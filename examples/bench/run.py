"""MuNet bench suite: correctness (parity with ONNX Runtime fp32) and speed of ONNX models per GPU and precision,
optionally against TensorRT on NVIDIA GPUs.

    python examples/bench/run.py                              # quick tier, every Vulkan GPU, fp32/mixed/fp16/int8
    python examples/bench/run.py --tier standard --device vulkan:1 --precision fp16,int8
    python examples/bench/run.py --models yolo11n,rtdetr --trt # also time TensorRT (NVIDIA GPUs, Docker)
    python examples/bench/report.py                           # matrix of every results file

Models (examples/bench/models.py) download from usls's release assets into $MUNET_BENCH_CACHE (default
~/.cache/munet-bench). Downloads are untrusted: the first download of a file records its SHA-256 in
examples/bench/models.lock.json (trust on first use; commit it) and every later download must match.

Each (model, device, precision) runs in its own process, so a crash or device loss is recorded as an ERROR and the
suite continues. Results go to examples/bench/results/<device>/<timestamp>.json.
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from models import MODELS, BASE, Input, select  # noqa: E402

CACHE = Path(os.environ.get("MUNET_BENCH_CACHE", Path.home() / ".cache" / "munet-bench"))
LOCK = HERE / "models.lock.json"
PRECISIONS = ("fp32", "mixed", "fp16", "int8")
# Parity thresholds against ONNX Runtime fp32: (cosine for PASS, cosine for WARN, max relative error for PASS).
THRESHOLDS = {"fp32": (0.99999, 0.9999, 1e-2), "mixed": (0.999, 0.99, 5e-2), "fp16": (0.995, 0.98, 1e-1),
              "int8": (0.98, 0.95, 3e-1)}


# ---------------------------------------------------------------------------------------------------------------------
# Downloads (pinned by SHA-256)

def _lock():
    return json.loads(LOCK.read_text()) if LOCK.exists() else {}


def fetch(release, name):
    """Path of a release asset in the cache, downloaded and verified against the lock file."""
    path = CACHE / release / name
    lock, key = _lock(), f"{release}/{name}"
    if path.exists() and lock.get(key) == _sha256(path):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = f"{BASE}/{release}/{name}"
    print(f"  downloading {url}", flush=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as tmp:
        with urllib.request.urlopen(url, timeout=120) as response:
            while chunk := response.read(1 << 20):
                tmp.write(chunk)
    digest = _sha256(Path(tmp.name))
    if key in lock and lock[key] != digest:
        os.unlink(tmp.name)
        raise RuntimeError(f"{key}: SHA-256 {digest} does not match the pinned {lock[key]} (models.lock.json)")
    os.replace(tmp.name, path)
    if key not in lock:
        lock[key] = digest
        LOCK.write_text(json.dumps(dict(sorted(lock.items())), indent=1) + "\n")
        print(f"  pinned {key} sha256={digest}", flush=True)
    return path


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------------------------
# Inputs and references

def _image(spec, shape):
    from PIL import Image  # Pillow: test dependency of the suite only
    release, name = spec.split("/", 1)
    img = Image.open(fetch(release, name)).convert("RGB").resize((shape[3], shape[2]), Image.BILINEAR)
    return np.asarray(img, np.float32).transpose(2, 0, 1)[None] / 255.0


def build_inputs(model, onnx_model, seed=0):
    """name -> array for every graph input, and the static shapes for munet.interop.from_onnx."""
    from onnx import TensorProto
    rng = np.random.default_rng(seed)
    init = {t.name for t in onnx_model.graph.initializer}
    arrays, shapes = {}, {}
    np_types = {TensorProto.FLOAT: np.float32, TensorProto.FLOAT16: np.float16, TensorProto.INT64: np.int64,
                TensorProto.INT32: np.int32, TensorProto.BOOL: np.bool_, TensorProto.UINT8: np.uint8}
    for vi in onnx_model.graph.input:
        if vi.name in init:
            continue
        t = vi.type.tensor_type
        spec = model.inputs.get(vi.name, Input())
        dtype = np.dtype(spec.dtype) if spec.dtype else np.dtype(np_types.get(t.elem_type, np.float32))
        dims = [d.dim_value if d.HasField("dim_value") and d.dim_value > 0 else None for d in t.shape.dim]
        if spec.shape:
            shape = tuple(spec.shape)
        else:  # symbolic dims: batch 1, spatial dims of 4-D inputs = model.size
            shape = tuple(d if d else (1 if i == 0 else 3 if len(dims) == 4 and i == 1 and t.elem_type == TensorProto.FLOAT
                                       else model.size if len(dims) == 4 and i >= 2 else 1) for i, d in enumerate(dims))
        kind = spec.kind or ("image" if len(shape) == 4 and shape[1] == 3 and dtype.kind == "f" else
                             "int" if dtype.kind in "iub" else "uniform")
        if kind == "image":
            a = _image(spec.image if spec.kind else model.image, shape)
            mean, std = spec.mean or model.norm.get("mean"), spec.std or model.norm.get("std")
            if mean:
                a = (a - np.asarray(mean, np.float32)[:, None, None]) / np.asarray(std, np.float32)[:, None, None]
            a = np.repeat(a, shape[0], 0)
        elif kind == "const":
            a = np.full(shape, spec.value)
        elif kind == "int":
            a = rng.integers(int(spec.low), int(max(spec.high, spec.low + 1)), size=shape)
        elif kind == "normal":
            a = rng.normal(size=shape)
        else:
            a = rng.uniform(spec.low, spec.high, size=shape)
        arrays[vi.name] = np.ascontiguousarray(a.astype(dtype))
        shapes[vi.name] = shape
    return arrays, shapes


def reference(model, path, arrays):
    """ONNX Runtime CPU fp32 outputs (cached next to the model)."""
    key = hashlib.sha256(b"".join(a.tobytes() for a in arrays.values())).hexdigest()[:16]
    cache = path.with_suffix(f".ref-{key}.npz")
    if cache.exists():
        data = np.load(cache)
        return [data[f"o{i}"] for i in range(len(data.files))]
    import onnxruntime as ort
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    outputs = session.run(None, arrays)
    np.savez(cache, **{f"o{i}": o for i, o in enumerate(outputs)})
    return outputs


# ---------------------------------------------------------------------------------------------------------------------
# Worker: one (model, device, precision) in its own process

def compare(outputs, refs):
    metrics = []
    for out, ref in zip(outputs, refs):
        out, ref = np.asarray(out), np.asarray(ref)
        m = {"shape": list(ref.shape)}
        if out.shape != ref.shape:
            m["error"] = f"shape {list(out.shape)} != {list(ref.shape)}"
        elif ref.dtype.kind in "iub":
            m["equal"] = float((out == ref).mean()) if ref.size else 1.0
        else:
            a, b = out.astype(np.float64).ravel(), ref.astype(np.float64).ravel()
            finite = np.isfinite(b)
            a, b = np.nan_to_num(a[finite]), b[finite]
            scale = float(np.abs(b).max()) if b.size else 0.0
            m["rel"] = float(np.abs(a - b).max() / scale) if scale > 0 else float(np.abs(a - b).max(initial=0.0))
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            m["cos"] = float(a @ b / (na * nb)) if na > 0 and nb > 0 else (1.0 if na == nb else 0.0)
            # Order-insensitive check (top-k / NMS outputs: near-tied candidates may sort differently): cosine of the
            # sorted values. Passing this but not the direct check means "same values, different order".
            sa, sb = np.sort(a), np.sort(b)
            m["cos_sorted"] = float(sa @ sb / (na * nb)) if na > 0 and nb > 0 else m["cos"]
            m["nonfinite"] = int((~np.isfinite(out.astype(np.float64).ravel()[finite])).sum())
        metrics.append(m)
    return metrics


def detections(outputs, refs, threshold=0.3):
    """DETR-style (labels [1,Q], boxes [1,Q,4] xyxy, scores [1,Q]): recall of the reference's confident detections
    among MuNet's (same label, IoU >= 0.7) and the mean score difference of the matches."""
    (la, ba, sa), (lr, br, sr) = [np.asarray(o)[0] for o in outputs[:3]], [np.asarray(o)[0] for o in refs[:3]]
    def iou(p, q):
        w = max(0.0, min(p[2], q[2]) - max(p[0], q[0])); h = max(0.0, min(p[3], q[3]) - max(p[1], q[1]))
        u = (p[2] - p[0]) * (p[3] - p[1]) + (q[2] - q[0]) * (q[3] - q[1]) - w * h
        return w * h / u if u > 0 else 0.0
    want = [i for i in range(len(sr)) if sr[i] > threshold]
    have = [j for j in range(len(sa)) if sa[j] > threshold * 0.8]
    used, matched, dscore = set(), 0, []
    for i in want:
        best = max(((iou(br[i], ba[j]), j) for j in have if j not in used and int(la[j]) == int(lr[i])), default=(0, -1))
        if best[0] >= 0.7:
            used.add(best[1]); matched += 1; dscore.append(abs(float(sa[best[1]]) - float(sr[i])))
    return {"reference": len(want), "matched": matched, "recall": matched / len(want) if want else 1.0,
            "mean_dscore": float(np.mean(dscore)) if dscore else 0.0}


DETECTION_RECALL = {"fp32": (0.98, 0.9), "mixed": (0.9, 0.8), "fp16": (0.9, 0.8), "int8": (0.8, 0.6)}


def status(precision, metrics):
    passing, warning, rel_ok = THRESHOLDS[precision]
    worst = "PASS"
    for m in metrics:
        if "error" in m or m.get("nonfinite", 0):
            return "FAIL"
        if "equal" in m:
            s = "PASS" if m["equal"] >= 0.99 else "WARN" if m["equal"] >= 0.9 else "FAIL"
        else:
            s = "PASS" if m["cos"] >= passing and m["rel"] <= rel_ok else "WARN" if m["cos"] >= warning else "FAIL"
            if s == "FAIL" and m.get("cos_sorted", 0) >= warning:
                s, m["note"] = "WARN", "order differs (near ties)"
        worst = max(worst, s, key=["PASS", "WARN", "FAIL"].index)
    return worst


def worker(job):
    import munet as mu
    from munet.interop import from_onnx
    path, device, precision, runs = Path(job["path"]), job["device"], job["precision"], job["runs"]
    data = np.load(job["inputs"])
    names = job["names"]
    arrays = [data[n] for n in names]
    refs = [np.load(job["refs"])[f"o{i}"] for i in range(job["outputs"])]
    shapes = {n: tuple(s) for n, s in job["shapes"].items()}
    result = {"model": job["model"], "device": device, "precision": precision}
    t0 = time.perf_counter()
    int8 = None
    base = precision
    if precision == "int8":
        from munet.quantization import calibrate_onnx
        rng = np.random.default_rng(1)
        jitter = lambda a: (a * rng.uniform(0.9, 1.1) + rng.normal(0, 0.02, a.shape)).astype(a.dtype) if a.dtype.kind == "f" else a
        batches = [tuple(jitter(a) for a in arrays) for _ in range(8)]
        try:
            int8 = calibrate_onnx(str(path), batches, device=device)
        except Exception as error:  # no int8-eligible layer: report as fp16 with a note
            result["note"] = f"no int8 layers ({str(error).splitlines()[0][:120]})"
        base = "fp16"
    program = from_onnx(str(path), device=device, precision=base, int8=int8, profile=False, input_shapes=shapes)
    outputs = program(*arrays)
    result["compile_s"] = time.perf_counter() - t0
    outputs = outputs if isinstance(outputs, (tuple, list)) else [outputs]
    outputs = [o.numpy() if hasattr(o, "numpy") else np.asarray(o) for o in outputs]
    result["metrics"] = compare(outputs, refs)
    result["status"] = status(precision, result["metrics"])
    if job.get("compare") == "detr":
        # top-k over mostly near-zero scores: raw tensors legitimately differ off the top; grade the confident detections
        d = result["detections"] = detections(outputs, refs)
        ok, warn = DETECTION_RECALL[precision]
        result["status"] = "PASS" if d["recall"] >= ok else "WARN" if d["recall"] >= warn else "FAIL"
    # timing: host latency per call (includes upload and readback) and GPU time of the compute section. Keep the GPU busy
    # ~1 s first: GPUs idling at low clocks (Intel Arc ramps slowly) otherwise read 2-3x slow.
    t = time.perf_counter()
    while time.perf_counter() - t < 1.0:
        o = program(*arrays)
        o0 = o[0] if isinstance(o, (tuple, list)) else o
        if hasattr(o0, "numpy"):
            o0.numpy()
    host, gpu = [], []
    for _ in range(runs):
        t = time.perf_counter()
        o = program(*arrays)
        o0 = o[0] if isinstance(o, (tuple, list)) else o
        if hasattr(o0, "numpy"):
            o0.numpy()
        host.append((time.perf_counter() - t) * 1e3)
        stats = getattr(program, "stats", lambda: {})()
        if "gpu_ms" in stats:
            gpu.append(stats["gpu_ms"])
    result["host_ms"] = float(np.median(host))
    if gpu:
        result["gpu_ms"] = float(min(gpu))
    if int8:
        result["int8_layers"] = len(int8)
    return result


# ---------------------------------------------------------------------------------------------------------------------
# TensorRT (NVIDIA, trtexec in a container)

def tensorrt(path, shapes, precision, image, gpu):
    flags = {"fp32": [], "fp16": ["--fp16"], "int8": ["--int8", "--fp16"]}[precision]
    shape_arg = ",".join(f"{n}:{'x'.join(str(d) for d in s)}" for n, s in shapes.items())
    cmd = ["docker", "run", "--rm", "--gpus", f"device={gpu}", "-v", f"{path.parent}:/w:ro", "--entrypoint", "trtexec", image,
           f"--onnx=/w/{path.name}", f"--shapes={shape_arg}", "--iterations=50", "--warmUp=500", "--noDataTransfers", *flags]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
    m = re.search(r"GPU Compute Time: min = ([\d.]+) ms.*median = ([\d.]+) ms", proc.stdout)
    if not m:
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-3:]
        return {"precision": precision, "error": " | ".join(tail)[:300]}
    return {"precision": precision, "gpu_ms": float(m.group(1)), "median_ms": float(m.group(2))}


# ---------------------------------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tier", default="quick", choices=("quick", "standard", "full"))
    ap.add_argument("--models", help="comma-separated name substrings (overrides --tier)")
    ap.add_argument("--device", action="append", help="vulkan:N (repeatable; default: every Vulkan GPU)")
    ap.add_argument("--precision", default=",".join(PRECISIONS))
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--trt", action="store_true", help="also time TensorRT fp32/fp16/int8 (NVIDIA GPUs, Docker)")
    ap.add_argument("--trt-image", default=os.environ.get("MUNET_BENCH_TRT_IMAGE", "rsq-inference:nvidia-test"))
    ap.add_argument("--trt-gpu", default="0", help="CUDA device index for TensorRT")
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        try:
            result = worker(json.loads(Path(args.worker).read_text()))
        except Exception as error:
            result = {"status": "ERROR", "error": f"{type(error).__name__}: {str(error).strip().splitlines()[0][:300]}"}
        print("RESULT " + json.dumps(result))
        return

    import onnx
    import munet as mu
    from munet import _native
    names = mu.devices()
    devices = args.device or [f"vulkan:{i}" for i in range(len(names))]
    precisions = [p for p in args.precision.split(",") if p]
    models = select(args.tier, args.models.split(",") if args.models else None)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    runs = {}
    for device in devices:
        index = int(device.split(":")[1])
        limits = _native.device_limits(index)
        runs[device] = {"device": names[index], "vendor_id": limits["vendor_id"], "driver_version": limits["driver_version"],
                        "coop_rows": _native.coop_rows(index), "munet": mu.__version__, "date": stamp, "results": [], "tensorrt": []}
    with tempfile.TemporaryDirectory() as work:
        for model in models:
            print(f"== {model.name} ({model.release}/{model.file})", flush=True)
            try:
                path = fetch(model.release, model.file)
                graph = onnx.load(str(path), load_external_data=False)
                arrays, shapes = build_inputs(model, graph)
                refs = reference(model, path, arrays)
            except Exception as error:
                print(f"  setup failed: {error}", flush=True)
                for device in devices:
                    runs[device]["results"].append({"model": model.name, "task": model.task, "status": "ERROR", "precision": "-",
                                                    "error": f"setup: {str(error).splitlines()[0][:300]}"})
                continue
            np.savez(f"{work}/inputs.npz", **arrays)
            np.savez(f"{work}/refs.npz", **{f"o{i}": r for i, r in enumerate(refs)})
            for device in devices:
                for precision in precisions:
                    job = {"model": model.name, "path": str(path), "device": device, "precision": precision, "runs": args.runs,
                           "inputs": f"{work}/inputs.npz", "refs": f"{work}/refs.npz", "names": list(arrays),
                           "shapes": {k: list(v) for k, v in shapes.items()}, "outputs": len(refs), "compare": model.compare}
                    Path(f"{work}/job.json").write_text(json.dumps(job))
                    try:
                        proc = subprocess.run([sys.executable, __file__, "--worker", f"{work}/job.json"], capture_output=True,
                                              text=True, timeout=3600)
                        line = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
                        result = json.loads(line[-1][7:]) if line else {
                            "status": "ERROR", "error": f"worker exited {proc.returncode}: {(proc.stderr.strip().splitlines() or [''])[-1][:300]}"}
                    except subprocess.TimeoutExpired:
                        result = {"status": "ERROR", "error": "timeout"}
                    result.update(model=model.name, task=model.task, device=device, precision=precision)
                    runs[device]["results"].append(result)
                    worst = min((m.get("cos", 1.0) for m in result.get("metrics", [])), default=None)
                    if "detections" in result:
                        worst = f"{result['detections']['matched']}/{result['detections']['reference']} detections"
                    print(f"  {device} {precision:5s} {result['status']:5s} gpu {result.get('gpu_ms', float('nan')):8.2f} ms"
                          f"  host {result.get('host_ms', float('nan')):8.2f} ms  min cos {worst}"
                          f"{'  ' + result.get('error', result.get('note', '')) if result['status'] == 'ERROR' or 'note' in result else ''}", flush=True)
                if args.trt and "NVIDIA" in runs[device]["device"]:
                    for precision in ("fp32", "fp16", "int8"):
                        t = tensorrt(path, shapes, precision, args.trt_image, args.trt_gpu)
                        t["model"] = model.name
                        runs[device]["tensorrt"].append(t)
                        print(f"  {device} TensorRT {precision:5s} {t.get('gpu_ms', t.get('error'))}", flush=True)
    for device, run in runs.items():
        slug = re.sub(r"[^A-Za-z0-9]+", "-", run["device"]).strip("-").lower()
        out = Path(args.out) / slug / f"{stamp}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(run, indent=1))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
