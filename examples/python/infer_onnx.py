"""Run any ONNX model on the GPU, time it, and export it as .mnet for C, C++ and Rust.

    python examples/python/infer_onnx.py model.onnx --shape images=1x3x640x640
    python examples/python/infer_onnx.py model.onnx --shape images=1x3x640x640 --precision fp16 --autotune --save model.mnet
    python examples/python/infer_onnx.py model.onnx --shape images=1x3x640x640 --compare     # check against onnxruntime

Inputs are random (float inputs uniform in [0, 1)); integer inputs need real values, so pass a model whose inputs are
float or adapt `make_inputs`.
"""
import argparse
import time

import numpy as np

import munet as mu
from munet.interop import from_onnx


def make_inputs(path, shapes):
    import onnx
    elem = {i.name: i.type.tensor_type.elem_type for i in onnx.load(path, load_external_data=False).graph.input}
    rng = np.random.default_rng(0)
    return [rng.random(s, dtype=np.float32) if elem.get(n, 1) == onnx.TensorProto.FLOAT else np.ones(s, np.int64)
            for n, s in shapes.items()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--shape", required=True, help="NAME=DxDx... for every input, comma-separated")
    ap.add_argument("--device", default="vulkan")
    ap.add_argument("--precision", default="mixed", choices=["fp32", "mixed", "fp16"])
    ap.add_argument("--autotune", action="store_true", help="time tile configurations once (cached per GPU)")
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--save", help="write the program as .mnet")
    ap.add_argument("--compare", action="store_true", help="compare outputs with onnxruntime (CPU, fp32)")
    args = ap.parse_args()

    shapes = {kv.split("=")[0]: tuple(int(d) for d in kv.split("=")[1].split("x")) for kv in args.shape.split(",")}
    inputs = make_inputs(args.model, shapes)

    t = time.perf_counter()
    program = from_onnx(args.model, device=args.device, precision=args.precision, input_shapes=shapes,
                        autotune=args.autotune or None)
    outputs = program(*inputs)  # compiles on the first call
    outputs = outputs if isinstance(outputs, (tuple, list)) else [outputs]
    print(f"import + compile + first run: {time.perf_counter() - t:.1f} s")

    for _ in range(5):
        program(*inputs)
    gpu, host = [], []
    for _ in range(args.runs):
        t = time.perf_counter()
        out = program(*inputs)
        (out[0] if isinstance(out, (tuple, list)) else out).numpy()  # wait for the result
        host.append(time.perf_counter() - t)
        gpu.append(program.stats()["gpu_ms"])
    print(f"GPU {min(gpu):.2f} ms, end-to-end {np.median(host) * 1e3:.2f} ms (median of {args.runs}, "
          f"includes upload and readback of the first output)")
    stats = program.stats()
    print(f"{stats['device']}: {stats['kernels']} kernels, {stats['arena_bytes'] / 2**20:.0f} MiB device arena")

    outputs = program(*inputs)
    outputs = outputs if isinstance(outputs, (tuple, list)) else [outputs]
    values = [o.numpy() if hasattr(o, "numpy") else np.asarray(o) for o in outputs]
    for i, v in enumerate(values):
        print(f"output {i}: shape {v.shape} dtype {v.dtype}")

    if args.compare:
        import onnxruntime as ort
        session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
        reference = session.run(None, dict(zip(shapes, inputs)))
        for i, (a, b) in enumerate(zip(values, reference)):
            a, b = a.astype(np.float64).ravel(), b.astype(np.float64).ravel()
            cos = a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30)
            print(f"output {i}: cosine {cos:.6f}, max error {np.abs(a - b).max():.3g} (|ref| max {np.abs(b).max():.3g})")

    if args.save:
        mu.save(program, args.save)
        print(f"saved {args.save}: load it with munet.load / munet_load / munet::Program::load")


if __name__ == "__main__":
    main()
