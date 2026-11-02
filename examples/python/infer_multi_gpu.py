"""Batch inference on every GPU: one program per GPU, fed from a shared queue so faster GPUs take more batches.

    python examples/python/infer_multi_gpu.py model.onnx --shape images=8x3x640x640
    python examples/python/infer_multi_gpu.py model.onnx --shape images=8x3x640x640 --devices vulkan:0 --batches 200

The inputs here are random; put your DataLoader where `batches()` is. Results come back per batch index, in any order.
"""
import argparse
import queue
import threading
import time

import numpy as np

import munet as mu
from munet.interop import from_onnx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--shape", required=True, help="NAME=BxCxHxW for each input, comma-separated")
    ap.add_argument("--devices", help="comma-separated, default: every Vulkan GPU")
    ap.add_argument("--precision", default="fp16")
    ap.add_argument("--batches", type=int, default=100)
    args = ap.parse_args()

    shapes = {kv.split("=")[0]: tuple(int(d) for d in kv.split("=")[1].split("x")) for kv in args.shape.split(",")}
    devices = args.devices.split(",") if args.devices else [f"vulkan:{i}" for i in range(len(mu.devices()))]

    # Load in parallel: each GPU compiles its own program (kernels are cached on disk after the first time).
    programs = [None] * len(devices)
    def load(i):
        programs[i] = from_onnx(args.model, device=devices[i], precision=args.precision, input_shapes=shapes)
        programs[i](*[np.zeros(s, np.float32) for s in shapes.values()])  # first call: compile and warm up
    loaders = [threading.Thread(target=load, args=(i,)) for i in range(len(devices))]
    [t.start() for t in loaders]
    [t.join() for t in loaders]

    rng = np.random.default_rng(0)
    samples = [tuple(rng.random(s, dtype=np.float32) for s in shapes.values()) for _ in range(4)]
    def batches():  # replace with your data (e.g. a munet.data.DataLoader): any iterable of input tuples
        for k in range(args.batches):
            yield samples[k % len(samples)]

    work, results, done = queue.Queue(maxsize=2 * len(devices)), {}, [0] * len(devices)

    def worker(i):
        while (item := work.get()) is not None:
            index, inputs = item
            out = programs[i](*inputs)
            outs = out if isinstance(out, (tuple, list)) else (out,)
            results[index] = [o.numpy() if hasattr(o, "numpy") else o for o in outs]  # copy before the next call
            done[i] += 1

    workers = [threading.Thread(target=worker, args=(i,)) for i in range(len(devices))]
    start = time.perf_counter()
    [t.start() for t in workers]
    for item in enumerate(batches()):
        work.put(item)
    for _ in workers:
        work.put(None)
    [t.join() for t in workers]
    elapsed = time.perf_counter() - start

    batch = next(iter(shapes.values()))[0]
    print(f"{len(results)} batches in {elapsed:.2f} s: {len(results) * batch / elapsed:,.1f} samples/s")
    for d, n in zip(devices, done):
        print(f"  {d} ({mu.devices()[int(d.split(':')[1])]}): {n} batches")


if __name__ == "__main__":
    main()
