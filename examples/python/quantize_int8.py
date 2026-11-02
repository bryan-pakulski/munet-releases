"""int8 inference for a CNN ONNX model: calibrate on sample images, compare accuracy and speed with fp16, save.

    python examples/python/quantize_int8.py v11-n-det.onnx calib_images/ --shape images=1x3x640x640
    python examples/python/quantize_int8.py model.onnx calib_images/ --shape images=1x3x640x640 --save model-int8.mnet

calib_images/ holds a few dozen to a few hundred images like the ones you will serve (jpg/png); they are resized to
the input size, RGB in [0, 1]. Add --mean/--std if your model normalizes inputs. Needs Pillow.
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

import munet as mu
from munet.interop import from_onnx
from munet.quantization import calibrate_onnx


def load_image(path, h, w, mean, std):
    x = np.asarray(Image.open(path).convert("RGB").resize((w, h), Image.BILINEAR), np.float32) / 255.0
    return np.ascontiguousarray(((x - mean) / std).transpose(2, 0, 1)[None], np.float32)


def timed(program, x, runs=50):
    """Best GPU time in ms over `runs` calls (stats()["gpu_ms"] is the last call's GPU time)."""
    best = float("inf")
    for i in range(runs + 5):
        program(x)
        if i >= 5:
            best = min(best, program.stats()["gpu_ms"])
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("images", help="directory of calibration images")
    ap.add_argument("--shape", required=True, help="NAME=1x3xHxW")
    ap.add_argument("--mean", type=float, nargs=3, default=(0.0, 0.0, 0.0))
    ap.add_argument("--std", type=float, nargs=3, default=(1.0, 1.0, 1.0))
    ap.add_argument("--save")
    args = ap.parse_args()
    name, dims = args.shape.split("=")
    shape = tuple(int(d) for d in dims.split("x"))
    files = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not files:
        raise SystemExit(f"no images in {args.images}")
    mean, std = np.array(args.mean, np.float32), np.array(args.std, np.float32)
    images = [load_image(f, shape[2], shape[3], mean, std) for f in files]

    # 1. Activation scales: runs the model in fp32 over the images and picks a clip value per convolution input.
    scales = calibrate_onnx(args.model, [(x,) for x in images], observer="mse")
    print(f"calibrated {len(scales)} convolutions on {len(images)} images")

    # 2. Same model, int8 convolutions (eligible ones; the rest stay fp16).
    shapes = {name: shape}
    fp16 = from_onnx(args.model, precision="fp16", input_shapes=shapes)
    int8 = from_onnx(args.model, precision="fp16", input_shapes=shapes, int8=scales)

    # 3. Compare on held-out inputs (here: the images themselves; use your validation set and task metric).
    cos = []
    for x in images[: min(20, len(images))]:
        a = fp16(x)
        a = (a[0] if isinstance(a, (tuple, list)) else a).numpy().ravel().astype(np.float64)
        b = int8(x)
        b = (b[0] if isinstance(b, (tuple, list)) else b).numpy().ravel().astype(np.float64)
        cos.append(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    print(f"int8 vs fp16 output cosine: mean {np.mean(cos):.5f}, worst {np.min(cos):.5f}")
    print(f"int8 kernels: {int8.stats()['int8_conv_kernels']}")
    print(f"GPU time: fp16 {timed(fp16, images[0]):.2f} ms, int8 {timed(int8, images[0]):.2f} ms")

    if args.save:
        mu.save(int8, args.save)  # the scales are part of the program: C, C++ and Rust load it as int8
        print(f"saved {args.save}")


if __name__ == "__main__":
    main()
