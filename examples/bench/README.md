# MuNet bench suite

Correctness and speed of real ONNX models on every GPU you have, at every precision, against ONNX Runtime (parity)
and TensorRT (speed, NVIDIA). Models come from [usls](https://github.com/jamjamjon/usls) (MIT): YOLO v5-v11
(detection, segmentation, pose, OBB, classification), RT-DETR / D-FINE / DEIM / RF-DETR, ConvNeXt, MobileOne, FastViT,
DINOv2, CLIP, Depth-Anything, SAM / SAM2 encoders, MODNet, PP-OCR detection and recognition, and usls's own int8 (q8)
exports.

```bash
pip install onnxruntime pillow                                   # reference runtime and demo images
python examples/bench/run.py                                     # quick tier, every Vulkan GPU, fp32/mixed/fp16/int8
python examples/bench/run.py --tier standard --trt               # more models, plus TensorRT timings on NVIDIA
python examples/bench/run.py --models rtdetr-v1-r18,dfine-n --device vulkan:1 --precision fp16,int8
python examples/bench/report.py --markdown report.md             # matrix of the latest run per GPU
```

## What it measures

For each model the runner downloads the ONNX file (and demo image), builds inputs (image inputs from the demo image,
others seeded random of the declared type), and computes the reference outputs with ONNX Runtime on the CPU in fp32.
Then, for each GPU and precision, a separate process:

- imports the model with `munet.interop.from_onnx` (static input shapes), runs it, and compares every output with the
  reference: cosine similarity, maximum error relative to the reference's largest magnitude, exact match for integer
  outputs, and non-finite values;
- times it: GPU time of the compute section (`gpu_ms`, minimum over `--runs`) and host latency per call including
  input upload and output readback (`host_ms`, median), plus import and compile time.

`int8` calibrates per-tensor activation scales on jittered copies of the inputs (`munet.quantization.calibrate_onnx`)
and runs eligible convolutions on int8 kernels (the rest in fp16); models without eligible layers say so in a note.
usls's `*-q8` files are pre-quantized ONNX graphs and test the quantized-operator import.

| status | fp32 | mixed | fp16 | int8 |
|---|---|---|---|---|
| PASS: every output cosine >= / relative error <= | 0.99999 / 1e-2 | 0.999 / 5e-2 | 0.995 / 0.1 | 0.98 / 0.3 |
| WARN: cosine >= | 0.9999 | 0.99 | 0.98 | 0.95 |

FAIL is anything worse, a shape mismatch or non-finite outputs; ERROR is an import, compile or run failure (the message
is in the results file). Random-input parity is a numerical check, not an accuracy metric: a WARN on fp16/int8 is worth
a look with real data; a FAIL or ERROR is a bug.

`--trt` times the same models with `trtexec` at the precisions given to `--precision` (int8 without calibration, so
speed only) in the `--trt-image` container (any TensorRT image works, e.g. `nvcr.io/nvidia/tensorrt:26.03-py3`) on CUDA
device `--trt-gpu`. NVIDIA GPUs are detected by PCI vendor ID.

`hardware_report.py` turns runs from several GPUs into the speed tables of
[docs/tested-hardware.md](../../docs/tested-hardware.md):
- MuNet milliseconds per model;
- speed relative to a baseline GPU;
- MuNet against TensorRT on the same GPU.

For example: `python examples/bench/hardware_report.py --baseline "RTX 3060" "RTX 3060=results/nvidia-geforce-rtx-3060" "L4=path/to/l4/results"`.

## Files

- `models.py`: the manifest (name, release asset, tier, task, sizes for symbolic dimensions, normalization, demo
  image, overrides for non-image inputs). Tiers: `quick` (one or two per family), `standard`, `full`.
- `models.lock.json`: SHA-256 of every downloaded file (trust on first use; downloads that differ are rejected).
  Commit it after adding models.
- `results/<gpu>/<timestamp>.json`: one file per GPU and run (git-ignored); `report.py` renders them, and
  `hardware_report.py` compares GPUs.
- Downloads cache in `$MUNET_BENCH_CACHE` (default `~/.cache/munet-bench`) with the ONNX Runtime references.

New GPU: run the suite on it and compare `report.py` matrices; models that fail on one vendor only point at driver or
layout-specific kernels (cooperative-matrix shapes, subgroup sizes, fp16 denormal handling).
