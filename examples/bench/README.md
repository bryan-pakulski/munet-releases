# MuNet bench suite

The bench suite runs real ONNX models on every GPU you have, at every precision. It checks correctness against ONNX
Runtime and, on NVIDIA, compares speed with TensorRT.

Models come from three places:

- [usls](https://github.com/jamjamjon/usls) release assets (MIT): YOLO v5-v11 (detection, segmentation, pose, OBB,
  classification), RT-DETR / D-FINE / DEIM / RF-DETR, ConvNeXt, MobileOne, FastViT, DINOv2, DeiT, CLIP,
  Depth-Anything, SAM / SAM2 encoders, MODNet, BiRefNet, LinkNet, Swin2SR, PP-OCR detection and recognition. usls's
  `*-q8` files are pre-quantized ONNX graphs and test the quantized-operator import.
- Hugging Face ONNX exports: SegFormer-B0, DistilBERT, BERT-base, the Whisper-tiny encoder, LaMa.
- Local exports by `export.py` (torchvision and PyTorch): ResNet-50, EfficientNet-B0, MobileNetV3, DeepLabV3,
  Swin-T, ViT-B/16, a U-Net, Real-ESRGAN x4 and Stable Diffusion's VAE decoder.

Batched variants (`-b4`, `-b8`) run detectors at the batch sizes a serving backend uses.

## Run

```bash
pip install onnxruntime pillow                                   # reference runtime and demo images
python examples/bench/run.py                                     # quick tier, every Vulkan GPU, fp32/mixed/fp16/int8
python examples/bench/run.py --tier standard --trt               # more models, plus TensorRT timings on NVIDIA
python examples/bench/run.py --models rtdetr-v1-r18,dfine-n --device vulkan:1 --precision fp16,int8
python examples/bench/report.py --markdown report.md             # matrix of the latest run per GPU
```

`run.py` options:

| Option | Default | Meaning |
|---|---|---|
| `--tier` | `quick` | `quick`, `standard` or `full` |
| `--models` | | comma-separated name substrings; overrides `--tier` |
| `--device` | every Vulkan GPU | `vulkan:N`; repeatable |
| `--precision` | `fp32,mixed,fp16,int8` | comma-separated |
| `--runs` | 20 | timed runs per model |
| `--trt` | off | also time TensorRT (NVIDIA GPUs, Docker) |
| `--trt-image` | `nvcr.io/nvidia/tensorrt:26.03-py3` | TensorRT container; `MUNET_BENCH_TRT_IMAGE` overrides the default |
| `--trt-gpu` | 0 | CUDA device index for TensorRT |
| `--out` | `examples/bench/results` | results directory |

## What it measures

For each model the runner:

1. downloads the ONNX file and demo image;
2. builds inputs: image inputs from the demo image, others seeded random of the declared type;
3. computes reference outputs with ONNX Runtime on the CPU in fp32;
4. for each GPU and precision, starts a separate process that:
   - imports the model with `munet.interop.from_onnx` (static input shapes), runs it and compares every output with
     the reference: cosine similarity, maximum error relative to the reference's largest magnitude, exact match for
     integer outputs, and non-finite values;
   - times it: GPU time of the compute section (`gpu_ms`, minimum over `--runs`), host latency per call including
     upload and readback (`host_ms`, median), and import plus compile time.

`int8` calibrates per-tensor activation scales on jittered copies of the inputs (`munet.quantization.calibrate_onnx`).
Eligible convolutions run on int8 kernels, the rest in fp16. Models without eligible layers say so in a note.

## Grading

| Status | fp32 | mixed | fp16 | int8 |
|---|---|---|---|---|
| PASS: every output cosine >= / relative error <= | 0.99999 / 1e-2 | 0.999 / 5e-2 | 0.995 / 0.1 | 0.98 / 0.3 |
| WARN: cosine >= | 0.9999 | 0.99 | 0.98 | 0.95 |

- Integer outputs PASS when at least 99% of values are equal, WARN at 90%.
- A FAIL whose sorted values still reach the WARN cosine becomes WARN, noted "order differs (near ties)".
- DETR-style detectors are graded on detections instead: recall of the reference's detections with score > 0.3
  (same label, IoU >= 0.7). PASS / WARN recall: fp32 0.98 / 0.9, mixed and fp16 0.9 / 0.8, int8 0.8 / 0.6.
- FAIL is anything worse, a shape mismatch or non-finite outputs.
- ERROR is an import, compile or run failure. The message is in the results file.

Random-input parity is a numerical check, not an accuracy metric. A WARN on fp16 or int8 is worth a look with real
data. A FAIL or ERROR is a bug.

## TensorRT timings

`--trt` times the same models with `trtexec` in the `--trt-image` container on CUDA device `--trt-gpu`, at the
precisions given to `--precision`. int8 runs without calibration, so its numbers are speed only. NVIDIA GPUs are
detected by PCI vendor ID.

## Hardware report

`hardware_report.py` turns runs from several GPUs into speed tables:

- MuNet milliseconds per model;
- speed relative to a baseline GPU;
- MuNet against TensorRT on the same GPU.

```bash
python examples/bench/hardware_report.py --baseline "GPU A" \
    "GPU A=path/to/a/results" "GPU B=path/to/b/results"
```

## Files

- `models.py`: the manifest (name, release asset, tier, task, sizes for symbolic dimensions, normalization, demo
  image, overrides for non-image inputs). Tiers: `quick` (one or two per family), `standard`, `full`.
- `models.lock.json`: SHA-256 of every downloaded file. The first download is trusted; later downloads that differ
  are rejected. Commit it after adding models.
- `results/<gpu>/<timestamp>.json`: one file per GPU and run (git-ignored). `report.py` renders them;
  `hardware_report.py` compares GPUs.
- `find_mismatch.py MODEL`: finds the first ONNX node whose output differs from ONNX Runtime.
- `export.py NAME OUT`: exports the models with no published ONNX file. `run.py` calls it on first use with
  `$MUNET_BENCH_TORCH_PYTHON` (default: its own interpreter; needs torch and torchvision).
- Local-only models (`local=` in the manifest, e.g. `local-detr-1280-b3`) are skipped when their file is absent.
  `MUNET_BENCH_LOCAL_DETR` names the file for those entries.
- Downloads, exports and ONNX Runtime references are cached in `$MUNET_BENCH_CACHE` (default `~/.cache/munet-bench`).

To check a new GPU, run the suite on it and compare `report.py` matrices. Failures on one vendor only point at
driver- or layout-specific kernels (cooperative-matrix shapes, subgroup sizes, fp16 denormal handling).
