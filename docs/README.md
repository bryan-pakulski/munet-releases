# MuNet documentation

## Guides

| I want to... | Guide |
|---|---|
| Install MuNet and check my GPU | [getting-started.md](getting-started.md) |
| Train a model (loop, losses, optimizers, schedules, checkpoints, precision) | [training.md](training.md) |
| Load training data (datasets, augmentation, workers, prefetch, sharding) | [data.md](data.md) |
| Train on several GPUs, including different sizes and vendors | [multi-gpu.md](multi-gpu.md) |
| Import / export ONNX, import PyTorch | [onnx.md](onnx.md) |
| Make inference fast (precision, autotuning, batching, measuring) | [inference.md](inference.md) |
| Quantize (int8 convolutions, int8 / int4 linear layers, pre-quantized ONNX) | [quantization.md](quantization.md) |
| Configure NVIDIA, Intel, AMD, mobile GPUs and containers | [gpus.md](gpus.md) |
| Deploy from C or C++ | [c.md](c.md) |
| Deploy from Rust | [../rust/README.md](../rust/README.md) |
| Detect objects in video without copies (frame pipeline, FFmpeg, Vulkan interop) | [zero-copy.md](zero-copy.md) |
| Look up an environment variable | [environment.md](environment.md) |
| Fix an error | [troubleshooting.md](troubleshooting.md) |
| Look up a Python function | [python.md](python.md) |

## Examples

| Example | Shows |
|---|---|
| [examples/python/train_classifier.py](../examples/python/train_classifier.py) | data loader, training, evaluation, LR schedule, checkpoint/resume, export to `.mnet` + ONNX |
| [examples/python/train_multi_gpu.py](../examples/python/train_multi_gpu.py) | DataParallel over all GPUs, `split="auto"` / `"memory"`, `max_rows` |
| [examples/python/infer_onnx.py](../examples/python/infer_onnx.py) | any ONNX model: run, time, compare with ONNX Runtime, save `.mnet` |
| [examples/python/detect_yolo.py](../examples/python/detect_yolo.py) | YOLO detection end to end (letterbox, NMS, drawing) |
| [examples/python/quantize_int8.py](../examples/python/quantize_int8.py) | int8 calibration, accuracy and speed vs fp16 |
| [examples/python/infer_multi_gpu.py](../examples/python/infer_multi_gpu.py) | batch inference on every GPU from a shared queue |
| [examples/c/run_model.cpp](../examples/c/run_model.cpp) | C/C++: load and run a `.mnet` |
| [rust/examples/detect_frame.rs](../rust/examples/detect_frame.rs) | Rust: NV12 frames → detections on the GPU, tiled |
| [examples/c/vulkan_interop.cpp](../examples/c/vulkan_interop.cpp) | zero copy with your own Vulkan code |
| [examples/bench](../examples/bench/README.md) | correctness and speed of ~40 public models on your GPUs, vs ONNX Runtime and TensorRT |

## Reference

- [op_coverage.md](op_coverage.md): ONNX operator coverage.
- [swarm.md](swarm.md): training across machines (owner and worker nodes).

