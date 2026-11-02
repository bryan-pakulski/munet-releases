# Python examples

Each script runs on its own: `python examples/python/<script>.py --help`. They use `device="vulkan"` (GPU 0); set
`MUNET_VULKAN_DEVICE=N` to use another GPU, or pass `--device cpu` where offered.

| Script | What it does | Needs |
|---|---|---|
| [train_classifier.py](train_classifier.py) | trains a CNN with a DataLoader, AdamW and an LR schedule; evaluates; checkpoints (`--resume`); exports `.mnet` and ONNX | nothing (synthetic data) |
| [train_multi_gpu.py](train_multi_gpu.py) | the same model on every GPU with `DataParallel`; `--split auto|memory|even|3,2,1`, `--max-rows` | 2+ GPUs (works with 1) |
| [infer_onnx.py](infer_onnx.py) | runs any ONNX model, times it, compares with ONNX Runtime (`--compare`), saves `.mnet` | `onnxruntime` for `--compare` |
| [detect_yolo.py](detect_yolo.py) | YOLOv8/11 detection on an image: letterbox, GPU inference, NMS, boxes drawn to `*.det.jpg` | `pillow`, a YOLO ONNX file |
| [quantize_int8.py](quantize_int8.py) | int8 calibration on a folder of images; accuracy and speed vs fp16; saves an int8 `.mnet` | `pillow` |
| [infer_multi_gpu.py](infer_multi_gpu.py) | batch inference on all GPUs from a shared queue | |

Try them with a public model:

```bash
pip install 'munet-nn[interop]' pillow onnxruntime
curl -LO https://github.com/jamjamjon/assets/releases/download/yolo/v11-n-det.onnx
curl -LO https://github.com/jamjamjon/assets/releases/download/yolo/demo-det.jpg

python examples/python/detect_yolo.py v11-n-det.onnx demo-det.jpg --precision fp16
python examples/python/infer_onnx.py v11-n-det.onnx --shape images=1x3x640x640 --precision fp16 --compare --save yolo.mnet
mkdir calib && cp demo-det.jpg calib/   # use a few dozen of your own images
python examples/python/quantize_int8.py v11-n-det.onnx calib --shape images=1x3x640x640
python examples/python/infer_multi_gpu.py v11-n-det.onnx --shape images=8x3x640x640

python examples/python/train_classifier.py
python examples/python/train_multi_gpu.py --split memory
```

The `.mnet` files these scripts save load from C, C++ ([examples/c/run_model.cpp](../c/run_model.cpp)) and Rust
([rust/examples](../../rust/examples)).
