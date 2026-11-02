# Getting started

## 1. Install a Vulkan driver

MuNet runs on any GPU through its Vulkan driver. Most desktop Linux installs already have one; check with
`vulkaninfo --summary` (package `vulkan-tools`).

| GPU | Debian / Ubuntu | Arch | Fedora |
|---|---|---|---|
| NVIDIA | the proprietary driver (`nvidia-driver-XXX`, which includes Vulkan) | `nvidia-utils` | `xorg-x11-drv-nvidia` (RPM Fusion) |
| AMD | `mesa-vulkan-drivers` (RADV) | `vulkan-radeon` | `mesa-vulkan-drivers` |
| Intel | `mesa-vulkan-drivers` (ANV) | `vulkan-intel` | `mesa-vulkan-drivers` |

- Intel Arc B-series (Battlemage) needs a recent Mesa (25.x or newer) and the `xe` kernel driver.
- GPU-specific settings are in [gpus.md](gpus.md).

## 2. Install the shader compiler

MuNet generates its GPU kernels for your model and compiles them with `glslangValidator`. Compiled kernels are
cached on disk, so this happens once per model and GPU.

```bash
sudo apt install glslang-tools        # Debian/Ubuntu
sudo pacman -S glslang                # Arch
sudo dnf install glslang              # Fedora
```

If it is not on `PATH`, set `MUNET_GLSLANG=/path/to/glslangValidator`.

## 3. Install MuNet

```bash
python -m pip install 'munet-nn[interop]'     # Python 3.10+, Linux x86_64 / aarch64
```

| Extra | Adds |
|---|---|
| (none) | the library; NumPy is its only dependency |
| `[interop]` | ONNX import/export (`onnx`) |
| `[torch]` | PyTorch import (`from_torch`) |

C, C++ and Rust: see [c.md](c.md) and [rust/README.md](../rust/README.md).

## 4. Check it works

```bash
python - <<'EOF'
import numpy as np, munet as mu
print(mu.devices())                                    # ['NVIDIA GeForce RTX 3060', ...]
f = mu.compile(lambda x: (x @ x.T).relu())             # device="vulkan" (GPU 0) by default
print(f(np.random.rand(256, 256).astype(np.float32)).numpy().shape, f.stats()["device"])
EOF
```

- **No devices, or an error about Vulkan:** the driver is missing or not visible (in containers, see the container notes for your vendor in
  [gpus.md](gpus.md)).
- **An error about glslangValidator:** repeat step 2.
- **The evaluation notice** is expected without a license key.

## 5. Next steps

| Task | Guide | Example |
|---|---|---|
| Run an ONNX model | [onnx.md](onnx.md), [inference.md](inference.md) | [infer_onnx.py](../examples/python/infer_onnx.py) |
| Detect objects in an image | [inference.md](inference.md) | [detect_yolo.py](../examples/python/detect_yolo.py) |
| Train a model | [training.md](training.md) | [train_classifier.py](../examples/python/train_classifier.py) |
| Train on several GPUs | [multi-gpu.md](multi-gpu.md) | [train_multi_gpu.py](../examples/python/train_multi_gpu.py) |
| int8 inference | [quantization.md](quantization.md) | [quantize_int8.py](../examples/python/quantize_int8.py) |
| Deploy from C/C++/Rust | [c.md](c.md), [rust/README.md](../rust/README.md) | [run_model.cpp](../examples/c/run_model.cpp) |
