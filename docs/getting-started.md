# Getting started

Install MuNet and check that it runs on your GPU.

## 1. Install a Vulkan driver

MuNet runs on any GPU through its Vulkan driver. Most desktop Linux installs already have one; check with
`vulkaninfo --summary` (package `vulkan-tools`).

| GPU | Debian / Ubuntu | Arch | Fedora |
|---|---|---|---|
| NVIDIA | the proprietary driver (`nvidia-driver-XXX`, which includes Vulkan) | `nvidia-utils` | `xorg-x11-drv-nvidia` (RPM Fusion) |
| AMD | `mesa-vulkan-drivers` (RADV) | `vulkan-radeon` | `mesa-vulkan-drivers` |
| Intel | `mesa-vulkan-drivers` (ANV) | `vulkan-intel` | `mesa-vulkan-drivers` |

- Intel Arc B-series (Battlemage) needs Mesa 25.x or newer and the `xe` kernel driver.
- GPU-specific settings and containers: [gpus.md](gpus.md).

## 2. Install the shader compiler

MuNet generates GPU kernels for your model and compiles them with `glslangValidator`. Compiled kernels are cached on
disk, so this happens once per model and GPU.

```bash
sudo apt install glslang-tools        # Debian/Ubuntu
sudo pacman -S glslang                # Arch
sudo dnf install glslang              # Fedora
```

If it is not on `PATH`, set `MUNET_GLSLANG=/path/to/glslangValidator`.

## 3. Install MuNet

```bash
python -m pip install 'munet-nn[interop]'
```

Wheels exist for CPython 3.10 to 3.14 on Linux x86_64 and aarch64 (manylinux, glibc 2.28 or newer).

| Extra | Adds |
|---|---|
| (none) | the library; NumPy is its only dependency |
| `[interop]` | ONNX import/export (`onnx`) |
| `[torch]` | PyTorch import with `from_torch` (`torch`, `onnx`, `onnxscript`) |
| `[vk]` | `onnx` and `onnxruntime`, to compare results with ONNX Runtime. It does not install a driver. |

- `import munet` and `import munet_nn` load the same library.
- Example model code (RT-DETR and others) is not in the wheel. It lives in the source examples.

C, C++ and Rust: see [c.md](c.md) and [rust/README.md](../rust/README.md).

## 4. Check it works

```bash
python - <<'EOF'
import numpy as np, munet as mu
print(mu.devices())                                    # ['<GPU name>', ...]
f = mu.compile(lambda x: (x @ x.T).relu())             # device="vulkan" (GPU 0) by default
print(f(np.random.rand(256, 256).astype(np.float32)).numpy().shape, f.stats()["device"])
EOF
```

- **`mu.devices()` raises a Vulkan error or returns an empty list:** the loader or driver is missing or not visible.
  In containers, see the notes for your vendor in [gpus.md](gpus.md).
- **An error about glslangValidator:** repeat step 2.
- **The evaluation notice** is expected without a license key.

## License key

Without a key, GPU execution runs in evaluation mode and prints one notice to stderr.

MuNet reads the key from the first of these that is set:

1. `MUNET_LICENSE`: the key itself.
2. `MUNET_LICENSE_FILE`: a file that contains the key.
3. `~/.config/munet/license.key`.

Check the status:

```python
import munet
print(munet.license())   # {'status': 'licensed', 'customer': ..., 'expires': ..., 'features': [...], 'source': ..., ...}
```

- `status` is `licensed`, `evaluation`, `invalid` or `expired`. For `invalid` and `expired`, `error` gives the reason.
- An invalid or expired key is not evaluation mode: GPU execution fails with "MuNet license invalid ... requires a
  valid license key". An unreadable `MUNET_LICENSE_FILE` counts as invalid.
- C: `munet_license_json`. Rust: `munet::license()`.
- The bindings and wrappers (Python, C header, Rust crate) are MIT. The native engine is proprietary; see LICENSE.

## Platforms and requirements

- Published wheels are Linux only: x86_64 and aarch64.
- CPU execution (`device="cpu"`) needs no Vulkan. MuNet loads Vulkan only when you use a GPU device.
- An unavailable GPU raises an error. MuNet never falls back to the CPU silently.
- `MUNET_VULKAN_LIBRARY` sets the Vulkan loader path. Set it before the first GPU call.
- Compiling new graphs for a GPU needs `glslangValidator` (step 2).

## Next steps

| Task | Guide | Example |
|---|---|---|
| Run an ONNX model | [onnx.md](onnx.md), [inference.md](inference.md) | [infer_onnx.py](../examples/python/infer_onnx.py) |
| Detect objects in an image | [inference.md](inference.md) | [detect_yolo.py](../examples/python/detect_yolo.py) |
| Train a model | [training.md](training.md) | [train_classifier.py](../examples/python/train_classifier.py) |
| Train on several GPUs | [multi-gpu.md](multi-gpu.md) | [train_multi_gpu.py](../examples/python/train_multi_gpu.py) |
| Train across machines | [pool.md](pool.md) | |
| int8 inference | [quantization.md](quantization.md) | [quantize_int8.py](../examples/python/quantize_int8.py) |
| Deploy from C/C++/Rust | [c.md](c.md), [rust/README.md](../rust/README.md) | [run_model.cpp](../examples/c/run_model.cpp) |
