# MuNet releases

Prebuilt MuNet binaries: a neural-network runtime that compiles models (ONNX imports) into fused Vulkan compute
kernels and runs them on any Vulkan GPU — AMD, Intel, NVIDIA, Apple (MoltenVK) and mobile — with fp16/int8
cooperative-matrix kernels where the GPU has them.

| Package | Install |
|---|---|
| Python | `pip install munet-nn` |
| Rust | `cargo add munet-nn` (imported as `munet`; the build script downloads the C API below and verifies its SHA-256) |
| C | `munet-c-<version>-<os>-<arch>` archives on the [releases](https://github.com/bryan-pakulski/munet-releases/releases): `include/munet/munet_c.h`, shared and static libraries |
| Native tools | `munet-node`, `munet-server` and the C++ SDK archives on the releases |

Linux x86_64 and aarch64 (glibc 2.28+) are built for every release. Running on a GPU needs a Vulkan 1.1 driver and
`glslangValidator` (`glslang-tools` / `glslang` packages or the Vulkan SDK).

<!-- munet-help:begin -->
## Help tune MuNet for your GPU

MuNet runs on any Vulkan GPU, but its kernels are tuned only on GPUs we have. If yours is slow, gives wrong results
or isn't detected, one command collects what we need to fix it:

```bash
pip install munet-nn
python -m munet.diagnose                 # 5-10 minutes per GPU; --quick for 1-2
```

Then [open a hardware report](https://github.com/bryan-pakulski/munet-releases/issues/new?template=hardware-report.yml)
and attach the `munet-diag-<gpu>.json.gz` file. AMD Radeon, Apple Silicon, Intel integrated / Arc A-series and mobile
GPUs are especially welcome: we have no numbers for them yet. What the file contains:
[Hardware report](docs/hardware-report.md).
<!-- munet-help:end -->

<!-- munet-docs:begin -->
## Documentation

| I want to... | Read |
|---|---|
| Install MuNet and check my GPU works | [Getting started](docs/getting-started.md) |
| Train a model | [Training](docs/training.md) |
| Feed training data (datasets, shuffling, workers, prefetch) | [Data loading](docs/data.md) |
| Train on several GPUs, including different sizes (e.g. 24 + 12 + 8 GB) | [Multiple GPUs](docs/multi-gpu.md) |
| Train across machines (owner and worker nodes) | [Training swarm](docs/swarm.md) |
| Import or export ONNX, or import PyTorch | [ONNX](docs/onnx.md) |
| Run models fast (precision, autotuning, batching, `.mnet`) | [Inference and performance](docs/inference.md) |
| Quantize to int8 / int4 | [Quantization](docs/quantization.md) |
| Configure NVIDIA, AMD or Intel GPUs, and containers | [GPU vendors](docs/gpus.md) |
| Check which GPUs have been tested | [Tested hardware](docs/tested-hardware.md) |
| Report how MuNet runs on my GPU | [Hardware report](docs/hardware-report.md) |
| Run models from C or C++ | [C and C++](docs/c.md) |
| Run models from Rust | [Rust](rust/README.md) |
| Detect objects in video with zero copies | [Zero copy and video](docs/zero-copy.md) |
| Look up an environment variable | [Environment variables](docs/environment.md) |
| Fix an error | [Troubleshooting](docs/troubleshooting.md) |
| Look up a Python function | [Python API](docs/python.md) |

Runnable examples are in [examples/python](examples/python/README.md), [examples/c](examples/c/run_model.cpp) and
[rust/examples](rust/examples). [examples/bench](examples/bench/README.md) checks correctness and speed of about 40
public models on your GPUs.

All docs: [docs/README.md](docs/README.md).
<!-- munet-docs:end -->

## License

The MuNet engine binaries are proprietary (see [LICENSE](LICENSE)). Without a license key they run in evaluation mode;
licensed use needs a key in `MUNET_LICENSE`, `MUNET_LICENSE_FILE` or `~/.config/munet/license.key`. The Python and
Rust bindings are MIT licensed. To obtain a license, open an issue in this repository.

Every release lists `SHA256SUMS` for its files.
