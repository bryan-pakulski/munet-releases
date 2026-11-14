# Environment variables

Python, C, C++ and Rust read the same variables, except where a row says otherwise. Set a switch to `1` to turn it
on. Some switches react to any value, so unset them rather than setting `0`.

## Setup

| Variable | Default | Effect |
|---|---|---|
| `MUNET_LICENSE` | | license key ([getting-started.md](getting-started.md#license-key)) |
| `MUNET_LICENSE_FILE` | | file holding the key. Without either variable, MuNet reads `~/.config/munet/license.key` (`%APPDATA%\munet\license.key` on Windows) |
| `MUNET_CACHE_DIR` | `~/.cache/munet-next` | compiled kernels (SPIR-V), driver pipeline caches and autotune results. Copy it to deploy pre-tuned, pre-compiled kernels. A read-only directory disables caching |
| `MUNET_PIPELINE_CACHE_MAX_MB` | `256` | a driver pipeline cache file larger than this is discarded and rebuilt (each program loads the whole file) |
| `MUNET_GLSLANG` | `glslangValidator` on `PATH` | shader compiler path |
| `MUNET_VULKAN_LIBRARY` | system loader | explicit `libvulkan.so` path |
| `MUNET_VULKAN_DEVICE` | `0` | Python only: which GPU `device="vulkan"` means. `"vulkan:N"` always picks N ([gpus.md](gpus.md#choose-a-gpu)) |
| `MUNET_NO_DRIVER_WARNING=1` | off | silence the stderr warning about a driver that hides the tensor cores (e.g. NVIDIA 535) |

## Performance

| Variable | Default | Effect |
|---|---|---|
| `MUNET_AUTOTUNE=1` | off | autotune every compile ([inference.md](inference.md#autotuning)) |
| `MUNET_AUTOTUNE_BUDGET_S` | `30` | seconds of tuning per program |
| `MUNET_MERGE_SIBLINGS=1` | off | ONNX import: merge sibling layers that read the same input. No bench model gains from it. Part of the native-import cache key |
| `MUNET_COOP2_CONV=1` | off | NVIDIA `VK_NV_cooperative_matrix2` convolution kernel (experimental, not faster) |

## Debugging

Switches to isolate a wrong result or a GPU crash.

| Variable | Effect |
|---|---|
| `MUNET_DISABLE_COOPMAT=1` | portable fp32 kernels only (no tensor / matrix cores) |
| `MUNET_DISABLE_COOP2=1` | ignore `VK_NV_cooperative_matrix2` |
| `MUNET_NO_INT8=1` | quantized layers run in fp16 |
| `MUNET_NO_FLASH=1` | attention as separate kernels instead of the fused flash kernel |
| `MUNET_NO_FLASH_BWD=1` | training: attention backward without the fused kernels |
| `MUNET_NO_TILED_POINTWISE=1` | permuted element-wise kernels without shared-memory tiling |
| `MUNET_CHANNELS_LAST=0`, `MUNET_CONCAT_ALIAS=0`, `MUNET_FUSE_Q8=0` | turn off individual layout / fusion passes |
| `MUNET_SUBMIT_KERNELS=N` | split each run into submissions of at most N kernels (drivers that reset on long submissions). Default: one submission |
| `MUNET_SUBMIT_SYNC=1` | wait after every submission; on failure, report the kernel range (find a kernel that resets the GPU) |
| `MUNET_PIPELINE_STATS=1` | print driver statistics (registers, spills) per kernel at load |
| `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation` | Vulkan validation layers |

## Vendor drivers

| Variable | Effect |
|---|---|
| `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics[,video]` | containers: NVIDIA exposes Vulkan only with `graphics` |
| `ANV_DEBUG=video-decode` | Intel Mesa: enable hardware video decode (zero-copy video) |

## Training swarm

| Variable | Effect |
|---|---|
| `MUNET_SWARM_TOKEN` | enrolment secret for `munet-server` and `munet-node` (at least 32 characters) ([swarm.md](swarm.md)) |
| `MUNET_CA_BUNDLE` | `munet-node`: CA bundle file that verifies the owner's TLS certificate (default: the system trust store) |

## Rust crate build

Read by `cargo build` of the `munet-nn` crate.

| Variable | Effect |
|---|---|
| `MUNET_LIB_DIR` | link the prebuilt static C API from this directory (`lib/static` of a `munet-c` archive) instead of downloading |
| `MUNET_ARCHIVE` | use a local `munet-c` archive instead of downloading |
| `MUNET_DOWNLOAD_URL` | override the release download base URL |
| `MUNET_SOURCE_DIR` | build the engine with CMake from this source tree |
| `MUNET_DEBUG` | with a source build: debug CMake configuration |

## Tests and development

Not needed to use MuNet.

| Variable | Effect |
|---|---|
| `MUNET_TEST_VULKAN=1` | run the GPU tests (fail rather than skip without a GPU) |
| `MUNET_TEST_INJECT_DEVICE_LOST=N` | the N-th queue submission reports device loss |
| `MUNET_BENCH_CACHE` | bench suite download cache (default `~/.cache/munet-bench`) |
| `MUNET_BENCH_TRT_IMAGE` | TensorRT container image for `examples/bench/run.py --trt` (default of `--trt-image`) |
| `MUNET_BENCH_TORCH_PYTHON` | Python with torch and torchvision for the bench's local model exports (default: the bench's own) |
| `MUNET_BENCH_LOCAL_DETR` | path of the local 1280 px detector ONNX for the bench's batched detector entries |
| `MUNET_TEST_SWARM`, `MUNET_SWARM_NODE`, `MUNET_SWARM_NODE_WRAPPER` | swarm integration tests |

Kernel-tuning overrides used during development (`MUNET_MM_CFG`, `MUNET_CONV_CFG`, `MUNET_FLASH_TILE`, `MUNET_ACC16`,
...) are not stable interfaces. Use autotuning instead.
