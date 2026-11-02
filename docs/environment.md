# Environment variables

Python, C, C++ and Rust read the same variables. For a boolean switch, `1` turns it on; unset means default.

## Setup

| Variable | Default | Effect |
|---|---|---|
| `MUNET_LICENSE` | | license key |
| `MUNET_LICENSE_FILE` | `~/.config/munet/license.key` | file holding the key |
| `MUNET_CACHE_DIR` | `~/.cache/munet-next` | compiled kernels (SPIR-V), driver pipeline caches and autotune results. Copy it to deploy pre-tuned, pre-compiled kernels; a read-only directory disables caching |
| `MUNET_GLSLANG` | `glslangValidator` on `PATH` | shader compiler path |
| `MUNET_VULKAN_LIBRARY` | system loader | explicit `libvulkan.so` path |
| `MUNET_VULKAN_DEVICE` | `0` | which GPU `device="vulkan"` means (Python); `"vulkan:N"` always picks N |

## Performance

| Variable | Default | Effect |
|---|---|---|
| `MUNET_AUTOTUNE=1` | off | autotune every compile ([inference.md](inference.md#autotuning)) |
| `MUNET_AUTOTUNE_BUDGET_S` | `30` | seconds of tuning per program |
| `MUNET_COOP2_CONV=1` | off | NVIDIA `VK_NV_cooperative_matrix2` convolution kernel (experimental, not faster yet) |

## Debugging

| Variable | Effect |
|---|---|
| `MUNET_DISABLE_COOPMAT=1` | portable fp32 kernels only (no tensor / matrix cores) |
| `MUNET_DISABLE_COOP2=1` | ignore `VK_NV_cooperative_matrix2` |
| `MUNET_NO_INT8=1` | quantized layers run in fp16 |
| `MUNET_NO_FLASH=1` | attention as separate kernels instead of the fused flash kernel |
| `MUNET_NO_FLASH_BWD=1` | training: attention backward without the fused kernels |
| `MUNET_NO_TILED_POINTWISE=1` | permuted element-wise kernels without shared-memory tiling |
| `MUNET_CHANNELS_LAST=0`, `MUNET_CONCAT_ALIAS=0`, `MUNET_FUSE_Q8=0` | turn off individual layout / fusion passes (isolate a wrong result) |
| `MUNET_SUBMIT_KERNELS=N` | split each run into submissions of at most N kernels (drivers that reset on long submissions) |
| `MUNET_SUBMIT_SYNC=1` | wait after every submission; on failure, report the kernel range (find a kernel that resets the GPU) |
| `MUNET_PIPELINE_STATS=1` | print driver statistics (registers, spills) per kernel at load |
| `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation` | Vulkan validation layers |

## Vendor drivers

| Variable | Effect |
|---|---|
| `NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics[,video]` | containers: NVIDIA exposes Vulkan only with `graphics` |
| `ANV_DEBUG=video-decode` | Intel Mesa: enable hardware video decode (zero-copy video) |

## Rust crate build

| Variable | Effect |
|---|---|
| `MUNET_LIB_DIR` | link a prebuilt C API from this directory instead of downloading |
| `MUNET_ARCHIVE` | use a local `munet-c` archive |
| `MUNET_DOWNLOAD_URL` | override the release download base URL |

## Tests and development

| Variable | Effect |
|---|---|
| `MUNET_TEST_VULKAN=1` | run the GPU tests (fail rather than skip without a GPU) |
| `MUNET_TEST_INJECT_DEVICE_LOST=N` | the N-th queue submission reports device loss |
| `MUNET_BENCH_CACHE` | bench suite download cache (default `~/.cache/munet-bench`) |
| `MUNET_SWARM_TOKEN`, `MUNET_CA_BUNDLE` | training swarm enrolment secret and TLS CA ([swarm.md](swarm.md)) |
| `MUNET_TEST_SWARM`, `MUNET_SWARM_NODE`, `MUNET_SWARM_NODE_WRAPPER` | swarm integration tests |

Kernel-tuning overrides used during development (`MUNET_MM_CFG`, `MUNET_CONV_CFG`, `MUNET_FLASH_TILE`, `MUNET_ACC16`,
...) are not stable interfaces. Use autotuning instead.
