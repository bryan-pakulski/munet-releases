# GPU vendors

MuNet uses one code path for every GPU: Vulkan compute, plus `VK_KHR_cooperative_matrix` for tensor / matrix cores.
It adapts to each GPU on its own (fragment shapes, subgroup sizes, fp16 denormal handling, tile sizes). You don't
need vendor flags to get correct results, and usually not for speed either.

GPUs MuNet has been validated on, with known issues: [tested-hardware.md](tested-hardware.md).

## See what MuNet detected

```python
import munet as mu
for i in range(len(mu.devices())):
    print(mu.device_info(i))
# {'index': 0, 'name': 'NVIDIA GeForce RTX 3060', 'vendor': 'NVIDIA', 'memory_bytes': 12884901888,
#  'api_version': '1.4.351', 'cooperative_matrix': True, 'fp16_accumulate': True, 'int8': True,
#  'coopmat2': True, 'subgroup_rows': 16, 'fp16_denormals': True}
```

| Field | If False |
|---|---|
| `cooperative_matrix` | `"mixed"` and `"fp16"` run fp32 kernels (correct, slower). Update the driver; older GPUs don't have matrix cores |
| `fp16_accumulate` | `"fp16"` runs like `"mixed"` |
| `int8` | int8 programs compute their quantized layers in fp16 |

## NVIDIA

- **Driver.** The proprietary driver includes Vulkan. Cooperative matrices need a driver that exposes
  `VK_KHR_cooperative_matrix`; check `cooperative_matrix`. The 535 branch (common on cloud images) exposes only
  `VK_NV_cooperative_matrix`, so MuNet runs fp32 kernels there and says so in `driver_warning`. The 595 branch works on
  Turing through Blackwell. See [tested-hardware.md](tested-hardware.md).
- **Precision.** GeForce RTX 30/40 run `"fp16"` about twice as fast as `"mixed"` on matmuls and convolutions.
- **`VK_NV_cooperative_matrix2`.** It is detected (`coopmat2`) and used by an opt-in convolution kernel,
  `MUNET_COOP2_CONV=1`. That kernel is correct but not faster yet, so leave it off. `MUNET_DISABLE_COOP2=1` hides the
  extension entirely.
- **Containers.** The NVIDIA container toolkit exposes Vulkan only with the **graphics** capability:
  ```bash
  docker run --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics ...
  # add ",video" for hardware video decode
  ```
  Without `graphics`, `munet.devices()` is empty. In Kubernetes, set the same variable on the container.

## Intel (Arc, Xe)

- **Driver.** Mesa ANV. Arc B-series (B580/B570) needs Mesa 25.x or newer and the `xe` kernel driver. Newer Mesa is
  noticeably faster.
- **Handled automatically.** 8x16 matrix fragments on 16-wide subgroups, preserving fp16 denormals (ANV flushes them
  by default, which would break training), and Intel-specific tile sizes.
- **Clocks.** Arc cards idle at low clocks and take about a second to ramp up. Warm up before timing; the autotuner
  does this itself.
- **Hardware video decode** (zero-copy pipelines): current Mesa still marks ANV video decode experimental. Enable it with
  `ANV_DEBUG=video-decode`.
- **Containers:** `--device /dev/dri`, and the user must be in the `render` group.

## AMD

- **Driver.** Mesa RADV (`mesa-vulkan-drivers` / `vulkan-radeon`). AMDVLK also works.
- **Matrix cores.** RDNA3 and newer expose cooperative matrices (WMMA) through RADV, used by `"mixed"` / `"fp16"`.
  Older GPUs run the fp32 kernels.
- **Testing.** AMD runs the same portable kernels as NVIDIA and Intel, but has not yet been benchmarked on hardware
  for this release. Run [examples/bench](../examples/bench/README.md) to check a card, and please report results.
- **Containers:** `--device /dev/dri` (Vulkan does not need `/dev/kfd`).

## Mobile and embedded (Arm Mali, Qualcomm Adreno)

These run the fp32 kernels unless the driver exposes cooperative matrices. Some mobile drivers reset the GPU on long
submissions; `MUNET_SUBMIT_KERNELS=64` splits each run into smaller ones.

## Several vendors in one machine

All GPUs are listed by `mu.devices()` and usable at the same time, for example an NVIDIA and an Intel card in one
`DataParallel` run ([multi-gpu.md](multi-gpu.md)). `MUNET_VULKAN_DEVICE=N` picks the GPU that `device="vulkan"`
means.

## Debugging switches

| Variable | Effect |
|---|---|
| `MUNET_DISABLE_COOPMAT=1` | portable fp32 kernels only: rules out matrix-core issues |
| `MUNET_NO_INT8=1` | quantized layers run fp16 |
| `MUNET_NO_FLASH=1` | attention as separate matmul + softmax kernels |
| `MUNET_SUBMIT_SYNC=1` | wait after each submission and name the kernels on failure (GPU resets) |
| `MUNET_PIPELINE_STATS=1` | print registers / spills per kernel from the driver |
| `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation` | Vulkan validation layers (MuNet is expected to be clean) |

The full list is in [environment.md](environment.md).
