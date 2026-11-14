# GPU vendors

MuNet uses one code path for every GPU: Vulkan compute, plus `VK_KHR_cooperative_matrix` for tensor / matrix cores.
It adapts fragment shapes, subgroup sizes, fp16 denormal handling and tile sizes to each GPU itself. You don't need
vendor flags for correct results, and usually not for speed either.

## Choose a GPU

- `mu.devices()` lists the Vulkan GPUs by name.
- `device="vulkan:N"` picks the N-th entry.
- `device="vulkan"` means GPU 0, unless `MUNET_VULKAN_DEVICE=N` is set. This moves a whole script or test suite to
  another GPU without code changes. Only the Python package reads this variable.
- `device="cpu"` is the reference executor: slow, meant for tests. MuNet never falls back to it silently.

```bash
MUNET_VULKAN_DEVICE=1 python train.py
```

From C, set `options.device = N` (`-1` for the CPU). From Rust, use `Device::Vulkan(N)` or `Device::Cpu`.

## See what MuNet detected

```python
import munet as mu
for i in range(len(mu.devices())):
    print(mu.device_info(i))
# {'index': 0, 'name': 'NVIDIA GeForce RTX 3060', 'vendor': 'NVIDIA', 'memory_bytes': 12884901888,
#  'api_version': '1.4.351', 'cooperative_matrix': True, 'fp16_accumulate': True, 'int8': True,
#  'coopmat2': True, 'subgroup_rows': 16, 'fp16_denormals': True, 'driver_warning': ''}
```

`device_info()` without an index describes the GPU that `device="vulkan"` uses.

| Field | Meaning |
|---|---|
| `cooperative_matrix` | False: `"mixed"` and `"fp16"` run fp32 kernels (correct, slower). Update the driver; older GPUs don't have matrix cores |
| `fp16_accumulate` | False: `"fp16"` runs like `"mixed"` |
| `int8` | False: int8 programs compute their quantized layers in fp16 |
| `driver_warning` | non-empty: the driver hides the tensor cores (e.g. NVIDIA 535) and an update would enable them. Printed once to stderr; `MUNET_NO_DRIVER_WARNING=1` silences it |

## NVIDIA

- **Driver.** The proprietary driver includes Vulkan. Cooperative matrices need a driver that exposes
  `VK_KHR_cooperative_matrix`; check `cooperative_matrix`. The 535 branch (common on cloud images) exposes only
  `VK_NV_cooperative_matrix`, so MuNet runs fp32 kernels there and says so in `driver_warning`. The 595 branch works on
  Turing through Blackwell. See [tested-hardware.md](tested-hardware.md).
- **Precision.** GeForce RTX 30/40 run `"fp16"` about twice as fast as `"mixed"` on matmuls and convolutions.
- **`VK_NV_cooperative_matrix2`.** It is detected (`coopmat2`) and used by an opt-in convolution kernel,
  `MUNET_COOP2_CONV=1`. That kernel is correct but not faster, so leave it off. `MUNET_DISABLE_COOP2=1` hides the
  extension entirely.
- **Containers.** The NVIDIA container toolkit exposes Vulkan only with the `graphics` capability:
  ```bash
  docker run --gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics ...
  # add ",video" for hardware video decode
  ```
  Without `graphics`, MuNet finds no GPU. In Kubernetes, set the same variable on the container.

## Intel (Arc, Xe)

- **Driver.** Mesa ANV. Arc B-series (B580/B570) needs Mesa 25.x or newer and the `xe` kernel driver. Newer Mesa is
  noticeably faster.
- **Handled automatically.** 8x16 matrix fragments on 16-wide subgroups, Intel tile sizes, and preserved fp16
  denormals (ANV flushes them by default, which would break training).
- **Clocks.** Arc cards idle at low clocks and take about a second to ramp up. Warm up before timing; the autotuner
  does this itself.
- **Hardware video decode** (zero-copy pipelines). Mesa still marks ANV video decode experimental. Enable it with
  `ANV_DEBUG=video-decode`.
- **Containers.** `--device /dev/dri`, and the user must be in the `render` group.

## AMD

- **Driver.** Mesa RADV (`mesa-vulkan-drivers` / `vulkan-radeon`). AMDVLK also works.
- **Matrix cores.** RDNA3 and newer expose cooperative matrices (WMMA) through RADV, used by `"mixed"` / `"fp16"`.
  Older GPUs run the fp32 kernels.
- **Testing.** AMD runs the same portable kernels as NVIDIA and Intel, but is not yet tested on hardware. Run
  `python -m munet.diagnose` and send us the file ([hardware-report.md](hardware-report.md)).
- **Containers.** `--device /dev/dri` (Vulkan does not need `/dev/kfd`).

## Mobile and embedded (Arm Mali, Qualcomm Adreno)

These run the fp32 kernels unless the driver exposes cooperative matrices.

Some mobile drivers reset the GPU on long submissions. `MUNET_SUBMIT_KERNELS=64` splits each run into smaller ones.

## Several vendors in one machine

`mu.devices()` lists every GPU, and all of them can run at the same time. For example, an NVIDIA and an Intel card can
share one `DataParallel` run ([multi-gpu.md](multi-gpu.md)).

## See also

- Debugging switches (`MUNET_DISABLE_COOPMAT`, `MUNET_SUBMIT_SYNC`, validation layers, ...):
  [environment.md](environment.md#debugging).
- Validated GPUs, known issues and speed: [tested-hardware.md](tested-hardware.md).
