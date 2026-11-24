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
# {'index': 0, 'name': '<GPU name>', 'vendor': 'NVIDIA', 'memory_bytes': 12884901888,
#  'api_version': '1.4.351', 'cooperative_matrix': True, 'fp16_accumulate': True, 'int8': True,
#  'coopmat2': True, 'subgroup_rows': 16, 'fp16_denormals': True, 'driver_warning': ''}
```

`device_info()` without an index describes the GPU that `device="vulkan"` uses.

| Field | Meaning |
|---|---|
| `cooperative_matrix` | False: `"mixed"` runs fp32 kernels (correct, slower); `"fp16"` runs portable fp16 kernels if shaders have fp16 arithmetic, else fp32. Update the driver; older and mobile GPUs don't have matrix cores |
| `fp16_accumulate` | False: `"fp16"` runs like `"mixed"` |
| `int8` | False: int8 programs compute their quantized layers in fp16 |
| `driver_warning` | non-empty: the driver hides the tensor cores (e.g. NVIDIA 535) and an update would enable them. Printed once to stderr; `MUNET_NO_DRIVER_WARNING=1` silences it |

## NVIDIA

- **Driver.** The proprietary driver includes Vulkan. Cooperative matrices need a driver that exposes
  `VK_KHR_cooperative_matrix`; check `cooperative_matrix`. The 535 branch (common on cloud images) exposes only
  `VK_NV_cooperative_matrix`, so MuNet runs fp32 kernels there and says so in `driver_warning`. The 595 branch works on
  Turing through Blackwell.
- **Precision.** GPUs with fast fp16 accumulation run `"fp16"` faster than `"mixed"` on matmuls and convolutions.
- **Idle clocks.** The GPU lowers its clocks after a few seconds of no or sparse work, and what follows runs slowly
  until they recover. On an RTX 3060 an fp32 1024^3 matmul takes 0.43 ms under load, 4.0 ms after 5 idle seconds, and
  needs 0.4 s of sustained work to recover. Warm up before timing; the autotuner and `munet.diagnose` do this
  themselves. A service with seconds between requests sees this on each request.
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

- **Driver.** Mesa ANV. Arc B-series needs Mesa 25.x or newer and the `xe` kernel driver. Newer Mesa is
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
- **Tiles.** `"fp16"` keeps tiles of at most 8 matrix fragments per subgroup: larger ones are several times slower
  on RADV. `probe_device()` measures this on your GPU ([inference.md](inference.md#device-probe)).
- **Testing.** Tuned from user hardware reports, not on hardware of our own. Run `python -m munet.diagnose` and send us
  the file ([hardware-report.md](hardware-report.md)).
- **Containers.** `--device /dev/dri` (Vulkan does not need `/dev/kfd`).

## Mobile and embedded (Arm Mali, Qualcomm Adreno)

Without cooperative matrices these run fp32 kernels, or portable fp16 kernels with `precision="fp16"` where shaders
have fp16 arithmetic.

Three kernel choices depend on the GPU. Built-in rules pick them from the vendor and model:

| Choice | Rule | Override |
|---|---|---|
| Direct fp16 convolution, widest layer it takes | Adreno 7xx: any width; Adreno 6xx: 64 output channels; others: 32 | `MUNET_CONV_DIRECT=N` (`0`: off) |
| GEMM operand staging in shared memory | Qualcomm: whole vec4s; others: single elements | `MUNET_GEMM_VEC_STAGING=0/1` |
| `"fp16"` matrix-core tiles of 16+ fragments per subgroup | NVIDIA: on; others: off | `MUNET_ACC16_WIDE=0/1` |

Single-element staging returns wrong results on some Qualcomm drivers. For a GPU the rules were not written
for, run the device probe once: [inference.md](inference.md#device-probe).

Some mobile drivers reset the GPU on long submissions. Android builds split each run into submissions of 16 kernels;
`MUNET_SUBMIT_KERNELS=N` changes the size on any platform (`0`: never split).

## Several vendors in one machine

`mu.devices()` lists every GPU, and all of them can run at the same time. For example, an NVIDIA and an Intel card can
share one `DataParallel` run ([multi-gpu.md](multi-gpu.md)).

## See also

- Debugging switches (`MUNET_DISABLE_COOPMAT`, `MUNET_SUBMIT_SYNC`, validation layers, ...):
  [environment.md](environment.md#debugging).
