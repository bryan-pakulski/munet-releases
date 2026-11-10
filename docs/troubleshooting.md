# Troubleshooting

## Setup

| Symptom | Fix |
|---|---|
| `Vulkan JIT needs glslangValidator` | install `glslang-tools` (Debian/Ubuntu) / `glslang` (Arch, Fedora), or set `MUNET_GLSLANG` |
| `munet.devices()` is empty, or a Vulkan error on first use | install the GPU's Vulkan driver ([getting-started.md](getting-started.md)); check `vulkaninfo --summary` |
| No GPU inside Docker (NVIDIA) | `--gpus all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics` ([gpus.md](gpus.md#nvidia)) |
| No GPU inside Docker (Intel / AMD) | `--device /dev/dri`; the user needs the `render` group |
| A notice about evaluation mode | expected without a license key; set `MUNET_LICENSE` or `MUNET_LICENSE_FILE` ([getting-started.md](getting-started.md#license-key)) |
| `MuNet license invalid (...)` or `MuNet license expired (...)` | the key is unreadable, malformed or past its expiry date. MuNet raises instead of falling back to evaluation mode. Fix or renew the key; `munet.license()` shows the status and error |
| `"mixed"` / `"fp16"` no faster than `"fp32"` | `munet.device_info()["cooperative_matrix"]` is False, or `driver_warning` is set (NVIDIA 535): update the driver, or the GPU has no matrix cores ([gpus.md](gpus.md#see-what-munet-detected)) |

## Compiled functions

| Error | Cause | Fix |
|---|---|---|
| `shape guard failed: expected [...], got [...]` | a compiled function or program has fixed input shapes | keep batch shapes constant (`DataLoader(drop_last=True)`), or compile one function per shape |
| `result storage was reused by a later call` | a `Result` was read after the next call of the same program | call `.numpy()` before calling again |
| `data-dependent Python control flow cannot be traced` | `if tensor > 0:` or similar inside a compiled function | `mu.where(cond, a, b)`, or move the decision outside |
| `one optimizer step per trace is supported` | two `optimizer.step()` calls in one compiled function | one optimizer with parameter groups |
| `call loss.backward() before optimizer.step()` | step without backward in the traced function | call `loss.backward()` first |
| `integer input values beyond 2**24 are not exact in float32` | large integer IDs as inputs | remap the IDs to a smaller range, or pass them as float yourself |
| A print or Python side effect runs only once | Python runs only while tracing | return values from the function and print outside |
| Changes to the model have no effect | the structure changed after the program was built | create a new compiled function. Weights can change any time (`load_state_dict`, `assign`) |

Rules of compiled functions: [training.md](training.md#rules-of-compiled-functions).

## Speed

| Symptom | Fix |
|---|---|
| First call takes seconds | compilation; cached in `MUNET_CACHE_DIR`, so later runs are fast. Ship the cache directory to deployment machines ([inference.md](inference.md)) |
| Every call is slow in a loop | you call `.numpy()` / `.item()` every step (forces a wait); read less often |
| Training a symbolic-shape ONNX model is slow | each new shape compiles; pass `input_shapes` |
| Intel Arc timings vary a lot | the GPU ramps clocks after idle: warm up ~1 s and take the minimum of several runs |
| Multi-GPU barely faster than one GPU | the model is too small for host gradient averaging to pay off; use bigger batches. For GPUs of different speed, use `split="auto"` ([multi-gpu.md](multi-gpu.md)) |
| Need a per-kernel breakdown | `profile=True`, then `describe()` ([inference.md](inference.md#measure)) |

## Accuracy

| Symptom | Fix |
|---|---|
| fp16 outputs are `inf` / NaN, or far from fp32 | use `precision="mixed"` (or `"fp32"`) ([inference.md](inference.md#precision)) |
| Training loss stalls or diverges in `"mixed"` | try `precision="mixed_safe"`; compare one step against `"fp32"` |
| int8 accuracy drop | calibrate on real inputs with `observer="mse"`; leave sensitive layers in fp16 with `autoquantize` ([quantization.md](quantization.md)) |
| ONNX model gives different results than ONNX Runtime | `examples/python/infer_onnx.py model.onnx --shape ... --compare --precision fp32`; if fp32 differs, find the layer with `examples/bench/find_mismatch.py` and report it |
| Results differ slightly between GPUs | expected within the precision's rounding (fp16: ~1e-3 relative); fp32 matches closely |

To isolate a wrong result, turn off optimizations one at a time: `fuse=False`, `MUNET_DISABLE_COOPMAT=1`,
`MUNET_NO_FLASH=1`, `MUNET_NO_INT8=1` ([environment.md](environment.md#debugging)).

## GPU crashes and memory

| Symptom | Fix |
|---|---|
| `DeviceLostError` | the GPU reset (driver crash or watchdog timeout). `mu.compile(..., recover="rebuild")` recompiles once and retries the call. Parameters restart from their last read-back value (`.numpy()`), else from the compiled-in value, so training progress since the last read is lost. To find the kernel: `MUNET_SUBMIT_SYNC=1` |
| Device lost on a GPU that also drives a display | the driver's watchdog killed a long submission: `MUNET_SUBMIT_KERNELS=64`, or use a GPU without a display |
| Out of device memory | smaller batch; check `program.stats()["arena_bytes"]`. With DataParallel, cap small GPUs with `max_rows` or use `split="memory"` ([multi-gpu.md](multi-gpu.md#gpus-of-different-size-and-speed)) |
| Vulkan validation errors | run with `VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation` and report them: MuNet is expected to be clean |

## Reporting a bug

Include:

- `munet.__version__`;
- `munet.device_info()` for the GPU involved;
- the driver version (`vulkaninfo --summary`);
- the model or a small script that reproduces it;
- the full error.
