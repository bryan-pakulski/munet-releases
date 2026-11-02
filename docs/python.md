# Python API reference

`pip install munet-nn`, then `import munet as mu` (`import munet_nn` also works). Each section links to the guide
that explains it.

## Top level: `munet`

| Name | What it does |
|---|---|
| `compile(fn=None, *, device="vulkan", precision="mixed", fuse=True, profile=False, autotune=None, recover=None)` | trace `fn` (a function or a module) into a GPU program; usable as `@mu.compile` or `@mu.compile(device=...)` ([training.md](training.md)) |
| `devices()` | names of the Vulkan GPUs; `"vulkan:N"` is the N-th |
| `device_info(index=None)` | memory, vendor, matrix-core support of a GPU ([gpus.md](gpus.md)) |
| `save(program, path)` / `load(path, *, device, precision)` | `.mnet` files for Python, C, C++, Rust ([onnx.md](onnx.md#save-as-mnet-for-deployment)) |
| `DataParallel(loss_fn, parameters, optimizer, devices, *, precision, split, max_rows)` | multi-GPU training: `.step(*batch)`, `.counts`, `.stats()` ([multi-gpu.md](multi-gpu.md)) |
| `quantize(model, calib_data=None, *, scheme)` | quantize `Linear` layers in place ([quantization.md](quantization.md)) |
| `Tensor`, `Parameter`, `Buffer` | traced values, trainable weights, non-trained state |
| `cat`, `stack`, `where`, `as_tensor`, `grad` | tensor functions |
| `license()` | license status |
| `VulkanError`, `DeviceLostError` | GPU errors (`DeviceLostError` after a GPU reset) |
| `metrics` | counters / histograms; `metrics.prometheus()` for scraping |

## Compiled programs

Returned by `mu.compile`, `from_onnx`, `from_torch` and `mu.load`.

| Member | |
|---|---|
| `program(*arrays)` | run; returns a `Result` or a tuple of them (integer outputs of ONNX models: NumPy arrays) |
| `Result.numpy()`, `.item()`, `.shape` | read back; valid until the next call |
| `stats()` | `gpu_ms` (last call), `kernels`, `arena_bytes`, upload/download bytes, compile times, int8 kernel counts, `device` |
| `describe()` | per kernel: path, fused ops, ONNX node names, `gpu_ms` with `profile=True` |
| `optimizations` | (ONNX imports) layer merges applied |

## Data: `munet.data` ([data.md](data.md))

| Name | |
|---|---|
| `DataLoader(dataset, batch_size, *, shuffle, drop_last=True, seed, workers, prefetch, collate, shard)` | batches of NumPy arrays; `set_epoch(n)`, `len()` |
| `ArrayDataset(*arrays, transform=None)` | samples are rows of arrays |
| `Dataset` | optional base class (`__len__`, `__getitem__`) |
| `default_collate(samples)` | stacks samples field by field |

## Layers and functions: `munet.nn` ([training.md](training.md#models))

- **Modules:** `Module` (`parameters()`, `named_parameters()`, `state_dict()`, `load_state_dict()`, `train()`,
  `eval()`, `requires_grad_()`), plus layers in the PyTorch style: `Linear`, `Conv1d/2d`, `ConvTranspose1d/2d`,
  `BatchNorm1d/2d`, `LayerNorm`, `GroupNorm`, `RMSNorm`, `MultiheadAttention`, `Transformer*`, `LSTM`, `GRU`,
  `Embedding`, pooling, activations, and the `*Loss` classes.
- **Functional:** `munet.nn.functional` holds activations, losses, `scaled_dot_product_attention`, `interpolate`,
  `grid_sample`, `pad`, `conv*`, normalization, `roi_align`, `nms`, `batched_nms`, `einsum`, and more.

## Optimizers: `munet.optim` ([training.md](training.md#optimizers-and-learning-rate-schedules))

| Name | |
|---|---|
| `SGD(params, lr, momentum=0, weight_decay=0, nesterov=False, dampening=0)` | parameter groups; live `param_groups[i]["lr"]` |
| `AdamW(params, lr=1e-3, betas, eps, weight_decay=0.01)` | parameter groups; live settings |
| `MultiStepLR(optimizer, milestones, gamma)` | step per epoch |
| `ModelEMA(model, decay, warmups)` | `update()` inside the step, `copy_to(model)` |
| `clip_grad_norm_(params, max_norm)` | between `backward()` and `step()` |

All have `state_dict()` / `load_state_dict()`.

## Checkpoints: `munet.checkpoint`

`save_state(tree, path)` / `load_state(path)`: atomic, data-only archives of nested dicts / lists of float32 arrays
and plain values ([training.md](training.md#checkpoints-and-resuming)).

## ONNX and PyTorch: `munet.interop` ([onnx.md](onnx.md))

| Name | |
|---|---|
| `from_onnx(path, *, device, precision, input_shapes, int8, autotune, optimize, profile)` | import |
| `from_torch(module, example_inputs, **from_onnx_options)` | import a PyTorch module through ONNX |
| `to_onnx(program, path)` | export |
| `SUPPORTED_ONNX` | supported operator names |

## Quantization: `munet.quantization` ([quantization.md](quantization.md))

| Name | |
|---|---|
| `calibrate_onnx(path, batches, *, observer="mse")` | int8 activation scales for `from_onnx(int8=...)` |
| `autoquantize(path, calibration, validation, *, metric, tolerance)` | choose int8 layers by measured speed and accuracy |
| `quantize(model, calib_data, *, scheme)` | `w8a8_dynamic`, `w8a8`, `w8`, `w4` for `Linear` layers |
| `compare(reference, candidate, inputs)` | cosine / max / relative error |

## Training across machines: `munet.swarm`

`create_job(...)` and the `munet-server` / `munet-node` commands: see [swarm.md](swarm.md).
