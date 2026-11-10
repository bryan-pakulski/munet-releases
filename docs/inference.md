# Inference and performance

How to run models fast from Python, and what carries over to C, C++ and Rust.

## Load once, call many times

```python
model = from_onnx("model.onnx", precision="fp16", input_shapes={"images": (8, 3, 640, 640)})   # or munet.load("model.mnet")
model(warmup_batch)                    # first call compiles (seconds; cached on disk after the first time)
for batch in batches:
    out = model(batch)                 # replays a recorded GPU command buffer
    detections = out.numpy()
```

What a call does:

- **Weights and intermediates stay on the GPU.** A call uploads the inputs as part of the same GPU submission, runs
  every kernel, and reads back only the outputs you `.numpy()`.
- **Compiling is the expensive part.** Keep the program object alive, and don't rebuild it per request.
- **`.mnet` loads faster than ONNX import.** It skips the ONNX parsing and graph optimization, and is what C, C++ and
  Rust load ([onnx.md](onnx.md#save-as-mnet-for-deployment)).
- **Threads.** One program serializes its own calls. For parallel requests, load one program per thread or per GPU.
  Calls release the GIL.

### Kernel cache

Compiled kernels (SPIR-V), driver pipeline caches and autotune results live in `MUNET_CACHE_DIR` (default
`~/.cache/munet-next`). Python, C, C++ and Rust share it.

- Copy the directory to deployment machines to skip compilation there.
- A read-only directory disables caching; programs still run.

## Precision

| `precision=` | Matmul / conv arithmetic | Speed | Accuracy |
|---|---|---|---|
| `"fp32"` | float32 | 1x | exact reference |
| `"mixed"` (default) | fp16 inputs, fp32 accumulation, on tensor / matrix cores | ~2-3x fp32 | relative error ~1e-3; safe default |
| `"mixed_safe"` | as `"mixed"`, plus rescaled matmul gradients | as `"mixed"` | training only: use it when training stalls or diverges in `"mixed"` |
| `"fp16"` | fp16 inputs and accumulation | up to 2x `"mixed"` on GPUs with fast fp16 accumulation (NVIDIA GeForce RTX 30/40), same as `"mixed"` elsewhere | validate per model |

- Everything outside matmuls and convolutions (activations, normalization, softmax) computes in fp32 in every mode.
- On GPUs without cooperative-matrix support, every mode runs fp32 kernels.

Start with `"fp16"` for inference. If outputs differ too much from `"fp32"`, use `"mixed"`.
[examples/python/infer_onnx.py](https://github.com/bryan-pakulski/munet-releases/blob/main/examples/python/infer_onnx.py) `--compare` checks against ONNX Runtime.

## Autotuning

By default MuNet picks kernel tile sizes with heuristics. Autotuning times the candidates on your GPU once and keeps
the fastest. That typically gains 5-10% on large models, more on unusual shapes.

```python
model = from_onnx("model.onnx", input_shapes=..., precision="fp16", autotune=True)
model(example)                          # tunes during this call (default budget: 30 s per program)
```

- **Shared cache.** Results are cached per GPU model, driver, precision and kernel shape in the kernel cache. Every
  later load of the same shapes on the same GPU uses them, from Python, C, C++ or Rust.
- **Tune what you deploy.** Run it once per GPU model, with the batch sizes and precision you ship. To deploy, copy
  the cache directory along with the `.mnet`, or run the tuning step on the target machine.
- **Budget.** `MUNET_AUTOTUNE_BUDGET_S=120` raises the per-program time limit. `MUNET_AUTOTUNE=1` turns tuning on
  for every compile.

## Batching

Per-call overhead is fixed, and GPUs need parallel work. Batching several images per call raises throughput,
especially for small models.

- **Static batch size.** The batch size is part of the program. Import or `.mnet`-export one program per batch size
  you serve, e.g. `b1` for latency and `b8` for throughput.
- **Several GPUs:** one program per GPU, see [multi-gpu.md](multi-gpu.md#inference-on-several-gpus).
- **High-resolution frames:** split them into tiles and run the tiles as one batch. The C/C++/Rust frame pipeline
  does this on the GPU ([zero-copy.md](zero-copy.md)).

## int8

CNN convolutions can run in int8 after calibration. The gain depends on model size: large convolutions gain most,
tiny models little. Check accuracy on your validation set.

How to calibrate and choose layers: [quantization.md](quantization.md). Runnable example:
[examples/python/quantize_int8.py](https://github.com/bryan-pakulski/munet-releases/blob/main/examples/python/quantize_int8.py).

## Measure

```python
model.stats()["gpu_ms"]       # GPU time of the last call (compute only)
model.stats()                 # kernels, memory (arena_bytes), upload/download bytes, compile times, int8 kernel counts, device
```

For a per-kernel breakdown, compile with `profile=True`. It serializes the kernels, so use it to find hot spots, not
to measure throughput:

```python
model = from_onnx("model.onnx", input_shapes=..., precision="fp16", profile=True)
model(x)
for k in sorted(model.describe(), key=lambda k: -k["gpu_ms"])[:10]:
    print(f"{k['gpu_ms']:.3f} ms {k['path']:24s} {k['names'][:3]}")
```

`MUNET_PIPELINE_STATS=1` prints each kernel's register and spill counts from the driver.

When timing:

- **Warm up first.** Some GPUs (Intel Arc especially) idle at low clocks. Run for about a second before timing,
  and use the minimum of several runs.
- **The first call includes compilation.** Exclude it.
- **End-to-end time** = upload + `gpu_ms` + readback. For big inputs, upload dominates on small models. The Vulkan
  zero-copy path in C/C++/Rust removes it ([zero-copy.md](zero-copy.md)).

## Checklist

1. Fix shapes with `input_shapes`. Don't use symbolic shapes in production.
2. Use `precision="fp16"` (or `"mixed"` if accuracy needs it).
3. Run `autotune=True` once on each GPU model you deploy to.
4. Batch requests. Export one program per batch size.
5. For CNNs, try int8 and measure accuracy.
6. Keep programs alive. Use `.mnet` for fast loading and for C/C++/Rust.
7. Read back only the outputs you need, as late as possible.
8. For video, use the GPU frame pipeline and zero copy ([zero-copy.md](zero-copy.md)).

GPU-specific settings: [gpus.md](gpus.md).

## Example: object detection end to end

[examples/python/detect_yolo.py](https://github.com/bryan-pakulski/munet-releases/blob/main/examples/python/detect_yolo.py) covers the whole path for a YOLO model:

- letterbox pre-processing;
- GPU inference;
- decoding and NMS in NumPy;
- drawing the boxes;
- saving a `.mnet`.

```bash
curl -LO https://github.com/jamjamjon/assets/releases/download/yolo/v11-n-det.onnx
python examples/python/detect_yolo.py v11-n-det.onnx street.jpg --precision fp16
```

DETR-style detectors (RT-DETR, D-FINE, DEIM) output `labels, boxes, scores` directly and need no NMS. Take
`scores > threshold`.
