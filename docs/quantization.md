# Quantization

| You have | Do this |
|---|---|
| An ONNX CNN (YOLO, ResNet, RT-DETR...) | [calibrate and import with int8 convolutions](#int8-for-onnx-models) |
| An ONNX model and a validation metric, and you want the best speed/accuracy trade-off | [`autoquantize`](#pick-layers-automatically) |
| An already-quantized ONNX model (QDQ or QLinear) | [import it as is](#pre-quantized-onnx-models) |
| A MuNet model with `Linear` layers (MLPs, transformers, LLM-style decode) | [`munet.quantize`](#quantize-munet-linear-layers) |
| Accuracy loss to recover | [quantization-aware training](#quantization-aware-training) |

All int8 programs save to `.mnet` with their scales. C, C++ and Rust load them as int8 with no extra setup.

## int8 for ONNX models

```python
from munet.quantization import calibrate_onnx
from munet.interop import from_onnx
import munet

calib = [(load_image(p),) for p in calibration_paths]   # 50-500 inputs like those you serve, as model input tuples
scales = calibrate_onnx("model.onnx", calib)             # {conv name: activation scale}
model = from_onnx("model.onnx", input_shapes={"images": (1, 3, 640, 640)}, precision="fp16", int8=scales)
munet.save(model, "model-int8.mnet")
```

- **Which convolutions run in int8.** Those with constant weights, a spatial kernel (not 1x1), and a multiple of 32
  channels per group. The rest stay fp16; `model.stats()["int8_conv_kernels"]` counts the int8 ones.
- **How values are quantized.** Weights per output channel, activations per tensor, both symmetric int8.
- **Calibration data matters more than its amount.** Use real inputs from your domain. The default observer `"mse"`
  is much better than `"minmax"` for detectors.
- **Check accuracy** on your validation set with your task metric before shipping.
- **Partial selection.** Pass a subset of `scales` to quantize only some layers. Sensitive layers are often the
  first convolution and a few encoder layers.

Runnable: [examples/python/quantize_int8.py](../examples/python/quantize_int8.py) (calibrate, compare with fp16, time,
save).

## Pick layers automatically

`autoquantize` measures each candidate layer's speed gain and accuracy cost on your GPU. It keeps the largest set of
layers whose combined accuracy stays within a tolerance:

```python
from munet.quantization import autoquantize

def metric(reference, candidate):           # lower is better; lists of per-batch output tuples
    return 1.0 - my_map50(candidate)        # e.g. detection AP against ground truth (ignore reference)

result = autoquantize("model.onnx", calibration_batches, validation_batches, metric=metric, tolerance=0.005)
result.save("int8-selection.json")
model = result.compile("model.onnx", precision="fp16")     # = from_onnx(path, int8=result.int8)
```

- **Candidates** are convolutions and MatMul/Gemm layers with constant weights.
- **Default metric:** without `metric`, the relative error against fp32 outputs.
- **Runtime:** minutes to tens of minutes for a large model. Save the result and reuse it.

## Pre-quantized ONNX models

QDQ graphs (`QuantizeLinear` / `DequantizeLinear` around Conv/MatMul) and QLinear operators (`QLinearConv`,
`QLinearMatMul`, `MatMulInteger`, `ConvInteger`) import directly:

```python
model = from_onnx("model-qdq.onnx", input_shapes=...)
```

- **int8 kernels:** symmetric int8 QDQ `MatMul`s run on the int8 GEMM.
- **Float path:** other quantized operators run as quantize → float op → dequantize, which is exact but not faster
  than fp16.

## Quantize MuNet Linear layers

```python
import munet as mu

model.eval()
mu.quantize(model, scheme="w8a8_dynamic")            # in place: Linear -> QuantLinear
program = mu.compile(model)
```

| Scheme | Weights | Activations | Best for | Measured |
|---|---|---|---|---|
| `w8a8_dynamic` | int8 per channel | int8 per row, computed at run time | large batches (prefill, big MLPs) | ~2x fp16 on large GEMMs |
| `w8a8` | int8 per channel | int8 per tensor, calibrated (pass `calib_data`) | same, slightly cheaper | |
| `w8` | int8 per channel | fp16 | batch 1-8 (decode): memory-bound | 2.3x fp16 at batch 1, half the memory |
| `w4` | int4, groups of 128 | fp16 | batch 1-8, memory-limited | 4x fp16 at batch 1, quarter the memory |

The measurements are from an RTX 3060. Above 8 rows the weight-only schemes run about as fast as fp16 and only save
memory.

Check accuracy with `mu.quantization.compare(fp32_program, quantized_program, inputs)`, which returns the cosine,
max error and relative error.

## Quantization-aware training

Quantized layers train like normal ones: gradients pass straight through the rounding. Quantize, then fine-tune
with a low learning rate:

```python
mu.quantize(model, calib_batches, scheme="w8a8")
model.train()
# ... the usual compiled train step and loop (training.md), a few epochs at lr ~1e-5 ...
```

## Notes

- **Measured on a detector.** An RT-DETR-style detector at 1280x1280: fp16 23.7 ms, mAP50 83.38; int8 on 25 of 32
  eligible convolutions 21.1 ms, mAP50 83.29; all 32: 19.8 ms, mAP50 82.36. The gain grows with convolution size;
  tiny models gain little.
- **What stays fp16:** 1x1 convolutions, asymmetric (zero-point) ranges, and batched right-hand matmul operands.
- **Accuracy without a GPU.** The quantize-dequantize arithmetic also runs on CPU and fp32 plans, so you can check a
  selection's accuracy on any machine.
- **Export.** `to_onnx` exports quantized programs as QDQ (`QuantizeLinear` / `DequantizeLinear`).
