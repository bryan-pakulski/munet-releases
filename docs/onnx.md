# ONNX import and export

- [Import an ONNX model](#import-an-onnx-model)
- [Input shapes](#input-shapes)
- [Inputs and outputs](#inputs-and-outputs)
- [Import a PyTorch model](#import-a-pytorch-model)
- [Export to ONNX](#export-to-onnx)
- [Save as .mnet for deployment](#save-as-mnet-for-deployment)
- [What is supported](#what-is-supported)
- [When import fails](#when-import-fails)

## Import an ONNX model

```python
from munet.interop import from_onnx

model = from_onnx("model.onnx",
                  input_shapes={"images": (1, 3, 640, 640)},   # fix symbolic dims (batch, size)
                  precision="fp16",                            # "fp32" | "mixed" (default) | "fp16"
                  device="vulkan")                             # "vulkan:N" | "cpu"
outputs = model(image)                                         # NumPy in
boxes = outputs.numpy()                                        # one output: a Result; several: a tuple
```

| Option | Default | Meaning |
|---|---|---|
| `input_shapes` | from the file | `{input name: shape}` for symbolic dimensions |
| `precision` | `"mixed"` | see [inference.md](inference.md#precision) |
| `int8` | None | activation scales from `calibrate_onnx`: int8 convolutions ([quantization.md](quantization.md)) |
| `autotune` | off | time kernel tile sizes once per GPU and cache them ([inference.md](inference.md#autotuning)) |
| `optimize` | True | merge layers before import: fold BatchNorm, re-parameterize parallel branches (RepVGG/MobileOne), merge sibling layers, fuse attention. `model.optimizations` lists what was done. |
| `profile` | False | per-kernel GPU times in `model.describe()` |

The first call compiles: it generates and compiles kernels (cached on disk), plans memory, and uploads weights.
Later calls only run.

## Input shapes

Programs have static shapes.

- **With `input_shapes`** (recommended), you get one program for exactly those shapes.
- **Without it,** a model with symbolic dimensions returns a `ShapeSpecializedProgram`. It compiles a new program the
  first time it sees each new input shape and keeps the 8 most recent (`max_specializations=8`). That is handy for
  experiments, but every new shape pays the compile cost.

For production, import once per batch size and resolution you serve.

## Inputs and outputs

- **Inputs** go in model input order as NumPy arrays (anything `np.asarray` accepts). Integer and bool inputs (for
  example `orig_target_sizes` in RT-DETR) are passed as their real types and converted exactly, up to 2^24.
- **Float outputs** are `Result` handles: `.numpy()` reads the data back. Outputs you don't read are never copied off
  the GPU. A `Result` is valid until the next call of the same model.
- **Integer outputs** come back as NumPy arrays of their declared type.
- **NonMaxSuppression outputs** come back as int64 `[num_selected, 3]`, with the padding removed.

```python
labels, boxes, scores = model(image, np.array([[640, 640]], np.int64))
keep = scores.numpy()[0] > 0.5
```

## Import a PyTorch model

```python
from munet.interop import from_torch

model = from_torch(torch_model.eval(), (example_input,), precision="fp16")   # takes every from_onnx option
```

This exports with `torch.onnx.export` (dynamo, opset 18) and imports the result. You need `pip install
'munet-nn[torch]'` (torch + onnxscript). Shapes are those of `example_input`. For older exporters, export the file
yourself (`torch.onnx.export(model, x, "m.onnx", opset_version=17)`) and use `from_onnx`.

## Export to ONNX

Any compiled inference program exports to a standard ONNX graph, whether it was built natively, imported from
ONNX, or loaded from `.mnet`:

```python
from munet.interop import to_onnx

program = mu.compile(model.eval())
program(example)                       # compile once first
to_onnx(program, "model.onnx")
```

- **Exported:** the computation (weights included).
- **Not exported:** the original layer structure, because the graph is MuNet's optimized form.
- **Training steps** can't be exported. Compile the trained model for inference and export that.

## Save as .mnet for deployment

```python
import munet
munet.save(model, "model.mnet")                       # graph, weights, input/output names and dtypes
again = munet.load("model.mnet", precision="fp16")    # Python
```

C, C++ and Rust load the same file (`munet_load`, `munet::Program::load`). `.mnet` stores the computation and
weights, plus int8 scales if the model is quantized. Precision is chosen when loading.

## What is supported

- **Opsets.** ONNX opsets 13-21. Files with opsets 7-12 (YOLOv5-v8, PP-OCR and other older exports) are upgraded
  automatically.
- **Operators.** All 191 default-domain operators are handled; [op_coverage.md](op_coverage.md) is the table.
  `munet.interop.SUPPORTED_ONNX` lists them in code.
- **Quantized models.** QDQ (`QuantizeLinear` / `DequantizeLinear`) and QLinear graphs run on int8 kernels.
- **Resolved at import.** Sequences, optionals, control flow on constants, and recurrent layers (unrolled).
- **Fused automatically.** Attention patterns (`MatMul → [scale] → [+ bias/mask] → Softmax → MatMul`) become one
  flash-attention kernel.
- **Tested end to end.** [examples/bench](../examples/bench/README.md) runs about 40 public models (YOLO v5-v11,
  RT-DETR, D-FINE, DINOv2, CLIP, SAM, Depth-Anything, PP-OCR, ConvNeXt, ...) and compares them against ONNX Runtime.

## When import fails

| Error | Cause | Fix |
|---|---|---|
| `UnsupportedOperatorError: ... data-dependent output size` | an output's size depends on input values (e.g. `NonZero` feeding later ops) | move that part out of the model, or use fixed-size ops (`nn.functional.nms` with `max_output`) |
| `... external tensor data` | weights stored in separate files | `onnx.load(path)` then `onnx.save(model, "single.onnx", save_as_external_data=False)` (models under 2 GB) |
| `declared output shape differs` | the file declares an output shape that doesn't match the computation | re-export, or fix the declared shape with `onnx.shape_inference` |
| NonMaxSuppression consumed inside the graph | dynamic result size | NMS must be a model output |
| values far off in fp16 | the model has activations beyond fp16 range | `precision="mixed"` or `"fp32"`; compare with `infer_onnx.py --compare` |

To check a model's accuracy, run [examples/python/infer_onnx.py](../examples/python/infer_onnx.py) with
`--compare`, which compares against ONNX Runtime. To find the first layer that differs, use
[examples/bench/find_mismatch.py](../examples/bench/find_mismatch.py).
