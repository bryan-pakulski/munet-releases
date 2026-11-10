# ONNX import and export

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
| `device` | `"vulkan"` | `"vulkan:N"` for GPU N, `"cpu"` for the reference executor |
| `precision` | `"mixed"` | see [inference.md](inference.md#precision) |
| `input_shapes` | `None` (shapes from the file) | `{input name: shape}` for symbolic dimensions |
| `max_specializations` | `8` | programs kept when shapes stay symbolic (see [Input shapes](#input-shapes)) |
| `int8` | `None` | activation scales from `calibrate_onnx`: int8 convolutions ([quantization.md](quantization.md)) |
| `autotune` | `None` (follows `MUNET_AUTOTUNE`) | time kernel tile sizes once per GPU and cache them ([inference.md](inference.md#autotuning)) |
| `optimize` | `True` | merge layers before import (see below) |
| `profile` | `False` | per-kernel GPU times in `model.describe()` |
| `native` | `False` | import with the engine's C++ importer (the one C and Rust use); see [native import](#native-import-load-onnx-without-python) |

`optimize=True` applies these rewrites. `model.optimizations` lists what it did.

- Fold BatchNorm into the neighbouring convolution.
- Re-parameterize parallel convolution branches into one (RepVGG, MobileOne).
- Merge linear chains with no nonlinearity between them (MatMul/Gemm pairs, a Conv followed by a 1x1 Conv).
- Fuse attention patterns into one flash-attention kernel.

Sibling-layer merging (layers that read the same input become one wider layer) is off. Set `MUNET_MERGE_SIBLINGS=1`
to turn it on.

The first call compiles: it generates and compiles kernels (cached on disk), plans memory, and uploads weights.
Later calls only run.

## Input shapes

Programs have static shapes.

- **With `input_shapes`** (recommended), you get one program for exactly those shapes.
- **Without it,** a model with symbolic dimensions returns a `ShapeSpecializedProgram`. It compiles a new program the
  first time it sees each new input shape and keeps the `max_specializations` most recent. Every new shape pays the
  compile cost.

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

- It exports with `torch.onnx.export` (dynamo, opset 18) and imports the result.
- It needs `pip install 'munet-nn[torch]'` (torch + onnxscript).
- Shapes are those of `example_input`.
- For older exporters, export the file yourself (`torch.onnx.export(model, x, "m.onnx", opset_version=17)`) and use
  `from_onnx`.

## Export to ONNX

Any compiled inference program exports to a standard ONNX graph, whether it was built natively, imported from
ONNX, or loaded from `.mnet`:

```python
from munet.interop import to_onnx

program = mu.compile(model.eval())
program(example)                       # compile once first
to_onnx(program, "model.onnx")
```

- **Exported:** the computation, weights included. Quantized programs export as QDQ (`QuantizeLinear` /
  `DequantizeLinear`).
- **Not exported:** the original layer structure, because the graph is MuNet's optimized form.
- **Training steps** can't be exported. Compile the trained model for inference and export that.

## Save as .mnet for deployment

```python
import munet
munet.save(model, "model.mnet")                       # graph, weights, input/output names and dtypes
again = munet.load("model.mnet", precision="fp16")    # defaults: device="vulkan", precision="mixed"
```

- `.mnet` stores the computation and weights, plus int8 scales if the model is quantized.
- Precision is chosen when loading.
- C, C++ and Rust load the same file (`munet_load`, `munet::Program::load`), or import the ONNX file directly (next
  section).

## Native import: load ONNX without Python

The engine also imports ONNX itself, in C++. A service can load `model.onnx` at startup, with no export step and no
`.mnet` to keep in sync with the MuNet version.

| Language | Call |
|---|---|
| C, C++ | `munet_load_onnx` (`munet_load_onnx_on` for a shared Vulkan context) |
| Rust | `munet::Program::load_onnx` |
| Python | `from_onnx(..., native=True)` |

```python
model = from_onnx("detector.onnx", input_shapes={"images": (3, 3, 1280, 1280), "orig_target_sizes": (3, 2)}, native=True)
```

It builds the same graph as the Python importer (verified node by node on the bench models).

### Options

- **Shapes.** Inputs with symbolic dimensions need a static shape (`input_shapes` / `munet_onnx_input_shape` /
  `OnnxOptions::input_shape`). Inputs you don't list keep the model's declared shape.
- **Optimize.** On by default (`optimize` / `munet_onnx_options.optimize`), with the same rewrites as Python.
- **Precision and device** are chosen at load, as for `.mnet` programs.
- **int8.** Pass activation scales (`int8=` / `munet_onnx_options.int8_scales` / `OnnxOptions::int8`). Calibrate
  natively with `munet_calibrate_onnx`, `munet::calibrate_onnx` or `calibrate_onnx(..., native=True)`.
- **Save.** `munet_save` / `Program::save` write the same `.mnet` format as `munet.save`.

### Cache

With `munet_onnx_options.cache_path` / `OnnxOptions::cache`, the import is kept in a `.mnet` file:

- The key is the SHA-256 of the ONNX file's bytes, the import options (input shapes, `optimize`, int8 scales,
  `MUNET_MERGE_SIBLINGS`), the MuNet version and the format version. It is stored in the file's manifest
  (`"munet_onnx_cache": {"key": ...}`).
- Precision, device and autotuning are not part of the key. They apply at load.
- A matching file is loaded without importing.
- A missing, different or unreadable file is replaced atomically after the import.
- A file that cannot be written is skipped; the imported program is still returned.
- Point `cache_path` at a file of its own: any `.mnet` there without a matching key is overwritten.

### Coverage and differences from Python

- **Operators.** Everything the Python importer imports: every operator family it supports, including random
  operators (same seeds and per-call counter), signal processing, recurrent layers and constant folds. QDQ / QLinear
  graphs, sequences and optionals import too.
- **Control flow.** `If` with a constant condition picks a branch. With a runtime condition both branches run and a
  select picks per call. `Loop` / `Scan` / `SequenceMap` unroll when their trip counts and conditions are known at
  import.
- **Limits.** Unrolling is capped at 65536 iterations per loop and 4 million graph nodes per import.
- **Rejections.** The same as Python: unknown operators, string tensors that reach the graph, external tensor data,
  model-local functions. Errors name the node and operator.
- **Folded constants.** Floating-point folds of constants (e.g. `Log` or `MatMul` of two constants) can differ from
  Python's in the last bit.
- **Legacy opsets.** Opset 7-12 operators are upgraded per operator.
- **NonMaxSuppression.** Only as a model output with constant limits and thresholds. The program returns a static
  `[batch x class x max_output_boxes_per_class, 3]` int64 tensor whose unused rows are -1. The output is marked padded
  (`munet_output_padded`, `TensorInfo::padded`, `"padded"` in the `.mnet` io metadata).
  `munet_read_output_rows` / `Program::read_rows` / `read_i64` return just the selected rows, as Python's program does.
- **Untrusted files.** The protobuf decoder is bounds-checked and fuzzed. Malformed files fail with an error, and
  shapes a model derives are capped at 2^31 elements per value.

## What is supported

- **Opsets.** ONNX opsets 13-21. Files with opsets 7-12 (YOLOv5-v8, PP-OCR and other older exports) are upgraded
  automatically.
- **Operators.** All 191 default-domain operators; [op_coverage.md](https://github.com/bryan-pakulski/munet-releases/blob/main/docs/op_coverage.md) is the table.
  `munet.interop.SUPPORTED_ONNX` lists them in code.
- **Quantized models.** QDQ (`QuantizeLinear` / `DequantizeLinear`) and QLinear graphs run on int8 kernels.
- **Resolved at import.** Sequences, optionals, control flow on constants, and recurrent layers (unrolled).
- **Fused automatically.** Attention patterns (`MatMul → [scale] → [+ bias/mask] → Softmax → MatMul`) become one
  flash-attention kernel.
- **Tested end to end.** [examples/bench](https://github.com/bryan-pakulski/munet-releases/blob/main/examples/bench/README.md) runs about 40 public models (YOLO v5-v11,
  RT-DETR, D-FINE, DINOv2, CLIP, SAM, Depth-Anything, PP-OCR, ConvNeXt, ...) and compares them against ONNX Runtime.

## When import fails

| Error | Cause | Fix |
|---|---|---|
| `UnsupportedOperatorError: ... data-dependent output size` | an output's size depends on input values (e.g. `NonZero` feeding later ops) | move that part out of the model, or use fixed-size ops (`nn.functional.nms` with `max_output`) |
| `... external tensor data` | weights stored in separate files | `onnx.load(path)` then `onnx.save(model, "single.onnx", save_as_external_data=False)` (models under 2 GB) |
| `declared output shape differs` | the file declares an output shape that doesn't match the computation | re-export, or fix the declared shape with `onnx.shape_inference` |
| NonMaxSuppression consumed inside the graph | dynamic result size | NMS must be a model output |
| values far off in fp16 | the model has activations beyond fp16 range | `precision="mixed"` or `"fp32"`; compare with `infer_onnx.py --compare` |

To check a model's accuracy, run [examples/python/infer_onnx.py](https://github.com/bryan-pakulski/munet-releases/blob/main/examples/python/infer_onnx.py) with
`--compare`, which compares against ONNX Runtime. To find the first layer that differs, use
[examples/bench/find_mismatch.py](https://github.com/bryan-pakulski/munet-releases/blob/main/examples/bench/find_mismatch.py).
