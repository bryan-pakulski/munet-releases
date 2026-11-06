# ONNX import and export

- [Import an ONNX model](#import-an-onnx-model)
- [Input shapes](#input-shapes)
- [Inputs and outputs](#inputs-and-outputs)
- [Import a PyTorch model](#import-a-pytorch-model)
- [Export to ONNX](#export-to-onnx)
- [Save as .mnet for deployment](#save-as-mnet-for-deployment)
- [Native import: load ONNX without Python](#native-import-load-onnx-without-python)
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
| `native` | False | import with the engine's C++ importer (the one C and Rust use); see [native import](#native-import-load-onnx-without-python) |

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

C, C++ and Rust load the same file (`munet_load`, `munet::Program::load`), or import the ONNX file directly (see
[native import](#native-import-load-onnx-without-python)). `.mnet` stores the computation and
weights, plus int8 scales if the model is quantized. Precision is chosen when loading.

## Native import: load ONNX without Python

The engine also imports ONNX itself, in C++: C and C++ call `munet_load_onnx`, Rust calls `munet::Program::load_onnx`,
and Python reaches the same importer with `from_onnx(..., native=True)`. It builds the same graph as the Python
importer (verified node by node on the examples/bench models), so a service can load `model.onnx` at startup the way
TensorRT builds an engine, with no export step and no `.mnet` to keep in sync with the MuNet version.

```python
model = from_onnx("detector.onnx", input_shapes={"images": (3, 3, 1280, 1280), "orig_target_sizes": (3, 2)}, native=True)
```

- **Shapes.** Inputs with symbolic dimensions need a static shape (`input_shapes` / `munet_onnx_input_shape` /
  `OnnxOptions::input_shape`); inputs you don't list keep the model's declared shape.
- **Same options.** Layer merging and attention fusion are on by default (`optimize` / `munet_onnx_options.optimize`);
  precision and device are chosen as for `.mnet` programs.
- **Coverage.** Everything the Python importer imports, except the parts below: QDQ / QLinear graphs, int8 layers
  (`int8=` scales, and calibration: `munet_calibrate_onnx`, `munet::calibrate_onnx`,
  `calibrate_onnx(..., native=True)`), `NonMaxSuppression` as a model output (see below), sequences, optionals and
  control flow (`If` with a constant condition picks a branch, with a runtime condition both branches run and a
  select picks per call; `Loop` / `Scan` / `SequenceMap` unroll when their trip counts and conditions are known at
  import, with the same rejections as Python). Unrolling is capped at 65536 iterations per loop and 4 million graph
  nodes per import.
- **Every operator family.** Random operators (same seeds and per-call state counter as Python, so the same samples on
  every call), `Einsum`, `RoiAlign`, losses, signal processing (`DFT`, `STFT`, window functions, `MelWeightMatrix`),
  scatter / gather-ND, recurrent layers (`RNN` / `GRU` / `LSTM`, unrolled), bitwise operators and the other
  compositions import natively, as do the constant folds the Python importer runs through onnx's ReferenceEvaluator
  (`EyeLike`, `OneHot`, `NonZero`, `Compress`, `Unique`, ...). What the native importer rejects, Python rejects too
  (unknown operators, string tensors that reach the graph, external tensor data, model-local functions); errors name
  the node and operator.
- **Folded constants.** Shape arithmetic folds at import as in Python; floating-point folds that numpy evaluates with
  its own math library or BLAS (e.g. `Log` or `MatMul` of two constants) can differ from Python's in the last bit.
- **Legacy opsets.** Opset 7-12 operators are upgraded per operator. A Softmax over the last axis stays one softmax
  (Python's version converter may add a flatten/reshape pair around it when it cannot infer the rank; the result is the
  same).
- **NonMaxSuppression.** As in Python, only as a model output with constant limits and thresholds: the program
  returns a static `[batch x class x max_output_boxes_per_class, 3]` int64 tensor whose unused rows are -1. The output
  is marked padded (`munet_output_padded`, `TensorInfo::padded`, `"padded"` in the `.mnet` io metadata);
  `munet_read_output_rows` / `Program::read_rows` / `read_i64` return just the selected rows, as Python's program does.
- **Save and cache.** `munet_save` / `Program::save` write the `.mnet` format `munet.save` writes (same manifest and
  tensor bytes), readable by `munet_load` and `munet.load`. With `munet_onnx_options.cache_path` /
  `OnnxOptions::cache`, `munet_load_onnx` keeps the import in a `.mnet` file: the key is the SHA-256 of the ONNX
  file's bytes, the import options (input shapes, `optimize`, int8 scales, `MUNET_MERGE_SIBLINGS`), the MuNet version
  and the format version, stored in the archive's manifest (`"munet_onnx_cache": {"key": ...}`). A matching file is
  loaded without importing; a missing, different or unreadable one is replaced atomically (written to a temporary file
  next to it, then renamed) after the import. Precision, device and autotuning are not part of the key (they apply at
  load). A cache file that cannot be written is skipped (the imported program is still returned). Point `cache_path`
  at a file of its own: any `.mnet` without a matching key there is overwritten.
- **Untrusted files.** The protobuf decoder is bounds-checked and fuzzed; malformed files fail with an error, and
  shapes a model derives are capped (2^31 elements per value).

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
