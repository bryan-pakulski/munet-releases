# MuNet from C and C++

The C API (`munet/munet_c.h`, API version 5, `MUNET_C_API_VERSION`) loads ONNX files or `.mnet` programs and runs
them on any Vulkan GPU, or on the CPU reference executor. It is a plain C ABI, so C++, Rust (`munet-nn`), Go, C#, Zig
and other languages use the same library.

## Get the library

Each release publishes `munet-c-<version>-<os>-<arch>.tar.gz` archives (Linux x86_64 and aarch64) on
[the release downloads](https://github.com/bryan-pakulski/munet-releases/releases), with SHA-256 sums:

```
include/munet/munet_c.h
lib/libmunet_c.so            # shared library
lib/static/libmunet_c.a      # static: link munet_c and munet_core, plus the C++ runtime, libdl and pthread
lib/static/libmunet_core.a
examples/run_model.cpp       # load and run a program
examples/vulkan_interop.cpp  # share MuNet's Vulkan device
docs/                        # the user guides this README links to
LICENSE, README.md
```

Link it shared or static:

```bash
# shared
c++ -std=c++17 app.cpp -I$SDK/include -L$SDK/lib -lmunet_c -Wl,-rpath,$SDK/lib -o app
# static (no runtime dependency on libmunet_c.so)
c++ -std=c++17 app.cpp -I$SDK/include $SDK/lib/static/libmunet_c.a $SDK/lib/static/libmunet_core.a -ldl -lpthread -o app
```

From CMake:

```cmake
add_library(munet_c STATIC IMPORTED)
set_target_properties(munet_c PROPERTIES
  IMPORTED_LOCATION ${MUNET_SDK}/lib/static/libmunet_c.a
  INTERFACE_INCLUDE_DIRECTORIES ${MUNET_SDK}/include
  INTERFACE_LINK_LIBRARIES "${MUNET_SDK}/lib/static/libmunet_core.a;${CMAKE_DL_LIBS};Threads::Threads")
find_package(Threads REQUIRED)
target_link_libraries(app PRIVATE munet_c)
```

To build from source in the engine checkout, configure with `cmake -DMUNET_C_API=ON` (add
`-DMUNET_C_API_SHARED=ON` for `libmunet_c.so`).

At run time you need:

- a Vulkan 1.1+ loader and driver (any vendor: AMD, NVIDIA, Intel, Arm Mali, Qualcomm Adreno, Apple through MoltenVK);
- `glslangValidator` on `PATH` (`glslang-tools` on Debian/Ubuntu), or its path in `MUNET_GLSLANG`.

Kernels are compiled when a program loads and cached in `~/.cache/munet-next` (`MUNET_CACHE_DIR`), so later loads
skip compilation. All environment variables: [environment.md](environment.md).

Without a license key the library runs in evaluation mode and prints a notice. `munet_license_json()` reports the
status ([key setup](getting-started.md#license-key)):

```c
char json[512];
munet_license_json(json, sizeof json);   /* {"status":"licensed","customer":...,"expires":"2027-12-31",...} */
```

## Models

Load an ONNX file directly with the engine's native importer (no Python), or a `.mnet` program. Shapes are static:
give symbolic input dimensions a shape, and load one program per batch size and resolution you serve.

```c
int64_t images[] = {1, 3, 640, 640}, sizes[] = {1, 2};
munet_onnx_input_shape shapes[] = {{"images", images, 4}, {"orig_target_sizes", sizes, 2}};
munet_onnx_options onnx = munet_default_onnx_options();     /* optimize on (layer merging, attention fusion) */
onnx.input_shapes = shapes; onnx.input_shape_count = 2;
munet_program* program = munet_load_onnx("detector.onnx", &onnx, &options);   /* munet_load_onnx_on: on a context */
```

The native importer builds the same graph as Python's `from_onnx`, including QDQ / int8 layers, NonMaxSuppression and
control flow. Details: [onnx.md](onnx.md#native-import-load-onnx-without-python).

`munet_onnx_options` also takes int8 activation scales and a cache file:

```c
munet_int8_scale int8[] = {{"conv1", 0.021f}, {"conv2", 0.013f}};    /* e.g. from munet_calibrate_onnx below */
onnx.int8_scales = int8; onnx.int8_scale_count = 2;
onnx.cache_path = "/var/cache/app/detector.mnet";   /* reused while the ONNX file and these options are unchanged */
```

- The cache file is a `.mnet` archive keyed by the ONNX file's SHA-256, the import options, the MuNet version and the
  format version.
- `munet_save(program, path)` writes any program as `.mnet` (what `munet.save` writes).

int8 calibration runs the model's int8-eligible layers in fp32 over your batches and returns per-layer scales. It uses
the same algorithm and observers as Python's `munet.quantization.calibrate_onnx`:

```c
const float* batch0[] = {images0, sizes0}, *batch1[] = {images1, sizes1};   /* one pointer per model input */
const float* const* batches[] = {batch0, batch1};
munet_calibration* cal = munet_calibrate_onnx("detector.onnx", &onnx, &options, batches, 2, "mse", 2048);
for (size_t i = 0; cal && i < munet_calibration_count(cal); ++i)
  printf("%s %g\n", munet_calibration_name(cal, i), munet_calibration_scale(cal, i));
munet_calibration_free(cal);
```

Random operators keep a step counter that advances every `munet_run`. `munet_save` / `munet_load` preserve it.

Models with external tensor data or model-local functions need the Python importer. Export those from Python
(`pip install munet-nn`):

```python
import munet
from munet.interop import from_onnx
program = from_onnx("detector.onnx", input_shapes={"images": (1, 3, 640, 640), "orig_target_sizes": (1, 2)})
munet.save(program, "detector.mnet")   # records input/output names and declared dtypes
```

## Load and run

The complete program is `examples/run_model.cpp` in the archive. Its core:

```cpp
#include "munet/munet_c.h"

munet_options options = munet_default_options();   // device 0, MIXED, autotune cache on, no profiling
options.precision = MUNET_PRECISION_FP16;
munet_program* program = munet_load("detector.mnet", &options);
if (!program) fprintf(stderr, "%s\n", munet_last_error());

// One contiguous float32 tensor per input, in declaration order.
std::vector<float> image(3 * 640 * 640), size = {640, 640};
const float* inputs[] = {image.data(), size.data()};
if (munet_run(program, inputs) != 0) fprintf(stderr, "%s\n", munet_last_error());

std::vector<float> scores(300);
munet_read_output(program, /* index */ 2, scores.data(), scores.size());   // count must equal the output's size
munet_free(program);
```

A C++ RAII wrapper:

```cpp
struct ProgramDeleter { void operator()(munet_program* p) const { munet_free(p); } };
using Program = std::unique_ptr<munet_program, ProgramDeleter>;
Program program(munet_load(path, &options));
```

How runs behave:

- Load is the expensive step. It compiles kernels (cached after the first time), uploads the weights and records
  a command buffer that every run replays. Load once and keep the program.
- Data stays on the GPU. `munet_run` copies the inputs into a staging buffer in the same submission as the kernels.
  Weights and intermediate tensors never leave device memory, and only the outputs you ask for are read back.
- `munet_run` returns once the work is submitted. `munet_read_output` waits for it to finish, then copies the
  output.
- Read every output you need before the next run, because the next run overwrites them.
- `options.device = -1` selects the CPU reference executor. It is slow and meant for tests. It is never used as a
  silent fallback.

## Inputs, outputs and dtypes

```c
size_t n = munet_input_count(program);
const char* name  = munet_input_name(program, i);    // "images"
const char* dtype = munet_input_dtype(program, i);   // the model's declared type: "float32", "int64", "bool", ...
int32_t rank = munet_input_shape(program, i, dims, 8);
```

The output functions mirror these.

MuNet computes every tensor as float32, so all data crosses the API as `float`:

- Integer and bool inputs are passed as exact integer-valued floats (exact up to 2^24).
- Integer outputs come back the same way; round them to recover the integers.
- `munet_output_padded(program, i)` marks `NonMaxSuppression` outputs: `[bound, 3]` rows, where the unused rows are
  `-1`. `munet_read_output_rows` returns only the selected `[batch, class, box]` rows:

```c
std::vector<float> rows(bound * 3); size_t n = 0;
munet_read_output_rows(program, index, rows.data(), rows.size(), &n);   // n selected rows, rows[3*r .. 3*r+2]
```

## Precision, int8 and autotuning

| `precision` | Python name | Matmul/conv arithmetic |
|---|---|---|
| `MUNET_PRECISION_MIXED` (default) | `"mixed"` | fp16 operands, fp32 accumulation; int8 on quantized layers |
| `MUNET_PRECISION_FP16` | `"fp16"` | fp16 operands and accumulation; fastest on GPUs with fp16-accumulate cooperative matrices; validate accuracy per model |
| `MUNET_PRECISION_FP32` | `"fp32"` | exact float32 (reference) |

- GPUs without cooperative matrices (most mobile GPUs) run fp32 kernels in every mode.
- How to choose and measure: [inference.md](inference.md#precision). GPU-specific notes: [gpus.md](gpus.md).
- int8 is part of the program, not an option. Import with `int8_scales` (from `munet_calibrate_onnx` or Python's
  `calibrate_onnx`), or load an int8 `.mnet`. Its quantized convolutions then run on int8 kernels.
  See [quantization.md](quantization.md).
- Autotuning: run the program once from Python with `autotune=True` and the same precision on the target GPU
  ([inference.md](inference.md#autotuning)). C loads apply the cached results when `use_autotune_cache` is set (the
  default). Untuned shapes use heuristics.

## Zero copy with Vulkan

MuNet uses one `VkDevice` per GPU, and your Vulkan code can share it: a video decoder, a renderer or your own
preprocessing. Tensors then never leave device memory. Walkthrough for video (host frames, FFmpeg Vulkan decode, your
own buffers): [zero-copy.md](zero-copy.md). The complete, validation-clean example is `examples/vulkan_interop.cpp`
in the archive.

```c
munet_vulkan* ctx = munet_vulkan_create(0, MUNET_VULKAN_VIDEO_DECODE);   // flags 0: the default shared device
munet_vulkan_handles h;
munet_vulkan_get_handles(ctx, &h);
/* h.instance, h.physical_device, h.device, h.get_instance_proc_addr: adopt the device (e.g. FFmpeg's
   AVVulkanDeviceContext). h.extensions / h.features: what is enabled. h.compute_family / h.compute_queue: MuNet's
   queue. h.decode_family / h.decode_queues: the video decode queue (UINT32_MAX / 0 when not requested). */

munet_program* program = munet_load_on(ctx, "detector.mnet", &options);
VkBuffer buffer = (VkBuffer)munet_program_vulkan_buffer(program);
int64_t images = munet_input_offset(program, 0);    // byte offset of input 0 (contiguous float32) in `buffer`
int64_t scores = munet_output_offset(program, 2);
```

Write the inputs into `buffer` at their offsets with your own commands, then run:

```c
munet_semaphore_sync sync = {
  .wait_semaphore = timeline, .wait_value = 1,     /* your writes signal 1 */
  .signal_semaphore = timeline, .signal_value = 2  /* outputs complete */
};
munet_run_device(program, &sync);   /* GPU-ordered; returns without waiting */
/* ... your Vulkan work waits on timeline >= 2 and reads `buffer` at the output offsets ... */
munet_wait(program);                /* only if the host needs to block */
```

Rules:

- Timeline semaphores: `wait_semaphore` and `signal_semaphore` must be timeline semaphores on the shared device.
  With `sync == NULL`, your writes must already be complete when you call `munet_run_device`.
- Queue locking: Vulkan queues are externally synchronized. Hold `munet_vulkan_lock_queue(ctx, family, index)`
  around every `vkQueueSubmit` or `vkQueueWaitIdle` you make on MuNet's queues, and unlock afterwards with
  `munet_vulkan_unlock_queue`. Give FFmpeg these functions as `AVVulkanDeviceContext.lock_queue` / `unlock_queue`.
- Mixing host and device runs: every program accepts both `munet_run` (host arrays) and `munet_run_device`
  (device inputs), and can switch between them on any call.
- Lifetime: `munet_vulkan_free` releases your reference. Programs keep the device alive until they are freed.
- Device loss: see [Threads, errors and device loss](#threads-errors-and-device-loss).

## Frame pipeline (video detection)

For DETR-style detectors (inputs `images` [B,3,H,W] and `orig_target_sizes` [B,2]; outputs `labels` [B,Q], `boxes`
[B,Q,4] as xyxy in input pixels, and `scores` [B,Q]), the frame pipeline runs NV12 colour conversion, the model,
box mapping and NMS on the GPU. Only the detections come back. Tiles, colour and decoder sync are explained in
[zero-copy.md](zero-copy.md).

```c
munet_frame_pipeline* pipe = munet_frame_pipeline_create(program, /* max frame */ 3840, 2160, /* max tiles */ 32);

munet_nv12_frame frame = {
  .image = decoded_vkimage,                  /* VK_FORMAT_G8_B8R8_2PLANE_420_UNORM, TRANSFER_SRC usage, same device */
  .layout = current_layout,                  /* left in MUNET_FRAME_LAYOUT_AFTER (TRANSFER_SRC_OPTIMAL) */
  .width = 1920, .height = 1080};
/* or host planes: .image = NULL, .y = y, .y_pitch = ..., .uv = uv, .uv_pitch = ... (only the NV12 bytes are uploaded) */

munet_tile tiles[3] = {...};                 /* frame = input_px * scale + offset; see zero-copy.md */
munet_detection found[300];
size_t count = 0;
munet_frame_detect(pipe, &frame, tiles, 3, /* color: NULL = BT.709 limited, no normalisation */ NULL,
                   /* score */ 0.15f, /* nms IoU */ 0.25f, &sync, found, 300, &count);
/* found[i]: x, y, width, height (frame pixels), score, label, input (the tile's input_index) */
munet_frame_pipeline_free(pipe);             /* before freeing the program */
```

- `munet_frame_detect` returns once the detections are on the host.
- `munet_frame_color` holds the YCbCr → RGB coefficients plus `mean` and `inv_std`, applied as
  `(RGB/255 - mean) * inv_std`.
- `sync` carries the decoder's timeline semaphore: the pipeline waits on `wait_semaphore` and signals
  `signal_semaphore`. Pass `NULL` for host frames.
- With more tiles than the program's batch size, the pipeline runs the program as many times as needed.

## Threads, errors and device loss

- Threads: a program must not be used from two threads at once. Use one program per thread; several programs on
  one context are fine.
- Errors: functions returning a pointer return `NULL` on failure, and functions returning `int32_t` return
  nonzero. `munet_last_error()` describes the most recent failure on the calling thread.
- Device loss: a lost device (GPU reset, driver crash) fails every later run on it and is never reused. Recover by
  reloading the program. With a shared context, create a new context first and reload the programs on it.

## Diagnostics

- `munet_gpu_ms(program)`: GPU time of the last run's compute section. It excludes transfers and is negative if
  unknown.
- `munet_stats_json(program, buf, size)`: the device name, load / shader compile / device setup times, the number of
  autotuned kernels, kernel and memory counters, run counts and the last `gpu_ms`, as JSON.
- `options.profile = 1` records a GPU timestamp per kernel. It serializes kernels, so use it to compare kernels,
  not to measure throughput.
- `MUNET_PIPELINE_STATS=1` prints the driver's statistics for each kernel (registers, spills, ...) to stderr when the
  program loads, on drivers that expose `VK_KHR_pipeline_executable_properties`.
- Vulkan validation layers (`VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`) are supported and expected to be clean.

## C++ engine SDK

Releases also publish `munet-sdk-<version>-linux-<arch>.tar.gz` (x86_64, aarch64). It holds the engine's C++ core,
not the C API above:

- `lib/libmunet_core.a`, the static engine library;
- `include/munet/core.hpp`, the C++ graph API (`munet::Graph`, `munet::Plan`);
- `lib/cmake/MuNet/`, a CMake package exporting `MuNet::core`.

Point CMake at the extracted directory with `-DCMAKE_PREFIX_PATH=/path/to/munet-sdk-<version>-linux-<arch>`:

```cmake
find_package(MuNet CONFIG REQUIRED)
target_link_libraries(app PRIVATE MuNet::core)   # then #include <munet/core.hpp>
```

It is a C++ ABI package: you need C++17 and a toolchain compatible with the release build. If yours differs, use the
C API or build from source.
