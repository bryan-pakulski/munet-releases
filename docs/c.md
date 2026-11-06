# MuNet from C and C++

The C API (`munet/munet_c.h`, API version 3) loads `.mnet` programs and runs them on any Vulkan GPU, or on the CPU
reference executor. It is a plain C ABI, so C++, Rust (`munet-nn`), Go, C#, Zig and other languages use the same
library. This guide is the developer reference: building and linking, the run loop, precision and tuning, zero copy
with your own Vulkan code, and the frame pipeline that takes video frames in and returns detections.

- [Get the library](#get-the-library)
- [Models](#models)
- [Load and run](#load-and-run)
- [Inputs, outputs and dtypes](#inputs-outputs-and-dtypes)
- [Precision, int8 and autotuning](#precision-int8-and-autotuning)
- [Zero copy with Vulkan](#zero-copy-with-vulkan)
- [Frame pipeline (video detection)](#frame-pipeline-video-detection)
- [Threads, errors and device loss](#threads-errors-and-device-loss)
- [Diagnostics](#diagnostics)
- [Environment variables](#environment-variables)

## Get the library

Each release publishes `munet-c-<version>-<os>-<arch>.tar.gz` archives (Linux x86_64 and aarch64) on
[the release downloads](https://github.com/bryan-pakulski/munet-releases/releases), with SHA-256 sums:

```
include/munet/munet_c.h
lib/libmunet_c.so            # shared library
lib/static/libmunet_c.a      # static: link munet_c and munet_core, plus the C++ runtime, libdl and pthread
lib/static/libmunet_core.a
LICENSE, README.md
```

Link it either way:

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

To build from source inside the engine checkout instead, configure with `cmake -DMUNET_C_API=ON` (add
`-DMUNET_C_API_SHARED=ON` for `libmunet_c.so`).

**At run time** you need:
- a Vulkan 1.1+ loader and driver (any vendor: AMD, NVIDIA, Intel, Arm Mali, Qualcomm Adreno, Apple through MoltenVK);
- `glslangValidator` on `PATH` (`glslang-tools` on Debian/Ubuntu), or its path in `MUNET_GLSLANG`. Kernels are
  compiled when a program loads and cached in `~/.cache/munet-next` (`MUNET_CACHE_DIR`), so later loads skip it.

**License.** The native library runs in evaluation mode without a key (it prints a notice). Production keys go in
`MUNET_LICENSE`, `MUNET_LICENSE_FILE` or `~/.config/munet/license.key`. `munet_license_json()` reports the status:

```c
char json[512];
munet_license_json(json, sizeof json);   /* {"status":"licensed","customer":...,"expires":"2027-12-31",...} */
```

## Models

Load an ONNX file directly (C API 5; the engine's native importer, no Python), or a `.mnet` program exported from
Python. Shapes are static: give symbolic input dimensions a shape, one program per batch size and resolution you serve.

```c
int64_t images[] = {1, 3, 640, 640}, sizes[] = {1, 2};
munet_onnx_input_shape shapes[] = {{"images", images, 4}, {"orig_target_sizes", sizes, 2}};
munet_onnx_options onnx = munet_default_onnx_options();     /* optimize on (layer merging, attention fusion) */
onnx.input_shapes = shapes; onnx.input_shape_count = 2;
munet_program* program = munet_load_onnx("detector.onnx", &onnx, &options);   /* munet_load_onnx_on: on a context */
```

The native importer builds the same graph as Python's `from_onnx`, including QDQ / int8 layers, NonMaxSuppression and
control flow. `munet_onnx_options` also takes int8 activation scales and a cache file:

```c
munet_int8_scale int8[] = {{"conv1", 0.021f}, {"conv2", 0.013f}};    /* e.g. from munet_calibrate_onnx below */
onnx.int8_scales = int8; onnx.int8_scale_count = 2;
onnx.cache_path = "/var/cache/app/detector.mnet";   /* reused while the ONNX file and these options are unchanged */
```

The cache file is a `.mnet` archive keyed by the ONNX file's SHA-256, the import options, the MuNet version and the
format version ([onnx.md](onnx.md#native-import-load-onnx-without-python) has the rules). `munet_save(program, path)`
writes any program as `.mnet` (what `munet.save` writes).

int8 calibration runs the model's int8-eligible layers in fp32 over your batches and returns per-layer scales
(`munet.quantization.calibrate_onnx`'s algorithm and observers):

```c
const float* batch0[] = {images0, sizes0}, *batch1[] = {images1, sizes1};   /* one pointer per model input */
const float* const* batches[] = {batch0, batch1};
munet_calibration* cal = munet_calibrate_onnx("detector.onnx", &onnx, &options, batches, 2, "mse", 2048);
for (size_t i = 0; cal && i < munet_calibration_count(cal); ++i)
  printf("%s %g\n", munet_calibration_name(cal, i), munet_calibration_scale(cal, i));
munet_calibration_free(cal);
```

The native importer covers the same operators as Python, random operators included: their programs keep a step
counter that advances every `munet_run`, and `munet_save` / `munet_load` keep it. Models with external tensor data or
model-local functions still need the Python importer; export those from Python (`pip install munet-nn`):

```python
import munet
from munet.interop import from_onnx
program = from_onnx("detector.onnx", input_shapes={"images": (1, 3, 640, 640), "orig_target_sizes": (1, 2)})
munet.save(program, "detector.mnet")   # records input/output names and declared dtypes
```

## Load and run

The complete program is `examples/run_model.cpp` in the archive (`examples/c/run_model.cpp` in the source tree). The
core of it:

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

A C++ RAII wrapper is three lines:

```cpp
struct ProgramDeleter { void operator()(munet_program* p) const { munet_free(p); } };
using Program = std::unique_ptr<munet_program, ProgramDeleter>;
Program program(munet_load(path, &options));
```

How runs behave:

- **Load is the expensive step.** It compiles kernels (cached after the first time), uploads the weights and records
  a command buffer that every run replays. Load once and keep the program.
- **Data stays on the GPU.** `munet_run` copies the inputs into a staging buffer and records that copy into the same
  submission as the kernels. Weights and intermediate tensors never leave device memory, and only the outputs you
  ask for are read back.
- `munet_run` returns once the work is submitted. `munet_read_output` waits for it to finish, then copies the
  output. Read every output you need before the next run, because the next run overwrites them.
- **CPU executor.** Setting `options.device = -1` selects the CPU reference executor. It is slow and meant for tests
  and correctness checks, and it is never used as a silent fallback.

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
  `-1`. `munet_read_output_rows` returns only the selected `[batch, class, box]` rows (what Python's program returns):

```c
std::vector<float> rows(bound * 3); size_t n = 0;
munet_read_output_rows(program, index, rows.data(), rows.size(), &n);   // n selected rows, rows[3*r .. 3*r+2]
```

## Precision, int8 and autotuning

> How to choose and measure: [inference.md](inference.md); int8 export: [quantization.md](quantization.md); GPU-specific notes: [gpus.md](gpus.md).

| `precision` | Matmul/conv arithmetic | Use |
|---|---|---|
| `MUNET_PRECISION_MIXED` (default) | fp16 operands, fp32 accumulation; int8 on quantized layers | general inference |
| `MUNET_PRECISION_FP16` | fp16 operands and accumulation | fastest on GPUs with fp16-accumulate cooperative matrices (GeForce Ampere/Ada: 2x the MIXED rate); validate accuracy per model |
| `MUNET_PRECISION_FP32` | exact float32 | reference |

- **GPUs without cooperative matrices** (most mobile GPUs) run fp32 kernels in every mode.
- **int8** is a property of the program, not an option: export it with int8 activation scales from Python (see
  the Python guide), and its convolutions then run on int8 cooperative-matrix kernels.
- **Autotuning** picks tile configurations per GEMM/conv shape. Run the same program once from Python with
  `autotune=True` and the same precision on the target GPU. The winners are cached in `MUNET_CACHE_DIR`, and C/Rust
  loads apply them when `use_autotune_cache` is set (the default). Untuned shapes use heuristics.

## Zero copy with Vulkan

> Walkthrough for video (host frames, FFmpeg Vulkan decode, your own buffers): [zero-copy.md](zero-copy.md).

MuNet uses one `VkDevice` per GPU, and your Vulkan code can share it, for example a video decoder, a renderer or your
own preprocessing. Tensors then never leave device memory, and nothing outside Vulkan is involved, so this works the
same on AMD, NVIDIA, Intel and mobile GPUs. The complete, validation-clean example is
`examples/vulkan_interop.cpp` in the archive (`examples/c/vulkan_interop.cpp` in the source tree).

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

- **Timeline semaphores.** `wait_semaphore` and `signal_semaphore` must be timeline semaphores on the shared device.
  With `sync == NULL`, your writes must already be complete when you call `munet_run_device`.
- **Queue locking.** Vulkan queues are externally synchronized, so hold `munet_vulkan_lock_queue(ctx, family, index)`
  around every `vkQueueSubmit` or `vkQueueWaitIdle` you make on MuNet's queues, and unlock afterwards. Give FFmpeg
  these lock functions as `AVVulkanDeviceContext.lock_queue` / `unlock_queue`.
- **Mixing host and device runs.** Every program accepts both `munet_run` (host arrays) and `munet_run_device`
  (device inputs), and can switch between them on any call.
- **Lifetime.** `munet_vulkan_free` releases your reference; programs keep the device alive until they are freed.
- **Device loss.** A lost device is never reused: create a new context and reload the programs on it.

## Frame pipeline (video detection)

*Since 0.5.* For DETR-style detectors (inputs `images` [B,3,H,W] and `orig_target_sizes` [B,2]; outputs `labels`
[B,Q], `boxes` [B,Q,4] as xyxy in input pixels, and `scores` [B,Q]), the frame pipeline runs the whole
frame-to-detections path on the GPU:

1. NV12 → RGB per tile, with resize, letterbox padding and normalisation.
2. The model, once per batch of tiles.
3. Score threshold, then tile → frame box mapping. Boxes cut by an interior tile edge are dropped, because the
   neighbouring tile sees that object whole.
4. Class-wise greedy NMS.

```c
munet_frame_pipeline* pipe = munet_frame_pipeline_create(program, /* max frame */ 3840, 2160, /* max tiles */ 32);

munet_nv12_frame frame = {
  .image = decoded_vkimage,                  /* VK_FORMAT_G8_B8R8_2PLANE_420_UNORM, TRANSFER_SRC usage, same device */
  .layout = current_layout,                  /* left in MUNET_FRAME_LAYOUT_AFTER (TRANSFER_SRC_OPTIMAL) */
  .width = 1920, .height = 1080};
/* or host planes: .image = NULL, .y = y, .y_pitch = ..., .uv = uv, .uv_pitch = ... (only the NV12 bytes are uploaded) */

munet_tile tiles[3] = {...};                 /* see below */
munet_detection found[300];
size_t count = 0;
munet_frame_detect(pipe, &frame, tiles, 3, /* color: NULL = BT.709 limited, no normalisation */ NULL,
                   /* score */ 0.15f, /* nms IoU */ 0.25f, &sync, found, 300, &count);
/* found[i]: x, y, width, height (frame pixels), score, label, input (the tile's input_index) */
munet_frame_pipeline_free(pipe);             /* before freeing the program */
```

**Tiles.** Each `munet_tile` describes one model input. It maps model input pixels to frame pixels as
`frame = input_px * scale + offset`, and model pixels outside `content_*` are black padding.

| Use | Settings |
|---|---|
| Stretch the whole frame | `scale = frame_size / model_size`, `offset = 0`, content = the full input, `has_region = 0` |
| Letterbox | content = the scaled image rectangle, `offset = -content_origin * scale` |
| Tile a large frame | one tile per crop; set `has_region = 1` and `region_*` to the frame area the tile is responsible for. Boxes touching an interior region edge (within 2 px) are rejected. |

If there are more tiles than the program's batch size, the pipeline runs the program as many times as needed.

**Colour.** `munet_frame_color` holds the YCbCr → RGB coefficients plus `mean` and `inv_std`, applied as
`(RGB/255 - mean) * inv_std`. Pick the matrix from the decoder's colour space and range, e.g. BT.601 for SD sources.

**Sync with a decoder.** The first submission waits on `sync->wait_semaphore`, i.e. the frame is decoded. The last
one signals `sync->signal_semaphore`, i.e. the frame is no longer read. These are exactly the semaphore and value
pair that an FFmpeg `AVVkFrame` carries. `munet_frame_detect` returns once the detections are on the host; only
those 32-byte records cross the bus.

## Threads, errors and device loss

- **Threads.** A program must not be used from two threads at once. Use one program per thread; several programs on
  one context are fine.
- **Errors.** Functions returning a pointer return `NULL` on failure, and functions returning `int32_t` return
  nonzero. `munet_last_error()` describes the most recent failure on the calling thread.
- **Device loss.** A lost device (GPU reset, driver crash) fails every later run on it. Recover by reloading the
  program; with a shared context, create a new context first.

## Diagnostics

- `munet_gpu_ms(program)`: GPU time of the last run's compute section. It excludes transfers, and is negative if
  unknown.
- `munet_stats_json(program, buf, size)`: the device name, load / shader compile / device setup times, the number of
  autotuned kernels, kernel and memory counters, run counts and the last `gpu_ms`, as JSON.
- `options.profile = 1` records a GPU timestamp per kernel. It serializes kernels, so use it to compare kernels,
  not to measure throughput.
- `MUNET_PIPELINE_STATS=1` prints the driver's statistics for each kernel (registers, spills, ...) to stderr when the
  program loads, on drivers that expose `VK_KHR_pipeline_executable_properties`.
- Vulkan validation layers (`VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`) are supported and expected to be clean.

## Environment variables

Full list: [environment.md](environment.md).

| Variable | Effect |
|---|---|
| `MUNET_LICENSE`, `MUNET_LICENSE_FILE` | license key, or the file holding it (default `~/.config/munet/license.key`) |
| `MUNET_CACHE_DIR` | SPIR-V and autotune cache (default `~/.cache/munet-next`), shared with Python |
| `MUNET_GLSLANG` | path of `glslangValidator` |
| `MUNET_VULKAN_LIBRARY` | explicit Vulkan loader path |
| `MUNET_DISABLE_COOPMAT=1` | fp32 kernels only (debugging) |
| `MUNET_SUBMIT_KERNELS=N` | split each run into submissions of at most N kernels (mobile drivers that reset on long submissions) |
| `MUNET_SUBMIT_SYNC=1` | wait after each submission and name the kernel range on failure (diagnosing device loss) |
| `MUNET_PIPELINE_STATS=1` | print driver statistics per kernel (registers, spills) to stderr at load |
