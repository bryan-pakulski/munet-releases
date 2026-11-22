# munet-nn (Rust)

Run neural networks on any Vulkan GPU (AMD, Intel, NVIDIA, Apple through MoltenVK, Android Adreno/Mali) from Rust.
MuNet compiles a model graph into fused Vulkan compute kernels: fp16 and int8 cooperative-matrix kernels where the
GPU has them, portable fp16 or fp32 kernels everywhere else. It needs no CUDA, ROCm or vendor SDK, only a Vulkan driver.

```toml
[dependencies]
munet-nn = "0.11"   # the library is imported as `munet`
```

- [Requirements, build and licensing](#requirements-build-and-licensing)
- [Models](#models)
- [Load and run](#load-and-run)
- [Inputs and outputs](#inputs-and-outputs)
- [Program state and memory limit](#program-state-and-memory-limit)
- [Precision, int8 and autotuning](#precision-int8-and-autotuning)
- [Device probe](#device-probe)
- [Frame pipeline: video frames to detections](#frame-pipeline-video-frames-to-detections)
- [Zero copy with your own Vulkan code](#zero-copy-with-your-own-vulkan-code)
- [Threads, errors and device loss](#threads-errors-and-device-loss)
- [Diagnostics](#diagnostics)

## Requirements, build and licensing

Build:

- You need Rust and `curl`/`tar`. The build script downloads the prebuilt MuNet library for the target (Linux x86_64
  or aarch64) from the [release downloads](https://github.com/bryan-pakulski/munet-releases/releases) and verifies
  it against the SHA-256 sums published with the crate.
- Offline builds: set `MUNET_ARCHIVE=/path/munet-c-<ver>-<os>-<arch>.tar.gz`, or `MUNET_LIB_DIR` to the archive's
  `lib/static`.
- Inside a MuNet engine checkout (or with `MUNET_SOURCE_DIR`), the build script builds the library from source with
  CMake.
- `MUNET_DOWNLOAD_URL` overrides the download base URL.
- Android: the crate has no prebuilt library for Android targets and is not tested there. Android apps use the C API
  archive (`munet-android-<version>.tar.gz`) through JNI:
  [android.md](https://github.com/bryan-pakulski/munet-releases/blob/main/docs/android.md).

Run:

- You need a Vulkan 1.1+ driver and `glslangValidator` on `PATH` (packages `glslang-tools` / `glslang`, or the
  Vulkan SDK; `MUNET_GLSLANG` overrides the path).
- Kernels are compiled once and cached in `~/.cache/munet-next` (`MUNET_CACHE_DIR`), shared with the Python package.

License:

- These bindings are MIT. The native library is proprietary (see `LICENSE`).
- Without a license key it runs in evaluation mode and prints a notice.
- Production use needs a key in `MUNET_LICENSE`, `MUNET_LICENSE_FILE` or `~/.config/munet/license.key`
  ([details](https://github.com/bryan-pakulski/munet-releases/blob/main/docs/getting-started.md#license-key)).
- `munet::license()` returns the status as JSON.

The C API reference (`munet_c.h` and its guide) ships in the `munet-c` release archives.

## Models

Load an ONNX file directly with the engine's native importer (no Python). Shapes are static: give symbolic input
dimensions a shape, and load one program per batch size and resolution you serve:

```rust,no_run
use munet::{OnnxOptions, Options, Program};

fn main() -> munet::Result<()> {
    let onnx = OnnxOptions::default()
        .input_shape("images", &[3, 3, 1280, 1280])
        .input_shape("orig_target_sizes", &[3, 2]);
    let program = Program::load_onnx("detector.onnx", &onnx, &Options::default())?;
    println!("{:?}", program.inputs());
    Ok(())
}
```

It builds the same graph as Python's `from_onnx`, including QDQ / int8 layers, NonMaxSuppression and control flow.
Every operator the Python importer supports imports natively. Random operators keep a per-run step counter, as in
Python.

`OnnxOptions` also takes int8 activation scales (from `munet::calibrate_onnx` or Python's `calibrate_onnx`) and a
`.mnet` cache file. The cache is keyed by the ONNX file's SHA-256, the options, the MuNet version and the format
version:

```rust,no_run
use munet::{calibrate_onnx, Observer, OnnxOptions, Options, Program};

fn main() -> munet::Result<()> {
    let onnx = OnnxOptions::default().input_shape("images", &[1, 3, 640, 640]);
    let batch = vec![0.5f32; 3 * 640 * 640];
    let scales = calibrate_onnx("detector.onnx", &onnx, &Options::default(), &[vec![&batch[..]]], Observer::Mse, 2048)?;
    let onnx = scales
        .into_iter()
        .fold(onnx, |o, (layer, scale)| o.int8(layer, scale as f32))
        .cache("detector-int8.mnet");                 // reused while the ONNX file and options are unchanged
    let mut program = Program::load_onnx("detector.onnx", &onnx, &Options::default())?;
    program.save("copy.mnet")?;                       // the format munet.save writes
    Ok(())
}
```

Native import rules (cache, coverage, limits):
[onnx.md](https://github.com/bryan-pakulski/munet-releases/blob/main/docs/onnx.md#native-import-load-onnx-without-python).

Models with external tensor data or model-local functions need the Python importer. For those, export a `.mnet`
program from Python (`pip install munet-nn`):

```python
import munet
from munet.interop import from_onnx
program = from_onnx("detector.onnx", input_shapes={"images": (3, 3, 1280, 1280), "orig_target_sizes": (3, 2)})
munet.save(program, "detector-b3.mnet")   # records input/output names and declared dtypes
```

## Load and run

```rust,no_run
use munet::{Options, Precision, Program};

fn main() -> munet::Result<()> {
    let options = Options { precision: Precision::Fp16, ..Options::default() };   // device 0, autotune cache on
    let mut program = Program::load("detector-b3.mnet", &options)?;

    let images = vec![0.5f32; program.inputs()[0].len()];
    let sizes = vec![1280.0f32; program.inputs()[1].len()];
    program.run(&[&images, &sizes])?;                 // one f32 slice per input, in program.inputs() order

    let scores = program.read_by_name("scores")?;     // Vec<f32>
    let mut boxes = vec![0f32; program.outputs()[1].len()];
    program.read_into(1, &mut boxes)?;                // into your own buffer
    println!("{} scores, {:?} ms on {:?}", scores.len(), program.gpu_ms(), munet::devices());
    Ok(())
}
```

- Load is the expensive step. It compiles kernels (cached on disk after the first time), uploads the weights and
  records a command buffer that every `run` replays. Load once, then run many times.
- Data stays on the GPU. `run` stages the inputs inside the same GPU submission as the kernels. Weights and
  intermediate tensors never leave device memory, and only the outputs you read come back.
- `run` returns once the work is submitted. Reading an output waits for the run to finish.
- Read everything you need before the next `run`, because the next run overwrites the outputs.
- `Options { device: Device::Cpu, .. }` runs MuNet's CPU reference executor. It is slow and meant for tests. It is
  never used as a silent fallback.
- [`examples/bench.rs`](https://github.com/bryan-pakulski/munet-releases/blob/main/rust/examples/bench.rs) times a
  program: `cargo run --release --example bench -- model.mnet 50`.

## Inputs and outputs

```rust,no_run
fn describe(program: &munet::Program) {
    for t in program.inputs() {
        println!("{} {:?} {:?} ({} values)", t.name, t.dtype, t.shape, t.len());
    }
    let images: Option<usize> = program.input_index("images");
    println!("images is input {images:?}");
}
```

MuNet computes every tensor as `f32`:

- Integers: inputs and outputs whose declared `dtype.is_integer()` carry exact integer values as floats (exact
  up to 2^24). Round them to get the integers back.
- NMS outputs: `TensorInfo::padded` marks `NonMaxSuppression` outputs: `[bound, 3]` rows, where the unused rows
  are `-1`. `read_rows` / `read_i64` return the selected rows only, as Python does.

## Program state and memory limit

A program with parameters, such as the gradient program a training-pool master hands out, can have its state set and
read, and can be loaded under a memory limit.

```rust,no_run
use munet::{Options, Program};

fn main() -> munet::Result<()> {
    let options = Options::default();
    // Limit in bytes, 0 = none. Over the limit nothing is allocated.
    let mut program = match Program::load_limited("gradients.mnet", &options, 2 << 30)? {
        Ok(program) => program,
        Err(over) => return Err(munet::Error(format!("needs {} bytes", over.needs))),
    };
    let bytes = program.buffer_bytes()?;                 // peak while loading, held afterwards
    println!("setup {} steady {}", bytes.setup, bytes.steady);

    let mut weights = vec![0f32; 12];
    program.read_state(1, &mut weights)?;                // a parameter, buffer or output, by graph node id
    program.write_state(1, &weights)?;                   // set a parameter
    Ok(())
}
```

- The id is a node id of the graph `munet.save` wrote. A pool master lists the ids in its manifest.
- The slice length must equal the value's element count.
- The limit counts the buffers MuNet allocates. Memory the GPU driver maps on top is not counted.

## Precision, int8 and autotuning

| `Precision` | Python name | Matmul/conv arithmetic |
|---|---|---|
| `Mixed` (default) | `"mixed"` | fp16 operands, fp32 accumulation; int8 on quantized layers |
| `Fp16` | `"fp16"` | fp16 operands and accumulation; fastest on GPUs with fp16-accumulate cooperative matrices; validate accuracy per model |
| `Fp32` | `"fp32"` | exact float32 (reference) |

- GPUs without cooperative matrices (most mobile GPUs): `Mixed` runs fp32 kernels. `Fp16` runs portable fp16 kernels
  where shaders have fp16 arithmetic, and fp32 kernels elsewhere.
- How to choose: [inference.md](https://github.com/bryan-pakulski/munet-releases/blob/main/docs/inference.md#precision).
- int8 is part of the program. Import with `OnnxOptions::int8` scales (from `munet::calibrate_onnx` or Python's
  `calibrate_onnx`), or load an int8 `.mnet`. Its quantized convolutions then run on int8 kernels.
- Autotuning: run the program once from Python on the target GPU with `autotune=True` and the same precision.
  `Program::load` applies the cached results when `Options::autotune_cache` is set (the default). Untuned shapes use
  heuristics.

## Device probe

A few kernel choices depend on the GPU and driver. The probe checks each one against the CPU reference executor on
the GPU itself, times the correct ones and stores a profile. Later loads use the profile instead of the built-in
rules.

```rust,no_run
fn main() -> munet::Result<()> {
    let profile = munet::probe_device(0)?;            // a few seconds; returns the profile as JSON
    println!("{profile}");
    println!("{}", munet::device_profile(0)?);         // the stored profile, `{}` if the GPU was never probed
    Ok(())
}
```

- Run it once per GPU and driver version, for example on first start. The profile is stored in the kernel cache
  (`MUNET_CACHE_DIR`).
- A choice that returns wrong results on this GPU is never selected.
- `MUNET_CONV_DIRECT` and `MUNET_GEMM_VEC_STAGING` override the profile. Unset them before probing.
- Details: [inference.md](https://github.com/bryan-pakulski/munet-releases/blob/main/docs/inference.md#device-probe).

## Frame pipeline: video frames to detections

For DETR-style detectors (inputs `images` [B,3,H,W] and `orig_target_sizes` [B,2]; outputs `labels`,
`boxes` (xyxy, input pixels) and `scores`), a `FramePipeline` runs everything from an NV12 frame to the final
detections on the GPU:

1. Per-tile colour conversion, resize, letterbox padding and normalisation.
2. The model, once per batch of tiles.
3. Score threshold, then mapping boxes from tiles to the frame.
4. Class-wise NMS.

Only the detections (32 bytes each) are copied back to the CPU.

```rust,no_run
use munet::{FrameColor, Nv12, Options, Program, Tile};

fn detect(y: &[u8], uv: &[u8]) -> munet::Result<()> {     // a 1920x1080 NV12 frame
    let program = Program::load("detector-b3.mnet", &Options::default())?;
    let mut pipeline = program
        .into_frame_pipeline(3840, 2160, 32)       // max frame size, max tiles per frame
        .map_err(|(error, _program)| error)?;      // on failure you get the program back

    // One tile stretching the frame onto a 1280x1280 input.
    let tile = Tile {
        scale_x: 1920.0 / 1280.0, scale_y: 1080.0 / 1280.0, offset_x: 0.0, offset_y: 0.0,
        content_x: 0, content_y: 0, content_width: 1280, content_height: 1280,
        region_x0: 0.0, region_y0: 0.0, region_x1: 1920.0, region_y1: 1080.0, has_region: 0, input_index: 0,
    };
    let frame = Nv12::Host { y, y_pitch: 1920, uv, uv_pitch: 1920 };
    let found = pipeline.detect(frame, 1920, 1080, &[tile], &FrameColor::BT709_LIMITED,
                                /* score */ 0.3, /* NMS IoU */ 0.5, /* sync */ None, /* max */ 300)?;
    for d in &found {
        println!("{} {:.2} at ({}, {}) {}x{}", d.label, d.score, d.x, d.y, d.width, d.height);
    }
    Ok(())
}
```

Tiles, colour and decoder sync are explained in
[zero-copy.md](https://github.com/bryan-pakulski/munet-releases/blob/main/docs/zero-copy.md). In short:

- Tiles: a `Tile` maps model input pixels to frame pixels as `frame = input_px * scale + offset`. Input pixels
  outside `content_*` are black padding.
- Large frames: give each tile a crop at `scale = 1`, `offset` = the crop origin. Set `has_region = 1` and
  `region_*` to the part of the frame the tile is responsible for. Boxes touching an interior region edge are
  dropped, because the neighbouring tile sees that object whole.
- Colour: `FrameColor::BT709_LIMITED` (HD) and `BT601_LIMITED` (SD) give RGB in [0, 1]. For models that expect
  normalised input, set `mean` and `inv_std`, applied as `(rgb - mean) * inv_std`.
- GPU frames: pass `Nv12::Image { image, layout }` for a decoded `VkImage` on the program's device. The pipeline
  waits for and signals the decoder's timeline semaphore through `sync`. The image is left in
  `FramePipeline::IMAGE_LAYOUT_AFTER` (`TRANSFER_SRC_OPTIMAL`).
- `pipeline.program_mut()` still runs the program on host inputs.

[`examples/detect_frame.rs`](https://github.com/bryan-pakulski/munet-releases/blob/main/rust/examples/detect_frame.rs)
builds overlapping tiles for any frame size:
`cargo run --release --example detect_frame -- detector.mnet frame.nv12 1920 1080`.

## Zero copy with your own Vulkan code

MuNet runs one `VkDevice` per GPU, and other Vulkan code in your process can adopt it: a video decoder (FFmpeg's
Vulkan hwaccel), a renderer or your own compute shaders. Tensors and frames then never leave device memory. Video
walkthrough, including FFmpeg Vulkan decode:
[zero-copy.md](https://github.com/bryan-pakulski/munet-releases/blob/main/docs/zero-copy.md).

```rust,no_run
use munet::{Options, Program, SemaphoreSync, VulkanContext};
use std::ffi::c_void;

fn shared_device(timeline: *mut c_void) -> munet::Result<()> {   // a timeline VkSemaphore on MuNet's device
    // MuNet's device for GPU 0, with a video decode queue and the Vulkan video extensions enabled.
    let context = VulkanContext::new(0, /* video_decode */ true)?;
    let h = context.handles()?;
    // h.instance / physical_device / device / get_instance_proc_addr: adopt the device (e.g. AVVulkanDeviceContext).
    // h.extensions / h.features: what is enabled. h.compute_family + h.compute_queue: MuNet's queue.
    // h.decode_family + h.decode_queues: the video decode queue.
    println!("decode queue family {:?}", h.decode_family);

    let mut program = Program::load_on(&context, "detector.mnet", &Options::default())?;
    let buffer = program.vulkan_buffer()?;   // VkBuffer: write inputs here with your own commands ...
    let images = program.input_offset(0)?;   // ... at these byte offsets (contiguous f32)
    let scores = program.output_offset(2)?;  // outputs are read from the same buffer
    println!("buffer {buffer:?}: images at {images}, scores at {scores}");

    // GPU-ordered run: waits for your writes (timeline = 1), signals completion (timeline = 2). Returns immediately.
    program.run_device(Some(&SemaphoreSync {
        wait_semaphore: timeline, wait_value: 1, signal_semaphore: timeline, signal_value: 2,
    }))?;
    program.wait()                           // only if the host must block
}
```

Rules:

- Timeline semaphores: the semaphores must be timeline semaphores created on the shared device.
- Queue locking: Vulkan queues are externally synchronized. Hold `context.lock_queue(family, index)`, a guard,
  around every submit or wait you make on MuNet's queues. For C callbacks (such as FFmpeg's `lock_queue` /
  `unlock_queue`), use `lock_queue_raw` / `unsafe unlock_queue_raw`.
- Host or device inputs: every program accepts both host inputs (`run`) and device inputs (`run_device`), and can
  switch between them on any call.
- Lifetime: programs keep their device alive.

## Threads, errors and device loss

- Threads: `Program` and `FramePipeline` are `Send` but not `Sync`: use one per thread (or behind a `Mutex`).
  `VulkanContext` is `Send + Sync`, so several programs can share one.
- Errors: every failure is a `munet::Error` with the native message: load, shader compilation, device or shape
  errors.
- Device loss: a lost device (GPU reset, driver crash) fails every later run on that program. Recover by
  reloading it. With a shared `VulkanContext`, create a new `VulkanContext` first and reload on it.

## Diagnostics

- `program.gpu_ms()`: GPU time of the last run's compute section.
- `program.stats_json()`: the device name, load / shader compile / setup times, the number of autotuned kernels,
  kernel and memory counters, and run counts.
- `Options { profile: true, .. }` records a GPU timestamp per kernel. It serializes the kernels, so use it to compare
  kernels, not to measure throughput.
- `munet::devices()`, `munet::device_type(index)`, `munet::device_pci_address(index)` and
  `munet::device_driver_warning(index)` describe the GPUs.
- `MUNET_PIPELINE_STATS=1` prints the driver's per-kernel statistics (registers, spills) to stderr at load.
- Vulkan validation layers (`VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`) are supported.
