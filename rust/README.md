# munet-nn (Rust)

Run neural networks on any Vulkan GPU (AMD, Intel, NVIDIA, Apple through MoltenVK, Android Adreno/Mali) from Rust.
MuNet compiles a model graph into fused Vulkan compute kernels: fp16 and int8 cooperative-matrix kernels where the
GPU has them, fp32 everywhere else. It needs no CUDA, ROCm or vendor SDK, only a Vulkan driver.

```toml
[dependencies]
munet-nn = "0.5"   # the library is imported as `munet`
```

- [Models](#models)
- [Load and run](#load-and-run)
- [Inputs and outputs](#inputs-and-outputs)
- [Precision, int8 and autotuning](#precision-int8-and-autotuning)
- [Frame pipeline: video frames to detections](#frame-pipeline-video-frames-to-detections)
- [Zero copy with your own Vulkan code](#zero-copy-with-your-own-vulkan-code)
- [Threads, errors and device loss](#threads-errors-and-device-loss)
- [Diagnostics](#diagnostics)
- [Requirements, build and licensing](#requirements-build-and-licensing)

## Models

Programs are `.mnet` files exported once from Python (`pip install munet-nn`). Shapes are static, so export one
program per batch size and resolution you serve:

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

- **Load is the expensive step.** It compiles kernels (cached on disk after the first time), uploads the weights and
  records a command buffer that every `run` replays. Load once, then run many times.
- **Data stays on the GPU.** `run` stages the inputs inside the same GPU submission as the kernels. Weights and
  intermediate tensors never leave device memory, and only the outputs you read come back.
- **Run timing.** `run` returns once the work is submitted. Reading an output waits for the run to finish. Read
  everything you need before the next `run`, because the next run overwrites the outputs.
- **CPU executor.** `Options { device: Device::Cpu, .. }` runs MuNet's CPU reference executor. It is slow and meant
  for tests; it is never used as a silent fallback.
- [`examples/bench.rs`](examples/bench.rs) times a program:
  `cargo run --release --example bench -- model.mnet 50`.

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

- **Integers.** Inputs and outputs whose declared `dtype.is_integer()` carry exact integer values as floats (exact
  up to 2^24). Round them to get the integers back.
- **NMS outputs.** `TensorInfo::padded` marks `NonMaxSuppression` outputs: `[bound, 3]` rows, where the unused rows
  are `-1` and you drop them.

## Precision, int8 and autotuning

| `Precision` | Matmul/conv arithmetic | Use |
|---|---|---|
| `Mixed` (default) | fp16 operands, fp32 accumulation; int8 on quantized layers | general inference |
| `Fp16` | fp16 operands and accumulation | fastest on GPUs with fp16-accumulate cooperative matrices (2x `Mixed` on GeForce Ampere/Ada); validate accuracy per model |
| `Fp32` | exact float32 | reference |

- **GPUs without cooperative matrices** (most mobile GPUs) run the fp32 kernels in every mode.
- **int8** is part of the program: export with int8 activation scales from Python (`from_onnx(..., int8=scales)` with
  `munet.quantization.calibrate_onnx`), and its convolutions run on int8 kernels.
- **Autotuning:** run the program once from Python on the target GPU with `autotune=True` and the same precision.
  The winning tile configurations are cached in `MUNET_CACHE_DIR`, and `Program::load` applies them (when
  `Options::autotune_cache` is set, the default). Untuned shapes use heuristics.

## Frame pipeline: video frames to detections

*Since 0.5.* For DETR-style detectors (inputs `images` [B,3,H,W] and `orig_target_sizes` [B,2]; outputs `labels`,
`boxes` (xyxy, input pixels) and `scores`), a `FramePipeline` runs everything from an NV12 frame to the final
detections on the GPU:

1. Per-tile colour conversion, resize, letterbox padding and normalisation.
2. The model, once per batch of tiles.
3. Score threshold, then mapping boxes from tiles to the frame.
4. Class-wise NMS.

The only data copied back to the CPU is the detections (32 bytes each).

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

**Tiles.** A `Tile` maps model input pixels to frame pixels as `frame = input_px * scale + offset`. Input pixels
outside `content_*` are black padding. To tile a large frame:
- Give each tile a crop at `scale = 1`, `offset` = the crop origin.
- Set `has_region = 1` and `region_*` to the part of the frame the tile is responsible for. Boxes touching an
  interior region edge are dropped, because the neighbouring tile sees that object whole.

[`examples/detect_frame.rs`](examples/detect_frame.rs) builds overlapping tiles for any frame size:
`cargo run --release --example detect_frame -- detector.mnet frame.nv12 1920 1080`.

**Colour.** `FrameColor::BT709_LIMITED` (HD) and `BT601_LIMITED` (SD) give RGB in [0, 1]. Set `mean` and `inv_std`
for models that expect normalised input, applied as `(rgb - mean) * inv_std`.

**Frames already on the GPU.** Pass `Nv12::Image { image, layout }` for a decoded `VkImage` on the program's device.
The pipeline then waits for and signals the decoder's timeline semaphore through `sync` (see the next section). The
image is left in `FramePipeline::IMAGE_LAYOUT_AFTER` (`TRANSFER_SRC_OPTIMAL`). `pipeline.program_mut()` still runs
the program on host inputs.

## Zero copy with your own Vulkan code

> Video walkthrough, including FFmpeg Vulkan decode: [docs/zero-copy.md](https://github.com/bryan-pakulski/munet-releases/blob/main/docs/zero-copy.md).

MuNet runs one `VkDevice` per GPU, and other Vulkan code in your process can adopt it, for example a video decoder
(FFmpeg's Vulkan hwaccel), a renderer or your own compute shaders. Tensors and frames then never leave device memory,
and nothing outside Vulkan is involved, so this works the same on AMD, NVIDIA, Intel and mobile GPUs.

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

- **Timeline semaphores.** The semaphores must be timeline semaphores created on the shared device.
- **Queue locking.** Vulkan queues are externally synchronized. Hold `context.lock_queue(family, index)`, a guard,
  around every submit or wait you make on MuNet's queues. For C callbacks (such as FFmpeg's `lock_queue` /
  `unlock_queue`), use `lock_queue_raw` / `unsafe unlock_queue_raw`.
- **Host or device inputs.** Every program accepts both host inputs (`run`) and device inputs (`run_device`), and can
  switch between them on any call.
- **Lifetime.** Programs keep their device alive. After a device loss, create a new `VulkanContext` and reload.

## Threads, errors and device loss

- **Threads.** `Program` and `FramePipeline` are `Send` but not `Sync`: use one per thread (or behind a `Mutex`).
  Several programs can share one device or `VulkanContext`.
- **Errors.** Every failure is a `munet::Error` with the native message: load, shader compilation, device or shape
  errors.
- **Device loss.** A lost device (GPU reset, driver crash) fails every later run on that program; recover by
  reloading it.

## Diagnostics

- `program.gpu_ms()`: GPU time of the last run's compute section.
- `program.stats_json()`: the device name, load / shader compile / setup times, the number of autotuned kernels,
  kernel and memory counters, and run counts.
- `Options { profile: true, .. }` records a GPU timestamp per kernel. It serializes the kernels, so use it to compare
  kernels, not to measure throughput.
- `MUNET_PIPELINE_STATS=1` prints the driver's per-kernel statistics (registers, spills) to stderr at load.
- Vulkan validation layers (`VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation`) are supported.

## Requirements, build and licensing

- **Build.** You need nothing beyond Rust and `curl`/`tar`. The build script downloads the prebuilt MuNet library for
  the target (Linux x86_64/aarch64; macOS arm64 and Windows x86_64 as they become available) from the
  [release downloads](https://github.com/bryan-pakulski/munet-releases/releases) and verifies it against the SHA-256
  sums published with the crate. Offline builds have two options:
  - `MUNET_ARCHIVE=/path/munet-c-<ver>-<os>-<arch>.tar.gz`;
  - `MUNET_LIB_DIR` pointing at the archive's `lib/static`.
- **Run.** You need a Vulkan 1.1+ driver and `glslangValidator` on `PATH` (packages `glslang-tools` / `glslang`, or
  the Vulkan SDK; `MUNET_GLSLANG` overrides the path). Kernels are compiled once and cached in `~/.cache/munet-next`
  (`MUNET_CACHE_DIR`), shared with the Python package.
- **License.** These bindings are MIT. The native library is proprietary (see `LICENSE`):
  - Without a license key it runs in evaluation mode and prints a notice.
  - Production use needs a key in `MUNET_LICENSE`, `MUNET_LICENSE_FILE` or `~/.config/munet/license.key`.
  - `munet::license()` returns the status as JSON.
- **Other languages.** The C API reference (`munet_c.h` and its guide) ships in the `munet-c` release archives.
