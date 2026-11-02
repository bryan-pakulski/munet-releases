# Zero copy and video detection

For video, uploading every decoded frame as float32 costs more than running the detector. MuNet can instead keep
frames on the GPU from the decoder to the detections: only the detections, 32 bytes each, come back to the CPU. These
are native APIs (C, C++, Rust) and work on any Vulkan GPU.

| Your frames are | Use | Copies |
|---|---|---|
| NV12 in CPU memory (software decode, camera SDK) | [frame pipeline, host frames](#1-frame-pipeline-with-host-frames) | NV12 bytes up (1.5 bytes/pixel instead of 12), detections down |
| Decoded on the GPU (FFmpeg Vulkan hwaccel, Vulkan Video) | [frame pipeline, GPU frames](#2-frames-decoded-on-the-gpu) | none |
| Your own tensors in Vulkan buffers (custom preprocessing, rendering) | [shared device, `run_device`](#3-your-own-vulkan-buffers) | none |

## Export the model

The frame pipeline runs DETR-style detectors: RT-DETR, D-FINE, DEIM and similar.

- **Inputs:** `images [B, 3, H, W]`, `orig_target_sizes [B, 2]`.
- **Outputs:** `labels [B, Q]`, `boxes [B, Q, 4]` (xyxy, input pixels), `scores [B, Q]`.

`B` is the number of tiles per model run:

```python
import munet
from munet.interop import from_onnx
model = from_onnx("rtdetr.onnx", input_shapes={"images": (3, 3, 1280, 1280), "orig_target_sizes": (3, 2)})
munet.save(model, "detector-b3.mnet")
# optional, on the target GPU: tune once, cached for C/C++/Rust loads
from_onnx("rtdetr.onnx", input_shapes={...}, precision="fp16", autotune=True)(*example_inputs)
```

For other model types (YOLO, segmentation), use [shared buffers with `run_device`](#3-your-own-vulkan-buffers) and do
the pre/post-processing yourself.

## 1. Frame pipeline with host frames

Per frame, the pipeline:

1. converts NV12 to RGB, resizes and letterboxes each tile, and normalises;
2. runs the detector over all tiles in batches;
3. filters by score and maps boxes back to the frame;
4. runs class-wise NMS.

All of it happens on the GPU.

**Rust:**

```rust
use munet::{FrameColor, Nv12, Options, Program, Tile};

let program = Program::load("detector-b3.mnet", &Options { precision: munet::Precision::Fp16, ..Default::default() })?;
let mut pipeline = program.into_frame_pipeline(3840, 2160, 32).map_err(|(e, _)| e)?;   // max frame, max tiles

let tile = Tile {                                    // whole 1920x1080 frame stretched onto the 1280x1280 input
    scale_x: 1920.0 / 1280.0, scale_y: 1080.0 / 1280.0, offset_x: 0.0, offset_y: 0.0,
    content_x: 0, content_y: 0, content_width: 1280, content_height: 1280,
    region_x0: 0.0, region_y0: 0.0, region_x1: 1920.0, region_y1: 1080.0, has_region: 0, input_index: 0,
};
loop {
    let (y, uv) = next_frame();                      // NV12 planes from your decoder
    let found = pipeline.detect(Nv12::Host { y, y_pitch: 1920, uv, uv_pitch: 1920 }, 1920, 1080, &[tile],
                                &FrameColor::BT709_LIMITED, 0.3, 0.5, None, 300)?;
    for d in &found { println!("{} {:.2} {} {} {} {}", d.label, d.score, d.x, d.y, d.width, d.height); }
}
```

**C / C++:**

```c
munet_program* program = munet_load("detector-b3.mnet", &options);
munet_frame_pipeline* pipe = munet_frame_pipeline_create(program, 3840, 2160, 32);
munet_nv12_frame frame = {.image = NULL, .y = y, .y_pitch = 1920, .uv = uv, .uv_pitch = 1920,
                          .width = 1920, .height = 1080};
munet_detection found[300]; size_t count = 0;
munet_frame_detect(pipe, &frame, tiles, n_tiles, NULL /* BT.709 limited */, 0.3f, 0.5f, NULL, found, 300, &count);
munet_frame_pipeline_free(pipe);                     /* before munet_free(program) */
```

Complete programs: [rust/examples/detect_frame.rs](../rust/examples/detect_frame.rs) (overlapping tiles for any frame
size), and the frame-pipeline sections of [c.md](c.md#frame-pipeline-video-detection) and
[rust/README.md](../rust/README.md#frame-pipeline-video-frames-to-detections).

### Tiles

A tile maps model input pixels to frame pixels: `frame = input_px * scale + offset`. Model pixels outside
`content_*` are padding.

| Goal | Tile settings |
|---|---|
| Whole frame, stretched | `scale = frame_size / model_size`, `offset = 0`, content = whole input |
| Whole frame, letterboxed (keeps aspect) | `scale = max(frame_w / model_w, frame_h / model_h)`, content = the image rectangle inside the input, `offset = -content_origin * scale` |
| Small objects in a large frame | split into overlapping crops at `scale = 1`, `offset` = crop origin; `has_region = 1`, `region_*` = the area this tile owns (the crop minus half the overlap) |

With regions, a box that touches an interior region edge is dropped, because the neighbouring tile sees that object
whole. Make the model batch size equal to the usual number of tiles per frame. More tiles than that run in several
batches.

### Colour

- `NULL` / `FrameColor::BT709_LIMITED` is HD video. `BT601_LIMITED` is SD video.
- For models trained on normalised input, set `mean` and `inv_std` (applied as `(rgb - mean) * inv_std`).
- Match the decoder's colour space and range, or colours (and accuracy) shift.

## 2. Frames decoded on the GPU

Create MuNet's Vulkan device first, with a video decode queue, and hand it to the decoder. The decoder then writes
frames into `VkImage`s on the same device, and the pipeline reads them in place.

```rust
let context = munet::VulkanContext::new(0, /* video_decode */ true)?;
let h = context.handles()?;          // instance, physical_device, device, get_instance_proc_addr,
                                     // extensions, features, compute/decode queue families and queues
let program = munet::Program::load_on(&context, "detector-b3.mnet", &options)?;
let mut pipeline = program.into_frame_pipeline(3840, 2160, 32).map_err(|(e, _)| e)?;
```

**With FFmpeg** (Vulkan hwaccel):

1. Allocate an `AV_HWDEVICE_TYPE_VULKAN` device context. Fill its `AVVulkanDeviceContext` from the handles: instance,
   physical device, device, `get_proc_addr`, the enabled extensions and features, and the queue families (compute and
   decode).
2. Route FFmpeg's queue locking to MuNet's, so the two never submit to a queue at the same time. FFmpeg passes its
   own context to the callbacks, so wrap them:
   ```c
   static munet_vulkan* g_ctx;   /* or keep it in AVHWDeviceContext.user_opaque */
   static void lock(AVHWDeviceContext* d, uint32_t family, uint32_t index)   { munet_vulkan_lock_queue(g_ctx, family, index); }
   static void unlock(AVHWDeviceContext* d, uint32_t family, uint32_t index) { munet_vulkan_unlock_queue(g_ctx, family, index); }
   vk->lock_queue = lock; vk->unlock_queue = unlock;
   ```
   In Rust, call `VulkanContext::lock_queue_raw` / `unlock_queue_raw` from the callbacks. Then call
   `av_hwdevice_ctx_init`.
3. Decode as usual. Each output `AVFrame` carries an `AVVkFrame` with `img[0]` (NV12, `G8_B8R8_2PLANE_420`),
   `layout[0]`, `sem[0]` and `sem_value[0]`.
4. Call detect with the image, its layout, and the frame's semaphore. The pipeline waits for the decode to finish
   and signals when it no longer reads the frame:

   ```rust
   let sync = munet::SemaphoreSync { wait_semaphore: sem, wait_value: value,
                                      signal_semaphore: sem, signal_value: value + 1 };
   let found = pipeline.detect(munet::Nv12::Image { image, layout }, w, h, &tiles, &color, 0.3, 0.5, Some(&sync), 300)?;
   // then: vk_frame.sem_value[0] = value + 1; vk_frame.layout[0] = FramePipeline::IMAGE_LAYOUT_AFTER;
   ```

Requirements:

- The decoder must use MuNet's device. Images from another `VkDevice` cannot be read.
- Images need `TRANSFER_SRC` usage, which FFmpeg's default frame pool sets.
- On Intel Mesa, set `ANV_DEBUG=video-decode` ([gpus.md](gpus.md#intel-arc-xe)).
- With NVIDIA containers, add `video` to `NVIDIA_DRIVER_CAPABILITIES`.

## 3. Your own Vulkan buffers

Any program can take its inputs from, and leave its outputs in, a `VkBuffer` that your own Vulkan work writes and
reads:

```c
munet_vulkan* ctx = munet_vulkan_create(0, 0);                 /* or adopt its handles in your engine */
munet_program* program = munet_load_on(ctx, "model.mnet", &options);
VkBuffer buffer = (VkBuffer)munet_program_vulkan_buffer(program);
int64_t in  = munet_input_offset(program, 0);                  /* write contiguous float32 input here */
int64_t out = munet_output_offset(program, 0);                 /* read the output from here */

/* your commands write `buffer` at `in`, then signal timeline = 1 */
munet_semaphore_sync sync = {.wait_semaphore = timeline, .wait_value = 1,
                             .signal_semaphore = timeline, .signal_value = 2};
munet_run_device(program, &sync);                              /* returns immediately */
/* your commands wait for timeline >= 2 and read `buffer` at `out` */
```

- **Synchronization.** Semaphores must be timeline semaphores on MuNet's device. Hold MuNet's queue lock around
  every submit you make on its queues.
- **Mixing paths.** `munet_run` (host arrays) and `munet_run_device` can be mixed freely.
- **Full example.** The validation-clean version is
  [examples/c/vulkan_interop.cpp](../examples/c/vulkan_interop.cpp); for Rust (`VulkanContext`,
  `Program::run_device`), see the [Rust guide](../rust/README.md#zero-copy-with-your-own-vulkan-code).

## Throughput tips

- **Stream layout.** Use one pipeline (and program) per video stream and thread. Several programs can share one
  `VulkanContext`.
- **Batch the tiles.** Export the batch size equal to your tiles per frame (e.g. 3 tiles of 1280² for 4K).
- **Use fp16 and autotuning** on the deployment GPU ([inference.md](inference.md)).
- **Sharing the decoder's GPU.** Decode and inference share the GPU. Use a second GPU (a second `VulkanContext`)
  for more streams.
