//! Detections for one NV12 frame with the GPU frame pipeline (DETR programs):
//! `cargo run --release --example detect_frame -- detector.mnet frame.nv12 1920 1080`
//!
//! Make a raw NV12 frame with `ffmpeg -i video.mp4 -frames:v 1 -pix_fmt nv12 -f rawvideo frame.nv12`. The frame is
//! split into square tiles of the model's input size (letterboxed where it is smaller), each tile is responsible for
//! its region of the frame, and detections are returned in frame pixels.
use munet::{FrameColor, Nv12, Options, Precision, Program, Tile};

fn tiles(width: u32, height: u32, size: u32) -> Vec<Tile> {
    // Tile starts along one axis: evenly spread, at least `size` apart where the frame is larger than a tile.
    let starts = |length: u32| -> Vec<u32> {
        if length <= size {
            return vec![0];
        }
        let count = (length + size - 1) / size; // div_ceil needs Rust 1.73 (MSRV 1.70)
        (0..count).map(|i| i * (length - size) / (count - 1)).collect()
    };
    let (xs, ys) = (starts(width), starts(height));
    let mut out = Vec::new();
    for (row, &y) in ys.iter().enumerate() {
        for (column, &x) in xs.iter().enumerate() {
            let (w, h) = (size.min(width - x), size.min(height - y));
            // Responsibility region: split the overlap between neighbouring tiles at its middle.
            let edge = |starts: &[u32], i: usize, end: bool, length: u32| -> f32 {
                match (end, i) {
                    (false, 0) => 0.0,
                    (false, i) => (starts[i] + starts[i - 1] + size) as f32 / 2.0,
                    (true, i) if i + 1 == starts.len() => length as f32,
                    (true, i) => (starts[i + 1] + starts[i] + size) as f32 / 2.0,
                }
            };
            out.push(Tile {
                scale_x: 1.0,
                scale_y: 1.0,
                offset_x: x as f32,
                offset_y: y as f32,
                content_x: 0,
                content_y: 0,
                content_width: w as i32,
                content_height: h as i32,
                region_x0: edge(&xs, column, false, width),
                region_y0: edge(&ys, row, false, height),
                region_x1: edge(&xs, column, true, width),
                region_y1: edge(&ys, row, true, height),
                has_region: i32::from(xs.len() > 1 || ys.len() > 1),
                input_index: out.len() as i32,
            });
        }
    }
    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let usage = "usage: detect_frame detector.mnet frame.nv12 width height";
    let (model, frame) = (args.get(1).ok_or(usage)?, args.get(2).ok_or(usage)?);
    let width: u32 = args.get(3).ok_or(usage)?.parse()?;
    let height: u32 = args.get(4).ok_or(usage)?.parse()?;
    let bytes = std::fs::read(frame)?;
    let luma = (width * height) as usize;
    if bytes.len() < luma + luma / 2 {
        return Err(format!("{frame} is smaller than a {width}x{height} NV12 frame").into());
    }

    let options = Options {
        precision: Precision::Fp16,
        ..Options::default()
    };
    let program = Program::load(model, &options)?;
    let size = program.inputs()[0].shape[3] as u32; // images: [B, 3, H, W]
    let mut pipeline = program
        .into_frame_pipeline(width, height, 32)
        .map_err(|(error, _program)| error)?;

    let tiles = tiles(width, height, size);
    // BT.709 limited range (HD video), RGB in [0, 1] as RT-DETR exports expect. Models trained on ImageNet-normalised
    // input instead set `mean: [0.485, 0.456, 0.406]` and `inv_std: [1/0.229, 1/0.224, 1/0.225]`.
    let color = FrameColor::BT709_LIMITED;
    let host = Nv12::Host {
        y: &bytes[..luma],
        y_pitch: width as usize,
        uv: &bytes[luma..luma + luma / 2],
        uv_pitch: width as usize,
    };
    let started = std::time::Instant::now();
    let detections = pipeline.detect(host, width, height, &tiles, &color, 0.3, 0.5, None, 300)?;
    println!(
        "{} tiles of {size}px, {} detections in {:.1} ms",
        tiles.len(),
        detections.len(),
        started.elapsed().as_secs_f64() * 1e3
    );
    for d in detections.iter().take(20) {
        println!(
            "label {:3} score {:.3} box ({:.0}, {:.0}) {:.0}x{:.0}",
            d.label, d.score, d.x, d.y, d.width, d.height
        );
    }
    Ok(())
}
