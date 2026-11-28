//! Autotunes a .mnet program on this GPU and shows what it gained, without Python:
//! `cargo run --release --example autotune -- model.mnet [--device N] [--precision mixed|fp32|fp16|bf16] [--budget SECONDS]`
//!
//! Times the program with MuNet's built-in kernel choices, tunes the kernels the cache lacks, loads the program again
//! and times it with the tuned kernels. Results are stored in the kernel cache (`MUNET_CACHE_DIR`), so later loads on
//! this GPU and driver use them. Tune with the precision and batch size you ship.
use munet::{Device, Options, Precision, Program};
use std::time::{Duration, Instant};

/// Median GPU milliseconds of one run, after keeping the GPU busy for a second (idle GPUs run at low clocks).
fn gpu_ms(program: &mut Program) -> Result<f64, Box<dyn std::error::Error>> {
    let data: Vec<Vec<f32>> = program
        .inputs()
        .iter()
        .map(|t| {
            (0..t.len())
                .map(|i| ((i * 7919) % 255) as f32 / 255.0)
                .collect()
        })
        .collect();
    let views: Vec<&[f32]> = data.iter().map(Vec::as_slice).collect();
    let warm = Instant::now();
    while warm.elapsed() < Duration::from_secs(1) {
        program.run(&views)?;
        program.wait()?;
    }
    let mut times = Vec::new();
    for _ in 0..30 {
        program.run(&views)?;
        program.wait()?;
        times.push(program.gpu_ms().ok_or("this GPU reports no timestamps")?);
    }
    times.sort_by(f64::total_cmp);
    Ok(times[times.len() / 2])
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let usage = "usage: autotune model.mnet [--device N] [--precision mixed|fp32|fp16|bf16] [--budget SECONDS]";
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (mut path, mut device, mut precision, mut budget) = (None, 0u32, Precision::Mixed, 600.0);
    let mut rest = args.iter();
    while let Some(arg) = rest.next() {
        let mut value = || rest.next().ok_or(usage);
        match arg.as_str() {
            "--device" => device = value()?.parse()?,
            "--budget" => budget = value()?.parse()?,
            "--precision" => {
                precision = match value()?.as_str() {
                    "mixed" => Precision::Mixed,
                    "fp32" => Precision::Fp32,
                    "fp16" => Precision::Fp16,
                    "bf16" => Precision::Bf16,
                    _ => return Err(usage.into()),
                }
            }
            _ if path.is_none() && !arg.starts_with("--") => path = Some(arg.clone()),
            _ => return Err(usage.into()),
        }
    }
    let path = path.ok_or(usage)?;
    let name = munet::devices()
        .get(device as usize)
        .cloned()
        .ok_or("no such Vulkan device")?;
    println!("MuNet {} on {name}", munet::version());

    // Built-in choices first: a load that ignores the cache.
    let builtin = Options {
        device: Device::Vulkan(device),
        precision,
        autotune_cache: false,
        ..Options::default()
    };
    let mut program = Program::load(&path, &builtin)?;
    let before = gpu_ms(&mut program)?;
    println!("built-in kernels: {before:.2} ms GPU per run");

    let report = program.autotune(Duration::from_secs_f64(budget))?;
    println!("{report}");
    drop(program);

    let tuned = Options {
        autotune_cache: true,
        ..builtin
    };
    let mut program = Program::load(&path, &tuned)?;
    let after = gpu_ms(&mut program)?;
    println!(
        "tuned kernels:    {after:.2} ms GPU per run ({:+.1}%)",
        (after / before - 1.0) * 100.0
    );
    Ok(())
}
