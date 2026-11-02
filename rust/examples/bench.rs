//! Times a .mnet program on random inputs: `cargo run --release --example bench -- model.mnet [runs] [fp32]`
use munet::{Options, Precision, Program};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let path = args.get(1).ok_or("usage: bench model.mnet [runs] [fp32]")?;
    let runs: usize = args.get(2).map(|s| s.parse()).transpose()?.unwrap_or(20);
    let precision = if args.get(3).map(String::as_str) == Some("fp32") {
        Precision::Fp32
    } else {
        Precision::Mixed
    };
    println!("MuNet {} on {:?}", munet::version(), munet::devices());
    let started = Instant::now();
    let mut program = Program::load(
        path,
        &Options {
            precision,
            ..Options::default()
        },
    )?;
    println!(
        "load {:.1} s; inputs {:?}",
        started.elapsed().as_secs_f64(),
        program.inputs()
    );
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
    let mut times = vec![];
    for run in 0..runs + 3 {
        let t = Instant::now();
        program.run(&views)?;
        for i in 0..program.outputs().len() {
            program.read(i)?;
        }
        if run >= 3 {
            times.push((
                t.elapsed().as_secs_f64() * 1e3,
                program.gpu_ms().unwrap_or(f64::NAN),
            ));
        }
    }
    times.sort_by(|a, b| a.0.total_cmp(&b.0));
    println!(
        "end-to-end min {:.2} ms median {:.2} ms; GPU at min {:.2} ms",
        times[0].0,
        times[times.len() / 2].0,
        times[0].1
    );
    println!("{}", program.stats_json()?);
    Ok(())
}
