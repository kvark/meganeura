//! Time the 128×128 scalar matmul tile against the 64×64 incumbent.
//!
//! On an RTX 5080 (cooperative kernels off, 512 MiB scratch) every 128
//! candidate lost. The best 4096³ race sample was 8.026 ms / 17.12 TFLOP/s
//! against the installed 64×64 k-stage 16 at 6.885 ms / 19.96 TFLOP/s, and
//! the whole step stayed 64-wide (7.360 ms / 18.67 TFLOP/s, workgroups
//! [64, 64, 1]). The best 2048³ 128 sample was 1.421 ms against 0.901 ms.
//! That is the spill signal for an 8×8 register tile on a 256-thread group.
//!
//! The search scratch has to cover the operand set. A 4096³ f32 product is
//! three 64 MiB buffers plus a staging copy, and the default 64 MiB budget
//! ScratchLimits that shape back to the default tile.
//!
//! One square per process is the safe way to run this; a few shapes in one
//! process is fine. Strict f32, cooperative kernels off:
//!
//!   MEGANEURA_DEVICE_ID=0x2c02 MEGANEURA_GPU_TIMING=1 MEGANEURA_DISABLE_COOP=1 \
//!   cargo run --release --example wide_tile_probe -- 4096 2048
//!
//! Race medians are per-dispatch scratch times. The `step` row matches
//! `op_bench`: median summed GPU pass time, then median wall time, then
//! TFLOP/s from the GPU time.

use meganeura::{
    Graph, Session, SessionConfig, TuneCalibration, TuneOptions, TuneReport, TuneStrategy,
};
use std::time::{Duration, Instant};

fn fill(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i * 31 + seed * 17) % 251) as f32 - 125.0) * 0.004)
        .collect()
}

fn median(mut values: Vec<f64>) -> f64 {
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    values[values.len() / 2]
}

fn tflops(m: usize, n: usize, k: usize, ms: f64) -> f64 {
    2.0 * (m * n * k) as f64 / ms / 1e9
}

fn measure(session: &mut Session, m: usize, n: usize, k: usize) {
    session.set_profiling_window(Some(0..0));
    let mut early = Vec::new();
    for _ in 0..6 {
        session.step();
        session.wait();
        let timings = session.gpu_timings();
        early.push(timings.iter().map(|(_, d)| d.as_secs_f64() * 1e3 / 30.0).sum::<f64>());
    }
    session.set_profiling_window(None);
    println!("early\t{:?}", early);
    let warm = Instant::now();
    while warm.elapsed().as_secs_f32() < 1.0 {
        session.step();
        session.wait();
    }
    let mut wall = Vec::new();
    for _ in 0..50 {
        let start = Instant::now();
        session.step();
        session.wait();
        wall.push(start.elapsed().as_secs_f64() * 1e3 / 30.0);
    }
    session.set_profiling_window(Some(0..0));
    let mut gpu = Vec::new();
    let mut passes = String::new();
    for _ in 0..30 {
        session.step();
        session.wait();
        let timings = session.gpu_timings();
        if timings.is_empty() {
            panic!("no GPU timestamps; set MEGANEURA_GPU_TIMING=1 before the context is created");
        }
        passes = timings
            .iter()
            .map(|(name, _)| name.clone())
            .collect::<Vec<_>>()
            .join("+");
        gpu.push(timings.iter().map(|(_, d)| d.as_secs_f64() * 1e3 / 30.0).sum());
    }
    session.set_profiling_window(None);
    let gpu_ms = median(gpu);
    let plan = session.plan();
    let workgroups = plan
        .dispatches
        .iter()
        .find(|dispatch| dispatch.shader == meganeura::compile::ShaderEntry::MatMul)
        .map(|dispatch| dispatch.workgroups)
        .unwrap_or([0, 0, 0]);
    println!(
        "step\t{m}x{n}x{k}\t{gpu_ms:.4}\t{:.4}\t{:.3}\twg={workgroups:?}\t{passes}",
        median(wall),
        tflops(m, n, k, gpu_ms),
    );
}

fn report_race(report: &TuneReport, m: usize, n: usize, k: usize) {
    println!(
        "search\t{m}x{n}x{k}\tclasses {}/{} outcomes {} elapsed {:.3}s budget {} scratch_peak {}",
        report.visited_classes,
        report.eligible_classes,
        report.outcomes.len(),
        report.elapsed.as_secs_f64(),
        report.time_budget_exhausted,
        report
            .scratch
            .map(|scratch| scratch.peak_bytes)
            .unwrap_or(0),
    );
    for race in &report.races {
        println!(
            "race\t{m}x{n}x{k}\tcandidates {}\trounds {}\t{:?} -> {:?}\tincumbent {:.4} ms\tcompile {:.3}s\ttimestamps {}\tcomplete {}",
            race.candidates,
            race.rounds,
            race.initial,
            race.selected,
            race.incumbent_ms.unwrap_or(f64::NAN),
            race.compile_time.as_secs_f64(),
            race.gpu_timestamps,
            race.complete,
        );
    }
    for outcome in &report.outcomes {
        let ms = outcome.candidate_median_ms.unwrap_or(f64::NAN);
        let rate = if ms.is_finite() && ms > 0.0 {
            format!("{:.3}", tflops(m, n, k, ms))
        } else {
            "-".to_string()
        };
        println!(
            "candidate\t{m}x{n}x{k}\t{:?}\t{:?}\t{ms:.4}\t{rate}\tqualified {}\t{}",
            outcome.candidate,
            outcome.decision,
            outcome.qualified,
            outcome.failure.as_deref().unwrap_or(""),
        );
    }
}

fn main() {
    env_logger::init();
    let sizes: Vec<usize> = std::env::args()
        .skip(1)
        .map(|arg| arg.parse().expect("square size"))
        .collect();
    assert!(
        !sizes.is_empty(),
        "pass one or more square sizes, e.g. 4096"
    );
    for size in sizes {
        let (m, n, k) = (size, size, size);
        let mut graph = Graph::new();
        let a = graph.input("a", &[m, k]);
        let b = graph.parameter("b", &[k, n]);
        let outputs = (0..30).map(|_| graph.matmul(a, b)).collect();
        graph.set_outputs(outputs);
        let mut config = SessionConfig::inference_from_env();
        config.optimize.mode = meganeura::optimize::OptimizeMode::Off;
        config.runtime.serial_dispatch = true;
        config.tune = false;
        let (mut session, _) = meganeura::build(&graph, config);
        println!("device\t{}", session.device_information().device_name);
        session.set_input("a", &fill(m * k, 7));
        session.set_parameter("b", &fill(k * n, 1));
        let report = session
            .tune_with(TuneOptions::default())
            .expect("tune options");
        for (name, stats) in session.get_pipeline_statistics() {
            for exec in stats {
                print!("stat\t{name}");
                for s in exec.statistics {
                    print!("\t{}={}", s.name, s.value);
                }
                println!();
            }
        }
        report_race(&report, m, n, k);
        measure(&mut session, m, n, k);
    }
}
