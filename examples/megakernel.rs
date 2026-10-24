//! Persistent worker pool versus separate dispatches for a residual MLP.
//! cargo run --release --example megakernel -- [workers=64] [jobs=1024]
//!     [width=64] [layers=2] [samples=20]
//! Outputs JSON with full-arena f64 checks, worker accounting, and paired times.

use meganeura::megakernel::{Execution, Executor, Instruction, Job, Options};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};

fn workload(count: usize, width: u32, layers: u32) -> (Vec<Job>, Vec<f32>, Vec<Vec<f64>>) {
    let n = width as usize;
    let weights: Vec<f32> = (0..2 * n * n)
        .map(|i| ((i * 17 % 61) as f32 - 30.0) / (30.0 * (n as f32).sqrt()))
        .collect();
    let mut jobs = Vec::new();
    let mut expected = Vec::new();
    for index in 0..count {
        let mut data = vec![f32::NAN; n * (1 + 4 * layers as usize)];
        for (i, x) in data[..n].iter_mut().enumerate() {
            *x = ((i * 13 + index * 7) % 67) as f32 / 33.0 - 1.0;
        }
        let mut want: Vec<f64> = data.iter().map(|&x| f64::from(x)).collect();
        let mut instructions = Vec::new();
        let mut src = 0;
        for layer in 0..layers {
            let norm = width * (1 + 4 * layer);
            let hidden = norm + width;
            let projected = hidden + width;
            let dst = projected + width;
            instructions.extend([
                Instruction::RmsNorm {
                    src,
                    dst: norm,
                    len: width,
                    epsilon: 1e-5,
                },
                Instruction::MatVec {
                    src: norm,
                    weights: 0,
                    dst: hidden,
                    rows: width,
                    columns: width,
                },
                Instruction::Relu {
                    src: hidden,
                    dst: hidden,
                    len: width,
                },
                Instruction::MatVec {
                    src: hidden,
                    weights: width * width,
                    dst: projected,
                    rows: width,
                    columns: width,
                },
                Instruction::Add {
                    a: src,
                    b: projected,
                    dst,
                    len: width,
                },
            ]);
            let rms = (want[src as usize..][..n].iter().map(|x| x * x).sum::<f64>() / n as f64
                + f64::from(1e-5f32))
            .sqrt();
            for i in 0..n {
                want[norm as usize + i] = want[src as usize + i] / rms;
            }
            for row in 0..n {
                want[hidden as usize + row] = (0..n)
                    .map(|k| f64::from(weights[row * n + k]) * want[norm as usize + k])
                    .sum::<f64>()
                    .max(0.0);
            }
            for row in 0..n {
                want[projected as usize + row] = (0..n)
                    .map(|k| f64::from(weights[n * n + row * n + k]) * want[hidden as usize + k])
                    .sum();
                want[dst as usize + row] =
                    want[src as usize + row] + want[projected as usize + row];
            }
            src = dst;
        }
        jobs.push(Job { data, instructions });
        expected.push(want);
    }
    (jobs, weights, expected)
}

fn check(executor: &Executor, expected: &[Vec<f64>]) -> f64 {
    let mut error = 0.0f64;
    for (j, want) in expected.iter().enumerate() {
        for (i, (got, &want)) in executor.read_job(j).into_iter().zip(want).enumerate() {
            let delta = (f64::from(got) - want).abs();
            assert!(
                got.is_finite() && delta <= 2e-4 * (1.0 + want.abs()),
                "job {j}, value {i}: {got} != {want}"
            );
            error = error.max(delta);
        }
    }
    assert_eq!(
        executor.statistics().completed_per_job,
        vec![1; expected.len()]
    );
    error
}

fn median(samples: &[f64]) -> f64 {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    (sorted[(sorted.len() - 1) / 2] + sorted[sorted.len() / 2]) / 2.0
}

fn main() {
    env_logger::init();
    let args: Vec<u32> = std::env::args()
        .skip(1)
        .map(|s| s.parse().expect("integer argument"))
        .collect();
    assert!(
        args.len() <= 5,
        "expected: workers jobs width layers samples"
    );
    let get = |index, default| args.get(index).copied().unwrap_or(default);
    let (workers, count, width, layers, samples) =
        (get(0, 64), get(1, 1024), get(2, 64), get(3, 2), get(4, 20));
    assert!(
        workers > 0
            && workers <= 65535
            && count <= 65535
            && width > 0
            && width <= 4096
            && layers > 0
            && layers <= 64
            && samples > 0
    );
    let gpu = Arc::new(
        meganeura::init_gpu_context_with(meganeura::GpuOptions {
            timing: true,
            ..meganeura::GpuOptions::from_env()
        })
        .unwrap(),
    );
    let (jobs, weights, expected) = workload(count as usize, width, layers);
    let mut executor = Executor::new(
        gpu.clone(),
        &jobs,
        &weights,
        Options {
            workers,
            gpu_timing: true,
        },
    )
    .unwrap();
    let modes = [Execution::Separate, Execution::Persistent];
    for mode in modes {
        executor.run(mode);
        check(&executor, &expected);
    }
    let warmup = Instant::now();
    while warmup.elapsed() < Duration::from_millis(500) {
        for mode in modes {
            executor.run(mode);
        }
    }
    let mut wall_samples = [Vec::new(), Vec::new()];
    let mut gpu_samples = [Vec::new(), Vec::new()];
    for round in 0..samples {
        for index in if round % 2 == 0 { [0, 1] } else { [1, 0] } {
            let measurement = executor.run(modes[index]);
            wall_samples[index].push(measurement.wall.as_secs_f64() * 1e6);
            gpu_samples[index].push(measurement.compute.unwrap().as_secs_f64() * 1e6);
        }
    }
    executor.run(Execution::Separate);
    let separate_error = check(&executor, &expected);
    executor.run(Execution::Persistent);
    let persistent_error = check(&executor, &expected);
    let stats = executor.statistics();
    assert_eq!(stats.per_worker.iter().map(|v| v[0]).sum::<u32>(), count);
    assert_eq!(
        stats.per_worker.iter().map(|v| v[1]).sum::<u32>(),
        count * layers * 5
    );
    let wall_us = wall_samples.each_ref().map(|s| median(s));
    let compute_us = gpu_samples.each_ref().map(|s| median(s));
    println!(
        "{}",
        serde_json::json!({
            "device": gpu.device_information().device_name,
            "workers": workers, "jobs": count, "width": width, "layers": layers,
        "warmup_ms": 500,
            "mode_order": ["separate", "persistent"],
            "dispatches": [if count == 0 { 0 } else { layers * 5 }, 1],
            "values_checked_per_mode": expected.iter().map(Vec::len).sum::<usize>(),
            "max_abs_error": [separate_error, persistent_error],
            "worker_jobs": stats.per_worker.iter().map(|v| v[0]).collect::<Vec<_>>(),
            "wall_median_us": wall_us, "compute_median_us": compute_us,
            "compute_speedup": compute_us[0] / compute_us[1],
            "wall_samples_us": wall_samples, "compute_samples_us": gpu_samples,
        })
    );
}
