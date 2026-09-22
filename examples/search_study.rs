//! Paired comparison of one-graph tuning and measured tile equalities.
//!
//! ```text
//! MEGANEURA_DEVICE_ID=12036 MEGANEURA_DISABLE_COOP=1 MEGANEURA_GPU_TIMING=1 \
//!   cargo run --release --example search_study
//! ```
//!
//! `greedy+tuner` extracts one graph by tensor traffic, then runs the private
//! kernel tuner. `egraph` keeps tile and split-K equalities and times whole
//! steps. Both use the same inputs, device, and profiled-median readout.
use std::time::{Duration, Instant};

use meganeura::train::{BuildSearchOptions, Mode, build, build_measured};
use meganeura::{CoopPolicy, Graph, OptimizeConfig, Session, SessionConfig, TuneOptions};

fn shapes() -> Vec<(usize, usize, usize, bool)> {
    vec![
        (50, 4096, 720, false),
        (50, 960, 720, false),
        (50, 720, 960, false),
        (50, 960, 720, true),
    ]
}

fn operands(m: usize, n: usize, k: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 13) as f32 - 6.0) * 0.01).collect();
    let b: Vec<f32> = (0..k * n).map(|i| ((i % 11) as f32 - 5.0) * 0.02).collect();
    let d: Vec<f32> = (0..m * n).map(|i| ((i % 7) as f32 - 3.0) * 0.03).collect();
    (a, b, d)
}

fn reference(
    m: usize,
    n: usize,
    k: usize,
    fuse: bool,
    a: &[f32],
    b: &[f32],
    d: &[f32],
) -> Vec<f32> {
    let mut out = vec![0.0; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut acc = 0.0f32;
            for t in 0..k {
                acc += a[row * k + t] * b[t * n + col];
            }
            if fuse {
                acc += d[row * n + col];
            }
            out[row * n + col] = acc;
        }
    }
    out
}

fn graph_for(m: usize, n: usize, k: usize, fuse: bool) -> Graph {
    let mut graph = Graph::new();
    let a = graph.input("a", &[m, k]);
    let b = graph.parameter("b", &[k, n]);
    let product = graph.matmul(a, b);
    let y = if fuse {
        let d = graph.input("d", &[m, n]);
        graph.add(product, d)
    } else {
        product
    };
    graph.set_outputs(vec![y]);
    graph
}

fn load(session: &mut Session, fuse: bool, a: &[f32], b: &[f32], d: &[f32]) {
    session.set_input("a", a);
    session.set_parameter("b", b);
    if fuse {
        session.set_input("d", d);
    }
}

fn check(session: &Session, expected: &[f32]) -> Result<(), String> {
    let got = session.read_output(expected.len());
    let mut max_abs = 0.0f32;
    for (got, expected) in got.iter().zip(expected) {
        max_abs = max_abs.max((got - expected).abs());
    }
    if max_abs > 1e-3 {
        Err(format!("max_abs {max_abs:.3e}"))
    } else {
        Ok(())
    }
}

fn profile(session: &mut Session) -> f64 {
    for _ in 0..12 {
        session.step();
    }
    session.wait();
    session.set_profiling(true);
    let mut samples = Vec::with_capacity(8);
    for _ in 0..8 {
        session.step();
        session.wait();
        let micros = session
            .profiled_dispatch_timings()
            .iter()
            .map(|(_, _, duration)| duration.as_secs_f64())
            .sum::<f64>()
            * 1e6;
        samples.push(micros);
    }
    samples.sort_by(|left, right| left.partial_cmp(right).unwrap());
    samples[samples.len() / 2]
}

fn describe(session: &Session) -> String {
    session
        .plan()
        .dispatches
        .iter()
        .map(|dispatch| {
            format!(
                "{:?} {:?} wg={:?} locked={}",
                dispatch.shader, dispatch.kernel, dispatch.workgroups, dispatch.schedule_locked
            )
        })
        .collect::<Vec<_>>()
        .join(" | ")
}

fn base_config(tune: bool) -> SessionConfig<'static> {
    let mut cfg = SessionConfig::inference_from_env();
    cfg.mode = Mode::Inference;
    cfg.optimize = OptimizeConfig::default();
    cfg.runtime.coop = CoopPolicy::Disabled;
    cfg.tune = tune;
    cfg
}

fn main() {
    for (m, n, k, fuse) in shapes() {
        let graph = graph_for(m, n, k, fuse);
        let (a, b, d) = operands(m, n, k);
        let expected = reference(m, n, k, fuse, &a, &b, &d);
        let label = format!("{m}x{n}x{k}{}", if fuse { "+add" } else { "" });

        let started = Instant::now();
        let (mut greedy, _) = build(&graph, base_config(true));
        load(&mut greedy, fuse, &a, &b, &d);
        greedy.step();
        greedy.wait();
        check(&greedy, &expected).unwrap_or_else(|error| panic!("{label} greedy {error}"));
        let greedy_us = profile(&mut greedy);
        println!(
            "greedy+tuner {label} gpu_us {greedy_us:.2} setup_s {:.1} plan {}",
            started.elapsed().as_secs_f64(),
            describe(&greedy)
        );

        let started = Instant::now();
        let (mut measured, report) = build_measured(
            &graph,
            base_config(false),
            BuildSearchOptions {
                max_graphs: 8,
                max_programs: 12,
                max_time: Duration::from_secs(40),
                warmup_runs: 1,
                tuning: TuneOptions {
                    max_time: Duration::from_secs(2),
                    sample_pairs: 4,
                    warmup_runs: 1,
                    dispatches_per_sample: 4,
                    ..TuneOptions::default()
                },
                ..BuildSearchOptions::default()
            },
            |session, _| {
                load(session, fuse, &a, &b, &d);
                Ok(())
            },
            |session| check(session, &expected),
        )
        .unwrap_or_else(|error| panic!("{label} egraph {error}"));
        let measured_us = profile(&mut measured);
        println!(
            "egraph {label} gpu_us {measured_us:.2} setup_s {:.1} selected {} trials {} plan {}",
            started.elapsed().as_secs_f64(),
            report.selected,
            report.trials.len(),
            describe(&measured)
        );
        if let Some(trial) = report.trials.get(report.selected) {
            println!("egraph-desc {label} {}", trial.description);
        }
    }
}
