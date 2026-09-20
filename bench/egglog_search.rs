//! Small-region experiment: retain egglog alternatives through kernel tuning.
//! Usage: egglog_search [M K N] [reverse]
use meganeura::{Graph, Session, SessionConfig, optimize::search, tune::TuneOptions};
use std::time::{Duration, Instant};

fn samples(session: &mut Session) -> Vec<f64> {
    (0..9)
        .map(|_| {
            let start = Instant::now();
            for _ in 0..20 {
                session.step();
                session.wait();
            }
            start.elapsed().as_secs_f64() * 50.0
        })
        .collect()
}

fn median(samples: &[f64]) -> f64 {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[sorted.len() / 2]
}

fn qualify(session: &mut Session, reference: &[f64]) -> f64 {
    session.step();
    session.wait();
    let out = session.read_output(reference.len());
    let mut error = 0.0_f64;
    for (&actual, &expected) in out.iter().zip(reference) {
        let diff = (f64::from(actual) - expected).abs();
        assert!(
            actual.is_finite() && diff < 2e-5 + 2e-4 * expected.abs(),
            "{actual} != {expected}"
        );
        error = error.max(diff);
    }
    error
}

fn main() {
    env_logger::init();
    let args: Vec<_> = std::env::args().skip(1).collect();
    let dims: Vec<usize> = args.iter().take(3).map(|s| s.parse().unwrap()).collect();
    let [m, k, n] = if dims.is_empty() {
        [50, 512, 512]
    } else {
        dims.try_into().unwrap()
    };
    let mut graph = Graph::new();
    let a = graph.input("a", &[m, k]);
    let b = graph.parameter("b", &[k, n]);
    let c = graph.input("c", &[m, n]);
    let mm = graph.matmul(a, b);
    let output = graph.add(mm, c);
    graph.set_outputs(vec![output]);
    let started = Instant::now();
    let mut candidates = search::candidates(&graph, 8).unwrap();
    let extraction_ms = started.elapsed().as_secs_f64() * 1000.0;
    if args.get(3).is_some_and(|s| s == "reverse") {
        candidates.reverse();
    }
    let pattern = |len, salt| {
        (0..len)
            .map(|i| (((i * 17 + salt) % 257) as f32 - 128.0) / 512.0)
            .collect::<Vec<_>>()
    };
    let a = pattern(m * k, 7);
    let b = pattern(k * n, 31);
    let c = pattern(m * n, 97);
    let reference: Vec<f64> = (0..m * n)
        .map(|i| {
            let (row, col) = (i / n, i % n);
            f64::from(c[i])
                + (0..k)
                    .map(|j| f64::from(a[row * k + j]) * f64::from(b[j * n + col]))
                    .sum::<f64>()
        })
        .collect();
    let mut gpu = None;
    let mut rows = Vec::new();
    for candidate in candidates {
        let mut cfg = SessionConfig::inference_from_env();
        cfg.gpu = gpu.clone();
        cfg.optimize.mode = meganeura::OptimizeMode::Off;
        cfg.options.fuse_dispatches = false;
        cfg.runtime.coop = meganeura::CoopPolicy::NativeF32;
        cfg.tune = false;
        let start = Instant::now();
        let mut session = meganeura::build(&candidate.graph, cfg).0;
        let build_ms = start.elapsed().as_secs_f64() * 1000.0;
        gpu = Some(session.context());
        session.set_input("a", &a);
        session.set_input("c", &c);
        session.set_parameter("b", &b);
        let initial_error = qualify(&mut session, &reference);
        let initial = samples(&mut session);
        let tuning = session
            .tune_with(TuneOptions {
                max_time: Duration::from_secs(15),
                max_scratch_bytes: 128 * 1024 * 1024,
                ..Default::default()
            })
            .unwrap();
        let tuned_error = qualify(&mut session, &reference);
        let tuned = samples(&mut session);
        rows.push(serde_json::json!({
            "expression": candidate.expression,
            "build_ms": build_ms,
            "dispatches": session.plan().dispatches.len(),
            "initial_ms": initial, "initial_median_ms": median(&initial),
            "tuned_ms": tuned, "tuned_median_ms": median(&tuned),
            "initial_max_abs_error": initial_error, "tuned_max_abs_error": tuned_error,
            "tuning": tuning,
        }));
    }
    println!(
        "{}",
        serde_json::json!({
            "shape": [m,k,n], "extraction_ms": extraction_ms,
            "device": gpu.unwrap().device_information().device_name,
            "candidates": rows,
        })
    );
}
