//! Dense GEMV layout diagnostic, with finite independent products.
//!
//! MEGANEURA_DEVICE_ID=... cargo run --release --example bench_gemv_layout -- K N nn|bt
//! Every output is checked against an f64 dot product before and after timing.
//! The bank contains at least 64 MiB of distinct weights, reused across steps;
//! reported bandwidth is effective weight bytes / completed wall time, not a
//! DRAM counter. Construction, uploads, validation and readback are not timed.

use meganeura::{
    CompileOptions, CoopPolicy, GemvReduction, GemvShape, Graph, Session, SessionOptions,
};
use serde_json::json;
use std::{
    sync::Arc,
    time::{Duration, Instant},
};

fn values(n: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state >> 8) as f32 / 16777216.0 - 0.5) * scale
        })
        .collect()
}

fn check(session: &mut Session, expected: &[Vec<f64>]) -> (f64, f64) {
    session.step();
    session.wait();
    let (mut worst, mut squared, mut norm) = (0.0_f64, 0.0, 0.0);
    for (index, output) in expected.iter().enumerate() {
        let mut got = vec![f32::NAN; output.len()];
        session.read_output_by_index(index, &mut got);
        for (column, (&a, &b)) in got.iter().zip(output).enumerate() {
            let error = (f64::from(a) - b).abs();
            assert!(
                a.is_finite() && b.is_finite() && error <= 2e-5 + 2e-4 * b.abs(),
                "output {index}, column {column}: {a} vs {b}"
            );
            worst = worst.max(error);
            squared += error * error;
            norm += b * b;
        }
    }
    assert!(norm > 1e-6, "degenerate reference signal");
    let relative_l2 = (squared / norm).sqrt();
    assert!(relative_l2 <= 2e-4);
    (worst, relative_l2)
}

fn batch(session: &mut Session) -> f64 {
    let start = Instant::now();
    for _ in 0..10 {
        session.step();
    }
    session.wait();
    start.elapsed().as_secs_f64() / 10.0
}

fn main() {
    env_logger::init();
    let args: Vec<_> = std::env::args().skip(1).collect();
    assert_eq!(args.len(), 3, "expected K N nn|bt");
    let k: usize = args[0].parse().unwrap();
    let n: usize = args[1].parse().unwrap();
    let bt = match args[2].as_str() {
        "nn" => false,
        "bt" => true,
        _ => panic!("nn|bt"),
    };
    assert!(k > 0 && n > 0 && k.is_multiple_of(4) && n.is_multiple_of(4));
    let copies = (64 * 1024 * 1024usize).div_ceil(k * n * 4);
    assert!(
        copies <= 256,
        "use larger matrices for the streaming diagnostic"
    );
    let x = values(k, 17, 1.0);
    let mut graph = Graph::new();
    let input = graph.input("x", &[1, k]);
    let mut outputs = Vec::new();
    let mut weights = Vec::new();
    let mut expected = Vec::new();
    for i in 0..copies {
        let w = values(k * n, i as u32 + 41, 1.0 / (k as f32).sqrt());
        let mut y = vec![0.0_f64; n];
        for row in 0..k {
            for col in 0..n {
                y[col] += f64::from(x[row])
                    * f64::from(w[if bt { col * k + row } else { row * n + col }]);
            }
        }
        let dimensions = if bt { [n, k] } else { [k, n] };
        let parameter = graph.parameter(&format!("w{i}"), &dimensions);
        outputs.push(if bt {
            graph.matmul_bt(input, parameter)
        } else {
            graph.matmul(input, parameter)
        });
        weights.push(w);
        expected.push(y);
    }
    graph.set_outputs(outputs);
    let options = meganeura::GpuOptions::from_env();
    let requested_device_id = options.device_id;
    let gpu = Arc::new(meganeura::init_gpu_context_with(options).unwrap());
    eprintln!("GPU: {}", gpu.device_information().device_name);
    let initial = meganeura::compile::compile(&graph).dispatches[0]
        .gemv_shape()
        .unwrap();
    let geometries = if bt {
        vec![
            initial,
            GemvShape {
                threads: 32,
                reduction: GemvReduction::Subgroup,
                bt_rows: 4,
                column_groups: 1,
            },
            GemvShape {
                threads: 64,
                reduction: GemvReduction::Subgroup,
                bt_rows: 2,
                column_groups: 1,
            },
        ]
    } else {
        vec![
            initial,
            GemvShape {
                threads: 32,
                reduction: GemvReduction::Tree,
                bt_rows: 1,
                column_groups: 4,
            },
            GemvShape {
                threads: 32,
                reduction: GemvReduction::Tree,
                bt_rows: 1,
                column_groups: 8,
            },
        ]
    };
    let mut variants = Vec::new();
    for shape in geometries {
        let plan = meganeura::compile::compile_with(
            &graph,
            &CompileOptions {
                gemv_shape: Some(shape),
                ..Default::default()
            },
        );
        assert_eq!(plan.dispatches.len(), copies);
        assert!(
            plan.dispatches
                .iter()
                .all(|d| d.gemv_shape() == Some(shape))
        );
        let mut session = Session::with_context_opts(
            plan,
            Arc::clone(&gpu),
            SessionOptions {
                coop: CoopPolicy::Disabled,
                ..Default::default()
            },
        );
        session.set_input("x", &x);
        for (i, w) in weights.iter().enumerate() {
            session.set_parameter(&format!("w{i}"), w);
        }
        let errors = check(&mut session, &expected);
        eprintln!("qualified {shape:?}: {errors:?}");
        variants.push((shape, session, Vec::new(), errors));
    }
    // Warm all variants at the same point in construction, then rotate their
    // position in each round and reverse alternating rounds to limit drift.
    for (_, session, _, _) in &mut variants {
        let start = Instant::now();
        while start.elapsed() < Duration::from_secs(10) {
            batch(session);
        }
    }
    let count = variants.len();
    let mut order = Vec::new();
    for round in 0..20 {
        let mut indices: Vec<_> = (0..count).map(|i| (i + round) % count).collect();
        if round % 2 == 1 {
            indices.reverse();
        }
        for &index in &indices {
            let (_, session, samples, _) = &mut variants[index];
            samples.push(batch(session));
        }
        order.push(indices);
    }
    let records: Vec<_> = variants.iter_mut().map(|(shape, session, samples, preflight)| {
        let after = check(session, &expected);
        let mut sorted = samples.clone();
        sorted.sort_by(f64::total_cmp);
        let median = (sorted[9] + sorted[10]) / 2.0;
        json!({"shape": shape, "seconds_per_step": samples, "median_us_per_product": median * 1e6 / copies as f64,
            "effective_weight_gb_s": (copies * k * n * 4) as f64 / median / 1e9,
            "preflight_max_abs_l2": preflight, "postflight_max_abs_l2": after})
    }).collect();
    let device = gpu.device_information();
    println!(
        "{}",
        serde_json::to_string_pretty(&json!({
            "protocol": "finite-independent-gemv-v1", "k": k, "n": n, "bt": bt,
            "copies": copies, "weight_bytes": copies * k * n * 4,
            "device": device.device_name, "driver": device.driver_info,
            "requested_device_id": requested_device_id,
            "warmup_seconds_per_variant": 10, "steps_per_sample": 10,
            "order": order, "variants": records,
        }))
        .unwrap()
    );
}
