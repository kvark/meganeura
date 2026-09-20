//! Small-region experiment: retain egglog alternatives through kernel tuning.
//! Usage: egglog_search [M K N] [reverse|forward] [split] [select]
use meganeura::{Graph, Session, SessionConfig, optimize::search};
use std::time::Instant;

fn sample(session: &mut Session) -> f64 {
    let start = Instant::now();
    for _ in 0..20 {
        session.step();
        session.wait();
    }
    start.elapsed().as_secs_f64() * 50.0
}

fn median(samples: &[f64]) -> f64 {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    sorted[sorted.len() / 2]
}

fn qualify(session: &mut Session, reference: &[f64]) -> Result<f64, String> {
    session.step();
    session.wait();
    let out = session.read_output(reference.len());
    let mut error = 0.0_f64;
    for (&actual, &expected) in out.iter().zip(reference) {
        let diff = (f64::from(actual) - expected).abs();
        if !actual.is_finite() || diff >= 2e-5 + 2e-4 * expected.abs() {
            return Err(format!("{actual} != {expected}"));
        }
        error = error.max(diff);
    }
    Ok(error)
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
    let space = search::candidates(&graph, 8).unwrap();
    assert!(!space.truncated);
    let mut candidates = space.candidates;
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
    let split_search = args.get(4).is_some_and(|s| s == "split");
    let select = args.get(5).is_some_and(|s| s == "select");
    assert!(!select || split_search);
    let tiles: &[u32] = if split_search { &[32, 64] } else { &[64] };
    let splits: &[u32] = if split_search { &[1, 2, 4, 8] } else { &[1] };
    let gpu = std::sync::Arc::new(meganeura::runtime::init_gpu_context().unwrap());
    let mut trials = Vec::new();
    let mut programs = Vec::new();
    for candidate in candidates {
        for (tile_size, splits, k_stage, interleave_columns) in tiles.iter().flat_map(|&tile| {
            splits.iter().flat_map(move |&splits| {
                [8, 16, 32]
                    .into_iter()
                    .flat_map(move |k| [false, true].map(move |i| (tile, splits, k, i)))
            })
        }) {
            let mut cfg = SessionConfig::inference_from_env();
            cfg.gpu = Some(gpu.clone());
            cfg.optimize.mode = meganeura::OptimizeMode::Off;
            cfg.options.fuse_dispatches = false;
            cfg.options.knobs.matmul_k_stage = k_stage;
            cfg.options.knobs.matmul_interleave_columns = interleave_columns;
            cfg.runtime.coop = meganeura::CoopPolicy::NativeF32;
            cfg.tune = false;
            let start = Instant::now();
            let mut session = if split_search {
                use meganeura::compile::{self, Kernel, ShaderEntry};
                let mut plan = compile::compile_with(&candidate.graph, &cfg.options);
                let shape = meganeura::codegen::ScalarMatmulShape {
                    tile_size,
                    k_stage,
                    interleave_columns,
                };
                if splits > 1 {
                    let Some(index) = plan
                        .dispatches
                        .iter()
                        .position(|d| d.shader == ShaderEntry::MatMul)
                    else {
                        continue;
                    };
                    if plan
                        .split_matmul(index, shape, splits, 64 * 1024 * 1024)
                        .is_err()
                    {
                        continue;
                    }
                } else {
                    for d in &mut plan.dispatches {
                        if matches!(d.shader, ShaderEntry::MatMul | ShaderEntry::FusedMatMulAdd) {
                            d.kernel = Kernel::ScalarMatmul(shape);
                            d.workgroups = [
                                (n as u32).div_ceil(tile_size),
                                (m as u32).div_ceil(tile_size),
                                1,
                            ];
                        }
                    }
                }
                if select {
                    programs.push(search::measure::Program {
                        description: format!("{}; tile={tile_size}, k={k_stage}, interleave={interleave_columns}, splits={splits}", candidate.expression),
                        plan,
                    });
                    continue;
                }
                Session::with_context_opts(plan, gpu.clone(), cfg.runtime)
            } else {
                meganeura::build(&candidate.graph, cfg).0
            };
            let build_ms = start.elapsed().as_secs_f64() * 1000.0;
            session.set_input("a", &a);
            session.set_input("c", &c);
            session.set_parameter("b", &b);
            let error = qualify(&mut session, &reference).unwrap();
            let row = serde_json::json!({
                "expression": candidate.expression,
                "k_stage": k_stage, "interleave_columns": interleave_columns,
                "tile_size": tile_size, "splits": splits,
                "build_ms": build_ms,
                "dispatches": session.plan().dispatches.len(),
                "max_abs_error": error,
            });
            trials.push((session, row, Vec::new()));
        }
    }
    if select {
        let (mut selected, report) = search::measure::select(
            programs,
            gpu.clone(),
            meganeura::SessionOptions {
                coop: meganeura::CoopPolicy::NativeF32,
                ..Default::default()
            },
            search::measure::Options {
                warmup_runs: 3,
                tuning: meganeura::TuneOptions {
                    max_time: std::time::Duration::from_millis(200),
                    min_improvement: 0.02,
                    sample_pairs: 12,
                    ..Default::default()
                },
                max_time: std::time::Duration::from_secs(60),
                max_programs: 64,
                max_plan_bytes: 256 * 1024 * 1024,
            },
            |session| {
                session.set_input("a", &a);
                session.set_input("c", &c);
                session.set_parameter("b", &b);
                Ok(())
            },
            |session| qualify(session, &reference).map(|_| ()),
        )
        .unwrap();
        let samples: Vec<_> = (0..12).map(|_| sample(&mut selected)).skip(3).collect();
        let error = qualify(&mut selected, &reference).unwrap();
        println!(
            "{}",
            serde_json::json!({
                "shape": [m,k,n], "device": gpu.device_information().device_name,
                "extraction_ms": extraction_ms, "report": report,
                "held_out_ms": samples, "median_ms": median(&samples), "max_abs_error": error,
            })
        );
        return;
    }
    for repeat in 0..12 {
        let len = trials.len();
        for i in 0..len {
            let index = if repeat % 2 == 0 {
                (i + repeat) % len
            } else {
                (len - 1 - i + repeat) % len
            };
            let (session, _, samples) = &mut trials[index];
            let time = sample(session);
            if repeat >= 3 {
                samples.push(time);
            }
        }
    }
    let rows: Vec<_> = trials
        .into_iter()
        .map(|(_, mut row, samples)| {
            row["median_ms"] = median(&samples).into();
            row["samples_ms"] = samples.into();
            row
        })
        .collect();
    println!(
        "{}",
        serde_json::json!({
            "shape": [m,k,n], "extraction_ms": extraction_ms,
            "device": gpu.device_information().device_name,
            "candidates": rows,
        })
    );
}
