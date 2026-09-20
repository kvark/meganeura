//! A stateful attention region, not an end-to-end language-model benchmark.
use meganeura::{Graph, Session, compile, optimize::search::measure, tune};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};

pub fn run(args: &[String]) {
    let block = args.first().map_or(16, |s| s.parse::<usize>().unwrap());
    let capacity = args.get(1).map_or(2048, |s| s.parse::<usize>().unwrap());
    assert!(block > 0 && block <= capacity && capacity <= 8192);
    let reverse = args.get(2).is_some_and(|s| s == "reverse");
    let (heads, kv_heads, dim) = (12, 4, 64);
    let position = capacity - block;
    let mut graph = Graph::new();
    let q = graph.input("q", &[block, heads * dim]);
    let q = graph.scale(q, 0.5);
    let k = graph.parameter("k", &[capacity, kv_heads * dim]);
    let v = graph.parameter("v", &[capacity, kv_heads * dim]);
    let new_k = graph.input("new_k", &[block, kv_heads * dim]);
    let new_v = graph.input("new_v", &[block, kv_heads * dim]);
    let pos = graph.input_u32("position", &[1]);
    let valid = graph.input_u32("valid", &[1]);
    let k = graph.cache_write_prefix(new_k, k, pos, valid);
    let v = graph.cache_write_prefix(new_v, v, pos, valid);
    let y = graph.cached_block_attention(
        q,
        k,
        v,
        pos,
        valid,
        heads as u32,
        kv_heads as u32,
        dim as u32,
        0,
    );
    let y = graph.scale(y, 0.75);
    graph.set_outputs(vec![y]);
    let plan = compile::compile(&graph);
    let first = plan
        .dispatches
        .iter()
        .position(|d| {
            matches!(
                d.shader,
                compile::ShaderEntry::CachedBlockAttention
                    | compile::ShaderEntry::CachedBlockAttentionSplit
            )
        })
        .unwrap();
    let pattern = |len, salt| {
        (0..len)
            .map(|i| (((i * 17 + salt) % 257) as f32 - 128.0) / 512.0)
            .collect::<Vec<_>>()
    };
    let queries = pattern(block * heads * dim, 7);
    let initial: Vec<_> = [31, 53]
        .map(|salt| pattern(capacity * kv_heads * dim, salt))
        .into();
    let updates: Vec<_> = [97, 131]
        .map(|salt| pattern(block * kv_heads * dim, salt))
        .into();
    let mut changed = initial.clone();
    for (cache, update) in changed.iter_mut().zip(&updates) {
        cache[position * kv_heads * dim..].copy_from_slice(update);
    }
    let mut reference = Vec::new();
    for row in 0..block {
        for head in 0..heads {
            let kv_head = head / (heads / kv_heads);
            let q = &queries[(row * heads + head) * dim..][..dim];
            let scores: Vec<f64> = (0..position + row + 1)
                .map(|token| {
                    (0..dim)
                        .map(|col| {
                            f64::from(q[col])
                                * 0.5
                                * f64::from(changed[0][(token * kv_heads + kv_head) * dim + col])
                        })
                        .sum::<f64>()
                        / (dim as f64).sqrt()
                })
                .collect();
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<_> = scores.iter().map(|s| (s - max).exp()).collect();
            let sum: f64 = weights.iter().sum();
            for col in 0..dim {
                reference.push(
                    weights
                        .iter()
                        .enumerate()
                        .map(|(token, w)| {
                            w * f64::from(changed[1][(token * kv_heads + kv_head) * dim + col])
                        })
                        .sum::<f64>()
                        / sum
                        * 0.75,
                );
            }
        }
    }
    let initialize = |s: &mut Session| {
        s.set_input("q", &queries);
        s.set_input("new_k", &updates[0]);
        s.set_input("new_v", &updates[1]);
        s.set_input_u32("position", &[position as u32]);
        s.set_input_u32("valid", &[block as u32]);
        s.set_parameter("k", &initial[0]);
        s.set_parameter("v", &initial[1]);
    };
    let qualify = |s: &mut Session| -> Result<(), String> {
        s.step();
        s.wait();
        if s.read_output(reference.len())
            .iter()
            .zip(&reference)
            .any(|(&a, &b)| !a.is_finite() || (f64::from(a) - b).abs() > 2e-5 + 2e-4 * b.abs())
        {
            return Err("complete output disagrees with independent f64 reference".into());
        }
        if s.read_params(&["k", "v"]) != changed {
            return Err("cache write mismatch".into());
        }
        Ok(())
    };
    let mut programs = vec![measure::Program {
        description: "compiler default".into(),
        plan: plan.clone(),
    }];
    let original_splits = plan.dispatches[first].params[6].max(1);
    let mut splits: Vec<_> = [1, 2, 4, 8, 16]
        .into_iter()
        .filter(|&n| n != original_splits)
        .collect();
    if reverse {
        splits.reverse();
    }
    for splits in splits {
        let mut plan = plan.clone();
        plan.set_attention_splits(&[(first, splits)], 64 << 20)
            .unwrap();
        programs.push(measure::Program {
            description: format!("{splits} attention splits"),
            plan,
        });
    }
    let gpu =
        Arc::new(meganeura::init_gpu_context_with(meganeura::GpuOptions::from_env()).unwrap());
    let runtime = meganeura::SessionOptions::from_env();
    let policy = tune::TuneOptions {
        max_time: Duration::from_secs(2),
        sample_pairs: 12,
        warmup_runs: 4,
        ..Default::default()
    };
    let (mut selected, report) = measure::select(
        programs,
        gpu.clone(),
        runtime.clone(),
        measure::Options {
            tuning: policy.clone(),
            warmup_runs: 16,
            max_time: Duration::from_secs(30),
            max_programs: 6,
            max_plan_bytes: 256 << 20,
        },
        |s, _| {
            initialize(s);
            Ok(())
        },
        qualify,
    )
    .unwrap();
    assert!(!report.truncated);
    assert!(report.trials.iter().all(|t| t.outcome.qualified));
    assert_eq!(selected.read_params(&["k", "v"]), initial);

    // Separate held-out comparison against the existing position-balanced tuner.
    let mut baseline = Session::with_context_opts(plan, gpu, runtime);
    initialize(&mut baseline);
    let baseline_tuning = baseline
        .tune_with(tune::TuneOptions {
            scope: tune::TuneScope::Attention,
            ..policy
        })
        .unwrap();
    qualify(&mut baseline).unwrap();
    qualify(&mut selected).unwrap();
    let sample = |s: &mut Session| {
        let start = Instant::now();
        s.step();
        s.wait();
        start.elapsed().as_secs_f64() * 1000.0
    };
    for _ in 0..30 {
        sample(&mut baseline);
        sample(&mut selected);
    }
    let (mut before, mut after) = (Vec::new(), Vec::new());
    for i in 0..40 {
        if i % 2 == 0 {
            before.push(sample(&mut baseline));
            after.push(sample(&mut selected));
        } else {
            after.push(sample(&mut selected));
            before.push(sample(&mut baseline));
        }
    }
    qualify(&mut baseline).unwrap();
    qualify(&mut selected).unwrap();
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({
            "device": selected.context().device_information().device_name,
            "block": block, "capacity": capacity, "position": position, "reverse": reverse,
            "heads": heads, "kv_heads": kv_heads, "head_dim": dim,
            "precision": "f32", "search": report, "legacy_tuning": baseline_tuning,
            "baseline_ms": before, "selected_ms": after,
            "selected_dispatches": selected.plan().dispatches.len(),
            "baseline_dispatches": baseline.plan().dispatches.len(),
        }))
        .unwrap()
    );
}
