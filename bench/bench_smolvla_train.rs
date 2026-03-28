/// Benchmark SmolVLA action expert forward + training step with meganeura.
///
/// Measures per-step latency for:
///   - Forward-only pass (inference session on the training graph)
///   - Full training step: forward + backward + SGD (training session)
///
/// Uses random weight initialization — no HuggingFace download required.
///
/// Usage:
///   cargo run --release --example bench_smolvla_train [-- --warmup 3 --runs 5]
use std::time::Instant;

use meganeura::{
    build_inference_session, build_session,
    models::smolvla::{self, SmolVLAConfig},
};

fn check_bench_preconditions(abort_on_warn: bool) {
    #[cfg(not(target_os = "linux"))]
    {
        eprintln!("preconditions: skipped (not Linux)");
        let _ = abort_on_warn;
        return;
    }

    #[cfg(target_os = "linux")]
    {
        let mut warnings = Vec::new();

        let on_ac = std::fs::read_dir("/sys/class/power_supply")
            .ok()
            .and_then(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .find(|e| {
                        let name = e.file_name();
                        let s = name.to_string_lossy();
                        s.starts_with("AC") || s.starts_with("ADP")
                    })
                    .map(|e| {
                        let online_path = e.path().join("online");
                        std::fs::read_to_string(online_path)
                            .ok()
                            .map(|s| s.trim() == "1")
                            .unwrap_or(false)
                    })
            })
            .unwrap_or(true);

        if !on_ac {
            warnings.push(
                "running on BATTERY — GPU clocks may be throttled, results unreliable".to_string(),
            );
        }

        let cards: Vec<_> = std::fs::read_dir("/sys/class/drm")
            .ok()
            .map(|entries| {
                entries
                    .filter_map(|e| e.ok())
                    .filter(|e| e.file_name().to_string_lossy().starts_with("card"))
                    .filter(|e| !e.file_name().to_string_lossy().contains('-'))
                    .collect()
            })
            .unwrap_or_default();

        for card in &cards {
            let dev = card.path().join("device");
            if let Some(pct) = std::fs::read_to_string(dev.join("gpu_busy_percent"))
                .ok()
                .and_then(|s| s.trim().parse::<u64>().ok())
            {
                if pct > 10 {
                    warnings.push(format!(
                        "{}: GPU {}% busy — results may be noisy",
                        card.file_name().to_string_lossy(),
                        pct
                    ));
                } else {
                    eprintln!(
                        "  {}: GPU busy {}% (ok)",
                        card.file_name().to_string_lossy(),
                        pct
                    );
                }
            }
        }

        if warnings.is_empty() {
            eprintln!("preconditions ok");
            return;
        }
        for w in &warnings {
            eprintln!("WARNING: {}", w);
        }
        if abort_on_warn {
            eprintln!("Aborting. Pass --force to run anyway.");
            std::process::exit(1);
        }
    }
}

fn main() {
    env_logger::init();

    let mut args = std::env::args().skip(1);
    let mut warmup: usize = 3;
    let mut runs: usize = 5;
    let mut force = false;
    let mut profile = false;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--warmup" => warmup = args.next().expect("--warmup value").parse().unwrap(),
            "--runs" => runs = args.next().expect("--runs value").parse().unwrap(),
            "--force" => force = true,
            "--profile" => profile = true,
            _ => {
                eprintln!("unknown arg: {}", arg);
                std::process::exit(1);
            }
        }
    }

    eprintln!("checking preconditions...");
    check_bench_preconditions(!force);

    let config = SmolVLAConfig::smolvla_base();
    let action_seq_len = config.chunk_size; // 50
    let vlm_seq_len = 16;

    eprintln!("SmolVLA action expert training benchmark (random weights)");
    eprintln!(
        "  chunk_size={}, vlm_seq_len={}, num_layers={}",
        action_seq_len, vlm_seq_len, config.expert.num_layers
    );

    // --- Build training graph ---
    eprintln!("building training graph...");
    let training_g = smolvla::build_action_expert_training(&config, action_seq_len, vlm_seq_len);

    // --- Build sessions ---
    eprintln!("compiling inference session (forward only)...");
    let mut infer_session = build_inference_session(&training_g);
    eprintln!(
        "  infer: {} buffers, {} dispatches, {} barrier groups",
        infer_session.plan().buffers.len(),
        infer_session.plan().dispatches.len(),
        infer_session.num_groups()
    );

    eprintln!("compiling training session (fwd + bwd + SGD)...");
    let mut train_session = build_session(&training_g);
    eprintln!(
        "  train: {} buffers, {} dispatches, {} barrier groups",
        train_session.plan().buffers.len(),
        train_session.plan().dispatches.len(),
        train_session.num_groups()
    );

    // --- Initialize parameters with deterministic random values ---
    eprintln!("initializing parameters...");
    for (name, buf_ref) in train_session.plan().param_buffers.clone() {
        let size_bytes = train_session.plan().buffers[buf_ref.0 as usize];
        let n = size_bytes / 4; // f32
        let data: Vec<f32> = (0..n)
            .map(|i| (i as f32 * 0.01 + 1.0).sin() * 0.1)
            .collect();
        train_session.set_parameter(&name, &data);
        infer_session.set_parameter(&name, &data);
    }

    // --- Prepare synthetic inputs ---
    let expert_hidden = config.expert.hidden_size;
    let kv_dim = config.expert.kv_dim();

    let noisy_actions: Vec<f32> = (0..action_seq_len * config.max_action_dim)
        .map(|i| (i as f32 * 0.01).sin() * 0.1)
        .collect();
    let timestep: Vec<f32> = (0..expert_hidden * 2)
        .map(|i| (i as f32 * 0.005).cos() * 0.1)
        .collect();
    let vlm_kv: Vec<f32> = (0..vlm_seq_len * kv_dim)
        .map(|i| (i as f32 * 0.002).sin() * 0.05)
        .collect();
    let target_actions = vec![0.0_f32; action_seq_len * config.max_action_dim];

    let set_inputs = |session: &mut meganeura::Session| {
        session.set_input("noisy_actions", &noisy_actions);
        session.set_input("timestep", &timestep);
        for i in 0..config.expert.num_layers {
            if i % config.expert.self_attn_every_n_layers != 0 {
                session.set_input(&format!("vlm_kv_layer_{}", i), &vlm_kv);
            }
        }
        session.set_input("target_actions", &target_actions);
    };

    // --- Warmup ---
    eprintln!("warming up ({} runs)...", warmup);
    for _ in 0..warmup {
        set_inputs(&mut infer_session);
        infer_session.step();
        infer_session.wait();

        set_inputs(&mut train_session);
        train_session.step();
        train_session.wait();
        train_session.sgd_step(1e-5);
        train_session.wait();
    }

    // --- Per-kernel profiling (--profile flag) ---
    if profile {
        // Dispatch-count breakdown by shader type (free, no pass limit)
        let count_dispatches = |plan: &meganeura::compile::ExecutionPlan| {
            let mut counts: std::collections::BTreeMap<String, usize> = Default::default();
            for d in &plan.dispatches {
                *counts.entry(format!("{:?}", d.shader)).or_default() += 1;
            }
            counts
        };

        eprintln!(
            "\n=== Forward pass dispatch counts ({} total) ===",
            infer_session.plan().dispatches.len()
        );
        for (name, count) in count_dispatches(infer_session.plan()) {
            eprintln!("  {:>30}: {}", name, count);
        }

        eprintln!(
            "\n=== Training step dispatch counts ({} total) ===",
            train_session.plan().dispatches.len()
        );
        for (name, count) in count_dispatches(train_session.plan()) {
            eprintln!("  {:>30}: {}", name, count);
        }

        // GPU timing breakdown — blade caps at 100 timestamps per submission,
        // so only the first 100 dispatches are timed.
        //
        // Blade's command encoder uses a 2-buffer ring: start() reads timestamps
        // from 2 submissions ago. So we need 3 steps to get step A's timings:
        //   step A (profiling=true) → step B (advances ring) → step C's start()
        //   reads A's data → dump_gpu_timings() shows A's per-shader breakdown.
        infer_session.set_profiling(true);
        eprintln!(
            "\n=== Forward pass GPU timings (first 100 of {} dispatches) ===",
            infer_session.plan().dispatches.len()
        );
        set_inputs(&mut infer_session);
        infer_session.step();
        infer_session.wait(); // step A — profiling run, records shader timestamps
        set_inputs(&mut infer_session);
        infer_session.step();
        infer_session.wait(); // step B — advances ring buffer
        set_inputs(&mut infer_session);
        infer_session.step(); // step C — start() reads step A's timestamps
        infer_session.dump_gpu_timings();
        infer_session.wait();

        train_session.set_profiling(true);
        eprintln!(
            "\n=== Training step GPU timings (first 100 of {} dispatches) ===",
            train_session.plan().dispatches.len()
        );
        // Skip sgd_step during profiling so the ring buffer captures
        // forward+backward timings (not SGD update timings).
        set_inputs(&mut train_session);
        train_session.step();
        train_session.wait(); // step A — profiling run
        set_inputs(&mut train_session);
        train_session.step();
        train_session.wait(); // step B — advances ring buffer
        set_inputs(&mut train_session);
        train_session.step(); // step C — start() reads step A's timestamps
        train_session.dump_gpu_timings();
        train_session.wait();

        return;
    }

    // --- Benchmark forward ---
    eprintln!("benchmarking forward ({} runs)...", runs);
    let mut fwd_latencies = Vec::new();
    for i in 0..runs {
        set_inputs(&mut infer_session);
        let t0 = Instant::now();
        infer_session.step();
        infer_session.wait();
        let elapsed = t0.elapsed().as_secs_f64();
        fwd_latencies.push(elapsed);
        eprintln!("  fwd run {}: {:.2}ms", i + 1, elapsed * 1000.0);
    }

    // --- Benchmark training step ---
    eprintln!("benchmarking training step ({} runs)...", runs);
    let mut train_latencies = Vec::new();
    for i in 0..runs {
        set_inputs(&mut train_session);
        let t0 = Instant::now();
        train_session.step();
        train_session.wait();
        train_session.sgd_step(1e-5);
        train_session.wait();
        let elapsed = t0.elapsed().as_secs_f64();
        train_latencies.push(elapsed);
        eprintln!("  train run {}: {:.2}ms", i + 1, elapsed * 1000.0);
    }

    // --- Statistics ---
    let stat = |v: &[f64]| -> (f64, f64) {
        let avg = v.iter().sum::<f64>() / v.len() as f64;
        let mut s = v.to_vec();
        s.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = s[v.len() / 2];
        (avg, median)
    };

    let (fwd_avg, fwd_median) = stat(&fwd_latencies);
    let (train_avg, train_median) = stat(&train_latencies);
    let approx_bwd_ms = (train_avg - fwd_avg) * 1000.0;

    // --- JSON output ---
    println!("{{");
    println!("  \"framework\": \"meganeura\",");
    println!("  \"model\": \"smolvla_action_expert\",");
    println!("  \"device\": \"blade-gpu\",");
    println!("  \"chunk_size\": {},", action_seq_len);
    println!("  \"vlm_seq_len\": {},", vlm_seq_len);
    println!("  \"num_layers\": {},", config.expert.num_layers);
    println!("  \"fwd_avg_ms\": {:.2},", fwd_avg * 1000.0);
    println!("  \"fwd_median_ms\": {:.2},", fwd_median * 1000.0);
    println!("  \"train_step_avg_ms\": {:.2},", train_avg * 1000.0);
    println!("  \"train_step_median_ms\": {:.2},", train_median * 1000.0);
    println!("  \"approx_bwd_ms\": {:.2}", approx_bwd_ms);
    println!("}}");
}
