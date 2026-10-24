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
use std::time::{Duration, Instant};

use meganeura::{compile_training_graph, models::smolvla};

mod paired;

fn check_bench_preconditions(abort_on_warn: bool) {
    #[cfg(not(target_os = "linux"))]
    {
        eprintln!("preconditions: skipped (not Linux)");
        let _ = abort_on_warn;
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
    let trace_path = std::env::var("MEGANEURA_TRACE").ok();
    if trace_path.is_some() {
        meganeura::profiler::init();
    }

    let mut paired_options = paired::Options::default();
    let mut args = std::env::args().skip(1);
    let mut warmup: usize = 3;
    let mut runs: usize = 5;
    let mut force = false;
    let mut profile = false;
    let mut chunk_override: Option<usize> = None;

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--warmup" => {
                warmup = args.next().expect("--warmup value").parse().unwrap();
                paired_options.warmup = warmup;
            }
            "--runs" => {
                runs = args.next().expect("--runs value").parse().unwrap();
                paired_options.runs = runs;
            }
            "--chunk-size" => {
                chunk_override = Some(args.next().expect("--chunk-size value").parse().unwrap())
            }
            "--force" => force = true,
            "--profile" => profile = true,
            _ => {
                if !paired_options.argument(&arg, &mut args) {
                    eprintln!("unknown arg: {}", arg);
                    std::process::exit(1);
                }
            }
        }
    }

    assert!(
        !paired_options.enabled || !profile,
        "use --diagnostics for paired GPU intervals and worker counters"
    );
    let paired_gpu = paired_options.enabled.then(|| paired_options.context());
    eprintln!("checking preconditions...");
    check_bench_preconditions(!force);

    let mut config = smolvla::Config::smolvla_base();
    if let Some(cs) = chunk_override {
        config.chunk_size = cs;
    }
    let action_seq_len = config.chunk_size;
    let vlm_seq_len = 16;

    eprintln!("SmolVLA action expert training benchmark (random weights)");
    eprintln!(
        "  chunk_size={}, vlm_seq_len={}, num_layers={}",
        action_seq_len, vlm_seq_len, config.expert.num_layers
    );

    // --- Build training graph ---
    eprintln!("building training graph...");
    let training_g = smolvla::build_action_expert_training(&config, action_seq_len, vlm_seq_len);

    // The legacy report measures compilation separately; paired construction
    // records the complete build for every session instead.
    let compile_time = if paired_options.enabled {
        Duration::ZERO
    } else {
        eprintln!("measuring compile time (autodiff + e-graph + compile)...");
        let start = Instant::now();
        let _ = compile_training_graph(&training_g);
        start.elapsed()
    };

    // --- Build sessions ---
    eprintln!("compiling inference session (forward only)...");
    let (mut infer_session, forward_construction_ms) = if let Some(ref gpu) = paired_gpu {
        paired_options.build(&training_g, gpu.clone(), false, false)
    } else {
        let start = Instant::now();
        let session =
            meganeura::build(&training_g, meganeura::SessionConfig::inference_from_env()).0;
        (session, start.elapsed().as_secs_f64() * 1000.0)
    };
    eprintln!(
        "  infer: {} buffers, {} dispatches, {} barrier groups",
        infer_session.plan().buffers.len(),
        infer_session.plan().dispatches.len(),
        infer_session.num_groups()
    );

    eprintln!("compiling training session (fwd + bwd + SGD)...");
    let (mut train_session, training_construction_ms) = if let Some(ref gpu) = paired_gpu {
        paired_options.build(&training_g, gpu.clone(), true, false)
    } else {
        let start = Instant::now();
        let session = meganeura::build(&training_g, meganeura::SessionConfig::from_env()).0;
        (session, start.elapsed().as_secs_f64() * 1000.0)
    };
    eprintln!(
        "  train: {} buffers, {} dispatches, {} barrier groups",
        train_session.plan().buffers.len(),
        train_session.plan().dispatches.len(),
        train_session.num_groups()
    );

    let dev_info = train_session.device_information();
    let device_name = dev_info.device_name.clone();
    let driver_name = dev_info.driver_name.clone();
    eprintln!("  device: {} ({})", device_name, driver_name);

    // --- Initialize parameters with deterministic random values ---
    eprintln!("initializing parameters...");
    let mut parameter_seed = Vec::new();
    for (name, _) in train_session.plan().param_buffers.clone() {
        let n = train_session.param_size(&name).unwrap();
        let data: Vec<f32> = (0..n)
            .map(|i| (i as f32 * 0.01 + 1.0).sin() * 0.1)
            .collect();
        train_session.set_parameter(&name, &data);
        infer_session.set_parameter(&name, &data);
        if paired_options.enabled {
            parameter_seed.push((name, data));
        }
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

    if let Some(gpu) = paired_gpu {
        let (mut candidate_infer, candidate_forward_ms) =
            paired_options.build(&training_g, gpu.clone(), false, true);
        let (mut candidate_train, candidate_training_ms) =
            paired_options.build(&training_g, gpu, true, true);
        let prepare = |session: &mut meganeura::Session| {
            for (name, values) in &parameter_seed {
                session.set_parameter(name, values);
            }
            set_inputs(session);
        };
        let execute = |session: &mut meganeura::Session| {
            session.step();
            session.wait();
        };
        let forward = paired::measure(
            "smolvla_forward",
            &mut infer_session,
            &mut candidate_infer,
            [forward_construction_ms, candidate_forward_ms],
            paired_options,
            prepare,
            execute,
            |session| session.read_output(1),
        );
        train_session.set_learning_rate(1e-5);
        candidate_train.set_learning_rate(1e-5);
        let training = paired::measure(
            "smolvla_forward_backward_sgd",
            &mut train_session,
            &mut candidate_train,
            [training_construction_ms, candidate_training_ms],
            paired_options,
            prepare,
            execute,
            |session| {
                let mut values = vec![session.read_loss()];
                let mut changed = false;
                let mut requests = Vec::new();
                for (index, (name, initial)) in parameter_seed.iter().enumerate() {
                    let parameter = session.param_buffer(name).unwrap();
                    requests.push((parameter, initial.len(), Some(index)));
                    if let Some(&(_, gradient)) = session
                        .plan()
                        .param_grad_pairs
                        .iter()
                        .find(|entry| entry.0 == parameter)
                    {
                        requests.push((gradient, initial.len(), None));
                    }
                }
                let buffers: Vec<_> = requests.iter().map(|entry| entry.0).collect();
                // Stage the complete state once. Probing mapped parameter
                // reads individually is prohibitively slow on discrete GPUs.
                for (value, &(_, len, parameter)) in
                    session.read_buffers(&buffers).into_iter().zip(&requests)
                {
                    let logical = &value[..len];
                    if let Some(index) = parameter {
                        changed |= logical != parameter_seed[index].1;
                    }
                    values.extend_from_slice(logical);
                }
                assert!(changed, "SmolVLA SGD benchmark did not update parameters");
                values
            },
        );
        println!(
            "{}",
            serde_json::json!({
                "benchmark": "smolvla_paired", "model": "smolvla_action_expert",
                "chunk_size": action_seq_len, "vlm_seq_len": vlm_seq_len, "num_layers": config.expert.num_layers,
                "boundary": "completed step; parameter restore, input upload, and output/state qualification excluded",
                "state": "identical deterministic initial parameters restored before every execution; fixed SGD learning rate 1e-5",
                "qualification": "full scalar forward loss; training loss, every available parameter gradient and every parameter after each sample",
                "cases": [forward, training],
            })
        );
        return;
    }

    // --- Warmup ---
    eprintln!("warming up ({} runs)...", warmup);
    train_session.set_learning_rate(1e-5);
    for _ in 0..warmup {
        set_inputs(&mut infer_session);
        infer_session.step();
        infer_session.wait();

        set_inputs(&mut train_session);
        train_session.step(); // includes fused SGD update
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

        // GPU timing breakdown. Blade resolves the completed pass timestamps
        // directly after the submission fence.
        infer_session.set_profiling(true);
        eprintln!(
            "\n=== Forward pass GPU timings ({} dispatches) ===",
            infer_session.plan().dispatches.len()
        );
        set_inputs(&mut infer_session);
        infer_session.step();
        infer_session.wait();
        infer_session.dump_gpu_timings();

        train_session.set_profiling(true);
        eprintln!(
            "\n=== Training step GPU timings ({} dispatches) ===",
            train_session.plan().dispatches.len()
        );
        // The configured SGD update is included in these pass intervals.
        set_inputs(&mut train_session);
        train_session.step();
        train_session.wait();
        train_session.dump_gpu_timings();

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

    // --- Benchmark training step (with fused SGD) ---
    eprintln!("benchmarking training step ({} runs)...", runs);
    train_session.set_learning_rate(1e-5);
    let mut train_latencies = Vec::new();
    for i in 0..runs {
        set_inputs(&mut train_session);
        let t0 = Instant::now();
        train_session.step(); // fwd + bwd + SGD in single submission
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
    println!("  \"device\": \"{} ({})\",", device_name, driver_name);
    println!("  \"chunk_size\": {},", action_seq_len);
    println!("  \"vlm_seq_len\": {},", vlm_seq_len);
    println!("  \"num_layers\": {},", config.expert.num_layers);
    println!("  \"compile_time_s\": {:.2},", compile_time.as_secs_f64());
    println!("  \"fwd_avg_ms\": {:.2},", fwd_avg * 1000.0);
    println!("  \"fwd_median_ms\": {:.2},", fwd_median * 1000.0);
    println!("  \"train_step_avg_ms\": {:.2},", train_avg * 1000.0);
    println!("  \"train_step_median_ms\": {:.2},", train_median * 1000.0);
    println!("  \"approx_bwd_ms\": {:.2}", approx_bwd_ms);
    println!("}}");

    if let Some(path) = trace_path {
        eprintln!("saving trace to {}...", path);
        meganeura::profiler::save(&path).expect("failed to save trace");
        eprintln!(
            "trace saved ({} events)",
            meganeura::profiler::event_count()
        );
    }
}
