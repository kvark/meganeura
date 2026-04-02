#![allow(dead_code, clippy::too_many_arguments)]
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

#[allow(dead_code, clippy::too_many_arguments)]
use meganeura::{build_inference_session, build_session, compile_training_graph};

// ---------------------------------------------------------------------------
// Inlined SmolVLA model (with SmolVLM2 types)
// ---------------------------------------------------------------------------
mod smolvla {
    use meganeura::graph::Graph;

    pub struct VisionConfig {
        pub image_size: usize,
        pub patch_size: usize,
        pub hidden_size: usize,
        pub num_attention_heads: u32,
        pub num_hidden_layers: usize,
        pub intermediate_size: usize,
        pub layer_norm_eps: f32,
    }

    pub struct TextConfig {
        pub vocab_size: usize,
        pub hidden_size: usize,
        pub num_hidden_layers: usize,
        pub num_attention_heads: u32,
        pub num_key_value_heads: u32,
        pub intermediate_size: usize,
        pub rms_norm_eps: f32,
        pub rope_theta: f32,
    }

    pub struct SmolVLM2Config {
        pub vision: VisionConfig,
        pub text: TextConfig,
        pub scale_factor: usize,
    }

    pub struct ExpertConfig {
        pub hidden_size: usize,
        pub num_layers: usize,
        pub num_attention_heads: u32,
        pub num_key_value_heads: u32,
        pub head_dim: u32,
        pub intermediate_size: usize,
        pub rms_norm_eps: f32,
        pub self_attn_every_n_layers: usize,
    }

    impl ExpertConfig {
        pub fn kv_dim(&self) -> usize {
            self.num_key_value_heads as usize * self.head_dim as usize
        }
    }

    pub struct SmolVLAConfig {
        pub vlm: SmolVLM2Config,
        pub expert: ExpertConfig,
        pub max_action_dim: usize,
        pub max_state_dim: usize,
        pub chunk_size: usize,
        pub num_steps: usize,
        pub num_vlm_layers: usize,
    }

    impl SmolVLAConfig {
        pub fn smolvla_base() -> Self {
            Self {
                vlm: SmolVLM2Config {
                    vision: VisionConfig {
                        image_size: 512,
                        patch_size: 16,
                        hidden_size: 768,
                        num_attention_heads: 12,
                        num_hidden_layers: 12,
                        intermediate_size: 3072,
                        layer_norm_eps: 1e-6,
                    },
                    text: TextConfig {
                        vocab_size: 49280,
                        hidden_size: 960,
                        num_hidden_layers: 32,
                        num_attention_heads: 15,
                        num_key_value_heads: 5,
                        intermediate_size: 2560,
                        rms_norm_eps: 1e-5,
                        rope_theta: 100000.0,
                    },
                    scale_factor: 4,
                },
                expert: ExpertConfig {
                    hidden_size: 720,
                    num_layers: 16,
                    num_attention_heads: 15,
                    num_key_value_heads: 5,
                    head_dim: 64,
                    intermediate_size: 2048,
                    rms_norm_eps: 1e-5,
                    self_attn_every_n_layers: 2,
                },
                max_action_dim: 32,
                max_state_dim: 32,
                chunk_size: 50,
                num_steps: 10,
                num_vlm_layers: 16,
            }
        }
    }

    pub fn build_action_expert_training(
        config: &SmolVLAConfig,
        action_seq_len: usize,
        vlm_seq_len: usize,
    ) -> Graph {
        let mut g = Graph::new();
        let expert = &config.expert;
        let expert_hidden = expert.hidden_size;
        let kv_dim = expert.kv_dim();
        let eps = expert.rms_norm_eps;

        let num_heads = expert.num_attention_heads;
        let num_kv_heads = expert.num_key_value_heads;
        let hd = expert.head_dim;
        let q_dim = (num_heads * hd) as usize;
        let kv_dim_full = (num_kv_heads * hd) as usize;

        let noisy_actions = g.input("noisy_actions", &[action_seq_len, config.max_action_dim]);
        let timestep = g.input("timestep", &[1, expert_hidden * 2]);

        let action_in_w = g.parameter(
            "model.action_in_proj.weight",
            &[config.max_action_dim, expert_hidden],
        );
        let action_in_b = g.parameter("model.action_in_proj.bias", &[expert_hidden]);
        let mut x = g.matmul(noisy_actions, action_in_w);
        x = g.bias_add(x, action_in_b);

        let time_in_w = g.parameter(
            "model.action_time_mlp_in.weight",
            &[expert_hidden * 2, expert_hidden],
        );
        let time_in_b = g.parameter("model.action_time_mlp_in.bias", &[expert_hidden]);
        let time_out_w = g.parameter(
            "model.action_time_mlp_out.weight",
            &[expert_hidden, expert_hidden],
        );
        let time_out_b = g.parameter("model.action_time_mlp_out.bias", &[expert_hidden]);
        let time_h = g.matmul(timestep, time_in_w);
        let time_h = g.bias_add(time_h, time_in_b);
        let time_h = g.silu(time_h);
        let time_h = g.matmul(time_h, time_out_w);
        let time_embed = g.bias_add(time_h, time_out_b);
        x = g.broadcast_add(x, time_embed);

        for i in 0..expert.num_layers {
            let prefix = format!("model.vlm_with_expert.lm_expert.layers.{}", i);
            let is_cross_attn = i % expert.self_attn_every_n_layers != 0;

            let ln1_w = g.parameter(
                &format!("{}.input_layernorm.weight", prefix),
                &[expert_hidden],
            );
            let h = g.rms_norm(x, ln1_w, eps);

            if is_cross_attn {
                let wq = g.parameter(
                    &format!("{}.self_attn.q_proj.weight", prefix),
                    &[expert_hidden, q_dim],
                );
                let q = g.matmul(h, wq);

                let vlm_kv = g.input(&format!("vlm_kv_layer_{}", i), &[vlm_seq_len, kv_dim]);
                let wk = g.parameter(
                    &format!("{}.self_attn.k_proj.weight", prefix),
                    &[kv_dim, kv_dim_full],
                );
                let wv = g.parameter(
                    &format!("{}.self_attn.v_proj.weight", prefix),
                    &[kv_dim, kv_dim_full],
                );
                let k = g.matmul(vlm_kv, wk);
                let v = g.matmul(vlm_kv, wv);

                let attn = g.multi_head_attn(q, k, v, num_heads, num_kv_heads, hd, true);

                let wo = g.parameter(
                    &format!("{}.self_attn.o_proj.weight", prefix),
                    &[q_dim, expert_hidden],
                );
                let attn_out = g.matmul(attn, wo);
                x = g.add(x, attn_out);
            } else {
                let wq = g.parameter(
                    &format!("{}.self_attn.q_proj.weight", prefix),
                    &[expert_hidden, q_dim],
                );
                let wk = g.parameter(
                    &format!("{}.self_attn.k_proj.weight", prefix),
                    &[expert_hidden, kv_dim_full],
                );
                let wv = g.parameter(
                    &format!("{}.self_attn.v_proj.weight", prefix),
                    &[expert_hidden, kv_dim_full],
                );
                let q = g.matmul(h, wq);
                let k = g.matmul(h, wk);
                let v = g.matmul(h, wv);

                let attn = g.multi_head_attn(q, k, v, num_heads, num_kv_heads, hd, false);

                let wo = g.parameter(
                    &format!("{}.self_attn.o_proj.weight", prefix),
                    &[q_dim, expert_hidden],
                );
                let attn_out = g.matmul(attn, wo);
                x = g.add(x, attn_out);
            }

            let ln2_w = g.parameter(
                &format!("{}.post_attention_layernorm.weight", prefix),
                &[expert_hidden],
            );
            let h = g.rms_norm(x, ln2_w, eps);

            let w_gate = g.parameter(
                &format!("{}.mlp.gate_proj.weight", prefix),
                &[expert_hidden, expert.intermediate_size],
            );
            let w_up = g.parameter(
                &format!("{}.mlp.up_proj.weight", prefix),
                &[expert_hidden, expert.intermediate_size],
            );
            let w_down = g.parameter(
                &format!("{}.mlp.down_proj.weight", prefix),
                &[expert.intermediate_size, expert_hidden],
            );
            let gate = g.matmul(h, w_gate);
            let up = g.matmul(h, w_up);
            let gate_up = g.swiglu(gate, up);
            let ffn_out = g.matmul(gate_up, w_down);
            x = g.add(x, ffn_out);
        }

        let action_out_w = g.parameter(
            "model.action_out_proj.weight",
            &[expert_hidden, config.max_action_dim],
        );
        let action_out_b = g.parameter("model.action_out_proj.bias", &[config.max_action_dim]);
        let out = g.matmul(x, action_out_w);
        let out = g.bias_add(out, action_out_b);

        let target = g.input("target_actions", &[action_seq_len, config.max_action_dim]);
        let neg_target = g.neg(target);
        let diff = g.add(out, neg_target);
        let sq_diff = g.mul(diff, diff);
        let loss = g.mean_all(sq_diff);
        g.set_outputs(vec![loss]);
        g
    }
}
use smolvla::SmolVLAConfig;

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
    let trace_path = std::env::var("MEGANEURA_TRACE").ok();
    if trace_path.is_some() {
        meganeura::profiler::init();
    }

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

    // --- Measure compile time (autodiff + e-graph + plan, no GPU init) ---
    eprintln!("measuring compile time (autodiff + e-graph + compile)...");
    let compile_t0 = Instant::now();
    let _ = compile_training_graph(&training_g);
    let compile_time = compile_t0.elapsed();
    eprintln!(
        "  compile time: {:.0}ms",
        compile_time.as_secs_f64() * 1000.0
    );

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

    let dev_info = train_session.device_information();
    let device_name = dev_info.device_name.clone();
    let driver_name = dev_info.driver_name.clone();
    eprintln!("  device: {} ({})", device_name, driver_name);

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

        // GPU timing breakdown — blade 0.8.1+ supports up to 1000 timestamps
        // per submission.
        //
        // Blade's command encoder uses a 2-buffer ring: start() reads timestamps
        // from 2 submissions ago. So we need 3 steps to get step A's timings:
        //   step A (profiling=true) → step B (advances ring) → step C's start()
        //   reads A's data → dump_gpu_timings() shows A's per-shader breakdown.
        infer_session.set_profiling(true);
        eprintln!(
            "\n=== Forward pass GPU timings ({} dispatches) ===",
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
            "\n=== Training step GPU timings ({} dispatches) ===",
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
