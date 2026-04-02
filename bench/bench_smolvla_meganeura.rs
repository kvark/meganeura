#![allow(dead_code, clippy::too_many_arguments)]
/// Benchmark SmolVLA action expert inference with meganeura.
///
/// Measures per-step latency for the action expert denoising loop
/// (the hot path in SmolVLA inference, called `num_steps` times per
/// action prediction). Uses synthetic inputs for consistent timing.
///
/// Usage:
///   cargo run --release --example bench_smolvla_meganeura [-- --steps 10 --runs 5]
use std::collections::HashSet;
use std::time::Instant;

#[allow(dead_code, clippy::too_many_arguments)]
use meganeura::{Graph, build_inference_session, data::safetensors::SafeTensorsModel};

// ---------------------------------------------------------------------------
// Inlined SmolVLA model (with SmolVLM2 types)
// ---------------------------------------------------------------------------
mod smolvla {
    use meganeura::graph::{Graph, NodeId};

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

    pub fn build_action_expert(
        g: &mut Graph,
        config: &SmolVLAConfig,
        action_seq_len: usize,
        vlm_seq_len: usize,
    ) -> NodeId {
        let expert = &config.expert;
        let expert_hidden = expert.hidden_size;
        let text_hidden = config.vlm.text.hidden_size;
        let kv_dim = expert.kv_dim();
        let attn_dim = expert.num_attention_heads as usize * expert.head_dim as usize;
        let eps = expert.rms_norm_eps;

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

        let _vlm_hidden = g.input("vlm_hidden", &[vlm_seq_len, text_hidden]);

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
                    &[expert_hidden, attn_dim],
                );
                let q = g.matmul(h, wq);

                let wk = g.parameter(
                    &format!("{}.self_attn.k_proj.weight", prefix),
                    &[kv_dim, kv_dim],
                );
                let wv = g.parameter(
                    &format!("{}.self_attn.v_proj.weight", prefix),
                    &[kv_dim, kv_dim],
                );

                let vlm_kv = g.input(&format!("vlm_kv_layer_{}", i), &[vlm_seq_len, kv_dim]);
                let k = g.matmul(vlm_kv, wk);
                let v = g.matmul(vlm_kv, wv);

                let attn = g.cross_attention(
                    q,
                    k,
                    v,
                    expert.num_attention_heads,
                    expert.num_key_value_heads,
                    expert.head_dim,
                );

                let wo = g.parameter(
                    &format!("{}.self_attn.o_proj.weight", prefix),
                    &[attn_dim, expert_hidden],
                );
                let attn_out = g.matmul(attn, wo);
                x = g.add(x, attn_out);
            } else {
                let wq = g.parameter(
                    &format!("{}.self_attn.q_proj.weight", prefix),
                    &[expert_hidden, attn_dim],
                );
                let wk = g.parameter(
                    &format!("{}.self_attn.k_proj.weight", prefix),
                    &[expert_hidden, kv_dim],
                );
                let wv = g.parameter(
                    &format!("{}.self_attn.v_proj.weight", prefix),
                    &[expert_hidden, kv_dim],
                );

                let q = g.matmul(h, wq);
                let k = g.matmul(h, wk);
                let v = g.matmul(h, wv);

                let attn = g.causal_attention(
                    q,
                    k,
                    v,
                    expert.num_attention_heads,
                    expert.num_key_value_heads,
                    expert.head_dim,
                );

                let wo = g.parameter(
                    &format!("{}.self_attn.o_proj.weight", prefix),
                    &[attn_dim, expert_hidden],
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
        g.bias_add(out, action_out_b)
    }

    pub fn expert_transposed_weight_names(config: &SmolVLAConfig) -> Vec<String> {
        let expert = &config.expert;
        let mut names = vec![
            "model.state_proj.weight".into(),
            "model.action_in_proj.weight".into(),
            "model.action_out_proj.weight".into(),
            "model.action_time_mlp_in.weight".into(),
            "model.action_time_mlp_out.weight".into(),
        ];

        for i in 0..expert.num_layers {
            let p = format!("model.vlm_with_expert.lm_expert.layers.{}", i);
            names.push(format!("{}.self_attn.q_proj.weight", p));
            names.push(format!("{}.self_attn.k_proj.weight", p));
            names.push(format!("{}.self_attn.v_proj.weight", p));
            names.push(format!("{}.self_attn.o_proj.weight", p));
            names.push(format!("{}.mlp.gate_up_proj.weight", p));
            names.push(format!("{}.mlp.down_proj.weight", p));
        }

        names
    }
}
use smolvla::SmolVLAConfig;

/// Check that the system is in a good state for reliable benchmarking.
///
/// Warns (or aborts) if:
///  - Running on battery power (throttled clocks)
///  - GPU is busy (> threshold %) from other processes
///  - GPU GTT memory is nearly exhausted
///
/// Only performs sysfs checks on Linux; on other platforms it's a no-op.
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

        // --- AC power check ---
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
            .unwrap_or(true); // assume AC if we can't check

        if !on_ac {
            warnings.push(
                "running on BATTERY — GPU clocks may be throttled, results unreliable".to_string(),
            );
        }

        // --- GPU busy + GTT + clock check (per card) ---
        const GPU_BUSY_THRESHOLD: u64 = 10; // percent
        const GTT_FREE_THRESHOLD: u64 = 500 * 1024 * 1024; // 500 MB
        const GPU_CLOCK_MIN_RATIO: f64 = 0.7; // must be >= 70% of peak clock

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

            // GPU busy %
            if let Some(pct) = std::fs::read_to_string(dev.join("gpu_busy_percent"))
                .ok()
                .and_then(|s| s.trim().parse::<u64>().ok())
            {
                if pct > GPU_BUSY_THRESHOLD {
                    warnings.push(format!(
                        "{}: GPU {}% busy — other processes using GPU, results may be noisy",
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

            // GTT memory (system RAM mapped for GPU) — main memory pool for iGPU
            let gtt_used = std::fs::read_to_string(dev.join("mem_info_gtt_used"))
                .ok()
                .and_then(|s| s.trim().parse::<u64>().ok());
            let gtt_total = std::fs::read_to_string(dev.join("mem_info_gtt_total"))
                .ok()
                .and_then(|s| s.trim().parse::<u64>().ok());
            if let (Some(used), Some(total)) = (gtt_used, gtt_total) {
                let free = total.saturating_sub(used);
                eprintln!(
                    "  {}: GTT {:.0}MB / {:.0}MB used ({:.0}MB free)",
                    card.file_name().to_string_lossy(),
                    used as f64 / 1e6,
                    total as f64 / 1e6,
                    free as f64 / 1e6
                );
                if free < GTT_FREE_THRESHOLD {
                    warnings.push(format!(
                        "{}: only {:.0}MB GTT memory free — risk of GPU OOM",
                        card.file_name().to_string_lossy(),
                        free as f64 / 1e6
                    ));
                }
            }

            // GPU clock frequency: read pp_dpm_sclk, find current (*) and max
            if let Ok(dpm) = std::fs::read_to_string(dev.join("pp_dpm_sclk")) {
                let mut cur_mhz: Option<u64> = None;
                let mut max_mhz: u64 = 0;
                for line in dpm.lines() {
                    let mhz = line
                        .split_whitespace()
                        .find(|t| t.ends_with("Mhz"))
                        .and_then(|t| t.trim_end_matches("Mhz").parse::<u64>().ok())
                        .unwrap_or(0);
                    if mhz > max_mhz {
                        max_mhz = mhz;
                    }
                    if line.contains('*') {
                        cur_mhz = Some(mhz);
                    }
                }
                if let Some(cur) = cur_mhz {
                    let ratio = cur as f64 / max_mhz.max(1) as f64;
                    // Clock will boost under load during warmup — print as info only,
                    // not a hard warning that blocks benchmarking.
                    let boost_note = if ratio < GPU_CLOCK_MIN_RATIO {
                        " — will boost under load"
                    } else {
                        ""
                    };
                    eprintln!(
                        "  {}: GPU clock {}MHz / {}MHz ({:.0}%){}",
                        card.file_name().to_string_lossy(),
                        cur,
                        max_mhz,
                        ratio * 100.0,
                        boost_note
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

const REPO_ID: &str = "lerobot/smolvla_base";

fn main() {
    env_logger::init();
    let trace_path = std::env::var("MEGANEURA_TRACE").ok();
    if trace_path.is_some() {
        meganeura::profiler::init();
    }

    let mut args = std::env::args().skip(1);
    let mut warmup: usize = 3;
    let mut runs: usize = 5;
    let mut num_steps: usize = 0; // 0 = use config default
    let mut force = false; // skip precondition abort

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--warmup" => warmup = args.next().expect("--warmup value").parse().unwrap(),
            "--runs" => runs = args.next().expect("--runs value").parse().unwrap(),
            "--steps" => num_steps = args.next().expect("--steps value").parse().unwrap(),
            "--force" => force = true,
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
    let vlm_seq_len = 16; // representative VLM context length
    let denoise_steps = if num_steps > 0 {
        num_steps
    } else {
        config.num_steps
    };

    eprintln!("SmolVLA action expert benchmark");
    eprintln!(
        "  chunk_size={}, vlm_seq_len={}, denoise_steps={}",
        action_seq_len, vlm_seq_len, denoise_steps
    );

    // --- Download model ---
    eprintln!("downloading model...");
    let model = SafeTensorsModel::download(REPO_ID).expect("download failed");

    // --- Build action expert graph ---
    eprintln!("building action expert graph...");
    let mut g = Graph::new();
    let action_out = smolvla::build_action_expert(&mut g, &config, action_seq_len, vlm_seq_len);
    g.set_outputs(vec![action_out]);

    eprintln!("compiling...");
    let mut session = build_inference_session(&g);
    eprintln!(
        "ready: {} buffers, {} dispatches",
        session.plan().buffers.len(),
        session.plan().dispatches.len()
    );

    // --- Load weights ---
    eprintln!("loading weights...");
    let transposed = smolvla::expert_transposed_weight_names(&config);
    let transposed_set: HashSet<&str> = transposed.iter().map(|s| s.as_str()).collect();

    for (name, _) in session.plan().param_buffers.clone() {
        // Skip VLM KV inputs (not parameters)
        if name.starts_with("vlm_kv_layer_") {
            continue;
        }
        if transposed_set.contains(name.as_str()) {
            let data = model
                .tensor_f32_auto_transposed(&name)
                .unwrap_or_else(|e| panic!("{}: {}", name, e));
            session.set_parameter(&name, &data);
        } else {
            let data = model
                .tensor_f32_auto(&name)
                .unwrap_or_else(|e| panic!("{}: {}", name, e));
            session.set_parameter(&name, &data);
        }
    }
    eprintln!("weights loaded.");

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

    // --- Run helper: one full denoising loop ---
    let run_denoise = |session: &mut meganeura::Session| -> f64 {
        let t0 = Instant::now();
        for _step in 0..denoise_steps {
            session.set_input("noisy_actions", &noisy_actions);
            session.set_input("timestep", &timestep);
            // Set cross-attention KV for each odd layer
            for i in 0..config.expert.num_layers {
                if i % config.expert.self_attn_every_n_layers != 0 {
                    session.set_input(&format!("vlm_kv_layer_{}", i), &vlm_kv);
                }
            }
            session.step();
            session.wait();
        }
        t0.elapsed().as_secs_f64()
    };

    // --- Single forward pass to dump output for correctness check ---
    {
        session.set_input("noisy_actions", &noisy_actions);
        session.set_input("timestep", &timestep);
        for i in 0..config.expert.num_layers {
            if i % config.expert.self_attn_every_n_layers != 0 {
                session.set_input(&format!("vlm_kv_layer_{}", i), &vlm_kv);
            }
        }
        session.step();
        session.wait();

        let output_len = action_seq_len * config.max_action_dim;
        let output = session.read_output(output_len);
        let output_path = "bench/results/smolvla_meganeura_output.json";
        let output_str: Vec<String> = output.iter().map(|v| format!("{:.8e}", v)).collect();
        let output_json = format!("[{}]", output_str.join(", "));
        std::fs::write(output_path, &output_json).expect("write output");
        eprintln!("output saved to {} ({} floats)", output_path, output.len());
    }

    // --- Warmup + GPU timing dump ---
    eprintln!("warming up ({} runs)...", warmup);
    for _ in 0..warmup {
        run_denoise(&mut session);
    }
    // Run one more step to trigger encoder.start() which collects timings
    // from the previous submission.
    {
        session.set_input("noisy_actions", &noisy_actions);
        session.set_input("timestep", &timestep);
        for i in 0..config.expert.num_layers {
            if i % config.expert.self_attn_every_n_layers != 0 {
                session.set_input(&format!("vlm_kv_layer_{}", i), &vlm_kv);
            }
        }
        session.step();
        session.dump_gpu_timings();
        session.wait();
    }

    // --- Benchmark ---
    eprintln!(
        "benchmarking ({} runs, {} denoise steps each)...",
        runs, denoise_steps
    );
    let mut latencies = Vec::new();

    for i in 0..runs {
        let elapsed = run_denoise(&mut session);
        latencies.push(elapsed);
        let per_step = elapsed / denoise_steps as f64;
        eprintln!(
            "  run {}: {:.1}ms total, {:.2}ms/step",
            i + 1,
            elapsed * 1000.0,
            per_step * 1000.0,
        );
    }

    // --- Statistics ---
    let avg = latencies.iter().sum::<f64>() / runs as f64;
    let mut sorted = latencies.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[runs / 2];
    let stdev = if runs > 1 {
        let var = latencies.iter().map(|l| (l - avg).powi(2)).sum::<f64>() / (runs - 1) as f64;
        var.sqrt()
    } else {
        0.0
    };

    let avg_per_step = avg / denoise_steps as f64;
    let steps_per_sec = denoise_steps as f64 / avg;

    // --- JSON output ---
    println!("{{");
    println!("  \"framework\": \"meganeura\",");
    println!("  \"model\": \"{}\",", REPO_ID);
    println!("  \"device\": \"blade-gpu\",");
    println!("  \"dtype\": \"float32\",");
    println!("  \"chunk_size\": {},", action_seq_len);
    println!("  \"vlm_seq_len\": {},", vlm_seq_len);
    println!("  \"denoise_steps\": {},", denoise_steps);
    println!("  \"runs\": {},", runs);
    println!("  \"avg_latency_ms\": {:.2},", avg * 1000.0);
    println!("  \"median_latency_ms\": {:.2},", median * 1000.0);
    println!("  \"stdev_latency_ms\": {:.2},", stdev * 1000.0);
    println!("  \"avg_per_step_ms\": {:.2},", avg_per_step * 1000.0);
    println!("  \"steps_per_second\": {:.2}", steps_per_sec);
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
