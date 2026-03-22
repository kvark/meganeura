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

use meganeura::{
    Graph, build_inference_session,
    data::safetensors::SafeTensorsModel,
    models::smolvla::{self, SmolVLAConfig},
};

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

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--warmup" => warmup = args.next().expect("--warmup value").parse().unwrap(),
            "--runs" => runs = args.next().expect("--runs value").parse().unwrap(),
            "--steps" => num_steps = args.next().expect("--steps value").parse().unwrap(),
            _ => {
                eprintln!("unknown arg: {}", arg);
                std::process::exit(1);
            }
        }
    }

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

    // --- Warmup ---
    eprintln!("warming up ({} runs)...", warmup);
    for _ in 0..warmup {
        run_denoise(&mut session);
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
