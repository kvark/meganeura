/// SD U-Net training benchmark for meganeura.
///
/// Architecture: Conv2d + GroupNorm + SiLU ResBlocks with skip connections,
/// structurally matching the Stable Diffusion 1.5 U-Net.
///
/// Outputs JSON to stdout (for compare.sh), human-readable to stderr.
use meganeura::models::sd_unet::{self, SDUNetConfig};
use meganeura::{Graph, build_session};
use std::time::Instant;

fn main() {
    env_logger::init();

    let trace_path = std::env::var("MEGANEURA_TRACE").ok();
    if trace_path.is_some() {
        meganeura::profiler::init();
    }

    let use_small = std::env::args().any(|a| a == "--small");
    let profile = std::env::var("MEGANEURA_PROFILE").is_ok();
    let warmup: usize = parse_arg("--warmup").unwrap_or(2);
    let runs: usize = parse_arg("--runs").unwrap_or(5);
    let steps_per_run: usize = parse_arg("--steps").unwrap_or(20);

    let cfg = if use_small {
        SDUNetConfig::small()
    } else {
        SDUNetConfig::tiny()
    };

    let batch = cfg.batch_size;
    let in_c = cfg.in_channels;
    let (h, w) = (cfg.height, cfg.width);
    let in_size = (batch * in_c * h * w) as usize;
    let lr = 1e-3_f32;

    let num_params = sd_unet::count_params(&cfg);
    let config_name = if use_small { "small" } else { "tiny" };
    eprintln!("=== SD U-Net Training Benchmark ===");
    eprintln!(
        "config:     {} ({}x{} latent, batch {}, {} levels, base_ch={})",
        config_name, h, w, batch, cfg.num_levels, cfg.base_channels,
    );
    eprintln!(
        "parameters: {} ({:.2} MB)",
        num_params,
        num_params as f64 * 4.0 / 1e6
    );

    // --- Build graph ---
    eprintln!("building computation graph...");
    let mut g = Graph::new();
    let loss = sd_unet::build_training_graph(&mut g, &cfg);
    g.set_outputs(vec![loss]);
    eprintln!("graph: {} nodes", g.nodes().len());

    // --- Compile ---
    eprintln!("compiling (autodiff + egglog + codegen)...");
    let t0 = Instant::now();
    let mut session = build_session(&g);
    let compile_time = t0.elapsed();
    let device_info = session.device_information().clone();
    let device_name = &device_info.device_name;
    let driver_name = &device_info.driver_name;
    eprintln!(
        "compiled in {:.2}s: {} buffers, {} dispatches",
        compile_time.as_secs_f64(),
        session.plan().buffers.len(),
        session.plan().dispatches.len(),
    );
    eprintln!("GPU: {} ({})", device_name, driver_name);
    eprintln!("memory: {}", session.memory_summary());

    // --- Initialize parameters ---
    for (name, buf) in session.plan().param_buffers.clone() {
        let size = session.plan().buffers[buf.0 as usize] / 4;
        session.set_parameter(&name, &xavier_init(size));
    }

    // --- PRNG ---
    let mut rng_state: u64 = 42;
    let mut next_f32 = || -> f32 {
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (rng_state >> 33) as f32 / (1u64 << 31) as f32 * 2.0 - 1.0
    };

    session.set_learning_rate(lr);

    // --- Warmup ---
    eprintln!("warmup ({warmup} runs)...");
    for _ in 0..warmup {
        let noisy: Vec<f32> = (0..in_size).map(|_| next_f32()).collect();
        let noise: Vec<f32> = (0..in_size).map(|_| next_f32() * 0.5).collect();
        session.set_input("noisy_latent", &noisy);
        session.set_input("noise_target", &noise);
        session.step();
        session.wait();
    }

    // --- GPU profiling (MEGANEURA_PROFILE=1) ---
    if profile {
        session.set_profiling(true);
        eprintln!(
            "\n=== GPU timings ({} dispatches) ===",
            session.plan().dispatches.len()
        );
        // 3-step dance for blade's 2-buffer ring:
        // step A (profiled) → step B (advance ring) → step C reads A's timestamps
        let noisy: Vec<f32> = (0..in_size).map(|_| next_f32()).collect();
        let noise: Vec<f32> = (0..in_size).map(|_| next_f32() * 0.5).collect();
        session.set_input("noisy_latent", &noisy);
        session.set_input("noise_target", &noise);
        session.step();
        session.wait();

        let noisy: Vec<f32> = (0..in_size).map(|_| next_f32()).collect();
        let noise: Vec<f32> = (0..in_size).map(|_| next_f32() * 0.5).collect();
        session.set_input("noisy_latent", &noisy);
        session.set_input("noise_target", &noise);
        session.step();
        session.wait();

        let noisy: Vec<f32> = (0..in_size).map(|_| next_f32()).collect();
        let noise: Vec<f32> = (0..in_size).map(|_| next_f32() * 0.5).collect();
        session.set_input("noisy_latent", &noisy);
        session.set_input("noise_target", &noise);
        session.step();
        session.dump_gpu_timings();
        session.wait();

        session.set_profiling(false);
        return;
    }

    // --- Timed runs ---
    eprintln!("benchmarking ({runs} runs x {steps_per_run} steps)...");
    let mut run_times: Vec<f64> = Vec::new();

    for r in 0..runs {
        let t_run = Instant::now();
        let mut run_loss = 0.0_f64;

        for _ in 0..steps_per_run {
            let noisy: Vec<f32> = (0..in_size).map(|_| next_f32()).collect();
            let noise: Vec<f32> = (0..in_size).map(|_| next_f32() * 0.5).collect();
            session.set_input("noisy_latent", &noisy);
            session.set_input("noise_target", &noise);
            session.step();
            session.wait();
            run_loss += session.read_loss() as f64;
        }

        let elapsed = t_run.elapsed().as_secs_f64();
        run_times.push(elapsed);
        let avg_loss = run_loss / steps_per_run as f64;
        eprintln!(
            "  run {}: {:.2}ms total, {:.2}ms/step, avg_loss={:.6}",
            r + 1,
            elapsed * 1000.0,
            elapsed * 1000.0 / steps_per_run as f64,
            avg_loss,
        );
    }

    // --- Statistics ---
    let mut sorted = run_times.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let avg_s = run_times.iter().sum::<f64>() / runs as f64;
    let median_s = sorted[runs / 2];
    let step_avg_ms = avg_s * 1000.0 / steps_per_run as f64;
    let step_median_ms = median_s * 1000.0 / steps_per_run as f64;
    let samples_per_sec = (steps_per_run * batch as usize) as f64 / avg_s;

    // --- JSON output ---
    println!("{{");
    println!("  \"framework\": \"meganeura\",");
    println!("  \"model\": \"sd_unet_{}\",", config_name);
    println!("  \"device\": \"{} ({})\",", device_name, driver_name);
    println!("  \"parameters\": {},", num_params);
    println!("  \"batch_size\": {},", batch);
    println!("  \"resolution\": \"{}x{}\",", h, w);
    println!("  \"compile_time_s\": {:.2},", compile_time.as_secs_f64());
    println!("  \"train_step_avg_ms\": {:.2},", step_avg_ms);
    println!("  \"train_step_median_ms\": {:.2},", step_median_ms);
    println!("  \"samples_per_sec\": {:.1},", samples_per_sec);
    println!(
        "  \"memory_mb\": {:.1}",
        session.memory_summary().total_buffer_bytes as f64 / 1e6
    );
    println!("}}");

    if let Some(ref trace_file) = trace_path {
        let path = std::path::Path::new(trace_file);
        meganeura::profiler::save(path).expect("failed to save profile");
        eprintln!("profile saved to {}", path.display());
    }
}

fn parse_arg(flag: &str) -> Option<usize> {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == flag)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
}

fn xavier_init(size: usize) -> Vec<f32> {
    use std::f32::consts::PI;
    let fan = (size as f32).sqrt();
    let scale = (2.0 / (fan + fan)).sqrt();
    (0..size)
        .map(|i| {
            let x = (i as f32 + 1.0) * 0.618_034;
            (x * PI * 2.0).sin() * scale
        })
        .collect()
}
