/// Stable Diffusion U-Net training benchmark for Meganeura.
///
/// Trains a U-Net denoiser on synthetic latent data via score matching:
///   1. Generate random 32×32×4 "latent" images
///   2. Add Gaussian noise at random levels
///   3. Train the U-Net to predict the noise (MSE loss)
///   4. Report training throughput, loss curve, and memory usage
///
/// Architecture: encoder (Conv2d + GroupNorm + SiLU + ResBlocks + Downsample)
///             → middle block → decoder (Upsample + skip concat + ResBlocks)
///
/// This is structurally equivalent to the Stable Diffusion 1.5 U-Net, scaled
/// down to fit in GPU memory and run quickly for benchmarking.
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
    let cfg = if use_small {
        SDUNetConfig::small()
    } else {
        SDUNetConfig::tiny()
    };

    let batch = cfg.batch_size;
    let in_c = cfg.in_channels;
    let (h, w) = (cfg.height, cfg.width);
    let in_size = (batch * in_c * h * w) as usize;
    let epochs = 3;
    let steps_per_epoch = 50;
    let lr = 1e-3_f32;

    let num_params = sd_unet::count_params(&cfg);
    println!("=== SD U-Net Training Benchmark ===");
    println!(
        "config:     {} ({}×{} latent, batch {}, {} levels, base_ch={})",
        if use_small { "small" } else { "tiny" },
        h,
        w,
        batch,
        cfg.num_levels,
        cfg.base_channels,
    );
    println!(
        "parameters: {num_params} ({:.2} MB)",
        num_params as f64 * 4.0 / 1e6
    );

    // --- Build graph ---
    println!("\nbuilding computation graph...");
    let mut g = Graph::new();
    let loss = sd_unet::build_training_graph(&mut g, &cfg);
    g.set_outputs(vec![loss]);
    println!("graph: {} nodes", g.nodes().len(),);

    // --- Compile (autodiff + optimize + GPU init) ---
    println!("compiling (autodiff + egglog + codegen)...");
    let t0 = Instant::now();
    let mut session = build_session(&g);
    let compile_time = t0.elapsed();
    println!(
        "compiled in {:.2}s: {} buffers, {} dispatches",
        compile_time.as_secs_f64(),
        session.plan().buffers.len(),
        session.plan().dispatches.len(),
    );
    println!("GPU memory: {}", session.memory_summary());

    // --- Initialize parameters (Xavier) ---
    for (name, _buf) in session.plan().param_buffers.clone() {
        let size = session.plan().buffers[_buf.0 as usize] / 4; // f32 elements
        let data = xavier_init(size);
        session.set_parameter(&name, &data);
    }

    // --- Generate synthetic data ---
    let mut rng_state: u64 = 42;
    let mut next_f32 = || -> f32 {
        rng_state = rng_state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (rng_state >> 33) as f32 / (1u64 << 31) as f32 * 2.0 - 1.0
    };

    // --- Training loop ---
    println!("\ntraining ({epochs} epochs × {steps_per_epoch} steps)...");
    session.set_learning_rate(lr);

    let t_train = Instant::now();
    for epoch in 0..epochs {
        let mut epoch_loss = 0.0_f64;

        for _step in 0..steps_per_epoch {
            // Generate random noisy latent and noise target
            let noisy: Vec<f32> = (0..in_size).map(|_| next_f32()).collect();
            let noise: Vec<f32> = (0..in_size).map(|_| next_f32() * 0.5).collect();

            session.set_input("noisy_latent", &noisy);
            session.set_input("noise_target", &noise);
            session.step();
            session.wait();

            let loss = session.read_loss();
            epoch_loss += loss as f64;
        }

        let avg_loss = epoch_loss / steps_per_epoch as f64;
        println!("  epoch {}: avg_loss = {:.6}", epoch + 1, avg_loss);
    }
    let train_time = t_train.elapsed();

    let total_steps = epochs * steps_per_epoch;
    let samples_per_sec = (total_steps * batch as usize) as f64 / train_time.as_secs_f64();
    let steps_per_sec = total_steps as f64 / train_time.as_secs_f64();

    println!("\n=== Results ===");
    println!("compile time:    {:.2}s", compile_time.as_secs_f64());
    println!("train time:      {:.2}s", train_time.as_secs_f64());
    println!("total steps:     {total_steps}");
    println!(
        "throughput:      {:.1} samples/s ({:.1} steps/s)",
        samples_per_sec, steps_per_sec
    );
    println!(
        "per-step:        {:.2}ms",
        train_time.as_secs_f64() * 1000.0 / total_steps as f64
    );

    if let Some(ref trace_file) = trace_path {
        let path = std::path::Path::new(trace_file);
        meganeura::profiler::save(path).expect("failed to save profile");
        println!("profile saved to {}", path.display());
    }
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
