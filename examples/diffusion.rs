/// Denoising diffusion benchmark for Meganeura.
///
/// Trains a denoising MLP via score matching — the core operation behind
/// diffusion models (DDPM, etc.):
///   1. Generate synthetic 28×28 image patterns
///   2. Corrupt with Gaussian noise at various levels
///   3. Train a 4-layer MLP to predict the added noise
///   4. Report training throughput and loss curve
///
/// The network uses the same ops as a real diffusion backbone
/// (matmul, bias_add, relu, MSE loss), just with an MLP instead of a U-Net.
///
/// MSE loss is built from primitives: `mean_all(mul(diff, diff))`
/// where `diff = add(pred, neg(target))`.
use meganeura::{DataLoader, Graph, TrainConfig, Trainer, build_session};
use std::time::Instant;

fn main() {
    env_logger::init();

    // Set up Perfetto profiling: MEGANEURA_TRACE=path.pftrace
    let trace_path = std::env::var("MEGANEURA_TRACE").ok();
    if trace_path.is_some() {
        meganeura::profiler::init();
    }

    let batch = 64;
    let img_dim = 784; // 28×28 flattened
    let hidden = 256;
    let epochs = 5;
    let lr = 0.001_f32;
    let n_samples = 6400;
    let noise_level = 0.5_f32;

    // --- Generate noisy data pairs ---
    println!(
        "generating {} denoising pairs (σ={})...",
        n_samples, noise_level
    );
    let (noisy_images, noise_targets) =
        generate_denoising_data(n_samples, img_dim, noise_level);
    let mut loader = DataLoader::new(noisy_images, noise_targets, img_dim, img_dim, batch);
    println!(
        "{} samples, {} batches/epoch",
        loader.len(),
        loader.num_batches()
    );

    // --- Build denoising MLP ---
    //   noisy_image → 256 → 256 → 256 → predicted_noise
    let mut g = Graph::new();

    let x = g.input("x", &[batch, img_dim]); // noisy image
    let target = g.input("labels", &[batch, img_dim]); // noise that was added

    // Layer 1
    let w1 = g.parameter("w1", &[img_dim, hidden]);
    let b1 = g.parameter("b1", &[hidden]);
    let mm1 = g.matmul(x, w1);
    let ba1 = g.bias_add(mm1, b1);
    let h1 = g.relu(ba1);

    // Layer 2
    let w2 = g.parameter("w2", &[hidden, hidden]);
    let b2 = g.parameter("b2", &[hidden]);
    let mm2 = g.matmul(h1, w2);
    let ba2 = g.bias_add(mm2, b2);
    let h2 = g.relu(ba2);

    // Layer 3
    let w3 = g.parameter("w3", &[hidden, hidden]);
    let b3 = g.parameter("b3", &[hidden]);
    let mm3 = g.matmul(h2, w3);
    let ba3 = g.bias_add(mm3, b3);
    let h3 = g.relu(ba3);

    // Output layer (linear — no activation)
    let w4 = g.parameter("w4", &[hidden, img_dim]);
    let b4 = g.parameter("b4", &[img_dim]);
    let mm4 = g.matmul(h3, w4);
    let pred = g.bias_add(mm4, b4);

    // MSE loss: mean((pred - target)²)
    let neg_target = g.neg(target);
    let diff = g.add(pred, neg_target);
    let sq = g.mul(diff, diff);
    let loss = g.mean_all(sq);
    g.set_outputs(vec![loss]);

    println!("denoising MLP: 4 layers, {} hidden units", hidden);
    println!("parameters: {}",
        2 * (img_dim * hidden + hidden)       // layers 1 & 4
        + 2 * (hidden * hidden + hidden)      // layers 2 & 3
    );

    // --- Build session ---
    println!("building session (autodiff + egglog + compile)...");
    let t0 = Instant::now();
    let mut session = build_session(&g);
    let compile_time = t0.elapsed();
    println!(
        "session ready in {:.2}s: {} buffers, {} dispatches",
        compile_time.as_secs_f32(),
        session.plan().buffers.len(),
        session.plan().dispatches.len(),
    );

    // --- Initialize parameters (Xavier) ---
    init_param(&mut session, "w1", img_dim, hidden);
    init_param(&mut session, "w2", hidden, hidden);
    init_param(&mut session, "w3", hidden, hidden);
    init_param(&mut session, "w4", hidden, img_dim);
    session.set_parameter("b1", &vec![0.0; hidden]);
    session.set_parameter("b2", &vec![0.0; hidden]);
    session.set_parameter("b3", &vec![0.0; hidden]);
    session.set_parameter("b4", &vec![0.0; img_dim]);

    // --- Train ---
    println!("\ntraining ({} epochs)...", epochs);
    let config = TrainConfig {
        learning_rate: lr,
        log_interval: 20,
        ..TrainConfig::default()
    };
    let mut trainer = Trainer::new(session, config);

    let t_train = Instant::now();
    let history = trainer.train(&mut loader, epochs);
    let elapsed = t_train.elapsed();

    // --- Report ---
    let total_steps: usize = history.epochs.iter().map(|e| e.steps).sum();
    let samples_per_sec = (total_steps * batch) as f64 / elapsed.as_secs_f64();

    println!("\n=== diffusion benchmark results ===");
    println!("model:           4-layer MLP ({img_dim}→{hidden}→{hidden}→{hidden}→{img_dim})");
    println!("batch size:      {batch}");
    println!("epochs:          {epochs}");
    println!("total steps:     {total_steps}");
    println!("compile time:    {:.2}s", compile_time.as_secs_f64());
    println!("train time:      {:.2}s", elapsed.as_secs_f64());
    println!("throughput:      {:.0} samples/s", samples_per_sec);
    for stats in &history.epochs {
        println!(
            "  epoch {:>2}: avg_loss = {:.6}  ({} steps)",
            stats.epoch, stats.avg_loss, stats.steps,
        );
    }
    if let Some(final_loss) = history.final_loss() {
        println!("final loss:      {:.6}", final_loss);
    }

    // Save Perfetto trace when profiling.
    if let Some(ref trace_file) = trace_path {
        let path = std::path::Path::new(trace_file);
        meganeura::profiler::save(path).expect("failed to save profile");
        println!("profile saved to {}", path.display());
    }
}

/// Generate (noisy_image, noise) pairs for denoising score matching.
///
/// Clean images are procedural 28×28 patterns (gradients, circles,
/// checkerboards). Noise is sampled from a simple deterministic PRNG
/// that approximates Gaussian via the Irwin-Hall method.
fn generate_denoising_data(
    n: usize,
    dim: usize,
    noise_level: f32,
) -> (Vec<f32>, Vec<f32>) {
    let side = (dim as f32).sqrt() as usize; // 28 for dim=784

    let mut noisy = vec![0.0_f32; n * dim];
    let mut noise = vec![0.0_f32; n * dim];

    // Deterministic LCG PRNG
    let mut state: u64 = 314159265;
    let mut next_uniform = || -> f32 {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as f32 / (1u64 << 31) as f32 // [0, 1)
    };
    // Approximate standard normal via sum of 4 uniforms (Irwin-Hall)
    let mut next_normal = || -> f32 {
        let sum = next_uniform() + next_uniform() + next_uniform() + next_uniform();
        sum - 2.0 // mean 0, std ≈ 0.577
    };

    for i in 0..n {
        let pattern = i % 5;
        let base = i * dim;
        let phase = (i / 5) as f32 * 0.1; // vary across samples

        for y in 0..side {
            for x in 0..side {
                let px = base + y * side + x;
                let fx = x as f32 / side as f32;
                let fy = y as f32 / side as f32;

                let clean = match pattern {
                    0 => fx, // horizontal gradient
                    1 => fy, // vertical gradient
                    2 => {
                        // circle
                        let dx = fx - 0.5;
                        let dy = fy - 0.5;
                        let r = (dx * dx + dy * dy).sqrt();
                        if r < 0.3 + phase * 0.1 {
                            1.0
                        } else {
                            0.0
                        }
                    }
                    3 => {
                        // checkerboard (varying frequency)
                        let freq = 4 + (i / 5) % 4;
                        if (x / freq + y / freq) % 2 == 0 {
                            0.8
                        } else {
                            0.2
                        }
                    }
                    _ => {
                        // diagonal gradient
                        ((fx + fy) * 0.5 + phase).fract()
                    }
                };

                let n_val = next_normal() * noise_level;
                noise[px] = n_val;
                noisy[px] = (clean + n_val).clamp(0.0, 1.0);
            }
        }
    }

    (noisy, noise)
}

/// Xavier initialization for a weight matrix.
fn init_param(session: &mut meganeura::Session, name: &str, fan_in: usize, fan_out: usize) {
    use std::f32::consts::PI;
    let scale = (2.0 / (fan_in + fan_out) as f32).sqrt();
    let data: Vec<f32> = (0..fan_in * fan_out)
        .map(|i| {
            let x = (i as f32 + 1.0) * 0.618_034;
            (x * PI * 2.0).sin() * scale
        })
        .collect();
    session.set_parameter(name, &data);
}
