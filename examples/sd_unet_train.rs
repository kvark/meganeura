#![allow(dead_code, clippy::too_many_arguments)]
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
#[allow(dead_code, clippy::too_many_arguments)]
use meganeura::{Graph, build_session};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Inlined SD U-Net model
// ---------------------------------------------------------------------------
mod sd_unet {
    use meganeura::graph::{Graph, NodeId, Op};

    pub struct SDUNetConfig {
        pub batch_size: u32,
        pub in_channels: u32,
        pub base_channels: u32,
        pub num_levels: usize,
        pub resolution: u32,
        pub num_groups: u32,
        pub gn_eps: f32,
    }

    impl SDUNetConfig {
        pub fn tiny() -> Self {
            Self {
                batch_size: 4,
                in_channels: 4,
                base_channels: 32,
                num_levels: 3,
                resolution: 32,
                num_groups: 8,
                gn_eps: 1e-5,
            }
        }

        pub fn small() -> Self {
            Self {
                batch_size: 2,
                in_channels: 4,
                base_channels: 64,
                num_levels: 3,
                resolution: 32,
                num_groups: 16,
                gn_eps: 1e-5,
            }
        }

        fn channel_mult(&self) -> Vec<u32> {
            (0..self.num_levels).map(|i| 1u32 << i).collect()
        }
    }

    struct SpatialState {
        h: u32,
        w: u32,
        c: u32,
    }

    fn resblock(
        g: &mut Graph,
        x: NodeId,
        prefix: &str,
        cfg: &SDUNetConfig,
        s: &SpatialState,
        out_c: u32,
    ) -> NodeId {
        let batch = cfg.batch_size;
        let spatial = s.h * s.w;
        let in_c = s.c;

        let gn1_w = g.parameter(&format!("{prefix}.norm1.weight"), &[in_c as usize]);
        let gn1_b = g.parameter(&format!("{prefix}.norm1.bias"), &[in_c as usize]);
        let h = g.group_norm(
            x,
            gn1_w,
            gn1_b,
            batch,
            in_c,
            spatial,
            cfg.num_groups,
            cfg.gn_eps,
        );
        let h = g.silu(h);
        let conv1_w = g.parameter(
            &format!("{prefix}.conv1.weight"),
            &[(out_c * in_c * 9) as usize],
        );
        let h = g.conv2d(h, conv1_w, batch, in_c, s.h, s.w, out_c, 3, 3, 1, 1);

        let gn2_w = g.parameter(&format!("{prefix}.norm2.weight"), &[out_c as usize]);
        let gn2_b = g.parameter(&format!("{prefix}.norm2.bias"), &[out_c as usize]);
        let h = g.group_norm(
            h,
            gn2_w,
            gn2_b,
            batch,
            out_c,
            spatial,
            cfg.num_groups,
            cfg.gn_eps,
        );
        let h = g.silu(h);
        let conv2_w = g.parameter(
            &format!("{prefix}.conv2.weight"),
            &[(out_c * out_c * 9) as usize],
        );
        let h = g.conv2d(h, conv2_w, batch, out_c, s.h, s.w, out_c, 3, 3, 1, 1);

        if in_c == out_c {
            g.add(x, h)
        } else {
            let res_w = g.parameter(
                &format!("{prefix}.res_conv.weight"),
                &[(out_c * in_c) as usize],
            );
            let x_proj = g.conv2d(x, res_w, batch, in_c, s.h, s.w, out_c, 1, 1, 1, 0);
            g.add(x_proj, h)
        }
    }

    pub fn build_training_graph(g: &mut Graph, cfg: &SDUNetConfig) -> NodeId {
        let batch = cfg.batch_size;
        let res = cfg.resolution;
        let in_c = cfg.in_channels;
        let in_size = (batch * in_c * res * res) as usize;
        let ch_mults = cfg.channel_mult();

        let noisy = g.input("noisy_latent", &[in_size]);
        let target = g.input("noise_target", &[in_size]);

        let base_c = cfg.base_channels;
        let conv_in_w = g.parameter("conv_in.weight", &[(base_c * in_c * 3 * 3) as usize]);
        let mut x = g.conv2d(noisy, conv_in_w, batch, in_c, res, res, base_c, 3, 3, 1, 1);

        let mut s = SpatialState {
            h: res,
            w: res,
            c: base_c,
        };

        let mut skip_connections: Vec<(NodeId, SpatialState)> = Vec::new();

        for (level, &mult) in ch_mults.iter().enumerate() {
            let out_c = base_c * mult;

            x = resblock(g, x, &format!("encoder.{level}.resblock"), cfg, &s, out_c);
            s.c = out_c;

            skip_connections.push((
                x,
                SpatialState {
                    h: s.h,
                    w: s.w,
                    c: s.c,
                },
            ));

            if level < cfg.num_levels - 1 {
                let down_w = g.parameter(
                    &format!("encoder.{level}.downsample.weight"),
                    &[(out_c * out_c * 3 * 3) as usize],
                );
                x = g.conv2d(x, down_w, batch, out_c, s.h, s.w, out_c, 3, 3, 2, 1);
                s.h = (s.h + 2 - 3) / 2 + 1;
                s.w = (s.w + 2 - 3) / 2 + 1;
            }
        }

        x = resblock(g, x, "middle.resblock", cfg, &s, s.c);

        for level in (0..cfg.num_levels).rev() {
            let out_c = base_c * ch_mults[level];

            if level < cfg.num_levels - 1 {
                x = g.upsample_2x(x, batch, s.c, s.h, s.w);
                s.h *= 2;
                s.w *= 2;
            }

            let &(skip, ref skip_s) = &skip_connections[level];
            assert_eq!(s.h, skip_s.h, "spatial mismatch at level {level}");
            assert_eq!(s.w, skip_s.w, "spatial mismatch at level {level}");
            let spatial = s.h * s.w;
            x = g.concat(x, skip, batch, s.c, skip_s.c, spatial);
            let concat_c = s.c + skip_s.c;

            let dec_s = SpatialState {
                h: s.h,
                w: s.w,
                c: concat_c,
            };
            x = resblock(
                g,
                x,
                &format!("decoder.{level}.resblock"),
                cfg,
                &dec_s,
                out_c,
            );
            s.c = out_c;
        }

        let gn_out_w = g.parameter("conv_out.norm.weight", &[base_c as usize]);
        let gn_out_b = g.parameter("conv_out.norm.bias", &[base_c as usize]);
        x = g.group_norm(
            x,
            gn_out_w,
            gn_out_b,
            batch,
            base_c,
            res * res,
            cfg.num_groups,
            cfg.gn_eps,
        );
        x = g.silu(x);
        let conv_out_w = g.parameter("conv_out.weight", &[(in_c * base_c * 3 * 3) as usize]);
        let pred = g.conv2d(x, conv_out_w, batch, base_c, res, res, in_c, 3, 3, 1, 1);

        let neg_target = g.neg(target);
        let diff = g.add(pred, neg_target);
        let sq = g.mul(diff, diff);
        g.mean_all(sq)
    }

    pub fn count_params(cfg: &SDUNetConfig) -> usize {
        let mut g = Graph::new();
        let _loss = build_training_graph(&mut g, cfg);
        g.nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::Parameter { .. }))
            .map(|n| n.ty.num_elements())
            .sum()
    }
}
use sd_unet::SDUNetConfig;

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
    let res = cfg.resolution;
    let in_size = (batch * in_c * res * res) as usize;
    let epochs = 3;
    let steps_per_epoch = 50;
    let lr = 1e-3_f32;

    let num_params = sd_unet::count_params(&cfg);
    println!("=== SD U-Net Training Benchmark ===");
    println!(
        "config:     {} ({}×{} latent, batch {}, {} levels, base_ch={})",
        if use_small { "small" } else { "tiny" },
        res,
        res,
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
