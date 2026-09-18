//! Gemma 4 GGUF decode through the generic `load::gguf` path, with
//! explicit step control for benchmarking, and tok/s next to a llama.cpp
//! command for the same file.
//!
//! ```text
//! cargo run --release --features gguf --example gemma4 -- \
//!     models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q4_0.gguf
//!
//! MEGANEURA_GPU_TIMING=1 cargo run --release --features gguf --example gemma4 -- \
//!     model.gguf --profile
//!
//! cargo run --release --features gguf --example gemma4 -- model.gguf --tune --tune-secs 90
//!
//! cargo run --release --features gguf --example gemma4 -- model.gguf --f32-activations
//! ```
//!
//! Quantized models decode with quantized (Q8_1) activations by default;
//! `--f32-activations` opts back out for an A/B.
//!
//! Set `MEGANEURA_DEVICE_ID` to the discrete GPU's PCI id when an iGPU is
//! also visible. Fair throughput uses a single GPU submission
//! (`set_submission_chunks(1)`); chunking of 8 is compositor fairness, not
//! llama.cpp's policy.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use meganeura::load::gguf::{self, ModelConfig};
use meganeura::profiler::{CaptureOptions, capture_session_profile};
use meganeura::tune::TuneOptions;
use meganeura::{Graph, Session, load_gguf};

/// One decode step: a single real token at `position`, with the
/// architecture's per-layer embedding rows gathered on the host.
fn step(
    session: &mut Session,
    gguf: &gguf::GgufModel,
    config: &ModelConfig,
    token: u32,
    position: u32,
) {
    session.set_input_u32("token_ids", &[token]);
    session.set_input_u32("position", &[position]);
    session.set_input_u32("valid", &[1]);
    if config.architecture.uses_per_layer_embeddings() {
        let ple = gguf::weights::gather_per_layer_embeddings(
            gguf.tensors
                .get("per_layer_token_embd.weight")
                .expect("per-layer embedding table"),
            config,
            &[token],
        )
        .expect("per-layer embedding gather");
        session.set_input("ple", &ple);
    }
    session.step();
}

fn main() {
    env_logger::init();

    let args: Vec<String> = std::env::args().collect();
    let path = args
        .iter()
        .find(|a| a.ends_with(".gguf"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q4_0.gguf"));
    let do_profile = args.iter().any(|a| a == "--profile");
    let do_tune = args.iter().any(|a| a == "--tune");
    let f32_activations = args.iter().any(|a| a == "--f32-activations");
    let steps: usize = args
        .iter()
        .position(|a| a == "--steps")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    let max_seq: usize = args
        .iter()
        .position(|a| a == "--ctx")
        .and_then(|i| args.get(i + 1))
        .and_then(|s| s.parse().ok())
        .unwrap_or(256);

    if do_profile && std::env::var_os("MEGANEURA_GPU_TIMING").is_none() {
        eprintln!("--profile needs MEGANEURA_GPU_TIMING=1 before process start");
        std::process::exit(2);
    }

    println!("loading {}...", path.display());
    let gguf = load_gguf(&path).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });
    let config = ModelConfig::from_gguf(&gguf).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });
    let mut graph = Graph::new();
    let built = gguf::graph::build(&mut graph, &gguf, &config, 1, max_seq).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });

    println!("compiling...");
    let mut session_config = meganeura::SessionConfig::inference_from_env();
    if f32_activations {
        session_config.options.quantized_activations = false;
    }
    graph.set_outputs(built.outputs());
    let (mut session, _) = meganeura::train::build(
        &graph,
        meganeura::train::SessionConfig {
            mode: meganeura::train::Mode::Inference,
            ..session_config
        },
    );
    session.set_submission_chunks(1);
    println!(
        "activations: {}",
        if f32_activations {
            "f32"
        } else {
            "Q8_1 int-dot"
        }
    );
    println!(
        "session: {} buffers, {} dispatches, {} groups, {:.1} MiB plan",
        session.plan().buffers.len(),
        session.plan().dispatches.len(),
        session.num_groups(),
        session.plan().buffers.iter().sum::<usize>() as f64 / (1024.0 * 1024.0),
    );

    println!("loading weights...");
    gguf::weights::load(&mut session, &gguf, &config).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });

    let token = 2u32; // BOS

    if do_tune {
        let tune_secs: u64 = args
            .iter()
            .position(|a| a == "--tune-secs")
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse().ok())
            .unwrap_or(90);
        println!("tune_with GEMV shapes (packed + RmsNorm-folded, {tune_secs}s budget)...");
        // Vocab Q8 GEMV is ~0.4 GiB of packed B; the default 64 MiB scratch
        // would skip the kernel that dominates decode.
        let options = TuneOptions {
            max_time: Duration::from_secs(tune_secs),
            max_scratch_bytes: 512 * 1024 * 1024,
            max_classes: 32,
            ..Default::default()
        };
        match session.tune_with(options) {
            Ok(report) => {
                println!(
                    "tune: visited {}/{} classes, {} outcomes, {} excluded, elapsed {:?}",
                    report.visited_classes,
                    report.eligible_classes,
                    report.outcomes.len(),
                    report.excluded_dispatches,
                    report.elapsed
                );
            }
            Err(e) => eprintln!("tune skipped: {e}"),
        }
    }

    println!("warmup...");
    for _ in 0..3 {
        step(&mut session, &gguf, &config, token, 8);
    }
    session.wait();

    println!("timing {steps} decode steps...");
    let t0 = Instant::now();
    for i in 0..steps {
        step(
            &mut session,
            &gguf,
            &config,
            token,
            (8 + i as u32) % max_seq as u32,
        );
    }
    session.wait();
    let ms = t0.elapsed().as_secs_f64() * 1000.0;
    let per = ms / steps as f64;
    println!(
        "meganeura: {per:.2} ms/tok  ({:.2} tok/s)  chunks=1",
        1000.0 / per
    );
    println!(
        "llama.cpp (same file, Vulkan, one stream):\n  \
         llama-bench -m {} -ngl 99 -p 64 -n 128 -b 1 -ub 1",
        path.display()
    );

    if !do_profile {
        return;
    }

    let profile = capture_session_profile(
        &mut session,
        |session| {
            step(session, &gguf, &config, token, 8);
        },
        CaptureOptions {
            samples: 3,
            unprofiled_median_ms: Some(per),
            ..CaptureOptions::default()
        },
    )
    .expect("capture profile");

    println!("\nprofile: {} dispatches", profile.plan.dispatch_count);
    println!("=== GPU time by kernel family ===");
    let mut families = profile.families.clone();
    families.sort_by(|a, b| {
        b.dispatch_median_sum_ms
            .total_cmp(&a.dispatch_median_sum_ms)
    });
    for family in families.iter().take(12) {
        println!(
            "  {:>10} {:>22}: {:>4}x {:>8.3}ms ({:>5.1}%)",
            family.phase,
            family.family,
            family.dispatch_count,
            family.dispatch_median_sum_ms,
            family.share_of_dispatch_median_sum_pct,
        );
    }
    println!("=== costliest dispatches ===");
    let mut dispatches = profile.dispatches.clone();
    dispatches.sort_by(|a, b| b.median_ms.total_cmp(&a.median_ms));
    for d in dispatches.iter().take(20) {
        println!(
            "  #{:<5} {:>8.3}ms ({:>4.1}%) {} [{}]",
            d.index, d.median_ms, d.share_of_dispatch_median_sum_pct, d.label, d.pipeline,
        );
    }
}
