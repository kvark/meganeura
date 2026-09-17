//! Gemma 4 GGUF decode: load a HuggingFace GGUF, run Meganeura, optionally
//! profile, and print tok/s next to a llama.cpp command for the same file.
//!
//! ```text
//! cargo run --release --example gemma4 -- \
//!     models/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-Q4_0.gguf
//!
//! MEGANEURA_GPU_TIMING=1 cargo run --release --example gemma4 -- \
//!     model.gguf --profile
//!
//! cargo run --release --example gemma4 -- model.gguf --tune --tune-secs 90
//! ```
//!
//! Set `MEGANEURA_DEVICE_ID` to the discrete GPU's PCI id when an iGPU is
//! also visible.
//!
//! Fair throughput uses a single GPU submission (`set_submission_chunks(1)`).
//! Chunking of 8 is compositor fairness, not llama.cpp's policy.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use meganeura::models::gemma4::{self, Gemma4Config};
use meganeura::profiler::{CaptureOptions, capture_session_profile};
use meganeura::tune::TuneOptions;
use meganeura::{Graph, load_gguf};

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
    let config = Gemma4Config::from_gguf(&gguf).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });
    println!(
        "gemma4: layers={} hidden={} vocab={} swa={} kv_from_start={} ple={}",
        config.num_hidden_layers,
        config.hidden_size,
        config.vocab_size,
        config.sliding_window,
        config.n_kv_from_start,
        config.n_embd_per_layer,
    );

    println!("building decode graph (ctx={max_seq})...");
    let mut g = Graph::new();
    let (logits, _k, _v) = gemma4::build_decode_graph(&mut g, &gguf, &config, max_seq)
        .unwrap_or_else(|e| {
            eprintln!("{e}");
            std::process::exit(1);
        });
    g.set_outputs(vec![logits]);

    println!("compiling...");
    let mut session = meganeura::build(&g, meganeura::SessionConfig::inference_from_env()).0;
    session.set_submission_chunks(1);
    println!(
        "session: {} buffers, {} dispatches, {} groups, {:.1} MiB plan",
        session.plan().buffers.len(),
        session.plan().dispatches.len(),
        session.num_groups(),
        session.plan().buffers.iter().sum::<usize>() as f64 / (1024.0 * 1024.0),
    );

    println!("loading weights...");
    gemma4::load_parameters(&mut session, &gguf).unwrap_or_else(|e| {
        eprintln!("{e}");
        std::process::exit(1);
    });

    if do_tune {
        let tune_secs: u64 = args
            .iter()
            .position(|a| a == "--tune-secs")
            .and_then(|i| args.get(i + 1))
            .and_then(|s| s.parse().ok())
            .unwrap_or(90);
        println!("tune_with GEMV shapes (packed + RmsNorm-folded, {tune_secs}s budget)...");
        let mut options = TuneOptions::default();
        options.max_time = Duration::from_secs(tune_secs);
        // Vocab Q8 GEMV is ~0.4 GiB of packed B; the default 64 MiB scratch
        // would skip the kernel that dominates decode.
        options.max_scratch_bytes = 512 * 1024 * 1024;
        options.max_classes = 32;
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
                let mut last = std::collections::BTreeMap::new();
                for outcome in &report.outcomes {
                    let c = &outcome.class;
                    let key = (
                        format!("{:?}", c.shader),
                        c.m,
                        c.n,
                        c.k,
                        c.gemv_rmsnorm,
                        format!("{:?}", c.weight_format),
                    );
                    last.entry(key)
                        .and_modify(|(_, selected)| *selected = outcome.selected)
                        .or_insert((outcome, outcome.selected));
                }
                for (outcome, selected) in last.values() {
                    let c = &outcome.class;
                    let tag = if *selected != outcome.initial {
                        "pin"
                    } else {
                        "keep"
                    };
                    println!(
                        "  {tag} {:?} {}x{}x{} rms={} fmt={:?}  {:?} -> {:?}",
                        c.shader,
                        c.m,
                        c.n,
                        c.k,
                        c.gemv_rmsnorm,
                        c.weight_format,
                        outcome.initial,
                        selected
                    );
                }
            }
            Err(e) => eprintln!("tune skipped: {e}"),
        }
    }

    let token = 2u32; // BOS
    let (x, ple) = gemma4::gather_token(&gguf, &config, token).unwrap_or_else(|e| {
        eprintln!("gather: {e}");
        std::process::exit(1);
    });
    session.set_input("x", &x);
    if let Some(ref ple_row) = ple {
        session.set_input("ple", ple_row);
    }
    session.set_input_u32("kv_pos", &[8]);
    session.set_input_u32("valid_len", &[1]);

    println!("warmup...");
    for _ in 0..3 {
        session.step();
    }
    session.wait();

    println!("timing {steps} decode steps...");
    let t0 = Instant::now();
    for i in 0..steps {
        session.set_input_u32("kv_pos", &[(8 + i as u32) % max_seq as u32]);
        session.step();
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
            session.set_input("x", &x);
            if let Some(ref ple) = ple {
                session.set_input("ple", ple);
            }
            session.set_input_u32("kv_pos", &[8]);
            session.set_input_u32("valid_len", &[1]);
        },
        CaptureOptions {
            samples: 3,
            unprofiled_median_ms: Some(per),
            ..CaptureOptions::default()
        },
    )
    .expect("capture profile");

    println!(
        "\nprofile: {} dispatches, {} windows",
        profile.plan.dispatch_count, profile.measurement.window_count
    );
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
