//! Profile a single SmolLM2-135M decode step with blade GPU timestamps.
//!
//! Builds the decode graph, fills weights with small random data (perf is
//! independent of weight values), then captures a structured profile. Prints
//! the timing breakdown by phase and kernel family and the costliest
//! individual dispatches, plus total GPU time vs wall time so we can see how
//! much is launch/submission overhead vs actual kernel work.
//!
//! A decode step is over a thousand dispatches, more than Blade timestamps in
//! one submission, so the capture replays it once per window of dispatches
//! and stitches the results — see `docs/performance-profiling.md`.
//!
//! Usage:
//!   MEGANEURA_GPU_TIMING=1 cargo run --release --example profile_smollm2_decode

use std::time::Instant;

use meganeura::profiler::{CaptureOptions, capture_session_profile};
use meganeura::{Graph, models::smollm2};

fn main() {
    env_logger::init();

    if std::env::var_os("MEGANEURA_GPU_TIMING").is_none() {
        eprintln!("set MEGANEURA_GPU_TIMING=1 before starting the example");
        std::process::exit(2);
    }

    let config = smollm2::SmolLM2Config::smollm2_135m();
    let max_seq_len = 128;

    eprintln!(
        "building SmolLM2-135M decode graph (max_seq_len={})",
        max_seq_len
    );
    let mut g = Graph::new();
    let (logits, _k_caches, _v_caches) = smollm2::build_decode_graph(&mut g, &config, max_seq_len);
    g.set_outputs(vec![logits]);

    eprintln!("compiling...");
    let mut session = meganeura::build(&g, meganeura::SessionConfig::inference_from_env()).0;
    eprintln!(
        "decode: {} buffers, {} dispatches, {} barrier groups",
        session.plan().buffers.len(),
        session.plan().dispatches.len(),
        session.num_groups(),
    );

    // Fill params with small pseudo-random data so the kernels have
    // representative workloads (zeros would produce NaN through softmax
    // after exp(x - max) = exp(0 - 0) = 1, fine actually, but set
    // small values for realism).
    eprintln!("filling weights with small random values...");
    let param_buffers = session.plan().param_buffers.clone();
    for (name, buf_ref) in &param_buffers {
        let n = session.plan().buffers[buf_ref.0 as usize] / 4;
        let data: Vec<f32> = (0..n)
            .map(|i| {
                let x = (i as u32).wrapping_mul(2654435761) as f32 / (1u32 << 31) as f32;
                x * 0.02
            })
            .collect();
        session.set_parameter(name, &data);
    }
    // Inputs: a token id and a kv_pos. The kv_pos of 10 is arbitrary.
    session.set_input_u32("token_ids", &[42]);
    session.set_input_u32("kv_pos", &[10]);

    eprintln!("warming up (5 runs)...");
    for _ in 0..5 {
        session.step();
    }
    session.wait();

    eprintln!("measuring baseline wall time (no profiling, 20 runs)...");
    let t0 = Instant::now();
    for _ in 0..20 {
        session.step();
    }
    session.wait();
    let baseline_per_step = t0.elapsed().as_secs_f64() * 1000.0 / 20.0;
    eprintln!("  baseline: {:.2}ms / decode step", baseline_per_step);

    eprintln!("\ncapturing a structured profile (one pass per timed dispatch)...");
    let profile = capture_session_profile(
        &mut session,
        |session| {
            session.set_input_u32("token_ids", &[42]);
            session.set_input_u32("kv_pos", &[10]);
        },
        CaptureOptions {
            samples: 3,
            unprofiled_median_ms: Some(baseline_per_step),
            ..CaptureOptions::default()
        },
    )
    .expect("capture structured GPU profile");

    let measurement = &profile.measurement;
    eprintln!(
        "  {} dispatches over {} window(s) x {} sample(s), \
         largest window {} dispatches",
        profile.plan.dispatch_count,
        measurement.window_count,
        measurement.sample_count,
        measurement.max_window_dispatches,
    );

    eprintln!("\n=== GPU time by kernel family ===");
    let mut families = profile.families.clone();
    families.sort_by(|a, b| {
        b.dispatch_median_sum_ms
            .total_cmp(&a.dispatch_median_sum_ms)
    });
    for family in families.iter().take(12) {
        eprintln!(
            "  {:>10} {:>18}: {:>4}x {:>8.3}ms ({:>5.1}%)",
            family.phase,
            family.family,
            family.dispatch_count,
            family.dispatch_median_sum_ms,
            family.share_of_dispatch_median_sum_pct,
        );
    }

    eprintln!("\n=== costliest individual dispatches ===");
    let mut dispatches = profile.dispatches.clone();
    dispatches.sort_by(|a, b| b.median_ms.total_cmp(&a.median_ms));
    for dispatch in dispatches.iter().take(15) {
        eprintln!(
            "  #{:<5} {:>8.3}ms ({:>4.1}%) {} [{}]",
            dispatch.index,
            dispatch.median_ms,
            dispatch.share_of_dispatch_median_sum_pct,
            dispatch.label,
            dispatch.pipeline,
        );
    }

    eprintln!(
        "\ntotal timestamped GPU time: {:.2}ms (median over samples)",
        measurement.gpu_total_median_ms,
    );
    eprintln!(
        "baseline wall-time per step: {:.2}ms. The gap is launch and \
         submission overhead, not kernel work.",
        baseline_per_step,
    );
    if measurement.window_count > 1 {
        eprintln!(
            "profiled wall time is not comparable to the baseline here: each \
             replay timed only {} of {} dispatches and ran the rest at normal \
             pass counts.",
            measurement.max_window_dispatches, profile.plan.dispatch_count,
        );
    }
}
