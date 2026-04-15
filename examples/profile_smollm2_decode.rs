//! Profile a single SmolLM2-135M decode step with blade GPU timestamps.
//!
//! Builds the decode graph, fills weights with small random data (perf is
//! independent of weight values), then runs with per-pass profiling enabled.
//! Prints the per-pass GPU timing breakdown aggregated by shader type, plus
//! total GPU time vs wall time so we can see how much is launch/submission
//! overhead vs actual kernel work.
//!
//! Usage:
//!   cargo run --release --example profile_smollm2_decode

use std::time::Instant;

use meganeura::{Graph, build_inference_session, models::smollm2};

fn main() {
    env_logger::init();

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
    let mut session = build_inference_session(&g);
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

    eprintln!("\nenabling profiling (one pass per dispatch)...");
    session.set_profiling(true);

    // 3-step dance: step A records, step B advances ring, step C's
    // start() reads step A's timestamps.
    session.step();
    session.wait();
    session.step();
    session.wait();
    session.step();
    eprintln!("\n=== GPU pass timings for a single decode step ===");
    session.dump_gpu_timings();
    session.wait();

    eprintln!(
        "\nnote: baseline wall-time per step = {:.2}ms. Compare to sum of GPU \
         pass timings above to localize overhead vs kernel time.",
        baseline_per_step,
    );
}
