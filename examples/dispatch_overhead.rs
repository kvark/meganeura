//! Measure Vulkan dispatch + barrier overhead on NVIDIA.
//!
//! Builds a graph with N chained trivial ops (relu on 1024 f32) and
//! measures wall time as N grows. The slope of time-vs-N is the per-
//! dispatch GPU-side overhead (kernel compute is ~constant and tiny
//! at this size, so any slope is fixed launch/barrier cost).
//!
//! Also runs an "independent" variant where all N ops read from the
//! same input but write to different outputs — meganeura's group builder
//! can schedule them in a single barrier group, so the difference
//! between chained and independent isolates barrier cost from
//! kernel-launch cost.

use std::time::Instant;

use meganeura::{CompileOptions, Graph, NodeId, build_inference_session_with};

fn raw_opts() -> CompileOptions {
    // Turn off the pointwise-fusion pass so a chain of unary ops stays as
    // N dispatches — what we want to measure here.
    CompileOptions {
        use_schedule_pointwise: false,
        ..Default::default()
    }
}

fn bench_chain(n_ops: usize, iters: usize) -> (f64, usize, usize) {
    let size = 1024;
    let mut g = Graph::new();
    let x = g.input("x", &[size]);
    // Alternate relu/neg so the optimizer can't collapse the chain
    // (relu(relu(x)) = relu(x), but relu(neg(relu(x))) ≠ relu(x)).
    let mut cur = x;
    for i in 0..n_ops {
        cur = if i % 2 == 0 { g.neg(cur) } else { g.relu(cur) };
    }
    g.set_outputs(vec![cur]);

    let mut s = build_inference_session_with(&g, &raw_opts());
    let n_dispatches = s.plan().dispatches.len();
    let n_groups = s.num_groups();

    let data = vec![1.0_f32; size];
    s.set_input("x", &data);

    // Warmup — lets the driver stabilize and any JIT finish.
    for _ in 0..10 {
        s.step();
    }
    s.wait();

    let t0 = Instant::now();
    for _ in 0..iters {
        s.step();
    }
    s.wait();
    let total = t0.elapsed().as_secs_f64();
    let per_step_us = total / iters as f64 * 1e6;
    (per_step_us, n_dispatches, n_groups)
}

fn bench_independent(n_ops: usize, iters: usize) -> (f64, usize, usize) {
    let size = 1024;
    let mut g = Graph::new();
    let x = g.input("x", &[size]);
    // N independent relu ops on the same input — no dataflow dependency
    // between them, so meganeura's group builder can put all N in the
    // same barrier group.
    let outputs: Vec<NodeId> = (0..n_ops).map(|_| g.relu(x)).collect();
    g.set_outputs(outputs);

    let mut s = build_inference_session_with(&g, &raw_opts());
    let n_dispatches = s.plan().dispatches.len();
    let n_groups = s.num_groups();

    let data = vec![1.0_f32; size];
    s.set_input("x", &data);

    for _ in 0..10 {
        s.step();
    }
    s.wait();

    let t0 = Instant::now();
    for _ in 0..iters {
        s.step();
    }
    s.wait();
    let total = t0.elapsed().as_secs_f64();
    let per_step_us = total / iters as f64 * 1e6;
    (per_step_us, n_dispatches, n_groups)
}

fn main() {
    env_logger::init();

    // Sanity: device name so the reader knows what we're measuring.
    let mut g = Graph::new();
    let x = g.input("x", &[4]);
    let y = g.relu(x);
    g.set_outputs(vec![y]);
    let sanity = build_inference_session_with(&g, &raw_opts());
    eprintln!("device: {}", sanity.device_information().device_name);
    drop(sanity);

    println!();
    println!("chained (N-op serial chain — each relu feeds the next, forcing N barrier groups):",);
    println!(
        "{:<8} {:>10} {:>10} {:>12} {:>12} {:>12}",
        "N", "dispatches", "groups", "µs/step", "µs/dispatch", "µs/group"
    );
    let iters = 400;
    for &n in &[1usize, 5, 10, 20, 50, 100, 200, 500] {
        let (us_per_step, nd, ng) = bench_chain(n, iters);
        let per_dispatch = us_per_step / nd as f64;
        let per_group = us_per_step / ng as f64;
        println!(
            "{:<8} {:>10} {:>10} {:>12.2} {:>12.3} {:>12.3}",
            n, nd, ng, us_per_step, per_dispatch, per_group,
        );
    }

    println!();
    println!("independent (N relu ops on same input — merge into 1 barrier group):",);
    println!(
        "{:<8} {:>10} {:>10} {:>12} {:>12} {:>12}",
        "N", "dispatches", "groups", "µs/step", "µs/dispatch", "µs/group"
    );
    for &n in &[1usize, 5, 10, 20, 50, 100, 200, 500] {
        let (us_per_step, nd, ng) = bench_independent(n, iters);
        let per_dispatch = us_per_step / nd as f64;
        let per_group = us_per_step / ng as f64;
        println!(
            "{:<8} {:>10} {:>10} {:>12.2} {:>12.3} {:>12.3}",
            n, nd, ng, us_per_step, per_dispatch, per_group,
        );
    }

    println!();
    println!("Slope of chained.µs/step = kernel + barrier + dispatch overhead");
    println!("Slope of independent.µs/step = kernel + dispatch overhead (no inter-op barriers)");
    println!("Difference ≈ per-barrier cost.");
}
