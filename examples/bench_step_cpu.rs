//! CPU-side cost of a training step, separated from GPU execution.
//!
//! Supports the host-cost measurements in items 3 and 4 of `docs/open-items.md`.
//! `Session::record` encodes a step into a caller's encoder without submitting,
//! measuring host cost. `Session::step` followed by `Session::wait` measures the
//! whole step. A configurable chain of blocks varies the dispatch and parameter
//! counts.
//!
//! Encoding-only samples run with the device idle. Whole-step samples report
//! the recording/submission call and the following completion wait separately.
//! These timings do not measure the benefit of overlapping CPU and GPU work;
//! that needs a comparison with an encoder rotation.
//!
//! Run:
//!   cargo run --release --example bench_step_cpu

use std::{
    sync::Arc,
    time::{Duration, Instant},
};

use meganeura::{CoopPolicy, Graph, Mode, Session, SessionConfig};

/// Build a chain of `blocks` sequential `matmul + bias_mul + gelu` units.
fn build(blocks: usize, width: usize, batch: usize) -> Graph {
    let mut g = Graph::new();
    // One input of [batch, width]: each block is a batched matmul against a
    // [width, width] weight, so dispatch count scales with depth while the
    // number of parameter elements scales with width squared.
    let mut current = g.input("x", &[batch, width]);
    for b in 0..blocks {
        let weight = g.parameter(&format!("w{b}"), &[width, width]);
        // One scaling vector per block.
        let scale = g.parameter(&format!("s{b}"), &[width]);
        let m = g.matmul(current, weight);
        let biased = g.bias_mul(m, scale);
        current = g.gelu(biased);
    }
    // `differentiate` requires a graph ending in a scalar loss, so reduce the
    // final block output.
    let loss = g.mean_all(current);
    g.set_outputs(vec![loss]);
    g
}

fn train_session(
    gpu: &Arc<blade_graphics::Context>,
    blocks: usize,
    width: usize,
    batch: usize,
) -> Session {
    let g = build(blocks, width, batch);
    let base = SessionConfig::from_env_with_gpu(Some(Arc::clone(gpu)));
    let (session, _) = meganeura::build(
        &g,
        SessionConfig {
            mode: Mode::Training,
            runtime: meganeura::SessionOptions {
                coop: CoopPolicy::Disabled,
                ..base.runtime
            },
            ..base
        },
    );
    session
}

/// Median of `samples` timings of one closure.
fn median(mut samples: Vec<Duration>) -> Duration {
    samples.sort_unstable();
    samples[samples.len() / 2]
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

/// Time host-side encoding separately from the whole step.
///
/// `Session::record` encodes a step into a caller-supplied encoder and does
/// not submit, so it times exactly the host work: walking the dispatches,
/// resolving each pipeline, building its binding struct and issuing the
/// dispatch call. Nothing is submitted, so there is no fence and no queueing.
///
/// `whole` covers `Session::step` and the completion wait. `step_call` includes
/// recording and submission; `wait` includes the remaining device work and
/// host synchronization, so it is not a hardware timestamp measurement.
struct Measurement {
    encode: f64,
    whole: f64,
    step_call: f64,
    wait: f64,
    dispatches: usize,
    params: usize,
}

fn measure(
    gpu: &Arc<blade_graphics::Context>,
    blocks: usize,
    width: usize,
    batch: usize,
    runs: usize,
) -> Measurement {
    let mut session = train_session(gpu, blocks, width, batch);
    let dispatches = session.plan().dispatches.len();
    let params = session.plan().param_grad_pairs.len();
    session.set_adam(1e-3, 0.9, 0.999, 1e-8);
    let feed: Vec<f32> = vec![0.5; batch * width];

    // An encoder we own, so `record` can encode into it without submitting.
    let mut encoder = gpu.create_command_encoder(blade_graphics::CommandEncoderDesc {
        name: "bench_step_cpu",
        buffer_count: 2,
        manual_barriers: false,
    });

    // Warm up: pipeline creation and shader compilation land on the first
    // steps and would otherwise dominate.
    for _ in 0..5 {
        session.set_input("x", &feed);
        session.step();
    }
    session.wait();

    let mut encode = Vec::with_capacity(runs);
    for _ in 0..runs {
        encoder.start();
        let start = Instant::now();
        session.record(&mut encoder).expect("record");
        encode.push(start.elapsed());
    }

    let mut whole = Vec::with_capacity(runs);
    let mut step_call = Vec::with_capacity(runs);
    let mut wait = Vec::with_capacity(runs);
    for _ in 0..runs {
        session.set_input("x", &feed);
        let start = Instant::now();
        session.step();
        let submitted = start.elapsed();
        session.wait();
        let completed = start.elapsed();
        step_call.push(submitted);
        wait.push(completed - submitted);
        whole.push(completed);
    }
    gpu.destroy_command_encoder(&mut encoder);

    Measurement {
        encode: ms(median(encode)),
        whole: ms(median(whole)),
        step_call: ms(median(step_call)),
        wait: ms(median(wait)),
        dispatches,
        params,
    }
}

fn main() {
    let gpu = Arc::new(
        meganeura::init_gpu_context_with(meganeura::GpuOptions::from_env()).expect("GPU context"),
    );
    println!("device: {:?}", gpu.device_information().device_name);
    println!("(set MEGANEURA_DEVICE_ID to choose the adapter)");
    println!();
    println!(
        "{:>7} {:>7} {:>5} {:>10} {:>7} {:>11} {:>11} {:>11} {:>12}",
        "blocks",
        "width",
        "batch",
        "dispatches",
        "params",
        "step_ms",
        "encode_ms",
        "step_call_ms",
        "wait_ms"
    );
    println!("{}", "-".repeat(89));

    // Vary depth, matrix width and batch size. Depth changes both dispatch and
    // parameter counts; width changes parameter sizes at a fixed count.
    let cases = [
        (4usize, 64usize, 1usize),
        (8, 64, 1),
        (16, 64, 1),
        (32, 64, 1),
        (8, 256, 1),
        (8, 1024, 1),
        (16, 1024, 1),
        (8, 256, 4),
    ];
    for (blocks, width, batch) in cases {
        let m = measure(&gpu, blocks, width, batch, 40);
        println!(
            "{:>7} {:>7} {:>5} {:>10} {:>7} {:>11.4} {:>11.4} {:>11.4} {:>12.4}",
            blocks, width, batch, m.dispatches, m.params, m.whole, m.encode, m.step_call, m.wait
        );
    }

    println!();
    println!(
        "`encode_ms` times `Session::record` on an idle device. `step_ms` times\n\
         `Session::step` plus `Session::wait`, excluding input upload.\n\
         `step_call_ms` includes recording/submission; `wait_ms` is the following\n\
         completion wait, including host synchronization. Columns are separate\n\
         medians and need not add up. Overlap savings require a separate experiment."
    );
}
