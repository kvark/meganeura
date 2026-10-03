//! CPU-side cost of a training step, separated from GPU execution.
//!
//! Items 3 and 4 of `docs/open-items.md` are blocked on the same missing
//! measurement: how much host time a step spends *recording* versus waiting on
//! the GPU. `bench_ci_latency` reports `train_step_median_ms` for a model small
//! enough that the two are indistinguishable, and it reports one number for
//! both together.
//!
//! This separates them, using `Session::record` — which encodes a step into a
//! caller's encoder and does not submit — as the host cost, and `Session::step`
//! as the whole step. The model is a configurable chain of blocks because the
//! interesting quantity is dispatches and parameters, not layers, so this
//! sweeps both.
//!
//! `idle_ms` repeats the encode with the device already idle. If it matches
//! `encode_ms` there is no fence hidden inside `encode_step` and the number is
//! real host work; that is the control, because a per-step rebuild could
//! otherwise be hiding behind a wait.
//!
//! Run:
//!   cargo run --release --example bench_step_cpu

use std::time::{Duration, Instant};

use meganeura::{CoopPolicy, Graph, Mode, Session, SessionConfig};

/// Build a chain of `blocks` independent `matmul + bias + gelu` units.
///
/// Independent rather than sequential so the plan has one barrier group per
/// block — that is what makes dispatch count, and therefore the per-dispatch
/// host work, scale with depth. `width` drives both the dispatch count and the
/// parameter count, which the optimizer's segment table is sized by.
fn build(blocks: usize, width: usize, batch: usize) -> (Graph, Vec<String>) {
    let mut g = Graph::new();
    let mut names = Vec::new();
    // One input of [batch, width]: each block is a batched matmul against a
    // [width, width] weight, so dispatch count scales with depth while the
    // parameter count scales with width squared.
    let x = g.input("x", &[batch, width]);
    let mut current = vec![x];
    for b in 0..blocks {
        let w = format!("w{b}");
        let weight = g.parameter(&w, &[width, width]);
        // A scalar-shaped parameter per block, so the parameter count is driven
        // by depth as well as by width.
        let scale = g.parameter(&format!("s{b}"), &[width]);
        let next: Vec<_> = current
            .iter()
            .map(|&node| {
                let m = g.matmul(node, weight);
                let biased = g.bias_mul(m, scale);
                g.gelu(biased)
            })
            .collect();
        names.push(w);
        current = next;
    }
    // `differentiate` requires a graph ending in a scalar loss, so reduce the
    // block outputs rather than exposing them directly.
    let mut total = current[0];
    for &node in &current[1..] {
        total = g.add(total, node);
    }
    let loss = g.mean_all(total);
    g.set_outputs(vec![loss]);
    (g, names)
}

fn train_session(blocks: usize, width: usize, batch: usize) -> Session {
    let (g, _) = build(blocks, width, batch);
    // `SessionConfig::default()` names no device, so it ignores
    // MEGANEURA_DEVICE_ID and always picks the same adapter. Going through
    // `from_env` honours it, and sharing one context across the sweep keeps
    // the driver's per-process context budget from running out.
    let base = meganeura::SessionConfig::from_env();
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
/// `whole` is an ordinary `Session::step`, which encodes and submits.
///
/// `whole - encode` is what an encoder rotation could in principle overlap
/// with GPU execution — which is the question item 4 asks.
struct Measurement {
    encode: f64,
    /// The same encode with the device already idle: if this matches `encode`
    /// there is no hidden fence in the recording path.
    encode_idle: f64,
    whole: f64,
    dispatches: usize,
    params: usize,
}

fn measure(blocks: usize, width: usize, batch: usize, runs: usize) -> Measurement {
    let mut session = train_session(blocks, width, batch);
    let dispatches = session.plan().dispatches.len();
    let params = session.plan().param_grad_pairs.len();
    session.set_adam(1e-3, 0.9, 0.999, 1e-8);
    let feed: Vec<f32> = vec![0.5; batch * width];

    // An encoder we own, so `record` can encode into it without submitting.
    let gpu = session.context();
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

    // The same encodes against an encoder with nothing in flight, and again
    // with the GPU deliberately idle, to separate a hidden fence from real
    // encoding work. `record` does not call `wait`, so a fence here would be
    // inside `encode_step` -- measuring with the device already idle is the
    // control.
    let mut encode_idle = Vec::with_capacity(runs);
    for _ in 0..runs {
        session.wait();
        encoder.start();
        let start = Instant::now();
        session.record(&mut encoder).expect("record");
        encode_idle.push(start.elapsed());
    }
    let idle = ms(median(encode_idle));

    let mut whole = Vec::with_capacity(runs);
    for _ in 0..runs {
        session.set_input("x", &feed);
        let start = Instant::now();
        session.step();
        whole.push(start.elapsed());
    }
    session.wait();

    Measurement {
        encode: ms(median(encode)),
        encode_idle: idle,
        whole: ms(median(whole)),
        dispatches,
        params,
    }
}

fn main() {
    println!("device: {:?}", {
        let s = train_session(1, 8, 1);
        s.context().device_information().device_name.clone()
    });
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
        "idle_ms",
        "overlappable"
    );
    println!("{}", "-".repeat(72));

    // Sweep dispatch count and parameter count independently: the segment
    // table scales with parameters, the pipeline lookup with dispatches.
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
        let m = measure(blocks, width, batch, 40);
        println!(
            "{:>7} {:>7} {:>5} {:>10} {:>7} {:>11.4} {:>11.4} {:>11.4} {:>12.4}",
            blocks,
            width,
            batch,
            m.dispatches,
            m.params,
            m.whole,
            m.encode,
            m.encode_idle,
            m.whole - m.encode
        );
    }

    println!();
    println!(
        "`encode_ms` is `Session::record`, which encodes without submitting, so\n\
         it is the host cost exactly. `overlappable` is `step_ms - encode_ms`:\n\
         what an encoder rotation could hide behind GPU execution."
    );
}
