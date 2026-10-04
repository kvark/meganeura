//! Isolated complete-sequence evidence, never installation or whole-step timing.
#[path = "support/experiment_io.rs"]
mod experiment_io;
#[path = "support/gpu_monitor.rs"]
mod gpu_monitor;
#[path = "support/tuning_measurement.rs"]
#[allow(dead_code)]
mod measurement;

use experiment_io::{command, host_sample, sha256, unix_ms, write_record};
use measurement::TensorComparison;
use meganeura::{CoopPolicy, Graph, Session, SessionOptions, TuneOptions, compile::ShaderEntry};
use serde_json::{Value, json};
use std::{error::Error, fs::OpenOptions, path::Path, sync::Arc, time::Duration};

const CASES: [(&str, [u32; 10]); 4] = [
    ("spatial-7x7", [1, 3, 224, 224, 64, 7, 7, 2, 3, 3]),
    ("pointwise", [1, 256, 56, 56, 64, 1, 1, 1, 0, 0]),
    ("rectangular-tail", [3, 5, 7, 9, 7, 2, 3, 1, 1, 0]),
    ("long-tail", [2, 3, 1, 32771, 5, 1, 1, 1, 0, 0]),
];

fn data(count: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            (state >> 8) as f32 / 16777216.0 - 0.5
        })
        .collect()
}

fn snapshot(session: &Session) -> Vec<(String, Vec<f32>)> {
    let mut state = Vec::new();
    for (name, buffer) in &session.plan().input_buffers {
        let mut values = vec![f32::NAN; session.plan().buffers[buffer.0 as usize] / 4];
        session.read_buffer(*buffer, &mut values);
        state.push((format!("input.{name}"), values));
    }
    let n = session.param_size("w").unwrap();
    let mut w = vec![f32::NAN; n];
    let mut dw = vec![f32::NAN; n];
    session.read_param("w", &mut w);
    session.read_param_grad("w", &mut dw);
    state.push(("parameter.w".into(), w));
    state.push(("gradient.w".into(), dw));
    let loss = session.plan().loss_buffer.unwrap();
    let mut partials = vec![f32::NAN; session.plan().buffers[loss.0 as usize] / 4];
    session.read_buffer(loss, &mut partials);
    state.push(("loss_partials".into(), partials));
    state.push(("loss".into(), vec![session.read_loss()]));
    state
}

fn run_case(
    index: usize,
    seed: usize,
    gpu: &Arc<blade_graphics::Context>,
) -> Result<Value, Box<dyn Error>> {
    let (name, shape) = CASES[index];
    let [batch, ci, h, w, co, kh, kw, stride, ph, pw] = shape;
    let (oh, ow) = (
        (h + 2 * ph - kh) / stride + 1,
        (w + 2 * pw - kw) / stride + 1,
    );
    let (nx, nw, ny) = (
        (batch * ci * h * w) as usize,
        (co * ci * kh * kw) as usize,
        (batch * co * oh * ow) as usize,
    );
    let mut graph = Graph::new();
    let input = graph.input("x", &[nx]);
    let weight = graph.parameter("w", &[nw]);
    let upstream = graph.input("dy", &[ny]);
    let output = graph.conv2d_hw(input, weight, batch, ci, h, w, co, kh, kw, stride, ph, pw);
    let weighted = graph.mul(output, upstream);
    let loss = graph.sum_all(weighted);
    graph.set_outputs(vec![loss]);
    let plan = meganeura::compile::compile(&meganeura::autodiff::differentiate(&graph));
    let mut session = Session::with_context_opts(
        plan,
        Arc::clone(gpu),
        SessionOptions {
            coop: CoopPolicy::Disabled,
            ..Default::default()
        },
    );
    session.set_input("x", &data(nx, seed as u32 * 7));
    session.set_input("dy", &data(ny, seed as u32 * 37));
    session.set_parameter("w", &data(nw, seed as u32 * 19));
    session.step();
    session.wait();
    let before = snapshot(&session);
    if before.iter().any(|(_, values)| {
        values.iter().any(|v| !v.is_finite()) || !values.iter().any(|&v| v != 0.0)
    }) {
        return Err("nonfinite or zero preflight signal".into());
    }
    let dispatch_index = session
        .plan()
        .dispatches
        .iter()
        .position(|d| {
            matches!(
                d.shader,
                ShaderEntry::Conv2dGradWeightGemm | ShaderEntry::Conv2dGradWeightGemmSmall
            )
        })
        .ok_or("missing scalar dW")?;
    let plan_before = serde_json::to_value(session.plan())?;
    let keys = session.dispatch_pipeline_keys();
    let memory = session.memory_summary().total_allocated_bytes();
    let mut splits = [2, 3, 4, 8];
    splits.rotate_left(seed - 1);
    let options = TuneOptions {
        max_time: Duration::from_secs(120),
        ..Default::default()
    };
    let start = host_sample();
    let report = session.measure_conv_weight_splits(dispatch_index, &splits, options)?;
    let finish = host_sample();
    let after = snapshot(&session);
    let comparisons: Vec<_> = before
        .iter()
        .zip(&after)
        .map(|((name, a), (other, b))| {
            assert_eq!(name, other);
            TensorComparison::new(name.clone(), a, b)
        })
        .collect();
    let unchanged = comparisons.iter().all(|c| c.exact && c.passed)
        && plan_before == serde_json::to_value(session.plan())?
        && keys == session.dispatch_pipeline_keys()
        && memory == session.memory_summary().total_allocated_bytes()
        && session.adam_step_count() == 0
        && session.memory_summary().adam_state_bytes == 0;
    eprintln!(
        "{name}: unchanged={unchanged}; {:?}",
        report
            .outcomes
            .iter()
            .map(|o| (
                o.candidate_split_k,
                o.decision,
                o.baseline_median_ms,
                o.candidate_median_ms,
                &o.failure
            ))
            .collect::<Vec<_>>()
    );
    Ok(
        json!({"name": name, "shape": shape, "status": if unchanged {"complete"} else {"state_failed"},
        "host_start": start, "host_finish": finish, "dispatch_index": dispatch_index,
        "dispatch": session.plan().dispatches[dispatch_index], "splits": splits, "report": report,
        "state_comparisons": comparisons, "plan_and_state_unchanged": unchanged,
        "pipeline_keys": keys, "resident_buffer_requests": memory, "adam_step": session.adam_step_count(),
        "adam_bytes": session.memory_summary().adam_state_bytes}),
    )
}

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    let mut args = std::env::args_os().skip(1);
    let path = args
        .next()
        .ok_or("usage: measure_split_k <new-output.json> <seed 1..4>")?;
    let seed: usize = args
        .next()
        .ok_or("missing seed")?
        .to_str()
        .ok_or("non-UTF8 seed")?
        .parse()?;
    if !(1..=4).contains(&seed) || args.next().is_some() {
        return Err("expected seed 1..4".into());
    }
    if !command("git", &["status", "--porcelain", "--untracked-files=no"])?.is_empty() {
        return Err("commit tracked source before measuring".into());
    }
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    let monitor = gpu_monitor::Monitor::start();
    let gpu = Arc::new(meganeura::init_gpu_context()?);
    let info = gpu.device_information();
    let mut record = json!({"protocol": "split-k-sequence-v1", "status": "running", "metadata": {
        "revision": command("git", &["rev-parse", "HEAD"])?, "tracked_source_clean": true,
        "executable_sha256": sha256(&std::env::current_exe()?)?,
        "cargo_lock_sha256": sha256(&Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.lock"))?,
        "rustc": command("rustc", &["--version"])?, "rustflags": std::env::var("RUSTFLAGS").ok(),
        "device": info.device_name, "driver": info.driver_info, "seed": seed,
        "started_unix_ms": unix_ms(), "host_start": host_sample(), "nvidia_smi_before": command("nvidia-smi", &[]).ok(),
        "contract": "strict f32; synthetic scratch; full final/partial f64; no live installation or whole-step timing"
    }, "cases": []});
    write_record(&mut file, &record)?;
    for position in 0..4 {
        let index = (position + seed - 1) % 4;
        let result = run_case(index, seed, &gpu).unwrap_or_else(|error| {
            json!({
                "name": CASES[index].0, "status": "error", "error": error.to_string()
            })
        });
        record["cases"].as_array_mut().unwrap().push(result);
        write_record(&mut file, &record)?;
    }
    record["telemetry"] = monitor.finish();
    record["finished_unix_ms"] = json!(unix_ms());
    let complete = record["cases"]
        .as_array()
        .unwrap()
        .iter()
        .all(|c| c["status"] == "complete");
    record["status"] = json!(if complete { "complete" } else { "failed" });
    write_record(&mut file, &record)?;
    if !complete {
        return Err("failed case retained in record".into());
    }
    Ok(())
}
