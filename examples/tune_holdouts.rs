//! Whole-step inference/training holdouts for the existing isolated tuner.
//! See docs/experiments/holdouts-2026-09-06/README.md for the fixed protocol.

#[path = "support/tuning_measurement.rs"]
mod measurement;

use measurement::{PairedTiming, TensorComparison};
use meganeura::{
    CoopPolicy, Graph, Mode, Session, SessionConfig, SessionOptions, TuneOptions,
    compile::CompileOptions,
    graph::{NodeId, Op},
};
use serde::Serialize;
use serde_json::{Value, json};
use std::{
    collections::{HashMap, HashSet},
    error::Error,
    fs::{File, OpenOptions},
    io::{Seek, SeekFrom, Write},
    path::Path,
    process::Command,
    sync::Arc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

const PREFIX_STEPS: usize = 3;
const WARMUPS: usize = 30;
const SETTLING_STEPS: usize = 5;
const SAMPLE_PAIRS: usize = 40;

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
enum Work {
    Inference,
    ForwardLossBackward,
    Sgd,
    Adam,
}

impl Work {
    fn adam_step_after(self, steps: usize) -> u32 {
        if self == Self::Adam { steps as u32 } else { 0 }
    }
}

enum Input {
    F32(&'static str, Vec<f32>),
    U32(&'static str, Vec<u32>),
}

struct Case {
    name: &'static str,
    description: Value,
    work: Work,
    graph: Graph,
    inputs: Vec<Input>,
    output_elements: usize,
}

fn data(count: usize, seed: u32, scale: f32) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(747796405).wrapping_add(2891336453);
            let word = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277803737);
            (((((word >> 22) ^ word) >> 8) as f32 / 16_777_216.0) - 0.5) * scale
        })
        .collect()
}

fn labels(rows: usize, classes: usize) -> Vec<f32> {
    let mut values = vec![0.0; rows * classes];
    for row in 0..rows {
        values[row * classes + (row + 1) % classes] = 1.0;
    }
    values
}

fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for work in [Work::Inference, Work::Adam] {
        let mut graph = Graph::new();
        let x = graph.input("input", &[127, 384]);
        let w1 = graph.parameter("first.weight", &[384, 640]);
        let b1 = graph.parameter("first.bias", &[640]);
        let h = graph.matmul(x, w1);
        let h = graph.bias_add(h, b1);
        let h = graph.gelu(h);
        let w2 = graph.parameter("second.weight", &[640, 256]);
        let b2 = graph.parameter("second.bias", &[256]);
        let y = graph.matmul(h, w2);
        let y = graph.bias_add(y, b2);
        let mut inputs = vec![Input::F32("input", data(127 * 384, 1, 2.0))];
        let output = if work == Work::Adam {
            let labels = graph.input("labels", &[127, 256]);
            inputs.push(Input::F32("labels", self::labels(127, 256)));
            graph.cross_entropy_loss(y, labels)
        } else {
            y
        };
        graph.set_outputs(vec![output]);
        cases.push(Case {
            name: if work == Work::Inference { "mlp-inference" } else { "mlp-adam" },
            description: json!({"rows": 127, "widths": [384, 640, 256], "activation": "GELU", "bias": true}),
            work,
            graph,
            inputs,
            output_elements: 127 * 256,
        });
    }
    for work in [Work::Inference, Work::Adam] {
        use meganeura::models::smollm2::{self, SmolLM2Config};
        let config = SmolLM2Config::medium_test();
        let seq = 127;
        let graph = if work == Work::Adam {
            smollm2::build_training_graph(&config, seq)
        } else {
            let mut graph = Graph::new();
            let y = smollm2::build_graph(&mut graph, &config, seq);
            graph.set_outputs(vec![y]);
            graph
        };
        let mut inputs = vec![Input::U32(
            "token_ids",
            (0..seq).map(|i| (i % config.vocab_size) as u32).collect(),
        )];
        if work == Work::Adam {
            inputs.push(Input::F32("labels", labels(seq, config.vocab_size)));
        }
        cases.push(Case {
            name: if work == Work::Inference { "smollm2-inference" } else { "smollm2-adam" },
            description: json!({"builder": "SmolLM2Config::medium_test", "sequence": seq, "layers": config.num_hidden_layers, "hidden": config.hidden_size, "ffn": config.intermediate_size, "heads": config.num_attention_heads, "kv_heads": config.num_key_value_heads, "vocabulary": config.vocab_size, "kv_cache": false}),
            work,
            graph,
            inputs,
            output_elements: seq * config.vocab_size,
        });
    }
    let config = meganeura::models::whisper::WhisperConfig::whisper_tiny();
    cases.push(Case {
        name: "whisper-sgd",
        description: json!({"builder": "WhisperConfig::whisper_tiny encoder", "batch": 1, "mel_frames": 100, "hidden": config.d_model, "layers": config.n_layers, "loss": "mean squared encoder output"}),
        work: Work::Sgd,
        graph: meganeura::models::whisper::build_training_graph(&config, 1, 100),
        inputs: vec![Input::F32("mel", data(config.n_mels * 100, 1, 2.0))],
        output_elements: 1,
    });
    cases.push(Case {
        name: "resnet50-flb",
        description: json!({"builder": "build_resnet50_training", "batch": 1, "image": [3, 224, 224], "classes": 1000, "batch_norm": "folded, not running-statistic training", "loss": "cross entropy"}),
        work: Work::ForwardLossBackward,
        graph: meganeura::models::resnet::build_resnet50_training(1),
        inputs: vec![
            Input::F32("image", data(3 * 224 * 224, 1, 2.0)),
            Input::F32("labels", labels(1, 1000)),
        ],
        output_elements: 1,
    });
    cases
}

fn initialize(case: &Case, session: &mut Session) {
    let mut ones = HashSet::<NodeId>::new();
    let mut zeros = HashSet::<NodeId>::new();
    let mut fan_in = HashMap::<NodeId, usize>::new();
    for node in case.graph.nodes() {
        match node.op {
            Op::RmsNorm { .. } => {
                ones.insert(node.inputs[1]);
            }
            Op::LayerNorm { .. } => {
                ones.insert(node.inputs[1]);
                zeros.insert(node.inputs[2]);
            }
            Op::BiasAdd | Op::AddPerChannel { .. } => {
                zeros.insert(node.inputs[1]);
            }
            Op::Conv2d {
                in_channels,
                kernel_h,
                kernel_w,
                ..
            } => {
                fan_in.insert(node.inputs[1], (in_channels * kernel_h * kernel_w) as usize);
            }
            _ => {}
        }
    }
    for node in case.graph.nodes() {
        let Op::Parameter { ref name } = node.op else {
            continue;
        };
        assert!(
            session.has_parameter(name),
            "original parameter {name} missing"
        );
        let values = if ones.contains(&node.id) {
            vec![1.0; node.ty.num_elements()]
        } else if zeros.contains(&node.id) {
            vec![0.0; node.ty.num_elements()]
        } else {
            let fan = fan_in.get(&node.id).copied().unwrap_or(node.ty.shape[0]);
            data(
                node.ty.num_elements(),
                node.id + 2,
                (12.0 / fan as f32).sqrt(),
            )
        };
        session.set_parameter(name, &values);
    }
    for input in &case.inputs {
        match input {
            Input::F32(name, values) => session.set_input(name, values),
            Input::U32(name, values) => session.set_input_u32(name, values),
        }
    }
    match case.work {
        Work::Adam => session.set_adam(1e-4, 0.9, 0.999, 1e-8),
        Work::Sgd => session.set_learning_rate(1e-3),
        _ => {}
    }
    if matches!(case.work, Work::Adam | Work::Sgd) {
        session.set_grad_clip_norm(1.0);
    }
}

fn compile_options() -> CompileOptions {
    CompileOptions {
        flash_forward_coop: false,
        flash_backward_coop: false,
        ..Default::default()
    }
}

fn runtime_options(policy: CoopPolicy) -> SessionOptions {
    SessionOptions {
        coop: policy,
        ..Default::default()
    }
}

fn make_session(
    case: &Case,
    gpu: &Arc<blade_graphics::Context>,
    policy: CoopPolicy,
) -> (Session, Value) {
    let start = Instant::now();
    let (mut session, _) = meganeura::build(
        &case.graph,
        SessionConfig {
            mode: if case.work == Work::Inference {
                Mode::Inference
            } else {
                Mode::Training
            },
            gpu: Some(Arc::clone(gpu)),
            runtime: runtime_options(policy),
            options: compile_options(),
            ..Default::default()
        },
    );
    let build_ms = start.elapsed().as_secs_f64() * 1e3;
    let start = Instant::now();
    initialize(case, &mut session);
    let initialization_ms = start.elapsed().as_secs_f64() * 1e3;
    (
        session,
        json!({"build_ms": build_ms, "initialization_ms": initialization_ms}),
    )
}

struct Snapshot {
    tensors: Vec<(String, Vec<f32>)>,
    adam_step: u32,
    adam_bytes: usize,
}

fn snapshot(session: &mut Session, case: &Case) -> Snapshot {
    session.wait();
    let names = session.param_names();
    let mut tensors: Vec<_> = names
        .iter()
        .zip(session.read_params(&names))
        .map(|(name, values)| (format!("parameter.{name}"), values))
        .collect();
    let gradient_names: Vec<_> = names
        .iter()
        .copied()
        .filter(|name| session.has_param_grad(name))
        .collect();
    for &name in &gradient_names {
        let mut values = vec![0.0; session.param_size(name).unwrap()];
        session.read_param_grad(name, &mut values);
        tensors.push((format!("gradient.{name}"), values));
    }
    let adam_bytes = session.memory_summary().adam_state_bytes;
    if adam_bytes != 0 {
        for (name, (m, v)) in gradient_names
            .iter()
            .zip(session.read_adam_states(&gradient_names))
        {
            tensors.push((format!("adam_m.{name}"), m));
            tensors.push((format!("adam_v.{name}"), v));
        }
    }
    if case.work == Work::Inference {
        tensors.push(("output".into(), session.read_output(case.output_elements)));
    } else {
        let loss = session.plan().loss_buffer.unwrap();
        let mut partials = vec![0.0; session.plan().buffers[loss.0 as usize] / 4];
        session.read_buffer(loss, &mut partials);
        tensors.push(("loss_partials".into(), partials));
        tensors.push(("loss".into(), vec![session.read_loss()]));
    }
    Snapshot {
        tensors,
        adam_step: session.adam_step_count(),
        adam_bytes,
    }
}

#[derive(Serialize)]
struct Comparison {
    stage: &'static str,
    expected_adam_step: u32,
    baseline_adam_step: u32,
    tuned_adam_step: u32,
    same_moment_allocation: bool,
    exact: bool,
    passed: bool,
    tensors: Vec<TensorComparison>,
}

fn compare(stage: &'static str, expected_adam_step: u32, a: &Snapshot, b: &Snapshot) -> Comparison {
    assert_eq!(a.tensors.len(), b.tensors.len());
    let tensors: Vec<_> = a
        .tensors
        .iter()
        .zip(&b.tensors)
        .map(|((name, a), (other, b))| {
            assert_eq!(name, other);
            TensorComparison::new(name.clone(), a, b)
        })
        .collect();
    let state_matches = a.adam_step == expected_adam_step
        && b.adam_step == expected_adam_step
        && a.adam_bytes == b.adam_bytes;
    Comparison {
        stage,
        expected_adam_step,
        baseline_adam_step: a.adam_step,
        tuned_adam_step: b.adam_step,
        same_moment_allocation: a.adam_bytes == b.adam_bytes,
        exact: state_matches && tensors.iter().all(|t| t.exact),
        passed: state_matches && tensors.iter().all(|t| t.passed),
        tensors,
    }
}

fn memory(session: &Session) -> Value {
    let m = session.memory_summary();
    json!({
        "plan_capacity_bytes": m.total_buffer_bytes,
        "graph_allocation_bytes": m.allocated_buffer_bytes,
        "adam_bytes": m.adam_state_bytes,
        "accumulator_bytes": m.grad_accumulator_bytes,
        "auxiliary_bytes": m.optimizer_aux_bytes,
        "resident_buffer_requests": m.total_allocated_bytes(),
        "device_local_bytes": m.device_local_bytes,
        "process_api_sample": session.device_memory_stats().map(|s| json!({"usage_bytes": s.usage_bytes, "budget_bytes": s.budget_bytes})),
    })
}

fn step_ms(session: &mut Session) -> f64 {
    let start = Instant::now();
    session.step();
    session.wait();
    start.elapsed().as_secs_f64() * 1e3
}

fn run_case(
    case: &Case,
    gpu: &Arc<blade_graphics::Context>,
    policy: CoopPolicy,
) -> Result<Value, Box<dyn Error>> {
    let (mut baseline, baseline_build) = make_session(case, gpu, policy);
    let (mut tuned, tuned_build) = make_session(case, gpu, policy);
    let initial_keys = baseline.dispatch_pipeline_keys();
    assert_eq!(initial_keys, tuned.dispatch_pipeline_keys());
    let mut record = json!({
        "name": case.name, "description": case.description, "work": case.work,
        "status": "checking_prefix", "baseline_build": baseline_build, "tuned_build": tuned_build,
        "baseline_pipeline_keys": initial_keys, "dispatches": baseline.plan().dispatches.len(),
        "baseline_memory": memory(&baseline), "tuned_memory_before_search": memory(&tuned),
    });
    for _ in 0..PREFIX_STEPS {
        step_ms(&mut baseline);
        step_ms(&mut tuned);
    }
    let a = snapshot(&mut baseline, case);
    let b = snapshot(&mut tuned, case);
    let initial = compare("prefix", case.work.adam_step_after(PREFIX_STEPS), &a, &b);
    let valid = initial.passed;
    record["prefix_comparison"] = serde_json::to_value(initial)?;
    if !valid {
        record["status"] = json!("prefix_parity_failed");
        return Ok(record);
    }
    drop(a);
    let report = match tuned.tune_with(TuneOptions {
        max_time: Duration::from_secs(10),
        ..Default::default()
    }) {
        Ok(report) => report,
        Err(error) => {
            record["status"] = json!("search_error");
            record["error"] = json!(error.to_string());
            return Ok(record);
        }
    };
    let after = snapshot(&mut tuned, case);
    let isolation = compare(
        "before_after_search",
        case.work.adam_step_after(PREFIX_STEPS),
        &b,
        &after,
    );
    let isolated = isolation.exact && isolation.passed;
    record["search_state_comparison"] = serde_json::to_value(isolation)?;
    record["search"] = serde_json::to_value(report)?;
    record["tuned_memory_after_search"] = memory(&tuned);
    drop(b);
    drop(after);
    if !isolated {
        record["status"] = json!("search_changed_state");
        return Ok(record);
    }
    for _ in 0..WARMUPS {
        step_ms(&mut baseline);
        step_ms(&mut tuned);
    }
    let warm = compare(
        "warmup",
        case.work.adam_step_after(PREFIX_STEPS + WARMUPS),
        &snapshot(&mut baseline, case),
        &snapshot(&mut tuned, case),
    );
    let valid = warm.passed;
    record["warmup_comparison"] = serde_json::to_value(warm)?;
    if !valid {
        record["status"] = json!("warmup_parity_failed");
        return Ok(record);
    }

    for _ in 0..SETTLING_STEPS {
        step_ms(&mut baseline);
        step_ms(&mut tuned);
    }

    let mut baseline_ms = Vec::new();
    let mut tuned_ms = Vec::new();
    let mut loss_trajectory = Vec::new();
    let mut loss_trajectory_passed = true;
    for pair in 0..SAMPLE_PAIRS {
        let (a, b) = if pair % 2 == 0 {
            (step_ms(&mut baseline), step_ms(&mut tuned))
        } else {
            let b = step_ms(&mut tuned);
            (step_ms(&mut baseline), b)
        };
        baseline_ms.push(a);
        tuned_ms.push(b);
        if case.work != Work::Inference {
            let a = baseline.read_loss();
            let b = tuned.read_loss();
            loss_trajectory_passed &= TensorComparison::new("loss".into(), &[a], &[b]).passed;
            loss_trajectory.push(json!({"step": PREFIX_STEPS + WARMUPS + SETTLING_STEPS + pair + 1, "baseline": a, "tuned": b}));
        }
    }
    let final_comparison = compare(
        "final",
        case.work
            .adam_step_after(PREFIX_STEPS + WARMUPS + SETTLING_STEPS + SAMPLE_PAIRS),
        &snapshot(&mut baseline, case),
        &snapshot(&mut tuned, case),
    );
    let valid = final_comparison.passed && loss_trajectory_passed;
    record["final_comparison"] = serde_json::to_value(final_comparison)?;
    let timing = PairedTiming::new(&baseline_ms, &tuned_ms)?;
    let final_keys = tuned.dispatch_pipeline_keys();
    let changed = initial_keys
        .iter()
        .zip(&final_keys)
        .filter(|(a, b)| a != b)
        .count();
    eprintln!(
        "{}: {} dispatches changed; {:.4} -> {:.4} ms ({:.3}x), parity={valid}",
        case.name, changed, timing.baseline_median_ms, timing.tuned_median_ms, timing.speedup
    );
    record["timing"] = serde_json::to_value(timing)?;
    record["baseline_ms"] = json!(baseline_ms);
    record["tuned_ms"] = json!(tuned_ms);
    record["loss_trajectory"] = json!(loss_trajectory);
    record["loss_trajectory_passed"] = json!(loss_trajectory_passed);
    record["dispatches_changed"] = json!(changed);
    record["tuned_pipeline_keys"] = json!(final_keys);
    record["status"] = json!(if valid {
        "complete"
    } else if !loss_trajectory_passed {
        "loss_trajectory_failed"
    } else {
        "final_parity_failed"
    });
    Ok(record)
}

fn command(program: &str, args: &[&str]) -> Result<String, Box<dyn Error>> {
    let output = Command::new(program)
        .args(args)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()?;
    if !output.status.success() {
        return Err(format!("{program}: {}", String::from_utf8_lossy(&output.stderr)).into());
    }
    Ok(String::from_utf8(output.stdout)?.trim().to_owned())
}

fn sha256(path: &Path) -> Result<String, Box<dyn Error>> {
    let path = path.to_str().ok_or("non-UTF8 path")?;
    let output =
        command("sha256sum", &[path]).or_else(|_| command("shasum", &["-a", "256", path]))?;
    Ok(output
        .split_whitespace()
        .next()
        .ok_or("missing hash")?
        .to_owned())
}

fn write_record(file: &mut File, record: &Value) -> Result<(), Box<dyn Error>> {
    file.seek(SeekFrom::Start(0))?;
    file.set_len(0)?;
    serde_json::to_writer_pretty(&mut *file, record)?;
    file.flush()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    env_logger::init();
    let path = std::env::args_os()
        .nth(1)
        .ok_or("usage: tune_holdouts <new-output.json> | --list")?;
    if path == "--list" {
        for case in cases() {
            println!("{} {:?}: {}", case.name, case.work, case.description);
        }
        return Ok(());
    }
    if !command("git", &["status", "--porcelain", "--untracked-files=no"])?.is_empty() {
        return Err("commit tracked source changes before measuring".into());
    }
    let mut output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&path)?;
    let nvidia_before = command("nvidia-smi", &[]).ok();
    let gpu = Arc::new(meganeura::init_gpu_context()?);
    let device = gpu.device_information();
    let caps = &gpu.capabilities().cooperative_matrix;
    let policy = if caps.f32_tile > 0 {
        CoopPolicy::Auto
    } else {
        CoopPolicy::Disabled
    };
    let mut document = json!({
        "schema_version": 1,
        "status": "running",
        "metadata": {
            "revision": command("git", &["rev-parse", "HEAD"] )?,
            "tracked_source_clean": true,
            "cargo_lock_sha256": sha256(&Path::new(env!("CARGO_MANIFEST_DIR")).join("Cargo.lock"))?,
            "executable_sha256": sha256(&std::env::current_exe()?)?,
            "rustc": command("rustc", &["--version"] )?,
            "os": std::env::consts::OS, "arch": std::env::consts::ARCH,
            "process_id": std::process::id(),
            "started_unix_seconds": SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs(),
            "device": {"name": device.device_name, "driver_name": device.driver_name, "driver_info": device.driver_info, "software_emulated": device.is_software_emulated, "f32_tile": caps.f32_tile, "f16_tile": caps.f16_tile},
            "cooperative_policy": format!("{policy:?}"),
            "flash_forward_coop": false, "flash_backward_coop": false,
            "compile_options": compile_options(),
            "optimize_config": meganeura::optimize::OptimizeConfig::default(),
            "runtime_options": format!("{:?}", runtime_options(policy)),
            "skip_full_optimize": false, "build_cache": false,
            "prefix_steps": PREFIX_STEPS, "warmups_per_session": WARMUPS, "settling_steps": SETTLING_STEPS, "sample_pairs": SAMPLE_PAIRS,
            "contract": "strict f32; synthetic weights/inputs; matched evolving trajectories; normal step+wait including stated optimizer/clip; construction/uploads/search/readbacks excluded",
            "optimizer": {"adam_lr": 1e-4, "sgd_lr": 1e-3, "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8, "clip_norm": 1.0, "clip_every": 1, "accumulation": false, "weight_decay": 0.0},
            "nvidia_smi_before": nvidia_before,
            "rust_log": std::env::var("RUST_LOG").ok(),
            "memory_contract": "retained buffer requests plus stage samples of API process usage with both sessions resident; not peak VRAM",
        },
        "cases": [],
    });
    write_record(&mut output, &document)?;
    let mut all_valid = true;
    for case in cases() {
        eprintln!("running {}", case.name);
        let record = run_case(&case, &gpu, policy).unwrap_or_else(
            |error| json!({"name": case.name, "status": "error", "error": error.to_string()}),
        );
        all_valid &= record["status"] == "complete";
        document["cases"].as_array_mut().unwrap().push(record);
        write_record(&mut output, &document)?;
    }
    document["status"] = json!(if all_valid {
        "complete"
    } else {
        "validation_failed"
    });
    document["finished_unix_seconds"] =
        json!(SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs());
    document["nvidia_smi_after"] = json!(command("nvidia-smi", &[]).ok());
    write_record(&mut output, &document)?;
    if !all_valid {
        return Err("holdout validation failed; all completed case records retained".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn holdout_case_roster_is_fixed() {
        let cases = cases();
        assert_eq!(
            cases.iter().map(|c| c.name).collect::<Vec<_>>(),
            [
                "mlp-inference",
                "mlp-adam",
                "smollm2-inference",
                "smollm2-adam",
                "whisper-sgd",
                "resnet50-flb"
            ]
        );
        assert_eq!(cases.iter().filter(|c| c.work == Work::Adam).count(), 2);
        assert_eq!(
            cases.iter().filter(|c| c.work == Work::Inference).count(),
            2
        );
    }

    #[test]
    fn equal_but_wrong_optimizer_counters_fail() {
        let snapshot = Snapshot {
            tensors: vec![("parameter".into(), vec![1.0])],
            adam_step: 3,
            adam_bytes: 8,
        };
        assert!(compare("test", 3, &snapshot, &snapshot).passed);
        let wrong = compare("test", 4, &snapshot, &snapshot);
        assert!(!wrong.passed && !wrong.exact);
    }

    #[test]
    #[ignore = "constructs GPU sessions; numerical preflight, not a performance run"]
    fn holdout_prefixes_are_finite_and_match_before_search() {
        let gpu = Arc::new(meganeura::init_gpu_context().unwrap());
        for case in cases() {
            eprintln!("preflight {}", case.name);
            let (mut a, _) = make_session(&case, &gpu, CoopPolicy::Disabled);
            let (mut b, _) = make_session(&case, &gpu, CoopPolicy::Disabled);
            for _ in 0..PREFIX_STEPS {
                a.step();
                a.wait();
                b.step();
                b.wait();
            }
            let comparison = compare(
                "prefix",
                case.work.adam_step_after(PREFIX_STEPS),
                &snapshot(&mut a, &case),
                &snapshot(&mut b, &case),
            );
            for tensor in &comparison.tensors {
                assert!(tensor.passed, "{} {}: {:?}", case.name, tensor.name, tensor);
            }
            assert!(comparison.passed, "{}", case.name);
            if case.work != Work::Inference {
                assert!(a.read_loss().is_finite() && a.read_loss() > 0.0);
                assert!(
                    comparison
                        .tensors
                        .iter()
                        .any(|t| t.name.starts_with("gradient.") && t.reference_sum_sq > 0.0)
                );
            }
            assert_eq!(
                a.adam_step_count(),
                if case.work == Work::Adam {
                    PREFIX_STEPS as u32
                } else {
                    0
                }
            );
        }
    }
}
