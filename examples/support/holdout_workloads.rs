//! Deterministic workloads and full-state comparisons shared by tuning experiments.

use super::measurement::TensorComparison;
use meganeura::{
    CoopPolicy, Graph, Mode, Session, SessionConfig, SessionOptions,
    compile::CompileOptions,
    graph::{NodeId, Op},
};
use serde::Serialize;
use serde_json::{Value, json};
use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::Instant,
};

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub enum Work {
    Inference,
    ForwardLossBackward,
    Sgd,
    Adam,
}

impl Work {
    pub fn adam_step_after(self, steps: usize) -> u32 {
        if self == Self::Adam { steps as u32 } else { 0 }
    }
}

enum Input {
    F32(&'static str, Vec<f32>),
    U32(&'static str, Vec<u32>),
}

pub struct Case {
    pub name: &'static str,
    pub description: Value,
    pub work: Work,
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

pub fn cases() -> Vec<Case> {
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

#[allow(dead_code)] // Shared module also builds in the original holdout runner.
pub fn crossover_cases() -> Vec<Case> {
    let mut graph = Graph::new();
    let mut x = graph.input("input", &[128, 512]);
    for layer in 0..8 {
        let w = graph.parameter(&format!("layer{layer}"), &[512, 512]);
        x = graph.matmul(x, w);
    }
    graph.set_outputs(vec![x]);
    let mut result = vec![Case {
        name: "dense-inference",
        description: json!({"rows": 128, "width": 512, "layers": 8, "activation": null}),
        work: Work::Inference,
        graph,
        inputs: vec![Input::F32("input", data(128 * 512, 1, 2.0))],
        output_elements: 128 * 512,
    }];
    result.extend(
        cases()
            .into_iter()
            .filter(|c| matches!(c.name, "mlp-adam" | "resnet50-flb")),
    );
    result
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

pub fn compile_options() -> CompileOptions {
    CompileOptions {
        flash_forward_coop: false,
        flash_backward_coop: false,
        ..Default::default()
    }
}

pub fn runtime_options(policy: CoopPolicy) -> SessionOptions {
    SessionOptions {
        coop: policy,
        ..Default::default()
    }
}

pub fn make_session(
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

pub struct Snapshot {
    pub tensors: Vec<(String, Vec<f32>)>,
    pub adam_step: u32,
    pub adam_bytes: usize,
}

pub fn snapshot(session: &mut Session, case: &Case) -> Snapshot {
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
pub struct Comparison {
    stage: &'static str,
    expected_adam_step: u32,
    baseline_adam_step: u32,
    tuned_adam_step: u32,
    same_moment_allocation: bool,
    pub exact: bool,
    pub passed: bool,
    pub tensors: Vec<TensorComparison>,
}

pub fn compare(
    stage: &'static str,
    expected_adam_step: u32,
    a: &Snapshot,
    b: &Snapshot,
) -> Comparison {
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

pub fn memory(session: &Session) -> Value {
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

#[allow(dead_code)] // The readback experiment continues these workloads without timing steps.
pub fn step_ms(session: &mut Session) -> f64 {
    let start = Instant::now();
    session.step();
    session.wait();
    start.elapsed().as_secs_f64() * 1e3
}
