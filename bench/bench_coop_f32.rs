//! Paired, full-f32 qualification of the cooperative matmul/backward kernels.
//!
//! Build first, then run on an otherwise idle device:
//! cargo build --release --example bench_coop_f32 --features models
//! target/release/examples/bench_coop_f32 [all|matmul|attention|training]
//!
//! One shared context, identical inputs, no search, no GPU timestamps. Reports
//! construction separately; warmup and ABBA batches time recording, submission
//! and completion, not compilation, uploads, readbacks or correctness checks.
//! Attention micro-sessions contain exactly one dQ or dK/dV dispatch, fed by
//! the same scalar forward/LSE/row-dot preparation. Both implementations are
//! qualified against all prepared scalar gradients before measuring.
use meganeura::{
    CoopPolicy, GpuOptions, Graph, Mode, Session, SessionConfig, SessionOptions,
    compile::{self, BufferRef, Dispatch, Kernel, ShaderEntry},
};
use serde_json::{Value, json};
use std::{
    sync::Arc,
    time::{Duration, Instant},
};

const REPLAYS: usize = 16;
const ROUNDS: usize = 8;

fn values(len: usize, salt: usize, scale: f32) -> Vec<f32> {
    (0..len)
        .map(|i| (((i * 17 + salt * 31) % 101) as f32 - 50.0) * scale)
        .collect()
}

fn config(gpu: &Arc<blade_graphics::Context>, coop: bool, mode: Mode) -> SessionConfig<'static> {
    SessionConfig {
        gpu: Some(gpu.clone()),
        mode,
        tune: false,
        runtime: SessionOptions {
            coop: if coop {
                CoopPolicy::NativeF32
            } else {
                CoopPolicy::Disabled
            },
            ..Default::default()
        },
        ..Default::default()
    }
}

fn median(samples: &[f64]) -> f64 {
    let mut sorted = samples.to_vec();
    sorted.sort_by(f64::total_cmp);
    (sorted[(sorted.len() - 1) / 2] + sorted[sorted.len() / 2]) * 0.5
}

fn relative_error(actual: &[f32], expected: &[f32], floor: f64) -> f64 {
    assert_eq!(actual.len(), expected.len());
    let mut error = 0.0;
    let mut norm = 0.0;
    for (&a, &b) in actual.iter().zip(expected) {
        assert!(a.is_finite() && b.is_finite());
        error += (f64::from(a) - f64::from(b)).powi(2);
        norm += f64::from(b).powi(2);
    }
    (error / norm.max(floor).max(f64::MIN_POSITIVE)).sqrt()
}

fn outputs(session: &Session, reads: &[(BufferRef, usize)]) -> Vec<f32> {
    reads
        .iter()
        .flat_map(|&(buffer, len)| {
            let mut data = vec![0.0; len];
            session.read_buffer(buffer, &mut data);
            data
        })
        .collect()
}

fn inventory(session: &Session) -> std::collections::BTreeMap<String, usize> {
    let mut counts = std::collections::BTreeMap::new();
    for d in &session.plan().dispatches {
        let ept = match d.kernel {
            Kernel::AttentionBackward { ept_cap } => Some(ept_cap),
            _ => None,
        };
        let key = format!(
            "{:?} grid={:?} params={:?} coop={} small={} prologue={} epilogue={} copies={} ept={ept:?}",
            d.shader,
            d.workgroups,
            d.params,
            d.use_coop(),
            d.use_small_tiles(),
            d.matmul_prologue.is_some(),
            d.matmul_epilogue.is_some(),
            d.horizontal_batch,
        );
        *counts.entry(key).or_default() += 1;
    }
    counts
}

fn measure(
    gpu: &Arc<blade_graphics::Context>,
    label: &str,
    metadata: Value,
    sessions: &mut [Session; 2],
    reads: [Vec<(BufferRef, usize)>; 2],
    build_ms: [f64; 2],
    tolerance: f64,
    floor: f64,
) -> Value {
    for session in &mut *sessions {
        session.step();
        session.wait();
    }
    let expected = outputs(&sessions[0], &reads[0]);
    let actual = outputs(&sessions[1], &reads[1]);
    assert!(floor > 0.0 || expected.iter().any(|&value| value != 0.0));
    let relative_l2 = relative_error(&actual, &expected, floor);
    assert!(
        relative_l2 <= tolerance,
        "{label}: relative L2 {relative_l2} > {tolerance}"
    );
    // Check each output separately too: dV's magnitude must not hide a
    // bad dK, nor a large model gradient hide a smaller parameter's error.
    let mut offset = 0;
    let mut per_output_l2 = Vec::new();
    for (index, &(_, len)) in reads[0].iter().enumerate() {
        assert_eq!(len, reads[1][index].1);
        let end = offset + len;
        let output_floor = if index == 0 {
            floor * len as f64 / expected.len() as f64
        } else {
            0.0
        };
        let error = relative_error(&actual[offset..end], &expected[offset..end], output_floor);
        assert!(
            error <= tolerance,
            "{label} output {index}: relative L2 {error} > {tolerance}"
        );
        per_output_l2.push(error);
        offset = end;
    }
    let mut encoder = gpu.create_command_encoder(blade_graphics::CommandEncoderDesc {
        name: "paired f32 wall timing",
        buffer_count: 2,
        manual_barriers: false,
    });
    let mut batch = |session: &mut Session| {
        let start = Instant::now();
        encoder.start();
        for _ in 0..REPLAYS {
            session.record(&mut encoder).unwrap();
        }
        let sync = gpu.submit(&mut encoder);
        session.track_submission(sync);
        session.wait();
        start.elapsed().as_secs_f64() * 1e3 / REPLAYS as f64
    };
    for session in &mut *sessions {
        let start = Instant::now();
        while start.elapsed() < Duration::from_millis(250) {
            batch(session);
        }
    }
    let mut samples = [Vec::new(), Vec::new()];
    for _ in 0..ROUNDS {
        for index in [0, 1, 1, 0] {
            samples[index].push(batch(&mut sessions[index]));
        }
    }
    gpu.destroy_command_encoder(&mut encoder);
    let times = [median(&samples[0]), median(&samples[1])];
    eprintln!(
        "{label}: scalar {:.4} ms, coop-f32 {:.4} ms, speedup {:.3}x, L2 {relative_l2:.2e}",
        times[0],
        times[1],
        times[0] / times[1]
    );
    json!({"case":label, "metadata":metadata, "construction_ms":build_ms,
        "scalar_ms":times[0], "coop_f32_ms":times[1], "speedup":times[0]/times[1],
        "relative_l2":relative_l2, "per_output_l2":per_output_l2,
        "tolerance":tolerance, "norm_floor":floor,
        "samples_ms":samples, "dispatch_inventory":sessions.each_ref().map(inventory)})
}

fn matmul(gpu: &Arc<blade_graphics::Context>, records: &mut Vec<Value>) {
    for (label, m, n, k, kind) in [
        ("gqa-projection", 128, 192, 576, "nn"),
        ("vocabulary", 128, 49_152, 576, "nn"),
        ("compact", 32, 512, 17, "nn"),
        ("ragged", 65, 272, 17, "nn"),
        ("transpose-a", 65, 256, 33, "at"),
        ("transpose-b", 65, 256, 33, "bt"),
        ("prologue", 65, 256, 33, "prologue"),
        ("addend", 65, 256, 33, "addend"),
        ("epilogue", 65, 256, 33, "epilogue"),
    ] {
        let mut graph = Graph::new();
        let a = graph.input("a", &if kind == "at" { [k, m] } else { [m, k] });
        let b = graph.input("b", &if kind == "bt" { [n, k] } else { [k, n] });
        let lhs = if kind == "prologue" {
            let norm = graph.input("norm", &[k]);
            graph.rms_norm(a, norm, 1e-5)
        } else {
            a
        };
        let y = match kind {
            "at" => graph.matmul_at(lhs, b),
            "bt" => graph.matmul_bt(lhs, b),
            _ => graph.matmul(lhs, b),
        };
        let y = match kind {
            "addend" => {
                let src = graph.input("src", &[m, n]);
                graph.add(y, src)
            }
            "epilogue" => graph.relu(y),
            _ => y,
        };
        graph.set_outputs(vec![y]);
        let mut construction = [0.0; 2];
        let mut sessions = [false, true].map(|coop| {
            let start = Instant::now();
            let mut session = meganeura::build(&graph, config(gpu, coop, Mode::Inference)).0;
            construction[usize::from(coop)] = start.elapsed().as_secs_f64() * 1e3;
            assert_eq!(
                session.plan().dispatches.iter().any(Dispatch::use_coop),
                coop
            );
            if coop {
                let matrix = session
                    .plan()
                    .dispatches
                    .iter()
                    .find(|d| d.use_coop())
                    .unwrap();
                match kind {
                    "prologue" => assert!(matrix.matmul_prologue.is_some()),
                    "epilogue" => assert!(matrix.matmul_epilogue.is_some()),
                    "addend" => assert_eq!(matrix.shader, ShaderEntry::FusedMatMulAdd),
                    _ => {}
                }
            }
            session.set_input("a", &values(m * k, 1, 0.002));
            session.set_input("b", &values(n * k, 2, 0.002));
            if kind == "prologue" {
                session.set_input("norm", &vec![1.0; k]);
            }
            if kind == "addend" {
                session.set_input("src", &values(m * n, 3, 0.001));
            }
            session
        });
        let reads = sessions
            .each_ref()
            .map(|s| vec![(s.plan().output_buffers[0], m * n)]);
        records.push(measure(
            gpu,
            label,
            json!({"mnk":[m,n,k],"kind":kind}),
            &mut sessions,
            reads,
            construction,
            3e-5,
            0.0,
        ));
    }
}

fn attention(gpu: &Arc<blade_graphics::Context>, records: &mut Vec<Value>) {
    for (label, q_seq, kv_seq, heads, kv_heads, causal, window) in [
        ("full-128", 128, 128, 4, 4, false, 0),
        ("gqa-129", 129, 145, 4, 2, false, 0),
        ("causal-128", 128, 128, 9, 3, true, 0),
        ("causal-129", 129, 129, 4, 2, true, 0),
        ("window-1", 129, 129, 4, 2, true, 1),
        ("window-17", 145, 145, 4, 2, true, 17),
        ("window-17-long", 1024, 1024, 4, 2, true, 17),
        ("full-256", 256, 256, 8, 8, false, 0),
        ("whisper-1500", 1500, 1500, 6, 6, false, 0),
    ] {
        let q_len = q_seq * heads as usize * 64;
        let kv_len = kv_seq * kv_heads as usize * 64;
        let mut graph = Graph::new();
        let q = graph.parameter("q", &[q_seq, heads as usize * 64]);
        let k = graph.parameter("k", &[kv_seq, kv_heads as usize * 64]);
        let v = graph.parameter("v", &[kv_seq, kv_heads as usize * 64]);
        let output = if window > 0 {
            graph.sliding_window_attention(q, k, v, heads, kv_heads, 64, window)
        } else if causal {
            graph.causal_attention(q, k, v, heads, kv_heads, 64)
        } else {
            graph.multi_head_attn(q, k, v, heads, kv_heads, 64, true)
        };
        let upstream = graph.input("upstream", &[q_seq, heads as usize * 64]);
        let weighted = graph.mul(output, upstream);
        let loss = graph.sum_all(weighted);
        graph.set_outputs(vec![loss]);
        let mut preparation_config = config(gpu, false, Mode::Training);
        preparation_config.runtime.debug = true;
        let mut prepared = meganeura::build(&graph, preparation_config).0;
        prepared.set_parameter("q", &values(q_len, 1, 0.016));
        prepared.set_parameter("k", &values(kv_len, 2, 0.016));
        prepared.set_parameter("v", &values(kv_len, 3, 0.016));
        prepared.set_input("upstream", &values(q_len, 4, 0.0002));
        prepared.step();
        prepared.wait();
        for (direction, scalar, cooperative, lengths) in [
            (
                "dq",
                ShaderEntry::FlashGradQ,
                ShaderEntry::FlashGradQCoopF32,
                vec![q_len],
            ),
            (
                "dkv",
                ShaderEntry::FlashGradKV,
                ShaderEntry::FlashGradKVCoopF32,
                vec![kv_len, kv_len],
            ),
        ] {
            let template = prepared
                .plan()
                .dispatches
                .iter()
                .find(|d| d.shader == scalar)
                .unwrap();
            let inputs = prepared.read_buffers(&template.input_buffers);
            let mut micro = Graph::new();
            let input_nodes: Vec<_> = inputs
                .iter()
                .enumerate()
                .map(|(i, data)| micro.input(&format!("input{i}"), &[data.len()]))
                .collect();
            let output_nodes: Vec<_> = lengths
                .iter()
                .map(|&len| micro.constant(vec![0.0; len], &[len]))
                .collect();
            micro.set_outputs(output_nodes);
            let mut plan = compile::compile(&micro);
            let mut dispatch = template.clone();
            dispatch.input_buffers = input_nodes
                .iter()
                .map(|&id| plan.node_buffers[id as usize].1)
                .collect();
            dispatch.output_buffer = plan.output_buffers[0];
            dispatch.extra_outputs = plan.output_buffers[1..].to_vec();
            plan.dispatches = vec![dispatch];
            let mut construction = [0.0; 2];
            let mut sessions = [false, true].map(|coop| {
                let mut selected = plan.clone();
                if coop {
                    selected.dispatches[0].shader = cooperative.clone();
                    selected.dispatches[0].kernel = Kernel::Default;
                    selected.dispatches[0].workgroups = if direction == "dq" {
                        [(q_seq as u32).div_ceil(16), heads, 1]
                    } else {
                        [(kv_seq as u32).div_ceil(16), kv_heads, 1]
                    };
                }
                let start = Instant::now();
                let mut session = Session::with_context_opts(
                    selected,
                    gpu.clone(),
                    config(gpu, true, Mode::Inference).runtime,
                );
                construction[usize::from(coop)] = start.elapsed().as_secs_f64() * 1e3;
                assert_eq!(session.plan().dispatches.len(), 1);
                assert_eq!(
                    session.plan().dispatches[0].shader,
                    if coop {
                        cooperative.clone()
                    } else {
                        scalar.clone()
                    }
                );
                for (i, data) in inputs.iter().enumerate() {
                    session.set_input(&format!("input{i}"), data);
                }
                session
            });
            let reads = sessions.each_ref().map(|s| {
                s.plan()
                    .output_buffers
                    .iter()
                    .copied()
                    .zip(lengths.iter().copied())
                    .collect()
            });
            // Same operand-scale floor as the f64 regression for the zero
            // dQ/dK of a one-key window; no floor for other attention cases.
            let floor = if window == 1 {
                (0.01_f64 * 0.8).powi(2) * q_len as f64
            } else {
                0.0
            };
            records.push(measure(gpu, &format!("{label}-{direction}"), json!({"q":q_seq,"kv":kv_seq,"heads":heads,"kv_heads":kv_heads,"causal":causal,"window":window,"forced_candidate":true}), &mut sessions, reads, construction, if q_seq == 1500 { 1e-4 } else { 3e-5 }, floor));
        }
    }
}

fn training(gpu: &Arc<blade_graphics::Context>, records: &mut Vec<Value>) {
    let model = meganeura::models::smollm2::Config::medium_test();
    let seq = 128;
    let graph = meganeura::models::smollm2::build_training_graph(&model, seq);
    let mut construction = [0.0; 2];
    let mut sessions = [false, true].map(|coop| {
        let start = Instant::now();
        let mut session = meganeura::build(&graph, config(gpu, coop, Mode::Training)).0;
        construction[usize::from(coop)] = start.elapsed().as_secs_f64() * 1e3;
        assert_eq!(
            session.plan().dispatches.iter().any(Dispatch::use_coop),
            coop
        );
        assert_eq!(
            session
                .plan()
                .dispatches
                .iter()
                .any(|d| d.shader == ShaderEntry::FlashGradQCoopF32),
            coop
        );
        for (name, buffer) in session.plan().param_buffers.clone() {
            let len = session.plan().param_types[&buffer].num_elements();
            let data = if name.contains("norm") {
                vec![1.0; len]
            } else {
                values(len, 7, 0.0004)
            };
            session.set_parameter(&name, &data);
        }
        session.set_input_u32(
            "token_ids",
            &(0..seq)
                .map(|i| (i % model.vocab_size) as u32)
                .collect::<Vec<_>>(),
        );
        let mut labels = vec![0.0; seq * model.vocab_size];
        for i in 0..seq {
            labels[i * model.vocab_size + (i + 1) % model.vocab_size] = 1.0;
        }
        session.set_input("labels", &labels);
        session
    });
    // Include loss and every complete parameter gradient. No optimizer update:
    // each replay differentiates exactly the same state and inputs.
    let reads = sessions.each_ref().map(|s| {
        let mut reads = vec![(s.plan().output_buffers[0], 1)];
        reads.extend(
            s.plan()
                .param_grad_pairs
                .iter()
                .map(|&(p, g)| (g, s.plan().param_types[&p].num_elements())),
        );
        reads
    });
    records.push(measure(gpu, "smollm2-medium-training", json!({"layers":8,"hidden":128,"heads":2,"head_dim":64,"vocab":64,"seq":seq,"optimizer_update":false}), &mut sessions, reads, construction, 1e-4, 0.0));
}

fn main() {
    env_logger::init();
    let suite = std::env::args().nth(1).unwrap_or_else(|| "all".into());
    assert!(["all", "matmul", "attention", "training"].contains(&suite.as_str()));
    let gpu = Arc::new(
        meganeura::init_gpu_context_with(GpuOptions {
            timing: false,
            ..Default::default()
        })
        .unwrap(),
    );
    let caps = gpu.capabilities();
    assert!(
        caps.cooperative_matrix
            .f32_shapes
            .iter()
            .any(|shape| matches!(*shape, [8, 8, 8] | [16, 16, 16]))
    );
    assert!(caps.max_compute_shared_memory_size >= 18_624);
    let mut records = Vec::new();
    if suite == "all" || suite == "matmul" {
        matmul(&gpu, &mut records);
    }
    if suite == "all" || suite == "attention" {
        attention(&gpu, &mut records);
    }
    if suite == "all" || suite == "training" {
        training(&gpu, &mut records);
    }
    println!("{}", serde_json::to_string_pretty(&json!({
        "device":gpu.device_information().device_name,
        "timing":"completed ABBA wall batches; time per replay includes recording and submission",
        "units":"milliseconds", "replays_per_batch":REPLAYS, "abba_rounds":ROUNDS,
        "warmup_ms_per_implementation":250, "search":false, "gpu_timestamps":false,
        "records":records,
    })).unwrap());
}
