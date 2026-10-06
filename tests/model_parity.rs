//! Every model builds the same plan from its decomposition into primitives.
//!
//! `Graph::decompose` spells each composite in primitives. Builds recognize
//! the composites again, so the decomposed model must compile to exactly
//! the plan of the model as written: the same kernels, parameters, grids
//! and buffers, and the same values flowing between them
//! ([`ExecutionPlan::dataflow_digest`]). Training recognizes only
//! composites with exact gradients, so there the comparison is with the
//! model whose other composites are spelled in primitives too. No GPU is
//! needed: plans compile on the CPU.
//!
//! [`ExecutionPlan::dataflow_digest`]: meganeura::compile::ExecutionPlan::dataflow_digest

use meganeura::graph::OpClass;
use meganeura::models::{efficientnet, resnet, sd_unet, smollm2, smolvla, smolvlm2, whisper};
use meganeura::{CompileOptions, Graph, Mode, NodeId, OptimizeConfig, compile_plan};

/// The plan's dispatch inventory, and the digest of what it computes.
fn plan(graph: &Graph, mode: Mode) -> (Vec<String>, u64) {
    let plan = compile_plan(
        graph,
        mode,
        OptimizeConfig::default(),
        &CompileOptions::default(),
    );
    (plan.dispatch_inventory(), plan.dataflow_digest())
}

#[track_caller]
fn assert_parity(what: &str, graph: &Graph, modes: &[Mode]) {
    // Models need only the public ops: primitives and composites.
    for node in graph.nodes() {
        assert_ne!(
            node.op.class(),
            OpClass::Private,
            "{what} builds the private op {:?}",
            node.op
        );
    }
    let decomposed = graph.decompose();
    assert!(
        decomposed
            .nodes()
            .iter()
            .all(|n| n.op.class() == OpClass::Primitive),
        "{what}: decomposition left a non-primitive"
    );
    let composites = graph
        .nodes()
        .iter()
        .filter(|n| n.op.class() == OpClass::Composite)
        .count();
    for &mode in modes {
        // Training recognizes only composites with exact gradients, so the
        // others build as their primitives do.
        let written = graph.decompose_for(mode);
        let (original, rebuilt) = (plan(&written, mode), plan(&decomposed, mode));
        if original != rebuilt {
            let only = |a: &[String], b: &[String]| {
                a.iter()
                    .filter(|x| !b.contains(x))
                    .map(|s| s.chars().take(160).collect::<String>())
                    .collect::<Vec<_>>()
            };
            panic!(
                "{what} ({mode:?}, {composites} composites): plans differ\n  only original: {:#?}\n  only decomposed: {:#?}",
                only(&original.0, &rebuilt.0),
                only(&rebuilt.0, &original.0)
            );
        }
    }
}

fn graph_with(outputs: impl FnOnce(&mut Graph) -> Vec<NodeId>) -> Graph {
    let mut g = Graph::new();
    let out = outputs(&mut g);
    g.set_outputs(out);
    g
}

const INFERENCE: &[Mode] = &[Mode::Inference];

#[test]
fn smollm2() {
    let config = smollm2::Config::small_test();
    let forward = graph_with(|g| vec![smollm2::build_graph(g, &config, 8)]);
    assert_parity("smollm2 forward", &forward, INFERENCE);
    let training = smollm2::build_training_graph(&config, 8);
    assert_parity("smollm2 training", &training, &[Mode::Training]);
    let prefill = graph_with(|g| {
        let (logits, k, v) = smollm2::build_prefill_graph(g, &config, 8);
        [vec![logits], k, v].concat()
    });
    assert_parity("smollm2 prefill", &prefill, INFERENCE);
    let decode = graph_with(|g| {
        let (logits, k, v) = smollm2::build_decode_graph(g, &config, 16);
        [vec![logits], k, v].concat()
    });
    assert_parity("smollm2 decode", &decode, INFERENCE);
}

#[test]
fn smolvla() {
    let config = smolvla::Config::small_test();
    let forward = graph_with(|g| vec![smolvla::build_action_expert(g, &config, 4, 6)]);
    assert_parity("smolvla expert", &forward, INFERENCE);
    let training = smolvla::build_action_expert_training(&config, 4, 6);
    assert_parity("smolvla training", &training, &[Mode::Training]);
}

#[test]
fn smolvlm2() {
    let config = smolvlm2::Config {
        vision: smolvlm2::VisionConfig {
            image_size: 64,
            patch_size: 16,
            hidden_size: 32,
            num_attention_heads: 2,
            num_hidden_layers: 2,
            intermediate_size: 64,
            layer_norm_eps: 1e-6,
        },
        text: smolvlm2::TextConfig {
            vocab_size: 64,
            hidden_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            intermediate_size: 64,
            rms_norm_eps: 1e-5,
            rope_theta: 100_000.0,
        },
        scale_factor: 2,
    };
    let forward = graph_with(|g| vec![smolvlm2::build_graph(g, &config, 4)]);
    assert_parity("smolvlm2", &forward, INFERENCE);
}

#[test]
fn whisper() {
    let config = whisper::Config::whisper_tiny();
    let forward = graph_with(|g| vec![whisper::build_encoder(g, &config, 1, 32)]);
    assert_parity("whisper encoder", &forward, INFERENCE);
    let training = whisper::build_training_graph(&config, 1, 32);
    assert_parity("whisper training", &training, &[Mode::Training]);
}

#[test]
fn sd_unet() {
    let config = sd_unet::Config::tiny();
    let forward = graph_with(|g| vec![sd_unet::build_unet(g, &config)]);
    assert_parity("sd_unet", &forward, INFERENCE);
    let training = graph_with(|g| vec![sd_unet::build_training_graph(g, &config)]);
    assert_parity("sd_unet training", &training, &[Mode::Training]);
}

#[test]
fn vision_classifiers() {
    let efficientnet = graph_with(|g| vec![efficientnet::build_graph(g, 1)]);
    assert_parity("efficientnet", &efficientnet, INFERENCE);
    let resnet = graph_with(|g| vec![resnet::build_graph(g, 1)]);
    assert_parity("resnet", &resnet, INFERENCE);
    let resnet50 = resnet::build_resnet50_training(1);
    assert_parity("resnet50 training", &resnet50, &[Mode::Training]);
}

#[test]
fn onnx_exports() {
    for name in ["bert_layer", "llama_layer", "llama_gqa_dynamic_layer"] {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests/fixtures/onnx")
            .join(format!("{name}.onnx"));
        let model = meganeura::load_onnx(&path).unwrap();
        assert_parity(name, &model.graph, INFERENCE);
    }
}

/// Build-time cost of recognition at full model size. Recomposition runs
/// in every build; this reports what it adds next to the rest of the
/// build. Run with `--release -- --ignored --nocapture`.
#[test]
#[ignore]
fn recomposition_cost() {
    use std::time::Instant;
    let config = smollm2::Config::smollm2_135m();
    let forward = graph_with(|g| vec![smollm2::build_graph(g, &config, 128)]);
    let training = smollm2::build_training_graph(&config, 128);
    for (what, graph, mode) in [
        ("smollm2-135m forward", &forward, Mode::Inference),
        ("smollm2-135m training", &training, Mode::Training),
    ] {
        let decomposed = graph.decompose();
        let start = Instant::now();
        let _ = graph.recompose();
        let as_written = start.elapsed();
        let start = Instant::now();
        let _ = decomposed.recompose();
        let from_primitives = start.elapsed();
        let start = Instant::now();
        let original = plan(graph, mode);
        let build = start.elapsed();
        let start = Instant::now();
        let rebuilt = plan(&decomposed, mode);
        let build_decomposed = start.elapsed();
        assert_eq!(original, rebuilt, "{what}");
        println!(
            "{what}: {} nodes, {} decomposed; recompose {:.1} ms as written, {:.1} ms from primitives; \
             build {:.0} ms as written, {:.0} ms from primitives",
            graph.nodes().len(),
            decomposed.nodes().len(),
            as_written.as_secs_f64() * 1e3,
            from_primitives.as_secs_f64() * 1e3,
            build.as_secs_f64() * 1e3,
            build_decomposed.as_secs_f64() * 1e3,
        );
    }
}

/// On the GPU, the model built from its decomposition computes bit for bit
/// what the model as written does, in the same time: it is the same plan.
#[test]
fn decomposed_model_runs_identically() {
    use meganeura::graph::{DType, Op};
    use std::time::Instant;
    let config = smollm2::Config::medium_test();
    let graph = graph_with(|g| vec![smollm2::build_graph(g, &config, 32)]);
    let decomposed = graph.decompose();
    let session = |g: &Graph| {
        let mut cfg = meganeura::SessionConfig::from_env();
        cfg.mode = Mode::Inference;
        cfg.gpu = Some(meganeura::reference::gpu::shared_context());
        let (mut session, _) = meganeura::build(g, cfg);
        for node in g.nodes() {
            match node.op {
                Op::Parameter { ref name } => {
                    let n = node.ty.num_elements();
                    let data: Vec<f32> = (0..n)
                        .map(|i| ((i * 7 + name.len()) as f32 * 0.013).sin() * 0.05)
                        .collect();
                    session.set_parameter(name, &data);
                }
                Op::Input { ref name } if node.ty.dtype == DType::U32 => {
                    let ids: Vec<u32> = (0..node.ty.num_elements() as u32)
                        .map(|i| i * 5 % config.vocab_size as u32)
                        .collect();
                    session.set_input_u32(name, &ids);
                }
                _ => {}
            }
        }
        session
    };
    let mut original = session(&graph);
    let mut rebuilt = session(&decomposed);
    // Interleaved rounds, best of each, so neither side pays for going
    // first.
    let mut times = [std::time::Duration::MAX; 2];
    for _ in 0..3 {
        for (k, s) in [&mut original, &mut rebuilt].into_iter().enumerate() {
            s.step();
            s.wait();
            let start = Instant::now();
            for _ in 0..10 {
                s.step();
            }
            s.wait();
            times[k] = times[k].min(start.elapsed());
        }
    }
    let outputs: Vec<Vec<f32>> = [&mut original, &mut rebuilt]
        .into_iter()
        .map(|s| {
            let mut out = vec![0.0f32; graph.node(graph.outputs()[0]).ty.num_elements()];
            s.read_output_by_index(0, &mut out);
            out
        })
        .collect();
    assert!(outputs[0].iter().all(|v| v.is_finite()));
    assert_eq!(
        outputs[0].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        outputs[1].iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        "outputs differ"
    );
    println!(
        "smollm2 medium_test, best of 3 x 10 steps: {:.2} ms as written, {:.2} ms decomposed",
        times[0].as_secs_f64() * 1e3,
        times[1].as_secs_f64() * 1e3
    );
}

/// Recomposition leaves a model written with composites as it is: the
/// pipeline before it built the same ops.
#[test]
fn recomposition_keeps_models_as_written() {
    let ops = |g: &Graph| {
        let live = meganeura::optimize::optimize_with_config(
            g,
            OptimizeConfig {
                mode: meganeura::OptimizeMode::Off,
                ..OptimizeConfig::default()
            },
        )
        .0;
        // Attributes that do not change the result take one canonical
        // spelling, lowered to the same kernels: full attention as
        // multi-head attention, an upsample's planes as rows, a gate's
        // channel count as its length.
        let canonical = |node: &meganeura::graph::Node| match node.op {
            meganeura::graph::Op::Upsample2x { in_w, .. } => format!(
                "{:?}",
                meganeura::graph::Op::Upsample2x {
                    channels: 1,
                    in_h: (node.ty.num_elements() / 4 / in_w as usize) as u32,
                    in_w,
                }
            ),
            meganeura::graph::Op::MulPerChannel { spatial, .. } => format!(
                "{:?}",
                meganeura::graph::Op::MulPerChannel {
                    channels: (node.ty.num_elements() / spatial as usize) as u32,
                    spatial,
                }
            ),
            meganeura::graph::Op::FullAttention {
                num_heads,
                num_kv_heads,
                head_dim,
            } => format!(
                "{:?}",
                meganeura::graph::Op::MultiHeadAttn {
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    is_cross: false,
                }
            ),
            ref other => format!("{other:?}"),
        };
        let mut ops: Vec<String> = live
            .nodes()
            .iter()
            .map(|n| format!("{} {:?}", canonical(n), n.ty.shape))
            .collect();
        ops.sort();
        ops
    };
    let config = smollm2::Config::small_test();
    let vla = smolvla::Config::small_test();
    let unet = sd_unet::Config::tiny();
    let cases = vec![
        (
            "smollm2",
            graph_with(|g| vec![smollm2::build_graph(g, &config, 8)]),
        ),
        (
            "smollm2 training",
            smollm2::build_training_graph(&config, 8),
        ),
        (
            "smollm2 decode",
            graph_with(|g| {
                let (logits, k, v) = smollm2::build_decode_graph(g, &config, 16);
                [vec![logits], k, v].concat()
            }),
        ),
        ("smolvla", smolvla::build_action_expert_training(&vla, 4, 6)),
        (
            "whisper",
            whisper::build_training_graph(&whisper::Config::whisper_tiny(), 1, 32),
        ),
        (
            "sd_unet",
            graph_with(|g| vec![sd_unet::build_training_graph(g, &unet)]),
        ),
        (
            "efficientnet",
            graph_with(|g| vec![efficientnet::build_graph(g, 1)]),
        ),
        ("resnet50", resnet::build_resnet50_training(1)),
    ];
    for (what, graph) in cases {
        let (before, after) = (ops(&graph), ops(&graph.recompose()));
        let only = |a: &[String], b: &[String]| {
            a.iter()
                .filter(|x| !b.contains(x))
                .cloned()
                .collect::<Vec<_>>()
        };
        assert_eq!(
            before,
            after,
            "{what}: only as written {:?}, only recomposed {:?}",
            only(&before, &after),
            only(&after, &before)
        );
    }
}

/// The digest tells apart plans that launch the same kernels on different
/// values: swapped operands, or a constant with other contents.
#[test]
fn dataflow_digest_sees_what_the_inventory_does_not() {
    let swapped = |swap: bool| {
        graph_with(|g| {
            let x = g.input("x", &[4, 8]);
            let y = g.input("y", &[4, 8]);
            let (a, b) = if swap { (y, x) } else { (x, y) };
            vec![g.sub(a, b)]
        })
    };
    let (p, q) = (
        plan(&swapped(false), Mode::Inference),
        plan(&swapped(true), Mode::Inference),
    );
    assert_eq!(p.0, q.0, "the same kernels");
    assert_ne!(p.1, q.1, "on swapped operands");

    let scaled = |value: f32| {
        graph_with(|g| {
            let x = g.input("x", &[4, 8]);
            let w = g.constant((0..8).map(|i| value + i as f32).collect(), &[8]);
            vec![g.bias_mul(x, w)]
        })
    };
    let (p, q) = (
        plan(&scaled(2.0), Mode::Inference),
        plan(&scaled(3.0), Mode::Inference),
    );
    assert_eq!(p.0, q.0, "the same kernels");
    assert_ne!(p.1, q.1, "with other constants");
    assert_eq!(p, plan(&scaled(2.0), Mode::Inference), "deterministic");
}

/// What a dispatch computes includes its fused metadata: changing a
/// matmul's epilogue, weight format or precision policy, while keeping its
/// shader, kernel, parameters, grid and buffers, changes the digest.
#[test]
fn dataflow_digest_covers_fused_metadata() {
    use meganeura::compile::{MatMulEpilogue, WeightFormat};
    use meganeura::schedule::{PointwiseDAG, Pw};
    let g = graph_with(|g| {
        let x = g.input("x", &[8, 16]);
        let w = g.parameter("w", &[16, 8]);
        vec![g.matmul(x, w)]
    });
    let base = compile_plan(
        &g,
        Mode::Inference,
        OptimizeConfig::default(),
        &CompileOptions::default(),
    );
    let index = base
        .dispatches
        .iter()
        .position(|d| d.matmul_epilogue.is_none() && format!("{:?}", d.shader).contains("MatMul"))
        .expect("a matmul dispatch");
    type Mutation = fn(&mut meganeura::compile::Dispatch);
    let mutations: [(&str, Mutation); 3] = [
        ("relu epilogue", |d| {
            d.matmul_epilogue = Some(MatMulEpilogue {
                dag: PointwiseDAG {
                    n_inputs: 1,
                    ops: vec![Pw::LoadInput(0), Pw::Relu(0)],
                    output: 1,
                },
                inputs: Vec::new(),
            })
        }),
        ("f16 weights", |d| d.weight_format = WeightFormat::F16),
        ("full precision", |d| {
            d.requires_full_precision = !d.requires_full_precision
        }),
    ];
    for (what, mutate) in mutations {
        let mut changed = base.clone();
        mutate(&mut changed.dispatches[index]);
        assert_ne!(base.dataflow_digest(), changed.dataflow_digest(), "{what}");
        assert_ne!(
            base.dispatch_inventory(),
            changed.dispatch_inventory(),
            "{what}"
        );
    }
}
