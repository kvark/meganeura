//! Transformer layers in the form PyTorch's exporter writes, end to end.
//!
//! `tests/fixtures/onnx/generate.py` authors the models node by node in
//! that form (they are not exporter output) and computes their expected
//! outputs with ONNX's reference evaluator. Each must import, run on the
//! GPU, and build its decomposed norms and softmax as fused kernels.

use std::path::PathBuf;

use meganeura::graph::Op;
use meganeura::{Mode, SessionConfig, load_onnx};

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/onnx")
        .join(name)
}

fn floats(name: &str) -> Vec<f32> {
    std::fs::read(fixture(name))
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&b| f32::from_le_bytes(b))
        .collect()
}

/// Run the export and return the recognized, optimized graph's ops.
fn run(name: &str) -> Vec<Op> {
    let model = load_onnx(&fixture(&format!("{name}.onnx")))
        .unwrap_or_else(|e| panic!("{name}: import failed: {e}"));
    let mut config = SessionConfig::from_env();
    config.mode = Mode::Inference;
    config.gpu = Some(meganeura::reference::gpu::shared_context());
    let (mut session, _) = meganeura::build(&model.graph, config);
    for (param, data) in &model.weights {
        session.set_parameter(param, data);
    }
    session.set_input("hidden_states", &floats(&format!("{name}.input.bin")));
    session.step();
    session.wait();
    let expected = floats(&format!("{name}.expected.bin"));
    let mut got = vec![0.0; expected.len()];
    session.read_output_by_index(0, &mut got);
    // Every value must be finite: a NaN would drop out of a max-fold.
    for (i, (&g, &e)) in got.iter().zip(&expected).enumerate() {
        assert!(g.is_finite() && e.is_finite(), "{name}: element {i} is {g}");
        assert!((g - e).abs() < 2e-4, "{name}: element {i} is {g}, want {e}");
    }
    let graph = meganeura::optimize::optimize(&model.graph.recompose());
    graph.nodes().iter().map(|node| node.op.clone()).collect()
}

fn count(ops: &[Op], matches: impl Fn(&Op) -> bool) -> usize {
    ops.iter().filter(|op| matches(op)).count()
}

#[test]
fn bert_layer() {
    let ops = run("bert_layer");
    assert_eq!(count(&ops, |op| matches!(op, Op::LayerNorm { .. })), 2);
    assert_eq!(count(&ops, |op| matches!(op, Op::Softmax)), 1);
}

#[test]
fn llama_layer() {
    let ops = run("llama_layer");
    assert_eq!(count(&ops, |op| matches!(op, Op::RmsNorm { .. })), 2);
    assert_eq!(count(&ops, |op| matches!(op, Op::Softmax)), 1);
    assert_eq!(
        count(&ops, |op| matches!(op, Op::SwiGLU | Op::SwiGLUConcat)),
        1
    );
}

#[test]
fn llama_gqa_dynamic_layer() {
    let ops = run("llama_gqa_dynamic_layer");
    assert_eq!(count(&ops, |op| matches!(op, Op::RmsNorm { .. })), 2);
    assert_eq!(count(&ops, |op| matches!(op, Op::Softmax)), 1);
    // Shape arithmetic is folded at import: nothing on the GPU computes it.
    assert_eq!(count(&ops, |op| matches!(op, Op::Constant { .. })), 0);
}
