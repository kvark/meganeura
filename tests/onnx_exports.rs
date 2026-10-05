//! Transformer layers as `torch.onnx.export` writes them, end to end.
//!
//! `tests/fixtures/onnx/generate.py` builds the models and their expected
//! outputs with ONNX's reference evaluator. Each must import, run on the
//! GPU, and fold its decomposed norms and softmax back into fused kernels.

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

/// Run the export and return the optimized graph's ops for inspection.
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
    let worst = got
        .iter()
        .zip(&expected)
        .map(|(g, e)| (g - e).abs())
        .fold(0.0f32, f32::max);
    assert!(worst < 2e-4, "{name}: max abs error {worst}");
    let graph = meganeura::optimize::optimize(&model.graph);
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
