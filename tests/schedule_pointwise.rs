//! Parity tests: a graph's output must match within f32 round-off whether
//! pointwise ops run through the hand-written `unary.wgsl` / `binary.wgsl`
//! shaders or the schedule-template codegen path. This is the gate that
//! lets us retire hand-written pointwise entries one at a time.

use meganeura::{
    CompileOptions, Graph, NodeId, build_inference_session, build_inference_session_with,
};

/// Build+run a graph once, optionally with schedule-pointwise codegen on.
///
/// `build` receives the graph plus any input NodeIds it asked for via
/// `input_names`, in order, and returns the output NodeId.
fn run_once(
    input_names: &[&str],
    inputs: &[&[f32]],
    n_out: usize,
    build: &dyn Fn(&mut Graph, &[NodeId]) -> NodeId,
    opts_on: bool,
) -> Vec<f32> {
    assert_eq!(input_names.len(), inputs.len());
    let mut g = Graph::new();
    let ids: Vec<NodeId> = input_names
        .iter()
        .zip(inputs.iter())
        .map(|(name, data)| g.input(name, &[data.len()]))
        .collect();
    let y = build(&mut g, &ids);
    g.set_outputs(vec![y]);

    let mut session = if opts_on {
        let opts = CompileOptions {
            use_schedule_pointwise: true,
        };
        build_inference_session_with(&g, &opts)
    } else {
        build_inference_session(&g)
    };
    for (name, data) in input_names.iter().zip(inputs.iter()) {
        session.set_input(name, data);
    }
    session.step();
    session.wait();
    session.read_output(n_out)
}

fn assert_parity(
    input_names: &[&str],
    inputs: &[&[f32]],
    n_out: usize,
    build: impl Fn(&mut Graph, &[NodeId]) -> NodeId,
) {
    let default = run_once(input_names, inputs, n_out, &build, false);
    let schedule = run_once(input_names, inputs, n_out, &build, true);
    assert_eq!(default.len(), schedule.len());
    for (i, (a, b)) in default.iter().zip(schedule.iter()).enumerate() {
        assert!(
            (a - b).abs() <= a.abs().max(b.abs()) * 1e-6 + 1e-7,
            "parity mismatch at [{i}]: default={a}, schedule={b}",
        );
    }
}

// ---- Unary ops ----

#[test]
fn relu_parity() {
    let input: Vec<f32> = (0..256).map(|i| (i as f32) * 0.1 - 12.0).collect();
    assert_parity(&["x"], &[&input], 256, |g, xs| g.relu(xs[0]));
}

#[test]
fn silu_parity() {
    let input: Vec<f32> = (0..256).map(|i| (i as f32) * 0.1 - 12.0).collect();
    assert_parity(&["x"], &[&input], 256, |g, xs| g.silu(xs[0]));
}

#[test]
fn chain_relu_neg_parity() {
    // Two unary ops in sequence exercise two separately-generated
    // pointwise pipelines in the same plan.
    let input: Vec<f32> = (0..256).map(|i| (i as f32) * 0.1 - 12.0).collect();
    assert_parity(&["x"], &[&input], 256, |g, xs| {
        let r = g.relu(xs[0]);
        g.neg(r)
    });
}

// ---- Binary ops ----

#[test]
fn add_parity() {
    let a: Vec<f32> = (0..256).map(|i| (i as f32) * 0.05 - 6.0).collect();
    let b: Vec<f32> = (0..256).map(|i| (i as f32) * 0.03 - 4.0).collect();
    assert_parity(&["a", "b"], &[&a, &b], 256, |g, xs| g.add(xs[0], xs[1]));
}

#[test]
fn mul_parity() {
    let a: Vec<f32> = (0..256).map(|i| (i as f32) * 0.05 - 6.0).collect();
    let b: Vec<f32> = (0..256).map(|i| (i as f32) * 0.03 - 4.0).collect();
    assert_parity(&["a", "b"], &[&a, &b], 256, |g, xs| g.mul(xs[0], xs[1]));
}

#[test]
fn greater_parity() {
    let a: Vec<f32> = (0..256).map(|i| (i as f32) * 0.05 - 6.0).collect();
    let b: Vec<f32> = (0..256).map(|i| -(i as f32) * 0.02 + 2.0).collect();
    assert_parity(&["a", "b"], &[&a, &b], 256, |g, xs| g.greater(xs[0], xs[1]));
}

#[test]
fn swiglu_parity() {
    let gate: Vec<f32> = (0..256).map(|i| (i as f32) * 0.05 - 6.0).collect();
    let up: Vec<f32> = (0..256).map(|i| (i as f32) * 0.02 - 2.0).collect();
    assert_parity(&["gate", "up"], &[&gate, &up], 256, |g, xs| {
        g.swiglu(xs[0], xs[1])
    });
}

// ---- Fusion pass ----

/// Confirm the fusion pass collapses a 3-op chain into 1 dispatch.
#[test]
fn fusion_reduces_dispatch_count() {
    use meganeura::compile::{CompileOptions, compile_with};

    let mut g = Graph::new();
    let x = g.input("x", &[256]);
    let a = g.relu(x);
    let b = g.neg(a);
    let c = g.silu(b);
    g.set_outputs(vec![c]);

    let default_plan = compile_with(&g, &CompileOptions::default());
    let opts = CompileOptions {
        use_schedule_pointwise: true,
    };
    let fused_plan = compile_with(&g, &opts);

    // Default: 3 dispatches (relu, neg, silu).
    assert_eq!(default_plan.dispatches.len(), 3);
    // Fused: the three pointwise dispatches should collapse into 1.
    assert_eq!(
        fused_plan.dispatches.len(),
        1,
        "expected pointwise chain to collapse to one dispatch"
    );
    assert!(fused_plan.dispatches[0].pointwise.is_some());
}

/// Parity of a 3-op chain, once the fusion pass has run.
#[test]
fn fused_chain_runtime_parity() {
    let input: Vec<f32> = (0..256).map(|i| (i as f32) * 0.1 - 12.0).collect();
    assert_parity(&["x"], &[&input], 256, |g, xs| {
        let a = g.relu(xs[0]);
        let b = g.neg(a);
        g.silu(b)
    });
}

/// Mixed binary+unary fusion: add(relu(a), b) should become one dispatch
/// with a 2-input fused DAG, and still match the hand-written path.
#[test]
fn fused_relu_into_add_runtime_parity() {
    let a: Vec<f32> = (0..256).map(|i| (i as f32) * 0.05 - 6.0).collect();
    let b: Vec<f32> = (0..256).map(|i| -(i as f32) * 0.02 + 2.0).collect();
    assert_parity(&["a", "b"], &[&a, &b], 256, |g, xs| {
        let r = g.relu(xs[0]);
        g.add(r, xs[1])
    });
}

#[test]
fn chain_add_relu_parity() {
    // Mixes a binary op with a unary op — exercises both generated-pipeline
    // kinds plus the BinaryData/UnaryData layout switching in one plan.
    let a: Vec<f32> = (0..256).map(|i| (i as f32) * 0.05 - 6.0).collect();
    let b: Vec<f32> = (0..256).map(|i| -(i as f32) * 0.02 + 2.0).collect();
    assert_parity(&["a", "b"], &[&a, &b], 256, |g, xs| {
        let s = g.add(xs[0], xs[1]);
        g.relu(s)
    });
}
