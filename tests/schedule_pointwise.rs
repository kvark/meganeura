//! Parity tests: a graph's output must match within f32 round-off whether
//! unary pointwise ops run through the hand-written `unary.wgsl` shader or
//! the schedule-template codegen path. This is the gate that lets us retire
//! hand-written unary entries one at a time.

use meganeura::{
    CompileOptions, Graph, NodeId, build_inference_session, build_inference_session_with,
};

fn run_once(
    build: &dyn Fn(&mut Graph, NodeId) -> NodeId,
    input: &[f32],
    opts_on: bool,
) -> Vec<f32> {
    let n = input.len();
    let mut g = Graph::new();
    let x = g.input("x", &[n]);
    let y = build(&mut g, x);
    g.set_outputs(vec![y]);

    let mut session = if opts_on {
        let opts = CompileOptions {
            use_schedule_pointwise: true,
        };
        build_inference_session_with(&g, &opts)
    } else {
        build_inference_session(&g)
    };
    session.set_input("x", input);
    session.step();
    session.wait();
    session.read_output(n)
}

fn assert_parity(build: impl Fn(&mut Graph, NodeId) -> NodeId, input: &[f32]) {
    let default = run_once(&build, input, false);
    let schedule = run_once(&build, input, true);
    assert_eq!(default.len(), schedule.len());
    for (i, (a, b)) in default.iter().zip(schedule.iter()).enumerate() {
        assert!(
            (a - b).abs() <= a.abs().max(b.abs()) * 1e-6 + 1e-7,
            "parity mismatch at [{i}]: default={a}, schedule={b}",
        );
    }
}

#[test]
fn relu_parity() {
    let input: Vec<f32> = (0..256).map(|i| (i as f32) * 0.1 - 12.0).collect();
    assert_parity(|g, x| g.relu(x), &input);
}

#[test]
fn silu_parity() {
    let input: Vec<f32> = (0..256).map(|i| (i as f32) * 0.1 - 12.0).collect();
    assert_parity(|g, x| g.silu(x), &input);
}

#[test]
fn chain_relu_neg_parity() {
    // Two unary ops in sequence exercise two separately-generated
    // pointwise pipelines in the same plan.
    let input: Vec<f32> = (0..256).map(|i| (i as f32) * 0.1 - 12.0).collect();
    assert_parity(
        |g, x| {
            let r = g.relu(x);
            g.neg(r)
        },
        &input,
    );
}
