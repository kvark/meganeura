//! Plan-shape contracts of the generated pointwise and reduction kernels:
//! chains collapse into one dispatch. Their numerics are checked against
//! the f64 reference by the `oracle` suite.

use meganeura::Graph;
use meganeura::compile::{CompileOptions, compile_with};

fn unfused() -> CompileOptions {
    CompileOptions {
        fuse_dispatches: false,
        ..Default::default()
    }
}

#[test]
fn softplus_compiles_to_one_pointwise_dispatch() {
    let mut graph = Graph::new();
    let x = graph.input("x", &[513]);
    let y = graph.softplus(x, 10.0);
    graph.set_outputs(vec![y]);

    let plan = compile_with(&graph, &CompileOptions::default());
    assert_eq!(plan.dispatches.len(), 1);
    let dispatch = &plan.dispatches[0];
    assert_eq!(dispatch.input_buffers.len(), 1);
    assert!(dispatch.pointwise().is_some());
}

/// The fusion pass collapses a 3-op chain into 1 dispatch.
#[test]
fn fusion_reduces_dispatch_count() {
    let mut g = Graph::new();
    let x = g.input("x", &[256]);
    let a = g.relu(x);
    let b = g.neg(a);
    let c = g.silu(b);
    g.set_outputs(vec![c]);

    assert_eq!(compile_with(&g, &unfused()).dispatches.len(), 3);
    let fused = compile_with(&g, &CompileOptions::default());
    assert_eq!(
        fused.dispatches.len(),
        1,
        "expected pointwise chain to collapse to one dispatch"
    );
    assert!(fused.dispatches[0].pointwise().is_some());
}

/// Ternary fusion: `add(mul(a, b), c)` — a producer binary feeds into a
/// consumer binary, producing an arity-3 fused DAG routed through
/// TernaryData.
#[test]
fn ternary_fusion_add_of_mul() {
    let mut g = Graph::new();
    let a = g.input("a", &[256]);
    let b = g.input("b", &[256]);
    let c = g.input("c", &[256]);
    let ab = g.mul(a, b);
    let out = g.add(ab, c);
    g.set_outputs(vec![out]);

    assert_eq!(compile_with(&g, &unfused()).dispatches.len(), 2);
    let fused = compile_with(&g, &CompileOptions::default());
    assert_eq!(
        fused.dispatches.len(),
        1,
        "expected mul+add to collapse into a single arity-3 dispatch"
    );
    let dag = fused.dispatches[0]
        .pointwise()
        .expect("fused dispatch should carry a DAG");
    assert_eq!(dag.n_inputs, 3);
}

/// Wider-than-3 fusion: `((a * b) + c) * d + e` collapses into a single
/// 5-input `PointwiseDAG` bound through `DynPointwiseData` and executes
/// vectorized via `vec4<f32>`.
#[test]
fn five_input_pointwise_fuses_and_runs_on_gpu() {
    const LEN: usize = 512;
    let mut g = Graph::new();
    let a = g.input("a", &[LEN]);
    let b = g.input("b", &[LEN]);
    let c = g.input("c", &[LEN]);
    let d = g.input("d", &[LEN]);
    let e = g.input("e", &[LEN]);
    let ab = g.mul(a, b);
    let abc = g.add(ab, c);
    let abcd = g.mul(abc, d);
    let out = g.add(abcd, e);
    g.set_outputs(vec![out]);

    let fused = compile_with(&g, &CompileOptions::default());
    assert_eq!(
        fused.dispatches.len(),
        1,
        "expected 5-input chain to fuse into 1 dispatch"
    );
    assert_eq!(fused.dispatches[0].pointwise().unwrap().n_inputs, 5);

    let vals = |seed: usize| -> Vec<f32> {
        (0..LEN)
            .map(|i| (((i * 13 + seed * 7) % 41) as f32 - 20.0) * 0.05)
            .collect()
    };
    let (va, vb, vc, vd, ve) = (vals(1), vals(2), vals(3), vals(4), vals(5));
    let mut session = meganeura::build_inference_session(&g);
    session.set_input("a", &va);
    session.set_input("b", &vb);
    session.set_input("c", &vc);
    session.set_input("d", &vd);
    session.set_input("e", &ve);
    session.step();
    session.wait();
    let actual = session.read_output(LEN);
    for i in 0..LEN {
        let expected = (va[i] * vb[i] + vc[i]) * vd[i] + ve[i];
        assert!(
            (actual[i] - expected).abs() < 1e-5,
            "idx {i}: got {}, want {expected}",
            actual[i]
        );
    }
}
