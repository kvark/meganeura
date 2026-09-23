//! Plan-shape contracts of the generated pointwise and reduction kernels:
//! chains collapse into one dispatch. Their numerics, under every
//! lowering, are checked against the f64 reference by the `oracle` suite.

use meganeura::Graph;

#[test]
fn softplus_compiles_to_one_pointwise_dispatch() {
    use meganeura::compile::{CompileOptions, compile_with};

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

    let baseline_plan = compile_with(
        &g,
        &CompileOptions {
            use_schedule_pointwise: false,
            ..Default::default()
        },
    );
    let fused_plan = compile_with(
        &g,
        &CompileOptions {
            use_schedule_pointwise: true,
            ..Default::default()
        },
    );

    // Baseline: 3 dispatches (relu, neg, silu).
    assert_eq!(baseline_plan.dispatches.len(), 3);
    // Fused: the three pointwise dispatches should collapse into 1.
    assert_eq!(
        fused_plan.dispatches.len(),
        1,
        "expected pointwise chain to collapse to one dispatch"
    );
    assert!(fused_plan.dispatches[0].pointwise().is_some());
}

/// Ternary fusion: `add(mul(a, b), c)` — a producer binary feeds into a
/// consumer binary, producing an arity-3 fused DAG routed through
/// TernaryData.
#[test]
fn ternary_fusion_add_of_mul() {
    use meganeura::compile::{CompileOptions, compile_with};

    let mut g = Graph::new();
    let a = g.input("a", &[256]);
    let b = g.input("b", &[256]);
    let c = g.input("c", &[256]);
    let ab = g.mul(a, b);
    let out = g.add(ab, c);
    g.set_outputs(vec![out]);

    // Unfused: 2 dispatches (mul, add).
    assert_eq!(
        compile_with(
            &g,
            &CompileOptions {
                use_schedule_pointwise: false,
                ..Default::default()
            },
        )
        .dispatches
        .len(),
        2
    );
    let fused = compile_with(
        &g,
        &CompileOptions {
            use_schedule_pointwise: true,
            ..Default::default()
        },
    );
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

/// With `use_schedule_reduction`, softmax should compile to two dispatches
/// (max reduction + sum/normalize). Without, it's one.
#[test]
fn softmax_schedule_emits_two_dispatches() {
    use meganeura::compile::{CompileOptions, compile_with};
    let mut g = Graph::new();
    let x = g.input("x", &[4, 64]);
    let y = g.softmax(x);
    g.set_outputs(vec![y]);

    let baseline = compile_with(
        &g,
        &CompileOptions {
            use_schedule_reduction: false,
            ..Default::default()
        },
    );
    let schedule = compile_with(
        &g,
        &CompileOptions {
            use_schedule_reduction: true,
            ..Default::default()
        },
    );
    assert_eq!(baseline.dispatches.len(), 1);
    assert_eq!(schedule.dispatches.len(), 2);
    assert!(schedule.dispatches[0].reduction().is_some());
    assert!(schedule.dispatches[1].reduction().is_some());
}
