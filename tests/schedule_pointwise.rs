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
/// `DynPointwiseData`.
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
    let mut config = crate::support::gpu::inference_config();
    config.tune = false;
    let mut session = meganeura::build(&g, config).0;
    assert_eq!(
        session.plan().dispatches[0].workgroups,
        [(LEN as u32 / 4).div_ceil(256), 1, 1],
        "expected vec4 workgroup grid"
    );
    assert!(
        session.dispatch_pipeline_keys()[0].contains(":v4"),
        "expected vectorized pointwise pipeline, got {:?}",
        session.dispatch_pipeline_keys()
    );
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

/// An 8-input expression requires 9 storage buffers if fused into one kernel,
/// which exceeds the portable 8-storage-buffer limit (`MAX_POINTWISE_INPUTS = 7`).
/// Fusion must keep every dispatch within 7 inputs (+ 1 output = 8 storage buffers)
/// across multiple dispatches and still match the reference on GPU.
#[test]
fn eight_input_pointwise_respects_eight_storage_buffer_cap() {
    const LEN: usize = 256;
    let mut g = Graph::new();
    let names = ["x0", "x1", "x2", "x3", "x4", "x5", "x6", "x7"];
    let inputs: Vec<_> = names.iter().map(|name| g.input(name, &[LEN])).collect();
    let mut acc = inputs[0];
    for &inp in &inputs[1..] {
        acc = g.add(acc, inp);
    }
    g.set_outputs(vec![acc]);

    let fused = compile_with(&g, &CompileOptions::default());
    assert!(
        fused.dispatches.len() >= 2,
        "8-input sum must split across >= 2 dispatches to stay within 8 storage buffers"
    );
    for d in &fused.dispatches {
        let dag = d.pointwise().expect("pointwise dispatch");
        assert!(
            dag.n_inputs <= meganeura::schedule::MAX_POINTWISE_INPUTS,
            "fused dispatch exceeded storage cap: {} inputs",
            dag.n_inputs
        );
        assert!(d.input_buffers.len() < 8);
    }

    let mut config = crate::support::gpu::inference_config();
    config.tune = false;
    let mut session = meganeura::build(&g, config).0;
    let data: Vec<Vec<f32>> = (0..8)
        .map(|s| (0..LEN).map(|i| ((i + s) % 17) as f32 * 0.125).collect())
        .collect();
    for (name, buf) in names.iter().zip(&data) {
        session.set_input(name, buf);
    }
    session.step();
    session.wait();
    let got = session.read_output(LEN);
    for i in 0..LEN {
        let want: f32 = data.iter().map(|v| v[i]).sum();
        assert!(
            (got[i] - want).abs() < 1e-5,
            "idx {i}: got {}, want {want}",
            got[i]
        );
    }
}

/// Subgroup `argmax_inner` over an all-NaN row must return a valid column
/// index (0.0) rather than the empty-index sentinel `0xffffffff` (4294967296.0).
#[test]
fn subgroup_argmax_all_nan_row_returns_valid_column_index() {
    let mut g = Graph::new();
    let x = g.input("x", &[2, 64]);
    let y = g.argmax_inner(x);
    g.set_outputs(vec![y]);

    let mut config = crate::support::gpu::inference_config();
    config.tune = false;
    let mut session = meganeura::build(&g, config).0;
    let mut data = vec![f32::NAN; 2 * 64];
    for c in 0..64 {
        data[64 + c] = if c == 37 { 10.0 } else { c as f32 * 0.01 };
    }
    session.set_input("x", &data);
    session.step();
    session.wait();
    let got = session.read_output(2);
    assert_eq!(
        got[0], 0.0,
        "all-NaN row must return column 0, got {}",
        got[0]
    );
    assert_eq!(got[1], 37.0, "finite row must still return argmax 37");
}
