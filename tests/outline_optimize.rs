//! End-to-end coverage for repeated-region outlining: graphs over the
//! egglog node cutoff must still get e-graph saturation (on one block
//! instance per region), the extractor's fusion choices must land in
//! every instance, and the optimized session must produce the same
//! numbers as a completely unoptimized one.

use meganeura::graph::Op;
use meganeura::{
    Graph, OptimizeConfig, OptimizeMode, autodiff, compile, optimize, runtime::Session,
};

/// Residual MLP with a decomposed-Silu activation. Each layer carries
/// two fusible patterns: Add(MatMul(h, w), h) → FusedMatMulAdd and
/// Mul(a, Sigmoid(a)) → Silu. 5 nodes per layer.
fn deep_residual_model(layers: usize, dim: usize, batch: usize) -> Graph {
    let mut g = Graph::new();
    let x = g.input("x", &[batch, dim]);
    let target = g.input("target", &[batch, dim]);
    let mut h = x;
    for l in 0..layers {
        let w = g.parameter(&format!("l{l}.w"), &[dim, dim]);
        let mm = g.matmul(h, w);
        let a = g.add(mm, h);
        let sig = g.sigmoid(a);
        h = g.mul(a, sig);
    }
    let loss = g.mse_loss(h, target);
    g.set_outputs(vec![loss, h]);
    g
}

#[test]
fn outlined_regions_get_egglog_saturation() {
    // 64 layers exceed the egglog node cutoff and are outlined; 4 layers stay
    // under it and use full-graph extraction. Both must fuse every layer.
    for layers in [64, 4] {
        let g = deep_residual_model(layers, 8, 2);
        let active = g.nodes().len();
        assert_eq!(
            active > 300,
            layers == 64,
            "{layers} layers: {active} nodes"
        );

        let (opt, report) = optimize::optimize_with_config(
            &g,
            OptimizeConfig {
                mode: OptimizeMode::EgglogOutlined,
                ..OptimizeConfig::default()
            },
        );

        if layers == 64 {
            assert!(
                report.outlined_regions >= 1,
                "expected at least one outlined region on a 64-layer model"
            );
            assert!(
                report.egglog_time > std::time::Duration::ZERO,
                "egglog saturation must actually run on the outlined block"
            );
        } else {
            assert_eq!(report.outlined_regions, 0);
        }
        assert!(
            report.num_enodes > 0,
            "{layers} layers: saturation should report e-graph stats"
        );

        let count = |pred: fn(&Op) -> bool| opt.nodes().iter().filter(|n| pred(&n.op)).count();
        assert_eq!(
            count(|op| matches!(op, Op::FusedMatMulAdd)),
            layers,
            "one FusedMatMulAdd per layer"
        );
        assert_eq!(
            count(|op| matches!(op, Op::Silu)),
            layers,
            "one Silu per layer"
        );
    }
}

#[test]
fn production_optimization_preserves_outputs() {
    let layers = 64;
    let dim = 8;
    let batch = 2;
    let g = deep_residual_model(layers, dim, batch);

    // Fully optimized production training session (forward optimize →
    // autodiff → full-graph optimize). Greedy is the production default;
    // the two tests above exercise outlined egglog explicitly.
    let mut opt_session = meganeura::build(&g, crate::support::gpu::config()).0;

    // Baseline: autodiff + compile with no optimization at all.
    let sorted = g.toposort();
    let full = autodiff::differentiate(&sorted);
    let plan = compile::compile(&full);
    let mut raw_session = Session::with_context(plan, crate::support::gpu::gpu());

    let x: Vec<f32> = (0..batch * dim).map(|i| 0.1 + 0.01 * i as f32).collect();
    let target = vec![0.0f32; batch * dim];
    let w = vec![0.01f32; dim * dim];
    for s in [&mut opt_session, &mut raw_session] {
        for l in 0..layers {
            s.set_parameter(&format!("l{l}.w"), &w);
        }
        s.set_input("x", &x);
        s.set_input("target", &target);
        s.set_learning_rate(0.0);
        s.step();
        s.wait();
    }

    let opt_loss = opt_session.read_loss();
    let raw_loss = raw_session.read_loss();
    assert!(
        (opt_loss - raw_loss).abs() <= 1e-5 * raw_loss.abs().max(1.0),
        "loss diverged: optimized {opt_loss} vs raw {raw_loss}"
    );

    let mut opt_h = vec![0.0f32; batch * dim];
    let mut raw_h = vec![0.0f32; batch * dim];
    opt_session.read_output_by_index(1, &mut opt_h);
    raw_session.read_output_by_index(1, &mut raw_h);
    for (i, (o, r)) in opt_h.iter().zip(raw_h.iter()).enumerate() {
        let diff = (o - r).abs();
        assert!(
            diff <= 1e-5 * r.abs().max(1.0),
            "hidden state elem {i} diverged: optimized {o} vs raw {r}"
        );
    }
}
