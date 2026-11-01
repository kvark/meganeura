/// An optimized execution plan saved to disk and reloaded is the same plan,
/// and runs to the same outputs.
use meganeura::runtime::Session;
use meganeura::{Graph, cache, compile, optimize};

fn build_graph() -> Graph {
    let batch = 2;
    let mut g = Graph::new();
    let x = g.input("x", &[batch, 8]);
    let w1 = g.parameter("w1", &[8, 5]);
    let b1 = g.parameter("b1", &[5]);
    let mm1 = g.matmul(x, w1);
    let h1 = g.bias_add(mm1, b1);
    let a1 = g.relu(h1);
    let w2 = g.parameter("w2", &[5, 3]);
    let b2 = g.parameter("b2", &[3]);
    let mm2 = g.matmul(a1, w2);
    let out = g.bias_add(mm2, b2);
    g.set_outputs(vec![out]);
    g
}

#[test]
fn cached_plan_round_trips_and_infers_identically() {
    let g = build_graph();
    let plan = compile::compile(&optimize::optimize(&g));

    let dir = std::env::temp_dir().join("meganeura_test_cache_inference");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("inference_plan.ron");
    cache::save_plan(&plan, &g, &path).unwrap();
    let loaded = cache::load_plan(&g, &path)
        .unwrap()
        .expect("cache should hit");
    let _ = std::fs::remove_file(&path);

    // The same dispatches and buffer sizes, reading the same values. The
    // digest's leaves are parameters and inputs by name.
    assert_eq!(plan.dispatch_inventory(), loaded.dispatch_inventory());
    assert_eq!(plan.dataflow_digest(), loaded.dataflow_digest());

    let run = |plan| {
        let mut session = Session::with_context(plan, crate::support::gpu::gpu());
        session.set_parameter("w1", &[0.1_f32; 8 * 5]);
        session.set_parameter("b1", &[0.01_f32; 5]);
        session.set_parameter("w2", &[0.2_f32; 5 * 3]);
        session.set_parameter("b2", &[0.02_f32; 3]);
        let x_data: Vec<f32> = (0..2 * 8).map(|i| (i as f32) * 0.1).collect();
        session.set_input("x", &x_data);
        session.step();
        session.wait();
        session.read_output(2 * 3)
    };
    let (original, reloaded) = (run(plan), run(loaded));
    assert!(
        original.iter().all(|v| v.is_finite() && v.abs() > 1e-10),
        "{original:?}"
    );
    assert_eq!(
        original.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        reloaded.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}
