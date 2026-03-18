/// Test that an optimized execution plan can be saved to disk, reloaded,
/// and produce identical inference results.
///
/// Pipeline: build graph → optimize → compile → save plan →
///           load plan → verify structure → infer from loaded plan.
use meganeura::compile::ExecutionPlan;
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

fn optimize_and_compile(g: &Graph) -> ExecutionPlan {
    let optimized = optimize::optimize(g);
    compile::compile(&optimized)
}

#[test]
fn cache_roundtrip_preserves_plan() {
    let g = build_graph();
    let plan = optimize_and_compile(&g);

    // Save plan to disk
    let dir = std::env::temp_dir().join("meganeura_test_cache_inference");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("inference_plan.ron");
    cache::save_plan(&plan, &g, &path).unwrap();

    // Load it back
    let loaded = cache::load_plan(&g, &path).unwrap().expect("cache should hit");

    // Every field must match
    assert_eq!(plan.buffers, loaded.buffers, "buffer sizes");
    assert_eq!(plan.param_buffers, loaded.param_buffers, "param_buffers");
    assert_eq!(plan.input_buffers, loaded.input_buffers, "input_buffers");
    assert_eq!(plan.loss_buffer, loaded.loss_buffer, "loss_buffer");
    assert_eq!(plan.param_grad_pairs, loaded.param_grad_pairs, "param_grad_pairs");
    assert_eq!(plan.dispatches.len(), loaded.dispatches.len(), "dispatch count");
    for (i, (a, b)) in plan.dispatches.iter().zip(loaded.dispatches.iter()).enumerate() {
        assert_eq!(a.shader, b.shader, "dispatch[{}] shader", i);
        assert_eq!(a.workgroups, b.workgroups, "dispatch[{}] workgroups", i);
        assert_eq!(a.input_buffers, b.input_buffers, "dispatch[{}] inputs", i);
        assert_eq!(a.output_buffer, b.output_buffer, "dispatch[{}] output", i);
        assert_eq!(a.params, b.params, "dispatch[{}] params", i);
    }

    let _ = std::fs::remove_file(&path);
}

#[test]
fn infer_from_loaded_plan() {
    let g = build_graph();
    let plan = optimize_and_compile(&g);

    // Save and reload
    let dir = std::env::temp_dir().join("meganeura_test_cache_inference2");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("inference_plan2.ron");
    cache::save_plan(&plan, &g, &path).unwrap();
    let loaded = cache::load_plan(&g, &path).unwrap().expect("cache should hit");

    // Build a GPU session from the loaded plan and run inference
    let mut session = Session::new(loaded);

    session.set_parameter("w1", &vec![0.1_f32; 8 * 5]);
    session.set_parameter("b1", &vec![0.01_f32; 5]);
    session.set_parameter("w2", &vec![0.2_f32; 5 * 3]);
    session.set_parameter("b2", &vec![0.02_f32; 3]);

    let x_data: Vec<f32> = (0..2 * 8).map(|i| (i as f32) * 0.1).collect();
    session.set_input("x", &x_data);

    session.step();
    session.wait();

    let output = session.read_output(2 * 3);
    assert_eq!(output.len(), 6, "expected 2*3 output elements");

    // All outputs should be finite and non-zero (matmul+bias with non-zero weights/input)
    for (i, &v) in output.iter().enumerate() {
        assert!(v.is_finite(), "output[{}] = {} is not finite", i, v);
        assert!(v.abs() > 1e-10, "output[{}] = {} is unexpectedly zero", i, v);
    }

    let _ = std::fs::remove_file(&path);
}
