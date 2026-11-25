//! Logical input sizes must survive allocation padding and plan serialization.

#[test]
fn logical_input_uploads_clear_padding_and_reject_wrong_shapes() {
    use meganeura::{CoopPolicy, Graph, Session, SessionOptions};
    let mut graph = Graph::new();
    let x = graph.input("x", &[4]);
    graph.set_outputs(vec![x]);
    let mut plan = meganeura::compile::compile(&graph);
    let buffer = plan.input_buffers[0].1;
    plan.buffers[buffer.0 as usize] = 32;
    let plan = serde_json::from_str(&serde_json::to_string(&plan).unwrap()).unwrap();
    let mut session = Session::with_context_opts(
        plan,
        crate::support::gpu::gpu().clone(),
        SessionOptions {
            coop: CoopPolicy::Disabled,
            ..Default::default()
        },
    );
    // Retain full-slot uploads for callers already supplying their own padding.
    session.set_input("x", &[1.0; 8]);
    session.set_input("x", &[2.0; 4]);
    assert_eq!(
        session.read_output(8),
        [2.0, 2.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0]
    );
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            session.set_input("x", &[3.0; 3]);
        }))
        .is_err()
    );
    assert_eq!(session.read_output(4), [2.0; 4]);
}

#[test]
fn ragged_padded_parameter_reports_logical_size_and_uploads_cleanly() {
    use meganeura::{CoopPolicy, Graph, Session, SessionOptions};
    let mut graph = Graph::new();
    let x = graph.input("x", &[5]);
    let w = graph.parameter("w", &[5]);
    let y = graph.add(x, w);
    graph.set_outputs(vec![y]);

    let mut plan = meganeura::compile::compile(&graph);
    let x_buf = plan.input_buffers[0].1;
    let w_buf = plan.param_buffers[0].1;
    let y_buf = plan.output_buffers[0];
    // Simulate target tile padding while dropping param_types/input_types to
    // exercise the logical_buffers fallback path in transfer.rs.
    plan.logical_buffers = plan.buffers.clone();
    plan.buffers[x_buf.0 as usize] = 64;
    plan.buffers[w_buf.0 as usize] = 64;
    plan.buffers[y_buf.0 as usize] = 64;
    plan.param_types.clear();
    plan.input_types.clear();

    let mut session = Session::with_context_opts(
        plan,
        crate::support::gpu::gpu().clone(),
        SessionOptions {
            coop: CoopPolicy::Disabled,
            debug: true,
            ..Default::default()
        },
    );
    assert_eq!(session.param_size("w"), Some(5));
    assert_eq!(session.input_host_ptr("x").map(|(_, sz)| sz), Some(20));

    session.set_input("x", &[1.0, 2.0, 3.0, 4.0, 5.0]);
    session.set_parameter("w", &[10.0, 20.0, 30.0, 40.0, 50.0]);
    session.step();
    session.wait();

    // read_node must return the 5 logical elements, not 16 padded elements.
    let node_vals = session.read_node(y).expect("debug session reads y");
    assert_eq!(node_vals, vec![11.0, 22.0, 33.0, 44.0, 55.0]);
}
