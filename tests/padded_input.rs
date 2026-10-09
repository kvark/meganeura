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
