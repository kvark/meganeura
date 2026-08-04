use meganeura::{Graph, build_session};

#[test]
fn materialize_copies_values_and_propagates_gradient() {
    let mut graph = Graph::new();
    let input = graph.parameter("input", &[2, 4]);
    let staged = graph.materialize(input);
    let loss = graph.sum_all(staged);
    graph.set_outputs(vec![loss, staged]);

    let expected = [1.0, -2.0, 3.5, -4.25, 5.0, 6.25, -7.5, 8.0];
    let mut session = build_session(&graph);
    session.set_parameter("input", &expected);
    session.set_learning_rate(0.0);
    session.step();
    session.wait();

    let mut actual = [0.0; 8];
    session.read_output_by_index(1, &mut actual);
    assert_eq!(actual, expected);

    let mut gradient = [0.0; 8];
    session.read_param_grad("input", &mut gradient);
    assert_eq!(gradient, [1.0; 8]);
}
