use meganeura::Graph;

#[test]
fn laprop_normalizes_before_accumulating_momentum() {
    let mut graph = Graph::new();
    let parameter = graph.parameter("parameter", &[1]);
    let loss = graph.mean_all(parameter);
    graph.set_outputs(vec![loss]);

    let mut session = meganeura::build_session(&graph);
    session.set_parameter("parameter", &[1.0]);
    session.write_adam_m("parameter", &[0.4]);
    session.write_adam_v("parameter", &[0.25]);
    session.set_adam_step_count(1);

    let learning_rate = 0.1_f32;
    let beta1 = 0.9_f32;
    let beta2 = 0.999_f32;
    let epsilon = 1.0e-8_f32;
    session.set_laprop(learning_rate, beta1, beta2, epsilon);
    session.step();
    session.wait();

    let step = 2_i32;
    let gradient = 1.0_f32;
    let variance = beta2 * 0.25 + (1.0 - beta2) * gradient * gradient;
    let variance_hat = variance / (1.0 - beta2.powi(step));
    let normalized = gradient / (variance_hat.sqrt() + epsilon);
    let momentum = beta1 * 0.4 + (1.0 - beta1) * normalized;
    let momentum_hat = momentum / (1.0 - beta1.powi(step));
    let expected_parameter = 1.0 - learning_rate * momentum_hat;

    let mut actual_parameter = [0.0];
    session.read_param("parameter", &mut actual_parameter);
    let mut actual_momentum = [0.0];
    session.read_adam_m("parameter", &mut actual_momentum);
    let mut actual_variance = [0.0];
    session.read_adam_v("parameter", &mut actual_variance);

    assert!((actual_parameter[0] - expected_parameter).abs() < 1.0e-5);
    assert!((actual_momentum[0] - momentum).abs() < 1.0e-5);
    assert!((actual_variance[0] - variance).abs() < 1.0e-5);
    assert_eq!(session.adam_step_count(), 2);
}
