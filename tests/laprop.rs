use meganeura::Graph;

fn optimizer_state(session: &meganeura::Session, n: usize) -> [Vec<f32>; 3] {
    let mut state = std::array::from_fn(|_| vec![0.0; n]);
    session.read_adam_v("parameter", &mut state[0]);
    session.read_adam_m("parameter", &mut state[1]);
    session.read_param("parameter", &mut state[2]);
    state
}

#[test]
fn adaptive_optimizers_preserve_tiny_variance_and_nonfinite_state() {
    use std::sync::Arc;

    let gpu =
        Arc::new(meganeura::init_gpu_context_with(meganeura::GpuOptions::from_env()).unwrap());
    let gradients = [
        0.0f32, 1.0, -1.0, 1e-6, -1e-6, 1e-16, -1e-16, 1e-17, -1e-17, 3e-18, -3e-18, 1e-18, -1e-18,
        1e-20, -1e-20, 1e-23,
    ];
    // Gradient, variance, momentum. Include finite controls beside poisoned state.
    let poisoned = [
        [1e-20, f32::NAN, 0.0],
        [1e-20, f32::INFINITY, 0.0],
        [1e-20, f32::NEG_INFINITY, 0.0],
        [1e-20, f32::from_bits(0xffc00001), 0.0],
        [f32::NAN, 0.0, 0.0],
        [f32::INFINITY, 0.0, 0.0],
        [f32::NEG_INFINITY, 0.0, 0.0],
        [1e-20, 0.0, f32::NAN],
        [1e-20, 0.0, f32::INFINITY],
        [1e-20, 0.0, f32::NEG_INFINITY],
        [1e-20, 0.0, 0.0],
        [-1e-20, 1.0, 0.0],
        [0.0, f32::NAN, 0.0],
        [0.0, f32::INFINITY, 0.0],
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
    ];
    let n = gradients.len() + poisoned.len();
    let mut graph = Graph::new();
    let parameter = graph.parameter("parameter", &[n]);
    let input = graph.input("input", &[n]);
    let product = graph.mul(parameter, input);
    let loss = graph.mean_all(product);
    graph.set_outputs(vec![loss]);
    for laprop in [false, true] {
        for beta2 in [0.0f32, 0.999] {
            let (plan, _) = meganeura::compile_training_graph(&graph);
            let mut session = meganeura::Session::with_context_opts(
                plan,
                gpu.clone(),
                meganeura::SessionOptions {
                    poison: true,
                    ..Default::default()
                },
            );
            if laprop {
                session.set_laprop(0.1, 0.9, beta2, 1e-20);
            } else {
                session.set_adam(0.1, 0.9, beta2, 1e-20);
            }
            let initial_v = if beta2 == 0.0 { 1.0 } else { 0.0 };
            let mut reference = [vec![initial_v; n], vec![0.0f32; n], vec![1.0f32; n]];
            reference[0][gradients.len()..].copy_from_slice(&poisoned.map(|c| c[1]));
            reference[1][gradients.len()..].copy_from_slice(&poisoned.map(|c| c[2]));
            session.set_parameter("parameter", &reference[2]);
            session.write_adam_v("parameter", &reference[0]);
            session.write_adam_m("parameter", &reference[1]);
            for step in 1..=8 {
                let sign = if step % 2 == 0 { -1.0 } else { 1.0 };
                let step_gradients: Vec<_> = gradients
                    .iter()
                    .map(|g| g * sign)
                    .chain(poisoned.iter().map(|c| c[0]))
                    .collect();
                let input: Vec<_> = step_gradients.iter().map(|g| g * n as f32).collect();
                session.set_input("input", &input);
                session.step();
                session.wait();
                assert_eq!(session.adam_step_count(), step as u32);
                let mut actual_gradient = vec![0.0; n];
                session.read_param_grad("parameter", &mut actual_gradient);
                let actual = optimizer_state(&session, n);
                let beta1 = 0.9f32;
                let bc1 = f64::from(1.0 / (1.0 - beta1.powi(step)));
                let bc2 = f64::from(1.0 / (1.0 - beta2.powi(step)));
                for i in 0..n {
                    let gradient = step_gradients[i];
                    assert!(
                        actual_gradient[i] == gradient
                            || (gradient.is_nan() && actual_gradient[i].is_nan()),
                        "raw gradient {step}/{i}: {} vs {gradient}",
                        actual_gradient[i],
                    );
                    let g = f64::from(gradient);
                    reference[0][i] = (f64::from(beta2) * f64::from(reference[0][i])
                        + f64::from(1.0 - beta2) * g * g)
                        as f32;
                    let denominator =
                        (f64::from(reference[0][i]) * bc2).sqrt() + f64::from(1e-20f32);
                    let normalized = if laprop { g / denominator } else { g };
                    reference[1][i] = (f64::from(beta1) * f64::from(reference[1][i])
                        + f64::from(1.0 - beta1) * normalized)
                        as f32;
                    let update =
                        f64::from(reference[1][i]) * bc1 / if laprop { 1.0 } else { denominator };
                    let parameter = f64::from(reference[2][i]);
                    // The shader's zero weight decay still evaluates 0 * Inf as NaN.
                    reference[2][i] =
                        (parameter - f64::from(0.1f32) * (update + 0.0 * parameter)) as f32;
                    for k in 0..3 {
                        let expected = f64::from(reference[k][i]);
                        let value = actual[k][i];
                        let matches = if expected.is_nan() {
                            value.is_nan()
                        } else if expected.is_infinite() {
                            f64::from(value) == expected
                        } else {
                            let tolerance = if k == 0 {
                                1e-5 * expected.abs() + 2.0 * f64::from(f32::from_bits(1))
                            } else {
                                1e-5 * (1.0 + expected.abs())
                            };
                            value.is_finite() && (f64::from(value) - expected).abs() <= tolerance
                        };
                        assert!(
                            matches,
                            "laprop={laprop} beta2={beta2} step={step} index={i} field={k}: {} vs {expected}",
                            actual[k][i]
                        );
                    }
                }
            }
        }
    }
}

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
