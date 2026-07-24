use meganeura::{Graph, Mode, SessionConfig};

fn run_forward(reverse: bool) -> Vec<f32> {
    let mut graph = Graph::new();
    let input = graph.input("input", &[2, 4]);
    let output = graph.exclusive_cumsum(input, reverse);
    graph.set_outputs(vec![output]);

    let (mut session, _) = meganeura::build(
        &graph,
        SessionConfig {
            mode: Mode::Inference,
            ..SessionConfig::default()
        },
    );
    session.set_input("input", &[1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.0, -0.5]);
    session.step();
    session.wait();
    session.read_output(8)
}

#[test]
fn exclusive_cumsum_matches_both_directions() {
    assert_eq!(
        run_forward(false),
        [0.0, 1.0, 3.0, 6.0, 0.0, -1.0, -0.5, 1.5]
    );
    assert_eq!(run_forward(true), [9.0, 7.0, 4.0, 0.0, 2.0, 1.5, -0.5, 0.0]);
}

fn run_gradient(reverse: bool) -> Vec<f32> {
    let mut graph = Graph::new();
    let input = graph.parameter("input", &[2, 4]);
    let prefix = graph.exclusive_cumsum(input, reverse);
    let loss = graph.sum_all(prefix);
    graph.set_outputs(vec![loss]);

    let mut session = meganeura::build_session(&graph);
    session.set_parameter("input", &[1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.0, -0.5]);
    session.set_learning_rate(0.0);
    session.step();
    session.wait();
    let mut gradient = vec![0.0; 8];
    session.read_param_grad("input", &mut gradient);
    gradient
}

#[test]
fn exclusive_cumsum_backward_uses_transpose_direction() {
    assert_eq!(
        run_gradient(false),
        [3.0, 2.0, 1.0, 0.0, 3.0, 2.0, 1.0, 0.0]
    );
    assert_eq!(run_gradient(true), [0.0, 1.0, 2.0, 3.0, 0.0, 1.0, 2.0, 3.0]);
}

fn run_shift(offset: i32) -> (Vec<f32>, Vec<f32>) {
    let mut graph = Graph::new();
    let input = graph.parameter("input", &[2, 4]);
    let shifted = graph.shift_inner(input, offset);
    let loss = graph.sum_all(shifted);
    graph.set_outputs(vec![loss, shifted]);

    let mut session = meganeura::build_session(&graph);
    session.set_parameter("input", &[1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.0, -0.5]);
    session.set_learning_rate(0.0);
    session.step();
    session.wait();
    let mut output = vec![0.0; 8];
    session.read_output_by_index(1, &mut output);
    let mut gradient = vec![0.0; 8];
    session.read_param_grad("input", &mut gradient);
    (output, gradient)
}

#[test]
fn shift_inner_matches_zero_fill_and_inverse_gradient() {
    let (right, right_gradient) = run_shift(1);
    assert_eq!(right, [0.0, 1.0, 2.0, 3.0, 0.0, -1.0, 0.5, 2.0]);
    assert_eq!(right_gradient, [1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0]);

    let (left, left_gradient) = run_shift(-1);
    assert_eq!(left, [2.0, 3.0, 4.0, 0.0, 0.5, 2.0, -0.5, 0.0]);
    assert_eq!(left_gradient, [0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0]);
}
