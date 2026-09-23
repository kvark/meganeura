//! Regressions found by the September 2026 shader audit.
//!
//! Every case compares the GPU against a CPU reference or a central finite
//! difference, so each one fails on the defect it names rather than on a
//! changed rounding order.

use meganeura::{Graph, Session};

fn values(n: usize, frequency: f32, phase: f32) -> Vec<f32> {
    (0..n)
        .map(|i| (i as f32 * frequency + phase).sin() * 0.2)
        .collect()
}

/// Central finite difference of the scalar loss for one parameter element.
fn finite_difference(
    session: &mut Session,
    parameters: &[(&str, Vec<f32>)],
    inputs: &[(&str, Vec<f32>)],
    name: &str,
    index: usize,
    step: f32,
) -> f32 {
    let mut evaluate = |delta: f32| {
        for (parameter, data) in parameters {
            if *parameter == name {
                let mut data = data.clone();
                data[index] += delta;
                session.set_parameter(parameter, &data);
            } else {
                session.set_parameter(parameter, data);
            }
        }
        for (input, data) in inputs {
            session.set_input(input, data);
        }
        session.step();
        session.wait();
        session.read_loss()
    };
    (evaluate(step) - evaluate(-step)) / (2.0 * step)
}

fn assert_close(label: &str, actual: f32, expected: f32) {
    let error = (actual - expected).abs();
    let scale = actual.abs().max(expected.abs()).max(1e-5);
    assert!(
        error < 2e-3 || error / scale < 0.05,
        "{label}: analytical={actual:+.6e}, numerical={expected:+.6e}"
    );
}

/// Heads wider than 64 with a sequence shorter than the flash query block
/// used to fall back to the 64-lane scalar kernels, which silently dropped
/// every dimension past 63.
#[test]
fn attention_backward_covers_heads_wider_than_64() {
    let (seq, heads, head_dim) = (4usize, 2u32, 128u32);
    let width = heads as usize * head_dim as usize;
    let mut graph = Graph::new();
    let q = graph.parameter("q", &[seq, width]);
    let k = graph.parameter("k", &[seq, width]);
    let v = graph.parameter("v", &[seq, width]);
    let attention = graph.multi_head_attn(q, k, v, heads, heads, head_dim, false);
    let weights = graph.input("weights", &[seq, width]);
    let weighted = graph.mul(attention, weights);
    let loss = graph.sum_all(weighted);
    graph.set_outputs(vec![loss]);

    let n = seq * width;
    let parameters = [
        ("q", values(n, 0.017, 0.3)),
        ("k", values(n, 0.019, 0.7)),
        ("v", values(n, 0.023, 1.1)),
    ];
    let inputs = [("weights", values(n, 0.013, 1.7))];

    let mut training = meganeura::build(&graph, meganeura::SessionConfig::from_env()).0;
    for (name, data) in &parameters {
        training.set_parameter(name, data);
    }
    training.set_input("weights", &inputs[0].1);
    training.set_learning_rate(0.0);
    training.step();
    training.wait();

    let mut inference = meganeura::build(&graph, meganeura::SessionConfig::inference_from_env()).0;
    for (name, _) in &parameters {
        let mut gradient = vec![0.0; n];
        training.read_param_grad(name, &mut gradient);
        // Both sides of lane 64 in both heads and in a later row.
        for index in [
            0usize,
            63,
            64,
            100,
            127,
            128 + 70,
            width + 90,
            3 * width + 255,
        ] {
            let numerical =
                finite_difference(&mut inference, &parameters, &inputs, name, index, 1e-3);
            assert_close(&format!("{name}[{index}]"), gradient[index], numerical);
        }
    }
}
