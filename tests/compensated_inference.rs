//! Explicit bounded-operand tensor-core inference; never an automatic gradient mode.
use meganeura::{CoopPolicy, Graph, Mode, SessionConfig, SessionOptions};
use std::sync::Arc;

fn values(length: usize, seed: usize) -> Vec<f32> {
    (0..length)
        .map(|i| ((i * 67 + seed * 97) % 1019) as f32 / 1019.0 - 0.5)
        .collect()
}

#[test]
#[ignore = "requires a cooperative-matrix GPU"]
fn compensated_matrix_products_match_scalar_with_fused_epilogue() {
    let gpu =
        Arc::new(meganeura::init_gpu_context_with(meganeura::GpuOptions::from_env()).unwrap());
    eprintln!(
        "testing compensated inference on {}",
        gpu.device_information().device_name
    );
    assert!(gpu.capabilities().cooperative_matrix.f16_tile > 0);
    let mut failures = Vec::new();
    for (m, k, n) in [(256, 64, 512), (196, 257, 1024)] {
        for (transpose, epilogue) in [(0, false), (0, true), (1, true), (2, true)] {
            let mut graph = Graph::new();
            let a_shape = if transpose == 1 { [k, m] } else { [m, k] };
            let b_shape = if transpose == 2 { [n, k] } else { [k, n] };
            let a = graph.input("a", &a_shape);
            let b = graph.input("b", &b_shape);
            let y = match transpose {
                1 => graph.matmul_at(a, b),
                2 => graph.matmul_bt(a, b),
                _ => graph.matmul(a, b),
            };
            let y = if epilogue { graph.tanh(y) } else { y };
            graph.set_outputs(vec![y]);
            let a = values(m * k, 1);
            let b = values(k * n, 2);
            let mut results = Vec::new();
            for coop in [
                CoopPolicy::Disabled,
                CoopPolicy::AllowF16,
                CoopPolicy::CompensatedF16,
            ] {
                let (mut session, _) = meganeura::build(
                    &graph,
                    SessionConfig {
                        mode: Mode::Inference,
                        gpu: Some(Arc::clone(&gpu)),
                        runtime: SessionOptions {
                            coop,
                            ..Default::default()
                        },
                        ..Default::default()
                    },
                );
                if coop == CoopPolicy::CompensatedF16 {
                    assert!(
                        session
                            .plan()
                            .dispatches
                            .iter()
                            .any(|d| d.use_coop_compensated)
                    );
                }
                session.set_input("a", &a);
                session.set_input("b", &b);
                session.step();
                session.wait();
                let mut output = vec![0.0; m * n];
                session.read_output_by_index(0, &mut output);
                results.push(output);
            }
            let error = |left: &[f32], right: &[f32]| {
                left.iter()
                    .zip(right)
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0_f32, f32::max)
            };
            let raw_error = error(&results[0], &results[1]);
            let compensated_error = error(&results[0], &results[2]);
            let correction = error(&results[1], &results[2]);
            eprintln!(
                "shape {m},{k},{n} transpose {transpose} tanh {epilogue}: raw {raw_error}, compensated {compensated_error}, correction {correction}"
            );
            if compensated_error >= 0.00005 {
                failures.push((m, k, n, transpose, epilogue, compensated_error));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "compensated accuracy failures: {failures:?}"
    );
}

#[test]
#[ignore = "requires a cooperative-matrix GPU"]
fn compensated_policy_does_not_zero_small_weight_gradients() {
    let mut graph = Graph::new();
    let input = graph.input("input", &[1023, 512]);
    let weight = graph.parameter("weight", &[512, 256]);
    let y = graph.matmul(input, weight);
    let mean = graph.mean_all(y);
    let scale = graph.scalar(1e-6);
    let loss = graph.mul(mean, scale);
    graph.set_outputs(vec![loss]);
    let (mut session, _) = meganeura::build(
        &graph,
        SessionConfig {
            runtime: SessionOptions {
                coop: CoopPolicy::CompensatedF16,
                ..Default::default()
            },
            ..SessionConfig::from_env()
        },
    );
    assert!(
        session
            .plan()
            .dispatches
            .iter()
            .all(|d| !d.requires_full_precision || !d.use_coop)
    );
    session.set_input("input", &vec![1.0; 1023 * 512]);
    session.set_parameter("weight", &vec![0.0; 512 * 256]);
    session.step();
    session.wait();
    let mut gradient = vec![0.0; 512 * 256];
    session.read_param_grad("weight", &mut gradient);
    assert!(gradient.iter().all(|&v| (v - 1e-6 / 256.0).abs() < 1e-12));
}
