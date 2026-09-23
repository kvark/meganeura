//! Regressions from the September 2026 shader audit that the `oracle`
//! suite cannot express: gradient-clipping measurements, which buffers the
//! flash backward binds, and finiteness where f32 saturates but f64 does
//! not. The audit's numerical regressions now live in `oracle`.

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

/// Global clipping measures the norm with many workgroups per gradient. With
/// SGD the step is exactly `-lr * g * min(1, max_norm / ||g||)`, so the
/// update reveals the norm the GPU computed across every workgroup.
#[test]
fn global_grad_clip_measures_large_gradients() {
    let sizes = [300_001usize, 5, 70_000];
    let mut graph = Graph::new();
    let mut terms = Vec::new();
    for (i, &n) in sizes.iter().enumerate() {
        let p = graph.parameter(&format!("p{i}"), &[n]);
        let t = graph.input(&format!("t{i}"), &[n]);
        let product = graph.mul(p, t);
        terms.push(graph.sum_all(product));
    }
    let loss = terms
        .into_iter()
        .reduce(|a, b| graph.add(a, b))
        .expect("at least one term");
    graph.set_outputs(vec![loss]);

    // d(loss)/d(p_i) = t_i exactly.
    let targets: Vec<Vec<f32>> = sizes
        .iter()
        .enumerate()
        .map(|(i, &n)| values(n, 0.37 + i as f32 * 0.1, 0.3))
        .collect();
    let norm = targets
        .iter()
        .flatten()
        .map(|&v| (v as f64).powi(2))
        .sum::<f64>()
        .sqrt();
    let max_norm = (norm * 0.25) as f32;
    let lr = 0.5f32;

    let mut session = meganeura::build(&graph, meganeura::SessionConfig::from_env()).0;
    for (i, &n) in sizes.iter().enumerate() {
        session.set_parameter(&format!("p{i}"), &vec![0.0; n]);
        session.set_input(&format!("t{i}"), &targets[i]);
    }
    session.set_grad_clip_norm(max_norm);
    session.set_learning_rate(lr);
    session.step();
    session.wait();

    let scale = max_norm as f64 / norm;
    for (i, &n) in sizes.iter().enumerate() {
        let mut after = vec![0.0; n];
        session.read_param(&format!("p{i}"), &mut after);
        for j in [0, n / 2, n - 1] {
            let want = -(lr as f64) * targets[i][j] as f64 * scale;
            assert!(
                (after[j] as f64 - want).abs() <= 1e-4 * want.abs().max(1e-3),
                "p{i}[{j}]: got {}, want {want}",
                after[j]
            );
        }
    }
}

/// Adaptive clipping measures every parameter with many workgroups and
/// scales each by min(1, clip * max(pmin, ||p||) / ||g||). With SGD the step
/// reveals the scale the GPU computed for each parameter.
#[test]
fn adaptive_grad_clip_measures_large_parameters() {
    let sizes = [300_001usize, 5, 70_000];
    let (clip, pmin, lr) = (0.05f32, 1e-3f32, 0.5f32);
    let mut graph = Graph::new();
    let mut terms = Vec::new();
    for (i, &n) in sizes.iter().enumerate() {
        let p = graph.parameter(&format!("p{i}"), &[n]);
        let t = graph.input(&format!("t{i}"), &[n]);
        let product = graph.mul(p, t);
        terms.push(graph.sum_all(product));
    }
    let loss = terms
        .into_iter()
        .reduce(|a, b| graph.add(a, b))
        .expect("at least one term");
    graph.set_outputs(vec![loss]);

    let initial: Vec<Vec<f32>> = sizes
        .iter()
        .enumerate()
        .map(|(i, &n)| values(n, 0.21 + i as f32 * 0.05, 1.1))
        .collect();
    let targets: Vec<Vec<f32>> = sizes
        .iter()
        .enumerate()
        .map(|(i, &n)| values(n, 0.37 + i as f32 * 0.1, 0.3))
        .collect();

    let mut session = meganeura::build(&graph, meganeura::SessionConfig::from_env()).0;
    for (i, _) in sizes.iter().enumerate() {
        session.set_parameter(&format!("p{i}"), &initial[i]);
        session.set_input(&format!("t{i}"), &targets[i]);
    }
    session.set_adaptive_grad_clip(clip, pmin);
    session.set_learning_rate(lr);
    session.step();
    session.wait();

    let norm = |v: &[f32]| v.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    for (i, &n) in sizes.iter().enumerate() {
        let upper = clip as f64 * norm(&initial[i]).max(pmin as f64);
        let scale = (upper / norm(&targets[i])).min(1.0);
        let mut after = vec![0.0; n];
        session.read_param(&format!("p{i}"), &mut after);
        for j in [0, n / 2, n - 1] {
            let want = initial[i][j] as f64 - lr as f64 * targets[i][j] as f64 * scale;
            assert!(
                (after[j] as f64 - want).abs() <= 1e-5 * want.abs().max(1e-2),
                "p{i}[{j}]: got {}, want {want}",
                after[j]
            );
        }
    }
}

/// Flash dQ and dK/dV share one per-row dot(dO, O) reduction, including the
/// register and shared-memory KV variants, GQA and causal masking.
#[test]
fn flash_attention_backward_uses_precomputed_row_dots() {
    use meganeura::compile::ShaderEntry;
    // (seq, heads, kv_heads, head_dim, causal): 64-wide heads split each KV
    // row across two threads; 32-wide heads give each row one thread.
    for (seq, heads, kv_heads, head_dim, causal) in [
        (130usize, 4u32, 2u32, 64u32, false),
        (130, 2, 2, 64, true),
        (260, 2, 1, 32, true),
    ] {
        let q_width = heads as usize * head_dim as usize;
        let kv_width = kv_heads as usize * head_dim as usize;
        let mut graph = Graph::new();
        let q = graph.parameter("q", &[seq, q_width]);
        let k = graph.parameter("k", &[seq, kv_width]);
        let v = graph.parameter("v", &[seq, kv_width]);
        let attention = if causal {
            graph.causal_attention(q, k, v, heads, kv_heads, head_dim)
        } else {
            graph.multi_head_attn(q, k, v, heads, kv_heads, head_dim, false)
        };
        let weights = graph.input("weights", &[seq, q_width]);
        let weighted = graph.mul(attention, weights);
        let loss = graph.sum_all(weighted);
        graph.set_outputs(vec![loss]);

        let parameters = [
            ("q", values(seq * q_width, 0.017, 0.3)),
            ("k", values(seq * kv_width, 0.019, 0.7)),
            ("v", values(seq * kv_width, 0.023, 1.1)),
        ];
        let inputs = [("weights", values(seq * q_width, 0.013, 1.7))];
        let mut training = meganeura::build(&graph, meganeura::SessionConfig::from_env()).0;
        let dispatches = &training.plan().dispatches;
        let query = dispatches
            .iter()
            .find(|d| d.shader == ShaderEntry::FlashGradQ)
            .unwrap();
        let kv = dispatches
            .iter()
            .find(|d| d.shader == ShaderEntry::FlashGradKV)
            .unwrap();
        assert_eq!(
            query.input_buffers[5], kv.input_buffers[5],
            "dQ and dK/dV must share the row reduction"
        );
        for (name, data) in &parameters {
            training.set_parameter(name, data);
        }
        training.set_input("weights", &inputs[0].1);
        training.set_learning_rate(0.0);
        training.step();
        training.wait();

        let mut inference =
            meganeura::build(&graph, meganeura::SessionConfig::inference_from_env()).0;
        for (name, data) in &parameters {
            let mut gradient = vec![0.0; data.len()];
            training.read_param_grad(name, &mut gradient);
            let width = data.len() / seq;
            for index in [0, width + 3, (seq / 2) * width + width - 1, data.len() - 1] {
                let numerical =
                    finite_difference(&mut inference, &parameters, &inputs, name, index, 1e-2);
                assert_close(
                    &format!("seq={seq} hd={head_dim} causal={causal} {name}[{index}]"),
                    gradient[index],
                    numerical,
                );
            }
        }
    }
}

/// BCE's backward divided by p(1 - p), which is exactly zero once a
/// sigmoid saturates in f32, while the forward clamps p and stays finite.
/// The denominator is floored the way PyTorch floors it.
#[test]
fn bce_gradient_is_finite_for_saturated_predictions() {
    let mut graph = Graph::new();
    let x = graph.parameter("x", &[4]);
    let p = graph.sigmoid(x);
    let t = graph.input("t", &[4]);
    let loss = graph.bce_loss(p, t);
    graph.set_outputs(vec![loss]);
    let mut session = meganeura::build(&graph, meganeura::SessionConfig::from_env()).0;
    let xs = [30.0f32, -30.0, 0.3, -1.2];
    let ts = [0.0f32, 1.0, 1.0, 0.0];
    session.set_parameter("x", &xs);
    session.set_input("t", &ts);
    session.set_learning_rate(0.0);
    session.step();
    session.wait();
    assert!(session.read_loss().is_finite());
    let mut grad = [0.0f32; 4];
    session.read_param_grad("x", &mut grad);
    assert!(grad.iter().all(|g| g.is_finite()), "gradient {grad:?}");
    // Unsaturated entries keep the exact sigmoid-BCE gradient (p - t) / N.
    for i in 2..4 {
        let p = 1.0 / (1.0 + (-xs[i]).exp());
        let want = (p - ts[i]) / 4.0;
        assert!(
            (grad[i] - want).abs() < 1e-5,
            "grad[{i}] = {}, want {want}",
            grad[i]
        );
    }
}
