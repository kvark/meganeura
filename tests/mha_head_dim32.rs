//! Regression test for scalar MHA backward with head dimensions below 64.
//!
//! The hand-written backward kernels use a fixed 64-thread reduction. Before
//! inactive lanes were zero-padded, `head_dim=32` made lanes 32..63 read the
//! next head (or the next sequence row), corrupting every Q/K/V derivative.

use meganeura::{Graph, Session, build_inference_session, build_session};

fn values(n: usize, frequency: f32, phase: f32) -> Vec<f32> {
    (0..n)
        .map(|i| (i as f32 * frequency + phase).sin() * 0.2)
        .collect()
}

fn set_parameters(session: &mut Session, q: &[f32], k: &[f32], v: &[f32]) {
    session.set_parameter("q", q);
    session.set_parameter("k", k);
    session.set_parameter("v", v);
}

fn build_graph() -> Graph {
    let (seq, heads, head_dim) = (4usize, 2u32, 32u32);
    let width = heads as usize * head_dim as usize;
    let mut graph = Graph::new();
    let q = graph.parameter("q", &[seq, width]);
    let k = graph.parameter("k", &[seq, width]);
    let v = graph.parameter("v", &[seq, width]);
    let attention = graph.multi_head_attn(q, k, v, heads, heads, head_dim, false);
    let weights = graph.input("weights", &[seq, width]);
    let weighted = graph.mul(attention, weights);
    let loss = graph.mean_all(weighted);
    graph.set_outputs(vec![loss]);
    graph
}

#[test]
fn scalar_attention_backward_masks_inactive_lanes() {
    let graph = build_graph();
    let n = 4 * 2 * 32;
    let q = values(n, 0.017, 0.3);
    let k = values(n, 0.019, 0.7);
    let v = values(n, 0.023, 1.1);
    let weights = values(n, 0.013, 1.7);

    let mut training = build_session(&graph);
    set_parameters(&mut training, &q, &k, &v);
    training.set_input("weights", &weights);
    training.step();
    training.wait();

    let mut analytical = [
        ("q", q.clone(), vec![0.0; n]),
        ("k", k.clone(), vec![0.0; n]),
        ("v", v.clone(), vec![0.0; n]),
    ];
    for (name, _, gradient) in &mut analytical {
        training.read_param_grad(name, gradient);
    }

    let mut inference = build_inference_session(&graph);
    let finite_difference =
        |session: &mut Session, name: &str, data: &mut [f32], index: usize| -> f32 {
            let original = data[index];
            data[index] = original + 1e-3;
            match name {
                "q" => set_parameters(session, data, &k, &v),
                "k" => set_parameters(session, &q, data, &v),
                "v" => set_parameters(session, &q, &k, data),
                _ => unreachable!(),
            }
            session.set_input("weights", &weights);
            session.step();
            session.wait();
            let plus = session.read_loss();

            data[index] = original - 1e-3;
            match name {
                "q" => set_parameters(session, data, &k, &v),
                "k" => set_parameters(session, &q, data, &v),
                "v" => set_parameters(session, &q, &k, data),
                _ => unreachable!(),
            }
            session.set_input("weights", &weights);
            session.step();
            session.wait();
            let minus = session.read_loss();
            data[index] = original;
            (plus - minus) / 2e-3
        };

    // Probe both sides of the 32-element head boundary and multiple rows.
    for (name, mut data, gradient) in analytical {
        for index in [0usize, 31, 32, 63, 129, 223] {
            let numerical = finite_difference(&mut inference, name, &mut data, index);
            let actual = gradient[index];
            let abs_error = (actual - numerical).abs();
            let scale = actual.abs().max(numerical.abs()).max(1e-5);
            assert!(
                abs_error < 2e-4 || abs_error / scale < 0.08,
                "{name}[{index}]: analytical={actual:+.6e}, numerical={numerical:+.6e}"
            );
        }
    }
}
