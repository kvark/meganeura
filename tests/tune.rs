//! Real-device scalar tile qualification. These tests execute GPU timings;
//! run explicitly on an idle device with `--ignored --test-threads=1`.

use meganeura::graph::Graph;
use meganeura::{CoopPolicy, Mode, SessionConfig, SessionOptions, TuneOptions, build};
use std::time::Duration;

#[test]
#[ignore = "GPU tuning requires an idle device"]
fn tune_preserves_correctness() {
    let mut g = Graph::new();
    let x = g.input("x", &[64, 64]);
    let w1 = g.parameter("w1", &[64, 64]);
    let w2 = g.parameter("w2", &[64, 64]);
    let h = g.matmul(x, w1);
    let y = g.matmul(h, w2);
    g.set_outputs(vec![y]);

    let (mut s, _) = build(
        &g,
        SessionConfig {
            mode: Mode::Inference,
            runtime: SessionOptions {
                coop: CoopPolicy::Disabled,
                ..Default::default()
            },
            ..Default::default()
        },
    );

    let x_data: Vec<f32> = (0..64 * 64).map(|i| ((i % 13) as f32) * 0.05).collect();
    let w_data: Vec<f32> = (0..64 * 64)
        .map(|i| ((i % 7) as f32) * 0.02 - 0.05)
        .collect();
    s.set_input("x", &x_data);
    s.set_parameter("w1", &w_data);
    s.set_parameter("w2", &w_data);

    s.step();
    s.wait();
    let before = s.read_output(64 * 64);

    let report = s
        .tune_with(TuneOptions {
            max_time: Duration::from_secs(60),
            ..Default::default()
        })
        .unwrap();
    assert!(!report.outcomes.is_empty());
    for o in &report.outcomes {
        assert!(o.dispatches > 0);
        assert!(o.qualified, "qualification failed: {o:?}");
        assert!(o.baseline_median_ms.unwrap() > 0.0);
        assert!(o.candidate_median_ms.unwrap() > 0.0);
    }
    // Tuning must not execute the graph or overwrite the previous output.
    assert_eq!(
        before.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
        s.read_output(64 * 64)
            .iter()
            .map(|x| x.to_bits())
            .collect::<Vec<_>>()
    );

    s.step();
    s.wait();
    let after = s.read_output(64 * 64);
    for (i, (&b, &a)) in before.iter().zip(after.iter()).enumerate() {
        assert!(
            (b - a).abs() <= 1e-4 * b.abs().max(1.0),
            "output[{i}] changed after tune: {b} vs {a}"
        );
    }
}

#[test]
#[ignore = "GPU tuning requires an idle device"]
fn tune_preserves_training_parameters_gradients_and_moments() {
    let mut graph = Graph::new();
    let x = graph.input("x", &[33, 17]);
    let weight = graph.parameter("weight", &[17, 65]);
    let y = graph.matmul(x, weight);
    let loss = graph.mean_all(y);
    graph.set_outputs(vec![loss]);
    let (mut session, _) = build(
        &graph,
        SessionConfig {
            runtime: SessionOptions {
                debug: true,
                coop: CoopPolicy::Disabled,
                ..Default::default()
            },
            ..Default::default()
        },
    );
    session.set_input("x", &vec![0.25; 33 * 17]);
    session.set_parameter("weight", &vec![0.125; 17 * 65]);
    session.set_adam(0.001, 0.9, 0.999, 1.0e-8);
    session.set_grad_accumulate(2);
    session.step();
    session.wait();
    let snapshot = |s: &meganeura::Session| {
        let mut values = vec![s.adam_step_count()];
        for (index, &bytes) in s.plan().buffers.iter().enumerate() {
            let mut data = vec![0.0; bytes / 4];
            s.read_buffer(meganeura::compile::BufferRef(index as u32), &mut data);
            values.extend(data.into_iter().map(f32::to_bits));
        }
        for (m, v) in s.read_adam_states(&["weight"]) {
            values.extend(m.into_iter().chain(v).map(f32::to_bits));
        }
        values
    };
    let before = snapshot(&session);
    let report = session
        .tune_with(TuneOptions {
            max_time: Duration::from_secs(60),
            ..Default::default()
        })
        .unwrap();
    assert!(report.outcomes.iter().any(|o| o.qualified), "{report:?}");
    assert_eq!(before, snapshot(&session));
}
