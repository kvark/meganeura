//! Session-build tuner: `Session::tune` measures each flippable kernel
//! family both ways on real `step()` wall-clock and the session stays
//! numerically correct with whichever variant it keeps.

use meganeura::graph::Graph;
use meganeura::{Mode, SessionConfig, build};

#[test]
fn tune_preserves_correctness() {
    // Big enough matmul chain that coop promotion (if the device supports
    // it) actually triggers; on scalar-only adapters tune() is a no-op.
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
            ..SessionConfig::from_env()
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

    let outcomes = s.tune();
    // On coop-capable devices at least the matmul family is measured; on
    // scalar-only devices there is nothing to flip.
    for o in &outcomes {
        assert!(o.dispatches > 0);
        assert!(o.coop_ms > 0.0 && o.scalar_ms > 0.0);
    }

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
