//! Eager evaluation: values while building, identical results to the
//! compiled session, valid NodeIds across graph growth.

use meganeura::eager::Eager;
use meganeura::graph::Graph;
use meganeura::{Mode, SessionConfig, build};

#[test]
fn eval_while_building_matches_compiled_session() {
    let mut g = Graph::new();
    let x = g.input("x", &[2, 4]);
    let w1 = g.parameter("w1", &[4, 3]);
    let h = g.matmul(x, w1);

    let x_data: Vec<f32> = (0..8).map(|i| (i as f32) * 0.25 - 1.0).collect();
    let w1_data: Vec<f32> = (0..12).map(|i| ((i % 5) as f32) * 0.3 - 0.6).collect();

    let mut e = Eager::new();
    e.set_input("x", x_data.clone());
    e.set_parameter("w1", w1_data.clone());

    // Evaluate mid-build.
    let t_h = e.eval(&g, h);
    assert_eq!(t_h.shape, vec![2, 3]);

    // CPU reference for h.
    for r in 0..2 {
        for c in 0..3 {
            let mut acc = 0.0f32;
            for k in 0..4 {
                acc += x_data[r * 4 + k] * w1_data[k * 3 + c];
            }
            let got = t_h.data[r * 3 + c];
            assert!((got - acc).abs() < 1e-5, "h[{r},{c}] {got} vs {acc}");
        }
    }

    // Keep building the same graph; earlier ids stay valid.
    let y = g.relu(h);
    let t_y = e.eval(&g, y);
    let t_h2 = e.eval(&g, h);
    assert_eq!(t_h.data, t_h2.data, "h unchanged after growth");
    for (a, b) in t_h.data.iter().zip(t_y.data.iter()) {
        assert_eq!(a.max(0.0), *b, "relu mismatch");
    }

    // The very same graph compiles for the fast path with identical results.
    g.set_outputs(vec![y]);
    let (mut s, _) = build(
        &g,
        SessionConfig {
            mode: Mode::Inference,
            ..SessionConfig::from_env()
        },
    );
    s.set_input("x", &x_data);
    s.set_parameter("w1", &w1_data);
    s.step();
    s.wait();
    let compiled = s.read_output(6);
    for (i, (a, b)) in t_y.data.iter().zip(compiled.iter()).enumerate() {
        assert!(
            (a - b).abs() <= 1e-5,
            "eager vs compiled mismatch at [{i}]: {a} vs {b}"
        );
    }
}

#[test]
fn eval_dead_branch_and_rebind() {
    let mut g = Graph::new();
    let x = g.input("x", &[4]);
    let a = g.relu(x);
    let _unused = g.neg(x); // a branch nothing consumes — still evaluable

    let mut e = Eager::new();
    e.set_input("x", vec![-1.0, 2.0, -3.0, 4.0]);
    assert_eq!(e.eval(&g, a).data, vec![0.0, 2.0, 0.0, 4.0]);
    assert_eq!(e.eval(&g, _unused).data, vec![1.0, -2.0, 3.0, -4.0]);

    // Rebinding an input re-executes.
    e.set_input("x", vec![1.0, -2.0, 3.0, -4.0]);
    assert_eq!(e.eval(&g, a).data, vec![1.0, 0.0, 3.0, 0.0]);
}
