//! Debug-session observability: read any node's value by name after a step,
//! and attribute the first NaN to a dispatch/graph node via `step_debug`.

use meganeura::graph::Graph;
use meganeura::nn::Linear;
use meganeura::runtime::ReadNodeError;
use meganeura::{Mode, SessionConfig, build};

fn build_mlp() -> Graph {
    let mut g = Graph::new();
    let x = g.input("x", &[2, 4]);
    let fc1 = Linear::no_bias(&mut g, "fc1", 4, 3);
    let h = fc1.forward(&mut g, x);
    let h = g.relu(h);
    let h = g.named(h, "act");
    let fc2 = Linear::no_bias(&mut g, "fc2", 3, 2);
    let y = fc2.forward(&mut g, h);
    g.set_outputs(vec![y]);
    g
}

#[test]
fn read_node_by_name_matches_cpu() {
    let g = build_mlp();
    let (mut s, _) = build(
        &g,
        SessionConfig {
            mode: Mode::Inference,
            debug: true,
            ..SessionConfig::default()
        },
    );

    let x = [1.0f32, -2.0, 3.0, 0.5, 0.0, 1.5, -1.0, 2.0];
    let w1: Vec<f32> = (0..12).map(|i| (i as f32) * 0.1 - 0.5).collect();
    let w2 = [0.3f32; 6];
    s.set_input("x", &x);
    s.set_parameter("fc1.weight", &w1);
    s.set_parameter("fc2.weight", &w2);
    s.step();
    s.wait();

    // CPU reference for fc1 output (x[2,4] @ w1[4,3]).
    let mut expect = [0.0f32; 6];
    for r in 0..2 {
        for c in 0..3 {
            for k in 0..4 {
                expect[r * 3 + c] += x[r * 4 + k] * w1[k * 3 + c];
            }
        }
    }

    let got = s.read_node_by_name("fc1").expect("fc1 readable");
    assert_eq!(got.len(), 6);
    for (i, (&g_v, &e_v)) in got.iter().zip(expect.iter()).enumerate() {
        assert!(
            (g_v - e_v).abs() < 1e-5,
            "fc1[{i}]: got {g_v}, expected {e_v}"
        );
    }

    // The relu'd activation is also readable by its name.
    let act = s.read_node_by_name("act").expect("act readable");
    for (i, (&a, &e)) in act.iter().zip(expect.iter()).enumerate() {
        assert!(
            (a - e.max(0.0)).abs() < 1e-5,
            "act[{i}]: got {a}, expected {}",
            e.max(0.0)
        );
    }

    // Unknown names fail with the available names listed.
    match s.read_node_by_name("nope") {
        Err(ReadNodeError::UnknownName(names)) => {
            assert!(names.iter().any(|n| n == "fc1"));
        }
        other => panic!("expected UnknownName, got {other:?}"),
    }
}

#[test]
fn step_debug_attributes_first_nan() {
    let g = build_mlp();
    let (mut s, _) = build(
        &g,
        SessionConfig {
            mode: Mode::Inference,
            debug: true,
            ..SessionConfig::default()
        },
    );

    let mut x = [0.5f32; 8];
    x[3] = f32::NAN;
    s.set_input("x", &x);
    s.set_parameter("fc1.weight", &[0.1; 12]);
    s.set_parameter("fc2.weight", &[0.2; 6]);

    let report = s.step_debug();
    let first = report.first_bad().expect("NaN input must be detected");
    assert!(first.has_nan);
    // The earliest offender must be the fc1 matmul (the first consumer of
    // the poisoned input), and its label must say so by name.
    assert!(
        first.label.contains("fc1"),
        "first anomaly label should name fc1, got {:?}",
        first.label
    );
    assert!(!first.origin.is_empty());
}
