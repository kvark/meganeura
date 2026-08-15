//! Verify the Adam-state read/write API: `read_adam_m`, `read_adam_v`,
//! `read_adam_states`, `write_adam_m`, `write_adam_v`, `adam_step_count`,
//! `set_adam_step_count`. These are the building blocks for carrying
//! optimizer state across a session rebuild — e.g. when a downstream
//! training loop reshapes a parameter table (RadFoam densification was
//! the original motivating use case).

use meganeura::Graph;

#[test]
fn adam_state_roundtrip_and_step_counter() {
    let n = 8usize;

    // Toy graph that gives `log_density` a non-zero gradient: pred = p
    // (plus a dead `x * 0` to satisfy the Trainer contract that `x` is
    // an input), mse against constant labels.
    let mut g = Graph::new();
    let x = g.input("x", &[1, n]);
    let labels = g.input("labels", &[1, n]);
    let p = g.parameter("log_density", &[1, n]);
    let zero = g.constant(vec![0.0_f32; n], &[1, n]);
    let dead = g.mul(x, zero);
    let pred = g.add(p, dead);
    let loss = g.mse_loss(pred, labels);
    g.set_outputs(vec![loss]);

    let mut sess = meganeura::build(&g, meganeura::SessionConfig::from_env()).0;
    sess.set_adam(0.01, 0.9, 0.999, 1e-8);
    sess.set_parameter("log_density", &vec![1.0_f32; n]);
    sess.set_input("x", &vec![0.0_f32; n]);
    sess.set_input("labels", &vec![0.0_f32; n]);
    sess.step();
    sess.wait();

    // After one Adam step with non-zero gradient, m should be non-zero.
    let mut m = vec![0.0_f32; n];
    sess.read_adam_m("log_density", &mut m);
    assert!(
        m.iter().any(|&v| v.abs() > 1e-9),
        "m all-zero after one Adam step; got {:?}",
        m
    );

    let mut v = vec![0.0_f32; n];
    sess.read_adam_v("log_density", &mut v);
    assert!(
        v.iter().any(|&val| val.abs() > 1e-9),
        "v all-zero after one Adam step; got {:?}",
        v
    );

    assert_eq!(
        sess.adam_step_count(),
        1,
        "adam_step_count should be 1 after one step"
    );

    // Overwrite m, v with a known pattern, set t to 42, verify roundtrip.
    let m_pattern: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
    let v_pattern: Vec<f32> = (0..n).map(|i| (i as f32) * 0.001 + 0.5).collect();
    sess.write_adam_m("log_density", &m_pattern);
    sess.write_adam_v("log_density", &v_pattern);
    sess.set_adam_step_count(42);

    let mut m_back = vec![0.0_f32; n];
    sess.read_adam_m("log_density", &mut m_back);
    for (i, (&got, &want)) in m_back.iter().zip(m_pattern.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-6,
            "m[{i}] roundtrip: got {got}, want {want}"
        );
    }
    let mut v_back = vec![0.0_f32; n];
    sess.read_adam_v("log_density", &mut v_back);
    for (i, (&got, &want)) in v_back.iter().zip(v_pattern.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-6,
            "v[{i}] roundtrip: got {got}, want {want}"
        );
    }
    assert_eq!(sess.adam_step_count(), 42);

    // Step counter increments on next step() — confirms set_adam_step_count
    // actually took effect (would still be 2 if it was ignored).
    sess.set_input("labels", &vec![0.6_f32; n]);
    sess.step();
    sess.wait();
    assert_eq!(
        sess.adam_step_count(),
        43,
        "step counter must increment from set value 42 to 43, not from 2 to 3"
    );
}

#[test]
fn batched_adam_state_read_preserves_names_and_values() {
    let mut graph = Graph::new();
    let a = graph.parameter("a", &[3]);
    let b = graph.parameter("b", &[2]);
    let a_mean = graph.mean_all(a);
    let b_mean = graph.mean_all(b);
    let loss = graph.add(a_mean, b_mean);
    graph.set_outputs(vec![loss]);

    let mut session = meganeura::build_session(&graph);
    session.set_adam(0.01, 0.9, 0.999, 1.0e-8);
    session.set_parameter("a", &[1.0, 2.0, 3.0]);
    session.set_parameter("b", &[4.0, 5.0]);
    session.step();
    session.wait();

    let a_m = [0.1, 0.2, 0.3];
    let a_v = [1.1, 1.2, 1.3];
    let b_m = [-0.4, -0.5];
    let b_v = [2.4, 2.5];
    session.write_adam_m("a", &a_m);
    session.write_adam_v("a", &a_v);
    session.write_adam_m("b", &b_m);
    session.write_adam_v("b", &b_v);

    let states = session.read_adam_states(&["b", "a"]);
    assert_eq!(states[0], (b_m.to_vec(), b_v.to_vec()));
    assert_eq!(states[1], (a_m.to_vec(), a_v.to_vec()));
}

#[test]
fn adam_grouped_grad_norm_accumulates_exact_temporal_magnitudes() {
    let mut graph = Graph::new();
    let positions = graph.parameter("positions", &[2, 3]);
    let loss = graph.mean_all(positions);
    graph.set_outputs(vec![loss]);

    let mut session = meganeura::build_session(&graph);
    session.set_parameter("positions", &[1.0; 6]);
    session.set_adam_grouped_grad_norm("positions", 3);
    session.set_adam(0.01, 0.9, 0.999, 1.0e-8);

    for _ in 0..2 {
        session.step();
        session.wait();
    }

    let accumulated = session.read_adam_grouped_grad_norm("positions");
    let expected = 2.0 * (3.0_f32 / 36.0).sqrt();
    for (index, actual) in accumulated.into_iter().enumerate() {
        assert!(
            (actual - expected).abs() < 1.0e-6,
            "group {index}: got {actual}, expected {expected}",
        );
    }
}
