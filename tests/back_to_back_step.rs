/// Verify that back-to-back `step()` calls (no intervening reset)
/// produce two separate parameter updates — the kindle SIL pattern.
///
/// The kindle SIL update fires immediately after the regular policy
/// update, with different inputs and a fresh `set_learning_rate`.
/// Empirically the SIL pattern has zero downstream effect; this test
/// isolates whether the meganeura session correctly applies a second
/// optimizer step in this pattern.
use meganeura::Graph;

fn build_linear_regression(batch_size: usize, in_dim: usize, out_dim: usize) -> Graph {
    let mut g = Graph::new();
    let x = g.input("x", &[batch_size, in_dim]);
    let target = g.input("target", &[batch_size, out_dim]);
    let w = meganeura::nn::Linear::new(&mut g, "w", in_dim, out_dim);
    let y = w.forward(&mut g, x);
    let loss = g.mse_loss(y, target);
    g.set_outputs(vec![loss, y]);
    g
}

fn snapshot_params(session: &meganeura::Session, name: &str, n: usize) -> Vec<f32> {
    let mut buf = vec![0.0f32; n];
    session.read_param(name, &mut buf);
    buf
}

#[test]
fn back_to_back_step_applies_two_distinct_updates() {
    let batch = 4;
    let in_d = 3;
    let out_d = 2;
    let g = build_linear_regression(batch, in_d, out_d);
    let mut s = meganeura::build(&g, meganeura::SessionConfig::from_env()).0;

    // Init params deterministically.
    let n_w = in_d * out_d;
    let n_b = out_d;
    let w_init: Vec<f32> = (0..n_w).map(|i| (i as f32 * 0.1).sin()).collect();
    let b_init: Vec<f32> = vec![0.0; n_b];
    s.set_parameter("w.weight", &w_init);
    s.set_parameter("w.bias", &b_init);

    // Two distinct (input, target) batches.
    let x_a: Vec<f32> = (0..batch * in_d).map(|i| (i as f32 * 0.3).cos()).collect();
    let t_a: Vec<f32> = (0..batch * out_d).map(|i| (i as f32 * 0.5).sin()).collect();
    let x_b: Vec<f32> = (0..batch * in_d)
        .map(|i| (i as f32 * 0.7 + 1.0).cos())
        .collect();
    let t_b: Vec<f32> = (0..batch * out_d)
        .map(|i| (i as f32 * 0.4 + 2.0).sin())
        .collect();

    // ============= Step 1: input A =============
    s.set_input("x", &x_a);
    s.set_input("target", &t_a);
    s.set_learning_rate(0.1);
    s.step();
    s.wait();
    let loss1 = s.read_loss();
    let p_after_1 = snapshot_params(&s, "w.weight", n_w);

    eprintln!("after step 1: loss={:.4}", loss1);
    let p_diff_1: f32 = p_after_1
        .iter()
        .zip(w_init.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    eprintln!("  param L1 change from init: {:.6}", p_diff_1);
    assert!(
        p_diff_1 > 1e-6,
        "step 1 must change params; saw L1 diff {}",
        p_diff_1
    );

    // ============= Step 2: input B (back-to-back, no reset) =============
    s.set_input("x", &x_b);
    s.set_input("target", &t_b);
    s.set_learning_rate(0.1);
    s.step();
    s.wait();
    let loss2 = s.read_loss();
    let p_after_2 = snapshot_params(&s, "w.weight", n_w);

    eprintln!("after step 2: loss={:.4}", loss2);
    let p_diff_2: f32 = p_after_2
        .iter()
        .zip(p_after_1.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    eprintln!("  param L1 change from step 1: {:.6}", p_diff_2);

    assert!(
        p_diff_2 > 1e-6,
        "step 2 (back-to-back, different input) must change params; saw L1 diff {}",
        p_diff_2
    );

    // ============= Step 3: input A again =============
    s.set_input("x", &x_a);
    s.set_input("target", &t_a);
    s.set_learning_rate(0.1);
    s.step();
    s.wait();
    let p_after_3 = snapshot_params(&s, "w.weight", n_w);
    let p_diff_3: f32 = p_after_3
        .iter()
        .zip(p_after_2.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    eprintln!("  param L1 change after step 3: {:.6}", p_diff_3);
    assert!(p_diff_3 > 1e-6, "step 3 must change params");
}

#[test]
fn set_adam_persists_across_steps_without_re_arming() {
    // Calling `set_adam` once should keep Adam updates running on every
    // subsequent `step()`. Previously the config was one-shot, silently
    // stopping after the first step.
    let batch = 4;
    let in_d = 3;
    let out_d = 2;
    let g = build_linear_regression(batch, in_d, out_d);
    let mut s = meganeura::build(&g, meganeura::SessionConfig::from_env()).0;

    let n_w = in_d * out_d;
    let w_init: Vec<f32> = (0..n_w).map(|i| (i as f32 * 0.1).sin()).collect();
    s.set_parameter("w.weight", &w_init);
    s.set_parameter("w.bias", &vec![0.0; out_d]);

    let x: Vec<f32> = (0..batch * in_d).map(|i| (i as f32 * 0.3).cos()).collect();
    let t: Vec<f32> = (0..batch * out_d).map(|i| (i as f32 * 0.5).sin()).collect();
    s.set_input("x", &x);
    s.set_input("target", &t);

    // Configure Adam ONCE.
    s.set_adam(0.1, 0.9, 0.999, 1e-8);

    let mut last = w_init.clone();
    for step_idx in 0..3 {
        s.step();
        s.wait();
        let now = snapshot_params(&s, "w.weight", n_w);
        let diff: f32 = now
            .iter()
            .zip(last.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-6,
            "step {step_idx} (Adam configured once, called repeatedly) must keep updating params; saw L1 diff {diff}",
        );
        last = now;
    }
}

#[test]
fn clear_optimizer_stops_updates() {
    // After `clear_optimizer`, `step()` runs forward+backward but no
    // parameter update — params stay frozen even with non-zero LR
    // previously set.
    let batch = 4;
    let in_d = 3;
    let out_d = 2;
    let g = build_linear_regression(batch, in_d, out_d);
    let mut s = meganeura::build(&g, meganeura::SessionConfig::from_env()).0;

    let n_w = in_d * out_d;
    let w_init: Vec<f32> = (0..n_w).map(|i| (i as f32 * 0.1).sin()).collect();
    s.set_parameter("w.weight", &w_init);
    s.set_parameter("w.bias", &vec![0.0; out_d]);

    let x: Vec<f32> = (0..batch * in_d).map(|i| (i as f32 * 0.3).cos()).collect();
    let t: Vec<f32> = (0..batch * out_d).map(|i| (i as f32 * 0.5).sin()).collect();
    s.set_input("x", &x);
    s.set_input("target", &t);

    s.set_learning_rate(0.1);
    s.step();
    s.wait();
    let after_train = snapshot_params(&s, "w.weight", n_w);

    s.clear_optimizer();
    s.step();
    s.wait();
    let after_freeze = snapshot_params(&s, "w.weight", n_w);

    let diff: f32 = after_train
        .iter()
        .zip(after_freeze.iter())
        .map(|(a, b)| (a - b).abs())
        .sum();
    assert!(
        diff < 1e-6,
        "post-clear_optimizer step must not change params; saw L1 diff {diff}",
    );
}

#[test]
fn back_to_back_with_zero_lr_in_between_does_not_compound() {
    // Confirms set_learning_rate(0) in a step is a no-op for params.
    let batch = 4;
    let in_d = 3;
    let out_d = 2;
    let g = build_linear_regression(batch, in_d, out_d);
    let mut s = meganeura::build(&g, meganeura::SessionConfig::from_env()).0;

    let n_w = in_d * out_d;
    let w_init: Vec<f32> = (0..n_w).map(|i| (i as f32 * 0.1).sin()).collect();
    s.set_parameter("w.weight", &w_init);
    s.set_parameter("w.bias", &vec![0.0; out_d]);

    let x_a: Vec<f32> = (0..batch * in_d).map(|i| (i as f32 * 0.3).cos()).collect();
    let t_a: Vec<f32> = (0..batch * out_d).map(|i| (i as f32 * 0.5).sin()).collect();

    // Step 1: lr=0.1
    s.set_input("x", &x_a);
    s.set_input("target", &t_a);
    s.set_learning_rate(0.1);
    s.step();
    s.wait();
    let p1 = snapshot_params(&s, "w.weight", n_w);

    // Step 2: lr=0 — should NOT change params
    s.set_input("x", &x_a);
    s.set_input("target", &t_a);
    s.set_learning_rate(0.0);
    s.step();
    s.wait();
    let p2 = snapshot_params(&s, "w.weight", n_w);
    let diff: f32 = p1.iter().zip(p2.iter()).map(|(a, b)| (a - b).abs()).sum();
    eprintln!("zero-lr step diff: {:.6}", diff);
    assert!(
        diff < 1e-6,
        "zero-LR step should not change params; saw L1 diff {}",
        diff
    );
}
