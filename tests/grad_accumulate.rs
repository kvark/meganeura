//! Native temporal gradient accumulation: K backward passes summed into
//! the persistent accumulator, then one optimizer step, must match a
//! single optimizer step on the mean of those K gradients.

use meganeura::{Graph, build_session};

fn build() -> Graph {
    // Simple linear regression: loss = mse(x @ w, target).
    let mut g = Graph::new();
    let x = g.input("x", &[2, 3]);
    let w = g.parameter("w", &[3, 2]);
    let y = g.matmul(x, w);
    let t = g.input("t", &[2, 2]);
    let loss = g.mse_loss(y, t);
    g.set_outputs(vec![loss]);
    g
}

const W_INIT: [f32; 6] = [0.1, -0.2, 0.3, 0.05, 0.4, -0.1];

/// Two micro-batches A and B. Accumulating A then B (scale 1/2) then one
/// SGD step must equal one SGD step whose grad is mean(gradA, gradB).
#[test]
fn accumulate_two_micro_equals_mean_grad_step() {
    let a_x = vec![1.0f32, 0.5, -0.3, 0.2, 0.8, 0.1];
    let a_t = vec![0.4f32, -0.1, 0.6, 0.2];
    let b_x = vec![-0.5f32, 0.9, 0.4, -0.2, 0.3, 0.7];
    let b_t = vec![0.1f32, 0.5, -0.4, 0.3];

    // --- Reference: read gradA and gradB separately, mean, manual SGD ---
    let g = build();
    let mut s = build_session(&g);
    s.set_parameter("w", &W_INIT);
    s.clear_optimizer();
    s.set_input("x", &a_x);
    s.set_input("t", &a_t);
    s.step();
    s.wait();
    let mut ga = [0.0f32; 6];
    s.read_param_grad("w", &mut ga);
    s.set_input("x", &b_x);
    s.set_input("t", &b_t);
    s.step();
    s.wait();
    let mut gb = [0.0f32; 6];
    s.read_param_grad("w", &mut gb);
    let lr = 0.1f32;
    let mut want = W_INIT;
    for i in 0..6 {
        want[i] -= lr * 0.5 * (ga[i] + gb[i]);
    }

    // --- Native accumulation: zero_grad, A, B (scale 1/2), then SGD ---
    let mut s = build_session(&g);
    s.set_parameter("w", &W_INIT);
    s.set_grad_accumulate(2);
    s.zero_grad();
    s.clear_optimizer(); // micro-batch A: accumulate only
    s.set_input("x", &a_x);
    s.set_input("t", &a_t);
    s.step();
    s.set_learning_rate(lr); // micro-batch B: accumulate + apply
    s.set_input("x", &b_x);
    s.set_input("t", &b_t);
    s.step();
    s.wait();
    let mut got = [0.0f32; 6];
    s.read_param("w", &mut got);

    let max = want
        .iter()
        .zip(&got)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    println!("want={want:?}\ngot ={got:?}\nmax diff={max:.3e}");
    assert!(max < 1e-5, "native accumulation != mean-grad step: {max}");
}

/// zero_grad must actually clear: a second accumulate window starting
/// with zero_grad reproduces the first (no carryover).
#[test]
fn zero_grad_clears_accumulator() {
    let x = vec![1.0f32, 0.5, -0.3, 0.2, 0.8, 0.1];
    let t = vec![0.4f32, -0.1, 0.6, 0.2];
    let g = build();
    let mut s = build_session(&g);
    s.set_parameter("w", &W_INIT);
    s.set_grad_accumulate(1.max(1)); // enable accumulator buffers
    s.set_grad_accumulate(2);

    let mut step_once = |s: &mut meganeura::Session| -> [f32; 6] {
        s.set_parameter("w", &W_INIT);
        s.zero_grad();
        s.clear_optimizer();
        s.set_input("x", &x);
        s.set_input("t", &t);
        s.step();
        s.set_learning_rate(0.1);
        s.set_input("x", &x);
        s.set_input("t", &t);
        s.step();
        s.wait();
        let mut p = [0.0f32; 6];
        s.read_param("w", &mut p);
        p
    };
    let first = step_once(&mut s);
    let second = step_once(&mut s);
    let max = first
        .iter()
        .zip(&second)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(max < 1e-6, "zero_grad failed to clear; carryover {max}");
}
