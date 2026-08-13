/// Verify gradient norm clipping integrates correctly with `step()`.
///
/// Sets up a single linear-regression step where the unclipped
/// gradient is large enough to push the parameter past the target
/// in one update; with `set_grad_clip_norm(small)` the update should
/// become much smaller.
use meganeura::Graph;

fn build_linreg(batch: usize, in_d: usize, out_d: usize) -> Graph {
    let mut g = Graph::new();
    let x = g.input("x", &[batch, in_d]);
    let target = g.input("target", &[batch, out_d]);
    let w = meganeura::nn::Linear::new(&mut g, "w", in_d, out_d);
    let y = w.forward(&mut g, x);
    let loss = g.mse_loss(y, target);
    g.set_outputs(vec![loss, y]);
    g
}

fn run_one_step(grad_clip: Option<f32>) -> Vec<f32> {
    let g = build_linreg(4, 3, 2);
    let mut s = meganeura::build(&g, meganeura::SessionConfig::from_env()).0;

    // Same init for every run so we can compare.
    let w_init: Vec<f32> = (0..6).map(|i| (i as f32 * 0.1).sin()).collect();
    s.set_parameter("w.weight", &w_init);
    s.set_parameter("w.bias", &[0.0; 2]);

    // Inputs designed to give a large, easy-to-reason-about gradient.
    let x: Vec<f32> = (0..12).map(|i| (i as f32 * 0.3).cos()).collect();
    let t: Vec<f32> = vec![10.0; 8]; // very far from y(x) under any reasonable init
    s.set_input("x", &x);
    s.set_input("target", &t);

    if let Some(max_norm) = grad_clip {
        s.set_grad_clip_norm(max_norm);
    }
    s.set_learning_rate(0.1);
    s.step();
    s.wait();

    let mut params = vec![0.0; 6];
    s.read_param("w.weight", &mut params);
    params
}

#[test]
fn grad_clip_reduces_param_change() {
    let init: Vec<f32> = (0..6).map(|i| (i as f32 * 0.1).sin()).collect();

    let p_unclipped = run_one_step(None);
    let p_clipped = run_one_step(Some(0.01)); // tight clip

    let l1 = |after: &[f32]| -> f32 {
        after
            .iter()
            .zip(init.iter())
            .map(|(a, b)| (a - b).abs())
            .sum()
    };

    let unclipped_change = l1(&p_unclipped);
    let clipped_change = l1(&p_clipped);

    eprintln!("unclipped param L1 change: {:.6}", unclipped_change);
    eprintln!("clipped   param L1 change: {:.6}", clipped_change);

    assert!(
        unclipped_change > 0.001,
        "unclipped step should produce nontrivial param movement; got {}",
        unclipped_change
    );
    assert!(
        clipped_change < unclipped_change,
        "clipped step should produce smaller param change than unclipped; got clipped={} unclipped={}",
        clipped_change,
        unclipped_change
    );
    // Tight clip should bound the change to roughly clip_norm * lr.
    // With max_norm=0.01 and lr=0.1, expected per-step delta ~ 0.001.
    // Allow generous slack for direction-dependent norm contributions.
    assert!(
        clipped_change < 0.01,
        "clipped change {} should be near max_norm*lr=0.001, well under 0.01",
        clipped_change
    );
}

#[test]
fn grad_clip_zero_disables() {
    let p_unclipped = run_one_step(None);
    let p_zero_clip = run_one_step(Some(0.0)); // 0.0 means disabled

    // They should produce the same result — passing 0.0 should be a no-op.
    let same: bool = p_unclipped
        .iter()
        .zip(p_zero_clip.iter())
        .all(|(a, b)| (a - b).abs() < 1e-6);
    assert!(same, "grad_clip=0.0 must be a no-op");
}

/// `grad_clip_every` skips the clip on `every-1` of every `every` steps.
/// With `every=10` and a single step, the clip should be skipped, so the
/// param change should match the unclipped run.
#[test]
fn grad_clip_every_skips_clip() {
    let g = build_linreg(4, 3, 2);
    let mut s = meganeura::build(&g, meganeura::SessionConfig::from_env()).0;
    let w_init: Vec<f32> = (0..6).map(|i| (i as f32 * 0.1).sin()).collect();
    s.set_parameter("w.weight", &w_init);
    s.set_parameter("w.bias", &[0.0; 2]);
    let x: Vec<f32> = (0..12).map(|i| (i as f32 * 0.3).cos()).collect();
    let t: Vec<f32> = vec![10.0; 8];
    s.set_input("x", &x);
    s.set_input("target", &t);
    s.set_grad_clip_norm(0.001); // very tight clip — would massively shrink update
    s.set_grad_clip_every(10); // but skip 9 of every 10 steps
    s.set_learning_rate(0.1);
    s.step();
    s.wait();
    let mut p1 = vec![0.0; 6];
    s.read_param("w.weight", &mut p1);

    // Compare to unclipped run with same setup
    let p_unclipped = run_one_step(None);

    // Step 1 with `every=10` should NOT clip (1 % 10 = 1 != 0), so the
    // result matches unclipped.
    let max_diff: f32 = p1
        .iter()
        .zip(p_unclipped.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-5,
        "step 1 with every=10 should match unclipped exactly; max diff {}",
        max_diff
    );
}

#[test]
fn cpu_grad_clip_stages_device_local_gradients() {
    let g = build_linreg(4, 3, 2);
    let mut session = meganeura::build(&g, meganeura::SessionConfig::from_env()).0;
    session.set_parameter("w.weight", &[1.0; 6]);
    session.set_parameter("w.bias", &[0.0; 2]);
    session.set_input("x", &[1.0; 12]);
    session.set_input("target", &[10.0; 8]);
    session.clear_optimizer();
    session.step();
    session.wait();

    let (before, clipped) = session.clip_grad_norm_cpu(0.01);
    assert!(clipped);
    assert!(before > 0.01);

    let mut weight_gradient = [0.0; 6];
    let mut bias_gradient = [0.0; 2];
    session.read_param_grad("w.weight", &mut weight_gradient);
    session.read_param_grad("w.bias", &mut bias_gradient);
    let after = weight_gradient
        .into_iter()
        .chain(bias_gradient)
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    assert!((after - 0.01).abs() < 1.0e-6, "clipped norm is {after}");
}

#[test]
fn constant_parameter_gradient_remains_host_visible() {
    let mut graph = Graph::new();
    let parameter = graph.parameter("value", &[1]);
    graph.set_outputs(vec![parameter]);
    let mut session = meganeura::build(&graph, meganeura::SessionConfig::from_env()).0;
    session.set_parameter("value", &[2.0]);
    session.clear_optimizer();
    session.step();
    session.wait();
    assert_eq!(session.read_loss(), 2.0);

    let mut gradient = [0.0];
    session.read_param_grad("value", &mut gradient);
    assert_eq!(gradient, [1.0]);
}
