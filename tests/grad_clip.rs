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

/// One SGD step at lr 0.1 after `configure` sets up clipping.
fn run_one_step(configure: impl FnOnce(&mut meganeura::Session)) -> Vec<f32> {
    let g = build_linreg(4, 3, 2);
    let mut s = meganeura::build(&g, crate::support::gpu::config()).0;

    // Same init for every run so we can compare.
    let w_init: Vec<f32> = (0..6).map(|i| (i as f32 * 0.1).sin()).collect();
    s.set_parameter("w.weight", &w_init);
    s.set_parameter("w.bias", &[0.0; 2]);

    // Inputs designed to give a large, easy-to-reason-about gradient.
    let x: Vec<f32> = (0..12).map(|i| (i as f32 * 0.3).cos()).collect();
    let t: Vec<f32> = vec![10.0; 8]; // very far from y(x) under any reasonable init
    s.set_input("x", &x);
    s.set_input("target", &t);

    configure(&mut s);
    s.set_learning_rate(0.1);
    s.step();
    s.wait();

    let mut params = vec![0.0; 6];
    s.read_param("w.weight", &mut params);
    params
}

/// Against one unclipped step: a tight clip shrinks the update to about
/// `max_norm * lr`, a zero norm disables clipping, and `grad_clip_every`
/// skips the clip on `every - 1` of every `every` steps, so the first step
/// of `every = 10` is unclipped even under a very tight norm.
#[test]
fn grad_clip_bounds_disables_and_skips_the_update() {
    let init: Vec<f32> = (0..6).map(|i| (i as f32 * 0.1).sin()).collect();
    let max_diff = |a: &[f32], b: &[f32]| -> f32 {
        a.iter()
            .zip(b)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
    };
    let l1 = |after: &[f32]| -> f32 { after.iter().zip(&init).map(|(a, b)| (a - b).abs()).sum() };

    let unclipped = run_one_step(|_| {});
    let unclipped_change = l1(&unclipped);
    assert!(
        unclipped_change > 0.001,
        "unclipped step should produce nontrivial param movement; got {unclipped_change}"
    );

    let clipped_change = l1(&run_one_step(|s| s.set_grad_clip_norm(0.01)));
    assert!(
        clipped_change < unclipped_change,
        "clipped step should move less than unclipped; got clipped={clipped_change} unclipped={unclipped_change}"
    );
    // With max_norm=0.01 and lr=0.1 the expected delta is ~0.001; allow
    // generous slack for direction-dependent norm contributions.
    assert!(
        clipped_change < 0.01,
        "clipped change {clipped_change} should be near max_norm*lr=0.001, well under 0.01"
    );

    let zero = run_one_step(|s| s.set_grad_clip_norm(0.0));
    assert!(
        max_diff(&zero, &unclipped) < 1e-6,
        "grad_clip=0.0 must be a no-op"
    );

    let skipped = run_one_step(|s| {
        s.set_grad_clip_norm(0.001);
        s.set_grad_clip_every(10);
    });
    let diff = max_diff(&skipped, &unclipped);
    assert!(
        diff < 1e-5,
        "step 1 with every=10 should match unclipped exactly; max diff {diff}"
    );
}

#[test]
fn adaptive_grad_clip_uses_the_parameter_relative_bound() {
    let mut graph = Graph::new();
    let parameter = graph.parameter("parameter", &[2]);
    let loss = graph.mean_all(parameter);
    graph.set_outputs(vec![loss]);

    let mut session = meganeura::build_session(&graph);
    session.set_parameter("parameter", &[0.0, 0.0]);
    session.set_adaptive_grad_clip(0.1, 1.0);
    session.set_learning_rate(1.0);
    session.step();
    session.wait();

    let mut updated = [0.0; 2];
    session.read_param("parameter", &mut updated);
    let update_norm = updated
        .iter()
        .map(|value| value * value)
        .sum::<f32>()
        .sqrt();
    assert!(
        (update_norm - 0.1).abs() < 1.0e-5,
        "AGC update norm was {update_norm}, expected the 0.1 relative bound"
    );
}

/// Clipping scales a gradient where it is stored, so that storage has to be
/// the update's own. Two gradients that no backward dispatch writes for one
/// parameter alone show it. The gradient of `p` in `p + x` is the constant
/// that seeds the backward pass, and a clipped step must leave it at one for
/// the unclipped step that follows. Two parameters added to the same sum
/// share one gradient, which must be scaled once for both.
#[test]
fn clipping_scales_only_the_gradients_of_its_step() {
    let mut graph = Graph::new();
    let p = graph.parameter("p", &[1]);
    let x = graph.input("x", &[1]);
    let loss = graph.add(p, x);
    graph.set_outputs(vec![loss]);
    let mut session = meganeura::build(&graph, crate::support::gpu::config()).0;
    session.set_parameter("p", &[10.0]);
    session.set_input("x", &[0.0]);
    session.set_learning_rate(1.0);
    let mut p = [0.0];
    for (max_norm, expected) in [(0.25, 9.75), (0.0, 8.75)] {
        session.set_grad_clip_norm(max_norm);
        session.step();
        session.wait();
        session.read_param("p", &mut p);
        assert!(
            (p[0] - expected).abs() < 1e-5,
            "p is {p:?} after a step clipped at {max_norm}"
        );
    }

    let mut graph = Graph::new();
    let x = graph.input("x", &[2]);
    let first = graph.parameter("first", &[2]);
    let second = graph.parameter("second", &[2]);
    let sum = graph.add(x, first);
    let sum = graph.add(sum, second);
    let square = graph.mul(sum, sum);
    let loss = graph.mean_all(square);
    graph.set_outputs(vec![loss]);
    let mut session = meganeura::build(&graph, crate::support::gpu::config()).0;
    session.set_parameter("first", &[0.0; 2]);
    session.set_parameter("second", &[0.0; 2]);
    session.set_input("x", &[3.0, 4.0]);
    session.set_learning_rate(1.0);
    // Each gradient is [3, 4], so together they have norm sqrt(50).
    session.set_grad_clip_norm(0.5 * 50.0f32.sqrt());
    session.step();
    session.wait();
    for name in ["first", "second"] {
        let mut updated = [0.0; 2];
        session.read_param(name, &mut updated);
        assert!(
            (updated[0] + 1.5).abs() < 1e-5 && (updated[1] + 2.0).abs() < 1e-5,
            "{name} is {updated:?} after half of its gradient"
        );
    }
}
