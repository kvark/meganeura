//! Bugs the differential tests found, kept as named cases.

use meganeura::Graph;
use meganeura::reference::{Feeds, gpu, gradients};

/// A parameter added straight into an activation receives the upstream
/// gradient unchanged, so its gradient buffer is the output of a pointwise
/// dispatch that also feeds the activation's backward. Dispatch fusion used
/// to fold that producer into its consumer and never write the gradient,
/// because the fusion passes did not treat gradient buffers as observable:
/// the parameter silently received a zero gradient.
#[test]
fn parameter_gradient_survives_pointwise_fusion() {
    for poison in [false, true] {
        let mut g = Graph::new();
        let p1 = g.parameter("p1", &[17, 8]);
        let s = g.silu(p1);
        let sp = g.softplus(s, 1.0);
        let p2 = g.parameter("p2", &[17, 8]);
        let a = g.add(sp, p2);
        let loss = gradients::weighted_loss(&mut g, a, 3, 0.5);
        g.set_outputs(vec![loss]);
        let mut feeds = Feeds::new();
        feeds.fill_random(&g, 1, 1.0);
        let options = gpu::Options {
            poison,
            ..Default::default()
        };
        gpu::check_training(&g, &feeds, &options)
            .unwrap()
            .assert_passed(&format!("poison={poison}"));
    }
}

/// Generated reduction kernels reuse a unary shader entry (`Relu`) as a
/// layout sentinel. The matmul epilogue pass read that entry as the op, so a
/// `SumInner` of a product became `relu(product)`, written into the `[M, 1]`
/// row-sum buffer. Gradients do not read the forward value, so only the
/// loss was wrong.
#[test]
fn reduction_after_matmul_is_not_fused_as_a_unary_epilogue() {
    let mut g = Graph::new();
    let x = g.parameter("x", &[17, 4]);
    let w = g.parameter("w", &[4, 3]);
    let m = g.matmul(x, w);
    let r = g.sum_inner(m);
    let b = g.broadcast_inner(r, 3);
    let loss = gradients::weighted_loss(&mut g, b, 23, 0.5);
    g.set_outputs(vec![loss]);
    let mut feeds = Feeds::new();
    feeds.fill_random(&g, 23, 1.0);
    gpu::check_training(&g, &feeds, &gpu::Options::default())
        .unwrap()
        .assert_passed("training");
    let mut g = Graph::new();
    let x = g.input("x", &[17, 4]);
    let w = g.parameter("w", &[4, 3]);
    let m = g.matmul(x, w);
    let r = g.sum_inner(m);
    g.set_outputs(vec![r]);
    gpu::check_inference(&g, &feeds, &gpu::Options::default())
        .unwrap()
        .assert_passed("inference");
}

/// Tanh-form GELU computed `1 + tanh(u)`, which rounds to exactly zero
/// once `tanh(u)` reaches −1 in f32 (x below about −5): the value and the
/// derivative flushed to zero there. The kernels now use the identical
/// `x · sigmoid(2u)`. Inputs lie only in the tail, so no larger value hides
/// the error behind the tolerance floor.
#[test]
fn gelu_negative_tail_is_not_flushed_to_zero() {
    let values: Vec<f32> = (0..24).map(|i| -9.0 + i as f32 * 0.25).collect();
    let mut g = Graph::new();
    let x = g.input("x", &[4, 6]);
    let gate = g.input("gate", &[4, 6]);
    let cat = g.input("cat", &[4, 12]);
    let a = g.gelu(x);
    let b = g.geglu(x, gate);
    let c = g.geglu_concat(cat);
    g.set_outputs(vec![a, b, c]);
    let mut cat_values = values.clone();
    cat_values.extend(vec![1.0f32; 24]);
    let mut feeds = Feeds::new();
    feeds.set("x", &values).set("gate", &[1.0; 24]);
    feeds.set("cat", &cat_values);
    gpu::check_inference(&g, &feeds, &gpu::Options::default())
        .unwrap()
        .assert_passed("forward");

    let mut g = Graph::new();
    let x = g.parameter("x", &[4, 6]);
    let y = g.gelu(x);
    let loss = gradients::weighted_loss(&mut g, y, 5, 1.0);
    g.set_outputs(vec![loss]);
    let mut feeds = Feeds::new();
    feeds.set("x", &values);
    gradients::check(&g, &feeds, &gradients::Options::default())
        .unwrap()
        .assert_passed("autodiff");
    gpu::check_training(&g, &feeds, &gpu::Options::default())
        .unwrap()
        .assert_passed("training");
}
