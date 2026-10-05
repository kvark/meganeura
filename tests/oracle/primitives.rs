//! The primitive op set (`Op::is_primitive`) and composites built from it.
//!
//! Each newer primitive runs against the reference under every lowering,
//! and its gradient against finite differences and on the GPU. Composites
//! written in primitives (`Graph::decomposed_*`) must compute the same
//! values whether the optimizer folds them into fused kernels or leaves
//! them as primitive dispatches.

use meganeura::reference::{Feeds, Rng, gpu, gradients};
use meganeura::{Graph, NodeId, OptimizeMode};

fn positive(feeds: &mut Feeds, name: &str, n: usize) {
    let mut rng = Rng::new(n as u64);
    let data: Vec<f32> = (0..n).map(|_| rng.uniform(0.25, 2.0)).collect();
    feeds.set(name, &data);
}

/// Every lowering, plus the graph left unoptimized.
fn variants() -> Vec<(&'static str, gpu::Options)> {
    let mut variants = gpu::Options::lowerings();
    let mut unoptimized = gpu::Options::default();
    unoptimized.optimize.mode = OptimizeMode::Off;
    variants.push(("unoptimized", unoptimized));
    variants
}

fn inference_case(what: &str, build: impl Fn(&mut Graph) -> NodeId, fix: impl Fn(&mut Feeds)) {
    let mut g = Graph::new();
    let y = build(&mut g);
    g.set_outputs(vec![y]);
    let mut feeds = Feeds::new();
    fix(&mut feeds);
    feeds.fill_random(&g, what.len() as u64, 1.0);
    for (label, options) in variants() {
        gpu::check_inference(&g, &feeds, &options)
            .unwrap_or_else(|e| panic!("{what} ({label}): {e}"))
            .assert_passed(&format!("{what} ({label})"));
    }
}

fn grad_case(what: &str, build: impl Fn(&mut Graph) -> NodeId, fix: impl Fn(&mut Feeds)) {
    let mut g = Graph::new();
    let y = build(&mut g);
    let loss = gradients::weighted_loss(&mut g, y, what.len() as u64, 0.75);
    g.set_outputs(vec![loss]);
    let mut feeds = Feeds::new();
    fix(&mut feeds);
    feeds.fill_random(&g, 1 + what.len() as u64, 1.0);
    gradients::check(&g, &feeds, &gradients::Options::default())
        .unwrap_or_else(|e| panic!("{what}: {e}"))
        .assert_passed(&format!("{what}: autodiff vs finite differences"));
    for (label, options) in variants() {
        gpu::check_training(&g, &feeds, &options)
            .unwrap_or_else(|e| panic!("{what} ({label}): {e}"))
            .assert_passed(&format!("{what} ({label}): training step vs reference"));
    }
}

#[test]
fn elementwise_primitives() {
    for (what, sqrt) in [("sqrt", true), ("rsqrt", false)] {
        let build = |g: &mut Graph| {
            let x = g.parameter("x", &[5, 7]);
            if sqrt { g.sqrt(x) } else { g.rsqrt(x) }
        };
        let fix = |feeds: &mut Feeds| positive(feeds, "x", 35);
        inference_case(what, build, fix);
        grad_case(what, build, fix);
    }
    let build = |g: &mut Graph| {
        let x = g.parameter("x", &[5, 7]);
        g.add_scalar(x, 0.375)
    };
    inference_case("add_scalar", build, |_| {});
    grad_case("add_scalar", build, |_| {});
}

/// Values a step away from both bounds, so finite differences never
/// straddle one.
#[test]
fn clamp_gradient() {
    grad_case(
        "clamp",
        |g| {
            let x = g.parameter("x", &[4, 6]);
            g.clamp(x, -0.5, 0.5)
        },
        |feeds| {
            let data: Vec<f32> = (0..24).map(|i| -0.95 + 0.08 * i as f32).collect();
            feeds.set("x", &data);
        },
    );
}

/// Narrow rows pack many rows per workgroup; wide rows take one each.
#[test]
fn max_inner() {
    for (rows, cols) in [(9, 5), (3, 300)] {
        let what = format!("max_inner {rows}x{cols}");
        let build = |g: &mut Graph| {
            let x = g.parameter("x", &[rows, cols]);
            g.max_inner(x)
        };
        inference_case(&what, build, |_| {});
        grad_case(&what, build, |_| {});
    }
}

#[test]
fn decomposed_softmax() {
    for (rows, cols) in [(6, 10), (2, 300)] {
        let what = format!("decomposed softmax {rows}x{cols}");
        let build = |g: &mut Graph| {
            let x = g.parameter("x", &[rows, cols]);
            g.decomposed_softmax(x)
        };
        inference_case(&what, build, |_| {});
        grad_case(&what, build, |_| {});
    }
}

#[test]
fn decomposed_rms_norm() {
    for (rows, cols) in [(6, 10), (2, 300)] {
        let what = format!("decomposed rms_norm {rows}x{cols}");
        let build = |g: &mut Graph| {
            let x = g.parameter("x", &[rows, cols]);
            let w = g.parameter("w", &[cols]);
            g.decomposed_rms_norm(x, w, 1e-5)
        };
        inference_case(&what, build, |_| {});
        grad_case(&what, build, |_| {});
    }
}

#[test]
fn decomposed_layer_norm() {
    for (rows, cols) in [(6, 10), (2, 300)] {
        let what = format!("decomposed layer_norm {rows}x{cols}");
        let build = |g: &mut Graph| {
            let x = g.parameter("x", &[rows, cols]);
            let w = g.parameter("w", &[cols]);
            let b = g.parameter("b", &[cols]);
            g.decomposed_layer_norm(x, w, b, 1e-5)
        };
        inference_case(&what, build, |_| {});
        grad_case(&what, build, |_| {});
    }
}

/// Folding back into `RmsNorm` restores the plan-level fusions keyed on
/// it: a decode-shaped norm feeding a GEMV becomes the GEMV's prologue,
/// exactly as when the graph names `rms_norm` itself.
#[test]
fn decomposed_rms_norm_reaches_plan_fusions() {
    let build = |decomposed: bool| {
        let mut g = Graph::new();
        let x = g.input("x", &[1, 64]);
        let w = g.parameter("w", &[64]);
        let proj = g.parameter("proj", &[64, 32]);
        let h = if decomposed {
            g.decomposed_rms_norm(x, w, 1e-5)
        } else {
            g.rms_norm(x, w, 1e-5)
        };
        let y = g.matmul(h, proj);
        g.set_outputs(vec![y]);
        g
    };
    let plan = |g: &Graph, mode: OptimizeMode| {
        let mut config = meganeura::SessionConfig::from_env();
        config.mode = meganeura::Mode::Inference;
        config.optimize.mode = mode;
        config.gpu = Some(gpu::shared_context());
        let (session, _) = meganeura::build(g, config);
        let dispatches = &session.plan().dispatches;
        (
            dispatches.len(),
            dispatches.iter().any(|d| d.gemv_rmsnorm.is_some()),
        )
    };
    let named = plan(&build(false), OptimizeMode::EgglogOutlined);
    assert!(named.1, "the reference graph should fuse the norm");
    assert_eq!(
        plan(&build(true), OptimizeMode::EgglogOutlined),
        named,
        "decomposed and named norms should lower identically"
    );
    let primitive = plan(&build(true), OptimizeMode::Off);
    assert!(!primitive.1);
    assert!(primitive.0 > named.0, "{primitive:?} vs {named:?}");
}
