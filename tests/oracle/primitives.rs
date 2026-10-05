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

/// Known values of erf, so the reference the GPU kernel is compared with is
/// itself right, in both its series and continued-fraction ranges.
#[test]
fn erf_reference_values() {
    let points: [f64; 10] = [0.0, 0.1, 0.5, 1.0, 2.0, 2.9, 3.5, 4.5, -1.5, 7.0];
    let known = [
        0.0,
        0.112_462_916_018_284_9,
        0.520_499_877_813_046_5,
        0.842_700_792_949_714_9,
        0.995_322_265_018_952_7,
        0.999_958_902_121_900_5,
        0.999_999_256_901_627_7,
        0.999_999_999_803_383_4,
        -0.966_105_146_475_310_7,
        1.0,
    ];
    let mut g = Graph::new();
    let x = g.input("x", &[1, points.len()]);
    let y = g.erf(x);
    g.set_outputs(vec![y]);
    let mut feeds = Feeds::new();
    feeds.set("x", &points.map(|v| v as f32));
    let got = meganeura::reference::evaluate_outputs(&g, &feeds).unwrap();
    for ((&p, &want), &got) in points.iter().zip(&known).zip(&got[0].data) {
        // The feed rounds each point to f32 first.
        let slope = std::f64::consts::FRAC_2_SQRT_PI * (-p * p).exp();
        let rounding = (f64::from(p as f32) - p).abs() * slope;
        assert!(
            (got - want).abs() <= 1e-14 + rounding * 1.01,
            "erf({p}) = {got}, want {want}"
        );
    }
}

#[test]
fn erf() {
    let build = |g: &mut Graph| {
        let x = g.parameter("x", &[4, 16]);
        g.erf(x)
    };
    // Spans the kernel's Taylor and erfc ranges and the saturated tail.
    let fix = |feeds: &mut Feeds| {
        let data: Vec<f32> = (0..64).map(|i| -4.1 + 0.13 * i as f32).collect();
        feeds.set("x", &data);
    };
    inference_case("erf", build, fix);
    grad_case("erf", build, fix);
}

/// Small shapes take the 32-wide tile, large ones the 64-wide tile; the
/// gradient of each form exercises the other two.
#[test]
fn batch_matmul() {
    for (batch, m, k, n) in [(3, 5, 7, 4), (2, 130, 70, 90)] {
        for form in ["nn", "at", "bt"] {
            let what = format!("batch_matmul {form} {batch}x{m}x{k}x{n}");
            let build = |g: &mut Graph| match form {
                "nn" => {
                    let a = g.parameter("a", &[batch, m, k]);
                    let b = g.parameter("b", &[batch, k, n]);
                    g.batch_matmul(a, b)
                }
                "at" => {
                    let a = g.parameter("a", &[batch, k, m]);
                    let b = g.parameter("b", &[batch, k, n]);
                    g.batch_matmul_at(a, b)
                }
                _ => {
                    let a = g.parameter("a", &[batch, m, k]);
                    let b = g.parameter("b", &[batch, n, k]);
                    g.batch_matmul_bt(a, b)
                }
            };
            inference_case(&what, build, |_| {});
            if m < 100 {
                grad_case(&what, build, |_| {});
            }
        }
    }
}

#[test]
fn permute() {
    for (shape, perm) in [
        (vec![2, 3, 4, 5], vec![0, 2, 1, 3]),
        (vec![2, 3, 4, 5], vec![0, 2, 3, 1]),
        (vec![2, 3, 4, 5], vec![3, 1, 0, 2]),
        (vec![4, 6, 5], vec![2, 0, 1]),
        (vec![7, 9], vec![1, 0]),
    ] {
        let what = format!("permute {shape:?} by {perm:?}");
        let build = |g: &mut Graph| {
            let x = g.parameter("x", &shape);
            g.permute(x, &perm)
        };
        inference_case(&what, build, |_| {});
        grad_case(&what, build, |_| {});
    }
}

#[test]
fn sin_cos() {
    for (what, sine) in [("sin", true), ("cos", false)] {
        let build = |g: &mut Graph| {
            let x = g.parameter("x", &[4, 9]);
            if sine { g.sin(x) } else { g.cos(x) }
        };
        let fix = |feeds: &mut Feeds| {
            let data: Vec<f32> = (0..36).map(|i| -9.0 + 0.5 * i as f32).collect();
            feeds.set("x", &data);
        };
        inference_case(what, build, fix);
        grad_case(what, build, fix);
    }
}

/// Positions arrive as `U32` and convert exactly.
#[test]
fn to_f32() {
    let mut g = Graph::new();
    let p = g.input_u32("p", &[2, 5]);
    let y = g.to_f32(p);
    let w = g.parameter("w", &[2, 5]);
    let y = g.mul(y, w);
    g.set_outputs(vec![y]);
    let mut feeds = Feeds::new();
    feeds.set_u32(
        "p",
        &[0, 1, 7, 255, 4096, 65_537, 1 << 20, 3, 9, (1 << 24) - 1],
    );
    feeds.fill_random(&g, 5, 1.0);
    for (label, options) in variants() {
        gpu::check_inference(&g, &feeds, &options)
            .unwrap()
            .assert_passed(&format!("to_f32 ({label})"));
    }
}

/// The squeeze-excite gate, now differentiable.
#[test]
fn mul_per_channel_gradient() {
    let build = |g: &mut Graph| {
        let x = g.parameter("x", &[2 * 3 * 8]);
        let gate = g.parameter("gate", &[6]);
        let y = g.mul_per_channel(x, gate, 3, 8);
        g.reshape(y, &[6, 8])
    };
    grad_case("mul_per_channel", build, |_| {});
}

/// Attention with an additive per-head bias, against the reference, at
/// head widths below, at and above one lane per dimension.
#[test]
fn biased_attention() {
    for (rows, keys, heads, kv, dim, causal) in [
        (5, 7, 2, 1, 8, false),
        (6, 6, 4, 2, 64, true),
        (3, 9, 2, 2, 80, false),
        (4, 4, 1, 1, 200, true),
    ] {
        let what = format!("biased attention {rows}x{keys} h{heads}/{kv} d{dim} causal={causal}");
        let mut g = Graph::new();
        let q = g.parameter("q", &[rows, heads * dim]);
        let k = g.parameter("k", &[keys, kv * dim]);
        let v = g.parameter("v", &[keys, kv * dim]);
        let bias = g.parameter("bias", &[heads, rows, keys]);
        let y = g.biased_attention(
            [q, k, v, bias],
            heads as u32,
            kv as u32,
            dim as u32,
            0.7,
            causal,
        );
        g.set_outputs(vec![y]);
        let mut feeds = Feeds::new();
        feeds.fill_random(&g, dim as u64, 1.0);
        for (label, options) in variants() {
            gpu::check_inference(&g, &feeds, &options)
                .unwrap_or_else(|e| panic!("{what} ({label}): {e}"))
                .assert_passed(&format!("{what} ({label})"));
        }
    }
}

#[test]
fn biased_cached_attention() {
    for (rows, max_seq, pos, heads, kv, dim) in [(1, 12, 6, 4, 2, 16), (3, 9, 8, 2, 1, 72)] {
        let what = format!("biased cached attention {rows} rows, pos {pos}, d{dim}");
        let mut g = Graph::new();
        let q = g.parameter("q", &[rows, heads * dim]);
        let k = g.parameter("k", &[max_seq, kv * dim]);
        let v = g.parameter("v", &[max_seq, kv * dim]);
        let p = g.input_u32("pos", &[1]);
        let bias = g.parameter("bias", &[heads, max_seq]);
        let y =
            g.biased_cached_attention([q, k, v, p, bias], heads as u32, kv as u32, dim as u32, 1.0);
        g.set_outputs(vec![y]);
        let mut feeds = Feeds::new();
        feeds.set_u32("pos", &[pos]);
        feeds.fill_random(&g, dim as u64, 1.0);
        for (label, options) in variants() {
            gpu::check_inference(&g, &feeds, &options)
                .unwrap_or_else(|e| panic!("{what} ({label}): {e}"))
                .assert_passed(&format!("{what} ({label})"));
        }
    }
}

/// Index math for gathers: rows picked by a U32 constant, and by indices
/// computed from a runtime position and converted with `to_u32`.
#[test]
fn computed_indices() {
    let mut g = Graph::new();
    let table = g.parameter("table", &[10, 4]);
    let fixed = g.constant_u32(&[3, 0, 9, 3], &[4]);
    let picked = g.embedding(fixed, table);
    // Rows pos + 2 - j for j in 0..3: descending from pos + 2.
    let pos = g.input_u32("pos", &[1]);
    let pos = g.to_f32(pos);
    let pos = g.reshape(pos, &[1, 1]);
    let pos = g.broadcast_inner(pos, 3);
    let pos = g.reshape(pos, &[3]);
    let down = g.constant(vec![2.0, 1.0, 0.0], &[3]);
    let index = g.add(pos, down);
    let index = g.to_u32(index);
    let walked = g.embedding(index, table);
    g.set_outputs(vec![picked, walked]);
    let mut feeds = Feeds::new();
    feeds.set_u32("pos", &[5]);
    feeds.fill_random(&g, 9, 1.0);
    for (label, options) in variants() {
        gpu::check_inference(&g, &feeds, &options)
            .unwrap()
            .assert_passed(&format!("computed indices ({label})"));
    }
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
    // Recomposition is not an optimization: it holds with the optimizer
    // off as well.
    assert_eq!(
        plan(&build(true), OptimizeMode::Off),
        plan(&build(false), OptimizeMode::Off)
    );
}
