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
    // 5·4 output tiles of 64 or fewer select the 32-wide tile; 257×257
    // (5·5 = 25 tiles) is the smallest square to select the 64-wide one.
    for (batch, m, k, n) in [(3, 5, 7, 4), (2, 130, 70, 90), (2, 257, 33, 257)] {
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
            // The tile the case is meant to cover is the one selected.
            let mut g = Graph::new();
            let y = build(&mut g);
            g.set_outputs(vec![y]);
            let plan = meganeura::compile_plan(
                &g,
                meganeura::Mode::Inference,
                meganeura::OptimizeConfig::default(),
                &meganeura::CompileOptions::default(),
            );
            let large = m.div_ceil(64) * n.div_ceil(64) >= 16;
            let kernel = &plan.dispatches[0].kernel;
            assert_eq!(
                matches!(kernel, meganeura::compile::Kernel::SmallTile),
                !large,
                "{what}: {kernel:?}"
            );
        }
    }
}

/// A row maximum shares its gradient between tied elements: the maximum
/// of a broadcast scalar is the scalar itself, with derivative exactly 1.
#[test]
fn max_inner_ties() {
    for width in [2, 3, 8] {
        let what = format!("max of a {width}-way broadcast");
        grad_case(
            &what,
            |g| {
                let t = g.parameter("t", &[4, 1]);
                let b = g.broadcast_inner(t, width);
                g.max_inner(b)
            },
            |_| {},
        );
    }
    // An all-equal row: each element receives a quarter of the gradient.
    let mut g = Graph::new();
    let x = g.parameter("x", &[3, 4]);
    let m = g.max_inner(x);
    let loss = g.sum_all(m);
    g.set_outputs(vec![loss]);
    let diff = meganeura::autodiff::differentiate(&g);
    let mut feeds = Feeds::new();
    feeds.set("x", &[0.5; 12]);
    let out = meganeura::reference::evaluate_outputs(&diff, &feeds).unwrap();
    assert_eq!(out[1].data, vec![0.25; 12]);
    gpu::check_training(&g, &feeds, &gpu::Options::default())
        .unwrap()
        .assert_passed("all-equal rows");
}

/// Concatenation and splits differentiate by element counts, whatever
/// the operands' rank.
#[test]
fn concat_of_matrices() {
    grad_case(
        "concat of matrices",
        |g| {
            let a = g.parameter("a", &[4, 3]);
            let b = g.parameter("b", &[4, 2]);
            let ab = g.concat(a, b, 4, 3, 2, 1);
            let ab = g.reshape(ab, &[4, 5]);
            g.split_a(ab, 4, 4, 1, 1)
        },
        |_| {},
    );
}

/// Running sums before (or after) each element of a row, over a width
/// spanning several workgroups' worth of elements.
#[test]
fn exclusive_cumsum() {
    for (width, reverse) in [(9, false), (9, true), (300, false), (300, true)] {
        let what = format!("exclusive cumsum over {width}, reverse {reverse}");
        inference_case(
            &what,
            |g| {
                let x = g.input("x", &[4, width]);
                g.exclusive_cumsum(x, reverse)
            },
            |_| {},
        );
        grad_case(
            &what,
            |g| {
                let x = g.parameter("x", &[4, width]);
                g.exclusive_cumsum(x, reverse)
            },
            |_| {},
        );
    }
}

/// Broadcasts along leading, interior, trailing and several axes at
/// once; the gradient sums over the repeated axes.
#[test]
fn broadcast_to() {
    let cases: [(&[usize], &[usize]); 5] = [
        (&[1, 3], &[4, 3]),
        (&[3, 1], &[3, 5]),
        (&[2, 1, 3], &[2, 4, 3]),
        (&[1, 2, 1, 3], &[2, 2, 3, 3]),
        (&[2, 1, 3, 1], &[2, 2, 3, 2]),
    ];
    for (from, to) in cases {
        let what = format!("broadcast {from:?} to {to:?}");
        grad_case(
            &what,
            |g| {
                let t = g.parameter("t", from);
                let b = g.broadcast_to(t, to);
                let rows = to[0];
                g.reshape(b, &[rows, to[1..].iter().product()])
            },
            |_| {},
        );
    }
}

#[test]
#[should_panic(expected = "permute moves F32 elements")]
fn permute_rejects_other_storage() {
    let mut g = Graph::new();
    let x = g.parameter_f16("x", &[2, 2]);
    g.permute(x, &[1, 0]);
}

/// Strided copies spread their grid over two axes, each within the
/// portable limit of 65535 workgroups.
#[test]
fn permute_grid_stays_within_limits() {
    for elements in [65_535 * 256, 65_536 * 256, 3 * 65_535 * 256 + 7] {
        let mut g = Graph::new();
        let x = g.input("x", &[elements / 4, 4]);
        let y = g.permute(x, &[1, 0]);
        g.set_outputs(vec![y]);
        let plan = meganeura::compile_plan(
            &g,
            meganeura::Mode::Inference,
            meganeura::OptimizeConfig::default(),
            &meganeura::CompileOptions::default(),
        );
        let [gx, gy, gz] = plan.dispatches[0].workgroups;
        assert!(
            gx <= 65_535 && gy <= 65_535 && gz == 1,
            "{elements}: {gx}x{gy}"
        );
        assert!(
            (gx * gy) as usize * 256 >= elements,
            "{elements}: {gx}x{gy}"
        );
    }
}

/// Shaders bound their own reads: a token id past the table reads its
/// last row instead of memory past the buffer.
#[test]
fn gather_past_the_table_stays_inside() {
    let mut g = Graph::new();
    let ids = g.input_u32("ids", &[3]);
    let table = g.parameter("table", &[4, 5]);
    let y = g.embedding(ids, table);
    g.set_outputs(vec![y]);
    let data: Vec<f32> = (0..20).map(|i| i as f32).collect();
    let mut config = meganeura::SessionConfig::from_env();
    config.mode = meganeura::Mode::Inference;
    config.gpu = Some(gpu::shared_context());
    let (mut session, _) = meganeura::build(&g, config);
    session.set_parameter("table", &data);
    session.set_input_u32("ids", &[1, 1_000_000, 3]);
    session.step();
    session.wait();
    let mut got = vec![0.0; 15];
    session.read_output_by_index(0, &mut got);
    let row = |r: usize| data[r * 5..r * 5 + 5].to_vec();
    assert_eq!(got, [row(1), row(3), row(3)].concat());
}

/// On the device, a permutation past the one-axis grid limit.
#[test]
fn permute_past_one_axis_grid() {
    let (a, b, c) = (2, 4096, 2049);
    let n = a * b * c;
    assert!(n > 65_536 * 256);
    let mut g = Graph::new();
    let x = g.parameter("x", &[a, b, c]);
    let y = g.permute(x, &[0, 2, 1]);
    g.set_outputs(vec![y]);
    let data: Vec<f32> = (0..n).map(|i| (i % 8191) as f32).collect();
    let mut config = meganeura::SessionConfig::from_env();
    config.mode = meganeura::Mode::Inference;
    config.gpu = Some(gpu::shared_context());
    let (mut session, _) = meganeura::build(&g, config);
    session.set_parameter("x", &data);
    session.step();
    session.wait();
    let mut got = vec![0.0; n];
    session.read_output_by_index(0, &mut got);
    for i in 0..a {
        for k in 0..c {
            for j in 0..b {
                let out = (i * c + k) * b + j;
                assert_eq!(got[out], data[(i * b + j) * c + k], "element {out}");
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
    // With the optimizer off the graph builds as written: the decomposed
    // norm stays in primitives, and nothing fuses it.
    let off = plan(&build(true), OptimizeMode::Off);
    assert!(!off.1, "the optimizer is off");
    assert!(off.0 > named.0, "{off:?} should run the primitives");
}
