//! Every differentiation rule for the basic ops, checked two ways:
//! `differentiate()` against `f64` finite differences on the CPU, then the
//! compiled training step's gradients against the reference on the GPU.
//!
//! Each op sits mid-graph under `gradients::weighted_loss`, so its backward
//! receives a non-uniform upstream gradient scaled by a coefficient.

use meganeura::graph::{Op, TensorType};
use meganeura::reference::{Feeds, Rng, gpu, gradients};
use meganeura::{Graph, NodeId};

fn grad_case(what: &str, build: impl FnOnce(&mut Graph) -> NodeId, fix: impl FnOnce(&mut Feeds)) {
    let mut g = Graph::new();
    let y = build(&mut g);
    let loss = gradients::weighted_loss(&mut g, y, what.len() as u64, 0.75);
    g.set_outputs(vec![loss]);
    let mut feeds = Feeds::new();
    fix(&mut feeds);
    feeds.fill_random(&g, 1 + what.len() as u64, 1.0);
    let cpu = gradients::check(&g, &feeds, &gradients::Options::default())
        .unwrap_or_else(|e| panic!("{what}: {e}"));
    cpu.assert_passed(&format!("{what}: autodiff vs finite differences"));
    let device = gpu::check_training(&g, &feeds, &gpu::Options::default())
        .unwrap_or_else(|e| panic!("{what}: {e}"));
    device.assert_passed(&format!("{what}: training step vs reference"));
}

fn positive(feeds: &mut Feeds, name: &str, n: usize, lo: f32) {
    let mut rng = Rng::new(n as u64);
    let data: Vec<f32> = (0..n).map(|_| rng.uniform(lo, 2.0)).collect();
    feeds.set(name, &data);
}

#[test]
fn contractions() {
    grad_case(
        "matmul",
        |g| {
            let a = g.parameter("a", &[3, 4]);
            let b = g.parameter("b", &[4, 5]);
            g.matmul(a, b)
        },
        |_| {},
    );
    grad_case(
        "matmul_at",
        |g| {
            let a = g.parameter("a", &[4, 3]);
            let b = g.parameter("b", &[4, 5]);
            g.matmul_at(a, b)
        },
        |_| {},
    );
    grad_case(
        "matmul_bt",
        |g| {
            let a = g.parameter("a", &[3, 4]);
            let b = g.parameter("b", &[5, 4]);
            g.matmul_bt(a, b)
        },
        |_| {},
    );
    grad_case(
        "gemv",
        |g| {
            let a = g.parameter("a", &[1, 8]);
            let b = g.parameter("b", &[8, 4]);
            g.matmul(a, b)
        },
        |_| {},
    );
    for (op, ty) in [
        (Op::FusedMatMulAdd, [3, 4, 4, 5]),
        (Op::FusedMatMulATAdd, [4, 3, 4, 5]),
        (Op::FusedMatMulBTAdd, [3, 4, 5, 4]),
    ] {
        let name = format!("{op:?}");
        grad_case(
            &name,
            |g| {
                let a = g.parameter("a", &ty[..2]);
                let b = g.parameter("b", &ty[2..]);
                let d = g.parameter("d", &[3, 5]);
                g.add_raw_node(op, vec![a, b, d], TensorType::f32(vec![3, 5]))
            },
            |_| {},
        );
    }
    grad_case(
        "block_matmul",
        |g| {
            let a = g.parameter("a", &[3, 2 * 4]);
            let b = g.parameter("b", &[2, 4, 3]);
            g.block_matmul(a, b)
        },
        |_| {},
    );
    grad_case(
        "block_matmul_bt",
        |g| {
            let a = g.parameter("a", &[3, 2 * 4]);
            let b = g.parameter("b", &[2, 3, 4]);
            g.block_matmul_bt(a, b)
        },
        |_| {},
    );
    grad_case(
        "block_matmul_at",
        |g| {
            let a = g.parameter("a", &[4, 2 * 3]);
            let b = g.parameter("b", &[4, 2 * 5]);
            g.block_matmul_at(a, b, 2)
        },
        |_| {},
    );
}

#[test]
fn elementwise() {
    type Unary = fn(&mut Graph, NodeId) -> NodeId;
    let unary: [(&str, Unary, f32); 13] = [
        ("relu", Graph::relu, -1.0),
        ("sigmoid", Graph::sigmoid, -1.0),
        ("tanh", Graph::tanh, -1.0),
        ("neg", Graph::neg, -1.0),
        ("abs", Graph::abs, -1.0),
        ("log", Graph::log, 0.1),
        ("recip", Graph::recip, 0.3),
        ("exp", Graph::exp, -1.0),
        ("silu", Graph::silu, -1.0),
        ("gelu", Graph::gelu, -1.0),
        ("softplus", |g, x| g.softplus(x, 1.5), -1.0),
        ("scale", |g, x| g.scale(x, -2.5), -1.0),
        ("materialize", Graph::materialize, -1.0),
    ];
    for (name, op, lo) in unary {
        grad_case(
            name,
            |g| {
                let x = g.parameter("x", &[4, 6]);
                op(g, x)
            },
            |f| {
                if lo > 0.0 {
                    positive(f, "x", 24, lo);
                }
            },
        );
    }
    grad_case(
        "add, mul, bias_add, bias_mul",
        |g| {
            let a = g.parameter("a", &[4, 6]);
            let b = g.parameter("b", &[4, 6]);
            let bias = g.parameter("bias", &[6]);
            let scale = g.parameter("scale", &[6]);
            let s = g.add(a, b);
            let p = g.mul(s, a);
            let q = g.bias_add(p, bias);
            g.bias_mul(q, scale)
        },
        |_| {},
    );
    grad_case(
        "gated units",
        |g| {
            let gate = g.parameter("gate", &[3, 5]);
            let up = g.parameter("up", &[3, 5]);
            let cat = g.parameter("cat", &[3, 10]);
            let s = g.swiglu(gate, up);
            let e = g.geglu(gate, up);
            let sc = g.swiglu_concat(cat);
            let ec = g.geglu_concat(cat);
            let a = g.add(s, e);
            let b = g.add(sc, ec);
            g.add(a, b)
        },
        |_| {},
    );
}

#[test]
fn reductions_and_rows() {
    grad_case(
        "sum_all, mean_all",
        |g| {
            let x = g.parameter("x", &[3, 4]);
            let y = g.parameter("y", &[5]);
            let s = g.sum_all(x);
            let m = g.mean_all(y);
            let t = g.mul(s, m);
            g.mul(t, s)
        },
        |_| {},
    );
    grad_case(
        "sum_rows, sum_inner, broadcast_inner",
        |g| {
            let x = g.parameter("x", &[5, 3]);
            let ty = TensorType::f32(vec![3]);
            let rows = g.sum_rows(x, &ty);
            let inner = g.sum_inner(x);
            let back = g.broadcast_inner(inner, 3);
            let r = g.reshape(rows, &[1, 3]);
            g.broadcast_add(back, r)
        },
        |_| {},
    );
    grad_case(
        "softmax, log_softmax",
        |g| {
            let x = g.parameter("x", &[3, 5]);
            let s = g.softmax(x);
            let l = g.log_softmax(x);
            g.add(s, l)
        },
        |_| {},
    );
    grad_case(
        "normalize_inner_sum",
        |g| {
            let x = g.parameter("x", &[4, 3]);
            g.normalize_inner_sum(x, 0.5)
        },
        |f| {
            let mut rng = Rng::new(3);
            // Two rows above the floor, two below.
            let data: Vec<f32> = (0..12)
                .map(|i| rng.uniform(0.0, if i / 3 % 2 == 0 { 1.0 } else { 0.1 }))
                .collect();
            f.set("x", &data);
        },
    );
    grad_case(
        "exclusive_cumsum, shift_inner, transpose, reshape",
        |g| {
            let x = g.parameter("x", &[3, 5]);
            let f = g.exclusive_cumsum(x, false);
            let r = g.exclusive_cumsum(f, true);
            let s = g.shift_inner(r, 2);
            let s = g.shift_inner(s, -1);
            let t = g.transpose(s);
            g.reshape(t, &[15])
        },
        |_| {},
    );
}

/// `StopGradient` is exempt from finite differences by definition: its
/// derivative is declared zero while its value depends on the input. Check
/// the declared behaviour instead: the detached branch contributes nothing.
#[test]
fn stop_gradient_detaches_one_branch() {
    let mut g = Graph::new();
    let x = g.parameter("x", &[2, 3]);
    let d = g.stop_gradient(x);
    let t = g.tanh(x);
    let y = g.mul(t, d);
    let loss = g.sum_all(y);
    g.set_outputs(vec![loss]);
    let mut feeds = Feeds::new();
    feeds.fill_random(&g, 4, 1.0);
    let backward = meganeura::autodiff::differentiate(&g);
    let values = meganeura::reference::evaluate(&backward, &feeds).unwrap();
    let grad = &values[backward.outputs()[1] as usize];
    let x = feeds.get("x").unwrap();
    for (i, &v) in x.iter().enumerate() {
        // d/dx [tanh(x) · c] with c = x held constant.
        let want = (1.0 - v.tanh().powi(2)) * v;
        assert!((grad.data[i] - want).abs() < 1e-12, "element {i}");
    }
}

#[test]
fn losses() {
    grad_case(
        "cross_entropy with unnormalized labels",
        |g| {
            let logits = g.parameter("logits", &[4, 5]);
            let labels = g.input("labels", &[4, 5]);
            let ce = g.cross_entropy_loss(logits, labels);
            g.mul(ce, ce)
        },
        |_| {},
    );
    grad_case(
        "bce",
        |g| {
            let z = g.parameter("z", &[3, 4]);
            let t = g.input("t", &[3, 4]);
            let p = g.sigmoid(z);
            let l = g.bce_loss(p, t);
            g.mul(l, l)
        },
        |f| {
            let mut rng = Rng::new(9);
            let t: Vec<f32> = (0..12).map(|_| rng.uniform(0.0, 1.0)).collect();
            f.set("t", &t);
        },
    );
    grad_case(
        "mse and l1",
        |g| {
            let p = g.parameter("p", &[3, 4]);
            let t = g.input("t", &[3, 4]);
            let a = g.mse_loss(p, t);
            let b = g.l1_loss(p, t);
            g.add(a, b)
        },
        |_| {},
    );
}

#[test]
fn gathers_and_layouts() {
    grad_case(
        "embedding with repeated indices",
        |g| {
            let idx = g.input_u32("idx", &[6]);
            let table = g.parameter("table", &[5, 3]);
            g.embedding(idx, table)
        },
        |f| {
            f.set_u32("idx", &[1, 4, 1, 0, 1, 4]);
        },
    );
    let (batch, c, h, w) = (2u32, 3u32, 2u32, 3u32);
    let s = h * w;
    let total = (batch * c * s) as usize;
    grad_case(
        "add_per_channel, global_avg_pool, concat, split, upsample",
        |g| {
            let x = g.parameter("x", &[total]);
            let bias = g.parameter("bias", &[c as usize]);
            let other = g.parameter("other", &[(batch * 2 * s) as usize]);
            let biased = g.add_per_channel(x, bias, c, s);
            let cat = g.concat(biased, other, batch, c, 2, s);
            let a = g.split_a(cat, batch, c, 2, s);
            let b = g.split_b(cat, batch, c, 2, s);
            let up = g.upsample_2x(a, batch, c, h, w);
            let pooled = g.global_avg_pool(up, batch, c, 4 * s);
            let pooled_b = g.global_avg_pool(b, batch, 2, s);
            let pa = g.sum_all(pooled);
            let pb = g.sum_all(pooled_b);
            let t = g.mul(pa, pb);
            let up_sum = g.sum_all(up);
            g.add(t, up_sum)
        },
        |_| {},
    );
}
