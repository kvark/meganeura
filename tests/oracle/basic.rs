//! Elementwise, contraction, reduction, loss, gather and layout ops on the
//! GPU against the reference, over shapes that reach each lowering.

use meganeura::graph::{Op, TensorType};
use meganeura::reference::{Feeds, Rng, gpu};
use meganeura::{Graph, NodeId};

/// Build a graph with `build`, fill its inputs, adjust them with `fix`, and
/// compare every output with the reference.
fn case(what: &str, build: impl FnOnce(&mut Graph) -> Vec<NodeId>, fix: impl FnOnce(&mut Feeds)) {
    let mut g = Graph::new();
    let outputs = build(&mut g);
    g.set_outputs(outputs);
    let mut feeds = Feeds::new();
    fix(&mut feeds);
    feeds.fill_random(&g, what.len() as u64, 1.0);
    let report = gpu::check_inference(&g, &feeds, &gpu::Options::default())
        .unwrap_or_else(|e| panic!("{what}: {e}"));
    report.assert_passed(what);
}

fn positive(feeds: &mut Feeds, name: &str, n: usize, seed: u64) {
    let mut rng = Rng::new(seed);
    let data: Vec<f32> = (0..n).map(|_| rng.uniform(0.05, 3.0)).collect();
    feeds.set(name, &data);
}

#[test]
fn matmul_lowerings() {
    // (m, k, n): tiled, M=1 GEMV (N % 4 == 0), M=1 with ragged N, tile edges.
    for (m, k, n) in [
        (7, 5, 3),
        (1, 64, 128),
        (1, 63, 132),
        (1, 64, 130),
        (64, 64, 64),
        (65, 33, 129),
        (130, 70, 1),
        (3, 300, 5),
    ] {
        case(
            &format!("matmul {m}x{k}x{n}"),
            |g| {
                let a = g.input("a", &[m, k]);
                let b = g.parameter("b", &[k, n]);
                vec![g.matmul(a, b)]
            },
            |_| {},
        );
    }
}

#[test]
fn matmul_by_unit_column_is_a_row_sum() {
    for k in [1, 7, 32, 33] {
        case(
            &format!("matmul ones k={k}"),
            |g| {
                let a = g.input("a", &[9, k]);
                let ones = g.constant(vec![1.0; k], &[k, 1]);
                vec![g.matmul(a, ones)]
            },
            |_| {},
        );
    }
}

#[test]
fn transposed_matmul_lowerings() {
    for (m, k, n) in [
        (7, 5, 3),
        (1, 64, 128),
        (1, 63, 9),
        (65, 33, 129),
        (4, 1, 6),
    ] {
        case(
            &format!("matmul_bt {m}x{k}x{n}"),
            |g| {
                let a = g.input("a", &[m, k]);
                let b = g.parameter("b", &[n, k]);
                vec![g.matmul_bt(a, b)]
            },
            |_| {},
        );
        case(
            &format!("matmul_at {m}x{k}x{n}"),
            |g| {
                let a = g.input("a", &[k, m]);
                let b = g.parameter("b", &[k, n]);
                vec![g.matmul_at(a, b)]
            },
            |_| {},
        );
    }
    case(
        "matmul_bt by unit row",
        |g| {
            let a = g.input("a", &[5, 1]);
            let ones = g.constant(vec![1.0; 6], &[6, 1]);
            vec![g.matmul_bt(a, ones)]
        },
        |_| {},
    );
}

#[test]
fn matmul_epilogues() {
    // Add and bias-add after a product fuse into its epilogue.
    for (m, k, n) in [(1, 64, 128), (1, 64, 6), (9, 17, 33)] {
        case(
            &format!("matmul+add {m}x{k}x{n}"),
            |g| {
                let a = g.input("a", &[m, k]);
                let b = g.parameter("b", &[k, n]);
                let d = g.input("d", &[m, n]);
                let p = g.matmul(a, b);
                vec![g.add(p, d)]
            },
            |_| {},
        );
        case(
            &format!("matmul_bt+add {m}x{k}x{n}"),
            |g| {
                let a = g.input("a", &[m, k]);
                let b = g.parameter("b", &[n, k]);
                let d = g.input("d", &[m, n]);
                let p = g.matmul_bt(a, b);
                vec![g.add(d, p)]
            },
            |_| {},
        );
        case(
            &format!("matmul_at+add {m}x{k}x{n}"),
            |g| {
                let a = g.input("a", &[k, m]);
                let b = g.parameter("b", &[k, n]);
                let d = g.input("d", &[m, n]);
                let p = g.matmul_at(a, b);
                vec![g.add(p, d)]
            },
            |_| {},
        );
        case(
            &format!("matmul+bias+relu {m}x{k}x{n}"),
            |g| {
                let a = g.input("a", &[m, k]);
                let b = g.parameter("b", &[k, n]);
                let bias = g.parameter("bias", &[n]);
                let p = g.matmul(a, b);
                let p = g.bias_add(p, bias);
                vec![g.relu(p)]
            },
            |_| {},
        );
    }
}

#[test]
fn horizontal_matmuls_sharing_an_input() {
    case(
        "three projections of one input",
        |g| {
            let x = g.input("x", &[6, 24]);
            let wq = g.parameter("wq", &[24, 16]);
            let wk = g.parameter("wk", &[24, 8]);
            let wv = g.parameter("wv", &[24, 8]);
            vec![g.matmul(x, wq), g.matmul(x, wk), g.matmul(x, wv)]
        },
        |_| {},
    );
}

#[test]
fn block_matmuls() {
    for (m, groups, k, n) in [(3, 2, 4, 5), (1, 3, 8, 4), (17, 4, 3, 9)] {
        case(
            &format!("block_matmul {m}x{groups}x{k}x{n}"),
            |g| {
                let a = g.input("a", &[m, groups * k]);
                let b = g.parameter("b", &[groups, k, n]);
                let bt = g.parameter("bt", &[groups, n, k]);
                let at_a = g.input("at_a", &[k, groups * m]);
                let at_b = g.input("at_b", &[k, groups * n]);
                vec![
                    g.block_matmul(a, b),
                    g.block_matmul_bt(a, bt),
                    g.block_matmul_at(at_a, at_b, groups),
                ]
            },
            |_| {},
        );
    }
}

#[test]
fn unary_ops() {
    for len in [1usize, 255, 256, 1000] {
        case(
            &format!("unary len={len}"),
            |g| {
                let x = g.input("x", &[len]);
                let p = g.input("p", &[len]);
                vec![
                    g.relu(x),
                    g.sigmoid(x),
                    g.tanh(x),
                    g.neg(x),
                    g.abs(x),
                    g.log(p),
                    g.recip(p),
                    g.exp(x),
                    g.silu(x),
                    g.gelu(x),
                    g.clamp(x, -0.25, 0.5),
                    g.scale(x, -1.75),
                ]
            },
            |f| positive(f, "p", len, 7),
        );
    }
}

#[test]
fn unary_ops_over_a_wide_range() {
    // Tails: saturating sigmoid/tanh, exp overflow edges, softplus tails.
    let values: Vec<f32> = (-60..=60).map(|i| i as f32 * 0.75).collect();
    let n = values.len();
    case(
        "unary tails",
        |g| {
            let x = g.input("x", &[n]);
            vec![
                g.sigmoid(x),
                g.tanh(x),
                g.silu(x),
                g.gelu(x),
                g.softplus(x, 1.0),
                g.softplus(x, 3.5),
                g.softplus(x, 0.2),
            ]
        },
        |f| {
            f.set("x", &values);
        },
    );
}

#[test]
fn softplus_gradient_kernel() {
    case(
        "softplus_grad",
        |g| {
            let go = g.input("go", &[300]);
            let x = g.input("x", &[300]);
            let ty = TensorType::f32(vec![300]);
            vec![g.add_raw_node(Op::SoftplusGrad { beta: 2.0 }, vec![go, x], ty)]
        },
        |f| {
            let values: Vec<f32> = (0..300).map(|i| (i as f32 - 150.0) * 0.2).collect();
            f.set("x", &values);
        },
    );
}

#[test]
fn binary_and_broadcast_ops() {
    for (m, n) in [(1, 1), (5, 7), (33, 64)] {
        case(
            &format!("binary {m}x{n}"),
            |g| {
                let a = g.input("a", &[m, n]);
                let b = g.input("b", &[m, n]);
                let row = g.input("row", &[n]);
                let row2 = g.input("row2", &[1, n]);
                let den = g.input("den", &[m, n]);
                vec![
                    g.add(a, b),
                    g.mul(a, b),
                    g.greater(a, b),
                    g.bias_add(a, row),
                    g.bias_mul(a, row),
                    g.broadcast_add(a, row2),
                    g.div(a, den),
                ]
            },
            |f| positive(f, "den", m * n, 3),
        );
    }
}

#[test]
fn fused_pointwise_chains() {
    case(
        "pointwise chain",
        |g| {
            let x = g.input("x", &[37, 19]);
            let y = g.input("y", &[37, 19]);
            let t = g.tanh(x);
            let e = g.exp(t);
            let m = g.mul(e, y);
            let s = g.sigmoid(m);
            let n = g.neg(s);
            let a = g.add(n, x);
            vec![g.scale(a, 0.5), e]
        },
        |_| {},
    );
}

#[test]
fn full_reductions() {
    // One workgroup, several partials, and the two-stage path.
    for len in [1usize, 1000, 16 * 1024 + 3, 300_000] {
        case(
            &format!("sum/mean len={len}"),
            |g| {
                let x = g.input("x", &[len]);
                vec![g.sum_all(x), g.mean_all(x)]
            },
            |_| {},
        );
    }
}

#[test]
fn row_and_column_reductions() {
    // SumRows splits tall narrow inputs across workgroups.
    for (m, n) in [(1, 5), (5, 7), (600, 3), (5000, 3), (64, 4096), (513, 33)] {
        case(
            &format!("sum_rows {m}x{n}"),
            |g| {
                let x = g.input("x", &[m, n]);
                let ty = TensorType::f32(vec![n]);
                vec![g.sum_rows(x, &ty), g.sum_inner(x)]
            },
            |_| {},
        );
    }
    case(
        "broadcast_inner",
        |g| {
            let x = g.input("x", &[9, 1]);
            vec![g.broadcast_inner(x, 13)]
        },
        |_| {},
    );
}

#[test]
fn row_normalization_and_scans() {
    for (m, n) in [(1, 1), (4, 9), (3, 300), (2, 1025)] {
        case(
            &format!("row ops {m}x{n}"),
            |g| {
                let x = g.input("x", &[m, n]);
                vec![
                    g.softmax(x),
                    g.log_softmax(x),
                    g.exclusive_cumsum(x, false),
                    g.exclusive_cumsum(x, true),
                    g.shift_inner(x, 2),
                    g.shift_inner(x, -1),
                    g.transpose(x),
                ]
            },
            |_| {},
        );
    }
}

#[test]
fn normalize_inner_sum_and_its_gradient() {
    // Rows above and below the floor.
    let (m, n) = (6, 5);
    case(
        "normalize_inner_sum",
        |g| {
            let x = g.input("x", &[m, n]);
            let go = g.input("go", &[m, n]);
            let y = g.normalize_inner_sum(x, 0.5);
            let ty = TensorType::f32(vec![m, n]);
            let dx = g.add_raw_node(
                Op::NormalizeInnerSumGrad {
                    inner: n as u32,
                    floor: 0.5,
                },
                vec![go, x],
                ty,
            );
            vec![y, dx]
        },
        |f| {
            let mut rng = Rng::new(11);
            let data: Vec<f32> = (0..m * n)
                .map(|i| rng.uniform(0.0, if i / n % 2 == 0 { 1.0 } else { 0.05 }))
                .collect();
            f.set("x", &data);
        },
    );
}

#[test]
fn losses_and_their_gradients() {
    for (b, c) in [(1, 3), (5, 300), (3, 1025)] {
        case(
            &format!("losses {b}x{c}"),
            |g| {
                let logits = g.input("logits", &[b, c]);
                let labels = g.input("labels", &[b, c]);
                let p = g.input("p", &[b, c]);
                let t = g.input("t", &[b, c]);
                let ce = g.cross_entropy_loss(logits, labels);
                let bce = g.bce_loss(p, t);
                let ty = TensorType::f32(vec![b, c]);
                let grad = g.add_raw_node(Op::CrossEntropyLogitsGrad, vec![logits, labels], ty);
                vec![ce, bce, grad]
            },
            |f| {
                let mut rng = Rng::new(5);
                let p: Vec<f32> = (0..b * c).map(|_| rng.uniform(0.02, 0.98)).collect();
                let t: Vec<f32> = (0..b * c).map(|_| rng.uniform(0.0, 1.0)).collect();
                f.set("p", &p).set("t", &t);
            },
        );
    }
}

#[test]
fn gated_units_and_their_gradients() {
    for (m, n) in [(1, 4), (7, 33)] {
        case(
            &format!("glu {m}x{n}"),
            |g| {
                let gate = g.input("gate", &[m, n]);
                let up = g.input("up", &[m, n]);
                let go = g.input("go", &[m, n]);
                let cat = g.input("cat", &[m, 2 * n]);
                let ty = TensorType::f32(vec![m, 2 * n]);
                let swiglu_grad = g.add_raw_node(Op::SwiGLUConcatGrad, vec![go, cat], ty.clone());
                let geglu_grad = g.add_raw_node(Op::GeGLUConcatGrad, vec![go, cat], ty);
                vec![
                    g.swiglu(gate, up),
                    g.geglu(gate, up),
                    g.swiglu_concat(cat),
                    g.geglu_concat(cat),
                    swiglu_grad,
                    geglu_grad,
                    g.swiglu_grad_gate(go, gate, up),
                    g.swiglu_grad_up(go, gate),
                    g.silu_grad(go, gate),
                ]
            },
            |_| {},
        );
    }
}

#[test]
fn gathers_and_scatters() {
    let (vocab, dim, seq) = (11, 6, 9);
    let indices: Vec<u32> = vec![3, 0, 10, 3, 3, 7, 1, 10, 0];
    case(
        "embedding and scatter_add",
        |g| {
            let idx = g.input_u32("idx", &[seq]);
            let table = g.parameter("table", &[vocab, dim]);
            let src = g.input("src", &[seq, dim]);
            let half = g.to_f16(table);
            vec![
                g.embedding(idx, table),
                g.embedding_f16(idx, half),
                g.scatter_add(idx, src, vocab),
            ]
        },
        |f| {
            f.set_u32("idx", &indices);
        },
    );
}

#[test]
fn cache_writes_and_prefix_selection() {
    let (rows, dim) = (8, 5);
    case(
        "cache_write",
        |g| {
            let new = g.input("new", &[1, dim]);
            let cache = g.input("cache", &[rows, dim]);
            let pos = g.input_u32("pos", &[1]);
            vec![g.cache_write(new, cache, pos)]
        },
        |f| {
            f.set_u32("pos", &[5]);
        },
    );
    case(
        "cache_write_prefix",
        |g| {
            let new = g.input("new", &[4, dim]);
            let cache = g.input("cache", &[rows, dim]);
            let pos = g.input_u32("pos", &[1]);
            let valid = g.input_u32("valid", &[1]);
            vec![g.cache_write_prefix(new, cache, pos, valid)]
        },
        |f| {
            f.set_u32("pos", &[2]).set_u32("valid", &[3]);
        },
    );
    case(
        "prefix_last",
        |g| {
            let x = g.input("x", &[rows, dim]);
            let valid = g.input_u32("valid", &[1]);
            vec![g.prefix_last(x, valid)]
        },
        |f| {
            f.set_u32("valid", &[6]);
        },
    );
}

#[test]
fn channel_layout_ops() {
    let (batch, c, h, w) = (2u32, 3u32, 5u32, 4u32);
    let s = (h * w) as usize;
    let total = (batch * c) as usize * s;
    case(
        "per-channel, pooling, concat, split, upsample",
        |g| {
            let x = g.input("x", &[total]);
            let gate = g.input("gate", &[(batch * c) as usize]);
            let bias = g.input("bias", &[c as usize]);
            let other = g.input("other", &[(batch * 2) as usize * s]);
            let big = g.input("big", &[total * 4]);
            let pooled_grad = g.input("pooled_grad", &[(batch * c) as usize]);
            let cat = g.concat(x, other, batch, c, 2, s as u32);
            let x_ty = TensorType::f32(vec![total]);
            vec![
                g.mul_per_channel(x, gate, c, s as u32),
                g.add_per_channel(x, bias, c, s as u32),
                g.global_avg_pool(x, batch, c, s as u32),
                g.add_raw_node(
                    Op::GlobalAvgPoolGrad {
                        channels: c,
                        spatial: s as u32,
                    },
                    vec![pooled_grad],
                    x_ty,
                ),
                cat,
                g.split_a(cat, batch, c, 2, s as u32),
                g.split_b(cat, batch, c, 2, s as u32),
                g.upsample_2x(x, batch, c, h, w),
                g.upsample_2x_grad(big, batch, c, h, w),
            ]
        },
        |_| {},
    );
}

#[test]
fn views_and_copies() {
    case(
        "reshape, materialize, stop_gradient",
        |g| {
            let x = g.input("x", &[6, 4]);
            let r = g.reshape(x, &[3, 8]);
            let m = g.materialize(r);
            let s = g.stop_gradient(x);
            let t = g.tanh(r);
            vec![m, g.relu(s), t]
        },
        |_| {},
    );
}
