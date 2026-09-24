//! Named cases: bugs the differential tests found, and scenarios moved here
//! from focused legacy tests that the family sweeps do not reach on their
//! own (training outputs, mixed head widths, gather-fused reductions,
//! softplus tails, saturated statistics).

use meganeura::reference::{Feeds, gpu, gradients};
use meganeura::{Graph, NodeId};

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

/// A training graph that also returns intermediates: a product that feeds
/// only the loss (the MatMul+Add fusion once dropped it from the outputs)
/// and a mid-network activation the backward pass reads (it was once
/// overwritten at batch sizes above one).
#[test]
fn training_outputs_survive_fusion_and_backward() {
    for batch in [1usize, 2, 4] {
        let mut g = Graph::new();
        let x = g.input("x", &[batch, 4]);
        let target = g.input("target", &[batch, 3]);
        let w1 = g.parameter("w1", &[4, 4]);
        let b1 = g.parameter("b1", &[4]);
        let norm = g.parameter("norm", &[4]);
        let w_out = g.parameter("w_out", &[4, 3]);
        let w2 = g.parameter("w2", &[3, 3]);
        let h = g.matmul(x, w1);
        let h = g.bias_add(h, b1);
        let h = g.relu(h);
        let h = g.rms_norm(h, norm, 1e-5);
        let y = g.matmul(h, w_out);
        let y2 = g.matmul(y, w2);
        let loss = g.mse_loss(y2, target);
        // A packed projection that is observed but does not reach the loss
        // must also be checked, without inventing a device gradient for it.
        let gate = g.parameter("gate", &[4, 3]);
        let up = g.parameter("up", &[4, 3]);
        let a = g.matmul(x, gate);
        let b = g.matmul(x, up);
        let observed = g.swiglu(a, b);
        g.set_outputs(vec![loss, y, y2, observed]);
        let mut feeds = Feeds::new();
        feeds.fill_random(&g, 40 + batch as u64, 1.0);
        gpu::check_training(&g, &feeds, &gpu::Options::default())
            .unwrap()
            .assert_passed(&format!("batch {batch}"));
    }
}

/// Attention blocks of different head widths in one session each get their
/// own specialized pipeline.
#[test]
fn mixed_attention_widths_in_one_session() {
    let seq = 4;
    let mut g = Graph::new();
    let mut outputs = Vec::new();
    for (i, head_dim) in [4usize, 8, 64].into_iter().enumerate() {
        let q = g.input(&format!("q{i}"), &[seq, head_dim]);
        let k = g.input(&format!("k{i}"), &[seq, head_dim]);
        let v = g.input(&format!("v{i}"), &[seq, head_dim]);
        outputs.push(g.full_attention(q, k, v, 1, 1, head_dim as u32));
    }
    g.set_outputs(outputs);
    let mut feeds = Feeds::new();
    feeds.fill_random(&g, 50, 1.0);
    gpu::check_inference(&g, &feeds, &gpu::Options::default())
        .unwrap()
        .assert_passed("mixed widths");
}

/// The f16 embedding table forward (rounded weights) and its straight-through
/// gradient into the f32 master parameter, with repeated indices.
#[test]
fn f16_embedding_trains_the_f32_master() {
    let (vocab, seq, hidden) = (5, 12, 8);
    let mut g = Graph::new();
    let idx = g.input_u32("idx", &[seq]);
    let w = g.parameter("w", &[vocab, hidden]);
    let x = g.input("x", &[seq, hidden]);
    let half = g.to_f16(w);
    let gathered = g.embedding_f16(idx, half);
    let product = g.mul(gathered, x);
    let loss = g.sum_all(product);
    g.set_outputs(vec![loss, gathered]);
    let mut feeds = Feeds::new();
    feeds.set_u32("idx", &[0, 3, 3, 1, 4, 0, 2, 3, 1, 1, 4, 0]);
    feeds.fill_random(&g, 60, 1.0);
    gpu::check_training(&g, &feeds, &gpu::Options::default())
        .unwrap()
        .assert_passed("f16 embedding");
}

/// The atomic scatter path (large vocabularies) with repeated indices and
/// signed zeros.
#[test]
fn large_scatter_add_with_repeated_indices() {
    let (vocab, seq, dim) = (4097, 256, 3);
    let mut g = Graph::new();
    let idx = g.input_u32("idx", &[seq]);
    let src = g.input("src", &[seq, dim]);
    let out = g.scatter_add(idx, src, vocab);
    g.set_outputs(vec![out]);
    let indices: Vec<u32> = (0..seq).map(|i| ((i * 17) % 31) as u32 + 4000).collect();
    let values: Vec<f32> = (0..seq * dim)
        .map(|i| match i % 19 {
            0 => (i as f32 - 200.0) * 0.001,
            1 => -0.0,
            _ => (i as f32).sin(),
        })
        .collect();
    let mut feeds = Feeds::new();
    feeds.set_u32("idx", &indices).set("src", &values);
    gpu::check_inference(&g, &feeds, &gpu::Options::default())
        .unwrap()
        .assert_passed("large scatter");
}

/// Softplus keeps relative accuracy far into its negative tail, forward and
/// backward, for several `beta`, fused and unfused. The tolerance has no
/// floor, so a tail value that is merely small next to others still counts.
#[test]
fn softplus_tail_keeps_relative_accuracy() {
    let scaled = [
        -80.0f32, -40.0, -30.0, -20.0, -18.5, -18.0, -17.749283, -17.187275, -17.0, -16.617382,
        -16.0, -10.0, -6.0, -1.0, 0.0, 1.0, 6.0, 16.0, 30.0, 80.0,
    ];
    let strict = gpu::Options {
        tolerance: meganeura::reference::Tolerance {
            rtol: 2e-4,
            floor: 0.0,
        },
        ..Default::default()
    };
    for beta in [0.25f32, 1.0, 10.0] {
        for fused in [false, true] {
            let input: Vec<f32> = scaled.iter().map(|x| x / beta).collect();
            let mut g = Graph::new();
            let x = g.parameter("x", &[input.len()]);
            let y = g.softplus(x, beta);
            let loss = gradients::weighted_loss(&mut g, y, 70, 1.0);
            g.set_outputs(vec![loss, y]);
            let mut feeds = Feeds::new();
            feeds.set("x", &input);
            let mut options = strict.clone();
            options.compile.fuse_dispatches = fused;
            gpu::check_training(&g, &feeds, &options)
                .unwrap()
                .assert_passed(&format!("beta {beta} fused {fused}"));
        }
    }

    // A normalized mixture of saturated selectors: the old lowering moved one
    // weight by 0.295 while preserving the total.
    let logits = [
        -28.584324f32,
        -55.08445,
        -57.773396,
        -17.187275,
        -43.64951,
        42.930416,
        -53.557167,
        -71.394745,
        -42.14466,
        -105.05025,
        -16.617382,
        0.0,
    ];
    let priors = [
        0.02f32,
        0.06,
        0.12,
        0.30,
        0.5,
        0.0,
        0.283_020_4,
        0.28315285,
        0.18885687,
        0.15745437,
        0.08751558,
        0.0,
    ];
    let mut g = Graph::new();
    let x = g.input("x", &[2, 6]);
    let p = g.input("prior", &[2, 6]);
    let y = g.softplus(x, 1.0);
    let floor = g.constant(vec![1e-8; 12], &[2, 6]);
    let negative_floor = g.neg(floor);
    let offset = g.add(y, negative_floor);
    let clamped = g.relu(offset);
    let positive = g.add(clamped, floor);
    let raw = g.mul(positive, p);
    let total = g.sum_inner(raw);
    let total = g.broadcast_inner(total, 6);
    let normalized = g.div(raw, total);
    g.set_outputs(vec![normalized]);
    let mut feeds = Feeds::new();
    feeds.set("x", &logits).set("prior", &priors);
    gpu::check_inference(&g, &feeds, &gpu::Options::default())
        .unwrap()
        .assert_passed("normalized mixture");
}

/// Reductions whose operands are embedding gathers fold the gathers into the
/// reduction as indexed streams; their training step fuses the row-scaled
/// scatter back into the tables. One gather times a basis, two gathers, and
/// one gather shared (with an offset) by two reductions.
#[test]
fn gather_fused_reductions() {
    let (vocab, m, n) = (5usize, 6usize, 64usize);
    let indices_a: Vec<u32> = (0..m).map(|i| (i * 3 % vocab) as u32).collect();
    let indices_b: Vec<u32> = (0..m).map(|i| ((i * 2 + 1) % vocab) as u32).collect();
    type Build = fn(&mut Graph, NodeId, NodeId, NodeId, NodeId) -> NodeId;
    let cases: [(&str, Build); 3] = [
        ("gather x basis", |g, ia, _, ta, _| {
            let basis = g.input("basis", &[6, 64]);
            let gathered = g.embedding(ia, ta);
            let product = g.mul(gathered, basis);
            g.sum_inner(product)
        }),
        ("two gathers", |g, ia, ib, ta, tb| {
            let a = g.embedding(ia, ta);
            let b = g.embedding(ib, tb);
            let product = g.mul(a, b);
            g.sum_inner(product)
        }),
        ("shared gather with offset", |g, ia, _, ta, _| {
            let offset = g.input("offset", &[6, 64]);
            let fa = g.input("factors_a", &[6, 64]);
            let fb = g.input("factors_b", &[6, 64]);
            let gathered = g.embedding(ia, ta);
            let relative = g.add(gathered, offset);
            let terms_a = g.mul(relative, fa);
            let reduced_a = g.sum_inner(terms_a);
            let terms_b = g.mul(relative, fb);
            let reduced_b = g.sum_inner(terms_b);
            g.add(reduced_a, reduced_b)
        }),
    ];
    for (label, build) in cases {
        for training in [false, true] {
            let mut g = Graph::new();
            let ia = g.input_u32("ia", &[m]);
            let ib = g.input_u32("ib", &[m]);
            let (ta, tb) = if training {
                (
                    g.parameter("ta", &[vocab, n]),
                    g.parameter("tb", &[vocab, n]),
                )
            } else {
                (g.input("ta", &[vocab, n]), g.input("tb", &[vocab, n]))
            };
            let y = build(&mut g, ia, ib, ta, tb);
            let mut feeds = Feeds::new();
            feeds.set_u32("ia", &indices_a).set_u32("ib", &indices_b);
            let report = if training {
                let loss = gradients::weighted_loss(&mut g, y, 80, 0.8);
                g.set_outputs(vec![loss, y]);
                feeds.fill_random(&g, 81, 1.0);
                gpu::check_training(&g, &feeds, &gpu::Options::default())
            } else {
                g.set_outputs(vec![y]);
                feeds.fill_random(&g, 82, 1.0);
                gpu::check_inference(&g, &feeds, &gpu::Options::default())
            };
            report
                .unwrap()
                .assert_passed(&format!("{label} training={training}"));
        }
    }
}

/// GroupNorm backward on inputs whose mean (50) dwarfs their spread (0.25),
/// with the input trained and frozen: in the frozen case only the weight
/// and bias gradients request the shared statistics.
#[test]
fn group_norm_backward_with_a_large_mean() {
    let (batch, channels, spatial, groups) = (2u32, 4u32, 64u32, 2u32);
    let n = (batch * channels * spatial) as usize;
    let mut rng = meganeura::reference::Rng::new(90);
    let x: Vec<f32> = (0..n).map(|_| rng.uniform(49.75, 50.25)).collect();
    for train_input in [true, false] {
        let mut g = Graph::new();
        let xs = if train_input {
            g.parameter("x", &[n])
        } else {
            g.input("x", &[n])
        };
        let w = g.parameter("w", &[channels as usize]);
        let b = g.parameter("b", &[channels as usize]);
        let y = g.group_norm(xs, w, b, batch, channels, spatial, groups, 1e-5);
        let loss = gradients::weighted_loss(&mut g, y, 91, 1.3);
        g.set_outputs(vec![loss]);
        let mut feeds = Feeds::new();
        feeds.set("x", &x);
        feeds.fill_random(&g, 92, 1.0);
        gpu::check_training(&g, &feeds, &gpu::Options::default())
            .unwrap()
            .assert_passed(&format!("train_input={train_input}"));
    }
}

/// A per-channel bias followed by ReLU fuses into one generated kernel.
#[test]
fn per_channel_bias_then_relu() {
    let (batch, channels, spatial) = (3usize, 7usize, 45usize);
    let mut g = Graph::new();
    let x = g.input("x", &[batch * channels * spatial]);
    let b = g.input("b", &[channels]);
    let y = g.add_per_channel(x, b, channels as u32, spatial as u32);
    let y = g.relu(y);
    g.set_outputs(vec![y]);
    let mut feeds = Feeds::new();
    feeds.fill_random(&g, 93, 1.0);
    for (lowering, options) in gpu::Options::lowerings() {
        gpu::check_inference(&g, &feeds, &options)
            .unwrap()
            .assert_passed(lowering);
    }
}
