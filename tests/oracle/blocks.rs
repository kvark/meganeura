//! Whole blocks as models compose them, checked both ways: autodiff against
//! f64 finite differences, and the compiled step against the reference.
//! Single-op sweeps cannot see how a block's ops interact after fusion.

use meganeura::reference::{Feeds, Tolerance, gpu, gradients};
use meganeura::{Graph, NodeId};

/// Errors compound through a block; a few operations deep, the final op's
/// own bound is not the whole budget.
fn block_options() -> gpu::Options {
    gpu::Options {
        tolerance: Tolerance {
            rtol: 1e-3,
            floor: 1e-3,
        },
        ..Default::default()
    }
}

fn check(label: &str, g: &Graph, feeds: &Feeds) {
    gradients::check(g, feeds, &gradients::Options::default())
        .unwrap()
        .assert_passed(&format!("{label}: autodiff"));
    gpu::check_training(g, feeds, &block_options())
        .unwrap()
        .assert_passed(&format!("{label}: training"));
}

/// Decoder layer: RMSNorm, Q/K/V projections, RoPE, causal GQA attention,
/// output projection and residual, then RMSNorm, SwiGLU MLP and residual,
/// into a cross-entropy over a small vocabulary.
fn decoder_layer(g: &mut Graph, seq: usize, x: NodeId) -> NodeId {
    let (heads, kv_heads, head_dim) = (2u32, 1u32, 8u32);
    let hidden = (heads * head_dim) as usize;
    let kv = (kv_heads * head_dim) as usize;
    let norm1 = g.parameter("norm1", &[hidden]);
    let wq = g.parameter("wq", &[hidden, hidden]);
    let wk = g.parameter("wk", &[hidden, kv]);
    let wv = g.parameter("wv", &[hidden, kv]);
    let wo = g.parameter("wo", &[hidden, hidden]);
    let h = g.rms_norm(x, norm1, 1e-5);
    let q = g.matmul(h, wq);
    let k = g.matmul(h, wk);
    let v = g.matmul(h, wv);
    let q = g.rope(q, 10_000.0, head_dim);
    let k = g.rope(k, 10_000.0, head_dim);
    let a = g.causal_attention(q, k, v, heads, kv_heads, head_dim);
    let a = g.matmul(a, wo);
    let x = g.add(x, a);
    let norm2 = g.parameter("norm2", &[hidden]);
    let gate = g.parameter("gate", &[hidden, 24]);
    let up = g.parameter("up", &[hidden, 24]);
    let down = g.parameter("down", &[24, hidden]);
    let h = g.rms_norm(x, norm2, 1e-5);
    let gated = g.matmul(h, gate);
    let upped = g.matmul(h, up);
    let m = g.swiglu(gated, upped);
    let m = g.matmul(m, down);
    let _ = seq;
    g.add(x, m)
}

#[test]
fn decoder_layer_to_cross_entropy() {
    let (seq, hidden, vocab) = (5, 16, 7);
    let mut g = Graph::new();
    let x = g.parameter("x", &[seq, hidden]);
    let y = decoder_layer(&mut g, seq, x);
    let head = g.parameter("head", &[hidden, vocab]);
    let logits = g.matmul(y, head);
    let labels = g.input("labels", &[seq, vocab]);
    let loss = g.cross_entropy_loss(logits, labels);
    g.set_outputs(vec![loss]);
    let mut feeds = Feeds::new();
    let mut one_hot = vec![0.0f32; seq * vocab];
    for row in 0..seq {
        one_hot[row * vocab + (row * 3) % vocab] = 1.0;
    }
    feeds.set("labels", &one_hot);
    feeds.fill_random(&g, 500, 0.8);
    check("decoder layer", &g, &feeds);
}

/// Vision-encoder layer: LayerNorm, full multi-head attention, residual,
/// LayerNorm, GELU MLP, residual.
#[test]
fn vision_encoder_layer() {
    let (patches, heads, head_dim) = (6usize, 2u32, 8u32);
    let hidden = (heads * head_dim) as usize;
    let mut g = Graph::new();
    let x = g.parameter("x", &[patches, hidden]);
    let ln1_w = g.parameter("ln1_w", &[hidden]);
    let ln1_b = g.parameter("ln1_b", &[hidden]);
    let h = g.layer_norm(x, ln1_w, ln1_b, 1e-6);
    let project = |g: &mut Graph, name: &str| {
        let w = g.parameter(name, &[hidden, hidden]);
        g.matmul(h, w)
    };
    let q = project(&mut g, "wq");
    let k = project(&mut g, "wk");
    let v = project(&mut g, "wv");
    let a = g.full_attention(q, k, v, heads, heads, head_dim);
    let wo = g.parameter("wo", &[hidden, hidden]);
    let a = g.matmul(a, wo);
    let x = g.add(x, a);
    let ln2_w = g.parameter("ln2_w", &[hidden]);
    let ln2_b = g.parameter("ln2_b", &[hidden]);
    let h = g.layer_norm(x, ln2_w, ln2_b, 1e-6);
    let fc1 = g.parameter("fc1", &[hidden, 32]);
    let b1 = g.parameter("b1", &[32]);
    let fc2 = g.parameter("fc2", &[32, hidden]);
    let m = g.matmul(h, fc1);
    let m = g.bias_add(m, b1);
    let m = g.gelu(m);
    let m = g.matmul(m, fc2);
    let y = g.add(x, m);
    let loss = gradients::weighted_loss(&mut g, y, 501, 0.6);
    g.set_outputs(vec![loss, y]);
    let mut feeds = Feeds::new();
    feeds.fill_random(&g, 502, 0.8);
    check("vision encoder layer", &g, &feeds);
}

/// U-Net-style residual block: GroupNorm, SiLU, 3×3 convolution, twice, a
/// per-channel bias from a time embedding, and a 1×1 skip projection.
#[test]
fn residual_conv_block() {
    let (batch, cin, cout, h, w, groups) = (2u32, 4u32, 6u32, 6u32, 5u32, 2u32);
    let spatial = h * w;
    let mut g = Graph::new();
    let x = g.parameter("x", &[(batch * cin * spatial) as usize]);
    let n1_w = g.parameter("n1_w", &[cin as usize]);
    let n1_b = g.parameter("n1_b", &[cin as usize]);
    let a = g.group_norm(x, n1_w, n1_b, batch, cin, spatial, groups, 1e-5);
    let a = g.silu(a);
    let k1 = g.parameter("k1", &[(cout * cin * 9) as usize]);
    let a = g.conv2d(a, k1, batch, cin, h, w, cout, 3, 3, 1, 1);
    let t = g.parameter("t", &[cout as usize]);
    let a = g.add_per_channel(a, t, cout, spatial);
    let n2_w = g.parameter("n2_w", &[cout as usize]);
    let n2_b = g.parameter("n2_b", &[cout as usize]);
    let a = g.group_norm(a, n2_w, n2_b, batch, cout, spatial, groups, 1e-5);
    let a = g.silu(a);
    let k2 = g.parameter("k2", &[(cout * cout * 9) as usize]);
    let a = g.conv2d(a, k2, batch, cout, h, w, cout, 3, 3, 1, 1);
    let skip = g.parameter("skip", &[(cout * cin) as usize]);
    let s = g.conv2d(x, skip, batch, cin, h, w, cout, 1, 1, 1, 0);
    let y = g.add(a, s);
    let loss = gradients::weighted_loss(&mut g, y, 503, 0.4);
    g.set_outputs(vec![loss]);
    let mut feeds = Feeds::new();
    feeds.fill_random(&g, 504, 1.0);
    check("residual conv block", &g, &feeds);
}
