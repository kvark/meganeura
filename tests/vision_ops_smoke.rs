/// Smoke tests for vision/VLA ops on GPU (lavapipe).
/// Tests LayerNorm, GELU, and FullAttention individually.
use meganeura::{Graph, build_inference_session};

#[test]
fn layer_norm_basic() {
    let mut g = Graph::new();
    let x = g.input("x", &[2, 4]); // 2 rows, 4 cols
    let w = g.parameter("w", &[4]);
    let b = g.parameter("b", &[4]);
    let y = g.layer_norm(x, w, b, 1e-5);
    g.set_outputs(vec![y]);

    let mut session = build_inference_session(&g);
    session.set_parameter("w", &[1.0, 1.0, 1.0, 1.0f32]);
    session.set_parameter("b", &[0.0, 0.0, 0.0, 0.0f32]);
    session.set_input("x", &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0f32]);

    session.step();
    session.wait();

    let out = session.read_output(8);
    println!("layer_norm output: {:?}", out);
    for (i, v) in out.iter().enumerate() {
        assert!(
            v.is_finite(),
            "layer_norm output[{}] = {} is not finite",
            i,
            v
        );
    }
    // Each row should be normalized to mean≈0, std≈1
    let row0_mean: f32 = out[0..4].iter().sum::<f32>() / 4.0;
    assert!(
        row0_mean.abs() < 0.01,
        "row0 mean should be ~0, got {}",
        row0_mean
    );
}

#[test]
fn gelu_basic() {
    let mut g = Graph::new();
    let x = g.input("x", &[4, 1]);
    let y = g.gelu(x);
    g.set_outputs(vec![y]);

    let mut session = build_inference_session(&g);
    session.set_input("x", &[-2.0, -1.0, 0.0, 1.0f32]);

    session.step();
    session.wait();

    let out = session.read_output(4);
    println!("gelu output: {:?}", out);
    for (i, v) in out.iter().enumerate() {
        assert!(v.is_finite(), "gelu output[{}] = {} is not finite", i, v);
    }
    // GELU(0) ≈ 0, GELU(1) ≈ 0.841
    assert!(out[2].abs() < 0.01, "GELU(0) should be ~0, got {}", out[2]);
    assert!(
        (out[3] - 0.841).abs() < 0.05,
        "GELU(1) should be ~0.841, got {}",
        out[3]
    );
}

#[test]
fn full_attention_basic() {
    // seq=4, num_heads=2, head_dim=3 → q_dim = k_dim = v_dim = 6
    let mut g = Graph::new();
    let q = g.input("q", &[4, 6]);
    let k = g.input("k", &[4, 6]);
    let v = g.input("v", &[4, 6]);
    let attn = g.full_attention(q, k, v, 2, 2, 3);
    g.set_outputs(vec![attn]);

    let mut session = build_inference_session(&g);

    // Simple identity-like attention: q=k so attention is uniform
    let qk_data: Vec<f32> = (0..24).map(|i| (i as f32) * 0.1).collect();
    let v_data: Vec<f32> = (0..24).map(|i| (i as f32) * 0.01).collect();

    session.set_input("q", &qk_data);
    session.set_input("k", &qk_data);
    session.set_input("v", &v_data);

    session.step();
    session.wait();

    let out = session.read_output(24);
    println!("full_attention output: {:?}", out);
    for (i, v) in out.iter().enumerate() {
        assert!(
            v.is_finite(),
            "full_attention output[{}] = {} is not finite",
            i,
            v
        );
    }
}

#[test]
fn vision_encoder_one_layer() {
    // Minimal vision encoder: 4 patches, 1 layer, 8 hidden, 2 heads
    let num_patches = 4;
    let hidden = 8;
    let num_heads = 2u32;
    let head_dim = (hidden / num_heads as usize) as u32;
    let _intermediate = 16;

    let mut g = Graph::new();
    let x = g.input("x", &[num_patches, hidden]);

    // LayerNorm
    let ln_w = g.parameter("ln1_w", &[hidden]);
    let ln_b = g.parameter("ln1_b", &[hidden]);
    let h = g.layer_norm(x, ln_w, ln_b, 1e-6);

    // QKV projections (no bias for simplicity)
    let wq = g.parameter("wq", &[hidden, hidden]);
    let wk = g.parameter("wk", &[hidden, hidden]);
    let wv = g.parameter("wv", &[hidden, hidden]);

    let q = g.matmul(h, wq);
    let k = g.matmul(h, wk);
    let v = g.matmul(h, wv);

    let attn = g.full_attention(q, k, v, num_heads, num_heads, head_dim);

    // Output projection
    let wo = g.parameter("wo", &[hidden, hidden]);
    let attn_out = g.matmul(attn, wo);

    // Residual
    let out = g.add(x, attn_out);
    g.set_outputs(vec![out]);

    let mut session = build_inference_session(&g);

    // Initialize with small random-ish values
    let ones: Vec<f32> = vec![1.0; hidden];
    let zeros: Vec<f32> = vec![0.0; hidden];
    session.set_parameter("ln1_w", &ones);
    session.set_parameter("ln1_b", &zeros);

    let small_weights: Vec<f32> = (0..hidden * hidden)
        .map(|i| (i as f32 * 0.37).sin() * 0.1)
        .collect();
    session.set_parameter("wq", &small_weights);
    session.set_parameter("wk", &small_weights);
    session.set_parameter("wv", &small_weights);
    session.set_parameter("wo", &small_weights);

    let input: Vec<f32> = (0..num_patches * hidden)
        .map(|i| (i as f32 * 0.13).sin() * 0.5)
        .collect();
    session.set_input("x", &input);

    session.step();
    session.wait();

    let out = session.read_output(num_patches * hidden);
    println!("vision_one_layer output: {:?}", out);
    for (i, v) in out.iter().enumerate() {
        assert!(
            v.is_finite(),
            "vision_one_layer output[{}] = {} is not finite",
            i,
            v
        );
    }
}
