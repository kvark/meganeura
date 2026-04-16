/// Gradient debugging: test individual components with large weights
use meganeura::{Graph, build_inference_session, build_session, compile::BufferRef};
use std::collections::HashMap;

fn name_seed(name: &str) -> f32 {
    let mut h: u32 = 0;
    for c in name.bytes() {
        h = h.wrapping_mul(31).wrapping_add(c as u32);
    }
    (h % 10000) as f32
}
#[allow(clippy::too_many_arguments)]
fn check_grad(
    label: &str,
    build_train: impl Fn() -> Graph,
    build_infer: impl Fn() -> Graph,
    params_init: &[(String, Vec<f32>)],
    inputs: &[(String, Vec<f32>)],
    inputs_u32: &[(String, Vec<u32>)],
    check_param: &str,
    check_indices: &[usize],
) {
    let g_train = build_train();
    let mut train_sess = build_session(&g_train);
    for (name, data) in params_init {
        train_sess.set_parameter(name, data);
    }
    for (name, data) in inputs {
        train_sess.set_input(name, data);
    }
    for (name, data) in inputs_u32 {
        train_sess.set_input_u32(name, data);
    }
    train_sess.step();
    train_sess.wait();
    let train_loss = train_sess.read_loss();

    let plan = train_sess.plan().clone();
    let param_bufs: HashMap<String, BufferRef> = plan.param_buffers.iter().cloned().collect();
    let grad_map: HashMap<BufferRef, BufferRef> = plan.param_grad_pairs.iter().cloned().collect();
    let buf = param_bufs[check_param];
    let grad_buf = grad_map[&buf];
    let n = plan.buffers[grad_buf.0 as usize] / 4;
    let mut grad = vec![0.0f32; n];
    train_sess.read_buffer(grad_buf, &mut grad);

    // Finite differences
    let g_infer = build_infer();
    let mut infer_sess = build_inference_session(&g_infer);
    let orig_data = params_init
        .iter()
        .find(|(name, _)| name == check_param)
        .unwrap()
        .1
        .clone();

    let eps = 1e-3f32;
    let mut max_rel = 0.0f32;
    for &idx in check_indices {
        if idx >= orig_data.len() {
            continue;
        }
        for (name, data) in params_init {
            infer_sess.set_parameter(name, data);
        }
        let mut perturbed = orig_data.clone();
        perturbed[idx] += eps;
        infer_sess.set_parameter(check_param, &perturbed);
        for (name, data) in inputs {
            infer_sess.set_input(name, data);
        }
        for (name, data) in inputs_u32 {
            infer_sess.set_input_u32(name, data);
        }
        infer_sess.step();
        infer_sess.wait();
        let lp = infer_sess.read_loss();

        perturbed[idx] -= 2.0 * eps;
        infer_sess.set_parameter(check_param, &perturbed);
        for (name, data) in inputs {
            infer_sess.set_input(name, data);
        }
        for (name, data) in inputs_u32 {
            infer_sess.set_input_u32(name, data);
        }
        infer_sess.step();
        infer_sess.wait();
        let lm = infer_sess.read_loss();

        let num = (lp - lm) / (2.0 * eps);
        let ana = grad[idx];
        let rel = (num - ana).abs() / num.abs().max(ana.abs()).max(1e-8);
        eprintln!("{label} {check_param}[{idx}]: ana={ana:.6e} num={num:.6e} rel={rel:.4}");
        max_rel = max_rel.max(rel);
    }
    eprintln!("{label}: loss={train_loss:.6} max_rel_error={max_rel:.6}");
    assert!(
        max_rel < 0.15,
        "{label} gradient error too large: {max_rel}"
    );
}

/// Test RMSNorm + matmul + cross_entropy (no attention)
#[test]
fn grad_rmsnorm_matmul_ce_large() {
    let seq = 4;
    let hidden = 64;
    let vocab = 32;
    let scale = 1.0f32;

    let make_graph = || {
        let mut g = Graph::new();
        let x = g.parameter("x", &[seq, hidden]);
        let w_norm = g.parameter("w_norm", &[hidden]);
        let h = g.rms_norm(x, w_norm, 1e-5);
        let w_proj = g.parameter("w_proj", &[hidden, vocab]);
        let logits = g.matmul(h, w_proj);
        let labels = g.input("labels", &[seq, vocab]);
        let loss = g.cross_entropy_loss(logits, labels);
        g.set_outputs(vec![loss]);
        g
    };

    let x_data: Vec<f32> = (0..seq * hidden)
        .map(|i| (i as f32 * 0.1).sin() * scale)
        .collect();
    let w_norm_data: Vec<f32> = (0..hidden)
        .map(|i| (i as f32 * 0.2 + 1.0).cos() * scale)
        .collect();
    let w_proj_data: Vec<f32> = (0..hidden * vocab)
        .map(|i| (i as f32 * 0.07).sin() * scale)
        .collect();
    let mut label_data = vec![0.0f32; seq * vocab];
    for pos in 0..seq {
        label_data[pos * vocab + (pos + 1) % vocab] = 1.0;
    }

    let params = vec![
        ("x".to_string(), x_data),
        ("w_norm".to_string(), w_norm_data),
        ("w_proj".to_string(), w_proj_data),
    ];
    let inputs = vec![("labels".to_string(), label_data)];

    check_grad(
        "RMSNorm+MatMul+CE(scale=1.0)",
        make_graph,
        make_graph,
        &params,
        &inputs,
        &[],
        "w_norm",
        &[0, 1, 10, 32, 63],
    );
}

/// Test causal attention backward with large inputs
#[test]
fn grad_causal_attn_large() {
    let seq = 4;
    let num_heads: u32 = 1;
    let num_kv_heads: u32 = 1;
    let head_dim: u32 = 64;
    let d = (num_heads * head_dim) as usize;
    let scale = 1.0f32;

    let make_graph = || {
        let mut g = Graph::new();
        let q = g.parameter("q", &[seq, d]);
        let k = g.parameter("k", &[seq, d]);
        let v = g.parameter("v", &[seq, d]);
        let out = g.causal_attention(q, k, v, num_heads, num_kv_heads, head_dim);
        let loss = g.mean_all(out);
        g.set_outputs(vec![loss]);
        g
    };

    let q_data: Vec<f32> = (0..seq * d)
        .map(|i| (i as f32 * 0.01).sin() * scale)
        .collect();
    let k_data: Vec<f32> = (0..seq * d)
        .map(|i| (i as f32 * 0.013 + 1.0).sin() * scale)
        .collect();
    let v_data: Vec<f32> = (0..seq * d)
        .map(|i| (i as f32 * 0.017 + 2.0).sin() * scale)
        .collect();

    let params = vec![
        ("q".to_string(), q_data),
        ("k".to_string(), k_data),
        ("v".to_string(), v_data),
    ];

    check_grad(
        "CausalAttn(scale=1.0)",
        make_graph,
        make_graph,
        &params,
        &[],
        &[],
        "q",
        &[0, 8, 32, 63, 128, 200, 255],
    );
}

/// Test RMSNorm → QKV projections → attention → residual → mean (like SmolLM2 layer)
#[test]
fn grad_layer_attn_block_large() {
    let seq = 4;
    let hidden = 64; // Must be divisible by num_heads and head_dim must be 64
    let num_heads: u32 = 1;
    let num_kv_heads: u32 = 1;
    let head_dim: u32 = 64;
    let scale = 1.0f32;

    let make_graph = || {
        let mut g = Graph::new();
        let x = g.parameter("x", &[seq, hidden]);
        let w_norm = g.parameter("w_norm", &[hidden]);
        let h = g.rms_norm(x, w_norm, 1e-5);
        let wq = g.parameter("wq", &[hidden, hidden]);
        let wk = g.parameter("wk", &[hidden, hidden]);
        let wv = g.parameter("wv", &[hidden, hidden]);
        let q = g.matmul(h, wq);
        let k = g.matmul(h, wk);
        let v = g.matmul(h, wv);
        let attn = g.causal_attention(q, k, v, num_heads, num_kv_heads, head_dim);
        let wo = g.parameter("wo", &[hidden, hidden]);
        let attn_out = g.matmul(attn, wo);
        let out = g.add(x, attn_out); // residual
        let loss = g.mean_all(out);
        g.set_outputs(vec![loss]);
        g
    };

    let x_data: Vec<f32> = (0..seq * hidden)
        .map(|i| (i as f32 * 0.1).sin() * scale)
        .collect();
    let w_norm_data: Vec<f32> = (0..hidden)
        .map(|i| (i as f32 * 0.2 + 1.0).cos() * scale)
        .collect();
    let wq_data: Vec<f32> = (0..hidden * hidden)
        .map(|i| (i as f32 * 0.007).sin() * scale)
        .collect();
    let wk_data: Vec<f32> = (0..hidden * hidden)
        .map(|i| (i as f32 * 0.009 + 0.5).sin() * scale)
        .collect();
    let wv_data: Vec<f32> = (0..hidden * hidden)
        .map(|i| (i as f32 * 0.011 + 1.0).sin() * scale)
        .collect();
    let wo_data: Vec<f32> = (0..hidden * hidden)
        .map(|i| (i as f32 * 0.013 + 1.5).sin() * scale)
        .collect();

    let params = vec![
        ("x".to_string(), x_data),
        ("w_norm".to_string(), w_norm_data),
        ("wq".to_string(), wq_data),
        ("wk".to_string(), wk_data),
        ("wv".to_string(), wv_data),
        ("wo".to_string(), wo_data),
    ];

    check_grad(
        "AttnBlock(scale=1.0)",
        make_graph,
        make_graph,
        &params,
        &[],
        &[],
        "w_norm",
        &[0, 1, 10, 32, 63],
    );
}

/// Test full transformer layer (attn + FFN + residuals) → cross_entropy
#[test]
fn grad_full_layer_ce_large() {
    let seq = 4;
    let hidden: usize = std::env::var("HIDDEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let num_heads: u32 = std::env::var("NHEADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let num_kv_heads: u32 = std::env::var("NKVHEADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let head_dim: u32 = hidden as u32 / num_heads;
    let ffn = hidden * 2;
    let vocab = 32;
    let scale: f32 = std::env::var("SCALE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1.0);
    eprintln!(
        "Config: hidden={hidden} heads={num_heads}/{num_kv_heads} head_dim={head_dim} scale={scale}"
    );

    let make_graph = || {
        let mut g = Graph::new();
        let token_ids = g.input_u32("token_ids", &[seq]);
        let embed = g.parameter("embed", &[vocab, hidden]);
        let mut x = g.embedding(token_ids, embed);

        // Attention block
        let ln1 = g.parameter("ln1", &[hidden]);
        let h = g.rms_norm(x, ln1, 1e-5);
        let kv_dim = (num_kv_heads * head_dim) as usize;
        let wq = g.parameter("wq", &[hidden, hidden]);
        let wk = g.parameter("wk", &[hidden, kv_dim]);
        let wv = g.parameter("wv", &[hidden, kv_dim]);
        let q = g.matmul(h, wq);
        let k = g.matmul(h, wk);
        let v = g.matmul(h, wv);
        // RoPE (can be disabled with NO_ROPE env var)
        let (q, k) = if std::env::var("NO_ROPE").is_err() {
            (g.rope(q, 10000.0, head_dim), g.rope(k, 10000.0, head_dim))
        } else {
            (q, k)
        };
        let attn = g.causal_attention(q, k, v, num_heads, num_kv_heads, head_dim);
        let wo = g.parameter("wo", &[hidden, hidden]);
        let attn_out = g.matmul(attn, wo);
        x = g.add(x, attn_out);

        // FFN block
        let ln2 = g.parameter("ln2", &[hidden]);
        let h = g.rms_norm(x, ln2, 1e-5);
        let w_gate = g.parameter("w_gate", &[hidden, ffn]);
        let w_up = g.parameter("w_up", &[hidden, ffn]);
        let w_down = g.parameter("w_down", &[ffn, hidden]);
        let gate = g.matmul(h, w_gate);
        let up = g.matmul(h, w_up);
        let ffn_out = g.swiglu(gate, up);
        let ffn_out = g.matmul(ffn_out, w_down);
        x = g.add(x, ffn_out);

        // Final norm + LM head (tied) + CE
        let ln_final = g.parameter("ln_final", &[hidden]);
        x = g.rms_norm(x, ln_final, 1e-5);
        let logits = g.matmul_bt(x, embed); // weight tying
        let labels = g.input("labels", &[seq, vocab]);
        let loss = g.cross_entropy_loss(logits, labels);
        g.set_outputs(vec![loss]);
        g
    };

    let seed = |name: &str| -> f32 { name_seed(name) };
    let init = |name: &str, n: usize| -> Vec<f32> {
        let s = seed(name);
        (0..n)
            .map(|i| (i as f32 * 0.01 + s).sin() * scale)
            .collect()
    };

    let mut label_data = vec![0.0f32; seq * vocab];
    for pos in 0..seq {
        label_data[pos * vocab + (pos + 1) % vocab] = 1.0;
    }

    let kv_dim = (num_kv_heads * head_dim) as usize;
    let params = vec![
        ("embed".to_string(), init("embed", vocab * hidden)),
        ("ln1".to_string(), init("ln1", hidden)),
        ("wq".to_string(), init("wq", hidden * hidden)),
        ("wk".to_string(), init("wk", hidden * kv_dim)),
        ("wv".to_string(), init("wv", hidden * kv_dim)),
        ("wo".to_string(), init("wo", hidden * hidden)),
        ("ln2".to_string(), init("ln2", hidden)),
        ("w_gate".to_string(), init("w_gate", hidden * ffn)),
        ("w_up".to_string(), init("w_up", hidden * ffn)),
        ("w_down".to_string(), init("w_down", ffn * hidden)),
        ("ln_final".to_string(), init("ln_final", hidden)),
    ];
    let inputs = vec![("labels".to_string(), label_data)];
    let inputs_u32 = vec![("token_ids".to_string(), (0..seq as u32).collect())];

    check_grad(
        "FullLayer+CE(scale=1.0)",
        make_graph,
        make_graph,
        &params,
        &inputs,
        &inputs_u32,
        "ln1",
        &[0, 1, 10, 32, 63],
    );
}

/// Test matmul + cross_entropy only (no RMSNorm)
#[test]
fn grad_matmul_ce_large() {
    let seq = 4;
    let hidden = 64;
    let vocab = 32;
    let scale = 1.0f32;

    let make_graph = || {
        let mut g = Graph::new();
        let x = g.parameter("x", &[seq, hidden]);
        let w = g.parameter("w", &[hidden, vocab]);
        let logits = g.matmul(x, w);
        let labels = g.input("labels", &[seq, vocab]);
        let loss = g.cross_entropy_loss(logits, labels);
        g.set_outputs(vec![loss]);
        g
    };

    let x_data: Vec<f32> = (0..seq * hidden)
        .map(|i| (i as f32 * 0.1).sin() * scale)
        .collect();
    let w_data: Vec<f32> = (0..hidden * vocab)
        .map(|i| (i as f32 * 0.07).sin() * scale)
        .collect();
    let mut label_data = vec![0.0f32; seq * vocab];
    for pos in 0..seq {
        label_data[pos * vocab + (pos + 1) % vocab] = 1.0;
    }

    let params = vec![("x".to_string(), x_data), ("w".to_string(), w_data)];
    let inputs = vec![("labels".to_string(), label_data)];

    check_grad(
        "MatMul+CE(scale=1.0)",
        make_graph,
        make_graph,
        &params,
        &inputs,
        &[],
        "w",
        &[0, 3, 50, 100, 500, 2047],
    );
}
