/// GPU smoke test: validates that all shaders compile with blade + lavapipe
/// and that a simple forward pass executes without errors.
use meganeura::{
    Graph, build_inference_session, build_session,
    compile::BufferRef,
    models::smolvla::{self, SmolVLAConfig},
};

#[test]
fn matmul_non_uniform_values() {
    // Non-uniform matmul: A (16×16) where A[i,j]=i+1, B (16×16) where B[i,j]=j+1.
    // C[i,j] = sum_{k=0}^{15} (i+1) * (k+1) = (i+1) * sum_{k=0}^{15}(k+1) = (i+1) * 136.
    // This catches A@B vs B@A bugs because B@A[i,j] = (j+1)*136 ≠ (i+1)*136 for i≠j.
    let m = 16;
    let k = 16;
    let n = 16;
    let mut g = Graph::new();
    let a = g.input("a", &[m, k]);
    let b = g.parameter("b", &[k, n]);
    let c = g.matmul(a, b);
    g.set_outputs(vec![c]);

    let mut session = build_inference_session(&g);

    let a_data: Vec<f32> = (0..m * k).map(|i| (i / k + 1) as f32).collect(); // A[i,j] = i+1
    let b_data: Vec<f32> = (0..k * n).map(|i| (i % n + 1) as f32).collect(); // B[i,j] = j+1
    session.set_input("a", &a_data);
    session.set_parameter("b", &b_data);

    session.step();
    session.wait();

    let output = session.read_output(m * n);
    // C[i,j] = sum_{k=0}^{15} A[i,k]*B[k,j] = sum_k (i+1)*(j+1) = 16*(i+1)*(j+1)
    // B@A[i,j] = sum_k (k+1)*(j+1) = (j+1)*136  — varies only with j, not i
    // A@B[i,j] = 16*(i+1)*(j+1)               — varies with both i and j
    for row in 0..m {
        for col in 0..n {
            let got = output[row * n + col];
            let expected = k as f32 * (row as f32 + 1.0) * (col as f32 + 1.0);
            assert!(
                (got - expected).abs() < expected.abs() * 0.02 + 1.0,
                "C[{},{}]: got {}, expected {} (f16 precision)",
                row,
                col,
                got,
                expected
            );
        }
    }
}

#[test]
fn shader_compilation_and_forward_pass() {
    // Build a small MLP graph
    let batch = 4;
    let mut g = Graph::new();
    let x = g.input("x", &[batch, 8]);
    let labels = g.input("labels", &[batch, 3]);
    let w1 = g.parameter("w1", &[8, 5]);
    let b1 = g.parameter("b1", &[5]);
    let mm1 = g.matmul(x, w1);
    let h1 = g.bias_add(mm1, b1);
    let a1 = g.relu(h1);
    let w2 = g.parameter("w2", &[5, 3]);
    let mm2 = g.matmul(a1, w2);
    let loss = g.cross_entropy_loss(mm2, labels);
    g.set_outputs(vec![loss]);

    // Build session (this compiles all shaders via blade)
    let mut session = build_session(&g);

    // Initialize with small data
    let w1_data = vec![0.1_f32; 8 * 5];
    let b1_data = vec![0.0_f32; 5];
    let w2_data = vec![0.1_f32; 5 * 3];
    session.set_parameter("w1", &w1_data);
    session.set_parameter("b1", &b1_data);
    session.set_parameter("w2", &w2_data);

    let x_data = vec![1.0_f32; batch * 8];
    let mut labels_data = vec![0.0_f32; batch * 3];
    for b in 0..batch {
        labels_data[b * 3] = 1.0;
    }
    session.set_input("x", &x_data);
    session.set_input("labels", &labels_data);

    // Execute forward + backward
    session.step();
    session.wait();

    // Read back loss - should be a finite number
    let loss_val = session.read_loss();
    assert!(
        loss_val.is_finite(),
        "loss should be finite, got {}",
        loss_val
    );
    assert!(
        loss_val > 0.0,
        "cross-entropy loss should be positive, got {}",
        loss_val
    );
}

#[test]
fn matmul_produces_correct_values() {
    // 16x32 @ 32x16 matmul — tile-aligned for cooperative matmul (16×16 tiles)
    // A: 16×32 filled with 0.1 → each output element = 32 * 0.1 * 0.1 = 0.32
    // B: 32×16 filled with 0.1
    let m = 16;
    let k = 32;
    let n = 16;

    let mut g = Graph::new();
    let a = g.input("a", &[m, k]);
    let b = g.parameter("b", &[k, n]);
    let c = g.matmul(a, b);
    g.set_outputs(vec![c]);

    let mut session = build_inference_session(&g);

    let a_data = vec![0.1_f32; m * k];
    let b_data = vec![0.1_f32; k * n];
    session.set_input("a", &a_data);
    session.set_parameter("b", &b_data);

    session.step();
    session.wait();

    let output = session.read_output(m * n);
    eprintln!("matmul output first 4: {:?}", &output[..4]);
    assert_eq!(output.len(), m * n);
    // Each element = sum_{i=0}^{k-1} 0.1 * 0.1 = k * 0.01 = 0.32
    let expected = k as f32 * 0.01;
    for (i, &got) in output.iter().enumerate() {
        assert!(
            (got - expected).abs() < 0.02, // f16 precision tolerance
            "matmul output[{}]: got {}, expected {}",
            i,
            got,
            expected
        );
    }
}

#[test]
fn fused_matmul_add_correct() {
    // Test FusedMatMulAdd: C = A × B + D
    // A: 16×32 all 0.1, B: 32×16 all 0.1 → A×B = 0.32 per element
    // D: 16×16 all 1.0 → result should be 1.32 per element
    let m = 16;
    let k = 32;
    let n = 16;

    let mut g = Graph::new();
    let a = g.input("a", &[m, k]);
    let b = g.parameter("b", &[k, n]);
    let d = g.input("d", &[m, n]);
    let mm = g.matmul(a, b);
    let out = g.add(mm, d);
    g.set_outputs(vec![out]);

    let mut session = build_inference_session(&g);

    session.set_input("a", &vec![0.1_f32; m * k]);
    session.set_parameter("b", &vec![0.1_f32; k * n]);
    session.set_input("d", &vec![1.0_f32; m * n]);

    session.step();
    session.wait();

    let output = session.read_output(m * n);
    eprintln!("fused matmul+add first 4: {:?}", &output[..4]);
    let expected = k as f32 * 0.01 + 1.0; // 0.32 + 1.0 = 1.32
    for (i, &got) in output.iter().enumerate() {
        assert!(
            (got - expected).abs() < 0.02,
            "fused_matmul_add output[{}]: got {}, expected {}",
            i,
            got,
            expected
        );
    }
}

#[test]
fn simple_sgd_decreases_loss() {
    // Verify the basic training loop (SGD on matmul+mean_all) actually decreases loss.
    let mut g = Graph::new();
    let x = g.input("x", &[4, 8]);
    let w = g.parameter("w", &[8, 4]);
    let y = g.matmul(x, w);
    let loss = g.mean_all(y);
    g.set_outputs(vec![loss]);

    let mut session = build_session(&g);
    session.set_parameter("w", &vec![0.1_f32; 8 * 4]);
    session.set_input("x", &vec![1.0_f32; 4 * 8]);
    session.step();
    session.wait();
    let initial_loss = session.read_loss();
    assert!(initial_loss.is_finite());

    session.sgd_step_cpu(0.1);
    session.set_input("x", &vec![1.0_f32; 4 * 8]);
    session.step();
    session.wait();
    let final_loss = session.read_loss();
    assert!(
        final_loss < initial_loss,
        "basic SGD should decrease loss: {} → {}",
        initial_loss,
        final_loss
    );
}

#[test]
fn silu_swiglu_rmsnorm_gradients() {
    // Smoke test: backward pass through Silu, SwiGLU, RmsNorm doesn't crash
    // and produces a finite loss.
    let seq = 4;
    let d = 8;
    let mut g = Graph::new();
    let x = g.input("x", &[seq, d]);
    let w1 = g.parameter("w1", &[d, d]);
    let mm1 = g.matmul(x, w1);
    let s = g.silu(mm1); // test Silu backward
    let w_gate = g.parameter("w_gate", &[d, d]);
    let w_up = g.parameter("w_up", &[d, d]);
    let gate = g.matmul(s, w_gate);
    let up = g.matmul(s, w_up);
    let ffn = g.swiglu(gate, up); // test SwiGLU backward
    let rn_w = g.parameter("rn_w", &[d]);
    let rn = g.rms_norm(ffn, rn_w, 1e-5); // test RmsNorm backward
    let loss = g.mean_all(rn);
    g.set_outputs(vec![loss]);

    let mut session = build_session(&g);
    session.set_parameter("w1", &vec![0.1_f32; d * d]);
    session.set_parameter("w_gate", &vec![0.1_f32; d * d]);
    session.set_parameter("w_up", &vec![0.1_f32; d * d]);
    session.set_parameter("rn_w", &vec![1.0_f32; d]);
    session.set_input("x", &vec![0.5_f32; seq * d]);

    session.step();
    session.wait();

    let loss_val = session.read_loss();
    assert!(
        loss_val.is_finite(),
        "loss should be finite after silu/swiglu/rmsnorm backward, got {}",
        loss_val
    );
}

#[test]
fn smolvla_training_backprop_smoke() {
    // MHA backward shaders produce incorrect gradients on lavapipe (software Vulkan).
    // The workgroup shared memory reductions give wrong results when multiple reductions
    // reuse the same wg array in a single shader invocation. Works correctly on real GPUs.
    if std::env::var("MEGANEURA_SKIP_BACKPROP").unwrap_or_default() == "1" {
        eprintln!("MEGANEURA_SKIP_BACKPROP set — skipping MHA backward test");
        return;
    }
    // Smoke test: SmolVLA action expert training graph compiles, runs,
    // and decreases loss over 5 gradient steps.
    let config = SmolVLAConfig::small_test();
    let action_seq_len = config.chunk_size; // 4
    let vlm_seq_len = 4;

    let training_g = smolvla::build_action_expert_training(&config, action_seq_len, vlm_seq_len);
    let mut session = build_session(&training_g);

    // Initialize with small uniform weights
    for (name, buf_ref) in session.plan().param_buffers.clone() {
        let size_bytes = session.plan().buffers[buf_ref.0 as usize];
        let n = size_bytes / 4;
        session.set_parameter(&name, &vec![0.01_f32; n]);
    }

    let expert_hidden = config.expert.hidden_size;
    let kv_dim = config.expert.kv_dim();
    let noisy_actions = vec![0.5_f32; action_seq_len * config.max_action_dim];
    let timestep = vec![0.1_f32; expert_hidden * 2];
    let vlm_kv = vec![0.1_f32; vlm_seq_len * kv_dim];
    let target_actions = vec![0.0_f32; action_seq_len * config.max_action_dim];

    let set_inputs = |s: &mut meganeura::Session| {
        s.set_input("noisy_actions", &noisy_actions);
        s.set_input("timestep", &timestep);
        for i in 0..config.expert.num_layers {
            if i % config.expert.self_attn_every_n_layers != 0 {
                s.set_input(&format!("vlm_kv_layer_{}", i), &vlm_kv);
            }
        }
        s.set_input("target_actions", &target_actions);
    };

    // Diagnostic: check session structure
    let grad_bufs: std::collections::HashSet<u32> = session
        .plan()
        .param_grad_pairs
        .iter()
        .map(|&(p, _)| p.0)
        .collect();
    for (name, buf_ref) in &session.plan().param_buffers {
        let has_grad = grad_bufs.contains(&buf_ref.0);
        eprintln!(
            "  param {:>50}: buf={:>3} grad={}",
            name, buf_ref.0, has_grad
        );
    }
    eprintln!(
        "param_buffers={}, param_grad_pairs={}",
        session.plan().param_buffers.len(),
        session.plan().param_grad_pairs.len()
    );
    assert!(
        !session.plan().param_grad_pairs.is_empty(),
        "no gradient pairs — autodiff may have failed"
    );

    // Step 1 — record initial loss
    set_inputs(&mut session);
    session.step();
    session.wait();
    let initial_loss = session.read_loss();
    assert!(
        initial_loss.is_finite(),
        "initial loss should be finite, got {}",
        initial_loss
    );

    // Steps 2-5 — train with SGD
    let lr = 0.01;
    for _ in 0..4 {
        session.sgd_step_cpu(lr);
        set_inputs(&mut session);
        session.step();
        session.wait();
        let l = session.read_loss();
        assert!(
            l.is_finite(),
            "loss diverged to NaN/inf during training: {}",
            l
        );
    }

    let final_loss = session.read_loss();
    assert!(
        final_loss < initial_loss,
        "loss should decrease after 5 gradient steps: initial={:.6}, final={:.6}",
        initial_loss,
        final_loss
    );
}

#[test]
fn multi_head_attn_gradient_check() {
    if std::env::var("MEGANEURA_SKIP_BACKPROP").unwrap_or_default() == "1" {
        eprintln!("MEGANEURA_SKIP_BACKPROP set — skipping MHA gradient check");
        return;
    }
    // Numerical gradient check for MultiHeadAttn backward pass.
    // Uses head_dim=64 to match the WG=64 hardcoding in attention shaders.
    // Compares analytical gradients (from backprop) to central-difference
    // finite differences for a subset of Q, K, V elements.

    let q_seq: usize = 4;
    let kv_seq: usize = 4;
    let num_heads: u32 = 1;
    let num_kv_heads: u32 = 1;
    let head_dim: u32 = 64;
    let d_q = (num_heads * head_dim) as usize;
    let d_kv = (num_kv_heads * head_dim) as usize;
    let n_q = q_seq * d_q;
    let n_kv = kv_seq * d_kv;

    // --- Analytical gradients via backprop ---
    let mut g_train = Graph::new();
    let qn = g_train.parameter("q", &[q_seq, d_q]);
    let kn = g_train.parameter("k", &[kv_seq, d_kv]);
    let vn = g_train.parameter("v", &[kv_seq, d_kv]);
    let out = g_train.multi_head_attn(qn, kn, vn, num_heads, num_kv_heads, head_dim, false);
    let loss_node = g_train.mean_all(out);
    g_train.set_outputs(vec![loss_node]);

    let mut train_sess = build_session(&g_train);

    // Varied initialisation — using sin patterns to avoid degenerate cases
    let q_data: Vec<f32> = (0..n_q).map(|i| (i as f32 * 0.01).sin() * 0.1).collect();
    let k_data: Vec<f32> = (0..n_kv)
        .map(|i| (i as f32 * 0.013 + 1.0).sin() * 0.1)
        .collect();
    let v_data: Vec<f32> = (0..n_kv)
        .map(|i| (i as f32 * 0.017 + 2.0).sin() * 0.1)
        .collect();

    train_sess.set_parameter("q", &q_data);
    train_sess.set_parameter("k", &k_data);
    train_sess.set_parameter("v", &v_data);

    train_sess.step();
    train_sess.wait();

    let loss_val = train_sess.read_loss();
    assert!(loss_val.is_finite(), "MHA loss is not finite: {}", loss_val);

    // Map parameter names → (param_buf, grad_buf)
    let param_buffers: std::collections::HashMap<String, BufferRef> =
        train_sess.plan().param_buffers.iter().cloned().collect();
    let grad_map: std::collections::HashMap<BufferRef, BufferRef> =
        train_sess.plan().param_grad_pairs.iter().cloned().collect();

    let q_buf = param_buffers["q"];
    let k_buf = param_buffers["k"];
    let v_buf = param_buffers["v"];

    let mut grad_q = vec![0.0f32; n_q];
    let mut grad_k = vec![0.0f32; n_kv];
    let mut grad_v = vec![0.0f32; n_kv];
    train_sess.read_buffer(grad_map[&q_buf], &mut grad_q);
    train_sess.read_buffer(grad_map[&k_buf], &mut grad_k);
    train_sess.read_buffer(grad_map[&v_buf], &mut grad_v);

    // --- Numerical gradients via central differences ---
    let mut g_infer = Graph::new();
    let qi = g_infer.parameter("q", &[q_seq, d_q]);
    let ki = g_infer.parameter("k", &[kv_seq, d_kv]);
    let vi = g_infer.parameter("v", &[kv_seq, d_kv]);
    let out_i = g_infer.multi_head_attn(qi, ki, vi, num_heads, num_kv_heads, head_dim, false);
    let loss_i = g_infer.mean_all(out_i);
    g_infer.set_outputs(vec![loss_i]);
    let mut infer_sess = build_inference_session(&g_infer);

    let fwd = |sess: &mut meganeura::Session, qd: &[f32], kd: &[f32], vd: &[f32]| -> f32 {
        sess.set_parameter("q", qd);
        sess.set_parameter("k", kd);
        sess.set_parameter("v", vd);
        sess.step();
        sess.wait();
        sess.read_loss()
    };

    let eps = 1e-3_f32;
    // Check a spread of element indices across Q, K, V
    let check_idxs = [0usize, 8, 32, 63, 128, 200, 255];
    let mut max_rel_err = 0.0f32;
    let mut checks = 0usize;

    // grad_q
    for &idx in &check_idxs {
        if idx >= n_q {
            continue;
        }
        let mut qd = q_data.clone();
        qd[idx] += eps;
        let lp = fwd(&mut infer_sess, &qd, &k_data, &v_data);
        qd[idx] -= 2.0 * eps;
        let lm = fwd(&mut infer_sess, &qd, &k_data, &v_data);
        let num = (lp - lm) / (2.0 * eps);
        let ana = grad_q[idx];
        let rel = (num - ana).abs() / (num.abs().max(ana.abs()).max(1e-6));
        eprintln!("grad_q[{idx}]: ana={ana:.6e} num={num:.6e} rel={rel:.4}");
        if max_rel_err < rel {
            max_rel_err = rel;
        }
        checks += 1;
    }

    // grad_k
    for &idx in &check_idxs {
        if idx >= n_kv {
            continue;
        }
        let mut kd = k_data.clone();
        kd[idx] += eps;
        let lp = fwd(&mut infer_sess, &q_data, &kd, &v_data);
        kd[idx] -= 2.0 * eps;
        let lm = fwd(&mut infer_sess, &q_data, &kd, &v_data);
        let num = (lp - lm) / (2.0 * eps);
        let ana = grad_k[idx];
        let rel = (num - ana).abs() / (num.abs().max(ana.abs()).max(1e-6));
        eprintln!("grad_k[{idx}]: ana={ana:.6e} num={num:.6e} rel={rel:.4}");
        if max_rel_err < rel {
            max_rel_err = rel;
        }
        checks += 1;
    }

    // grad_v
    for &idx in &check_idxs {
        if idx >= n_kv {
            continue;
        }
        let mut vd = v_data.clone();
        vd[idx] += eps;
        let lp = fwd(&mut infer_sess, &q_data, &k_data, &vd);
        vd[idx] -= 2.0 * eps;
        let lm = fwd(&mut infer_sess, &q_data, &k_data, &vd);
        let num = (lp - lm) / (2.0 * eps);
        let ana = grad_v[idx];
        let rel = (num - ana).abs() / (num.abs().max(ana.abs()).max(1e-6));
        eprintln!("grad_v[{idx}]: ana={ana:.6e} num={num:.6e} rel={rel:.4}");
        if max_rel_err < rel {
            max_rel_err = rel;
        }
        checks += 1;
    }

    assert!(checks > 0, "no gradient elements were checked");
    assert!(
        max_rel_err < 0.05,
        "MultiHeadAttn gradient check FAILED: max relative error {:.4} (>5%) across {} elements",
        max_rel_err,
        checks,
    );
    eprintln!(
        "MultiHeadAttn gradient check PASSED: max relative error {:.4} across {} elements",
        max_rel_err, checks
    );
}

#[test]
fn swiglu_concat_gradient_check() {
    if std::env::var("MEGANEURA_SKIP_BACKPROP").unwrap_or_default() == "1" {
        return;
    }
    // Numerical gradient check for SwiGLUConcat backward.
    // Graph: W_gate_up[d, 2*d] matmul → SwiGLUConcat → mean_all loss.
    // Verifies the concatenated gate+up gradient computation.
    let seq = 4;
    let d = 16;

    // --- Analytical gradients ---
    let mut g_train = Graph::new();
    let x = g_train.input("x", &[seq, d]);
    let w = g_train.parameter("w", &[d, d * 2]);
    let mm = g_train.matmul(x, w);
    let swiglu = g_train.swiglu_concat(mm);
    let loss = g_train.mean_all(swiglu);
    g_train.set_outputs(vec![loss]);

    let mut sess = build_session(&g_train);
    let x_data: Vec<f32> = (0..seq * d)
        .map(|i| (i as f32 * 0.13).sin() * 1.0)
        .collect();
    let w_data: Vec<f32> = (0..d * d * 2)
        .map(|i| (i as f32 * 0.07 + 1.0).sin() * 0.5)
        .collect();
    sess.set_parameter("w", &w_data);
    sess.set_input("x", &x_data);
    sess.step();
    sess.wait();

    let loss_val = sess.read_loss();
    eprintln!("loss = {:.8}", loss_val);
    assert!(loss_val.is_finite(), "loss not finite: {}", loss_val);

    let w_buf = sess
        .plan()
        .param_buffers
        .iter()
        .find(|(n, _)| n == "w")
        .unwrap()
        .1;
    let grad_buf = sess
        .plan()
        .param_grad_pairs
        .iter()
        .find(|(p, _)| *p == w_buf)
        .unwrap()
        .1;
    let mut grad_w = vec![0.0f32; d * d * 2];
    sess.read_buffer(grad_buf, &mut grad_w);

    // --- Numerical gradients via central differences ---
    let mut g_infer = Graph::new();
    let xi = g_infer.input("x", &[seq, d]);
    let wi = g_infer.parameter("w", &[d, d * 2]);
    let mmi = g_infer.matmul(xi, wi);
    let swi = g_infer.swiglu_concat(mmi);
    let li = g_infer.mean_all(swi);
    g_infer.set_outputs(vec![li]);
    let mut isess = build_inference_session(&g_infer);

    let fwd = |s: &mut meganeura::Session, wd: &[f32]| -> f32 {
        s.set_parameter("w", wd);
        s.set_input("x", &x_data);
        s.step();
        s.wait();
        s.read_loss()
    };

    let eps = 1e-3f32;
    let check_idxs: Vec<usize> = (0..d * d * 2).step_by(7).collect();
    let mut max_rel_err = 0.0f32;
    let mut checks = 0;
    for &idx in &check_idxs {
        if idx >= d * d * 2 {
            continue;
        }
        let mut wd = w_data.clone();
        wd[idx] += eps;
        let lp = fwd(&mut isess, &wd);
        wd[idx] -= 2.0 * eps;
        let lm = fwd(&mut isess, &wd);
        let num = (lp - lm) / (2.0 * eps);
        let ana = grad_w[idx];
        let rel = (num - ana).abs() / (num.abs().max(ana.abs()).max(1e-6));
        eprintln!("grad_w[{idx}]: ana={ana:.6e} num={num:.6e} rel={rel:.4}");
        if max_rel_err < rel {
            max_rel_err = rel;
        }
        checks += 1;
    }

    assert!(checks > 0);
    assert!(
        max_rel_err < 0.05,
        "SwiGLUConcat gradient check FAILED: max rel err {:.4} (>5%)",
        max_rel_err,
    );
    eprintln!(
        "SwiGLUConcat gradient check PASSED: max rel err {:.4} across {} elements",
        max_rel_err, checks
    );
}

#[test]
fn abs_log_recip_ops() {
    // Verify Abs, Log, Recip produce correct values on GPU.
    let n = 8;
    let mut g = Graph::new();
    let x = g.input("x", &[1, n]);
    let a = g.abs(x);
    g.set_outputs(vec![a]);

    let mut session = build_inference_session(&g);
    let input: Vec<f32> = vec![-3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0];
    session.set_input("x", &input);
    session.step();
    session.wait();

    let output = session.read_output(n);
    let expected: Vec<f32> = input.iter().map(|x| x.abs()).collect();
    for i in 0..n {
        assert!(
            (output[i] - expected[i]).abs() < 1e-4,
            "abs[{}]: got {}, expected {}",
            i,
            output[i],
            expected[i]
        );
    }

    // Log
    let mut g = Graph::new();
    let x = g.input("x", &[1, n]);
    let l = g.log(x);
    g.set_outputs(vec![l]);

    let mut session = build_inference_session(&g);
    let input: Vec<f32> = vec![0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 100.0];
    session.set_input("x", &input);
    session.step();
    session.wait();

    let output = session.read_output(n);
    for i in 0..n {
        let exp = input[i].ln();
        assert!(
            (output[i] - exp).abs() < 1e-3,
            "log[{}]: got {}, expected {}",
            i,
            output[i],
            exp
        );
    }

    // Recip
    let mut g = Graph::new();
    let x = g.input("x", &[1, n]);
    let r = g.recip(x);
    g.set_outputs(vec![r]);

    let mut session = build_inference_session(&g);
    let input: Vec<f32> = vec![0.1, 0.5, 1.0, 2.0, 4.0, 5.0, 10.0, 100.0];
    session.set_input("x", &input);
    session.step();
    session.wait();

    let output = session.read_output(n);
    for i in 0..n {
        let exp = 1.0 / input[i];
        assert!(
            (output[i] - exp).abs() < 1e-3,
            "recip[{}]: got {}, expected {}",
            i,
            output[i],
            exp
        );
    }
}

#[test]
fn mse_loss_training() {
    // Verify MSE loss produces correct value and SGD decreases it.
    let mut g = Graph::new();
    let x = g.input("x", &[4, 4]);
    let w = g.parameter("w", &[4, 4]);
    let pred = g.matmul(x, w);
    let target = g.input("target", &[4, 4]);
    let loss = g.mse_loss(pred, target);
    g.set_outputs(vec![loss]);

    let mut session = build_session(&g);
    session.set_parameter("w", &vec![0.1_f32; 16]);
    session.set_input("x", &vec![1.0_f32; 16]);
    session.set_input("target", &vec![0.0_f32; 16]);

    session.step();
    session.wait();
    let loss0 = session.read_loss();
    assert!(
        loss0.is_finite() && loss0 > 0.0,
        "initial MSE loss: {}",
        loss0
    );

    session.sgd_step_cpu(0.01);
    session.set_input("x", &vec![1.0_f32; 16]);
    session.set_input("target", &vec![0.0_f32; 16]);
    session.step();
    session.wait();
    let loss1 = session.read_loss();
    assert!(
        loss1 < loss0,
        "SGD should decrease MSE loss: {} -> {}",
        loss0,
        loss1
    );
}

#[test]
fn bce_loss_forward() {
    let n = 4;
    let mut g = Graph::new();
    let pred = g.input("pred", &[1, n]);
    let labels = g.input("labels", &[1, n]);
    let loss = g.bce_loss(pred, labels);
    g.set_outputs(vec![loss]);

    let mut session = build_inference_session(&g);
    session.set_input("pred", &[0.9, 0.1, 0.8, 0.2]);
    session.set_input("labels", &[1.0, 0.0, 1.0, 0.0]);

    session.step();
    session.wait();

    let loss_val = session.read_output(1)[0];
    assert!(
        loss_val.is_finite() && loss_val > 0.0,
        "BCE loss should be finite and positive, got {}",
        loss_val
    );
    // Expected ≈ 0.164 for these well-calibrated predictions
    assert!(
        (loss_val - 0.164).abs() < 0.05,
        "BCE loss expected ~0.164, got {}",
        loss_val
    );
}

#[test]
fn checkpoint_round_trip() {
    let mut g = Graph::new();
    let x = g.input("x", &[4, 8]);
    let w = g.parameter("w", &[8, 4]);
    let y = g.matmul(x, w);
    let loss = g.mean_all(y);
    g.set_outputs(vec![loss]);

    let mut session = build_session(&g);
    session.set_parameter("w", &vec![0.1_f32; 8 * 4]);
    session.set_input("x", &vec![1.0_f32; 4 * 8]);

    // Train 3 steps with Adam
    for _ in 0..3 {
        session.set_input("x", &vec![1.0_f32; 4 * 8]);
        session.adam_step(0.01, 0.9, 0.999, 1e-8);
        session.step();
        session.wait();
    }
    let loss_before = session.read_loss();

    // Save checkpoint
    let tmp = std::env::temp_dir().join("meganeura_test_ckpt.safetensors");
    session.save_checkpoint(&tmp).expect("save checkpoint");

    // Read back parameter
    let w_buf = session
        .plan()
        .param_buffers
        .iter()
        .find(|(n, _)| n == "w")
        .unwrap()
        .1;
    let mut w_saved = vec![0.0f32; 32];
    session.read_buffer(w_buf, &mut w_saved);

    // Fresh session, load checkpoint
    let mut session2 = build_session(&g);
    session2.load_checkpoint(&tmp).expect("load checkpoint");

    let mut w_loaded = vec![0.0f32; 32];
    session2.read_buffer(w_buf, &mut w_loaded);
    for i in 0..32 {
        assert!(
            (w_saved[i] - w_loaded[i]).abs() < 1e-6,
            "w[{}]: saved={} loaded={}",
            i,
            w_saved[i],
            w_loaded[i]
        );
    }

    // Same loss after restore
    session2.set_input("x", &vec![1.0_f32; 4 * 8]);
    session2.step();
    session2.wait();
    let loss_after = session2.read_loss();
    assert!(
        (loss_before - loss_after).abs() < 1e-4,
        "loss mismatch: {} vs {}",
        loss_before,
        loss_after
    );

    std::fs::remove_file(&tmp).ok();
}

/// Smoke test: KV cache ops (cache_write, cached_attention, rope_dynamic) compile and run.
#[test]
fn kv_cache_ops_smoke() {
    let num_heads: u32 = 2;
    let num_kv_heads: u32 = 2;
    let head_dim: u32 = 64;
    let kv_dim = (num_kv_heads * head_dim) as usize;
    let q_dim = (num_heads * head_dim) as usize;
    let max_seq: usize = 16;

    let mut g = Graph::new();
    let q_input = g.input("q", &[1, q_dim]);
    let k_input = g.input("k", &[1, kv_dim]);
    let v_input = g.input("v", &[1, kv_dim]);
    let kv_pos = g.input_u32("kv_pos", &[1]);

    // Pre-allocated cache buffers
    let k_cache = g.parameter("k_cache", &[max_seq, kv_dim]);
    let v_cache = g.parameter("v_cache", &[max_seq, kv_dim]);

    // Write new K/V into cache
    let _k_updated = g.cache_write(k_input, k_cache, kv_pos);
    let _v_updated = g.cache_write(v_input, v_cache, kv_pos);

    // RoPE with dynamic offset
    let q_rope = g.rope_dynamic_offset(q_input, 10000.0, kv_pos);

    // Cached attention
    let attn = g.cached_attention(
        q_rope,
        k_cache,
        v_cache,
        kv_pos,
        num_heads,
        num_kv_heads,
        head_dim,
    );

    g.set_outputs(vec![attn]);

    let mut session = build_inference_session(&g);

    // Initialize caches to zero
    session.set_parameter("k_cache", &vec![0.0f32; max_seq * kv_dim]);
    session.set_parameter("v_cache", &vec![0.0f32; max_seq * kv_dim]);

    // Step 0: write first K/V
    let q_data: Vec<f32> = (0..q_dim).map(|i| (i as f32 * 0.01).sin()).collect();
    let k_data: Vec<f32> = (0..kv_dim).map(|i| (i as f32 * 0.02).sin()).collect();
    let v_data: Vec<f32> = (0..kv_dim).map(|i| (i as f32 * 0.03).sin()).collect();

    session.set_input("q", &q_data);
    session.set_input("k", &k_data);
    session.set_input("v", &v_data);
    session.set_input_u32("kv_pos", &[0]);

    session.step();
    session.wait();

    let out = session.read_output(q_dim);
    assert_eq!(out.len(), q_dim);
    // With only 1 cached position, attention output should be non-zero
    let sum: f32 = out.iter().map(|x| x.abs()).sum();
    assert!(
        sum > 0.0,
        "attention output should be non-zero, got sum={}",
        sum
    );

    // Step 1: add another K/V entry
    session.set_input_u32("kv_pos", &[1]);
    session.step();
    session.wait();

    let out2 = session.read_output(q_dim);
    assert_eq!(out2.len(), q_dim);
    let sum2: f32 = out2.iter().map(|x| x.abs()).sum();
    assert!(sum2 > 0.0, "second step output should be non-zero");
}

/// Smoke test: Conv2d forward compiles and produces correct output.
#[test]
fn conv2d_forward_smoke() {
    // Tiny conv: 1 batch, 1 input channel, 4×4 image, 1 output channel, 3×3 kernel, stride=1, pad=0
    let batch = 1u32;
    let in_c = 1u32;
    let h = 4u32;
    let w = 4u32;
    let out_c = 1u32;
    let kh = 3u32;
    let kw = 3u32;
    let stride = 1u32;
    let padding = 0u32;
    let out_h = (h + 2 * padding - kh) / stride + 1; // 2
    let out_w = (w + 2 * padding - kw) / stride + 1; // 2

    let in_size = (batch * in_c * h * w) as usize;
    let kernel_size = (out_c * in_c * kh * kw) as usize;
    let out_size = (batch * out_c * out_h * out_w) as usize;

    let mut g = Graph::new();
    let input = g.input("input", &[in_size]);
    let kernel = g.parameter("kernel", &[kernel_size]);
    let output = g.conv2d(
        input, kernel, batch, in_c, h, w, out_c, kh, kw, stride, padding,
    );
    g.set_outputs(vec![output]);

    let mut session = build_inference_session(&g);

    // All-ones input, all-ones kernel → each output = 9.0 (sum of 3×3 ones)
    session.set_input("input", &vec![1.0f32; in_size]);
    session.set_parameter("kernel", &vec![1.0f32; kernel_size]);

    session.step();
    session.wait();

    let out = session.read_output(out_size);
    assert_eq!(out.len(), out_size);
    for (i, &v) in out.iter().enumerate() {
        assert!(
            (v - 9.0).abs() < 1e-4,
            "output[{}] = {}, expected 9.0",
            i,
            v
        );
    }
}

/// Smoke test: Conv2d with padding.
#[test]
fn conv2d_padding_smoke() {
    // 1 batch, 1 channel, 3×3 input, 1 output channel, 3×3 kernel, stride=1, padding=1
    // Output should be 3×3 (same size as input)
    let batch = 1u32;
    let in_c = 1u32;
    let h = 3u32;
    let w = 3u32;
    let out_c = 1u32;
    let kh = 3u32;
    let kw = 3u32;
    let stride = 1u32;
    let padding = 1u32;
    let out_h = (h + 2 * padding - kh) / stride + 1; // 3
    let out_w = (w + 2 * padding - kw) / stride + 1; // 3

    let in_size = (batch * in_c * h * w) as usize;
    let kernel_size = (out_c * in_c * kh * kw) as usize;
    let out_size = (batch * out_c * out_h * out_w) as usize;

    let mut g = Graph::new();
    let input = g.input("input", &[in_size]);
    let kernel = g.parameter("kernel", &[kernel_size]);
    let output = g.conv2d(
        input, kernel, batch, in_c, h, w, out_c, kh, kw, stride, padding,
    );
    g.set_outputs(vec![output]);

    let mut session = build_inference_session(&g);
    session.set_input("input", &vec![1.0f32; in_size]);
    session.set_parameter("kernel", &vec![1.0f32; kernel_size]);

    session.step();
    session.wait();

    let out = session.read_output(out_size);
    assert_eq!(out.len(), out_size);
    // Center pixel: full 3×3 overlap → 9.0
    assert!(
        (out[4] - 9.0).abs() < 1e-4,
        "center = {}, expected 9.0",
        out[4]
    );
    // Corner pixel: only 2×2 overlap → 4.0
    assert!(
        (out[0] - 4.0).abs() < 1e-4,
        "corner = {}, expected 4.0",
        out[0]
    );
    // Edge pixel: 2×3 or 3×2 overlap → 6.0
    assert!(
        (out[1] - 6.0).abs() < 1e-4,
        "edge = {}, expected 6.0",
        out[1]
    );
}

/// Smoke test: Conv2d backward (gradient check via finite differences).
#[test]
fn conv2d_backward_smoke() {
    use meganeura::autodiff;

    let batch = 1u32;
    let in_c = 1u32;
    let h = 4u32;
    let w = 4u32;
    let out_c = 1u32;
    let kh = 3u32;
    let kw = 3u32;
    let stride = 1u32;
    let padding = 0u32;

    let in_size = (batch * in_c * h * w) as usize;
    let kernel_size = (out_c * in_c * kh * kw) as usize;
    // Forward + loss = sum_all(conv2d(input, kernel))
    let mut g = Graph::new();
    let input = g.input("input", &[in_size]);
    let kernel_node = g.parameter("kernel", &[kernel_size]);
    let conv_out = g.conv2d(
        input,
        kernel_node,
        batch,
        in_c,
        h,
        w,
        out_c,
        kh,
        kw,
        stride,
        padding,
    );
    let loss = g.sum_all(conv_out);
    g.set_outputs(vec![loss]);

    // Differentiate
    let diff_graph = autodiff::differentiate(&g);
    let plan = meganeura::compile::compile(&diff_graph);
    let mut session = meganeura::Session::new(plan);

    let input_data: Vec<f32> = (0..in_size).map(|i| (i as f32 + 1.0) * 0.1).collect();
    let kernel_data: Vec<f32> = (0..kernel_size).map(|i| (i as f32 + 1.0) * 0.05).collect();

    session.set_input("input", &input_data);
    session.set_parameter("kernel", &kernel_data);

    session.step();
    session.wait();

    // Read loss
    let loss_val = session.read_output(1);
    assert!(loss_val[0].abs() > 0.0, "loss should be non-zero");

    // Read kernel gradient (output index 1 = first param gradient)
    let mut grad_kernel = vec![0.0f32; kernel_size];
    session.read_output_by_index(1, &mut grad_kernel);

    // Finite difference check for kernel gradient
    let eps = 1e-3;
    for idx in 0..kernel_size.min(4) {
        let mut k_plus = kernel_data.clone();
        k_plus[idx] += eps;

        // Re-run forward with perturbed kernel
        let mut g2 = Graph::new();
        let inp2 = g2.input("input", &[in_size]);
        let kern2 = g2.parameter("kernel", &[kernel_size]);
        let conv2 = g2.conv2d(
            inp2, kern2, batch, in_c, h, w, out_c, kh, kw, stride, padding,
        );
        let loss2 = g2.sum_all(conv2);
        g2.set_outputs(vec![loss2]);

        let mut s2 = build_inference_session(&g2);
        s2.set_input("input", &input_data);
        s2.set_parameter("kernel", &k_plus);
        s2.step();
        s2.wait();
        let loss_plus = s2.read_output(1)[0];

        let numerical = (loss_plus - loss_val[0]) / eps;
        let analytical = grad_kernel[idx];
        let diff = (numerical - analytical).abs();
        assert!(
            diff < 0.1,
            "kernel grad[{}]: numerical={:.4}, analytical={:.4}, diff={:.4}",
            idx,
            numerical,
            analytical,
            diff
        );
    }
}
