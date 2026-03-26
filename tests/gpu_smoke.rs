/// GPU smoke test: validates that all shaders compile with blade + lavapipe
/// and that a simple forward pass executes without errors.
use meganeura::{
    Graph, build_inference_session, build_session,
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
    // Smoke test: SmolVLA action expert training graph compiles, runs,
    // and decreases loss over 5 gradient steps.
    let config = SmolVLAConfig::small_test();
    let action_seq_len = config.chunk_size; // 4
    let vlm_seq_len = 4;

    let training_g =
        smolvla::build_action_expert_training(&config, action_seq_len, vlm_seq_len);
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
        assert!(l.is_finite(), "loss diverged to NaN/inf during training: {}", l);
    }

    let final_loss = session.read_loss();
    assert!(
        final_loss < initial_loss,
        "loss should decrease after 5 gradient steps: initial={:.6}, final={:.6}",
        initial_loss,
        final_loss
    );
}
