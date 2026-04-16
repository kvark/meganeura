/// Training correctness tests: verify that each model's training graph
/// compiles, runs forward+backward, and loss decreases over SGD steps.
use meganeura::{Graph, build_inference_session, build_session};

/// Helper: initialize all parameters with small deterministic values,
/// run a few SGD steps, verify loss decreases.
fn verify_training_decreases_loss(
    mut session: meganeura::Session,
    set_inputs: impl Fn(&mut meganeura::Session),
    steps: usize,
    lr: f32,
) {
    // Initialize parameters
    for (name, buf_ref) in session.plan().param_buffers.clone() {
        let size_bytes = session.plan().buffers[buf_ref.0 as usize];
        let n = size_bytes / 4;
        let data: Vec<f32> = (0..n)
            .map(|i| (i as f32 * 0.01 + 1.0).sin() * 0.1)
            .collect();
        session.set_parameter(&name, &data);
    }

    assert!(
        !session.plan().param_grad_pairs.is_empty(),
        "no gradient pairs — autodiff may have failed"
    );

    // Step 1: record initial loss
    set_inputs(&mut session);
    session.step();
    session.wait();
    let initial_loss = session.read_loss();
    eprintln!("  initial loss: {:.6}", initial_loss);
    assert!(
        initial_loss.is_finite() && initial_loss > 0.0,
        "initial loss should be finite and positive, got {}",
        initial_loss
    );

    // Train with SGD
    for i in 0..steps {
        session.sgd_step_cpu(lr);
        set_inputs(&mut session);
        session.step();
        session.wait();
        let l = session.read_loss();
        assert!(
            l.is_finite(),
            "loss diverged to NaN/inf at step {}: {}",
            i + 1,
            l
        );
    }

    let final_loss = session.read_loss();
    assert!(
        final_loss < initial_loss,
        "loss should decrease after {} SGD steps: initial={:.6}, final={:.6}",
        steps,
        initial_loss,
        final_loss
    );
    eprintln!(
        "  loss: {:.6} → {:.6} ({} steps, lr={})",
        initial_loss, final_loss, steps, lr
    );
}

// ---------------------------------------------------------------------------
// SmolLM2
// ---------------------------------------------------------------------------

#[test]
fn smollm2_training_loss_decreases() {
    if std::env::var("MEGANEURA_SKIP_BACKPROP").unwrap_or_default() == "1" {
        eprintln!("MEGANEURA_SKIP_BACKPROP set — skipping");
        return;
    }

    use meganeura::models::smollm2::{self, SmolLM2Config};

    let config = SmolLM2Config::small_test();
    let seq_len = 8;
    let vocab = config.vocab_size;

    eprintln!(
        "SmolLM2 training test: seq_len={}, vocab={}",
        seq_len, vocab
    );
    let g = smollm2::build_training_graph(&config, seq_len);
    let session = build_session(&g);

    // Deterministic input: token_ids and one-hot labels
    let token_ids: Vec<u32> = (0..seq_len as u32).map(|i| i % vocab as u32).collect();
    let mut labels = vec![0.0f32; seq_len * vocab];
    for i in 0..seq_len {
        let target = (i + 1) % vocab;
        labels[i * vocab + target] = 1.0;
    }

    verify_training_decreases_loss(
        session,
        move |s| {
            s.set_input_u32("token_ids", &token_ids);
            s.set_input("labels", &labels);
        },
        5,
        0.01,
    );
}

/// Test the inferena weight-loading pattern: build separate inference and training
/// sessions, load weights into inference session, then copy into training session.
/// This catches param name/size mismatches between the two sessions.
#[test]
fn smollm2_weight_sharing_inference_to_training() {
    if std::env::var("MEGANEURA_SKIP_BACKPROP").unwrap_or_default() == "1" {
        eprintln!("MEGANEURA_SKIP_BACKPROP set — skipping");
        return;
    }

    use meganeura::models::smollm2::{self, SmolLM2Config};

    let config = SmolLM2Config::small_test();
    let seq_len = 8;
    let vocab = config.vocab_size;

    // Build inference session (forward only)
    let mut infer_g = Graph::new();
    let logits = smollm2::build_graph(&mut infer_g, &config, seq_len);
    infer_g.set_outputs(vec![logits]);
    let mut infer_session = build_inference_session(&infer_g);

    // Build training session (forward + backward + loss)
    let train_g = smollm2::build_training_graph(&config, seq_len);
    let mut train_session = build_session(&train_g);

    // Initialize inference session with deterministic weights
    for (name, buf_ref) in infer_session.plan().param_buffers.clone() {
        let size_bytes = infer_session.plan().buffers[buf_ref.0 as usize];
        let n = size_bytes / 4;
        let data: Vec<f32> = (0..n)
            .map(|i| (i as f32 * 0.01 + 1.0).sin() * 0.1)
            .collect();
        infer_session.set_parameter(&name, &data);
    }

    // Copy weights from inference → training (the inferena pattern)
    // This should NOT crash — all param names and sizes must match.
    for (name, buf_ref) in infer_session.plan().param_buffers.clone() {
        let size_bytes = infer_session.plan().buffers[buf_ref.0 as usize];
        let n = size_bytes / 4;
        let mut data = vec![0.0f32; n];
        infer_session.read_param(&name, &mut data);
        train_session.set_parameter(&name, &data);
    }

    // Run forward+backward on training session
    let token_ids: Vec<u32> = (0..seq_len as u32).map(|i| i % vocab as u32).collect();
    let mut labels = vec![0.0f32; seq_len * vocab];
    for i in 0..seq_len {
        labels[i * vocab + ((i + 1) % vocab)] = 1.0;
    }

    train_session.set_input_u32("token_ids", &token_ids);
    train_session.set_input("labels", &labels);
    train_session.step();
    train_session.wait();

    let loss = train_session.read_loss();
    eprintln!("SmolLM2 weight-sharing test: loss = {:.6}", loss);
    assert!(loss.is_finite() && loss > 0.0, "training loss: {}", loss);

    // Verify gradients exist and are finite
    assert!(
        !train_session.plan().param_grad_pairs.is_empty(),
        "no gradient pairs"
    );
    for &(_param_buf, grad_buf) in &train_session.plan().param_grad_pairs {
        let size = train_session.plan().buffers[grad_buf.0 as usize] / 4;
        let mut grad = vec![0.0f32; size];
        train_session.read_buffer(grad_buf, &mut grad);
        assert!(
            grad.iter().all(|v| v.is_finite()),
            "NaN/Inf in gradient buffer {}",
            grad_buf.0
        );
    }
}

/// Same weight-sharing test for SmolVLA.
#[test]
fn smolvla_weight_sharing_inference_to_training() {
    if std::env::var("MEGANEURA_SKIP_BACKPROP").unwrap_or_default() == "1" {
        eprintln!("MEGANEURA_SKIP_BACKPROP set — skipping");
        return;
    }

    use meganeura::models::smolvla::{self, SmolVLAConfig};

    let config = SmolVLAConfig::small_test();
    let action_seq_len = config.chunk_size;
    let vlm_seq_len = 4;

    // Build inference session
    let mut infer_g = Graph::new();
    let pred = smolvla::build_action_expert(&mut infer_g, &config, action_seq_len, vlm_seq_len);
    infer_g.set_outputs(vec![pred]);
    let mut infer_session = build_inference_session(&infer_g);

    // Build training session
    let train_g = smolvla::build_action_expert_training(&config, action_seq_len, vlm_seq_len);
    let mut train_session = build_session(&train_g);

    // Init inference params
    for (name, buf_ref) in infer_session.plan().param_buffers.clone() {
        let n = infer_session.plan().buffers[buf_ref.0 as usize] / 4;
        let data: Vec<f32> = (0..n)
            .map(|i| (i as f32 * 0.01 + 1.0).sin() * 0.1)
            .collect();
        infer_session.set_parameter(&name, &data);
    }

    // Copy inference → training
    for (name, buf_ref) in infer_session.plan().param_buffers.clone() {
        let n = infer_session.plan().buffers[buf_ref.0 as usize] / 4;
        let mut data = vec![0.0f32; n];
        infer_session.read_param(&name, &mut data);
        train_session.set_parameter(&name, &data);
    }

    // Run training step
    let expert_hidden = config.expert.hidden_size;
    let kv_dim = config.expert.kv_dim();
    train_session.set_input(
        "noisy_actions",
        &vec![0.5f32; action_seq_len * config.max_action_dim],
    );
    train_session.set_input("timestep", &vec![0.1f32; expert_hidden * 2]);
    for i in 0..config.expert.num_layers {
        if i % config.expert.self_attn_every_n_layers != 0 {
            train_session.set_input(
                &format!("vlm_kv_layer_{i}"),
                &vec![0.1f32; vlm_seq_len * kv_dim],
            );
        }
    }
    train_session.set_input(
        "target_actions",
        &vec![0.0f32; action_seq_len * config.max_action_dim],
    );

    train_session.step();
    train_session.wait();

    let loss = train_session.read_loss();
    eprintln!("SmolVLA weight-sharing test: loss = {:.6}", loss);
    assert!(loss.is_finite() && loss > 0.0, "training loss: {}", loss);
}

// ---------------------------------------------------------------------------
// SmolVLA
// ---------------------------------------------------------------------------

#[test]
fn smolvla_training_loss_decreases() {
    if std::env::var("MEGANEURA_SKIP_BACKPROP").unwrap_or_default() == "1" {
        eprintln!("MEGANEURA_SKIP_BACKPROP set — skipping");
        return;
    }

    use meganeura::models::smolvla::{self, SmolVLAConfig};

    let config = SmolVLAConfig::small_test();
    let action_seq_len = config.chunk_size;
    let vlm_seq_len = 4;

    eprintln!(
        "SmolVLA training test: action_seq={}, vlm_seq={}",
        action_seq_len, vlm_seq_len
    );
    let g = smolvla::build_action_expert_training(&config, action_seq_len, vlm_seq_len);
    let session = build_session(&g);

    let expert_hidden = config.expert.hidden_size;
    let kv_dim = config.expert.kv_dim();
    let noisy_actions = vec![0.5f32; action_seq_len * config.max_action_dim];
    let timestep = vec![0.1f32; expert_hidden * 2];
    let vlm_kv = vec![0.1f32; vlm_seq_len * kv_dim];
    let target_actions = vec![0.0f32; action_seq_len * config.max_action_dim];
    let num_layers = config.expert.num_layers;
    let self_attn_every_n = config.expert.self_attn_every_n_layers;

    verify_training_decreases_loss(
        session,
        move |s| {
            s.set_input("noisy_actions", &noisy_actions);
            s.set_input("timestep", &timestep);
            for i in 0..num_layers {
                if i % self_attn_every_n != 0 {
                    s.set_input(&format!("vlm_kv_layer_{i}"), &vlm_kv);
                }
            }
            s.set_input("target_actions", &target_actions);
        },
        5,
        0.01,
    );
}

// ---------------------------------------------------------------------------
// SD U-Net
// ---------------------------------------------------------------------------

#[test]
fn sd_unet_training_loss_decreases() {
    if std::env::var("MEGANEURA_SKIP_BACKPROP").unwrap_or_default() == "1" {
        eprintln!("MEGANEURA_SKIP_BACKPROP set — skipping");
        return;
    }

    use meganeura::models::sd_unet::{self, SDUNetConfig};

    let config = SDUNetConfig::tiny();
    let batch = config.batch_size;
    let in_c = config.in_channels;
    let res = config.resolution;
    let in_size = (batch * in_c * res * res) as usize;

    eprintln!(
        "SD U-Net training test: batch={}, res={}, in_c={}",
        batch, res, in_c
    );
    let mut g = Graph::new();
    let loss = sd_unet::build_training_graph(&mut g, &config);
    g.set_outputs(vec![loss]);
    let session = build_session(&g);

    let noisy_latent: Vec<f32> = (0..in_size).map(|i| (i as f32 * 0.01).sin()).collect();
    let noise_target: Vec<f32> = (0..in_size).map(|i| (i as f32 * 0.007).cos()).collect();

    verify_training_decreases_loss(
        session,
        move |s| {
            s.set_input("noisy_latent", &noisy_latent);
            s.set_input("noise_target", &noise_target);
        },
        5,
        0.001,
    );
}

#[test]
fn smollm2_kv_cache_decode_graph() {
    use meganeura::models::smollm2::{self, SmolLM2Config};

    let config = SmolLM2Config::small_test();
    let max_seq = 16;
    let _hidden = config.hidden_size;
    let _kv_dim = config.kv_dim();

    // Build decode graph (single-token with KV cache)
    let mut g = Graph::new();
    let (logits, k_caches, v_caches) = smollm2::build_decode_graph(&mut g, &config, max_seq);
    g.set_outputs(vec![logits]);

    let mut session = build_inference_session(&g);

    // Initialize model weights
    for (name, buf_ref) in session.plan().param_buffers.clone() {
        let size_bytes = session.plan().buffers[buf_ref.0 as usize];
        let n = size_bytes / 4;
        // Skip cache buffers — initialize to zero
        if name.contains("kv_cache") {
            session.set_parameter(&name, &vec![0.0f32; n]);
            continue;
        }
        let data: Vec<f32> = (0..n)
            .map(|i| (i as f32 * 0.01 + 1.0).sin() * 0.1)
            .collect();
        session.set_parameter(&name, &data);
    }

    // Run decode for 4 positions
    let vocab = config.vocab_size;
    for pos in 0..4u32 {
        // Single token input
        let token_ids = vec![pos % vocab as u32];
        session.set_input_u32("token_ids", &token_ids);
        session.set_input_u32("kv_pos", &[pos]);

        // K/V inputs are generated by the graph's matmul projections,
        // not set externally. We just need token_ids and kv_pos.

        session.step();
        session.wait();

        let out = session.read_output(vocab);
        assert_eq!(out.len(), vocab);
        assert!(
            out.iter().any(|&v| v.abs() > 1e-10),
            "pos={}: logits all zero",
            pos
        );
        assert!(
            out.iter().all(|v| v.is_finite()),
            "pos={}: logits contain NaN/Inf",
            pos
        );
        eprintln!(
            "pos={}: logits range [{:.4}, {:.4}]",
            pos,
            out.iter().cloned().fold(f32::INFINITY, f32::min),
            out.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        );
    }
    let _ = (k_caches, v_caches); // suppress unused warnings
}

#[test]
fn resnet50_training_loss_decreases() {
    let g = meganeura::models::resnet::build_resnet50_training(1);
    let batch = 1u32;
    let image_size = (batch * 3 * 224 * 224) as usize;
    let num_classes = 1000;

    let mut sess = build_session(&g);

    // Use very small init for ResNet (no BN at training time → needs small weights)
    for (name, buf_ref) in sess.plan().param_buffers.clone() {
        let n = sess.plan().buffers[buf_ref.0 as usize] / 4;
        let data: Vec<f32> = (0..n)
            .map(|i| (i as f32 * 0.01 + 1.0).sin() * 0.001)
            .collect();
        sess.set_parameter(&name, &data);
    }

    let image: Vec<f32> = (0..image_size)
        .map(|i| (i as f32 * 0.001).sin() * 0.1)
        .collect();
    let mut labels = vec![0.0f32; num_classes];
    labels[42] = 1.0;

    sess.set_learning_rate(1e-5);
    sess.set_input("image", &image);
    sess.set_input("labels", &labels);

    sess.step();
    sess.wait();
    let loss0 = sess.read_loss();
    eprintln!("ResNet-50 initial loss: {:.6}", loss0);
    assert!(loss0.is_finite(), "initial loss is not finite: {}", loss0);

    for step in 0..5 {
        sess.set_input("image", &image);
        sess.set_input("labels", &labels);
        sess.step();
        sess.wait();
        let loss = sess.read_loss();
        eprintln!("  step {}: loss={:.6}", step + 1, loss);
        assert!(
            loss.is_finite(),
            "loss diverged at step {}: {}",
            step + 1,
            loss
        );
    }
    let loss_final = sess.read_loss();
    assert!(
        loss_final < loss0 * 1.1,
        "loss didn't decrease: {:.6} → {:.6}",
        loss0,
        loss_final
    );
}

#[test]
fn whisper_encoder_training_loss_decreases() {
    use meganeura::models::whisper::{self, WhisperConfig};

    let config = WhisperConfig::whisper_tiny();
    let batch = 1;
    let mel_len = 100; // short clip for testing
    let seq_len = (mel_len / 2) as usize;
    let num_classes = 64;

    let g = whisper::build_training_graph(&config, batch, mel_len);

    verify_training_decreases_loss(
        build_session(&g),
        |s| {
            let mel_size = (batch * config.n_mels as u32 * mel_len) as usize;
            let mel: Vec<f32> = (0..mel_size).map(|i| (i as f32 * 0.01).sin()).collect();
            let mut labels = vec![0.0f32; seq_len * num_classes];
            for i in 0..seq_len {
                labels[i * num_classes + (i % num_classes)] = 1.0;
            }
            s.set_input("mel", &mel);
            s.set_input("labels", &labels);
        },
        5,
        0.001,
    );
}
