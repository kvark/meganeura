/// Training correctness tests: verify that each model's training graph
/// compiles, runs forward+backward, and loss decreases over SGD steps.
use meganeura::{Graph, build_session};

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
        let target = ((i + 1) % vocab) as usize;
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
