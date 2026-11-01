/// Training correctness tests: verify that each model's training graph
/// compiles, runs forward+backward, and loss decreases over SGD steps.
use meganeura::Graph;

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

    use meganeura::models::smollm2;

    let config = smollm2::Config::small_test();
    let seq_len = 8;
    let vocab = config.vocab_size;

    eprintln!(
        "SmolLM2 training test: seq_len={}, vocab={}",
        seq_len, vocab
    );
    let g = smollm2::build_training_graph(&config, seq_len);
    let session = meganeura::build(&g, crate::support::gpu::config()).0;

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

/// The inferena weight-loading pattern: weights loaded into an inference
/// session are copied into a training session by name, so every inference
/// parameter must exist in the training plan with the same size.
/// `set_parameter` panics on an unknown name or a size mismatch. The plans
/// compile on the CPU; the training step itself is covered by the
/// `*_training_loss_decreases` tests.
#[test]
fn inference_parameters_load_into_training() {
    use meganeura::models::{smollm2, smolvla};
    use meganeura::{CompileOptions, Mode, OptimizeConfig, compile_plan};

    let parameters = |graph: &Graph, mode: Mode| -> Vec<(String, usize)> {
        let plan = compile_plan(
            graph,
            mode,
            OptimizeConfig::default(),
            &CompileOptions::default(),
        );
        plan.param_buffers
            .iter()
            .map(|(name, buf)| (name.clone(), plan.buffers[buf.0 as usize]))
            .collect()
    };

    let lm = smollm2::Config::small_test();
    let mut lm_inference = Graph::new();
    let logits = smollm2::build_graph(&mut lm_inference, &lm, 8);
    lm_inference.set_outputs(vec![logits]);
    let lm_training = smollm2::build_training_graph(&lm, 8);

    let vla = smolvla::Config::small_test();
    let mut vla_inference = Graph::new();
    let pred = smolvla::build_action_expert(&mut vla_inference, &vla, vla.chunk_size, 4);
    vla_inference.set_outputs(vec![pred]);
    let vla_training = smolvla::build_action_expert_training(&vla, vla.chunk_size, 4);

    for (what, inference, training) in [
        ("SmolLM2", &lm_inference, &lm_training),
        ("SmolVLA", &vla_inference, &vla_training),
    ] {
        let training: std::collections::HashMap<_, _> =
            parameters(training, Mode::Training).into_iter().collect();
        let inference = parameters(inference, Mode::Inference);
        assert!(!inference.is_empty(), "{what}: no inference parameters");
        for (name, bytes) in inference {
            assert_eq!(
                training.get(&name),
                Some(&bytes),
                "{what}: inference parameter {name} ({bytes} bytes) does not load into training"
            );
        }
    }
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

    use meganeura::models::smolvla;

    let config = smolvla::Config::small_test();
    let action_seq_len = config.chunk_size;
    let vlm_seq_len = 4;

    eprintln!(
        "SmolVLA training test: action_seq={}, vlm_seq={}",
        action_seq_len, vlm_seq_len
    );
    let g = smolvla::build_action_expert_training(&config, action_seq_len, vlm_seq_len);
    let session = meganeura::build(&g, crate::support::gpu::config()).0;

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
// Conditioned diffusion U-Net
// ---------------------------------------------------------------------------

#[test]
fn sd_unet_training_loss_decreases() {
    if std::env::var("MEGANEURA_SKIP_BACKPROP").unwrap_or_default() == "1" {
        eprintln!("MEGANEURA_SKIP_BACKPROP set — skipping");
        return;
    }

    use meganeura::models::sd_unet;

    let config = sd_unet::Config::tiny();
    let batch = config.batch_size;
    let in_c = config.in_channels;
    let res = config.resolution;
    let in_size = (batch * in_c * res * res) as usize;

    eprintln!(
        "Conditioned diffusion U-Net training test: batch={}, res={}, in_c={}",
        batch, res, in_c
    );
    let mut g = Graph::new();
    let loss = sd_unet::build_training_graph(&mut g, &config);
    g.set_outputs(vec![loss]);
    let session = meganeura::build(&g, crate::support::gpu::config()).0;

    let noisy_latent: Vec<f32> = (0..in_size).map(|i| (i as f32 * 0.01).sin()).collect();
    let noise_target: Vec<f32> = (0..in_size).map(|i| (i as f32 * 0.007).cos()).collect();
    let timestep_embedding: Vec<f32> = (0..config.time_input_dim as usize)
        .map(|i| (i as f32 * 0.03).sin())
        .collect();
    let text_context: Vec<f32> = (0..(config.context_len * config.context_dim) as usize)
        .map(|i| (i as f32 * 0.001).cos() * 0.1)
        .collect();

    verify_training_decreases_loss(
        session,
        move |s| {
            s.set_input("noisy_latent", &noisy_latent);
            s.set_input("timestep_embedding", &timestep_embedding);
            s.set_input("text_context", &text_context);
            s.set_input("noise_target", &noise_target);
        },
        5,
        0.001,
    );
}

#[test]
#[ignore = "slow: full 224x224 ResNet-50 training; run with --include-ignored"]
fn resnet50_training_loss_decreases() {
    let g = meganeura::models::resnet::build_resnet50_training(1);
    let batch = 1u32;
    let image_size = (batch * 3 * 224 * 224) as usize;
    let num_classes = 1000;

    let mut sess = meganeura::build(&g, crate::support::gpu::config()).0;

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
    use meganeura::models::whisper;

    // One narrow layer covers the same training graph as whisper_tiny.
    let config = whisper::Config {
        d_model: 64,
        n_heads: 2,
        n_layers: 1,
        ffn_dim: 256,
        ..whisper::Config::whisper_tiny()
    };
    let batch = 1;
    let mel_len = 100; // short clip for testing

    let g = whisper::build_training_graph(&config, batch, mel_len);

    verify_training_decreases_loss(
        meganeura::build(&g, crate::support::gpu::config()).0,
        |s| {
            let mel_size = (batch * config.n_mels as u32 * mel_len) as usize;
            let mel: Vec<f32> = (0..mel_size).map(|i| (i as f32 * 0.01).sin()).collect();
            s.set_input("mel", &mel);
        },
        5,
        0.001,
    );
}
