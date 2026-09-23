//! Stack behaviour on the GPU beyond op numerics: optimizer updates,
//! checkpoints, gradient inspection, dispatch geometry at extreme sizes,
//! packed quantized weights, and model-level gradient checks. Every op's
//! values and derivatives are checked by the `oracle` suite.
use meganeura::Graph;
#[cfg(feature = "models")]
use meganeura::{build_session, build_session_unoptimized, compile::BufferRef, models::smolvla};

#[cfg(feature = "models")]
fn grad_rel_err(num: f32, ana: f32) -> f32 {
    let abs_err = (num - ana).abs();
    const ABS_TOL: f32 = 3e-6;
    if abs_err < ABS_TOL {
        0.0
    } else {
        abs_err / num.abs().max(ana.abs()).max(1e-6)
    }
}

#[test]
fn tall_matmul_splits_dispatch_across_z() {
    const ROWS: usize = 65_536 * 64;

    let mut graph = Graph::new();
    let input = graph.input("input", &[ROWS, 1]);
    let weight = graph.parameter("weight", &[1, 1]);
    let output = graph.matmul(input, weight);
    graph.set_outputs(vec![output]);

    let mut session = meganeura::build(&graph, meganeura::SessionConfig::inference_from_env()).0;
    session.set_input("input", &vec![1.25_f32; ROWS]);
    session.set_parameter("weight", &[2.0]);
    session.step();
    session.wait();

    let actual = session.read_output(ROWS);
    for index in [0, ROWS / 2, ROWS - 1] {
        assert!((actual[index] - 2.5).abs() < 1.0e-6);
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

    let mut session = meganeura::build(&g, meganeura::SessionConfig::from_env()).0;
    session.set_parameter("w", &[0.1_f32; 8 * 4]);
    session.set_input("x", &[1.0_f32; 4 * 8]);
    session.step();
    session.wait();
    let initial_loss = session.read_loss();
    assert!(initial_loss.is_finite());

    session.sgd_step_cpu(0.1);
    session.set_input("x", &[1.0_f32; 4 * 8]);
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
#[cfg(feature = "models")]
fn smolvla_training_backprop_smoke() {
    // GPU-less validation jobs may opt out of the full model backpropagation test.
    if std::env::var("MEGANEURA_SKIP_BACKPROP").unwrap_or_default() == "1" {
        eprintln!("MEGANEURA_SKIP_BACKPROP set — skipping MHA backward test");
        return;
    }
    // Smoke test: SmolVLA action expert training graph compiles, runs,
    // and decreases loss over 5 gradient steps.
    let config = smolvla::Config::small_test();
    let action_seq_len = config.chunk_size; // 4
    let vlm_seq_len = 4;

    let training_g = smolvla::build_action_expert_training(&config, action_seq_len, vlm_seq_len);
    let mut session = meganeura::build(&training_g, meganeura::SessionConfig::from_env()).0;

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

/// Check that all non-fused parameters have non-zero gradients.
/// Returns (total_params, zero_param_names).
#[cfg(feature = "models")]
fn check_ffn_gradients(session: &meganeura::Session) -> (usize, Vec<String>) {
    let param_buffers: std::collections::HashMap<String, BufferRef> =
        session.plan().param_buffers.iter().cloned().collect();
    let grad_map: std::collections::HashMap<BufferRef, BufferRef> =
        session.plan().param_grad_pairs.iter().cloned().collect();

    let mut zero_params = Vec::new();
    let mut total = 0usize;
    for (name, buf_ref) in &param_buffers {
        if let Some(&grad_buf) = grad_map.get(buf_ref) {
            let n = session.plan().buffers[buf_ref.0 as usize] / 4;
            let grad_n = session.plan().buffers[grad_buf.0 as usize] / 4;
            if n != grad_n {
                continue;
            } // skip fused/dead params
            let mut grad = vec![0.0f32; n];
            session.read_buffer(grad_buf, &mut grad);
            let norm: f64 = grad.iter().map(|&v| (v as f64).powi(2)).sum::<f64>().sqrt();
            total += 1;
            if norm < 1e-12 {
                zero_params.push(name.clone());
            }
        }
    }
    zero_params.sort();
    (total, zero_params)
}

/// Run SmolLM2 training graph with given session builder and check gradients.
#[cfg(feature = "models")]
fn run_smollm2_gradient_check(
    config: &meganeura::models::smollm2::Config,
    builder: fn(&Graph) -> meganeura::Session,
) -> (usize, Vec<String>) {
    let seq = 8;
    let g = meganeura::models::smollm2::build_training_graph(config, seq);
    let mut session = builder(&g);

    for (name, buf_ref) in session.plan().param_buffers.clone() {
        let n = session.plan().buffers[buf_ref.0 as usize] / 4;
        let data: Vec<f32> = (0..n)
            .map(|i| (i as f32 * 0.01 + 1.0).sin() * 0.1)
            .collect();
        session.set_parameter(&name, &data);
    }

    let input_ids: Vec<u32> = (0..seq as u32).collect();
    let vocab = config.vocab_size;
    let mut labels = vec![0.0f32; seq * vocab];
    for pos in 0..seq {
        labels[pos * vocab + ((pos + 1) % vocab)] = 1.0;
    }
    session.set_input_u32("token_ids", &input_ids);
    session.set_input("labels", &labels);

    session.step();
    session.wait();

    check_ffn_gradients(&session)
}

#[test]
#[cfg(feature = "models")]
fn smollm2_ffn_gradients_nonzero() {
    use meganeura::models::smollm2;
    let config = smollm2::Config::small_test();
    let (total, zeros) = run_smollm2_gradient_check(&config, build_session);
    eprintln!("optimized: {total} params, {} zero", zeros.len());
    assert!(
        zeros.is_empty(),
        "Zero gradients (optimized):\n  {}",
        zeros.join("\n  ")
    );
}

#[test]
#[cfg(feature = "models")]
fn smollm2_ffn_gradients_unoptimized() {
    use meganeura::models::smollm2;
    let config = smollm2::Config::small_test();
    let (total, zeros) = run_smollm2_gradient_check(&config, build_session_unoptimized);
    eprintln!("unoptimized: {total} params, {} zero", zeros.len());
    assert!(
        zeros.is_empty(),
        "Zero gradients (unoptimized):\n  {}",
        zeros.join("\n  ")
    );
}

#[test]
#[cfg(feature = "models")]
#[ignore] // ~22 min in debug mode; run with --release --ignored
fn smollm2_medium_ffn_gradients_optimized() {
    use meganeura::models::smollm2;
    let config = smollm2::Config::medium_test();
    let (total, zeros) = run_smollm2_gradient_check(&config, build_session);
    eprintln!("medium optimized: {total} params, {} zero", zeros.len());
    assert!(
        zeros.is_empty(),
        "Zero gradients (medium optimized):\n  {}",
        zeros.join("\n  ")
    );
}

#[test]
#[cfg(feature = "models")]
#[ignore] // ~22 min in debug mode; run with --release --ignored
fn smollm2_medium_ffn_gradients_unoptimized() {
    use meganeura::models::smollm2;
    let config = smollm2::Config::medium_test();
    let (total, zeros) = run_smollm2_gradient_check(&config, build_session_unoptimized);
    eprintln!("medium unoptimized: {total} params, {} zero", zeros.len());
    assert!(
        zeros.is_empty(),
        "Zero gradients (medium unoptimized):\n  {}",
        zeros.join("\n  ")
    );
}

/// End-to-end SmolLM2 gradient check via finite differences.
/// Uses 1 layer with head_dim=64 (matching production shaders).
#[test]
#[cfg(feature = "models")]
fn smollm2_e2e_gradient_finite_diff() {
    if std::env::var("MEGANEURA_SKIP_BACKPROP").unwrap_or_default() == "1" {
        eprintln!("MEGANEURA_SKIP_BACKPROP set — skipping SmolLM2 e2e gradient check");
        return;
    }
    use meganeura::models::smollm2;

    let num_layers = std::env::var("SMOLLM2_LAYERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let tie = std::env::var("TIE_WEIGHTS").unwrap_or_default() != "0";
    let config = smollm2::Config {
        vocab_size: 256,
        hidden_size: 576,
        num_hidden_layers: num_layers,
        num_attention_heads: 9,
        num_key_value_heads: 3,
        intermediate_size: 1536,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        tie_word_embeddings: tie,
    };
    let seq = 8;

    // --- Build training session ---
    let g = meganeura::models::smollm2::build_training_graph(&config, seq);
    let use_unopt = std::env::var("UNOPT").is_ok();
    let mut train_sess = if use_unopt {
        meganeura::build(&g, meganeura::SessionConfig::unoptimized_from_env()).0
    } else {
        meganeura::build(&g, meganeura::SessionConfig::from_env()).0
    };

    // Deterministic init
    fn name_seed(name: &str) -> f32 {
        let mut h: u32 = 0;
        for c in name.bytes() {
            h = h.wrapping_mul(31).wrapping_add(c as u32);
        }
        (h % 10000) as f32
    }
    let scale: f32 = std::env::var("WEIGHT_SCALE")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.1);
    for (name, buf_ref) in train_sess.plan().param_buffers.clone() {
        let n = train_sess.plan().buffers[buf_ref.0 as usize] / 4;
        let seed = name_seed(&name);
        let data: Vec<f32> = (0..n)
            .map(|i| (i as f32 * 0.01 + seed).sin() * scale)
            .collect();
        train_sess.set_parameter(&name, &data);
    }

    // Inputs
    let input_ids: Vec<u32> = (0..seq as u32).collect();
    let mut labels = vec![0.0f32; seq * config.vocab_size];
    for pos in 0..seq {
        labels[pos * config.vocab_size + (pos + 1) % config.vocab_size] = 1.0;
    }
    train_sess.set_input_u32("token_ids", &input_ids);
    train_sess.set_input("labels", &labels);
    train_sess.step();
    train_sess.wait();

    let train_loss = train_sess.read_loss();
    eprintln!("e2e training loss: {train_loss:.6}");

    // Read gradient for a few params
    let plan = train_sess.plan().clone();
    let param_bufs: std::collections::HashMap<String, BufferRef> =
        plan.param_buffers.iter().cloned().collect();
    let grad_map: std::collections::HashMap<BufferRef, BufferRef> =
        plan.param_grad_pairs.iter().cloned().collect();

    // --- Build inference session for finite differences ---
    let gi = meganeura::models::smollm2::build_training_graph(&config, seq);
    let mut infer_sess = meganeura::build(&gi, meganeura::SessionConfig::inference_from_env()).0;

    // Same init (must use the same scale)
    for (name, buf_ref) in infer_sess.plan().param_buffers.clone() {
        let n = infer_sess.plan().buffers[buf_ref.0 as usize] / 4;
        let seed = name_seed(&name);
        let data: Vec<f32> = (0..n)
            .map(|i| (i as f32 * 0.01 + seed).sin() * scale)
            .collect();
        infer_sess.set_parameter(&name, &data);
    }

    let fwd = |sess: &mut meganeura::Session| -> f32 {
        sess.set_input_u32("token_ids", &input_ids);
        sess.set_input("labels", &labels);
        sess.step();
        sess.wait();
        sess.read_loss()
    };

    // Verify inference loss matches
    let infer_loss = fwd(&mut infer_sess);
    eprintln!("e2e inference loss: {infer_loss:.6}");
    assert!(
        (train_loss - infer_loss).abs() / train_loss.abs().max(1e-6) < 0.01,
        "train loss {train_loss} != infer loss {infer_loss}"
    );

    // Check gradient of a few parameters via finite differences
    let eps = 1e-3f32;
    let mut max_rel = 0.0f32;
    let test_params = [
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.self_attn.q_proj.weight",
        "model.layers.0.self_attn.v_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.embed_tokens.weight",
    ];

    for param_name in test_params {
        let buf = match param_bufs.get(param_name) {
            Some(b) => *b,
            None => {
                eprintln!("  skipping {param_name} (not found)");
                continue;
            }
        };
        let grad_buf = match grad_map.get(&buf) {
            Some(g) => *g,
            None => {
                eprintln!("  skipping {param_name} (no gradient)");
                continue;
            }
        };
        let n = plan.buffers[buf.0 as usize] / 4;
        let grad_n = plan.buffers[grad_buf.0 as usize] / 4;
        if n != grad_n {
            eprintln!("  skipping {param_name} (fused, n={n} grad_n={grad_n})");
            continue;
        }

        let mut grad = vec![0.0f32; n];
        train_sess.read_buffer(grad_buf, &mut grad);

        // Check a few indices — only where gradient is large enough for finite diff
        let seed = name_seed(param_name);
        let orig_data: Vec<f32> = (0..n)
            .map(|i| (i as f32 * 0.01 + seed).sin() * scale)
            .collect();

        let check_indices: Vec<usize> = [0, 1, n / 4, n / 2, n - 1]
            .iter()
            .filter(|&&i| i < n)
            .copied()
            .collect();

        for &idx in &check_indices {
            let mut perturbed = orig_data.clone();
            perturbed[idx] += eps;
            infer_sess.set_parameter(param_name, &perturbed);
            let lp = fwd(&mut infer_sess);
            perturbed[idx] -= 2.0 * eps;
            infer_sess.set_parameter(param_name, &perturbed);
            let lm = fwd(&mut infer_sess);
            // Restore
            infer_sess.set_parameter(param_name, &orig_data);

            let num = (lp - lm) / (2.0 * eps);
            let ana = grad[idx];
            let rel = grad_rel_err(num, ana);
            // Only count errors where BOTH values are above f32 noise floor.
            // For large models with small weights, many gradient elements are too small
            // for finite differences to detect (loss change < f32 epsilon).
            let significant = num.abs() > 1e-3 && ana.abs() > 1e-3;
            if rel > 0.1 || idx == 0 {
                let tag = if significant { "" } else { " [below noise]" };
                eprintln!("  {param_name}[{idx}]: ana={ana:.6e} num={num:.6e} rel={rel:.4}{tag}");
            }
            if significant {
                max_rel = max_rel.max(rel);
            }
        }
    }

    eprintln!("SmolLM2 e2e gradient check: max relative error {max_rel:.6}");

    // Print all per-param gradient norms for comparison with PyTorch
    let mut param_norms: Vec<(String, f64)> = Vec::new();
    let mut total_sq = 0.0f64;
    for (name, buf_ref) in plan.param_buffers.iter() {
        if let Some(&(_, g)) = plan.param_grad_pairs.iter().find(|&&(p, _)| p == *buf_ref) {
            let n = plan.buffers[buf_ref.0 as usize] / 4;
            let gn = plan.buffers[g.0 as usize] / 4;
            if n != gn {
                continue;
            }
            let mut grad = vec![0.0f32; n];
            train_sess.read_buffer(g, &mut grad);
            let sq: f64 = grad.iter().map(|&v| (v as f64).powi(2)).sum();
            total_sq += sq;
            param_norms.push((name.clone(), sq.sqrt()));
        }
    }
    param_norms.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    eprintln!(
        "Meganeura grad_norm: {:.6} ({} params)",
        total_sq.sqrt(),
        param_norms.len()
    );
    for (name, norm) in &param_norms {
        eprintln!("  {name}: {norm:.6e}");
    }

    // Print specific gradient values
    for pname in [
        "model.embed_tokens.weight",
        "model.layers.0.input_layernorm.weight",
    ] {
        if let Some(buf) = param_bufs.get(pname)
            && let Some(&(_, g)) = plan.param_grad_pairs.iter().find(|&&(p, _)| p == *buf)
        {
            let n = plan.buffers[g.0 as usize] / 4;
            let mut grad = vec![0.0f32; n];
            train_sess.read_buffer(g, &mut grad);
            eprintln!("  {pname} grad[0..3]: {:?}", &grad[..3.min(n)]);
        }
    }

    assert!(
        max_rel < 0.25,
        "SmolLM2 e2e gradient error too large: {max_rel}"
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

    let mut session = meganeura::build(&g, meganeura::SessionConfig::from_env()).0;
    session.set_parameter("w", &[0.1_f32; 8 * 4]);
    session.set_input("x", &[1.0_f32; 4 * 8]);

    // Train 3 steps with Adam
    for _ in 0..3 {
        session.set_input("x", &[1.0_f32; 4 * 8]);
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
    let mut session2 = meganeura::build(&g, meganeura::SessionConfig::from_env()).0;
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
    session2.set_input("x", &[1.0_f32; 4 * 8]);
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

#[test]
fn batched_parameter_read_matches_uploaded_values() {
    let mut graph = Graph::new();
    let a = graph.parameter("a", &[2, 3]);
    let b = graph.parameter("b", &[5]);
    let a_mean = graph.mean_all(a);
    let b_mean = graph.mean_all(b);
    let output = graph.add(a_mean, b_mean);
    graph.set_outputs(vec![output]);

    let a_values = [0.25_f32, -1.0, 2.5, 7.0, -3.25, 0.125];
    let b_values = [4.0_f32, 3.0, 2.0, 1.0, -5.0];
    let mut session = meganeura::build(&graph, meganeura::SessionConfig::inference_from_env()).0;
    session.set_parameter("a", &a_values);
    session.set_parameter("b", &b_values);

    let values = session.read_params(&["b", "a"]);
    assert_eq!(values, [b_values.to_vec(), a_values.to_vec()]);
    let a = session.param_buffer("a").unwrap();
    let b = session.param_buffer("b").unwrap();
    assert_eq!(
        session.read_buffers(&[b, a, b]),
        [b_values.to_vec(), a_values.to_vec(), b_values.to_vec()]
    );
    assert!(session.read_buffers(&[]).is_empty());
}

#[test]
fn checkpoint_round_trip_preserves_odd_f16_tail() {
    let mut g = Graph::new();
    let x = g.input("x", &[1, 3]);
    let w = g.parameter_f16("w", &[3, 1]);
    let y = g.matmul(x, w);
    g.set_outputs(vec![y]);

    let values = [0.25_f32, -0.5, 0.75];
    let input = [1.0_f32, 2.0, 3.0];
    let tmp = std::env::temp_dir().join("meganeura_test_odd_f16_ckpt.safetensors");
    let mut session = meganeura::build(&g, meganeura::SessionConfig::inference_from_env()).0;
    session.set_parameter("w", &values);
    session.save_checkpoint(&tmp).expect("save checkpoint");

    let mut restored = meganeura::build(&g, meganeura::SessionConfig::inference_from_env()).0;
    restored.load_checkpoint(&tmp).expect("load checkpoint");
    restored.set_input("x", &input);
    restored.step();
    restored.wait();

    let actual = restored.read_output(1)[0];
    let expected = values
        .iter()
        .zip(input)
        .map(|(&weight, value)| weight * value)
        .sum::<f32>();
    assert!((actual - expected).abs() < 1.0e-3, "{actual} != {expected}");
    std::fs::remove_file(tmp).unwrap();
}

/// Verify the bulk gradient-inspection API: param_names enumerates,
/// has_param_grad reports correctly, read_all_param_grad_norms returns
/// finite norms in compile order matching individual read_param_grad.
#[test]
fn grad_inspection_api_basic() {
    use meganeura::Graph;
    let mut g = Graph::new();
    let x = g.input("x", &[2, 3]);
    let w = g.parameter("weights", &[3, 2]);
    let b = g.parameter("bias", &[2]);
    let h = g.matmul(x, w);
    let h = g.bias_add(h, b);
    let target = g.input("target", &[2, 2]);
    let loss = g.mse_loss(h, target);
    g.set_outputs(vec![loss]);
    let mut session = meganeura::build(&g, meganeura::SessionConfig::from_env()).0;

    session.set_input("x", &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    session.set_input("target", &[0.5, 0.5, 0.5, 0.5]);
    session.set_parameter("weights", &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
    session.set_parameter("bias", &[0.1, -0.1]);
    session.step();
    session.wait();

    // param_names enumerates both params.
    let names = session.param_names();
    assert!(names.contains(&"weights"), "weights in param_names");
    assert!(names.contains(&"bias"), "bias in param_names");

    // has_param_grad true for both since loss flows through both.
    assert!(session.has_param_grad("weights"), "weights has grad");
    assert!(session.has_param_grad("bias"), "bias has grad");
    assert!(!session.has_param_grad("nonexistent"), "missing → false");

    // param_size matches buffer size in f32 elements.
    assert_eq!(session.param_size("weights"), Some(6));
    assert_eq!(session.param_size("bias"), Some(2));
    assert_eq!(session.param_size("missing"), None);

    // Bulk grad norms should agree with individual read_param_grad
    // computations (modulo float rounding).
    let bulk = session.read_all_param_grad_norms();
    assert_eq!(bulk.len(), 2, "two params with grads");
    for (name, bulk_norm) in &bulk {
        let n = session.param_size(name).unwrap();
        let mut grad = vec![0.0f32; n];
        session.read_param_grad(name, &mut grad);
        let manual: f32 = grad.iter().map(|&v| v * v).sum::<f32>().sqrt();
        let diff = (bulk_norm - manual).abs();
        assert!(
            diff < 1e-5 || diff < manual * 1e-4,
            "bulk norm for {} ({}) should match manual ({})",
            name,
            bulk_norm,
            manual
        );
        assert!(bulk_norm.is_finite(), "norm finite for {}", name);
    }

    // Bulk weight norms cover all params (including any without grads).
    let weights = session.read_all_param_norms();
    assert_eq!(weights.len(), 2);
    for (name, n) in &weights {
        assert!(n.is_finite() && *n > 0.0, "{} weight norm > 0", name);
    }

    // dump_grad_summary doesn't panic.
    session.dump_grad_summary(2);
}

/// Verify per-parameter LR multipliers actually scale the SGD update.
/// Two parameters; multiplier on one of them; after one SGD step the
/// scaled param should have moved that-many-times-more.
#[test]
fn lr_multipliers_apply_to_sgd_update() {
    use meganeura::Graph;
    let mut g = Graph::new();
    let x = g.input("x", &[2, 2]);
    let w_a = g.parameter("a.weight", &[2, 2]);
    let w_b = g.parameter("b.weight", &[2, 2]);
    let h = g.matmul(x, w_a);
    let h = g.matmul(h, w_b);
    let target = g.input("target", &[2, 2]);
    let loss = g.mse_loss(h, target);
    g.set_outputs(vec![loss]);
    let mut session = meganeura::build(&g, meganeura::SessionConfig::from_env()).0;

    let init = vec![1.0, 0.1, 0.1, 1.0];

    // Pass 1: default LR (no multipliers). Observe per-param updates.
    session.set_parameter("a.weight", &init);
    session.set_parameter("b.weight", &init);
    session.set_input("x", &[1.0, 0.0, 0.0, 1.0]);
    session.set_input("target", &[0.0, 0.0, 0.0, 0.0]);
    let base_lr = 0.01;
    session.set_learning_rate(base_lr);
    session.step();
    session.wait();

    let mut a_after = vec![0.0; 4];
    let mut b_after = vec![0.0; 4];
    session.read_param("a.weight", &mut a_after);
    session.read_param("b.weight", &mut b_after);
    let delta_a_default: f32 = a_after
        .iter()
        .zip(init.iter())
        .map(|(p, i)| (p - i).abs())
        .sum();
    let delta_b_default: f32 = b_after
        .iter()
        .zip(init.iter())
        .map(|(p, i)| (p - i).abs())
        .sum();

    // Pass 2: reset, apply 5x multiplier to b.weight only, observe.
    session.set_parameter("a.weight", &init);
    session.set_parameter("b.weight", &init);
    session.set_lr_multiplier("b.", 5.0);
    session.set_learning_rate(base_lr);
    session.step();
    session.wait();

    let mut a_after2 = vec![0.0; 4];
    let mut b_after2 = vec![0.0; 4];
    session.read_param("a.weight", &mut a_after2);
    session.read_param("b.weight", &mut b_after2);
    let delta_a_scaled: f32 = a_after2
        .iter()
        .zip(init.iter())
        .map(|(p, i)| (p - i).abs())
        .sum();
    let delta_b_scaled: f32 = b_after2
        .iter()
        .zip(init.iter())
        .map(|(p, i)| (p - i).abs())
        .sum();

    // a.weight's update should be unchanged (no multiplier applies).
    let a_diff = (delta_a_scaled - delta_a_default).abs();
    assert!(
        a_diff < delta_a_default * 0.05 + 1e-6,
        "a.weight update should be unaffected: default={} scaled={}",
        delta_a_default,
        delta_a_scaled
    );
    // b.weight's update should be ~5x its default.
    let b_ratio = delta_b_scaled / delta_b_default;
    assert!(
        (b_ratio - 5.0).abs() < 0.5,
        "b.weight update should scale by ~5x: default={} scaled={} ratio={}",
        delta_b_default,
        delta_b_scaled,
        b_ratio
    );

    // Pass 3: clear multipliers, verify both params back to default.
    session.set_parameter("a.weight", &init);
    session.set_parameter("b.weight", &init);
    session.clear_lr_multipliers();
    session.set_learning_rate(base_lr);
    session.step();
    session.wait();
    let mut b_after3 = vec![0.0; 4];
    session.read_param("b.weight", &mut b_after3);
    let delta_b_cleared: f32 = b_after3
        .iter()
        .zip(init.iter())
        .map(|(p, i)| (p - i).abs())
        .sum();
    assert!(
        (delta_b_cleared - delta_b_default).abs() < delta_b_default * 0.05 + 1e-6,
        "b.weight update should return to default after clear: default={} cleared={}",
        delta_b_default,
        delta_b_cleared
    );
}

#[test]
fn full_precision_weight_gradient_preserves_tiny_values() {
    // This shape reaches the f16 cooperative-matrix promotion threshold on
    // NVIDIA (16 * 8 = 128 output workgroups). The loss scale makes every
    // dL/dY element smaller than f16's minimum subnormal, while the f32
    // A^T*dY result remains representable. Routing this derivative matmul
    // through even compensated f16 therefore turns the whole weight gradient
    // into zero.
    // Keep the elementwise output dispatch below Vulkan's 65,535-workgroup
    // per-axis limit while retaining the same weight-gradient tile geometry.
    const ROWS: usize = 1_023;
    const INPUTS: usize = 512;
    const OUTPUTS: usize = 256;
    const LOSS_SCALE: f32 = 1.0e-6;

    let mut graph = Graph::new();
    let input = graph.input("input", &[ROWS, INPUTS]);
    let weight = graph.parameter("weight", &[INPUTS, OUTPUTS]);
    let prediction = graph.matmul(input, weight);
    let mean = graph.mean_all(prediction);
    let scale = graph.scalar(LOSS_SCALE);
    let loss = graph.mul(mean, scale);
    graph.set_outputs(vec![loss]);

    let mut session = meganeura::build(&graph, meganeura::SessionConfig::from_env()).0;
    session.set_input("input", &vec![1.0; ROWS * INPUTS]);
    session.set_parameter("weight", &vec![0.0; INPUTS * OUTPUTS]);
    session.step();
    session.wait();

    let plan = session.plan();
    let parameter = plan
        .param_buffers
        .iter()
        .find_map(|(name, buffer)| (name == "weight").then_some(*buffer))
        .expect("weight parameter buffer");
    let gradient = plan
        .param_grad_pairs
        .iter()
        .find_map(|(candidate, gradient)| (*candidate == parameter).then_some(*gradient))
        .expect("weight gradient buffer");
    let mut actual = vec![0.0; INPUTS * OUTPUTS];
    session.read_buffer(gradient, &mut actual);

    let expected = LOSS_SCALE / OUTPUTS as f32;
    let max_error = actual
        .iter()
        .map(|value| (value - expected).abs())
        .fold(0.0_f32, f32::max);
    assert!(actual.iter().all(|value| value.is_finite()));
    assert!(
        actual.iter().all(|value| *value != 0.0),
        "full-precision weight gradient underflowed to zero"
    );
    assert!(
        max_error <= expected.abs() * 1.0e-4,
        "weight gradient error {max_error:e}, expected {expected:e}"
    );
}

#[test]
fn q4_matmul_correctness() {
    // Q4 matmul correctness: compare GPU Q4 against CPU Q4 reference.
    let m = 6;
    let k = 1024;
    let n = 2048;

    // Build f32 reference graph
    let mut g_ref = Graph::new();
    let a_ref = g_ref.input("a", &[m, k]);
    let b_ref = g_ref.parameter("b", &[k, n]);
    let c_ref = g_ref.matmul(a_ref, b_ref);
    g_ref.set_outputs(vec![c_ref]);

    // Build Q4 graph
    let mut g_q4 = Graph::new();
    let a_q4 = g_q4.input("a", &[m, k]);
    let b_q4 = g_q4.parameter_q4("b", &[k, n]);
    let c_q4 = g_q4.matmul(a_q4, b_q4);
    g_q4.set_outputs(vec![c_q4]);

    let mut sess_ref = meganeura::build(&g_ref, meganeura::SessionConfig::inference_from_env()).0;
    let mut sess_q4 = meganeura::build(&g_q4, meganeura::SessionConfig::inference_from_env()).0;

    // Use values closer to real model scale
    let a_data: Vec<f32> = (0..m * k).map(|i| ((i % 7) as f32 - 3.0) * 0.5).collect();
    let b_data: Vec<f32> = (0..k * n)
        .map(|i| {
            let v = ((i % 11) as f32 - 5.0) * 0.3;
            // Add some outliers like real weights
            if i % 1000 == 0 { v * 5.0 } else { v }
        })
        .collect();

    sess_ref.set_input("a", &a_data);
    sess_ref.set_parameter("b", &b_data);
    sess_ref.step();
    sess_ref.wait();
    let ref_output = sess_ref.read_output(m * n);

    sess_q4.set_input("a", &a_data);
    sess_q4.set_parameter("b", &b_data);
    sess_q4.step();
    sess_q4.wait();
    let q4_output = sess_q4.read_output(m * n);

    // CPU Q4 reference: quantize then dequantize on CPU, multiply manually
    let b_dequant = meganeura::runtime::dequantize_q4_0(
        &meganeura::runtime::quantize_q4_0(&b_data, k, n),
        k,
        n,
    );
    let mut cpu_q4_output = vec![0.0f32; m * n];
    for row in 0..m {
        for col in 0..n {
            let mut sum = 0.0;
            for kk in 0..k {
                sum += a_data[row * k + kk] * b_dequant[kk * n + col];
            }
            cpu_q4_output[row * n + col] = sum;
        }
    }

    let nans = q4_output.iter().filter(|v| v.is_nan()).count();
    eprintln!("Q4 matmul test ({}x{}x{}): nans={}", m, k, n, nans);
    let mut max_err = 0.0f32;
    for i in 0..m * n {
        // Compare GPU Q4 output against CPU Q4 reference (not f32 ref)
        let err = (cpu_q4_output[i] - q4_output[i]).abs();
        if err > max_err {
            max_err = err;
        }
        if i < 16 || err > 0.1 {
            eprintln!(
                "  [{},{}]: cpu_q4={:.4}, gpu_q4={:.4}, f32_ref={:.4}, err={:.4}",
                i / n,
                i % n,
                cpu_q4_output[i],
                q4_output[i],
                ref_output[i],
                err,
            );
        }
    }
    // GPU should match CPU Q4 to within f32 precision + GPU rounding
    assert!(
        max_err < 0.1,
        "Q4 GPU vs CPU max error {:.4} exceeds tolerance",
        max_err,
    );
    eprintln!("Q4 matmul PASSED: GPU-CPU max error {:.4}", max_err);
}

/// `q4_matmul_correctness` uses m=6, which only reaches the tiled matmul.
/// Decode multiplies one row at a time, and `compile.rs` sends
/// `m == 1 && n % 4 == 0` to the K-split GEMV instead — a separate shader
/// that needed its own Q4 variant. Without one it emitted the f32 GEMV
/// over packed blocks and returned ~1e37.
#[test]
fn q4_matmul_single_row_matches_reference() {
    // SmolLM2-135M widths: n % 4 == 0 takes the GEMV path, 1534 the tiled
    // one as a control.
    for (k, n) in [
        (576usize, 1536usize),
        (576, 49152),
        (1536, 576),
        (576, 1534),
    ] {
        let a: Vec<f32> = (0..k).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
        let b: Vec<f32> = (0..k * n)
            .map(|i| ((i % 97) as f32 - 48.0) * 0.01)
            .collect();

        let mut g = Graph::new();
        let a_in = g.input("a", &[1, k]);
        let b_q4 = g.parameter_q4("b", &[k, n]);
        let c = g.matmul(a_in, b_q4);
        g.set_outputs(vec![c]);
        let mut session = meganeura::build(&g, meganeura::SessionConfig::inference_from_env()).0;
        session.set_input("a", &a);
        session.set_parameter("b", &b);
        session.step();
        session.wait();
        let gpu = session.read_output(n);

        let b_deq =
            meganeura::runtime::dequantize_q4_0(&meganeura::runtime::quantize_q4_0(&b, k, n), k, n);
        let mut max_err = 0.0f32;
        for col in 0..n {
            let mut want = 0.0f32;
            for i in 0..k {
                want += a[i] * b_deq[i * n + col];
            }
            max_err = max_err.max((gpu[col] - want).abs());
        }
        assert!(
            gpu.iter().all(|v| v.is_finite()),
            "Q4 m=1 {k}x{n}: non-finite output"
        );
        // The GEMV K-splits the sum, so this is f32 reassociation only.
        assert!(
            max_err < 1e-2,
            "Q4 m=1 {k}x{n}: GPU vs CPU Q4 max error {max_err}"
        );
    }

    // A fused matmul+add at m=1 goes to MatMulGemvAdd, a second shader
    // that needed the same Q4 variant.
    let (k, n) = (576usize, 1536usize);
    let a: Vec<f32> = (0..k).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
    let b: Vec<f32> = (0..k * n)
        .map(|i| ((i % 97) as f32 - 48.0) * 0.01)
        .collect();
    let bias: Vec<f32> = (0..n).map(|i| ((i % 7) as f32 - 3.0) * 0.02).collect();

    let mut g = Graph::new();
    let a_in = g.input("a", &[1, k]);
    let b_q4 = g.parameter_q4("b", &[k, n]);
    let d_in = g.input("d", &[1, n]);
    let mm = g.matmul(a_in, b_q4);
    let c = g.add(mm, d_in);
    g.set_outputs(vec![c]);
    let mut session = meganeura::build(&g, meganeura::SessionConfig::inference_from_env()).0;
    session.set_input("a", &a);
    session.set_parameter("b", &b);
    session.set_input("d", &bias);
    session.step();
    session.wait();
    let gpu = session.read_output(n);

    let b_deq =
        meganeura::runtime::dequantize_q4_0(&meganeura::runtime::quantize_q4_0(&b, k, n), k, n);
    let mut max_err = 0.0f32;
    for col in 0..n {
        let mut want = bias[col];
        for i in 0..k {
            want += a[i] * b_deq[i * n + col];
        }
        max_err = max_err.max((gpu[col] - want).abs());
    }
    assert!(
        max_err < 1e-2,
        "Q4 fused m=1 {k}x{n}: GPU vs CPU Q4 max error {max_err}"
    );
}

/// Roadmap: "quantized matmul, required before E4B". A decode graph is
/// all m=1 matmuls, so this is the shape the Q4 GEMV was for.
///
/// Only the seven per-layer projections are packed. The embedding table
/// stays f32 (`embedding` has no Q4 gather, and a tied `lm_head` shares
/// it), as do the norms and the KV cache.
#[test]
#[cfg(feature = "models")]
fn smollm2_q4_projections_match_f32_decode() {
    use meganeura::models::smollm2;

    fn decode(config: &smollm2::Config, weights: smollm2::ProjectionWeights) -> (Vec<f32>, usize) {
        let mut g = Graph::new();
        let (logits, _k, _v) = smollm2::build_decode_graph_with(&mut g, config, 16, weights);
        g.set_outputs(vec![logits]);
        let mut s = meganeura::build(&g, meganeura::SessionConfig::inference_from_env()).0;
        let mut param_bytes = 0usize;
        for (name, buf) in s.plan().param_buffers.clone() {
            let bytes = s.plan().buffers[buf.0 as usize];
            param_bytes += bytes;
            if name.contains("kv_cache") {
                s.set_parameter(&name, &vec![0.0f32; bytes / 4]);
                continue;
            }
            // A packed buffer is smaller than its logical element count, so
            // size the upload from the weight shape, not the buffer.
            let n = s
                .plan()
                .weight_buffers
                .get(&buf)
                .map(|&(_, r, c)| r * c)
                .unwrap_or(bytes / 4);
            let seed = name.bytes().fold(2166136261u32, |h, b| {
                h.wrapping_mul(16777619) ^ u32::from(b)
            });
            let data: Vec<f32> = (0..n)
                .map(|i| {
                    let x = seed.wrapping_add((i as u32).wrapping_mul(2654435761)) as f32
                        / (1u32 << 31) as f32;
                    (x - 1.0) * 0.2
                })
                .collect();
            s.set_parameter(&name, &data);
        }
        let mut logits = Vec::new();
        for (pos, tok) in [1u32, 2, 3, 4, 5].iter().enumerate() {
            s.set_input_u32("token_ids", &[*tok]);
            s.set_input_u32("kv_pos", &[pos as u32]);
            s.step();
            s.wait();
            logits = s.read_output(config.vocab_size);
        }
        (logits, param_bytes)
    }

    let config = smollm2::Config {
        hidden_size: 64, // One 64-wide head keeps the decode fixture small.
        num_attention_heads: 1,
        num_key_value_heads: 1,
        ..smollm2::Config::small_test()
    };
    let (f32_logits, f32_bytes) = decode(&config, smollm2::ProjectionWeights::F32);
    let (q4_logits, q4_bytes) = decode(&config, smollm2::ProjectionWeights::Q4);

    assert!(
        q4_logits.iter().all(|v| v.is_finite()),
        "Q4 decode produced non-finite logits"
    );
    let scale = f32_logits
        .iter()
        .cloned()
        .fold(0.0f32, |m, v| m.max(v.abs()));
    let err = f32_logits
        .iter()
        .zip(&q4_logits)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    };
    // Bound quantization error relative to the f32 logit scale.
    assert!(
        err / scale < 0.05,
        "Q4 decode diverged from f32: max_abs_err={err} (logit range {scale})"
    );
    assert_eq!(
        argmax(&f32_logits),
        argmax(&q4_logits),
        "Q4 decode would generate a different token"
    );
    assert!(
        q4_bytes * 2 < f32_bytes,
        "Q4 parameters should be far smaller: {q4_bytes} vs {f32_bytes} bytes"
    );
}

/// RmsNorm folds into a following GEMV, and that fused pipeline is a
/// `Variant::GemvRmsNorm` which does not compose with `Variant::Weight`.
/// Folding it into a Q4 GEMV therefore ran the f32 kernel over packed
/// blocks, under a binding layout carrying an extra buffer — so the
/// corruption landed in neighbouring buffers (a KV cache, in the graph
/// that found this) rather than only in the result.
///
/// Unreachable until Q4 gained a GEMV variant, since Q4 was kept off that
/// path entirely before then.
#[test]
fn q4_matmul_after_rms_norm_matches_reference() {
    let (k, n) = (576usize, 1536usize);
    let x: Vec<f32> = (0..k).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
    let nw: Vec<f32> = (0..k).map(|i| 1.0 + ((i % 5) as f32 - 2.0) * 0.1).collect();
    let w: Vec<f32> = (0..k * n)
        .map(|i| ((i % 97) as f32 - 48.0) * 0.01)
        .collect();
    let eps = 1e-5f32;

    let mut g = Graph::new();
    let x_in = g.input("x", &[1, k]);
    let norm_w = g.parameter("norm_w", &[k]);
    let h = g.rms_norm(x_in, norm_w, eps);
    let w_q4 = g.parameter_q4("w", &[k, n]);
    let out = g.matmul(h, w_q4);
    g.set_outputs(vec![out]);
    let mut session = meganeura::build(&g, meganeura::SessionConfig::inference_from_env()).0;
    session.set_input("x", &x);
    session.set_parameter("norm_w", &nw);
    session.set_parameter("w", &w);
    session.step();
    session.wait();
    let gpu = session.read_output(n);

    let ms = x.iter().map(|v| v * v).sum::<f32>() / k as f32;
    let inv = (ms + eps).sqrt().recip();
    let normed: Vec<f32> = x.iter().zip(&nw).map(|(v, g)| v * inv * g).collect();
    let w_deq =
        meganeura::runtime::dequantize_q4_0(&meganeura::runtime::quantize_q4_0(&w, k, n), k, n);
    let mut max_err = 0.0f32;
    let mut scale = 0.0f32;
    for col in 0..n {
        let mut want = 0.0f32;
        for i in 0..k {
            want += normed[i] * w_deq[i * n + col];
        }
        scale = scale.max(want.abs());
        max_err = max_err.max((gpu[col] - want).abs());
    }
    assert!(
        gpu.iter().all(|v| v.is_finite()),
        "Q4 matmul after RmsNorm produced non-finite values"
    );
    assert!(
        max_err / scale < 1e-3,
        "Q4 matmul after RmsNorm diverged: max_abs_err={max_err} (scale {scale})"
    );
}

/// A store-side unary epilogue fused onto a Q4 tiled matmul.
///
/// The epilogue generator emits the packed-B declaration and the pack8
/// staging path, so this covers the combination that `fuse_epilogues`
/// admits for reduced-storage weights: one dispatch, packed B, and the
/// activation applied at the store.
#[test]
fn q4_matmul_with_relu_epilogue_matches_cpu() {
    let (m, k, n) = (8usize, 256usize, 512usize);
    let a: Vec<f32> = (0..m * k).map(|i| ((i % 17) as f32 - 8.0) * 0.05).collect();
    let w: Vec<f32> = (0..k * n)
        .map(|i| ((i % 97) as f32 - 48.0) * 0.01)
        .collect();

    let mut g = Graph::new();
    let x = g.input("x", &[m, k]);
    let w_q4 = g.parameter_q4("w", &[k, n]);
    let mm = g.matmul(x, w_q4);
    let out = g.relu(mm);
    g.set_outputs(vec![out]);
    let mut session = meganeura::build(&g, meganeura::SessionConfig::inference_from_env()).0;
    session.set_input("x", &a);
    session.set_parameter("w", &w);
    session.step();
    session.wait();
    let gpu = session.read_output(m * n);

    let w_deq =
        meganeura::runtime::dequantize_q4_0(&meganeura::runtime::quantize_q4_0(&w, k, n), k, n);
    let mut max_err = 0.0f32;
    let mut scale = 0.0f32;
    for row in 0..m {
        for col in 0..n {
            let mut want = 0.0f32;
            for i in 0..k {
                want += a[row * k + i] * w_deq[i * n + col];
            }
            let want = want.max(0.0);
            scale = scale.max(want.abs());
            max_err = max_err.max((gpu[row * n + col] - want).abs());
        }
    }
    assert!(
        gpu.iter().all(|v| v.is_finite()),
        "Q4 + relu epilogue produced non-finite values"
    );
    assert!(
        gpu.iter().all(|&v| v >= 0.0),
        "Q4 + relu epilogue emitted a negative value"
    );
    assert!(
        max_err / scale.max(1e-6) < 1e-3,
        "Q4 + relu epilogue diverged: max_abs_err={max_err} (scale {scale})"
    );
}

/// A fused epilogue on a matmul small enough to be demoted to 32×32 tiles.
///
/// `select_variants` rewrites `workgroups` for the smaller tile, so the
/// epilogue pipeline has to be generated for the same geometry — and with
/// the matching staging maps, since the 64-wide and 32-wide skeletons
/// split the flat thread index differently. Getting the tile right but the
/// maps wrong stages the wrong elements into shared memory, which is a
/// value error rather than a dispatch-shape one, so this compares against
/// a CPU reference. `small_tile_demotion_survives_epilogue_fusion` covers
/// the dispatch geometry itself.
#[test]
fn small_tile_matmul_with_epilogue_matches_cpu() {
    let dim = 64usize;
    let a: Vec<f32> = (0..dim * dim)
        .map(|i| ((i % 23) as f32 - 11.0) * 0.05)
        .collect();
    let w: Vec<f32> = (0..dim * dim)
        .map(|i| ((i % 31) as f32 - 15.0) * 0.03)
        .collect();

    let mut g = Graph::new();
    let x = g.input("x", &[dim, dim]);
    let p = g.parameter("w", &[dim, dim]);
    let mm = g.matmul(x, p);
    let out = g.relu(mm);
    g.set_outputs(vec![out]);
    let mut session = meganeura::build(&g, meganeura::SessionConfig::inference_from_env()).0;
    session.set_input("x", &a);
    session.set_parameter("w", &w);
    session.step();
    session.wait();
    let gpu = session.read_output(dim * dim);

    let mut max_err = 0.0f32;
    let mut scale = 0.0f32;
    for row in 0..dim {
        for col in 0..dim {
            let mut want = 0.0f32;
            for i in 0..dim {
                want += a[row * dim + i] * w[i * dim + col];
            }
            let want = want.max(0.0);
            scale = scale.max(want.abs());
            max_err = max_err.max((gpu[row * dim + col] - want).abs());
        }
    }
    assert!(
        max_err / scale.max(1e-6) < 1e-4,
        "small-tile epilogue diverged: max_abs_err={max_err} (scale {scale})"
    );
}

/// Build one Q4_K superblock with a realistic spread: per-sub-block scales
/// and mins that actually differ, and quants across the whole nibble range.
/// Packed exactly the way `get_scale_min_k4` expects to read it back.
#[cfg(feature = "gguf")]
fn q4k_superblock(seed: u32) -> Vec<u8> {
    let mut st = seed | 1;
    let mut rnd = || {
        st = st.wrapping_mul(747796405).wrapping_add(2891336453);
        let w = ((st >> ((st >> 28) + 4)) ^ st).wrapping_mul(277803737);
        (w >> 22) ^ w
    };
    let mut b = vec![0u8; 144];
    b[0..2].copy_from_slice(&half::f16::from_f32(0.0035).to_bits().to_le_bytes());
    b[2..4].copy_from_slice(&half::f16::from_f32(0.0021).to_bits().to_le_bytes());
    let sc: Vec<u8> = (0..8).map(|_| (rnd() % 64) as u8).collect();
    let mn: Vec<u8> = (0..8).map(|_| (rnd() % 64) as u8).collect();
    for j in 0..4 {
        b[4 + j] = sc[j] & 63;
        b[8 + j] = mn[j] & 63;
    }
    for j in 4..8 {
        b[8 + j] = (sc[j] & 0x0F) | ((mn[j] & 0x0F) << 4);
        b[j] |= (sc[j] >> 4) << 6;
        b[4 + j] |= (mn[j] >> 4) << 6;
    }
    for i in 0..128 {
        b[16 + i] = (rnd() % 256) as u8;
    }
    b
}

#[cfg(feature = "gguf")]
fn assert_gguf_packed_matmul(
    case: &str,
    tensor: meganeura::load::gguf::GgufTensor,
    m: usize,
    tolerance: f32,
) {
    use meganeura::graph::DType;

    let [k, n] = tensor.dims.as_slice() else {
        panic!("{case}: expected a matrix, got {:?}", tensor.dims);
    };
    let (k, n) = (*k, *n);
    let reference = tensor.to_f32().unwrap();
    let (dtype, packed) = tensor.to_packed().unwrap();
    let input: Vec<f32> = (0..m * k)
        .map(|i| ((i % 29) as f32 - 14.0) * 0.03)
        .collect();
    let mut graph = Graph::new();
    let x = graph.input("x", &[m, k]);
    let w = match dtype {
        DType::Q4_0 => graph.parameter_q4("w", &[k, n]),
        DType::Q40 => graph.parameter_q40("w", &[k, n]),
        DType::Q8_0 => graph.parameter_q8("w", &[k, n]),
        DType::Q4K => graph.parameter_q4k("w", &[k, n]),
        DType::Q6K => graph.parameter_q6k("w", &[k, n]),
        DType::Q5K => graph.parameter_q5k("w", &[k, n]),
        DType::Q3K => graph.parameter_q3k("w", &[k, n]),
        other => panic!("{case}: unexpected packed dtype {other:?}"),
    };
    let out = graph.matmul(x, w);
    graph.set_outputs(vec![out]);
    // This compares against GGML's own dequantizer at f32 tolerance, which
    // is a claim about the decode path. Quantized activations deliberately
    // compute something else, so they are pinned off here rather than
    // letting the environment decide what this test measures.
    let mut session = meganeura::build(
        &graph,
        meganeura::SessionConfig {
            options: meganeura::compile::CompileOptions {
                quantized_activations: false,
                ..meganeura::compile::CompileOptions::from_env()
            },
            ..meganeura::SessionConfig::inference_from_env()
        },
    )
    .0;
    session.set_input("x", &input);
    session.set_parameter_packed("w", &packed);
    session.step();
    session.wait();
    let actual = session.read_output(m * n);

    let mut max_error = 0.0f32;
    let mut scale = 0.0f32;
    for row in 0..m {
        for col in 0..n {
            let expected = (0..k)
                .map(|i| input[row * k + i] * reference[i * n + col])
                .sum::<f32>();
            scale = scale.max(expected.abs());
            max_error = max_error.max((actual[row * n + col] - expected).abs());
        }
    }
    assert!(
        actual.iter().all(|v| v.is_finite()),
        "{case}: non-finite output"
    );
    assert!(
        max_error / scale.max(1e-6) < tolerance,
        "{case}: max_abs_err={max_error}, scale={scale}, tolerance={tolerance}"
    );
}

/// Every native K-quant refuses the two things it cannot do, in one test
/// rather than one per format: there is no host encoder, so `set_parameter`
/// must send callers to `set_parameter_packed`; and blocks run along the
/// parameter's first dimension while decoders index along K, so a
/// transposed B has no correct reading.
#[test]
fn k_quants_are_load_only_and_refuse_transposed_b() {
    use meganeura::graph::DType;

    // Quiet the per-case backtraces; the assertions below report failures.
    let previous = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));

    let build = |g: &mut Graph, dtype: DType, name: &str, shape: &[usize]| match dtype {
        DType::Q4K => g.parameter_q4k(name, shape),
        DType::Q5K => g.parameter_q5k(name, shape),
        DType::Q6K => g.parameter_q6k(name, shape),
        DType::Q3K => g.parameter_q3k(name, shape),
        DType::Q40 => g.parameter_q40(name, shape),
        other => panic!("unexpected dtype {other:?}"),
    };

    let mut failures = Vec::new();
    // GGML Q4_0 is load-only for the same reason as the K-quants: the
    // encoder Meganeura has produces its own asymmetric Q4, not this.
    for dtype in [DType::Q4K, DType::Q5K, DType::Q6K, DType::Q3K, DType::Q40] {
        let (k, n) = (256usize, 4usize);

        // No host encoder: f32 upload must be refused.
        let upload = std::panic::catch_unwind(|| {
            let mut g = Graph::new();
            let x = g.input("x", &[1, k]);
            let w = build(&mut g, dtype, "w", &[k, n]);
            let out = g.matmul(x, w);
            g.set_outputs(vec![out]);
            let mut session =
                meganeura::build(&g, meganeura::SessionConfig::inference_from_env()).0;
            session.set_parameter("w", &vec![0.1f32; k * n]);
        });
        if upload.is_ok() {
            failures.push(format!("{dtype:?}: set_parameter should have been refused"));
        }

        // Transposed B has no correct reading for a block format.
        let bt = std::panic::catch_unwind(|| {
            let mut g = Graph::new();
            let x = g.input("x", &[1, 512]);
            let w = build(&mut g, dtype, "w", &[256, 512]);
            let out = g.matmul_bt(x, w);
            g.set_outputs(vec![out]);
            let _ = meganeura::build(&g, meganeura::SessionConfig::inference_from_env());
        });
        if bt.is_ok() {
            failures.push(format!("{dtype:?}: matmul_bt should have been refused"));
        }
    }

    std::panic::set_hook(previous);
    assert!(failures.is_empty(), "{failures:#?}");
}

/// Every GGUF packed weight format against the loader's CPU references,
/// which are themselves written from `ggml-quants.c`.
///
/// One test rather than one per format and shape: these all exercise the
/// same decode-and-multiply path, and the per-case label says which row
/// failed. The shapes cover what actually differs —
///
/// * `tiled` — the batched-staging path, one output tile.
/// * `GEMV` — `m = 1`, the separate K-split shader decode runs.
/// * `multi-tile` — 65 x 512 x 68, past one 64x64 tile in both M and N and
///   two superblocks deep in K, so tile offsets and K-block progression are
///   exercised rather than a single tile.
/// * `odd pad` — an odd superblock count for the formats whose blocks are
///   not a whole number of words, so half of them start at byte 2 and the
///   buffer needs a tail.
#[test]
#[cfg(feature = "gguf")]
fn gguf_packed_matmul_variants_match_ggml_reference() {
    use meganeura::load::gguf::{GgmlType, GgufTensor};

    type Build = fn(u32) -> Vec<u8>;
    // (type, builder, elements per block, bytes per block)
    let formats: [(GgmlType, Build, usize, usize); 5] = [
        (GgmlType::Q4K, q4k_superblock as Build, 256, 144),
        (GgmlType::Q5K, q5k_superblock as Build, 256, 176),
        // 210, 110 and 18 bytes: not whole words, so the odd-count case
        // matters for these three.
        (GgmlType::Q6K, q6k_superblock as Build, 256, 210),
        (GgmlType::Q3K, q3k_superblock as Build, 256, 110),
        (GgmlType::Q4_0, q40_block as Build, 32, 18),
    ];

    for (ty, build, block, bytes) in formats {
        let unaligned = !bytes.is_multiple_of(4);
        // (label, m, k, n)
        let mut shapes = vec![
            ("tiled", 3usize, 512usize, 4usize),
            ("GEMV", 1, 256, 8),
            ("multi-tile", 65, 512, 68),
        ];
        if unaligned {
            shapes.push(("odd pad", 2, 256, 3));
        }
        for (label, m, k, n) in shapes {
            let blocks = k / block * n;
            let mut data = Vec::new();
            for s in 0..blocks {
                data.extend_from_slice(&build(s as u32 + 1));
            }
            assert_eq!(data.len(), blocks * bytes, "{ty:?} {label}: block size");
            if unaligned && !(blocks * bytes).is_multiple_of(4) {
                assert_ne!(data.len() % 4, 0, "{ty:?} {label}: expected a ragged tail");
            }
            assert_gguf_packed_matmul(
                &format!("{ty:?} {label}"),
                GgufTensor::new(vec![k, n], ty, data),
                m,
                1e-5,
            );
        }
    }
}

/// One GGML Q4_0 block: an f16 scale and sixteen nibble bytes spanning the
/// full 0..15 range, so both halves of the split-nibble layout and both ends
/// of the -8 bias are exercised.
#[cfg(feature = "gguf")]
fn q40_block(seed: u32) -> Vec<u8> {
    let mut st = seed | 1;
    let mut rnd = || {
        st = st.wrapping_mul(747796405).wrapping_add(2891336453);
        let w = ((st >> ((st >> 28) + 4)) ^ st).wrapping_mul(277803737);
        (w >> 22) ^ w
    };
    let mut b = vec![0u8; 18];
    // A scale that varies per block, so a decoder reading the wrong block's
    // header shows up rather than cancelling out.
    let d = 0.015 + (seed % 7) as f32 * 0.004;
    b[0..2].copy_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
    for byte in b[2..18].iter_mut() {
        *byte = (rnd() & 0xFF) as u8;
    }
    // Pin the extremes: nibble 0 decodes to -8d and nibble 15 to +7d.
    b[2] = 0x0F;
    b[3] = 0xF0;
    b
}

/// One Q5_K superblock: a spread of scales and mins, plus full-range
/// nibbles and high bits.
#[cfg(feature = "gguf")]
fn q5k_superblock(seed: u32) -> Vec<u8> {
    let mut st = seed | 1;
    let mut rnd = || {
        st = st.wrapping_mul(747796405).wrapping_add(2891336453);
        let w = ((st >> ((st >> 28) + 4)) ^ st).wrapping_mul(277803737);
        (w >> 22) ^ w
    };
    let mut b = vec![0u8; 176];
    b[0..2].copy_from_slice(&half::f16::from_f32(0.0031).to_bits().to_le_bytes());
    b[2..4].copy_from_slice(&half::f16::from_f32(0.0019).to_bits().to_le_bytes());
    let sc: Vec<u8> = (0..8).map(|_| (rnd() % 64) as u8).collect();
    let mn: Vec<u8> = (0..8).map(|_| (rnd() % 64) as u8).collect();
    for j in 0..4 {
        b[4 + j] = sc[j] & 63;
        b[8 + j] = mn[j] & 63;
    }
    for j in 4..8 {
        b[8 + j] = (sc[j] & 0x0F) | ((mn[j] & 0x0F) << 4);
        b[j] |= (sc[j] >> 4) << 6;
        b[4 + j] |= (mn[j] >> 4) << 6;
    }
    // qh at 16..48 then qs at 48..176.
    for byte in b[16..176].iter_mut() {
        *byte = (rnd() % 256) as u8;
    }
    b
}

/// One Q3_K superblock: full-range hmask, 2-bit quants and packed scales.
#[cfg(feature = "gguf")]
fn q3k_superblock(seed: u32) -> Vec<u8> {
    let mut st = seed | 1;
    let mut rnd = || {
        st = st.wrapping_mul(747796405).wrapping_add(2891336453);
        let w = ((st >> ((st >> 28) + 4)) ^ st).wrapping_mul(277803737);
        (w >> 22) ^ w
    };
    let mut b = vec![0u8; 110];
    for byte in b[..108].iter_mut() {
        *byte = (rnd() % 256) as u8;
    }
    b[108..110].copy_from_slice(&half::f16::from_f32(0.0042).to_bits().to_le_bytes());
    b
}

/// The packed SwiGLU concat restages a derived `gate+up` from two uploads.
/// Q3_K is the interesting one: each source pads to a word, but that tail
/// must not land between the two sources' superblocks.
#[test]
#[cfg(feature = "gguf")]
fn q3k_swiglu_packed_concat_matches_reference() {
    use meganeura::load::gguf::{GgmlType, GgufTensor};

    let (m, k, n) = (2usize, 256usize, 4usize);
    let mut gate_data = Vec::new();
    let mut up_data = Vec::new();
    for s in 0..(k / 256 * n) {
        gate_data.extend_from_slice(&q3k_superblock(s as u32 + 5));
        up_data.extend_from_slice(&q3k_superblock(s as u32 + 23));
    }
    let gate_t = GgufTensor::new(vec![k, n], GgmlType::Q3K, gate_data);
    let up_t = GgufTensor::new(vec![k, n], GgmlType::Q3K, up_data);
    let gate_ref = gate_t.to_f32().unwrap();
    let up_ref = up_t.to_f32().unwrap();
    let (_, gate_packed) = gate_t.to_packed().unwrap();
    let (_, up_packed) = up_t.to_packed().unwrap();

    let a: Vec<f32> = (0..m * k)
        .map(|i| ((i % 29) as f32 - 14.0) * 0.03)
        .collect();

    let mut g = Graph::new();
    let x = g.input("x", &[m, k]);
    let gate_w = g.parameter_q3k("gate", &[k, n]);
    let up_w = g.parameter_q3k("up", &[k, n]);
    let gate = g.matmul(x, gate_w);
    let up = g.matmul(x, up_w);
    let out = g.swiglu(gate, up);
    g.set_outputs(vec![out]);
    let mut session = meganeura::build(&g, meganeura::SessionConfig::inference_from_env()).0;
    assert!(
        session.has_parameter("gate+up"),
        "expected SwiGLU concat fusion so packed upload restages the derived weight"
    );
    session.set_input("x", &a);
    // Upload order reversed relative to the declaration order.
    session.set_parameter_packed("up", &up_packed);
    session.set_parameter_packed("gate", &gate_packed);
    session.step();
    session.wait();
    let gpu = session.read_output(m * n);

    let silu = |v: f32| v / (1.0 + (-v).exp());
    let mut max_err = 0.0f32;
    let mut scale = 0.0f32;
    for row in 0..m {
        for col in 0..n {
            let gate_v = (0..k)
                .map(|i| a[row * k + i] * gate_ref[i * n + col])
                .sum::<f32>();
            let up_v = (0..k)
                .map(|i| a[row * k + i] * up_ref[i * n + col])
                .sum::<f32>();
            let want = silu(gate_v) * up_v;
            scale = scale.max(want.abs());
            max_err = max_err.max((gpu[row * n + col] - want).abs());
        }
    }
    assert!(
        max_err / scale.max(1e-6) < 1e-4,
        "Q3_K SwiGLU concat diverged: max_abs_err={max_err} (scale {scale})"
    );
}

/// One Q6_K superblock with signed scales spanning both polarities and
/// quants across the full 6-bit range.
#[cfg(feature = "gguf")]
fn q6k_superblock(seed: u32) -> Vec<u8> {
    let mut st = seed | 1;
    let mut rnd = || {
        st = st.wrapping_mul(747796405).wrapping_add(2891336453);
        let w = ((st >> ((st >> 28) + 4)) ^ st).wrapping_mul(277803737);
        (w >> 22) ^ w
    };
    let mut b = vec![0u8; 210];
    // ql and qh: full range.
    for byte in b[..192].iter_mut() {
        *byte = (rnd() % 256) as u8;
    }
    // int8 scales, deliberately straddling zero.
    for byte in b[192..208].iter_mut() {
        *byte = ((rnd() % 80) as i32 - 40) as i8 as u8;
    }
    b[208..210].copy_from_slice(&half::f16::from_f32(0.0012).to_bits().to_le_bytes());
    b
}

/// A subnormal f16 block scale is an ordinary f32, and must survive.
///
/// `0x0100` is 2^-16: subnormal as a half, exactly representable as a
/// float. The hand-assembled decoder this replaced flushed `expo == 0` to
/// zero, which erased the whole superblock. Every other fixture here uses
/// a normal scale, so nothing else would catch it.
#[test]
#[cfg(feature = "gguf")]
fn q6k_preserves_subnormal_block_scales() {
    use meganeura::load::gguf::{GgmlType, GgufTensor};

    let (k, n) = (256usize, 4usize);
    let mut block = vec![0u8; 210];
    // ql = qh = 0xff gives the maximum 6-bit quant, 63 -> 63 - 32 = 31.
    for b in block[..192].iter_mut() {
        *b = 0xFF;
    }
    for b in block[192..208].iter_mut() {
        *b = 127; // int8 subscale
    }
    block[208..210].copy_from_slice(&0x0100u16.to_le_bytes()); // d = 2^-16

    let mut data = Vec::new();
    for _ in 0..n {
        data.extend_from_slice(&block);
    }
    let tensor = GgufTensor::new(vec![k, n], GgmlType::Q6K, data.clone());
    let w_ref = tensor.to_f32().unwrap();
    // 2^-16 * 127 * 31, an ordinary number.
    let want_elem = (2.0f32).powi(-16) * 127.0 * 31.0;
    assert!(
        (w_ref[0] - want_elem).abs() < 1e-9,
        "CPU reference lost the subnormal scale: {}",
        w_ref[0]
    );

    let (_, packed) = tensor.to_packed().unwrap();
    // One-hot input, so each output is a single weight.
    let mut a = vec![0.0f32; k];
    a[0] = 1.0;

    let mut g = Graph::new();
    let x = g.input("x", &[1, k]);
    let w = g.parameter_q6k("w", &[k, n]);
    let out = g.matmul(x, w);
    g.set_outputs(vec![out]);
    let mut session = meganeura::build(&g, meganeura::SessionConfig::inference_from_env()).0;
    session.set_input("x", &a);
    session.set_parameter_packed("w", &packed);
    session.step();
    session.wait();
    let gpu = session.read_output(n);

    for (col, &got) in gpu.iter().enumerate() {
        assert!(
            got != 0.0,
            "column {col}: subnormal scale decoded to zero on the GPU"
        );
        assert!(
            (got - w_ref[col]).abs() / w_ref[col].abs() < 1e-5,
            "column {col}: got {got}, want {}",
            w_ref[col]
        );
    }
}

/// Default extraction concatenates SwiGLU gate/up into one matmul.
/// `set_parameter_packed` has to restage that derived buffer; uploading
/// only the named sources leaves the fused weight uninitialized.
#[test]
#[cfg(feature = "gguf")]
fn q4k_swiglu_packed_concat_matches_reference() {
    use meganeura::load::gguf::{GgmlType, GgufTensor};

    let (m, k, n) = (2usize, 256usize, 4usize);
    let mut gate_data = Vec::new();
    let mut up_data = Vec::new();
    for s in 0..(k / 256 * n) {
        gate_data.extend_from_slice(&q4k_superblock(s as u32 + 3));
        up_data.extend_from_slice(&q4k_superblock(s as u32 + 19));
    }
    let gate_t = GgufTensor::new(vec![k, n], GgmlType::Q4K, gate_data);
    let up_t = GgufTensor::new(vec![k, n], GgmlType::Q4K, up_data);
    let gate_ref = gate_t.to_f32().unwrap();
    let up_ref = up_t.to_f32().unwrap();
    let (_, gate_packed) = gate_t.to_packed().unwrap();
    let (_, up_packed) = up_t.to_packed().unwrap();

    let a: Vec<f32> = (0..m * k)
        .map(|i| ((i % 29) as f32 - 14.0) * 0.03)
        .collect();

    let mut g = Graph::new();
    let x = g.input("x", &[m, k]);
    let gate_w = g.parameter_q4k("gate", &[k, n]);
    let up_w = g.parameter_q4k("up", &[k, n]);
    let gate = g.matmul(x, gate_w);
    let up = g.matmul(x, up_w);
    let out = g.swiglu(gate, up);
    g.set_outputs(vec![out]);
    let mut session = meganeura::build(&g, meganeura::SessionConfig::inference_from_env()).0;
    assert!(
        session.has_parameter("gate+up"),
        "expected SwiGLU concat fusion so packed upload restages the derived weight"
    );
    session.set_input("x", &a);
    session.set_parameter_packed("gate", &gate_packed);
    session.set_parameter_packed("up", &up_packed);
    session.step();
    session.wait();
    let gpu = session.read_output(m * n);

    let silu = |v: f32| v / (1.0 + (-v).exp());
    let mut max_err = 0.0f32;
    let mut scale = 0.0f32;
    for row in 0..m {
        for col in 0..n {
            let mut gate_acc = 0.0f32;
            let mut up_acc = 0.0f32;
            for i in 0..k {
                gate_acc += a[row * k + i] * gate_ref[i * n + col];
                up_acc += a[row * k + i] * up_ref[i * n + col];
            }
            let want = silu(gate_acc) * up_acc;
            scale = scale.max(want.abs());
            max_err = max_err.max((gpu[row * n + col] - want).abs());
        }
    }
    assert!(
        gpu.iter().all(|v| v.is_finite()),
        "Q4_K SwiGLU produced non-finite values"
    );
    assert!(
        max_err / scale.max(1e-6) < 1e-4,
        "Q4_K SwiGLU concat diverged from the reference: \
         max_abs_err={max_err} (scale {scale})"
    );
}

/// Exercise both repacked legacy formats on the paths their layout is most
/// likely to break: multi-column Q4 and K-split Q8 GEMV.
#[test]
#[cfg(feature = "gguf")]
fn gguf_repacked_matmuls_match_reference() {
    use meganeura::load::gguf::{GgmlType, GgufTensor};

    fn ggml_q4_0_block(d: f32, nibbles: [u8; 32]) -> Vec<u8> {
        let mut b = Vec::with_capacity(18);
        b.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
        for j in 0..16 {
            b.push(nibbles[j] | (nibbles[j + 16] << 4));
        }
        b
    }

    let (m, k, n) = (2usize, 32usize, 4usize);
    let mut data = Vec::new();
    for col in 0..n {
        let nibbles: [u8; 32] = std::array::from_fn(|i| ((i + col * 3) % 16) as u8);
        data.extend(ggml_q4_0_block(0.25 * (col as f32 + 1.0), nibbles));
    }
    assert_gguf_packed_matmul(
        "GGUF Q4_0 multi-column tiled matmul",
        GgufTensor::new(vec![k, n], GgmlType::Q4_0, data),
        m,
        1e-5,
    );

    let mut q8 = Vec::new();
    for col in 0..n {
        q8.extend_from_slice(
            &half::f16::from_f32(0.02 * (col + 1) as f32)
                .to_bits()
                .to_le_bytes(),
        );
        q8.extend((0..32).map(|i| (i - 16 + col as i32) as i8 as u8));
    }
    assert_gguf_packed_matmul(
        "GGUF Q8_0 GEMV",
        GgufTensor::new(vec![k, n], GgmlType::Q8_0, q8),
        1,
        1e-5,
    );
}

/// Packed blocks run along the parameter's first dimension, which differs
/// from K for transposed B. Cover the plain, add-fused, and epilogue
/// routes without emitting a GPU shader for any of them.
#[test]
fn block_quantized_matmul_bt_variants_are_refused() {
    let (m, k, n) = (1usize, 512usize, 256usize);
    for (case, q4, add, relu) in [
        ("Q4 plain", true, false, false),
        ("Q4_K plain", false, false, false),
        ("Q4_K fused add", false, true, false),
        ("Q4_K fused add+relu", false, true, true),
    ] {
        let rejected = std::panic::catch_unwind(|| {
            let mut graph = Graph::new();
            let x = graph.input("x", &[m, k]);
            let w = if q4 {
                graph.parameter_q4("w", &[n, k])
            } else {
                graph.parameter_q4k("w", &[n, k])
            };
            let mut out = graph.matmul_bt(x, w);
            if add {
                let bias = graph.parameter("bias", &[m, n]);
                out = graph.add(out, bias);
            }
            if relu {
                out = graph.relu(out);
            }
            graph.set_outputs(vec![out]);
            let optimized = meganeura::optimize::optimize(&graph);
            meganeura::compile::compile(&optimized);
        });
        assert!(rejected.is_err(), "{case} reached codegen");
    }
}

/// K-quants have no host encoder; silently substituting a cruder quantizer
/// would discard the packed weights the caller already has.
#[test]
fn k_quants_reject_f32_parameter_upload() {
    let (k, n) = (256usize, 4usize);
    for q4 in [true, false] {
        let rejected = std::panic::catch_unwind(|| {
            let mut graph = Graph::new();
            let x = graph.input("x", &[1, k]);
            let w = if q4 {
                graph.parameter_q4k("w", &[k, n])
            } else {
                graph.parameter_q6k("w", &[k, n])
            };
            let out = graph.matmul(x, w);
            graph.set_outputs(vec![out]);
            let mut session =
                meganeura::build(&graph, meganeura::SessionConfig::inference_from_env()).0;
            session.set_parameter("w", &vec![0.1; k * n]);
        });
        assert!(
            rejected.is_err(),
            "{} accepted f32",
            if q4 { "Q4_K" } else { "Q6_K" }
        );
    }
}

/// The reduction extent has to fill whole 256-element superblocks.
#[test]
#[should_panic(expected = "multiple of 256")]
fn q4k_rejects_unaligned_reduction_extent() {
    let mut g = Graph::new();
    let _ = g.parameter_q4k("w", &[128, 4]);
}

/// Incrementally build a 1-layer transformer with Q4 weights.
/// Output each intermediate result to find where NaN first appears.
#[test]
fn q4_layer_nan_hunt() {
    let seq = 6;
    let hidden = 1024;
    let qd = 2048; // 16 heads * 128 head_dim
    let kv = 1024; // 8 kv_heads * 128 head_dim
    let ffn = 3072;
    let head_dim = 128u32;
    let num_heads = 16u32;
    let num_kv_heads = 8u32;

    // Random-ish weight data (deterministic, larger range like real weights)
    let make_weights = |n: usize| -> Vec<f32> {
        (0..n)
            .map(|i| {
                let x = (i as f64 * 0.618033988) % 1.0; // golden ratio hash
                (x * 2.0 - 1.0) as f32 * 0.5
            })
            .collect()
    };

    // Test stages: each adds one more operation
    let stages: &[&str] = &["embed+rmsnorm+qkv+rope+attn+o_proj+residual", "full_layer"];

    for &stage in stages {
        let mut g = Graph::new();
        let token_ids = g.input_u32("token_ids", &[seq]);
        let embed = g.parameter("embed", &[4096, hidden]); // small vocab for test
        let mut x = g.embedding(token_ids, embed);

        let ln1_w = g.parameter("ln1", &[hidden]);
        let h = g.rms_norm(x, ln1_w, 1e-6);

        let wq = g.parameter_q4("wq", &[hidden, qd]);
        let q = g.matmul(h, wq);

        let wk = g.parameter_q4("wk", &[hidden, kv]);
        let wv = g.parameter_q4("wv", &[hidden, kv]);
        let k = g.matmul(h, wk);
        let v = g.matmul(h, wv);
        let q_rope = g.rope(q, 1e6, head_dim);
        let k_rope = g.rope(k, 1e6, head_dim);

        {
            let attn = g.causal_attention(q_rope, k_rope, v, num_heads, num_kv_heads, head_dim);
            let wo = g.parameter_q4("wo", &[qd, hidden]);
            let attn_out = g.matmul(attn, wo);
            x = g.add(x, attn_out);

            if stage == "full_layer" {
                let ln2_w = g.parameter("ln2", &[hidden]);
                let h2 = g.rms_norm(x, ln2_w, 1e-6);
                let wg = g.parameter_q4("wg", &[hidden, ffn]);
                let wu = g.parameter_q4("wu", &[hidden, ffn]);
                let wd = g.parameter_q4("wd", &[ffn, hidden]);
                let gate = g.matmul(h2, wg);
                let up = g.matmul(h2, wu);
                let ffn_out = g.swiglu(gate, up);
                let ffn_out = g.matmul(ffn_out, wd);
                x = g.add(x, ffn_out);
            }
            g.set_outputs(vec![x]);
        }

        let mut session = meganeura::build(&g, meganeura::SessionConfig::inference_from_env()).0;

        // Set dummy inputs
        session.set_input_u32("token_ids", &[10, 20, 30, 40, 50, 60]);
        session.set_parameter("embed", &make_weights(4096 * hidden));
        session.set_parameter("ln1", &vec![1.0f32; hidden]);
        session.set_parameter("wq", &make_weights(hidden * qd));
        session.set_parameter("wk", &make_weights(hidden * kv));
        session.set_parameter("wv", &make_weights(hidden * kv));
        if stage.contains("attn") || stage == "full_layer" {
            session.set_parameter("wo", &make_weights(qd * hidden));
        }
        if stage == "full_layer" {
            session.set_parameter("ln2", &vec![1.0f32; hidden]);
            session.set_parameter("wg", &make_weights(hidden * ffn));
            session.set_parameter("wu", &make_weights(hidden * ffn));
            session.set_parameter("wd", &make_weights(ffn * hidden));
        }

        session.step();
        session.wait();

        // Read only the logically valid output elements
        let out_buf = session.plan().output_buffers[0];
        let buf_size = session.plan().buffers[out_buf.0 as usize] / 4;
        let all_output = session.read_output(buf_size);
        let logical_size = seq * hidden;
        let output = &all_output[..logical_size.min(all_output.len())];
        let nans = output.iter().filter(|v| v.is_nan()).count();
        let infs = output.iter().filter(|v| v.is_infinite()).count();
        let finite: Vec<f32> = output.iter().copied().filter(|v| v.is_finite()).collect();
        let (min_v, max_v) = if finite.is_empty() {
            (f32::NAN, f32::NAN)
        } else {
            (
                finite.iter().copied().fold(f32::INFINITY, f32::min),
                finite.iter().copied().fold(f32::NEG_INFINITY, f32::max),
            )
        };

        eprintln!(
            "  {:<50} nans={}/{}, infs={}, range=[{:.2}, {:.2}]",
            stage,
            nans,
            output.len(),
            infs,
            min_v,
            max_v,
        );

        if nans > 0 {
            eprintln!("  FOUND NaN at stage: {}", stage);
            // Don't break — continue to see all stages
        }
    }
}

/// Regression test for the 1×1-conv dispatch-axes bug.
///
/// The 1×1 stride-1 conv path in `compile.rs` reinterprets `conv2d` as a
/// MatMulBT with `M = batch*HW` rows and `N = out_channels` columns.  The
/// shader's convention is `wgid.x → N tiles`, `wgid.y → M tiles`, matching
/// every other matmul dispatch in the file.  An earlier version of the
/// 1×1 shortcut accidentally swapped them — `[m.div_ceil, n.div_ceil, 1]`
/// instead of `[n.div_ceil, m.div_ceil, 1]` — which silently dropped every
/// output row past the first workgroup column whenever `out_channels <
/// tile_size` (= every 1×1 conv in EfficientNet-V2, since Co ∈ {24, 48,
/// 64, 128, 160} are all < 64).  For batch=N>1 lanes 1..N got all zeros.
///
/// This test triggers the path with the same shape conditions:
/// `batch=4, in_ch=96, H=W=48, out_ch=48` (the features.2.0 project conv)
/// and verifies all four batches produce identical, non-zero output for
/// bit-identical input.
#[test]
fn conv2d_1x1_batch_replicated_input_is_uniform() {
    let batch = 4u32;
    let in_ch = 96u32;
    let out_ch = 48u32;
    let h = 48u32;
    let w = 48u32;
    let in_size = (batch * in_ch * h * w) as usize;
    let out_size = (batch * out_ch * h * w) as usize;

    let mut g = Graph::new();
    let input = g.input("x", &[in_size]);
    let kernel = g.parameter("w", &[(out_ch * in_ch) as usize]);
    let y = g.conv2d(input, kernel, batch, in_ch, h, w, out_ch, 1, 1, 1, 0);
    g.set_outputs(vec![y]);

    let mut session = meganeura::build(&g, meganeura::SessionConfig::inference_from_env()).0;

    // Deterministic kernel weights so every conv output is nonzero.
    let kernel_data: Vec<f32> = (0..(out_ch * in_ch))
        .map(|i| (i as f32 * 0.0173).sin())
        .collect();
    session.set_parameter("w", &kernel_data);

    // Replicate identical [in_ch, H, W] across all batches.
    let single: Vec<f32> = (0..(in_ch * h * w))
        .map(|i| ((i as f32) * 0.0237).cos())
        .collect();
    let mut input_data = Vec::with_capacity(in_size);
    for _ in 0..batch {
        input_data.extend_from_slice(&single);
    }
    session.set_input("x", &input_data);

    session.step();
    session.wait();

    let out = session.read_output(out_size);
    let per_batch = (out_ch * h * w) as usize;
    let lane0 = &out[..per_batch];

    // Every batch slice must match lane 0.
    for b in 1..batch as usize {
        let slice = &out[b * per_batch..(b + 1) * per_batch];
        let mut max_diff = 0.0f32;
        for i in 0..per_batch {
            let d = (slice[i] - lane0[i]).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
        assert!(
            max_diff < 1e-4,
            "lane {b}: max abs diff vs lane 0 = {max_diff} (input was bit-identical \
             across batches — divergence here is the 1×1-conv dispatch-axes regression)"
        );
    }

    // Lane 0 must actually contain real conv output (not all zeros — that's
    // the failure mode this bug produced).
    let lane0_max_abs = lane0.iter().map(|v| v.abs()).fold(0.0, f32::max);
    assert!(
        lane0_max_abs > 0.1,
        "lane 0 max-abs = {lane0_max_abs} — buffer is essentially zero, conv didn't compute"
    );
}

/// Regression test for the kindle batch>1 silent-zero-gradient bug.
///
/// Models the canonical failure mode: graph declares a parameter buffer
/// of size `[batch * channels]` (e.g. a per-batch broadcast bias), but
/// `set_parameter` is called with only the per-channel `[channels]`
/// slice from a safetensors file.  Before the `upload_buffer`
/// size-check, the GPU buffer's tail past the supplied data stayed at
/// whatever it was initialized to (zero on first allocation), and the
/// kernel reading the buffer silently produced garbage for the
/// unwritten portion.  The repro in mind-games was kindle's
/// EfficientNet V2-S BN bias at `batch_size=4`: lane 0 trained, lanes
/// 1-3 silently saw zero features for 50k steps.
///
/// Now `upload_buffer` asserts data byte-length equals the buffer's
/// declared byte size.  This test must panic with the new message.
#[test]
#[should_panic(expected = "byte-size mismatch")]
fn upload_buffer_rejects_undersized_parameter_upload() {
    // Build a graph that declares a `[48]`-element bias (mimicking a
    // per-batch broadcast: 2 batches × 24 channels), used as a 1D bias
    // over a `[1, 48]` input.  Then upload only `[24]` floats — the
    // shape a per-channel safetensors slice would supply.
    let mut g = Graph::new();
    let x = g.input("x", &[1, 48]);
    let bias = g.parameter("bias", &[48]);
    let y = g.bias_add(x, bias);
    g.set_outputs(vec![y]);

    let mut session = meganeura::build(&g, meganeura::SessionConfig::inference_from_env()).0;
    let undersized = vec![0.5_f32; 24]; // 24 floats == 96 bytes
    // Buffer is 48 floats == 192 bytes.  upload_buffer must panic with
    // "byte-size mismatch ... got 96, slot expects 192".
    session.set_parameter("bias", &undersized);
}
