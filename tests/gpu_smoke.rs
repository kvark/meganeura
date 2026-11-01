//! Stack behaviour on the GPU beyond op numerics: optimizer updates,
//! checkpoints, gradient inspection, dispatch geometry at extreme sizes,
//! packed quantized weights, and model-level gradient checks. Every op's
//! values and derivatives are checked by the `oracle` suite.
use meganeura::Graph;
#[cfg(feature = "models")]
use meganeura::{build_session, build_session_unoptimized, compile::BufferRef};

#[test]
fn tall_matmul_splits_dispatch_across_z() {
    const ROWS: usize = 65_536 * 64;

    let mut graph = Graph::new();
    let input = graph.input("input", &[ROWS, 1]);
    let weight = graph.parameter("weight", &[1, 1]);
    let output = graph.matmul(input, weight);
    graph.set_outputs(vec![output]);

    let mut session = meganeura::build(&graph, crate::support::gpu::inference_config()).0;
    session.set_input("input", &vec![1.25_f32; ROWS]);
    session.set_parameter("weight", &[2.0]);
    session.step();
    session.wait();

    let actual = session.read_output(ROWS);
    for index in [0, ROWS / 2, ROWS - 1] {
        assert!((actual[index] - 2.5).abs() < 1.0e-6);
    }
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
fn checkpoint_round_trip() {
    let mut g = Graph::new();
    let x = g.input("x", &[4, 8]);
    let w = g.parameter("w", &[8, 4]);
    let y = g.matmul(x, w);
    let loss = g.mean_all(y);
    g.set_outputs(vec![loss]);

    let mut session = meganeura::build(&g, crate::support::gpu::config()).0;
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
    let mut session2 = meganeura::build(&g, crate::support::gpu::config()).0;
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
    let mut session = meganeura::build(&graph, crate::support::gpu::inference_config()).0;
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
    let mut session = meganeura::build(&g, crate::support::gpu::inference_config()).0;
    session.set_parameter("w", &values);
    session.save_checkpoint(&tmp).expect("save checkpoint");

    let mut restored = meganeura::build(&g, crate::support::gpu::inference_config()).0;
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
    let mut session = meganeura::build(&g, crate::support::gpu::config()).0;

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
    let mut session = meganeura::build(&g, crate::support::gpu::config()).0;

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

    let mut session = meganeura::build(&graph, crate::support::gpu::config()).0;
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

/// Packed Q4 weights on every route a Q4 matmul takes, each against the
/// CPU's own quantize-dequantize round trip, so the tolerances bound the
/// kernel rather than the quantization:
///
/// * `tiled` — m > 1, with outliers like real weights.
/// * `GEMV` — decode multiplies one row at a time, and `compile.rs` sends
///   `m == 1 && n % 4 == 0` to the K-split GEMV, a separate shader that
///   needed its own Q4 variant: without one it emitted the f32 GEMV over
///   packed blocks and returned ~1e37. The GEMV K-splits the sum, so the
///   error is f32 reassociation only. `n = 1534` is the tiled control.
/// * `GEMV + add` — a fused matmul+add at m = 1 goes to `MatMulGemvAdd`, a
///   second shader that needed the same variant.
/// * `RmsNorm + GEMV` — RmsNorm folds into a following GEMV as
///   `Variant::GemvRmsNorm`, which does not compose with `Variant::Weight`.
///   Folding it into a Q4 GEMV ran the f32 kernel over packed blocks under a
///   binding layout with an extra buffer, so the corruption landed in
///   neighbouring buffers (a KV cache, in the graph that found it).
/// * `relu epilogue` — a store-side unary epilogue on the tiled matmul: the
///   epilogue generator emits the packed-B declaration and the pack8 staging
///   path, so this is one dispatch with packed B and the activation applied
///   at the store.
#[test]
fn q4_matmul_routes_match_cpu_quantization() {
    #[derive(Clone, Copy, Debug, PartialEq)]
    enum Route {
        Plain,
        Add,
        RmsNorm,
        Relu,
    }
    #[derive(Clone, Copy, Debug)]
    enum Tolerance {
        Absolute(f32),
        Relative(f32),
    }
    use Route::*;
    use Tolerance::*;
    // `((i % period) - period / 2) * step`.
    let ramp =
        |i: usize, (period, step): (usize, f32)| ((i % period) as f32 - (period / 2) as f32) * step;
    // Input ramp, weight ramp, and whether every 1000th weight is a 5x
    // outlier, as in real weights.
    let tiled = ((7, 0.5), (11, 0.3), true);
    let decode = ((13, 0.05), (97, 0.01), false);
    let epilogue = ((17, 0.05), (97, 0.01), false);
    let cases = [
        (Plain, [6, 1024, 2048], tiled, Absolute(0.1)),
        (Plain, [1, 576, 1536], decode, Absolute(1e-2)),
        (Plain, [1, 1536, 576], decode, Absolute(1e-2)),
        // n % 4 != 0: the tiled control.
        (Plain, [1, 576, 1534], decode, Absolute(1e-2)),
        (Add, [1, 576, 1536], decode, Absolute(1e-2)),
        (RmsNorm, [1, 576, 1536], decode, Relative(1e-3)),
        (Relu, [8, 256, 512], epilogue, Relative(1e-3)),
    ];
    let eps = 1e-5f32;
    let mut failures = Vec::new();
    for (route, [m, k, n], (input, weight, outliers), tolerance) in cases {
        let x: Vec<f32> = (0..m * k).map(|i| ramp(i, input)).collect();
        let w: Vec<f32> = (0..k * n)
            .map(|i| {
                let v = ramp(i, weight);
                if outliers && i % 1000 == 0 {
                    v * 5.0
                } else {
                    v
                }
            })
            .collect();
        let addend: Vec<f32> = (0..m * n).map(|i| ramp(i, (7, 0.02))).collect();
        let norm_w: Vec<f32> = (0..k).map(|i| 1.0 + ramp(i, (5, 0.1))).collect();

        let mut g = Graph::new();
        let x_in = g.input("x", &[m, k]);
        let h = if route == RmsNorm {
            let nw = g.parameter("norm_w", &[k]);
            g.rms_norm(x_in, nw, eps)
        } else {
            x_in
        };
        let w_q4 = g.parameter_q4("w", &[k, n]);
        let mut out = g.matmul(h, w_q4);
        if route == Add {
            let d = g.input("d", &[m, n]);
            out = g.add(out, d);
        } else if route == Relu {
            out = g.relu(out);
        }
        g.set_outputs(vec![out]);
        let mut session = meganeura::build(&g, crate::support::gpu::inference_config()).0;
        session.set_input("x", &x);
        session.set_parameter("w", &w);
        if route == Add {
            session.set_input("d", &addend);
        }
        if route == RmsNorm {
            session.set_parameter("norm_w", &norm_w);
        }
        session.step();
        session.wait();
        let gpu = session.read_output(m * n);

        let rows: Vec<f32> = if route == RmsNorm {
            x.chunks(k)
                .flat_map(|row| {
                    let ms = row.iter().map(|v| v * v).sum::<f32>() / k as f32;
                    let inv = (ms + eps).sqrt().recip();
                    row.iter().zip(&norm_w).map(move |(v, g)| v * inv * g)
                })
                .collect()
        } else {
            x
        };
        let w_deq =
            meganeura::runtime::dequantize_q4_0(&meganeura::runtime::quantize_q4_0(&w, k, n), k, n);
        let (mut max_err, mut scale) = (0.0f32, 0.0f32);
        for row in 0..m {
            for col in 0..n {
                let mut want = if route == Add {
                    addend[row * n + col]
                } else {
                    0.0
                };
                for i in 0..k {
                    want += rows[row * k + i] * w_deq[i * n + col];
                }
                if route == Relu {
                    want = want.max(0.0);
                }
                scale = scale.max(want.abs());
                max_err = max_err.max((gpu[row * n + col] - want).abs());
            }
        }
        let within = match tolerance {
            Absolute(t) => max_err < t,
            Relative(t) => max_err / scale < t,
        };
        let finite = gpu.iter().all(|v| v.is_finite());
        let signs = route != Relu || gpu.iter().all(|&v| v >= 0.0);
        if !(within && finite && signs) {
            failures.push(format!(
                "{route:?} {m}x{k}x{n}: max_abs_err={max_err}, scale={scale}, \
                 {tolerance:?}, finite={finite}, relu signs={signs}"
            ));
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// Roadmap: "quantized matmul, required before E4B". A decode graph is
/// all m=1 matmuls, so this is the shape the Q4 GEMV was for.
///
/// Only the seven per-layer projections are packed. The embedding table
/// stays f32 (`embedding` has no Q4 gather, and a tied `lm_head` shares
/// it), as do the norms and the KV cache. The f32 run doubles as the plain
/// KV-cache decode check: its logits must be finite and nonzero.
#[test]
#[cfg(feature = "models")]
fn smollm2_q4_projections_match_f32_decode() {
    use meganeura::models::smollm2;

    fn decode(config: &smollm2::Config, weights: smollm2::ProjectionWeights) -> (Vec<f32>, usize) {
        let mut g = Graph::new();
        let (logits, _k, _v) = smollm2::build_decode_graph_with(&mut g, config, 16, weights);
        g.set_outputs(vec![logits]);
        let mut s = meganeura::build(&g, crate::support::gpu::inference_config()).0;
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

    for (what, logits) in [("f32", &f32_logits), ("Q4", &q4_logits)] {
        assert!(
            logits.iter().all(|v| v.is_finite()),
            "{what} decode produced non-finite logits"
        );
    }
    let scale = f32_logits
        .iter()
        .cloned()
        .fold(0.0f32, |m, v| m.max(v.abs()));
    assert!(scale > 1e-10, "f32 decode logits are all zero");
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
    let mut session = meganeura::build(&g, crate::support::gpu::inference_config()).0;
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

/// The PCG-style stream every GGUF block fixture draws its bytes from.
#[cfg(feature = "gguf")]
fn block_rng(seed: u32) -> impl FnMut() -> u32 {
    let mut st = seed | 1;
    move || {
        st = st.wrapping_mul(747796405).wrapping_add(2891336453);
        let w = ((st >> ((st >> 28) + 4)) ^ st).wrapping_mul(277803737);
        (w >> 22) ^ w
    }
}

/// Build one Q4_K superblock with a realistic spread: per-sub-block scales
/// and mins that actually differ, and quants across the whole nibble range.
/// Packed exactly the way `get_scale_min_k4` expects to read it back.
#[cfg(feature = "gguf")]
fn q4k_superblock(seed: u32) -> Vec<u8> {
    let mut rnd = block_rng(seed);
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
            ..crate::support::gpu::inference_config()
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
            let mut session = meganeura::build(&g, crate::support::gpu::inference_config()).0;
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
            let _ = meganeura::build(&g, crate::support::gpu::inference_config());
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
    let mut rnd = block_rng(seed);
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
    let mut rnd = block_rng(seed);
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
    let mut rnd = block_rng(seed);
    let mut b = vec![0u8; 110];
    for byte in b[..108].iter_mut() {
        *byte = (rnd() % 256) as u8;
    }
    b[108..110].copy_from_slice(&half::f16::from_f32(0.0042).to_bits().to_le_bytes());
    b
}

/// One Q6_K superblock with signed scales spanning both polarities and
/// quants across the full 6-bit range.
#[cfg(feature = "gguf")]
fn q6k_superblock(seed: u32) -> Vec<u8> {
    let mut rnd = block_rng(seed);
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
    let mut session = meganeura::build(&g, crate::support::gpu::inference_config()).0;
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

/// Default extraction concatenates SwiGLU gate/up into one matmul, so
/// `set_parameter_packed` has to restage that derived buffer from the two
/// uploads; uploading only the named sources leaves it uninitialized. Q3_K is
/// the interesting format: each source pads to a word, but that tail must not
/// land between the two sources' superblocks. Its uploads arrive in reverse
/// declaration order.
#[test]
#[cfg(feature = "gguf")]
fn swiglu_packed_concat_matches_reference() {
    use meganeura::load::gguf::{GgmlType, GgufTensor};

    type Build = fn(u32) -> Vec<u8>;
    let (m, k, n) = (2usize, 256usize, 4usize);
    // (format, superblock builder, gate and up seeds, upload `up` first)
    let cases: [(GgmlType, Build, u32, u32, bool); 2] = [
        (GgmlType::Q4K, q4k_superblock, 3, 19, false),
        (GgmlType::Q3K, q3k_superblock, 5, 23, true),
    ];
    for (ty, build, gate_seed, up_seed, up_first) in cases {
        let tensor = |seed: u32| {
            let data = (0..(k / 256 * n) as u32)
                .flat_map(|s| build(s + seed))
                .collect();
            GgufTensor::new(vec![k, n], ty, data)
        };
        let (gate_t, up_t) = (tensor(gate_seed), tensor(up_seed));
        let gate_ref = gate_t.to_f32().unwrap();
        let up_ref = up_t.to_f32().unwrap();
        let (_, gate_packed) = gate_t.to_packed().unwrap();
        let (_, up_packed) = up_t.to_packed().unwrap();

        let a: Vec<f32> = (0..m * k)
            .map(|i| ((i % 29) as f32 - 14.0) * 0.03)
            .collect();

        let mut g = Graph::new();
        let x = g.input("x", &[m, k]);
        let param = |g: &mut Graph, name: &str| match ty {
            GgmlType::Q4K => g.parameter_q4k(name, &[k, n]),
            _ => g.parameter_q3k(name, &[k, n]),
        };
        let gate_w = param(&mut g, "gate");
        let up_w = param(&mut g, "up");
        let gate = g.matmul(x, gate_w);
        let up = g.matmul(x, up_w);
        let out = g.swiglu(gate, up);
        g.set_outputs(vec![out]);
        let mut session = meganeura::build(&g, crate::support::gpu::inference_config()).0;
        assert!(
            session.has_parameter("gate+up"),
            "{ty:?}: expected SwiGLU concat fusion so packed upload restages the derived weight"
        );
        session.set_input("x", &a);
        let mut uploads = [("gate", &gate_packed), ("up", &up_packed)];
        if up_first {
            uploads.reverse();
        }
        for (name, packed) in uploads {
            session.set_parameter_packed(name, packed);
        }
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
            "{ty:?} SwiGLU produced non-finite values"
        );
        assert!(
            max_err / scale.max(1e-6) < 1e-4,
            "{ty:?} SwiGLU concat diverged from the reference: \
             max_abs_err={max_err} (scale {scale})"
        );
    }
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

/// The reduction extent has to fill whole 256-element superblocks.
#[test]
#[should_panic(expected = "multiple of 256")]
fn q4k_rejects_unaligned_reduction_extent() {
    let mut g = Graph::new();
    let _ = g.parameter_q4k("w", &[128, 4]);
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

    let mut session = meganeura::build(&g, crate::support::gpu::inference_config()).0;
    let undersized = vec![0.5_f32; 24]; // 24 floats == 96 bytes
    // Buffer is 48 floats == 192 bytes.  upload_buffer must panic with
    // "byte-size mismatch ... got 96, slot expects 192".
    session.set_parameter("bias", &undersized);
}
