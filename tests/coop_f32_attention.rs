//! F32 cooperative attention gradients, including tile tails, GQA, causal
//! masks and sliding windows. The f64 interpreter checks every gradient.
use meganeura::{CoopPolicy, Graph, build, compile::ShaderEntry, reference};

fn supported() -> bool {
    let gpu = crate::support::gpu::gpu();
    let caps = gpu.capabilities();
    caps.max_compute_shared_memory_size >= (4 * 1024 + 2 * 256 + 3 * 16) * 4
        && caps
            .cooperative_matrix
            .f32_shapes
            .iter()
            .any(|shape| matches!(*shape, [8, 8, 8] | [16, 16, 16]))
}

#[test]
fn coop_f32_gradients_match_f64_with_small_upstream_derivatives() {
    if !supported() {
        return;
    }
    for (q_seq, kv_seq, heads, kv_heads, window, causal) in [
        (128, 128, 1, 1, 0, false),
        (129, 145, 4, 2, 0, false),
        (129, 129, 3, 1, 0, true),
        (145, 145, 2, 2, 17, true),
        (129, 129, 2, 1, 1, true),
    ] {
        check_case(
            (q_seq, kv_seq, heads, kv_heads, window, causal),
            3e5,
            &[1e-12, 0.1],
            3e-5,
        );
    }
}

/// Manual stress check: the quadratic CPU f64 reference is too slow for CI.
/// Run explicitly on a GPU with 8x8 f32 cooperative matrices:
/// ```sh
/// cargo test --release --all-features --test regression \
///   coop_f32_attention::coop_f32_long_attention_gradients_match_f64 \
///   -- --ignored --exact
/// ```
#[test]
#[ignore = "expensive f64 reference; run manually with --ignored"]
fn coop_f32_long_attention_gradients_match_f64() {
    if !supported() {
        return;
    }
    // At 1500 rows this smooth, cancellation-heavy case gives the previous
    // scalar dQ an f64-relative L2 error of 4.2e-5 on the M3, and cooperative f32 dQ
    // 6.2e-5. Bound both accumulation orders, not their bitwise agreement.
    check_case((1500, 1500, 6, 6, 0, false), 0.8, &[0.01], 1e-4);
}

fn check_case(
    (q_seq, kv_seq, heads, kv_heads, window, causal): (usize, usize, u32, u32, u32, bool),
    value_scale: f32,
    upstream_scales: &[f32],
    tolerance: f64,
) {
    let q_width = heads as usize * 64;
    let kv_width = kv_heads as usize * 64;
    let mut graph = Graph::new();
    let q = graph.parameter("q", &[q_seq, q_width]);
    let k = graph.parameter("k", &[kv_seq, kv_width]);
    let v = graph.parameter("v", &[kv_seq, kv_width]);
    let attention = if window > 0 {
        graph.sliding_window_attention(q, k, v, heads, kv_heads, 64, window)
    } else if causal {
        graph.causal_attention(q, k, v, heads, kv_heads, 64)
    } else {
        graph.multi_head_attn(q, k, v, heads, kv_heads, 64, true)
    };
    let weights = graph.input("weights", &[q_seq, q_width]);
    let weighted = graph.mul(attention, weights);
    let loss = graph.sum_all(weighted);
    graph.set_outputs(vec![loss]);
    let values = |len: usize, frequency: f32, scale: f32| -> Vec<f32> {
        (0..len)
            .map(|i| (i as f32 * frequency + 0.7).sin() * scale)
            .collect()
    };
    let parameters = [
        ("q", values(q_seq * q_width, 0.017, 0.8)),
        ("k", values(kv_seq * kv_width, 0.023, 0.8)),
        ("v", values(kv_seq * kv_width, 0.031, value_scale)),
    ];
    let mut config = crate::support::gpu::config();
    config.runtime.coop = CoopPolicy::NativeF32;
    config.runtime.poison = true;
    config.tune = false;
    let (mut session, _) = build(&graph, config);
    assert!(
        session
            .plan()
            .dispatches
            .iter()
            .any(|dispatch| dispatch.shader == ShaderEntry::FlashGradKVCoopF32)
    );
    let backward = meganeura::autodiff::differentiate(&graph);
    assert!(
        session
            .plan()
            .dispatches
            .iter()
            .any(|dispatch| dispatch.shader == ShaderEntry::FlashGradQCoopF32)
    );
    for &scale in upstream_scales {
        let weights = values(q_seq * q_width, 0.019, scale);
        let mut feeds = reference::Feeds::new();
        for (name, data) in &parameters {
            feeds.set(name, data);
            session.set_parameter(name, data);
        }
        feeds.set("weights", &weights);
        session.set_input("weights", &weights);
        let expected = reference::evaluate_outputs(&backward, &feeds).unwrap();
        session.step();
        session.wait();
        for ((name, data), reference) in parameters.iter().zip(&expected[1..]) {
            let mut actual = vec![0.0; data.len()];
            session.read_param_grad(name, &mut actual);
            let mut error = 0.0;
            let mut magnitude = 0.0;
            for (&actual, &expected) in actual.iter().zip(&reference.data) {
                assert!(actual.is_finite());
                error += (f64::from(actual) - expected).powi(2);
                magnitude += expected.powi(2);
            }
            // A one-key window has mathematically zero dQ/dK. Bound
            // cancellation there relative to the unmasked operand scale.
            let floor = if window == 1 && *name != "v" {
                (f64::from(scale) * f64::from(value_scale)).powi(2) * data.len() as f64
            } else {
                0.0
            };
            let relative = (error / magnitude.max(floor)).sqrt();
            assert!(
                relative < tolerance,
                "{q_seq}/{kv_seq} h={heads}/{kv_heads} window={window} causal={causal} {name} scale={scale}: relative error {relative}"
            );
        }
    }
}

#[test]
fn coop_f32_forward_matches_f64_across_heads_masks_and_tails() {
    if !crate::support::gpu::gpu()
        .capabilities()
        .cooperative_matrix
        .f32_shapes
        .contains(&[16, 16, 16])
    {
        return;
    }
    for hd in [16, 32, 64, 128, 256] {
        for (qs, ks, heads, kv_heads, window, causal) in [
            (17, 19, 4, 2, 0, false),
            (33, 33, 3, 1, 0, true),
            (65, 65, 2, 1, 17, true),
        ] {
            let mut graph = Graph::new();
            let q = graph.input("q", &[qs, heads * hd]);
            let k = graph.input("k", &[ks, kv_heads * hd]);
            let v = graph.input("v", &[ks, kv_heads * hd]);
            let y = if window > 0 {
                graph.sliding_window_attention(
                    q,
                    k,
                    v,
                    heads as u32,
                    kv_heads as u32,
                    hd as u32,
                    window,
                )
            } else if causal {
                graph.causal_attention(q, k, v, heads as u32, kv_heads as u32, hd as u32)
            } else {
                graph.multi_head_attn(q, k, v, heads as u32, kv_heads as u32, hd as u32, true)
            };
            graph.set_outputs(vec![y]);
            let mut config = crate::support::gpu::inference_config();
            config.runtime.coop = CoopPolicy::NativeF32;
            config.runtime.poison = true;
            config.tune = false;
            let (mut session, _) = build(&graph, config);
            assert!(
                session
                    .plan()
                    .dispatches
                    .iter()
                    .any(|d| d.shader == ShaderEntry::FlashAttentionCoopF32)
            );
            let mut feeds = reference::Feeds::new();
            for (name, len, scale, phase) in [
                ("q", qs * heads * hd, 0.8, 0.3),
                ("k", ks * kv_heads * hd, 0.8, 0.7),
                ("v", ks * kv_heads * hd, 3.0e5, 1.1),
            ] {
                let values: Vec<f32> = (0..len)
                    .map(|i| (i as f32 * 0.017 + phase).sin() * scale)
                    .collect();
                feeds.set(name, &values);
                session.set_input(name, &values);
            }
            let expected = reference::evaluate_outputs(&graph, &feeds).unwrap();
            session.step();
            session.wait();
            let actual = session.read_output(qs * heads * hd);
            let mut error = 0.0;
            let mut norm = 0.0;
            for (&a, &b) in actual.iter().zip(&expected[0].data) {
                assert!(a.is_finite());
                error += (a as f64 - b).powi(2);
                norm += b.powi(2);
            }
            assert!(
                (error / norm).sqrt() < 1e-5,
                "hd={hd}, qs={qs}, ks={ks}, window={window}: {}",
                (error / norm).sqrt()
            );
        }
    }
}
