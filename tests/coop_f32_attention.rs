//! Cooperative attention gradients, including tile tails, GQA, causal
//! masks and sliding windows. The f64 interpreter checks every gradient.
use meganeura::{CoopPolicy, Graph, build, compile::ShaderEntry, reference};

use meganeura::kernels::attention_grad::{AttentionGrad, Operands, Part, Path};

#[derive(Clone, Copy, Debug)]
enum Keys {
    Sinusoid,
    Dyadic(f32),
    SharedSinusoid(f32),
}

const COOP_F32: Path = Path::Cooperative(Operands::F32);

fn grad(part: Part, path: Path) -> ShaderEntry {
    ShaderEntry::AttentionGrad(AttentionGrad::new(part, path))
}

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
            64,
            3e5,
            &[1e-12, 0.1],
            3e-5,
            COOP_F32,
            Keys::Sinusoid,
        );
    }
}

#[test]
fn scaled_f16_gradients_keep_small_upstream_values_across_masks_and_tails() {
    let gpu = crate::support::gpu::gpu();
    if meganeura::runtime::auto_tune(&gpu, 0)
        .coop_caps
        .f16_attention_rows
        == 0
    {
        return;
    }
    for head_dim in [16, 32, 64, 128] {
        for shape in [
            (7, 5, 2, 1, 0, false),
            (33, 19, 3, 1, 0, false),
            (65, 65, 3, 1, 0, true),
            (45, 45, 2, 2, 17, true),
            (17, 17, 2, 1, 1, true),
        ] {
            check_case(
                shape,
                head_dim,
                3e5,
                &[1e-30, 1e-12, 0.1],
                0.01,
                Path::Cooperative(Operands::F16),
                Keys::Sinusoid,
            );
        }
    }
    for head_dim in [16, 64, 128] {
        for keys in [
            Keys::Dyadic(0.0),
            Keys::Dyadic(16.0),
            Keys::SharedSinusoid(0.0),
            Keys::SharedSinusoid(16.0),
        ] {
            check_case(
                (33, 19, 3, 1, 0, false),
                head_dim,
                3e5,
                &[1e-30, 1e-12, 0.1],
                0.01,
                Path::Cooperative(Operands::F16),
                keys,
            );
        }
    }
}

/// Manual stress check: the quadratic CPU f64 reference is too slow for CI.
/// Run explicitly on a GPU with native f32 cooperative matrices:
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
    check_case(
        (1500, 1500, 6, 6, 0, false),
        64,
        0.8,
        &[0.01],
        1e-4,
        COOP_F32,
        Keys::Sinusoid,
    );
}

fn check_case(
    (q_seq, kv_seq, heads, kv_heads, window, causal): (usize, usize, u32, u32, u32, bool),
    head_dim: u32,
    value_scale: f32,
    upstream_scales: &[f32],
    tolerance: f64,
    path: Path,
    keys: Keys,
) {
    check_case_with_forward(
        (q_seq, kv_seq, heads, kv_heads, window, causal),
        head_dim,
        value_scale,
        upstream_scales,
        tolerance,
        (path, false),
        keys,
    );
}

#[test]
fn f16_products_lse_and_output_support_independent_backward_paths() {
    let gpu = crate::support::gpu::gpu();
    if meganeura::runtime::auto_tune(&gpu, 0)
        .coop_caps
        .f16_attention_rows
        == 0
    {
        return;
    }
    for path in [Path::Rowwise, Path::Cooperative(Operands::F16)] {
        for shape in [
            (7, 5, 2, 1, 0, false),
            (33, 19, 3, 1, 0, false),
            (65, 65, 3, 1, 0, true),
            (45, 45, 2, 2, 17, true),
        ] {
            check_case_with_forward(
                shape,
                64,
                0.4,
                &[1e-12, 0.1],
                0.01,
                (path, true),
                Keys::Sinusoid,
            );
        }
    }
    check_case_with_forward(
        (33, 19, 3, 1, 0, false),
        48,
        0.4,
        &[0.1],
        0.01,
        (Path::Rowwise, true),
        Keys::Sinusoid,
    );
}

fn check_case_with_forward(
    (q_seq, kv_seq, heads, kv_heads, window, causal): (usize, usize, u32, u32, u32, bool),
    head_dim: u32,
    value_scale: f32,
    upstream_scales: &[f32],
    tolerance: f64,
    (path, forward_products): (Path, bool),
    keys: Keys,
) {
    let q_width = heads as usize * head_dim as usize;
    let kv_width = kv_heads as usize * head_dim as usize;
    let mut graph = Graph::new();
    let q = graph.parameter("q", &[q_seq, q_width]);
    let k = graph.parameter("k", &[kv_seq, kv_width]);
    let v = graph.parameter("v", &[kv_seq, kv_width]);
    let attention = if window > 0 {
        graph.sliding_window_attention(q, k, v, heads, kv_heads, head_dim, window)
    } else if causal {
        graph.causal_attention(q, k, v, heads, kv_heads, head_dim)
    } else {
        graph.multi_head_attn(q, k, v, heads, kv_heads, head_dim, true)
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
    let q = match keys {
        Keys::Dyadic(_) => (0..q_seq * q_width)
            .map(|i| ((i % 13) as f32 - 6.0) / 32.0)
            .collect(),
        _ => values(q_seq * q_width, 0.017, 0.8),
    };
    let k = match keys {
        Keys::Sinusoid => values(kv_seq * kv_width, 0.023, 0.8),
        Keys::Dyadic(offset) => (0..kv_seq * kv_width)
            .map(|i| offset + ((i % 11) as f32 - 5.0) / 64.0)
            .collect(),
        Keys::SharedSinusoid(offset) => values(kv_seq * kv_width, 0.023, 0.05)
            .into_iter()
            .map(|v| offset + v)
            .collect(),
    };
    let parameters = [
        ("q", q),
        ("k", k),
        ("v", values(kv_seq * kv_width, 0.031, value_scale)),
    ];
    let mut config = crate::support::gpu::config();
    config.runtime.coop = if path == COOP_F32 {
        CoopPolicy::NativeF32
    } else {
        CoopPolicy::Auto
    };
    config.runtime.poison = true;
    config.tune = false;
    config.options.flash_backward_coop = path == Path::Cooperative(Operands::F16);
    config.options.flash_forward_coop = false;
    config.options.prefer_attention_grad = Some(path);
    if path == COOP_F32
        && crate::support::gpu::gpu()
            .capabilities()
            .cooperative_matrix
            .f32_shapes
            .contains(&[16, 16, 16])
    {
        graph.nodes_mut()[attention as usize].attention_impl =
            Some(meganeura::graph::AttentionImpl::CooperativeF32);
        config.optimize.mode = meganeura::optimize::OptimizeMode::Off;
    }
    let runtime = config.runtime.clone();
    let (mut session, _) = build(&graph, config);
    if forward_products {
        let mut plan = session.plan().clone();
        for d in &mut plan.dispatches {
            if d.shader.is_attention() {
                d.shader = ShaderEntry::FlashAttentionProducts;
                d.workgroups = [(q_seq as u32).div_ceil(16), heads, 1];
            }
        }
        drop(session);
        session = meganeura::Session::with_context_opts(
            plan,
            crate::support::gpu::gpu().clone(),
            runtime,
        );
        assert!(
            session
                .plan()
                .dispatches
                .iter()
                .any(|d| d.shader == ShaderEntry::FlashAttentionProducts)
        );
    }
    assert!(
        session
            .plan()
            .dispatches
            .iter()
            .any(|dispatch| dispatch.shader == grad(Part::KV, path))
    );
    let backward = meganeura::autodiff::differentiate(&graph);
    assert!(
        session
            .plan()
            .dispatches
            .iter()
            .any(|dispatch| dispatch.shader == grad(Part::Q, path))
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
                "{q_seq}/{kv_seq} h={heads}/{kv_heads} window={window} causal={causal} head_dim={head_dim} {name} scale={scale}, keys={keys:?}: relative error {relative}"
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
            (16, 19, 2, 1, 0, false),
            (17, 17, 2, 1, 0, true),
            (129, 19, 4, 2, 0, false),
            (145, 145, 3, 1, 0, true),
            (161, 161, 2, 1, 17, true),
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
            graph.nodes_mut()[y as usize].attention_impl =
                Some(meganeura::graph::AttentionImpl::CooperativeF32);
            let mut config = crate::support::gpu::inference_config();
            config.runtime.coop = CoopPolicy::NativeF32;
            config.runtime.poison = true;
            config.tune = false;
            config.optimize.mode = meganeura::optimize::OptimizeMode::Off;
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

#[test]
fn f16_products_forward_matches_f64_across_padded_heads_and_masks() {
    use meganeura::{Session, SessionOptions};
    let gpu = crate::support::gpu::gpu();
    if !matches!(
        meganeura::runtime::auto_tune(&gpu, 0)
            .coop_caps
            .f16_attention_rows,
        8 | 16
    ) {
        return;
    }
    for (qs, ks, hd, heads, kv_heads, window, causal) in [
        (7, 5, 16, 2, 1, 0, false),
        (33, 19, 48, 3, 1, 0, false),
        (65, 65, 80, 3, 1, 0, true),
        (45, 45, 128, 2, 2, 17, true),
        (17, 17, 64, 2, 1, 1, true),
        (33, 19, 5, 3, 1, 0, false),
        (128, 128, 64, 9, 3, 0, true),
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
        config.tune = false;
        config.optimize.mode = meganeura::optimize::OptimizeMode::Off;
        let (baseline, _) = build(&graph, config);
        let mut plan = baseline.plan().clone();
        drop(baseline);
        for d in &mut plan.dispatches {
            if d.shader.is_attention() {
                d.shader = ShaderEntry::FlashAttentionProducts;
                d.workgroups = [(qs as u32).div_ceil(16), heads as u32, 1];
            }
        }
        let mut session = Session::with_context_opts(
            plan,
            gpu.clone(),
            SessionOptions {
                poison: true,
                ..Default::default()
            },
        );
        assert!(
            session
                .plan()
                .dispatches
                .iter()
                .any(|d| d.shader == ShaderEntry::FlashAttentionProducts)
        );
        let mut feeds = reference::Feeds::new();
        for (name, len, scale, phase) in [
            ("q", qs * heads * hd, 0.8, 0.3),
            ("k", ks * kv_heads * hd, 0.8, 0.7),
            ("v", ks * kv_heads * hd, 0.4, 1.1),
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
        let (mut err, mut norm, mut maximum, mut reference_maximum) = (0., 0., 0f64, 0f64);
        for (&a, &b) in actual.iter().zip(&expected[0].data) {
            assert!(a.is_finite());
            err += (f64::from(a) - b).powi(2);
            norm += b * b;
            maximum = maximum.max((f64::from(a) - b).abs());
            reference_maximum = reference_maximum.max(b.abs());
        }
        assert!(
            (err / norm.max(1e-300)).sqrt() < 0.01 && maximum < 1e-3 + 0.01 * reference_maximum,
            "q={qs}, k={ks}, d={hd}, window={window}: relL2={}, max={maximum}",
            (err / norm).sqrt()
        );
    }
}
