//! Short-sequence cooperative forward/backward parity on GPUs with f16 tiles.
//!
//! SmolVLA uses Q=50 with both KV=16 (cross attention) and KV=50 (self
//! attention). These shapes must not inherit the scalar flash kernel's much
//! larger BQ heuristic when selecting the independent BKV=16 coop kernel.

use std::sync::Arc;

use meganeura::{Graph, Mode, SessionConfig, build, compile::ShaderEntry};

fn run(
    gpu: Arc<blade_graphics::Context>,
    q_seq: usize,
    kv_seq: usize,
    window: u32,
    cooperative: bool,
) -> ([Vec<f32>; 3], bool) {
    unsafe {
        std::env::set_var(
            "MEGANEURA_FLASH_FWD_COOP",
            if cooperative { "1" } else { "0" },
        );
        std::env::set_var(
            "MEGANEURA_FLASH_BWD_COOP",
            if cooperative { "1" } else { "0" },
        );
    }

    let (num_heads, num_kv_heads, head_dim) = (3, 1, 64);
    let mut graph = Graph::new();
    let q = graph.parameter("q", &[q_seq, num_heads as usize * head_dim as usize]);
    let k = graph.parameter("k", &[kv_seq, num_kv_heads as usize * head_dim as usize]);
    let v = graph.parameter("v", &[kv_seq, num_kv_heads as usize * head_dim as usize]);
    let attention = if window == 0 {
        graph.multi_head_attn(q, k, v, num_heads, num_kv_heads, head_dim, q_seq != kv_seq)
    } else {
        graph.sliding_window_attention(q, k, v, num_heads, num_kv_heads, head_dim, window)
    };
    let loss = graph.mean_all(attention);
    graph.set_outputs(vec![loss]);

    let (mut session, _) = build(
        &graph,
        SessionConfig {
            mode: Mode::Training,
            gpu: Some(gpu),
            ..SessionConfig::from_env()
        },
    );
    let uses_coop = session
        .plan()
        .dispatches
        .iter()
        .any(|dispatch| dispatch.shader == ShaderEntry::FlashGradKVCoop);

    let q_data: Vec<f32> = (0..q_seq * num_heads as usize * head_dim as usize)
        .map(|i| ((i as f32 * 0.017) + 0.3).sin() * 0.1)
        .collect();
    let k_data: Vec<f32> = (0..kv_seq * num_kv_heads as usize * head_dim as usize)
        .map(|i| ((i as f32 * 0.019) + 0.7).sin() * 0.1)
        .collect();
    let v_data: Vec<f32> = (0..kv_seq * num_kv_heads as usize * head_dim as usize)
        .map(|i| ((i as f32 * 0.023) + 1.1).sin() * 0.1)
        .collect();
    session.set_parameter("q", &q_data);
    session.set_parameter("k", &k_data);
    session.set_parameter("v", &v_data);
    session.step();
    session.wait();

    let gradients = [
        ("q", q_data.len()),
        ("k", k_data.len()),
        ("v", v_data.len()),
    ]
    .map(|(name, len)| {
        let mut gradient = vec![0.0; len];
        session.read_param_grad(name, &mut gradient);
        gradient
    });
    (gradients, uses_coop)
}

fn assert_close(label: &str, scalar: &[f32], cooperative: &[f32]) {
    let max_abs = scalar
        .iter()
        .zip(cooperative)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    let scale = scalar
        .iter()
        .map(|value| value.abs())
        .fold(0.0_f32, f32::max)
        .max(1e-6);
    assert!(
        max_abs / scale < 0.02,
        "{label}: cooperative gradient differs from scalar by {:.3}% (max abs {max_abs:.3e})",
        max_abs / scale * 100.0,
    );
}

#[test]
fn short_cross_self_and_window_attention_gradients_match_scalar() {
    let gpu = Arc::new(
        meganeura::init_gpu_context_with(meganeura::GpuOptions::from_env()).expect("GPU context"),
    );
    let has_coop = gpu.capabilities().cooperative_matrix.f16_tile == 16;

    for (label, q_seq, kv_seq, window) in [
        ("cross", 50, 16, 0),
        ("self", 50, 50, 0),
        ("window", 50, 50, 17),
    ] {
        let (scalar, scalar_used_coop) = run(gpu.clone(), q_seq, kv_seq, window, false);
        let (cooperative, coop_used_coop) = run(gpu.clone(), q_seq, kv_seq, window, true);
        assert!(!scalar_used_coop);
        assert_eq!(coop_used_coop, has_coop);
        for (i, name) in ["dQ", "dK", "dV"].into_iter().enumerate() {
            assert_close(&format!("{label} {name}"), &scalar[i], &cooperative[i]);
        }
    }
}
