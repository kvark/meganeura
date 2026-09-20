//! A session may contain attention blocks with different head widths (for
//! example, a multimodal vision encoder and text decoder). Each generated
//! attention pipeline must be specialized independently.

use meganeura::Graph;

fn values(len: usize, phase: f32) -> Vec<f32> {
    (0..len)
        .map(|i| ((i as f32 * 0.173) + phase).sin() * 0.4)
        .collect()
}

fn run_single(head_dim: usize, q: &[f32], k: &[f32], v: &[f32]) -> Vec<f32> {
    let seq = q.len() / head_dim;
    let mut graph = Graph::new();
    let qn = graph.input("q", &[seq, head_dim]);
    let kn = graph.input("k", &[seq, head_dim]);
    let vn = graph.input("v", &[seq, head_dim]);
    let out = graph.full_attention(qn, kn, vn, 1, 1, head_dim as u32);
    graph.set_outputs(vec![out]);

    let mut session = meganeura::build(&graph, meganeura::SessionConfig::inference_from_env()).0;
    session.set_input("q", q);
    session.set_input("k", k);
    session.set_input("v", v);
    session.step();
    session.wait();
    session.read_output(q.len())
}

#[test]
fn mixed_head_dims_match_independent_sessions() {
    let seq = 4;
    let (hd_a, hd_b) = (4, 8);
    let qa = values(seq * hd_a, 0.1);
    let ka = values(seq * hd_a, 0.7);
    let va = values(seq * hd_a, 1.3);
    let qb = values(seq * hd_b, 0.2);
    let kb = values(seq * hd_b, 0.8);
    let vb = values(seq * hd_b, 1.4);
    let expected_a = run_single(hd_a, &qa, &ka, &va);
    let expected_b = run_single(hd_b, &qb, &kb, &vb);

    let mut graph = Graph::new();
    let qa_node = graph.input("qa", &[seq, hd_a]);
    let ka_node = graph.input("ka", &[seq, hd_a]);
    let va_node = graph.input("va", &[seq, hd_a]);
    let qb_node = graph.input("qb", &[seq, hd_b]);
    let kb_node = graph.input("kb", &[seq, hd_b]);
    let vb_node = graph.input("vb", &[seq, hd_b]);
    let out_a = graph.full_attention(qa_node, ka_node, va_node, 1, 1, hd_a as u32);
    let out_b = graph.full_attention(qb_node, kb_node, vb_node, 1, 1, hd_b as u32);
    graph.set_outputs(vec![out_a, out_b]);

    let mut session = meganeura::build(&graph, meganeura::SessionConfig::inference_from_env()).0;
    for (name, data) in [
        ("qa", qa.as_slice()),
        ("ka", ka.as_slice()),
        ("va", va.as_slice()),
        ("qb", qb.as_slice()),
        ("kb", kb.as_slice()),
        ("vb", vb.as_slice()),
    ] {
        session.set_input(name, data);
    }
    session.step();
    session.wait();
    let mut actual_a = vec![0.0; expected_a.len()];
    let mut actual_b = vec![0.0; expected_b.len()];
    session.read_output_by_index(0, &mut actual_a);
    session.read_output_by_index(1, &mut actual_b);

    for (label, expected, actual) in [
        ("head_dim=4", expected_a, actual_a),
        ("head_dim=8", expected_b, actual_b),
    ] {
        let max_abs = expected
            .iter()
            .zip(&actual)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0_f32, f32::max);
        assert!(max_abs < 1e-5, "{label} pipeline mismatch: {max_abs}");
    }
}

#[test]
#[ignore = "cooperative attention source experiment; run on a GPU"]
fn cooperative_attention_matches_full_reference_with_masks_and_tails() {
    use meganeura::{codegen, compile};
    let gpu = std::sync::Arc::new(meganeura::runtime::init_gpu_context().unwrap());
    for (q_len, kv_len, hd, causal, window) in [
        (33, 49, 16, false, 0),
        (49, 49, 64, true, 0),
        (50, 50, 128, true, 17),
        (50, 50, 64, true, 1),
    ] {
        let (heads, kv_heads) = (3, 1);
        let q = values(q_len * heads * hd, 0.3);
        let k = values(kv_len * kv_heads * hd, 1.3);
        let v = values(kv_len * kv_heads * hd, 2.3);
        let mut expected = vec![0.0f64; q.len()];
        for row in 0..q_len {
            let end = if causal { row + 1 } else { kv_len };
            let begin = if window == 0 {
                0
            } else {
                end.saturating_sub(window)
            };
            for head in 0..heads {
                let kv_head = head / (heads / kv_heads);
                let scores: Vec<f64> = (begin..end)
                    .map(|key| {
                        (0..hd)
                            .map(|d| {
                                f64::from(q[(row * heads + head) * hd + d])
                                    * f64::from(k[(key * kv_heads + kv_head) * hd + d])
                            })
                            .sum::<f64>()
                            / (hd as f64).sqrt()
                    })
                    .collect();
                let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                let probabilities: Vec<_> =
                    scores.iter().map(|score| (score - maximum).exp()).collect();
                let sum: f64 = probabilities.iter().sum();
                for d in 0..hd {
                    expected[(row * heads + head) * hd + d] = probabilities
                        .iter()
                        .enumerate()
                        .map(|(key, p)| {
                            p / sum * f64::from(v[((begin + key) * kv_heads + kv_head) * hd + d])
                        })
                        .sum();
                }
            }
        }
        for (cooperative, query_tiles) in [(false, 0), (true, 0), (true, 1), (true, 2), (true, 4)] {
            if query_tiles > 0
                && (gpu.capabilities().fixed_compute_subgroup_size != Some(32)
                    || gpu.capabilities().cooperative_matrix.f16_tile != 16
                    || !codegen::cooperative_attention_tile_is_legal(hd as u32, query_tiles))
            {
                continue;
            }
            let mut graph = Graph::new();
            let qn = graph.input("q", &[q_len, heads * hd]);
            let kn = graph.input("k", &[kv_len, kv_heads * hd]);
            let vn = graph.input("v", &[kv_len, kv_heads * hd]);
            let output = if window != 0 {
                graph.sliding_window_attention(
                    qn,
                    kn,
                    vn,
                    heads as u32,
                    kv_heads as u32,
                    hd as u32,
                    window as u32,
                )
            } else if causal {
                graph.causal_attention(qn, kn, vn, heads as u32, kv_heads as u32, hd as u32)
            } else {
                graph.cross_attention(qn, kn, vn, heads as u32, kv_heads as u32, hd as u32)
            };
            graph.set_outputs(vec![output]);
            let advertised = gpu.capabilities().cooperative_matrix;
            let mut plan = compile::compile_with_caps(
                &graph,
                &compile::CompileOptions {
                    flash_forward_coop: cooperative,
                    ..Default::default()
                },
                codegen::CoopCaps {
                    f16_tile: advertised.f16_tile,
                    f32_tile: advertised.f32_tile,
                },
            );
            if query_tiles > 0 {
                for d in &mut plan.dispatches {
                    if d.shader == compile::ShaderEntry::FlashAttentionCoop {
                        d.kernel = compile::Kernel::CooperativeAttention { query_tiles };
                        d.workgroups[0] = d.params[0].div_ceil(16 * query_tiles);
                    }
                }
            }
            let mut session = meganeura::Session::with_context(plan, gpu.clone());
            assert_eq!(
                session
                    .plan()
                    .dispatches
                    .iter()
                    .any(|d| d.shader == compile::ShaderEntry::FlashAttentionCoop),
                cooperative && advertised.f16_tile == 16
            );
            session.set_input("q", &q);
            session.set_input("k", &k);
            session.set_input("v", &v);
            session.step();
            session.wait();
            let actual = session.read_output(q.len());
            let error: f64 = actual
                .iter()
                .zip(&expected)
                .map(|(&a, &b)| (f64::from(a) - b).powi(2))
                .sum();
            let norm: f64 = expected.iter().map(|v| v * v).sum();
            let relative = (error / norm.max(1e-24)).sqrt();
            assert!(
                actual.iter().all(|v| v.is_finite()) && relative < 0.002,
                "Q={q_len}, KV={kv_len}, HD={hd}, causal={causal}, window={window}, cooperative={cooperative}, query_tiles={query_tiles}: relL2={relative}"
            );
        }
    }
}
