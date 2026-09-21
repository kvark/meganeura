use meganeura::{CoopPolicy, Graph, Mode, SessionConfig};
use std::sync::Arc;

fn values(n: usize, seed: u32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let mut x = (i as u32)
                .wrapping_add(seed)
                .wrapping_mul(747796405)
                .wrapping_add(2891336453);
            x = ((x >> ((x >> 28) + 4)) ^ x).wrapping_mul(277803737);
            ((x >> 22) ^ x) as f32 / u32::MAX as f32 * 4.0 - 2.0
        })
        .collect()
}

fn main() {
    let gpu =
        Arc::new(meganeura::init_gpu_context_with(meganeura::GpuOptions::from_env()).unwrap());
    let mut args = std::env::args().skip(1);
    let threads: u32 = args.next().unwrap_or("256".into()).parse().unwrap();
    let keys: u32 = args.next().unwrap_or("8".into()).parse().unwrap();
    let heads = if args.any(|arg| arg == "--wide") { vec![1024usize] } else { vec![32, 64, 128, 256] };
    for hd in heads {
        for mask in [0, 1, 7] {
            let (q_len, kv_len, heads, kv_heads) =
                (257usize, if mask == 0 { 263 } else { 257 }, 4usize, 2usize);
            let q_data = values(q_len * heads * hd, 43);
            let k_data = values(kv_len * kv_heads * hd, 1231);
            let v_data = values(kv_len * kv_heads * hd, 7521);
            let upstream = values(q_data.len(), 73515);
            let mut reference = vec![0.0; q_data.len()];
            let mut gradients = [
                vec![0.0; q_data.len()],
                vec![0.0; k_data.len()],
                vec![0.0; v_data.len()],
            ];
            for pos in 0..q_len {
                let end = if mask == 0 { kv_len } else { pos + 1 };
                let start = if mask > 1 {
                    end.saturating_sub(mask)
                } else {
                    0
                };
                for head in 0..heads {
                    let q_base = (pos * heads + head) * hd;
                    let kv_head = head / (heads / kv_heads);
                    let mut scores: Vec<f64> = (start..end)
                        .map(|key| {
                            let k_base = (key * kv_heads + kv_head) * hd;
                            (0..hd)
                                .map(|d| {
                                    f64::from(q_data[q_base + d]) * f64::from(k_data[k_base + d])
                                })
                                .sum::<f64>()
                                / (hd as f64).sqrt()
                        })
                        .collect();
                    let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
                    for score in &mut scores {
                        *score = (*score - max).exp();
                    }
                    let sum: f64 = scores.iter().sum();
                    for score in &mut scores {
                        *score /= sum;
                    }
                    let mut dp = vec![0.0; scores.len()];
                    for (offset, &p) in scores.iter().enumerate() {
                        let v_base = ((start + offset) * kv_heads + kv_head) * hd;
                        for d in 0..hd {
                            let dout = f64::from(upstream[q_base + d]) / q_data.len() as f64;
                            reference[q_base + d] += p * f64::from(v_data[v_base + d]);
                            dp[offset] += dout * f64::from(v_data[v_base + d]);
                            gradients[2][v_base + d] += p * dout;
                        }
                    }
                    let correction: f64 = dp.iter().zip(&scores).map(|(dp, p)| dp * p).sum();
                    for (offset, &p) in scores.iter().enumerate() {
                        let k_base = ((start + offset) * kv_heads + kv_head) * hd;
                        let ds = p * (dp[offset] - correction) / (hd as f64).sqrt();
                        for d in 0..hd {
                            gradients[0][q_base + d] += ds * f64::from(k_data[k_base + d]);
                            gradients[1][k_base + d] += ds * f64::from(q_data[q_base + d]);
                        }
                    }
                }
            }
            for ept in [8, 16, 32] {
                for interleave in [false, true] {
                    for mode in [Mode::Inference, Mode::Training] {
                        let mut graph = Graph::new();
                        let q = graph.parameter("q", &[q_len, heads * hd]);
                        let k = graph.parameter("k", &[kv_len, kv_heads * hd]);
                        let v = graph.parameter("v", &[kv_len, kv_heads * hd]);
                        let output = if mask == 0 {
                            graph.multi_head_attn(
                                q,
                                k,
                                v,
                                heads as u32,
                                kv_heads as u32,
                                hd as u32,
                                true,
                            )
                        } else if mask == 1 {
                            graph.causal_attention(
                                q,
                                k,
                                v,
                                heads as u32,
                                kv_heads as u32,
                                hd as u32,
                            )
                        } else {
                            graph.sliding_window_attention(
                                q,
                                k,
                                v,
                                heads as u32,
                                kv_heads as u32,
                                hd as u32,
                                mask as u32,
                            )
                        };
                        let root = if mode == Mode::Training {
                            let weight = graph.input("upstream", &[q_len, heads * hd]);
                            let weighted = graph.mul(output, weight);
                            graph.mean_all(weighted)
                        } else {
                            output
                        };
                        graph.set_outputs(vec![root]);
                        let mut config = SessionConfig {
                            mode,
                            gpu: Some(gpu.clone()),
                            ..Default::default()
                        };
                        config.options.knobs.flash_ept_cap = ept;
                        config.options.knobs.flash = meganeura::codegen::FlashAttentionShape {
                            threads,
                            keys,
                            interleave,
                        };
                        config.options.flash_forward_coop = false;
                        config.runtime.coop = CoopPolicy::Disabled;
                        let mut session = meganeura::build(&graph, config).0;
                        assert!(
                            session.plan().dispatches.iter().any(
                                |d| d.shader == meganeura::compile::ShaderEntry::FlashAttention
                            )
                        );
                        session.set_parameter("q", &q_data);
                        session.set_parameter("k", &k_data);
                        session.set_parameter("v", &v_data);
                        if mode == Mode::Training {
                            session.set_input("upstream", &upstream);
                        }
                        session.step();
                        session.wait();
                        let mut max_error = 0.0f64;
                        if mode == Mode::Inference {
                            for (actual, expected) in session
                                .read_output(reference.len())
                                .into_iter()
                                .zip(&reference)
                            {
                                assert!(
                                    actual.is_finite()
                                        && (f64::from(actual) - expected).abs() < 2e-6,
                                    "mask {mask} ept {ept}: {actual} != {expected}"
                                );
                                max_error = max_error.max((f64::from(actual) - expected).abs());
                            }
                        } else {
                            for (name, expected) in ["q", "k", "v"].iter().zip(&gradients) {
                                let mut actual = vec![0.0; expected.len()];
                                session.read_param_grad(name, &mut actual);
                                for (&actual, &expected) in actual.iter().zip(expected) {
                                    assert!(
                                        actual.is_finite()
                                            && (f64::from(actual) - expected).abs() < 2e-8,
                                        "gradient {name} mask {mask} ept {ept}: {actual} != {expected}"
                                    );
                                    max_error = max_error.max((f64::from(actual) - expected).abs());
                                }
                            }
                        }
                        println!(
                            "threads={threads} keys={keys} hd={hd} mask={mask} ept={ept} interleave={interleave} mode={mode:?} max_error={max_error:.3e}"
                        );
                    }
                }
            }
        }
    }
}
