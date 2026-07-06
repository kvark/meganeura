//! Numerical correctness for the LLM temporal decoder *stack*
//! (`build_temporal_decoder_stack`): token embed → N temporal decoder layers →
//! final RMSNorm → weight-tied logits.
//!
//! Built with small random weights and zero rel-pos tables (plain causal
//! self-attention), run on the GPU, and compared against an independent CPU
//! reference. Extends `llm_decoder_correctness.rs` (single layer) to the
//! stacked forward: layer chaining, the embedding lookup, the final norm, and
//! the transpose-weight-tied logits projection.

use std::collections::HashMap;

use meganeura::Graph;
use meganeura::models::magenta_rt::llm::{LlmConfig, build_temporal_decoder_stack};

const LAYERS: usize = 2;

fn tiny_cfg() -> LlmConfig {
    LlmConfig {
        embed_dim: 16,
        head_dim: 8,
        num_heads: 2,
        mlp_dim: 32,
        num_encoder_layers: 1,
        num_temporal_decoder_layers: LAYERS as u32,
        num_depth_decoder_layers: 1,
        num_levels: 16,
        vocab_size: 12,
        encoder_seq_len: 4,
        decoder_seq_len: 3,
        rel_pos_num_buckets: 8,
        rel_pos_max_distance: 16,
        depth_rel_pos_num_buckets: 8,
        depth_rel_pos_max_distance: 16,
        layer_norm_eps: 1e-6,
    }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        let u = (x.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as u32;
        (u as f32 / (1u32 << 24) as f32 - 0.5) * 0.3
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next()).collect()
    }
}

fn matmul(x: &[f32], w: &[f32], rows: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; rows * n];
    for r in 0..rows {
        for j in 0..n {
            let mut acc = 0.0;
            for d in 0..k {
                acc += x[r * k + d] * w[d * n + j];
            }
            out[r * n + j] = acc;
        }
    }
    out
}

fn rms_norm(x: &[f32], scale: &[f32], rows: usize, dim: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0_f32; rows * dim];
    for r in 0..rows {
        let row = &x[r * dim..(r + 1) * dim];
        let ms = row.iter().map(|v| v * v).sum::<f32>() / dim as f32;
        let inv = 1.0 / (ms + eps).sqrt();
        for d in 0..dim {
            out[r * dim + d] = row[d] * inv * scale[d];
        }
    }
    out
}

fn gelu_tanh(v: f32) -> f32 {
    let c = (2.0_f32 / std::f32::consts::PI).sqrt();
    0.5 * v * (1.0 + (c * (v + 0.044715 * v * v * v)).tanh())
}

#[allow(clippy::too_many_arguments)]
fn sdpa(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    q_seq: usize,
    kv_seq: usize,
    heads: usize,
    head_dim: usize,
    causal: bool,
) -> Vec<f32> {
    let dim = heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0_f32; q_seq * dim];
    for h in 0..heads {
        let off = h * head_dim;
        for i in 0..q_seq {
            let limit = if causal { i + 1 } else { kv_seq };
            let mut scores = vec![0.0_f32; limit];
            for (j, sc) in scores.iter_mut().enumerate() {
                let mut s = 0.0;
                for d in 0..head_dim {
                    s += q[i * dim + off + d] * k[j * dim + off + d];
                }
                *sc = s * scale;
            }
            let mx = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0;
            for s in scores.iter_mut() {
                *s = (*s - mx).exp();
                sum += *s;
            }
            for d in 0..head_dim {
                let mut acc = 0.0;
                for (j, &sc) in scores.iter().enumerate() {
                    acc += (sc / sum) * v[j * dim + off + d];
                }
                out[i * dim + off + d] = acc;
            }
        }
    }
    out
}

/// CPU reference for one temporal decoder layer (zero rel-pos bias).
fn layer_ref(
    cfg: &LlmConfig,
    w: &HashMap<String, Vec<f32>>,
    prefix: &str,
    x_in: &[f32],
    enc: &[f32],
    dec_seq: usize,
    enc_seq: usize,
) -> Vec<f32> {
    let embed = cfg.embed_dim as usize;
    let attn_dim = (cfg.num_heads * cfg.head_dim) as usize;
    let mlp = cfg.mlp_dim as usize;
    let heads = cfg.num_heads as usize;
    let hd = cfg.head_dim as usize;
    let eps = cfg.layer_norm_eps;
    let g = |n: &str| -> &[f32] { w.get(&format!("{prefix}.{n}")).unwrap().as_slice() };
    let mut x = x_in.to_vec();

    let h = rms_norm(&x, g("pre_self_attn_norm"), dec_seq, embed, eps);
    let q = matmul(&h, g("self_attn.q"), dec_seq, embed, attn_dim);
    let k = matmul(&h, g("self_attn.k"), dec_seq, embed, attn_dim);
    let v = matmul(&h, g("self_attn.v"), dec_seq, embed, attn_dim);
    let sa = sdpa(&q, &k, &v, dec_seq, dec_seq, heads, hd, true);
    let sa_out = matmul(&sa, g("self_attn.o"), dec_seq, attn_dim, embed);
    for i in 0..dec_seq * embed {
        x[i] += sa_out[i];
    }

    let h = rms_norm(&x, g("pre_cross_attn_norm"), dec_seq, embed, eps);
    let cq = matmul(&h, g("cross_attn.q"), dec_seq, embed, attn_dim);
    let ck = matmul(enc, g("cross_attn.k"), enc_seq, embed, attn_dim);
    let cv = matmul(enc, g("cross_attn.v"), enc_seq, embed, attn_dim);
    let ca = sdpa(&cq, &ck, &cv, dec_seq, enc_seq, heads, hd, false);
    let ca_out = matmul(&ca, g("cross_attn.o"), dec_seq, attn_dim, embed);
    for i in 0..dec_seq * embed {
        x[i] += ca_out[i];
    }

    let h = rms_norm(&x, g("pre_mlp_norm"), dec_seq, embed, eps);
    let gate = matmul(&h, g("mlp.w_gate"), dec_seq, embed, mlp);
    let up = matmul(&h, g("mlp.w_up"), dec_seq, embed, mlp);
    let ffn: Vec<f32> = gate.iter().zip(up.iter()).map(|(&gt, &u)| gelu_tanh(gt) * u).collect();
    let down = matmul(&ffn, g("mlp.w_down"), dec_seq, mlp, embed);
    for i in 0..dec_seq * embed {
        x[i] += down[i];
    }
    x
}

fn layer_weights(cfg: &LlmConfig, w: &mut HashMap<String, Vec<f32>>, prefix: &str, r: &mut Rng) {
    let embed = cfg.embed_dim as usize;
    let attn_dim = (cfg.num_heads * cfg.head_dim) as usize;
    let mlp = cfg.mlp_dim as usize;
    for nm in ["pre_self_attn_norm", "pre_cross_attn_norm", "pre_mlp_norm"] {
        w.insert(format!("{prefix}.{nm}"), (0..embed).map(|_| 1.0 + r.next()).collect());
    }
    for kind in ["self_attn", "cross_attn"] {
        for n in ["q", "k", "v"] {
            w.insert(format!("{prefix}.{kind}.{n}"), r.vec(embed * attn_dim));
        }
        w.insert(format!("{prefix}.{kind}.o"), r.vec(attn_dim * embed));
    }
    w.insert(format!("{prefix}.mlp.w_gate"), r.vec(embed * mlp));
    w.insert(format!("{prefix}.mlp.w_up"), r.vec(embed * mlp));
    w.insert(format!("{prefix}.mlp.w_down"), r.vec(mlp * embed));
}

#[test]
fn temporal_decoder_stack_matches_cpu_reference() {
    let cfg = tiny_cfg();
    let embed = cfg.embed_dim as usize;
    let vocab = cfg.vocab_size as usize;
    let dec_seq = cfg.decoder_seq_len as usize;
    let enc_seq = cfg.encoder_seq_len as usize;

    let mut r = Rng(0xFACEFEED01);
    let mut weights: HashMap<String, Vec<f32>> = HashMap::new();
    weights.insert("shared_token_embedder".into(), r.vec(vocab * embed));
    weights.insert("decoder.decoder_norm".into(), (0..embed).map(|_| 1.0 + r.next()).collect());
    for i in 0..LAYERS {
        layer_weights(&cfg, &mut weights, &format!("decoder.temporal_layers.{i}"), &mut r);
    }
    let tokens: Vec<u32> = vec![1, 5, 2];
    let enc = r.vec(enc_seq * embed);

    // GPU.
    let mut g = Graph::new();
    let tok_node = g.input_u32("dec_tokens", &[dec_seq]);
    let enc_node = g.input("enc_out", &[enc_seq, embed]);
    let logits = build_temporal_decoder_stack(&mut g, &cfg, tok_node, enc_node);
    g.set_outputs(vec![logits]);
    let mut s = meganeura::build_inference_session(&g);
    for (name, data) in &weights {
        s.set_parameter(name, data);
    }
    let n_buckets = (cfg.num_heads * cfg.rel_pos_num_buckets) as usize;
    for i in 0..LAYERS {
        s.set_parameter(
            &format!("decoder.temporal_layers.{i}.self_attn.rel_pos_bias_table"),
            &vec![0.0_f32; n_buckets],
        );
    }
    s.set_input_u32("dec_tokens", &tokens);
    s.set_input("enc_out", &enc);
    s.step();
    s.wait();
    let gpu = s.read_output(dec_seq * vocab);

    // CPU reference.
    let table = &weights["shared_token_embedder"];
    let mut x = vec![0.0_f32; dec_seq * embed];
    for (si, &t) in tokens.iter().enumerate() {
        x[si * embed..(si + 1) * embed]
            .copy_from_slice(&table[t as usize * embed..(t as usize + 1) * embed]);
    }
    for i in 0..LAYERS {
        x = layer_ref(&cfg, &weights, &format!("decoder.temporal_layers.{i}"), &x, &enc, dec_seq, enc_seq);
    }
    x = rms_norm(&x, &weights["decoder.decoder_norm"], dec_seq, embed, cfg.layer_norm_eps);
    // Weight-tied logits: logits[s, vc] = dot(x[s], table[vc]).
    let mut cpu = vec![0.0_f32; dec_seq * vocab];
    for si in 0..dec_seq {
        for vc in 0..vocab {
            let mut acc = 0.0;
            for d in 0..embed {
                acc += x[si * embed + d] * table[vc * embed + d];
            }
            cpu[si * vocab + vc] = acc;
        }
    }

    assert_eq!(gpu.len(), cpu.len());
    let mut max_abs = 0.0_f32;
    for (i, (&a, &b)) in gpu.iter().zip(cpu.iter()).enumerate() {
        let d = (a - b).abs();
        max_abs = max_abs.max(d);
        assert!(d <= 1e-3 * b.abs().max(1.0), "elem {i}: gpu {a} vs cpu {b} (diff {d})");
    }
    eprintln!("temporal stack max abs diff GPU vs CPU: {max_abs:.2e}");
}
