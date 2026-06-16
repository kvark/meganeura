//! Numerical correctness for the LLM temporal decoder layer.
//!
//! Builds one `build_decoder_layer` (causal self-attention + cross-attention to
//! the encoder + GeGLU MLP) with small random weights and a **zero** rel-pos
//! bias table — so the self-attention reduces to plain causal scaled-dot-product
//! — runs it on the GPU, and compares against an independent CPU reference. This
//! validates the decoder layer's wiring (the three pre-norm sublayers, the
//! causal mask, cross-attention to a different-length encoder sequence, and the
//! GeGLU MLP) and meganeura's ops compose as intended, independent of the
//! (unavailable, and architecturally unsettled) real T5X decoder weights.

use std::collections::HashMap;

use meganeura::Graph;
use meganeura::models::magenta_rt::llm::{LlmConfig, build_decoder_layer};

fn tiny_cfg() -> LlmConfig {
    LlmConfig {
        embed_dim: 16,
        head_dim: 8,
        num_heads: 2, // num_heads * head_dim == embed_dim
        mlp_dim: 32,
        num_encoder_layers: 1,
        num_temporal_decoder_layers: 1,
        num_depth_decoder_layers: 1,
        num_levels: 16,
        vocab_size: 64,
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

// ---- CPU reference ops ----

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

/// Scaled-dot-product attention; `causal` masks j > i (q_seq must equal kv_seq).
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

fn make_weights(cfg: &LlmConfig) -> HashMap<String, Vec<f32>> {
    let embed = cfg.embed_dim as usize;
    let attn_dim = (cfg.num_heads * cfg.head_dim) as usize;
    let mlp = cfg.mlp_dim as usize;
    let mut r = Rng(0xDEADBEEF1234);
    let mut w = HashMap::new();
    let put = |w: &mut HashMap<String, Vec<f32>>, n: String, sz: usize, r: &mut Rng| {
        w.insert(n, r.vec(sz));
    };
    let p = "dec.layer";
    // RMSNorm scales near 1.
    for nm in ["pre_self_attn_norm", "pre_cross_attn_norm", "pre_mlp_norm"] {
        w.insert(format!("{p}.{nm}"), (0..embed).map(|_| 1.0 + r.next()).collect());
    }
    for kind in ["self_attn", "cross_attn"] {
        for n in ["q", "k", "v"] {
            put(&mut w, format!("{p}.{kind}.{n}"), embed * attn_dim, &mut r);
        }
        put(&mut w, format!("{p}.{kind}.o"), attn_dim * embed, &mut r);
    }
    put(&mut w, format!("{p}.mlp.w_gate"), embed * mlp, &mut r);
    put(&mut w, format!("{p}.mlp.w_up"), embed * mlp, &mut r);
    put(&mut w, format!("{p}.mlp.w_down"), mlp * embed, &mut r);
    w
}

/// CPU reference for one temporal decoder layer (zero rel-pos bias).
fn reference(
    cfg: &LlmConfig,
    w: &HashMap<String, Vec<f32>>,
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
    let g = |n: &str| -> &[f32] { w.get(n).unwrap().as_slice() };
    let p = "dec.layer";
    let mut x = x_in.to_vec();

    // 1. Causal self-attention.
    let h = rms_norm(&x, g(&format!("{p}.pre_self_attn_norm")), dec_seq, embed, eps);
    let q = matmul(&h, g(&format!("{p}.self_attn.q")), dec_seq, embed, attn_dim);
    let k = matmul(&h, g(&format!("{p}.self_attn.k")), dec_seq, embed, attn_dim);
    let v = matmul(&h, g(&format!("{p}.self_attn.v")), dec_seq, embed, attn_dim);
    let sa = sdpa(&q, &k, &v, dec_seq, dec_seq, heads, hd, true);
    let sa_out = matmul(&sa, g(&format!("{p}.self_attn.o")), dec_seq, attn_dim, embed);
    for i in 0..dec_seq * embed {
        x[i] += sa_out[i];
    }

    // 2. Cross-attention to encoder.
    let h = rms_norm(&x, g(&format!("{p}.pre_cross_attn_norm")), dec_seq, embed, eps);
    let cq = matmul(&h, g(&format!("{p}.cross_attn.q")), dec_seq, embed, attn_dim);
    let ck = matmul(enc, g(&format!("{p}.cross_attn.k")), enc_seq, embed, attn_dim);
    let cv = matmul(enc, g(&format!("{p}.cross_attn.v")), enc_seq, embed, attn_dim);
    let ca = sdpa(&cq, &ck, &cv, dec_seq, enc_seq, heads, hd, false);
    let ca_out = matmul(&ca, g(&format!("{p}.cross_attn.o")), dec_seq, attn_dim, embed);
    for i in 0..dec_seq * embed {
        x[i] += ca_out[i];
    }

    // 3. GeGLU MLP.
    let h = rms_norm(&x, g(&format!("{p}.pre_mlp_norm")), dec_seq, embed, eps);
    let gate = matmul(&h, g(&format!("{p}.mlp.w_gate")), dec_seq, embed, mlp);
    let up = matmul(&h, g(&format!("{p}.mlp.w_up")), dec_seq, embed, mlp);
    let ffn: Vec<f32> = gate.iter().zip(up.iter()).map(|(&gt, &u)| gelu_tanh(gt) * u).collect();
    let down = matmul(&ffn, g(&format!("{p}.mlp.w_down")), dec_seq, mlp, embed);
    for i in 0..dec_seq * embed {
        x[i] += down[i];
    }
    x
}

#[test]
fn decoder_layer_matches_cpu_reference() {
    let cfg = tiny_cfg();
    let dec_seq = cfg.decoder_seq_len as usize;
    let enc_seq = cfg.encoder_seq_len as usize;
    let embed = cfg.embed_dim as usize;
    let weights = make_weights(&cfg);

    let mut rin = Rng(0x0BADC0DE);
    let x_in = rin.vec(dec_seq * embed);
    let enc = rin.vec(enc_seq * embed);

    // GPU.
    let mut g = Graph::new();
    let x_node = g.input("dec_x", &[dec_seq, embed]);
    let enc_node = g.input("enc_out", &[enc_seq, embed]);
    let rel = g.parameter("dec.rel_pos", &[(cfg.num_heads * cfg.rel_pos_num_buckets) as usize]);
    let out = build_decoder_layer(&mut g, &cfg, x_node, enc_node, rel, "dec.layer");
    g.set_outputs(vec![out]);
    let mut s = meganeura::build_inference_session(&g);
    for (name, data) in &weights {
        s.set_parameter(name, data);
    }
    // Zero rel-pos table → plain causal self-attention.
    s.set_parameter("dec.rel_pos", &vec![0.0_f32; (cfg.num_heads * cfg.rel_pos_num_buckets) as usize]);
    s.set_input("dec_x", &x_in);
    s.set_input("enc_out", &enc);
    s.step();
    s.wait();
    let gpu = s.read_output(dec_seq * embed);

    let cpu = reference(&cfg, &weights, &x_in, &enc, dec_seq, enc_seq);

    assert_eq!(gpu.len(), cpu.len());
    let mut max_abs = 0.0_f32;
    for (i, (&a, &b)) in gpu.iter().zip(cpu.iter()).enumerate() {
        let d = (a - b).abs();
        max_abs = max_abs.max(d);
        assert!(
            d <= 1e-3 * b.abs().max(1.0),
            "elem {i}: gpu {a} vs cpu {b} (abs diff {d})"
        );
    }
    eprintln!("decoder layer max abs diff GPU vs CPU: {max_abs:.2e}");
}
