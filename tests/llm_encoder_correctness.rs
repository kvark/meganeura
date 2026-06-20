//! Numerical correctness for the LLM encoder (`build_encoder_graph`): token
//! embed → + sinusoidal absolute PE → N bidirectional pre-norm layers (no
//! rel-pos) → final RMSNorm. Random weights, GPU (lavapipe) vs an independent
//! CPU reference (mirrors `tools/magenta_rt/llm_numpy_ref.py`), within 1e-3.
//! head_dim=64 (real value; exercises the fixed attention kernel).
//!
//! This validates the encoder's op composition — the computed sinusoidal PE, the
//! plain bidirectional self-attention, the GeGLU MLP, layer chaining, and the
//! final norm. It does NOT validate the position *scheme* against the real model
//! (sinusoidal-vs-scale is unverified; see `LLM_FINDINGS.md`).

use std::collections::HashMap;

use meganeura::models::magenta_rt::llm::{build_encoder_graph, LlmConfig};
use meganeura::Graph;

const LAYERS: usize = 2;
const SEQ: usize = 6;

fn tiny_cfg() -> LlmConfig {
    LlmConfig {
        embed_dim: 128,
        head_dim: 64,
        num_heads: 2,
        mlp_dim: 64,
        num_encoder_layers: LAYERS as u32,
        num_temporal_decoder_layers: 1,
        num_depth_decoder_layers: 1,
        num_levels: 16,
        vocab_size: 20,
        encoder_seq_len: SEQ as u32,
        decoder_seq_len: 800,
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

/// Bidirectional (non-causal) multi-head SDPA, no rel-pos bias.
fn sdpa(q: &[f32], k: &[f32], v: &[f32], seq: usize, heads: usize, hd: usize) -> Vec<f32> {
    let dim = heads * hd;
    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = vec![0.0_f32; seq * dim];
    for h in 0..heads {
        let off = h * hd;
        for i in 0..seq {
            let mut scores = vec![0.0_f32; seq];
            for (j, sc) in scores.iter_mut().enumerate() {
                let mut s = 0.0;
                for d in 0..hd {
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
            for d in 0..hd {
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

/// Same sinusoidal PE the encoder builds internally.
fn sinusoidal(seq: usize, embed: usize) -> Vec<f32> {
    let mut pe = vec![0.0_f32; seq * embed];
    for p in 0..seq {
        for i in 0..embed / 2 {
            let inv = 1.0_f64 / 10000.0_f64.powf(2.0 * i as f64 / embed as f64);
            let a = p as f64 * inv;
            pe[p * embed + 2 * i] = a.sin() as f32;
            pe[p * embed + 2 * i + 1] = a.cos() as f32;
        }
    }
    pe
}

fn layer_weights(cfg: &LlmConfig, w: &mut HashMap<String, Vec<f32>>, prefix: &str, r: &mut Rng) {
    let embed = cfg.embed_dim as usize;
    let attn_dim = (cfg.num_heads * cfg.head_dim) as usize;
    let mlp = cfg.mlp_dim as usize;
    for nm in ["pre_attn_norm", "pre_mlp_norm"] {
        w.insert(
            format!("{prefix}.{nm}"),
            (0..embed).map(|_| 1.0 + r.next()).collect(),
        );
    }
    for n in ["q", "k", "v"] {
        w.insert(format!("{prefix}.attn.{n}"), r.vec(embed * attn_dim));
    }
    w.insert(format!("{prefix}.attn.o"), r.vec(attn_dim * embed));
    w.insert(format!("{prefix}.mlp.w_gate"), r.vec(embed * mlp));
    w.insert(format!("{prefix}.mlp.w_up"), r.vec(embed * mlp));
    w.insert(format!("{prefix}.mlp.w_down"), r.vec(mlp * embed));
}

#[test]
fn encoder_matches_cpu_reference() {
    let cfg = tiny_cfg();
    let embed = cfg.embed_dim as usize;
    let vocab = cfg.vocab_size as usize;
    let attn_dim = (cfg.num_heads * cfg.head_dim) as usize;
    let mlp = cfg.mlp_dim as usize;
    let heads = cfg.num_heads as usize;
    let hd = cfg.head_dim as usize;
    let eps = cfg.layer_norm_eps;

    let mut r = Rng(0x0000_E5C0_DE77);
    let mut weights: HashMap<String, Vec<f32>> = HashMap::new();
    weights.insert("shared_token_embedder".into(), r.vec(vocab * embed));
    weights.insert(
        "encoder.final_norm".into(),
        (0..embed).map(|_| 1.0 + r.next()).collect(),
    );
    for i in 0..LAYERS {
        layer_weights(&cfg, &mut weights, &format!("encoder.layers.{i}"), &mut r);
    }
    let tokens: Vec<u32> = (0..SEQ).map(|i| ((i * 3 + 1) % vocab) as u32).collect();

    // GPU.
    let mut g = Graph::new();
    let out = build_encoder_graph(&mut g, &cfg, SEQ);
    g.set_outputs(vec![out]);
    let mut s = meganeura::build_inference_session(&g);
    for (n, d) in &weights {
        s.set_parameter(n, d);
    }
    s.set_input_u32("encoder_input_tokens", &tokens);
    s.step();
    s.wait();
    let gpu = s.read_output(SEQ * embed);

    // CPU reference.
    let table = &weights["shared_token_embedder"];
    let mut x = vec![0.0_f32; SEQ * embed];
    for (i, &t) in tokens.iter().enumerate() {
        x[i * embed..(i + 1) * embed]
            .copy_from_slice(&table[t as usize * embed..(t as usize + 1) * embed]);
    }
    let pe = sinusoidal(SEQ, embed);
    for i in 0..SEQ * embed {
        x[i] += pe[i];
    }
    for l in 0..LAYERS {
        let prefix = format!("encoder.layers.{l}");
        let g = |n: &str| -> &[f32] { weights.get(&format!("{prefix}.{n}")).unwrap().as_slice() };
        let h = rms_norm(&x, g("pre_attn_norm"), SEQ, embed, eps);
        let q = matmul(&h, g("attn.q"), SEQ, embed, attn_dim);
        let k = matmul(&h, g("attn.k"), SEQ, embed, attn_dim);
        let v = matmul(&h, g("attn.v"), SEQ, embed, attn_dim);
        let a = sdpa(&q, &k, &v, SEQ, heads, hd);
        let ao = matmul(&a, g("attn.o"), SEQ, attn_dim, embed);
        for i in 0..SEQ * embed {
            x[i] += ao[i];
        }
        let h = rms_norm(&x, g("pre_mlp_norm"), SEQ, embed, eps);
        let gate = matmul(&h, g("mlp.w_gate"), SEQ, embed, mlp);
        let up = matmul(&h, g("mlp.w_up"), SEQ, embed, mlp);
        let ffn: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(&gt, &u)| gelu_tanh(gt) * u)
            .collect();
        let down = matmul(&ffn, g("mlp.w_down"), SEQ, mlp, embed);
        for i in 0..SEQ * embed {
            x[i] += down[i];
        }
    }
    let cpu = rms_norm(&x, &weights["encoder.final_norm"], SEQ, embed, eps);

    assert_eq!(gpu.len(), cpu.len());
    let mut max_abs = 0.0_f32;
    for (i, (&a, &b)) in gpu.iter().zip(cpu.iter()).enumerate() {
        let d = (a - b).abs();
        max_abs = max_abs.max(d);
        assert!(
            d <= 1e-3 * b.abs().max(1.0),
            "elem {i}: gpu {a} vs cpu {b} (diff {d})"
        );
    }
    eprintln!("encoder max abs diff GPU vs CPU: {max_abs:.2e}");
}
