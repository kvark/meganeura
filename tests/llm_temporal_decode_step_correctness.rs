//! Numerical correctness for the incremental KV-cached temporal decode
//! (`build_temporal_decode_step`).
//!
//! Drives the single-frame decode step F times (the per-layer K/V caches persist
//! across `step()` calls as mutable parameter buffers, the smollm2 pattern), and
//! compares the per-frame temporal states against an independent CPU reference
//! forward, on the GPU (lavapipe), within 1e-3. A **non-zero** shared rel-pos
//! table is used, exercising `cached_attention_rel_pos` (the cached self-attn
//! applies the learned T5 rel-pos bias just like the parallel full attention).
//!
//! This validates the autoregressive machinery — `cache_write` + repeated
//! rel-pos cached attention against a growing cache, the per-step mean-pool, the
//! cross-attention, and layer chaining — against ground truth, and that it
//! agrees with the parallel `build_temporal_decoder` it mirrors.

use std::collections::HashMap;

use meganeura::models::magenta_rt::llm::{
    build_temporal_decode_step, build_temporal_decoder, LlmConfig,
};
use meganeura::Graph;

const LAYERS: usize = 2;
const FRAMES: usize = 4;
const LEVELS: usize = 4;

fn tiny_cfg() -> LlmConfig {
    // head_dim must be 64: `cached_attention` reduces the per-head dot product
    // over its full 64-lane workgroup (no `tid < head_dim` mask), so it is only
    // correct at head_dim == 64 — which is the real Magenta-RT value.
    LlmConfig {
        embed_dim: 128,
        head_dim: 64,
        num_heads: 2,
        mlp_dim: 64,
        num_encoder_layers: 1,
        num_temporal_decoder_layers: LAYERS as u32,
        num_depth_decoder_layers: 1,
        num_levels: LEVELS as u32,
        vocab_size: 20,
        encoder_seq_len: 4,
        decoder_seq_len: (FRAMES * LEVELS) as u32,
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

/// T5 relative-position bucket — ports the WGSL `rel_pos_bucket` helper exactly.
fn rel_pos_bucket(
    q_minus_k: i32,
    num_buckets: u32,
    max_distance: u32,
    bidirectional: bool,
) -> usize {
    let mut n = q_minus_k;
    let mut ret = 0u32;
    let mut nb = num_buckets;
    if bidirectional {
        nb /= 2;
        if n < 0 {
            ret = nb;
            n = -n;
        }
    } else if n < 0 {
        n = 0;
    }
    let max_exact = nb / 2;
    let n_u = n as u32;
    if n_u < max_exact {
        return (ret + n_u) as usize;
    }
    let log_n = ((n_u as f32) / (max_exact as f32)).ln();
    let log_max = ((max_distance as f32) / (max_exact as f32)).ln();
    let val_large = (max_exact as f32 + log_n / log_max * (nb - max_exact) as f32) as u32;
    (ret + val_large.min(nb - 1)) as usize
}

/// Causal self-attention with a per-head T5 rel-pos bias on the scores.
#[allow(clippy::too_many_arguments)]
fn self_attn_relpos(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    seq: usize,
    heads: usize,
    hd: usize,
    rel: &[f32],
    num_buckets: u32,
    max_distance: u32,
) -> Vec<f32> {
    let dim = heads * hd;
    let scale = 1.0 / (hd as f32).sqrt();
    let mut out = vec![0.0_f32; seq * dim];
    for h in 0..heads {
        let off = h * hd;
        for i in 0..seq {
            let mut scores = vec![0.0_f32; i + 1];
            for (j, sc) in scores.iter_mut().enumerate() {
                let mut s = 0.0;
                for d in 0..hd {
                    s += q[i * dim + off + d] * k[j * dim + off + d];
                }
                let b = rel_pos_bucket(i as i32 - j as i32, num_buckets, max_distance, false);
                *sc = s * scale + rel[h * num_buckets as usize + b];
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

/// CPU reference for the parallel temporal forward over all FRAMES, returning
/// per-frame states `[FRAMES, embed]`. The incremental GPU decode must reproduce
/// these frame by frame. `rel` is the shared temporal rel-pos bias table.
fn temporal_ref(
    cfg: &LlmConfig,
    w: &HashMap<String, Vec<f32>>,
    tokens: &[u32],
    enc: &[f32],
    rel: &[f32],
) -> Vec<f32> {
    let embed = cfg.embed_dim as usize;
    let attn_dim = (cfg.num_heads * cfg.head_dim) as usize;
    let mlp = cfg.mlp_dim as usize;
    let heads = cfg.num_heads as usize;
    let hd = cfg.head_dim as usize;
    let levels = cfg.num_levels as usize;
    let enc_seq = cfg.encoder_seq_len as usize;
    let eps = cfg.layer_norm_eps;
    let table = &w["shared_token_embedder"];

    // Mean-pool each frame's level embeddings → [FRAMES, embed].
    let mut x = vec![0.0_f32; FRAMES * embed];
    for f in 0..FRAMES {
        for l in 0..levels {
            let t = tokens[f * levels + l] as usize;
            for d in 0..embed {
                x[f * embed + d] += table[t * embed + d] / levels as f32;
            }
        }
    }

    for i in 0..LAYERS {
        let prefix = format!("decoder.temporal_layers.{i}");
        let g = |n: &str| -> &[f32] { w.get(&format!("{prefix}.{n}")).unwrap().as_slice() };

        let h = rms_norm(&x, g("pre_self_attn_norm"), FRAMES, embed, eps);
        let q = matmul(&h, g("self_attn.q"), FRAMES, embed, attn_dim);
        let k = matmul(&h, g("self_attn.k"), FRAMES, embed, attn_dim);
        let v = matmul(&h, g("self_attn.v"), FRAMES, embed, attn_dim);
        let sa = self_attn_relpos(
            &q,
            &k,
            &v,
            FRAMES,
            heads,
            hd,
            rel,
            cfg.rel_pos_num_buckets,
            cfg.rel_pos_max_distance,
        );
        let sa_out = matmul(&sa, g("self_attn.o"), FRAMES, attn_dim, embed);
        for j in 0..FRAMES * embed {
            x[j] += sa_out[j];
        }

        let h = rms_norm(&x, g("pre_cross_attn_norm"), FRAMES, embed, eps);
        let cq = matmul(&h, g("cross_attn.q"), FRAMES, embed, attn_dim);
        let ck = matmul(enc, g("cross_attn.k"), enc_seq, embed, attn_dim);
        let cv = matmul(enc, g("cross_attn.v"), enc_seq, embed, attn_dim);
        let ca = sdpa(&cq, &ck, &cv, FRAMES, enc_seq, heads, hd, false);
        let ca_out = matmul(&ca, g("cross_attn.o"), FRAMES, attn_dim, embed);
        for j in 0..FRAMES * embed {
            x[j] += ca_out[j];
        }

        let h = rms_norm(&x, g("pre_mlp_norm"), FRAMES, embed, eps);
        let gate = matmul(&h, g("mlp.w_gate"), FRAMES, embed, mlp);
        let up = matmul(&h, g("mlp.w_up"), FRAMES, embed, mlp);
        let ffn: Vec<f32> = gate
            .iter()
            .zip(up.iter())
            .map(|(&gt, &u)| gelu_tanh(gt) * u)
            .collect();
        let down = matmul(&ffn, g("mlp.w_down"), FRAMES, mlp, embed);
        for j in 0..FRAMES * embed {
            x[j] += down[j];
        }
    }
    x
}

fn layer_weights(cfg: &LlmConfig, w: &mut HashMap<String, Vec<f32>>, prefix: &str, r: &mut Rng) {
    let embed = cfg.embed_dim as usize;
    let attn_dim = (cfg.num_heads * cfg.head_dim) as usize;
    let mlp = cfg.mlp_dim as usize;
    for nm in ["pre_self_attn_norm", "pre_cross_attn_norm", "pre_mlp_norm"] {
        w.insert(
            format!("{prefix}.{nm}"),
            (0..embed).map(|_| 1.0 + r.next()).collect(),
        );
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
fn incremental_temporal_decode_matches_cpu_reference() {
    let cfg = tiny_cfg();
    let embed = cfg.embed_dim as usize;
    let vocab = cfg.vocab_size as usize;
    let levels = cfg.num_levels as usize;
    let enc_seq = cfg.encoder_seq_len as usize;
    let attn_dim = (cfg.num_heads * cfg.head_dim) as usize;
    let grid = FRAMES * levels;

    let mut r = Rng(0x00AB_1234_5678);
    let mut weights: HashMap<String, Vec<f32>> = HashMap::new();
    weights.insert("shared_token_embedder".into(), r.vec(vocab * embed));
    for i in 0..LAYERS {
        layer_weights(
            &cfg,
            &mut weights,
            &format!("decoder.temporal_layers.{i}"),
            &mut r,
        );
    }
    let tokens: Vec<u32> = (0..grid).map(|i| ((i * 3 + 2) % vocab) as u32).collect();
    let enc = r.vec(enc_seq * embed);
    // Non-zero shared temporal rel-pos table, to exercise the rel-pos path.
    let rel = r.vec((cfg.num_heads * cfg.rel_pos_num_buckets) as usize);

    // Incremental decode: one frame per step, caches persist across steps.
    let mut g = Graph::new();
    let step_tok = g.input_u32("step_tokens", &[levels]);
    let enc_node = g.input("enc_out", &[enc_seq, embed]);
    let kv_pos = g.input_u32("kv_pos", &[1]);
    let state = build_temporal_decode_step(&mut g, &cfg, step_tok, enc_node, kv_pos, FRAMES);
    g.set_outputs(vec![state]);
    let mut s = meganeura::build_inference_session(&g);
    for (n, d) in &weights {
        s.set_parameter(n, d);
    }
    for i in 0..LAYERS {
        s.set_parameter(
            &format!("decoder.temporal_kv_cache.{i}.k"),
            &vec![0.0_f32; FRAMES * attn_dim],
        );
        s.set_parameter(
            &format!("decoder.temporal_kv_cache.{i}.v"),
            &vec![0.0_f32; FRAMES * attn_dim],
        );
    }
    s.set_parameter("decoder.temporal_decoder.rel_pos_bias_table", &rel);
    s.set_input("enc_out", &enc);

    let mut incremental = vec![0.0_f32; FRAMES * embed];
    for t in 0..FRAMES {
        let frame: Vec<u32> = tokens[t * levels..(t + 1) * levels].to_vec();
        s.set_input_u32("step_tokens", &frame);
        s.set_input_u32("kv_pos", &[t as u32]);
        s.step();
        s.wait();
        let out = s.read_output(embed);
        incremental[t * embed..(t + 1) * embed].copy_from_slice(&out);
    }

    // Parallel forward (independently CPU-verified at head_dim=8) for isolation.
    let parallel = {
        let mut g = Graph::new();
        let tok = g.input_u32("dec_tokens", &[grid]);
        let enc_node = g.input("enc_out", &[enc_seq, embed]);
        let states = build_temporal_decoder(&mut g, &cfg, tok, enc_node, FRAMES);
        g.set_outputs(vec![states]);
        let mut s = meganeura::build_inference_session(&g);
        for (n, d) in &weights {
            s.set_parameter(n, d);
        }
        s.set_parameter("decoder.temporal_decoder.rel_pos_bias_table", &rel);
        s.set_input_u32("dec_tokens", &tokens);
        s.set_input("enc_out", &enc);
        s.step();
        s.wait();
        s.read_output(FRAMES * embed)
    };

    let cpu = temporal_ref(&cfg, &weights, &tokens, &enc, &rel);
    let diff = |a: &[f32], b: &[f32]| {
        a.iter()
            .zip(b)
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max)
    };
    eprintln!("parallel-vs-cpu max diff: {:.3e}", diff(&parallel, &cpu));
    eprintln!(
        "incremental-vs-parallel max diff: {:.3e}",
        diff(&incremental, &parallel)
    );

    assert_eq!(cpu.len(), incremental.len());
    let mut max_abs = 0.0_f32;
    let mut max_at = 0;
    for (i, (&a, &b)) in incremental.iter().zip(cpu.iter()).enumerate() {
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
            max_at = i;
        }
    }
    eprintln!(
        "incremental temporal decode max abs diff GPU vs CPU: {max_abs:.3e} at elem {max_at} (frame {})",
        max_at / embed
    );
    for f in 0..FRAMES {
        let mut fmax = 0.0_f32;
        for d in 0..embed {
            fmax = fmax.max((incremental[f * embed + d] - cpu[f * embed + d]).abs());
        }
        eprintln!("  frame {f}: max abs diff {fmax:.3e}");
    }
    assert!(max_abs <= 1e-3, "max abs diff {max_abs} exceeds 1e-3");
}
