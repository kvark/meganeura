//! Numerical correctness for the MusicCoCa text-encoder graph.
//!
//! Builds the encoder with small random weights, runs it on the GPU, and
//! compares against an independent CPU reference forward pass that mirrors
//! `tools/magenta_rt/musiccoca_numpy_ref.py`. This validates that the graph
//! wiring (pre-norm layers, residuals, attention pooling) and meganeura's
//! `full_attention` / `cross_attention` / `layer_norm` / `gelu` ops compose
//! into the intended computation — independent of the (unavailable) real
//! MusicCoCa weights.

use std::collections::HashMap;

use meganeura::Graph;
use meganeura::models::magenta_rt::musiccoca::{
    MusicCoCaConfig, build_text_encoder_graph, sinusoidal_pos_embedding,
};

/// Tiny deterministic config so the CPU reference stays cheap.
fn tiny_cfg() -> MusicCoCaConfig {
    MusicCoCaConfig {
        embed_dim: 16,
        num_heads: 2,
        head_dim: 8, // num_heads * head_dim == embed_dim
        mlp_dim: 32,
        num_layers: 2,
        pool_head_dim: 8, // pool_dim = num_heads * pool_head_dim = 16
        vocab_size: 10,
        rvq_depth: 12,
        codebook_size: 1024,
        layer_norm_eps: 1e-6,
    }
}

/// Deterministic small pseudo-random weights in roughly [-0.15, 0.15].
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        // xorshift64*
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        let u = (x.wrapping_mul(0x2545F4914F6CDD1D) >> 40) as u32; // 24 bits
        (u as f32 / (1u32 << 24) as f32 - 0.5) * 0.3
    }
    fn vec(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| self.next()).collect()
    }
}

/// Build the full weight set keyed by the exact param names
/// `build_text_encoder_graph` declares, so the GPU session and the CPU
/// reference consume identical tensors.
fn make_weights(cfg: &MusicCoCaConfig) -> HashMap<String, Vec<f32>> {
    let embed = cfg.embed_dim as usize;
    let attn_dim = (cfg.num_heads * cfg.head_dim) as usize;
    let mlp = cfg.mlp_dim as usize;
    let pool_dim = (cfg.num_heads * cfg.pool_head_dim) as usize;
    let mut r = Rng(0x9E3779B97F4A7C15);
    let mut w = HashMap::new();
    let put = |w: &mut HashMap<String, Vec<f32>>, name: String, n: usize, r: &mut Rng| {
        w.insert(name, r.vec(n));
    };

    put(&mut w, "text_encoder.embed_table".into(), cfg.vocab_size as usize * embed, &mut r);
    for i in 0..cfg.num_layers {
        let p = format!("text_encoder.layers.{i}");
        // LayerNorm scale near 1 (graph applies it as a plain weight).
        let ln_scale = |w: &mut HashMap<String, Vec<f32>>, name: String, r: &mut Rng| {
            let v: Vec<f32> = (0..embed).map(|_| 1.0 + r.next()).collect();
            w.insert(name, v);
        };
        ln_scale(&mut w, format!("{p}.pre_attn_norm.scale"), &mut r);
        put(&mut w, format!("{p}.pre_attn_norm.bias"), embed, &mut r);
        for n in ["q", "k", "v"] {
            put(&mut w, format!("{p}.attn.{n}.kernel"), embed * attn_dim, &mut r);
            put(&mut w, format!("{p}.attn.{n}.bias"), attn_dim, &mut r);
        }
        put(&mut w, format!("{p}.attn.o.kernel"), attn_dim * embed, &mut r);
        put(&mut w, format!("{p}.attn.o.bias"), embed, &mut r);
        ln_scale(&mut w, format!("{p}.pre_mlp_norm.scale"), &mut r);
        put(&mut w, format!("{p}.pre_mlp_norm.bias"), embed, &mut r);
        put(&mut w, format!("{p}.mlp.wi.kernel"), embed * mlp, &mut r);
        put(&mut w, format!("{p}.mlp.wi.bias"), mlp, &mut r);
        put(&mut w, format!("{p}.mlp.wo.kernel"), mlp * embed, &mut r);
        put(&mut w, format!("{p}.mlp.wo.bias"), embed, &mut r);
    }
    put(&mut w, "text_encoder.pool.query".into(), embed, &mut r);
    for n in ["q", "k", "v"] {
        put(&mut w, format!("text_encoder.pool.{n}.kernel"), embed * pool_dim, &mut r);
        put(&mut w, format!("text_encoder.pool.{n}.bias"), pool_dim, &mut r);
    }
    put(&mut w, "text_encoder.pool.o.kernel".into(), pool_dim * embed, &mut r);
    put(&mut w, "text_encoder.pool.o.bias".into(), embed, &mut r);
    ln_scale_final(&mut w, "text_encoder.final_norm.scale".into(), embed, &mut r);
    put(&mut w, "text_encoder.final_norm.bias".into(), embed, &mut r);
    w
}

fn ln_scale_final(w: &mut HashMap<String, Vec<f32>>, name: String, embed: usize, r: &mut Rng) {
    let v: Vec<f32> = (0..embed).map(|_| 1.0 + r.next()).collect();
    w.insert(name, v);
}

// ---- CPU reference ops (mirror musiccoca_numpy_ref.py) ----

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

fn bias_add(x: &mut [f32], bias: &[f32], rows: usize, n: usize) {
    for r in 0..rows {
        for j in 0..n {
            x[r * n + j] += bias[j];
        }
    }
}

fn layer_norm(x: &[f32], w: &[f32], b: &[f32], rows: usize, dim: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0.0_f32; rows * dim];
    for r in 0..rows {
        let row = &x[r * dim..(r + 1) * dim];
        let mean = row.iter().sum::<f32>() / dim as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / dim as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for d in 0..dim {
            out[r * dim + d] = (row[d] - mean) * inv * w[d] + b[d];
        }
    }
    out
}

fn gelu(x: &[f32]) -> Vec<f32> {
    // sqrt(2/pi), matching shaders/unary.wgsl::gelu in f32.
    let c = (2.0_f32 / std::f32::consts::PI).sqrt();
    x.iter()
        .map(|&v| 0.5 * v * (1.0 + (c * (v + 0.044715 * v * v * v)).tanh()))
        .collect()
}

/// Scaled-dot-product multi-head attention matching meganeura's layout:
/// head `h` occupies columns `[h*head_dim, (h+1)*head_dim)`, scale 1/sqrt(hd).
fn mha(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    q_seq: usize,
    kv_seq: usize,
    heads: usize,
    head_dim: usize,
) -> Vec<f32> {
    let dim = heads * head_dim;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let mut out = vec![0.0_f32; q_seq * dim];
    for h in 0..heads {
        let off = h * head_dim;
        for i in 0..q_seq {
            let mut scores = vec![0.0_f32; kv_seq];
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

/// Full CPU reference: ids → [1, embed] contrastive embedding (pre-L2).
fn reference(cfg: &MusicCoCaConfig, w: &HashMap<String, Vec<f32>>, ids: &[u32]) -> Vec<f32> {
    let embed = cfg.embed_dim as usize;
    let attn_dim = (cfg.num_heads * cfg.head_dim) as usize;
    let mlp = cfg.mlp_dim as usize;
    let pool_dim = (cfg.num_heads * cfg.pool_head_dim) as usize;
    let heads = cfg.num_heads as usize;
    let seq = ids.len();
    let eps = cfg.layer_norm_eps;
    let g = |n: &str| -> &[f32] { w.get(n).unwrap().as_slice() };

    // Token embed + sinusoidal PE.
    let table = g("text_encoder.embed_table");
    let mut x = vec![0.0_f32; seq * embed];
    for (s, &id) in ids.iter().enumerate() {
        x[s * embed..(s + 1) * embed]
            .copy_from_slice(&table[id as usize * embed..(id as usize + 1) * embed]);
    }
    let pe = sinusoidal_pos_embedding(seq, embed);
    for i in 0..seq * embed {
        x[i] += pe[i];
    }

    for li in 0..cfg.num_layers {
        let p = format!("text_encoder.layers.{li}");
        let h = layer_norm(&x, g(&format!("{p}.pre_attn_norm.scale")), g(&format!("{p}.pre_attn_norm.bias")), seq, embed, eps);
        let proj = |name: &str| {
            let mut m = matmul(&h, g(&format!("{p}.attn.{name}.kernel")), seq, embed, attn_dim);
            bias_add(&mut m, g(&format!("{p}.attn.{name}.bias")), seq, attn_dim);
            m
        };
        let q = proj("q");
        let k = proj("k");
        let v = proj("v");
        let attn = mha(&q, &k, &v, seq, seq, heads, cfg.head_dim as usize);
        let mut o = matmul(&attn, g(&format!("{p}.attn.o.kernel")), seq, attn_dim, embed);
        bias_add(&mut o, g(&format!("{p}.attn.o.bias")), seq, embed);
        for i in 0..seq * embed {
            x[i] += o[i];
        }
        let h = layer_norm(&x, g(&format!("{p}.pre_mlp_norm.scale")), g(&format!("{p}.pre_mlp_norm.bias")), seq, embed, eps);
        let mut up = matmul(&h, g(&format!("{p}.mlp.wi.kernel")), seq, embed, mlp);
        bias_add(&mut up, g(&format!("{p}.mlp.wi.bias")), seq, mlp);
        let act = gelu(&up);
        let mut down = matmul(&act, g(&format!("{p}.mlp.wo.kernel")), seq, mlp, embed);
        bias_add(&mut down, g(&format!("{p}.mlp.wo.bias")), seq, embed);
        for i in 0..seq * embed {
            x[i] += down[i];
        }
    }

    // Attention pool: single learned query over encoder positions.
    let pool_proj_q = {
        let mut m = matmul(g("text_encoder.pool.query"), g("text_encoder.pool.q.kernel"), 1, embed, pool_dim);
        bias_add(&mut m, g("text_encoder.pool.q.bias"), 1, pool_dim);
        m
    };
    let pool_proj = |name: &str| {
        let mut m = matmul(&x, g(&format!("text_encoder.pool.{name}.kernel")), seq, embed, pool_dim);
        bias_add(&mut m, g(&format!("text_encoder.pool.{name}.bias")), seq, pool_dim);
        m
    };
    let pool_k = pool_proj("k");
    let pool_v = pool_proj("v");
    let pooled = mha(&pool_proj_q, &pool_k, &pool_v, 1, seq, heads, cfg.pool_head_dim as usize);
    let mut pool_out = matmul(&pooled, g("text_encoder.pool.o.kernel"), 1, pool_dim, embed);
    bias_add(&mut pool_out, g("text_encoder.pool.o.bias"), 1, embed);

    layer_norm(&pool_out, g("text_encoder.final_norm.scale"), g("text_encoder.final_norm.bias"), 1, embed, eps)
}

#[test]
fn text_encoder_matches_cpu_reference() {
    let cfg = tiny_cfg();
    let ids: Vec<u32> = vec![1, 3, 0];
    let weights = make_weights(&cfg);

    // GPU forward.
    let mut g = Graph::new();
    let out = build_text_encoder_graph(&mut g, &cfg, ids.len());
    g.set_outputs(vec![out]);
    let mut s = meganeura::build_inference_session(&g);
    for (name, data) in &weights {
        s.set_parameter(name, data);
    }
    s.set_input_u32("text_tokens", &ids);
    s.step();
    s.wait();
    let gpu = s.read_output(cfg.embed_dim as usize);

    // CPU reference.
    let cpu = reference(&cfg, &weights, &ids);

    assert_eq!(gpu.len(), cpu.len());
    let mut max_abs = 0.0_f32;
    for (i, (&a, &b)) in gpu.iter().zip(cpu.iter()).enumerate() {
        let d = (a - b).abs();
        if d > max_abs {
            max_abs = d;
        }
        assert!(
            d <= 1e-3 * b.abs().max(1.0),
            "elem {i}: gpu {a} vs cpu {b} (abs diff {d})"
        );
    }
    eprintln!("max abs diff GPU vs CPU reference: {max_abs:.2e}");
}
