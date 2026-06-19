//! Numerical correctness for the *full* Depthformer decoder (`build_decoder`):
//! SOS-padded RVQ token grid → embed → mean-pool levels → temporal layers
//! (cross-attn to the encoder) → per-frame depth inputs
//! `concat([temporal_state, embed(prev levels)])` → depth layers → shared
//! `decoder_norm` → non-tied `logits_dense` → `[num_frames*num_levels, vocab]`.
//!
//! Built with small random weights and zero (shared) rel-pos tables, run on the
//! GPU (lavapipe), and compared against an independent CPU reference. This pins
//! down the temporal→depth wiring: the level mean-pool, the per-frame depth
//! prefix construction (slice + concat), the block-per-frame depth attention,
//! and the shared head — the whole teacher-forcing forward end to end.

use std::collections::HashMap;

use meganeura::models::magenta_rt::llm::{build_decoder, LlmConfig};
use meganeura::Graph;

const TEMPORAL_LAYERS: usize = 2;
const DEPTH_LAYERS: usize = 2;
const FRAMES: usize = 3;
const LEVELS: usize = 4;

fn tiny_cfg() -> LlmConfig {
    LlmConfig {
        embed_dim: 16,
        head_dim: 8,
        num_heads: 2,
        mlp_dim: 32,
        num_encoder_layers: 1,
        num_temporal_decoder_layers: TEMPORAL_LAYERS as u32,
        num_depth_decoder_layers: DEPTH_LAYERS as u32,
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

/// CPU reference for one temporal decoder layer (self-attn + cross-attn + MLP).
fn temporal_layer_ref(
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
    let ffn: Vec<f32> = gate
        .iter()
        .zip(up.iter())
        .map(|(&gt, &u)| gelu_tanh(gt) * u)
        .collect();
    let down = matmul(&ffn, g("mlp.w_down"), dec_seq, mlp, embed);
    for i in 0..dec_seq * embed {
        x[i] += down[i];
    }
    x
}

/// CPU reference for one depth decoder layer (causal self-attn + MLP, no cross).
fn depth_layer_ref(
    cfg: &LlmConfig,
    w: &HashMap<String, Vec<f32>>,
    prefix: &str,
    x_in: &[f32],
    seq: usize,
) -> Vec<f32> {
    let embed = cfg.embed_dim as usize;
    let attn_dim = (cfg.num_heads * cfg.head_dim) as usize;
    let mlp = cfg.mlp_dim as usize;
    let heads = cfg.num_heads as usize;
    let hd = cfg.head_dim as usize;
    let eps = cfg.layer_norm_eps;
    let g = |n: &str| -> &[f32] { w.get(&format!("{prefix}.{n}")).unwrap().as_slice() };
    let mut x = x_in.to_vec();

    let h = rms_norm(&x, g("pre_self_attn_norm"), seq, embed, eps);
    let q = matmul(&h, g("self_attn.q"), seq, embed, attn_dim);
    let k = matmul(&h, g("self_attn.k"), seq, embed, attn_dim);
    let v = matmul(&h, g("self_attn.v"), seq, embed, attn_dim);
    let sa = sdpa(&q, &k, &v, seq, seq, heads, hd, true);
    let sa_out = matmul(&sa, g("self_attn.o"), seq, attn_dim, embed);
    for i in 0..seq * embed {
        x[i] += sa_out[i];
    }

    let h = rms_norm(&x, g("pre_mlp_norm"), seq, embed, eps);
    let gate = matmul(&h, g("mlp.w_gate"), seq, embed, mlp);
    let up = matmul(&h, g("mlp.w_up"), seq, embed, mlp);
    let ffn: Vec<f32> = gate
        .iter()
        .zip(up.iter())
        .map(|(&gt, &u)| gelu_tanh(gt) * u)
        .collect();
    let down = matmul(&ffn, g("mlp.w_down"), seq, mlp, embed);
    for i in 0..seq * embed {
        x[i] += down[i];
    }
    x
}

fn temporal_layer_weights(
    cfg: &LlmConfig,
    w: &mut HashMap<String, Vec<f32>>,
    prefix: &str,
    r: &mut Rng,
) {
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

fn depth_layer_weights(
    cfg: &LlmConfig,
    w: &mut HashMap<String, Vec<f32>>,
    prefix: &str,
    r: &mut Rng,
) {
    let embed = cfg.embed_dim as usize;
    let attn_dim = (cfg.num_heads * cfg.head_dim) as usize;
    let mlp = cfg.mlp_dim as usize;
    for nm in ["pre_self_attn_norm", "pre_mlp_norm"] {
        w.insert(
            format!("{prefix}.{nm}"),
            (0..embed).map(|_| 1.0 + r.next()).collect(),
        );
    }
    for n in ["q", "k", "v"] {
        w.insert(format!("{prefix}.self_attn.{n}"), r.vec(embed * attn_dim));
    }
    w.insert(format!("{prefix}.self_attn.o"), r.vec(attn_dim * embed));
    w.insert(format!("{prefix}.mlp.w_gate"), r.vec(embed * mlp));
    w.insert(format!("{prefix}.mlp.w_up"), r.vec(embed * mlp));
    w.insert(format!("{prefix}.mlp.w_down"), r.vec(mlp * embed));
}

#[test]
fn full_decoder_matches_cpu_reference() {
    let cfg = tiny_cfg();
    let embed = cfg.embed_dim as usize;
    let vocab = cfg.vocab_size as usize;
    let levels = cfg.num_levels as usize;
    let enc_seq = cfg.encoder_seq_len as usize;
    let padded = (FRAMES + 1) * levels;

    let mut r = Rng(0x00C0_FFEE_4242);
    let mut weights: HashMap<String, Vec<f32>> = HashMap::new();
    weights.insert("shared_token_embedder".into(), r.vec(vocab * embed));
    weights.insert(
        "decoder.decoder_norm".into(),
        (0..embed).map(|_| 1.0 + r.next()).collect(),
    );
    weights.insert("decoder.logits_dense".into(), r.vec(embed * vocab));
    for i in 0..TEMPORAL_LAYERS {
        temporal_layer_weights(
            &cfg,
            &mut weights,
            &format!("decoder.temporal_layers.{i}"),
            &mut r,
        );
    }
    for i in 0..DEPTH_LAYERS {
        depth_layer_weights(
            &cfg,
            &mut weights,
            &format!("decoder.depth_layers.{i}"),
            &mut r,
        );
    }
    // SOS-padded grid: (FRAMES+1) frames × LEVELS levels of token ids.
    let tokens: Vec<u32> = (0..padded).map(|i| ((i * 5 + 1) % vocab) as u32).collect();
    let enc = r.vec(enc_seq * embed);

    // GPU.
    let mut g = Graph::new();
    let tok_node = g.input_u32("dec_tokens", &[padded]);
    let enc_node = g.input("enc_out", &[enc_seq, embed]);
    let logits = build_decoder(&mut g, &cfg, tok_node, enc_node, FRAMES);
    g.set_outputs(vec![logits]);
    let mut s = meganeura::build_inference_session(&g);
    for (name, data) in &weights {
        s.set_parameter(name, data);
    }
    s.set_parameter(
        "decoder.temporal_decoder.rel_pos_bias_table",
        &vec![0.0_f32; (cfg.num_heads * cfg.rel_pos_num_buckets) as usize],
    );
    s.set_parameter(
        "decoder.depth_decoder.rel_pos_bias_table",
        &vec![0.0_f32; (cfg.num_heads * cfg.depth_rel_pos_num_buckets) as usize],
    );
    s.set_input_u32("dec_tokens", &tokens);
    s.set_input("enc_out", &enc);
    s.step();
    s.wait();
    let gpu = s.read_output(FRAMES * levels * vocab);

    // CPU reference.
    let table = &weights["shared_token_embedder"];
    let emb = |tok: u32| -> &[f32] {
        let t = tok as usize;
        &table[t * embed..(t + 1) * embed]
    };
    // Embed the padded grid.
    let mut embedded = vec![0.0_f32; padded * embed];
    for (i, &tok) in tokens.iter().enumerate() {
        embedded[i * embed..(i + 1) * embed].copy_from_slice(emb(tok));
    }
    // Temporal input: mean over levels of padded frames 0..FRAMES-1.
    let mut temporal = vec![0.0_f32; FRAMES * embed];
    for f in 0..FRAMES {
        for l in 0..levels {
            for d in 0..embed {
                temporal[f * embed + d] += embedded[(f * levels + l) * embed + d] / levels as f32;
            }
        }
    }
    for i in 0..TEMPORAL_LAYERS {
        temporal = temporal_layer_ref(
            &cfg,
            &weights,
            &format!("decoder.temporal_layers.{i}"),
            &temporal,
            &enc,
            FRAMES,
            enc_seq,
        );
    }
    // Per-frame depth.
    let mut cpu = vec![0.0_f32; FRAMES * levels * vocab];
    for t in 0..FRAMES {
        // depth input: [temporal_state[t]] ++ embeddings of padded frame t+1, levels 0..L-2.
        let mut depth_in = vec![0.0_f32; levels * embed];
        depth_in[0..embed].copy_from_slice(&temporal[t * embed..(t + 1) * embed]);
        for l in 0..levels - 1 {
            let src = ((t + 1) * levels + l) * embed;
            depth_in[(l + 1) * embed..(l + 2) * embed].copy_from_slice(&embedded[src..src + embed]);
        }
        let mut x = depth_in;
        for i in 0..DEPTH_LAYERS {
            x = depth_layer_ref(
                &cfg,
                &weights,
                &format!("decoder.depth_layers.{i}"),
                &x,
                levels,
            );
        }
        x = rms_norm(
            &x,
            &weights["decoder.decoder_norm"],
            levels,
            embed,
            cfg.layer_norm_eps,
        );
        let frame_logits = matmul(&x, &weights["decoder.logits_dense"], levels, embed, vocab);
        cpu[t * levels * vocab..(t + 1) * levels * vocab].copy_from_slice(&frame_logits);
    }

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
    eprintln!("full decoder max abs diff GPU vs CPU: {max_abs:.2e}");
}
