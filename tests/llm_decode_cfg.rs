//! CFG + sampling for the autoregressive decode driver (`decode`).
//!
//! CFG is host-side: two batch=1 passes (positive / negative encoder output)
//! whose per-level logits are combined with `cfg_combine` before sampling. This
//! checks that (1) CFG-greedy matches the argmax of the two-pass parallel
//! `build_decoder` logits combined the same way, (2) top-k sampling is
//! reproducible from a seed and only ever emits tokens within the top-k of the
//! parallel combined logits. On lavapipe; head_dim=64 (real value).

use std::collections::HashMap;

use meganeura::models::magenta_rt::llm::{
    build_decoder_faithful, build_depth_decoder_stack, build_temporal_decode_step, decode,
    DecodeOptions, LlmConfig,
};
use meganeura::models::magenta_rt::sampling::{argmax, cfg_combine};
use meganeura::Graph;

const TEMPORAL_LAYERS: usize = 2;
const DEPTH_LAYERS: usize = 2;
const FRAMES: usize = 3;
const LEVELS: usize = 4;
const SOS_ID: u32 = 1;
const GUIDANCE: f32 = 3.0;

fn cfg() -> LlmConfig {
    LlmConfig {
        embed_dim: 128,
        head_dim: 64,
        num_heads: 2,
        mlp_dim: 64,
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

fn temporal_layer_weights(
    c: &LlmConfig,
    w: &mut HashMap<String, Vec<f32>>,
    prefix: &str,
    r: &mut Rng,
) {
    let embed = c.embed_dim as usize;
    let attn_dim = (c.num_heads * c.head_dim) as usize;
    let mlp = c.mlp_dim as usize;
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
    c: &LlmConfig,
    w: &mut HashMap<String, Vec<f32>>,
    prefix: &str,
    r: &mut Rng,
) {
    let embed = c.embed_dim as usize;
    let attn_dim = (c.num_heads * c.head_dim) as usize;
    let mlp = c.mlp_dim as usize;
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

fn load(s: &mut meganeura::Session, w: &HashMap<String, Vec<f32>>) {
    for (name, _) in s.plan().param_buffers.clone() {
        let data = w
            .get(&name)
            .unwrap_or_else(|| panic!("missing weight: {name}"));
        s.set_parameter(&name, data);
    }
}

fn zero_caches(s: &mut meganeura::Session, c: &LlmConfig) {
    let attn_dim = (c.num_heads * c.head_dim) as usize;
    for i in 0..TEMPORAL_LAYERS {
        s.set_parameter(
            &format!("decoder.temporal_kv_cache.{i}.k"),
            &vec![0.0; FRAMES * attn_dim],
        );
        s.set_parameter(
            &format!("decoder.temporal_kv_cache.{i}.v"),
            &vec![0.0; FRAMES * attn_dim],
        );
    }
}

/// Build the shared weight map (caches included, zeroed).
fn make_weights(c: &LlmConfig) -> HashMap<String, Vec<f32>> {
    let embed = c.embed_dim as usize;
    let vocab = c.vocab_size as usize;
    let attn_dim = (c.num_heads * c.head_dim) as usize;
    let mut r = Rng(0x0000_0CF6_5EED);
    let mut w: HashMap<String, Vec<f32>> = HashMap::new();
    w.insert("shared_token_embedder".into(), r.vec(vocab * embed));
    w.insert(
        "decoder.decoder_norm".into(),
        (0..embed).map(|_| 1.0 + r.next()).collect(),
    );
    w.insert("decoder.logits_dense".into(), r.vec(embed * vocab));
    w.insert(
        "decoder.temporal_decoder.rel_pos_bias_table".into(),
        r.vec((c.num_heads * c.rel_pos_num_buckets) as usize),
    );
    w.insert(
        "decoder.depth_decoder.rel_pos_bias_table".into(),
        r.vec((c.num_heads * c.depth_rel_pos_num_buckets) as usize),
    );
    for i in 0..TEMPORAL_LAYERS {
        temporal_layer_weights(c, &mut w, &format!("decoder.temporal_layers.{i}"), &mut r);
    }
    for i in 0..DEPTH_LAYERS {
        depth_layer_weights(c, &mut w, &format!("decoder.depth_layers.{i}"), &mut r);
    }
    for i in 0..TEMPORAL_LAYERS {
        w.insert(
            format!("decoder.temporal_kv_cache.{i}.k"),
            vec![0.0; FRAMES * attn_dim],
        );
        w.insert(
            format!("decoder.temporal_kv_cache.{i}.v"),
            vec![0.0; FRAMES * attn_dim],
        );
    }
    w
}

fn build_temporal(c: &LlmConfig, w: &HashMap<String, Vec<f32>>, enc: &[f32]) -> meganeura::Session {
    let embed = c.embed_dim as usize;
    let levels = c.num_levels as usize;
    let enc_seq = c.encoder_seq_len as usize;
    let mut g = Graph::new();
    let step_tok = g.input_u32("step_tokens", &[levels]);
    let enc_node = g.input("enc_out", &[enc_seq, embed]);
    let kv_pos = g.input_u32("kv_pos", &[1]);
    let state = build_temporal_decode_step(&mut g, c, step_tok, enc_node, kv_pos, FRAMES);
    g.set_outputs(vec![state]);
    let mut s = meganeura::build_inference_session(&g);
    load(&mut s, w);
    s.set_input("enc_out", enc);
    s
}

fn build_depth(c: &LlmConfig, w: &HashMap<String, Vec<f32>>) -> meganeura::Session {
    let embed = c.embed_dim as usize;
    let levels = c.num_levels as usize;
    let mut g = Graph::new();
    let depth_in = g.input("depth_inputs", &[levels, embed]);
    let dlogits = build_depth_decoder_stack(&mut g, c, depth_in);
    g.set_outputs(vec![dlogits]);
    let mut s = meganeura::build_inference_session(&g);
    load(&mut s, w);
    s
}

/// Teacher-force `dit` (the flat shift_right(decoded) grid `[FRAMES*LEVELS]`)
/// through `build_decoder_faithful` with encoder `enc`, returning per-position
/// logits `[FRAMES*LEVELS, vocab]`.
fn parallel_logits(
    c: &LlmConfig,
    w: &HashMap<String, Vec<f32>>,
    dit: &[u32],
    enc: &[f32],
) -> Vec<f32> {
    let embed = c.embed_dim as usize;
    let enc_seq = c.encoder_seq_len as usize;
    let levels = c.num_levels as usize;
    let vocab = c.vocab_size as usize;
    let mut g = Graph::new();
    let tok = g.input_u32("dec_tokens", &[FRAMES * levels]);
    let enc_node = g.input("enc_out", &[enc_seq, embed]);
    let logits = build_decoder_faithful(&mut g, c, tok, enc_node, FRAMES);
    g.set_outputs(vec![logits]);
    let mut s = meganeura::build_inference_session(&g);
    load(&mut s, w);
    s.set_input_u32("dec_tokens", dit);
    s.set_input("enc_out", enc);
    s.step();
    s.wait();
    s.read_output(FRAMES * levels * vocab)
}

/// shift_right: prepend SOS, drop the last token → the decoder input grid.
fn shift_right(tokens: &[u32], sos: u32) -> Vec<u32> {
    let mut dit = vec![sos];
    dit.extend_from_slice(&tokens[..tokens.len() - 1]);
    dit
}

#[test]
fn cfg_greedy_matches_parallel_combined_argmax() {
    let c = cfg();
    let embed = c.embed_dim as usize;
    let enc_seq = c.encoder_seq_len as usize;
    let levels = c.num_levels as usize;
    let vocab = c.vocab_size as usize;

    let w = make_weights(&c);
    let mut r = Rng(0x0000_E5C0_0011);
    let enc_pos = r.vec(enc_seq * embed);
    let enc_neg = r.vec(enc_seq * embed);
    let table = w["shared_token_embedder"].clone();

    let mut tpos = build_temporal(&c, &w, &enc_pos);
    let mut tneg = build_temporal(&c, &w, &enc_neg);
    let mut depth = build_depth(&c, &w);

    let opts = DecodeOptions {
        num_frames: FRAMES,
        sos_id: SOS_ID,
        guidance_weight: GUIDANCE,
        temperature: 0.0, // greedy
        top_k: 0,
        seed: 0,
    };
    let tokens = decode(&c, &mut tpos, Some(&mut tneg), &mut depth, &table, &opts);
    assert_eq!(tokens.len(), FRAMES * levels);
    eprintln!("CFG-greedy grid: {tokens:?}");

    // Cross-check: two-pass parallel logits, combined the same way, argmax.
    let dit = shift_right(&tokens, SOS_ID);
    let pos = parallel_logits(&c, &w, &dit, &enc_pos);
    let neg = parallel_logits(&c, &w, &dit, &enc_neg);
    for p in 0..FRAMES * levels {
        let mut combined = vec![0.0_f32; vocab];
        cfg_combine(
            &pos[p * vocab..(p + 1) * vocab],
            &neg[p * vocab..(p + 1) * vocab],
            GUIDANCE,
            &mut combined,
        );
        assert_eq!(
            tokens[p],
            argmax(&combined),
            "pos {p}: CFG-greedy {} vs parallel combined argmax {}",
            tokens[p],
            argmax(&combined)
        );
    }
    eprintln!("CFG-greedy matches parallel combined argmax at all positions");
}

#[test]
fn cfg_topk_sampling_reproducible_and_within_topk() {
    let c = cfg();
    let embed = c.embed_dim as usize;
    let enc_seq = c.encoder_seq_len as usize;
    let levels = c.num_levels as usize;
    let vocab = c.vocab_size as usize;
    let top_k = 5usize;

    let w = make_weights(&c);
    let mut r = Rng(0x0000_5A11_9999);
    let enc_pos = r.vec(enc_seq * embed);
    let enc_neg = r.vec(enc_seq * embed);
    let table = w["shared_token_embedder"].clone();

    let mut tpos = build_temporal(&c, &w, &enc_pos);
    let mut tneg = build_temporal(&c, &w, &enc_neg);
    let mut depth = build_depth(&c, &w);

    let opts = DecodeOptions {
        num_frames: FRAMES,
        sos_id: SOS_ID,
        guidance_weight: GUIDANCE,
        temperature: 0.9,
        top_k,
        seed: 777,
    };

    let t1 = decode(&c, &mut tpos, Some(&mut tneg), &mut depth, &table, &opts);
    // Reproducibility: re-zero caches and decode again with the same seed.
    zero_caches(&mut tpos, &c);
    zero_caches(&mut tneg, &c);
    let t2 = decode(&c, &mut tpos, Some(&mut tneg), &mut depth, &table, &opts);
    assert_eq!(t1, t2, "same seed must reproduce the same tokens");
    eprintln!("CFG top-k grid: {t1:?}");

    // Every sampled token must be within the top-k of the parallel combined logits.
    let dit = shift_right(&t1, SOS_ID);
    let pos = parallel_logits(&c, &w, &dit, &enc_pos);
    let neg = parallel_logits(&c, &w, &dit, &enc_neg);
    for p in 0..FRAMES * levels {
        let mut combined = vec![0.0_f32; vocab];
        cfg_combine(
            &pos[p * vocab..(p + 1) * vocab],
            &neg[p * vocab..(p + 1) * vocab],
            GUIDANCE,
            &mut combined,
        );
        let mut idx: Vec<usize> = (0..vocab).collect();
        idx.sort_by(|&a, &b| combined[b].partial_cmp(&combined[a]).unwrap());
        let topk = &idx[..top_k];
        assert!(
            topk.contains(&(t1[p] as usize)),
            "pos {p}: sampled token {} not in top-{top_k} {:?}",
            t1[p],
            topk
        );
    }
    eprintln!("CFG top-k samples all within parallel top-{top_k}");
}
