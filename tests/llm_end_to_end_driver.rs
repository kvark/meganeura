//! End-to-end LLM orchestration (`driver::generate_token_grid`): assemble an
//! encoder-input token sequence, run the LLM **encoder** on the positive and the
//! masked-style negative passes, feed each encoder output into its temporal
//! decode-step session, and run the CFG decode loop to a token grid — all on the
//! GPU (lavapipe). This is the weight-independent half of the full pipeline
//! (everything except the SpectroStream/MusicCoCa front-ends and the audio
//! decoder), wired exactly as the deployed system runs it.
//!
//! Cross-check: with `guidance_weight = 1` the CFG-combined logits
//! (`neg + w·(pos − neg)`) collapse to the positive logits, so the grid must
//! equal the plain greedy decode against the positive encoder output. That ties
//! the orchestration to the already-verified `decode`/`decode_greedy` path.
//! head_dim=64 (the real value the cached/full attention kernels require).

use std::collections::HashMap;

use meganeura::models::magenta_rt::driver::{assemble_encoder_input, generate_token_grid};
use meganeura::models::magenta_rt::llm::{
    build_depth_decoder_stack, build_encoder_graph, build_temporal_decode_step, decode_greedy,
    DecodeOptions, LlmConfig,
};
use meganeura::models::magenta_rt::MagentaRtConfig;
use meganeura::{build_inference_session, Graph, Session};

const TEMPORAL_LAYERS: usize = 2;
const DEPTH_LAYERS: usize = 2;
const FRAMES: usize = 3;
const LEVELS: usize = 4;
const ENCODER_LAYERS: usize = 2;
const SOS_ID: u32 = 1;

/// A small Magenta-RT vocab/layout so the assembled encoder input is short
/// enough to run the whole encoder on lavapipe, while keeping the real per-level
/// offset arithmetic exercised by `assemble_encoder_input`.
fn mrt_cfg() -> MagentaRtConfig {
    MagentaRtConfig {
        chunk_length_sec: 2.0,
        context_length_sec: 0.2, // 5 context frames (× 4 levels = 20 codec tokens)
        crossfade_length_sec: 0.04,
        codec_sample_rate: 48000,
        codec_frame_rate: 25,
        codec_num_channels: 2,
        codec_rvq_codebook_size: 4,
        style_embedding_dim: 768,
        style_rvq_codebook_size: 4,
        encoder_codec_rvq_depth: 4,
        encoder_style_rvq_depth: 6,
        decoder_codec_rvq_depth: 16,
    }
}

fn llm_cfg(enc_seq: usize, vocab: usize) -> LlmConfig {
    LlmConfig {
        embed_dim: 128,
        head_dim: 64,
        num_heads: 2,
        mlp_dim: 64,
        num_encoder_layers: ENCODER_LAYERS as u32,
        num_temporal_decoder_layers: TEMPORAL_LAYERS as u32,
        num_depth_decoder_layers: DEPTH_LAYERS as u32,
        num_levels: LEVELS as u32,
        vocab_size: vocab as u32,
        encoder_seq_len: enc_seq as u32,
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

fn attn_block(c: &LlmConfig, w: &mut HashMap<String, Vec<f32>>, prefix: &str, r: &mut Rng) {
    let embed = c.embed_dim as usize;
    let attn_dim = (c.num_heads * c.head_dim) as usize;
    let mlp = c.mlp_dim as usize;
    for kind in ["self_attn", "cross_attn"] {
        for n in ["q", "k", "v"] {
            w.insert(format!("{prefix}.{kind}.{n}"), r.vec(embed * attn_dim));
        }
        w.insert(format!("{prefix}.{kind}.o"), r.vec(attn_dim * embed));
    }
    for nm in ["pre_self_attn_norm", "pre_cross_attn_norm", "pre_mlp_norm"] {
        w.insert(
            format!("{prefix}.{nm}"),
            (0..embed).map(|_| 1.0 + r.next()).collect(),
        );
    }
    w.insert(format!("{prefix}.mlp.w_gate"), r.vec(embed * mlp));
    w.insert(format!("{prefix}.mlp.w_up"), r.vec(embed * mlp));
    w.insert(format!("{prefix}.mlp.w_down"), r.vec(mlp * embed));
}

fn load(s: &mut Session, w: &HashMap<String, Vec<f32>>) {
    for (name, _) in s.plan().param_buffers.clone() {
        let data = w
            .get(&name)
            .unwrap_or_else(|| panic!("missing weight: {name}"));
        s.set_parameter(&name, data);
    }
}

#[test]
fn end_to_end_orchestration_matches_greedy() {
    let mrt = mrt_cfg();
    let enc_seq = mrt.encoder_input_length() as usize;
    // The LLM vocab must cover the assembled style offsets.
    let vocab = mrt.vocab_size() as usize;
    let c = llm_cfg(enc_seq, vocab);
    let embed = c.embed_dim as usize;
    let attn_dim = (c.num_heads * c.head_dim) as usize;
    let levels = c.num_levels as usize;

    // --- Assemble the encoder inputs (positive + masked-style negative) ---
    let ctx_frames = mrt.context_length_frames() as usize;
    let depth = mrt.encoder_codec_rvq_depth as usize;
    let context: Vec<u32> = (0..ctx_frames * depth)
        .map(|i| (i as u32) % mrt.codec_rvq_codebook_size)
        .collect();
    let style: Vec<u32> = (0..mrt.encoder_style_rvq_depth)
        .map(|i| i % mrt.style_rvq_codebook_size)
        .collect();
    let pos_tokens = assemble_encoder_input(&context, Some(&style), &mrt);
    let neg_tokens = assemble_encoder_input(&context, None, &mrt);
    assert_eq!(pos_tokens.len(), enc_seq);
    assert!(pos_tokens.iter().all(|&t| (t as usize) < vocab));
    assert!(neg_tokens.iter().all(|&t| (t as usize) < vocab));

    // --- Shared weights for all graphs ---
    let mut r = Rng(0xA11C_E5ED);
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
        attn_block(&c, &mut w, &format!("decoder.temporal_layers.{i}"), &mut r);
    }
    for i in 0..DEPTH_LAYERS {
        // Depth layers have no cross-attn; reuse attn_block and just don't read it.
        let prefix = format!("decoder.depth_layers.{i}");
        let embedn = embed;
        for n in ["q", "k", "v"] {
            w.insert(format!("{prefix}.self_attn.{n}"), r.vec(embedn * attn_dim));
        }
        w.insert(format!("{prefix}.self_attn.o"), r.vec(attn_dim * embedn));
        for nm in ["pre_self_attn_norm", "pre_mlp_norm"] {
            w.insert(
                format!("{prefix}.{nm}"),
                (0..embedn).map(|_| 1.0 + r.next()).collect(),
            );
        }
        w.insert(
            format!("{prefix}.mlp.w_gate"),
            r.vec(embedn * c.mlp_dim as usize),
        );
        w.insert(
            format!("{prefix}.mlp.w_up"),
            r.vec(embedn * c.mlp_dim as usize),
        );
        w.insert(
            format!("{prefix}.mlp.w_down"),
            r.vec(c.mlp_dim as usize * embedn),
        );
    }
    for i in 0..ENCODER_LAYERS {
        let prefix = format!("encoder.layers.{i}");
        // Encoder self-attn (bidirectional) params: `attn.{q,k,v,o}`.
        for n in ["q", "k", "v"] {
            w.insert(format!("{prefix}.attn.{n}"), r.vec(embed * attn_dim));
        }
        w.insert(format!("{prefix}.attn.o"), r.vec(attn_dim * embed));
        for nm in ["pre_attn_norm", "pre_mlp_norm"] {
            w.insert(
                format!("{prefix}.{nm}"),
                (0..embed).map(|_| 1.0 + r.next()).collect(),
            );
        }
        w.insert(
            format!("{prefix}.mlp.w_gate"),
            r.vec(embed * c.mlp_dim as usize),
        );
        w.insert(
            format!("{prefix}.mlp.w_up"),
            r.vec(embed * c.mlp_dim as usize),
        );
        w.insert(
            format!("{prefix}.mlp.w_down"),
            r.vec(c.mlp_dim as usize * embed),
        );
    }
    w.insert(
        "encoder.final_norm".into(),
        (0..embed).map(|_| 1.0 + r.next()).collect(),
    );
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

    // --- Build sessions ---
    let mut eg = Graph::new();
    let enc_out = build_encoder_graph(&mut eg, &c, enc_seq);
    eg.set_outputs(vec![enc_out]);
    let mut encoder = build_inference_session(&eg);
    load(&mut encoder, &w);

    let build_temporal = |w: &HashMap<String, Vec<f32>>| -> Session {
        let mut tg = Graph::new();
        let step_tok = tg.input_u32("step_tokens", &[levels]);
        let enc_node = tg.input("enc_out", &[enc_seq, embed]);
        let kv_pos = tg.input_u32("kv_pos", &[1]);
        let state = build_temporal_decode_step(&mut tg, &c, step_tok, enc_node, kv_pos, FRAMES);
        tg.set_outputs(vec![state]);
        let mut s = build_inference_session(&tg);
        load(&mut s, w);
        s
    };
    let mut temporal_pos = build_temporal(&w);
    let mut temporal_neg = build_temporal(&w);

    let mut dg = Graph::new();
    let depth_in = dg.input("depth_inputs", &[levels, embed]);
    let dlogits = build_depth_decoder_stack(&mut dg, &c, depth_in);
    dg.set_outputs(vec![dlogits]);
    let mut depth = build_inference_session(&dg);
    load(&mut depth, &w);

    let table = w["shared_token_embedder"].clone();

    // --- Orchestrated CFG decode with guidance_weight = 1 (≡ positive only) ---
    // cfg_combine = neg + w·(pos − neg), so w = 1 collapses exactly to the
    // positive logits — the orchestration must then equal greedy(positive).
    let opts = DecodeOptions {
        num_frames: FRAMES,
        sos_id: SOS_ID,
        guidance_weight: 1.0,
        temperature: 0.0, // greedy
        top_k: 0,
        seed: 0,
    };
    let grid = generate_token_grid(
        &c,
        &mut encoder,
        &pos_tokens,
        Some(&neg_tokens),
        &mut temporal_pos,
        Some(&mut temporal_neg),
        &mut depth,
        &table,
        &opts,
    );
    assert_eq!(grid.len(), FRAMES * levels);
    assert!(grid.iter().all(|&t| (t as usize) < vocab));

    // --- Reference: plain greedy against the positive encoder output ---
    // Rebuild fresh temporal/depth sessions so caches start clean.
    let mut temporal_ref = build_temporal(&w);
    let mut dg2 = Graph::new();
    let depth_in2 = dg2.input("depth_inputs", &[levels, embed]);
    let dlogits2 = build_depth_decoder_stack(&mut dg2, &c, depth_in2);
    dg2.set_outputs(vec![dlogits2]);
    let mut depth_ref = build_inference_session(&dg2);
    load(&mut depth_ref, &w);

    encoder.set_input_u32("encoder_input_tokens", &pos_tokens);
    encoder.step();
    encoder.wait();
    let enc_pos = encoder.read_output(enc_seq * embed);
    temporal_ref.set_input("enc_out", &enc_pos);

    let want = decode_greedy(
        &c,
        &mut temporal_ref,
        &mut depth_ref,
        &table,
        FRAMES,
        SOS_ID,
    );
    assert_eq!(
        grid, want,
        "orchestrated CFG(w=0) must equal greedy(positive)"
    );
    eprintln!(
        "end-to-end orchestration matches greedy at all {} tokens",
        grid.len()
    );
}
