//! End-to-end greedy autoregressive decode (`decode_greedy`): drives the
//! incremental temporal KV-cache decode + per-frame depth decode + greedy
//! sampling + token feedback, and checks self-consistency against the
//! independently-verified parallel decoder (`build_decoder`).
//!
//! The check: greedily decode a token grid, then teacher-force the SOS-padded
//! grid through `build_decoder` and assert each decoded token is the argmax of
//! the corresponding parallel-logits row. This exercises the whole generation
//! loop on the GPU (lavapipe) and ties it to the verified parallel forward.
//! head_dim=64 (real value, required by the cached-attention kernels).

use std::collections::HashMap;

use meganeura::models::magenta_rt::llm::{
    build_decoder_faithful, build_depth_decoder_stack, build_temporal_decode_step, decode_greedy,
    LlmConfig,
};
use meganeura::models::magenta_rt::sampling::argmax;
use meganeura::Graph;

const TEMPORAL_LAYERS: usize = 2;
const DEPTH_LAYERS: usize = 2;
const FRAMES: usize = 3;
const LEVELS: usize = 4;
const SOS_ID: u32 = 1;

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

/// Set every parameter a session declares from the weight map (panics if any is
/// missing, which keeps the map honest).
fn load(s: &mut meganeura::Session, w: &HashMap<String, Vec<f32>>) {
    for (name, _) in s.plan().param_buffers.clone() {
        let data = w
            .get(&name)
            .unwrap_or_else(|| panic!("missing weight: {name}"));
        s.set_parameter(&name, data);
    }
}

#[test]
fn greedy_decode_consistent_with_parallel() {
    let c = cfg();
    let embed = c.embed_dim as usize;
    let vocab = c.vocab_size as usize;
    let levels = c.num_levels as usize;
    let enc_seq = c.encoder_seq_len as usize;
    let attn_dim = (c.num_heads * c.head_dim) as usize;

    // --- Shared weights (one set drives all three graphs) ---
    let mut r = Rng(0x0000_DEC0_DE99);
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
        temporal_layer_weights(&c, &mut w, &format!("decoder.temporal_layers.{i}"), &mut r);
    }
    for i in 0..DEPTH_LAYERS {
        depth_layer_weights(&c, &mut w, &format!("decoder.depth_layers.{i}"), &mut r);
    }
    // Temporal KV caches (zero).
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
    let enc = r.vec(enc_seq * embed);

    // --- Temporal-step session ---
    let mut tg = Graph::new();
    let step_tok = tg.input_u32("step_tokens", &[levels]);
    let enc_node = tg.input("enc_out", &[enc_seq, embed]);
    let kv_pos = tg.input_u32("kv_pos", &[1]);
    let state = build_temporal_decode_step(&mut tg, &c, step_tok, enc_node, kv_pos, FRAMES);
    tg.set_outputs(vec![state]);
    let mut temporal = meganeura::build_inference_session(&tg);
    load(&mut temporal, &w);
    temporal.set_input("enc_out", &enc);

    // --- Depth-stack session ---
    let mut dg = Graph::new();
    let depth_in = dg.input("depth_inputs", &[levels, embed]);
    let dlogits = build_depth_decoder_stack(&mut dg, &c, depth_in);
    dg.set_outputs(vec![dlogits]);
    let mut depth = meganeura::build_inference_session(&dg);
    load(&mut depth, &w);

    // --- Greedy decode ---
    let table = w["shared_token_embedder"].clone();
    let tokens = decode_greedy(&c, &mut temporal, &mut depth, &table, FRAMES, SOS_ID);
    assert_eq!(tokens.len(), FRAMES * levels);
    assert!(tokens.iter().all(|&t| (t as usize) < vocab));
    eprintln!("decoded grid: {tokens:?}");

    // --- Parallel cross-check against the faithful decoder (PE + edge-pad) ---
    // The faithful decoder consumes the flat `decoder_input_tokens` the model
    // actually receives = shift_right(decoded grid): a BOS/SOS at position 0,
    // then the decoded tokens (drop the last). Teacher-forcing it must reproduce
    // the greedy argmax at every output position (the incremental decode now adds
    // the same FixedEmbed PE the faithful path does — LLM_FINDINGS §2.6).
    let mut dit = vec![SOS_ID; 1];
    dit.extend_from_slice(&tokens[..FRAMES * levels - 1]);
    let mut pg = Graph::new();
    let ptok = pg.input_u32("dec_tokens", &[FRAMES * levels]);
    let penc = pg.input("enc_out", &[enc_seq, embed]);
    let plogits = build_decoder_faithful(&mut pg, &c, ptok, penc, FRAMES);
    pg.set_outputs(vec![plogits]);
    let mut parallel = meganeura::build_inference_session(&pg);
    load(&mut parallel, &w);
    parallel.set_input_u32("dec_tokens", &dit);
    parallel.set_input("enc_out", &enc);
    parallel.step();
    parallel.wait();
    let logits = parallel.read_output(FRAMES * levels * vocab);

    for pos in 0..FRAMES * levels {
        let row = &logits[pos * vocab..(pos + 1) * vocab];
        let want = argmax(row);
        assert_eq!(
            tokens[pos],
            want,
            "pos {pos} (frame {}, level {}): greedy {} vs parallel argmax {}",
            pos / levels,
            pos % levels,
            tokens[pos],
            want
        );
    }
    eprintln!(
        "greedy decode matches parallel argmax at all {} positions",
        FRAMES * levels
    );
}
