//! Full Magenta-RT pipeline, end to end on real weights: a **text prompt** →
//! 2 s of 48 kHz stereo audio, exercising every ported component.
//!
//! ```text
//! prompt ─► MusicCoCa text tower ─► RVQ ─► 6 style tokens ┐
//! cold-start silence context ─────────────► 250×4 codec ──┴► assemble (1006)
//!                                                  │  LLM encoder (pos + masked neg)
//!                                                  ▼  LLM CFG decode (50×16 tokens)
//!                                            SpectroStream dequantize + decode body
//!                                                  ▼  host iSTFT
//!                                            2 s audio ─► WAV
//! ```
//!
//! This is the capstone that ties the individually-verified stages
//! (`musiccoca`, `driver`, `llm`, `spectrostream`) into one run. It is *not* a
//! correctness gate — each stage has its own real-weight gate test — but a
//! living demonstration that the assembled pipeline produces audio from text.
//!
//! Reads from `magenta_rt_codec_dump/`:
//!   - `weights_musiccoca.safetensors` + `musiccoca_gate.safetensors` (codebooks)
//!   - `weights_llm_base.safetensors`
//!   - `weights_spectrostream.safetensors`
//! Writes `/tmp/magenta_rt_generate.wav`.
//!
//! Run (slow on lavapipe; set MRT_FRAMES=4 for a quick wiring smoke):
//!   cargo run --release --example magenta_rt_generate

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use meganeura::data::safetensors::SafeTensorsModel;
use meganeura::models::magenta_rt::driver::{assemble_encoder_input, generate_token_grid};
use meganeura::models::magenta_rt::llm::{
    build_depth_decoder_stack, build_encoder_graph, build_temporal_decode_step, DecodeOptions,
    LlmConfig,
};
use meganeura::models::magenta_rt::musiccoca::{
    build_text_encoder_graph, load_text_encoder_weights, rvq_quantize, MusicCoCaConfig,
};
use meganeura::models::magenta_rt::spectrostream::{
    build_decoder_graph, decoder_body_to_audio, dequantize_tokens, input_layer_preprocess,
    load_decoder_weights, IstftConfig, SpectroStreamConfig,
};
use meganeura::models::magenta_rt::MagentaRtConfig;
use meganeura::{build_inference_session, Graph, Session};

const DUMP: &str = "magenta_rt_codec_dump";
const OUT_WAV: &str = "/tmp/magenta_rt_generate.wav";

/// Prompt tokenized offline with `musiccoca_vocab.model` (lowercase →
/// SentencePiece → SOS=1 prefix): "funky upbeat jazz".
const PROMPT: &str = "funky upbeat jazz";
const PROMPT_IDS: &[u32] = &[1, 534, 354, 397];

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key).ok().and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn require(path: &str) -> SafeTensorsModel {
    if !Path::new(path).exists() {
        eprintln!("missing {path} — dump the weights first (tools/magenta_rt/)");
        std::process::exit(1);
    }
    SafeTensorsModel::load(path.into()).unwrap_or_else(|e| panic!("load {path}: {e}"))
}

fn main() {
    env_logger::init();
    let mrt = MagentaRtConfig::default();
    let num_frames = env_usize("MRT_FRAMES", mrt.chunk_length_frames() as usize); // 50

    // ===================== 1. Text → 6 style tokens (MusicCoCa) =====================
    let style_tokens = musiccoca_style_tokens(&mrt);
    println!("style tokens (6): {style_tokens:?}");

    // ===================== 2. Context codec tokens (cold start) =====================
    // The first generated chunk has no audio history — Magenta-RT cold-starts
    // from a silent context. A warm/streaming start would substitute the
    // SpectroStream encoder's tokens for the trailing 10 s of generated audio
    // (`spectrostream_encoder::{encode, rvq_encode}`, taking the first 4 levels).
    let ctx_frames = mrt.context_length_frames() as usize; // 250
    let ctx_depth = mrt.encoder_codec_rvq_depth as usize; // 4
    let context_codec = vec![0u32; ctx_frames * ctx_depth];

    // ===================== 3. Assemble the LLM encoder inputs =====================
    let pos_tokens = assemble_encoder_input(&context_codec, Some(&style_tokens), &mrt);
    let neg_tokens = assemble_encoder_input(&context_codec, None, &mrt);
    println!("encoder input length: {} (pos/neg)", pos_tokens.len());

    // ===================== 4. LLM encode + CFG decode → 50×16 grid =====================
    let grid = run_llm(&pos_tokens, &neg_tokens, num_frames);
    println!("decoded grid: {} tokens ({num_frames} frames × 16 levels)", grid.len());

    // ===================== 5. SpectroStream: tokens → 2 s audio =====================
    let audio = decode_audio(&grid, num_frames);
    println!(
        "audio: {} interleaved stereo samples, range [{:.4}, {:.4}]",
        audio.len(),
        audio.iter().cloned().fold(f32::INFINITY, f32::min),
        audio.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
    );

    write_wav_pcm16(OUT_WAV, &audio, mrt.codec_sample_rate).unwrap();
    println!("\nWrote {OUT_WAV}  ({PROMPT:?})");
}

/// Run the MusicCoCa text tower on the prompt and RVQ-quantize to the first 6
/// style tokens the LLM encoder consumes.
fn musiccoca_style_tokens(mrt: &MagentaRtConfig) -> Vec<u32> {
    let cfg = MusicCoCaConfig::default();
    let weights = require(&format!("{DUMP}/weights_musiccoca.safetensors"));
    // Codebooks (numeric RVQ-level order, [depth, embed, codebook_size]) ship in
    // the gate bundle, the same source the real-weight gate uses.
    let gate = require(&format!("{DUMP}/musiccoca_gate.safetensors"));
    let codebooks = gate.tensor_f32("codebooks").expect("gate codebooks");

    let seq = PROMPT_IDS.len();
    let mut g = Graph::new();
    let out = build_text_encoder_graph(&mut g, &cfg, seq);
    g.set_outputs(vec![out]);
    let mut s = build_inference_session(&g);
    load_text_encoder_weights(&weights, &mut s, &cfg).expect("load musiccoca weights");
    s.set_input_u32("text_tokens", PROMPT_IDS);
    s.step();
    s.wait();
    let emb = s.read_output(cfg.embed_dim as usize);

    let tokens = rvq_quantize(
        &emb,
        &codebooks,
        cfg.rvq_depth as usize,
        cfg.embed_dim as usize,
        cfg.codebook_size as usize,
    );
    // The LLM encoder takes the first `encoder_style_rvq_depth` (6) of the 12.
    tokens[..mrt.encoder_style_rvq_depth as usize].to_vec()
}

/// Build the base LLM (encoder + temporal pos/neg decode-step + depth) on real
/// weights and run the CFG decode loop to a `[num_frames * 16]` RVQ grid.
fn run_llm(pos_tokens: &[u32], neg_tokens: &[u32], num_frames: usize) -> Vec<u32> {
    let c = LlmConfig::base();
    let weights = require(&format!("{DUMP}/weights_llm_base.safetensors"));
    let enc_seq = pos_tokens.len();
    let embed = c.embed_dim as usize;
    let levels = c.num_levels as usize;

    // Encoder.
    let mut eg = Graph::new();
    let enc_out = build_encoder_graph(&mut eg, &c, enc_seq);
    eg.set_outputs(vec![enc_out]);
    let mut encoder = build_inference_session(&eg);
    load_and_zero_caches(&mut encoder, &weights, &c, num_frames);

    // Temporal decode-step sessions (positive + masked negative for CFG).
    let build_temporal = |weights: &SafeTensorsModel| -> Session {
        let mut tg = Graph::new();
        let step_tok = tg.input_u32("step_tokens", &[levels]);
        let enc_node = tg.input("enc_out", &[enc_seq, embed]);
        let kv_pos = tg.input_u32("kv_pos", &[1]);
        let state = build_temporal_decode_step(&mut tg, &c, step_tok, enc_node, kv_pos, num_frames);
        tg.set_outputs(vec![state]);
        let mut s = build_inference_session(&tg);
        load_and_zero_caches(&mut s, weights, &c, num_frames);
        s
    };
    let mut temporal_pos = build_temporal(&weights);
    let mut temporal_neg = build_temporal(&weights);

    // Depth decoder stack.
    let mut dg = Graph::new();
    let depth_in = dg.input("depth_inputs", &[levels, embed]);
    let dlogits = build_depth_decoder_stack(&mut dg, &c, depth_in);
    dg.set_outputs(vec![dlogits]);
    let mut depth = build_inference_session(&dg);
    load_and_zero_caches(&mut depth, &weights, &c, num_frames);

    let table = weights
        .tensor_f32_auto("target.token_embedder.embedding")
        .expect("token embedder");

    // Classifier-free guidance with the production sampling knobs.
    let opts = DecodeOptions {
        num_frames,
        sos_id: MagentaRtConfig::default().vocab_mask_token(),
        guidance_weight: 4.0,
        temperature: 1.1,
        top_k: 40,
        seed: 0x4D52_5447,
    };
    generate_token_grid(
        &c,
        &mut encoder,
        pos_tokens,
        Some(neg_tokens),
        &mut temporal_pos,
        Some(&mut temporal_neg),
        &mut depth,
        &table,
        &opts,
    )
}

/// Load the checkpoint params and zero-initialise any skipped runtime buffers
/// (the temporal KV caches, which carry no checkpoint tensor).
fn load_and_zero_caches(s: &mut Session, weights: &SafeTensorsModel, c: &LlmConfig, cache_frames: usize) {
    let skipped =
        meganeura::models::magenta_rt::llm_weights::load_llm_weights(s, weights, c).expect("load llm");
    for name in skipped {
        // Encoder PE has no params; only the KV caches need explicit zeroing.
        // The temporal caches are `[max_frames, attn]`, built with max_frames ==
        // cache_frames (the chunk's frame count).
        if name.contains("temporal_kv_cache") {
            let attn = (c.num_heads * c.head_dim) as usize;
            s.set_parameter(&name, &vec![0.0_f32; cache_frames * attn]);
        }
    }
}

/// Dequantize the LLM's 16-level grid, run the SpectroStream decoder body on the
/// GPU, and iSTFT to interleaved stereo audio.
fn decode_audio(grid: &[u32], num_frames: usize) -> Vec<f32> {
    let cfg = SpectroStreamConfig::default();
    let weights = require(&format!("{DUMP}/weights_spectrostream.safetensors"));
    let depth = 16usize; // the LLM produces 16 RVQ levels per frame
    let codebooks = weights.tensor_f32_auto("quantizer.rvq_codebooks").unwrap();

    let embed = dequantize_tokens(
        grid,
        num_frames,
        &codebooks,
        depth,
        cfg.codebook_size as usize,
        cfg.embedding_dim as usize,
    );
    let preprocessed = input_layer_preprocess(&embed, num_frames, &weights, &cfg);

    let mut g = Graph::new();
    let out = build_decoder_graph(&mut g, &cfg, num_frames as u32);
    g.set_outputs(vec![out]);
    let mut s = build_inference_session(&g);
    load_decoder_weights(&weights, &mut s).unwrap();
    s.set_input("decoder_input_preprocessed", &preprocessed);
    s.step();
    s.wait();
    // Body output T = (num_frames + temporal_pad) × 4.
    let out_frames = (num_frames + cfg.temporal_pad as usize) * 4;
    let body = s.read_output(out_frames * 480 * 4);
    decoder_body_to_audio(&body, out_frames, &IstftConfig::default())
}

/// Write 48 kHz stereo PCM-16 WAV.
fn write_wav_pcm16(path: &str, interleaved: &[f32], sample_rate: u32) -> std::io::Result<()> {
    let n_channels: u16 = 2;
    let bits: u16 = 16;
    let byte_rate = sample_rate * n_channels as u32 * (bits / 8) as u32;
    let block_align = n_channels * (bits / 8);
    let data_bytes = (interleaved.len() as u32) * 2;
    let f = File::create(path)?;
    let mut w = BufWriter::new(f);
    w.write_all(b"RIFF")?;
    w.write_all(&(36 + data_bytes).to_le_bytes())?;
    w.write_all(b"WAVE")?;
    w.write_all(b"fmt ")?;
    w.write_all(&16u32.to_le_bytes())?;
    w.write_all(&1u16.to_le_bytes())?;
    w.write_all(&n_channels.to_le_bytes())?;
    w.write_all(&sample_rate.to_le_bytes())?;
    w.write_all(&byte_rate.to_le_bytes())?;
    w.write_all(&block_align.to_le_bytes())?;
    w.write_all(&bits.to_le_bytes())?;
    w.write_all(b"data")?;
    w.write_all(&data_bytes.to_le_bytes())?;
    for &s in interleaved {
        let v = (s.clamp(-1.0, 1.0) * 32767.0) as i16;
        w.write_all(&v.to_le_bytes())?;
    }
    Ok(())
}
