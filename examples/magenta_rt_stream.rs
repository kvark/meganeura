//! **Continuous** (streaming) Magenta-RT generation on real weights: a text
//! prompt → an arbitrary number of seamlessly-stitched 2 s chunks of 48 kHz
//! stereo audio.
//!
//! This extends `magenta_rt_generate.rs` (one 2 s chunk) into the real streaming
//! loop. Each chunk conditions on the previous 10 s of **generated codec tokens**
//! (token-domain continuation — no audio re-encoding in the loop), mirroring
//! `MagentaRTT5X.generate_chunk` in magenta-realtime v1:
//!
//! ```text
//!   prompt ─► MusicCoCa ─► 6 style tokens ──────────────┐  (fixed across chunks)
//!                                                        │
//!   ┌────────────────────────── per chunk ──────────────┼─────────────────────┐
//!   │ StreamState.context_codec (250×4) ────────────────┴► assemble (1006)     │
//!   │        │                                              LLM CFG decode      │
//!   │        │                                              ▼  50×16 grid       │
//!   │  boundary_frame ++ grid ─► SpectroStream decode ─► 51-frame audio         │
//!   │        │                                              │                   │
//!   │  StreamState.push_chunk(grid)  ◄──────────────────────┘                   │
//!   └──────────────────────────────────────────────────────────────────────────┘
//!     chunks ─► crossfade_chunks (40 ms eqpower seam) ─► continuous WAV
//! ```
//!
//! Each decoded chunk prepends the previous chunk's last (boundary) frame, so
//! consecutive chunks share exactly one overlap frame; [`crossfade_chunks`]
//! eqpower-blends that 40 ms seam. `N` chunks ⇒ `N * 2 s` of continuous audio.
//!
//! Reads the same `magenta_rt_codec_dump/` bundle as `magenta_rt_generate.rs`,
//! plus `musiccoca_vocab.model` (SentencePiece) so prompts are tokenized at
//! runtime. Writes one WAV per prompt to `/tmp/magenta_rt_stream_<i>.wav`.
//!
//! Run (slow on lavapipe — each chunk is a full 50-frame CFG decode):
//!   cargo run --release --example magenta_rt_stream
//! Env knobs: `MRT_FRAMES` (frames/chunk, default 50), `MRT_CHUNKS` (chunks/
//! prompt, default 3), `MRT_PROMPTS` (`;`-separated prompt list).

use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

use meganeura::data::safetensors::SafeTensorsModel;
use meganeura::models::magenta_rt::driver::{
    assemble_encoder_input, crossfade_chunks, generate_token_grid, llm_grid_to_rvq, CrossfadeStyle,
    StreamState,
};
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
use meganeura::models::magenta_rt::tokenizer::{musiccoca_token_ids, SpmModel};
use meganeura::models::magenta_rt::MagentaRtConfig;
use meganeura::{build_inference_session, Graph, Session};

const DUMP: &str = "magenta_rt_codec_dump";

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
    let frames = env_usize("MRT_FRAMES", mrt.chunk_length_frames() as usize); // 50
    let chunks = env_usize("MRT_CHUNKS", 3);
    let prompts: Vec<String> = std::env::var("MRT_PROMPTS")
        .ok()
        .map(|s| s.split(';').map(|p| p.trim().to_string()).filter(|p| !p.is_empty()).collect())
        .unwrap_or_else(|| {
            vec![
                "funky upbeat jazz".to_string(),
                "ambient synth pads, slow and dreamy".to_string(),
                "driving techno with a heavy bassline".to_string(),
            ]
        });

    let spm = {
        let path = format!("{DUMP}/musiccoca_vocab.model");
        if !Path::new(&path).exists() {
            eprintln!("missing {path} — needed to tokenize prompts");
            std::process::exit(1);
        }
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("read {path}: {e}"));
        SpmModel::from_bytes(&bytes).unwrap_or_else(|e| panic!("parse spm: {e}"))
    };

    // Build the (prompt-independent) LLM + SpectroStream machinery once.
    let mut llm = Llm::build(frames);
    let mut codec = Codec::build(frames + 1); // decode boundary-frame + chunk

    for (pi, prompt) in prompts.iter().enumerate() {
        println!("\n=== prompt {pi}: {prompt:?} ({chunks} chunks × {frames} frames) ===");
        let style = style_tokens(&spm, prompt, &mrt);
        println!("style tokens: {style:?}");

        let mut state = StreamState::cold_start(&mrt);
        let mut chunk_audios: Vec<Vec<f32>> = Vec::with_capacity(chunks);

        for ci in 0..chunks {
            // 1. Assemble this chunk's encoder inputs from the rolling context.
            let ctx = state.context_codec();
            let pos = assemble_encoder_input(&ctx, Some(&style), &mrt);
            let neg = assemble_encoder_input(&ctx, None, &mrt);

            // 2. LLM CFG decode → 50×16 grid (unified-vocab tokens) → raw RVQ.
            let vocab_grid = llm.generate(&pos, &neg, frames);
            let (grid, oor) = llm_grid_to_rvq(&vocab_grid, &mrt);
            if oor > 0 {
                println!("  chunk {ci}: {oor} out-of-range tokens clamped");
            }

            // 3. Decode [boundary ++ grid] so the chunk's first frame re-decodes
            //    the previous chunk's last frame — the 1-frame crossfade seam.
            let mut dec_in = Vec::with_capacity((frames + 1) * 16);
            dec_in.extend_from_slice(state.boundary_frame());
            dec_in.extend_from_slice(&grid);
            let mut audio = codec.decode(&dec_in, frames + 1);

            // Chunk 0 has no previous chunk to seam with: its boundary is the
            // cold-start frame (not real music), so drop that leading frame for a
            // clean onset. Later chunks keep it — that frame is the overlap the
            // crossfade consumes.
            let frame_samples = mrt.frame_length_samples() as usize * mrt.codec_num_channels as usize;
            if ci == 0 {
                audio.drain(..frame_samples.min(audio.len()));
            }

            // 4. Slide the token window for the next chunk.
            state.push_chunk(&grid);

            let secs = audio.len() as f32 / 2.0 / mrt.codec_sample_rate as f32;
            println!(
                "  chunk {ci}: grid {} tok, audio {} samples ({:.2}s), range [{:.3}, {:.3}]",
                grid.len(),
                audio.len(),
                secs,
                audio.iter().cloned().fold(f32::INFINITY, f32::min),
                audio.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
            );
            chunk_audios.push(audio);
            std::io::stdout().flush().ok();
        }

        // Stitch the chunks: each shares a 1-frame (1920-sample) eqpower seam.
        let refs: Vec<&[f32]> = chunk_audios.iter().map(|c| c.as_slice()).collect();
        let overlap = mrt.crossfade_length_samples() as usize; // 1920
        let mut stream = crossfade_chunks(&refs, overlap, mrt.codec_num_channels as usize, CrossfadeStyle::EqPower);

        // Peak-normalize the whole stream to -0.5 dBFS for clip-free inspection
        // (the codec output can run hot; scaling preserves the waveform shape and
        // relative dynamics across the stream).
        let peak = stream.iter().fold(0.0_f32, |m, &s| m.max(s.abs()));
        if peak > 1e-6 {
            let gain = 0.944 / peak; // -0.5 dBFS
            for s in &mut stream {
                *s *= gain;
            }
        }

        let out = format!("/tmp/magenta_rt_stream_{pi}.wav");
        write_wav_pcm16(&out, &stream, mrt.codec_sample_rate).unwrap();
        let secs = stream.len() as f32 / 2.0 / mrt.codec_sample_rate as f32;
        println!("  wrote {out}  ({:.2}s continuous, raw peak {:.3})", secs, peak);
    }
}

/// MusicCoCa text tower: prompt → first 6 style RVQ tokens.
fn style_tokens(spm: &SpmModel, prompt: &str, mrt: &MagentaRtConfig) -> Vec<u32> {
    let cfg = MusicCoCaConfig::default();
    let weights = require(&format!("{DUMP}/weights_musiccoca.safetensors"));
    let gate = require(&format!("{DUMP}/musiccoca_gate.safetensors"));
    let codebooks = gate.tensor_f32("codebooks").expect("gate codebooks");

    let ids = musiccoca_token_ids(spm, prompt);
    let mut g = Graph::new();
    let out = build_text_encoder_graph(&mut g, &cfg, ids.len());
    g.set_outputs(vec![out]);
    let mut s = build_inference_session(&g);
    load_text_encoder_weights(&weights, &mut s, &cfg).expect("load musiccoca weights");
    s.set_input_u32("text_tokens", &ids);
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
    tokens[..mrt.encoder_style_rvq_depth as usize].to_vec()
}

/// The base LLM (encoder + temporal pos/neg + depth), built once and reused for
/// every chunk. Each chunk re-runs the encoder (new context) and re-zeros the
/// temporal KV caches before the 50-frame CFG decode.
struct Llm {
    cfg: LlmConfig,
    encoder: Session,
    temporal_pos: Session,
    temporal_neg: Session,
    depth: Session,
    table: Vec<f32>,
    cache_names: Vec<String>,
    cache_frames: usize,
    opts_seed: u64,
}

impl Llm {
    fn build(num_frames: usize) -> Self {
        let c = LlmConfig::base();
        let weights = require(&format!("{DUMP}/weights_llm_base.safetensors"));
        let mrt = MagentaRtConfig::default();
        let enc_seq = mrt.encoder_input_length() as usize;
        let embed = c.embed_dim as usize;
        let levels = c.num_levels as usize;

        let mut eg = Graph::new();
        let enc_out = build_encoder_graph(&mut eg, &c, enc_seq);
        eg.set_outputs(vec![enc_out]);
        let mut encoder = build_inference_session(&eg);
        zero_caches(&mut encoder, &weights, &c, num_frames);

        let build_temporal = |weights: &SafeTensorsModel| -> (Session, Vec<String>) {
            let mut tg = Graph::new();
            let step_tok = tg.input_u32("step_tokens", &[levels]);
            let enc_node = tg.input("enc_out", &[enc_seq, embed]);
            let kv_pos = tg.input_u32("kv_pos", &[1]);
            let state = build_temporal_decode_step(&mut tg, &c, step_tok, enc_node, kv_pos, num_frames);
            tg.set_outputs(vec![state]);
            let mut s = build_inference_session(&tg);
            let names = zero_caches(&mut s, weights, &c, num_frames);
            (s, names)
        };
        let (temporal_pos, cache_names) = build_temporal(&weights);
        let (temporal_neg, _) = build_temporal(&weights);

        let mut dg = Graph::new();
        let depth_in = dg.input("depth_inputs", &[levels, embed]);
        let dlogits = build_depth_decoder_stack(&mut dg, &c, depth_in);
        dg.set_outputs(vec![dlogits]);
        let mut depth = build_inference_session(&dg);
        zero_caches(&mut depth, &weights, &c, num_frames);

        let table = weights
            .tensor_f32_auto("target.token_embedder.embedding")
            .expect("token embedder");

        Llm {
            cfg: c,
            encoder,
            temporal_pos,
            temporal_neg,
            depth,
            table,
            cache_names,
            cache_frames: num_frames,
            opts_seed: 0x4D52_5447,
        }
    }

    fn generate(&mut self, pos: &[u32], neg: &[u32], num_frames: usize) -> Vec<u32> {
        // Each chunk is an independent 50-frame autoregressive decode; clear the
        // temporal KV caches so no state leaks across chunk boundaries.
        let attn = (self.cfg.num_heads * self.cfg.head_dim) as usize;
        let zeros = vec![0.0_f32; self.cache_frames * attn];
        for name in &self.cache_names {
            self.temporal_pos.set_parameter(name, &zeros);
            self.temporal_neg.set_parameter(name, &zeros);
        }

        let opts = DecodeOptions {
            num_frames,
            sos_id: MagentaRtConfig::default().vocab_mask_token(),
            guidance_weight: 4.0,
            temperature: 1.1,
            top_k: 40,
            seed: self.opts_seed,
        };
        generate_token_grid(
            &self.cfg,
            &mut self.encoder,
            pos,
            Some(neg),
            &mut self.temporal_pos,
            Some(&mut self.temporal_neg),
            &mut self.depth,
            &self.table,
            &opts,
        )
    }
}

/// The SpectroStream decoder, built once for a fixed frame count and reused.
struct Codec {
    cfg: SpectroStreamConfig,
    session: Session,
    codebooks: Vec<f32>,
    weights: SafeTensorsModel,
}

impl Codec {
    fn build(num_frames: usize) -> Self {
        let cfg = SpectroStreamConfig::default();
        let weights = require(&format!("{DUMP}/weights_spectrostream.safetensors"));
        let codebooks = weights.tensor_f32_auto("quantizer.rvq_codebooks").unwrap();
        let mut g = Graph::new();
        let out = build_decoder_graph(&mut g, &cfg, num_frames as u32);
        g.set_outputs(vec![out]);
        let mut session = build_inference_session(&g);
        load_decoder_weights(&weights, &mut session).unwrap();
        Codec { cfg, session, codebooks, weights }
    }

    /// Decode a `[num_frames × 16]` token grid → interleaved stereo, trimmed to
    /// exactly `num_frames` frames (drops the decoder's trailing temporal-pad
    /// frame so chunks tile cleanly).
    fn decode(&mut self, grid: &[u32], num_frames: usize) -> Vec<f32> {
        let depth = 16usize;
        let embed = dequantize_tokens(
            grid,
            num_frames,
            &self.codebooks,
            depth,
            self.cfg.codebook_size as usize,
            self.cfg.embedding_dim as usize,
        );
        let preprocessed = input_layer_preprocess(&embed, num_frames, &self.weights, &self.cfg);
        self.session.set_input("decoder_input_preprocessed", &preprocessed);
        self.session.step();
        self.session.wait();
        let out_frames = (num_frames + self.cfg.temporal_pad as usize) * 4;
        let body = self.session.read_output(out_frames * 480 * 4);
        let audio = decoder_body_to_audio(&body, out_frames, &IstftConfig::default());
        // istft yields (num_frames + temporal_pad) audio frames; keep the first
        // num_frames (× 1920 samples × 2 channels).
        let keep = num_frames * 1920 * 2;
        audio[..keep.min(audio.len())].to_vec()
    }
}

/// Load checkpoint params and zero-initialise the temporal KV caches (no
/// checkpoint tensor); returns the cache parameter names.
fn zero_caches(s: &mut Session, weights: &SafeTensorsModel, c: &LlmConfig, cache_frames: usize) -> Vec<String> {
    let skipped =
        meganeura::models::magenta_rt::llm_weights::load_llm_weights(s, weights, c).expect("load llm");
    let mut names = Vec::new();
    for name in skipped {
        if name.contains("temporal_kv_cache") {
            let attn = (c.num_heads * c.head_dim) as usize;
            s.set_parameter(&name, &vec![0.0_f32; cache_frames * attn]);
            names.push(name);
        }
    }
    names
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
