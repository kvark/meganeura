//! End-to-end SpectroStream decoder demo: tokens → meganeura GPU body →
//! host iSTFT → WAV file.
//!
//! Reads:
//!   - `magenta_rt_codec_dump/weights_spectrostream.safetensors` — decoder
//!     weights (incl. quantizer codebooks for token dequantization)
//!   - `magenta_rt_codec_dump/reference_codec.safetensors` — the reference
//!     token sequence and TF's reference reconstructed audio (for comparison)
//!
//! Writes:
//!   - `/tmp/magenta_rt_demo.wav` — 48 kHz stereo PCM-16 reconstruction
//!
//! Run:
//!   cargo run --example magenta_rt_demo --release

use meganeura::build_inference_session;
use meganeura::data::safetensors::SafeTensorsModel;
use meganeura::models::magenta_rt::spectrostream::{
    build_decoder_graph, decoder_body_to_audio, dequantize_tokens, input_layer_preprocess,
    load_decoder_weights, IstftConfig, SpectroStreamConfig,
};
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

const WEIGHTS: &str = "magenta_rt_codec_dump/weights_spectrostream.safetensors";
const REF_CODEC: &str = "magenta_rt_codec_dump/reference_codec.safetensors";
const OUT_WAV: &str = "/tmp/magenta_rt_demo.wav";

const NUM_FRAMES: u32 = 50;
const SAMPLE_RATE: u32 = 48000;

fn main() {
    env_logger::init();

    if !Path::new(WEIGHTS).exists() {
        eprintln!("missing {WEIGHTS} — dump SpectroStream weights first");
        std::process::exit(1);
    }
    if !Path::new(REF_CODEC).exists() {
        eprintln!("missing {REF_CODEC} — dump reference tokens first");
        std::process::exit(1);
    }

    let weights = SafeTensorsModel::load(WEIGHTS.into()).unwrap();
    let refs = SafeTensorsModel::load(REF_CODEC.into()).unwrap();
    let cfg = SpectroStreamConfig::default();

    // 1. Load tokens [50, 64] and host-side preprocess: dequantize via 64
    // codebooks → embed [50, 256] → temporal_pad [51, 256] → input_layer
    // (conv1x1_a + ELU + conv1x1_b + parallel conv1x1) → reshape to NHWC
    // [1, 51, 5, 512] → transpose to NCHW [1, 512, 51, 5] for meganeura.
    let tokens_u = load_tokens_i32_as_usize(REF_CODEC, "tokens");
    println!("tokens: {} elements", tokens_u.len());

    // Full 64-level codec round-trip: dequantize → input_layer expansion → NCHW.
    let depth = 64usize;
    let codebook_size = 1024usize;
    let embed_dim = cfg.embedding_dim as usize;
    let tokens_u32: Vec<u32> = tokens_u.iter().map(|&t| t as u32).collect();
    let codebooks = weights.tensor_f32_auto("quantizer.rvq_codebooks").unwrap();
    let embed = dequantize_tokens(
        &tokens_u32,
        NUM_FRAMES as usize,
        &codebooks,
        depth,
        codebook_size,
        embed_dim,
    );
    let preprocessed_nchw = input_layer_preprocess(&embed, NUM_FRAMES as usize, &weights, &cfg);
    println!("preprocessed NCHW: {} f32 values", preprocessed_nchw.len());

    // 2. Build meganeura decoder graph through the body output, run inference.
    let mut g = meganeura::Graph::new();
    let out = build_decoder_graph(&mut g, &cfg, NUM_FRAMES);
    g.set_outputs(vec![out]);
    let mut s = build_inference_session(&g);
    load_decoder_weights(&weights, &mut s).unwrap();
    s.set_input("decoder_input_preprocessed", &preprocessed_nchw);
    println!("Running meganeura SpectroStream decoder body...");
    let t0 = std::time::Instant::now();
    s.step();
    s.wait();
    let elapsed = t0.elapsed();
    let body_out_nchw = s.read_output(1 * 4 * 200 * 480);
    println!("body inference: {elapsed:.2?}");
    let nans = body_out_nchw.iter().filter(|v| !v.is_finite()).count();
    if nans > 0 {
        println!("WARNING: {nans}/{} body samples are NaN/Inf — clamped to 0 \
                  for a playable WAV (RADV bug, not meganeura)",
                 body_out_nchw.len());
    }

    // 3. Host-side iSTFT: NCHW [1, 4, 200, 480] → audio [96000, 2]
    //    (decoder_body_to_audio zeroes non-finite samples internally).
    let audio = decoder_body_to_audio(&body_out_nchw, 200, &IstftConfig::default());

    // 4. Write WAV.
    println!("\nWriting {OUT_WAV}...");
    write_wav_pcm16(OUT_WAV, &audio, SAMPLE_RATE).unwrap();
    println!("Done. Range [{:.4}, {:.4}].", audio.iter().cloned().fold(f32::INFINITY, f32::min),
             audio.iter().cloned().fold(f32::NEG_INFINITY, f32::max));

    // Quality check against reference.
    if let Ok(ref_audio) = refs.tensor_f32_auto("reconstructed_audio") {
        // ref_audio is [96000, 2] flat (interleaved or not? safetensors).
        if ref_audio.len() == audio.len() {
            let mut nan_count = 0;
            let mut sum_diff = 0.0_f64;
            let mut sum_ref = 0.0_f64;
            for (i, (&our, &theirs)) in audio.iter().zip(ref_audio.iter()).enumerate() {
                if !our.is_finite() {
                    nan_count += 1;
                    if nan_count <= 3 { println!("  first non-finite at {i}: {our}"); }
                    continue;
                }
                sum_diff += (our - theirs).abs() as f64;
                sum_ref += (theirs as f64).abs();
            }
            let rel = sum_diff / sum_ref.max(1e-9);
            println!("Audio vs TF reference: rel_err={rel:.4e}  non-finite={nan_count}/{}", audio.len());
        }
    }
}

/// safetensors's int-tensor reader is not exposed by SafeTensorsModel; parse
/// the file's header + data inline. tokens are stored as I32 in
/// reference_codec.safetensors.
fn load_tokens_i32_as_usize(path: &str, tensor_name: &str) -> Vec<usize> {
    let bytes = std::fs::read(path).unwrap();
    let st = safetensors::SafeTensors::deserialize(&bytes).unwrap();
    let view = st.tensor(tensor_name).unwrap();
    assert_eq!(view.dtype(), safetensors::Dtype::I32);
    view.data()
        .chunks_exact(4)
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as usize)
        .collect()
}

fn write_wav_pcm16(path: &str, samples_interleaved: &[f32], sample_rate: u32) -> std::io::Result<()> {
    let n_channels: u16 = 2;
    let bits_per_sample: u16 = 16;
    let byte_rate = sample_rate * n_channels as u32 * (bits_per_sample / 8) as u32;
    let block_align = n_channels * (bits_per_sample / 8);
    let data_bytes = (samples_interleaved.len() as u32) * 2;
    let file_size = 36 + data_bytes;

    let f = File::create(path)?;
    let mut w = BufWriter::new(f);
    w.write_all(b"RIFF")?;
    w.write_all(&file_size.to_le_bytes())?;
    w.write_all(b"WAVE")?;
    w.write_all(b"fmt ")?;
    w.write_all(&16u32.to_le_bytes())?;  // fmt chunk size
    w.write_all(&1u16.to_le_bytes())?;    // PCM format
    w.write_all(&n_channels.to_le_bytes())?;
    w.write_all(&sample_rate.to_le_bytes())?;
    w.write_all(&byte_rate.to_le_bytes())?;
    w.write_all(&block_align.to_le_bytes())?;
    w.write_all(&bits_per_sample.to_le_bytes())?;
    w.write_all(b"data")?;
    w.write_all(&data_bytes.to_le_bytes())?;

    for &s in samples_interleaved {
        let clamped = s.clamp(-1.0, 1.0);
        let i16_val = (clamped * 32767.0) as i16;
        w.write_all(&i16_val.to_le_bytes())?;
    }
    Ok(())
}
