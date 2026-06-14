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
    build_decoder_graph, load_decoder_weights, SpectroStreamConfig,
};
use rustfft::FftPlanner;
use rustfft::num_complex::Complex32;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::Path;

const WEIGHTS: &str = "magenta_rt_codec_dump/weights_spectrostream.safetensors";
const REF_CODEC: &str = "magenta_rt_codec_dump/reference_codec.safetensors";
const OUT_WAV: &str = "/tmp/magenta_rt_demo.wav";

const NUM_FRAMES: u32 = 50;
const FRAME_LENGTH: usize = 960;
const FRAME_STEP: usize = 480;
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

    let preprocessed_nchw = host_preprocess(&weights, &tokens_u);
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
    let mut body_out_nchw = s.read_output(1 * 4 * 200 * 480);
    println!("body inference: {elapsed:.2?}");
    // Sanitize: replace NaN with 0 (known RADV non-determinism bug at GPU
    // level — separate from meganeura). Without this every NaN body sample
    // contaminates ~960 audio samples through the IRFFT.
    let nans = body_out_nchw.iter().filter(|v| !v.is_finite()).count();
    if nans > 0 {
        println!("WARNING: {nans}/{} body samples are NaN/Inf — clamping to 0 \
                  for playable WAV (RADV bug, not meganeura)",
                 body_out_nchw.len());
        for v in body_out_nchw.iter_mut() {
            if !v.is_finite() { *v = 0.0; }
        }
    }

    // 3. Host-side iSTFT: NCHW [1, 4, 200, 480] → audio [96000, 2].
    let body_nhwc = nchw_to_nhwc(&body_out_nchw, 1, 4, 200, 480);
    let audio = istft_body_to_audio(&body_nhwc, 200, 480);

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

fn host_preprocess(weights: &SafeTensorsModel, tokens: &[usize]) -> Vec<f32> {
    let n_codebooks = 64usize;
    let codebook_size = 1024usize;
    let embed_dim = 256usize;
    let depth = 64usize; // use all 64 codebooks (codec round-trip)
    let n_frames = tokens.len() / n_codebooks;
    assert_eq!(n_frames, NUM_FRAMES as usize);

    let codebooks = weights.tensor_f32_auto("quantizer.rvq_codebooks").unwrap();
    assert_eq!(codebooks.len(), n_codebooks * codebook_size * embed_dim);

    // embed[i, d] = sum_k codebooks[k, tokens[i, k], d]
    let mut embed = vec![0.0_f32; n_frames * embed_dim];
    for i in 0..n_frames {
        for k in 0..depth {
            let tok = tokens[i * n_codebooks + k];
            let cb_offset = (k * codebook_size + tok) * embed_dim;
            for d in 0..embed_dim {
                embed[i * embed_dim + d] += codebooks[cb_offset + d];
            }
        }
    }

    // Temporal pad [0, 1] on T axis: pad 1 zero frame at the end → [51, 256].
    let t_padded = n_frames + 1;
    let mut embed_padded = vec![0.0_f32; t_padded * embed_dim];
    embed_padded[..n_frames * embed_dim].copy_from_slice(&embed);

    // input_layer parallel conv1x1 paths:
    //   main_a = conv1x1_a(x)           [T, 2560]
    //   main_b = conv1x1_b(ELU(main_a)) [T, 2560]
    //   parallel = conv1x1(x)            [T, 2560]
    //   out = main_b + parallel
    let k_a = weights.tensor_f32_auto("decoder.input_layer.conv1x1_first.conv1x1_a.weight_norm.rescaled.kernel").unwrap();
    let b_a = weights.tensor_f32_auto("decoder.input_layer.conv1x1_first.conv1x1_a.weight_norm.bias").unwrap();
    let k_b = weights.tensor_f32_auto("decoder.input_layer.conv1x1_first.conv1x1_b.weight_norm.rescaled.kernel").unwrap();
    let b_b = weights.tensor_f32_auto("decoder.input_layer.conv1x1_first.conv1x1_b.weight_norm.bias").unwrap();
    let k_p = weights.tensor_f32_auto("decoder.input_layer.conv1x1_first.conv1x1.weight_norm.rescaled.kernel").unwrap();
    let b_p = weights.tensor_f32_auto("decoder.input_layer.conv1x1_first.conv1x1.weight_norm.bias").unwrap();

    let out_2560 = 2560usize;
    let main_a = matmul_add(&embed_padded, &k_a, &b_a, t_padded, embed_dim, out_2560);
    let main_a_elu = main_a.iter().map(|&v| if v > 0.0 { v } else { v.exp() - 1.0 }).collect::<Vec<f32>>();
    let main_b = matmul_add(&main_a_elu, &k_b, &b_b, t_padded, out_2560, out_2560);
    let parallel = matmul_add(&embed_padded, &k_p, &b_p, t_padded, embed_dim, out_2560);

    // Sum + reshape to NHWC [1, T, 5, 512], then transpose to NCHW [1, 512, T, 5].
    let init_freq = 5usize;
    let init_channels = 512usize;
    let mut nchw = vec![0.0_f32; init_channels * t_padded * init_freq];
    for t in 0..t_padded {
        for f in 0..init_freq {
            for c in 0..init_channels {
                let val = main_b[t * out_2560 + f * init_channels + c]
                        + parallel[t * out_2560 + f * init_channels + c];
                // NCHW index: ((b=0)*C + c)*H + t)*W + f.
                let dst = (c * t_padded + t) * init_freq + f;
                nchw[dst] = val;
            }
        }
    }
    nchw
}

/// out[t, n] = sum_d in[t, d] * k[d, n] + b[n].
fn matmul_add(input: &[f32], kernel: &[f32], bias: &[f32],
              n_frames: usize, in_dim: usize, out_dim: usize) -> Vec<f32> {
    assert_eq!(kernel.len(), in_dim * out_dim);
    assert_eq!(bias.len(), out_dim);
    let mut out = vec![0.0_f32; n_frames * out_dim];
    for t in 0..n_frames {
        let row = &input[t * in_dim..(t + 1) * in_dim];
        for n in 0..out_dim {
            let mut acc = bias[n];
            for d in 0..in_dim {
                acc += row[d] * kernel[d * out_dim + n];
            }
            out[t * out_dim + n] = acc;
        }
    }
    out
}

fn nchw_to_nhwc(data: &[f32], n: usize, c: usize, h: usize, w: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; n * h * w * c];
    for bi in 0..n {
        for ci in 0..c {
            for hi in 0..h {
                for wi in 0..w {
                    let src = ((bi * c + ci) * h + hi) * w + wi;
                    let dst = ((bi * h + hi) * w + wi) * c + ci;
                    out[dst] = data[src];
                }
            }
        }
    }
    out
}

/// Compute TF's inverse_stft_window_fn(frame_step, hann_window): the Hann
/// window divided by sum-of-squared-overlapping-shifts.
fn inverse_stft_window(frame_length: usize, frame_step: usize) -> Vec<f32> {
    let n = frame_length;
    // tf.signal.hann_window(N, periodic=True): w[t] = 0.5 - 0.5*cos(2π t / N).
    let mut forward = vec![0.0_f32; n];
    for t in 0..n {
        forward[t] = 0.5 - 0.5 * ((2.0 * std::f32::consts::PI * t as f32) / n as f32).cos();
    }
    let mut denom = vec![0.0_f32; n];
    let overlap = n / frame_step;
    for shift in 0..overlap {
        let offset = shift * frame_step;
        for t in 0..n {
            let src = (t + n - offset % n) % n;
            denom[t] += forward[src] * forward[src];
        }
    }
    let mut inv = vec![0.0_f32; n];
    for t in 0..n {
        inv[t] = forward[t] / denom[t].max(1e-10);
    }
    inv
}

/// body_nhwc: [1, T, 480, 4] → audio [96000, 2] interleaved (L, R, L, R, ...).
fn istft_body_to_audio(body_nhwc: &[f32], n_frames: usize, n_freq: usize) -> Vec<f32> {
    assert_eq!(body_nhwc.len(), 1 * n_frames * n_freq * 4);
    let n_freq_with_nyquist = n_freq + 1;  // 481

    // For each stereo channel, build the complex STFT [T, 481] and iSTFT.
    let mut planner = FftPlanner::<f32>::new();
    let ifft = planner.plan_fft_inverse(FRAME_LENGTH);
    let inv_window = inverse_stft_window(FRAME_LENGTH, FRAME_STEP);

    let out_len = (n_frames - 1) * FRAME_STEP + FRAME_LENGTH;
    let mut audio_l = vec![0.0_f32; out_len];
    let mut audio_r = vec![0.0_f32; out_len];

    // Helper: do iSTFT for one stereo channel.
    let mut process = |re_offset: usize, im_offset: usize, audio: &mut Vec<f32>| {
        for f in 0..n_frames {
            let mut full = vec![Complex32::new(0.0, 0.0); FRAME_LENGTH];
            // Bins 0..480 from body.
            for bin in 0..n_freq {
                let idx = ((0 * n_frames + f) * n_freq + bin) * 4;
                full[bin] = Complex32::new(body_nhwc[idx + re_offset], body_nhwc[idx + im_offset]);
            }
            // bin 480 (Nyquist) = 0 (already from init).
            // Reflect for negative frequencies: bin k → bin N-k = conj(bin k).
            for bin in 1..n_freq_with_nyquist - 1 {
                full[FRAME_LENGTH - bin] = full[bin].conj();
            }
            // IFFT — note rustfft does NOT normalize, so divide by N.
            ifft.process(&mut full);
            for t in 0..FRAME_LENGTH {
                let time_val = full[t].re / FRAME_LENGTH as f32;
                audio[f * FRAME_STEP + t] += time_val * inv_window[t];
            }
        }
    };

    process(0, 1, &mut audio_l);
    process(2, 3, &mut audio_r);

    // Interleave L, R to 96000-sample stereo.
    let target_samples = 96000;
    let mut interleaved = Vec::with_capacity(target_samples * 2);
    for i in 0..target_samples {
        interleaved.push(if i < audio_l.len() { audio_l[i] } else { 0.0 });
        interleaved.push(if i < audio_r.len() { audio_r[i] } else { 0.0 });
    }
    interleaved
}

/// Write a 48 kHz stereo PCM-16 WAV file.
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
