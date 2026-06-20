//! Correctness of the SpectroStream encoder STFT front-end
//! (`spectrostream_encoder::stft_features`) against an independent naive-DFT
//! reference: validates the framing, the periodic Hann window, the FFT, the
//! keep-DC bin selection, and the `[re, im]`-per-channel output layout.
//!
//! This is op-correctness (vs a hand-rolled DFT with the same framing), not a
//! match against the real codec's `semicausal` alignment — see the module docs.

use std::f32::consts::PI;

use meganeura::models::magenta_rt::spectrostream_encoder::{stft_features, StftConfig};

fn hann_periodic(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / n as f32).cos())
        .collect()
}

/// Naive reference: same causal framing/window, then a direct DFT per bin.
fn ref_stft(audio: &[f32], cfg: &StftConfig) -> (Vec<f32>, usize) {
    let c = cfg.num_audio_channels;
    let num_samples = audio.len() / c;
    let num_bins = cfg.fft_length / 2;
    let pad_left = cfg.frame_length - cfg.frame_step;
    let num_frames = num_samples / cfg.frame_step;
    let out_channels = 2 * c;
    let window = hann_periodic(cfg.frame_length);
    let n_fft = cfg.fft_length;

    let mut out = vec![0.0_f32; num_frames * num_bins * out_channels];
    for ch in 0..c {
        for f in 0..num_frames {
            // Windowed (zero-padded to fft_length) frame.
            let mut frame = vec![0.0_f32; n_fft];
            for (n, fr) in frame.iter_mut().enumerate().take(cfg.frame_length) {
                let idx = (f * cfg.frame_step + n) as isize - pad_left as isize;
                if idx >= 0 && (idx as usize) < num_samples {
                    *fr = audio[idx as usize * c + ch] * window[n];
                }
            }
            for bin in 0..num_bins {
                let k = if cfg.keep_dc { bin } else { bin + 1 };
                let mut re = 0.0_f64;
                let mut im = 0.0_f64;
                for (n, &fr) in frame.iter().enumerate() {
                    let ang = -2.0 * std::f64::consts::PI * k as f64 * n as f64 / n_fft as f64;
                    re += fr as f64 * ang.cos();
                    im += fr as f64 * ang.sin();
                }
                let base = (f * num_bins + bin) * out_channels + ch * 2;
                out[base] = re as f32;
                out[base + 1] = im as f32;
            }
        }
    }
    (out, num_frames)
}

#[test]
fn stft_matches_naive_dft() {
    // Small config (so the O(F·bins·N) reference is cheap) with the real layout.
    let cfg = StftConfig {
        frame_length: 64,
        frame_step: 32,
        fft_length: 64,
        num_audio_channels: 2,
        keep_dc: true,
    };
    // 8 frames of stereo audio: num_samples = 8 * frame_step = 256.
    let num_samples = 8 * cfg.frame_step;
    let mut seed = 0x1234_5678u64;
    let mut rng = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        ((seed >> 40) as f32 / (1u32 << 24) as f32 - 0.5) * 2.0
    };
    let audio: Vec<f32> = (0..num_samples * cfg.num_audio_channels)
        .map(|_| rng())
        .collect();

    let got = stft_features(&audio, &cfg);
    let (want, frames) = ref_stft(&audio, &cfg);
    assert_eq!(got.num_frames, frames);
    assert_eq!(got.data.len(), want.len());

    let mut max_abs = 0.0_f32;
    for (a, b) in got.data.iter().zip(want.iter()) {
        max_abs = max_abs.max((a - b).abs());
    }
    eprintln!("STFT vs naive DFT max abs diff: {max_abs:.3e}");
    assert!(
        max_abs <= 1e-3,
        "STFT differs from DFT reference by {max_abs}"
    );
}

#[test]
fn keep_dc_false_drops_dc_bin() {
    // With keep_dc=false the first stored bin is DFT bin 1 (DC dropped).
    let cfg = StftConfig {
        frame_length: 32,
        frame_step: 16,
        fft_length: 32,
        num_audio_channels: 1,
        keep_dc: false,
    };
    let audio: Vec<f32> = (0..16 * 4).map(|i| (i as f32 * 0.1).sin()).collect();
    let got = stft_features(&audio, &cfg);
    let (want, _) = ref_stft(&audio, &cfg);
    let mut max_abs = 0.0_f32;
    for (a, b) in got.data.iter().zip(want.iter()) {
        max_abs = max_abs.max((a - b).abs());
    }
    assert!(max_abs <= 1e-3, "keep_dc=false path differs by {max_abs}");
}
