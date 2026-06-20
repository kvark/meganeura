//! SpectroStream **encoder** front-end (audio → spectrogram features).
//!
//! The SpectroStream codec operates on an STFT spectrogram, not raw audio. The
//! encoder is: `audio → STFT → strided conv-residual stack → RVQ quantize →
//! tokens`. This module implements the **STFT front-end** — the audio→feature
//! transform that mirrors the decoder's host-side iSTFT and produces features in
//! the same `[frames, num_bins, 2*channels]` layout the decoder body uses.
//!
//! Parameters are the real 48 kHz stereo codec (`ssv2_48k_stereo`), confirmed
//! from `magenta_rt/mlx/spectrostream/modeling.py`: `frame_length=960`,
//! `frame_step=480`, `fft_length=960` (Hann window), `keep_dc=true` (keep the DC
//! bin, drop Nyquist → 480 bins), `num_channels=4` (stereo × complex). The last
//! feature axis is `[L_real, L_imag, R_real, R_imag]` per `(frame, bin)` — the
//! complex STFT bit-cast to real channels, matching `istft_test.py`.
//!
//! Not yet built (blocked on the encoder weight manifest — channel counts per
//! block): the strided conv-residual stack (`ratios = ((1,2),(1,2),(1,3),(1,2),
//! (1,2),(2,2),(2,1))` — ÷4 in time, ÷96 in freq: 200→50 frames, 480→5 bins) and
//! the RVQ quantizer. The STFT here is the verifiable, weight-free entry point.
//!
//! **Unconfirmed detail:** the codec configures the STFT `time_padding` as
//! `semicausal`; the exact frame alignment isn't recoverable from the public
//! source (sequence-layers isn't vendored). This uses causal-style left padding
//! (`frame_length - frame_step`), which yields the right frame count
//! (`num_samples / frame_step`); the alignment needs a real-audio round-trip to
//! confirm.

use std::f32::consts::PI;

use rustfft::num_complex::Complex;
use rustfft::FftPlanner;

/// STFT front-end parameters for the SpectroStream encoder.
#[derive(Clone, Debug)]
pub struct StftConfig {
    /// Window / frame length in samples (960).
    pub frame_length: usize,
    /// Hop between frames in samples (480).
    pub frame_step: usize,
    /// FFT size (960).
    pub fft_length: usize,
    /// Number of audio channels (2 = stereo).
    pub num_audio_channels: usize,
    /// Keep the DC bin and drop Nyquist (`true`, the real codec) vs. the reverse.
    pub keep_dc: bool,
}

impl Default for StftConfig {
    fn default() -> Self {
        Self {
            frame_length: 960,
            frame_step: 480,
            fft_length: 960,
            num_audio_channels: 2,
            keep_dc: true,
        }
    }
}

/// Computed STFT features.
pub struct StftFeatures {
    /// Row-major `[num_frames, num_bins, 2 * num_audio_channels]`; the last axis
    /// is `[ch0_real, ch0_imag, ch1_real, ch1_imag, …]`.
    pub data: Vec<f32>,
    pub num_frames: usize,
    /// `fft_length / 2` (480) — one of the `fft_length/2 + 1` bins is dropped.
    pub num_bins: usize,
    /// `2 * num_audio_channels` (4).
    pub channels: usize,
}

/// Periodic Hann window (`tf.signal.hann_window` default, `periodic=True`):
/// `w[n] = 0.5 - 0.5·cos(2π n / N)`.
fn hann_periodic(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| 0.5 - 0.5 * (2.0 * PI * i as f32 / n as f32).cos())
        .collect()
}

/// Compute the SpectroStream encoder STFT features from interleaved audio.
///
/// `audio` is `num_samples * num_audio_channels` interleaved samples
/// (`[L, R, L, R, …]`). Returns features in the decoder-body layout
/// `[num_frames, num_bins, 2*num_audio_channels]` with `num_frames =
/// num_samples / frame_step` (the input is left-padded by `frame_length -
/// frame_step`).
pub fn stft_features(audio: &[f32], cfg: &StftConfig) -> StftFeatures {
    let c = cfg.num_audio_channels;
    assert_eq!(
        audio.len() % c,
        0,
        "audio length must be a multiple of channels"
    );
    let num_samples = audio.len() / c;
    let num_bins = cfg.fft_length / 2; // drop one of the fft_length/2+1 bins
    let pad_left = cfg.frame_length - cfg.frame_step;
    let num_frames = num_samples / cfg.frame_step;
    let out_channels = 2 * c;
    let window = hann_periodic(cfg.frame_length);

    let mut planner = FftPlanner::<f32>::new();
    let fft = planner.plan_fft_forward(cfg.fft_length);
    let mut buf = vec![Complex::new(0.0_f32, 0.0); cfg.fft_length];

    let mut out = vec![0.0_f32; num_frames * num_bins * out_channels];
    for ch in 0..c {
        for f in 0..num_frames {
            // Window the frame: padded index p = f*step + n maps to audio index
            // p - pad_left (causal left padding; zeros outside [0, num_samples)).
            for (n, slot) in buf.iter_mut().enumerate() {
                let re = if n < cfg.frame_length {
                    let idx = (f * cfg.frame_step + n) as isize - pad_left as isize;
                    if idx >= 0 && (idx as usize) < num_samples {
                        audio[idx as usize * c + ch] * window[n]
                    } else {
                        0.0
                    }
                } else {
                    0.0 // zero-pad if fft_length > frame_length
                };
                *slot = Complex::new(re, 0.0);
            }
            fft.process(&mut buf);
            for bin in 0..num_bins {
                // keep_dc: bins 0..num_bins (drop Nyquist); else 1..=num_bins (drop DC).
                let src = if cfg.keep_dc { bin } else { bin + 1 };
                let v = buf[src];
                let base = (f * num_bins + bin) * out_channels + ch * 2;
                out[base] = v.re;
                out[base + 1] = v.im;
            }
        }
    }

    StftFeatures {
        data: out,
        num_frames,
        num_bins,
        channels: out_channels,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_count_matches_codec() {
        // 2 s stereo at 48 kHz → 96000 samples/channel, hop 480 → 200 frames.
        let cfg = StftConfig::default();
        let audio = vec![0.0_f32; 96000 * 2];
        let feats = stft_features(&audio, &cfg);
        assert_eq!(feats.num_frames, 200);
        assert_eq!(feats.num_bins, 480);
        assert_eq!(feats.channels, 4);
    }
}
