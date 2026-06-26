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
//! The full encoder is implemented below as a host (CPU) function: STFT →
//! strided conv-residual stack (`ratios = ((1,2),(1,2),(1,3),(1,2),(1,2),(2,2),
//! (2,1))` — ÷4 in time, ÷96 in freq: 200→50 frames, 480→5 bins) → gated
//! bottleneck → `embed [50,256]`, then RVQ quantize → tokens. The architecture,
//! per-block strides/paddings, and the STFT frame alignment were reverse-
//! engineered from the TF SavedModel graph and verified bit-exact against it
//! (`embed` rel ~1e-4; RVQ logic reproduces the TF tokens exactly on the TF
//! embedding, and the first 4 RVQ levels — the ones the LLM consumes — match TF
//! exactly end-to-end). See `tools/magenta_rt/encoder_reference.py` (NumPy
//! reference) and `tests/spectrostream_encoder_vs_tf.rs` (real-weight gate).
//!
//! Frame alignment (resolved): the encoder STFT uses `tf.signal.stft` with
//! `pad_end=True` and **no** left padding — frame `f` covers samples
//! `[f*step, f*step+frame_length)` with the final frame zero-padded at the end.
//! This is the `pad_left = 0` default in [`StftConfig`].

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
    /// Left zero-padding (samples) before frame 0. The real SpectroStream encoder
    /// uses **0** (`tf.signal.stft` with `pad_end=True`: frame `f` = samples
    /// `[f*step, f*step+frame_length)`, last frame zero-padded at the end). Set to
    /// `frame_length - frame_step` for a centered/causal STFT.
    pub pad_left: usize,
}

impl Default for StftConfig {
    fn default() -> Self {
        Self {
            frame_length: 960,
            frame_step: 480,
            fft_length: 960,
            num_audio_channels: 2,
            keep_dc: true,
            pad_left: 0,
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
    let pad_left = cfg.pad_left;
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

// ===================== Encoder conv stack (host / CPU) =====================
//
// A faithful CPU port of the SpectroStream encoder, verified bit-exact against
// the TF SavedModel (embed rel 1.6e-4, RVQ tokens 100% — see
// `tools/magenta_rt/encoder_reference.py` and `tests/spectrostream_encoder_vs_tf.rs`).
// The encoder runs once per ~10 s audio context (not the generation hot path),
// so it lives on the CPU; a GPU graph port would need new ops (avg-pool,
// asymmetric strided/causal conv) and is a follow-up.

use crate::data::safetensors::SafeTensorsModel;

/// NHWC tensor `[b, h, w, c]` as a flat row-major buffer.
struct T4 {
    d: Vec<f32>,
    b: usize,
    h: usize,
    w: usize,
    c: usize,
}

impl T4 {
    fn zeros(b: usize, h: usize, w: usize, c: usize) -> Self {
        T4 {
            d: vec![0.0; b * h * w * c],
            b,
            h,
            w,
            c,
        }
    }
    #[inline]
    fn at(&self, b: usize, h: usize, w: usize, c: usize) -> f32 {
        self.d[((b * self.h + h) * self.w + w) * self.c + c]
    }
    fn meta(&self) -> (usize, usize, usize, usize) {
        (self.b, self.h, self.w, self.c)
    }
}

fn elu_t(x: &T4) -> T4 {
    let d =
        x.d.iter()
            .map(|&v| if v > 0.0 { v } else { v.exp() - 1.0 })
            .collect();
    let (b, h, w, c) = x.meta();
    T4 { d, b, h, w, c }
}

fn add_t(a: &T4, b: &T4) -> T4 {
    let d = a.d.iter().zip(&b.d).map(|(x, y)| x + y).collect();
    let (bb, h, w, c) = a.meta();
    T4 { d, b: bb, h, w, c }
}

/// Zero-pad H/W: `(top, bottom)` on H, `(left, right)` on W.
fn pad_hw(x: &T4, top: usize, bottom: usize, left: usize, right: usize) -> T4 {
    let mut o = T4::zeros(x.b, x.h + top + bottom, x.w + left + right, x.c);
    for b in 0..x.b {
        for h in 0..x.h {
            for w in 0..x.w {
                let src = ((b * x.h + h) * x.w + w) * x.c;
                let dst = ((b * o.h + (h + top)) * o.w + (w + left)) * o.c;
                o.d[dst..dst + x.c].copy_from_slice(&x.d[src..src + x.c]);
            }
        }
    }
    o
}

/// VALID conv (NHWC), kernel `[kh,kw,ci,co]` row-major (HWIO), stride `(sh,sw)`.
fn conv2d_valid(
    x: &T4,
    k: &[f32],
    kh: usize,
    kw: usize,
    co: usize,
    b: &[f32],
    sh: usize,
    sw: usize,
) -> T4 {
    let ci = x.c;
    let ho = (x.h - kh) / sh + 1;
    let wo = (x.w - kw) / sw + 1;
    let mut o = T4::zeros(x.b, ho, wo, co);
    for bi in 0..x.b {
        for oh in 0..ho {
            for ow in 0..wo {
                let dst = ((bi * ho + oh) * wo + ow) * co;
                for (oc, ob) in b.iter().enumerate().take(co) {
                    let mut acc = *ob;
                    for i in 0..kh {
                        for j in 0..kw {
                            let xbase = ((bi * x.h + oh * sh + i) * x.w + ow * sw + j) * ci;
                            let kbase = (i * kw + j) * ci * co + oc;
                            for c in 0..ci {
                                acc += x.d[xbase + c] * k[kbase + c * co];
                            }
                        }
                    }
                    o.d[dst + oc] = acc;
                }
            }
        }
    }
    o
}

/// AvgPool window=stride=`(sh,sw)`, VALID.
fn avgpool(x: &T4, sh: usize, sw: usize) -> T4 {
    let ho = x.h / sh;
    let wo = x.w / sw;
    let mut o = T4::zeros(x.b, ho, wo, x.c);
    let inv = 1.0 / (sh * sw) as f32;
    for bi in 0..x.b {
        for oh in 0..ho {
            for ow in 0..wo {
                let dst = ((bi * ho + oh) * wo + ow) * x.c;
                for i in 0..sh {
                    for j in 0..sw {
                        let sbase = ((bi * x.h + oh * sh + i) * x.w + ow * sw + j) * x.c;
                        for c in 0..x.c {
                            o.d[dst + c] += x.d[sbase + c] * inv;
                        }
                    }
                }
            }
        }
    }
    o
}

struct EBlock {
    name: &'static str,
    dc: &'static str,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    wpad: (usize, usize),
    has_sc: bool,
}

const EBLOCKS: [EBlock; 7] = [
    EBlock {
        name: "encoder_0",
        dc: "conv2d_3x4_a",
        kh: 3,
        kw: 4,
        sh: 1,
        sw: 2,
        wpad: (1, 1),
        has_sc: true,
    },
    EBlock {
        name: "encoder_1",
        dc: "conv2d_3x4_a",
        kh: 3,
        kw: 4,
        sh: 1,
        sw: 2,
        wpad: (1, 1),
        has_sc: false,
    },
    EBlock {
        name: "encoder_2",
        dc: "conv2d_3x6_a",
        kh: 3,
        kw: 6,
        sh: 1,
        sw: 3,
        wpad: (1, 2),
        has_sc: true,
    },
    EBlock {
        name: "encoder_3",
        dc: "conv2d_3x4_a",
        kh: 3,
        kw: 4,
        sh: 1,
        sw: 2,
        wpad: (1, 1),
        has_sc: false,
    },
    EBlock {
        name: "encoder_4",
        dc: "conv2d_3x4_a",
        kh: 3,
        kw: 4,
        sh: 1,
        sw: 2,
        wpad: (1, 1),
        has_sc: false,
    },
    EBlock {
        name: "encoder_5",
        dc: "conv2d_4x4_a",
        kh: 4,
        kw: 4,
        sh: 2,
        sw: 2,
        wpad: (1, 1),
        has_sc: true,
    },
    EBlock {
        name: "encoder_6",
        dc: "conv2d_4x3_a",
        kh: 4,
        kw: 3,
        sh: 2,
        sw: 1,
        wpad: (1, 1),
        has_sc: true,
    },
];

fn wkey(m: &SafeTensorsModel, name: &str) -> (Vec<f32>, Vec<f32>) {
    let k = m
        .tensor_f32_auto(&format!("{name}.weight_norm.rescaled.kernel"))
        .unwrap_or_else(|e| panic!("{name}.kernel: {e}"));
    let b = m
        .tensor_f32_auto(&format!("{name}.weight_norm.bias"))
        .unwrap_or_else(|e| panic!("{name}.bias: {e}"));
    (k, b)
}

/// Causal 3×3 conv (H pad `[2,0]`, W pad `[1,1]`, VALID).
fn causal_3x3(x: &T4, k: &[f32], co: usize, b: &[f32]) -> T4 {
    let p = pad_hw(x, 2, 0, 1, 1);
    conv2d_valid(&p, k, 3, 3, co, b, 1, 1)
}

/// SpectroStream encoder forward: interleaved stereo `audio` → continuous
/// embedding `[num_frames * 256]`. Verified bit-exact against the TF encoder.
pub fn encode(audio: &[f32], model: &SafeTensorsModel) -> Vec<f32> {
    let cfg = StftConfig::default();
    let feats = stft_features(audio, &cfg);
    let t = feats.num_frames;
    // STFT [1,T,480,4] → fold 4ch → batch×2; channels [L_re,L_im,R_re,R_im].
    let mut x = T4::zeros(2, t, 480, 2);
    for f in 0..t {
        for w in 0..480 {
            let src = (f * 480 + w) * 4;
            for sb in 0..2 {
                for ch in 0..2 {
                    let dst = ((sb * t + f) * 480 + w) * 2 + ch;
                    x.d[dst] = feats.data[src + sb * 2 + ch];
                }
            }
        }
    }
    let (k, b) = wkey(model, "encoder.base_conv_first");
    let p = pad_hw(&x, 6, 0, 3, 3);
    x = conv2d_valid(&p, &k, 7, 7, 32, &b, 1, 1);

    for blk in EBLOCKS.iter() {
        if blk.name == "encoder_6" {
            // batch→channel fold: [2,T,W,C] → [1,T,W,2C].
            let mut f = T4::zeros(1, x.h, x.w, x.c * 2);
            for h in 0..x.h {
                for w in 0..x.w {
                    for sb in 0..2 {
                        for c in 0..x.c {
                            let dst = (h * f.w + w) * f.c + sb * x.c + c;
                            f.d[dst] = x.at(sb, h, w, c);
                        }
                    }
                }
            }
            x = f;
        }
        let (rb, rh, rw, rc) = x.meta();
        let res = T4 {
            d: x.d.clone(),
            b: rb,
            h: rh,
            w: rw,
            c: rc,
        };
        let h = elu_t(&x);
        let (k3, b3) = wkey(model, &format!("encoder.{}.conv2d_3x3", blk.name));
        let h = causal_3x3(&h, &k3, b3.len(), &b3);
        let h = elu_t(&h);
        let (kd, bd) = wkey(model, &format!("encoder.{}.{}", blk.name, blk.dc));
        let hp = pad_hw(&h, 2, 0, blk.wpad.0, blk.wpad.1);
        let main = conv2d_valid(&hp, &kd, blk.kh, blk.kw, bd.len(), &bd, blk.sh, blk.sw);
        let mut sc = avgpool(&res, blk.sh, blk.sw);
        if blk.has_sc {
            let (ks, bs) = wkey(model, &format!("encoder.{}.shortcut.conv1x1", blk.name));
            sc = conv2d_valid(&sc, &ks, 1, 1, bs.len(), &bs, 1, 1);
        }
        x = add_t(&main, &sc);
    }

    // bottleneck: residual (conv2d_3x3 then conv2d_3x3_a), then gated conv1x1_last.
    let (rb, rh, rw, rc) = x.meta();
    let res = T4 {
        d: x.d.clone(),
        b: rb,
        h: rh,
        w: rw,
        c: rc,
    };
    let h = elu_t(&x);
    let (k1, b1) = wkey(model, "encoder.bottleneck.conv2d_3x3");
    let h = causal_3x3(&h, &k1, b1.len(), &b1);
    let h = elu_t(&h);
    let (k2, b2) = wkey(model, "encoder.bottleneck.conv2d_3x3_a");
    let h = causal_3x3(&h, &k2, b2.len(), &b2);
    x = add_t(&res, &h);
    let flat = T4 {
        d: x.d.clone(),
        b: 1,
        h: x.h,
        w: 1,
        c: x.w * x.c,
    };
    // gated conv1x1_last: conv1x1_b(elu(conv1x1_a(elu(x)))) + conv1x1(x).
    let (ka, ba) = wkey(model, "encoder.bottleneck.conv1x1_last.conv1x1_a");
    let ma = conv2d_valid(&elu_t(&flat), &ka, 1, 1, ba.len(), &ba, 1, 1);
    let (kb, bb) = wkey(model, "encoder.bottleneck.conv1x1_last.conv1x1_b");
    let mb = conv2d_valid(&elu_t(&ma), &kb, 1, 1, bb.len(), &bb, 1, 1);
    let (kp, bp) = wkey(model, "encoder.bottleneck.conv1x1_last.conv1x1");
    let par = conv2d_valid(&flat, &kp, 1, 1, bp.len(), &bp, 1, 1);
    add_t(&mb, &par).d // [1, T, 1, 256] → flat [T*256]
}

/// RVQ-encode an embedding grid `[num_frames, dim]` → `[num_frames, depth]`
/// tokens (per frame/level: nearest codebook centroid, emit index, subtract).
/// `codebooks` is `[depth, codebook_size, dim]`. Verified 100% vs the TF quantizer.
pub fn rvq_encode(
    embed: &[f32],
    num_frames: usize,
    dim: usize,
    codebooks: &[f32],
    depth: usize,
    codebook_size: usize,
) -> Vec<u32> {
    assert_eq!(embed.len(), num_frames * dim);
    assert_eq!(codebooks.len(), depth * codebook_size * dim);
    let mut tokens = vec![0u32; num_frames * depth];
    for f in 0..num_frames {
        let mut res: Vec<f32> = embed[f * dim..(f + 1) * dim].to_vec();
        for level in 0..depth {
            let cb = &codebooks[level * codebook_size * dim..(level + 1) * codebook_size * dim];
            let mut best = 0usize;
            let mut best_d = f32::INFINITY;
            for e in 0..codebook_size {
                let cen = &cb[e * dim..(e + 1) * dim];
                let mut d = 0.0_f32;
                for (rk, ck) in res.iter().zip(cen) {
                    let diff = rk - ck;
                    d += diff * diff;
                }
                if d < best_d {
                    best_d = d;
                    best = e;
                }
            }
            tokens[f * depth + level] = best as u32;
            let cen = &cb[best * dim..(best + 1) * dim];
            for (rk, ck) in res.iter_mut().zip(cen) {
                *rk -= ck;
            }
        }
    }
    tokens
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
