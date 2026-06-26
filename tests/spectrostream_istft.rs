//! Verifies the Rust iSTFT (`spectrostream::istft_to_audio`) reproduces
//! `tf.signal.inverse_stft` as the SpectroStream decoder applies it: a real
//! decoder `body_out [200, 480, 4]` → stereo audio `[96000, 2]`.
//!
//! The reference (`istft_ref.safetensors`: `body_out` + `tf_audio`) is produced
//! by `tools/magenta_rt/istft_derive.py` and isn't in CI (needs TF + the codec
//! SavedModel), so this is `#[ignore]`d. The mapping was confirmed bit-exact
//! (rms-ratio 0.0) at derivation time.

use meganeura::data::safetensors::SafeTensorsModel;
use meganeura::models::magenta_rt::spectrostream::{istft_to_audio, IstftConfig};
use std::path::Path;

const REF: &str = "magenta_rt_codec_dump/istft_ref.safetensors";

#[test]
#[ignore = "requires magenta_rt_codec_dump/istft_ref.safetensors (istft_derive.py)"]
fn istft_matches_tf_inverse_stft() {
    if !Path::new(REF).exists() {
        eprintln!("skip: {REF} not found");
        return;
    }
    let m = SafeTensorsModel::load(REF.into()).unwrap();
    let body = m.tensor_f32_auto("body_out").unwrap(); // [200, 480, 4] NHWC
    let tf_audio = m.tensor_f32_auto("tf_audio").unwrap(); // [96000, 2] interleaved

    let frames = 200usize;
    let cfg = IstftConfig::default();
    let audio = istft_to_audio(&body, frames, &cfg);

    // Compare the first 96000 samples/channel against TF (the codec trims to the
    // 2-second chunk; TF's inverse_stft tail beyond that is dropped).
    let n = tf_audio.len().min(audio.len());
    let mut sq = 0.0_f64;
    let mut ref_sq = 0.0_f64;
    for i in 0..n {
        let d = (audio[i] - tf_audio[i]) as f64;
        sq += d * d;
        ref_sq += (tf_audio[i] as f64).powi(2);
    }
    let rms_ratio = (sq / n as f64).sqrt() / (ref_sq / n as f64).sqrt().max(1e-12);
    eprintln!("iSTFT vs TF rms-ratio: {rms_ratio:.6e} (n={n})");
    assert!(rms_ratio < 1e-4, "iSTFT rms-ratio {rms_ratio} too high");
}
