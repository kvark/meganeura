//! Real-weight gate for the SpectroStream encoder (`spectrostream_encoder::{encode,
//! rvq_encode}`): audio → continuous embedding → RVQ tokens, vs the TF encoder +
//! quantizer on the reference 2-second clip.
//!
//! Reference (`encoder_ref.safetensors`: `audio` [96000,2], `embed` [50,256],
//! `tokens` [50,64]) is produced by `tools/magenta_rt/encoder_reference.py` /
//! `dump_codec_local.py`; not in CI, so `#[ignore]`d.
//!
//! Verification splits the two concerns that an all-levels token-match conflates:
//!   * embedding accuracy — `embed` rel < 1e-3 (we get ~1e-4);
//!   * RVQ logic — `rvq_encode(ref_embed)` reproduces the TF tokens *exactly*.
//! End-to-end (our embed → tokens), the first 4 RVQ levels (the only ones the LLM
//! consumes) match TF exactly; deeper levels are embedding-noise-limited and
//! reported for information only.

use meganeura::data::safetensors::SafeTensorsModel;
use meganeura::models::magenta_rt::spectrostream_encoder::{encode, rvq_encode};
use std::path::Path;

const WEIGHTS: &str = "magenta_rt_codec_dump/weights_spectrostream.safetensors";
const REF: &str = "magenta_rt_codec_dump/encoder_ref.safetensors";

#[test]
#[ignore = "requires magenta_rt_codec_dump/{weights_spectrostream,encoder_ref}.safetensors"]
fn encoder_matches_tf() {
    if !Path::new(WEIGHTS).exists() || !Path::new(REF).exists() {
        eprintln!("skip: codec dump not found");
        return;
    }
    let model = SafeTensorsModel::load(WEIGHTS.into()).unwrap();
    let refm = SafeTensorsModel::load(REF.into()).unwrap();
    let audio = refm.tensor_f32_auto("audio").unwrap(); // [96000,2] interleaved
    let ref_embed = refm.tensor_f32_auto("embed").unwrap(); // [50,256]
    let ref_tokens = refm.tensor_f32_auto("tokens").unwrap(); // [50,64] (as f32)

    let frames = 50usize;
    let dim = 256usize;

    // --- embedding ---
    let embed = encode(&audio, &model);
    assert_eq!(embed.len(), frames * dim, "embed size");
    let mut max_abs = 0.0_f32;
    let mut ref_absmax = 0.0_f32;
    for (a, b) in embed.iter().zip(&ref_embed) {
        max_abs = max_abs.max((a - b).abs());
        ref_absmax = ref_absmax.max(b.abs());
    }
    let rel = max_abs / ref_absmax;
    eprintln!("encoder embed vs TF: max abs diff {max_abs:.3e}, rel {rel:.3e}");
    assert!(rel < 1e-3, "embed rel error {rel} too high");

    // --- RVQ tokens ---
    let codebooks = model.tensor_f32_auto("quantizer.rvq_codebooks").unwrap(); // [64,1024,256]
    let depth = 64usize;
    let cb_size = 1024usize;

    // (a) RVQ logic, isolated from embedding noise: encode the *reference* TF
    // embedding and require an exact reproduction of the TF tokens. This is the
    // true test of the quantizer math (nearest-centroid + residual subtraction).
    let tokens_ref = rvq_encode(&ref_embed, frames, dim, &codebooks, depth, cb_size);
    let mut matches_ref = 0usize;
    for (t, r) in tokens_ref.iter().zip(&ref_tokens) {
        if *t == *r as u32 {
            matches_ref += 1;
        }
    }
    let pct_ref = 100.0 * matches_ref as f32 / (frames * depth) as f32;
    eprintln!("RVQ(ref embed) vs TF: {matches_ref}/{} = {pct_ref:.2}%", frames * depth);
    assert!(
        pct_ref >= 99.9,
        "RVQ logic must reproduce TF tokens exactly from the TF embedding, got {pct_ref}%"
    );

    // (b) End-to-end on our own embedding. A ~1e-4 embedding error flips a few
    // *deep* RVQ choices (residuals there are at noise level), so all-64-levels
    // agreement is embedding-limited (~90-96%) and reported for information only.
    // What the LLM actually consumes is the first 4 levels — those operate on the
    // largest-magnitude residuals and must match TF exactly.
    let tokens = rvq_encode(&embed, frames, dim, &codebooks, depth, cb_size);
    let mut matches = 0usize;
    for (t, r) in tokens.iter().zip(&ref_tokens) {
        if *t == *r as u32 {
            matches += 1;
        }
    }
    let pct = 100.0 * matches as f32 / (frames * depth) as f32;
    eprintln!(
        "RVQ(our embed) vs TF: {matches}/{} = {pct:.1}% (all levels; embedding-limited)",
        frames * depth
    );

    let mut m4 = 0usize;
    for f in 0..frames {
        for l in 0..4 {
            if tokens[f * depth + l] == ref_tokens[f * depth + l] as u32 {
                m4 += 1;
            }
        }
    }
    eprintln!("first-4-level match: {m4}/{}", frames * 4);
    assert_eq!(m4, frames * 4, "first-4 RVQ levels must match TF exactly");
}
