//! Real-weight numeric gate for the full Depthformer decoder
//! (`build_decoder_faithful`): the first independent check of the temporal+depth
//! decoder against the faithful NumPy port of the v1 `depthformer/modules.py`
//! (`tools/magenta_rt/llm_numpy_ref.py`), both on the *same* real weights.
//!
//! Gate interface = the flat `decoder_input_tokens` `[T*Q]` the decoder receives
//! (upstream shift is identical for any consumer). Both sides use the
//! deterministic input `tok[i] = (i*53+11) % vocab` and encoder output
//! `enc[i,j] = 0.1*sin(0.7i + 0.013j)`.
//!
//! Unlike the random-weight `llm_full_decoder_correctness.rs` (zero rel-pos,
//! no PE), this exercises: the FixedEmbed absolute PE, the edge-pad/mean
//! (temporal) and edge-pad/concat (depth) input construction, the **nonzero**
//! T5 rel-pos bias (temporal 128-bucket + depth 16-bucket bucketing), the
//! cross-attention to a real encoder output, and the shared head — all at real
//! weight magnitudes.
//!
//! Gated behind env vars (the 1.3 GB checkpoint + the reference aren't in CI):
//!
//! - `MEGANEURA_LLM_WEIGHTS=<weights_llm_base.safetensors>` (from `dump_llm.py`)
//! - `MEGANEURA_DEC_REF=<dec_ref.bin>` — raw LE f32 `[T*Q, vocab]` logits, plus a
//!   sidecar `<dec_ref.bin>.enc` = `[enc_seq, embed]`, written by
//!   `llm_numpy_ref.py` (`MEGANEURA_DEC_REF_OUT=…`).
//! - `MEGANEURA_DEC_REF_FRAMES` (default 3), `MEGANEURA_DEC_REF_ENCSEQ` (default 5)
//!   — must match the reference run.

use meganeura::data::safetensors::SafeTensorsModel;
use meganeura::models::magenta_rt::llm::{build_decoder_faithful, LlmConfig};
use meganeura::models::magenta_rt::llm_weights::load_llm_weights;
use meganeura::Graph;

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(default)
}

fn read_f32_le(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
#[ignore = "requires MEGANEURA_LLM_WEIGHTS + MEGANEURA_DEC_REF + a Vulkan device"]
fn decoder_matches_numpy_reference_on_real_weights() {
    let weights_path = std::env::var("MEGANEURA_LLM_WEIGHTS").expect("set MEGANEURA_LLM_WEIGHTS");
    let ref_path = std::env::var("MEGANEURA_DEC_REF").expect("set MEGANEURA_DEC_REF");
    let frames = env_usize("MEGANEURA_DEC_REF_FRAMES", 3);
    let enc_seq = env_usize("MEGANEURA_DEC_REF_ENCSEQ", 5);

    let cfg = LlmConfig::base();
    let embed = cfg.embed_dim as usize;
    let vocab = cfg.vocab_size as usize;
    let levels = cfg.num_levels as usize;
    let tq = frames * levels;

    // Deterministic input — must match ref_decoder_input_tokens / ref_encoder_out.
    let tokens: Vec<u32> = (0..tq).map(|i| ((i * 53 + 11) % vocab) as u32).collect();
    let enc: Vec<f32> = (0..enc_seq)
        .flat_map(|i| (0..embed).map(move |j| 0.1_f32 * (0.7 * i as f32 + 0.013 * j as f32).sin()))
        .collect();

    let mut g = Graph::new();
    let tok = g.input_u32("dec_tokens", &[tq]);
    let enc_node = g.input("enc_out", &[enc_seq, embed]);
    let logits = build_decoder_faithful(&mut g, &cfg, tok, enc_node, frames);
    g.set_outputs(vec![logits]);
    let mut s = meganeura::build_inference_session(&g);

    let model =
        SafeTensorsModel::load(std::path::PathBuf::from(&weights_path)).expect("open safetensors");
    let skipped = load_llm_weights(&mut s, &model, &cfg).expect("load weights");
    assert!(
        skipped.is_empty(),
        "decoder params not covered: {skipped:?}"
    );

    s.set_input_u32("dec_tokens", &tokens);
    s.set_input("enc_out", &enc);
    s.step();
    s.wait();
    let gpu = s.read_output(tq * vocab);

    // Reference: logits and the encoder output used to produce them (sanity-check
    // the enc sidecar matches what we fed).
    let reference = read_f32_le(&ref_path);
    assert_eq!(reference.len(), tq * vocab, "ref logits size mismatch");
    let enc_ref = read_f32_le(&format!("{ref_path}.enc"));
    assert_eq!(enc_ref.len(), enc.len(), "ref enc size mismatch");
    let enc_drift = enc
        .iter()
        .zip(&enc_ref)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(enc_drift < 1e-5, "encoder input drift vs ref: {enc_drift}");

    // Logits are O(10–60); compare with abs+rel tolerance. A missing-PE or
    // rel-pos-bucket bug shifts logits by O(1) (the PE effect alone is ~2).
    let mut max_abs = 0.0_f32;
    let mut argmax_mismatch = 0usize;
    for row in 0..tq {
        let g_row = &gpu[row * vocab..(row + 1) * vocab];
        let r_row = &reference[row * vocab..(row + 1) * vocab];
        for (a, b) in g_row.iter().zip(r_row) {
            max_abs = max_abs.max((a - b).abs());
        }
        let am = |v: &[f32]| {
            v.iter()
                .enumerate()
                .max_by(|x, y| x.1.partial_cmp(y.1).unwrap())
                .unwrap()
                .0
        };
        if am(g_row) != am(r_row) {
            argmax_mismatch += 1;
        }
    }
    eprintln!(
        "decoder real-weight gate (frames={frames}): max abs diff {max_abs:.3e}, argmax mismatches {argmax_mismatch}/{tq}"
    );
    assert!(
        max_abs <= 5e-2,
        "decoder logits differ from NumPy reference by {max_abs}"
    );
    assert_eq!(
        argmax_mismatch, 0,
        "greedy argmax disagrees on {argmax_mismatch} rows"
    );
}
