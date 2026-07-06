//! Real-weight numeric gate for the LLM encoder (`build_encoder_graph`).
//!
//! Loads the real `llm_base_x4286_c1860k` weights into the encoder graph, runs a
//! forward on the GPU (lavapipe), and compares the output against an independent
//! NumPy reference (`tools/magenta_rt/llm_numpy_ref.py`) computed from the *same*
//! weights. Both sides use the deterministic input `ids[i] = (i*101 + 7) % vocab`.
//!
//! Unlike `tests/llm_encoder_correctness.rs` (random weights, op-composition
//! only), this runs at **real weight magnitudes** and cross-checks a *separate*
//! implementation, so it gates:
//!
//! - the flat-copy `target.*` → graph weight mapping (a transpose bug in any
//!   Q/K/V/O or MLP kernel would surface here — random-symmetric weights can
//!   hide it, real asymmetric ones cannot),
//! - the encoder op composition (attention, GeGLU, RMSNorm, final norm) end to
//!   end against NumPy einsum code.
//!
//! It does **not** settle the position *scheme*: both sides add the same computed
//! sinusoidal PE (and omit `Scale(sqrt(embed))`). The checkpoint carries no
//! encoder PE/rel-pos tensor, so that choice is unsettleable from the weights
//! alone — see `LLM_FINDINGS.md`.
//!
//! Gated behind two env vars (the 1.3 GB checkpoint isn't in CI):
//!
//! - `MEGANEURA_LLM_WEIGHTS=<weights_llm_base.safetensors>` (from `dump_llm.py`)
//! - `MEGANEURA_ENC_REF=<enc_ref.bin>` — raw little-endian f32 `[seq, embed]`
//!   written by `llm_numpy_ref.py` (`MEGANEURA_ENC_REF_OUT=…`), same seq.
//!
//! Run both with the same `MEGANEURA_ENC_REF_SEQ` (default 32).

use meganeura::data::safetensors::SafeTensorsModel;
use meganeura::models::magenta_rt::llm::{build_encoder_graph, LlmConfig};
use meganeura::models::magenta_rt::llm_weights::load_llm_weights;
use meganeura::Graph;

/// Deterministic encoder input — must match `ref_input_ids` in `llm_numpy_ref.py`.
fn ref_input_ids(seq: usize, vocab: usize) -> Vec<u32> {
    (0..seq).map(|i| ((i * 101 + 7) % vocab) as u32).collect()
}

fn read_f32_le(path: &str) -> Vec<f32> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    assert!(
        bytes.len().is_multiple_of(4),
        "{path}: not a multiple of 4 bytes"
    );
    bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

#[test]
#[ignore = "requires MEGANEURA_LLM_WEIGHTS + MEGANEURA_ENC_REF + a Vulkan device"]
fn encoder_matches_numpy_reference_on_real_weights() {
    let weights_path = std::env::var("MEGANEURA_LLM_WEIGHTS").expect("set MEGANEURA_LLM_WEIGHTS");
    let ref_path = std::env::var("MEGANEURA_ENC_REF").expect("set MEGANEURA_ENC_REF");
    let seq: usize = std::env::var("MEGANEURA_ENC_REF_SEQ")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);

    let cfg = LlmConfig::base();
    let embed = cfg.embed_dim as usize;
    let vocab = cfg.vocab_size as usize;

    // Build the encoder at the reference seq length and load the real weights.
    let mut g = Graph::new();
    let out = build_encoder_graph(&mut g, &cfg, seq);
    g.set_outputs(vec![out]);
    let mut s = meganeura::build_inference_session(&g);

    let model =
        SafeTensorsModel::load(std::path::PathBuf::from(&weights_path)).expect("open safetensors");
    let skipped = load_llm_weights(&mut s, &model, &cfg).expect("load weights");
    // The encoder graph's params are all checkpoint-backed (the PE is computed),
    // so nothing should be skipped.
    assert!(
        skipped.is_empty(),
        "encoder params not covered: {skipped:?}"
    );

    let tokens = ref_input_ids(seq, vocab);
    s.set_input_u32("encoder_input_tokens", &tokens);
    s.step();
    s.wait();
    let gpu = s.read_output(seq * embed);

    let reference = read_f32_le(&ref_path);
    assert_eq!(
        reference.len(),
        seq * embed,
        "reference has {} elems, expected seq*embed={}",
        reference.len(),
        seq * embed
    );

    // Compare. Encoder outputs are small (post-RMSNorm, O(0.1)), so use an
    // absolute+relative tolerance; a transpose/mapping bug diverges by O(1).
    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    for (i, (&a, &b)) in gpu.iter().zip(reference.iter()).enumerate() {
        let d = (a - b).abs();
        max_abs = max_abs.max(d);
        max_rel = max_rel.max(d / b.abs().max(1e-3));
        assert!(
            d <= 2e-3 + 1e-2 * b.abs(),
            "elem {i}: gpu {a} vs numpy {b} (abs diff {d})"
        );
    }
    eprintln!(
        "encoder real-weight gate OK (seq={seq}): max abs diff {max_abs:.2e}, max rel {max_rel:.2e}"
    );
}
