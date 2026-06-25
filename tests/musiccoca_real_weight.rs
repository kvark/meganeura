//! Real-weight numeric gate for the MusicCoCa text tower
//! (`build_text_encoder_graph` + `rvq_quantize`).
//!
//! Loads the real `musiccoca_mv212f` weights into the Rust text encoder, runs
//! the 26 testdata prompts, and asserts the contrastive style embedding matches
//! the reference (`testdata/musiccoca_mv212/embeddings`) by cosine, and that the
//! 12 RVQ style tokens match the reference (`.../tokens`). This is the first
//! end-to-end real-weight check of the loader (`load_text_encoder_weights`) — its
//! opaque `tf_var_leaves` arg→param mapping and the three load-time transforms
//! (sqrt-d embed scale, +1 LayerNorm offset, output-projection transpose) — plus
//! the attention-pool head and the RVQ quantizer + codebook level ordering.
//!
//! The token recipe (lowercase → SentencePiece → SOS=1 prefix, unpadded) was
//! verified to reproduce the reference embeddings at cosine 1.0000 via the
//! SavedModel oracle; the gate bundle ships the resulting ids so this test needs
//! no tokenizer/TF. Codebooks are pre-arranged in RVQ-level order
//! (`numeric_codebook_order`, = `[0,1,4,5,6,7,8,9,10,11,2,3]`).
//!
//! Gated behind env vars (the ~1.2 GB dump + bundle aren't in CI):
//!
//! - `MEGANEURA_MUSICCOCA_WEIGHTS=<weights_musiccoca.safetensors>` (dump_musiccoca.py)
//! - `MEGANEURA_MUSICCOCA_GATE=<musiccoca_gate.safetensors>` (ids/lens/ref_embeddings/
//!   ref_tokens/codebooks, all f32)

use meganeura::data::safetensors::SafeTensorsModel;
use meganeura::models::magenta_rt::musiccoca::{
    build_text_encoder_graph, load_text_encoder_weights, numeric_codebook_order, rvq_quantize,
    MusicCoCaConfig,
};
use meganeura::Graph;

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb + 1e-9)
}

#[test]
#[ignore = "requires MEGANEURA_MUSICCOCA_WEIGHTS + MEGANEURA_MUSICCOCA_GATE + a Vulkan device"]
fn musiccoca_text_tower_matches_testdata_on_real_weights() {
    let weights_path =
        std::env::var("MEGANEURA_MUSICCOCA_WEIGHTS").expect("set MEGANEURA_MUSICCOCA_WEIGHTS");
    let gate_path =
        std::env::var("MEGANEURA_MUSICCOCA_GATE").expect("set MEGANEURA_MUSICCOCA_GATE");

    let cfg = MusicCoCaConfig::default();
    let embed = cfg.embed_dim as usize;
    let depth = cfg.rvq_depth as usize;
    let cbsz = cfg.codebook_size as usize;

    let model = SafeTensorsModel::load(std::path::PathBuf::from(&weights_path))
        .expect("open musiccoca weights");
    let gate =
        SafeTensorsModel::load(std::path::PathBuf::from(&gate_path)).expect("open gate bundle");

    // Gate bundle (all f32; ids/lens/tokens are exact integers).
    let ids_f = gate.tensor_f32("ids").expect("ids");
    let lens_f = gate.tensor_f32("lens").expect("lens");
    let ref_emb = gate.tensor_f32("ref_embeddings").expect("ref_embeddings");
    let ref_tok = gate.tensor_f32("ref_tokens").expect("ref_tokens");
    let codebooks = gate.tensor_f32("codebooks").expect("codebooks");
    let n = lens_f.len();
    let maxlen = ids_f.len() / n;
    assert_eq!(ref_emb.len(), n * embed);
    assert_eq!(codebooks.len(), depth * embed * cbsz);

    // Sanity: the documented ordering matches the bundle's pre-arrangement.
    assert_eq!(
        numeric_codebook_order(depth),
        vec![0, 1, 4, 5, 6, 7, 8, 9, 10, 11, 2, 3],
        "codebook level order changed — regenerate the gate bundle"
    );

    // Group prompts by sequence length so we build/load one session per distinct
    // length (the Rust encoder has no padding/masking — feed exactly the valid
    // tokens).
    let lens: Vec<usize> = lens_f.iter().map(|&x| x as usize).collect();
    let mut distinct: Vec<usize> = lens.clone();
    distinct.sort_unstable();
    distinct.dedup();

    let mut cos_sum = 0.0_f32;
    let mut cos_min = 1.0_f32;
    let mut tok_match = 0usize;
    let mut tok_total = 0usize;

    for &seq_len in &distinct {
        let mut g = Graph::new();
        let out = build_text_encoder_graph(&mut g, &cfg, seq_len);
        g.set_outputs(vec![out]);
        let mut s = meganeura::build_inference_session(&g);
        load_text_encoder_weights(&model, &mut s, &cfg).expect("load text encoder weights");

        for i in 0..n {
            if lens[i] != seq_len {
                continue;
            }
            let toks: Vec<u32> = (0..seq_len).map(|j| ids_f[i * maxlen + j] as u32).collect();
            s.set_input_u32("text_tokens", &toks);
            s.step();
            s.wait();
            let emb = s.read_output(embed);

            let c = cosine(&emb, &ref_emb[i * embed..(i + 1) * embed]);
            cos_sum += c;
            cos_min = cos_min.min(c);

            let got = rvq_quantize(&emb, &codebooks, depth, embed, cbsz);
            for q in 0..depth {
                if got[q] == ref_tok[i * depth + q] as u32 {
                    tok_match += 1;
                }
                tok_total += 1;
            }
        }
    }

    let cos_mean = cos_sum / n as f32;
    let tok_pct = 100.0 * tok_match as f32 / tok_total as f32;
    eprintln!(
        "MusicCoCa real-weight gate: cosine mean={cos_mean:.4} min={cos_min:.4}; RVQ token match {tok_match}/{tok_total} = {tok_pct:.1}%"
    );
    // Thresholds match the model's inherent accuracy: the independent NumPy
    // reference hits mean 0.9993 / min 0.9965 / 93.3% on this testdata, and the
    // Rust encoder now reproduces those exactly.
    assert!(cos_mean >= 0.999, "mean cosine {cos_mean} < 0.999");
    assert!(cos_min >= 0.99, "min cosine {cos_min} < 0.99");
    assert!(tok_pct >= 90.0, "RVQ token match {tok_pct}% < 90%");
}
