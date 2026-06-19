//! Verifies the LLM real-weight mapping (`llm_weights::checkpoint_param_map`)
//! against the committed checkpoint manifest — no weight download needed.
//!
//! Asserts (1) every mapped checkpoint tensor exists in the manifest with an
//! element count matching the graph parameter, and (2) every `target.*` tensor
//! in the manifest is consumed by exactly one mapping (100% checkpoint
//! coverage). This pins the flaxformer↔graph naming and the no-transpose claim.

use std::collections::HashSet;

use meganeura::models::magenta_rt::llm::LlmConfig;
use meganeura::models::magenta_rt::llm_weights::checkpoint_param_map;

const MANIFEST: &str = include_str!("../tools/magenta_rt/llm_base_manifest.json");

fn manifest_shapes() -> std::collections::HashMap<String, Vec<usize>> {
    let v: serde_json::Value = serde_json::from_str(MANIFEST).expect("parse manifest json");
    let tensors = v["tensors"].as_object().expect("tensors object");
    tensors
        .iter()
        .map(|(name, info)| {
            let shape: Vec<usize> = info["shape"]
                .as_array()
                .expect("shape array")
                .iter()
                .map(|x| x.as_u64().expect("dim") as usize)
                .collect();
            (name.clone(), shape)
        })
        .collect()
}

#[test]
fn mapping_matches_base_checkpoint_manifest() {
    let cfg = LlmConfig::base();
    let shapes = manifest_shapes();
    let map = checkpoint_param_map(&cfg);

    // (1) Every mapped checkpoint tensor exists with the expected element count.
    for p in &map {
        let shape = shapes
            .get(&p.ckpt)
            .unwrap_or_else(|| panic!("mapping references unknown checkpoint tensor: {}", p.ckpt));
        let got: usize = shape.iter().product();
        assert_eq!(
            got, p.numel,
            "{}: manifest shape {:?} = {} elems, graph param {} expects {}",
            p.ckpt, shape, got, p.graph, p.numel
        );
    }

    // (2) Every target.* manifest tensor is consumed exactly once.
    let mapped: Vec<&String> = map.iter().map(|p| &p.ckpt).collect();
    let mapped_set: HashSet<&String> = mapped.iter().copied().collect();
    assert_eq!(
        mapped.len(),
        mapped_set.len(),
        "duplicate checkpoint tensor in mapping"
    );

    let target_tensors: Vec<&String> = shapes.keys().filter(|k| k.starts_with("target.")).collect();
    for t in &target_tensors {
        assert!(
            mapped_set.contains(*t),
            "checkpoint tensor not consumed by any mapping: {t}"
        );
    }
    assert_eq!(
        mapped.len(),
        target_tensors.len(),
        "mapping covers {} tensors but manifest has {} target.* tensors",
        mapped.len(),
        target_tensors.len()
    );

    // Graph param names must be unique too.
    let graph_set: HashSet<&String> = map.iter().map(|p| &p.graph).collect();
    assert_eq!(
        graph_set.len(),
        map.len(),
        "duplicate graph param in mapping"
    );

    eprintln!(
        "LLM weight map: {} tensors, 100% of checkpoint covered",
        map.len()
    );
}

/// Real-weight load smoke test — runs only when `MEGANEURA_LLM_WEIGHTS` points
/// at a `weights_llm_base.safetensors` dumped by `tools/magenta_rt/dump_llm.py`
/// (the ~1.3 GB checkpoint isn't downloaded by the CI harness). Builds a full
/// parallel decoder, loads the real weights, and asserts nothing was skipped
/// (the decoder graph's params are 100% covered by the checkpoint).
#[test]
#[ignore = "requires MEGANEURA_LLM_WEIGHTS=<weights_llm_base.safetensors>"]
fn load_real_weights_into_decoder() {
    use meganeura::data::safetensors::SafeTensorsModel;
    use meganeura::models::magenta_rt::llm::build_decoder;
    use meganeura::models::magenta_rt::llm_weights::load_llm_weights;
    use meganeura::Graph;

    let path = std::env::var("MEGANEURA_LLM_WEIGHTS").expect("set MEGANEURA_LLM_WEIGHTS");
    let model = SafeTensorsModel::load(std::path::PathBuf::from(&path)).expect("open safetensors");
    let cfg = LlmConfig::base();
    let embed = cfg.embed_dim as usize;
    let enc_seq = cfg.encoder_seq_len as usize;
    let levels = cfg.num_levels as usize;
    let frames = cfg.decoder_seq_len as usize / levels;

    let mut g = Graph::new();
    let tok = g.input_u32("dec_tokens", &[(frames + 1) * levels]);
    let enc = g.input("enc_out", &[enc_seq, embed]);
    let logits = build_decoder(&mut g, &cfg, tok, enc, frames);
    g.set_outputs(vec![logits]);
    let mut s = meganeura::build_inference_session(&g);

    let skipped = load_llm_weights(&mut s, &model, &cfg).expect("load weights");
    assert!(
        skipped.is_empty(),
        "decoder graph should be fully covered, skipped: {skipped:?}"
    );
    eprintln!("loaded all decoder weights from {path}");
}
