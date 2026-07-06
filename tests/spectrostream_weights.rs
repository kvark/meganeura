//! Verify that the parameters declared in `magenta_rt::spectrostream::build_decoder_graph`
//! exactly match the tensors dumped to `magenta_rt_codec_dump/weights_spectrostream.safetensors`
//! by `tools/magenta_rt/dump_codec_local.py`. This catches naming mistakes
//! before we wire the forward pass.
//!
//! Run with `cargo test --test spectrostream_weights -- --ignored` when the
//! dump file is present locally. Skipped by default since the file is ~427 MB
//! and not in the repo.

use std::collections::{HashMap, HashSet};

use meganeura::Graph;
use meganeura::models::magenta_rt::spectrostream::{SpectroStreamConfig, build_decoder_graph};

const DUMP: &str = "magenta_rt_codec_dump/weights_spectrostream.safetensors";

fn load_safetensors_shapes() -> Option<HashMap<String, Vec<usize>>> {
    if !std::path::Path::new(DUMP).exists() {
        return None;
    }
    let bytes = std::fs::read(DUMP).expect("read dump");
    let st = safetensors::SafeTensors::deserialize(&bytes).expect("parse safetensors");
    let mut out = HashMap::new();
    for (name, tensor) in st.tensors() {
        out.insert(name.to_string(), tensor.shape().to_vec());
    }
    Some(out)
}

fn skeleton_param_names_and_sizes() -> Vec<(String, usize)> {
    let cfg = SpectroStreamConfig::default();
    let mut g = Graph::new();
    let _ = build_decoder_graph(&mut g, &cfg, 50);
    g.nodes()
        .iter()
        .filter_map(|n| {
            if let meganeura::graph::Op::Parameter { name } = &n.op {
                let size: usize = n.ty.shape.iter().product();
                Some((name.clone(), size))
            } else {
                None
            }
        })
        .collect()
}

#[test]
#[ignore]
fn skeleton_param_names_exist_in_safetensors() {
    let Some(st) = load_safetensors_shapes() else {
        eprintln!("skipping: {DUMP} not found (run tools/magenta_rt/dump_codec_local.py first)");
        return;
    };
    let st_names: HashSet<&str> = st.keys().map(String::as_str).collect();
    let ours = skeleton_param_names_and_sizes();
    let missing: Vec<&str> = ours
        .iter()
        .map(|(n, _)| n.as_str())
        .filter(|n| !st_names.contains(n))
        .collect();
    assert!(
        missing.is_empty(),
        "{} declared params not present in safetensors:\n  {}",
        missing.len(),
        missing.join("\n  ")
    );
}

#[test]
#[ignore]
fn skeleton_param_shapes_match_safetensors_byte_count() {
    let Some(st) = load_safetensors_shapes() else {
        eprintln!("skipping: {DUMP} not found");
        return;
    };
    let ours = skeleton_param_names_and_sizes();
    let mut mismatches = Vec::new();
    for (name, declared_size) in &ours {
        if let Some(st_shape) = st.get(name) {
            let st_size: usize = st_shape.iter().product();
            if st_size != *declared_size {
                mismatches.push(format!(
                    "  {name}: declared {} elements, safetensors has {} (shape {:?})",
                    declared_size, st_size, st_shape,
                ));
            }
        }
    }
    assert!(
        mismatches.is_empty(),
        "{} shape size mismatches:\n{}",
        mismatches.len(),
        mismatches.join("\n")
    );
}

#[test]
#[ignore]
fn loader_runs_end_to_end_without_panicking() {
    // Build the decoder graph, load weights via load_decoder_weights, run with
    // a synthetic preprocessed input. Validates only that the loader applies
    // the right permutations (no shape errors) and that the GPU graph runs.
    // Does NOT validate audio output — that's the next step.
    use meganeura::data::safetensors::SafeTensorsModel;
    use meganeura::models::magenta_rt::spectrostream::{load_decoder_weights, SpectroStreamConfig};
    use meganeura::{build_inference_session, Graph};
    use meganeura::models::magenta_rt::spectrostream::build_decoder_graph;

    if !std::path::Path::new(DUMP).exists() {
        eprintln!("skipping: {DUMP} not found");
        return;
    }
    let model = SafeTensorsModel::load(DUMP.into()).expect("safetensors load");

    let cfg = SpectroStreamConfig::default();
    let num_frames = 50;
    let mut g = Graph::new();
    let out = build_decoder_graph(&mut g, &cfg, num_frames);
    g.set_outputs(vec![out]);
    let mut session = build_inference_session(&g);

    let leftover = load_decoder_weights(&model, &mut session).expect("load weights");
    println!("loaded; {} unused safetensors keys (weight_norm artifacts)", leftover.len());

    // Run with synthetic input.
    let h_padded = num_frames + cfg.temporal_pad;
    let in_size = (cfg.initial_channels * h_padded * cfg.initial_freq_bins) as usize;
    let input: Vec<f32> = (0..in_size).map(|i| (i as f32 * 1e-4).sin()).collect();
    session.set_input("decoder_input_preprocessed", &input);
    session.step();
    session.wait();
    // Output shape: [1, 2, T_final=50, F_final=1920] = 192000 elements.
    // T_final: h_padded - 24 (blocks net -14, input_layer -4, base_conv_last -6).
    // F_final: 5 → 1920 via the 7 conv-transpose stride_w pattern (2·2·2·2·3·2·2=192).
    let expected = 2 * 50 * 1920;
    let out = session.read_output(expected);
    assert_eq!(out.len(), expected, "unexpected output length");
    let zeros = out.iter().filter(|&&v| v == 0.0).count();
    let nans = out.iter().filter(|v| v.is_nan()).count();
    let infs = out.iter().filter(|v| v.is_infinite()).count();
    let max_abs = out.iter().filter(|v| v.is_finite()).fold(0.0_f32, |a, &v| a.max(v.abs()));
    let mean_abs: f32 = out.iter().filter(|v| v.is_finite()).map(|v| v.abs()).sum::<f32>()
        / out.iter().filter(|v| v.is_finite()).count().max(1) as f32;
    println!("output diagnostics:");
    println!("  total:   {expected}");
    println!("  zeros:   {zeros} ({:.1}%)", 100.0 * zeros as f32 / expected as f32);
    println!("  NaNs:    {nans}");
    println!("  Infs:    {infs}");
    println!("  max abs: {max_abs:.4e}");
    println!("  mean abs: {mean_abs:.4e}");
    println!("  first 8: {:?}", &out[..8]);
    println!("  middle 8: {:?}", &out[expected/2..expected/2 + 8]);
}

#[test]
#[ignore]
fn skeleton_covers_all_decoder_safetensors_keys() {
    // The reverse direction: every `decoder.*` and `quantizer.*` tensor in the
    // safetensors file (modulo weight_norm artifacts we intentionally skip)
    // should be declared by our skeleton. Catches missing layers.
    let Some(st) = load_safetensors_shapes() else {
        eprintln!("skipping: {DUMP} not found");
        return;
    };
    let ours: HashSet<String> = skeleton_param_names_and_sizes().into_iter().map(|(n, _)| n).collect();

    // Filter ignored artifacts: meganeura uses the post-norm `rescaled.kernel`,
    // not the raw `kernel`, scale `g`, or boolean `initialized`. Encoder tensors
    // are out of scope for the decoder-only skeleton.
    let st_relevant: Vec<&str> = st
        .keys()
        .filter(|k| !k.starts_with("encoder."))
        .filter(|k| !k.ends_with(".weight_norm.initialized"))
        .filter(|k| !k.ends_with(".weight_norm.g"))
        // The raw `.weight_norm.kernel` is unused (we use `.rescaled.kernel`).
        // But the *bare* `.conv.kernel` and `.conv.bias` (base_conv_last,
        // which is NOT weight-normed) IS used.
        .filter(|k| {
            !k.ends_with(".weight_norm.kernel") || k.ends_with(".rescaled.kernel")
        })
        .map(String::as_str)
        .collect();

    let missing: Vec<&str> = st_relevant.iter().filter(|k| !ours.contains(**k)).cloned().collect();
    assert!(
        missing.is_empty(),
        "{} safetensors keys not declared by skeleton:\n  {}",
        missing.len(),
        missing.join("\n  ")
    );
}
