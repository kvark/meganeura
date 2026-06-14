//! Validate meganeura's SpectroStream GPU graph against a NumPy reference.
//!
//! Both implementations consume the SAME synthetic preprocessed input
//! (sin(i * 1e-7)) so we isolate the GPU graph from the host-side
//! preprocessing (codebook dequant + input_layer expansion).
//!
//! Run the NumPy reference first to produce `decoder_reference.safetensors`:
//!   nix-shell -p python3Packages.numpy --run \
//!     "python3 tools/magenta_rt/decoder_reference.py"
//! Then this test compares meganeura's intermediates against it stage-by-stage.

use meganeura::data::safetensors::SafeTensorsModel;
use meganeura::models::magenta_rt::spectrostream::{
    build_decoder_graph_through, load_decoder_weights, DecoderStage, SpectroStreamConfig,
};
use meganeura::{build_inference_session, Graph};
use std::path::Path;

const WEIGHTS: &str = "magenta_rt_codec_dump/weights_spectrostream.safetensors";
const REF: &str = "magenta_rt_codec_dump/decoder_reference.safetensors";

fn stats(label: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len(), "{label}: length mismatch {} vs {}", a.len(), b.len());
    let mut max_abs_diff = 0.0_f32;
    let mut max_abs_ref = 0.0_f32;
    let mut sum_diff = 0.0_f64;
    let mut sum_ref = 0.0_f64;
    let mut nan_or_inf = 0usize;
    for (&x, &y) in a.iter().zip(b.iter()) {
        if !x.is_finite() || !y.is_finite() {
            nan_or_inf += 1;
            continue;
        }
        let d = (x - y).abs();
        if d > max_abs_diff { max_abs_diff = d; }
        if y.abs() > max_abs_ref { max_abs_ref = y.abs(); }
        sum_diff += d as f64;
        sum_ref += y.abs() as f64;
    }
    let mean_abs_diff = sum_diff / a.len() as f64;
    let mean_abs_ref = sum_ref / a.len() as f64;
    let rel = if mean_abs_ref > 0.0 { mean_abs_diff / mean_abs_ref } else { f64::NAN };
    println!(
        "[{label}] n={} max_abs_diff={max_abs_diff:.3e} max_abs_ref={max_abs_ref:.3e} mean_abs_diff={mean_abs_diff:.3e} rel={rel:.4} nan/inf={nan_or_inf}",
        a.len()
    );
}

/// Standalone conv-T check: feed Python's `block0_after_elu` into
/// meganeura's V1 `conv_transpose_2d_hw` (NOT through SpectroStream's
/// dilate+forward rewrite) and compare against Python's `block0_after_convT`.
/// If meganeura V1 agrees with Python NumPy, the disagreement at Block(0)
/// must be downstream (3x3 conv, shortcut, pixel_shuffle, or bias).
#[test]
#[ignore]
fn standalone_convt_block0() {
    if !Path::new(WEIGHTS).exists() || !Path::new(REF).exists() { return; }
    let model = SafeTensorsModel::load(WEIGHTS.into()).unwrap();
    let reference = SafeTensorsModel::load(REF.into()).unwrap();

    // Python's block0_after_elu = ELU(stage_input_layer). Shape [1, 512, 70, 5].
    let input_elu = reference.tensor_f32_auto("block0_after_elu").unwrap();
    let bsize = 1u32 * 512 * 70 * 5;
    assert_eq!(input_elu.len(), bsize as usize);

    // Build a graph: x → conv_transpose_2d_hw (4x3 stride (1,2), no padding).
    let mut g = meganeura::Graph::new();
    let x = g.input("x", &[bsize as usize]);
    let k = g.parameter("k", &[4 * 3 * 512 * 1024]);
    let y = g.conv_transpose_2d_hw(
        x, k,
        1, 512, 70, 5,
        1024, 4, 3,
        1, 2, 0, 0,
    );
    let bias = g.parameter("bias", &[1024]);
    let y = g.add_per_channel(y, bias, 1024, 73 * 11); // post-bias
    g.set_outputs(vec![y]);
    let mut s = build_inference_session(&g);
    s.set_input("x", &input_elu);

    // Load the ORIGINAL (PyTorch-layout) conv-T kernel: [in_c=512, out_c=1024, kh=4, kw=3].
    // That's what conv_transpose_2d_hw expects (NOT the SpectroStream flipped+transposed version).
    let tf_kernel = model.tensor_f32_auto("decoder.decoder_0.conv2dtranspose_4x3.weight_norm.rescaled.kernel").unwrap();
    // TF layout [kh, kw, out_c=1024, in_c=512]. Permute to [in_c, out_c, kh, kw].
    let mut permuted = vec![0.0_f32; tf_kernel.len()];
    for kh in 0..4 { for kw in 0..3 { for oc in 0..1024 { for ic in 0..512 {
        let src = ((kh * 3 + kw) * 1024 + oc) * 512 + ic;
        let dst = ((ic * 1024 + oc) * 4 + kh) * 3 + kw;
        permuted[dst] = tf_kernel[src];
    }}}}
    s.set_parameter("k", &permuted);
    let tf_bias = model.tensor_f32_auto("decoder.decoder_0.conv2dtranspose_4x3.weight_norm.bias").unwrap();
    s.set_parameter("bias", &tf_bias);

    s.step();
    s.wait();
    let our = s.read_output((1024 * 73 * 11) as usize);
    let ref_data = reference.tensor_f32_auto("block0_after_convT").unwrap();
    println!("ref len = {}, our len = {}", ref_data.len(), our.len());
    stats("standalone-convT", &our, &ref_data);
}

/// Standalone shortcut path for Block(0): input → conv1x1 → upsample_w → slice2d.
#[test]
#[ignore]
fn standalone_block0_shortcut() {
    if !Path::new(WEIGHTS).exists() || !Path::new(REF).exists() { return; }
    let model = SafeTensorsModel::load(WEIGHTS.into()).unwrap();
    let reference = SafeTensorsModel::load(REF.into()).unwrap();

    let mut g = meganeura::Graph::new();
    // Shortcut input is `stage_input_layer` (BEFORE the elu).
    let x = g.input("x", &[(1*512*70*5) as usize]);
    let sc = g.parameter("sc", &[1 * 1 * 512 * 1024]);
    let sc_bias = g.parameter("sc_bias", &[1024]);
    // 1x1 conv: 512 → 1024.
    let after_conv1x1 = g.conv2d_hw(x, sc, 1, 512, 70, 5, 1024, 1, 1, 1, 0, 0);
    let after_conv1x1 = g.add_per_channel(after_conv1x1, sc_bias, 1024, 70 * 5);
    let after_upsample = g.upsample_nearest(after_conv1x1, 1, 1024, 70, 5, 1, 2);
    let after_slice = g.slice_2d(after_upsample, 1, 1024, 70, 10, 1, 1, 0, 0);
    g.set_outputs(vec![after_slice]);
    let mut s = build_inference_session(&g);

    s.set_input("x", &reference.tensor_f32_auto("stage_input_layer").unwrap());
    let tf_sc = model.tensor_f32_auto("decoder.decoder_0.shortcut.conv1x1.weight_norm.rescaled.kernel").unwrap();
    let mut perm_sc = vec![0.0_f32; tf_sc.len()];
    // TF [1, 1, in_c=512, out_c=1024] → meganeura [out_c, in_c, 1, 1].
    for ic in 0..512 { for oc in 0..1024 {
        let src = ic * 1024 + oc;
        let dst = oc * 512 + ic;
        perm_sc[dst] = tf_sc[src];
    }}
    s.set_parameter("sc", &perm_sc);
    s.set_parameter("sc_bias", &model.tensor_f32_auto("decoder.decoder_0.shortcut.conv1x1.weight_norm.bias").unwrap());

    s.step();
    s.wait();
    let out = s.read_output((1024 * 68 * 10) as usize);
    stats("block0_shortcut", &out, &reference.tensor_f32_auto("block0_shortcut").unwrap());
}

/// Full Block(0) main path standalone: input → conv-T → crop → elu → conv2d_3x3.
/// Compare each intermediate against Python.
#[test]
#[ignore]
fn standalone_block0_main_path() {
    if !Path::new(WEIGHTS).exists() || !Path::new(REF).exists() { return; }
    let model = SafeTensorsModel::load(WEIGHTS.into()).unwrap();
    let reference = SafeTensorsModel::load(REF.into()).unwrap();

    let mut g = meganeura::Graph::new();
    let x_after_elu = g.input("x", &[(1*512*70*5) as usize]);
    let kt = g.parameter("kt", &[4 * 3 * 512 * 1024]);
    let kt_bias = g.parameter("kt_bias", &[1024]);
    let c3 = g.parameter("c3", &[3 * 3 * 1024 * 1024]);
    let c3_bias = g.parameter("c3_bias", &[1024]);

    let after_convt = g.conv_transpose_2d_hw(
        x_after_elu, kt, 1, 512, 70, 5, 1024, 4, 3, 1, 2, 0, 0);
    let after_convt = g.add_per_channel(after_convt, kt_bias, 1024, 73 * 11);
    let after_crop = g.slice_2d(after_convt, 1, 1024, 73, 11, 1, 2, 0, 1);
    let after_second_elu = g.elu(after_crop);
    let after_3x3 = g.conv2d_hw(
        after_second_elu, c3, 1, 1024, 70, 10, 1024, 3, 3, 1, 0, 1);
    let after_3x3 = g.add_per_channel(after_3x3, c3_bias, 1024, 68 * 10);
    g.set_outputs(vec![after_convt, after_crop, after_3x3]);
    let mut s = build_inference_session(&g);

    s.set_input("x", &reference.tensor_f32_auto("block0_after_elu").unwrap());
    // ConvT kernel: load TF kernel and permute to [in_c=512, out_c=1024, kh=4, kw=3].
    let tf_kt = model.tensor_f32_auto("decoder.decoder_0.conv2dtranspose_4x3.weight_norm.rescaled.kernel").unwrap();
    let mut perm_kt = vec![0.0_f32; tf_kt.len()];
    for kh in 0..4 { for kw in 0..3 { for oc in 0..1024 { for ic in 0..512 {
        let src = ((kh * 3 + kw) * 1024 + oc) * 512 + ic;
        let dst = ((ic * 1024 + oc) * 4 + kh) * 3 + kw;
        perm_kt[dst] = tf_kt[src];
    }}}}
    s.set_parameter("kt", &perm_kt);
    s.set_parameter("kt_bias", &model.tensor_f32_auto("decoder.decoder_0.conv2dtranspose_4x3.weight_norm.bias").unwrap());

    // 3x3 kernel: TF [3, 3, in_c=1024, out_c=1024] → meganeura [out_c, in_c, kh, kw].
    let tf_c3 = model.tensor_f32_auto("decoder.decoder_0.conv2d_3x3.weight_norm.rescaled.kernel").unwrap();
    let mut perm_c3 = vec![0.0_f32; tf_c3.len()];
    for kh in 0..3 { for kw in 0..3 { for ic in 0..1024 { for oc in 0..1024 {
        let src = ((kh * 3 + kw) * 1024 + ic) * 1024 + oc;
        let dst = ((oc * 1024 + ic) * 3 + kh) * 3 + kw;
        perm_c3[dst] = tf_c3[src];
    }}}}
    s.set_parameter("c3", &perm_c3);
    s.set_parameter("c3_bias", &model.tensor_f32_auto("decoder.decoder_0.conv2d_3x3.weight_norm.bias").unwrap());

    s.step();
    s.wait();
    let mut convt_out = vec![0.0_f32; (1024 * 73 * 11) as usize];
    s.read_output_by_index(0, &mut convt_out);
    let mut crop_out = vec![0.0_f32; (1024 * 70 * 10) as usize];
    s.read_output_by_index(1, &mut crop_out);
    let mut conv3_out = vec![0.0_f32; (1024 * 68 * 10) as usize];
    s.read_output_by_index(2, &mut conv3_out);

    stats("block0_after_convT", &convt_out, &reference.tensor_f32_auto("block0_after_convT").unwrap());
    stats("block0_after_crop",  &crop_out,  &reference.tensor_f32_auto("block0_after_crop").unwrap());
    stats("block0_main_after_3x3", &conv3_out, &reference.tensor_f32_auto("block0_main_after_3x3").unwrap());
}

#[test]
#[ignore]
fn compare_against_numpy_reference() {
    if !Path::new(WEIGHTS).exists() {
        eprintln!("skip: {WEIGHTS} not found");
        return;
    }
    if !Path::new(REF).exists() {
        eprintln!("skip: {REF} not found — run tools/magenta_rt/decoder_reference.py first");
        return;
    }
    let model = SafeTensorsModel::load(WEIGHTS.into()).unwrap();
    let reference = SafeTensorsModel::load(REF.into()).unwrap();
    let cfg = SpectroStreamConfig::default();
    let num_frames = std::env::var("NUM_FRAMES").ok().and_then(|s| s.parse().ok()).unwrap_or(50u32);
    let h_padded = num_frames + cfg.temporal_pad;
    let in_size = (cfg.initial_channels * h_padded * cfg.initial_freq_bins) as usize;
    // Same synthetic input as decoder_reference.py and tests/spectrostream_bisect.rs s10.
    let input: Vec<f32> = (0..in_size).map(|i| (i as f32 * 1e-7).sin()).collect();

    let stages: &[(&str, &str, DecoderStage)] = &[
        ("stage_input_layer", "InputLayer", DecoderStage::InputLayer),
        ("stage_block_0",     "Block(0)",   DecoderStage::Block(0)),
        ("stage_block_1",     "Block(1)",   DecoderStage::Block(1)),
        ("stage_block_2",     "Block(2)",   DecoderStage::Block(2)),
        ("stage_block_3",     "Block(3)",   DecoderStage::Block(3)),
        ("stage_block_4",     "Block(4)",   DecoderStage::Block(4)),
        ("stage_block_5",     "Block(5)",   DecoderStage::Block(5)),
        ("stage_block_6",     "Block(6)",   DecoderStage::Block(6)),
        ("stage_output",      "Output",     DecoderStage::Output),
    ];
    for &(ref_name, label, stage) in stages {
        let mut g = Graph::new();
        let out = build_decoder_graph_through(&mut g, &cfg, num_frames, stage);
        g.set_outputs(vec![out]);
        let mut s = build_inference_session(&g);
        load_decoder_weights(&model, &mut s).unwrap();
        s.set_input("decoder_input_preprocessed", &input);
        s.step();
        s.wait();
        let out_size = g.node(out).ty.shape.iter().product::<usize>();
        let our = s.read_output(out_size);
        let ref_data = reference.tensor_f32_auto(ref_name).unwrap();
        stats(label, &our, &ref_data);
    }
}
