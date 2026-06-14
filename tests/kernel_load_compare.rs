//! Verify meganeura's flip+transpose of conv-T kernels matches the equivalent
//! NumPy ref operation. Both should produce identical post-transformation data.

use meganeura::data::safetensors::SafeTensorsModel;
use meganeura::models::magenta_rt::spectrostream::{
    build_decoder_graph_through, load_decoder_weights, DecoderStage, SpectroStreamConfig,
};
use meganeura::{build_inference_session, Graph};
use std::path::Path;

const WEIGHTS: &str = "magenta_rt_codec_dump/weights_spectrostream.safetensors";

#[test]
#[ignore]
fn meganeura_kernel_matches_python_flipped_transposed() {
    if !Path::new(WEIGHTS).exists() {
        return;
    }
    let model = SafeTensorsModel::load(WEIGHTS.into()).unwrap();
    let cfg = SpectroStreamConfig::default();
    let mut g = Graph::new();
    let _ = build_decoder_graph_through(&mut g, &cfg, 50, DecoderStage::Block(0));
    let mut s = build_inference_session(&g);
    load_decoder_weights(&model, &mut s).unwrap();

    // decoder_0's conv2dtranspose: TF kh=4, kw=3, out_c=1024, in_c=512.
    let name = "decoder.decoder_0.conv2dtranspose_4x3.weight_norm.rescaled.kernel";
    let raw_tf = model.tensor_f32_auto(name).unwrap();
    let info = &model.tensor_info()[name];
    let kh = info.shape[0]; let kw = info.shape[1];
    let out_c = info.shape[2]; let in_c = info.shape[3];
    println!("TF shape: kh={kh} kw={kw} out_c={out_c} in_c={in_c}");

    // Read what meganeura stored in the parameter buffer.
    let n_elems = kh * kw * out_c * in_c;
    let mut loaded = vec![0.0_f32; n_elems];
    s.read_param(name, &mut loaded);

    // Compute the equivalent transformation in plain Rust (= what Python's NumPy
    // does via `kernel_hwoi[::-1, ::-1].transpose(0, 1, 3, 2)` then read as
    // `[kh, kw, in_c, out_c]` flat).
    // Input layout: [kh, kw, out_c, in_c] (TF, row-major flat).
    // Python's kernel_fwd[kh', kw', ic, oc] = raw_tf[kH-1-kh', kW-1-kw', oc, ic]
    let mut python_fwd = vec![0.0_f32; n_elems];
    for kh_idx in 0..kh {
        for kw_idx in 0..kw {
            for ic in 0..in_c {
                for oc in 0..out_c {
                    let src = ((kh - 1 - kh_idx) * kw + (kw - 1 - kw_idx)) * out_c * in_c
                            + oc * in_c + ic;
                    // Python flat layout: [kh, kw, ic, oc]
                    let dst = (kh_idx * kw + kw_idx) * in_c * out_c + ic * out_c + oc;
                    python_fwd[dst] = raw_tf[src];
                }
            }
        }
    }

    // meganeura stores [out_c, in_c, kh, kw] flat.
    // Convert to python's [kh, kw, in_c, out_c] flat for comparison.
    let mut megan_in_python_layout = vec![0.0_f32; n_elems];
    for oc in 0..out_c {
        for ic in 0..in_c {
            for kh_idx in 0..kh {
                for kw_idx in 0..kw {
                    let src = ((oc * in_c + ic) * kh + kh_idx) * kw + kw_idx;
                    let dst = (kh_idx * kw + kw_idx) * in_c * out_c + ic * out_c + oc;
                    megan_in_python_layout[dst] = loaded[src];
                }
            }
        }
    }

    let mut max_diff = 0.0_f32;
    let mut n_diff = 0;
    for i in 0..n_elems {
        let d = (python_fwd[i] - megan_in_python_layout[i]).abs();
        if d > 1e-10 { n_diff += 1; }
        if d > max_diff { max_diff = d; }
    }
    println!("max_diff = {max_diff:.6e}, n_diff/total = {n_diff}/{n_elems}");
    assert!(max_diff < 1e-5, "kernels should match");
}
