//! Compare meganeura's SpectroStream decoder output against TF's body output
//! for the reference token sequence.
//!
//! Requires:
//!   - `magenta_rt_codec_dump/weights_spectrostream.safetensors`
//!   - `magenta_rt_codec_dump/decoder_reference_v2.safetensors` (provides
//!     the correctly preprocessed input; rebuild with
//!     `USE_TOKENS=1 python3 tools/magenta_rt/decoder_reference_v2.py`)
//!   - `magenta_rt_codec_dump/tf_intermediates.safetensors` (provides
//!     `body_out` ground truth for the reference tokens)
//!
//! Compares meganeura's `[1, 4, 200, 480]` NCHW body output against TF's
//! `[1, 200, 480, 4]` NHWC `body_out` (transposed for comparison).

use meganeura::build_inference_session;
use meganeura::data::safetensors::SafeTensorsModel;
use meganeura::models::magenta_rt::spectrostream::{
    build_decoder_graph, load_decoder_weights, SpectroStreamConfig,
};
use std::path::Path;

const WEIGHTS: &str = "magenta_rt_codec_dump/weights_spectrostream.safetensors";
const REF_V2: &str = "magenta_rt_codec_dump/decoder_reference_v2.safetensors";
const TF_INT: &str = "magenta_rt_codec_dump/tf_intermediates.safetensors";

#[test]
#[ignore]
fn spectrostream_body_matches_tf() {
    if !Path::new(WEIGHTS).exists() {
        eprintln!("skip: {WEIGHTS} not found");
        return;
    }
    if !Path::new(REF_V2).exists() {
        eprintln!("skip: {REF_V2} not found — run decoder_reference_v2.py first");
        return;
    }
    if !Path::new(TF_INT).exists() {
        eprintln!("skip: {TF_INT} not found");
        return;
    }

    let model = SafeTensorsModel::load(WEIGHTS.into()).unwrap();
    let ref_v2 = SafeTensorsModel::load(REF_V2.into()).unwrap();
    let tf_int = SafeTensorsModel::load(TF_INT.into()).unwrap();
    let cfg = SpectroStreamConfig::default();
    let num_frames: u32 = 50;

    // v2's `preprocessed_input` is NHWC [B=1, T_pad=51, W=5, C=512].
    // meganeura expects NCHW [1, 512, 51, 5] in row-major contiguous order.
    let preprocessed_nhwc = ref_v2.tensor_f32_auto("preprocessed_input").unwrap();
    assert_eq!(preprocessed_nhwc.len(), (1 * 51 * 5 * 512) as usize,
               "preprocessed_input has wrong size");
    let preprocessed_nchw = nhwc_to_nchw_f32(&preprocessed_nhwc, 1, 51, 5, 512);

    let mut g = meganeura::Graph::new();
    let out = build_decoder_graph(&mut g, &cfg, num_frames);
    g.set_outputs(vec![out]);
    let mut s = build_inference_session(&g);
    load_decoder_weights(&model, &mut s).unwrap();
    s.set_input("decoder_input_preprocessed", &preprocessed_nchw);
    s.step();
    s.wait();

    // Meganeura output: [B=1, C=4, H=200, W=480] in NCHW row-major.
    let out_size = 1usize * 4 * 200 * 480;
    let our = s.read_output(out_size);

    // TF body_out: [B=1, T=200, W=480, C=4] in NHWC row-major.
    let tf_body = tf_int.tensor_f32_auto("body_out").unwrap();
    assert_eq!(tf_body.len(), out_size, "tf body_out has wrong size");

    // Transpose meganeura NCHW → NHWC for direct comparison.
    let ours_nhwc = nchw_to_nhwc_f32(&our, 1, 4, 200, 480);
    assert_eq!(ours_nhwc.len(), tf_body.len());

    // Count NaN/Inf for diagnosis (known RADV non-determinism produces NaN
    // in some late-dispatch positions; see commit a3c40e7).
    let total = our.len();
    let nans = our.iter().filter(|v| v.is_nan()).count();
    let infs = our.iter().filter(|v| !v.is_nan() && !v.is_finite()).count();
    let nan_frac = nans as f64 / total as f64;
    println!("NCHW output: {nans}/{total} NaN ({:.2}%), {infs} Inf", nan_frac * 100.0);

    stats("body_out", &ours_nhwc, &tf_body);

    // Pass criterion: among the FINITE positions, average relative error
    // < 1%. Architecture is verified bit-exact in NumPy ref; meganeura
    // should match within float32 + GPU dispatch ordering noise. NaN
    // positions are a separate Blade/RADV bug — tracked at commit a3c40e7.
    let (mean_diff, mean_ref) = mean_diff_and_ref_finite(&ours_nhwc, &tf_body);
    let rel = mean_diff / mean_ref;
    println!("finite rel err = {rel:.4e}  (NaN fraction: {:.2}%)", nan_frac * 100.0);
    assert!(rel < 0.01, "relative error {rel:.4e} > 1%");
    // Sanity: don't accept a graph that's mostly NaN.
    assert!(nan_frac < 0.1, "too many NaN positions ({nans}/{total})");
}

fn nhwc_to_nchw_f32(data: &[f32], n: u32, h: u32, w: u32, c: u32) -> Vec<f32> {
    let (n, h, w, c) = (n as usize, h as usize, w as usize, c as usize);
    let mut out = vec![0.0_f32; n * c * h * w];
    for bi in 0..n {
        for hi in 0..h {
            for wi in 0..w {
                for ci in 0..c {
                    let src = ((bi * h + hi) * w + wi) * c + ci;
                    let dst = ((bi * c + ci) * h + hi) * w + wi;
                    out[dst] = data[src];
                }
            }
        }
    }
    out
}

fn nchw_to_nhwc_f32(data: &[f32], n: u32, c: u32, h: u32, w: u32) -> Vec<f32> {
    let (n, c, h, w) = (n as usize, c as usize, h as usize, w as usize);
    let mut out = vec![0.0_f32; n * h * w * c];
    for bi in 0..n {
        for ci in 0..c {
            for hi in 0..h {
                for wi in 0..w {
                    let src = ((bi * c + ci) * h + hi) * w + wi;
                    let dst = ((bi * h + hi) * w + wi) * c + ci;
                    out[dst] = data[src];
                }
            }
        }
    }
    out
}

fn stats(label: &str, a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    let mut max_diff = 0.0_f32;
    let mut max_ref = 0.0_f32;
    let mut sum_diff = 0.0_f64;
    let mut sum_ref = 0.0_f64;
    let mut n_finite = 0usize;
    for (&x, &y) in a.iter().zip(b.iter()) {
        if !x.is_finite() || !y.is_finite() { continue; }
        let d = (x - y).abs();
        if d > max_diff { max_diff = d; }
        if y.abs() > max_ref { max_ref = y.abs(); }
        sum_diff += d as f64;
        sum_ref += y.abs() as f64;
        n_finite += 1;
    }
    let n = n_finite.max(1) as f64;
    println!(
        "[{label}] max_diff={max_diff:.3e} max_ref={max_ref:.3e} \
         mean_diff={:.3e} mean_ref={:.3e} rel={:.4e}  (n_finite={n_finite})",
        sum_diff / n,
        sum_ref / n,
        sum_diff / (sum_ref + 1e-20),
    );
}

fn mean_diff_and_ref_finite(a: &[f32], b: &[f32]) -> (f64, f64) {
    let mut sum_diff = 0.0_f64;
    let mut sum_ref = 0.0_f64;
    let mut n = 0usize;
    for (&x, &y) in a.iter().zip(b.iter()) {
        if !x.is_finite() || !y.is_finite() { continue; }
        sum_diff += (x - y).abs() as f64;
        sum_ref += y.abs() as f64;
        n += 1;
    }
    if n == 0 { return (0.0, 0.0); }
    (sum_diff / n as f64, sum_ref / n as f64)
}
