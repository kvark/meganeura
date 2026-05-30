//! Tests for `Graph::conv_transpose_2d_hw` (separate stride_h, stride_w).
//! Mirrors tests/conv_transpose.rs but for the asymmetric-stride variant.

use meganeura::{Graph, build_inference_session};

/// Reference: pure-Rust ConvTranspose forward.
/// kernel layout: [C_in, C_out, kH, kW] (PyTorch).
#[allow(clippy::too_many_arguments)]
fn conv_transpose_ref(
    input: &[f32],
    kernel: &[f32],
    batch: usize,
    in_c: usize,
    in_h: usize,
    in_w: usize,
    out_c: usize,
    kh: usize,
    kw: usize,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> Vec<f32> {
    let out_h = (in_h - 1) * stride_h + kh - 2 * pad_h;
    let out_w = (in_w - 1) * stride_w + kw - 2 * pad_w;
    let mut out = vec![0.0_f32; batch * out_c * out_h * out_w];
    for n in 0..batch {
        for co in 0..out_c {
            for oy in 0..out_h {
                for ox in 0..out_w {
                    let mut acc = 0.0_f32;
                    for ci in 0..in_c {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let iy_s = oy + pad_h;
                                let ix_s = ox + pad_w;
                                if iy_s < ky || ix_s < kx { continue; }
                                let iy = iy_s - ky;
                                let ix = ix_s - kx;
                                if iy % stride_h != 0 || ix % stride_w != 0 { continue; }
                                let iy = iy / stride_h;
                                let ix = ix / stride_w;
                                if iy >= in_h || ix >= in_w { continue; }
                                let in_idx = ((n * in_c + ci) * in_h + iy) * in_w + ix;
                                let k_idx = ((ci * out_c + co) * kh + ky) * kw + kx;
                                acc += input[in_idx] * kernel[k_idx];
                            }
                        }
                    }
                    let o_idx = ((n * out_c + co) * out_h + oy) * out_w + ox;
                    out[o_idx] = acc;
                }
            }
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn run_gpu(
    input: &[f32],
    kernel: &[f32],
    batch: u32,
    in_c: u32,
    in_h: u32,
    in_w: u32,
    out_c: u32,
    kh: u32,
    kw: u32,
    stride_h: u32,
    stride_w: u32,
) -> Vec<f32> {
    let out_h = (in_h - 1) * stride_h + kh;
    let out_w = (in_w - 1) * stride_w + kw;
    let in_size = (batch * in_c * in_h * in_w) as usize;
    let k_size = (in_c * out_c * kh * kw) as usize;
    let out_size = (batch * out_c * out_h * out_w) as usize;

    let mut g = Graph::new();
    let inp = g.input("input", &[in_size]);
    let ker = g.parameter("kernel", &[k_size]);
    let y = g.conv_transpose_2d_hw(
        inp, ker, batch, in_c, in_h, in_w, out_c, kh, kw, stride_h, stride_w, 0, 0,
    );
    g.set_outputs(vec![y]);

    let mut s = build_inference_session(&g);
    s.set_input("input", input);
    s.set_parameter("kernel", kernel);
    s.step();
    s.wait();
    s.read_output(out_size)
}

#[test]
fn conv_transpose_hw_stride_h1_w2_matches_reference() {
    // SpectroStream decoder_0 shape: kernel 4x3, stride (1, 2). Tiny batch
    // 1 ch in, 2 ch out, [1, 1, 3, 3] input.
    let in_h = 3u32; let in_w = 3u32;
    let in_c = 1u32; let out_c = 2u32;
    let kh = 4u32; let kw = 3u32;
    let stride_h = 1u32; let stride_w = 2u32;
    let input: Vec<f32> = (0..(in_c * in_h * in_w) as usize).map(|i| (i as f32) * 0.1).collect();
    let kernel: Vec<f32> = (0..(in_c * out_c * kh * kw) as usize).map(|i| (i as f32) * 0.05 - 0.4).collect();
    let got = run_gpu(&input, &kernel, 1, in_c, in_h, in_w, out_c, kh, kw, stride_h, stride_w);
    let want = conv_transpose_ref(
        &input, &kernel, 1, in_c as usize, in_h as usize, in_w as usize,
        out_c as usize, kh as usize, kw as usize, stride_h as usize, stride_w as usize, 0, 0,
    );
    assert_eq!(got.len(), want.len(), "len mismatch");
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let tol = w.abs() * 1e-5 + 1e-6;
        assert!((g - w).abs() <= tol, "mismatch at {i}: got {g}, expected {w}");
    }
}

#[test]
fn conv_transpose_hw_stride_h1_w3_matches_reference() {
    // SpectroStream decoder_4 shape: kernel 3x6, stride (1, 3).
    let in_h = 2u32; let in_w = 2u32;
    let in_c = 2u32; let out_c = 1u32;
    let kh = 3u32; let kw = 6u32;
    let stride_h = 1u32; let stride_w = 3u32;
    let input: Vec<f32> = (0..(in_c * in_h * in_w) as usize).map(|i| (i as f32) * 0.07 - 0.2).collect();
    let kernel: Vec<f32> = (0..(in_c * out_c * kh * kw) as usize).map(|i| (i as f32) * 0.03 - 0.5).collect();
    let got = run_gpu(&input, &kernel, 1, in_c, in_h, in_w, out_c, kh, kw, stride_h, stride_w);
    let want = conv_transpose_ref(
        &input, &kernel, 1, in_c as usize, in_h as usize, in_w as usize,
        out_c as usize, kh as usize, kw as usize, stride_h as usize, stride_w as usize, 0, 0,
    );
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        let tol = w.abs() * 1e-5 + 1e-6;
        assert!((g - w).abs() <= tol, "mismatch at {i}: got {g}, expected {w}");
    }
}

#[test]
fn conv_transpose_hw_equal_strides_matches_symmetric() {
    // With stride_h == stride_w, the HW variant should match the symmetric
    // conv_transpose_2d (regression check for the new shader).
    let in_h = 2u32; let in_w = 3u32;
    let in_c = 2u32; let out_c = 2u32;
    let kh = 3u32; let kw = 3u32;
    let stride = 2u32;
    let input: Vec<f32> = (0..(in_c * in_h * in_w) as usize).map(|i| (i as f32) * 0.1).collect();
    let kernel: Vec<f32> = (0..(in_c * out_c * kh * kw) as usize).map(|i| (i as f32) * 0.02 - 0.3).collect();

    let from_hw = run_gpu(&input, &kernel, 1, in_c, in_h, in_w, out_c, kh, kw, stride, stride);

    // Compare with conv_transpose_2d (single stride).
    let in_size = (in_c * in_h * in_w) as usize;
    let k_size = (in_c * out_c * kh * kw) as usize;
    let out_h = (in_h - 1) * stride + kh;
    let out_w = (in_w - 1) * stride + kw;
    let out_size = (out_c * out_h * out_w) as usize;
    let mut g = Graph::new();
    let inp = g.input("input", &[in_size]);
    let ker = g.parameter("kernel", &[k_size]);
    let y = g.conv_transpose_2d(inp, ker, 1, in_c, in_h, in_w, out_c, kh, kw, stride, 0, 0);
    g.set_outputs(vec![y]);
    let mut s = build_inference_session(&g);
    s.set_input("input", &input);
    s.set_parameter("kernel", &kernel);
    s.step();
    s.wait();
    let from_symmetric = s.read_output(out_size);

    assert_eq!(from_hw.len(), from_symmetric.len());
    for (i, (a, b)) in from_hw.iter().zip(from_symmetric.iter()).enumerate() {
        let tol = b.abs() * 1e-5 + 1e-6;
        assert!((a - b).abs() <= tol, "HW vs symmetric mismatch at {i}: {a} vs {b}");
    }
}
