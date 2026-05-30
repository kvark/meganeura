//! Tests for `Graph::conv_transpose_2d`. Implemented as a thin wrapper over
//! `conv2d_grad_input`, so these tests prove the wrapper signs/sizes things
//! correctly and matches the standard ConvTranspose formula.

use meganeura::{Graph, build_inference_session};

/// Reference: pure-Rust ConvTranspose forward for verification.
/// `input` is [N, C_in, H_in, W_in] flat NCHW.
/// `kernel` is [C_in, C_out, kH, kW] flat (PyTorch ConvTranspose layout).
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
    stride: usize,
    pad_h: usize,
    pad_w: usize,
) -> Vec<f32> {
    let out_h = (in_h - 1) * stride + kh - 2 * pad_h;
    let out_w = (in_w - 1) * stride + kw - 2 * pad_w;
    let mut out = vec![0.0_f32; batch * out_c * out_h * out_w];
    for n in 0..batch {
        for co in 0..out_c {
            for oy in 0..out_h {
                for ox in 0..out_w {
                    let mut acc = 0.0_f32;
                    for ci in 0..in_c {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                // ConvTranspose: output(oy,ox) gets contribution from
                                // input(iy, ix) where oy + pad_h == iy*stride + ky.
                                let iy_s = oy + pad_h;
                                let ix_s = ox + pad_w;
                                if iy_s < ky || ix_s < kx { continue; }
                                let iy = iy_s - ky;
                                let ix = ix_s - kx;
                                if iy % stride != 0 || ix % stride != 0 { continue; }
                                let iy = iy / stride;
                                let ix = ix / stride;
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

fn run_conv_transpose_gpu(
    input_data: &[f32],
    kernel_data: &[f32],
    batch: u32,
    in_c: u32,
    in_h: u32,
    in_w: u32,
    out_c: u32,
    kh: u32,
    kw: u32,
    stride: u32,
    pad_h: u32,
    pad_w: u32,
) -> Vec<f32> {
    let out_h = (in_h - 1) * stride + kh - 2 * pad_h;
    let out_w = (in_w - 1) * stride + kw - 2 * pad_w;
    let in_size = (batch * in_c * in_h * in_w) as usize;
    let k_size = (out_c * in_c * kh * kw) as usize;
    let out_size = (batch * out_c * out_h * out_w) as usize;
    assert_eq!(input_data.len(), in_size);
    assert_eq!(kernel_data.len(), k_size);

    let mut g = Graph::new();
    let input = g.input("input", &[in_size]);
    let kernel = g.parameter("kernel", &[k_size]);
    let y = g.conv_transpose_2d(
        input, kernel, batch, in_c, in_h, in_w, out_c, kh, kw, stride, pad_h, pad_w,
    );
    g.set_outputs(vec![y]);

    let mut s = build_inference_session(&g);
    s.set_input("input", input_data);
    s.set_parameter("kernel", kernel_data);
    s.step();
    s.wait();
    s.read_output(out_size)
}

#[test]
fn conv_transpose_1d_stride2_basic() {
    // Conv1d-as-Conv2d with W=1. 3 input frames upsampled 2x with kernel size 2.
    let input = vec![1.0f32, 2.0, 3.0];
    let kernel = vec![10.0f32, 20.0]; // (C_out=1, C_in=1, kH=2, kW=1)
    let got = run_conv_transpose_gpu(&input, &kernel, 1, 1, 3, 1, 1, 2, 1, 2, 0, 0);
    // Hand-computed: each input scattered with kernel at i*stride
    //   i=0: [10, 20, .., .., .., ..]
    //   i=1: [.., .., 20, 40, .., ..]
    //   i=2: [.., .., .., .., 30, 60]
    let expected = vec![10.0f32, 20.0, 20.0, 40.0, 30.0, 60.0];
    assert_eq!(got, expected, "ConvT1d basic stride-2 mismatch");
}

#[test]
fn conv_transpose_1d_stride4_kernel8() {
    // Closer to SoundStream decoder shapes: stride=4, kernel=8.
    // 4 input frames, 2 channels, single output channel.
    let batch = 1u32;
    let in_c = 2u32;
    let in_h = 4u32;
    let in_w = 1u32;
    let out_c = 1u32;
    let kh = 8u32;
    let kw = 1u32;
    let stride = 4u32;
    let pad_h = 2u32;
    let pad_w = 0u32;
    let in_size = (batch * in_c * in_h * in_w) as usize;
    let k_size = (out_c * in_c * kh * kw) as usize;
    let input: Vec<f32> = (0..in_size).map(|i| (i as f32) * 0.1 - 0.5).collect();
    let kernel: Vec<f32> = (0..k_size).map(|i| (i as f32) * 0.05).collect();
    let got = run_conv_transpose_gpu(
        &input, &kernel, batch, in_c, in_h, in_w, out_c, kh, kw, stride, pad_h, pad_w,
    );
    let expected = conv_transpose_ref(
        &input, &kernel,
        batch as usize, in_c as usize, in_h as usize, in_w as usize,
        out_c as usize, kh as usize, kw as usize, stride as usize,
        pad_h as usize, pad_w as usize,
    );
    assert_eq!(got.len(), expected.len(), "length mismatch");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let tol = e.abs() * 1e-5 + 1e-6;
        assert!(
            (g - e).abs() <= tol,
            "ConvT1d stride-4 mismatch at {i}: got {g}, expected {e}",
        );
    }
}

#[test]
fn conv_transpose_2d_stride2_multichannel() {
    // Full 2D case: small image, 3 in channels → 2 out channels.
    let batch = 1u32;
    let in_c = 3u32;
    let in_h = 2u32;
    let in_w = 2u32;
    let out_c = 2u32;
    let kh = 3u32;
    let kw = 3u32;
    let stride = 2u32;
    let pad_h = 1u32;
    let pad_w = 1u32;
    let in_size = (batch * in_c * in_h * in_w) as usize;
    let k_size = (out_c * in_c * kh * kw) as usize;
    let input: Vec<f32> = (0..in_size).map(|i| (i as f32) * 0.07 - 0.3).collect();
    let kernel: Vec<f32> = (0..k_size).map(|i| (i as f32) * 0.03 - 0.4).collect();
    let got = run_conv_transpose_gpu(
        &input, &kernel, batch, in_c, in_h, in_w, out_c, kh, kw, stride, pad_h, pad_w,
    );
    let expected = conv_transpose_ref(
        &input, &kernel,
        batch as usize, in_c as usize, in_h as usize, in_w as usize,
        out_c as usize, kh as usize, kw as usize, stride as usize,
        pad_h as usize, pad_w as usize,
    );
    assert_eq!(got.len(), expected.len(), "length mismatch");
    for (i, (g, e)) in got.iter().zip(expected.iter()).enumerate() {
        let tol = e.abs() * 1e-5 + 1e-6;
        assert!(
            (g - e).abs() <= tol,
            "ConvT2d stride-2 mismatch at {i}: got {g}, expected {e}",
        );
    }
}
