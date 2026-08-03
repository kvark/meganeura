//! CPU-reference parity for the inference-only fusion paths at
//! production-like sizes. Training sessions never use WinogradConv2d or
//! GroupNormSilu (autodiff rejects them), and the existing fusion tests
//! run at small sizes — so a size-dependent bug here produces healthy
//! training but garbage sampling, which is what van-world SR sampling
//! shows at step 10k-20k.

use meganeura::Graph;

fn ramp(n: usize, scale: f32, offset: f32) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 37 + 11) % 19) as f32 / 9.0 - 1.0)
        .map(|v| v * scale + offset)
        .collect()
}

fn max_diff(got: &[f32], want: &[f32]) -> f32 {
    got.iter()
        .zip(want)
        .map(|(g, w)| (g - w).abs())
        .fold(0.0f32, f32::max)
}

/// 3x3 stride-1 conv at 128x128, C=64 — inference mode rewrites this to
/// WinogradConv2d. Compare against a naive CPU convolution.
fn check_conv3x3(c_in: u32, c_out: u32, hw: u32) {
    let in_size = (c_in * hw * hw) as usize;
    let out_size = (c_out * hw * hw) as usize;
    let k_size = (c_out * c_in * 9) as usize;

    let mut g = Graph::new();
    let x = g.input("x", &[in_size]);
    let k = g.parameter("k", &[k_size]);
    let y = g.conv2d(x, k, 1, c_in, hw, hw, c_out, 3, 3, 1, 1);
    g.set_outputs(vec![y]);

    let mut session = meganeura::build(&g, meganeura::SessionConfig::inference_from_env()).0;
    let x_data = ramp(in_size, 0.7, 0.1);
    let k_data = ramp(k_size, 0.2, 0.0);
    session.set_parameter("k", &k_data);
    session.set_input("x", &x_data);
    session.step();
    session.wait();
    let got = session.read_output(out_size);

    let (h, w) = (hw as usize, hw as usize);
    let mut want = vec![0.0f32; out_size];
    for co in 0..c_out as usize {
        for oy in 0..h {
            for ox in 0..w {
                let mut s = 0.0f32;
                for ci in 0..c_in as usize {
                    for ky in 0..3usize {
                        for kx in 0..3usize {
                            let iy = oy as isize + ky as isize - 1;
                            let ix = ox as isize + kx as isize - 1;
                            if iy >= 0 && iy < h as isize && ix >= 0 && ix < w as isize {
                                s += x_data[(ci * h + iy as usize) * w + ix as usize]
                                    * k_data[((co * c_in as usize + ci) * 3 + ky) * 3 + kx];
                            }
                        }
                    }
                }
                want[(co * h + oy) * w + ox] = s;
            }
        }
    }
    let d = max_diff(&got, &want);
    assert!(
        d < 1e-2,
        "conv3x3 Cin={c_in} Cout={c_out} {hw}x{hw}: max_diff={d:.4}, got[..4]={:?} want[..4]={:?}",
        &got[..4],
        &want[..4]
    );
}

#[test]
fn winograd_conv_parity_small() {
    check_conv3x3(4, 4, 16);
}

#[test]
fn winograd_conv_parity_sr_scale() {
    check_conv3x3(64, 64, 128);
}

/// group_norm -> silu at C=64, spatial 128x128, 16 groups (the SR model
/// shape) — inference mode fuses this into GroupNormSilu.
#[test]
fn group_norm_silu_parity_sr_scale() {
    let (c, hw, groups) = (64u32, 128u32, 16u32);
    let spatial = (hw * hw) as usize;
    let n = c as usize * spatial;

    let mut g = Graph::new();
    let x = g.input("x", &[n]);
    let w = g.parameter("w", &[c as usize]);
    let b = g.parameter("b", &[c as usize]);
    let y = g.group_norm(x, w, b, 1, c, hw * hw, groups, 1e-5);
    let y = g.silu(y);
    g.set_outputs(vec![y]);

    let mut session = meganeura::build(&g, meganeura::SessionConfig::inference_from_env()).0;
    let x_data = ramp(n, 1.0, 0.4);
    let w_data = ramp(c as usize, 0.2, 1.0);
    let b_data = ramp(c as usize, 0.2, 0.0);
    session.set_parameter("w", &w_data);
    session.set_parameter("b", &b_data);
    session.set_input("x", &x_data);
    session.step();
    session.wait();
    let got = session.read_output(n);

    let cpg = (c / groups) as usize;
    let group_size = cpg * spatial;
    let mut want = vec![0.0f32; n];
    for grp in 0..groups as usize {
        let start = grp * group_size;
        let xs = &x_data[start..start + group_size];
        let mean = xs.iter().sum::<f32>() / group_size as f32;
        let var = xs.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / group_size as f32;
        let inv_std = 1.0 / (var + 1e-5).sqrt();
        for j in 0..group_size {
            let ch = grp * cpg + j / spatial;
            let z = (xs[j] - mean) * inv_std * w_data[ch] + b_data[ch];
            want[start + j] = z * (1.0 / (1.0 + (-z).exp()));
        }
    }
    let d = max_diff(&got, &want);
    assert!(
        d < 1e-2,
        "group_norm_silu C={c} {hw}x{hw}: max_diff={d:.4}, got[..4]={:?} want[..4]={:?}",
        &got[..4],
        &want[..4]
    );
}

/// 1x1 stride-1 conv — the compile shortcut views NCHW input as
/// [batch*H*W, Ci] row-major, but NCHW flat data is [Ci, H*W] per
/// image: the matmul would mix VALUES ACROSS PIXELS if that view is
/// wrong. Compare against the CPU channel-mix reference.
#[test]
fn conv1x1_layout_parity() {
    let (c_in, c_out, hw) = (4u32, 3u32, 8u32);
    let spatial = (hw * hw) as usize;
    let in_size = c_in as usize * spatial;
    let out_size = c_out as usize * spatial;
    let k_size = (c_out * c_in) as usize;

    let mut g = Graph::new();
    let x = g.input("x", &[in_size]);
    let k = g.parameter("k", &[k_size]);
    let y = g.conv2d(x, k, 1, c_in, hw, hw, c_out, 1, 1, 1, 0);
    g.set_outputs(vec![y]);

    let mut session = meganeura::build(&g, meganeura::SessionConfig::inference_from_env()).0;
    let x_data = ramp(in_size, 0.7, 0.1);
    let k_data = ramp(k_size, 0.4, 0.1);
    session.set_parameter("k", &k_data);
    session.set_input("x", &x_data);
    session.step();
    session.wait();
    let got = session.read_output(out_size);

    // NCHW: out[co][p] = sum_ci k[co][ci] * x[ci][p]
    let mut want = vec![0.0f32; out_size];
    for co in 0..c_out as usize {
        for p in 0..spatial {
            let mut s = 0.0;
            for ci in 0..c_in as usize {
                s += k_data[co * c_in as usize + ci] * x_data[ci * spatial + p];
            }
            want[co * spatial + p] = s;
        }
    }
    let d = max_diff(&got, &want);
    assert!(
        d < 1e-3,
        "conv1x1 layout mismatch: max_diff={d:.4}, got[..6]={:?} want[..6]={:?}",
        &got[..6],
        &want[..6]
    );
}

/// Whisper's Conv1d emulation uses H×1 tensors. Cover a spatial length
/// larger than one GEMM tile so reducing the dispatch from H workgroups to
/// ceil(H/tile) cannot accidentally drop valid output rows.
#[test]
fn conv3x1_flat_spatial_dispatch_parity() {
    let (c_in, c_out, h) = (3usize, 5usize, 129usize);
    let in_size = c_in * h;
    let out_size = c_out * h;

    let mut g = Graph::new();
    let x = g.input("x", &[in_size]);
    let k = g.parameter("k", &[c_out * c_in * 3]);
    let y = g.conv2d_hw(
        x,
        k,
        1,
        c_in as u32,
        h as u32,
        1,
        c_out as u32,
        3,
        1,
        1,
        1,
        0,
    );
    g.set_outputs(vec![y]);

    let mut session = meganeura::build(&g, meganeura::SessionConfig::inference_from_env()).0;
    let x_data = ramp(in_size, 0.7, 0.1);
    let k_data = ramp(c_out * c_in * 3, 0.2, 0.0);
    session.set_parameter("k", &k_data);
    session.set_input("x", &x_data);
    session.step();
    session.wait();
    let got = session.read_output(out_size);

    let mut want = vec![0.0; out_size];
    for co in 0..c_out {
        for oh in 0..h {
            let mut sum = 0.0;
            for ci in 0..c_in {
                for kh in 0..3 {
                    let ih = oh as isize + kh as isize - 1;
                    if (0..h as isize).contains(&ih) {
                        sum += x_data[ci * h + ih as usize] * k_data[(co * c_in + ci) * 3 + kh];
                    }
                }
            }
            want[co * h + oh] = sum;
        }
    }

    let d = max_diff(&got, &want);
    assert!(d < 1e-3, "conv3x1 H×1 parity mismatch: max_diff={d:.4}");
}
