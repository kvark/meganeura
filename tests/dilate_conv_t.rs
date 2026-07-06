//! Verify that ConvTranspose2D == dilate_zeros_w(input) → forward Conv2D
//! (with kernel transposed in [Ci,Co] dims and spatially flipped).
//!
//! If this holds, SpectroStream can route its expensive `Conv2dGradInputHW`
//! dispatches through the fast cooperative-matrix `Conv2dGemm` path.

use meganeura::{Graph, build_inference_session};

/// Transform a conv-T kernel `[Ci_T, Co_T, kH, kW]` to the equivalent
/// forward-conv kernel `[Co_T, Ci_T, kH, kW]` with the spatial axes
/// flipped: `out[co, ci, kh, kw] = in[ci, co, kH-1-kh, kW-1-kw]`.
fn flip_and_transpose_conv_t_kernel(
    data: &[f32], ci_t: usize, co_t: usize, kh: usize, kw: usize,
) -> Vec<f32> {
    assert_eq!(data.len(), ci_t * co_t * kh * kw);
    let mut out = vec![0.0_f32; data.len()];
    for co in 0..co_t {
        for ci in 0..ci_t {
            for r in 0..kh {
                for c in 0..kw {
                    let src = ((ci * co_t + co) * kh + (kh - 1 - r)) * kw + (kw - 1 - c);
                    let dst = ((co * ci_t + ci) * kh + r) * kw + c;
                    out[dst] = data[src];
                }
            }
        }
    }
    out
}

/// Time the new dilate+forward-conv path at SpectroStream's d67 shape.
/// d67 conv-T: input [1, 128, 60, 480] → output [1, 128, 62, 962]
///   kernel (3, 4), stride (1, 2), padding (0, 0).
/// Chain N copies + slice2d between to keep the chain dimensions constant
/// (matches the failing pattern in blade_radv_repro).
#[test]
#[ignore]
fn dilate_chain_timing() {
    use std::time::Instant;
    let n_chain = std::env::var("N").ok().and_then(|s| s.parse().ok()).unwrap_or(6usize);
    let batch = 1u32;
    let in_c = 128u32;
    let out_c = 128u32;
    let in_h = 60u32;
    let in_w = 480u32;
    let kh = 3u32;
    let kw = 4u32;
    let stride_w = 2u32;
    let pad_h = 0u32;
    let pad_w = 0u32;

    let in_size = (batch * in_c * in_h * in_w) as usize;
    let k_size = (in_c * out_c * kh * kw) as usize;

    let mut g = Graph::new();
    let mut x = g.input("x", &[in_size]);
    let mut param_names = Vec::new();
    for i in 0..n_chain {
        let k = g.parameter(&format!("k{i}"), &[k_size]);
        param_names.push(format!("k{i}"));
        // dilate W: 480 → 959
        let dilated = g.dilate_zeros_w(x, batch, in_c, in_h, in_w, stride_w);
        let dil_w = in_w * stride_w - (stride_w - 1);
        // forward conv2d: 60x959 padded to 64x965, kernel 3x4 → output 62x962
        let big = g.conv2d_hw(
            dilated, k,
            batch, in_c, in_h, dil_w,
            out_c, kh, kw,
            1, // stride
            kh - 1 - pad_h, kw - 1 - pad_w,
        );
        // slice2d: 62x962 → 60x480 (start_h=1, end_h=1, start_w=1, end_w=481)
        let cropped = g.slice_2d(big, batch, in_c, 62, 962, 1, 1, 1, 481);
        x = cropped;
    }
    g.set_outputs(vec![x]);

    let mut s = build_inference_session(&g);
    let input: Vec<f32> = (0..in_size).map(|i| (i as f32 * 0.001).sin() + 1.0).collect();
    s.set_input("x", &input);
    let k_data: Vec<f32> = (0..k_size).map(|i| (i as f32 * 0.01).cos() * 0.05).collect();
    for name in &param_names { s.set_parameter(name, &k_data); }

    let t0 = Instant::now();
    s.step();
    s.wait();
    let elapsed = t0.elapsed();
    let out = s.read_output(in_size);
    let nz = out.iter().filter(|&&v| v != 0.0 && v.is_finite()).count();
    let max_abs = out.iter().fold(0.0_f32, |a, &v| a.max(v.abs()));
    println!("[dilate-chain] N={n_chain} wall={:.3}s nz={nz}/{in_size} ({:.1}%) max_abs={max_abs:.3e}",
             elapsed.as_secs_f64(), 100.0 * nz as f32 / in_size as f32);
}

#[test]
#[ignore]
fn dilate_plus_conv2d_matches_conv_transpose() {
    let batch = 1u32;
    let in_c = 4u32;
    let out_c = 8u32;
    let in_h = 5u32;
    let in_w = 6u32;
    let kh = 3u32;
    let kw = 4u32;
    let stride_h = 1u32;
    let stride_w = 2u32;
    let pad_h = 0u32;
    let pad_w = 0u32;

    // Shared input + kernel data.
    let in_size = (batch * in_c * in_h * in_w) as usize;
    let in_data: Vec<f32> = (0..in_size)
        .map(|i| (i as f32 * 0.13).sin() + 0.5)
        .collect();
    let k_size = (in_c * out_c * kh * kw) as usize;
    let k_data: Vec<f32> = (0..k_size)
        .map(|i| (i as f32 * 0.07).cos() * 0.3)
        .collect();

    // --- Reference: existing conv_transpose_2d_hw path ---
    let ref_out = {
        let mut g = Graph::new();
        let x = g.input("x", &[in_size]);
        let k = g.parameter("k", &[k_size]);
        let y = g.conv_transpose_2d_hw(
            x, k, batch, in_c, in_h, in_w, out_c, kh, kw, stride_h, stride_w, pad_h, pad_w,
        );
        let out_h = (in_h - 1) * stride_h + kh - 2 * pad_h;
        let out_w = (in_w - 1) * stride_w + kw - 2 * pad_w;
        let out_size = (batch * out_c * out_h * out_w) as usize;
        g.set_outputs(vec![y]);
        let mut s = build_inference_session(&g);
        s.set_input("x", &in_data);
        s.set_parameter("k", &k_data);
        s.step();
        s.wait();
        s.read_output(out_size)
    };

    // --- New path: dilate → forward conv2d with flipped/transposed kernel ---
    let new_out = {
        let mut g = Graph::new();
        let x = g.input("x", &[in_size]);
        let k_fwd_data = flip_and_transpose_conv_t_kernel(
            &k_data, in_c as usize, out_c as usize, kh as usize, kw as usize,
        );
        let k = g.parameter("k", &[k_size]);
        let dilated = g.dilate_zeros_w(x, batch, in_c, in_h, in_w, stride_w);
        let dil_w = if stride_w == 1 { in_w } else { in_w * stride_w - (stride_w - 1) };
        // Forward conv2d: stride 1, padding = (kh-1-pad_h, kw-1-pad_w).
        let fwd_pad_h = kh - 1 - pad_h;
        let fwd_pad_w = kw - 1 - pad_w;
        let y = g.conv2d_hw(
            dilated, k,
            batch, in_c, in_h, dil_w,
            out_c, kh, kw,
            1, // stride
            fwd_pad_h, fwd_pad_w,
        );
        let out_h = in_h + 2 * fwd_pad_h - kh + 1;
        let out_w = dil_w + 2 * fwd_pad_w - kw + 1;
        let out_size = (batch * out_c * out_h * out_w) as usize;
        g.set_outputs(vec![y]);
        let mut s = build_inference_session(&g);
        s.set_input("x", &in_data);
        s.set_parameter("k", &k_fwd_data);
        s.step();
        s.wait();
        s.read_output(out_size)
    };

    assert_eq!(ref_out.len(), new_out.len(),
        "output sizes differ: ref={} new={}", ref_out.len(), new_out.len());
    let mut max_diff = 0.0_f32;
    let mut max_ref = 0.0_f32;
    for (i, (&r, &n)) in ref_out.iter().zip(new_out.iter()).enumerate() {
        let d = (r - n).abs();
        if d > max_diff {
            max_diff = d;
            eprintln!("  largest mismatch at i={i}: ref={r:.6}  new={n:.6}  diff={d:.6}");
        }
        max_ref = max_ref.max(r.abs());
    }
    println!("dilate-vs-convT max_diff={max_diff:.6e}  max_ref={max_ref:.6e}");
    assert!(max_diff < 1e-4, "mismatch: max_diff={max_diff} max_ref={max_ref}");
}
