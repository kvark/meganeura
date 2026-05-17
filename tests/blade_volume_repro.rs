//! End-to-end regression test for blade-volume-train's volumetric forward.
//!
//! Mirrors the graph shape (for SH degree 0):
//!
//!   inputs:
//!     cell_indices : u32 [P*L]
//!     dt           : f32 [P*L]
//!     mask         : f32 [P*L]
//!     labels       : f32 [1, P*3]
//!   parameters:
//!     log_density, sh_r, sh_g, sh_b : f32 [N, 1]
//!   forward:
//!     density = relu(embedding(cell_indices, log_density))  [P*L, 1]
//!     reshape → [P, L]
//!     raw     = density * reshape(dt, [P, L]) * reshape(mask, [P, L])
//!     cumsum  = raw @ strict_lower_triangular(L,L)          [P, L]
//!     T       = recip(sigmoid(cumsum)) + neg(ones)
//!     alpha   = 2*ones + neg(recip(sigmoid(raw)))
//!     weight  = T * alpha * mask
//!     per-channel pixel = matmul(weight * (SH_C0 * embed(sh_c) + 0.5), ones_L1)
//!     loss = l1_loss(pixel_r, target_r) + ... + ...
//!
//! Originally [P=1024, L=16] returned NaN because `fuse_epilogues` rewrote
//! the cumsum-matmul's output buffer to fold sigmoid in as an epilogue,
//! then the runtime's coop-matrix auto-promotion further rewrote
//! `workgroups` to coop geometry — but `Pipelines::get` keeps using the
//! scalar epilogue pipeline, so every output row past 64 silently zeroed.
//! Recip then saw `sigmoid(0)=0.5` → `recip` → 2.0 for those rows… wait,
//! the bug was tighter: only the first `M=64` rows of the [P, L] tile were
//! written at all; the rest stayed uninitialised, sigmoid-of-garbage was
//! ~0, recip blew up to inf. Fixed by skipping coop promotion when a
//! matmul carries an epilogue.

use meganeura::{build, Graph, Mode, SessionConfig};

fn strict_lower_triangular(l: usize) -> Vec<f32> {
    let mut m = vec![0.0_f32; l * l];
    for i in 0..l {
        for k in (i + 1)..l {
            m[i * l + k] = 1.0;
        }
    }
    m
}

/// Stage at which to truncate the graph for inspection.
#[derive(Clone, Copy)]
enum Stage {
    Cumsum,
    SigCum,
    RecSigCum,
    Transmittance,
    Alpha,
    Weight,
    Pixel,
}

/// Build the same forward as blade-volume-train::diff_render::build_volumetric_graph
/// and run one inference step with hand-crafted finite inputs. Asserts the
/// output is finite.
fn forward_finite_smoke(p: usize, l: usize, n_cells: usize) {
    forward_at_stage(p, l, n_cells, Stage::Pixel);
}

fn forward_at_stage(p: usize, l: usize, n_cells: usize, stage: Stage) {
    let pl = p * l;
    let mut g = Graph::new();
    let cell_indices = g.input_u32("cell_indices", &[pl]);
    let dt = g.input("dt", &[pl]);
    let mask = g.input("mask", &[pl]);
    let _target = g.input("labels", &[1, p * 3]);
    let log_density = g.parameter("log_density", &[n_cells, 1]);
    let sh_r = g.parameter("sh_r", &[n_cells, 1]);
    let _sh_g = g.parameter("sh_g", &[n_cells, 1]);
    let _sh_b = g.parameter("sh_b", &[n_cells, 1]);

    let density_flat = g.embedding(cell_indices, log_density);
    let density_flat = g.relu(density_flat);
    let density = g.reshape(density_flat, &[p, l]);
    let dt_2d = g.reshape(dt, &[p, l]);
    let mask_2d = g.reshape(mask, &[p, l]);

    let raw = g.mul(density, dt_2d);
    let raw_masked = g.mul(raw, mask_2d);

    let cum_data = strict_lower_triangular(l);
    let cum_matrix = g.constant(cum_data, &[l, l]);
    let cumsum = g.matmul(raw_masked, cum_matrix);

    let ones_pl = g.constant(vec![1.0; pl], &[p, l]);
    let twos_pl = g.constant(vec![2.0; pl], &[p, l]);
    let sig_cum = g.sigmoid(cumsum);
    let rec_sig_cum = g.recip(sig_cum);
    let neg_ones_pl = g.neg(ones_pl);
    let t = g.add(rec_sig_cum, neg_ones_pl);

    let sig_raw = g.sigmoid(raw_masked);
    let rec_sig_raw = g.recip(sig_raw);
    let neg_rec_sig_raw = g.neg(rec_sig_raw);
    let alpha = g.add(twos_pl, neg_rec_sig_raw);

    let weight = g.mul(t, alpha);
    let weight = g.mul(weight, mask_2d);

    // Channel pipeline (just R; G/B identical).
    let scale = g.constant(vec![0.282_094_8_f32; pl], &[p, l]);
    let bias = g.constant(vec![0.5_f32; pl], &[p, l]);
    let color_flat = g.embedding(cell_indices, sh_r);
    let color = g.reshape(color_flat, &[p, l]);
    let scaled = g.mul(color, scale);
    let biased = g.add(scaled, bias);
    let weighted = g.mul(weight, biased);
    let ones_l1 = g.constant(vec![1.0_f32; l], &[l, 1]);
    let pixel = g.matmul(weighted, ones_l1);

    let output = match stage {
        Stage::Cumsum => cumsum,
        Stage::SigCum => sig_cum,
        Stage::RecSigCum => rec_sig_cum,
        Stage::Transmittance => t,
        Stage::Alpha => alpha,
        Stage::Weight => weight,
        Stage::Pixel => pixel,
    };
    g.set_outputs(vec![output]);

    let (mut session, _) = build(
        &g,
        SessionConfig {
            mode: Mode::Inference,
            ..Default::default()
        },
    );

    // Hand-crafted finite inputs.
    let mut cell_buf = vec![0u32; pl];
    for (i, slot) in cell_buf.iter_mut().enumerate() {
        *slot = (i as u32) % (n_cells as u32);
    }
    let dt_buf = vec![0.5_f32; pl];
    let mask_buf = vec![1.0_f32; pl];
    let labels_buf = vec![0.0_f32; p * 3];

    session.set_input_u32("cell_indices", &cell_buf);
    session.set_input("dt", &dt_buf);
    session.set_input("mask", &mask_buf);
    session.set_input("labels", &labels_buf);
    session.set_parameter("log_density", &vec![1.0_f32; n_cells]);
    session.set_parameter("sh_r", &vec![0.0_f32; n_cells]);
    session.set_parameter("sh_g", &vec![0.0_f32; n_cells]);
    session.set_parameter("sh_b", &vec![0.0_f32; n_cells]);

    session.step();
    session.wait();

    let out_len = match stage {
        Stage::Pixel => p,
        _ => p * l,
    };
    let mut out = vec![0.0_f32; out_len];
    session.read_output_by_index(0, &mut out);

    let n_nan = out.iter().filter(|x| x.is_nan()).count();
    let n_inf = out.iter().filter(|x| x.is_infinite()).count();
    let stage_name = match stage {
        Stage::Cumsum => "cumsum",
        Stage::SigCum => "sig_cum",
        Stage::RecSigCum => "rec_sig_cum",
        Stage::Transmittance => "T",
        Stage::Alpha => "alpha",
        Stage::Weight => "weight",
        Stage::Pixel => "pixel",
    };
    assert_eq!(n_nan, 0, "P={p} L={l} stage={stage_name}: NaN");
    assert_eq!(n_inf, 0, "P={p} L={l} stage={stage_name}: inf");
}

#[test]
fn blade_volume_p256_l32_works() {
    forward_finite_smoke(256, 32, 2000);
}

#[test]
fn blade_volume_p576_l24_works() {
    forward_finite_smoke(576, 24, 2000);
}

#[test]
fn blade_volume_p784_l16() {
    forward_finite_smoke(784, 16, 2000);
}

#[test]
fn blade_volume_p1024_l16() {
    forward_finite_smoke(1024, 16, 2000);
}

#[test]
fn bisect_cumsum_p1024_l16() {
    forward_at_stage(1024, 16, 2000, Stage::Cumsum);
}

#[test]
fn bisect_sig_cum_p1024_l16() {
    forward_at_stage(1024, 16, 2000, Stage::SigCum);
}

#[test]
fn bisect_rec_sig_cum_p1024_l16() {
    forward_at_stage(1024, 16, 2000, Stage::RecSigCum);
}

#[test]
fn bisect_transmittance_p1024_l16() {
    forward_at_stage(1024, 16, 2000, Stage::Transmittance);
}

#[test]
fn bisect_alpha_p1024_l16() {
    forward_at_stage(1024, 16, 2000, Stage::Alpha);
}

#[test]
fn bisect_weight_p1024_l16() {
    forward_at_stage(1024, 16, 2000, Stage::Weight);
}
