//! Bisect the layer where V2-S forward starts producing per-lane divergent
//! outputs for bit-identical input. Each stage builder ends at a tap; we
//! run the partial graph + safetensor weights + replicated input, then
//! compare per-batch slices of the output for divergence.
//!
//! Run:
//!   HSA_OVERRIDE_GFX_VERSION=11.0.0 cargo run --release --example v2s_lane_bisect
//!
//! Env vars:
//!   V2S_WEIGHTS    path to safetensors (default: ../mind-games/ext/efficientnet_v2_s.safetensors)
//!   V2S_STAGE      which stage to stop at: stem | f1 | f2 | f3 | f4 | f5 (default: stem)
//!   V2S_BATCH      batch size (default: 4)

use meganeura::Graph;
use meganeura::NodeId;
use meganeura::build_inference_session;
use meganeura::data::safetensors::SafeTensorsModel;
use meganeura::models::efficientnet;
use meganeura::models::efficientnet::{Spatial, fused_mbconv, mbconv};

const BATCH: u32 = 4;
const H: u32 = 192;
const W: u32 = 192;

fn main() {
    env_logger::init();

    let batch: u32 = std::env::var("V2S_BATCH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(BATCH);
    let stage: String = std::env::var("V2S_STAGE").unwrap_or_else(|_| "stem".to_string());
    let weights_path = std::env::var("V2S_WEIGHTS").unwrap_or_else(|_| {
        "/x/Code/mind-games/ext/efficientnet_v2_s.safetensors".to_string()
    });

    println!("== V2-S lane bisect ==");
    println!("batch={batch}  stage={stage}  weights={weights_path}");

    let mut g = Graph::new();
    let image = g.input("image", &[(batch * 3 * H * W) as usize]);

    // -------- Stem (always) --------
    let w0 = g.parameter("features.0.0.weight", &[24 * 3 * 3 * 3]);
    let x = g.conv2d_hw(image, w0, batch, 3, H, W, 24, 3, 3, 2, 1, 1);
    let stem_after_conv = x;
    // After stride-2 conv: spatial 96x96
    let bn0 = g.parameter("features.0.bn.fused_bias", &[24]);
    let x = g.add_per_channel(x, bn0, 24, 96 * 96);
    let stem_after_addperch = x;
    let x = g.silu(x);
    let stem_after_silu = x;

    // -------- Inline a SINGLE fused_mbconv block (features.1.0) so we
    // can tap at its mid-points without depending on the private helper.
    // Block: 3×3 conv → AddPerChannel → silu → residual add.
    let in_c = 24u32;
    let out_c = 24u32;
    let kernel = 3u32;
    let stride = 1u32;
    let padding = 1u32;
    let s_h = 96u32;
    let s_w = 96u32;
    let s1_area = s_h * s_w; // stride 1 → same area
    let f10_w = g.parameter(
        "features.1.0.block.0.0.weight",
        &[(out_c * in_c * kernel * kernel) as usize],
    );
    let f10_conv = g.conv2d_hw(stem_after_silu, f10_w, batch, in_c, s_h, s_w, out_c, kernel, kernel, stride, padding, padding);
    let f10_bn = g.parameter("features.1.0.block.0.bn.fused_bias", &[out_c as usize]);
    let f10_after_apc = g.add_per_channel(f10_conv, f10_bn, out_c, s1_area);
    let f10_after_silu = g.silu(f10_after_apc);
    // Residual add (stride==1 && in_c==out_c)
    let f10_after_residual = g.add(f10_after_silu, stem_after_silu);

    // -------- Continue: chain through features.1.1 .. features.5.8 --------
    // using public helpers, capturing tap after each block.
    let s96 = Spatial { h: 96, w: 96 };
    let f11 = fused_mbconv(&mut g, f10_after_residual, &s96, batch, 24, 24, 1, 3, 1, "features.1.1");

    // -------- features.2.0 INLINED to expose mid-block taps --------
    // expand_ratio=4, 24→48, kernel=3, stride=2.  Internal pieces:
    //   3×3 expand (24→96 channels, stride 2, 96→48 spatial)
    //   AddPerChannel  → silu
    //   1×1 project (96→48 channels)  ← first 1×1 conv in the network
    //   AddPerChannel
    let f20_w_e = g.parameter("features.2.0.block.0.0.weight", &[(96 * 24 * 3 * 3) as usize]);
    let f20_3x3 = g.conv2d_hw(f11, f20_w_e, batch, 24, 96, 96, 96, 3, 3, 2, 1, 1);
    // After stride-2 3×3 conv: 48×48 spatial, 96 channels
    let f20_bn_e = g.parameter("features.2.0.block.0.bn.fused_bias", &[96]);
    let f20_after_bn_e = g.add_per_channel(f20_3x3, f20_bn_e, 96, 48 * 48);
    let f20_after_silu = g.silu(f20_after_bn_e);
    // 1×1 project conv — the suspect.
    let f20_w_p = g.parameter("features.2.0.block.1.0.weight", &[(48 * 96) as usize]);
    let f20_1x1 = g.conv2d(f20_after_silu, f20_w_p, batch, 96, 48, 48, 48, 1, 1, 1, 0);
    let f20_bn_p = g.parameter("features.2.0.block.1.bn.fused_bias", &[48]);
    let f20 = g.add_per_channel(f20_1x1, f20_bn_p, 48, 48 * 48);

    let s48 = s96.after_conv(3, 2, 1); // 48
    let f21 = fused_mbconv(&mut g, f20, &s48, batch, 48, 48, 4, 3, 1, "features.2.1");
    let f22 = fused_mbconv(&mut g, f21, &s48, batch, 48, 48, 4, 3, 1, "features.2.2");
    let f23 = fused_mbconv(&mut g, f22, &s48, batch, 48, 48, 4, 3, 1, "features.2.3");

    // features.3.* — FusedMBConv e=4 48→64
    let f30 = fused_mbconv(&mut g, f23, &s48, batch, 48, 64, 4, 3, 2, "features.3.0");
    let s24 = s48.after_conv(3, 2, 1); // 24

    // features.4.0 — first MBConv (SE) block, 64→128 stride 2
    let f40 = mbconv(&mut g, f30, &s24, batch, 64, 128, 4, 3, 2, "features.4.0");
    let _s12 = s24.after_conv(3, 2, 1); // 12

    // Expose intermediates as outputs based on chosen stage.
    let (out_label, out_node, out_channels, out_h, out_w): (&'static str, NodeId, u32, u32, u32) = match stage.as_str() {
        "stem_conv" => ("stem.conv", stem_after_conv, 24, 96, 96),
        "stem_addperch" => ("stem.add_per_channel", stem_after_addperch, 24, 96, 96),
        "stem" | "stem_silu" => ("stem.silu", stem_after_silu, 24, 96, 96),
        "f10_conv" => ("features.1.0.conv", f10_conv, 24, 96, 96),
        "f10_addperch" => ("features.1.0.add_per_channel", f10_after_apc, 24, 96, 96),
        "f10_silu" => ("features.1.0.silu", f10_after_silu, 24, 96, 96),
        "f10" | "f10_residual" => ("features.1.0.residual", f10_after_residual, 24, 96, 96),
        "f11" => ("features.1.1", f11, 24, 96, 96),
        "f20_3x3" => ("features.2.0 3×3 expand", f20_3x3, 96, 48, 48),
        "f20_after_bn_e" => ("features.2.0 expand AddPerChannel", f20_after_bn_e, 96, 48, 48),
        "f20_after_silu" => ("features.2.0 expand silu", f20_after_silu, 96, 48, 48),
        "f20_1x1" => ("features.2.0 1×1 project", f20_1x1, 48, 48, 48),
        "f20" => ("features.2.0", f20, 48, 48, 48),
        "f21" => ("features.2.1", f21, 48, 48, 48),
        "f22" => ("features.2.2", f22, 48, 48, 48),
        "f23" => ("features.2.3", f23, 48, 48, 48),
        "f30" => ("features.3.0", f30, 64, 24, 24),
        "f40" => ("features.4.0", f40, 128, 12, 12),
        other => panic!("unknown stage {other:?}; supported: stem_conv|stem_addperch|stem|f10_conv|f10_addperch|f10_silu|f10|f11|f20|f21|f22|f23|f30|f40"),
    };
    g.set_outputs(vec![out_node]);

    let mut session = build_inference_session(&g);

    println!("loading weights from {weights_path}...");
    let weights = SafeTensorsModel::load(weights_path.clone().into()).expect("safetensors load");
    // Only set parameters that the partial graph actually requires.
    // Load only the parameters that our partial graph references — which
    // depends on the chosen stage. Easiest: try all, ignore not-in-graph.
    for name in efficientnet::weight_names() {
        if let Ok(data) = weights.tensor_f32(&name) {
            // Try set_parameter; if the param isn't in the current graph
            // it'll panic via "unknown parameter" — catch by checking the
            // session's plan first via a permissive helper.
            let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                session.set_parameter(&name, &data);
            }));
        }
    }

    // Build replicated input: a simple gradient image, repeated across batches.
    let single = {
        let mut v = vec![0.0f32; (3 * H * W) as usize];
        for c in 0..3 {
            for h in 0..H {
                for w in 0..W {
                    let idx = ((c * H + h) * W + w) as usize;
                    // Deterministic non-trivial pattern so the conv produces nonzero output.
                    v[idx] = ((h as f32 / H as f32) + (w as f32 / W as f32) + (c as f32 * 0.3)).sin();
                }
            }
        }
        v
    };
    let mut input = Vec::with_capacity((batch * 3 * H * W) as usize);
    for _ in 0..batch {
        input.extend_from_slice(&single);
    }
    session.set_input("image", &input);

    session.step();
    session.wait();

    let total = (batch as usize)
        * (out_channels as usize)
        * (out_h as usize)
        * (out_w as usize);
    let out = session.read_output(total);

    // For each batch, compute mean / max-abs of its slice.
    let per_batch_elems = (out_channels as usize) * (out_h as usize) * (out_w as usize);
    println!("\nOutput tap: {out_label}");
    println!("Per-batch summary (input is bit-identical across {batch} lanes):");
    println!(
        "{:>6}  {:>14}  {:>14}  {:>14}  {:>14}",
        "lane", "mean", "max-abs", "min", "max"
    );
    let mut max_diff = 0.0f32;
    let mut ref_slice: Vec<f32> = Vec::new();
    for b in 0..batch as usize {
        let start = b * per_batch_elems;
        let s = &out[start..start + per_batch_elems];
        let mean = s.iter().sum::<f32>() / s.len() as f32;
        let max_abs = s.iter().map(|x| x.abs()).fold(0.0, f32::max);
        let min = s.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max = s.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        println!("{b:>6}  {mean:>14.6}  {max_abs:>14.6}  {min:>14.6}  {max:>14.6}");
        if b == 0 {
            ref_slice = s.to_vec();
        } else {
            for i in 0..per_batch_elems {
                let d = (s[i] - ref_slice[i]).abs();
                if d > max_diff {
                    max_diff = d;
                }
            }
        }
    }
    println!("\nMax abs diff vs lane 0 across all later lanes: {max_diff:.6}");
    if max_diff < 1e-3 {
        println!("RESULT: lanes uniform at tap {out_label} ✓ (bug is downstream)");
    } else {
        println!("RESULT: lanes DIVERGENT at tap {out_label} ‼ (bug is here or upstream)");
    }
}
