//! EfficientNetV2-S features[0:6] model definition for meganeura.
//!
//! V2-S is the visual frontend used by the kindle agent's
//! `KindleVisualActor`. We deliberately use V2 (not V1) because:
//! - V2's Fused-MBConv early stages train ~20% faster on accelerators
//!   than V1's depthwise-heavy stem (paper §4.1).
//! - Stock torchvision V2-S weights (ImageNet-21k → 1k) are stronger
//!   than V1-B0's ImageNet-1k for downstream transfer.
//! - We're not reusing P2P's V1 fine-tuning, so the V1 architectural
//!   advantage (weight portability) doesn't apply here.
//!
//! On a 192×192 RGB input it outputs a `[N, 160, 12, 12]` feature map.
//!
//! Architecture (torchvision `efficientnet_v2_s`, sliced to `features[:6]`):
//! - `0` — stem conv 3×3 stride 2 + BN + SiLU                  (24 channels)
//! - `1` — 2× FusedMBConv, e=1, k=3, stride 1                  (24 → 24)
//! - `2` — 4× FusedMBConv, e=4, k=3, strides (2,1,1,1)         (24 → 48)
//! - `3` — 4× FusedMBConv, e=4, k=3, strides (2,1,1,1)         (48 → 64)
//! - `4` — 6× MBConv,      e=4, k=3, strides (2,1,1,1,1,1)     (64 → 128) [SE]
//! - `5` — 9× MBConv,      e=6, k=3, strides (1×9)             (128 → 160) [SE]
//!
//! BatchNorm is fused into the preceding conv's weight + a per-channel
//! bias at load time (mirrors the resnet pattern). The SE branch is a
//! GAP → 1×1 conv → SiLU → 1×1 conv → sigmoid → channel-wise multiply.
//!
//! Weight names follow the torchvision convention exactly so that
//! `efficientnet_v2_s` checkpoints load with no key remapping.
//! Channel-bias parameters are renamed to `<name>.bn.fused_bias` to
//! make the BN-fusion explicit.

use crate::graph::{Graph, NodeId};

/// Spatial dims tracked through the network.
#[derive(Clone, Copy)]
struct Spatial {
    h: u32,
    w: u32,
}

impl Spatial {
    fn after_conv(&self, kernel: u32, stride: u32, padding: u32) -> Self {
        Self {
            h: (self.h + 2 * padding - kernel) / stride + 1,
            w: (self.w + 2 * padding - kernel) / stride + 1,
        }
    }
    fn area(&self) -> u32 {
        self.h * self.w
    }
}

/// Build the EfficientNetV2-S features[0:6] graph.
///
/// Input is `"image"` with shape `[batch * 3 * 192 * 192]` in NCHW.
/// Returns the feature map node `[batch * 160 * 12 * 12]`.
pub fn build_graph(g: &mut Graph, batch: u32) -> NodeId {
    let s = Spatial { h: 192, w: 192 };

    // -------- Stem (features.0) --------
    let image = g.input("image", &[(batch * 3 * 192 * 192) as usize]);
    let w0 = g.parameter("features.0.0.weight", &[24 * 3 * 3 * 3]);
    let x = g.conv2d_hw(image, w0, batch, 3, s.h, s.w, 24, 3, 3, 2, 1, 1);
    let s = s.after_conv(3, 2, 1); // 96
    let bn0 = g.parameter("features.0.bn.fused_bias", &[24]);
    let x = g.add_per_channel(x, bn0, 24, s.area());
    let x = g.silu(x);

    // -------- features.1 — 2× FusedMBConv e=1, 24→24, s=1 --------
    let x = fused_mbconv(g, x, &s, batch, 24, 24, 1, 3, 1, "features.1.0");
    let x = fused_mbconv(g, x, &s, batch, 24, 24, 1, 3, 1, "features.1.1");

    // -------- features.2 — 4× FusedMBConv e=4, 24→48 --------
    let x = fused_mbconv(g, x, &s, batch, 24, 48, 4, 3, 2, "features.2.0");
    let s = s.after_conv(3, 2, 1); // 48
    let x = fused_mbconv(g, x, &s, batch, 48, 48, 4, 3, 1, "features.2.1");
    let x = fused_mbconv(g, x, &s, batch, 48, 48, 4, 3, 1, "features.2.2");
    let x = fused_mbconv(g, x, &s, batch, 48, 48, 4, 3, 1, "features.2.3");

    // -------- features.3 — 4× FusedMBConv e=4, 48→64 --------
    let x = fused_mbconv(g, x, &s, batch, 48, 64, 4, 3, 2, "features.3.0");
    let s = s.after_conv(3, 2, 1); // 24
    let x = fused_mbconv(g, x, &s, batch, 64, 64, 4, 3, 1, "features.3.1");
    let x = fused_mbconv(g, x, &s, batch, 64, 64, 4, 3, 1, "features.3.2");
    let x = fused_mbconv(g, x, &s, batch, 64, 64, 4, 3, 1, "features.3.3");

    // -------- features.4 — 6× MBConv e=4 SE, 64→128 --------
    let x = mbconv(g, x, &s, batch, 64, 128, 4, 3, 2, "features.4.0");
    let s = s.after_conv(3, 2, 1); // 12
    let x = mbconv(g, x, &s, batch, 128, 128, 4, 3, 1, "features.4.1");
    let x = mbconv(g, x, &s, batch, 128, 128, 4, 3, 1, "features.4.2");
    let x = mbconv(g, x, &s, batch, 128, 128, 4, 3, 1, "features.4.3");
    let x = mbconv(g, x, &s, batch, 128, 128, 4, 3, 1, "features.4.4");
    let x = mbconv(g, x, &s, batch, 128, 128, 4, 3, 1, "features.4.5");

    // -------- features.5 — 9× MBConv e=6 SE, 128→160, all stride 1 --------
    let x = mbconv(g, x, &s, batch, 128, 160, 6, 3, 1, "features.5.0");
    let x = mbconv(g, x, &s, batch, 160, 160, 6, 3, 1, "features.5.1");
    let x = mbconv(g, x, &s, batch, 160, 160, 6, 3, 1, "features.5.2");
    let x = mbconv(g, x, &s, batch, 160, 160, 6, 3, 1, "features.5.3");
    let x = mbconv(g, x, &s, batch, 160, 160, 6, 3, 1, "features.5.4");
    let x = mbconv(g, x, &s, batch, 160, 160, 6, 3, 1, "features.5.5");
    let x = mbconv(g, x, &s, batch, 160, 160, 6, 3, 1, "features.5.6");
    let x = mbconv(g, x, &s, batch, 160, 160, 6, 3, 1, "features.5.7");
    mbconv(g, x, &s, batch, 160, 160, 6, 3, 1, "features.5.8")
}

/// Build the same graph as [`build_graph`] but return one `NodeId` per
/// stage (stage 0 = stem post-SiLU; stages 1..=5 = feature stage outputs).
/// Used by the bisection correctness test to compare each stage against
/// torchvision separately. Don't ship downstream code calling this —
/// it's strictly a debugging hook.
pub fn build_graph_stage_outputs(g: &mut Graph, batch: u32) -> [NodeId; 6] {
    let s = Spatial { h: 192, w: 192 };

    let image = g.input("image", &[(batch * 3 * 192 * 192) as usize]);
    let w0 = g.parameter("features.0.0.weight", &[24 * 3 * 3 * 3]);
    let x = g.conv2d_hw(image, w0, batch, 3, s.h, s.w, 24, 3, 3, 2, 1, 1);
    let s = s.after_conv(3, 2, 1);
    let bn0 = g.parameter("features.0.bn.fused_bias", &[24]);
    let x = g.add_per_channel(x, bn0, 24, s.area());
    let stage0 = g.silu(x);

    let x = fused_mbconv(g, stage0, &s, batch, 24, 24, 1, 3, 1, "features.1.0");
    let stage1 = fused_mbconv(g, x, &s, batch, 24, 24, 1, 3, 1, "features.1.1");

    let x = fused_mbconv(g, stage1, &s, batch, 24, 48, 4, 3, 2, "features.2.0");
    let s = s.after_conv(3, 2, 1);
    let x = fused_mbconv(g, x, &s, batch, 48, 48, 4, 3, 1, "features.2.1");
    let x = fused_mbconv(g, x, &s, batch, 48, 48, 4, 3, 1, "features.2.2");
    let stage2 = fused_mbconv(g, x, &s, batch, 48, 48, 4, 3, 1, "features.2.3");

    let x = fused_mbconv(g, stage2, &s, batch, 48, 64, 4, 3, 2, "features.3.0");
    let s = s.after_conv(3, 2, 1);
    let x = fused_mbconv(g, x, &s, batch, 64, 64, 4, 3, 1, "features.3.1");
    let x = fused_mbconv(g, x, &s, batch, 64, 64, 4, 3, 1, "features.3.2");
    let stage3 = fused_mbconv(g, x, &s, batch, 64, 64, 4, 3, 1, "features.3.3");

    let x = mbconv(g, stage3, &s, batch, 64, 128, 4, 3, 2, "features.4.0");
    let s = s.after_conv(3, 2, 1);
    let x = mbconv(g, x, &s, batch, 128, 128, 4, 3, 1, "features.4.1");
    let x = mbconv(g, x, &s, batch, 128, 128, 4, 3, 1, "features.4.2");
    let x = mbconv(g, x, &s, batch, 128, 128, 4, 3, 1, "features.4.3");
    let x = mbconv(g, x, &s, batch, 128, 128, 4, 3, 1, "features.4.4");
    let stage4 = mbconv(g, x, &s, batch, 128, 128, 4, 3, 1, "features.4.5");

    let x = mbconv(g, stage4, &s, batch, 128, 160, 6, 3, 1, "features.5.0");
    let x = mbconv(g, x, &s, batch, 160, 160, 6, 3, 1, "features.5.1");
    let x = mbconv(g, x, &s, batch, 160, 160, 6, 3, 1, "features.5.2");
    let x = mbconv(g, x, &s, batch, 160, 160, 6, 3, 1, "features.5.3");
    let x = mbconv(g, x, &s, batch, 160, 160, 6, 3, 1, "features.5.4");
    let x = mbconv(g, x, &s, batch, 160, 160, 6, 3, 1, "features.5.5");
    let x = mbconv(g, x, &s, batch, 160, 160, 6, 3, 1, "features.5.6");
    let x = mbconv(g, x, &s, batch, 160, 160, 6, 3, 1, "features.5.7");
    let stage5 = mbconv(g, x, &s, batch, 160, 160, 6, 3, 1, "features.5.8");

    [stage0, stage1, stage2, stage3, stage4, stage5]
}

/// Fused-MBConv block (V2 early stages — replaces V1's expand+DW combo
/// with a single 3×3 conv).  Two layouts:
/// - `expand_ratio == 1`: single `block.0` 3×3 conv (no project step).
/// - `expand_ratio  > 1`: `block.0` 3×3 expand + `block.1` 1×1 project.
///
/// SE is intentionally absent — V2's design moves SE into MBConv only.
#[allow(clippy::too_many_arguments)]
fn fused_mbconv(
    g: &mut Graph,
    x: NodeId,
    s: &Spatial,
    batch: u32,
    in_c: u32,
    out_c: u32,
    expand_ratio: u32,
    kernel: u32,
    stride: u32,
    name: &str,
) -> NodeId {
    let padding = kernel / 2;
    let s1 = s.after_conv(kernel, stride, padding);

    let h = if expand_ratio == 1 {
        // Single 3×3 conv directly. No project.
        let w = g.parameter(
            &format!("{name}.block.0.0.weight"),
            &[(out_c * in_c * kernel * kernel) as usize],
        );
        let h = g.conv2d_hw(x, w, batch, in_c, s.h, s.w, out_c, kernel, kernel, stride, padding, padding);
        let bn = g.parameter(&format!("{name}.block.0.bn.fused_bias"), &[out_c as usize]);
        let h = g.add_per_channel(h, bn, out_c, s1.area());
        g.silu(h)
    } else {
        // Expand 3×3 (in_c → expanded), then project 1×1 (expanded → out_c).
        let expanded_c = in_c * expand_ratio;
        let w_e = g.parameter(
            &format!("{name}.block.0.0.weight"),
            &[(expanded_c * in_c * kernel * kernel) as usize],
        );
        let h = g.conv2d_hw(x, w_e, batch, in_c, s.h, s.w, expanded_c, kernel, kernel, stride, padding, padding);
        let bn_e = g.parameter(
            &format!("{name}.block.0.bn.fused_bias"),
            &[expanded_c as usize],
        );
        let h = g.add_per_channel(h, bn_e, expanded_c, s1.area());
        let h = g.silu(h);

        let w_p = g.parameter(
            &format!("{name}.block.1.0.weight"),
            &[(out_c * expanded_c) as usize],
        );
        let h = g.conv2d(h, w_p, batch, expanded_c, s1.h, s1.w, out_c, 1, 1, 1, 0);
        let bn_p = g.parameter(&format!("{name}.block.1.bn.fused_bias"), &[out_c as usize]);
        g.add_per_channel(h, bn_p, out_c, s1.area())
    };

    if stride == 1 && in_c == out_c {
        g.add(h, x)
    } else {
        h
    }
}

/// Mobile Inverted Bottleneck Conv (MBConv) block — V1-style with SE.
///
/// Composition: `expand 1×1 → DW k×k → SE → project 1×1`, with an
/// optional residual skip when `stride == 1 && in_c == out_c`.
///
/// V2-S always uses expansion ratio > 1 in its MBConv stages, so this
/// helper assumes `expand_ratio > 1` (V1's MBConv1-style is handled
/// by `fused_mbconv` instead in the V2 architecture).
#[allow(clippy::too_many_arguments)]
fn mbconv(
    g: &mut Graph,
    x: NodeId,
    s: &Spatial,
    batch: u32,
    in_c: u32,
    out_c: u32,
    expand_ratio: u32,
    kernel: u32,
    stride: u32,
    name: &str,
) -> NodeId {
    assert!(expand_ratio > 1, "use fused_mbconv for expand_ratio == 1");
    let expanded_c = in_c * expand_ratio;
    let padding = kernel / 2;
    let s_dw = s.after_conv(kernel, stride, padding);

    // -------- Expand 1×1 --------
    let w_e = g.parameter(
        &format!("{name}.block.0.0.weight"),
        &[(expanded_c * in_c) as usize],
    );
    let h = g.conv2d(x, w_e, batch, in_c, s.h, s.w, expanded_c, 1, 1, 1, 0);
    let bn_e = g.parameter(
        &format!("{name}.block.0.bn.fused_bias"),
        &[expanded_c as usize],
    );
    let h = g.add_per_channel(h, bn_e, expanded_c, s.area());
    let h = g.silu(h);

    // -------- Depthwise k×k --------
    let dw_w = g.parameter(
        &format!("{name}.block.1.0.weight"),
        &[(expanded_c * kernel * kernel) as usize],
    );
    let h = g.conv2d_dw(
        h, dw_w, batch, expanded_c, s.h, s.w, kernel, kernel, stride, padding, padding,
    );
    let bn_d = g.parameter(
        &format!("{name}.block.1.bn.fused_bias"),
        &[expanded_c as usize],
    );
    let h = g.add_per_channel(h, bn_d, expanded_c, s_dw.area());
    let h = g.silu(h);

    // -------- Squeeze-and-Excitation --------
    let sq = se_squeeze_channels(name, expanded_c);
    let pooled = g.global_avg_pool(h, batch, expanded_c, s_dw.area());
    let pooled_2d = g.reshape(pooled, &[batch as usize, expanded_c as usize]);
    let fc1_w = g.parameter(
        &format!("{name}.block.2.fc1.weight"),
        &[sq as usize, expanded_c as usize],
    );
    let fc1_b = g.parameter(&format!("{name}.block.2.fc1.bias"), &[sq as usize]);
    let z = g.matmul_bt(pooled_2d, fc1_w);
    let z = g.bias_add(z, fc1_b);
    let z = g.silu(z);

    let fc2_w = g.parameter(
        &format!("{name}.block.2.fc2.weight"),
        &[expanded_c as usize, sq as usize],
    );
    let fc2_b = g.parameter(
        &format!("{name}.block.2.fc2.bias"),
        &[expanded_c as usize],
    );
    let g_ate = g.matmul_bt(z, fc2_w);
    let g_ate = g.bias_add(g_ate, fc2_b);
    let g_ate = g.sigmoid(g_ate);
    let g_ate = g.reshape(g_ate, &[(batch * expanded_c) as usize]);
    let h = g.mul_per_channel(h, g_ate, expanded_c, s_dw.area());

    // -------- Project 1×1 --------
    let proj_w = g.parameter(
        &format!("{name}.block.3.0.weight"),
        &[(out_c * expanded_c) as usize],
    );
    let h = g.conv2d(h, proj_w, batch, expanded_c, s_dw.h, s_dw.w, out_c, 1, 1, 1, 0);
    let bn_p = g.parameter(&format!("{name}.block.3.bn.fused_bias"), &[out_c as usize]);
    let h = g.add_per_channel(h, bn_p, out_c, s_dw.area());

    if stride == 1 && in_c == out_c {
        g.add(h, x)
    } else {
        h
    }
}

/// SE squeeze channel count for V2-S MBConv stages.  Matches the
/// torchvision shapes: features.4.* uses sq=16 then 32, features.5.*
/// uses sq=32 (e=6 on 128-c) then 40 (e=6 on 160-c).
fn se_squeeze_channels(name: &str, expanded: u32) -> u32 {
    // Direct lookup for V2-S feature.<i>.<j>.
    let map: &[(&str, u32)] = &[
        ("features.4.0", 16), // expanded=256 → 16
        ("features.4.1", 32), // expanded=512 → 32
        ("features.4.2", 32),
        ("features.4.3", 32),
        ("features.4.4", 32),
        ("features.4.5", 32),
        ("features.5.0", 32), // expanded=768 → 32
        ("features.5.1", 40), // expanded=960 → 40
        ("features.5.2", 40),
        ("features.5.3", 40),
        ("features.5.4", 40),
        ("features.5.5", 40),
        ("features.5.6", 40),
        ("features.5.7", 40),
        ("features.5.8", 40),
    ];
    for &(k, v) in map {
        if name == k {
            return v;
        }
    }
    (expanded / 24).max(1)
}

/// Stage descriptor used by [`weight_names`] to enumerate every
/// parameter in topo order.
///
/// Fields: `(stage, block, in_c, out_c, expand_ratio, kernel, stride, kind)`
/// where `kind` is `"fused"` for V2 FusedMBConv and `"mbconv"` for the
/// V1-style MBConv-with-SE block used in features[4..].
type StageDesc = (u32, u32, u32, u32, u32, u32, u32, &'static str);

/// Names of all parameters in the model — useful for the weight loader.
///
/// Listed in topo order so a sequential checkpoint reader sees them
/// in the same order they're allocated.
pub fn weight_names() -> Vec<String> {
    let mut names = Vec::new();
    names.push("features.0.0.weight".into());
    names.push("features.0.bn.fused_bias".into());

    let stages: &[StageDesc] = &[
        // features.1 — 2× FusedMBConv e=1
        (1, 0, 24, 24, 1, 3, 1, "fused"),
        (1, 1, 24, 24, 1, 3, 1, "fused"),
        // features.2 — 4× FusedMBConv e=4
        (2, 0, 24, 48, 4, 3, 2, "fused"),
        (2, 1, 48, 48, 4, 3, 1, "fused"),
        (2, 2, 48, 48, 4, 3, 1, "fused"),
        (2, 3, 48, 48, 4, 3, 1, "fused"),
        // features.3 — 4× FusedMBConv e=4
        (3, 0, 48, 64, 4, 3, 2, "fused"),
        (3, 1, 64, 64, 4, 3, 1, "fused"),
        (3, 2, 64, 64, 4, 3, 1, "fused"),
        (3, 3, 64, 64, 4, 3, 1, "fused"),
        // features.4 — 6× MBConv e=4 SE
        (4, 0, 64, 128, 4, 3, 2, "mbconv"),
        (4, 1, 128, 128, 4, 3, 1, "mbconv"),
        (4, 2, 128, 128, 4, 3, 1, "mbconv"),
        (4, 3, 128, 128, 4, 3, 1, "mbconv"),
        (4, 4, 128, 128, 4, 3, 1, "mbconv"),
        (4, 5, 128, 128, 4, 3, 1, "mbconv"),
        // features.5 — 9× MBConv e=6 SE
        (5, 0, 128, 160, 6, 3, 1, "mbconv"),
        (5, 1, 160, 160, 6, 3, 1, "mbconv"),
        (5, 2, 160, 160, 6, 3, 1, "mbconv"),
        (5, 3, 160, 160, 6, 3, 1, "mbconv"),
        (5, 4, 160, 160, 6, 3, 1, "mbconv"),
        (5, 5, 160, 160, 6, 3, 1, "mbconv"),
        (5, 6, 160, 160, 6, 3, 1, "mbconv"),
        (5, 7, 160, 160, 6, 3, 1, "mbconv"),
        (5, 8, 160, 160, 6, 3, 1, "mbconv"),
    ];
    for &(stage, block, _in_c, _out_c, expand_ratio, _kernel, _stride, kind) in stages {
        let name = format!("features.{stage}.{block}");
        match kind {
            "fused" => {
                names.push(format!("{name}.block.0.0.weight"));
                names.push(format!("{name}.block.0.bn.fused_bias"));
                if expand_ratio > 1 {
                    names.push(format!("{name}.block.1.0.weight"));
                    names.push(format!("{name}.block.1.bn.fused_bias"));
                }
            }
            "mbconv" => {
                names.push(format!("{name}.block.0.0.weight"));
                names.push(format!("{name}.block.0.bn.fused_bias"));
                names.push(format!("{name}.block.1.0.weight"));
                names.push(format!("{name}.block.1.bn.fused_bias"));
                names.push(format!("{name}.block.2.fc1.weight"));
                names.push(format!("{name}.block.2.fc1.bias"));
                names.push(format!("{name}.block.2.fc2.weight"));
                names.push(format!("{name}.block.2.fc2.bias"));
                names.push(format!("{name}.block.3.0.weight"));
                names.push(format!("{name}.block.3.bn.fused_bias"));
            }
            _ => unreachable!(),
        }
    }
    names
}
