//! Stable Diffusion U-Net model definition for meganeura.
//!
//! Implements a scaled-down version of the SD 1.5 U-Net architecture:
//! - Encoder: Conv2d → [ResBlock + Downsample] × N
//! - Middle: ResBlock
//! - Decoder: [ResBlock + Upsample + skip concat] × N → Conv2d
//!
//! Each ResBlock: GroupNorm → SiLU → Conv3×3 → GroupNorm → SiLU → Conv3×3 + residual.
//!
//! All tensors are flat 1D arrays in NCHW layout.

use crate::graph::{Graph, NodeId};

/// Configuration for the tiny SD U-Net.
pub struct SDUNetConfig {
    /// Number of images in a batch.
    pub batch_size: u32,
    /// Number of input/output channels (e.g. 4 for latent space).
    pub in_channels: u32,
    /// Base channel width (doubled at each level).
    pub base_channels: u32,
    /// Number of downsampling levels.
    pub num_levels: usize,
    /// Spatial resolution of the input (square: H = W = resolution).
    pub resolution: u32,
    /// Number of groups for GroupNorm.
    pub num_groups: u32,
    /// GroupNorm epsilon.
    pub gn_eps: f32,
}

impl SDUNetConfig {
    /// A tiny configuration suitable for benchmarking.
    /// ~120K parameters, 32×32 latent, batch 4.
    pub fn tiny() -> Self {
        Self {
            batch_size: 4,
            in_channels: 4,
            base_channels: 32,
            num_levels: 3,
            resolution: 32,
            num_groups: 8,
            gn_eps: 1e-5,
        }
    }

    /// A small configuration closer to real SD dimensions.
    /// ~2M parameters, 32×32 latent, batch 2.
    pub fn small() -> Self {
        Self {
            batch_size: 2,
            in_channels: 4,
            base_channels: 64,
            num_levels: 3,
            resolution: 32,
            num_groups: 16,
            gn_eps: 1e-5,
        }
    }

    fn channel_mult(&self) -> Vec<u32> {
        (0..self.num_levels).map(|i| 1u32 << i).collect()
    }
}

/// Spatial state tracked during graph construction.
struct SpatialState {
    h: u32,
    w: u32,
    c: u32,
}

/// Build a resblock: GroupNorm → SiLU → Conv3×3 → GroupNorm → SiLU → Conv3×3 + residual.
/// If in_c != out_c, adds a 1×1 residual projection.
fn resblock(
    g: &mut Graph,
    x: NodeId,
    prefix: &str,
    cfg: &SDUNetConfig,
    s: &SpatialState,
    out_c: u32,
) -> NodeId {
    let batch = cfg.batch_size;
    let spatial = s.h * s.w;
    let in_c = s.c;

    // GroupNorm1 → SiLU → Conv3×3
    let gn1_w = g.parameter(&format!("{prefix}.norm1.weight"), &[in_c as usize]);
    let gn1_b = g.parameter(&format!("{prefix}.norm1.bias"), &[in_c as usize]);
    let h = g.group_norm(
        x,
        gn1_w,
        gn1_b,
        batch,
        in_c,
        spatial,
        cfg.num_groups,
        cfg.gn_eps,
    );
    let h = g.silu(h);
    let conv1_w = g.parameter(
        &format!("{prefix}.conv1.weight"),
        &[(out_c * in_c * 9) as usize],
    );
    let h = g.conv2d(h, conv1_w, batch, in_c, s.h, s.w, out_c, 3, 3, 1, 1);

    // GroupNorm2 → SiLU → Conv3×3
    let gn2_w = g.parameter(&format!("{prefix}.norm2.weight"), &[out_c as usize]);
    let gn2_b = g.parameter(&format!("{prefix}.norm2.bias"), &[out_c as usize]);
    let h = g.group_norm(
        h,
        gn2_w,
        gn2_b,
        batch,
        out_c,
        spatial,
        cfg.num_groups,
        cfg.gn_eps,
    );
    let h = g.silu(h);
    let conv2_w = g.parameter(
        &format!("{prefix}.conv2.weight"),
        &[(out_c * out_c * 9) as usize],
    );
    let h = g.conv2d(h, conv2_w, batch, out_c, s.h, s.w, out_c, 3, 3, 1, 1);

    // Residual connection
    if in_c == out_c {
        g.add(x, h)
    } else {
        // 1×1 projection for channel change
        let res_w = g.parameter(
            &format!("{prefix}.res_conv.weight"),
            &[(out_c * in_c) as usize],
        );
        let x_proj = g.conv2d(x, res_w, batch, in_c, s.h, s.w, out_c, 1, 1, 1, 0);
        g.add(x_proj, h)
    }
}

/// Build the SD U-Net training graph.
///
/// Returns the MSE loss node. The graph expects:
/// - Input "noisy_latent": flat `[batch * in_c * res * res]`
/// - Input "noise_target": flat `[batch * in_c * res * res]` (the noise to predict)
pub fn build_training_graph(g: &mut Graph, cfg: &SDUNetConfig) -> NodeId {
    let batch = cfg.batch_size;
    let res = cfg.resolution;
    let in_c = cfg.in_channels;
    let in_size = (batch * in_c * res * res) as usize;
    let ch_mults = cfg.channel_mult();

    // Inputs
    let noisy = g.input("noisy_latent", &[in_size]);
    let target = g.input("noise_target", &[in_size]);

    // Input conv: in_channels → base_channels
    let base_c = cfg.base_channels;
    let conv_in_w = g.parameter("conv_in.weight", &[(base_c * in_c * 3 * 3) as usize]);
    let mut x = g.conv2d(noisy, conv_in_w, batch, in_c, res, res, base_c, 3, 3, 1, 1);

    let mut s = SpatialState {
        h: res,
        w: res,
        c: base_c,
    };

    // ---- Encoder ----
    let mut skip_connections: Vec<(NodeId, SpatialState)> = Vec::new();

    for (level, &mult) in ch_mults.iter().enumerate() {
        let out_c = base_c * mult;

        // ResBlock
        x = resblock(g, x, &format!("encoder.{level}.resblock"), cfg, &s, out_c);
        s.c = out_c;

        // Save skip connection
        skip_connections.push((
            x,
            SpatialState {
                h: s.h,
                w: s.w,
                c: s.c,
            },
        ));

        // Downsample (stride-2 conv) except at last level
        if level < cfg.num_levels - 1 {
            let down_w = g.parameter(
                &format!("encoder.{level}.downsample.weight"),
                &[(out_c * out_c * 3 * 3) as usize],
            );
            x = g.conv2d(x, down_w, batch, out_c, s.h, s.w, out_c, 3, 3, 2, 1);
            s.h = (s.h + 2 - 3) / 2 + 1; // padding=1, stride=2, kernel=3
            s.w = (s.w + 2 - 3) / 2 + 1;
        }
    }

    // ---- Middle ----
    x = resblock(g, x, "middle.resblock", cfg, &s, s.c);

    // ---- Decoder ----
    for level in (0..cfg.num_levels).rev() {
        let out_c = base_c * ch_mults[level];

        // Upsample (except at the highest-res level)
        if level < cfg.num_levels - 1 {
            x = g.upsample_2x(x, batch, s.c, s.h, s.w);
            s.h *= 2;
            s.w *= 2;
        }

        // Concat with skip connection
        let &(skip, ref skip_s) = &skip_connections[level];
        assert_eq!(s.h, skip_s.h, "spatial mismatch at level {level}");
        assert_eq!(s.w, skip_s.w, "spatial mismatch at level {level}");
        let spatial = s.h * s.w;
        x = g.concat(x, skip, batch, s.c, skip_s.c, spatial);
        let concat_c = s.c + skip_s.c;

        // ResBlock (input channels = concat_c, output = out_c)
        let dec_s = SpatialState {
            h: s.h,
            w: s.w,
            c: concat_c,
        };
        x = resblock(
            g,
            x,
            &format!("decoder.{level}.resblock"),
            cfg,
            &dec_s,
            out_c,
        );
        s.c = out_c;
    }

    // Output: GroupNorm → SiLU → Conv3×3 → in_channels
    let gn_out_w = g.parameter("conv_out.norm.weight", &[base_c as usize]);
    let gn_out_b = g.parameter("conv_out.norm.bias", &[base_c as usize]);
    x = g.group_norm(
        x,
        gn_out_w,
        gn_out_b,
        batch,
        base_c,
        res * res,
        cfg.num_groups,
        cfg.gn_eps,
    );
    x = g.silu(x);
    let conv_out_w = g.parameter("conv_out.weight", &[(in_c * base_c * 3 * 3) as usize]);
    let pred = g.conv2d(x, conv_out_w, batch, base_c, res, res, in_c, 3, 3, 1, 1);

    // MSE loss: mean((pred - target)²)
    let neg_target = g.neg(target);
    let diff = g.add(pred, neg_target);
    let sq = g.mul(diff, diff);
    g.mean_all(sq)
}

/// Count the total number of parameters in the U-Net.
pub fn count_params(cfg: &SDUNetConfig) -> usize {
    let mut g = Graph::new();
    let _loss = build_training_graph(&mut g, cfg);
    g.nodes()
        .iter()
        .filter(|n| matches!(n.op, crate::graph::Op::Parameter { .. }))
        .map(|n| n.ty.num_elements())
        .sum()
}
