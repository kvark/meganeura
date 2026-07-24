//! Conditioned latent-diffusion U-Net model definition for meganeura.
//!
//! Implements a scaled, structurally representative diffusion U-Net:
//! - Encoder: Conv2d → [ResBlock + Downsample] × N
//! - Timestep MLP injected into every residual block
//! - Spatial transformer blocks with self- and text cross-attention
//! - Middle: ResBlock + spatial transformer
//! - Decoder: [ResBlock + Upsample + skip concat] × N → Conv2d
//!
//! This is intentionally much smaller than the 860M-parameter SD 1.5 U-Net,
//! but it retains the operator mix that distinguishes conditioned latent
//! diffusion from a generic convolutional U-Net.
//!
//! All tensors are flat 1D arrays in NCHW layout.

use crate::graph::{Graph, NodeId};

/// Configuration for the scaled conditioned U-Net.
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
    /// Width of the caller-provided sinusoidal timestep embedding.
    pub time_input_dim: u32,
    /// Hidden/output width of the timestep MLP.
    pub time_embed_dim: u32,
    /// Token count of the text-conditioning sequence.
    pub context_len: u32,
    /// Width of each text-conditioning token (768 for CLIP SD 1.x).
    pub context_dim: u32,
    /// Per-head attention width in spatial transformer blocks.
    pub attention_head_dim: u32,
}

impl SDUNetConfig {
    /// A tiny configuration suitable for quick smoke tests.
    ///
    /// This configuration omits no operator families from [`Self::small`],
    /// but uses half the channel width and a smaller timestep MLP.
    pub fn tiny() -> Self {
        Self {
            batch_size: 1,
            in_channels: 4,
            base_channels: 32,
            num_levels: 3,
            resolution: 32,
            num_groups: 8,
            gn_eps: 1e-5,
            time_input_dim: 32,
            time_embed_dim: 128,
            context_len: 77,
            context_dim: 768,
            attention_head_dim: 32,
        }
    }

    /// Paper workload: a reduced-width, conditioned latent-diffusion U-Net.
    ///
    /// Batch 1 is required because the current differentiable attention
    /// primitive represents one sequence per op. The 77×768 text context and
    /// the timestep-conditioning structure match SD 1.x conventions.
    pub fn small() -> Self {
        Self {
            batch_size: 1,
            in_channels: 4,
            base_channels: 64,
            num_levels: 3,
            resolution: 32,
            num_groups: 16,
            gn_eps: 1e-5,
            time_input_dim: 64,
            time_embed_dim: 256,
            context_len: 77,
            context_dim: 768,
            attention_head_dim: 32,
        }
    }

    fn channel_mult(&self) -> Vec<u32> {
        (0..self.num_levels).map(|i| 1u32 << i).collect()
    }

    fn has_attention(&self, level: usize) -> bool {
        // At reduced 32×32 latent resolution, keep spatial transformers at
        // 16×16 and below. This controls benchmark size while retaining both
        // self-attention and the characteristic 77-token text cross-attention.
        level > 0
    }
}

/// Spatial state tracked during graph construction.
struct SpatialState {
    h: u32,
    w: u32,
    c: u32,
}

fn linear(
    g: &mut Graph,
    x: NodeId,
    prefix: &str,
    in_features: u32,
    out_features: u32,
    bias: bool,
) -> NodeId {
    let weight = g.parameter(
        &format!("{prefix}.weight"),
        &[in_features as usize, out_features as usize],
    );
    let y = g.matmul(x, weight);
    if bias {
        let bias = g.parameter(&format!("{prefix}.bias"), &[out_features as usize]);
        g.bias_add(y, bias)
    } else {
        y
    }
}

fn timestep_embedding(g: &mut Graph, cfg: &SDUNetConfig) -> NodeId {
    let input = g.input(
        "timestep_embedding",
        &[cfg.batch_size as usize, cfg.time_input_dim as usize],
    );
    let hidden = linear(
        g,
        input,
        "time_mlp.fc1",
        cfg.time_input_dim,
        cfg.time_embed_dim,
        true,
    );
    let hidden = g.silu(hidden);
    linear(
        g,
        hidden,
        "time_mlp.fc2",
        cfg.time_embed_dim,
        cfg.time_embed_dim,
        true,
    )
}

/// Build a diffusion ResBlock:
/// GroupNorm → SiLU → Conv3×3 → add(projected timestep)
/// → GroupNorm → SiLU → Conv3×3 + residual.
/// If in_c != out_c, adds a 1×1 residual projection.
fn resblock(
    g: &mut Graph,
    x: NodeId,
    time_emb: NodeId,
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
    let mut h = g.conv2d(h, conv1_w, batch, in_c, s.h, s.w, out_c, 3, 3, 1, 1);

    // Project the timestep embedding to channels and broadcast it over the
    // NCHW spatial plane. MatMul([N*C,1], [1,HW]) gives exactly the flat
    // [N,C,H,W] ordering used by the convolution kernels.
    let time = linear(
        g,
        time_emb,
        &format!("{prefix}.time_proj"),
        cfg.time_embed_dim,
        out_c,
        true,
    );
    let time = g.reshape(time, &[(batch * out_c) as usize, 1]);
    let spatial_ones = g.constant(vec![1.0; spatial as usize], &[1, spatial as usize]);
    let time = g.matmul(time, spatial_ones);
    let time = g.reshape(time, &[(batch * out_c * spatial) as usize]);
    h = g.add(h, time);

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

fn token_layer_norm(g: &mut Graph, x: NodeId, prefix: &str, channels: u32, eps: f32) -> NodeId {
    let weight = g.parameter(&format!("{prefix}.weight"), &[channels as usize]);
    let bias = g.parameter(&format!("{prefix}.bias"), &[channels as usize]);
    g.layer_norm(x, weight, bias, eps)
}

/// Stable-Diffusion-style spatial transformer:
///
/// GroupNorm + 1×1 projection → self-attention → text cross-attention
/// → GELU feed-forward → 1×1 projection + spatial residual.
fn spatial_transformer(
    g: &mut Graph,
    x: NodeId,
    context: NodeId,
    prefix: &str,
    cfg: &SDUNetConfig,
    s: &SpatialState,
) -> NodeId {
    assert_eq!(
        cfg.batch_size, 1,
        "spatial transformer currently supports one sequence per attention op"
    );
    assert_eq!(
        s.c % cfg.attention_head_dim,
        0,
        "channels must be divisible by attention_head_dim"
    );
    let spatial = s.h * s.w;
    let channels = s.c;
    let num_heads = channels / cfg.attention_head_dim;

    let norm_w = g.parameter(&format!("{prefix}.norm.weight"), &[channels as usize]);
    let norm_b = g.parameter(&format!("{prefix}.norm.bias"), &[channels as usize]);
    let h = g.group_norm(
        x,
        norm_w,
        norm_b,
        1,
        channels,
        spatial,
        cfg.num_groups,
        cfg.gn_eps,
    );
    let proj_in_w = g.parameter(
        &format!("{prefix}.proj_in.weight"),
        &[(channels * channels) as usize],
    );
    let h = g.conv2d(h, proj_in_w, 1, channels, s.h, s.w, channels, 1, 1, 1, 0);
    let h = g.reshape(h, &[channels as usize, spatial as usize]);
    let mut tokens = g.transpose(h);

    // Self-attention.
    let norm = token_layer_norm(
        g,
        tokens,
        &format!("{prefix}.transformer.norm1"),
        channels,
        cfg.gn_eps,
    );
    let q = linear(
        g,
        norm,
        &format!("{prefix}.transformer.self_attn.q_proj"),
        channels,
        channels,
        false,
    );
    let k = linear(
        g,
        norm,
        &format!("{prefix}.transformer.self_attn.k_proj"),
        channels,
        channels,
        false,
    );
    let v = linear(
        g,
        norm,
        &format!("{prefix}.transformer.self_attn.v_proj"),
        channels,
        channels,
        false,
    );
    let attended = g.multi_head_attn(q, k, v, num_heads, num_heads, cfg.attention_head_dim, false);
    let attended = linear(
        g,
        attended,
        &format!("{prefix}.transformer.self_attn.out_proj"),
        channels,
        channels,
        false,
    );
    tokens = g.add(tokens, attended);

    // Text cross-attention against a 77×768 CLIP-style context.
    let norm = token_layer_norm(
        g,
        tokens,
        &format!("{prefix}.transformer.norm2"),
        channels,
        cfg.gn_eps,
    );
    let q = linear(
        g,
        norm,
        &format!("{prefix}.transformer.cross_attn.q_proj"),
        channels,
        channels,
        false,
    );
    let k = linear(
        g,
        context,
        &format!("{prefix}.transformer.cross_attn.k_proj"),
        cfg.context_dim,
        channels,
        false,
    );
    let v = linear(
        g,
        context,
        &format!("{prefix}.transformer.cross_attn.v_proj"),
        cfg.context_dim,
        channels,
        false,
    );
    let attended = g.multi_head_attn(q, k, v, num_heads, num_heads, cfg.attention_head_dim, true);
    let attended = linear(
        g,
        attended,
        &format!("{prefix}.transformer.cross_attn.out_proj"),
        channels,
        channels,
        false,
    );
    tokens = g.add(tokens, attended);

    // Transformer feed-forward. SD 1.x uses GEGLU; GELU keeps this reduced
    // workload within the current primitive set while preserving the dense
    // expansion/contraction profile.
    let norm = token_layer_norm(
        g,
        tokens,
        &format!("{prefix}.transformer.norm3"),
        channels,
        cfg.gn_eps,
    );
    let ff = linear(
        g,
        norm,
        &format!("{prefix}.transformer.ff.fc1"),
        channels,
        4 * channels,
        true,
    );
    let ff = g.gelu(ff);
    let ff = linear(
        g,
        ff,
        &format!("{prefix}.transformer.ff.fc2"),
        4 * channels,
        channels,
        true,
    );
    tokens = g.add(tokens, ff);

    let h = g.transpose(tokens);
    let h = g.reshape(h, &[(channels * spatial) as usize]);
    let proj_out_w = g.parameter(
        &format!("{prefix}.proj_out.weight"),
        &[(channels * channels) as usize],
    );
    let h = g.conv2d(h, proj_out_w, 1, channels, s.h, s.w, channels, 1, 1, 1, 0);
    g.add(x, h)
}

/// Build the SD U-Net forward graph (no loss).
///
/// Returns the noise prediction node. The graph expects:
/// - Input "noisy_latent": flat `[batch * in_c * res * res]`
/// - Input "timestep_embedding": `[batch, time_input_dim]`
/// - Input "text_context": `[context_len, context_dim]`
pub fn build_unet(g: &mut Graph, cfg: &SDUNetConfig) -> NodeId {
    build_unet_inner(g, cfg)
}

/// Build the SD U-Net training graph.
///
/// Returns the MSE loss node. The graph expects:
/// - Input "noisy_latent": flat `[batch * in_c * res * res]`
/// - Input "timestep_embedding": `[batch, time_input_dim]`
/// - Input "text_context": `[context_len, context_dim]`
/// - Input "noise_target": flat `[batch * in_c * res * res]` (the noise to predict)
pub fn build_training_graph(g: &mut Graph, cfg: &SDUNetConfig) -> NodeId {
    let batch = cfg.batch_size;
    let res = cfg.resolution;
    let in_c = cfg.in_channels;
    let in_size = (batch * in_c * res * res) as usize;

    let pred = build_unet_inner(g, cfg);
    let target = g.input("noise_target", &[in_size]);

    // MSE loss: mean((pred - target)²)
    let neg_target = g.neg(target);
    let diff = g.add(pred, neg_target);
    let sq = g.mul(diff, diff);
    g.mean_all(sq)
}

/// Inner U-Net forward pass that returns the noise prediction tensor.
fn build_unet_inner(g: &mut Graph, cfg: &SDUNetConfig) -> NodeId {
    let batch = cfg.batch_size;
    let res = cfg.resolution;
    let in_c = cfg.in_channels;
    let in_size = (batch * in_c * res * res) as usize;
    let ch_mults = cfg.channel_mult();

    // Inputs
    let noisy = g.input("noisy_latent", &[in_size]);
    let time_emb = timestep_embedding(g, cfg);
    let context = g.input(
        "text_context",
        &[cfg.context_len as usize, cfg.context_dim as usize],
    );

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
        x = resblock(
            g,
            x,
            time_emb,
            &format!("encoder.{level}.resblock"),
            cfg,
            &s,
            out_c,
        );
        s.c = out_c;
        if cfg.has_attention(level) {
            x = spatial_transformer(g, x, context, &format!("encoder.{level}.attn"), cfg, &s);
        }

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
    x = resblock(g, x, time_emb, "middle.resblock", cfg, &s, s.c);
    x = spatial_transformer(g, x, context, "middle.attn", cfg, &s);

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
            time_emb,
            &format!("decoder.{level}.resblock"),
            cfg,
            &dec_s,
            out_c,
        );
        s.c = out_c;
        if cfg.has_attention(level) {
            x = spatial_transformer(g, x, context, &format!("decoder.{level}.attn"), cfg, &s);
        }
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
    g.conv2d(x, conv_out_w, batch, base_c, res, res, in_c, 3, 3, 1, 1)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Op;

    #[test]
    fn paper_workload_shape_is_stable() {
        let cfg = SDUNetConfig::small();
        assert_eq!(count_params(&cfg), 10_928_768);

        let mut graph = Graph::new();
        let loss = build_training_graph(&mut graph, &cfg);
        graph.set_outputs(vec![loss]);

        let input_names: Vec<&str> = graph
            .nodes()
            .iter()
            .filter_map(|node| match node.op {
                Op::Input { ref name } => Some(name.as_str()),
                _ => None,
            })
            .collect();
        assert_eq!(
            input_names,
            [
                "noisy_latent",
                "timestep_embedding",
                "text_context",
                "noise_target"
            ]
        );
        assert_eq!(
            graph
                .nodes()
                .iter()
                .filter(|node| matches!(node.op, Op::MultiHeadAttn { .. }))
                .count(),
            10
        );
    }
}
