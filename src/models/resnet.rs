//! ResNet-18 model definition for meganeura.
//!
//! Architecture: conv7x7 → maxpool → 4 stages of BasicBlocks → GAP → FC(1000).
//! BatchNorm is fused into convolution weights at load time (no runtime BN op).
//!
//! Reference: <https://arxiv.org/abs/1512.03385>
//! Weights: torchvision `resnet18(weights=ResNet18_Weights.DEFAULT)`

use crate::graph::{Graph, NodeId};

/// Spatial dimensions tracked through the network.
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
}

/// Build the ResNet-18 inference graph.
///
/// Returns the logits output node `[batch, 1000]`. Input is `"image"` with
/// shape `[batch * 3 * 224 * 224]` in NCHW layout.
///
/// Weight names follow the torchvision convention:
/// `conv1.weight`, `bn1.weight`, `layer1.0.conv1.weight`, etc.
pub fn build_graph(g: &mut Graph, batch: u32) -> NodeId {
    let s = Spatial { h: 224, w: 224 };

    // --- Stem: conv7x7/2 → BN → ReLU → maxpool3x3/2 ---
    let image = g.input("image", &[(batch * 3 * 224 * 224) as usize]);
    let conv1_w = g.parameter("conv1.weight", &[64 * 3 * 7 * 7]);
    let x = g.conv2d(image, conv1_w, batch, 3, s.h, s.w, 64, 7, 7, 2, 3);
    let s = s.after_conv(7, 2, 3); // 112x112

    // BN1 is fused at load time — we just store the fused bias
    let bn1_bias = g.parameter("bn1.fused_bias", &[(batch * 64 * s.h * s.w) as usize]);
    let x = g.add(x, bn1_bias);
    let x = g.relu(x);

    let x = g.max_pool_2d(x, batch, 64, s.h, s.w, 3, 3, 2, 1);
    let s = s.after_conv(3, 2, 1); // 56x56

    // --- Layer 1: 2 BasicBlocks, 64 channels, no downsample ---
    let (x, s) = basic_block(g, x, &s, batch, 64, 64, 1, "layer1.0");
    let (x, s) = basic_block(g, x, &s, batch, 64, 64, 1, "layer1.1");

    // --- Layer 2: 2 BasicBlocks, 128 channels, stride 2 ---
    let (x, s) = basic_block(g, x, &s, batch, 64, 128, 2, "layer2.0");
    let (x, s) = basic_block(g, x, &s, batch, 128, 128, 1, "layer2.1");

    // --- Layer 3: 2 BasicBlocks, 256 channels, stride 2 ---
    let (x, s) = basic_block(g, x, &s, batch, 128, 256, 2, "layer3.0");
    let (x, s) = basic_block(g, x, &s, batch, 256, 256, 1, "layer3.1");

    // --- Layer 4: 2 BasicBlocks, 512 channels, stride 2 ---
    let (x, s) = basic_block(g, x, &s, batch, 256, 512, 2, "layer4.0");
    let (x, _s) = basic_block(g, x, &s, batch, 512, 512, 1, "layer4.1");

    // --- Global average pool → FC ---
    let spatial = 7 * 7; // after all downsampling: 224 / 32 = 7
    let x = g.global_avg_pool(x, batch, 512, spatial);
    // x is now [batch * 512] (flat)

    // FC layer: [batch, 512] → [batch, 1000]
    let fc_w = g.parameter("fc.weight", &[512, 1000]);
    let fc_b = g.parameter("fc.bias", &[1000]);
    let logits = g.matmul(x, fc_w);
    g.bias_add(logits, fc_b)
}

/// BasicBlock: two 3x3 convolutions with residual connection.
///
/// If `stride > 1` or `in_c != out_c`, a 1x1 conv shortcut is used.
/// BatchNorm is fused into conv weights at load time.
fn basic_block(
    g: &mut Graph,
    x: NodeId,
    s: &Spatial,
    batch: u32,
    in_c: u32,
    out_c: u32,
    stride: u32,
    name: &str,
) -> (NodeId, Spatial) {
    let s1 = s.after_conv(3, stride, 1);

    // Conv1: 3x3, may downsample
    let w1 = g.parameter(
        &format!("{name}.conv1.weight"),
        &[(out_c * in_c * 3 * 3) as usize],
    );
    let h = g.conv2d(x, w1, batch, in_c, s.h, s.w, out_c, 3, 3, stride, 1);
    // BN1 fused
    let bn1_b = g.parameter(
        &format!("{name}.bn1.fused_bias"),
        &[(batch * out_c * s1.h * s1.w) as usize],
    );
    let h = g.add(h, bn1_b);
    let h = g.relu(h);

    // Conv2: 3x3, no stride
    let w2 = g.parameter(
        &format!("{name}.conv2.weight"),
        &[(out_c * out_c * 3 * 3) as usize],
    );
    let h = g.conv2d(h, w2, batch, out_c, s1.h, s1.w, out_c, 3, 3, 1, 1);
    // BN2 fused
    let bn2_b = g.parameter(
        &format!("{name}.bn2.fused_bias"),
        &[(batch * out_c * s1.h * s1.w) as usize],
    );
    let h = g.add(h, bn2_b);

    // Shortcut: identity or 1x1 conv
    let shortcut = if stride > 1 || in_c != out_c {
        let ds_w = g.parameter(
            &format!("{name}.downsample.0.weight"),
            &[(out_c * in_c) as usize],
        );
        let ds = g.conv2d(x, ds_w, batch, in_c, s.h, s.w, out_c, 1, 1, stride, 0);
        let ds_bn_b = g.parameter(
            &format!("{name}.downsample.1.fused_bias"),
            &[(batch * out_c * s1.h * s1.w) as usize],
        );
        g.add(ds, ds_bn_b)
    } else {
        x
    };

    let out = g.add(h, shortcut);
    let out = g.relu(out);
    (out, s1)
}

/// Names of all parameters in the model.
pub fn weight_names(batch: u32) -> Vec<String> {
    let mut names = Vec::new();
    names.push("conv1.weight".into());
    names.push("bn1.fused_bias".into());

    for (layer_idx, &(in_c, out_c, stride)) in [
        (64u32, 64u32, 1u32),
        (64, 64, 1),
        (64, 128, 2),
        (128, 128, 1),
        (128, 256, 2),
        (256, 256, 1),
        (256, 512, 2),
        (512, 512, 1),
    ]
    .iter()
    .enumerate()
    {
        let stage = layer_idx / 2 + 1;
        let block = layer_idx % 2;
        let name = format!("layer{stage}.{block}");

        names.push(format!("{name}.conv1.weight"));
        names.push(format!("{name}.bn1.fused_bias"));
        names.push(format!("{name}.conv2.weight"));
        names.push(format!("{name}.bn2.fused_bias"));

        if stride > 1 || in_c != out_c {
            names.push(format!("{name}.downsample.0.weight"));
            names.push(format!("{name}.downsample.1.fused_bias"));
        }
    }

    names.push("fc.weight".into());
    names.push("fc.bias".into());
    let _ = batch;
    names
}

/// Fuse BatchNorm parameters into the preceding conv's weight and a bias tensor.
///
/// Given conv weight `W [Co, Ci, kH, kW]` and BN params
/// `(scale, bias, mean, var, eps)`, returns `(W_fused, bias_fused)` where:
/// - `W_fused[co, ci, kh, kw] = W[co, ci, kh, kw] * scale[co] / sqrt(var[co] + eps)`
/// - `bias_fused[n, co, h, w] = bias[co] - mean[co] * scale[co] / sqrt(var[co] + eps)`
///   (broadcast to full spatial tensor for element-wise add)
pub fn fuse_bn_into_conv(
    conv_weight: &[f32],
    scale: &[f32],
    bias: &[f32],
    mean: &[f32],
    var: &[f32],
    eps: f32,
    out_channels: usize,
    kernel_size: usize,
    batch: usize,
    out_h: usize,
    out_w: usize,
) -> (Vec<f32>, Vec<f32>) {
    let in_channels = conv_weight.len() / (out_channels * kernel_size);

    // Fuse conv weight
    let mut w_fused = conv_weight.to_vec();
    for co in 0..out_channels {
        let inv_std = scale[co] / (var[co] + eps).sqrt();
        let start = co * in_channels * kernel_size;
        let end = start + in_channels * kernel_size;
        for v in &mut w_fused[start..end] {
            *v *= inv_std;
        }
    }

    // Fuse bias (broadcast to full spatial)
    let spatial = out_h * out_w;
    let full_size = batch * out_channels * spatial;
    let mut b_fused = vec![0.0f32; full_size];
    for n in 0..batch {
        for co in 0..out_channels {
            let inv_std = scale[co] / (var[co] + eps).sqrt();
            let b = bias[co] - mean[co] * inv_std;
            for s in 0..spatial {
                b_fused[(n * out_channels + co) * spatial + s] = b;
            }
        }
    }

    (w_fused, b_fused)
}

// ---------------------------------------------------------------------------
// ResNet-50 (Bottleneck blocks)
// ---------------------------------------------------------------------------

/// Build the ResNet-50 inference graph.
///
/// Same stem as ResNet-18 but uses Bottleneck blocks (1x1 → 3x3 → 1x1)
/// with 4x expansion. Output: `[batch, 1000]`.
pub fn build_resnet50(g: &mut Graph, batch: u32) -> NodeId {
    let s = Spatial { h: 224, w: 224 };

    // Stem: identical to ResNet-18
    let image = g.input("image", &[(batch * 3 * 224 * 224) as usize]);
    let conv1_w = g.parameter("conv1.weight", &[64 * 3 * 7 * 7]);
    let x = g.conv2d(image, conv1_w, batch, 3, s.h, s.w, 64, 7, 7, 2, 3);
    let s = s.after_conv(7, 2, 3);
    let bn1_bias = g.parameter("bn1.fused_bias", &[(batch * 64 * s.h * s.w) as usize]);
    let x = g.add(x, bn1_bias);
    let x = g.relu(x);
    let x = g.max_pool_2d(x, batch, 64, s.h, s.w, 3, 3, 2, 1);
    let s = s.after_conv(3, 2, 1);

    // Layer 1: 3 Bottleneck blocks, 64→256 (stride 1, but needs downsample for expansion)
    let (x, s) = bottleneck(g, x, &s, batch, 64, 64, 256, 1, "layer1.0");
    let (x, s) = bottleneck(g, x, &s, batch, 256, 64, 256, 1, "layer1.1");
    let (x, s) = bottleneck(g, x, &s, batch, 256, 64, 256, 1, "layer1.2");

    // Layer 2: 4 Bottleneck blocks, 256→512 (first stride 2)
    let (x, s) = bottleneck(g, x, &s, batch, 256, 128, 512, 2, "layer2.0");
    let (x, s) = bottleneck(g, x, &s, batch, 512, 128, 512, 1, "layer2.1");
    let (x, s) = bottleneck(g, x, &s, batch, 512, 128, 512, 1, "layer2.2");
    let (x, s) = bottleneck(g, x, &s, batch, 512, 128, 512, 1, "layer2.3");

    // Layer 3: 6 Bottleneck blocks, 512→1024 (first stride 2)
    let (x, s) = bottleneck(g, x, &s, batch, 512, 256, 1024, 2, "layer3.0");
    let (x, s) = bottleneck(g, x, &s, batch, 1024, 256, 1024, 1, "layer3.1");
    let (x, s) = bottleneck(g, x, &s, batch, 1024, 256, 1024, 1, "layer3.2");
    let (x, s) = bottleneck(g, x, &s, batch, 1024, 256, 1024, 1, "layer3.3");
    let (x, s) = bottleneck(g, x, &s, batch, 1024, 256, 1024, 1, "layer3.4");
    let (x, s) = bottleneck(g, x, &s, batch, 1024, 256, 1024, 1, "layer3.5");

    // Layer 4: 3 Bottleneck blocks, 1024→2048 (first stride 2)
    let (x, s) = bottleneck(g, x, &s, batch, 1024, 512, 2048, 2, "layer4.0");
    let (x, s) = bottleneck(g, x, &s, batch, 2048, 512, 2048, 1, "layer4.1");
    let (x, _) = bottleneck(g, x, &s, batch, 2048, 512, 2048, 1, "layer4.2");

    // GAP → FC
    let x = g.global_avg_pool(x, batch, 2048, 7 * 7);
    let fc_w = g.parameter("fc.weight", &[2048, 1000]);
    let fc_b = g.parameter("fc.bias", &[1000]);
    let logits = g.matmul(x, fc_w);
    g.bias_add(logits, fc_b)
}

/// Bottleneck block: conv1x1 → BN → ReLU → conv3x3 → BN → ReLU → conv1x1 → BN → residual → ReLU.
///
/// `mid_c` is the bottleneck width (e.g., 64 for layer1). `out_c` = `mid_c * 4`.
fn bottleneck(
    g: &mut Graph,
    x: NodeId,
    s: &Spatial,
    batch: u32,
    in_c: u32,
    mid_c: u32,
    out_c: u32,
    stride: u32,
    name: &str,
) -> (NodeId, Spatial) {
    // Conv1: 1x1, reduce channels
    let w1 = g.parameter(&format!("{name}.conv1.weight"), &[(mid_c * in_c) as usize]);
    let h = g.conv2d(x, w1, batch, in_c, s.h, s.w, mid_c, 1, 1, 1, 0);
    let bn1_b = g.parameter(
        &format!("{name}.bn1.fused_bias"),
        &[(batch * mid_c * s.h * s.w) as usize],
    );
    let h = g.add(h, bn1_b);
    let h = g.relu(h);

    // Conv2: 3x3, may downsample
    let s1 = s.after_conv(3, stride, 1);
    let w2 = g.parameter(
        &format!("{name}.conv2.weight"),
        &[(mid_c * mid_c * 9) as usize],
    );
    let h = g.conv2d(h, w2, batch, mid_c, s.h, s.w, mid_c, 3, 3, stride, 1);
    let bn2_b = g.parameter(
        &format!("{name}.bn2.fused_bias"),
        &[(batch * mid_c * s1.h * s1.w) as usize],
    );
    let h = g.add(h, bn2_b);
    let h = g.relu(h);

    // Conv3: 1x1, expand channels
    let w3 = g.parameter(&format!("{name}.conv3.weight"), &[(out_c * mid_c) as usize]);
    let h = g.conv2d(h, w3, batch, mid_c, s1.h, s1.w, out_c, 1, 1, 1, 0);
    let bn3_b = g.parameter(
        &format!("{name}.bn3.fused_bias"),
        &[(batch * out_c * s1.h * s1.w) as usize],
    );
    let h = g.add(h, bn3_b);

    // Shortcut: always downsample when in_c != out_c or stride > 1
    let shortcut = if stride > 1 || in_c != out_c {
        let ds_w = g.parameter(
            &format!("{name}.downsample.0.weight"),
            &[(out_c * in_c) as usize],
        );
        let ds = g.conv2d(x, ds_w, batch, in_c, s.h, s.w, out_c, 1, 1, stride, 0);
        let ds_bn_b = g.parameter(
            &format!("{name}.downsample.1.fused_bias"),
            &[(batch * out_c * s1.h * s1.w) as usize],
        );
        g.add(ds, ds_bn_b)
    } else {
        x
    };

    let out = g.add(h, shortcut);
    let out = g.relu(out);
    (out, s1)
}
