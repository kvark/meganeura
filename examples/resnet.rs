#![allow(dead_code, clippy::too_many_arguments)]
/// Run ResNet-18 inference using meganeura.
///
/// Downloads weights from HuggingFace, builds the computation graph
/// manually, fuses BatchNorm into conv weights, and classifies a
/// synthetic image.
///
/// Usage:
///   cargo run --release --example resnet
#[allow(dead_code, clippy::too_many_arguments)]
use meganeura::{Graph, NodeId, build_inference_session, data::safetensors::SafeTensorsModel};

const REPO_ID: &str = "microsoft/resnet-18";

// ---------------------------------------------------------------------------
// Model: ResNet-18
// ---------------------------------------------------------------------------

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

fn build_graph(g: &mut Graph, batch: u32) -> NodeId {
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

fn weight_names(batch: u32) -> Vec<String> {
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

fn fuse_bn_into_conv(
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
// Example main
// ---------------------------------------------------------------------------

fn main() {
    env_logger::init();

    let batch = 1u32;

    // --- Build graph ---
    println!("building ResNet-18 graph...");
    let mut g = Graph::new();
    let logits = build_graph(&mut g, batch);
    g.set_outputs(vec![logits]);

    // --- Compile ---
    println!("compiling...");
    let mut session = build_inference_session(&g);
    println!(
        "  {} buffers, {} dispatches",
        session.plan().buffers.len(),
        session.plan().dispatches.len()
    );

    // --- Download and load weights ---
    println!("downloading {} weights...", REPO_ID);
    let model = SafeTensorsModel::download(REPO_ID).expect("download failed");

    println!("model tensors:");
    let mut names: Vec<_> = model.tensor_info().keys().collect();
    names.sort();
    for name in names.iter().take(10) {
        let info = &model.tensor_info()[*name];
        println!("  {}: shape={:?}", name, info.shape);
    }
    if names.len() > 10 {
        println!("  ... and {} more", names.len() - 10);
    }

    // Load conv weights with BatchNorm fusion
    println!("loading weights (fusing BatchNorm)...");
    load_resnet_weights(&mut session, &model, batch);

    // --- Inference on synthetic image ---
    let image: Vec<f32> = (0..3 * 224 * 224)
        .map(|i| {
            let pixel = ((i * 7 + 13) % 256) as f32 / 255.0;
            let ch = i / (224 * 224);
            let mean = [0.485, 0.456, 0.406][ch];
            let std = [0.229, 0.224, 0.225][ch];
            (pixel - mean) / std
        })
        .collect();

    session.set_input("image", &image);
    session.step();
    session.wait();

    let logits = session.read_output(1000);
    println!("\ntop-5 predictions (synthetic image):");
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    for (rank, &(class_id, logit)) in indexed.iter().take(5).enumerate() {
        println!("  #{}: class {} (logit {:.3})", rank + 1, class_id, logit);
    }
}

/// Load torchvision ResNet-18 weights, fusing BatchNorm into conv at load time.
///
/// The HuggingFace `microsoft/resnet-18` model uses torchvision naming:
/// `conv1.weight`, `bn1.weight`, `bn1.bias`, `bn1.running_mean`, `bn1.running_var`,
/// `layer1.0.conv1.weight`, `layer1.0.bn1.weight`, etc.
///
/// Since our graph expects `conv1.weight` (pre-fused) and `bn1.fused_bias`,
/// we fuse at load time: W_fused = W * scale/sqrt(var+eps), b_fused = b - mean*w.
fn load_resnet_weights(session: &mut meganeura::Session, model: &SafeTensorsModel, batch: u32) {
    let eps = 1e-5f32;

    // Helper: load, fuse, and set conv+bn parameters
    let fuse_and_load = |session: &mut meganeura::Session,
                         model: &SafeTensorsModel,
                         conv_name: &str,
                         bn_name: &str,
                         out_c: usize,
                         spatial: usize| {
        let w = model.tensor_f32_auto(conv_name).expect(conv_name);
        let scale = model
            .tensor_f32_auto(&format!("{bn_name}.weight"))
            .expect("bn weight");
        let bias = model
            .tensor_f32_auto(&format!("{bn_name}.bias"))
            .expect("bn bias");
        let mean = model
            .tensor_f32_auto(&format!("{bn_name}.running_mean"))
            .expect("bn mean");
        let var = model
            .tensor_f32_auto(&format!("{bn_name}.running_var"))
            .expect("bn var");

        let (w_fused, b_fused) = fuse_bn_into_conv(
            &w,
            &scale,
            &bias,
            &mean,
            &var,
            eps,
            out_c,
            0,
            batch as usize,
            0,
            spatial,
        );

        session.set_parameter(conv_name, &w_fused);
        session.set_parameter(&format!("{bn_name}.fused_bias"), &b_fused);
    };

    // Stem
    fuse_and_load(session, model, "conv1.weight", "bn1", 64, 112 * 112);

    // Residual blocks
    for (stage, channels, first_stride) in &[(1, 64, 1), (2, 128, 2), (3, 256, 2), (4, 512, 2)] {
        for block in 0..2 {
            let name = format!("layer{stage}.{block}");
            let in_c = if block == 0 && *stage > 1 {
                channels / 2
            } else {
                *channels
            };
            let _ = in_c;
            let stride = if block == 0 { *first_stride } else { 1 };

            fuse_and_load(
                session,
                model,
                &format!("{name}.conv1.weight"),
                &format!("{name}.bn1"),
                *channels,
                0, // spatial determined by fuse_bn_into_conv
            );
            fuse_and_load(
                session,
                model,
                &format!("{name}.conv2.weight"),
                &format!("{name}.bn2"),
                *channels,
                0,
            );

            // Downsample shortcut
            if stride > 1 || (block == 0 && *stage > 1) {
                fuse_and_load(
                    session,
                    model,
                    &format!("{name}.downsample.0.weight"),
                    &format!("{name}.downsample.1"),
                    *channels,
                    0,
                );
            }
        }
    }

    // FC layer (needs transposing: HF stores [out, in])
    let fc_w = model
        .tensor_f32_auto_transposed("fc.weight")
        .expect("fc weight");
    session.set_parameter("fc.weight", &fc_w);
    let fc_b = model.tensor_f32_auto("fc.bias").expect("fc bias");
    session.set_parameter("fc.bias", &fc_b);
}
