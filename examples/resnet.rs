/// Run ResNet-18 inference using meganeura.
///
/// Downloads weights from HuggingFace, builds the computation graph
/// manually, fuses BatchNorm into conv weights, and classifies a
/// synthetic image.
///
/// Usage:
///   cargo run --release --example resnet
use meganeura::{
    Graph, build_inference_session, data::safetensors::SafeTensorsModel, models::resnet,
};

const REPO_ID: &str = "microsoft/resnet-18";

fn main() {
    env_logger::init();

    let batch = 1u32;

    // --- Build graph ---
    println!("building ResNet-18 graph...");
    let mut g = Graph::new();
    let logits = resnet::build_graph(&mut g, batch);
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
/// Map torchvision ResNet names to HuggingFace transformers names.
fn hf_name(torchvision_name: &str) -> String {
    // Stem: conv1 → resnet.embedder.embedder.convolution
    //       bn1  → resnet.embedder.embedder.normalization
    if torchvision_name == "conv1.weight" {
        return "resnet.embedder.embedder.convolution.weight".to_string();
    }
    if let Some(suffix) = torchvision_name.strip_prefix("bn1.") {
        return format!("resnet.embedder.embedder.normalization.{suffix}");
    }
    // FC: fc.weight/bias → classifier.1.weight/bias
    if let Some(suffix) = torchvision_name.strip_prefix("fc.") {
        return format!("classifier.1.{suffix}");
    }
    // Blocks: layer{s}.{b}.conv{i}.weight → resnet.encoder.stages.{s-1}.layers.{b}.layer.{i-1}.convolution.weight
    //         layer{s}.{b}.bn{i}.X        → resnet.encoder.stages.{s-1}.layers.{b}.layer.{i-1}.normalization.X
    //         layer{s}.{b}.downsample.0.weight → resnet.encoder.stages.{s-1}.layers.{b}.shortcut.convolution.weight
    //         layer{s}.{b}.downsample.1.X      → resnet.encoder.stages.{s-1}.layers.{b}.shortcut.normalization.X
    if let Some(rest) = torchvision_name.strip_prefix("layer") {
        let parts: Vec<&str> = rest.splitn(2, '.').collect();
        if parts.len() == 2 {
            let stage = parts[0].parse::<usize>().unwrap_or(1) - 1;
            let remainder = parts[1]; // e.g. "0.conv1.weight" or "0.downsample.0.weight"
            let block_parts: Vec<&str> = remainder.splitn(2, '.').collect();
            if block_parts.len() == 2 {
                let block = block_parts[0];
                let field = block_parts[1]; // e.g. "conv1.weight" or "downsample.0.weight"
                if let Some(suffix) = field.strip_prefix("conv") {
                    // conv1.weight → layer.0.convolution.weight, conv2.weight → layer.1.convolution.weight
                    let idx_rest: Vec<&str> = suffix.splitn(2, '.').collect();
                    let layer_idx = idx_rest[0].parse::<usize>().unwrap_or(1) - 1;
                    let attr = idx_rest.get(1).unwrap_or(&"weight");
                    return format!(
                        "resnet.encoder.stages.{stage}.layers.{block}.layer.{layer_idx}.convolution.{attr}"
                    );
                }
                if let Some(suffix) = field.strip_prefix("bn") {
                    let idx_rest: Vec<&str> = suffix.splitn(2, '.').collect();
                    let layer_idx = idx_rest[0].parse::<usize>().unwrap_or(1) - 1;
                    let attr = idx_rest.get(1).unwrap_or(&"weight");
                    return format!(
                        "resnet.encoder.stages.{stage}.layers.{block}.layer.{layer_idx}.normalization.{attr}"
                    );
                }
                if let Some(suffix) = field.strip_prefix("downsample.0.") {
                    return format!(
                        "resnet.encoder.stages.{stage}.layers.{block}.shortcut.convolution.{suffix}"
                    );
                }
                if let Some(suffix) = field.strip_prefix("downsample.1.") {
                    return format!(
                        "resnet.encoder.stages.{stage}.layers.{block}.shortcut.normalization.{suffix}"
                    );
                }
            }
        }
    }
    // Fallback: return as-is
    torchvision_name.to_string()
}

fn load_resnet_weights(session: &mut meganeura::Session, model: &SafeTensorsModel, batch: u32) {
    let eps = 1e-5f32;

    // Helper: load, fuse, and set conv+bn parameters
    let fuse_and_load = |session: &mut meganeura::Session,
                         model: &SafeTensorsModel,
                         conv_name: &str,
                         bn_name: &str,
                         out_c: usize,
                         spatial: usize| {
        let hf_conv = hf_name(conv_name);
        let w = model.tensor_f32_auto(&hf_conv).expect(conv_name);
        let scale = model
            .tensor_f32_auto(&hf_name(&format!("{bn_name}.weight")))
            .expect("bn weight");
        let bias = model
            .tensor_f32_auto(&hf_name(&format!("{bn_name}.bias")))
            .expect("bn bias");
        let mean = model
            .tensor_f32_auto(&hf_name(&format!("{bn_name}.running_mean")))
            .expect("bn mean");
        let var = model
            .tensor_f32_auto(&hf_name(&format!("{bn_name}.running_var")))
            .expect("bn var");

        // Infer kernel spatial size from weight tensor shape (Co, Ci, kH, kW)
        let info = &model.tensor_info()[&hf_conv];
        let kernel_hw = if info.shape.len() == 4 {
            info.shape[2] * info.shape[3]
        } else {
            1
        };

        let (w_fused, b_fused) = resnet::fuse_bn_into_conv(
            &w,
            &scale,
            &bias,
            &mean,
            &var,
            eps,
            out_c,
            kernel_hw,
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
        .tensor_f32_auto_transposed(&hf_name("fc.weight"))
        .expect("fc weight");
    session.set_parameter("fc.weight", &fc_w);
    let fc_b = model.tensor_f32_auto(&hf_name("fc.bias")).expect("fc bias");
    session.set_parameter("fc.bias", &fc_b);
}
