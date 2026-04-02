/// Compare ResNet mini-model output against PyTorch reference.
///
/// Run `python3 scripts/gen_reference.py` first to generate the reference.
use meganeura::{Graph, build_inference_session};

fn parse_f32_array(json: &str, key: &str) -> Vec<f32> {
    let needle = format!("\"{key}\": [");
    let start = json.find(&needle).expect(key) + needle.len();
    let end = start + json[start..].find(']').unwrap();
    json[start..end]
        .split(',')
        .map(|s| s.trim().parse::<f32>().unwrap())
        .collect()
}

fn expand_channel_bias(bias: &[f32], spatial: usize) -> Vec<f32> {
    let c = bias.len();
    let mut out = vec![0.0f32; c * spatial];
    for ch in 0..c {
        for s in 0..spatial {
            out[ch * spatial + s] = bias[ch];
        }
    }
    out
}

fn transpose_2d(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

#[test]
fn resnet_mini_matches_pytorch() {
    let path = "bench/results/resnet_reference.json";
    if !std::path::Path::new(path).exists() {
        eprintln!("SKIP: {path} not found (run: python3 scripts/gen_reference.py)");
        return;
    }
    let json = std::fs::read_to_string(path).unwrap();

    let input = parse_f32_array(&json, "input");
    let expected = parse_f32_array(&json, "output");

    let batch = 1u32;
    let (h, w) = (64u32, 64u32);

    let mut g = Graph::new();
    let image = g.input("image", &[(batch * 3 * h * w) as usize]);

    // Conv1: 7x7, stride=2, padding=3, 3→64
    let conv1_w = g.parameter("conv1_weight", &[64 * 3 * 7 * 7]);
    let x = g.conv2d(image, conv1_w, batch, 3, h, w, 64, 7, 7, 2, 3);
    let oh = (h + 6 - 7) / 2 + 1; // 32
    let ow = (w + 6 - 7) / 2 + 1;
    let bn1_b = g.parameter("bn1_bias", &[(batch * 64 * oh * ow) as usize]);
    let x = g.add(x, bn1_b);
    let x = g.relu(x);

    // MaxPool: 3x3, stride=2, padding=1
    let x = g.max_pool_2d(x, batch, 64, oh, ow, 3, 3, 2, 1);
    let ph = (oh + 2 - 3) / 2 + 1; // 16
    let pw = (ow + 2 - 3) / 2 + 1;

    // Block: conv→bn→relu → conv→bn → residual+relu
    let bw1 = g.parameter("bconv1_w", &[64 * 64 * 9]);
    let bx = g.conv2d(x, bw1, batch, 64, ph, pw, 64, 3, 3, 1, 1);
    let bb1 = g.parameter("bbn1_b", &[(batch * 64 * ph * pw) as usize]);
    let bx = g.add(bx, bb1);
    let bx = g.relu(bx);

    let bw2 = g.parameter("bconv2_w", &[64 * 64 * 9]);
    let bx = g.conv2d(bx, bw2, batch, 64, ph, pw, 64, 3, 3, 1, 1);
    let bb2 = g.parameter("bbn2_b", &[(batch * 64 * ph * pw) as usize]);
    let bx = g.add(bx, bb2);
    let x = g.add(bx, x);
    let x = g.relu(x);

    // GAP → FC
    let x = g.global_avg_pool(x, batch, 64, ph * pw);
    let fc_w = g.parameter("fc_w", &[64, 10]);
    let fc_b = g.parameter("fc_b", &[10]);
    let logits = g.matmul(x, fc_w);
    let logits = g.bias_add(logits, fc_b);
    g.set_outputs(vec![logits]);

    let mut session = build_inference_session(&g);

    // Load weights
    session.set_parameter("conv1_weight", &parse_f32_array(&json, "conv1_weight"));
    session.set_parameter(
        "bn1_bias",
        &expand_channel_bias(&parse_f32_array(&json, "bn1_bias"), (oh * ow) as usize),
    );
    session.set_parameter("bconv1_w", &parse_f32_array(&json, "block_conv1_weight"));
    session.set_parameter(
        "bbn1_b",
        &expand_channel_bias(
            &parse_f32_array(&json, "block_bn1_bias"),
            (ph * pw) as usize,
        ),
    );
    session.set_parameter("bconv2_w", &parse_f32_array(&json, "block_conv2_weight"));
    session.set_parameter(
        "bbn2_b",
        &expand_channel_bias(
            &parse_f32_array(&json, "block_bn2_bias"),
            (ph * pw) as usize,
        ),
    );
    session.set_parameter(
        "fc_w",
        &transpose_2d(&parse_f32_array(&json, "fc_weight"), 10, 64),
    );
    session.set_parameter("fc_b", &parse_f32_array(&json, "fc_bias"));

    session.set_input("image", &input);
    session.step();
    session.wait();

    let output = session.read_output(10);

    eprintln!("PyTorch:   {:?}", &expected[..5]);
    eprintln!("Meganeura: {:?}", &output[..5]);

    let max_diff = output
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("max diff: {:.6e}", max_diff);

    assert!(
        max_diff < 0.1,
        "ResNet output diverges from PyTorch: max_diff={max_diff:.6e}"
    );
}
