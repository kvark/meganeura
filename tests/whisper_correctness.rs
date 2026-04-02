/// Compare Whisper conv stem + FFN against PyTorch reference.
///
/// Tests Conv1d (via Conv2d H=mel_len,W=1), GELU, transpose, LayerNorm,
/// and Linear — the ops unique to Whisper. Attention is verified by
/// existing gpu_smoke tests.
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

fn parse_int(json: &str, key: &str) -> usize {
    let needle = format!("\"{key}\": ");
    let start = json.find(&needle).expect(key) + needle.len();
    let end = start + json[start..].find(|c: char| !c.is_ascii_digit()).unwrap();
    json[start..end].parse().unwrap()
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
fn whisper_conv_stem_ffn_matches_pytorch() {
    let path = "bench/results/whisper_reference.json";
    if !std::path::Path::new(path).exists() {
        eprintln!("SKIP: {path} not found (run: python3 scripts/gen_reference.py)");
        return;
    }
    let json = std::fs::read_to_string(path).unwrap();

    let input = parse_f32_array(&json, "input");
    let expected = parse_f32_array(&json, "output");
    let d = parse_int(&json, "d_model");
    let n_mels = parse_int(&json, "n_mels") as u32;
    let mel_len = parse_int(&json, "mel_len") as u32;
    let seq_len = parse_int(&json, "seq_len");
    let ffn_dim = parse_int(&json, "ffn_dim");

    let batch = 1u32;
    let mut g = Graph::new();
    let mel = g.input("mel", &[(batch * n_mels * mel_len) as usize]);

    // Conv1: n_mels → d, kernel=3, stride=1, padding_h=1 (temporal on H, W=1)
    let w1 = g.parameter("conv1.weight", &[d * n_mels as usize * 3]);
    let b1 = g.parameter("conv1.bias_exp", &[d * mel_len as usize]);
    let x = g.conv2d_hw(mel, w1, batch, n_mels, mel_len, 1, d as u32, 3, 1, 1, 1, 0);
    let x = g.add(x, b1);
    let x = g.gelu(x);

    // Conv2: d → d, kernel=3, stride=2, padding_h=1
    let w2 = g.parameter("conv2.weight", &[d * d * 3]);
    let b2 = g.parameter("conv2.bias_exp", &[d * seq_len]);
    let x = g.conv2d_hw(x, w2, batch, d as u32, mel_len, 1, d as u32, 3, 1, 2, 1, 0);
    let x = g.add(x, b2);
    let x = g.gelu(x);

    // Reshape + transpose: [d * seq_len] → [d, seq_len] → [seq_len, d]
    let x = g.reshape(x, &[d, seq_len]);
    let x = g.transpose(x);

    // LayerNorm + GELU FFN + residual
    let ln_w = g.parameter("ln.weight", &[d]);
    let ln_b = g.parameter("ln.bias", &[d]);
    let h = g.layer_norm(x, ln_w, ln_b, 1e-5);

    let fc1_w = g.parameter("fc1.weight", &[d, ffn_dim]);
    let fc1_b = g.parameter("fc1.bias", &[ffn_dim]);
    let h = g.matmul(h, fc1_w);
    let h = g.bias_add(h, fc1_b);
    let h = g.gelu(h);

    let fc2_w = g.parameter("fc2.weight", &[ffn_dim, d]);
    let fc2_b = g.parameter("fc2.bias", &[d]);
    let h = g.matmul(h, fc2_w);
    let h = g.bias_add(h, fc2_b);
    let x = g.add(x, h);

    // Final LN
    let fln_w = g.parameter("final_ln.weight", &[d]);
    let fln_b = g.parameter("final_ln.bias", &[d]);
    let x = g.layer_norm(x, fln_w, fln_b, 1e-5);
    g.set_outputs(vec![x]);

    let mut session = build_inference_session(&g);

    // Load weights
    session.set_parameter("conv1.weight", &parse_f32_array(&json, "conv1.weight"));
    session.set_parameter(
        "conv1.bias_exp",
        &expand_channel_bias(&parse_f32_array(&json, "conv1.bias"), mel_len as usize),
    );
    session.set_parameter("conv2.weight", &parse_f32_array(&json, "conv2.weight"));
    session.set_parameter(
        "conv2.bias_exp",
        &expand_channel_bias(&parse_f32_array(&json, "conv2.bias"), seq_len),
    );
    session.set_parameter("ln.weight", &parse_f32_array(&json, "ln.weight"));
    session.set_parameter("ln.bias", &parse_f32_array(&json, "ln.bias"));
    session.set_parameter(
        "fc1.weight",
        &transpose_2d(&parse_f32_array(&json, "fc1.weight"), ffn_dim, d),
    );
    session.set_parameter("fc1.bias", &parse_f32_array(&json, "fc1.bias"));
    session.set_parameter(
        "fc2.weight",
        &transpose_2d(&parse_f32_array(&json, "fc2.weight"), d, ffn_dim),
    );
    session.set_parameter("fc2.bias", &parse_f32_array(&json, "fc2.bias"));
    session.set_parameter(
        "final_ln.weight",
        &parse_f32_array(&json, "final_ln.weight"),
    );
    session.set_parameter("final_ln.bias", &parse_f32_array(&json, "final_ln.bias"));

    session.set_input("mel", &input);
    session.step();
    session.wait();

    let output = session.read_output(seq_len * d);

    eprintln!("PyTorch:   {:?}", &expected[..5]);
    eprintln!("Meganeura: {:?}", &output[..5]);

    let max_diff = output
        .iter()
        .zip(expected.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("max diff: {:.6e}", max_diff);

    assert!(
        max_diff < 0.01,
        "Whisper conv stem + FFN diverges from PyTorch: max_diff={max_diff:.6e}"
    );
}
