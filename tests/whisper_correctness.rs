/// Compare Whisper mini-encoder output against PyTorch reference.
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
fn whisper_mini_encoder_matches_pytorch() {
    let path = "bench/results/whisper_reference.json";
    if !std::path::Path::new(path).exists() {
        eprintln!("SKIP: {path} not found (run: python3 scripts/gen_reference.py)");
        return;
    }
    let json = std::fs::read_to_string(path).unwrap();

    let input = parse_f32_array(&json, "input");
    let expected = parse_f32_array(&json, "output");
    let d = parse_int(&json, "d_model");
    let n_heads = parse_int(&json, "n_heads") as u32;
    let n_mels = parse_int(&json, "n_mels") as u32;
    let mel_len = parse_int(&json, "mel_len") as u32;
    let seq_len = parse_int(&json, "seq_len");
    let ffn_dim = parse_int(&json, "ffn_dim");

    let batch = 1u32;

    let mut g = Graph::new();
    let mel = g.input("mel", &[(batch * n_mels * mel_len) as usize]);

    // Conv1d emulated as Conv2d with W=1, temporal axis on H.
    // Conv1: n_mels → d, kernel=3, stride=1, padding=1
    let conv1_w = g.parameter("conv1.weight", &[d * n_mels as usize * 3]);
    let conv1_b = g.parameter("conv1.bias_expanded", &[d * mel_len as usize]);
    let x = g.conv2d_hw(
        mel, conv1_w, batch, n_mels, mel_len, 1, d as u32, 3, 1, 1, 1, 0,
    );
    let x = g.add(x, conv1_b);
    let x = g.gelu(x);

    // Conv2: d → d, kernel=3, stride=2, padding_h=1, padding_w=0
    let conv2_w = g.parameter("conv2.weight", &[d * d * 3]);
    let conv2_b = g.parameter("conv2.bias_expanded", &[d * seq_len]);
    let x = g.conv2d_hw(
        x, conv2_w, batch, d as u32, mel_len, 1, d as u32, 3, 1, 2, 1, 0,
    );
    let x = g.add(x, conv2_b);
    let x = g.gelu(x);

    // Reshape [d * seq_len] → [d, seq_len], then transpose to [seq_len, d]
    let x = g.reshape(x, &[d, seq_len]);
    let x = g.transpose(x);

    // One transformer layer (pre-norm)
    let ln1_w = g.parameter("ln1.weight", &[d]);
    let ln1_b = g.parameter("ln1.bias", &[d]);
    let h = g.layer_norm(x, ln1_w, ln1_b, 1e-5);

    let wq = g.parameter("q.weight", &[d, d]);
    let wk = g.parameter("k.weight", &[d, d]);
    let wv = g.parameter("v.weight", &[d, d]);
    let q_b = g.parameter("q.bias", &[d]);
    let v_b = g.parameter("v.bias", &[d]);

    let q = g.matmul(h, wq);
    let q = g.bias_add(q, q_b);
    let k = g.matmul(h, wk);
    let v = g.matmul(h, wv);
    let v = g.bias_add(v, v_b);

    let head_dim = d as u32 / n_heads;
    let attn = g.full_attention(q, k, v, n_heads, n_heads, head_dim);

    let wo = g.parameter("out.weight", &[d, d]);
    let wo_b = g.parameter("out.bias", &[d]);
    let attn_out = g.matmul(attn, wo);
    let attn_out = g.bias_add(attn_out, wo_b);
    let x = g.add(x, attn_out);

    // FFN
    let ln2_w = g.parameter("ln2.weight", &[d]);
    let ln2_b = g.parameter("ln2.bias", &[d]);
    let h = g.layer_norm(x, ln2_w, ln2_b, 1e-5);
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

    // Load weights from reference
    session.set_parameter("conv1.weight", &parse_f32_array(&json, "conv1.weight"));
    session.set_parameter(
        "conv1.bias_expanded",
        &expand_channel_bias(&parse_f32_array(&json, "conv1.bias"), mel_len as usize),
    );
    session.set_parameter("conv2.weight", &parse_f32_array(&json, "conv2.weight"));
    session.set_parameter(
        "conv2.bias_expanded",
        &expand_channel_bias(&parse_f32_array(&json, "conv2.bias"), seq_len),
    );

    // Linear weights need transposing [out, in] → [in, out]
    session.set_parameter("ln1.weight", &parse_f32_array(&json, "ln1.weight"));
    session.set_parameter("ln1.bias", &parse_f32_array(&json, "ln1.bias"));
    session.set_parameter(
        "q.weight",
        &transpose_2d(&parse_f32_array(&json, "q.weight"), d, d),
    );
    session.set_parameter("q.bias", &parse_f32_array(&json, "q.bias"));
    session.set_parameter(
        "k.weight",
        &transpose_2d(&parse_f32_array(&json, "k.weight"), d, d),
    );
    session.set_parameter(
        "v.weight",
        &transpose_2d(&parse_f32_array(&json, "v.weight"), d, d),
    );
    session.set_parameter("v.bias", &parse_f32_array(&json, "v.bias"));
    session.set_parameter(
        "out.weight",
        &transpose_2d(&parse_f32_array(&json, "out.weight"), d, d),
    );
    session.set_parameter("out.bias", &parse_f32_array(&json, "out.bias"));
    session.set_parameter("ln2.weight", &parse_f32_array(&json, "ln2.weight"));
    session.set_parameter("ln2.bias", &parse_f32_array(&json, "ln2.bias"));
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
        max_diff < 0.1,
        "Whisper encoder output diverges from PyTorch: max_diff={max_diff:.6e}"
    );
}
