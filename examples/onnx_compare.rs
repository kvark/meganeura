/// Compare native and ONNX-imported model inference.
///
/// Loads the same model weights into two graphs:
/// 1. Native: hand-built Meganeura graph (like examples/huggingface.rs)
/// 2. ONNX: imported from an ONNX file via the new loader
///
/// Runs both on the same input and compares logits numerically.
///
/// Usage:
///   # First, export the ONNX model:
///   pip install torch safetensors huggingface_hub onnx
///   python scripts/export_onnx.py mnist-mlp
///
///   # Then run the comparison:
///   cargo run --release --example onnx_compare
///
/// Expects:
///   - models/onnx/mnist-mlp.onnx  (from the export script)
///   - data/t10k-images-idx3-ubyte.gz  (MNIST test set, optional for accuracy test)
use meganeura::{
    Graph, build_inference_session, data::safetensors::SafeTensorsModel, load::onnx::load_onnx,
};
use std::path::Path;

fn main() {
    env_logger::init();

    let onnx_path = Path::new("models/onnx/mnist-mlp.onnx");
    if !onnx_path.exists() {
        eprintln!(
            "ONNX model not found at {}.\n\
             Run: python scripts/export_onnx.py mnist-mlp",
            onnx_path.display()
        );
        std::process::exit(1);
    }

    // ─── Load ONNX model ───
    println!("loading ONNX model from {}...", onnx_path.display());
    let onnx_model = load_onnx(onnx_path).expect("failed to load ONNX model");
    println!(
        "  ONNX graph: {} nodes, {} weights",
        onnx_model.graph.nodes().len(),
        onnx_model.weights.len()
    );
    println!("  ONNX weights:");
    for (name, data) in &onnx_model.weights {
        println!("    {}: {} elements", name, data.len());
    }

    // ─── Build native graph (same architecture) ───
    println!("\nbuilding native graph...");
    let (native_graph, native_weight_map) = build_native_mnist_mlp();

    // ─── Compile both ───
    println!("compiling ONNX session...");
    let mut onnx_session = build_inference_session(&onnx_model.graph);
    println!(
        "  {} buffers, {} dispatches",
        onnx_session.plan().buffers.len(),
        onnx_session.plan().dispatches.len()
    );

    println!("compiling native session...");
    let mut native_session = build_inference_session(&native_graph);
    println!(
        "  {} buffers, {} dispatches",
        native_session.plan().buffers.len(),
        native_session.plan().dispatches.len()
    );

    // ─── Load weights into both sessions ───
    // ONNX session: weights come from the ONNX file
    println!("\nloading ONNX weights...");
    for (name, _) in onnx_session.plan().param_buffers.clone() {
        if let Some(data) = onnx_model.weights.get(&name) {
            onnx_session.set_parameter(&name, data);
        } else {
            eprintln!("  WARNING: ONNX weight '{}' not found in model", name);
        }
    }

    // Native session: weights from HuggingFace safetensors (with transposition)
    println!("loading native weights from HuggingFace...");
    let hf = SafeTensorsModel::download("dacorvo/mnist-mlp").expect("failed to download model");
    for (name, hf_name, transpose) in &native_weight_map {
        let data = if *transpose {
            hf.tensor_f32_transposed(hf_name)
        } else {
            hf.tensor_f32(hf_name)
        }
        .unwrap_or_else(|e| panic!("{}: {}", hf_name, e));
        native_session.set_parameter(name, &data);
    }

    // ─── Compare on test inputs ───
    println!("\n=== Comparison ===\n");

    // Test 1: zeros input
    let zeros = vec![0.0f32; 784];
    let (native_out, onnx_out) = run_both(&mut native_session, &mut onnx_session, &zeros, 10);
    print_comparison("zeros input", &native_out, &onnx_out);

    // Test 2: ones input
    let ones = vec![1.0f32; 784];
    let (native_out, onnx_out) = run_both(&mut native_session, &mut onnx_session, &ones, 10);
    print_comparison("ones input", &native_out, &onnx_out);

    // Test 3: random-ish input (deterministic)
    let rand_input: Vec<f32> = (0..784)
        .map(|i| ((i * 7 + 13) % 256) as f32 / 255.0)
        .collect();
    let (native_out, onnx_out) = run_both(&mut native_session, &mut onnx_session, &rand_input, 10);
    print_comparison("pseudo-random input", &native_out, &onnx_out);

    // Test 4: MNIST-normalized input (if data available)
    let data_dir = Path::new("data");
    let gz_images = data_dir.join("t10k-images-idx3-ubyte.gz");
    if gz_images.exists() {
        let gz_labels = data_dir.join("t10k-labels-idx1-ubyte.gz");
        let mnist =
            meganeura::MnistDataset::load_gz(&gz_images, &gz_labels).expect("failed to load MNIST");
        println!("\n--- MNIST accuracy comparison (first 1000 images) ---");

        let mut native_correct = 0;
        let mut onnx_correct = 0;
        let num_test = 1000.min(mnist.n);

        for i in 0..num_test {
            let raw = &mnist.images[i * 784..(i + 1) * 784];
            let image: Vec<f32> = raw.iter().map(|&v| (v - 0.1307) / 0.3081).collect();
            let label = &mnist.labels[i * 10..(i + 1) * 10];
            let true_label = label
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .unwrap()
                .0;

            let (native_out, onnx_out) =
                run_both(&mut native_session, &mut onnx_session, &image, 10);

            let native_pred = argmax(&native_out);
            let onnx_pred = argmax(&onnx_out);

            if native_pred == true_label {
                native_correct += 1;
            }
            if onnx_pred == true_label {
                onnx_correct += 1;
            }
        }

        println!(
            "  native accuracy: {}/{} ({:.2}%)",
            native_correct,
            num_test,
            native_correct as f64 / num_test as f64 * 100.0
        );
        println!(
            "  ONNX   accuracy: {}/{} ({:.2}%)",
            onnx_correct,
            num_test,
            onnx_correct as f64 / num_test as f64 * 100.0
        );
        if native_correct == onnx_correct {
            println!("  MATCH: both models agree on all predictions");
        } else {
            println!(
                "  DIFF: {} disagreements",
                (native_correct as i64 - onnx_correct as i64).unsigned_abs()
            );
        }
    } else {
        println!(
            "\n(MNIST data not found at {}, skipping accuracy test)",
            data_dir.display()
        );
    }
}

/// Build the native MNIST MLP graph (same as examples/huggingface.rs).
/// Returns (graph, weight_map) where weight_map is (param_name, hf_name, needs_transpose).
fn build_native_mnist_mlp() -> (Graph, Vec<(&'static str, &'static str, bool)>) {
    let mut g = Graph::new();
    let batch = 1;
    let input_dim = 784;
    let hidden = 256;
    let classes = 10;

    let x = g.input("x", &[batch, input_dim]);

    let w1 = g.parameter("input_layer.weight", &[input_dim, hidden]);
    let b1 = g.parameter("input_layer.bias", &[hidden]);
    let h1 = g.matmul(x, w1);
    let h1 = g.bias_add(h1, b1);
    let h1 = g.relu(h1);

    let w2 = g.parameter("mid_layer.weight", &[hidden, hidden]);
    let b2 = g.parameter("mid_layer.bias", &[hidden]);
    let h2 = g.matmul(h1, w2);
    let h2 = g.bias_add(h2, b2);
    let h2 = g.relu(h2);

    let w3 = g.parameter("output_layer.weight", &[hidden, classes]);
    let b3 = g.parameter("output_layer.bias", &[classes]);
    let logits = g.matmul(h2, w3);
    let logits = g.bias_add(logits, b3);

    // Output raw logits (not softmax) to match ONNX export
    g.set_outputs(vec![logits]);

    let weight_map = vec![
        ("input_layer.weight", "input_layer.weight", true),
        ("input_layer.bias", "input_layer.bias", false),
        ("mid_layer.weight", "mid_layer.weight", true),
        ("mid_layer.bias", "mid_layer.bias", false),
        ("output_layer.weight", "output_layer.weight", true),
        ("output_layer.bias", "output_layer.bias", false),
    ];

    (g, weight_map)
}

fn run_both(
    native: &mut meganeura::Session,
    onnx: &mut meganeura::Session,
    input: &[f32],
    output_size: usize,
) -> (Vec<f32>, Vec<f32>) {
    native.set_input("x", input);
    native.step();
    native.wait();
    let native_out = native.read_output(output_size);

    onnx.set_input("x", input);
    onnx.step();
    onnx.wait();
    let onnx_out = onnx.read_output(output_size);

    (native_out, onnx_out)
}

fn print_comparison(label: &str, native: &[f32], onnx: &[f32]) {
    println!("--- {} ---", label);
    println!("  native: {:>9.4?}", &native[..native.len().min(10)]);
    println!("  ONNX:   {:>9.4?}", &onnx[..onnx.len().min(10)]);

    let max_diff = native
        .iter()
        .zip(onnx.iter())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    let mean_diff: f32 = native
        .iter()
        .zip(onnx.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<f32>()
        / native.len() as f32;

    println!("  max_diff: {:.6e}, mean_diff: {:.6e}", max_diff, mean_diff);
    if max_diff < 1e-3 {
        println!("  PASS (max diff < 1e-3)");
    } else if max_diff < 1e-1 {
        println!("  WARN (max diff < 1e-1, possible precision issue)");
    } else {
        println!("  FAIL (max diff >= 1e-1)");
    }
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0
}
