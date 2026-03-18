/// Load a pre-trained MNIST MLP from HuggingFace and run inference.
///
/// Downloads `dacorvo/mnist-mlp` — a 3-layer MLP (784→256→256→10)
/// trained on MNIST — from the HuggingFace Hub, loads the safetensors
/// weights into a meganeura graph, and classifies real MNIST test images.
///
/// The model expects flattened 28×28 images normalized with
/// mean=0.1307, std=0.3081.
///
/// Usage:
///   cargo run --example huggingface [model.safetensors]
///
/// MNIST test data is expected in `data/` (gzipped or raw):
///   data/t10k-images-idx3-ubyte.gz
///   data/t10k-labels-idx1-ubyte.gz
use meganeura::{
    Graph, MnistDataset, build_inference_session,
    data::safetensors::SafeTensorsModel,
};
use std::path::{Path, PathBuf};

fn main() {
    env_logger::init();

    let batch = 1;
    let input_dim = 784;
    let hidden = 256;
    let classes = 10;

    // --- Load model: CLI path or download from HuggingFace ---
    let hf = if let Some(path) = std::env::args().nth(1) {
        println!("loading model from {}...", path);
        SafeTensorsModel::load(PathBuf::from(path)).expect("failed to load model")
    } else {
        println!("downloading dacorvo/mnist-mlp from HuggingFace Hub...");
        SafeTensorsModel::download("dacorvo/mnist-mlp").expect("failed to download model")
    };

    // Print tensor info
    println!("model tensors:");
    let mut names: Vec<_> = hf.tensor_info().keys().collect();
    names.sort();
    for name in &names {
        let info = &hf.tensor_info()[*name];
        println!("  {}: shape={:?} dtype={:?}", name, info.shape, info.dtype);
    }

    // --- Build the inference graph ---
    // Architecture: Linear(784,256)+ReLU → Linear(256,256)+ReLU → Linear(256,10)+Softmax
    let mut g = Graph::new();

    let x = g.input("x", &[batch, input_dim]);

    // Layer 1: input_layer
    let w1 = g.parameter("input_layer.weight", &[input_dim, hidden]);
    let b1 = g.parameter("input_layer.bias", &[hidden]);
    let h1 = g.matmul(x, w1);
    let h1 = g.bias_add(h1, b1);
    let h1 = g.relu(h1);

    // Layer 2: mid_layer
    let w2 = g.parameter("mid_layer.weight", &[hidden, hidden]);
    let b2 = g.parameter("mid_layer.bias", &[hidden]);
    let h2 = g.matmul(h1, w2);
    let h2 = g.bias_add(h2, b2);
    let h2 = g.relu(h2);

    // Layer 3: output_layer + softmax
    let w3 = g.parameter("output_layer.weight", &[hidden, classes]);
    let b3 = g.parameter("output_layer.bias", &[classes]);
    let logits = g.matmul(h2, w3);
    let logits = g.bias_add(logits, b3);
    let probs = g.softmax(logits);

    g.set_outputs(vec![probs]);

    // --- Build inference session ---
    println!("compiling inference session...");
    let mut session = build_inference_session(&g);
    println!(
        "session ready: {} buffers, {} dispatches",
        session.plan().buffers.len(),
        session.plan().dispatches.len()
    );

    // --- Load weights from safetensors ---
    // PyTorch Linear stores weights as (out_features, in_features),
    // meganeura matmul expects (in_features, out_features) → transpose.
    println!("loading weights...");
    for name in ["input_layer.weight", "mid_layer.weight", "output_layer.weight"] {
        let data = hf
            .tensor_f32_transposed(name)
            .unwrap_or_else(|e| panic!("failed to load {}: {}", name, e));
        session.set_parameter(name, &data);
    }
    for name in ["input_layer.bias", "mid_layer.bias", "output_layer.bias"] {
        let data = hf
            .tensor_f32(name)
            .unwrap_or_else(|e| panic!("failed to load {}: {}", name, e));
        session.set_parameter(name, &data);
    }
    println!("weights loaded.");

    // --- Load MNIST test data ---
    let data_dir = Path::new("data");
    let mnist = load_mnist_test(data_dir);
    println!("loaded {} MNIST test images", mnist.n);

    // --- Run inference on all test images ---
    let mut correct = 0usize;
    let mut total = 0usize;

    for i in 0..mnist.n {
        // Extract and normalize one image: (pixel - 0.1307) / 0.3081
        let raw = &mnist.images[i * 784..(i + 1) * 784];
        let image: Vec<f32> = raw.iter().map(|&v| (v - 0.1307) / 0.3081).collect();

        // True label (argmax of one-hot)
        let label_slice = &mnist.labels[i * 10..(i + 1) * 10];
        let true_label = label_slice
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;

        session.set_input("x", &image);
        session.step();
        session.wait();

        let probs = session.read_output(classes);
        let predicted = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;

        if predicted == true_label {
            correct += 1;
        }
        total += 1;

        // Show first 10 predictions in detail
        if i < 10 {
            println!(
                "  sample {:>5}: true={}, predicted={}, confidence={:.1}% {}",
                i,
                true_label,
                predicted,
                probs[predicted] * 100.0,
                if predicted == true_label { "OK" } else { "WRONG" }
            );
        }
    }

    let accuracy = correct as f64 / total as f64 * 100.0;
    println!(
        "\naccuracy: {}/{} ({:.2}%)",
        correct, total, accuracy
    );
}

fn load_mnist_test(data_dir: &Path) -> MnistDataset {
    let gz_images = data_dir.join("t10k-images-idx3-ubyte.gz");
    let gz_labels = data_dir.join("t10k-labels-idx1-ubyte.gz");
    let raw_images = data_dir.join("t10k-images-idx3-ubyte");
    let raw_labels = data_dir.join("t10k-labels-idx1-ubyte");

    if gz_images.exists() && gz_labels.exists() {
        return MnistDataset::load_gz(&gz_images, &gz_labels)
            .expect("failed to parse MNIST gz files");
    }
    if raw_images.exists() && raw_labels.exists() {
        return MnistDataset::load(&raw_images, &raw_labels)
            .expect("failed to parse MNIST files");
    }
    panic!(
        "MNIST test data not found in {}.\n\
         Download from https://yann.lecun.com/exdb/mnist/ :\n  \
         t10k-images-idx3-ubyte.gz\n  \
         t10k-labels-idx1-ubyte.gz",
        data_dir.display()
    );
}
