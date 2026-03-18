/// MNIST training example for Meganeura.
///
/// Demonstrates the full pipeline:
///   Graph definition → autodiff → egglog optimization → compile → GPU execution
///
/// If MNIST data files are present in `data/`, loads them automatically.
/// Otherwise falls back to synthetic data. Download MNIST from:
///   <https://yann.lecun.com/exdb/mnist/>
///
/// Expected files (gzipped or raw):
///   data/train-images-idx3-ubyte.gz  (or without .gz)
///   data/train-labels-idx1-ubyte.gz  (or without .gz)
use meganeura::{DataLoader, Graph, MnistDataset, TrainConfig, Trainer, build_session};
use std::path::Path;

fn main() {
    env_logger::init();

    // Set up Perfetto profiling when MEGANEURA_PROFILE is set.
    let profiling = std::env::var("MEGANEURA_PROFILE").is_ok();
    if profiling {
        meganeura::profiler::init();
    }

    let batch = 32;
    let input_dim = 784; // 28x28
    let hidden = 128;
    let classes = 10;
    let epochs = 10;
    let lr = 0.01_f32;

    // --- Load data ---
    let mut loader = load_mnist_or_synthetic(batch, input_dim, classes);
    let steps_per_epoch = loader.num_batches();
    println!("{} samples, {} batches/epoch", loader.len(), steps_per_epoch);

    // --- Build the model graph ---
    let mut g = Graph::new();

    // Inputs
    let x = g.input("x", &[batch, input_dim]);
    let labels = g.input("labels", &[batch, classes]);

    // Layer 1: linear + relu
    let w1 = g.parameter("w1", &[input_dim, hidden]);
    let b1 = g.parameter("b1", &[hidden]);
    let mm1 = g.matmul(x, w1);
    let h1 = g.bias_add(mm1, b1);
    let a1 = g.relu(h1);

    // Layer 2: linear → logits
    let w2 = g.parameter("w2", &[hidden, classes]);
    let b2 = g.parameter("b2", &[classes]);
    let mm2 = g.matmul(a1, w2);
    let logits = g.bias_add(mm2, b2);

    // Loss
    let loss = g.cross_entropy_loss(logits, labels);
    g.set_outputs(vec![loss]);

    println!("forward graph:\n{}", g);

    // --- Build training session ---
    // This runs: autodiff → egglog optimize → compile → GPU init
    println!("building session (autodiff + egglog + compile)...");
    let mut session = build_session(&g);
    println!(
        "session ready: {} buffers, {} dispatches",
        session.plan().buffers.len(),
        session.plan().dispatches.len()
    );

    // --- Initialize parameters ---
    // Xavier initialization
    let w1_data = xavier_init(input_dim, hidden);
    let b1_data = vec![0.0_f32; hidden];
    let w2_data = xavier_init(hidden, classes);
    let b2_data = vec![0.0_f32; classes];

    session.set_parameter("w1", &w1_data);
    session.set_parameter("b1", &b1_data);
    session.set_parameter("w2", &w2_data);
    session.set_parameter("b2", &b2_data);

    // --- Training loop ---
    println!("training...");
    let config = TrainConfig {
        learning_rate: lr,
        log_interval: 50,
        ..TrainConfig::default()
    };
    let mut trainer = Trainer::new(session, config);
    let history = trainer.train(&mut loader, epochs);

    if let Some(final_loss) = history.final_loss() {
        println!("done! final avg_loss = {:.4}", final_loss);
    } else {
        println!("done! (no epochs ran)");
    }

    // Save Perfetto trace when profiling.
    if profiling {
        let path = Path::new("trace.pftrace");
        meganeura::profiler::save(path).expect("failed to save profile");
        println!("profile saved to {}", path.display());
    }
}

fn load_mnist_or_synthetic(batch: usize, input_dim: usize, classes: usize) -> DataLoader {
    let data_dir = Path::new("data");

    // Try gzipped files first, then raw
    let gz_images = data_dir.join("train-images-idx3-ubyte.gz");
    let gz_labels = data_dir.join("train-labels-idx1-ubyte.gz");
    let raw_images = data_dir.join("train-images-idx3-ubyte");
    let raw_labels = data_dir.join("train-labels-idx1-ubyte");

    if gz_images.exists() && gz_labels.exists() {
        println!("loading MNIST from {} (gzipped)...", data_dir.display());
        let mnist = MnistDataset::load_gz(&gz_images, &gz_labels)
            .expect("failed to parse MNIST gz files");
        println!("loaded {} MNIST images", mnist.n);
        return mnist.loader(batch);
    }

    if raw_images.exists() && raw_labels.exists() {
        println!("loading MNIST from {} (raw)...", data_dir.display());
        let mnist = MnistDataset::load(&raw_images, &raw_labels)
            .expect("failed to parse MNIST files");
        println!("loaded {} MNIST images", mnist.n);
        return mnist.loader(batch);
    }

    println!("MNIST not found in data/, using synthetic data");
    let n = 3200; // enough for 100 batches of 32
    synthetic_loader(n, input_dim, classes, batch)
}

fn synthetic_loader(n: usize, input_dim: usize, classes: usize, batch: usize) -> DataLoader {
    let images: Vec<f32> = (0..n * input_dim)
        .map(|i| ((i % 256) as f32) / 255.0)
        .collect();
    let mut labels = vec![0.0_f32; n * classes];
    for b in 0..n {
        labels[b * classes + (b % classes)] = 1.0;
    }
    DataLoader::new(images, labels, input_dim, classes, batch)
}

fn xavier_init(fan_in: usize, fan_out: usize) -> Vec<f32> {
    use std::f32::consts::PI;
    let scale = (2.0 / (fan_in + fan_out) as f32).sqrt();
    let n = fan_in * fan_out;
    // Simple pseudo-random using sine (deterministic, no external deps)
    (0..n)
        .map(|i| {
            let x = (i as f32 + 1.0) * 0.618_034; // golden ratio
            (x * PI * 2.0).sin() * scale
        })
        .collect()
}
