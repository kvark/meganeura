//! Minimal train -> checkpoint -> inference demonstration.
//!
//! The example intentionally uses a tiny deterministic classifier. Its purpose
//! is to exercise the complete deployment path, not to benchmark model quality:
//! optimizer-backed GPU training, a safetensors checkpoint, and a fresh
//! inference-only session all use the same graph/runtime API.

use meganeura::{
    DataLoader, Graph, Optimizer, TrainConfig, Trainer, build_inference_session, build_session,
};

const BATCH: usize = 8;
const FEATURES: usize = 2;
const CLASSES: usize = 2;
const EPOCHS: usize = 40;

fn main() {
    env_logger::init();

    let features = vec![
        -2.0, -1.0, -1.0, -2.0, -2.0, 1.0, -1.0, 2.0, 1.0, -2.0, 2.0, -1.0, 1.0, 2.0, 2.0, 1.0,
    ];
    let labels = vec![
        1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0,
    ];

    let mut training = build_session(&classifier_graph(true));
    training.set_parameter("weight", &[0.0; FEATURES * CLASSES]);
    training.set_parameter("bias", &[0.0; CLASSES]);

    let mut loader = DataLoader::new(features.clone(), labels, FEATURES, CLASSES, BATCH);
    let mut trainer = Trainer::new(
        training,
        TrainConfig {
            optimizer: Optimizer::sgd(0.4),
            learning_rate: 0.4,
            log_interval: 0,
        },
    );
    let history = trainer.train(&mut loader, EPOCHS);
    let first_loss = history.epochs.first().expect("training ran").avg_loss;
    let final_loss = history.final_loss().expect("training ran");
    assert!(
        final_loss < first_loss * 0.1,
        "loss did not converge: {first_loss:.6} -> {final_loss:.6}",
    );

    let checkpoint = std::env::temp_dir().join(format!(
        "meganeura-train-deploy-{}.safetensors",
        std::process::id()
    ));
    assert!(
        !checkpoint.exists(),
        "temporary checkpoint already exists: {}",
        checkpoint.display()
    );
    trainer
        .session_mut()
        .save_checkpoint(&checkpoint)
        .expect("save checkpoint");

    let mut inference = build_inference_session(&classifier_graph(false));
    inference
        .load_checkpoint(&checkpoint)
        .expect("load checkpoint into inference session");
    inference.set_input("x", &features);
    inference.step();
    inference.wait();

    let logits = inference.read_output(BATCH * CLASSES);
    let correct = logits
        .as_chunks::<CLASSES>()
        .0
        .iter()
        .enumerate()
        .filter(|&(sample, scores)| {
            let predicted = usize::from(scores[1] > scores[0]);
            let expected = usize::from(sample >= BATCH / 2);
            predicted == expected
        })
        .count();

    assert_eq!(
        correct, BATCH,
        "reloaded model classified {correct}/{BATCH}"
    );
    println!(
        "loss {first_loss:.6} -> {final_loss:.6}; reloaded inference accuracy {correct}/{BATCH}"
    );
    println!("checkpoint: {}", checkpoint.display());

    std::fs::remove_file(&checkpoint).expect("remove temporary checkpoint");
}

fn classifier_graph(training: bool) -> Graph {
    let mut graph = Graph::new();
    let x = graph.input("x", &[BATCH, FEATURES]);
    let weight = graph.parameter("weight", &[FEATURES, CLASSES]);
    let bias = graph.parameter("bias", &[CLASSES]);
    let projected = graph.matmul(x, weight);
    let logits = graph.bias_add(projected, bias);

    if training {
        let labels = graph.input("labels", &[BATCH, CLASSES]);
        let loss = graph.cross_entropy_loss(logits, labels);
        graph.set_outputs(vec![loss]);
    } else {
        graph.set_outputs(vec![logits]);
    }
    graph
}
