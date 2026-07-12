//! `Trainer` end-to-end with a graph that takes more than two inputs.
//!
//! Three input streams (`x`, `bias`, `labels`) — `x + bias` is regressed
//! against `labels` with MSE. The point is that `Trainer`/`DataLoader`
//! used to hardcode `data_input` / `label_input` (single-data, single-
//! label) and so couldn't drive graphs like blade-volume-train's
//! volumetric forward that has 3+ inputs per step. After the multi-
//! input rework, the loader declares N named streams and the trainer
//! binds them by name.

use meganeura::{DataLoader, Graph, TrainConfig, Trainer, build_session, data::InputStream, nn};

fn build_xy_plus_bias_regressor(batch: usize, in_d: usize, out_d: usize) -> Graph {
    let mut g = Graph::new();
    let x = g.input("x", &[batch, in_d]);
    // Extra per-sample input that the canonical (data, labels) Trainer
    // can't deliver. Bias is added before the projection.
    let bias = g.input("bias", &[batch, in_d]);
    let target = g.input("labels", &[batch, out_d]);
    let xb = g.add(x, bias);
    let w = nn::Linear::new(&mut g, "w", in_d, out_d);
    let y = w.forward(&mut g, xb);
    let loss = g.mse_loss(y, target);
    g.set_outputs(vec![loss, y]);
    g
}

#[test]
fn trainer_drives_three_named_streams() {
    let batch = 4;
    let in_d = 3;
    let out_d = 2;
    let n = 32;

    let g = build_xy_plus_bias_regressor(batch, in_d, out_d);
    let session = build_session(&g);

    // Deterministic synthetic data.
    let x: Vec<f32> = (0..n * in_d).map(|i| (i as f32 * 0.1).sin()).collect();
    let bias: Vec<f32> = (0..n * in_d).map(|i| (i as f32 * 0.05).cos()).collect();
    let labels: Vec<f32> = (0..n * out_d)
        .map(|i| 0.5 + (i as f32 * 0.07).sin())
        .collect();

    let mut loader = DataLoader::with_streams(
        vec![
            InputStream::new("x", x, in_d),
            InputStream::new("bias", bias, in_d),
            InputStream::new("labels", labels, out_d),
        ],
        batch,
    );

    let mut trainer = Trainer::new(
        session,
        TrainConfig {
            optimizer: meganeura::Optimizer::adam(0.05),
            log_interval: 0,
            ..TrainConfig::default()
        },
    );

    let history = trainer.train(&mut loader, 4);
    let first = history.epochs.first().unwrap().avg_loss;
    let last = history.epochs.last().unwrap().avg_loss;
    assert!(
        last < first,
        "loss should decrease across epochs; first={first} last={last}",
    );
    assert!(last.is_finite(), "loss must stay finite; got {last}");
}
