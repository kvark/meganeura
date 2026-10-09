//! Loss ops write one partial per row or workgroup. Their value is the sum,
//! whether read by `read_loss` or consumed by another node.

use meganeura::Graph;
use meganeura::reference::{Feeds, evaluate_outputs, gpu, gradients};

fn run(consume: bool, bce: bool) {
    let mut g = Graph::new();
    let x = g.input("x", &[5, 300]);
    let labels = g.input("labels", &[5, 300]);
    let l = if bce {
        let p = g.sigmoid(x);
        g.bce_loss(p, labels)
    } else {
        g.cross_entropy_loss(x, labels)
    };
    let out = if consume { g.scale(l, 0.5) } else { l };
    g.set_outputs(vec![out]);
    let mut feeds = Feeds::new();
    feeds.fill_random(&g, 3, 1.0);
    gpu::check_inference(&g, &feeds, &gpu::Options::default())
        .unwrap()
        .assert_passed(&format!("consume={consume} bce={bce}"));
}

#[test]
fn loss_value_is_the_sum_of_partials_for_every_consumer() {
    for bce in [false, true] {
        for consume in [false, true] {
            run(consume, bce);
        }
    }
}

#[test]
fn class_indices_match_dense_targets_and_scaled_gradients() {
    for classes in [3usize, 300] {
        let indices = [0, classes as u32 - 1, 1, 0, u32::MAX];
        let mut dense_targets = vec![0.0; indices.len() * classes];
        for (row, &index) in indices.iter().enumerate() {
            if index < classes as u32 {
                dense_targets[row * classes + index as usize] = 1.0;
            }
        }
        let graph = |indexed| {
            let mut g = Graph::new();
            let x = g.parameter("x", &[indices.len(), classes]);
            let loss = if indexed {
                let labels = g.input_u32("indices", &[indices.len()]);
                g.cross_entropy_loss_indices(x, labels)
            } else {
                let labels = g.input("dense", &[indices.len(), classes]);
                g.cross_entropy_loss(x, labels)
            };
            let loss = g.scale(loss, 0.375);
            g.set_outputs(vec![loss]);
            g
        };
        let (indexed, dense) = (graph(true), graph(false));
        let mut feeds = Feeds::new();
        feeds.set_u32("indices", &indices);
        feeds.set("dense", &dense_targets);
        feeds.fill_random(&indexed, 37, 4.0);
        let expected =
            evaluate_outputs(&meganeura::autodiff::differentiate(&dense), &feeds).unwrap();
        let actual =
            evaluate_outputs(&meganeura::autodiff::differentiate(&indexed), &feeds).unwrap();
        assert_eq!(actual, expected);
        gradients::check(&indexed, &feeds, &gradients::Options::default())
            .unwrap()
            .assert_passed("indexed targets: finite differences");
        for (name, options) in gpu::Options::lowerings() {
            gpu::check_inference(&indexed, &feeds, &options)
                .unwrap()
                .assert_passed(&format!("indexed loss: classes={classes}, {name}"));
            gpu::check_training(&indexed, &feeds, &options)
                .unwrap()
                .assert_passed(&format!("indexed targets: classes={classes}, {name}"));
        }
    }
}
