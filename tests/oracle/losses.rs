//! Loss ops write one partial per row or workgroup. Their value is the sum,
//! whether read by `read_loss` or consumed by another node.

use meganeura::Graph;
use meganeura::reference::{Feeds, gpu};

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
