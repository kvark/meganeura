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
    let report = gpu::check_inference(&g, &feeds, &gpu::Options::default()).unwrap();
    println!("consume={consume} bce={bce}\n{report}");
}

#[test]
fn loss_consumers() {
    for bce in [false, true] {
        for consume in [false, true] {
            run(consume, bce);
        }
    }
}
