use meganeura::Graph;
use meganeura::reference::{Feeds, gpu, gradients};

#[test]
fn smoke() {
    let mut g = Graph::new();
    let x = g.input("x", &[5, 7]);
    let w = g.parameter("w", &[7, 3]);
    let b = g.parameter("b", &[3]);
    let y = g.matmul(x, w);
    let y = g.bias_add(y, b);
    let s = g.softmax(y);
    g.set_outputs(vec![s, y]);
    let mut feeds = Feeds::new();
    feeds.fill_random(&g, 1, 1.0);
    let report = gpu::check_inference(&g, &feeds, &gpu::Options::default()).unwrap();
    println!("{report}");
    report.assert_passed("inference");

    let mut g = Graph::new();
    let x = g.input("x", &[5, 7]);
    let w = g.parameter("w", &[7, 3]);
    let b = g.parameter("b", &[3]);
    let labels = g.input("labels", &[5, 3]);
    let y = g.matmul(x, w);
    let y = g.bias_add(y, b);
    let y = g.tanh(y);
    let l = g.cross_entropy_loss(y, labels);
    let l = g.scale(l, 0.7);
    g.set_outputs(vec![l]);
    let mut feeds = Feeds::new();
    feeds.fill_random(&g, 2, 1.0);
    let report = gradients::check(&g, &feeds, &gradients::Options::default()).unwrap();
    println!("{report}");
    report.assert_passed("autodiff");
    let report = gpu::check_training(&g, &feeds, &gpu::Options::default()).unwrap();
    println!("{report}");
    report.assert_passed("training");
}
