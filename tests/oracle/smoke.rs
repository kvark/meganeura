use meganeura::Graph;
use meganeura::reference::{Feeds, gpu, gradients};

#[test]
fn comparison_checks_every_element_with_a_finite_bound() {
    use meganeura::reference::{Tolerance, check};
    let tolerance = Tolerance::default();
    assert!(std::panic::catch_unwind(|| check(&[100.0], &[1.0], &[], tolerance)).is_err());
    for magnitude in [f64::NAN, f64::INFINITY, -1.0] {
        assert!(check(&[100.0], &[1.0], &[magnitude], tolerance).is_err());
    }
    assert!(check(&[1.0], &[1.0], &[1.0], tolerance).is_ok());
    assert!(check(&[f32::NAN], &[1.0], &[1.0], tolerance).is_err());
    // Some ops intentionally return nonfinite values, such as log(0).
    assert!(
        check(
            &[f32::NEG_INFINITY],
            &[f64::NEG_INFINITY],
            &[f64::INFINITY],
            tolerance
        )
        .is_ok()
    );
}

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

/// With poisoning, memory nothing wrote reads as NaN: an input the caller
/// never set produces NaN rather than a plausible zero.
#[test]
fn poison_reaches_unwritten_memory() {
    let mut g = Graph::new();
    let x = g.input("x", &[300]);
    let y = g.neg(x);
    g.set_outputs(vec![y]);
    let mut config = meganeura::SessionConfig::from_env();
    config.mode = meganeura::Mode::Inference;
    config.runtime.poison = true;
    let (mut session, _) = meganeura::build(&g, config);
    session.step();
    session.wait();
    let mut out = vec![0.0f32; 300];
    session.read_output_by_index(0, &mut out);
    assert!(out.iter().all(|v| v.is_nan()), "{:?}", &out[..4]);
}
