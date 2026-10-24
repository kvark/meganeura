use meganeura::{
    Graph,
    reference::{Feeds, Tolerance, check, error_scales, evaluate, gpu, gradients},
};

#[test]
fn comparison_checks_every_element_with_a_finite_bound() {
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
fn rotation_preserves_upstream_error_scales() {
    for offset in [0, 1, 3] {
        let mut g = Graph::new();
        let a = g.constant(vec![1.0, -1.0], &[1, 2]);
        let b = g.constant(vec![1.0, 2.0, 1.0, 2.0], &[2, 2]);
        let x = g.matmul(a, b);
        let forward = g.rope_with_offset(x, 10_000.0, offset, 2);
        let backward = g.rope_grad(x, 10_000.0, offset, 2);
        let positions = g.input_u32("positions", &[1]);
        let indexed = g.rope_with_positions(x, 10_000.0, positions, 2);
        let mut feeds = Feeds::new();
        feeds.set_u32("positions", &[offset]);
        let values = evaluate(&g, &feeds).unwrap();
        let scales = error_scales(&g, &values).unwrap();
        assert_eq!(values[x as usize].data, [0.0, 0.0]);
        assert_eq!(scales[x as usize], [2.0, 4.0]);
        let (sin, cos) = f64::from(offset).sin_cos();
        let expected = [
            2.0 * cos.abs() + 4.0 * sin.abs(),
            2.0 * sin.abs() + 4.0 * cos.abs(),
        ];
        for node in [forward, backward, indexed] {
            assert_eq!(scales[node as usize], expected);
            assert!(
                check(
                    &[1.0, 0.0],
                    &[0.0, 0.0],
                    &scales[node as usize],
                    Tolerance::default()
                )
                .is_err()
            );
        }
    }
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

#[test]
fn persistent_execution_reports_actual_coverage() {
    let mut graph = Graph::new();
    let input = graph.input("x", &[3, 259]);
    let mapped = graph.silu(input);
    let reduced = graph.sum_inner(mapped);
    graph.set_outputs(vec![mapped, reduced]);
    let mut feeds = Feeds::new();
    feeds.fill_random(&graph, 421, 1.0);
    for persistent in [
        None,
        Some(meganeura::PersistentOptions {
            workers: 3,
            allow_fallback: false,
            ..Default::default()
        }),
        Some(meganeura::PersistentOptions {
            workers: 3,
            allow_fallback: false,
            device_scheduling: true,
            ..Default::default()
        }),
    ] {
        if persistent.is_some_and(|options| options.device_scheduling)
            && !gpu::shared_context().capabilities().device_memory_fence
        {
            continue;
        }
        let options = gpu::Options {
            persistent,
            ..Default::default()
        };
        let report = gpu::check_inference(&graph, &feeds, &options).unwrap();
        report.assert_passed("persistent coverage");
        let execution = report.execution.unwrap();
        assert_eq!(execution.persistent, persistent);
        assert_eq!(
            execution
                .regions
                .iter()
                .all(|region| region.device_scheduled),
            persistent.is_some_and(|options| options.device_scheduling)
        );
        assert!(!execution.regions.is_empty());
        assert_eq!(execution.persistent_tasks > 0, persistent.is_some());
        assert!(
            execution
                .regions
                .iter()
                .all(|region| region.fallback_reason.is_none())
        );
        assert_eq!(
            execution.regions.iter().all(|region| region.workers > 0),
            persistent.is_some()
        );
    }
}
