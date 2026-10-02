use meganeura::{CoopPolicy, Graph, SessionConfig, SessionOptions};

/// A few ULP of relative slack; see [`close`].
const TOLERANCE: f32 = 1.0e-5;

/// Relative agreement, the same form `gguf_model` uses.
///
/// Bit identity is the wrong contract here. This test exists to prove that
/// allocation padding never reaches the arithmetic, and it does — the
/// poisoned tails are excluded by `i < s.len` in every optimizer, clip and
/// accumulation pass, and the gradients themselves are bit-identical across
/// the paddings. What the padding *can* legitimately perturb is the order in
/// which f32 values are summed, and that is enough to move a result by a
/// unit in the last place: the LaProp and adaptive-clip path reduces a
/// workgroup-sized tree whose lane occupancy depends on the tile layout, and
/// reassociating six squares in a different order is not exact.
///
/// Demanding `f32` equality therefore failed on rounding, not on a defect,
/// and the fix is to state the property the test actually means: the
/// padding-relative result must agree to within f32 noise. The tolerance is
/// loose enough to absorb a few ULP and far tighter than the failure it
/// guards against — a single leaked `1000.0` tail element inflates the
/// adaptive-clip norm by roughly 2400x, so noise and a real leak stay about
/// five orders of magnitude apart.
fn close(a: &[f32], b: &[f32], tolerance: f32) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(x, y)| (x - y).abs() <= tolerance * (1.0 + x.abs().max(y.abs())))
}

/// What one configuration of the padding test produced: the parameter, the
/// `(m, v)` moments per parameter, and the grouped gradient norm.
type Observed = (Vec<Vec<f32>>, Vec<(Vec<f32>, Vec<f32>)>, Vec<f32>);

fn close_observed(left: &Observed, right: &Observed) -> bool {
    close(&left.0[0], &right.0[0], TOLERANCE)
        && left.1.len() == right.1.len()
        && left
            .1
            .iter()
            .zip(&right.1)
            .all(|((lm, lv), (rm, rv))| close(lm, rm, TOLERANCE) && close(lv, rv, TOLERANCE))
        && close(&left.2, &right.2, TOLERANCE)
}

fn session(debug: bool) -> meganeura::Session {
    let mut graph = Graph::new();
    let a = graph.parameter("a", &[3]);
    let b = graph.parameter("b", &[5]);
    let c = graph.parameter("c", &[7]);
    let a = graph.mean_all(a);
    let b = graph.mean_all(b);
    let c = graph.mean_all(c);
    let sum = graph.add(a, b);
    let loss = graph.add(sum, c);
    graph.set_outputs(vec![loss]);
    meganeura::build(
        &graph,
        SessionConfig {
            runtime: SessionOptions {
                debug,
                coop: CoopPolicy::Disabled,
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .0
}

#[test]
fn moments_are_lazy_and_accumulators_are_counted_once() {
    for debug in [false, true] {
        let mut s = session(debug);
        let before = s.memory_summary();
        assert_eq!(before.adam_state_bytes, 0);
        assert_eq!(before.grad_accumulator_bytes, 0);
        // The clip total, a two-float partial slot per (single-workgroup)
        // parameter, and one adaptive-clip scale per parameter.
        let clip_scratch = 3 * 8 + 3 * 4;
        assert_eq!(before.optimizer_aux_bytes, 4 + clip_scratch);
        assert_eq!(
            s.read_adam_states(&["b"]),
            vec![(vec![0.0; 5], vec![0.0; 5])]
        );
        let mut m = [9.0; 3];
        s.read_adam_m("a", &mut m);
        assert_eq!(m, [0.0; 3]);
        s.read_adam_v("a", &mut m);
        assert_eq!(m, [0.0; 3]);
        s.set_learning_rate(0.01);
        s.step();
        s.wait();
        assert_eq!(s.memory_summary().adam_state_bytes, 0);

        s.set_grad_accumulate(3);
        let accumulated = s.memory_summary();
        // Optimizer state follows the parameter arena, whose slots start at
        // storage-binding offsets: one 256-byte slot per parameter here.
        let slots = 3 * 256;
        assert_eq!(accumulated.grad_accumulator_bytes, slots);
        assert_eq!(
            accumulated.total_allocated_bytes() - before.total_allocated_bytes(),
            slots
        );
        assert_eq!(
            accumulated.device_local_bytes - before.device_local_bytes,
            if debug { 0 } else { slots }
        );

        s.set_adam_grouped_grad_norm("b", 1);
        assert_eq!(s.memory_summary().optimizer_aux_bytes, 24 + clip_scratch);
        assert_eq!(s.memory_summary().adam_state_bytes, 0);
        s.set_adam(0.001, 0.9, 0.999, 1e-8);
        let allocated = s.memory_summary();
        assert_eq!(allocated.adam_state_bytes, slots * 2);
        assert_eq!(
            allocated.device_local_bytes - accumulated.device_local_bytes,
            if debug { 0 } else { slots * 2 }
        );
        s.step();
        s.wait();
        let states = s.read_adam_states(&["a", "b", "c"]);
        s.clear_optimizer();
        s.set_learning_rate(0.01);
        s.set_laprop(0.001, 0.9, 0.999, 1e-8);
        assert_eq!(s.read_adam_states(&["a", "b", "c"]), states);
        assert_eq!(
            s.memory_summary().total_allocated_bytes(),
            allocated.total_allocated_bytes()
        );
        println!(
            "debug={debug}: graph={} B, no-Adam={} B, Adam={} B, accumulators={} B",
            before.allocated_buffer_bytes,
            before.total_allocated_bytes(),
            allocated.adam_state_bytes,
            allocated.grad_accumulator_bytes
        );
    }
}

#[test]
fn explicit_moment_write_initializes_storage_without_configuring_updates() {
    let mut s = session(false);
    s.write_adam_m("a", &[1.0, 2.0, 3.0]);
    assert_eq!(
        s.read_adam_states(&["a"]),
        vec![(vec![1.0, 2.0, 3.0], vec![0.0; 3])]
    );
    assert_eq!(s.memory_summary().adam_state_bytes, 3 * 256 * 2);
    s.step();
    s.wait();
    assert_eq!(s.adam_step_count(), 0);
    assert_eq!(s.read_adam_states(&["a"])[0].0, vec![1.0, 2.0, 3.0]);
}

#[test]
fn a_million_f32_parameters_do_not_reserve_eight_mib_of_unused_moments() {
    let mut graph = Graph::new();
    let p = graph.parameter("p", &[1024, 1024]);
    let loss = graph.mean_all(p);
    graph.set_outputs(vec![loss]);
    let mut s = meganeura::build(&graph, crate::support::gpu::config()).0;
    let before = s.memory_summary();
    assert_eq!(before.adam_state_bytes, 0);
    s.set_adam(0.001, 0.9, 0.999, 1e-8);
    let after = s.memory_summary();
    assert_eq!(after.adam_state_bytes, 8 * 1024 * 1024);
    assert_eq!(
        after.total_allocated_bytes() - before.total_allocated_bytes(),
        8 * 1024 * 1024
    );
    println!(
        "1,048,576 F32 parameters: {} -> {} requested resident bytes; Adam adds {} bytes",
        before.total_allocated_bytes(),
        after.total_allocated_bytes(),
        after.adam_state_bytes
    );
}

#[test]
fn optimizer_clipping_and_diagnostics_ignore_poisoned_allocation_padding() {
    for debug in [false, true] {
        for mode in 0..6 {
            let run = |param_padding, grad_padding| -> Observed {
                let mut graph = Graph::new();
                let p = graph.parameter("p", &[2, 3]);
                let loss = graph.mean_all(p);
                graph.set_outputs(vec![loss]);
                let mut plan =
                    meganeura::compile::compile(&meganeura::autodiff::differentiate(&graph));
                let (parameter, gradient) = plan.param_grad_pairs[0];
                plan.buffers[parameter.0 as usize] += param_padding;
                plan.buffers[gradient.0 as usize] += grad_padding;
                // Seed the whole gradient slot; backward overwrites only its
                // logical prefix, leaving finite poison in the allocation tail.
                let poison = vec![1000.0f32; (24 + grad_padding) / 4];
                plan.constant_buffers
                    .push((gradient, bytemuck::cast_slice(&poison).to_vec()));
                let mut s = meganeura::Session::with_context_opts(
                    plan,
                    crate::support::gpu::gpu(),
                    SessionOptions {
                        debug,
                        coop: CoopPolicy::Disabled,
                        ..Default::default()
                    },
                );
                let mut weights = vec![1000.0f32; (24 + param_padding) / 4];
                weights[..6].copy_from_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
                s.set_parameter("p", &weights);
                if mode < 3 {
                    s.set_grad_accumulate(2);
                    s.step();
                    match mode {
                        0 => s.set_learning_rate(0.1),
                        1 => s.set_adam(0.01, 0.9, 0.999, 1e-8),
                        _ => s.set_laprop(0.01, 0.9, 0.999, 1e-8),
                    }
                    if mode == 2 {
                        s.set_adaptive_grad_clip(0.01, 0.001);
                    } else {
                        s.set_grad_clip_norm(0.1);
                    }
                    if mode != 0 {
                        s.set_adam_grouped_grad_norm("p", 2);
                    }
                    s.step();
                } else {
                    s.step();
                    let (norm, clipped) = s.clip_grad_norm_cpu(0.1);
                    assert!((norm - (1.0f32 / 6.0).sqrt()).abs() < 1e-6);
                    assert!(clipped);
                    match mode {
                        3 => s.sgd_step(0.1),
                        4 => s.sgd_step_cpu(0.1),
                        _ => s.adam_step(0.01, 0.9, 0.999, 1e-8),
                    }
                }
                s.wait();
                s.read_buffer(parameter, &mut weights);
                assert!(weights[6..].iter().all(|v| *v == 1000.0));
                let mut gradient_slot = vec![0.0; (24 + grad_padding) / 4];
                s.read_buffer(gradient, &mut gradient_slot);
                assert!(gradient_slot[6..].iter().all(|v| *v == 1000.0));
                let diagnostic = if mode == 1 || mode == 2 {
                    s.read_adam_grouped_grad_norm("p")
                } else {
                    Vec::new()
                };
                (
                    s.read_params(&["p"]),
                    s.read_adam_states(&["p"]),
                    diagnostic,
                )
            };
            let expected = run(0, 0);
            for (param_padding, grad_padding) in [(64, 128), (64, 0), (0, 128)] {
                let got = run(param_padding, grad_padding);
                assert!(
                    close_observed(&got, &expected),
                    "mode={mode}, debug={debug}, padding=({param_padding},{grad_padding})\n\
                     got:      {got:?}\n\
                     expected: {expected:?}"
                );
            }
        }
    }
}

#[test]
fn parameter_reads_reject_wrong_storage_and_oversized_views() {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    let s = session(false);
    let buffer = s.param_buffer("a").unwrap();
    assert!(catch_unwind(AssertUnwindSafe(|| s.read_buffer(buffer, &mut [0.0; 4]))).is_err());
    let mut graph = Graph::new();
    let p = graph.parameter_f16("p", &[3]);
    graph.set_outputs(vec![p]);
    let s = meganeura::Session::new(meganeura::compile::compile(&graph));
    assert_eq!(s.param_size("p"), Some(3));
    assert!(catch_unwind(AssertUnwindSafe(|| s.read_param("p", &mut [0.0; 3]))).is_err());
    assert!(catch_unwind(AssertUnwindSafe(|| s.read_params(&["p"]))).is_err());
    assert!(catch_unwind(AssertUnwindSafe(|| s.read_all_param_norms())).is_err());
}

/// Every optimizer feature gives bit-identical results whether the
/// parameters share one arena chunk (one dispatch per pass) or each has
/// its own: the per-element arithmetic and the partial-norm order agree.
#[test]
fn arena_chunking_does_not_change_updates() {
    let run = |chunk: Option<usize>| {
        let mut graph = Graph::new();
        let mut terms = Vec::new();
        for (name, len) in [("a", 3), ("b", 300), ("c", 1500)] {
            let p = graph.parameter(name, &[len]);
            let squared = graph.mul(p, p);
            terms.push(graph.sum_all(squared));
        }
        let sum = graph.add(terms[0], terms[1]);
        let loss = graph.add(sum, terms[2]);
        graph.set_outputs(vec![loss]);
        let mut s = meganeura::build(
            &graph,
            SessionConfig {
                runtime: SessionOptions {
                    coop: CoopPolicy::Disabled,
                    arena_chunk_bytes: chunk,
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .0;
        for (name, len) in [("a", 3), ("b", 300), ("c", 1500)] {
            let values: Vec<f32> = (0..len)
                .map(|i| ((i * 37 % 101) as f32 - 50.0) / 25.0)
                .collect();
            s.set_parameter(name, &values);
        }
        s.set_lr_multiplier("b", 2.0);
        s.set_grad_clip_norm(5.0);
        s.set_adam(0.01, 0.9, 0.999, 1e-8);
        s.set_adam_grouped_grad_norm("c", 3);
        for _ in 0..2 {
            s.step();
        }
        s.set_adaptive_grad_clip(0.05, 1e-3);
        s.set_grad_accumulate(2);
        s.zero_grad();
        for _ in 0..2 {
            s.step();
        }
        s.set_learning_rate(0.02);
        s.step();
        s.wait();
        let names = ["a", "b", "c"];
        (
            s.read_params(&names),
            s.read_adam_states(&names),
            s.read_adam_grouped_grad_norm("c"),
        )
    };
    let packed = run(None);
    let separate = run(Some(256));
    assert_eq!(packed, separate);
    // The updates did happen.
    assert!(packed.2.iter().all(|&norm| norm > 0.0));
}
