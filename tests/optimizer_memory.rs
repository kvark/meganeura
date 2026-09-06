use meganeura::{CoopPolicy, Graph, SessionConfig, SessionOptions};

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
        assert_eq!(before.optimizer_aux_bytes, 4);
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
        assert_eq!(accumulated.grad_accumulator_bytes, (3 + 5 + 7) * 4);
        assert_eq!(
            accumulated.total_allocated_bytes() - before.total_allocated_bytes(),
            60
        );
        assert_eq!(
            accumulated.device_local_bytes - before.device_local_bytes,
            if debug { 0 } else { 60 }
        );

        s.set_adam_grouped_grad_norm("b", 1);
        assert_eq!(s.memory_summary().optimizer_aux_bytes, 24);
        assert_eq!(s.memory_summary().adam_state_bytes, 0);
        s.set_adam(0.001, 0.9, 0.999, 1e-8);
        let allocated = s.memory_summary();
        assert_eq!(allocated.adam_state_bytes, (3 + 5 + 7) * 4 * 2);
        assert_eq!(
            allocated.device_local_bytes - accumulated.device_local_bytes,
            if debug { 0 } else { 120 }
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
    assert_eq!(s.memory_summary().adam_state_bytes, 120);
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
    let mut s = meganeura::build(&graph, SessionConfig::default()).0;
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
            let run = |param_padding, grad_padding| {
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
                    std::sync::Arc::new(meganeura::init_gpu_context().unwrap()),
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
                assert_eq!(
                    run(param_padding, grad_padding),
                    expected,
                    "mode={mode}, debug={debug}, padding=({param_padding},{grad_padding})"
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
