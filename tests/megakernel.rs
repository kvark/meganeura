use meganeura::megakernel::{Execution, Executor, Instruction, Job, Options};

#[test]
fn session_whole_graph_narrow_scalar_tiles() {
    check_scalar_tile_edges(false);
    check_scalar_tile_edges(true);
}

fn check_scalar_tile_edges(transposed_b: bool) {
    let gpu = crate::support::gpu::gpu();
    let supports_graph = gpu.capabilities().device_memory_fence && gpu.capabilities().binding_array;
    let (m, n, k) = (35, 65, 1031);
    let mut graph = meganeura::Graph::new();
    let a = graph.input("a", &[m, k]);
    let b = graph.input("b", &if transposed_b { [n, k] } else { [k, n] });
    let d = graph.input("d", &[m, n]);
    let y = if transposed_b {
        graph.matmul_bt(a, b)
    } else {
        graph.matmul(a, b)
    };
    graph.nodes_mut()[y as usize].op = if transposed_b {
        meganeura::graph::Op::FusedMatMulBTAdd
    } else {
        meganeura::graph::Op::FusedMatMulAdd
    };
    graph.nodes_mut()[y as usize].inputs.push(d);
    graph.set_outputs(vec![y]);
    for workers in [0, 1, 3, 256] {
        if workers != 0 && !supports_graph {
            continue;
        }
        let mut config = meganeura::SessionConfig::inference_from_env_on(gpu.clone());
        config.optimize.mode = meganeura::OptimizeMode::Off;
        config.tune = false;
        config.runtime.poison = true;
        config.runtime.persistent = (workers != 0).then_some(meganeura::PersistentOptions {
            workers,
            device_scheduling: true,
            whole_graph: true,
            allow_fallback: false,
            track_workers: true,
        });
        let mut session = meganeura::build(&graph, config).0;
        let dispatch = &session.plan().dispatches[0];
        let (rows, cols) = if transposed_b {
            assert_eq!(dispatch.kernel, meganeura::compile::Kernel::Default);
            (64, 64)
        } else {
            let meganeura::compile::Kernel::ScalarMatmul(shape) = dispatch.kernel else {
                panic!(
                    "narrow probe did not select scalar geometry: {:?}",
                    dispatch.kernel
                );
            };
            assert_eq!(shape.rows(), 32);
            assert_eq!(shape.k_stage, 32);
            (shape.rows(), shape.cols())
        };
        assert_eq!(
            dispatch.workgroups,
            [(n as u32).div_ceil(cols), (m as u32).div_ceil(rows), 1]
        );
        for seed in [3, 7] {
            let values = |len, salt| {
                (0..len)
                    .map(|i| ((i * seed + salt) % 7) as f32 / 32.0 - 3.0 / 32.0)
                    .collect::<Vec<_>>()
            };
            let av = values(m * k, 1);
            let bv = values(k * n, 2);
            let dv = values(m * n, 4);
            let mut expected = dv.clone();
            for row in 0..m {
                for col in 0..n {
                    let mut sum = 0.0;
                    for inner in 0..k {
                        let b_index = if transposed_b {
                            col * k + inner
                        } else {
                            inner * n + col
                        };
                        sum += av[row * k + inner] * bv[b_index];
                    }
                    expected[row * n + col] += sum;
                }
            }
            session.set_input("a", &av);
            session.set_input("b", &bv);
            session.set_input("d", &dv);
            session.step();
            session.wait();
            if workers != 0 {
                check_worker_tasks(&mut session);
                assert_eq!(session.execution_report().compute_dispatches, 1);
            }
            let mut actual = vec![0.0; m * n];
            session.read_output_by_index(0, &mut actual);
            assert_eq!(
                actual, expected,
                "transposed_b={transposed_b}, workers={workers}, seed={seed}"
            );
        }
    }
}

fn check_worker_tasks(session: &mut meganeura::Session) {
    session.validate_persistent_completion().unwrap();
    let report = session.execution_report();
    let enabled = report
        .persistent
        .is_some_and(|options| options.track_workers)
        && report.regions.iter().any(|region| region.workers != 0);
    let rows = session.read_persistent_worker_tasks();
    assert_eq!(rows.is_some(), enabled);
    if let Some(rows) = rows {
        assert_eq!(rows.len(), report.regions.len());
        for (row, region) in rows.iter().zip(&report.regions) {
            assert_eq!(row.len(), region.workers as usize);
            if region.workers != 0 {
                assert_eq!(
                    row.iter().map(|&count| u64::from(count)).sum::<u64>(),
                    region.logical_tasks
                );
            }
        }
    }
    let appended = session.read_persistent_appended_worker_tasks();
    assert_eq!(
        appended.is_some(),
        report
            .persistent
            .is_some_and(|options| options.track_workers)
            && report.appended.iter().any(|row| row.workers != 0)
    );
    if let Some(rows) = appended {
        assert_eq!(rows.len(), report.appended.len());
        for (row, dispatch) in rows.iter().zip(&report.appended) {
            assert_eq!(row.len(), dispatch.workers as usize);
            assert_eq!(
                row.iter().map(|&count| u64::from(count)).sum::<u64>(),
                dispatch.logical_tasks
            );
        }
    }
}

#[test]
fn session_persistent_optimizer_tiles_preserve_training_state() {
    check_optimizer_tiles(false);
}

#[test]
fn session_whole_graph_optimizer_uses_one_dispatch() {
    check_optimizer_tiles(true);
}

#[test]
fn session_whole_graph_wait_preserves_unsubmitted_parameters() {
    let gpu = crate::support::gpu::gpu();
    if !gpu.capabilities().device_memory_fence || !gpu.capabilities().binding_array {
        return;
    }
    let mut graph = meganeura::Graph::new();
    let p = graph.parameter("p", &[257]);
    let x = graph.input("x", &[257]);
    let loss = graph.mse_loss(p, x);
    graph.set_outputs(vec![loss]);
    let mut sessions: Vec<_> = [false, true]
        .into_iter()
        .map(|persistent| {
            let mut config = crate::support::gpu::config();
            config.runtime.persistent = persistent.then_some(meganeura::PersistentOptions {
                workers: 3,
                device_scheduling: true,
                whole_graph: true,
                allow_fallback: false,
                track_workers: true,
            });
            let mut session = meganeura::build(&graph, config).0;
            session.set_parameter("p", &vec![1.0; 257]);
            session.set_input("x", &vec![0.5; 257]);
            session
        })
        .collect();
    for lr in [0.01, 0.02, 0.03] {
        sessions[0].set_learning_rate(lr);
        sessions[0].step();
    }
    sessions[0].wait();
    let session = &mut sessions[1];
    let mut first = gpu.create_command_encoder(blade_graphics::CommandEncoderDesc {
        name: "submitted graph",
        buffer_count: 1,
        manual_barriers: false,
    });
    let mut later = gpu.create_command_encoder(blade_graphics::CommandEncoderDesc {
        name: "pending graphs",
        buffer_count: 1,
        manual_barriers: false,
    });
    first.start();
    session.set_learning_rate(0.01);
    session.record(&mut first).unwrap();
    session.track_submission(gpu.submit(&mut first));
    later.start();
    session.set_learning_rate(0.02);
    session.record(&mut later).unwrap();
    session.wait();
    session.set_learning_rate(0.03);
    session.record(&mut later).unwrap();
    session.track_submission(gpu.submit(&mut later));
    session.wait();
    check_worker_tasks(session);
    gpu.destroy_command_encoder(&mut first);
    gpu.destroy_command_encoder(&mut later);
    let mut expected = vec![0.0; 257];
    let mut actual = expected.clone();
    sessions[0].read_param("p", &mut expected);
    sessions[1].read_param("p", &mut actual);
    assert_eq!(actual, expected);
}

#[test]
fn session_whole_graph_mixes_128_and_256_lane_reductions() {
    let gpu = crate::support::gpu::gpu();
    if !gpu.capabilities().device_memory_fence || !gpu.capabilities().binding_array {
        return;
    }
    let mut graph = meganeura::Graph::new();
    let x = graph.input("x", &[37, 259]);
    let w = graph.parameter("w", &[259]);
    let b = graph.parameter("b", &[259]);
    let norm = graph.layer_norm(x, w, b, 1e-5);
    let pool = graph.global_avg_pool(norm, 37, 1, 259);
    graph.set_outputs(vec![norm, pool]);
    let plan = meganeura::compile::compile_with(
        &graph,
        &meganeura::CompileOptions {
            fuse_dispatches: false,
            ..Default::default()
        },
    );
    for workers in [1, 64, 4096] {
        let mut sessions: Vec<_> = [false, true]
            .into_iter()
            .map(|persistent| {
                meganeura::Session::with_context_opts(
                    plan.clone(),
                    gpu.clone(),
                    meganeura::SessionOptions {
                        persistent: persistent.then_some(meganeura::PersistentOptions {
                            workers,
                            device_scheduling: true,
                            whole_graph: true,
                            track_workers: true,
                            allow_fallback: false,
                        }),
                        poison: true,
                        coop: meganeura::CoopPolicy::Disabled,
                        ..meganeura::SessionOptions::from_env()
                    },
                )
            })
            .collect();
        for replay in 0..3 {
            for session in &mut sessions {
                session.set_input(
                    "x",
                    &(0..37 * 259)
                        .map(|i| ((i * 17 + replay * 19) % 113) as f32 / 16.0 - 3.0)
                        .collect::<Vec<_>>(),
                );
                session.set_parameter(
                    "w",
                    &(0..259).map(|i| (i % 11) as f32 / 16.0).collect::<Vec<_>>(),
                );
                session.set_parameter(
                    "b",
                    &(0..259).map(|i| (i % 7) as f32 / 32.0).collect::<Vec<_>>(),
                );
                session.step();
                check_worker_tasks(session);
            }
            assert_eq!(sessions[1].execution_report().compute_dispatches, 1);
            for (index, len) in [37 * 259, 37].into_iter().enumerate() {
                let mut expected = vec![0.0; len];
                let mut actual = expected.clone();
                sessions[0].read_output_by_index(index, &mut expected);
                sessions[1].read_output_by_index(index, &mut actual);
                assert_eq!(
                    actual, expected,
                    "workers={workers}, replay={replay}, output={index}"
                );
            }
        }
    }
}

fn check_optimizer_tiles(whole_graph: bool) {
    let gpu = crate::support::gpu::gpu();
    if whole_graph && !gpu.capabilities().binding_array {
        return;
    }
    // Whole-graph SGD must exercise a large grid with a partial final batch,
    // followed by small chunks sharing the same cached optimizer handler.
    let first_len = if whole_graph { 1024 * 1024 + 17 } else { 8193 };
    let parameters = [("a", "x", first_len), ("b", "y", 1031), ("c", "z", 33)];
    let names = ["a", "b", "c"];
    let mut graph = meganeura::Graph::new();
    let mut losses = Vec::new();
    for (name, input, len) in parameters {
        let param = graph.parameter(name, &[1, len]);
        let target = graph.input(input, &[1, len]);
        losses.push(graph.mse_loss(param, target));
    }
    let sum = graph.add(losses[0], losses[1]);
    let loss = graph.add(sum, losses[2]);
    graph.set_outputs(vec![loss]);
    for device_scheduling in [false, true] {
        if whole_graph && !device_scheduling {
            continue;
        }
        if device_scheduling && !gpu.capabilities().device_memory_fence {
            continue;
        }
        for workers in [1, 3, 64, 4096] {
            let mut sessions: Vec<_> = [false, true]
                .into_iter()
                .map(|persistent| {
                    let mut config = crate::support::gpu::config();
                    config.runtime.coop = meganeura::CoopPolicy::Disabled;
                    config.runtime.poison = true;
                    config.runtime.arena_chunk_bytes = Some(1024);
                    config.runtime.persistent =
                        persistent.then_some(meganeura::PersistentOptions {
                            workers,
                            track_workers: true,
                            allow_fallback: false,
                            device_scheduling,
                            whole_graph,
                        });
                    let mut session = meganeura::build(&graph, config).0;
                    for (name, _, len) in parameters {
                        let values: Vec<_> =
                            (0..len).map(|i| (i % 31) as f32 / 16.0 - 0.75).collect();
                        session.set_parameter(name, &values);
                    }
                    session.set_adam_grouped_grad_norm("c", 3);
                    session.set_submission_chunks(if whole_graph { 1 } else { 3 });
                    session
                })
                .collect();
            for stage in 0..8 {
                for session in &mut sessions {
                    for (_, input, len) in parameters {
                        let values: Vec<_> = (0..len)
                            .map(|i| ((i * 7 + stage * 19) % 53) as f32 / 32.0 - 0.5)
                            .collect();
                        session.set_input(input, &values);
                    }
                    match stage {
                        0 => {
                            session.set_learning_rate(0.01);
                            session.set_grad_clip_norm(0.02);
                        }
                        2 => {
                            session.set_adam(0.001, 0.9, 0.99, 1e-8);
                            session.set_adaptive_grad_clip(0.001, 0.01);
                        }
                        4 => {
                            session.set_laprop(0.001, 0.9, 0.99, 1e-8);
                            session.set_grad_clip_norm(0.02);
                        }
                        5 => {
                            session.set_grad_accumulate(2);
                            session.zero_grad();
                            session.clear_optimizer();
                        }
                        6 => {
                            session.set_adam(0.001, 0.9, 0.99, 1e-8);
                            session.set_adaptive_grad_clip(0.001, 0.01);
                        }
                        7 => {
                            session.clear_grad_accumulate();
                            session.clear_optimizer();
                            session.disable_grad_clip();
                        }
                        _ => {}
                    }
                    session.set_profiling(stage == 3);
                    if stage == 1 || stage == 3 {
                        let mut encoder =
                            gpu.create_command_encoder(blade_graphics::CommandEncoderDesc {
                                name: "persistent training reset epochs",
                                buffer_count: 1,
                                manual_barriers: false,
                            });
                        encoder.start();
                        session.record(&mut encoder).unwrap();
                        session.record(&mut encoder).unwrap();
                        let sync = gpu.submit(&mut encoder);
                        session.track_submission(sync);
                        session.wait();
                        gpu.destroy_command_encoder(&mut encoder);
                    } else {
                        session.step();
                        session.wait();
                    }
                    if stage == 7 {
                        session.sgd_step(0.01);
                    }
                    check_worker_tasks(session);
                }
                let ordinary = &sessions[0];
                let selected = &sessions[1];
                let report = selected.execution_report();
                if whole_graph {
                    assert_eq!(report.compute_dispatches, 1);
                    assert!(report.appended.iter().all(|row| row.device_scheduled));
                }
                assert!(report.appended_fallback.is_empty());
                assert!(
                    report
                        .appended
                        .iter()
                        .all(|row| row.workers == workers && row.fallback_reason.is_none())
                );
                assert_eq!(
                    report.appended.len(),
                    match stage {
                        0 | 1 | 4 => 10,
                        2 | 3 => 12,
                        6 => 15,
                        _ => 3,
                    }
                );
                let reference = ordinary.execution_report();
                for (row, expected) in report.appended.iter().zip(&reference.appended) {
                    assert_eq!(
                        (&row.label, &row.implementation_key, row.workgroups),
                        (
                            &expected.label,
                            &expected.implementation_key,
                            expected.workgroups
                        )
                    );
                }
                for (name, _, len) in parameters {
                    let mut expected = vec![0.0; len];
                    let mut actual = vec![0.0; len];
                    ordinary.read_param(name, &mut expected);
                    selected.read_param(name, &mut actual);
                    assert_eq!(
                        actual, expected,
                        "parameter {name}, stage={stage}, workers={workers}"
                    );
                    ordinary.read_param_grad(name, &mut expected);
                    selected.read_param_grad(name, &mut actual);
                    assert_eq!(
                        actual, expected,
                        "gradient {name}, stage={stage}, workers={workers}"
                    );
                }
                assert_eq!(
                    selected.read_adam_states(&names),
                    ordinary.read_adam_states(&names)
                );
                assert_eq!(selected.adam_step_count(), ordinary.adam_step_count());
                assert_eq!(
                    selected.read_adam_grouped_grad_norm("c"),
                    ordinary.read_adam_grouped_grad_norm("c")
                );
            }
        }
    }
}

#[test]
fn session_device_graphs_release_dependencies_across_worker_tiles() {
    check_device_graphs(false);
}

#[test]
fn session_whole_graph_releases_all_nodes() {
    check_device_graphs(true);
}

fn check_device_graphs(whole_graph: bool) {
    let gpu = crate::support::gpu::gpu();
    if whole_graph && !gpu.capabilities().binding_array {
        return;
    }
    if !gpu.capabilities().device_memory_fence {
        eprintln!("device graph fences are unsupported on this backend");
        return;
    }
    for case in 0..5 {
        let mut graph = meganeura::Graph::new();
        let shape = if case == 2 { [0, 259] } else { [37, 259] };
        let x = graph.input("x", &shape);
        let positive = graph.relu(x);
        if case == 0 || case == 4 {
            let negative = graph.neg(x);
            let sum = graph.add(positive, negative);
            let mut outputs = vec![positive, negative, sum];
            if case == 4 {
                outputs.push(graph.neg(sum));
            }
            graph.set_outputs(outputs);
        } else if case == 1 {
            let sum = graph.sum_inner(positive);
            let negative = graph.neg(sum);
            graph.set_outputs(vec![positive, sum, negative]);
        } else if case == 2 {
            graph.set_outputs(vec![positive]);
        } else {
            let empty = graph.input("empty", &[0]);
            let empty = graph.relu(empty);
            let negative = graph.neg(positive);
            graph.set_outputs(vec![positive, empty, negative]);
        }
        for (workers, serial) in [
            (1, false),
            (3, false),
            (64, false),
            (4096, false),
            (3, true),
        ] {
            let mut config = meganeura::SessionConfig::inference_from_env_on(gpu.clone());
            config.optimize.mode = meganeura::OptimizeMode::Off;
            config.options.fuse_dispatches = false;
            config.runtime.poison = true;
            config.runtime.serial_dispatch = serial;
            config.runtime.persistent = Some(meganeura::PersistentOptions {
                workers,
                track_workers: workers == 3,
                allow_fallback: false,
                device_scheduling: true,
                whole_graph,
            });
            let mut session = meganeura::build(&graph, config).0;
            let report = session.execution_report();
            let operations: usize = if case == 2 {
                1
            } else if case == 4 {
                4
            } else {
                3
            };
            assert_eq!(
                report.graph_dispatches,
                if whole_graph {
                    1
                } else if serial {
                    operations
                } else {
                    operations.div_ceil(3)
                }
            );
            assert!(report.regions.iter().all(|region| region.device_scheduled
                == (whole_graph || (region.dispatches.len() > 1 && case != 2))
                && !region.worker_owned_chain));
            assert_eq!(
                report
                    .regions
                    .iter()
                    .map(|region| region.dispatches.len())
                    .sum::<usize>(),
                operations
            );
            assert_eq!(
                session.memory_summary().scheduler_bytes,
                report.scheduler_bytes
            );
            for replay in 0..3 {
                let input: Vec<_> = (0..shape[0] * shape[1])
                    .map(|i| ((i * 17 + replay * 23) % 101) as f32 / 32.0 - 1.0)
                    .collect();
                let positive: Vec<_> = input.iter().map(|value| value.max(0.0)).collect();
                let expected = if case == 0 || case == 4 {
                    let mut values = vec![
                        positive,
                        input.iter().map(|value| -value).collect(),
                        input.iter().map(|value| (-value).max(0.0)).collect(),
                    ];
                    if case == 4 {
                        values.push(values[2].iter().map(|value| -value).collect());
                    }
                    values
                } else if case == 1 {
                    let sums: Vec<f32> = positive.chunks(259).map(|row| row.iter().sum()).collect();
                    let negative = sums.iter().map(|value| -value).collect();
                    vec![positive, sums, negative]
                } else if case == 2 {
                    vec![positive]
                } else {
                    let negative = positive.iter().map(|value| -value).collect();
                    vec![positive, Vec::new(), negative]
                };
                session.set_input("x", &input);
                if case == 3 {
                    session.set_input("empty", &[]);
                }
                session.set_submission_chunks(if whole_graph { 1 } else { 3 });
                session.set_profiling(replay == 1);
                if replay < 2 {
                    session.step();
                    session.wait();
                } else {
                    let mut encoder =
                        gpu.create_command_encoder(blade_graphics::CommandEncoderDesc {
                            name: "device region reset epochs",
                            buffer_count: 1,
                            manual_barriers: false,
                        });
                    encoder.start();
                    session.record(&mut encoder).unwrap();
                    session.record(&mut encoder).unwrap();
                    let sync = gpu.submit(&mut encoder);
                    session.track_submission(sync);
                    session.wait();
                    gpu.destroy_command_encoder(&mut encoder);
                }
                check_worker_tasks(&mut session);
                for (index, expected) in expected.into_iter().enumerate() {
                    let mut actual = vec![f32::NAN; expected.len()];
                    session.read_output_by_index(index, &mut actual);
                    assert_eq!(
                        actual, expected,
                        "case={case}, workers={workers}, replay={replay}"
                    );
                }
            }
        }
    }
    let mut graph = meganeura::Graph::new();
    let x = graph.input("x", &[1025]);
    let positive = graph.relu(x);
    let negative = graph.neg(x);
    let sum = graph.add(positive, negative);
    graph.set_outputs(vec![positive, negative, sum]);
    let mut plan = meganeura::compile::compile_with(
        &graph,
        &meganeura::CompileOptions {
            fuse_dispatches: false,
            ..Default::default()
        },
    );
    assert_eq!(plan.dispatches.len(), 3);
    let input = plan.input_buffers[0].1;
    assert_eq!(
        plan.dispatches[2].input_buffers[1],
        plan.dispatches[1].output_buffer
    );
    // Reuse the input for the second and third writes: RAW, WAR and WAW
    // edges must all follow physical accesses, including the in-place read.
    plan.dispatches[1].output_buffer = input;
    plan.dispatches[2].input_buffers[1] = input;
    plan.dispatches[2].output_buffer = input;
    plan.output_buffers[1] = input;
    plan.output_buffers[2] = input;
    for workers in [1, 3, 4096] {
        let options = meganeura::SessionOptions {
            persistent: Some(meganeura::PersistentOptions {
                workers,
                device_scheduling: true,
                allow_fallback: false,
                track_workers: true,
                whole_graph,
            }),
            ..meganeura::SessionOptions::from_env()
        };
        let mut session = meganeura::Session::with_context_opts(plan.clone(), gpu.clone(), options);
        assert_eq!(session.execution_dispatch_count(), 1);
        for seed in 0..8 {
            let input: Vec<_> = (0..1025)
                .map(|i| ((i * 19 + seed * 17) % 101) as f32 - 50.0)
                .collect();
            session.set_input("x", &input);
            session.step();
            check_worker_tasks(&mut session);
            let mut actual = vec![f32::NAN; input.len()];
            session.read_output_by_index(0, &mut actual);
            assert_eq!(actual, input.iter().map(|x| x.max(0.0)).collect::<Vec<_>>());
            session.read_output_by_index(2, &mut actual);
            assert_eq!(
                actual,
                input.iter().map(|x| (-x).max(0.0)).collect::<Vec<_>>()
            );
        }
    }
}

#[test]
fn session_persistent_tiles_use_normal_graph_storage_and_reset() {
    let gpu = crate::support::gpu::gpu();
    let mut graph = meganeura::Graph::new();
    let x = graph.input("x", &[37, 259]);
    let relu = graph.relu(x);
    let square = graph.mul(relu, relu);
    let sum = graph.sum_inner(square);
    graph.set_outputs(vec![relu, sum]);
    for workers in [1, 3, 64, 256] {
        let mut config = meganeura::SessionConfig::inference_from_env_on(gpu.clone());
        config.optimize.mode = meganeura::OptimizeMode::Off;
        config.options.fuse_dispatches = false;
        config.runtime.poison = true;
        config.runtime.persistent = Some(meganeura::PersistentOptions {
            workers,
            track_workers: workers == 3,
            device_scheduling: false,
            allow_fallback: false,
            whole_graph: false,
        });
        let mut session = meganeura::build(&graph, config).0;
        let report = session.execution_report();
        assert!(report.persistent_tasks > 0);
        assert!(report.graph_dispatches < session.plan().dispatches.len());
        assert!(
            report
                .regions
                .iter()
                .any(|region| region.worker_owned_chain)
        );
        assert!(
            report
                .regions
                .iter()
                .all(|r| r.workers == workers && r.fallback_reason.is_none())
        );
        assert_eq!(
            session.memory_summary().scheduler_bytes,
            report.scheduler_bytes
        );
        assert!(report.regions.iter().any(|r| r.logical_tasks > 1));
        for iteration in 0..3 {
            let input: Vec<_> = (0..37 * 259)
                .map(|i| ((i * 17 + iteration * 23) % 101) as f32 / 32.0 - 1.0)
                .collect();
            session.set_input("x", &input);
            if iteration == 0 {
                session.set_submission_chunks(3);
                session.step();
                session.wait();
                check_worker_tasks(&mut session);
            } else if iteration == 1 {
                session.set_profiling(true);
                session.step();
                session.wait();
                check_worker_tasks(&mut session);
                session.set_profiling(false);
            } else {
                let mut encoder = gpu.create_command_encoder(blade_graphics::CommandEncoderDesc {
                    name: "persistent repeated records",
                    buffer_count: 1,
                    manual_barriers: false,
                });
                encoder.start();
                session.record(&mut encoder).unwrap();
                session.record(&mut encoder).unwrap();
                let sync = gpu.submit(&mut encoder);
                session.track_submission(sync);
                session.wait();
                check_worker_tasks(&mut session);
                gpu.destroy_command_encoder(&mut encoder);
            }
            let mut actual = vec![f32::NAN; input.len()];
            session.read_output_by_index(0, &mut actual);
            for (&value, &input) in actual.iter().zip(&input) {
                assert_eq!(value, input.max(0.0));
            }
            let mut reduced = vec![f32::NAN; 37];
            session.read_output_by_index(1, &mut reduced);
            for (&value, row) in reduced.iter().zip(input.as_chunks::<259>().0) {
                let expected: f64 = row.iter().map(|&x| f64::from(x.max(0.0)).powi(2)).sum();
                assert!(
                    (f64::from(value) - expected).abs() <= 2e-5 * expected.max(1.0),
                    "{value} vs {expected}"
                );
            }
        }
    }

    // Explicit in-place storage is separate from lifetime allocation reuse.
    // Keep the intermediate as a graph output so compilation retains both ops.
    let mut graph = meganeura::Graph::new();
    let input = graph.input("x", &[513]);
    let mapped = graph.relu(input);
    let output = graph.neg(mapped);
    graph.set_outputs(vec![mapped, output]);
    let mut plan = meganeura::compile::compile_with(
        &graph,
        &meganeura::CompileOptions {
            fuse_dispatches: false,
            ..Default::default()
        },
    );
    assert_eq!(plan.dispatches.len(), 2);
    let input_buffer = plan.input_buffers[0].1;
    let intermediate = plan.dispatches[0].output_buffer;
    plan.dispatches[0].output_buffer = input_buffer;
    assert_eq!(plan.dispatches[1].input_buffers[0], intermediate);
    plan.dispatches[1].input_buffers[0] = input_buffer;
    plan.output_buffers[0] = input_buffer;
    plan.node_buffers[mapped as usize].1 = input_buffer;
    for (workers, device_scheduling) in [(1, false), (64, false), (1, true), (64, true)] {
        if device_scheduling && !gpu.capabilities().device_memory_fence {
            continue;
        }
        let mut options = meganeura::SessionOptions::from_env();
        options.persistent = Some(meganeura::PersistentOptions {
            workers,
            track_workers: workers == 3,
            device_scheduling,
            allow_fallback: false,
            whole_graph: false,
        });
        let mut session = meganeura::Session::with_context_opts(plan.clone(), gpu.clone(), options);
        let input: Vec<_> = (0..513).map(|i| (i % 17) as f32 - 8.0).collect();
        session.set_input("x", &input);
        session.step();
        session.wait();
        check_worker_tasks(&mut session);
        let mut mapped = vec![f32::NAN; 513];
        session.read_output_by_index(0, &mut mapped);
        let mut output = vec![f32::NAN; 513];
        session.read_output_by_index(1, &mut output);
        assert_eq!(mapped, input.iter().map(|x| x.max(0.0)).collect::<Vec<_>>());
        assert_eq!(output, mapped.iter().map(|x| -x).collect::<Vec<_>>());
        assert!(
            session
                .execution_report()
                .regions
                .iter()
                .all(|region| region.workers == workers)
        );
    }
}

#[test]
fn session_persistent_regions_combine_independent_ragged_tasks() {
    let gpu = crate::support::gpu::gpu();
    let mut graph = meganeura::Graph::new();
    let lengths = [1, 17, 513, 4097, 7939];
    let mut outputs = Vec::new();
    for (index, &length) in lengths.iter().enumerate() {
        let input = graph.input(&format!("x{index}"), &[length]);
        let positive = graph.relu(input);
        let negative = graph.neg(positive);
        outputs.extend([positive, negative]);
    }
    graph.set_outputs(outputs);
    for workers in [1, 3, 256] {
        let mut config = meganeura::SessionConfig::inference_from_env_on(gpu.clone());
        config.optimize.mode = meganeura::OptimizeMode::Off;
        config.options.fuse_dispatches = false;
        config.runtime.persistent = Some(meganeura::PersistentOptions {
            workers,
            track_workers: workers == 3,
            device_scheduling: false,
            allow_fallback: false,
            whole_graph: false,
        });
        let mut session = meganeura::build(&graph, config).0;
        let report = session.execution_report();
        assert_eq!(session.plan().dispatches.len(), 10);
        assert_eq!(report.graph_dispatches, 4);
        assert_eq!(session.execution_dispatch_count(), 4);
        let keys = session.dispatch_pipeline_keys();
        assert!(
            report
                .regions
                .iter()
                .all(|region| region.workers == workers && region.fallback_reason.is_none())
        );
        assert_eq!(
            report
                .regions
                .iter()
                .map(|region| region.dispatches.len())
                .sum::<usize>(),
            10
        );
        for region in &report.regions {
            assert!((2..=3).contains(&region.dispatches.len()));
            assert!(
                region
                    .dispatches
                    .clone()
                    .all(|index| keys[index] == keys[region.dispatches.start])
            );
        }
        for replay in 0..3 {
            let inputs: Vec<Vec<f32>> = lengths
                .iter()
                .enumerate()
                .map(|(index, &length)| {
                    (0..length)
                        .map(|i| ((i * 17 + index * 31 + replay * 13) % 53) as f32 - 26.0)
                        .collect()
                })
                .collect();
            for (index, input) in inputs.iter().enumerate() {
                session.set_input(&format!("x{index}"), input);
            }
            session.set_profiling(replay == 1);
            session.set_submission_chunks(3);
            if replay < 2 {
                session.step();
                session.wait();
                check_worker_tasks(&mut session);
            } else {
                let mut encoder = gpu.create_command_encoder(blade_graphics::CommandEncoderDesc {
                    name: "merged region replay",
                    buffer_count: 1,
                    manual_barriers: false,
                });
                encoder.start();
                session.record(&mut encoder).unwrap();
                session.record(&mut encoder).unwrap();
                let sync = gpu.submit(&mut encoder);
                session.track_submission(sync);
                session.wait();
                check_worker_tasks(&mut session);
                gpu.destroy_command_encoder(&mut encoder);
            }
            for (index, input) in inputs.iter().enumerate() {
                let mut output = vec![f32::NAN; input.len()];
                session.read_output_by_index(2 * index, &mut output);
                assert_eq!(
                    output,
                    input.iter().map(|value| value.max(0.0)).collect::<Vec<_>>()
                );
                session.read_output_by_index(2 * index + 1, &mut output);
                assert_eq!(
                    output,
                    input
                        .iter()
                        .map(|value| -value.max(0.0))
                        .collect::<Vec<_>>()
                );
            }
        }
    }
    let mut graph = meganeura::Graph::new();
    let shapes = [[37, 259], [9, 513], [17, 1025]];
    let outputs = shapes
        .iter()
        .enumerate()
        .map(|(index, shape)| {
            let input = graph.input(&format!("x{index}"), shape);
            graph.sum_inner(input)
        })
        .collect();
    graph.set_outputs(outputs);
    let config = || {
        let mut config = meganeura::SessionConfig::inference_from_env_on(gpu.clone());
        config.optimize.mode = meganeura::OptimizeMode::Off;
        config.options.fuse_dispatches = false;
        config.runtime.persistent = Some(meganeura::PersistentOptions {
            workers: 3,
            track_workers: true,
            device_scheduling: false,
            allow_fallback: false,
            whole_graph: false,
        });
        config
    };
    let mut session = meganeura::build(&graph, config()).0;
    assert!(
        session
            .execution_report()
            .regions
            .iter()
            .any(|region| region.dispatches.len() > 1)
    );
    for replay in 0..3 {
        for (index, shape) in shapes.iter().enumerate() {
            let input: Vec<_> = (0..shape[0] * shape[1])
                .map(|i| ((i * 7 + replay * 13) % 53) as f32 / 16.0 - 1.0)
                .collect();
            session.set_input(&format!("x{index}"), &input);
        }
        session.step();
        session.wait();
        check_worker_tasks(&mut session);
        for (index, shape) in shapes.iter().enumerate() {
            let mut output = vec![f32::NAN; shape[0]];
            session.read_output_by_index(index, &mut output);
            for (row, value) in output.into_iter().enumerate() {
                let expected: f64 = (row * shape[1]..(row + 1) * shape[1])
                    .map(|i| ((i * 7 + replay * 13) % 53) as f64 / 16.0 - 1.0)
                    .sum();
                assert!((f64::from(value) - expected).abs() < 1e-4);
            }
        }
    }
    let mut diamond = meganeura::Graph::new();
    let input = diamond.input("x", &[513]);
    let positive = diamond.relu(input);
    let negative = diamond.neg(input);
    let joined = diamond.add(positive, negative);
    diamond.set_outputs(vec![positive, negative, joined]);
    let mut session = meganeura::build(&diamond, config()).0;
    assert_eq!(session.execution_dispatch_count(), 2);
    assert_eq!(session.execution_report().regions[0].dispatches.len(), 2);
    let input: Vec<_> = (0..513).map(|i| (i % 17) as f32 - 8.0).collect();
    session.set_input("x", &input);
    session.step();
    session.wait();
    check_worker_tasks(&mut session);
    let mut output = vec![f32::NAN; input.len()];
    session.read_output_by_index(2, &mut output);
    assert_eq!(
        output,
        input
            .iter()
            .map(|value| (-value).max(0.0))
            .collect::<Vec<_>>()
    );
    let mut empty = meganeura::Graph::new();
    let input = empty.input("x", &[0]);
    let output = empty.relu(input);
    empty.set_outputs(vec![output]);
    let mut session = meganeura::build(&empty, config()).0;
    assert_eq!(session.execution_report().persistent_tasks, 0);
    session.step();
    session.wait();
    check_worker_tasks(&mut session);
    session.step();
    session.wait();
    check_worker_tasks(&mut session);
    // Rebinding can introduce a physical dependency between distinct logical
    // parameters. The queue must be rebuilt before those tasks execute.
    let mut source_graph = meganeura::Graph::new();
    let a = source_graph.parameter("a", &[513]);
    let b = source_graph.parameter("b", &[513]);
    source_graph.set_outputs(vec![a, b]);
    let mut source_plan = meganeura::compile::compile_with(&source_graph, &Default::default());
    source_plan.param_buffers[1].1 = source_plan.param_buffers[0].1;
    let mut source_options = meganeura::SessionOptions::from_env();
    source_options.persistent = None;
    let mut source =
        meganeura::Session::with_context_opts(source_plan, gpu.clone(), source_options);
    let mut graph = meganeura::Graph::new();
    let a = graph.parameter("a", &[513]);
    let b = graph.parameter("b", &[513]);
    let positive = graph.relu(a);
    let negative = graph.neg(b);
    graph.set_outputs(vec![positive, negative]);
    let mut plan = meganeura::compile::compile_with(
        &graph,
        &meganeura::CompileOptions {
            fuse_dispatches: false,
            ..Default::default()
        },
    );
    let a_buffer = plan.param_buffers[0].1;
    assert_eq!(plan.dispatches[0].input_buffers[0], a_buffer);
    plan.dispatches[0].output_buffer = a_buffer;
    plan.output_buffers[0] = a_buffer;
    plan.node_buffers[positive as usize].1 = a_buffer;
    for (workers, device_scheduling) in [(1, false), (64, false), (1, true), (64, true)] {
        if device_scheduling && !gpu.capabilities().device_memory_fence {
            continue;
        }
        let mut options = meganeura::SessionOptions::from_env();
        options.persistent = Some(meganeura::PersistentOptions {
            workers,
            track_workers: workers == 3,
            device_scheduling,
            allow_fallback: false,
            whole_graph: false,
        });
        let mut session = meganeura::Session::with_context_opts(plan.clone(), gpu.clone(), options);
        assert_eq!(session.execution_dispatch_count(), 1);
        session.share_parameter_from(&mut source, "a").unwrap();
        session.share_parameter_from(&mut source, "b").unwrap();
        assert_eq!(
            session.execution_dispatch_count(),
            if device_scheduling { 1 } else { 2 }
        );
        let input: Vec<_> = (0..513).map(|i| (i % 17) as f32 - 8.0).collect();
        source.set_parameter("a", &input);
        session.step();
        session.wait();
        check_worker_tasks(&mut session);
        let mut output = vec![f32::NAN; input.len()];
        session.read_output_by_index(1, &mut output);
        assert_eq!(
            output,
            input
                .iter()
                .map(|value| -value.max(0.0))
                .collect::<Vec<_>>()
        );
    }
}

#[test]
fn session_persistent_lifetime_reuse_orders_physical_dependencies() {
    check_lifetime_reuse(false);
}

#[test]
fn session_whole_graph_lifetime_reuse() {
    check_lifetime_reuse(true);
}

#[test]
fn session_whole_graph_gradient_write_ranges() {
    let gpu = crate::support::gpu::gpu();
    if !gpu.capabilities().device_memory_fence || !gpu.capabilities().binding_array {
        return;
    }
    let (rows, inner, columns) = (17, 33, 65);
    let mut graph = meganeura::Graph::new();
    let x = graph.input("x", &[rows, inner]);
    let target = graph.input("target", &[rows, columns]);
    let mut losses = Vec::new();
    for i in 0..3 {
        let w = graph.parameter(&format!("w{i}"), &[inner, columns]);
        let b = graph.parameter(&format!("b{i}"), &[columns]);
        let product = graph.matmul(x, w);
        let biased = graph.bias_add(product, b);
        losses.push(graph.mse_loss(biased, target));
    }
    let sum = graph.add(losses[0], losses[1]);
    let sum = graph.add(sum, losses[2]);
    graph.set_outputs(vec![sum]);
    for workers in [1, 3, 64, 4096] {
        let mut sessions: Vec<_> = [false, true]
            .into_iter()
            .map(|persistent| {
                let mut config = crate::support::gpu::config();
                config.optimize.mode = meganeura::OptimizeMode::Off;
                config.tune = false;
                config.runtime.coop = meganeura::CoopPolicy::Disabled;
                config.runtime.poison = true;
                config.runtime.arena_chunk_bytes = Some(1 << 20);
                config.runtime.persistent = persistent.then_some(meganeura::PersistentOptions {
                    workers,
                    device_scheduling: true,
                    whole_graph: true,
                    allow_fallback: false,
                    track_workers: true,
                });
                let mut session = meganeura::build(&graph, config).0;
                for i in 0..3 {
                    session.set_parameter(
                        &format!("w{i}"),
                        &vec![(i + 1) as f32 / 64.0; inner * columns],
                    );
                    session.set_parameter(&format!("b{i}"), &vec![-0.125; columns]);
                }
                assert!(
                    session
                        .plan()
                        .dispatches
                        .iter()
                        .any(|d| d.shader == meganeura::compile::ShaderEntry::MatMulAT)
                );
                assert!(
                    session
                        .plan()
                        .dispatches
                        .iter()
                        .any(|d| d.shader == meganeura::compile::ShaderEntry::SumRows)
                );
                session
            })
            .collect();
        assert_eq!(
            sessions[0].dispatch_implementation_keys(),
            sessions[1].dispatch_implementation_keys()
        );
        for replay in 0..3 {
            for session in &mut sessions {
                session.set_input(
                    "x",
                    &(0..rows * inner)
                        .map(|i| ((i * 7 + replay * 13) % 31) as f32 / 64.0 - 0.25)
                        .collect::<Vec<_>>(),
                );
                session.set_input("target", &vec![0.0625; rows * columns]);
                if replay == 0 {
                    session.set_learning_rate(0.01);
                }
                if replay == 1 {
                    session.set_adam(0.001, 0.9, 0.99, 1e-8);
                    session.set_grad_clip_norm(0.02);
                }
                if replay == 2 {
                    let mut encoder =
                        gpu.create_command_encoder(blade_graphics::CommandEncoderDesc {
                            name: "gradient ranges external records",
                            buffer_count: 1,
                            manual_barriers: false,
                        });
                    encoder.start();
                    session.record(&mut encoder).unwrap();
                    session.record(&mut encoder).unwrap();
                    let sync = gpu.submit(&mut encoder);
                    session.track_submission(sync);
                    session.wait();
                    gpu.destroy_command_encoder(&mut encoder);
                } else {
                    session.step();
                    session.wait();
                }
                check_worker_tasks(session);
            }
            assert_eq!(sessions[1].execution_report().compute_dispatches, 1);
            for i in 0..3 {
                for (name, len) in [
                    (format!("w{i}"), inner * columns),
                    (format!("b{i}"), columns),
                ] {
                    let mut a = vec![0.0; len];
                    let mut b = a.clone();
                    sessions[0].read_param(&name, &mut a);
                    sessions[1].read_param(&name, &mut b);
                    assert_eq!(a, b, "parameter {name}, workers={workers}, replay={replay}");
                    sessions[0].read_param_grad(&name, &mut a);
                    sessions[1].read_param_grad(&name, &mut b);
                    assert_eq!(a, b, "gradient {name}, workers={workers}, replay={replay}");
                }
            }
            assert_eq!(
                sessions[0].read_adam_states(&["w0", "w1", "w2", "b0", "b1", "b2"]),
                sessions[1].read_adam_states(&["w0", "w1", "w2", "b0", "b1", "b2"])
            );
        }
    }

    // Rebinding can make two bounded outputs overlap after the first frame.
    // Keep WAW order and both consumers' RAW order when that happens.
    let mut donor = meganeura::Graph::new();
    let a = donor.parameter("a", &[33, 33]);
    let b = donor.parameter("b", &[33, 33]);
    donor.set_outputs(vec![a, b]);
    let mut donor_plan = meganeura::compile::compile_with(&donor, &Default::default());
    donor_plan.param_buffers[1].1 = donor_plan.param_buffers[0].1;
    let mut donor_options = meganeura::SessionOptions::from_env();
    donor_options.persistent = None;
    let mut donor = meganeura::Session::with_context_opts(donor_plan, gpu.clone(), donor_options);
    let mut graph = meganeura::Graph::new();
    let a = graph.parameter("a", &[33, 33]);
    let b = graph.parameter("b", &[33, 33]);
    let x = graph.input("x", &[17, 33]);
    let y = graph.input("y", &[17, 33]);
    let ax = graph.matmul_at(x, x);
    let by = graph.matmul_at(y, y);
    let sa = graph.sum_inner(a);
    let sb = graph.sum_inner(b);
    graph.set_outputs(vec![ax, by, sa, sb]);
    let mut plan = meganeura::compile::compile_with(
        &graph,
        &meganeura::CompileOptions {
            fuse_dispatches: false,
            ..Default::default()
        },
    );
    for (output, name) in [(ax, "a"), (by, "b")] {
        let old = plan.node_buffers[output as usize].1;
        let new = plan.param_buffers.iter().find(|p| p.0 == name).unwrap().1;
        plan.dispatches
            .iter_mut()
            .find(|d| d.output_buffer == old)
            .unwrap()
            .output_buffer = new;
        for slot in &mut plan.output_buffers {
            if *slot == old {
                *slot = new;
            }
        }
        plan.node_buffers[output as usize].1 = new;
    }
    // The original independent reductions may have been scheduled first.
    // After redirecting the matrix outputs, make this fixture write then read.
    plan.dispatches
        .sort_by_key(|d| d.shader != meganeura::compile::ShaderEntry::MatMulAT);
    for workers in [1, 3, 64, 4096] {
        let mut options = meganeura::SessionOptions::from_env();
        options.coop = meganeura::CoopPolicy::Disabled;
        options.poison = true;
        options.persistent = Some(meganeura::PersistentOptions {
            workers,
            device_scheduling: true,
            whole_graph: true,
            allow_fallback: false,
            track_workers: true,
        });
        let mut session = meganeura::Session::with_context_opts(plan.clone(), gpu.clone(), options);
        session.set_input("x", &vec![0.25; 17 * 33]);
        session.set_input("y", &vec![-0.5; 17 * 33]);
        for replay in 0..3 {
            if replay == 1 {
                session.share_parameter_from(&mut donor, "a").unwrap();
                session.share_parameter_from(&mut donor, "b").unwrap();
            }
            session.step();
            session.wait();
            check_worker_tasks(&mut session);
            assert_eq!(session.execution_report().compute_dispatches, 1);
            let first = if replay == 0 { 1.0625 } else { 4.25 };
            for (index, count, value) in [
                (0, 33 * 33, first),
                (1, 33 * 33, 4.25),
                (2, 33, first * 33.0),
                (3, 33, 4.25 * 33.0),
            ] {
                let mut output = vec![f32::NAN; count];
                session.read_output_by_index(index, &mut output);
                assert!(
                    output.iter().all(|&v| v == value),
                    "rebind workers={workers}, replay={replay}, output={index}"
                );
            }
        }
    }
}

fn check_lifetime_reuse(whole_graph: bool) {
    let gpu = crate::support::gpu::gpu();
    if whole_graph && !gpu.capabilities().binding_array {
        return;
    }
    let mut graph = meganeura::Graph::new();
    let x = graph.input("x", &[8193]);
    let a = graph.neg(x);
    let e = graph.relu(x);
    let extra = graph.abs(x);
    let b = graph.neg(a);
    let f = graph.neg(e);
    let c = graph.neg(f);
    let h = graph.neg(c);
    graph.set_outputs(vec![e, extra, b, h]);
    let plan = meganeura::compile::compile_with(
        &graph,
        &meganeura::CompileOptions {
            fuse_dispatches: false,
            ..Default::default()
        },
    );
    let options = meganeura::SessionOptions {
        persistent: None,
        no_alias: true,
        poison: true,
        ..meganeura::SessionOptions::from_env()
    };
    let mut ordinary =
        meganeura::Session::with_context_opts(plan.clone(), gpu.clone(), options.clone());
    for device_scheduling in [false, true] {
        if whole_graph && !device_scheduling {
            continue;
        }
        if device_scheduling && !gpu.capabilities().device_memory_fence {
            continue;
        }
        for workers in [1, 3, 64, 4096] {
            let mut session = meganeura::Session::with_context_opts(
                plan.clone(),
                gpu.clone(),
                meganeura::SessionOptions {
                    no_alias: false,
                    persistent: Some(meganeura::PersistentOptions {
                        workers,
                        device_scheduling,
                        allow_fallback: false,
                        track_workers: true,
                        whole_graph,
                    }),
                    ..options.clone()
                },
            );
            assert!(
                session.memory_summary().allocated_buffer_bytes
                    < ordinary.memory_summary().allocated_buffer_bytes
            );
            assert!(
                session.memory_summary().num_allocations
                    < ordinary.memory_summary().num_allocations
            );
            if device_scheduling {
                // C reuses A's allocation while B reads A. There is no logical
                // edge B -> C; the native region must add that physical WAR edge.
                assert!(session.execution_report().regions.iter().any(|region| {
                    region.device_scheduled
                        && region.origin.contains(&b)
                        && region.origin.contains(&c)
                }));
            }
            session.set_submission_chunks(if whole_graph { 1 } else { 3 });
            for replay in 0..3 {
                let input: Vec<_> = (0..8193)
                    .map(|i| ((i * 17 + replay * 31) % 101) as f32 - 50.0)
                    .collect();
                ordinary.set_input("x", &input);
                ordinary.step();
                session.set_input("x", &input);
                if replay < 2 {
                    session.step();
                } else {
                    let mut encoder =
                        gpu.create_command_encoder(blade_graphics::CommandEncoderDesc {
                            name: "persistent allocation reuse replay",
                            buffer_count: 1,
                            manual_barriers: false,
                        });
                    encoder.start();
                    session.record(&mut encoder).unwrap();
                    session.record(&mut encoder).unwrap();
                    let sync = gpu.submit(&mut encoder);
                    session.track_submission(sync);
                    session.wait();
                    gpu.destroy_command_encoder(&mut encoder);
                }
                check_worker_tasks(&mut session);
                for index in 0..4 {
                    let mut expected = vec![f32::NAN; input.len()];
                    let mut actual = expected.clone();
                    ordinary.read_output_by_index(index, &mut expected);
                    session.read_output_by_index(index, &mut actual);
                    assert_eq!(actual, expected);
                }
            }
        }
    }
}

#[test]
fn session_persistent_owned_chains_keep_tile_dependencies_local() {
    let gpu = crate::support::gpu::gpu();
    let mut graph = meganeura::Graph::new();
    let mut node = graph.input("x", &[513]);
    let mut outputs = Vec::new();
    for _ in 0..8 {
        node = graph.neg(node);
        outputs.push(node);
    }
    graph.set_outputs(outputs);
    let plan = meganeura::compile::compile_with(
        &graph,
        &meganeura::CompileOptions {
            fuse_dispatches: false,
            ..Default::default()
        },
    );
    assert_eq!(plan.dispatches.len(), 8);
    for (workers, serial) in [(1, false), (3, false), (256, false), (3, true)] {
        let mut options = meganeura::SessionOptions::from_env();
        options.serial_dispatch = serial;
        options.persistent = Some(meganeura::PersistentOptions {
            workers,
            track_workers: workers == 3,
            device_scheduling: false,
            allow_fallback: false,
            whole_graph: false,
        });
        let mut session = meganeura::Session::with_context_opts(plan.clone(), gpu.clone(), options);
        let report = session.execution_report();
        assert_eq!(report.graph_dispatches, if serial { 8 } else { 3 });
        assert!(
            report
                .regions
                .iter()
                .all(|region| region.worker_owned_chain != serial)
        );
        for replay in 0..3 {
            let input: Vec<_> = (0..513).map(|i| (i % 17) as f32 - replay as f32).collect();
            session.set_input("x", &input);
            session.step();
            session.wait();
            check_worker_tasks(&mut session);
            for index in 0..8 {
                let mut output = vec![f32::NAN; input.len()];
                session.read_output_by_index(index, &mut output);
                let sign = if index % 2 == 0 { -1.0 } else { 1.0 };
                assert_eq!(
                    output,
                    input.iter().map(|value| sign * value).collect::<Vec<_>>()
                );
            }
        }
    }
    // Same grid and extent alone do not prove locality: every tile below
    // reads element zero from the preceding operation, owned by another tile.
    let mut broadcast = plan;
    if let meganeura::compile::Kernel::Pointwise(ref mut dag) = broadcast.dispatches[1].kernel {
        dag.ops[0] = meganeura::schedule::Pw::LoadBroadcast {
            input: 0,
            divisor: 1,
            modulus: 1,
        };
    } else {
        panic!("expected generated pointwise consumer");
    }
    let mut options = meganeura::SessionOptions::from_env();
    options.persistent = Some(meganeura::PersistentOptions {
        workers: 64,
        allow_fallback: false,
        ..Default::default()
    });
    let mut session = meganeura::Session::with_context_opts(broadcast, gpu, options);
    let report = session.execution_report();
    assert_eq!(report.regions[0].dispatches, 0..1);
    assert_eq!(report.regions[1].dispatches, 1..2);
    for value in [7.0, -3.0, 11.0] {
        let mut input = vec![-19.0; 513];
        input[0] = value;
        session.set_input("x", &input);
        session.step();
        session.wait();
        check_worker_tasks(&mut session);
        let mut output = vec![f32::NAN; 513];
        session.read_output_by_index(7, &mut output);
        assert_eq!(output, vec![value; 513]);
    }
}

#[test]
fn session_persistent_selected_kernels_preserve_geometry_and_scratch() {
    check_selected_kernels(false);
}

#[test]
fn session_whole_graph_selected_geometry_and_scratch() {
    check_selected_kernels(true);
}

#[test]
fn session_whole_graph_packed_matrix_tiles_preserve_scratch() {
    let gpu = crate::support::gpu::gpu();
    if !gpu.capabilities().device_memory_fence
        || !gpu.capabilities().binding_array
        || meganeura::runtime::auto_tune(&gpu, 0).coop_caps.f16_tile == 0
    {
        return;
    }
    for (rows, inner, cols) in [
        (53, 720, 2080),
        (53, 720, 4096),
        (53, 720, 4128),
        (53, 717, 4096),
        (53, 718, 4128),
        (33, 719, 2080),
    ] {
        let mut graph = meganeura::Graph::new();
        let input = graph.input("x", &[rows, inner]);
        let norm = graph.parameter("norm", &[inner]);
        let normalized = graph.rms_norm(input, norm, 1e-5);
        let weights = graph.parameter("weights", &[inner, cols]);
        let output = graph.matmul(normalized, weights);
        let mixed_columns = if (rows, inner, cols) == (33, 719, 2080) {
            // The same handler sees two- and four-tile claims in one graph.
            let norm = graph.parameter("norm_second", &[inner]);
            let normalized = graph.rms_norm(input, norm, 1e-5);
            let weights = graph.parameter("weights_second", &[inner, 4096]);
            let second = graph.matmul(normalized, weights);
            graph.set_outputs(vec![output, second]);
            Some(4096)
        } else {
            graph.set_outputs(vec![output]);
            None
        };
        let config = |persistent| {
            let mut config = crate::support::gpu::inference_config();
            config.tune = false;
            config.runtime.coop = meganeura::CoopPolicy::AllowF16;
            config.runtime.persistent = persistent;
            config.runtime.poison = true;
            config
        };
        let mut ordinary = meganeura::build(&graph, config(None)).0;
        assert!(
            ordinary
                .dispatch_implementation_keys()
                .iter()
                .any(|key| key.contains("cooperative-prologue"))
        );
        if mixed_columns.is_some() {
            let keys = ordinary.dispatch_implementation_keys();
            let prologues: Vec<_> = keys
                .iter()
                .filter(|key| key.contains("cooperative-prologue"))
                .collect();
            assert_eq!(prologues.len(), 2);
            assert_eq!(prologues[0], prologues[1]);
        }
        for workers in [1, 3, 64, 4096] {
            let options = meganeura::PersistentOptions {
                workers,
                device_scheduling: true,
                whole_graph: true,
                allow_fallback: false,
                track_workers: true,
            };
            let mut packed = meganeura::build(&graph, config(Some(options))).0;
            assert_eq!(
                ordinary.dispatch_implementation_keys(),
                packed.dispatch_implementation_keys()
            );
            for replay in 0..3 {
                let values = |len| {
                    (0..len)
                        .map(|i| (((i * 17 + replay * 23) % 61) as f32 - 30.0) / 64.0)
                        .collect::<Vec<_>>()
                };
                let input = values(rows * inner);
                let norm = values(inner);
                let weights = values(inner * cols);
                for session in [&mut ordinary, &mut packed] {
                    session.set_input("x", &input);
                    session.set_parameter("norm", &norm);
                    session.set_parameter("weights", &weights);
                    if let Some(columns) = mixed_columns {
                        session.set_parameter("norm_second", &values(inner));
                        session.set_parameter("weights_second", &values(inner * columns));
                    }
                    session.step();
                    session.wait();
                    check_worker_tasks(session);
                }
                assert_eq!(packed.execution_report().compute_dispatches, 1);
                assert_eq!(
                    ordinary.read_output(rows * cols),
                    packed.read_output(rows * cols)
                );
                if let Some(columns) = mixed_columns {
                    let mut expected = vec![0.0; rows * columns];
                    let mut actual = vec![0.0; rows * columns];
                    ordinary.read_output_by_index(1, &mut expected);
                    packed.read_output_by_index(1, &mut actual);
                    assert_eq!(expected, actual);
                }
            }
        }
    }
}

#[test]
fn session_whole_graph_packed_attention_heads_preserve_scratch() {
    use meganeura::graph::Op;
    use meganeura::kernels::attention_grad::Path;
    let gpu = crate::support::gpu::gpu();
    if !gpu.capabilities().device_memory_fence || !gpu.capabilities().binding_array {
        return;
    }
    let mut graph = meganeura::Graph::new();
    let mut inputs = Vec::new();
    let mut outputs = Vec::new();
    let mut lengths = Vec::new();
    // Both rowwise handlers are shared by nodes with different per-claim
    // widths. Include three active groups, the inactive fourth group, and
    // smaller head counts using the same pooled scratch banks.
    for (index, (heads, kv_heads, dim)) in [
        (1, 1, 63),
        (2, 1, 65),
        (3, 3, 129),
        (4, 2, 64),
        (5, 5, 193),
        (15, 3, 64),
        (12, 6, 64),
        (14, 7, 65),
    ]
    .into_iter()
    .enumerate()
    {
        let rows = 7 + index % 3;
        let is_cross = index % 3 == 1;
        let kv_rows = rows + if is_cross { 2 } else { 0 };
        let q_shape = [rows, (heads * dim) as usize];
        let kv_shape = [kv_rows, (kv_heads * dim) as usize];
        let mut nodes = Vec::new();
        for (name, shape) in [
            ("q", q_shape),
            ("k", kv_shape),
            ("v", kv_shape),
            ("do", q_shape),
        ] {
            let name = format!("{name}_{index}");
            nodes.push(graph.input(&name, &shape));
            inputs.push((name, shape.into_iter().product::<usize>()));
        }
        let [q, k, v, d_out] = nodes.try_into().unwrap();
        let fwd_node = match index % 3 {
            0 => graph.causal_attention(q, k, v, heads, kv_heads, dim),
            1 => graph.multi_head_attn(q, k, v, heads, kv_heads, dim, true),
            _ => graph.sliding_window_attention(q, k, v, heads, kv_heads, dim, 3),
        };
        outputs.push(fwd_node);
        lengths.push(q_shape.into_iter().product());
        for (op, original) in [
            (
                Op::MultiHeadAttnGradQ {
                    fwd_node,
                    num_heads: heads,
                    num_kv_heads: kv_heads,
                    head_dim: dim,
                    is_cross,
                },
                q,
            ),
            (
                Op::MultiHeadAttnGradK {
                    fwd_node,
                    num_heads: heads,
                    num_kv_heads: kv_heads,
                    head_dim: dim,
                    is_cross,
                },
                k,
            ),
            (
                Op::MultiHeadAttnGradV {
                    fwd_node,
                    num_heads: heads,
                    num_kv_heads: kv_heads,
                    head_dim: dim,
                    is_cross,
                },
                v,
            ),
        ] {
            let ty = graph.node(original).ty.clone();
            lengths.push(
                if original == q { q_shape } else { kv_shape }
                    .into_iter()
                    .product(),
            );
            outputs.push(graph.add_raw_node(op, vec![d_out, q, k, v], ty));
        }
    }
    graph.set_outputs(outputs);
    let plan = meganeura::compile::compile_with(
        &graph,
        &meganeura::CompileOptions {
            fuse_dispatches: false,
            prefer_attention_grad: Some(Path::Rowwise),
            ..Default::default()
        },
    );
    let options = meganeura::SessionOptions {
        persistent: None,
        coop: meganeura::CoopPolicy::Disabled,
        poison: true,
        ..meganeura::SessionOptions::from_env()
    };
    let mut ordinary =
        meganeura::Session::with_context_opts(plan.clone(), gpu.clone(), options.clone());
    for workers in [1, 3, 64, 4096] {
        let mut packed = meganeura::Session::with_context_opts(
            plan.clone(),
            gpu.clone(),
            meganeura::SessionOptions {
                persistent: Some(meganeura::PersistentOptions {
                    workers,
                    device_scheduling: true,
                    whole_graph: true,
                    allow_fallback: false,
                    track_workers: true,
                }),
                ..options.clone()
            },
        );
        assert_eq!(packed.execution_report().compute_dispatches, 1);
        for replay in 0..3 {
            for (name, len) in &inputs {
                let values = (0..*len)
                    .map(|i| ((i * 17 + replay * 23 + name.len()) % 61) as f32 / 32.0 - 1.0)
                    .collect::<Vec<_>>();
                ordinary.set_input(name, &values);
                packed.set_input(name, &values);
            }
            for session in [&mut ordinary, &mut packed] {
                session.step();
                session.wait();
            }
            check_worker_tasks(&mut packed);
            for (index, &len) in lengths.iter().enumerate() {
                let mut expected = vec![f32::NAN; len];
                let mut actual = expected.clone();
                ordinary.read_output_by_index(index, &mut expected);
                packed.read_output_by_index(index, &mut actual);
                assert_eq!(
                    actual, expected,
                    "workers={workers}, replay={replay}, output={index}"
                );
            }
        }
    }
}

fn check_selected_kernels(whole_graph: bool) {
    let gpu = crate::support::gpu::gpu();
    if whole_graph && !gpu.capabilities().binding_array {
        return;
    }
    for attention in [false, true] {
        let (x_shape, a_shape, b_shape, output_lengths) = if attention {
            ([33, 64], [33, 32], [33, 32], [33 * 64; 3])
        } else {
            ([37, 19], [19, 41], [41, 23], [37 * 41, 37 * 23, 23 * 37])
        };
        let mut graph = meganeura::Graph::new();
        let x = graph.input("x", &x_shape);
        let a = graph.parameter("a", &a_shape);
        let b = graph.parameter("b", &b_shape);
        let first = if attention {
            graph.causal_attention(x, a, b, 2, 1, 32)
        } else {
            graph.matmul(x, a)
        };
        let second = if attention {
            graph.full_attention(first, a, b, 2, 1, 32)
        } else {
            graph.matmul(first, b)
        };
        let transposed = graph.transpose(second);
        let output = graph.relu(transposed);
        graph.set_outputs(vec![first, second, output]);
        let plan = meganeura::compile::compile_with(
            &graph,
            &meganeura::CompileOptions {
                fuse_dispatches: false,
                flash_forward_coop: attention,
                ..Default::default()
            },
        );
        let ordinary_options = meganeura::SessionOptions {
            persistent: None,
            coop: if attention {
                meganeura::CoopPolicy::default()
            } else {
                meganeura::CoopPolicy::Disabled
            },
            poison: true,
            ..meganeura::SessionOptions::from_env()
        };
        let mut ordinary = meganeura::Session::with_context_opts(
            plan.clone(),
            gpu.clone(),
            ordinary_options.clone(),
        );
        for device_scheduling in [false, true] {
            if whole_graph && !device_scheduling {
                continue;
            }
            if device_scheduling && !gpu.capabilities().device_memory_fence {
                continue;
            }
            for workers in [1, 3, 64, 4096] {
                let mut session = meganeura::Session::with_context_opts(
                    plan.clone(),
                    gpu.clone(),
                    meganeura::SessionOptions {
                        persistent: Some(meganeura::PersistentOptions {
                            workers,
                            device_scheduling,
                            allow_fallback: false,
                            track_workers: true,
                            whole_graph,
                        }),
                        ..ordinary_options.clone()
                    },
                );
                let report = session.execution_report();
                assert!(
                    report
                        .regions
                        .iter()
                        .all(|region| region.workers == workers && region.fallback_reason.is_none())
                );
                assert!(report.regions.iter().any(|region| region.workgroup_shape
                    == Some(if whole_graph {
                        [256, 1, 1]
                    } else {
                        [16, 16, 1]
                    })));
                assert_eq!(
                    report.regions.iter().any(|region| region.device_scheduled),
                    device_scheduling
                );
                for replay in 0..3 {
                    let values = |len| {
                        (0..len)
                            .map(|i| (((i * 17 + replay * 23) % 61) as f32 - 30.0) / 32.0)
                            .collect::<Vec<_>>()
                    };
                    let input = values(x_shape.into_iter().product());
                    let a = values(a_shape.into_iter().product());
                    let b = values(b_shape.into_iter().product());
                    for current in [&mut ordinary, &mut session] {
                        current.set_input("x", &input);
                        current.set_parameter("a", &a);
                        current.set_parameter("b", &b);
                        current.step();
                        current.wait();
                        check_worker_tasks(current);
                    }
                    for (index, len) in output_lengths.into_iter().enumerate() {
                        let mut expected = vec![f32::NAN; len];
                        let mut actual = expected.clone();
                        ordinary.read_output_by_index(index, &mut expected);
                        session.read_output_by_index(index, &mut actual);
                        assert_eq!(actual, expected);
                    }
                }
            }
        }
    }
}

#[test]
fn session_persistent_fallback_is_explicit_and_strict_rejects_it() {
    let mut graph = meganeura::Graph::new();
    let x = graph.input("x", &[2, 3]);
    let w = graph.parameter("w", &[3, 5]);
    let product = graph.matmul(x, w);
    let out = graph.relu(product);
    graph.set_outputs(vec![out]);
    let config = |allow_fallback| {
        let mut config = crate::support::gpu::inference_config();
        config.optimize.mode = meganeura::OptimizeMode::Off;
        config.options.fuse_dispatches = false;
        config.runtime.persistent = Some(meganeura::PersistentOptions {
            allow_fallback,
            track_workers: true,
            device_scheduling: false,
            ..Default::default()
        });
        config
    };
    let mut session = meganeura::build(&graph, config(false)).0;
    let report = session.execution_report();
    assert!(
        report
            .regions
            .iter()
            .all(|r| r.workers != 0 && r.fallback_reason.is_none())
    );
    assert!(report.persistent_tasks > 0);
    session.set_input("x", &[1.0; 6]);
    session.set_parameter("w", &[2.0; 15]);
    session.step();
    session.wait();
    check_worker_tasks(&mut session);
    assert_eq!(session.read_output(10), vec![6.0; 10]);
    // Construction-only limit check: this grid cannot fit the persistent
    // counter. Do not submit this deliberately oversized ordinary fallback.
    let mut oversized = session.plan().clone();
    oversized.dispatches[0].workgroups = [u32::MAX, 1, 1];
    let fallback = meganeura::Session::with_context_opts(
        oversized.clone(),
        crate::support::gpu::gpu(),
        config(true).runtime,
    );
    assert!(
        fallback
            .execution_report()
            .regions
            .iter()
            .any(|region| region.workers == 0 && region.fallback_reason.is_some())
    );
    let failure = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        meganeura::Session::with_context_opts(
            oversized,
            crate::support::gpu::gpu(),
            config(false).runtime,
        )
    }));
    assert!(
        failure.is_err(),
        "strict execution silently accepted a queue-capacity fallback"
    );

    let mut training = meganeura::Graph::new();
    let parameter = training.parameter("p", &[1]);
    let loss = training.relu(parameter);
    training.set_outputs(vec![loss]);
    let mut training_config = config(false);
    training_config.mode = meganeura::Mode::Training;
    let mut session = meganeura::build(&training, training_config).0;
    session.set_parameter("p", &[1.0]);
    session.step();
    session.wait();
    check_worker_tasks(&mut session);
    session.set_learning_rate(0.1);
    assert!(session.execution_report().appended_fallback.is_empty());
    session.step();
    check_worker_tasks(&mut session);
    assert!(
        session
            .execution_report()
            .appended
            .iter()
            .all(|row| row.workers != 0 && row.fallback_reason.is_none())
    );
    let mut parameter = [0.0];
    session.read_param("p", &mut parameter);
    assert_eq!(parameter, [0.9]);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| session.sgd_step_cpu(0.1)))
            .is_err()
    );
    session.read_param("p", &mut parameter);
    assert_eq!(
        parameter,
        [0.9],
        "strict CPU rejection must precede the update"
    );
}

fn reference(job: &Job, weights: &[f32]) -> Vec<f64> {
    let mut data: Vec<f64> = job.data.iter().map(|&x| f64::from(x)).collect();
    for instruction in &job.instructions {
        let (dst, result): (u32, Vec<f64>) = match *instruction {
            Instruction::Affine {
                src,
                dst,
                len,
                scale,
                bias,
            } => (
                dst,
                data[src as usize..][..len as usize]
                    .iter()
                    .map(|&x| x * f64::from(scale) + f64::from(bias))
                    .collect(),
            ),
            Instruction::Add { a, b, dst, len } => (
                dst,
                (0..len as usize)
                    .map(|i| data[a as usize + i] + data[b as usize + i])
                    .collect(),
            ),
            Instruction::Relu { src, dst, len } => (
                dst,
                data[src as usize..][..len as usize]
                    .iter()
                    .map(|&x| x.max(0.0))
                    .collect(),
            ),
            Instruction::RmsNorm {
                src,
                dst,
                len,
                epsilon,
            } => {
                let input = &data[src as usize..][..len as usize];
                let rms = (input.iter().map(|x| x * x).sum::<f64>() / f64::from(len)
                    + f64::from(epsilon))
                .sqrt();
                (dst, input.iter().map(|x| x / rms).collect())
            }
            Instruction::MatVec {
                src,
                weights: offset,
                dst,
                rows,
                columns,
            } => (
                dst,
                weights[offset as usize..][..(rows * columns) as usize]
                    .chunks_exact(columns as usize)
                    .map(|row| {
                        row.iter()
                            .zip(&data[src as usize..])
                            .map(|(&a, &b)| f64::from(a) * b)
                            .sum()
                    })
                    .collect(),
            ),
        };
        data[dst as usize..][..result.len()].copy_from_slice(&result);
    }
    data
}

fn workload(count: usize) -> (Vec<Job>, Vec<f32>) {
    let mut weights = Vec::new();
    let mut jobs: Vec<_> = (0..count)
        .map(|index| {
            let n = [1, 31, 63, 64, 65, 129, 257][index % 7];
            let offset = weights.len() as u32;
            weights.extend((0..n * 3).map(|i| ((i * 17 + index as u32) % 23) as f32 / 23.0 - 0.5));
            let mut data = vec![f32::NAN; (4 * n + 3) as usize];
            for (i, x) in data[..n as usize].iter_mut().enumerate() {
                *x = ((i * 13 + index) % 37) as f32 / 19.0 - 1.0;
            }
            let instructions = vec![
                Instruction::Affine {
                    src: 0,
                    dst: n,
                    len: n,
                    scale: -0.75,
                    bias: 0.125,
                },
                Instruction::RmsNorm {
                    src: n,
                    dst: 2 * n,
                    len: n,
                    epsilon: 1e-5,
                },
                Instruction::Add {
                    a: 0,
                    b: 2 * n,
                    dst: 3 * n,
                    len: n,
                },
                Instruction::Relu {
                    src: 3 * n,
                    dst: 3 * n,
                    len: n,
                },
                Instruction::MatVec {
                    src: 3 * n,
                    weights: offset,
                    dst: 4 * n,
                    rows: 3,
                    columns: n,
                },
                Instruction::RmsNorm {
                    src: 4 * n,
                    dst: 4 * n,
                    len: 3,
                    epsilon: 1e-5,
                },
            ];
            let mut job = Job { data, instructions };
            // Different job lengths exercise dispatch and termination independently
            // of the queue index. Zero-op jobs preserve their initial data.
            job.instructions.truncate(index % 7);
            job
        })
        .collect();
    if count > 0 {
        jobs.push(Job::default());
    }
    (jobs, weights)
}

fn check(executor: &Executor, jobs: &[Job], expected: &[Vec<f64>], mode: Execution) {
    for (j, want) in expected.iter().enumerate() {
        for (i, (got, &want)) in executor.read_job(j).into_iter().zip(want).enumerate() {
            if want.is_nan() {
                assert!(got.is_nan(), "job {j} scratch {i} overwritten");
            } else {
                assert!(
                    got.is_finite() && (f64::from(got) - want).abs() <= 2e-5 * (1.0 + want.abs()),
                    "job {j}, value {i}: {got} != {want}"
                );
            }
        }
    }
    let stats = executor.statistics();
    assert_eq!(stats.completed_per_job, vec![1; jobs.len()]);
    if mode == Execution::Persistent {
        assert_eq!(
            stats
                .per_worker
                .iter()
                .map(|v| v[0] as usize)
                .sum::<usize>(),
            jobs.len()
        );
        assert_eq!(
            stats
                .per_worker
                .iter()
                .map(|v| v[1] as usize)
                .sum::<usize>(),
            jobs.iter().map(|j| j.instructions.len()).sum::<usize>()
        );
        if jobs.len() > stats.per_worker.len() {
            assert!(
                stats.per_worker.iter().any(|v| v[0] > 1),
                "workers must claim multiple jobs"
            );
        }
    }
}

#[test]
fn persistent_queue_matches_f64_and_separate_dispatches() {
    let gpu = crate::support::gpu::gpu();
    for (workers, count) in [(1, 137), (3, 137), (64, 137), (4096, 3), (7, 0)] {
        let (jobs, weights) = workload(count);
        let expected: Vec<_> = jobs.iter().map(|job| reference(job, &weights)).collect();
        let mut executor = Executor::new(
            gpu.clone(),
            &jobs,
            &weights,
            Options {
                workers,
                ..Options::default()
            },
        )
        .unwrap();
        for mode in [
            Execution::Persistent,
            Execution::Persistent,
            Execution::Separate,
            Execution::Persistent,
        ] {
            let measurement = executor.run(mode);
            if mode == Execution::Persistent {
                assert_eq!(measurement.dispatches, 1);
            }
            check(&executor, &jobs, &expected, mode);
        }
    }
}

#[test]
fn invalid_work_is_rejected_before_dispatch() {
    let gpu = crate::support::gpu::gpu();
    let instructions = [
        Instruction::Relu {
            src: u32::MAX,
            dst: 0,
            len: 2,
        },
        Instruction::Relu {
            src: 0,
            dst: 0,
            len: 0,
        },
        Instruction::Relu {
            src: 0,
            dst: 1,
            len: 3,
        },
        Instruction::Add {
            a: 0,
            b: 3,
            dst: 0,
            len: 2,
        },
        Instruction::RmsNorm {
            src: 0,
            dst: 0,
            len: 4,
            epsilon: f32::NAN,
        },
        Instruction::MatVec {
            src: 0,
            weights: 0,
            dst: 0,
            rows: 2,
            columns: 2,
        },
        Instruction::MatVec {
            src: 0,
            weights: 3,
            dst: 2,
            rows: 2,
            columns: 2,
        },
        Instruction::MatVec {
            src: 0,
            weights: 0,
            dst: 2,
            rows: u32::MAX,
            columns: 2,
        },
    ];
    for instruction in instructions {
        let job = Job {
            data: vec![1.0; 4],
            instructions: vec![instruction],
        };
        assert!(Executor::new(gpu.clone(), &[job], &[1.0; 4], Options::default()).is_err());
    }
    for workers in [0, 65_536] {
        assert!(
            Executor::new(
                gpu.clone(),
                &[],
                &[],
                Options {
                    workers,
                    ..Options::default()
                }
            )
            .is_err()
        );
    }
}
