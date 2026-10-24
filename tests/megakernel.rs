use meganeura::megakernel::{Execution, Executor, Instruction, Job, Options};

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
            allow_fallback: false,
        });
        let mut session = meganeura::build(&graph, config).0;
        let report = session.execution_report();
        assert!(report.persistent_tasks > 0);
        assert_eq!(report.graph_dispatches, session.plan().dispatches.len());
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
            } else if iteration == 1 {
                session.set_profiling(true);
                session.step();
                session.wait();
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
    for workers in [1, 64] {
        let mut options = meganeura::SessionOptions::from_env();
        options.persistent = Some(meganeura::PersistentOptions {
            workers,
            allow_fallback: false,
        });
        let mut session = meganeura::Session::with_context_opts(plan.clone(), gpu.clone(), options);
        let input: Vec<_> = (0..513).map(|i| (i % 17) as f32 - 8.0).collect();
        session.set_input("x", &input);
        session.step();
        session.wait();
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
            ..Default::default()
        });
        config
    };
    let mut session = meganeura::build(&graph, config(true)).0;
    let report = session.execution_report();
    assert!(
        report
            .regions
            .iter()
            .any(|r| r.workers == 0 && r.fallback_reason.is_some())
    );
    assert!(report.persistent_tasks > 0);
    session.set_input("x", &[1.0; 6]);
    session.set_parameter("w", &[2.0; 15]);
    session.step();
    session.wait();
    assert_eq!(session.read_output(10), vec![6.0; 10]);
    let failure = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        meganeura::build(&graph, config(false))
    }));
    assert!(
        failure.is_err(),
        "strict execution silently accepted a matrix fallback"
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
    session.set_learning_rate(0.1);
    assert_eq!(
        session.execution_report().appended_fallback,
        ["optimizer update"]
    );
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| session.step())).is_err());
    let mut parameter = [0.0];
    session.read_param("p", &mut parameter);
    assert_eq!(parameter, [1.0], "strict rejection must precede the update");
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
