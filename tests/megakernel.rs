use meganeura::megakernel::{Execution, Executor, Instruction, Job, Options};

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
