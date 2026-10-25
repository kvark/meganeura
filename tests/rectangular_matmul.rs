//! Exercise rectangular candidates through ordinary session tuning, including
//! fused addends, rounded outputs, complete split-K reductions and device fallback.
use meganeura::{
    CoopPolicy, DType, Graph, MatmulTile, TuneDecision, TuneOptions, TuneStrategy,
    codegen::{MatrixTypes, WorkgroupMatmulShape},
};
use std::time::Duration;

fn graph(storage: DType, layout: usize, add: bool, round: bool, dims: [usize; 3]) -> Graph {
    let [m, n, k] = dims;
    let mut g = Graph::new();
    let ashape = if layout == 1 { [k, m] } else { [m, k] };
    let bshape = if layout == 2 { [n, k] } else { [k, n] };
    let (a, b) = if storage == DType::F16 {
        (g.parameter_f16("a", &ashape), g.parameter_f16("b", &bshape))
    } else {
        let a = g.parameter("a", &ashape);
        let b = g.parameter("b", &bshape);
        if storage == DType::BF16 {
            (g.to_bf16(a), g.to_bf16(b))
        } else {
            (a, b)
        }
    };
    let bias = add.then(|| g.parameter("bias", &[m, n]));
    let y = match layout {
        0 => g.matmul(a, b),
        1 => g.matmul_at(a, b),
        _ => g.matmul_bt(a, b),
    };
    if let Some(bias) = bias {
        let node = &mut g.nodes_mut()[y as usize];
        node.op = match layout {
            0 => meganeura::graph::Op::FusedMatMulAdd,
            1 => meganeura::graph::Op::FusedMatMulATAdd,
            _ => meganeura::graph::Op::FusedMatMulBTAdd,
        };
        node.inputs.push(bias);
    }
    let y = if round {
        let rounded = g.to_bf16(y);
        g.from_bf16(rounded)
    } else {
        y
    };
    g.set_outputs(vec![y]);
    g
}

#[test]
fn rectangular_candidates_preserve_live_outputs_and_precision() {
    check_candidates(TuneStrategy::Sequential);
}

#[test]
fn rectangular_races_preserve_live_outputs_and_precision() {
    check_candidates(TuneStrategy::Race);
}

fn check_candidates(strategy: TuneStrategy) {
    let gpu = crate::support::gpu::gpu();
    let mut probes = 0;
    for storage in [DType::F32, DType::F16, DType::BF16] {
        if storage != DType::F32 && !meganeura::runtime::typed_matrix_supported(&gpu, storage) {
            continue;
        }
        for layout in 0..3 {
            for (add, round, dims) in [
                (false, false, [256, 512, 256]),
                (true, false, [256, 512, 256]),
                (true, true, [128, 256, 128]),
                (false, false, [64, 128, 1024]),
            ] {
                if storage == DType::F32 && (round || dims[2] == 1024) {
                    continue;
                }
                let [m, n, k] = dims;
                eprintln!(
                    "rectangular case {storage:?} layout={layout} add={add} round={round} {dims:?}"
                );
                let graph = graph(storage, layout, add, round, dims);
                let mut config = crate::support::gpu::inference_config();
                config.tune = false;
                config.runtime.coop = if storage == DType::F32 {
                    CoopPolicy::AllowF16
                } else {
                    CoopPolicy::NativeF32
                };
                config.runtime.poison = true;
                let (mut session, _) = meganeura::build(&graph, config);
                // Positive exact layout fixtures keep this partition-equivalence
                // check separate from the signed native-accuracy tests.
                let values = |len, seed| {
                    (0..len)
                        .map(|i| (i * seed % 31 + 1) as f32 / 64.)
                        .collect::<Vec<_>>()
                };
                let a = values(m * k, 17);
                let b = values(n * k, 23);
                let bias = values(m * n, 11);
                session.set_parameter("a", &a);
                session.set_parameter("b", &b);
                if add {
                    session.set_parameter("bias", &bias);
                }
                session.step();
                session.wait();
                let before = session.read_output(m * n);
                // Check every output against an independent scalar dot. The
                // periodic exact operands let us sum one period and the tail.
                for row in 0..m {
                    for col in 0..n {
                        let product = |inner| {
                            let ai = if layout == 1 {
                                inner * m + row
                            } else {
                                row * k + inner
                            };
                            let bi = if layout == 2 {
                                col * k + inner
                            } else {
                                inner * n + col
                            };
                            f64::from(a[ai]) * f64::from(b[bi])
                        };
                        let period: f64 = (0..31).map(product).sum();
                        let tail: f64 = (0..k % 31).map(product).sum();
                        let expected = (period * (k / 31) as f64 + tail) as f32
                            + if add { bias[row * n + col] } else { 0. };
                        let expected = if round {
                            half::bf16::from_f32(expected).to_f32()
                        } else {
                            expected
                        };
                        assert_eq!(
                            before[row * n + col],
                            expected,
                            "{storage:?} layout={layout} add={add} round={round} {dims:?} row={row} col={col}"
                        );
                    }
                }
                let report = session
                    .tune_with(TuneOptions {
                        strategy,
                        max_time: Duration::from_secs(60),
                        max_classes: 1,
                        warmup_runs: 1,
                        sample_pairs: 4,
                        dispatches_per_sample: 1,
                        target_sample_time: None,
                        ..Default::default()
                    })
                    .unwrap();
                let workgroup: Vec<_> = report
                    .outcomes
                    .iter()
                    .filter(|o| matches!(o.candidate, MatmulTile::CooperativeWorkgroup { .. }))
                    .collect();
                // A native-F32 cooperative key does not imply F16 arithmetic,
                // and workgroup support is specific to types, shapes and limits.
                // CPU tests separately check classes filtered out before probing.
                let supported = report.outcomes.iter().any(|outcome| {
                    let class = &outcome.class;
                    let types = class
                        .matrix_types
                        .or_else(|| class.cooperative_f16.then_some(MatrixTypes::F32_AS_F16));
                    types.is_some_and(|types| {
                        WorkgroupMatmulShape::all().any(|shape| {
                            shape.supports(&gpu.capabilities(), types)
                                && shape.fits(class.m, class.n, class.k)
                        })
                    })
                });
                assert_eq!(
                    !workgroup.is_empty(),
                    supported,
                    "{storage:?} layout={layout} add={add} round={round}"
                );
                for outcome in &report.outcomes {
                    assert!(
                        if strategy == TuneStrategy::Sequential {
                            outcome.qualified
                        } else {
                            matches!(
                                outcome.decision,
                                TuneDecision::FasterCandidate
                                    | TuneDecision::KeepBaseline
                                    | TuneDecision::Eliminated
                            ) && (outcome.decision != TuneDecision::FasterCandidate
                                || outcome.qualified)
                        },
                        "{storage:?} layout={layout} add={add} round={round}: {outcome:?}"
                    );
                    assert_eq!(outcome.class.matrix_types.is_some(), storage != DType::F32);
                    if storage == DType::F32
                        && matches!(outcome.candidate, MatmulTile::CooperativeWorkgroup { .. })
                    {
                        assert!(outcome.class.cooperative_f16);
                    }
                }
                probes += workgroup.len();
                session.step();
                session.wait();
                assert_eq!(
                    session.read_output(m * n),
                    before,
                    "tuning changed live values or arithmetic"
                );
            }
        }
    }
    println!("rectangular_workgroup_candidates_reported={probes}");
}

#[test]
fn typed_split_add_keeps_bias_in_first_partition() {
    use meganeura::compile::{BufferRef, Dispatch, Kernel, ShaderEntry};
    let gpu = crate::support::gpu::gpu();
    for operand in [DType::F16, DType::BF16] {
        if !meganeura::runtime::typed_matrix_supported(&gpu, operand) {
            continue;
        }
        for storage in [DType::F32, operand] {
            for (dims, splits) in [([64, 64, 256], 8), ([33, 65, 1031], 4), ([16, 48, 65], 3)] {
                let [m, n, k] = dims;
                for layout in 0..3 {
                    let g = graph(storage, layout, true, false, dims);
                    let mut plan = meganeura::compile::compile(&g);
                    let index = plan
                        .dispatches
                        .iter()
                        .position(|d| d.shader.is_matmul())
                        .unwrap();
                    let output = plan.dispatches[index].output_buffer;
                    let partial = BufferRef(plan.buffers.len() as u32);
                    plan.buffers.push(m * n * splits * 4);
                    let types = MatrixTypes {
                        a: storage,
                        b: storage,
                        multiply: operand,
                        output: DType::F32,
                    };
                    let product = &mut plan.dispatches[index];
                    product.kernel = Kernel::CooperativeTyped {
                        types,
                        splits: splits as u32,
                    };
                    product.output_buffer = partial;
                    product.workgroups = [
                        (m as u32).div_ceil(32),
                        (n as u32).div_ceil(32),
                        splits as u32,
                    ];
                    let sum = Dispatch {
                        shader: ShaderEntry::SumRows,
                        input_buffers: vec![partial],
                        output_buffer: output,
                        params: vec![splits as u32, (m * n) as u32, 1, 0],
                        workgroups: [((m * n) as u32).div_ceil(256), 1, 1],
                        ..Default::default()
                    };
                    plan.dispatches.insert(index + 1, sum);
                    plan.groups.clear();
                    let mut session = meganeura::Session::with_context_opts(
                        plan,
                        gpu.clone(),
                        meganeura::SessionOptions {
                            poison: true,
                            no_alias: true,
                            ..Default::default()
                        },
                    );
                    let a: Vec<_> = (0..m * k)
                        .map(|i| ((i * 17 % 31) as f32 - 15.) / 64.)
                        .collect();
                    let b: Vec<_> = (0..n * k)
                        .map(|i| ((i * 23 % 29) as f32 - 14.) / 64.)
                        .collect();
                    let bias: Vec<_> = (0..m * n)
                        .map(|i| {
                            if i % 17 == 0 {
                                70_000.125
                            } else {
                                (i % 13) as f32 / 127. - 0.05
                            }
                        })
                        .collect();
                    for zero_product in [false, true] {
                        session.set_parameter(
                            "a",
                            &if zero_product {
                                vec![0.; a.len()]
                            } else {
                                a.clone()
                            },
                        );
                        session.set_parameter("b", &b);
                        session.set_parameter("bias", &bias);
                        session.step();
                        session.wait();
                        let mut partials = vec![0.; m * n * splits];
                        session.read_buffer(partial, &mut partials);
                        let actual = session.read_output(m * n);
                        for row in 0..m {
                            for col in 0..n {
                                let offset = row * n + col;
                                let (mut total, mut total_abs) = (0., 0.);
                                let blocks = k.div_ceil(32);
                                for split in 0..splits {
                                    let first = (split * (blocks / splits)
                                        + split.min(blocks % splits))
                                        * 32;
                                    let end = (first
                                        + (blocks / splits + usize::from(split < blocks % splits))
                                            * 32)
                                        .min(k);
                                    let mut expected = if split == 0 {
                                        f64::from(bias[offset])
                                    } else {
                                        0.
                                    };
                                    let mut sum_abs = expected.abs();
                                    if !zero_product {
                                        for inner in first..end {
                                            let ai = if layout == 1 {
                                                inner * m + row
                                            } else {
                                                row * k + inner
                                            };
                                            let bi = if layout == 2 {
                                                col * k + inner
                                            } else {
                                                inner * n + col
                                            };
                                            let term = f64::from(a[ai]) * f64::from(b[bi]);
                                            expected += term;
                                            sum_abs += term.abs();
                                        }
                                    }
                                    let got = f64::from(partials[split * m * n + offset]);
                                    let bound = meganeura::reference::native_matrix::error_bound(
                                        end - first + 1,
                                        sum_abs,
                                    )
                                    .unwrap();
                                    assert!(
                                        got.is_finite() && (got - expected).abs() <= bound,
                                        "{operand:?}/{storage:?} {dims:?} layout={layout} split={split}: {got} != {expected}"
                                    );
                                    total += expected;
                                    total_abs += sum_abs;
                                }
                                let got = f64::from(actual[offset]);
                                let bound = meganeura::reference::native_matrix::error_bound(
                                    k + splits,
                                    total_abs,
                                )
                                .unwrap();
                                assert!(got.is_finite() && (got - total).abs() <= bound);
                                if zero_product {
                                    assert_eq!(actual[offset], bias[offset]);
                                }
                            }
                        }
                    }
                    if operand == DType::F16
                        && storage == DType::F32
                        && dims == [33, 65, 1031]
                        && layout == 0
                    {
                        let report = session
                            .tune_with(TuneOptions {
                                max_time: Duration::from_secs(10),
                                max_classes: 1,
                                warmup_runs: 1,
                                sample_pairs: 4,
                                dispatches_per_sample: 1,
                                target_sample_time: None,
                                ..crate::support::sequential_tuning()
                            })
                            .unwrap();
                        assert_eq!(report.eligible_classes, 1);
                        assert!(!report.outcomes.is_empty());
                        assert!(report.outcomes.iter().all(|o| o.qualified), "{report:?}");
                        session.step();
                        session.wait();
                        assert_eq!(session.read_output(m * n), bias);
                    }
                }
            }
        }
    }
}
