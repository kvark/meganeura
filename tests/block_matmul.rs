//! CPU graph/chain-rule checks; native checks require exclusive GPU ownership.
use meganeura::{Graph, graph::Op};
use std::collections::HashMap;

#[derive(Clone, Copy, Debug)]
enum Product {
    Forward,
    AT,
    BT,
}

fn product(g: &mut Graph, kind: Product, a: u32, b: u32, groups: usize) -> u32 {
    match kind {
        Product::Forward => g.block_matmul(a, b),
        Product::AT => g.block_matmul_at(a, b, groups),
        Product::BT => g.block_matmul_bt(a, b),
    }
}

fn shapes(kind: Product, groups: usize, m: usize, n: usize, k: usize) -> (Vec<usize>, Vec<usize>) {
    match kind {
        Product::Forward => (vec![m, groups * k], vec![groups, k, n]),
        Product::AT => (vec![k, groups * m], vec![k, groups * n]),
        Product::BT => (vec![m, groups * k], vec![groups, n, k]),
    }
}

fn reference(
    kind: Product,
    groups: usize,
    m: usize,
    n: usize,
    k: usize,
    a: &[f64],
    b: &[f64],
) -> Vec<f64> {
    let mut c = vec![0.0; groups * m * n];
    for group in 0..groups {
        for row in 0..m {
            for col in 0..n {
                let c_index = match kind {
                    Product::AT => (group * m + row) * n + col,
                    _ => (row * groups + group) * n + col,
                };
                for inner in 0..k {
                    let a_index = match kind {
                        Product::AT => (inner * groups + group) * m + row,
                        _ => (row * groups + group) * k + inner,
                    };
                    let b_index = match kind {
                        Product::Forward => (group * k + inner) * n + col,
                        Product::AT => (inner * groups + group) * n + col,
                        Product::BT => (group * n + col) * k + inner,
                    };
                    c[c_index] += a[a_index] * b[b_index];
                }
            }
        }
    }
    c
}

// Test-only scalar execution of the actual differentiated graph. Finite
// differences exercise the forward graph separately, including nonunit loss
// coefficients, both parameter paths and their accumulation.
fn evaluate(g: &Graph, leaves: &HashMap<String, Vec<f64>>) -> Vec<Vec<f64>> {
    let mut values: Vec<Vec<f64>> = Vec::new();
    for node in g.nodes() {
        let inputs: Vec<_> = node.inputs.iter().map(|&i| &values[i as usize]).collect();
        let result = match &node.op {
            Op::Input { name } | Op::Parameter { name } => leaves[name].clone(),
            Op::Constant { data } => data.iter().map(|&v| f64::from(v)).collect(),
            Op::Identity => inputs[0].clone(),
            Op::Add => inputs[0]
                .iter()
                .zip(inputs[1])
                .map(|(a, b)| a + b)
                .collect(),
            Op::Mul => inputs[0]
                .iter()
                .zip(inputs[1])
                .map(|(a, b)| a * b)
                .collect(),
            Op::Scale { factor } => inputs[0].iter().map(|v| v * f64::from(*factor)).collect(),
            Op::SumAll => vec![inputs[0].iter().sum()],
            Op::MatMul => {
                let a = &g.node(node.inputs[0]).ty.shape;
                let b = &g.node(node.inputs[1]).ty.shape;
                reference(Product::Forward, 1, a[0], b[1], a[1], inputs[0], inputs[1])
            }
            Op::BlockMatMul | Op::BlockMatMulAT { .. } | Op::BlockMatMulBT => {
                let a = &g.node(node.inputs[0]).ty.shape;
                let b = &g.node(node.inputs[1]).ty.shape;
                let (kind, groups, m, n, k) = match node.op {
                    Op::BlockMatMul => (Product::Forward, b[0], a[0], b[2], b[1]),
                    Op::BlockMatMulAT { groups } => {
                        (Product::AT, groups, a[1] / groups, b[1] / groups, a[0])
                    }
                    Op::BlockMatMulBT => (Product::BT, b[0], a[0], b[1], b[2]),
                    _ => unreachable!(),
                };
                reference(kind, groups, m, n, k, inputs[0], inputs[1])
            }
            other => panic!("unimplemented test operator: {other:?}"),
        };
        assert_eq!(result.len(), node.ty.num_elements(), "{:?}", node.op);
        values.push(result);
    }
    g.outputs()
        .iter()
        .map(|&i| values[i as usize].clone())
        .collect()
}

fn loss_graph(
    kind: Product,
    groups: usize,
    m: usize,
    n: usize,
    k: usize,
) -> (Graph, HashMap<String, Vec<f64>>) {
    let mut g = Graph::new();
    let (a_shape, b_shape) = shapes(kind, groups, m, n, k);
    let a = g.parameter("a", &a_shape);
    let b = g.parameter("b", &b_shape);
    let y = product(&mut g, kind, a, b, groups);
    let square = g.mul(y, y);
    let upstream = g.input("upstream", &g.node(y).ty.shape.clone());
    let weighted = g.mul(square, upstream);
    let sum = g.sum_all(weighted);
    let loss = g.scale(sum, -0.7);
    g.set_outputs(vec![loss]);
    let mut leaves = HashMap::new();
    for (name, len, multiplier) in [
        ("a", a_shape.iter().product(), 7),
        ("b", b_shape.iter().product(), 11),
        ("upstream", groups * m * n, 5),
    ] {
        leaves.insert(
            name.to_owned(),
            (0..len)
                .map(|i| ((i * multiplier % 23) as f64 - 11.0) * 0.03125)
                .collect(),
        );
    }
    (g, leaves)
}

#[test]
fn cpu_hand_calculated_independent_blocks() {
    assert_eq!(
        reference(
            Product::Forward,
            2,
            2,
            1,
            2,
            &[1., 2., 10., 20., 3., 4., 30., 40.],
            &[2., 3., 5., 7.]
        ),
        [8., 190., 18., 430.]
    );
    assert_eq!(
        reference(
            Product::AT,
            2,
            2,
            1,
            2,
            &[1., 2., 10., 20., 3., 4., 30., 40.],
            &[2., 5., 3., 7.]
        ),
        [11., 16., 260., 380.]
    );
}

#[test]
fn cpu_all_three_chain_rules_match_finite_differences() {
    for kind in [Product::Forward, Product::AT, Product::BT] {
        for groups in [1, 3] {
            let (g, mut leaves) = loss_graph(kind, groups, 2, 3, 4);
            let backward = meganeura::autodiff::differentiate(&g);
            let outputs = evaluate(&backward, &leaves);
            assert_eq!(outputs.len(), 3);
            for (index, name) in ["a", "b"].iter().enumerate() {
                for i in 0..leaves[*name].len() {
                    let value = leaves[*name][i];
                    let epsilon = 1e-5;
                    leaves.get_mut(*name).unwrap()[i] = value + epsilon;
                    let plus = evaluate(&g, &leaves)[0][0];
                    leaves.get_mut(*name).unwrap()[i] = value - epsilon;
                    let minus = evaluate(&g, &leaves)[0][0];
                    leaves.get_mut(*name).unwrap()[i] = value;
                    let numerical = (plus - minus) / (2.0 * epsilon);
                    assert!(
                        (outputs[index + 1][i] - numerical).abs() < 1e-8,
                        "{kind:?}/{groups}/{name}[{i}]: {} != {numerical}",
                        outputs[index + 1][i]
                    );
                }
            }
            let plan = meganeura::compile::compile(&backward);
            assert_eq!(plan.param_grad_pairs.len(), 2);
            let block_gradients: Vec<_> = plan
                .dispatches
                .iter()
                .filter(|d| {
                    matches!(
                        d.shader,
                        meganeura::compile::ShaderEntry::BlockMatMul
                            | meganeura::compile::ShaderEntry::BlockMatMulAT
                            | meganeura::compile::ShaderEntry::BlockMatMulBT
                    ) && d.requires_full_precision
                })
                .collect();
            assert_eq!(block_gradients.len(), 2);
            assert!(block_gradients.iter().all(|d| !d.use_coop));
        }
    }
}

#[test]
fn cpu_production_shapes_use_one_dispatch_per_product() {
    use meganeura::compile::ShaderEntry;
    for (rows, k, n) in [(1, 5, 7), (6, 1024, 256), (16, 256, 768), (1024, 256, 768)] {
        for kind in [Product::Forward, Product::AT, Product::BT] {
            let mut g = Graph::new();
            let (a_shape, b_shape) = shapes(kind, 8, rows, n, k);
            let a = g.input("a", &a_shape);
            let b = g.parameter("b", &b_shape);
            let out = product(&mut g, kind, a, b, 8);
            g.set_outputs(vec![out]);
            for graph in [&g, &meganeura::optimize::optimize(&g)] {
                let plan = meganeura::compile::compile(graph);
                assert_eq!(plan.dispatches.len(), 1);
                let d = &plan.dispatches[0];
                assert!(matches!(
                    d.shader,
                    ShaderEntry::BlockMatMul
                        | ShaderEntry::BlockMatMulAT
                        | ShaderEntry::BlockMatMulBT
                ));
                assert_eq!(d.params, [rows as u32, n as u32, k as u32, 8]);
                let tile = if d.use_small_tiles { 32 } else { 64 };
                assert_eq!(
                    d.workgroups,
                    [(n as u32).div_ceil(tile), (rows as u32).div_ceil(tile), 8]
                );
                assert!(!d.use_coop);
            }
        }
    }
}

#[test]
fn cpu_both_tile_sizes_generate_valid_f32_shaders() {
    use meganeura::codegen::{ShaderGroup, generate_module, generate_module_small};
    for group in [
        ShaderGroup::BlockMatMul,
        ShaderGroup::BlockMatMulAT,
        ShaderGroup::BlockMatMulBT,
        ShaderGroup::MatMul,
        ShaderGroup::MatMulAT,
        ShaderGroup::MatMulBT,
    ] {
        for module in [generate_module(group), generate_module_small(group)] {
            assert!(!module.source.contains('$'));
            assert!(!module.source.contains("enable f16"));
            let info = naga::valid::Validator::new(
                // Blade assigns bindings when it creates the pipeline.
                naga::valid::ValidationFlags::all() ^ naga::valid::ValidationFlags::BINDINGS,
                naga::valid::Capabilities::empty(),
            )
            .validate(&module.module)
            .unwrap();
            #[cfg(not(target_vendor = "apple"))]
            naga::back::spv::write_vec(
                &module.module,
                &info,
                &naga::back::spv::Options::default(),
                Some(&naga::back::spv::PipelineOptions {
                    shader_stage: naga::ShaderStage::Compute,
                    entry_point: "main".into(),
                }),
            )
            .unwrap();
        }
    }
}

#[test]
fn cpu_invalid_shapes_and_storage_are_rejected() {
    let cases = [
        (Product::Forward, vec![2, 6], vec![3, 2, 4], 3, true),
        (Product::Forward, vec![2, 7], vec![3, 2, 4], 3, false),
        (Product::BT, vec![2, 7], vec![3, 4, 2], 3, false),
        (Product::Forward, vec![2, 6], vec![6, 4], 3, false),
        (Product::Forward, vec![0, 6], vec![3, 2, 4], 3, false),
        (Product::AT, vec![2, 6], vec![2, 12], 0, false),
        (Product::AT, vec![2, 7], vec![2, 12], 3, false),
        (Product::AT, vec![2, 6], vec![3, 12], 3, false),
    ];
    for (kind, a_shape, b_shape, groups, f16) in cases {
        assert!(
            std::panic::catch_unwind(|| {
                let mut g = Graph::new();
                let a = g.input("a", &a_shape);
                let b = if f16 {
                    g.parameter_f16("b", &b_shape)
                } else {
                    g.parameter("b", &b_shape)
                };
                product(&mut g, kind, a, b, groups);
            })
            .is_err()
        );
    }
}

#[test]
#[ignore = "requires an exclusive GPU; CPU checks do not establish native parity"]
fn gpu_composed_losses_and_all_gradients_match_f64() {
    for kind in [Product::Forward, Product::AT, Product::BT] {
        for (groups, m, n, k) in [
            (1, 1, 3, 5),
            (3, 6, 7, 5),
            (8, 16, 17, 33),
            (2, 193, 193, 5), // Exercise the 64×64 output tile too.
        ] {
            let (g, leaves) = loss_graph(kind, groups, m, n, k);
            let expected = evaluate(&meganeura::autodiff::differentiate(&g), &leaves);
            let mut session = meganeura::build(&g, meganeura::SessionConfig::from_env()).0;
            for name in ["a", "b"] {
                session.set_parameter(
                    name,
                    &leaves[name].iter().map(|&v| v as f32).collect::<Vec<_>>(),
                );
            }
            session.set_input(
                "upstream",
                &leaves["upstream"]
                    .iter()
                    .map(|&v| v as f32)
                    .collect::<Vec<_>>(),
            );
            session.set_adam(0.0, 0.9, 0.999, 1e-8);
            session.step();
            session.wait();
            let mut actual = vec![vec![session.read_loss()]];
            for name in ["a", "b"] {
                let mut grad = vec![0.; leaves[name].len()];
                session.read_param_grad(name, &mut grad);
                actual.push(grad);
            }
            for (index, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                for (element, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
                    assert!(
                        actual.is_finite()
                            && (f64::from(actual) - expected).abs() <= 3e-6 + 3e-5 * expected.abs(),
                        "{kind:?}/{groups}/{m}/{n}/{k} output {index}[{element}]: {actual} != {expected}"
                    );
                }
            }
        }
    }
}
