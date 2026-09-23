//! Block products compile to one dispatch each, at both tile sizes, and
//! reject invalid shapes. Their values and derivatives are checked against
//! the f64 reference by the `oracle` suite.
use meganeura::Graph;

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
                let tile = if d.use_small_tiles() { 32 } else { 64 };
                assert_eq!(
                    d.workgroups,
                    [(n as u32).div_ceil(tile), (rows as u32).div_ceil(tile), 8]
                );
                assert!(!d.use_coop());
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
        for module in [
            generate_module(group, meganeura::codegen::MatmulKnobs::default()),
            generate_module_small(group, meganeura::codegen::MatmulKnobs::default()),
        ] {
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
