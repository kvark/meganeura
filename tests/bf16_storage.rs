use meganeura::{Graph, compile::ShaderEntry, graph::DType};

#[test]
fn bf16_storage_roundtrip_and_f32_master_gradient() {
    let context = crate::support::gpu::gpu();
    if !context.capabilities().shader_bfloat16 {
        eprintln!("native BF16 storage unavailable; skipping device-specific coverage");
        return;
    }
    let memory = context.memory_stats();
    assert!(memory.budget.saturating_sub(memory.usage) >= 2 << 30);
    let mut values: Vec<_> = (0..1025)
        .map(|i| ((i * 37 % 1031) as f32 - 515.0) / 67.0)
        .collect();
    values.extend([
        0.0,
        -0.0,
        1.0 + 1.0 / 256.0,
        1.0 + 3.0 / 256.0,
        -1.0 - 1.0 / 256.0,
        -1.0 - 3.0 / 256.0,
        1e-30,
        -1e-30,
        1e30,
        -1e30,
    ]);
    let mut graph = Graph::new();
    let input = graph.input("x", &[values.len()]);
    let narrow = graph.to_bf16(input);
    let wide = graph.from_bf16(narrow);
    graph.set_outputs(vec![wide]);
    let mut config = crate::support::gpu::inference_config();
    config.tune = false;
    let (mut session, _) = meganeura::build(&graph, config);
    assert_eq!(
        session
            .plan()
            .dispatches
            .iter()
            .filter(|d| matches!(d.shader, ShaderEntry::ToBF16 | ShaderEntry::FromBF16))
            .count(),
        2
    );
    let narrow_buffer = session
        .plan()
        .node_buffers
        .iter()
        .find(|(id, _)| *id == narrow)
        .unwrap()
        .1;
    assert!(session.plan().buffers[narrow_buffer.0 as usize] < values.len() * 4);
    session.set_input("x", &values);
    session.step();
    session.wait();
    let mut actual = vec![0.0; values.len()];
    session.read_output_by_index(0, &mut actual);
    for (index, (&value, &got)) in values.iter().zip(&actual).enumerate() {
        let want = half::bf16::from_f32(value).to_f32();
        assert_eq!(got.to_bits(), want.to_bits(), "conversion {index}: {value}");
    }
    drop(session);

    let mut graph = Graph::new();
    let weight = graph.parameter("weight", &[17]);
    let scale = graph.input("scale", &[17]);
    let narrow = graph.to_bf16(weight);
    let wide = graph.from_bf16(narrow);
    let product = graph.mul(wide, scale);
    let loss = graph.sum_all(product);
    graph.set_outputs(vec![loss, wide]);
    let differentiated = meganeura::autodiff::differentiate(&graph);
    assert!(
        differentiated
            .nodes()
            .iter()
            .filter(|node| matches!(node.op, meganeura::graph::Op::Parameter { .. }))
            .all(|node| node.ty.dtype == DType::F32)
    );
    let mut feeds = meganeura::reference::Feeds::new();
    feeds.set("weight", &values[0..17]);
    feeds.set(
        "scale",
        &(0..17)
            .map(|i| (i as f32 - 8.0) * 0.125)
            .collect::<Vec<_>>(),
    );
    let report =
        meganeura::reference::gpu::check_training(&graph, &feeds, &Default::default()).unwrap();
    report.assert_passed("BF16 storage with F32 master gradients");
    let memory = context.memory_stats();
    assert!(memory.budget.saturating_sub(memory.usage) >= 2 << 30);
    println!(
        "bf16_storage_verified conversions={} f32_master_elements=17 memory={memory:?}",
        values.len()
    );
}

#[test]
#[should_panic(expected = "BF16 currently supports internal conversion storage only")]
fn bf16_storage_rejects_untyped_arithmetic() {
    let mut graph = Graph::new();
    let input = graph.input("x", &[7]);
    let narrow = graph.to_bf16(input);
    let output = graph.add(narrow, narrow);
    graph.set_outputs(vec![output]);
    meganeura::compile::compile(&graph);
}

#[test]
fn bf16_matrix_shaders_validate() {
    for (transpose_a, transpose_b) in [(false, false), (true, false), (false, true)] {
        for addend in [false, true] {
            let entry = ShaderEntry::MatMulBF16 {
                transpose_a,
                transpose_b,
                addend,
            };
            let mut shader =
                meganeura::codegen::generate_module(entry.shader_group(), Default::default());
            let mut binding = 0;
            for (_, var) in shader.module.global_variables.iter_mut() {
                if matches!(
                    var.space,
                    naga::AddressSpace::Storage { .. } | naga::AddressSpace::Uniform
                ) {
                    var.binding = Some(naga::ResourceBinding { group: 0, binding });
                    binding += 1;
                }
            }
            let info = naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&shader.module)
            .unwrap();
            let options = naga::back::spv::Options {
                lang_version: (1, 3),
                capabilities: None,
                ..Default::default()
            };
            naga::back::spv::write_vec(&shader.module, &info, &options, None).unwrap();
            let expected = meganeura::runtime::shader_data_layout(&entry);
            for (name, _) in expected.bindings {
                assert!(
                    shader
                        .module
                        .global_variables
                        .iter()
                        .any(|(_, var)| var.name.as_deref() == Some(name))
                );
            }
        }
    }
}

#[test]
fn bf16_matrix_values_and_f32_gradients() {
    let context = crate::support::gpu::gpu();
    if !context
        .capabilities()
        .cooperative_matrix
        .bf16_f32_shapes
        .contains(&[16, 16, 16])
    {
        eprintln!(
            "native BF16 cooperative matrices unavailable; skipping device-specific coverage"
        );
        return;
    }
    let memory = context.memory_stats();
    assert!(memory.budget.saturating_sub(memory.usage) >= 2 << 30);
    let mut comparisons = 0;
    for (m, n, k) in [(19, 23, 17), (32, 48, 64), (1, 7, 1)] {
        for layout in 0..3 {
            for addend in [false, true] {
                let mut graph = Graph::new();
                let a = graph.parameter("a", &if layout == 1 { [k, m] } else { [m, k] });
                let b = graph.parameter("b", &if layout == 2 { [n, k] } else { [k, n] });
                let aq = graph.to_bf16(a);
                let bq = graph.to_bf16(b);
                let product = match layout {
                    0 => graph.matmul(aq, bq),
                    1 => graph.matmul_at(aq, bq),
                    _ => graph.matmul_bt(aq, bq),
                };
                let output = if addend {
                    let bias = graph.parameter("bias", &[m, n]);
                    graph.add(product, bias)
                } else {
                    product
                };
                let loss = graph.sum_all(output);
                graph.set_outputs(vec![loss, output]);
                let mut feeds = meganeura::reference::Feeds::new();
                feeds.set(
                    "a",
                    &(0..m * k)
                        .map(|i| ((i * 37 + 11) % 53) as f32 / 17.0 - 1.5)
                        .collect::<Vec<_>>(),
                );
                feeds.set(
                    "b",
                    &(0..k * n)
                        .map(|i| ((i * 23 + 7) % 47) as f32 / 13.0 - 1.75)
                        .collect::<Vec<_>>(),
                );
                if addend {
                    feeds.set("bias", &vec![0.125; m * n]);
                }
                let options = meganeura::reference::gpu::Options::default();
                let plan =
                    meganeura::reference::gpu::plan(&graph, &options, meganeura::Mode::Training);
                assert!(
                    plan.dispatches
                        .iter()
                        .any(|d| matches!(d.shader, ShaderEntry::MatMulBF16 { .. }))
                );
                let report =
                    meganeura::reference::gpu::check_training(&graph, &feeds, &options).unwrap();
                report.assert_passed("BF16 matrix values and F32 gradients");
                comparisons += 1;
            }
        }
    }
    let memory = context.memory_stats();
    assert!(memory.budget.saturating_sub(memory.usage) >= 2 << 30);
    println!("bf16_matrices_verified cases={comparisons} memory={memory:?}");
}
