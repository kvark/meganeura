use meganeura::{
    Graph,
    codegen::MatrixTypes,
    compile::{Kernel, ShaderEntry},
    graph::DType,
};

fn parse_with_bindings(source: &str) -> naga::Module {
    let mut module = naga::front::wgsl::parse_str(source).unwrap();
    let mut binding = 0;
    for (_, var) in module.global_variables.iter_mut() {
        if matches!(
            var.space,
            naga::AddressSpace::Storage { .. } | naga::AddressSpace::Uniform
        ) {
            var.binding = Some(naga::ResourceBinding { group: 0, binding });
            binding += 1;
        }
    }
    module
}

#[test]
fn bf16_storage_roundtrip_and_f32_master_gradient() {
    let context = crate::support::gpu::gpu();
    if !context.capabilities().shader_bfloat16 {
        eprintln!("native BF16 storage unavailable; skipping device-specific coverage");
        return;
    }
    crate::support::gpu::assert_protocol_headroom(&context);
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
    crate::support::gpu::assert_protocol_headroom(&context);
    let memory = context.memory_stats();
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
fn unsupported_device_reports_bf16_requirement_before_shader_validation() {
    let context = crate::support::gpu::gpu();
    let caps = context.capabilities();
    if caps.shader_bfloat16
        && meganeura::codegen::checked_cooperative_shape(&caps.cooperative_matrix.bf16_f32_shapes)
            .is_some()
    {
        eprintln!(
            "native BF16 storage and matrices available; skipping unsupported-device coverage"
        );
        return;
    }
    let mut graph = Graph::new();
    let input = graph.input("x", &[3, 7]);
    let narrow = graph.to_bf16(input);
    let (output, expected) = if caps.shader_bfloat16 {
        let weight = graph.parameter("w", &[7, 5]);
        let weight = graph.to_bf16(weight);
        (
            graph.matmul(narrow, weight),
            "BF16 matrix products require native 16x16x16 or 8x16x16 BF16/F32 cooperative matrices",
        )
    } else {
        (
            graph.from_bf16(narrow),
            "selected plan requires native BF16 shader support",
        )
    };
    graph.set_outputs(vec![output]);
    let plan = meganeura::compile::compile(&graph);
    let panic = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        meganeura::Session::with_context(plan, context)
    })) {
        Ok(_) => panic!("unsupported device accepted a BF16 plan"),
        Err(panic) => panic,
    };
    let message = panic
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| panic.downcast_ref::<&str>().copied())
        .unwrap_or("non-string panic");
    assert!(message.contains(expected), "unexpected panic: {message}");
}

#[test]
fn bf16_matrix_shaders_validate() {
    for entry in [
        ShaderEntry::MatMul,
        ShaderEntry::MatMulAT,
        ShaderEntry::MatMulBT,
        ShaderEntry::FusedMatMulAdd,
        ShaderEntry::FusedMatMulATAdd,
        ShaderEntry::FusedMatMulBTAdd,
    ] {
        for output_bf16 in [false, true] {
            let types = MatrixTypes {
                output: if output_bf16 { DType::BF16 } else { DType::F32 },
                ..MatrixTypes::BF16
            };
            let shader =
                meganeura::codegen::generate_module_coop_typed(entry.shader_group(), types);
            let module = parse_with_bindings(&shader.source);
            let info = naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap();
            let options = naga::back::spv::Options {
                lang_version: (1, 3),
                capabilities: None,
                ..Default::default()
            };
            naga::back::spv::write_vec(&module, &info, &options, None).unwrap();
            let expected = meganeura::runtime::shader_data_layout(&entry);
            for (name, _) in expected.bindings {
                assert!(
                    module
                        .global_variables
                        .iter()
                        .any(|(_, var)| var.name.as_deref() == Some(name))
                );
            }
        }
    }
}

fn connected_graph() -> (Graph, meganeura::reference::Feeds) {
    let mut graph = Graph::new();
    let input = graph.input("input", &[19, 17]);
    let w1 = graph.parameter("w1", &[17, 23]);
    let w2 = graph.parameter("w2", &[23, 7]);
    let x = graph.to_bf16(input);
    let w1q = graph.to_bf16(w1);
    let w1q = graph.reshape(w1q, &[17, 23]);
    let hidden = graph.matmul(x, w1q);
    let hidden = graph.to_bf16(hidden);
    let hidden = graph.from_bf16(hidden);
    let hidden = graph.clamp(hidden, 0.0, 8.0);
    let hidden = graph.scale(hidden, 0.5);
    let hidden = graph.to_bf16(hidden);
    let w2q = graph.to_bf16(w2);
    let output = graph.matmul(hidden, w2q);
    let square = graph.mul(output, output);
    let loss = graph.mean_all(square);
    graph.set_outputs(vec![loss, output]);
    let mut feeds = meganeura::reference::Feeds::new();
    for (name, count, divisor) in [
        ("input", 19 * 17, 8.0),
        ("w1", 17 * 23, 64.0),
        ("w2", 23 * 7, 64.0),
    ] {
        feeds.set(
            name,
            &(0..count)
                .map(|i| ((i * 37 + 11) % 53) as f32 / divisor - 26.0 / divisor)
                .collect::<Vec<_>>(),
        );
    }
    (graph, feeds)
}

#[test]
fn bf16_connected_plan_cpu() {
    let (graph, feeds) = connected_graph();
    let full = meganeura::autodiff::differentiate_with_bf16_gradients(&graph);
    let reference = meganeura::reference::evaluate(&full, &feeds).unwrap();
    for &id in full.outputs() {
        assert_eq!(full.node(id).ty.dtype, DType::F32);
        assert!(reference[id as usize].data.iter().all(|v| v.is_finite()));
    }
    for node in full
        .nodes()
        .iter()
        .filter(|n| matches!(n.op, meganeura::graph::Op::Identity))
    {
        assert_eq!(node.ty.dtype, full.node(node.inputs[0]).ty.dtype);
    }
    let plan = meganeura::compile::compile(&full);
    let plain = meganeura::compile::compile_with(
        &full,
        &meganeura::CompileOptions {
            fuse_dispatches: false,
            ..Default::default()
        },
    );
    assert!(plan.dispatches.len() < plain.dispatches.len());
    assert!(plan.dispatches.iter().any(|d| matches!(
        d.kernel,
        Kernel::CooperativeTyped {
            types: MatrixTypes {
                output: DType::BF16,
                ..
            },
            ..
        }
    )));
    assert!(
        plan.dispatches
            .iter()
            .any(|d| d.pointwise_storage.bf16_inputs != 0 && d.pointwise_storage.bf16_output)
    );
    assert!(plan.param_types.values().all(|ty| ty.dtype == DType::F32));
    for &(param, grad) in &plan.param_grad_pairs {
        assert_eq!(
            plan.buffers[param.0 as usize],
            plan.buffers[grad.0 as usize]
        );
    }
    println!(
        "bf16_connected_plan dispatches={}->{}",
        plain.dispatches.len(),
        plan.dispatches.len()
    );

    // Visible F32 outputs must survive even when their internal consumers
    // fold conversions. The repeated input also exercises both typed slots.
    let mut visible = Graph::new();
    let input = visible.input("x", &[37]);
    let scaled = visible.scale(input, 0.75);
    let narrow = visible.to_bf16(scaled);
    let wide = visible.from_bf16(narrow);
    let square = visible.mul(wide, wide);
    let narrow_square = visible.to_bf16(square);
    let output = visible.from_bf16(narrow_square);
    visible.set_outputs(vec![scaled, wide, output]);
    let plan = meganeura::compile::compile(&visible);
    assert!(
        plan.dispatches
            .iter()
            .any(|d| d.pointwise_storage.bf16_inputs == 3 && d.pointwise_storage.bf16_output)
    );
    for &buffer in &plan.output_buffers {
        assert_eq!(plan.buffers[buffer.0 as usize], 37 * 4);
        assert!(plan.dispatches.iter().any(|d| d.output_buffer == buffer));
    }
}

#[test]
fn bf16_pointwise_shaders_validate() {
    use meganeura::schedule::{PointwiseDAG, PointwiseStorage, Pw};
    let dag = PointwiseDAG {
        n_inputs: 3,
        ops: vec![
            Pw::LoadInput(0),
            Pw::LoadInput(1),
            Pw::LoadBroadcast {
                input: 2,
                divisor: 1,
                modulus: 7,
            },
            Pw::Mul(0, 1),
            Pw::Add(3, 2),
        ],
        output: 4,
    };
    for inputs in 0..8 {
        for output in [false, true] {
            let shader = meganeura::schedule::lower_pointwise_storage(
                &dag,
                Default::default(),
                PointwiseStorage {
                    bf16_inputs: inputs,
                    bf16_output: output,
                },
            );
            let module = parse_with_bindings(&shader.source);
            let info = naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap();
            naga::back::spv::write_vec(
                &module,
                &info,
                &naga::back::spv::Options {
                    lang_version: (1, 3),
                    capabilities: None,
                    ..Default::default()
                },
                None,
            )
            .unwrap();
        }
    }
}

#[test]
fn bf16_connected_forward_backward_and_updates() {
    let context = crate::support::gpu::gpu();
    if !meganeura::codegen::checked_cooperative_shape(
        &context.capabilities().cooperative_matrix.bf16_f32_shapes,
    )
    .is_some()
    {
        eprintln!("native BF16 matrices unavailable; skipping device-specific coverage");
        return;
    }
    crate::support::gpu::assert_protocol_headroom(&context);
    let (graph, feeds) = connected_graph();
    for fuse in [false, true] {
        let mut options = meganeura::reference::gpu::Options::default();
        options.compile.fuse_dispatches = fuse;
        options.compile.bf16_matrix_gradients = true;
        meganeura::reference::gpu::check_training(&graph, &feeds, &options)
            .unwrap()
            .assert_passed("connected BF16 forward/backward");
    }
    let mut config = crate::support::gpu::config();
    config.tune = false;
    config.options.bf16_matrix_gradients = true;
    let (mut session, _) = meganeura::build(&graph, config);
    session.set_input("input", &feeds.f32("input").unwrap());
    for name in ["w1", "w2"] {
        session.set_parameter(name, &feeds.f32(name).unwrap());
    }
    let mut first_loss = 0.0;
    let mut final_loss = 0.0;
    for step in 0..32 {
        session.step();
        session.wait();
        let loss = session.read_loss();
        assert!(loss.is_finite() && loss >= 0.0);
        if step == 0 {
            first_loss = loss;
        }
        final_loss = loss;
        session.adam_step(1e-3, 0.9, 0.999, 1e-8);
    }
    session.wait();
    assert!(final_loss < first_loss);
    for (name, count) in [("w1", 17 * 23), ("w2", 23 * 7)] {
        let mut values = vec![0.0; count];
        session.read_param(name, &mut values);
        assert!(values.iter().all(|v| v.is_finite()));
        session.read_adam_m(name, &mut values);
        assert!(values.iter().all(|v| v.is_finite()));
        session.read_adam_v(name, &mut values);
        assert!(values.iter().all(|v| v.is_finite() && *v >= 0.0));
    }
    crate::support::gpu::assert_protocol_headroom(&context);
    let memory = context.memory_stats();
    println!(
        "bf16_connected_verified updates=32 losses={first_loss}->{final_loss} memory={memory:?}"
    );
}

#[test]
fn bf16_matrix_reduced_gradients() {
    let context = crate::support::gpu::gpu();
    if !meganeura::codegen::checked_cooperative_shape(
        &context.capabilities().cooperative_matrix.bf16_f32_shapes,
    )
    .is_some()
    {
        eprintln!("native BF16 matrices unavailable; skipping device-specific coverage");
        return;
    }
    let mut cases = 0;
    for layout in 0..3 {
        for addend in [false, true] {
            for scale in [1e-20, 1.0, 1e20] {
                let (m, n, k) = (35, 37, 41);
                let mut graph = Graph::new();
                let a = graph.parameter("a", &if layout == 1 { [k, m] } else { [m, k] });
                let b = graph.parameter("b", &if layout == 2 { [n, k] } else { [k, n] });
                let aq = graph.to_bf16(a);
                let bq = graph.to_bf16(b);
                let output = match layout {
                    0 => graph.matmul(aq, bq),
                    1 => graph.matmul_at(aq, bq),
                    _ => graph.matmul_bt(aq, bq),
                };
                let output = if addend {
                    let bias = graph.parameter("bias", &[m, n]);
                    graph.add(output, bias)
                } else {
                    output
                };
                let output = graph.to_bf16(output);
                let output = graph.from_bf16(output);
                let weights = graph.input("weights", &[m, n]);
                let weighted = graph.mul(output, weights);
                let loss = graph.sum_all(weighted);
                graph.set_outputs(vec![loss, output]);
                let mut feeds = meganeura::reference::Feeds::new();
                for (name, count) in [("a", m * k), ("b", k * n)] {
                    feeds.set(
                        name,
                        &(0..count)
                            .map(|i| ((i * 37 + 11) % 53) as f32 / 128.0 - 26.0 / 128.0)
                            .collect::<Vec<_>>(),
                    );
                }
                if addend {
                    feeds.set("bias", &vec![0.125; m * n]);
                }
                feeds.set(
                    "weights",
                    &(0..m * n)
                        .map(|i| ((i % 7) as f32 - 3.0) * scale * 1.0015)
                        .collect::<Vec<_>>(),
                );
                let mut options = meganeura::reference::gpu::Options::default();
                options.compile.bf16_matrix_gradients = true;
                let report =
                    meganeura::reference::gpu::check_training(&graph, &feeds, &options).unwrap();
                report.assert_passed("BF16 reduced derivative inputs");
                cases += 1;
            }
        }
    }
    println!("bf16_reduced_gradients_verified cases={cases}");
}

#[test]
fn bf16_selected_dense_graph_gradients() {
    let context = crate::support::gpu::gpu();
    if !meganeura::codegen::checked_cooperative_shape(
        &context.capabilities().cooperative_matrix.bf16_f32_shapes,
    )
    .is_some()
    {
        eprintln!("native BF16 matrices unavailable; skipping device-specific coverage");
        return;
    }
    let mut graph = Graph::new();
    let x = graph.input("x", &[19, 17]);
    let w1 = graph.parameter("w1", &[17, 23]);
    let w2 = graph.parameter("w2", &[23, 7]);
    let hidden = graph.matmul(x, w1);
    let hidden = graph.clamp(hidden, 0.0, 8.0);
    let hidden = graph.scale(hidden, 0.5);
    let hidden = graph.reshape(hidden, &[19, 23]);
    let output = graph.matmul(hidden, w2);
    let square = graph.mul(output, output);
    let loss = graph.mean_all(square);
    graph.set_outputs(vec![loss, output]);
    let graph = graph.into_bf16_matmuls();
    let mut feeds = meganeura::reference::Feeds::new();
    for (name, count, divisor) in [
        ("x", 19 * 17, 8.0),
        ("w1", 17 * 23, 64.0),
        ("w2", 23 * 7, 64.0),
    ] {
        feeds.set(
            name,
            &(0..count)
                .map(|i| ((i * 37 + 11) % 53) as f32 / divisor - 26.0 / divisor)
                .collect::<Vec<_>>(),
        );
    }
    for fuse in [false, true] {
        let mut options = meganeura::reference::gpu::Options::default();
        options.compile.fuse_dispatches = fuse;
        options.compile.bf16_matrix_gradients = true;
        meganeura::reference::gpu::check_training(&graph, &feeds, &options)
            .unwrap()
            .assert_passed("selected BF16 dense graph");
    }
    println!("bf16_selection_verified fused_unfused=2");
}

#[test]
fn bf16_matrix_values_and_f32_gradients() {
    let context = crate::support::gpu::gpu();
    if !meganeura::codegen::checked_cooperative_shape(
        &context.capabilities().cooperative_matrix.bf16_f32_shapes,
    )
    .is_some()
    {
        eprintln!(
            "native BF16 cooperative matrices unavailable; skipping device-specific coverage"
        );
        return;
    }
    crate::support::gpu::assert_protocol_headroom(&context);
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
                        .any(|d| matches!(d.kernel, Kernel::CooperativeTyped { .. }))
                );
                let report =
                    meganeura::reference::gpu::check_training(&graph, &feeds, &options).unwrap();
                report.assert_passed("BF16 matrix values and F32 gradients");
                comparisons += 1;
            }
        }
    }
    crate::support::gpu::assert_protocol_headroom(&context);
    let memory = context.memory_stats();
    println!("bf16_matrices_verified cases={comparisons} memory={memory:?}");
}
