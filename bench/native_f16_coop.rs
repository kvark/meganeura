//! Source-only F16-weight cooperative-matmul diagnostic; not a runtime selection.
use blade_graphics::{self as bg, ShaderData};
use meganeura::codegen::{self, MatMulOptions, MatMulTile, ShaderGroup, ShaderModule};
use std::time::Instant;

struct Inputs {
    shape: [u32; 3],
    label: String,
    a: Vec<f32>,
    b: Vec<f32>,
    src: Vec<f32>,
    model_add: Option<bool>,
    model_output: Vec<f32>,
}

fn model_inputs(gpu: std::sync::Arc<bg::Context>, path: &str) -> Vec<Inputs> {
    use meganeura::{Graph, SessionConfig, compile::ShaderEntry, load::gguf};
    let model = gguf::load_gguf(std::path::Path::new(path)).unwrap();
    let config = gguf::arch::ModelConfig::from_gguf(&model).unwrap();
    let mut graph = Graph::new();
    let built = gguf::graph::build(&mut graph, &model, &config, 128, 256).unwrap();
    graph.set_outputs(built.outputs());
    let mut cfg = SessionConfig::inference_from_env();
    cfg.gpu = Some(gpu);
    cfg.tune = false;
    cfg.runtime.no_alias = true;
    cfg.runtime.gpu_timing = true;
    cfg.runtime.coop = meganeura::CoopPolicy::NativeF32;
    let mut session = meganeura::build(&graph, cfg).0;
    gguf::weights::load(&mut session, &model, &config).unwrap();
    gguf::weights::reset_caches(&mut session, &built, &config);
    let tokens: Vec<_> = (0..128).map(|i| 42 + i % 31).collect();
    session.set_input_u32("token_ids", &tokens);
    session.set_input_u32("position", &[0]);
    session.set_input_u32("valid", &[128]);
    session.step();
    session.wait();
    let read = |buffer, count| {
        let mut values = vec![0.0f32; count];
        session.read_buffer(buffer, &mut values);
        values
    };
    let mut classes = std::collections::BTreeMap::<_, Vec<_>>::new();
    for (index, dispatch) in session.plan().dispatches.iter().enumerate() {
        if !matches!(
            dispatch.shader,
            ShaderEntry::MatMulBT | ShaderEntry::FusedMatMulBTAdd
        ) || dispatch.weight_format != meganeura::compile::WeightFormat::F16
            || dispatch.matmul_prologue.is_some()
            || dispatch.matmul_epilogue.is_some()
            || !dispatch.epilogue.is_empty()
            || dispatch.horizontal_batch >= 2
        {
            continue;
        }
        let add = dispatch.shader == ShaderEntry::FusedMatMulBTAdd;
        let shape = [dispatch.params[0], dispatch.params[1], dispatch.params[2]];
        classes
            .entry((shape, add))
            .or_default()
            .push((index, dispatch));
    }
    let mut inputs = Vec::new();
    for (([m, n, k], add), members) in classes {
        let mut selected = vec![0, members.len() / 2, members.len() - 1];
        selected.dedup();
        for member in selected {
            let (index, dispatch) = members[member];
            let raw = read(dispatch.input_buffers[1], (n * k / 2) as usize);
            let b = bytemuck::cast_slice::<f32, u16>(&raw)
                .iter()
                .map(|&bits| half::f16::from_bits(bits).to_f32())
                .collect();
            let label = format!("dispatch {index}: {}", dispatch.label);
            eprintln!("capture {label}, {m}x{n}x{k}, add={add}");
            inputs.push(Inputs {
                shape: [m, n, k],
                label,
                a: read(dispatch.input_buffers[0], (m * k) as usize),
                b,
                src: if add {
                    read(dispatch.input_buffers[2], (m * n) as usize)
                } else {
                    vec![0.0; (m * n) as usize]
                },
                model_add: Some(add),
                model_output: read(dispatch.output_buffer, (m * n) as usize),
            });
        }
    }
    assert!(!inputs.is_empty());
    inputs
}

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Params {
    m: u32,
    n: u32,
    k: u32,
    pad: u32,
}

#[derive(blade_macros::ShaderData, Clone, Copy)]
struct Data {
    matrix_a: bg::BufferPiece,
    matrix_b: bg::BufferPiece,
    matrix_c: bg::BufferPiece,
    src: bg::BufferPiece,
    params: Params,
}

fn cooperative(
    tile: [u32; 3],
    block: [u32; 3],
    width: u32,
    subgroups: u32,
    add: bool,
    scalar: bool,
    direct: bool,
) -> ShaderModule {
    let [tm, tn, tk] = tile;
    let [bm, bn, bk] = block;
    assert!(bm.is_multiple_of(tm) && bn.is_multiple_of(tn) && bk.is_multiple_of(tk));
    let stride = bn;
    let tile_count = (bm / tm) * (bn / tn);
    let mut accumulators = String::new();
    let mut multiplies = String::new();
    let mut stores = String::new();
    assert!(tile_count.is_multiple_of(subgroups));
    for index in 0..tile_count / subgroups {
        accumulators.push_str(&format!("var c{index} = coop_mat16x16<f32,C>();\n"));
        let coordinates = format!(
            "let tile = {index}u * {subgroups}u + subgroup;
                 let row = (tile / {}u) * {tm}u;
                 let col = (tile % {}u) * {tn}u;",
            bn / tn,
            bn / tn,
        );
        if direct && add {
            accumulators.push_str(&format!(
                "{{ {coordinates}
                    c{index} = coopLoadT<coop_mat16x16<f32,C>>(
                        &src[(base_m + row) * params.n + base_n + col], params.n);
                }}\n"
            ));
        }
        multiplies.push_str(&format!(
                "{{
                    {coordinates}
                    let a = coopLoadT<coop_mat16x16<f16,A>>(&shared_a[row * {bk}u + kk], {bk}u);
                    let b = coopLoadT<coop_mat16x16<f16,B>>(&shared_b[kk * {stride}u + col], {stride}u);
                    c{index} = coopMultiplyAdd(a, b, c{index});
                }}\n",
            ));
        stores.push_str(&format!(
            "{{
                    {coordinates}
                    coopStoreT(c{index}, &shared_c[row * {bn}u + col], {bn}u);
                }}\n",
        ));
    }
    if scalar {
        let threads = width * subgroups;
        let count = bm * bn / threads;
        accumulators = format!("var accum: array<f32, {count}>;");
        multiplies = format!(
            "for (var e = 0u; e < {count}u; e++) {{
                let output = e * {threads}u + lane;
                let row = output / {bn}u;
                let col = output % {bn}u;
                for (var inner = 0u; inner < {tk}u; inner++) {{
                    let a = row * {bk}u + kk + inner;
                    let b = (kk + inner) * {stride}u + col;
                    accum[e] += f32(shared_a[a]) * f32(shared_b[b]);
                }}
            }}"
        );
        stores = format!(
            "for (var e = 0u; e < {count}u; e++) {{ shared_c[e * {threads}u + lane] = accum[e]; }}"
        );
    }
    let mut source = include_str!("native_f16_coop.wgsl").to_string();
    if direct {
        assert!(!scalar);
        stores = stores.replace(
            &format!("&shared_c[row * {bn}u + col], {bn}u"),
            "&matrix_c[(base_m + row) * params.n + base_n + col], params.n",
        );
        source.truncate(source.find("$STORES").unwrap() + "$STORES".len());
        source.push_str("\n}");
    }
    for (name, value) in [
        ("A_SIZE", bm * bk),
        ("B_SIZE", bk * stride),
        ("C_SIZE", bm * bn),
        ("BM", bm),
        ("BN", bn),
        ("BK", bk),
        ("TK", tk),
        ("B_STRIDE", stride),
        ("THREADS", width * subgroups),
    ] {
        source = source.replace(&format!("${name}"), &format!("{value}u"));
    }
    source = source
        .replace("$ACCUMULATORS", &accumulators)
        .replace("$MULTIPLIES", &multiplies)
        .replace("$STORES", &stores)
        .replace(
            "$ADD_DECL",
            if add {
                "var<storage> src: array<f32>;"
            } else {
                ""
            },
        )
        .replace("$ADD", if add { "+ src[index]" } else { "" });
    let mut module = naga::front::wgsl::parse_str(&source).expect("cooperative source");
    // WGSL only spells square cooperative types. Naga IR and SPIR-V represent
    // rows and columns independently; preserve the type handles while setting
    // exactly the dimensions reported by the device.
    let size = |value| match value {
        8 => naga::CooperativeSize::Eight,
        16 => naga::CooperativeSize::Sixteen,
        _ => unreachable!(),
    };
    let dimensions = |role| match role {
        naga::CooperativeRole::A => (size(tm), size(tk)),
        naga::CooperativeRole::B => (size(tk), size(tn)),
        naga::CooperativeRole::C => (size(tm), size(tn)),
    };
    let types: Vec<_> = module
        .types
        .iter()
        .filter_map(|(handle, ty)| {
            let naga::TypeInner::CooperativeMatrix { scalar, role, .. } = ty.inner else {
                return None;
            };
            let (rows, columns) = dimensions(role);
            let mut replacement = ty.clone();
            replacement.inner = naga::TypeInner::CooperativeMatrix {
                rows,
                columns,
                scalar,
                role,
            };
            (replacement != *ty).then_some((handle, replacement))
        })
        .collect();
    for (handle, ty) in types {
        module.types.replace(handle, ty);
    }
    for function in module
        .functions
        .iter_mut()
        .map(|(_, f)| f)
        .chain(module.entry_points.iter_mut().map(|ep| &mut ep.function))
    {
        for (_, expression) in function.expressions.iter_mut() {
            if let naga::Expression::CooperativeLoad {
                rows,
                columns,
                role,
                ..
            } = expression
            {
                (*rows, *columns) = dimensions(*role);
            }
        }
    }
    ShaderModule {
        module,
        source,
        hint: "native_f16_coop",
    }
}

fn values(count: usize, seed: u32) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state.wrapping_mul(1664525).wrapping_add(1013904223);
            ((state >> 24) as i32 - 128) as f32 / 128.0
        })
        .collect()
}

#[test]
fn cooperative_store_emits_its_arguments() {
    let module = naga::front::wgsl::parse_str(
        "enable wgpu_cooperative_matrix;
        @group(0) @binding(0) var<storage, read_write> output: array<f32>;
        @group(0) @binding(1) var<uniform> size: vec4<u32>;
        @compute @workgroup_size(32)
        fn main() {
            var value = coop_mat16x16<f32,C>();
            coopStoreT(value, &output[size.x * size.y], size.x + 1u);
        }",
    )
    .unwrap();
    let info = naga::valid::Validator::new(
        naga::valid::ValidationFlags::all(),
        naga::valid::Capabilities::COOPERATIVE_MATRIX,
    )
    .validate(&module)
    .unwrap();
    naga::back::spv::write_vec(&module, &info, &Default::default(), None).unwrap();
}

fn main() {
    env_logger::init();
    let gpu = std::sync::Arc::new(unsafe {
        bg::Context::init(bg::ContextDesc {
            timing: true,
            validation: cfg!(debug_assertions),
            capture: std::env::var_os("MEGANEURA_GPU_CAPTURE").is_some(),
            ..Default::default()
        })
        .unwrap()
    });
    let caps = gpu.capabilities();
    assert!(
        caps.timing
            && !caps.cooperative_matrix.f16.is_empty()
            && caps.cooperative_matrix.subgroup_size > 0
    );
    eprintln!("cooperative capabilities: {:?}", caps.cooperative_matrix);
    let mut encoder = gpu.create_command_encoder(bg::CommandEncoderDesc {
        name: "native_f16_coop",
        buffer_count: 2,
        manual_barriers: false,
    });
    let mut records = Vec::new();
    let model_path = std::env::args().nth(1);
    let workloads = if let Some(ref path) = model_path {
        model_inputs(std::sync::Arc::clone(&gpu), path)
    } else {
        [
            [17, 35, 41],
            [128, 576, 576],
            [128, 3072, 576],
            [128, 576, 1536],
        ]
        .into_iter()
        .map(|[m, n, k]| Inputs {
            shape: [m, n, k],
            label: "synthetic".into(),
            a: values((m * k) as usize, 42),
            b: values((n * k) as usize, 73),
            src: values((m * n) as usize, 101),
            model_add: None,
            model_output: Vec::new(),
        })
        .collect()
    };
    for inputs in workloads {
        let Inputs {
            shape: [m, n, k],
            label,
            a,
            b,
            src,
            model_add,
            model_output,
        } = inputs;
        let params = Params { m, n, k, pad: 0 };
        let mut reference = vec![0.0f64; (m * n) as usize];
        let mut rounded_reference = reference.clone();
        let rounded_a: Vec<_> = a.iter().map(|&v| half::f16::from_f32(v).to_f32()).collect();
        for row in 0..m as usize {
            for col in 0..n as usize {
                let mut sum = 0.0f64;
                let mut rounded_sum = 0.0f64;
                for inner in 0..k as usize {
                    sum += f64::from(a[row * k as usize + inner])
                        * f64::from(b[col * k as usize + inner]);
                    rounded_sum += f64::from(rounded_a[row * k as usize + inner])
                        * f64::from(b[col * k as usize + inner]);
                }
                reference[row * n as usize + col] = sum;
                rounded_reference[row * n as usize + col] = rounded_sum;
            }
        }
        let packed: Vec<u16> = b
            .iter()
            .map(|&x| half::f16::from_f32(x).to_bits())
            .collect();
        let sentinel = f32::from_bits(0x7fc0_baad);
        let initial = vec![sentinel; (m * n) as usize + 256];
        let bytes: [&[u8]; 5] = [
            bytemuck::cast_slice(&a),
            bytemuck::cast_slice(&packed),
            bytemuck::cast_slice(&initial),
            bytemuck::cast_slice(&src),
            bytemuck::cast_slice(&b),
        ];
        let mut upload = Vec::new();
        let mut device = Vec::new();
        encoder.start();
        {
            let mut pass = encoder.transfer("upload");
            for data in bytes {
                let size = data.len().next_multiple_of(256) as u64;
                let staging = gpu.create_buffer(bg::BufferDesc {
                    name: "upload",
                    size,
                    memory: bg::Memory::Download,
                });
                let local = gpu.create_buffer(bg::BufferDesc {
                    name: "local",
                    size,
                    memory: bg::Memory::Device,
                });
                unsafe {
                    std::ptr::write_bytes(staging.data(), 0, size as usize);
                    std::ptr::copy_nonoverlapping(data.as_ptr(), staging.data(), data.len());
                }
                pass.copy_buffer_to_buffer(staging.at(0), local.at(0), size);
                upload.push(staging);
                device.push(local);
            }
        }
        let sync = gpu.submit(&mut encoder);
        assert!(gpu.wait_for(&sync, !0).unwrap());
        let data = Data {
            matrix_a: device[0].at(0),
            matrix_b: device[1].at(0),
            matrix_c: device[2].at(0),
            src: device[3].at(0),
            params,
        };
        for add in [false, true] {
            if model_add.is_some_and(|wanted| wanted != add) {
                continue;
            }
            assert!(model_output.iter().all(|v| v.is_finite()));
            let capture_mismatches = model_output
                .iter()
                .enumerate()
                .filter(|&(i, &actual)| {
                    let expected = reference[i] + if add { f64::from(src[i]) } else { 0.0 };
                    (f64::from(actual) - expected).abs() > 1e-5 + 2e-4 * expected.abs()
                })
                .count();
            let mut candidates = Vec::new();
            for (tile, width) in [(MatMulTile::Small, 32), (MatMulTile::Large, 64)] {
                let sm = codegen::generate_matmul_with_epilogue(
                    if add {
                        ShaderGroup::MatMulBTAdd
                    } else {
                        ShaderGroup::MatMulBT
                    },
                    codegen::EpilogueSource::Ops(&[]),
                    MatMulOptions {
                        format: meganeura::compile::WeightFormat::F16,
                        tile,
                        ..Default::default()
                    },
                );
                candidates.push((
                    format!("scalar-{width}"),
                    sm,
                    [n.div_ceil(width), m.div_ceil(width), 1],
                ));
            }
            if m.is_multiple_of(32)
                && n.is_multiple_of(32)
                && k.is_multiple_of(16)
                && caps.cooperative_matrix.f16.contains(&[16, 16, 16])
            {
                candidates.push((
                    "production-coop".into(),
                    codegen::generate_module_coop(
                        if add {
                            ShaderGroup::MatMulBTAdd
                        } else {
                            ShaderGroup::MatMulBT
                        },
                        &codegen::CoopConfig {
                            tile_size: 16,
                            use_f16_input: true,
                            compensated: false,
                        },
                    ),
                    [m / 32, n / 32, 1],
                ));
                let mut packed = codegen::generate_module_coop(
                    if add {
                        ShaderGroup::MatMulBTAdd
                    } else {
                        ShaderGroup::MatMulBT
                    },
                    &codegen::CoopConfig {
                        tile_size: 16,
                        use_f16_input: true,
                        compensated: false,
                    },
                );
                packed.source = packed
                    .source
                    .replace("matrix_b: array<vec4<f32>>", "matrix_b: array<vec4<f16>>");
                packed.module = naga::front::wgsl::parse_str(&packed.source).unwrap();
                candidates.push(("production-native-f16".into(), packed, [m / 32, n / 32, 1]));
            }
            let diagnostic_tile = caps.cooperative_matrix.f16[0];
            candidates.push((
                "scalar-staging".into(),
                cooperative(
                    diagnostic_tile,
                    [32, 32, 16],
                    caps.cooperative_matrix.subgroup_size,
                    2,
                    add,
                    true,
                    false,
                ),
                [m.div_ceil(32), n.div_ceil(32), 1],
            ));
            for &tile in &caps.cooperative_matrix.f16 {
                for (block, subgroups) in [
                    (tile, 1),
                    ([32, 32, 16], 1),
                    ([32, 32, 16], 2),
                    ([32, 32, 16], 4),
                    ([32, 32, 32], 4),
                    ([32, 64, 32], 2),
                    ([32, 64, 32], 4),
                ] {
                    let direct = m.is_multiple_of(block[0]) && n.is_multiple_of(block[1]);
                    candidates.push((
                        format!("coop-{tile:?}-{block:?}-sg{subgroups}-direct{direct}"),
                        cooperative(
                            tile,
                            block,
                            caps.cooperative_matrix.subgroup_size,
                            subgroups,
                            add,
                            false,
                            direct,
                        ),
                        [m.div_ceil(block[0]), n.div_ceil(block[1]), 1],
                    ));
                }
            }
            for (name, sm, groups) in candidates {
                let data = if name == "production-coop" {
                    Data {
                        matrix_b: device[4].at(0),
                        ..data
                    }
                } else {
                    data
                };
                eprintln!("{m}x{n}x{k}, add={add}, {name}");
                let started = Instant::now();
                let shader = gpu.create_shader(bg::ShaderDesc {
                    source: &sm.source,
                    naga_module: Some(sm.module),
                });
                let mut pipeline = gpu.create_compute_pipeline(bg::ComputePipelineDesc {
                    name: &name,
                    data_layouts: &[&Data::layout()],
                    compute: shader.at("main"),
                });
                let compile_ms = started.elapsed().as_secs_f64() * 1000.0;
                let stats: Vec<_> = gpu.get_pipeline_statistics(&pipeline).iter().map(|executable| {
                    serde_json::json!({ "name": executable.name, "statistics": executable.statistics.iter()
                        .map(|stat| serde_json::json!({"name":stat.name,"value":stat.value})).collect::<Vec<_>>() })
                }).collect();
                let mut gpu_us = Vec::new();
                let mut wall_us = Vec::new();
                let mut max_abs = 0.0f64;
                let mut failure = None;
                let mut full_precision_mismatches = 0;
                let mut relative_l2 = 0.0;
                let mut rounding_max_abs = 0.0f64;
                let half_arithmetic = name != "scalar-32" && name != "scalar-64";
                let arithmetic_reference = if half_arithmetic {
                    &rounded_reference
                } else {
                    &reference
                };
                let repeats = if model_path.is_some() { 1 } else { 32 };
                'samples: for sample in 0..if model_path.is_some() { 1 } else { 10 } {
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            bytes[2].as_ptr(),
                            upload[2].data(),
                            bytes[2].len(),
                        );
                    }
                    encoder.start();
                    encoder.transfer("reset").copy_buffer_to_buffer(
                        upload[2].at(0),
                        device[2].at(0),
                        bytes[2].len() as u64,
                    );
                    let sync = gpu.submit(&mut encoder);
                    assert!(gpu.wait_for(&sync, !0).unwrap());
                    let started = Instant::now();
                    encoder.start();
                    for _ in 0..repeats {
                        let mut pass = encoder.compute(&name);
                        let mut pc = pass.with(&pipeline);
                        pc.bind(0, &data);
                        pc.dispatch(groups);
                    }
                    encoder.transfer("readback").copy_buffer_to_buffer(
                        device[2].at(0),
                        upload[2].at(0),
                        bytes[2].len() as u64,
                    );
                    let sync = gpu.submit(&mut encoder);
                    assert!(gpu.wait_for(&sync, !0).unwrap());
                    let elapsed = started.elapsed().as_secs_f64() * 1e6 / f64::from(repeats);
                    let output = unsafe {
                        std::slice::from_raw_parts(upload[2].data().cast::<f32>(), initial.len())
                    }
                    .to_vec();
                    let mut difference = 0.0;
                    let mut norm = 0.0;
                    for (i, (&actual, &expected)) in
                        output.iter().zip(arithmetic_reference).enumerate()
                    {
                        let expected = expected + if add { f64::from(src[i]) } else { 0.0 };
                        let error = (f64::from(actual) - expected).abs();
                        let tolerance = if model_path.is_some() {
                            1e-5 + 2e-4 * expected.abs()
                        } else {
                            0.0001
                        };
                        if !actual.is_finite() || error > tolerance {
                            failure.get_or_insert_with(|| {
                                format!("at {i}: actual {actual}, expected {expected}")
                            });
                        }
                        max_abs = max_abs.max(error);
                        let full = reference[i] + if add { f64::from(src[i]) } else { 0.0 };
                        let delta = (f64::from(actual) - full).abs();
                        full_precision_mismatches += usize::from(delta > 1e-5 + 2e-4 * full.abs());
                        rounding_max_abs = rounding_max_abs.max(delta);
                        difference += (f64::from(actual) - full).powi(2);
                        norm += full.powi(2);
                    }
                    relative_l2 = if norm > 0.0 {
                        (difference / norm).sqrt()
                    } else {
                        0.0
                    };
                    if output[reference.len()..]
                        .iter()
                        .any(|x| x.to_bits() != sentinel.to_bits())
                    {
                        failure = Some("output guard overwritten".into());
                        break 'samples;
                    }
                    if failure.is_some() {
                        break 'samples;
                    }
                    if sample >= 3 {
                        wall_us.push(elapsed);
                        let timing = encoder.last_timing();
                        for pair in timing.passes.windows(2) {
                            gpu_us.push(pair[1].1.duration_since(pair[0].1).as_secs_f64() * 1e6);
                        }
                    }
                }
                if let Some(ref failure) = failure {
                    eprintln!("rejected: {failure}");
                    gpu_us.clear();
                    wall_us.clear();
                }
                records.push(serde_json::json!({ "shape": [m,n,k], "label": label, "add": add, "candidate": name,
                    "capture_f32_mismatches": capture_mismatches,
                    "f16_activation_arithmetic": half_arithmetic, "full_precision_mismatches": full_precision_mismatches,
                    "f32_reference_relative_l2": relative_l2, "f32_reference_max_abs": rounding_max_abs,
                    "compile_ms": compile_ms, "statistics": stats, "failure": failure, "max_abs": max_abs, "gpu_us": gpu_us, "wall_us": wall_us }));
                gpu.destroy_compute_pipeline(&mut pipeline);
            }
        }
        for buffer in device.into_iter().chain(upload) {
            gpu.destroy_buffer(buffer);
        }
    }
    println!(
        "{}",
        serde_json::json!({ "device": gpu.device_information().device_name,
        "model": model_path, "qualification_only": model_path.is_some(),
        "capture": std::env::var_os("MEGANEURA_GPU_CAPTURE").is_some(),
        "validation": cfg!(debug_assertions),
        "subgroup_size": caps.cooperative_matrix.subgroup_size,
        "f16_shapes": caps.cooperative_matrix.f16, "records": records })
    );
    gpu.destroy_command_encoder(&mut encoder);
}
