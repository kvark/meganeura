//! Source-only F16-weight cooperative-matmul diagnostic; not a runtime selection.
use blade_graphics::{self as bg, ShaderData};
use meganeura::codegen::{self, MatMulOptions, MatMulTile, ShaderGroup, ShaderModule};
use std::time::Instant;
mod scalar_layout;

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

#[derive(blade_macros::ShaderData, Clone, Copy)]
struct Reduction {
    src: bg::BufferPiece,
    dst: bg::BufferPiece,
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
    transposed: bool,
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
                    let output_index = row * {bn}u + col;
                    let output_ptr = &shared_c[output_index];
                    let result = c{index};
                    coopStoreT(result, output_ptr, {bn}u);
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
    let mut source = include_str!("rectangular.wgsl").to_string();
    if !transposed {
        source = source
            .replace(
                "let row = base_n + 2u * (i / $BK);",
                "let row = base_n + 2u * (i % ($BN / 2u));",
            )
            .replace(
                "let col = base_k + i % $BK;",
                "let col = base_k + i / ($BN / 2u);",
            )
            .replace(
                "weight(row * params.k + col)",
                "weight(col * params.n + row)",
            )
            .replace(
                "weight((row + 1u) * params.k + col)",
                "weight(col * params.n + row + 1u)",
            )
            .replace(
                "let dst = (i % $BK) * $B_STRIDE + 2u * (i / $BK);",
                "let dst = 2u * i;",
            );
    }
    if direct {
        assert!(!scalar);
        stores = stores.replace(
            "&shared_c[output_index];",
            "&matrix_c[(base_m + row) * params.n + base_n + col];",
        );
        stores = stores.replace(&format!("output_ptr, {bn}u"), "output_ptr, n");
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

fn partitioned_production(add: bool, transposed: bool) -> ShaderModule {
    let mut module = codegen::generate_module_coop(
        if !transposed && add {
            ShaderGroup::MatMulAdd
        } else if !transposed {
            ShaderGroup::MatMul
        } else if add {
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
    module.source = module
        .source
        .replace("if sg == 0u {", "{")
        .replace(
            "var<workgroup> shared_b0: array<f16, 256>;",
            "var<workgroup> shared_b0: array<f16, 512>;",
        )
        .replace("var<workgroup> shared_b1: array<f16, 256>;", "")
        .replace("shared_b1[", "shared_b0[256u + ")
        .replace("&shared_b0[0]", "&shared_b0[sg * 256u]")
        .replace(
            "let c00 = tile_row * n",
            "let c00 = (tile_row + sg * 16u) * n",
        )
        .replace(
            "let c01 = tile_row * n",
            "let c01 = (tile_row + sg * 16u) * n",
        );
    module.source = module
        .source
        .lines()
        .filter(|line| {
            !line.contains("acc10")
                && !line.contains("acc11")
                && !line.contains("let a1 = coopLoadT")
        })
        .collect::<Vec<_>>()
        .join("\n");
    module.module = naga::front::wgsl::parse_str(&module.source).unwrap();
    module.hint = "partitioned_coop";
    module
}

fn one_subgroup_production(add: bool, transposed: bool) -> ShaderModule {
    let group = match (transposed, add) {
        (false, false) => ShaderGroup::MatMul,
        (false, true) => ShaderGroup::MatMulAdd,
        (true, false) => ShaderGroup::MatMulBT,
        (true, true) => ShaderGroup::MatMulBTAdd,
    };
    let mut module = codegen::generate_module_coop(group, &codegen::CoopConfig {
        tile_size: 16, use_f16_input: true, compensated: false,
    });
    module.source = module.source
        .replace("// Cooperative matrix multiply-add: C += A × B", "if sg == 0u {\n // Cooperative matrix multiply-add: C += A × B")
        .replace("workgroupBarrier();\n        t +=", "}\n        workgroupBarrier();\n        t +=");
    module.module = naga::front::wgsl::parse_str(&module.source).unwrap();
    module.hint = "one_subgroup_coop";
    module
}

#[test]
fn cooperative_store_with_hoisted_operands_emits_valid_spirv() {
    let module = naga::front::wgsl::parse_str(
        "enable wgpu_cooperative_matrix;
        @group(0) @binding(0) var<storage, read_write> output: array<f32>;
        @group(0) @binding(1) var<uniform> size: vec4<u32>;
        @compute @workgroup_size(32)
        fn main() {
            var value = coop_mat16x16<f32,C>();
            let result = value;
            let pointer = &output[size.x * size.y];
            let stride = size.x + 1u;
            coopStoreT(result, pointer, stride);
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
    let scalar_search = std::env::args().any(|a| a == "--scalar-nn");
    let split_search = std::env::args().any(|a| a == "--split-nn");
    let transposed = !scalar_search && !split_search && !std::env::args().any(|a| a == "--nn");
    let gpu = unsafe {
        bg::Context::init(bg::ContextDesc {
            timing: true,
            validation: cfg!(debug_assertions),
            capture: std::env::var_os("MEGANEURA_GPU_CAPTURE").is_some(),
            ..Default::default()
        })
        .unwrap()
    };
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
    let reduction_module = codegen::generate_module(ShaderGroup::SumRows, Default::default());
    let reduction_shader = gpu.create_shader(bg::ShaderDesc {
        source: &reduction_module.source,
        naga_module: Some(reduction_module.module),
    });
    let mut reduction_pipeline = gpu.create_compute_pipeline(bg::ComputePipelineDesc {
        name: "partial sums",
        data_layouts: &[&Reduction::layout()],
        compute: reduction_shader.at("sum_rows"),
    });
    for [m, n, k] in [
        [17, 35, 41],
        [50, 4096, 720],
        [50, 720, 2048],
        [50, 720, 960],
        [1500, 1536, 384],
        [128, 3072, 576],
    ] {
        let params = Params { m, n, k, pad: 0 };
        let a = values((m * k) as usize, 42);
        let b = values((n * k) as usize, 73);
        let src = values((m * n) as usize, 101);
        let mut reference = vec![0.0f32; (m * n) as usize];
        for row in 0..m as usize {
            for col in 0..n as usize {
                let mut sum = 0.0f64;
                for inner in 0..k as usize {
                    sum += f64::from(a[row * k as usize + inner])
                        * f64::from(
                            b[if !transposed {
                                inner * n as usize + col
                            } else {
                                col * k as usize + inner
                            }],
                        );
                }
                reference[row * n as usize + col] = sum as f32;
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
        if split_search {
            device.push(gpu.create_buffer(bg::BufferDesc {
                name: "split partials",
                size: u64::from(m) * u64::from(n) * 8 * 4,
                memory: bg::Memory::Device,
            }));
        }
        let data = Data {
            matrix_a: device[0].at(0),
            matrix_b: device[4].at(0),
            matrix_c: device[2].at(0),
            src: device[3].at(0),
            params,
        };
        for add in [false, true] {
            let mut candidates = Vec::new();
            for (tile, width) in [(MatMulTile::Small, 32), (MatMulTile::Large, 64)] {
                let sm = codegen::generate_matmul_with_epilogue(
                    if !transposed && add {
                        ShaderGroup::MatMulAdd
                    } else if !transposed {
                        ShaderGroup::MatMul
                    } else if add {
                        ShaderGroup::MatMulBTAdd
                    } else {
                        ShaderGroup::MatMulBT
                    },
                    None,
                    MatMulOptions {
                        format: meganeura::compile::WeightFormat::F32,
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
            if split_search {
                for (tile, width) in [(MatMulTile::Small, 32), (MatMulTile::Large, 64)] {
                    for stage in [8, 16, 32] {
                        for interleave in [false, true] {
                            for splits in [2, 4, 8] {
                                let mut sm = codegen::generate_matmul_with_epilogue(
                                    if add {
                                        ShaderGroup::MatMulAdd
                                    } else {
                                        ShaderGroup::MatMul
                                    },
                                    None,
                                    MatMulOptions {
                                        tile,
                                        knobs: codegen::MatmulKnobs {
                                            k_stage: stage,
                                            interleave_columns: interleave,
                                            integer_dot: false,
                                        },
                                        ..Default::default()
                                    },
                                );
                                sm.source = sm.source
                                    .replace("const SPLITS: u32 = 1u;", &format!("const SPLITS: u32 = {splits}u;"))
                                    .replace(" + src[idx]", " + select(0.0, src[idx - split_id * params.m * params.n], split_id == 0u)");
                                sm.module = naga::front::wgsl::parse_str(&sm.source).unwrap();
                                candidates.push((
                                    format!(
                                        "split{splits}-tile{width}-k{stage}-interleave{interleave}"
                                    ),
                                    sm,
                                    [n.div_ceil(width), m.div_ceil(width), splits],
                                ));
                            }
                        }
                    }
                }
            } else if scalar_search {
                for block in [[32, 32], [64, 64], [32, 64], [64, 32], [64, 128], [128, 64]] {
                    for threads in [[16, 16], [16, 8], [8, 16], [8, 8]] {
                        for stage in [8, 16, 32] {
                            for interleave in [false, true] {
                                if 4 * (block[0] * (stage + 1) + stage * (block[1] + 1))
                                    > caps.max_compute_shared_memory_size
                                {
                                    continue;
                                }
                                candidates.push((
                                    format!("scalar-{block:?}-wg{threads:?}-k{stage}-interleave{interleave}"),
                                    scalar_layout::generate(block, threads, stage, interleave, add),
                                    [n.div_ceil(block[1]), m.div_ceil(block[0]), 1],
                                ));
                            }
                        }
                    }
                }
            } else {
                if m.is_multiple_of(32)
                    && n.is_multiple_of(32)
                    && k.is_multiple_of(16)
                    && caps.cooperative_matrix.f16.contains(&[16, 16, 16])
                {
                    candidates.push((
                        "production-coop".into(),
                        codegen::generate_module_coop(
                            if !transposed && add {
                                ShaderGroup::MatMulAdd
                            } else if !transposed {
                                ShaderGroup::MatMul
                            } else if add {
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
                    if caps.cooperative_matrix.subgroup_size == 32 {
                        if std::env::args().any(|a| a == "--guard") {
                            candidates.push((
                                "one-subgroup-coop".into(),
                                one_subgroup_production(add, transposed),
                                [m / 32, n / 32, 1],
                            ));
                        }
                        candidates.push((
                            "partitioned-coop".into(),
                            partitioned_production(add, transposed),
                            [m / 32, n / 32, 1],
                        ));
                    }
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
                        transposed,
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
                                transposed,
                            ),
                            [m.div_ceil(block[0]), n.div_ceil(block[1]), 1],
                        ));
                    }
                }
            }
            for (name, sm, groups) in candidates {
                let split = name.starts_with("split");
                let data = if split {
                    Data {
                        matrix_c: device[5].at(0),
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
                let mut max_abs = 0.0f32;
                let mut failure = None;
                'samples: for sample in 0..10 {
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
                    for _ in 0..32 {
                        {
                            let mut pass = encoder.compute(&name);
                            let mut pc = pass.with(&pipeline);
                            pc.bind(0, &data);
                            pc.dispatch(groups);
                        }
                        if split {
                            let mut pass = encoder.compute("partial sums");
                            let mut pc = pass.with(&reduction_pipeline);
                            pc.bind(
                                0,
                                &Reduction {
                                    src: device[5].at(0),
                                    dst: device[2].at(0),
                                    params: Params {
                                        m: groups[2],
                                        n: m * n,
                                        k: 1,
                                        pad: 0,
                                    },
                                },
                            );
                            pc.dispatch([(m * n).div_ceil(256), 1, 1]);
                        }
                    }
                    encoder.transfer("readback").copy_buffer_to_buffer(
                        device[2].at(0),
                        upload[2].at(0),
                        bytes[2].len() as u64,
                    );
                    let sync = gpu.submit(&mut encoder);
                    assert!(gpu.wait_for(&sync, !0).unwrap());
                    let elapsed = started.elapsed().as_secs_f64() * 1e6 / 32.0;
                    let output = unsafe {
                        std::slice::from_raw_parts(upload[2].data().cast::<f32>(), initial.len())
                    }
                    .to_vec();
                    for (i, (&actual, &expected)) in output.iter().zip(&reference).enumerate() {
                        let expected = expected + if add { src[i] } else { 0.0 };
                        let error = (actual - expected).abs();
                        if !actual.is_finite() || error > 0.0001 {
                            failure = Some(format!("at {i}: actual {actual}, expected {expected}"));
                            break 'samples;
                        }
                        max_abs = max_abs.max(error);
                    }
                    if output[reference.len()..]
                        .iter()
                        .any(|x| x.to_bits() != sentinel.to_bits())
                    {
                        failure = Some("output guard overwritten".into());
                        break 'samples;
                    }
                    if sample >= 3 {
                        wall_us.push(elapsed);
                        let timing = encoder.last_timing();
                        if split {
                            for pair in timing.passes.windows(3).step_by(2).take(31) {
                                gpu_us
                                    .push(pair[2].1.duration_since(pair[0].1).as_secs_f64() * 1e6);
                            }
                        } else {
                            for pair in timing.passes.windows(2).take(31) {
                                gpu_us
                                    .push(pair[1].1.duration_since(pair[0].1).as_secs_f64() * 1e6);
                            }
                        }
                    }
                }
                if let Some(ref failure) = failure {
                    eprintln!("rejected: {failure}");
                    gpu_us.clear();
                    wall_us.clear();
                }
                records.push(serde_json::json!({ "shape": [m,n,k], "add": add, "candidate": name,
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
        "orientation": if transposed { "NT" } else { "NN" },
        "capture": std::env::var_os("MEGANEURA_GPU_CAPTURE").is_some(),
        "validation": cfg!(debug_assertions),
        "subgroup_size": caps.cooperative_matrix.subgroup_size,
        "f16_shapes": caps.cooperative_matrix.f16, "records": records })
    );
    gpu.destroy_compute_pipeline(&mut reduction_pipeline);
    gpu.destroy_command_encoder(&mut encoder);
}
