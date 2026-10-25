use super::{ConvolutionPass, MatrixTypes};
use super::{MatMulCoopVariant, ShaderGroup, ShaderModule, coop_shape, matmul_module, preprocess};
use crate::DType;

/// Four subgroup tiles sharing full-range f32 operands through workgroup memory.
/// Each workgroup writes 64 rows and `columns` columns. The caller must supply
/// full output tiles and a K dimension divisible by `k_stage`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct CooperativeMatmulShape {
    pub columns: u32,
    pub k_stage: u32,
    pub prefetch: bool,
}

impl CooperativeMatmulShape {
    pub const ROWS: u32 = 64;
    pub const THREADS: u32 = 256;

    pub fn legal(self) -> bool {
        matches!(self.columns, 64 | 128) && matches!(self.k_stage, 16 | 32)
    }

    /// Every legal shape, in the order tuning challenges them.
    pub fn all() -> impl Iterator<Item = Self> {
        [64, 128].into_iter().flat_map(|columns| {
            [16, 32].into_iter().flat_map(move |k_stage| {
                [false, true].map(move |prefetch| Self {
                    columns,
                    k_stage,
                    prefetch,
                })
            })
        })
    }

    pub fn fits_dimensions(self, m: u32, n: u32, k: u32) -> bool {
        self.legal()
            && m != 0
            && n != 0
            && k != 0
            && m.is_multiple_of(Self::ROWS)
            && n.is_multiple_of(self.columns)
            && k.is_multiple_of(self.k_stage)
    }

    pub fn shared_bytes(self) -> u32 {
        // Padding avoids repeatedly mapping adjacent matrix rows onto the
        // same LDS banks on CDNA. It does not change the logical matrices.
        4 * (Self::ROWS * (self.k_stage + 4) + self.k_stage * (self.columns + 16))
    }

    pub fn typed_shared_bytes(self, types: MatrixTypes, addend: bool) -> u32 {
        let bytes = MatrixTypes::bytes(types.multiply) as u32;
        bytes * (Self::ROWS * (self.k_stage + 16 / bytes) + self.k_stage * (self.columns + 16))
            + if types.output != DType::F32 || addend {
                Self::ROWS * self.columns * 4
            } else {
                0
            }
    }
}

/// Generate an aligned native 16x16 f32 matrix product, optionally partitioned
/// along K. Partitions write compact M*N slices; the caller must sum them in a
/// separate dispatch. An addend seeds partition zero only.
pub fn generate_tiled_coop_matmul(
    group: ShaderGroup,
    shape: CooperativeMatmulShape,
    splits: u32,
    prologue: Option<&crate::compile::MatMulPrologue>,
) -> ShaderModule {
    assert!(shape.legal() && (1..=65535).contains(&splits));
    let packed = matches!(group, ShaderGroup::Conv2dPacked { .. });
    let weight_gradient = matches!(
        group,
        ShaderGroup::Conv2dPacked {
            pass: ConvolutionPass::WeightGradient,
            ..
        }
    );
    let input_gradient = matches!(
        group,
        ShaderGroup::Conv2dPacked {
            pass: ConvolutionPass::InputGradient,
            ..
        }
    );
    let grouped = matches!(group, ShaderGroup::BlockMatMulTyped { .. });
    let typed = packed || grouped || matches!(group, ShaderGroup::MatMulTyped { .. });
    let (addend, variant, types) = match group {
        ShaderGroup::Conv2dPacked { types, .. } => {
            assert!(prologue.is_none() && (weight_gradient || splits == 1));
            assert_eq!(
                shape,
                CooperativeMatmulShape {
                    columns: 64,
                    k_stage: 32,
                    prefetch: false
                }
            );
            (
                false,
                if weight_gradient {
                    MatMulCoopVariant::AT
                } else {
                    MatMulCoopVariant::BT
                },
                types,
            )
        }
        ShaderGroup::BlockMatMulTyped {
            transpose_a,
            transpose_b,
            types,
        } => {
            assert!(!(transpose_a && transpose_b) && prologue.is_none());
            (
                false,
                if transpose_a {
                    MatMulCoopVariant::AT
                } else if transpose_b {
                    MatMulCoopVariant::BT
                } else {
                    MatMulCoopVariant::Normal
                },
                types,
            )
        }
        ShaderGroup::MatMulTyped {
            transpose_a,
            transpose_b,
            addend,
            types,
        } => {
            assert!(!(transpose_a && transpose_b));
            assert!(prologue.is_none());
            (
                addend,
                if transpose_a {
                    MatMulCoopVariant::AT
                } else if transpose_b {
                    MatMulCoopVariant::BT
                } else {
                    MatMulCoopVariant::Normal
                },
                types,
            )
        }
        _ => {
            let (addend, variant) = coop_shape(group).expect("tiled cooperative matrix group");
            (
                addend,
                variant,
                MatrixTypes {
                    a: DType::F32,
                    b: DType::F32,
                    multiply: DType::F32,
                    output: DType::F32,
                },
            )
        }
    };
    types.validate();
    let multiply_type = MatrixTypes::name(types.multiply);
    let scratch = packed || types.output != DType::F32 || (typed && addend);
    assert!(
        !addend || prologue.is_none(),
        "prologue layout has no addend binding"
    );
    let rows = CooperativeMatmulShape::ROWS;
    let lanes = if typed {
        128
    } else {
        CooperativeMatmulShape::THREADS
    };
    let cols = shape.columns;
    let stage = shape.k_stage;
    let stride_a = (if weight_gradient { rows } else { stage })
        + 16 / MatrixTypes::bytes(types.multiply) as u32;
    let stride_b = if weight_gradient {
        cols + 16 / MatrixTypes::bytes(types.multiply) as u32
    } else if packed {
        stride_a
    } else {
        cols + 16
    };
    let wave_cols = cols / 2;
    let (decl, cache_decl, cache_init, transform) = prologue
        .map(|p| super::matmul_prologue_to_wgsl(p, rows))
        .unwrap_or_default();
    let mut init = String::new();
    let mut add_loads = String::new();
    let mut stores = String::new();
    for i in 0..2 {
        for j in 0..wave_cols / 16 {
            init += &format!("var acc{i}_{j} = coop_mat16x16<f32,C>();\n");
            let index = format!(
                "(tile_row + wr * 32u + {}u) * output_stride + tile_col + wc * {wave_cols}u + {}u",
                i * 16,
                j * 16
            );
            add_loads += &format!(
                "let ci{i}_{j} = {index};\nacc{i}_{j} = coopLoadT<coop_mat16x16<f32,C>>(&src[ci{i}_{j}], n);\n"
            );
            if scratch {
                stores += &format!(
                    "let oi{i}_{j} = (wr * 32u + {}u) * {cols}u + wc * {wave_cols}u + {}u;\ncoopStoreT(acc{i}_{j}, &results[oi{i}_{j}], {cols}u);\n",
                    i * 16,
                    j * 16
                );
            } else {
                stores += &format!(
                    "let oi{i}_{j} = output_base + {index};\ncoopStoreT(acc{i}_{j}, &$C_BUFFER[oi{i}_{j}], output_stride);\n"
                );
            }
        }
    }
    if addend && !typed {
        init += &format!("if wgid.z == 0u {{\n{add_loads}}}\n");
    }
    let (partial_init, partial_step, partial_finish) = if weight_gradient {
        let names = (0..2)
            .flat_map(|i| (0..wave_cols / 16).map(move |j| format!("acc{i}_{j}")))
            .collect::<Vec<_>>();
        super::qualified_cooperative::compensated_partials(&names, "begin", &format!("(t + {stage}u)"), "end")
    } else {
        Default::default()
    };
    init += &partial_init;
    let mut variables = String::new();
    let mut loads = String::new();
    let mut writes = String::new();
    for a in [true, false] {
        // Vector loads always follow the contiguous axis of the global
        // operand. AT and BT scatter components into the shared tile.
        let transposed = if a {
            variant == MatMulCoopVariant::AT
        } else {
            variant == MatMulCoopVariant::BT
        };
        let (prefix, buffer, shared, count, vstride) = if a {
            (
                "a",
                "matrix_a",
                "sa",
                rows * stage / 4,
                if transposed { rows / 4 } else { stage / 4 },
            )
        } else {
            (
                "b",
                "$B_BUFFER",
                "sb",
                cols * stage / 4,
                if transposed { stage / 4 } else { cols / 4 },
            )
        };
        assert!(count.is_multiple_of(lanes));
        for part in 0..count / lanes {
            variables += &format!(
                "let {prefix}r{part} = (lid.x + {}u) / {vstride}u;\nlet {prefix}c{part} = ((lid.x + {}u) % {vstride}u) * 4u;\nvar {prefix}v{part} = vec4<{multiply_type}>(0.0);\n",
                part * lanes,
                part * lanes
            );
            if weight_gradient && !a {
                variables += &format!(
                    "let bc_full{part} = tile_col + bc{part};\nlet btap{part} = bc_full{part} / channels;\nlet bchannel{part} = bc_full{part} % channels;\nlet bh_offset{part} = i32(btap{part} / KW) - i32(params.padding_h);\nlet bw_offset{part} = i32(btap{part} % KW) - i32(params.padding_w);\n"
                );
            } else if packed && !a {
                let coordinate = if input_gradient {
                    format!(
                        "let ih0_{part} = i32(bh{part} + params.padding_h);\nlet iw0_{part} = i32(bw{part} + params.padding_w);\n"
                    )
                } else {
                    format!(
                        "let ih0_{part} = i32(bh{part} * STRIDE) - i32(params.padding_h);\nlet iw0_{part} = i32(bw{part} * STRIDE) - i32(params.padding_w);\n"
                    )
                };
                let width = if input_gradient { "in_w" } else { "out_w" };
                variables += &format!(
                    "let bp{part} = tile_col + br{part};\nlet bh{part} = divide_exact(bp{part}, params.{width}, params.column_width_multiplier);\nlet bw{part} = bp{part} - bh{part} * params.{width};\n{coordinate}"
                );
            }
            let index = if grouped {
                match (a, transposed) {
                    (true, false) => format!(
                        "(((tile_row + ar{part}) * params.groups + block_id) * k + t + ac{part}) / 4u"
                    ),
                    (true, true) => format!(
                        "(((t + ar{part}) * params.groups + block_id) * m + tile_row + ac{part}) / 4u"
                    ),
                    (false, true) => {
                        format!("((block_id * n + tile_col + br{part}) * k + t + bc{part}) / 4u")
                    }
                    (false, false) if variant == MatMulCoopVariant::AT => format!(
                        "(((t + br{part}) * params.groups + block_id) * n + tile_col + bc{part}) / 4u"
                    ),
                    (false, false) => {
                        format!("((block_id * k + t + br{part}) * n + tile_col + bc{part}) / 4u")
                    }
                }
            } else if weight_gradient {
                if a {
                    format!("((t + ar{part}) * output_channels + tile_row + ac{part}) / 4u")
                } else {
                    format!(
                        "(((image{part} * params.in_h + u32(ih{part})) * params.in_w + u32(iw{part})) * channels + bchannel{part}) / 4u"
                    )
                }
            } else if input_gradient && !a {
                format!(
                    "(((wgid.z * params.out_h + u32(ih{part}) / STRIDE) * params.out_w + u32(iw{part}) / STRIDE) * channels + t % channels + bc{part}) / 4u"
                )
            } else if packed && !a {
                format!(
                    "(((wgid.z * params.in_h + u32(ih{part})) * params.in_w + u32(iw{part})) * channels + t % channels + bc{part}) / 4u"
                )
            } else {
                match (a, transposed) {
                    (true, false) => format!("((tile_row + ar{part}) * k + t + ac{part}) / 4u"),
                    (true, true) => format!("((t + ar{part}) * m + tile_row + ac{part}) / 4u"),
                    (false, false) => format!("((t + br{part}) * n + tile_col + bc{part}) / 4u"),
                    (false, true) => format!("((tile_col + br{part}) * k + t + bc{part}) / 4u"),
                }
            };
            let k_component = if a != transposed {
                format!("{prefix}c{part}")
            } else {
                format!("{prefix}r{part}")
            };
            let source = format!("{buffer}[{index}]");
            let storage_type = if a { types.a } else { types.b };
            let value = if storage_type == types.multiply {
                source
            } else if types.multiply == DType::BF16 {
                let components: Vec<_> = ["x", "y", "z", "w"]
                    .map(|c| format!("round_bf16({source}.{c})"))
                    .into();
                format!("vec4<bf16>({})", components.join(", "))
            } else {
                format!("vec4<{multiply_type}>({source})")
            };
            if packed {
                loads += &format!("{prefix}v{part} = vec4<{multiply_type}>(0.0);\n");
                if weight_gradient && a {
                    loads += &format!(
                        "if tile_row + ac{part} < m && t + ar{part} < end {{\n{prefix}v{part} = {value};\n}}\n"
                    );
                } else if weight_gradient {
                    loads += &format!(
                        "let reduction{part} = t + br{part};\nlet spatial{part} = params.out_h * params.out_w;\nlet image{part} = divide_exact(reduction{part}, spatial{part}, params.output_spatial_multiplier);\nlet position{part} = reduction{part} - image{part} * spatial{part};\nlet oh{part} = divide_exact(position{part}, params.out_w, params.column_width_multiplier);\nlet ow{part} = position{part} - oh{part} * params.out_w;\nlet ih{part} = i32(oh{part} * STRIDE) + bh_offset{part};\nlet iw{part} = i32(ow{part} * STRIDE) + bw_offset{part};\nif reduction{part} < end && bc_full{part} < n && ih{part} >= 0 && iw{part} >= 0 && ih{part} < i32(params.in_h) && iw{part} < i32(params.in_w) {{\n{prefix}v{part} = {value};\n}}\n"
                    );
                } else if a {
                    loads +=
                        &format!("if tile_row + ar{part} < m {{\n{prefix}v{part} = {value};\n}}\n");
                } else if input_gradient {
                    loads += &format!(
                        "let tap{part} = t / channels;\nlet ih{part} = ih0_{part} - i32(tap{part} / KW);\nlet iw{part} = iw0_{part} - i32(tap{part} % KW);\nif bp{part} < n && ih{part} >= 0 && iw{part} >= 0 && ih{part} % i32(STRIDE) == 0 && iw{part} % i32(STRIDE) == 0 && u32(ih{part}) / STRIDE < params.out_h && u32(iw{part}) / STRIDE < params.out_w {{\n{prefix}v{part} = {value};\n}}\n"
                    );
                } else {
                    loads += &format!(
                        "let tap{part} = t / channels;\nlet ih{part} = ih0_{part} + i32(tap{part} / KW);\nlet iw{part} = iw0_{part} + i32(tap{part} % KW);\nif bp{part} < n && ih{part} >= 0 && iw{part} >= 0 && ih{part} < i32(params.in_h) && iw{part} < i32(params.in_w) {{\n{prefix}v{part} = {value};\n}}\n"
                    );
                }
                let stride = if a { stride_a } else { stride_b } / 4;
                writes += &format!(
                    "{shared}[{prefix}r{part} * {stride}u + {prefix}c{part} / 4u] = {prefix}v{part};\n"
                );
                continue;
            } else if !typed && stage == 32 && splits > 1 {
                loads += &format!(
                    "{prefix}v{part} = vec4<{multiply_type}>(0.0);\nif t + {k_component} < end {{\n{prefix}v{part} = {value};\n"
                );
            } else {
                loads += &format!("{prefix}v{part} = {value};\n");
            }
            for (lane, component) in ["x", "y", "z", "w"].iter().enumerate() {
                let r = if transposed {
                    format!("({prefix}c{part} + {lane}u)")
                } else {
                    format!("{prefix}r{part}")
                };
                let c = if transposed {
                    format!("{prefix}r{part}")
                } else {
                    format!("({prefix}c{part} + {lane}u)")
                };
                if a && !transform.is_empty() {
                    loads += &format!(
                        "{{ let gr = tile_row + {r}; let tc = t + {c};\n{prefix}v{part}.{component} = {prefix}v{part}.{component}{transform}; }}\n"
                    );
                }
                let stride = if a { stride_a } else { stride_b };
                writes +=
                    &format!("{shared}[{r} * {stride}u + {c}] = {prefix}v{part}.{component};\n");
            }
            if !typed && stage == 32 && splits > 1 {
                loads += "}\n";
            }
        }
    }
    let mut multiply = String::new();
    for q in 0..stage / 16 {
        for i in 0..2 {
            if weight_gradient {
                multiply += &format!(
                    "let ai{i}_{q} = {}u * {stride_a}u + wr * 32u + {}u;\nlet a{i}_{q} = coopLoad<coop_mat16x16<{multiply_type},A>>(&sa[ai{i}_{q} / 4u], {}u);\n",
                    q * 16,
                    i * 16,
                    stride_a / 4
                );
                continue;
            }
            let pointer = if packed {
                format!("&sa[ai{i}_{q} / 4u]")
            } else {
                format!("&sa[ai{i}_{q}]")
            };
            // Cooperative strides are in pointee units, including vectors.
            let load_stride = if packed { stride_a / 4 } else { stride_a };
            multiply += &format!(
                "let ai{i}_{q} = (wr * 32u + {}u) * {stride_a}u + {}u;\nlet a{i}_{q} = coopLoadT<coop_mat16x16<{multiply_type},A>>({pointer}, {load_stride}u);\n",
                i * 16,
                q * 16
            );
        }
        for j in 0..wave_cols / 16 {
            if weight_gradient {
                multiply += &format!(
                    "let bi{j}_{q} = {}u * {stride_b}u + wc * {wave_cols}u + {}u;\nlet b{j}_{q} = coopLoadT<coop_mat16x16<{multiply_type},B>>(&sb[bi{j}_{q} / 4u], {}u);\n",
                    q * 16,
                    j * 16,
                    stride_b / 4
                );
                continue;
            }
            if packed {
                let load_stride = stride_b / 4;
                multiply += &format!(
                    "let bi{j}_{q} = (wc * {wave_cols}u + {}u) * {stride_b}u + {}u;\nlet b{j}_{q} = coopLoad<coop_mat16x16<{multiply_type},B>>(&sb[bi{j}_{q} / 4u], {load_stride}u);\n",
                    j * 16,
                    q * 16
                );
                continue;
            }
            multiply += &format!(
                "let bi{j}_{q} = {}u + wc * {wave_cols}u + {}u;\nlet b{j}_{q} = coopLoadT<coop_mat16x16<{multiply_type},B>>(&sb[bi{j}_{q}], {stride_b}u);\n",
                q * 16 * stride_b,
                j * 16
            );
        }
        for i in 0..2 {
            for j in 0..wave_cols / 16 {
                multiply +=
                    &format!("acc{i}_{j} = coopMultiplyAdd(a{i}_{q}, b{j}_{q}, acc{i}_{j});\n");
            }
        }
    }
    let (first_load, step_load, next_load) = if shape.prefetch {
        (
            format!("{{ let t = begin; if t < end {{\n{loads}}} }}"),
            String::new(),
            format!("{{ let t = t + {stage}u; if t < end {{\n{loads}}} }}"),
        )
    } else {
        (String::new(), loads, String::new())
    };
    let mut extensions = String::new();
    let fields = [types.a, types.b, types.multiply, types.output];
    if fields.contains(&DType::F16) {
        extensions.push_str("enable f16;\n");
    }
    if fields.contains(&DType::BF16) {
        extensions.push_str("enable wgpu_bfloat16;\n");
    }
    let range = if grouped {
        "let block_id = wgid.z / params.splits; let part_id = wgid.z % params.splits;\nlet span = grouped_span(k, params.splits, part_id); let begin = span.x; let end = span.y;".to_owned()
    } else if weight_gradient {
        format!(
            "let tiles = (k + 15u) / 16u; let count = tiles / {splits}u; let extra = tiles % {splits}u;\nlet first_tile = wgid.z * count + min(wgid.z, extra);\nlet begin = first_tile * 16u; let end = min(k, (first_tile + count + select(0u, 1u, wgid.z < extra)) * 16u);"
        )
    } else if packed {
        "let begin = 0u; let end = k;".to_owned()
    } else if typed {
        format!(
            "let chunk = (k / {splits}u / 32u) * 32u;\nlet begin = wgid.z * chunk;\nlet end = select(begin + chunk, k, wgid.z + 1u == {splits}u);"
        )
    } else {
        format!(
            "let chunk = ((k / 16u + {splits}u - 1u) / {splits}u) * 16u;\nlet begin = wgid.z * chunk;\nlet end = min(k, begin + chunk);"
        )
    };
    let output_type = MatrixTypes::name(types.output);
    let bounds = if weight_gradient {
        " && tile_row + row < m && tile_col + col < n && (tile_col + col) % channels < params.in_channels"
    } else if packed {
        " && tile_row + row < m && tile_col + col < n"
    } else {
        ""
    };
    let scatter = if scratch {
        let convert = if types.output == DType::BF16 {
            "round_bf16"
        } else {
            output_type
        };
        let add = if addend { " + src[index]" } else { "" };
        let output_index = if weight_gradient {
            "((wgid.z * m + tile_row + row) * params.in_channels + (tile_col + col) % channels) * KH * KW + (tile_col + col) / channels"
        } else {
            "output_base + (tile_row + row) * output_stride + tile_col + col"
        };
        format!(
            "workgroupBarrier();\nfor (var index_local = lid.x; index_local < {}u; index_local += {lanes}u) {{\nlet row = index_local / {cols}u;\nlet col = index_local % {cols}u;\nlet tile = (row / 32u) * 2u + col / {wave_cols}u;\nif tile >= wave_base && tile < wave_base + {lanes}u / sg_size{bounds} {{\nlet index = {output_index};\n$C_BUFFER[index] = {convert}(results[index_local]{add});\n}}\n}}\nworkgroupBarrier();",
            rows * cols
        )
    } else {
        String::new()
    };
    let (params, dimensions) = match group {
        ShaderGroup::BlockMatMulTyped { .. } => (
            include_str!("../shaders/matmul_grouped_params.wgsl").to_owned(),
            "let m = params.m; let n = params.n; let k = params.k;",
        ),
        ShaderGroup::Conv2dPacked {
            kernel_h,
            kernel_w,
            stride,
            ..
        } => (
            format!(
                "{}\n{}\nconst KH: u32 = {kernel_h}u;\nconst KW: u32 = {kernel_w}u;\nconst STRIDE: u32 = {stride}u;\n",
                include_str!("../shaders/conv2d_params.wgsl"),
                crate::divisor::SHADER
            ),
            if weight_gradient {
                "let m = params.out_channels; let channels = ((params.in_channels + 31u) / 32u) * 32u; let output_channels = ((params.out_channels + 31u) / 32u) * 32u; let n = channels * KH * KW; let k = params.batch * params.out_h * params.out_w;"
            } else if input_gradient {
                "let m = params.in_channels; let n = params.in_h * params.in_w; let channels = ((params.out_channels + 31u) / 32u) * 32u; let k = channels * KH * KW;"
            } else {
                "let m = params.out_channels;\nlet n = params.out_h * params.out_w;\nlet channels = ((params.in_channels + 31u) / 32u) * 32u;\nlet k = channels * KH * KW;"
            },
        ),
        _ => (
            "struct Params { m: u32, n: u32, k: u32, _pad: u32, }".to_owned(),
            "let m = params.m; let n = params.n; let k = params.k;",
        ),
    };
    let shared_decl = if packed {
        format!(
            "var<workgroup> sa: array<vec4<{multiply_type}>, {}>;\nvar<workgroup> sb: array<vec4<{multiply_type}>, {}>;",
            (if weight_gradient { stage } else { rows }) * stride_a / 4,
            (if weight_gradient { stage } else { cols }) * stride_b / 4
        )
    } else {
        format!(
            "var<workgroup> sa: array<{multiply_type}, {}>;\nvar<workgroup> sb: array<{multiply_type}, {}>;",
            rows * stride_a,
            stage * stride_b
        )
    };
    let source = preprocess(
        include_str!("../shaders/matmul_coop_tiled.wgsl"),
        &[
            ("$PARAMS_TYPE", &params),
            ("$DIMS", dimensions),
            ("$SHARED_DECL", &shared_decl),
            (
                "$OUTPUT_LAYOUT",
                if grouped {
                    if variant == MatMulCoopVariant::AT {
                        "let output_base = (part_id * params.groups + block_id) * m * n; let output_stride = n;"
                    } else {
                        "let output_base = part_id * params.groups * m * n + block_id * n; let output_stride = params.groups * n;"
                    }
                } else {
                    "let output_base = wgid.z * m * n; let output_stride = n;"
                },
            ),
            ("$COLUMNS", &cols.to_string()),
            ("$EXTENSIONS", &extensions),
            ("$A_TYPE", MatrixTypes::name(types.a)),
            ("$MULTIPLY_TYPE", multiply_type),
            ("$THREADS", &lanes.to_string()),
            ("$RANGE", &range),
            (
                "$RESULT_DECL",
                &if scratch {
                    format!("var<workgroup> results: array<f32, {}>;", rows * cols)
                } else {
                    String::new()
                },
            ),
            ("$SCATTER", &scatter),
            (
                "$ROUND",
                if fields.contains(&DType::BF16) {
                    include_str!("../shaders/bf16_round.wgsl")
                } else {
                    ""
                },
            ),
            ("$STAGE", &stage.to_string()),
            ("$A_SIZE", &(rows * stride_a).to_string()),
            ("$B_SIZE", &(stage * stride_b).to_string()),
            ("$SPLITS", &splits.to_string()),
            (
                "$ADD_DECL",
                if addend {
                    "var<storage> src: array<f32>;"
                } else {
                    ""
                },
            ),
            ("$PROLOGUE_DECL", &decl),
            ("$CACHE_DECL", &cache_decl),
            ("$CACHE_INIT", &cache_init),
            ("$ACC_INIT", &init),
            ("$PARTIAL_STEP", &partial_step),
            ("$PARTIAL_FINISH", &partial_finish),
            ("$VARIABLES", &variables),
            ("$FIRST_LOAD", &first_load),
            ("$STEP_LOAD", &step_load),
            ("$NEXT_LOAD", &next_load),
            ("$WRITES", &writes),
            ("$MULTIPLY", &multiply),
            ("$STORES", &stores),
        ],
    );
    let module = matmul_module(
        &source,
        &format!("array<vec4<{}>>", MatrixTypes::name(types.b)),
        &format!("array<{output_type}>"),
        Some(lanes),
        1,
    );
    if packed {
        let (a_name, b_name) = if weight_gradient {
            ("grad_out", "src")
        } else if input_gradient {
            ("weight", "grad_out")
        } else {
            ("weight", "src")
        };
        ShaderModule::new(
            &module
                .source
                .replace("matrix_a", a_name)
                .replace("matrix_b", b_name)
                .replace("matrix_c", "dst"),
        )
    } else {
        module
    }
}

pub(super) fn generate_channel_pack(dtype: DType, transpose: bool) -> ShaderModule {
    let extension = match dtype {
        DType::F32 => "",
        DType::F16 => "enable f16;",
        DType::BF16 => "enable wgpu_bfloat16;",
        _ => panic!("unsupported channel storage"),
    };
    ShaderModule::new(&preprocess(
        include_str!("../shaders/pack_channels.wgsl"),
        &[
            ("$EXTENSIONS", extension),
            ("$TYPE", MatrixTypes::name(dtype)),
            (
                "$INDEX_A",
                if transpose {
                    "(channel * params.outer + outer) * params.spatial + spatial"
                } else {
                    "(outer * params.channels + channel) * params.spatial + spatial"
                },
            ),
            (
                "$INDEX_B",
                if transpose {
                    "((channel + 1u) * params.outer + outer) * params.spatial + spatial"
                } else {
                    "(outer * params.channels + channel + 1u) * params.spatial + spatial"
                },
            ),
        ],
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grouped_typed_modules_validate_and_emit_spirv() {
        let mut count = 0;
        for multiply in [DType::F32, DType::F16, DType::BF16] {
            let storage = if multiply == DType::F32 {
                vec![DType::F32]
            } else {
                vec![DType::F32, multiply]
            };
            for &a in &storage {
                for &b in &storage {
                    for &output in &storage {
                        for (transpose_a, transpose_b) in
                            [(false, false), (true, false), (false, true)]
                        {
                            let group = ShaderGroup::BlockMatMulTyped {
                                transpose_a,
                                transpose_b,
                                types: MatrixTypes {
                                    a,
                                    b,
                                    multiply,
                                    output,
                                },
                            };
                            for tiled in [false, true] {
                                let mut module = if tiled {
                                    generate_tiled_coop_matmul(
                                        group,
                                        CooperativeMatmulShape {
                                            columns: 64,
                                            k_stage: 32,
                                            prefetch: true,
                                        },
                                        3,
                                        None,
                                    )
                                } else {
                                    super::super::qualified_cooperative::generate(group)
                                };
                                let mut binding = 0;
                                for (_, var) in module.module.global_variables.iter_mut() {
                                    if matches!(
                                        var.space,
                                        naga::AddressSpace::Storage { .. }
                                            | naga::AddressSpace::Uniform
                                    ) {
                                        var.binding =
                                            Some(naga::ResourceBinding { group: 0, binding });
                                        binding += 1;
                                    }
                                }
                                let info = naga::valid::Validator::new(
                                    naga::valid::ValidationFlags::all(),
                                    naga::valid::Capabilities::all(),
                                )
                                .validate(&module.module)
                                .unwrap();
                                let words = naga::back::spv::write_vec(
                                    &module.module,
                                    &info,
                                    &naga::back::spv::Options {
                                        lang_version: (1, 3),
                                        flags: naga::back::spv::WriterFlags::empty(),
                                        ..Default::default()
                                    },
                                    None,
                                )
                                .unwrap();
                                assert!(!words.is_empty());
                                count += 1;
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(count, 102);
    }

    #[test]
    fn channel_tiled_modules_validate_and_emit_spirv() {
        let mut modules = Vec::new();
        for multiply in [DType::F32, DType::F16, DType::BF16] {
            modules.push(generate_channel_pack(multiply, false));
            modules.push(generate_channel_pack(multiply, true));
            let storage = if multiply == DType::F32 {
                vec![DType::F32]
            } else {
                vec![DType::F32, multiply]
            };
            for &input in &storage {
                for &output in &storage {
                    modules.push(generate_tiled_coop_matmul(
                        ShaderGroup::Conv2dPacked {
                            pass: ConvolutionPass::Forward,
                            splits: 1,
                            kernel_h: 3,
                            kernel_w: 5,
                            stride: 2,
                            types: MatrixTypes {
                                a: input,
                                b: input,
                                multiply,
                                output,
                            },
                        },
                        CooperativeMatmulShape {
                            columns: 64,
                            k_stage: 32,
                            prefetch: false,
                        },
                        1,
                        None,
                    ));
                }
            }
        }
        for multiply in [DType::F32, DType::F16, DType::BF16] {
            for pass in [
                ConvolutionPass::InputGradient,
                ConvolutionPass::WeightGradient,
            ] {
                for splits in [1, 3] {
                    if pass == ConvolutionPass::InputGradient && splits > 1 {
                        continue;
                    }
                    modules.push(generate_tiled_coop_matmul(
                        ShaderGroup::Conv2dPacked {
                            pass,
                            splits,
                            kernel_h: 2,
                            kernel_w: 4,
                            stride: 2,
                            types: MatrixTypes {
                                a: multiply,
                                b: multiply,
                                multiply,
                                output: DType::F32,
                            },
                        },
                        CooperativeMatmulShape {
                            columns: 64,
                            k_stage: 32,
                            prefetch: false,
                        },
                        splits,
                        None,
                    ));
                }
            }
        }
        assert_eq!(modules.len(), 24);
        for mut module in modules {
            let mut binding = 0;
            for (_, var) in module.module.global_variables.iter_mut() {
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
            .validate(&module.module)
            .unwrap();
            let words = naga::back::spv::write_vec(
                &module.module,
                &info,
                &naga::back::spv::Options {
                    lang_version: (1, 3),
                    flags: naga::back::spv::WriterFlags::empty(),
                    ..Default::default()
                },
                None,
            )
            .unwrap();
            assert!(!words.is_empty());
        }
    }

    #[test]
    fn typed_tiled_modules_validate_and_emit_spirv() {
        let shape = CooperativeMatmulShape {
            columns: 64,
            k_stage: 32,
            prefetch: true,
        };
        for multiply in [DType::F32, DType::F16, DType::BF16] {
            for storage in [DType::F32, multiply] {
                for output in [DType::F32, multiply] {
                    for addend in [false, true] {
                        for (transpose_a, transpose_b) in
                            [(false, false), (true, false), (false, true)]
                        {
                            let mut module = generate_tiled_coop_matmul(
                                ShaderGroup::MatMulTyped {
                                    transpose_a,
                                    transpose_b,
                                    addend,
                                    types: MatrixTypes {
                                        a: storage,
                                        b: storage,
                                        multiply,
                                        output,
                                    },
                                },
                                shape,
                                if addend { 1 } else { 4 },
                                None,
                            );
                            let mut binding = 0;
                            for (_, var) in module.module.global_variables.iter_mut() {
                                if matches!(
                                    var.space,
                                    naga::AddressSpace::Storage { .. }
                                        | naga::AddressSpace::Uniform
                                ) {
                                    var.binding = Some(naga::ResourceBinding { group: 0, binding });
                                    binding += 1;
                                }
                            }
                            let info = naga::valid::Validator::new(
                                naga::valid::ValidationFlags::all(),
                                naga::valid::Capabilities::all(),
                            )
                            .validate(&module.module)
                            .unwrap();
                            let words = naga::back::spv::write_vec(
                                &module.module,
                                &info,
                                &naga::back::spv::Options {
                                    lang_version: (1, 3),
                                    flags: naga::back::spv::WriterFlags::empty(),
                                    ..Default::default()
                                },
                                None,
                            )
                            .unwrap();
                            assert!(!words.is_empty());
                        }
                    }
                }
            }
        }
    }
}
