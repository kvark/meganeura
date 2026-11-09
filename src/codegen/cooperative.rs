use super::{
    MatMulCoopVariant, MatrixTypes, ShaderGroup, ShaderModule, TypedMatmulLayout, coop_shape,
    preprocess,
};
use crate::{DType, kernels::conv::Params};

/// The output tile of one checked workgroup. Each subgroup owns a 32×32 block
/// of native accumulators, so a wider tile stages every operand element once
/// for more outputs. Every output still adds the same products in the same
/// order, so all tiles compute identical values.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct CheckedTile {
    pub rows: u32,
    pub columns: u32,
}

impl Default for CheckedTile {
    fn default() -> Self {
        Self::BASE
    }
}

impl CheckedTile {
    /// The tile of every checked product unless tuning measures another.
    pub const BASE: Self = Self {
        rows: 32,
        columns: 32,
    };
    /// Wider tiles, which forward convolutions may use.
    pub const WIDE: [Self; 5] = [
        Self {
            rows: 32,
            columns: 64,
        },
        Self {
            rows: 64,
            columns: 32,
        },
        Self {
            rows: 64,
            columns: 64,
        },
        Self {
            rows: 64,
            columns: 128,
        },
        Self {
            rows: 128,
            columns: 64,
        },
    ];

    pub fn valid(self) -> bool {
        self == Self::BASE || Self::WIDE.contains(&self)
    }

    /// 32×32 blocks, one per subgroup when the width matches the hint.
    pub fn blocks(self) -> u32 {
        (self.rows / 32) * (self.columns / 32)
    }

    /// Invocations of a wide tile: one subgroup of the device's default
    /// `subgroup` width (32 when unknown) per block, from one full 64-lane
    /// subgroup up to the portable 256.
    pub fn threads(self, subgroup: u32) -> u32 {
        let width = if subgroup == 0 { 32 } else { subgroup };
        width.saturating_mul(self.blocks()).clamp(64, 256)
    }

    /// Workgroup memory of a wide tile: both staged operands, plus an F32
    /// result tile unless accumulators store straight to the output.
    pub fn shared_bytes(self, multiply: DType, direct_output: bool) -> u32 {
        let staged = (self.rows + self.columns) * 32 * MatrixTypes::bytes(multiply) as u32;
        staged
            + if direct_output {
                0
            } else {
                self.rows * self.columns * 4
            }
    }
}

// Keep partition boundaries identical across native tile shapes.
// A 32-element block contains whole 8-wide or 16-wide K tiles.
pub(super) const SPLIT_RANGE: &str = "let blocks = (k + 31u) / 32u;\n\
    let base = blocks / params.splits;\n\
    let extra = blocks % params.splits;\n\
    let first = (wgid.z * base + min(wgid.z, extra)) * 32u;\n\
    let count = base + select(0u, 1u, wgid.z < extra);\n\
    let end = min(first + count * 32u, k);";

/// Dense and implicit-GEMM convolutions share the same checked cooperative
/// loop. Aligned native operands load directly; conversions and tails stage
/// in the multiplication type. Accumulators and split partials remain F32.
pub(super) fn generate(
    group: ShaderGroup,
    types: MatrixTypes,
    shape: [u32; 3],
    subgroup: u32,
    layout: TypedMatmulLayout,
) -> ShaderModule {
    generate_with_params(
        group,
        types,
        shape,
        None,
        subgroup,
        layout,
        CheckedTile::BASE,
    )
}

pub(super) fn generate_with_params(
    group: ShaderGroup,
    types: MatrixTypes,
    shape: [u32; 3],
    params: Option<&Params>,
    subgroup: u32,
    layout: TypedMatmulLayout,
    tile: CheckedTile,
) -> ShaderModule {
    types.validate();
    assert!(
        super::checked_cooperative_shape(&[shape]).is_some(),
        "unsupported cooperative tile {shape:?}"
    );
    let wide = tile != CheckedTile::BASE;
    assert!(
        !wide || (tile.valid() && matches!(group, ShaderGroup::Conv2dChecked { .. })),
        "{tile:?} is not a forward convolution tile"
    );
    let [m, n, k] = shape;
    let pointwise = group == ShaderGroup::Conv2dPointwiseGradInput;
    let variant = if pointwise {
        Some(MatMulCoopVariant::AT)
    } else {
        coop_shape(group).map(|(_, variant)| variant)
    };
    let split_range = SPLIT_RANGE.replace("wgid.", "group.");
    let vector_conv = !pointwise && params.is_some_and(Params::vectorized_forward);
    let vector_dense = layout.vectorized && variant.is_some();
    let global_a = layout.direct_inputs && vector_dense && types.a == types.multiply;
    let global_b = layout.direct_inputs && vector_dense && types.b == types.multiply;
    let (declaration, dimensions, range, a, b, index, destination, add) = match group {
        ShaderGroup::MatMul
        | ShaderGroup::MatMulAT
        | ShaderGroup::MatMulBT
        | ShaderGroup::MatMulAdd
        | ShaderGroup::MatMulATAdd
        | ShaderGroup::MatMulBTAdd => {
            let (addend, variant) = coop_shape(group).unwrap();
            let mut declaration = String::from(
                "struct Params { m: u32, n: u32, k: u32, splits: u32 }\n\
                 var<storage> matrix_a: array<$A_TYPE>;\n\
                 var<storage> matrix_b: array<$B_TYPE>;\n\
                 var<storage, read_write> matrix_c: array<$OUT_TYPE>;\n\
                 var<uniform> params: Params;\n",
            );
            if addend {
                declaration.push_str("var<storage> src: array<f32>;\n");
            }
            if vector_dense {
                if !global_a {
                    declaration = declaration.replace("array<$A_TYPE>", "array<vec4<$A_TYPE>>");
                }
                if !global_b {
                    declaration = declaration.replace("array<$B_TYPE>", "array<vec4<$B_TYPE>>");
                }
            }
            (
                declaration,
                "let m = params.m; let n = params.n; let k = params.k;",
                split_range.as_str(),
                if variant == MatMulCoopVariant::AT {
                    "$DENSE_A_AT"
                } else {
                    "$DENSE_A_NN"
                },
                if variant == MatMulCoopVariant::BT {
                    "$DENSE_B_BT"
                } else {
                    "$DENSE_B_NN"
                },
                "(group.z * m + r) * n + c",
                "matrix_c",
                if addend { " + src[index]" } else { "" },
            )
        }
        ShaderGroup::Conv2dBF16 { .. }
        | ShaderGroup::Conv2dChecked { .. }
        | ShaderGroup::Conv2dPointwiseGradInput => {
            let (kernel_h, kernel_w, stride) = match group {
                ShaderGroup::Conv2dBF16 {
                    kernel_h,
                    kernel_w,
                    stride,
                    ..
                }
                | ShaderGroup::Conv2dChecked {
                    kernel_h,
                    kernel_w,
                    stride,
                } => (kernel_h, kernel_w, stride),
                ShaderGroup::Conv2dPointwiseGradInput => (1, 1, 1),
                _ => unreachable!(),
            };
            assert!(kernel_h > 0 && kernel_w > 0 && stride > 0);
            if let ShaderGroup::Conv2dBF16 { output_bf16, .. } = group {
                assert_eq!(
                    types,
                    MatrixTypes {
                        output: if output_bf16 { DType::BF16 } else { DType::F32 },
                        ..MatrixTypes::BF16
                    }
                );
            }
            let declaration = format!(
                "{}\n{}\nconst KH: u32 = {kernel_h}u;\nconst KW: u32 = {kernel_w}u;\n\
                 const STRIDE: u32 = {stride}u;\n{}",
                include_str!("../shaders/conv2d_params.wgsl"),
                if params.is_some() {
                    crate::divisor::NATIVE_SHADER
                } else {
                    crate::divisor::SHADER
                },
                if vector_conv {
                    "var<storage> src: array<vec4<$B_TYPE>>;\nvar<storage> weight: array<vec4<$A_TYPE>>;\nvar<storage, read_write> dst: array<$OUT_TYPE>;"
                } else if pointwise && vector_dense {
                    "var<storage> grad_out: array<vec4<$B_TYPE>>; var<storage> weight: array<vec4<$A_TYPE>>; var<storage,read_write> dst: array<$OUT_TYPE>;"
                } else if pointwise {
                    "var<storage> grad_out: array<$B_TYPE>; var<storage> weight: array<$A_TYPE>; var<storage,read_write> dst: array<$OUT_TYPE>;"
                } else {
                    include_str!("../shaders/conv2d_checked.wgsl")
                }
            );
            let params_decl = match params {
                Some(p) => format!(
                    "const params = Params({}u, {}u, {}u, {}u, {}u, {}u, {}u, {}u, {}u, {}u, {}u, {}u, {}u, {}u, {}u, {}u);",
                    p.batch,
                    p.in_channels,
                    p.in_h,
                    p.in_w,
                    p.out_channels,
                    p.kernel_h,
                    p.kernel_w,
                    p.stride,
                    p.padding_h,
                    p.out_h,
                    p.out_w,
                    p.padding_w,
                    p.kernel_w_multiplier,
                    p.kernel_hw_multiplier,
                    p.column_width_multiplier,
                    p.output_spatial_multiplier,
                ),
                None => "var<uniform> params: Params;".to_owned(),
            };
            (
                format!("{declaration}\n{params_decl}"),
                if pointwise {
                    "let m=params.in_channels; let n=params.in_h*params.in_w; let k=params.out_channels;"
                } else {
                    "let m = params.out_channels; let n = params.out_h * params.out_w; let k = params.in_channels * KH * KW;"
                },
                "let first = 0u; let end = k;",
                if pointwise {
                    "weight[ak * m + ar]"
                } else {
                    "weight[ar * k + ak]"
                },
                if pointwise {
                    "grad_out[(group.z * k + bk) * n + bc]"
                } else {
                    "load_pixel(group.z, bk, bc)"
                },
                "(group.z * m + r) * n + c",
                "dst",
                "",
            )
        }
        _ => unreachable!(),
    };
    assert!(!layout.direct_output || (types.output == DType::F32 && add.is_empty()));
    let direct_output = layout.direct_output;
    // Callers select this only for whole native output tiles. Partial 32x32
    // workgroups still guard each subgroup's tile before the global store.
    let result_pointer = |rows: u32, columns: u32| {
        if direct_output {
            format!(
                "&{destination}[(group.z * m + row + a_offset / 32u + {rows}u) * n + col + b_offset + {columns}u]"
            )
        } else {
            format!("&result[a_offset + b_offset + {}u]", rows * 32 + columns)
        }
    };
    let template = include_str!("../shaders/matmul_checked.wgsl");
    let scatter = super::template_section(template, "scatter");
    let mut extensions = String::from("enable wgpu_cooperative_matrix;\n");
    if types.uses(DType::F16) {
        extensions.push_str("enable f16;\n");
    }
    if types.uses(DType::BF16) {
        extensions.push_str("enable wgpu_bfloat16;\n");
    }
    // F32 storage reaches a BF16 multiplication through nearest-even
    // rounding; every other operand conversion is exact or a plain cast.
    let cast = |storage| {
        if types.multiply == DType::BF16 && storage == DType::F32 {
            "round_bf16"
        } else {
            MatrixTypes::name(types.multiply)
        }
    };
    let (a_cast, b_cast) = (cast(types.a), cast(types.b));
    let convert = |value: &str, storage| {
        let cast = cast(storage);
        if types.multiply == DType::BF16 && storage == DType::F32 {
            format!(
                "vec4<bf16>({cast}({value}.x), {cast}({value}.y), {cast}({value}.z), {cast}({value}.w))"
            )
        } else {
            format!("vec4<{cast}>({value})")
        }
    };
    // Logical output and K coordinates map to one storage index for both
    // vector staging and direct cooperative loads.
    let operand = |a_side: bool| {
        let transposed = variant
            == Some(if a_side {
                MatMulCoopVariant::AT
            } else {
                MatMulCoopVariant::BT
            });
        let (name, base, limit) = if a_side {
            ("a", "row", "m")
        } else {
            ("b", "col", "n")
        };
        (name, base, limit, transposed, a_side != transposed)
    };
    let address = |a_side: bool, output: &str, reduction: &str| {
        let (_, _, limit, _, output_outer) = operand(a_side);
        let (index, stride) = if output_outer {
            (format!("({output}) * k + ({reduction})"), "k")
        } else {
            (format!("({reduction}) * {limit} + ({output})"), limit)
        };
        (
            if pointwise && !a_side {
                format!("group.z * k * n + ({index})")
            } else {
                index
            },
            stride,
        )
    };
    let stage = if vector_dense {
        let mut body = String::new();
        for (a_side, global, storage) in [(true, global_a, types.a), (false, global_b, types.b)] {
            if global {
                continue;
            }
            let (name, base, limit, transposed, output_outer) = operand(a_side);
            body += &preprocess(
                include_str!("../shaders/matmul_checked_vector.wgsl"),
                &[
                    ("$OPERAND", name),
                    (
                        "$STORAGE",
                        if pointwise {
                            if a_side { "weight" } else { "grad_out" }
                        } else if a_side {
                            "matrix_a"
                        } else {
                            "matrix_b"
                        },
                    ),
                    ("$OUTPUT_BASE", base),
                    ("$OUTPUT_LIMIT", limit),
                    ("$OUTPUT_OUTER", if output_outer { "true" } else { "false" }),
                    ("$TRANSPOSE", if transposed { "true" } else { "false" }),
                    ("$GLOBAL_INDEX", &address(a_side, "out_pos", "k_pos").0),
                    (
                        "$UNCHECKED",
                        if global_a || global_b {
                            "true"
                        } else {
                            "false"
                        },
                    ),
                    ("$VECTOR", &convert("raw", storage)),
                    ("$CAST", cast(storage)),
                ],
            );
        }
        if body.is_empty() {
            body
        } else {
            format!("for (var index = lane; index < 256u; index += $THREADSu) {{\n{body}}}\n")
        }
    } else if vector_conv {
        include_str!("../shaders/conv2d_checked_vector.wgsl").to_owned()
    } else {
        include_str!("../shaders/matmul_checked_stage.wgsl").to_owned()
    };
    let matrix_load = |a_side: bool, extra_output: &str| {
        let (name, base, _, transposed, _) = operand(a_side);
        let ty = if a_side { "$A_TILE" } else { "$B_TILE" };
        if if a_side { global_a } else { global_b } {
            let tile_output = if a_side { "a_offset / 32u" } else { "b_offset" };
            let (index, stride) = address(
                a_side,
                &format!("{base} + {tile_output} + {extra_output}"),
                "start + inner",
            );
            let load = if transposed { "coopLoad" } else { "coopLoadT" };
            format!("{load}<{ty}>(&matrix_{name}[{index}], {stride})")
        } else if a_side {
            // Output offsets count rows; each scratch row spans 32 elements.
            format!("coopLoadT<{ty}>(&stage_a[a_offset + {extra_output} * 32u + inner], 32u)")
        } else {
            format!("coopLoadT<{ty}>(&stage_b[b_offset + inner * 32u + {extra_output}], 32u)")
        }
    };
    let rounds = types.output == DType::BF16
        || (types.multiply == DType::BF16 && (types.a == DType::F32 || types.b == DType::F32));
    let multiply = MatrixTypes::name(types.multiply);
    // Use one subgroup per rectangular convolution tile where possible. The
    // runtime width may differ, so the shader still pairs or loops over tiles.
    let threads = if m == 8 && group.is_checked_convolution() {
        if subgroup == 0 {
            256
        } else {
            (8 * subgroup).clamp(128, 256)
        }
    } else {
        128
    };
    let a_vector = convert("v", types.a);
    let b_vector = convert("v", types.b);
    // WGSL names cooperative matrices columns × rows, whereas device
    // capabilities and the product shape are M × N × K.
    let a_tile = format!("coop_mat{k}x{m}<{multiply}, A>");
    let b_tile = format!("coop_mat{n}x{k}<{multiply}, B>");
    let c_tile = format!("coop_mat{n}x{m}<f32, C>");
    let (tile_m, tile_n, tile_k) = (m.to_string(), n.to_string(), k.to_string());
    // Substituted after each kernel's own placeholders, which contain these.
    let tail: [(&str, &str); 29] = [
        ("$A_VECTOR", &a_vector),
        ("$B_VECTOR", &b_vector),
        ("$EXTENSIONS", &extensions),
        ("$DECL", &declaration),
        ("$DIMS", dimensions),
        ("$RANGE", range),
        ("$A_LOAD", a),
        ("$B_LOAD", b),
        (
            "$DENSE_A_AT",
            if vector_dense {
                "matrix_a[(ak * m + ar) / 4u][(ak * m + ar) % 4u]"
            } else {
                "matrix_a[ak * m + ar]"
            },
        ),
        (
            "$DENSE_A_NN",
            if vector_dense {
                "matrix_a[(ar * k + ak) / 4u][(ar * k + ak) % 4u]"
            } else {
                "matrix_a[ar * k + ak]"
            },
        ),
        (
            "$DENSE_B_BT",
            if vector_dense {
                "matrix_b[(bc * k + bk) / 4u][(bc * k + bk) % 4u]"
            } else {
                "matrix_b[bc * k + bk]"
            },
        ),
        (
            "$DENSE_B_NN",
            if vector_dense {
                "matrix_b[(bk * n + bc) / 4u][(bk * n + bc) % 4u]"
            } else {
                "matrix_b[bk * n + bc]"
            },
        ),
        ("$INDEX", index),
        ("$DEST", destination),
        (
            "$ROUND",
            if rounds {
                include_str!("../shaders/bf16_round.wgsl")
            } else {
                ""
            },
        ),
        ("$A_TYPE", MatrixTypes::name(types.a)),
        ("$B_TYPE", MatrixTypes::name(types.b)),
        ("$A_CAST", a_cast),
        ("$B_CAST", b_cast),
        ("$OUT_TYPE", MatrixTypes::name(types.output)),
        (
            "$STORE",
            if types.output == DType::BF16 {
                "round_bf16"
            } else {
                MatrixTypes::name(types.output)
            },
        ),
        ("$ADD", add),
        ("$A_TILE", &a_tile),
        ("$B_TILE", &b_tile),
        ("$C_TILE", &c_tile),
        ("$MULTIPLY_TYPE", multiply),
        ("$TILE_M", &tile_m),
        ("$TILE_N", &tile_n),
        ("$TILE_K", &tile_k),
    ];
    if wide {
        let own = wide_substitutions(tile, [m, n], subgroup, direct_output, vector_conv);
        let vars: Vec<_> = own
            .iter()
            .map(|&(key, ref value)| (key, value.as_str()))
            .chain(tail)
            .collect();
        let template = include_str!("../shaders/conv2d_checked_wide.wgsl");
        return ShaderModule::new(&preprocess(
            super::template_section(template, "main"),
            &vars,
        ));
    }
    let a_stage_decl = if global_a {
        String::new()
    } else {
        format!("var<workgroup> stage_a: array<{multiply}, 1024>;")
    };
    let b_stage_decl = if global_b {
        String::new()
    } else {
        format!("var<workgroup> stage_b: array<{multiply}, 1024>;")
    };
    let own: [(&str, &str); 19] = [
        ("$STAGE", &stage),
        ("$A_STAGE_DECL", &a_stage_decl),
        ("$B_STAGE_DECL", &b_stage_decl),
        (
            "$INPUT_BARRIER",
            if global_a && global_b {
                ""
            } else {
                "workgroupBarrier();"
            },
        ),
        ("$A_MATRIX", &matrix_load(true, "0u")),
        ("$A_LOWER_MATRIX", &matrix_load(true, "lower_row")),
        ("$B_MATRIX", &matrix_load(false, "0u")),
        ("$B_RIGHT_MATRIX", &matrix_load(false, "right_col")),
        (
            "$RESULT_DECL",
            if direct_output {
                ""
            } else {
                "var<workgroup> result: array<f32, 1024>;"
            },
        ),
        ("$RESULT_SCATTER", if direct_output { "" } else { scatter }),
        (
            "$STORE_VALID",
            if direct_output {
                " && row + a_offset / 32u < m && col + b_offset < n"
            } else {
                ""
            },
        ),
        ("$RESULT_STRIDE", if direct_output { "n" } else { "32u" }),
        ("$C_LOWER_RIGHT", &result_pointer(8, 8)),
        ("$C_LOWER", &result_pointer(8, 0)),
        ("$C_RIGHT", &result_pointer(0, 8)),
        ("$C_BASE", &result_pointer(0, 0)),
        ("$THREADS", &threads.to_string()),
        (
            "$TRANSPOSE_A",
            if variant == Some(MatMulCoopVariant::AT) {
                "true"
            } else {
                "false"
            },
        ),
        (
            "$TRANSPOSE_B",
            if variant == Some(MatMulCoopVariant::BT) {
                "true"
            } else {
                "false"
            },
        ),
    ];
    let vars: Vec<_> = own.into_iter().chain(tail).collect();
    ShaderModule::new(&preprocess(
        super::template_section(template, "main"),
        &vars,
    ))
}

/// Placeholders of the wide forward convolution kernel: its geometry, its
/// staging, and the 32×32 block of native accumulators each subgroup owns.
fn wide_substitutions(
    tile: CheckedTile,
    [native_rows, native_columns]: [u32; 2],
    subgroup: u32,
    direct_output: bool,
    vector_conv: bool,
) -> Vec<(&'static str, String)> {
    let template = include_str!("../shaders/conv2d_checked_wide.wgsl");
    let CheckedTile { rows, columns } = tile;
    let (row_blocks, column_blocks) = (32 / native_rows, 32 / native_columns);
    let mut accumulators = String::new();
    let mut products = String::new();
    let mut stores = String::new();
    for i in 0..row_blocks {
        let r = i * native_rows;
        products += &format!(
            "                let a{i} = coopLoadT<$A_TILE>(&stage_a[(block_row + {r}u) * 32u + inner], 32u);\n"
        );
    }
    for j in 0..column_blocks {
        let c = j * native_columns;
        products += &format!(
            "                let b{j} = coopLoadT<$B_TILE>(&stage_b[inner * {columns}u + block_col + {c}u], {columns}u);\n"
        );
    }
    for i in 0..row_blocks {
        for j in 0..column_blocks {
            let (r, c) = (i * native_rows, j * native_columns);
            accumulators += &format!("        var acc{i}_{j} = $C_TILE();\n");
            products +=
                &format!("                acc{i}_{j} = coopMultiplyAdd(a{i}, b{j}, acc{i}_{j});\n");
            stores += &if direct_output {
                format!(
                    "            if row + block_row + {r}u < m && col + block_col + {c}u < n {{\n\
                     \x20               coopStoreT(acc{i}_{j}, &dst[(group.z * m + row + block_row + {r}u) * n + col + block_col + {c}u], n);\n\
                     \x20           }}\n"
                )
            } else {
                format!(
                    "            coopStoreT(acc{i}_{j}, &result[(block_row + {r}u) * {columns}u + block_col + {c}u], {columns}u);\n"
                )
            };
        }
    }
    let stage = if vector_conv { "vector_stage" } else { "stage" };
    vec![
        (
            "$WIDE_STAGE",
            super::template_section(template, stage).to_owned(),
        ),
        ("$WIDE_ACCUMULATORS", accumulators),
        ("$WIDE_PRODUCTS", products),
        ("$WIDE_STORES", stores),
        (
            "$RESULT_DECL",
            if direct_output {
                String::new()
            } else {
                format!("var<workgroup> result: array<f32, {}>;", rows * columns)
            },
        ),
        (
            "$RESULT_SCATTER",
            if direct_output {
                String::new()
            } else {
                super::template_section(template, "scatter").to_owned()
            },
        ),
        ("$WIDE_ROWS", rows.to_string()),
        ("$WIDE_COLUMNS", columns.to_string()),
        ("$WIDE_BLOCKS", tile.blocks().to_string()),
        ("$WIDE_BLOCK_COLUMNS", (columns / 32).to_string()),
        ("$WIDE_A_SIZE", (rows * 32).to_string()),
        ("$WIDE_B_SIZE", (32 * columns).to_string()),
        ("$WIDE_A_VECTORS", (rows * 8).to_string()),
        ("$WIDE_B_VECTORS", (8 * columns).to_string()),
        ("$WIDE_COLUMN_VECTORS", (columns / 4).to_string()),
        ("$WIDE_RESULT", (rows * columns).to_string()),
        ("$THREADS", tile.threads(subgroup).to_string()),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_shape_selection_requires_an_implemented_product() {
        use crate::codegen::checked_cooperative_shape as select;
        assert_eq!(select(&[]), None);
        assert_eq!(select(&[[16, 8, 16], [8, 16, 32]]), None);
        assert_eq!(select(&[[8, 8, 8]]), Some([8, 8, 8]));
        assert_eq!(select(&[[8, 16, 16]]), Some([8, 16, 16]));
        assert_eq!(select(&[[8, 8, 8], [8, 16, 16]]), Some([8, 16, 16]));
        assert_eq!(select(&[[8, 16, 16], [16, 16, 16]]), Some([16, 16, 16]));
    }

    #[test]
    fn typed_operand_combinations_are_validated() {
        for types in [
            MatrixTypes::BF16,
            MatrixTypes::reduced(DType::F16),
            MatrixTypes::reduced(DType::F32),
            MatrixTypes {
                a: DType::F32,
                b: DType::F16,
                multiply: DType::F16,
                output: DType::BF16,
            },
            MatrixTypes {
                a: DType::BF16,
                b: DType::F16,
                multiply: DType::F32,
                output: DType::F32,
            },
        ] {
            assert!(types.is_valid(), "{types:?}");
        }
        for types in [
            // Neither half-precision format converts exactly to the other.
            MatrixTypes {
                b: DType::F16,
                ..MatrixTypes::BF16
            },
            MatrixTypes {
                output: DType::U32,
                ..MatrixTypes::BF16
            },
        ] {
            assert!(!types.is_valid(), "{types:?}");
        }
    }

    /// Every group meets every valid operand, multiply and output type;
    /// tile shapes and staging layouts cycle so each group meets every pair.
    #[test]
    fn convolution_and_dense_modules_validate_and_emit_spirv() {
        let capabilities = naga::valid::Capabilities::SHADER_BFLOAT16
            | naga::valid::Capabilities::SHADER_FLOAT16
            | naga::valid::Capabilities::COOPERATIVE_MATRIX
            | naga::valid::Capabilities::SUBGROUP;
        let shapes = [[16, 16, 16], [8, 16, 16], [8, 8, 8]];
        let mut cases = Vec::new();
        for (i, output) in [DType::F32, DType::BF16].into_iter().enumerate() {
            for (j, &shape) in shapes.iter().enumerate() {
                let output_bf16 = output == DType::BF16;
                cases.push((
                    ShaderGroup::Conv2dBF16 {
                        kernel_h: 2,
                        kernel_w: 4,
                        stride: 2,
                        output_bf16,
                    },
                    MatrixTypes {
                        output,
                        ..MatrixTypes::BF16
                    },
                    shape,
                    i + j,
                ));
            }
        }
        let mut types = Vec::new();
        for output in [DType::F32, DType::BF16] {
            for storage in [DType::BF16, DType::F16, DType::F32] {
                for multiply in [DType::BF16, DType::F16, DType::F32] {
                    for b in [DType::F32, multiply] {
                        let candidate = MatrixTypes {
                            a: storage,
                            b,
                            multiply,
                            output,
                        };
                        if candidate.is_valid() && !types.contains(&candidate) {
                            types.push(candidate);
                        }
                    }
                }
            }
        }
        let mut index = 0;
        for group in [
            ShaderGroup::Conv2dChecked {
                kernel_h: 2,
                kernel_w: 4,
                stride: 2,
            },
            ShaderGroup::MatMul,
            ShaderGroup::MatMulAT,
            ShaderGroup::MatMulBT,
            ShaderGroup::MatMulAdd,
            ShaderGroup::MatMulATAdd,
            ShaderGroup::MatMulBTAdd,
        ] {
            for &types in &types {
                cases.push((group, types, shapes[(index / 3) % 3], index));
                index += 1;
            }
        }
        for (group, types, shape, index) in cases {
            let layout = [
                super::super::TypedMatmulLayout::default(),
                super::super::TypedMatmulLayout {
                    vectorized: true,
                    direct_output: false,
                    ..Default::default()
                },
                super::super::TypedMatmulLayout::for_product(
                    types,
                    coop_shape(group).is_some_and(|(addend, _)| addend),
                    [64, 64, 64],
                ),
            ][index % 3];
            let subgroup = if layout.vectorized { 16 } else { 32 };
            let mut module = super::super::generate_module_coop_typed_with_subgroup(
                group, types, shape, layout, subgroup,
            );
            if shape[0] == 8 && group.is_checked_convolution() {
                assert_eq!(
                    module.module.entry_points[0].workgroup_size,
                    [8 * subgroup, 1, 1]
                );
            }
            super::super::validate_shader_for_test(&mut module.module, capabilities, (1, 3))
                .unwrap_or_else(|error| panic!("{group:?} {types:?} {shape:?}: {error}"));
        }
    }

    #[test]
    fn specialized_convolutions_validate() {
        let capabilities = naga::valid::Capabilities::SHADER_BFLOAT16
            | naga::valid::Capabilities::SHADER_FLOAT16
            | naga::valid::Capabilities::COOPERATIVE_MATRIX
            | naga::valid::Capabilities::SUBGROUP;
        for shape in [[16, 16, 16], [8, 16, 16], [8, 8, 8]] {
            for multiply in [DType::F16, DType::BF16] {
                for (h, w, ci, kh, kw, stride, pad) in [
                    (14, 14, 64, 1, 1, 1, 0),
                    (7, 7, 64, 1, 1, 1, 0),
                    (11, 13, 3, 3, 3, 2, 1),
                ] {
                    let params = Params {
                        batch: 2,
                        in_channels: ci,
                        in_h: h,
                        in_w: w,
                        out_channels: 65,
                        kernel_h: kh,
                        kernel_w: kw,
                        stride,
                        padding_h: pad,
                        out_h: (h + 2 * pad - kh) / stride + 1,
                        out_w: (w + 2 * pad - kw) / stride + 1,
                        padding_w: pad,
                        kernel_w_multiplier: 0,
                        kernel_hw_multiplier: 0,
                        column_width_multiplier: 0,
                        output_spatial_multiplier: 0,
                    };
                    for (subgroup, threads) in [(0, 256), (16, 128), (32, 256)] {
                        let mut module = generate_with_params(
                            ShaderGroup::Conv2dChecked {
                                kernel_h: kh,
                                kernel_w: kw,
                                stride,
                            },
                            MatrixTypes {
                                multiply,
                                ..MatrixTypes::F32_AS_F16
                            },
                            shape,
                            Some(&params),
                            subgroup,
                            TypedMatmulLayout::default(),
                            CheckedTile::BASE,
                        );
                        assert_eq!(
                            module.module.entry_points[0].workgroup_size,
                            [if shape[0] == 8 { threads } else { 128 }, 1, 1],
                        );
                        let mut binding = 0;
                        for (_, global) in module.module.global_variables.iter_mut() {
                            if matches!(
                                global.space,
                                naga::AddressSpace::Uniform | naga::AddressSpace::Storage { .. }
                            ) {
                                global.binding = Some(naga::ResourceBinding { group: 0, binding });
                                binding += 1;
                            }
                        }
                        let info = naga::valid::Validator::new(
                            naga::valid::ValidationFlags::all(),
                            capabilities,
                        )
                        .validate(&module.module)
                        .unwrap();
                        naga::back::spv::write_vec(
                            &module.module,
                            &info,
                            &naga::back::spv::Options::default(),
                            None,
                        )
                        .unwrap();
                    }
                }
            }
        }
    }

    /// Every wide tile meets every native shape, vector and scalar staging,
    /// and direct and staged stores; types and subgroup widths cycle.
    #[test]
    fn wide_convolution_tiles_validate_and_emit_spirv() {
        let capabilities = naga::valid::Capabilities::SHADER_BFLOAT16
            | naga::valid::Capabilities::SHADER_FLOAT16
            | naga::valid::Capabilities::COOPERATIVE_MATRIX
            | naga::valid::Capabilities::SUBGROUP;
        let types = [
            MatrixTypes {
                multiply: DType::F16,
                ..MatrixTypes::F32_AS_F16
            },
            MatrixTypes {
                multiply: DType::BF16,
                ..MatrixTypes::F32_AS_F16
            },
            MatrixTypes {
                output: DType::BF16,
                ..MatrixTypes::BF16
            },
        ];
        let mut index = 0;
        for tile in CheckedTile::WIDE {
            for shape in [[16, 16, 16], [8, 16, 16], [8, 8, 8]] {
                // (in_h, in_w, in_channels, out_channels, kernel, stride, padding)
                for (h, w, ci, co, kernel, stride, pad) in [
                    (16, 16, 64, 64, 1, 1, 0),
                    (14, 14, 64, 96, 1, 1, 0),
                    (11, 13, 3, 65, 3, 2, 1),
                    (16, 16, 32, 48, 3, 1, 1),
                ] {
                    let (out_h, out_w) = (
                        (h + 2 * pad - kernel) / stride + 1,
                        (w + 2 * pad - kernel) / stride + 1,
                    );
                    let params = Params {
                        batch: 2,
                        in_channels: ci,
                        in_h: h,
                        in_w: w,
                        out_channels: co,
                        kernel_h: kernel,
                        kernel_w: kernel,
                        stride,
                        padding_h: pad,
                        out_h,
                        out_w,
                        padding_w: pad,
                        kernel_w_multiplier: crate::divisor::multiplier(kernel),
                        kernel_hw_multiplier: crate::divisor::multiplier(kernel * kernel),
                        column_width_multiplier: crate::divisor::multiplier(out_w),
                        output_spatial_multiplier: crate::divisor::multiplier(out_h * out_w),
                    };
                    let types = types[index % types.len()];
                    let subgroup = [0, 16, 32, 64][index % 4];
                    index += 1;
                    let group = ShaderGroup::Conv2dChecked {
                        kernel_h: kernel,
                        kernel_w: kernel,
                        stride,
                    };
                    let mut module = super::super::generate_checked_conv(
                        group, types, shape, &params, subgroup, tile,
                    );
                    let what = format!("{tile:?} {shape:?} {types:?} {params:?} {subgroup}");
                    assert_eq!(
                        module.module.entry_points[0].workgroup_size,
                        [tile.threads(subgroup), 1, 1],
                        "{what}"
                    );
                    let accumulators = (32 / shape[0]) * (32 / shape[1]);
                    let last = format!("acc{}_{} =", 32 / shape[0] - 1, 32 / shape[1] - 1);
                    assert!(module.source.contains(&last), "{what}");
                    assert_eq!(
                        module.source.matches("coopMultiplyAdd").count() as u32,
                        accumulators,
                        "{what}"
                    );
                    let direct = types.output == DType::F32
                        && co.is_multiple_of(16)
                        && (out_h * out_w).is_multiple_of(16);
                    assert_eq!(
                        !module.source.contains("var<workgroup> result"),
                        direct,
                        "{what}"
                    );
                    assert_eq!(
                        module.source.contains("weight[(ar * k + ak) / 4u]"),
                        params.vectorized_forward(),
                        "{what}"
                    );
                    let mut binding = 0;
                    for (_, global) in module.module.global_variables.iter_mut() {
                        if matches!(
                            global.space,
                            naga::AddressSpace::Uniform | naga::AddressSpace::Storage { .. }
                        ) {
                            global.binding = Some(naga::ResourceBinding { group: 0, binding });
                            binding += 1;
                        }
                    }
                    let info = naga::valid::Validator::new(
                        naga::valid::ValidationFlags::all(),
                        capabilities,
                    )
                    .validate(&module.module)
                    .unwrap_or_else(|error| panic!("{what}: {error:?}"));
                    naga::back::spv::write_vec(
                        &module.module,
                        &info,
                        &naga::back::spv::Options::default(),
                        None,
                    )
                    .unwrap_or_else(|error| panic!("{what}: {error:?}"));
                }
            }
        }
    }

    #[test]
    fn wide_tile_geometry() {
        for tile in CheckedTile::WIDE {
            assert!(tile.valid() && tile.blocks() >= 2, "{tile:?}");
            // One subgroup per block at the hinted width, inside the portable
            // limit and never below one full 64-lane subgroup.
            for subgroup in [0, 16, 32, 64] {
                let threads = tile.threads(subgroup);
                assert!((64..=256).contains(&threads), "{tile:?} {subgroup}");
            }
            assert_eq!(tile.threads(32), 32 * tile.blocks());
        }
        assert!(
            !CheckedTile {
                rows: 128,
                columns: 128
            }
            .valid()
        );
        // Both staged operands, plus an F32 result unless stored directly.
        let tile = CheckedTile {
            rows: 64,
            columns: 128,
        };
        assert_eq!(tile.shared_bytes(DType::F16, true), (64 + 128) * 32 * 2);
        assert_eq!(
            tile.shared_bytes(DType::F16, false),
            (64 + 128) * 32 * 2 + 64 * 128 * 4
        );
    }

    #[test]
    fn typed_dense_uses_shared_staging_and_register_tiles() {
        use super::super::{TypedMatmulLayout, generate_module_coop_typed_with_layout};
        for storage in [DType::F16, DType::BF16] {
            let types = MatrixTypes::reduced(storage);
            let layout = TypedMatmulLayout::for_product(types, false, [64, 64, 64]);
            let module = generate_module_coop_typed_with_layout(
                ShaderGroup::MatMul,
                types,
                [16, 16, 16],
                layout,
            );
            assert!(module.source.contains(&format!(
                "matrix_a: array<vec4<{}>>",
                MatrixTypes::name(storage)
            )));
            assert!(
                module
                    .source
                    .contains("acc11 = coopMultiplyAdd(a1, b1, acc11)")
            );
            assert!(!module.source.contains("var<workgroup> shared_c"));
            assert!(module.source.contains("wgid.z * m * n"));
        }
    }
}
