use super::{MatMulCoopVariant, MatrixTypes, ShaderGroup, ShaderModule, coop_shape};
use crate::DType;

/// A complete workgroup output tile, reduction stage and invocation count.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct WorkgroupMatmulShape {
    pub rows: u32,
    pub columns: u32,
    pub k_stage: u32,
    pub threads: u32,
}

impl WorkgroupMatmulShape {
    /// Bounded search space, including both orientations of rectangular tiles.
    /// Device granularity, memory, buffer extents and precision filter it later.
    pub fn all() -> impl Iterator<Item = Self> {
        [
            (64, 64, 64),
            (32, 64, 64),
            (64, 32, 64),
            (64, 128, 128),
            (128, 64, 128),
            (128, 128, 128),
            (128, 256, 256),
            (256, 128, 256),
        ]
        .into_iter()
        .flat_map(|(rows, columns, threads)| {
            [32, 64].map(|k_stage| Self {
                rows,
                columns,
                k_stage,
                threads,
            })
        })
    }

    pub fn supports(self, caps: &blade_graphics::Capabilities, types: MatrixTypes) -> bool {
        use blade_graphics::CooperativeMatrixOperand;
        let matrix = &caps.cooperative_matrix.workgroup;
        let operand = match types.multiply {
            DType::F16 => CooperativeMatrixOperand::F16,
            DType::BF16 => CooperativeMatrixOperand::Bf16,
            _ => return false,
        };
        types.is_valid()
            && (!types.uses(DType::F16) || caps.shader_float16)
            && (!types.uses(DType::BF16) || caps.shader_bfloat16)
            && [self.rows, self.columns, self.k_stage]
                .iter()
                .all(|&s| s > 0 && s.is_multiple_of(4))
            && matrix.supports(
                operand,
                [self.rows, self.columns, self.k_stage],
                self.threads,
            )
            && self
                .shared_bytes(types)
                .and_then(|bytes| bytes.checked_add(matrix.reserved_shared_memory))
                .is_some_and(|bytes| bytes <= caps.max_compute_shared_memory_size)
    }

    pub fn fits(self, m: u32, n: u32, k: u32) -> bool {
        self.fits_with_row_bounds(m, n, k) && m.is_multiple_of(self.rows)
    }

    pub(crate) fn fits_with_row_bounds(self, m: u32, n: u32, k: u32) -> bool {
        self.rows != 0
            && self.columns != 0
            && self.k_stage != 0
            && self.threads != 0
            && m != 0
            && n != 0
            && k != 0
            && n.is_multiple_of(self.columns)
            && k.is_multiple_of(self.k_stage)
    }

    pub fn shared_bytes(self, types: MatrixTypes) -> Option<u32> {
        self.shared_bytes_with_row_bounds(types, false)
    }

    pub(crate) fn supports_dimensions(
        self,
        caps: &blade_graphics::Capabilities,
        types: MatrixTypes,
        m: u32,
        n: u32,
        k: u32,
    ) -> bool {
        self.fits_with_row_bounds(m, n, k)
            && self.supports(caps, types)
            && self
                .shared_bytes_with_row_bounds(types, !m.is_multiple_of(self.rows))
                .and_then(|bytes| {
                    bytes.checked_add(caps.cooperative_matrix.workgroup.reserved_shared_memory)
                })
                .is_some_and(|bytes| bytes <= caps.max_compute_shared_memory_size)
    }

    fn shared_bytes_with_row_bounds(self, types: MatrixTypes, row_bounds: bool) -> Option<u32> {
        let mut bytes = 0u32;
        for (storage, extent, bounded) in [
            (types.a, self.rows, row_bounds),
            (types.b, self.columns, false),
        ] {
            if storage != types.multiply || bounded {
                bytes = bytes.checked_add(
                    extent
                        .checked_mul(self.k_stage)?
                        .checked_mul(MatrixTypes::bytes(types.multiply) as u32)?,
                )?;
            }
        }
        if types.output != DType::F32 || row_bounds {
            bytes = bytes.checked_add(self.rows.checked_mul(self.columns)?.checked_mul(4)?)?;
        }
        Some(bytes)
    }
}

/// Aligned NN/AT/BT products, including addends and complete K partitions.
/// Matching storage loads directly; only operands needing conversion stage in
/// shared memory. Reduced outputs round once after F32 accumulation and addition.
pub fn generate_workgroup_matmul(
    group: ShaderGroup,
    shape: WorkgroupMatmulShape,
    types: MatrixTypes,
) -> ShaderModule {
    generate_workgroup_matmul_with_row_bounds(group, shape, types, false)
}

/// Row tails use shared A/result tiles so every cooperative operation is uniform,
/// while global reads and stores stay within the logical bindings, including
/// adjacent split-K partitions. Aligned products retain their original shader.
pub(crate) fn generate_workgroup_matmul_with_row_bounds(
    group: ShaderGroup,
    shape: WorkgroupMatmulShape,
    types: MatrixTypes,
    row_bounds: bool,
) -> ShaderModule {
    types.validate();
    assert!(matches!(types.multiply, DType::F16 | DType::BF16));
    let (addend, variant) = coop_shape(group).expect("workgroup matmul group");
    let WorkgroupMatmulShape {
        rows: m,
        columns: n,
        k_stage: k,
        threads,
    } = shape;
    assert!([m, n, k].iter().all(|&s| s > 0 && s.is_multiple_of(4)) && threads > 0);
    let multiply = MatrixTypes::name(types.multiply);
    let mut source = String::from("enable wgpu_cooperative_matrix;\n");
    if types.uses(DType::F16) {
        source += "enable f16;\n";
    }
    if types.uses(DType::BF16) {
        source += "enable wgpu_bfloat16;\n";
        source += include_str!("../shaders/bf16_round.wgsl");
    }
    // K stages are four-aligned. NN/BT A can keep vector loads even when
    // its matching F16 storage still needs a guarded shared-memory tile.
    let vector_a = row_bounds
        && variant != MatMulCoopVariant::AT
        && types.a == DType::F16
        && types.multiply == DType::F16;
    source +=
        "struct Params { m: u32, n: u32, k: u32, splits: u32 }\nvar<uniform> params: Params;\n";
    for (name, storage, extent) in [("a", types.a, m), ("b", types.b, n)] {
        let scalar = MatrixTypes::name(storage);
        let storage_name = if storage != types.multiply || (name == "a" && vector_a) {
            format!("vec4<{scalar}>")
        } else {
            scalar.to_owned()
        };
        source += &format!("var<storage> matrix_{name}: array<{storage_name}>;\n");
        if storage != types.multiply || (name == "a" && row_bounds) {
            source += &format!(
                "var<workgroup> s{name}: array<{multiply}, {}>;\n",
                extent * k
            );
        }
    }
    source += &format!(
        "var<storage, read_write> matrix_c: array<{}>;\n",
        MatrixTypes::name(types.output)
    );
    if addend {
        source += "var<storage> src: array<f32>;\n";
    }
    if types.output != DType::F32 || row_bounds {
        source += &format!("var<workgroup> result: array<f32, {}>;\n", m * n);
    }
    source += &format!(
        "@compute @workgroup_size({threads})\nfn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lid: u32) {{\nlet m = params.m;\nlet n = params.n;\nlet k = params.k;\nlet row = group.x * {m}u;\nlet col = group.y * {n}u;\nlet blocks = k / {k}u;\nlet base = blocks / params.splits;\nlet extra = blocks % params.splits;\nlet first = (group.z * base + min(group.z, extra)) * {k}u;\nlet end = first + (base + select(0u, 1u, group.z < extra)) * {k}u;\nvar acc = coop_mat{n}x{m}<f32,C,workgroup>();\nfor (var t = first; t < end; t += {k}u) {{\n"
    );
    for (a, storage) in [(true, types.a), (false, types.b)] {
        let transpose = if a {
            variant == MatMulCoopVariant::AT
        } else {
            variant == MatMulCoopVariant::BT
        };
        let (name, rows, cols, role) = if a {
            ("a", m, k, "A")
        } else {
            ("b", k, n, "B")
        };
        if storage != types.multiply || (a && vector_a) {
            let width = if transpose { rows } else { cols };
            let index = match (a, transpose) {
                (true, false) => "(row + r) * k + t + c",
                (true, true) => "(t + r) * m + row + c",
                (false, false) => "(t + r) * n + col + c",
                (false, true) => "(col + r) * k + t + c",
            };
            let cast = if types.multiply == DType::BF16 {
                "round_bf16"
            } else {
                multiply
            };
            source += &format!(
                "for (var i = lid; i < {}u; i += {threads}u) {{\nlet r = i / {}u;\nlet c = (i % {}u) * 4u;\n",
                rows * cols / 4,
                width / 4,
                width / 4
            );
            if a && row_bounds {
                let scalar = MatrixTypes::name(storage);
                source += &format!("var v = vec4<{scalar}>(0.0);\n");
                if transpose {
                    // An odd M stride can straddle two vec4s. Gather valid
                    // lanes instead of rounding the address or reading a tail.
                    for (lane, field) in ["x", "y", "z", "w"].iter().enumerate() {
                        source += &format!(
                            "if (row + c + {lane}u < m) {{ v.{field} = matrix_a[((({index}) + {lane}u) / 4u)][((({index}) + {lane}u) % 4u)]; }}\n"
                        );
                    }
                } else {
                    source += &format!("if (row + r < m) {{ v = matrix_a[({index}) / 4u]; }}\n");
                }
            } else {
                source += &format!("let v = matrix_{name}[({index}) / 4u];\n");
            }
            for (lane, field) in ["x", "y", "z", "w"].iter().enumerate() {
                let index = if transpose {
                    format!("(c + {lane}u) * {cols}u + r")
                } else {
                    format!("r * {cols}u + c + {lane}u")
                };
                source += &format!("s{name}[{index}] = {cast}(v.{field});\n");
            }
            source += "}\nworkgroupBarrier();\n";
            source += &format!(
                "let {name} = coopLoadT<coop_mat{cols}x{rows}<{multiply},{role},workgroup>>(&s{name}[0], {cols}u);\n"
            );
        } else if a && row_bounds {
            let width = if transpose { rows } else { cols };
            let (index, stage, axis) = if transpose {
                ("(t + r) * m + row + c", format!("c * {cols}u + r"), "c")
            } else {
                ("(row + r) * k + t + c", format!("r * {cols}u + c"), "r")
            };
            source += &format!(
                "for (var i = lid; i < {}u; i += {threads}u) {{\nlet r = i / {width}u;\nlet c = i % {width}u;\nvar value = {multiply}(0.0);\nif (row + {axis} < m) {{ value = matrix_a[{index}]; }}\nsa[{stage}] = value;\n}}\nworkgroupBarrier();\nlet a = coopLoadT<coop_mat{cols}x{rows}<{multiply},A,workgroup>>(&sa[0], {cols}u);\n",
                rows * cols
            );
        } else {
            let load = if transpose { "coopLoad" } else { "coopLoadT" };
            let (index, stride) = match (a, transpose) {
                (true, false) => ("row * k + t", "k"),
                (true, true) => ("t * m + row", "m"),
                (false, false) => ("t * n + col", "n"),
                (false, true) => ("col * k + t", "k"),
            };
            source += &format!(
                "let {name} = {load}<coop_mat{cols}x{rows}<{multiply},{role},workgroup>>(&matrix_{name}[{index}], {stride});\n"
            );
        }
    }
    source += "acc = coopMultiplyAdd(a,b,acc);\n";
    if row_bounds || types.a != types.multiply || types.b != types.multiply {
        source += "workgroupBarrier();\n";
    }
    source += "}\n";
    if addend && !row_bounds {
        source += &format!(
            "acc = acc + coopLoadT<coop_mat{n}x{m}<f32,C,workgroup>>(&src[row * n + col], n);\n"
        );
    }
    if row_bounds {
        let index = format!("(group.z * m + row + i / {n}u) * n + col + i % {n}u");
        let value = if addend {
            format!("result[i] + src[(row + i / {n}u) * n + col + i % {n}u]")
        } else {
            "result[i]".to_owned()
        };
        let value = match types.output {
            DType::F32 => value,
            DType::BF16 => format!("round_bf16({value})"),
            dtype => format!("{}({value})", MatrixTypes::name(dtype)),
        };
        source += &format!(
            "coopStoreT(acc, &result[0], {n}u);\nworkgroupBarrier();\nfor (var i = lid; i < {}u; i += {threads}u) {{\nif (row + i / {n}u < m) {{ matrix_c[{index}] = {value}; }}\n}}\n",
            m * n
        );
    } else if types.output == DType::F32 {
        source += "coopStoreT(acc, &matrix_c[(group.z * m + row) * n + col], n);\n";
    } else {
        let cast = if types.output == DType::BF16 {
            "round_bf16"
        } else {
            MatrixTypes::name(types.output)
        };
        source += &format!(
            "coopStoreT(acc, &result[0], {n}u);\nworkgroupBarrier();\nfor (var i = lid; i < {}u; i += {threads}u) {{\nmatrix_c[(row + i / {n}u) * n + col + i % {n}u] = {cast}(result[i]);\n}}\n",
            m * n
        );
    }
    source += "}\n";
    ShaderModule::with_hint(source, "workgroup-matmul")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_aligned_generator_keeps_its_complete_tile_contract() {
        let shape = WorkgroupMatmulShape {
            rows: 64,
            columns: 64,
            k_stage: 32,
            threads: 64,
        };
        assert!(shape.fits(1536, 384, 384));
        for m in [1, 65, 129, 1500, 1501] {
            assert!(!shape.fits(m, 384, 384));
            assert!(shape.fits_with_row_bounds(m, 384, 384));
        }
    }

    #[test]
    fn rectangular_typed_modules_validate_and_emit_spirv() {
        let capabilities = naga::valid::Capabilities::SHADER_BFLOAT16
            | naga::valid::Capabilities::SHADER_FLOAT16
            | naga::valid::Capabilities::COOPERATIVE_MATRIX
            | naga::valid::Capabilities::COOPERATIVE_MATRIX_WORKGROUP;
        for shape in WorkgroupMatmulShape::all() {
            for group in [
                ShaderGroup::MatMul,
                ShaderGroup::MatMulAT,
                ShaderGroup::MatMulBT,
                ShaderGroup::MatMulAdd,
                ShaderGroup::MatMulATAdd,
                ShaderGroup::MatMulBTAdd,
            ] {
                for multiply in [DType::F16, DType::BF16] {
                    for (a, b) in [
                        (multiply, multiply),
                        (DType::F32, multiply),
                        (multiply, DType::F32),
                        (DType::F32, DType::F32),
                    ] {
                        for output in [DType::F32, DType::F16, DType::BF16] {
                            let types = MatrixTypes {
                                a,
                                b,
                                multiply,
                                output,
                            };
                            for row_bounds in [false, true] {
                                let mut module = generate_workgroup_matmul_with_row_bounds(
                                    group, shape, types, row_bounds,
                                );
                                let mut binding = 0;
                                for (_, global) in module.module.global_variables.iter_mut() {
                                    if matches!(
                                        global.space,
                                        naga::AddressSpace::Uniform
                                            | naga::AddressSpace::Storage { .. }
                                    ) {
                                        global.binding =
                                            Some(naga::ResourceBinding { group: 0, binding });
                                        binding += 1;
                                    }
                                }
                                let info = naga::valid::Validator::new(
                                    naga::valid::ValidationFlags::all(),
                                    capabilities,
                                )
                                .validate(&module.module)
                                .unwrap_or_else(|error| {
                                    panic!("{group:?} {types:?} {shape:?}: {error:?}")
                                });
                                let words = naga::back::spv::write_vec(
                                    &module.module,
                                    &info,
                                    &naga::back::spv::Options {
                                        lang_version: (1, 6),
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
}
