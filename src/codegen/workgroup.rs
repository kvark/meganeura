use super::{MatMulCoopVariant, MatrixTypes, ShaderGroup, ShaderModule, coop_shape};
use crate::DType;

/// A complete workgroup output tile, reduction stage and invocation count.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct WorkgroupMatmulShape {
    pub rows: u32,
    pub columns: u32,
    pub k_stage: u32,
    pub threads: u32,
    /// Read both operands as vectors into padded shared-memory tiles.
    #[serde(default)]
    pub stage_operands: bool,
    /// Subgroup-scope tiles: the rows of a grid of `threads / 32` subgroups,
    /// each holding 16x16 accumulators of the workgroup tile. Zero selects
    /// one workgroup-scope matrix.
    #[serde(default)]
    pub subgroup_rows: u32,
    /// Subgroup tiles only: hold the next K stage's operand vectors in
    /// registers while this stage multiplies, so global latency overlaps the
    /// products. It costs registers, so tiles with one accumulator column and
    /// more subgroups can lose; the menu offers each tile both ways.
    #[serde(default)]
    pub prefetch: bool,
}

/// Subgroup-tiled shapes: `(subgroup rows, subgroup columns, tile rows,
/// tile columns, k_stage)`, tiles counted in 16x16 accumulators per subgroup.
/// `gemm_lab libsweep` timed 168 tilings, with up to four K partitions each,
/// in F16 on the RTX 5070 over the accelerated training products of
/// Whisper-tiny and SmolLM2. These eight come within 5% of the best tiling
/// on each of those 17 products; the 64x64 tile of four subgroups is best
/// for most. Each runs with and without register prefetch: over 15 cold
/// products of the five models' training programs, choosing both ways per
/// product takes the best F16 tile to 0.968 and the best BF16 one to 0.956
/// of the plain menu (SmolVLA's 4096-deep product 0.84, Whisper's 0.91–0.94),
/// while prefetch on every tile loses up to 9% on Whisper's weight gradients.
const SUBGROUP_TILES: [(u32, u32, u32, u32, u32); 8] = [
    (2, 2, 2, 2, 32),
    (1, 4, 4, 1, 16),
    (2, 2, 2, 1, 32),
    (2, 1, 2, 4, 64),
    (4, 2, 2, 1, 32),
    (2, 4, 2, 1, 64),
    (1, 2, 2, 2, 64),
    (2, 1, 1, 1, 64),
];

/// Default subgroup width when no device capability overrides it.
pub(crate) const SUBGROUP_WIDTH: u32 = 32;

/// Subgroup widths supported by the subgroup-tiled cooperative kernels
/// (Intel Xe2 16/32, NVIDIA/Metal 32, AMD/Adreno 64).
pub(crate) fn supported_subgroup_width(subgroup: u32) -> bool {
    matches!(subgroup, 16 | 32 | 64)
}

/// Native cooperative matrix shape and subgroup width for `multiply` on `caps`.
pub(crate) fn native_subgroup_shape(
    caps: &blade_graphics::Capabilities,
    multiply: DType,
) -> Option<([u32; 3], u32)> {
    if !supported_subgroup_width(caps.subgroup_size) {
        return None;
    }
    let shapes = match multiply {
        DType::F16 => &caps.cooperative_matrix.f16_f32_shapes,
        DType::BF16 => &caps.cooperative_matrix.bf16_f32_shapes,
        _ => return None,
    };
    let native = super::checked_cooperative_shape(shapes)?;
    Some((native, caps.subgroup_size))
}

impl WorkgroupMatmulShape {
    /// Bounded search space for the default 16×16×16 / 32-lane target.
    pub fn all() -> impl Iterator<Item = Self> {
        Self::all_for([16, 16, 16], SUBGROUP_WIDTH)
    }

    /// Bounded search space, including both orientations of rectangular tiles
    /// and subgroup-tiled shapes sized to `(native, subgroup)`.
    /// Device granularity, memory, buffer extents and precision filter it later.
    pub fn all_for(native: [u32; 3], subgroup: u32) -> impl Iterator<Item = Self> {
        let [nm, nn, _] = native;
        let subgroup_ok = super::checked_cooperative_shape(&[native]).is_some()
            && supported_subgroup_width(subgroup);
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
            [32, 64].into_iter().flat_map(move |k_stage| {
                [false, true]
                    .into_iter()
                    // Keep staging to one small and one tall tile. Expanding
                    // every tile spends the model search budget on compilation.
                    .filter(move |&stage| !stage || (columns == 64 && matches!(rows, 32 | 128)))
                    .map(move |stage_operands| Self {
                        rows,
                        columns,
                        k_stage,
                        threads,
                        stage_operands,
                        subgroup_rows: 0,
                        prefetch: false,
                    })
            })
        })
        .chain(
            SUBGROUP_TILES
                .into_iter()
                .filter(move |&(wr, wc, ..)| subgroup_ok && wr * wc * subgroup <= 256)
                .flat_map(move |(wr, wc, tr, tc, k_stage)| {
                    [false, true].map(move |prefetch| Self {
                        rows: wr * tr * nm,
                        columns: wc * tc * nn,
                        k_stage,
                        threads: wr * wc * subgroup,
                        stage_operands: false,
                        subgroup_rows: wr,
                        prefetch,
                    })
                }),
        )
    }

    /// `(subgroup rows, subgroup columns, tile rows, tile columns)` of a
    /// subgroup-tiled shape on a 16×16×16 / 32-lane device.
    pub fn subgroup_grid(self) -> Option<(u32, u32, u32, u32)> {
        self.subgroup_grid_for([16, 16, 16], SUBGROUP_WIDTH)
    }

    /// `(subgroup rows, subgroup columns, tile rows, tile columns)` of a
    /// subgroup-tiled shape, in subgroups and `native` accumulators.
    pub(crate) fn subgroup_grid_for(
        self,
        native: [u32; 3],
        subgroup: u32,
    ) -> Option<(u32, u32, u32, u32)> {
        let [nm, nn, _] = native;
        let wr = self.subgroup_rows;
        if wr == 0
            || !supported_subgroup_width(subgroup)
            || nm == 0
            || nn == 0
            || self.threads == 0
            || !self.threads.is_multiple_of(subgroup * wr)
        {
            return None;
        }
        let wc = self.threads / subgroup / wr;
        if wc == 0 || !self.rows.is_multiple_of(nm * wr) || !self.columns.is_multiple_of(nn * wc) {
            return None;
        }
        let tr = self.rows / nm / wr;
        let tc = self.columns / nn / wc;
        (tr > 0 && tc > 0).then_some((wr, wc, tr, tc))
    }

    /// Shared bytes of a subgroup-tiled shape: padded operand tiles in the
    /// multiplication type (doubled for double-buffered `prefetch`), and one
    /// `nm × nn` f32 tile per subgroup when stores pass through shared memory.
    fn subgroup_shared_bytes_for(
        self,
        types: MatrixTypes,
        native: [u32; 3],
        subgroup: u32,
        row_bounds: bool,
    ) -> Option<u32> {
        let (wr, wc, ..) = self.subgroup_grid_for(native, subgroup)?;
        let element = MatrixTypes::bytes(types.multiply) as u32;
        let stages = if self.prefetch { 2u32 } else { 1u32 };
        let operands = self
            .rows
            .checked_add(self.columns)?
            .checked_mul(self.k_stage.checked_add(8)?)?
            .checked_add(16 * self.k_stage)?
            .checked_mul(element)?
            .checked_mul(stages)?;
        let scratch = if row_bounds || types.output != DType::F32 {
            wr.checked_mul(wc)?
                .checked_mul(native[0])?
                .checked_mul(native[1])?
                .checked_mul(4)?
        } else {
            0
        };
        operands.checked_add(scratch)
    }

    pub fn supports(
        self,
        caps: &blade_graphics::Capabilities,
        types: MatrixTypes,
        m: u32,
        n: u32,
        k: u32,
    ) -> bool {
        self.supports_layout(caps, types, None, m, n, k)
    }

    /// [`supports`](Self::supports) for one product layout of `group`:
    /// subgroup tiles then need whole vectors only along each operand's
    /// contiguous axis, not along both M and K.
    pub fn supports_layout(
        self,
        caps: &blade_graphics::Capabilities,
        types: MatrixTypes,
        group: Option<ShaderGroup>,
        m: u32,
        n: u32,
        k: u32,
    ) -> bool {
        use blade_graphics::CooperativeMatrixOperand;
        if self.prefetch && self.subgroup_rows == 0 {
            return false;
        }
        if self.subgroup_rows != 0 {
            let Some((native, subgroup)) = native_subgroup_shape(caps, types.multiply) else {
                return false;
            };
            // Storage is F32 or the multiplication type itself.
            return self.fits_layout_for(group, native, subgroup, m, n, k)
                && !self.stage_operands
                && types.is_valid()
                && [types.a, types.b]
                    .iter()
                    .all(|&storage| storage == DType::F32 || storage == types.multiply)
                && (!types.uses(DType::F16) || caps.shader_float16)
                && (!types.uses(DType::BF16) || caps.shader_bfloat16)
                && matches!(self.k_stage, 16 | 32 | 64)
                && self.k_stage.is_multiple_of(native[2])
                && self.threads <= 256
                && self
                    .subgroup_shared_bytes_for(types, native, subgroup, self.ragged_rows(m))
                    .is_some_and(|bytes| bytes <= caps.max_compute_shared_memory_size);
        }
        let matrix = &caps.cooperative_matrix.workgroup;
        let operand = match types.multiply {
            DType::F16 => CooperativeMatrixOperand::F16,
            DType::BF16 => CooperativeMatrixOperand::Bf16,
            _ => return false,
        };
        self.fits(m, n, k)
            && types.is_valid()
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
                .shared_bytes(types, self.ragged_rows(m))
                .and_then(|bytes| bytes.checked_add(matrix.reserved_shared_memory))
                .is_some_and(|bytes| bytes <= caps.max_compute_shared_memory_size)
    }

    pub fn fits(self, m: u32, n: u32, k: u32) -> bool {
        self.fits_layout(None, m, n, k)
    }

    /// [`fits`](Self::fits) for the layout of `group`, when known, across any
    /// supported native cooperative matrix shape and subgroup width.
    pub fn fits_layout(self, group: Option<ShaderGroup>, m: u32, n: u32, k: u32) -> bool {
        if self.subgroup_rows == 0 {
            return self.fits_layout_for(group, [16, 16, 16], SUBGROUP_WIDTH, m, n, k);
        }
        [[16, 16, 16], [8, 16, 16], [8, 8, 8]]
            .into_iter()
            .any(|native| {
                [16, 32, 64]
                    .into_iter()
                    .any(|subgroup| self.fits_layout_for(group, native, subgroup, m, n, k))
            })
    }

    /// [`fits_layout`](Self::fits_layout) for a specific `(native, subgroup)`.
    pub(crate) fn fits_layout_for(
        self,
        group: Option<ShaderGroup>,
        native: [u32; 3],
        subgroup: u32,
        m: u32,
        n: u32,
        k: u32,
    ) -> bool {
        let reduction = if self.subgroup_rows != 0 {
            // Guarded vector loads: K may end inside a stage. Every
            // operand's contiguous axis holds whole vectors: K for A and for
            // a transposed B, M for a transposed A; N holds whole tiles.
            // Without a layout, both K and M must.
            let contiguous = match group.and_then(coop_shape).map(|(_, variant)| variant) {
                Some(MatMulCoopVariant::AT) => m.is_multiple_of(4),
                Some(_) => k.is_multiple_of(4),
                None => k.is_multiple_of(4) && m.is_multiple_of(4),
            };
            self.subgroup_grid_for(native, subgroup).is_some() && contiguous
        } else {
            k.is_multiple_of(self.k_stage)
        };
        self.rows != 0
            && self.columns != 0
            && self.k_stage != 0
            && self.threads != 0
            && m != 0
            && n != 0
            && k != 0
            && n.is_multiple_of(self.columns)
            && reduction
    }

    /// K stages a partition can own.
    pub fn k_blocks(self, k: u32) -> u32 {
        if self.subgroup_rows != 0 {
            k.div_ceil(self.k_stage.max(1))
        } else {
            k / self.k_stage.max(1)
        }
    }

    pub(crate) fn ragged_rows(self, m: u32) -> bool {
        !m.is_multiple_of(self.rows)
    }

    fn stages(self, storage: DType, multiply: DType) -> bool {
        self.stage_operands || storage != multiply
    }

    fn pitch(self, cols: u32) -> Option<u32> {
        // Eight half elements rotate the shared-memory bank assignment.
        cols.checked_add(if self.stage_operands { 8 } else { 0 })
    }

    pub fn shared_bytes(self, types: MatrixTypes, row_bounds: bool) -> Option<u32> {
        self.shared_bytes_for(types, [16, 16, 16], SUBGROUP_WIDTH, row_bounds)
    }

    pub(crate) fn shared_bytes_for(
        self,
        types: MatrixTypes,
        native: [u32; 3],
        subgroup: u32,
        row_bounds: bool,
    ) -> Option<u32> {
        if self.subgroup_rows != 0 {
            return self.subgroup_shared_bytes_for(types, native, subgroup, row_bounds);
        }
        let mut bytes = 0u32;
        for (storage, rows, cols, bounded) in [
            (types.a, self.rows, self.k_stage, row_bounds),
            (types.b, self.k_stage, self.columns, false),
        ] {
            if self.stages(storage, types.multiply) || bounded {
                bytes = bytes.checked_add(
                    rows.checked_mul(self.pitch(cols)?)?
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

/// NN/AT/BT products, with optional guarded final rows. Cooperative operations
/// stay uniform; shared A/result tiles keep ragged global accesses in bounds.
pub fn generate_workgroup_matmul(
    group: ShaderGroup,
    shape: WorkgroupMatmulShape,
    types: MatrixTypes,
    row_bounds: bool,
) -> ShaderModule {
    generate_workgroup_matmul_for_device(
        group,
        shape,
        types,
        [16, 16, 16],
        SUBGROUP_WIDTH,
        row_bounds,
        None,
    )
}

/// [`generate_workgroup_matmul`] specialized to a device's `(native, subgroup)`
/// and optional unary [`crate::compile::MatMulEpilogue`].
pub(crate) fn generate_workgroup_matmul_for_device(
    group: ShaderGroup,
    shape: WorkgroupMatmulShape,
    types: MatrixTypes,
    native: [u32; 3],
    subgroup: u32,
    row_bounds: bool,
    epilogue: Option<&crate::compile::MatMulEpilogue>,
) -> ShaderModule {
    types.validate();
    assert!(matches!(types.multiply, DType::F16 | DType::BF16));
    if shape.subgroup_rows != 0 {
        return generate_subgroup_matmul(
            group, shape, types, native, subgroup, row_bounds, epilogue,
        );
    }
    assert!(
        epilogue.is_none(),
        "workgroup-scope matmul does not stage fused epilogues"
    );
    let (addend, variant) = coop_shape(group).expect("workgroup matmul group");
    let WorkgroupMatmulShape {
        rows: m,
        columns: n,
        k_stage: k,
        threads,
        ..
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
    source +=
        "struct Params { m: u32, n: u32, k: u32, splits: u32 }\nvar<uniform> params: Params;\n";
    for (name, storage, rows, cols) in [("a", types.a, m, k), ("b", types.b, k, n)] {
        let scalar = MatrixTypes::name(storage);
        let storage_name = if shape.stages(storage, types.multiply) {
            format!("vec4<{scalar}>")
        } else {
            scalar.to_owned()
        };
        source += &format!("var<storage> matrix_{name}: array<{storage_name}>;\n");
        if shape.stages(storage, types.multiply) || (name == "a" && row_bounds) {
            source += &format!(
                "var<workgroup> s{name}: array<{multiply}, {}>;\n",
                rows * shape.pitch(cols).expect("workgroup operand pitch overflow")
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
        let pitch = shape.pitch(cols).expect("workgroup operand pitch overflow");
        if shape.stages(storage, types.multiply) {
            let width = if transpose { rows } else { cols };
            let index = match (a, transpose) {
                (true, false) => "(row + r) * k + t + c",
                (true, true) => "(t + r) * m + row + c",
                (false, false) => "(t + r) * n + col + c",
                (false, true) => "(col + r) * k + t + c",
            };
            let cast = if storage == types.multiply {
                ""
            } else if types.multiply == DType::BF16 {
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
                    format!("(c + {lane}u) * {pitch}u + r")
                } else {
                    format!("r * {pitch}u + c + {lane}u")
                };
                source += &format!("s{name}[{index}] = {cast}(v.{field});\n");
            }
            source += "}\nworkgroupBarrier();\n";
            source += &format!(
                "let {name} = coopLoadT<coop_mat{cols}x{rows}<{multiply},{role},workgroup>>(&s{name}[0], {pitch}u);\n"
            );
        } else if a && row_bounds {
            let width = if transpose { rows } else { cols };
            let (index, stage, axis) = if transpose {
                ("(t + r) * m + row + c", format!("c * {pitch}u + r"), "c")
            } else {
                ("(row + r) * k + t + c", format!("r * {pitch}u + c"), "r")
            };
            source += &format!(
                "for (var i = lid; i < {}u; i += {threads}u) {{\nlet r = i / {width}u;\nlet c = i % {width}u;\nvar value = {multiply}(0.0);\nif (row + {axis} < m) {{ value = matrix_a[{index}]; }}\nsa[{stage}] = value;\n}}\nworkgroupBarrier();\nlet a = coopLoadT<coop_mat{cols}x{rows}<{multiply},A,workgroup>>(&sa[0], {pitch}u);\n",
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
    if row_bounds || shape.stages(types.a, types.multiply) || shape.stages(types.b, types.multiply)
    {
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
    ShaderModule::with_hint(source, "workgroup-matmul").with_subgroup(0, [threads, 1, 1])
}

fn conv_split_images(split: bool, batch: u32) -> Images {
    let batch = batch.max(1);
    if split {
        Images {
            a: "0u".into(),
            b: "0u".into(),
            c: format!("(chunk * {batch}u + image) * m * n"),
            batch: Some(batch),
        }
    } else {
        Images {
            a: "0u".into(),
            b: "0u".into(),
            c: "image * m * n".into(),
            batch: None,
        }
    }
}

/// Subgroup-tiled NN/AT/BT products: `threads / subgroup` subgroups in a
/// `subgroup_rows` grid split the workgroup tile, each accumulating `native`
/// F32 matrices over operands staged through shared memory in the
/// multiplication type. Vector loads guard every edge, so M may be ragged
/// and K may end inside a stage. Partitions own whole K stages and write
/// compact M*N slices; ragged rows, fused epilogues and reduced outputs leave
/// through a per-subgroup shared tile.
fn generate_subgroup_matmul(
    group: ShaderGroup,
    shape: WorkgroupMatmulShape,
    types: MatrixTypes,
    native: [u32; 3],
    subgroup: u32,
    row_bounds: bool,
    epilogue: Option<&crate::compile::MatMulEpilogue>,
) -> ShaderModule {
    let (addend, variant) = coop_shape(group).expect("workgroup matmul group");
    let mut header = String::from(
        "struct Params { m: u32, n: u32, k: u32, splits: u32 }\nvar<uniform> params: Params;\n",
    );
    for (name, storage) in [("a", types.a), ("b", types.b)] {
        header += &format!(
            "var<storage> matrix_{name}: array<vec4<{}>>;\n",
            MatrixTypes::name(storage)
        );
    }
    header += &format!(
        "var<storage, read_write> matrix_c: array<{}>;\n",
        MatrixTypes::name(types.output)
    );
    if addend {
        header += "var<storage> src: array<f32>;\n";
    }
    subgroup_kernel(
        shape,
        types,
        native,
        subgroup,
        &SubgroupLayout {
            variant,
            header,
            dimensions: "let m = params.m;\nlet n = params.n;\nlet k = params.k;".into(),
            a: "matrix_a",
            b: "matrix_b",
            c: "matrix_c",
            addend: addend.then_some("src"),
            epilogue,
            images: None,
            row_bounds,
            column_bounds: false,
            gather: [None, None],
            scatter: None,
        },
    )
}

/// A pointwise (1x1, stride 1, unpadded) NCHW convolution as one subgroup-
/// tiled product per image: the forward `W[Co, Ci] · X[Ci, HW]` or the input
/// gradient `Wᵀ · dY[Co, HW]`. Dimensions are constants; workgroup Z is the
/// image.
pub(crate) fn generate_subgroup_pointwise_conv(
    backward: bool,
    shape: WorkgroupMatmulShape,
    types: MatrixTypes,
    native: [u32; 3],
    subgroup: u32,
    params: &crate::kernels::conv::Params,
) -> ShaderModule {
    assert!(params.vectorized_forward());
    assert_eq!(
        (types.a, types.b, types.output),
        (DType::F32, DType::F32, DType::F32)
    );
    let spatial = params.in_h * params.in_w;
    let (m, k, b) = if backward {
        (params.in_channels, params.out_channels, "grad_out")
    } else {
        (params.out_channels, params.in_channels, "src")
    };
    let header = format!(
        "var<storage> {b}: array<vec4<f32>>;\nvar<storage> weight: array<vec4<f32>>;\nvar<storage, read_write> dst: array<f32>;\n"
    );
    subgroup_kernel(
        shape,
        types,
        native,
        subgroup,
        &SubgroupLayout {
            variant: if backward {
                MatMulCoopVariant::AT
            } else {
                MatMulCoopVariant::Normal
            },
            header,
            dimensions: format!("let m = {m}u;\nlet n = {spatial}u;\nlet k = {k}u;"),
            a: "weight",
            b: if backward { "grad_out" } else { "src" },
            c: "dst",
            addend: None,
            epilogue: None,
            images: Some(Images {
                a: "0u".into(),
                b: "image * k * n".into(),
                c: "image * m * n".into(),
                batch: None,
            }),
            row_bounds: !m.is_multiple_of(shape.rows),
            column_bounds: !spatial.is_multiple_of(shape.columns),
            gather: [None, None],
            scatter: None,
        },
    )
}

/// The weight gradient of a pointwise convolution, `Σ dY[Co, HW] · X[Ci, HW]ᵀ`
/// over images, as per-image BT products split into K chunks. Workgroup Z
/// covers `batch × chunks` partitions; each writes its own `Co × Ci` partial
/// for a following SumRows, or the gradient itself when there is one.
pub(crate) fn generate_subgroup_pointwise_weight_grad(
    shape: WorkgroupMatmulShape,
    types: MatrixTypes,
    native: [u32; 3],
    subgroup: u32,
    params: &crate::kernels::conv::Params,
) -> ShaderModule {
    assert!(params.vectorized_forward());
    assert_eq!(
        (types.a, types.b, types.output),
        (DType::F32, DType::F32, DType::F32)
    );
    let (m, n, k) = (
        params.out_channels,
        params.in_channels,
        params.in_h * params.in_w,
    );
    subgroup_kernel(
        shape,
        types,
        native,
        subgroup,
        &SubgroupLayout {
            variant: MatMulCoopVariant::BT,
            header: "var<storage> grad_out: array<vec4<f32>>;\nvar<storage> src: array<vec4<f32>>;\nvar<storage, read_write> dst: array<f32>;\n".into(),
            dimensions: format!("let m = {m}u;\nlet n = {n}u;\nlet k = {k}u;"),
            a: "grad_out",
            b: "src",
            c: "dst",
            addend: None,
            epilogue: None,
            images: Some(Images {
                a: "image * m * k".into(),
                b: "image * n * k".into(),
                c: "group.z * m * n".into(),
                batch: Some(params.batch.max(1)),
            }),
            row_bounds: !m.is_multiple_of(shape.rows),
            column_bounds: !n.is_multiple_of(shape.columns),
            gather: [None, None],
            scatter: None,
        },
    )
}

/// The weight gradient of any convolution, `Σ dY[Co, N·oHW] · col(X)[Ci·kH·kW, N·oHW]ᵀ`,
/// as one BT product over the whole batch: K runs over images and output
/// positions, split into chunks over workgroup Z, each writing a
/// `Co × Ci·kH·kW` partial (or the gradient itself when there is one). Both
/// operands gather their elements: dY because `oH·oW` need not hold whole
/// vectors, and the im2col of X because stride and padding break its rows.
/// Unlike the pointwise kernel's per-image products, a large gradient needs
/// no partials at all.
pub(crate) fn generate_subgroup_gathered_weight_grad(
    shape: WorkgroupMatmulShape,
    types: MatrixTypes,
    native: [u32; 3],
    subgroup: u32,
    params: &crate::kernels::conv::Params,
) -> ShaderModule {
    assert_eq!(
        (types.a, types.b, types.output),
        (DType::F32, DType::F32, DType::F32)
    );
    let p = params;
    let khw = p.kernel_h * p.kernel_w;
    let ohw = p.out_h * p.out_w;
    let (m, n, k) = (p.out_channels, p.in_channels * khw, p.batch.max(1) * ohw);
    // Whole dY vectors per image: four consecutive K load as one.
    let vector_dy = ohw.is_multiple_of(4);
    let (dy_type, dy_at) = if vector_dy {
        ("vec4<f32>", "grad_out[i / 4u][i % 4u]")
    } else {
        ("f32", "grad_out[i]")
    };
    let gather_dy4 = if vector_dy {
        format!(
            "// Four consecutive kk of one dY row: one image's aligned vector.
fn gather_dy4(_chunk: u32, co: u32, kk0: u32, end: u32) -> vec4<f32> {{
    if co >= {m}u || kk0 >= min(end, {k}u) {{ return vec4<f32>(0.0); }}
    return grad_out[(((kk0 / {ohw}u) * {m}u + co) * {ohw}u + kk0 % {ohw}u) / 4u];
}}
"
        )
    } else {
        format!(
            "// Four consecutive kk of one dY row: the image and position step along.
fn gather_dy4(_chunk: u32, co: u32, kk0: u32, end: u32) -> vec4<f32> {{
    var out = vec4<f32>(0.0);
    if co >= {m}u {{ return out; }}
    var image = kk0 / {ohw}u;
    var pos = kk0 % {ohw}u;
    for (var e = 0u; e < 4u; e++) {{
        if kk0 + e < min(end, {k}u) {{ out[e] = grad_out[(image * {m}u + co) * {ohw}u + pos]; }}
        pos += 1u;
        if pos == {ohw}u {{ pos = 0u; image += 1u; }}
    }}
    return out;
}}
"
        )
    };
    let header = format!(
        "var<storage> grad_out: array<{dy_type}>;
var<storage> src: array<f32>;
var<storage, read_write> dst: array<f32>;
// dY[image, co, pos] with kk = image·oHW + pos.
fn gather_dy(_chunk: u32, co: u32, kk: u32) -> f32 {{
    if co >= {m}u || kk >= {k}u {{ return 0.0; }}
    let i = ((kk / {ohw}u) * {m}u + co) * {ohw}u + kk % {ohw}u;
    return {dy_at};
}}
// im2col(X)[(ci, ky, kx), (image, oh, ow)] = X[image, ci, oh·s + ky − pH, ow·s + kx − pW].
fn gather_x(_chunk: u32, column: u32, kk: u32) -> f32 {{
    if column >= {n}u || kk >= {k}u {{ return 0.0; }}
    let image = kk / {ohw}u;
    let pos = kk % {ohw}u;
    let ci = column / {khw}u;
    let tap = column % {khw}u;
    let ih = i32((pos / {ow}u) * {s}u + tap / {kw}u) - {ph};
    let iw = i32((pos % {ow}u) * {s}u + tap % {kw}u) - {pw};
    if ih < 0 || iw < 0 || ih >= {h} || iw >= {w} {{ return 0.0; }}
    return src[((image * {ci}u + ci) * {h}u + u32(ih)) * {w}u + u32(iw)];
}}
{gather_dy4}// Four consecutive kk of one im2col row: (ci, ky, kx) decompose once, the
// output position steps along.
fn gather_x4(_chunk: u32, column: u32, kk0: u32, end: u32) -> vec4<f32> {{
    var out = vec4<f32>(0.0);
    if column >= {n}u {{ return out; }}
    let ci = column / {khw}u;
    let tap = column % {khw}u;
    let ky = tap / {kw}u;
    let kx = tap % {kw}u;
    var image = kk0 / {ohw}u;
    let pos0 = kk0 % {ohw}u;
    var oy = pos0 / {ow}u;
    var ox = pos0 % {ow}u;
    // Four positions in one output row share the input row.
    if {row_fast} && ox + 3u < {ow}u && kk0 + 3u < min(end, {k}u) {{
        let ih = i32(oy * {s}u + ky) - {ph};
        if ih < 0 || ih >= {h} {{ return out; }}
        let iw0 = i32(ox * {s}u + kx) - {pw};
        let row = ((image * {ci}u + ci) * {h}u + u32(ih)) * {w}u;
        for (var e = 0u; e < 4u; e++) {{
            let iw = iw0 + i32(e * {s}u);
            if iw >= 0 && iw < {w} {{ out[e] = src[row + u32(iw)]; }}
        }}
        return out;
    }}
    for (var e = 0u; e < 4u; e++) {{
        let ih = i32(oy * {s}u + ky) - {ph};
        let iw = i32(ox * {s}u + kx) - {pw};
        if kk0 + e < min(end, {k}u) && ih >= 0 && iw >= 0 && ih < {h} && iw < {w} {{
            out[e] = src[((image * {ci}u + ci) * {h}u + u32(ih)) * {w}u + u32(iw)];
        }}
        ox += 1u;
        if ox == {ow}u {{
            ox = 0u;
            oy += 1u;
            if oy == {oh_n}u {{
                oy = 0u;
                image += 1u;
            }}
        }}
    }}
    return out;
}}
",
        ow = p.out_w,
        // Rows of whole vectors: no vector of positions straddles two rows,
        // so the branch is uniform (narrower rows measured slower with it).
        row_fast = ohw.is_multiple_of(4) && p.out_w.is_multiple_of(4),
        oh_n = p.out_h,
        s = p.stride,
        kw = p.kernel_w,
        ph = p.padding_h,
        pw = p.padding_w,
        h = p.in_h,
        w = p.in_w,
        ci = p.in_channels,
    );
    subgroup_kernel(
        shape,
        types,
        native,
        subgroup,
        &SubgroupLayout {
            variant: MatMulCoopVariant::BT,
            header,
            dimensions: format!("let m = {m}u;\nlet n = {n}u;\nlet k = {k}u;"),
            a: "grad_out",
            b: "src",
            c: "dst",
            addend: None,
            epilogue: None,
            // One "image" whose K is the whole batch: workgroup Z is the chunk.
            images: Some(Images {
                a: "0u".into(),
                b: "0u".into(),
                c: "group.z * m * n".into(),
                batch: Some(1),
            }),
            row_bounds: !m.is_multiple_of(shape.rows),
            column_bounds: !n.is_multiple_of(shape.columns),
            gather: [
                Some(Gather {
                    name: "gather_dy",
                    vector: true,
                }),
                Some(Gather {
                    name: "gather_x",
                    vector: true,
                }),
            ],
            scatter: None,
        },
    )
}

/// A forward convolution of any window, `Y[Co, oHW] = W[Co, Ci·kH·kW] · im2col(X)ᵀ`
/// per image (workgroup Z), as a BT product whose B rows are output positions
/// and whose K runs over the taps. im2col(X) gathers its elements; W loads
/// vectors when its rows hold whole ones and gathers otherwise (a 7×7 stem
/// over three channels has 147 taps).
pub(crate) fn generate_subgroup_gathered_conv(
    shape: WorkgroupMatmulShape,
    types: MatrixTypes,
    native: [u32; 3],
    subgroup: u32,
    params: &crate::kernels::conv::Params,
    split: bool,
) -> ShaderModule {
    assert_eq!(
        (types.a, types.b, types.output),
        (DType::F32, DType::F32, DType::F32)
    );
    let p = params;
    let khw = p.kernel_h * p.kernel_w;
    let (m, n, k) = (p.out_channels, p.out_h * p.out_w, p.in_channels * khw);
    let vector_weights = k.is_multiple_of(4);
    let weight_type = if vector_weights { "vec4<f32>" } else { "f32" };
    let gather_w = if vector_weights {
        String::new()
    } else {
        format!(
            "// W[co, tap] with tap = (ci, ky, kx), row-major.
fn gather_w(_image: u32, co: u32, tap: u32) -> f32 {{
    if co >= {m}u || tap >= {k}u {{ return 0.0; }}
    return weight[co * {k}u + tap];
}}
"
        )
    };
    let header = format!(
        "var<storage> src: array<f32>;
var<storage> weight: array<{weight_type}>;
var<storage, read_write> dst: array<f32>;
{gather_w}// im2col(X)[image, (oh, ow), (ci, ky, kx)] = X[image, ci, oh·s + ky − pH, ow·s + kx − pW].
fn gather_x(image: u32, pos: u32, tap: u32) -> f32 {{
    if pos >= {n}u || tap >= {k}u {{ return 0.0; }}
    let ci = tap / {khw}u;
    let t = tap % {khw}u;
    let ih = i32((pos / {ow}u) * {s}u + t / {kw}u) - {ph};
    let iw = i32((pos % {ow}u) * {s}u + t % {kw}u) - {pw};
    if ih < 0 || iw < 0 || ih >= {h} || iw >= {w} {{ return 0.0; }}
    return src[((image * {ci}u + ci) * {h}u + u32(ih)) * {w}u + u32(iw)];
}}
// Four consecutive taps of one output position: the position and the first
// tap decompose once, and the rest step (ci, ky, kx) along.
fn gather_x4(image: u32, pos: u32, tap0: u32, end: u32) -> vec4<f32> {{
    var out = vec4<f32>(0.0);
    if pos >= {n}u {{ return out; }}
    let oy = i32((pos / {ow}u) * {s}u) - {ph};
    let ox = i32((pos % {ow}u) * {s}u) - {pw};
    var ci = tap0 / {khw}u;
    let t0 = tap0 % {khw}u;
    var ky = t0 / {kw}u;
    var kx = t0 % {kw}u;
    for (var e = 0u; e < 4u; e++) {{
        let ih = oy + i32(ky);
        let iw = ox + i32(kx);
        if tap0 + e < min(end, {k}u) && ih >= 0 && iw >= 0 && ih < {h} && iw < {w} {{
            out[e] = src[((image * {ci}u + ci) * {h}u + u32(ih)) * {w}u + u32(iw)];
        }}
        kx += 1u;
        if kx == {kw}u {{
            kx = 0u;
            ky += 1u;
            if ky == {kh}u {{
                ky = 0u;
                ci += 1u;
            }}
        }}
    }}
    return out;
}}
",
        ow = p.out_w,
        s = p.stride,
        kw = p.kernel_w,
        kh = p.kernel_h,
        ph = p.padding_h,
        pw = p.padding_w,
        h = p.in_h,
        w = p.in_w,
        ci = p.in_channels,
    );
    subgroup_kernel(
        shape,
        types,
        native,
        subgroup,
        &SubgroupLayout {
            variant: MatMulCoopVariant::BT,
            header,
            dimensions: format!("let m = {m}u;\nlet n = {n}u;\nlet k = {k}u;"),
            a: "weight",
            b: "src",
            c: "dst",
            addend: None,
            epilogue: None,
            // Split K: workgroup Z is (image, chunk), and each chunk writes its
            // partial at [chunk][image] for the SumRows that follows.
            images: Some(conv_split_images(split, p.batch)),
            row_bounds: !m.is_multiple_of(shape.rows),
            column_bounds: !n.is_multiple_of(shape.columns),
            gather: [
                (!vector_weights).then_some(Gather {
                    name: "gather_w",
                    vector: false,
                }),
                Some(Gather {
                    name: "gather_x",
                    vector: true,
                }),
            ],
            scatter: None,
        },
    )
}

/// The input gradient of a pointwise convolution whose positions take no
/// whole vectors (a 7×7 stage): per image, `dX[Ci, HW] = Wᵀ[Ci, Co] · dY[HW, Co]ᵀ`
/// as a BT product with both operands gathered. With `split`, workgroup Z
/// is (image, K chunk), and each chunk writes a partial for a SumRows.
pub(crate) fn generate_subgroup_gathered_pointwise_grad_input(
    shape: WorkgroupMatmulShape,
    types: MatrixTypes,
    native: [u32; 3],
    subgroup: u32,
    params: &crate::kernels::conv::Params,
    split: bool,
) -> ShaderModule {
    assert_eq!(
        (types.a, types.b, types.output),
        (DType::F32, DType::F32, DType::F32)
    );
    let p = params;
    let (m, n, k) = (p.in_channels, p.in_h * p.in_w, p.out_channels);
    let header = format!(
        "var<storage> grad_out: array<f32>;
var<storage> weight: array<f32>;
var<storage, read_write> dst: array<f32>;
// Wᵀ[ci, co] = W[co, ci].
fn gather_wt(_image: u32, ci: u32, co: u32) -> f32 {{
    if ci >= {m}u || co >= {k}u {{ return 0.0; }}
    return weight[co * {m}u + ci];
}}
fn gather_wt4(_image: u32, ci: u32, co0: u32, end: u32) -> vec4<f32> {{
    var out = vec4<f32>(0.0);
    if ci >= {m}u {{ return out; }}
    for (var e = 0u; e < 4u; e++) {{
        if co0 + e < min(end, {k}u) {{ out[e] = weight[(co0 + e) * {m}u + ci]; }}
    }}
    return out;
}}
// dY[image, co, pos].
fn gather_dy(image: u32, pos: u32, co: u32) -> f32 {{
    if pos >= {n}u || co >= {k}u {{ return 0.0; }}
    return grad_out[(image * {k}u + co) * {n}u + pos];
}}
fn gather_dy4(image: u32, pos: u32, co0: u32, end: u32) -> vec4<f32> {{
    var out = vec4<f32>(0.0);
    if pos >= {n}u {{ return out; }}
    for (var e = 0u; e < 4u; e++) {{
        if co0 + e < min(end, {k}u) {{ out[e] = grad_out[(image * {k}u + co0 + e) * {n}u + pos]; }}
    }}
    return out;
}}
"
    );
    subgroup_kernel(
        shape,
        types,
        native,
        subgroup,
        &SubgroupLayout {
            variant: MatMulCoopVariant::BT,
            header,
            dimensions: format!("let m = {m}u;\nlet n = {n}u;\nlet k = {k}u;"),
            a: "weight",
            b: "grad_out",
            c: "dst",
            addend: None,
            epilogue: None,
            images: Some(conv_split_images(split, p.batch)),
            row_bounds: !m.is_multiple_of(shape.rows),
            column_bounds: !n.is_multiple_of(shape.columns),
            gather: [
                Some(Gather {
                    name: "gather_wt",
                    vector: true,
                }),
                Some(Gather {
                    name: "gather_dy",
                    vector: true,
                }),
            ],
            scatter: None,
        },
    )
}

/// The input gradient of a strided convolution by stride phases. Input
/// position (s·a + p, s·b + q) only receives the taps kh ≡ p + pH and
/// kw ≡ q + pW (mod s), each reading dY at (a + dh, b + dw) for a fixed
/// offset per tap. Per image and phase (workgroup Z = rank · batch + image,
/// phases ranked by taps, most first),
/// `dX[Ci, phase positions] = Wᵀ[Ci, Co·taps] · D[phase positions, Co·taps]`
/// is a BT product with that phase's K, both operands gathered, and the
/// output scattered to the phase's positions. A phase with no taps stores
/// zeros. Phases of an extent the stride does not divide differ in size,
/// and one past the extent (a width-1 image's odd column) is empty.
pub(crate) fn generate_subgroup_phased_grad_input(
    shape: WorkgroupMatmulShape,
    types: MatrixTypes,
    native: [u32; 3],
    subgroup: u32,
    params: &crate::kernels::conv::Params,
    transposed: bool,
) -> ShaderModule {
    use std::fmt::Write as _;
    assert_eq!(
        (types.a, types.b, types.output),
        (DType::F32, DType::F32, DType::F32)
    );
    let p = params;
    let s = p.stride;
    assert!(s > 1);
    let (m, co_n) = (p.in_channels, p.out_channels);
    assert!(
        !transposed || co_n.is_multiple_of(4),
        "transposed phased weights require vector-aligned output channels"
    );
    let (kh_n, kw_n, oh_n, ow_n) = (p.kernel_h, p.kernel_w, p.out_h, p.out_w);
    // Per phase: first taps, tap counts, dY offsets of the first taps, and
    // the phase's rows and columns of the input.
    struct Phase {
        kh0: u32,
        kw0: u32,
        th: u32,
        tw: u32,
        dh0: u32,
        dw0: u32,
        hp: u32,
        wp: u32,
    }
    let phase = |index: u32| {
        let (pp, qq) = (index / s, index % s);
        let first = |pos: u32, pad: u32, kernel: u32| {
            let k0 = (pos + pad) % s;
            let taps = if k0 < kernel {
                (kernel - k0).div_ceil(s)
            } else {
                0
            };
            (k0, taps, (pos + pad - k0) / s)
        };
        let (kh0, th, dh0) = first(pp, p.padding_h, kh_n);
        let (kw0, tw, dw0) = first(qq, p.padding_w, kw_n);
        let extent = |size: u32, pos: u32| {
            if pos < size {
                (size - pos).div_ceil(s)
            } else {
                0
            }
        };
        Phase {
            kh0,
            kw0,
            th,
            tw,
            dh0,
            dw0,
            hp: extent(p.in_h, pp),
            wp: extent(p.in_w, qq),
        }
    };
    // Each phase's taps and extents are known here, so its gathers divide by
    // literals: the phase is uniform over a workgroup and selects its case.
    let quotient = |x: &str, d: u32| {
        if d == 1 {
            x.to_owned()
        } else {
            format!("({x} / {d}u)")
        }
    };
    let remainder = |x: &str, d: u32| {
        if d == 1 {
            "0u".to_owned()
        } else {
            format!("({x} % {d}u)")
        }
    };
    let (mut phase_cases, mut wt_cases, mut dy_cases) =
        (String::new(), String::new(), String::new());
    let (mut wt4_cases, mut dy4_cases) = (String::new(), String::new());
    for index in 0..s * s {
        let f = phase(index);
        let (taps, n) = (f.th * f.tw, f.hp * f.wp);
        let _ = writeln!(
            phase_cases,
            "case {index}u: {{ taps = {taps}u; ph_n = {n}u; ph_wp = {}u; }}",
            f.wp.max(1)
        );
        if taps == 0 || n == 0 {
            continue;
        }
        let (co, t) = if transposed {
            (format!("(kk % {co_n}u)"), quotient("kk", co_n))
        } else {
            (quotient("kk", taps), remainder("kk", taps))
        };
        let (ty, tx) = (quotient("t", f.tw), remainder("t", f.tw));
        let _ = writeln!(
            wt_cases,
            "case {index}u: {{
    if kk < {k}u {{
        let co = {co};
        let t = {t};
        v = {weight_at};
    }}
}}",
            k = co_n * taps,
            weight_at = if transposed {
                format!(
                    "weight[(((ci * {kh_n}u + {}u + {s}u * {ty}) * {kw_n}u + {}u + {s}u * {tx}) * {co_n}u + co) / 4u][co % 4u]",
                    f.kh0, f.kw0
                )
            } else {
                format!(
                    "weight[((co * {m}u + ci) * {kh_n}u + {}u + {s}u * {ty}) * {kw_n}u + {}u + {s}u * {tx}]",
                    f.kh0, f.kw0
                )
            },
        );
        if transposed {
            // Four consecutive K: one tap, four output channels.
            let _ = writeln!(
                wt4_cases,
                "case {index}u: {{
    if kk0 < min(end, {k}u) {{
        let t = kk0 / {co_n}u;
        let co = kk0 % {co_n}u;
        v = weight[(((ci * {kh_n}u + {kh0}u + {s}u * (t / {tw}u)) * {kw_n}u + {kw0}u + {s}u * (t % {tw}u)) * {co_n}u + co) / 4u];
    }}
}}",
                k = co_n * taps,
                kh0 = f.kh0,
                kw0 = f.kw0,
                tw = f.tw,
            );
            let _ = writeln!(
                dy4_cases,
                "case {index}u: {{
    if pos < {n}u && kk0 < min(end, {k}u) {{
        let t = kk0 / {co_n}u;
        let co = kk0 % {co_n}u;
        let oh = i32(pos / {wp}u) + {dh0} - i32(t / {tw}u);
        let ow = i32(pos % {wp}u) + {dw0} - i32(t % {tw}u);
        if oh >= 0 && ow >= 0 && oh < {oh_n} && ow < {ow_n} {{
            let at = ((ph_image * {co_n}u + co) * {oh_n}u + u32(oh)) * {ow_n}u + u32(ow);
            v = vec4<f32>(grad_out[at], grad_out[at + {plane}u], grad_out[at + {plane2}u], grad_out[at + {plane3}u]);
        }}
    }}
}}",
                k = co_n * taps,
                wp = f.wp.max(1),
                tw = f.tw,
                dh0 = f.dh0,
                dw0 = f.dw0,
                plane = oh_n * ow_n,
                plane2 = 2 * oh_n * ow_n,
                plane3 = 3 * oh_n * ow_n,
            );
        }
        let (a, b) = (quotient("pos", f.wp), remainder("pos", f.wp));
        let _ = writeln!(
            dy_cases,
            "case {index}u: {{
    if pos < {n}u && kk < {k}u {{
        let co = {co};
        let t = {t};
        let oh = i32({a}) + {dh0} - i32({ty});
        let ow = i32({b}) + {dw0} - i32({tx});
        if oh >= 0 && ow >= 0 && oh < {oh_n} && ow < {ow_n} {{
            v = grad_out[((ph_image * {co_n}u + co) * {oh_n}u + u32(oh)) * {ow_n}u + u32(ow)];
        }}
    }}
}}",
            k = co_n * taps,
            dh0 = f.dh0,
            dw0 = f.dw0,
        );
    }
    // Workgroup Z runs phases with the most taps first, so the longest K
    // walks start in the first wave instead of the last.
    let mut order: Vec<u32> = (0..s * s).collect();
    order.sort_by_key(|&index| {
        let f = phase(index);
        std::cmp::Reverse(f.th * f.tw * u32::from(f.hp * f.wp > 0))
    });
    let mut order_cases = String::new();
    for (rank, index) in order.iter().enumerate() {
        let _ = writeln!(order_cases, "case {rank}u: {{ ph_phase = {index}u; }}");
    }
    let weight_type = if transposed { "vec4<f32>" } else { "f32" };
    let vector_gathers = if transposed {
        format!(
            "fn gather_wt4(_z: u32, ci: u32, kk0: u32, end: u32) -> vec4<f32> {{
    var v = vec4<f32>(0.0);
    if ci < {m}u {{
        switch ph_phase {{
{wt4_cases}default: {{}}
        }}
    }}
    return v;
}}
fn gather_dy4(_z: u32, pos: u32, kk0: u32, end: u32) -> vec4<f32> {{
    var v = vec4<f32>(0.0);
    switch ph_phase {{
{dy4_cases}default: {{}}
    }}
    return v;
}}
"
        )
    } else {
        String::new()
    };
    let header = format!(
        "var<storage> grad_out: array<f32>;
var<storage> weight: array<{weight_type}>;
var<storage, read_write> dst: array<f32>;
// This workgroup's image and stride phase (p, q), and the phase's columns.
var<private> ph_image: u32;
var<private> ph_phase: u32;
var<private> ph_wp: u32;
// Wᵀ[ci, (co, tap)] = W[co, ci, kh0 + s·(tap / tw), kw0 + s·(tap % tw)].
fn gather_wt(_z: u32, ci: u32, kk: u32) -> f32 {{
    var v = 0.0;
    if ci < {m}u {{
        switch ph_phase {{
{wt_cases}default: {{}}
        }}
    }}
    return v;
}}
// D[(a, b), (co, tap)] = dY[image, co, a + dh0 − tap / tw, b + dw0 − tap % tw].
fn gather_dy(_z: u32, pos: u32, kk: u32) -> f32 {{
    var v = 0.0;
    switch ph_phase {{
{dy_cases}default: {{}}
    }}
    return v;
}}
{vector_gathers}"
    );
    let dimensions = format!(
        "ph_image = group.z % {batch}u;
switch group.z / {batch}u {{
{order_cases}default: {{}}
}}
var taps = 0u;
var ph_n = 0u;
switch ph_phase {{
{phase_cases}default: {{}}
}}
let m = {m}u;
let n = ph_n;
let k = {co_n}u * taps;",
        batch = p.batch.max(1),
    );
    // Output column c = a · wp + b lands at (s·a + p, s·b + q) of the image.
    let scatter = format!(
        "((ph_image * {m}u + r) * {h}u + {s}u * (c / ph_wp) + ph_phase / {s}u) * {w}u + {s}u * (c % ph_wp) + ph_phase % {s}u",
        h = p.in_h,
        w = p.in_w,
    );
    subgroup_kernel(
        shape,
        types,
        native,
        subgroup,
        &SubgroupLayout {
            variant: MatMulCoopVariant::BT,
            header,
            dimensions,
            a: "weight",
            b: "grad_out",
            c: "dst",
            addend: None,
            epilogue: None,
            images: Some(Images {
                a: "0u".into(),
                b: "0u".into(),
                c: "0u".into(),
                batch: None,
            }),
            row_bounds: !m.is_multiple_of(shape.rows),
            // Phase sizes differ; every column tile checks its phase's.
            column_bounds: true,
            gather: [
                Some(Gather {
                    name: "gather_wt",
                    vector: transposed,
                }),
                Some(Gather {
                    name: "gather_dy",
                    vector: transposed,
                }),
            ],
            scatter: Some(scatter),
        },
    )
}

/// Subgroup-tiled shape for a Winograd `[rows, columns]` output tile on
/// `(native, subgroup)`, choosing one subgroup per `nm × nn` tile where it
/// fits the 256-thread limit, or tiling multiple accumulators per subgroup.
pub(crate) fn winograd_subgroup_shape(
    tile: [u32; 2],
    native: [u32; 3],
    subgroup: u32,
) -> Option<WorkgroupMatmulShape> {
    let [nm, nn, _] = native;
    let [rows, columns] = tile;
    if !supported_subgroup_width(subgroup)
        || !rows.is_multiple_of(nm)
        || !columns.is_multiple_of(nn)
    {
        return None;
    }
    let max_subgroups = 256 / subgroup;
    let max_wr = rows / nm;
    let max_wc = columns / nn;
    // Prefer as many subgroups as fit (at most one per native tile).
    let (mut wr, mut wc) = (max_wr, max_wc);
    while wr * wc > max_subgroups {
        if wr >= wc && wr > 1 && wr.is_multiple_of(2) {
            wr /= 2;
        } else if wc > 1 && wc.is_multiple_of(2) {
            wc /= 2;
        } else {
            return None;
        }
    }
    let shape = WorkgroupMatmulShape {
        rows,
        columns,
        k_stage: 32,
        threads: wr * wc * subgroup,
        stage_operands: false,
        subgroup_rows: wr,
        prefetch: false,
    };
    shape.subgroup_grid_for(native, subgroup).map(|_| shape)
}

/// The batched Winograd product `C[z] = A[z] · B[z]` over the 16 transform
/// planes (workgroup Z), from the reduced transform buffers, on the
/// `[rows, columns]` output grid the reduced Winograd lowering sizes: staged
/// through shared memory where the typed kernel it replaces loaded each
/// fragment from device memory. N may end inside a column tile when it is a
/// multiple of 4. Bindings and parameters are the typed kernel's
/// (`matrix_a`, `matrix_b`, `matrix_c`).
pub(crate) fn generate_subgroup_winograd(
    storage: DType,
    tile: [u32; 2],
    native: [u32; 3],
    subgroup: u32,
) -> ShaderModule {
    assert!(matches!(storage, DType::F16 | DType::BF16));
    let shape = winograd_subgroup_shape(tile, native, subgroup).unwrap_or_else(|| {
        panic!("unsupported Winograd subgroup tile {tile:?} on {native:?}/{subgroup}")
    });
    let types = MatrixTypes {
        a: storage,
        b: storage,
        multiply: storage,
        output: DType::F32,
    };
    let ty = MatrixTypes::name(storage);
    let header = format!(
        "struct Params {{ m: u32, n: u32, k: u32, batches: u32 }}\nvar<storage> matrix_a: array<vec4<{ty}>>;\nvar<storage> matrix_b: array<vec4<{ty}>>;\nvar<storage, read_write> matrix_c: array<f32>;\nvar<uniform> params: Params;\n"
    );
    subgroup_kernel(
        shape,
        types,
        native,
        subgroup,
        &SubgroupLayout {
            variant: MatMulCoopVariant::Normal,
            header,
            dimensions: "let m = params.m;\nlet n = params.n;\nlet k = params.k;".into(),
            a: "matrix_a",
            b: "matrix_b",
            c: "matrix_c",
            addend: None,
            epilogue: None,
            images: Some(Images {
                a: "image * m * k".into(),
                b: "image * k * n".into(),
                c: "image * m * n".into(),
                batch: None,
            }),
            // Channels fill whole row tiles; the transform's tile count may end
            // inside a column tile, in whole vectors.
            row_bounds: false,
            column_bounds: true,
            gather: [None, None],
            scatter: None,
        },
    )
}

/// Whether [`generate_subgroup_pointwise_conv`] serves a pointwise
/// convolution, with the 32x32 tile of four subgroups; [`conv_tile_for`]
/// picks the tile it runs.
pub(crate) fn pointwise_conv_subgroup_shape(
    types: MatrixTypes,
    native: [u32; 3],
    subgroup: u32,
    params: &crate::kernels::conv::Params,
) -> Option<WorkgroupMatmulShape> {
    params
        .vectorized_forward()
        .then(|| gathered_conv_subgroup_shape(types, native, subgroup))
        .flatten()
}

/// Whether the device runs the generated convolutions whose operands gather
/// their elements ([`generate_subgroup_gathered_weight_grad`]): any kernel,
/// stride and padding. The 32x32 tile it returns is the smallest;
/// [`conv_tile_for`] picks the tile each one runs.
pub(crate) fn gathered_conv_subgroup_shape(
    types: MatrixTypes,
    native: [u32; 3],
    subgroup: u32,
) -> Option<WorkgroupMatmulShape> {
    if super::checked_cooperative_shape(&[native]).is_none()
        || !supported_subgroup_width(subgroup)
        || !matches!(types.multiply, DType::F16 | DType::BF16)
        || (types.a, types.b, types.output) != (DType::F32, DType::F32, DType::F32)
    {
        return None;
    }
    let shape = conv_shape((32, 32), subgroup);
    shape.subgroup_grid_for(native, subgroup).map(|_| shape)
}

/// Subgroup tiles of the generated convolutions, largest first: 64x64,
/// 64x32, and 32x32 over a 2×2 subgroup grid (4 subgroups, ≤ 256 threads
/// for 16-, 32- and 64-lane subgroups alike).
const CONV_TILES: [(u32, u32); 3] = [(64, 64), (64, 32), (32, 32)];

fn conv_shape((rows, columns): (u32, u32), subgroup: u32) -> WorkgroupMatmulShape {
    WorkgroupMatmulShape {
        rows,
        columns,
        k_stage: 32,
        threads: 4 * subgroup,
        stage_operands: false,
        subgroup_rows: 2,
        prefetch: false,
    }
}

/// Which product of a convolution a generated kernel computes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum ConvProduct {
    /// `Y[Co, oH·oW]` per image.
    Forward,
    /// A pointwise `dX[Ci, H·W]` per image.
    InputGradient,
    /// `dX[Ci, phase positions]` per image and stride phase.
    PhasedInputGradient,
    /// `dW[Co, Ci·kH·kW]` over K partitions.
    WeightGradient,
}

/// The subgroup tile of a generated convolution on a 32-lane device.
pub(crate) fn conv_tile(
    product: ConvProduct,
    p: &crate::kernels::conv::Params,
) -> WorkgroupMatmulShape {
    conv_tile_for(product, p, SUBGROUP_WIDTH)
}

/// The subgroup tile of a generated convolution on `subgroup`-wide subgroups.
/// Larger tiles load each operand element for more products, so the largest
/// tile that still launches
/// [`MIN_F16_WORKGROUPS`](crate::kernels::conv::MIN_F16_WORKGROUPS)
/// workgroups runs.
pub(crate) fn conv_tile_for(
    product: ConvProduct,
    p: &crate::kernels::conv::Params,
    subgroup: u32,
) -> WorkgroupMatmulShape {
    let batch = p.batch.max(1);
    match product {
        ConvProduct::Forward => {
            conv_product_tile_for(p.out_channels, p.out_h * p.out_w, batch, subgroup)
        }
        ConvProduct::InputGradient => {
            conv_product_tile_for(p.in_channels, p.in_h * p.in_w, batch, subgroup)
        }
        ConvProduct::PhasedInputGradient => {
            let s = p.stride.max(1);
            conv_product_tile_for(
                p.in_channels,
                p.in_h.div_ceil(s) * p.in_w.div_ceil(s),
                batch * s * s,
                subgroup,
            )
        }
        ConvProduct::WeightGradient => conv_shape(
            if p.out_channels >= 64 && p.in_channels * p.kernel_h * p.kernel_w >= 32 {
                CONV_TILES[1]
            } else {
                CONV_TILES[2]
            },
            subgroup,
        ),
    }
}

/// [`conv_tile`] of an `m × n` product over `layers` workgroup layers.
pub(crate) fn conv_product_tile(m: u32, n: u32, layers: u32) -> WorkgroupMatmulShape {
    conv_product_tile_for(m, n, layers, SUBGROUP_WIDTH)
}

pub(crate) fn conv_product_tile_for(
    m: u32,
    n: u32,
    layers: u32,
    subgroup: u32,
) -> WorkgroupMatmulShape {
    CONV_TILES
        .into_iter()
        .map(|dims| conv_shape(dims, subgroup))
        .find(|s| {
            // A tile past the product's extent only multiplies zeros.
            s.rows <= m
                && s.columns <= n
                && u64::from(m.div_ceil(s.rows))
                    * u64::from(n.div_ceil(s.columns))
                    * u64::from(layers)
                    >= crate::kernels::conv::MIN_F16_WORKGROUPS
        })
        .unwrap_or_else(|| conv_shape(CONV_TILES[2], subgroup))
}

/// How a subgroup-tiled kernel binds its operands.
struct SubgroupLayout<'a> {
    variant: MatMulCoopVariant,
    /// Buffer and parameter declarations.
    header: String,
    /// Statements defining `m`, `n` and `k`.
    dimensions: String,
    a: &'static str,
    b: &'static str,
    c: &'static str,
    addend: Option<&'static str>,
    /// Optional fused unary elementwise epilogue applied in the store loop.
    epilogue: Option<&'a crate::compile::MatMulEpilogue>,
    /// Per-image products in place of K partitions over workgroup Z.
    images: Option<Images>,
    row_bounds: bool,
    column_bounds: bool,
    /// Gathers for A and B in place of their vector loads: operands with no
    /// contiguous K vectors (an im2col of a strided or padded image). Only
    /// K-contiguous layouts (A rows, BT rows) gather.
    gather: [Option<Gather>; 2],
    /// The element offset in C of output row `r`, column `c`, in place of
    /// `c_base + r * ldc + c`, for outputs whose columns land at scattered
    /// positions (one stride phase of an image). Stores pass through shared
    /// memory.
    scatter: Option<String>,
}

/// A WGSL function `fn(image, row, k) -> f32` in a kernel's header that
/// computes an f32 element of an operand, zero outside it.
#[derive(Clone, Copy)]
struct Gather {
    name: &'static str,
    /// The header also defines `{name}4(image, row, k0, end) -> vec4<f32>`,
    /// which gathers four consecutive K elements and shares their index work.
    vector: bool,
}

/// Element offsets of A, B and C for an image, in terms of `image`, `m`,
/// `n`, `k` and `group`. With `batch`, workgroup Z covers `batch` images
/// times `num_workgroups.z / batch` K chunks, and each Z layer writes its
/// own partial product; otherwise Z is the image and K is whole.
struct Images {
    a: String,
    b: String,
    c: String,
    batch: Option<u32>,
}

/// Loads or stores of one staged operand vector at a K index or shared-memory offset.
type StageEmit = Box<dyn Fn(&str) -> String>;

fn subgroup_kernel(
    shape: WorkgroupMatmulShape,
    types: MatrixTypes,
    native: [u32; 3],
    subgroup: u32,
    layout: &SubgroupLayout<'_>,
) -> ShaderModule {
    use std::fmt::Write as _;
    let [nm, nn, nk] = native;
    let variant = layout.variant;
    let (wr, wc, tr, tc) = shape
        .subgroup_grid_for(native, subgroup)
        .expect("subgroup-tiled shape");
    let (bm, bn, ks, threads) = (shape.rows, shape.columns, shape.k_stage, shape.threads);
    assert!(matches!(ks, 16 | 32 | 64) && ks.is_multiple_of(nk));
    let bf16 = types.multiply == DType::BF16;
    let multiply = MatrixTypes::name(types.multiply);
    for storage in [types.a, types.b] {
        assert!(storage == DType::F32 || storage == types.multiply);
    }
    let (epi_decl, epi_body) = match layout.epilogue {
        Some(epi) => {
            assert!(
                epi.inputs.is_empty(),
                "subgroup_kernel epilogue expects unary DAG without extra buffer bindings"
            );
            super::matmul_epilogue_to_wgsl(epi)
        }
        None => (String::new(), String::new()),
    };
    let a_t = variant == MatMulCoopVariant::AT;
    let b_t = variant == MatMulCoopVariant::BT;
    // Eight elements of padding a line rotate the shared-memory banks.
    let a_pitch = if a_t { bm + 8 } else { ks + 8 };
    let b_pitch = if b_t { ks + 8 } else { bn + 8 };
    let a_size = if a_t { ks * a_pitch } else { bm * a_pitch };
    let b_size = if b_t { bn * b_pitch } else { ks * b_pitch };
    // Operand tiles move as vec4 slots of the multiplication type.
    let lanes = 4;
    let a_stage4 = a_size / lanes;
    let b_stage4 = b_size / lanes;
    let stages = if shape.prefetch { 2 } else { 1 };
    let slot = format!("vec4<{multiply}>");
    let bounded = layout.row_bounds || layout.column_bounds;
    let scratch = bounded
        || types.output != DType::F32
        || layout.scatter.is_some()
        || layout.epilogue.is_some();
    let tile_elems = nm * nn;
    let a_mat = format!("coop_mat{nk}x{nm}<{multiply},A>");
    let b_mat = format!("coop_mat{nn}x{nk}<{multiply},B>");
    let c_mat = format!("coop_mat{nn}x{nm}<f32,C>");
    let (lda, ldb, ldc) = (
        if a_t { "m" } else { "k" },
        if b_t { "k" } else { "n" },
        "n",
    );
    let (a_off, b_off, c_off) = match layout.images {
        Some(ref images) => (images.a.clone(), images.b.clone(), images.c.clone()),
        None => (
            "0u".to_owned(),
            "0u".to_owned(),
            "group.z * m * n".to_owned(),
        ),
    };
    let mut s = String::from("enable wgpu_cooperative_matrix;\n");
    if types.uses(DType::F16) {
        s += "enable f16;\n";
    }
    if types.uses(DType::BF16) {
        s += "enable wgpu_bfloat16;\n";
        s += include_str!("../shaders/bf16_round.wgsl");
    }
    s += &layout.header;
    if !epi_decl.is_empty() {
        s += &epi_decl;
        s.push('\n');
    }
    let _ = writeln!(
        s,
        "var<workgroup> sa: array<{slot}, {}>;",
        a_stage4 * stages
    );
    let _ = writeln!(
        s,
        "var<workgroup> sb: array<{slot}, {}>;",
        b_stage4 * stages
    );
    if scratch {
        let _ = writeln!(
            s,
            "var<workgroup> scratch: array<f32, {}>;",
            wr * wc * tile_elems
        );
    }
    let range = if let Some(Images {
        batch: Some(batch), ..
    }) = layout.images
    {
        format!(
            "let chunks = groups.z / {batch}u;
let image = group.z / chunks;
let chunk = group.z % chunks;
let blocks = (k + {ks}u - 1u) / {ks}u;
let base = blocks / chunks;
let extra = blocks % chunks;
let first = (chunk * base + min(chunk, extra)) * {ks}u;
let end = min(k, first + (base + select(0u, 1u, chunk < extra)) * {ks}u);"
        )
    } else if layout.images.is_some() {
        "let image = group.z;\nlet first = 0u;\nlet end = k;".to_owned()
    } else {
        format!(
            "let blocks = (k + {ks}u - 1u) / {ks}u;
let base = blocks / params.splits;
let extra = blocks % params.splits;
let first = (group.z * base + min(group.z, extra)) * {ks}u;
let end = min(k, first + (base + select(0u, 1u, group.z < extra)) * {ks}u);"
        )
    };
    let _ = writeln!(
        s,
        "@compute @workgroup_size({threads})
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(num_workgroups) groups: vec3<u32>, @builtin(local_invocation_index) lid: u32, @builtin(subgroup_id) sg: u32, @builtin(subgroup_invocation_id) lane: u32) {{
{dimensions}
let row = group.x * {bm}u;
let col = group.y * {bn}u;
{range}
let a_base = {a_off};
let b_base = {b_off};
let c_base = {c_off};
let lda = {lda};
let ldb = {ldb};
let ldc = {ldc};
let wrow = (sg / {wc}u) * {}u;
let wcol = (sg % {wc}u) * {}u;",
        tr * nm,
        tc * nn,
        dimensions = layout.dimensions,
    );
    for i in 0..tr {
        for j in 0..tc {
            let _ = writeln!(s, "var acc{i}_{j} = {c_mat}();");
        }
    }
    // Staging: lines along each operand's strided axis, vectors along its
    // contiguous one, with (line, column) relative to the tile at stage `t`.
    struct Plan {
        name: &'static str,
        buffer: &'static str,
        storage: DType,
        lines: u32,
        width: u32,
        pitch: u32,
        index: &'static str,
        guard: &'static str,
    }
    let plans = [
        if a_t {
            Plan {
                name: "a",
                buffer: layout.a,
                storage: types.a,
                lines: ks,
                width: bm,
                pitch: a_pitch,
                index: "(a_base + (T + L) * lda + row + C) / 4u",
                guard: "T + L < end && row + C < m",
            }
        } else {
            Plan {
                name: "a",
                buffer: layout.a,
                storage: types.a,
                lines: bm,
                width: ks,
                pitch: a_pitch,
                index: "(a_base + (row + L) * lda + T + C) / 4u",
                guard: "row + L < m && T + C < end",
            }
        },
        if b_t {
            Plan {
                name: "b",
                buffer: layout.b,
                storage: types.b,
                lines: bn,
                width: ks,
                pitch: b_pitch,
                index: "(b_base + (col + L) * ldb + T + C) / 4u",
                guard: if layout.column_bounds {
                    "col + L < n && T + C < end"
                } else {
                    "T + C < end"
                },
            }
        } else {
            Plan {
                name: "b",
                buffer: layout.b,
                storage: types.b,
                lines: ks,
                width: bn,
                pitch: b_pitch,
                index: "(b_base + (T + L) * ldb + col + C) / 4u",
                guard: if layout.column_bounds {
                    "T + L < end && col + C < n"
                } else {
                    "T + L < end"
                },
            }
        },
    ];
    // With double-buffered prefetch, global loads for stage `t + ks` overlap
    // with the cooperative products of stage `t`, then write into the idle
    // shared-memory buffer so each K stage pays one workgroupBarrier() instead
    // of two.
    let prefetch = shape.prefetch;
    let mut stage = String::new();
    let mut declare = String::new();
    let mut load_at: Vec<StageEmit> = Vec::new();
    let mut store_at: Vec<StageEmit> = Vec::new();
    for plan in &plans {
        let count = plan.lines * plan.width / 4;
        let scalar = MatrixTypes::name(plan.storage);
        let stage_base = if plan.name == "a" { "BASE_A" } else { "BASE_B" };
        for e in 0..count.div_ceil(threads) {
            let line = format!("{}l{e}", plan.name);
            let column = format!("{}c{e}", plan.name);
            let partial = (e + 1) * threads > count;
            let _ = writeln!(
                s,
                "let {line} = (lid + {o}u) / {w}u;\nlet {column} = ((lid + {o}u) % {w}u) * 4u;",
                o = e * threads,
                w = plan.width / 4
            );
            let index = plan.index.replace('L', &line).replace('C', &column);
            let mut guard = plan.guard.replace('L', &line).replace('C', &column);
            let in_tile = if partial {
                format!("lid + {}u < {count}u && ", e * threads)
            } else {
                String::new()
            };
            let value = format!("v{}{e}", plan.name);
            let mut loaded = format!("{}[{index}]", plan.buffer);
            // `T` stands for the stage's first K index until a stage is chosen.
            if let Some(Gather { name: f, vector }) = layout.gather[usize::from(plan.name == "b")] {
                assert_eq!(plan.storage, DType::F32, "gathers compute f32 elements");
                let r = match (plan.name, a_t, b_t) {
                    ("a", false, _) => format!("row + {line}"),
                    ("b", _, true) => format!("col + {line}"),
                    _ => panic!("gathers take K-contiguous operands"),
                };
                loaded = if vector {
                    format!("{f}4(image, {r}, T + {column}, end)")
                } else {
                    let parts: Vec<String> = (0..4)
                        .map(|e| format!("select(0.0, {f}(image, {r}, T + {column} + {e}u), T + {column} + {e}u < end)"))
                        .collect();
                    format!("vec4<f32>({})", parts.join(", "))
                };
                guard = "true".into();
            }
            let _ = writeln!(
                stage,
                "var {value} = vec4<{scalar}>(0.0);\nif {in_tile}{} {{ {value} = {}; }}",
                guard.replace('T', "t"),
                loaded.replace('T', "t"),
            );
            let _ = writeln!(declare, "var {value} = vec4<{scalar}>(0.0);");
            {
                let (value, scalar, guard, loaded, in_tile) = (
                    value.clone(),
                    scalar.to_owned(),
                    guard.clone(),
                    loaded.clone(),
                    in_tile.clone(),
                );
                load_at.push(Box::new(move |k: &str| {
                    format!(
                        "{value} = vec4<{scalar}>(0.0);\nif {in_tile}{} {{ {value} = {}; }}\n",
                        guard.replace('T', k),
                        loaded.replace('T', k)
                    )
                }));
            }
            let at = format!("{line} * {}u + {column}", plan.pitch);
            let shared = format!("s{}", plan.name);
            let rhs = if bf16 && plan.storage == DType::F32 {
                format!(
                    "vec4<bf16>(round_bf16({value}.x), round_bf16({value}.y), round_bf16({value}.z), round_bf16({value}.w))"
                )
            } else {
                format!("vec4<{multiply}>({value})")
            };
            let plain_store = format!("{shared}[({at}) / 4u] = {rhs};");
            if partial {
                let _ = writeln!(
                    stage,
                    "if lid + {}u < {count}u {{ {plain_store} }}",
                    e * threads
                );
            } else {
                let _ = writeln!(stage, "{plain_store}");
            }
            let off_expr = format!("({at}) / 4u");
            let offset_threads = e * threads;
            store_at.push(Box::new(move |bases: &str| {
                // `bases` is "0u|0u" or "nxt_a|nxt_b".
                let (ba, bb) = bases.split_once('|').unwrap();
                let base = if stage_base == "BASE_A" { ba } else { bb };
                let idx = if base == "0u" {
                    off_expr.clone()
                } else {
                    format!("{base} + {off_expr}")
                };
                let stmt = format!("{shared}[{idx}] = {rhs};");
                if partial {
                    format!("if lid + {offset_threads}u < {count}u {{ {stmt} }}\n")
                } else {
                    format!("{stmt}\n")
                }
            }));
        }
    }
    let emit_product = |base_a: &str, base_b: &str| -> String {
        let mut product = String::new();
        for q in 0..ks / nk {
            for i in 0..tr {
                let (load, offset, stride) = if a_t {
                    (
                        "coopLoad",
                        format!("{}u * {a_pitch}u + wrow + {}u", q * nk, i * nm),
                        a_pitch,
                    )
                } else {
                    (
                        "coopLoadT",
                        format!("(wrow + {}u) * {a_pitch}u + {}u", i * nm, q * nk),
                        a_pitch,
                    )
                };
                let idx = if base_a.is_empty() {
                    format!("({offset}) / {lanes}u")
                } else {
                    format!("{base_a} + ({offset}) / {lanes}u")
                };
                let _ = writeln!(
                    product,
                    "let a{i}_{q} = {load}<{a_mat}>(&sa[{idx}], {}u);",
                    stride / lanes
                );
            }
            for j in 0..tc {
                let (load, offset, stride) = if b_t {
                    (
                        "coopLoad",
                        format!("(wcol + {}u) * {b_pitch}u + {}u", j * nn, q * nk),
                        b_pitch,
                    )
                } else {
                    (
                        "coopLoadT",
                        format!("{}u * {b_pitch}u + wcol + {}u", q * nk, j * nn),
                        b_pitch,
                    )
                };
                let idx = if base_b.is_empty() {
                    format!("({offset}) / {lanes}u")
                } else {
                    format!("{base_b} + ({offset}) / {lanes}u")
                };
                let _ = writeln!(
                    product,
                    "let b{j}_{q} = {load}<{b_mat}>(&sb[{idx}], {}u);",
                    stride / lanes
                );
            }
            for i in 0..tr {
                for j in 0..tc {
                    let _ = writeln!(
                        product,
                        "acc{i}_{j} = coopMultiplyAdd(a{i}_{q}, b{j}_{q}, acc{i}_{j});"
                    );
                }
            }
        }
        product
    };
    if prefetch {
        let first_load: String = load_at.iter().map(|f| f("first")).collect();
        let first_store: String = store_at.iter().map(|f| f("0u|0u")).collect();
        let next_load: String = load_at.iter().map(|f| f("tn")).collect();
        let next_store: String = store_at.iter().map(|f| f("nxt_a|nxt_b")).collect();
        let product = emit_product("cur_a", "cur_b");
        let _ = writeln!(
            s,
            "{declare}if first < end {{\n{first_load}{first_store}}}\nworkgroupBarrier();\nvar buf = 0u;\nfor (var t = first; t < end; t += {ks}u) {{\nlet tn = t + {ks}u;\nlet cur_a = buf * {a_stage4}u;\nlet cur_b = buf * {b_stage4}u;\nlet nxt = 1u - buf;\nlet nxt_a = nxt * {a_stage4}u;\nlet nxt_b = nxt * {b_stage4}u;\nif tn < end {{\n{next_load}}}\n{product}if tn < end {{\n{next_store}}}\nworkgroupBarrier();\nbuf = nxt;\n}}"
        );
    } else {
        let product = emit_product("", "");
        let _ = writeln!(
            s,
            "for (var t = first; t < end; t += {ks}u) {{\n{stage}workgroupBarrier();\n{product}workgroupBarrier();\n}}"
        );
    }
    let tile_origin = |i: u32, j: u32| {
        (
            format!("row + wrow + {}u", i * nm),
            format!("col + wcol + {}u", j * nn),
        )
    };
    let c = layout.c;
    let mut direct = String::new();
    for i in 0..tr {
        for j in 0..tc {
            let (r, cc) = tile_origin(i, j);
            if let Some(src) = layout.addend {
                let _ = writeln!(
                    direct,
                    "acc{i}_{j} = acc{i}_{j} + coopLoadT<{c_mat}>(&{src}[({r}) * ldc + {cc}], ldc);"
                );
            }
            let _ = writeln!(
                direct,
                "coopStoreT(acc{i}_{j}, &{c}[c_base + ({r}) * ldc + {cc}], ldc);"
            );
        }
    }
    let mut staged = String::new();
    if scratch {
        for i in 0..tr {
            for j in 0..tc {
                let (r, cc) = tile_origin(i, j);
                let stored = format!("scratch[sg * {tile_elems}u + e]");
                let base_val = match layout.addend {
                    Some(src) => format!("{stored} + {src}[r * ldc + c]"),
                    None => stored,
                };
                let index = layout.scatter.as_deref().unwrap_or("c_base + r * ldc + c");
                let apply = if layout.epilogue.is_some() {
                    let final_val = match types.output {
                        DType::F32 => "val".to_owned(),
                        DType::BF16 => "round_bf16(val)".to_owned(),
                        dtype => format!("{}(val)", MatrixTypes::name(dtype)),
                    };
                    format!(
                        "if r < m && c < n {{ let idx = r * ldc + c; let col = c; var val = {base_val};\n{epi_body}\n{c}[{index}] = {final_val}; }}"
                    )
                } else {
                    let value = match types.output {
                        DType::F32 => base_val,
                        DType::BF16 => format!("round_bf16({base_val})"),
                        dtype => format!("{}({base_val})", MatrixTypes::name(dtype)),
                    };
                    format!("if r < m && c < n {{ {c}[{index}] = {value}; }}")
                };
                let _ = writeln!(
                    staged,
                    "coopStoreT(acc{i}_{j}, &scratch[sg * {tile_elems}u], {nn}u);
workgroupBarrier();
for (var e = lane; e < {tile_elems}u; e += {subgroup}u) {{
let r = {r} + e / {nn}u;
let c = {cc} + e % {nn}u;
{apply}
}}
workgroupBarrier();"
                );
            }
        }
    }
    if !scratch {
        s += &direct;
    } else if types.output == DType::F32 && layout.scatter.is_none() && layout.epilogue.is_none() {
        // Only the last row and column of workgroups are ragged; the branch
        // is uniform. `&&` would lower through a local variable, which
        // naga's uniformity analysis treats as varying.
        let mut whole = Vec::new();
        if layout.row_bounds {
            whole.push(format!("(row + {bm}u <= m)"));
        }
        if layout.column_bounds {
            whole.push(format!("(col + {bn}u <= n)"));
        }
        let _ = writeln!(
            s,
            "if {} {{\n{direct}}} else {{\n{staged}}}",
            whole.join(" & ")
        );
    } else {
        s += &staged;
    }
    s += "}\n";
    ShaderModule::with_hint(s, "subgroup-matmul").with_subgroup(subgroup, [threads, 1, 1])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cycle tiles and output storage through layouts and operand conversions,
    /// exercising both staging modes and guarded rows.
    #[test]
    fn rectangular_typed_modules_validate_and_emit_spirv() {
        let capabilities = naga::valid::Capabilities::SHADER_BFLOAT16
            | naga::valid::Capabilities::SHADER_FLOAT16
            | naga::valid::Capabilities::COOPERATIVE_MATRIX
            | naga::valid::Capabilities::COOPERATIVE_MATRIX_WORKGROUP
            | naga::valid::Capabilities::SUBGROUP;
        let shapes: Vec<_> = WorkgroupMatmulShape::all().collect();
        let outputs = [DType::F32, DType::F16, DType::BF16];
        let mut index = 0;
        for (g, group) in [
            ShaderGroup::MatMul,
            ShaderGroup::MatMulAT,
            ShaderGroup::MatMulBT,
            ShaderGroup::MatMulAdd,
            ShaderGroup::MatMulATAdd,
            ShaderGroup::MatMulBTAdd,
        ]
        .into_iter()
        .enumerate()
        {
            for (p, (a_f32, b_f32)) in [(false, false), (true, false), (false, true), (true, true)]
                .into_iter()
                .enumerate()
            {
                let multiply = [DType::F16, DType::BF16][(g + p) % 2];
                let operand = |f32| if f32 { DType::F32 } else { multiply };
                for row_bounds in [false, true] {
                    for stage_operands in [false, true] {
                        let types = MatrixTypes {
                            a: operand(a_f32),
                            b: operand(b_f32),
                            multiply,
                            output: outputs[index % outputs.len()],
                        };
                        let shape = WorkgroupMatmulShape {
                            stage_operands,
                            ..shapes[(index / 2) % shapes.len()]
                        };
                        index += 1;
                        let mut module = generate_workgroup_matmul(group, shape, types, row_bounds);
                        super::super::validate_shader_for_test(
                            &mut module.module,
                            capabilities,
                            (1, 6),
                        )
                        .unwrap_or_else(|error| panic!("{group:?} {types:?} {shape:?}: {error}"));
                    }
                }
            }
        }
        // Each shape takes two consecutive slots, so every shape is generated.
        assert!(
            index / 2 >= shapes.len(),
            "{index} slots for {} shapes",
            shapes.len()
        );
    }

    /// Pointwise convolutions with ragged rows and columns, both directions.
    #[test]
    fn subgroup_pointwise_convolutions_validate() {
        let capabilities = naga::valid::Capabilities::SHADER_BFLOAT16
            | naga::valid::Capabilities::SHADER_FLOAT16
            | naga::valid::Capabilities::COOPERATIVE_MATRIX
            | naga::valid::Capabilities::SUBGROUP;
        for (ci, co, h, w) in [
            (20u32, 36u32, 4u32, 5u32),
            (64, 64, 8, 8),
            (256, 64, 56, 56),
        ] {
            let params = crate::kernels::conv::Params {
                batch: 2,
                in_channels: ci,
                in_h: h,
                in_w: w,
                out_channels: co,
                kernel_h: 1,
                kernel_w: 1,
                stride: 1,
                padding_h: 0,
                out_h: h,
                out_w: w,
                padding_w: 0,
                kernel_w_multiplier: 0,
                kernel_hw_multiplier: 0,
                column_width_multiplier: 0,
                output_spatial_multiplier: 0,
            };
            for multiply in [DType::F16, DType::BF16] {
                let types = MatrixTypes {
                    multiply,
                    ..MatrixTypes::F32_AS_F16
                };
                let shape =
                    pointwise_conv_subgroup_shape(types, [16, 16, 16], SUBGROUP_WIDTH, &params)
                        .expect("pointwise subgroup shape");
                assert_eq!((shape.rows, shape.columns), (32, 32));
                for backward in [false, true] {
                    let mut module = generate_subgroup_pointwise_conv(
                        backward,
                        shape,
                        types,
                        [16, 16, 16],
                        SUBGROUP_WIDTH,
                        &params,
                    );
                    super::super::validate_shader_for_test(
                        &mut module.module,
                        capabilities,
                        (1, 6),
                    )
                    .unwrap_or_else(|error| {
                        panic!("{ci}x{co} {h}x{w} {multiply:?} {backward}: {error}")
                    });
                }
            }
            assert!(
                pointwise_conv_subgroup_shape(MatrixTypes::F32_AS_F16, [16, 8, 16], 32, &params)
                    .is_none()
            );
            assert!(
                pointwise_conv_subgroup_shape(MatrixTypes::F32_AS_F16, [16, 16, 16], 64, &params)
                    .is_some()
            );
            assert!(
                pointwise_conv_subgroup_shape(MatrixTypes::F32_AS_F16, [8, 16, 16], 16, &params)
                    .is_some()
            );
            assert!(
                pointwise_conv_subgroup_shape(MatrixTypes::F32_AS_F16, [8, 8, 8], 32, &params)
                    .is_some()
            );
            assert!(
                pointwise_conv_subgroup_shape(MatrixTypes::F32_AS_F16, [16, 16, 16], 8, &params)
                    .is_none()
            );
        }
    }

    /// Subgroup tiles validate across all supported native matrix shapes
    /// (`16×16×16`, `8×16×16`, `8×8×8`), subgroup widths (`16`, `32`, `64`),
    /// double-buffered `prefetch`, and fused unary epilogues.
    #[test]
    fn subgroup_portable_prefetch_and_epilogue_validate() {
        let capabilities = naga::valid::Capabilities::SHADER_BFLOAT16
            | naga::valid::Capabilities::SHADER_FLOAT16
            | naga::valid::Capabilities::COOPERATIVE_MATRIX
            | naga::valid::Capabilities::SUBGROUP;
        let relu = crate::compile::MatMulEpilogue {
            dag: crate::schedule::PointwiseDAG {
                n_inputs: 1,
                ops: vec![
                    crate::schedule::Pw::LoadInput(0),
                    crate::schedule::Pw::Relu(0),
                ],
                output: 1,
            },
            inputs: Vec::new(),
        };
        for (native, subgroup) in [
            ([16, 16, 16], 32),
            ([16, 16, 16], 64),
            ([8, 16, 16], 16),
            ([8, 16, 16], 32),
            ([8, 8, 8], 32),
        ] {
            let shapes: Vec<_> = WorkgroupMatmulShape::all_for(native, subgroup)
                .filter(|s| s.subgroup_rows != 0)
                .collect();
            assert!(!shapes.is_empty(), "no shapes for {native:?}/{subgroup}");
            for (idx, group) in [
                ShaderGroup::MatMul,
                ShaderGroup::MatMulAT,
                ShaderGroup::MatMulBT,
                ShaderGroup::MatMulAdd,
            ]
            .into_iter()
            .enumerate()
            {
                for prefetch in [false, true] {
                    let shape = shapes
                        .iter()
                        .copied()
                        .find(|s| s.prefetch == prefetch)
                        .expect("prefetch shape");
                    let multiply = if idx % 2 == 0 {
                        DType::F16
                    } else {
                        DType::BF16
                    };
                    let types = MatrixTypes {
                        multiply,
                        ..MatrixTypes::F32_AS_F16
                    };
                    let epi = (prefetch && idx % 2 == 0).then_some(&relu);
                    let mut module = generate_workgroup_matmul_for_device(
                        group,
                        shape,
                        types,
                        native,
                        subgroup,
                        idx % 2 == 1,
                        epi,
                    );
                    super::super::validate_shader_for_test(
                        &mut module.module,
                        capabilities,
                        (1, 6),
                    )
                    .unwrap_or_else(|error| {
                        panic!(
                            "{native:?}/{subgroup} {group:?} {shape:?} epi={}: {error}",
                            epi.is_some()
                        )
                    });
                }
            }
        }
    }

    /// GPU f64-parity test for double-buffered `prefetch` subgroup matmul
    /// (NN, AT, BT, ragged rows, and fused ReLU epilogue).
    #[test]
    fn subgroup_double_buffered_prefetch_and_epilogue_match_f64() {
        let gpu = std::sync::Arc::new(crate::init_gpu_context().expect("gpu"));
        let caps = gpu.capabilities();
        let Some((native, subgroup)) = native_subgroup_shape(&caps, DType::F16) else {
            return;
        };
        if !crate::runtime::typed_matrix_supported(&gpu, DType::F16) {
            return;
        }
        let shape = WorkgroupMatmulShape::all_for(native, subgroup)
            .find(|s| s.subgroup_rows != 0 && s.prefetch && s.k_stage == 32)
            .expect("double-buffered subgroup shape");
        let [m, k, n] = [
            shape.rows as usize + 12,
            128usize,
            shape.columns as usize * 2,
        ];
        let mut g = crate::Graph::new();
        let a = g.input("a", &[m, k]);
        let b = g.input("b", &[k, n]);
        let y = g.matmul(a, b);
        let z = g.relu(y);
        g.set_outputs(vec![z]);
        let a_data: Vec<f32> = (0..m * k)
            .map(|i| (((i * 37 + 11) % 97) as f32 - 48.0) * 0.02)
            .collect();
        let b_data: Vec<f32> = (0..k * n)
            .map(|i| (((i * 29 + 5) % 89) as f32 - 44.0) * 0.02)
            .collect();
        let mut ref_out = vec![0.0f64; m * n];
        for r in 0..m {
            for c in 0..n {
                let mut sum = 0.0f64;
                for t in 0..k {
                    sum += f64::from(a_data[r * k + t]) * f64::from(b_data[t * n + c]);
                }
                ref_out[r * n + c] = sum.max(0.0);
            }
        }
        let plan = crate::compile::compile(&g);
        let opts = crate::SessionOptions {
            coop: crate::CoopPolicy::AllowF16,
            workgroup_matmul: Some(shape),
            ..Default::default()
        };
        let mut session = crate::Session::with_context_opts(plan, gpu, opts);
        assert!(
            session.plan().dispatches.iter().any(|d| matches!(
                d.kernel,
                crate::compile::Kernel::CooperativeWorkgroup { .. }
            )),
            "expected subgroup workgroup kernel with fused ReLU epilogue"
        );
        session.set_input("a", &a_data);
        session.set_input("b", &b_data);
        session.step();
        session.wait();
        let got = session.read_output(m * n);
        let max_err = got
            .iter()
            .zip(&ref_out)
            .fold(0.0f64, |acc, (&g, &r)| acc.max((f64::from(g) - r).abs()));
        assert!(
            max_err < 1e-2,
            "double-buffered subgroup matmul + ReLU epilogue diverged: max_err={max_err}"
        );
    }

    /// Subgroup tiles in every layout, storage, output and row-bounds case.
    #[test]
    fn subgroup_typed_modules_validate_and_emit_spirv() {
        let capabilities = naga::valid::Capabilities::SHADER_BFLOAT16
            | naga::valid::Capabilities::SHADER_FLOAT16
            | naga::valid::Capabilities::COOPERATIVE_MATRIX
            | naga::valid::Capabilities::SUBGROUP;
        let shapes = [
            (2, 2, 2, 2, 32),
            (4, 1, 1, 4, 64),
            (1, 4, 4, 2, 32),
            (2, 2, 1, 1, 16),
        ]
        .map(|(wr, wc, tr, tc, k_stage)| WorkgroupMatmulShape {
            rows: wr * tr * 16,
            columns: wc * tc * 16,
            k_stage,
            threads: wr * wc * SUBGROUP_WIDTH,
            stage_operands: false,
            subgroup_rows: wr,
            prefetch: false,
        });
        let mut index = 0;
        for group in [
            ShaderGroup::MatMul,
            ShaderGroup::MatMulAT,
            ShaderGroup::MatMulBT,
            ShaderGroup::MatMulAdd,
            ShaderGroup::MatMulATAdd,
            ShaderGroup::MatMulBTAdd,
        ] {
            for multiply in [DType::F16, DType::BF16] {
                for (a_f32, b_f32) in [(true, true), (false, true), (true, false), (false, false)] {
                    for row_bounds in [false, true] {
                        let operand = |f32| if f32 { DType::F32 } else { multiply };
                        let types = MatrixTypes {
                            a: operand(a_f32),
                            b: operand(b_f32),
                            multiply,
                            output: [DType::F32, multiply][index % 2],
                        };
                        let shape = shapes[index % shapes.len()];
                        index += 1;
                        assert!(shape.subgroup_grid().is_some());
                        let mut module = generate_workgroup_matmul(group, shape, types, row_bounds);
                        super::super::validate_shader_for_test(
                            &mut module.module,
                            capabilities,
                            (1, 6),
                        )
                        .unwrap_or_else(|error| panic!("{group:?} {types:?} {shape:?}: {error}"));
                    }
                }
            }
        }
    }
}
