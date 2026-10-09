use super::{MatMulCoopVariant, ShaderGroup, ShaderModule, coop_shape, matmul_module, preprocess};

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
    let (addend, variant) = coop_shape(group).expect("tiled cooperative matrix group");
    assert!(
        !addend || prologue.is_none(),
        "prologue layout has no addend binding"
    );
    let rows = CooperativeMatmulShape::ROWS;
    let lanes = CooperativeMatmulShape::THREADS;
    let cols = shape.columns;
    let stage = shape.k_stage;
    let stride_a = stage + 4;
    let stride_b = cols + 16;
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
                "(tile_row + wr * 32u + {}u) * n + tile_col + wc * {wave_cols}u + {}u",
                i * 16,
                j * 16
            );
            add_loads += &format!(
                "let ci{i}_{j} = {index};\nacc{i}_{j} = coopLoadT<coop_mat16x16<f32,C>>(&src[ci{i}_{j}], n);\n"
            );
            stores += &format!(
                "let oi{i}_{j} = output_base + {index};\ncoopStoreT(acc{i}_{j}, &$C_BUFFER[oi{i}_{j}], n);\n"
            );
        }
    }
    if addend {
        init += &format!("if wgid.z == 0u {{\n{add_loads}}}\n");
    }
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
                "let {prefix}r{part} = (lid.x + {}u) / {vstride}u;\nlet {prefix}c{part} = ((lid.x + {}u) % {vstride}u) * 4u;\nvar {prefix}v{part} = vec4<f32>(0.0);\n",
                part * lanes,
                part * lanes
            );
            let index = match (a, transposed) {
                (true, false) => format!("((tile_row + ar{part}) * k + t + ac{part}) / 4u"),
                (true, true) => format!("((t + ar{part}) * m + tile_row + ac{part}) / 4u"),
                (false, false) => format!("((t + br{part}) * n + tile_col + bc{part}) / 4u"),
                (false, true) => format!("((tile_col + br{part}) * k + t + bc{part}) / 4u"),
            };
            loads += &format!("{prefix}v{part} = {buffer}[{index}];\n");
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
        }
    }
    let mut multiply = String::new();
    for q in 0..stage / 16 {
        for i in 0..2 {
            multiply += &format!(
                "let ai{i}_{q} = (wr * 32u + {}u) * {stride_a}u + {}u;\nlet a{i}_{q} = coopLoadT<coop_mat16x16<f32,A>>(&sa[ai{i}_{q}], {stride_a}u);\n",
                i * 16,
                q * 16
            );
        }
        for j in 0..wave_cols / 16 {
            multiply += &format!(
                "let bi{j}_{q} = {}u + wc * {wave_cols}u + {}u;\nlet b{j}_{q} = coopLoadT<coop_mat16x16<f32,B>>(&sb[bi{j}_{q}], {stride_b}u);\n",
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
    let source = preprocess(
        include_str!("../shaders/matmul_coop_tiled.wgsl"),
        &[
            ("$COLUMNS", &cols.to_string()),
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
            ("$VARIABLES", &variables),
            ("$FIRST_LOAD", &first_load),
            ("$STEP_LOAD", &step_load),
            ("$NEXT_LOAD", &next_load),
            ("$WRITES", &writes),
            ("$MULTIPLY", &multiply),
            ("$STORES", &stores),
        ],
    );
    matmul_module(&source, "array<vec4<f32>>", Some(lanes), 1)
}
