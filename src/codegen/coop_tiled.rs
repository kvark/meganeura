use super::workgroup::{
    SubgroupLayout, WorkgroupMatmulShape, checked_subgroup_kernel_width, subgroup_kernel,
};
use super::{MatrixTypes, ShaderGroup, ShaderModule, coop_shape};

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
        // Matches `subgroup_kernel`'s 8-element bank padding (doubled for
        // double-buffered `prefetch`).
        let stages = if self.prefetch { 2 } else { 1 };
        stages * 4 * (Self::ROWS * (self.k_stage + 8) + self.k_stage * (self.columns + 8))
    }
}

/// Generate an aligned native 16x16 f32 matrix product, optionally partitioned
/// along K, lowered through the unified subgroup-tiled product kernel and
/// specialized to `subgroup`-wide subgroups (8, 16, 32, or 64; `0` defaults to
/// 32). Partitions write compact M*N slices; the caller must sum them in a
/// separate dispatch. An addend seeds partition zero only.
pub fn generate_tiled_coop_matmul(
    group: ShaderGroup,
    shape: CooperativeMatmulShape,
    splits: u32,
    prologue: Option<&crate::compile::MatMulPrologue>,
    subgroup: u32,
) -> ShaderModule {
    assert!(shape.legal() && (1..=65535).contains(&splits));
    let (addend, variant) = coop_shape(group).expect("tiled cooperative matrix group");
    assert!(
        !addend || prologue.is_none(),
        "prologue layout has no addend binding"
    );
    let subgroup = checked_subgroup_kernel_width(subgroup);
    let wg_shape = WorkgroupMatmulShape {
        rows: CooperativeMatmulShape::ROWS,
        columns: shape.columns,
        k_stage: shape.k_stage,
        threads: 4 * subgroup,
        stage_operands: false,
        subgroup_rows: 2,
        prefetch: shape.prefetch,
    };
    let mut header = String::from(
        "struct Params { m: u32, n: u32, k: u32, _pad: u32 }\n\
         var<storage> matrix_a: array<vec4<f32>>;\n\
         var<storage> matrix_b: array<vec4<f32>>;\n\
         var<storage, read_write> matrix_c: array<f32>;\n",
    );
    if addend {
        header += "var<storage> src: array<f32>;\n";
    }
    header += "var<uniform> params: Params;\n";
    subgroup_kernel(
        wg_shape,
        MatrixTypes::F32,
        [16, 16, 16],
        subgroup,
        &SubgroupLayout {
            variant,
            header,
            dimensions: "let m = params.m;\nlet n = params.n;\nlet k = params.k;".into(),
            a: "matrix_a",
            b: "matrix_b",
            c: "matrix_c",
            addend: addend.then_some("src"),
            prologue,
            epilogue: None,
            images: None,
            fixed_splits: Some(splits),
            unaligned_loads: false,
            copies: 1,
            row_bounds: false,
            column_bounds: false,
            gather: [None, None],
            scatter: None,
        },
    )
}
