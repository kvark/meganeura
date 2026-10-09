//! Shader codegen via WGSL templates.
//!
//! Shaders are written as `.wgsl` files in `src/shaders/` and parsed at
//! runtime by the naga WGSL frontend. The `preprocess()` helper performs
//! `$VAR` substitution for parameterized shaders before parsing.
//!
//! Modules are passed directly to blade via `naga_module` for SPIR-V
//! compilation.

use crate::compile::WeightFormat;
use naga::Module;

mod coop_tiled;
pub use coop_tiled::{CooperativeMatmulShape, generate_tiled_coop_matmul};

/// Forward attention staging and lane layout, independent of the EPT cap.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct FlashAttentionShape {
    pub threads: u32,
    pub keys: u32,
    pub interleave: bool,
}

impl Default for FlashAttentionShape {
    fn default() -> Self {
        Self {
            threads: 256,
            keys: 8,
            interleave: false,
        }
    }
}

impl FlashAttentionShape {
    pub(crate) fn fit_shared_memory(&mut self, head_dim: u32, bytes: u32) {
        if bytes == 0 {
            return;
        }
        while self.keys > 1 && self.shared_bytes(head_dim) > u64::from(bytes) {
            self.keys /= 2;
        }
        assert!(
            self.shared_bytes(head_dim) <= u64::from(bytes),
            "attention staging exceeds device shared memory"
        );
    }

    pub(crate) fn shared_bytes(self, head_dim: u32) -> u64 {
        let (keys, threads, head_dim) = (
            u64::from(self.keys),
            u64::from(self.threads),
            u64::from(head_dim),
        );
        4 * (2 * keys * head_dim + keys * threads + threads)
    }
}

/// How a K-split GEMV combines the per-lane partial sums.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum GemvReduction {
    /// Halving tree through workgroup memory, one `workgroupBarrier` per
    /// level. Portable and the historical default.
    Tree,
    /// One `subgroupAdd` per wave, then a single partial per wave through
    /// workgroup memory. The wave-wide add replaces `log2(wave)` tree levels
    /// and their barriers, so the wider the wave the more it removes — six
    /// levels on AMD's 64-wide wave against five on a 32-wide one.
    ///
    /// Subgroup IDs/counts index the partials without assuming a width or a
    /// mapping from local lanes. A one-subgroup workgroup needs no barrier.
    Subgroup,
}

/// Workgroup geometry and reduction style for the K-split GEMV family.
///
/// Plain, fused-add, reduced-storage and RMSNorm-folded kernels share the
/// width/reduction generator. Dense transposed-B kernels also share row
/// grouping. The winner depends on both device and shape; `Session::tune_with`
/// measures the alternatives without changing the arithmetic or bindings.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct GemvShape {
    /// K-reduction lanes per column group: 32, 64, 128 or 256.
    pub threads: u32,
    pub reduction: GemvReduction,
    /// Contiguous rows of transposed B per workgroup: 1, 2 or 4.
    /// Applies only to transposed B.
    #[serde(default = "GemvShape::one_row")]
    pub bt_rows: u32,
    /// Groups of four adjacent output columns in a forward workgroup.
    /// Grouped columns use the tree reduction and at most 256 total threads.
    #[serde(default = "GemvShape::one_row")]
    pub column_groups: u32,
}

impl GemvShape {
    /// Widths a K-split GEMV can be generated at.
    pub const WIDTHS: [u32; 4] = [32, 64, 128, 256];
    /// Contiguous row groups supported by the transposed-B GEMV source.
    pub const BT_ROWS: [u32; 3] = [1, 2, 4];

    fn one_row() -> u32 {
        1
    }

    pub(crate) fn for_group(mut self, group: ShaderGroup) -> Self {
        self.validate();
        if !matches!(
            group,
            ShaderGroup::MatMulGemvBT | ShaderGroup::MatMulGemvBTAdd
        ) {
            self.bt_rows = 1;
        } else {
            self.column_groups = 1;
        }
        self
    }

    /// The shape a group is generated with when nothing has chosen one.
    ///
    /// Every GEMV kernel is written at the width that suits its own access
    /// pattern — the plain form wide enough to hide DRAM latency at M=1, the
    /// fused-add and transposed forms narrow enough to stay coalesced — so
    /// the default is per group rather than one number.
    ///
    /// [`crate::compile::CompileOptions::gemv_shape`] replaces it, and
    /// measurement challenges whichever is in force.
    pub(crate) fn initial(group: ShaderGroup) -> Self {
        let threads = match group {
            ShaderGroup::MatMulGemv => 256,
            ShaderGroup::MatMulGemvAdd
            | ShaderGroup::MatMulGemvBT
            | ShaderGroup::MatMulGemvBTAdd => 32,
            _ => panic!("{group:?} is not a GEMV group"),
        };
        Self {
            threads,
            reduction: GemvReduction::Tree,
            bt_rows: 1,
            column_groups: 1,
        }
    }

    pub(crate) fn valid_columns(self) -> bool {
        self.column_groups == 1
            || (matches!(self.column_groups, 4 | 8)
                && self.reduction == GemvReduction::Tree
                && self.threads <= 256 / self.column_groups)
    }

    /// Reject geometry no GEMV source can be generated at.
    pub(crate) fn validate(self) {
        assert!(self.valid_columns(), "unsupported GEMV column grouping");
        assert!(
            Self::WIDTHS.contains(&self.threads),
            "GEMV workgroup width must be one of {:?}, got {}",
            Self::WIDTHS,
            self.threads
        );
        assert!(
            Self::BT_ROWS.contains(&self.bt_rows),
            "unsupported GEMV row count"
        );
    }
}

/// Configuration for cooperative matrix tile size and precision.
///
/// Derived from `blade_graphics::CooperativeMatrix` capabilities at runtime.
/// Determines which shader variant is generated for coop matmul.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CoopConfig {
    /// Cooperative matrix tile dimension (8 for Apple Silicon, 16 for RDNA3/Volta+).
    pub tile_size: u32,
    /// Use f16 input with f32 accumulators when true, or all-f32 when false.
    pub use_f16_input: bool,
    /// Split each f32 operand into `hi = f16(x)` and `lo = f16(x - f32(hi))`
    /// and accumulate `hi·hi + hi·lo + lo·hi` in f32 (Ootomo & Yokota).
    /// This improves mantissa precision but does not preserve f32's exponent
    /// range, so automatic selection never uses it for full-precision work.
    pub compensated: bool,
}

impl CoopConfig {
    /// Output tile per workgroup = 2 × tile_size (2×2 grid of coop tiles).
    pub fn output_tile(&self) -> u32 {
        2 * self.tile_size
    }

    /// Dense f32 8x8 cooperative matrices use four independent SIMD groups,
    /// each computing a 16x16 quadrant of a 32x32 workgroup tile.
    /// Convolution retains its separately generated two-by-two layout.
    pub fn matmul_output_tile(&self) -> u32 {
        if self.tile_size == 8 && !self.use_f16_input {
            32
        } else {
            self.output_tile()
        }
    }
}

/// Replace `$VAR` occurrences in `source` with the corresponding values.
/// Replacements run in order: insert fragments before filling their slots.
fn preprocess(source: &str, vars: &[(&str, &str)]) -> String {
    let mut s = source.to_string();
    for &(key, val) in vars {
        s = s.replace(key, val);
    }
    s
}

/// Named fragments in WGSL templates are delimited by `// @section NAME`.
fn template_section<'a>(source: &'a str, name: &str) -> &'a str {
    let marker = format!("// @section {name}\n");
    let (_, rest) = source
        .split_once(&marker)
        .unwrap_or_else(|| panic!("missing shader template section: {name}"));
    rest.split_once("// @section ")
        .map_or(rest, |(section, _)| section)
}

/// A parsed shader module together with the WGSL source text.
///
/// Blade needs the source for SPIR-V debug info (OpLine) in debug builds.
pub struct ShaderModule {
    pub module: Module,
    pub source: String,
    pub hint: &'static str,
}

impl ShaderModule {
    pub(crate) fn new(source: &str) -> Self {
        Self {
            module: parse_source(source).expect("WGSL parse failed"),
            source: source.to_string(),
            hint: "",
        }
    }

    /// Dump WGSL source into the configured directory.
    pub fn dump(&self, dir: &str) {
        let hint = if self.hint.is_empty() {
            self.module
                .entry_points
                .first()
                .map(|e| e.name.as_str())
                .unwrap_or("shader")
        } else {
            self.hint
        };

        let _ = std::fs::create_dir_all(dir);
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        std::hash::Hash::hash(&self.source, &mut hasher);
        let h = std::hash::Hasher::finish(&hasher);
        let path = std::path::Path::new(&dir).join(format!("{hint}_{h:x}.wgsl"));
        let _ = std::fs::write(path, self.source.as_bytes());
    }
}

fn parse_source(source: &str) -> Result<Module, naga::front::wgsl::ParseError> {
    let _span = tracing::info_span!("naga_parse", source_bytes = source.len()).entered();
    naga::front::wgsl::parse_str(source)
}

/// Generate WGSL for a [`crate::compile::MatMulEpilogue`].
/// Returns (declarations, body).
///
/// The DAG's `LoadInput(0)` maps to `val` (the matmul accumulator).
/// `LoadInput(1+)` maps to `epi_buf_{n}` indexed by either `idx`
/// (per-element) or `col` (per-column broadcast) based on `EpilogueLoadKind`.
pub fn matmul_epilogue_to_wgsl(epi: &crate::compile::MatMulEpilogue) -> (String, String) {
    use crate::compile::EpilogueLoadKind;

    let mut decls = Vec::new();
    for (i, _) in epi.inputs.iter().enumerate() {
        decls.push(format!("var<storage> epi_buf_{}: array<f32>;", i));
    }

    let body = epi.dag.emit_body(|idx| {
        if idx == 0 {
            "val".to_string()
        } else {
            let (_, ref kind) = epi.inputs[(idx - 1) as usize];
            let index_var = match *kind {
                EpilogueLoadKind::PerElement => "idx",
                EpilogueLoadKind::PerCol => "col",
            };
            format!("epi_buf_{}[{}]", idx - 1, index_var)
        }
    });

    // The DAG body emits `let v0 = ...; let v1 = ...; ...`. We need to
    // assign the final value back to `val`.
    let assign = format!("val = v{};", epi.dag.output);
    let full_body = format!("{}\n                {}", body.trim_end(), assign);

    (decls.join("\n"), full_body)
}

/// Generate WGSL for a [`crate::compile::MatMulPrologue`] — the multiplicative factors
/// applied during A-tile staging in the coop matmul.
///
/// Returns `(declarations, cache_decl, cache_init, transform_expression)`:
///   - `declarations` — `var<storage>` lines for prologue buffers (global).
///   - `cache_decl` — `var<workgroup>` lines for per-row caches (if any).
///   - `cache_init` — WG-entry code that reads PerRow factors into shared
///     memory, ending with a barrier. Empty when no PerRow factor exists.
///   - `transform_expression` — expression suffix. PerRow factors read from
///     shared cache (cheap); PerKCol stay global.
///
/// Caching PerRow factors eliminates the per-A-element global read that
/// caused the previous fused-prologue attempt to regress TTFT ~40%.
pub fn matmul_prologue_to_wgsl(
    prologue: &crate::compile::MatMulPrologue,
    output_tile: u32,
) -> (String, String, String, String) {
    use crate::compile::PrologueLoadKind;

    let mut decls = Vec::new();
    let mut cache_decls = Vec::new();
    let mut cache_inits = Vec::new();
    let mut expr = String::new();
    let mut has_per_row = false;
    #[allow(clippy::pattern_type_mismatch)]
    for (i, (_, kind)) in prologue.factors.iter().enumerate() {
        let buf_name = format!("prologue_buf_{}", i);
        decls.push(format!("var<storage> {}: array<f32>;", buf_name));
        match *kind {
            PrologueLoadKind::PerRow => {
                has_per_row = true;
                let cache_name = format!("prologue_cache_{}", i);
                cache_decls.push(format!(
                    "var<workgroup> {cache_name}: array<f32, {output_tile}>;"
                ));
                // Each thread in 0..output_tile loads one entry. For output_tile
                // ≤ 64 (wg_size), this is one load per thread max.
                cache_inits.push(format!(
                    "    if lid.x < {output_tile}u {{\n\
                     \x20       let gr = tile_row + lid.x;\n\
                     \x20       var factor = 0.0;\n\
                     \x20       if gr < m {{ factor = {buf_name}[gr]; }}\n\
                     \x20       {cache_name}[lid.x] = factor;\n\
                     \x20   }}"
                ));
                // Access: use shared cache with row index relative to tile_row.
                expr.push_str(&format!(" * {cache_name}[gr - tile_row]"));
            }
            PrologueLoadKind::PerKCol => {
                expr.push_str(&format!(" * {buf_name}[tc]"));
            }
        }
    }
    let cache_init = if has_per_row {
        format!("{}\n    workgroupBarrier();", cache_inits.join("\n"))
    } else {
        String::new()
    };
    (decls.join("\n"), cache_decls.join("\n"), cache_init, expr)
}

/// Tuning knobs for the register-tiled scalar matmul codegen.
///
/// Defaults are what the plain kernel ships with: 32-row K staging and
/// sequential columns. Resolved values come from `TuningKnobs`
/// (`crate::compile`) and travel with `MatMulOptions` — nothing reads a
/// process global.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MatmulKnobs {
    /// K staging depth: how many B rows each shared-memory stage loads
    /// before the accumulator loop consumes them. 8 | 16 | 32.
    pub k_stage: u32,
    /// Stagger the B loads across columns instead of tying a thread to
    /// `tm` consecutive ones.
    pub interleave_columns: bool,
    /// Emit the hardware `dot4I8Packed` in the int-dot GEMV kernels where
    /// the device reports `shader_integer_dot_product`; the exact scalar
    /// expansion is the fallback.
    pub integer_dot: bool,
    /// Straight-line K loop instead of a counted loop.
    pub unroll_k: bool,
}

impl Default for MatmulKnobs {
    fn default() -> Self {
        Self {
            k_stage: 32,
            interleave_columns: false,
            integer_dot: false,
            unroll_k: false,
        }
    }
}

/// Measured scalar layout for plain F32/F16 weights, with F32 accumulation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ScalarMatmulShape {
    /// Rows of C covered by one workgroup.
    pub tile_size: u32,
    /// Columns of C covered by one workgroup. Zero means a square block.
    #[serde(default)]
    pub tile_n: u32,
    pub k_stage: u32,
    pub interleave_columns: bool,
    /// Emit a straight-line K stage. The counted loop remains an alternative.
    #[serde(default)]
    pub unroll_k: bool,
}

impl ScalarMatmulShape {
    /// Rows of C covered by one workgroup.
    pub fn rows(self) -> u32 {
        self.tile_size
    }

    /// Columns of C covered by one workgroup.
    pub fn cols(self) -> u32 {
        if self.tile_n == 0 {
            self.tile_size
        } else {
            self.tile_n
        }
    }

    /// The staged tiles have to be an integer number of 256-thread passes,
    /// and each thread's register tile is BM/16 by BN/16.
    pub fn legal(self) -> bool {
        let (bm, bn, k) = (self.rows(), self.cols(), self.k_stage);
        matches!(k, 8 | 16 | 32)
            && matches!(bm, 16 | 32 | 64)
            && matches!(bn, 16 | 32 | 64)
            && bm * k % 256 == 0
            && bn * k % 256 == 0
    }

    pub fn geometry(self) -> MatMulTile {
        match (self.rows(), self.cols()) {
            (64, 64) => MatMulTile::Large,
            (32, 32) => MatMulTile::Small,
            (bm, bn) => MatMulTile::Rect { bm, bn },
        }
    }
}

/// How to specialize the matmul the epilogue is fused into.
///
/// [`Default`] is the plain f32 64×64 kernel, so a caller that only wants
/// an epilogue and nothing else can pass `Default::default()`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MatMulOptions {
    /// B-buffer storage format: drives its declaration and its load.
    pub format: WeightFormat,
    /// Tile geometry. Must match what the dispatch's workgroup count was
    /// computed for, or the grid and the kernel disagree about coverage.
    pub tile: MatMulTile,
    /// K staging depth and column layout for the tiled skeleton; only
    /// consulted for F32/F16 B storage; block-quantized formats have their own.
    pub knobs: MatmulKnobs,
}

/// Generate a matmul shader module with a fused epilogue chain.
///
/// Used by the runtime when a dispatch carries an epilogue. The ops are
/// compiled into WGSL statements that transform each output element
/// before it is stored. The epilogue never inspects B, so the same
/// `$STORE_BODY` hook serves every weight format.
pub fn generate_matmul_with_epilogue(
    group: ShaderGroup,
    epilogue: Option<&crate::compile::MatMulEpilogue>,
    options: MatMulOptions,
) -> ShaderModule {
    generate_partitioned_matmul(group, epilogue, options, 1, 1)
}

pub(crate) fn generate_split_matmul(
    group: ShaderGroup,
    shape: ScalarMatmulShape,
    splits: u32,
    format: WeightFormat,
    knobs: MatmulKnobs,
) -> ShaderModule {
    assert!(splits >= 2);
    assert!(group.is_matmul());
    assert!(matches!(format, WeightFormat::F32 | WeightFormat::F16));
    generate_partitioned_matmul(
        group,
        None,
        MatMulOptions {
            format,
            tile: shape.geometry(),
            knobs: MatmulKnobs {
                k_stage: shape.k_stage,
                interleave_columns: shape.interleave_columns,
                unroll_k: shape.unroll_k,
                ..knobs
            },
        },
        splits,
        1,
    )
}

fn generate_partitioned_matmul(
    group: ShaderGroup,
    epilogue: Option<&crate::compile::MatMulEpilogue>,
    options: MatMulOptions,
    splits: u32,
    copies: u32,
) -> ShaderModule {
    if matches!(group, ShaderGroup::MatMulBT | ShaderGroup::MatMulBTAdd)
        && options.format.is_quantized()
    {
        panic!(
            "matmul_bt does not support block-quantized weights: their blocks \
             run along the parameter's first dimension, which is N here, not K"
        );
    }
    let (epi_decl, epi_body) = epilogue.map(matmul_epilogue_to_wgsl).unwrap_or_default();
    let (a_idx, b_idx, fused_decl, fused_expr) = match group {
        ShaderGroup::MatMul => (MATMUL_A_FWD, MATMUL_B_FWD, "", ""),
        ShaderGroup::MatMulAdd => (
            MATMUL_A_FWD,
            MATMUL_B_FWD,
            "var<storage> src: array<f32>;",
            " + src[idx]",
        ),
        ShaderGroup::MatMulAT => (MATMUL_A_AT, MATMUL_B_FWD, "", ""),
        ShaderGroup::MatMulBT => (MATMUL_A_FWD, MATMUL_B_BT, "", ""),
        ShaderGroup::MatMulATAdd => (
            MATMUL_A_AT,
            MATMUL_B_FWD,
            "var<storage> src: array<f32>;",
            " + src[idx]",
        ),
        ShaderGroup::MatMulBTAdd => (
            MATMUL_A_FWD,
            MATMUL_B_BT,
            "var<storage> src: array<f32>;",
            " + src[idx]",
        ),
        _ => panic!("epilogue fusion not supported for {:?}", group),
    };
    let fused_expr = if splits > 1 && !fused_expr.is_empty() {
        " + select(0.0, src[select(0u, idx, split_id == 0u)], split_id == 0u)"
    } else {
        fused_expr
    };
    let (a_row, a_col, b_row, b_col) = matmul_stage_maps(group);
    matmul_vars_tiled(
        MatMulIndexing {
            a_idx,
            b_idx,
            a_row,
            a_col,
            b_row,
            b_col,
            tile_row: "wgid.y + wgid.z * params._pad",
            c_idx: "row * params.n + col",
        },
        fused_decl,
        fused_expr,
        &epi_decl,
        &epi_body,
        options,
        splits,
        copies,
    )
}

/// Thread-to-element staging maps. The widths are `$BM_U`, `$BN_U` and
/// `$K_TILE_U`, so one mapping covers every scalar matmul tile.
fn matmul_stage_maps(
    group: ShaderGroup,
) -> (&'static str, &'static str, &'static str, &'static str) {
    let a_transposed = matches!(group, ShaderGroup::MatMulAT | ShaderGroup::MatMulATAdd);
    let b_transposed = matches!(group, ShaderGroup::MatMulBT | ShaderGroup::MatMulBTAdd);
    let (a_row, a_col) = if a_transposed {
        (A_ROW_AT, A_COL_AT)
    } else {
        (A_ROW_FWD, A_COL_FWD)
    };
    let (b_row, b_col) = if b_transposed {
        (B_ROW_BT, B_COL_BT)
    } else {
        (B_ROW_FWD, B_COL_FWD)
    };
    (a_row, a_col, b_row, b_col)
}

// ---------------------------------------------------------------------------
// Shader groups — each group is a naga::Module with one or more entry points
// ---------------------------------------------------------------------------

/// A shader group corresponds to a single `naga::Module` that may
/// contain multiple entry points (e.g. `Unary` has relu, sigmoid, neg).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShaderGroup {
    /// Generated pointwise and reduction kernels, lowered from their
    /// dispatch's `kernel` rather than from a module of this group.
    Generated,
    Sgd,
    Adam,
    Transpose,
    MatMul,
    MatMulAdd,
    MatMulAT,
    MatMulBT,
    BlockMatMul,
    BlockMatMulAT,
    BlockMatMulBT,
    BatchMatMul,
    BatchMatMulAT,
    BatchMatMulBT,
    MatMulATAdd,
    MatMulBTAdd,
    /// M=1 matmul (GEMV): `C[1,N] = A[1,K] × B[K,N]`. One thread per
    /// output column; dispatched for batch-1 decode on transformers.
    MatMulGemv,
    /// M=1 matmul with fused residual add: `C[1,N] = A×B + D[1,N]`.
    /// Same shape as MatMulGemv plus one extra storage input.
    MatMulGemvAdd,
    /// M=1 MatMulBT (`B` stored `[N,K]`): `C[1,N] = A × Bᵀ`. K-split with
    /// coalesced contiguous-K vec4 loads.
    MatMulGemvBT,
    MatMulGemvBTAdd,
    Reduce,
    CrossEntropy,
    RmsNormAdd,
    CachedBlockAttentionSplit,
    CachedBlockAttentionCombine,
    Embedding,
    ToF16,
    RoPE,
    RoPEGrad,
    LayerNorm,
    MultiHeadAttn,
    /// Flash Attention 2 forward: BQ>1 multi-query tiling with shared K staging.
    FlashAttention,
    /// Cooperative-matrix flash attention forward.
    /// Phase 1: coop_mat for QK^T, scalar softmax + PV. Opt-in via
    /// `MEGANEURA_FLASH_FWD_COOP=1`.
    FlashAttentionCoop,
    /// Attention backward, generated by its kernel family.
    AttentionGrad(crate::kernels::attention_grad::AttentionGrad),
    SwiGLUGrad,
    SwiGLUConcat,
    SumRows,
    RmsNormGrad,
    RmsNormGradWRowPar,
    LayerNormGrad,
    ScatterAdd,
    ScatterAddAtomic,
    BceLoss,
    RmsNormRsqrt,
    GroupNorm,
    GroupNormGrad,
    Concat,
    Split,
    Permute,
    Upsample,
    UpsampleGrad,
    /// Depthwise Conv2d (groups == channels). Used by EfficientNet MBConv.
    Conv2dDw,
    /// Per-channel broadcast mul: `dst[n,c,h,w] = src[n,c,h,w] * gate[n,c]`.
    /// Used by EfficientNet Squeeze-and-Excitation.
    MulPerChannel,
    Conv2dGemm,
    Conv2dGemmSmall,
    Conv2dGemm16,
    Conv2dGemmCoop,
    Conv2dGradInputGemm,
    Conv2dGradInputGemmSmall,
    Conv2dGradInputGemm16,
    Conv2dGradInputGemmCoop,
    GroupNormSilu,
    WinogradInputTransform,
    WinogradOutputTransform,
    WinogradBatchedMatMul,
    WinogradWeightTransform,
    Conv2dGradWeightGemm,
    Conv2dGradWeightGemmSmall,
    Conv2dGradWeightGemm16,
    Conv2dGradWeightGemmSplit,
    Conv2dGradWeightGemmSplitSmall,
    Conv2dGradWeightGemmSplit16,
    CacheWrite,
    CacheWritePrefix,
    CachedAttention,
    BiasedAttention,
    CachedQueryAttention,
    CachedBlockAttention,
    ChunkedRelativeAttention,
    PrefixLast,
    RoPEDynamic,
    MaxPool2d,
    MaxPool2dGrad,
    GlobalAvgPool,
    GlobalAvgPoolGrad,
    PairwiseGrad,
    GradClipNormSq,
    GradClipScale,
    AdaptiveGradClip,
    GradAccum,
}

impl ShaderGroup {
    pub const fn is_matmul(&self) -> bool {
        matches!(
            self,
            Self::MatMul
                | Self::MatMulAdd
                | Self::MatMulAT
                | Self::MatMulBT
                | Self::MatMulATAdd
                | Self::MatMulBTAdd
        )
    }
}

/// Generate a `naga::Module` for a shader group.
pub fn generate_module(group: ShaderGroup, knobs: MatmulKnobs) -> ShaderModule {
    match group {
        ShaderGroup::Generated => {
            unreachable!("generated kernels are lowered from their dispatch's kernel")
        }
        ShaderGroup::Sgd => optimizer_module(include_str!("shaders/sgd.wgsl")),
        ShaderGroup::Adam => optimizer_module(include_str!("shaders/adam.wgsl")),
        ShaderGroup::Transpose => ShaderModule::new(include_str!("shaders/transpose.wgsl")),
        ShaderGroup::MatMul
        | ShaderGroup::MatMulAdd
        | ShaderGroup::MatMulAT
        | ShaderGroup::MatMulBT
        | ShaderGroup::MatMulATAdd
        | ShaderGroup::MatMulBTAdd => generate_matmul_with_epilogue(
            group,
            None,
            MatMulOptions {
                knobs,
                ..Default::default()
            },
        ),
        ShaderGroup::BlockMatMul
        | ShaderGroup::BlockMatMulAT
        | ShaderGroup::BlockMatMulBT
        | ShaderGroup::BatchMatMul
        | ShaderGroup::BatchMatMulAT
        | ShaderGroup::BatchMatMulBT => gen_block_matmul(group, MatMulTile::Large),
        ShaderGroup::MatMulGemv
        | ShaderGroup::MatMulGemvAdd
        | ShaderGroup::MatMulGemvBT
        | ShaderGroup::MatMulGemvBTAdd => {
            generate_module_gemv(group, WeightFormat::F32, GemvShape::initial(group))
        }
        ShaderGroup::Reduce => ShaderModule::new(include_str!("shaders/reduce.wgsl")),
        ShaderGroup::CrossEntropy => ShaderModule::new(include_str!("shaders/cross_entropy.wgsl")),
        ShaderGroup::RmsNormAdd => generate_rms_norm_add_module(),
        ShaderGroup::CachedBlockAttentionSplit | ShaderGroup::CachedBlockAttentionCombine => {
            generate_cached_attention_module(group, None)
        }
        ShaderGroup::Embedding => ShaderModule::new(include_str!("shaders/embedding.wgsl")),
        ShaderGroup::ToF16 => ShaderModule::new(include_str!("shaders/to_f16.wgsl")),
        ShaderGroup::RoPE => ShaderModule::new(include_str!("shaders/rope.wgsl")),
        ShaderGroup::RoPEGrad => ShaderModule::new(include_str!("shaders/rope_grad.wgsl")),
        ShaderGroup::LayerNorm => ShaderModule::new(include_str!("shaders/layer_norm.wgsl")),
        ShaderGroup::MultiHeadAttn => {
            // Default head_dim=64 fallback; runtime calls
            // generate_attention_module(head_dim) directly for the actual value.
            generate_attention_module(64)
        }
        ShaderGroup::FlashAttention => {
            // Default head_dim=64 fallback; runtime calls
            // generate_flash_attention_module(head_dim) directly.
            generate_flash_attention_module(
                64,
                crate::compile::TuningKnobs::default().flash_ept_cap,
                FlashAttentionShape::default(),
            )
        }
        ShaderGroup::FlashAttentionCoop => generate_flash_attention_coop_module(64),
        ShaderGroup::AttentionGrad(kernel) => {
            // Default head_dim=64; runtime generates for the actual width.
            kernel.generate(64, None, &crate::compile::TuningKnobs::default())
        }
        ShaderGroup::SwiGLUGrad => ShaderModule::new(include_str!("shaders/swiglu_grad.wgsl")),
        ShaderGroup::SwiGLUConcat => ShaderModule::new(include_str!("shaders/swiglu_concat.wgsl")),
        ShaderGroup::SumRows => ShaderModule::new(include_str!("shaders/sum_rows.wgsl")),
        ShaderGroup::RmsNormGrad => ShaderModule::new(include_str!("shaders/rms_norm_grad.wgsl")),
        ShaderGroup::RmsNormGradWRowPar => {
            ShaderModule::new(include_str!("shaders/rms_norm_grad_w_rowpar.wgsl"))
        }
        ShaderGroup::LayerNormGrad => {
            ShaderModule::new(include_str!("shaders/layer_norm_grad.wgsl"))
        }
        ShaderGroup::RmsNormRsqrt => ShaderModule::new(include_str!("shaders/rms_norm_rsqrt.wgsl")),
        ShaderGroup::ScatterAdd => ShaderModule::new(include_str!("shaders/scatter_add.wgsl")),
        ShaderGroup::ScatterAddAtomic => {
            ShaderModule::new(include_str!("shaders/scatter_add_atomic.wgsl"))
        }
        ShaderGroup::BceLoss => ShaderModule::new(include_str!("shaders/bce.wgsl")),
        ShaderGroup::GroupNorm => ShaderModule::new(include_str!("shaders/group_norm.wgsl")),
        ShaderGroup::GroupNormGrad => {
            ShaderModule::new(include_str!("shaders/group_norm_grad.wgsl"))
        }
        ShaderGroup::Concat => ShaderModule::new(include_str!("shaders/concat.wgsl")),
        ShaderGroup::Split => ShaderModule::new(include_str!("shaders/split.wgsl")),
        ShaderGroup::Permute => ShaderModule::new(include_str!("shaders/permute.wgsl")),
        ShaderGroup::BiasedAttention => {
            ShaderModule::new(include_str!("shaders/biased_attention.wgsl"))
        }
        ShaderGroup::Upsample => ShaderModule::new(include_str!("shaders/upsample.wgsl")),
        ShaderGroup::UpsampleGrad => ShaderModule::new(include_str!("shaders/upsample_grad.wgsl")),
        ShaderGroup::Conv2dDw => ShaderModule::new(include_str!("shaders/conv2d_dw.wgsl")),
        ShaderGroup::MulPerChannel => {
            ShaderModule::new(include_str!("shaders/mul_per_channel.wgsl"))
        }
        ShaderGroup::Conv2dGemm
        | ShaderGroup::Conv2dGemmSmall
        | ShaderGroup::Conv2dGemm16
        | ShaderGroup::Conv2dGradInputGemm
        | ShaderGroup::Conv2dGradInputGemmSmall
        | ShaderGroup::Conv2dGradInputGemm16 => generate_conv_module(group, 16, None),
        ShaderGroup::Conv2dGemmCoop | ShaderGroup::Conv2dGradInputGemmCoop => {
            panic!(
                "conv coop kernels are generated per (kernel, stride) via generate_conv2d_coop_module"
            )
        }
        ShaderGroup::GroupNormSilu => {
            ShaderModule::new(include_str!("shaders/group_norm_silu.wgsl"))
        }
        ShaderGroup::WinogradInputTransform => {
            ShaderModule::new(include_str!("shaders/winograd_input_transform.wgsl"))
        }
        ShaderGroup::WinogradOutputTransform => {
            ShaderModule::new(include_str!("shaders/winograd_output_transform.wgsl"))
        }
        ShaderGroup::WinogradBatchedMatMul => {
            ShaderModule::new(include_str!("shaders/winograd_matmul.wgsl"))
        }
        ShaderGroup::WinogradWeightTransform => {
            ShaderModule::new(include_str!("shaders/winograd_weight_transform.wgsl"))
        }
        ShaderGroup::Conv2dGradWeightGemm => {
            conv_grad_weight_tiled(MatMulTile::Large, false, 16, None)
        }
        ShaderGroup::Conv2dGradWeightGemmSmall => {
            conv_grad_weight_tiled(MatMulTile::Small, false, 16, None)
        }
        ShaderGroup::Conv2dGradWeightGemm16 => {
            conv_grad_weight_tiled(MatMulTile::Occupancy, false, 16, None)
        }
        ShaderGroup::Conv2dGradWeightGemmSplit => {
            conv_grad_weight_tiled(MatMulTile::Large, true, 16, None)
        }
        ShaderGroup::Conv2dGradWeightGemmSplitSmall => {
            conv_grad_weight_tiled(MatMulTile::Small, true, 16, None)
        }
        ShaderGroup::Conv2dGradWeightGemmSplit16 => {
            conv_grad_weight_tiled(MatMulTile::Occupancy, true, 16, None)
        }
        ShaderGroup::CacheWrite => ShaderModule::new(include_str!("shaders/cache_write.wgsl")),
        ShaderGroup::CacheWritePrefix => {
            ShaderModule::new(include_str!("shaders/cache_write_prefix.wgsl"))
        }
        ShaderGroup::CachedAttention => {
            ShaderModule::new(include_str!("shaders/cached_attention.wgsl"))
        }
        ShaderGroup::CachedQueryAttention => generate_cached_query_attention_module(64),
        ShaderGroup::CachedBlockAttention => generate_module_block_attention(),
        ShaderGroup::ChunkedRelativeAttention => {
            ShaderModule::new(include_str!("shaders/chunked_relative_attention.wgsl"))
        }
        ShaderGroup::PrefixLast => ShaderModule::new(include_str!("shaders/prefix_last.wgsl")),
        ShaderGroup::RoPEDynamic => ShaderModule::new(include_str!("shaders/rope_dynamic.wgsl")),
        ShaderGroup::MaxPool2d => ShaderModule::new(include_str!("shaders/max_pool_2d.wgsl")),
        ShaderGroup::MaxPool2dGrad => {
            ShaderModule::new(include_str!("shaders/max_pool_2d_grad.wgsl"))
        }
        ShaderGroup::GlobalAvgPool => {
            ShaderModule::new(include_str!("shaders/global_avg_pool.wgsl"))
        }
        ShaderGroup::GlobalAvgPoolGrad => {
            ShaderModule::new(include_str!("shaders/global_avg_pool_grad.wgsl"))
        }
        ShaderGroup::PairwiseGrad => ShaderModule::new(include_str!("shaders/pairwise_grad.wgsl")),
        ShaderGroup::GradClipNormSq => {
            optimizer_module(include_str!("shaders/grad_clip_norm_sq.wgsl"))
        }
        ShaderGroup::GradClipScale => {
            optimizer_module(include_str!("shaders/grad_clip_scale.wgsl"))
        }
        ShaderGroup::AdaptiveGradClip => {
            optimizer_module(include_str!("shaders/adaptive_grad_clip.wgsl"))
        }
        ShaderGroup::GradAccum => optimizer_module(include_str!("shaders/grad_accum.wgsl")),
    }
}

fn generate_rms_norm_add_module() -> ShaderModule {
    let source = preprocess(
        include_str!("shaders/rms_norm.wgsl"),
        &[(
            "$EPILOGUE",
            include_str!("shaders/rms_norm_add_epilogue.wgsl"),
        )],
    );
    ShaderModule::new(&source)
}

/// The cooperative-matrix form of a matmul group: whether it fuses a
/// residual add, and which operand (if any) is transposed.
///
/// `None` for groups that have no cooperative form. This is the single
/// definition of that mapping - the module, prologue and epilogue
/// generators all route through it.
pub(crate) fn coop_shape(group: ShaderGroup) -> Option<(bool, MatMulCoopVariant)> {
    Some(match group {
        ShaderGroup::MatMul => (false, MatMulCoopVariant::Normal),
        ShaderGroup::MatMulAdd => (true, MatMulCoopVariant::Normal),
        ShaderGroup::MatMulAT => (false, MatMulCoopVariant::AT),
        ShaderGroup::MatMulATAdd => (true, MatMulCoopVariant::AT),
        ShaderGroup::MatMulBT => (false, MatMulCoopVariant::BT),
        ShaderGroup::MatMulBTAdd => (true, MatMulCoopVariant::BT),
        _ => return None,
    })
}

/// Generate the cooperative-matrix form of a shader group, with the given
/// tile config. Takes the scalar group: cooperative execution is a
/// modifier on a dispatch, not a group of its own.
pub fn generate_module_coop(group: ShaderGroup, config: &CoopConfig) -> ShaderModule {
    let (fused_add, variant) =
        coop_shape(group).unwrap_or_else(|| panic!("no cooperative form for {group:?}"));
    gen_matmul_coop_wgsl(fused_add, variant, config)
}

/// Pack `count` same-A matmuls into one dispatch (`workgroups.z = count`).
/// Each z-slice uses `matrix_b{i}` / `matrix_c{i}`.
pub fn generate_horizontal_matmul(
    group: ShaderGroup,
    count: u32,
    coop: Option<&CoopConfig>,
) -> ShaderModule {
    assert!((2..=3).contains(&count));
    match coop {
        Some(config) => {
            let (fused_add, variant) =
                coop_shape(group).unwrap_or_else(|| panic!("no cooperative form for {group:?}"));
            gen_matmul_coop_wgsl_full(fused_add, variant, config, None, None, count, 1)
        }
        None => generate_partitioned_matmul(group, None, MatMulOptions::default(), 1, count),
    }
}

/// Compose declarations and kernel functions through named slots, before parsing.
fn matmul_module(
    source: &str,
    b_storage: &str,
    coop_threads: Option<u32>,
    count: u32,
) -> ShaderModule {
    let interface = include_str!("shaders/matmul_entry.wgsl");
    let workgroup_size = coop_threads.map_or_else(|| "16, 16".to_owned(), |n| n.to_string());
    let coop = coop_threads.is_some();
    let entry = |name: &str, compute: bool| {
        preprocess(
            template_section(
                interface,
                if compute {
                    "entry_signature"
                } else {
                    "helper_signature"
                },
            ),
            &[
                ("$NAME", name),
                ("$WORKGROUP_SIZE", &workgroup_size),
                (
                    "$SUBGROUP",
                    if !coop {
                        ""
                    } else if compute {
                        ", @builtin(subgroup_id) sg: u32, @builtin(subgroup_size) sg_size: u32"
                    } else {
                        ", sg: u32, sg_size: u32"
                    },
                ),
            ],
        )
    };
    let mut bindings = String::new();
    let mut kernels = String::new();
    let mut calls = String::new();
    for i in 0..count {
        let suffix = if count == 1 {
            String::new()
        } else {
            i.to_string()
        };
        let b = format!("matrix_b{suffix}");
        let c = format!("matrix_c{suffix}");
        bindings.push_str(&preprocess(
            template_section(interface, "bindings"),
            &[
                ("$B_BUFFER", &b),
                ("$C_BUFFER", &c),
                ("$B_STORAGE", b_storage),
            ],
        ));
        let name = if count == 1 {
            "main".to_owned()
        } else {
            format!("horiz_{i}")
        };
        kernels.push_str(&preprocess(
            template_section(source, "kernel"),
            &[
                ("$ENTRY_SIGNATURE", &entry(&name, count == 1)),
                ("$B_BUFFER", &b),
                ("$C_BUFFER", &c),
            ],
        ));
        if count > 1 {
            let condition = if i == 0 {
                format!("if wgid.z == {i}u")
            } else if i + 1 == count {
                "else".to_owned()
            } else {
                format!("else if wgid.z == {i}u")
            };
            calls.push_str(&preprocess(
                template_section(interface, "call"),
                &[
                    ("$CONDITION", &condition),
                    ("$NAME", &name),
                    ("$SUBGROUP_ARG", if coop { ", sg, sg_size" } else { "" }),
                ],
            ));
        }
    }
    let mut source = preprocess(
        template_section(source, "header"),
        &[("$MATRIX_BINDINGS", &bindings)],
    );
    source.push_str(&kernels);
    if count > 1 {
        source.push_str(&preprocess(
            template_section(interface, "dispatch"),
            &[
                ("$ENTRY_SIGNATURE", &entry("main", true)),
                ("$CALLS", &calls),
            ],
        ));
    }
    ShaderModule::new(&source)
}

/// Generate WGSL source for a shader group.
pub fn generate_wgsl(group: ShaderGroup) -> String {
    let sm = generate_module(group, MatmulKnobs::default());
    let capabilities = match group {
        ShaderGroup::Conv2dGemmCoop | ShaderGroup::Conv2dGradInputGemmCoop => {
            naga::valid::Capabilities::COOPERATIVE_MATRIX
                | naga::valid::Capabilities::SHADER_FLOAT16
                | naga::valid::Capabilities::SUBGROUP
        }
        ShaderGroup::ToF16 => naga::valid::Capabilities::SHADER_FLOAT16,
        _ => naga::valid::Capabilities::empty(),
    };
    module_to_wgsl(&sm.module, capabilities)
}

/// Convert a naga Module to WGSL source text.
pub fn module_to_wgsl(module: &Module, capabilities: naga::valid::Capabilities) -> String {
    let flags = naga::valid::ValidationFlags::all() ^ naga::valid::ValidationFlags::BINDINGS;
    let info = naga::valid::Validator::new(flags, capabilities)
        .validate(module)
        .expect("generated module failed validation");
    naga::back::wgsl::write_string(module, &info, naga::back::wgsl::WriterFlags::empty())
        .expect("WGSL write failed")
}

// ---------------------------------------------------------------------------
// matmul.wgsl — 4×4 register-tiled matrix multiply (64×64 output tiles)
//
// Workgroup [16, 16, 1] = 256 threads, dispatched as [ceil(N/64), ceil(M/64), 1].
// Each thread computes a 4×4 sub-tile of the output using register blocking.
// Shared memory tiles: shared_a[64*16], shared_b[16*64].
// ---------------------------------------------------------------------------

/// Register-tiled matmul: C = A × B via Naga IR with shared memory.
///
/// BM=64, BN=64, KTILE=16, TM=4, TN=4.
/// Workgroup [16, 16, 1], dispatched as [ceil(N/64), ceil(M/64), 1].
///
/// Template variables for global memory indices:
const MATMUL_A_FWD: &str = "a_row * params.k + a_col"; // A[m,k] row-major
const MATMUL_B_FWD: &str = "b_row * params.n + b_col"; // B[k,n] row-major
const MATMUL_A_AT: &str = "a_col * params.m + a_row"; // A^T[m,k] = A[k*M+m]
const MATMUL_B_BT: &str = "b_col * params.k + b_row"; // B^T[k,n] = B[n*K+k]

/// Thread-to-tile mapping for coalesced global memory access.
///
/// Large-tile (64×64) load mappings. Adjacent threads follow each operand's
/// fast dimension: the selected K extent for A[M,K] and B[N,K], or the
/// output tile width for A[K,M] and B[K,N].
const A_ROW_FWD: &str = "flat / $K_TILE_U"; // M varies slowly (good for [M,K])
const A_COL_FWD: &str = "flat % $K_TILE_U"; // K varies fast (coalesced in [M,K])
const A_ROW_AT: &str = "flat % $BM_U"; // M varies fast (coalesced in [K,M])
const A_COL_AT: &str = "flat / $BM_U"; // K varies slowly
const B_ROW_FWD: &str = "flat / $BN_U"; // K varies slowly (good for [K,N])
const B_COL_FWD: &str = "flat % $BN_U"; // N varies fast (coalesced in [K,N])
const B_ROW_BT: &str = "flat % $K_TILE_U"; // K varies fast (coalesced in [N,K])
const B_COL_BT: &str = "flat / $K_TILE_U"; // N varies slowly

/// Tile geometry for the register-tiled scalar matmul skeleton.
///
/// Public because the epilogue generators are driven by the runtime's
/// small-tile demotion: a dispatch whose workgroup count was recomputed
/// for 32×32 tiles must be paired with a 32×32 shader, so the choice
/// travels with `Dispatch::use_small_tiles` into the pipeline key.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum MatMulTile {
    /// BM=BN=64, TM=TN=4 — the default tile.
    #[default]
    Large,
    /// BM=BN=32, TM=TN=2 — 4× more workgroups for low-occupancy shapes.
    Small,
    /// BM=BN=16, TM=TN=1 — one output per thread when 32-wide tiles still
    /// leave the device idle.
    Occupancy,
    /// Short-M or other non-square block. `bm` and `bn` are multiples of 16.
    Rect { bm: u32, bn: u32 },
}

impl MatMulTile {
    fn bm(self) -> u32 {
        match self {
            MatMulTile::Large => 64,
            MatMulTile::Small => 32,
            MatMulTile::Occupancy => 16,
            MatMulTile::Rect { bm, .. } => bm,
        }
    }
    fn bn(self) -> u32 {
        match self {
            MatMulTile::Large => 64,
            MatMulTile::Small => 32,
            MatMulTile::Occupancy => 16,
            MatMulTile::Rect { bn, .. } => bn,
        }
    }
    fn tm(self) -> u32 {
        self.bm() / 16
    }
    fn tn(self) -> u32 {
        self.bn() / 16
    }
}

/// One straight-line copy of the register tile per K step. Each step is its
/// own block so the `let` accumulators can be repeated.
fn unroll_k_math(compute_body: &str, k_tile: u32) -> String {
    let mut out = String::new();
    for kk in 0..k_tile {
        out.push_str("{\n");
        out.push_str(&compute_body.replace("kk", &format!("{kk}u")));
        out.push_str("}\n");
    }
    out
}

/// Generate the unrolled `(acc_decl, compute_body, acc_array)` sections of
/// the tiled-matmul skeleton for a TM×TN register tile.
fn tiled_matmul_body(
    tile: MatMulTile,
    k_tile: u32,
    interleave_columns: bool,
) -> (String, String, String) {
    tiled_gemm_body(
        tile.tm(),
        tile.tn(),
        k_tile + 1,
        tile.bn() + 1,
        interleave_columns,
    )
}

/// An optimizer kernel over the segments of one arena chunk.
fn optimizer_module(source: &str) -> ShaderModule {
    ShaderModule::new(&format!(
        "{}\n{source}",
        include_str!("shaders/optimizer_segments.wgsl")
    ))
}

/// Generate an ordinary scalar convolution with a measured K-tile candidate.
pub(crate) fn generate_conv_module(
    group: ShaderGroup,
    k_tile: u32,
    params: Option<&[u32]>,
) -> ShaderModule {
    let (source, tile) = match group {
        ShaderGroup::Conv2dGemm => (include_str!("shaders/conv2d_gemm.wgsl"), MatMulTile::Large),
        ShaderGroup::Conv2dGemmSmall => {
            (include_str!("shaders/conv2d_gemm.wgsl"), MatMulTile::Small)
        }
        ShaderGroup::Conv2dGemm16 => (
            include_str!("shaders/conv2d_gemm.wgsl"),
            MatMulTile::Occupancy,
        ),
        ShaderGroup::Conv2dGradInputGemm => (
            include_str!("shaders/conv2d_grad_input_gemm.wgsl"),
            MatMulTile::Large,
        ),
        ShaderGroup::Conv2dGradInputGemmSmall => (
            include_str!("shaders/conv2d_grad_input_gemm.wgsl"),
            MatMulTile::Small,
        ),
        ShaderGroup::Conv2dGradInputGemm16 => (
            include_str!("shaders/conv2d_grad_input_gemm.wgsl"),
            MatMulTile::Occupancy,
        ),
        ShaderGroup::Conv2dGradWeightGemm => {
            return conv_grad_weight_tiled(MatMulTile::Large, false, k_tile, params);
        }
        ShaderGroup::Conv2dGradWeightGemmSmall => {
            return conv_grad_weight_tiled(MatMulTile::Small, false, k_tile, params);
        }
        ShaderGroup::Conv2dGradWeightGemm16 => {
            return conv_grad_weight_tiled(MatMulTile::Occupancy, false, k_tile, params);
        }
        ShaderGroup::Conv2dGradWeightGemmSplit => {
            return conv_grad_weight_tiled(MatMulTile::Large, true, k_tile, params);
        }
        ShaderGroup::Conv2dGradWeightGemmSplitSmall => {
            return conv_grad_weight_tiled(MatMulTile::Small, true, k_tile, params);
        }
        ShaderGroup::Conv2dGradWeightGemmSplit16 => {
            return conv_grad_weight_tiled(MatMulTile::Occupancy, true, k_tile, params);
        }
        _ => panic!("not an ordinary scalar convolution: {group:?}"),
    };
    conv_gemm_tiled(source, tile, k_tile, params)
}

/// All three implicit-GEMM conv skeletons stage A at stride K and B at
/// stride BM, with BM·K/256 elements per thread.
fn conv_gemm_tiled(
    src: &str,
    tile: MatMulTile,
    k_tile: u32,
    params: Option<&[u32]>,
) -> ShaderModule {
    assert!(matches!(k_tile, 16 | 32));
    let bm = tile.bm();
    let tm = tile.tm();
    // Pad shared-memory strides by one so a wave that spans two rows does not
    // hit the same bank, matching the scalar matmul skeleton. The padding
    // column is never stored or read.
    let a_stride = k_tile + 1;
    let b_stride = bm + 1;
    let (acc_decl, compute_body, acc_array) = tiled_gemm_body(tm, tm, a_stride, b_stride, false);
    let (declaration, divisor) = if let Some(values) = params {
        assert_eq!(values.len(), 16, "Conv2dParams layout");
        let arguments = values.iter().map(|v| format!("{v}u")).collect::<Vec<_>>();
        (
            format!("const params = Params({});", arguments.join(", ")),
            // Divisors are constants in this specialization, so integer `/`
            // is exact and lowers to a short multiply-high. The uniform shader
            // keeps the software reciprocal because its divisor is dynamic.
            crate::divisor::NATIVE_SHADER,
        )
    } else {
        (
            "var<uniform> params: Params;".into(),
            crate::divisor::SHADER,
        )
    };
    let helpers = format!("{divisor}\n{}", include_str!("shaders/digits.wgsl"));
    let src = preprocess(
        src,
        &[
            ("$DIVISOR", &helpers),
            ("$PARAMS_TYPE", include_str!("shaders/conv2d_params.wgsl")),
            ("$PARAMS_DECL", &declaration),
            ("$BM_U", &format!("{bm}u")),
            ("$TM_U", &format!("{tm}u")),
            ("$KTILE_U", &format!("{k_tile}u")),
            ("$A_STRIDE_U", &format!("{a_stride}u")),
            ("$B_STRIDE_U", &format!("{b_stride}u")),
            ("$STAGE_EPT_U", &format!("{}u", bm * k_tile / 256)),
            ("$SHARED_A_SIZE", &(bm * a_stride).to_string()),
            ("$SHARED_B_SIZE", &(k_tile * b_stride).to_string()),
            ("$ACC_DECL", &acc_decl),
            ("$COMPUTE_BODY", &compute_body),
            ("$ACC_ARRAY", &acc_array),
        ],
    );
    ShaderModule::new(&src)
}

fn conv_grad_weight_tiled(
    tile: MatMulTile,
    split_k: bool,
    k_tile: u32,
    params: Option<&[u32]>,
) -> ShaderModule {
    assert!(!split_k || k_tile == 16, "split-K partitioning uses K=16");
    let (counts, range, start, end, offset) = if split_k {
        (
            ", @builtin(num_workgroups) counts: vec3<u32>",
            "let tiles = (k_total + 15u) / 16u;\n\
             let per_split = tiles / counts.z;\n\
             let extra = tiles - per_split * counts.z;\n\
             let first = wgid.z * per_split + min(wgid.z, extra);\n\
             let last = first + per_split + u32(wgid.z < extra);\n\
             let k_end = min(last * 16u, k_total);",
            "first * 16u",
            "k_end",
            "wgid.z * m_total * n_total + ",
        )
    } else {
        ("", "", "0u", "k_total", "")
    };
    let source = preprocess(
        include_str!("shaders/conv2d_grad_weight_gemm.wgsl"),
        &[
            ("$COUNTS", counts),
            ("$K_RANGE", range),
            ("$K_START", start),
            ("$K_END", end),
            ("$OUTPUT_OFFSET", offset),
        ],
    );
    conv_gemm_tiled(&source, tile, k_tile, params)
}

/// Shared unroll generator for every register-tiled GEMM skeleton
/// (matmul.wgsl and the implicit-GEMM conv kernels): accumulator
/// declarations, the KTILE inner-loop FMA body, and the store array,
/// parameterized by register tile size and shared-memory strides.
fn tiled_gemm_body(
    tm: u32,
    tn: u32,
    a_stride: u32,
    b_stride: u32,
    interleave_columns: bool,
) -> (String, String, String) {
    use std::fmt::Write;

    let mut acc_decl = String::new();
    for i in 0..tm {
        for j in 0..tn {
            let _ = write!(acc_decl, "var s{i}_{j} = 0.0; ");
        }
        acc_decl.push_str("\n    ");
    }

    let mut body = String::new();
    for i in 0..tm {
        let _ = writeln!(
            body,
            "            let a{i} = shared_a[(ty * {tm}u + {i}u) * {a_stride}u + kk];"
        );
    }
    for j in 0..tn {
        let column = if interleave_columns {
            format!("tx + {j}u * 16u")
        } else {
            format!("tx * {tn}u + {j}u")
        };
        let _ = writeln!(
            body,
            "            let b{j} = shared_b[kk * {b_stride}u + {column}];"
        );
    }
    for i in 0..tm {
        body.push_str("            ");
        for j in 0..tn {
            let _ = write!(body, "s{i}_{j} += a{i} * b{j}; ");
        }
        body.push('\n');
    }

    let mut acc_array = format!("array<array<f32, {tn}>, {tm}>(\n");
    for i in 0..tm {
        let cols: Vec<String> = (0..tn).map(|j| format!("s{i}_{j}")).collect();
        let _ = writeln!(acc_array, "        array<f32, {tn}>({}),", cols.join(", "));
    }
    acc_array.push_str("    )");

    (acc_decl, body, acc_array)
}

/// Where the tiled skeleton reads A and B.
///
/// `*_idx` address the storage buffers; `*_row`/`*_col` split the flat
/// thread index across the shared tile, so they depend on the tile width
/// as well as on the transposition.
#[derive(Clone, Copy)]
struct MatMulIndexing<'a> {
    a_idx: &'a str,
    b_idx: &'a str,
    a_row: &'a str,
    a_col: &'a str,
    b_row: &'a str,
    b_col: &'a str,
    tile_row: &'a str,
    c_idx: &'a str,
}

fn matmul_vars_tiled(
    indexing: MatMulIndexing<'_>,
    fused_decl: &str,
    fused_expr: &str,
    epilogue_decl: &str,
    epilogue_body: &str,
    options: MatMulOptions,
    splits: u32,
    copies: u32,
) -> ShaderModule {
    let MatMulIndexing {
        a_idx,
        b_idx,
        a_row,
        a_col,
        b_row,
        b_col,
        tile_row,
        c_idx,
    } = indexing;
    let MatMulOptions {
        format: b_mode,
        tile,
        knobs,
    } = options;
    let src = include_str!("shaders/matmul.wgsl");
    let full_decl = if epilogue_decl.is_empty() {
        fused_decl.to_string()
    } else {
        format!("{}\n{}", fused_decl, epilogue_decl)
    };
    let store_body = if epilogue_body.is_empty() {
        format!("$C_BUFFER[idx] = s[i][j]{};", fused_expr)
    } else {
        format!(
            "var val = s[i][j]{};\n                {}\n                $C_BUFFER[idx] = val;",
            fused_expr, epilogue_body
        )
    };
    let (enable_f16, b_storage, b_load_expr, b_dequant_fn) = match b_mode {
        WeightFormat::F32 => (
            "",
            "array<f32>",
            format!("$B_BUFFER[select(0u, {}, in_bounds)]", b_idx),
            String::new(),
        ),
        WeightFormat::F16 => (
            "enable f16;",
            "array<f16>",
            format!("f32($B_BUFFER[select(0u, {}, in_bounds)])", b_idx),
            String::new(),
        ),
        WeightFormat::Q4 => (
            "",
            "array<u32>",
            "dequant_q4(select(0u, b_row, in_bounds), select(0u, b_col, in_bounds))".to_string(),
            format!("{F16_DECODE_FN}{Q4_DEQUANT_FN}"),
        ),
        WeightFormat::Q8 => (
            "",
            "array<u32>",
            "dequant_q8(select(0u, b_row, in_bounds), select(0u, b_col, in_bounds))".to_string(),
            format!("{F16_DECODE_FN}{Q8_DEQUANT_FN}"),
        ),
        WeightFormat::Q40 => (
            "",
            "array<u32>",
            "dequant_q40(select(0u, b_row, in_bounds), select(0u, b_col, in_bounds))".to_string(),
            format!("{F16_DECODE_FN}{Q40_DEQUANT_FN}"),
        ),
        WeightFormat::Q4K => (
            "",
            "array<u32>",
            "dequant_q4k(select(0u, b_row, in_bounds), select(0u, b_col, in_bounds))".to_string(),
            format!("{F16_DECODE_FN}{K_SCALE_MIN_FN}{Q4K_DEQUANT_FN}"),
        ),
        WeightFormat::Q6K => (
            "",
            "array<u32>",
            "dequant_q6k(select(0u, b_row, in_bounds), select(0u, b_col, in_bounds))".to_string(),
            format!("{F16_DECODE_FN}{Q6K_DEQUANT_FN}"),
        ),
        WeightFormat::Q5K => (
            "",
            "array<u32>",
            "dequant_q5k(select(0u, b_row, in_bounds), select(0u, b_col, in_bounds))".to_string(),
            format!("{F16_DECODE_FN}{K_SCALE_MIN_FN}{Q5K_DEQUANT_FN}"),
        ),
        WeightFormat::Q3K => (
            "",
            "array<u32>",
            "dequant_q3k(select(0u, b_row, in_bounds), select(0u, b_col, in_bounds))".to_string(),
            format!("{F16_DECODE_FN}{Q3K_DEQUANT_FN}"),
        ),
    };
    // Nibble-packed large tile: 32×64 B tile, 256 threads → each thread owns
    // 8 consecutive K of one column, which share a block header and, for Q4,
    // a single data word. Small tile and Q8 stay on the per-element path.
    let pack8 = match b_mode {
        WeightFormat::Q4 if tile == MatMulTile::Large => Some("dequant_q4_pack8"),
        WeightFormat::Q4K if tile == MatMulTile::Large => Some("dequant_q4k_pack8"),
        WeightFormat::Q6K if tile == MatMulTile::Large => Some("dequant_q6k_pack8"),
        WeightFormat::Q5K if tile == MatMulTile::Large => Some("dequant_q5k_pack8"),
        WeightFormat::Q3K if tile == MatMulTile::Large => Some("dequant_q3k_pack8"),
        _ => None,
    };
    let b_stage_body = if let Some(pack8) = pack8 {
        format!(
            "\
        let n_local = tid % $BN_U;\n\
        let k_base = (tid / $BN_U) * 8u;\n\
        let b_col = tile_col + n_local;\n\
        // Blocks are 32 wide, so 8 values from inside K stay inside it;\n\
        // lanes past the matrix decode the first group and discard it.\n\
        let in_group = (t + k_base < params.k) && (b_col < params.n);\n\
        let unpacked = {pack8}(select(0u, t + k_base, in_group), select(0u, b_col, in_group));\n\
        for (var i = 0u; i < 8u; i++) {{\n\
            let b_row = t + k_base + i;\n\
            let in_bounds = (b_row < params.k) && (b_col < params.n);\n\
            shared_b[(k_base + i) * $B_STRIDE_U + n_local] = select(0.0, unpacked[i], in_bounds);\n\
        }}"
        )
    } else {
        "\
        for (var e = 0u; e < $B_STAGE_EPT_U; e++) {\n\
            let flat = tid + e * 256u;\n\
            let row_local = $B_ROW;\n\
            let col_local = $B_COL;\n\
            let b_row = t + row_local;\n\
            let b_col = tile_col + col_local;\n\
            let in_bounds = (b_row < params.k) && (b_col < params.n);\n\
            shared_b[row_local * $B_STRIDE_U + col_local] = select(0.0, $B_LOAD_EXPR, in_bounds);\n\
        }"
        .to_string()
    };
    let bm = tile.bm();
    let bn = tile.bn();
    let tm = tile.tm();
    let tn = tile.tn();
    // Block decoders have fixed layouts; plain F32/F16 share the same skeleton.
    let k_tile = match b_mode {
        WeightFormat::F32 | WeightFormat::F16 => knobs.k_stage,
        _ => 32,
    };
    assert!(
        bm * k_tile % 256 == 0 && bn * k_tile % 256 == 0,
        "matmul stage {bm}x{k_tile} and {bn}x{k_tile} must cover whole warps"
    );
    assert!(
        matches!(knobs.k_stage, 8 | 16 | 32),
        "unsupported scalar matmul K stage: {}",
        knobs.k_stage
    );
    let interleave_columns = !b_mode.is_quantized() && knobs.interleave_columns;
    let (acc_decl, compute_body, acc_array) = tiled_matmul_body(tile, k_tile, interleave_columns);
    let k_math = if knobs.unroll_k {
        unroll_k_math(&compute_body, k_tile)
    } else {
        compute_body
    };
    let output_column = if interleave_columns {
        "tx + j * 16u".to_string()
    } else {
        format!("tx * {tn}u + j")
    };
    let src = preprocess(
        src,
        &[
            ("$COMPUTE_BODY", &k_math),
            (
                "$K_UNROLL_U",
                &format!("{}u", if knobs.unroll_k { k_tile } else { 1 }),
            ),
            ("$B_STAGE_BODY", &b_stage_body),
            ("$ENABLE_F16", enable_f16),
            ("$B_LOAD_EXPR", &b_load_expr),
            ("$B_DEQUANT_FN", &b_dequant_fn),
            ("$A_INDEX", a_idx),
            ("$B_INDEX", b_idx),
            ("$TILE_ROW", tile_row),
            ("$C_INDEX", c_idx),
            ("$SPLITS_U", &format!("{splits}u")),
            ("$A_ROW", a_row),
            ("$A_COL", a_col),
            ("$B_ROW", b_row),
            ("$B_COL", b_col),
            ("$FUSED_ADD_DECL", &full_decl),
            ("$STORE_BODY", &store_body),
            ("$OUTPUT_COLUMN", &output_column),
            ("$BM_U", &format!("{bm}u")),
            ("$BN_U", &format!("{bn}u")),
            ("$TM_U", &format!("{tm}u")),
            ("$TN_U", &format!("{tn}u")),
            ("$B_STRIDE_U", &format!("{}u", bn + 1)),
            ("$A_STRIDE_U", &format!("{}u", k_tile + 1)),
            ("$K_TILE_U", &format!("{k_tile}u")),
            ("$A_STAGE_EPT_U", &format!("{}u", bm * k_tile / 256)),
            ("$B_STAGE_EPT_U", &format!("{}u", bn * k_tile / 256)),
            ("$SHARED_A_SIZE", &(bm * (k_tile + 1)).to_string()),
            ("$SHARED_B_SIZE", &(k_tile * (bn + 1)).to_string()),
            ("$ACC_DECL", &acc_decl),
            ("$ACC_ARRAY", &acc_array),
        ],
    );
    matmul_module(&src, b_storage, None, copies)
}

/// f16 → f32 for the packed-weight kernels' block scales.
///
/// `unpack2x16float` handles subnormals, Inf and NaN, so imported GGUF
/// scales are not ours to constrain: `0x0100` is 2^-16, which a quantizer
/// emits for a block of very small weights and which the hand-assembled
/// version this replaces folded to zero.
///
/// Every packed format shares this one copy. Naga gates the builtin behind
/// `SHADER_FLOAT16_IN_FLOAT32`, which Blade enables — it stores
/// f16-precision values inside f32 rather than needing the `f16`
/// extension, and lowers to core `GlslStd450 UnpackHalf2x16` on SPIR-V.
const F16_DECODE_FN: &str = include_str!("shaders/dequant_f16_decode.wgsl");

/// Meganeura asymmetric Q4 (Q4_1-style) dequantization helper for WGSL.
/// Buffer layout: [scales as packed f16 pairs (u32)][packed nibble data (u32)].
/// Column-wise blocking: blocks of 32 elements along the K dimension per column.
///
/// `dequant_q4` stays the scalar entry used by GEMV. Tiled staging uses
/// `dequant_q4_pack8`: one (d, m) header and one data word → 8 values.
const Q4_DEQUANT_FN: &str = include_str!("shaders/dequant_q4.wgsl");

/// GGML's own Q4_0: 18-byte blocks of an f16 `d` and 16 nibble bytes, read
/// byte-addressed because 18 is not a whole number of words.
///
/// Distinct from [`Q4_DEQUANT_FN`] in both the arithmetic — symmetric
/// `d * (q - 8)` against Meganeura's `q * d + m` — and the nibble order:
/// GGML splits a block across halves where Meganeura pairs neighbours.
const Q40_DEQUANT_FN: &str = include_str!("shaders/dequant_q40.wgsl");

/// The `get_scale_min_k4` scale/min decoder, shared by Q4_K and Q5_K.
///
/// Both store eight 6-bit pairs in twelve bytes the same way, and both
/// have word-aligned superblocks, so the byte reader is word-indexed.
const K_SCALE_MIN_FN: &str = include_str!("shaders/dequant_k_scale_min.wgsl");

/// GGML Q4_K dequantization, reading GGUF's bytes verbatim.
///
/// A 256-element superblock is 36 u32s: `[d|dmin]`, three words of eight
/// 6-bit scale/min pairs, then 32 words of nibbles. The value is
/// `d * sc_j * q - dmin * m_j` for sub-block `j = (k % 256) / 32`, which is
/// the same `q * scale + min` shape the Q4 kernels already compute — only
/// the pair is derived per sub-block instead of read outright.
///
/// The scale unpacking mirrors `get_scale_min_k4` in `ggml-quants.c`, and
/// the nibble split mirrors `dequantize_row_q4_K`: within each 64-element
/// span the low nibbles feed the first 32 elements and the high nibbles
/// the second.
const Q4K_DEQUANT_FN: &str = include_str!("shaders/dequant_q4k.wgsl");

/// GGML Q6_K dequantization, reading GGUF's bytes verbatim.
///
/// A 256-element superblock is 210 bytes: 128 low-nibble bytes `ql`, 64
/// bytes of high bit-pairs `qh`, sixteen signed 8-bit sub-block scales, and
/// an f16 `d`. Each 6-bit quant is `(ql nibble) | (qh bit-pair << 4)`,
/// biased by 32, and one scale covers 16 elements.
///
/// 210 is not a multiple of 4, so a superblock starts at byte 0 or 2 of a
/// word depending on its index, and every field is read byte-addressed.
/// The f16 `d` at offset 208 lands on 0 or 2 either way, so it never spans
/// two words.
///
/// The element mapping mirrors `dequantize_row_q6_K`: each 128-element half
/// is walked in 32-element strides `j`, drawing the low nibble of
/// `ql[l + (j & 1) * 32]` for `j < 2` and the high nibble for `j >= 2`, with
/// the bit-pair at `qh[l] >> (j * 2)` and the scale at `j * 2 + l / 16`.
const Q6K_DEQUANT_FN: &str = include_str!("shaders/dequant_q6k.wgsl");

/// GGML Q5_K dequantization, reading GGUF's bytes verbatim.
///
/// [`Q4K_DEQUANT_FN`] plus one bit. A 256-element superblock is 44 u32s:
/// `[d|dmin]`, three words of eight 6-bit scale/min pairs, eight words of
/// high bits, then 32 words of nibbles. The 5-bit quant is the nibble with
/// `qh`'s bit for this sub-block contributing 16, so the value is
/// `d * sc_j * (nibble + 16*bit) - dmin * m_j`.
///
/// The scale packing is `get_scale_min_k4`, identical to Q4_K. The high
/// bit lives at `qh[l] >> (e / 32)`, where `l` is the element's position
/// within its 32-element stride — `qh` is indexed by `l` alone and shared
/// across all four 64-element spans, which is why the bit index is the
/// sub-block number rather than an offset into `qh`.
const Q5K_DEQUANT_FN: &str = include_str!("shaders/dequant_q5k.wgsl");

/// GGML Q3_K dequantization, reading GGUF's bytes verbatim.
///
/// A 256-element superblock is 110 bytes: 32 bytes of `hmask`, 64 bytes of
/// 2-bit quants, 12 bytes of packed 6-bit scales, and an f16 `d`. Like
/// Q6_K that is not a whole number of words, so every field is read
/// byte-addressed off a byte base.
///
/// Two things here have no analogue in the other K-quants. The high bit is
/// **inverted** — a *clear* `hmask` bit subtracts 4 — so the quant is
/// `(qs >> 2j) & 3` minus 4 unless the bit is set. And the sixteen 6-bit
/// scales are not `get_scale_min_k4`: they come from a shuffle of the 12
/// bytes that pairs each low or high nibble with two bits drawn from the
/// last four bytes, then biases by 32.
const Q3K_DEQUANT_FN: &str = include_str!("shaders/dequant_q3k.wgsl");

/// Q8_0 dequantization: symmetric 8-bit, 32-element blocks.
/// Layout: [9 u32s per block: 1 scale_u32 + 8 data_u32s].
/// Block i starts at matrix_b[i * 9]. scale_f16 in low 16 bits of first u32.
const Q8_DEQUANT_FN: &str = include_str!("shaders/dequant_q8.wgsl");

/// Generate the 32×32 small-tile form of a matmul group.
pub fn generate_module_small(group: ShaderGroup, knobs: MatmulKnobs) -> ShaderModule {
    match group {
        ShaderGroup::BlockMatMul
        | ShaderGroup::BlockMatMulAT
        | ShaderGroup::BlockMatMulBT
        | ShaderGroup::BatchMatMul
        | ShaderGroup::BatchMatMulAT
        | ShaderGroup::BatchMatMulBT => gen_block_matmul(group, MatMulTile::Small),
        ShaderGroup::MatMul
        | ShaderGroup::MatMulAdd
        | ShaderGroup::MatMulAT
        | ShaderGroup::MatMulBT => generate_matmul_with_epilogue(
            group,
            None,
            MatMulOptions {
                tile: MatMulTile::Small,
                knobs,
                ..Default::default()
            },
        ),
        _ => generate_module(group, knobs),
    }
}

fn gen_block_matmul(group: ShaderGroup, tile: MatMulTile) -> ShaderModule {
    let (ordinary, a_idx, b_idx, c_idx) = match group {
        ShaderGroup::BlockMatMul => (
            ShaderGroup::MatMul,
            "(a_row * params._pad + wgid.z) * params.k + a_col",
            "(wgid.z * params.k + b_row) * params.n + b_col",
            "(row * params._pad + wgid.z) * params.n + col",
        ),
        ShaderGroup::BlockMatMulAT => (
            ShaderGroup::MatMulAT,
            "(a_col * params._pad + wgid.z) * params.m + a_row",
            "(b_row * params._pad + wgid.z) * params.n + b_col",
            "(wgid.z * params.m + row) * params.n + col",
        ),
        ShaderGroup::BlockMatMulBT => (
            ShaderGroup::MatMulBT,
            "(a_row * params._pad + wgid.z) * params.k + a_col",
            "(wgid.z * params.n + b_col) * params.k + b_row",
            "(row * params._pad + wgid.z) * params.n + col",
        ),
        // Batch-major: one whole matrix per `wgid.z`.
        ShaderGroup::BatchMatMul => (
            ShaderGroup::MatMul,
            "(wgid.z * params.m + a_row) * params.k + a_col",
            "(wgid.z * params.k + b_row) * params.n + b_col",
            "(wgid.z * params.m + row) * params.n + col",
        ),
        ShaderGroup::BatchMatMulAT => (
            ShaderGroup::MatMulAT,
            "(wgid.z * params.k + a_col) * params.m + a_row",
            "(wgid.z * params.k + b_row) * params.n + b_col",
            "(wgid.z * params.m + row) * params.n + col",
        ),
        ShaderGroup::BatchMatMulBT => (
            ShaderGroup::MatMulBT,
            "(wgid.z * params.m + a_row) * params.k + a_col",
            "(wgid.z * params.n + b_col) * params.k + b_row",
            "(wgid.z * params.m + row) * params.n + col",
        ),
        _ => unreachable!(),
    };
    let (a_row, a_col, b_row, b_col) = matmul_stage_maps(ordinary);
    matmul_vars_tiled(
        MatMulIndexing {
            a_idx,
            b_idx,
            a_row,
            a_col,
            b_row,
            b_col,
            tile_row: "wgid.y",
            c_idx,
        },
        "",
        "",
        "",
        "",
        MatMulOptions {
            format: WeightFormat::F32,
            tile,
            ..Default::default()
        },
        1,
        1,
    )
}

pub fn generate_module_weighted(
    group: ShaderGroup,
    format: WeightFormat,
    knobs: MatmulKnobs,
) -> ShaderModule {
    match group {
        ShaderGroup::MatMul
        | ShaderGroup::MatMulAdd
        | ShaderGroup::MatMulAT
        | ShaderGroup::MatMulATAdd
        | ShaderGroup::MatMulBT
        | ShaderGroup::MatMulBTAdd => generate_matmul_with_epilogue(
            group,
            None,
            MatMulOptions {
                format,
                knobs,
                ..Default::default()
            },
        ),
        // The f16 embedding is a variant of the same gather, selected by the
        // table's dtype rather than by a separate op and shader group.
        ShaderGroup::Embedding if format == WeightFormat::F16 => {
            ShaderModule::new(include_str!("shaders/embedding_f16.wgsl"))
        }
        // Every block-packed format takes the same K-split GEMV with its
        // own decoder substituted, so the format picks the helper rather
        // than the arm.
        ShaderGroup::MatMulGemv
        | ShaderGroup::MatMulGemvAdd
        | ShaderGroup::MatMulGemvBT
        | ShaderGroup::MatMulGemvBTAdd => {
            generate_module_gemv(group, format, GemvShape::initial(group))
        }
        // Unsupported packed routes must fail closed: falling through would
        // read compressed bytes as f32 and produce plausible garbage.
        _ if format.is_quantized() => panic!(
            "no {format:?} variant for {group:?}; block-quantized weights are \
             supported on forward tiled matmul groups and K-split GEMV only"
        ),
        _ => generate_module(group, knobs),
    }
}

const ATTENTION_PARAMS_WGSL: &str = include_str!("shaders/attention_params.wgsl");
const CACHED_ATTENTION_PARAMS_WGSL: &str = include_str!("shaders/cached_attention_params.wgsl");
const GEMV_TEMPLATE: &str = include_str!("shaders/matmul_gemv.wgsl");

/// Apply width and reduction slots shared by floating-point and integer-dot GEMV.
fn specialize_gemv(source: &str, shape: GemvShape) -> String {
    shape.validate();
    let columns = shape.column_groups;
    let (reduction, subgroup_args, total) = match shape.reduction {
        GemvReduction::Tree => {
            let mut body = template_section(GEMV_TEMPLATE, "tree_start").to_owned();
            let mut stride = shape.threads / 2;
            while stride > 1 {
                body.push_str(&preprocess(
                    template_section(GEMV_TEMPLATE, "tree_step"),
                    &[
                        ("$STRIDE", &stride.to_string()),
                        ("$REDUCE_STRIDE", &(stride * columns).to_string()),
                    ],
                ));
                stride /= 2;
            }
            // Read by lane zero, where lid.x is the column within the group.
            (
                body,
                "",
                format!("reduce_buf[lid.x] + reduce_buf[lid.x + {columns}u]"),
            )
        }
        GemvReduction::Subgroup => (
            template_section(GEMV_TEMPLATE, "subgroup_reduce").to_owned(),
            ", @builtin(subgroup_invocation_id) sg_id: u32, @builtin(subgroup_id) wave_id: u32, @builtin(num_subgroups) wave_count: u32",
            "group_total".to_owned(),
        ),
    };
    preprocess(
        source,
        &[
            ("$REDUCTION", &reduction),
            ("$SUBGROUP_ARGS", subgroup_args),
            ("$TOTAL", &total),
            ("$LANES", &shape.threads.to_string()),
            ("$COLUMN_GROUPS", &columns.to_string()),
            ("$WORKGROUP_SIZE", &(shape.threads * columns).to_string()),
        ],
    )
}

/// Helpers and scalar load function for a block-packed B format.
fn packed_decoder(mode: WeightFormat) -> Option<(String, &'static str)> {
    // Q4_K and Q5_K share the `get_scale_min_k4` block, so it is prepended
    // rather than duplicated in each decoder.
    match mode {
        WeightFormat::Q4 => Some((Q4_DEQUANT_FN.to_string(), "dequant_q4")),
        WeightFormat::Q8 => Some((Q8_DEQUANT_FN.to_string(), "dequant_q8")),
        WeightFormat::Q40 => Some((Q40_DEQUANT_FN.to_string(), "dequant_q40")),
        WeightFormat::Q4K => Some((format!("{K_SCALE_MIN_FN}{Q4K_DEQUANT_FN}"), "dequant_q4k")),
        WeightFormat::Q6K => Some((Q6K_DEQUANT_FN.to_string(), "dequant_q6k")),
        WeightFormat::Q5K => Some((format!("{K_SCALE_MIN_FN}{Q5K_DEQUANT_FN}"), "dequant_q5k")),
        WeightFormat::Q3K => Some((Q3K_DEQUANT_FN.to_string(), "dequant_q3k")),
        _ => None,
    }
}

pub fn generate_module_gemv_rmsnorm(
    group: ShaderGroup,
    shape: GemvShape,
    format: WeightFormat,
) -> ShaderModule {
    assert!(matches!(
        group,
        ShaderGroup::MatMulGemv | ShaderGroup::MatMulGemvBT
    ));
    generate_gemv(group, format, shape, true)
}

pub(crate) fn generate_module_gemv(
    group: ShaderGroup,
    format: WeightFormat,
    shape: GemvShape,
) -> ShaderModule {
    generate_gemv(group, format, shape, false)
}

fn generate_gemv(
    group: ShaderGroup,
    format: WeightFormat,
    shape: GemvShape,
    norm: bool,
) -> ShaderModule {
    let (transposed, add) = match group {
        ShaderGroup::MatMulGemv => (false, false),
        ShaderGroup::MatMulGemvAdd => (false, true),
        ShaderGroup::MatMulGemvBT => (true, false),
        ShaderGroup::MatMulGemvBTAdd => (true, true),
        _ => panic!("{group:?} is not a GEMV group"),
    };
    assert!(
        !transposed || !format.is_quantized(),
        "no {format:?} variant for {group:?}; block-quantized weights cannot serve a transposed B"
    );
    let shape = shape.for_group(group);
    assert!(shape.column_groups == 1 || matches!(format, WeightFormat::F32 | WeightFormat::F16));
    let rows = shape.bt_rows;
    let fragment = |name| template_section(GEMV_TEMPLATE, name);
    let b_storage = match format {
        WeightFormat::F32 => "array<vec4<f32>>",
        WeightFormat::F16 => "array<vec4<f16>>",
        _ => "array<u32>",
    };
    let load = |index: &str| {
        if format == WeightFormat::F16 {
            format!("vec4<f32>(matrix_b[{index}])")
        } else {
            format!("matrix_b[{index}]")
        }
    };
    let addend = |index: &str| {
        if add {
            format!(" + src[{index}]")
        } else {
            String::new()
        }
    };
    let (helpers, weight_load) = match packed_decoder(format) {
        Some((helpers, call)) => (
            format!("{F16_DECODE_FN}{helpers}"),
            preprocess(fragment("packed_load"), &[("$DEQUANT", call)]),
        ),
        None => (
            String::new(),
            preprocess(
                fragment("weight_load"),
                &[(
                    "$B_VALUE",
                    &load(if shape.column_groups == 1 {
                        "kk * n_v4 + col4"
                    } else {
                        "select(0u, kk * n_v4 + col4, col4 < n_v4)"
                    }),
                )],
            ),
        ),
    };
    let (accumulate, store) = if rows == 1 {
        (
            preprocess(
                fragment("bt_accumulate"),
                &[("$B_VALUE", &load("row_off + kk_v4"))],
            ),
            preprocess(fragment("bt_store"), &[("$ADDEND", &addend("col"))]),
        )
    } else {
        let mut accumulate = String::new();
        let mut store = fragment("bt_total").to_owned();
        for (row, component) in ["x", "y", "z", "w"].iter().take(rows as usize).enumerate() {
            let index = format!("col + {row}u");
            let vars = [
                ("$ROW", row.to_string()),
                ("$COMPONENT", component.to_string()),
                (
                    "$B_VALUE",
                    load(&format!("row_off + {row}u * k_v4 + kk_v4")),
                ),
                ("$ADDEND", addend(&index)),
            ];
            let vars: Vec<_> = vars
                .iter()
                .map(|&(key, ref value)| (key, value.as_str()))
                .collect();
            accumulate.push_str(&preprocess(fragment("bt_row_accumulate"), &vars));
            store.push_str(&preprocess(fragment("bt_row_store"), &vars));
        }
        (accumulate, store)
    };
    let norm_prologue = if norm {
        let prologue = preprocess(
            include_str!("shaders/matmul_gemv_norm_prologue.wgsl"),
            &[(
                "$NORM_VALUE",
                if transposed {
                    "matrix_a[si / 4u][si % 4u]"
                } else {
                    "matrix_a[si]"
                },
            )],
        );
        format!(
            "{prologue}{}",
            if transposed {
                fragment("bt_norm_end")
            } else {
                "let rs = inv_rms;"
            }
        )
    } else {
        String::new()
    };
    let acc_type = if rows == 1 {
        "f32".to_owned()
    } else {
        format!("vec{rows}<f32>")
    };
    let source = preprocess(
        fragment(if transposed { "transposed" } else { "forward" }),
        &[
            ("$ACCUMULATE", &accumulate),
            ("$STORE", &store),
            ("$WEIGHT_LOAD", &weight_load),
            ("$WEIGHT_HELPERS", &helpers),
            (
                "$ENABLE_F16",
                if format == WeightFormat::F16 {
                    "enable f16;"
                } else {
                    ""
                },
            ),
            ("$EPS_FIELD", if norm { "eps_bits" } else { "_pad" }),
            (
                "$NORM_WEIGHT",
                if norm { fragment("norm_weight") } else { "" },
            ),
            (
                "$NORM_SCRATCH",
                if norm { fragment("norm_scratch") } else { "" },
            ),
            (
                "$NORM_K",
                if norm && transposed {
                    fragment("norm_k")
                } else {
                    ""
                },
            ),
            ("$NORM_PROLOGUE", &norm_prologue),
            (
                "$EARLY_RETURN",
                if transposed && !norm {
                    fragment("bt_guard")
                } else {
                    ""
                },
            ),
            ("$A_LOAD", fragment(if norm { "bt_norm_a" } else { "bt_a" })),
            ("$A_SCALE", if norm { " * rs * norm_w[kk]" } else { "" }),
            ("$B_STORAGE", b_storage),
            (
                "$ADDEND_DECL",
                if !add {
                    ""
                } else if transposed {
                    fragment("bt_addend")
                } else {
                    fragment("addend")
                },
            ),
            ("$ADDEND", &addend("col4")),
            (
                "$COL_EXPR",
                &if rows == 1 {
                    "wgid.x + grid.x * wgid.y".to_owned()
                } else {
                    format!("(wgid.x + grid.x * wgid.y) * {rows}u")
                },
            ),
            ("$ACC_TYPE", &acc_type),
            (
                "$ACC_ZERO",
                &if rows == 1 {
                    "0.0".to_owned()
                } else {
                    format!("{acc_type}(0.0)")
                },
            ),
        ],
    );
    ShaderModule::new(&specialize_gemv(&source, shape))
}

/// The Q8_1-activation, integer-dot GEMV at an explicit shape.
///
/// The weight format selects the kernel: GGML Q4_0 feeds
/// `dot_q4_q8_packed` through its split-nibble blocks, which hand out two
/// int8x4 vectors of consecutive elements per word, exactly what
/// `dot4I8Packed` wants. Meganeura Q8 stores four int8s per whole word and
/// pairs directly as `dot_q8_q8_packed`. See the two shader files.
///
/// `packed_dot` selects the hardware `dot4I8Packed` (DP4A) intrinsic where
/// the device reports `shader_integer_dot_product`; other devices run the
/// exact scalar expansion of the same integer arithmetic. `norm` folds an
/// RmsNorm prologue in, matching the non-integer fused form.
pub(crate) fn generate_module_gemv_int_dot(
    group: ShaderGroup,
    format: WeightFormat,
    shape: GemvShape,
    packed_dot: bool,
    norm: bool,
) -> ShaderModule {
    assert_eq!(
        shape.column_groups, 1,
        "integer-dot GEMV has no grouped-column variant"
    );
    let dot_helpers = if packed_dot {
        include_str!("shaders/matmul_gemv_int_dot_packed.wgsl")
    } else {
        include_str!("shaders/matmul_gemv_int_dot_scalar.wgsl")
    };
    let (addend_decl, addend) = match group {
        ShaderGroup::MatMulGemv => ("", ""),
        ShaderGroup::MatMulGemvAdd => ("var<storage> src: array<vec4<f32>>;", " + src[col4]"),
        _ => panic!("no int-dot variant for {group:?}"),
    };
    let base = match format {
        WeightFormat::Q40 => include_str!("shaders/matmul_gemv_q40_q8.wgsl"),
        WeightFormat::Q8 => include_str!("shaders/matmul_gemv_q8_q8.wgsl"),
        WeightFormat::Q4K | WeightFormat::Q5K | WeightFormat::Q6K | WeightFormat::Q3K => {
            include_str!("shaders/matmul_gemv_qk_q8.wgsl")
        }
        _ => unreachable!("checked above"),
    };
    let weight_helpers = match format {
        WeightFormat::Q40 | WeightFormat::Q8 => String::new(),
        WeightFormat::Q4K | WeightFormat::Q5K | WeightFormat::Q6K | WeightFormat::Q3K => {
            preprocess(
                include_str!("shaders/matmul_gemv_qk_helpers.wgsl"),
                &[
                    ("$F16_DECODE_FN", F16_DECODE_FN),
                    ("$K_SCALE_MIN_FN", K_SCALE_MIN_FN),
                ],
            )
        }
        _ => unreachable!("checked above"),
    };
    let block_dot = match format {
        WeightFormat::Q4K => "q4k_block_dot",
        WeightFormat::Q5K => "q5k_block_dot",
        WeightFormat::Q6K => "q6k_block_dot",
        WeightFormat::Q3K => "q3k_block_dot",
        WeightFormat::Q40 | WeightFormat::Q8 => "",
        _ => unreachable!("checked above"),
    };
    let (norm_decl, a_fn, norm_prologue) = if norm {
        (
            include_str!("shaders/matmul_gemv_int_dot_norm_decl.wgsl"),
            include_str!("shaders/matmul_gemv_int_dot_norm_a.wgsl"),
            include_str!("shaders/matmul_gemv_norm_prologue.wgsl"),
        )
    } else {
        ("", include_str!("shaders/matmul_gemv_int_dot_a.wgsl"), "")
    };
    let source = preprocess(
        base,
        &[
            ("$PACKED_DOT_HELPER", dot_helpers),
            ("$A_FN_DECL", a_fn),
            ("$NORM_DECL", norm_decl),
            ("$NORM_PROLOGUE", norm_prologue),
            ("$NORM_VALUE", "matrix_a[si]"),
            ("$ADDEND_DECL", addend_decl),
            ("$ADDEND", addend),
            ("$WEIGHT_HELPERS", weight_helpers.as_str()),
            ("$BLOCK_DOT", block_dot),
        ],
    );
    ShaderModule::new(&specialize_gemv(&source, shape))
}

fn gen_matmul_coop_wgsl(
    fused_add: bool,
    variant: MatMulCoopVariant,
    config: &CoopConfig,
) -> ShaderModule {
    gen_matmul_coop_wgsl_full(fused_add, variant, config, None, None, 1, 1)
}

/// Generate coop matmul with an optional [`crate::compile::MatMulPrologue`].
pub fn gen_matmul_coop_with_prologue(
    fused_add: bool,
    variant: MatMulCoopVariant,
    config: &CoopConfig,
    prologue: &crate::compile::MatMulPrologue,
) -> ShaderModule {
    gen_matmul_coop_wgsl_full(fused_add, variant, config, Some(prologue), None, 1, 1)
}

/// Generate a cooperative matmul that stages its f32 accumulators through
/// workgroup memory before applying a scalar PointwiseDAG epilogue.
///
/// Cooperative matrix stores do not expose individual accumulator lanes.
/// The workgroup staging step is therefore the portable bridge between the
/// matrix operation and arbitrary per-element WGSL. Extra epilogue buffers
/// are intentionally rejected until the runtime has a dynamic binding layout;
/// unary chains (the current compiler fusion set) need no extra bindings.
pub fn generate_coop_matmul_with_dag_epilogue(
    group: ShaderGroup,
    config: &CoopConfig,
    epilogue: &crate::compile::MatMulEpilogue,
) -> ShaderModule {
    assert!(
        epilogue.inputs.is_empty(),
        "cooperative matmul epilogues with extra buffers are not supported"
    );
    let (fused_add, variant) = coop_shape(group)
        .unwrap_or_else(|| panic!("cooperative epilogue not supported for {group:?}"));
    gen_matmul_coop_wgsl_full(fused_add, variant, config, None, Some(epilogue), 1, 1)
}

/// Generate one K partition of a native 16x16 f32 product.
/// The caller supplies full 32-row/column tiles, a Z grid of `splits`, and
/// `splits * M * N` output elements, then sums the partial matrices.
/// An addend seeds partition zero only. No nonlinear epilogue may be split.
pub fn generate_split_coop_matmul(
    group: ShaderGroup,
    config: &CoopConfig,
    splits: u32,
    prologue: Option<&crate::compile::MatMulPrologue>,
) -> ShaderModule {
    assert!(config.tile_size == 16 && !config.use_f16_input && !config.compensated);
    assert!((2..=65535).contains(&splits));
    let (add, variant) = coop_shape(group).expect("split cooperative matrix group");
    gen_matmul_coop_wgsl_full(add, variant, config, prologue, None, 1, splits)
}

fn gen_matmul_coop_wgsl_full(
    fused_add: bool,
    variant: MatMulCoopVariant,
    config: &CoopConfig,
    prologue: Option<&crate::compile::MatMulPrologue>,
    epilogue: Option<&crate::compile::MatMulEpilogue>,
    copies: u32,
    splits: u32,
) -> ShaderModule {
    assert!(
        splits == 1
            || (config.tile_size == 16
                && !config.use_f16_input
                && copies == 1
                && epilogue.is_none())
    );
    if config.tile_size == 8 && !config.use_f16_input {
        return gen_matmul_coop_f32_8x8(fused_add, variant, prologue, epilogue, copies);
    }
    let tile = config.tile_size;
    let output_tile = config.output_tile();
    let shared_size = tile * tile;
    let wg_size: u32 = 64;
    let staging_iters = shared_size / wg_size;
    let row_stride = wg_size / tile;
    let tile_mask = tile - 1;
    let tile_shift = tile.trailing_zeros();

    let compensated = config.compensated && config.use_f16_input;
    let (elem_type, enable_f16) = if config.use_f16_input {
        ("f16", "enable f16;")
    } else {
        ("f32", "")
    };
    let store = |shared: &str, idx: &str, f32_expr: &str| -> String {
        if compensated {
            format!(
                "{{ let _x = ({f32_expr}); let _h = f16(_x); {shared}[{idx}] = _h; {shared}_lo[{idx}] = f16(_x - f32(_h)); }}"
            )
        } else if config.use_f16_input {
            format!("{shared}[{idx}] = f16({f32_expr});")
        } else {
            format!("{shared}[{idx}] = {f32_expr};")
        }
    };
    let ab_type = if config.use_f16_input { "f16" } else { "f32" };
    // `coop_mat{columns}x{rows}`. Square tiles pass the same extent three times.
    let (tile_m, tile_n, tile_k) = (tile, tile, tile);
    let coop_ab = format!("coop_mat{tile_k}x{tile_m}<{ab_type},A>");
    let coop_ba = format!("coop_mat{tile_n}x{tile_k}<{ab_type},B>");
    let coop_c = format!("coop_mat{tile_n}x{tile_m}<f32,C>");

    let (prologue_decl, prologue_cache_decl, prologue_cache_init, a_transform) = match prologue {
        Some(p) => matmul_prologue_to_wgsl(p, output_tile),
        None => (String::new(), String::new(), String::new(), String::new()),
    };
    let (epilogue_decl, epilogue_body) = match epilogue {
        Some(e) => matmul_epilogue_to_wgsl(e),
        None => (String::new(), String::new()),
    };

    // Vec4 staging uses 128-bit loads when tile is 16+ (64 threads × 4 = 256 = 16×16).
    // "Direct" vec4: load along the shared-memory column axis (consecutive writes).
    //   B: Normal/AT (B[K,N], load along N)
    //   A: Normal/BT (A[M,K], load along K)
    // "Transposed" vec4: load along the contiguous global-memory axis, write strided.
    //   B: BT (B[N,K], load along K, write rows-of-shared)
    //   A: AT (A[K,M], load along M, write rows-of-shared)
    let use_vec4 = tile >= 16;
    // All variants use vec4 for both A and B (direct or transposed).
    // Transposed staging writes strided into shared (4 rows × 1 col per thread).
    // `vec4_b` is the "direct" staging that assumes B is [K,N]; it must be
    // FALSE for BT so the `vec4_b_transposed` branch is taken. Same reasoning
    // for `vec4_a` vs `vec4_a_transposed` on the AT variant.
    let vec4_b_transposed = use_vec4 && variant == MatMulCoopVariant::BT;
    let vec4_a_transposed = use_vec4 && variant == MatMulCoopVariant::AT && prologue.is_none();
    let vec4_b = use_vec4 && !vec4_b_transposed;
    let vec4_a = use_vec4 && prologue.is_none() && !vec4_a_transposed;

    // Both "direct" vec4 (vec4_{a,b}) and "transposed" vec4 staging use 128-bit
    // loads, so the backing storage must be `array<vec4<f32>>` in either case.
    let a_storage = if vec4_a || vec4_a_transposed {
        "array<vec4<f32>>"
    } else {
        "array<f32>"
    };
    let b_storage = if vec4_b || vec4_b_transposed {
        "array<vec4<f32>>"
    } else {
        "array<f32>"
    };

    // Generate hoisted staging index variables
    let staging_vars = {
        let mut s = String::new();
        if use_vec4 {
            s += "let v4_row = lid.x >> 2u;\n    let v4_col = (lid.x & 3u) << 2u;";
        }
        if !vec4_b || !vec4_a {
            if !s.is_empty() {
                s += "\n    ";
            }
            s += &format!(
                "let src_col = lid.x & {}u;\n    let base_row = lid.x >> {}u;",
                tile_mask, tile_shift
            );
        }
        if !vec4_b {
            s += &format!(
                "\n    let cc = tile_col + src_col;\
                 \n    let in_n = cc < n;\
                 \n    let cc1 = cc + {}u;\
                 \n    let in_n1 = cc1 < n;",
                tile
            );
        }
        s
    };

    // Generate B staging blocks (shared_a0, shared_a1)
    let (b_stage_0, b_stage_1) = if vec4_b {
        // Normal/AT: B[K,N], load vec4 along N (consecutive in memory).
        //
        // The fast vec4 packs the 4 lanes as four consecutive N-elements
        // for the same K-row, valid when `n % 4 == 0` and the tile fits.
        // Per-lane fallback handles non-multiple-of-4 N, but note that the
        // *output store* (`coopStoreT` with stride `n` for a 16-wide tile)
        // can still corrupt adjacent rows when N is not a multiple of 16.
        // The runtime auto-switch in runtime.rs refuses coop for non-16-
        // aligned N to keep the store safe — the slow-staging path here
        // is only exercised when N >= 16 but N % 4 != 0 (e.g., N=20).
        let gen_vec4_b = |shared: &str, col_offset: &str| -> String {
            let st_x = store(shared, "flat", "v.x");
            let st_y = store(shared, "flat + 1u", "v.y");
            let st_z = store(shared, "flat + 2u", "v.z");
            let st_w = store(shared, "flat + 3u", "v.w");
            let st_m0 = store(shared, "flat", "select(0.0, v0, m0)");
            let st_m1 = store(shared, "flat + 1u", "select(0.0, v1, m1)");
            let st_m2 = store(shared, "flat + 2u", "select(0.0, v2, m2)");
            let st_m3 = store(shared, "flat + 3u", "select(0.0, v3, m3)");
            let st_z0 = store(shared, "flat", "0.0");
            let st_z1 = store(shared, "flat + 1u", "0.0");
            let st_z2 = store(shared, "flat + 2u", "0.0");
            let st_z3 = store(shared, "flat + 3u", "0.0");
            format!(
                "{{\
               \n            let tr = t + v4_row;\
               \n            let cc4 = {col} + v4_col;\
               \n            let flat = v4_row * {t}u + v4_col;\
               \n            if tr < k && (cc4 + 4u) <= n && (n & 3u) == 0u {{\
               \n                let v = $B_BUFFER[(tr * n + cc4) >> 2u];\
               \n                {st_x}\
               \n                {st_y}\
               \n                {st_z}\
               \n                {st_w}\
               \n            }} else if tr < k {{\
               \n                let m0 = (cc4 + 0u) < n;\
               \n                let m1 = (cc4 + 1u) < n;\
               \n                let m2 = (cc4 + 2u) < n;\
               \n                let m3 = (cc4 + 3u) < n;\
               \n                let lastn = n - 1u;\
               \n                let a0 = tr * n + min(cc4 + 0u, lastn);\
               \n                let a1 = tr * n + min(cc4 + 1u, lastn);\
               \n                let a2 = tr * n + min(cc4 + 2u, lastn);\
               \n                let a3 = tr * n + min(cc4 + 3u, lastn);\
               \n                let v0 = $B_BUFFER[a0 >> 2u][a0 & 3u];\
               \n                let v1 = $B_BUFFER[a1 >> 2u][a1 & 3u];\
               \n                let v2 = $B_BUFFER[a2 >> 2u][a2 & 3u];\
               \n                let v3 = $B_BUFFER[a3 >> 2u][a3 & 3u];\
               \n                {st_m0}\
               \n                {st_m1}\
               \n                {st_m2}\
               \n                {st_m3}\
               \n            }} else {{\
               \n                {st_z0}\
               \n                {st_z1}\
               \n                {st_z2}\
               \n                {st_z3}\
               \n            }}\
               \n        }}",
                col = col_offset,
                t = tile,
            )
        };
        (
            gen_vec4_b("shared_a0", "tile_col"),
            gen_vec4_b("shared_a1", &format!("(tile_col + {}u)", tile)),
        )
    } else if vec4_b_transposed {
        // BT: B[N,K], load vec4 along K (consecutive in memory), write
        // transposed to shared.
        //
        // v4_row → N (cc direction, shared col), v4_col → K (tr direction,
        // shared rows). shared[row * tile + col] where row = K-offset,
        // col = N-offset.
        //
        // The fast vec4 load `$B_BUFFER[(cc * k + tr4) >> 2u]` packs the
        // 4 lanes as four consecutive K-elements for the same N-row. That
        // packing is only correct when `k % 4 == 0` (else the lanes pull
        // from the next N-row and the kernel misinterprets memory) *and*
        // the tile fully fits in K (`tr4 + 4 <= k`). For backward passes
        // where K is small (e.g. K=1 in MatMulBT(d_loss/d_pred [N,1],
        // w_out [hidden,1])) or non-multiple-of-4, take the slow path: a
        // per-lane scalar load through `array<vec4<f32>>` indexed by
        // `(addr >> 2u)[addr & 3u]` with K-bounds masking.
        let gen_vec4_bt = |shared: &str, col_offset: &str| -> String {
            let ix0 = format!("v4_col * {tile}u + v4_row");
            let ix1 = format!("(v4_col + 1u) * {tile}u + v4_row");
            let ix2 = format!("(v4_col + 2u) * {tile}u + v4_row");
            let ix3 = format!("(v4_col + 3u) * {tile}u + v4_row");
            let st_x = store(shared, &ix0, "v.x");
            let st_y = store(shared, &ix1, "v.y");
            let st_z = store(shared, &ix2, "v.z");
            let st_w = store(shared, &ix3, "v.w");
            let st_m0 = store(shared, &ix0, "select(0.0, v0, m0)");
            let st_m1 = store(shared, &ix1, "select(0.0, v1, m1)");
            let st_m2 = store(shared, &ix2, "select(0.0, v2, m2)");
            let st_m3 = store(shared, &ix3, "select(0.0, v3, m3)");
            let st_z0 = store(shared, &ix0, "0.0");
            let st_z1 = store(shared, &ix1, "0.0");
            let st_z2 = store(shared, &ix2, "0.0");
            let st_z3 = store(shared, &ix3, "0.0");
            format!(
                "{{\
               \n            let cc = {col} + v4_row;\
               \n            let tr4 = t + v4_col;\
               \n            if cc < n && (tr4 + 4u) <= k && (k & 3u) == 0u {{\
               \n                let v = $B_BUFFER[(cc * k + tr4) >> 2u];\
               \n                {st_x}\
               \n                {st_y}\
               \n                {st_z}\
               \n                {st_w}\
               \n            }} else if cc < n {{\
               \n                let m0 = (tr4 + 0u) < k;\
               \n                let m1 = (tr4 + 1u) < k;\
               \n                let m2 = (tr4 + 2u) < k;\
               \n                let m3 = (tr4 + 3u) < k;\
               \n                let lastk = k - 1u;\
               \n                let a0 = cc * k + min(tr4 + 0u, lastk);\
               \n                let a1 = cc * k + min(tr4 + 1u, lastk);\
               \n                let a2 = cc * k + min(tr4 + 2u, lastk);\
               \n                let a3 = cc * k + min(tr4 + 3u, lastk);\
               \n                let v0 = $B_BUFFER[a0 >> 2u][a0 & 3u];\
               \n                let v1 = $B_BUFFER[a1 >> 2u][a1 & 3u];\
               \n                let v2 = $B_BUFFER[a2 >> 2u][a2 & 3u];\
               \n                let v3 = $B_BUFFER[a3 >> 2u][a3 & 3u];\
               \n                {st_m0}\
               \n                {st_m1}\
               \n                {st_m2}\
               \n                {st_m3}\
               \n            }} else {{\
               \n                {st_z0}\
               \n                {st_z1}\
               \n                {st_z2}\
               \n                {st_z3}\
               \n            }}\
               \n        }}",
                col = col_offset,
            )
        };
        (
            gen_vec4_bt("shared_a0", "tile_col"),
            gen_vec4_bt("shared_a1", &format!("(tile_col + {}u)", tile)),
        )
    } else {
        // Scalar B staging (tile_size < 16 fallback)
        let (bi0, bi1) = match variant {
            MatMulCoopVariant::Normal | MatMulCoopVariant::AT => ("tr * n + cc", "tr * n + cc1"),
            MatMulCoopVariant::BT => ("cc * k + tr", "cc1 * k + tr"),
        };
        let gen_scalar_b = |shared: &str, in_col: &str, b_index: &str| -> String {
            let st = store(shared, "flat", &format!("$B_BUFFER[{b_index}]"));
            let stz = store(shared, "flat", "0.0");
            format!(
                "{{\
               \n            for (var e = 0u; e < {iters}u; e++) {{\
               \n                let flat = lid.x + e * 64u;\
               \n                let tr = t + base_row + e * {stride}u;\
               \n                let in_bounds = (tr < k) && {ic};\
               \n                if in_bounds {{\
               \n                    {st}\
               \n                }} else {{\
               \n                    {stz}\
               \n                }}\
               \n            }}\
               \n        }}",
                iters = staging_iters,
                stride = row_stride,
                ic = in_col,
            )
        };
        (
            gen_scalar_b("shared_a0", "in_n", bi0),
            gen_scalar_b("shared_a1", "in_n1", bi1),
        )
    };

    // Generate A staging blocks (shared_b0, shared_b1)
    let (a_stage_0, a_stage_1) = if vec4_a {
        // Normal/BT: A[M,K], load vec4 along K (consecutive in memory).
        //
        // The fast vec4 path packs 4 lanes as 4 consecutive K-elements for
        // the same M-row, which is only correct when `k % 4 == 0` and the
        // full vec4 lies within bounds. Otherwise fall back to per-lane
        // scalar loads via `(addr >> 2u)[addr & 3u]` with K-bounds masking.
        // Required for backward passes whose effective K is small (e.g.
        // d_loss/d_pred has shape [N, 1] in the BT of an MLP output head).
        let gen_vec4_a = |shared: &str, row_offset: &str| -> String {
            let st_x = store(shared, "flat", "v.x");
            let st_y = store(shared, "flat + 1u", "v.y");
            let st_z = store(shared, "flat + 2u", "v.z");
            let st_w = store(shared, "flat + 3u", "v.w");
            let st_m0 = store(shared, "flat", "select(0.0, v0, m0)");
            let st_m1 = store(shared, "flat + 1u", "select(0.0, v1, m1)");
            let st_m2 = store(shared, "flat + 2u", "select(0.0, v2, m2)");
            let st_m3 = store(shared, "flat + 3u", "select(0.0, v3, m3)");
            let st_z0 = store(shared, "flat", "0.0");
            let st_z1 = store(shared, "flat + 1u", "0.0");
            let st_z2 = store(shared, "flat + 2u", "0.0");
            let st_z3 = store(shared, "flat + 3u", "0.0");
            format!(
                "{{\
               \n            let gr = {row} + v4_row;\
               \n            let tc4 = t + v4_col;\
               \n            let flat = v4_row * {t}u + v4_col;\
               \n            if gr < m && (tc4 + 4u) <= k && (k & 3u) == 0u {{\
               \n                let v = matrix_a[(gr * k + tc4) >> 2u];\
               \n                {st_x}\
               \n                {st_y}\
               \n                {st_z}\
               \n                {st_w}\
               \n            }} else if gr < m {{\
               \n                let m0 = (tc4 + 0u) < k;\
               \n                let m1 = (tc4 + 1u) < k;\
               \n                let m2 = (tc4 + 2u) < k;\
               \n                let m3 = (tc4 + 3u) < k;\
               \n                let lastk = k - 1u;\
               \n                let a0 = gr * k + min(tc4 + 0u, lastk);\
               \n                let a1 = gr * k + min(tc4 + 1u, lastk);\
               \n                let a2 = gr * k + min(tc4 + 2u, lastk);\
               \n                let a3 = gr * k + min(tc4 + 3u, lastk);\
               \n                let v0 = matrix_a[a0 >> 2u][a0 & 3u];\
               \n                let v1 = matrix_a[a1 >> 2u][a1 & 3u];\
               \n                let v2 = matrix_a[a2 >> 2u][a2 & 3u];\
               \n                let v3 = matrix_a[a3 >> 2u][a3 & 3u];\
               \n                {st_m0}\
               \n                {st_m1}\
               \n                {st_m2}\
               \n                {st_m3}\
               \n            }} else {{\
               \n                {st_z0}\
               \n                {st_z1}\
               \n                {st_z2}\
               \n                {st_z3}\
               \n            }}\
               \n        }}",
                row = row_offset,
                t = tile,
            )
        };
        (
            gen_vec4_a("shared_b0", "tile_row"),
            gen_vec4_a("shared_b1", &format!("(tile_row + {}u)", tile)),
        )
    } else if vec4_a_transposed {
        // AT: A[K,M], load vec4 along M (consecutive in memory), write
        // transposed to shared. The packed vec4 load is valid only when M is
        // divisible by four: otherwise successive K rows start at different
        // lanes of the storage vec4. Partial rows and unaligned row strides
        // use scalar lane extraction, mirroring the Normal/BT staging paths.
        let gen_vec4_at = |shared: &str, row_offset: &str| -> String {
            let ix0 = format!("v4_col * {tile}u + v4_row");
            let ix1 = format!("(v4_col + 1u) * {tile}u + v4_row");
            let ix2 = format!("(v4_col + 2u) * {tile}u + v4_row");
            let ix3 = format!("(v4_col + 3u) * {tile}u + v4_row");
            let st_x = store(shared, &ix0, "v.x");
            let st_y = store(shared, &ix1, "v.y");
            let st_z = store(shared, &ix2, "v.z");
            let st_w = store(shared, &ix3, "v.w");
            let st_m0 = store(shared, &ix0, "select(0.0, v0, m0)");
            let st_m1 = store(shared, &ix1, "select(0.0, v1, m1)");
            let st_m2 = store(shared, &ix2, "select(0.0, v2, m2)");
            let st_m3 = store(shared, &ix3, "select(0.0, v3, m3)");
            let st_z0 = store(shared, &ix0, "0.0");
            let st_z1 = store(shared, &ix1, "0.0");
            let st_z2 = store(shared, &ix2, "0.0");
            let st_z3 = store(shared, &ix3, "0.0");
            format!(
                "{{\
               \n            let tc = t + v4_row;\
               \n            let gr4 = {row} + v4_col;\
               \n            if tc < k && (gr4 + 4u) <= m && (m & 3u) == 0u {{\
               \n                let v = matrix_a[(tc * m + gr4) >> 2u];\
               \n                {st_x}\
               \n                {st_y}\
               \n                {st_z}\
               \n                {st_w}\
               \n            }} else if tc < k {{\
               \n                let m0 = (gr4 + 0u) < m;\
               \n                let m1 = (gr4 + 1u) < m;\
               \n                let m2 = (gr4 + 2u) < m;\
               \n                let m3 = (gr4 + 3u) < m;\
               \n                let last = m - 1u;\
               \n                let a0 = tc * m + min(gr4 + 0u, last);\
               \n                let a1 = tc * m + min(gr4 + 1u, last);\
               \n                let a2 = tc * m + min(gr4 + 2u, last);\
               \n                let a3 = tc * m + min(gr4 + 3u, last);\
               \n                let v0 = matrix_a[a0 >> 2u][a0 & 3u];\
               \n                let v1 = matrix_a[a1 >> 2u][a1 & 3u];\
               \n                let v2 = matrix_a[a2 >> 2u][a2 & 3u];\
               \n                let v3 = matrix_a[a3 >> 2u][a3 & 3u];\
               \n                {st_m0}\
               \n                {st_m1}\
               \n                {st_m2}\
               \n                {st_m3}\
               \n            }} else {{\
               \n                {st_z0}\
               \n                {st_z1}\
               \n                {st_z2}\
               \n                {st_z3}\
               \n            }}\
               \n        }}",
                row = row_offset,
            )
        };
        (
            gen_vec4_at("shared_b0", "tile_row"),
            gen_vec4_at("shared_b1", &format!("(tile_row + {}u)", tile)),
        )
    } else {
        // Scalar A staging (prologue or tile_size < 16)
        let a_idx = match variant {
            MatMulCoopVariant::Normal | MatMulCoopVariant::BT => "gr * k + tc",
            MatMulCoopVariant::AT => "tc * m + gr",
        };
        let gen_scalar_a = |shared: &str, row_offset: &str| -> String {
            let st = store(shared, "flat", &format!("a_val{a_transform}"));
            let stz = store(shared, "flat", "0.0");
            format!(
                "{{\
               \n            let tc = t + src_col;\
               \n            let in_k = tc < k;\
               \n            for (var e = 0u; e < {iters}u; e++) {{\
               \n                let flat = lid.x + e * 64u;\
               \n                let gr = {row} + base_row + e * {stride}u;\
               \n                let in_bounds = (gr < m) && in_k;\
               \n                if in_bounds {{\
               \n                    let a_val = matrix_a[{ai}];\
               \n                    {st}\
               \n                }} else {{\
               \n                    {stz}\
               \n                }}\
               \n            }}\
               \n        }}",
                iters = staging_iters,
                stride = row_stride,
                row = row_offset,
                ai = a_idx,
            )
        };
        (
            gen_scalar_a("shared_b0", "tile_row"),
            gen_scalar_a("shared_b1", &format!("(tile_row + {}u)", tile)),
        )
    };

    let (fused_decl, mut acc_init) = if fused_add {
        (
            "var<storage> src: array<f32>;".to_string(),
            format!(
                "var acc00 = coopLoadT<{coop_c}>(&src[c00], n);\n\
                 \x20   var acc01 = coopLoadT<{coop_c}>(&src[c01], n);\n\
                 \x20   var acc10 = coopLoadT<{coop_c}>(&src[c10], n);\n\
                 \x20   var acc11 = coopLoadT<{coop_c}>(&src[c11], n);"
            ),
        )
    } else {
        (
            String::new(),
            format!(
                "var acc00 = {coop_c}();\n\
                 \x20   var acc01 = {coop_c}();\n\
                 \x20   var acc10 = {coop_c}();\n\
                 \x20   var acc11 = {coop_c}();"
            ),
        )
    };

    if fused_add && splits > 1 {
        acc_init.clear();
        for name in ["00", "01", "10", "11"] {
            acc_init.push_str(&format!("var acc{name} = {coop_c}();\n"));
        }
        acc_init.push_str("if wgid.z == 0u {\n");
        for name in ["00", "01", "10", "11"] {
            acc_init.push_str(&format!(
                "acc{name} = coopLoadT<{coop_c}>(&src[c{name}], n);\n"
            ));
        }
        acc_init.push_str("}\n");
    }
    let partition = if splits > 1 {
        format!(
            "let chunk = ((k + 15u) / 16u + {last}u) / {splits}u * 16u;\n    let begin = wgid.z * chunk;\n    let end = min(k, begin + chunk);",
            last = splits - 1
        )
    } else {
        String::new()
    };

    let output_tile_u = format!("{}u", output_tile);
    let tile_size_u = format!("{}u", tile);
    let shared_size_s = format!("{}", shared_size);
    let result_shared_size = output_tile * output_tile;
    let (result_shared_decl, result_store) = if epilogue.is_some() {
        // Cooperative matrices are subgroup-scoped. Only one subgroup may
        // store these shared tiles; every invocation still reaches the barrier.
        let store_iters = result_shared_size.div_ceil(wg_size);
        (
            format!(
                "var<workgroup> shared_c: array<f32, {}>;",
                result_shared_size
            ),
            format!(
                "if sg == 0u {{\n\
                 \x20   coopStoreT(acc00, &shared_c[0], {output_tile}u);\n\
                 \x20   coopStoreT(acc01, &shared_c[{tile}u], {output_tile}u);\n\
                 \x20   coopStoreT(acc10, &shared_c[{}u], {output_tile}u);\n\
                 \x20   coopStoreT(acc11, &shared_c[{}u], {output_tile}u);\n\
                 \x20   }}\n\
                 \x20   workgroupBarrier();\n\
                 \n\
                 \x20   for (var e = 0u; e < {store_iters}u; e++) {{\n\
                 \x20       let local_idx = lid.x + e * {wg_size}u;\n\
                 \x20       if local_idx < {result_shared_size}u {{\n\
                 \x20           let local_row = local_idx / {output_tile}u;\n\
                 \x20           let local_col = local_idx - local_row * {output_tile}u;\n\
                 \x20           let row = tile_row + local_row;\n\
                 \x20           let col = tile_col + local_col;\n\
                 \x20           if row < m && col < n {{\n\
                 \x20               let idx = row * n + col;\n\
                 \x20               var val = shared_c[local_idx];\n\
                 \x20               {epilogue_body}\n\
                 \x20               $C_BUFFER[idx] = val;\n\
                 \x20           }}\n\
                 \x20       }}\n\
                 \x20   }}",
                tile * output_tile,
                tile * output_tile + tile,
            ),
        )
    } else {
        (
            String::new(),
            "if sg == 0u {\n\
             \x20   coopStoreT(acc00, &$C_BUFFER[o00], n);\n\
             \x20   if n1_valid {\n\
             \x20       coopStoreT(acc01, &$C_BUFFER[o01], n);\n\
             \x20   }\n\
             \x20   if m1_valid {\n\
             \x20       coopStoreT(acc10, &$C_BUFFER[o10], n);\n\
             \x20   }\n\
             \x20   if n1_valid && m1_valid {\n\
             \x20       coopStoreT(acc11, &$C_BUFFER[o11], n);\n\
             \x20   }\n\
             \x20   }"
                .to_string(),
        )
    };

    let (shared_lo_decl, compensated_mma) = if compensated {
        (
            format!(
                "var<workgroup> shared_a0_lo: array<f16, {shared_size}>;\n\
                 var<workgroup> shared_a1_lo: array<f16, {shared_size}>;\n\
                 var<workgroup> shared_b0_lo: array<f16, {shared_size}>;\n\
                 var<workgroup> shared_b1_lo: array<f16, {shared_size}>;"
            ),
            format!(
                "let a0_lo = coopLoadT<{coop_ab}>(&shared_b0_lo[0], {tile}u);\n\
                 \x20   let a1_lo = coopLoadT<{coop_ab}>(&shared_b1_lo[0], {tile}u);\n\
                 \x20   let b0_lo = coopLoadT<{coop_ba}>(&shared_a0_lo[0], {tile}u);\n\
                 \x20   let b1_lo = coopLoadT<{coop_ba}>(&shared_a1_lo[0], {tile}u);\n\
                 \x20   acc00 = coopMultiplyAdd(a0, b0_lo, acc00);\n\
                 \x20   acc00 = coopMultiplyAdd(a0_lo, b0, acc00);\n\
                 \x20   acc01 = coopMultiplyAdd(a0, b1_lo, acc01);\n\
                 \x20   acc01 = coopMultiplyAdd(a0_lo, b1, acc01);\n\
                 \x20   acc10 = coopMultiplyAdd(a1, b0_lo, acc10);\n\
                 \x20   acc10 = coopMultiplyAdd(a1_lo, b0, acc10);\n\
                 \x20   acc11 = coopMultiplyAdd(a1, b1_lo, acc11);\n\
                 \x20   acc11 = coopMultiplyAdd(a1_lo, b1, acc11);"
            ),
        )
    } else {
        (String::new(), String::new())
    };

    let src = include_str!("shaders/matmul_coop.wgsl");
    let src = preprocess(
        src,
        &[
            ("$PARTITION", &partition),
            ("$K_BEGIN", if splits > 1 { "begin" } else { "0u" }),
            ("$K_END", if splits > 1 { "end" } else { "k" }),
            (
                "$OUTPUT_BASE",
                if splits > 1 { "wgid.z * m * n" } else { "0u" },
            ),
            ("$ENABLE_F16", enable_f16),
            ("$ELEM_TYPE", elem_type),
            ("$SHARED_SIZE", &shared_size_s),
            ("$OUTPUT_M_U", &output_tile_u),
            ("$OUTPUT_N_U", &output_tile_u),
            ("$TILE_M_U", &tile_size_u),
            ("$TILE_N_U", &tile_size_u),
            ("$TILE_K_U", &tile_size_u),
            ("$COOP_AB", &coop_ab),
            ("$COOP_BA", &coop_ba),
            ("$A_STORAGE", a_storage),
            ("$STAGING_VARS", &staging_vars),
            ("$B_STAGE_0", &b_stage_0),
            ("$B_STAGE_1", &b_stage_1),
            ("$A_STAGE_0", &a_stage_0),
            ("$A_STAGE_1", &a_stage_1),
            ("$PROLOGUE_DECL", &prologue_decl),
            ("$EPILOGUE_DECL", &epilogue_decl),
            ("$PROLOGUE_CACHE_DECL", &prologue_cache_decl),
            ("$PROLOGUE_CACHE_INIT", &prologue_cache_init),
            ("$FUSED_ADD_DECL", &fused_decl),
            ("$ACC_INIT", &acc_init),
            ("$RESULT_SHARED_DECL", &result_shared_decl),
            ("$RESULT_STORE", &result_store),
            ("$SHARED_LO_DECL", &shared_lo_decl),
            ("$COMPENSATED_MMA", &compensated_mma),
        ],
    );

    matmul_module(&src, b_storage, Some(wg_size), copies)
}

/// F32 8x8 cooperative matrices: four SIMD groups own disjoint
/// output quadrants, sharing 32 values along K between barriers.
fn gen_matmul_coop_f32_8x8(
    fused_add: bool,
    variant: MatMulCoopVariant,
    prologue: Option<&crate::compile::MatMulPrologue>,
    epilogue: Option<&crate::compile::MatMulEpilogue>,
    copies: u32,
) -> ShaderModule {
    let (prologue_decl, cache_decl, cache_init, a_transform) = prologue
        .map(|p| matmul_prologue_to_wgsl(p, 32))
        .unwrap_or_default();
    let (epilogue_decl, epilogue_body) = epilogue.map(matmul_epilogue_to_wgsl).unwrap_or_default();
    // Always load along the contiguous global axis. Transposed variants
    // scatter the four components into the shared tile after the load.
    // Keep scalar storage elements: a trailing partial vec4 must not extend
    // beyond the logical buffer binding, even when an allocation is padded.
    let stage = |is_a: bool, transposed: bool| {
        let (buffer, shared, row_base, col_base, rows, stride) = match (is_a, transposed) {
            (true, false) => ("matrix_a", "shared_a", "tile_row", "t", "m", "k"),
            (true, true) => ("matrix_a", "shared_a", "t", "tile_row", "k", "m"),
            (false, false) => ("$B_BUFFER", "shared_b", "t", "tile_col", "k", "n"),
            (false, true) => ("$B_BUFFER", "shared_b", "tile_col", "t", "n", "k"),
        };
        let mut fast = String::new();
        let mut edge = String::new();
        for (lane, component) in ["x", "y", "z", "w"].iter().enumerate() {
            let shared_index = if transposed {
                format!("(local_col + {lane}u) * 32u + local_row")
            } else {
                format!("local_row * 32u + local_col + {lane}u")
            };
            let coordinates = if is_a && !a_transform.is_empty() {
                if transposed {
                    format!("let gr = gc + {lane}u; let tc = gr_global;\n")
                } else {
                    format!("let gr = gr_global; let tc = gc + {lane}u;\n")
                }
            } else {
                String::new()
            };
            let transform = if is_a { a_transform.as_str() } else { "" };
            fast.push_str(&format!(
                "{{ {coordinates} {shared}[{shared_index}] = v.{component}{transform}; }}\n"
            ));
            edge.push_str(&format!(
                "{{\n\
                    var value = 0.0;\n\
                    if gr_global < {rows} && gc + {lane}u < {stride} {{\n\
                        let address = gr_global * {stride} + gc + {lane}u;\n\
                        {coordinates}\
                        value = {buffer}[address]{transform};\n\
                    }}\n\
                    {shared}[{shared_index}] = value;\n\
                }}\n"
            ));
        }
        format!(
            "for (var flat = lid.x; flat < 256u; flat += 128u) {{\n\
                let local_row = flat / 8u;\n\
                let local_col = (flat % 8u) * 4u;\n\
                let gr_global = {row_base} + local_row;\n\
                let gc = {col_base} + local_col;\n\
                if gr_global < {rows} && gc + 4u <= {stride} {{\n\
                    let address = gr_global * {stride} + gc;\n\
                    let v = vec4<f32>({buffer}[address], {buffer}[address + 1u], {buffer}[address + 2u], {buffer}[address + 3u]);\n\
                    {fast}\n\
                }} else {{\n\
                    {edge}\n\
                }}\n\
            }}"
        )
    };
    let a_stage = stage(true, variant == MatMulCoopVariant::AT);
    let b_stage = stage(false, variant == MatMulCoopVariant::BT);
    let mut acc_init = String::new();
    let mut result_store = String::new();
    for (name, row, col) in [("00", 0, 0), ("01", 0, 8), ("10", 8, 0), ("11", 8, 8)] {
        let offset = format!("(tile_row + sg_row + {row}u) * n + tile_col + sg_col + {col}u");
        // Hoist pointer indices before the cooperative store statement.
        // Naga's SPIR-V backend needs their scalar expressions emitted first.
        acc_init.push_str(&format!("let c{name} = {offset};\n"));
        let init = if fused_add {
            acc_init.push_str(&format!(
                "let add{name} = (sg_row + {row}u) * 32u + sg_col + {col}u;\n"
            ));
            format!("coopLoadT<coop_mat8x8<f32,C>>(&shared_a[add{name}], 32u)")
        } else {
            "coop_mat8x8<f32,C>()".to_owned()
        };
        acc_init.push_str(&format!("var acc{name} = {init};\n"));
        if epilogue.is_some() {
            acc_init.push_str(&format!(
                "let s{name} = (sg_row + {row}u) * 32u + sg_col + {col}u;\n"
            ));
            result_store.push_str(&format!(
                "if tile_sg < 4u {{ coopStoreT(acc{name}, &shared_c[s{name}], 32u); }}\n"
            ));
        } else {
            result_store.push_str(&format!(
                "if tile_sg < 4u && tile_row + sg_row + {row}u < m && tile_col + sg_col + {col}u < n {{\n\
                 coopStoreT(acc{name}, &$C_BUFFER[c{name}], n);\n}}\n"
            ));
        }
    }
    if epilogue.is_some() {
        result_store.push_str(&format!(
            "workgroupBarrier();\n\
             for (var local_idx = lid.x; local_idx < 1024u; local_idx += 128u) {{\n\
                 let row = tile_row + local_idx / 32u;\n\
                 let col = tile_col + local_idx % 32u;\n\
                 let owner = (local_idx / 512u) * 2u + (local_idx % 32u) / 16u;\n\
                 if row < m && col < n && owner >= group_base && owner < group_base + 128u / sg_size {{\n\
                     let idx = row * n + col;\n\
                     var val = shared_c[local_idx];\n\
                     {epilogue_body}\n\
                     $C_BUFFER[idx] = val;\n\
                 }}\n\
             }}"
        ));
    }
    let src = preprocess(
        include_str!("shaders/matmul_coop_f32_8x8.wgsl"),
        &[
            ("$PROLOGUE_DECL", &prologue_decl),
            ("$PROLOGUE_CACHE_DECL", &cache_decl),
            ("$PROLOGUE_CACHE_INIT", &cache_init),
            ("$EPILOGUE_DECL", &epilogue_decl),
            ("$A_STAGE", &a_stage),
            ("$B_STAGE", &b_stage),
            ("$ACC_INIT", &acc_init),
            (
                "$ADDEND_STAGE",
                if fused_add {
                    "for (var flat = lid.x; flat < 1024u; flat += 128u) {\n\
                        let row = tile_row + flat / 32u;\n\
                        let col = tile_col + flat % 32u;\n\
                        var value = 0.0;\n\
                        if row < m && col < n { value = src[row * n + col]; }\n\
                        shared_a[flat] = value;\n\
                    }\n\
                    workgroupBarrier();"
                } else {
                    ""
                },
            ),
            (
                "$ACC_READY",
                if fused_add { "workgroupBarrier();" } else { "" },
            ),
            ("$RESULT_STORE", &result_store),
            (
                "$FUSED_ADD_DECL",
                if fused_add {
                    "var<storage> src: array<f32>;"
                } else {
                    ""
                },
            ),
            (
                "$RESULT_SHARED_DECL",
                if epilogue.is_some() {
                    "var<workgroup> shared_c: array<f32, 1024>;"
                } else {
                    ""
                },
            ),
        ],
    );
    matmul_module(&src, "array<f32>", Some(128), copies)
}

/// Variant selector for gen_matmul_coop_inner.
#[derive(Clone, Copy, PartialEq)]
pub enum MatMulCoopVariant {
    /// C = A @ B  (standard)
    Normal,
    /// `C = A @ B^T` (`B` is `[N,K]`, accessed transposed)
    BT,
    /// `C = A^T @ B` (`A` is `[K,M]`, accessed transposed)
    AT,
}

/// Cooperative-matrix tile sizes supported by the device, filtered by precision policy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct CoopCaps {
    pub f16_tile: u32,
    pub f32_tile: u32,
}

impl CoopCaps {
    pub fn is_supported(&self) -> bool {
        self.f16_tile > 0 || self.f32_tile > 0
    }
    /// True when the kernels that hardcode `coop_mat16x16<f16, ...>`
    /// can run (NVIDIA, RDNA3, Xe-HPG). False on Apple's 8x8 f32 path
    /// or any GPU without KHR_cooperative_matrix at the right tile size.
    pub fn supports_16x16_f16(&self) -> bool {
        self.f16_tile == 16
    }
}

#[cfg(test)]
mod coop_caps_tests {
    use super::CoopCaps;

    #[test]
    fn cooperative_caps_respect_tile_and_precision_requirements() {
        assert!(
            CoopCaps {
                f16_tile: 16,
                f32_tile: 0,
            }
            .supports_16x16_f16()
        );
        for tile in [0, 8, 32] {
            assert!(
                !CoopCaps {
                    f16_tile: tile,
                    f32_tile: 0,
                }
                .supports_16x16_f16(),
                "hard-coded 16x16 shaders must reject tile {tile}",
            );
        }
        use crate::CoopPolicy;
        for f32_tile in [0, 8, 16] {
            let caps = CoopCaps {
                f16_tile: 16,
                f32_tile,
            };
            let strict = CoopPolicy::NativeF32.filter_caps(caps);
            assert_eq!(strict.f32_tile, f32_tile);
            assert!(!strict.supports_16x16_f16());
            assert_eq!(strict.is_supported(), f32_tile != 0);
            assert_eq!(CoopPolicy::Disabled.filter_caps(caps), CoopCaps::default());
            assert_eq!(CoopPolicy::Auto.filter_caps(caps), caps);
            assert_eq!(CoopPolicy::AllowF16.filter_caps(caps), caps);
        }
    }
}

mod attention;
pub use attention::*;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::attention_grad::{AttentionGrad, Operands, Part, Path};

    const COOP_F16: Path = Path::Cooperative(Operands::F16);

    #[test]
    fn ordinary_weight_gradients_have_no_split_partition_logic() {
        for tile in [MatMulTile::Small, MatMulTile::Large] {
            let ordinary = conv_grad_weight_tiled(tile, false, 16, None);
            assert!(!ordinary.source.contains("num_workgroups"));
            assert!(!ordinary.source.contains("k_end"));
            assert!(ordinary.source.contains("var t = 0u;"));
            let split = conv_grad_weight_tiled(tile, true, 16, None);
            assert!(split.source.contains("num_workgroups"));
            assert!(split.source.contains("var t = first * 16u;"));
        }
    }

    #[test]
    fn all_shaders_generate_valid_modules() {
        let groups = [
            (
                ShaderGroup::ChunkedRelativeAttention,
                naga::valid::Capabilities::empty(),
            ),
            (ShaderGroup::Sgd, naga::valid::Capabilities::empty()),
            (ShaderGroup::Adam, naga::valid::Capabilities::empty()),
            (ShaderGroup::Transpose, naga::valid::Capabilities::empty()),
            (ShaderGroup::MatMul, naga::valid::Capabilities::empty()),
            (ShaderGroup::MatMulAdd, naga::valid::Capabilities::empty()),
            (ShaderGroup::MatMulAT, naga::valid::Capabilities::empty()),
            (ShaderGroup::MatMulBT, naga::valid::Capabilities::empty()),
            (ShaderGroup::BlockMatMul, naga::valid::Capabilities::empty()),
            (
                ShaderGroup::BlockMatMulAT,
                naga::valid::Capabilities::empty(),
            ),
            (
                ShaderGroup::BlockMatMulBT,
                naga::valid::Capabilities::empty(),
            ),
            (ShaderGroup::BatchMatMul, naga::valid::Capabilities::empty()),
            (
                ShaderGroup::BatchMatMulAT,
                naga::valid::Capabilities::empty(),
            ),
            (
                ShaderGroup::BatchMatMulBT,
                naga::valid::Capabilities::empty(),
            ),
            (ShaderGroup::MatMulATAdd, naga::valid::Capabilities::empty()),
            (ShaderGroup::MatMulBTAdd, naga::valid::Capabilities::empty()),
            // The GEMV family's reduction is a generation-time choice, and
            // the subgroup form needs `SUBGROUP`. Derive the requirement from
            // the shape these groups are actually generated with rather than
            // asserting the tree's, or this table silently stops describing
            // what `generate_module` emits.
            (ShaderGroup::MatMulGemv, gemv_caps(ShaderGroup::MatMulGemv)),
            (
                ShaderGroup::MatMulGemvAdd,
                gemv_caps(ShaderGroup::MatMulGemvAdd),
            ),
            (
                ShaderGroup::MatMulGemvBT,
                gemv_caps(ShaderGroup::MatMulGemvBT),
            ),
            (
                ShaderGroup::MatMulGemvBTAdd,
                gemv_caps(ShaderGroup::MatMulGemvBTAdd),
            ),
            (ShaderGroup::Reduce, naga::valid::Capabilities::empty()),
            (
                ShaderGroup::CrossEntropy,
                naga::valid::Capabilities::empty(),
            ),
            (ShaderGroup::RmsNormAdd, naga::valid::Capabilities::empty()),
            (ShaderGroup::Embedding, naga::valid::Capabilities::empty()),
            (
                ShaderGroup::ToF16,
                naga::valid::Capabilities::SHADER_FLOAT16,
            ),
            (ShaderGroup::RoPE, naga::valid::Capabilities::empty()),
            (ShaderGroup::RoPEGrad, naga::valid::Capabilities::empty()),
            (ShaderGroup::LayerNorm, naga::valid::Capabilities::empty()),
            (
                ShaderGroup::MultiHeadAttn,
                naga::valid::Capabilities::empty(),
            ),
            (
                ShaderGroup::FlashAttention,
                naga::valid::Capabilities::empty(),
            ),
            (
                ShaderGroup::FlashAttentionCoop,
                naga::valid::Capabilities::COOPERATIVE_MATRIX
                    | naga::valid::Capabilities::SHADER_FLOAT16
                    | naga::valid::Capabilities::SUBGROUP,
            ),
            (ShaderGroup::SwiGLUGrad, naga::valid::Capabilities::empty()),
            (
                ShaderGroup::SwiGLUConcat,
                naga::valid::Capabilities::empty(),
            ),
            (ShaderGroup::SumRows, naga::valid::Capabilities::empty()),
            (
                ShaderGroup::Conv2dGradWeightGemm,
                naga::valid::Capabilities::empty(),
            ),
            (
                ShaderGroup::Conv2dGradWeightGemmSmall,
                naga::valid::Capabilities::empty(),
            ),
            (
                ShaderGroup::Conv2dGradWeightGemm16,
                naga::valid::Capabilities::empty(),
            ),
            (
                ShaderGroup::Conv2dGradWeightGemmSplit,
                naga::valid::Capabilities::empty(),
            ),
            (
                ShaderGroup::Conv2dGradWeightGemmSplitSmall,
                naga::valid::Capabilities::empty(),
            ),
            (
                ShaderGroup::Conv2dGradWeightGemmSplit16,
                naga::valid::Capabilities::empty(),
            ),
            (ShaderGroup::RmsNormGrad, naga::valid::Capabilities::empty()),
            (
                ShaderGroup::RmsNormGradWRowPar,
                naga::valid::Capabilities::empty(),
            ),
            (ShaderGroup::ScatterAdd, naga::valid::Capabilities::empty()),
            (
                ShaderGroup::ScatterAddAtomic,
                naga::valid::Capabilities::empty(),
            ),
            (ShaderGroup::BceLoss, naga::valid::Capabilities::empty()),
            (
                ShaderGroup::GlobalAvgPoolGrad,
                naga::valid::Capabilities::empty(),
            ),
            (
                ShaderGroup::PairwiseGrad,
                naga::valid::Capabilities::empty(),
            ),
            (
                ShaderGroup::GradClipNormSq,
                naga::valid::Capabilities::empty(),
            ),
            (
                ShaderGroup::GradClipScale,
                naga::valid::Capabilities::empty(),
            ),
            (
                ShaderGroup::AdaptiveGradClip,
                naga::valid::Capabilities::empty(),
            ),
            (ShaderGroup::GradAccum, naga::valid::Capabilities::empty()),
        ];
        let groups: Vec<_> = groups
            .iter()
            .copied()
            .chain(
                AttentionGrad::ALL
                    .map(|kernel| (ShaderGroup::AttentionGrad(kernel), kernel.capabilities())),
            )
            .collect();

        let flags = naga::valid::ValidationFlags::all() ^ naga::valid::ValidationFlags::BINDINGS;
        for &(group, caps) in &groups {
            let sm = generate_module(group, MatmulKnobs::default());
            naga::valid::Validator::new(flags, caps)
                .validate(&sm.module)
                .unwrap_or_else(|e| {
                    panic!("{group:?}: generated module failed validation: {e:#?}")
                });
        }

        // Cooperative execution is a modifier rather than a group, so its
        // modules are reached through the scalar group they derive from.
        let coop_caps = naga::valid::Capabilities::COOPERATIVE_MATRIX
            | naga::valid::Capabilities::SHADER_FLOAT16
            | naga::valid::Capabilities::SUBGROUP;
        let config = CoopConfig {
            tile_size: 16,
            use_f16_input: true,
            compensated: false,
        };
        for &(group, caps) in &groups {
            if coop_shape(group).is_none() {
                continue;
            }
            let sm = generate_module_coop(group, &config);
            naga::valid::Validator::new(flags, caps | coop_caps)
                .validate(&sm.module)
                .unwrap_or_else(|e| {
                    panic!("{group:?} (coop): generated module failed validation: {e:#?}")
                });
        }
    }

    /// Verify the generated modules contain the expected entry points.
    #[test]
    fn entry_points_present() {
        let m = generate_module(ShaderGroup::Reduce, MatmulKnobs::default());
        let names: Vec<&str> = m
            .module
            .entry_points
            .iter()
            .map(|ep| ep.name.as_str())
            .collect();
        assert!(names.contains(&"sum_all"));
        assert!(names.contains(&"mean_all"));

        let m = generate_module(ShaderGroup::SumRows, MatmulKnobs::default());
        let names: Vec<&str> = m
            .module
            .entry_points
            .iter()
            .map(|ep| ep.name.as_str())
            .collect();
        assert!(names.contains(&"sum_rows"));
    }

    #[test]
    fn test_rms_norm_wgsl() {
        let _ = generate_wgsl(ShaderGroup::RmsNormAdd);
    }

    #[test]
    fn test_embedding_wgsl() {
        let _ = generate_wgsl(ShaderGroup::Embedding);
    }

    #[test]
    fn test_rope_wgsl() {
        let _ = generate_wgsl(ShaderGroup::RoPE);
    }

    #[test]
    fn test_rope_grad_wgsl() {
        let _ = generate_wgsl(ShaderGroup::RoPEGrad);
    }

    #[test]
    fn test_unified_attention_wgsl() {
        let _ = generate_attention_module(64);
        let _ = generate_attention_module(32);
        let _ = generate_attention_module(128);
    }

    #[test]
    fn cooperative_attention_staging_types_and_storage() {
        for head_dim in [16, 32, 64, 128, 256, 512] {
            for (group, shader, scalar_arrays) in [
                (
                    ShaderGroup::FlashAttentionCoop,
                    generate_flash_attention_coop_module(head_dim),
                    &["shared_v"][..],
                ),
                (
                    ShaderGroup::AttentionGrad(AttentionGrad::new(Part::Q, COOP_F16)),
                    generate_flash_grad_q_coop_f16_module(head_dim),
                    &["shared_k"][..],
                ),
                (
                    ShaderGroup::AttentionGrad(AttentionGrad::new(Part::KV, COOP_F16)),
                    generate_flash_grad_kv_coop_f16_module(head_dim),
                    &["shared_q", "shared_do"][..],
                ),
            ] {
                for &name in scalar_arrays {
                    let (_, var) = shader
                        .module
                        .global_variables
                        .iter()
                        .find(|&(_, var)| var.name.as_deref() == Some(name))
                        .expect("scalar staging array");
                    let naga::TypeInner::Array { base, .. } = shader.module.types[var.ty].inner
                    else {
                        panic!("{group:?}: {name} must be an array");
                    };
                    assert!(
                        matches!(
                            shader.module.types[base].inner,
                            naga::TypeInner::Scalar(naga::Scalar {
                                kind: naga::ScalarKind::Float,
                                width: 4
                            })
                        ),
                        "{group:?}: {name} must preserve f32 precision"
                    );
                }
                let mut layout = naga::proc::Layouter::default();
                layout.update(shader.module.to_ctx()).unwrap();
                let mut bytes = 0u32;
                for (_, var) in shader.module.global_variables.iter() {
                    if var.space == naga::AddressSpace::WorkGroup {
                        let ty = layout[var.ty];
                        bytes = ty.alignment.round_up(bytes) + ty.size;
                    }
                }
                assert_eq!(
                    attention_coop_shared_bytes(group, head_dim),
                    u64::from(bytes),
                    "{group:?}, head_dim={head_dim}",
                );
            }
        }
    }

    #[test]
    fn scalar_attention_backward_staging_fits_16_kib() {
        for head_dim in [16, 32, 64, 80, 128, 256, 512, 1024] {
            for cap in [4, 8, 16, 32, 64] {
                for shader in [
                    generate_flash_grad_q_module(head_dim, cap),
                    generate_flash_grad_kv_module(head_dim, cap),
                ] {
                    naga::valid::Validator::new(
                        naga::valid::ValidationFlags::all()
                            & !naga::valid::ValidationFlags::BINDINGS,
                        naga::valid::Capabilities::all(),
                    )
                    .validate(&shader.module)
                    .unwrap();
                    let mut layout = naga::proc::Layouter::default();
                    layout.update(shader.module.to_ctx()).unwrap();
                    let mut bytes = 0;
                    for (_, var) in shader.module.global_variables.iter() {
                        if var.space == naga::AddressSpace::WorkGroup {
                            let ty = layout[var.ty];
                            bytes = ty.alignment.round_up(bytes) + ty.size;
                        }
                    }
                    assert!(
                        bytes <= 16384,
                        "{}: head={head_dim}, cap={cap}, bytes={bytes}",
                        shader.hint
                    );
                }
            }
        }
    }

    #[test]
    fn coop_f32_attention_staging_preserves_f32_and_matches_storage_gate() {
        for (shader, expected_bytes) in [
            (
                generate_flash_grad_q_coop_f32_module(64),
                FLASH_GRAD_COOP_F32_SHARED_BYTES,
            ),
            (
                generate_flash_grad_kv_coop_f32_module(64),
                FLASH_GRAD_COOP_F32_SHARED_BYTES,
            ),
        ] {
            assert!(!shader.source.contains("f16"));
            let mut layout = naga::proc::Layouter::default();
            layout.update(shader.module.to_ctx()).unwrap();
            let mut bytes = 0;
            for (_, var) in shader.module.global_variables.iter() {
                if var.space != naga::AddressSpace::WorkGroup {
                    continue;
                }
                let naga::TypeInner::Array { base, .. } = shader.module.types[var.ty].inner else {
                    panic!("f32 cooperative attention workgroup storage must be an f32 array");
                };
                assert!(matches!(
                    shader.module.types[base].inner,
                    naga::TypeInner::Scalar(naga::Scalar {
                        kind: naga::ScalarKind::Float,
                        width: 4
                    })
                ));
                bytes += layout[var.ty].size;
            }
            assert_eq!(bytes, expected_bytes);
        }
    }

    #[test]
    fn test_flash_attention_wgsl() {
        let mut shape = FlashAttentionShape::default();
        shape.fit_shared_memory(1024, 32768);
        assert_eq!(shape.keys, 2);
        assert!(shape.shared_bytes(1024) <= 32768);
        for hd in [32, 64, 128, 256] {
            for ept in [8, 16, 32] {
                for interleave in [false, true] {
                    for (threads, keys) in [(256, 8), (128, 16), (256, 2), (128, 4)] {
                        let sm = generate_flash_attention_module(
                            hd,
                            ept,
                            FlashAttentionShape {
                                threads,
                                keys,
                                interleave,
                            },
                        );
                        naga::valid::Validator::new(
                            naga::valid::ValidationFlags::all()
                                ^ naga::valid::ValidationFlags::BINDINGS,
                            naga::valid::Capabilities::empty(),
                        )
                        .validate(&sm.module)
                        .unwrap();
                    }
                }
            }
        }
    }

    #[test]
    fn grad_clip_norm_avoids_device_scope_storage_atomics() {
        let module = generate_module(ShaderGroup::GradClipNormSq, MatmulKnobs::default());
        assert!(!module.source.contains("array<atomic"));
        assert!(!module.source.contains("atomicLoad"));
    }

    #[test]
    fn scatter_add_uses_float_cas() {
        let module = generate_module(ShaderGroup::ScatterAddAtomic, MatmulKnobs::default());
        assert!(module.source.contains("array<atomic"));
        assert!(module.source.contains("atomicLoad"));
        assert!(module.source.contains("atomicCompareExchangeWeak"));
    }

    #[test]
    fn horizontal_matmul_modules_validate() {
        use naga::valid::{Capabilities, ValidationFlags, Validator};
        let flags = ValidationFlags::all() ^ ValidationFlags::BINDINGS;
        let coop = CoopConfig {
            tile_size: 16,
            use_f16_input: true,
            compensated: false,
        };
        let compensated = CoopConfig {
            tile_size: 16,
            use_f16_input: true,
            compensated: true,
        };
        for count in [2u32, 3] {
            for group in [
                ShaderGroup::MatMul,
                ShaderGroup::MatMulAT,
                ShaderGroup::MatMulBT,
            ] {
                let module = generate_horizontal_matmul(group, count, None);
                assert!(module.source.contains("matrix_b0"));
                assert!(module.source.contains("horiz_0"));
                Validator::new(flags, Capabilities::empty())
                    .validate(&module.module)
                    .unwrap_or_else(|e| panic!("horizontal {group:?} {count} failed: {e:#?}"));
            }
            for (label, cfg, caps) in [
                (
                    "coop",
                    coop,
                    Capabilities::COOPERATIVE_MATRIX
                        | Capabilities::SHADER_FLOAT16
                        | Capabilities::SUBGROUP,
                ),
                (
                    "compensated",
                    compensated,
                    Capabilities::COOPERATIVE_MATRIX
                        | Capabilities::SHADER_FLOAT16
                        | Capabilities::SUBGROUP,
                ),
            ] {
                let module = generate_horizontal_matmul(ShaderGroup::MatMul, count, Some(&cfg));
                assert!(module.source.contains("matrix_b0"));
                Validator::new(flags, caps)
                    .validate(&module.module)
                    .unwrap_or_else(|e| panic!("horizontal {label} {count} failed: {e:#?}"));
            }
        }
    }

    #[test]
    fn coop_f32_multi_simd_matmul_modules_validate() {
        use crate::compile::{BufferRef, MatMulEpilogue, MatMulPrologue, PrologueLoadKind};
        use crate::schedule::{PointwiseDAG, Pw};
        use naga::valid::{Capabilities, ValidationFlags, Validator};
        let config = CoopConfig {
            tile_size: 8,
            use_f16_input: false,
            compensated: false,
        };
        assert_eq!(config.matmul_output_tile(), 32);
        assert_eq!(
            config.output_tile(),
            16,
            "convolution geometry is unchanged"
        );
        let prologue = MatMulPrologue {
            factors: vec![
                (BufferRef(3), PrologueLoadKind::PerRow),
                (BufferRef(4), PrologueLoadKind::PerKCol),
            ],
        };
        let epilogue = MatMulEpilogue {
            dag: PointwiseDAG {
                n_inputs: 1,
                ops: vec![Pw::LoadInput(0), Pw::Relu(0)],
                output: 1,
            },
            inputs: Vec::new(),
        };
        for group in [
            ShaderGroup::MatMul,
            ShaderGroup::MatMulAdd,
            ShaderGroup::MatMulAT,
            ShaderGroup::MatMulATAdd,
            ShaderGroup::MatMulBT,
            ShaderGroup::MatMulBTAdd,
        ] {
            for copies in [1, 2, 3] {
                for (prologue, epilogue) in [
                    (None, None),
                    (Some(&prologue), None),
                    (None, Some(&epilogue)),
                ] {
                    let (fused_add, variant) = coop_shape(group).unwrap();
                    let module = gen_matmul_coop_wgsl_full(
                        fused_add, variant, &config, prologue, epilogue, copies, 1,
                    );
                    assert_eq!(module.module.entry_points[0].workgroup_size, [128, 1, 1]);
                    assert!(!module.source.contains("enable f16"));
                    assert!(module.source.contains("group_base += 128u / sg_size"));
                    Validator::new(
                        ValidationFlags::all() ^ ValidationFlags::BINDINGS,
                        Capabilities::COOPERATIVE_MATRIX | Capabilities::SUBGROUP,
                    )
                    .validate(&module.module)
                    .unwrap_or_else(|error| panic!("{group:?}, {copies} copies: {error:#?}"));
                }
            }
        }
    }

    #[test]
    fn compensated_f16_coop_validates() {
        use naga::valid::{Capabilities, ValidationFlags, Validator};
        let config = CoopConfig {
            tile_size: 16,
            use_f16_input: true,
            compensated: true,
        };
        for group in [
            ShaderGroup::MatMul,
            ShaderGroup::MatMulAT,
            ShaderGroup::MatMulBT,
        ] {
            let module = generate_module_coop(group, &config);
            assert!(
                module.source.contains("shared_a0_lo"),
                "{group:?} missing residual shared"
            );
            Validator::new(
                ValidationFlags::all() ^ ValidationFlags::BINDINGS,
                Capabilities::COOPERATIVE_MATRIX
                    | Capabilities::SHADER_FLOAT16
                    | Capabilities::SUBGROUP,
            )
            .validate(&module.module)
            .unwrap_or_else(|error| panic!("{group:?} compensated coop failed: {error:#?}"));
        }
    }

    #[test]
    fn cooperative_matmul_epilogue_stages_and_validates() {
        use crate::compile::MatMulEpilogue;
        use crate::schedule::{PointwiseDAG, Pw};
        use naga::valid::{Capabilities, ValidationFlags, Validator};

        let epilogue = MatMulEpilogue {
            dag: PointwiseDAG {
                n_inputs: 1,
                ops: vec![Pw::LoadInput(0), Pw::Silu(0), Pw::Relu(1)],
                output: 2,
            },
            inputs: Vec::new(),
        };
        let config = CoopConfig {
            tile_size: 16,
            use_f16_input: true,
            compensated: false,
        };
        let capabilities = Capabilities::COOPERATIVE_MATRIX
            | Capabilities::SHADER_FLOAT16
            | Capabilities::SUBGROUP;
        let flags = ValidationFlags::all() ^ ValidationFlags::BINDINGS;

        for group in [
            ShaderGroup::MatMul,
            ShaderGroup::MatMulAdd,
            ShaderGroup::MatMulAT,
            ShaderGroup::MatMulBT,
        ] {
            let module = generate_coop_matmul_with_dag_epilogue(group, &config, &epilogue);
            assert!(module.source.contains("shared_c"));
            assert!(module.source.contains("var val = shared_c[local_idx]"));
            Validator::new(flags, capabilities)
                .validate(&module.module)
                .unwrap_or_else(|error| {
                    panic!("{group:?} cooperative epilogue failed validation: {error:#?}")
                });
        }
    }

    /// Verify every shader group compiles to SPIR-V without panics.
    /// This catches "Expression [N] is not cached!" bugs in hand-built IR.
    /// Skipped on Apple targets where naga's spv-out backend is not available.
    #[test]
    #[cfg(not(target_vendor = "apple"))]
    fn all_shaders_compile_to_spirv() {
        let empty = naga::valid::Capabilities::empty();
        let f16 = naga::valid::Capabilities::SHADER_FLOAT16;
        let coop = naga::valid::Capabilities::COOPERATIVE_MATRIX
            | naga::valid::Capabilities::SHADER_FLOAT16
            | naga::valid::Capabilities::SUBGROUP;
        let groups: &[(ShaderGroup, naga::valid::Capabilities)] = &[
            (ShaderGroup::Sgd, empty),
            (ShaderGroup::Adam, empty),
            (ShaderGroup::Transpose, empty),
            (ShaderGroup::MatMul, empty),
            (ShaderGroup::MatMulAdd, empty),
            (ShaderGroup::MatMulAT, empty),
            (ShaderGroup::MatMulBT, empty),
            (ShaderGroup::BlockMatMul, empty),
            (ShaderGroup::BlockMatMulAT, empty),
            (ShaderGroup::BlockMatMulBT, empty),
            (ShaderGroup::BatchMatMul, empty),
            (ShaderGroup::BatchMatMulAT, empty),
            (ShaderGroup::BatchMatMulBT, empty),
            (ShaderGroup::MatMulATAdd, empty),
            (ShaderGroup::MatMulBTAdd, empty),
            (ShaderGroup::Reduce, empty),
            (ShaderGroup::CrossEntropy, empty),
            (ShaderGroup::RmsNormAdd, empty),
            (ShaderGroup::Embedding, empty),
            (ShaderGroup::ToF16, f16),
            (ShaderGroup::RoPE, empty),
            (ShaderGroup::RoPEGrad, empty),
            (ShaderGroup::LayerNorm, empty),
            (ShaderGroup::MultiHeadAttn, empty),
            (ShaderGroup::FlashAttention, empty),
            (ShaderGroup::FlashAttentionCoop, coop),
            (ShaderGroup::SwiGLUGrad, empty),
            (ShaderGroup::SwiGLUConcat, empty),
            (ShaderGroup::SumRows, empty),
            (ShaderGroup::Conv2dGradWeightGemm, empty),
            (ShaderGroup::Conv2dGradWeightGemmSmall, empty),
            (ShaderGroup::Conv2dGradWeightGemm16, empty),
            (ShaderGroup::Conv2dGradWeightGemmSplit, empty),
            (ShaderGroup::Conv2dGradWeightGemmSplitSmall, empty),
            (ShaderGroup::Conv2dGradWeightGemmSplit16, empty),
            (ShaderGroup::RmsNormGrad, empty),
            (ShaderGroup::RmsNormGradWRowPar, empty),
            (ShaderGroup::ScatterAdd, empty),
            (ShaderGroup::ScatterAddAtomic, empty),
            (ShaderGroup::BceLoss, empty),
            (ShaderGroup::GlobalAvgPoolGrad, empty),
            (ShaderGroup::PairwiseGrad, empty),
            (ShaderGroup::GradClipNormSq, empty),
            (ShaderGroup::GradClipScale, empty),
            (ShaderGroup::AdaptiveGradClip, empty),
            (ShaderGroup::GradAccum, empty),
        ];
        let groups: Vec<_> = groups
            .iter()
            .copied()
            .chain(
                AttentionGrad::ALL
                    .map(|kernel| (ShaderGroup::AttentionGrad(kernel), kernel.capabilities())),
            )
            .collect();

        let flags = naga::valid::ValidationFlags::all() ^ naga::valid::ValidationFlags::BINDINGS;
        let options = naga::back::spv::Options {
            lang_version: (1, 0),
            flags: naga::back::spv::WriterFlags::empty(),
            capabilities: None,
            bounds_check_policies: naga::proc::BoundsCheckPolicies::default(),
            binding_map: Default::default(),
            ..Default::default()
        };

        let mut failed = Vec::new();
        for &(group, caps) in &groups {
            // See note in all_shaders_generate_valid_modules
            if matches!(
                group,
                ShaderGroup::Conv2dGemmCoop | ShaderGroup::Conv2dGradInputGemmCoop
            ) {
                continue;
            }
            let sm = generate_module(group, MatmulKnobs::default());
            let info = match naga::valid::Validator::new(flags, caps).validate(&sm.module) {
                Ok(info) => info,
                Err(e) => {
                    failed.push(format!("{group:?}: validation failed: {e}"));
                    continue;
                }
            };
            // Try each entry point
            for ep in &sm.module.entry_points {
                let pipeline_options = naga::back::spv::PipelineOptions {
                    shader_stage: naga::ShaderStage::Compute,
                    entry_point: ep.name.clone(),
                };
                let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    naga::back::spv::write_vec(&sm.module, &info, &options, Some(&pipeline_options))
                }));
                match result {
                    Ok(Ok(_)) => {}
                    Ok(Err(e)) => failed.push(format!("{group:?}/{}: SPIR-V error: {e}", ep.name)),
                    Err(e) => {
                        let msg = e
                            .downcast_ref::<String>()
                            .map(|s| s.as_str())
                            .or_else(|| e.downcast_ref::<&str>().copied())
                            .unwrap_or("unknown panic");
                        failed.push(format!("{group:?}/{}: SPIR-V panic: {msg}", ep.name));
                    }
                }
            }
        }
        if !failed.is_empty() {
            panic!("SPIR-V compilation failures:\n{}", failed.join("\n"));
        }
    }

    /// Verify that shader global variable names match the runtime ShaderData
    /// struct field names. Blade resolves bindings by name — a mismatch causes
    /// a runtime panic ("Unable to resolve binding for ...").
    #[test]
    fn shader_globals_match_runtime_bindings() {
        use crate::compile::ShaderEntry;
        use std::collections::HashSet;

        // Expected global variable names for each ShaderEntry, derived from
        // the runtime ShaderData structs. Workgroup vars (tile_a, tile_b) and
        // builtin args are not bound by blade and can be ignored.
        fn expected_globals(entry: &ShaderEntry) -> Vec<&'static str> {
            match *entry {
                // Generated kernels bind by their kernel's own layout.
                ShaderEntry::Generated => Vec::new(),
                ShaderEntry::BlockMatMul
                | ShaderEntry::BlockMatMulAT
                | ShaderEntry::BlockMatMulBT
                | ShaderEntry::BatchMatMul
                | ShaderEntry::BatchMatMulAT
                | ShaderEntry::BatchMatMulBT => vec!["matrix_a", "matrix_b", "matrix_c", "params"],
                ShaderEntry::MatMul
                | ShaderEntry::MatMulAT
                | ShaderEntry::MatMulBT
                | ShaderEntry::MatMulGemv
                | ShaderEntry::MatMulGemvBT => {
                    vec!["matrix_a", "matrix_b", "matrix_c", "params"]
                }
                ShaderEntry::MatMulGemvAdd | ShaderEntry::MatMulGemvBTAdd => {
                    vec!["matrix_a", "matrix_b", "matrix_c", "src", "params"]
                }
                ShaderEntry::FusedMatMulAdd
                | ShaderEntry::FusedMatMulATAdd
                | ShaderEntry::FusedMatMulBTAdd => {
                    vec!["matrix_a", "matrix_b", "matrix_c", "src", "params"]
                }
                ShaderEntry::SumAll
                | ShaderEntry::MeanAll
                | ShaderEntry::SumRows
                | ShaderEntry::RoPE
                | ShaderEntry::RoPEGrad => vec!["src", "dst", "params"],
                ShaderEntry::SgdUpdate => vec!["segments", "param", "grad", "params"],
                ShaderEntry::AdamUpdate => vec![
                    "segments",
                    "param",
                    "grad",
                    "m",
                    "v",
                    "grouped_grad_norm",
                    "params",
                ],
                ShaderEntry::ScatterAdd => vec!["indices", "src", "dst", "params"],
                ShaderEntry::ScatterAddAtomic => {
                    vec!["indices", "src", "row_scale", "dst", "params"]
                }
                ShaderEntry::BceLoss => vec!["pred", "labels", "loss_out", "params"],
                ShaderEntry::CrossEntropyLoss | ShaderEntry::CrossEntropyLossIndices => {
                    vec!["logits", "labels", "grad_out", "loss_out", "params"]
                }
                ShaderEntry::Transpose => vec!["src", "dst", "params"],
                ShaderEntry::RmsNormAdd => vec!["src", "bias", "residual", "dst", "params"],
                ShaderEntry::Embedding => vec!["indices", "src", "dst", "params"],
                ShaderEntry::ToF16 => vec!["src", "dst", "params"],
                ShaderEntry::LayerNorm => vec!["src", "src_b", "bias", "dst", "params"],
                ShaderEntry::MultiHeadAttn
                | ShaderEntry::FlashAttention
                | ShaderEntry::FlashAttentionCoop => {
                    vec!["src_a", "src_b", "bias", "dst", "lse", "params"]
                }
                ShaderEntry::AttentionGrad(kernel) => kernel.globals().to_vec(),
                // All three SwiGLUGrad entries share the same module globals
                ShaderEntry::SwiGLUGradGate | ShaderEntry::SwiGLUGradUp | ShaderEntry::SiluGrad => {
                    vec!["src_a", "src_b", "src_c", "dst", "params"]
                }
                ShaderEntry::SwiGLUConcat
                | ShaderEntry::SwiGLUConcatGrad
                | ShaderEntry::GeGLUConcat
                | ShaderEntry::GeGLUConcatGrad => {
                    vec!["src_a", "src_b", "dst", "params"]
                }
                ShaderEntry::RmsNormGradW
                | ShaderEntry::RmsNormGradWRowPar
                | ShaderEntry::RmsNormGradX => {
                    vec!["src_a", "src_b", "bias", "dst", "params"]
                }
                ShaderEntry::LayerNormGradWB | ShaderEntry::LayerNormGradX => {
                    vec!["src_a", "src_b", "bias", "dst", "params"]
                }
                ShaderEntry::RmsNormRsqrt => vec!["src", "dst", "params"],
                ShaderEntry::CacheWrite => vec!["src", "dst", "kv_pos_buf", "params"],
                ShaderEntry::CacheWritePrefix => {
                    vec!["src", "dst", "kv_pos_buf", "valid_len_buf", "params"]
                }
                ShaderEntry::CachedAttention | ShaderEntry::CachedQueryAttention => {
                    vec!["src_a", "src_b", "bias", "kv_pos_buf", "dst", "params"]
                }
                ShaderEntry::CachedBlockAttention | ShaderEntry::CachedBlockAttentionSplit => {
                    vec![
                        "src_a",
                        "src_b",
                        "bias",
                        "kv_pos_buf",
                        "valid_len_buf",
                        "dst",
                        "params",
                    ]
                }
                ShaderEntry::CachedBlockAttentionCombine => {
                    vec!["partials", "dst", "params"]
                }
                ShaderEntry::ChunkedRelativeAttention => {
                    vec!["src_a", "src_b", "bias", "relative_k", "dst", "params"]
                }
                ShaderEntry::PrefixLast => vec!["src", "valid_len_buf", "dst", "params"],
                ShaderEntry::GroupNorm | ShaderEntry::GroupNormSilu => {
                    vec!["src", "src_b", "bias", "dst", "params"]
                }
                ShaderEntry::GroupNormApply => {
                    vec!["src", "src_b", "bias", "dst", "partials", "params"]
                }
                ShaderEntry::GroupNormStats => vec!["src", "dst", "params"],
                ShaderEntry::GroupNormGradInput
                | ShaderEntry::GroupNormGradWeightBias
                | ShaderEntry::GroupNormGradStats => {
                    vec!["src_a", "src_b", "bias", "dst", "stats", "params"]
                }
                ShaderEntry::Concat => vec!["src_a", "src_b", "dst", "params"],
                ShaderEntry::BiasedAttention => {
                    vec!["q", "k", "v", "bias", "kv_pos", "dst", "params"]
                }
                ShaderEntry::SplitA | ShaderEntry::SplitB | ShaderEntry::Permute => {
                    vec!["src", "dst", "params"]
                }
                ShaderEntry::Upsample2x | ShaderEntry::Upsample2xGrad => {
                    vec!["src", "dst", "params"]
                }
                ShaderEntry::Conv2dDw => {
                    vec!["src", "weight", "dst", "params"]
                }
                ShaderEntry::MulPerChannel => vec!["src", "gate", "dst", "params"],
                ShaderEntry::Conv2dGemm
                | ShaderEntry::Conv2dGemmSmall
                | ShaderEntry::Conv2dGemm16
                | ShaderEntry::Conv2dGemmCoopGen(..) => vec!["src", "weight", "dst", "params"],
                ShaderEntry::Conv2dGradInputGemm
                | ShaderEntry::Conv2dGradInputGemmSmall
                | ShaderEntry::Conv2dGradInputGemm16
                | ShaderEntry::Conv2dGradInputGemmCoopGen(..) => {
                    vec!["grad_out", "weight", "dst", "params"]
                }
                ShaderEntry::Conv2dGradWeightGemm
                | ShaderEntry::Conv2dGradWeightGemmSmall
                | ShaderEntry::Conv2dGradWeightGemm16
                | ShaderEntry::Conv2dGradWeightGemmSplit
                | ShaderEntry::Conv2dGradWeightGemmSplitSmall
                | ShaderEntry::Conv2dGradWeightGemmSplit16 => {
                    vec!["grad_out", "src", "dst", "params"]
                }
                ShaderEntry::RoPEDynamic | ShaderEntry::RoPEPositions => {
                    vec!["src", "dst", "pos_offset_buf", "params"]
                }
                ShaderEntry::RoPEDynamicFactors => {
                    vec!["src", "dst", "pos_offset_buf", "factors", "params"]
                }
                ShaderEntry::MaxPool2d
                | ShaderEntry::GlobalAvgPool
                | ShaderEntry::GlobalAvgPoolGrad => vec!["src", "dst", "params"],
                ShaderEntry::MaxPool2dGrad => vec!["grad_out", "src", "dst", "params"],
                ShaderEntry::PairwiseGrad => {
                    vec!["src_a", "src_b", "src_c", "dst", "params"]
                }
                ShaderEntry::WinogradInputTransform | ShaderEntry::WinogradOutputTransform => {
                    vec!["src", "dst", "params"]
                }
                ShaderEntry::WinogradBatchedMatMul => {
                    vec!["matrix_a", "matrix_b", "matrix_c", "params"]
                }
                ShaderEntry::WinogradWeightTransform => vec!["src", "dst", "params"],
                ShaderEntry::GradClipNormSq
                | ShaderEntry::GradClipScale
                | ShaderEntry::GradAccum => vec!["segments", "grad", "acc", "params"],
                ShaderEntry::AdaptiveGradClip => {
                    vec!["segments", "param", "grad", "partials", "scales", "params"]
                }
            }
        }

        let entries = [
            ShaderEntry::BlockMatMul,
            ShaderEntry::BlockMatMulAT,
            ShaderEntry::BlockMatMulBT,
            ShaderEntry::BatchMatMul,
            ShaderEntry::BatchMatMulAT,
            ShaderEntry::BatchMatMulBT,
            ShaderEntry::MatMul,
            ShaderEntry::MatMulAT,
            ShaderEntry::MatMulBT,
            ShaderEntry::MatMulGemv,
            ShaderEntry::MatMulGemvAdd,
            ShaderEntry::MatMulGemvBT,
            ShaderEntry::MatMulGemvBTAdd,
            ShaderEntry::FusedMatMulAdd,
            ShaderEntry::FusedMatMulATAdd,
            ShaderEntry::FusedMatMulBTAdd,
            ShaderEntry::SgdUpdate,
            ShaderEntry::GradClipNormSq,
            ShaderEntry::GradClipScale,
            ShaderEntry::AdaptiveGradClip,
            ShaderEntry::GradAccum,
            ShaderEntry::SumAll,
            ShaderEntry::MeanAll,
            ShaderEntry::CrossEntropyLoss,
            ShaderEntry::CrossEntropyLossIndices,
            ShaderEntry::Transpose,
            ShaderEntry::Embedding,
            ShaderEntry::RoPE,
            ShaderEntry::RoPEGrad,
            ShaderEntry::LayerNorm,
            ShaderEntry::MultiHeadAttn,
            ShaderEntry::FlashAttention,
            ShaderEntry::FlashAttentionCoop,
            ShaderEntry::SwiGLUGradGate,
            ShaderEntry::SwiGLUGradUp,
            ShaderEntry::SwiGLUConcat,
            ShaderEntry::SwiGLUConcatGrad,
            ShaderEntry::GeGLUConcat,
            ShaderEntry::GeGLUConcatGrad,
            ShaderEntry::SiluGrad,
            ShaderEntry::RmsNormGradW,
            ShaderEntry::RmsNormGradWRowPar,
            ShaderEntry::RmsNormGradX,
            ShaderEntry::LayerNormGradWB,
            ShaderEntry::LayerNormGradX,
            ShaderEntry::RmsNormRsqrt,
            ShaderEntry::AdamUpdate,
            ShaderEntry::ScatterAdd,
            ShaderEntry::ScatterAddAtomic,
            ShaderEntry::BceLoss,
            ShaderEntry::GroupNorm,
            ShaderEntry::GroupNormSilu,
            ShaderEntry::GroupNormStats,
            ShaderEntry::GroupNormApply,
            ShaderEntry::GroupNormGradInput,
            ShaderEntry::GroupNormGradWeightBias,
            ShaderEntry::GroupNormGradStats,
            ShaderEntry::Concat,
            ShaderEntry::SplitA,
            ShaderEntry::SplitB,
            ShaderEntry::Permute,
            ShaderEntry::BiasedAttention,
            ShaderEntry::Upsample2x,
            ShaderEntry::Upsample2xGrad,
            ShaderEntry::Conv2dDw,
            ShaderEntry::MulPerChannel,
            ShaderEntry::Conv2dGemm,
            ShaderEntry::Conv2dGemmSmall,
            ShaderEntry::Conv2dGemm16,
            ShaderEntry::Conv2dGradInputGemm,
            ShaderEntry::Conv2dGradInputGemmSmall,
            ShaderEntry::Conv2dGradInputGemm16,
            ShaderEntry::Conv2dGradWeightGemm,
            ShaderEntry::Conv2dGradWeightGemmSmall,
            ShaderEntry::Conv2dGradWeightGemm16,
            ShaderEntry::Conv2dGradWeightGemmSplit,
            ShaderEntry::Conv2dGradWeightGemmSplitSmall,
            ShaderEntry::Conv2dGradWeightGemmSplit16,
            ShaderEntry::WinogradInputTransform,
            ShaderEntry::WinogradOutputTransform,
            ShaderEntry::WinogradBatchedMatMul,
            ShaderEntry::CacheWrite,
            ShaderEntry::CacheWritePrefix,
            ShaderEntry::CachedAttention,
            ShaderEntry::CachedQueryAttention,
            ShaderEntry::CachedBlockAttention,
            ShaderEntry::ChunkedRelativeAttention,
            ShaderEntry::PrefixLast,
            ShaderEntry::RoPEDynamic,
            ShaderEntry::RoPEDynamicFactors,
            ShaderEntry::RoPEPositions,
            ShaderEntry::MaxPool2d,
            ShaderEntry::MaxPool2dGrad,
            ShaderEntry::GlobalAvgPool,
            ShaderEntry::GlobalAvgPoolGrad,
            ShaderEntry::PairwiseGrad,
        ];
        let entries: Vec<_> = entries
            .into_iter()
            .chain(AttentionGrad::ALL.map(ShaderEntry::AttentionGrad))
            .collect();

        for entry in &entries {
            let group = entry.shader_group();
            let expected: HashSet<&str> = expected_globals(entry).into_iter().collect();

            let sm = generate_module(group, MatmulKnobs::default());
            let info = naga::valid::Validator::new(
                naga::valid::ValidationFlags::all() ^ naga::valid::ValidationFlags::BINDINGS,
                naga::valid::Capabilities::all(),
            )
            .validate(&sm.module)
            .unwrap();
            let ep_index = sm
                .module
                .entry_points
                .iter()
                .position(|ep| ep.name == entry.entry_point())
                .unwrap();
            let ep_info = info.get_entry_point(ep_index);

            let actual: HashSet<&str> = sm
                .module
                .global_variables
                .iter()
                .filter_map(|(handle, gv)| {
                    // Blade binds only resources used by this entry point.
                    if gv.space == naga::AddressSpace::WorkGroup || ep_info[handle].is_empty() {
                        return None;
                    }
                    gv.name.as_deref()
                })
                .collect();

            assert!(
                actual.is_subset(&expected),
                "{entry:?} (group {group:?}): shader globals {actual:?} include resources \
                 absent from the runtime bindings {expected:?}"
            );
        }
    }

    /// Verify that `generate_conv2d_coop_module` produces valid WGSL+naga modules
    /// for various kernel configs, both forward and backward.
    #[test]
    fn generated_conv2d_coop_modules_are_valid() {
        use naga::valid::{Capabilities, ValidationFlags, Validator};

        let coop_caps = Capabilities::COOPERATIVE_MATRIX
            | Capabilities::SHADER_FLOAT16
            | Capabilities::SUBGROUP;
        let flags = ValidationFlags::all() ^ ValidationFlags::BINDINGS;
        let configs = [
            CoopConfig {
                tile_size: 8,
                use_f16_input: false,
                compensated: false,
            },
            CoopConfig {
                tile_size: 16,
                use_f16_input: false,
                compensated: false,
            },
            CoopConfig {
                tile_size: 16,
                use_f16_input: true,
                compensated: false,
            },
        ];

        // Test several kernel configs x direction combinations
        let cases = [
            (1, 1, 1, Conv2dCoopDirection::Forward),
            (1, 1, 1, Conv2dCoopDirection::GradInput),
            (3, 3, 1, Conv2dCoopDirection::Forward),
            (3, 3, 1, Conv2dCoopDirection::GradInput),
            (3, 3, 2, Conv2dCoopDirection::Forward),
            (3, 3, 2, Conv2dCoopDirection::GradInput),
            (2, 4, 1, Conv2dCoopDirection::GradInput),
            (3, 2, 2, Conv2dCoopDirection::GradInput),
            (5, 5, 1, Conv2dCoopDirection::GradInput),
            (7, 7, 2, Conv2dCoopDirection::Forward),
        ];

        for config in configs {
            for &(kh, kw, stride, direction) in &cases {
                let sm = generate_conv2d_coop_module(kh, kw, stride, direction, &config);
                let mut validator = Validator::new(flags, coop_caps);
                let result = validator.validate(&sm.module);
                assert!(
                    result.is_ok(),
                    "Conv2d coop gen (tile {} {kh}x{kw} s{stride} {direction:?}) failed validation: {:?}",
                    config.tile_size,
                    result.err()
                );
                // f32 grad-input kernels store only fully covered
                // workgroups directly; edge workgroups (right or bottom) go
                // through bounds-checked scalar stores.
                let full_tile_gate = ") <= n_total && (tile_row + ";
                if !config.use_f16_input && direction == Conv2dCoopDirection::GradInput {
                    assert!(sm.source.contains(full_tile_gate));
                    assert!(sm.source.contains(") <= m_total {"));
                    assert!(sm.source.contains("shared_b0[flat]"));
                } else {
                    assert!(!sm.source.contains(full_tile_gate));
                }
            }
        }
    }

    /// The BT groups are deliberately absent: Q4 blocks run along the
    /// parameter's first dimension while the decoder indexes along K, so
    /// the shader this used to generate read the wrong block. Covered by
    /// `quantized_formats_refuse_transposed_b`.
    #[test]
    fn q4_matmul_shader_generates() {
        for group in [
            ShaderGroup::MatMul,
            ShaderGroup::MatMulAdd,
            ShaderGroup::MatMulAT,
            ShaderGroup::MatMulATAdd,
        ] {
            let sm = generate_module_weighted(group, WeightFormat::Q4, MatmulKnobs::default());
            assert!(
                sm.source.contains("dequant_q4"),
                "Q4 {group:?}: missing dequant_q4"
            );
            assert!(
                sm.source.contains("array<u32>"),
                "Q4 {group:?}: missing array<u32>"
            );
            if group == ShaderGroup::MatMul {
                assert!(
                    sm.source.contains("let unpacked = dequant_q4_pack8("),
                    "Q4 tiled MatMul must call pack8 from its B staging"
                );
            }
            eprintln!("Q4 {group:?} shader: {} chars", sm.source.len());
        }
    }

    /// Naga capabilities a GEMV group's generated module needs.
    fn gemv_caps(group: ShaderGroup) -> naga::valid::Capabilities {
        match GemvShape::initial(group).reduction {
            GemvReduction::Tree => naga::valid::Capabilities::empty(),
            GemvReduction::Subgroup => naga::valid::Capabilities::SUBGROUP,
        }
    }

    #[test]
    fn every_gemv_shape_composes_with_every_weight_format() {
        let legacy: GemvShape =
            serde_json::from_str(r#"{"threads":32,"reduction":"Tree"}"#).unwrap();
        assert_eq!(legacy.bt_rows, 1);
        assert_eq!(legacy.column_groups, 1);
        let formats = [
            (WeightFormat::F32, "matrix_b: array<vec4<f32>>"),
            (WeightFormat::F16, "array<vec4<f16>>"),
            (WeightFormat::Q4, "dequant_q4("),
            (WeightFormat::Q8, "dequant_q8("),
            (WeightFormat::Q40, "dequant_q40("),
            (WeightFormat::Q4K, "dequant_q4k("),
            (WeightFormat::Q6K, "dequant_q6k("),
            (WeightFormat::Q5K, "dequant_q5k("),
            (WeightFormat::Q3K, "dequant_q3k("),
        ];
        for (format, marker) in formats {
            for group in [
                ShaderGroup::MatMulGemv,
                ShaderGroup::MatMulGemvAdd,
                ShaderGroup::MatMulGemvBT,
                ShaderGroup::MatMulGemvBTAdd,
            ] {
                if format.is_quantized()
                    && matches!(
                        group,
                        ShaderGroup::MatMulGemvBT | ShaderGroup::MatMulGemvBTAdd
                    )
                {
                    continue;
                }
                for threads in [32, 64, 128, 256] {
                    for reduction in [GemvReduction::Tree, GemvReduction::Subgroup] {
                        for bt_rows in GemvShape::BT_ROWS {
                            let shape = GemvShape {
                                threads,
                                reduction,
                                bt_rows,
                                column_groups: 1,
                            }
                            .for_group(group);
                            if shape.bt_rows != bt_rows {
                                continue;
                            }
                            let module = generate_module_gemv(group, format, shape);
                            let caps = match reduction {
                                GemvReduction::Tree => naga::valid::Capabilities::empty(),
                                GemvReduction::Subgroup => naga::valid::Capabilities::SUBGROUP,
                            } | match format {
                                // f16 storage reads real `f16` values.
                                WeightFormat::F16 => naga::valid::Capabilities::SHADER_FLOAT16,
                                // Block scales are f16 bit patterns decoded with
                                // `unpack2x16float` into f32, which is the weaker
                                // capability — it needs no f16 arithmetic type.
                                f if f.is_quantized() => {
                                    naga::valid::Capabilities::SHADER_FLOAT16_IN_FLOAT32
                                }
                                _ => naga::valid::Capabilities::empty(),
                            };
                            let flags = naga::valid::ValidationFlags::all()
                                ^ naga::valid::ValidationFlags::BINDINGS;
                            naga::valid::Validator::new(flags, caps)
                                .validate(&module.module)
                                .unwrap_or_else(|e| {
                                    panic!(
                                        "{format:?} {group:?} {shape:?} failed validation: {e:#?}"
                                    )
                                });
                            let source = module.source;
                            assert!(
                                source.contains(marker),
                                "{format:?} {group:?} {shape:?} lost its B representation"
                            );
                            assert!(
                                source.contains(&format!("const LANES: u32 = {threads}u;")),
                                "{format:?} {group:?} {shape:?} kept the declared width"
                            );
                            let subgroup = reduction == GemvReduction::Subgroup;
                            assert_eq!(
                                source.contains("subgroupAdd"),
                                subgroup,
                                "{format:?} {group:?} {shape:?} reduction mismatch"
                            );
                            // The tree leaves two partials; the subgroup form produces a total.
                            assert_eq!(
                                source.contains("reduce_buf[lid.x] + reduce_buf[lid.x + 1u]"),
                                !subgroup,
                                "{format:?} {group:?} {shape:?} store expression mismatch"
                            );
                            // The fused add is applied after the reduction.
                            if group == ShaderGroup::MatMulGemvAdd {
                                assert!(
                                    source.contains("src[col4]"),
                                    "{format:?} {shape:?} dropped the fused addend"
                                );
                            }
                            if group == ShaderGroup::MatMulGemvBTAdd {
                                assert!(source.contains(if bt_rows == 1 {
                                    "src[col]"
                                } else {
                                    "src[col + 0u]"
                                }));
                            }
                        }
                    }
                }
            }
        }
    }

    /// Integer-dot templates share the shape slots and select hardware or scalar dot products.
    #[test]
    fn the_int_dot_gemv_takes_every_shape() {
        for group in [ShaderGroup::MatMulGemv, ShaderGroup::MatMulGemvAdd] {
            for format in [
                crate::compile::WeightFormat::Q40,
                crate::compile::WeightFormat::Q8,
            ] {
                let dot_call = match format {
                    crate::compile::WeightFormat::Q40 => "return dot4I8Packed(q4, q8)",
                    crate::compile::WeightFormat::Q8 => "return dot4I8Packed(a, b)",
                    _ => unreachable!(),
                };
                let helper_name = match format {
                    crate::compile::WeightFormat::Q40 => "fn dot_q4_q8_packed",
                    crate::compile::WeightFormat::Q8 => "fn dot_q8_q8_packed",
                    _ => unreachable!(),
                };
                for threads in [32, 64, 128, 256] {
                    for reduction in [GemvReduction::Tree, GemvReduction::Subgroup] {
                        let shape = GemvShape {
                            threads,
                            reduction,
                            bt_rows: 1,
                            column_groups: 1,
                        };
                        for packed_dot in [true, false] {
                            for norm in [false, true] {
                                let module = generate_module_gemv_int_dot(
                                    group, format, shape, packed_dot, norm,
                                );
                                let caps = naga::valid::Capabilities::SHADER_FLOAT16_IN_FLOAT32
                                    | match reduction {
                                        GemvReduction::Tree => naga::valid::Capabilities::empty(),
                                        GemvReduction::Subgroup => {
                                            naga::valid::Capabilities::SUBGROUP
                                        }
                                    };
                                let flags = naga::valid::ValidationFlags::all()
                                    ^ naga::valid::ValidationFlags::BINDINGS;
                                naga::valid::Validator::new(flags, caps)
                                    .validate(&module.module)
                                    .unwrap_or_else(|e| panic!("{group:?} {shape:?}: {e:#?}"));
                                assert!(module.source.contains(helper_name));
                                assert_eq!(
                                    module.source.contains(dot_call),
                                    packed_dot,
                                    "{group:?} {format:?} {shape:?}: wrong packed-dot implementation"
                                );
                                assert_eq!(
                                    module.source.contains("src[col4]"),
                                    group == ShaderGroup::MatMulGemvAdd,
                                    "{group:?} {shape:?}: wrong residual handling"
                                );
                                assert!(
                                    module
                                        .source
                                        .contains(&format!("const LANES: u32 = {threads}u;")),
                                    "{group:?} {format:?} {shape:?}: wrong width"
                                );
                                assert!(module.source.contains("blk += LANES;"));
                                assert_eq!(
                                    module.source.contains("subgroupAdd"),
                                    reduction == GemvReduction::Subgroup,
                                    "{group:?} {shape:?}: reduction mismatch"
                                );
                                assert_eq!(
                                    module.source.contains("params.eps_bits"),
                                    norm,
                                    "{group:?} {format:?}: wrong norm prologue"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn packed_gemv_rmsnorm_keeps_the_decoder() {
        for format in [
            WeightFormat::F32,
            WeightFormat::F16,
            WeightFormat::Q40,
            WeightFormat::Q4K,
            WeightFormat::Q8,
        ] {
            for group in [ShaderGroup::MatMulGemv, ShaderGroup::MatMulGemvBT] {
                if group == ShaderGroup::MatMulGemvBT && format.is_quantized() {
                    continue;
                }
                let sm = generate_module_gemv_rmsnorm(
                    group,
                    GemvShape {
                        threads: 64,
                        reduction: GemvReduction::Subgroup,
                        bt_rows: 4,
                        column_groups: 1,
                    },
                    format,
                );
                let flags =
                    naga::valid::ValidationFlags::all() ^ naga::valid::ValidationFlags::BINDINGS;
                naga::valid::Validator::new(flags, naga::valid::Capabilities::all())
                    .validate(&sm.module)
                    .unwrap();
                assert!(
                    sm.source.contains("inv_rms"),
                    "{format:?} fused GEMV lost the RmsNorm prologue"
                );
                assert!(
                    sm.source.contains("norm_w"),
                    "{format:?} fused GEMV lost the norm-weight binding"
                );
                match format {
                    WeightFormat::F32 => assert!(sm.source.contains("matrix_b: array<vec4<f32>>")),
                    WeightFormat::Q40 => assert!(sm.source.contains("dequant_q40(")),
                    WeightFormat::Q4K => assert!(sm.source.contains("dequant_q4k(")),
                    WeightFormat::Q8 => assert!(sm.source.contains("dequant_q8(")),
                    _ => {}
                }
            }
        }
    }

    #[test]
    fn subgroup_reduction_replaces_the_barrier_chain() {
        let barriers = |shape| {
            generate_module_gemv(ShaderGroup::MatMulGemv, WeightFormat::F32, shape)
                .source
                .matches("workgroupBarrier()")
                .count()
        };
        for threads in [32, 64, 128, 256] {
            let tree = barriers(GemvShape {
                threads,
                reduction: GemvReduction::Tree,
                bt_rows: 1,
                column_groups: 1,
            });
            let subgroup = barriers(GemvShape {
                threads,
                reduction: GemvReduction::Subgroup,
                bt_rows: 1,
                column_groups: 1,
            });
            // The only barrier gathers partials when there is more than
            // one subgroup; a single subgroup keeps its sum in registers.
            assert_eq!(
                subgroup, 1,
                "the subgroup reduction needs one conditional barrier at {threads} threads"
            );
            assert_eq!(
                tree,
                threads.ilog2() as usize,
                "the tree needs one barrier per halving level at {threads} threads"
            );
        }
    }

    #[test]
    fn reduced_storage_gemv_variants_keep_typed_b() {
        for (format, marker) in [
            (WeightFormat::F16, "array<vec4<f16>>"),
            (WeightFormat::Q4, "dequant_q4("),
            (WeightFormat::Q8, "dequant_q8("),
            (WeightFormat::Q40, "dequant_q40("),
            (WeightFormat::Q4K, "dequant_q4k("),
            (WeightFormat::Q6K, "dequant_q6k("),
            (WeightFormat::Q5K, "dequant_q5k("),
            (WeightFormat::Q3K, "dequant_q3k("),
        ] {
            for group in [ShaderGroup::MatMulGemv, ShaderGroup::MatMulGemvAdd] {
                let source = generate_module_weighted(group, format, MatmulKnobs::default()).source;
                assert!(
                    source.contains(marker) && !source.contains("matrix_b: array<vec4<f32>>"),
                    "{format:?} {group:?} did not retain its B representation"
                );
            }
        }
    }

    #[test]
    fn q4_matmul_with_sigmoid_epilogue_keeps_packed_b() {
        use crate::compile::MatMulEpilogue;
        use crate::schedule::{PointwiseDAG, Pw};

        let epi = MatMulEpilogue {
            dag: PointwiseDAG {
                n_inputs: 1,
                ops: vec![Pw::LoadInput(0), Pw::Sigmoid(0)],
                output: 1,
            },
            inputs: vec![],
        };
        let sm = generate_matmul_with_epilogue(
            ShaderGroup::MatMul,
            Some(&epi),
            MatMulOptions {
                format: WeightFormat::Q4,
                ..Default::default()
            },
        );
        assert!(
            sm.source.contains("dequant_q4"),
            "Q4+sigmoid must keep the packed B path"
        );
        assert!(
            sm.source.contains("array<u32>"),
            "Q4+sigmoid must declare matrix_b as array<u32>"
        );
        assert!(
            !sm.source.contains("matrix_b: array<f32>"),
            "Q4+sigmoid must not fall back to f32 B"
        );
        assert!(
            sm.source.contains("exp(-"),
            "Q4+sigmoid must emit a store-side sigmoid"
        );
    }

    #[test]
    #[should_panic(expected = "does not support block-quantized")]
    fn quantized_bt_epilogue_is_refused() {
        let _ = generate_matmul_with_epilogue(
            ShaderGroup::MatMulBTAdd,
            None,
            MatMulOptions {
                format: WeightFormat::Q4K,
                ..Default::default()
            },
        );
    }

    /// Q4_K is read out of a GGUF file unmodified, so the shader must
    /// declare B as words and decode superblocks rather than load floats.
    #[test]
    fn q6k_shaders_read_packed_superblocks() {
        for group in [
            ShaderGroup::MatMul,
            ShaderGroup::MatMulAdd,
            ShaderGroup::MatMulAT,
        ] {
            let sm = generate_module_weighted(group, WeightFormat::Q6K, MatmulKnobs::default());
            assert!(
                sm.source.contains("dequant_q6k"),
                "Q6_K {group:?}: missing the superblock decoder"
            );
            assert!(
                sm.source.contains("array<u32>") && !sm.source.contains("matrix_b: array<f32>"),
                "Q6_K {group:?}: B must be words, not floats"
            );
            // 210-byte superblocks, read byte-addressed.
            assert!(
                sm.source.contains("210u"),
                "Q6_K {group:?}: missing the 210-byte superblock stride"
            );
            assert!(
                sm.source.contains("select(0, 256, raw >= 128u)"),
                "Q6_K {group:?}: sub-block scales are signed and must sign-extend"
            );
        }
        let tiled = generate_module_weighted(
            ShaderGroup::MatMul,
            WeightFormat::Q6K,
            MatmulKnobs::default(),
        );
        assert!(
            tiled.source.contains("let unpacked = dequant_q6k_pack8("),
            "Q6_K tiled MatMul must call batched staging"
        );
    }

    #[test]
    fn q4k_shaders_read_packed_superblocks() {
        for group in [
            ShaderGroup::MatMul,
            ShaderGroup::MatMulAdd,
            ShaderGroup::MatMulAT,
        ] {
            let sm = generate_module_weighted(group, WeightFormat::Q4K, MatmulKnobs::default());
            assert!(
                sm.source.contains("dequant_q4k"),
                "Q4_K {group:?}: missing the superblock decoder"
            );
            assert!(
                sm.source.contains("array<u32>") && !sm.source.contains("matrix_b: array<f32>"),
                "Q4_K {group:?}: B must be words, not floats"
            );
            // 36 u32s per superblock is the layout the decoder assumes.
            assert!(
                sm.source.contains("36u"),
                "Q4_K {group:?}: missing the 144-byte superblock stride"
            );
        }
        // The large tile stages eight elements per thread through one scale
        // unpack; losing that silently falls back to eight scalar decodes.
        let tiled = generate_module_weighted(
            ShaderGroup::MatMul,
            WeightFormat::Q4K,
            MatmulKnobs::default(),
        );
        assert!(
            tiled.source.contains("let unpacked = dequant_q4k_pack8("),
            "Q4_K tiled MatMul must call batched staging"
        );
    }

    #[test]
    fn block_formats_refuse_groups_without_a_variant() {
        for format in [
            WeightFormat::Q4,
            WeightFormat::Q8,
            WeightFormat::Q40,
            WeightFormat::Q4K,
            WeightFormat::Q6K,
            WeightFormat::Q5K,
            WeightFormat::Q3K,
        ] {
            assert!(
                std::panic::catch_unwind(|| {
                    generate_module_weighted(
                        ShaderGroup::MatMulGemvBT,
                        format,
                        MatmulKnobs::default(),
                    )
                })
                .is_err(),
                "{format:?} GEMV-BT fell through to an f32 shader"
            );
        }
    }

    /// Packing runs along the parameter's first dimension; every packed
    /// decoder indexes along K. Those are the same axis for a forward
    /// `[K, N]` weight and different axes for a transposed `[N, K]` one, so
    /// no block format has a correct reading on the BT groups — Q4 and Q8
    /// included, whose arms here were dead and wrong rather than unused.
    /// Q5_K and Q3_K read GGUF bytes directly, so the shader must declare
    /// B as words and carry each format's own superblock stride.
    ///
    /// The batched-staging check looks for the *call*, not the helper name:
    /// the decoder block always declares `fn dequant_*_pack8`, so a
    /// `contains` on the name alone stays true even when the staging
    /// selector has dropped the format back to the scalar path.
    #[test]
    fn new_k_quant_shaders_read_packed_superblocks() {
        for (mode, decoder, stride) in [
            (WeightFormat::Q5K, "dequant_q5k", "44u"),
            (WeightFormat::Q3K, "dequant_q3k", "110u"),
        ] {
            for group in [
                ShaderGroup::MatMul,
                ShaderGroup::MatMulAdd,
                ShaderGroup::MatMulAT,
            ] {
                let sm = generate_module_weighted(group, mode, MatmulKnobs::default());
                assert!(
                    sm.source.contains(decoder),
                    "{mode:?} {group:?}: missing the superblock decoder"
                );
                assert!(
                    sm.source.contains("array<u32>") && !sm.source.contains("matrix_b: array<f32>"),
                    "{mode:?} {group:?}: B must be words, not floats"
                );
                assert!(
                    sm.source.contains(stride),
                    "{mode:?} {group:?}: missing the superblock stride"
                );
            }
            let tiled = generate_module_weighted(ShaderGroup::MatMul, mode, MatmulKnobs::default());
            assert!(
                tiled
                    .source
                    .contains(&format!("let unpacked = {decoder}_pack8(")),
                "{mode:?} tiled MatMul must call batched staging"
            );
        }
        // Q3_K's high bit is inverted: a clear hmask bit subtracts 4.
        let q3 = generate_module_weighted(
            ShaderGroup::MatMul,
            WeightFormat::Q3K,
            MatmulKnobs::default(),
        );
        assert!(
            q3.source.contains("select(4, 0, hbit == 1u)"),
            "Q3_K must subtract 4 when the hmask bit is clear"
        );
    }

    #[test]
    fn quantized_formats_refuse_transposed_b() {
        for group in [ShaderGroup::MatMulBT, ShaderGroup::MatMulBTAdd] {
            for mode in [
                WeightFormat::Q4,
                WeightFormat::Q8,
                WeightFormat::Q4K,
                WeightFormat::Q6K,
                WeightFormat::Q5K,
                WeightFormat::Q3K,
            ] {
                let caught = std::panic::catch_unwind(|| {
                    let _ = generate_module_weighted(group, mode, MatmulKnobs::default());
                });
                assert!(
                    caught.is_err(),
                    "{mode:?} on {group:?} must be refused, not decoded along the wrong axis"
                );
            }
        }
        // f16 is an elementwise cast at the same index, so it still works.
        assert!(
            generate_module_weighted(
                ShaderGroup::MatMulBT,
                WeightFormat::F16,
                MatmulKnobs::default()
            )
            .source
            .contains("array<f16>")
        );
        // The forward groups are unaffected.
        assert!(
            generate_module_weighted(
                ShaderGroup::MatMul,
                WeightFormat::Q4K,
                MatmulKnobs::default()
            )
            .source
            .contains("dequant_q4k")
        );
    }

    /// The epilogue skeleton has to be specialized for the tile the
    /// dispatch was sized for, staging maps included: a 64-wide map over
    /// a 32-wide tile reads the wrong elements into shared memory.
    #[test]
    fn small_tile_epilogue_uses_small_tile_geometry() {
        for group in [
            ShaderGroup::MatMul,
            ShaderGroup::MatMulAdd,
            ShaderGroup::MatMulAT,
            ShaderGroup::MatMulBT,
        ] {
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
            let small = generate_matmul_with_epilogue(
                group,
                Some(&relu),
                MatMulOptions {
                    tile: MatMulTile::Small,
                    ..Default::default()
                },
            );
            let large = generate_matmul_with_epilogue(group, Some(&relu), MatMulOptions::default());
            assert_ne!(
                small.source, large.source,
                "{group:?}: small and large epilogue shaders must differ"
            );
            let k_stage = crate::codegen::MatMulOptions::default().knobs.k_stage;
            assert!(
                small
                    .source
                    .contains(&format!("array<f32, {}>", 32 * (k_stage + 1)))
                    && small
                        .source
                        .contains(&format!("array<f32, {}>", k_stage * 33)),
                "{group:?}: small epilogue must stage 32-wide tiles at the requested K depth"
            );
            assert!(
                !small.source.contains("flat / 64u") && !small.source.contains("flat % 64u"),
                "{group:?}: small epilogue must not use 64-wide staging maps"
            );
        }
    }
}
