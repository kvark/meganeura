//! Shader codegen via WGSL templates.
//!
//! Shaders are written as `.wgsl` files in `src/shaders/` and parsed at
//! runtime by the naga WGSL frontend. The `preprocess()` helper performs
//! `$VAR` substitution for parameterized shaders before parsing.
//!
//! Modules are passed directly to blade via `naga_module` for SPIR-V
//! compilation.

use naga::Module;

/// Configuration for cooperative matrix tile size and precision.
///
/// Derived from `blade_graphics::CooperativeMatrix` capabilities at runtime.
/// Determines which shader variant is generated for coop matmul.
#[derive(Clone, Copy, Debug)]
pub struct CoopConfig {
    /// Cooperative matrix tile dimension (8 for Apple Silicon, 16 for RDNA3/Volta+).
    pub tile_size: u32,
    /// Use f16 input with f32 accumulator (true for Vulkan), or all-f32 (true for Metal).
    pub use_f16_input: bool,
}

impl CoopConfig {
    /// Output tile per workgroup = 2 × tile_size (2×2 grid of coop tiles).
    pub fn output_tile(&self) -> u32 {
        2 * self.tile_size
    }
}

/// Replace `$VAR` occurrences in `source` with the corresponding values.
fn preprocess(source: &str, vars: &[(&str, &str)]) -> String {
    let mut s = source.to_string();
    for &(key, val) in vars {
        s = s.replace(key, val);
    }
    s
}

/// A parsed shader module together with the WGSL source text.
///
/// Blade needs the source for SPIR-V debug info (OpLine) in debug builds.
pub struct ShaderModule {
    pub module: Module,
    pub source: String,
}

/// Parse a WGSL source string into a [`ShaderModule`].
fn parse_wgsl(source: &str) -> ShaderModule {
    let module = naga::front::wgsl::parse_str(source).expect("WGSL parse failed");
    ShaderModule {
        module,
        source: source.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Shader groups — each group is a naga::Module with one or more entry points
// ---------------------------------------------------------------------------

/// A shader group corresponds to a single `naga::Module` that may
/// contain multiple entry points (e.g. `Unary` has relu, sigmoid, neg).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ShaderGroup {
    Unary,
    Binary,
    BiasAdd,
    Sgd,
    Adam,
    Transpose,
    MatMul,
    MatMulAdd,
    MatMulAT,
    MatMulBT,
    MatMulATAdd,
    MatMulBTAdd,
    MatMulSmall,
    MatMulSmallAdd,
    MatMulSmallAT,
    MatMulSmallBT,
    MatMulCoop,
    MatMulCoopAdd,
    MatMulCoopAT,
    MatMulCoopBT,
    Reduce,
    Softmax,
    CrossEntropy,
    RmsNorm,
    Embedding,
    RoPE,
    CausalAttention,
    LayerNorm,
    FullAttention,
    CrossAttention,
    MultiHeadAttn,
    MultiHeadAttnGradQ,
    MultiHeadAttnGradK,
    MultiHeadAttnGradV,
    SwiGLUGrad,
    SwiGLUConcat,
    SumRows,
    RmsNormGrad,
    ScatterAdd,
    BceLoss,
    FusedRmsNormMatMul,
    GroupNorm,
    GroupNormGrad,
    Concat,
    Split,
    Upsample,
    UpsampleGrad,
    Conv2d,
    Conv2dGemm,
    Conv2dGemmSmall,
    Conv2dGradInput,
    Conv2dGradInputGemm,
    Conv2dGradInputGemmSmall,
    Conv2dGradInputGemmCoop,
    Conv2dGradWeight,
    Conv2dGradWeightGemm,
    CacheWrite,
    CachedAttention,
    RoPEDynamic,
}

/// Generate a `naga::Module` for a shader group.
pub fn generate_module(group: ShaderGroup) -> ShaderModule {
    match group {
        ShaderGroup::Unary => gen_unary(),
        ShaderGroup::Binary => gen_binary(),
        ShaderGroup::BiasAdd => gen_bias_add(),
        ShaderGroup::Sgd => gen_sgd(),
        ShaderGroup::Adam => gen_adam(),
        ShaderGroup::Transpose => gen_transpose(),
        ShaderGroup::MatMul => gen_matmul(),
        ShaderGroup::MatMulAdd => gen_matmul_add(),
        ShaderGroup::MatMulAT => gen_matmul_at(),
        ShaderGroup::MatMulBT => gen_matmul_bt(),
        ShaderGroup::MatMulATAdd => gen_matmul_at_add(),
        ShaderGroup::MatMulBTAdd => gen_matmul_bt_add(),
        ShaderGroup::MatMulSmall => gen_matmul_small(),
        ShaderGroup::MatMulSmallAdd => gen_matmul_small_add(),
        ShaderGroup::MatMulSmallAT => gen_matmul_small_at(),
        ShaderGroup::MatMulSmallBT => gen_matmul_small_bt(),
        ShaderGroup::MatMulCoop => gen_matmul_coop(),
        ShaderGroup::MatMulCoopAdd => gen_matmul_coop_add(),
        ShaderGroup::MatMulCoopAT => gen_matmul_coop_at(),
        ShaderGroup::MatMulCoopBT => gen_matmul_coop_bt(),
        ShaderGroup::Reduce => gen_reduce(),
        ShaderGroup::Softmax => gen_softmax(),
        ShaderGroup::CrossEntropy => gen_cross_entropy(),
        ShaderGroup::RmsNorm => gen_rms_norm(),
        ShaderGroup::Embedding => gen_embedding(),
        ShaderGroup::RoPE => gen_rope(),
        ShaderGroup::CausalAttention => gen_causal_attention(),
        ShaderGroup::LayerNorm => gen_layer_norm(),
        ShaderGroup::FullAttention => gen_full_attention(),
        ShaderGroup::CrossAttention => gen_cross_attention(),
        ShaderGroup::MultiHeadAttn => gen_mha_forward(),
        ShaderGroup::MultiHeadAttnGradQ => gen_mha_grad_q(),
        ShaderGroup::MultiHeadAttnGradK => gen_mha_grad_k(),
        ShaderGroup::MultiHeadAttnGradV => gen_mha_grad_v(),
        ShaderGroup::SwiGLUGrad => gen_swiglu_grad(),
        ShaderGroup::SwiGLUConcat => gen_swiglu_concat(),
        ShaderGroup::SumRows => gen_sum_rows(),
        ShaderGroup::RmsNormGrad => gen_rms_norm_grad(),
        ShaderGroup::FusedRmsNormMatMul => gen_fused_rms_norm_matmul(),
        ShaderGroup::ScatterAdd => gen_scatter_add(),
        ShaderGroup::BceLoss => gen_bce_loss(),
        ShaderGroup::GroupNorm => gen_group_norm(),
        ShaderGroup::GroupNormGrad => gen_group_norm_grad(),
        ShaderGroup::Concat => gen_concat(),
        ShaderGroup::Split => gen_split(),
        ShaderGroup::Upsample => gen_upsample(),
        ShaderGroup::UpsampleGrad => gen_upsample_grad(),
        ShaderGroup::Conv2d => gen_conv2d(),
        ShaderGroup::Conv2dGemm => gen_conv2d_gemm(),
        ShaderGroup::Conv2dGemmSmall => gen_conv2d_gemm_small(),
        ShaderGroup::Conv2dGradInput => gen_conv2d_grad_input(),
        ShaderGroup::Conv2dGradInputGemm => gen_conv2d_grad_input_gemm(),
        ShaderGroup::Conv2dGradInputGemmSmall => gen_conv2d_grad_input_gemm_small(),
        ShaderGroup::Conv2dGradInputGemmCoop => gen_conv2d_grad_input_gemm_coop(),
        ShaderGroup::Conv2dGradWeight => gen_conv2d_grad_weight(),
        ShaderGroup::Conv2dGradWeightGemm => gen_conv2d_grad_weight_gemm(),
        ShaderGroup::CacheWrite => gen_cache_write(),
        ShaderGroup::CachedAttention => gen_cached_attention(),
        ShaderGroup::RoPEDynamic => gen_rope_dynamic(),
    }
}

/// Generate a cooperative matrix shader module with the given tile config.
pub fn generate_coop_module(group: ShaderGroup, config: &CoopConfig) -> ShaderModule {
    match group {
        ShaderGroup::MatMulCoop => gen_matmul_coop_wgsl(false, MatMulCoopVariant::Normal, config),
        ShaderGroup::MatMulCoopAdd => gen_matmul_coop_wgsl(true, MatMulCoopVariant::Normal, config),
        ShaderGroup::MatMulCoopBT => gen_matmul_coop_wgsl(false, MatMulCoopVariant::BT, config),
        ShaderGroup::MatMulCoopAT => gen_matmul_coop_wgsl(false, MatMulCoopVariant::AT, config),
        ShaderGroup::Conv2dGradInputGemmCoop => gen_conv2d_grad_input_gemm_coop_wgsl(config),
        _ => panic!("not a coop shader group: {:?}", group),
    }
}

/// Generate WGSL source for a shader group.
pub fn generate_wgsl(group: ShaderGroup) -> String {
    let sm = generate_module(group);
    let capabilities = match group {
        ShaderGroup::MatMulCoop
        | ShaderGroup::MatMulCoopAdd
        | ShaderGroup::MatMulCoopAT
        | ShaderGroup::Conv2dGradInputGemmCoop
        | ShaderGroup::MatMulCoopBT => {
            naga::valid::Capabilities::COOPERATIVE_MATRIX
                | naga::valid::Capabilities::SHADER_FLOAT16
        }
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
// unary.wgsl: relu, sigmoid, neg
// ---------------------------------------------------------------------------

fn gen_unary() -> ShaderModule {
    parse_wgsl(include_str!("shaders/unary.wgsl"))
}

// ---------------------------------------------------------------------------
// binary.wgsl: add, mul, greater
// ---------------------------------------------------------------------------

fn gen_binary() -> ShaderModule {
    parse_wgsl(include_str!("shaders/binary.wgsl"))
}

// ---------------------------------------------------------------------------
// swiglu_grad: fused backward kernels for SwiGLU and Silu
//   swiglu_grad_gate: (src_a=grad_out, src_b=gate, src_c=up) → dst
//   swiglu_grad_up:   (src_a=grad_out, src_b=gate)            → dst
//   silu_grad:        (src_a=grad_out, src_b=x)               → dst
// ---------------------------------------------------------------------------

fn gen_swiglu_grad() -> ShaderModule {
    parse_wgsl(include_str!("shaders/swiglu_grad.wgsl"))
}

fn gen_swiglu_concat() -> ShaderModule {
    parse_wgsl(include_str!("shaders/swiglu_concat.wgsl"))
}

// ---------------------------------------------------------------------------
// bias_add.wgsl
// ---------------------------------------------------------------------------

fn gen_bias_add() -> ShaderModule {
    parse_wgsl(include_str!("shaders/bias_add.wgsl"))
}

// ---------------------------------------------------------------------------
// sgd.wgsl
// ---------------------------------------------------------------------------

fn gen_sgd() -> ShaderModule {
    parse_wgsl(include_str!("shaders/sgd.wgsl"))
}

// ---------------------------------------------------------------------------
// adam.wgsl
// ---------------------------------------------------------------------------

fn gen_adam() -> ShaderModule {
    parse_wgsl(include_str!("shaders/adam.wgsl"))
}

// ---------------------------------------------------------------------------
// scatter_add.wgsl — embedding gradient accumulation
// ---------------------------------------------------------------------------

fn gen_scatter_add() -> ShaderModule {
    parse_wgsl(include_str!("shaders/scatter_add.wgsl"))
}

// ---------------------------------------------------------------------------
// bce.wgsl — binary cross-entropy loss with gradient
// ---------------------------------------------------------------------------

fn gen_bce_loss() -> ShaderModule {
    parse_wgsl(include_str!("shaders/bce.wgsl"))
}

// ---------------------------------------------------------------------------
// transpose.wgsl
// ---------------------------------------------------------------------------

fn gen_transpose() -> ShaderModule {
    parse_wgsl(include_str!("shaders/transpose.wgsl"))
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
/// For row-major A[M,K]: K is the fast dimension → col = flat%16 (fast)
/// For transposed A[K,M]: M is the fast dimension → row = flat%64 (fast)
/// For row-major B[K,N]: N is the fast dimension → col = flat%64 (fast)
/// For transposed B[N,K]: K is the fast dimension → row = flat%16 (fast)
const A_ROW_FWD: &str = "flat / 16u"; // M varies slowly (good for [M,K])
const A_COL_FWD: &str = "flat % 16u"; // K varies fast (coalesced in [M,K])
const A_ROW_AT: &str = "flat % 64u"; // M varies fast (coalesced in [K,M])
const A_COL_AT: &str = "flat / 64u"; // K varies slowly
const B_ROW_FWD: &str = "flat / 64u"; // K varies slowly (good for [K,N])
const B_COL_FWD: &str = "flat % 64u"; // N varies fast (coalesced in [K,N])
const B_ROW_BT: &str = "flat % 16u"; // K varies fast (coalesced in [N,K])
const B_COL_BT: &str = "flat / 16u"; // N varies slowly

// Small-tile (32×32) load mappings — BM=32, BN=32, KTILE=16
// A tile: [32, 16] = 512 elements, 2 per thread
// B tile: [16, 32] = 512 elements, 2 per thread
const A_ROW_FWD_S: &str = "flat / 16u"; // same as large (M slow, K fast)
const A_COL_FWD_S: &str = "flat % 16u";
const A_ROW_AT_S: &str = "flat % 32u"; // M fast (32 not 64)
const A_COL_AT_S: &str = "flat / 32u";
const B_ROW_FWD_S: &str = "flat / 32u"; // K slow (32 not 64)
const B_COL_FWD_S: &str = "flat % 32u"; // N fast (32 not 64)
const B_ROW_BT_S: &str = "flat % 16u"; // same as large
const B_COL_BT_S: &str = "flat / 16u";

fn matmul_vars(
    a_idx: &str,
    b_idx: &str,
    a_row: &str,
    a_col: &str,
    b_row: &str,
    b_col: &str,
    fused_decl: &str,
    fused_expr: &str,
) -> ShaderModule {
    let src = include_str!("shaders/matmul.wgsl");
    let src = preprocess(
        src,
        &[
            ("$A_INDEX", a_idx),
            ("$B_INDEX", b_idx),
            ("$A_ROW", a_row),
            ("$A_COL", a_col),
            ("$B_ROW", b_row),
            ("$B_COL", b_col),
            ("$FUSED_ADD_DECL", fused_decl),
            ("$FUSED_ADD_EXPR", fused_expr),
        ],
    );
    parse_wgsl(&src)
}

fn matmul_small_vars(
    a_idx: &str,
    b_idx: &str,
    a_row: &str,
    a_col: &str,
    b_row: &str,
    b_col: &str,
    fused_decl: &str,
    fused_expr: &str,
) -> ShaderModule {
    let src = include_str!("shaders/matmul_small.wgsl");
    let src = preprocess(
        src,
        &[
            ("$A_INDEX", a_idx),
            ("$B_INDEX", b_idx),
            ("$A_ROW_S", a_row),
            ("$A_COL_S", a_col),
            ("$B_ROW_S", b_row),
            ("$B_COL_S", b_col),
            ("$FUSED_ADD_DECL", fused_decl),
            ("$FUSED_ADD_EXPR", fused_expr),
        ],
    );
    parse_wgsl(&src)
}

fn gen_matmul_small() -> ShaderModule {
    matmul_small_vars(
        MATMUL_A_FWD,
        MATMUL_B_FWD,
        A_ROW_FWD_S,
        A_COL_FWD_S,
        B_ROW_FWD_S,
        B_COL_FWD_S,
        "",
        "",
    )
}
fn gen_matmul_small_add() -> ShaderModule {
    matmul_small_vars(
        MATMUL_A_FWD,
        MATMUL_B_FWD,
        A_ROW_FWD_S,
        A_COL_FWD_S,
        B_ROW_FWD_S,
        B_COL_FWD_S,
        "var<storage> src: array<f32>;",
        " + src[idx]",
    )
}
fn gen_matmul_small_at() -> ShaderModule {
    matmul_small_vars(
        MATMUL_A_AT,
        MATMUL_B_FWD,
        A_ROW_AT_S,
        A_COL_AT_S,
        B_ROW_FWD_S,
        B_COL_FWD_S,
        "",
        "",
    )
}
fn gen_matmul_small_bt() -> ShaderModule {
    matmul_small_vars(
        MATMUL_A_FWD,
        MATMUL_B_BT,
        A_ROW_FWD_S,
        A_COL_FWD_S,
        B_ROW_BT_S,
        B_COL_BT_S,
        "",
        "",
    )
}

fn gen_matmul() -> ShaderModule {
    matmul_vars(
        MATMUL_A_FWD,
        MATMUL_B_FWD,
        A_ROW_FWD,
        A_COL_FWD,
        B_ROW_FWD,
        B_COL_FWD,
        "",
        "",
    )
}

fn gen_matmul_add() -> ShaderModule {
    matmul_vars(
        MATMUL_A_FWD,
        MATMUL_B_FWD,
        A_ROW_FWD,
        A_COL_FWD,
        B_ROW_FWD,
        B_COL_FWD,
        "var<storage> src: array<f32>;",
        " + src[idx]",
    )
}

/// FusedMatMulATAdd: C = A^T × B + D  (A=[K,M], B=[K,N], D=[M,N], C=[M,N])
fn gen_matmul_at_add() -> ShaderModule {
    matmul_vars(
        MATMUL_A_AT,
        MATMUL_B_FWD,
        A_ROW_AT,
        A_COL_AT,
        B_ROW_FWD,
        B_COL_FWD,
        "var<storage> src: array<f32>;",
        " + src[idx]",
    )
}

/// FusedMatMulBTAdd: C = A × B^T + D  (A=[M,K], B=[N,K], D=[M,N], C=[M,N])
fn gen_matmul_bt_add() -> ShaderModule {
    matmul_vars(
        MATMUL_A_FWD,
        MATMUL_B_BT,
        A_ROW_FWD,
        A_COL_FWD,
        B_ROW_BT,
        B_COL_BT,
        "var<storage> src: array<f32>;",
        " + src[idx]",
    )
}

/// MatMulBT: C = A @ B^T  (A=[M,K], B=[N,K], C=[M,N])
///
/// Coalesced B load: consecutive threads read adjacent K values from B[N,K]
/// (K is the row-major fast dimension), then store transposed into shared_b.
fn gen_matmul_bt() -> ShaderModule {
    matmul_vars(
        MATMUL_A_FWD,
        MATMUL_B_BT,
        A_ROW_FWD,
        A_COL_FWD,
        B_ROW_BT,
        B_COL_BT,
        "",
        "",
    )
}

/// MatMulAT: C = A^T @ B  (A=[K,M], B=[K,N], C=[M,N])
///
/// Coalesced A load: consecutive threads read adjacent M values from A[K,M]
/// (M is the row-major fast dimension), then store transposed into shared_a.
fn gen_matmul_at() -> ShaderModule {
    matmul_vars(
        MATMUL_A_AT,
        MATMUL_B_FWD,
        A_ROW_AT,
        A_COL_AT,
        B_ROW_FWD,
        B_COL_FWD,
        "",
        "",
    )
}

// ---------------------------------------------------------------------------
// matmul_coop.wgsl — cooperative matrix multiply (16×16 tiles)
//
// Uses cooperative matrix operations for hardware-accelerated matrix multiply
// on supported GPUs (VK_KHR_cooperative_matrix on Vulkan, simdgroup_matrix
// on Metal).
//
// Workgroup [8, 8, 1], dispatched as [ceil(M/8), ceil(N/8), 1].
// Each workgroup computes one 8×8 output tile, iterating over K in
// steps of 8.
// ---------------------------------------------------------------------------

/// Cooperative matrix matmul: C = A × B.
///
/// Parameterized by `CoopConfig` to support different tile sizes and precisions:
/// - 16×16 f16 tiles (RDNA3/Volta+): mixed-precision f16×f16+f32
/// -  8×8  f32 tiles (Apple Silicon): all-f32 via simdgroup_matrix
///
/// The default `generate_module` path uses 16×16 f16 for backward compat.
/// Use `generate_coop_module` with a `CoopConfig` for runtime-detected config.
fn gen_matmul_coop() -> ShaderModule {
    let default_config = CoopConfig {
        tile_size: 16,
        use_f16_input: true,
    };
    gen_matmul_coop_wgsl(false, MatMulCoopVariant::Normal, &default_config)
}

fn gen_matmul_coop_add() -> ShaderModule {
    let default_config = CoopConfig {
        tile_size: 16,
        use_f16_input: true,
    };
    gen_matmul_coop_wgsl(true, MatMulCoopVariant::Normal, &default_config)
}

fn gen_matmul_coop_bt() -> ShaderModule {
    let default_config = CoopConfig {
        tile_size: 16,
        use_f16_input: true,
    };
    gen_matmul_coop_wgsl(false, MatMulCoopVariant::BT, &default_config)
}

fn gen_matmul_coop_at() -> ShaderModule {
    let default_config = CoopConfig {
        tile_size: 16,
        use_f16_input: true,
    };
    gen_matmul_coop_wgsl(false, MatMulCoopVariant::AT, &default_config)
}

fn gen_matmul_coop_wgsl(
    fused_add: bool,
    variant: MatMulCoopVariant,
    config: &CoopConfig,
) -> ShaderModule {
    let tile = config.tile_size;
    let output_tile = config.output_tile();
    let shared_size = tile * tile;
    let wg_size: u32 = 64;
    let staging_iters = shared_size / wg_size;
    let row_stride = wg_size / tile;
    let tile_mask = tile - 1;
    let tile_shift = tile.trailing_zeros();

    let (elem_type, enable_f16, elem_zero, cast_open, cast_close) = if config.use_f16_input {
        ("f16", "enable f16;", "f16(0.0)", "f16(", ")")
    } else {
        ("f32", "", "0.0", "", "")
    };
    let ab_type = if config.use_f16_input { "f16" } else { "f32" };
    let coop_ab = format!("coop_mat{}x{}<{},A>", tile, tile, ab_type);
    let coop_ba = format!("coop_mat{}x{}<{},B>", tile, tile, ab_type);
    let coop_c = format!("coop_mat{}x{}<f32,C>", tile, tile);

    let (b_idx_0, b_idx_1) = match variant {
        MatMulCoopVariant::Normal | MatMulCoopVariant::AT => ("tr * n + cc", "tr * n + cc1"),
        MatMulCoopVariant::BT => ("cc * k + tr", "cc1 * k + tr"),
    };
    let (a_idx_0, a_idx_1) = match variant {
        MatMulCoopVariant::Normal | MatMulCoopVariant::BT => ("gr * k + tc", "gr * k + tc"),
        MatMulCoopVariant::AT => ("tc * m + gr", "tc * m + gr"),
    };
    let (fused_decl, acc_init) = if fused_add {
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

    let output_tile_u = format!("{}u", output_tile);
    let tile_size_u = format!("{}u", tile);
    let tile_mask_u = format!("{}u", tile_mask);
    let tile_shift_u = format!("{}u", tile_shift);
    let staging_iters_u = format!("{}u", staging_iters);
    let row_stride_u = format!("{}u", row_stride);
    let shared_size_s = format!("{}", shared_size);

    let src = include_str!("shaders/matmul_coop.wgsl");
    let src = preprocess(
        src,
        &[
            ("$ENABLE_F16", enable_f16),
            ("$ELEM_TYPE", elem_type),
            ("$ELEM_ZERO", elem_zero),
            ("$SHARED_SIZE", &shared_size_s),
            ("$OUTPUT_TILE_U", &output_tile_u),
            ("$TILE_SIZE_U", &tile_size_u),
            ("$TILE_MASK_U", &tile_mask_u),
            ("$TILE_SHIFT_U", &tile_shift_u),
            ("$STAGING_ITERS_U", &staging_iters_u),
            ("$ROW_STRIDE_U", &row_stride_u),
            ("$CAST_OPEN", cast_open),
            ("$CAST_CLOSE", cast_close),
            ("$COOP_AB", &coop_ab),
            ("$COOP_BA", &coop_ba),
            ("$B_INDEX_0", b_idx_0),
            ("$B_INDEX_1", b_idx_1),
            ("$A_INDEX_0", a_idx_0),
            ("$A_INDEX_1", a_idx_1),
            ("$FUSED_ADD_DECL", &fused_decl),
            ("$ACC_INIT", &acc_init),
        ],
    );
    parse_wgsl(&src)
}

/// Variant selector for gen_matmul_coop_inner.
#[derive(Clone, Copy, PartialEq)]
enum MatMulCoopVariant {
    /// C = A @ B  (standard)
    Normal,
    /// C = A @ B^T  (B is [N,K], accessed transposed)
    BT,
    /// C = A^T @ B  (A is [K,M], accessed transposed)
    AT,
}

// ---------------------------------------------------------------------------
// reduce.wgsl: sum_all, mean_all
// ---------------------------------------------------------------------------

fn gen_reduce() -> ShaderModule {
    parse_wgsl(include_str!("shaders/reduce.wgsl"))
}

// ---------------------------------------------------------------------------
// sum_rows.wgsl: [M, N] → [N], column-wise sum
// ---------------------------------------------------------------------------

fn gen_sum_rows() -> ShaderModule {
    parse_wgsl(include_str!("shaders/sum_rows.wgsl"))
}

// ---------------------------------------------------------------------------
// softmax.wgsl
// ---------------------------------------------------------------------------

fn gen_softmax() -> ShaderModule {
    parse_wgsl(include_str!("shaders/softmax.wgsl"))
}

// ---------------------------------------------------------------------------
// cross_entropy.wgsl
// ---------------------------------------------------------------------------

fn gen_cross_entropy() -> ShaderModule {
    parse_wgsl(include_str!("shaders/cross_entropy.wgsl"))
}

// ---------------------------------------------------------------------------
// rms_norm.wgsl: y[i,j] = x[i,j] / sqrt(mean(x[i,:]²) + eps) * weight[j]
// ---------------------------------------------------------------------------

/// Parallel RMSNorm: one workgroup (256 threads) per row.
///
/// Each thread handles cols/256 elements, then tree-reduces the partial
/// sums-of-squares in shared memory. Much faster than the serial version
/// for wide rows (e.g. 720 columns in SmolVLA).
///
/// Dispatch: [rows, 1, 1].
fn gen_rms_norm() -> ShaderModule {
    parse_wgsl(include_str!("shaders/rms_norm.wgsl"))
}

// ---------------------------------------------------------------------------
// rms_norm_grad: exact backward for RmsNorm
// Two entry points: rms_norm_grad_w (dispatch [ceil(cols/256), 1, 1])
//                   rms_norm_grad_x (dispatch [rows, 1, 1])
// Bindings: src_a (dy, ro), src_b (x, ro), bias (w, ro), dst (rw), params (uniform)
// Params: rows (field 0), cols (field 1), eps_bits (field 2), _pad (field 3)
// ---------------------------------------------------------------------------

fn gen_rms_norm_grad() -> ShaderModule {
    parse_wgsl(include_str!("shaders/rms_norm_grad.wgsl"))
}

// ---------------------------------------------------------------------------
// matmul_rms_norm.wgsl: Fused RmsNorm + MatMul
// C = RmsNorm(X, W_norm) × W_proj
// ---------------------------------------------------------------------------

fn gen_fused_rms_norm_matmul() -> ShaderModule {
    parse_wgsl(include_str!("shaders/matmul_rms_norm.wgsl"))
}

// ---------------------------------------------------------------------------
// embedding.wgsl: dst[i*hidden+j] = table[indices[i]*hidden+j]
// ---------------------------------------------------------------------------

fn gen_embedding() -> ShaderModule {
    parse_wgsl(include_str!("shaders/embedding.wgsl"))
}

// ---------------------------------------------------------------------------
// rope.wgsl: Rotary position embeddings
// For each position pos and pair (2i, 2i+1):
//   cos_t = cos(pos * theta^(-2i/dim))
//   sin_t = sin(pos * theta^(-2i/dim))
//   out[2i]   = x[2i]*cos_t - x[2i+1]*sin_t
//   out[2i+1] = x[2i]*sin_t + x[2i+1]*cos_t
// ---------------------------------------------------------------------------

fn gen_rope() -> ShaderModule {
    parse_wgsl(include_str!("shaders/rope.wgsl"))
}

// ---------------------------------------------------------------------------
// causal_attention.wgsl: Fused multi-head causal attention with GQA
// One workgroup per (position, head) pair.
// params: seq, num_heads, num_kv_heads, head_dim
// inputs: q[seq, num_heads*head_dim], k[seq, num_kv_heads*head_dim], v[seq, ...]
// output: [seq, num_heads*head_dim]
// ---------------------------------------------------------------------------

/// Single-pass causal attention with online softmax.
///
/// Computes multi-head causal attention in one pass over key positions,
/// maintaining a running output accumulator. Compared to the 3-pass
/// approach, this reduces compute from O(D²·N) to O(D·N) per (pos, head).
///
/// Algorithm per (pos, head):
///   max_score = -inf, sum_exp = 0, out[d] = 0
///   for t in 0..pos+1:
///     score = Q[pos]·K[t] * scale
///     new_max = max(max_score, score)
///     correction = exp(max_score - new_max)
///     sum_exp = sum_exp * correction + exp(score - new_max)
///     for d: out[d] = out[d] * correction + exp(score - new_max) * V[t,d]
///     max_score = new_max
///   for d: dst[d] = out[d] / sum_exp
fn gen_causal_attention() -> ShaderModule {
    let src = include_str!("shaders/attention.wgsl");
    let src = preprocess(
        src,
        &[
            (
                "$PARAM_FIELDS",
                "seq: u32, num_heads: u32, num_kv_heads: u32, head_dim: u32,",
            ),
            (
                "$PARSE_PARAMS",
                "let q_seq = params.seq;\n    let num_heads = params.num_heads;\n    let num_kv_heads = params.num_kv_heads;\n    let head_dim = params.head_dim;\n    let kv_len = pos + 1u;",
            ),
        ],
    );
    parse_wgsl(&src)
}

#[allow(clippy::empty_line_after_doc_comments)]
/// Parallel attention kernel: 64 threads per workgroup, one per head_dim element.
/// Dispatch [q_seq, num_heads, 1] workgroups; workgroup_id gives (pos, head), local_id.x is tid.
///
/// Algorithm: single-pass online softmax across KV positions.
/// - All 64 threads compute partial dot Q[tid]*K[t,tid] → store to wg_dot[tid]
/// - 6-stage parallel reduction gives scalar score = sum_d Q[d]*K[t,d]
/// - Online softmax update (same scalar ops for all threads)
/// - Each thread accumulates its own V dimension: out[tid] += weight * V[t,tid]
/// - Output: dst[q_base + tid] = out[tid] / sum_exp

// ---------------------------------------------------------------------------
// layer_norm.wgsl: y[i,j] = (x[i,j] - mean) / sqrt(var + eps) * weight[j] + bias[j]
// ---------------------------------------------------------------------------

fn gen_layer_norm() -> ShaderModule {
    parse_wgsl(include_str!("shaders/layer_norm.wgsl"))
}

// ---------------------------------------------------------------------------
// full_attention.wgsl: non-causal multi-head attention with GQA
// Same as causal_attention but attends to all positions (no mask).
// ---------------------------------------------------------------------------

fn gen_full_attention() -> ShaderModule {
    let src = include_str!("shaders/attention.wgsl");
    let src = preprocess(
        src,
        &[
            (
                "$PARAM_FIELDS",
                "seq: u32, num_heads: u32, num_kv_heads: u32, head_dim: u32,",
            ),
            (
                "$PARSE_PARAMS",
                "let q_seq = params.seq;\n    let num_heads = params.num_heads;\n    let num_kv_heads = params.num_kv_heads;\n    let head_dim = params.head_dim;\n    let kv_len = q_seq;",
            ),
        ],
    );
    parse_wgsl(&src)
}

// ---------------------------------------------------------------------------
// cross_attention.wgsl: cross-attention where q and k/v have different seq lengths
// params: q_seq, kv_seq, (num_heads<<16)|num_kv_heads, head_dim
// ---------------------------------------------------------------------------

fn gen_cross_attention() -> ShaderModule {
    let src = include_str!("shaders/attention.wgsl");
    let src = preprocess(
        src,
        &[
            (
                "$PARAM_FIELDS",
                "q_seq: u32, kv_seq: u32, packed_heads: u32, head_dim: u32,",
            ),
            (
                "$PARSE_PARAMS",
                "let q_seq = params.q_seq;\n    let num_heads = params.packed_heads >> 16u;\n    let num_kv_heads = params.packed_heads & 0xFFFFu;\n    let head_dim = params.head_dim;\n    let kv_len = params.kv_seq;",
            ),
        ],
    );
    parse_wgsl(&src)
}

// ---------------------------------------------------------------------------
// cache_write.wgsl: write [1, dim] into row kv_pos of [max_seq, dim]
// ---------------------------------------------------------------------------

fn gen_rope_dynamic() -> ShaderModule {
    parse_wgsl(include_str!("shaders/rope_dynamic.wgsl"))
}

// ---------------------------------------------------------------------------
// group_norm.wgsl — Group normalization forward
// ---------------------------------------------------------------------------

fn gen_group_norm() -> ShaderModule {
    parse_wgsl(include_str!("shaders/group_norm.wgsl"))
}

// ---------------------------------------------------------------------------
// group_norm_grad.wgsl — GroupNorm backward
// ---------------------------------------------------------------------------

fn gen_group_norm_grad() -> ShaderModule {
    parse_wgsl(include_str!("shaders/group_norm_grad.wgsl"))
}

// ---------------------------------------------------------------------------
// concat.wgsl — Channel concatenation
// ---------------------------------------------------------------------------

fn gen_concat() -> ShaderModule {
    parse_wgsl(include_str!("shaders/concat.wgsl"))
}

// ---------------------------------------------------------------------------
// split.wgsl — Channel split (backward of concat)
// ---------------------------------------------------------------------------

fn gen_split() -> ShaderModule {
    parse_wgsl(include_str!("shaders/split.wgsl"))
}

// ---------------------------------------------------------------------------
// upsample.wgsl — Nearest-neighbor 2x upsample
// ---------------------------------------------------------------------------

fn gen_upsample() -> ShaderModule {
    parse_wgsl(include_str!("shaders/upsample.wgsl"))
}

// ---------------------------------------------------------------------------
// upsample_grad.wgsl — Backward of 2x upsample
// ---------------------------------------------------------------------------

fn gen_upsample_grad() -> ShaderModule {
    parse_wgsl(include_str!("shaders/upsample_grad.wgsl"))
}

// ---------------------------------------------------------------------------
// conv2d.wgsl — 2D convolution forward
// ---------------------------------------------------------------------------

fn gen_conv2d() -> ShaderModule {
    parse_wgsl(include_str!("shaders/conv2d.wgsl"))
}

// ---------------------------------------------------------------------------
// conv2d_gemm.wgsl — Conv2d forward via implicit GEMM
// ---------------------------------------------------------------------------

fn gen_conv2d_gemm() -> ShaderModule {
    parse_wgsl(include_str!("shaders/conv2d_gemm.wgsl"))
}

fn gen_conv2d_gemm_small() -> ShaderModule {
    parse_wgsl(include_str!("shaders/conv2d_gemm_small.wgsl"))
}

// ---------------------------------------------------------------------------
// conv2d_grad_input.wgsl — Conv2d backward w.r.t. input
// ---------------------------------------------------------------------------

fn gen_conv2d_grad_input() -> ShaderModule {
    parse_wgsl(include_str!("shaders/conv2d_grad_input.wgsl"))
}

// ---------------------------------------------------------------------------
// conv2d_grad_input_gemm.wgsl — Conv2d backward w.r.t. input via implicit GEMM
// ---------------------------------------------------------------------------

fn gen_conv2d_grad_input_gemm() -> ShaderModule {
    parse_wgsl(include_str!("shaders/conv2d_grad_input_gemm.wgsl"))
}

fn gen_conv2d_grad_input_gemm_small() -> ShaderModule {
    parse_wgsl(include_str!("shaders/conv2d_grad_input_gemm_small.wgsl"))
}

fn gen_conv2d_grad_input_gemm_coop() -> ShaderModule {
    let default_config = CoopConfig {
        tile_size: 16,
        use_f16_input: true,
    };
    gen_conv2d_grad_input_gemm_coop_wgsl(&default_config)
}

fn gen_conv2d_grad_input_gemm_coop_wgsl(config: &CoopConfig) -> ShaderModule {
    let tile = config.tile_size;
    let output_tile = config.output_tile();
    let shared_size = tile * tile;
    let wg_size: u32 = 64;
    let staging_iters = shared_size / wg_size;
    let row_stride = wg_size / tile;
    let tile_mask = tile - 1;
    let tile_shift = tile.trailing_zeros();

    let (elem_type, enable_f16, elem_zero, cast_open, cast_close) = if config.use_f16_input {
        ("f16", "enable f16;", "f16(0.0)", "f16(", ")")
    } else {
        ("f32", "", "0.0", "", "")
    };
    let ab_type = if config.use_f16_input { "f16" } else { "f32" };
    let coop_ab = format!("coop_mat{}x{}<{},A>", tile, tile, ab_type);
    let coop_ba = format!("coop_mat{}x{}<{},B>", tile, tile, ab_type);
    let coop_c = format!("coop_mat{}x{}<f32,C>", tile, tile);

    let acc_init = format!(
        "var acc00 = {coop_c}();\n\
         \x20   var acc01 = {coop_c}();\n\
         \x20   var acc10 = {coop_c}();\n\
         \x20   var acc11 = {coop_c}();"
    );

    let output_tile_u = format!("{}u", output_tile);
    let tile_size_u = format!("{}u", tile);
    let tile_mask_u = format!("{}u", tile_mask);
    let tile_shift_u = format!("{}u", tile_shift);
    let staging_iters_u = format!("{}u", staging_iters);
    let row_stride_u = format!("{}u", row_stride);
    let shared_size_s = format!("{}", shared_size);

    let src = include_str!("shaders/conv2d_grad_input_gemm_coop.wgsl");
    let src = preprocess(
        src,
        &[
            ("$ENABLE_F16", enable_f16),
            ("$ELEM_TYPE", elem_type),
            ("$ELEM_ZERO", elem_zero),
            ("$SHARED_SIZE", &shared_size_s),
            ("$OUTPUT_TILE_U", &output_tile_u),
            ("$TILE_SIZE_U", &tile_size_u),
            ("$TILE_MASK_U", &tile_mask_u),
            ("$TILE_SHIFT_U", &tile_shift_u),
            ("$STAGING_ITERS_U", &staging_iters_u),
            ("$ROW_STRIDE_U", &row_stride_u),
            ("$CAST_OPEN", cast_open),
            ("$CAST_CLOSE", cast_close),
            ("$COOP_AB", &coop_ab),
            ("$COOP_BA", &coop_ba),
            ("$ACC_INIT", &acc_init),
        ],
    );
    parse_wgsl(&src)
}

// ---------------------------------------------------------------------------
// conv2d_grad_weight.wgsl — Conv2d backward w.r.t. kernel weights
// ---------------------------------------------------------------------------

fn gen_conv2d_grad_weight() -> ShaderModule {
    parse_wgsl(include_str!("shaders/conv2d_grad_weight.wgsl"))
}

fn gen_conv2d_grad_weight_gemm() -> ShaderModule {
    parse_wgsl(include_str!("shaders/conv2d_grad_weight_gemm.wgsl"))
}

fn gen_cache_write() -> ShaderModule {
    parse_wgsl(include_str!("shaders/cache_write.wgsl"))
}

// ---------------------------------------------------------------------------
// cached_attention.wgsl: single-token attention with KV cache
// ---------------------------------------------------------------------------

fn gen_cached_attention() -> ShaderModule {
    parse_wgsl(include_str!("shaders/cached_attention.wgsl"))
}

// ---------------------------------------------------------------------------
// multi_head_attn (forward, saves LSE for backward)
// Same as gen_attention_parallel(false, true) but with an extra `lse` binding.
// After normalization, thread 0 writes lse[pos * num_heads + head] = max + log(sum_exp).
// ---------------------------------------------------------------------------

fn gen_mha_forward() -> ShaderModule {
    parse_wgsl(include_str!("shaders/mha_forward.wgsl"))
}

// ---------------------------------------------------------------------------
// gen_mha_grad_q: dQ computation for MultiHeadAttn backward
// dispatch [q_seq, num_heads, 1], WG=64
// inputs: [dO, Q, K, V, LSE, O], output: dQ
// ---------------------------------------------------------------------------

fn gen_mha_grad_q() -> ShaderModule {
    parse_wgsl(include_str!("shaders/mha_grad_q.wgsl"))
}

// ---------------------------------------------------------------------------
// gen_mha_grad_k: dK computation for MultiHeadAttn backward
// dispatch [kv_seq, num_kv_heads, 1], WG=64
// ---------------------------------------------------------------------------

fn gen_mha_grad_k() -> ShaderModule {
    parse_wgsl(include_str!("shaders/mha_grad_k.wgsl"))
}

// ---------------------------------------------------------------------------
// gen_mha_grad_v: dV computation for MultiHeadAttn backward
// dispatch [kv_seq, num_kv_heads, 1], WG=64
// ---------------------------------------------------------------------------

fn gen_mha_grad_v() -> ShaderModule {
    parse_wgsl(include_str!("shaders/mha_grad_v.wgsl"))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Verify every shader group generates a valid Naga module.
    #[test]
    fn all_shaders_generate_valid_modules() {
        let groups = [
            (ShaderGroup::Unary, naga::valid::Capabilities::empty()),
            (ShaderGroup::Binary, naga::valid::Capabilities::empty()),
            (ShaderGroup::BiasAdd, naga::valid::Capabilities::empty()),
            (ShaderGroup::Sgd, naga::valid::Capabilities::empty()),
            (ShaderGroup::Adam, naga::valid::Capabilities::empty()),
            (ShaderGroup::Transpose, naga::valid::Capabilities::empty()),
            (ShaderGroup::MatMul, naga::valid::Capabilities::empty()),
            (ShaderGroup::MatMulAdd, naga::valid::Capabilities::empty()),
            (ShaderGroup::MatMulAT, naga::valid::Capabilities::empty()),
            (ShaderGroup::MatMulBT, naga::valid::Capabilities::empty()),
            (ShaderGroup::MatMulATAdd, naga::valid::Capabilities::empty()),
            (ShaderGroup::MatMulBTAdd, naga::valid::Capabilities::empty()),
            (
                ShaderGroup::MatMulCoop,
                naga::valid::Capabilities::COOPERATIVE_MATRIX
                    | naga::valid::Capabilities::SHADER_FLOAT16,
            ),
            (
                ShaderGroup::MatMulCoopAdd,
                naga::valid::Capabilities::COOPERATIVE_MATRIX
                    | naga::valid::Capabilities::SHADER_FLOAT16,
            ),
            (
                ShaderGroup::MatMulCoopAT,
                naga::valid::Capabilities::COOPERATIVE_MATRIX
                    | naga::valid::Capabilities::SHADER_FLOAT16,
            ),
            (
                ShaderGroup::MatMulCoopBT,
                naga::valid::Capabilities::COOPERATIVE_MATRIX
                    | naga::valid::Capabilities::SHADER_FLOAT16,
            ),
            (
                ShaderGroup::Conv2dGradInputGemmCoop,
                naga::valid::Capabilities::COOPERATIVE_MATRIX
                    | naga::valid::Capabilities::SHADER_FLOAT16,
            ),
            (ShaderGroup::Reduce, naga::valid::Capabilities::empty()),
            (ShaderGroup::Softmax, naga::valid::Capabilities::empty()),
            (
                ShaderGroup::CrossEntropy,
                naga::valid::Capabilities::empty(),
            ),
            (ShaderGroup::RmsNorm, naga::valid::Capabilities::empty()),
            (ShaderGroup::Embedding, naga::valid::Capabilities::empty()),
            (ShaderGroup::RoPE, naga::valid::Capabilities::empty()),
            (
                ShaderGroup::CausalAttention,
                naga::valid::Capabilities::empty(),
            ),
            (ShaderGroup::LayerNorm, naga::valid::Capabilities::empty()),
            (
                ShaderGroup::FullAttention,
                naga::valid::Capabilities::empty(),
            ),
            (
                ShaderGroup::CrossAttention,
                naga::valid::Capabilities::empty(),
            ),
            (
                ShaderGroup::MultiHeadAttn,
                naga::valid::Capabilities::empty(),
            ),
            (
                ShaderGroup::MultiHeadAttnGradQ,
                naga::valid::Capabilities::empty(),
            ),
            (
                ShaderGroup::MultiHeadAttnGradK,
                naga::valid::Capabilities::empty(),
            ),
            (
                ShaderGroup::MultiHeadAttnGradV,
                naga::valid::Capabilities::empty(),
            ),
            (ShaderGroup::SwiGLUGrad, naga::valid::Capabilities::empty()),
            (
                ShaderGroup::SwiGLUConcat,
                naga::valid::Capabilities::empty(),
            ),
            (ShaderGroup::SumRows, naga::valid::Capabilities::empty()),
            (ShaderGroup::RmsNormGrad, naga::valid::Capabilities::empty()),
            (ShaderGroup::ScatterAdd, naga::valid::Capabilities::empty()),
            (ShaderGroup::BceLoss, naga::valid::Capabilities::empty()),
            (
                ShaderGroup::FusedRmsNormMatMul,
                naga::valid::Capabilities::empty(),
            ),
        ];

        let flags = naga::valid::ValidationFlags::all() ^ naga::valid::ValidationFlags::BINDINGS;
        for &(group, caps) in &groups {
            let sm = generate_module(group);
            naga::valid::Validator::new(flags, caps)
                .validate(&sm.module)
                .unwrap_or_else(|e| {
                    panic!("{group:?}: generated module failed validation: {e:#?}")
                });
        }
    }

    /// Verify the generated modules contain the expected entry points.
    #[test]
    fn entry_points_present() {
        let m = generate_module(ShaderGroup::Unary);
        let names: Vec<&str> = m
            .module
            .entry_points
            .iter()
            .map(|ep| ep.name.as_str())
            .collect();
        assert!(names.contains(&"relu"), "missing relu");
        assert!(names.contains(&"sigmoid"), "missing sigmoid");
        assert!(names.contains(&"neg"), "missing neg");
        assert!(names.contains(&"silu"), "missing silu");

        let m = generate_module(ShaderGroup::Binary);
        let names: Vec<&str> = m
            .module
            .entry_points
            .iter()
            .map(|ep| ep.name.as_str())
            .collect();
        assert!(names.contains(&"add"));
        assert!(names.contains(&"mul"));
        assert!(names.contains(&"greater"));

        let m = generate_module(ShaderGroup::Reduce);
        let names: Vec<&str> = m
            .module
            .entry_points
            .iter()
            .map(|ep| ep.name.as_str())
            .collect();
        assert!(names.contains(&"sum_all"));
        assert!(names.contains(&"mean_all"));
    }

    #[test]
    fn test_rms_norm_wgsl() {
        let _ = generate_wgsl(ShaderGroup::RmsNorm);
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
    fn test_causal_attention_wgsl() {
        let _ = generate_wgsl(ShaderGroup::CausalAttention);
    }

    /// Verify every shader group compiles to SPIR-V without panics.
    /// This catches "Expression [N] is not cached!" bugs in hand-built IR.
    /// Skipped on Apple targets where naga's spv-out backend is not available.
    #[test]
    #[cfg(not(target_vendor = "apple"))]
    fn all_shaders_compile_to_spirv() {
        let empty = naga::valid::Capabilities::empty();
        let coop = naga::valid::Capabilities::COOPERATIVE_MATRIX
            | naga::valid::Capabilities::SHADER_FLOAT16;
        let groups: &[(ShaderGroup, naga::valid::Capabilities)] = &[
            (ShaderGroup::Unary, empty),
            (ShaderGroup::Binary, empty),
            (ShaderGroup::BiasAdd, empty),
            (ShaderGroup::Sgd, empty),
            (ShaderGroup::Adam, empty),
            (ShaderGroup::Transpose, empty),
            (ShaderGroup::MatMul, empty),
            (ShaderGroup::MatMulAdd, empty),
            (ShaderGroup::MatMulAT, empty),
            (ShaderGroup::MatMulBT, empty),
            (ShaderGroup::MatMulATAdd, empty),
            (ShaderGroup::MatMulBTAdd, empty),
            (ShaderGroup::MatMulCoop, coop),
            (ShaderGroup::MatMulCoopAdd, coop),
            (ShaderGroup::MatMulCoopAT, coop),
            (ShaderGroup::MatMulCoopBT, coop),
            (ShaderGroup::Reduce, empty),
            (ShaderGroup::Softmax, empty),
            (ShaderGroup::CrossEntropy, empty),
            (ShaderGroup::RmsNorm, empty),
            (ShaderGroup::Embedding, empty),
            (ShaderGroup::RoPE, empty),
            (ShaderGroup::CausalAttention, empty),
            (ShaderGroup::LayerNorm, empty),
            (ShaderGroup::FullAttention, empty),
            (ShaderGroup::CrossAttention, empty),
            (ShaderGroup::SwiGLUGrad, empty),
            (ShaderGroup::SwiGLUConcat, empty),
            (ShaderGroup::SumRows, empty),
            (ShaderGroup::RmsNormGrad, empty),
            (ShaderGroup::ScatterAdd, empty),
            (ShaderGroup::BceLoss, empty),
            (ShaderGroup::FusedRmsNormMatMul, empty),
        ];

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
        for &(group, caps) in groups {
            // See note in all_shaders_generate_valid_modules
            if matches!(
                group,
                ShaderGroup::MatMulCoop
                    | ShaderGroup::MatMulCoopAdd
                    | ShaderGroup::MatMulCoopAT
                    | ShaderGroup::MatMulCoopBT
                    | ShaderGroup::Conv2dGradInputGemmCoop
            ) {
                continue;
            }
            let sm = generate_module(group);
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
            match entry {
                ShaderEntry::MatMul | ShaderEntry::MatMulAT | ShaderEntry::MatMulBT => {
                    vec!["matrix_a", "matrix_b", "matrix_c", "params"]
                }
                ShaderEntry::FusedMatMulAdd
                | ShaderEntry::FusedMatMulATAdd
                | ShaderEntry::FusedMatMulBTAdd => {
                    vec!["matrix_a", "matrix_b", "matrix_c", "src", "params"]
                }
                ShaderEntry::Relu
                | ShaderEntry::Sigmoid
                | ShaderEntry::Neg
                | ShaderEntry::Abs
                | ShaderEntry::Log
                | ShaderEntry::Recip
                | ShaderEntry::Silu
                | ShaderEntry::Gelu
                | ShaderEntry::SumAll
                | ShaderEntry::MeanAll
                | ShaderEntry::SumRows
                | ShaderEntry::RoPE => vec!["src", "dst", "params"],
                ShaderEntry::Add
                | ShaderEntry::Mul
                | ShaderEntry::Greater
                | ShaderEntry::SwiGLU => {
                    vec!["src_a", "src_b", "dst", "params"]
                }
                ShaderEntry::BiasAdd => vec!["src", "bias", "dst", "params"],
                ShaderEntry::SgdUpdate => vec!["param", "grad", "dst", "params"],
                ShaderEntry::AdamUpdate => vec!["param", "grad", "m", "v", "params"],
                ShaderEntry::ScatterAdd => vec!["indices", "src", "dst", "params"],
                ShaderEntry::BceLoss => vec!["pred", "labels", "grad_out", "loss_out", "params"],
                ShaderEntry::Softmax => vec!["src", "dst", "params"],
                ShaderEntry::CrossEntropyLoss => {
                    vec!["logits", "labels", "grad_out", "loss_out", "params"]
                }
                ShaderEntry::Transpose => vec!["src", "dst", "params"],
                ShaderEntry::RmsNorm => vec!["src", "bias", "dst", "params"],
                ShaderEntry::Embedding => vec!["indices", "src", "dst", "params"],
                ShaderEntry::CausalAttention
                | ShaderEntry::FullAttention
                | ShaderEntry::CrossAttention => vec!["src_a", "src_b", "bias", "dst", "params"],
                ShaderEntry::LayerNorm => vec!["src", "src_b", "bias", "dst", "params"],
                ShaderEntry::MultiHeadAttn => {
                    vec!["src_a", "src_b", "bias", "dst", "lse", "params"]
                }
                ShaderEntry::MultiHeadAttnGradQ
                | ShaderEntry::MultiHeadAttnGradK
                | ShaderEntry::MultiHeadAttnGradV => {
                    vec![
                        "d_out", "src_a", "src_b", "bias", "lse", "fwd_dst", "dst", "params",
                    ]
                }
                // All three SwiGLUGrad entries share the same module globals
                ShaderEntry::SwiGLUGradGate | ShaderEntry::SwiGLUGradUp | ShaderEntry::SiluGrad => {
                    vec!["src_a", "src_b", "src_c", "dst", "params"]
                }
                ShaderEntry::SwiGLUConcat | ShaderEntry::SwiGLUConcatGrad => {
                    vec!["src_a", "src_b", "dst", "params"]
                }
                ShaderEntry::RmsNormGradW | ShaderEntry::RmsNormGradX => {
                    vec!["src_a", "src_b", "bias", "dst", "params"]
                }
                ShaderEntry::FusedRmsNormMatMul => {
                    vec!["src_a", "src_b", "bias", "dst", "params"]
                }
                ShaderEntry::CacheWrite => vec!["src", "dst", "kv_pos_buf", "params"],
                ShaderEntry::CachedAttention => {
                    vec!["src_a", "src_b", "bias", "kv_pos_buf", "dst", "params"]
                }
                ShaderEntry::GroupNorm => vec!["src", "src_b", "bias", "dst", "params"],
                ShaderEntry::GroupNormGradInput => vec!["src_a", "src_b", "bias", "dst", "params"],
                ShaderEntry::GroupNormGradWeightBias => {
                    vec!["src_a", "src_b", "bias", "dst", "params"]
                }
                ShaderEntry::Concat => vec!["src_a", "src_b", "dst", "params"],
                ShaderEntry::SplitA | ShaderEntry::SplitB => vec!["src", "dst", "params"],
                ShaderEntry::Upsample2x | ShaderEntry::Upsample2xGrad => {
                    vec!["src", "dst", "params"]
                }
                ShaderEntry::Conv2d => vec!["src", "weight", "dst", "params"],
                ShaderEntry::Conv2dGemm | ShaderEntry::Conv2dGemmSmall => {
                    vec!["src", "weight", "dst", "params"]
                }
                ShaderEntry::Conv2dGradInput => vec!["grad_out", "weight", "dst", "params"],
                ShaderEntry::Conv2dGradInputGemm | ShaderEntry::Conv2dGradInputGemmSmall => {
                    vec!["grad_out", "weight", "dst", "params"]
                }
                ShaderEntry::Conv2dGradInputGemmCoop => {
                    vec!["grad_out", "weight", "dst", "params"]
                }
                ShaderEntry::Conv2dGradWeight | ShaderEntry::Conv2dGradWeightGemm => {
                    vec!["grad_out", "src", "dst", "params"]
                }
                ShaderEntry::RoPEDynamic => vec!["src", "dst", "pos_offset_buf", "params"],
            }
        }

        let entries = [
            ShaderEntry::MatMul,
            ShaderEntry::MatMulAT,
            ShaderEntry::MatMulBT,
            ShaderEntry::FusedMatMulAdd,
            ShaderEntry::FusedMatMulATAdd,
            ShaderEntry::FusedMatMulBTAdd,
            ShaderEntry::Relu,
            ShaderEntry::Sigmoid,
            ShaderEntry::Neg,
            ShaderEntry::Abs,
            ShaderEntry::Log,
            ShaderEntry::Recip,
            ShaderEntry::Add,
            ShaderEntry::Mul,
            ShaderEntry::Greater,
            ShaderEntry::BiasAdd,
            ShaderEntry::SgdUpdate,
            ShaderEntry::SumAll,
            ShaderEntry::MeanAll,
            ShaderEntry::Softmax,
            ShaderEntry::CrossEntropyLoss,
            ShaderEntry::Transpose,
            ShaderEntry::Silu,
            ShaderEntry::RmsNorm,
            ShaderEntry::Embedding,
            ShaderEntry::RoPE,
            ShaderEntry::CausalAttention,
            ShaderEntry::Gelu,
            ShaderEntry::LayerNorm,
            ShaderEntry::FullAttention,
            ShaderEntry::CrossAttention,
            ShaderEntry::MultiHeadAttn,
            ShaderEntry::MultiHeadAttnGradQ,
            ShaderEntry::MultiHeadAttnGradK,
            ShaderEntry::MultiHeadAttnGradV,
            ShaderEntry::SwiGLUGradGate,
            ShaderEntry::SwiGLUGradUp,
            ShaderEntry::SwiGLUConcat,
            ShaderEntry::SwiGLUConcatGrad,
            ShaderEntry::SiluGrad,
            ShaderEntry::RmsNormGradW,
            ShaderEntry::RmsNormGradX,
            ShaderEntry::FusedRmsNormMatMul,
            ShaderEntry::AdamUpdate,
            ShaderEntry::ScatterAdd,
            ShaderEntry::BceLoss,
            ShaderEntry::GroupNorm,
            ShaderEntry::GroupNormGradInput,
            ShaderEntry::GroupNormGradWeightBias,
            ShaderEntry::Concat,
            ShaderEntry::SplitA,
            ShaderEntry::SplitB,
            ShaderEntry::Upsample2x,
            ShaderEntry::Upsample2xGrad,
            ShaderEntry::Conv2d,
            ShaderEntry::Conv2dGemm,
            ShaderEntry::Conv2dGemmSmall,
            ShaderEntry::Conv2dGradInput,
            ShaderEntry::Conv2dGradInputGemm,
            ShaderEntry::Conv2dGradInputGemmSmall,
            ShaderEntry::Conv2dGradWeight,
            ShaderEntry::CacheWrite,
            ShaderEntry::CachedAttention,
            ShaderEntry::RoPEDynamic,
        ];

        for entry in &entries {
            let group = entry.shader_group();
            let expected: HashSet<&str> = expected_globals(entry).into_iter().collect();

            let sm = generate_module(group);

            let actual: HashSet<&str> = sm
                .module
                .global_variables
                .iter()
                .filter_map(|(_, gv)| {
                    // Skip workgroup variables — blade doesn't bind those
                    if gv.space == naga::AddressSpace::WorkGroup {
                        return None;
                    }
                    gv.name.as_deref()
                })
                .collect();

            assert_eq!(
                expected, actual,
                "{entry:?} (group {group:?}): shader globals {actual:?} \
                 don't match expected runtime bindings {expected:?}"
            );
        }
    }
}
