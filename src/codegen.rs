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

/// Generate WGSL declarations and body for a fused epilogue chain.
///
/// Returns (declarations, body) where declarations are `var<storage>`
/// lines for extra buffers, and body is a sequence of WGSL statements
/// that transform `val` (the matmul result for one output element).
pub fn epilogue_to_wgsl(epilogue: &[crate::compile::EpilogueOp]) -> (String, String) {
    use crate::compile::EpilogueOp;
    let mut decls = Vec::new();
    let mut body = Vec::new();
    let mut declared = std::collections::HashSet::new();

    for op in epilogue {
        #[allow(clippy::pattern_type_mismatch)]
        match op {
            EpilogueOp::Add(buf_idx) => {
                let name = format!("epi_buf_{}", buf_idx);
                if declared.insert(*buf_idx) {
                    decls.push(format!("var<storage> {}: array<f32>;", name));
                }
                body.push(format!("val = val + {}[idx];", name));
            }
            EpilogueOp::BiasAdd(buf_idx) => {
                let name = format!("epi_buf_{}", buf_idx);
                if declared.insert(*buf_idx) {
                    decls.push(format!("var<storage> {}: array<f32>;", name));
                }
                body.push(format!("val = val + {}[col];", name));
            }
            EpilogueOp::Relu => {
                body.push("val = max(val, 0.0);".to_string());
            }
            EpilogueOp::Silu => {
                body.push("val = val / (1.0 + exp(-val));".to_string());
            }
            EpilogueOp::Sigmoid => {
                body.push("val = 1.0 / (1.0 + exp(-val));".to_string());
            }
            EpilogueOp::Neg => {
                body.push("val = -val;".to_string());
            }
        }
    }
    (decls.join("\n"), body.join("\n                "))
}

/// Generate a matmul shader module with a fused epilogue chain.
///
/// Used by the runtime when a dispatch has a non-empty epilogue field.
/// The epilogue ops are compiled into WGSL statements that transform
/// each output element before storing it.
pub fn generate_matmul_with_epilogue(
    group: ShaderGroup,
    epilogue: &[crate::compile::EpilogueOp],
) -> ShaderModule {
    let (epi_decl, epi_body) = epilogue_to_wgsl(epilogue);
    match group {
        ShaderGroup::MatMul => matmul_vars_epilogue(
            MATMUL_A_FWD,
            MATMUL_B_FWD,
            A_ROW_FWD,
            A_COL_FWD,
            B_ROW_FWD,
            B_COL_FWD,
            "",
            "",
            &epi_decl,
            &epi_body,
        ),
        ShaderGroup::MatMulAdd => matmul_vars_epilogue(
            MATMUL_A_FWD,
            MATMUL_B_FWD,
            A_ROW_FWD,
            A_COL_FWD,
            B_ROW_FWD,
            B_COL_FWD,
            "var<storage> src: array<f32>;",
            " + src[idx]",
            &epi_decl,
            &epi_body,
        ),
        ShaderGroup::MatMulAT => matmul_vars_epilogue(
            MATMUL_A_AT,
            MATMUL_B_FWD,
            A_ROW_AT,
            A_COL_AT,
            B_ROW_FWD,
            B_COL_FWD,
            "",
            "",
            &epi_decl,
            &epi_body,
        ),
        ShaderGroup::MatMulBT => matmul_vars_epilogue(
            MATMUL_A_FWD,
            MATMUL_B_BT,
            A_ROW_FWD,
            A_COL_FWD,
            B_ROW_BT,
            B_COL_BT,
            "",
            "",
            &epi_decl,
            &epi_body,
        ),
        ShaderGroup::MatMulATAdd => matmul_vars_epilogue(
            MATMUL_A_AT,
            MATMUL_B_FWD,
            A_ROW_AT,
            A_COL_AT,
            B_ROW_FWD,
            B_COL_FWD,
            "var<storage> src: array<f32>;",
            " + src[idx]",
            &epi_decl,
            &epi_body,
        ),
        ShaderGroup::MatMulBTAdd => matmul_vars_epilogue(
            MATMUL_A_FWD,
            MATMUL_B_BT,
            A_ROW_FWD,
            A_COL_FWD,
            B_ROW_BT,
            B_COL_BT,
            "var<storage> src: array<f32>;",
            " + src[idx]",
            &epi_decl,
            &epi_body,
        ),
        _ => panic!("epilogue fusion not supported for {:?}", group),
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
    /// M=1 matmul (GEMV): C[1,N] = A[1,K] × B[K,N]. One thread per
    /// output column; dispatched for batch-1 decode on transformers.
    MatMulGemv,
    /// M=1 matmul with fused residual add: C[1,N] = A×B + D[1,N].
    /// Same shape as MatMulGemv plus one extra storage input.
    MatMulGemvAdd,
    /// M=1 MatMulBT (B stored [N,K]): C[1,N] = A × Bᵀ. K-split with
    /// coalesced contiguous-K vec4 loads.
    MatMulGemvBT,
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
    RoPEGrad,
    CausalAttention,
    CausalAttentionRoPE,
    SlidingWindowAttention,
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
    LayerNormGrad,
    ScatterAdd,
    BceLoss,
    FusedRmsNormMatMul,
    FusedRmsNormMatMulCoop,
    RmsNormRsqrt,
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
    GroupNormSilu,
    Conv2dGradWeight,
    Conv2dGradWeightGemm,
    Conv2dGradWeightGemmSmall,
    CacheWrite,
    CachedAttention,
    RoPEDynamic,
    MaxPool2d,
    GlobalAvgPool,
    GlobalAvgPoolGrad,
}

/// Generate a `naga::Module` for a shader group.
pub fn generate_module(group: ShaderGroup) -> ShaderModule {
    match group {
        ShaderGroup::Unary => parse_wgsl(include_str!("shaders/unary.wgsl")),
        ShaderGroup::Binary => parse_wgsl(include_str!("shaders/binary.wgsl")),
        ShaderGroup::BiasAdd => parse_wgsl(include_str!("shaders/bias_add.wgsl")),
        ShaderGroup::Sgd => parse_wgsl(include_str!("shaders/sgd.wgsl")),
        ShaderGroup::Adam => parse_wgsl(include_str!("shaders/adam.wgsl")),
        ShaderGroup::Transpose => parse_wgsl(include_str!("shaders/transpose.wgsl")),
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
        ShaderGroup::MatMulGemv => parse_wgsl(include_str!("shaders/matmul_gemv.wgsl")),
        ShaderGroup::MatMulGemvAdd => parse_wgsl(include_str!("shaders/matmul_gemv_add.wgsl")),
        ShaderGroup::MatMulGemvBT => parse_wgsl(include_str!("shaders/matmul_gemv_bt.wgsl")),
        ShaderGroup::MatMulCoop => gen_matmul_coop(),
        ShaderGroup::MatMulCoopAdd => gen_matmul_coop_add(),
        ShaderGroup::MatMulCoopAT => gen_matmul_coop_at(),
        ShaderGroup::MatMulCoopBT => gen_matmul_coop_bt(),
        ShaderGroup::Reduce => parse_wgsl(include_str!("shaders/reduce.wgsl")),
        ShaderGroup::Softmax => parse_wgsl(include_str!("shaders/softmax.wgsl")),
        ShaderGroup::CrossEntropy => parse_wgsl(include_str!("shaders/cross_entropy.wgsl")),
        ShaderGroup::RmsNorm => parse_wgsl(include_str!("shaders/rms_norm.wgsl")),
        ShaderGroup::Embedding => parse_wgsl(include_str!("shaders/embedding.wgsl")),
        ShaderGroup::RoPE => parse_wgsl(include_str!("shaders/rope.wgsl")),
        ShaderGroup::RoPEGrad => parse_wgsl(include_str!("shaders/rope_grad.wgsl")),
        ShaderGroup::CausalAttention => gen_causal_attention(),
        ShaderGroup::CausalAttentionRoPE => gen_causal_attention(), // TODO: generate RoPE-fused shader
        ShaderGroup::SlidingWindowAttention => gen_sliding_window_attention(),
        ShaderGroup::LayerNorm => parse_wgsl(include_str!("shaders/layer_norm.wgsl")),
        ShaderGroup::FullAttention => gen_full_attention(),
        ShaderGroup::CrossAttention => gen_cross_attention(),
        ShaderGroup::MultiHeadAttn => parse_wgsl(include_str!("shaders/mha_forward.wgsl")),
        ShaderGroup::MultiHeadAttnGradQ => parse_wgsl(include_str!("shaders/mha_grad_q.wgsl")),
        ShaderGroup::MultiHeadAttnGradK => parse_wgsl(include_str!("shaders/mha_grad_k.wgsl")),
        ShaderGroup::MultiHeadAttnGradV => parse_wgsl(include_str!("shaders/mha_grad_v.wgsl")),
        ShaderGroup::SwiGLUGrad => parse_wgsl(include_str!("shaders/swiglu_grad.wgsl")),
        ShaderGroup::SwiGLUConcat => parse_wgsl(include_str!("shaders/swiglu_concat.wgsl")),
        ShaderGroup::SumRows => parse_wgsl(include_str!("shaders/sum_rows.wgsl")),
        ShaderGroup::RmsNormGrad => parse_wgsl(include_str!("shaders/rms_norm_grad.wgsl")),
        ShaderGroup::LayerNormGrad => parse_wgsl(include_str!("shaders/layer_norm_grad.wgsl")),
        ShaderGroup::FusedRmsNormMatMul => parse_wgsl(include_str!("shaders/matmul_rms_norm.wgsl")),
        ShaderGroup::RmsNormRsqrt => parse_wgsl(include_str!("shaders/rms_norm_rsqrt.wgsl")),
        // Coop groups must use generate_coop_module(), not generate_module()
        ShaderGroup::FusedRmsNormMatMulCoop => {
            panic!("use generate_coop_module for FusedRmsNormMatMulCoop")
        }
        ShaderGroup::ScatterAdd => parse_wgsl(include_str!("shaders/scatter_add.wgsl")),
        ShaderGroup::BceLoss => parse_wgsl(include_str!("shaders/bce.wgsl")),
        ShaderGroup::GroupNorm => parse_wgsl(include_str!("shaders/group_norm.wgsl")),
        ShaderGroup::GroupNormGrad => parse_wgsl(include_str!("shaders/group_norm_grad.wgsl")),
        ShaderGroup::Concat => parse_wgsl(include_str!("shaders/concat.wgsl")),
        ShaderGroup::Split => parse_wgsl(include_str!("shaders/split.wgsl")),
        ShaderGroup::Upsample => parse_wgsl(include_str!("shaders/upsample.wgsl")),
        ShaderGroup::UpsampleGrad => parse_wgsl(include_str!("shaders/upsample_grad.wgsl")),
        ShaderGroup::Conv2d => parse_wgsl(include_str!("shaders/conv2d.wgsl")),
        ShaderGroup::Conv2dGemm => parse_wgsl(include_str!("shaders/conv2d_gemm.wgsl")),
        ShaderGroup::Conv2dGemmSmall => parse_wgsl(include_str!("shaders/conv2d_gemm_small.wgsl")),
        ShaderGroup::Conv2dGradInput => parse_wgsl(include_str!("shaders/conv2d_grad_input.wgsl")),
        ShaderGroup::Conv2dGradInputGemm => {
            parse_wgsl(include_str!("shaders/conv2d_grad_input_gemm.wgsl"))
        }
        ShaderGroup::Conv2dGradInputGemmSmall => {
            parse_wgsl(include_str!("shaders/conv2d_grad_input_gemm_small.wgsl"))
        }
        ShaderGroup::Conv2dGradInputGemmCoop => gen_conv2d_grad_input_gemm_coop(),
        ShaderGroup::GroupNormSilu => parse_wgsl(include_str!("shaders/group_norm_silu.wgsl")),
        ShaderGroup::Conv2dGradWeight => {
            parse_wgsl(include_str!("shaders/conv2d_grad_weight.wgsl"))
        }
        ShaderGroup::Conv2dGradWeightGemm => {
            parse_wgsl(include_str!("shaders/conv2d_grad_weight_gemm.wgsl"))
        }
        ShaderGroup::Conv2dGradWeightGemmSmall => {
            parse_wgsl(include_str!("shaders/conv2d_grad_weight_gemm_small.wgsl"))
        }
        ShaderGroup::CacheWrite => parse_wgsl(include_str!("shaders/cache_write.wgsl")),
        ShaderGroup::CachedAttention => parse_wgsl(include_str!("shaders/cached_attention.wgsl")),
        ShaderGroup::RoPEDynamic => parse_wgsl(include_str!("shaders/rope_dynamic.wgsl")),
        ShaderGroup::MaxPool2d => parse_wgsl(include_str!("shaders/max_pool_2d.wgsl")),
        ShaderGroup::GlobalAvgPool => parse_wgsl(include_str!("shaders/global_avg_pool.wgsl")),
        ShaderGroup::GlobalAvgPoolGrad => {
            parse_wgsl(include_str!("shaders/global_avg_pool_grad.wgsl"))
        }
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
        ShaderGroup::FusedRmsNormMatMulCoop => gen_fused_rms_norm_matmul_coop_wgsl(config),
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
        | ShaderGroup::MatMulCoopBT
        | ShaderGroup::FusedRmsNormMatMulCoop => {
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
/// Large-tile (64×64) load mappings — BM=64, BN=64, KTILE=32
/// A tile: [64, 32] = 2048 elements, 8 per thread
/// B tile: [32, 64] = 2048 elements, 8 per thread
/// For transposed B[N,K]: K is the fast dimension → row = flat%32 (fast)
const A_ROW_FWD: &str = "flat / 32u"; // M varies slowly (good for [M,K])
const A_COL_FWD: &str = "flat % 32u"; // K varies fast (coalesced in [M,K])
const A_ROW_AT: &str = "flat % 64u"; // M varies fast (coalesced in [K,M])
const A_COL_AT: &str = "flat / 64u"; // K varies slowly
const B_ROW_FWD: &str = "flat / 64u"; // K varies slowly (good for [K,N])
const B_COL_FWD: &str = "flat % 64u"; // N varies fast (coalesced in [K,N])
const B_ROW_BT: &str = "flat % 32u"; // K varies fast (coalesced in [N,K])
const B_COL_BT: &str = "flat / 32u"; // N varies slowly

// Small-tile (32×32) load mappings — BM=32, BN=32, KTILE=32
// A tile: [32, 32] = 1024 elements, 4 per thread
// B tile: [32, 32] = 1024 elements, 4 per thread
const A_ROW_FWD_S: &str = "flat / 32u"; // M slow, K fast
const A_COL_FWD_S: &str = "flat % 32u";
const A_ROW_AT_S: &str = "flat % 32u"; // M fast (coalesced for [K,M])
const A_COL_AT_S: &str = "flat / 32u";
const B_ROW_FWD_S: &str = "flat / 32u"; // K slow
const B_COL_FWD_S: &str = "flat % 32u"; // N fast
const B_ROW_BT_S: &str = "flat % 32u"; // K fast (coalesced for [N,K])
const B_COL_BT_S: &str = "flat / 32u";

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
    matmul_vars_epilogue(
        a_idx, b_idx, a_row, a_col, b_row, b_col, fused_decl, fused_expr, "", "",
    )
}

fn matmul_vars_epilogue(
    a_idx: &str,
    b_idx: &str,
    a_row: &str,
    a_col: &str,
    b_row: &str,
    b_col: &str,
    fused_decl: &str,
    fused_expr: &str,
    epilogue_decl: &str,
    epilogue_body: &str,
) -> ShaderModule {
    let src = include_str!("shaders/matmul.wgsl");
    let full_decl = if epilogue_decl.is_empty() {
        fused_decl.to_string()
    } else {
        format!("{}\n{}", fused_decl, epilogue_decl)
    };
    // When there's an epilogue, use var val + epilogue + store.
    // When there's no epilogue, use direct store (avoids SPIR-V regression
    // from the extra variable assignment).
    let store_body = if epilogue_body.is_empty() {
        format!("matrix_c[idx] = s[i][j]{};", fused_expr)
    } else {
        format!(
            "var val = s[i][j]{};\n                {}\n                matrix_c[idx] = val;",
            fused_expr, epilogue_body
        )
    };
    let src = preprocess(
        src,
        &[
            ("$A_INDEX", a_idx),
            ("$B_INDEX", b_idx),
            ("$A_ROW", a_row),
            ("$A_COL", a_col),
            ("$B_ROW", b_row),
            ("$B_COL", b_col),
            ("$FUSED_ADD_DECL", &full_decl),
            ("$STORE_BODY", &store_body),
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
    let store_body = format!("matrix_c[idx] = s[i][j]{};", fused_expr);
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
            ("$STORE_BODY", &store_body),
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
            ("$A_TRANSFORM_0", ""),
            ("$A_TRANSFORM_1", ""),
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
/// Default attention template vars for no-RoPE variants.
const ATTN_NO_ROPE: &[(&str, &str)] = &[
    ("$ROPE_DECL", ""),
    ("$ROPE_Q_APPLY", ""),
    ("$Q_VAL_EXPR", "q_raw"),
    ("$K_VAL_EXPR", "src_b[k_base + tid]"),
    ("$K_VAL_TAIL_EXPR", "src_b[k_base + tid]"),
];

fn gen_causal_attention() -> ShaderModule {
    let src = include_str!("shaders/attention.wgsl");
    let mut vars = vec![
        (
            "$PARAM_FIELDS",
            "seq: u32, num_heads: u32, num_kv_heads: u32, head_dim: u32,",
        ),
        (
            "$PARSE_PARAMS",
            "let q_seq = params.seq;\n    let num_heads = params.num_heads;\n    let num_kv_heads = params.num_kv_heads;\n    let head_dim = params.head_dim;\n    let kv_len = pos + 1u;",
        ),
        ("$KV_START", "0u"),
    ];
    vars.extend_from_slice(ATTN_NO_ROPE);
    let src = preprocess(src, &vars);
    parse_wgsl(&src)
}

// ---------------------------------------------------------------------------
// sliding_window_attention: same as causal but with bounded window
// ---------------------------------------------------------------------------

fn gen_sliding_window_attention() -> ShaderModule {
    let src = include_str!("shaders/sliding_window_attention.wgsl");
    let src = preprocess(
        src,
        &[
            (
                "$PARAM_FIELDS",
                "seq: u32, num_heads: u32, num_kv_heads: u32, head_dim: u32, window_size: u32, _pad0: u32, _pad1: u32, _pad2: u32,",
            ),
            (
                "$PARSE_PARAMS",
                "let q_seq = params.seq;\n    let num_heads = params.num_heads;\n    let num_kv_heads = params.num_kv_heads;\n    let head_dim = params.head_dim;\n    let window_size = params.window_size;\n    let kv_start = select(0u, pos + 1u - window_size, pos >= window_size);\n    let kv_len = pos + 1u;",
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
// full_attention.wgsl: non-causal multi-head attention with GQA
// Same as causal_attention but attends to all positions (no mask).
// ---------------------------------------------------------------------------

fn gen_full_attention() -> ShaderModule {
    let src = include_str!("shaders/attention.wgsl");
    let mut vars = vec![
        (
            "$PARAM_FIELDS",
            "seq: u32, num_heads: u32, num_kv_heads: u32, head_dim: u32,",
        ),
        (
            "$PARSE_PARAMS",
            "let q_seq = params.seq;\n    let num_heads = params.num_heads;\n    let num_kv_heads = params.num_kv_heads;\n    let head_dim = params.head_dim;\n    let kv_len = q_seq;",
        ),
        ("$KV_START", "0u"),
    ];
    vars.extend_from_slice(ATTN_NO_ROPE);
    let src = preprocess(src, &vars);
    parse_wgsl(&src)
}

fn gen_cross_attention() -> ShaderModule {
    let src = include_str!("shaders/attention.wgsl");
    let mut vars = vec![
        (
            "$PARAM_FIELDS",
            "q_seq: u32, kv_seq: u32, packed_heads: u32, head_dim: u32,",
        ),
        (
            "$PARSE_PARAMS",
            "let q_seq = params.q_seq;\n    let num_heads = params.packed_heads >> 16u;\n    let num_kv_heads = params.packed_heads & 0xFFFFu;\n    let head_dim = params.head_dim;\n    let kv_len = params.kv_seq;",
        ),
        ("$KV_START", "0u"),
    ];
    vars.extend_from_slice(ATTN_NO_ROPE);
    let src = preprocess(src, &vars);
    parse_wgsl(&src)
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

#[allow(dead_code)]
fn gen_fused_rms_norm_matmul_coop() -> ShaderModule {
    let default_config = CoopConfig {
        tile_size: 16,
        use_f16_input: true,
    };
    gen_fused_rms_norm_matmul_coop_wgsl(&default_config)
}

fn gen_fused_rms_norm_matmul_coop_wgsl(config: &CoopConfig) -> ShaderModule {
    // Use the standalone matmul_rms_norm_coop.wgsl which has:
    // - 64-thread cooperative rsqrt prologue (tree reduction)
    // - on-the-fly normalization during A-staging
    // - FourBufData-compatible globals (src_a, src_b, bias, dst)
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

    let src = include_str!("shaders/matmul_rms_norm_coop.wgsl");
    let src = preprocess(
        src,
        &[
            ("$ENABLE_F16", enable_f16),
            ("$ELEM_TYPE", elem_type),
            ("$ELEM_ZERO", elem_zero),
            ("$SHARED_SIZE", &shared_size.to_string()),
            ("$OUTPUT_TILE_U", &format!("{}u", output_tile)),
            ("$TILE_SIZE_U", &format!("{}u", tile)),
            ("$TILE_MASK_U", &format!("{}u", tile_mask)),
            ("$TILE_SHIFT_U", &format!("{}u", tile_shift)),
            ("$STAGING_ITERS_U", &format!("{}u", staging_iters)),
            ("$ROW_STRIDE_U", &format!("{}u", row_stride)),
            ("$CAST_OPEN", cast_open),
            ("$CAST_CLOSE", cast_close),
            ("$COOP_AB", &coop_ab),
            ("$COOP_BA", &coop_ba),
            ("$COOP_OUT", &coop_c),
        ],
    );
    parse_wgsl(&src)
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
            (ShaderGroup::MatMulGemv, naga::valid::Capabilities::empty()),
            (
                ShaderGroup::MatMulGemvAdd,
                naga::valid::Capabilities::empty(),
            ),
            (
                ShaderGroup::MatMulGemvBT,
                naga::valid::Capabilities::empty(),
            ),
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
            (ShaderGroup::RoPEGrad, naga::valid::Capabilities::empty()),
            (
                ShaderGroup::CausalAttention,
                naga::valid::Capabilities::empty(),
            ),
            (
                ShaderGroup::SlidingWindowAttention,
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
            (
                ShaderGroup::GlobalAvgPoolGrad,
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
    fn test_rope_grad_wgsl() {
        let _ = generate_wgsl(ShaderGroup::RoPEGrad);
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
            (ShaderGroup::RoPEGrad, empty),
            (ShaderGroup::CausalAttention, empty),
            (ShaderGroup::SlidingWindowAttention, empty),
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
            (ShaderGroup::GlobalAvgPoolGrad, empty),
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
                ShaderEntry::MatMul
                | ShaderEntry::MatMulAT
                | ShaderEntry::MatMulBT
                | ShaderEntry::MatMulGemv
                | ShaderEntry::MatMulGemvBT => {
                    vec!["matrix_a", "matrix_b", "matrix_c", "params"]
                }
                ShaderEntry::MatMulGemvAdd => {
                    vec!["matrix_a", "matrix_b", "matrix_c", "src", "params"]
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
                | ShaderEntry::Tanh
                | ShaderEntry::SumAll
                | ShaderEntry::MeanAll
                | ShaderEntry::SumRows
                | ShaderEntry::RoPE
                | ShaderEntry::RoPEGrad => vec!["src", "dst", "params"],
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
                | ShaderEntry::CausalAttentionRoPE
                | ShaderEntry::FullAttention
                | ShaderEntry::CrossAttention => {
                    vec!["src_a", "src_b", "bias", "dst", "lse", "params"]
                }
                ShaderEntry::SlidingWindowAttention => {
                    vec!["src_a", "src_b", "bias", "dst", "lse", "params"]
                }
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
                ShaderEntry::LayerNormGradWB | ShaderEntry::LayerNormGradX => {
                    vec!["src_a", "src_b", "bias", "dst", "params"]
                }
                ShaderEntry::RmsNormRsqrt => vec!["src", "dst", "params"],
                ShaderEntry::FusedRmsNormMatMul => {
                    vec!["src_a", "src_b", "bias", "dst", "params"]
                }
                ShaderEntry::CacheWrite => vec!["src", "dst", "kv_pos_buf", "params"],
                ShaderEntry::CachedAttention => {
                    vec!["src_a", "src_b", "bias", "kv_pos_buf", "dst", "params"]
                }
                ShaderEntry::GroupNorm | ShaderEntry::GroupNormSilu => {
                    vec!["src", "src_b", "bias", "dst", "params"]
                }
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
                ShaderEntry::Conv2dGradWeight
                | ShaderEntry::Conv2dGradWeightGemm
                | ShaderEntry::Conv2dGradWeightGemmSmall => {
                    vec!["grad_out", "src", "dst", "params"]
                }
                ShaderEntry::RoPEDynamic => vec!["src", "dst", "pos_offset_buf", "params"],
                ShaderEntry::MaxPool2d
                | ShaderEntry::GlobalAvgPool
                | ShaderEntry::GlobalAvgPoolGrad => vec!["src", "dst", "params"],
            }
        }

        let entries = [
            ShaderEntry::MatMul,
            ShaderEntry::MatMulAT,
            ShaderEntry::MatMulBT,
            ShaderEntry::MatMulGemv,
            ShaderEntry::MatMulGemvAdd,
            ShaderEntry::MatMulGemvBT,
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
            ShaderEntry::RoPEGrad,
            ShaderEntry::CausalAttention,
            ShaderEntry::CausalAttentionRoPE,
            ShaderEntry::SlidingWindowAttention,
            ShaderEntry::Gelu,
            ShaderEntry::Tanh,
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
            ShaderEntry::LayerNormGradWB,
            ShaderEntry::LayerNormGradX,
            ShaderEntry::RmsNormRsqrt,
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
            ShaderEntry::MaxPool2d,
            ShaderEntry::GlobalAvgPool,
            ShaderEntry::GlobalAvgPoolGrad,
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
