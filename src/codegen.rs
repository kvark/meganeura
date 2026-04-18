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

/// Generate WGSL for a [`MatMulEpilogue`] — the PointwiseDAG-based
/// replacement for `epilogue_to_wgsl`. Returns (declarations, body).
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

/// Generate WGSL for a [`MatMulPrologue`] — the multiplicative factors
/// applied during A-tile staging in the coop matmul.
///
/// Returns `(declarations, transform_expression)` where:
///   - `declarations` = `var<storage>` lines for prologue buffers.
///   - `transform_expression` = expression suffix like `* buf_0[gr] * buf_1[tc]`.
pub fn matmul_prologue_to_wgsl(prologue: &crate::compile::MatMulPrologue) -> (String, String) {
    use crate::compile::PrologueLoadKind;

    let mut decls = Vec::new();
    let mut expr = String::new();
    #[allow(clippy::pattern_type_mismatch)]
    for (i, (_, kind)) in prologue.factors.iter().enumerate() {
        let name = format!("prologue_buf_{}", i);
        decls.push(format!("var<storage> {}: array<f32>;", name));
        let idx = match *kind {
            PrologueLoadKind::PerRow => "gr",
            PrologueLoadKind::PerKCol => "tc",
        };
        expr.push_str(&format!(" * {}[{}]", name, idx));
    }
    (decls.join("\n"), expr)
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
    generate_matmul_with_epilogue_wgsl(group, &epi_decl, &epi_body)
}

/// Same as above but from a [`MatMulEpilogue`] (PointwiseDAG-based).
pub fn generate_matmul_with_dag_epilogue(
    group: ShaderGroup,
    epilogue: &crate::compile::MatMulEpilogue,
) -> ShaderModule {
    let (epi_decl, epi_body) = matmul_epilogue_to_wgsl(epilogue);
    generate_matmul_with_epilogue_wgsl(group, &epi_decl, &epi_body)
}

fn generate_matmul_with_epilogue_wgsl(
    group: ShaderGroup,
    epi_decl: &str,
    epi_body: &str,
) -> ShaderModule {
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
            epi_decl,
            epi_body,
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
            epi_decl,
            epi_body,
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
            epi_decl,
            epi_body,
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
            epi_decl,
            epi_body,
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
            epi_decl,
            epi_body,
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
            epi_decl,
            epi_body,
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
    LayerNorm,
    MultiHeadAttn,
    /// Flash Attention 2 forward: BQ>1 multi-query tiling with shared K staging.
    FlashAttention,
    MultiHeadAttnGradQ,
    FlashGradQ,
    MultiHeadAttnGradK,
    MultiHeadAttnGradKV,
    FlashGradKV,
    MultiHeadAttnGradV,
    SwiGLUGrad,
    SwiGLUConcat,
    SumRows,
    RmsNormGrad,
    RmsNormGradWRowPar,
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
    Conv2dGemmCoop,
    Conv2dGradInput,
    Conv2dGradInputGemm,
    Conv2dGradInputGemmSmall,
    Conv2dGradInputGemmCoop,
    Conv2dGradInputGemmCoop3x3,
    GroupNormSilu,
    WinogradInputTransform,
    WinogradOutputTransform,
    WinogradBatchedMatMul,
    WinogradBatchedMatMulSmall,
    WinogradWeightTransform,
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
        ShaderGroup::LayerNorm => parse_wgsl(include_str!("shaders/layer_norm.wgsl")),
        ShaderGroup::MultiHeadAttn => {
            // Default head_dim=64 fallback; runtime calls
            // generate_attention_module(head_dim) directly for the actual value.
            generate_attention_module(64)
        }
        ShaderGroup::FlashAttention => {
            // Default head_dim=64 fallback; runtime calls
            // generate_flash_attention_module(head_dim) directly.
            generate_flash_attention_module(64)
        }
        ShaderGroup::MultiHeadAttnGradQ => parse_wgsl(include_str!("shaders/mha_grad_q.wgsl")),
        ShaderGroup::FlashGradQ => generate_flash_grad_q_module(64),
        ShaderGroup::MultiHeadAttnGradK => parse_wgsl(include_str!("shaders/mha_grad_k.wgsl")),
        ShaderGroup::MultiHeadAttnGradKV => parse_wgsl(include_str!("shaders/mha_grad_kv.wgsl")),
        ShaderGroup::FlashGradKV => generate_flash_grad_kv_module(64),
        ShaderGroup::MultiHeadAttnGradV => parse_wgsl(include_str!("shaders/mha_grad_v.wgsl")),
        ShaderGroup::SwiGLUGrad => parse_wgsl(include_str!("shaders/swiglu_grad.wgsl")),
        ShaderGroup::SwiGLUConcat => parse_wgsl(include_str!("shaders/swiglu_concat.wgsl")),
        ShaderGroup::SumRows => parse_wgsl(include_str!("shaders/sum_rows.wgsl")),
        ShaderGroup::RmsNormGrad => parse_wgsl(include_str!("shaders/rms_norm_grad.wgsl")),
        ShaderGroup::RmsNormGradWRowPar => {
            parse_wgsl(include_str!("shaders/rms_norm_grad_w_rowpar.wgsl"))
        }
        ShaderGroup::LayerNormGrad => parse_wgsl(include_str!("shaders/layer_norm_grad.wgsl")),
        ShaderGroup::FusedRmsNormMatMul => parse_wgsl(include_str!("shaders/matmul_rms_norm.wgsl")),
        ShaderGroup::RmsNormRsqrt => parse_wgsl(include_str!("shaders/rms_norm_rsqrt.wgsl")),
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
        ShaderGroup::Conv2dGemmCoop => gen_conv2d_gemm_coop(),
        ShaderGroup::Conv2dGradInputGemmCoop => gen_conv2d_grad_input_gemm_coop(),
        ShaderGroup::Conv2dGradInputGemmCoop3x3 => gen_conv2d_grad_input_gemm_coop_3x3(),
        ShaderGroup::GroupNormSilu => parse_wgsl(include_str!("shaders/group_norm_silu.wgsl")),
        ShaderGroup::WinogradInputTransform => {
            parse_wgsl(include_str!("shaders/winograd_input_transform.wgsl"))
        }
        ShaderGroup::WinogradOutputTransform => {
            parse_wgsl(include_str!("shaders/winograd_output_transform.wgsl"))
        }
        ShaderGroup::WinogradBatchedMatMul | ShaderGroup::WinogradBatchedMatMulSmall => {
            parse_wgsl(include_str!("shaders/winograd_matmul.wgsl"))
        }
        ShaderGroup::WinogradWeightTransform => {
            parse_wgsl(include_str!("shaders/winograd_weight_transform.wgsl"))
        }
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
        ShaderGroup::Conv2dGemmCoop => gen_conv2d_gemm_coop_wgsl(config),
        ShaderGroup::Conv2dGradInputGemmCoop => gen_conv2d_grad_input_gemm_coop_wgsl(config),
        ShaderGroup::Conv2dGradInputGemmCoop3x3 => gen_conv2d_grad_input_gemm_coop_3x3_wgsl(config),
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
        | ShaderGroup::Conv2dGemmCoop
        | ShaderGroup::Conv2dGradInputGemmCoop
        | ShaderGroup::Conv2dGradInputGemmCoop3x3
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
    gen_matmul_coop_wgsl_prologue(fused_add, variant, config, None)
}

/// Generate coop matmul with an optional [`MatMulPrologue`].
pub fn gen_matmul_coop_with_prologue(
    fused_add: bool,
    variant: MatMulCoopVariant,
    config: &CoopConfig,
    prologue: &crate::compile::MatMulPrologue,
) -> ShaderModule {
    gen_matmul_coop_wgsl_prologue(fused_add, variant, config, Some(prologue))
}

fn gen_matmul_coop_wgsl_prologue(
    fused_add: bool,
    variant: MatMulCoopVariant,
    config: &CoopConfig,
    prologue: Option<&crate::compile::MatMulPrologue>,
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

    let (prologue_decl, a_transform) = match prologue {
        Some(p) => matmul_prologue_to_wgsl(p),
        None => (String::new(), String::new()),
    };

    // Vec4 staging: use 128-bit loads when tile is 16+ (64 threads × 4 = 256 = 16×16).
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
    let b_stage_0;
    let b_stage_1;
    if vec4_b {
        let gen_vec4_b = |shared: &str, col_offset: &str| -> String {
            format!(
                "{{\
               \n            let tr = t + v4_row;\
               \n            let cc4 = {col} + v4_col;\
               \n            let flat = v4_row * {t}u + v4_col;\
               \n            if tr < k && (cc4 + 4u) <= n {{\
               \n                let v = matrix_b[(tr * n + cc4) >> 2u];\
               \n                {s}[flat] = {co}v.x{cc};\
               \n                {s}[flat + 1u] = {co}v.y{cc};\
               \n                {s}[flat + 2u] = {co}v.z{cc};\
               \n                {s}[flat + 3u] = {co}v.w{cc};\
               \n            }} else {{\
               \n                let z = {z};\
               \n                {s}[flat] = z;\
               \n                {s}[flat + 1u] = z;\
               \n                {s}[flat + 2u] = z;\
               \n                {s}[flat + 3u] = z;\
               \n            }}\
               \n        }}",
                col = col_offset,
                t = tile,
                s = shared,
                co = cast_open,
                cc = cast_close,
                z = elem_zero,
            )
        };
        b_stage_0 = gen_vec4_b("shared_a0", "tile_col");
        b_stage_1 = gen_vec4_b("shared_a1", &format!("(tile_col + {}u)", tile));
    } else if vec4_b_transposed {
        // BT: B[N,K], load vec4 along K (consecutive in memory), write transposed to shared.
        // v4_row → N (cc direction, shared col), v4_col → K (tr direction, shared rows).
        // shared[row * tile + col] where row = K-offset, col = N-offset.
        let gen_vec4_bt = |shared: &str, col_offset: &str| -> String {
            format!(
                "{{\
               \n            let cc = {col} + v4_row;\
               \n            let tr4 = t + v4_col;\
               \n            if cc < n && (tr4 + 4u) <= k {{\
               \n                let v = matrix_b[(cc * k + tr4) >> 2u];\
               \n                {s}[v4_col * {t}u + v4_row] = {co}v.x{cc2};\
               \n                {s}[(v4_col + 1u) * {t}u + v4_row] = {co}v.y{cc2};\
               \n                {s}[(v4_col + 2u) * {t}u + v4_row] = {co}v.z{cc2};\
               \n                {s}[(v4_col + 3u) * {t}u + v4_row] = {co}v.w{cc2};\
               \n            }} else {{\
               \n                let z = {z};\
               \n                {s}[v4_col * {t}u + v4_row] = z;\
               \n                {s}[(v4_col + 1u) * {t}u + v4_row] = z;\
               \n                {s}[(v4_col + 2u) * {t}u + v4_row] = z;\
               \n                {s}[(v4_col + 3u) * {t}u + v4_row] = z;\
               \n            }}\
               \n        }}",
                col = col_offset,
                t = tile,
                s = shared,
                co = cast_open,
                cc2 = cast_close,
                z = elem_zero,
            )
        };
        b_stage_0 = gen_vec4_bt("shared_a0", "tile_col");
        b_stage_1 = gen_vec4_bt("shared_a1", &format!("(tile_col + {}u)", tile));
    } else {
        // Scalar B staging (tile_size < 16 fallback)
        let (bi0, bi1) = match variant {
            MatMulCoopVariant::Normal | MatMulCoopVariant::AT => ("tr * n + cc", "tr * n + cc1"),
            MatMulCoopVariant::BT => ("cc * k + tr", "cc1 * k + tr"),
        };
        let gen_scalar_b = |shared: &str, in_col: &str, b_index: &str| -> String {
            format!(
                "{{\
               \n            let zero_val = {z};\
               \n            for (var e = 0u; e < {iters}u; e++) {{\
               \n                let flat = lid.x + e * 64u;\
               \n                let tr = t + base_row + e * {stride}u;\
               \n                let in_bounds = (tr < k) && {ic};\
               \n                if in_bounds {{\
               \n                    {s}[flat] = {co}matrix_b[{bi}]{cc};\
               \n                }} else {{\
               \n                    {s}[flat] = zero_val;\
               \n                }}\
               \n            }}\
               \n        }}",
                z = elem_zero,
                iters = staging_iters,
                stride = row_stride,
                ic = in_col,
                s = shared,
                co = cast_open,
                cc = cast_close,
                bi = b_index,
            )
        };
        b_stage_0 = gen_scalar_b("shared_a0", "in_n", bi0);
        b_stage_1 = gen_scalar_b("shared_a1", "in_n1", bi1);
    }

    // Generate A staging blocks (shared_b0, shared_b1)
    let a_stage_0;
    let a_stage_1;
    if vec4_a {
        let gen_vec4_a = |shared: &str, row_offset: &str| -> String {
            format!(
                "{{\
               \n            let gr = {row} + v4_row;\
               \n            let tc4 = t + v4_col;\
               \n            let flat = v4_row * {t}u + v4_col;\
               \n            if gr < m && (tc4 + 4u) <= k {{\
               \n                let v = matrix_a[(gr * k + tc4) >> 2u];\
               \n                {s}[flat] = {co}v.x{cc};\
               \n                {s}[flat + 1u] = {co}v.y{cc};\
               \n                {s}[flat + 2u] = {co}v.z{cc};\
               \n                {s}[flat + 3u] = {co}v.w{cc};\
               \n            }} else {{\
               \n                let z = {z};\
               \n                {s}[flat] = z;\
               \n                {s}[flat + 1u] = z;\
               \n                {s}[flat + 2u] = z;\
               \n                {s}[flat + 3u] = z;\
               \n            }}\
               \n        }}",
                row = row_offset,
                t = tile,
                s = shared,
                co = cast_open,
                cc = cast_close,
                z = elem_zero,
            )
        };
        a_stage_0 = gen_vec4_a("shared_b0", "tile_row");
        a_stage_1 = gen_vec4_a("shared_b1", &format!("(tile_row + {}u)", tile));
    } else if vec4_a_transposed {
        // AT: A[K,M], load vec4 along M (consecutive in memory), write transposed to shared.
        // v4_row → K (tc direction, shared col), v4_col → M (gr direction, shared rows).
        // shared[row * tile + col] where row = M-offset, col = K-offset.
        let gen_vec4_at = |shared: &str, row_offset: &str| -> String {
            format!(
                "{{\
               \n            let tc = t + v4_row;\
               \n            let gr4 = {row} + v4_col;\
               \n            if tc < k && (gr4 + 4u) <= m {{\
               \n                let v = matrix_a[(tc * m + gr4) >> 2u];\
               \n                {s}[v4_col * {t}u + v4_row] = {co}v.x{cc};\
               \n                {s}[(v4_col + 1u) * {t}u + v4_row] = {co}v.y{cc};\
               \n                {s}[(v4_col + 2u) * {t}u + v4_row] = {co}v.z{cc};\
               \n                {s}[(v4_col + 3u) * {t}u + v4_row] = {co}v.w{cc};\
               \n            }} else {{\
               \n                let z = {z};\
               \n                {s}[v4_col * {t}u + v4_row] = z;\
               \n                {s}[(v4_col + 1u) * {t}u + v4_row] = z;\
               \n                {s}[(v4_col + 2u) * {t}u + v4_row] = z;\
               \n                {s}[(v4_col + 3u) * {t}u + v4_row] = z;\
               \n            }}\
               \n        }}",
                row = row_offset,
                t = tile,
                s = shared,
                co = cast_open,
                cc = cast_close,
                z = elem_zero,
            )
        };
        a_stage_0 = gen_vec4_at("shared_b0", "tile_row");
        a_stage_1 = gen_vec4_at("shared_b1", &format!("(tile_row + {}u)", tile));
    } else {
        // Scalar A staging (prologue or tile_size < 16)
        let a_idx = match variant {
            MatMulCoopVariant::Normal | MatMulCoopVariant::BT => "gr * k + tc",
            MatMulCoopVariant::AT => "tc * m + gr",
        };
        let gen_scalar_a = |shared: &str, row_offset: &str| -> String {
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
               \n                    {s}[flat] = {co}a_val{xf}{cc};\
               \n                }} else {{\
               \n                    {s}[flat] = {z};\
               \n                }}\
               \n            }}\
               \n        }}",
                iters = staging_iters,
                stride = row_stride,
                row = row_offset,
                ai = a_idx,
                s = shared,
                co = cast_open,
                cc = cast_close,
                z = elem_zero,
                xf = a_transform,
            )
        };
        a_stage_0 = gen_scalar_a("shared_b0", "tile_row");
        a_stage_1 = gen_scalar_a("shared_b1", &format!("(tile_row + {}u)", tile));
    }

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
    let shared_size_s = format!("{}", shared_size);

    let src = include_str!("shaders/matmul_coop.wgsl");
    let src = preprocess(
        src,
        &[
            ("$ENABLE_F16", enable_f16),
            ("$ELEM_TYPE", elem_type),
            ("$SHARED_SIZE", &shared_size_s),
            ("$OUTPUT_TILE_U", &output_tile_u),
            ("$TILE_SIZE_U", &tile_size_u),
            ("$COOP_AB", &coop_ab),
            ("$COOP_BA", &coop_ba),
            ("$A_STORAGE", a_storage),
            ("$B_STORAGE", b_storage),
            ("$STAGING_VARS", &staging_vars),
            ("$B_STAGE_0", &b_stage_0),
            ("$B_STAGE_1", &b_stage_1),
            ("$A_STAGE_0", &a_stage_0),
            ("$A_STAGE_1", &a_stage_1),
            ("$PROLOGUE_DECL", &prologue_decl),
            ("$FUSED_ADD_DECL", &fused_decl),
            ("$ACC_INIT", &acc_init),
        ],
    );
    parse_wgsl(&src)
}

/// Variant selector for gen_matmul_coop_inner.
#[derive(Clone, Copy, PartialEq)]
pub enum MatMulCoopVariant {
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
// Unified attention: BKV=8 tiled, runtime causal detection, parameterized head_dim.
// Replaces gen_causal_attention, gen_full_attention, gen_cross_attention, and
// the schedule-template `lower_attention` for MultiHeadAttn.
//
// Param layout matches MultiHeadAttnData (AttentionParams):
//   params = [q_seq, kv_seq, packed_heads, head_dim, window_size, ...]
//   kv_seq == 0 → causal (kv_len = pos + 1)
//   kv_seq >  0 → non-causal (kv_len = kv_seq)
//   window_size > 0 → sliding window (kv_start = max(0, pos+1-window))
// ---------------------------------------------------------------------------

/// Generate a BKV=8 tiled attention shader parameterized by `head_dim`.
///
/// The shader uses online softmax with 8-way KV tiling to reduce
/// workgroup barriers by 8× compared to the un-tiled archetype.
/// Runtime causal detection via `kv_seq == 0` avoids separate shaders
/// for causal vs non-causal masks.
pub fn generate_attention_module(head_dim: u32) -> ShaderModule {
    use std::fmt::Write;
    assert!(
        head_dim.is_power_of_two() && head_dim >= 2,
        "attention head_dim must be a power of 2 ≥ 2, got {head_dim}"
    );

    let hd = head_dim;
    let bkv: u32 = 8;
    let mut src = String::new();

    // Params struct (matches AttentionParams, 8 u32 = 32 bytes)
    src.push_str(
        "struct Params {\n    q_seq: u32,\n    kv_seq: u32,\n    packed_heads: u32,\n    head_dim: u32,\n    window_size: u32,\n    _pad0: u32,\n    _pad1: u32,\n    _pad2: u32,\n}\n\n",
    );
    src.push_str("var<storage> src_a: array<f32>;\n"); // Q
    src.push_str("var<storage> src_b: array<f32>;\n"); // K
    src.push_str("var<storage> bias: array<f32>;\n"); // V
    src.push_str("var<storage, read_write> dst: array<f32>;\n"); // O
    src.push_str("var<storage, read_write> lse: array<f32>;\n"); // LSE
    src.push_str("var<uniform> params: Params;\n\n");

    // Shared memory: BKV * head_dim for tiled scores, head_dim for tail
    let _ = writeln!(src, "var<workgroup> wg_scores: array<f32, {}>;\n", bkv * hd);
    let _ = writeln!(src, "var<workgroup> wg_dot: array<f32, {}>;\n", hd);

    // tree_reduce_8: reduce BKV=8 dot products simultaneously
    src.push_str("fn tree_reduce_8(tid: u32) {\n");
    let mut stride = hd / 2;
    while stride > 0 {
        src.push_str("    workgroupBarrier();\n");
        let _ = writeln!(src, "    if tid < {stride}u {{");
        let _ = writeln!(src, "        for (var i = 0u; i < {bkv}u; i++) {{");
        let _ = writeln!(
            src,
            "            wg_scores[i * {hd}u + tid] += wg_scores[i * {hd}u + tid + {stride}u];"
        );
        src.push_str("        }\n    }\n");
        stride /= 2;
    }
    src.push_str("    workgroupBarrier();\n}\n\n");

    // tree_reduce: single dot product for tail
    src.push_str("fn tree_reduce(tid: u32) {\n");
    stride = hd / 2;
    while stride > 0 {
        src.push_str("    workgroupBarrier();\n");
        let _ = writeln!(
            src,
            "    if tid < {stride}u {{ wg_dot[tid] += wg_dot[tid + {stride}u]; }}"
        );
        stride /= 2;
    }
    src.push_str("    workgroupBarrier();\n}\n\n");

    // Main kernel
    let _ = writeln!(src, "@compute @workgroup_size({hd})");
    src.push_str(
        "fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {\n",
    );
    src.push_str("    let pos = wgid.x;\n");
    src.push_str("    let head = wgid.y;\n");
    src.push_str("    let tid = lid.x;\n");
    src.push_str("    let q_seq = params.q_seq;\n");
    src.push_str("    let kv_seq = params.kv_seq;\n");
    src.push_str("    let num_heads = params.packed_heads >> 16u;\n");
    src.push_str("    let num_kv_heads = params.packed_heads & 0xFFFFu;\n");
    src.push_str("    let head_dim = params.head_dim;\n");
    src.push_str("    if pos >= q_seq || head >= num_heads { return; }\n\n");

    // Runtime causal detection: kv_seq=0 means causal (kv_len = pos + 1)
    src.push_str("    let kv_len = select(kv_seq, pos + 1u, kv_seq == 0u);\n");
    // Sliding window: window_size>0 limits how far back we attend
    src.push_str("    let window_size = params.window_size;\n");
    src.push_str(
        "    let kv_start = select(0u, kv_len - min(kv_len, window_size), window_size > 0u);\n\n",
    );

    // GQA head mapping
    src.push_str("    let kv_head = head / (num_heads / num_kv_heads);\n");
    src.push_str("    let kv_head_off = kv_head * head_dim;\n");
    src.push_str("    let kv_dim = num_kv_heads * head_dim;\n");
    src.push_str("    let scale = inverseSqrt(f32(head_dim));\n");
    src.push_str("    let q_base = pos * (num_heads * head_dim) + head * head_dim;\n");
    src.push_str("    let q_val = src_a[q_base + tid];\n\n");

    // Online softmax accumulators
    src.push_str("    var my_out = 0.0;\n");
    src.push_str("    var max_score = -1e30;\n");
    src.push_str("    var sum_exp = 0.0;\n\n");

    // --- Tiled KV loop: process BKV positions per reduction ---
    let _ = writeln!(src, "    let kv_range = kv_len - kv_start;");
    let _ = writeln!(
        src,
        "    let tile_end = kv_start + (kv_range / {bkv}u) * {bkv}u;"
    );
    src.push_str("    var t = kv_start;\n");
    let _ = writeln!(src, "    for (; t < tile_end; t += {bkv}u) {{");
    let _ = writeln!(src, "        for (var i = 0u; i < {bkv}u; i++) {{");
    src.push_str("            let k_base = (t + i) * kv_dim + kv_head_off;\n");
    let _ = writeln!(
        src,
        "            wg_scores[i * {hd}u + tid] = q_val * src_b[k_base + tid];"
    );
    src.push_str("        }\n");
    src.push_str("        tree_reduce_8(tid);\n\n");
    let _ = writeln!(src, "        for (var i = 0u; i < {bkv}u; i++) {{");
    let _ = writeln!(src, "            let score = wg_scores[i * {hd}u] * scale;");
    src.push_str("            let new_max = max(max_score, score);\n");
    src.push_str("            let correction = exp(max_score - new_max);\n");
    src.push_str("            let weight = exp(score - new_max);\n");
    src.push_str("            sum_exp = sum_exp * correction + weight;\n");
    src.push_str("            let v_base = (t + i) * kv_dim + kv_head_off;\n");
    src.push_str("            my_out = my_out * correction + weight * bias[v_base + tid];\n");
    src.push_str("            max_score = new_max;\n");
    src.push_str("        }\n");
    src.push_str("    }\n\n");

    // --- Tail: remaining KV positions one at a time ---
    src.push_str("    for (; t < kv_len; t++) {\n");
    src.push_str("        let k_base = t * kv_dim + kv_head_off;\n");
    src.push_str("        wg_dot[tid] = q_val * src_b[k_base + tid];\n");
    src.push_str("        tree_reduce(tid);\n");
    src.push_str("        let score = wg_dot[0] * scale;\n\n");
    src.push_str("        let new_max = max(max_score, score);\n");
    src.push_str("        let correction = exp(max_score - new_max);\n");
    src.push_str("        let weight = exp(score - new_max);\n");
    src.push_str("        sum_exp = sum_exp * correction + weight;\n");
    src.push_str("        my_out = my_out * correction + weight * bias[k_base + tid];\n");
    src.push_str("        max_score = new_max;\n");
    src.push_str("    }\n\n");

    // Final output
    src.push_str("    let safe_sum = select(sum_exp, 1.0, sum_exp == 0.0);\n");
    src.push_str("    dst[q_base + tid] = my_out / safe_sum;\n\n");

    // LSE output for backward pass
    src.push_str("    if tid == 0u {\n");
    src.push_str("        let idx = (pos * num_heads + head) * 2u;\n");
    src.push_str("        lse[idx] = max_score;\n");
    src.push_str("        lse[idx + 1u] = select(log(sum_exp), -1e30, sum_exp == 0.0);\n");
    src.push_str("    }\n");
    src.push_str("}\n");

    let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| {
        panic!(
            "generated unified attention WGSL failed to parse:\n{}\n---\n{}",
            e, src
        )
    });
    ShaderModule {
        module,
        source: src,
    }
}

// ---------------------------------------------------------------------------
// Flash Attention 2 forward: multiple query positions per workgroup.
//
// BQ = 256 / head_dim query positions per workgroup (e.g. 4 for hd=64).
// Each group of head_dim threads handles one query position. K tiles are
// staged in shared memory and reused across all BQ groups, reducing
// global memory reads by BQ×.
//
// Falls back to generate_attention_module (BQ=1) when head_dim > 128.
// ---------------------------------------------------------------------------

/// Generate a Flash Attention 2 forward kernel with BQ>1 multi-query tiling.
///
/// Flash Attention 2 forward with multi-query tiling and vectorized threads.
///
/// Each thread handles EPT (elements per thread) head_dim elements, reducing
/// the tree reduction depth from log2(head_dim) to log2(head_dim/EPT).
/// This dramatically cuts workgroup barriers while increasing per-thread
/// compute and register-based V accumulation.
///
/// For head_dim=64, EPT=8: TPQ=8 threads/query, BQ=32 queries/WG,
/// tree depth=3 (vs 6), workgroups reduced 8x.
pub fn generate_flash_attention_module(head_dim: u32) -> ShaderModule {
    use std::fmt::Write;
    assert!(
        head_dim.is_power_of_two() && head_dim >= 2,
        "attention head_dim must be a power of 2 ≥ 2, got {head_dim}"
    );

    let hd = head_dim;
    // Elements per thread: each thread handles EPT consecutive head_dim elements.
    // Choose EPT so that threads_per_query (TPQ) fits well in 256-thread WGs.
    // Each thread handles EPT consecutive head_dim elements, computing partial
    // dot products in registers. Higher EPT = fewer tree-reduce barriers but
    // more register pressure. EPT=hd eliminates all dot-product reductions.
    let ept: u32 = hd.min(64);
    let tpq = hd / ept; // threads per query
    let bq: u32 = (256 / tpq).max(1);
    // Fall back to BQ=1 kernel when multi-query isn't beneficial
    if bq <= 1 {
        return generate_attention_module(head_dim);
    }
    let wg_size = bq * tpq;
    let bkv: u32 = 8;
    let mut src = String::new();

    // Params struct (matches AttentionParams: 8 u32 = 32 bytes)
    src.push_str(
        "struct Params {\n    q_seq: u32,\n    kv_seq: u32,\n    packed_heads: u32,\n    head_dim: u32,\n    window_size: u32,\n    _pad0: u32,\n    _pad1: u32,\n    _pad2: u32,\n}\n\n",
    );
    src.push_str("var<storage> src_a: array<f32>;\n"); // Q
    src.push_str("var<storage> src_b: array<f32>;\n"); // K
    src.push_str("var<storage> bias: array<f32>;\n"); // V
    src.push_str("var<storage, read_write> dst: array<f32>;\n"); // O
    src.push_str("var<storage, read_write> lse: array<f32>;\n"); // LSE
    src.push_str("var<uniform> params: Params;\n\n");

    // Shared memory:
    //   shared_k: K tile [BKV, hd] loaded once, reused by BQ groups
    //   wg_scores: [BQ][BKV][TPQ] partial dot products for grouped reduction
    //   wg_dot: [BQ][TPQ] tail reduction
    let _ = writeln!(src, "var<workgroup> shared_k: array<f32, {}>;\n", bkv * hd);
    let _ = writeln!(
        src,
        "var<workgroup> wg_scores: array<f32, {}>;\n",
        bq * bkv * tpq
    );
    let _ = writeln!(src, "var<workgroup> wg_dot: array<f32, {}>;\n", bq * tpq);

    // Grouped tree_reduce for BKV scores: each group of TPQ threads reduces independently.
    src.push_str("fn tree_reduce_bkv_grouped(tid: u32) {\n");
    let _ = writeln!(src, "    let qi = tid / {tpq}u;");
    let _ = writeln!(src, "    let local = tid % {tpq}u;");
    let _ = writeln!(src, "    let base = qi * {}u;", bkv * tpq);
    let mut stride = tpq / 2;
    while stride > 0 {
        src.push_str("    workgroupBarrier();\n");
        let _ = writeln!(src, "    if local < {stride}u {{");
        let _ = writeln!(src, "        for (var i = 0u; i < {bkv}u; i++) {{");
        let _ = writeln!(
            src,
            "            wg_scores[base + i * {tpq}u + local] += wg_scores[base + i * {tpq}u + local + {stride}u];"
        );
        src.push_str("        }\n    }\n");
        stride /= 2;
    }
    src.push_str("    workgroupBarrier();\n}\n\n");

    // Grouped tree_reduce for tail (single dot product)
    src.push_str("fn tree_reduce_grouped(tid: u32) {\n");
    let _ = writeln!(src, "    let qi = tid / {tpq}u;");
    let _ = writeln!(src, "    let local = tid % {tpq}u;");
    let _ = writeln!(src, "    let base = qi * {tpq}u;");
    stride = tpq / 2;
    while stride > 0 {
        src.push_str("    workgroupBarrier();\n");
        let _ = writeln!(
            src,
            "    if local < {stride}u {{ wg_dot[base + local] += wg_dot[base + local + {stride}u]; }}"
        );
        stride /= 2;
    }
    src.push_str("    workgroupBarrier();\n}\n\n");

    // Main kernel
    let _ = writeln!(src, "@compute @workgroup_size({wg_size})");
    src.push_str(
        "fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {\n",
    );
    let _ = writeln!(src, "    let qi = lid.x / {tpq}u;"); // query within tile
    let _ = writeln!(src, "    let lane = lid.x % {tpq}u;"); // lane within query group
    let _ = writeln!(src, "    let d_base = lane * {ept}u;"); // first head_dim element for this thread
    let _ = writeln!(src, "    let pos = wgid.x * {bq}u + qi;"); // global query position
    src.push_str("    let head = wgid.y;\n");
    src.push_str("    let q_seq = params.q_seq;\n");
    src.push_str("    let kv_seq = params.kv_seq;\n");
    src.push_str("    let num_heads = params.packed_heads >> 16u;\n");
    src.push_str("    let num_kv_heads = params.packed_heads & 0xFFFFu;\n");
    src.push_str("    let head_dim = params.head_dim;\n");
    src.push_str("    let valid = pos < q_seq && head < num_heads;\n\n");

    // Per-position KV range (causal + sliding window)
    src.push_str(
        "    let my_kv_len = select(kv_seq, select(pos + 1u, 0u, !valid), kv_seq == 0u);\n",
    );
    src.push_str("    let window_size = params.window_size;\n");
    src.push_str("    let my_kv_start = select(0u, my_kv_len - min(my_kv_len, window_size), window_size > 0u);\n\n");

    // Workgroup-wide loop bounds
    let _ = writeln!(
        src,
        "    let last_pos = min(wgid.x * {bq}u + {bq}u - 1u, q_seq - 1u);"
    );
    let _ = writeln!(src, "    let first_pos = wgid.x * {bq}u;");
    src.push_str("    let max_kv_len = select(kv_seq, last_pos + 1u, kv_seq == 0u);\n");
    src.push_str("    let first_kv_len = select(kv_seq, first_pos + 1u, kv_seq == 0u);\n");
    src.push_str("    let min_kv_start = select(0u, first_kv_len - min(first_kv_len, window_size), window_size > 0u);\n\n");

    // GQA head mapping
    src.push_str("    let kv_head = head / (num_heads / max(num_kv_heads, 1u));\n");
    src.push_str("    let kv_head_off = kv_head * head_dim;\n");
    src.push_str("    let kv_dim = num_kv_heads * head_dim;\n");
    src.push_str("    let scale = inverseSqrt(f32(head_dim));\n");

    // Load Q values for this thread's EPT elements (registers)
    for e in 0..ept {
        let _ = writeln!(src, "    var q{e} = 0.0;");
    }
    src.push_str("    if valid {\n");
    src.push_str("        let q_base = pos * (num_heads * head_dim) + head * head_dim;\n");
    for e in 0..ept {
        let _ = writeln!(src, "        q{e} = src_a[q_base + d_base + {e}u];");
    }
    src.push_str("    }\n\n");

    // Online softmax accumulators: EPT output elements per thread
    for e in 0..ept {
        let _ = writeln!(src, "    var out{e} = 0.0;");
    }
    src.push_str("    var max_score = -1e30;\n");
    src.push_str("    var sum_exp = 0.0;\n\n");

    // --- Tiled KV loop with shared K staging ---
    let _ = writeln!(src, "    let kv_range = max_kv_len - min_kv_start;");
    let _ = writeln!(
        src,
        "    let tile_end = min_kv_start + (kv_range / {bkv}u) * {bkv}u;"
    );
    src.push_str("    var t = min_kv_start;\n");
    let _ = writeln!(src, "    for (; t < tile_end; t += {bkv}u) {{");

    // Cooperatively load K tile into shared memory
    let k_tile_size = bkv * hd;
    let loads_per_thread = k_tile_size.div_ceil(wg_size);
    for l in 0..loads_per_thread {
        let offset = l * wg_size;
        if offset == 0 {
            let _ = writeln!(src, "        if lid.x < {k_tile_size}u {{");
            let _ = writeln!(src, "            let ki = lid.x / {hd}u;");
            src.push_str(
                "            shared_k[lid.x] = src_b[(t + ki) * kv_dim + kv_head_off + (lid.x % head_dim)];\n",
            );
            src.push_str("        }\n");
        } else {
            let _ = writeln!(src, "        if lid.x + {offset}u < {k_tile_size}u {{");
            let _ = writeln!(src, "            let ki2 = (lid.x + {offset}u) / {hd}u;");
            let _ = writeln!(
                src,
                "            shared_k[lid.x + {offset}u] = src_b[(t + ki2) * kv_dim + kv_head_off + ((lid.x + {offset}u) % head_dim)];"
            );
            src.push_str("        }\n");
        }
    }
    src.push_str("        workgroupBarrier();\n\n");

    // Each thread computes partial dot product (EPT elements) for BKV positions
    let _ = writeln!(src, "        let grp_base = qi * {}u;", bkv * tpq);
    let _ = writeln!(src, "        for (var i = 0u; i < {bkv}u; i++) {{");
    // Compute partial dot product across EPT elements
    src.push_str("            var pdot = 0.0;\n");
    for e in 0..ept {
        let _ = writeln!(
            src,
            "            pdot += q{e} * shared_k[i * {hd}u + d_base + {e}u];"
        );
    }
    let _ = writeln!(
        src,
        "            wg_scores[grp_base + i * {tpq}u + lane] = pdot;"
    );
    src.push_str("        }\n");
    src.push_str("        tree_reduce_bkv_grouped(lid.x);\n\n");

    // Online softmax + V accumulation for BKV positions
    let _ = writeln!(src, "        for (var i = 0u; i < {bkv}u; i++) {{");
    src.push_str("            let kv_pos = t + i;\n");
    src.push_str("            if valid && kv_pos >= my_kv_start && kv_pos < my_kv_len {\n");
    let _ = writeln!(
        src,
        "                let score = wg_scores[grp_base + i * {tpq}u] * scale;"
    );
    src.push_str("                let new_max = max(max_score, score);\n");
    src.push_str("                let correction = exp(max_score - new_max);\n");
    src.push_str("                let weight = exp(score - new_max);\n");
    src.push_str("                sum_exp = sum_exp * correction + weight;\n");
    src.push_str("                let v_base = kv_pos * kv_dim + kv_head_off;\n");
    // Accumulate EPT V elements in registers
    for e in 0..ept {
        let _ = writeln!(
            src,
            "                out{e} = out{e} * correction + weight * bias[v_base + d_base + {e}u];"
        );
    }
    src.push_str("                max_score = new_max;\n");
    src.push_str("            }\n");
    src.push_str("        }\n");
    src.push_str("        workgroupBarrier();\n");
    src.push_str("    }\n\n");

    // --- Tail: remaining KV positions one at a time ---
    src.push_str("    for (; t < max_kv_len; t++) {\n");
    // Load single K position into shared_k
    let _ = writeln!(src, "        if lid.x < {hd}u {{");
    src.push_str("            shared_k[lid.x] = src_b[t * kv_dim + kv_head_off + lid.x];\n");
    src.push_str("        }\n");
    src.push_str("        workgroupBarrier();\n\n");

    // Each thread computes partial dot product
    let _ = writeln!(src, "        let dot_base = qi * {tpq}u;");
    src.push_str("        var pdot2 = 0.0;\n");
    for e in 0..ept {
        let _ = writeln!(src, "        pdot2 += q{e} * shared_k[d_base + {e}u];");
    }
    src.push_str("        wg_dot[dot_base + lane] = pdot2;\n");
    src.push_str("        tree_reduce_grouped(lid.x);\n");
    let _ = writeln!(src, "        let score = wg_dot[qi * {tpq}u] * scale;\n");

    src.push_str("        if valid && t >= my_kv_start && t < my_kv_len {\n");
    src.push_str("            let new_max = max(max_score, score);\n");
    src.push_str("            let correction = exp(max_score - new_max);\n");
    src.push_str("            let weight = exp(score - new_max);\n");
    src.push_str("            sum_exp = sum_exp * correction + weight;\n");
    src.push_str("            let v_base2 = t * kv_dim + kv_head_off;\n");
    for e in 0..ept {
        let _ = writeln!(
            src,
            "            out{e} = out{e} * correction + weight * bias[v_base2 + d_base + {e}u];"
        );
    }
    src.push_str("            max_score = new_max;\n");
    src.push_str("        }\n");
    src.push_str("        workgroupBarrier();\n");
    src.push_str("    }\n\n");

    // Final output + LSE
    src.push_str("    if valid {\n");
    src.push_str("        let q_base = pos * (num_heads * head_dim) + head * head_dim;\n");
    src.push_str("        let safe_sum = select(sum_exp, 1.0, sum_exp == 0.0);\n");
    for e in 0..ept {
        let _ = writeln!(
            src,
            "        dst[q_base + d_base + {e}u] = out{e} / safe_sum;"
        );
    }

    // LSE output: only first thread in each group
    src.push_str("        if lane == 0u {\n");
    src.push_str("            let idx = (pos * num_heads + head) * 2u;\n");
    src.push_str("            lse[idx] = max_score;\n");
    src.push_str("            lse[idx + 1u] = select(log(sum_exp), -1e30, sum_exp == 0.0);\n");
    src.push_str("        }\n");
    src.push_str("    }\n");
    src.push_str("}\n");

    let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| {
        panic!(
            "generated flash attention WGSL failed to parse:\n{}\n---\n{}",
            e, src
        )
    });
    ShaderModule {
        module,
        source: src,
    }
}

/// Generate a Flash Attention 2 backward dQ kernel using vectorized register
/// pattern (each thread computes full Q·K / dO·V dot products in registers).
///
/// Mirrors the forward kernel's EPT/TPQ/BQ pattern. When TPQ=1 (EPT==hd)
/// there are NO workgroup barriers inside the KV loop — each thread owns
/// one full query row and sequentially accumulates dQ by loading K/V
/// scalars directly from global memory.
pub fn generate_flash_grad_q_module(head_dim: u32) -> ShaderModule {
    use std::fmt::Write;
    assert!(head_dim.is_power_of_two() && head_dim >= 2);

    let hd = head_dim;
    let ept: u32 = hd.min(64);
    let tpq = hd / ept; // threads per query
    let bq: u32 = (256 / tpq).max(1);
    if bq <= 1 {
        // Fall back to hand-written shader
        return parse_wgsl(include_str!("shaders/mha_grad_q.wgsl"));
    }
    let wg_size = bq * tpq;
    let mut src = String::new();

    // Params + bindings (match MultiHeadAttnGradData)
    src.push_str("struct Params {\n    q_seq: u32,\n    kv_seq: u32,\n    packed_heads: u32,\n    head_dim: u32,\n    window_size: u32,\n    _pad0: u32,\n    _pad1: u32,\n    _pad2: u32,\n}\n\n");
    src.push_str("var<storage> d_out: array<f32>;\n");
    src.push_str("var<storage> src_a: array<f32>;\n"); // Q
    src.push_str("var<storage> src_b: array<f32>;\n"); // K
    src.push_str("var<storage> bias: array<f32>;\n"); // V
    src.push_str("var<storage> lse: array<f32>;\n");
    src.push_str("var<storage> fwd_dst: array<f32>;\n"); // O
    src.push_str("var<storage, read_write> dst: array<f32>;\n"); // dQ
    src.push_str("var<uniform> params: Params;\n\n");

    // Shared K/V staging with BKV tiling: amortize barrier cost by loading
    // BKV KV positions worth of K and V at once, then looping in-register.
    let bkv: u32 = 8;
    let _ = writeln!(src, "var<workgroup> shared_k: array<f32, {}>;", bkv * hd);
    let _ = writeln!(src, "var<workgroup> shared_v: array<f32, {}>;\n", bkv * hd);

    // Shared memory only needed when TPQ > 1 (for cross-lane reductions).
    if tpq > 1 {
        // wg_score[qi*tpq + lane] partial Q·K
        // wg_dp   [qi*tpq + lane] partial dO·V
        // wg_rs   [qi*tpq + lane] partial dO·O for row_sum
        let _ = writeln!(src, "var<workgroup> wg_score: array<f32, {}>;", bq * tpq);
        let _ = writeln!(src, "var<workgroup> wg_dp: array<f32, {}>;", bq * tpq);
        let _ = writeln!(src, "var<workgroup> wg_rs: array<f32, {}>;\n", bq * tpq);

        // Grouped tree_reduce for wg_rs
        src.push_str("fn reduce_rs(tid: u32) {\n");
        let _ = writeln!(src, "    let local = tid % {tpq}u;");
        let _ = writeln!(src, "    let base = (tid / {tpq}u) * {tpq}u;");
        let mut stride = tpq / 2;
        while stride > 0 {
            src.push_str("    workgroupBarrier();\n");
            let _ = writeln!(
                src,
                "    if local < {stride}u {{ wg_rs[base + local] += wg_rs[base + local + {stride}u]; }}"
            );
            stride /= 2;
        }
        src.push_str("    workgroupBarrier();\n}\n\n");

        // Grouped tree_reduce for wg_score and wg_dp simultaneously
        src.push_str("fn reduce_score_dp(tid: u32) {\n");
        let _ = writeln!(src, "    let local = tid % {tpq}u;");
        let _ = writeln!(src, "    let base = (tid / {tpq}u) * {tpq}u;");
        let mut stride = tpq / 2;
        while stride > 0 {
            src.push_str("    workgroupBarrier();\n");
            let _ = writeln!(
                src,
                "    if local < {stride}u {{ wg_score[base + local] += wg_score[base + local + {stride}u]; wg_dp[base + local] += wg_dp[base + local + {stride}u]; }}"
            );
            stride /= 2;
        }
        src.push_str("    workgroupBarrier();\n}\n\n");
    }

    // Main kernel
    let _ = writeln!(src, "@compute @workgroup_size({wg_size})");
    src.push_str("fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {\n");
    let _ = writeln!(src, "    let qi = lid.x / {tpq}u;");
    let _ = writeln!(src, "    let lane = lid.x % {tpq}u;");
    let _ = writeln!(src, "    let d_base = lane * {ept}u;");
    let _ = writeln!(src, "    let pos = wgid.x * {bq}u + qi;");
    src.push_str("    let head = wgid.y;\n");
    src.push_str("    let q_seq = params.q_seq;\n");
    src.push_str("    let kv_seq = params.kv_seq;\n");
    src.push_str("    let num_heads = params.packed_heads >> 16u;\n");
    src.push_str("    let num_kv_heads = params.packed_heads & 0xFFFFu;\n");
    src.push_str("    let head_dim = params.head_dim;\n");
    src.push_str("    let valid = pos < q_seq && head < num_heads;\n\n");

    src.push_str("    let kv_head = head / (num_heads / max(num_kv_heads, 1u));\n");
    src.push_str("    let kv_head_off = kv_head * head_dim;\n");
    src.push_str("    let kv_dim = num_kv_heads * head_dim;\n");
    src.push_str("    let scale = inverseSqrt(f32(head_dim));\n");
    src.push_str("    var max_s = 0.0;\n    var log_sum = 0.0;\n    var q_base = 0u;\n");
    // Load Q and dO into thread-local registers
    for e in 0..ept {
        let _ = writeln!(src, "    var q{e} = 0.0;");
        let _ = writeln!(src, "    var do{e} = 0.0;");
    }
    src.push_str("    if valid {\n");
    src.push_str("        q_base = pos * (num_heads * head_dim) + head * head_dim;\n");
    for e in 0..ept {
        let _ = writeln!(src, "        q{e} = src_a[q_base + d_base + {e}u];");
        let _ = writeln!(src, "        do{e} = d_out[q_base + d_base + {e}u];");
    }
    src.push_str("        let lse_idx = (pos * num_heads + head) * 2u;\n");
    src.push_str("        max_s = lse[lse_idx];\n");
    src.push_str("        log_sum = lse[lse_idx + 1u];\n");
    src.push_str("    }\n\n");

    // row_sum = sum(dO * O) — partial across EPT elements, then reduce across TPQ (if needed)
    src.push_str("    var row_sum_part = 0.0;\n");
    src.push_str("    if valid {\n");
    for e in 0..ept {
        let _ = writeln!(
            src,
            "        row_sum_part += do{e} * fwd_dst[q_base + d_base + {e}u];"
        );
    }
    src.push_str("    }\n");
    if tpq > 1 {
        let _ = writeln!(src, "    wg_rs[qi * {tpq}u + lane] = row_sum_part;");
        src.push_str("    reduce_rs(lid.x);\n");
        let _ = writeln!(src, "    let row_sum = wg_rs[qi * {tpq}u];\n");
    } else {
        src.push_str("    let row_sum = row_sum_part;\n\n");
    }

    // Per-position KV range
    src.push_str(
        "    let my_kv_len = select(kv_seq, select(pos + 1u, 0u, !valid), kv_seq == 0u);\n",
    );
    src.push_str("    let window = params.window_size;\n");
    src.push_str(
        "    let my_kv_start = select(0u, my_kv_len - min(my_kv_len, window), window > 0u);\n",
    );
    // Workgroup-wide bounds (all threads must agree for potential barriers)
    let _ = writeln!(
        src,
        "    let last_pos = min(wgid.x * {bq}u + {bq}u - 1u, q_seq - 1u);"
    );
    let _ = writeln!(src, "    let first_pos = wgid.x * {bq}u;");
    src.push_str("    let max_kv_len = select(kv_seq, last_pos + 1u, kv_seq == 0u);\n");
    src.push_str("    let first_kv_len = select(kv_seq, first_pos + 1u, kv_seq == 0u);\n");
    src.push_str("    let min_kv_start = select(0u, first_kv_len - min(first_kv_len, window), window > 0u);\n\n");

    // Per-thread dQ accumulators (EPT elements)
    for e in 0..ept {
        let _ = writeln!(src, "    var dq{e} = 0.0;");
    }
    src.push('\n');

    if tpq == 1 {
        // Tiled KV loop with BKV positions per barrier (only when tpq=1).
        let _ = writeln!(src, "    let kv_range = max_kv_len - min_kv_start;");
        let _ = writeln!(
            src,
            "    let tile_end = min_kv_start + (kv_range / {bkv}u) * {bkv}u;"
        );
        src.push_str("    var t = min_kv_start;\n");
        let _ = writeln!(src, "    for (; t < tile_end; t += {bkv}u) {{");
        // Cooperative tile load (wg_size threads load BKV*hd elements)
        let tile_size = bkv * hd;
        let loads_per_thread = tile_size.div_ceil(wg_size);
        for l in 0..loads_per_thread {
            let off = l * wg_size;
            if off == 0 {
                let _ = writeln!(src, "        if lid.x < {tile_size}u {{");
                let _ = writeln!(src, "            let ki = lid.x / {hd}u;");
                let _ = writeln!(src, "            let kd = lid.x % {hd}u;");
                src.push_str("            let kb = (t + ki) * kv_dim + kv_head_off;\n");
                src.push_str("            shared_k[lid.x] = src_b[kb + kd];\n");
                src.push_str("            shared_v[lid.x] = bias[kb + kd];\n");
                src.push_str("        }\n");
            } else {
                let _ = writeln!(src, "        if lid.x + {off}u < {tile_size}u {{");
                let _ = writeln!(src, "            let ki = (lid.x + {off}u) / {hd}u;");
                let _ = writeln!(src, "            let kd = (lid.x + {off}u) % {hd}u;");
                src.push_str("            let kb = (t + ki) * kv_dim + kv_head_off;\n");
                let _ = writeln!(
                    src,
                    "            shared_k[lid.x + {off}u] = src_b[kb + kd];"
                );
                let _ = writeln!(src, "            shared_v[lid.x + {off}u] = bias[kb + kd];");
                src.push_str("        }\n");
            }
        }
        src.push_str("        workgroupBarrier();\n\n");

        // Inner loop: BKV positions, in registers, no barriers
        let _ = writeln!(src, "        for (var i = 0u; i < {bkv}u; i++) {{");
        src.push_str("            let kv_pos = t + i;\n");
        let _ = writeln!(src, "            let k_off = i * {hd}u + d_base;");
        src.push_str("            var score_part = 0.0;\n");
        src.push_str("            var dp_part = 0.0;\n");
        for e in 0..ept {
            let _ = writeln!(
                src,
                "            score_part += q{e} * shared_k[k_off + {e}u];"
            );
            let _ = writeln!(
                src,
                "            dp_part += do{e} * shared_v[k_off + {e}u];"
            );
        }
        src.push_str("            let score = score_part * scale;\n");
        src.push_str("            if valid && kv_pos >= my_kv_start && kv_pos < my_kv_len {\n");
        src.push_str("                let p_t = exp(min(score - max_s, 0.0) - log_sum);\n");
        src.push_str("                let ds_t = p_t * (dp_part - row_sum);\n");
        src.push_str("                let w = ds_t * scale;\n");
        for e in 0..ept {
            let _ = writeln!(src, "                dq{e} += w * shared_k[k_off + {e}u];");
        }
        src.push_str("            }\n");
        src.push_str("        }\n");
        src.push_str("        workgroupBarrier();\n");
        src.push_str("    }\n\n");

        // Tail: remaining KV positions one at a time
        src.push_str("    for (; t < max_kv_len; t++) {\n");
        src.push_str("        let k_base = t * kv_dim + kv_head_off;\n");
        let _ = writeln!(src, "        if lid.x < {hd}u {{");
        src.push_str("            shared_k[lid.x] = src_b[k_base + lid.x];\n");
        src.push_str("            shared_v[lid.x] = bias[k_base + lid.x];\n");
        src.push_str("        }\n");
        src.push_str("        workgroupBarrier();\n");
        src.push_str("        var sp2 = 0.0;\n");
        src.push_str("        var dp2 = 0.0;\n");
        for e in 0..ept {
            let _ = writeln!(src, "        sp2 += q{e} * shared_k[d_base + {e}u];");
            let _ = writeln!(src, "        dp2 += do{e} * shared_v[d_base + {e}u];");
        }
        src.push_str("        let score2 = sp2 * scale;\n");
        src.push_str("        if valid && t >= my_kv_start && t < my_kv_len {\n");
        src.push_str("            let p_t = exp(min(score2 - max_s, 0.0) - log_sum);\n");
        src.push_str("            let ds_t = p_t * (dp2 - row_sum);\n");
        src.push_str("            let w = ds_t * scale;\n");
        for e in 0..ept {
            let _ = writeln!(src, "            dq{e} += w * shared_k[d_base + {e}u];");
        }
        src.push_str("        }\n");
        src.push_str("        workgroupBarrier();\n");
        src.push_str("    }\n\n");
    } else {
        // TPQ>1 path: single KV position per iteration with cross-lane reduction.
        src.push_str("    for (var t = min_kv_start; t < max_kv_len; t++) {\n");
        src.push_str("        let k_base = t * kv_dim + kv_head_off;\n");
        let _ = writeln!(src, "        if lid.x < {hd}u {{");
        src.push_str("            shared_k[lid.x] = src_b[k_base + lid.x];\n");
        src.push_str("            shared_v[lid.x] = bias[k_base + lid.x];\n");
        src.push_str("        }\n");
        src.push_str("        workgroupBarrier();\n\n");

        for e in 0..ept {
            let _ = writeln!(src, "        let k{e} = shared_k[d_base + {e}u];");
            let _ = writeln!(src, "        let v{e} = shared_v[d_base + {e}u];");
        }
        src.push_str("        var score_part = 0.0;\n");
        src.push_str("        var dp_part = 0.0;\n");
        for e in 0..ept {
            let _ = writeln!(src, "        score_part += q{e} * k{e};");
            let _ = writeln!(src, "        dp_part += do{e} * v{e};");
        }
        let _ = writeln!(src, "        wg_score[qi * {tpq}u + lane] = score_part;");
        let _ = writeln!(src, "        wg_dp[qi * {tpq}u + lane] = dp_part;");
        src.push_str("        reduce_score_dp(lid.x);\n");
        let _ = writeln!(src, "        let score = wg_score[qi * {tpq}u] * scale;");
        let _ = writeln!(src, "        let dp_t = wg_dp[qi * {tpq}u];\n");
        src.push_str("        if valid && t >= my_kv_start && t < my_kv_len {\n");
        src.push_str("            let p_t = exp(min(score - max_s, 0.0) - log_sum);\n");
        src.push_str("            let ds_t = p_t * (dp_t - row_sum);\n");
        src.push_str("            let w = ds_t * scale;\n");
        for e in 0..ept {
            let _ = writeln!(src, "            dq{e} += w * k{e};");
        }
        src.push_str("        }\n");
        src.push_str("        workgroupBarrier();\n");
        src.push_str("    }\n\n");
    }

    src.push_str("    if valid {\n");
    for e in 0..ept {
        let _ = writeln!(src, "        dst[q_base + d_base + {e}u] = dq{e};");
    }
    src.push_str("    }\n");
    src.push_str("}\n");

    let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| {
        panic!(
            "generated flash grad_q WGSL failed to parse:\n{}\n---\n{}",
            e, src
        )
    });
    ShaderModule {
        module,
        source: src,
    }
}

/// Generate a Flash Attention 2 backward dK/dV kernel using vectorized
/// register pattern (each thread computes full Q·K, dO·O, dO·V dot
/// products in registers).
///
/// Mirrors the forward kernel's EPT/TPQ/BKV pattern. When TPQ=1 (EPT==hd)
/// there are NO workgroup barriers inside the Q loop — each thread owns
/// one full (kv_pos, head_range) output row and sequentially accumulates
/// dK/dV by loading Q/dO/O scalars directly from global memory.
pub fn generate_flash_grad_kv_module(head_dim: u32) -> ShaderModule {
    use std::fmt::Write;
    assert!(head_dim.is_power_of_two() && head_dim >= 2);

    let hd = head_dim;
    let ept: u32 = hd.min(64);
    let tpq = hd / ept; // threads per KV position
    let bkv: u32 = (256 / tpq).max(1);
    if bkv <= 1 {
        return parse_wgsl(include_str!("shaders/mha_grad_kv.wgsl"));
    }
    let wg_size = bkv * tpq;
    let mut src = String::new();

    // Params + bindings (match MultiHeadAttnGradKVData)
    src.push_str("struct Params {\n    q_seq: u32,\n    kv_seq: u32,\n    packed_heads: u32,\n    head_dim: u32,\n    window_size: u32,\n    _pad0: u32,\n    _pad1: u32,\n    _pad2: u32,\n}\n\n");
    src.push_str("var<storage> d_out: array<f32>;\n");
    src.push_str("var<storage> src_a: array<f32>;\n"); // Q
    src.push_str("var<storage> src_b: array<f32>;\n"); // K
    src.push_str("var<storage> bias: array<f32>;\n"); // V
    src.push_str("var<storage> lse: array<f32>;\n");
    src.push_str("var<storage> fwd_dst: array<f32>;\n"); // O
    src.push_str("var<storage, read_write> dst: array<f32>;\n"); // dK
    src.push_str("var<storage, read_write> dst2: array<f32>;\n"); // dV
    src.push_str("var<uniform> params: Params;\n\n");

    // Shared Q/dO/O staging: all BKV groups share the same q position `pos`
    // in the inner loop, so staging once per WG amortizes global reads.
    let _ = writeln!(src, "var<workgroup> shared_q: array<f32, {hd}>;");
    let _ = writeln!(src, "var<workgroup> shared_do: array<f32, {hd}>;");
    let _ = writeln!(src, "var<workgroup> shared_o: array<f32, {hd}>;\n");

    // Shared memory only needed when TPQ > 1 (cross-lane reductions)
    if tpq > 1 {
        let _ = writeln!(src, "var<workgroup> wg_score: array<f32, {}>;", bkv * tpq);
        let _ = writeln!(src, "var<workgroup> wg_rs: array<f32, {}>;", bkv * tpq);
        let _ = writeln!(src, "var<workgroup> wg_dp: array<f32, {}>;\n", bkv * tpq);
        src.push_str("fn reduce_triple(tid: u32) {\n");
        let _ = writeln!(src, "    let local = tid % {tpq}u;");
        let _ = writeln!(src, "    let base = (tid / {tpq}u) * {tpq}u;");
        let mut stride = tpq / 2;
        while stride > 0 {
            src.push_str("    workgroupBarrier();\n");
            let _ = writeln!(
                src,
                "    if local < {stride}u {{ wg_score[base + local] += wg_score[base + local + {stride}u]; wg_rs[base + local] += wg_rs[base + local + {stride}u]; wg_dp[base + local] += wg_dp[base + local + {stride}u]; }}"
            );
            stride /= 2;
        }
        src.push_str("    workgroupBarrier();\n}\n\n");
    }

    // Main kernel
    let _ = writeln!(src, "@compute @workgroup_size({wg_size})");
    src.push_str("fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {\n");
    let _ = writeln!(src, "    let ki = lid.x / {tpq}u;"); // KV position within tile
    let _ = writeln!(src, "    let lane = lid.x % {tpq}u;");
    let _ = writeln!(src, "    let d_base = lane * {ept}u;");
    let _ = writeln!(src, "    let t = wgid.x * {bkv}u + ki;"); // global KV position
    src.push_str("    let kv_head = wgid.y;\n");
    src.push_str("    let q_seq = params.q_seq;\n");
    src.push_str("    let kv_seq = params.kv_seq;\n");
    src.push_str("    let num_heads = params.packed_heads >> 16u;\n");
    src.push_str("    let num_kv_heads = params.packed_heads & 0xFFFFu;\n");
    src.push_str("    let head_dim = params.head_dim;\n\n");

    src.push_str("    let effective_kv_seq = select(kv_seq, q_seq, kv_seq == 0u);\n");
    src.push_str("    let valid = t < effective_kv_seq && kv_head < num_kv_heads;\n");
    src.push_str("    let heads_per_kv = num_heads / max(num_kv_heads, 1u);\n");
    src.push_str("    let kv_dim = num_kv_heads * head_dim;\n");
    src.push_str("    let q_dim = num_heads * head_dim;\n");
    src.push_str("    let kv_base = t * kv_dim + kv_head * head_dim;\n");
    src.push_str("    let scale = inverseSqrt(f32(head_dim));\n\n");

    // Load K/V slices for this thread into registers
    for e in 0..ept {
        let _ = writeln!(src, "    var k{e} = 0.0;");
        let _ = writeln!(src, "    var v{e} = 0.0;");
    }
    src.push_str("    if valid {\n");
    for e in 0..ept {
        let _ = writeln!(src, "        k{e} = src_b[kv_base + d_base + {e}u];");
        let _ = writeln!(src, "        v{e} = bias[kv_base + d_base + {e}u];");
    }
    src.push_str("    }\n\n");

    // Per-thread dK/dV accumulators (EPT elements each)
    for e in 0..ept {
        let _ = writeln!(src, "    var dk{e} = 0.0;");
        let _ = writeln!(src, "    var dv{e} = 0.0;");
    }
    src.push('\n');

    // Q loop range bounds
    src.push_str("    let start_pos = select(0u, t, kv_seq == 0u);\n");
    src.push_str("    let window = params.window_size;\n");
    src.push_str("    let end_pos = select(q_seq, min(q_seq, t + window), window > 0u);\n\n");

    // Workgroup-wide loop bounds (used for barrier consistency when TPQ>1)
    let _ = writeln!(src, "    let first_t = wgid.x * {bkv}u;");
    let _ = writeln!(
        src,
        "    let last_t = min(wgid.x * {bkv}u + {bkv}u - 1u, effective_kv_seq - 1u);"
    );
    src.push_str("    let wg_start = select(0u, first_t, kv_seq == 0u);\n");
    src.push_str("    let wg_end = select(q_seq, min(q_seq, last_t + window), window > 0u);\n\n");

    // Simple per-position cooperative load (fast path for BKV=4, hd=64).
    src.push_str("    for (var pos = wg_start; pos < wg_end; pos++) {\n");
    src.push_str("        for (var head_rel = 0u; head_rel < heads_per_kv; head_rel++) {\n");
    src.push_str("            let head = kv_head * heads_per_kv + head_rel;\n");
    src.push_str("            let q_base = pos * q_dim + head * head_dim;\n\n");

    let _ = writeln!(src, "            if lid.x < {hd}u {{");
    src.push_str("                shared_q[lid.x] = src_a[q_base + lid.x];\n");
    src.push_str("                shared_do[lid.x] = d_out[q_base + lid.x];\n");
    src.push_str("                shared_o[lid.x] = fwd_dst[q_base + lid.x];\n");
    src.push_str("            }\n");
    src.push_str("            workgroupBarrier();\n\n");

    for e in 0..ept {
        let _ = writeln!(src, "            let q{e} = shared_q[d_base + {e}u];");
        let _ = writeln!(src, "            let do{e} = shared_do[d_base + {e}u];");
        let _ = writeln!(src, "            let o{e} = shared_o[d_base + {e}u];");
    }
    src.push_str("            var score_part = 0.0;\n");
    src.push_str("            var rs_part = 0.0;\n");
    src.push_str("            var dp_part = 0.0;\n");
    for e in 0..ept {
        let _ = writeln!(src, "            score_part += q{e} * k{e};");
        let _ = writeln!(src, "            rs_part += do{e} * o{e};");
        let _ = writeln!(src, "            dp_part += do{e} * v{e};");
    }
    if tpq > 1 {
        let _ = writeln!(
            src,
            "            wg_score[ki * {tpq}u + lane] = score_part;"
        );
        let _ = writeln!(src, "            wg_rs[ki * {tpq}u + lane] = rs_part;");
        let _ = writeln!(src, "            wg_dp[ki * {tpq}u + lane] = dp_part;");
        src.push_str("            reduce_triple(lid.x);\n");
        let _ = writeln!(
            src,
            "            let score = wg_score[ki * {tpq}u] * scale;"
        );
        let _ = writeln!(src, "            let row_sum = wg_rs[ki * {tpq}u];");
        let _ = writeln!(src, "            let dp_t = wg_dp[ki * {tpq}u];\n");
    } else {
        src.push_str("            let score = score_part * scale;\n");
        src.push_str("            let row_sum = rs_part;\n");
        src.push_str("            let dp_t = dp_part;\n");
    }
    src.push_str("            if valid && pos >= start_pos && pos < end_pos {\n");
    src.push_str("                let lse_idx = (pos * num_heads + head) * 2u;\n");
    src.push_str(
        "                let p_t = exp(min(score - lse[lse_idx], 0.0) - lse[lse_idx + 1u]);\n",
    );
    src.push_str("                let ds_t = p_t * (dp_t - row_sum);\n");
    src.push_str("                let w_dk = ds_t * scale;\n");
    for e in 0..ept {
        let _ = writeln!(src, "                dk{e} += w_dk * q{e};");
        let _ = writeln!(src, "                dv{e} += p_t * do{e};");
    }
    src.push_str("            }\n");
    src.push_str("            workgroupBarrier();\n");
    src.push_str("        }\n");
    src.push_str("    }\n\n");

    src.push_str("    if valid {\n");
    for e in 0..ept {
        let _ = writeln!(src, "        dst[kv_base + d_base + {e}u] = dk{e};");
        let _ = writeln!(src, "        dst2[kv_base + d_base + {e}u] = dv{e};");
    }
    src.push_str("    }\n");
    src.push_str("}\n");

    let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| {
        panic!(
            "generated flash grad_kv WGSL failed to parse:\n{}\n---\n{}",
            e, src
        )
    });
    ShaderModule {
        module,
        source: src,
    }
}

fn gen_conv2d_gemm_coop() -> ShaderModule {
    let default_config = CoopConfig {
        tile_size: 16,
        use_f16_input: true,
    };
    gen_conv2d_gemm_coop_wgsl(&default_config)
}

fn gen_conv2d_gemm_coop_wgsl(config: &CoopConfig) -> ShaderModule {
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

    let src = include_str!("shaders/conv2d_gemm_coop.wgsl");
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

fn gen_conv2d_grad_input_gemm_coop_3x3() -> ShaderModule {
    let default_config = CoopConfig {
        tile_size: 16,
        use_f16_input: true,
    };
    gen_conv2d_grad_input_gemm_coop_3x3_wgsl(&default_config)
}

fn gen_conv2d_grad_input_gemm_coop_3x3_wgsl(config: &CoopConfig) -> ShaderModule {
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

    let src = include_str!("shaders/conv2d_grad_input_gemm_coop_3x3.wgsl");
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
                ShaderGroup::Conv2dGemmCoop,
                naga::valid::Capabilities::COOPERATIVE_MATRIX
                    | naga::valid::Capabilities::SHADER_FLOAT16,
            ),
            (
                ShaderGroup::Conv2dGradInputGemmCoop,
                naga::valid::Capabilities::COOPERATIVE_MATRIX
                    | naga::valid::Capabilities::SHADER_FLOAT16,
            ),
            (
                ShaderGroup::Conv2dGradInputGemmCoop3x3,
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
                ShaderGroup::MultiHeadAttnGradQ,
                naga::valid::Capabilities::empty(),
            ),
            (ShaderGroup::FlashGradQ, naga::valid::Capabilities::empty()),
            (
                ShaderGroup::MultiHeadAttnGradK,
                naga::valid::Capabilities::empty(),
            ),
            (
                ShaderGroup::MultiHeadAttnGradKV,
                naga::valid::Capabilities::empty(),
            ),
            (ShaderGroup::FlashGradKV, naga::valid::Capabilities::empty()),
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
            (
                ShaderGroup::RmsNormGradWRowPar,
                naga::valid::Capabilities::empty(),
            ),
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
    fn test_unified_attention_wgsl() {
        let _ = generate_attention_module(64);
        let _ = generate_attention_module(32);
        let _ = generate_attention_module(128);
    }

    #[test]
    fn test_flash_attention_wgsl() {
        // BQ=4 for hd=64, BQ=8 for hd=32, BQ=2 for hd=128
        let _ = generate_flash_attention_module(64);
        let _ = generate_flash_attention_module(32);
        let _ = generate_flash_attention_module(128);
        // hd=256 should fall back to BQ=1 (regular attention)
        let _ = generate_flash_attention_module(256);
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
            (ShaderGroup::LayerNorm, empty),
            (ShaderGroup::SwiGLUGrad, empty),
            (ShaderGroup::SwiGLUConcat, empty),
            (ShaderGroup::SumRows, empty),
            (ShaderGroup::RmsNormGrad, empty),
            (ShaderGroup::RmsNormGradWRowPar, empty),
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
                    | ShaderGroup::Conv2dGemmCoop
                    | ShaderGroup::Conv2dGradInputGemmCoop
                    | ShaderGroup::Conv2dGradInputGemmCoop3x3
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
            match *entry {
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
                ShaderEntry::LayerNorm => vec!["src", "src_b", "bias", "dst", "params"],
                ShaderEntry::MultiHeadAttn | ShaderEntry::FlashAttention => {
                    vec!["src_a", "src_b", "bias", "dst", "lse", "params"]
                }
                ShaderEntry::MultiHeadAttnGradQ
                | ShaderEntry::FlashGradQ
                | ShaderEntry::MultiHeadAttnGradK
                | ShaderEntry::MultiHeadAttnGradV => {
                    vec![
                        "d_out", "src_a", "src_b", "bias", "lse", "fwd_dst", "dst", "params",
                    ]
                }
                ShaderEntry::MultiHeadAttnGradKV | ShaderEntry::FlashGradKV => {
                    vec![
                        "d_out", "src_a", "src_b", "bias", "lse", "fwd_dst", "dst", "dst2",
                        "params",
                    ]
                }
                // All three SwiGLUGrad entries share the same module globals
                ShaderEntry::SwiGLUGradGate | ShaderEntry::SwiGLUGradUp | ShaderEntry::SiluGrad => {
                    vec!["src_a", "src_b", "src_c", "dst", "params"]
                }
                ShaderEntry::SwiGLUConcat | ShaderEntry::SwiGLUConcatGrad => {
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
                ShaderEntry::Conv2dGemm
                | ShaderEntry::Conv2dGemmSmall
                | ShaderEntry::Conv2dGemmCoop => vec!["src", "weight", "dst", "params"],
                ShaderEntry::Conv2dGradInput => vec!["grad_out", "weight", "dst", "params"],
                ShaderEntry::Conv2dGradInputGemm | ShaderEntry::Conv2dGradInputGemmSmall => {
                    vec!["grad_out", "weight", "dst", "params"]
                }
                ShaderEntry::Conv2dGradInputGemmCoop
                | ShaderEntry::Conv2dGradInputGemmCoop3x3 => {
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
                ShaderEntry::WinogradInputTransform | ShaderEntry::WinogradOutputTransform => {
                    vec!["src", "dst", "params"]
                }
                ShaderEntry::WinogradBatchedMatMul | ShaderEntry::WinogradBatchedMatMulSmall => {
                    vec!["matrix_a", "matrix_b", "matrix_c", "params"]
                }
                ShaderEntry::WinogradWeightTransform => vec!["src", "dst", "params"],
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
            ShaderEntry::Gelu,
            ShaderEntry::Tanh,
            ShaderEntry::LayerNorm,
            ShaderEntry::MultiHeadAttn,
            ShaderEntry::FlashAttention,
            ShaderEntry::MultiHeadAttnGradQ,
            ShaderEntry::FlashGradQ,
            ShaderEntry::MultiHeadAttnGradK,
            ShaderEntry::MultiHeadAttnGradKV,
            ShaderEntry::FlashGradKV,
            ShaderEntry::MultiHeadAttnGradV,
            ShaderEntry::SwiGLUGradGate,
            ShaderEntry::SwiGLUGradUp,
            ShaderEntry::SwiGLUConcat,
            ShaderEntry::SwiGLUConcatGrad,
            ShaderEntry::SiluGrad,
            ShaderEntry::RmsNormGradW,
            ShaderEntry::RmsNormGradWRowPar,
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
            ShaderEntry::WinogradInputTransform,
            ShaderEntry::WinogradOutputTransform,
            ShaderEntry::WinogradBatchedMatMul,
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
