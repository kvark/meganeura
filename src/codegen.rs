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
    /// Split flash-attention dK kernel — alternative to `FlashGradKV`
    /// with lower register pressure, dispatched back-to-back with
    /// `FlashGradV` when the cost model prefers the split.
    FlashGradK,
    /// Split flash-attention dV kernel — see [`ShaderGroup::FlashGradK`].
    FlashGradV,
    /// Cooperative-matrix flash attention forward.
    /// Phase 1: coop_mat for QK^T, scalar softmax + PV. Opt-in via
    /// `MEGANEURA_FLASH_FWD_COOP=1`.
    FlashAttentionCoop,
    /// Cooperative-matrix flash backward dQ kernel (3 coop matmuls
    /// per KV tile: score, dp, ds·K). Opt-in via
    /// `MEGANEURA_FLASH_BWD_COOP=1`.
    FlashGradQCoop,
    /// Cooperative-matrix flash backward dK + dV kernel (fused).
    /// Two coop matmuls per Q-tile: score = K·Q^T, dp = V·dO^T.
    /// dV/dK accumulate in per-thread registers across the Q loop.
    FlashGradKVCoop,
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
        ShaderGroup::FlashAttentionCoop => generate_flash_attention_coop_module(64),
        ShaderGroup::FlashGradQCoop => generate_flash_grad_q_coop_module(64),
        ShaderGroup::FlashGradKVCoop => generate_flash_grad_kv_coop_module(64),
        ShaderGroup::MultiHeadAttnGradQ => parse_wgsl(include_str!("shaders/mha_grad_q.wgsl")),
        ShaderGroup::FlashGradQ => generate_flash_grad_q_module(64),
        ShaderGroup::MultiHeadAttnGradK => parse_wgsl(include_str!("shaders/mha_grad_k.wgsl")),
        ShaderGroup::MultiHeadAttnGradKV => parse_wgsl(include_str!("shaders/mha_grad_kv.wgsl")),
        ShaderGroup::FlashGradKV => generate_flash_grad_kv_module(64),
        ShaderGroup::FlashGradK => generate_flash_grad_k_module(64),
        ShaderGroup::FlashGradV => generate_flash_grad_v_module(64),
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
/// Identifies which flash-attention kernel an EPT decision applies to.
///
/// Each kernel has different register pressure (the backward `GradKv`
/// kernel hits 210 regs/thread at EPT=32 on Blackwell, while forward
/// only sees 80) so different optimal EPTs. The `flash_ept_for()`
/// lookup picks the right cap per kernel.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum FlashKernel {
    /// Forward attention (`generate_flash_attention_module`).
    Forward,
    /// Backward dQ (`generate_flash_grad_q_module`).
    GradQ,
    /// Fused backward dK+dV (`generate_flash_grad_kv_module`).
    GradKv,
    /// Split backward dK (`generate_flash_grad_k_module`).
    GradK,
    /// Split backward dV (`generate_flash_grad_v_module`).
    GradV,
}

/// Per-kernel EPT cap selection, populated by
/// [`crate::runtime::auto_tune_flash_ept`] at session creation.
///
/// Each cap is the largest power-of-two EPT whose measured register
/// count keeps the kernel under the occupancy threshold (~128 regs →
/// 2 wg/SM on a 64K-reg Blackwell SM with 256-thread workgroups).
/// `head_dim.min(cap)` is the EPT actually used by codegen.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlashEptConfig {
    pub forward_cap: u32,
    pub grad_q_cap: u32,
    pub grad_kv_cap: u32,
    pub grad_k_cap: u32,
    pub grad_v_cap: u32,
}

impl Default for FlashEptConfig {
    fn default() -> Self {
        // Pre-pipeline-stats default: cap everything at 32 (same value
        // the legacy code used for all kernels). Auto-tune overrides
        // this with measured per-kernel optima.
        Self {
            forward_cap: 32,
            grad_q_cap: 32,
            grad_kv_cap: 32,
            grad_k_cap: 32,
            grad_v_cap: 32,
        }
    }
}

static FLASH_EPT_CONFIG: std::sync::OnceLock<FlashEptConfig> = std::sync::OnceLock::new();

/// Cooperative-matrix capabilities snapshot the compiler consults
/// when choosing between scalar and coop kernel variants. Mirrors
/// Blade's `CooperativeMatrix` struct minus the methods. Compile-time
/// dispatch can't ask the GPU directly (no Blade context), so we
/// cache what we need here. Set by `runtime::install_auto_tune`,
/// defaults to all-zero (= no coop, scalar everywhere).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CoopCaps {
    pub f16_tile: u32,
    pub f32_tile: u32,
}

impl CoopCaps {
    pub fn is_supported(&self) -> bool {
        self.f16_tile > 0 || self.f32_tile > 0
    }
    /// True when the kernels that hardcode `coop_mat16x16<f16,...>`
    /// can run (NVIDIA, RDNA3, Xe-HPG). False on Apple (8x8 f32) or
    /// any GPU without KHR_cooperative_matrix at the right tile size.
    pub fn supports_16x16_f16(&self) -> bool {
        self.f16_tile >= 16
    }
}

static COOP_CAPS: std::sync::OnceLock<CoopCaps> = std::sync::OnceLock::new();

pub fn set_coop_caps(caps: CoopCaps) {
    let _ = COOP_CAPS.set(caps);
}

pub fn coop_caps() -> CoopCaps {
    *COOP_CAPS.get().unwrap_or(&CoopCaps::default())
}

/// Backwards-compatible — true when the GPU supports any
/// cooperative_matrix path (f16 or f32).
pub fn coop_matrix_available() -> bool {
    coop_caps().is_supported()
}

/// Backwards-compatible setter. Conservatively assumes f16 16x16
/// (the common case the old boolean gate implied).
pub fn set_coop_matrix_available(available: bool) {
    let caps = if available {
        CoopCaps {
            f16_tile: 16,
            f32_tile: 0,
        }
    } else {
        CoopCaps::default()
    };
    let _ = COOP_CAPS.set(caps);
}

/// Install a process-wide EPT config (typically the result of
/// `runtime::auto_tune_flash_ept`). Has no effect after the first
/// successful call — the config locks in for the rest of the process.
pub fn set_flash_ept_config(cfg: FlashEptConfig) {
    let _ = FLASH_EPT_CONFIG.set(cfg);
}

/// Retrieve the active EPT config.
pub fn flash_ept_config() -> FlashEptConfig {
    *FLASH_EPT_CONFIG.get().unwrap_or(&FlashEptConfig::default())
}

/// Per-kernel EPT actually used in codegen and dispatch.
///
/// Resolution order:
///   1. `MEGANEURA_FLASH_EPT_CAP` env var (overrides everything,
///      applied uniformly to all kernels — for benchmarking sweeps).
///   2. The installed `FlashEptConfig` (typically from auto-tune).
///   3. Default cap of 32.
///
/// **Important**: the same value must be observed by both codegen and
/// `Compiler::attention_dispatch*`, otherwise generated tile sizes
/// won't match the workgroup counts and kernels produce wrong results.
/// Both sites call this function with the same `FlashKernel` argument.
pub fn flash_ept_for(kernel: FlashKernel, head_dim: u32) -> u32 {
    if let Some(cap) = std::env::var("MEGANEURA_FLASH_EPT_CAP")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|v| v.is_power_of_two() && *v >= 2)
    {
        return head_dim.min(cap);
    }
    let cfg = flash_ept_config();
    let cap = match kernel {
        FlashKernel::Forward => cfg.forward_cap,
        FlashKernel::GradQ => cfg.grad_q_cap,
        FlashKernel::GradKv => cfg.grad_kv_cap,
        FlashKernel::GradK => cfg.grad_k_cap,
        FlashKernel::GradV => cfg.grad_v_cap,
    };
    head_dim.min(cap)
}

/// Legacy global-cap accessor, retained for callers that don't have a
/// `FlashKernel` context (e.g. analyze_shaders default sweep).
/// New code should prefer [`flash_ept_for`].
pub fn flash_ept_cap() -> u32 {
    if let Some(cap) = std::env::var("MEGANEURA_FLASH_EPT_CAP")
        .ok()
        .and_then(|s| s.parse::<u32>().ok())
        .filter(|v| v.is_power_of_two() && *v >= 2)
    {
        return cap;
    }
    flash_ept_config().forward_cap
}

pub fn generate_flash_attention_module(head_dim: u32) -> ShaderModule {
    use std::fmt::Write;
    assert!(
        head_dim.is_power_of_two() && head_dim >= 2,
        "attention head_dim must be a power of 2 ≥ 2, got {head_dim}"
    );

    let hd = head_dim;
    // Per-kernel EPT picked by `flash_ept_for(FlashKernel::Forward, ..)`,
    // which honors MEGANEURA_FLASH_EPT_CAP overrides.
    let ept: u32 = flash_ept_for(FlashKernel::Forward, hd);
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

/// Cooperative-matrix flash attention forward (Phase 1).
///
/// Replaces only the QK^T product with a `coop_mat` MMA — softmax and
/// PV stay scalar — to keep the change tractable and validate the
/// coop path before tackling PV. Critical design point: the O
/// accumulator is held in *registers* across the entire KV loop (not
/// re-staged through shared memory each iteration), and the per-row
/// rescale runs INSIDE the same thread that owns the accumulator.
/// Each row's softmax math is duplicated 4× (once per d-chunk) but
/// that's a small constant (16 ops/row) vs the cost of the shared-mem
/// roundtrip.
///
/// Workgroup layout (64 threads = 16 rows × 4 d-chunks):
///   * `BQ = 16`, `BKV = 16` — match the coop_mat tile size.
///   * Each thread owns one (row, d_chunk) pair: holds
///     `O_acc[chunk_hd]` and `local_max` / `local_sum` in registers.
///   * Q is staged once per workgroup into shared as f16 [BQ × hd].
///   * K is staged per KV tile, **transposed** into shared as f16
///     [hd × BKV] so the MMA reads it as the B operand for Q @ K^T.
///   * V is staged per KV tile as f16 [BKV × hd] for the scalar PV.
///   * The score tile (after MMA) goes to shared_score[BQ × BKV];
///     each thread reads its row's BKV scores and runs softmax
///     locally.
///
/// Caller dispatch must use workgroups = `[ceil(q_seq/16), num_heads, 1]`.
/// `head_dim` must be a multiple of 16.
pub fn generate_flash_attention_coop_module(head_dim: u32) -> ShaderModule {
    use std::fmt::Write;
    assert!(
        head_dim >= 16 && head_dim.is_multiple_of(16),
        "coop flash attention requires head_dim multiple of 16, got {head_dim}"
    );
    let hd = head_dim;
    let hd_tiles = hd / 16;
    let bq: u32 = 16;
    let bkv: u32 = 16;
    let wg_size: u32 = 64;
    assert!(
        wg_size == bq * 4,
        "coop flash assumes wg_size=64 / BQ=16 / 4 hd-chunks per row"
    );
    let chunks_per_row: u32 = 4;
    let chunk_hd: u32 = hd / chunks_per_row;

    let mut src = String::new();
    src.push_str("enable f16;\n");
    src.push_str("enable wgpu_cooperative_matrix;\n\n");
    src.push_str("struct Params {\n    q_seq: u32,\n    kv_seq: u32,\n    packed_heads: u32,\n    head_dim: u32,\n    window_size: u32,\n    _pad0: u32,\n    _pad1: u32,\n    _pad2: u32,\n}\n\n");
    src.push_str("var<storage> src_a: array<f32>;\n"); // Q
    src.push_str("var<storage> src_b: array<f32>;\n"); // K
    src.push_str("var<storage> bias: array<f32>;\n"); // V
    src.push_str("var<storage, read_write> dst: array<f32>;\n"); // O
    src.push_str("var<storage, read_write> lse: array<f32>;\n");
    src.push_str("var<uniform> params: Params;\n\n");

    let _ = writeln!(src, "var<workgroup> shared_q: array<f16, {}>;", bq * hd);
    let _ = writeln!(src, "var<workgroup> shared_k_t: array<f16, {}>;", hd * bkv);
    let _ = writeln!(src, "var<workgroup> shared_v: array<f16, {}>;", bkv * hd);
    let _ = writeln!(
        src,
        "var<workgroup> shared_score: array<f32, {}>;",
        bq * bkv
    );
    src.push('\n');

    let _ = writeln!(src, "@compute @workgroup_size({wg_size})");
    src.push_str("fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {\n");
    let _ = writeln!(src, "    let pos_base = wgid.x * {bq}u;");
    src.push_str("    let head = wgid.y;\n");
    src.push_str("    let q_seq = params.q_seq;\n");
    src.push_str("    let kv_seq = params.kv_seq;\n");
    src.push_str("    let num_heads = params.packed_heads >> 16u;\n");
    src.push_str("    let num_kv_heads = params.packed_heads & 0xFFFFu;\n");
    src.push_str("    let head_dim = params.head_dim;\n");
    src.push_str("    let window_size = params.window_size;\n");
    src.push_str("    let kv_head = head / (num_heads / max(num_kv_heads, 1u));\n");
    src.push_str("    let kv_head_off = kv_head * head_dim;\n");
    src.push_str("    let kv_dim = num_kv_heads * head_dim;\n");
    src.push_str("    let scale = inverseSqrt(f32(head_dim));\n\n");

    // Per-thread (row, chunk) — index hoisted to top.
    let _ = writeln!(src, "    let row = lid.x / {chunks_per_row}u;");
    let _ = writeln!(src, "    let chunk = lid.x % {chunks_per_row}u;");
    let _ = writeln!(src, "    let d_off = chunk * {chunk_hd}u;");
    src.push_str("    let qpos = pos_base + row;\n");
    src.push_str("    let q_valid = qpos < q_seq && head < num_heads;\n\n");

    // Per-row valid KV range (causal + sliding window).
    src.push_str("    let row_kv_len = select(kv_seq, qpos + 1u, kv_seq == 0u);\n");
    src.push_str(
        "    let row_kv_start = select(0u, row_kv_len - min(row_kv_len, window_size), window_size > 0u);\n\n",
    );

    // Workgroup-wide bounds (drives all threads through the same
    // outer KV loop).
    src.push_str("    let last_pos = min(pos_base + 15u, q_seq - 1u);\n");
    src.push_str("    let max_kv_len = select(kv_seq, last_pos + 1u, kv_seq == 0u);\n");
    src.push_str(
        "    let min_kv_start = select(0u, max_kv_len - min(max_kv_len, window_size), window_size > 0u);\n\n",
    );

    // Per-thread O accumulator and softmax state — REGISTERS.
    let _ = writeln!(src, "    var local_o: array<f32, {chunk_hd}>;");
    let _ = writeln!(src, "    for (var e = 0u; e < {chunk_hd}u; e = e + 1u) {{");
    src.push_str("        local_o[e] = 0.0;\n");
    src.push_str("    }\n");
    src.push_str("    var local_max: f32 = -1e30;\n");
    src.push_str("    var local_sum: f32 = 0.0;\n\n");

    // ---- Stage Q once per workgroup ----
    let q_total = bq * hd;
    let _ = writeln!(
        src,
        "    for (var i = lid.x; i < {q_total}u; i = i + {wg_size}u) {{"
    );
    let _ = writeln!(src, "        let r = i / {hd}u;");
    let _ = writeln!(src, "        let col = i % {hd}u;");
    src.push_str("        let qp = pos_base + r;\n");
    src.push_str(
        "        if qp < q_seq { shared_q[i] = f16(src_a[qp * (num_heads * head_dim) + head * head_dim + col]); } else { shared_q[i] = f16(0.0); }\n",
    );
    src.push_str("    }\n");
    src.push_str("    workgroupBarrier();\n\n");

    // ---- KV tile loop ----
    let _ = writeln!(
        src,
        "    let tile_end = min_kv_start + ((max_kv_len - min_kv_start) / {bkv}u) * {bkv}u;"
    );
    src.push_str("    var t = min_kv_start;\n");
    let _ = writeln!(src, "    for (; t < tile_end; t = t + {bkv}u) {{");

    // Stage K transposed [hd × bkv] and V natural [bkv × hd] into shared.
    let kv_total = bkv * hd;
    let _ = writeln!(
        src,
        "        for (var i = lid.x; i < {kv_total}u; i = i + {wg_size}u) {{"
    );
    let _ = writeln!(src, "            let ki = i / {hd}u;");
    let _ = writeln!(src, "            let d  = i % {hd}u;");
    src.push_str("            let kv_pos = t + ki;\n");
    src.push_str(
        "            shared_k_t[d * 16u + ki] = f16(src_b[kv_pos * kv_dim + kv_head_off + d]);\n",
    );
    src.push_str(
        "            shared_v[ki * head_dim + d] = f16(bias[kv_pos * kv_dim + kv_head_off + d]);\n",
    );
    src.push_str("        }\n");
    src.push_str("        workgroupBarrier();\n\n");

    // Cooperative QK^T → shared_score.
    src.push_str("        var score_acc = coop_mat16x16<f32,C>();\n");
    let _ = writeln!(
        src,
        "        for (var ht = 0u; ht < {hd_tiles}u; ht = ht + 1u) {{"
    );
    let _ = writeln!(
        src,
        "            let a = coopLoadT<coop_mat16x16<f16,A>>(&shared_q[ht * 16u], {hd}u);"
    );
    let _ = writeln!(
        src,
        "            let b = coopLoadT<coop_mat16x16<f16,B>>(&shared_k_t[ht * 16u * {bkv}u], {bkv}u);"
    );
    src.push_str("            score_acc = coopMultiplyAdd(a, b, score_acc);\n");
    src.push_str("        }\n");
    let _ = writeln!(
        src,
        "        coopStoreT(score_acc, &shared_score[0], {bkv}u);"
    );
    src.push_str("        workgroupBarrier();\n\n");

    // Per-thread row softmax + PV. Each thread owns one (row, chunk).
    // The 4 chunks of a row redundantly compute the same row max/sum
    // (16 mul/add/exp ops per row — small constant).
    src.push_str("        var rowmax = -1e30;\n");
    let _ = writeln!(src, "        for (var j = 0u; j < {bkv}u; j = j + 1u) {{");
    src.push_str("            let kv_pos = t + j;\n");
    src.push_str("            if q_valid && kv_pos >= row_kv_start && kv_pos < row_kv_len {\n");
    let _ = writeln!(
        src,
        "                rowmax = max(rowmax, shared_score[row * {bkv}u + j] * scale);"
    );
    src.push_str("            }\n");
    src.push_str("        }\n");
    src.push_str("        let new_max = max(local_max, rowmax);\n");
    src.push_str("        let correction = select(exp(local_max - new_max), 0.0, !q_valid);\n");

    // Apply correction to the O accumulator (registers).
    let _ = writeln!(
        src,
        "        for (var e = 0u; e < {chunk_hd}u; e = e + 1u) {{"
    );
    src.push_str("            local_o[e] = local_o[e] * correction;\n");
    src.push_str("        }\n");

    // PV: for each j compute p_j = exp(score-new_max), accumulate into local_o.
    src.push_str("        var rowsum = 0.0;\n");
    let _ = writeln!(src, "        for (var j = 0u; j < {bkv}u; j = j + 1u) {{");
    src.push_str("            let kv_pos = t + j;\n");
    src.push_str(
        "            let masked = !(q_valid && kv_pos >= row_kv_start && kv_pos < row_kv_len);\n",
    );
    let _ = writeln!(
        src,
        "            let score = shared_score[row * {bkv}u + j] * scale;"
    );
    src.push_str("            let p = select(exp(score - new_max), 0.0, masked);\n");
    src.push_str("            rowsum = rowsum + p;\n");
    let _ = writeln!(
        src,
        "            for (var e = 0u; e < {chunk_hd}u; e = e + 1u) {{"
    );
    let _ = writeln!(
        src,
        "                local_o[e] = local_o[e] + p * f32(shared_v[j * {hd}u + d_off + e]);"
    );
    src.push_str("            }\n");
    src.push_str("        }\n");
    src.push_str("        local_sum = local_sum * correction + rowsum;\n");
    src.push_str("        local_max = select(local_max, new_max, q_valid);\n");
    src.push_str("        workgroupBarrier();\n");
    src.push_str("    }\n\n");

    // ---- Tail KV (positions not in a full BKV tile) ----
    src.push_str("    for (; t < max_kv_len; t = t + 1u) {\n");
    src.push_str("        let masked = !(q_valid && t >= row_kv_start && t < row_kv_len);\n");
    // Compute QK for this single position from shared_q (everyone reads the same Q, ok).
    src.push_str("        var dot = 0.0;\n");
    let _ = writeln!(src, "        for (var d = 0u; d < {hd}u; d = d + 1u) {{");
    let _ = writeln!(src, "            let qv = f32(shared_q[row * {hd}u + d]);");
    src.push_str("            let kv = src_b[t * kv_dim + kv_head_off + d];\n");
    src.push_str("            dot = dot + qv * kv;\n");
    src.push_str("        }\n");
    src.push_str("        let score = dot * scale;\n");
    src.push_str("        let new_max = select(local_max, max(local_max, score), !masked);\n");
    src.push_str("        let correction = exp(local_max - new_max);\n");
    src.push_str("        let p = select(exp(score - new_max), 0.0, masked);\n");
    let _ = writeln!(
        src,
        "        for (var e = 0u; e < {chunk_hd}u; e = e + 1u) {{"
    );
    src.push_str("            let v = bias[t * kv_dim + kv_head_off + d_off + e];\n");
    src.push_str("            local_o[e] = local_o[e] * correction + p * v;\n");
    src.push_str("        }\n");
    src.push_str("        local_sum = local_sum * correction + p;\n");
    src.push_str("        local_max = select(local_max, new_max, q_valid);\n");
    src.push_str("    }\n\n");

    // ---- Final write: divide by sum_exp, write output + LSE ----
    src.push_str("    if q_valid {\n");
    src.push_str("        let safe_sum = select(local_sum, 1.0, local_sum == 0.0);\n");
    src.push_str("        let q_base = qpos * (num_heads * head_dim) + head * head_dim;\n");
    let _ = writeln!(
        src,
        "        for (var e = 0u; e < {chunk_hd}u; e = e + 1u) {{"
    );
    src.push_str("            dst[q_base + d_off + e] = local_o[e] / safe_sum;\n");
    src.push_str("        }\n");
    // LSE: only the first chunk-thread per row writes (avoids 4-way duplicate write).
    src.push_str("        if chunk == 0u {\n");
    src.push_str("            let idx = (qpos * num_heads + head) * 2u;\n");
    src.push_str("            lse[idx] = local_max;\n");
    src.push_str("            lse[idx + 1u] = select(log(local_sum), -1e30, local_sum == 0.0);\n");
    src.push_str("        }\n");
    src.push_str("    }\n");
    src.push_str("}\n");

    let module = naga::front::wgsl::parse_str(&src)
        .unwrap_or_else(|e| panic!("generated coop flash WGSL failed to parse:\n{e}\n---\n{src}"));
    ShaderModule {
        module,
        source: src,
    }
}

/// Cooperative-matrix flash backward dQ kernel.
///
/// Three matmuls per KV tile, all coop_mat:
///   1. score = Q @ K^T          (BQxBKV from Q[BQ,hd], K^T[hd,BKV])
///   2. dp    = dO @ V^T         (BQxBKV from dO[BQ,hd], V^T[hd,BKV])
///   3. dQ   += ds @ K * scale   (BQxhd_chunk from ds[BQ,BKV], K[BKV,hd_chunk]),
///      one MMA per hd_chunk → 4 separate `coop_mat<f32,C>` accumulators
///      that live across the entire KV loop.
///
/// Per-row row_sum is precomputed once at the top (sum_d dO[i,d]·O[i,d]).
/// ds = p · (dp − row_sum) where p = exp(score·scale − lse[row]) is
/// computed elementwise in scalar code (1 thread per (row,col) of the
/// 16×16 ds tile = 64 threads · 4 elements/thread).
///
/// Workgroup layout (64 threads = 16 rows × 4 d-chunks):
///   * BQ = BKV = 16 — match coop_mat tile size.
///   * Each workgroup processes one head and 16 query positions, all KV.
///
/// Caller dispatch must use `[ceil(q_seq/16), num_heads, 1]`.
/// `head_dim` must be a multiple of 16.
pub fn generate_flash_grad_q_coop_module(head_dim: u32) -> ShaderModule {
    use std::fmt::Write;
    assert!(
        head_dim >= 16 && head_dim.is_multiple_of(16),
        "coop flash grad_q requires head_dim multiple of 16, got {head_dim}"
    );
    let hd = head_dim;
    let hd_tiles = hd / 16;
    let bq: u32 = 16;
    let bkv: u32 = 16;
    let wg_size: u32 = 64;
    assert!(
        wg_size == 64,
        "coop grad_q assumes wg_size=64 (one warp set)"
    );

    let mut src = String::new();
    src.push_str("enable f16;\n");
    src.push_str("enable wgpu_cooperative_matrix;\n\n");
    src.push_str("struct Params {\n    q_seq: u32,\n    kv_seq: u32,\n    packed_heads: u32,\n    head_dim: u32,\n    window_size: u32,\n    _pad0: u32,\n    _pad1: u32,\n    _pad2: u32,\n}\n\n");
    src.push_str("var<storage> d_out: array<f32>;\n");
    src.push_str("var<storage> src_a: array<f32>;\n"); // Q
    src.push_str("var<storage> src_b: array<f32>;\n"); // K
    src.push_str("var<storage> bias: array<f32>;\n"); // V
    src.push_str("var<storage> lse: array<f32>;\n");
    src.push_str("var<storage> fwd_dst: array<f32>;\n"); // O
    src.push_str("var<storage, read_write> dst: array<f32>;\n"); // dQ
    src.push_str("var<uniform> params: Params;\n\n");

    let _ = writeln!(src, "var<workgroup> shared_q: array<f16, {}>;", bq * hd);
    let _ = writeln!(src, "var<workgroup> shared_do: array<f16, {}>;", bq * hd);
    let _ = writeln!(src, "var<workgroup> shared_k: array<f16, {}>;", bkv * hd);
    let _ = writeln!(src, "var<workgroup> shared_k_t: array<f16, {}>;", hd * bkv);
    let _ = writeln!(src, "var<workgroup> shared_v_t: array<f16, {}>;", hd * bkv);
    let _ = writeln!(
        src,
        "var<workgroup> shared_score: array<f32, {}>;",
        bq * bkv
    );
    let _ = writeln!(src, "var<workgroup> shared_dp: array<f32, {}>;", bq * bkv);
    // shared_ds is f32 instead of f16 — we don't feed it into a coop
    // MMA anymore (dQ uses scalar PV-style accumulation instead) and
    // f32 keeps the per-row scaling tighter.
    let _ = writeln!(src, "var<workgroup> shared_ds: array<f32, {}>;", bq * bkv);
    let _ = writeln!(src, "var<workgroup> wg_row_sum: array<f32, {bq}>;");
    let _ = writeln!(src, "var<workgroup> wg_lse_max: array<f32, {bq}>;");
    let _ = writeln!(src, "var<workgroup> wg_lse_log: array<f32, {bq}>;");
    src.push('\n');

    let _ = writeln!(src, "@compute @workgroup_size({wg_size})");
    src.push_str("fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {\n");
    let _ = writeln!(src, "    let pos_base = wgid.x * {bq}u;");
    src.push_str("    let head = wgid.y;\n");
    src.push_str("    let q_seq = params.q_seq;\n");
    src.push_str("    let kv_seq = params.kv_seq;\n");
    src.push_str("    let num_heads = params.packed_heads >> 16u;\n");
    src.push_str("    let num_kv_heads = params.packed_heads & 0xFFFFu;\n");
    src.push_str("    let head_dim = params.head_dim;\n");
    src.push_str("    let window_size = params.window_size;\n");
    src.push_str("    let kv_head = head / (num_heads / max(num_kv_heads, 1u));\n");
    src.push_str("    let kv_head_off = kv_head * head_dim;\n");
    src.push_str("    let kv_dim = num_kv_heads * head_dim;\n");
    src.push_str("    let scale = inverseSqrt(f32(head_dim));\n\n");

    // Workgroup-wide KV iteration bounds.
    src.push_str("    let last_pos = min(pos_base + 15u, q_seq - 1u);\n");
    src.push_str("    let max_kv_len = select(kv_seq, last_pos + 1u, kv_seq == 0u);\n");
    src.push_str(
        "    let min_kv_start = select(0u, max_kv_len - min(max_kv_len, window_size), window_size > 0u);\n\n",
    );

    // ---- Stage Q + dO into shared (once per workgroup) ----
    let q_total = bq * hd;
    let _ = writeln!(
        src,
        "    for (var i = lid.x; i < {q_total}u; i = i + {wg_size}u) {{"
    );
    let _ = writeln!(src, "        let r = i / {hd}u;");
    let _ = writeln!(src, "        let col = i % {hd}u;");
    src.push_str("        let qp = pos_base + r;\n");
    src.push_str("        if qp < q_seq {\n");
    src.push_str("            let qi = qp * (num_heads * head_dim) + head * head_dim + col;\n");
    src.push_str("            shared_q[i] = f16(src_a[qi]);\n");
    src.push_str("            shared_do[i] = f16(d_out[qi]);\n");
    src.push_str("        } else {\n");
    src.push_str("            shared_q[i] = f16(0.0);\n");
    src.push_str("            shared_do[i] = f16(0.0);\n");
    src.push_str("        }\n");
    src.push_str("    }\n");
    src.push_str("    workgroupBarrier();\n\n");

    // ---- Precompute per-row row_sum = sum_d(dO[r, d] * O[r, d]) and
    //      cache lse{max,log}. 16 threads (one per row).
    let _ = writeln!(src, "    if lid.x < {bq}u {{");
    src.push_str("        let r = lid.x;\n");
    src.push_str("        let qp = pos_base + r;\n");
    src.push_str("        if qp < q_seq && head < num_heads {\n");
    src.push_str("            var s = 0.0;\n");
    src.push_str("            let q_base = qp * (num_heads * head_dim) + head * head_dim;\n");
    let _ = writeln!(
        src,
        "            for (var d = 0u; d < {hd}u; d = d + 1u) {{"
    );
    src.push_str("                s = s + d_out[q_base + d] * fwd_dst[q_base + d];\n");
    src.push_str("            }\n");
    src.push_str("            wg_row_sum[r] = s;\n");
    src.push_str("            let li = (qp * num_heads + head) * 2u;\n");
    src.push_str("            wg_lse_max[r] = lse[li];\n");
    src.push_str("            wg_lse_log[r] = lse[li + 1u];\n");
    src.push_str("        } else {\n");
    src.push_str("            wg_row_sum[r] = 0.0;\n");
    src.push_str("            wg_lse_max[r] = 0.0;\n");
    src.push_str("            wg_lse_log[r] = 0.0;\n");
    src.push_str("        }\n");
    src.push_str("    }\n\n");

    // ---- Per-thread (row, chunk) hoisted indices. Each thread owns
    //      one (row, hd_chunk) of dQ in registers, accumulated across
    //      the entire KV loop.
    let chunks_per_row: u32 = 4;
    let chunk_hd: u32 = hd / chunks_per_row;
    let _ = writeln!(src, "    let row = lid.x / {chunks_per_row}u;");
    let _ = writeln!(src, "    let chunk = lid.x % {chunks_per_row}u;");
    let _ = writeln!(src, "    let d_off = chunk * {chunk_hd}u;");
    src.push_str("    let qpos_thread = pos_base + row;\n");
    src.push_str("    let q_valid_thread = qpos_thread < q_seq && head < num_heads;\n\n");

    // Per-thread dQ accumulator (registers).
    let _ = writeln!(src, "    var local_dq: array<f32, {chunk_hd}>;");
    let _ = writeln!(src, "    for (var e = 0u; e < {chunk_hd}u; e = e + 1u) {{");
    src.push_str("        local_dq[e] = 0.0;\n");
    src.push_str("    }\n\n");

    // ---- KV tile loop ----
    let _ = writeln!(
        src,
        "    let tile_end = min_kv_start + ((max_kv_len - min_kv_start) / {bkv}u) * {bkv}u;"
    );
    src.push_str("    var t = min_kv_start;\n");
    let _ = writeln!(src, "    for (; t < tile_end; t = t + {bkv}u) {{");

    // Stage K both ways and V transposed.
    let kv_total = bkv * hd;
    let _ = writeln!(
        src,
        "        for (var i = lid.x; i < {kv_total}u; i = i + {wg_size}u) {{"
    );
    let _ = writeln!(src, "            let ki = i / {hd}u;");
    let _ = writeln!(src, "            let d  = i % {hd}u;");
    src.push_str("            let kv_pos = t + ki;\n");
    src.push_str("            let k_v = src_b[kv_pos * kv_dim + kv_head_off + d];\n");
    src.push_str("            let v_v = bias[kv_pos * kv_dim + kv_head_off + d];\n");
    src.push_str("            shared_k[i] = f16(k_v);\n");
    let _ = writeln!(src, "            shared_k_t[d * {bkv}u + ki] = f16(k_v);");
    let _ = writeln!(src, "            shared_v_t[d * {bkv}u + ki] = f16(v_v);");
    src.push_str("        }\n");
    src.push_str("        workgroupBarrier();\n\n");

    // score = Q @ K^T (B operand from shared_k_t).
    src.push_str("        var score_acc = coop_mat16x16<f32,C>();\n");
    let _ = writeln!(
        src,
        "        for (var ht = 0u; ht < {hd_tiles}u; ht = ht + 1u) {{"
    );
    let _ = writeln!(
        src,
        "            let a = coopLoadT<coop_mat16x16<f16,A>>(&shared_q[ht * 16u], {hd}u);"
    );
    let _ = writeln!(
        src,
        "            let b = coopLoadT<coop_mat16x16<f16,B>>(&shared_k_t[ht * 16u * {bkv}u], {bkv}u);"
    );
    src.push_str("            score_acc = coopMultiplyAdd(a, b, score_acc);\n");
    src.push_str("        }\n");
    let _ = writeln!(
        src,
        "        coopStoreT(score_acc, &shared_score[0], {bkv}u);"
    );

    // dp = dO @ V^T.
    src.push_str("        var dp_acc = coop_mat16x16<f32,C>();\n");
    let _ = writeln!(
        src,
        "        for (var ht = 0u; ht < {hd_tiles}u; ht = ht + 1u) {{"
    );
    let _ = writeln!(
        src,
        "            let a = coopLoadT<coop_mat16x16<f16,A>>(&shared_do[ht * 16u], {hd}u);"
    );
    let _ = writeln!(
        src,
        "            let b = coopLoadT<coop_mat16x16<f16,B>>(&shared_v_t[ht * 16u * {bkv}u], {bkv}u);"
    );
    src.push_str("            dp_acc = coopMultiplyAdd(a, b, dp_acc);\n");
    src.push_str("        }\n");
    let _ = writeln!(src, "        coopStoreT(dp_acc, &shared_dp[0], {bkv}u);");
    src.push_str("        workgroupBarrier();\n\n");

    // ds = p * (dp - row_sum). 64 threads × 4 elements each = 256 entries
    // (the 16x16 ds tile). Each thread handles 4 consecutive entries.
    let ds_total = bq * bkv;
    let _ = writeln!(src, "        for (var k = 0u; k < 4u; k = k + 1u) {{");
    let _ = writeln!(src, "            let idx = lid.x * 4u + k;");
    let _ = writeln!(src, "            if idx < {ds_total}u {{");
    let _ = writeln!(src, "                let r = idx / {bkv}u;");
    let _ = writeln!(src, "                let j = idx % {bkv}u;");
    src.push_str("                let qpos = pos_base + r;\n");
    src.push_str("                let kv_pos = t + j;\n");
    src.push_str("                let row_kv_len = select(kv_seq, qpos + 1u, kv_seq == 0u);\n");
    src.push_str(
        "                let row_kv_start = select(0u, row_kv_len - min(row_kv_len, window_size), window_size > 0u);\n",
    );
    src.push_str("                let q_valid = qpos < q_seq && head < num_heads;\n");
    src.push_str(
        "                let masked = !(q_valid && kv_pos >= row_kv_start && kv_pos < row_kv_len);\n",
    );
    let _ = writeln!(
        src,
        "                let s = shared_score[r * {bkv}u + j] * scale;"
    );
    src.push_str("                let p = exp(min(s - wg_lse_max[r], 0.0) - wg_lse_log[r]);\n");
    let _ = writeln!(src, "                let dp_v = shared_dp[r * {bkv}u + j];");
    src.push_str("                let ds = select(p * (dp_v - wg_row_sum[r]), 0.0, masked);\n");
    let _ = writeln!(src, "                shared_ds[r * {bkv}u + j] = ds;");
    src.push_str("            }\n");
    src.push_str("        }\n");
    src.push_str("        workgroupBarrier();\n\n");

    // dQ += ds @ K * scale, scalar per-thread accumulation.
    // Each thread (row, chunk) reads its row of shared_ds (16 entries)
    // and shared_k (one chunk per j), accumulates into local_dq.
    let _ = writeln!(src, "        for (var j = 0u; j < {bkv}u; j = j + 1u) {{");
    let _ = writeln!(src, "            let p = shared_ds[row * {bkv}u + j];");
    let _ = writeln!(
        src,
        "            for (var e = 0u; e < {chunk_hd}u; e = e + 1u) {{"
    );
    let _ = writeln!(
        src,
        "                let kv = f32(shared_k[j * {hd}u + d_off + e]);"
    );
    src.push_str("                local_dq[e] = local_dq[e] + p * kv;\n");
    src.push_str("            }\n");
    src.push_str("        }\n");
    src.push_str("        workgroupBarrier();\n");
    src.push_str("    }\n\n");

    // ---- Tail KV (positions outside a full BKV tile) ----
    // Each thread handles its own (row, chunk) — replicated softmax
    // (4-way), unique chunk-of-K accumulation.
    src.push_str("    for (; t < max_kv_len; t = t + 1u) {\n");
    src.push_str("        let row_kv_len = select(kv_seq, qpos_thread + 1u, kv_seq == 0u);\n");
    src.push_str(
        "        let row_kv_start = select(0u, row_kv_len - min(row_kv_len, window_size), window_size > 0u);\n",
    );
    src.push_str(
        "        let masked = !(q_valid_thread && t >= row_kv_start && t < row_kv_len);\n",
    );
    src.push_str("        if !masked {\n");
    src.push_str("            var dot_qk = 0.0;\n");
    src.push_str("            var dot_dov = 0.0;\n");
    let _ = writeln!(
        src,
        "            for (var d = 0u; d < {hd}u; d = d + 1u) {{"
    );
    let _ = writeln!(
        src,
        "                let qv = f32(shared_q[row * {hd}u + d]);"
    );
    let _ = writeln!(
        src,
        "                let dov = f32(shared_do[row * {hd}u + d]);"
    );
    src.push_str("                let kv = src_b[t * kv_dim + kv_head_off + d];\n");
    src.push_str("                let vv = bias[t * kv_dim + kv_head_off + d];\n");
    src.push_str("                dot_qk = dot_qk + qv * kv;\n");
    src.push_str("                dot_dov = dot_dov + dov * vv;\n");
    src.push_str("            }\n");
    src.push_str(
        "            let p = exp(min(dot_qk * scale - wg_lse_max[row], 0.0) - wg_lse_log[row]);\n",
    );
    src.push_str("            let ds = p * (dot_dov - wg_row_sum[row]);\n");
    let _ = writeln!(
        src,
        "            for (var e = 0u; e < {chunk_hd}u; e = e + 1u) {{"
    );
    src.push_str("                let kv = src_b[t * kv_dim + kv_head_off + d_off + e];\n");
    src.push_str("                local_dq[e] = local_dq[e] + ds * kv;\n");
    src.push_str("            }\n");
    src.push_str("        }\n");
    src.push_str("    }\n\n");

    // ---- Final write: scale and store local_dq to global dst ----
    src.push_str("    if q_valid_thread {\n");
    src.push_str("        let dst_row_stride = num_heads * head_dim;\n");
    src.push_str("        let q_base = qpos_thread * dst_row_stride + head * head_dim + d_off;\n");
    let _ = writeln!(
        src,
        "        for (var e = 0u; e < {chunk_hd}u; e = e + 1u) {{"
    );
    src.push_str("            dst[q_base + e] = local_dq[e] * scale;\n");
    src.push_str("        }\n");
    src.push_str("    }\n");
    src.push_str("}\n");

    let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| {
        panic!("generated coop flash grad_q WGSL failed to parse:\n{e}\n---\n{src}")
    });
    ShaderModule {
        module,
        source: src,
    }
}

/// Cooperative-matrix flash backward dK + dV kernel.
///
/// Workgroup processes one head and 16 KV positions, iterating through
/// all queries in tiles of 16. Per Q-tile:
///   1. row_sum[q] = sum_d(dO[q,d]·O[q,d]) precomputed (16 threads).
///   2. score = K @ Q^T  via coop_mat → shared_score[BKV, BQ]
///   3. dp    = V @ dO^T via coop_mat → shared_dp   [BKV, BQ]
///   4. p[kv,q]  = exp(score·scale − lse[q]),
///      ds[kv,q] = p · (dp − row_sum[q])             scalar elementwise
///   5. dV[kv,d] += sum_q(p[kv,q]  · dO[q,d])         scalar per-thread
///   6. dK[kv,d] += sum_q(ds[kv,q] · Q[q,d]) · scale  scalar per-thread
///
/// dV and dK accumulate in per-thread registers (chunk_hd=16 each)
/// across the entire query loop — same design as the forward and
/// GradQ coop kernels (the alternative `coop_mat` accumulator
/// spanning the loop hits a naga / shared-memory race).
///
/// Workgroup layout (64 threads = 16 kv-rows × 4 d-chunks).
/// Caller dispatch must use `[ceil(dispatch_kv/16), num_kv_heads, 1]`.
/// `head_dim` must be a multiple of 16.
pub fn generate_flash_grad_kv_coop_module(head_dim: u32) -> ShaderModule {
    use std::fmt::Write;
    assert!(
        head_dim >= 16 && head_dim.is_multiple_of(16),
        "coop flash grad_kv requires head_dim multiple of 16, got {head_dim}"
    );
    let hd = head_dim;
    let hd_tiles = hd / 16;
    let bq: u32 = 16;
    let bkv: u32 = 16;
    let wg_size: u32 = 64;
    let chunks_per_row: u32 = 4;
    let chunk_hd: u32 = hd / chunks_per_row;

    let mut src = String::new();
    src.push_str("enable f16;\n");
    src.push_str("enable wgpu_cooperative_matrix;\n\n");
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

    let _ = writeln!(src, "var<workgroup> shared_k: array<f16, {}>;", bkv * hd);
    let _ = writeln!(src, "var<workgroup> shared_v: array<f16, {}>;", bkv * hd);
    let _ = writeln!(src, "var<workgroup> shared_q: array<f16, {}>;", bq * hd);
    let _ = writeln!(src, "var<workgroup> shared_q_t: array<f16, {}>;", hd * bq);
    let _ = writeln!(src, "var<workgroup> shared_do: array<f16, {}>;", bq * hd);
    let _ = writeln!(src, "var<workgroup> shared_do_t: array<f16, {}>;", hd * bq);
    let _ = writeln!(
        src,
        "var<workgroup> shared_score: array<f32, {}>;",
        bkv * bq
    );
    let _ = writeln!(src, "var<workgroup> shared_dp: array<f32, {}>;", bkv * bq);
    let _ = writeln!(src, "var<workgroup> shared_p: array<f32, {}>;", bkv * bq);
    let _ = writeln!(src, "var<workgroup> shared_ds: array<f32, {}>;", bkv * bq);
    let _ = writeln!(src, "var<workgroup> wg_row_sum: array<f32, {bq}>;");
    let _ = writeln!(src, "var<workgroup> wg_lse_max: array<f32, {bq}>;");
    let _ = writeln!(src, "var<workgroup> wg_lse_log: array<f32, {bq}>;");
    src.push('\n');

    let _ = writeln!(src, "@compute @workgroup_size({wg_size})");
    src.push_str("fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {\n");
    let _ = writeln!(src, "    let kv_base = wgid.x * {bkv}u;");
    src.push_str("    let kv_head = wgid.y;\n");
    src.push_str("    let q_seq = params.q_seq;\n");
    src.push_str("    let kv_seq = params.kv_seq;\n");
    src.push_str("    let num_heads = params.packed_heads >> 16u;\n");
    src.push_str("    let num_kv_heads = params.packed_heads & 0xFFFFu;\n");
    src.push_str("    let head_dim = params.head_dim;\n");
    src.push_str("    let window_size = params.window_size;\n");
    src.push_str("    let heads_per_kv = num_heads / max(num_kv_heads, 1u);\n");
    src.push_str("    let kv_head_off = kv_head * head_dim;\n");
    src.push_str("    let kv_dim = num_kv_heads * head_dim;\n");
    src.push_str("    let q_dim = num_heads * head_dim;\n");
    src.push_str("    let scale = inverseSqrt(f32(head_dim));\n");
    src.push_str("    let effective_kv_seq = select(kv_seq, q_seq, kv_seq == 0u);\n\n");

    // Per-thread (kv_row, chunk) hoisted indices.
    let _ = writeln!(src, "    let kv_row = lid.x / {chunks_per_row}u;");
    let _ = writeln!(src, "    let chunk = lid.x % {chunks_per_row}u;");
    let _ = writeln!(src, "    let d_off = chunk * {chunk_hd}u;");
    src.push_str("    let kv_pos_thread = kv_base + kv_row;\n");
    src.push_str(
        "    let kv_valid_thread = kv_pos_thread < effective_kv_seq && kv_head < num_kv_heads;\n\n",
    );

    // Per-thread dV and dK accumulators (registers).
    let _ = writeln!(src, "    var local_dv: array<f32, {chunk_hd}>;");
    let _ = writeln!(src, "    var local_dk: array<f32, {chunk_hd}>;");
    let _ = writeln!(src, "    for (var e = 0u; e < {chunk_hd}u; e = e + 1u) {{");
    src.push_str("        local_dv[e] = 0.0;\n");
    src.push_str("        local_dk[e] = 0.0;\n");
    src.push_str("    }\n\n");

    // ---- Stage K and V once per workgroup ----
    let kv_total = bkv * hd;
    let _ = writeln!(
        src,
        "    for (var i = lid.x; i < {kv_total}u; i = i + {wg_size}u) {{"
    );
    let _ = writeln!(src, "        let ki = i / {hd}u;");
    let _ = writeln!(src, "        let d  = i % {hd}u;");
    src.push_str("        let kp = kv_base + ki;\n");
    src.push_str("        if kp < effective_kv_seq {\n");
    src.push_str("            let kv_off = kp * kv_dim + kv_head_off + d;\n");
    src.push_str("            shared_k[i] = f16(src_b[kv_off]);\n");
    src.push_str("            shared_v[i] = f16(bias[kv_off]);\n");
    src.push_str("        } else {\n");
    src.push_str("            shared_k[i] = f16(0.0);\n");
    src.push_str("            shared_v[i] = f16(0.0);\n");
    src.push_str("        }\n");
    src.push_str("    }\n");
    src.push_str("    workgroupBarrier();\n\n");

    // GQA: this kernel processes one KV head, but multiple Q heads
    // map to it. Iterate q-heads (heads_per_kv) and within each, all
    // queries in BQ tiles. Outer loop is q-head, inner loop is q-pos.
    src.push_str("    for (var qh = 0u; qh < heads_per_kv; qh = qh + 1u) {\n");
    src.push_str("        let q_head = kv_head * heads_per_kv + qh;\n\n");

    // Per-row valid Q range (causal: only q >= kv_pos can attend; i.e.
    // for KV position kv_pos_thread, valid q_pos in [kv_pos_thread, q_seq))
    src.push_str("        let row_q_start = select(0u, kv_pos_thread, kv_seq == 0u);\n");
    // For sliding window: q must be in (kv_pos_thread - window, kv_pos_thread]
    // — implemented elementwise inside the softmax phase below.

    // ---- Q-tile loop ----
    let _ = writeln!(src, "        let tile_end = (q_seq / {bq}u) * {bq}u;");
    src.push_str("        var t = 0u;\n");
    let _ = writeln!(src, "        for (; t < tile_end; t = t + {bq}u) {{");

    // Stage Q, dO, O for this tile (and their transposes for coop).
    let q_total = bq * hd;
    let _ = writeln!(
        src,
        "            for (var i = lid.x; i < {q_total}u; i = i + {wg_size}u) {{"
    );
    let _ = writeln!(src, "                let qi = i / {hd}u;");
    let _ = writeln!(src, "                let d  = i % {hd}u;");
    src.push_str("                let qp = t + qi;\n");
    src.push_str("                if qp < q_seq {\n");
    src.push_str("                    let q_off = qp * q_dim + q_head * head_dim + d;\n");
    src.push_str("                    let qv = src_a[q_off];\n");
    src.push_str("                    let dov = d_out[q_off];\n");
    src.push_str("                    shared_q[i] = f16(qv);\n");
    src.push_str("                    shared_do[i] = f16(dov);\n");
    let _ = writeln!(
        src,
        "                    shared_q_t[d * {bq}u + qi] = f16(qv);"
    );
    let _ = writeln!(
        src,
        "                    shared_do_t[d * {bq}u + qi] = f16(dov);"
    );
    src.push_str("                } else {\n");
    src.push_str("                    shared_q[i] = f16(0.0);\n");
    src.push_str("                    shared_do[i] = f16(0.0);\n");
    let _ = writeln!(
        src,
        "                    shared_q_t[d * {bq}u + qi] = f16(0.0);"
    );
    let _ = writeln!(
        src,
        "                    shared_do_t[d * {bq}u + qi] = f16(0.0);"
    );
    src.push_str("                }\n");
    src.push_str("            }\n");
    src.push_str("            workgroupBarrier();\n\n");

    // Precompute row_sum[qi] and lse cache (16 threads).
    let _ = writeln!(src, "            if lid.x < {bq}u {{");
    src.push_str("                let qi = lid.x;\n");
    src.push_str("                let qp = t + qi;\n");
    src.push_str("                if qp < q_seq && q_head < num_heads {\n");
    src.push_str("                    var s = 0.0;\n");
    src.push_str("                    let q_base = qp * q_dim + q_head * head_dim;\n");
    let _ = writeln!(
        src,
        "                    for (var d = 0u; d < {hd}u; d = d + 1u) {{"
    );
    src.push_str("                        s = s + d_out[q_base + d] * fwd_dst[q_base + d];\n");
    src.push_str("                    }\n");
    src.push_str("                    wg_row_sum[qi] = s;\n");
    src.push_str("                    let li = (qp * num_heads + q_head) * 2u;\n");
    src.push_str("                    wg_lse_max[qi] = lse[li];\n");
    src.push_str("                    wg_lse_log[qi] = lse[li + 1u];\n");
    src.push_str("                } else {\n");
    src.push_str("                    wg_row_sum[qi] = 0.0;\n");
    src.push_str("                    wg_lse_max[qi] = 0.0;\n");
    src.push_str("                    wg_lse_log[qi] = 0.0;\n");
    src.push_str("                }\n");
    src.push_str("            }\n");
    src.push_str("            workgroupBarrier();\n\n");

    // score = K @ Q^T (BKV x BQ).
    src.push_str("            var score_acc = coop_mat16x16<f32,C>();\n");
    let _ = writeln!(
        src,
        "            for (var ht = 0u; ht < {hd_tiles}u; ht = ht + 1u) {{"
    );
    let _ = writeln!(
        src,
        "                let a_k = coopLoadT<coop_mat16x16<f16,A>>(&shared_k[ht * 16u], {hd}u);"
    );
    let _ = writeln!(
        src,
        "                let b_qt = coopLoadT<coop_mat16x16<f16,B>>(&shared_q_t[ht * 16u * {bq}u], {bq}u);"
    );
    src.push_str("                score_acc = coopMultiplyAdd(a_k, b_qt, score_acc);\n");
    src.push_str("            }\n");
    let _ = writeln!(
        src,
        "            coopStoreT(score_acc, &shared_score[0], {bq}u);"
    );

    // dp = V @ dO^T (BKV x BQ).
    src.push_str("            var dp_acc = coop_mat16x16<f32,C>();\n");
    let _ = writeln!(
        src,
        "            for (var ht = 0u; ht < {hd_tiles}u; ht = ht + 1u) {{"
    );
    let _ = writeln!(
        src,
        "                let a_v = coopLoadT<coop_mat16x16<f16,A>>(&shared_v[ht * 16u], {hd}u);"
    );
    let _ = writeln!(
        src,
        "                let b_dot = coopLoadT<coop_mat16x16<f16,B>>(&shared_do_t[ht * 16u * {bq}u], {bq}u);"
    );
    src.push_str("                dp_acc = coopMultiplyAdd(a_v, b_dot, dp_acc);\n");
    src.push_str("            }\n");
    let _ = writeln!(src, "            coopStoreT(dp_acc, &shared_dp[0], {bq}u);");
    src.push_str("            workgroupBarrier();\n\n");

    // p[kv, q] = exp(score * scale - lse[q]); ds[kv, q] = p * (dp - row_sum[q]).
    // 64 threads × 4 entries each = 256 = full BKV*BQ tile.
    let pq_total = bkv * bq;
    src.push_str("            for (var k = 0u; k < 4u; k = k + 1u) {\n");
    let _ = writeln!(src, "                let idx = lid.x * 4u + k;");
    let _ = writeln!(src, "                if idx < {pq_total}u {{");
    let _ = writeln!(src, "                    let kv = idx / {bq}u;");
    let _ = writeln!(src, "                    let q  = idx % {bq}u;");
    src.push_str("                    let kp = kv_base + kv;\n");
    src.push_str("                    let qp = t + q;\n");
    src.push_str(
        "                    let masked = !(kp < effective_kv_seq && kv_head < num_kv_heads && qp < q_seq && q_head < num_heads);\n",
    );
    // Causal: qp >= kp. Sliding window: qp - window < kp <= qp.
    src.push_str("                    let row_kv_len = select(kv_seq, qp + 1u, kv_seq == 0u);\n");
    src.push_str(
        "                    let row_kv_start = select(0u, row_kv_len - min(row_kv_len, window_size), window_size > 0u);\n",
    );
    src.push_str(
        "                    let attn_masked = masked || (kp < row_kv_start) || (kp >= row_kv_len);\n",
    );
    let _ = writeln!(
        src,
        "                    let s = shared_score[kv * {bq}u + q] * scale;"
    );
    src.push_str("                    let p = exp(min(s - wg_lse_max[q], 0.0) - wg_lse_log[q]);\n");
    let _ = writeln!(
        src,
        "                    let dp_v = shared_dp[kv * {bq}u + q];"
    );
    src.push_str("                    let ds = p * (dp_v - wg_row_sum[q]);\n");
    src.push_str("                    let p_safe = select(p, 0.0, attn_masked);\n");
    src.push_str("                    let ds_safe = select(ds, 0.0, attn_masked);\n");
    let _ = writeln!(
        src,
        "                    shared_p[kv * {bq}u + q] = p_safe;"
    );
    let _ = writeln!(
        src,
        "                    shared_ds[kv * {bq}u + q] = ds_safe;"
    );
    src.push_str("                }\n");
    src.push_str("            }\n");
    src.push_str("            workgroupBarrier();\n\n");

    // dV[kv,d] += sum_q(p[kv,q] · dO[q,d])  scalar
    // dK[kv,d] += sum_q(ds[kv,q] · Q[q,d])  scalar (scale applied at final write)
    let _ = writeln!(
        src,
        "            for (var q = 0u; q < {bq}u; q = q + 1u) {{"
    );
    let _ = writeln!(src, "                let p = shared_p[kv_row * {bq}u + q];");
    let _ = writeln!(
        src,
        "                let ds = shared_ds[kv_row * {bq}u + q];"
    );
    let _ = writeln!(
        src,
        "                for (var e = 0u; e < {chunk_hd}u; e = e + 1u) {{"
    );
    let _ = writeln!(
        src,
        "                    let dov = f32(shared_do[q * {hd}u + d_off + e]);"
    );
    let _ = writeln!(
        src,
        "                    let qv  = f32(shared_q [q * {hd}u + d_off + e]);"
    );
    src.push_str("                    local_dv[e] = local_dv[e] + p  * dov;\n");
    src.push_str("                    local_dk[e] = local_dk[e] + ds * qv;\n");
    src.push_str("                }\n");
    src.push_str("            }\n");
    src.push_str("            workgroupBarrier();\n");
    src.push_str("        }\n\n");

    // Tail Q (positions outside a full BQ tile) — scalar fallback.
    src.push_str("        for (; t < q_seq; t = t + 1u) {\n");
    src.push_str("            let masked = !(kv_valid_thread && q_head < num_heads);\n");
    src.push_str("            if !masked {\n");
    src.push_str("                let row_kv_len = select(kv_seq, t + 1u, kv_seq == 0u);\n");
    src.push_str(
        "                let row_kv_start = select(0u, row_kv_len - min(row_kv_len, window_size), window_size > 0u);\n",
    );
    src.push_str(
        "                let attn_masked = (kv_pos_thread < row_kv_start) || (kv_pos_thread >= row_kv_len);\n",
    );
    src.push_str("                if !attn_masked {\n");
    src.push_str("                    var dot_qk = 0.0;\n");
    src.push_str("                    var dot_dov = 0.0;\n");
    src.push_str("                    var dot_doo = 0.0;\n");
    let _ = writeln!(
        src,
        "                    for (var d = 0u; d < {hd}u; d = d + 1u) {{"
    );
    src.push_str(
        "                        let kv = src_b[kv_pos_thread * kv_dim + kv_head_off + d];\n",
    );
    src.push_str(
        "                        let vv = bias[kv_pos_thread * kv_dim + kv_head_off + d];\n",
    );
    src.push_str("                        let qv = src_a[t * q_dim + q_head * head_dim + d];\n");
    src.push_str("                        let dov = d_out[t * q_dim + q_head * head_dim + d];\n");
    src.push_str("                        let ov = fwd_dst[t * q_dim + q_head * head_dim + d];\n");
    src.push_str("                        dot_qk = dot_qk + qv * kv;\n");
    src.push_str("                        dot_dov = dot_dov + dov * vv;\n");
    src.push_str("                        dot_doo = dot_doo + dov * ov;\n");
    src.push_str("                    }\n");
    src.push_str("                    let li = (t * num_heads + q_head) * 2u;\n");
    src.push_str("                    let lmax = lse[li];\n");
    src.push_str("                    let llog = lse[li + 1u];\n");
    src.push_str("                    let p = exp(min(dot_qk * scale - lmax, 0.0) - llog);\n");
    src.push_str("                    let ds = p * (dot_dov - dot_doo);\n");
    let _ = writeln!(
        src,
        "                    for (var e = 0u; e < {chunk_hd}u; e = e + 1u) {{"
    );
    src.push_str(
        "                        let dov = d_out[t * q_dim + q_head * head_dim + d_off + e];\n",
    );
    src.push_str(
        "                        let qv = src_a[t * q_dim + q_head * head_dim + d_off + e];\n",
    );
    src.push_str("                        local_dv[e] = local_dv[e] + p  * dov;\n");
    src.push_str("                        local_dk[e] = local_dk[e] + ds * qv;\n");
    src.push_str("                    }\n");
    src.push_str("                }\n");
    src.push_str("            }\n");
    src.push_str("        }\n");
    src.push_str("    }\n\n"); // end qh loop

    // ---- Final write: scale (only dK), store to global dK / dV ----
    src.push_str("    if kv_valid_thread {\n");
    src.push_str("        let kv_dst_off = kv_pos_thread * kv_dim + kv_head_off + d_off;\n");
    let _ = writeln!(
        src,
        "        for (var e = 0u; e < {chunk_hd}u; e = e + 1u) {{"
    );
    src.push_str("            dst[kv_dst_off + e] = local_dk[e] * scale;\n");
    src.push_str("            dst2[kv_dst_off + e] = local_dv[e];\n");
    src.push_str("        }\n");
    src.push_str("    }\n");
    src.push_str("}\n");

    let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| {
        panic!("generated coop flash grad_kv WGSL failed to parse:\n{e}\n---\n{src}")
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
    // Per-kernel EPT picked by `flash_ept_for(FlashKernel::GradQ, ..)`.
    let ept: u32 = flash_ept_for(FlashKernel::GradQ, hd);
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
    // Per-kernel EPT picked by `flash_ept_for(FlashKernel::GradKv, ..)`.
    // The fused dK+dV kernel reports 210 regs at EPT=32 on Blackwell,
    // so the auto-tune typically chooses a smaller value here.
    let ept: u32 = flash_ept_for(FlashKernel::GradKv, hd);
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

    // Shared Q/dO/O staging: only needed when TPQ > 1 (multiple threads
    // per KV position need coordinated access). When TPQ == 1, each thread
    // loads Q/dO/O directly from global memory — all threads read the same
    // addresses, hitting L2 cache, and no barriers are needed.
    if tpq > 1 {
        let _ = writeln!(src, "var<workgroup> shared_q: array<f32, {hd}>;");
        let _ = writeln!(src, "var<workgroup> shared_do: array<f32, {hd}>;");
        let _ = writeln!(src, "var<workgroup> shared_o: array<f32, {hd}>;\n");
    }

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

    // Inner Q loop: iterate all Q positions that attend to this KV position.
    src.push_str("    for (var pos = wg_start; pos < wg_end; pos++) {\n");
    src.push_str("        for (var head_rel = 0u; head_rel < heads_per_kv; head_rel++) {\n");
    src.push_str("            let head = kv_head * heads_per_kv + head_rel;\n");
    src.push_str("            let q_base = pos * q_dim + head * head_dim;\n\n");

    if tpq == 1 {
        // TPQ=1: each thread handles the full head_dim. Load Q/dO/O directly
        // from global memory — all threads read the same addresses, hitting L2.
        // This eliminates ALL barriers in the inner loop.
        for e in 0..ept {
            let _ = writeln!(src, "            let q{e} = src_a[q_base + {e}u];");
            let _ = writeln!(src, "            let do{e} = d_out[q_base + {e}u];");
            let _ = writeln!(src, "            let o{e} = fwd_dst[q_base + {e}u];");
        }
    } else {
        // TPQ>1: cooperative staging into shared memory (needs barriers).
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
    if tpq > 1 {
        src.push_str("            workgroupBarrier();\n");
    }
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

/// Which side of the KV gradient the split codegen produces.
#[derive(Clone, Copy, PartialEq, Eq)]
enum FlashKvKind {
    K,
    V,
}

/// Shared codegen for the *split* flash-attention KV-gradient kernels.
///
/// The fused `generate_flash_grad_kv_module` reports 210 regs/thread on
/// Blackwell (1 wg/SM). Splitting drops register pressure:
///   * `K` variant: ~158 regs (drops per-thread dV accumulator).
///   * `V` variant: ~109 regs (drops V/O loads and row_sum/dp_t reductions,
///     getting 2 wg/SM).
///
/// The split is not a universal win: each kernel independently recomputes
/// `score = Q·K · scale` in its inner loop, which doubles compute on
/// work-bound shapes (e.g. Whisper's non-causal seq=1500 encoder, where
/// the inner Q loop dominates). Dispatch-size-aware selection between
/// split and fused belongs in the e-graph cost model.
///
/// Both split kernels use the `MultiHeadAttnGradData` binding layout.
fn gen_flash_grad_kv_split_impl(head_dim: u32, kind: FlashKvKind) -> ShaderModule {
    use std::fmt::Write;
    assert!(head_dim.is_power_of_two() && head_dim >= 2);

    let hd = head_dim;
    // Per-kernel EPT — split kernels each have their own slot.
    let kernel = match kind {
        FlashKvKind::K => FlashKernel::GradK,
        FlashKvKind::V => FlashKernel::GradV,
    };
    let ept: u32 = flash_ept_for(kernel, hd);
    let tpq = hd / ept;
    let bkv: u32 = (256 / tpq).max(1);
    assert!(
        bkv > 1,
        "flash grad kv split assumes bkv>1 (head_dim {hd} too large)",
    );
    let need_v = kind == FlashKvKind::K; // V only needed for dp_t in dK pass.
    let need_o = kind == FlashKvKind::K; // O only needed for row_sum in dK pass.
    let wg_size = bkv * tpq;
    let mut src = String::new();

    src.push_str("struct Params {\n    q_seq: u32,\n    kv_seq: u32,\n    packed_heads: u32,\n    head_dim: u32,\n    window_size: u32,\n    _pad0: u32,\n    _pad1: u32,\n    _pad2: u32,\n}\n\n");
    src.push_str("var<storage> d_out: array<f32>;\n");
    src.push_str("var<storage> src_a: array<f32>;\n"); // Q
    src.push_str("var<storage> src_b: array<f32>;\n"); // K
    src.push_str("var<storage> bias: array<f32>;\n"); // V (unused in V kind, but in binding)
    src.push_str("var<storage> lse: array<f32>;\n");
    src.push_str("var<storage> fwd_dst: array<f32>;\n"); // O (unused in V kind)
    src.push_str("var<storage, read_write> dst: array<f32>;\n"); // dK or dV
    src.push_str("var<uniform> params: Params;\n\n");

    if tpq > 1 {
        let _ = writeln!(src, "var<workgroup> shared_q: array<f32, {hd}>;");
        let _ = writeln!(src, "var<workgroup> shared_do: array<f32, {hd}>;");
        if need_o {
            let _ = writeln!(src, "var<workgroup> shared_o: array<f32, {hd}>;");
        }
        src.push('\n');

        let _ = writeln!(src, "var<workgroup> wg_score: array<f32, {}>;", bkv * tpq);
        if kind == FlashKvKind::K {
            let _ = writeln!(src, "var<workgroup> wg_rs: array<f32, {}>;", bkv * tpq);
            let _ = writeln!(src, "var<workgroup> wg_dp: array<f32, {}>;", bkv * tpq);
        }
        src.push('\n');

        match kind {
            FlashKvKind::K => {
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
            FlashKvKind::V => {
                src.push_str("fn reduce_score(tid: u32) {\n");
                let _ = writeln!(src, "    let local = tid % {tpq}u;");
                let _ = writeln!(src, "    let base = (tid / {tpq}u) * {tpq}u;");
                let mut stride = tpq / 2;
                while stride > 0 {
                    src.push_str("    workgroupBarrier();\n");
                    let _ = writeln!(
                        src,
                        "    if local < {stride}u {{ wg_score[base + local] += wg_score[base + local + {stride}u]; }}"
                    );
                    stride /= 2;
                }
                src.push_str("    workgroupBarrier();\n}\n\n");
            }
        }
    }

    let _ = writeln!(src, "@compute @workgroup_size({wg_size})");
    src.push_str("fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {\n");
    let _ = writeln!(src, "    let ki = lid.x / {tpq}u;");
    let _ = writeln!(src, "    let lane = lid.x % {tpq}u;");
    let _ = writeln!(src, "    let d_base = lane * {ept}u;");
    let _ = writeln!(src, "    let t = wgid.x * {bkv}u + ki;");
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

    for e in 0..ept {
        let _ = writeln!(src, "    var k{e} = 0.0;");
        if need_v {
            let _ = writeln!(src, "    var v{e} = 0.0;");
        }
    }
    src.push_str("    if valid {\n");
    for e in 0..ept {
        let _ = writeln!(src, "        k{e} = src_b[kv_base + d_base + {e}u];");
        if need_v {
            let _ = writeln!(src, "        v{e} = bias[kv_base + d_base + {e}u];");
        }
    }
    src.push_str("    }\n\n");

    let acc = match kind {
        FlashKvKind::K => "dk",
        FlashKvKind::V => "dv",
    };
    for e in 0..ept {
        let _ = writeln!(src, "    var {acc}{e} = 0.0;");
    }
    src.push('\n');

    src.push_str("    let start_pos = select(0u, t, kv_seq == 0u);\n");
    src.push_str("    let window = params.window_size;\n");
    src.push_str("    let end_pos = select(q_seq, min(q_seq, t + window), window > 0u);\n\n");

    let _ = writeln!(src, "    let first_t = wgid.x * {bkv}u;");
    let _ = writeln!(
        src,
        "    let last_t = min(wgid.x * {bkv}u + {bkv}u - 1u, effective_kv_seq - 1u);"
    );
    src.push_str("    let wg_start = select(0u, first_t, kv_seq == 0u);\n");
    src.push_str("    let wg_end = select(q_seq, min(q_seq, last_t + window), window > 0u);\n\n");

    src.push_str("    for (var pos = wg_start; pos < wg_end; pos++) {\n");
    src.push_str("        for (var head_rel = 0u; head_rel < heads_per_kv; head_rel++) {\n");
    src.push_str("            let head = kv_head * heads_per_kv + head_rel;\n");
    src.push_str("            let q_base = pos * q_dim + head * head_dim;\n\n");

    if tpq == 1 {
        for e in 0..ept {
            let _ = writeln!(src, "            let q{e} = src_a[q_base + {e}u];");
            let _ = writeln!(src, "            let do{e} = d_out[q_base + {e}u];");
            if need_o {
                let _ = writeln!(src, "            let o{e} = fwd_dst[q_base + {e}u];");
            }
        }
    } else {
        let _ = writeln!(src, "            if lid.x < {hd}u {{");
        src.push_str("                shared_q[lid.x] = src_a[q_base + lid.x];\n");
        src.push_str("                shared_do[lid.x] = d_out[q_base + lid.x];\n");
        if need_o {
            src.push_str("                shared_o[lid.x] = fwd_dst[q_base + lid.x];\n");
        }
        src.push_str("            }\n");
        src.push_str("            workgroupBarrier();\n\n");

        for e in 0..ept {
            let _ = writeln!(src, "            let q{e} = shared_q[d_base + {e}u];");
            let _ = writeln!(src, "            let do{e} = shared_do[d_base + {e}u];");
            if need_o {
                let _ = writeln!(src, "            let o{e} = shared_o[d_base + {e}u];");
            }
        }
    }
    src.push_str("            var score_part = 0.0;\n");
    if kind == FlashKvKind::K {
        src.push_str("            var rs_part = 0.0;\n");
        src.push_str("            var dp_part = 0.0;\n");
    }
    for e in 0..ept {
        let _ = writeln!(src, "            score_part += q{e} * k{e};");
        if kind == FlashKvKind::K {
            let _ = writeln!(src, "            rs_part += do{e} * o{e};");
            let _ = writeln!(src, "            dp_part += do{e} * v{e};");
        }
    }
    if tpq > 1 {
        let _ = writeln!(
            src,
            "            wg_score[ki * {tpq}u + lane] = score_part;"
        );
        if kind == FlashKvKind::K {
            let _ = writeln!(src, "            wg_rs[ki * {tpq}u + lane] = rs_part;");
            let _ = writeln!(src, "            wg_dp[ki * {tpq}u + lane] = dp_part;");
            src.push_str("            reduce_triple(lid.x);\n");
        } else {
            src.push_str("            reduce_score(lid.x);\n");
        }
        let _ = writeln!(
            src,
            "            let score = wg_score[ki * {tpq}u] * scale;"
        );
        if kind == FlashKvKind::K {
            let _ = writeln!(src, "            let row_sum = wg_rs[ki * {tpq}u];");
            let _ = writeln!(src, "            let dp_t = wg_dp[ki * {tpq}u];");
        }
        src.push('\n');
    } else {
        src.push_str("            let score = score_part * scale;\n");
        if kind == FlashKvKind::K {
            src.push_str("            let row_sum = rs_part;\n");
            src.push_str("            let dp_t = dp_part;\n");
        }
    }
    src.push_str("            if valid && pos >= start_pos && pos < end_pos {\n");
    src.push_str("                let lse_idx = (pos * num_heads + head) * 2u;\n");
    src.push_str(
        "                let p_t = exp(min(score - lse[lse_idx], 0.0) - lse[lse_idx + 1u]);\n",
    );
    match kind {
        FlashKvKind::K => {
            src.push_str("                let ds_t = p_t * (dp_t - row_sum);\n");
            src.push_str("                let w_dk = ds_t * scale;\n");
            for e in 0..ept {
                let _ = writeln!(src, "                dk{e} += w_dk * q{e};");
            }
        }
        FlashKvKind::V => {
            for e in 0..ept {
                let _ = writeln!(src, "                dv{e} += p_t * do{e};");
            }
        }
    }
    src.push_str("            }\n");
    if tpq > 1 {
        src.push_str("            workgroupBarrier();\n");
    }
    src.push_str("        }\n");
    src.push_str("    }\n\n");

    src.push_str("    if valid {\n");
    for e in 0..ept {
        let _ = writeln!(src, "        dst[kv_base + d_base + {e}u] = {acc}{e};");
    }
    src.push_str("    }\n");
    src.push_str("}\n");

    let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| {
        panic!("generated flash grad_{acc} WGSL failed to parse:\n{e}\n---\n{src}")
    });
    ShaderModule {
        module,
        source: src,
    }
}

/// Split flash-attention backward: dK-only kernel.
///
/// Loads K, V, Q, dO, O; writes dK. Register pressure ≈ 158 on Blackwell
/// (RTX 5080, head_dim=64). See [`gen_flash_grad_kv_split_impl`] for the
/// tradeoff vs the fused `generate_flash_grad_kv_module`.
pub fn generate_flash_grad_k_module(head_dim: u32) -> ShaderModule {
    gen_flash_grad_kv_split_impl(head_dim, FlashKvKind::K)
}

/// Split flash-attention backward: dV-only kernel.
///
/// Loads K, Q, dO; writes dV. Register pressure ≈ 109 on Blackwell
/// (RTX 5080, head_dim=64), unlocking 2 wg/SM. Skips V/O loads and
/// the row_sum/dp reductions since dV = p · dO needs neither.
pub fn generate_flash_grad_v_module(head_dim: u32) -> ShaderModule {
    gen_flash_grad_kv_split_impl(head_dim, FlashKvKind::V)
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

/// Conv2d cooperative-matrix GEMM direction.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Conv2dCoopDirection {
    /// Forward: C[Co, oH*oW] = A[Co, K] * B[K, oH*oW], K = Ci*kH*kW
    Forward,
    /// Backward w.r.t. input: C[Ci, H*W] = A[Ci, K] * B[K, H*W], K = Co*kH*kW
    GradInput,
}

/// Generate a specialized conv2d cooperative-matrix GEMM kernel with
/// compile-time kernel size and stride constants.
///
/// When `kernel_h`, `kernel_w`, and `stride` are baked into the WGSL as
/// constants, the SPIR-V compiler can constant-fold the im2col index
/// decomposition (divisions become constant-divisor ops) and eliminate
/// dead stride branches entirely.
///
/// The generated shader uses the SAME bindings and Params struct as
/// the template-based variants (`conv2d_gemm_coop.wgsl` and
/// `conv2d_grad_input_gemm_coop.wgsl`) so the runtime bind/dispatch
/// code needs no changes.
pub fn generate_conv2d_coop_module(
    kernel_h: u32,
    kernel_w: u32,
    stride: u32,
    direction: Conv2dCoopDirection,
    config: &CoopConfig,
) -> ShaderModule {
    use std::fmt::Write;

    let tile = config.tile_size;
    let output_tile = config.output_tile();
    let shared_size = tile * tile;
    let wg_size: u32 = 64;
    let staging_iters = shared_size / wg_size;
    let row_stride = wg_size / tile;
    let tile_mask = tile - 1;
    let tile_shift = tile.trailing_zeros();

    let (elem_type, enable_f16, elem_zero, cast_open, cast_close) = if config.use_f16_input {
        ("f16", "enable f16;\n", "f16(0.0)", "f16(", ")")
    } else {
        ("f32", "", "0.0", "", "")
    };
    let ab_type = if config.use_f16_input { "f16" } else { "f32" };
    let coop_ab = format!("coop_mat{tile}x{tile}<{ab_type},A>");
    let coop_ba = format!("coop_mat{tile}x{tile}<{ab_type},B>");
    let coop_c = format!("coop_mat{tile}x{tile}<f32,C>");

    let kernel_hw = kernel_h * kernel_w;
    let backward = direction == Conv2dCoopDirection::GradInput;

    let mut src = String::with_capacity(8192);

    // Header
    let dir_str = if backward {
        "backward (grad_input)"
    } else {
        "forward"
    };
    let _ = writeln!(
        src,
        "// Conv2d {dir_str} via implicit GEMM — cooperative matrix variant."
    );
    let _ = writeln!(
        src,
        "// Specialized for {kernel_h}x{kernel_w} stride-{stride} convolutions."
    );
    src.push('\n');
    src.push_str(enable_f16);
    src.push_str("enable wgpu_cooperative_matrix;\n\n");

    // Params struct — identical layout to Conv2dParams for binding compatibility.
    // kernel_h/kernel_w/stride fields are still present but ignored in favor of constants.
    src.push_str(
        "struct Params {\n\
         \x20   batch: u32,\n\
         \x20   in_channels: u32,\n\
         \x20   in_h: u32,\n\
         \x20   in_w: u32,\n\
         \x20   out_channels: u32,\n\
         \x20   kernel_h: u32,\n\
         \x20   kernel_w: u32,\n\
         \x20   stride: u32,\n\
         \x20   padding_h: u32,\n\
         \x20   out_h: u32,\n\
         \x20   out_w: u32,\n\
         \x20   padding_w: u32,\n\
         \x20   inv_kernel_w: f32,\n\
         \x20   inv_kernel_hw: f32,\n\
         \x20   inv_col_w: f32,\n\
         \x20   inv_go_spatial: f32,\n\
         }\n\n",
    );

    // Storage bindings — must match Conv2dData / Conv2dGradInputData layout
    if backward {
        src.push_str("var<storage> grad_out: array<f32>;\n");
        src.push_str("var<storage> weight: array<f32>;\n");
    } else {
        src.push_str("var<storage> src: array<f32>;\n");
        src.push_str("var<storage> weight: array<vec4<f32>>;\n");
    }
    src.push_str("var<storage, read_write> dst: array<f32>;\n");
    src.push_str("var<uniform> params: Params;\n");
    let _ = writeln!(
        src,
        "var<workgroup> shared_a0: array<{elem_type}, {shared_size}>;"
    );
    let _ = writeln!(
        src,
        "var<workgroup> shared_a1: array<{elem_type}, {shared_size}>;"
    );
    let _ = writeln!(
        src,
        "var<workgroup> shared_b0: array<{elem_type}, {shared_size}>;"
    );
    let _ = writeln!(
        src,
        "var<workgroup> shared_b1: array<{elem_type}, {shared_size}>;"
    );
    src.push('\n');

    // Compile-time constants for kernel geometry
    let _ = writeln!(src, "const KERNEL_H: u32 = {kernel_h}u;");
    let _ = writeln!(src, "const KERNEL_W: u32 = {kernel_w}u;");
    let _ = writeln!(src, "const KERNEL_HW: u32 = {kernel_hw}u;");
    let _ = writeln!(src, "const STRIDE: u32 = {stride}u;");
    src.push('\n');

    // Main function
    let _ = writeln!(src, "@compute @workgroup_size(64)");
    src.push_str(
        "fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {\n",
    );

    if backward {
        // Backward: M = Ci, N = H*W (input spatial), K = Co*kH*kW
        let _ = writeln!(src, "    let tile_row = wgid.x * {output_tile}u;");
        let _ = writeln!(src, "    let tile_col = wgid.y * {output_tile}u;");
        src.push_str("    let n = wgid.z;\n\n");
        src.push_str("    let m_total = params.in_channels;\n");
        src.push_str("    let n_total = params.in_h * params.in_w;\n");
        let _ = writeln!(src, "    let k_total = params.out_channels * KERNEL_HW;");
        src.push_str("    let go_spatial = params.out_h * params.out_w;\n\n");

        // Padding computation for backward
        let _ = writeln!(
            src,
            "    let pad_h = i32(KERNEL_H) - 1 - i32(params.padding_h);"
        );
        let _ = writeln!(
            src,
            "    let pad_w = i32(KERNEL_W) - 1 - i32(params.padding_w);"
        );
    } else {
        // Forward: M = Co, N = oH*oW, K = Ci*kH*kW
        let _ = writeln!(src, "    let tile_row = wgid.x * {output_tile}u;");
        let _ = writeln!(src, "    let tile_col = wgid.y * {output_tile}u;");
        src.push_str("    let n = wgid.z;\n\n");
        src.push_str("    let m_total = params.out_channels;\n");
        src.push_str("    let n_total = params.out_h * params.out_w;\n");
        let _ = writeln!(src, "    let k_total = params.in_channels * KERNEL_HW;");
        src.push_str("    let input_stride = params.in_channels * params.in_h * params.in_w;\n\n");
    }

    // C offsets for the 4 output tiles
    src.push_str("    let c00 = n * m_total * n_total + tile_row * n_total + tile_col;\n");
    let _ = writeln!(
        src,
        "    let c01 = n * m_total * n_total + tile_row * n_total + (tile_col + {tile}u);"
    );
    let _ = writeln!(
        src,
        "    let c10 = n * m_total * n_total + (tile_row + {tile}u) * n_total + tile_col;"
    );
    let _ = writeln!(
        src,
        "    let c11 = n * m_total * n_total + (tile_row + {tile}u) * n_total + (tile_col + {tile}u);"
    );
    src.push('\n');
    let _ = writeln!(src, "    let n1_valid = (tile_col + {tile}u) < n_total;");
    let _ = writeln!(src, "    let m1_valid = (tile_row + {tile}u) < m_total;");
    src.push('\n');

    // Accumulator init
    let _ = writeln!(src, "    var acc00 = {coop_c}();");
    let _ = writeln!(src, "    var acc01 = {coop_c}();");
    let _ = writeln!(src, "    var acc10 = {coop_c}();");
    let _ = writeln!(src, "    var acc11 = {coop_c}();");
    src.push('\n');

    // Hoisted staging index components
    if backward {
        let _ = writeln!(src, "    let src_col = lid.x & {tile_mask}u;");
        let _ = writeln!(src, "    let base_row = lid.x >> {tile_shift}u;");
    } else {
        src.push_str("    let v4_row = lid.x >> 2u;\n");
        src.push_str("    let v4_col = (lid.x & 3u) << 2u;\n");
        let _ = writeln!(src, "    let src_col = lid.x & {tile_mask}u;");
        let _ = writeln!(src, "    let base_row = lid.x >> {tile_shift}u;");
    }
    src.push('\n');

    // Main loop
    src.push_str("    var t = 0u;\n");
    src.push_str("    loop {\n");
    src.push_str("        if t >= k_total { break; }\n\n");
    let _ = writeln!(src, "        let zero_val = {elem_zero};");
    src.push('\n');

    if backward {
        // === BACKWARD STAGING ===
        // Stage sa0: B-tile im2col(grad_out)^T
        emit_grad_input_im2col_stage(
            &mut src,
            "shared_a0",
            "tile_col",
            "cc0",
            "in_n0",
            "ih0",
            "iw0",
            tile,
            tile_mask,
            staging_iters,
            row_stride,
            kernel_hw,
            stride,
            cast_open,
            cast_close,
            elem_zero,
        );

        // Stage sa1: second column block
        let _ = writeln!(src, "        let cc1 = tile_col + {tile}u + src_col;");
        emit_grad_input_im2col_stage(
            &mut src,
            "shared_a1",
            &format!("tile_col + {tile}u"),
            "cc1",
            "in_n1",
            "ih1",
            "iw1",
            tile,
            tile_mask,
            staging_iters,
            row_stride,
            kernel_hw,
            stride,
            cast_open,
            cast_close,
            elem_zero,
        );

        // Stage sb0: A-tile weight_T
        emit_grad_input_weight_stage(
            &mut src,
            "shared_b0",
            "tile_row",
            false,
            tile,
            staging_iters,
            row_stride,
            kernel_hw,
            cast_open,
            cast_close,
            elem_zero,
        );

        // Stage sb1: A-tile weight_T second row block
        emit_grad_input_weight_stage(
            &mut src,
            "shared_b1",
            "tile_row",
            true,
            tile,
            staging_iters,
            row_stride,
            kernel_hw,
            cast_open,
            cast_close,
            elem_zero,
        );
    } else {
        // === FORWARD STAGING ===
        // Stage sb0: A-tile weight[Co, K] via vec4
        emit_forward_weight_stage(
            &mut src,
            "shared_b0",
            "tile_row",
            false,
            tile,
            cast_open,
            cast_close,
            elem_zero,
        );

        // Stage sb1: A-tile weight second row block
        emit_forward_weight_stage(
            &mut src,
            "shared_b1",
            "tile_row",
            true,
            tile,
            cast_open,
            cast_close,
            elem_zero,
        );

        // Stage sa0: B-tile im2col(input)
        emit_forward_im2col_stage(
            &mut src,
            "shared_a0",
            "tile_col",
            "cc0",
            "in_n0",
            tile,
            tile_mask,
            staging_iters,
            row_stride,
            kernel_hw,
            stride,
            cast_open,
            cast_close,
            elem_zero,
        );

        // Stage sa1: B-tile second column block
        let _ = writeln!(src, "        let cc1 = tile_col + {tile}u + src_col;");
        emit_forward_im2col_stage(
            &mut src,
            "shared_a1",
            &format!("tile_col + {tile}u"),
            "cc1",
            "in_n1",
            tile,
            tile_mask,
            staging_iters,
            row_stride,
            kernel_hw,
            stride,
            cast_open,
            cast_close,
            elem_zero,
        );
    }

    // Barrier + cooperative matmul
    src.push_str("\n        workgroupBarrier();\n\n");
    let _ = writeln!(
        src,
        "        let a0 = coopLoadT<{coop_ab}>(&shared_b0[0], {tile}u);"
    );
    let _ = writeln!(
        src,
        "        let a1 = coopLoadT<{coop_ab}>(&shared_b1[0], {tile}u);"
    );
    let _ = writeln!(
        src,
        "        let b0 = coopLoadT<{coop_ba}>(&shared_a0[0], {tile}u);"
    );
    let _ = writeln!(
        src,
        "        let b1 = coopLoadT<{coop_ba}>(&shared_a1[0], {tile}u);"
    );
    src.push_str("        acc00 = coopMultiplyAdd(a0, b0, acc00);\n");
    src.push_str("        acc01 = coopMultiplyAdd(a0, b1, acc01);\n");
    src.push_str("        acc10 = coopMultiplyAdd(a1, b0, acc10);\n");
    src.push_str("        acc11 = coopMultiplyAdd(a1, b1, acc11);\n\n");
    src.push_str("        workgroupBarrier();\n");
    let _ = writeln!(src, "        t += {tile}u;");
    src.push_str("    }\n\n");

    // Store results
    let store_comment = if backward {
        "grad_input [N, Ci, H, W]"
    } else {
        "output [N, Co, oH, oW]"
    };
    let _ = writeln!(
        src,
        "    // Store results to {store_comment} in NCHW layout"
    );
    src.push_str("    coopStoreT(acc00, &dst[c00], n_total);\n");
    src.push_str("    if n1_valid {\n");
    src.push_str("        coopStoreT(acc01, &dst[c01], n_total);\n");
    src.push_str("    }\n");
    src.push_str("    if m1_valid {\n");
    src.push_str("        coopStoreT(acc10, &dst[c10], n_total);\n");
    src.push_str("    }\n");
    src.push_str("    if n1_valid && m1_valid {\n");
    src.push_str("        coopStoreT(acc11, &dst[c11], n_total);\n");
    src.push_str("    }\n");
    src.push_str("}\n");

    parse_wgsl(&src)
}

/// Emit the im2col staging loop for grad_input (backward) direction.
///
/// Loads from `grad_out` with compile-time kernel decomposition.
#[allow(clippy::too_many_arguments)]
fn emit_grad_input_im2col_stage(
    src: &mut String,
    shared_name: &str,
    tile_col_expr: &str,
    cc_var: &str,
    in_n_var: &str,
    ih_var: &str,
    iw_var: &str,
    _tile: u32,
    _tile_mask: u32,
    staging_iters: u32,
    row_stride: u32,
    _kernel_hw: u32,
    stride: u32,
    cast_open: &str,
    cast_close: &str,
    _elem_zero: &str,
) {
    use std::fmt::Write;

    // First stage (sa0) uses tile_col + src_col directly; subsequent stages
    // have cc_var pre-computed above the call.
    let is_first = cc_var == "cc0";
    if is_first {
        let _ = writeln!(src, "        let {cc_var} = {tile_col_expr} + src_col;");
    }
    let _ = writeln!(src, "        let {in_n_var} = {cc_var} < n_total;");

    // Pre-decompose spatial position (invariant across e iterations)
    let _ = writeln!(
        src,
        "        let {ih_var} = u32(f32({cc_var}) * params.inv_col_w);"
    );
    let _ = writeln!(
        src,
        "        let {iw_var} = {cc_var} - {ih_var} * params.in_w;"
    );

    let _ = writeln!(
        src,
        "        for (var e = 0u; e < {staging_iters}u; e++) {{"
    );
    src.push_str("            let flat = lid.x + e * 64u;\n");
    let _ = writeln!(
        src,
        "            let tr = t + base_row + e * {row_stride}u;"
    );
    src.push_str("            var val = zero_val;\n");
    let _ = writeln!(src, "            if tr < k_total && {in_n_var} {{");

    // Decompose tr into (co, kh, kw) using compile-time constants
    let _ = writeln!(src, "                let co = tr / KERNEL_HW;");
    src.push_str("                let k_rem = tr - co * KERNEL_HW;\n");
    let _ = writeln!(src, "                let kh = k_rem / KERNEL_W;");
    src.push_str("                let kw = k_rem - kh * KERNEL_W;\n");

    if stride == 1 {
        // Stride-1 only path: oh = ih + pad_h - kh
        let _ = writeln!(
            src,
            "                let oh = i32({ih_var}) + pad_h - i32(kh);"
        );
        let _ = writeln!(
            src,
            "                let ow = i32({iw_var}) + pad_w - i32(kw);"
        );
        src.push_str(
            "                if oh >= 0 && u32(oh) < params.out_h && ow >= 0 && u32(ow) < params.out_w {\n",
        );
        let _ = writeln!(
            src,
            "                    val = {cast_open}grad_out[n * params.out_channels * go_spatial + co * go_spatial + u32(oh) * params.out_w + u32(ow)]{cast_close};"
        );
        src.push_str("                }\n");
    } else {
        // General stride path
        let _ = writeln!(
            src,
            "                let h_off = i32({ih_var}) + i32(params.padding_h) - i32(kh);"
        );
        let _ = writeln!(
            src,
            "                let w_off = i32({iw_var}) + i32(params.padding_w) - i32(kw);"
        );
        let _ = writeln!(src, "                let i_stride = i32(STRIDE);");
        src.push_str(
            "                if h_off >= 0 && w_off >= 0 && (h_off % i_stride) == 0 && (w_off % i_stride) == 0 {\n",
        );
        let _ = writeln!(src, "                    let oh = u32(h_off) / STRIDE;");
        let _ = writeln!(src, "                    let ow = u32(w_off) / STRIDE;");
        src.push_str("                    if oh < params.out_h && ow < params.out_w {\n");
        let _ = writeln!(
            src,
            "                        val = {cast_open}grad_out[n * params.out_channels * go_spatial + co * go_spatial + oh * params.out_w + ow]{cast_close};"
        );
        src.push_str("                    }\n");
        src.push_str("                }\n");
    }

    src.push_str("            }\n");
    let _ = writeln!(src, "            {shared_name}[flat] = val;");
    src.push_str("        }\n\n");
}

/// Emit the weight staging for grad_input (backward) direction.
///
/// Weight is stored as [Co, Ci, kH, kW]; we load weight_T[Ci, Co*kH*kW].
#[allow(clippy::too_many_arguments)]
fn emit_grad_input_weight_stage(
    src: &mut String,
    shared_name: &str,
    tile_row_expr: &str,
    is_second_block: bool,
    tile: u32,
    staging_iters: u32,
    row_stride: u32,
    _kernel_hw: u32,
    cast_open: &str,
    cast_close: &str,
    _elem_zero: &str,
) {
    use std::fmt::Write;

    // Only emit tc decomposition once (for the first weight stage)
    if !is_second_block {
        src.push_str("        let tc = t + src_col;\n");
        src.push_str("        let in_k = tc < k_total;\n");
        src.push_str("        let tc_co = tc / KERNEL_HW;\n");
        src.push_str("        let tc_k_rem = tc - tc_co * KERNEL_HW;\n");
        src.push_str("        let tc_kh = tc_k_rem / KERNEL_W;\n");
        src.push_str("        let tc_kw = tc_k_rem - tc_kh * KERNEL_W;\n");
        let _ = writeln!(
            src,
            "        let tc_weight_offset = tc_kh * KERNEL_W + tc_kw;"
        );
    }

    let row_offset = if is_second_block {
        format!("{tile_row_expr} + {tile}u + ")
    } else {
        format!("{tile_row_expr} + ")
    };

    let _ = writeln!(
        src,
        "        for (var e = 0u; e < {staging_iters}u; e++) {{"
    );
    src.push_str("            let flat = lid.x + e * 64u;\n");
    let _ = writeln!(
        src,
        "            let gr = {row_offset}base_row + e * {row_stride}u;"
    );
    src.push_str("            var val = zero_val;\n");
    src.push_str("            if gr < m_total && in_k {\n");
    let _ = writeln!(
        src,
        "                val = {cast_open}weight[(tc_co * m_total + gr) * KERNEL_HW + tc_weight_offset]{cast_close};"
    );
    src.push_str("            }\n");
    let _ = writeln!(src, "            {shared_name}[flat] = val;");
    src.push_str("        }\n\n");
}

/// Emit the vec4 weight staging for forward direction.
///
/// Weight is stored as dense [Co, K] row-major with vec4 loads.
#[allow(clippy::too_many_arguments)]
fn emit_forward_weight_stage(
    src: &mut String,
    shared_name: &str,
    tile_row_expr: &str,
    is_second_block: bool,
    tile: u32,
    cast_open: &str,
    cast_close: &str,
    _elem_zero: &str,
) {
    use std::fmt::Write;

    src.push_str("        {\n");
    let row_offset = if is_second_block {
        format!("({tile_row_expr} + {tile}u) + v4_row")
    } else {
        format!("{tile_row_expr} + v4_row")
    };
    let _ = writeln!(src, "            let gr = {row_offset};");
    src.push_str("            let tc4 = t + v4_col;\n");
    let _ = writeln!(src, "            let flat = v4_row * {tile}u + v4_col;");
    src.push_str("            if gr < m_total && (tc4 + 4u) <= k_total {\n");
    src.push_str("                let v = weight[(gr * k_total + tc4) >> 2u];\n");
    let _ = writeln!(
        src,
        "                {shared_name}[flat] = {cast_open}v.x{cast_close};"
    );
    let _ = writeln!(
        src,
        "                {shared_name}[flat + 1u] = {cast_open}v.y{cast_close};"
    );
    let _ = writeln!(
        src,
        "                {shared_name}[flat + 2u] = {cast_open}v.z{cast_close};"
    );
    let _ = writeln!(
        src,
        "                {shared_name}[flat + 3u] = {cast_open}v.w{cast_close};"
    );
    src.push_str("            } else {\n");
    let _ = writeln!(src, "                {shared_name}[flat] = zero_val;");
    let _ = writeln!(src, "                {shared_name}[flat + 1u] = zero_val;");
    let _ = writeln!(src, "                {shared_name}[flat + 2u] = zero_val;");
    let _ = writeln!(src, "                {shared_name}[flat + 3u] = zero_val;");
    src.push_str("            }\n");
    src.push_str("        }\n\n");
}

/// Emit the im2col staging loop for forward direction.
///
/// Loads from `src` (input) with compile-time kernel decomposition.
#[allow(clippy::too_many_arguments)]
fn emit_forward_im2col_stage(
    src: &mut String,
    shared_name: &str,
    tile_col_expr: &str,
    cc_var: &str,
    in_n_var: &str,
    _tile: u32,
    _tile_mask: u32,
    staging_iters: u32,
    row_stride: u32,
    _kernel_hw: u32,
    _stride: u32,
    cast_open: &str,
    cast_close: &str,
    _elem_zero: &str,
) {
    use std::fmt::Write;

    let is_first = cc_var == "cc0";
    if is_first {
        let _ = writeln!(src, "        let {cc_var} = {tile_col_expr} + src_col;");
    }
    let _ = writeln!(src, "        let {in_n_var} = {cc_var} < n_total;");

    let _ = writeln!(
        src,
        "        for (var e = 0u; e < {staging_iters}u; e++) {{"
    );
    src.push_str("            let flat = lid.x + e * 64u;\n");
    let _ = writeln!(
        src,
        "            let tr = t + base_row + e * {row_stride}u;"
    );
    src.push_str("            var val = zero_val;\n");
    let _ = writeln!(src, "            if tr < k_total && {in_n_var} {{");

    // Decompose k_idx into (ci, kh, kw) using compile-time constants
    let _ = writeln!(src, "                let ci = tr / KERNEL_HW;");
    src.push_str("                let k_rem = tr - ci * KERNEL_HW;\n");
    let _ = writeln!(src, "                let kh = k_rem / KERNEL_W;");
    src.push_str("                let kw = k_rem - kh * KERNEL_W;\n");

    // Decompose hw_idx -> (oh, ow) -> (ih, iw)
    let _ = writeln!(
        src,
        "                let oh = u32(f32({cc_var}) * params.inv_col_w);"
    );
    let _ = writeln!(
        src,
        "                let ow = {cc_var} - oh * params.out_w;"
    );

    // Use compile-time stride constant
    let _ = writeln!(
        src,
        "                let ih = i32(oh * STRIDE + kh) - i32(params.padding_h);"
    );
    let _ = writeln!(
        src,
        "                let iw = i32(ow * STRIDE + kw) - i32(params.padding_w);"
    );
    src.push_str(
        "                if ih >= 0 && u32(ih) < params.in_h && iw >= 0 && u32(iw) < params.in_w {\n",
    );
    let _ = writeln!(
        src,
        "                    val = {cast_open}src[n * input_stride + ci * params.in_h * params.in_w + u32(ih) * params.in_w + u32(iw)]{cast_close};"
    );
    src.push_str("                }\n");

    src.push_str("            }\n");
    let _ = writeln!(src, "            {shared_name}[flat] = val;");
    src.push_str("        }\n\n");
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
                ShaderGroup::FlashAttentionCoop,
                naga::valid::Capabilities::COOPERATIVE_MATRIX
                    | naga::valid::Capabilities::SHADER_FLOAT16,
            ),
            (
                ShaderGroup::FlashGradQCoop,
                naga::valid::Capabilities::COOPERATIVE_MATRIX
                    | naga::valid::Capabilities::SHADER_FLOAT16,
            ),
            (
                ShaderGroup::FlashGradKVCoop,
                naga::valid::Capabilities::COOPERATIVE_MATRIX
                    | naga::valid::Capabilities::SHADER_FLOAT16,
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
            (ShaderGroup::FlashGradK, naga::valid::Capabilities::empty()),
            (ShaderGroup::FlashGradV, naga::valid::Capabilities::empty()),
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
                ShaderEntry::MultiHeadAttn
                | ShaderEntry::FlashAttention
                | ShaderEntry::FlashAttentionCoop => {
                    vec!["src_a", "src_b", "bias", "dst", "lse", "params"]
                }
                ShaderEntry::MultiHeadAttnGradQ
                | ShaderEntry::FlashGradQ
                | ShaderEntry::FlashGradQCoop
                | ShaderEntry::MultiHeadAttnGradK
                | ShaderEntry::FlashGradK
                | ShaderEntry::MultiHeadAttnGradV
                | ShaderEntry::FlashGradV => {
                    vec![
                        "d_out", "src_a", "src_b", "bias", "lse", "fwd_dst", "dst", "params",
                    ]
                }
                ShaderEntry::MultiHeadAttnGradKV
                | ShaderEntry::FlashGradKV
                | ShaderEntry::FlashGradKVCoop => {
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
                | ShaderEntry::Conv2dGemmCoop
                | ShaderEntry::Conv2dGemmCoopGen(..) => vec!["src", "weight", "dst", "params"],
                ShaderEntry::Conv2dGradInput => vec!["grad_out", "weight", "dst", "params"],
                ShaderEntry::Conv2dGradInputGemm | ShaderEntry::Conv2dGradInputGemmSmall => {
                    vec!["grad_out", "weight", "dst", "params"]
                }
                ShaderEntry::Conv2dGradInputGemmCoop
                | ShaderEntry::Conv2dGradInputGemmCoop3x3
                | ShaderEntry::Conv2dGradInputGemmCoopGen(..) => {
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
            ShaderEntry::FlashAttentionCoop,
            ShaderEntry::FlashGradQCoop,
            ShaderEntry::FlashGradKVCoop,
            ShaderEntry::MultiHeadAttnGradQ,
            ShaderEntry::FlashGradQ,
            ShaderEntry::MultiHeadAttnGradK,
            ShaderEntry::MultiHeadAttnGradKV,
            ShaderEntry::FlashGradKV,
            ShaderEntry::FlashGradK,
            ShaderEntry::FlashGradV,
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

    /// Verify that `generate_conv2d_coop_module` produces valid WGSL+naga modules
    /// for various kernel configs, both forward and backward.
    #[test]
    fn generated_conv2d_coop_modules_are_valid() {
        use naga::valid::{Capabilities, ValidationFlags, Validator};

        let coop_caps = Capabilities::COOPERATIVE_MATRIX | Capabilities::SHADER_FLOAT16;
        let flags = ValidationFlags::all() ^ ValidationFlags::BINDINGS;
        let config = CoopConfig {
            tile_size: 16,
            use_f16_input: true,
        };

        // Test several kernel configs x direction combinations
        let cases = [
            (1, 1, 1, Conv2dCoopDirection::Forward),
            (1, 1, 1, Conv2dCoopDirection::GradInput),
            (3, 3, 1, Conv2dCoopDirection::Forward),
            (3, 3, 1, Conv2dCoopDirection::GradInput),
            (3, 3, 2, Conv2dCoopDirection::Forward),
            (3, 3, 2, Conv2dCoopDirection::GradInput),
            (5, 5, 1, Conv2dCoopDirection::GradInput),
            (7, 7, 2, Conv2dCoopDirection::Forward),
        ];

        for (kh, kw, stride, direction) in &cases {
            let sm = generate_conv2d_coop_module(*kh, *kw, *stride, *direction, &config);
            let mut validator = Validator::new(flags, coop_caps);
            let result = validator.validate(&sm.module);
            assert!(
                result.is_ok(),
                "Conv2d coop gen ({kh}x{kw} s{stride} {direction:?}) failed validation: {:?}",
                result.err()
            );
        }
    }
}
