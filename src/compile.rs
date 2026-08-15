use crate::graph::{DType, Graph, Node, NodeId, Op, PairwiseGradKind};
use crate::schedule::{PointwiseDAG, Pw, ReductionEpilogue, ReductionKernel};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Weight storage format for matmul B operands.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WeightFormat {
    #[default]
    F32,
    F16,
    Q4,
    Q8,
}

impl WeightFormat {
    pub fn is_quantized(self) -> bool {
        matches!(self, Self::Q4 | Self::Q8)
    }

    /// Uses a B-buffer representation other than ordinary IEEE f32.
    ///
    /// This deliberately includes f16 as well as block-quantized formats;
    /// these formats share the weighted-shader codegen path even though f16
    /// is not quantization.
    pub fn uses_reduced_storage(self) -> bool {
        !matches!(self, Self::F32)
    }

    pub fn from_dtype(dtype: DType) -> Self {
        match dtype {
            DType::F16 => Self::F16,
            DType::Q4_0 => Self::Q4,
            DType::Q8_0 => Self::Q8,
            _ => Self::F32,
        }
    }
}

/// Performance-tuning knobs that shape generated kernels and dispatch
/// geometry. A knob is *data*: it lives in `CompileOptions`, is stamped into
/// the compiled `ExecutionPlan` (geometry and generated WGSL must agree), and
/// participates in the plan-cache fingerprint automatically. Defaults come
/// from capability-signature heuristics plus `MEGANEURA_FLASH_*` env
/// overrides; a session-build tuner can substitute measured values instead.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TuningKnobs {
    /// Elements-per-thread cap for flash-attention forward codegen.
    pub flash_ept_cap: u32,
    /// EPT cap for the flash dQ backward kernel.
    pub flash_grad_q_ept_cap: u32,
    /// EPT cap for the fused flash dK/dV backward kernel.
    pub flash_grad_kv_ept_cap: u32,
}

impl Default for TuningKnobs {
    /// Pure per-platform defaults. Apple Silicon benefits from the extra
    /// parallelism of smaller EPT; 32 keeps register count below the
    /// spilling cliff on Ampere/Blackwell. `MEGANEURA_FLASH_*_EPT_CAP`
    /// overrides are applied only by [`TuningKnobs::from_env`] (in
    /// `crate::config`) — the library itself never reads the environment.
    fn default() -> Self {
        let apple = cfg!(all(target_vendor = "apple", target_arch = "aarch64"));
        let fwd = if apple { 16 } else { 32 };
        Self {
            flash_ept_cap: fwd,
            flash_grad_q_ept_cap: fwd,
            flash_grad_kv_ept_cap: if apple { 8 } else { 32 },
        }
    }
}

/// Options controlling graph → execution-plan compilation.
///
/// Wire these to env vars or CLI flags in your own harness if you want —
/// the library itself takes only this typed struct.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompileOptions {
    /// Route unary + binary pointwise ops through the schedule-template
    /// codegen path (with chain fusion) instead of the hand-written
    /// unary.wgsl / binary.wgsl shaders. The generated WGSL uses the same
    /// UnaryData / BinaryData / TernaryData binding layouts, so no runtime
    /// surface changes for callers.
    pub use_schedule_pointwise: bool,
    /// Route reduction-shaped ops (currently Softmax) through the
    /// schedule-template reduction archetype instead of hand-written
    /// shaders. Generated kernels use workgroup-per-row tree reduction,
    /// which is much more parallel than the 1-thread-per-row loops in
    /// the existing softmax.wgsl. Enabled by default after parity validation.
    pub use_schedule_reduction: bool,
    /// Apply dispatch-level fusion passes (matmul epilogues, pointwise
    /// chains, reduction prologues, RmsNorm→matmul prologues). These are
    /// numerics-neutral performance transforms; debug sessions disable them
    /// so every graph node's value stays materialized and readable.
    pub fuse_dispatches: bool,
    /// Tuning knobs stamped into the compiled plan.
    pub knobs: TuningKnobs,
    /// Use the cooperative flash-attention forward kernel when the device
    /// supports it.
    pub flash_forward_coop: bool,
    /// Enable the experimental reduced-precision cooperative flash
    /// backward kernels.
    pub flash_backward_coop: bool,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            use_schedule_pointwise: true,
            use_schedule_reduction: true,
            fuse_dispatches: true,
            knobs: TuningKnobs::default(),
            flash_forward_coop: true,
            flash_backward_coop: false,
        }
    }
}

/// Identifies which shader and entry point to use.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ShaderEntry {
    #[default]
    MatMul,
    MatMulAT,
    MatMulBT,
    /// M=1 GEMV specialization of MatMul. Selected when `C = A × B` has
    /// a single row on the output side (LM decode path).
    MatMulGemv,
    /// M=1 GEMV with fused residual add: `C = A × B + D`.
    /// Selected for FusedMatMulAdd when M=1.
    MatMulGemvAdd,
    /// M=1 MatMulBT specialization (`B` stored `[N,K]`). K-split with
    /// naturally coalesced vec4 reads along the contiguous K axis.
    MatMulGemvBT,
    FusedMatMulAdd,
    FusedMatMulATAdd,
    FusedMatMulBTAdd,
    Relu,
    Sigmoid,
    Tanh,
    Neg,
    Abs,
    Log,
    Recip,
    Add,
    Mul,
    Greater,
    BiasAdd,
    SgdUpdate,
    AdamUpdate,
    ScatterAdd,
    ScatterAddAtomic,
    SumAll,
    MeanAll,
    Softmax,
    CrossEntropyLoss,
    BceLoss,
    Transpose,
    Silu,
    SwiGLU,
    RmsNorm,
    Embedding,
    ToF16,
    RoPE,
    RoPEGrad,
    Gelu,
    LayerNorm,
    MultiHeadAttn,
    /// Flash Attention 2 forward: BQ>1 multi-query tiling.
    FlashAttention,
    /// Cooperative-matrix flash attention forward (Phase 1: coop QK^T,
    /// scalar softmax + PV). Enabled on compatible devices unless
    /// `MEGANEURA_FLASH_FWD_COOP=0`.
    /// BQ=BKV=16, dispatched as `[ceil(q_seq/16), num_heads, 1]`.
    FlashAttentionCoop,
    /// Cooperative-matrix flash backward dQ kernel. Three coop matmuls
    /// per KV tile: score=Q·K^T, dp=dO·V^T, dQ+=ds·K. Opt-in via
    /// `MEGANEURA_FLASH_BWD_COOP=1`. BQ=BKV=16,
    /// dispatched as `[ceil(q_seq/16), num_heads, 1]`.
    FlashGradQCoop,
    /// Cooperative-matrix flash backward dK + dV kernel (fused).
    /// Two coop matmuls per Q-tile: score = K·Q^T, dp = V·dO^T.
    /// dV/dK accumulate per-thread. Dispatched as
    /// `[ceil(dispatch_kv/16), num_kv_heads, 1]`.
    FlashGradKVCoop,
    MultiHeadAttnGradQ,
    FlashGradQ,
    MultiHeadAttnGradKV,
    FlashGradKV,
    SwiGLUGradGate,
    SwiGLUGradUp,
    SiluGrad,
    SwiGLUConcat,
    SwiGLUConcatGrad,
    SumRows,
    RmsNormGradW,
    RmsNormGradWRowPar,
    RmsNormGradX,
    LayerNormGradWB,
    LayerNormGradX,
    /// Precompute rsqrt for RmsNorm (phase 1 of two-phase fusion)
    RmsNormRsqrt,
    GroupNorm,
    GroupNormSilu,
    GroupNormApply,
    GroupNormGradInput,
    GroupNormGradWeightBias,
    Concat,
    SplitA,
    SplitB,
    Upsample2x,
    Upsample2xGrad,
    /// Depthwise Conv2d forward (groups == channels). Weight shape
    /// `[C, 1, kH, kW]`; each output channel reads one input channel.
    /// Used by EfficientNet MBConv blocks.
    Conv2dDw,
    /// Per-channel broadcast multiply: `dst[n,c,h,w] = src[n,c,h,w] * gate[n,c]`.
    /// Used by EfficientNet Squeeze-and-Excitation.
    MulPerChannel,
    /// Per-channel broadcast add: `dst[n,c,h,w] = src[n,c,h,w] + bias[c]`.
    /// Used to apply a fused-BN per-channel bias to a conv output.
    AddPerChannel,
    Conv2dGemm,
    Conv2dGemmSmall,
    /// Generated conv2d forward coop kernel specialized for (kernel_h, kernel_w, stride).
    Conv2dGemmCoopGen(u32, u32, u32),
    Conv2dGradInputGemm,
    Conv2dGradInputGemmSmall,
    /// Generated conv2d grad_input coop kernel specialized for (kernel_h, kernel_w, stride).
    Conv2dGradInputGemmCoopGen(u32, u32, u32),
    Conv2dGradWeightGemm,
    Conv2dGradWeightGemmSmall,
    CacheWrite,
    CachedAttention,
    RoPEDynamic,
    MaxPool2d,
    GlobalAvgPool,
    GlobalAvgPoolGrad,
    PairwiseGrad,
    WinogradInputTransform,
    WinogradOutputTransform,
    WinogradBatchedMatMul,
    WinogradWeightTransform,
    /// Zero a 1-element scalar accumulator buffer (gradient-clip pre-pass).
    GradClipZero,
    /// Sum-of-squares of a gradient buffer, added to a shared 1-element
    /// scalar buffer. One workgroup of 256 threads per dispatch tree-reduces
    /// in shared memory; the runtime barriers parameter dispatches. Pair with
    /// `GradClipZero` (pre-pass) and `GradClipScale` (post-pass).
    GradClipNormSq,
    /// In-place scale a gradient buffer by `min(1, max_norm / norm)`,
    /// where `norm = sqrt(acc[0])` from the accumulator
    /// produced by `GradClipNormSq`.
    GradClipScale,
    /// Temporal gradient accumulation: `acc[i] += grad[i] * scale`. Adds
    /// each step's fresh (overwritten) grad into a persistent accumulator
    /// so gradients sum across `step()` calls. Cleared by `zero_grad`.
    GradAccum,
}

impl ShaderEntry {
    /// Coarse workload family used by structured performance profiles.
    ///
    /// These categories intentionally match the paper-level decomposition:
    /// they are stable enough to compare profiles across revisions while the
    /// individual shader variants continue to evolve.
    pub fn profile_family(&self) -> &'static str {
        match *self {
            ShaderEntry::MatMul
            | ShaderEntry::MatMulAT
            | ShaderEntry::MatMulBT
            | ShaderEntry::MatMulGemv
            | ShaderEntry::MatMulGemvAdd
            | ShaderEntry::MatMulGemvBT
            | ShaderEntry::FusedMatMulAdd
            | ShaderEntry::FusedMatMulATAdd
            | ShaderEntry::FusedMatMulBTAdd => "matrix",

            ShaderEntry::MultiHeadAttn
            | ShaderEntry::FlashAttention
            | ShaderEntry::FlashAttentionCoop
            | ShaderEntry::FlashGradQCoop
            | ShaderEntry::FlashGradKVCoop
            | ShaderEntry::MultiHeadAttnGradQ
            | ShaderEntry::FlashGradQ
            | ShaderEntry::MultiHeadAttnGradKV
            | ShaderEntry::FlashGradKV
            | ShaderEntry::CachedAttention => "attention",

            ShaderEntry::Conv2dDw
            | ShaderEntry::Conv2dGemm
            | ShaderEntry::Conv2dGemmSmall
            | ShaderEntry::Conv2dGemmCoopGen(..)
            | ShaderEntry::Conv2dGradInputGemm
            | ShaderEntry::Conv2dGradInputGemmSmall
            | ShaderEntry::Conv2dGradInputGemmCoopGen(..)
            | ShaderEntry::Conv2dGradWeightGemm
            | ShaderEntry::Conv2dGradWeightGemmSmall
            | ShaderEntry::Upsample2x
            | ShaderEntry::Upsample2xGrad
            | ShaderEntry::MaxPool2d
            | ShaderEntry::WinogradInputTransform
            | ShaderEntry::WinogradOutputTransform
            | ShaderEntry::WinogradBatchedMatMul
            | ShaderEntry::WinogradWeightTransform => "convolution_spatial",

            ShaderEntry::SumAll
            | ShaderEntry::MeanAll
            | ShaderEntry::Softmax
            | ShaderEntry::CrossEntropyLoss
            | ShaderEntry::BceLoss
            | ShaderEntry::RmsNorm
            | ShaderEntry::LayerNorm
            | ShaderEntry::SumRows
            | ShaderEntry::RmsNormGradW
            | ShaderEntry::RmsNormGradWRowPar
            | ShaderEntry::RmsNormGradX
            | ShaderEntry::LayerNormGradWB
            | ShaderEntry::LayerNormGradX
            | ShaderEntry::RmsNormRsqrt
            | ShaderEntry::GroupNorm
            | ShaderEntry::GroupNormSilu
            | ShaderEntry::GroupNormApply
            | ShaderEntry::GroupNormGradInput
            | ShaderEntry::GroupNormGradWeightBias
            | ShaderEntry::GlobalAvgPool
            | ShaderEntry::GlobalAvgPoolGrad
            | ShaderEntry::PairwiseGrad => "normalization_reduction",

            ShaderEntry::SgdUpdate
            | ShaderEntry::AdamUpdate
            | ShaderEntry::GradClipZero
            | ShaderEntry::GradClipNormSq
            | ShaderEntry::GradClipScale
            | ShaderEntry::GradAccum => "optimizer",

            ShaderEntry::ScatterAdd
            | ShaderEntry::ScatterAddAtomic
            | ShaderEntry::Transpose
            | ShaderEntry::Embedding
            | ShaderEntry::ToF16
            | ShaderEntry::Concat
            | ShaderEntry::SplitA
            | ShaderEntry::SplitB
            | ShaderEntry::CacheWrite => "data_movement",

            ShaderEntry::Relu
            | ShaderEntry::Sigmoid
            | ShaderEntry::Tanh
            | ShaderEntry::Neg
            | ShaderEntry::Abs
            | ShaderEntry::Log
            | ShaderEntry::Recip
            | ShaderEntry::Add
            | ShaderEntry::Mul
            | ShaderEntry::Greater
            | ShaderEntry::BiasAdd
            | ShaderEntry::Silu
            | ShaderEntry::SwiGLU
            | ShaderEntry::RoPE
            | ShaderEntry::RoPEGrad
            | ShaderEntry::Gelu
            | ShaderEntry::SwiGLUGradGate
            | ShaderEntry::SwiGLUGradUp
            | ShaderEntry::SiluGrad
            | ShaderEntry::SwiGLUConcat
            | ShaderEntry::SwiGLUConcatGrad
            | ShaderEntry::MulPerChannel
            | ShaderEntry::AddPerChannel
            | ShaderEntry::RoPEDynamic => "pointwise",
        }
    }

    pub fn shader_group(&self) -> crate::codegen::ShaderGroup {
        use crate::codegen::ShaderGroup;
        match *self {
            ShaderEntry::MatMul => ShaderGroup::MatMul,
            ShaderEntry::MatMulAT => ShaderGroup::MatMulAT,
            ShaderEntry::MatMulBT => ShaderGroup::MatMulBT,
            ShaderEntry::MatMulGemv => ShaderGroup::MatMulGemv,
            ShaderEntry::MatMulGemvAdd => ShaderGroup::MatMulGemvAdd,
            ShaderEntry::MatMulGemvBT => ShaderGroup::MatMulGemvBT,
            ShaderEntry::FusedMatMulAdd => ShaderGroup::MatMulAdd,
            ShaderEntry::FusedMatMulATAdd => ShaderGroup::MatMulATAdd,
            ShaderEntry::FusedMatMulBTAdd => ShaderGroup::MatMulBTAdd,
            ShaderEntry::Relu
            | ShaderEntry::Sigmoid
            | ShaderEntry::Tanh
            | ShaderEntry::Neg
            | ShaderEntry::Abs
            | ShaderEntry::Log
            | ShaderEntry::Recip => ShaderGroup::Unary,
            ShaderEntry::Add | ShaderEntry::Mul | ShaderEntry::Greater => ShaderGroup::Binary,
            ShaderEntry::BiasAdd => ShaderGroup::BiasAdd,
            ShaderEntry::SgdUpdate => ShaderGroup::Sgd,
            ShaderEntry::AdamUpdate => ShaderGroup::Adam,
            ShaderEntry::ScatterAdd => ShaderGroup::ScatterAdd,
            ShaderEntry::ScatterAddAtomic => ShaderGroup::ScatterAddAtomic,
            ShaderEntry::SumAll | ShaderEntry::MeanAll => ShaderGroup::Reduce,
            ShaderEntry::Softmax => ShaderGroup::Softmax,
            ShaderEntry::CrossEntropyLoss => ShaderGroup::CrossEntropy,
            ShaderEntry::BceLoss => ShaderGroup::BceLoss,
            ShaderEntry::Transpose => ShaderGroup::Transpose,
            ShaderEntry::Silu => ShaderGroup::Unary,
            ShaderEntry::SwiGLU => ShaderGroup::Binary,
            ShaderEntry::RmsNorm => ShaderGroup::RmsNorm,
            ShaderEntry::Embedding => ShaderGroup::Embedding,
            ShaderEntry::ToF16 => ShaderGroup::ToF16,
            ShaderEntry::RoPE => ShaderGroup::RoPE,
            ShaderEntry::RoPEGrad => ShaderGroup::RoPEGrad,
            ShaderEntry::Gelu => ShaderGroup::Unary,
            ShaderEntry::LayerNorm => ShaderGroup::LayerNorm,
            ShaderEntry::MultiHeadAttn => ShaderGroup::MultiHeadAttn,
            ShaderEntry::FlashAttention => ShaderGroup::FlashAttention,
            ShaderEntry::FlashAttentionCoop => ShaderGroup::FlashAttentionCoop,
            ShaderEntry::FlashGradQCoop => ShaderGroup::FlashGradQCoop,
            ShaderEntry::FlashGradKVCoop => ShaderGroup::FlashGradKVCoop,
            ShaderEntry::MultiHeadAttnGradQ => ShaderGroup::MultiHeadAttnGradQ,
            ShaderEntry::FlashGradQ => ShaderGroup::FlashGradQ,
            ShaderEntry::MultiHeadAttnGradKV => ShaderGroup::MultiHeadAttnGradKV,
            ShaderEntry::FlashGradKV => ShaderGroup::FlashGradKV,
            ShaderEntry::SwiGLUGradGate | ShaderEntry::SwiGLUGradUp | ShaderEntry::SiluGrad => {
                ShaderGroup::SwiGLUGrad
            }
            ShaderEntry::SwiGLUConcat | ShaderEntry::SwiGLUConcatGrad => ShaderGroup::SwiGLUConcat,
            ShaderEntry::SumRows => ShaderGroup::SumRows,
            ShaderEntry::RmsNormGradW | ShaderEntry::RmsNormGradX => ShaderGroup::RmsNormGrad,
            ShaderEntry::RmsNormGradWRowPar => ShaderGroup::RmsNormGradWRowPar,
            ShaderEntry::LayerNormGradWB | ShaderEntry::LayerNormGradX => {
                ShaderGroup::LayerNormGrad
            }
            ShaderEntry::RmsNormRsqrt => ShaderGroup::RmsNormRsqrt,
            ShaderEntry::GroupNorm => ShaderGroup::GroupNorm,
            ShaderEntry::GroupNormSilu => ShaderGroup::GroupNormSilu,
            ShaderEntry::GroupNormApply => ShaderGroup::GroupNorm,
            ShaderEntry::GroupNormGradInput => ShaderGroup::GroupNormGrad,
            ShaderEntry::GroupNormGradWeightBias => ShaderGroup::GroupNormGrad,
            ShaderEntry::Concat => ShaderGroup::Concat,
            ShaderEntry::SplitA | ShaderEntry::SplitB => ShaderGroup::Split,
            ShaderEntry::Upsample2x => ShaderGroup::Upsample,
            ShaderEntry::Upsample2xGrad => ShaderGroup::UpsampleGrad,
            ShaderEntry::Conv2dDw => ShaderGroup::Conv2dDw,
            ShaderEntry::MulPerChannel => ShaderGroup::MulPerChannel,
            ShaderEntry::AddPerChannel => ShaderGroup::AddPerChannel,
            ShaderEntry::Conv2dGemm => ShaderGroup::Conv2dGemm,
            ShaderEntry::Conv2dGemmCoopGen(..) => ShaderGroup::Conv2dGemmCoop,
            ShaderEntry::Conv2dGemmSmall => ShaderGroup::Conv2dGemmSmall,
            ShaderEntry::Conv2dGradInputGemm => ShaderGroup::Conv2dGradInputGemm,
            ShaderEntry::Conv2dGradInputGemmSmall => ShaderGroup::Conv2dGradInputGemmSmall,
            ShaderEntry::Conv2dGradInputGemmCoopGen(..) => ShaderGroup::Conv2dGradInputGemmCoop,
            ShaderEntry::Conv2dGradWeightGemm => ShaderGroup::Conv2dGradWeightGemm,
            ShaderEntry::Conv2dGradWeightGemmSmall => ShaderGroup::Conv2dGradWeightGemmSmall,
            ShaderEntry::CacheWrite => ShaderGroup::CacheWrite,
            ShaderEntry::CachedAttention => ShaderGroup::CachedAttention,
            ShaderEntry::RoPEDynamic => ShaderGroup::RoPEDynamic,
            ShaderEntry::MaxPool2d => ShaderGroup::MaxPool2d,
            ShaderEntry::GlobalAvgPool => ShaderGroup::GlobalAvgPool,
            ShaderEntry::GlobalAvgPoolGrad => ShaderGroup::GlobalAvgPoolGrad,
            ShaderEntry::PairwiseGrad => ShaderGroup::PairwiseGrad,
            ShaderEntry::WinogradInputTransform => ShaderGroup::WinogradInputTransform,
            ShaderEntry::WinogradOutputTransform => ShaderGroup::WinogradOutputTransform,
            ShaderEntry::WinogradBatchedMatMul => ShaderGroup::WinogradBatchedMatMul,
            ShaderEntry::WinogradWeightTransform => ShaderGroup::WinogradWeightTransform,
            ShaderEntry::GradClipZero => ShaderGroup::GradClipZero,
            ShaderEntry::GradClipNormSq => ShaderGroup::GradClipNormSq,
            ShaderEntry::GradClipScale => ShaderGroup::GradClipScale,
            ShaderEntry::GradAccum => ShaderGroup::GradAccum,
        }
    }

    pub fn entry_point(&self) -> &'static str {
        match *self {
            ShaderEntry::MatMul
            | ShaderEntry::MatMulAT
            | ShaderEntry::MatMulBT
            | ShaderEntry::MatMulGemv
            | ShaderEntry::MatMulGemvAdd
            | ShaderEntry::MatMulGemvBT
            | ShaderEntry::FusedMatMulAdd
            | ShaderEntry::FusedMatMulATAdd
            | ShaderEntry::FusedMatMulBTAdd
            | ShaderEntry::BiasAdd
            | ShaderEntry::SgdUpdate
            | ShaderEntry::AdamUpdate
            | ShaderEntry::ScatterAdd
            | ShaderEntry::ScatterAddAtomic
            | ShaderEntry::Softmax
            | ShaderEntry::CrossEntropyLoss
            | ShaderEntry::BceLoss
            | ShaderEntry::Transpose => "main",
            ShaderEntry::Relu => "relu",
            ShaderEntry::Sigmoid => "sigmoid",
            ShaderEntry::Tanh => "tanh_",
            ShaderEntry::Neg => "neg",
            ShaderEntry::Abs => "abs_",
            ShaderEntry::Log => "log_",
            ShaderEntry::Recip => "recip",
            ShaderEntry::Add => "add",
            ShaderEntry::Mul => "mul",
            ShaderEntry::Greater => "greater",
            ShaderEntry::SumAll => "sum_all",
            ShaderEntry::MeanAll => "mean_all",
            ShaderEntry::Silu => "silu",
            ShaderEntry::SwiGLU => "swiglu",
            ShaderEntry::RmsNorm => "main",
            ShaderEntry::Embedding => "main",
            ShaderEntry::ToF16 => "main",
            ShaderEntry::RoPE => "main",
            ShaderEntry::RoPEGrad => "main",
            ShaderEntry::Gelu => "gelu",
            ShaderEntry::LayerNorm => "main",
            ShaderEntry::MultiHeadAttn
            | ShaderEntry::FlashAttention
            | ShaderEntry::FlashAttentionCoop
            | ShaderEntry::MultiHeadAttnGradQ
            | ShaderEntry::FlashGradQ
            | ShaderEntry::FlashGradQCoop
            | ShaderEntry::MultiHeadAttnGradKV
            | ShaderEntry::FlashGradKV
            | ShaderEntry::FlashGradKVCoop => "main",
            ShaderEntry::SwiGLUGradGate => "swiglu_grad_gate",
            ShaderEntry::SwiGLUGradUp => "swiglu_grad_up",
            ShaderEntry::SiluGrad => "silu_grad",
            ShaderEntry::SwiGLUConcat => "swiglu_concat",
            ShaderEntry::SwiGLUConcatGrad => "swiglu_concat_grad",
            ShaderEntry::SumRows => "sum_rows",
            ShaderEntry::RmsNormGradW => "rms_norm_grad_w",
            ShaderEntry::RmsNormGradWRowPar => "rms_norm_grad_w_rowpar",
            ShaderEntry::RmsNormGradX => "rms_norm_grad_x",
            ShaderEntry::LayerNormGradWB => "layer_norm_grad_wb",
            ShaderEntry::LayerNormGradX => "layer_norm_grad_x",
            ShaderEntry::RmsNormRsqrt => "main",
            ShaderEntry::GroupNorm | ShaderEntry::GroupNormSilu => "main",
            ShaderEntry::GroupNormApply => "apply",
            ShaderEntry::GroupNormGradInput => "grad_input",
            ShaderEntry::GroupNormGradWeightBias => "grad_weight_bias",
            ShaderEntry::Concat => "main",
            ShaderEntry::SplitA => "split_a",
            ShaderEntry::SplitB => "split_b",
            ShaderEntry::Upsample2x => "main",
            ShaderEntry::Upsample2xGrad => "main",
            ShaderEntry::Conv2dDw => "main",
            ShaderEntry::MulPerChannel => "main",
            ShaderEntry::AddPerChannel => "main",
            ShaderEntry::Conv2dGemm
            | ShaderEntry::Conv2dGemmSmall
            | ShaderEntry::Conv2dGemmCoopGen(..) => "main",
            ShaderEntry::Conv2dGradInputGemm
            | ShaderEntry::Conv2dGradInputGemmSmall
            | ShaderEntry::Conv2dGradInputGemmCoopGen(..) => "main",
            ShaderEntry::Conv2dGradWeightGemm | ShaderEntry::Conv2dGradWeightGemmSmall => "main",
            ShaderEntry::CacheWrite => "main",
            ShaderEntry::CachedAttention => "main",
            ShaderEntry::RoPEDynamic => "main",
            ShaderEntry::MaxPool2d => "max_pool_2d",
            ShaderEntry::GlobalAvgPool => "global_avg_pool",
            ShaderEntry::GlobalAvgPoolGrad => "main",
            ShaderEntry::PairwiseGrad => "main",
            ShaderEntry::WinogradInputTransform
            | ShaderEntry::WinogradOutputTransform
            | ShaderEntry::WinogradBatchedMatMul
            | ShaderEntry::WinogradWeightTransform => "main",
            ShaderEntry::GradClipZero
            | ShaderEntry::GradClipNormSq
            | ShaderEntry::GradClipScale
            | ShaderEntry::GradAccum => "main",
        }
    }
}

impl Dispatch {
    /// Coarse workload family for this concrete dispatch.
    ///
    /// Generated schedule kernels retain a legacy `shader` entry for binding
    /// layout compatibility, so their schedule kind takes precedence over
    /// that placeholder when profiling.
    pub fn profile_family(&self) -> &'static str {
        if self.is_row_data_movement() {
            "data_movement"
        } else if self.reduction.is_some() {
            "normalization_reduction"
        } else if self.pointwise.is_some() {
            "pointwise"
        } else {
            self.shader.profile_family()
        }
    }

    fn is_inner_broadcast(&self) -> bool {
        self.shader == ShaderEntry::GlobalAvgPoolGrad && self.params.get(2) == Some(&1)
    }

    fn is_row_data_movement(&self) -> bool {
        self.shader == ShaderEntry::GlobalAvgPoolGrad
            && self.params.get(2).is_some_and(|&mode| mode != 0)
    }

    fn is_zero_fill(&self) -> bool {
        self.pointwise.as_ref().is_some_and(|dag| {
            dag.n_inputs == 1 && dag.ops == [Pw::const_f32(0.0)] && dag.output == 0
        })
    }

    #[cfg(test)]
    fn is_row_scaled_atomic_scatter(&self) -> bool {
        self.shader == ShaderEntry::ScatterAddAtomic
            && self.params.get(3).is_some_and(|&mode| mode != 0)
    }
}

/// Legacy enum — kept for serde backward compat of cached plans.
/// New code should use `MatMulEpilogue` (a `PointwiseDAG`).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EpilogueOp {
    Add(u8),
    BiasAdd(u8),
    Relu,
    Silu,
    Sigmoid,
    Neg,
}

/// A RmsNorm folded into a GEMV's A operand.
///
/// A modifier on `ShaderEntry::MatMulGemv` rather than a shader entry of
/// its own, in the same way `weight_format` is: variants of a kernel are
/// resolved when the pipeline is picked, so fusions compose with the other
/// axes instead of multiplying the entry enum.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GemvRmsNorm {
    /// The norm's weight vector, applied per element alongside the scale.
    pub weight: BufferRef,
    /// `f32::to_bits` of the norm's epsilon.
    pub eps_bits: u32,
}

/// How an epilogue buffer is indexed in the matmul store loop.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EpilogueLoadKind {
    /// Load at `row * N + col` (per-element, same shape as output).
    PerElement,
    /// Load at `col` (per-column broadcast, e.g. bias).
    PerCol,
}

/// A fused epilogue applied in the matmul store loop, expressed as a
/// [`PointwiseDAG`]. Replaces the closed `EpilogueOp` enum so arbitrary
/// per-element transforms can be fused without new enum variants.
///
/// `LoadInput(0)` in the DAG = `val` (the matmul accumulator result).
/// `LoadInput(1+)` indexes into `inputs`, each with its own buffer +
/// load-indexing kind (per-element or per-col broadcast).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MatMulEpilogue {
    pub dag: PointwiseDAG,
    pub inputs: Vec<(BufferRef, EpilogueLoadKind)>,
}

/// How a prologue buffer is indexed during matmul A-tile staging.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PrologueLoadKind {
    /// Load at `gr` (global row of the A matrix).
    PerRow,
    /// Load at `tc` (K-column within the current K-tile).
    PerKCol,
}

/// Multiplicative prologue applied during matmul A-tile staging.
///
/// Each factor is multiplied into `a_val` before it enters shared memory:
/// `a_staged = a_val * buf_0[idx] * buf_1[idx] * ...`
///
/// Generalizes the `$A_TRANSFORM` template in `matmul_coop.wgsl` so
/// that fusions like RmsNorm+MatMul don't need a dedicated shader file.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MatMulPrologue {
    pub factors: Vec<(BufferRef, PrologueLoadKind)>,
}
/// Slices to split each GroupNorm group into, so the normalisation's
/// parallelism follows the image instead of the batch.
///
/// The single-pass kernel launches `batch * num_groups` workgroups, which at
/// inference is a handful however large the image is. Enough slices to fill
/// the device, but not so many that a workgroup has less than a few thousand
/// elements to chew on, nor that combining the partials costs more than
/// splitting saved.
pub fn group_norm_chunks(batch: u32, channels: u32, spatial: u32, num_groups: u32) -> u32 {
    const TARGET_WORKGROUPS: u32 = 1024;
    const MIN_PER_WORKGROUP: u32 = 2048;
    let group_size = (channels / num_groups.max(1)) * spatial;
    let by_occupancy = TARGET_WORKGROUPS.div_ceil((batch * num_groups).max(1));
    let by_work = (group_size / MIN_PER_WORKGROUP).max(1);
    let mut chunks = by_occupancy.min(by_work).clamp(1, 256);
    // The generated reduction archetype treats every row as the same length.
    // Pick the closest exact divisor so each chunk is one contiguous row and
    // no bespoke ragged-tail statistics kernel is needed.
    while !group_size.is_multiple_of(chunks) {
        chunks -= 1;
    }
    chunks
}

/// Pack independent same-A matmuls that share a barrier group into one
/// dispatch (`workgroups[2] = N`, `horizontal_batch = N`).
pub fn fuse_horizontal_matmuls(
    dispatches: &mut Vec<Dispatch>,
    groups: &mut Vec<std::ops::Range<usize>>,
) {
    let old = std::mem::take(dispatches);
    let mut new = Vec::with_capacity(old.len());
    let mut new_groups = Vec::with_capacity(groups.len());
    for group in groups.iter() {
        let members: Vec<usize> = group.clone().collect();
        let mut used = vec![false; members.len()];
        let start = new.len();
        for i in 0..members.len() {
            if used[i] {
                continue;
            }
            let mut batch = vec![members[i]];
            for j in (i + 1)..members.len() {
                if used[j] {
                    continue;
                }
                if batch.len() < 3 && can_horizontal_fuse(&old[members[i]], &old[members[j]]) {
                    batch.push(members[j]);
                    used[j] = true;
                }
            }
            used[i] = true;
            if batch.len() >= 2 {
                new.push(merge_horizontal(&old, &batch));
            } else {
                new.push(old[members[i]].clone());
            }
        }
        new_groups.push(start..new.len());
    }
    *dispatches = new;
    *groups = new_groups;
}

fn can_horizontal_fuse(a: &Dispatch, b: &Dispatch) -> bool {
    matches!(
        a.shader,
        ShaderEntry::MatMul | ShaderEntry::MatMulAT | ShaderEntry::MatMulBT
    ) && a.shader == b.shader
        && a.workgroups == b.workgroups
        && a.workgroups[2] == 1
        && a.params == b.params
        && a.use_coop == b.use_coop
        && a.use_coop_compensated == b.use_coop_compensated
        && a.use_small_tiles == b.use_small_tiles
        && !a.weight_format.uses_reduced_storage()
        && a.weight_format == b.weight_format
        && a.matmul_prologue.is_none()
        && b.matmul_prologue.is_none()
        && a.matmul_epilogue.is_none()
        && b.matmul_epilogue.is_none()
        && a.epilogue.is_empty()
        && b.epilogue.is_empty()
        && a.pointwise.is_none()
        && b.pointwise.is_none()
        && a.reduction.is_none()
        && b.reduction.is_none()
        && a.gemv_rmsnorm.is_none()
        && b.gemv_rmsnorm.is_none()
        && a.input_buffers.len() == 2
        && b.input_buffers.len() == 2
        && a.input_buffers[0] == b.input_buffers[0]
}

fn merge_horizontal(dispatches: &[Dispatch], batch: &[usize]) -> Dispatch {
    let mut merged = dispatches[batch[0]].clone();
    let n = batch.len() as u32;
    merged.horizontal_batch = n;
    merged.workgroups[2] = n;
    merged.input_buffers = vec![merged.input_buffers[0]];
    merged.extra_outputs.clear();
    for (k, &idx) in batch.iter().enumerate() {
        let d = &dispatches[idx];
        merged.input_buffers.push(d.input_buffers[1]);
        if k == 0 {
            merged.output_buffer = d.output_buffer;
        } else {
            merged.extra_outputs.push(d.output_buffer);
        }
    }
    merged.label = format!("{}x{}", merged.label, n);
    merged
}

fn group_norm_stats_kernel() -> crate::schedule::ReductionKernel {
    use crate::schedule::{PointwiseDAG, Pw, ReduceOp, ReductionKernel};
    ReductionKernel {
        op: ReduceOp::Sum,
        prologue: PointwiseDAG {
            n_inputs: 1,
            ops: vec![Pw::LoadInput(0)],
            output: 0,
        },
        extra_prologues: vec![PointwiseDAG {
            n_inputs: 1,
            ops: vec![Pw::LoadInput(0), Pw::Mul(0, 0)],
            output: 1,
        }],
        epilogue: None,
        n_per_elem: 1,
        n_per_row: 0,
        workgroup_size: 256,
        rows_per_workgroup: 1,
        gather_elem: Vec::new(),
        input_row_repeats: Vec::new(),
    }
}

/// A single GPU dispatch in the execution plan.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Dispatch {
    pub shader: ShaderEntry,
    pub workgroups: [u32; 3],
    /// Buffer bindings: maps the node IDs for inputs/outputs to buffer slots.
    pub input_buffers: Vec<BufferRef>,
    pub output_buffer: BufferRef,
    /// Extra output buffers (e.g. LSE + scores for attention forward).
    pub extra_outputs: Vec<BufferRef>,
    /// Extra params to upload as a uniform buffer.
    pub params: Vec<u32>,
    /// When true, this dispatch uses the cooperative matrix pipeline
    /// (set at runtime based on per-dispatch eligibility).
    #[serde(default)]
    pub use_coop: bool,
    /// Cooperative f16 path with hi/lo residual staging (C1). Only set
    /// together with `use_coop` on `requires_full_precision` dispatches.
    #[serde(default)]
    pub use_coop_compensated: bool,
    /// Number of same-A sibling matmuls packed into this dispatch (D1).
    /// 0/1 = not packed. Extra B operands follow A in `input_buffers`;
    /// extra C outputs are `extra_outputs`. `workgroups[2]` is the pack
    /// count when this is ≥ 2 (only applied when the original Z was 1).
    #[serde(default)]
    pub horizontal_batch: u32,
    /// When true, use the 32×32 small-tile matmul pipeline instead of 64×64.
    #[serde(default)]
    pub use_small_tiles: bool,
    /// The dispatch belongs to numerically sensitive derivative work and may
    /// not be promoted to a reduced-input-precision implementation. Native
    /// f32 cooperative kernels remain eligible.
    #[serde(default)]
    pub requires_full_precision: bool,
    /// Prevent this dispatch from being absorbed into a producer or consumer.
    /// Used for explicit memory-placement boundaries such as [`Op::Materialize`].
    #[serde(default)]
    pub fusion_barrier: bool,
    /// Fused elementwise epilogue (PointwiseDAG) applied in the matmul
    /// store loop. `None` = no epilogue (default). When present, saves
    /// one dispatch + barrier per fused op.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub matmul_epilogue: Option<MatMulEpilogue>,
    /// RmsNorm folded into this GEMV's A operand. See [`GemvRmsNorm`].
    #[serde(default)]
    pub gemv_rmsnorm: Option<GemvRmsNorm>,
    /// Multiplicative prologue applied during matmul A-tile staging.
    /// When present, the coop matmul fills `$A_TRANSFORM` and
    /// `$PROLOGUE_DECL` template variables from the prologue's factors.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub matmul_prologue: Option<MatMulPrologue>,
    /// Legacy fields — kept for serde backward compat of cached plans.
    /// New code uses `matmul_epilogue` instead.
    #[serde(default)]
    pub epilogue: Vec<EpilogueOp>,
    #[serde(default)]
    pub epilogue_buffers: Vec<BufferRef>,
    /// Human-readable label for profiling (e.g. `"MatMul[50,720,960]"`).
    #[serde(default)]
    pub label: String,
    /// Graph node ids this dispatch implements. One entry normally; several
    /// after dispatch-level fusion absorbs a neighbor. Provenance only —
    /// execution never reads it, but labels, plan dumps, profiler rows, and
    /// `Session::read_node` do.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub origin: Vec<NodeId>,
    /// The scalar `(shader, workgroups)` this dispatch had before
    /// cooperative-matrix promotion. Recorded by variant selection so the
    /// session-build tuner can measure both variants and keep the faster
    /// one. Not serialized: cached plans are stored pre-selection and
    /// re-selected per session.
    #[serde(skip)]
    pub scalar_fallback: Option<(ShaderEntry, [u32; 3])>,
    /// When `Some`, this dispatch uses a schedule-template-generated
    /// pointwise kernel. The runtime compiles a dedicated pipeline from the
    /// DAG (keyed by `PointwiseDAG::hash_key`) and binds it using the same
    /// `UnaryData` / `BinaryData` layout that `shader` already selects.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pointwise: Option<PointwiseDAG>,
    /// When `Some`, this dispatch uses a schedule-template-generated
    /// reduction kernel. Mutually exclusive with `pointwise`. The runtime
    /// compiles a dedicated pipeline from the kernel spec (keyed by
    /// `ReductionKernel::hash_key`) and picks the binding layout based on
    /// the kernel's buffer-input arity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reduction: Option<ReductionKernel>,
    /// Storage format of the B (weight) input buffer.
    #[serde(default)]
    pub weight_format: WeightFormat,
}

/// Reference to a GPU buffer in the execution plan.
#[derive(Clone, Debug, Default, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BufferRef(pub u32);

/// The complete execution plan: a static sequence of dispatches.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExecutionPlan {
    /// Buffer sizes in bytes, indexed by BufferRef.
    pub buffers: Vec<usize>,
    /// Which buffers hold parameters (need initialization).
    pub param_buffers: Vec<(String, BufferRef)>,
    /// Which buffers hold inputs (filled each step).
    pub input_buffers: Vec<(String, BufferRef)>,
    /// Constant buffers with their initial data (uploaded once at session creation).
    pub constant_buffers: Vec<(BufferRef, Vec<f32>)>,
    /// The dispatch sequence. For a training graph, this includes
    /// forward, backward, and parameter update dispatches.
    pub dispatches: Vec<Dispatch>,
    /// Index of the loss buffer (first graph output, for reading back).
    pub loss_buffer: Option<BufferRef>,
    /// All graph output buffers (for reading back multiple outputs).
    pub output_buffers: Vec<BufferRef>,
    /// Parameter buffer → gradient buffer mapping (for SGD).
    pub param_grad_pairs: Vec<(BufferRef, BufferRef)>,
    /// LSE buffers allocated for MultiHeadAttn forward nodes: (node_id, buffer).
    pub lse_buffers: Vec<(NodeId, BufferRef)>,
    /// Derived parameters: buffer computed from source parameters.
    /// Created by the optimizer when fusing e.g. gate+up projections or Winograd weight transforms.
    /// Format: (derived_buf, [(source_name, num_elements), ...], transform)
    #[allow(clippy::type_complexity)] //TODO
    pub derived_params: Vec<(
        BufferRef,
        Vec<(String, usize)>,
        crate::graph::ParamTransform,
    )>,
    /// Quantized weight buffers: format + matrix dimensions (rows, cols).
    #[serde(default)]
    pub weight_buffers: HashMap<BufferRef, (WeightFormat, usize, usize)>,
    /// Node id → buffer holding that node's value, for every graph node.
    /// Debug/introspection only (`Session::read_node`); aliasing may reuse
    /// the physical allocation once the value's live range ends unless the
    /// session pins it (debug mode).
    #[serde(default)]
    pub node_buffers: Vec<(NodeId, BufferRef)>,
    /// Names attached via `Graph::named`, for name-based lookup in sessions,
    /// dumps, and profiler rows.
    #[serde(default)]
    pub node_names: Vec<(NodeId, String)>,
    /// The tuning knobs this plan was compiled with. Dispatch geometry
    /// depends on them, and pipeline creation must generate WGSL with the
    /// same values — they travel with the plan, not as ambient globals.
    #[serde(default)]
    pub knobs: TuningKnobs,
}

/// Compile a differentiated graph into an ExecutionPlan.
/// Topological sort of graph nodes (Kahn's algorithm).
/// Returns node IDs in dependency order: producers before consumers.
fn topological_order(graph: &Graph) -> Vec<NodeId> {
    let n = graph.nodes().len();
    let mut in_degree = vec![0u32; n];
    // Consumer edges with multiplicity: a node listing the same input
    // twice (Mul(x, x), grad-sum Add(g, g)) contributes two edges and
    // needs two decrements to activate.
    let mut consumers: Vec<Vec<NodeId>> = vec![Vec::new(); n];
    for node in graph.nodes() {
        in_degree[node.id as usize] = node.inputs.len() as u32;
        for &input in &node.inputs {
            consumers[input as usize].push(node.id);
        }
    }

    let mut queue: std::collections::VecDeque<NodeId> = std::collections::VecDeque::new();
    for node in graph.nodes() {
        if in_degree[node.id as usize] == 0 {
            queue.push_back(node.id);
        }
    }

    let mut order = Vec::with_capacity(n);
    while let Some(id) = queue.pop_front() {
        order.push(id);
        for &c in &consumers[id as usize] {
            in_degree[c as usize] -= 1;
            if in_degree[c as usize] == 0 {
                queue.push_back(c);
            }
        }
    }

    // Any unvisited nodes (cycles or disconnected) — append in ID order
    if order.len() < n {
        let mut visited = vec![false; n];
        for &id in &order {
            visited[id as usize] = true;
        }
        for node in graph.nodes() {
            if !visited[node.id as usize] {
                order.push(node.id);
            }
        }
    }

    order
}

pub fn compile(graph: &Graph) -> ExecutionPlan {
    compile_with(graph, &CompileOptions::default())
}

pub fn compile_with(graph: &Graph, options: &CompileOptions) -> ExecutionPlan {
    compile_with_caps(graph, options, crate::codegen::coop_caps())
}

/// Compile for a concrete cooperative-matrix capability set.
///
/// `build()` uses this path after probing the context it will attach to the
/// session. Keeping the target explicit avoids a process-global "first GPU
/// wins" decision when an application owns multiple adapters. The public
/// [`compile_with`] entry point retains the installed global capabilities for
/// callers that intentionally compile a plan before constructing a session.
pub(crate) fn compile_with_caps(
    graph: &Graph,
    options: &CompileOptions,
    coop_caps: crate::codegen::CoopCaps,
) -> ExecutionPlan {
    let allow_reduced_precision_attention_backward = options.flash_backward_coop;
    compile_with_caps_policy(
        graph,
        options,
        coop_caps,
        allow_reduced_precision_attention_backward,
    )
}

fn compile_with_caps_policy(
    graph: &Graph,
    options: &CompileOptions,
    coop_caps: crate::codegen::CoopCaps,
    allow_reduced_precision_attention_backward: bool,
) -> ExecutionPlan {
    let mut compiler = Compiler::new_with_options(
        graph,
        options.clone(),
        coop_caps,
        allow_reduced_precision_attention_backward,
    );
    compiler.compile();

    // Propagate derived parameter info from graph to plan
    for dp in &graph.derived_params {
        if let Some(&(_, buf_ref)) = compiler
            .plan
            .param_buffers
            .iter()
            .find(|entry| entry.0 == dp.name)
        {
            let sources: Vec<(String, usize)> = dp
                .sources
                .iter()
                .map(|entry| (entry.0.clone(), entry.1))
                .collect();
            compiler
                .plan
                .derived_params
                .push((buf_ref, sources, dp.transform.clone()));
        }
    }

    if options.fuse_dispatches {
        fuse_epilogues(&mut compiler.plan);
        if options.use_schedule_pointwise {
            fuse_pointwise_chains(&mut compiler.plan);
        }
        if options.use_schedule_reduction {
            fuse_reduction_chains(&mut compiler.plan);
        }
        fuse_row_scaled_scatters(&mut compiler.plan);
    }
    // RmsNorm+MatMul prologue fusion is applied later in the runtime, after
    // per-dispatch coop selection — the prologue path currently only has a
    // coop-matmul implementation. See Session::with_context.

    compiler.plan
}

/// Fuse the table-gradient chain produced by
/// `sum_inner(embedding(indices, table) * factors)`:
///
/// `BroadcastInner(row_grad) -> Mul(factors) -> ScatterAddAtomic`
///
/// into one row-scaled atomic scatter. For narrow rows the fused shader maps
/// one invocation to a complete source row; wider rows retain the scalar work
/// mapping. The zeroing pass stays unchanged and the two large intermediate
/// buffers disappear.
fn fuse_row_scaled_scatters(plan: &mut ExecutionPlan) {
    use crate::schedule::Pw;
    use std::collections::{HashMap, HashSet};

    let mut protected: HashSet<BufferRef> = HashSet::new();
    protected.extend(plan.output_buffers.iter().copied());
    if let Some(buffer) = plan.loss_buffer {
        protected.insert(buffer);
    }
    protected.extend(plan.param_buffers.iter().map(|entry| entry.1));
    protected.extend(plan.input_buffers.iter().map(|entry| entry.1));
    protected.extend(plan.constant_buffers.iter().map(|entry| entry.0));
    protected.extend(plan.lse_buffers.iter().map(|entry| entry.1));
    for dispatch in &plan.dispatches {
        protected.extend(dispatch.extra_outputs.iter().copied());
    }

    loop {
        let mut producer = HashMap::new();
        let mut reads = HashMap::new();
        for (index, dispatch) in plan.dispatches.iter().enumerate() {
            producer.insert(dispatch.output_buffer, index);
            for buffer in &dispatch.input_buffers {
                *reads.entry(*buffer).or_insert(0usize) += 1;
            }
            for buffer in &dispatch.epilogue_buffers {
                *reads.entry(*buffer).or_insert(0usize) += 1;
            }
        }

        let mut candidate = None;
        for (scatter_index, scatter) in plan.dispatches.iter().enumerate() {
            if scatter.shader != ShaderEntry::ScatterAddAtomic
                || scatter.params.get(3) != Some(&0)
                || scatter.input_buffers.len() != 3
            {
                continue;
            }
            let indices = scatter.input_buffers[0];
            let product = scatter.input_buffers[1];
            let Some(&mul_index) = producer.get(&product) else {
                continue;
            };
            let mul = &plan.dispatches[mul_index];
            let plain_mul = mul.shader == ShaderEntry::Mul
                && mul.input_buffers.len() == 2
                && mul.reduction.is_none()
                && match mul.pointwise.as_ref() {
                    None => true,
                    Some(dag) => {
                        dag.n_inputs == 2
                            && dag.ops == [Pw::LoadInput(0), Pw::LoadInput(1), Pw::Mul(0, 1)]
                            && dag.output == 2
                    }
                };
            if !plain_mul || protected.contains(&product) {
                continue;
            }

            let broadcast_side = mul
                .input_buffers
                .iter()
                .enumerate()
                .find_map(|(side, buffer)| match producer.get(buffer).copied() {
                    Some(index) if plan.dispatches[index].is_inner_broadcast() => {
                        Some((side, index))
                    }
                    _ => None,
                });
            let Some((broadcast_side, broadcast_index)) = broadcast_side else {
                continue;
            };
            let broadcast = &plan.dispatches[broadcast_index];
            let broadcast_output = broadcast.output_buffer;
            let factors = mul.input_buffers[1 - broadcast_side];
            if broadcast.input_buffers.len() != 1
                || protected.contains(&broadcast_output)
                || reads.get(&broadcast_output).copied() != Some(1)
                || reads.get(&product).copied() != Some(2)
            {
                continue;
            }

            let Some((zero_index, _)) = plan.dispatches.iter().enumerate().find(|entry| {
                let dispatch = entry.1;
                dispatch.is_zero_fill()
                    && dispatch.output_buffer == scatter.output_buffer
                    && dispatch.params.first() == scatter.params.first()
            }) else {
                continue;
            };
            if scatter.params.len() < 4 {
                continue;
            }
            let source_len = scatter.params[1].saturating_mul(scatter.params[2]);
            if mul.params != [source_len, 0, 0, 0]
                || broadcast.params != [source_len, scatter.params[2], 1, 0]
                || scatter.workgroups != [scatter.params[2].div_ceil(256), 1, 1]
            {
                continue;
            }
            candidate = Some((
                scatter_index,
                zero_index,
                mul_index,
                broadcast_index,
                indices,
                factors,
                broadcast.input_buffers[0],
            ));
            break;
        }

        let Some((
            scatter_index,
            zero_index,
            mul_index,
            broadcast_index,
            indices,
            factors,
            row_scale,
        )) = candidate
        else {
            break;
        };

        let output = plan.dispatches[scatter_index].output_buffer;
        let total = plan.dispatches[scatter_index].params[0];
        let scatter = &mut plan.dispatches[scatter_index];
        scatter.input_buffers = vec![indices, factors, row_scale, output];
        let small_row = scatter.params[2] <= 16;
        scatter.params[3] = if small_row { 2 } else { 1 };
        // Column-parallel non-atomic kernel: one thread per embedding
        // column, regardless of the historical row-mapping pad flag.
        scatter.workgroups = [scatter.params[2].div_ceil(256), 1, 1];
        scatter.pointwise = None;
        scatter.label = format!("ScatterAddAtomicRowMul[{total}]");
        // The zero entry point does not read `src`, but its shared binding
        // layout still requires a valid buffer. Stop it from retaining the
        // now-eliminated product buffer.
        plan.dispatches[zero_index].input_buffers = vec![factors];

        let mut remove = [mul_index, broadcast_index];
        remove.sort_unstable();
        for index in remove.into_iter().rev() {
            plan.dispatches.remove(index);
        }
    }
}

/// Post-compile pass: merge sequential single-use pointwise dispatches into
/// a single deeper-DAG dispatch, eliminating the intermediate buffer and
/// the barrier between them. Only runs when
/// `CompileOptions::use_schedule_pointwise` is set (pointwise dispatches
/// carry the DAG the pass needs).
///
/// Conservative criteria — a producer P is fused into consumer C only when:
///   1. Both `P.pointwise` and `C.pointwise` are `Some`.
///   2. P's output buffer is read by exactly one dispatch (C) and appears
///      in no plan-level role (output/loss/param/input/constant/extra).
///   3. C's workgroups match P's (same output length).
///   4. Exactly one of C's input buffers equals P's output buffer (no
///      diamond — the intermediate is consumed in one slot only).
fn fuse_pointwise_chains(plan: &mut ExecutionPlan) {
    use std::collections::{HashMap, HashSet};

    // Buffers we must not eliminate. Anything here is preserved even if it
    // looks single-use from the dispatch list alone.
    let mut protected: HashSet<BufferRef> = HashSet::new();
    protected.extend(plan.output_buffers.iter().copied());
    if let Some(b) = plan.loss_buffer {
        protected.insert(b);
    }
    for entry in &plan.param_buffers {
        protected.insert(entry.1);
    }
    for entry in &plan.input_buffers {
        protected.insert(entry.1);
    }
    for entry in &plan.constant_buffers {
        protected.insert(entry.0);
    }
    for entry in &plan.lse_buffers {
        protected.insert(entry.1);
    }

    // Iterate until no more fusions apply.
    loop {
        // Recompute indices each pass since dispatches shift.
        let n = plan.dispatches.len();

        // Producer: output_buffer -> dispatch index.
        let mut producer: HashMap<BufferRef, usize> = HashMap::new();
        for (i, d) in plan.dispatches.iter().enumerate() {
            producer.insert(d.output_buffer, i);
        }

        // Reader counts.
        let mut reads: HashMap<BufferRef, usize> = HashMap::new();
        for d in &plan.dispatches {
            for b in &d.input_buffers {
                *reads.entry(*b).or_default() += 1;
            }
            // extra_outputs are also "referenced"; count them as protected.
            for b in &d.extra_outputs {
                protected.insert(*b);
            }
            for b in &d.epilogue_buffers {
                *reads.entry(*b).or_default() += 1;
            }
        }

        let mut fused_any = false;
        for ci in 0..n {
            let c = &plan.dispatches[ci];
            if c.pointwise.is_none() || c.fusion_barrier {
                continue;
            }

            // Find a fusion candidate: exactly one input slot that resolves
            // to a pointwise producer satisfying the criteria.
            let mut candidate: Option<(u8, usize)> = None; // (input_idx, producer_dispatch_idx)
            for (slot_idx, buf) in c.input_buffers.iter().enumerate() {
                if protected.contains(buf) {
                    continue;
                }
                let Some(&pi) = producer.get(buf) else {
                    continue;
                };
                if pi == ci {
                    continue;
                }
                let p = &plan.dispatches[pi];
                if p.pointwise.is_none() || p.fusion_barrier {
                    continue;
                }
                if reads.get(buf).copied().unwrap_or(0) != 1 {
                    continue;
                }
                // Same per-element workload (len).
                if p.workgroups != c.workgroups {
                    continue;
                }
                if p.params.first() != c.params.first() {
                    continue;
                }
                // The consumer must read this buffer in exactly one slot.
                let slot_count = c.input_buffers.iter().filter(|b| *b == buf).count();
                if slot_count != 1 {
                    continue;
                }
                // Arity cap: the runtime binds pointwise pipelines via
                // UnaryData (n=1), BinaryData (n=2), or TernaryData (n=3).
                // A higher-arity fused DAG would need a wider layout we
                // don't plumb yet.
                let new_arity = p.input_buffers.len() + c.input_buffers.len() - 1;
                if new_arity > 3 {
                    continue;
                }
                candidate = Some((slot_idx as u8, pi));
                break;
            }

            let Some((input_idx, pi)) = candidate else {
                continue;
            };

            // Perform the fusion.
            let producer_d = plan.dispatches[pi].clone();
            let consumer_d = &mut plan.dispatches[ci];

            let p_dag = producer_d.pointwise.expect("checked above");
            let c_dag = consumer_d
                .pointwise
                .as_ref()
                .expect("checked above")
                .clone();
            let fused_dag = c_dag.fuse_input(input_idx, &p_dag);

            // Rebuild consumer input_buffers: producer inputs, then
            // consumer inputs with the fused slot removed, in order.
            let mut new_inputs: Vec<BufferRef> = producer_d.input_buffers.clone();
            for (idx, b) in consumer_d.input_buffers.iter().enumerate() {
                if idx as u8 != input_idx {
                    new_inputs.push(*b);
                }
            }
            consumer_d.input_buffers = new_inputs;
            consumer_d.pointwise = Some(fused_dag);
            consumer_d.origin.extend(producer_d.origin.iter().copied());
            // The consumer now reads from more buffers; its ShaderEntry
            // (used only to pick the data layout) must reflect the new
            // arity. The runtime binds via UnaryData for n=1, BinaryData
            // for n=2; arities >2 would need a wider layout we don't yet
            // plumb. Guard against that.
            // Update the sentinel `shader` so the (legacy) pipeline
            // lookup still resolves — actual binding/pipeline come from
            // the `pointwise` DAG's arity.
            let new_arity = consumer_d.input_buffers.len();
            consumer_d.shader = match new_arity {
                1 => ShaderEntry::Relu,
                2 => ShaderEntry::Add,
                3 => ShaderEntry::SwiGLUGradGate, // dummy for TernaryData layout
                _ => unreachable!("arity capped at 3 by candidate-selection guard"),
            };

            // Drop the producer dispatch.
            plan.dispatches.remove(pi);
            fused_any = true;
            break;
        }

        if !fused_any {
            break;
        }
    }
}

fn shared_pointwise_consumers_are_foldable(
    plan: &ExecutionPlan,
    buffer: BufferRef,
    producer: &Dispatch,
    read_count: usize,
) -> bool {
    let mut foldable_reads = 0usize;
    for dispatch in &plan.dispatches {
        if dispatch.epilogue_buffers.contains(&buffer) {
            return false;
        }
        let occurrences = dispatch
            .input_buffers
            .iter()
            .filter(|&&input| input == buffer)
            .count();
        if occurrences == 0 {
            continue;
        }
        if occurrences > 1 {
            return false;
        }
        let Some(ref kernel) = dispatch.reduction else {
            return false;
        };
        if kernel.input_row_repeats.iter().any(|&factor| factor != 1) {
            return false;
        }
        let per_elem = kernel.n_per_elem as usize;
        let input_index = dispatch
            .input_buffers
            .iter()
            .position(|&input| input == buffer)
            .unwrap();
        if kernel.gather_elem.iter().any(|&g| g)
            || kernel.n_per_row != 0
            || input_index >= per_elem
            || producer.params.first().copied()
                != Some(dispatch.params[0].saturating_mul(dispatch.params[1]))
            || per_elem - 1 + producer.input_buffers.len() > 3
        {
            return false;
        }
        foldable_reads += 1;
    }
    foldable_reads == read_count
}

fn shared_embedding_consumers_are_foldable(
    plan: &ExecutionPlan,
    buffer: BufferRef,
    outer: u32,
    inner: u32,
    read_count: usize,
) -> bool {
    let mut foldable_reads = 0usize;
    for dispatch in &plan.dispatches {
        if dispatch.epilogue_buffers.contains(&buffer) {
            return false;
        }
        let occurrences = dispatch
            .input_buffers
            .iter()
            .filter(|&&input| input == buffer)
            .count();
        if occurrences == 0 {
            continue;
        }
        if occurrences > 1 {
            return false;
        }
        let Some(ref kernel) = dispatch.reduction else {
            return false;
        };
        if kernel.input_row_repeats.iter().any(|&factor| factor != 1) {
            return false;
        }
        if dispatch.params.first().copied() != Some(outer)
            || dispatch.params.get(1).copied() != Some(inner)
        {
            return false;
        }
        let mut stream_position = 0usize;
        let mut is_direct_stream = false;
        for stream in 0..kernel.n_per_elem as usize {
            let is_gather = kernel.gather_elem.get(stream).copied().unwrap_or(false);
            if !is_gather && dispatch.input_buffers.get(stream_position) == Some(&buffer) {
                is_direct_stream = true;
                break;
            }
            stream_position += if is_gather { 2 } else { 1 };
        }
        if !is_direct_stream {
            return false;
        }
        foldable_reads += 1;
    }
    foldable_reads == read_count
}

/// Post-compile pass: fold producers into a reduction dispatch's prologue,
/// eliminating intermediate buffers. Two phases:
///
///   Phase 1 (pointwise → prologue): a per-element input of the reduction
///   produced by a pointwise dispatch is folded into the prologue DAG via
///   `PointwiseDAG::fuse_input` (grows `n_per_elem`). Shared producers are
///   cloned only when all of their consumers are compatible reductions.
///   Runs while the reduction has no gather streams, so `input_buffers`
///   stays 1:1 with prologue inputs (no gather expansion to track).
///
///   Phase 2 (gather → prologue): a per-element input produced by an
///   `Embedding` (indexed load) is marked a gather stream —
///   `gather_elem[s] = true`, and the stream's buffer is replaced by the
///   table with the indices buffer spliced in right after. Shared embeddings
///   are folded only when all consumers are compatible reductions. This is
///   valid because the gathered axis is the reduced (inner) axis: `embedding`
///   params `[seq, hidden] = [outer, inner]`.
///
/// Together these let `sum_inner(mul(embedding(idx,tbl), x))` collapse to
/// one fused reduction kernel — the SH colour path — discovered from
/// primitive ops, not hand-written.
fn fuse_reduction_chains(plan: &mut ExecutionPlan) {
    use std::collections::{HashMap, HashSet};

    let protected = |plan: &ExecutionPlan| -> HashSet<BufferRef> {
        let mut p: HashSet<BufferRef> = HashSet::new();
        p.extend(plan.output_buffers.iter().copied());
        if let Some(b) = plan.loss_buffer {
            p.insert(b);
        }
        for e in &plan.param_buffers {
            p.insert(e.1);
        }
        for e in &plan.input_buffers {
            p.insert(e.1);
        }
        for e in &plan.constant_buffers {
            p.insert(e.0);
        }
        for e in &plan.lse_buffers {
            p.insert(e.1);
        }
        for d in &plan.dispatches {
            for b in &d.extra_outputs {
                p.insert(*b);
            }
        }
        p
    };

    // Helper: producer map + read counts for the current plan.
    let scan = |plan: &ExecutionPlan| -> (HashMap<BufferRef, usize>, HashMap<BufferRef, usize>) {
        let mut producer = HashMap::new();
        for (i, d) in plan.dispatches.iter().enumerate() {
            producer.insert(d.output_buffer, i);
        }
        let mut reads: HashMap<BufferRef, usize> = HashMap::new();
        for d in &plan.dispatches {
            for b in &d.input_buffers {
                *reads.entry(*b).or_default() += 1;
            }
            for b in &d.epilogue_buffers {
                *reads.entry(*b).or_default() += 1;
            }
        }
        (producer, reads)
    };

    // --- Phase 1: fold pointwise producers into reduction prologues. ---
    loop {
        let prot = protected(plan);
        let (producer, reads) = scan(plan);
        let mut fused = false;

        'outer: for ci in 0..plan.dispatches.len() {
            let c = &plan.dispatches[ci];
            let Some(kernel) = c.reduction.as_ref() else {
                continue;
            };
            // Phase 1 invariant: no gather streams yet, so input_buffers
            // is 1:1 with prologue inputs (per-elem first, then per-row).
            if kernel.gather_elem.iter().any(|&g| g)
                || kernel.input_row_repeats.iter().any(|&factor| factor != 1)
            {
                continue;
            }
            let outer = c.params[0];
            let inner = c.params[1];
            let per_elem = kernel.n_per_elem as usize;

            for s in 0..per_elem {
                let buf = c.input_buffers[s];
                if prot.contains(&buf) {
                    continue;
                }
                let Some(&pi) = producer.get(&buf) else {
                    continue;
                };
                if pi == ci {
                    continue;
                }
                let p = &plan.dispatches[pi];
                if p.pointwise.is_none() || p.reduction.is_some() || p.fusion_barrier {
                    continue;
                }
                // Producer must cover the per-element domain (outer*inner).
                if p.params.first().copied() != Some(outer.saturating_mul(inner)) {
                    continue;
                }
                // Arity cap (binding vocab supports ≤3 per-elem streams).
                let new_n_per_elem = per_elem - 1 + p.input_buffers.len();
                if new_n_per_elem > 3 || kernel.n_per_row != 0 {
                    continue;
                }

                let read_count = reads.get(&buf).copied().unwrap_or(0);
                if read_count > 1
                    && !shared_pointwise_consumers_are_foldable(plan, buf, p, read_count)
                {
                    continue;
                }

                let producer_d = plan.dispatches[pi].clone();
                let p_dag = producer_d.pointwise.clone().expect("checked");
                let c = &mut plan.dispatches[ci];
                let kernel = c.reduction.as_mut().expect("checked");
                kernel.prologue = kernel.prologue.fuse_input(s as u8, &p_dag);
                for prologue in &mut kernel.extra_prologues {
                    *prologue = prologue.fuse_input(s as u8, &p_dag);
                }
                // The reduction epilogue sees the same per-element streams
                // before its per-column and reduced-value inputs. If it
                // references the folded stream, expand that input there as
                // well; otherwise its declared arity no longer matches
                // n_per_elem and lowering aborts.
                if let Some(epilogue) = kernel.epilogue.as_mut() {
                    epilogue.dag = epilogue.dag.fuse_input(s as u8, &p_dag);
                }
                kernel.n_per_elem = new_n_per_elem as u8;
                kernel.gather_elem = Vec::new();
                // Rebuild input_buffers: producer inputs first (matching
                // fuse_input's ordering), then consumer's others.
                let mut new_inputs = producer_d.input_buffers.clone();
                for (idx, b) in c.input_buffers.iter().enumerate() {
                    if idx != s {
                        new_inputs.push(*b);
                    }
                }
                c.input_buffers = new_inputs;
                c.origin.extend(producer_d.origin.iter().copied());
                if read_count == 1 {
                    plan.dispatches.remove(pi);
                }
                fused = true;
                break 'outer;
            }
        }
        if !fused {
            break;
        }
    }

    // --- Phase 2: fold Embedding producers as gather streams. ---
    loop {
        let prot = protected(plan);
        let (producer, reads) = scan(plan);
        let mut fused = false;

        'outer2: for ci in 0..plan.dispatches.len() {
            let c = &plan.dispatches[ci];
            let Some(kernel) = c.reduction.as_ref() else {
                continue;
            };
            if kernel.input_row_repeats.iter().any(|&factor| factor != 1) {
                continue;
            }
            let outer = c.params[0];
            let inner = c.params[1];
            let per_elem = kernel.n_per_elem as usize;

            // Flat position of each per-element stream in input_buffers,
            // accounting for earlier gather streams (which occupy 2 slots).
            let mut flat = Vec::with_capacity(per_elem);
            let mut pos = 0usize;
            for s in 0..per_elem {
                flat.push(pos);
                pos += if kernel.gather_elem.get(s).copied().unwrap_or(false) {
                    2
                } else {
                    1
                };
            }

            for (s, &flat_pos) in flat.iter().enumerate() {
                if kernel.gather_elem.get(s).copied().unwrap_or(false) {
                    continue; // already a gather leaf
                }
                let buf = c.input_buffers[flat_pos];
                if prot.contains(&buf) {
                    continue;
                }
                let Some(&pi) = producer.get(&buf) else {
                    continue;
                };
                if pi == ci {
                    continue;
                }
                let p = &plan.dispatches[pi];
                // Plain Embedding dispatch: indexed load, gathered axis ==
                // reduced axis (params [seq, hidden] = [outer, inner]).
                let is_embedding = p.shader == ShaderEntry::Embedding
                    && p.reduction.is_none()
                    && p.pointwise.is_none()
                    && p.params.first().copied() == Some(outer)
                    && p.params.get(1).copied() == Some(inner)
                    && p.input_buffers.len() == 2;
                if !is_embedding {
                    continue;
                }
                let read_count = reads.get(&buf).copied().unwrap_or(0);
                if read_count > 1
                    && !shared_embedding_consumers_are_foldable(plan, buf, outer, inner, read_count)
                {
                    continue;
                }
                let idx_buf = p.input_buffers[0]; // Embedding inputs[0] = indices
                let table_buf = p.input_buffers[1]; // inputs[1] = table
                let producer_origin = p.origin.clone();

                let c = &mut plan.dispatches[ci];
                let kernel = c.reduction.as_mut().expect("checked");
                if kernel.gather_elem.is_empty() {
                    kernel.gather_elem = vec![false; per_elem];
                }
                kernel.gather_elem[s] = true;
                // Replace stream s's buffer (the embedding output) with the
                // table, and splice the indices buffer right after it.
                c.input_buffers[flat_pos] = table_buf;
                c.input_buffers.insert(flat_pos + 1, idx_buf);
                c.origin.extend(producer_origin);
                // Drop the embedding dispatch if it's now unused.
                if reads.get(&buf).copied().unwrap_or(0) == 1 {
                    plan.dispatches.remove(pi);
                }
                fused = true;
                break 'outer2;
            }
        }
        if !fused {
            break;
        }
    }
}

/// Post-compile pass: fuse single-consumer `RmsNorm → MatMul` pairs into
/// `RmsNormRsqrt + MatMul-with-prologue`. Instead of computing the full
/// normalized output and writing it to DRAM, the matmul reads raw x and
/// multiplies by pre-computed rsqrt and weight during A-tile staging.
///
/// Unlike the deleted single-dispatch fused form (which computed rsqrt
/// INSIDE the matmul via 64-thread tree reduction → 25% regression), this
/// uses a separate lightweight `RmsNormRsqrt` dispatch, so the matmul
/// prologue is only two scalar multiplies per A element — essentially free.
/// Dispatches must already have their runtime `use_coop` decision; scalar
/// matmuls are deliberately left unchanged.
/// Fold a RmsNorm into every GEMV that consumes it, removing both the
/// norm's dispatch and the boundary that separated it.
///
/// The norm's own work is trivial — one workgroup over `hidden` elements —
/// while each consuming GEMV already streams a `K`-element input vector, so
/// reading the un-normalized one and scaling on the way past costs no extra
/// traffic; only the per-workgroup sum of squares is duplicated. Requires
/// every consumer to be a GEMV and nothing else to read the normalized
/// values.
pub fn fuse_rmsnorm_into_gemv(plan: &mut ExecutionPlan) {
    use std::collections::HashMap;

    let mut readers: HashMap<BufferRef, Vec<usize>> = HashMap::new();
    for (i, d) in plan.dispatches.iter().enumerate() {
        for buf in &d.input_buffers {
            readers.entry(*buf).or_default().push(i);
        }
    }
    let mut external: std::collections::HashSet<BufferRef> = Default::default();
    external.extend(plan.output_buffers.iter().copied());
    if let Some(b) = plan.loss_buffer {
        external.insert(b);
    }
    for entry in &plan.param_buffers {
        external.insert(entry.1);
    }
    for entry in &plan.input_buffers {
        external.insert(entry.1);
    }

    let mut drop_norm: Vec<usize> = Vec::new();
    let mut rewrite: Vec<(usize, BufferRef, BufferRef, u32)> = Vec::new();
    for (ni, norm) in plan.dispatches.iter().enumerate() {
        if norm.shader != ShaderEntry::RmsNorm || norm.input_buffers.len() < 2 {
            continue;
        }
        let normed = norm.output_buffer;
        if external.contains(&normed) {
            continue;
        }
        // Only fuse a single-row norm; the fused kernel assumes M = 1.
        if norm.params.first().copied().unwrap_or(0) != 1 {
            continue;
        }
        let Some(consumers) = readers.get(&normed) else {
            continue;
        };
        if consumers.is_empty()
            || consumers.iter().any(|&c| {
                let d = &plan.dispatches[c];
                d.shader != ShaderEntry::MatMulGemv || d.input_buffers.first() != Some(&normed)
            })
        {
            continue;
        }
        let (src, weight, eps_bits) = (
            norm.input_buffers[0],
            norm.input_buffers[1],
            norm.params.get(2).copied().unwrap_or(0),
        );
        for &c in consumers {
            rewrite.push((c, src, weight, eps_bits));
        }
        drop_norm.push(ni);
    }
    if drop_norm.is_empty() {
        return;
    }

    for (idx, src, weight, eps_bits) in rewrite {
        let d = &mut plan.dispatches[idx];
        d.input_buffers[0] = src;
        d.gemv_rmsnorm = Some(GemvRmsNorm { weight, eps_bits });
    }
    let dropped = drop_norm.len();
    drop_norm.sort_unstable();
    for ni in drop_norm.into_iter().rev() {
        plan.dispatches.remove(ni);
    }
    log::info!("fuse_rmsnorm_into_gemv: folded {dropped} RmsNorm dispatches into their GEMVs");
}

pub fn fuse_rmsnorm_prologues(plan: &mut ExecutionPlan) {
    use std::collections::HashMap;

    let mut producer: HashMap<BufferRef, usize> = HashMap::new();
    for (i, d) in plan.dispatches.iter().enumerate() {
        producer.insert(d.output_buffer, i);
    }

    let mut read_count: HashMap<BufferRef, usize> = HashMap::new();
    for d in &plan.dispatches {
        for buf in &d.input_buffers {
            *read_count.entry(*buf).or_default() += 1;
        }
    }

    // Collect protected buffers (graph outputs, params, etc.)
    let mut external: std::collections::HashSet<BufferRef> = Default::default();
    external.extend(plan.output_buffers.iter().copied());
    if let Some(b) = plan.loss_buffer {
        external.insert(b);
    }
    for entry in &plan.param_buffers {
        external.insert(entry.1);
    }
    for entry in &plan.input_buffers {
        external.insert(entry.1);
    }

    let mut to_fuse: Vec<(usize, usize)> = Vec::new(); // (norm_idx, matmul_idx)

    for (i, d) in plan.dispatches.iter().enumerate() {
        // Find MatMul dispatches whose input_buffers[0] comes from a
        // single-consumer RmsNorm. The scalar pipeline cannot execute a
        // matmul prologue, so only transform dispatches that runtime policy
        // has already selected for cooperative matrices.
        if !d.use_coop
            || !matches!(
                d.shader,
                ShaderEntry::MatMul
                    | ShaderEntry::MatMulAT
                    | ShaderEntry::FusedMatMulAdd
                    | ShaderEntry::FusedMatMulATAdd
            )
        {
            continue;
        }
        // Skip GEMV variants (M=1) — those use a different kernel path.
        if matches!(
            d.shader,
            ShaderEntry::MatMulGemv | ShaderEntry::MatMulGemvAdd | ShaderEntry::MatMulGemvBT
        ) {
            continue;
        }
        if d.input_buffers.is_empty() {
            continue;
        }
        let a_buf = d.input_buffers[0];
        if external.contains(&a_buf) {
            continue;
        }
        let Some(&norm_idx) = producer.get(&a_buf) else {
            continue;
        };
        let norm = &plan.dispatches[norm_idx];
        if norm.shader != ShaderEntry::RmsNorm {
            continue;
        }
        // RmsNorm output must be single-consumer (only this matmul reads it).
        if read_count.get(&a_buf).copied().unwrap_or(0) != 1 {
            continue;
        }
        to_fuse.push((norm_idx, i));
    }

    for &(norm_idx, matmul_idx) in &to_fuse {
        let norm = &plan.dispatches[norm_idx];
        let x_buf = norm.input_buffers[0]; // raw x
        let w_norm_buf = norm.input_buffers[1]; // norm weight
        let rows = norm.params[0];
        let cols = norm.params[1];
        let eps_bits = norm.params[2];

        // Allocate rsqrt_cache buffer: one f32 per row.
        let rsqrt_buf_idx = plan.buffers.len() as u32;
        plan.buffers.push((rows as usize) * 4);
        let rsqrt_buf = BufferRef(rsqrt_buf_idx);

        // Replace the RmsNorm dispatch with RmsNormRsqrt, keeping its
        // provenance.
        let norm_origin = plan.dispatches[norm_idx].origin.clone();
        let norm_label = plan.dispatches[norm_idx].label.clone();
        plan.dispatches[norm_idx] = Dispatch {
            shader: ShaderEntry::RmsNormRsqrt,
            workgroups: [rows, 1, 1],
            input_buffers: vec![x_buf],
            output_buffer: rsqrt_buf,
            extra_outputs: vec![],
            params: vec![rows, cols, eps_bits, 0],
            use_coop: false,
            use_small_tiles: false,
            origin: norm_origin.clone(),
            label: norm_label,
            ..Default::default()
        };

        // Modify the matmul: read raw x instead of normalized x. The
        // normalization now happens inside the matmul, so it inherits the
        // norm node's provenance too.
        plan.dispatches[matmul_idx].input_buffers[0] = x_buf;
        plan.dispatches[matmul_idx].origin.extend(norm_origin);

        // Attach the prologue: multiply A-elements by rsqrt[gr] and
        // w_norm[tc]. Also declare both factor buffers as dispatch inputs so
        // scheduling and memory planning preserve the producer dependency and
        // lifetime; shader binding still uses the typed prologue metadata.
        let factors = vec![
            (rsqrt_buf, PrologueLoadKind::PerRow),
            (w_norm_buf, PrologueLoadKind::PerKCol),
        ];
        plan.dispatches[matmul_idx]
            .input_buffers
            .extend(factors.iter().map(|&(buffer, _)| buffer));
        plan.dispatches[matmul_idx].matmul_prologue = Some(MatMulPrologue { factors });
    }

    if !to_fuse.is_empty() {
        log::info!(
            "fuse_rmsnorm_prologues: fused {} RmsNorm+MatMul pairs",
            to_fuse.len()
        );
    }
}

/// Absorb unary, single-consumer pointwise dispatches into a preceding f32
/// matmul. Runtime selection may subsequently use either the scalar or
/// cooperative epilogue implementation.
fn fuse_epilogues(plan: &mut ExecutionPlan) {
    use std::collections::{HashMap, HashSet};
    // Buffers that must keep their producer unchanged: anything the rest
    // of the plan references by buffer ref (user outputs, loss, params,
    // inputs, constants, …). If the matmul's output_buffer is one of
    // these, remapping it to the elementwise op's output buffer would
    // leave the protected buffer unwritten.
    let mut protected: HashSet<BufferRef> = HashSet::new();
    protected.extend(plan.output_buffers.iter().copied());
    if let Some(b) = plan.loss_buffer {
        protected.insert(b);
    }
    for entry in &plan.param_buffers {
        protected.insert(entry.1);
    }
    for entry in &plan.input_buffers {
        protected.insert(entry.1);
    }
    for entry in &plan.constant_buffers {
        protected.insert(entry.0);
    }

    let dispatches = &mut plan.dispatches;
    // Map: output buffer → dispatch index that writes it.
    let mut producer: HashMap<BufferRef, usize> = HashMap::new();
    for (i, d) in dispatches.iter().enumerate() {
        producer.insert(d.output_buffer, i);
    }

    // Count how many dispatches read each buffer (consumers).
    let mut read_count: HashMap<BufferRef, usize> = HashMap::new();
    for d in dispatches.iter() {
        for buf in &d.input_buffers {
            *read_count.entry(*buf).or_default() += 1;
        }
    }

    let mut to_remove = Vec::new();

    for i in 0..dispatches.len() {
        let d = &dispatches[i];
        // Only consider single-input unary elementwise ops.
        // Binary ops (BiasAdd, Add) would require extra buffer bindings in the
        // shader data layout, which is a larger change. TODO: extend shader data
        // layouts to support dynamic extra bindings for binary epilogues.
        if d.input_buffers.len() != 1 {
            continue;
        }

        // If this dispatch carries a schedule-template DAG, it can still be
        // absorbed into a matmul epilogue as long as the DAG corresponds to
        // an epilogue-supported op (Relu/Silu/Sigmoid/Neg — checked below
        // via `d.shader`, which we preserve as the sentinel entry). The
        // DAG field gets discarded along with the dispatch itself.
        // Map the consumed op to a PointwiseDAG single-op applied to val.
        use crate::schedule::Pw;
        let d_shader = d.shader.clone();
        let pw_op = match d_shader {
            ShaderEntry::Relu => Pw::Relu(0),
            ShaderEntry::Sigmoid => Pw::Sigmoid(0),
            ShaderEntry::Neg => Pw::Neg(0),
            ShaderEntry::Silu => Pw::Silu(0),
            _ => continue,
        };
        let primary_buf = d.input_buffers[0];
        let elem_output = d.output_buffer;
        let consumer_requires_full_precision = d.requires_full_precision;

        // The elementwise op reads from primary_buf. Find the matmul that produced it.
        let Some(&prod_idx) = producer.get(&primary_buf) else {
            continue;
        };
        let prod = &dispatches[prod_idx];

        // Weighted kernels have a different B-buffer declaration and are
        // compiled through `generate_module_weighted`. The epilogue generator
        // currently emits only the f32 layout, so keep f16/Q4/Q8 matmuls
        // separate until those two codegen paths are composed explicitly.
        let is_matmul = matches!(
            prod.shader,
            ShaderEntry::MatMul
                | ShaderEntry::MatMulAT
                | ShaderEntry::MatMulBT
                | ShaderEntry::FusedMatMulAdd
                | ShaderEntry::FusedMatMulATAdd
                | ShaderEntry::FusedMatMulBTAdd
        );
        if !is_matmul || prod.use_coop || prod.weight_format != WeightFormat::F32 {
            continue;
        }

        // The matmul output must be consumed by ONLY this elementwise op
        // (otherwise we can't modify the output in-place).
        if read_count.get(&primary_buf).copied().unwrap_or(0) != 1 {
            continue;
        }

        // The matmul's output buffer cannot be remapped if downstream code
        // reads it by buffer-ref (user output, loss, param, input, constant).
        if protected.contains(&primary_buf) {
            continue;
        }

        // Build or extend the MatMulEpilogue DAG on the producer.
        if let Some(ref mut epi) = dispatches[prod_idx].matmul_epilogue {
            // Chain: append the new op referencing the current output.
            let prev_out = epi.dag.output;
            epi.dag.ops.push(pw_op.clone());
            // The new op's input index (0 in the single-op) needs to
            // reference prev_out. Since pw_op was built with Pw::Relu(0)
            // etc., the "0" refers to LoadInput(0). But we want it to
            // refer to the PREVIOUS output. Adjust: replace the inner
            // index with prev_out.
            let last = epi.dag.ops.len() - 1;
            epi.dag.ops[last] = match pw_op {
                Pw::Relu(_) => Pw::Relu(prev_out),
                Pw::Sigmoid(_) => Pw::Sigmoid(prev_out),
                Pw::Neg(_) => Pw::Neg(prev_out),
                Pw::Silu(_) => Pw::Silu(prev_out),
                _ => unreachable!(),
            };
            epi.dag.output = last as u16;
        } else {
            // First epilogue op: create DAG with LoadInput(0) = val.
            dispatches[prod_idx].matmul_epilogue = Some(MatMulEpilogue {
                dag: PointwiseDAG {
                    n_inputs: 1,
                    ops: vec![Pw::LoadInput(0), pw_op],
                    output: 1,
                },
                inputs: vec![],
            });
        }
        // Also maintain the legacy fields for backward compat.
        let legacy_op = match d_shader {
            ShaderEntry::Relu => EpilogueOp::Relu,
            ShaderEntry::Sigmoid => EpilogueOp::Sigmoid,
            ShaderEntry::Neg => EpilogueOp::Neg,
            ShaderEntry::Silu => EpilogueOp::Silu,
            _ => unreachable!(),
        };
        dispatches[prod_idx].epilogue.push(legacy_op);

        dispatches[prod_idx].requires_full_precision |= consumer_requires_full_precision;
        dispatches[prod_idx].output_buffer = elem_output;
        let absorbed_origin = dispatches[i].origin.clone();
        dispatches[prod_idx].origin.extend(absorbed_origin);
        producer.insert(elem_output, prod_idx);

        to_remove.push(i);
    }

    // Remove fused dispatches (iterate in reverse to preserve indices)
    for &idx in to_remove.iter().rev() {
        dispatches.remove(idx);
    }
}

/// Build a 1-input pointwise DAG equivalent to `shader`, or `None` if the
/// shader isn't a unary elementwise op. Used when
/// `CompileOptions::use_schedule_pointwise` is set to route unary ops
/// through the generated codegen path instead of the hand-written shader.
fn unary_shader_to_pointwise(shader: &ShaderEntry) -> Option<PointwiseDAG> {
    use crate::schedule::Pw;
    let op = match *shader {
        ShaderEntry::Relu => Pw::Relu(0),
        ShaderEntry::Sigmoid => Pw::Sigmoid(0),
        ShaderEntry::Tanh => Pw::Tanh(0),
        ShaderEntry::Neg => Pw::Neg(0),
        ShaderEntry::Abs => Pw::Abs(0),
        ShaderEntry::Log => Pw::Log(0),
        ShaderEntry::Recip => Pw::Recip(0),
        ShaderEntry::Silu => Pw::Silu(0),
        ShaderEntry::Gelu => {
            // Tanh-approx GELU, bit-matching unary.wgsl's `gelu` entry:
            // 0.5 * x * (1 + tanh(0.7978845608 * (x + 0.044715 * x³)))
            return Some(PointwiseDAG {
                n_inputs: 1,
                ops: vec![
                    Pw::LoadInput(0),           // v0 = x
                    Pw::Mul(0, 0),              // v1 = x²
                    Pw::Mul(1, 0),              // v2 = x³
                    Pw::const_f32(0.044715),    // v3
                    Pw::Mul(2, 3),              // v4 = 0.044715·x³
                    Pw::Add(0, 4),              // v5 = x + 0.044715·x³
                    Pw::const_f32(0.797_884_6), // v6 = sqrt(2/π)
                    Pw::Mul(5, 6),              // v7 = inner
                    Pw::Tanh(7),                // v8
                    Pw::const_f32(1.0),         // v9
                    Pw::Add(8, 9),              // v10 = 1 + tanh
                    Pw::const_f32(0.5),         // v11
                    Pw::Mul(0, 11),             // v12 = 0.5·x
                    Pw::Mul(12, 10),            // v13 = gelu
                ],
                output: 13,
            });
        }
        _ => return None,
    };
    Some(PointwiseDAG {
        n_inputs: 1,
        ops: vec![Pw::LoadInput(0), op],
        output: 1,
    })
}

/// Build a 2-input pointwise DAG equivalent to `shader`, or `None` if the
/// shader isn't a binary elementwise op using the shared `BinaryData`
/// binding layout.
fn binary_shader_to_pointwise(shader: &ShaderEntry) -> Option<PointwiseDAG> {
    use crate::schedule::Pw;
    // All binary shaders read two streams, named src_a/src_b by
    // `schedule::PointwiseDAG::input_binding_names(2)` to match binary.wgsl.
    let (ops, output) = match *shader {
        ShaderEntry::Add => (vec![Pw::LoadInput(0), Pw::LoadInput(1), Pw::Add(0, 1)], 2),
        ShaderEntry::Mul => (vec![Pw::LoadInput(0), Pw::LoadInput(1), Pw::Mul(0, 1)], 2),
        ShaderEntry::Greater => (
            vec![Pw::LoadInput(0), Pw::LoadInput(1), Pw::Greater(0, 1)],
            2,
        ),
        ShaderEntry::SwiGLU => (
            // swiglu(a, b) = silu(a) * b
            vec![
                Pw::LoadInput(0),
                Pw::LoadInput(1),
                Pw::Silu(0),
                Pw::Mul(2, 1),
            ],
            3,
        ),
        _ => return None,
    };
    Some(PointwiseDAG {
        n_inputs: 2,
        ops,
        output,
    })
}

const MAX_COMPUTE_WORKGROUPS_PER_DIMENSION: u32 = 65_535;

/// Tile a scalar matmul across Y and Z without exceeding the portable
/// per-dimension compute-dispatch limit.
fn matmul_workgroups(m: u32, n: u32, tile: u32) -> [u32; 3] {
    let columns = n.div_ceil(tile);
    assert!(
        columns <= MAX_COMPUTE_WORKGROUPS_PER_DIMENSION,
        "matmul needs {columns} workgroups on X, exceeding the portable limit"
    );
    let rows = m.div_ceil(tile);
    let depth = rows.div_ceil(MAX_COMPUTE_WORKGROUPS_PER_DIMENSION).max(1);
    assert!(
        depth <= MAX_COMPUTE_WORKGROUPS_PER_DIMENSION,
        "matmul needs {depth} workgroup layers, exceeding the portable limit"
    );
    let height = rows.div_ceil(depth);
    debug_assert!(height <= MAX_COMPUTE_WORKGROUPS_PER_DIMENSION);
    [columns, height, depth]
}

struct Compiler<'a> {
    graph: &'a Graph,
    plan: ExecutionPlan,
    /// Map from NodeId → BufferRef for each node's output.
    node_buffers: HashMap<NodeId, BufferRef>,
    options: CompileOptions,
    /// Capabilities of the GPU this plan is being compiled for. Flash
    /// attention selection is plan-time (unlike ordinary matmul selection),
    /// so this must be the eventual session's target rather than global state.
    coop_caps: crate::codegen::CoopCaps,
    /// Experimental f16-input attention backward is deliberately separate
    /// from device capability: availability does not imply adequate gradient
    /// accuracy.
    allow_reduced_precision_attention_backward: bool,
    /// Fused GradKV: maps fwd_node → pre-allocated dV buffer.
    /// When GradK is compiled, it emits a fused GradKV dispatch and
    /// pre-allocates the dV buffer here. When GradV is later compiled
    /// for the same fwd_node, it reuses this buffer and skips dispatching.
    fused_grad_kv_dv: HashMap<NodeId, BufferRef>,
}

impl<'a> Compiler<'a> {
    fn new_with_options(
        graph: &'a Graph,
        options: CompileOptions,
        coop_caps: crate::codegen::CoopCaps,
        allow_reduced_precision_attention_backward: bool,
    ) -> Self {
        Self {
            graph,
            plan: ExecutionPlan {
                buffers: vec![],
                param_buffers: vec![],
                input_buffers: Vec::new(),
                constant_buffers: Vec::new(),
                dispatches: Vec::new(),
                loss_buffer: None,
                output_buffers: Vec::new(),
                param_grad_pairs: Vec::new(),
                lse_buffers: Vec::new(),
                derived_params: Vec::new(),
                weight_buffers: HashMap::new(),
                node_buffers: Vec::new(),
                node_names: Vec::new(),
                knobs: options.knobs,
            },
            node_buffers: HashMap::new(),
            options,
            coop_caps,
            allow_reduced_precision_attention_backward,
            fused_grad_kv_dv: HashMap::new(),
        }
    }

    fn alloc_buffer(&mut self, size_bytes: usize) -> BufferRef {
        let idx = self.plan.buffers.len() as u32;
        self.plan.buffers.push(size_bytes);
        BufferRef(idx)
    }

    /// Buffer already allocated for a `CrossEntropyLogitsGrad` on the same
    /// `(logits, labels)` pair. Present only on training graphs.
    fn ce_logits_grad_buffer(&self, logits: NodeId, labels: NodeId) -> Option<BufferRef> {
        self.graph.nodes().iter().find_map(|node| {
            if matches!(node.op, Op::CrossEntropyLogitsGrad)
                && node.inputs.first().copied() == Some(logits)
                && node.inputs.get(1).copied() == Some(labels)
            {
                Some(self.get_buffer(node.id))
            } else {
                None
            }
        })
    }

    /// Choose between FlashAttention (BQ>1) and MultiHeadAttn (BQ=1) for
    /// a forward attention dispatch. Returns (shader_entry, workgroups_x).
    fn attention_dispatch(
        &self,
        q_seq: u32,
        head_dim: u32,
        num_heads: u32,
    ) -> (ShaderEntry, [u32; 3]) {
        // Pick the coop-matrix flash forward when the GPU has the
        // 16x16 f16 cooperative_matrix path (NVIDIA, RDNA3, Xe-HPG)
        // and the shape is compatible. ~3.2x faster per dispatch than
        // the scalar kernel on Blackwell. The env var
        // `MEGANEURA_FLASH_FWD_COOP=0` opts back to scalar (regression
        // escape hatch).
        let coop_disabled = !self.options.flash_forward_coop;
        if !coop_disabled
            && self.coop_caps.supports_16x16_f16()
            && head_dim >= 16
            && head_dim.is_multiple_of(16)
            && q_seq >= 16
        {
            return (
                ShaderEntry::FlashAttentionCoop,
                [q_seq.div_ceil(16), num_heads, 1],
            );
        }
        // EPT (elements per thread) must match forward codegen.
        let ept = head_dim.min(self.options.knobs.flash_ept_cap);
        let tpq = head_dim / ept; // threads per query
        let bq = (256 / tpq).max(1);
        if bq >= 2 && q_seq >= bq {
            (
                ShaderEntry::FlashAttention,
                [q_seq.div_ceil(bq), num_heads, 1],
            )
        } else {
            (ShaderEntry::MultiHeadAttn, [q_seq, num_heads, 1])
        }
    }

    /// Backward attention dispatch — same EPT/TPQ/BQ pattern as the
    /// forward kernel.
    fn attention_dispatch_bwd(
        q_seq: u32,
        head_dim: u32,
        num_heads: u32,
        ept_cap: u32,
    ) -> (ShaderEntry, [u32; 3]) {
        let ept = head_dim.min(ept_cap);
        let tpq = head_dim / ept;
        let bq = (256 / tpq).max(1);
        if bq >= 2 && q_seq >= bq {
            (
                ShaderEntry::FlashAttention,
                [q_seq.div_ceil(bq), num_heads, 1],
            )
        } else {
            (ShaderEntry::MultiHeadAttn, [q_seq, num_heads, 1])
        }
    }

    fn get_buffer(&self, node: NodeId) -> BufferRef {
        self.node_buffers[&node]
    }

    fn is_f32_unit_constant(&self, node: NodeId) -> bool {
        let node = self.graph.node(node);
        node.ty.dtype == DType::F32
            && matches!(
                node.op,
                Op::Constant { ref data }
                    if !data.is_empty()
                        && data.iter().all(|value| value.to_bits() == 1.0_f32.to_bits())
            )
    }

    fn emit_sum_inner(&mut self, input: BufferRef, output: BufferRef, rows: u32, inner: u32) {
        const WORKGROUP_SIZE: u32 = 256;
        let rows_per_workgroup = if inner <= 32 { WORKGROUP_SIZE } else { 1 };
        self.emit_sum_inner_with_rows_per_workgroup(input, output, rows, inner, rows_per_workgroup);
    }

    fn emit_sum_inner_with_rows_per_workgroup(
        &mut self,
        input: BufferRef,
        output: BufferRef,
        rows: u32,
        inner: u32,
        rows_per_workgroup: u32,
    ) {
        use crate::schedule::ReduceOp;

        const WORKGROUP_SIZE: u32 = 256;
        let kernel = ReductionKernel {
            op: ReduceOp::Sum,
            prologue: PointwiseDAG {
                n_inputs: 1,
                ops: vec![Pw::LoadInput(0)],
                output: 0,
            },
            extra_prologues: vec![],
            epilogue: None,
            n_per_elem: 1,
            n_per_row: 0,
            workgroup_size: WORKGROUP_SIZE,
            rows_per_workgroup,
            gather_elem: Vec::new(),
            input_row_repeats: Vec::new(),
        };
        self.plan.dispatches.push(Dispatch {
            shader: ShaderEntry::Relu, // sentinel; routing is via `reduction`
            workgroups: [rows.div_ceil(rows_per_workgroup), 1, 1],
            input_buffers: vec![input],
            output_buffer: output,
            extra_outputs: vec![],
            params: vec![rows, inner, 1.0_f32.to_bits(), 0],
            use_coop: false,
            use_small_tiles: false,
            reduction: Some(kernel),
            ..Default::default()
        });
    }

    fn emit_broadcast_inner(&mut self, input: BufferRef, output: BufferRef, rows: u32, inner: u32) {
        let total = rows
            .checked_mul(inner)
            .expect("inner broadcast element count exceeds u32");
        self.plan.dispatches.push(Dispatch {
            shader: ShaderEntry::GlobalAvgPoolGrad,
            workgroups: [total.div_ceil(256), 1, 1],
            input_buffers: vec![input],
            output_buffer: output,
            extra_outputs: vec![],
            params: vec![total, inner, 1, 0],
            use_coop: false,
            use_small_tiles: false,
            ..Default::default()
        });
    }

    fn compile(&mut self) {
        // First pass: allocate buffers for all nodes
        for node in self.graph.nodes() {
            // Identity and StopGradient are zero-cost: alias the input
            // buffer. (Identity may also reshape; StopGradient is forward-
            // identity with backward zero, handled in autodiff.)
            if matches!(node.op, Op::Identity | Op::StopGradient) && !node.inputs.is_empty() {
                if let Some(&input_buf) = self.node_buffers.get(&node.inputs[0]) {
                    self.node_buffers.insert(node.id, input_buf);
                    continue;
                }
            }
            // Loss ops output scalar [1] but the shader writes per-batch
            // or per-workgroup partial losses. Allocate enough space for
            // the shader; read_loss() sums all elements on the CPU side.
            let size = match node.op {
                Op::CrossEntropyLoss => {
                    let batch = self.graph.node(node.inputs[0]).ty.shape[0];
                    batch * 4
                }
                Op::BceLoss => {
                    let n: usize = self.graph.node(node.inputs[0]).ty.shape.iter().product();
                    n.div_ceil(256) * 4
                }
                _ => node.ty.size_bytes(),
            };
            let buf = self.alloc_buffer(size);
            self.node_buffers.insert(node.id, buf);

            match node.op {
                Op::Parameter { ref name } => {
                    self.plan.param_buffers.push((name.clone(), buf));
                    let wf = WeightFormat::from_dtype(node.ty.dtype);
                    if wf.uses_reduced_storage() {
                        let shape = &node.ty.shape;
                        let (rows, cols) = if shape.len() >= 2 {
                            (shape[0], shape[1])
                        } else {
                            (shape[0], 1)
                        };
                        self.plan.weight_buffers.insert(buf, (wf, rows, cols));
                    }
                }
                Op::Input { ref name } => {
                    self.plan.input_buffers.push((name.clone(), buf));
                }
                Op::Constant { ref data } => {
                    self.plan.constant_buffers.push((buf, data.clone()));
                }
                Op::MultiHeadAttn { num_heads, .. }
                | Op::CausalAttention { num_heads, .. }
                | Op::CausalAttentionRoPE { num_heads, .. }
                | Op::SlidingWindowAttention { num_heads, .. }
                | Op::FullAttention { num_heads, .. }
                | Op::CrossAttention { num_heads, .. } => {
                    let q_seq = node.ty.shape[0];
                    // LSE buffer: [0..q_seq*num_heads*2): LSE data (max_score, log_sum_exp per pos×head)
                    let lse_part = q_seq * num_heads as usize * 2;
                    let lse_buf = self.alloc_buffer(lse_part * 4);
                    self.plan.lse_buffers.push((node.id, lse_buf));
                }
                _ => {}
            }
        }

        // Second pass: emit dispatches in topological order.
        // The optimizer may create new nodes at high IDs that are referenced
        // by existing nodes at lower IDs (e.g. SwiGLU concat fusion creates
        // a new MatMul at the end, referenced by the original SwiGLU node).
        // Processing in ID order would dispatch consumers before producers.
        let topo = topological_order(self.graph);
        for &node_id in &topo {
            let first_new = self.plan.dispatches.len();
            self.compile_node(&self.graph.nodes()[node_id as usize]);
            // Stamp provenance on every dispatch this node emitted.
            for d in &mut self.plan.dispatches[first_new..] {
                d.origin.push(node_id);
            }
        }

        // Record the node → buffer map and node names for debug readback.
        let mut node_buffers: Vec<(NodeId, BufferRef)> =
            self.node_buffers.iter().map(|(&n, &b)| (n, b)).collect();
        node_buffers.sort_unstable_by_key(|&(n, _)| n);
        self.plan.node_buffers = node_buffers;
        self.plan.node_names = self
            .graph
            .nodes()
            .iter()
            .filter_map(|n| n.name.clone().map(|name| (n.id, name)))
            .collect();

        // Generate labels for profiling
        for d in &mut self.plan.dispatches {
            d.label = match d.shader {
                ShaderEntry::MatMul
                | ShaderEntry::FusedMatMulAdd
                | ShaderEntry::MatMulGemv
                | ShaderEntry::MatMulGemvAdd => {
                    format!(
                        "{:?}[{}x{}x{}]",
                        d.shader, d.params[0], d.params[2], d.params[1]
                    )
                }
                ShaderEntry::MatMulAT
                | ShaderEntry::MatMulBT
                | ShaderEntry::MatMulGemvBT
                | ShaderEntry::FusedMatMulATAdd
                | ShaderEntry::FusedMatMulBTAdd => {
                    format!(
                        "{:?}[{}x{}x{}]",
                        d.shader, d.params[0], d.params[1], d.params[2]
                    )
                }
                ShaderEntry::MultiHeadAttn
                | ShaderEntry::MultiHeadAttnGradQ
                | ShaderEntry::MultiHeadAttnGradKV => {
                    let nh = d.params[2] >> 16;
                    let nkv = d.params[2] & 0xFFFF;
                    format!(
                        "{:?}[q={},kv={},h={}/{}]",
                        d.shader, d.params[0], d.params[1], nh, nkv
                    )
                }
                ShaderEntry::RmsNorm
                | ShaderEntry::RmsNormGradW
                | ShaderEntry::RmsNormGradWRowPar
                | ShaderEntry::RmsNormGradX
                | ShaderEntry::LayerNormGradWB
                | ShaderEntry::LayerNormGradX => {
                    format!("{:?}[{}x{}]", d.shader, d.params[0], d.params[1])
                }
                _ => {
                    if d.params[0] > 0 {
                        format!("{:?}[{}]", d.shader, d.params[0])
                    } else {
                        format!("{:?}", d.shader)
                    }
                }
            };
            // Prefix the graph-level name when the originating node has one,
            // so profiler rows and dumps read "blk3.mlp.gate: MatMul[...]"
            // instead of a dozen identical "MatMul[...]" entries.
            if let Some(name) = d.origin.iter().find_map(|&n| self.graph.node_name(n)) {
                d.label = format!("{name}: {}", d.label);
            }
        }

        // Outputs layout (from autodiff): [user_outputs..., param_grads...]
        // Where `param_grads` is the last `num_param_grad_outputs()` entries
        // (one per Parameter node, exactly aligned with param_buffers order).
        // For inference (no autodiff), num_param_grad_outputs() == 0 and every
        // output is user-facing.
        let outputs = self.graph.outputs();
        let num_grads = self.graph.num_param_grad_outputs();
        let num_user = outputs.len() - num_grads;

        // Collect user-facing output buffers (accessible via read_output_by_index).
        for &out_id in &outputs[..num_user] {
            self.plan.output_buffers.push(self.get_buffer(out_id));
        }
        if let Some(&loss_id) = outputs.first() {
            self.plan.loss_buffer = Some(self.get_buffer(loss_id));
        }

        // Build param→grad pairs from the trailing grad outputs.
        if num_grads > 0 {
            let param_nodes: Vec<&Node> = self
                .graph
                .nodes()
                .iter()
                .filter(|node| matches!(node.op, Op::Parameter { .. }))
                .collect();
            assert_eq!(
                param_nodes.len(),
                num_grads,
                "autodiff must emit one grad output per Parameter",
            );
            for i in 0..num_grads {
                let grad_node = self.graph.node(outputs[num_user + i]);
                if grad_node.ty.num_elements() != param_nodes[i].ty.num_elements() {
                    // Autodiff uses a scalar zero as the positional placeholder
                    // for a parameter that receives no gradient (for example,
                    // one behind StopGradient). Do not expose that sentinel as
                    // a real gradient: optimizers iterate over the parameter's
                    // full length and would otherwise read past the scalar.
                    assert!(
                        matches!(grad_node.op, Op::Constant { ref data } if data == &[0.0]),
                        "parameter gradient shape mismatch without a zero placeholder",
                    );
                    continue;
                }
                let param_buf = self.plan.param_buffers[i].1;
                let grad_buf = self.get_buffer(outputs[num_user + i]);
                self.plan.param_grad_pairs.push((param_buf, grad_buf));
            }
        }
    }

    fn compile_node(&mut self, node: &Node) {
        let out_buf = self.get_buffer(node.id);
        let dispatch_start = self.plan.dispatches.len();

        match node.op {
            // Leaf nodes and dead nodes: no dispatch needed
            Op::Input { .. }
            | Op::Parameter { .. }
            | Op::Constant { .. }
            | Op::Nop
            | Op::Identity
            | Op::StopGradient
            | Op::CrossEntropyLogitsGrad => {}

            Op::Materialize => {
                let input = self.get_buffer(node.inputs[0]);
                let len = node.ty.num_elements() as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::Relu,
                    workgroups: [len.div_ceil(256), 1, 1],
                    input_buffers: vec![input],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![len, 0, 0, 0],
                    pointwise: Some(PointwiseDAG {
                        n_inputs: 1,
                        ops: vec![Pw::LoadInput(0)],
                        output: 0,
                    }),
                    fusion_barrier: true,
                    ..Default::default()
                });
            }

            Op::MatMul => {
                let a = self.get_buffer(node.inputs[0]);
                let b = self.get_buffer(node.inputs[1]);
                let a_shape = &self.graph.node(node.inputs[0]).ty.shape;
                let b_shape = &self.graph.node(node.inputs[1]).ty.shape;
                let wf = WeightFormat::from_dtype(self.graph.node(node.inputs[1]).ty.dtype);
                let m = a_shape[0] as u32;
                let k = a_shape[1] as u32;
                let n = b_shape[1] as u32;
                if n == 1 && k <= 32 && self.is_f32_unit_constant(node.inputs[1]) {
                    // A narrow multiplication by an all-ones column is a
                    // scalar-order row sum. Keep MatMul in the graph so its
                    // autodiff topology and accumulation order are unchanged;
                    // specialize only the physical forward dispatch.
                    self.emit_sum_inner(a, out_buf, m, k);
                } else if m == 1 && n.is_multiple_of(4) {
                    // K-split GEMV: one WG per 4 output columns (vec4),
                    // 32 threads cooperatively K-split with a shared-
                    // memory tree reduction. Many more WGs than N/128,
                    // giving occupancy to hide DRAM latency at M=1.
                    self.plan.dispatches.push(Dispatch {
                        shader: ShaderEntry::MatMulGemv,
                        workgroups: [n / 4, 1, 1],
                        input_buffers: vec![a, b],
                        output_buffer: out_buf,
                        extra_outputs: vec![],
                        params: vec![m, k, n, 0],
                        use_coop: false,
                        use_small_tiles: false,
                        weight_format: wf,
                        ..Default::default()
                    });
                } else {
                    self.plan.dispatches.push(Dispatch {
                        shader: ShaderEntry::MatMul,
                        workgroups: matmul_workgroups(m, n, 64),
                        input_buffers: vec![a, b],
                        output_buffer: out_buf,
                        extra_outputs: vec![],
                        params: vec![m, k, n, 0],
                        use_coop: false,
                        use_small_tiles: false,
                        weight_format: wf,
                        ..Default::default()
                    });
                }
            }

            Op::MatMulAT => {
                // C = A^T @ B  (A is [K, M], B is [K, N], C is [M, N])
                let a = self.get_buffer(node.inputs[0]);
                let b = self.get_buffer(node.inputs[1]);
                let a_shape = &self.graph.node(node.inputs[0]).ty.shape;
                let b_shape = &self.graph.node(node.inputs[1]).ty.shape;
                let wf = WeightFormat::from_dtype(self.graph.node(node.inputs[1]).ty.dtype);
                let k = a_shape[0] as u32; // A is [K, M]
                let m = a_shape[1] as u32;
                let n = b_shape[1] as u32; // B is [K, N]
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::MatMulAT,
                    workgroups: matmul_workgroups(m, n, 64),
                    input_buffers: vec![a, b],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![m, n, k, 0],
                    use_coop: false,
                    use_small_tiles: false,
                    weight_format: wf,
                    ..Default::default()
                });
            }

            Op::MatMulBT => {
                // C = A @ B^T  (A is [M, K], B is [N, K], C is [M, N])
                let a = self.get_buffer(node.inputs[0]);
                let b = self.get_buffer(node.inputs[1]);
                let a_shape = &self.graph.node(node.inputs[0]).ty.shape;
                let b_shape = &self.graph.node(node.inputs[1]).ty.shape;
                let wf = WeightFormat::from_dtype(self.graph.node(node.inputs[1]).ty.dtype);
                let m = a_shape[0] as u32; // A is [M, K]
                let k = a_shape[1] as u32;
                let n = b_shape[0] as u32; // B is [N, K]
                if k == 1 && self.is_f32_unit_constant(node.inputs[1]) {
                    // MatMul's input gradient against a unit column is an
                    // exact row broadcast. Preserve the MatMulBT graph node
                    // and its dependency order while avoiding tiled GEMM.
                    self.emit_broadcast_inner(a, out_buf, m, n);
                } else if m == 1 && k.is_multiple_of(4) {
                    self.plan.dispatches.push(Dispatch {
                        shader: ShaderEntry::MatMulGemvBT,
                        workgroups: [n, 1, 1],
                        input_buffers: vec![a, b],
                        output_buffer: out_buf,
                        extra_outputs: vec![],
                        params: vec![m, n, k, 0],
                        use_coop: false,
                        use_small_tiles: false,
                        weight_format: wf,
                        ..Default::default()
                    });
                } else {
                    self.plan.dispatches.push(Dispatch {
                        shader: ShaderEntry::MatMulBT,
                        workgroups: matmul_workgroups(m, n, 64),
                        input_buffers: vec![a, b],
                        output_buffer: out_buf,
                        extra_outputs: vec![],
                        params: vec![m, n, k, 0],
                        use_coop: false,
                        use_small_tiles: false,
                        weight_format: wf,
                        ..Default::default()
                    });
                }
            }

            Op::FusedMatMulAdd => {
                // C = A × B + D (inputs: [a, b, d])
                let a = self.get_buffer(node.inputs[0]);
                let b = self.get_buffer(node.inputs[1]);
                let d = self.get_buffer(node.inputs[2]);
                let a_shape = &self.graph.node(node.inputs[0]).ty.shape;
                let b_shape = &self.graph.node(node.inputs[1]).ty.shape;
                let wf = WeightFormat::from_dtype(self.graph.node(node.inputs[1]).ty.dtype);
                let m = a_shape[0] as u32;
                let k = a_shape[1] as u32;
                let n = b_shape[1] as u32;
                if m == 1 && n.is_multiple_of(4) {
                    self.plan.dispatches.push(Dispatch {
                        shader: ShaderEntry::MatMulGemvAdd,
                        workgroups: [n / 4, 1, 1],
                        input_buffers: vec![a, b, d],
                        output_buffer: out_buf,
                        extra_outputs: vec![],
                        params: vec![m, k, n, 0],
                        use_coop: false,
                        use_small_tiles: false,
                        weight_format: wf,
                        ..Default::default()
                    });
                } else {
                    self.plan.dispatches.push(Dispatch {
                        shader: ShaderEntry::FusedMatMulAdd,
                        workgroups: matmul_workgroups(m, n, 64),
                        input_buffers: vec![a, b, d],
                        output_buffer: out_buf,
                        extra_outputs: vec![],
                        params: vec![m, k, n, 0],
                        use_coop: false,
                        use_small_tiles: false,
                        weight_format: wf,
                        ..Default::default()
                    });
                }
            }

            Op::FusedMatMulATAdd => {
                // C = A^T × B + D (inputs: [a, b, d])
                let a = self.get_buffer(node.inputs[0]);
                let b = self.get_buffer(node.inputs[1]);
                let d = self.get_buffer(node.inputs[2]);
                let a_shape = &self.graph.node(node.inputs[0]).ty.shape;
                let b_shape = &self.graph.node(node.inputs[1]).ty.shape;
                let wf = WeightFormat::from_dtype(self.graph.node(node.inputs[1]).ty.dtype);
                let k = a_shape[0] as u32;
                let m = a_shape[1] as u32;
                let n = b_shape[1] as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::FusedMatMulATAdd,
                    workgroups: matmul_workgroups(m, n, 64),
                    input_buffers: vec![a, b, d],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![m, n, k, 0],
                    use_coop: false,
                    use_small_tiles: false,
                    weight_format: wf,
                    ..Default::default()
                });
            }

            Op::FusedMatMulBTAdd => {
                // C = A × B^T + D (inputs: [a, b, d])
                let a = self.get_buffer(node.inputs[0]);
                let b = self.get_buffer(node.inputs[1]);
                let d = self.get_buffer(node.inputs[2]);
                let a_shape = &self.graph.node(node.inputs[0]).ty.shape;
                let b_shape = &self.graph.node(node.inputs[1]).ty.shape;
                let wf = WeightFormat::from_dtype(self.graph.node(node.inputs[1]).ty.dtype);
                let m = a_shape[0] as u32;
                let k = a_shape[1] as u32;
                let n = b_shape[0] as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::FusedMatMulBTAdd,
                    workgroups: matmul_workgroups(m, n, 64),
                    input_buffers: vec![a, b, d],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![m, n, k, 0],
                    use_coop: false,
                    use_small_tiles: false,
                    weight_format: wf,
                    ..Default::default()
                });
            }

            Op::Add => {
                self.emit_binary(ShaderEntry::Add, node, out_buf);
            }
            Op::Mul => {
                self.emit_binary(ShaderEntry::Mul, node, out_buf);
            }
            Op::Greater => {
                self.emit_binary(ShaderEntry::Greater, node, out_buf);
            }

            Op::BiasAdd => {
                let a = self.get_buffer(node.inputs[0]);
                let b = self.get_buffer(node.inputs[1]);
                let len = node.ty.num_elements() as u32;
                let bias_len = self.graph.node(node.inputs[1]).ty.num_elements() as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::BiasAdd,
                    workgroups: [len.div_ceil(256), 1, 1],
                    input_buffers: vec![a, b],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![len, bias_len, 0, 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::Relu => {
                self.emit_unary(ShaderEntry::Relu, node, out_buf);
            }
            Op::Sigmoid => {
                self.emit_unary(ShaderEntry::Sigmoid, node, out_buf);
            }
            Op::Tanh => {
                self.emit_unary(ShaderEntry::Tanh, node, out_buf);
            }
            Op::Neg => {
                self.emit_unary(ShaderEntry::Neg, node, out_buf);
            }
            Op::Abs => {
                self.emit_unary(ShaderEntry::Abs, node, out_buf);
            }
            Op::Log => {
                self.emit_unary(ShaderEntry::Log, node, out_buf);
            }
            Op::Recip => {
                self.emit_unary(ShaderEntry::Recip, node, out_buf);
            }
            Op::Exp => {
                self.emit_generated_unary(Pw::Exp(0), node, out_buf);
            }
            Op::Softplus { beta } => {
                let input = self.get_buffer(node.inputs[0]);
                let len = node.ty.num_elements() as u32;
                let pointwise = PointwiseDAG {
                    n_inputs: 1,
                    ops: vec![
                        Pw::LoadInput(0),
                        Pw::const_f32(beta),
                        Pw::Mul(0, 1),
                        Pw::Relu(2),
                        Pw::Abs(2),
                        Pw::Sigmoid(4),
                        Pw::Log(5),
                        Pw::Neg(6),
                        Pw::Add(3, 7),
                        Pw::const_f32(beta.recip()),
                        Pw::Mul(8, 9),
                    ],
                    output: 10,
                };
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::Relu,
                    workgroups: [len.div_ceil(256), 1, 1],
                    input_buffers: vec![input],
                    output_buffer: out_buf,
                    params: vec![len, 0, 0, 0],
                    pointwise: Some(pointwise),
                    ..Default::default()
                });
            }
            Op::SoftplusGrad { beta } => {
                let grad_output = self.get_buffer(node.inputs[0]);
                let input = self.get_buffer(node.inputs[1]);
                let len = node.ty.num_elements() as u32;
                let pointwise = PointwiseDAG {
                    n_inputs: 2,
                    ops: vec![
                        Pw::LoadInput(0),            // 0: upstream gradient
                        Pw::LoadInput(1),            // 1: x
                        Pw::const_f32(beta),         // 2: beta
                        Pw::Mul(1, 2),               // 3: beta * x
                        Pw::const_f32(beta.recip()), // 4: 1 / beta
                        Pw::Mul(0, 4),               // 5: gradient after final scale
                        Pw::Abs(3),                  // 6: abs(beta * x)
                        Pw::Sigmoid(6),              // 7: stable probability
                        Pw::Neg(5),                  // 8: gradient through negation
                        Pw::Recip(7),                // 9: gradient through logarithm
                        Pw::Mul(8, 9),               // 10: log input gradient
                        Pw::const_f32(1.0),          // 11: one
                        Pw::Neg(7),                  // 12: -probability
                        Pw::Add(11, 12),             // 13: 1 - probability
                        Pw::Mul(7, 13),              // 14: sigmoid derivative
                        Pw::Mul(10, 14),             // 15: abs output gradient
                        Pw::const_f32(0.0),          // 16: zero
                        Pw::Greater(3, 16),          // 17: positive mask for abs
                        Pw::const_f32(2.0),          // 18: two
                        Pw::Mul(17, 18),             // 19: doubled mask
                        Pw::const_f32(1.0),          // 20: one
                        Pw::Neg(20),                 // 21: negative one
                        Pw::Add(19, 21),             // 22: sign(beta * x)
                        Pw::Mul(15, 22),             // 23: abs branch gradient
                        Pw::const_f32(0.0),          // 24: zero
                        Pw::Greater(3, 24),          // 25: ReLU mask
                        Pw::Mul(5, 25),              // 26: ReLU branch gradient
                        Pw::Add(23, 26),             // 27: accumulated beta*x gradient
                        Pw::Mul(27, 2),              // 28: input gradient
                    ],
                    output: 28,
                };
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::Add,
                    workgroups: [len.div_ceil(256), 1, 1],
                    input_buffers: vec![grad_output, input],
                    output_buffer: out_buf,
                    params: vec![len, 0, 0, 0],
                    pointwise: Some(pointwise),
                    ..Default::default()
                });
            }

            Op::SumAll => {
                let input = self.get_buffer(node.inputs[0]);
                let len = self.graph.node(node.inputs[0]).ty.num_elements() as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::SumAll,
                    workgroups: [1, 1, 1],
                    input_buffers: vec![input],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![len, 0, 0, 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::MeanAll => {
                let input = self.get_buffer(node.inputs[0]);
                let len = self.graph.node(node.inputs[0]).ty.num_elements() as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::MeanAll,
                    workgroups: [1, 1, 1],
                    input_buffers: vec![input],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![len, 0, 0, 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::SumRows => {
                // [M, N] → [N]: one workgroup per 32-column tile; eight
                // row lanes cooperate on each column.
                let input = self.get_buffer(node.inputs[0]);
                let in_shape = &self.graph.node(node.inputs[0]).ty.shape;
                let m = in_shape[0] as u32;
                let n = in_shape[1] as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::SumRows,
                    workgroups: [n.div_ceil(32), 1, 1],
                    input_buffers: vec![input],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![m, n, 0, 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::ExclusiveCumsum { reverse } => {
                // One invocation handles one row. This is deliberately
                // linear in N; the dense-matmul spelling is quadratic.
                let input = self.get_buffer(node.inputs[0]);
                let in_shape = &self.graph.node(node.inputs[0]).ty.shape;
                let m = in_shape[0] as u32;
                let n = in_shape[1] as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::GlobalAvgPoolGrad,
                    workgroups: [m.div_ceil(256), 1, 1],
                    input_buffers: vec![input],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![m, n, 3, u32::from(reverse)],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::ShiftInner { offset } => {
                let input = self.get_buffer(node.inputs[0]);
                let in_shape = &self.graph.node(node.inputs[0]).ty.shape;
                let m = in_shape[0] as u32;
                let n = in_shape[1] as u32;
                let len = m
                    .checked_mul(n)
                    .expect("shift_inner element count exceeds u32");
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::GlobalAvgPoolGrad,
                    workgroups: [len.div_ceil(256), 1, 1],
                    input_buffers: vec![input],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![len, n, 2, offset as u32],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::SumInner => {
                // [M, N] → [M, 1]: per-row reduction over the inner axis.
                // Pack narrow rows into one workgroup so SH-width reductions
                // do not launch hundreds of idle lanes per output scalar.
                // Wide reductions keep the existing producer-fusion path.
                // A one-lane packed path retains scalar column-order
                // summation while still allowing pointwise and gather
                // producers to fold into the reduction prologue.
                let input = self.get_buffer(node.inputs[0]);
                let in_shape = &self.graph.node(node.inputs[0]).ty.shape;
                let m = in_shape[0] as u32;
                let n = in_shape[1] as u32;
                self.emit_sum_inner(input, out_buf, m, n);
            }

            Op::BroadcastInner { inner } => {
                let input = self.get_buffer(node.inputs[0]);
                let rows = node.ty.shape[0] as u32;
                self.emit_broadcast_inner(input, out_buf, rows, inner);
            }

            Op::NormalizeInnerSum { inner, floor } => {
                let input = self.get_buffer(node.inputs[0]);
                let rows = u32::try_from(node.ty.shape[0])
                    .expect("normalize_inner_sum row count exceeds u32");
                let floor = Pw::const_f32(floor);
                let kernel = ReductionKernel {
                    op: crate::schedule::ReduceOp::Sum,
                    prologue: PointwiseDAG {
                        n_inputs: 1,
                        ops: vec![Pw::LoadInput(0)],
                        output: 0,
                    },
                    extra_prologues: vec![],
                    epilogue: Some(ReductionEpilogue {
                        dag: PointwiseDAG {
                            n_inputs: 2,
                            ops: vec![
                                Pw::LoadInput(0),
                                Pw::LoadInput(1),
                                floor,
                                Pw::Sub(1, 2),
                                Pw::Relu(3),
                                Pw::Add(4, 2),
                                Pw::Div(0, 5),
                            ],
                            output: 6,
                        },
                        n_per_col_inputs: 0,
                    }),
                    n_per_elem: 1,
                    n_per_row: 0,
                    workgroup_size: 256,
                    rows_per_workgroup: 256,
                    gather_elem: vec![],
                    input_row_repeats: vec![],
                };
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::Relu,
                    workgroups: [rows.div_ceil(256), 1, 1],
                    input_buffers: vec![input],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![rows, inner, 1.0_f32.to_bits(), 0],
                    use_coop: false,
                    use_small_tiles: false,
                    reduction: Some(kernel),
                    ..Default::default()
                });
            }

            Op::NormalizeInnerSumGrad { inner, floor } => {
                let grad_output = self.get_buffer(node.inputs[0]);
                let input = self.get_buffer(node.inputs[1]);
                let rows = u32::try_from(node.ty.shape[0])
                    .expect("normalize_inner_sum gradient row count exceeds u32");
                let sum = self.alloc_buffer(rows as usize * std::mem::size_of::<f32>());
                self.emit_sum_inner_with_rows_per_workgroup(input, sum, rows, inner, 256);
                let floor = Pw::const_f32(floor);
                let kernel = ReductionKernel {
                    op: crate::schedule::ReduceOp::Sum,
                    prologue: PointwiseDAG {
                        n_inputs: 3,
                        ops: vec![
                            Pw::LoadInput(0),
                            Pw::LoadInput(1),
                            Pw::LoadInput(2),
                            floor.clone(),
                            Pw::Sub(2, 3),
                            Pw::Relu(4),
                            Pw::Add(5, 3),
                            Pw::Recip(6),
                            Pw::Mul(7, 7),
                            Pw::Neg(8),
                            Pw::Mul(0, 1),
                            Pw::Mul(10, 9),
                            Pw::Greater(2, 3),
                            Pw::Mul(11, 12),
                        ],
                        output: 13,
                    },
                    extra_prologues: vec![],
                    epilogue: Some(ReductionEpilogue {
                        dag: PointwiseDAG {
                            n_inputs: 4,
                            ops: vec![
                                Pw::LoadInput(0),
                                Pw::LoadInput(1),
                                Pw::LoadInput(2),
                                Pw::LoadInput(3),
                                floor,
                                Pw::Sub(2, 4),
                                Pw::Relu(5),
                                Pw::Add(6, 4),
                                Pw::Recip(7),
                                Pw::Mul(0, 8),
                                Pw::Add(9, 3),
                            ],
                            output: 10,
                        },
                        n_per_col_inputs: 0,
                    }),
                    n_per_elem: 2,
                    n_per_row: 1,
                    workgroup_size: 256,
                    rows_per_workgroup: 256,
                    gather_elem: vec![],
                    input_row_repeats: vec![],
                };
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::Relu,
                    workgroups: [rows.div_ceil(256), 1, 1],
                    input_buffers: vec![grad_output, input, sum],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![rows, inner, 1.0_f32.to_bits(), 0],
                    use_coop: false,
                    use_small_tiles: false,
                    reduction: Some(kernel),
                    ..Default::default()
                });
            }

            Op::PairwiseSquaredDistance { pairs } => {
                let left = self.get_buffer(node.inputs[0]);
                let right = self.get_buffer(node.inputs[1]);
                let inner = u32::try_from(self.graph.node(node.inputs[0]).ty.shape[1])
                    .expect("pairwise distance width exceeds u32");
                let total = u32::try_from(node.ty.num_elements())
                    .expect("pairwise distance output element count exceeds u32");
                let kernel = ReductionKernel {
                    op: crate::schedule::ReduceOp::Sum,
                    prologue: PointwiseDAG {
                        n_inputs: 2,
                        ops: vec![
                            Pw::LoadInput(0),
                            Pw::LoadInput(1),
                            Pw::Sub(0, 1),
                            Pw::Mul(2, 2),
                        ],
                        output: 3,
                    },
                    extra_prologues: vec![],
                    epilogue: None,
                    n_per_elem: 2,
                    n_per_row: 0,
                    workgroup_size: 256,
                    rows_per_workgroup: 256,
                    gather_elem: vec![],
                    input_row_repeats: vec![pairs, 1],
                };
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::Relu,
                    workgroups: [total.div_ceil(256), 1, 1],
                    input_buffers: vec![left, right],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![total, inner, 1.0_f32.to_bits(), 0],
                    use_coop: false,
                    use_small_tiles: false,
                    reduction: Some(kernel),
                    ..Default::default()
                });
            }

            Op::PairwiseGrad { kind, inner, pairs } => {
                let grad_output = self.get_buffer(node.inputs[0]);
                let first = self.get_buffer(node.inputs[1]);
                let second = self.get_buffer(node.inputs[2]);
                let total = u32::try_from(node.ty.num_elements())
                    .expect("pairwise gradient element count exceeds u32");
                let mode = match kind {
                    PairwiseGradKind::DistanceLeft => 0,
                    PairwiseGradKind::DistanceRight => 1,
                    PairwiseGradKind::RejectionDirections => 2,
                };
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::PairwiseGrad,
                    workgroups: [total.div_ceil(256), 1, 1],
                    input_buffers: vec![grad_output, first, second],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![total, inner, pairs, mode],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::PairwiseVectorRejection { pairs } => {
                let vectors = self.get_buffer(node.inputs[0]);
                let directions = self.get_buffer(node.inputs[1]);
                let inner = u32::try_from(self.graph.node(node.inputs[1]).ty.shape[1])
                    .expect("pairwise vector width exceeds u32");
                let vector_rows =
                    u32::try_from(node.ty.shape[0]).expect("pairwise vector row count exceeds u32");
                let kernel = ReductionKernel {
                    op: crate::schedule::ReduceOp::Sum,
                    prologue: PointwiseDAG {
                        n_inputs: 2,
                        ops: vec![Pw::LoadInput(0), Pw::LoadInput(1), Pw::Mul(0, 1)],
                        output: 2,
                    },
                    extra_prologues: vec![],
                    epilogue: Some(ReductionEpilogue {
                        dag: PointwiseDAG {
                            n_inputs: 3,
                            ops: vec![
                                Pw::LoadInput(0),
                                Pw::LoadInput(1),
                                Pw::LoadInput(2),
                                Pw::Mul(1, 2),
                                Pw::Sub(0, 3),
                            ],
                            output: 4,
                        },
                        n_per_col_inputs: 0,
                    }),
                    n_per_elem: 2,
                    n_per_row: 0,
                    workgroup_size: 256,
                    rows_per_workgroup: 256,
                    gather_elem: vec![],
                    input_row_repeats: vec![1, pairs],
                };
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::Relu,
                    workgroups: [vector_rows.div_ceil(256), 1, 1],
                    input_buffers: vec![vectors, directions],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![vector_rows, inner, 1.0_f32.to_bits(), 0],
                    use_coop: false,
                    use_small_tiles: false,
                    reduction: Some(kernel),
                    ..Default::default()
                });
            }

            Op::Softmax => {
                let input = self.get_buffer(node.inputs[0]);
                let shape = &self.graph.node(node.inputs[0]).ty.shape;
                let batch = shape[0] as u32;
                let features = shape[1] as u32;
                if self.options.use_schedule_reduction {
                    self.emit_softmax_schedule(input, out_buf, batch, features);
                } else {
                    self.plan.dispatches.push(Dispatch {
                        shader: ShaderEntry::Softmax,
                        workgroups: [batch.div_ceil(256), 1, 1],
                        input_buffers: vec![input],
                        output_buffer: out_buf,
                        extra_outputs: vec![],
                        params: vec![batch, features, 0, 0],
                        use_coop: false,
                        use_small_tiles: false,
                        ..Default::default()
                    });
                }
            }

            Op::LogSoftmax => {
                // Two-pass: Softmax → temp_buf, then elementwise Log →
                // out_buf. The previous implementation dispatched only
                // the Softmax shader and labeled the output as
                // log_softmax, producing softmax(x) values where the
                // backward (correctly) assumed log_softmax(x). Result:
                // forward and backward computed gradients for different
                // mathematical functions. Caught by `gradcheck.rs::
                // log_softmax_mid_chain` after the autodiff fix exposed
                // the disagreement.
                //
                // TODO: replace with a single dedicated log_softmax
                // shader (computes `x - log_sum_exp(x)` in one pass —
                // also more numerically stable than `log(softmax(x))`
                // for very negative log_softmax values). The two-pass
                // version is correct but allocates an extra batch*features
                // f32 buffer per LogSoftmax node.
                let input = self.get_buffer(node.inputs[0]);
                let shape = &self.graph.node(node.inputs[0]).ty.shape;
                let batch = shape[0] as u32;
                let features = shape[1] as u32;
                let n_bytes = (batch as usize) * (features as usize) * 4;
                let softmax_buf = self.alloc_buffer(n_bytes);
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::Softmax,
                    workgroups: [batch.div_ceil(256), 1, 1],
                    input_buffers: vec![input],
                    output_buffer: softmax_buf,
                    extra_outputs: vec![],
                    params: vec![batch, features, 0, 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
                let len = batch * features;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::Log,
                    workgroups: [len.div_ceil(256), 1, 1],
                    input_buffers: vec![softmax_buf],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![len, 0, 0, 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::CrossEntropyLoss => {
                let logits = self.get_buffer(node.inputs[0]);
                let labels = self.get_buffer(node.inputs[1]);
                let shape = &self.graph.node(node.inputs[0]).ty.shape;
                let batch = shape[0] as u32;
                let features = shape[1] as u32;
                // Training autodiff emits CrossEntropyLogitsGrad on the same
                // (logits, labels). Reuse that pre-allocated buffer so the
                // forward kernel's fused grad is the backward value — no
                // second softmax / ones-row matmul. Inference leaves
                // write_grad=0 and binds the loss buffer as a dummy.
                let grad_buf = self.ce_logits_grad_buffer(node.inputs[0], node.inputs[1]);
                let write_grad = u32::from(grad_buf.is_some());
                let grad_buf = grad_buf.unwrap_or(out_buf);
                // One workgroup per batch item (256 threads each).
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::CrossEntropyLoss,
                    workgroups: [batch, 1, 1],
                    input_buffers: vec![logits, labels],
                    output_buffer: grad_buf,
                    extra_outputs: vec![out_buf],
                    params: vec![batch, features, write_grad, 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::BceLoss => {
                let pred = self.get_buffer(node.inputs[0]);
                let labels = self.get_buffer(node.inputs[1]);
                let len = self.graph.node(node.inputs[0]).ty.num_elements() as u32;
                let num_wgs = len.div_ceil(256);
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::BceLoss,
                    workgroups: [num_wgs, 1, 1],
                    input_buffers: vec![pred, labels],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![len, 0, 0, 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::Transpose => {
                let input = self.get_buffer(node.inputs[0]);
                let shape = &self.graph.node(node.inputs[0]).ty.shape;
                let m = shape[0] as u32;
                let n = shape[1] as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::Transpose,
                    workgroups: [n.div_ceil(16), m.div_ceil(16), 1],
                    input_buffers: vec![input],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![m, n, 0, 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::Silu => {
                self.emit_unary(ShaderEntry::Silu, node, out_buf);
            }

            Op::SwiGLU => {
                self.emit_binary(ShaderEntry::SwiGLU, node, out_buf);
            }

            Op::SwiGLUConcat => {
                // input[M, 2*N] → output[M, N]
                let input = self.get_buffer(node.inputs[0]);
                let out_len = node.ty.num_elements() as u32;
                let half_n = node.ty.shape[1] as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::SwiGLUConcat,
                    workgroups: [out_len.div_ceil(256), 1, 1],
                    input_buffers: vec![input, input], // src_b unused in forward
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![out_len, half_n, 0, 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::SwiGLUConcatGrad => {
                // (grad_out[M,N], input[M,2*N]) → grad_input[M,2*N]
                let grad_out = self.get_buffer(node.inputs[0]);
                let input = self.get_buffer(node.inputs[1]);
                let grad_out_len = self.graph.node(node.inputs[0]).ty.num_elements() as u32;
                let half_n = self.graph.node(node.inputs[0]).ty.shape[1] as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::SwiGLUConcatGrad,
                    workgroups: [grad_out_len.div_ceil(256), 1, 1],
                    input_buffers: vec![input, grad_out],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![grad_out_len, half_n, 0, 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::RmsNorm { eps } => {
                let x = self.get_buffer(node.inputs[0]);
                let w = self.get_buffer(node.inputs[1]);
                let shape = &self.graph.node(node.inputs[0]).ty.shape;
                let rows = shape[0] as u32;
                let cols = shape[1] as u32;
                if self.options.use_schedule_reduction {
                    self.emit_rmsnorm_schedule(x, w, out_buf, rows, cols, eps);
                } else {
                    self.plan.dispatches.push(Dispatch {
                        shader: ShaderEntry::RmsNorm,
                        workgroups: [rows, 1, 1],
                        input_buffers: vec![x, w],
                        output_buffer: out_buf,
                        extra_outputs: vec![],
                        params: vec![rows, cols, eps.to_bits(), 0],
                        use_coop: false,
                        use_small_tiles: false,
                        ..Default::default()
                    });
                }
            }

            Op::Embedding => {
                let indices = self.get_buffer(node.inputs[0]);
                let table = self.get_buffer(node.inputs[1]);
                let idx_shape = &self.graph.node(node.inputs[0]).ty.shape;
                let tbl_node = self.graph.node(node.inputs[1]);
                let tbl_shape = &tbl_node.ty.shape;
                let seq = idx_shape[0] as u32;
                let hidden = tbl_shape[1] as u32;
                // The table's storage format picks the gather variant, so an
                // f16 table cannot silently be read by the f32 kernel.
                let wf = WeightFormat::from_dtype(tbl_node.ty.dtype);
                let workgroups = seq
                    .checked_mul(hidden)
                    .expect("embedding output exceeds u32 indexing")
                    .div_ceil(256);
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::Embedding,
                    workgroups: [workgroups, 1, 1],
                    input_buffers: vec![indices, table],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![seq, hidden, 0, 0],
                    use_coop: false,
                    use_small_tiles: false,
                    weight_format: wf,
                    ..Default::default()
                });
            }

            Op::ToF16 => {
                // Elementwise f32 → f16 cast. UnaryData layout (src, dst,
                // params); dst is an f16 buffer.
                let src = self.get_buffer(node.inputs[0]);
                let len = node.ty.num_elements() as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::ToF16,
                    workgroups: [len.div_ceil(256), 1, 1],
                    input_buffers: vec![src],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![len, 0, 0, 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::ScatterAdd { vocab_size } => {
                let indices = self.get_buffer(node.inputs[0]);
                let src = self.get_buffer(node.inputs[1]);
                let src_shape = &self.graph.node(node.inputs[1]).ty.shape;
                let seq_len = src_shape[0] as u32;
                let embed_dim = src_shape[1] as u32;
                let total = vocab_size as u32 * embed_dim;
                let params = vec![total, seq_len, embed_dim, 0];
                if u64::from(vocab_size as u32) * u64::from(seq_len) > 1_000_000 {
                    self.plan.dispatches.push(Dispatch {
                        shader: ShaderEntry::Relu,
                        workgroups: [total.div_ceil(256), 1, 1],
                        input_buffers: vec![src],
                        output_buffer: out_buf,
                        extra_outputs: vec![],
                        params: vec![total, 0, 0, 0],
                        pointwise: Some(PointwiseDAG {
                            n_inputs: 1,
                            ops: vec![Pw::const_f32(0.0)],
                            output: 0,
                        }),
                        use_coop: false,
                        use_small_tiles: false,
                        ..Default::default()
                    });
                    self.plan.dispatches.push(Dispatch {
                        shader: ShaderEntry::ScatterAddAtomic,
                        workgroups: [embed_dim.div_ceil(256), 1, 1],
                        // The output is also an input so scheduling inserts a
                        // global barrier after the zeroing dispatch.
                        input_buffers: vec![indices, src, out_buf],
                        output_buffer: out_buf,
                        extra_outputs: vec![],
                        params,
                        use_coop: false,
                        use_small_tiles: false,
                        ..Default::default()
                    });
                } else {
                    self.plan.dispatches.push(Dispatch {
                        shader: ShaderEntry::ScatterAdd,
                        workgroups: [total.div_ceil(256), 1, 1],
                        input_buffers: vec![indices, src],
                        output_buffer: out_buf,
                        extra_outputs: vec![],
                        params,
                        use_coop: false,
                        use_small_tiles: false,
                        ..Default::default()
                    });
                }
            }

            Op::RoPE {
                theta,
                pos_offset,
                head_dim,
            } => {
                let input = self.get_buffer(node.inputs[0]);
                let shape = &self.graph.node(node.inputs[0]).ty.shape;
                let seq = shape[0] as u32;
                let dim = shape[1] as u32;
                if node.inputs.len() == 2 {
                    // Dynamic offset: read pos_offset from input buffer
                    let offset_buf = self.get_buffer(node.inputs[1]);
                    self.plan.dispatches.push(Dispatch {
                        shader: ShaderEntry::RoPEDynamic,
                        workgroups: [(seq * dim / 2).div_ceil(256), 1, 1],
                        input_buffers: vec![input, offset_buf],
                        output_buffer: out_buf,
                        extra_outputs: vec![],
                        params: vec![seq, dim, theta.to_bits(), 0, head_dim, 0, 0, 0],
                        use_coop: false,
                        use_small_tiles: false,
                        ..Default::default()
                    });
                } else {
                    self.plan.dispatches.push(Dispatch {
                        shader: ShaderEntry::RoPE,
                        workgroups: [(seq * dim / 2).div_ceil(256), 1, 1],
                        input_buffers: vec![input],
                        output_buffer: out_buf,
                        extra_outputs: vec![],
                        params: vec![seq, dim, theta.to_bits(), pos_offset, head_dim, 0, 0, 0],
                        use_coop: false,
                        use_small_tiles: false,
                        ..Default::default()
                    });
                }
            }

            Op::CausalAttention {
                num_heads,
                num_kv_heads,
                head_dim,
            }
            | Op::CausalAttentionRoPE {
                num_heads,
                num_kv_heads,
                head_dim,
                ..
            } => {
                // Route through unified attention shader.
                // kv_seq=0 signals causal mask at runtime.
                let q = self.get_buffer(node.inputs[0]);
                let k = self.get_buffer(node.inputs[1]);
                let v = self.get_buffer(node.inputs[2]);
                let seq = self.graph.node(node.inputs[0]).ty.shape[0] as u32;
                let lse_buf = self.find_lse_buffer(node.id);
                let (shader, workgroups) = self.attention_dispatch(seq, head_dim, num_heads);
                self.plan.dispatches.push(Dispatch {
                    shader,
                    workgroups,
                    input_buffers: vec![q, k, v],
                    output_buffer: out_buf,
                    extra_outputs: vec![lse_buf],
                    params: vec![seq, 0, (num_heads << 16) | num_kv_heads, head_dim],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::SlidingWindowAttention {
                num_heads,
                num_kv_heads,
                head_dim,
                window_size,
            } => {
                // Route through unified attention shader.
                // kv_seq=0 signals causal, window_size>0 limits the window.
                let q = self.get_buffer(node.inputs[0]);
                let k = self.get_buffer(node.inputs[1]);
                let v = self.get_buffer(node.inputs[2]);
                let seq = self.graph.node(node.inputs[0]).ty.shape[0] as u32;
                let lse_buf = self.find_lse_buffer(node.id);
                let (shader, workgroups) = self.attention_dispatch(seq, head_dim, num_heads);
                self.plan.dispatches.push(Dispatch {
                    shader,
                    workgroups,
                    input_buffers: vec![q, k, v],
                    output_buffer: out_buf,
                    extra_outputs: vec![lse_buf],
                    params: vec![
                        seq,
                        0,
                        (num_heads << 16) | num_kv_heads,
                        head_dim,
                        window_size,
                    ],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::RoPEGrad {
                theta,
                pos_offset,
                head_dim,
            } => {
                let grad_out = self.get_buffer(node.inputs[0]);
                let shape = &self.graph.node(node.inputs[0]).ty.shape;
                let seq = shape[0] as u32;
                let dim = shape[1] as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::RoPEGrad,
                    workgroups: [(seq * dim / 2).div_ceil(256), 1, 1],
                    input_buffers: vec![grad_out],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![seq, dim, theta.to_bits(), pos_offset, head_dim, 0, 0, 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::GroupNorm {
                num_groups,
                eps,
                channels,
                spatial,
            } => {
                let x = self.get_buffer(node.inputs[0]);
                let weight = self.get_buffer(node.inputs[1]);
                let bias = self.get_buffer(node.inputs[2]);
                let total = node.ty.shape[0] as u32;
                let batch = total / (channels * spatial);
                // Two passes so the parallelism scales with the image: one
                // workgroup per slice of a group, rather than one per group.
                let chunks = group_norm_chunks(batch, channels, spatial, num_groups);
                if chunks == 1 {
                    // On small tensors the original kernel is already busy
                    // enough. Keep its single dispatch instead of paying for
                    // an intermediate buffer, another full tensor pass, and a
                    // global barrier merely to split the group once.
                    self.plan.dispatches.push(Dispatch {
                        shader: ShaderEntry::GroupNorm,
                        workgroups: [batch * num_groups, 1, 1],
                        input_buffers: vec![x, weight, bias],
                        output_buffer: out_buf,
                        extra_outputs: vec![],
                        params: vec![batch, channels, spatial, num_groups, eps.to_bits(), 0, 0, 0],
                        use_coop: false,
                        use_small_tiles: false,
                        ..Default::default()
                    });
                } else {
                    let group_size = (channels / num_groups) * spatial;
                    let slices = batch * num_groups * chunks;
                    let partials = self.alloc_buffer(slices as usize * 2 * 4);
                    let params = vec![
                        batch,
                        channels,
                        spatial,
                        num_groups,
                        eps.to_bits(),
                        chunks,
                        0,
                        0,
                    ];
                    self.plan.dispatches.push(Dispatch {
                        // Generated-reduction routing takes priority over the
                        // sentinel entry in pipeline selection and binding.
                        shader: ShaderEntry::GroupNorm,
                        workgroups: [slices, 1, 1],
                        input_buffers: vec![x],
                        output_buffer: partials,
                        extra_outputs: vec![],
                        params: vec![slices, group_size / chunks, 0, 0],
                        use_coop: false,
                        use_small_tiles: false,
                        reduction: Some(group_norm_stats_kernel()),
                        ..Default::default()
                    });
                    self.plan.dispatches.push(Dispatch {
                        shader: ShaderEntry::GroupNormApply,
                        workgroups: [slices, 1, 1],
                        input_buffers: vec![x, partials, weight, bias],
                        output_buffer: out_buf,
                        extra_outputs: vec![],
                        params,
                        use_coop: false,
                        use_small_tiles: false,
                        ..Default::default()
                    });
                }
            }

            Op::GroupNormSilu {
                num_groups,
                eps,
                channels,
                spatial,
            } => {
                let x = self.get_buffer(node.inputs[0]);
                let weight = self.get_buffer(node.inputs[1]);
                let bias = self.get_buffer(node.inputs[2]);
                let total = node.ty.shape[0] as u32;
                let batch = total / (channels * spatial);
                // Two passes so the parallelism scales with the image: one
                // workgroup per slice of a group, rather than one per group.
                let chunks = group_norm_chunks(batch, channels, spatial, num_groups);
                if chunks == 1 {
                    self.plan.dispatches.push(Dispatch {
                        shader: ShaderEntry::GroupNormSilu,
                        workgroups: [batch * num_groups, 1, 1],
                        input_buffers: vec![x, weight, bias],
                        output_buffer: out_buf,
                        extra_outputs: vec![],
                        params: vec![batch, channels, spatial, num_groups, eps.to_bits(), 0, 0, 0],
                        use_coop: false,
                        use_small_tiles: false,
                        ..Default::default()
                    });
                } else {
                    let group_size = (channels / num_groups) * spatial;
                    let slices = batch * num_groups * chunks;
                    let partials = self.alloc_buffer(slices as usize * 2 * 4);
                    let params = vec![
                        batch,
                        channels,
                        spatial,
                        num_groups,
                        eps.to_bits(),
                        chunks,
                        1,
                        0,
                    ];
                    self.plan.dispatches.push(Dispatch {
                        shader: ShaderEntry::GroupNorm,
                        workgroups: [slices, 1, 1],
                        input_buffers: vec![x],
                        output_buffer: partials,
                        extra_outputs: vec![],
                        params: vec![slices, group_size / chunks, 0, 0],
                        use_coop: false,
                        use_small_tiles: false,
                        reduction: Some(group_norm_stats_kernel()),
                        ..Default::default()
                    });
                    self.plan.dispatches.push(Dispatch {
                        shader: ShaderEntry::GroupNormApply,
                        workgroups: [slices, 1, 1],
                        input_buffers: vec![x, partials, weight, bias],
                        output_buffer: out_buf,
                        extra_outputs: vec![],
                        params,
                        use_coop: false,
                        use_small_tiles: false,
                        ..Default::default()
                    });
                }
            }

            Op::GroupNormGradInput {
                num_groups,
                eps,
                channels,
                spatial,
            } => {
                let grad_out = self.get_buffer(node.inputs[0]);
                let input = self.get_buffer(node.inputs[1]);
                let weight = self.get_buffer(node.inputs[2]);
                let total = node.ty.shape[0] as u32;
                let batch = total / (channels * spatial);
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::GroupNormGradInput,
                    workgroups: [batch * num_groups, 1, 1],
                    input_buffers: vec![grad_out, input, weight],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![batch, channels, spatial, num_groups, eps.to_bits(), 0, 0, 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::GroupNormGradWeightBias {
                num_groups,
                eps,
                channels,
                spatial,
            } => {
                let grad_out = self.get_buffer(node.inputs[0]);
                let input = self.get_buffer(node.inputs[1]);
                let go_total = self.graph.node(node.inputs[0]).ty.shape[0] as u32;
                let batch = go_total / (channels * spatial);
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::GroupNormGradWeightBias,
                    workgroups: [channels, 1, 1],
                    input_buffers: vec![grad_out, input],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![batch, channels, spatial, num_groups, eps.to_bits(), 0, 0, 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::Concat {
                channels_a,
                channels_b,
                spatial,
            } => {
                let a = self.get_buffer(node.inputs[0]);
                let b = self.get_buffer(node.inputs[1]);
                let total = node.ty.shape[0] as u32;
                let batch = total / ((channels_a + channels_b) * spatial);
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::Concat,
                    workgroups: [total.div_ceil(256), 1, 1],
                    input_buffers: vec![a, b],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![batch, channels_a, channels_b, spatial],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::SplitA {
                channels_a,
                channels_b,
                spatial,
            } => {
                let x = self.get_buffer(node.inputs[0]);
                let total = node.ty.shape[0] as u32;
                let batch = total / (channels_a * spatial);
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::SplitA,
                    workgroups: [total.div_ceil(256), 1, 1],
                    input_buffers: vec![x],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![batch, channels_a, channels_b, spatial],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::SplitB {
                channels_a,
                channels_b,
                spatial,
            } => {
                let x = self.get_buffer(node.inputs[0]);
                let total = node.ty.shape[0] as u32;
                let batch = total / (channels_b * spatial);
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::SplitB,
                    workgroups: [total.div_ceil(256), 1, 1],
                    input_buffers: vec![x],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![batch, channels_a, channels_b, spatial],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::Upsample2x {
                channels,
                in_h,
                in_w,
            } => {
                let x = self.get_buffer(node.inputs[0]);
                let total = node.ty.shape[0] as u32;
                let batch = total / (channels * in_h * 2 * in_w * 2);
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::Upsample2x,
                    workgroups: [total.div_ceil(256), 1, 1],
                    input_buffers: vec![x],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![batch, channels, in_h, in_w],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::Upsample2xGrad {
                channels,
                in_h,
                in_w,
            } => {
                let grad = self.get_buffer(node.inputs[0]);
                let total = node.ty.shape[0] as u32;
                let batch = total / (channels * in_h * in_w);
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::Upsample2xGrad,
                    workgroups: [total.div_ceil(256), 1, 1],
                    input_buffers: vec![grad],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![batch, channels, in_h, in_w],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::Conv2d {
                in_channels,
                in_h,
                in_w,
                out_channels,
                kernel_h,
                kernel_w,
                stride,
                padding_h,
                padding_w,
            } => {
                let input = self.get_buffer(node.inputs[0]);
                let kernel = self.get_buffer(node.inputs[1]);
                let in_shape = &self.graph.node(node.inputs[0]).ty.shape;
                let out_h = (in_h + 2 * padding_h - kernel_h) / stride + 1;
                let out_w = (in_w + 2 * padding_w - kernel_w) / stride + 1;
                let batch = in_shape[0] as u32 / (in_channels * in_h * in_w);

                // No 1×1 MatMul shortcut: the previous one viewed NCHW
                // input as [batch*H*W, Ci] row-major, which is a transposed
                // (pixel-scrambling) read of the actual [Ci, H*W] layout —
                // and its output was [HW, Co] while every consumer expects
                // [Co, HW]. Matching transposed views in the backward
                // shortcuts made training self-consistent, so gradchecks
                // passed while the network fought a fixed scrambler on
                // every residual projection. The general GEMM path is
                // CPU-parity-verified (tests/inference_parity_large.rs)
                // and handles kernel 1×1 as a degenerate im2col.
                {
                    // Use implicit GEMM: output = weight @ im2col(input)^T
                    // M=Co, N=oH*oW, K=Ci*kH*kW, batched in z dimension
                    // Use small (32×32) tiles when workgroup count per batch is low.
                    let spatial = out_h * out_w;
                    let wgs_64 = spatial.div_ceil(64) * out_channels.div_ceil(64);
                    let use_small = wgs_64 < 16;
                    let tile = if use_small { 32 } else { 64 };
                    self.plan.dispatches.push(Dispatch {
                        shader: if use_small {
                            ShaderEntry::Conv2dGemmSmall
                        } else {
                            ShaderEntry::Conv2dGemm
                        },
                        workgroups: [spatial.div_ceil(tile), out_channels.div_ceil(tile), batch],
                        input_buffers: vec![input, kernel],
                        output_buffer: out_buf,
                        extra_outputs: vec![],
                        params: vec![
                            batch,
                            in_channels,
                            in_h,
                            in_w,
                            out_channels,
                            kernel_h,
                            kernel_w,
                            stride,
                            padding_h,
                            out_h,
                            out_w,
                            padding_w,
                        ],
                        use_coop: false,
                        use_small_tiles: false,
                        ..Default::default()
                    });
                } // else (non-1x1 conv)
            }

            Op::MulPerChannel { channels, spatial } => {
                let src = self.get_buffer(node.inputs[0]);
                let gate = self.get_buffer(node.inputs[1]);
                let len = node.ty.shape[0] as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::MulPerChannel,
                    workgroups: [len.div_ceil(256), 1, 1],
                    input_buffers: vec![src, gate],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![len, spatial, channels, 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::AddPerChannel { channels, spatial } => {
                let src = self.get_buffer(node.inputs[0]);
                let bias = self.get_buffer(node.inputs[1]);
                let len = node.ty.shape[0] as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::AddPerChannel,
                    workgroups: [len.div_ceil(256), 1, 1],
                    input_buffers: vec![src, bias],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![len, spatial, channels, 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::Conv2dDw {
                channels,
                in_h,
                in_w,
                kernel_h,
                kernel_w,
                stride,
                padding_h,
                padding_w,
            } => {
                let input = self.get_buffer(node.inputs[0]);
                let kernel = self.get_buffer(node.inputs[1]);
                let in_shape = &self.graph.node(node.inputs[0]).ty.shape;
                let out_h = (in_h + 2 * padding_h - kernel_h) / stride + 1;
                let out_w = (in_w + 2 * padding_w - kernel_w) / stride + 1;
                let batch = in_shape[0] as u32 / (channels * in_h * in_w);
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::Conv2dDw,
                    workgroups: [out_w.div_ceil(16), out_h.div_ceil(16), batch * channels],
                    input_buffers: vec![input, kernel],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![
                        batch, channels, in_h, in_w, kernel_h, kernel_w, stride, padding_h, out_h,
                        out_w, padding_w,
                    ],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::WinogradConv2d {
                in_channels,
                in_h,
                in_w,
                out_channels,
                padding,
            } => {
                let out_h = in_h + 2 * padding - 2; // 3x3 stride 1
                let out_w = in_w + 2 * padding - 2;
                let batch_size = node.ty.shape[0] as u32 / (out_channels * out_h * out_w);
                let tiles_h = out_h.div_ceil(2);
                let tiles_w = out_w.div_ceil(2);
                let total_tiles = batch_size * tiles_h * tiles_w;

                // Temp buffers
                let input_xform_size = (16 * in_channels * total_tiles * 4) as usize;
                let mm_out_size = (16 * out_channels * total_tiles * 4) as usize;
                let input_xform_buf = self.alloc_buffer(input_xform_size);
                let mm_out_buf = self.alloc_buffer(mm_out_size);

                let input = self.get_buffer(node.inputs[0]);
                let weight_xform = self.get_buffer(node.inputs[1]); // Winograd-transformed weights [16*Co*Ci]
                // input[2] is the original weight [Co*Ci*9] (for backward, and for re-transform)
                let original_weight = self.get_buffer(node.inputs[2]);

                // Dispatch 0: Weight transform (re-transform every step for training)
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::WinogradWeightTransform,
                    workgroups: [out_channels * in_channels.div_ceil(256), 1, 1],
                    input_buffers: vec![original_weight],
                    output_buffer: weight_xform,
                    extra_outputs: vec![],
                    params: vec![out_channels, in_channels, 0, 0, 0, 0, 0, 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });

                // Dispatch 1: Input transform
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::WinogradInputTransform,
                    workgroups: [(total_tiles * in_channels).div_ceil(256), 1, 1],
                    input_buffers: vec![input],
                    output_buffer: input_xform_buf,
                    extra_outputs: vec![],
                    params: vec![
                        batch_size,
                        in_channels,
                        in_h,
                        in_w,
                        padding,
                        tiles_h,
                        tiles_w,
                        total_tiles,
                    ],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });

                // Dispatch 2: Batched matmul
                // weight_xform[16, Co, Ci] × input_xform[16, Ci, P] → mm_out[16, Co, P]
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::WinogradBatchedMatMul,
                    workgroups: [total_tiles.div_ceil(64), out_channels.div_ceil(64), 16],
                    input_buffers: vec![weight_xform, input_xform_buf],
                    output_buffer: mm_out_buf,
                    extra_outputs: vec![],
                    params: vec![out_channels, total_tiles, in_channels, 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });

                // Dispatch 3: Output transform
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::WinogradOutputTransform,
                    workgroups: [(total_tiles * out_channels).div_ceil(256), 1, 1],
                    input_buffers: vec![mm_out_buf],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![
                        batch_size,
                        out_channels,
                        out_h,
                        out_w,
                        tiles_h,
                        tiles_w,
                        total_tiles,
                        0,
                    ],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::Conv2dGradInput {
                in_channels,
                in_h,
                in_w,
                out_channels,
                kernel_h,
                kernel_w,
                stride,
                padding_h,
                padding_w,
            } => {
                let grad_out = self.get_buffer(node.inputs[0]);
                let kernel = self.get_buffer(node.inputs[1]);
                let out_h = (in_h + 2 * padding_h - kernel_h) / stride + 1;
                let out_w = (in_w + 2 * padding_w - kernel_w) / stride + 1;
                let out_size = node.ty.shape[0] as u32;
                let batch = out_size / (in_channels * in_h * in_w);

                // No 1×1 MatMul shortcut — see the Op::Conv2d arm: the
                // shortcut family used a transposed (NHWC-ish) view of
                // NCHW data. The GEMM path handles kernel 1×1 correctly.
                {
                    // Use implicit GEMM: grad_input = weight_T @ im2col(grad_out)^T
                    // M=Ci, N=H*W, K=Co*kH*kW, batched in z dimension.
                    {
                        let spatial = in_h * in_w;
                        let wgs_64 = spatial.div_ceil(64) * in_channels.div_ceil(64);
                        let use_small = wgs_64 < 16;
                        let tile = if use_small { 32 } else { 64 };
                        self.plan.dispatches.push(Dispatch {
                            shader: if use_small {
                                ShaderEntry::Conv2dGradInputGemmSmall
                            } else {
                                ShaderEntry::Conv2dGradInputGemm
                            },
                            workgroups: [spatial.div_ceil(tile), in_channels.div_ceil(tile), batch],
                            input_buffers: vec![grad_out, kernel],
                            output_buffer: out_buf,
                            extra_outputs: vec![],
                            params: vec![
                                batch,
                                in_channels,
                                in_h,
                                in_w,
                                out_channels,
                                kernel_h,
                                kernel_w,
                                stride,
                                padding_h,
                                out_h,
                                out_w,
                                padding_w,
                            ],
                            use_coop: false,
                            use_small_tiles: false,
                            ..Default::default()
                        });
                    }
                }
            }

            Op::Conv2dGradWeight {
                in_channels,
                in_h,
                in_w,
                out_channels,
                kernel_h,
                kernel_w,
                stride,
                padding_h,
                padding_w,
            } => {
                let grad_out = self.get_buffer(node.inputs[0]);
                let input = self.get_buffer(node.inputs[1]);
                let out_h = (in_h + 2 * padding_h - kernel_h) / stride + 1;
                let out_w = (in_w + 2 * padding_w - kernel_w) / stride + 1;
                let out_size = self.graph.node(node.inputs[0]).ty.shape[0] as u32;
                let batch = out_size / (out_channels * out_h * out_w);

                // No 1×1 MatMulAT shortcut — see the Op::Conv2d arm: the
                // shortcut family used a transposed (NHWC-ish) view of
                // NCHW data. The GEMM path handles kernel 1×1 correctly.
                {
                    // Use GEMM formulation: grad_weight[Co, Ci*kH*kW] = grad_out_flat[Co, N*oH*oW] @ im2col(input)[N*oH*oW, Ci*kH*kW]
                    let n_total = in_channels * kernel_h * kernel_w; // Ci*kH*kW
                    let m_total = out_channels; // Co
                    let wgs_64 = n_total.div_ceil(64) * m_total.div_ceil(64);
                    let use_small = wgs_64 < 16;
                    let tile = if use_small { 32 } else { 64 };
                    self.plan.dispatches.push(Dispatch {
                        shader: if use_small {
                            ShaderEntry::Conv2dGradWeightGemmSmall
                        } else {
                            ShaderEntry::Conv2dGradWeightGemm
                        },
                        workgroups: [n_total.div_ceil(tile), m_total.div_ceil(tile), 1],
                        input_buffers: vec![grad_out, input],
                        output_buffer: out_buf,
                        extra_outputs: vec![],
                        params: vec![
                            batch,
                            in_channels,
                            in_h,
                            in_w,
                            out_channels,
                            kernel_h,
                            kernel_w,
                            stride,
                            padding_h,
                            out_h,
                            out_w,
                            padding_w,
                        ],
                        use_coop: false,
                        use_small_tiles: false,
                        ..Default::default()
                    });
                }
            }

            Op::CacheWrite => {
                let new_kv = self.get_buffer(node.inputs[0]);
                let cache = self.get_buffer(node.inputs[1]);
                let kv_pos_input = self.get_buffer(node.inputs[2]);
                let dim = self.graph.node(node.inputs[0]).ty.shape[1] as u32;
                // Output aliases the cache buffer (in-place write)
                self.node_buffers.insert(node.id, cache);
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::CacheWrite,
                    workgroups: [dim.div_ceil(256), 1, 1],
                    input_buffers: vec![new_kv, cache, kv_pos_input],
                    output_buffer: cache,
                    extra_outputs: vec![],
                    params: vec![dim, 0, 0, 0], // kv_pos read from input buffer at runtime
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::CachedAttention {
                num_heads,
                num_kv_heads,
                head_dim,
            } => {
                let q = self.get_buffer(node.inputs[0]);
                let k_cache = self.get_buffer(node.inputs[1]);
                let v_cache = self.get_buffer(node.inputs[2]);
                let kv_pos_input = self.get_buffer(node.inputs[3]);
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::CachedAttention,
                    workgroups: [1, num_heads, 1],
                    input_buffers: vec![q, k_cache, v_cache, kv_pos_input],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![0, num_heads, num_kv_heads, head_dim], // kv_len read from input buffer
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::MaxPool2d {
                channels,
                in_h,
                in_w,
                kernel_h,
                kernel_w,
                stride,
                padding,
            } => {
                let input = self.get_buffer(node.inputs[0]);
                let in_shape = &self.graph.node(node.inputs[0]).ty.shape;
                let batch = in_shape[0] as u32 / (channels * in_h * in_w);
                let out_h = (in_h + 2 * padding - kernel_h) / stride + 1;
                let out_w = (in_w + 2 * padding - kernel_w) / stride + 1;
                let total = batch * channels * out_h * out_w;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::MaxPool2d,
                    workgroups: [total.div_ceil(256), 1, 1],
                    input_buffers: vec![input],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![
                        batch, channels, in_h, in_w, kernel_h, kernel_w, stride, padding, out_h,
                        out_w, 0, 0,
                    ],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::GlobalAvgPool { channels, spatial } => {
                let input = self.get_buffer(node.inputs[0]);
                let total_out = node.ty.num_elements() as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::GlobalAvgPool,
                    workgroups: [total_out.div_ceil(256), 1, 1],
                    input_buffers: vec![input],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![channels, spatial, total_out, 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::GlobalAvgPoolGrad {
                channels: _,
                spatial,
            } => {
                let grad_output = self.get_buffer(node.inputs[0]);
                let total = node.ty.num_elements() as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::GlobalAvgPoolGrad,
                    workgroups: [total.div_ceil(256), 1, 1],
                    input_buffers: vec![grad_output],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![total, spatial, 0, 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::Gelu => {
                self.emit_unary(ShaderEntry::Gelu, node, out_buf);
            }

            Op::LayerNorm { eps } => {
                let x = self.get_buffer(node.inputs[0]);
                let w = self.get_buffer(node.inputs[1]);
                let bias = self.get_buffer(node.inputs[2]);
                let shape = &self.graph.node(node.inputs[0]).ty.shape;
                let rows = shape[0] as u32;
                let cols = shape[1] as u32;
                if self.options.use_schedule_reduction {
                    self.emit_layernorm_schedule(x, w, bias, out_buf, rows, cols, eps);
                } else {
                    // One workgroup per row — threads inside cooperate on the
                    // mean/variance reduction. Matches the hand-written
                    // shader's workgroup-cooperative layout.
                    self.plan.dispatches.push(Dispatch {
                        shader: ShaderEntry::LayerNorm,
                        workgroups: [rows, 1, 1],
                        input_buffers: vec![x, w, bias],
                        output_buffer: out_buf,
                        extra_outputs: vec![],
                        params: vec![rows, cols, eps.to_bits(), 0],
                        use_coop: false,
                        use_small_tiles: false,
                        ..Default::default()
                    });
                }
            }

            Op::FullAttention {
                num_heads,
                num_kv_heads,
                head_dim,
            } => {
                // Route through unified attention shader.
                // kv_seq=q_seq for full (non-causal) attention.
                let q = self.get_buffer(node.inputs[0]);
                let k = self.get_buffer(node.inputs[1]);
                let v = self.get_buffer(node.inputs[2]);
                let seq = self.graph.node(node.inputs[0]).ty.shape[0] as u32;
                let lse_buf = self.find_lse_buffer(node.id);
                let (shader, workgroups) = self.attention_dispatch(seq, head_dim, num_heads);
                self.plan.dispatches.push(Dispatch {
                    shader,
                    workgroups,
                    input_buffers: vec![q, k, v],
                    output_buffer: out_buf,
                    extra_outputs: vec![lse_buf],
                    params: vec![seq, seq, (num_heads << 16) | num_kv_heads, head_dim],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::CrossAttention {
                num_heads,
                num_kv_heads,
                head_dim,
            } => {
                // Route through unified attention shader.
                let q = self.get_buffer(node.inputs[0]);
                let k = self.get_buffer(node.inputs[1]);
                let v = self.get_buffer(node.inputs[2]);
                let q_seq = self.graph.node(node.inputs[0]).ty.shape[0] as u32;
                let kv_seq = self.graph.node(node.inputs[1]).ty.shape[0] as u32;
                let lse_buf = self.find_lse_buffer(node.id);
                let (shader, workgroups) = self.attention_dispatch(q_seq, head_dim, num_heads);
                self.plan.dispatches.push(Dispatch {
                    shader,
                    workgroups,
                    input_buffers: vec![q, k, v],
                    output_buffer: out_buf,
                    extra_outputs: vec![lse_buf],
                    params: vec![q_seq, kv_seq, (num_heads << 16) | num_kv_heads, head_dim],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::MultiHeadAttn {
                num_heads,
                num_kv_heads,
                head_dim,
                ..
            } => {
                let q = self.get_buffer(node.inputs[0]);
                let k = self.get_buffer(node.inputs[1]);
                let v = self.get_buffer(node.inputs[2]);
                let q_seq = self.graph.node(node.inputs[0]).ty.shape[0] as u32;
                let kv_seq = self.graph.node(node.inputs[1]).ty.shape[0] as u32;
                let lse_buf = self.find_lse_buffer(node.id);
                let (shader, workgroups) = self.attention_dispatch(q_seq, head_dim, num_heads);
                self.plan.dispatches.push(Dispatch {
                    shader,
                    workgroups,
                    input_buffers: vec![q, k, v],
                    output_buffer: out_buf,
                    extra_outputs: vec![lse_buf],
                    params: vec![q_seq, kv_seq, (num_heads << 16) | num_kv_heads, head_dim],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::MultiHeadAttnGradQ {
                fwd_node,
                num_heads,
                num_kv_heads,
                head_dim,
                ..
            } => {
                let d_out = self.get_buffer(node.inputs[0]);
                let q = self.get_buffer(node.inputs[1]);
                let k = self.get_buffer(node.inputs[2]);
                let v = self.get_buffer(node.inputs[3]);
                let fwd_o = self.get_buffer(fwd_node);
                let lse_buf = self.find_lse_buffer(fwd_node);
                let q_seq = self.graph.node(node.inputs[1]).ty.shape[0] as u32;
                let fwd_op = &self.graph.node(fwd_node).op;
                let is_causal = matches!(
                    fwd_op,
                    Op::CausalAttention { .. }
                        | Op::CausalAttentionRoPE { .. }
                        | Op::SlidingWindowAttention { .. }
                );
                let kv_seq = if is_causal {
                    0
                } else {
                    self.graph.node(node.inputs[2]).ty.shape[0] as u32
                };
                let window_size = match *fwd_op {
                    Op::SlidingWindowAttention { window_size, .. } => window_size,
                    _ => 0,
                };
                // The cooperative backward path rounds dO to f16 before its
                // matrix products. Unlike forward activations, small
                // derivatives can lose material information there, so keep
                // this experimental path explicit rather than enabling it
                // merely because the device advertises f16 matrices.
                let bwd_coop_enabled = self.allow_reduced_precision_attention_backward
                    && self.coop_caps.supports_16x16_f16()
                    && head_dim >= 16
                    && head_dim.is_multiple_of(16)
                    && q_seq >= 16;
                let (grad_q_shader, grad_q_wgs) = if bwd_coop_enabled {
                    (
                        ShaderEntry::FlashGradQCoop,
                        [q_seq.div_ceil(16), num_heads, 1],
                    )
                } else {
                    let (raw_shader, wgs) = Self::attention_dispatch_bwd(
                        q_seq,
                        head_dim,
                        num_heads,
                        self.options.knobs.flash_grad_q_ept_cap,
                    );
                    let mapped = match raw_shader {
                        ShaderEntry::FlashAttention => ShaderEntry::FlashGradQ,
                        _ => ShaderEntry::MultiHeadAttnGradQ,
                    };
                    (mapped, wgs)
                };
                self.plan.dispatches.push(Dispatch {
                    shader: grad_q_shader,
                    workgroups: grad_q_wgs,
                    input_buffers: vec![d_out, q, k, v, lse_buf, fwd_o],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![
                        q_seq,
                        kv_seq,
                        (num_heads << 16) | num_kv_heads,
                        head_dim,
                        window_size,
                    ],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::MultiHeadAttnGradK {
                fwd_node,
                num_heads,
                num_kv_heads,
                head_dim,
                ..
            } => {
                let d_out = self.get_buffer(node.inputs[0]);
                let q = self.get_buffer(node.inputs[1]);
                let k = self.get_buffer(node.inputs[2]);
                let v = self.get_buffer(node.inputs[3]);
                let fwd_o = self.get_buffer(fwd_node);
                let lse_buf = self.find_lse_buffer(fwd_node);
                let q_seq = self.graph.node(node.inputs[1]).ty.shape[0] as u32;
                let fwd_op = &self.graph.node(fwd_node).op;
                let is_causal = matches!(
                    fwd_op,
                    Op::CausalAttention { .. }
                        | Op::CausalAttentionRoPE { .. }
                        | Op::SlidingWindowAttention { .. }
                );
                let kv_seq = if is_causal {
                    0
                } else {
                    self.graph.node(node.inputs[2]).ty.shape[0] as u32
                };
                let window_size = match *fwd_op {
                    Op::SlidingWindowAttention { window_size, .. } => window_size,
                    _ => 0,
                };
                let dispatch_kv = if is_causal { q_seq } else { kv_seq };

                // GradKV: pre-allocate dV buffer. When GradV is later
                // compiled for the same fwd_node, it reuses this buffer.
                let dv_buf = self.alloc_buffer(self.graph.node(node.inputs[3]).ty.size_bytes());
                self.fused_grad_kv_dv.insert(fwd_node, dv_buf);
                let attention_params = vec![
                    q_seq,
                    kv_seq,
                    (num_heads << 16) | num_kv_heads,
                    head_dim,
                    window_size,
                ];
                let (grad_kv_shader, grad_kv_wgs) = Self::attention_dispatch_bwd(
                    dispatch_kv,
                    head_dim,
                    num_kv_heads,
                    self.options.knobs.flash_grad_kv_ept_cap,
                );
                // See GradQ above: reduced-input precision in backward is an
                // experimental opt-in until its error is bounded (or loss
                // scaling keeps the derivative operands representable).
                let bwd_coop_enabled = self.allow_reduced_precision_attention_backward
                    && self.coop_caps.supports_16x16_f16()
                    && head_dim >= 16
                    && head_dim.is_multiple_of(16)
                    && dispatch_kv >= 16;
                let (shader, workgroups) = if bwd_coop_enabled {
                    (
                        ShaderEntry::FlashGradKVCoop,
                        [dispatch_kv.div_ceil(16), num_kv_heads, 1],
                    )
                } else {
                    let s = match grad_kv_shader {
                        ShaderEntry::FlashAttention => ShaderEntry::FlashGradKV,
                        _ => ShaderEntry::MultiHeadAttnGradKV,
                    };
                    (s, grad_kv_wgs)
                };
                {
                    self.plan.dispatches.push(Dispatch {
                        shader,
                        workgroups,
                        input_buffers: vec![d_out, q, k, v, lse_buf, fwd_o],
                        output_buffer: out_buf,
                        extra_outputs: vec![dv_buf],
                        params: attention_params,
                        use_coop: false,
                        use_small_tiles: false,
                        ..Default::default()
                    });
                }
            }

            Op::MultiHeadAttnGradV { fwd_node, .. } => {
                // GradK is deliberately compiled as a fused dK+dV dispatch.
                // Autodiff appends GradK before GradV, and topological sorting
                // preserves that dependency-equivalent ID order.
                let dv_buf = *self.fused_grad_kv_dv.get(&fwd_node).unwrap_or_else(|| {
                    panic!(
                        "attention GradV for forward node {fwd_node} compiled before fused GradKV"
                    )
                });
                self.node_buffers.insert(node.id, dv_buf);
                return;
            }

            Op::SwiGLUGradGate => {
                // inputs: [grad_out, gate, up]
                let grad_out = self.get_buffer(node.inputs[0]);
                let gate = self.get_buffer(node.inputs[1]);
                let up = self.get_buffer(node.inputs[2]);
                let len = node.ty.num_elements() as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::SwiGLUGradGate,
                    workgroups: [len.div_ceil(256), 1, 1],
                    input_buffers: vec![grad_out, gate, up],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![len, 0, 0, 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::SwiGLUGradUp => {
                // inputs: [grad_out, gate]
                let grad_out = self.get_buffer(node.inputs[0]);
                let gate = self.get_buffer(node.inputs[1]);
                let len = node.ty.num_elements() as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::SwiGLUGradUp,
                    workgroups: [len.div_ceil(256), 1, 1],
                    input_buffers: vec![grad_out, gate],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![len, 0, 0, 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::SiluGrad => {
                // inputs: [grad_out, x]
                let grad_out = self.get_buffer(node.inputs[0]);
                let x = self.get_buffer(node.inputs[1]);
                let len = node.ty.num_elements() as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::SiluGrad,
                    workgroups: [len.div_ceil(256), 1, 1],
                    input_buffers: vec![grad_out, x],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![len, 0, 0, 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::RmsNormGradW { eps } => {
                let dy = self.get_buffer(node.inputs[0]);
                let x = self.get_buffer(node.inputs[1]);
                let w = self.get_buffer(node.inputs[2]);
                let x_shape = &self.graph.node(node.inputs[1]).ty.shape;
                let rows = x_shape[0] as u32;
                let cols = x_shape[1] as u32;
                if rows >= 4 {
                    // Two-pass row-parallel approach for better GPU occupancy:
                    // Pass 1: one WG per row computes partial[row,col] = dy*x*rsqrt
                    // Pass 2: SumRows reduces partial → grad_w[col]
                    let temp_buf = self.alloc_buffer((rows as usize) * (cols as usize) * 4);
                    self.plan.dispatches.push(Dispatch {
                        shader: ShaderEntry::RmsNormGradWRowPar,
                        workgroups: [rows, 1, 1],
                        input_buffers: vec![dy, x, w],
                        output_buffer: temp_buf,
                        extra_outputs: vec![],
                        params: vec![rows, cols, eps.to_bits(), 0],
                        use_coop: false,
                        use_small_tiles: false,
                        ..Default::default()
                    });
                    self.plan.dispatches.push(Dispatch {
                        shader: ShaderEntry::SumRows,
                        workgroups: [cols.div_ceil(32), 1, 1],
                        input_buffers: vec![temp_buf],
                        output_buffer: out_buf,
                        extra_outputs: vec![],
                        params: vec![rows, cols, 0, 0],
                        use_coop: false,
                        use_small_tiles: false,
                        ..Default::default()
                    });
                } else {
                    // Small row count: single-pass is fine
                    self.plan.dispatches.push(Dispatch {
                        shader: ShaderEntry::RmsNormGradW,
                        workgroups: [cols.div_ceil(256), 1, 1],
                        input_buffers: vec![dy, x, w],
                        output_buffer: out_buf,
                        extra_outputs: vec![],
                        params: vec![rows, cols, eps.to_bits(), 0],
                        use_coop: false,
                        use_small_tiles: false,
                        ..Default::default()
                    });
                }
            }

            Op::RmsNormGradX { eps } => {
                let dy = self.get_buffer(node.inputs[0]);
                let x = self.get_buffer(node.inputs[1]);
                let w = self.get_buffer(node.inputs[2]);
                let x_shape = &self.graph.node(node.inputs[1]).ty.shape;
                let rows = x_shape[0] as u32;
                let cols = x_shape[1] as u32;
                const WG: u32 = 256;
                let lanes_per_row = if (2..=32).contains(&cols) {
                    cols.next_power_of_two()
                } else {
                    0
                };
                let workgroups = WG
                    .checked_div(lanes_per_row)
                    .map_or(rows, |packed_rows| rows.div_ceil(packed_rows));
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::RmsNormGradX,
                    workgroups: [workgroups, 1, 1],
                    input_buffers: vec![dy, x, w],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![rows, cols, eps.to_bits(), lanes_per_row],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::LayerNormGradWB { eps } => {
                let dy = self.get_buffer(node.inputs[0]);
                let x = self.get_buffer(node.inputs[1]);
                let w = self.get_buffer(node.inputs[2]);
                let x_shape = &self.graph.node(node.inputs[1]).ty.shape;
                let rows = x_shape[0] as u32;
                let cols = x_shape[1] as u32;
                if rows >= 4 {
                    // Row-parallel: each WG handles one row, SumRows reduces.
                    let temp_buf = self.alloc_buffer((rows as usize) * (cols as usize) * 4);
                    self.plan.dispatches.push(Dispatch {
                        shader: ShaderEntry::LayerNormGradWB,
                        workgroups: [rows, 1, 1],
                        input_buffers: vec![dy, x, w],
                        output_buffer: temp_buf,
                        extra_outputs: vec![],
                        params: vec![rows, cols, eps.to_bits(), 0],
                        use_coop: false,
                        use_small_tiles: false,
                        ..Default::default()
                    });
                    self.plan.dispatches.push(Dispatch {
                        shader: ShaderEntry::SumRows,
                        workgroups: [cols.div_ceil(32), 1, 1],
                        input_buffers: vec![temp_buf],
                        output_buffer: out_buf,
                        extra_outputs: vec![],
                        params: vec![rows, cols, 0, 0],
                        use_coop: false,
                        use_small_tiles: false,
                        ..Default::default()
                    });
                } else {
                    self.plan.dispatches.push(Dispatch {
                        shader: ShaderEntry::LayerNormGradWB,
                        workgroups: [rows, 1, 1],
                        input_buffers: vec![dy, x, w],
                        output_buffer: out_buf,
                        extra_outputs: vec![],
                        params: vec![rows, cols, eps.to_bits(), 0],
                        use_coop: false,
                        use_small_tiles: false,
                        ..Default::default()
                    });
                }
            }

            Op::LayerNormGradX { eps } => {
                let dy = self.get_buffer(node.inputs[0]);
                let x = self.get_buffer(node.inputs[1]);
                let w = self.get_buffer(node.inputs[2]);
                let x_shape = &self.graph.node(node.inputs[1]).ty.shape;
                let rows = x_shape[0] as u32;
                let cols = x_shape[1] as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::LayerNormGradX,
                    workgroups: [rows, 1, 1],
                    input_buffers: vec![dy, x, w],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![rows, cols, eps.to_bits(), 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }
        }

        // A single graph node can lower to multiple dispatches (for example,
        // row-parallel normalization gradients). Preserve the node's numeric
        // policy on every emitted dispatch so runtime kernel selection cannot
        // silently turn f32 autodiff math into f16-input math.
        for dispatch in &mut self.plan.dispatches[dispatch_start..] {
            dispatch.requires_full_precision = node.requires_full_precision;
        }
    }

    fn find_lse_buffer(&self, fwd_node: NodeId) -> BufferRef {
        self.plan
            .lse_buffers
            .iter()
            .find(|item| item.0 == fwd_node)
            .expect("LSE buffer not found for MultiHeadAttn forward node")
            .1
    }

    /// Emit softmax as two schedule-template reductions:
    ///
    ///   1. Max reduction (identity prologue) → `row_max` buffer `[batch]`.
    ///   2. Sum reduction with prologue `exp(src - row_max)` and epilogue
    ///      `exp(src - row_max) / row_sum` → output `[batch, features]`.
    ///
    /// Matches softmax.wgsl semantics bit-for-bit on finite inputs. No
    /// workgroup-size autotune yet — hardcoded at 256, which matches the
    /// existing softmax's tile size.
    fn emit_softmax_schedule(
        &mut self,
        input: BufferRef,
        out_buf: BufferRef,
        batch: u32,
        features: u32,
    ) {
        use crate::schedule::{PointwiseDAG, Pw, ReduceOp, ReductionEpilogue, ReductionKernel};

        const WG: u32 = 256;

        // Allocate intermediate row_max buffer: batch × f32.
        let row_max = self.alloc_buffer((batch as usize) * 4);

        // --- Dispatch 1: row-wise max reduction ---
        let max_prologue = PointwiseDAG {
            n_inputs: 1,
            ops: vec![Pw::LoadInput(0)],
            output: 0,
        };
        let max_kernel = ReductionKernel {
            op: ReduceOp::Max,
            prologue: max_prologue,
            extra_prologues: vec![],
            epilogue: None,
            n_per_elem: 1,
            n_per_row: 0,
            workgroup_size: WG,
            rows_per_workgroup: 1,
            gather_elem: Vec::new(),
            input_row_repeats: Vec::new(),
        };
        self.plan.dispatches.push(Dispatch {
            // Sentinel shader for runtime data-layout selection (UnaryData).
            shader: ShaderEntry::Relu,
            workgroups: [batch, 1, 1],
            input_buffers: vec![input],
            output_buffer: row_max,
            extra_outputs: vec![],
            params: vec![batch, features, 0, 0],
            use_coop: false,
            use_small_tiles: false,
            reduction: Some(max_kernel),
            ..Default::default()
        });

        // --- Dispatch 2: sum reduction with exp-subtract prologue + normalize epilogue ---
        // Prologue DAG: inputs are 0=src (per-elem), 1=row_max (per-row).
        //   exp(src - row_max) → scalar contribution.
        let sum_prologue = PointwiseDAG {
            n_inputs: 2,
            ops: vec![
                Pw::LoadInput(0), // v0 = src
                Pw::LoadInput(1), // v1 = row_max
                Pw::Sub(0, 1),    // v2 = src - row_max
                Pw::Exp(2),       // v3 = exp(src - row_max)
            ],
            output: 3,
        };
        // Epilogue: inputs 0=src (per-elem), 1=row_max (per-row),
        //           2=row_sum (reduced scalar, always last).
        let sum_epilogue_dag = PointwiseDAG {
            n_inputs: 3,
            ops: vec![
                Pw::LoadInput(0), // v0 = src
                Pw::LoadInput(1), // v1 = row_max
                Pw::LoadInput(2), // v2 = row_sum
                Pw::Sub(0, 1),    // v3 = src - row_max
                Pw::Exp(3),       // v4 = exp(src - row_max)
                Pw::Div(4, 2),    // v5 = exp(...) / row_sum
            ],
            output: 5,
        };
        let sum_kernel = ReductionKernel {
            op: ReduceOp::Sum,
            prologue: sum_prologue,
            extra_prologues: vec![],
            epilogue: Some(ReductionEpilogue {
                dag: sum_epilogue_dag,
                n_per_col_inputs: 0,
            }),
            n_per_elem: 1,
            n_per_row: 1,
            workgroup_size: WG,
            rows_per_workgroup: 1,
            gather_elem: Vec::new(),
            input_row_repeats: Vec::new(),
        };
        self.plan.dispatches.push(Dispatch {
            // Sentinel for runtime data-layout: 1 per-elem + 1 per-row → 2
            // buffer inputs → we key the runtime off `reduction.is_some()`
            // and the kernel's arity, so the shader field is purely a
            // historical leftover here.
            shader: ShaderEntry::Add,
            workgroups: [batch, 1, 1],
            input_buffers: vec![input, row_max],
            output_buffer: out_buf,
            extra_outputs: vec![],
            params: vec![batch, features, 0, 0],
            use_coop: false,
            use_small_tiles: false,
            reduction: Some(sum_kernel),
            ..Default::default()
        });
    }

    /// Emit LayerNorm as a single schedule-template reduction with two
    /// accumulators:
    ///   prologues: x and x*x (sum and sum-of-squares)
    ///   op: Sum
    ///   epilogue: (x - mean) * rsqrt(var + eps) * weight[col] + bias[col]
    ///   where mean = r0/cols, var = r1/cols - mean².
    #[allow(clippy::too_many_arguments)]
    fn emit_layernorm_schedule(
        &mut self,
        x: BufferRef,
        w: BufferRef,
        bias: BufferRef,
        out_buf: BufferRef,
        rows: u32,
        cols: u32,
        eps: f32,
    ) {
        use crate::schedule::{PointwiseDAG, Pw, ReduceOp, ReductionEpilogue, ReductionKernel};

        const WG: u32 = 256;

        let sum_x = PointwiseDAG {
            n_inputs: 1,
            ops: vec![Pw::LoadInput(0)],
            output: 0,
        };
        let sum_x2 = PointwiseDAG {
            n_inputs: 1,
            ops: vec![Pw::LoadInput(0), Pw::Mul(0, 0)],
            output: 1,
        };

        // Epilogue inputs (canonical layout):
        //   0 = x[row, col]   (per-elem)
        //   1 = weight[col]   (per-col)
        //   2 = bias[col]     (per-col)
        //   3 = sum(x)        (reduced 0)
        //   4 = sum(x*x)      (reduced 1)
        let inv_cols = Pw::const_f32(1.0 / cols as f32);
        let eps_c = Pw::const_f32(eps);
        let epilogue_dag = PointwiseDAG {
            n_inputs: 5,
            ops: vec![
                Pw::LoadInput(0), // v0 = x
                Pw::LoadInput(1), // v1 = weight
                Pw::LoadInput(2), // v2 = bias
                Pw::LoadInput(3), // v3 = sum(x)
                Pw::LoadInput(4), // v4 = sum(x*x)
                inv_cols,         // v5 = 1/cols
                eps_c,            // v6 = eps
                Pw::Mul(3, 5),    // v7 = mean
                Pw::Mul(4, 5),    // v8 = E[x²]
                Pw::Mul(7, 7),    // v9 = mean²
                Pw::Sub(8, 9),    // v10 = var
                Pw::Add(10, 6),   // v11 = var + eps
                Pw::Rsqrt(11),    // v12 = rsqrt(var + eps)
                Pw::Sub(0, 7),    // v13 = x - mean
                Pw::Mul(13, 12),  // v14 = normalized
                Pw::Mul(14, 1),   // v15 = * weight
                Pw::Add(15, 2),   // v16 = + bias
            ],
            output: 16,
        };

        let kernel = ReductionKernel {
            op: ReduceOp::Sum,
            prologue: sum_x,
            extra_prologues: vec![sum_x2],
            epilogue: Some(ReductionEpilogue {
                dag: epilogue_dag,
                n_per_col_inputs: 2,
            }),
            n_per_elem: 1,
            n_per_row: 0,
            workgroup_size: WG,
            rows_per_workgroup: 1,
            gather_elem: Vec::new(),
            input_row_repeats: Vec::new(),
        };

        self.plan.dispatches.push(Dispatch {
            shader: ShaderEntry::LayerNorm, // sentinel for layout family
            workgroups: [rows, 1, 1],
            input_buffers: vec![x, w, bias],
            output_buffer: out_buf,
            extra_outputs: vec![],
            params: vec![rows, cols, eps.to_bits(), 0],
            use_coop: false,
            use_small_tiles: false,
            reduction: Some(kernel),
            ..Default::default()
        });
    }

    /// Emit RmsNorm as a single schedule-template reduction:
    ///   prologue: v*v (sum-of-squares)
    ///   op: Sum
    ///   epilogue: src * rsqrt(sum_sq / cols + eps) * weight[col]
    fn emit_rmsnorm_schedule(
        &mut self,
        x: BufferRef,
        w: BufferRef,
        out_buf: BufferRef,
        rows: u32,
        cols: u32,
        eps: f32,
    ) {
        use crate::schedule::{PointwiseDAG, Pw, ReduceOp, ReductionEpilogue, ReductionKernel};

        const WG: u32 = 256;
        // A power-of-two lane group at least as wide as the row preserves the
        // existing tree-reduction order. Pack several narrow rows into the
        // otherwise idle lanes of one workgroup.
        let rows_per_workgroup = if (2..=32).contains(&cols) {
            WG / cols.next_power_of_two()
        } else {
            1
        };

        // Prologue: v*v → scalar contribution to sum-of-squares.
        let prologue = PointwiseDAG {
            n_inputs: 1,
            ops: vec![Pw::LoadInput(0), Pw::Mul(0, 0)],
            output: 1,
        };

        // Epilogue inputs (per the canonical layout):
        //   0 = src[row, col]    (per-elem)
        //   1 = weight[col]      (per-col)
        //   2 = sum_of_squares   (reduced scalar, always last)
        //
        // Computes: src * rsqrt(sum_sq * inv_cols + eps) * weight
        let inv_cols = Pw::const_f32(1.0 / cols as f32);
        let eps_c = Pw::const_f32(eps);
        let epilogue_dag = PointwiseDAG {
            n_inputs: 3,
            ops: vec![
                Pw::LoadInput(0), // v0 = src[row, col]
                Pw::LoadInput(1), // v1 = weight[col]
                Pw::LoadInput(2), // v2 = sum_of_squares
                inv_cols,         // v3 = 1/cols
                eps_c,            // v4 = eps
                Pw::Mul(2, 3),    // v5 = mean_sq
                Pw::Add(5, 4),    // v6 = mean_sq + eps
                Pw::Rsqrt(6),     // v7 = rsqrt(...)
                Pw::Mul(0, 7),    // v8 = src * rsqrt
                Pw::Mul(8, 1),    // v9 = (src * rsqrt) * weight
            ],
            output: 9,
        };

        let kernel = ReductionKernel {
            op: ReduceOp::Sum,
            prologue,
            extra_prologues: vec![],
            epilogue: Some(ReductionEpilogue {
                dag: epilogue_dag,
                n_per_col_inputs: 1,
            }),
            n_per_elem: 1,
            n_per_row: 0,
            workgroup_size: WG,
            rows_per_workgroup,
            gather_elem: Vec::new(),
            input_row_repeats: Vec::new(),
        };

        // Uses RmsNormData layout: src + bias (per-col weight) + dst + params.
        self.plan.dispatches.push(Dispatch {
            shader: ShaderEntry::RmsNorm, // sentinel for layout
            workgroups: [rows.div_ceil(rows_per_workgroup), 1, 1],
            input_buffers: vec![x, w],
            output_buffer: out_buf,
            extra_outputs: vec![],
            // The schedule shader embeds epsilon in its DAG, but retain the
            // bits in dispatch metadata as well: runtime may rewrite this
            // dispatch to RmsNormRsqrt for a cooperative matmul prologue.
            params: vec![rows, cols, eps.to_bits(), 0],
            use_coop: false,
            use_small_tiles: false,
            reduction: Some(kernel),
            ..Default::default()
        });
    }

    fn emit_generated_unary(&mut self, op: Pw, node: &Node, out_buf: BufferRef) {
        let input = self.get_buffer(node.inputs[0]);
        let len = node.ty.num_elements() as u32;
        self.plan.dispatches.push(Dispatch {
            shader: ShaderEntry::Relu,
            workgroups: [len.div_ceil(256), 1, 1],
            input_buffers: vec![input],
            output_buffer: out_buf,
            extra_outputs: vec![],
            params: vec![len, 0, 0, 0],
            use_coop: false,
            use_small_tiles: false,
            pointwise: Some(PointwiseDAG {
                n_inputs: 1,
                ops: vec![Pw::LoadInput(0), op],
                output: 1,
            }),
            ..Default::default()
        });
    }

    fn emit_unary(&mut self, shader: ShaderEntry, node: &Node, out_buf: BufferRef) {
        let input = self.get_buffer(node.inputs[0]);
        let len = node.ty.num_elements() as u32;
        let pointwise = if self.options.use_schedule_pointwise {
            unary_shader_to_pointwise(&shader)
        } else {
            None
        };
        self.plan.dispatches.push(Dispatch {
            shader,
            workgroups: [len.div_ceil(256), 1, 1],
            input_buffers: vec![input],
            output_buffer: out_buf,
            extra_outputs: vec![],
            params: vec![len, 0, 0, 0],
            use_coop: false,
            use_small_tiles: false,
            pointwise,
            ..Default::default()
        });
    }

    fn emit_binary(&mut self, shader: ShaderEntry, node: &Node, out_buf: BufferRef) {
        let a = self.get_buffer(node.inputs[0]);
        let b = self.get_buffer(node.inputs[1]);
        let len = node.ty.num_elements() as u32;
        let pointwise = if self.options.use_schedule_pointwise {
            binary_shader_to_pointwise(&shader)
        } else {
            None
        };
        self.plan.dispatches.push(Dispatch {
            shader,
            workgroups: [len.div_ceil(256), 1, 1],
            input_buffers: vec![a, b],
            output_buffer: out_buf,
            extra_outputs: vec![],
            params: vec![len, 0, 0, 0],
            use_coop: false,
            use_small_tiles: false,
            pointwise,
            ..Default::default()
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;

    #[test]
    fn test_compile_simple() {
        let mut g = Graph::new();
        let x = g.input("x", &[4, 784]);
        let w = g.parameter("w", &[784, 128]);
        let y = g.matmul(x, w);
        let h = g.relu(y);
        g.set_outputs(vec![h]);

        let plan = compile(&g);
        assert_eq!(plan.input_buffers.len(), 1);
        assert_eq!(plan.param_buffers.len(), 1);
        assert_eq!(plan.dispatches.len(), 1); // matmul with fused relu epilogue
    }

    #[test]
    fn test_compile_fused() {
        let mut g = Graph::new();
        let x = g.input("x", &[4, 784]);
        let w = g.parameter("w", &[784, 128]);
        let y = g.matmul(x, w);
        let h = g.relu(y);
        g.set_outputs(vec![h]);

        let optimized = crate::optimize::optimize(&g);
        let plan = compile(&optimized);
        // MatMul with Relu fused into epilogue (epilogue fusion pass)
        assert_eq!(plan.dispatches.len(), 1);
        assert_eq!(plan.dispatches[0].shader, ShaderEntry::MatMul);
        assert_eq!(plan.dispatches[0].epilogue, vec![EpilogueOp::Relu]);
    }

    #[test]
    fn test_compile_all_unary_ops() {
        let mut g = Graph::new();
        let x = g.input("x", &[4, 8]);
        let r = g.relu(x);
        let s = g.sigmoid(x);
        let n = g.neg(x);
        let e = g.exp(x);
        g.set_outputs(vec![r, s, n, e]);

        let plan = compile(&g);
        assert_eq!(plan.dispatches.len(), 4);
        assert_eq!(plan.dispatches[0].shader, ShaderEntry::Relu);
        assert_eq!(plan.dispatches[1].shader, ShaderEntry::Sigmoid);
        assert_eq!(plan.dispatches[2].shader, ShaderEntry::Neg);
        assert_eq!(plan.dispatches[3].shader, ShaderEntry::Relu);
        assert!(plan.dispatches[3].pointwise.is_some());
        // All unary ops: params = [len, 0, 0, 0]
        for d in &plan.dispatches {
            assert_eq!(d.params[0], 32); // 4*8
            assert_eq!(d.input_buffers.len(), 1);
        }
    }

    #[test]
    fn test_materialize_is_a_distinct_non_fused_dispatch() {
        let mut g = Graph::new();
        let x = g.input("x", &[2, 4]);
        let staged = g.materialize(x);
        let first = g.split_a(staged, 2, 3, 1, 1);
        g.set_outputs(vec![first]);

        let optimized = crate::optimize::optimize(&g);
        let plan = compile(&optimized);

        assert_eq!(plan.dispatches.len(), 2);
        let copy = &plan.dispatches[0];
        assert_eq!(copy.shader, ShaderEntry::Relu);
        assert_eq!(copy.params, [8, 0, 0, 0]);
        assert_eq!(copy.workgroups, [1, 1, 1]);
        assert_eq!(copy.input_buffers.len(), 1);
        assert_ne!(copy.input_buffers[0], copy.output_buffer);
        assert!(copy.pointwise.is_some());
        assert!(copy.fusion_barrier);

        let split = &plan.dispatches[1];
        assert_eq!(split.shader, ShaderEntry::SplitA);
        assert_eq!(split.input_buffers, [copy.output_buffer]);
    }

    #[test]
    fn test_compile_all_binary_ops() {
        let mut g = Graph::new();
        let a = g.input("a", &[4, 8]);
        let b = g.input("b", &[4, 8]);
        let add = g.add(a, b);
        let mul = g.mul(a, b);
        let gt = g.greater(a, b);
        g.set_outputs(vec![add, mul, gt]);

        let plan = compile(&g);
        assert_eq!(plan.dispatches.len(), 3);
        assert_eq!(plan.dispatches[0].shader, ShaderEntry::Add);
        assert_eq!(plan.dispatches[1].shader, ShaderEntry::Mul);
        assert_eq!(plan.dispatches[2].shader, ShaderEntry::Greater);
        for d in &plan.dispatches {
            assert_eq!(d.input_buffers.len(), 2);
            assert_eq!(d.params[0], 32);
        }
    }

    #[test]
    fn test_compile_bias_add() {
        let mut g = Graph::new();
        let x = g.input("x", &[4, 128]);
        let b = g.parameter("b", &[128]);
        let out = g.bias_add(x, b);
        g.set_outputs(vec![out]);

        let plan = compile(&g);
        assert_eq!(plan.dispatches.len(), 1);
        assert_eq!(plan.dispatches[0].shader, ShaderEntry::BiasAdd);
        assert_eq!(plan.dispatches[0].params[0], 512); // 4*128
        assert_eq!(plan.dispatches[0].params[1], 128); // bias len
    }

    #[test]
    fn test_compile_reductions() {
        let mut g = Graph::new();
        let x = g.input("x", &[4, 8]);
        let sa = g.sum_all(x);
        let ma = g.mean_all(x);
        g.set_outputs(vec![sa, ma]);

        let plan = compile(&g);
        assert_eq!(plan.dispatches.len(), 2);
        assert_eq!(plan.dispatches[0].shader, ShaderEntry::SumAll);
        assert_eq!(plan.dispatches[1].shader, ShaderEntry::MeanAll);
        // params = [len, 0, 0, 0]
        for d in &plan.dispatches {
            assert_eq!(d.params[0], 32);
        }
    }

    #[test]
    fn sum_inner_packs_narrow_rows_only() {
        let mut narrow = Graph::new();
        let input = narrow.input("input", &[100, 9]);
        let output = narrow.sum_inner(input);
        narrow.set_outputs(vec![output]);
        let narrow_plan = compile(&narrow);
        let narrow_dispatch = &narrow_plan.dispatches[0];
        assert_eq!(narrow_dispatch.workgroups, [1, 1, 1]);
        assert_eq!(
            narrow_dispatch
                .reduction
                .as_ref()
                .unwrap()
                .rows_per_workgroup,
            256
        );

        let mut wide = Graph::new();
        let input = wide.input("input", &[100, 33]);
        let output = wide.sum_inner(input);
        wide.set_outputs(vec![output]);
        let wide_plan = compile(&wide);
        let wide_dispatch = &wide_plan.dispatches[0];
        assert_eq!(wide_dispatch.workgroups, [100, 1, 1]);
        assert_eq!(
            wide_dispatch.reduction.as_ref().unwrap().rows_per_workgroup,
            1
        );

        let mut product = Graph::new();
        let a = product.input("a", &[100, 9]);
        let b = product.input("b", &[100, 9]);
        let terms = product.mul(a, b);
        let output = product.sum_inner(terms);
        product.set_outputs(vec![output]);
        let product_plan = compile(&product);
        assert_eq!(
            product_plan.dispatches.len(),
            1,
            "narrow reductions should fold their pointwise producer"
        );
        let product_kernel = product_plan.dispatches[0].reduction.as_ref().unwrap();
        assert_eq!(product_kernel.n_per_elem, 2);
    }

    #[test]
    fn unit_column_matmuls_use_exact_inner_kernels() {
        let mut forward = Graph::new();
        let input = forward.input("input", &[100, 3]);
        let ones = forward.constant(vec![1.0; 3], &[3, 1]);
        let output = forward.matmul(input, ones);
        forward.set_outputs(vec![output]);
        let forward_plan = compile(&forward);
        assert_eq!(forward_plan.dispatches.len(), 1);
        let reduction = &forward_plan.dispatches[0];
        assert!(reduction.reduction.is_some());
        assert_eq!(reduction.params[..2], [100, 3]);
        assert_eq!(reduction.input_buffers.len(), 1);

        let mut backward = Graph::new();
        let row_gradient = backward.input("row_gradient", &[100, 1]);
        let ones = backward.constant(vec![1.0; 3], &[3, 1]);
        let output = backward.matmul_bt(row_gradient, ones);
        backward.set_outputs(vec![output]);
        let backward_plan = compile(&backward);
        assert_eq!(backward_plan.dispatches.len(), 1);
        let broadcast = &backward_plan.dispatches[0];
        assert!(broadcast.is_inner_broadcast());
        assert_eq!(broadcast.params, [300, 3, 1, 0]);
        assert_eq!(broadcast.input_buffers.len(), 1);

        let mut non_unit = Graph::new();
        let input = non_unit.input("input", &[100, 3]);
        let weights = non_unit.constant(vec![1.0, 2.0, 1.0], &[3, 1]);
        let output = non_unit.matmul(input, weights);
        non_unit.set_outputs(vec![output]);
        let non_unit_plan = compile(&non_unit);
        assert_eq!(non_unit_plan.dispatches.len(), 1);
        assert_eq!(non_unit_plan.dispatches[0].shader, ShaderEntry::MatMul);
        assert!(non_unit_plan.dispatches[0].reduction.is_none());
    }

    #[test]
    fn sum_inner_gradient_uses_direct_row_broadcast() {
        let mut graph = Graph::new();
        let input = graph.parameter("input", &[513, 16]);
        let rows = graph.sum_inner(input);
        let loss = graph.sum_all(rows);
        graph.set_outputs(vec![loss]);

        let differentiated = crate::autodiff::differentiate(&graph);
        let plan = compile(&differentiated);
        let broadcast = plan
            .dispatches
            .iter()
            .find(|dispatch| {
                dispatch.is_inner_broadcast() && dispatch.params == [513 * 16, 16, 1, 0]
            })
            .expect("sum_inner backward should emit a direct row broadcast");
        assert_eq!(broadcast.workgroups, [33, 1, 1]);
    }

    #[test]
    fn rms_norm_packs_narrow_rows_only() {
        let mut narrow = Graph::new();
        let x = narrow.input("x", &[100, 3]);
        let weight = narrow.input("weight", &[3]);
        let output = narrow.rms_norm(x, weight, 1.0e-6);
        narrow.set_outputs(vec![output]);
        let narrow_plan = compile(&narrow);
        let forward = &narrow_plan.dispatches[0];
        assert_eq!(forward.workgroups, [2, 1, 1]);
        assert_eq!(forward.reduction.as_ref().unwrap().rows_per_workgroup, 64);

        let mut narrow_grad = Graph::new();
        let dy = narrow_grad.input("dy", &[100, 3]);
        let x = narrow_grad.input("x", &[100, 3]);
        let weight = narrow_grad.input("weight", &[3]);
        let output = narrow_grad.rms_norm_grad_x(dy, x, weight, 1.0e-6);
        narrow_grad.set_outputs(vec![output]);
        let narrow_grad_plan = compile(&narrow_grad);
        let grad_x = &narrow_grad_plan.dispatches[0];
        assert_eq!(grad_x.workgroups, [2, 1, 1]);
        assert_eq!(grad_x.params[3], 4);

        let mut wide = Graph::new();
        let dy = wide.input("dy", &[100, 33]);
        let x = wide.input("x", &[100, 33]);
        let weight = wide.input("weight", &[33]);
        let output = wide.rms_norm_grad_x(dy, x, weight, 1.0e-6);
        wide.set_outputs(vec![output]);
        let wide_plan = compile(&wide);
        assert_eq!(wide_plan.dispatches[0].workgroups, [100, 1, 1]);
        assert_eq!(wide_plan.dispatches[0].params[3], 0);
    }

    #[test]
    fn embedding_dispatch_covers_only_output_elements() {
        let mut f32_graph = Graph::new();
        let indices = f32_graph.input_u32("indices", &[513]);
        let table = f32_graph.input("table", &[17, 3]);
        let output = f32_graph.embedding(indices, table);
        f32_graph.set_outputs(vec![output]);
        let f32_plan = compile(&f32_graph);
        assert_eq!(f32_plan.dispatches[0].shader, ShaderEntry::Embedding);
        assert_eq!(f32_plan.dispatches[0].workgroups, [7, 1, 1]);

        let mut f16_graph = Graph::new();
        let indices = f16_graph.input_u32("indices", &[513]);
        let table = f16_graph.input("table", &[17, 257]);
        let table = f16_graph.to_f16(table);
        let output = f16_graph.embedding_f16(indices, table);
        f16_graph.set_outputs(vec![output]);
        let f16_plan = compile(&f16_graph);
        let embedding = f16_plan
            .dispatches
            .iter()
            .find(|dispatch| dispatch.shader == ShaderEntry::Embedding)
            .unwrap();
        assert_eq!(embedding.workgroups, [516, 1, 1]);
        assert_eq!(embedding.weight_format, WeightFormat::F16);
    }

    #[test]
    fn scatter_add_uses_atomic_path_only_for_large_workloads() {
        let mut small = Graph::new();
        let small_indices = small.input_u32("indices", &[2]);
        let small_src = small.input("src", &[2, 3]);
        let small_output = small.scatter_add(small_indices, small_src, 4);
        small.set_outputs(vec![small_output]);
        let small_plan = compile(&small);
        assert_eq!(small_plan.dispatches.len(), 1);
        assert_eq!(small_plan.dispatches[0].shader, ShaderEntry::ScatterAdd);

        let mut large = Graph::new();
        let large_indices = large.input_u32("indices", &[256]);
        let large_src = large.input("src", &[256, 3]);
        let large_output = large.scatter_add(large_indices, large_src, 4097);
        large.set_outputs(vec![large_output]);
        let large_plan = compile(&large);
        assert_eq!(large_plan.dispatches.len(), 2);
        assert!(large_plan.dispatches[0].is_zero_fill());
        assert_eq!(
            large_plan.dispatches[1].shader,
            ShaderEntry::ScatterAddAtomic
        );
        assert!(
            large_plan.dispatches[1]
                .input_buffers
                .contains(&large_plan.dispatches[1].output_buffer)
        );
    }

    #[test]
    fn gather_reduction_gradient_fuses_row_scaled_atomic_scatter() {
        const SEQ: usize = 256;
        const VOCAB: usize = 4097;
        const INNER: usize = 16;

        let mut graph = Graph::new();
        let indices = graph.input_u32("indices", &[SEQ]);
        let table = graph.parameter("table", &[VOCAB, INNER]);
        let factors = graph.input("factors", &[SEQ, INNER]);
        let gathered = graph.embedding(indices, table);
        let terms = graph.mul(gathered, factors);
        let rows = graph.sum_inner(terms);
        let row_scale = graph.input("row_scale", &[SEQ, 1]);
        let weighted = graph.mul(rows, row_scale);
        let loss = graph.sum_all(weighted);
        graph.set_outputs(vec![loss]);

        let (plan, _) = crate::train::compile_training_graph(&graph);
        let fused = plan
            .dispatches
            .iter()
            .find(|dispatch| dispatch.is_row_scaled_atomic_scatter())
            .expect("gathered row reduction should fuse its table gradient");
        assert_eq!(
            fused.params,
            [VOCAB as u32 * INNER as u32, SEQ as u32, INNER as u32, 2]
        );
        assert_eq!(fused.workgroups, [1, 1, 1]);
        assert_eq!(fused.input_buffers.len(), 4);
        assert_eq!(fused.input_buffers[3], fused.output_buffer);
        assert!(!plan.dispatches.iter().any(|dispatch| {
            dispatch.is_inner_broadcast()
                && dispatch.params == [SEQ as u32 * INNER as u32, INNER as u32, 1, 0]
        }));

        let zero = plan
            .dispatches
            .iter()
            .find(|dispatch| {
                dispatch.is_zero_fill() && dispatch.output_buffer == fused.output_buffer
            })
            .expect("fused atomic scatter still needs its zeroing pass");
        assert_eq!(zero.input_buffers[0], fused.input_buffers[1]);
    }

    #[test]
    fn wide_gather_reduction_keeps_scalar_atomic_mapping() {
        const SEQ: usize = 256;
        const VOCAB: usize = 4097;
        const INNER: usize = 32;

        let mut graph = Graph::new();
        let indices = graph.input_u32("indices", &[SEQ]);
        let table = graph.parameter("table", &[VOCAB, INNER]);
        let factors = graph.input("factors", &[SEQ, INNER]);
        let gathered = graph.embedding(indices, table);
        let terms = graph.mul(gathered, factors);
        let rows = graph.sum_inner(terms);
        let row_scale = graph.input("row_scale", &[SEQ, 1]);
        let weighted = graph.mul(rows, row_scale);
        let loss = graph.sum_all(weighted);
        graph.set_outputs(vec![loss]);

        let (plan, _) = crate::train::compile_training_graph(&graph);
        let fused = plan
            .dispatches
            .iter()
            .find(|dispatch| dispatch.is_row_scaled_atomic_scatter())
            .expect("gathered row reduction should fuse its table gradient");
        assert_eq!(fused.params[3], 1);
        assert_eq!(fused.workgroups, [INNER.div_ceil(256) as u32, 1, 1]);
    }

    #[test]
    fn test_compile_softmax() {
        let mut g = Graph::new();
        let x = g.input("x", &[4, 10]);
        let sm = g.softmax(x);
        g.set_outputs(vec![sm]);

        let plan = compile(&g);
        // With use_schedule_reduction=true (default), softmax compiles to
        // 2 Reduction dispatches (max + sum/normalize). With =false, it's
        // 1 Softmax dispatch. Check that it compiles and has the right
        // batch/features params.
        assert!(!plan.dispatches.is_empty());
        assert_eq!(plan.dispatches[0].params[0], 4); // batch/outer
        assert_eq!(plan.dispatches[0].params[1], 10); // features/inner
    }

    #[test]
    fn pointwise_into_softmax_updates_reduction_epilogue_arity() {
        let mut g = Graph::new();
        let x = g.input("x", &[4, 10]);
        let shifted = g.neg(x);
        let sm = g.softmax(shifted);
        g.set_outputs(vec![sm]);

        let plan = compile(&g);
        assert!(!plan.dispatches.is_empty());
        for dispatch in &plan.dispatches {
            if let Some(reduction) = dispatch.reduction.as_ref()
                && let Some(epilogue) = reduction.epilogue.as_ref()
            {
                assert_eq!(
                    epilogue.dag.n_inputs,
                    reduction.n_per_elem + reduction.n_per_row + epilogue.n_per_col_inputs + 1
                );
            }
        }
    }

    #[test]
    fn test_compile_cross_entropy() {
        let mut g = Graph::new();
        let logits = g.input("logits", &[4, 10]);
        let labels = g.input("labels", &[4, 10]);
        let loss = g.cross_entropy_loss(logits, labels);
        g.set_outputs(vec![loss]);

        let plan = compile(&g);
        assert_eq!(plan.dispatches.len(), 1);
        assert_eq!(plan.dispatches[0].shader, ShaderEntry::CrossEntropyLoss);
        assert_eq!(plan.dispatches[0].workgroups, [4, 1, 1]);
        assert_eq!(plan.dispatches[0].params[0], 4);
        assert_eq!(plan.dispatches[0].params[1], 10);
        assert_eq!(plan.dispatches[0].params[2], 0);
    }

    #[test]
    fn training_cross_entropy_reuses_fused_logits_grad() {
        let mut g = Graph::new();
        let logits = g.parameter("logits", &[4, 10]);
        let labels = g.input("labels", &[4, 10]);
        let loss = g.cross_entropy_loss(logits, labels);
        g.set_outputs(vec![loss]);

        let (plan, _) = crate::train::compile_training_graph(&g);
        let ce = plan
            .dispatches
            .iter()
            .find(|d| d.shader == ShaderEntry::CrossEntropyLoss)
            .expect("CE forward");
        assert_eq!(ce.params[2], 1, "training CE must write the fused grad");
        let softmax_dispatches = plan
            .dispatches
            .iter()
            .filter(|d| d.shader == ShaderEntry::Softmax || d.reduction.is_some())
            .count();
        assert_eq!(
            softmax_dispatches, 0,
            "fused CE grad should not rebuild softmax"
        );
    }

    #[test]
    fn test_compile_transpose() {
        let mut g = Graph::new();
        let x = g.input("x", &[4, 8]);
        let t = g.transpose(x);
        g.set_outputs(vec![t]);

        let plan = compile(&g);
        assert_eq!(plan.dispatches.len(), 1);
        assert_eq!(plan.dispatches[0].shader, ShaderEntry::Transpose);
        assert_eq!(plan.dispatches[0].params[0], 4); // m
        assert_eq!(plan.dispatches[0].params[1], 8); // n
    }

    #[test]
    fn horizontal_fuse_packs_same_a_matmuls() {
        let mut dispatches = vec![
            Dispatch {
                shader: ShaderEntry::MatMul,
                workgroups: [2, 2, 1],
                input_buffers: vec![BufferRef(0), BufferRef(1)],
                output_buffer: BufferRef(2),
                params: vec![32, 32, 32, 0],
                ..Default::default()
            },
            Dispatch {
                shader: ShaderEntry::MatMul,
                workgroups: [2, 2, 1],
                input_buffers: vec![BufferRef(0), BufferRef(3)],
                output_buffer: BufferRef(4),
                params: vec![32, 32, 32, 0],
                ..Default::default()
            },
            Dispatch {
                shader: ShaderEntry::MatMul,
                workgroups: [2, 2, 1],
                input_buffers: vec![BufferRef(0), BufferRef(5)],
                output_buffer: BufferRef(6),
                params: vec![32, 32, 32, 0],
                ..Default::default()
            },
        ];
        let mut groups: Vec<std::ops::Range<usize>> = Vec::new();
        groups.push(0..3);
        fuse_horizontal_matmuls(&mut dispatches, &mut groups);
        assert_eq!(dispatches.len(), 1);
        assert_eq!(dispatches[0].horizontal_batch, 3);
        assert_eq!(dispatches[0].workgroups[2], 3);
        assert_eq!(dispatches[0].input_buffers.len(), 4);
        assert_eq!(dispatches[0].extra_outputs.len(), 2);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0], 0..1);
    }

    fn mm_dispatch(a: u32, b: u32, c: u32, wgz: u32, n: u32) -> Dispatch {
        Dispatch {
            shader: ShaderEntry::MatMul,
            workgroups: [2, 2, wgz],
            input_buffers: vec![BufferRef(a), BufferRef(b)],
            output_buffer: BufferRef(c),
            params: vec![32, 32, n, 0],
            ..Default::default()
        }
    }

    #[test]
    fn horizontal_fuse_skips_mismatched_siblings() {
        let cases: &[(&str, Vec<Dispatch>)] = &[
            (
                "different A",
                vec![mm_dispatch(0, 1, 2, 1, 32), mm_dispatch(7, 3, 4, 1, 32)],
            ),
            (
                "different N",
                vec![mm_dispatch(0, 1, 2, 1, 32), mm_dispatch(0, 3, 4, 1, 64)],
            ),
            (
                "tall-split uses Z",
                vec![mm_dispatch(0, 1, 2, 2, 32), mm_dispatch(0, 3, 4, 2, 32)],
            ),
            (
                "fused add has three inputs",
                vec![
                    Dispatch {
                        shader: ShaderEntry::FusedMatMulAdd,
                        workgroups: [2, 2, 1],
                        input_buffers: vec![BufferRef(0), BufferRef(1), BufferRef(7)],
                        output_buffer: BufferRef(2),
                        params: vec![32, 32, 32, 0],
                        ..Default::default()
                    },
                    Dispatch {
                        shader: ShaderEntry::FusedMatMulAdd,
                        workgroups: [2, 2, 1],
                        input_buffers: vec![BufferRef(0), BufferRef(3), BufferRef(8)],
                        output_buffer: BufferRef(4),
                        params: vec![32, 32, 32, 0],
                        ..Default::default()
                    },
                ],
            ),
        ];
        for &(label, ref dispatches) in cases {
            let mut dispatches = dispatches.clone();
            let n = dispatches.len();
            let mut groups: Vec<std::ops::Range<usize>> = Vec::new();
            groups.push(0..n);
            fuse_horizontal_matmuls(&mut dispatches, &mut groups);
            assert_eq!(dispatches.len(), n, "{label} should not pack");
            assert!(
                dispatches.iter().all(|d| d.horizontal_batch < 2),
                "{label} should leave horizontal_batch unset"
            );
        }
    }

    #[test]
    fn test_compile_matmul_workgroups() {
        let mut g = Graph::new();
        let a = g.input("a", &[33, 64]);
        let b = g.input("b", &[64, 17]);
        let y = g.matmul(a, b);
        g.set_outputs(vec![y]);

        let plan = compile(&g);
        let d = &plan.dispatches[0];
        // workgroups = [ceil(N/64), ceil(M/64), 1] = [1, 1, 1] (4×4 register-tiled)
        assert_eq!(d.workgroups, [1, 1, 1]);
        assert_eq!(d.params, vec![33, 64, 17, 0]);
    }

    #[test]
    fn tall_matmul_splits_portable_dispatch_limit_across_z() {
        let mut graph = Graph::new();
        let rows = 65_536 * 64;
        let a = graph.input("a", &[rows, 1]);
        let b = graph.input("b", &[1, 1]);
        let output = graph.matmul(a, b);
        graph.set_outputs(vec![output]);

        let plan = compile(&graph);
        let dispatch = &plan.dispatches[0];
        assert_eq!(dispatch.shader, ShaderEntry::MatMul);
        assert_eq!(dispatch.workgroups, [1, 32_768, 2]);
        assert!(
            dispatch
                .workgroups
                .iter()
                .all(|&count| count <= MAX_COMPUTE_WORKGROUPS_PER_DIMENSION)
        );
    }

    #[test]
    fn rmsnorm_prologue_requires_coop_and_declares_factor_reads() {
        let mut g = Graph::new();
        let x = g.input("x", &[32, 64]);
        let w_norm = g.parameter("w_norm", &[64]);
        let normalized = g.rms_norm(x, w_norm, 1e-5);
        let projection = g.parameter("projection", &[64, 64]);
        let output = g.matmul(normalized, projection);
        g.set_outputs(vec![output]);

        let mut scalar_plan = compile(&g);
        fuse_rmsnorm_prologues(&mut scalar_plan);
        assert!(
            scalar_plan
                .dispatches
                .iter()
                .any(|dispatch| dispatch.shader == ShaderEntry::RmsNorm)
        );
        assert!(
            scalar_plan
                .dispatches
                .iter()
                .all(|dispatch| dispatch.matmul_prologue.is_none())
        );

        let mut coop_plan = compile(&g);
        let matmul_index = coop_plan
            .dispatches
            .iter()
            .position(|dispatch| dispatch.shader == ShaderEntry::MatMul)
            .expect("matmul dispatch");
        coop_plan.dispatches[matmul_index].use_coop = true;
        fuse_rmsnorm_prologues(&mut coop_plan);

        let rsqrt = coop_plan
            .dispatches
            .iter()
            .find(|dispatch| dispatch.shader == ShaderEntry::RmsNormRsqrt)
            .expect("RmsNorm rsqrt dispatch");
        assert_eq!(rsqrt.params[2], 1e-5f32.to_bits());
        let matmul = &coop_plan.dispatches[matmul_index];
        let prologue = matmul.matmul_prologue.as_ref().expect("matmul prologue");
        assert_eq!(prologue.factors.len(), 2);
        for &(factor, _) in &prologue.factors {
            assert!(matmul.input_buffers.contains(&factor));
        }
    }

    #[test]
    fn conv1d_emulation_dispatches_flat_spatial_tiles() {
        // Whisper represents its temporal convolutions as H×1 Conv2d. The
        // spatial workgroup axis tiles H*W, rather than tiling W once per H.
        let mut g = Graph::new();
        let x = g.parameter("x", &[80 * 3000]);
        let w = g.parameter("w", &[512 * 80 * 3]);
        let y = g.conv2d_hw(x, w, 1, 80, 3000, 1, 512, 3, 1, 1, 1, 0);
        let loss = g.sum_all(y);
        g.set_outputs(vec![loss]);

        let plan = compile(&g);
        let forward = plan
            .dispatches
            .iter()
            .find(|dispatch| dispatch.shader == ShaderEntry::Conv2dGemm)
            .unwrap();
        assert_eq!(forward.workgroups, [3000u32.div_ceil(64), 8, 1]);

        let differentiated = crate::autodiff::differentiate(&g);
        let training = compile(&differentiated);
        let grad_input = training
            .dispatches
            .iter()
            .find(|dispatch| dispatch.shader == ShaderEntry::Conv2dGradInputGemm)
            .unwrap();
        assert_eq!(grad_input.workgroups, [3000u32.div_ceil(64), 2, 1]);
    }

    #[test]
    fn test_compile_loss_buffer() {
        let mut g = Graph::new();
        let x = g.input("x", &[4, 8]);
        let loss = g.mean_all(x);
        g.set_outputs(vec![loss]);

        let plan = compile(&g);
        assert!(plan.loss_buffer.is_some());
    }

    #[test]
    fn test_compile_param_grad_pairs() {
        let mut g = Graph::new();
        let x = g.input("x", &[4, 3]);
        let w = g.parameter("w", &[3, 2]);
        let y = g.matmul(x, w);
        let loss = g.mean_all(y);
        g.set_outputs(vec![loss]);

        let diff = crate::autodiff::differentiate(&g);
        let plan = compile(&diff);
        assert_eq!(plan.param_grad_pairs.len(), 1);
        // param buffer and grad buffer should be different
        assert_ne!(plan.param_grad_pairs[0].0, plan.param_grad_pairs[0].1);
    }

    #[test]
    fn frozen_parameter_is_excluded_from_param_grad_pairs() {
        let mut g = Graph::new();
        let trained = g.parameter("trained", &[8]);
        let frozen = g.parameter("frozen", &[8]);
        let frozen = g.stop_gradient(frozen);
        let sum = g.add(trained, frozen);
        let loss = g.mean_all(sum);
        g.set_outputs(vec![loss]);

        let diff = crate::autodiff::differentiate(&g);
        let plan = compile(&diff);
        assert_eq!(plan.param_buffers.len(), 2);
        assert_eq!(plan.param_grad_pairs.len(), 1);
        assert_eq!(plan.param_grad_pairs[0].0, plan.param_buffers[0].1);
    }

    #[test]
    fn test_compile_nop_skipped() {
        use crate::graph::{Op, TensorType};
        let mut g = Graph::new();
        let x = g.input("x", &[4, 8]);
        let _nop = g.add_raw_node(Op::Nop, vec![], TensorType::f32(vec![1]));
        let r = g.relu(x);
        g.set_outputs(vec![r]);

        let plan = compile(&g);
        // Nop should produce no dispatch
        assert_eq!(plan.dispatches.len(), 1);
        assert_eq!(plan.dispatches[0].shader, ShaderEntry::Relu);
    }

    #[test]
    fn test_compile_matmul_bias_relu_unfused() {
        let mut g = Graph::new();
        let x = g.input("x", &[4, 8]);
        let w = g.parameter("w", &[8, 4]);
        let b = g.parameter("b", &[4]);
        let mm = g.matmul(x, w);
        let ba = g.bias_add(mm, b);
        let h = g.relu(ba);
        g.set_outputs(vec![h]);

        let opt = crate::optimize::optimize(&g);
        let plan = compile(&opt);
        // With cooperative matrix, matmul+bias_add+relu are separate dispatches
        assert_eq!(plan.dispatches.len(), 3);
        assert_eq!(plan.dispatches[0].shader, ShaderEntry::MatMul);
        assert_eq!(plan.dispatches[1].shader, ShaderEntry::BiasAdd);
        assert_eq!(plan.dispatches[2].shader, ShaderEntry::Relu);
    }

    #[test]
    fn weighted_matmul_keeps_pointwise_epilogue_separate() {
        let mut g = Graph::new();
        let x = g.input("x", &[4, 32]);
        let w = g.parameter_q4("w", &[32, 64]);
        let mm = g.matmul(x, w);
        let output = g.sigmoid(mm);
        g.set_outputs(vec![output]);

        let plan = compile(&g);
        assert_eq!(plan.dispatches.len(), 2);
        assert_eq!(plan.dispatches[0].weight_format, WeightFormat::Q4);
        assert!(plan.dispatches[0].matmul_epilogue.is_none());
        assert!(plan.dispatches[0].epilogue.is_empty());
        assert_eq!(plan.dispatches[1].shader, ShaderEntry::Sigmoid);
    }

    #[test]
    fn test_shader_entry_mappings() {
        // Verify all shader entries have valid group and entry_point
        let entries = [
            ShaderEntry::MatMul,
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
            ShaderEntry::AdamUpdate,
            ShaderEntry::ScatterAdd,
            ShaderEntry::ScatterAddAtomic,
            ShaderEntry::SumAll,
            ShaderEntry::MeanAll,
            ShaderEntry::SumRows,
            ShaderEntry::Softmax,
            ShaderEntry::CrossEntropyLoss,
            ShaderEntry::BceLoss,
            ShaderEntry::Transpose,
            ShaderEntry::SwiGLUGradGate,
            ShaderEntry::SwiGLUGradUp,
            ShaderEntry::SiluGrad,
            ShaderEntry::RmsNormGradW,
            ShaderEntry::RmsNormGradX,
            ShaderEntry::LayerNormGradWB,
            ShaderEntry::LayerNormGradX,
        ];
        for entry in &entries {
            let _group = entry.shader_group();
            let ep = entry.entry_point();
            assert!(!ep.is_empty());
        }
    }

    /// Verify that the EPT/TPQ/BQ values computed in the dispatch functions
    /// match those used by the codegen shader generators. A mismatch means
    /// the compile-time workgroup counts won't match the shader's tile sizes,
    /// producing wrong results for attention kernels.
    #[test]
    fn test_attention_dispatch_matches_codegen_ept() {
        // Dispatch and the corresponding codegen path must agree on
        // EPT/TPQ/BQ. Forward and backward may use different caps.
        let graph = Graph::new();
        let compiler = Compiler::new_with_options(
            &graph,
            CompileOptions::default(),
            crate::codegen::CoopCaps::default(),
            false,
        );
        for hd_log2 in 1..=8 {
            let hd: u32 = 1 << hd_log2;
            let fwd_ept = hd.min(TuningKnobs::default().flash_ept_cap);
            let fwd_tpq = hd / fwd_ept;
            let fwd_bq: u32 = (256 / fwd_tpq).max(1);
            let grad_q_ept = hd.min(TuningKnobs::default().flash_grad_q_ept_cap);
            let grad_q_tpq = hd / grad_q_ept;
            let grad_q_bq: u32 = (256 / grad_q_tpq).max(1);
            let grad_kv_ept = hd.min(TuningKnobs::default().flash_grad_kv_ept_cap);
            let grad_kv_tpq = hd / grad_kv_ept;
            let grad_kv_bq: u32 = (256 / grad_kv_tpq).max(1);

            let (fwd_entry, fwd_wg) = compiler.attention_dispatch(256, hd, 1);
            let (grad_q_entry, grad_q_wg) =
                Compiler::attention_dispatch_bwd(256, hd, 1, grad_q_ept);
            let (grad_kv_entry, grad_kv_wg) =
                Compiler::attention_dispatch_bwd(256, hd, 1, grad_kv_ept);
            if fwd_bq >= 2 {
                assert_eq!(fwd_entry, ShaderEntry::FlashAttention);
                assert_eq!(fwd_wg[0], 256u32.div_ceil(fwd_bq));
            }
            if grad_q_bq >= 2 {
                assert_eq!(grad_q_entry, ShaderEntry::FlashAttention);
                assert_eq!(grad_q_wg[0], 256u32.div_ceil(grad_q_bq));
            }
            if grad_kv_bq >= 2 {
                assert_eq!(grad_kv_entry, ShaderEntry::FlashAttention);
                assert_eq!(grad_kv_wg[0], 256u32.div_ceil(grad_kv_bq));
            }
        }
    }

    /// A 16×16 f16-only capability set is what several NVIDIA Vulkan
    /// drivers expose (and is also used by RDNA3). Keep this plan-time path
    /// covered even when CI has no matching physical adapter.
    #[test]
    fn f16_only_target_keeps_attention_backward_safe_by_default() {
        let mut g = Graph::new();
        let q = g.parameter("q", &[256, 64]);
        let k = g.parameter("k", &[256, 64]);
        let v = g.parameter("v", &[256, 64]);
        let attention = g.causal_attention(q, k, v, 1, 1, 64);
        let loss = g.sum_all(attention);
        g.set_outputs(vec![loss]);
        let differentiated = crate::autodiff::differentiate(&g);

        let f16_only = crate::codegen::CoopCaps {
            f16_tile: 16,
            f32_tile: 0,
        };
        let safe =
            compile_with_caps_policy(&differentiated, &CompileOptions::default(), f16_only, false);
        let safe_entries: Vec<_> = safe
            .dispatches
            .iter()
            .map(|dispatch| dispatch.shader.clone())
            .collect();
        assert!(safe_entries.contains(&ShaderEntry::FlashAttentionCoop));
        assert!(!safe_entries.contains(&ShaderEntry::FlashGradQCoop));
        assert!(!safe_entries.contains(&ShaderEntry::FlashGradKVCoop));

        let experimental =
            compile_with_caps_policy(&differentiated, &CompileOptions::default(), f16_only, true);
        let experimental_entries: Vec<_> = experimental
            .dispatches
            .iter()
            .map(|dispatch| dispatch.shader.clone())
            .collect();
        assert!(experimental_entries.contains(&ShaderEntry::FlashGradQCoop));
        assert!(experimental_entries.contains(&ShaderEntry::FlashGradKVCoop));

        let scalar = compile_with_caps_policy(
            &differentiated,
            &CompileOptions::default(),
            crate::codegen::CoopCaps::default(),
            false,
        );
        assert!(scalar.dispatches.iter().all(|dispatch| !matches!(
            &dispatch.shader,
            ShaderEntry::FlashAttentionCoop
                | ShaderEntry::FlashGradQCoop
                | ShaderEntry::FlashGradKVCoop
        )));
    }

    #[test]
    fn short_cross_attention_selects_coop_grad_kv() {
        // The scalar flash heuristic uses BQ=128 at head_dim=64, so a
        // SmolVLA-shaped 16-position KV span selects MultiHeadAttn. The
        // cooperative GradKV kernel has its own BKV=16 geometry and must not
        // inherit that unrelated scalar decision.
        let mut g = Graph::new();
        let q = g.parameter("q", &[50, 15 * 64]);
        let k = g.parameter("k", &[16, 5 * 64]);
        let v = g.parameter("v", &[16, 5 * 64]);
        let attention = g.multi_head_attn(q, k, v, 15, 5, 64, true);
        let loss = g.sum_all(attention);
        g.set_outputs(vec![loss]);
        let differentiated = crate::autodiff::differentiate(&g);

        let plan = compile_with_caps_policy(
            &differentiated,
            &CompileOptions::default(),
            crate::codegen::CoopCaps {
                f16_tile: 16,
                f32_tile: 0,
            },
            true,
        );

        assert!(
            plan.dispatches
                .iter()
                .any(|dispatch| dispatch.shader == ShaderEntry::FlashGradKVCoop)
        );
        assert!(
            plan.dispatches
                .iter()
                .all(|dispatch| { dispatch.shader != ShaderEntry::MultiHeadAttnGradKV })
        );
    }

    #[test]
    fn profile_families_cover_representative_kernel_shapes() {
        assert_eq!(ShaderEntry::MatMul.profile_family(), "matrix");
        assert_eq!(ShaderEntry::FlashAttention.profile_family(), "attention");
        assert_eq!(
            ShaderEntry::Conv2dGemmCoopGen(3, 3, 1).profile_family(),
            "convolution_spatial"
        );
        assert_eq!(
            ShaderEntry::LayerNormGradX.profile_family(),
            "normalization_reduction"
        );
        assert_eq!(ShaderEntry::SwiGLU.profile_family(), "pointwise");
        assert_eq!(ShaderEntry::Transpose.profile_family(), "data_movement");
        assert_eq!(ShaderEntry::AdamUpdate.profile_family(), "optimizer");

        let mut graph = Graph::new();
        let input = graph.input("input", &[4, 8]);
        let output = graph.softmax(input);
        graph.set_outputs(vec![output]);
        let plan = compile(&graph);
        assert!(
            plan.dispatches
                .iter()
                .any(|dispatch| dispatch.reduction.is_some())
        );
        assert!(
            plan.dispatches
                .iter()
                .filter(|dispatch| dispatch.reduction.is_some())
                .all(|dispatch| dispatch.profile_family() == "normalization_reduction")
        );
    }

    #[test]
    fn group_norm_only_splits_when_there_is_parallel_work_to_gain() {
        let make_plan = |spatial: u32| {
            let mut graph = Graph::new();
            let channels = 8;
            let groups = 8;
            let input = graph.input("input", &[(channels * spatial) as usize]);
            let weight = graph.parameter("weight", &[channels as usize]);
            let bias = graph.parameter("bias", &[channels as usize]);
            let output =
                graph.group_norm(input, weight, bias, 1, channels, spatial, groups, 1.0e-5);
            graph.set_outputs(vec![output]);
            compile(&graph)
        };

        let small = make_plan(256);
        assert_eq!(small.dispatches.len(), 1);
        assert_eq!(small.dispatches[0].shader, ShaderEntry::GroupNorm);

        let large = make_plan(8202);
        assert_eq!(large.dispatches.len(), 2);
        assert!(large.dispatches[0].reduction.is_some());
        assert_eq!(large.dispatches[1].shader, ShaderEntry::GroupNormApply);
        let chunks = large.dispatches[1].params[5];
        assert_eq!(chunks, 3);
        assert_eq!(8202 % chunks, 0);
        assert_eq!(large.dispatches[0].params[..2], [8 * chunks, 8202 / chunks]);
    }
}
