use crate::codegen::ShaderGroup;
use crate::graph::{DType, Graph, Node, NodeId, Op, PairwiseGradKind};
use crate::schedule::{PointwiseDAG, Pw, ReductionEpilogue, ReductionKernel};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

mod softplus;
mod split_k;

/// Host layout of the cached block-attention WGSL uniform. Dispatch encoding,
/// runtime binding and tuning all use this layout rather than indexing words.
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
pub(crate) struct CachedBlockAttentionParams {
    pub window_size: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub block_len: u32,
    pub max_seq: u32,
    pub splits: u32,
    pub _pad: u32,
}

impl CachedBlockAttentionParams {
    pub fn from_words(words: &[u32]) -> Option<Self> {
        bytemuck::try_from_bytes(bytemuck::cast_slice(words))
            .ok()
            .copied()
    }

    pub fn to_words(self) -> Vec<u32> {
        bytemuck::cast_slice(std::slice::from_ref(&self)).to_vec()
    }
}

/// Weight storage format for matmul B operands.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WeightFormat {
    #[default]
    F32,
    F16,
    Q4,
    Q8,
    /// GGML Q4_K: 256-element superblocks with 6-bit sub-block scales.
    /// Load-only; see [`crate::graph::DType::Q4K`].
    Q4K,
    /// GGML Q6_K: 256-element superblocks with signed 8-bit sub-block
    /// scales. Load-only; see [`crate::graph::DType::Q6K`].
    Q6K,
    /// GGML Q5_K: 256-element superblocks, nibble plus a high bit.
    /// Load-only; see [`crate::graph::DType::Q5K`].
    Q5K,
    /// GGML Q3_K: 256-element superblocks, 2-bit quants with an inverted
    /// high bit. Load-only; see [`crate::graph::DType::Q3K`].
    Q3K,
    /// GGML Q4_0: 32-element blocks of an f16 scale and 16 nibble bytes,
    /// symmetric with no minimum. Load-only; see [`crate::graph::DType::Q40`].
    ///
    /// Distinct from [`WeightFormat::Q4`], which is Meganeura's own
    /// asymmetric packing at half a bit per weight more.
    Q40,
}

impl WeightFormat {
    /// The stored dtype, excluding the separately packed Q4/Q8 upload paths.
    pub fn dtype(self) -> Option<crate::graph::DType> {
        use crate::graph::DType;
        match self {
            Self::Q4K => Some(DType::Q4K),
            Self::Q6K => Some(DType::Q6K),
            Self::Q5K => Some(DType::Q5K),
            Self::Q3K => Some(DType::Q3K),
            Self::Q40 => Some(DType::Q40),
            Self::F32 => Some(DType::F32),
            Self::F16 => Some(DType::F16),
            Self::Q4 | Self::Q8 => None,
        }
    }

    pub fn is_quantized(self) -> bool {
        matches!(
            self,
            Self::Q4 | Self::Q8 | Self::Q40 | Self::Q4K | Self::Q6K | Self::Q5K | Self::Q3K
        )
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
            DType::Q40 => Self::Q40,
            DType::Q4K => Self::Q4K,
            DType::Q6K => Self::Q6K,
            DType::Q5K => Self::Q5K,
            DType::Q3K => Self::Q3K,
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TuningKnobs {
    /// Elements-per-thread cap for flash-attention forward codegen.
    pub flash_ept_cap: u32,
    /// Forward attention workgroup, shared tile and lane layout.
    #[serde(default)]
    pub flash: crate::codegen::FlashAttentionShape,
    /// EPT cap for the flash dQ backward kernel.
    pub flash_grad_q_ept_cap: u32,
    /// EPT cap for the fused flash dK/dV backward kernel.
    pub flash_grad_kv_ept_cap: u32,
    /// K staging depth of the scalar tiled matmul: 8 | 16 | 32.
    pub matmul_k_stage: u32,
    /// Stagger scalar-matmul B loads across columns instead of routing a
    /// thread through consecutive ones.
    pub matmul_interleave_columns: bool,
}

impl Default for TuningKnobs {
    /// Pure per-platform defaults. Apple Silicon benefits from the extra
    /// parallelism of smaller EPT; 32 keeps register count below the
    /// spilling cliff on Ampere/Blackwell. `MEGANEURA_FLASH_*_EPT_CAP`
    /// overrides are applied only by [`TuningKnobs::from_env`] (in
    /// `crate::config`) — the library itself never reads the environment.
    /// Matmul K stage and column interleave stay at these defaults. Measured
    /// extraction chooses other schedules; they are not environment switches.
    fn default() -> Self {
        let apple = cfg!(all(target_vendor = "apple", target_arch = "aarch64"));
        let fwd = if apple { 16 } else { 32 };
        Self {
            flash_ept_cap: fwd,
            flash: crate::codegen::FlashAttentionShape::default(),
            flash_grad_q_ept_cap: fwd,
            flash_grad_kv_ept_cap: if apple { 8 } else { 32 },
            matmul_k_stage: 32,
            matmul_interleave_columns: false,
        }
    }
}

/// Options controlling graph → execution-plan compilation.
///
/// Wire these to env vars or CLI flags in your own harness if you want —
/// the library itself takes only this typed struct.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CompileOptions {
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
    /// Starting workgroup geometry and cross-lane reduction for the K-split
    /// GEMV family, overriding each group's own default.
    ///
    /// This is a starting point, not a decision: `Session::tune_with`
    /// challenges whichever shape is in force and installs what measures
    /// faster. Set it to pin a shape for a benchmark or a reproduction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gemv_shape: Option<crate::codegen::GemvShape>,
    /// Cached-attention implementation: 1 is unsplit, 2..=16 use partials.
    /// None retains the ordinary lowering; measured construction searches this.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cached_attention_splits: Option<u32>,
    /// Opt-in two-pass scalar convolution weight gradients. Changes reduction
    /// order; qualify values/gradients and whole-step timing before deployment.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conv_weight_splits: Option<ConvWeightSplits>,
    /// Quantize the activation row to Q8_1 inside the K-split GEMV and do
    /// the inner product with integer dot products, for the weight formats
    /// that have an int-dot kernel (GGML Q4_0 and Meganeura Q8).
    ///
    /// On by default: a model that ships quantized weights is being decoded
    /// quantized, so its activations join them (llama.cpp's Q8_1 policy),
    /// and no caller wiring should be needed to get it. This is the one
    /// switch here that changes the numbers — the activation loses
    /// precision on top of the weight — so measurement may reshape the
    /// selected kernel but never flips it. Set it false to keep f32
    /// activations, e.g. to pin packed-weight decode fidelity in a test.
    pub quantized_activations: bool,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            fuse_dispatches: true,
            knobs: TuningKnobs::default(),
            flash_forward_coop: true,
            flash_backward_coop: false,
            gemv_shape: None,
            cached_attention_splits: None,
            conv_weight_splits: None,
            quantized_activations: true,
        }
    }
}

/// Bounded split-K lowering for convolution gradients with too few workgroups.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct ConvWeightSplits {
    /// Only split dispatches with fewer than this many unsplit workgroups.
    pub workgroup_threshold: u32,
    /// Maximum reduction positions per partition; a positive multiple of 16.
    pub reduction_chunk: u32,
    /// Total logical partial-buffer budget, before aliasing. Selections that
    /// would exceed the remaining budget keep their original implementation.
    pub max_partial_bytes: usize,
}

impl CompileOptions {
    fn gemv_kernel(&self, group: ShaderGroup, format: WeightFormat) -> Kernel {
        Kernel::Gemv {
            shape: self.gemv_shape.map_or_else(
                || crate::codegen::GemvShape::initial(group),
                |shape| shape.for_group(group),
            ),
            integer_dot: self.quantized_activations
                && matches!(group, ShaderGroup::MatMulGemv | ShaderGroup::MatMulGemvAdd)
                && matches!(
                    format,
                    WeightFormat::Q40
                        | WeightFormat::Q8
                        | WeightFormat::Q4K
                        | WeightFormat::Q5K
                        | WeightFormat::Q6K
                        | WeightFormat::Q3K
                ),
        }
    }
}

/// Identifies which shader and entry point to use.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ShaderEntry {
    #[default]
    MatMul,
    /// A generated pointwise or reduction kernel. The dispatch's `kernel`
    /// says what it computes and determines its binding layout.
    Generated,
    MatMulAT,
    MatMulBT,
    BlockMatMul,
    BlockMatMulAT,
    BlockMatMulBT,
    /// Batch-major products: `[B, M, K] × [B, K, N] → [B, M, N]`.
    BatchMatMul,
    BatchMatMulAT,
    BatchMatMulBT,
    /// M=1 GEMV specialization of MatMul. Selected when `C = A × B` has
    /// a single row on the output side (LM decode path).
    MatMulGemv,
    /// M=1 GEMV with fused residual add: `C = A × B + D`.
    /// Selected for FusedMatMulAdd when M=1.
    MatMulGemvAdd,
    /// M=1 MatMulBT specialization (`B` stored `[N,K]`). K-split with
    /// naturally coalesced vec4 reads along the contiguous K axis.
    MatMulGemvBT,
    /// M=1 transposed-B GEMV with a fused addend.
    MatMulGemvBTAdd,
    FusedMatMulAdd,
    FusedMatMulATAdd,
    FusedMatMulBTAdd,
    SgdUpdate,
    AdamUpdate,
    ScatterAdd,
    ScatterAddAtomic,
    SumAll,
    MeanAll,
    CrossEntropyLoss,
    BceLoss,
    Transpose,
    /// RmsNorm with the consumer's residual add folded in. Selected by
    /// `fuse_rmsnorm_into_add` for a norm whose only reader is an `Add`;
    /// computes the same values with one dispatch instead of two.
    RmsNormAdd,
    Embedding,
    ToF16,
    RoPE,
    RoPEGrad,
    LayerNorm,
    MultiHeadAttn,
    /// Flash Attention 2 forward: BQ>1 multi-query tiling.
    FlashAttention,
    /// Cooperative-matrix flash attention forward (Phase 1: coop QK^T,
    /// scalar softmax + PV). Enabled on compatible devices unless
    /// `MEGANEURA_FLASH_FWD_COOP=0`.
    /// BQ=BKV=16, dispatched as `[ceil(q_seq/16), num_heads, 1]`.
    FlashAttentionCoop,
    FlashAttentionCoopF32,
    /// F16-input cooperative products for score=Q·K^T and dp=dO·V^T,
    /// with scalar f32 dQ+=ds·K accumulation. Opt-in via
    /// `MEGANEURA_FLASH_BWD_COOP=1`. BQ=BKV=16,
    /// dispatched as `[ceil(q_seq/16), num_heads, 1]`.
    FlashGradQCoopF16,
    /// F16-input cooperative flash backward dK + dV kernel (fused).
    /// Two coop matmuls per Q-tile: score = K·Q^T, dp = V·dO^T.
    /// dV/dK accumulate per-thread. Dispatched as
    /// `[ceil(dispatch_kv/16), num_kv_heads, 1]`.
    FlashGradKVCoopF16,
    /// F32 cooperative dK/dV, 16 keys per tile, specialized for 64-wide heads.
    FlashGradKVCoopF32,
    /// F32 cooperative dQ, 16 queries per tile, specialized for 64-wide heads.
    FlashGradQCoopF32,
    MultiHeadAttnGradQ,
    FlashGradQ,
    MultiHeadAttnGradKV,
    FlashGradKV,
    SwiGLUGradGate,
    SwiGLUGradUp,
    SiluGrad,
    SwiGLUConcat,
    SwiGLUConcatGrad,
    GeGLUConcat,
    GeGLUConcatGrad,
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
    /// Per-slice (sum, M2) of a group, first pass of chunked GroupNorm.
    GroupNormStats,
    GroupNormApply,
    GroupNormGradInput,
    GroupNormGradWeightBias,
    /// Per-(batch, group) mean and inverse deviation shared by both
    /// GroupNorm backward kernels.
    GroupNormGradStats,
    Concat,
    SplitA,
    /// Axis permutation of a tensor of rank at most 4.
    Permute,
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
    Conv2dGemm,
    Conv2dGemmSmall,
    /// 16×16 register tile for convolutions whose 32-wide grid is still tiny.
    Conv2dGemm16,
    /// Generated conv2d forward coop kernel specialized for (kernel_h, kernel_w, stride).
    Conv2dGemmCoopGen(u32, u32, u32),
    Conv2dGradInputGemm,
    Conv2dGradInputGemmSmall,
    Conv2dGradInputGemm16,
    /// Generated conv2d grad_input coop kernel specialized for (kernel_h, kernel_w, stride).
    Conv2dGradInputGemmCoopGen(u32, u32, u32),
    Conv2dGradWeightGemm,
    Conv2dGradWeightGemmSmall,
    Conv2dGradWeightGemm16,
    /// Split reduction tiles into `[split, Co, Ci*Kh*Kw]` partials for SumRows.
    Conv2dGradWeightGemmSplit,
    Conv2dGradWeightGemmSplitSmall,
    Conv2dGradWeightGemmSplit16,
    CacheWrite,
    CacheWritePrefix,
    CachedAttention,
    /// Attention with an additive per-head bias, one workgroup per query
    /// row and head: full, causal or over a cache prefix.
    BiasedAttention,
    CachedQueryAttention,
    CachedBlockAttention,
    /// Flash-decoding split-K partial: per-slice online-softmax partials.
    CachedBlockAttentionSplit,
    /// Merges the split-K partials into the attention output.
    CachedBlockAttentionCombine,
    ChunkedRelativeAttention,
    PrefixLast,
    RoPEDynamic,
    RoPEDynamicFactors,
    RoPEPositions,
    MaxPool2d,
    MaxPool2dGrad,
    GlobalAvgPool,
    GlobalAvgPoolGrad,
    PairwiseGrad,
    WinogradInputTransform,
    WinogradOutputTransform,
    WinogradBatchedMatMul,
    WinogradWeightTransform,
    /// Squared-gradient partial sums, one slot per workgroup, then a
    /// single-workgroup total of those partials. Pair with `GradClipScale`.
    GradClipNormSq,
    /// In-place scale a gradient buffer by `min(1, max_norm / norm)`,
    /// where `norm = sqrt(acc[0])` from the accumulator
    /// produced by `GradClipNormSq`.
    GradClipScale,
    /// Per-parameter adaptive gradient clipping. One workgroup measures the
    /// parameter and gradient norms and scales the gradient in place.
    AdaptiveGradClip,
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
            ShaderEntry::Generated => "pointwise",
            ShaderEntry::MatMul
            | ShaderEntry::MatMulAT
            | ShaderEntry::MatMulBT
            | ShaderEntry::BlockMatMul
            | ShaderEntry::BlockMatMulAT
            | ShaderEntry::BlockMatMulBT
            | ShaderEntry::BatchMatMul
            | ShaderEntry::BatchMatMulAT
            | ShaderEntry::BatchMatMulBT
            | ShaderEntry::MatMulGemv
            | ShaderEntry::MatMulGemvAdd
            | ShaderEntry::MatMulGemvBT
            | ShaderEntry::MatMulGemvBTAdd
            | ShaderEntry::FusedMatMulAdd
            | ShaderEntry::FusedMatMulATAdd
            | ShaderEntry::FusedMatMulBTAdd => "matrix",

            ShaderEntry::MultiHeadAttn
            | ShaderEntry::FlashAttention
            | ShaderEntry::FlashAttentionCoop
            | ShaderEntry::FlashAttentionCoopF32
            | ShaderEntry::FlashGradQCoopF16
            | ShaderEntry::FlashGradKVCoopF16
            | ShaderEntry::FlashGradKVCoopF32
            | ShaderEntry::FlashGradQCoopF32
            | ShaderEntry::MultiHeadAttnGradQ
            | ShaderEntry::FlashGradQ
            | ShaderEntry::MultiHeadAttnGradKV
            | ShaderEntry::FlashGradKV
            | ShaderEntry::CachedAttention
            | ShaderEntry::BiasedAttention
            | ShaderEntry::CachedQueryAttention
            | ShaderEntry::CachedBlockAttention
            | ShaderEntry::CachedBlockAttentionSplit
            | ShaderEntry::CachedBlockAttentionCombine
            | ShaderEntry::ChunkedRelativeAttention => "attention",

            ShaderEntry::Conv2dDw
            | ShaderEntry::Conv2dGemm
            | ShaderEntry::Conv2dGemmSmall
            | ShaderEntry::Conv2dGemm16
            | ShaderEntry::Conv2dGemmCoopGen(..)
            | ShaderEntry::Conv2dGradInputGemm
            | ShaderEntry::Conv2dGradInputGemmSmall
            | ShaderEntry::Conv2dGradInputGemm16
            | ShaderEntry::Conv2dGradInputGemmCoopGen(..)
            | ShaderEntry::Conv2dGradWeightGemm
            | ShaderEntry::Conv2dGradWeightGemmSmall
            | ShaderEntry::Conv2dGradWeightGemm16
            | ShaderEntry::Conv2dGradWeightGemmSplit
            | ShaderEntry::Conv2dGradWeightGemmSplitSmall
            | ShaderEntry::Conv2dGradWeightGemmSplit16
            | ShaderEntry::Upsample2x
            | ShaderEntry::Upsample2xGrad
            | ShaderEntry::MaxPool2d
            | ShaderEntry::MaxPool2dGrad
            | ShaderEntry::WinogradInputTransform
            | ShaderEntry::WinogradOutputTransform
            | ShaderEntry::WinogradBatchedMatMul
            | ShaderEntry::WinogradWeightTransform => "convolution_spatial",

            ShaderEntry::SumAll
            | ShaderEntry::MeanAll
            | ShaderEntry::CrossEntropyLoss
            | ShaderEntry::BceLoss
            | ShaderEntry::RmsNormAdd
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
            | ShaderEntry::GroupNormStats
            | ShaderEntry::GroupNormApply
            | ShaderEntry::GroupNormGradInput
            | ShaderEntry::GroupNormGradWeightBias
            | ShaderEntry::GroupNormGradStats
            | ShaderEntry::GlobalAvgPool
            | ShaderEntry::GlobalAvgPoolGrad
            | ShaderEntry::PairwiseGrad => "normalization_reduction",

            ShaderEntry::SgdUpdate
            | ShaderEntry::AdamUpdate
            | ShaderEntry::GradClipNormSq
            | ShaderEntry::GradClipScale
            | ShaderEntry::AdaptiveGradClip
            | ShaderEntry::GradAccum => "optimizer",

            ShaderEntry::ScatterAdd
            | ShaderEntry::ScatterAddAtomic
            | ShaderEntry::Transpose
            | ShaderEntry::Embedding
            | ShaderEntry::ToF16
            | ShaderEntry::Concat
            | ShaderEntry::SplitA
            | ShaderEntry::SplitB
            | ShaderEntry::Permute
            | ShaderEntry::CacheWrite
            | ShaderEntry::CacheWritePrefix
            | ShaderEntry::PrefixLast => "data_movement",

            ShaderEntry::RoPE
            | ShaderEntry::RoPEGrad
            | ShaderEntry::SwiGLUGradGate
            | ShaderEntry::SwiGLUGradUp
            | ShaderEntry::SiluGrad
            | ShaderEntry::SwiGLUConcat
            | ShaderEntry::SwiGLUConcatGrad
            | ShaderEntry::GeGLUConcat
            | ShaderEntry::GeGLUConcatGrad
            | ShaderEntry::MulPerChannel
            | ShaderEntry::RoPEDynamic
            | ShaderEntry::RoPEDynamicFactors
            | ShaderEntry::RoPEPositions => "pointwise",
        }
    }

    pub const fn is_matmul(&self) -> bool {
        matches!(
            self,
            Self::MatMul
                | Self::MatMulAT
                | Self::MatMulBT
                | Self::FusedMatMulAdd
                | Self::FusedMatMulATAdd
                | Self::FusedMatMulBTAdd
        )
    }

    pub const fn is_attention(&self) -> bool {
        matches!(
            self,
            Self::FlashAttention
                | Self::MultiHeadAttn
                | Self::FlashAttentionCoop
                | Self::FlashAttentionCoopF32
        )
    }

    pub fn shader_group(&self) -> crate::codegen::ShaderGroup {
        use crate::codegen::ShaderGroup;
        match *self {
            ShaderEntry::Generated => ShaderGroup::Generated,
            ShaderEntry::MatMul => ShaderGroup::MatMul,
            ShaderEntry::MatMulAT => ShaderGroup::MatMulAT,
            ShaderEntry::MatMulBT => ShaderGroup::MatMulBT,
            ShaderEntry::BlockMatMul => ShaderGroup::BlockMatMul,
            ShaderEntry::BlockMatMulAT => ShaderGroup::BlockMatMulAT,
            ShaderEntry::BlockMatMulBT => ShaderGroup::BlockMatMulBT,
            ShaderEntry::BatchMatMul => ShaderGroup::BatchMatMul,
            ShaderEntry::BatchMatMulAT => ShaderGroup::BatchMatMulAT,
            ShaderEntry::BatchMatMulBT => ShaderGroup::BatchMatMulBT,
            ShaderEntry::MatMulGemv => ShaderGroup::MatMulGemv,
            ShaderEntry::MatMulGemvAdd => ShaderGroup::MatMulGemvAdd,
            ShaderEntry::MatMulGemvBT => ShaderGroup::MatMulGemvBT,
            ShaderEntry::MatMulGemvBTAdd => ShaderGroup::MatMulGemvBTAdd,
            ShaderEntry::FusedMatMulAdd => ShaderGroup::MatMulAdd,
            ShaderEntry::FusedMatMulATAdd => ShaderGroup::MatMulATAdd,
            ShaderEntry::FusedMatMulBTAdd => ShaderGroup::MatMulBTAdd,
            ShaderEntry::SgdUpdate => ShaderGroup::Sgd,
            ShaderEntry::AdamUpdate => ShaderGroup::Adam,
            ShaderEntry::ScatterAdd => ShaderGroup::ScatterAdd,
            ShaderEntry::ScatterAddAtomic => ShaderGroup::ScatterAddAtomic,
            ShaderEntry::SumAll | ShaderEntry::MeanAll => ShaderGroup::Reduce,
            ShaderEntry::CrossEntropyLoss => ShaderGroup::CrossEntropy,
            ShaderEntry::BceLoss => ShaderGroup::BceLoss,
            ShaderEntry::Transpose => ShaderGroup::Transpose,
            ShaderEntry::RmsNormAdd => ShaderGroup::RmsNormAdd,
            ShaderEntry::Embedding => ShaderGroup::Embedding,
            ShaderEntry::ToF16 => ShaderGroup::ToF16,
            ShaderEntry::RoPE => ShaderGroup::RoPE,
            ShaderEntry::RoPEGrad => ShaderGroup::RoPEGrad,
            ShaderEntry::LayerNorm => ShaderGroup::LayerNorm,
            ShaderEntry::MultiHeadAttn => ShaderGroup::MultiHeadAttn,
            ShaderEntry::FlashAttention => ShaderGroup::FlashAttention,
            ShaderEntry::FlashAttentionCoop => ShaderGroup::FlashAttentionCoop,
            ShaderEntry::FlashAttentionCoopF32 => ShaderGroup::FlashAttentionCoopF32,
            ShaderEntry::FlashGradQCoopF16 => ShaderGroup::FlashGradQCoopF16,
            ShaderEntry::FlashGradKVCoopF16 => ShaderGroup::FlashGradKVCoopF16,
            ShaderEntry::FlashGradKVCoopF32 => ShaderGroup::FlashGradKVCoopF32,
            ShaderEntry::FlashGradQCoopF32 => ShaderGroup::FlashGradQCoopF32,
            ShaderEntry::MultiHeadAttnGradQ => ShaderGroup::MultiHeadAttnGradQ,
            ShaderEntry::FlashGradQ => ShaderGroup::FlashGradQ,
            ShaderEntry::MultiHeadAttnGradKV => ShaderGroup::MultiHeadAttnGradKV,
            ShaderEntry::FlashGradKV => ShaderGroup::FlashGradKV,
            ShaderEntry::SwiGLUGradGate | ShaderEntry::SwiGLUGradUp | ShaderEntry::SiluGrad => {
                ShaderGroup::SwiGLUGrad
            }
            ShaderEntry::SwiGLUConcat
            | ShaderEntry::SwiGLUConcatGrad
            | ShaderEntry::GeGLUConcat
            | ShaderEntry::GeGLUConcatGrad => ShaderGroup::SwiGLUConcat,
            ShaderEntry::SumRows => ShaderGroup::SumRows,
            ShaderEntry::RmsNormGradW | ShaderEntry::RmsNormGradX => ShaderGroup::RmsNormGrad,
            ShaderEntry::RmsNormGradWRowPar => ShaderGroup::RmsNormGradWRowPar,
            ShaderEntry::LayerNormGradWB | ShaderEntry::LayerNormGradX => {
                ShaderGroup::LayerNormGrad
            }
            ShaderEntry::RmsNormRsqrt => ShaderGroup::RmsNormRsqrt,
            ShaderEntry::GroupNorm => ShaderGroup::GroupNorm,
            ShaderEntry::GroupNormSilu => ShaderGroup::GroupNormSilu,
            ShaderEntry::GroupNormApply | ShaderEntry::GroupNormStats => ShaderGroup::GroupNorm,
            ShaderEntry::GroupNormGradInput => ShaderGroup::GroupNormGrad,
            ShaderEntry::GroupNormGradWeightBias => ShaderGroup::GroupNormGrad,
            ShaderEntry::GroupNormGradStats => ShaderGroup::GroupNormGrad,
            ShaderEntry::Concat => ShaderGroup::Concat,
            ShaderEntry::SplitA | ShaderEntry::SplitB => ShaderGroup::Split,
            ShaderEntry::Permute => ShaderGroup::Permute,
            ShaderEntry::Upsample2x => ShaderGroup::Upsample,
            ShaderEntry::Upsample2xGrad => ShaderGroup::UpsampleGrad,
            ShaderEntry::Conv2dDw => ShaderGroup::Conv2dDw,
            ShaderEntry::MulPerChannel => ShaderGroup::MulPerChannel,
            ShaderEntry::Conv2dGemm => ShaderGroup::Conv2dGemm,
            ShaderEntry::Conv2dGemmCoopGen(..) => ShaderGroup::Conv2dGemmCoop,
            ShaderEntry::Conv2dGemmSmall => ShaderGroup::Conv2dGemmSmall,
            ShaderEntry::Conv2dGemm16 => ShaderGroup::Conv2dGemm16,
            ShaderEntry::Conv2dGradInputGemm => ShaderGroup::Conv2dGradInputGemm,
            ShaderEntry::Conv2dGradInputGemmSmall => ShaderGroup::Conv2dGradInputGemmSmall,
            ShaderEntry::Conv2dGradInputGemm16 => ShaderGroup::Conv2dGradInputGemm16,
            ShaderEntry::Conv2dGradInputGemmCoopGen(..) => ShaderGroup::Conv2dGradInputGemmCoop,
            ShaderEntry::Conv2dGradWeightGemm => ShaderGroup::Conv2dGradWeightGemm,
            ShaderEntry::Conv2dGradWeightGemmSmall => ShaderGroup::Conv2dGradWeightGemmSmall,
            ShaderEntry::Conv2dGradWeightGemm16 => ShaderGroup::Conv2dGradWeightGemm16,
            ShaderEntry::Conv2dGradWeightGemmSplit => ShaderGroup::Conv2dGradWeightGemmSplit,
            ShaderEntry::Conv2dGradWeightGemmSplitSmall => {
                ShaderGroup::Conv2dGradWeightGemmSplitSmall
            }
            ShaderEntry::Conv2dGradWeightGemmSplit16 => ShaderGroup::Conv2dGradWeightGemmSplit16,
            ShaderEntry::CacheWrite => ShaderGroup::CacheWrite,
            ShaderEntry::CacheWritePrefix => ShaderGroup::CacheWritePrefix,
            ShaderEntry::CachedAttention => ShaderGroup::CachedAttention,
            ShaderEntry::BiasedAttention => ShaderGroup::BiasedAttention,
            ShaderEntry::CachedQueryAttention => ShaderGroup::CachedQueryAttention,
            ShaderEntry::CachedBlockAttention => ShaderGroup::CachedBlockAttention,
            ShaderEntry::CachedBlockAttentionSplit => ShaderGroup::CachedBlockAttentionSplit,
            ShaderEntry::CachedBlockAttentionCombine => ShaderGroup::CachedBlockAttentionCombine,
            ShaderEntry::ChunkedRelativeAttention => ShaderGroup::ChunkedRelativeAttention,
            ShaderEntry::PrefixLast => ShaderGroup::PrefixLast,
            ShaderEntry::RoPEDynamic | ShaderEntry::RoPEDynamicFactors => ShaderGroup::RoPEDynamic,
            ShaderEntry::RoPEPositions => ShaderGroup::RoPEDynamic,
            ShaderEntry::MaxPool2d => ShaderGroup::MaxPool2d,
            ShaderEntry::MaxPool2dGrad => ShaderGroup::MaxPool2dGrad,
            ShaderEntry::GlobalAvgPool => ShaderGroup::GlobalAvgPool,
            ShaderEntry::GlobalAvgPoolGrad => ShaderGroup::GlobalAvgPoolGrad,
            ShaderEntry::PairwiseGrad => ShaderGroup::PairwiseGrad,
            ShaderEntry::WinogradInputTransform => ShaderGroup::WinogradInputTransform,
            ShaderEntry::WinogradOutputTransform => ShaderGroup::WinogradOutputTransform,
            ShaderEntry::WinogradBatchedMatMul => ShaderGroup::WinogradBatchedMatMul,
            ShaderEntry::WinogradWeightTransform => ShaderGroup::WinogradWeightTransform,
            ShaderEntry::GradClipNormSq => ShaderGroup::GradClipNormSq,
            ShaderEntry::GradClipScale => ShaderGroup::GradClipScale,
            ShaderEntry::AdaptiveGradClip => ShaderGroup::AdaptiveGradClip,
            ShaderEntry::GradAccum => ShaderGroup::GradAccum,
        }
    }

    pub fn entry_point(&self) -> &'static str {
        match *self {
            ShaderEntry::Generated => crate::schedule::POINTWISE_ENTRY,
            ShaderEntry::BlockMatMul
            | ShaderEntry::BlockMatMulAT
            | ShaderEntry::BlockMatMulBT
            | ShaderEntry::BatchMatMul
            | ShaderEntry::BatchMatMulAT
            | ShaderEntry::BatchMatMulBT => "main",
            ShaderEntry::MatMul
            | ShaderEntry::MatMulAT
            | ShaderEntry::MatMulBT
            | ShaderEntry::MatMulGemv
            | ShaderEntry::MatMulGemvAdd
            | ShaderEntry::MatMulGemvBT
            | ShaderEntry::MatMulGemvBTAdd
            | ShaderEntry::FusedMatMulAdd
            | ShaderEntry::FusedMatMulATAdd
            | ShaderEntry::FusedMatMulBTAdd
            | ShaderEntry::SgdUpdate
            | ShaderEntry::AdamUpdate
            | ShaderEntry::ScatterAdd
            | ShaderEntry::ScatterAddAtomic
            | ShaderEntry::CrossEntropyLoss
            | ShaderEntry::BceLoss
            | ShaderEntry::Transpose => "main",
            ShaderEntry::SumAll => "sum_all",
            ShaderEntry::MeanAll => "mean_all",
            ShaderEntry::RmsNormAdd => "main",
            ShaderEntry::Embedding => "main",
            ShaderEntry::ToF16 => "main",
            ShaderEntry::RoPE => "main",
            ShaderEntry::RoPEGrad => "main",
            ShaderEntry::LayerNorm => "main",
            ShaderEntry::MultiHeadAttn
            | ShaderEntry::FlashAttention
            | ShaderEntry::FlashAttentionCoop
            | ShaderEntry::FlashAttentionCoopF32
            | ShaderEntry::MultiHeadAttnGradQ
            | ShaderEntry::FlashGradQ
            | ShaderEntry::FlashGradQCoopF16
            | ShaderEntry::MultiHeadAttnGradKV
            | ShaderEntry::FlashGradKV
            | ShaderEntry::FlashGradKVCoopF16
            | ShaderEntry::FlashGradQCoopF32
            | ShaderEntry::FlashGradKVCoopF32 => "main",
            ShaderEntry::SwiGLUGradGate => "swiglu_grad_gate",
            ShaderEntry::SwiGLUGradUp => "swiglu_grad_up",
            ShaderEntry::SiluGrad => "silu_grad",
            ShaderEntry::SwiGLUConcat => "swiglu_concat",
            ShaderEntry::SwiGLUConcatGrad => "swiglu_concat_grad",
            ShaderEntry::GeGLUConcat => "geglu_concat",
            ShaderEntry::GeGLUConcatGrad => "geglu_concat_grad",
            ShaderEntry::SumRows => "sum_rows",
            ShaderEntry::RmsNormGradW => "rms_norm_grad_w",
            ShaderEntry::RmsNormGradWRowPar => "rms_norm_grad_w_rowpar",
            ShaderEntry::RmsNormGradX => "rms_norm_grad_x",
            ShaderEntry::LayerNormGradWB => "layer_norm_grad_wb",
            ShaderEntry::LayerNormGradX => "layer_norm_grad_x",
            ShaderEntry::RmsNormRsqrt => "main",
            ShaderEntry::GroupNorm | ShaderEntry::GroupNormSilu => "main",
            ShaderEntry::GroupNormApply => "apply",
            ShaderEntry::GroupNormStats => "stats",
            ShaderEntry::GroupNormGradInput => "grad_input",
            ShaderEntry::GroupNormGradWeightBias => "grad_weight_bias",
            ShaderEntry::GroupNormGradStats => "grad_stats",
            ShaderEntry::Concat => "main",
            ShaderEntry::SplitA => "split_a",
            ShaderEntry::Permute => "main",
            ShaderEntry::SplitB => "split_b",
            ShaderEntry::Upsample2x => "main",
            ShaderEntry::Upsample2xGrad => "main",
            ShaderEntry::Conv2dDw => "main",
            ShaderEntry::MulPerChannel => "main",
            ShaderEntry::Conv2dGemm
            | ShaderEntry::Conv2dGemmSmall
            | ShaderEntry::Conv2dGemm16
            | ShaderEntry::Conv2dGemmCoopGen(..) => "main",
            ShaderEntry::Conv2dGradInputGemm
            | ShaderEntry::Conv2dGradInputGemmSmall
            | ShaderEntry::Conv2dGradInputGemm16
            | ShaderEntry::Conv2dGradInputGemmCoopGen(..) => "main",
            ShaderEntry::Conv2dGradWeightGemm
            | ShaderEntry::Conv2dGradWeightGemmSmall
            | ShaderEntry::Conv2dGradWeightGemm16
            | ShaderEntry::Conv2dGradWeightGemmSplit
            | ShaderEntry::Conv2dGradWeightGemmSplitSmall
            | ShaderEntry::Conv2dGradWeightGemmSplit16 => "main",
            ShaderEntry::CacheWrite => "main",
            ShaderEntry::CacheWritePrefix => "main",
            ShaderEntry::CachedAttention => "main",
            ShaderEntry::BiasedAttention => "main",
            ShaderEntry::CachedQueryAttention => "main",
            ShaderEntry::CachedBlockAttention => "main",
            ShaderEntry::CachedBlockAttentionSplit => "main",
            ShaderEntry::CachedBlockAttentionCombine => "main",
            ShaderEntry::ChunkedRelativeAttention => "main",
            ShaderEntry::PrefixLast => "main",
            ShaderEntry::RoPEDynamic => "main",
            ShaderEntry::RoPEDynamicFactors => "with_factors",
            ShaderEntry::RoPEPositions => "with_positions",
            ShaderEntry::MaxPool2d => "max_pool_2d",
            ShaderEntry::MaxPool2dGrad => "main",
            ShaderEntry::GlobalAvgPool => "global_avg_pool",
            ShaderEntry::GlobalAvgPoolGrad => "main",
            ShaderEntry::PairwiseGrad => "main",
            ShaderEntry::WinogradInputTransform
            | ShaderEntry::WinogradOutputTransform
            | ShaderEntry::WinogradBatchedMatMul
            | ShaderEntry::WinogradWeightTransform => "main",
            ShaderEntry::GradClipNormSq
            | ShaderEntry::GradClipScale
            | ShaderEntry::AdaptiveGradClip
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
        } else if self.reduction().is_some() {
            "normalization_reduction"
        } else if self.pointwise().is_some() {
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
        self.pointwise().is_some_and(|dag| {
            dag.n_inputs == 1 && dag.ops == [Pw::const_f32(0.0)] && dag.output == 0
        })
    }

    #[cfg(test)]
    fn is_row_scaled_atomic_scatter(&self) -> bool {
        self.shader == ShaderEntry::ScatterAddAtomic
            && self.params.get(3).is_some_and(|&mode| mode != 0)
    }
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
/// [`PointwiseDAG`], so arbitrary per-element transforms can be fused
/// without new enum variants.
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

/// Rows each workgroup folds into one partial row of a norm weight gradient.
///
/// Folding rows shrinks the `[blocks, cols]` partial buffer that SumRows
/// reads back, at the cost of workgroups. Blocks only grow once there would
/// still be at least 1024 workgroups, enough to occupy any current GPU, so
/// short inputs keep one row per workgroup. The kernels split their 256
/// lanes evenly across a block's rows, so a block is a power of two up to 32.
fn norm_weight_grad_rows_per_workgroup(rows: u32) -> u32 {
    const MIN_WORKGROUPS: u32 = 1024;
    let block = (rows / MIN_WORKGROUPS).clamp(1, 32);
    1 << block.ilog2()
}

/// Order dependent dispatches, form barrier groups, then fuse within groups.
pub fn schedule_dispatches(plan: &mut ExecutionPlan, serial: bool, fuse_horizontal: bool) {
    reorder_by_level(&mut plan.dispatches);
    let mut groups = if serial {
        (0..plan.dispatches.len()).map(|i| i..i + 1).collect()
    } else {
        compute_groups(&plan.dispatches)
    };
    if fuse_horizontal {
        fuse_horizontal_matmuls(&mut plan.dispatches, &mut groups);
    }
    warn_on_hazards(&plan.dispatches, &groups);
    plan.groups = groups;
}

/// Stable dependency ordering places independent dispatches next to each other.
fn reorder_by_level(dispatches: &mut Vec<Dispatch>) {
    let n = dispatches.len();
    if n == 0 {
        return;
    }
    // Map: buffer id → index of the dispatch that writes it.
    let mut producer: HashMap<u32, usize> = HashMap::new();
    let mut levels = vec![0u32; n];
    for (i, dispatch) in dispatches.iter().enumerate() {
        let level = dispatch
            .input_buffers
            .iter()
            .filter_map(|b| producer.get(&b.0))
            .map(|&pred| levels[pred] + 1)
            .max()
            .unwrap_or(0);
        levels[i] = level;
        producer.insert(dispatch.output_buffer.0, i);
        for &extra in &dispatch.extra_outputs {
            producer.insert(extra.0, i);
        }
    }
    // Stable sort by level keeps topological order within a level.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| levels[i]);
    let old = std::mem::take(dispatches);
    *dispatches = order.iter().map(|&i| old[i].clone()).collect();
}

/// Start a new compute pass when a dispatch reads an earlier write.
fn compute_groups(dispatches: &[Dispatch]) -> Vec<std::ops::Range<usize>> {
    let mut groups = Vec::new();
    let mut dirty = HashSet::<u32>::new();
    let mut start = 0;
    for (i, dispatch) in dispatches.iter().enumerate() {
        if dispatch.input_buffers.iter().any(|b| dirty.contains(&b.0)) {
            groups.push(start..i);
            start = i;
            dirty.clear();
        }
        dirty.insert(dispatch.output_buffer.0);
        for &extra in &dispatch.extra_outputs {
            dirty.insert(extra.0);
        }
    }
    if !dispatches.is_empty() {
        groups.push(start..dispatches.len());
    }
    groups
}

/// Diagnose missing barriers after horizontal fusion.
fn warn_on_hazards(dispatches: &[Dispatch], groups: &[std::ops::Range<usize>]) {
    for group in groups {
        let mut written = HashSet::<u32>::new();
        for i in group.clone() {
            let d = &dispatches[i];
            for ib in &d.input_buffers {
                if written.contains(&ib.0) {
                    log::warn!(
                        "RAW hazard in group: dispatch {} ({:?}) reads buf {} written earlier in same group",
                        i,
                        d.shader,
                        ib.0
                    );
                }
            }
            written.insert(d.output_buffer.0);
            for &extra in &d.extra_outputs {
                written.insert(extra.0);
            }
        }
    }
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
        && a.kernel == b.kernel
        && matches!(
            a.kernel,
            Kernel::Default
                | Kernel::SmallTile
                | Kernel::Cooperative
                | Kernel::CooperativeCompensated
        )
        && !a.weight_format.uses_reduced_storage()
        && a.weight_format == b.weight_format
        && a.matmul_prologue.is_none()
        && b.matmul_prologue.is_none()
        && a.matmul_epilogue.is_none()
        && b.matmul_epilogue.is_none()
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
        merged.requires_full_precision |= d.requires_full_precision;
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

/// A single GPU dispatch in the execution plan.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
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
    /// Exactly one implementation of the shader's binding contract.
    #[serde(default)]
    pub kernel: Kernel,
    /// The matrix implementation was fixed by extraction. Later kernel
    /// probes must not replace it.
    #[serde(default)]
    pub schedule_locked: bool,
    /// Number of same-A sibling matmuls packed into this dispatch (D1).
    /// 0/1 = not packed. Extra B operands follow A in `input_buffers`;
    /// extra C outputs are `extra_outputs`. `workgroups[2]` is the pack
    /// count when this is ≥ 2 (only applied when the original Z was 1).
    #[serde(default)]
    pub horizontal_batch: u32,
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
    /// Human-readable label for profiling (e.g. `"MatMul[50,720,960]"`).
    #[serde(default)]
    pub label: String,
    /// Graph node ids this dispatch implements. One entry normally; several
    /// after dispatch-level fusion absorbs a neighbor. Provenance only —
    /// execution never reads it, but labels, plan dumps, profiler rows, and
    /// `Session::read_node` do.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub origin: Vec<NodeId>,
    /// Storage format of the B (weight) input buffer.
    #[serde(default)]
    pub weight_format: WeightFormat,
}

/// Mutually exclusive implementations. Bindings and launch geometry live on
/// the dispatch; shader-specific configuration lives only in its variant.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Kernel {
    #[default]
    Default,
    SmallTile,
    ScalarMatmul(crate::codegen::ScalarMatmulShape),
    SplitMatmul {
        shape: crate::codegen::ScalarMatmulShape,
        splits: u32,
    },
    SpecializedConv {
        k_tile: u32,
    },
    Cooperative,
    /// Experimental f16 hi/lo staging, not a full-range f32 implementation.
    CooperativeCompensated,
    Gemv {
        shape: crate::codegen::GemvShape,
        /// Precision policy, never enabled by measurement.
        integer_dot: bool,
    },
    Pointwise(PointwiseDAG),
    Reduction(ReductionKernel),
    AttentionBackward {
        ept_cap: u32,
    },
}

impl Dispatch {
    /// Logical `(m, n, k)` dimensions, accounting for shader parameter order.
    pub fn mnk(&self) -> Option<(u32, u32, u32)> {
        let p = &self.params;
        let (a, b, c) = (p.first().copied()?, p.get(1).copied()?, p.get(2).copied()?);
        match self.shader {
            ShaderEntry::MatMul
            | ShaderEntry::MatMulGemv
            | ShaderEntry::MatMulGemvAdd
            | ShaderEntry::FusedMatMulAdd => Some((a, c, b)),
            ShaderEntry::MatMulAT
            | ShaderEntry::MatMulBT
            | ShaderEntry::MatMulGemvBT
            | ShaderEntry::MatMulGemvBTAdd
            | ShaderEntry::FusedMatMulATAdd
            | ShaderEntry::FusedMatMulBTAdd
            | ShaderEntry::BlockMatMul
            | ShaderEntry::BlockMatMulAT
            | ShaderEntry::BlockMatMulBT
            | ShaderEntry::BatchMatMul
            | ShaderEntry::BatchMatMulAT
            | ShaderEntry::BatchMatMulBT => Some((a, b, c)),
            _ => None,
        }
    }

    pub fn use_coop(&self) -> bool {
        matches!(
            self.kernel,
            Kernel::Cooperative | Kernel::CooperativeCompensated
        )
    }

    pub fn use_coop_compensated(&self) -> bool {
        matches!(self.kernel, Kernel::CooperativeCompensated)
    }

    pub fn use_small_tiles(&self) -> bool {
        matches!(
            self.kernel,
            Kernel::SmallTile
                | Kernel::ScalarMatmul(crate::codegen::ScalarMatmulShape { tile_size: 32, .. })
        )
    }

    pub fn scalar_matmul(&self) -> Option<crate::codegen::ScalarMatmulShape> {
        match self.kernel {
            Kernel::ScalarMatmul(shape) => Some(shape),
            _ => None,
        }
    }

    pub fn conv_k_tile(&self) -> Option<u32> {
        match self.kernel {
            Kernel::SpecializedConv { k_tile } => Some(k_tile),
            _ => None,
        }
    }

    pub fn gemv_shape(&self) -> Option<crate::codegen::GemvShape> {
        match self.kernel {
            Kernel::Gemv { shape, .. } => Some(shape),
            _ => None,
        }
    }

    pub fn gemv_int_dot(&self) -> bool {
        matches!(
            self.kernel,
            Kernel::Gemv {
                integer_dot: true,
                ..
            }
        )
    }

    pub fn pointwise(&self) -> Option<&PointwiseDAG> {
        match self.kernel {
            Kernel::Pointwise(ref dag) => Some(dag),
            _ => None,
        }
    }

    pub fn reduction(&self) -> Option<&ReductionKernel> {
        match self.kernel {
            Kernel::Reduction(ref kernel) => Some(kernel),
            _ => None,
        }
    }

    pub fn reduction_mut(&mut self) -> Option<&mut ReductionKernel> {
        match self.kernel {
            Kernel::Reduction(ref mut kernel) => Some(kernel),
            _ => None,
        }
    }
}

/// Reference to a GPU buffer in the execution plan.
#[derive(Clone, Debug, Default, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BufferRef(pub u32);

/// The complete execution plan: a static sequence of dispatches.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExecutionPlan {
    /// Buffer sizes in bytes, indexed by BufferRef.
    pub buffers: Vec<usize>,
    /// Which buffers hold parameters (need initialization).
    pub param_buffers: Vec<(String, BufferRef)>,
    /// Logical parameter types, unaffected by runtime allocation padding.
    #[serde(default)]
    pub param_types: HashMap<BufferRef, crate::graph::TensorType>,
    /// Which buffers hold inputs (filled each step).
    pub input_buffers: Vec<(String, BufferRef)>,
    /// Constant buffers with their initial data (uploaded once at session creation).
    pub constant_buffers: Vec<(BufferRef, Vec<f32>)>,
    /// The dispatch sequence. For a training graph, this includes
    /// forward, backward, and parameter update dispatches.
    pub dispatches: Vec<Dispatch>,
    /// Barrier groups: each range of dispatch indices shares one compute
    /// pass, and pass boundaries emit the barriers that order them.
    ///
    /// Filled by [`schedule_dispatches`] rather than by `compile`, because
    /// the two steps are not separable — horizontal fusion changes how many
    /// dispatches there are, so the groups can only be computed once the
    /// fusion has run. `runtime` records these; `memplan` uses them to find
    /// each buffer's live range. Defaults to one group per dispatch so a
    /// deserialized or hand-built plan is still recordable.
    #[serde(default)]
    pub groups: Vec<std::ops::Range<usize>>,
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

impl ExecutionPlan {
    /// The dispatches the plan runs, independent of their order and buffer
    /// numbering: each one's shader, kernel, parameters, grid, operand
    /// counts and fused prologue, epilogue, norm, weight format and
    /// precision policy, sorted, then the buffer sizes. Equal inventories
    /// launch the same kernels on the same shapes. They do not show that the
    /// kernels read the same values, which [`Self::dataflow_digest`] does,
    /// and neither measures scheduling, memory traffic or device time.
    pub fn dispatch_inventory(&self) -> Vec<String> {
        let mut out: Vec<String> = self
            .dispatches
            .iter()
            .map(|d| {
                let kinds = |e: &MatMulEpilogue| {
                    e.inputs.iter().map(|i| i.1.clone()).collect::<Vec<_>>()
                };
                format!(
                    "{:?} {:?} {:?} {:?} in={} extra={} epilogue={:?} prologue={:?} norm={:?} weights={:?} batch={} full={}",
                    d.shader,
                    d.kernel,
                    d.params,
                    d.workgroups,
                    d.input_buffers.len(),
                    d.extra_outputs.len(),
                    d.matmul_epilogue.as_ref().map(|e| (&e.dag, kinds(e))),
                    d.matmul_prologue
                        .as_ref()
                        .map(|p| p.factors.iter().map(|f| f.1.clone()).collect::<Vec<_>>()),
                    d.gemv_rmsnorm.as_ref().map(|n| n.eps_bits),
                    d.weight_format,
                    d.horizontal_batch,
                    d.requires_full_precision,
                )
            })
            .collect();
        out.sort();
        let mut buffers = self.buffers.clone();
        buffers.sort_unstable();
        out.push(format!("buffers {buffers:?}"));
        out
    }

    /// A digest of what the plan computes. Each value written is hashed
    /// from everything its dispatch computes it with: shader, kernel,
    /// parameters, grid, fused prologue, epilogue and norm, weight format
    /// and precision policy, and the digests of the values it reads,
    /// including those the fused metadata reads. Leaves are parameters and
    /// inputs by name, derived parameters by source, and constants by
    /// content. The digest covers every value written and the outputs, so
    /// it is independent of dispatch order and buffer numbering but changes
    /// when any dispatch computes differently or reads another value.
    pub fn dataflow_digest(&self) -> u64 {
        use crate::graph::key::structural_key;
        use std::hash::{Hash, Hasher};
        fn hash(value: impl Hash) -> u64 {
            let mut h = std::collections::hash_map::DefaultHasher::new();
            value.hash(&mut h);
            h.finish()
        }
        let mut current: HashMap<BufferRef, u64> = HashMap::new();
        for &(ref name, buf) in self.param_buffers.iter().chain(&self.input_buffers) {
            current.insert(buf, hash(("leaf", name)));
        }
        for &(buf, ref data) in &self.constant_buffers {
            let bits: Vec<u32> = data.iter().map(|v| v.to_bits()).collect();
            current.insert(buf, hash(("constant", bits)));
        }
        for &(buf, ref sources, ref transform) in &self.derived_params {
            current.insert(buf, hash(("derived", sources, structural_key(transform))));
        }
        let unwritten = hash("unwritten");
        let mut written = Vec::with_capacity(self.dispatches.len());
        for d in &self.dispatches {
            let value = |b: &BufferRef| current.get(b).copied().unwrap_or(unwritten);
            let reads: Vec<u64> = d.input_buffers.iter().map(value).collect();
            let epilogue = d.matmul_epilogue.as_ref().map(|e| {
                let inputs: Vec<_> = e
                    .inputs
                    .iter()
                    .map(|entry| (value(&entry.0), entry.1.clone()))
                    .collect();
                (structural_key(&e.dag), inputs)
            });
            let prologue = d.matmul_prologue.as_ref().map(|p| {
                p.factors
                    .iter()
                    .map(|entry| (value(&entry.0), entry.1.clone()))
                    .collect::<Vec<_>>()
            });
            let norm = d
                .gemv_rmsnorm
                .as_ref()
                .map(|n| (value(&n.weight), n.eps_bits));
            let produced = hash((
                (structural_key(&d.shader), structural_key(&d.kernel)),
                (&d.params, d.workgroups, reads),
                (epilogue, prologue, norm),
                structural_key(&d.weight_format),
                (d.horizontal_batch, d.requires_full_precision),
            ));
            for (slot, out) in std::iter::once(&d.output_buffer)
                .chain(&d.extra_outputs)
                .enumerate()
            {
                let value = hash((produced, slot));
                current.insert(*out, value);
                written.push(value);
            }
        }
        written.sort_unstable();
        let outputs: Vec<u64> = self
            .output_buffers
            .iter()
            .map(|b| current.get(b).copied().unwrap_or(unwritten))
            .collect();
        hash((written, outputs))
    }

    fn node_buffer(&self, node_id: NodeId) -> BufferRef {
        let &(mapped_id, buffer) = self
            .node_buffers
            .get(node_id as usize)
            .expect("compiled plan is missing a graph node buffer");
        assert_eq!(
            mapped_id, node_id,
            "compiled node buffers are not ID-sorted"
        );
        buffer
    }

    fn attach_borrowed_constant_buffers(&mut self, graph: &Graph) {
        for node in graph.nodes() {
            if let Op::Constant { ref data } = node.op {
                let buffer = self.node_buffer(node.id);
                self.constant_buffers
                    .push((buffer, constant_words(data.clone(), node.ty.dtype)));
            }
        }
    }

    fn attach_owned_constant_buffers(&mut self, graph: &mut Graph) {
        for node in graph.nodes_mut() {
            let node_id = node.id;
            let dtype = node.ty.dtype;
            if let Op::Constant { ref mut data } = node.op {
                let buffer = self.node_buffer(node_id);
                self.constant_buffers
                    .push((buffer, constant_words(std::mem::take(data), dtype)));
            }
        }
    }

    /// Buffers whose contents are observed outside the dispatch that writes
    /// them: graph outputs, the loss, parameters and their gradients,
    /// derived parameters, inputs, constants, attention LSE side outputs,
    /// and every dispatch's extra outputs. A fusion pass may elide the write
    /// of an intermediate only when it is not one of these.
    pub(crate) fn externally_visible_buffers(&self) -> std::collections::HashSet<BufferRef> {
        let mut visible: std::collections::HashSet<BufferRef> =
            self.output_buffers.iter().copied().collect();
        visible.extend(self.loss_buffer);
        visible.extend(self.param_buffers.iter().map(|entry| entry.1));
        visible.extend(
            self.param_grad_pairs
                .iter()
                .flat_map(|&(param, grad)| [param, grad]),
        );
        visible.extend(self.derived_params.iter().map(|entry| entry.0));
        visible.extend(self.input_buffers.iter().map(|entry| entry.1));
        visible.extend(self.constant_buffers.iter().map(|entry| entry.0));
        visible.extend(self.lse_buffers.iter().map(|entry| entry.1));
        for dispatch in &self.dispatches {
            visible.extend(dispatch.extra_outputs.iter().copied());
        }
        visible
    }

    fn finish(mut self, options: &CompileOptions) -> Self {
        if options.fuse_dispatches {
            fuse_epilogues(&mut self);
            fold_uniform_constants(&mut self);
            fuse_pointwise_chains(&mut self);
            fuse_reduction_chains(&mut self);
            fuse_row_scaled_scatters(&mut self);
        }
        // RmsNorm+MatMul prologue fusion is applied later in the runtime,
        // after per-dispatch coop selection — the prologue path currently
        // only has a coop-matmul implementation. See Session::with_context.

        if let Some(splits) = options.conv_weight_splits {
            self.split_low_occupancy_conv_weights(splits);
        }

        self
    }
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

/// The words a constant uploads: its values, or for `U32` constants (held
/// as exact values in the graph) their integer bits.
fn constant_words(data: Vec<f32>, dtype: crate::graph::DType) -> Vec<f32> {
    match dtype {
        crate::graph::DType::U32 => data.into_iter().map(|v| f32::from_bits(v as u32)).collect(),
        _ => data,
    }
}

pub fn compile(graph: &Graph) -> ExecutionPlan {
    compile_with(graph, &CompileOptions::default())
}

pub fn compile_with(graph: &Graph, options: &CompileOptions) -> ExecutionPlan {
    // No coop capability — callers that know the target hardware go
    // through the capabilities-taking paths below.
    compile_with_caps(graph, options, crate::codegen::CoopCaps::default(), 0)
}

/// Compile for a concrete cooperative-matrix capability set.
///
/// `build()` uses this path after probing the context it will attach to the
/// session. Keeping the target explicit avoids a process-global "first GPU
/// wins" decision when an application owns multiple adapters.
pub(crate) fn compile_with_caps(
    graph: &Graph,
    options: &CompileOptions,
    coop_caps: crate::codegen::CoopCaps,
    shared_memory_bytes: u32,
) -> ExecutionPlan {
    let allow_reduced_precision_attention_backward = options.flash_backward_coop;
    compile_with_caps_policy(
        graph,
        options,
        coop_caps,
        shared_memory_bytes,
        allow_reduced_precision_attention_backward,
    )
}

/// Compile an owned graph, moving constant payloads into the execution plan.
///
/// A differentiated recurrent graph can contain large generated constants.
/// The ordinary borrowed API must clone those values into the plan; session
/// construction no longer needs the optimized graph afterward and can avoid
/// holding both copies at once.
pub(crate) fn compile_owned_with_caps(
    mut graph: Graph,
    options: &CompileOptions,
    coop_caps: crate::codegen::CoopCaps,
    shared_memory_bytes: u32,
) -> ExecutionPlan {
    let allow_reduced_precision_attention_backward = options.flash_backward_coop;
    let mut plan = Compiler::new_with_options(
        &graph,
        options.clone(),
        coop_caps,
        shared_memory_bytes,
        allow_reduced_precision_attention_backward,
    )
    .into_plan();
    plan.attach_owned_constant_buffers(&mut graph);
    plan.finish(options)
}

fn compile_with_caps_policy(
    graph: &Graph,
    options: &CompileOptions,
    coop_caps: crate::codegen::CoopCaps,
    shared_memory_bytes: u32,
    allow_reduced_precision_attention_backward: bool,
) -> ExecutionPlan {
    let mut plan = Compiler::new_with_options(
        graph,
        options.clone(),
        coop_caps,
        shared_memory_bytes,
        allow_reduced_precision_attention_backward,
    )
    .into_plan();
    plan.attach_borrowed_constant_buffers(graph);
    plan.finish(options)
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

    let protected: HashSet<BufferRef> = plan.externally_visible_buffers();

    loop {
        let mut producer = HashMap::new();
        let mut reads = HashMap::new();
        for (index, dispatch) in plan.dispatches.iter().enumerate() {
            producer.insert(dispatch.output_buffer, index);
            for buffer in &dispatch.input_buffers {
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
            let plain_mul = mul.input_buffers.len() == 2
                && mul.pointwise() == Some(&pointwise(2, [Pw::Mul(0, 1)]));
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
                || scatter.workgroups != [source_len.div_ceil(256), 1, 1]
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
        if small_row {
            scatter.workgroups = [scatter.params[1].div_ceil(256), 1, 1];
        }
        scatter.kernel = Kernel::Default;
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

/// Replace pointwise reads of uniform constant tensors with literals.
///
/// Autodiff seeds ReLU masks with `zeros[n]` and mean gradients with
/// `scale[n]`. Read from memory, each costs a full tensor pass from a pinned
/// host-visible allocation and takes one of the three input slots a fused
/// chain may use. As literals they cost nothing, so more chains fuse.
/// The constants themselves stay in the plan so `read_node` still sees them.
fn fold_uniform_constants(plan: &mut ExecutionPlan) {
    use crate::schedule::Pw;
    use std::collections::HashMap;

    let uniform: HashMap<BufferRef, u32> = plan
        .constant_buffers
        .iter()
        .filter_map(|&(buffer, ref data)| {
            let bits = data.first()?.to_bits();
            data.iter()
                .all(|value| value.to_bits() == bits)
                .then_some((buffer, bits))
        })
        .collect();
    if uniform.is_empty() {
        return;
    }

    for dispatch in &mut plan.dispatches {
        let Some(dag) = dispatch.pointwise() else {
            continue;
        };
        let mut dag = dag.clone();
        let mut inputs = dispatch.input_buffers.clone();
        // Walk backwards so earlier slot indices stay valid. Keep one input:
        // pointwise dispatches bind at least one stream.
        for slot in (0..inputs.len()).rev() {
            if inputs.len() == 1 {
                break;
            }
            let Some(&bits) = uniform.get(&inputs[slot]) else {
                continue;
            };
            let literal = PointwiseDAG {
                n_inputs: 0,
                ops: vec![Pw::Const(bits)],
                output: 0,
            };
            dag = dag.fuse_input(slot as u8, &literal);
            inputs.remove(slot);
        }
        if inputs.len() != dispatch.input_buffers.len() {
            dispatch.shader = ShaderEntry::Generated;
            dispatch.input_buffers = inputs;
            dispatch.kernel = Kernel::Pointwise(dag);
        }
    }
}

/// Post-compile pass: merge sequential single-use pointwise dispatches into
/// a single deeper-DAG dispatch, eliminating the intermediate buffer and
/// the barrier between them.
///
/// Conservative criteria — a producer P is fused into consumer C only when:
///   1. Both `P.pointwise()` and `C.pointwise()` are `Some`.
///   2. P's output buffer is read by exactly one dispatch (C) and appears
///      in no plan-level role (output/loss/param/input/constant/extra).
///   3. C's workgroups match P's (same output length).
///   4. Exactly one of C's input buffers equals P's output buffer (no
///      diamond — the intermediate is consumed in one slot only).
fn fuse_pointwise_chains(plan: &mut ExecutionPlan) {
    use std::collections::{HashMap, HashSet};

    // Buffers we must not eliminate. Anything here is preserved even if it
    // looks single-use from the dispatch list alone.
    let mut protected: HashSet<BufferRef> = plan.externally_visible_buffers();

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
        }

        let mut fused_any = false;
        for ci in 0..n {
            let c = &plan.dispatches[ci];
            if c.pointwise().is_none() || c.fusion_barrier {
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
                if p.pointwise().is_none() || p.fusion_barrier {
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
                // A broadcast read wants the producer at another element.
                if c.pointwise()
                    .is_some_and(|dag| dag.broadcasts_input(slot_idx as u8))
                {
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

            let p_dag = producer_d.pointwise().expect("checked above");
            let c_dag = consumer_d.pointwise().expect("checked above").clone();
            let fused_dag = c_dag.fuse_input(input_idx, p_dag);

            // Rebuild consumer input_buffers: producer inputs, then
            // consumer inputs with the fused slot removed, in order.
            let mut new_inputs: Vec<BufferRef> = producer_d.input_buffers.clone();
            for (idx, b) in consumer_d.input_buffers.iter().enumerate() {
                if idx as u8 != input_idx {
                    new_inputs.push(*b);
                }
            }
            consumer_d.input_buffers = new_inputs;
            consumer_d.kernel = Kernel::Pointwise(fused_dag);
            consumer_d.origin.extend(producer_d.origin.iter().copied());
            // The consumer now reads from more buffers; its ShaderEntry
            // (used only to pick the data layout) must reflect the new
            // arity. The runtime binds via UnaryData for n=1, BinaryData
            // for n=2; arities >2 would need a wider layout we don't yet
            // plumb. Guard against that.
            // Update the sentinel `shader` so the (legacy) pipeline
            // lookup still resolves — actual binding/pipeline come from
            // the `pointwise` DAG's arity.
            consumer_d.shader = ShaderEntry::Generated;

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
        let Some(kernel) = dispatch.reduction() else {
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
        let Some(kernel) = dispatch.reduction() else {
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

    let protected =
        |plan: &ExecutionPlan| -> HashSet<BufferRef> { plan.externally_visible_buffers() };

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
            let Some(kernel) = c.reduction() else {
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
                if p.pointwise().is_none_or(PointwiseDAG::has_broadcast) || p.fusion_barrier {
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
                let p_dag = producer_d.pointwise().expect("checked");
                let c = &mut plan.dispatches[ci];
                let kernel = c.reduction_mut().expect("checked");
                kernel.prologue = kernel.prologue.fuse_input(s as u8, p_dag);
                for prologue in &mut kernel.extra_prologues {
                    *prologue = prologue.fuse_input(s as u8, p_dag);
                }
                // The reduction epilogue sees the same per-element streams
                // before its per-column and reduced-value inputs. If it
                // references the folded stream, expand that input there as
                // well; otherwise its declared arity no longer matches
                // n_per_elem and lowering aborts.
                if let Some(epilogue) = kernel.epilogue.as_mut() {
                    epilogue.dag = epilogue.dag.fuse_input(s as u8, p_dag);
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
            let Some(kernel) = c.reduction() else {
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
                    && p.reduction().is_none()
                    && p.pointwise().is_none()
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
                // Gathered indices are clamped to the table's rows, one
                // count per kernel: every gathered table must share it.
                let table_rows = p.params[2];
                let kernel_rows = plan.dispatches[ci].params[3];
                if kernel_rows != 0 && kernel_rows != table_rows {
                    continue;
                }

                let c = &mut plan.dispatches[ci];
                c.params[3] = table_rows;
                let kernel = c.reduction_mut().expect("checked");
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

fn rmsnorm_kernel(cols: u32, eps: f32) -> ReductionKernel {
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

    ReductionKernel {
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
    }
}

fn is_plain_rmsnorm(dispatch: &Dispatch) -> bool {
    // The canonical generated norm only: after pointwise or gather fusion
    // the reduction differs, and a dedicated runtime shader cannot replace it.
    dispatch.input_buffers.len() == 2
        && dispatch.params.len() >= 3
        && dispatch.reduction()
            == Some(&rmsnorm_kernel(
                dispatch.params[1],
                f32::from_bits(dispatch.params[2]),
            ))
}

/// Fold a plain RmsNorm into its GEMV consumers, provided no other operation
/// needs the materialized normalized values.
pub fn fuse_rmsnorm_into_gemv(plan: &mut ExecutionPlan) {
    use std::collections::HashMap;

    let mut readers: HashMap<BufferRef, Vec<usize>> = HashMap::new();
    for (i, d) in plan.dispatches.iter().enumerate() {
        for buf in &d.input_buffers {
            readers.entry(*buf).or_default().push(i);
        }
    }
    let external = plan.externally_visible_buffers();

    let mut drop_norm: Vec<usize> = Vec::new();
    let mut rewrite: Vec<(usize, BufferRef, BufferRef, u32)> = Vec::new();
    for (ni, norm) in plan.dispatches.iter().enumerate() {
        if !is_plain_rmsnorm(norm) {
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
        // Folding is a variant of MatMulGemv: the fused pipeline is
        // `Variant::GemvRmsNorm` — or its int-dot form, `GemvRmsNormIntDot`
        // — keyed by weight format and shape, so a packed GEMV keeps its
        // decoder. Plain GEMV and dense transposed-B GEMV are eligible;
        // fused-add remains separate. Int-dot GEMVs fold too: their activation
        // quantizer runs inside the same workgroup as the prologue, so the
        // integer arithmetic sees exactly the row the unfused path would.
        if consumers.is_empty()
            || consumers.iter().any(|&c| {
                let d = &plan.dispatches[c];
                !matches!(
                    d.shader,
                    ShaderEntry::MatMulGemv | ShaderEntry::MatMulGemvBT
                ) || d.input_buffers.first() != Some(&normed)
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

/// Fuse a single-reader `RmsNorm → Add` pair into one `RmsNormAdd`
/// dispatch.
///
/// The Gemma-style decode keeps a norm *after* every block's output GEMV
/// and feeds it straight into the residual add; those two dispatches are a
/// round trip of `hidden` floats for no reason, since the norm kernel
/// already streams the row. The fused kernel computes the same values in
/// the same order, so this is numerics-neutral and requires nothing from
/// the caller. The norm must have exactly one reader (the add), the add
/// must not be its own output, and neither output may be graph-visible —
/// the fused kernel overwrites the add's buffer and drops both others.
pub fn fuse_rmsnorm_into_add(plan: &mut ExecutionPlan) {
    use std::collections::HashMap;

    let plain_add = pointwise(2, [Pw::Add(0, 1)]);

    let mut readers: HashMap<BufferRef, Vec<usize>> = HashMap::new();
    for (i, d) in plan.dispatches.iter().enumerate() {
        for buf in &d.input_buffers {
            readers.entry(*buf).or_default().push(i);
        }
    }
    let external = plan.externally_visible_buffers();

    let mut drop_dispatches: Vec<usize> = Vec::new();
    let mut rewrite: Vec<(usize, usize, BufferRef)> = Vec::new();
    for (ni, norm) in plan.dispatches.iter().enumerate() {
        if !is_plain_rmsnorm(norm) {
            continue;
        }
        let normed = norm.output_buffer;
        if external.contains(&normed) {
            continue;
        }
        let Some(cons) = readers.get(&normed) else {
            continue;
        };
        if cons.len() != 1 {
            continue;
        }
        let ai = cons[0];
        let add = &plan.dispatches[ai];
        // Only the canonical Add qualifies. Pointwise fusion may have
        // absorbed a consumer (for example Gemma4's `(norm + embedding) /
        // sqrt(2)`) into its DAG, and RmsNormAdd has no pointwise epilogue,
        // so replacing that dispatch would silently drop the consumer.
        if add.pointwise() != Some(&plain_add) || add.output_buffer == normed {
            continue;
        }
        // The add's other input carries the residual.
        let residual = *add
            .input_buffers
            .iter()
            .find(|&&b| b != normed)
            .expect("an add has two inputs");
        // The add's result must not be read while the norm is rewritten
        // beneath it — checked by re-examining the add's own readers below.
        rewrite.push((ni, ai, residual));
        drop_dispatches.push(ai);
    }
    if rewrite.is_empty() {
        return;
    }
    let fused = rewrite.len();
    for (ni, ai, residual) in rewrite {
        let out = plan.dispatches[ai].output_buffer;
        let d = &mut plan.dispatches[ni];
        d.shader = ShaderEntry::RmsNormAdd;
        // A scheduled RmsNorm carries a generated reduction kernel which
        // implements only the original normalization. If it survives this
        // rewrite, pipeline selection prefers that kernel over RmsNormAdd
        // and silently drops the residual. Route the fused operation through
        // its dedicated shader and restore its one-workgroup-per-row shape.
        d.kernel = Kernel::Default;
        d.workgroups = [d.params[0], 1, 1];
        d.input_buffers.push(residual);
        d.output_buffer = out;
    }
    drop_dispatches.sort_unstable();
    drop_dispatches.dedup();
    for i in drop_dispatches.into_iter().rev() {
        plan.dispatches.remove(i);
    }
    log::info!("fuse_rmsnorm_into_add: fused {fused} norm+add pairs");
}

/// Replace a single-consumer plain RmsNorm with a row factor and cooperative
/// matmul prologue. Scalar matmuls keep their materialized normalized input.
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
    let external = plan.externally_visible_buffers();

    let mut to_fuse: Vec<(usize, usize)> = Vec::new(); // (norm_idx, matmul_idx)

    for (i, d) in plan.dispatches.iter().enumerate() {
        // Find MatMul dispatches whose input_buffers[0] comes from a
        // single-consumer RmsNorm. The scalar pipeline cannot execute a
        // matmul prologue, so only transform dispatches that runtime policy
        // has already selected for cooperative matrices.
        if !d.use_coop()
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
            ShaderEntry::MatMulGemv
                | ShaderEntry::MatMulGemvAdd
                | ShaderEntry::MatMulGemvBT
                | ShaderEntry::MatMulGemvBTAdd
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
        if !is_plain_rmsnorm(norm) {
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
    let protected: HashSet<BufferRef> = plan.externally_visible_buffers();

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

        // Every elementwise op lowers to a generated DAG; the dispatch's
        // shader entry only names its binding layout.
        let epilogue_dag = match d.kernel {
            Kernel::Pointwise(ref dag) if dag.n_inputs == 1 && !dag.has_broadcast() => dag.clone(),
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

        // Store-side unary epilogues do not inspect B, so they compose
        // with f16/Q4/Q8 tiled matmuls through
        // `generate_matmul_with_dag_epilogue_fmt`. Cooperative matmuls
        // stay out: their epilogue path is a separate generator and
        // quantized weights still do not feed coop tiles.
        let is_matmul = matches!(
            prod.shader,
            ShaderEntry::MatMul
                | ShaderEntry::MatMulAT
                | ShaderEntry::MatMulBT
                | ShaderEntry::FusedMatMulAdd
                | ShaderEntry::FusedMatMulATAdd
                | ShaderEntry::FusedMatMulBTAdd
        );
        if !is_matmul || prod.use_coop() {
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
            epi.dag = epilogue_dag.fuse_input(0, &epi.dag);
        } else {
            dispatches[prod_idx].matmul_epilogue = Some(MatMulEpilogue {
                dag: epilogue_dag,
                inputs: vec![],
            });
        }
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

/// A pointwise DAG over `n_inputs` loads followed by `ops`, whose operand
/// indices count the loads first. The last op is the output.
fn pointwise(n_inputs: u8, ops: impl IntoIterator<Item = Pw>) -> PointwiseDAG {
    let mut all: Vec<Pw> = (0..n_inputs).map(Pw::LoadInput).collect();
    all.extend(ops);
    PointwiseDAG {
        n_inputs,
        output: (all.len() - 1) as u16,
        ops: all,
    }
}

/// Tanh-form GELU of value `x`, appended to a DAG whose next value index is
/// `next`: 0.5·x·(1 + tanh(u)) = x·sigmoid(2u), u = √(2/π)·(x + 0.044715·x³).
/// The sigmoid spelling does not cancel to zero for negative x.
fn gelu_ops(x: u16, next: u16) -> [Pw; 9] {
    let n = next;
    [
        Pw::Mul(x, x),                    // n   = x²
        Pw::Mul(n, x),                    // n+1 = x³
        Pw::const_f32(0.044715),          // n+2
        Pw::Mul(n + 1, n + 2),            // n+3 = 0.044715·x³
        Pw::Add(x, n + 3),                // n+4 = x + 0.044715·x³
        Pw::const_f32(2.0 * 0.797_884_6), // n+5 = 2·√(2/π)
        Pw::Mul(n + 4, n + 5),            // n+6 = 2u
        Pw::Sigmoid(n + 6),               // n+7
        Pw::Mul(x, n + 7),                // n+8 = gelu(x)
    ]
}

const MAX_COMPUTE_WORKGROUPS_PER_DIMENSION: u32 = 65_535;

/// Conservative launch domain shared by automatic promotion and tuning.
/// A supported matrix shape does not imply a larger dispatch-axis limit.
pub(crate) fn workgroups_within_portable_limits(workgroups: [u32; 3]) -> bool {
    workgroups
        .into_iter()
        .all(|count| count > 0 && count <= MAX_COMPUTE_WORKGROUPS_PER_DIMENSION)
}

/// Reserve complete cooperative output tiles without wrapping shader indices
/// or host byte counts. In particular, a valid 4 GiB allocation is not zero.
pub(crate) fn cooperative_output_bytes(m: u32, n: u32, batch: u32, tile: u32) -> Option<usize> {
    let rows = m.div_ceil(tile).checked_mul(tile)?;
    let cols = n.div_ceil(tile).checked_mul(tile)?;
    let elements = rows.checked_mul(cols)?.checked_mul(batch)?;
    usize::try_from(elements).ok()?.checked_mul(4)
}

#[cfg(test)]
mod dispatch_limits {
    use super::{cooperative_output_bytes, workgroups_within_portable_limits};

    #[test]
    fn cooperative_padding_checks_shader_and_host_extents() {
        assert_eq!(cooperative_output_bytes(65, 272, 1, 32), Some(96 * 288 * 4));
        assert_eq!(
            cooperative_output_bytes(32_767, 32_768, 1, 32),
            usize::try_from(1_u64 << 32).ok(),
        );
        assert_eq!(cooperative_output_bytes(65_535, 65_536, 1, 32), None);
        assert_eq!(cooperative_output_bytes(32, 32, u32::MAX, 32), None);
    }

    #[test]
    fn every_axis_must_be_nonzero_and_within_the_portable_limit() {
        assert!(workgroups_within_portable_limits([65_535; 3]));
        for axis in 0..3 {
            for count in [0, 65_536, u32::MAX] {
                let mut grid = [1; 3];
                grid[axis] = count;
                assert!(!workgroups_within_portable_limits(grid));
            }
        }
    }
}

/// Tile row-GEMV workgroups across X and Y within the portable limit.
/// `groups` workgroups of a one-dimensional kernel over X and Y, each
/// within the portable per-axis limit; the kernel flattens them back.
pub(crate) fn linear_grid(groups: u32) -> [u32; 3] {
    let y = groups.div_ceil(MAX_COMPUTE_WORKGROUPS_PER_DIMENSION).max(1);
    [groups.div_ceil(y).max(1), y, 1]
}

pub(crate) fn row_gemv_workgroups(n: u32) -> [u32; 3] {
    // Large vocabularies can exceed the portable X workgroup limit. Spread
    // rows over Y as well; the kernel flattens the actual dispatch grid.
    let y = n.div_ceil(65_535).max(1);
    [n.div_ceil(y), y, 1]
}

fn matmul_workgroups(m: u32, n: u32, tile: u32) -> [u32; 3] {
    matmul_workgroups_rect(m, n, tile, tile)
}

fn matmul_workgroups_rect(m: u32, n: u32, row_tile: u32, col_tile: u32) -> [u32; 3] {
    let columns = n.div_ceil(col_tile);
    assert!(
        columns <= MAX_COMPUTE_WORKGROUPS_PER_DIMENSION,
        "matmul needs {columns} workgroups on X, exceeding the portable limit"
    );
    let rows = m.div_ceil(row_tile);
    let depth = rows.div_ceil(MAX_COMPUTE_WORKGROUPS_PER_DIMENSION).max(1);
    assert!(
        depth <= MAX_COMPUTE_WORKGROUPS_PER_DIMENSION,
        "matmul needs {depth} workgroup layers, exceeding the portable limit"
    );
    let height = rows.div_ceil(depth);
    debug_assert!(height <= MAX_COMPUTE_WORKGROUPS_PER_DIMENSION);
    [columns, height, depth]
}

/// Largest of 64/32/16 whose launch grid has at least 64 workgroups.
///
/// A 64-wide tile with only a few workgroups leaves most of the GPU idle.
/// F32 cooperative kernels are selected on that same 64-wide entry
/// once its grid reaches 16 workgroups, so devices that have one keep it.
/// The 16-wide tile is the tail: one accumulator per thread, used when the
/// wider launches would sit almost empty.
fn conv_register_tile(rows: u32, cols: u32, batch: u32, f32_coop: bool) -> u32 {
    let batch = batch.max(1);
    let groups = |tile: u32| rows.div_ceil(tile) * cols.div_ceil(tile) * batch;
    if f32_coop && groups(64) >= 16 {
        return 64;
    }
    for tile in [64u32, 32, 16] {
        if groups(tile) >= 64 {
            return tile;
        }
    }
    16
}

#[cfg(test)]
mod conv_tile {
    use super::conv_register_tile;

    #[test]
    fn keeps_a_wide_grid_on_the_64_tile() {
        // 256 x 3136 is 4 * 49 workgroups at tile 64.
        assert_eq!(conv_register_tile(256, 3136, 1, false), 64);
    }

    #[test]
    fn narrows_a_7x7_weight_gradient_to_16() {
        // Co=64, Ci*k=147: tile 64 is 3 workgroups, tile 32 is 10, tile 16 is 40.
        assert_eq!(conv_register_tile(64, 147, 1, false), 16);
    }

    #[test]
    fn keeps_64_when_native_f32_coop_can_use_the_grid() {
        assert_eq!(conv_register_tile(256, 196, 1, true), 64);
        assert_eq!(conv_register_tile(64, 147, 1, true), 16);
    }

    #[test]
    fn splits_only_tall_narrow_row_reductions() {
        // 12544 x 64 is the ResNet stem bias reduction: 2 column groups.
        assert_eq!(super::Compiler::row_reduction_splits(12544, 64), 49);
        assert_eq!(super::Compiler::row_reduction_splits(3136, 64), 13);
        // Few rows, and matrices wide enough to fill the launch, stay one reduction.
        assert_eq!(super::Compiler::row_reduction_splits(32, 64), 1);
        assert_eq!(super::Compiler::row_reduction_splits(10_000, 4096), 1);
    }
}

/// Scalar convolution with geometry baked into the pipeline.
///
/// The uniform software divisor stays available as a measured alternative.
/// This kernel uses the same exact reciprocal, with the multipliers as
/// constants, and the K stage the uniform shader already uses.
fn exact_conv_kernel() -> Kernel {
    Kernel::SpecializedConv { k_tile: 16 }
}

fn conv_gemm_entry(kind: u8, tile: u32) -> ShaderEntry {
    match (kind, tile) {
        (0, 16) => ShaderEntry::Conv2dGemm16,
        (0, 32) => ShaderEntry::Conv2dGemmSmall,
        (0, _) => ShaderEntry::Conv2dGemm,
        (1, 16) => ShaderEntry::Conv2dGradInputGemm16,
        (1, 32) => ShaderEntry::Conv2dGradInputGemmSmall,
        (1, _) => ShaderEntry::Conv2dGradInputGemm,
        (2, 16) => ShaderEntry::Conv2dGradWeightGemm16,
        (2, 32) => ShaderEntry::Conv2dGradWeightGemmSmall,
        _ => ShaderEntry::Conv2dGradWeightGemm,
    }
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
    shared_memory_bytes: u32,
    /// Experimental f16-input attention backward is deliberately separate
    /// from device capability: availability does not imply adequate gradient
    /// accuracy.
    allow_reduced_precision_attention_backward: bool,
    /// Fused GradKV: maps fwd_node → pending, pre-allocated dV buffer.
    /// Allocate GradV's destination before any Identity/reshape views alias
    /// it. GradK writes both gradients; GradV emits no separate dispatch.
    pending_grad_v_buffers: HashMap<NodeId, BufferRef>,
    attention_row_dots: HashMap<(BufferRef, BufferRef, u32, u32), BufferRef>,
    /// GroupNorm backward statistics per (input node, eps bits), computed
    /// once for the input and weight/bias gradients.
    group_norm_grad_stats: HashMap<(NodeId, u32), BufferRef>,
}

// The emitter lives in `compile/emit.rs`: graph traversal that turns nodes
// into dispatches, kept apart from the plan and option types it writes.
mod emit;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;

    /// `Dispatch::mnk` must agree with the order each contraction shader's
    /// constructor writes into `params`.
    ///
    /// The rule was written out in three places — the horizontal-batch binding,
    /// the cooperative-prologue binding, and the arms of `bind_dispatch` — and
    /// the first two each listed a different subset of the shaders that need it.
    /// This test states the intended mapping in one place: a `MatMulAT` reads
    /// `(m, n, k)` straight out of `params`, and a `MatMul` reads `(m, k, n)` and
    /// swaps.
    ///
    /// This test only covers the shaders its own table names, so it cannot
    /// detect a contraction shader `mnk` forgot — it was written after one was.
    /// The backstop is the `mnk` closure in `bind_dispatch`, which panics naming
    /// the shader; adding a contraction binding without a row in the table fails
    /// loudly the first time any test reaches it.
    #[test]
    fn mnk_matches_the_params_order_each_shader_is_built_with() {
        /// One row: the shader, the `params` it is constructed with, and what
        /// `mnk` must return for that `params`.
        type Case = (ShaderEntry, [u32; 3], (u32, u32, u32));
        let cases: &[Case] = &[
            // A is [M, K], B is [K, N]: params hold (m, k, n).
            (ShaderEntry::MatMul, [10, 20, 30], (10, 30, 20)),
            (ShaderEntry::MatMulGemv, [10, 20, 30], (10, 30, 20)),
            (ShaderEntry::MatMulGemvAdd, [10, 20, 30], (10, 30, 20)),
            (ShaderEntry::FusedMatMulAdd, [10, 20, 30], (10, 30, 20)),
            // A is already [K, M]: params hold (m, n, k) already.
            (ShaderEntry::MatMulAT, [10, 20, 30], (10, 20, 30)),
            (ShaderEntry::MatMulBT, [10, 20, 30], (10, 20, 30)),
            (ShaderEntry::MatMulGemvBT, [10, 20, 30], (10, 20, 30)),
            (ShaderEntry::MatMulGemvBTAdd, [10, 20, 30], (10, 20, 30)),
            (ShaderEntry::FusedMatMulATAdd, [10, 20, 30], (10, 20, 30)),
            (ShaderEntry::FusedMatMulBTAdd, [10, 20, 30], (10, 20, 30)),
            // Block matmuls are per-block tiles with the block count in
            // `params[3]`, so `(m, n, k)` is already in that order.
            (ShaderEntry::BlockMatMul, [10, 20, 30], (10, 20, 30)),
            (ShaderEntry::BlockMatMulAT, [10, 20, 30], (10, 20, 30)),
            (ShaderEntry::BlockMatMulBT, [10, 20, 30], (10, 20, 30)),
        ];

        for &(ref shader, params, want) in cases.iter() {
            let d = Dispatch {
                shader: shader.clone(),
                params: params.to_vec(),
                ..Dispatch::default()
            };
            assert_eq!(
                d.mnk(),
                Some(want),
                "{shader:?}: params are {params:?}, so mnk must be {want:?}"
            );
        }
    }

    /// A dispatch that is not a contraction must not hand back three numbers
    /// from `params` that merely happen to be there.
    #[test]
    fn mnk_is_none_for_a_non_contraction() {
        let d = Dispatch {
            shader: ShaderEntry::LayerNorm,
            params: vec![64, 64, 0],
            ..Dispatch::default()
        };
        assert_eq!(d.mnk(), None);
    }

    /// `params` shorter than three is not a shape.
    #[test]
    fn mnk_is_none_when_params_are_short() {
        for shader in [ShaderEntry::MatMul, ShaderEntry::MatMulAT] {
            let d = Dispatch {
                shader: shader.clone(),
                params: vec![10, 20],
                ..Dispatch::default()
            };
            assert_eq!(d.mnk(), None, "{shader:?} with two params");
        }
    }

    #[test]
    fn attention_value_gradient_views_share_fused_output() {
        let mut g = Graph::new();
        // A flattened V parameter makes autodiff reshape dV before using it.
        // Check two attention nodes so their fused outputs cannot be shared.
        let q = g.parameter("q", &[3, 8]);
        let k = g.parameter("k", &[5, 8]);
        let flat_v = g.parameter("v", &[40]);
        let v = g.reshape(flat_v, &[5, 8]);
        let first = g.multi_head_attn(q, k, v, 2, 2, 4, true);
        let flat_v2 = g.parameter("v2", &[40]);
        let v2 = g.reshape(flat_v2, &[5, 8]);
        let second = g.multi_head_attn(first, k, v2, 2, 2, 4, true);
        let loss = g.sum_all(second);
        g.set_outputs(vec![loss]);
        let backward = crate::autodiff::differentiate(&g);
        let plan = compile(&backward);
        let buffers: HashMap<_, _> = plan.node_buffers.iter().copied().collect();
        let mut checked = 0;
        for node in backward.nodes() {
            if let Op::MultiHeadAttnGradV { fwd_node, .. } = node.op {
                let fused = plan
                    .dispatches
                    .iter()
                    .find(|d| {
                        d.origin.iter().any(|&id| {
                            matches!(backward.node(id).op,
                            Op::MultiHeadAttnGradK { fwd_node: fwd, .. } if fwd == fwd_node)
                        })
                    })
                    .expect("fused dK/dV producer");
                assert_eq!(fused.extra_outputs, vec![buffers[&node.id]]);
                checked += 1;
            }
            if matches!(node.op, Op::Identity | Op::StopGradient) {
                assert_eq!(
                    buffers[&node.id], buffers[&node.inputs[0]],
                    "view {} must alias its input {}",
                    node.id, node.inputs[0]
                );
            }
        }
        assert_eq!(checked, 2);
    }

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
    fn scheduled_matmul_impl_locks_the_lowered_kernel() {
        use crate::graph::MatmulImpl;

        let mut g = Graph::new();
        let a = g.input("a", &[50, 720]);
        let b = g.parameter("b", &[720, 960]);
        let y = g.matmul(a, b);
        g.set_outputs(vec![y]);
        g.nodes_mut()[y as usize].matmul_impl = Some(MatmulImpl {
            shape: crate::codegen::ScalarMatmulShape {
                tile_size: 64,
                tile_n: 32,
                k_stage: 32,
                interleave_columns: false,
                unroll_k: true,
            },
            splits: 1,
        });
        let plan = compile(&g);
        let dispatch = &plan.dispatches[0];
        assert!(dispatch.schedule_locked);
        match dispatch.kernel {
            Kernel::ScalarMatmul(shape) => {
                assert_eq!((shape.rows(), shape.cols(), shape.k_stage), (64, 32, 32));
            }
            ref other => panic!("expected scalar 64x32, got {other:?}"),
        }
        assert_eq!(dispatch.workgroups, [960u32.div_ceil(32), 1, 1]);

        g.nodes_mut()[y as usize].matmul_impl = Some(MatmulImpl {
            shape: crate::codegen::ScalarMatmulShape {
                tile_size: 64,
                tile_n: 0,
                k_stage: 8,
                interleave_columns: false,
                unroll_k: true,
            },
            splits: 8,
        });
        let plan = compile(&g);
        assert!(plan.dispatches[0].schedule_locked);
        match plan.dispatches[0].kernel {
            Kernel::SplitMatmul { splits, shape } => {
                assert_eq!(splits, 8);
                assert_eq!((shape.rows(), shape.cols(), shape.k_stage), (64, 64, 8));
            }
            ref other => panic!("expected split-K, got {other:?}"),
        }
        assert_eq!(plan.dispatches[0].workgroups[2], 8);
        assert_eq!(plan.dispatches[1].shader, ShaderEntry::SumRows);
        assert!(!plan.dispatches[1].schedule_locked);

        // K is too short for eight splits, so the single kernel stays locked.
        let mut tiny = Graph::new();
        let a = tiny.input("a", &[3, 33]);
        let b = tiny.parameter("b", &[33, 5]);
        let y = tiny.matmul(a, b);
        tiny.set_outputs(vec![y]);
        tiny.nodes_mut()[y as usize].matmul_impl = Some(MatmulImpl {
            shape: crate::codegen::ScalarMatmulShape {
                tile_size: 64,
                tile_n: 0,
                k_stage: 8,
                interleave_columns: false,
                unroll_k: true,
            },
            splits: 8,
        });
        let plan = compile(&tiny);
        assert_eq!(plan.dispatches.len(), 1);
        assert!(plan.dispatches[0].schedule_locked);
        assert!(matches!(
            &plan.dispatches[0].kernel,
            Kernel::ScalarMatmul(shape) if shape.k_stage == 8 && shape.rows() == 64
        ));
    }

    #[test]
    fn cached_block_attention_splits_decode_but_not_batched_prefill() {
        let compile_shape = |block_len: usize| {
            let mut g = Graph::new();
            let head_dim = 8;
            let max_seq = 128;
            let q = g.input("q", &[block_len, head_dim]);
            let k = g.parameter("k", &[max_seq, head_dim]);
            let v = g.parameter("v", &[max_seq, head_dim]);
            let position = g.input_u32("position", &[1]);
            let valid = g.input_u32("valid", &[1]);
            let output =
                g.cached_block_attention(q, k, v, position, valid, 1, 1, head_dim as u32, 0);
            g.set_outputs(vec![output]);
            compile(&g)
        };

        let decode = compile_shape(1);
        assert!(
            decode
                .dispatches
                .iter()
                .any(|dispatch| dispatch.shader == ShaderEntry::CachedBlockAttentionSplit)
        );
        assert!(
            decode
                .dispatches
                .iter()
                .any(|dispatch| dispatch.shader == ShaderEntry::CachedBlockAttentionCombine)
        );

        let prefill = compile_shape(4);
        assert!(
            prefill
                .dispatches
                .iter()
                .any(|dispatch| dispatch.shader == ShaderEntry::CachedBlockAttention)
        );
        assert!(prefill.dispatches.iter().all(|dispatch| !matches!(
            dispatch.shader,
            ShaderEntry::CachedBlockAttentionSplit | ShaderEntry::CachedBlockAttentionCombine
        )));
    }

    #[test]
    fn owned_compile_preserves_constant_plan_data() {
        let mut g = Graph::new();
        let x = g.input("x", &[2, 2]);
        let constant = g.constant(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let output = g.add(x, constant);
        g.set_outputs(vec![output]);

        let options = CompileOptions::default();
        let caps = crate::codegen::CoopCaps::default();
        let borrowed = compile_with_caps(&g, &options, caps, 0);
        let owned = compile_owned_with_caps(g.deep_clone(), &options, caps, 0);

        assert_eq!(
            serde_json::to_value(borrowed).unwrap(),
            serde_json::to_value(owned).unwrap(),
        );
        let Op::Constant { data } = g.node(constant).op.clone() else {
            panic!(
                "expected retained source constant, got {:?}",
                g.node(constant).op,
            );
        };
        assert_eq!(data, [1.0, 2.0, 3.0, 4.0]);
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
        assert_eq!(
            plan.dispatches[0].matmul_epilogue.as_ref().unwrap().dag.ops,
            [
                crate::schedule::Pw::LoadInput(0),
                crate::schedule::Pw::Relu(0)
            ]
        );
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
        use crate::schedule::Pw;
        let last_op = |d: &Dispatch| d.pointwise().expect("generated").ops.last().cloned();
        assert_eq!(last_op(&plan.dispatches[0]), Some(Pw::Relu(0)));
        assert_eq!(last_op(&plan.dispatches[1]), Some(Pw::Sigmoid(0)));
        assert_eq!(last_op(&plan.dispatches[2]), Some(Pw::Neg(0)));
        assert_eq!(last_op(&plan.dispatches[3]), Some(Pw::Exp(0)));
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
        assert_eq!(copy.shader, ShaderEntry::Generated);
        assert_eq!(copy.params, [8, 0, 0, 0]);
        assert_eq!(copy.workgroups, [1, 1, 1]);
        assert_eq!(copy.input_buffers.len(), 1);
        assert_ne!(copy.input_buffers[0], copy.output_buffer);
        assert!(copy.pointwise().is_some());
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
        use crate::schedule::Pw;
        let last_op = |d: &Dispatch| d.pointwise().expect("generated").ops.last().cloned();
        assert_eq!(last_op(&plan.dispatches[0]), Some(Pw::Add(0, 1)));
        assert_eq!(last_op(&plan.dispatches[1]), Some(Pw::Mul(0, 1)));
        assert_eq!(last_op(&plan.dispatches[2]), Some(Pw::Greater(0, 1)));
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
        assert_eq!(plan.dispatches[0].params[0], 512); // 4*128
        let dag = plan.dispatches[0].pointwise().expect("broadcast pointwise");
        assert!(dag.ops.contains(&Pw::LoadBroadcast {
            input: 1,
            divisor: 1,
            modulus: 128,
        }));
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
        assert_eq!(narrow_dispatch.reduction().unwrap().rows_per_workgroup, 256);

        let mut wide = Graph::new();
        let input = wide.input("input", &[100, 33]);
        let output = wide.sum_inner(input);
        wide.set_outputs(vec![output]);
        let wide_plan = compile(&wide);
        let wide_dispatch = &wide_plan.dispatches[0];
        assert_eq!(wide_dispatch.workgroups, [100, 1, 1]);
        assert_eq!(wide_dispatch.reduction().unwrap().rows_per_workgroup, 1);

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
        let product_kernel = product_plan.dispatches[0].reduction().unwrap();
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
        assert!(reduction.reduction().is_some());
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
        assert!(non_unit_plan.dispatches[0].reduction().is_none());
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
        assert_eq!(forward.reduction().unwrap().rows_per_workgroup, 64);

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
        assert_eq!(large_plan.dispatches[1].workgroups, [3, 1, 1]);
        assert!(
            large_plan.dispatches[1]
                .input_buffers
                .contains(&large_plan.dispatches[1].output_buffer)
        );
    }

    #[test]
    fn gather_reduction_gradient_fuses_row_scaled_atomic_scatter() {
        const SEQ: usize = 1024;
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
        assert_eq!(fused.workgroups, [4, 1, 1]);
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
        assert_eq!(fused.workgroups, [(SEQ * INNER).div_ceil(256) as u32, 1, 1]);
    }

    #[test]
    fn test_compile_softmax() {
        let mut g = Graph::new();
        let x = g.input("x", &[100, 10]);
        let sm = g.softmax(x);
        g.set_outputs(vec![sm]);

        let plan = compile(&g);
        // Softmax compiles to 2 Reduction dispatches (max, then
        // sum/normalize). Check that it has the right batch/features params.
        assert_eq!(plan.dispatches.len(), 2);
        assert_eq!(plan.dispatches[0].params[0], 100); // batch/outer
        assert_eq!(plan.dispatches[0].params[1], 10); // features/inner
        for dispatch in &plan.dispatches {
            assert_eq!(dispatch.workgroups, [7, 1, 1]);
            assert_eq!(dispatch.reduction().unwrap().rows_per_workgroup, 16);
        }
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
            if let Some(reduction) = dispatch.reduction()
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
        // One partial per row, then their sum into the scalar loss.
        assert_eq!(plan.dispatches.len(), 2);
        assert_eq!(plan.dispatches[0].shader, ShaderEntry::CrossEntropyLoss);
        assert_eq!(plan.dispatches[0].workgroups, [4, 1, 1]);
        assert_eq!(plan.dispatches[0].params[0], 4);
        assert_eq!(plan.dispatches[0].params[1], 10);
        assert_eq!(plan.dispatches[0].params[2], 0);
        assert_eq!(plan.dispatches[1].shader, ShaderEntry::SumAll);
        assert_eq!(plan.loss_buffer, Some(plan.dispatches[1].output_buffer));
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
            .filter(|d| d.reduction().is_some())
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
    fn horizontal_fusion_preserves_precision() {
        let mut dispatches = vec![mm_dispatch(0, 1, 2, 1, 32), mm_dispatch(0, 3, 4, 1, 32)];
        for d in &mut dispatches {
            d.kernel = crate::compile::Kernel::Cooperative;
        }
        dispatches[1].requires_full_precision = true;
        let mut groups = Vec::new();
        groups.push(0..2);

        fuse_horizontal_matmuls(&mut dispatches, &mut groups);

        assert_eq!(dispatches.len(), 1);
        let packed = &dispatches[0];
        assert_eq!(packed.horizontal_batch, 2);
        assert_eq!(packed.workgroups, [2, 2, 2]);
        assert!(packed.use_coop());
        assert!(packed.requires_full_precision);
        assert_eq!(packed.extra_outputs, [BufferRef(4)]);
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
    fn rmsnorm_runtime_fusions_preserve_scheduled_inputs() {
        for unary in [false, true] {
            for consumer in 0..3 {
                let mut graph = Graph::new();
                let rows = if consumer == 2 { 32 } else { 1 };
                let input = graph.input("input", &[rows, 64]);
                let residual = graph.input("residual", &[rows, 64]);
                let source = if unary {
                    graph.silu(input)
                } else {
                    graph.add(input, residual)
                };
                let weight = graph.parameter("weight", &[64]);
                let normalized = graph.rms_norm(source, weight, 1e-5);
                let projection = graph.parameter("projection", &[64, 64]);
                let output = if consumer == 0 {
                    graph.add(normalized, residual)
                } else {
                    graph.matmul(normalized, projection)
                };
                graph.set_outputs(vec![output]);
                let mut plan = compile(&graph);
                let normalized = plan
                    .dispatches
                    .iter()
                    .find(|dispatch| dispatch.reduction().is_some())
                    .expect("scheduled normalization")
                    .clone();
                assert!(normalized.reduction().is_some());
                assert!(!is_plain_rmsnorm(&normalized));
                for dispatch in &mut plan.dispatches {
                    if dispatch.shader == ShaderEntry::MatMul {
                        dispatch.kernel = Kernel::Cooperative;
                    }
                }
                match consumer {
                    0 => fuse_rmsnorm_into_add(&mut plan),
                    1 => fuse_rmsnorm_into_gemv(&mut plan),
                    _ => fuse_rmsnorm_prologues(&mut plan),
                }
                assert!(plan.dispatches.contains(&normalized));
            }
        }
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
        assert!(scalar_plan.dispatches.iter().any(is_plain_rmsnorm));
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
        coop_plan.dispatches[matmul_index].kernel = crate::compile::Kernel::Cooperative;
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
        assert!(plan.dispatches[0].pointwise().is_some());
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
        // The matmul keeps no epilogue; the bias add and ReLU fuse into one
        // broadcast pointwise dispatch after it.
        assert_eq!(plan.dispatches.len(), 2);
        assert_eq!(plan.dispatches[0].shader, ShaderEntry::MatMul);
        let dag = plan.dispatches[1].pointwise().expect("fused bias and ReLU");
        assert!(dag.has_broadcast());
        assert!(dag.ops.iter().any(|op| matches!(op, Pw::Relu(_))));
    }

    #[test]
    fn relu_gradient_reads_no_constant_tensors() {
        // Autodiff compares against a zeros tensor and scales by a
        // mean-gradient tensor. Both must become literals, which leaves room
        // to fold the mask into the gradient product.
        let mut g = Graph::new();
        let x = g.parameter("x", &[1000]);
        let y = g.relu(x);
        let loss = g.mean_all(y);
        g.set_outputs(vec![loss]);
        let diff = crate::autodiff::differentiate(&g);
        let plan = compile(&diff);
        let constants: std::collections::HashSet<_> =
            plan.constant_buffers.iter().map(|entry| entry.0).collect();
        for dispatch in plan.dispatches.iter().filter(|d| d.pointwise().is_some()) {
            assert!(
                dispatch
                    .input_buffers
                    .iter()
                    .all(|b| !constants.contains(b)),
                "{} still reads a constant tensor",
                dispatch.label
            );
        }
        assert!(
            plan.dispatches.iter().all(|d| !d
                .pointwise()
                .is_some_and(|dag| dag.ops.contains(&crate::schedule::Pw::Greater(0, 1)))),
            "the ReLU mask was not fused into its consumer"
        );
    }

    #[test]
    fn per_channel_bias_fuses_into_relu() {
        let mut g = Graph::new();
        let x = g.input("x", &[2 * 8 * 36]);
        let b = g.parameter("b", &[8]);
        let y = g.add_per_channel(x, b, 8, 36);
        let y = g.relu(y);
        g.set_outputs(vec![y]);
        let plan = compile(&g);
        assert_eq!(plan.dispatches.len(), 1, "{:#?}", plan.dispatches);
        let dag = plan.dispatches[0]
            .pointwise()
            .expect("fused pointwise kernel");
        assert!(dag.has_broadcast());
        assert!(dag.ops.contains(&Pw::Relu(2)));
    }

    #[test]
    fn weighted_matmul_fuses_pointwise_epilogue() {
        let mut g = Graph::new();
        let x = g.input("x", &[4, 32]);
        let w = g.parameter_q4("w", &[32, 64]);
        let mm = g.matmul(x, w);
        let output = g.sigmoid(mm);
        g.set_outputs(vec![output]);

        let plan = compile(&g);
        assert_eq!(plan.dispatches.len(), 1);
        assert_eq!(plan.dispatches[0].weight_format, WeightFormat::Q4);
        assert!(plan.dispatches[0].matmul_epilogue.is_some());
        assert_eq!(plan.dispatches[0].shader, ShaderEntry::MatMul);
    }

    #[test]
    fn generated_clamp_keeps_its_exact_matmul_epilogue() {
        let mut g = Graph::new();
        let x = g.input("x", &[2, 32]);
        let w = g.parameter_f16("w", &[20, 32]);
        let mm = g.matmul_bt(x, w);
        let output = g.clamp(mm, -3.25, 4.5);
        g.set_outputs(vec![output]);

        let plan = compile(&g);
        assert_eq!(plan.dispatches.len(), 1);
        let epilogue = plan.dispatches[0]
            .matmul_epilogue
            .as_ref()
            .expect("clamp should fuse into the matmul store");
        assert_eq!(
            epilogue.dag,
            PointwiseDAG {
                n_inputs: 1,
                ops: vec![
                    Pw::LoadInput(0),
                    Pw::const_f32(-3.25),
                    Pw::const_f32(4.5),
                    Pw::Max(0, 1),
                    Pw::Min(3, 2),
                ],
                output: 4,
            }
        );
    }

    #[test]
    fn test_shader_entry_mappings() {
        // Verify all shader entries have valid group and entry_point
        let entries = [
            ShaderEntry::MatMul,
            ShaderEntry::SgdUpdate,
            ShaderEntry::AdamUpdate,
            ShaderEntry::ScatterAdd,
            ShaderEntry::ScatterAddAtomic,
            ShaderEntry::SumAll,
            ShaderEntry::MeanAll,
            ShaderEntry::SumRows,
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
        let mut compiler = Compiler::new_with_options(
            &graph,
            CompileOptions::default(),
            crate::codegen::CoopCaps::default(),
            0,
            false,
        );
        for hd_log2 in 1..=8 {
            let hd: u32 = 1 << hd_log2;
            let fwd_ept = hd.min(TuningKnobs::default().flash_ept_cap);
            let fwd_tpq = hd / fwd_ept;
            let grad_q_ept = hd.min(TuningKnobs::default().flash_grad_q_ept_cap);
            let grad_q_tpq = hd / grad_q_ept;
            let grad_q_bq: u32 = (256 / grad_q_tpq).max(1);
            let grad_kv_ept = hd.min(TuningKnobs::default().flash_grad_kv_ept_cap);
            let grad_kv_tpq = hd / grad_kv_ept;
            let grad_kv_bq: u32 = (256 / grad_kv_tpq).max(1);

            for threads in [128, 256] {
                compiler.options.knobs.flash.threads = threads;
                let fwd_bq = (threads / fwd_tpq).max(1);
                let (fwd_entry, fwd_wg) = compiler.attention_dispatch(256, hd, 1, false);
                if fwd_bq >= 2 {
                    assert_eq!(fwd_entry, ShaderEntry::FlashAttention);
                    assert_eq!(fwd_wg[0], 256u32.div_ceil(fwd_bq));
                }
            }
            let (grad_q_entry, grad_q_wg) =
                Compiler::attention_dispatch_bwd(256, hd, 1, grad_q_ept);
            let (grad_kv_entry, grad_kv_wg) =
                Compiler::attention_dispatch_bwd(256, hd, 1, grad_kv_ept);
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

    #[test]
    fn native_f32_forward_attention_respects_precision_storage_and_grid_bounds() {
        let graph = Graph::new();
        for (tile, head_dim, rows, bytes, selected) in [
            (16, 16, 17, 4096, true),
            (16, 16, 17, 4095, false),
            (16, 64, 15, 65536, false),
            (16, 64, 1500, 65536, true),
            (16, 48, 128, 65536, false),
            (16, 512, 128, 65536, false),
            (16, 64, 16 * 65536, 65536, false),
            (8, 64, 128, 65536, false),
        ] {
            let compiler = Compiler::new_with_options(
                &graph,
                CompileOptions::default(),
                crate::codegen::CoopCaps {
                    f32_tile: tile,
                    f16_tile: 16,
                },
                bytes,
                false,
            );
            for full_precision in [false, true] {
                let (entry, grid) = compiler.attention_dispatch(rows, head_dim, 3, full_precision);
                assert_eq!(entry == ShaderEntry::FlashAttentionCoopF32, selected);
                if selected {
                    assert_eq!(grid, [rows.div_ceil(16), 3, 1]);
                }
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
        let safe = compile_with_caps_policy(
            &differentiated,
            &CompileOptions::default(),
            f16_only,
            49_152,
            false,
        );
        let safe_entries: Vec<_> = safe
            .dispatches
            .iter()
            .map(|dispatch| dispatch.shader.clone())
            .collect();
        assert!(safe_entries.contains(&ShaderEntry::FlashAttentionCoop));
        assert!(!safe_entries.contains(&ShaderEntry::FlashGradQCoopF16));
        assert!(!safe_entries.contains(&ShaderEntry::FlashGradKVCoopF16));

        let experimental = compile_with_caps_policy(
            &differentiated,
            &CompileOptions::default(),
            f16_only,
            49_152,
            true,
        );
        let experimental_entries: Vec<_> = experimental
            .dispatches
            .iter()
            .map(|dispatch| dispatch.shader.clone())
            .collect();
        assert!(experimental_entries.contains(&ShaderEntry::FlashGradQCoopF16));
        assert!(experimental_entries.contains(&ShaderEntry::FlashGradKVCoopF16));

        g.nodes_mut()[attention as usize].requires_full_precision = true;
        let full =
            compile_with_caps_policy(&g, &CompileOptions::default(), f16_only, 49_152, false);
        assert!(
            full.dispatches
                .iter()
                .any(|d| d.shader == ShaderEntry::FlashAttention)
        );
        assert!(
            full.dispatches
                .iter()
                .all(|d| d.shader != ShaderEntry::FlashAttentionCoop)
        );

        let scalar = compile_with_caps_policy(
            &differentiated,
            &CompileOptions::default(),
            crate::codegen::CoopCaps::default(),
            0,
            false,
        );
        assert!(scalar.dispatches.iter().all(|dispatch| !matches!(
            &dispatch.shader,
            ShaderEntry::FlashAttentionCoop
                | ShaderEntry::FlashGradQCoopF16
                | ShaderEntry::FlashGradKVCoopF16
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
            49_152,
            true,
        );

        assert!(
            plan.dispatches
                .iter()
                .any(|dispatch| dispatch.shader == ShaderEntry::FlashGradKVCoopF16)
        );
        assert!(
            plan.dispatches
                .iter()
                .all(|dispatch| { dispatch.shader != ShaderEntry::MultiHeadAttnGradKV })
        );
    }

    #[test]
    fn coop_f32_attention_backward_requires_qualified_shape_and_storage() {
        for (tile, head_dim, q_seq, kv_seq, bytes, expected) in [
            (8, 64, 129, 145, 18_624, [true, true]),
            (8, 64, 129, 145, 18_623, [false, false]),
            (0, 64, 129, 145, 32_768, [false, false]),
            (16, 64, 129, 145, 18_624, [true, true]),
            (16, 64, 129, 145, 18_623, [false, false]),
            (8, 128, 129, 145, 32_768, [false, false]),
            (8, 64, 50, 145, 32_768, [false, false]),
            (8, 64, 129, 16, 32_768, [false, false]),
            (8, 64, 16 * 65_535, 128, 32_768, [true, true]),
            (8, 64, 16 * 65_536, 128, 32_768, [false, true]),
            (8, 64, 128, 16 * 65_535, 32_768, [true, true]),
            (8, 64, 128, 16 * 65_536, 32_768, [true, false]),
        ] {
            let mut graph = Graph::new();
            let q = graph.parameter("q", &[q_seq, head_dim]);
            let k = graph.parameter("k", &[kv_seq, head_dim]);
            let v = graph.parameter("v", &[kv_seq, head_dim]);
            let attention = graph.multi_head_attn(q, k, v, 1, 1, head_dim as u32, true);
            let loss = graph.sum_all(attention);
            graph.set_outputs(vec![loss]);
            let backward = crate::autodiff::differentiate(&graph);
            let plan = compile_with_caps(
                &backward,
                &CompileOptions::default(),
                crate::codegen::CoopCaps {
                    f16_tile: 0,
                    f32_tile: tile,
                },
                bytes,
            );
            for (shader, expected) in [
                (ShaderEntry::FlashGradQCoopF32, expected[0]),
                (ShaderEntry::FlashGradKVCoopF32, expected[1]),
            ] {
                assert_eq!(
                    plan.dispatches
                        .iter()
                        .any(|dispatch| dispatch.shader == shader),
                    expected,
                    "{shader:?}",
                );
            }
        }
    }

    #[test]
    fn coop_f32_attention_backward_respects_independent_scalar_layouts() {
        for (q_cap, kv_cap) in [(Some(4), None), (None, Some(8))] {
            let mut graph = Graph::new();
            let q = graph.parameter("q", &[129, 64]);
            let k = graph.parameter("k", &[145, 64]);
            let v = graph.parameter("v", &[145, 64]);
            let attention = graph.multi_head_attn(q, k, v, 1, 1, 64, true);
            let loss = graph.sum_all(attention);
            graph.set_outputs(vec![loss]);
            let mut backward = crate::autodiff::differentiate(&graph);
            for node in backward.nodes_mut() {
                match node.op {
                    Op::MultiHeadAttnGradQ { .. } => node.attention_ept_cap = q_cap,
                    Op::MultiHeadAttnGradK { .. } => node.attention_ept_cap = kv_cap,
                    _ => {}
                }
            }
            let plan = compile_with_caps(
                &backward,
                &CompileOptions::default(),
                crate::codegen::CoopCaps {
                    f16_tile: 0,
                    f32_tile: 8,
                },
                32_768,
            );
            for (cap, cooperative, scalar) in [
                (
                    q_cap,
                    ShaderEntry::FlashGradQCoopF32,
                    ShaderEntry::FlashGradQ,
                ),
                (
                    kv_cap,
                    ShaderEntry::FlashGradKVCoopF32,
                    ShaderEntry::FlashGradKV,
                ),
            ] {
                let dispatch = plan
                    .dispatches
                    .iter()
                    .find(|dispatch| dispatch.shader == cooperative || dispatch.shader == scalar)
                    .unwrap();
                if let Some(ept_cap) = cap {
                    assert_eq!(dispatch.shader, scalar);
                    assert_eq!(dispatch.kernel, Kernel::AttentionBackward { ept_cap });
                } else {
                    assert_eq!(dispatch.shader, cooperative);
                    assert_eq!(dispatch.kernel, Kernel::Default);
                }
            }
        }
    }

    #[test]
    fn cooperative_attention_respects_shared_memory_limits() {
        let caps = crate::codegen::CoopCaps {
            f16_tile: 16,
            f32_tile: 0,
        };
        let options = CompileOptions {
            flash_backward_coop: true,
            ..CompileOptions::default()
        };
        for (head_dim, bytes, expected) in [
            (64, 0, [false, false, false]),
            (64, 16_384, [true, true, false]),
            (128, 32_768, [true, true, false]),
            (256, 49_152, [true, false, false]),
            (256, 52_415, [true, false, false]),
            (256, 52_416, [true, true, false]),
            (256, 70_080, [true, true, true]),
            (512, 65_536, [false, false, false]),
        ] {
            let mut graph = Graph::new();
            let shape = [32, head_dim as usize];
            let q = graph.parameter("q", &shape);
            let k = graph.parameter("k", &shape);
            let v = graph.parameter("v", &shape);
            let attention = graph.causal_attention(q, k, v, 1, 1, head_dim);
            let loss = graph.sum_all(attention);
            graph.set_outputs(vec![loss]);
            let differentiated = crate::autodiff::differentiate(&graph);
            let plan = compile_with_caps(&differentiated, &options, caps, bytes);
            for (shader, wanted) in [
                ShaderEntry::FlashAttentionCoop,
                ShaderEntry::FlashGradQCoopF16,
                ShaderEntry::FlashGradKVCoopF16,
            ]
            .into_iter()
            .zip(expected)
            {
                assert_eq!(
                    plan.dispatches
                        .iter()
                        .any(|dispatch| dispatch.shader == shader),
                    wanted,
                    "{shader:?}, head_dim={head_dim}, shared_memory_bytes={bytes}",
                );
            }
        }
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
        assert_eq!(ShaderEntry::SwiGLUGradGate.profile_family(), "pointwise");
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
                .any(|dispatch| dispatch.reduction().is_some())
        );
        assert!(
            plan.dispatches
                .iter()
                .filter(|dispatch| dispatch.reduction().is_some())
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
        assert_eq!(large.dispatches[0].shader, ShaderEntry::GroupNormStats);
        assert_eq!(large.dispatches[1].shader, ShaderEntry::GroupNormApply);
        let chunks = large.dispatches[1].params[5];
        assert_eq!(chunks, 3);
        assert_eq!(8202 % chunks, 0);
        assert_eq!(large.dispatches[0].workgroups[0], 8 * chunks);
        assert_eq!(large.dispatches[0].params, large.dispatches[1].params);
    }
}
