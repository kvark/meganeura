//! The primitive op set, and composite ops written in terms of it.
//!
//! A model built only from primitives runs without any model-specific
//! kernel: every primitive has a lowering and a reference implementation,
//! and all but a few have a gradient (depthwise convolution and
//! `PrefixLast` are inference-only). Fused kernels are the compiler's
//! business: optimized builds recognize composites spelled in primitives
//! ([`Graph::recompose`]), so a new model needs a new kernel only to run
//! faster, not to run at all.
//!
//! The class is a structural role. What an op supports (storage types,
//! shapes, training) is stated by its builder.
//!
//! The set follows StableHLO's meaning for each op, restricted to what a
//! statically planned graph needs: static shapes, reductions and broadcasts
//! along the inner axis of a 2D tensor, and scalar attributes in place of
//! broadcast constants.

use super::{Graph, NodeId, Op};

/// Where an op sits in the architecture.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OpClass {
    /// Part of the primitive set: its lowering and reference
    /// implementation, and its gradient where it has one, are defined
    /// directly on it.
    Primitive,
    /// A named operation with a decomposition into primitives
    /// ([`Graph::decompose`]). Optimized builds recognize the decomposition
    /// again ([`Graph::recompose_for`]), training builds only where the
    /// gradient is the decomposition's.
    Composite,
    /// A fused kernel or gradient helper that only the optimizer and
    /// autodiff create.
    Private,
}

impl Op {
    /// The op's place in the architecture. The match is exhaustive, so
    /// every new op has to declare one.
    pub fn class(&self) -> OpClass {
        use OpClass::{Composite, Primitive, Private};
        match *self {
            // Leaves.
            Op::Parameter { .. } | Op::Input { .. } | Op::Constant { .. } => Primitive,
            // `dot_general`, in each operand layout.
            Op::MatMul
            | Op::MatMulAT
            | Op::MatMulBT
            | Op::BlockMatMul
            | Op::BlockMatMulAT { .. }
            | Op::BlockMatMulBT
            | Op::BatchMatMul
            | Op::BatchMatMulAT
            | Op::BatchMatMulBT => Primitive,
            // Elementwise math and comparison.
            Op::Add
            | Op::Mul
            | Op::Greater
            | Op::Neg
            | Op::Abs
            | Op::Exp
            | Op::Erf
            | Op::Log
            | Op::Recip
            | Op::Sqrt
            | Op::Rsqrt
            | Op::Sin
            | Op::Cos
            | Op::Tanh
            | Op::Sigmoid
            | Op::Relu
            | Op::Scale { .. }
            | Op::Offset { .. }
            | Op::Clamp { .. } => Primitive,
            // Broadcasts, reductions and data movement.
            Op::BiasAdd
            | Op::BiasMul
            | Op::BroadcastInner { .. }
            | Op::SumInner
            | Op::MaxInner
            | Op::SumAll
            | Op::ExclusiveCumsum { .. }
            | Op::Transpose
            | Op::Permute { .. }
            | Op::BroadcastTo
            | Op::Identity
            | Op::Materialize
            | Op::StopGradient
            | Op::Concat { .. }
            | Op::SplitA { .. }
            | Op::SplitB { .. }
            | Op::ToF16
            | Op::ToF32
            | Op::ToU32 => Primitive,
            // Gather, scatter and dynamic slices.
            Op::Embedding
            | Op::ScatterAdd { .. }
            | Op::CacheWrite
            | Op::CacheWritePrefix
            | Op::PrefixLast => Primitive,
            // Convolution and `reduce_window`.
            Op::Conv2d { .. } | Op::Conv2dDw { .. } | Op::MaxPool2d { .. } => Primitive,

            Op::Softplus { .. }
            | Op::Silu
            | Op::Gelu
            | Op::SwiGLU
            | Op::GeGLU
            | Op::MeanAll
            | Op::SumRows
            | Op::GlobalAvgPool { .. }
            | Op::NormalizeInnerSum { .. }
            | Op::PairwiseSquaredDistance { .. }
            | Op::PairwiseVectorRejection { .. }
            | Op::ShiftInner { .. }
            | Op::Softmax
            | Op::LogSoftmax
            | Op::CrossEntropyLoss
            | Op::BceLoss
            | Op::RmsNorm { .. }
            | Op::LayerNorm { .. }
            | Op::GroupNorm { .. }
            | Op::MulPerChannel { .. }
            | Op::AddPerChannel { .. }
            | Op::Upsample2x { .. }
            | Op::RoPE { .. }
            | Op::RoPEPositions { .. }
            | Op::CausalAttention { .. }
            | Op::FullAttention { .. }
            | Op::CrossAttention { .. }
            | Op::MultiHeadAttn { .. }
            | Op::SlidingWindowAttention { .. }
            | Op::CachedAttention { .. }
            | Op::CachedBlockAttention { .. }
            | Op::ChunkedRelativeAttention { .. }
            | Op::BiasedAttention { .. }
            | Op::BiasedCachedAttention { .. } => Composite,

            Op::SoftplusGrad { .. }
            | Op::NormalizeInnerSumGrad { .. }
            | Op::PairwiseGrad { .. }
            | Op::CrossEntropyLogitsGrad
            | Op::FusedMatMulAdd
            | Op::FusedMatMulATAdd
            | Op::FusedMatMulBTAdd
            | Op::Nop
            | Op::SwiGLUConcat
            | Op::SwiGLUConcatGrad
            | Op::GeGLUConcat
            | Op::GeGLUConcatGrad
            | Op::SwiGLUGradGate
            | Op::SwiGLUGradUp
            | Op::SiluGrad
            | Op::RoPEGrad { .. }
            // Causal attention rotating Q and K itself; nothing builds it.
            | Op::CausalAttentionRoPE { .. }
            | Op::MultiHeadAttnGradQ { .. }
            | Op::MultiHeadAttnGradK { .. }
            | Op::MultiHeadAttnGradV { .. }
            | Op::RmsNormGradW { .. }
            | Op::RmsNormGradX { .. }
            | Op::LayerNormGradWB { .. }
            | Op::LayerNormGradX { .. }
            | Op::Conv2dGradInput { .. }
            | Op::Conv2dGradWeight { .. }
            | Op::MaxPool2dGrad { .. }
            | Op::GlobalAvgPoolGrad { .. }
            | Op::WinogradConv2d { .. }
            | Op::GroupNormSilu { .. }
            | Op::GroupNormGradInput { .. }
            | Op::GroupNormGradWeightBias { .. }
            | Op::Upsample2xGrad { .. } => Private,
        }
    }

    /// Whether this op belongs to the primitive set.
    pub fn is_primitive(&self) -> bool {
        self.class() == OpClass::Primitive
    }

    /// Whether autodiff differentiates this composite, applied to `arity`
    /// inputs, exactly as it does its decomposition, so that recognizing it
    /// in a training graph changes no gradient. A list of the cases the
    /// gradient comparison in `tests/oracle/composites.rs` verifies; any
    /// other op is not recognized for training. The losses are absent
    /// because their gradients treat the labels as constant and skip the
    /// clamp, RoPE is present only in its static form, and the cached,
    /// biased and chunked attention forms are inference-only.
    pub(crate) fn differentiates_as_decomposed(&self, arity: usize) -> bool {
        match *self {
            Op::RoPE { freq_factors, .. } => arity == 1 && !freq_factors,
            Op::Softplus { .. }
            | Op::Silu
            | Op::Gelu
            | Op::SwiGLU
            | Op::GeGLU
            | Op::MeanAll
            | Op::SumRows
            | Op::GlobalAvgPool { .. }
            | Op::NormalizeInnerSum { .. }
            | Op::PairwiseSquaredDistance { .. }
            | Op::PairwiseVectorRejection { .. }
            | Op::ShiftInner { .. }
            | Op::Softmax
            | Op::LogSoftmax
            | Op::RmsNorm { .. }
            | Op::LayerNorm { .. }
            | Op::GroupNorm { .. }
            | Op::MulPerChannel { .. }
            | Op::AddPerChannel { .. }
            | Op::Upsample2x { .. }
            | Op::CausalAttention { .. }
            | Op::FullAttention { .. }
            | Op::MultiHeadAttn { .. }
            | Op::SlidingWindowAttention { .. } => true,
            _ => false,
        }
    }
}

/// The `1 / N` factor of a mean over `inner` elements, which
/// [`Graph::mean_inner`] scales by. The optimizer recognizes a mean by
/// comparing against this exact value.
pub(crate) fn mean_factor(inner: usize) -> f32 {
    1.0 / inner as f32
}

impl Graph {
    /// Numerically stable softmax over the inner axis of `x: [M, N]`,
    /// built from primitives:
    /// `exp(x - max(x)) / sum(exp(x - max(x)))`.
    ///
    /// This is [`Op::Softmax`]'s decomposition; builds recognize it as one.
    #[track_caller]
    pub fn decomposed_softmax(&mut self, x: NodeId) -> NodeId {
        let inner = self.node(x).ty.shape[1];
        let max = self.max_inner(x);
        let max = self.broadcast_inner(max, inner);
        let shifted = self.sub(x, max);
        let exp = self.exp(shifted);
        let sum = self.sum_inner(exp);
        let inv = self.recip(sum);
        let inv = self.broadcast_inner(inv, inner);
        self.mul(exp, inv)
    }

    /// RMS normalization of `x: [M, N]` with weight `[N]`, built from
    /// primitives: `x * rsqrt(mean(x²) + eps) * weight`.
    ///
    /// This is [`Op::RmsNorm`]'s decomposition; builds recognize it as one.
    #[track_caller]
    pub fn decomposed_rms_norm(&mut self, x: NodeId, weight: NodeId, eps: f32) -> NodeId {
        let inner = self.node(x).ty.shape[1];
        let square = self.mul(x, x);
        let mean = self.mean_inner(square);
        let mean = self.add_scalar(mean, eps);
        let inv = self.rsqrt(mean);
        let inv = self.broadcast_inner(inv, inner);
        let normalized = self.mul(x, inv);
        self.bias_mul(normalized, weight)
    }

    /// Layer normalization of `x: [M, N]` with weight and bias `[N]`,
    /// built from primitives:
    /// `(x - mean(x)) * rsqrt(mean((x - mean(x))²) + eps) * weight + bias`.
    ///
    /// This is [`Op::LayerNorm`]'s full decomposition; builds recognize it
    /// as one.
    #[track_caller]
    pub fn decomposed_layer_norm(
        &mut self,
        x: NodeId,
        weight: NodeId,
        bias: NodeId,
        eps: f32,
    ) -> NodeId {
        let inner = self.node(x).ty.shape[1];
        let mean = self.mean_inner(x);
        let mean = self.broadcast_inner(mean, inner);
        let centered = self.sub(x, mean);
        let square = self.mul(centered, centered);
        let variance = self.mean_inner(square);
        let variance = self.add_scalar(variance, eps);
        let inv = self.rsqrt(variance);
        let inv = self.broadcast_inner(inv, inner);
        let normalized = self.mul(centered, inv);
        let scaled = self.bias_mul(normalized, weight);
        self.bias_add(scaled, bias)
    }
}
