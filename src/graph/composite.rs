//! The primitive op set, and composite ops written in terms of it.
//!
//! A model built only from primitives runs without any model-specific
//! kernel: every primitive has a lowering, a gradient and a reference
//! implementation. Fused kernels are the optimizer's business. Its rewrite
//! rules recognize a decomposition such as [`Graph::decomposed_softmax`]
//! and replace it with the fused op, so a new model needs a new kernel only
//! to run faster, not to run at all.
//!
//! The set follows StableHLO's meaning for each op, restricted to what a
//! statically planned graph needs: static shapes, reductions and broadcasts
//! along the inner axis of a 2D tensor, and scalar attributes in place of
//! broadcast constants.

use super::{Graph, NodeId, Op};

impl Op {
    /// Whether this op belongs to the primitive set, which model builders
    /// and loaders can rely on: everything else is a fused kernel, a
    /// composite with a decomposition, or a gradient helper.
    pub fn is_primitive(&self) -> bool {
        matches!(
            *self,
            Op::Parameter { .. }
                | Op::Input { .. }
                | Op::Constant { .. }
                // Elementwise
                | Op::Add
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
                | Op::Tanh
                | Op::Sigmoid
                | Op::Relu
                | Op::Scale { .. }
                | Op::Offset { .. }
                | Op::Clamp { .. }
                // Broadcasts along rows and along the inner axis
                | Op::BiasAdd
                | Op::BiasMul
                | Op::BroadcastInner { .. }
                // Reductions
                | Op::SumInner
                | Op::MaxInner
                | Op::SumAll
                // Contraction and data movement
                | Op::MatMul
                | Op::BatchMatMul
                | Op::Transpose
                | Op::Permute { .. }
                | Op::Identity
                | Op::Embedding
                | Op::StopGradient
        )
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
    /// The optimizer replaces this with the fused [`Op::Softmax`] kernel.
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
    /// The optimizer replaces this with the fused [`Op::RmsNorm`] kernel.
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
    /// The optimizer replaces this with the fused [`Op::LayerNorm`] kernel.
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
