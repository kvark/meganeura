//! Shader binding layouts and generic stack/heap binding carriers.
//!
//! Blade's `PipelineEncoder::bind` ignores `ShaderData::layout()` and binds
//! slots `0..=N` against the pipeline's own [`bg::ShaderDataLayout`], which
//! matches WGSL globals by name. Every non-optimizer compute dispatch binds a
//! run of storage buffers followed by one `params` uniform, so [`BuffersData`]
//! (fixed arity, stack-allocated) and [`BufferListData`] (dynamic arity) cover
//! all of them without a per-shader `#[derive(ShaderData)]` wrapper struct.

use super::optimizer;
use crate::compile::{
    AttentionParams, CachedBlockAttentionParams, RoPEParams, ShaderEntry, WinogradInputParams,
    WinogradOutputParams, WinogradWeightParams,
};
use crate::kernels::attention_grad::Part as AttentionGradPart;
use crate::kernels::conv::Params as Conv2dParams;
use blade_graphics as bg;

// ---- Generic ShaderData carriers ----

/// Fixed-arity storage buffers followed by a single `params` uniform.
///
/// `layout()` is a placeholder: the pipeline owns the real [`bg::ShaderDataLayout`]
/// produced by [`buffers_layout`], and `fill()` writes `buffers[0..N]` into
/// slots `0..N` and `params` into slot `N`.
pub(super) struct BuffersData<const N: usize, P> {
    pub(super) buffers: [bg::BufferPiece; N],
    pub(super) params: P,
}

impl<const N: usize, P: bg::ShaderBindable + Copy> bg::ShaderData for BuffersData<N, P> {
    fn layout() -> bg::ShaderDataLayout {
        bg::ShaderDataLayout::default()
    }
    fn fill(&self, mut context: bg::PipelineContext) {
        use bg::ShaderBindable as _;
        for (index, buffer) in self.buffers.iter().enumerate() {
            buffer.bind_to(&mut context, index as u32);
        }
        self.params.bind_to(&mut context, N as u32);
    }
}

/// Bind `N` storage buffers and one `params` uniform at binding group 0.
pub(super) fn bind_buffers<const N: usize, P: bg::ShaderBindable + Copy>(
    pc: &mut impl bg::traits::PipelineEncoder,
    buffers: [bg::BufferPiece; N],
    params: P,
) {
    pc.bind(0, &BuffersData { buffers, params });
}

/// A variable-length run of storage buffers followed by one `params` uniform.
///
/// Used when a shader's buffer arity is parameterized (horizontal matmul packs
/// and wide multi-gather reductions). Like [`BuffersData`], the pipeline owns
/// the real [`bg::ShaderDataLayout`].
pub(super) struct BufferListData<P> {
    pub(super) buffers: Vec<bg::BufferPiece>,
    pub(super) params: P,
}

impl<P: bg::ShaderBindable + Copy> bg::ShaderData for BufferListData<P> {
    fn layout() -> bg::ShaderDataLayout {
        bg::ShaderDataLayout::default()
    }
    fn fill(&self, mut context: bg::PipelineContext) {
        use bg::ShaderBindable as _;
        let mut index = 0u32;
        for b in &self.buffers {
            b.bind_to(&mut context, index);
            index += 1;
        }
        self.params.bind_to(&mut context, index);
    }
}

/// Horizontal matmul bindings: every matrix in the pack, then `params`.
pub(super) type HorizMatMulData = BufferListData<MatMulParams>;

/// Gather-reduction bindings: one buffer per gathered stream, then `params`.
pub(super) type DynReductionData = BufferListData<ReductionParams>;

/// Build a [`bg::ShaderDataLayout`] for storage buffers named `names` followed
/// by a `"params"` plain uniform of size `size_of::<P>()`.
pub(super) fn buffers_layout<P>(names: &[&'static str]) -> bg::ShaderDataLayout {
    let mut bindings = Vec::with_capacity(names.len() + 1);
    for &name in names {
        bindings.push((name, bg::ShaderBinding::Buffer));
    }
    bindings.push((
        "params",
        bg::ShaderBinding::Plain {
            size: std::mem::size_of::<P>() as u32,
        },
    ));
    bg::ShaderDataLayout { bindings }
}

// ---- Uniform parameter structs ----

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
pub(super) struct ScatterAddParams {
    pub(super) total: u32,
    pub(super) seq_len: u32,
    pub(super) embed_dim: u32,
    pub(super) _pad: u32,
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
pub(super) struct MatMulParams {
    pub(super) m: u32,
    pub(super) n: u32,
    pub(super) k: u32,
    pub(super) _pad: u32,
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
pub(super) struct MatMulRmsNormParams {
    pub(super) m: u32,
    pub(super) n: u32,
    pub(super) k: u32,
    pub(super) eps_bits: u32,
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
pub(super) struct UnaryParams {
    pub(super) len: u32,
    pub(super) _pad0: u32,
    pub(super) _pad1: u32,
    pub(super) _pad2: u32,
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
pub(super) struct PermuteParams {
    pub(super) total: u32,
    pub(super) dims: [u32; 3],
    pub(super) strides: [u32; 4],
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
pub(super) struct BiasAddParams {
    pub(super) len: u32,
    pub(super) bias_len: u32,
    pub(super) _pad0: u32,
    pub(super) _pad1: u32,
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
pub(super) struct MulPerChannelParams {
    pub(super) len: u32,
    pub(super) spatial: u32,
    pub(super) _pad0: u32,
    pub(super) _pad1: u32,
}

/// Schedule-template reduction params: `{ outer, inner, round_one_bits, table_rows }`.
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
pub(super) struct ReductionParams {
    pub(super) outer: u32,
    pub(super) inner: u32,
    pub(super) round_one_bits: u32,
    pub(super) table_rows: u32,
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
pub(super) struct GroupNormParams {
    pub(super) batch: u32,
    pub(super) channels: u32,
    pub(super) spatial: u32,
    pub(super) num_groups: u32,
    pub(super) eps_bits: u32,
    /// Slices each group is split into, so the parallelism follows the image
    /// rather than the batch. One for the legacy single-pass kernels.
    pub(super) chunks: u32,
    pub(super) apply_silu: u32,
    pub(super) _pad2: u32,
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
pub(super) struct Conv2dDwParams {
    pub(super) batch: u32,
    pub(super) channels: u32,
    pub(super) in_h: u32,
    pub(super) in_w: u32,
    pub(super) kernel_h: u32,
    pub(super) kernel_w: u32,
    pub(super) stride: u32,
    pub(super) padding_h: u32,
    pub(super) out_h: u32,
    pub(super) out_w: u32,
    pub(super) padding_w: u32,
    pub(super) _pad: u32,
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
pub(super) struct MaxPool2dParams {
    pub(super) batch: u32,
    pub(super) channels: u32,
    pub(super) in_h: u32,
    pub(super) in_w: u32,
    pub(super) kernel_h: u32,
    pub(super) kernel_w: u32,
    pub(super) stride: u32,
    pub(super) padding: u32,
    pub(super) out_h: u32,
    pub(super) out_w: u32,
    pub(super) _pad0: u32,
    pub(super) _pad1: u32,
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
pub(super) struct GlobalAvgPoolParams {
    pub(super) channels: u32,
    pub(super) spatial: u32,
    pub(super) total_out: u32,
    pub(super) _pad: u32,
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
pub(super) struct CachedAttentionParams {
    pub(super) queries: u32,
    pub(super) num_heads: u32,
    pub(super) num_kv_heads: u32,
    pub(super) head_dim: u32,
    pub(super) max_seq: u32,
    pub(super) _pad: [u32; 3],
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
pub(super) struct ChunkedRelativeAttentionParams {
    pub(super) seq_len: u32,
    pub(super) num_heads: u32,
    pub(super) head_dim: u32,
    pub(super) left_context: u32,
    pub(super) softcap_bits: u32,
    pub(super) _pad0: u32,
    pub(super) _pad1: u32,
    pub(super) _pad2: u32,
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
pub(super) struct SoftmaxParams {
    pub(super) batch: u32,
    pub(super) features: u32,
    pub(super) _pad0: u32,
    pub(super) _pad1: u32,
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
pub(super) struct TransposeParams {
    pub(super) m: u32,
    pub(super) n: u32,
    pub(super) _pad0: u32,
    pub(super) _pad1: u32,
}

// ---- ShaderDataLayout builders ----

/// Layout for a GEMV with its input RmsNorm folded in.
pub(super) fn matmul_rmsnorm_layout() -> bg::ShaderDataLayout {
    buffers_layout::<MatMulRmsNormParams>(&["matrix_a", "norm_w", "matrix_b", "matrix_c"])
}

pub(super) fn horizontal_matmul_layout(count: u32) -> bg::ShaderDataLayout {
    let mut names = Vec::with_capacity(1 + 2 * count as usize);
    names.push("matrix_a");
    for i in 0..count {
        names.push(match i {
            0 => "matrix_b0",
            1 => "matrix_b1",
            _ => "matrix_b2",
        });
    }
    for i in 0..count {
        names.push(match i {
            0 => "matrix_c0",
            1 => "matrix_c1",
            _ => "matrix_c2",
        });
    }
    buffers_layout::<MatMulParams>(&names)
}

/// Ordered storage-buffer binding names for a reduction kernel's input
/// streams: each per-element stream, with its `_idx` indices buffer right
/// after it when that stream is a gather; then per-row; then per-col.
/// Excludes `dst` / `params`. Names align with the globals emitted by
/// `schedule::lower_reduction`.
fn reduction_input_binding_names(k: &crate::schedule::ReductionKernel) -> Vec<&'static str> {
    let per_elem: &[&str] = match k.n_per_elem {
        1 => &["src"],
        2 => &["src_a", "src_b"],
        3 => &["src_a", "src_b", "src_c"],
        n => panic!("reduction n_per_elem {n} unsupported"),
    };
    let idx: &[&str] = match k.n_per_elem {
        1 => &["src_idx"],
        2 => &["src_a_idx", "src_b_idx"],
        3 => &["src_a_idx", "src_b_idx", "src_c_idx"],
        _ => &[],
    };
    let per_row: &[&str] = match k.n_per_row {
        0 => &[],
        1 => &["per_row_src"],
        2 => &["per_row_src_a", "per_row_src_b"],
        n => panic!("reduction n_per_row {n} unsupported"),
    };
    let n_per_col = k.epilogue.as_ref().map_or(0, |e| e.n_per_col_inputs);
    let per_col: &[&str] = match n_per_col {
        0 => &[],
        1 => &["bias"],
        2 => &["bias_a", "bias_b"],
        n => panic!("reduction n_per_col {n} unsupported"),
    };
    let mut names = Vec::new();
    for i in 0..k.n_per_elem as usize {
        names.push(per_elem[i]);
        if k.gather_elem.get(i).copied().unwrap_or(false) {
            names.push(idx[i]);
        }
    }
    names.extend_from_slice(per_row);
    names.extend_from_slice(per_col);
    names
}

/// ShaderDataLayout for a schedule-template reduction pipeline.
/// Names align with the bindings emitted by `schedule::lower` for reductions.
pub(super) fn reduction_data_layout(
    kernel: &crate::schedule::ReductionKernel,
) -> bg::ShaderDataLayout {
    let mut names = reduction_input_binding_names(kernel);
    names.push("dst");
    buffers_layout::<ReductionParams>(&names)
}

/// ShaderDataLayout for a schedule-template pointwise pipeline, chosen by
/// DAG input arity. Names align with the bindings emitted by
/// `schedule::lower` for each arity.
pub(super) fn pointwise_data_layout(n_inputs: u8) -> bg::ShaderDataLayout {
    match n_inputs {
        1 => buffers_layout::<UnaryParams>(&["src", "dst"]),
        2 => buffers_layout::<UnaryParams>(&["src_a", "src_b", "dst"]),
        3 => buffers_layout::<UnaryParams>(&["src_a", "src_b", "src_c", "dst"]),
        n => panic!("pointwise arity {n} has no runtime data layout"),
    }
}

/// Get the ShaderDataLayout for a coop matmul with an N-factor prologue.
pub fn matmul_with_prologue_layout(n_factors: usize) -> bg::ShaderDataLayout {
    match n_factors {
        2 => buffers_layout::<MatMulParams>(&[
            "matrix_a",
            "matrix_b",
            "matrix_c",
            "prologue_buf_0",
            "prologue_buf_1",
        ]),
        n => panic!("matmul prologue arity {n} has no runtime data layout"),
    }
}

/// Get the ShaderDataLayout for a given shader entry.
pub fn shader_data_layout(entry: &ShaderEntry) -> bg::ShaderDataLayout {
    use bg::ShaderData;
    match *entry {
        // Generated kernels take their layout from their kernel.
        ShaderEntry::Generated => bg::ShaderDataLayout {
            bindings: Vec::new(),
        },
        ShaderEntry::MatMul
        | ShaderEntry::MatMulAT
        | ShaderEntry::MatMulBT
        | ShaderEntry::MatMulGemv
        | ShaderEntry::MatMulGemvBT
        | ShaderEntry::BlockMatMul
        | ShaderEntry::BlockMatMulAT
        | ShaderEntry::BlockMatMulBT
        | ShaderEntry::BatchMatMul
        | ShaderEntry::BatchMatMulAT
        | ShaderEntry::BatchMatMulBT
        | ShaderEntry::WinogradBatchedMatMul => {
            buffers_layout::<MatMulParams>(&["matrix_a", "matrix_b", "matrix_c"])
        }
        ShaderEntry::FusedMatMulAdd
        | ShaderEntry::FusedMatMulATAdd
        | ShaderEntry::FusedMatMulBTAdd
        | ShaderEntry::MatMulGemvBTAdd
        | ShaderEntry::MatMulGemvAdd => {
            buffers_layout::<MatMulParams>(&["matrix_a", "matrix_b", "matrix_c", "src"])
        }
        ShaderEntry::PairwiseGrad | ShaderEntry::SwiGLUGradGate => {
            buffers_layout::<UnaryParams>(&["src_a", "src_b", "src_c", "dst"])
        }
        ShaderEntry::SgdUpdate => optimizer::SgdData::layout(),
        ShaderEntry::AdamUpdate => optimizer::AdamData::layout(),
        ShaderEntry::ScatterAdd => buffers_layout::<ScatterAddParams>(&["indices", "src", "dst"]),
        ShaderEntry::ScatterAddAtomic => {
            buffers_layout::<ScatterAddParams>(&["indices", "src", "row_scale", "dst"])
        }
        ShaderEntry::SwiGLUConcat
        | ShaderEntry::SwiGLUConcatGrad
        | ShaderEntry::GeGLUConcat
        | ShaderEntry::GeGLUConcatGrad
        | ShaderEntry::SwiGLUGradUp
        | ShaderEntry::SiluGrad
        | ShaderEntry::Concat => buffers_layout::<UnaryParams>(&["src_a", "src_b", "dst"]),
        ShaderEntry::SumAll
        | ShaderEntry::MeanAll
        | ShaderEntry::SumRows
        | ShaderEntry::ToF16
        | ShaderEntry::ToBF16
        | ShaderEntry::FromBF16
        | ShaderEntry::RmsNormRsqrt
        | ShaderEntry::SplitA
        | ShaderEntry::SplitB
        | ShaderEntry::Upsample2x
        | ShaderEntry::Upsample2xGrad
        | ShaderEntry::GlobalAvgPoolGrad => buffers_layout::<UnaryParams>(&["src", "dst"]),
        ShaderEntry::CrossEntropyLoss | ShaderEntry::CrossEntropyLossIndices => {
            buffers_layout::<SoftmaxParams>(&["logits", "labels", "grad_out", "loss_out"])
        }
        ShaderEntry::BceLoss => buffers_layout::<UnaryParams>(&["pred", "labels", "loss_out"]),
        ShaderEntry::Transpose => buffers_layout::<TransposeParams>(&["src", "dst"]),
        ShaderEntry::Embedding => buffers_layout::<UnaryParams>(&["indices", "src", "dst"]),
        ShaderEntry::RoPE | ShaderEntry::RoPEGrad => buffers_layout::<RoPEParams>(&["src", "dst"]),
        ShaderEntry::LayerNorm => buffers_layout::<MatMulParams>(&["src", "src_b", "bias", "dst"]),
        ShaderEntry::RmsNormAdd => {
            buffers_layout::<BiasAddParams>(&["src", "bias", "residual", "dst"])
        }
        ShaderEntry::MultiHeadAttn
        | ShaderEntry::FlashAttention
        | ShaderEntry::FlashAttentionCoop
        | ShaderEntry::FlashAttentionProducts
        | ShaderEntry::FlashAttentionCoopF32 => {
            buffers_layout::<AttentionParams>(&["src_a", "src_b", "bias", "dst", "lse"])
        }
        ShaderEntry::AttentionGrad(kernel) => match kernel.part {
            AttentionGradPart::Q => buffers_layout::<AttentionParams>(&[
                "d_out", "src_a", "src_b", "bias", "lse", "fwd_dst", "dst",
            ]),
            AttentionGradPart::KV => buffers_layout::<AttentionParams>(&[
                "d_out", "src_a", "src_b", "bias", "lse", "fwd_dst", "dst", "dst2",
            ]),
        },
        ShaderEntry::RmsNormGradW
        | ShaderEntry::RmsNormGradWRowPar
        | ShaderEntry::RmsNormGradX
        | ShaderEntry::LayerNormGradWB
        | ShaderEntry::LayerNormGradX => {
            buffers_layout::<MatMulParams>(&["src_a", "src_b", "bias", "dst"])
        }
        ShaderEntry::GroupNorm | ShaderEntry::GroupNormSilu => {
            buffers_layout::<GroupNormParams>(&["src", "src_b", "bias", "dst"])
        }
        ShaderEntry::GroupNormApply => {
            buffers_layout::<GroupNormParams>(&["src", "src_b", "bias", "dst", "partials"])
        }
        ShaderEntry::GroupNormStats => buffers_layout::<GroupNormParams>(&["src", "dst"]),
        ShaderEntry::GroupNormGradInput
        | ShaderEntry::GroupNormGradWeightBias
        | ShaderEntry::GroupNormGradStats => {
            buffers_layout::<GroupNormParams>(&["src_a", "src_b", "bias", "dst", "stats"])
        }
        ShaderEntry::Permute => buffers_layout::<PermuteParams>(&["src", "dst"]),
        ShaderEntry::BiasedAttention => {
            buffers_layout::<[u32; 8]>(&["q", "k", "v", "bias", "kv_pos", "dst"])
        }
        ShaderEntry::Conv2dDw => buffers_layout::<Conv2dDwParams>(&["src", "weight", "dst"]),
        ShaderEntry::MulPerChannel => {
            buffers_layout::<MulPerChannelParams>(&["src", "gate", "dst"])
        }
        ShaderEntry::Conv2dGemm
        | ShaderEntry::Conv2dBF16 { .. }
        | ShaderEntry::Conv2dGemmSmall
        | ShaderEntry::Conv2dGemm16
        | ShaderEntry::Conv2dGemmCoopGen(..) => {
            buffers_layout::<Conv2dParams>(&["src", "weight", "dst"])
        }
        ShaderEntry::Conv2dGradInputGemm
        | ShaderEntry::Conv2dGradInputGemmSmall
        | ShaderEntry::Conv2dGradInputGemm16
        | ShaderEntry::Conv2dGradInputGemmCoopGen(..) => {
            buffers_layout::<Conv2dParams>(&["grad_out", "weight", "dst"])
        }
        ShaderEntry::Conv2dGradWeightGemm
        | ShaderEntry::Conv2dGradWeightGemmSmall
        | ShaderEntry::Conv2dGradWeightGemm16
        | ShaderEntry::Conv2dGradWeightGemmSplit
        | ShaderEntry::Conv2dGradWeightGemmSplitSmall
        | ShaderEntry::Conv2dGradWeightGemmSplit16 => {
            buffers_layout::<Conv2dParams>(&["grad_out", "src", "dst"])
        }
        ShaderEntry::RoPEDynamic | ShaderEntry::RoPEPositions | ShaderEntry::AxialRoPE => {
            buffers_layout::<RoPEParams>(&["src", "dst", "pos_offset_buf"])
        }
        ShaderEntry::RoPEDynamicFactors => {
            buffers_layout::<RoPEParams>(&["src", "dst", "pos_offset_buf", "factors"])
        }
        ShaderEntry::CacheWrite => buffers_layout::<UnaryParams>(&["src", "dst", "kv_pos_buf"]),
        ShaderEntry::CacheWritePrefix => {
            buffers_layout::<MatMulParams>(&["src", "dst", "kv_pos_buf", "valid_len_buf"])
        }
        ShaderEntry::CachedAttention | ShaderEntry::CachedQueryAttention => {
            buffers_layout::<CachedAttentionParams>(&[
                "src_a",
                "src_b",
                "bias",
                "kv_pos_buf",
                "dst",
            ])
        }
        ShaderEntry::CachedBlockAttention | ShaderEntry::CachedBlockAttentionSplit => {
            buffers_layout::<CachedBlockAttentionParams>(&[
                "src_a",
                "src_b",
                "bias",
                "kv_pos_buf",
                "valid_len_buf",
                "dst",
            ])
        }
        ShaderEntry::CachedBlockAttentionCombine => {
            buffers_layout::<CachedBlockAttentionParams>(&["partials", "dst"])
        }
        ShaderEntry::ChunkedRelativeAttention => {
            buffers_layout::<ChunkedRelativeAttentionParams>(&[
                "src_a",
                "src_b",
                "bias",
                "relative_k",
                "dst",
            ])
        }
        ShaderEntry::PrefixLast => buffers_layout::<MatMulParams>(&["src", "valid_len_buf", "dst"]),
        ShaderEntry::MaxPool2d => buffers_layout::<MaxPool2dParams>(&["src", "dst"]),
        ShaderEntry::MaxPool2dGrad => {
            buffers_layout::<MaxPool2dParams>(&["grad_out", "src", "dst"])
        }
        ShaderEntry::GlobalAvgPool => buffers_layout::<GlobalAvgPoolParams>(&["src", "dst"]),
        ShaderEntry::WinogradInputTransform => {
            buffers_layout::<WinogradInputParams>(&["src", "dst"])
        }
        ShaderEntry::WinogradOutputTransform { fused: false } => {
            buffers_layout::<WinogradOutputParams>(&["src", "dst"])
        }
        ShaderEntry::WinogradOutputTransform { fused: true } => {
            buffers_layout::<WinogradOutputParams>(&["src", "dst", "residual"])
        }
        ShaderEntry::WinogradWeightTransform => {
            buffers_layout::<WinogradWeightParams>(&["src", "dst"])
        }
        ShaderEntry::GradClipNormSq | ShaderEntry::GradClipScale | ShaderEntry::GradAccum => {
            optimizer::GradPassData::layout()
        }
        ShaderEntry::AdaptiveGradClip => optimizer::AdaptiveGradClipData::layout(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffers_layout_appends_plain_params_slot() {
        let layout = buffers_layout::<MatMulParams>(&["matrix_a", "matrix_b", "matrix_c"]);
        assert_eq!(layout.bindings.len(), 4);
        assert_eq!(layout.bindings[0], ("matrix_a", bg::ShaderBinding::Buffer));
        assert_eq!(layout.bindings[1], ("matrix_b", bg::ShaderBinding::Buffer));
        assert_eq!(layout.bindings[2], ("matrix_c", bg::ShaderBinding::Buffer));
        assert_eq!(
            layout.bindings[3],
            (
                "params",
                bg::ShaderBinding::Plain {
                    size: std::mem::size_of::<MatMulParams>() as u32,
                }
            )
        );
    }

    #[test]
    fn winograd_fused_layout_places_params_last_for_buffers_data() {
        let layout = shader_data_layout(&ShaderEntry::WinogradOutputTransform { fused: true });
        let names: Vec<&str> = layout.bindings.iter().map(|&(n, _)| n).collect();
        assert_eq!(names, ["src", "dst", "residual", "params"]);
    }
}
