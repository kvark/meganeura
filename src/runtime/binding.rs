//! Bind dispatch operands and uniforms to their shader layout.

use super::shader_data::{
    BiasAddParams, CachedAttentionParams, ChunkedRelativeAttentionParams, Conv2dDwParams,
    DynReductionData, GlobalAvgPoolParams, GroupNormParams, HorizMatMulData, MatMulParams,
    MatMulRmsNormParams, MaxPool2dParams, MulPerChannelParams, PermuteParams, ReductionParams,
    ScatterAddParams, SoftmaxParams, TransposeParams, UnaryParams, bind_buffers,
};
use super::{AttentionParams, Conv2dParams, RoPEParams, Session};
use crate::compile::{
    BufferRef, CachedBlockAttentionParams, Dispatch, DispatchParams, Kernel, ShaderEntry,
    WinogradInputParams, WinogradOutputParams, WinogradWeightParams,
};
use crate::kernels::attention_grad::{AttentionGrad, Part as AttentionGradPart};

fn dense_params(dispatch: &Dispatch) -> MatMulParams {
    let (m, n, k) = dispatch.mnk().expect("dense matrix dimensions");
    MatMulParams {
        m,
        n,
        k,
        _pad: match dispatch.kernel {
            Kernel::CooperativeTyped { splits, .. }
            | Kernel::CooperativeWorkgroup { splits, .. } => splits,
            _ => dispatch.workgroups[1],
        },
    }
}

impl Session {
    pub(super) fn bind_dispatch(
        buffers: &[blade_graphics::BufferPiece],
        dispatch: &Dispatch,
        pc: &mut impl blade_graphics::traits::PipelineEncoder,
    ) {
        let buf = |r: BufferRef| buffers[r.0 as usize];
        let input = |i: usize| buf(dispatch.input_buffers[i]);
        let output = || buf(dispatch.output_buffer);
        let mnk = || {
            dispatch
                .mnk()
                .unwrap_or_else(|| panic!("missing matrix dimensions for {:?}", dispatch.shader))
        };
        if dispatch.horizontal_batch >= 2 {
            let count = dispatch.horizontal_batch as usize;
            let mut pieces = vec![input(0)];
            for i in 0..count {
                pieces.push(input(1 + i));
            }
            pieces.push(output());
            pieces.extend(dispatch.extra_outputs.iter().map(|&r| buf(r)));
            let (m, n, k) = mnk();
            pc.bind(
                0,
                &HorizMatMulData {
                    buffers: pieces,
                    params: MatMulParams { m, n, k, _pad: 0 },
                },
            );
            return;
        }
        // A GEMV with its RmsNorm folded in takes the norm's weight vector
        // as an extra binding and carries eps in the params' spare slot.
        if let Some(ref rn) = dispatch.gemv_rmsnorm {
            let (m, n, k) = mnk();
            bind_buffers(
                pc,
                [input(0), buf(rn.weight), input(1), output()],
                MatMulRmsNormParams {
                    m,
                    n,
                    k,
                    eps_bits: rn.eps_bits,
                },
            );
            return;
        }
        // Schedule-template reduction dispatches bind every input stream in
        // `reduction_input_binding_names` order, followed by `dst` and `params`.
        if let Some(kernel) = dispatch.reduction() {
            // Plain RmsNorm (`arity (1, 0, 1)`) carries `eps_bits` in
            // `dispatch.params[2]` for `fuse_rmsnorm_into_add`; keep the
            // reduction template's `round_one_bits` slot at identity `1.0`
            // so single-lane rows are never scaled by eps.
            let n_per_col = kernel.epilogue.as_ref().map_or(0, |e| e.n_per_col_inputs);
            let round_one_bits =
                if kernel.n_per_elem == 1 && kernel.n_per_row == 0 && n_per_col == 1 {
                    1.0f32.to_bits()
                } else {
                    dispatch.params.get(2).copied().unwrap_or(0)
                };
            let params = ReductionParams {
                outer: dispatch.params[0],
                inner: dispatch.params[1],
                round_one_bits,
                table_rows: dispatch.params.get(3).copied().unwrap_or(0),
            };
            match dispatch.input_buffers.as_slice() {
                &[a] => bind_buffers(pc, [buf(a), output()], params),
                &[a, b] => bind_buffers(pc, [buf(a), buf(b), output()], params),
                &[a, b, c] => bind_buffers(pc, [buf(a), buf(b), buf(c), output()], params),
                inputs => {
                    let mut pieces: Vec<blade_graphics::BufferPiece> =
                        inputs.iter().map(|&r| buf(r)).collect();
                    pieces.push(output());
                    pc.bind(
                        0,
                        &DynReductionData {
                            buffers: pieces,
                            params,
                        },
                    );
                }
            }
            return;
        }
        // Schedule-template pointwise dispatches route by DAG arity, not
        // by the dummy `shader` entry — arity may be 3 after fusion, which
        // no ShaderEntry variant represents.
        if let Some(dag) = dispatch.pointwise() {
            let params = UnaryParams {
                len: dispatch.params[0],
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            };
            match dag.n_inputs {
                1 => bind_buffers(pc, [input(0), output()], params),
                2 => bind_buffers(pc, [input(0), input(1), output()], params),
                3 => bind_buffers(pc, [input(0), input(1), input(2), output()], params),
                n => panic!("pointwise arity {n} has no runtime data layout"),
            }
            return;
        }
        // Prologue-fused coop matmul: 2-factor prologue (RmsNorm rsqrt + w_norm).
        // Only applies when use_coop is set AND a prologue is attached.
        if let Some(ref prologue) = dispatch.matmul_prologue
            && dispatch.use_coop()
            && prologue.factors.len() == 2
        {
            let (m, n, k) = dispatch
                .mnk()
                .expect("a fused prologue is attached to a contraction");
            bind_buffers(
                pc,
                [
                    input(0),
                    input(1),
                    output(),
                    buf(prologue.factors[0].0),
                    buf(prologue.factors[1].0),
                ],
                MatMulParams { m, n, k, _pad: 0 },
            );
            return;
        }

        let p = &dispatch.params;
        match dispatch.shader {
            ShaderEntry::Generated => {
                unreachable!("generated kernels are bound by their kernel above")
            }
            ShaderEntry::BlockMatMul
            | ShaderEntry::BlockMatMulAT
            | ShaderEntry::BlockMatMulBT
            | ShaderEntry::BatchMatMul
            | ShaderEntry::BatchMatMulAT
            | ShaderEntry::BatchMatMulBT => {
                let (m, n, k) = mnk();
                bind_buffers(
                    pc,
                    [input(0), input(1), output()],
                    MatMulParams {
                        m,
                        n,
                        k,
                        _pad: p[3],
                    },
                );
            }
            ShaderEntry::MatMul
            | ShaderEntry::MatMulGemv
            | ShaderEntry::MatMulAT
            | ShaderEntry::MatMulBT
            | ShaderEntry::MatMulGemvBT => {
                bind_buffers(pc, [input(0), input(1), output()], dense_params(dispatch));
            }
            ShaderEntry::FusedMatMulAdd
            | ShaderEntry::MatMulGemvAdd
            | ShaderEntry::FusedMatMulATAdd
            | ShaderEntry::FusedMatMulBTAdd
            | ShaderEntry::MatMulGemvBTAdd => {
                bind_buffers(
                    pc,
                    [input(0), input(1), output(), input(2)],
                    dense_params(dispatch),
                );
            }
            ShaderEntry::ToF16 | ShaderEntry::ToBF16 | ShaderEntry::FromBF16 => {
                bind_buffers(
                    pc,
                    [input(0), output()],
                    UnaryParams {
                        len: p[0],
                        _pad0: 0,
                        _pad1: 0,
                        _pad2: 0,
                    },
                );
            }
            ShaderEntry::SwiGLUConcat
            | ShaderEntry::SwiGLUConcatGrad
            | ShaderEntry::GeGLUConcat
            | ShaderEntry::GeGLUConcatGrad => {
                bind_buffers(
                    pc,
                    [input(0), input(1), output()],
                    UnaryParams {
                        len: p[0],
                        _pad0: p[1], // half_n
                        _pad1: 0,
                        _pad2: 0,
                    },
                );
            }
            ShaderEntry::PairwiseGrad => {
                bind_buffers(
                    pc,
                    [input(0), input(1), input(2), output()],
                    UnaryParams {
                        len: p[0],
                        _pad0: p[1],
                        _pad1: p[2],
                        _pad2: p[3],
                    },
                );
            }
            ShaderEntry::SumAll | ShaderEntry::MeanAll => {
                bind_buffers(
                    pc,
                    [input(0), output()],
                    UnaryParams {
                        len: p[0],
                        _pad0: p.get(1).copied().unwrap_or(0), // mean divisor
                        _pad1: 0,
                        _pad2: 0,
                    },
                );
            }
            ShaderEntry::SumRows => {
                // Rows, columns, and optional serial-row layout.
                bind_buffers(
                    pc,
                    [input(0), output()],
                    UnaryParams {
                        len: p[0],   // m
                        _pad0: p[1], // n
                        _pad1: p.get(2).copied().unwrap_or(0),
                        _pad2: p.get(3).copied().unwrap_or(0),
                    },
                );
            }
            ShaderEntry::CrossEntropyLoss | ShaderEntry::CrossEntropyLossIndices => {
                let loss_buf = dispatch
                    .extra_outputs
                    .first()
                    .copied()
                    .unwrap_or(dispatch.output_buffer);
                bind_buffers(
                    pc,
                    [input(0), input(1), output(), buf(loss_buf)],
                    SoftmaxParams {
                        batch: p[0],
                        features: p[1],
                        _pad0: p.get(2).copied().unwrap_or(0),
                        _pad1: 0,
                    },
                );
            }
            ShaderEntry::BceLoss => {
                bind_buffers(
                    pc,
                    [input(0), input(1), output()],
                    UnaryParams {
                        len: p[0],
                        _pad0: 0,
                        _pad1: 0,
                        _pad2: 0,
                    },
                );
            }
            ShaderEntry::Transpose => {
                bind_buffers(
                    pc,
                    [input(0), output()],
                    TransposeParams {
                        m: p[0],
                        n: p[1],
                        _pad0: 0,
                        _pad1: 0,
                    },
                );
            }
            ShaderEntry::RmsNormAdd => {
                bind_buffers(
                    pc,
                    [input(0), input(1), input(2), output()],
                    BiasAddParams {
                        len: p[0],
                        bias_len: p[1],
                        _pad0: p[2], // eps_bits
                        _pad1: 0,
                    },
                );
            }
            ShaderEntry::Embedding => {
                bind_buffers(
                    pc,
                    [input(0), input(1), output()],
                    UnaryParams {
                        len: p[0],
                        _pad0: p[1],
                        _pad1: p[2], // table rows
                        _pad2: 0,
                    },
                );
            }
            ShaderEntry::RoPE | ShaderEntry::RoPEGrad => {
                bind_buffers(
                    pc,
                    [input(0), output()],
                    RoPEParams::from_words(p).expect("invalid RoPE parameters"),
                );
            }
            ShaderEntry::LayerNorm => {
                bind_buffers(
                    pc,
                    [input(0), input(1), input(2), output()],
                    MatMulParams {
                        m: p[0],
                        n: p[1],
                        k: p[2],
                        _pad: p[3],
                    },
                );
            }
            ShaderEntry::MultiHeadAttn
            | ShaderEntry::FlashAttention
            | ShaderEntry::FlashAttentionCoop
            | ShaderEntry::FlashAttentionProducts
            | ShaderEntry::FlashAttentionCoopF32 => {
                bind_buffers(
                    pc,
                    [
                        input(0),
                        input(1),
                        input(2),
                        output(),
                        buf(dispatch.extra_outputs[0]),
                    ],
                    AttentionParams::from_words(p).expect("invalid attention parameters"),
                );
            }
            ShaderEntry::AttentionGrad(AttentionGrad {
                part: AttentionGradPart::KV,
                ..
            }) => {
                bind_buffers(
                    pc,
                    [
                        input(0),
                        input(1),
                        input(2),
                        input(3),
                        input(4),
                        input(5),
                        output(),
                        buf(dispatch.extra_outputs[0]),
                    ],
                    AttentionParams::from_words(p).expect("invalid attention parameters"),
                );
            }
            ShaderEntry::AttentionGrad(AttentionGrad {
                part: AttentionGradPart::Q,
                ..
            }) => {
                bind_buffers(
                    pc,
                    [
                        input(0),
                        input(1),
                        input(2),
                        input(3),
                        input(4),
                        input(5),
                        output(),
                    ],
                    AttentionParams::from_words(p).expect("invalid attention parameters"),
                );
            }
            ShaderEntry::SwiGLUGradGate => {
                bind_buffers(
                    pc,
                    [input(0), input(1), input(2), output()],
                    UnaryParams {
                        len: p[0],
                        _pad0: 0,
                        _pad1: 0,
                        _pad2: 0,
                    },
                );
            }
            ShaderEntry::SwiGLUGradUp | ShaderEntry::SiluGrad => {
                bind_buffers(
                    pc,
                    [input(0), input(1), output()],
                    UnaryParams {
                        len: p[0],
                        _pad0: 0,
                        _pad1: 0,
                        _pad2: 0,
                    },
                );
            }
            ShaderEntry::RmsNormGradW
            | ShaderEntry::RmsNormGradWRowPar
            | ShaderEntry::RmsNormGradX
            | ShaderEntry::LayerNormGradWB
            | ShaderEntry::LayerNormGradX => {
                bind_buffers(
                    pc,
                    [input(0), input(1), input(2), output()],
                    MatMulParams {
                        m: p[0], // rows
                        n: p[1], // cols
                        k: p[2], // eps_bits
                        _pad: p[3],
                    },
                );
            }
            ShaderEntry::RmsNormRsqrt
            | ShaderEntry::SplitA
            | ShaderEntry::SplitB
            | ShaderEntry::Upsample2x
            | ShaderEntry::Upsample2xGrad
            | ShaderEntry::GlobalAvgPoolGrad => {
                bind_buffers(
                    pc,
                    [input(0), output()],
                    UnaryParams {
                        len: p[0],
                        _pad0: p[1],
                        _pad1: p[2],
                        _pad2: p[3],
                    },
                );
            }
            ShaderEntry::SgdUpdate | ShaderEntry::AdamUpdate => {
                unreachable!("optimizer updates are encoded by the optimizer passes")
            }
            ShaderEntry::GradClipNormSq
            | ShaderEntry::GradClipScale
            | ShaderEntry::AdaptiveGradClip
            | ShaderEntry::GradAccum => {
                unreachable!(
                    "Grad-clip/accum shaders are dispatched directly from step(), \
                     not via bind_dispatch"
                );
            }
            ShaderEntry::ScatterAdd => {
                bind_buffers(
                    pc,
                    [input(0), input(1), output()],
                    ScatterAddParams {
                        total: p[0],
                        seq_len: p[1],
                        embed_dim: p[2],
                        _pad: 0,
                    },
                );
            }
            ShaderEntry::ScatterAddAtomic => {
                let row_scale = buf(dispatch
                    .input_buffers
                    .get(2)
                    .copied()
                    .unwrap_or(dispatch.input_buffers[1]));
                bind_buffers(
                    pc,
                    [input(0), input(1), row_scale, output()],
                    ScatterAddParams {
                        total: p[0],
                        seq_len: p[1],
                        embed_dim: p[2],
                        _pad: p[3],
                    },
                );
            }
            ShaderEntry::GroupNorm | ShaderEntry::GroupNormSilu => {
                bind_buffers(
                    pc,
                    [input(0), input(1), input(2), output()],
                    GroupNormParams {
                        batch: p[0],
                        channels: p[1],
                        spatial: p[2],
                        num_groups: p[3],
                        eps_bits: p[4],
                        chunks: 1,
                        apply_silu: 0,
                        _pad2: 0,
                    },
                );
            }
            ShaderEntry::GroupNormStats => {
                bind_buffers(
                    pc,
                    [input(0), output()],
                    GroupNormParams {
                        batch: p[0],
                        channels: p[1],
                        spatial: p[2],
                        num_groups: p[3],
                        eps_bits: p[4],
                        chunks: p[5],
                        apply_silu: 0,
                        _pad2: 0,
                    },
                );
            }
            ShaderEntry::GroupNormApply => {
                bind_buffers(
                    pc,
                    [input(0), input(2), input(3), output(), input(1)],
                    GroupNormParams {
                        batch: p[0],
                        channels: p[1],
                        spatial: p[2],
                        num_groups: p[3],
                        eps_bits: p[4],
                        chunks: p[5],
                        apply_silu: p[6],
                        _pad2: 0,
                    },
                );
            }
            ShaderEntry::GroupNormGradInput
            | ShaderEntry::GroupNormGradWeightBias
            | ShaderEntry::GroupNormGradStats => {
                let inputs = &dispatch.input_buffers;
                // Unused bindings of an entry point take a buffer it reads.
                let (src_a, src_b, bias, stats) = match dispatch.shader {
                    ShaderEntry::GroupNormGradInput => (inputs[0], inputs[1], inputs[2], inputs[3]),
                    ShaderEntry::GroupNormGradWeightBias => {
                        (inputs[0], inputs[1], inputs[1], inputs[2])
                    }
                    _ => (inputs[0], inputs[0], inputs[0], inputs[0]),
                };
                bind_buffers(
                    pc,
                    [buf(src_a), buf(src_b), buf(bias), output(), buf(stats)],
                    GroupNormParams {
                        batch: p[0],
                        channels: p[1],
                        spatial: p[2],
                        num_groups: p[3],
                        eps_bits: p[4],
                        chunks: 1,
                        apply_silu: 0,
                        _pad2: 0,
                    },
                );
            }
            ShaderEntry::Concat => {
                bind_buffers(
                    pc,
                    [input(0), input(1), output()],
                    UnaryParams {
                        len: p[0],
                        _pad0: p[1],
                        _pad1: p[2],
                        _pad2: p[3],
                    },
                );
            }
            ShaderEntry::BiasedAttention => {
                bind_buffers(
                    pc,
                    [input(0), input(1), input(2), input(3), input(4), output()],
                    [p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]],
                );
            }
            ShaderEntry::Permute => {
                bind_buffers(
                    pc,
                    [input(0), output()],
                    PermuteParams {
                        total: p[0],
                        dims: [p[1], p[2], p[3]],
                        strides: [p[4], p[5], p[6], p[7]],
                    },
                );
            }
            ShaderEntry::MulPerChannel => {
                bind_buffers(
                    pc,
                    [input(0), input(1), output()],
                    MulPerChannelParams {
                        len: p[0],
                        spatial: p[1],
                        _pad0: 0,
                        _pad1: 0,
                    },
                );
            }
            ShaderEntry::Conv2dDw => {
                bind_buffers(
                    pc,
                    [input(0), input(1), output()],
                    Conv2dDwParams {
                        batch: p[0],
                        channels: p[1],
                        in_h: p[2],
                        in_w: p[3],
                        kernel_h: p[4],
                        kernel_w: p[5],
                        stride: p[6],
                        padding_h: p[7],
                        out_h: p[8],
                        out_w: p[9],
                        padding_w: p[10],
                        _pad: 0,
                    },
                );
            }
            ShaderEntry::Conv2dGemm
            | ShaderEntry::Conv2dBF16 { .. }
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
            | ShaderEntry::Conv2dGradWeightGemmSplit16 => {
                bind_buffers(
                    pc,
                    [input(0), input(1), output()],
                    Conv2dParams::from(dispatch),
                );
            }
            ShaderEntry::RoPEDynamicFactors => {
                bind_buffers(
                    pc,
                    [input(0), output(), input(1), input(2)],
                    RoPEParams::from_words(p).expect("invalid RoPE parameters"),
                );
            }
            ShaderEntry::RoPEDynamic | ShaderEntry::RoPEPositions | ShaderEntry::AxialRoPE => {
                bind_buffers(
                    pc,
                    [input(0), output(), input(1)],
                    RoPEParams::from_words(p).expect("invalid RoPE parameters"),
                );
            }
            ShaderEntry::CacheWrite => {
                bind_buffers(
                    pc,
                    [input(0), output(), input(2)],
                    UnaryParams {
                        len: p[0],   // dim
                        _pad0: p[1], // cache rows
                        _pad1: 0,
                        _pad2: 0,
                    },
                );
            }
            ShaderEntry::CacheWritePrefix => {
                bind_buffers(
                    pc,
                    [input(0), output(), input(2), input(3)],
                    MatMulParams {
                        m: p[0],
                        n: p[1],
                        k: p[2],
                        _pad: p[3],
                    },
                );
            }
            ShaderEntry::CachedAttention | ShaderEntry::CachedQueryAttention => {
                bind_buffers(
                    pc,
                    [input(0), input(1), input(2), input(3), output()],
                    CachedAttentionParams {
                        queries: p[0],
                        num_heads: p[1],
                        num_kv_heads: p[2],
                        head_dim: p[3],
                        max_seq: p[4],
                        _pad: [0; 3],
                    },
                );
            }
            ShaderEntry::CachedBlockAttention | ShaderEntry::CachedBlockAttentionSplit => {
                bind_buffers(
                    pc,
                    [input(0), input(1), input(2), input(3), input(4), output()],
                    CachedBlockAttentionParams::from_words(p)
                        .expect("cached attention parameter layout"),
                );
            }
            ShaderEntry::CachedBlockAttentionCombine => {
                bind_buffers(
                    pc,
                    [input(0), output()],
                    CachedBlockAttentionParams::from_words(p)
                        .expect("cached attention parameter layout"),
                );
            }
            ShaderEntry::ChunkedRelativeAttention => {
                bind_buffers(
                    pc,
                    [input(0), input(1), input(2), input(3), output()],
                    ChunkedRelativeAttentionParams {
                        seq_len: p[0],
                        num_heads: p[1],
                        head_dim: p[2],
                        left_context: p[3],
                        softcap_bits: p[4],
                        _pad0: 0,
                        _pad1: 0,
                        _pad2: 0,
                    },
                );
            }
            ShaderEntry::PrefixLast => {
                bind_buffers(
                    pc,
                    [input(0), input(1), output()],
                    MatMulParams {
                        m: p[0],
                        n: p[1],
                        k: 0,
                        _pad: 0,
                    },
                );
            }
            ShaderEntry::MaxPool2d => {
                bind_buffers(
                    pc,
                    [input(0), output()],
                    MaxPool2dParams {
                        batch: p[0],
                        channels: p[1],
                        in_h: p[2],
                        in_w: p[3],
                        kernel_h: p[4],
                        kernel_w: p[5],
                        stride: p[6],
                        padding: p[7],
                        out_h: p[8],
                        out_w: p[9],
                        _pad0: p[10],
                        _pad1: p[11],
                    },
                );
            }
            ShaderEntry::MaxPool2dGrad => {
                bind_buffers(
                    pc,
                    [input(0), input(1), output()],
                    MaxPool2dParams {
                        batch: p[0],
                        channels: p[1],
                        in_h: p[2],
                        in_w: p[3],
                        kernel_h: p[4],
                        kernel_w: p[5],
                        stride: p[6],
                        padding: p[7],
                        out_h: p[8],
                        out_w: p[9],
                        _pad0: p[10],
                        _pad1: p[11],
                    },
                );
            }
            ShaderEntry::GlobalAvgPool => {
                bind_buffers(
                    pc,
                    [input(0), output()],
                    GlobalAvgPoolParams {
                        channels: p[0],
                        spatial: p[1],
                        total_out: p[2],
                        _pad: p[3],
                    },
                );
            }
            ShaderEntry::WinogradInputTransform => {
                bind_buffers(
                    pc,
                    [input(0), output()],
                    WinogradInputParams::from_words(p)
                        .expect("Winograd input transform parameters"),
                );
            }
            ShaderEntry::WinogradWeightTransform => {
                bind_buffers(
                    pc,
                    [input(0), output()],
                    WinogradWeightParams::from_words(p)
                        .expect("Winograd weight transform parameters"),
                );
            }
            ShaderEntry::WinogradOutputTransform { fused } => {
                let params = WinogradOutputParams::from_words(p)
                    .expect("Winograd output transform parameters");
                if fused {
                    bind_buffers(pc, [input(0), output(), input(1)], params);
                } else {
                    bind_buffers(pc, [input(0), output()], params);
                }
            }
            ShaderEntry::WinogradBatchedMatMul => {
                bind_buffers(
                    pc,
                    [input(0), input(1), output()],
                    MatMulParams {
                        m: p[0],
                        n: p[1],
                        k: p[2],
                        _pad: p[3],
                    },
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_uniforms_distinguish_typed_splits_from_gemv_columns() {
        for addend in [false, true] {
            for splits in [1, 4, 16] {
                let dispatch = Dispatch {
                    shader: if addend {
                        ShaderEntry::FusedMatMulAdd
                    } else {
                        ShaderEntry::MatMul
                    },
                    kernel: Kernel::CooperativeTyped {
                        types: crate::codegen::MatrixTypes::BF16,
                        splits,
                        specialize: false,
                    },
                    params: vec![128, 2560, 256, 0],
                    workgroups: [4, 8, splits],
                    ..Default::default()
                };
                let params = dense_params(&dispatch);
                assert_eq!(
                    [params.m, params.n, params.k, params._pad],
                    [128, 256, 2560, splits]
                );
            }
        }
        for shader in [
            ShaderEntry::MatMul,
            ShaderEntry::MatMulGemv,
            ShaderEntry::MatMulGemvAdd,
        ] {
            let params = dense_params(&Dispatch {
                shader,
                params: vec![1, 2048, 256],
                workgroups: [1, 8, 1],
                ..Default::default()
            });
            assert_eq!(
                [params.m, params.n, params.k, params._pad],
                [1, 256, 2048, 8]
            );
        }
    }
}
