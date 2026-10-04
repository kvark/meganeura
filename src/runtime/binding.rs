//! Bind the typed operation selected by the compiler.

use super::{
    AttentionParams, BceData, BiasAddParams, BinaryData, CacheWriteData, CacheWritePrefixData,
    CachedAttentionData, CachedBlockAttentionCombineData, CachedBlockAttentionData,
    CachedBlockAttentionParams, ChunkedRelativeAttentionData, ChunkedRelativeAttentionParams,
    Conv2dData, Conv2dDwData, Conv2dDwParams, Conv2dGradInputData, Conv2dGradWeightData,
    Conv2dParams, CrossEntropyData, DynReductionData, EmbeddingData, FourBufData,
    FusedMatMulAddData, GlobalAvgPoolData, GlobalAvgPoolParams, GroupNormApplyData, GroupNormData,
    GroupNormGradData, GroupNormParams, GroupNormStatsData, HorizMatMulData, LayerNormData,
    MatMulData, MatMulParams, MatMulPrologue2Data, MatMulRmsNormData, MatMulRmsNormParams,
    MaxPool2dData, MaxPool2dGradData, MaxPool2dParams, MulPerChannelData, MulPerChannelParams,
    MultiHeadAttnData, MultiHeadAttnGradData, MultiHeadAttnGradKVData, PrefixLastData,
    ReductionParams, ReductionPass1Data, ReductionPass2RowData, RmsNormAddData, RmsNormData,
    RoPEData, RoPEDynamicData, RoPEDynamicFactorsData, RoPEParams, ScatterAddAtomicData,
    ScatterAddData, ScatterAddParams, Session, SoftmaxParams, TernaryData, TransposeData,
    TransposeParams, UnaryData, UnaryParams, WinogradTransformData, WinogradTransformParams,
    reduction_is_dynamic,
};
use crate::compile::{BufferRef, Dispatch, DispatchOp, dispatch};

impl Session {
    pub(super) fn bind_dispatch(
        buffers: &[blade_graphics::BufferPiece],
        dispatch: &Dispatch,
        pc: &mut impl blade_graphics::traits::PipelineEncoder,
    ) {
        let buf = |r: BufferRef| buffers[r.0 as usize];
        match dispatch.op {
            DispatchOp::Matmul(ref op) => Self::bind_matmul(buffers, op, dispatch.workgroups, pc),
            DispatchOp::Convolution(ref op) => Self::bind_convolution(buffers, op, pc),
            DispatchOp::Pointwise(ref op) => Self::bind_pointwise(buffers, op, pc),
            DispatchOp::Reduction(ref op) => Self::bind_reduction(buffers, op, pc),
            DispatchOp::ScatterAddAtomic(ref op) => pc.bind(
                0,
                &ScatterAddAtomicData {
                    indices: buf(op.indices),
                    src: buf(op.src),
                    row_scale: buf(op.row_scale.unwrap_or(op.src)),
                    dst: buf(op.dst),
                    params: ScatterAddParams {
                        total: op.total,
                        seq_len: op.seq_len,
                        embed_dim: op.embed_dim,
                        _pad: if op.row_scale.is_none() {
                            0
                        } else if op.serial_rows {
                            2
                        } else {
                            1
                        },
                    },
                },
            ),
            DispatchOp::Conv2dDw(ref op) => {
                pc.bind(
                    0,
                    &Conv2dDwData {
                        src: buf(op.src),
                        weight: buf(op.weight),
                        dst: buf(op.dst),
                        params: Conv2dDwParams {
                            batch: op.batch,
                            channels: op.channels,
                            in_h: op.in_h,
                            in_w: op.in_w,
                            kernel_h: op.kernel_h,
                            kernel_w: op.kernel_w,
                            stride: op.stride,
                            padding_h: op.padding_h,
                            out_h: op.out_h,
                            out_w: op.out_w,
                            padding_w: op.padding_w,
                            _pad: 0,
                        },
                    },
                );
            }
            DispatchOp::WinogradInputTransform(ref op) => {
                pc.bind(
                    0,
                    &WinogradTransformData {
                        src: buf(op.src),
                        dst: buf(op.dst),
                        params: WinogradTransformParams {
                            p0: op.batch,
                            p1: op.in_channels,
                            p2: op.in_h,
                            p3: op.in_w,
                            p4: op.padding,
                            p5: op.tiles_h,
                            p6: op.tiles_w,
                            p7: op.total_tiles,
                        },
                    },
                );
            }
            DispatchOp::WinogradOutputTransform(ref op) => {
                pc.bind(
                    0,
                    &WinogradTransformData {
                        src: buf(op.src),
                        dst: buf(op.dst),
                        params: WinogradTransformParams {
                            p0: op.batch,
                            p1: op.out_channels,
                            p2: op.out_h,
                            p3: op.out_w,
                            p4: op.tiles_h,
                            p5: op.tiles_w,
                            p6: op.total_tiles,
                            p7: 0,
                        },
                    },
                );
            }
            DispatchOp::WinogradWeightTransform(ref op) => {
                pc.bind(
                    0,
                    &WinogradTransformData {
                        src: buf(op.src),
                        dst: buf(op.dst),
                        params: WinogradTransformParams {
                            p0: op.out_channels,
                            p1: op.in_channels,
                            p2: op.adjoint,
                            p3: 0,
                            p4: 0,
                            p5: 0,
                            p6: 0,
                            p7: 0,
                        },
                    },
                );
            }
            DispatchOp::MaxPool2d(ref op) => {
                pc.bind(
                    0,
                    &MaxPool2dData {
                        src: buf(op.src),
                        dst: buf(op.dst),
                        params: MaxPool2dParams {
                            batch: op.batch,
                            channels: op.channels,
                            in_h: op.in_h,
                            in_w: op.in_w,
                            kernel_h: op.kernel_h,
                            kernel_w: op.kernel_w,
                            stride: op.stride,
                            padding: op.padding,
                            out_h: op.out_h,
                            out_w: op.out_w,
                            _pad0: 0,
                            _pad1: 0,
                        },
                    },
                );
            }
            DispatchOp::MaxPool2dGrad(ref op) => {
                pc.bind(
                    0,
                    &MaxPool2dGradData {
                        grad_out: buf(op.dy),
                        src: buf(op.src),
                        dst: buf(op.dst),
                        params: MaxPool2dParams {
                            batch: op.batch,
                            channels: op.channels,
                            in_h: op.in_h,
                            in_w: op.in_w,
                            kernel_h: op.kernel_h,
                            kernel_w: op.kernel_w,
                            stride: op.stride,
                            padding: op.padding,
                            out_h: op.out_h,
                            out_w: op.out_w,
                            _pad0: 0,
                            _pad1: 0,
                        },
                    },
                );
            }
            DispatchOp::GlobalAvgPool(ref op) => {
                pc.bind(
                    0,
                    &GlobalAvgPoolData {
                        src: buf(op.src),
                        dst: buf(op.dst),
                        params: GlobalAvgPoolParams {
                            channels: op.channels,
                            spatial: op.spatial,
                            total_out: op.total_out,
                            _pad: 0,
                        },
                    },
                );
            }
            DispatchOp::GlobalAvgPoolGrad(ref op) => {
                pc.bind(
                    0,
                    &UnaryData {
                        src: buf(op.src),
                        dst: buf(op.dst),
                        params: UnaryParams {
                            len: op.len,
                            _pad0: op.inner,
                            _pad1: op.mode,
                            _pad2: op.offset,
                        },
                    },
                );
            }
            DispatchOp::Upsample2x(ref op) | DispatchOp::Upsample2xGrad(ref op) => {
                pc.bind(
                    0,
                    &UnaryData {
                        src: buf(op.src),
                        dst: buf(op.dst),
                        params: UnaryParams {
                            len: op.batch,
                            _pad0: op.channels,
                            _pad1: op.in_h,
                            _pad2: op.in_w,
                        },
                    },
                );
            }
            DispatchOp::MultiHeadAttn(ref op)
            | DispatchOp::FlashAttention(ref op)
            | DispatchOp::FlashAttentionCoop(ref op) => {
                pc.bind(
                    0,
                    &MultiHeadAttnData {
                        src_a: buf(op.q),
                        src_b: buf(op.k),
                        bias: buf(op.v),
                        dst: buf(op.dst),
                        lse: buf(op.lse),
                        params: AttentionParams {
                            q_seq: op.q_seq,
                            kv_seq: op.kv_seq,
                            packed_heads: op.packed_heads,
                            head_dim: op.head_dim,
                            window_size: op.window_size,
                            _pad0: 0,
                            _pad1: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            DispatchOp::MultiHeadAttnGradKV(ref op)
            | DispatchOp::FlashGradKV(ref op)
            | DispatchOp::FlashGradKVCoop(ref op) => {
                pc.bind(
                    0,
                    &MultiHeadAttnGradKVData {
                        d_out: buf(op.d_out),
                        src_a: buf(op.q),
                        src_b: buf(op.k),
                        bias: buf(op.v),
                        lse: buf(op.lse),
                        fwd_dst: buf(op.row_source),
                        dst: buf(op.dk),
                        dst2: buf(op.dv),
                        params: AttentionParams {
                            q_seq: op.q_seq,
                            kv_seq: op.kv_seq,
                            packed_heads: op.packed_heads,
                            head_dim: op.head_dim,
                            window_size: op.window_size,
                            _pad0: 0,
                            _pad1: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            DispatchOp::MultiHeadAttnGradQ(ref op)
            | DispatchOp::FlashGradQ(ref op)
            | DispatchOp::FlashGradQCoop(ref op) => {
                pc.bind(
                    0,
                    &MultiHeadAttnGradData {
                        d_out: buf(op.d_out),
                        src_a: buf(op.q),
                        src_b: buf(op.k),
                        bias: buf(op.v),
                        lse: buf(op.lse),
                        fwd_dst: buf(op.row_source),
                        dst: buf(op.dst),
                        params: AttentionParams {
                            q_seq: op.q_seq,
                            kv_seq: op.kv_seq,
                            packed_heads: op.packed_heads,
                            head_dim: op.head_dim,
                            window_size: op.window_size,
                            _pad0: 0,
                            _pad1: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            DispatchOp::CachedAttention(ref op) | DispatchOp::CachedQueryAttention(ref op) => {
                pc.bind(
                    0,
                    &CachedAttentionData {
                        src_a: buf(op.q),             // Q
                        src_b: buf(op.k_cache),       // K cache
                        bias: buf(op.v_cache),        // V cache
                        kv_pos_buf: buf(op.position), // kv_pos
                        dst: buf(op.dst),
                        params: MatMulParams {
                            m: op.q_seq,
                            n: op.num_heads,
                            k: op.num_kv_heads,
                            _pad: op.head_dim,
                        },
                    },
                );
            }
            DispatchOp::CachedBlockAttention(ref op)
            | DispatchOp::CachedBlockAttentionSplit(ref op) => {
                pc.bind(
                    0,
                    &CachedBlockAttentionData {
                        src_a: buf(op.q),
                        src_b: buf(op.k_cache),
                        bias: buf(op.v_cache),
                        kv_pos_buf: buf(op.position),
                        valid_len_buf: buf(op.valid_len),
                        dst: buf(op.dst),
                        params: CachedBlockAttentionParams {
                            window_size: op.window_size,
                            num_heads: op.num_heads,
                            num_kv_heads: op.num_kv_heads,
                            head_dim: op.head_dim,
                            block_len: op.block_len,
                            max_seq: op.max_seq,
                            splits: op.splits,
                            _pad: 0,
                        },
                    },
                );
            }
            DispatchOp::CachedBlockAttentionCombine(ref op) => {
                pc.bind(
                    0,
                    &CachedBlockAttentionCombineData {
                        partials: buf(op.partials),
                        dst: buf(op.dst),
                        params: CachedBlockAttentionParams {
                            window_size: op.window_size,
                            num_heads: op.num_heads,
                            num_kv_heads: op.num_kv_heads,
                            head_dim: op.head_dim,
                            block_len: op.block_len,
                            max_seq: op.max_seq,
                            splits: op.splits,
                            _pad: 0,
                        },
                    },
                );
            }
            DispatchOp::ChunkedRelativeAttention(ref op) => {
                pc.bind(
                    0,
                    &ChunkedRelativeAttentionData {
                        src_a: buf(op.q),
                        src_b: buf(op.k),
                        bias: buf(op.v),
                        relative_k: buf(op.relative_k),
                        dst: buf(op.dst),
                        params: ChunkedRelativeAttentionParams {
                            seq_len: op.seq_len,
                            num_heads: op.num_heads,
                            head_dim: op.head_dim,
                            left_context: op.left_context,
                            softcap_bits: op.softcap_bits,
                            _pad0: 0,
                            _pad1: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            DispatchOp::PrefixLast(ref op) => {
                pc.bind(
                    0,
                    &PrefixLastData {
                        src: buf(op.src),
                        valid_len_buf: buf(op.valid_len),
                        dst: buf(op.dst),
                        params: MatMulParams {
                            m: op.cols,
                            n: op.rows,
                            k: 0,
                            _pad: 0,
                        },
                    },
                );
            }
            DispatchOp::CacheWrite(ref op) => {
                pc.bind(
                    0,
                    &CacheWriteData {
                        src: buf(op.src),
                        dst: buf(op.cache),
                        kv_pos_buf: buf(op.position),
                        params: UnaryParams {
                            len: op.dim, // dim
                            _pad0: 0,
                            _pad1: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            DispatchOp::CacheWritePrefix(ref op) => {
                pc.bind(
                    0,
                    &CacheWritePrefixData {
                        src: buf(op.src),
                        dst: buf(op.cache),
                        kv_pos_buf: buf(op.position),
                        valid_len_buf: buf(op.valid_len),
                        params: MatMulParams {
                            m: op.dim,
                            n: op.block_len,
                            k: op.max_seq,
                            _pad: 0,
                        },
                    },
                );
            }
            DispatchOp::RmsNormAdd(ref op) => {
                pc.bind(
                    0,
                    &RmsNormAddData {
                        src: buf(op.src),
                        bias: buf(op.weight),
                        residual: buf(op.residual),
                        dst: buf(op.dst),
                        params: BiasAddParams {
                            len: op.rows,
                            bias_len: op.cols,
                            _pad0: op.eps_bits, // eps_bits
                            _pad1: 0,
                        },
                    },
                );
            }
            DispatchOp::LayerNorm(ref op) => {
                pc.bind(
                    0,
                    &LayerNormData {
                        src: buf(op.src),
                        src_b: buf(op.weight),
                        bias: buf(op.bias),
                        dst: buf(op.dst),
                        params: MatMulParams {
                            m: op.rows,
                            n: op.cols,
                            k: op.eps_bits,
                            _pad: op.block_rows,
                        },
                    },
                );
            }
            DispatchOp::RmsNormGradW(ref op)
            | DispatchOp::RmsNormGradWRowPar(ref op)
            | DispatchOp::RmsNormGradX(ref op)
            | DispatchOp::LayerNormGradWB(ref op)
            | DispatchOp::LayerNormGradX(ref op) => {
                pc.bind(
                    0,
                    &FourBufData {
                        src_a: buf(op.dy),    // dy
                        src_b: buf(op.src),   // x
                        bias: buf(op.weight), // w
                        dst: buf(op.dst),
                        params: MatMulParams {
                            m: op.rows,     // rows
                            n: op.cols,     // cols
                            k: op.eps_bits, // eps_bits
                            _pad: op.block_rows,
                        },
                    },
                );
            }
            DispatchOp::RmsNormRsqrt(ref op) => {
                // bindings: src=X, dst=rsqrt, params=(rows, cols, eps_bits, _pad)
                pc.bind(
                    0,
                    &UnaryData {
                        src: buf(op.src),
                        dst: buf(op.dst),
                        params: UnaryParams {
                            len: op.rows,
                            _pad0: op.cols,
                            _pad1: op.eps_bits,
                            _pad2: 0,
                        },
                    },
                );
            }
            DispatchOp::GroupNorm(ref op) | DispatchOp::GroupNormSilu(ref op) => {
                pc.bind(
                    0,
                    &GroupNormData {
                        src: buf(op.src),
                        src_b: buf(op.weight),
                        bias: buf(op.bias),
                        dst: buf(op.dst),
                        params: GroupNormParams {
                            batch: op.batch,
                            channels: op.channels,
                            spatial: op.spatial,
                            num_groups: op.num_groups,
                            eps_bits: op.eps_bits,
                            chunks: 1,
                            apply_silu: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            DispatchOp::GroupNormStats(ref op) => {
                pc.bind(
                    0,
                    &GroupNormStatsData {
                        src: buf(op.src),
                        dst: buf(op.dst),
                        params: GroupNormParams {
                            batch: op.batch,
                            channels: op.channels,
                            spatial: op.spatial,
                            num_groups: op.num_groups,
                            eps_bits: op.eps_bits,
                            chunks: op.chunks,
                            apply_silu: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            DispatchOp::GroupNormApply(ref op) => {
                pc.bind(
                    0,
                    &GroupNormApplyData {
                        src: buf(op.src),
                        src_b: buf(op.weight),
                        bias: buf(op.bias),
                        dst: buf(op.dst),
                        partials: buf(op.partials),
                        params: GroupNormParams {
                            batch: op.batch,
                            channels: op.channels,
                            spatial: op.spatial,
                            num_groups: op.num_groups,
                            eps_bits: op.eps_bits,
                            chunks: op.chunks,
                            apply_silu: op.apply_silu,
                            _pad2: 0,
                        },
                    },
                );
            }
            DispatchOp::GroupNormGradInput(ref op) => {
                // Unused bindings of an entry point take a buffer it reads.
                let (src_a, src_b, bias, stats) = (op.dy, op.src, op.weight, op.stats);
                pc.bind(
                    0,
                    &GroupNormGradData {
                        src_a: buf(src_a),
                        src_b: buf(src_b),
                        bias: buf(bias),
                        dst: buf(op.dst),
                        stats: buf(stats),
                        params: GroupNormParams {
                            batch: op.batch,
                            channels: op.channels,
                            spatial: op.spatial,
                            num_groups: op.num_groups,
                            eps_bits: op.eps_bits,
                            chunks: 1,
                            apply_silu: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            DispatchOp::GroupNormGradWeightBias(ref op) => {
                // Unused bindings of an entry point take a buffer it reads.
                let (src_a, src_b, bias, stats) = (op.dy, op.src, op.src, op.stats);
                pc.bind(
                    0,
                    &GroupNormGradData {
                        src_a: buf(src_a),
                        src_b: buf(src_b),
                        bias: buf(bias),
                        dst: buf(op.dst),
                        stats: buf(stats),
                        params: GroupNormParams {
                            batch: op.batch,
                            channels: op.channels,
                            spatial: op.spatial,
                            num_groups: op.num_groups,
                            eps_bits: op.eps_bits,
                            chunks: 1,
                            apply_silu: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            DispatchOp::GroupNormGradStats(ref op) => {
                // Unused bindings of an entry point take a buffer it reads.
                let (src_a, src_b, bias, stats) = (op.src, op.src, op.src, op.src);
                pc.bind(
                    0,
                    &GroupNormGradData {
                        src_a: buf(src_a),
                        src_b: buf(src_b),
                        bias: buf(bias),
                        dst: buf(op.dst),
                        stats: buf(stats),
                        params: GroupNormParams {
                            batch: op.batch,
                            channels: op.channels,
                            spatial: op.spatial,
                            num_groups: op.num_groups,
                            eps_bits: op.eps_bits,
                            chunks: 1,
                            apply_silu: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            DispatchOp::SwiGLUConcat(ref op)
            | DispatchOp::SwiGLUConcatGrad(ref op)
            | DispatchOp::GeGLUConcat(ref op)
            | DispatchOp::GeGLUConcatGrad(ref op) => {
                pc.bind(
                    0,
                    &BinaryData {
                        src_a: buf(op.src_a),
                        src_b: buf(op.src_b),
                        dst: buf(op.dst),
                        params: UnaryParams {
                            len: op.len,
                            _pad0: op.half_width, // half_n
                            _pad1: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            DispatchOp::SwiGLUGradGate(ref op) => {
                pc.bind(
                    0,
                    &TernaryData {
                        src_a: buf(op.dy),   // grad_out
                        src_b: buf(op.gate), // gate
                        src_c: buf(op.up),   // up
                        dst: buf(op.dst),
                        params: UnaryParams {
                            len: op.len,
                            _pad0: 0,
                            _pad1: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            DispatchOp::SwiGLUGradUp(ref op) => {
                pc.bind(
                    0,
                    &BinaryData {
                        src_a: buf(op.dy),  // grad_out
                        src_b: buf(op.src), // gate
                        dst: buf(op.dst),
                        params: UnaryParams {
                            len: op.len,
                            _pad0: 0,
                            _pad1: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            DispatchOp::SiluGrad(ref op) => {
                pc.bind(
                    0,
                    &BinaryData {
                        src_a: buf(op.dy),  // grad_out
                        src_b: buf(op.src), // x
                        dst: buf(op.dst),
                        params: UnaryParams {
                            len: op.len,
                            _pad0: 0,
                            _pad1: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            DispatchOp::PairwiseGrad(ref op) => {
                pc.bind(
                    0,
                    &TernaryData {
                        src_a: buf(op.gradient),
                        src_b: buf(op.a),
                        src_c: buf(op.b),
                        dst: buf(op.dst),
                        params: UnaryParams {
                            len: op.total,
                            _pad0: op.inner,
                            _pad1: op.pairs,
                            _pad2: op.mode,
                        },
                    },
                );
            }
            DispatchOp::SumAll(ref op) | DispatchOp::MeanAll(ref op) => {
                pc.bind(
                    0,
                    &UnaryData {
                        src: buf(op.src),
                        dst: buf(op.dst),
                        params: UnaryParams {
                            len: op.len,
                            _pad0: op.divisor, // mean divisor
                            _pad1: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            DispatchOp::SumRows(ref op) => {
                // Rows, columns, and optional serial-row layout.
                pc.bind(
                    0,
                    &UnaryData {
                        src: buf(op.src),
                        dst: buf(op.dst),
                        params: UnaryParams {
                            len: op.rows,   // m
                            _pad0: op.cols, // n
                            _pad1: op.serial_rows,
                            _pad2: op.splits,
                        },
                    },
                );
            }
            DispatchOp::CrossEntropyLoss(ref op) => {
                pc.bind(
                    0,
                    &CrossEntropyData {
                        logits: buf(op.logits),
                        labels: buf(op.labels),
                        grad_out: buf(op.gradient),
                        loss_out: buf(op.loss),
                        params: SoftmaxParams {
                            batch: op.batch,
                            features: op.features,
                            _pad0: op.write_grad,
                            _pad1: 0,
                        },
                    },
                );
            }
            DispatchOp::BceLoss(ref op) => {
                pc.bind(
                    0,
                    &BceData {
                        pred: buf(op.prediction),
                        labels: buf(op.labels),
                        loss_out: buf(op.dst),
                        params: UnaryParams {
                            len: op.len,
                            _pad0: 0,
                            _pad1: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            DispatchOp::ToF16(ref op) => {
                pc.bind(
                    0,
                    &UnaryData {
                        src: buf(op.src),
                        dst: buf(op.dst),
                        params: UnaryParams {
                            len: op.len,
                            _pad0: 0,
                            _pad1: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            DispatchOp::Transpose(ref op) => {
                pc.bind(
                    0,
                    &TransposeData {
                        src: buf(op.src),
                        dst: buf(op.dst),
                        params: TransposeParams {
                            m: op.rows,
                            n: op.cols,
                            _pad0: 0,
                            _pad1: 0,
                        },
                    },
                );
            }
            DispatchOp::Embedding(ref op) => {
                pc.bind(
                    0,
                    &EmbeddingData {
                        indices: buf(op.indices),
                        src: buf(op.table),
                        dst: buf(op.dst),
                        params: UnaryParams {
                            len: op.rows,
                            _pad0: op.embed_dim,
                            _pad1: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            DispatchOp::RoPE(ref op) | DispatchOp::RoPEGrad(ref op) => {
                pc.bind(
                    0,
                    &RoPEData {
                        src: buf(op.src),
                        dst: buf(op.dst),
                        params: RoPEParams {
                            seq: op.seq,
                            dim: op.dim,
                            theta_bits: op.theta_bits,
                            pos_offset: op.pos_offset,
                            head_dim: op.head_dim,
                            _pad0: 0,
                            _pad1: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            DispatchOp::RoPEDynamicFactors(ref op) => {
                pc.bind(
                    0,
                    &RoPEDynamicFactorsData {
                        src: buf(op.src),
                        dst: buf(op.dst),
                        pos_offset_buf: buf(op.position),
                        factors: buf(op.factors),
                        params: RoPEParams {
                            seq: op.seq,
                            dim: op.dim,
                            theta_bits: op.theta_bits,
                            pos_offset: op.pos_offset,
                            head_dim: op.head_dim,
                            _pad0: 0,
                            _pad1: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            DispatchOp::RoPEDynamic(ref op) | DispatchOp::RoPEPositions(ref op) => {
                pc.bind(
                    0,
                    &RoPEDynamicData {
                        src: buf(op.src),
                        dst: buf(op.dst),
                        pos_offset_buf: buf(op.position),
                        params: RoPEParams {
                            seq: op.seq,
                            dim: op.dim,
                            theta_bits: op.theta_bits,
                            pos_offset: op.pos_offset,
                            head_dim: op.head_dim,
                            _pad0: 0,
                            _pad1: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            DispatchOp::Concat(ref op) => {
                pc.bind(
                    0,
                    &BinaryData {
                        src_a: buf(op.a),
                        src_b: buf(op.b),
                        dst: buf(op.dst),
                        params: UnaryParams {
                            len: op.batch,
                            _pad0: op.channels_a,
                            _pad1: op.channels_b,
                            _pad2: op.spatial,
                        },
                    },
                );
            }
            DispatchOp::SplitA(ref op) | DispatchOp::SplitB(ref op) => {
                pc.bind(
                    0,
                    &UnaryData {
                        src: buf(op.src),
                        dst: buf(op.dst),
                        params: UnaryParams {
                            len: op.batch,
                            _pad0: op.channels_a,
                            _pad1: op.channels_b,
                            _pad2: op.spatial,
                        },
                    },
                );
            }
            DispatchOp::MulPerChannel(ref op) => {
                pc.bind(
                    0,
                    &MulPerChannelData {
                        src: buf(op.src),
                        gate: buf(op.gate),
                        dst: buf(op.dst),
                        params: MulPerChannelParams {
                            len: op.len,
                            spatial: op.spatial,
                            _pad0: 0,
                            _pad1: 0,
                        },
                    },
                );
            }
            DispatchOp::ScatterAdd(ref op) => {
                pc.bind(
                    0,
                    &ScatterAddData {
                        indices: buf(op.indices),
                        src: buf(op.src),
                        dst: buf(op.dst),
                        params: ScatterAddParams {
                            total: op.total,
                            seq_len: op.seq_len,
                            embed_dim: op.embed_dim,
                            _pad: 0,
                        },
                    },
                );
            }
        }
    }

    fn bind_matmul(
        buffers: &[blade_graphics::BufferPiece],
        op: &dispatch::Matmul,
        workgroups: [u32; 3],
        pc: &mut impl blade_graphics::traits::PipelineEncoder,
    ) {
        let buf = |r: BufferRef| buffers[r.0 as usize];
        if !op.siblings.is_empty() {
            let mut pieces = vec![buf(op.a), buf(op.b)];
            pieces.extend(op.siblings.iter().map(|&(b, _)| buf(b)));
            pieces.push(buf(op.dst));
            pieces.extend(op.siblings.iter().map(|&(_, c)| buf(c)));
            pc.bind(
                0,
                &HorizMatMulData {
                    buffers: pieces,
                    params: MatMulParams {
                        m: op.m,
                        n: op.n,
                        k: op.k,
                        _pad: 0,
                    },
                },
            );
            return;
        }
        if let Some(ref norm) = op.rmsnorm {
            pc.bind(
                0,
                &MatMulRmsNormData {
                    matrix_a: buf(op.a),
                    norm_w: buf(norm.weight),
                    matrix_b: buf(op.b),
                    matrix_c: buf(op.dst),
                    params: MatMulRmsNormParams {
                        m: op.m,
                        n: op.n,
                        k: op.k,
                        eps_bits: norm.eps_bits,
                    },
                },
            );
            return;
        }
        if let Some(ref prologue) = op.prologue {
            if matches!(
                op.implementation,
                dispatch::MatmulImplementation::Cooperative
                    | dispatch::MatmulImplementation::CooperativeCompensated
            ) && prologue.factors.len() == 2
            {
                pc.bind(
                    0,
                    &MatMulPrologue2Data {
                        matrix_a: buf(op.a),
                        matrix_b: buf(op.b),
                        matrix_c: buf(op.dst),
                        prologue_buf_0: buf(prologue.factors[0].0),
                        prologue_buf_1: buf(prologue.factors[1].0),
                        params: MatMulParams {
                            m: op.m,
                            n: op.n,
                            k: op.k,
                            _pad: 0,
                        },
                    },
                );
                return;
            }
        }
        let pad = match op.kind {
            dispatch::MatmulKind::Block { batches }
            | dispatch::MatmulKind::BlockAT { batches }
            | dispatch::MatmulKind::BlockBT { batches } => batches,
            dispatch::MatmulKind::Winograd { planes } => planes,
            _ => workgroups[1],
        };
        let params = MatMulParams {
            m: op.m,
            n: op.n,
            k: op.k,
            _pad: pad,
        };
        if let Some(addend) = op.addend() {
            pc.bind(
                0,
                &FusedMatMulAddData {
                    matrix_a: buf(op.a),
                    matrix_b: buf(op.b),
                    matrix_c: buf(op.dst),
                    src: buf(addend),
                    params,
                },
            );
        } else {
            pc.bind(
                0,
                &MatMulData {
                    matrix_a: buf(op.a),
                    matrix_b: buf(op.b),
                    matrix_c: buf(op.dst),
                    params,
                },
            );
        }
    }

    fn bind_convolution(
        buffers: &[blade_graphics::BufferPiece],
        op: &dispatch::Convolution,
        pc: &mut impl blade_graphics::traits::PipelineEncoder,
    ) {
        let buf = |r: BufferRef| buffers[r.0 as usize];
        let params = Conv2dParams::from(op);
        match op.kind {
            dispatch::ConvolutionKind::Forward => pc.bind(
                0,
                &Conv2dData {
                    src: buf(op.args.a),
                    weight: buf(op.args.b),
                    dst: buf(op.args.dst),
                    params,
                },
            ),
            dispatch::ConvolutionKind::InputGradient => pc.bind(
                0,
                &Conv2dGradInputData {
                    grad_out: buf(op.args.a),
                    weight: buf(op.args.b),
                    dst: buf(op.args.dst),
                    params,
                },
            ),
            dispatch::ConvolutionKind::WeightGradient => pc.bind(
                0,
                &Conv2dGradWeightData {
                    grad_out: buf(op.args.a),
                    src: buf(op.args.b),
                    dst: buf(op.args.dst),
                    params,
                },
            ),
        }
    }
    fn bind_pointwise(
        buffers: &[blade_graphics::BufferPiece],
        op: &dispatch::Pointwise,
        pc: &mut impl blade_graphics::traits::PipelineEncoder,
    ) {
        let buf = |r: BufferRef| buffers[r.0 as usize];
        let dag = &op.dag;

        let params = UnaryParams {
            len: op.len,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
        };
        match dag.n_inputs {
            1 => {
                pc.bind(
                    0,
                    &UnaryData {
                        src: buf(op.inputs[0]),
                        dst: buf(op.dst),
                        params,
                    },
                );
            }
            2 => {
                pc.bind(
                    0,
                    &BinaryData {
                        src_a: buf(op.inputs[0]),
                        src_b: buf(op.inputs[1]),
                        dst: buf(op.dst),
                        params,
                    },
                );
            }
            3 => {
                pc.bind(
                    0,
                    &TernaryData {
                        src_a: buf(op.inputs[0]),
                        src_b: buf(op.inputs[1]),
                        src_c: buf(op.inputs[2]),
                        dst: buf(op.dst),
                        params,
                    },
                );
            }
            n => panic!("pointwise arity {} has no runtime data layout", n),
        }
    }
    fn bind_reduction(
        buffers: &[blade_graphics::BufferPiece],
        op: &dispatch::Reduction,
        pc: &mut impl blade_graphics::traits::PipelineEncoder,
    ) {
        let buf = |r: BufferRef| buffers[r.0 as usize];
        let k = &op.kernel;

        let params = ReductionParams {
            outer: op.outer,
            inner: op.inner,
            round_one_bits: op.round_one_bits,
            _pad1: 0,
        };
        if reduction_is_dynamic(k) {
            // Buffers in binding order: each input stream (gather idx
            // buffers are already interleaved into `input_buffers` by
            // the fusion pass, right after their table stream), then
            // `dst`. `params` is bound last by `fill`.
            let mut buffers: Vec<blade_graphics::BufferPiece> =
                op.inputs.iter().map(|&r| buf(r)).collect();
            buffers.push(buf(op.dst));
            pc.bind(0, &DynReductionData { buffers, params });
            return;
        }
        let n_per_col = k.epilogue.as_ref().map_or(0, |e| e.n_per_col_inputs);
        match (k.n_per_elem, k.n_per_row, n_per_col) {
            (1, 0, 0) => {
                pc.bind(
                    0,
                    &ReductionPass1Data {
                        src: buf(op.inputs[0]),
                        dst: buf(op.dst),
                        params,
                    },
                );
            }
            (1, 1, 0) => {
                pc.bind(
                    0,
                    &ReductionPass2RowData {
                        src: buf(op.inputs[0]),
                        per_row_src: buf(op.inputs[1]),
                        dst: buf(op.dst),
                        params,
                    },
                );
            }
            (1, 0, 1) => {
                pc.bind(
                    0,
                    &RmsNormData {
                        src: buf(op.inputs[0]),
                        bias: buf(op.inputs[1]),
                        dst: buf(op.dst),
                        params: BiasAddParams {
                            len: params.outer,
                            bias_len: params.inner,
                            _pad0: 0,
                            _pad1: 0,
                        },
                    },
                );
            }
            other => {
                panic!(
                    "reduction kernel with arity {:?} has no runtime binding layout",
                    other
                )
            }
        }
    }
}
