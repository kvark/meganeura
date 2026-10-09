//! Bind dispatch operands and uniforms to their shader layout.

use super::{
    AttentionParams, BceData, BiasAddParams, BinaryData, CacheWriteData, CacheWritePrefixData,
    CachedAttentionData, CachedAttentionParams, CachedBlockAttentionCombineData,
    CachedBlockAttentionData, ChunkedRelativeAttentionData, ChunkedRelativeAttentionParams,
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
use crate::compile::{BufferRef, CachedBlockAttentionParams, Dispatch, ShaderEntry};
use crate::kernels::attention_grad::{AttentionGrad, Part as AttentionGradPart};

impl Session {
    pub(super) fn bind_dispatch(
        buffers: &[blade_graphics::BufferPiece],
        dispatch: &Dispatch,
        pc: &mut impl blade_graphics::traits::PipelineEncoder,
    ) {
        let buf = |r: BufferRef| buffers[r.0 as usize];
        let mnk = || {
            dispatch
                .mnk()
                .unwrap_or_else(|| panic!("missing matrix dimensions for {:?}", dispatch.shader))
        };
        if dispatch.horizontal_batch >= 2 {
            let count = dispatch.horizontal_batch as usize;
            let mut pieces = vec![buf(dispatch.input_buffers[0])];
            for i in 0..count {
                pieces.push(buf(dispatch.input_buffers[1 + i]));
            }
            pieces.push(buf(dispatch.output_buffer));
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
            pc.bind(
                0,
                &MatMulRmsNormData {
                    matrix_a: buf(dispatch.input_buffers[0]),
                    norm_w: buf(rn.weight),
                    matrix_b: buf(dispatch.input_buffers[1]),
                    matrix_c: buf(dispatch.output_buffer),
                    params: MatMulRmsNormParams {
                        m,
                        n,
                        k,
                        eps_bits: rn.eps_bits,
                    },
                },
            );
            return;
        }
        // Schedule-template reduction dispatches have priority and route
        // by kernel arity (n_per_elem, n_per_row, n_per_col).
        if let Some(k) = dispatch.reduction() {
            let params = ReductionParams {
                outer: dispatch.params[0],
                inner: dispatch.params[1],
                round_one_bits: dispatch.params.get(2).copied().unwrap_or(0),
                table_rows: dispatch.params.get(3).copied().unwrap_or(0),
            };
            if reduction_is_dynamic(k) {
                // Buffers in binding order: each input stream (gather idx
                // buffers are already interleaved into `input_buffers` by
                // the fusion pass, right after their table stream), then
                // `dst`. `params` is bound last by `fill`.
                let mut buffers: Vec<blade_graphics::BufferPiece> =
                    dispatch.input_buffers.iter().map(|&r| buf(r)).collect();
                buffers.push(buf(dispatch.output_buffer));
                pc.bind(0, &DynReductionData { buffers, params });
                return;
            }
            let n_per_col = k.epilogue.as_ref().map_or(0, |e| e.n_per_col_inputs);
            match (k.n_per_elem, k.n_per_row, n_per_col) {
                (1, 0, 0) => {
                    pc.bind(
                        0,
                        &ReductionPass1Data {
                            src: buf(dispatch.input_buffers[0]),
                            dst: buf(dispatch.output_buffer),
                            params,
                        },
                    );
                }
                (1, 1, 0) => {
                    pc.bind(
                        0,
                        &ReductionPass2RowData {
                            src: buf(dispatch.input_buffers[0]),
                            per_row_src: buf(dispatch.input_buffers[1]),
                            dst: buf(dispatch.output_buffer),
                            params,
                        },
                    );
                }
                (1, 0, 1) => {
                    pc.bind(
                        0,
                        &RmsNormData {
                            src: buf(dispatch.input_buffers[0]),
                            bias: buf(dispatch.input_buffers[1]),
                            dst: buf(dispatch.output_buffer),
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
                1 => {
                    pc.bind(
                        0,
                        &UnaryData {
                            src: buf(dispatch.input_buffers[0]),
                            dst: buf(dispatch.output_buffer),
                            params,
                        },
                    );
                }
                2 => {
                    pc.bind(
                        0,
                        &BinaryData {
                            src_a: buf(dispatch.input_buffers[0]),
                            src_b: buf(dispatch.input_buffers[1]),
                            dst: buf(dispatch.output_buffer),
                            params,
                        },
                    );
                }
                3 => {
                    pc.bind(
                        0,
                        &TernaryData {
                            src_a: buf(dispatch.input_buffers[0]),
                            src_b: buf(dispatch.input_buffers[1]),
                            src_c: buf(dispatch.input_buffers[2]),
                            dst: buf(dispatch.output_buffer),
                            params,
                        },
                    );
                }
                n => panic!("pointwise arity {} has no runtime data layout", n),
            }
            return;
        }
        // Prologue-fused coop matmul: 2-factor prologue (RmsNorm rsqrt + w_norm).
        // Only applies when use_coop is set AND a prologue is attached.
        if let Some(ref prologue) = dispatch.matmul_prologue {
            if dispatch.use_coop() && prologue.factors.len() == 2 {
                let (m, n, k) = dispatch
                    .mnk()
                    .expect("a fused prologue is attached to a contraction");
                pc.bind(
                    0,
                    &MatMulPrologue2Data {
                        matrix_a: buf(dispatch.input_buffers[0]),
                        matrix_b: buf(dispatch.input_buffers[1]),
                        matrix_c: buf(dispatch.output_buffer),
                        prologue_buf_0: buf(prologue.factors[0].0),
                        prologue_buf_1: buf(prologue.factors[1].0),
                        params: MatMulParams { m, n, k, _pad: 0 },
                    },
                );
                return;
            }
        }

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
                pc.bind(
                    0,
                    &MatMulData {
                        matrix_a: buf(dispatch.input_buffers[0]),
                        matrix_b: buf(dispatch.input_buffers[1]),
                        matrix_c: buf(dispatch.output_buffer),
                        params: MatMulParams {
                            m,
                            n,
                            k,
                            _pad: dispatch.params[3],
                        },
                    },
                );
            }
            ShaderEntry::MatMul | ShaderEntry::MatMulGemv => {
                let (m, n, k) = mnk();
                pc.bind(
                    0,
                    &MatMulData {
                        matrix_a: buf(dispatch.input_buffers[0]),
                        matrix_b: buf(dispatch.input_buffers[1]),
                        matrix_c: buf(dispatch.output_buffer),
                        params: MatMulParams {
                            m,
                            n,
                            k,
                            _pad: dispatch.workgroups[1],
                        },
                    },
                );
            }
            ShaderEntry::MatMulAT | ShaderEntry::MatMulBT | ShaderEntry::MatMulGemvBT => {
                let (m, n, k) = mnk();
                pc.bind(
                    0,
                    &MatMulData {
                        matrix_a: buf(dispatch.input_buffers[0]),
                        matrix_b: buf(dispatch.input_buffers[1]),
                        matrix_c: buf(dispatch.output_buffer),
                        params: MatMulParams {
                            m,
                            n,
                            k,
                            _pad: dispatch.workgroups[1],
                        },
                    },
                );
            }
            ShaderEntry::FusedMatMulAdd | ShaderEntry::MatMulGemvAdd => {
                let (m, n, k) = mnk();
                pc.bind(
                    0,
                    &FusedMatMulAddData {
                        matrix_a: buf(dispatch.input_buffers[0]),
                        matrix_b: buf(dispatch.input_buffers[1]),
                        matrix_c: buf(dispatch.output_buffer),
                        src: buf(dispatch.input_buffers[2]), // addend
                        params: MatMulParams {
                            m,
                            n,
                            k,
                            _pad: dispatch.workgroups[1],
                        },
                    },
                );
            }
            ShaderEntry::FusedMatMulATAdd
            | ShaderEntry::FusedMatMulBTAdd
            | ShaderEntry::MatMulGemvBTAdd => {
                let (m, n, k) = mnk();
                pc.bind(
                    0,
                    &FusedMatMulAddData {
                        matrix_a: buf(dispatch.input_buffers[0]),
                        matrix_b: buf(dispatch.input_buffers[1]),
                        matrix_c: buf(dispatch.output_buffer),
                        src: buf(dispatch.input_buffers[2]), // addend
                        params: MatMulParams {
                            m,
                            n,
                            k,
                            _pad: dispatch.workgroups[1],
                        },
                    },
                );
            }
            ShaderEntry::ToF16 => {
                pc.bind(
                    0,
                    &UnaryData {
                        src: buf(dispatch.input_buffers[0]),
                        dst: buf(dispatch.output_buffer),
                        params: UnaryParams {
                            len: dispatch.params[0],
                            _pad0: 0,
                            _pad1: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            ShaderEntry::SwiGLUConcat
            | ShaderEntry::SwiGLUConcatGrad
            | ShaderEntry::GeGLUConcat
            | ShaderEntry::GeGLUConcatGrad => {
                pc.bind(
                    0,
                    &BinaryData {
                        src_a: buf(dispatch.input_buffers[0]),
                        src_b: buf(dispatch.input_buffers[1]),
                        dst: buf(dispatch.output_buffer),
                        params: UnaryParams {
                            len: dispatch.params[0],
                            _pad0: dispatch.params[1], // half_n
                            _pad1: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            ShaderEntry::PairwiseGrad => {
                pc.bind(
                    0,
                    &TernaryData {
                        src_a: buf(dispatch.input_buffers[0]),
                        src_b: buf(dispatch.input_buffers[1]),
                        src_c: buf(dispatch.input_buffers[2]),
                        dst: buf(dispatch.output_buffer),
                        params: UnaryParams {
                            len: dispatch.params[0],
                            _pad0: dispatch.params[1],
                            _pad1: dispatch.params[2],
                            _pad2: dispatch.params[3],
                        },
                    },
                );
            }
            ShaderEntry::SumAll | ShaderEntry::MeanAll => {
                pc.bind(
                    0,
                    &UnaryData {
                        src: buf(dispatch.input_buffers[0]),
                        dst: buf(dispatch.output_buffer),
                        params: UnaryParams {
                            len: dispatch.params[0],
                            _pad0: dispatch.params.get(1).copied().unwrap_or(0), // mean divisor
                            _pad1: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            ShaderEntry::SumRows => {
                // Rows, columns, and optional serial-row layout.
                pc.bind(
                    0,
                    &UnaryData {
                        src: buf(dispatch.input_buffers[0]),
                        dst: buf(dispatch.output_buffer),
                        params: UnaryParams {
                            len: dispatch.params[0],   // m
                            _pad0: dispatch.params[1], // n
                            _pad1: dispatch.params.get(2).copied().unwrap_or(0),
                            _pad2: dispatch.params.get(3).copied().unwrap_or(0),
                        },
                    },
                );
            }
            ShaderEntry::CrossEntropyLoss | ShaderEntry::CrossEntropyLossIndices => {
                let loss_buf = dispatch
                    .extra_outputs
                    .first()
                    .copied()
                    .unwrap_or(dispatch.output_buffer);
                pc.bind(
                    0,
                    &CrossEntropyData {
                        logits: buf(dispatch.input_buffers[0]),
                        labels: buf(dispatch.input_buffers[1]),
                        grad_out: buf(dispatch.output_buffer),
                        loss_out: buf(loss_buf),
                        params: SoftmaxParams {
                            batch: dispatch.params[0],
                            features: dispatch.params[1],
                            _pad0: dispatch.params.get(2).copied().unwrap_or(0),
                            _pad1: 0,
                        },
                    },
                );
            }
            ShaderEntry::BceLoss => {
                pc.bind(
                    0,
                    &BceData {
                        pred: buf(dispatch.input_buffers[0]),
                        labels: buf(dispatch.input_buffers[1]),
                        loss_out: buf(dispatch.output_buffer),
                        params: UnaryParams {
                            len: dispatch.params[0],
                            _pad0: 0,
                            _pad1: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            ShaderEntry::Transpose => {
                pc.bind(
                    0,
                    &TransposeData {
                        src: buf(dispatch.input_buffers[0]),
                        dst: buf(dispatch.output_buffer),
                        params: TransposeParams {
                            m: dispatch.params[0],
                            n: dispatch.params[1],
                            _pad0: 0,
                            _pad1: 0,
                        },
                    },
                );
            }
            ShaderEntry::RmsNormAdd => {
                pc.bind(
                    0,
                    &RmsNormAddData {
                        src: buf(dispatch.input_buffers[0]),
                        bias: buf(dispatch.input_buffers[1]),
                        residual: buf(dispatch.input_buffers[2]),
                        dst: buf(dispatch.output_buffer),
                        params: BiasAddParams {
                            len: dispatch.params[0],
                            bias_len: dispatch.params[1],
                            _pad0: dispatch.params[2], // eps_bits
                            _pad1: 0,
                        },
                    },
                );
            }
            ShaderEntry::Embedding => {
                pc.bind(
                    0,
                    &EmbeddingData {
                        indices: buf(dispatch.input_buffers[0]),
                        src: buf(dispatch.input_buffers[1]),
                        dst: buf(dispatch.output_buffer),
                        params: UnaryParams {
                            len: dispatch.params[0],
                            _pad0: dispatch.params[1],
                            _pad1: dispatch.params[2], // table rows
                            _pad2: 0,
                        },
                    },
                );
            }
            ShaderEntry::RoPE | ShaderEntry::RoPEGrad => {
                pc.bind(
                    0,
                    &RoPEData {
                        src: buf(dispatch.input_buffers[0]),
                        dst: buf(dispatch.output_buffer),
                        params: RoPEParams {
                            seq: dispatch.params[0],
                            dim: dispatch.params[1],
                            theta_bits: dispatch.params[2],
                            pos_offset: dispatch.params[3],
                            head_dim: dispatch.params[4],
                            _pad0: 0,
                            _pad1: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            ShaderEntry::LayerNorm => {
                pc.bind(
                    0,
                    &LayerNormData {
                        src: buf(dispatch.input_buffers[0]),
                        src_b: buf(dispatch.input_buffers[1]),
                        bias: buf(dispatch.input_buffers[2]),
                        dst: buf(dispatch.output_buffer),
                        params: MatMulParams {
                            m: dispatch.params[0],
                            n: dispatch.params[1],
                            k: dispatch.params[2],
                            _pad: dispatch.params[3],
                        },
                    },
                );
            }
            ShaderEntry::MultiHeadAttn
            | ShaderEntry::FlashAttention
            | ShaderEntry::FlashAttentionCoop
            | ShaderEntry::FlashAttentionCoopF32 => {
                pc.bind(
                    0,
                    &MultiHeadAttnData {
                        src_a: buf(dispatch.input_buffers[0]),
                        src_b: buf(dispatch.input_buffers[1]),
                        bias: buf(dispatch.input_buffers[2]),
                        dst: buf(dispatch.output_buffer),
                        lse: buf(dispatch.extra_outputs[0]),
                        params: AttentionParams {
                            q_seq: dispatch.params[0],
                            kv_seq: dispatch.params[1],
                            packed_heads: dispatch.params[2],
                            head_dim: dispatch.params[3],
                            window_size: *dispatch.params.get(4).unwrap_or(&0),
                            _pad0: 0,
                            _pad1: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            ShaderEntry::AttentionGrad(AttentionGrad {
                part: AttentionGradPart::KV,
                ..
            }) => {
                pc.bind(
                    0,
                    &MultiHeadAttnGradKVData {
                        d_out: buf(dispatch.input_buffers[0]),
                        src_a: buf(dispatch.input_buffers[1]),
                        src_b: buf(dispatch.input_buffers[2]),
                        bias: buf(dispatch.input_buffers[3]),
                        lse: buf(dispatch.input_buffers[4]),
                        fwd_dst: buf(dispatch.input_buffers[5]),
                        dst: buf(dispatch.output_buffer),
                        dst2: buf(dispatch.extra_outputs[0]),
                        params: AttentionParams {
                            q_seq: dispatch.params[0],
                            kv_seq: dispatch.params[1],
                            packed_heads: dispatch.params[2],
                            head_dim: dispatch.params[3],
                            window_size: dispatch.params[4],
                            _pad0: 0,
                            _pad1: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            ShaderEntry::AttentionGrad(AttentionGrad {
                part: AttentionGradPart::Q,
                ..
            }) => {
                pc.bind(
                    0,
                    &MultiHeadAttnGradData {
                        d_out: buf(dispatch.input_buffers[0]),
                        src_a: buf(dispatch.input_buffers[1]),
                        src_b: buf(dispatch.input_buffers[2]),
                        bias: buf(dispatch.input_buffers[3]),
                        lse: buf(dispatch.input_buffers[4]),
                        fwd_dst: buf(dispatch.input_buffers[5]),
                        dst: buf(dispatch.output_buffer),
                        params: AttentionParams {
                            q_seq: dispatch.params[0],
                            kv_seq: dispatch.params[1],
                            packed_heads: dispatch.params[2],
                            head_dim: dispatch.params[3],
                            window_size: dispatch.params[4],
                            _pad0: 0,
                            _pad1: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            ShaderEntry::SwiGLUGradGate => {
                pc.bind(
                    0,
                    &TernaryData {
                        src_a: buf(dispatch.input_buffers[0]), // grad_out
                        src_b: buf(dispatch.input_buffers[1]), // gate
                        src_c: buf(dispatch.input_buffers[2]), // up
                        dst: buf(dispatch.output_buffer),
                        params: UnaryParams {
                            len: dispatch.params[0],
                            _pad0: 0,
                            _pad1: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            ShaderEntry::SwiGLUGradUp => {
                pc.bind(
                    0,
                    &BinaryData {
                        src_a: buf(dispatch.input_buffers[0]), // grad_out
                        src_b: buf(dispatch.input_buffers[1]), // gate
                        dst: buf(dispatch.output_buffer),
                        params: UnaryParams {
                            len: dispatch.params[0],
                            _pad0: 0,
                            _pad1: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            ShaderEntry::SiluGrad => {
                pc.bind(
                    0,
                    &BinaryData {
                        src_a: buf(dispatch.input_buffers[0]), // grad_out
                        src_b: buf(dispatch.input_buffers[1]), // x
                        dst: buf(dispatch.output_buffer),
                        params: UnaryParams {
                            len: dispatch.params[0],
                            _pad0: 0,
                            _pad1: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            ShaderEntry::RmsNormGradW
            | ShaderEntry::RmsNormGradWRowPar
            | ShaderEntry::RmsNormGradX => {
                pc.bind(
                    0,
                    &FourBufData {
                        src_a: buf(dispatch.input_buffers[0]), // dy
                        src_b: buf(dispatch.input_buffers[1]), // x
                        bias: buf(dispatch.input_buffers[2]),  // w
                        dst: buf(dispatch.output_buffer),
                        params: MatMulParams {
                            m: dispatch.params[0], // rows
                            n: dispatch.params[1], // cols
                            k: dispatch.params[2], // eps_bits
                            _pad: dispatch.params[3],
                        },
                    },
                );
            }
            ShaderEntry::LayerNormGradWB | ShaderEntry::LayerNormGradX => {
                pc.bind(
                    0,
                    &FourBufData {
                        src_a: buf(dispatch.input_buffers[0]), // dy
                        src_b: buf(dispatch.input_buffers[1]), // x
                        bias: buf(dispatch.input_buffers[2]),  // w
                        dst: buf(dispatch.output_buffer),
                        params: MatMulParams {
                            m: dispatch.params[0], // rows
                            n: dispatch.params[1], // cols
                            k: dispatch.params[2], // eps_bits
                            _pad: dispatch.params[3],
                        },
                    },
                );
            }
            ShaderEntry::RmsNormRsqrt => {
                // bindings: src=X, dst=rsqrt, params=(rows, cols, eps_bits, _pad)
                pc.bind(
                    0,
                    &UnaryData {
                        src: buf(dispatch.input_buffers[0]),
                        dst: buf(dispatch.output_buffer),
                        params: UnaryParams {
                            len: dispatch.params[0],
                            _pad0: dispatch.params[1],
                            _pad1: dispatch.params[2],
                            _pad2: dispatch.params[3],
                        },
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
                pc.bind(
                    0,
                    &ScatterAddData {
                        indices: buf(dispatch.input_buffers[0]),
                        src: buf(dispatch.input_buffers[1]),
                        dst: buf(dispatch.output_buffer),
                        params: ScatterAddParams {
                            total: dispatch.params[0],
                            seq_len: dispatch.params[1],
                            embed_dim: dispatch.params[2],
                            _pad: 0,
                        },
                    },
                );
            }
            ShaderEntry::ScatterAddAtomic => {
                pc.bind(
                    0,
                    &ScatterAddAtomicData {
                        indices: buf(dispatch.input_buffers[0]),
                        src: buf(dispatch.input_buffers[1]),
                        row_scale: buf(dispatch
                            .input_buffers
                            .get(2)
                            .copied()
                            .unwrap_or(dispatch.input_buffers[1])),
                        dst: buf(dispatch.output_buffer),
                        params: ScatterAddParams {
                            total: dispatch.params[0],
                            seq_len: dispatch.params[1],
                            embed_dim: dispatch.params[2],
                            _pad: dispatch.params[3],
                        },
                    },
                );
            }
            ShaderEntry::GroupNorm | ShaderEntry::GroupNormSilu => {
                let p = &dispatch.params;
                pc.bind(
                    0,
                    &GroupNormData {
                        src: buf(dispatch.input_buffers[0]),
                        src_b: buf(dispatch.input_buffers[1]),
                        bias: buf(dispatch.input_buffers[2]),
                        dst: buf(dispatch.output_buffer),
                        params: GroupNormParams {
                            batch: p[0],
                            channels: p[1],
                            spatial: p[2],
                            num_groups: p[3],
                            eps_bits: p[4],
                            chunks: 1,
                            apply_silu: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            ShaderEntry::GroupNormStats => {
                let p = &dispatch.params;
                pc.bind(
                    0,
                    &GroupNormStatsData {
                        src: buf(dispatch.input_buffers[0]),
                        dst: buf(dispatch.output_buffer),
                        params: GroupNormParams {
                            batch: p[0],
                            channels: p[1],
                            spatial: p[2],
                            num_groups: p[3],
                            eps_bits: p[4],
                            chunks: p[5],
                            apply_silu: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            ShaderEntry::GroupNormApply => {
                let p = &dispatch.params;
                pc.bind(
                    0,
                    &GroupNormApplyData {
                        src: buf(dispatch.input_buffers[0]),
                        src_b: buf(dispatch.input_buffers[2]),
                        bias: buf(dispatch.input_buffers[3]),
                        dst: buf(dispatch.output_buffer),
                        partials: buf(dispatch.input_buffers[1]),
                        params: GroupNormParams {
                            batch: p[0],
                            channels: p[1],
                            spatial: p[2],
                            num_groups: p[3],
                            eps_bits: p[4],
                            chunks: p[5],
                            apply_silu: p[6],
                            _pad2: 0,
                        },
                    },
                );
            }
            ShaderEntry::GroupNormGradInput
            | ShaderEntry::GroupNormGradWeightBias
            | ShaderEntry::GroupNormGradStats => {
                let p = &dispatch.params;
                let inputs = &dispatch.input_buffers;
                // Unused bindings of an entry point take a buffer it reads.
                let (src_a, src_b, bias, stats) = match dispatch.shader {
                    ShaderEntry::GroupNormGradInput => (inputs[0], inputs[1], inputs[2], inputs[3]),
                    ShaderEntry::GroupNormGradWeightBias => {
                        (inputs[0], inputs[1], inputs[1], inputs[2])
                    }
                    _ => (inputs[0], inputs[0], inputs[0], inputs[0]),
                };
                pc.bind(
                    0,
                    &GroupNormGradData {
                        src_a: buf(src_a),
                        src_b: buf(src_b),
                        bias: buf(bias),
                        dst: buf(dispatch.output_buffer),
                        stats: buf(stats),
                        params: GroupNormParams {
                            batch: p[0],
                            channels: p[1],
                            spatial: p[2],
                            num_groups: p[3],
                            eps_bits: p[4],
                            chunks: 1,
                            apply_silu: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            ShaderEntry::Concat => {
                let p = &dispatch.params;
                pc.bind(
                    0,
                    &BinaryData {
                        src_a: buf(dispatch.input_buffers[0]),
                        src_b: buf(dispatch.input_buffers[1]),
                        dst: buf(dispatch.output_buffer),
                        params: UnaryParams {
                            len: p[0],
                            _pad0: p[1],
                            _pad1: p[2],
                            _pad2: p[3],
                        },
                    },
                );
            }
            ShaderEntry::BiasedAttention => {
                let p = &dispatch.params;
                let input = |i: usize| buf(dispatch.input_buffers[i]);
                pc.bind(
                    0,
                    &super::BiasedAttentionData {
                        q: input(0),
                        k: input(1),
                        v: input(2),
                        bias: input(3),
                        kv_pos: input(4),
                        dst: buf(dispatch.output_buffer),
                        params: [p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7]],
                    },
                );
            }
            ShaderEntry::Permute => {
                let p = &dispatch.params;
                pc.bind(
                    0,
                    &super::PermuteData {
                        src: buf(dispatch.input_buffers[0]),
                        dst: buf(dispatch.output_buffer),
                        params: super::PermuteParams {
                            total: p[0],
                            dims: [p[1], p[2], p[3]],
                            strides: [p[4], p[5], p[6], p[7]],
                        },
                    },
                );
            }
            ShaderEntry::SplitA | ShaderEntry::SplitB => {
                let p = &dispatch.params;
                pc.bind(
                    0,
                    &UnaryData {
                        src: buf(dispatch.input_buffers[0]),
                        dst: buf(dispatch.output_buffer),
                        params: UnaryParams {
                            len: p[0],
                            _pad0: p[1],
                            _pad1: p[2],
                            _pad2: p[3],
                        },
                    },
                );
            }
            ShaderEntry::Upsample2x | ShaderEntry::Upsample2xGrad => {
                let p = &dispatch.params;
                pc.bind(
                    0,
                    &UnaryData {
                        src: buf(dispatch.input_buffers[0]),
                        dst: buf(dispatch.output_buffer),
                        params: UnaryParams {
                            len: p[0],
                            _pad0: p[1],
                            _pad1: p[2],
                            _pad2: p[3],
                        },
                    },
                );
            }
            ShaderEntry::MulPerChannel => {
                let p = &dispatch.params;
                pc.bind(
                    0,
                    &MulPerChannelData {
                        src: buf(dispatch.input_buffers[0]),
                        gate: buf(dispatch.input_buffers[1]),
                        dst: buf(dispatch.output_buffer),
                        params: MulPerChannelParams {
                            len: p[0],
                            spatial: p[1],
                            _pad0: 0,
                            _pad1: 0,
                        },
                    },
                );
            }
            ShaderEntry::Conv2dDw => {
                let p = &dispatch.params;
                pc.bind(
                    0,
                    &Conv2dDwData {
                        src: buf(dispatch.input_buffers[0]),
                        weight: buf(dispatch.input_buffers[1]),
                        dst: buf(dispatch.output_buffer),
                        params: Conv2dDwParams {
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
                    },
                );
            }
            ShaderEntry::Conv2dGemm
            | ShaderEntry::Conv2dGemmSmall
            | ShaderEntry::Conv2dGemm16
            | ShaderEntry::Conv2dGemmCoopGen(..) => {
                pc.bind(
                    0,
                    &Conv2dData {
                        src: buf(dispatch.input_buffers[0]),
                        weight: buf(dispatch.input_buffers[1]),
                        dst: buf(dispatch.output_buffer),
                        params: Conv2dParams::from(dispatch),
                    },
                );
            }
            ShaderEntry::Conv2dGradInputGemm
            | ShaderEntry::Conv2dGradInputGemmSmall
            | ShaderEntry::Conv2dGradInputGemm16
            | ShaderEntry::Conv2dGradInputGemmCoopGen(..) => {
                pc.bind(
                    0,
                    &Conv2dGradInputData {
                        grad_out: buf(dispatch.input_buffers[0]),
                        weight: buf(dispatch.input_buffers[1]),
                        dst: buf(dispatch.output_buffer),
                        params: Conv2dParams::from(dispatch),
                    },
                );
            }
            ShaderEntry::Conv2dGradWeightGemm
            | ShaderEntry::Conv2dGradWeightGemmSmall
            | ShaderEntry::Conv2dGradWeightGemm16
            | ShaderEntry::Conv2dGradWeightGemmSplit
            | ShaderEntry::Conv2dGradWeightGemmSplitSmall
            | ShaderEntry::Conv2dGradWeightGemmSplit16 => {
                pc.bind(
                    0,
                    &Conv2dGradWeightData {
                        grad_out: buf(dispatch.input_buffers[0]),
                        src: buf(dispatch.input_buffers[1]),
                        dst: buf(dispatch.output_buffer),
                        params: Conv2dParams::from(dispatch),
                    },
                );
            }
            ShaderEntry::RoPEDynamicFactors => {
                pc.bind(
                    0,
                    &RoPEDynamicFactorsData {
                        src: buf(dispatch.input_buffers[0]),
                        dst: buf(dispatch.output_buffer),
                        pos_offset_buf: buf(dispatch.input_buffers[1]),
                        factors: buf(dispatch.input_buffers[2]),
                        params: RoPEParams {
                            seq: dispatch.params[0],
                            dim: dispatch.params[1],
                            theta_bits: dispatch.params[2],
                            pos_offset: dispatch.params[3],
                            head_dim: dispatch.params[4],
                            _pad0: 0,
                            _pad1: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            ShaderEntry::RoPEDynamic | ShaderEntry::RoPEPositions => {
                pc.bind(
                    0,
                    &RoPEDynamicData {
                        src: buf(dispatch.input_buffers[0]),
                        dst: buf(dispatch.output_buffer),
                        pos_offset_buf: buf(dispatch.input_buffers[1]),
                        params: RoPEParams {
                            seq: dispatch.params[0],
                            dim: dispatch.params[1],
                            theta_bits: dispatch.params[2],
                            pos_offset: dispatch.params[3],
                            head_dim: dispatch.params[4],
                            _pad0: 0,
                            _pad1: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            ShaderEntry::CacheWrite => {
                pc.bind(
                    0,
                    &CacheWriteData {
                        src: buf(dispatch.input_buffers[0]),
                        dst: buf(dispatch.output_buffer),
                        kv_pos_buf: buf(dispatch.input_buffers[2]),
                        params: UnaryParams {
                            len: dispatch.params[0],   // dim
                            _pad0: dispatch.params[1], // cache rows
                            _pad1: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            ShaderEntry::CacheWritePrefix => {
                pc.bind(
                    0,
                    &CacheWritePrefixData {
                        src: buf(dispatch.input_buffers[0]),
                        dst: buf(dispatch.output_buffer),
                        kv_pos_buf: buf(dispatch.input_buffers[2]),
                        valid_len_buf: buf(dispatch.input_buffers[3]),
                        params: MatMulParams {
                            m: dispatch.params[0],
                            n: dispatch.params[1],
                            k: dispatch.params[2],
                            _pad: dispatch.params[3],
                        },
                    },
                );
            }
            ShaderEntry::CachedAttention | ShaderEntry::CachedQueryAttention => {
                pc.bind(
                    0,
                    &CachedAttentionData {
                        src_a: buf(dispatch.input_buffers[0]),      // Q
                        src_b: buf(dispatch.input_buffers[1]),      // K cache
                        bias: buf(dispatch.input_buffers[2]),       // V cache
                        kv_pos_buf: buf(dispatch.input_buffers[3]), // kv_pos
                        dst: buf(dispatch.output_buffer),
                        params: CachedAttentionParams {
                            queries: dispatch.params[0],
                            num_heads: dispatch.params[1],
                            num_kv_heads: dispatch.params[2],
                            head_dim: dispatch.params[3],
                            max_seq: dispatch.params[4],
                            _pad: [0; 3],
                        },
                    },
                );
            }
            ShaderEntry::CachedBlockAttention | ShaderEntry::CachedBlockAttentionSplit => {
                pc.bind(
                    0,
                    &CachedBlockAttentionData {
                        src_a: buf(dispatch.input_buffers[0]),
                        src_b: buf(dispatch.input_buffers[1]),
                        bias: buf(dispatch.input_buffers[2]),
                        kv_pos_buf: buf(dispatch.input_buffers[3]),
                        valid_len_buf: buf(dispatch.input_buffers[4]),
                        dst: buf(dispatch.output_buffer),
                        params: CachedBlockAttentionParams::from_words(&dispatch.params)
                            .expect("cached attention parameter layout"),
                    },
                );
            }
            ShaderEntry::CachedBlockAttentionCombine => {
                pc.bind(
                    0,
                    &CachedBlockAttentionCombineData {
                        partials: buf(dispatch.input_buffers[0]),
                        dst: buf(dispatch.output_buffer),
                        params: CachedBlockAttentionParams::from_words(&dispatch.params)
                            .expect("cached attention parameter layout"),
                    },
                );
            }
            ShaderEntry::ChunkedRelativeAttention => {
                pc.bind(
                    0,
                    &ChunkedRelativeAttentionData {
                        src_a: buf(dispatch.input_buffers[0]),
                        src_b: buf(dispatch.input_buffers[1]),
                        bias: buf(dispatch.input_buffers[2]),
                        relative_k: buf(dispatch.input_buffers[3]),
                        dst: buf(dispatch.output_buffer),
                        params: ChunkedRelativeAttentionParams {
                            seq_len: dispatch.params[0],
                            num_heads: dispatch.params[1],
                            head_dim: dispatch.params[2],
                            left_context: dispatch.params[3],
                            softcap_bits: dispatch.params[4],
                            _pad0: 0,
                            _pad1: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            ShaderEntry::PrefixLast => {
                pc.bind(
                    0,
                    &PrefixLastData {
                        src: buf(dispatch.input_buffers[0]),
                        valid_len_buf: buf(dispatch.input_buffers[1]),
                        dst: buf(dispatch.output_buffer),
                        params: MatMulParams {
                            m: dispatch.params[0],
                            n: dispatch.params[1],
                            k: 0,
                            _pad: 0,
                        },
                    },
                );
            }
            ShaderEntry::MaxPool2d => {
                pc.bind(
                    0,
                    &MaxPool2dData {
                        src: buf(dispatch.input_buffers[0]),
                        dst: buf(dispatch.output_buffer),
                        params: MaxPool2dParams {
                            batch: dispatch.params[0],
                            channels: dispatch.params[1],
                            in_h: dispatch.params[2],
                            in_w: dispatch.params[3],
                            kernel_h: dispatch.params[4],
                            kernel_w: dispatch.params[5],
                            stride: dispatch.params[6],
                            padding: dispatch.params[7],
                            out_h: dispatch.params[8],
                            out_w: dispatch.params[9],
                            _pad0: dispatch.params[10],
                            _pad1: dispatch.params[11],
                        },
                    },
                );
            }
            ShaderEntry::MaxPool2dGrad => {
                let p = &dispatch.params;
                pc.bind(
                    0,
                    &MaxPool2dGradData {
                        grad_out: buf(dispatch.input_buffers[0]),
                        src: buf(dispatch.input_buffers[1]),
                        dst: buf(dispatch.output_buffer),
                        params: MaxPool2dParams {
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
                    },
                );
            }
            ShaderEntry::GlobalAvgPool => {
                pc.bind(
                    0,
                    &GlobalAvgPoolData {
                        src: buf(dispatch.input_buffers[0]),
                        dst: buf(dispatch.output_buffer),
                        params: GlobalAvgPoolParams {
                            channels: dispatch.params[0],
                            spatial: dispatch.params[1],
                            total_out: dispatch.params[2],
                            _pad: dispatch.params[3],
                        },
                    },
                );
            }
            ShaderEntry::GlobalAvgPoolGrad => {
                let p = &dispatch.params;
                pc.bind(
                    0,
                    &UnaryData {
                        src: buf(dispatch.input_buffers[0]),
                        dst: buf(dispatch.output_buffer),
                        params: UnaryParams {
                            len: p[0],
                            _pad0: p[1],
                            _pad1: p[2],
                            _pad2: p[3],
                        },
                    },
                );
            }
            ShaderEntry::WinogradInputTransform
            | ShaderEntry::WinogradOutputTransform
            | ShaderEntry::WinogradWeightTransform => {
                let p = &dispatch.params;
                pc.bind(
                    0,
                    &WinogradTransformData {
                        src: buf(dispatch.input_buffers[0]),
                        dst: buf(dispatch.output_buffer),
                        params: WinogradTransformParams {
                            p0: p[0],
                            p1: p[1],
                            p2: p[2],
                            p3: p[3],
                            p4: p[4],
                            p5: p[5],
                            p6: p[6],
                            p7: p[7],
                        },
                    },
                );
            }
            ShaderEntry::WinogradBatchedMatMul => {
                pc.bind(
                    0,
                    &MatMulData {
                        matrix_a: buf(dispatch.input_buffers[0]),
                        matrix_b: buf(dispatch.input_buffers[1]),
                        matrix_c: buf(dispatch.output_buffer),
                        params: MatMulParams {
                            m: dispatch.params[0],
                            n: dispatch.params[1],
                            k: dispatch.params[2],
                            _pad: dispatch.params[3],
                        },
                    },
                );
            }
        }
    }
}
