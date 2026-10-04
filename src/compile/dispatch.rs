//! Typed operations in a compiled execution plan.

use super::{
    BufferRef, GemvRmsNorm, Kernel, MatMulEpilogue, MatMulPrologue, ShaderEntry, WeightFormat,
};
use crate::{
    codegen,
    schedule::{PointwiseDAG, ReductionKernel},
};
use serde::{Deserialize, Serialize};

trait Payload {
    fn visit_inputs(&self, visit: &mut impl FnMut(BufferRef));

    fn inputs(&self) -> Vec<BufferRef> {
        let mut inputs = Vec::new();
        self.visit_inputs(&mut |buffer| inputs.push(buffer));
        inputs
    }

    fn visit_outputs(&self, visit: &mut impl FnMut(BufferRef)) {
        visit(self.output());
        for buffer in self.extra_outputs() {
            visit(buffer);
        }
    }

    fn output(&self) -> BufferRef;
    fn extra_outputs(&self) -> Vec<BufferRef>;
    fn map_buffers(&mut self, map: &mut impl FnMut(BufferRef) -> BufferRef);
    fn set_input(&mut self, index: usize, buffer: BufferRef);
    fn set_output(&mut self, buffer: BufferRef);
    fn parameter_words(&self) -> Vec<u32>;
}

// Fixed binding contracts. Buffer roles and parameter names are declared once;
// visitors used by scheduling and memory planning follow the same declaration.
macro_rules! payloads {
    ($($name:ident {
        inputs: [$($input:ident),*], output: $output:ident,
        extras: [$($extra:ident),*], params: [$($param:ident),*]
        $(, config: [$($config:ident: $config_ty:ty),*])?
    })*) => {$(
        #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
        pub struct $name {
            $(pub $input: BufferRef,)*
            pub $output: BufferRef,
            $(pub $extra: BufferRef,)*
            $(pub $param: u32,)*
            $($(pub $config: $config_ty,)*)?
        }

        impl Payload for $name {
            fn visit_inputs(&self, visit: &mut impl FnMut(BufferRef)) {
                $(visit(self.$input);)*
            }

            fn output(&self) -> BufferRef {
                self.$output
            }

            fn extra_outputs(&self) -> Vec<BufferRef> {
                vec![$(self.$extra),*]
            }

            fn visit_outputs(&self, visit: &mut impl FnMut(BufferRef)) {
                visit(self.$output);
                $(visit(self.$extra);)*
            }

            fn map_buffers(&mut self, map: &mut impl FnMut(BufferRef) -> BufferRef) {
                $(self.$input = map(self.$input);)*
                self.$output = map(self.$output);
                $(self.$extra = map(self.$extra);)*
            }

            fn set_input(&mut self, index: usize, buffer: BufferRef) {
                let fields = [$(&mut self.$input),*];
                *fields.into_iter().nth(index).expect("input index") = buffer;
            }

            fn set_output(&mut self, buffer: BufferRef) {
                self.$output = buffer;
            }

            fn parameter_words(&self) -> Vec<u32> {
                let mut words = vec![$(self.$param),*];
                words.resize(words.len().max(4), 0);
                words
            }
        }
    )*};
}

payloads! {
    ConvolutionArgs { inputs: [a, b], output: dst, extras: [],
        params: [batch, in_channels, in_h, in_w, out_channels, kernel_h, kernel_w, stride, padding_h, out_h, out_w, padding_w] }
    AttentionForward { inputs: [q, k, v], output: dst, extras: [lse],
        params: [q_seq, kv_seq, packed_heads, head_dim, window_size] }
    AttentionGradQ { inputs: [d_out, q, k, v, lse, row_source], output: dst, extras: [],
        params: [q_seq, kv_seq, packed_heads, head_dim, window_size] }
    AttentionGradKV { inputs: [d_out, q, k, v, lse, row_source], output: dk, extras: [dv],
        params: [q_seq, kv_seq, packed_heads, head_dim, window_size] }
    CachedAttention { inputs: [q, k_cache, v_cache, position], output: dst, extras: [],
        params: [q_seq, num_heads, num_kv_heads, head_dim] }
    CachedBlockAttention { inputs: [q, k_cache, v_cache, position, valid_len], output: dst, extras: [],
        params: [window_size, num_heads, num_kv_heads, head_dim, block_len, max_seq, splits] }
    CachedAttentionCombine { inputs: [partials], output: dst, extras: [],
        params: [window_size, num_heads, num_kv_heads, head_dim, block_len, max_seq, splits] }
    ChunkedRelativeAttention { inputs: [q, k, v, relative_k], output: dst, extras: [],
        params: [seq_len, num_heads, head_dim, left_context, softcap_bits] }
    PrefixLast { inputs: [src, valid_len], output: dst, extras: [], params: [cols, rows] }
    NormResidual { inputs: [src, weight, residual], output: dst, extras: [], params: [rows, cols, eps_bits] }
    LayerNorm { inputs: [src, weight, bias], output: dst, extras: [], params: [rows, cols, eps_bits, block_rows] }
    NormGradient { inputs: [dy, src, weight], output: dst, extras: [], params: [rows, cols, eps_bits, block_rows] }
    RmsNormRsqrt { inputs: [src], output: dst, extras: [], params: [rows, cols, eps_bits] }
    GroupNorm { inputs: [src, weight, bias], output: dst, extras: [],
        params: [batch, channels, spatial, num_groups, eps_bits] }
    GroupNormStats { inputs: [src], output: dst, extras: [],
        params: [batch, channels, spatial, num_groups, eps_bits, chunks] }
    GroupNormApply { inputs: [src, partials, weight, bias], output: dst, extras: [],
        params: [batch, channels, spatial, num_groups, eps_bits, chunks, apply_silu] }
    GroupNormGradInput { inputs: [dy, src, weight, stats], output: dst, extras: [],
        params: [batch, channels, spatial, num_groups, eps_bits] }
    GroupNormGradWeightBias { inputs: [dy, src, stats], output: dst, extras: [],
        params: [batch, channels, spatial, num_groups, eps_bits] }
    GatedActivation { inputs: [src_a, src_b], output: dst, extras: [], params: [len, half_width] }
    GateGradient { inputs: [dy, gate, up], output: dst, extras: [], params: [len] }
    BinaryGradient { inputs: [dy, src], output: dst, extras: [], params: [len] }
    PairwiseGradient { inputs: [gradient, a, b], output: dst, extras: [], params: [total, inner, pairs, mode] }
    ReduceAll { inputs: [src], output: dst, extras: [], params: [len, divisor] }
    SumRows { inputs: [src], output: dst, extras: [], params: [rows, cols, serial_rows, splits] }
    CrossEntropy { inputs: [logits, labels], output: gradient, extras: [loss], params: [batch, features, write_grad] }
    BinaryLoss { inputs: [prediction, labels], output: dst, extras: [], params: [len] }
    Unary { inputs: [src], output: dst, extras: [], params: [len] }
    Transpose { inputs: [src], output: dst, extras: [], params: [rows, cols] }
    Embedding { inputs: [indices, table], output: dst, extras: [], params: [rows, embed_dim], config: [weight_format: WeightFormat] }
    Rope { inputs: [src], output: dst, extras: [], params: [seq, dim, theta_bits, pos_offset, head_dim] }
    RopeDynamic { inputs: [src, position], output: dst, extras: [], params: [seq, dim, theta_bits, pos_offset, head_dim] }
    RopeFactors { inputs: [src, position, factors], output: dst, extras: [], params: [seq, dim, theta_bits, pos_offset, head_dim] }
    Concat { inputs: [a, b], output: dst, extras: [], params: [batch, channels_a, channels_b, spatial] }
    Split { inputs: [src], output: dst, extras: [], params: [batch, channels_a, channels_b, spatial] }
    MulPerChannel { inputs: [src, gate], output: dst, extras: [], params: [len, spatial, channels] }
    DepthwiseConv { inputs: [src, weight], output: dst, extras: [],
        params: [batch, channels, in_h, in_w, kernel_h, kernel_w, stride, padding_h, out_h, out_w, padding_w] }
    WinogradWeightTransform { inputs: [src], output: dst, extras: [], params: [out_channels, in_channels, adjoint] }
    WinogradInputTransform { inputs: [src], output: dst, extras: [], params: [batch, in_channels, in_h, in_w, padding, tiles_h, tiles_w, total_tiles] }
    WinogradOutputTransform { inputs: [src], output: dst, extras: [], params: [batch, out_channels, out_h, out_w, tiles_h, tiles_w, total_tiles] }
    MaxPool { inputs: [src], output: dst, extras: [],
        params: [batch, channels, in_h, in_w, kernel_h, kernel_w, stride, padding, out_h, out_w] }
    MaxPoolGradient { inputs: [dy, src], output: dst, extras: [],
        params: [batch, channels, in_h, in_w, kernel_h, kernel_w, stride, padding, out_h, out_w] }
    GlobalAvgPool { inputs: [src], output: dst, extras: [], params: [channels, spatial, total_out] }
    RowBroadcast { inputs: [src], output: dst, extras: [], params: [len, inner, mode, offset] }
    Upsample { inputs: [src], output: dst, extras: [], params: [batch, channels, in_h, in_w] }
}

impl ConvolutionArgs {
    pub(crate) fn geometry_words(&self) -> [u32; 12] {
        [
            self.batch,
            self.in_channels,
            self.in_h,
            self.in_w,
            self.out_channels,
            self.kernel_h,
            self.kernel_w,
            self.stride,
            self.padding_h,
            self.out_h,
            self.out_w,
            self.padding_w,
        ]
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MatmulKind {
    Plain,
    TransposeA,
    TransposeB,
    Block { batches: u32 },
    BlockAT { batches: u32 },
    BlockBT { batches: u32 },
    Gemv,
    GemvBT,
    Add { addend: BufferRef },
    AddAT { addend: BufferRef },
    AddBT { addend: BufferRef },
    GemvAdd { addend: BufferRef },
    GemvBTAdd { addend: BufferRef },
    Winograd { planes: u32 },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MatmulImplementation {
    Default,
    SmallTile,
    Scalar(codegen::ScalarMatmulShape),
    Split {
        shape: codegen::ScalarMatmulShape,
        splits: u32,
    },
    Cooperative,
    CooperativeCompensated,
    Gemv {
        shape: codegen::GemvShape,
        integer_dot: bool,
    },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Matmul {
    pub kind: MatmulKind,
    pub a: BufferRef,
    pub b: BufferRef,
    pub dst: BufferRef,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub implementation: MatmulImplementation,
    pub weight_format: WeightFormat,
    /// Additional (B, C) pairs sharing A in a horizontal batch.
    pub siblings: Vec<(BufferRef, BufferRef)>,
    pub epilogue: Option<MatMulEpilogue>,
    pub prologue: Option<MatMulPrologue>,
    pub rmsnorm: Option<GemvRmsNorm>,
}

impl Matmul {
    pub fn new(
        kind: MatmulKind,
        a: BufferRef,
        b: BufferRef,
        dst: BufferRef,
        [m, n, k]: [u32; 3],
    ) -> Self {
        Self {
            kind,
            a,
            b,
            dst,
            m,
            n,
            k,
            implementation: MatmulImplementation::Default,
            weight_format: WeightFormat::F32,
            siblings: Vec::new(),
            epilogue: None,
            prologue: None,
            rmsnorm: None,
        }
    }

    pub fn select_shader(&mut self, shader: ShaderEntry) {
        let batches = self.batch_count();
        self.kind = match shader {
            ShaderEntry::MatMul => MatmulKind::Plain,
            ShaderEntry::MatMulAT => MatmulKind::TransposeA,
            ShaderEntry::MatMulBT => MatmulKind::TransposeB,
            ShaderEntry::BlockMatMul => MatmulKind::Block { batches },
            ShaderEntry::BlockMatMulAT => MatmulKind::BlockAT { batches },
            ShaderEntry::BlockMatMulBT => MatmulKind::BlockBT { batches },
            ShaderEntry::MatMulGemv => MatmulKind::Gemv,
            ShaderEntry::MatMulGemvBT => MatmulKind::GemvBT,
            ShaderEntry::FusedMatMulAdd => MatmulKind::Add {
                addend: self.addend().expect("matmul addend"),
            },
            ShaderEntry::FusedMatMulATAdd => MatmulKind::AddAT {
                addend: self.addend().expect("matmul addend"),
            },
            ShaderEntry::FusedMatMulBTAdd => MatmulKind::AddBT {
                addend: self.addend().expect("matmul addend"),
            },
            ShaderEntry::MatMulGemvAdd => MatmulKind::GemvAdd {
                addend: self.addend().expect("matmul addend"),
            },
            ShaderEntry::MatMulGemvBTAdd => MatmulKind::GemvBTAdd {
                addend: self.addend().expect("matmul addend"),
            },
            ShaderEntry::WinogradBatchedMatMul => MatmulKind::Winograd { planes: batches },
            _ => panic!("not a matrix operation"),
        };
    }

    pub fn kernel(&self) -> Kernel {
        match self.implementation {
            MatmulImplementation::Default => Kernel::Default,
            MatmulImplementation::SmallTile => Kernel::SmallTile,
            MatmulImplementation::Scalar(shape) => Kernel::ScalarMatmul(shape),
            MatmulImplementation::Split { shape, splits } => Kernel::SplitMatmul { shape, splits },
            MatmulImplementation::Cooperative => Kernel::Cooperative,
            MatmulImplementation::CooperativeCompensated => Kernel::CooperativeCompensated,
            MatmulImplementation::Gemv { shape, integer_dot } => {
                Kernel::Gemv { shape, integer_dot }
            }
        }
    }

    pub fn set_kernel(&mut self, kernel: Kernel) {
        self.implementation = match kernel {
            Kernel::Default => MatmulImplementation::Default,
            Kernel::SmallTile => MatmulImplementation::SmallTile,
            Kernel::ScalarMatmul(shape) => MatmulImplementation::Scalar(shape),
            Kernel::SplitMatmul { shape, splits } => MatmulImplementation::Split { shape, splits },
            Kernel::Cooperative => MatmulImplementation::Cooperative,
            Kernel::CooperativeCompensated => MatmulImplementation::CooperativeCompensated,
            Kernel::Gemv { shape, integer_dot } => {
                MatmulImplementation::Gemv { shape, integer_dot }
            }
            _ => panic!("implementation does not apply to matmul"),
        };
    }

    pub fn shader(&self) -> ShaderEntry {
        match self.kind {
            MatmulKind::Plain => ShaderEntry::MatMul,
            MatmulKind::TransposeA => ShaderEntry::MatMulAT,
            MatmulKind::TransposeB => ShaderEntry::MatMulBT,
            MatmulKind::Block { .. } => ShaderEntry::BlockMatMul,
            MatmulKind::BlockAT { .. } => ShaderEntry::BlockMatMulAT,
            MatmulKind::BlockBT { .. } => ShaderEntry::BlockMatMulBT,
            MatmulKind::Gemv => ShaderEntry::MatMulGemv,
            MatmulKind::GemvBT => ShaderEntry::MatMulGemvBT,
            MatmulKind::Add { .. } => ShaderEntry::FusedMatMulAdd,
            MatmulKind::AddAT { .. } => ShaderEntry::FusedMatMulATAdd,
            MatmulKind::AddBT { .. } => ShaderEntry::FusedMatMulBTAdd,
            MatmulKind::GemvAdd { .. } => ShaderEntry::MatMulGemvAdd,
            MatmulKind::GemvBTAdd { .. } => ShaderEntry::MatMulGemvBTAdd,
            MatmulKind::Winograd { .. } => ShaderEntry::WinogradBatchedMatMul,
        }
    }

    pub fn addend(&self) -> Option<BufferRef> {
        match self.kind {
            MatmulKind::Add { addend }
            | MatmulKind::AddAT { addend }
            | MatmulKind::AddBT { addend }
            | MatmulKind::GemvAdd { addend }
            | MatmulKind::GemvBTAdd { addend } => Some(addend),
            _ => None,
        }
    }

    pub fn batch_count(&self) -> u32 {
        match self.kind {
            MatmulKind::Block { batches }
            | MatmulKind::BlockAT { batches }
            | MatmulKind::BlockBT { batches } => batches,
            MatmulKind::Winograd { planes } => planes,
            _ => 0,
        }
    }

    fn visit_input_mut(&mut self, visit: &mut impl FnMut(&mut BufferRef)) {
        visit(&mut self.a);
        visit(&mut self.b);
        match self.kind {
            MatmulKind::Add { ref mut addend }
            | MatmulKind::AddAT { ref mut addend }
            | MatmulKind::AddBT { ref mut addend }
            | MatmulKind::GemvAdd { ref mut addend }
            | MatmulKind::GemvBTAdd { ref mut addend } => visit(addend),
            _ => {}
        }
        for &mut (ref mut b, _) in &mut self.siblings {
            visit(b);
        }
        if let Some(ref mut epilogue) = self.epilogue {
            for &mut (ref mut buffer, _) in &mut epilogue.inputs {
                visit(buffer);
            }
        }
        if let Some(ref mut prologue) = self.prologue {
            for &mut (ref mut buffer, _) in &mut prologue.factors {
                visit(buffer);
            }
        }
        if let Some(ref mut rmsnorm) = self.rmsnorm {
            visit(&mut rmsnorm.weight);
        }
    }
}

impl Payload for Matmul {
    fn visit_inputs(&self, visit: &mut impl FnMut(BufferRef)) {
        visit(self.a);
        visit(self.b);
        if let Some(buffer) = self.addend() {
            visit(buffer);
        }
        for &(b, _) in &self.siblings {
            visit(b);
        }
        if let Some(ref epilogue) = self.epilogue {
            for &(b, _) in &epilogue.inputs {
                visit(b);
            }
        }
        if let Some(ref prologue) = self.prologue {
            for &(b, _) in &prologue.factors {
                visit(b);
            }
        }
        if let Some(ref rmsnorm) = self.rmsnorm {
            visit(rmsnorm.weight);
        }
    }

    fn visit_outputs(&self, visit: &mut impl FnMut(BufferRef)) {
        visit(self.dst);
        for &(_, c) in &self.siblings {
            visit(c);
        }
    }

    fn output(&self) -> BufferRef {
        self.dst
    }

    fn extra_outputs(&self) -> Vec<BufferRef> {
        self.siblings.iter().map(|&(_, c)| c).collect()
    }

    fn map_buffers(&mut self, map: &mut impl FnMut(BufferRef) -> BufferRef) {
        self.visit_input_mut(&mut |buffer| *buffer = map(*buffer));
        self.dst = map(self.dst);
        for &mut (_, ref mut c) in &mut self.siblings {
            *c = map(*c);
        }
    }

    fn set_input(&mut self, index: usize, buffer: BufferRef) {
        let mut current = 0;
        self.visit_input_mut(&mut |field| {
            if current == index {
                *field = buffer;
            }
            current += 1;
        });
        assert!(index < current, "input index");
    }

    fn set_output(&mut self, buffer: BufferRef) {
        self.dst = buffer;
    }

    fn parameter_words(&self) -> Vec<u32> {
        match self.kind {
            MatmulKind::Plain
            | MatmulKind::Gemv
            | MatmulKind::Add { .. }
            | MatmulKind::GemvAdd { .. } => vec![self.m, self.k, self.n, self.batch_count()],
            _ => vec![self.m, self.n, self.k, self.batch_count()],
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConvolutionKind {
    Forward,
    InputGradient,
    WeightGradient,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConvolutionImplementation {
    Scalar { tile: u32, k_tile: Option<u32> },
    Cooperative { specialized: bool },
    Split { tile: u32 },
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Convolution {
    pub kind: ConvolutionKind,
    pub implementation: ConvolutionImplementation,
    pub args: ConvolutionArgs,
}

impl Convolution {
    pub fn select_shader(&mut self, shader: ShaderEntry) {
        let (kind, tile, split, specialized) = match shader {
            ShaderEntry::Conv2dGemm => (ConvolutionKind::Forward, 64, false, false),
            ShaderEntry::Conv2dGemmSmall => (ConvolutionKind::Forward, 32, false, false),
            ShaderEntry::Conv2dGemm16 => (ConvolutionKind::Forward, 16, false, false),
            ShaderEntry::Conv2dGemmCoopGen(..) => (ConvolutionKind::Forward, 64, false, true),
            ShaderEntry::Conv2dGradInputGemm => (ConvolutionKind::InputGradient, 64, false, false),
            ShaderEntry::Conv2dGradInputGemmSmall => {
                (ConvolutionKind::InputGradient, 32, false, false)
            }
            ShaderEntry::Conv2dGradInputGemm16 => {
                (ConvolutionKind::InputGradient, 16, false, false)
            }
            ShaderEntry::Conv2dGradInputGemmCoopGen(..) => {
                (ConvolutionKind::InputGradient, 64, false, true)
            }
            ShaderEntry::Conv2dGradWeightGemm => {
                (ConvolutionKind::WeightGradient, 64, false, false)
            }
            ShaderEntry::Conv2dGradWeightGemmSmall => {
                (ConvolutionKind::WeightGradient, 32, false, false)
            }
            ShaderEntry::Conv2dGradWeightGemm16 => {
                (ConvolutionKind::WeightGradient, 16, false, false)
            }
            ShaderEntry::Conv2dGradWeightGemmSplit => {
                (ConvolutionKind::WeightGradient, 64, true, false)
            }
            ShaderEntry::Conv2dGradWeightGemmSplitSmall => {
                (ConvolutionKind::WeightGradient, 32, true, false)
            }
            ShaderEntry::Conv2dGradWeightGemmSplit16 => {
                (ConvolutionKind::WeightGradient, 16, true, false)
            }
            _ => panic!("not a convolution"),
        };
        self.kind = kind;
        self.implementation = if split {
            ConvolutionImplementation::Split { tile }
        } else if specialized {
            ConvolutionImplementation::Cooperative { specialized }
        } else {
            ConvolutionImplementation::Scalar { tile, k_tile: None }
        };
    }

    pub fn kernel(&self) -> Kernel {
        match self.implementation {
            ConvolutionImplementation::Scalar {
                k_tile: Some(k_tile),
                ..
            } => Kernel::SpecializedConv { k_tile },
            ConvolutionImplementation::Cooperative { .. } => Kernel::Cooperative,
            _ => Kernel::Default,
        }
    }

    pub fn set_kernel(&mut self, kernel: Kernel) {
        let tile = match self.implementation {
            ConvolutionImplementation::Scalar { tile, .. }
            | ConvolutionImplementation::Split { tile } => tile,
            ConvolutionImplementation::Cooperative { .. } => 64,
        };
        self.implementation = match kernel {
            Kernel::Default => match self.implementation {
                ConvolutionImplementation::Split { .. } => self.implementation,
                _ => ConvolutionImplementation::Scalar { tile, k_tile: None },
            },
            Kernel::SmallTile => ConvolutionImplementation::Scalar {
                tile: 32,
                k_tile: None,
            },
            Kernel::SpecializedConv { k_tile } => ConvolutionImplementation::Scalar {
                tile,
                k_tile: Some(k_tile),
            },
            Kernel::Cooperative => ConvolutionImplementation::Cooperative {
                specialized: matches!(
                    self.implementation,
                    ConvolutionImplementation::Cooperative { specialized: true }
                ),
            },
            _ => panic!("implementation does not apply to convolution"),
        };
    }

    pub fn shader(&self) -> ShaderEntry {
        let (tile, specialized, split) = match self.implementation {
            ConvolutionImplementation::Scalar { tile, .. } => (tile, false, false),
            ConvolutionImplementation::Cooperative { specialized } => (64, specialized, false),
            ConvolutionImplementation::Split { tile } => (tile, false, true),
        };
        match (self.kind, tile, specialized, split) {
            (ConvolutionKind::Forward, _, true, _) => ShaderEntry::Conv2dGemmCoopGen(
                self.args.kernel_h,
                self.args.kernel_w,
                self.args.stride,
            ),
            (ConvolutionKind::InputGradient, _, true, _) => {
                ShaderEntry::Conv2dGradInputGemmCoopGen(
                    self.args.kernel_h,
                    self.args.kernel_w,
                    self.args.stride,
                )
            }
            (ConvolutionKind::WeightGradient, 16, _, true) => {
                ShaderEntry::Conv2dGradWeightGemmSplit16
            }
            (ConvolutionKind::WeightGradient, 32, _, true) => {
                ShaderEntry::Conv2dGradWeightGemmSplitSmall
            }
            (ConvolutionKind::WeightGradient, _, _, true) => ShaderEntry::Conv2dGradWeightGemmSplit,
            (ConvolutionKind::Forward, 16, _, _) => ShaderEntry::Conv2dGemm16,
            (ConvolutionKind::Forward, 32, _, _) => ShaderEntry::Conv2dGemmSmall,
            (ConvolutionKind::Forward, _, _, _) => ShaderEntry::Conv2dGemm,
            (ConvolutionKind::InputGradient, 16, _, _) => ShaderEntry::Conv2dGradInputGemm16,
            (ConvolutionKind::InputGradient, 32, _, _) => ShaderEntry::Conv2dGradInputGemmSmall,
            (ConvolutionKind::InputGradient, _, _, _) => ShaderEntry::Conv2dGradInputGemm,
            (ConvolutionKind::WeightGradient, 16, _, _) => ShaderEntry::Conv2dGradWeightGemm16,
            (ConvolutionKind::WeightGradient, 32, _, _) => ShaderEntry::Conv2dGradWeightGemmSmall,
            (ConvolutionKind::WeightGradient, _, _, _) => ShaderEntry::Conv2dGradWeightGemm,
        }
    }
}

macro_rules! delegate_payload {
    ($ty:ty, $field:ident) => {
        impl Payload for $ty {
            fn visit_inputs(&self, visit: &mut impl FnMut(BufferRef)) {
                self.$field.visit_inputs(visit);
            }

            fn visit_outputs(&self, visit: &mut impl FnMut(BufferRef)) {
                self.$field.visit_outputs(visit);
            }

            fn output(&self) -> BufferRef {
                self.$field.output()
            }

            fn extra_outputs(&self) -> Vec<BufferRef> {
                self.$field.extra_outputs()
            }

            fn map_buffers(&mut self, map: &mut impl FnMut(BufferRef) -> BufferRef) {
                self.$field.map_buffers(map);
            }

            fn set_input(&mut self, index: usize, buffer: BufferRef) {
                self.$field.set_input(index, buffer);
            }

            fn set_output(&mut self, buffer: BufferRef) {
                self.$field.set_output(buffer);
            }

            fn parameter_words(&self) -> Vec<u32> {
                self.$field.parameter_words()
            }
        }
    };
}
delegate_payload!(Convolution, args);

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pointwise {
    pub inputs: Vec<BufferRef>,
    pub dst: BufferRef,
    pub len: u32,
    pub dag: PointwiseDAG,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Reduction {
    pub inputs: Vec<BufferRef>,
    pub dst: BufferRef,
    pub outer: u32,
    pub inner: u32,
    pub round_one_bits: u32,
    pub kernel: ReductionKernel,
}

macro_rules! scheduled_payload {
    ($name:ident, [$($param:ident),*]) => {
        impl Payload for $name {
            fn visit_inputs(&self, visit: &mut impl FnMut(BufferRef)) {
                for &buffer in &self.inputs {
                    visit(buffer);
                }
            }

            fn output(&self) -> BufferRef {
                self.dst
            }

            fn extra_outputs(&self) -> Vec<BufferRef> {
                Vec::new()
            }

            fn map_buffers(&mut self, map: &mut impl FnMut(BufferRef) -> BufferRef) {
                for buffer in &mut self.inputs {
                    *buffer = map(*buffer);
                }
                self.dst = map(self.dst);
            }

            fn set_input(&mut self, index: usize, buffer: BufferRef) {
                self.inputs[index] = buffer;
            }

            fn set_output(&mut self, buffer: BufferRef) {
                self.dst = buffer;
            }

            fn parameter_words(&self) -> Vec<u32> {
                let mut words = vec![$(self.$param),*];
                words.resize(4, 0);
                words
            }
        }
    };
}
scheduled_payload!(Pointwise, [len]);
scheduled_payload!(Reduction, [outer, inner, round_one_bits]);

macro_rules! inplace_payloads {
    ($($name:ident {
        inputs: [$($input:ident),*], output: $output:ident,
        params: [$($param:ident),*]
    })*) => {$(
        #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
        pub struct $name {
            $(pub $input: BufferRef,)*
            $(pub $param: u32,)*
        }

        impl Payload for $name {
            fn visit_inputs(&self, visit: &mut impl FnMut(BufferRef)) {
                $(visit(self.$input);)*
            }

            fn output(&self) -> BufferRef {
                self.$output
            }

            fn extra_outputs(&self) -> Vec<BufferRef> {
                Vec::new()
            }

            fn map_buffers(&mut self, map: &mut impl FnMut(BufferRef) -> BufferRef) {
                $(self.$input = map(self.$input);)*
            }

            fn set_input(&mut self, index: usize, buffer: BufferRef) {
                *[$(&mut self.$input),*].into_iter().nth(index).expect("input index") = buffer;
            }

            fn set_output(&mut self, buffer: BufferRef) {
                self.$output = buffer;
            }

            fn parameter_words(&self) -> Vec<u32> {
                let mut words = vec![$(self.$param),*];
                words.resize(words.len().max(4), 0);
                words
            }
        }
    )*};
}

inplace_payloads! {
    CacheWrite { inputs: [src, cache, position], output: cache, params: [dim] }
    CacheWritePrefix { inputs: [src, cache, position, valid_len], output: cache, params: [dim, block_len, max_seq] }
    Scatter { inputs: [indices, src, dst], output: dst, params: [total, seq_len, embed_dim] }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct AtomicScatter {
    pub indices: BufferRef,
    pub src: BufferRef,
    pub dst: BufferRef,
    pub row_scale: Option<BufferRef>,
    pub serial_rows: bool,
    pub total: u32,
    pub seq_len: u32,
    pub embed_dim: u32,
}

impl Payload for AtomicScatter {
    fn visit_inputs(&self, visit: &mut impl FnMut(BufferRef)) {
        visit(self.indices);
        visit(self.src);
        if let Some(buffer) = self.row_scale {
            visit(buffer);
        }
        visit(self.dst);
    }

    fn output(&self) -> BufferRef {
        self.dst
    }

    fn extra_outputs(&self) -> Vec<BufferRef> {
        Vec::new()
    }

    fn map_buffers(&mut self, map: &mut impl FnMut(BufferRef) -> BufferRef) {
        self.indices = map(self.indices);
        self.src = map(self.src);
        self.dst = map(self.dst);
        if let Some(ref mut buffer) = self.row_scale {
            *buffer = map(*buffer);
        }
    }

    fn set_input(&mut self, index: usize, buffer: BufferRef) {
        match index {
            0 => self.indices = buffer,
            1 => self.src = buffer,
            2 if self.row_scale.is_some() => self.row_scale = Some(buffer),
            2 => self.dst = buffer,
            3 if self.row_scale.is_some() => self.dst = buffer,
            _ => panic!("input index"),
        }
    }

    fn set_output(&mut self, buffer: BufferRef) {
        self.dst = buffer;
    }

    fn parameter_words(&self) -> Vec<u32> {
        vec![
            self.total,
            self.seq_len,
            self.embed_dim,
            if self.row_scale.is_none() {
                0
            } else if self.serial_rows {
                2
            } else {
                1
            },
        ]
    }
}

macro_rules! operations {
    ($($variant:ident($payload:ident)),* $(,)?) => {
        #[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
        pub enum DispatchOp {
            Matmul(Matmul),
            Convolution(Convolution),
            Pointwise(Pointwise),
            Reduction(Reduction),
            $($variant($payload),)*
        }

        impl DispatchOp {
            pub fn shader(&self) -> ShaderEntry {
                match *self {
                    DispatchOp::Matmul(ref op) => op.shader(),
                    DispatchOp::Convolution(ref op) => op.shader(),
                    DispatchOp::Pointwise(_) | DispatchOp::Reduction(_) => ShaderEntry::Generated,
                    $(DispatchOp::$variant(_) => ShaderEntry::$variant,)*
                }
            }

            pub fn inputs(&self) -> Vec<BufferRef> {
                match *self {
                    DispatchOp::Matmul(ref op) => op.inputs(),
                    DispatchOp::Convolution(ref op) => op.inputs(),
                    DispatchOp::Pointwise(ref op) => op.inputs(),
                    DispatchOp::Reduction(ref op) => op.inputs(),
                    $(DispatchOp::$variant(ref op) => op.inputs(),)*
                }
            }

            pub fn visit_inputs(&self, mut visit: impl FnMut(BufferRef)) {
                match *self {
                    DispatchOp::Matmul(ref op) => op.visit_inputs(&mut visit),
                    DispatchOp::Convolution(ref op) => op.visit_inputs(&mut visit),
                    DispatchOp::Pointwise(ref op) => op.visit_inputs(&mut visit),
                    DispatchOp::Reduction(ref op) => op.visit_inputs(&mut visit),
                    $(DispatchOp::$variant(ref op) => op.visit_inputs(&mut visit),)*
                }
            }

            pub fn visit_outputs(&self, mut visit: impl FnMut(BufferRef)) {
                match *self {
                    DispatchOp::Matmul(ref op) => op.visit_outputs(&mut visit),
                    DispatchOp::Convolution(ref op) => op.visit_outputs(&mut visit),
                    DispatchOp::Pointwise(ref op) => op.visit_outputs(&mut visit),
                    DispatchOp::Reduction(ref op) => op.visit_outputs(&mut visit),
                    $(DispatchOp::$variant(ref op) => op.visit_outputs(&mut visit),)*
                }
            }

            pub fn output(&self) -> BufferRef {
                match *self {
                    DispatchOp::Matmul(ref op) => op.output(),
                    DispatchOp::Convolution(ref op) => op.output(),
                    DispatchOp::Pointwise(ref op) => op.output(),
                    DispatchOp::Reduction(ref op) => op.output(),
                    $(DispatchOp::$variant(ref op) => op.output(),)*
                }
            }

            pub fn extra_outputs(&self) -> Vec<BufferRef> {
                match *self {
                    DispatchOp::Matmul(ref op) => op.extra_outputs(),
                    DispatchOp::Convolution(ref op) => op.extra_outputs(),
                    DispatchOp::Pointwise(ref op) => op.extra_outputs(),
                    DispatchOp::Reduction(ref op) => op.extra_outputs(),
                    $(DispatchOp::$variant(ref op) => op.extra_outputs(),)*
                }
            }

            pub fn map_buffers(&mut self, mut map: impl FnMut(BufferRef) -> BufferRef) {
                match *self {
                    DispatchOp::Matmul(ref mut op) => op.map_buffers(&mut map),
                    DispatchOp::Convolution(ref mut op) => op.map_buffers(&mut map),
                    DispatchOp::Pointwise(ref mut op) => op.map_buffers(&mut map),
                    DispatchOp::Reduction(ref mut op) => op.map_buffers(&mut map),
                    $(DispatchOp::$variant(ref mut op) => op.map_buffers(&mut map),)*
                }
            }

            pub fn set_input(&mut self, index: usize, buffer: BufferRef) {
                match *self {
                    DispatchOp::Matmul(ref mut op) => op.set_input(index, buffer),
                    DispatchOp::Convolution(ref mut op) => op.set_input(index, buffer),
                    DispatchOp::Pointwise(ref mut op) => op.set_input(index, buffer),
                    DispatchOp::Reduction(ref mut op) => op.set_input(index, buffer),
                    $(DispatchOp::$variant(ref mut op) => op.set_input(index, buffer),)*
                }
            }

            pub fn set_output(&mut self, buffer: BufferRef) {
                match *self {
                    DispatchOp::Matmul(ref mut op) => op.set_output(buffer),
                    DispatchOp::Convolution(ref mut op) => op.set_output(buffer),
                    DispatchOp::Pointwise(ref mut op) => op.set_output(buffer),
                    DispatchOp::Reduction(ref mut op) => op.set_output(buffer),
                    $(DispatchOp::$variant(ref mut op) => op.set_output(buffer),)*
                }
            }

            pub fn parameter_words(&self) -> Vec<u32> {
                match *self {
                    DispatchOp::Matmul(ref op) => op.parameter_words(),
                    DispatchOp::Convolution(ref op) => op.parameter_words(),
                    DispatchOp::Pointwise(ref op) => op.parameter_words(),
                    DispatchOp::Reduction(ref op) => op.parameter_words(),
                    $(DispatchOp::$variant(ref op) => op.parameter_words(),)*
                }
            }
        }
    };
}

operations! {
    MultiHeadAttn(AttentionForward),
    FlashAttention(AttentionForward),
    FlashAttentionCoop(AttentionForward),
    MultiHeadAttnGradQ(AttentionGradQ),
    FlashGradQ(AttentionGradQ),
    FlashGradQCoop(AttentionGradQ),
    MultiHeadAttnGradKV(AttentionGradKV),
    FlashGradKV(AttentionGradKV),
    FlashGradKVCoop(AttentionGradKV),
    CachedAttention(CachedAttention),
    CachedQueryAttention(CachedAttention),
    CachedBlockAttention(CachedBlockAttention),
    CachedBlockAttentionSplit(CachedBlockAttention),
    CachedBlockAttentionCombine(CachedAttentionCombine),
    ChunkedRelativeAttention(ChunkedRelativeAttention),
    PrefixLast(PrefixLast),
    CacheWrite(CacheWrite),
    CacheWritePrefix(CacheWritePrefix),
    RmsNormAdd(NormResidual),
    LayerNorm(LayerNorm),
    RmsNormRsqrt(RmsNormRsqrt),
    RmsNormGradW(NormGradient),
    RmsNormGradWRowPar(NormGradient),
    RmsNormGradX(NormGradient),
    LayerNormGradWB(NormGradient),
    LayerNormGradX(NormGradient),
    GroupNorm(GroupNorm),
    GroupNormSilu(GroupNorm),
    GroupNormStats(GroupNormStats),
    GroupNormApply(GroupNormApply),
    GroupNormGradInput(GroupNormGradInput),
    GroupNormGradWeightBias(GroupNormGradWeightBias),
    GroupNormGradStats(GroupNormStats),
    SwiGLUConcat(GatedActivation),
    SwiGLUConcatGrad(GatedActivation),
    GeGLUConcat(GatedActivation),
    GeGLUConcatGrad(GatedActivation),
    SwiGLUGradGate(GateGradient),
    SwiGLUGradUp(BinaryGradient),
    SiluGrad(BinaryGradient),
    PairwiseGrad(PairwiseGradient),
    SumAll(ReduceAll),
    MeanAll(ReduceAll),
    SumRows(SumRows),
    CrossEntropyLoss(CrossEntropy),
    BceLoss(BinaryLoss),
    ToF16(Unary),
    Transpose(Transpose),
    Embedding(Embedding),
    RoPE(Rope),
    RoPEGrad(Rope),
    RoPEDynamic(RopeDynamic),
    RoPEPositions(RopeDynamic),
    RoPEDynamicFactors(RopeFactors),
    Concat(Concat),
    SplitA(Split),
    SplitB(Split),
    MulPerChannel(MulPerChannel),
    ScatterAdd(Scatter),
    ScatterAddAtomic(AtomicScatter),
    Conv2dDw(DepthwiseConv),
    WinogradInputTransform(WinogradInputTransform),
    WinogradOutputTransform(WinogradOutputTransform),
    WinogradWeightTransform(WinogradWeightTransform),
    MaxPool2d(MaxPool),
    MaxPool2dGrad(MaxPoolGradient),
    GlobalAvgPool(GlobalAvgPool),
    GlobalAvgPoolGrad(RowBroadcast),
    Upsample2x(Upsample),
    Upsample2xGrad(Upsample),
}
