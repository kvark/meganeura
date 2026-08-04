use std::fmt;

pub type NodeId = u32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum DType {
    F32,
    F16,
    U32,
    /// Asymmetric 4-bit quantization (Q4_1-style), 32-element blocks.
    /// Each block: 1 f16 scale + 1 f16 min (4 bytes) + 16 packed nibble bytes = 20 bytes.
    /// The `Q4_0` variant name is retained for API compatibility; unlike the
    /// GGML Q4_0 format, this representation stores a per-block minimum.
    Q4_0,
    /// Q8_0: symmetric 8-bit quantization, 32-element blocks.
    /// Each block: 1 f16 scale (padded to u32) + 32 int8 values (8 u32s) = 36 bytes.
    Q8_0,
}

impl DType {
    #[track_caller]
    pub fn size_bytes(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 => 2,
            DType::U32 => 4,
            DType::Q4_0 | DType::Q8_0 => panic!("quantized types use block-level sizing"),
        }
    }
}

#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct TensorType {
    pub shape: Vec<usize>,
    pub dtype: DType,
}

impl TensorType {
    pub fn new(shape: Vec<usize>, dtype: DType) -> Self {
        Self { shape, dtype }
    }

    pub fn f32(shape: Vec<usize>) -> Self {
        Self::new(shape, DType::F32)
    }

    pub fn f16(shape: Vec<usize>) -> Self {
        Self::new(shape, DType::F16)
    }

    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn size_bytes(&self) -> usize {
        match self.dtype {
            DType::Q4_0 => {
                // Q4_1: 32-element blocks, 20 bytes each.
                // Per block: 1 u32 (d_f16 | m_f16) + 4 u32s (16 bytes nibbles) = 5 u32s.
                let blocks = self.num_elements().div_ceil(32);
                blocks * 5 * 4
            }
            DType::Q8_0 => {
                // Q8_0: 32-element blocks, 36 bytes each.
                // Per block: 1 u32 (scale_f16 padded) + 8 u32s (32 int8s) = 9 u32s.
                let blocks = self.num_elements().div_ceil(32);
                blocks * 9 * 4
            }
            _ => self.num_elements() * self.dtype.size_bytes(),
        }
    }

    pub fn rank(&self) -> usize {
        self.shape.len()
    }
}

impl fmt::Display for TensorType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}<{:?}>", self.dtype, self.shape)
    }
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum Op {
    // Leaf nodes
    Parameter {
        name: String,
    },
    Input {
        name: String,
    },
    Constant {
        data: Vec<f32>,
    },

    // Binary
    MatMul,
    // MatMulAT: C = A^T @ B  (A stored as [K,M], B stored as [K,N], C is [M,N])
    MatMulAT,
    // MatMulBT: C = A @ B^T  (A stored as [M,K], B stored as [N,K], C is [M,N])
    MatMulBT,
    Add,
    Mul,

    // Unary
    Relu,
    Sigmoid,
    Tanh,
    Neg,
    Abs,
    Log,
    Recip,
    Exp,

    // Reduction
    SumAll,
    MeanAll,
    /// Column-wise sum: `[M, N]` → `[N]` (sum over rows)
    SumRows,
    /// Row-wise sum over the inner axis: [M, N] → [M, 1]  (sum over cols).
    /// Lowers to a schedule-template `ReductionKernel` (per-row reduction),
    /// so the fusion pass can fold a pointwise/gather producer into its
    /// prologue. The differentiable equivalent of `matmul(x, ones[N,1])`.
    SumInner,
    /// Backward helper for [`Op::SumInner`]: repeat each `[M, 1]` row across
    /// the inner axis, `[M, 1]` → `[M, N]`.
    BroadcastInner {
        inner: u32,
    },
    /// Normalize every `[M, N]` row by its sum, using
    /// `relu(sum - floor) + floor` as the denominator.
    NormalizeInnerSum {
        inner: u32,
        floor: f32,
    },
    /// Backward helper for [`Op::NormalizeInnerSum`].
    NormalizeInnerSumGrad {
        inner: u32,
        floor: f32,
    },
    /// Repeat each complete inner row `repeats` times:
    /// `[M, N] → [M, N * repeats]`.
    TileInner {
        repeats: u32,
    },
    /// Backward helper for [`Op::TileInner`].
    TileInnerGrad {
        inner: u32,
        repeats: u32,
    },
    /// Squared distance from each `[M, D]` row to `pairs` consecutive
    /// `[M * pairs, D]` rows, producing `[M, pairs]`.
    PairwiseSquaredDistance {
        pairs: u32,
    },
    /// Backward helper for the left input of [`Op::PairwiseSquaredDistance`].
    PairwiseSquaredDistanceGradLeft {
        inner: u32,
        pairs: u32,
    },
    /// Backward helper for the right input of [`Op::PairwiseSquaredDistance`].
    PairwiseSquaredDistanceGradRight {
        inner: u32,
        pairs: u32,
    },
    /// Remove each shared row direction from `pairs` consecutive vectors.
    PairwiseVectorRejection {
        pairs: u32,
    },
    /// Backward helper for vector inputs to [`Op::PairwiseVectorRejection`].
    PairwiseVectorRejectionGradVectors {
        inner: u32,
        pairs: u32,
    },
    /// Backward helper for shared directions in [`Op::PairwiseVectorRejection`].
    PairwiseVectorRejectionGradDirections {
        inner: u32,
        pairs: u32,
    },
    /// Exclusive cumulative sum over the inner axis of a 2D tensor.
    ///
    /// Forward order emits `y[m, n] = sum(x[m, 0..n])`; reverse order emits
    /// `y[m, n] = sum(x[m, n+1..N])`. The two directions are transposes of
    /// one another, so the reverse form is also the forward form's backward.
    ExclusiveCumsum {
        reverse: bool,
    },
    /// Shift rows along the inner axis, filling uncovered elements with zero.
    ShiftInner {
        offset: i32,
    },
    Softmax,

    // Loss
    CrossEntropyLoss,
    BceLoss,

    // Comparison (for autodiff)
    Greater,

    // Transpose (swap last two dims)
    Transpose,

    // Broadcast add (bias add: [M,N] + [N])
    BiasAdd,

    // Fused MatMul + Add: C = A × B + D (inputs: [a, b, d])
    FusedMatMulAdd,
    // Fused MatMulAT + Add: C = A^T × B + D (inputs: [a, b, d])
    FusedMatMulATAdd,
    // Fused MatMulBT + Add: C = A × B^T + D (inputs: [a, b, d])
    FusedMatMulBTAdd,

    // Dead node (consumed by fusion, skip during compilation)
    Nop,
    /// Identity / reshape: zero-cost view with potentially different shape.
    /// Compiled as buffer alias (no GPU dispatch). Backward reshapes grad back.
    Identity,
    /// Copy a tensor into a distinct intermediate buffer.
    ///
    /// Unlike [`Op::Identity`], this is an explicit GPU dispatch and must not
    /// be fused away. It is useful when consumers benefit from staging an
    /// externally-backed or otherwise non-local buffer in device-local memory.
    /// Backward is the identity.
    Materialize,
    /// Forward identity, backward zero. Lets a graph use a value in two
    /// places where one branch's gradient should not flow back to the
    /// shared producer (the standard "detach" / "stop_gradient" op in
    /// other frameworks). Compiled as a buffer alias, so zero forward
    /// cost; in autodiff, no gradient is accumulated for the input.
    StopGradient,

    // Log-softmax (for numerical stability)
    LogSoftmax,

    /// Scatter-add: accumulate src rows into output indexed by indices.
    ScatterAdd {
        vocab_size: usize,
    },

    // --- Transformer ops ---

    // SiLU activation: x * sigmoid(x)
    Silu,

    // SwiGLU: silu(gate) * up  (inputs: [gate, up])
    SwiGLU,

    // SwiGLU on concatenated input: input[M, 2*N] → output[M, N]
    // gate = input[:, :N], up = input[:, N:], out = silu(gate) * up
    SwiGLUConcat,
    // Backward for SwiGLUConcat: (grad_out[M,N], input[M,2*N]) → grad_input[M,2*N]
    SwiGLUConcatGrad,

    // Fused backward gradient ops for SwiGLU and Silu
    // SwiGLUGradGate: (grad_out, gate, up) → grad_gate
    SwiGLUGradGate,
    // SwiGLUGradUp: (grad_out, gate) → grad_up
    SwiGLUGradUp,
    // SiluGrad: (grad_out, x) → grad_x
    SiluGrad,

    // RMSNorm: x / sqrt(mean(x²) + eps) * weight
    // inputs: [x, weight], eps stored as f32 bits in params
    RmsNorm {
        eps: f32,
    },

    // Embedding lookup: indices → table rows
    // inputs: [indices (U32), table (F32)]
    Embedding,
    // inputs: [indices (U32), table (F16)] → f32 output. f16-table gather.
    // input: [x (F32)] → f16. Cast f32→f16; backward = identity.
    ToF16,

    // Rotary position embeddings
    // inputs: [x] or [x, pos_offset_input]
    // pos_offset: static offset added to each row's position (0 for prefill)
    // When inputs has 2 elements, the second is a u32 buffer whose value is
    // added to position (for decode, this is kv_pos).
    RoPE {
        theta: f32,
        pos_offset: u32,
        /// Dimension of each attention head. RoPE rotations are applied
        /// independently within each head. When equal to the last dim of
        /// the input tensor, the behavior is identical to "global" RoPE.
        head_dim: u32,
    },
    /// Backward gradient op for RoPE: applies inverse (transpose) rotation.
    /// Inputs: `[grad_output]`.
    RoPEGrad {
        theta: f32,
        pos_offset: u32,
        head_dim: u32,
    },

    // Fused causal multi-head attention with GQA
    // inputs: [q, k, v] as 2D: q=[seq, num_heads*head_dim], k/v=[seq, num_kv_heads*head_dim]
    CausalAttention {
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
    },
    /// CausalAttention with on-the-fly RoPE: takes un-rotated Q, K, V.
    /// Applies RoPE rotation inside the attention kernel's dot product,
    /// eliminating separate RoPE dispatches. inputs: [Q, K, V]
    CausalAttentionRoPE {
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        rope_theta: f32,
    },

    // --- Vision / VLA ops ---

    // GELU activation: x * 0.5 * (1 + erf(x / sqrt(2)))
    Gelu,

    // Standard Layer Normalization: (x - mean) / sqrt(var + eps) * weight + bias
    // inputs: [x, weight, bias]
    LayerNorm {
        eps: f32,
    },

    // Non-causal (full) multi-head attention with GQA
    // inputs: [q, k, v] as 2D: q=[seq, num_heads*head_dim], k/v=[seq, num_kv_heads*head_dim]
    FullAttention {
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
    },

    // Cross-attention: query attends to key/value from a different sequence
    // inputs: [q, k, v] where q=[q_seq, num_heads*head_dim], k/v=[kv_seq, num_kv_heads*head_dim]
    CrossAttention {
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
    },

    // Differentiable multi-head attention (GQA-capable) with LSE saved for backward.
    // inputs: [q, k, v]
    // output: O [q_seq, num_heads*head_dim]
    // During compilation, also allocates an LSE buffer [q_seq * num_heads].
    // Params reuse cross-attention encoding: [q_seq, kv_seq, (num_heads<<16)|num_kv_heads, head_dim]
    MultiHeadAttn {
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        is_cross: bool,
    },

    // Backward gradient ops for MultiHeadAttn.
    // inputs: [dO, q, k, v]
    // fwd_node: NodeId of the forward MultiHeadAttn — compile looks up O and LSE buffers.
    MultiHeadAttnGradQ {
        fwd_node: NodeId,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        is_cross: bool,
    },
    MultiHeadAttnGradK {
        fwd_node: NodeId,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        is_cross: bool,
    },
    MultiHeadAttnGradV {
        fwd_node: NodeId,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        is_cross: bool,
    },

    // Exact RmsNorm backward: grad_w[j] = sum_i(dy[i,j] * x[i,j] * rsqrt_i)
    // inputs: [dy, x, w] → [cols]
    RmsNormGradW {
        eps: f32,
    },
    // Exact RmsNorm backward: grad_x[i,j] = rsqrt_i * (dy[i,j]*w[j] - x[i,j]*s_i)
    // inputs: [dy, x, w] → [rows, cols]
    RmsNormGradX {
        eps: f32,
    },

    // LayerNorm backward: grad_w and grad_bias (combined shader output [2*cols])
    // inputs: [dy, x, w] → [2 * cols]  (first cols = grad_w, last cols = grad_b)
    LayerNormGradWB {
        eps: f32,
    },
    // LayerNorm backward: grad_x
    // inputs: [dy, x, w] → [rows, cols]
    LayerNormGradX {
        eps: f32,
    },

    // --- Conv2d ops ---

    // 2D convolution: input[N,C_in,H,W] * kernel[C_out,C_in,kH,kW] → output[N,C_out,oH,oW]
    // inputs: [input, kernel]
    // Tensor is stored as a flat 1D array in NCHW order.
    // Shape is tracked as [N*C_out*oH*oW] in the graph (flat), with spatial
    // metadata encoded in the op for dispatch.
    Conv2d {
        // Input spatial: channels, height, width
        in_channels: u32,
        in_h: u32,
        in_w: u32,
        // Kernel spatial
        out_channels: u32,
        kernel_h: u32,
        kernel_w: u32,
        stride: u32,
        padding_h: u32,
        padding_w: u32,
    },

    /// Per-channel broadcast multiply: `dst[n,c,h,w] = src[n,c,h,w] * gate[n,c]`.
    /// inputs: [src, gate] where src is `[N*C*H*W]` and gate is `[N*C]`.
    /// Used by Squeeze-and-Excitation in EfficientNet MBConv blocks.
    MulPerChannel {
        channels: u32,
        spatial: u32,
    },

    /// Per-channel broadcast add: `dst[n,c,h,w] = src[n,c,h,w] + bias[c]`.
    /// inputs: [src, bias] where src is `[N*C*H*W]` and bias is `[C]`.
    /// Used to apply a fused-BN per-channel bias to a conv output without
    /// pre-replicating the `[C]` parameter into a `[N*C*H*W]` buffer.
    AddPerChannel {
        channels: u32,
        spatial: u32,
    },

    /// Depthwise 2D convolution (groups == channels).
    /// inputs: [input, kernel] where input is `[N*C*H*W]` and kernel is `[C*kH*kW]`.
    /// Each output channel `c` reads input channel `c` only.  Used by
    /// EfficientNet MBConv blocks.  No autodiff path — frozen-weights only.
    Conv2dDw {
        channels: u32,
        in_h: u32,
        in_w: u32,
        kernel_h: u32,
        kernel_w: u32,
        stride: u32,
        padding_h: u32,
        padding_w: u32,
    },

    // Conv2d backward w.r.t. input: given grad_output and kernel, produce grad_input.
    // inputs: [grad_output, kernel]
    Conv2dGradInput {
        in_channels: u32,
        in_h: u32,
        in_w: u32,
        out_channels: u32,
        kernel_h: u32,
        kernel_w: u32,
        stride: u32,
        padding_h: u32,
        padding_w: u32,
    },

    // Conv2d backward w.r.t. kernel: given grad_output and input, produce grad_kernel.
    // inputs: [grad_output, input]
    Conv2dGradWeight {
        in_channels: u32,
        in_h: u32,
        in_w: u32,
        out_channels: u32,
        kernel_h: u32,
        kernel_w: u32,
        stride: u32,
        padding_h: u32,
        padding_w: u32,
    },

    /// 2D max pooling: `input[N,C,H,W]` → `output[N,C,oH,oW]`
    MaxPool2d {
        channels: u32,
        in_h: u32,
        in_w: u32,
        kernel_h: u32,
        kernel_w: u32,
        stride: u32,
        padding: u32,
    },

    /// Global average pooling: `input[N*C*H*W]` → `output[N*C]`
    /// Averages over the spatial dimensions (H,W) for each channel.
    GlobalAvgPool {
        channels: u32,
        spatial: u32, // H * W
    },
    /// Backward of GlobalAvgPool: broadcast `grad_output[batch*channels]` →
    /// `[batch*channels*spatial]`
    /// then divide by spatial.
    GlobalAvgPoolGrad {
        channels: u32,
        spatial: u32,
    },

    // --- GroupNorm ---

    // Group normalization: input[N*C*H*W] with weight[C], bias[C]
    // inputs: [x, weight, bias]
    GroupNorm {
        num_groups: u32,
        eps: f32,
        channels: u32,
        spatial: u32, // H * W
    },

    /// Winograd F(2,3) convolution for 3×3 stride-1 kernels.
    /// Inputs: `[input, winograd_weight]` where `winograd_weight` is `[16*Co*Ci]`.
    /// Compiler emits 3 dispatches: input transform, batched matmul, output transform.
    WinogradConv2d {
        in_channels: u32,
        in_h: u32,
        in_w: u32,
        out_channels: u32,
        padding: u32,
    },

    /// Fused GroupNorm + SiLU: normalize then apply SiLU activation.
    /// inputs: [x, weight, bias], same shape as GroupNorm.
    GroupNormSilu {
        num_groups: u32,
        eps: f32,
        channels: u32,
        spatial: u32,
    },

    // GroupNorm backward w.r.t. input
    // inputs: [grad_output, input, weight]
    GroupNormGradInput {
        num_groups: u32,
        eps: f32,
        channels: u32,
        spatial: u32,
    },

    // GroupNorm backward w.r.t. weight and bias (concatenated output [2*C])
    // inputs: [grad_output, input]
    GroupNormGradWeightBias {
        num_groups: u32,
        eps: f32,
        channels: u32,
        spatial: u32,
    },

    // --- Concat / Split ---

    // Concatenate along channel dim: [N,Ca,H,W] ++ [N,Cb,H,W] → [N,Ca+Cb,H,W]
    // inputs: [a, b]
    Concat {
        channels_a: u32,
        channels_b: u32,
        spatial: u32,
    },

    // Split (backward of Concat): extract first Ca channels
    // inputs: [grad_output]  (from [N, Ca+Cb, H, W])
    SplitA {
        channels_a: u32,
        channels_b: u32,
        spatial: u32,
    },

    // Split: extract last Cb channels
    SplitB {
        channels_a: u32,
        channels_b: u32,
        spatial: u32,
    },

    // --- Upsample ---

    // Nearest-neighbor 2x upsample: [N,C,H,W] → [N,C,2H,2W]
    // inputs: [x]
    Upsample2x {
        channels: u32,
        in_h: u32,
        in_w: u32,
    },

    // Backward of Upsample2x: [N,C,2H,2W] → [N,C,H,W] (sum 2×2 blocks)
    Upsample2xGrad {
        channels: u32,
        in_h: u32,
        in_w: u32,
    },

    // --- KV cache ops ---

    // Sliding-window causal attention with GQA.
    // Same as CausalAttention but only attends to the last `window_size` positions.
    // inputs: [q, k, v] as 2D: q=[seq, num_heads*head_dim], k/v=[seq, num_kv_heads*head_dim]
    SlidingWindowAttention {
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        window_size: u32,
    },

    // Write [1, dim] into row kv_pos of [max_seq, dim] cache buffer.
    // inputs: [new_kv, cache_buf], kv_pos read from a u32 input.
    // output: cache_buf (in-place write at row kv_pos)
    CacheWrite,

    // Attention with Q from current token and K/V from pre-allocated cache.
    // inputs: [q, k_cache, v_cache, kv_pos_input]
    // q: [1, num_heads*head_dim], k_cache/v_cache: [max_seq, kv_dim]
    // kv_pos_input: u32 scalar (number of valid cached positions)
    CachedAttention {
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
    },
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct Node {
    pub id: NodeId,
    pub op: Op,
    pub inputs: Vec<NodeId>,
    pub ty: TensorType,
    /// Prevent reduced-precision kernel promotion for numerically sensitive
    /// work derived by autodiff. Forward tensors remain logically f32 too;
    /// this flag only constrains optional runtime accelerations such as
    /// f16-input cooperative matrices.
    #[serde(default)]
    pub requires_full_precision: bool,
    /// Optional human-readable name (e.g. `"blk7.mlp.gate"`). Carried through
    /// autodiff, rewrites, and compilation so dispatch labels, profiler rows,
    /// plan dumps, and debug readback can address values by name instead of
    /// bare node ids.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

/// How a derived parameter is computed from its source(s).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum ParamTransform {
    /// Horizontal concatenation: interleave source columns per row.
    HorizontalConcat,
    /// Winograd F(2,3) weight transform: [Co, Ci, 3, 3] → [16, Co, Ci].
    Winograd3x3 {
        out_channels: usize,
        in_channels: usize,
    },
}

/// A derived parameter is created by the optimizer when fusing ops
/// that require concatenating multiple weights (e.g. gate+up projections).
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct DerivedParam {
    /// Name of the new parameter (e.g. "gate_proj.weight+up_proj.weight")
    pub name: String,
    /// Source parameters to concatenate horizontally: (name, cols)
    pub sources: Vec<(String, usize)>,
    /// Total rows (shared across all sources)
    pub rows: usize,
    /// How to compute this parameter from sources.
    pub transform: ParamTransform,
}

pub struct Graph {
    nodes: Vec<Node>,
    outputs: Vec<NodeId>,
    /// Precision policy inherited by newly appended nodes. Autodiff switches
    /// this on after copying the forward graph so all derivative work is
    /// marked without giving gradient ops a parallel set of IR variants.
    new_nodes_require_full_precision: bool,
    /// Number of trailing entries in `outputs` that are parameter gradients
    /// appended by autodiff (positionally aligned with `param_buffers` in the
    /// compiled plan). The leading `outputs.len() - num_param_grad_outputs`
    /// entries are user-facing outputs supplied via `set_outputs`.
    num_param_grad_outputs: usize,
    /// Parameters created by the optimizer from concatenating original params.
    pub derived_params: Vec<DerivedParam>,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            outputs: Vec::new(),
            new_nodes_require_full_precision: false,
            num_param_grad_outputs: 0,
            derived_params: Vec::new(),
        }
    }

    /// Rebuild the graph with nodes in topological order, removing Nop nodes.
    /// Returns a new graph with consecutive IDs where every node's inputs
    /// have lower IDs than the node itself.
    #[track_caller]
    pub fn toposort(&self) -> Graph {
        // Build adjacency: for each node, which nodes depend on it
        let n = self.nodes.len();
        let mut in_degree = vec![0u32; n];
        let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut is_nop = vec![false; n];

        for (i, node) in self.nodes.iter().enumerate() {
            if matches!(node.op, Op::Nop) {
                is_nop[i] = true;
                continue;
            }
            for &inp in &node.inputs {
                let inp = inp as usize;
                if !is_nop[inp] {
                    in_degree[i] += 1;
                    dependents[inp].push(i);
                }
            }
        }

        // Kahn's algorithm: process nodes with in_degree 0
        let mut queue: Vec<usize> = Vec::new();
        for i in 0..n {
            if !is_nop[i] && in_degree[i] == 0 {
                queue.push(i);
            }
        }

        let mut order: Vec<usize> = Vec::new();
        let mut old_to_new: Vec<Option<NodeId>> = vec![None; n];

        while let Some(old_id) = queue.first().copied() {
            queue.remove(0);
            let new_id = order.len() as NodeId;
            old_to_new[old_id] = Some(new_id);
            order.push(old_id);

            for &dep in &dependents[old_id] {
                in_degree[dep] -= 1;
                if in_degree[dep] == 0 {
                    queue.push(dep);
                }
            }
        }

        // Build new graph with remapped IDs
        let mut new_graph = Graph::new();
        for &old_id in &order {
            let node = &self.nodes[old_id];
            let mut op = node.op.clone();
            match op {
                Op::MultiHeadAttnGradQ {
                    ref mut fwd_node, ..
                }
                | Op::MultiHeadAttnGradK {
                    ref mut fwd_node, ..
                }
                | Op::MultiHeadAttnGradV {
                    ref mut fwd_node, ..
                } => {
                    *fwd_node = old_to_new[*fwd_node as usize]
                        .expect("toposort: attention gradient refers to removed forward node");
                }
                _ => {}
            }
            let new_inputs: Vec<NodeId> = node
                .inputs
                .iter()
                .filter_map(|&inp| old_to_new[inp as usize])
                .collect();
            let new_id = new_graph.add_raw_node_with_precision(
                op,
                new_inputs,
                node.ty.clone(),
                node.requires_full_precision,
            );
            new_graph.nodes[new_id as usize].name = node.name.clone();
        }

        // Remap outputs. An output being Nop'd would silently drop it and
        // shift downstream indices — assert loudly instead so fusion bugs
        // surface at build time rather than as `read_output_by_index`
        // panics at inference time.
        let mut new_outputs: Vec<NodeId> = Vec::with_capacity(self.outputs.len());
        for (i, &out) in self.outputs.iter().enumerate() {
            match old_to_new[out as usize] {
                Some(new_id) => new_outputs.push(new_id),
                None => panic!(
                    "toposort: output #{i} (node {out}, op {:?}) was removed as Nop. \
                     A fusion pass Nop'd a node that's listed in `set_outputs`. \
                     Add an `is_output(id)` guard to that fusion.",
                    self.nodes[out as usize].op,
                ),
            }
        }
        new_graph.set_outputs(new_outputs);
        // Preserve the user/grad output boundary (set_outputs resets it).
        new_graph.num_param_grad_outputs = self.num_param_grad_outputs;
        new_graph.derived_params = self.derived_params.clone();
        new_graph
    }

    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    pub fn node(&self, id: NodeId) -> &Node {
        &self.nodes[id as usize]
    }

    pub fn outputs(&self) -> &[NodeId] {
        &self.outputs
    }

    /// Returns true if `id` is a graph output (user-facing or param-grad).
    /// Fusions must not Nop nodes that are outputs — their values are
    /// read back by the user or consumed by the optimizer.
    pub fn is_output(&self, id: NodeId) -> bool {
        self.outputs.contains(&id)
    }

    /// Number of user-supplied outputs (those passed to `set_outputs`).
    /// The remaining `num_param_grad_outputs()` entries are parameter
    /// gradients appended by autodiff.
    pub fn num_user_outputs(&self) -> usize {
        self.outputs.len() - self.num_param_grad_outputs
    }

    /// Number of trailing outputs that are param gradients (appended by autodiff).
    pub fn num_param_grad_outputs(&self) -> usize {
        self.num_param_grad_outputs
    }

    pub fn set_outputs(&mut self, outputs: Vec<NodeId>) {
        self.outputs = outputs;
        self.num_param_grad_outputs = 0;
    }

    /// Append parameter-gradient outputs after user outputs. Only called by
    /// autodiff. `num_grads` must equal the number of entries appended.
    pub fn append_param_grad_outputs(&mut self, grads: &[NodeId]) {
        self.outputs.extend_from_slice(grads);
        self.num_param_grad_outputs += grads.len();
    }

    pub fn add_raw_node(&mut self, op: Op, inputs: Vec<NodeId>, ty: TensorType) -> NodeId {
        self.add_raw_node_with_precision(op, inputs, ty, self.new_nodes_require_full_precision)
    }

    pub(crate) fn add_raw_node_with_precision(
        &mut self,
        op: Op,
        inputs: Vec<NodeId>,
        ty: TensorType,
        requires_full_precision: bool,
    ) -> NodeId {
        let id = self.nodes.len() as NodeId;
        self.nodes.push(Node {
            id,
            op,
            inputs,
            ty,
            requires_full_precision,
            name: None,
        });
        id
    }

    /// Attach a human-readable name to a node. Names survive autodiff,
    /// rewrites, and toposort, and surface in dispatch labels, profiler rows,
    /// plan dumps, and `Session::read_node`. Returns the id so naming can
    /// wrap a binding: `let h = g.matmul(x, w); let h = g.named(h, "blk0.qkv");`
    pub fn named(&mut self, id: NodeId, name: impl Into<String>) -> NodeId {
        self.nodes[id as usize].name = Some(name.into());
        id
    }

    /// The node's name, if one was attached via [`Graph::named`].
    pub fn node_name(&self, id: NodeId) -> Option<&str> {
        self.nodes[id as usize].name.as_deref()
    }

    /// Field-complete copy. Prefer this over hand-rolled node-by-node
    /// reconstruction, which silently drops any `Node` field it doesn't
    /// know about (that bug has happened with `name`).
    pub fn deep_clone(&self) -> Graph {
        Graph {
            nodes: self.nodes.clone(),
            outputs: self.outputs.clone(),
            new_nodes_require_full_precision: self.new_nodes_require_full_precision,
            num_param_grad_outputs: self.num_param_grad_outputs,
            derived_params: self.derived_params.clone(),
        }
    }

    /// Mark subsequently appended nodes as numerically sensitive derivative
    /// work. This is intentionally crate-private: graph authors express f32
    /// tensors as usual, while autodiff owns the forward/backward boundary.
    pub(crate) fn begin_full_precision_region(&mut self) {
        self.new_nodes_require_full_precision = true;
    }

    pub fn nodes_mut(&mut self) -> &mut Vec<Node> {
        &mut self.nodes
    }

    fn add_node(&mut self, op: Op, inputs: Vec<NodeId>, ty: TensorType) -> NodeId {
        self.add_raw_node(op, inputs, ty)
    }

    // --- Leaf nodes ---

    pub fn input(&mut self, name: &str, shape: &[usize]) -> NodeId {
        let ty = TensorType::f32(shape.to_vec());
        self.add_node(
            Op::Input {
                name: name.to_string(),
            },
            vec![],
            ty,
        )
    }

    pub fn parameter(&mut self, name: &str, shape: &[usize]) -> NodeId {
        let ty = TensorType::f32(shape.to_vec());
        self.add_node(
            Op::Parameter {
                name: name.to_string(),
            },
            vec![],
            ty,
        )
    }

    /// Create a parameter stored as f16 on GPU (half the VRAM of f32).
    /// Weights are converted from f32 to f16 during upload.
    /// Shaders read f16 and cast to f32 for computation.
    pub fn parameter_f16(&mut self, name: &str, shape: &[usize]) -> NodeId {
        let ty = TensorType::f16(shape.to_vec());
        self.add_node(
            Op::Parameter {
                name: name.to_string(),
            },
            vec![],
            ty,
        )
    }

    /// Create a parameter stored in Meganeura's asymmetric Q4 format
    /// (~5 bits/element, Q4_1-style scale + minimum metadata).
    ///
    /// The underlying [`DType::Q4_0`] name is historical; this is not the
    /// symmetric GGML Q4_0 wire format.
    pub fn parameter_q4(&mut self, name: &str, shape: &[usize]) -> NodeId {
        let ty = TensorType::new(shape.to_vec(), DType::Q4_0);
        self.add_node(
            Op::Parameter {
                name: name.to_string(),
            },
            vec![],
            ty,
        )
    }

    /// Create a parameter stored as Q8_0 on GPU (8-bit symmetric, ~9 bits/element).
    pub fn parameter_q8(&mut self, name: &str, shape: &[usize]) -> NodeId {
        let ty = TensorType::new(shape.to_vec(), DType::Q8_0);
        self.add_node(
            Op::Parameter {
                name: name.to_string(),
            },
            vec![],
            ty,
        )
    }

    #[track_caller]
    pub fn constant(&mut self, data: Vec<f32>, shape: &[usize]) -> NodeId {
        assert_eq!(data.len(), shape.iter().product::<usize>());
        let ty = TensorType::f32(shape.to_vec());
        self.add_node(Op::Constant { data }, vec![], ty)
    }

    pub fn scalar(&mut self, value: f32) -> NodeId {
        self.constant(vec![value], &[1])
    }

    // --- Binary ops ---

    #[track_caller]
    pub fn matmul(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let a_shape = &self.node(a).ty.shape;
        let b_shape = &self.node(b).ty.shape;
        assert_eq!(a_shape.len(), 2, "matmul requires 2D tensors");
        assert_eq!(b_shape.len(), 2, "matmul requires 2D tensors");
        assert_eq!(a_shape[1], b_shape[0], "matmul inner dimensions must match");
        let ty = TensorType::f32(vec![a_shape[0], b_shape[1]]);
        self.add_node(Op::MatMul, vec![a, b], ty)
    }

    /// C = A^T @ B  (A is [K, M], B is [K, N], C is [M, N])
    #[track_caller]
    pub fn matmul_at(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let a_shape = &self.node(a).ty.shape;
        let b_shape = &self.node(b).ty.shape;
        assert_eq!(a_shape.len(), 2);
        assert_eq!(b_shape.len(), 2);
        assert_eq!(a_shape[0], b_shape[0], "MatMulAT: K dimensions must match");
        let ty = TensorType::f32(vec![a_shape[1], b_shape[1]]);
        self.add_node(Op::MatMulAT, vec![a, b], ty)
    }

    /// C = A @ B^T  (A is [M, K], B is [N, K], C is [M, N])
    #[track_caller]
    pub fn matmul_bt(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let a_shape = &self.node(a).ty.shape;
        let b_shape = &self.node(b).ty.shape;
        assert_eq!(a_shape.len(), 2);
        assert_eq!(b_shape.len(), 2);
        assert_eq!(a_shape[1], b_shape[1], "MatMulBT: K dimensions must match");
        let ty = TensorType::f32(vec![a_shape[0], b_shape[0]]);
        self.add_node(Op::MatMulBT, vec![a, b], ty)
    }

    #[track_caller]
    pub fn add(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let a_ty = &self.node(a).ty;
        let b_ty = &self.node(b).ty;
        assert_eq!(a_ty.shape, b_ty.shape, "add requires matching shapes");
        let ty = a_ty.clone();
        self.add_node(Op::Add, vec![a, b], ty)
    }

    #[track_caller]
    pub fn bias_add(&mut self, a: NodeId, bias: NodeId) -> NodeId {
        let a_shape = &self.node(a).ty.shape;
        let b_shape = &self.node(bias).ty.shape;
        assert_eq!(a_shape.len(), 2, "bias_add requires 2D input");
        assert_eq!(b_shape.len(), 1, "bias must be 1D");
        assert_eq!(a_shape[1], b_shape[0], "bias size must match last dim");
        let ty = self.node(a).ty.clone();
        self.add_node(Op::BiasAdd, vec![a, bias], ty)
    }

    /// Broadcast-add a `[1, N]` tensor across a `[M, N]` tensor.
    ///
    /// Uses the BiasAdd shader which does `dst[i] = a[i] + b[i % N]`.
    #[track_caller]
    pub fn broadcast_add(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let a_shape = &self.node(a).ty.shape;
        let b_shape = &self.node(b).ty.shape;
        assert_eq!(a_shape.len(), 2, "broadcast_add requires 2D input");
        assert_eq!(b_shape.len(), 2, "broadcast_add requires 2D addend");
        assert_eq!(
            b_shape[0], 1,
            "broadcast_add requires addend with first dim = 1"
        );
        assert_eq!(
            a_shape[1], b_shape[1],
            "broadcast_add requires matching last dim"
        );
        let ty = self.node(a).ty.clone();
        self.add_node(Op::BiasAdd, vec![a, b], ty)
    }

    #[track_caller]
    pub fn mul(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let a_ty = &self.node(a).ty;
        let b_ty = &self.node(b).ty;
        assert_eq!(a_ty.shape, b_ty.shape, "mul requires matching shapes");
        let ty = a_ty.clone();
        self.add_node(Op::Mul, vec![a, b], ty)
    }

    #[track_caller]
    pub fn greater(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let a_ty = &self.node(a).ty;
        let b_ty = &self.node(b).ty;
        assert_eq!(a_ty.shape, b_ty.shape, "greater requires matching shapes");
        let ty = a_ty.clone();
        self.add_node(Op::Greater, vec![a, b], ty)
    }

    // --- Unary ops ---

    pub fn relu(&mut self, x: NodeId) -> NodeId {
        let ty = self.node(x).ty.clone();
        self.add_node(Op::Relu, vec![x], ty)
    }

    pub fn sigmoid(&mut self, x: NodeId) -> NodeId {
        let ty = self.node(x).ty.clone();
        self.add_node(Op::Sigmoid, vec![x], ty)
    }

    pub fn tanh(&mut self, x: NodeId) -> NodeId {
        let ty = self.node(x).ty.clone();
        self.add_node(Op::Tanh, vec![x], ty)
    }

    pub fn neg(&mut self, x: NodeId) -> NodeId {
        let ty = self.node(x).ty.clone();
        self.add_node(Op::Neg, vec![x], ty)
    }

    pub fn abs(&mut self, x: NodeId) -> NodeId {
        let ty = self.node(x).ty.clone();
        self.add_node(Op::Abs, vec![x], ty)
    }

    pub fn log(&mut self, x: NodeId) -> NodeId {
        let ty = self.node(x).ty.clone();
        self.add_node(Op::Log, vec![x], ty)
    }

    pub fn recip(&mut self, x: NodeId) -> NodeId {
        let ty = self.node(x).ty.clone();
        self.add_node(Op::Recip, vec![x], ty)
    }

    pub fn exp(&mut self, x: NodeId) -> NodeId {
        let ty = self.node(x).ty.clone();
        self.add_node(Op::Exp, vec![x], ty)
    }

    /// Reshape: reinterpret the tensor with a new shape (same element count).
    ///
    /// Implemented as `x + 0` with the target shape. The e-graph optimizer
    /// or a future pass could eliminate this, but it's cheap (one element-wise
    /// add of zeros).
    #[track_caller]
    pub fn reshape(&mut self, x: NodeId, new_shape: &[usize]) -> NodeId {
        let old_elems = self.node(x).ty.num_elements();
        let new_elems: usize = new_shape.iter().product();
        assert_eq!(
            old_elems, new_elems,
            "reshape: element count mismatch ({old_elems} vs {new_elems})"
        );
        // Reshape is a zero-cost view — just reinterprets the shape.
        // Uses Identity op (compiled as buffer alias, no GPU dispatch).
        self.add_raw_node(Op::Identity, vec![x], TensorType::f32(new_shape.to_vec()))
    }

    /// Copy `x` into a distinct intermediate buffer.
    ///
    /// This is a memory-placement barrier rather than an algebraic operation:
    /// compilation always emits the copy and pointwise fusion does not remove
    /// it. The returned tensor has the same type and propagates gradients to
    /// `x` unchanged.
    pub fn materialize(&mut self, x: NodeId) -> NodeId {
        let ty = self.node(x).ty.clone();
        assert_eq!(
            ty.dtype,
            DType::F32,
            "materialize only supports f32 tensors"
        );
        self.add_raw_node(Op::Materialize, vec![x], ty)
    }

    /// Forward identity, backward zero — the "detach" / `stop_gradient`
    /// op. Use when a value is consumed by two branches and only one
    /// should send gradient back to the producer.
    pub fn stop_gradient(&mut self, x: NodeId) -> NodeId {
        let ty = self.node(x).ty.clone();
        self.add_node(Op::StopGradient, vec![x], ty)
    }

    /// Element-wise division: `a / b` = `a * recip(b)`.
    pub fn div(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let r = self.recip(b);
        self.mul(a, r)
    }

    // --- Loss ---

    /// Mean squared error: `mean((pred - target)²)`.
    pub fn mse_loss(&mut self, pred: NodeId, target: NodeId) -> NodeId {
        let diff = self.neg(target);
        let diff = self.add(pred, diff);
        let sq = self.mul(diff, diff);
        self.mean_all(sq)
    }

    /// L1 / mean absolute error: `mean(|pred - target|)`.
    pub fn l1_loss(&mut self, pred: NodeId, target: NodeId) -> NodeId {
        let diff = self.neg(target);
        let diff = self.add(pred, diff);
        let a = self.abs(diff);
        self.mean_all(a)
    }

    #[track_caller]
    pub fn transpose(&mut self, x: NodeId) -> NodeId {
        let x_shape = &self.node(x).ty.shape;
        assert_eq!(x_shape.len(), 2, "transpose requires 2D tensor");
        let ty = TensorType::f32(vec![x_shape[1], x_shape[0]]);
        self.add_node(Op::Transpose, vec![x], ty)
    }

    // --- Reductions ---

    pub fn sum_all(&mut self, x: NodeId) -> NodeId {
        let ty = TensorType::f32(vec![1]);
        self.add_node(Op::SumAll, vec![x], ty)
    }

    pub fn mean_all(&mut self, x: NodeId) -> NodeId {
        let ty = TensorType::f32(vec![1]);
        self.add_node(Op::MeanAll, vec![x], ty)
    }

    /// Row-wise sum over the inner axis: `x: [M, N] → [M, 1]`. Equivalent
    /// to `matmul(x, ones[N, 1])` but lowers to a fused-able reduction
    /// kernel.
    #[track_caller]
    pub fn sum_inner(&mut self, x: NodeId) -> NodeId {
        let shape = &self.node(x).ty.shape;
        assert_eq!(
            shape.len(),
            2,
            "sum_inner expects a 2D [M, N] input, got shape {shape:?}"
        );
        let m = shape[0];
        let ty = TensorType::f32(vec![m, 1]);
        self.add_node(Op::SumInner, vec![x], ty)
    }

    /// Repeat each scalar row of `x: [M, 1]` across `inner` columns.
    pub fn broadcast_inner(&mut self, x: NodeId, inner: usize) -> NodeId {
        let shape = &self.node(x).ty.shape;
        assert_eq!(
            shape.len(),
            2,
            "broadcast_inner expects a 2D [M, 1] input, got shape {shape:?}"
        );
        assert_eq!(shape[1], 1, "broadcast_inner expects one input column");
        assert!(
            inner > 0,
            "broadcast_inner needs at least one output column"
        );
        let inner_u32 = u32::try_from(inner).expect("broadcast_inner width exceeds u32");
        let ty = TensorType::f32(vec![shape[0], inner]);
        self.add_raw_node(Op::BroadcastInner { inner: inner_u32 }, vec![x], ty)
    }

    /// Normalize each row of `x: [M, N]` by
    /// `relu(sum(row) - floor) + floor`.
    ///
    /// Rows whose sum does not exceed `floor` keep a normalized sum below one
    /// instead of being forced to sum to one. This is useful for non-negative
    /// weights that may all underflow to zero.
    pub fn normalize_inner_sum(&mut self, x: NodeId, floor: f32) -> NodeId {
        let ty = self.node(x).ty.clone();
        assert_eq!(
            ty.shape.len(),
            2,
            "normalize_inner_sum expects a 2D [M, N] input, got {:?}",
            ty.shape
        );
        assert_eq!(
            ty.dtype,
            DType::F32,
            "normalize_inner_sum expects an f32 input"
        );
        assert!(
            ty.shape[1] > 0,
            "normalize_inner_sum needs a non-empty inner row"
        );
        assert!(
            floor.is_finite() && floor > 0.0,
            "normalize_inner_sum floor must be positive and finite"
        );
        let inner = u32::try_from(ty.shape[1]).expect("normalize_inner_sum width exceeds u32");
        self.add_raw_node(Op::NormalizeInnerSum { inner, floor }, vec![x], ty)
    }

    /// Repeat each complete inner row of `x: [M, N]` `repeats` times.
    pub fn tile_inner(&mut self, x: NodeId, repeats: usize) -> NodeId {
        let shape = &self.node(x).ty.shape;
        assert_eq!(
            shape.len(),
            2,
            "tile_inner expects a 2D [M, N] input, got shape {shape:?}"
        );
        assert!(shape[1] > 0, "tile_inner needs a non-empty inner row");
        assert!(repeats > 0, "tile_inner needs at least one repetition");
        let output_inner = shape[1]
            .checked_mul(repeats)
            .expect("tile_inner output width overflows usize");
        u32::try_from(shape[1]).expect("tile_inner input width exceeds u32");
        let repeats = u32::try_from(repeats).expect("tile_inner repeat count exceeds u32");
        let ty = TensorType::f32(vec![shape[0], output_inner]);
        self.add_raw_node(Op::TileInner { repeats }, vec![x], ty)
    }

    /// Compute squared distances from every `[M, D]` row in `left` to its
    /// `pairs` consecutive rows in `right: [M * pairs, D]`.
    pub fn pairwise_squared_distance(
        &mut self,
        left: NodeId,
        right: NodeId,
        pairs: usize,
    ) -> NodeId {
        let left_shape = self.node(left).ty.shape.clone();
        let right_shape = self.node(right).ty.shape.clone();
        assert_eq!(
            left_shape.len(),
            2,
            "pairwise_squared_distance expects a 2D left input, got {left_shape:?}"
        );
        assert_eq!(
            right_shape.len(),
            2,
            "pairwise_squared_distance expects a 2D right input, got {right_shape:?}"
        );
        assert!(
            left_shape[1] > 0,
            "pairwise distance rows must be non-empty"
        );
        assert!(pairs > 0, "pairwise distance needs at least one pair");
        let right_rows = left_shape[0]
            .checked_mul(pairs)
            .expect("pairwise distance row count overflows usize");
        assert_eq!(
            right_shape,
            [right_rows, left_shape[1]],
            "pairwise distance right shape must be [M * pairs, D]"
        );
        u32::try_from(left_shape[1]).expect("pairwise distance width exceeds u32");
        let pairs = u32::try_from(pairs).expect("pairwise distance pair count exceeds u32");
        let ty = TensorType::f32(vec![left_shape[0], pairs as usize]);
        self.add_raw_node(Op::PairwiseSquaredDistance { pairs }, vec![left, right], ty)
    }

    /// Remove each `[M, D]` unit direction's component from its `pairs`
    /// consecutive vectors in `[M * pairs, D]`.
    pub fn pairwise_vector_rejection(
        &mut self,
        vectors: NodeId,
        directions: NodeId,
        pairs: usize,
    ) -> NodeId {
        let vector_ty = self.node(vectors).ty.clone();
        let direction_shape = self.node(directions).ty.shape.clone();
        assert_eq!(
            vector_ty.shape.len(),
            2,
            "pairwise_vector_rejection expects 2D vectors, got {:?}",
            vector_ty.shape
        );
        assert_eq!(
            direction_shape.len(),
            2,
            "pairwise_vector_rejection expects 2D directions, got {direction_shape:?}"
        );
        assert!(
            vector_ty.shape[1] > 0,
            "pairwise vector rows must be non-empty"
        );
        assert!(
            pairs > 0,
            "pairwise vector rejection needs at least one pair"
        );
        let vector_rows = direction_shape[0]
            .checked_mul(pairs)
            .expect("pairwise vector row count overflows usize");
        assert_eq!(
            vector_ty.shape,
            [vector_rows, direction_shape[1]],
            "pairwise vectors must have shape [M * pairs, D]"
        );
        u32::try_from(direction_shape[1]).expect("pairwise vector width exceeds u32");
        let pairs = u32::try_from(pairs).expect("pairwise vector pair count exceeds u32");
        self.add_raw_node(
            Op::PairwiseVectorRejection { pairs },
            vec![vectors, directions],
            vector_ty,
        )
    }

    /// Exclusive cumulative sum over the inner axis of a 2D tensor.
    ///
    /// With `reverse = false`, each output contains the sum of values before
    /// it in the row. With `reverse = true`, it contains the sum after it.
    pub fn exclusive_cumsum(&mut self, x: NodeId, reverse: bool) -> NodeId {
        let ty = self.node(x).ty.clone();
        assert_eq!(
            ty.shape.len(),
            2,
            "exclusive_cumsum expects a 2D [M, N] input, got shape {:?}",
            ty.shape
        );
        assert_eq!(
            ty.dtype,
            DType::F32,
            "exclusive_cumsum expects an f32 input"
        );
        self.add_node(Op::ExclusiveCumsum { reverse }, vec![x], ty)
    }

    /// Shift each row by `offset` elements along its inner axis.
    ///
    /// Positive offsets move values toward larger indices; negative offsets
    /// move them toward smaller indices. Newly uncovered elements are zero.
    pub fn shift_inner(&mut self, x: NodeId, offset: i32) -> NodeId {
        let ty = self.node(x).ty.clone();
        assert_eq!(
            ty.shape.len(),
            2,
            "shift_inner expects a 2D [M, N] input, got shape {:?}",
            ty.shape
        );
        assert_eq!(ty.dtype, DType::F32, "shift_inner expects an f32 input");
        assert_ne!(
            offset,
            i32::MIN,
            "shift_inner offset must be safely negatable"
        );
        self.add_node(Op::ShiftInner { offset }, vec![x], ty)
    }

    pub fn softmax(&mut self, x: NodeId) -> NodeId {
        let ty = self.node(x).ty.clone();
        self.add_node(Op::Softmax, vec![x], ty)
    }

    pub fn log_softmax(&mut self, x: NodeId) -> NodeId {
        let ty = self.node(x).ty.clone();
        self.add_node(Op::LogSoftmax, vec![x], ty)
    }

    // --- Transformer ops ---

    pub fn silu(&mut self, x: NodeId) -> NodeId {
        let ty = self.node(x).ty.clone();
        self.add_node(Op::Silu, vec![x], ty)
    }

    /// Fused SwiGLU: silu(gate) * up. gate and up must have the same shape.
    pub fn swiglu(&mut self, gate: NodeId, up: NodeId) -> NodeId {
        let ty = self.node(gate).ty.clone();
        self.add_node(Op::SwiGLU, vec![gate, up], ty)
    }

    /// SwiGLU on concatenated input: input[M, 2*N] → output[M, N].
    /// Reads gate from first half, up from second half.
    #[track_caller]
    pub fn swiglu_concat(&mut self, input: NodeId) -> NodeId {
        let in_shape = &self.node(input).ty.shape;
        assert_eq!(in_shape.len(), 2);
        assert_eq!(in_shape[1] % 2, 0, "SwiGLUConcat requires even N");
        let ty = TensorType::f32(vec![in_shape[0], in_shape[1] / 2]);
        self.add_raw_node(Op::SwiGLUConcat, vec![input], ty)
    }

    /// Fused SwiGLU backward: grad_gate = grad_out * up * dsilu(gate)
    pub fn swiglu_grad_gate(&mut self, grad_out: NodeId, gate: NodeId, up: NodeId) -> NodeId {
        let ty = self.node(gate).ty.clone();
        self.add_raw_node(Op::SwiGLUGradGate, vec![grad_out, gate, up], ty)
    }

    /// Fused SwiGLU backward: grad_up = grad_out * silu(gate)
    pub fn swiglu_grad_up(&mut self, grad_out: NodeId, gate: NodeId) -> NodeId {
        let ty = self.node(gate).ty.clone();
        self.add_raw_node(Op::SwiGLUGradUp, vec![grad_out, gate], ty)
    }

    /// Fused Silu backward: grad_x = grad_out * dsilu(x)
    pub fn silu_grad(&mut self, grad_out: NodeId, x: NodeId) -> NodeId {
        let ty = self.node(x).ty.clone();
        self.add_raw_node(Op::SiluGrad, vec![grad_out, x], ty)
    }

    #[track_caller]
    pub fn rms_norm(&mut self, x: NodeId, weight: NodeId, eps: f32) -> NodeId {
        let x_shape = &self.node(x).ty.shape;
        let w_shape = &self.node(weight).ty.shape;
        assert_eq!(x_shape.len(), 2, "rms_norm requires 2D input");
        assert_eq!(w_shape.len(), 1, "rms_norm weight must be 1D");
        assert_eq!(
            x_shape[1], w_shape[0],
            "rms_norm weight size must match last dim"
        );
        let ty = self.node(x).ty.clone();
        self.add_node(Op::RmsNorm { eps }, vec![x, weight], ty)
    }

    pub fn rms_norm_grad_w(&mut self, dy: NodeId, x: NodeId, w: NodeId, eps: f32) -> NodeId {
        let w_ty = self.node(w).ty.clone();
        self.add_raw_node(Op::RmsNormGradW { eps }, vec![dy, x, w], w_ty)
    }

    pub fn rms_norm_grad_x(&mut self, dy: NodeId, x: NodeId, w: NodeId, eps: f32) -> NodeId {
        let x_ty = self.node(x).ty.clone();
        self.add_raw_node(Op::RmsNormGradX { eps }, vec![dy, x, w], x_ty)
    }

    pub fn layer_norm_grad_wb(&mut self, dy: NodeId, x: NodeId, w: NodeId, eps: f32) -> NodeId {
        let w_ty = self.node(w).ty.clone();
        self.add_raw_node(Op::LayerNormGradWB { eps }, vec![dy, x, w], w_ty)
    }

    pub fn layer_norm_grad_x(&mut self, dy: NodeId, x: NodeId, w: NodeId, eps: f32) -> NodeId {
        let x_ty = self.node(x).ty.clone();
        self.add_raw_node(Op::LayerNormGradX { eps }, vec![dy, x, w], x_ty)
    }

    pub fn input_u32(&mut self, name: &str, shape: &[usize]) -> NodeId {
        let ty = TensorType::new(shape.to_vec(), DType::U32);
        self.add_node(
            Op::Input {
                name: name.to_string(),
            },
            vec![],
            ty,
        )
    }

    #[track_caller]
    pub fn embedding(&mut self, indices: NodeId, table: NodeId) -> NodeId {
        let idx_shape = &self.node(indices).ty.shape;
        let tbl_shape = &self.node(table).ty.shape;
        assert_eq!(
            self.node(indices).ty.dtype,
            DType::U32,
            "embedding indices must be U32"
        );
        assert_eq!(idx_shape.len(), 1, "embedding indices must be 1D");
        assert_eq!(tbl_shape.len(), 2, "embedding table must be 2D");
        let seq_len = idx_shape[0];
        let hidden = tbl_shape[1];
        let ty = TensorType::f32(vec![seq_len, hidden]);
        self.add_node(Op::Embedding, vec![indices, table], ty)
    }

    /// Cast an f32 tensor to f16 (same shape, `DType::F16`). Forward
    /// rounds to half precision; backward is the identity (the standard
    /// mixed-precision "straight-through" — f32 master weights, f16 copy
    /// for a bandwidth-bound forward read). Pair with [`Self::embedding_f16`]
    /// to halve the bytes read by a scattered gather.
    pub fn to_f16(&mut self, x: NodeId) -> NodeId {
        let shape = self.node(x).ty.shape.clone();
        let ty = TensorType::new(shape, DType::F16);
        self.add_node(Op::ToF16, vec![x], ty)
    }

    /// Embedding gather from an **f16** table → f32 output. Same as
    /// [`Self::embedding`] but the table is `DType::F16` (half the bytes
    /// read per gathered element). Backward scatter-adds f32 gradients
    /// into the table's grad (so it composes with `to_f16` → the f32
    /// parameter).
    #[track_caller]
    pub fn embedding_f16(&mut self, indices: NodeId, table: NodeId) -> NodeId {
        let idx_shape = &self.node(indices).ty.shape;
        let tbl = self.node(table);
        assert_eq!(
            self.node(indices).ty.dtype,
            DType::U32,
            "embedding_f16 indices must be U32"
        );
        assert_eq!(
            tbl.ty.dtype,
            DType::F16,
            "embedding_f16 table must be F16 (use to_f16)"
        );
        assert_eq!(idx_shape.len(), 1, "embedding_f16 indices must be 1D");
        assert_eq!(tbl.ty.shape.len(), 2, "embedding_f16 table must be 2D");
        let seq_len = idx_shape[0];
        let hidden = tbl.ty.shape[1];
        let ty = TensorType::f32(vec![seq_len, hidden]);
        self.add_node(Op::Embedding, vec![indices, table], ty)
    }

    /// Scatter-add: accumulate `src[i]` rows into `output[indices[i]]`.
    #[track_caller]
    pub fn scatter_add(&mut self, indices: NodeId, src: NodeId, vocab_size: usize) -> NodeId {
        let src_shape = &self.node(src).ty.shape;
        assert_eq!(src_shape.len(), 2);
        let embed_dim = src_shape[1];
        let ty = TensorType::f32(vec![vocab_size, embed_dim]);
        self.add_node(Op::ScatterAdd { vocab_size }, vec![indices, src], ty)
    }

    pub fn rope(&mut self, x: NodeId, theta: f32, head_dim: u32) -> NodeId {
        self.rope_with_offset(x, theta, 0, head_dim)
    }

    pub fn rope_grad(
        &mut self,
        grad_output: NodeId,
        theta: f32,
        pos_offset: u32,
        head_dim: u32,
    ) -> NodeId {
        let ty = self.node(grad_output).ty.clone();
        self.add_raw_node(
            Op::RoPEGrad {
                theta,
                pos_offset,
                head_dim,
            },
            vec![grad_output],
            ty,
        )
    }

    #[track_caller]
    pub fn rope_with_offset(
        &mut self,
        x: NodeId,
        theta: f32,
        pos_offset: u32,
        head_dim: u32,
    ) -> NodeId {
        let x_shape = &self.node(x).ty.shape;
        assert_eq!(x_shape.len(), 2, "rope requires 2D input");
        let dim = x_shape[1] as u32;
        assert_eq!(dim % 2, 0, "rope requires even last dim");
        assert_eq!(dim % head_dim, 0, "rope: dim must be divisible by head_dim");
        assert_eq!(head_dim % 2, 0, "rope: head_dim must be even");
        let ty = self.node(x).ty.clone();
        self.add_node(
            Op::RoPE {
                theta,
                pos_offset,
                head_dim,
            },
            vec![x],
            ty,
        )
    }

    /// RoPE with a dynamic position offset read from an input buffer.
    /// The position for each row is `row_index + offset_buf[0]`.
    #[track_caller]
    pub fn rope_dynamic_offset(
        &mut self,
        x: NodeId,
        theta: f32,
        offset_input: NodeId,
        head_dim: u32,
    ) -> NodeId {
        let x_shape = &self.node(x).ty.shape;
        assert_eq!(x_shape.len(), 2, "rope requires 2D input");
        let dim = x_shape[1] as u32;
        assert_eq!(dim % 2, 0, "rope requires even last dim");
        assert_eq!(dim % head_dim, 0, "rope: dim must be divisible by head_dim");
        assert_eq!(head_dim % 2, 0, "rope: head_dim must be even");
        let ty = self.node(x).ty.clone();
        self.add_node(
            Op::RoPE {
                theta,
                pos_offset: 0,
                head_dim,
            },
            vec![x, offset_input],
            ty,
        )
    }

    #[track_caller]
    pub fn causal_attention(
        &mut self,
        q: NodeId,
        k: NodeId,
        v: NodeId,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
    ) -> NodeId {
        let q_shape = &self.node(q).ty.shape;
        let k_shape = &self.node(k).ty.shape;
        let v_shape = &self.node(v).ty.shape;
        assert_eq!(q_shape.len(), 2, "q must be 2D");
        assert_eq!(k_shape.len(), 2, "k must be 2D");
        assert_eq!(v_shape.len(), 2, "v must be 2D");
        let seq = q_shape[0];
        assert_eq!(
            q_shape[1],
            (num_heads * head_dim) as usize,
            "q dim mismatch"
        );
        assert_eq!(k_shape[0], seq, "k seq must match q seq");
        assert_eq!(
            k_shape[1],
            (num_kv_heads * head_dim) as usize,
            "k dim mismatch"
        );
        assert_eq!(v_shape[0], seq, "v seq must match q seq");
        assert_eq!(
            v_shape[1],
            (num_kv_heads * head_dim) as usize,
            "v dim mismatch"
        );
        let ty = TensorType::f32(vec![seq, (num_heads * head_dim) as usize]);
        self.add_node(
            Op::CausalAttention {
                num_heads,
                num_kv_heads,
                head_dim,
            },
            vec![q, k, v],
            ty,
        )
    }

    /// Sliding-window causal attention with GQA.
    ///
    /// Same as `causal_attention` but each position only attends to the
    /// last `window_size` positions (inclusive).
    #[allow(clippy::too_many_arguments)]
    #[track_caller]
    pub fn sliding_window_attention(
        &mut self,
        q: NodeId,
        k: NodeId,
        v: NodeId,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        window_size: u32,
    ) -> NodeId {
        let q_shape = &self.node(q).ty.shape;
        let k_shape = &self.node(k).ty.shape;
        let v_shape = &self.node(v).ty.shape;
        assert_eq!(q_shape.len(), 2, "q must be 2D");
        assert_eq!(k_shape.len(), 2, "k must be 2D");
        assert_eq!(v_shape.len(), 2, "v must be 2D");
        let seq = q_shape[0];
        assert_eq!(
            q_shape[1],
            (num_heads * head_dim) as usize,
            "q dim mismatch"
        );
        assert_eq!(k_shape[0], seq, "k seq must match q seq");
        assert_eq!(
            k_shape[1],
            (num_kv_heads * head_dim) as usize,
            "k dim mismatch"
        );
        assert_eq!(v_shape[0], seq, "v seq must match q seq");
        assert_eq!(
            v_shape[1],
            (num_kv_heads * head_dim) as usize,
            "v dim mismatch"
        );
        assert!(window_size > 0, "window_size must be > 0");
        let ty = TensorType::f32(vec![seq, (num_heads * head_dim) as usize]);
        self.add_node(
            Op::SlidingWindowAttention {
                num_heads,
                num_kv_heads,
                head_dim,
                window_size,
            },
            vec![q, k, v],
            ty,
        )
    }

    // --- GroupNorm ops ---

    /// Group normalization. Input is flat `[N*C*H*W]`, weight `[C]`, bias `[C]`.
    #[allow(clippy::too_many_arguments)]
    pub fn group_norm(
        &mut self,
        x: NodeId,
        weight: NodeId,
        bias: NodeId,
        _batch: u32,
        channels: u32,
        spatial: u32,
        num_groups: u32,
        eps: f32,
    ) -> NodeId {
        let ty = self.node(x).ty.clone();
        self.add_node(
            Op::GroupNorm {
                num_groups,
                eps,
                channels,
                spatial,
            },
            vec![x, weight, bias],
            ty,
        )
    }

    /// GroupNorm backward w.r.t. input.
    #[allow(clippy::too_many_arguments)]
    pub fn group_norm_grad_input(
        &mut self,
        grad_output: NodeId,
        input: NodeId,
        weight: NodeId,
        batch: u32,
        channels: u32,
        spatial: u32,
        num_groups: u32,
        eps: f32,
    ) -> NodeId {
        let in_size = batch as usize * channels as usize * spatial as usize;
        let ty = TensorType::f32(vec![in_size]);
        self.add_raw_node(
            Op::GroupNormGradInput {
                num_groups,
                eps,
                channels,
                spatial,
            },
            vec![grad_output, input, weight],
            ty,
        )
    }

    /// GroupNorm backward w.r.t. weight+bias (concatenated `[2*C]` output).
    #[allow(clippy::too_many_arguments)]
    pub fn group_norm_grad_weight_bias(
        &mut self,
        grad_output: NodeId,
        input: NodeId,
        channels: u32,
        spatial: u32,
        num_groups: u32,
        eps: f32,
    ) -> NodeId {
        let ty = TensorType::f32(vec![2 * channels as usize]);
        self.add_raw_node(
            Op::GroupNormGradWeightBias {
                num_groups,
                eps,
                channels,
                spatial,
            },
            vec![grad_output, input],
            ty,
        )
    }

    // --- Concat / Split ops ---

    /// Concatenate two tensors along the channel dimension (NCHW).
    /// Both inputs must be flat 1D tensors. `spatial` = H * W.
    pub fn concat(
        &mut self,
        a: NodeId,
        b: NodeId,
        batch: u32,
        channels_a: u32,
        channels_b: u32,
        spatial: u32,
    ) -> NodeId {
        let total = batch as usize * (channels_a + channels_b) as usize * spatial as usize;
        let ty = TensorType::f32(vec![total]);
        self.add_node(
            Op::Concat {
                channels_a,
                channels_b,
                spatial,
            },
            vec![a, b],
            ty,
        )
    }

    /// Split first Ca channels from `[N, Ca+Cb, H, W]`.
    pub fn split_a(
        &mut self,
        x: NodeId,
        batch: u32,
        channels_a: u32,
        channels_b: u32,
        spatial: u32,
    ) -> NodeId {
        let total = batch as usize * channels_a as usize * spatial as usize;
        let ty = TensorType::f32(vec![total]);
        self.add_raw_node(
            Op::SplitA {
                channels_a,
                channels_b,
                spatial,
            },
            vec![x],
            ty,
        )
    }

    /// Split last Cb channels from `[N, Ca+Cb, H, W]`.
    pub fn split_b(
        &mut self,
        x: NodeId,
        batch: u32,
        channels_a: u32,
        channels_b: u32,
        spatial: u32,
    ) -> NodeId {
        let total = batch as usize * channels_b as usize * spatial as usize;
        let ty = TensorType::f32(vec![total]);
        self.add_raw_node(
            Op::SplitB {
                channels_a,
                channels_b,
                spatial,
            },
            vec![x],
            ty,
        )
    }

    // --- Upsample ops ---

    /// Nearest-neighbor 2x upsampling: `[N,C,H,W]` → `[N,C,2H,2W]`.
    pub fn upsample_2x(
        &mut self,
        x: NodeId,
        batch: u32,
        channels: u32,
        in_h: u32,
        in_w: u32,
    ) -> NodeId {
        let total = batch as usize * channels as usize * (in_h * 2) as usize * (in_w * 2) as usize;
        let ty = TensorType::f32(vec![total]);
        self.add_node(
            Op::Upsample2x {
                channels,
                in_h,
                in_w,
            },
            vec![x],
            ty,
        )
    }

    /// Backward of 2x upsample: `[N,C,2H,2W]` → `[N,C,H,W]`.
    pub fn upsample_2x_grad(
        &mut self,
        grad_output: NodeId,
        batch: u32,
        channels: u32,
        in_h: u32,
        in_w: u32,
    ) -> NodeId {
        let total = batch as usize * channels as usize * in_h as usize * in_w as usize;
        let ty = TensorType::f32(vec![total]);
        self.add_raw_node(
            Op::Upsample2xGrad {
                channels,
                in_h,
                in_w,
            },
            vec![grad_output],
            ty,
        )
    }

    // --- Conv2d ops ---

    /// 2D convolution: input[N, C_in, H, W] * kernel[C_out, C_in, kH, kW] → output[N, C_out, oH, oW].
    ///
    /// Tensors are flat 1D arrays in NCHW order. `input` shape must be `[N * C_in * H * W]`
    /// and `kernel` shape `[C_out * C_in * kH * kW]` (both stored as single-dim in the graph).
    #[allow(clippy::too_many_arguments)]
    pub fn conv2d(
        &mut self,
        input: NodeId,
        kernel: NodeId,
        batch: u32,
        in_channels: u32,
        in_h: u32,
        in_w: u32,
        out_channels: u32,
        kernel_h: u32,
        kernel_w: u32,
        stride: u32,
        padding: u32,
    ) -> NodeId {
        self.conv2d_hw(
            input,
            kernel,
            batch,
            in_channels,
            in_h,
            in_w,
            out_channels,
            kernel_h,
            kernel_w,
            stride,
            padding,
            padding,
        )
    }

    /// Conv2d with separate height/width padding (for Conv1d emulation etc.).
    #[allow(clippy::too_many_arguments)]
    pub fn conv2d_hw(
        &mut self,
        input: NodeId,
        kernel: NodeId,
        batch: u32,
        in_channels: u32,
        in_h: u32,
        in_w: u32,
        out_channels: u32,
        kernel_h: u32,
        kernel_w: u32,
        stride: u32,
        padding_h: u32,
        padding_w: u32,
    ) -> NodeId {
        let out_h = (in_h + 2 * padding_h - kernel_h) / stride + 1;
        let out_w = (in_w + 2 * padding_w - kernel_w) / stride + 1;
        let out_size = batch as usize * out_channels as usize * out_h as usize * out_w as usize;
        let ty = TensorType::f32(vec![out_size]);
        self.add_node(
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
            },
            vec![input, kernel],
            ty,
        )
    }

    /// Per-channel broadcast multiply: `dst[n,c,h,w] = src[n,c,h,w] * gate[n,c]`.
    ///
    /// `src` shape `[N*C*H*W]`; `gate` shape `[N*C]`.  Output matches `src`.
    /// Used by EfficientNet's Squeeze-and-Excitation block.
    pub fn mul_per_channel(
        &mut self,
        src: NodeId,
        gate: NodeId,
        channels: u32,
        spatial: u32,
    ) -> NodeId {
        let ty = self.node(src).ty.clone();
        self.add_node(Op::MulPerChannel { channels, spatial }, vec![src, gate], ty)
    }

    /// Per-channel broadcast add: `dst[n,c,h,w] = src[n,c,h,w] + bias[c]`.
    ///
    /// `src` shape `[N*C*H*W]`; `bias` shape `[C]`.  Output matches `src`.
    /// Lets a fused-BN per-channel bias term be applied to a conv output
    /// without pre-replicating the safetensor's `[C]` data into a
    /// `[N*C*H*W]`-sized parameter buffer.  Compare `mul_per_channel`,
    /// whose gate is `[N*C]` (per-batch-and-channel) for the SE pathway.
    pub fn add_per_channel(
        &mut self,
        src: NodeId,
        bias: NodeId,
        channels: u32,
        spatial: u32,
    ) -> NodeId {
        let ty = self.node(src).ty.clone();
        self.add_node(Op::AddPerChannel { channels, spatial }, vec![src, bias], ty)
    }

    /// Depthwise Conv2d (groups == channels).
    ///
    /// `kernel` shape is `[C * kH * kW]`. Output shape is
    /// `[N, C, oH, oW]` flattened (same NCHW layout convention as
    /// [`Self::conv2d`]). No autodiff support — used only with frozen
    /// pretrained weights (e.g. P2P-fine-tuned EfficientNet).
    #[allow(clippy::too_many_arguments)]
    pub fn conv2d_dw(
        &mut self,
        input: NodeId,
        kernel: NodeId,
        batch: u32,
        channels: u32,
        in_h: u32,
        in_w: u32,
        kernel_h: u32,
        kernel_w: u32,
        stride: u32,
        padding_h: u32,
        padding_w: u32,
    ) -> NodeId {
        let out_h = (in_h + 2 * padding_h - kernel_h) / stride + 1;
        let out_w = (in_w + 2 * padding_w - kernel_w) / stride + 1;
        let out_size = batch as usize * channels as usize * out_h as usize * out_w as usize;
        let ty = TensorType::f32(vec![out_size]);
        self.add_node(
            Op::Conv2dDw {
                channels,
                in_h,
                in_w,
                kernel_h,
                kernel_w,
                stride,
                padding_h,
                padding_w,
            },
            vec![input, kernel],
            ty,
        )
    }

    /// Conv2d backward w.r.t. input.
    #[allow(clippy::too_many_arguments)]
    pub fn conv2d_grad_input(
        &mut self,
        grad_output: NodeId,
        kernel: NodeId,
        batch: u32,
        in_channels: u32,
        in_h: u32,
        in_w: u32,
        out_channels: u32,
        kernel_h: u32,
        kernel_w: u32,
        stride: u32,
        padding_h: u32,
        padding_w: u32,
    ) -> NodeId {
        let in_size = batch as usize * in_channels as usize * in_h as usize * in_w as usize;
        let ty = TensorType::f32(vec![in_size]);
        self.add_raw_node(
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
            },
            vec![grad_output, kernel],
            ty,
        )
    }

    /// Conv2d backward w.r.t. kernel weights.
    #[allow(clippy::too_many_arguments)]
    pub fn conv2d_grad_weight(
        &mut self,
        grad_output: NodeId,
        input: NodeId,
        in_channels: u32,
        in_h: u32,
        in_w: u32,
        out_channels: u32,
        kernel_h: u32,
        kernel_w: u32,
        stride: u32,
        padding_h: u32,
        padding_w: u32,
    ) -> NodeId {
        let kernel_size =
            out_channels as usize * in_channels as usize * kernel_h as usize * kernel_w as usize;
        let ty = TensorType::f32(vec![kernel_size]);
        self.add_raw_node(
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
            },
            vec![grad_output, input],
            ty,
        )
    }

    pub fn max_pool_2d(
        &mut self,
        input: NodeId,
        batch: u32,
        channels: u32,
        in_h: u32,
        in_w: u32,
        kernel_h: u32,
        kernel_w: u32,
        stride: u32,
        padding: u32,
    ) -> NodeId {
        let out_h = (in_h + 2 * padding - kernel_h) / stride + 1;
        let out_w = (in_w + 2 * padding - kernel_w) / stride + 1;
        let out_size = batch as usize * channels as usize * out_h as usize * out_w as usize;
        self.add_node(
            Op::MaxPool2d {
                channels,
                in_h,
                in_w,
                kernel_h,
                kernel_w,
                stride,
                padding,
            },
            vec![input],
            TensorType::f32(vec![out_size]),
        )
    }

    pub fn global_avg_pool(
        &mut self,
        input: NodeId,
        batch: u32,
        channels: u32,
        spatial: u32,
    ) -> NodeId {
        self.add_node(
            Op::GlobalAvgPool { channels, spatial },
            vec![input],
            TensorType::f32(vec![batch as usize, channels as usize]),
        )
    }

    /// Write `new_kv` [1, dim] into row `kv_pos` of `cache` [max_seq, dim].
    /// Returns a node representing the updated cache buffer.
    #[track_caller]
    pub fn cache_write(&mut self, new_kv: NodeId, cache: NodeId, kv_pos: NodeId) -> NodeId {
        let nk_shape = &self.node(new_kv).ty.shape;
        let c_shape = &self.node(cache).ty.shape;
        assert_eq!(nk_shape.len(), 2, "new_kv must be 2D");
        assert_eq!(nk_shape[0], 1, "new_kv must have seq_len=1");
        assert_eq!(c_shape.len(), 2, "cache must be 2D");
        assert_eq!(nk_shape[1], c_shape[1], "dim must match");
        let ty = self.node(cache).ty.clone();
        self.add_node(Op::CacheWrite, vec![new_kv, cache, kv_pos], ty)
    }

    /// Cached attention: Q attends to K/V cache.
    /// q: [1, num_heads*head_dim], k_cache/v_cache: [max_seq, kv_dim],
    /// kv_pos: u32 scalar (number of valid positions in cache).
    #[track_caller]
    pub fn cached_attention(
        &mut self,
        q: NodeId,
        k_cache: NodeId,
        v_cache: NodeId,
        kv_pos: NodeId,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
    ) -> NodeId {
        let q_shape = &self.node(q).ty.shape;
        assert_eq!(q_shape.len(), 2, "q must be 2D");
        assert_eq!(q_shape[0], 1, "q must have seq_len=1 for cached attention");
        assert_eq!(
            q_shape[1],
            (num_heads * head_dim) as usize,
            "q dim mismatch"
        );
        let ty = TensorType::f32(vec![1, (num_heads * head_dim) as usize]);
        self.add_node(
            Op::CachedAttention {
                num_heads,
                num_kv_heads,
                head_dim,
            },
            vec![q, k_cache, v_cache, kv_pos],
            ty,
        )
    }

    // --- Vision / VLA ops ---

    pub fn gelu(&mut self, x: NodeId) -> NodeId {
        let ty = self.node(x).ty.clone();
        self.add_node(Op::Gelu, vec![x], ty)
    }

    #[track_caller]
    pub fn layer_norm(&mut self, x: NodeId, weight: NodeId, bias: NodeId, eps: f32) -> NodeId {
        let x_shape = &self.node(x).ty.shape;
        let w_shape = &self.node(weight).ty.shape;
        let b_shape = &self.node(bias).ty.shape;
        assert_eq!(x_shape.len(), 2, "layer_norm requires 2D input");
        assert_eq!(w_shape.len(), 1, "layer_norm weight must be 1D");
        assert_eq!(b_shape.len(), 1, "layer_norm bias must be 1D");
        assert_eq!(
            x_shape[1], w_shape[0],
            "layer_norm weight size must match last dim"
        );
        assert_eq!(
            x_shape[1], b_shape[0],
            "layer_norm bias size must match last dim"
        );
        let ty = self.node(x).ty.clone();
        self.add_node(Op::LayerNorm { eps }, vec![x, weight, bias], ty)
    }

    #[track_caller]
    pub fn full_attention(
        &mut self,
        q: NodeId,
        k: NodeId,
        v: NodeId,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
    ) -> NodeId {
        let q_shape = &self.node(q).ty.shape;
        let k_shape = &self.node(k).ty.shape;
        let v_shape = &self.node(v).ty.shape;
        assert_eq!(q_shape.len(), 2, "q must be 2D");
        assert_eq!(k_shape.len(), 2, "k must be 2D");
        assert_eq!(v_shape.len(), 2, "v must be 2D");
        let seq = q_shape[0];
        assert_eq!(
            q_shape[1],
            (num_heads * head_dim) as usize,
            "q dim mismatch"
        );
        assert_eq!(k_shape[0], seq, "k seq must match q seq");
        assert_eq!(
            k_shape[1],
            (num_kv_heads * head_dim) as usize,
            "k dim mismatch"
        );
        assert_eq!(v_shape[0], seq, "v seq must match q seq");
        assert_eq!(
            v_shape[1],
            (num_kv_heads * head_dim) as usize,
            "v dim mismatch"
        );
        let ty = TensorType::f32(vec![seq, (num_heads * head_dim) as usize]);
        self.add_node(
            Op::FullAttention {
                num_heads,
                num_kv_heads,
                head_dim,
            },
            vec![q, k, v],
            ty,
        )
    }

    #[track_caller]
    pub fn cross_attention(
        &mut self,
        q: NodeId,
        k: NodeId,
        v: NodeId,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
    ) -> NodeId {
        let q_shape = &self.node(q).ty.shape;
        let k_shape = &self.node(k).ty.shape;
        let v_shape = &self.node(v).ty.shape;
        assert_eq!(q_shape.len(), 2, "q must be 2D");
        assert_eq!(k_shape.len(), 2, "k must be 2D");
        assert_eq!(v_shape.len(), 2, "v must be 2D");
        let q_seq = q_shape[0];
        let kv_seq = k_shape[0];
        assert_eq!(
            q_shape[1],
            (num_heads * head_dim) as usize,
            "q dim mismatch"
        );
        assert_eq!(
            k_shape[1],
            (num_kv_heads * head_dim) as usize,
            "k dim mismatch"
        );
        assert_eq!(v_shape[0], kv_seq, "v seq must match k seq");
        assert_eq!(
            v_shape[1],
            (num_kv_heads * head_dim) as usize,
            "v dim mismatch"
        );
        let ty = TensorType::f32(vec![q_seq, (num_heads * head_dim) as usize]);
        self.add_node(
            Op::CrossAttention {
                num_heads,
                num_kv_heads,
                head_dim,
            },
            vec![q, k, v],
            ty,
        )
    }

    /// Differentiable multi-head attention with LSE output for backward.
    /// Handles both self-attention (q_seq == kv_seq, is_cross=false) and
    /// cross-attention (q_seq != kv_seq, is_cross=true).
    #[track_caller]
    pub fn multi_head_attn(
        &mut self,
        q: NodeId,
        k: NodeId,
        v: NodeId,
        num_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        is_cross: bool,
    ) -> NodeId {
        let q_shape = &self.node(q).ty.shape;
        let k_shape = &self.node(k).ty.shape;
        let v_shape = &self.node(v).ty.shape;
        assert_eq!(q_shape.len(), 2, "q must be 2D");
        assert_eq!(k_shape.len(), 2, "k must be 2D");
        assert_eq!(v_shape.len(), 2, "v must be 2D");
        let q_seq = q_shape[0];
        assert_eq!(
            q_shape[1],
            (num_heads * head_dim) as usize,
            "q dim mismatch"
        );
        assert_eq!(
            k_shape[1],
            (num_kv_heads * head_dim) as usize,
            "k dim mismatch"
        );
        assert_eq!(v_shape[0], k_shape[0], "v seq must match k seq");
        assert_eq!(
            v_shape[1],
            (num_kv_heads * head_dim) as usize,
            "v dim mismatch"
        );
        let ty = TensorType::f32(vec![q_seq, (num_heads * head_dim) as usize]);
        self.add_node(
            Op::MultiHeadAttn {
                num_heads,
                num_kv_heads,
                head_dim,
                is_cross,
            },
            vec![q, k, v],
            ty,
        )
    }

    // --- Loss ---

    #[track_caller]
    pub fn cross_entropy_loss(&mut self, logits: NodeId, labels: NodeId) -> NodeId {
        let l_shape = &self.node(logits).ty.shape;
        let t_shape = &self.node(labels).ty.shape;
        assert_eq!(l_shape, t_shape, "logits and labels must match");
        // Scalar output. The shader writes per-batch partial losses into an
        // oversized buffer; read_loss() sums them on the CPU side.
        let ty = TensorType::f32(vec![1]);
        self.add_node(Op::CrossEntropyLoss, vec![logits, labels], ty)
    }

    /// Binary cross-entropy loss: `-mean(t*log(p) + (1-t)*log(1-p))`.
    ///
    /// `pred` should be in (0, 1) (e.g. after sigmoid).
    /// Both `pred` and `labels` must have the same shape; output is scalar `[1]`.
    #[track_caller]
    pub fn bce_loss(&mut self, pred: NodeId, labels: NodeId) -> NodeId {
        let p_shape = &self.node(pred).ty.shape;
        let l_shape = &self.node(labels).ty.shape;
        assert_eq!(p_shape, l_shape, "pred and labels must match");
        // Scalar output. The shader writes per-workgroup partial losses
        // into an oversized buffer; read_loss() sums them on the CPU side.
        let ty = TensorType::f32(vec![1]);
        self.add_node(Op::BceLoss, vec![pred, labels], ty)
    }
}

impl fmt::Display for Graph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for node in &self.nodes {
            match node.name {
                Some(ref name) => write!(f, "%{} \"{}\" = {:?}(", node.id, name, node.op)?,
                None => write!(f, "%{} = {:?}(", node.id, node.op)?,
            }
            for (i, input) in node.inputs.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "%{}", input)?;
            }
            writeln!(f, ") : {}", node.ty)?;
        }
        write!(f, "outputs: ")?;
        for (i, out) in self.outputs.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            write!(f, "%{}", out)?;
        }
        writeln!(f)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_simple_graph() {
        let mut g = Graph::new();
        let x = g.input("x", &[4, 784]);
        let w = g.parameter("w", &[784, 128]);
        let y = g.matmul(x, w);
        let h = g.relu(y);
        g.set_outputs(vec![h]);

        assert_eq!(g.nodes().len(), 4);
        assert_eq!(g.node(y).ty.shape, vec![4, 128]);
        assert_eq!(g.node(h).ty.shape, vec![4, 128]);
    }

    #[test]
    fn tensor_type_bytes() {
        let t = TensorType::f32(vec![32, 784]);
        assert_eq!(t.num_elements(), 32 * 784);
        assert_eq!(t.size_bytes(), 32 * 784 * 4);
    }

    #[test]
    fn tensor_type_rank() {
        assert_eq!(TensorType::f32(vec![4, 3]).rank(), 2);
        assert_eq!(TensorType::f32(vec![1]).rank(), 1);
        assert_eq!(TensorType::f32(vec![2, 3, 4]).rank(), 3);
    }

    #[test]
    fn build_all_unary_ops() {
        let mut g = Graph::new();
        let x = g.input("x", &[4, 8]);
        let r = g.relu(x);
        let s = g.sigmoid(x);
        let n = g.neg(x);
        let e = g.exp(x);
        let t = g.transpose(x);
        g.set_outputs(vec![r, s, n, e, t]);

        assert_eq!(g.node(r).ty.shape, vec![4, 8]);
        assert_eq!(g.node(s).ty.shape, vec![4, 8]);
        assert_eq!(g.node(n).ty.shape, vec![4, 8]);
        assert_eq!(g.node(e).ty.shape, vec![4, 8]);
        assert_eq!(g.node(t).ty.shape, vec![8, 4]); // transposed
    }

    #[test]
    fn build_all_binary_ops() {
        let mut g = Graph::new();
        let a = g.input("a", &[4, 8]);
        let b = g.input("b", &[4, 8]);
        let add = g.add(a, b);
        let mul = g.mul(a, b);
        let gt = g.greater(a, b);
        g.set_outputs(vec![add, mul, gt]);

        for &id in &[add, mul, gt] {
            assert_eq!(g.node(id).ty.shape, vec![4, 8]);
        }
    }

    #[test]
    fn build_bias_add() {
        let mut g = Graph::new();
        let x = g.input("x", &[4, 128]);
        let b = g.parameter("b", &[128]);
        let out = g.bias_add(x, b);
        assert_eq!(g.node(out).ty.shape, vec![4, 128]);
    }

    #[test]
    fn build_reductions() {
        let mut g = Graph::new();
        let x = g.input("x", &[4, 8]);
        let sa = g.sum_all(x);
        let ma = g.mean_all(x);
        let sm = g.softmax(x);
        let lsm = g.log_softmax(x);
        g.set_outputs(vec![sa, ma, sm, lsm]);

        assert_eq!(g.node(sa).ty.shape, vec![1]);
        assert_eq!(g.node(ma).ty.shape, vec![1]);
        assert_eq!(g.node(sm).ty.shape, vec![4, 8]);
        assert_eq!(g.node(lsm).ty.shape, vec![4, 8]);
    }

    #[test]
    fn build_cross_entropy_loss() {
        let mut g = Graph::new();
        let logits = g.input("logits", &[4, 10]);
        let labels = g.input("labels", &[4, 10]);
        let loss = g.cross_entropy_loss(logits, labels);
        assert_eq!(g.node(loss).ty.shape, vec![1]);
    }

    #[test]
    fn build_constant_and_scalar() {
        let mut g = Graph::new();
        let c = g.constant(vec![1.0, 2.0, 3.0, 4.0], &[2, 2]);
        let s = g.scalar(42.0);
        assert_eq!(g.node(c).ty.shape, vec![2, 2]);
        assert_eq!(g.node(s).ty.shape, vec![1]);
        if let Op::Constant { ref data } = g.node(s).op {
            assert_eq!(data, &[42.0]);
        } else {
            panic!("expected Constant");
        }
    }

    #[test]
    fn graph_display() {
        let mut g = Graph::new();
        let x = g.input("x", &[2, 3]);
        let w = g.parameter("w", &[3, 4]);
        let y = g.matmul(x, w);
        g.set_outputs(vec![y]);
        let display = format!("{}", g);
        assert!(display.contains("%0"));
        assert!(display.contains("%2"));
        assert!(display.contains("outputs: %2"));
    }

    #[test]
    fn add_raw_node() {
        let mut g = Graph::new();
        let id = g.add_raw_node(
            Op::Input {
                name: "raw".to_string(),
            },
            vec![],
            TensorType::f32(vec![2, 3]),
        );
        assert_eq!(id, 0);
        assert_eq!(g.nodes().len(), 1);
    }

    #[test]
    fn toposort_remaps_attention_forward_node() {
        let mut g = Graph::new();
        let q = g.input("q", &[2, 4]);
        g.add_raw_node(Op::Nop, Vec::new(), TensorType::f32(Vec::new()));
        let k = g.input("k", &[2, 4]);
        let v = g.input("v", &[2, 4]);
        let grad = g.input("grad", &[2, 4]);
        let forward = g.add_raw_node(
            Op::MultiHeadAttn {
                num_heads: 1,
                num_kv_heads: 1,
                head_dim: 4,
                is_cross: false,
            },
            vec![q, k, v],
            TensorType::f32(vec![2, 4]),
        );
        let backward = g.add_raw_node(
            Op::MultiHeadAttnGradQ {
                fwd_node: forward,
                num_heads: 1,
                num_kv_heads: 1,
                head_dim: 4,
                is_cross: false,
            },
            vec![grad, q, k, v],
            TensorType::f32(vec![2, 4]),
        );
        g.set_outputs(vec![backward]);

        let sorted = g.toposort();
        let forward = sorted
            .nodes()
            .iter()
            .find(|node| matches!(node.op, Op::MultiHeadAttn { .. }))
            .expect("forward attention node not found");
        let backward = sorted.node(sorted.outputs()[0]);
        let Op::MultiHeadAttnGradQ { ref fwd_node, .. } = backward.op else {
            panic!("expected attention gradient output");
        };
        assert_eq!(*fwd_node, forward.id);
    }

    #[test]
    #[should_panic(expected = "matmul inner dimensions must match")]
    fn matmul_shape_mismatch() {
        let mut g = Graph::new();
        let a = g.input("a", &[4, 3]);
        let b = g.input("b", &[5, 2]); // 5 != 3
        g.matmul(a, b);
    }

    #[test]
    #[should_panic(expected = "add requires matching shapes")]
    fn add_shape_mismatch() {
        let mut g = Graph::new();
        let a = g.input("a", &[4, 3]);
        let b = g.input("b", &[4, 5]);
        g.add(a, b);
    }

    #[test]
    #[should_panic(expected = "transpose requires 2D tensor")]
    fn transpose_non_2d() {
        let mut g = Graph::new();
        let x = g.add_raw_node(
            Op::Input {
                name: "x".to_string(),
            },
            vec![],
            TensorType::f32(vec![2, 3, 4]),
        );
        g.transpose(x);
    }

    #[test]
    #[should_panic(expected = "bias must be 1D")]
    fn bias_add_wrong_bias_rank() {
        let mut g = Graph::new();
        let x = g.input("x", &[4, 8]);
        let b = g.input("b", &[4, 8]); // 2D, not 1D
        g.bias_add(x, b);
    }
}
