use std::fmt;

pub type NodeId = u32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
    U32,
}

impl DType {
    pub fn size_bytes(self) -> usize {
        match self {
            DType::F32 => 4,
            DType::U32 => 4,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
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

    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn size_bytes(&self) -> usize {
        self.num_elements() * self.dtype.size_bytes()
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

#[derive(Clone, Debug)]
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

    // Reduction
    SumAll,
    MeanAll,
    /// Column-wise sum: [M, N] → [N]  (sum over rows)
    SumRows,
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
    /// inputs: [grad_output]
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

    // Fused RmsNorm + MatMul: C = RmsNorm(X, W_norm) × W_proj
    // inputs: [x, w_norm, w_proj], output: [M, N] where x=[M,K], w_proj=[K,N]
    FusedRmsNormMatMul {
        eps: f32,
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

    /// 2D max pooling: input[N,C,H,W] → output[N,C,oH,oW]
    MaxPool2d {
        channels: u32,
        in_h: u32,
        in_w: u32,
        kernel_h: u32,
        kernel_w: u32,
        stride: u32,
        padding: u32,
    },

    /// Global average pooling: input[N*C*H*W] → output[N*C]
    /// Averages over the spatial dimensions (H,W) for each channel.
    GlobalAvgPool {
        channels: u32,
        spatial: u32, // H * W
    },
    /// Backward of GlobalAvgPool: broadcast grad_output[batch*channels] → [batch*channels*spatial]
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

#[derive(Clone, Debug)]
pub struct Node {
    pub id: NodeId,
    pub op: Op,
    pub inputs: Vec<NodeId>,
    pub ty: TensorType,
}

/// A derived parameter is created by the optimizer when fusing ops
/// that require concatenating multiple weights (e.g. gate+up projections).
#[derive(Clone, Debug)]
pub struct DerivedParam {
    /// Name of the new parameter (e.g. "gate_proj.weight+up_proj.weight")
    pub name: String,
    /// Source parameters to concatenate horizontally: (name, cols)
    pub sources: Vec<(String, usize)>,
    /// Total rows (shared across all sources)
    pub rows: usize,
}

pub struct Graph {
    nodes: Vec<Node>,
    outputs: Vec<NodeId>,
    /// Parameters created by the optimizer from concatenating original params.
    pub derived_params: Vec<DerivedParam>,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            outputs: Vec::new(),
            derived_params: Vec::new(),
        }
    }

    /// Rebuild the graph with nodes in topological order, removing Nop nodes.
    /// Returns a new graph with consecutive IDs where every node's inputs
    /// have lower IDs than the node itself.
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
            let new_inputs: Vec<NodeId> = node
                .inputs
                .iter()
                .filter_map(|&inp| old_to_new[inp as usize])
                .collect();
            new_graph.add_raw_node(node.op.clone(), new_inputs, node.ty.clone());
        }

        // Remap outputs
        let new_outputs: Vec<NodeId> = self
            .outputs
            .iter()
            .filter_map(|&out| old_to_new[out as usize])
            .collect();
        new_graph.set_outputs(new_outputs);
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

    pub fn set_outputs(&mut self, outputs: Vec<NodeId>) {
        self.outputs = outputs;
    }

    pub fn add_raw_node(&mut self, op: Op, inputs: Vec<NodeId>, ty: TensorType) -> NodeId {
        let id = self.nodes.len() as NodeId;
        self.nodes.push(Node { id, op, inputs, ty });
        id
    }

    pub fn nodes_mut(&mut self) -> &mut Vec<Node> {
        &mut self.nodes
    }

    fn add_node(&mut self, op: Op, inputs: Vec<NodeId>, ty: TensorType) -> NodeId {
        let id = self.nodes.len() as NodeId;
        self.nodes.push(Node { id, op, inputs, ty });
        id
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

    pub fn constant(&mut self, data: Vec<f32>, shape: &[usize]) -> NodeId {
        assert_eq!(data.len(), shape.iter().product::<usize>());
        let ty = TensorType::f32(shape.to_vec());
        self.add_node(Op::Constant { data }, vec![], ty)
    }

    pub fn scalar(&mut self, value: f32) -> NodeId {
        self.constant(vec![value], &[1])
    }

    // --- Binary ops ---

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
    pub fn matmul_bt(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let a_shape = &self.node(a).ty.shape;
        let b_shape = &self.node(b).ty.shape;
        assert_eq!(a_shape.len(), 2);
        assert_eq!(b_shape.len(), 2);
        assert_eq!(a_shape[1], b_shape[1], "MatMulBT: K dimensions must match");
        let ty = TensorType::f32(vec![a_shape[0], b_shape[0]]);
        self.add_node(Op::MatMulBT, vec![a, b], ty)
    }

    pub fn add(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let a_ty = &self.node(a).ty;
        let b_ty = &self.node(b).ty;
        assert_eq!(a_ty.shape, b_ty.shape, "add requires matching shapes");
        let ty = a_ty.clone();
        self.add_node(Op::Add, vec![a, b], ty)
    }

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

    pub fn mul(&mut self, a: NodeId, b: NodeId) -> NodeId {
        let a_ty = &self.node(a).ty;
        let b_ty = &self.node(b).ty;
        assert_eq!(a_ty.shape, b_ty.shape, "mul requires matching shapes");
        let ty = a_ty.clone();
        self.add_node(Op::Mul, vec![a, b], ty)
    }

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

    /// Reshape: reinterpret the tensor with a new shape (same element count).
    ///
    /// Implemented as `x + 0` with the target shape. The e-graph optimizer
    /// or a future pass could eliminate this, but it's cheap (one element-wise
    /// add of zeros).
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

    /// Scatter-add: accumulate `src[i]` rows into `output[indices[i]]`.
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

    pub fn cross_entropy_loss(&mut self, logits: NodeId, labels: NodeId) -> NodeId {
        let l_shape = &self.node(logits).ty.shape;
        let t_shape = &self.node(labels).ty.shape;
        assert_eq!(l_shape, t_shape, "logits and labels must match");
        let ty = TensorType::f32(vec![1]);
        self.add_node(Op::CrossEntropyLoss, vec![logits, labels], ty)
    }

    /// Binary cross-entropy loss: `-mean(t*log(p) + (1-t)*log(1-p))`.
    ///
    /// `pred` should be in (0, 1) (e.g. after sigmoid).
    /// Both `pred` and `labels` must have the same shape; output is scalar `[1]`.
    pub fn bce_loss(&mut self, pred: NodeId, labels: NodeId) -> NodeId {
        let p_shape = &self.node(pred).ty.shape;
        let l_shape = &self.node(labels).ty.shape;
        assert_eq!(p_shape, l_shape, "pred and labels must match");
        let ty = TensorType::f32(vec![1]);
        self.add_node(Op::BceLoss, vec![pred, labels], ty)
    }
}

impl fmt::Display for Graph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for node in &self.nodes {
            write!(f, "%{} = {:?}(", node.id, node.op)?;
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
        let t = g.transpose(x);
        g.set_outputs(vec![r, s, n, t]);

        assert_eq!(g.node(r).ty.shape, vec![4, 8]);
        assert_eq!(g.node(s).ty.shape, vec![4, 8]);
        assert_eq!(g.node(n).ty.shape, vec![4, 8]);
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
