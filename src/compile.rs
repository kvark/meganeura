use crate::graph::{Graph, Node, NodeId, Op};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Identifies which shader and entry point to use.
#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ShaderEntry {
    #[default]
    MatMul,
    MatMulAT,
    MatMulBT,
    FusedMatMulAdd,
    FusedMatMulATAdd,
    FusedMatMulBTAdd,
    Relu,
    Sigmoid,
    Tanh,
    Neg,
    Abs,
    Log,
    Recip,
    Add,
    Mul,
    Greater,
    BiasAdd,
    SgdUpdate,
    AdamUpdate,
    ScatterAdd,
    SumAll,
    MeanAll,
    Softmax,
    CrossEntropyLoss,
    BceLoss,
    Transpose,
    Silu,
    SwiGLU,
    RmsNorm,
    Embedding,
    RoPE,
    RoPEGrad,
    CausalAttention,
    CausalAttentionRoPE,
    SlidingWindowAttention,
    Gelu,
    LayerNorm,
    FullAttention,
    CrossAttention,
    MultiHeadAttn,
    MultiHeadAttnGradQ,
    MultiHeadAttnGradK,
    MultiHeadAttnGradV,
    SwiGLUGradGate,
    SwiGLUGradUp,
    SiluGrad,
    SwiGLUConcat,
    SwiGLUConcatGrad,
    SumRows,
    RmsNormGradW,
    RmsNormGradX,
    FusedRmsNormMatMul,
    /// Precompute rsqrt for RmsNorm (phase 1 of two-phase fusion)
    RmsNormRsqrt,
    GroupNorm,
    GroupNormSilu,
    GroupNormGradInput,
    GroupNormGradWeightBias,
    Concat,
    SplitA,
    SplitB,
    Upsample2x,
    Upsample2xGrad,
    Conv2d,
    Conv2dGemm,
    Conv2dGemmSmall,
    Conv2dGradInput,
    Conv2dGradInputGemm,
    Conv2dGradInputGemmSmall,
    Conv2dGradInputGemmCoop,
    Conv2dGradWeight,
    Conv2dGradWeightGemm,
    Conv2dGradWeightGemmSmall,
    CacheWrite,
    CachedAttention,
    RoPEDynamic,
    MaxPool2d,
    GlobalAvgPool,
    GlobalAvgPoolGrad,
}

impl ShaderEntry {
    pub fn shader_group(&self) -> crate::codegen::ShaderGroup {
        use crate::codegen::ShaderGroup;
        match *self {
            ShaderEntry::MatMul => ShaderGroup::MatMul,
            ShaderEntry::MatMulAT => ShaderGroup::MatMulAT,
            ShaderEntry::MatMulBT => ShaderGroup::MatMulBT,
            ShaderEntry::FusedMatMulAdd => ShaderGroup::MatMulAdd,
            ShaderEntry::FusedMatMulATAdd => ShaderGroup::MatMulATAdd,
            ShaderEntry::FusedMatMulBTAdd => ShaderGroup::MatMulBTAdd,
            ShaderEntry::Relu
            | ShaderEntry::Sigmoid
            | ShaderEntry::Tanh
            | ShaderEntry::Neg
            | ShaderEntry::Abs
            | ShaderEntry::Log
            | ShaderEntry::Recip => ShaderGroup::Unary,
            ShaderEntry::Add | ShaderEntry::Mul | ShaderEntry::Greater => ShaderGroup::Binary,
            ShaderEntry::BiasAdd => ShaderGroup::BiasAdd,
            ShaderEntry::SgdUpdate => ShaderGroup::Sgd,
            ShaderEntry::AdamUpdate => ShaderGroup::Adam,
            ShaderEntry::ScatterAdd => ShaderGroup::ScatterAdd,
            ShaderEntry::SumAll | ShaderEntry::MeanAll => ShaderGroup::Reduce,
            ShaderEntry::Softmax => ShaderGroup::Softmax,
            ShaderEntry::CrossEntropyLoss => ShaderGroup::CrossEntropy,
            ShaderEntry::BceLoss => ShaderGroup::BceLoss,
            ShaderEntry::Transpose => ShaderGroup::Transpose,
            ShaderEntry::Silu => ShaderGroup::Unary,
            ShaderEntry::SwiGLU => ShaderGroup::Binary,
            ShaderEntry::RmsNorm => ShaderGroup::RmsNorm,
            ShaderEntry::Embedding => ShaderGroup::Embedding,
            ShaderEntry::RoPE => ShaderGroup::RoPE,
            ShaderEntry::RoPEGrad => ShaderGroup::RoPEGrad,
            ShaderEntry::CausalAttention => ShaderGroup::CausalAttention,
            ShaderEntry::CausalAttentionRoPE => ShaderGroup::CausalAttentionRoPE,
            ShaderEntry::SlidingWindowAttention => ShaderGroup::SlidingWindowAttention,
            ShaderEntry::Gelu => ShaderGroup::Unary,
            ShaderEntry::LayerNorm => ShaderGroup::LayerNorm,
            ShaderEntry::FullAttention => ShaderGroup::FullAttention,
            ShaderEntry::CrossAttention => ShaderGroup::CrossAttention,
            ShaderEntry::MultiHeadAttn => ShaderGroup::MultiHeadAttn,
            ShaderEntry::MultiHeadAttnGradQ => ShaderGroup::MultiHeadAttnGradQ,
            ShaderEntry::MultiHeadAttnGradK => ShaderGroup::MultiHeadAttnGradK,
            ShaderEntry::MultiHeadAttnGradV => ShaderGroup::MultiHeadAttnGradV,
            ShaderEntry::SwiGLUGradGate | ShaderEntry::SwiGLUGradUp | ShaderEntry::SiluGrad => {
                ShaderGroup::SwiGLUGrad
            }
            ShaderEntry::SwiGLUConcat | ShaderEntry::SwiGLUConcatGrad => ShaderGroup::SwiGLUConcat,
            ShaderEntry::SumRows => ShaderGroup::SumRows,
            ShaderEntry::RmsNormGradW | ShaderEntry::RmsNormGradX => ShaderGroup::RmsNormGrad,
            ShaderEntry::FusedRmsNormMatMul => ShaderGroup::FusedRmsNormMatMul,
            ShaderEntry::RmsNormRsqrt => ShaderGroup::RmsNormRsqrt,
            ShaderEntry::GroupNorm => ShaderGroup::GroupNorm,
            ShaderEntry::GroupNormSilu => ShaderGroup::GroupNormSilu,
            ShaderEntry::GroupNormGradInput => ShaderGroup::GroupNormGrad,
            ShaderEntry::GroupNormGradWeightBias => ShaderGroup::GroupNormGrad,
            ShaderEntry::Concat => ShaderGroup::Concat,
            ShaderEntry::SplitA | ShaderEntry::SplitB => ShaderGroup::Split,
            ShaderEntry::Upsample2x => ShaderGroup::Upsample,
            ShaderEntry::Upsample2xGrad => ShaderGroup::UpsampleGrad,
            ShaderEntry::Conv2d => ShaderGroup::Conv2d,
            ShaderEntry::Conv2dGemm => ShaderGroup::Conv2dGemm,
            ShaderEntry::Conv2dGemmSmall => ShaderGroup::Conv2dGemmSmall,
            ShaderEntry::Conv2dGradInput => ShaderGroup::Conv2dGradInput,
            ShaderEntry::Conv2dGradInputGemm => ShaderGroup::Conv2dGradInputGemm,
            ShaderEntry::Conv2dGradInputGemmSmall => ShaderGroup::Conv2dGradInputGemmSmall,
            ShaderEntry::Conv2dGradInputGemmCoop => ShaderGroup::Conv2dGradInputGemmCoop,
            ShaderEntry::Conv2dGradWeight => ShaderGroup::Conv2dGradWeight,
            ShaderEntry::Conv2dGradWeightGemm => ShaderGroup::Conv2dGradWeightGemm,
            ShaderEntry::Conv2dGradWeightGemmSmall => ShaderGroup::Conv2dGradWeightGemmSmall,
            ShaderEntry::CacheWrite => ShaderGroup::CacheWrite,
            ShaderEntry::CachedAttention => ShaderGroup::CachedAttention,
            ShaderEntry::RoPEDynamic => ShaderGroup::RoPEDynamic,
            ShaderEntry::MaxPool2d => ShaderGroup::MaxPool2d,
            ShaderEntry::GlobalAvgPool => ShaderGroup::GlobalAvgPool,
            ShaderEntry::GlobalAvgPoolGrad => ShaderGroup::GlobalAvgPoolGrad,
        }
    }

    pub fn entry_point(&self) -> &'static str {
        match *self {
            ShaderEntry::MatMul
            | ShaderEntry::MatMulAT
            | ShaderEntry::MatMulBT
            | ShaderEntry::FusedMatMulAdd
            | ShaderEntry::FusedMatMulATAdd
            | ShaderEntry::FusedMatMulBTAdd
            | ShaderEntry::BiasAdd
            | ShaderEntry::SgdUpdate
            | ShaderEntry::AdamUpdate
            | ShaderEntry::ScatterAdd
            | ShaderEntry::Softmax
            | ShaderEntry::CrossEntropyLoss
            | ShaderEntry::BceLoss
            | ShaderEntry::Transpose => "main",
            ShaderEntry::Relu => "relu",
            ShaderEntry::Sigmoid => "sigmoid",
            ShaderEntry::Tanh => "tanh_",
            ShaderEntry::Neg => "neg",
            ShaderEntry::Abs => "abs_",
            ShaderEntry::Log => "log_",
            ShaderEntry::Recip => "recip",
            ShaderEntry::Add => "add",
            ShaderEntry::Mul => "mul",
            ShaderEntry::Greater => "greater",
            ShaderEntry::SumAll => "sum_all",
            ShaderEntry::MeanAll => "mean_all",
            ShaderEntry::Silu => "silu",
            ShaderEntry::SwiGLU => "swiglu",
            ShaderEntry::RmsNorm => "main",
            ShaderEntry::Embedding => "main",
            ShaderEntry::RoPE => "main",
            ShaderEntry::RoPEGrad => "main",
            ShaderEntry::CausalAttention => "main",
            ShaderEntry::CausalAttentionRoPE => "main",
            ShaderEntry::SlidingWindowAttention => "main",
            ShaderEntry::Gelu => "gelu",
            ShaderEntry::LayerNorm => "main",
            ShaderEntry::FullAttention => "main",
            ShaderEntry::CrossAttention => "main",
            ShaderEntry::MultiHeadAttn
            | ShaderEntry::MultiHeadAttnGradQ
            | ShaderEntry::MultiHeadAttnGradK
            | ShaderEntry::MultiHeadAttnGradV => "main",
            ShaderEntry::SwiGLUGradGate => "swiglu_grad_gate",
            ShaderEntry::SwiGLUGradUp => "swiglu_grad_up",
            ShaderEntry::SiluGrad => "silu_grad",
            ShaderEntry::SwiGLUConcat => "swiglu_concat",
            ShaderEntry::SwiGLUConcatGrad => "swiglu_concat_grad",
            ShaderEntry::SumRows => "sum_rows",
            ShaderEntry::RmsNormGradW => "rms_norm_grad_w",
            ShaderEntry::RmsNormGradX => "rms_norm_grad_x",
            ShaderEntry::FusedRmsNormMatMul => "main",
            ShaderEntry::RmsNormRsqrt => "main",
            ShaderEntry::GroupNorm | ShaderEntry::GroupNormSilu => "main",
            ShaderEntry::GroupNormGradInput => "grad_input",
            ShaderEntry::GroupNormGradWeightBias => "grad_weight_bias",
            ShaderEntry::Concat => "main",
            ShaderEntry::SplitA => "split_a",
            ShaderEntry::SplitB => "split_b",
            ShaderEntry::Upsample2x => "main",
            ShaderEntry::Upsample2xGrad => "main",
            ShaderEntry::Conv2d => "main",
            ShaderEntry::Conv2dGemm | ShaderEntry::Conv2dGemmSmall => "main",
            ShaderEntry::Conv2dGradInput => "main",
            ShaderEntry::Conv2dGradInputGemm
            | ShaderEntry::Conv2dGradInputGemmSmall
            | ShaderEntry::Conv2dGradInputGemmCoop => "main",
            ShaderEntry::Conv2dGradWeight
            | ShaderEntry::Conv2dGradWeightGemm
            | ShaderEntry::Conv2dGradWeightGemmSmall => "main",
            ShaderEntry::CacheWrite => "main",
            ShaderEntry::CachedAttention => "main",
            ShaderEntry::RoPEDynamic => "main",
            ShaderEntry::MaxPool2d => "max_pool_2d",
            ShaderEntry::GlobalAvgPool => "global_avg_pool",
            ShaderEntry::GlobalAvgPoolGrad => "main",
        }
    }
}

/// An elementwise operation fused into a matmul's store loop.
///
/// Each step transforms `val` (the matmul result for one element):
///   - Unary ops modify val in-place: `val = relu(val)`
///   - Binary ops read from an extra buffer: `val = val + extra[idx]`
///
/// The extra buffer index refers to `epilogue_buffers` on the Dispatch.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EpilogueOp {
    /// val = val + extra_buffer[row * N + col]
    Add(u8),
    /// val = val + extra_buffer[col]  (row-broadcast bias)
    BiasAdd(u8),
    /// val = max(val, 0)
    Relu,
    /// val = val * sigmoid(val)
    Silu,
    /// val = 1 / (1 + exp(-val))
    Sigmoid,
    /// val = -val
    Neg,
}

/// A single GPU dispatch in the execution plan.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
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
    /// When true, this dispatch uses the cooperative matrix pipeline
    /// (set at runtime based on per-dispatch eligibility).
    #[serde(default)]
    pub use_coop: bool,
    /// When true, use the 32×32 small-tile matmul pipeline instead of 64×64.
    #[serde(default)]
    pub use_small_tiles: bool,
    /// Fused elementwise epilogue chain applied in the matmul store loop.
    /// Empty = no epilogue (default). Non-empty = one or more ops fused
    /// into the matmul, each saving a dispatch + barrier.
    #[serde(default)]
    pub epilogue: Vec<EpilogueOp>,
    /// Extra buffer refs for epilogue binary ops (Add, BiasAdd).
    #[serde(default)]
    pub epilogue_buffers: Vec<BufferRef>,
    /// Human-readable label for profiling (e.g. "MatMul[50,720,960]").
    #[serde(default)]
    pub label: String,
}

/// Reference to a GPU buffer in the execution plan.
#[derive(Clone, Debug, Default, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BufferRef(pub u32);

/// The complete execution plan: a static sequence of dispatches.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ExecutionPlan {
    /// Buffer sizes in bytes, indexed by BufferRef.
    pub buffers: Vec<usize>,
    /// Which buffers hold parameters (need initialization).
    pub param_buffers: Vec<(String, BufferRef)>,
    /// Which buffers hold inputs (filled each step).
    pub input_buffers: Vec<(String, BufferRef)>,
    /// Constant buffers with their initial data (uploaded once at session creation).
    pub constant_buffers: Vec<(BufferRef, Vec<f32>)>,
    /// The dispatch sequence. For a training graph, this includes
    /// forward, backward, and parameter update dispatches.
    pub dispatches: Vec<Dispatch>,
    /// Index of the loss buffer (first graph output, for reading back).
    pub loss_buffer: Option<BufferRef>,
    /// All graph output buffers (for reading back multiple outputs).
    pub output_buffers: Vec<BufferRef>,
    /// Parameter buffer → gradient buffer mapping (for SGD).
    pub param_grad_pairs: Vec<(BufferRef, BufferRef)>,
    /// LSE buffers allocated for MultiHeadAttn forward nodes: (node_id, buffer).
    pub lse_buffers: Vec<(NodeId, BufferRef)>,
    /// Derived parameters: buffer = horizontal concat of source parameters.
    /// Created by the optimizer when fusing e.g. gate+up projections.
    /// Format: (derived_buf, [(source_name, num_elements), ...])
    pub derived_params: Vec<(BufferRef, Vec<(String, usize)>)>,
}

/// Compile a differentiated graph into an ExecutionPlan.
/// Topological sort of graph nodes (Kahn's algorithm).
/// Returns node IDs in dependency order: producers before consumers.
fn topological_order(graph: &Graph) -> Vec<NodeId> {
    let n = graph.nodes().len();
    let mut in_degree = vec![0u32; n];
    for node in graph.nodes() {
        in_degree[node.id as usize] = node.inputs.len() as u32;
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
        // For each node that depends on `id`, decrement in-degree
        for node in graph.nodes() {
            if node.inputs.contains(&id) {
                in_degree[node.id as usize] -= 1;
                if in_degree[node.id as usize] == 0 {
                    queue.push_back(node.id);
                }
            }
        }
    }

    // Any unvisited nodes (cycles or disconnected) — append in ID order
    if order.len() < n {
        for node in graph.nodes() {
            if !order.contains(&node.id) {
                order.push(node.id);
            }
        }
    }

    order
}

pub fn compile(graph: &Graph) -> ExecutionPlan {
    let mut compiler = Compiler::new(graph);
    compiler.compile();

    // Propagate derived parameter info from graph to plan
    for dp in &graph.derived_params {
        if let Some(&(_, buf_ref)) = compiler
            .plan
            .param_buffers
            .iter()
            .find(|entry| entry.0 == dp.name)
        {
            let sources: Vec<(String, usize)> = dp
                .sources
                .iter()
                .map(|entry| (entry.0.clone(), entry.1))
                .collect();
            compiler.plan.derived_params.push((buf_ref, sources));
        }
    }

    fuse_epilogues(&mut compiler.plan.dispatches);

    compiler.plan
}

/// Post-compile pass: absorb single-use elementwise dispatches into
/// the preceding matmul's epilogue. Each fused dispatch is replaced
/// with a no-op (empty shader + zero workgroups) and removed later.
///
/// Only fuses into scalar matmul dispatches (not coop) because the
/// coop store uses coopStoreT which doesn't support per-element epilogues.
fn fuse_epilogues(dispatches: &mut Vec<Dispatch>) {
    use std::collections::HashMap;
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

        let (epilogue_op, primary_buf, extra_buf) = match d.shader {
            ShaderEntry::Relu => (EpilogueOp::Relu, d.input_buffers[0], None),
            ShaderEntry::Sigmoid => (EpilogueOp::Sigmoid, d.input_buffers[0], None),
            ShaderEntry::Neg => (EpilogueOp::Neg, d.input_buffers[0], None),
            ShaderEntry::Silu => (EpilogueOp::Silu, d.input_buffers[0], None),
            _ => continue,
        };

        // The elementwise op reads from primary_buf. Find the matmul that produced it.
        let Some(&prod_idx) = producer.get(&primary_buf) else {
            continue;
        };
        let prod = &dispatches[prod_idx];

        // Only fuse into matmul dispatches (not coop — coop uses coopStoreT)
        let is_matmul = matches!(
            prod.shader,
            ShaderEntry::MatMul
                | ShaderEntry::MatMulAT
                | ShaderEntry::MatMulBT
                | ShaderEntry::FusedMatMulAdd
                | ShaderEntry::FusedMatMulATAdd
                | ShaderEntry::FusedMatMulBTAdd
        );
        if !is_matmul || prod.use_coop {
            continue;
        }

        // The matmul output must be consumed by ONLY this elementwise op
        // (otherwise we can't modify the output in-place).
        if read_count.get(&primary_buf).copied().unwrap_or(0) != 1 {
            continue;
        }

        // The elementwise op must write to a different buffer (or same).
        // We redirect the matmul to write directly to the elementwise output.
        let elem_output = dispatches[i].output_buffer;

        // Apply the fusion
        let epi_buf_idx = dispatches[prod_idx].epilogue_buffers.len() as u8;
        let op = match epilogue_op {
            EpilogueOp::BiasAdd(_) => EpilogueOp::BiasAdd(epi_buf_idx),
            EpilogueOp::Add(_) => EpilogueOp::Add(epi_buf_idx),
            other => other,
        };
        dispatches[prod_idx].epilogue.push(op);
        if let Some(buf) = extra_buf {
            dispatches[prod_idx].epilogue_buffers.push(buf);
        }
        // Redirect matmul output to the elementwise op's output buffer
        dispatches[prod_idx].output_buffer = elem_output;
        // Update producer map
        producer.insert(elem_output, prod_idx);

        to_remove.push(i);
    }

    // Remove fused dispatches (iterate in reverse to preserve indices)
    for &idx in to_remove.iter().rev() {
        dispatches.remove(idx);
    }
}

struct Compiler<'a> {
    graph: &'a Graph,
    plan: ExecutionPlan,
    /// Map from NodeId → BufferRef for each node's output.
    node_buffers: HashMap<NodeId, BufferRef>,
}

impl<'a> Compiler<'a> {
    fn new(graph: &'a Graph) -> Self {
        Self {
            graph,
            plan: ExecutionPlan {
                buffers: Vec::new(),
                param_buffers: Vec::new(),
                input_buffers: Vec::new(),
                constant_buffers: Vec::new(),
                dispatches: Vec::new(),
                loss_buffer: None,
                output_buffers: Vec::new(),
                param_grad_pairs: Vec::new(),
                lse_buffers: Vec::new(),
                derived_params: Vec::new(),
            },
            node_buffers: HashMap::new(),
        }
    }

    fn alloc_buffer(&mut self, size_bytes: usize) -> BufferRef {
        let idx = self.plan.buffers.len() as u32;
        self.plan.buffers.push(size_bytes);
        BufferRef(idx)
    }

    fn get_buffer(&self, node: NodeId) -> BufferRef {
        self.node_buffers[&node]
    }

    fn compile(&mut self) {
        // First pass: allocate buffers for all nodes
        for node in self.graph.nodes() {
            let size = node.ty.size_bytes();
            let buf = self.alloc_buffer(size);
            self.node_buffers.insert(node.id, buf);

            match node.op {
                Op::Parameter { ref name } => {
                    self.plan.param_buffers.push((name.clone(), buf));
                }
                Op::Input { ref name } => {
                    self.plan.input_buffers.push((name.clone(), buf));
                }
                Op::Constant { ref data } => {
                    self.plan.constant_buffers.push((buf, data.clone()));
                }
                Op::MultiHeadAttn { num_heads, .. }
                | Op::CausalAttention { num_heads, .. }
                | Op::CausalAttentionRoPE { num_heads, .. }
                | Op::SlidingWindowAttention { num_heads, .. }
                | Op::FullAttention { num_heads, .. }
                | Op::CrossAttention { num_heads, .. } => {
                    let q_seq = node.ty.shape[0];
                    // LSE buffer: [0..q_seq*num_heads*2): LSE data (max_score, log_sum_exp per pos×head)
                    let lse_part = q_seq * num_heads as usize * 2;
                    let lse_buf = self.alloc_buffer(lse_part * 4);
                    self.plan.lse_buffers.push((node.id, lse_buf));
                }
                _ => {}
            }
        }

        // Second pass: emit dispatches in topological order.
        // The optimizer may create new nodes at high IDs that are referenced
        // by existing nodes at lower IDs (e.g. SwiGLU concat fusion creates
        // a new MatMul at the end, referenced by the original SwiGLU node).
        // Processing in ID order would dispatch consumers before producers.
        let topo = topological_order(self.graph);
        for &node_id in &topo {
            self.compile_node(&self.graph.nodes()[node_id as usize]);
        }

        // Generate labels for profiling
        for d in &mut self.plan.dispatches {
            d.label = match d.shader {
                ShaderEntry::MatMul | ShaderEntry::FusedMatMulAdd => {
                    format!(
                        "{:?}[{}x{}x{}]",
                        d.shader, d.params[0], d.params[2], d.params[1]
                    )
                }
                ShaderEntry::MatMulAT
                | ShaderEntry::MatMulBT
                | ShaderEntry::FusedMatMulATAdd
                | ShaderEntry::FusedMatMulBTAdd => {
                    format!(
                        "{:?}[{}x{}x{}]",
                        d.shader, d.params[0], d.params[1], d.params[2]
                    )
                }
                ShaderEntry::MultiHeadAttn
                | ShaderEntry::MultiHeadAttnGradQ
                | ShaderEntry::MultiHeadAttnGradK
                | ShaderEntry::MultiHeadAttnGradV => {
                    let nh = d.params[2] >> 16;
                    let nkv = d.params[2] & 0xFFFF;
                    format!(
                        "{:?}[q={},kv={},h={}/{}]",
                        d.shader, d.params[0], d.params[1], nh, nkv
                    )
                }
                ShaderEntry::RmsNorm | ShaderEntry::RmsNormGradW | ShaderEntry::RmsNormGradX => {
                    format!("{:?}[{}x{}]", d.shader, d.params[0], d.params[1])
                }
                ShaderEntry::FusedRmsNormMatMul => {
                    format!(
                        "{:?}[{}x{}x{}]",
                        d.shader, d.params[0], d.params[2], d.params[1]
                    )
                }
                _ => {
                    if d.params[0] > 0 {
                        format!("{:?}[{}]", d.shader, d.params[0])
                    } else {
                        format!("{:?}", d.shader)
                    }
                }
            };
        }

        // Set loss buffer (first output) and collect all output buffers
        for &out_id in self.graph.outputs() {
            self.plan.output_buffers.push(self.get_buffer(out_id));
        }
        if let Some(&loss_id) = self.graph.outputs().first() {
            self.plan.loss_buffer = Some(self.get_buffer(loss_id));
        }

        // Build param→grad pairs from outputs
        // Outputs are [loss, grad_param1, grad_param2, ...]
        let outputs = self.graph.outputs();
        if outputs.len() > 1 {
            let param_names: Vec<String> = self
                .plan
                .param_buffers
                .iter()
                .map(|entry| entry.0.clone())
                .collect();
            for (i, _name) in param_names.iter().enumerate() {
                if i + 1 < outputs.len() {
                    let param_buf = self.plan.param_buffers[i].1;
                    let grad_buf = self.get_buffer(outputs[i + 1]);
                    self.plan.param_grad_pairs.push((param_buf, grad_buf));
                }
            }
        }
    }

    fn compile_node(&mut self, node: &Node) {
        let out_buf = self.get_buffer(node.id);

        match node.op {
            // Leaf nodes and dead nodes: no dispatch needed
            Op::Input { .. } | Op::Parameter { .. } | Op::Constant { .. } | Op::Nop => {}

            Op::MatMul => {
                let a = self.get_buffer(node.inputs[0]);
                let b = self.get_buffer(node.inputs[1]);
                let a_shape = &self.graph.node(node.inputs[0]).ty.shape;
                let b_shape = &self.graph.node(node.inputs[1]).ty.shape;
                let m = a_shape[0] as u32;
                let k = a_shape[1] as u32;
                let n = b_shape[1] as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::MatMul,
                    workgroups: [n.div_ceil(64), m.div_ceil(64), 1],
                    input_buffers: vec![a, b],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![m, k, n, 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::MatMulAT => {
                // C = A^T @ B  (A is [K, M], B is [K, N], C is [M, N])
                let a = self.get_buffer(node.inputs[0]);
                let b = self.get_buffer(node.inputs[1]);
                let a_shape = &self.graph.node(node.inputs[0]).ty.shape;
                let b_shape = &self.graph.node(node.inputs[1]).ty.shape;
                let k = a_shape[0] as u32; // A is [K, M]
                let m = a_shape[1] as u32;
                let n = b_shape[1] as u32; // B is [K, N]
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::MatMulAT,
                    workgroups: [n.div_ceil(64), m.div_ceil(64), 1],
                    input_buffers: vec![a, b],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![m, n, k, 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::MatMulBT => {
                // C = A @ B^T  (A is [M, K], B is [N, K], C is [M, N])
                let a = self.get_buffer(node.inputs[0]);
                let b = self.get_buffer(node.inputs[1]);
                let a_shape = &self.graph.node(node.inputs[0]).ty.shape;
                let b_shape = &self.graph.node(node.inputs[1]).ty.shape;
                let m = a_shape[0] as u32; // A is [M, K]
                let k = a_shape[1] as u32;
                let n = b_shape[0] as u32; // B is [N, K]
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::MatMulBT,
                    workgroups: [n.div_ceil(64), m.div_ceil(64), 1],
                    input_buffers: vec![a, b],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![m, n, k, 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::FusedMatMulAdd => {
                // C = A × B + D (inputs: [a, b, d])
                let a = self.get_buffer(node.inputs[0]);
                let b = self.get_buffer(node.inputs[1]);
                let d = self.get_buffer(node.inputs[2]);
                let a_shape = &self.graph.node(node.inputs[0]).ty.shape;
                let b_shape = &self.graph.node(node.inputs[1]).ty.shape;
                let m = a_shape[0] as u32;
                let k = a_shape[1] as u32;
                let n = b_shape[1] as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::FusedMatMulAdd,
                    workgroups: [n.div_ceil(64), m.div_ceil(64), 1],
                    input_buffers: vec![a, b, d],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![m, k, n, 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::FusedMatMulATAdd => {
                // C = A^T × B + D (inputs: [a, b, d])
                let a = self.get_buffer(node.inputs[0]);
                let b = self.get_buffer(node.inputs[1]);
                let d = self.get_buffer(node.inputs[2]);
                let a_shape = &self.graph.node(node.inputs[0]).ty.shape;
                let b_shape = &self.graph.node(node.inputs[1]).ty.shape;
                let k = a_shape[0] as u32; // A is [K, M]
                let m = a_shape[1] as u32;
                let n = b_shape[1] as u32; // B is [K, N]
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::FusedMatMulATAdd,
                    workgroups: [n.div_ceil(64), m.div_ceil(64), 1],
                    input_buffers: vec![a, b, d],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![m, n, k, 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::FusedMatMulBTAdd => {
                // C = A × B^T + D (inputs: [a, b, d])
                let a = self.get_buffer(node.inputs[0]);
                let b = self.get_buffer(node.inputs[1]);
                let d = self.get_buffer(node.inputs[2]);
                let a_shape = &self.graph.node(node.inputs[0]).ty.shape;
                let b_shape = &self.graph.node(node.inputs[1]).ty.shape;
                let m = a_shape[0] as u32; // A is [M, K]
                let k = a_shape[1] as u32;
                let n = b_shape[0] as u32; // B is [N, K]
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::FusedMatMulBTAdd,
                    workgroups: [n.div_ceil(64), m.div_ceil(64), 1],
                    input_buffers: vec![a, b, d],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![m, n, k, 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::Add => {
                self.emit_binary(ShaderEntry::Add, node, out_buf);
            }
            Op::Mul => {
                self.emit_binary(ShaderEntry::Mul, node, out_buf);
            }
            Op::Greater => {
                self.emit_binary(ShaderEntry::Greater, node, out_buf);
            }

            Op::BiasAdd => {
                let a = self.get_buffer(node.inputs[0]);
                let b = self.get_buffer(node.inputs[1]);
                let len = node.ty.num_elements() as u32;
                let bias_len = self.graph.node(node.inputs[1]).ty.num_elements() as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::BiasAdd,
                    workgroups: [len.div_ceil(256), 1, 1],
                    input_buffers: vec![a, b],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![len, bias_len, 0, 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::Relu => {
                self.emit_unary(ShaderEntry::Relu, node, out_buf);
            }
            Op::Sigmoid => {
                self.emit_unary(ShaderEntry::Sigmoid, node, out_buf);
            }
            Op::Tanh => {
                self.emit_unary(ShaderEntry::Tanh, node, out_buf);
            }
            Op::Neg => {
                self.emit_unary(ShaderEntry::Neg, node, out_buf);
            }
            Op::Abs => {
                self.emit_unary(ShaderEntry::Abs, node, out_buf);
            }
            Op::Log => {
                self.emit_unary(ShaderEntry::Log, node, out_buf);
            }
            Op::Recip => {
                self.emit_unary(ShaderEntry::Recip, node, out_buf);
            }

            Op::SumAll => {
                let input = self.get_buffer(node.inputs[0]);
                let len = self.graph.node(node.inputs[0]).ty.num_elements() as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::SumAll,
                    workgroups: [1, 1, 1],
                    input_buffers: vec![input],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![len, 0, 0, 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::MeanAll => {
                let input = self.get_buffer(node.inputs[0]);
                let len = self.graph.node(node.inputs[0]).ty.num_elements() as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::MeanAll,
                    workgroups: [1, 1, 1],
                    input_buffers: vec![input],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![len, 0, 0, 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::SumRows => {
                // [M, N] → [N]: one thread per column, loops over M rows
                let input = self.get_buffer(node.inputs[0]);
                let in_shape = &self.graph.node(node.inputs[0]).ty.shape;
                let m = in_shape[0] as u32;
                let n = in_shape[1] as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::SumRows,
                    workgroups: [n.div_ceil(256), 1, 1],
                    input_buffers: vec![input],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![m, n, 0, 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::Softmax => {
                let input = self.get_buffer(node.inputs[0]);
                let shape = &self.graph.node(node.inputs[0]).ty.shape;
                let batch = shape[0] as u32;
                let features = shape[1] as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::Softmax,
                    workgroups: [batch.div_ceil(256), 1, 1],
                    input_buffers: vec![input],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![batch, features, 0, 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::LogSoftmax => {
                // Same as softmax for now, log applied in place
                let input = self.get_buffer(node.inputs[0]);
                let shape = &self.graph.node(node.inputs[0]).ty.shape;
                let batch = shape[0] as u32;
                let features = shape[1] as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::Softmax,
                    workgroups: [batch.div_ceil(256), 1, 1],
                    input_buffers: vec![input],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![batch, features, 0, 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::CrossEntropyLoss => {
                let logits = self.get_buffer(node.inputs[0]);
                let labels = self.get_buffer(node.inputs[1]);
                let shape = &self.graph.node(node.inputs[0]).ty.shape;
                let batch = shape[0] as u32;
                let features = shape[1] as u32;
                // The cross_entropy shader writes both grad_out (batch*features f32s)
                // and loss_out (scalar). Separate buffer for grad_out avoids OOB.
                let grad_buf = self.alloc_buffer(shape.iter().product::<usize>() * 4);
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::CrossEntropyLoss,
                    workgroups: [1, 1, 1],
                    input_buffers: vec![logits, labels],
                    output_buffer: grad_buf,
                    extra_outputs: vec![out_buf],
                    params: vec![batch, features, 0, 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::BceLoss => {
                let pred = self.get_buffer(node.inputs[0]);
                let labels = self.get_buffer(node.inputs[1]);
                let len = self.graph.node(node.inputs[0]).ty.num_elements() as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::BceLoss,
                    workgroups: [1, 1, 1],
                    input_buffers: vec![pred, labels],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![len, 0, 0, 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::Transpose => {
                let input = self.get_buffer(node.inputs[0]);
                let shape = &self.graph.node(node.inputs[0]).ty.shape;
                let m = shape[0] as u32;
                let n = shape[1] as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::Transpose,
                    workgroups: [n.div_ceil(16), m.div_ceil(16), 1],
                    input_buffers: vec![input],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![m, n, 0, 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::Silu => {
                self.emit_unary(ShaderEntry::Silu, node, out_buf);
            }

            Op::SwiGLU => {
                self.emit_binary(ShaderEntry::SwiGLU, node, out_buf);
            }

            Op::SwiGLUConcat => {
                // input[M, 2*N] → output[M, N]
                let input = self.get_buffer(node.inputs[0]);
                let out_len = node.ty.num_elements() as u32;
                let half_n = node.ty.shape[1] as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::SwiGLUConcat,
                    workgroups: [out_len.div_ceil(256), 1, 1],
                    input_buffers: vec![input, input], // src_b unused in forward
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![out_len, half_n, 0, 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::SwiGLUConcatGrad => {
                // (grad_out[M,N], input[M,2*N]) → grad_input[M,2*N]
                let grad_out = self.get_buffer(node.inputs[0]);
                let input = self.get_buffer(node.inputs[1]);
                let grad_out_len = self.graph.node(node.inputs[0]).ty.num_elements() as u32;
                let half_n = self.graph.node(node.inputs[0]).ty.shape[1] as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::SwiGLUConcatGrad,
                    workgroups: [grad_out_len.div_ceil(256), 1, 1],
                    input_buffers: vec![input, grad_out],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![grad_out_len, half_n, 0, 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::RmsNorm { eps } => {
                let x = self.get_buffer(node.inputs[0]);
                let w = self.get_buffer(node.inputs[1]);
                let shape = &self.graph.node(node.inputs[0]).ty.shape;
                let rows = shape[0] as u32;
                let cols = shape[1] as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::RmsNorm,
                    workgroups: [rows, 1, 1], // one workgroup per row
                    input_buffers: vec![x, w],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![rows, cols, eps.to_bits(), 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::Embedding => {
                let indices = self.get_buffer(node.inputs[0]);
                let table = self.get_buffer(node.inputs[1]);
                let idx_shape = &self.graph.node(node.inputs[0]).ty.shape;
                let tbl_shape = &self.graph.node(node.inputs[1]).ty.shape;
                let seq = idx_shape[0] as u32;
                let hidden = tbl_shape[1] as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::Embedding,
                    workgroups: [seq * hidden.div_ceil(256), 1, 1],
                    input_buffers: vec![indices, table],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![seq, hidden, 0, 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::ScatterAdd { vocab_size } => {
                let indices = self.get_buffer(node.inputs[0]);
                let src = self.get_buffer(node.inputs[1]);
                let src_shape = &self.graph.node(node.inputs[1]).ty.shape;
                let seq_len = src_shape[0] as u32;
                let embed_dim = src_shape[1] as u32;
                let total = vocab_size as u32 * embed_dim;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::ScatterAdd,
                    workgroups: [total.div_ceil(256), 1, 1],
                    input_buffers: vec![indices, src],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![total, seq_len, embed_dim, 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::RoPE {
                theta,
                pos_offset,
                head_dim,
            } => {
                let input = self.get_buffer(node.inputs[0]);
                let shape = &self.graph.node(node.inputs[0]).ty.shape;
                let seq = shape[0] as u32;
                let dim = shape[1] as u32;
                if node.inputs.len() == 2 {
                    // Dynamic offset: read pos_offset from input buffer
                    let offset_buf = self.get_buffer(node.inputs[1]);
                    self.plan.dispatches.push(Dispatch {
                        shader: ShaderEntry::RoPEDynamic,
                        workgroups: [(seq * dim / 2).div_ceil(256), 1, 1],
                        input_buffers: vec![input, offset_buf],
                        output_buffer: out_buf,
                        extra_outputs: vec![],
                        params: vec![seq, dim, theta.to_bits(), 0, head_dim, 0, 0, 0],
                        use_coop: false,
                        use_small_tiles: false,
                        ..Default::default()
                    });
                } else {
                    self.plan.dispatches.push(Dispatch {
                        shader: ShaderEntry::RoPE,
                        workgroups: [(seq * dim / 2).div_ceil(256), 1, 1],
                        input_buffers: vec![input],
                        output_buffer: out_buf,
                        extra_outputs: vec![],
                        params: vec![seq, dim, theta.to_bits(), pos_offset, head_dim, 0, 0, 0],
                        use_coop: false,
                        use_small_tiles: false,
                        ..Default::default()
                    });
                }
            }

            Op::CausalAttention {
                num_heads,
                num_kv_heads,
                head_dim,
            } => {
                let q = self.get_buffer(node.inputs[0]);
                let k = self.get_buffer(node.inputs[1]);
                let v = self.get_buffer(node.inputs[2]);
                let seq = self.graph.node(node.inputs[0]).ty.shape[0] as u32;
                let lse_buf = self.find_lse_buffer(node.id);
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::CausalAttention,
                    workgroups: [seq.div_ceil(1), num_heads, 1],
                    input_buffers: vec![q, k, v],
                    output_buffer: out_buf,
                    extra_outputs: vec![lse_buf],
                    params: vec![seq, num_heads, num_kv_heads, head_dim],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::CausalAttentionRoPE {
                num_heads,
                num_kv_heads,
                head_dim,
                rope_theta: _,
            } => {
                let q = self.get_buffer(node.inputs[0]);
                let k = self.get_buffer(node.inputs[1]);
                let v = self.get_buffer(node.inputs[2]);
                let seq = self.graph.node(node.inputs[0]).ty.shape[0] as u32;
                let lse_buf = self.find_lse_buffer(node.id);
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::CausalAttentionRoPE,
                    workgroups: [seq.div_ceil(1), num_heads, 1],
                    input_buffers: vec![q, k, v],
                    output_buffer: out_buf,
                    extra_outputs: vec![lse_buf],
                    // TODO: pass rope_theta via params when the RoPE-fused shader is implemented
                    params: vec![seq, num_heads, num_kv_heads, head_dim],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::SlidingWindowAttention {
                num_heads,
                num_kv_heads,
                head_dim,
                window_size,
            } => {
                let q = self.get_buffer(node.inputs[0]);
                let k = self.get_buffer(node.inputs[1]);
                let v = self.get_buffer(node.inputs[2]);
                let seq = self.graph.node(node.inputs[0]).ty.shape[0] as u32;
                let lse_buf = self.find_lse_buffer(node.id);
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::SlidingWindowAttention,
                    workgroups: [seq.div_ceil(1), num_heads, 1],
                    input_buffers: vec![q, k, v],
                    output_buffer: out_buf,
                    extra_outputs: vec![lse_buf],
                    params: vec![seq, num_heads, num_kv_heads, head_dim, window_size],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::RoPEGrad {
                theta,
                pos_offset,
                head_dim,
            } => {
                let grad_out = self.get_buffer(node.inputs[0]);
                let shape = &self.graph.node(node.inputs[0]).ty.shape;
                let seq = shape[0] as u32;
                let dim = shape[1] as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::RoPEGrad,
                    workgroups: [(seq * dim / 2).div_ceil(256), 1, 1],
                    input_buffers: vec![grad_out],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![seq, dim, theta.to_bits(), pos_offset, head_dim, 0, 0, 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::GroupNorm {
                num_groups,
                eps,
                channels,
                spatial,
            } => {
                let x = self.get_buffer(node.inputs[0]);
                let weight = self.get_buffer(node.inputs[1]);
                let bias = self.get_buffer(node.inputs[2]);
                let total = node.ty.shape[0] as u32;
                let batch = total / (channels * spatial);
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::GroupNorm,
                    workgroups: [batch * num_groups, 1, 1],
                    input_buffers: vec![x, weight, bias],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![batch, channels, spatial, num_groups, eps.to_bits(), 0, 0, 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::GroupNormSilu {
                num_groups,
                eps,
                channels,
                spatial,
            } => {
                let x = self.get_buffer(node.inputs[0]);
                let weight = self.get_buffer(node.inputs[1]);
                let bias = self.get_buffer(node.inputs[2]);
                let total = node.ty.shape[0] as u32;
                let batch = total / (channels * spatial);
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::GroupNormSilu,
                    workgroups: [batch * num_groups, 1, 1],
                    input_buffers: vec![x, weight, bias],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![batch, channels, spatial, num_groups, eps.to_bits(), 0, 0, 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::GroupNormGradInput {
                num_groups,
                eps,
                channels,
                spatial,
            } => {
                let grad_out = self.get_buffer(node.inputs[0]);
                let input = self.get_buffer(node.inputs[1]);
                let weight = self.get_buffer(node.inputs[2]);
                let total = node.ty.shape[0] as u32;
                let batch = total / (channels * spatial);
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::GroupNormGradInput,
                    workgroups: [batch * num_groups, 1, 1],
                    input_buffers: vec![grad_out, input, weight],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![batch, channels, spatial, num_groups, eps.to_bits(), 0, 0, 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::GroupNormGradWeightBias {
                num_groups,
                eps,
                channels,
                spatial,
            } => {
                let grad_out = self.get_buffer(node.inputs[0]);
                let input = self.get_buffer(node.inputs[1]);
                let go_total = self.graph.node(node.inputs[0]).ty.shape[0] as u32;
                let batch = go_total / (channels * spatial);
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::GroupNormGradWeightBias,
                    workgroups: [channels, 1, 1],
                    input_buffers: vec![grad_out, input],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![batch, channels, spatial, num_groups, eps.to_bits(), 0, 0, 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::Concat {
                channels_a,
                channels_b,
                spatial,
            } => {
                let a = self.get_buffer(node.inputs[0]);
                let b = self.get_buffer(node.inputs[1]);
                let total = node.ty.shape[0] as u32;
                let batch = total / ((channels_a + channels_b) * spatial);
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::Concat,
                    workgroups: [total.div_ceil(256), 1, 1],
                    input_buffers: vec![a, b],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![batch, channels_a, channels_b, spatial],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::SplitA {
                channels_a,
                channels_b,
                spatial,
            } => {
                let x = self.get_buffer(node.inputs[0]);
                let total = node.ty.shape[0] as u32;
                let batch = total / (channels_a * spatial);
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::SplitA,
                    workgroups: [total.div_ceil(256), 1, 1],
                    input_buffers: vec![x],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![batch, channels_a, channels_b, spatial],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::SplitB {
                channels_a,
                channels_b,
                spatial,
            } => {
                let x = self.get_buffer(node.inputs[0]);
                let total = node.ty.shape[0] as u32;
                let batch = total / (channels_b * spatial);
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::SplitB,
                    workgroups: [total.div_ceil(256), 1, 1],
                    input_buffers: vec![x],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![batch, channels_a, channels_b, spatial],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::Upsample2x {
                channels,
                in_h,
                in_w,
            } => {
                let x = self.get_buffer(node.inputs[0]);
                let total = node.ty.shape[0] as u32;
                let batch = total / (channels * in_h * 2 * in_w * 2);
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::Upsample2x,
                    workgroups: [total.div_ceil(256), 1, 1],
                    input_buffers: vec![x],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![batch, channels, in_h, in_w],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::Upsample2xGrad {
                channels,
                in_h,
                in_w,
            } => {
                let grad = self.get_buffer(node.inputs[0]);
                let total = node.ty.shape[0] as u32;
                let batch = total / (channels * in_h * in_w);
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::Upsample2xGrad,
                    workgroups: [total.div_ceil(256), 1, 1],
                    input_buffers: vec![grad],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![batch, channels, in_h, in_w],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

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
            } => {
                let input = self.get_buffer(node.inputs[0]);
                let kernel = self.get_buffer(node.inputs[1]);
                let in_shape = &self.graph.node(node.inputs[0]).ty.shape;
                let out_h = (in_h + 2 * padding_h - kernel_h) / stride + 1;
                let out_w = (in_w + 2 * padding_w - kernel_w) / stride + 1;
                let batch = in_shape[0] as u32 / (in_channels * in_h * in_w);
                // Use implicit GEMM: output = weight @ im2col(input)^T
                // M=Co, N=oH*oW, K=Ci*kH*kW, batched in z dimension
                // Use small (32×32) tiles when workgroup count per batch is low.
                let wgs_64 = out_h * out_w.div_ceil(64) * out_channels.div_ceil(64);
                let use_small = wgs_64 < 16;
                let tile = if use_small { 32 } else { 64 };
                self.plan.dispatches.push(Dispatch {
                    shader: if use_small {
                        ShaderEntry::Conv2dGemmSmall
                    } else {
                        ShaderEntry::Conv2dGemm
                    },
                    workgroups: [
                        out_h * out_w.div_ceil(tile),
                        out_channels.div_ceil(tile),
                        batch,
                    ],
                    input_buffers: vec![input, kernel],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![
                        batch,
                        in_channels,
                        in_h,
                        in_w,
                        out_channels,
                        kernel_h,
                        kernel_w,
                        stride,
                        padding_h,
                        out_h,
                        out_w,
                        padding_w,
                    ],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

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
            } => {
                let grad_out = self.get_buffer(node.inputs[0]);
                let kernel = self.get_buffer(node.inputs[1]);
                let out_h = (in_h + 2 * padding_h - kernel_h) / stride + 1;
                let out_w = (in_w + 2 * padding_w - kernel_w) / stride + 1;
                let out_size = node.ty.shape[0] as u32;
                let batch = out_size / (in_channels * in_h * in_w);
                // Use implicit GEMM: grad_input = weight_T @ im2col(grad_out)^T
                // M=Ci, N=H*W, K=Co*kH*kW, batched in z dimension
                let wgs_64 = in_h * in_w.div_ceil(64) * in_channels.div_ceil(64);
                let use_small = wgs_64 < 16;
                let tile = if use_small { 32 } else { 64 };
                self.plan.dispatches.push(Dispatch {
                    shader: if use_small {
                        ShaderEntry::Conv2dGradInputGemmSmall
                    } else {
                        ShaderEntry::Conv2dGradInputGemm
                    },
                    workgroups: [
                        in_h * in_w.div_ceil(tile),
                        in_channels.div_ceil(tile),
                        batch,
                    ],
                    input_buffers: vec![grad_out, kernel],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![
                        batch,
                        in_channels,
                        in_h,
                        in_w,
                        out_channels,
                        kernel_h,
                        kernel_w,
                        stride,
                        padding_h,
                        out_h,
                        out_w,
                        padding_w,
                    ],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

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
            } => {
                let grad_out = self.get_buffer(node.inputs[0]);
                let input = self.get_buffer(node.inputs[1]);
                let out_h = (in_h + 2 * padding_h - kernel_h) / stride + 1;
                let out_w = (in_w + 2 * padding_w - kernel_w) / stride + 1;
                let out_size = self.graph.node(node.inputs[0]).ty.shape[0] as u32;
                let batch = out_size / (out_channels * out_h * out_w);
                // Use GEMM formulation: grad_weight[Co, Ci*kH*kW] = grad_out_flat[Co, N*oH*oW] @ im2col(input)[N*oH*oW, Ci*kH*kW]
                let n_total = in_channels * kernel_h * kernel_w; // Ci*kH*kW
                let m_total = out_channels; // Co
                let wgs_64 = n_total.div_ceil(64) * m_total.div_ceil(64);
                let use_small = wgs_64 < 16;
                let tile = if use_small { 32 } else { 64 };
                self.plan.dispatches.push(Dispatch {
                    shader: if use_small {
                        ShaderEntry::Conv2dGradWeightGemmSmall
                    } else {
                        ShaderEntry::Conv2dGradWeightGemm
                    },
                    workgroups: [n_total.div_ceil(tile), m_total.div_ceil(tile), 1],
                    input_buffers: vec![grad_out, input],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![
                        batch,
                        in_channels,
                        in_h,
                        in_w,
                        out_channels,
                        kernel_h,
                        kernel_w,
                        stride,
                        padding_h,
                        out_h,
                        out_w,
                        padding_w,
                    ],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::CacheWrite => {
                let new_kv = self.get_buffer(node.inputs[0]);
                let cache = self.get_buffer(node.inputs[1]);
                let kv_pos_input = self.get_buffer(node.inputs[2]);
                let dim = self.graph.node(node.inputs[0]).ty.shape[1] as u32;
                // Output aliases the cache buffer (in-place write)
                self.node_buffers.insert(node.id, cache);
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::CacheWrite,
                    workgroups: [dim.div_ceil(256), 1, 1],
                    input_buffers: vec![new_kv, cache, kv_pos_input],
                    output_buffer: cache,
                    extra_outputs: vec![],
                    params: vec![dim, 0, 0, 0], // kv_pos read from input buffer at runtime
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::CachedAttention {
                num_heads,
                num_kv_heads,
                head_dim,
            } => {
                let q = self.get_buffer(node.inputs[0]);
                let k_cache = self.get_buffer(node.inputs[1]);
                let v_cache = self.get_buffer(node.inputs[2]);
                let kv_pos_input = self.get_buffer(node.inputs[3]);
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::CachedAttention,
                    workgroups: [1, num_heads, 1],
                    input_buffers: vec![q, k_cache, v_cache, kv_pos_input],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![0, num_heads, num_kv_heads, head_dim], // kv_len read from input buffer
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::MaxPool2d {
                channels,
                in_h,
                in_w,
                kernel_h,
                kernel_w,
                stride,
                padding,
            } => {
                let input = self.get_buffer(node.inputs[0]);
                let in_shape = &self.graph.node(node.inputs[0]).ty.shape;
                let batch = in_shape[0] as u32 / (channels * in_h * in_w);
                let out_h = (in_h + 2 * padding - kernel_h) / stride + 1;
                let out_w = (in_w + 2 * padding - kernel_w) / stride + 1;
                let total = batch * channels * out_h * out_w;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::MaxPool2d,
                    workgroups: [total.div_ceil(256), 1, 1],
                    input_buffers: vec![input],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![
                        batch, channels, in_h, in_w, kernel_h, kernel_w, stride, padding, out_h,
                        out_w, 0, 0,
                    ],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::GlobalAvgPool { channels, spatial } => {
                let input = self.get_buffer(node.inputs[0]);
                let total_out = node.ty.num_elements() as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::GlobalAvgPool,
                    workgroups: [total_out.div_ceil(256), 1, 1],
                    input_buffers: vec![input],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![channels, spatial, total_out, 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::GlobalAvgPoolGrad {
                channels: _,
                spatial,
            } => {
                let grad_output = self.get_buffer(node.inputs[0]);
                let total = node.ty.num_elements() as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::GlobalAvgPoolGrad,
                    workgroups: [total.div_ceil(256), 1, 1],
                    input_buffers: vec![grad_output],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![total, spatial, 0, 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::Gelu => {
                self.emit_unary(ShaderEntry::Gelu, node, out_buf);
            }

            Op::LayerNorm { eps } => {
                let x = self.get_buffer(node.inputs[0]);
                let w = self.get_buffer(node.inputs[1]);
                let bias = self.get_buffer(node.inputs[2]);
                let shape = &self.graph.node(node.inputs[0]).ty.shape;
                let rows = shape[0] as u32;
                let cols = shape[1] as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::LayerNorm,
                    workgroups: [rows.div_ceil(256), 1, 1],
                    input_buffers: vec![x, w, bias],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![rows, cols, eps.to_bits(), 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::FullAttention {
                num_heads,
                num_kv_heads,
                head_dim,
            } => {
                let q = self.get_buffer(node.inputs[0]);
                let k = self.get_buffer(node.inputs[1]);
                let v = self.get_buffer(node.inputs[2]);
                let seq = self.graph.node(node.inputs[0]).ty.shape[0] as u32;
                let lse_buf = self.find_lse_buffer(node.id);
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::FullAttention,
                    workgroups: [seq.div_ceil(1), num_heads, 1],
                    input_buffers: vec![q, k, v],
                    output_buffer: out_buf,
                    extra_outputs: vec![lse_buf],
                    params: vec![seq, num_heads, num_kv_heads, head_dim],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::CrossAttention {
                num_heads,
                num_kv_heads,
                head_dim,
            } => {
                let q = self.get_buffer(node.inputs[0]);
                let k = self.get_buffer(node.inputs[1]);
                let v = self.get_buffer(node.inputs[2]);
                let q_seq = self.graph.node(node.inputs[0]).ty.shape[0] as u32;
                let kv_seq = self.graph.node(node.inputs[1]).ty.shape[0] as u32;
                let lse_buf = self.find_lse_buffer(node.id);
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::CrossAttention,
                    workgroups: [q_seq.div_ceil(1), num_heads, 1],
                    input_buffers: vec![q, k, v],
                    output_buffer: out_buf,
                    extra_outputs: vec![lse_buf],
                    params: vec![q_seq, kv_seq, (num_heads << 16) | num_kv_heads, head_dim],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::MultiHeadAttn {
                num_heads,
                num_kv_heads,
                head_dim,
                ..
            } => {
                let q = self.get_buffer(node.inputs[0]);
                let k = self.get_buffer(node.inputs[1]);
                let v = self.get_buffer(node.inputs[2]);
                let q_seq = self.graph.node(node.inputs[0]).ty.shape[0] as u32;
                let kv_seq = self.graph.node(node.inputs[1]).ty.shape[0] as u32;
                let lse_buf = self.find_lse_buffer(node.id);
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::MultiHeadAttn,
                    workgroups: [q_seq, num_heads, 1],
                    input_buffers: vec![q, k, v],
                    output_buffer: out_buf,
                    extra_outputs: vec![lse_buf],
                    params: vec![q_seq, kv_seq, (num_heads << 16) | num_kv_heads, head_dim],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::MultiHeadAttnGradQ {
                fwd_node,
                num_heads,
                num_kv_heads,
                head_dim,
                ..
            } => {
                let d_out = self.get_buffer(node.inputs[0]);
                let q = self.get_buffer(node.inputs[1]);
                let k = self.get_buffer(node.inputs[2]);
                let v = self.get_buffer(node.inputs[3]);
                let fwd_o = self.get_buffer(fwd_node);
                let lse_buf = self.find_lse_buffer(fwd_node);
                let q_seq = self.graph.node(node.inputs[1]).ty.shape[0] as u32;
                let fwd_op = &self.graph.node(fwd_node).op;
                let is_causal = matches!(
                    fwd_op,
                    Op::CausalAttention { .. }
                        | Op::CausalAttentionRoPE { .. }
                        | Op::SlidingWindowAttention { .. }
                );
                let kv_seq = if is_causal {
                    0
                } else {
                    self.graph.node(node.inputs[2]).ty.shape[0] as u32
                };
                let window_size = match *fwd_op {
                    Op::SlidingWindowAttention { window_size, .. } => window_size,
                    _ => 0,
                };
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::MultiHeadAttnGradQ,
                    workgroups: [q_seq, num_heads, 1],
                    input_buffers: vec![d_out, q, k, v, lse_buf, fwd_o],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![
                        q_seq,
                        kv_seq,
                        (num_heads << 16) | num_kv_heads,
                        head_dim,
                        window_size,
                    ],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::MultiHeadAttnGradK {
                fwd_node,
                num_heads,
                num_kv_heads,
                head_dim,
                ..
            } => {
                let d_out = self.get_buffer(node.inputs[0]);
                let q = self.get_buffer(node.inputs[1]);
                let k = self.get_buffer(node.inputs[2]);
                let v = self.get_buffer(node.inputs[3]);
                let fwd_o = self.get_buffer(fwd_node);
                let lse_buf = self.find_lse_buffer(fwd_node);
                let q_seq = self.graph.node(node.inputs[1]).ty.shape[0] as u32;
                let fwd_op = &self.graph.node(fwd_node).op;
                let is_causal = matches!(
                    fwd_op,
                    Op::CausalAttention { .. }
                        | Op::CausalAttentionRoPE { .. }
                        | Op::SlidingWindowAttention { .. }
                );
                let kv_seq = if is_causal {
                    0
                } else {
                    self.graph.node(node.inputs[2]).ty.shape[0] as u32
                };
                let window_size = match *fwd_op {
                    Op::SlidingWindowAttention { window_size, .. } => window_size,
                    _ => 0,
                };
                let dispatch_kv = if is_causal { q_seq } else { kv_seq };
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::MultiHeadAttnGradK,
                    workgroups: [dispatch_kv, num_kv_heads, 1],
                    input_buffers: vec![d_out, q, k, v, lse_buf, fwd_o],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![
                        q_seq,
                        kv_seq,
                        (num_heads << 16) | num_kv_heads,
                        head_dim,
                        window_size,
                    ],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::MultiHeadAttnGradV {
                fwd_node,
                num_heads,
                num_kv_heads,
                head_dim,
                ..
            } => {
                let d_out = self.get_buffer(node.inputs[0]);
                let q = self.get_buffer(node.inputs[1]);
                let k = self.get_buffer(node.inputs[2]);
                let v = self.get_buffer(node.inputs[3]);
                let fwd_o = self.get_buffer(fwd_node);
                let lse_buf = self.find_lse_buffer(fwd_node);
                let q_seq = self.graph.node(node.inputs[1]).ty.shape[0] as u32;
                let fwd_op = &self.graph.node(fwd_node).op;
                let is_causal = matches!(
                    fwd_op,
                    Op::CausalAttention { .. }
                        | Op::CausalAttentionRoPE { .. }
                        | Op::SlidingWindowAttention { .. }
                );
                let kv_seq = if is_causal {
                    0
                } else {
                    self.graph.node(node.inputs[2]).ty.shape[0] as u32
                };
                let window_size = match *fwd_op {
                    Op::SlidingWindowAttention { window_size, .. } => window_size,
                    _ => 0,
                };
                let dispatch_kv = if is_causal { q_seq } else { kv_seq };
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::MultiHeadAttnGradV,
                    workgroups: [dispatch_kv, num_kv_heads, 1],
                    input_buffers: vec![d_out, q, k, v, lse_buf, fwd_o],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![
                        q_seq,
                        kv_seq,
                        (num_heads << 16) | num_kv_heads,
                        head_dim,
                        window_size,
                    ],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::SwiGLUGradGate => {
                // inputs: [grad_out, gate, up]
                let grad_out = self.get_buffer(node.inputs[0]);
                let gate = self.get_buffer(node.inputs[1]);
                let up = self.get_buffer(node.inputs[2]);
                let len = node.ty.num_elements() as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::SwiGLUGradGate,
                    workgroups: [len.div_ceil(256), 1, 1],
                    input_buffers: vec![grad_out, gate, up],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![len, 0, 0, 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::SwiGLUGradUp => {
                // inputs: [grad_out, gate]
                let grad_out = self.get_buffer(node.inputs[0]);
                let gate = self.get_buffer(node.inputs[1]);
                let len = node.ty.num_elements() as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::SwiGLUGradUp,
                    workgroups: [len.div_ceil(256), 1, 1],
                    input_buffers: vec![grad_out, gate],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![len, 0, 0, 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::SiluGrad => {
                // inputs: [grad_out, x]
                let grad_out = self.get_buffer(node.inputs[0]);
                let x = self.get_buffer(node.inputs[1]);
                let len = node.ty.num_elements() as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::SiluGrad,
                    workgroups: [len.div_ceil(256), 1, 1],
                    input_buffers: vec![grad_out, x],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![len, 0, 0, 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::FusedRmsNormMatMul { eps } => {
                // Single dispatch: coop matmul with cooperative rsqrt prologue.
                // The coop shader (matmul_rms_norm_coop.wgsl) computes rsqrt using
                // 64-thread tree reduction in the prologue, then applies normalization
                // during the A-staging phase with tensor cores.
                let x = self.get_buffer(node.inputs[0]);
                let w_norm = self.get_buffer(node.inputs[1]);
                let w_proj = self.get_buffer(node.inputs[2]);
                let x_shape = &self.graph.node(node.inputs[0]).ty.shape;
                let w_proj_shape = &self.graph.node(node.inputs[2]).ty.shape;
                let m = x_shape[0] as u32;
                let k = x_shape[1] as u32;
                let n = w_proj_shape[1] as u32;

                // Single fused dispatch: rsqrt prologue + coop matmul
                // input_buffers: [x, w_proj, w_norm] for scalar path
                //                [x, w_proj, (unused), w_norm] for coop path
                // The coop selection will mark this for coop and the runtime
                // binds FusedRmsNormMatMulCoopData with rsqrt_cache in shared mem.
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::FusedRmsNormMatMul,
                    workgroups: [n.div_ceil(64), m.div_ceil(64), 1],
                    input_buffers: vec![x, w_norm, w_proj],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![m, n, k, eps.to_bits()],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::RmsNormGradW { eps } => {
                let dy = self.get_buffer(node.inputs[0]);
                let x = self.get_buffer(node.inputs[1]);
                let w = self.get_buffer(node.inputs[2]);
                let x_shape = &self.graph.node(node.inputs[1]).ty.shape;
                let rows = x_shape[0] as u32;
                let cols = x_shape[1] as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::RmsNormGradW,
                    workgroups: [cols.div_ceil(256), 1, 1],
                    input_buffers: vec![dy, x, w],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![rows, cols, eps.to_bits(), 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }

            Op::RmsNormGradX { eps } => {
                let dy = self.get_buffer(node.inputs[0]);
                let x = self.get_buffer(node.inputs[1]);
                let w = self.get_buffer(node.inputs[2]);
                let x_shape = &self.graph.node(node.inputs[1]).ty.shape;
                let rows = x_shape[0] as u32;
                let cols = x_shape[1] as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::RmsNormGradX,
                    workgroups: [rows, 1, 1],
                    input_buffers: vec![dy, x, w],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![rows, cols, eps.to_bits(), 0],
                    use_coop: false,
                    use_small_tiles: false,
                    ..Default::default()
                });
            }
        }
    }

    fn find_lse_buffer(&self, fwd_node: NodeId) -> BufferRef {
        self.plan
            .lse_buffers
            .iter()
            .find(|item| item.0 == fwd_node)
            .expect("LSE buffer not found for MultiHeadAttn forward node")
            .1
    }

    fn emit_unary(&mut self, shader: ShaderEntry, node: &Node, out_buf: BufferRef) {
        let input = self.get_buffer(node.inputs[0]);
        let len = node.ty.num_elements() as u32;
        self.plan.dispatches.push(Dispatch {
            shader,
            workgroups: [len.div_ceil(256), 1, 1],
            input_buffers: vec![input],
            output_buffer: out_buf,
            extra_outputs: vec![],
            params: vec![len, 0, 0, 0],
            use_coop: false,
            use_small_tiles: false,
            ..Default::default()
        });
    }

    fn emit_binary(&mut self, shader: ShaderEntry, node: &Node, out_buf: BufferRef) {
        let a = self.get_buffer(node.inputs[0]);
        let b = self.get_buffer(node.inputs[1]);
        let len = node.ty.num_elements() as u32;
        self.plan.dispatches.push(Dispatch {
            shader,
            workgroups: [len.div_ceil(256), 1, 1],
            input_buffers: vec![a, b],
            output_buffer: out_buf,
            extra_outputs: vec![],
            params: vec![len, 0, 0, 0],
            use_coop: false,
            use_small_tiles: false,
            ..Default::default()
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;

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
        assert_eq!(plan.dispatches[0].epilogue, vec![EpilogueOp::Relu]);
    }

    #[test]
    fn test_compile_all_unary_ops() {
        let mut g = Graph::new();
        let x = g.input("x", &[4, 8]);
        let r = g.relu(x);
        let s = g.sigmoid(x);
        let n = g.neg(x);
        g.set_outputs(vec![r, s, n]);

        let plan = compile(&g);
        assert_eq!(plan.dispatches.len(), 3);
        assert_eq!(plan.dispatches[0].shader, ShaderEntry::Relu);
        assert_eq!(plan.dispatches[1].shader, ShaderEntry::Sigmoid);
        assert_eq!(plan.dispatches[2].shader, ShaderEntry::Neg);
        // All unary ops: params = [len, 0, 0, 0]
        for d in &plan.dispatches {
            assert_eq!(d.params[0], 32); // 4*8
            assert_eq!(d.input_buffers.len(), 1);
        }
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
        assert_eq!(plan.dispatches[0].shader, ShaderEntry::Add);
        assert_eq!(plan.dispatches[1].shader, ShaderEntry::Mul);
        assert_eq!(plan.dispatches[2].shader, ShaderEntry::Greater);
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
        assert_eq!(plan.dispatches[0].shader, ShaderEntry::BiasAdd);
        assert_eq!(plan.dispatches[0].params[0], 512); // 4*128
        assert_eq!(plan.dispatches[0].params[1], 128); // bias len
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
    fn test_compile_softmax() {
        let mut g = Graph::new();
        let x = g.input("x", &[4, 10]);
        let sm = g.softmax(x);
        g.set_outputs(vec![sm]);

        let plan = compile(&g);
        assert_eq!(plan.dispatches.len(), 1);
        assert_eq!(plan.dispatches[0].shader, ShaderEntry::Softmax);
        assert_eq!(plan.dispatches[0].params[0], 4); // batch
        assert_eq!(plan.dispatches[0].params[1], 10); // features
    }

    #[test]
    fn test_compile_cross_entropy() {
        let mut g = Graph::new();
        let logits = g.input("logits", &[4, 10]);
        let labels = g.input("labels", &[4, 10]);
        let loss = g.cross_entropy_loss(logits, labels);
        g.set_outputs(vec![loss]);

        let plan = compile(&g);
        assert_eq!(plan.dispatches.len(), 1);
        assert_eq!(plan.dispatches[0].shader, ShaderEntry::CrossEntropyLoss);
        assert_eq!(plan.dispatches[0].workgroups, [1, 1, 1]);
        assert_eq!(plan.dispatches[0].params[0], 4);
        assert_eq!(plan.dispatches[0].params[1], 10);
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
        assert_eq!(plan.dispatches[0].shader, ShaderEntry::Relu);
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
        // With cooperative matrix, matmul+bias_add+relu are separate dispatches
        assert_eq!(plan.dispatches.len(), 3);
        assert_eq!(plan.dispatches[0].shader, ShaderEntry::MatMul);
        assert_eq!(plan.dispatches[1].shader, ShaderEntry::BiasAdd);
        assert_eq!(plan.dispatches[2].shader, ShaderEntry::Relu);
    }

    #[test]
    fn test_shader_entry_mappings() {
        // Verify all shader entries have valid group and entry_point
        let entries = [
            ShaderEntry::MatMul,
            ShaderEntry::Relu,
            ShaderEntry::Sigmoid,
            ShaderEntry::Neg,
            ShaderEntry::Abs,
            ShaderEntry::Log,
            ShaderEntry::Recip,
            ShaderEntry::Add,
            ShaderEntry::Mul,
            ShaderEntry::Greater,
            ShaderEntry::BiasAdd,
            ShaderEntry::SgdUpdate,
            ShaderEntry::AdamUpdate,
            ShaderEntry::ScatterAdd,
            ShaderEntry::SumAll,
            ShaderEntry::MeanAll,
            ShaderEntry::SumRows,
            ShaderEntry::Softmax,
            ShaderEntry::CrossEntropyLoss,
            ShaderEntry::BceLoss,
            ShaderEntry::Transpose,
            ShaderEntry::SwiGLUGradGate,
            ShaderEntry::SwiGLUGradUp,
            ShaderEntry::SiluGrad,
            ShaderEntry::RmsNormGradW,
            ShaderEntry::RmsNormGradX,
            ShaderEntry::FusedRmsNormMatMul,
        ];
        for entry in &entries {
            let _group = entry.shader_group();
            let ep = entry.entry_point();
            assert!(!ep.is_empty());
        }
    }
}
