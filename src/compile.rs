use crate::graph::{Graph, Node, NodeId, Op};
use std::collections::HashMap;

/// Identifies which shader and entry point to use.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum ShaderEntry {
    MatMul,
    MatMulRelu,
    MatMulBiasRelu,
    Relu,
    Sigmoid,
    Neg,
    Add,
    Mul,
    Greater,
    BiasAdd,
    SgdUpdate,
    SumAll,
    MeanAll,
    Softmax,
    CrossEntropyLoss,
    Transpose,
}

impl ShaderEntry {
    pub fn shader_group(&self) -> crate::codegen::ShaderGroup {
        use crate::codegen::ShaderGroup;
        match *self {
            ShaderEntry::MatMul => ShaderGroup::MatMul,
            ShaderEntry::MatMulRelu => ShaderGroup::MatMulRelu,
            ShaderEntry::MatMulBiasRelu => ShaderGroup::MatMulBiasRelu,
            ShaderEntry::Relu | ShaderEntry::Sigmoid | ShaderEntry::Neg => ShaderGroup::Unary,
            ShaderEntry::Add | ShaderEntry::Mul | ShaderEntry::Greater => ShaderGroup::Binary,
            ShaderEntry::BiasAdd => ShaderGroup::BiasAdd,
            ShaderEntry::SgdUpdate => ShaderGroup::Sgd,
            ShaderEntry::SumAll | ShaderEntry::MeanAll => ShaderGroup::Reduce,
            ShaderEntry::Softmax => ShaderGroup::Softmax,
            ShaderEntry::CrossEntropyLoss => ShaderGroup::CrossEntropy,
            ShaderEntry::Transpose => ShaderGroup::Transpose,
        }
    }

    pub fn entry_point(&self) -> &'static str {
        match *self {
            ShaderEntry::MatMul
            | ShaderEntry::MatMulRelu
            | ShaderEntry::MatMulBiasRelu
            | ShaderEntry::BiasAdd
            | ShaderEntry::SgdUpdate
            | ShaderEntry::Softmax
            | ShaderEntry::CrossEntropyLoss
            | ShaderEntry::Transpose => "main",
            ShaderEntry::Relu => "relu",
            ShaderEntry::Sigmoid => "sigmoid",
            ShaderEntry::Neg => "neg",
            ShaderEntry::Add => "add",
            ShaderEntry::Mul => "mul",
            ShaderEntry::Greater => "greater",
            ShaderEntry::SumAll => "sum_all",
            ShaderEntry::MeanAll => "mean_all",
        }
    }
}

/// A single GPU dispatch in the execution plan.
#[derive(Clone, Debug)]
pub struct Dispatch {
    pub shader: ShaderEntry,
    pub workgroups: [u32; 3],
    /// Buffer bindings: maps the node IDs for inputs/outputs to buffer slots.
    pub input_buffers: Vec<BufferRef>,
    pub output_buffer: BufferRef,
    /// Extra params to upload as a uniform buffer.
    pub params: Vec<u32>,
}

/// Reference to a GPU buffer in the execution plan.
#[derive(Clone, Debug, Copy, PartialEq, Eq, Hash)]
pub struct BufferRef(pub u32);

/// The complete execution plan: a static sequence of dispatches.
#[derive(Clone, Debug)]
pub struct ExecutionPlan {
    /// Buffer sizes in bytes, indexed by BufferRef.
    pub buffers: Vec<usize>,
    /// Which buffers hold parameters (need initialization).
    pub param_buffers: Vec<(String, BufferRef)>,
    /// Which buffers hold inputs (filled each step).
    pub input_buffers: Vec<(String, BufferRef)>,
    /// The dispatch sequence. For a training graph, this includes
    /// forward, backward, and parameter update dispatches.
    pub dispatches: Vec<Dispatch>,
    /// Index of the loss buffer (for reading back).
    pub loss_buffer: Option<BufferRef>,
    /// Parameter buffer → gradient buffer mapping (for SGD).
    pub param_grad_pairs: Vec<(BufferRef, BufferRef)>,
}

/// Compile a differentiated graph into an ExecutionPlan.
pub fn compile(graph: &Graph) -> ExecutionPlan {
    let mut compiler = Compiler::new(graph);
    compiler.compile();
    compiler.plan
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
                dispatches: Vec::new(),
                loss_buffer: None,
                param_grad_pairs: Vec::new(),
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
                _ => {}
            }
        }

        // Second pass: emit dispatches for each non-leaf node
        for node in self.graph.nodes() {
            self.compile_node(node);
        }

        // Set loss buffer (first output)
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
                    workgroups: [ceil_div(n, 16), ceil_div(m, 16), 1],
                    input_buffers: vec![a, b],
                    output_buffer: out_buf,
                    params: vec![m, k, n, 0],
                });
            }

            Op::FusedMatMulRelu => {
                let a = self.get_buffer(node.inputs[0]);
                let b = self.get_buffer(node.inputs[1]);
                let a_shape = &self.graph.node(node.inputs[0]).ty.shape;
                let b_shape = &self.graph.node(node.inputs[1]).ty.shape;
                let m = a_shape[0] as u32;
                let k = a_shape[1] as u32;
                let n = b_shape[1] as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::MatMulRelu,
                    workgroups: [ceil_div(n, 16), ceil_div(m, 16), 1],
                    input_buffers: vec![a, b],
                    output_buffer: out_buf,
                    params: vec![m, k, n, 0],
                });
            }

            Op::FusedMatMulBiasRelu => {
                let a = self.get_buffer(node.inputs[0]);
                let b = self.get_buffer(node.inputs[1]);
                let bias = self.get_buffer(node.inputs[2]);
                let a_shape = &self.graph.node(node.inputs[0]).ty.shape;
                let b_shape = &self.graph.node(node.inputs[1]).ty.shape;
                let m = a_shape[0] as u32;
                let k = a_shape[1] as u32;
                let n = b_shape[1] as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::MatMulBiasRelu,
                    workgroups: [ceil_div(n, 16), ceil_div(m, 16), 1],
                    input_buffers: vec![a, b, bias],
                    output_buffer: out_buf,
                    params: vec![m, k, n, 0],
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
                    workgroups: [ceil_div(len, 256), 1, 1],
                    input_buffers: vec![a, b],
                    output_buffer: out_buf,
                    params: vec![len, bias_len, 0, 0],
                });
            }

            Op::Relu => {
                self.emit_unary(ShaderEntry::Relu, node, out_buf);
            }
            Op::Sigmoid => {
                self.emit_unary(ShaderEntry::Sigmoid, node, out_buf);
            }
            Op::Neg => {
                self.emit_unary(ShaderEntry::Neg, node, out_buf);
            }

            Op::SumAll => {
                let input = self.get_buffer(node.inputs[0]);
                let len = self.graph.node(node.inputs[0]).ty.num_elements() as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::SumAll,
                    workgroups: [ceil_div(len, 256), 1, 1],
                    input_buffers: vec![input],
                    output_buffer: out_buf,
                    params: vec![len, 0, 0, 0],
                });
            }

            Op::MeanAll => {
                let input = self.get_buffer(node.inputs[0]);
                let len = self.graph.node(node.inputs[0]).ty.num_elements() as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::MeanAll,
                    workgroups: [ceil_div(len, 256), 1, 1],
                    input_buffers: vec![input],
                    output_buffer: out_buf,
                    params: vec![len, 0, 0, 0],
                });
            }

            Op::Softmax => {
                let input = self.get_buffer(node.inputs[0]);
                let shape = &self.graph.node(node.inputs[0]).ty.shape;
                let batch = shape[0] as u32;
                let features = shape[1] as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::Softmax,
                    workgroups: [ceil_div(batch, 256), 1, 1],
                    input_buffers: vec![input],
                    output_buffer: out_buf,
                    params: vec![batch, features, 0, 0],
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
                    workgroups: [ceil_div(batch, 256), 1, 1],
                    input_buffers: vec![input],
                    output_buffer: out_buf,
                    params: vec![batch, features, 0, 0],
                });
            }

            Op::CrossEntropyLoss => {
                let logits = self.get_buffer(node.inputs[0]);
                let labels = self.get_buffer(node.inputs[1]);
                let shape = &self.graph.node(node.inputs[0]).ty.shape;
                let batch = shape[0] as u32;
                let features = shape[1] as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::CrossEntropyLoss,
                    workgroups: [1, 1, 1],
                    input_buffers: vec![logits, labels],
                    output_buffer: out_buf,
                    params: vec![batch, features, 0, 0],
                });
            }

            Op::Transpose => {
                let input = self.get_buffer(node.inputs[0]);
                let shape = &self.graph.node(node.inputs[0]).ty.shape;
                let m = shape[0] as u32;
                let n = shape[1] as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::Transpose,
                    workgroups: [ceil_div(n, 16), ceil_div(m, 16), 1],
                    input_buffers: vec![input],
                    output_buffer: out_buf,
                    params: vec![m, n, 0, 0],
                });
            }
        }
    }

    fn emit_unary(&mut self, shader: ShaderEntry, node: &Node, out_buf: BufferRef) {
        let input = self.get_buffer(node.inputs[0]);
        let len = node.ty.num_elements() as u32;
        self.plan.dispatches.push(Dispatch {
            shader,
            workgroups: [ceil_div(len, 256), 1, 1],
            input_buffers: vec![input],
            output_buffer: out_buf,
            params: vec![len, 0, 0, 0],
        });
    }

    fn emit_binary(&mut self, shader: ShaderEntry, node: &Node, out_buf: BufferRef) {
        let a = self.get_buffer(node.inputs[0]);
        let b = self.get_buffer(node.inputs[1]);
        let len = node.ty.num_elements() as u32;
        self.plan.dispatches.push(Dispatch {
            shader,
            workgroups: [ceil_div(len, 256), 1, 1],
            input_buffers: vec![a, b],
            output_buffer: out_buf,
            params: vec![len, 0, 0, 0],
        });
    }
}

fn ceil_div(a: u32, b: u32) -> u32 {
    a.div_ceil(b)
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
        assert_eq!(plan.dispatches.len(), 2); // matmul + relu
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
        // After fusion, matmul+relu → single FusedMatMulRelu dispatch
        assert_eq!(plan.dispatches.len(), 1);
        assert_eq!(plan.dispatches[0].shader, ShaderEntry::MatMulRelu);
    }
}
