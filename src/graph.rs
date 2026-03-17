use std::fmt;

pub type NodeId = u32;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    F32,
}

impl DType {
    pub fn size_bytes(self) -> usize {
        match self {
            DType::F32 => 4,
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
    Parameter { name: String },
    Input { name: String },
    Constant { data: Vec<f32> },

    // Binary
    MatMul,
    Add,
    Mul,

    // Unary
    Relu,
    Sigmoid,
    Neg,

    // Reduction
    SumAll,
    MeanAll,
    Softmax,

    // Loss
    CrossEntropyLoss,

    // Comparison (for autodiff)
    Greater,

    // Fused ops (created by optimizer)
    FusedMatMulRelu,
    FusedMatMulBiasRelu,

    // Transpose (swap last two dims)
    Transpose,

    // Broadcast add (bias add: [M,N] + [N])
    BiasAdd,

    // Dead node (consumed by fusion, skip during compilation)
    Nop,

    // Log-softmax (for numerical stability)
    LogSoftmax,
}

#[derive(Clone, Debug)]
pub struct Node {
    pub id: NodeId,
    pub op: Op,
    pub inputs: Vec<NodeId>,
    pub ty: TensorType,
}

pub struct Graph {
    nodes: Vec<Node>,
    outputs: Vec<NodeId>,
}

impl Graph {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            outputs: Vec::new(),
        }
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
        self.add_node(Op::Input { name: name.to_string() }, vec![], ty)
    }

    pub fn parameter(&mut self, name: &str, shape: &[usize]) -> NodeId {
        let ty = TensorType::f32(shape.to_vec());
        self.add_node(Op::Parameter { name: name.to_string() }, vec![], ty)
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

    pub fn neg(&mut self, x: NodeId) -> NodeId {
        let ty = self.node(x).ty.clone();
        self.add_node(Op::Neg, vec![x], ty)
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

    // --- Loss ---

    pub fn cross_entropy_loss(&mut self, logits: NodeId, labels: NodeId) -> NodeId {
        let l_shape = &self.node(logits).ty.shape;
        let t_shape = &self.node(labels).ty.shape;
        assert_eq!(l_shape, t_shape, "logits and labels must match");
        let ty = TensorType::f32(vec![1]);
        self.add_node(Op::CrossEntropyLoss, vec![logits, labels], ty)
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
}
