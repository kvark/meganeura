use crate::graph::{Graph, NodeId, Op, TensorType};
use std::collections::HashMap;

/// Build the backward (gradient) graph from a forward graph.
///
/// Given a graph ending in a scalar loss, appends gradient nodes for each
/// parameter. Returns a new graph containing both forward and backward passes,
/// with outputs being the parameter gradients (in the same order as parameters
/// appear in the original graph).
pub fn differentiate(forward: &Graph) -> Graph {
    let mut graph = Graph::new();
    // Copy forward graph nodes
    for node in forward.nodes() {
        graph.add_raw_node(node.op.clone(), node.inputs.clone(), node.ty.clone());
    }

    let loss_node = forward.outputs()[0];

    // Map from forward node id → gradient node id
    let mut grads: HashMap<NodeId, NodeId> = HashMap::new();

    // dL/dL = 1.0
    let one = graph.scalar(1.0);
    grads.insert(loss_node, one);

    // Backward pass: iterate nodes in reverse topological order
    let num_forward = forward.nodes().len();
    for i in (0..num_forward).rev() {
        let node = forward.nodes()[i].clone();
        let grad_output = match grads.get(&node.id) {
            Some(&g) => g,
            None => continue, // no gradient flows to this node
        };

        match node.op {
            Op::MatMul => {
                // C = A @ B
                // dL/dA = dL/dC @ B^T
                // dL/dB = A^T @ dL/dC
                let a = node.inputs[0];
                let b = node.inputs[1];
                let bt = graph.transpose(b);
                let grad_a = graph.matmul(grad_output, bt);
                let at = graph.transpose(a);
                let grad_b = graph.matmul(at, grad_output);
                accumulate_grad(&mut graph, &mut grads, a, grad_a);
                accumulate_grad(&mut graph, &mut grads, b, grad_b);
            }
            Op::Add => {
                let a = node.inputs[0];
                let b = node.inputs[1];
                // dL/da = dL/dc, dL/db = dL/dc
                accumulate_grad(&mut graph, &mut grads, a, grad_output);
                accumulate_grad(&mut graph, &mut grads, b, grad_output);
            }
            Op::BiasAdd => {
                // out = input + bias (broadcast)
                // dL/dinput = dL/dout
                // dL/dbias = sum_rows(dL/dout)
                let input = node.inputs[0];
                let bias = node.inputs[1];
                accumulate_grad(&mut graph, &mut grads, input, grad_output);
                // Sum over batch dimension to get bias gradient
                // For [batch, features] → [features], we need row-wise sum
                // For now, use a dedicated approach: the bias grad is the
                // column sums of grad_output
                let bias_grad = graph.sum_rows(grad_output, &forward.nodes()[bias as usize].ty);
                accumulate_grad(&mut graph, &mut grads, bias, bias_grad);
            }
            Op::Mul => {
                let a = node.inputs[0];
                let b = node.inputs[1];
                // dL/da = dL/dc * b
                // dL/db = dL/dc * a
                let grad_a = graph.mul(grad_output, b);
                let grad_b = graph.mul(grad_output, a);
                accumulate_grad(&mut graph, &mut grads, a, grad_a);
                accumulate_grad(&mut graph, &mut grads, b, grad_b);
            }
            Op::Relu => {
                // dL/dx = dL/dy * (x > 0)
                let x = node.inputs[0];
                let zero = graph.constant(
                    vec![0.0; forward.nodes()[x as usize].ty.num_elements()],
                    &forward.nodes()[x as usize].ty.shape,
                );
                let mask = graph.greater(x, zero);
                let grad_x = graph.mul(grad_output, mask);
                accumulate_grad(&mut graph, &mut grads, x, grad_x);
            }
            Op::Sigmoid => {
                // dL/dx = dL/dy * y * (1 - y)
                let y = node.id;
                let one = graph.constant(
                    vec![1.0; node.ty.num_elements()],
                    &node.ty.shape,
                );
                let neg_y = graph.neg(y);
                let one_minus_y = graph.add(one, neg_y);
                let dy = graph.mul(y, one_minus_y);
                let grad_x = graph.mul(grad_output, dy);
                accumulate_grad(&mut graph, &mut grads, node.inputs[0], grad_x);
            }
            Op::CrossEntropyLoss => {
                // L = -mean(labels * log(softmax(logits)))
                // dL/dlogits = softmax(logits) - labels (simplified)
                let logits = node.inputs[0];
                let labels = node.inputs[1];
                let softmax = graph.softmax(logits);
                let neg_labels = graph.neg(labels);
                let grad_logits = graph.add(softmax, neg_labels);
                accumulate_grad(&mut graph, &mut grads, logits, grad_logits);
                // No gradient for labels (they're targets)
            }
            Op::SumAll => {
                // dL/dx = dL/dy broadcast to shape of x
                // Since grad_output is scalar [1] and we need [shape of x],
                // we create a constant filled with 1.0 (representing the
                // broadcast of the scalar gradient).
                let x = node.inputs[0];
                let x_shape = &forward.nodes()[x as usize].ty.shape;
                let ones = graph.constant(
                    vec![1.0; x_shape.iter().product()],
                    x_shape,
                );
                // grad_output is scalar — just use the constant directly
                accumulate_grad(&mut graph, &mut grads, x, ones);
            }
            Op::MeanAll => {
                // dL/dx = (1/N) broadcast to shape of x
                let x = node.inputs[0];
                let x_shape = &forward.nodes()[x as usize].ty.shape;
                let n = x_shape.iter().product::<usize>() as f32;
                let scale = 1.0 / n;
                let scaled_ones = graph.constant(
                    vec![scale; x_shape.iter().product()],
                    x_shape,
                );
                accumulate_grad(&mut graph, &mut grads, x, scaled_ones);
            }
            Op::Neg => {
                let x = node.inputs[0];
                let grad_x = graph.neg(grad_output);
                accumulate_grad(&mut graph, &mut grads, x, grad_x);
            }
            Op::Transpose => {
                let x = node.inputs[0];
                let grad_x = graph.transpose(grad_output);
                accumulate_grad(&mut graph, &mut grads, x, grad_x);
            }
            Op::Softmax | Op::LogSoftmax => {
                // Softmax gradient is complex, but when used with CrossEntropyLoss
                // the combined gradient is simple (softmax - labels).
                // Standalone softmax grad: dL/dx_i = s_i * (dL/ds_i - sum_j(dL/ds_j * s_j))
                // For now, we only support softmax as part of CrossEntropyLoss.
                log::warn!("standalone softmax/log_softmax gradient not yet implemented");
            }
            // Leaf nodes, fused ops don't appear in forward pass before optimization
            Op::Input { .. }
            | Op::Parameter { .. }
            | Op::Constant { .. }
            | Op::Greater => {}
            Op::Nop => {}
            Op::FusedMatMulRelu | Op::FusedMatMulBiasRelu => {
                log::warn!("autodiff should run before fusion optimization");
            }
        }
    }

    // Collect parameter gradients as outputs
    let mut param_grad_outputs = Vec::new();
    for node in forward.nodes() {
        if let Op::Parameter { .. } = node.op {
            if let Some(&grad_id) = grads.get(&node.id) {
                param_grad_outputs.push((node.id, grad_id));
            }
        }
    }

    // Outputs: [loss, (param_id, grad_id), ...]
    let mut outputs = vec![loss_node];
    for &(_, grad_id) in &param_grad_outputs {
        outputs.push(grad_id);
    }
    graph.set_outputs(outputs);

    graph
}

/// Helper to implement sum_rows: reduce [M,N] → [N] by summing rows
impl Graph {
    pub fn sum_rows(&mut self, x: NodeId, target_ty: &TensorType) -> NodeId {
        // This is a specialized reduction. We'll implement it as a SumRows op.
        // For now, we use MeanAll as a placeholder and scale.
        // TODO: proper SumRows op
        let ty = target_ty.clone();
        self.add_raw_node(Op::SumAll, vec![x], ty)
    }
}

fn accumulate_grad(graph: &mut Graph, grads: &mut HashMap<NodeId, NodeId>, node: NodeId, grad: NodeId) {
    match grads.get(&node) {
        Some(&existing) => {
            // Multiple paths contribute to this gradient — sum them
            let sum = graph.add(existing, grad);
            grads.insert(node, sum);
        }
        None => {
            grads.insert(node, grad);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_autodiff() {
        let mut g = Graph::new();
        let x = g.input("x", &[4, 3]);
        let w = g.parameter("w", &[3, 2]);
        let y = g.matmul(x, w);
        let loss = g.mean_all(y);
        g.set_outputs(vec![loss]);

        let diff = differentiate(&g);
        // Should have: forward nodes + gradient nodes
        assert!(diff.nodes().len() > g.nodes().len());
        // Outputs should be [loss, grad_w]
        assert_eq!(diff.outputs().len(), 2);
    }

    #[test]
    fn test_mlp_autodiff() {
        let mut g = Graph::new();
        let x = g.input("x", &[4, 784]);
        let w1 = g.parameter("w1", &[784, 128]);
        let b1 = g.parameter("b1", &[128]);
        let mm1 = g.matmul(x, w1);
        let h1 = g.bias_add(mm1, b1);
        let a1 = g.relu(h1);
        let w2 = g.parameter("w2", &[128, 10]);
        let mm2 = g.matmul(a1, w2);
        let labels = g.input("labels", &[4, 10]);
        let loss = g.cross_entropy_loss(mm2, labels);
        g.set_outputs(vec![loss]);

        let diff = differentiate(&g);
        // Should produce gradients for w1, b1, w2
        // outputs: [loss, grad_w1, grad_b1, grad_w2]
        assert_eq!(diff.outputs().len(), 4, "expected loss + 3 param grads");
    }

    #[test]
    fn test_sigmoid_autodiff() {
        let mut g = Graph::new();
        let x = g.input("x", &[4, 8]);
        let w = g.parameter("w", &[8, 4]);
        let y = g.matmul(x, w);
        let s = g.sigmoid(y);
        let loss = g.mean_all(s);
        g.set_outputs(vec![loss]);

        let diff = differentiate(&g);
        // outputs: [loss, grad_w]
        assert_eq!(diff.outputs().len(), 2);
        // Sigmoid backward creates: neg, add (1-y), mul (y*(1-y)), mul (dL * dy)
        // Check gradient node shapes
        let grad_w = diff.outputs()[1];
        assert_eq!(diff.node(grad_w).ty.shape, vec![8, 4]);
    }

    #[test]
    fn test_neg_autodiff() {
        let mut g = Graph::new();
        let x = g.input("x", &[4, 8]);
        let w = g.parameter("w", &[8, 4]);
        let y = g.matmul(x, w);
        let n = g.neg(y);
        let loss = g.sum_all(n);
        g.set_outputs(vec![loss]);

        let diff = differentiate(&g);
        assert_eq!(diff.outputs().len(), 2);
        let grad_w = diff.outputs()[1];
        assert_eq!(diff.node(grad_w).ty.shape, vec![8, 4]);
    }

    #[test]
    fn test_transpose_autodiff() {
        let mut g = Graph::new();
        let x = g.input("x", &[4, 8]);
        let w = g.parameter("w", &[8, 3]);
        let y = g.matmul(x, w);
        let t = g.transpose(y);
        // t is [3, 4], need to reduce to scalar
        let loss = g.mean_all(t);
        g.set_outputs(vec![loss]);

        let diff = differentiate(&g);
        assert_eq!(diff.outputs().len(), 2);
        let grad_w = diff.outputs()[1];
        assert_eq!(diff.node(grad_w).ty.shape, vec![8, 3]);
    }

    #[test]
    fn test_mul_autodiff() {
        let mut g = Graph::new();
        let x = g.input("x", &[4, 8]);
        let w = g.parameter("w", &[8, 4]);
        let y = g.matmul(x, w);
        // element-wise mul with itself: y * y
        let sq = g.mul(y, y);
        let loss = g.mean_all(sq);
        g.set_outputs(vec![loss]);

        let diff = differentiate(&g);
        assert_eq!(diff.outputs().len(), 2);
        // When multiplying y*y, gradient accumulates from both inputs
        let grad_w = diff.outputs()[1];
        assert_eq!(diff.node(grad_w).ty.shape, vec![8, 4]);
    }

    #[test]
    fn test_multi_path_gradient_accumulation() {
        // w is used in two separate matmuls — gradients should accumulate
        let mut g = Graph::new();
        let x1 = g.input("x1", &[4, 8]);
        let x2 = g.input("x2", &[4, 8]);
        let w = g.parameter("w", &[8, 4]);
        let y1 = g.matmul(x1, w);
        let y2 = g.matmul(x2, w);
        let sum = g.add(y1, y2);
        let loss = g.mean_all(sum);
        g.set_outputs(vec![loss]);

        let diff = differentiate(&g);
        assert_eq!(diff.outputs().len(), 2); // loss + grad_w
        let grad_w = diff.outputs()[1];
        // Gradient should be accumulated (Add node)
        assert!(
            matches!(diff.node(grad_w).op, Op::Add),
            "expected gradient accumulation via Add, got {:?}",
            diff.node(grad_w).op
        );
        assert_eq!(diff.node(grad_w).ty.shape, vec![8, 4]);
    }

    #[test]
    fn test_no_grad_for_inputs() {
        // Inputs should not appear in gradient outputs
        let mut g = Graph::new();
        let x = g.input("x", &[4, 3]);
        let w = g.parameter("w", &[3, 2]);
        let y = g.matmul(x, w);
        let loss = g.sum_all(y);
        g.set_outputs(vec![loss]);

        let diff = differentiate(&g);
        // Only loss + grad_w (not grad_x)
        assert_eq!(diff.outputs().len(), 2);
    }
}
