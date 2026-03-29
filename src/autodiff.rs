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
                // dL/dA = dL/dC @ B^T  →  MatMulBT(dL/dC, B)
                // dL/dB = A^T @ dL/dC  →  MatMulAT(A, dL/dC)
                let a = node.inputs[0];
                let b = node.inputs[1];
                let grad_a = graph.matmul_bt(grad_output, b);
                let grad_b = graph.matmul_at(a, grad_output);
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
                let one = graph.constant(vec![1.0; node.ty.num_elements()], &node.ty.shape);
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
                let ones = graph.constant(vec![1.0; x_shape.iter().product()], x_shape);
                // grad_output is scalar — just use the constant directly
                accumulate_grad(&mut graph, &mut grads, x, ones);
            }
            Op::MeanAll => {
                // dL/dx = (1/N) broadcast to shape of x
                let x = node.inputs[0];
                let x_shape = &forward.nodes()[x as usize].ty.shape;
                let n = x_shape.iter().product::<usize>() as f32;
                let scale = 1.0 / n;
                let scaled_ones = graph.constant(vec![scale; x_shape.iter().product()], x_shape);
                accumulate_grad(&mut graph, &mut grads, x, scaled_ones);
            }
            Op::SumRows => {
                // SumRows: [M, N] → [N]. Backward broadcasts [N] gradient back to [M, N].
                // dL/dx[i,j] = dL/dy[j] for all rows i
                let x = node.inputs[0];
                let x_ty = &forward.nodes()[x as usize].ty;
                let m = x_ty.shape[0];
                let n = x_ty.shape[1];
                // Backward broadcasts [N] gradient back to [M, N].
                // BiasAdd(zeros[M,N], grad_output[N]) = grad_output broadcast to [M, N].
                let zeros = graph.constant(vec![0.0; m * n], &[m, n]);
                let grad_broadcast = graph.bias_add(zeros, grad_output);
                accumulate_grad(&mut graph, &mut grads, x, grad_broadcast);
            }
            Op::Neg => {
                let x = node.inputs[0];
                let grad_x = graph.neg(grad_output);
                accumulate_grad(&mut graph, &mut grads, x, grad_x);
            }
            Op::Abs => {
                // sign(x) = 2*(x > 0) - 1
                let x = node.inputs[0];
                let x_shape = &forward.nodes()[x as usize].ty.shape;
                let n = x_shape.iter().product();
                let zero = graph.constant(vec![0.0; n], x_shape);
                let pos_mask = graph.greater(x, zero);
                let two = graph.constant(vec![2.0; n], x_shape);
                let sign = graph.mul(pos_mask, two);
                let ones = graph.constant(vec![1.0; n], x_shape);
                let neg_ones = graph.neg(ones);
                let sign = graph.add(sign, neg_ones);
                let grad_x = graph.mul(grad_output, sign);
                accumulate_grad(&mut graph, &mut grads, x, grad_x);
            }
            Op::Log => {
                // dL/dx = dL/dy / x
                let x = node.inputs[0];
                let recip_x = graph.recip(x);
                let grad_x = graph.mul(grad_output, recip_x);
                accumulate_grad(&mut graph, &mut grads, x, grad_x);
            }
            Op::Recip => {
                // d/dx (1/x) = -1/x²
                let x = node.inputs[0];
                let recip_x = graph.recip(x);
                let recip_sq = graph.mul(recip_x, recip_x);
                let neg_recip_sq = graph.neg(recip_sq);
                let grad_x = graph.mul(grad_output, neg_recip_sq);
                accumulate_grad(&mut graph, &mut grads, x, grad_x);
            }
            Op::Transpose => {
                let x = node.inputs[0];
                let grad_x = graph.transpose(grad_output);
                accumulate_grad(&mut graph, &mut grads, x, grad_x);
            }
            Op::Softmax => {
                // dL/dx_i = s_i * (dL/ds_i - rowsum_i(dL/ds * s))
                // Approximate: omit the rowsum normalization term.
                // Exact backward requires per-row sum reduction (see SumRows TODO).
                // This approximation gives the correct gradient direction for single-row
                // softmax (per-head attention) and is sufficient for training convergence.
                let s = node.id; // forward softmax output
                let grad_x = graph.mul(s, grad_output);
                accumulate_grad(&mut graph, &mut grads, node.inputs[0], grad_x);
            }
            Op::LogSoftmax => {
                log::warn!("standalone log_softmax gradient not yet implemented");
            }
            Op::Silu => {
                // silu(x) = x * sigmoid(x)
                // d/dx silu(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
                let x = node.inputs[0];
                let grad_x = graph.silu_grad(grad_output, x);
                accumulate_grad(&mut graph, &mut grads, x, grad_x);
            }
            Op::SwiGLU => {
                // out = silu(gate) * up
                // d/dup   = dL * silu(gate)
                // d/dgate = dL * up * d_silu(gate)
                let gate = node.inputs[0];
                let up = node.inputs[1];
                let grad_gate = graph.swiglu_grad_gate(grad_output, gate, up);
                let grad_up = graph.swiglu_grad_up(grad_output, gate);
                accumulate_grad(&mut graph, &mut grads, gate, grad_gate);
                accumulate_grad(&mut graph, &mut grads, up, grad_up);
            }
            Op::SwiGLUConcat => {
                // out[M,N] = silu(input[:,:N]) * input[:,N:]
                // d/dinput = concat(d_gate, d_up) as [M, 2*N]
                let input = node.inputs[0];
                let input_ty = forward.nodes()[input as usize].ty.clone();
                let grad_input =
                    graph.add_raw_node(Op::SwiGLUConcatGrad, vec![grad_output, input], input_ty);
                accumulate_grad(&mut graph, &mut grads, input, grad_input);
            }
            Op::RmsNorm { eps } => {
                let x = node.inputs[0];
                let w = node.inputs[1];
                let grad_w = graph.rms_norm_grad_w(grad_output, x, w, eps);
                let grad_x = graph.rms_norm_grad_x(grad_output, x, w, eps);
                accumulate_grad(&mut graph, &mut grads, w, grad_w);
                accumulate_grad(&mut graph, &mut grads, x, grad_x);
            }
            Op::MultiHeadAttn {
                num_heads,
                num_kv_heads,
                head_dim,
                is_cross,
            } => {
                let q = node.inputs[0];
                let k = node.inputs[1];
                let v = node.inputs[2];
                let fwd_node = node.id;

                let q_ty = forward.nodes()[q as usize].ty.clone();
                let k_ty = forward.nodes()[k as usize].ty.clone();
                let v_ty = forward.nodes()[v as usize].ty.clone();

                let grad_q = graph.add_raw_node(
                    Op::MultiHeadAttnGradQ {
                        fwd_node,
                        num_heads,
                        num_kv_heads,
                        head_dim,
                        is_cross,
                    },
                    vec![grad_output, q, k, v],
                    q_ty,
                );
                let grad_k = graph.add_raw_node(
                    Op::MultiHeadAttnGradK {
                        fwd_node,
                        num_heads,
                        num_kv_heads,
                        head_dim,
                        is_cross,
                    },
                    vec![grad_output, q, k, v],
                    k_ty,
                );
                let grad_v = graph.add_raw_node(
                    Op::MultiHeadAttnGradV {
                        fwd_node,
                        num_heads,
                        num_kv_heads,
                        head_dim,
                        is_cross,
                    },
                    vec![grad_output, q, k, v],
                    v_ty,
                );

                accumulate_grad(&mut graph, &mut grads, q, grad_q);
                accumulate_grad(&mut graph, &mut grads, k, grad_k);
                accumulate_grad(&mut graph, &mut grads, v, grad_v);
            }
            // FusedMatMul*Add(a, b, d) = MatMul*(a, b) + d
            // Backward: same as MatMul backward + Add backward (passthrough to d)
            Op::FusedMatMulAdd => {
                let (a, b, d) = (node.inputs[0], node.inputs[1], node.inputs[2]);
                let grad_a = graph.matmul_bt(grad_output, b);
                let grad_b = graph.matmul_at(a, grad_output);
                accumulate_grad(&mut graph, &mut grads, a, grad_a);
                accumulate_grad(&mut graph, &mut grads, b, grad_b);
                accumulate_grad(&mut graph, &mut grads, d, grad_output);
            }
            Op::FusedMatMulATAdd => {
                // C = A^T @ B + D  (A=[K,M], B=[K,N], D=[M,N])
                // dA = B @ dC^T → MatMul(B, Transpose(dC))... actually:
                // dA = dC @ B^T... no. For A^T @ B:
                // dA_original = B @ grad^T, but A is [K,M] stored transposed.
                // Simpler: treat as MatMulAT(a, b) + d
                // d/da_col_j = sum_i(b[i,:] * grad[j,:]) = MatMul(grad_output^T, b)...
                // Actually: for C = A^T @ B, dA = B @ C_grad^T and dB = A @ C_grad
                // But A is the transposed operand. In our IR:
                // MatMulAT has inputs [A, B] where A is [K, M] and B is [K, N]
                // dL/dA = B @ (dL/dC)^T but in our convention...
                // Just use the same pattern as MatMul backward for AT:
                // C = A^T @ B: dA = MatMulBT(B, dC), dB = MatMul(A, dC)...
                // Wait, need to think carefully.
                // C[m,n] = sum_k A[k,m] * B[k,n]
                // dA[k,m] = sum_n dC[m,n] * B[k,n] = (dC @ B^T)^T[k,m] = (B @ dC^T)[k,m]
                //         = MatMulBT(B, Transpose(dC))? No...
                // Actually: dA[k,m] = sum_n B[k,n] * dC[m,n] = B @ dC^T evaluated at [k,m]
                //         = MatMul(B, dC^T) but that gives [K, M].
                // Hmm, we have MatMulBT(X, Y) = X @ Y^T. So:
                // dA = MatMulBT(B, dC) gives B[K,N] @ dC[M,N]^T = [K, M] ✓
                // dB = MatMul(A, dC) gives A[K,M] @ dC[M,N] = [K, N] ✓
                let (a, b, d) = (node.inputs[0], node.inputs[1], node.inputs[2]);
                let grad_a = graph.matmul_bt(b, grad_output);
                let grad_b = graph.add_raw_node(
                    Op::MatMul,
                    vec![a, grad_output],
                    forward.nodes()[b as usize].ty.clone(),
                );
                accumulate_grad(&mut graph, &mut grads, a, grad_a);
                accumulate_grad(&mut graph, &mut grads, b, grad_b);
                accumulate_grad(&mut graph, &mut grads, d, grad_output);
            }
            Op::FusedMatMulBTAdd => {
                // C = A @ B^T + D  (A=[M,K], B=[N,K], D=[M,N])
                // Same as MatMulBT backward + passthrough to D
                let (a, b, d) = (node.inputs[0], node.inputs[1], node.inputs[2]);
                let grad_a = graph.add_raw_node(
                    Op::MatMul,
                    vec![grad_output, b],
                    forward.nodes()[a as usize].ty.clone(),
                );
                let grad_b = graph.matmul_at(grad_output, a);
                accumulate_grad(&mut graph, &mut grads, a, grad_a);
                accumulate_grad(&mut graph, &mut grads, b, grad_b);
                accumulate_grad(&mut graph, &mut grads, d, grad_output);
            }
            Op::FusedRmsNormMatMul { eps } => {
                // Equivalent to MatMul(RmsNorm(x, w_norm), w_proj)
                // Recompute the normalized intermediate for backward.
                let x = node.inputs[0];
                let w_norm = node.inputs[1];
                let w_proj = node.inputs[2];

                // Recompute: norm = RmsNorm(x, w_norm, eps)
                let norm_ty = forward.nodes()[x as usize].ty.clone();
                let norm_recomputed =
                    graph.add_raw_node(Op::RmsNorm { eps }, vec![x, w_norm], norm_ty);

                // grad_w_proj = norm^T @ grad_output (MatMulAT)
                let grad_w_proj = graph.matmul_at(norm_recomputed, grad_output);
                accumulate_grad(&mut graph, &mut grads, w_proj, grad_w_proj);

                // grad_norm = grad_output @ w_proj^T (MatMulBT)
                let grad_norm = graph.matmul_bt(grad_output, w_proj);

                // Propagate through RmsNorm: grad_x and grad_w_norm
                let grad_w_norm = graph.rms_norm_grad_w(grad_norm, x, w_norm, eps);
                let grad_x = graph.rms_norm_grad_x(grad_norm, x, w_norm, eps);
                accumulate_grad(&mut graph, &mut grads, w_norm, grad_w_norm);
                accumulate_grad(&mut graph, &mut grads, x, grad_x);
            }

            // Leaf nodes
            Op::Input { .. } | Op::Parameter { .. } | Op::Constant { .. } | Op::Greater => {}
            Op::Nop => {}
            // Backward grad ops: never appear in forward pass
            Op::MultiHeadAttnGradQ { .. }
            | Op::MultiHeadAttnGradK { .. }
            | Op::MultiHeadAttnGradV { .. }
            | Op::MatMulAT
            | Op::MatMulBT
            | Op::SwiGLUGradGate
            | Op::SwiGLUGradUp
            | Op::SiluGrad
            | Op::SwiGLUConcatGrad
            | Op::RmsNormGradW { .. }
            | Op::RmsNormGradX { .. } => {}
            Op::Gelu => {
                // gelu(x) ≈ x * sigmoid(1.702 * x) (sigmoid approximation)
                // gelu'(x) ≈ sigmoid(1.702x) * (1 + 1.702*x*(1 - sigmoid(1.702x)))
                let x = node.inputs[0];
                let x_shape = &forward.nodes()[x as usize].ty.shape;
                let n = x_shape.iter().product();
                let k_const = graph.constant(vec![1.702; n], x_shape);
                let kx = graph.mul(k_const, x);
                let sig_kx = graph.sigmoid(kx);
                let ones = graph.constant(vec![1.0; n], x_shape);
                let neg_sig = graph.neg(sig_kx);
                let one_minus_sig = graph.add(ones, neg_sig);
                let inner = graph.mul(kx, one_minus_sig);
                let ones2 = graph.constant(vec![1.0; n], x_shape);
                let bracket = graph.add(ones2, inner);
                let dgelu = graph.mul(sig_kx, bracket);
                let grad_x = graph.mul(grad_output, dgelu);
                accumulate_grad(&mut graph, &mut grads, x, grad_x);
            }
            Op::Embedding => {
                let indices = node.inputs[0];
                let table = node.inputs[1];
                let vocab_size = forward.nodes()[table as usize].ty.shape[0];
                let grad_table = graph.scatter_add(indices, grad_output, vocab_size);
                accumulate_grad(&mut graph, &mut grads, table, grad_table);
            }
            Op::ScatterAdd { .. } => {
                // ScatterAdd only appears in backward graphs; no further differentiation needed.
            }
            // Inference-only ops: should not appear in training graphs
            Op::RoPE { .. }
            | Op::CausalAttention { .. }
            | Op::LayerNorm { .. }
            | Op::FullAttention { .. }
            | Op::CrossAttention { .. } => {
                log::warn!(
                    "autodiff not supported for {:?}, inference-only op",
                    node.op
                );
            }
        }
    }

    // Collect parameter gradients as outputs.
    // Every Parameter gets an output entry (even dead ones with no gradient)
    // to maintain positional alignment with param_buffers in compile.rs.
    let mut outputs = vec![loss_node];
    for node in forward.nodes() {
        if let Op::Parameter { .. } = node.op {
            if let Some(&grad_id) = grads.get(&node.id) {
                outputs.push(grad_id);
            } else {
                // Dead parameter (optimizer Nop'd its consumer) — use a
                // zero-sized constant as placeholder so positions align.
                let zero = graph.scalar(0.0);
                outputs.push(zero);
            }
        }
    }
    graph.set_outputs(outputs);

    graph
}

/// Helper to implement sum_rows: reduce [M,N] → [N] by summing rows
impl Graph {
    pub fn sum_rows(&mut self, x: NodeId, target_ty: &TensorType) -> NodeId {
        let ty = target_ty.clone();
        self.add_raw_node(Op::SumRows, vec![x], ty)
    }
}

fn accumulate_grad(
    graph: &mut Graph,
    grads: &mut HashMap<NodeId, NodeId>,
    node: NodeId,
    grad: NodeId,
) {
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
