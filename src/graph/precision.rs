use super::{DType, Graph, NodeId, Op};
use std::collections::HashMap;

impl Graph {
    /// Explicit mixed precision for ordinary dense products in a forward graph.
    /// Operands and stored results use BF16; accumulators, other operations,
    /// parameters and graph interfaces remain F32. Call before autodiff and
    /// enable `CompileOptions::bf16_matrix_gradients` for reduced derivatives.
    /// Block/batched products, convolution and full-precision regions are not
    /// silently converted. Native BF16 cooperative support is required.
    pub fn into_bf16_matmuls(self) -> Self {
        self.into_bf16_selected(|op| {
            matches!(
                op,
                Op::MatMul
                    | Op::MatMulAT
                    | Op::MatMulBT
                    | Op::FusedMatMulAdd
                    | Op::FusedMatMulATAdd
                    | Op::FusedMatMulBTAdd
            )
        })
    }

    /// Explicit BF16 convolution operands and stored results, with F32
    /// accumulation and derivatives. Parameters and graph interfaces stay F32.
    /// Compose with `into_bf16_matmuls` before differentiation as needed.
    pub fn into_bf16_convolutions(self) -> Self {
        self.into_bf16_selected(|op| matches!(op, Op::Conv2d { .. }))
    }

    fn into_bf16_selected(self, select: impl Fn(&Op) -> bool) -> Self {
        assert_eq!(
            self.num_param_grad_outputs, 0,
            "cast forward graphs before autodiff"
        );
        let sorted = self.into_toposort();
        let mut result = Graph::new();
        let mut remap = vec![0; sorted.nodes.len()];
        let mut casts = HashMap::new();
        result.derived_params = sorted.derived_params;
        for mut node in sorted.nodes {
            let old_id = node.id as usize;
            node.inputs = node
                .inputs
                .into_iter()
                .map(|id| remap[id as usize])
                .collect();
            // Explicit backward primitives may occur in diagnostic graphs.
            match node.op {
                Op::MultiHeadAttnGradQ {
                    ref mut fwd_node, ..
                }
                | Op::MultiHeadAttnGradK {
                    ref mut fwd_node, ..
                }
                | Op::MultiHeadAttnGradV {
                    ref mut fwd_node, ..
                } => *fwd_node = remap[*fwd_node as usize],
                _ => (),
            }
            let cast = !node.requires_full_precision
                && select(&node.op)
                && node.ty.dtype == DType::F32
                && node.inputs[..2]
                    .iter()
                    .all(|&id| result.node(id).ty.dtype == DType::F32);
            if cast {
                for id in &mut node.inputs[..2] {
                    *id = narrow(&mut result, &mut casts, *id);
                }
                // An implementation selected for F32 is not a BF16 kernel pin.
                node.matmul_impl = None;
            }
            node.id = result.nodes.len() as NodeId;
            let mut id = node.id;
            let name = if cast { node.name.take() } else { None };
            result.nodes.push(node);
            if cast {
                let half = result.to_bf16(id);
                id = result.from_bf16(half);
                result.nodes[id as usize].name = name;
            }
            remap[old_id] = id;
        }
        result.outputs = sorted
            .outputs
            .into_iter()
            .map(|id| remap[id as usize])
            .collect();
        result.new_nodes_require_full_precision = sorted.new_nodes_require_full_precision;
        result
    }
}

fn narrow(graph: &mut Graph, casts: &mut HashMap<NodeId, NodeId>, id: NodeId) -> NodeId {
    if matches!(graph.node(id).op, Op::FromBF16) {
        return graph.node(id).inputs[0];
    }
    if let Some(&half) = casts.get(&id) {
        return half;
    }
    let half = if matches!(graph.node(id).op, Op::Identity | Op::StopGradient) {
        let node = graph.node(id).clone();
        let source = narrow(graph, casts, node.inputs[0]);
        let mut ty = node.ty;
        ty.dtype = DType::BF16;
        graph.add_raw_node(node.op, vec![source], ty)
    } else {
        graph.to_bf16(id)
    };
    casts.insert(id, half);
    half
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bf16_convolution_is_not_rewritten_to_f32_winograd() {
        let mut graph = Graph::new();
        let x = graph.input("x", &[64 * 8 * 8]);
        let w = graph.parameter("w", &[64 * 64 * 3 * 3]);
        let y = graph.conv2d(x, w, 1, 64, 8, 8, 64, 3, 3, 1, 1);
        graph.set_outputs(vec![y]);
        let mut bf16 = graph.deep_clone().into_bf16_convolutions();
        let mut fusions = Vec::new();
        crate::optimize::apply_winograd_conv_fusions(&mut bf16, &mut fusions, &Default::default());
        assert!(fusions.is_empty());
        crate::optimize::apply_winograd_conv_fusions(&mut graph, &mut fusions, &Default::default());
        assert_eq!(fusions.len(), 1);
    }

    #[test]
    fn convolution_selection_preserves_strict_regions_and_is_idempotent() {
        let mut graph = Graph::new();
        let x = graph.input("x", &[3 * 5 * 7]);
        let w = graph.parameter("w", &[5 * 3 * 3 * 3]);
        let y = graph.conv2d(x, w, 1, 3, 5, 7, 5, 3, 3, 1, 1);
        graph.begin_full_precision_region();
        let strict = graph.conv2d(x, w, 1, 3, 5, 7, 5, 3, 3, 1, 1);
        graph.set_outputs(vec![y, strict]);
        let selected = graph.into_bf16_convolutions();
        let convs: Vec<_> = selected
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::Conv2d { .. }))
            .collect();
        assert_eq!(convs.len(), 2);
        assert!(
            convs[0]
                .inputs
                .iter()
                .all(|&i| selected.node(i).ty.dtype == DType::BF16)
        );
        assert!(
            convs[1]
                .inputs
                .iter()
                .all(|&i| selected.node(i).ty.dtype == DType::F32)
        );
        assert!(
            selected
                .outputs()
                .iter()
                .all(|&i| selected.node(i).ty.dtype == DType::F32)
        );
        assert_eq!(
            selected.deep_clone().into_bf16_convolutions().nodes().len(),
            selected.nodes().len()
        );
        assert_eq!(
            selected.deep_clone().into_bf16_matmuls().nodes().len(),
            selected.nodes().len()
        );
    }

    #[test]
    fn dense_selection_preserves_masters_interfaces_and_explicit_f32() {
        let mut graph = Graph::new();
        let x = graph.input("x", &[3, 5]);
        let w = graph.parameter("w", &[5, 5]);
        let first = graph.matmul(x, w);
        let second = graph.matmul(first, w);
        graph.begin_full_precision_region();
        let strict = graph.matmul(second, w);
        graph.set_outputs(vec![first, second, strict]);
        let cast = graph.into_bf16_matmuls();
        assert_eq!(
            cast.nodes
                .iter()
                .filter(|n| matches!(n.op, Op::ToBF16))
                .count(),
            4
        );
        for &output in cast.outputs() {
            assert_eq!(cast.node(output).ty.dtype, DType::F32);
        }
        let matrices: Vec<_> = cast
            .nodes
            .iter()
            .filter(|n| matches!(n.op, Op::MatMul))
            .collect();
        assert_eq!(matrices.len(), 3);
        for matrix in &matrices[..2] {
            assert!(
                matrix
                    .inputs
                    .iter()
                    .all(|&id| cast.node(id).ty.dtype == DType::BF16)
            );
        }
        assert!(
            matrices[2]
                .inputs
                .iter()
                .all(|&id| cast.node(id).ty.dtype == DType::F32)
        );
        assert!(
            cast.nodes
                .iter()
                .filter(|n| matches!(n.op, Op::Parameter { .. }))
                .all(|n| n.ty.dtype == DType::F32)
        );
        let same = cast.deep_clone().into_bf16_matmuls();
        assert_eq!(
            same.nodes.len(),
            cast.nodes.len(),
            "conversion must be idempotent"
        );
    }
}
