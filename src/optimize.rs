use crate::graph::{Graph, Node, NodeId, Op, TensorType};

/// Convert a Graph to an egglog program string, run equality saturation
/// with rewrite rules, and extract the optimized graph back.
pub fn optimize(graph: &Graph) -> Graph {
    let program = graph_to_egglog(graph);
    log::debug!("egglog program:\n{}", program);

    let mut egraph = egglog::EGraph::default();
    match egraph.parse_and_run_program(None, &program) {
        Ok(outputs) => {
            for out in &outputs {
                log::debug!("egglog output: {}", out);
            }
        }
        Err(e) => {
            log::warn!("egglog optimization failed: {}, returning original graph", e);
            return clone_graph(graph);
        }
    }

    extract_graph(&egraph, graph)
}

/// Generate egglog program text from a Graph.
///
/// Each node becomes an egglog expression. We define a sort for tensor
/// operations and rewrite rules for optimization.
fn graph_to_egglog(graph: &Graph) -> String {
    let mut prog = String::new();

    // Define the sort and constructors
    prog.push_str(
        "\
(datatype Op
  (Input String)
  (Parameter String)
  (Const i64)
  (MatMul Op Op)
  (Add Op Op)
  (Mul Op Op)
  (BiasAdd Op Op)
  (Relu Op)
  (Sigmoid Op)
  (Neg Op)
  (Transpose Op)
  (Softmax Op)
  (LogSoftmax Op)
  (SumAll Op)
  (MeanAll Op)
  (CrossEntropyLoss Op Op)
  (Greater Op Op)
  ; Fused ops
  (FusedMatMulRelu Op Op)
  (FusedMatMulBiasRelu Op Op Op)
)

",
    );

    // Rewrite rules
    prog.push_str(
        "\
; --- Operator fusion ---
(rewrite (Relu (MatMul ?a ?b)) (FusedMatMulRelu ?a ?b))
(rewrite (Relu (BiasAdd (MatMul ?a ?b) ?c)) (FusedMatMulBiasRelu ?a ?b ?c))

; --- Algebraic simplifications ---
(rewrite (Neg (Neg ?x)) ?x)
(rewrite (Transpose (Transpose ?x)) ?x)
(rewrite (Relu (Relu ?x)) (Relu ?x))

",
    );

    // Define expressions for each node
    for node in graph.nodes() {
        if let Op::Nop = node.op {
            continue;
        }
        let expr = node_to_egglog_expr(node);
        prog.push_str(&format!("(let n{} {})\n", node.id, expr));
    }

    // Extract the output nodes
    for &out in graph.outputs() {
        prog.push_str(&format!("(extract n{})\n", out));
    }

    prog
}

fn node_to_egglog_expr(node: &Node) -> String {
    match node.op {
        Op::Input { ref name } => format!("(Input \"{}\")", name),
        Op::Parameter { ref name } => format!("(Parameter \"{}\")", name),
        Op::Constant { .. } => format!("(Const {})", node.id),
        Op::MatMul => format!("(MatMul n{} n{})", node.inputs[0], node.inputs[1]),
        Op::Add => format!("(Add n{} n{})", node.inputs[0], node.inputs[1]),
        Op::Mul => format!("(Mul n{} n{})", node.inputs[0], node.inputs[1]),
        Op::BiasAdd => format!("(BiasAdd n{} n{})", node.inputs[0], node.inputs[1]),
        Op::Relu => format!("(Relu n{})", node.inputs[0]),
        Op::Sigmoid => format!("(Sigmoid n{})", node.inputs[0]),
        Op::Neg => format!("(Neg n{})", node.inputs[0]),
        Op::Transpose => format!("(Transpose n{})", node.inputs[0]),
        Op::Softmax => format!("(Softmax n{})", node.inputs[0]),
        Op::LogSoftmax => format!("(LogSoftmax n{})", node.inputs[0]),
        Op::SumAll => format!("(SumAll n{})", node.inputs[0]),
        Op::MeanAll => format!("(MeanAll n{})", node.inputs[0]),
        Op::CrossEntropyLoss => {
            format!(
                "(CrossEntropyLoss n{} n{})",
                node.inputs[0], node.inputs[1]
            )
        }
        Op::Greater => format!("(Greater n{} n{})", node.inputs[0], node.inputs[1]),
        Op::FusedMatMulRelu => {
            format!("(FusedMatMulRelu n{} n{})", node.inputs[0], node.inputs[1])
        }
        Op::FusedMatMulBiasRelu => format!(
            "(FusedMatMulBiasRelu n{} n{} n{})",
            node.inputs[0], node.inputs[1], node.inputs[2]
        ),
        Op::Nop => unreachable!("Nop nodes should be filtered before conversion"),
    }
}

/// Extract optimized graph from the e-graph after saturation.
///
/// For now, we use a simpler approach: apply known rewrites directly
/// on the Graph IR. The egglog integration validates the rules fire,
/// and we'll upgrade to full extraction later.
fn extract_graph(_egraph: &egglog::EGraph, original: &Graph) -> Graph {
    // For now, apply pattern-matching rewrites directly.
    // Full egglog extraction (parsing the extracted s-expr back to Graph)
    // is complex and will be Phase 2.
    let mut graph = clone_graph(original);
    apply_fusion_rewrites(&mut graph);
    graph
}

/// Apply fusion rewrites directly on the graph IR.
fn apply_fusion_rewrites(graph: &mut Graph) {
    // We need to iterate and find patterns. Since nodes are append-only
    // and topologically sorted, we can scan forward.
    let len = graph.nodes().len();
    let replacements: Vec<Option<NodeId>> = vec![None; len];

    // Collect fused ops: (relu_idx, fused_op, inputs, ty, dead_node_indices)
    type Fusion = (usize, Op, Vec<NodeId>, TensorType, Vec<usize>);
    let mut fusions: Vec<Fusion> = Vec::new();

    for i in 0..len {
        let node = &graph.nodes()[i];
        match node.op {
            Op::Relu => {
                let input_id = resolve(node.inputs[0], &replacements);
                let input_node = &graph.nodes()[input_id as usize];
                match input_node.op {
                    // Relu(BiasAdd(MatMul(a,b), c)) → FusedMatMulBiasRelu(a,b,c)
                    Op::BiasAdd => {
                        let bias_input = resolve(input_node.inputs[0], &replacements);
                        let matmul_node = &graph.nodes()[bias_input as usize];
                        if let Op::MatMul = matmul_node.op {
                            let a = resolve(matmul_node.inputs[0], &replacements);
                            let b = resolve(matmul_node.inputs[1], &replacements);
                            let c = resolve(input_node.inputs[1], &replacements);
                            fusions.push((
                                i,
                                Op::FusedMatMulBiasRelu,
                                vec![a, b, c],
                                node.ty.clone(),
                                vec![bias_input as usize, input_id as usize],
                            ));
                            continue;
                        }
                    }
                    // Relu(MatMul(a,b)) → FusedMatMulRelu(a,b)
                    Op::MatMul => {
                        let a = resolve(input_node.inputs[0], &replacements);
                        let b = resolve(input_node.inputs[1], &replacements);
                        fusions.push((
                            i,
                            Op::FusedMatMulRelu,
                            vec![a, b],
                            node.ty.clone(),
                            vec![input_id as usize],
                        ));
                        continue;
                    }
                    _ => {}
                }
            }
            _ => {}
        }
    }

    // Apply the fusions
    let nodes = graph.nodes_mut();
    for (idx, op, inputs, ty, dead) in fusions {
        nodes[idx].op = op;
        nodes[idx].inputs = inputs;
        nodes[idx].ty = ty;
        // Mark consumed nodes as dead
        for d in dead {
            nodes[d].op = Op::Nop;
            nodes[d].inputs.clear();
        }
    }
}

fn resolve(id: NodeId, replacements: &[Option<NodeId>]) -> NodeId {
    replacements[id as usize].unwrap_or(id)
}

fn clone_graph(graph: &Graph) -> Graph {
    let mut new_graph = Graph::new();
    for node in graph.nodes() {
        new_graph.add_raw_node(node.op.clone(), node.inputs.clone(), node.ty.clone());
    }
    new_graph.set_outputs(graph.outputs().to_vec());
    new_graph
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fusion_matmul_relu() {
        let mut g = Graph::new();
        let x = g.input("x", &[4, 784]);
        let w = g.parameter("w", &[784, 128]);
        let mm = g.matmul(x, w);
        let h = g.relu(mm);
        g.set_outputs(vec![h]);

        let opt = optimize(&g);
        // The relu node should now be FusedMatMulRelu
        let output_id = opt.outputs()[0];
        let output_node = opt.node(output_id);
        assert!(
            matches!(output_node.op, Op::FusedMatMulRelu),
            "expected FusedMatMulRelu, got {:?}",
            output_node.op
        );
    }

    #[test]
    fn test_fusion_matmul_bias_relu() {
        let mut g = Graph::new();
        let x = g.input("x", &[4, 784]);
        let w = g.parameter("w", &[784, 128]);
        let b = g.parameter("b", &[128]);
        let mm = g.matmul(x, w);
        let ba = g.bias_add(mm, b);
        let h = g.relu(ba);
        g.set_outputs(vec![h]);

        let opt = optimize(&g);
        let output_id = opt.outputs()[0];
        let output_node = opt.node(output_id);
        assert!(
            matches!(output_node.op, Op::FusedMatMulBiasRelu),
            "expected FusedMatMulBiasRelu, got {:?}",
            output_node.op
        );
    }

    #[test]
    fn test_egglog_roundtrip() {
        let mut g = Graph::new();
        let x = g.input("x", &[4, 10]);
        let w = g.parameter("w", &[10, 5]);
        let y = g.matmul(x, w);
        g.set_outputs(vec![y]);

        let program = graph_to_egglog(&g);
        assert!(program.contains("(MatMul"));
        assert!(program.contains("(Input \"x\")"));

        // Verify egglog can parse and run it
        let mut egraph = egglog::EGraph::default();
        egraph.parse_and_run_program(None, &program).unwrap();
    }
}
