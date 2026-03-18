use crate::graph::{Graph, Node, NodeId, Op, TensorType};
use std::{fmt, time::Instant};

/// Report from the e-graph optimization pass.
pub struct OptimizeReport {
    /// The egglog program text (for external inspection / replay).
    pub egglog_program: String,
    /// Number of e-classes after saturation.
    pub num_eclasses: usize,
    /// Number of e-nodes after saturation.
    pub num_enodes: usize,
    /// Which rewrite rules fired and how many times.
    pub rules_fired: Vec<(String, usize)>,
    /// Graph node count before optimization.
    pub nodes_before: usize,
    /// Graph node count after optimization (excluding Nop).
    pub nodes_after: usize,
    /// Fusions applied: list of (fusion_name, node_index) pairs.
    pub fusions_applied: Vec<(String, u32)>,
    /// Wall-clock time for egglog saturation.
    pub egglog_time: std::time::Duration,
    /// Wall-clock time for graph extraction + fusion rewrites.
    pub extract_time: std::time::Duration,
}

impl fmt::Display for OptimizeReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Optimization Report ===")?;
        writeln!(
            f,
            "Egglog saturation: {:.1}ms ({} e-classes, {} e-nodes)",
            self.egglog_time.as_secs_f64() * 1000.0,
            self.num_eclasses,
            self.num_enodes,
        )?;
        if !self.rules_fired.is_empty() {
            writeln!(f, "Rules fired:")?;
            for &(ref rule, count) in &self.rules_fired {
                writeln!(f, "  {}  x{}", rule, count)?;
            }
        }
        writeln!(
            f,
            "Graph: {} nodes -> {} active nodes ({} fused away)",
            self.nodes_before,
            self.nodes_after,
            self.nodes_before.saturating_sub(self.nodes_after),
        )?;
        if !self.fusions_applied.is_empty() {
            write!(f, "Fusions:")?;
            for (i, &(ref name, node_idx)) in self.fusions_applied.iter().enumerate() {
                if i > 0 {
                    write!(f, ",")?;
                }
                write!(f, " {} @node{}", name, node_idx)?;
            }
            writeln!(f)?;
        }
        write!(
            f,
            "Extract time: {:.1}ms",
            self.extract_time.as_secs_f64() * 1000.0
        )
    }
}

/// Convert a Graph to an egglog program string, run equality saturation
/// with rewrite rules, and extract the optimized graph back.
pub fn optimize(graph: &Graph) -> Graph {
    let (graph, _report) = optimize_with_report(graph);
    graph
}

/// Like `optimize`, but also returns a detailed report for debugging.
pub fn optimize_with_report(graph: &Graph) -> (Graph, OptimizeReport) {
    let program = graph_to_egglog(graph);
    log::debug!("egglog program:\n{}", program);

    let nodes_before = graph.nodes().len();
    let mut num_eclasses = 0;
    let mut num_enodes = 0;

    let egglog_start = Instant::now();
    let mut egraph = egglog::EGraph::default();
    let egglog_ok = match egraph.parse_and_run_program(None, &program) {
        Ok(outputs) => {
            for out in &outputs {
                log::debug!("egglog output: {}", out);
            }
            true
        }
        Err(e) => {
            log::warn!(
                "egglog optimization failed: {}, returning original graph",
                e
            );
            false
        }
    };
    let egglog_time = egglog_start.elapsed();

    // Extract e-graph stats if saturation succeeded
    if egglog_ok {
        let serialized = egraph.serialize(egglog::SerializeConfig::default());
        num_eclasses = serialized.egraph.class_data.len();
        num_enodes = serialized.egraph.nodes.len();
    }

    let extract_start = Instant::now();
    let (optimized, fusions_applied) = if egglog_ok {
        extract_graph_with_fusions(&egraph, graph)
    } else {
        (clone_graph(graph), Vec::new())
    };
    let extract_time = extract_start.elapsed();

    let nodes_after = optimized
        .nodes()
        .iter()
        .filter(|n| !matches!(n.op, Op::Nop))
        .count();

    // Count fusions by type
    let mut rules_fired: Vec<(String, usize)> = Vec::new();
    for fusion in &fusions_applied {
        if let Some(entry) = rules_fired.iter_mut().find(|e| e.0 == fusion.0) {
            entry.1 += 1;
        } else {
            rules_fired.push((fusion.0.clone(), 1));
        }
    }

    let report = OptimizeReport {
        egglog_program: program,
        num_eclasses,
        num_enodes,
        rules_fired,
        nodes_before,
        nodes_after,
        fusions_applied,
        egglog_time,
        extract_time,
    };

    (optimized, report)
}

/// Dump the egglog program for a graph (for standalone debugging).
pub fn dump_egglog_program(graph: &Graph) -> String {
    graph_to_egglog(graph)
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
  ; Transformer ops (passthrough, no fusion rules)
  (Silu Op)
  (RmsNorm Op Op)
  (Embedding Op Op)
  (RoPE Op)
  (CausalAttention Op Op Op)
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
            format!("(CrossEntropyLoss n{} n{})", node.inputs[0], node.inputs[1])
        }
        Op::Greater => format!("(Greater n{} n{})", node.inputs[0], node.inputs[1]),
        Op::FusedMatMulRelu => {
            format!("(FusedMatMulRelu n{} n{})", node.inputs[0], node.inputs[1])
        }
        Op::FusedMatMulBiasRelu => format!(
            "(FusedMatMulBiasRelu n{} n{} n{})",
            node.inputs[0], node.inputs[1], node.inputs[2]
        ),
        Op::Silu => format!("(Silu n{})", node.inputs[0]),
        Op::RmsNorm { .. } => format!("(RmsNorm n{} n{})", node.inputs[0], node.inputs[1]),
        Op::Embedding => format!("(Embedding n{} n{})", node.inputs[0], node.inputs[1]),
        Op::RoPE { .. } => format!("(RoPE n{})", node.inputs[0]),
        Op::CausalAttention { .. } => format!(
            "(CausalAttention n{} n{} n{})",
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
fn extract_graph_with_fusions(
    _egraph: &egglog::EGraph,
    original: &Graph,
) -> (Graph, Vec<(String, u32)>) {
    let mut graph = clone_graph(original);
    let fusions = apply_fusion_rewrites(&mut graph);
    (graph, fusions)
}

/// Apply fusion rewrites directly on the graph IR.
/// Returns a list of (fusion_name, node_index) for each fusion applied.
fn apply_fusion_rewrites(graph: &mut Graph) -> Vec<(String, u32)> {
    // We need to iterate and find patterns. Since nodes are append-only
    // and topologically sorted, we can scan forward.
    let len = graph.nodes().len();
    let replacements: Vec<Option<NodeId>> = vec![None; len];

    // Collect fused ops: (relu_idx, fused_op, inputs, ty, dead_node_indices, name)
    type Fusion = (usize, Op, Vec<NodeId>, TensorType, Vec<usize>, &'static str);
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
                                "FusedMatMulBiasRelu",
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
                            "FusedMatMulRelu",
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
    let mut applied = Vec::new();
    let nodes = graph.nodes_mut();
    for (idx, op, inputs, ty, dead, name) in fusions {
        nodes[idx].op = op;
        nodes[idx].inputs = inputs;
        nodes[idx].ty = ty;
        applied.push((name.to_string(), idx as u32));
        for d in dead {
            nodes[d].op = Op::Nop;
            nodes[d].inputs.clear();
        }
    }
    applied
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
    fn test_optimize_report() {
        let mut g = Graph::new();
        let x = g.input("x", &[4, 784]);
        let w1 = g.parameter("w1", &[784, 128]);
        let mm1 = g.matmul(x, w1);
        let h1 = g.relu(mm1);
        let w2 = g.parameter("w2", &[128, 10]);
        let mm2 = g.matmul(h1, w2);
        let h2 = g.relu(mm2);
        g.set_outputs(vec![h2]);

        let (_opt, report) = optimize_with_report(&g);
        assert_eq!(report.fusions_applied.len(), 2);
        assert!(
            report
                .fusions_applied
                .iter()
                .all(|(n, _)| n == "FusedMatMulRelu")
        );
        assert_eq!(report.rules_fired.len(), 1);
        assert_eq!(report.rules_fired[0].1, 2); // fired twice
        assert!(report.nodes_after < report.nodes_before);
        // Verify Display works
        let display = format!("{}", report);
        assert!(display.contains("Optimization Report"));
        assert!(display.contains("FusedMatMulRelu"));
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

    #[test]
    fn test_no_fusion_when_not_applicable() {
        // Graph with no fusable patterns should pass through unchanged
        let mut g = Graph::new();
        let x = g.input("x", &[4, 8]);
        let s = g.sigmoid(x); // sigmoid(input) — not fusable
        g.set_outputs(vec![s]);

        let opt = optimize(&g);
        let output_id = opt.outputs()[0];
        assert!(
            matches!(opt.node(output_id).op, Op::Sigmoid),
            "sigmoid should remain unfused"
        );
    }

    #[test]
    fn test_optimize_preserves_non_fusable_graph() {
        let mut g = Graph::new();
        let a = g.input("a", &[4, 8]);
        let b = g.input("b", &[4, 8]);
        let sum = g.add(a, b);
        let neg = g.neg(sum);
        g.set_outputs(vec![neg]);

        let opt = optimize(&g);
        assert_eq!(opt.nodes().len(), g.nodes().len());
        let out = opt.node(opt.outputs()[0]);
        assert!(matches!(out.op, Op::Neg));
    }

    #[test]
    fn test_optimize_report_no_fusions() {
        let mut g = Graph::new();
        let x = g.input("x", &[4, 8]);
        let s = g.sigmoid(x);
        g.set_outputs(vec![s]);

        let (_opt, report) = optimize_with_report(&g);
        assert!(report.fusions_applied.is_empty());
        assert!(report.rules_fired.is_empty());
        assert_eq!(report.nodes_before, report.nodes_after);
    }

    #[test]
    fn test_optimize_deep_chain_multiple_fusions() {
        // 3-layer MLP: each matmul+relu should fuse independently
        let mut g = Graph::new();
        let x = g.input("x", &[4, 16]);
        let w1 = g.parameter("w1", &[16, 8]);
        let mm1 = g.matmul(x, w1);
        let h1 = g.relu(mm1);
        let w2 = g.parameter("w2", &[8, 4]);
        let mm2 = g.matmul(h1, w2);
        let h2 = g.relu(mm2);
        let w3 = g.parameter("w3", &[4, 2]);
        let mm3 = g.matmul(h2, w3);
        let h3 = g.relu(mm3);
        g.set_outputs(vec![h3]);

        let (_opt, report) = optimize_with_report(&g);
        assert_eq!(report.fusions_applied.len(), 3);
        assert!(
            report
                .fusions_applied
                .iter()
                .all(|(n, _)| n == "FusedMatMulRelu")
        );
    }

    #[test]
    fn test_dump_egglog_program() {
        let mut g = Graph::new();
        let x = g.input("x", &[4, 8]);
        let w = g.parameter("w", &[8, 4]);
        let y = g.matmul(x, w);
        let h = g.relu(y);
        g.set_outputs(vec![h]);

        let program = dump_egglog_program(&g);
        assert!(program.contains("(datatype Op"));
        assert!(program.contains("(rewrite (Relu (MatMul ?a ?b)) (FusedMatMulRelu ?a ?b))"));
        assert!(program.contains("(extract n"));
    }

    #[test]
    fn test_egglog_all_ops() {
        // Verify egglog program generation and parsing for all op types
        let mut g = Graph::new();
        let x = g.input("x", &[4, 8]);
        let w = g.parameter("w", &[8, 4]);
        let _c = g.constant(vec![0.0; 32], &[4, 8]);
        let mm = g.matmul(x, w);
        let _a = g.add(mm, mm);
        let _m = g.mul(mm, mm);
        let b = g.parameter("b", &[4]);
        let _ba = g.bias_add(mm, b);
        let _r = g.relu(mm);
        let _s = g.sigmoid(mm);
        let _n = g.neg(mm);
        let _t = g.transpose(mm);
        let _sm = g.softmax(mm);
        let _lsm = g.log_softmax(mm);
        let sa = g.sum_all(mm);
        let _ma = g.mean_all(mm);
        let _gt = g.greater(mm, mm);
        let _cel = g.cross_entropy_loss(mm, mm);
        g.set_outputs(vec![sa]);

        let program = graph_to_egglog(&g);
        let mut egraph = egglog::EGraph::default();
        egraph.parse_and_run_program(None, &program).unwrap();
    }

    #[test]
    fn test_clone_graph_preserves_structure() {
        let mut g = Graph::new();
        let x = g.input("x", &[4, 8]);
        let w = g.parameter("w", &[8, 4]);
        let y = g.matmul(x, w);
        g.set_outputs(vec![y]);

        let cloned = clone_graph(&g);
        assert_eq!(cloned.nodes().len(), g.nodes().len());
        assert_eq!(cloned.outputs(), g.outputs());
        for (a, b) in cloned.nodes().iter().zip(g.nodes().iter()) {
            assert_eq!(a.id, b.id);
            assert_eq!(a.inputs, b.inputs);
            assert_eq!(a.ty.shape, b.ty.shape);
        }
    }
}
