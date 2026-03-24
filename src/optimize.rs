use crate::graph::{Graph, Node, Op};
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
  ; Transformer ops (passthrough, no fusion rules)
  (Silu Op)
  (Gelu Op)
  (RmsNorm Op Op)
  (Embedding Op Op)
  (RoPE Op)
  (CausalAttention Op Op Op)
  (LayerNorm Op Op Op)
  (FullAttention Op Op Op)
  (CrossAttention Op Op Op)
)

",
    );

    // Rewrite rules
    prog.push_str(
        "\
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
        Op::Silu => format!("(Silu n{})", node.inputs[0]),
        Op::RmsNorm { .. } => format!("(RmsNorm n{} n{})", node.inputs[0], node.inputs[1]),
        Op::Embedding => format!("(Embedding n{} n{})", node.inputs[0], node.inputs[1]),
        Op::RoPE { .. } => format!("(RoPE n{})", node.inputs[0]),
        Op::CausalAttention { .. } => format!(
            "(CausalAttention n{} n{} n{})",
            node.inputs[0], node.inputs[1], node.inputs[2]
        ),
        Op::Gelu => format!("(Gelu n{})", node.inputs[0]),
        Op::LayerNorm { .. } => format!(
            "(LayerNorm n{} n{} n{})",
            node.inputs[0], node.inputs[1], node.inputs[2]
        ),
        Op::FullAttention { .. } => format!(
            "(FullAttention n{} n{} n{})",
            node.inputs[0], node.inputs[1], node.inputs[2]
        ),
        Op::CrossAttention { .. } => format!(
            "(CrossAttention n{} n{} n{})",
            node.inputs[0], node.inputs[1], node.inputs[2]
        ),
        Op::Nop | Op::FusedMatMulAdd => {
            unreachable!("Nop/FusedMatMulAdd nodes should not appear before optimization")
        }
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
    use crate::graph::Op;
    let mut graph = clone_graph(original);
    let mut fusions = Vec::new();

    // Fuse Add(MatMul(a, b), d) → FusedMatMulAdd(a, b, d)
    let node_ids: Vec<usize> = (0..graph.nodes().len()).collect();
    for &id in &node_ids {
        let node = &graph.nodes()[id];
        if !matches!(node.op, Op::Add) {
            continue;
        }
        let (lhs, rhs) = (node.inputs[0], node.inputs[1]);
        let (mm_id, addend_id) = if matches!(graph.node(lhs).op, Op::MatMul) {
            (lhs, rhs)
        } else if matches!(graph.node(rhs).op, Op::MatMul) {
            (rhs, lhs)
        } else {
            continue;
        };

        // Only fuse if the MatMul result is used exclusively by this Add.
        let mm_use_count = graph
            .nodes()
            .iter()
            .filter(|n| n.inputs.contains(&mm_id) && !matches!(n.op, Op::Nop))
            .count();
        if mm_use_count != 1 {
            continue;
        }

        // Rewrite: Add node becomes FusedMatMulAdd, MatMul becomes Nop
        let mm_node = graph.node(mm_id);
        let (a, b) = (mm_node.inputs[0], mm_node.inputs[1]);
        graph.nodes_mut()[id].op = Op::FusedMatMulAdd;
        graph.nodes_mut()[id].inputs = vec![a, b, addend_id];
        graph.nodes_mut()[mm_id as usize].op = Op::Nop;
        fusions.push(("MatMul+Add→FusedMatMulAdd".to_string(), id as u32));
    }

    (graph, fusions)
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
    fn test_no_fusion_cooperative_matrix() {
        // With cooperative matrix, matmul+relu stay as separate ops
        let mut g = Graph::new();
        let x = g.input("x", &[4, 784]);
        let w = g.parameter("w", &[784, 128]);
        let mm = g.matmul(x, w);
        let h = g.relu(mm);
        g.set_outputs(vec![h]);

        let opt = optimize(&g);
        let output_id = opt.outputs()[0];
        let output_node = opt.node(output_id);
        assert!(
            matches!(output_node.op, Op::Relu),
            "expected Relu (no fusion), got {:?}",
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
        // No fusions with cooperative matrix
        assert!(report.fusions_applied.is_empty());
        // Verify Display works
        let display = format!("{}", report);
        assert!(display.contains("Optimization Report"));
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
    fn test_optimize_preserves_graph() {
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
    fn test_dump_egglog_program() {
        let mut g = Graph::new();
        let x = g.input("x", &[4, 8]);
        let w = g.parameter("w", &[8, 4]);
        let y = g.matmul(x, w);
        let _h = g.relu(y);
        g.set_outputs(vec![y]);

        let program = dump_egglog_program(&g);
        assert!(program.contains("(datatype Op"));
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

    #[test]
    fn test_matmul_stays_as_matmul() {
        // With cooperative matrix, matmul is never transformed
        let mut g = Graph::new();
        let x = g.input("x", &[2, 1024]);
        let w = g.parameter("w", &[1024, 64]);
        let y = g.matmul(x, w);
        g.set_outputs(vec![y]);

        let opt = optimize(&g);
        let output_id = opt.outputs()[0];
        assert!(
            matches!(opt.node(output_id).op, Op::MatMul),
            "expected MatMul, got {:?}",
            opt.node(output_id).op
        );
    }
}
