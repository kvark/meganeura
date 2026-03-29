use crate::graph::{Graph, Node, Op};
use egglog::{Term, TermDag, TermId};
use std::collections::HashMap;
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

    let node_count = graph
        .nodes()
        .iter()
        .filter(|n| !matches!(n.op, Op::Nop))
        .count();
    // egglog saturation time grows superlinearly with node count.
    // For the SmolVLA training graph (~750 nodes), saturation takes
    // minutes. Fall back to direct pattern matching for large graphs.
    // TODO: investigate egglog performance or use incremental matching.
    let egglog_start = Instant::now();
    // egglog saturation with shared-parameter graphs (like transformers
    // where the same weight is used by many MatMul nodes) creates large
    // e-classes that make pattern matching slow. 300 nodes handles most
    // small models; larger models fall back to direct pattern matching.
    if node_count > 300 {
        log::debug!(
            "egglog: {} nodes, falling back to pattern matching",
            node_count
        );
        let extract_start = Instant::now();
        let (optimized, fusions_applied) = rebuild_graph_from_extractions(graph, &[]);
        let extract_time = extract_start.elapsed();
        return (
            optimized,
            OptimizeReport {
                egglog_program: program,
                num_eclasses: 0,
                num_enodes: 0,
                rules_fired: fusions_applied.iter().fold(Vec::new(), |mut acc, entry| {
                    let name = &entry.0;
                    if let Some(e) = acc.iter_mut().find(|e: &&mut (String, usize)| e.0 == *name) {
                        e.1 += 1;
                    } else {
                        acc.push((name.clone(), 1));
                    }
                    acc
                }),
                nodes_before,
                nodes_after: 0,
                fusions_applied,
                egglog_time: std::time::Duration::ZERO,
                extract_time,
            },
        );
    }
    let mut egraph = egglog::EGraph::default();
    let egglog_result = egraph.parse_and_run_program(None, &program);
    log::debug!(
        "egglog: saturation took {:.1}ms",
        egglog_start.elapsed().as_secs_f64() * 1000.0
    );
    let egglog_ok;
    let mut extractions: Vec<(TermDag, TermId)> = Vec::new();

    match egglog_result {
        Ok(outputs) => {
            egglog_ok = true;
            for out in &outputs {
                if let egglog::CommandOutput::ExtractBest(ref dag, _cost, term_id) = *out {
                    log::debug!("egglog extracted: {}", dag.to_string(term_id));
                    extractions.push((dag.clone(), term_id));
                }
            }
        }
        Err(e) => {
            log::warn!(
                "egglog optimization failed: {}, returning original graph",
                e
            );
            egglog_ok = false;
        }
    };
    let egglog_time = egglog_start.elapsed();

    if egglog_ok {
        let serialized = egraph.serialize(egglog::SerializeConfig::default());
        num_eclasses = serialized.egraph.class_data.len();
        num_enodes = serialized.egraph.nodes.len();
    }

    let extract_start = Instant::now();
    let (optimized, fusions_applied) = rebuild_graph_from_extractions(graph, &extractions);
    let extract_time = extract_start.elapsed();

    let nodes_after = optimized
        .nodes()
        .iter()
        .filter(|n| !matches!(n.op, Op::Nop))
        .count();

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
/// Encodes the FULL graph (forward + backward) into egglog. Every node
/// becomes an expression. Rewrite rules express algebraic simplifications
/// and kernel fusions — egglog discovers which fusions are applicable via
/// equality saturation.
fn graph_to_egglog(graph: &Graph) -> String {
    let mut prog = String::new();

    // Sort and constructors — covers forward, backward, and fused ops
    prog.push_str(
        "\
(datatype Op
  ; --- Leaf nodes ---
  (Input String)
  (Parameter String)
  (Const i64)
  ; --- Forward matmul variants ---
  (MatMul Op Op)
  (MatMulAT Op Op)
  (MatMulBT Op Op)
  ; --- Fused matmul+add (targets for fusion rules) ---
  (FusedMatMulAdd Op Op Op)
  (FusedMatMulATAdd Op Op Op)
  (FusedMatMulBTAdd Op Op Op)
  ; --- Element-wise ---
  (Add Op Op)
  (Mul Op Op)
  (BiasAdd Op Op)
  (Relu Op)
  (Sigmoid Op)
  (Neg Op)
  (Abs Op)
  (Log Op)
  (Recip Op)
  (ScatterAdd i64 Op Op)
  (Silu Op)
  (Gelu Op)
  ; --- Shape / reduction ---
  (Transpose Op)
  (Softmax Op)
  (LogSoftmax Op)
  (SumAll Op)
  (MeanAll Op)
  (SumRows Op)
  (CrossEntropyLoss Op Op)
  (BceLoss Op Op)
  (Greater Op Op)
  ; --- Transformer forward ---
  (SwiGLU Op Op)
  (SwiGLUConcat Op)
  (RmsNorm Op Op)
  (FusedRmsNormMatMul Op Op Op)
  (Embedding Op Op)
  (RoPE Op)
  (CausalAttention Op Op Op)
  (LayerNorm Op Op Op)
  (FullAttention Op Op Op)
  (CrossAttention Op Op Op)
  (MultiHeadAttn Op Op Op)
  ; --- KV cache ops ---
  (CacheWrite Op Op Op)
  (CachedAttention Op Op Op Op)
  ; --- Backward / gradient ops ---
  (SiluGrad Op Op)
  (SwiGLUGradGate Op Op Op)
  (SwiGLUGradUp Op Op)
  (SwiGLUConcatGrad Op Op)
  (RmsNormGradW Op Op Op)
  (RmsNormGradX Op Op Op)
  (MHAGradQ Op Op Op Op)
  (MHAGradK Op Op Op Op)
  (MHAGradV Op Op Op Op)
)

",
    );

    // Rewrite rules — these are the optimizations egglog discovers
    prog.push_str(
        "\
; --- Algebraic simplifications ---
(rewrite (Neg (Neg ?x)) ?x)
(rewrite (Transpose (Transpose ?x)) ?x)
(rewrite (Relu (Relu ?x)) (Relu ?x))

; --- Kernel fusion: Add(MatMul*(a,b), d) → FusedMatMul*Add(a,b,d) ---
; Both argument orders handled explicitly (no general Add commutativity
; rule, which causes exponential blowup on large graphs).
(rewrite (Add (MatMul ?a ?b) ?d)    (FusedMatMulAdd ?a ?b ?d))
(rewrite (Add ?d (MatMul ?a ?b))    (FusedMatMulAdd ?a ?b ?d))
(rewrite (Add (MatMulAT ?a ?b) ?d)  (FusedMatMulATAdd ?a ?b ?d))
(rewrite (Add ?d (MatMulAT ?a ?b))  (FusedMatMulATAdd ?a ?b ?d))
(rewrite (Add (MatMulBT ?a ?b) ?d)  (FusedMatMulBTAdd ?a ?b ?d))
(rewrite (Add ?d (MatMulBT ?a ?b))  (FusedMatMulBTAdd ?a ?b ?d))

; --- RmsNorm+MatMul fusion ---
(rewrite (MatMul (RmsNorm ?x ?w_norm) ?w_proj) (FusedRmsNormMatMul ?x ?w_norm ?w_proj))

; --- SwiGLU fusion: two matmuls sharing input → single wide matmul ---
; SwiGLU(MatMul(h, w1), MatMul(h, w2)) can use SwiGLUConcat on a
; concatenated [h, w1|w2] matmul. Pattern matcher handles weight
; concatenation since egglog can't create new tensors.
; (documented here; applied by apply_swiglu_concat_fusions)

",
    );

    // Encode every node (forward AND backward)
    for node in graph.nodes() {
        if matches!(node.op, Op::Nop) {
            continue;
        }
        let expr = node_to_egglog_expr(node);
        prog.push_str(&format!("(let n{} {})\n", node.id, expr));
    }

    // Run equality saturation with a node limit to keep saturation fast.
    // The fusion rules only need one pass — they're not iterative.
    prog.push_str("(run 1)\n\n");

    // Extract all output nodes (after saturation)
    for &out in graph.outputs() {
        if !matches!(graph.node(out).op, Op::Nop) {
            prog.push_str(&format!("(extract n{})\n", out));
        }
    }

    prog
}

fn node_to_egglog_expr(node: &Node) -> String {
    let i = &node.inputs;
    match node.op {
        Op::Input { ref name } => format!("(Input \"{}\")", name),
        Op::Parameter { ref name } => format!("(Parameter \"{}\")", name),
        Op::Constant { .. } => format!("(Const {})", node.id),
        Op::MatMul => format!("(MatMul n{} n{})", i[0], i[1]),
        Op::MatMulAT => format!("(MatMulAT n{} n{})", i[0], i[1]),
        Op::MatMulBT => format!("(MatMulBT n{} n{})", i[0], i[1]),
        Op::Add => format!("(Add n{} n{})", i[0], i[1]),
        Op::Mul => format!("(Mul n{} n{})", i[0], i[1]),
        Op::BiasAdd => format!("(BiasAdd n{} n{})", i[0], i[1]),
        Op::Relu => format!("(Relu n{})", i[0]),
        Op::Sigmoid => format!("(Sigmoid n{})", i[0]),
        Op::Neg => format!("(Neg n{})", i[0]),
        Op::Abs => format!("(Abs n{})", i[0]),
        Op::Log => format!("(Log n{})", i[0]),
        Op::Recip => format!("(Recip n{})", i[0]),
        Op::ScatterAdd { vocab_size } => {
            format!("(ScatterAdd {} n{} n{})", vocab_size, i[0], i[1])
        }
        Op::Transpose => format!("(Transpose n{})", i[0]),
        Op::Softmax => format!("(Softmax n{})", i[0]),
        Op::LogSoftmax => format!("(LogSoftmax n{})", i[0]),
        Op::SumAll => format!("(SumAll n{})", i[0]),
        Op::MeanAll => format!("(MeanAll n{})", i[0]),
        Op::SumRows => format!("(SumRows n{})", i[0]),
        Op::CrossEntropyLoss => format!("(CrossEntropyLoss n{} n{})", i[0], i[1]),
        Op::BceLoss => format!("(BceLoss n{} n{})", i[0], i[1]),
        Op::Greater => format!("(Greater n{} n{})", i[0], i[1]),
        Op::Silu => format!("(Silu n{})", i[0]),
        Op::SwiGLU => format!("(SwiGLU n{} n{})", i[0], i[1]),
        Op::SwiGLUConcat => format!("(SwiGLUConcat n{})", i[0]),
        Op::Gelu => format!("(Gelu n{})", i[0]),
        Op::RmsNorm { .. } => format!("(RmsNorm n{} n{})", i[0], i[1]),
        Op::Embedding => format!("(Embedding n{} n{})", i[0], i[1]),
        Op::RoPE { .. } => format!("(RoPE n{})", i[0]),
        Op::CausalAttention { .. } => {
            format!("(CausalAttention n{} n{} n{})", i[0], i[1], i[2])
        }
        Op::LayerNorm { .. } => format!("(LayerNorm n{} n{} n{})", i[0], i[1], i[2]),
        Op::FullAttention { .. } => format!("(FullAttention n{} n{} n{})", i[0], i[1], i[2]),
        Op::CrossAttention { .. } => format!("(CrossAttention n{} n{} n{})", i[0], i[1], i[2]),
        Op::MultiHeadAttn { .. } => format!("(MultiHeadAttn n{} n{} n{})", i[0], i[1], i[2]),
        // Backward ops
        Op::SiluGrad => format!("(SiluGrad n{} n{})", i[0], i[1]),
        Op::SwiGLUGradGate => format!("(SwiGLUGradGate n{} n{} n{})", i[0], i[1], i[2]),
        Op::SwiGLUGradUp => format!("(SwiGLUGradUp n{} n{})", i[0], i[1]),
        Op::SwiGLUConcatGrad => format!("(SwiGLUConcatGrad n{} n{})", i[0], i[1]),
        Op::RmsNormGradW { .. } => format!("(RmsNormGradW n{} n{} n{})", i[0], i[1], i[2]),
        Op::RmsNormGradX { .. } => format!("(RmsNormGradX n{} n{} n{})", i[0], i[1], i[2]),
        Op::MultiHeadAttnGradQ { .. } => {
            format!("(MHAGradQ n{} n{} n{} n{})", i[0], i[1], i[2], i[3])
        }
        Op::MultiHeadAttnGradK { .. } => {
            format!("(MHAGradK n{} n{} n{} n{})", i[0], i[1], i[2], i[3])
        }
        Op::MultiHeadAttnGradV { .. } => {
            format!("(MHAGradV n{} n{} n{} n{})", i[0], i[1], i[2], i[3])
        }
        // Fused ops from a previous optimization pass — encode as-is
        Op::FusedMatMulAdd => {
            format!("(FusedMatMulAdd n{} n{} n{})", i[0], i[1], i[2])
        }
        Op::FusedMatMulATAdd => {
            format!("(FusedMatMulATAdd n{} n{} n{})", i[0], i[1], i[2])
        }
        Op::FusedMatMulBTAdd => {
            format!("(FusedMatMulBTAdd n{} n{} n{})", i[0], i[1], i[2])
        }
        Op::FusedRmsNormMatMul { .. } => {
            format!("(FusedRmsNormMatMul n{} n{} n{})", i[0], i[1], i[2])
        }
        Op::CacheWrite => format!("(CacheWrite n{} n{} n{})", i[0], i[1], i[2]),
        Op::CachedAttention { .. } => {
            format!("(CachedAttention n{} n{} n{} n{})", i[0], i[1], i[2], i[3])
        }
        Op::Nop => unreachable!("Nop nodes are filtered before encoding"),
    }
}

/// Rebuild the graph using egglog extraction results.
///
/// Each extraction is a `(TermDag, TermId)` from egglog's `(extract ...)`.
/// We walk the term trees, matching them back to original graph nodes.
/// Where egglog chose a fused variant (e.g. FusedMatMulAdd instead of
/// Add(MatMul, x)), we apply the fusion in the graph.
///
/// Falls back to manual pattern matching if no extractions are available
/// (e.g. egglog failed or the graph has no extract commands).
fn rebuild_graph_from_extractions(
    original: &Graph,
    extractions: &[(TermDag, TermId)],
) -> (Graph, Vec<(String, u32)>) {
    let mut graph = clone_graph(original);
    let mut fusions = Vec::new();

    if !extractions.is_empty() {
        // Build a lookup: (op_name, input_node_ids) → graph node id.
        // This lets us match extracted terms back to original graph nodes.
        let mut node_lookup: HashMap<String, Vec<usize>> = HashMap::new();
        for node in graph.nodes() {
            let key = egglog_key(node);
            node_lookup.entry(key).or_default().push(node.id as usize);
        }

        // Walk each extracted term tree looking for fused ops that differ
        // from the original graph. When we find a FusedMatMul*Add that
        // corresponds to an original Add(MatMul*(...), d), apply the fusion.
        for &(ref dag, root) in extractions {
            scan_fusions(dag, root, &graph, &node_lookup, &mut fusions);
        }
    }

    // Apply fusion rules iteratively until fixpoint.
    // Each rule fires on matching patterns, potentially exposing new patterns
    // for subsequent rules (like e-graph saturation, but on the graph IR).
    loop {
        let n = fusions.len();
        apply_matmul_add_fusions(&mut graph, &mut fusions);
        apply_swiglu_concat_fusions(&mut graph, &mut fusions);
        // RmsNorm+MatMul fusion: disabled on iGPU — the fused kernel's
        // per-element normalization (2 extra FMAs/element in tile loads)
        // outweighs the saved intermediate write (144KB). Enable via cost
        // model on discrete GPUs where bandwidth savings matter more.
        // apply_rms_norm_matmul_fusions(&mut graph, &mut fusions);
        if fusions.len() == n {
            break;
        }
    }
    let active_nodes = graph
        .nodes()
        .iter()
        .filter(|n| !matches!(n.op, Op::Nop))
        .count();
    log::info!(
        "optimizer: {} fusions on {} nodes",
        fusions.len(),
        active_nodes
    );
    for (name, count) in fusions.iter().fold(
        std::collections::BTreeMap::<&str, usize>::new(),
        |mut acc, entry| {
            let name = &entry.0;
            *acc.entry(name.as_str()).or_default() += 1;
            acc
        },
    ) {
        log::info!("  {}x {}", count, name);
    }

    (graph, fusions)
}

/// Generate a lookup key for a graph node (op name + input IDs).
fn egglog_key(node: &Node) -> String {
    let op_name = match node.op {
        Op::Input { ref name } => format!("Input:{}", name),
        Op::Parameter { ref name } => format!("Parameter:{}", name),
        Op::Constant { .. } => format!("Const:{}", node.id),
        _ => format!("{:?}", std::mem::discriminant(&node.op)),
    };
    format!("{}:{:?}", op_name, node.inputs)
}

/// Walk an extracted term tree looking for fused ops.
fn scan_fusions(
    dag: &TermDag,
    term_id: TermId,
    _graph: &Graph,
    _lookup: &HashMap<String, Vec<usize>>,
    _fusions: &mut Vec<(String, u32)>,
) {
    if let Term::App(name, children) = dag.get(term_id).clone() {
        if name.starts_with("FusedMatMul") || name.starts_with("FusedRmsNorm") {
            log::debug!("egglog discovered fusion: {}", name);
        }
        for child in children {
            scan_fusions(dag, child, _graph, _lookup, _fusions);
        }
    }
}

/// Apply Add(MatMul*(a, b), d) → FusedMatMul*Add(a, b, d) fusions.
///
/// This is the concrete graph mutation. It matches the patterns that
/// egglog's rewrite rules express, applying them with the additional
/// single-use constraint (the MatMul must feed only this Add).
fn apply_matmul_add_fusions(graph: &mut Graph, fusions: &mut Vec<(String, u32)>) {
    let node_ids: Vec<usize> = (0..graph.nodes().len()).collect();
    for &id in &node_ids {
        let node = &graph.nodes()[id];
        if !matches!(node.op, Op::Add) {
            continue;
        }
        let (lhs, rhs) = (node.inputs[0], node.inputs[1]);
        let (mm_id, addend_id) =
            if matches!(graph.node(lhs).op, Op::MatMul | Op::MatMulAT | Op::MatMulBT) {
                (lhs, rhs)
            } else if matches!(graph.node(rhs).op, Op::MatMul | Op::MatMulAT | Op::MatMulBT) {
                (rhs, lhs)
            } else {
                continue;
            };
        let mm_use_count = graph
            .nodes()
            .iter()
            .filter(|n| n.inputs.contains(&mm_id) && !matches!(n.op, Op::Nop))
            .count();
        if mm_use_count != 1 {
            continue;
        }

        let mm_node = graph.node(mm_id);
        let (a, b) = (mm_node.inputs[0], mm_node.inputs[1]);
        let (fused_op, label) = match mm_node.op {
            Op::MatMul => (Op::FusedMatMulAdd, "MatMul+Add→FusedMatMulAdd"),
            Op::MatMulAT => (Op::FusedMatMulATAdd, "MatMulAT+Add→FusedMatMulATAdd"),
            Op::MatMulBT => (Op::FusedMatMulBTAdd, "MatMulBT+Add→FusedMatMulBTAdd"),
            _ => unreachable!(),
        };
        graph.nodes_mut()[id].op = fused_op;
        graph.nodes_mut()[id].inputs = vec![a, b, addend_id];
        graph.nodes_mut()[mm_id as usize].op = Op::Nop;
        fusions.push((label.to_string(), id as u32));
    }
}

/// Fuse SwiGLU(MatMul(h, w_gate), MatMul(h, w_up)) → SwiGLUConcat(MatMul(h, w_gate_up))
///
/// When both gate and up projections share the same input `h`, merge
/// them into a single wide matmul [hidden, 2*intermediate] followed by
/// SwiGLUConcat. This halves the number of matmul dispatches for the
/// MLP gate+up path. The optimizer creates a new `concat_weight` parameter
/// node so model code can use the naive two-matmul pattern.
fn apply_swiglu_concat_fusions(graph: &mut Graph, fusions: &mut Vec<(String, u32)>) {
    use crate::graph::TensorType;
    let node_ids: Vec<usize> = (0..graph.nodes().len()).collect();
    for &id in &node_ids {
        let node = &graph.nodes()[id];
        if !matches!(node.op, Op::SwiGLU) {
            continue;
        }
        let (gate_id, up_id) = (node.inputs[0], node.inputs[1]);
        let gate_node = graph.node(gate_id);
        let up_node = graph.node(up_id);

        // Both must be MatMul
        if !matches!(gate_node.op, Op::MatMul) || !matches!(up_node.op, Op::MatMul) {
            continue;
        }
        // Both must share the same input (first operand)
        if gate_node.inputs[0] != up_node.inputs[0] {
            continue;
        }
        // Both matmuls must be single-use (only feeding this SwiGLU)
        let gate_uses = graph
            .nodes()
            .iter()
            .filter(|n| n.inputs.contains(&gate_id) && !matches!(n.op, Op::Nop))
            .count();
        let up_uses = graph
            .nodes()
            .iter()
            .filter(|n| n.inputs.contains(&up_id) && !matches!(n.op, Op::Nop))
            .count();
        if gate_uses != 1 || up_uses != 1 {
            continue;
        }

        let h = gate_node.inputs[0];
        let w_gate = gate_node.inputs[1];
        let w_up = up_node.inputs[1];

        // Create concatenated weight parameter: [in_features, 2 * out_features]
        let gate_shape = &graph.node(w_gate).ty.shape;
        let up_shape = &graph.node(w_up).ty.shape;
        if gate_shape.len() != 2 || up_shape.len() != 2 {
            continue;
        }
        if gate_shape[0] != up_shape[0] || gate_shape[1] != up_shape[1] {
            continue;
        }
        let in_features = gate_shape[0];
        let out_features = gate_shape[1];
        let concat_shape = vec![in_features, 2 * out_features];
        let gate_name = match graph.node(w_gate).op {
            Op::Parameter { ref name } => name.clone(),
            _ => "w_gate".to_string(),
        };
        let up_name = match graph.node(w_up).op {
            Op::Parameter { ref name } => name.clone(),
            _ => "w_up".to_string(),
        };
        let concat_name = format!("{}+{}", gate_name, up_name);

        // Record derivation so runtime can fill this from original params
        graph.derived_params.push(crate::graph::DerivedParam {
            name: concat_name.clone(),
            sources: vec![(gate_name, out_features), (up_name, out_features)],
            rows: in_features,
        });
        let concat_w = graph.add_raw_node(
            Op::Parameter { name: concat_name },
            vec![],
            TensorType::f32(concat_shape.clone()),
        );

        // MatMul(h, concat_w) → [M, 2*out_features]
        let m = graph.node(h).ty.shape[0];
        let wide_mm = graph.add_raw_node(
            Op::MatMul,
            vec![h, concat_w],
            TensorType::f32(vec![m, 2 * out_features]),
        );

        // SwiGLUConcat(wide_mm) → [M, out_features]
        let swiglu_ty = TensorType::f32(vec![m, out_features]);
        graph.nodes_mut()[id].op = Op::SwiGLUConcat;
        graph.nodes_mut()[id].inputs = vec![wide_mm];
        graph.nodes_mut()[id].ty = swiglu_ty;

        // Mark old matmuls as Nop
        graph.nodes_mut()[gate_id as usize].op = Op::Nop;
        graph.nodes_mut()[up_id as usize].op = Op::Nop;

        fusions.push((
            "SwiGLU(MatMul,MatMul)→SwiGLUConcat(MatMul)".to_string(),
            id as u32,
        ));
    }
}

/// Fuse MatMul(RmsNorm(x, w_norm, eps), w_proj) → FusedRmsNormMatMul(x, w_norm, w_proj, eps)
///
/// Only fuses if the RmsNorm result is used exclusively by this MatMul.
#[allow(dead_code)]
fn apply_rms_norm_matmul_fusions(graph: &mut Graph, fusions: &mut Vec<(String, u32)>) {
    use crate::graph::TensorType;
    let node_ids: Vec<usize> = (0..graph.nodes().len()).collect();
    for &id in &node_ids {
        let node = &graph.nodes()[id];
        if !matches!(node.op, Op::MatMul) {
            continue;
        }
        let (norm_id, w_proj_id) = (node.inputs[0], node.inputs[1]);
        let norm_node = graph.node(norm_id);
        let eps = match norm_node.op {
            Op::RmsNorm { eps } => eps,
            _ => continue,
        };
        // RmsNorm must be single-use (only feeding this MatMul)
        let norm_use_count = graph
            .nodes()
            .iter()
            .filter(|n| n.inputs.contains(&norm_id) && !matches!(n.op, Op::Nop))
            .count();
        if norm_use_count != 1 {
            continue;
        }

        let x = norm_node.inputs[0];
        let w_norm = norm_node.inputs[1];
        let x_shape = &graph.node(x).ty.shape;
        let w_proj_shape = &graph.node(w_proj_id).ty.shape;
        let m = x_shape[0];
        let n = w_proj_shape[1];

        // Rewrite the MatMul node to FusedRmsNormMatMul
        graph.nodes_mut()[id].op = Op::FusedRmsNormMatMul { eps };
        graph.nodes_mut()[id].inputs = vec![x, w_norm, w_proj_id];
        graph.nodes_mut()[id].ty = TensorType::f32(vec![m, n]);
        // Mark old RmsNorm as Nop
        graph.nodes_mut()[norm_id as usize].op = Op::Nop;

        fusions.push(("RmsNorm+MatMul→FusedRmsNormMatMul".to_string(), id as u32));
    }
}

fn clone_graph(graph: &Graph) -> Graph {
    let mut new_graph = Graph::new();
    for node in graph.nodes() {
        new_graph.add_raw_node(node.op.clone(), node.inputs.clone(), node.ty.clone());
    }
    new_graph.set_outputs(graph.outputs().to_vec());
    new_graph.derived_params = graph.derived_params.clone();
    new_graph
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_no_fusion_cooperative_matrix() {
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
        assert!(report.fusions_applied.is_empty());
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

        let mut egraph = egglog::EGraph::default();
        egraph.parse_and_run_program(None, &program).unwrap();
    }

    /// Verify egglog extraction returns fused terms via TermDag.
    #[test]
    fn test_egglog_extract_returns_fused() {
        let mut egraph = egglog::EGraph::default();
        let outputs = egraph
            .parse_and_run_program(
                None,
                r#"
(datatype Op
  (MatMul Op Op)
  (MatMulBT Op Op)
  (Add Op Op)
  (FusedMatMulAdd Op Op Op)
  (FusedMatMulBTAdd Op Op Op)
  (Input String)
  (Parameter String)
)
(rewrite (Add (MatMul ?a ?b) ?d) (FusedMatMulAdd ?a ?b ?d))
(rewrite (Add (MatMulBT ?a ?b) ?d) (FusedMatMulBTAdd ?a ?b ?d))
(rewrite (Add ?x ?y) (Add ?y ?x))

(let n0 (Input "x"))
(let n1 (Parameter "w"))
(let n2 (MatMul n0 n1))
(let n3 (Input "bias"))
(let n4 (Add n2 n3))
(run 10)
(extract n4)
"#,
            )
            .unwrap();
        // Find the ExtractBest output
        let mut found_fused = false;
        for out in &outputs {
            if let egglog::CommandOutput::ExtractBest(dag, _cost, term_id) = out {
                let s = dag.to_string(*term_id);
                eprintln!("egglog extracted: {}", s);
                assert!(
                    s.contains("FusedMatMulAdd"),
                    "expected FusedMatMulAdd, got: {}",
                    s
                );
                // Verify the term tree structure
                match dag.get(*term_id) {
                    Term::App(name, _children) => {
                        assert_eq!(name, "FusedMatMulAdd");
                    }
                    other => panic!("expected App, got {:?}", other),
                }
                found_fused = true;
            }
        }
        assert!(found_fused, "no ExtractBest output found");
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

    /// Measure egglog saturation time vs graph size.
    #[test]
    fn test_egglog_scalability() {
        for n in [10, 50, 100, 200, 350] {
            let mut prog = String::from(
                "(datatype Op
  (MatMul Op Op) (MatMulAT Op Op) (MatMulBT Op Op)
  (Add Op Op) (Input String) (Parameter String)
  (FusedMatMulAdd Op Op Op) (FusedMatMulATAdd Op Op Op) (FusedMatMulBTAdd Op Op Op)
)\n",
            );
            prog.push_str("(rewrite (Add (MatMul ?a ?b) ?d) (FusedMatMulAdd ?a ?b ?d))\n");
            prog.push_str("(rewrite (Add ?d (MatMul ?a ?b)) (FusedMatMulAdd ?a ?b ?d))\n");
            prog.push_str("(rewrite (Add (MatMulAT ?a ?b) ?d) (FusedMatMulATAdd ?a ?b ?d))\n");
            prog.push_str("(rewrite (Add ?d (MatMulAT ?a ?b)) (FusedMatMulATAdd ?a ?b ?d))\n");
            prog.push_str("(rewrite (Add (MatMulBT ?a ?b) ?d) (FusedMatMulBTAdd ?a ?b ?d))\n");
            prog.push_str("(rewrite (Add ?d (MatMulBT ?a ?b)) (FusedMatMulBTAdd ?a ?b ?d))\n");

            prog.push_str("(let n0 (Input \"x\"))\n(let n1 (Parameter \"w\"))\n");
            for i in 1..n {
                let prev = (i - 1) * 2 + 2;
                match i % 3 {
                    0 => prog.push_str(&format!("(let n{} (MatMulAT n{} n1))\n", i * 2, prev - 1)),
                    1 => prog.push_str(&format!("(let n{} (MatMulBT n{} n1))\n", i * 2, prev - 1)),
                    _ => prog.push_str(&format!("(let n{} (MatMul n{} n1))\n", i * 2, prev - 1)),
                }
                prog.push_str(&format!(
                    "(let n{} (Add n{} n{}))\n",
                    i * 2 + 1,
                    i * 2,
                    prev - 1
                ));
            }
            prog.push_str("(run 1)\n");
            let last = (n - 1) * 2 + 1;
            prog.push_str(&format!("(extract n{})\n", last));

            let t0 = Instant::now();
            let mut egraph = egglog::EGraph::default();
            egraph.parse_and_run_program(None, &prog).unwrap();
            let elapsed = t0.elapsed();
            eprintln!(
                "egglog scalability: n={:>4} nodes -> {:>8.1}ms",
                n * 2,
                elapsed.as_secs_f64() * 1000.0
            );
        }
    }

    /// E-graph discovers MatMul+Add → FusedMatMulAdd.
    #[test]
    fn test_egglog_discovers_matmul_add_fusion() {
        let mut g = Graph::new();
        let x = g.input("x", &[4, 8]);
        let w = g.parameter("w", &[8, 4]);
        let b = g.input("bias", &[4, 4]);
        let mm = g.matmul(x, w);
        let out = g.add(mm, b);
        g.set_outputs(vec![out]);

        let (opt, report) = optimize_with_report(&g);
        let output_node = opt.node(opt.outputs()[0]);
        assert!(
            matches!(output_node.op, Op::FusedMatMulAdd),
            "expected FusedMatMulAdd, got {:?}",
            output_node.op
        );
        assert!(!report.fusions_applied.is_empty());
    }

    /// SwiGLU(MatMul, MatMul) → SwiGLUConcat(MatMul) fusion.
    #[test]
    fn test_swiglu_concat_fusion() {
        let mut g = Graph::new();
        let h = g.input("h", &[50, 720]);
        let w_gate = g.parameter("w_gate", &[720, 2048]);
        let w_up = g.parameter("w_up", &[720, 2048]);
        let gate = g.matmul(h, w_gate);
        let up = g.matmul(h, w_up);
        let out = g.swiglu(gate, up);
        g.set_outputs(vec![out]);

        let (opt, report) = optimize_with_report(&g);
        let output_node = opt.node(opt.outputs()[0]);
        assert!(
            matches!(output_node.op, Op::SwiGLUConcat),
            "expected SwiGLUConcat, got {:?}",
            output_node.op
        );
        assert!(
            report
                .fusions_applied
                .iter()
                .any(|(name, _)| name.contains("SwiGLU")),
            "no SwiGLU fusion in report: {:?}",
            report.fusions_applied
        );
        // The fused matmul should have shape [50, 4096] (2*2048)
        let mm_id = output_node.inputs[0];
        let mm_node = opt.node(mm_id);
        assert!(matches!(mm_node.op, Op::MatMul));
        assert_eq!(mm_node.ty.shape, vec![50, 4096]);
    }

    /// Backward ops are encoded into egglog (not skipped).
    #[test]
    fn test_egglog_encodes_backward_ops() {
        let mut g = Graph::new();
        let x = g.input("x", &[4, 8]);
        let w = g.parameter("w", &[8, 4]);
        let at = g.add_raw_node(
            Op::MatMulAT,
            vec![x, x],
            crate::graph::TensorType::f32(vec![8, 8]),
        );
        let bt = g.add_raw_node(
            Op::MatMulBT,
            vec![x, w],
            crate::graph::TensorType::f32(vec![4, 8]),
        );
        g.set_outputs(vec![at, bt]);

        let program = graph_to_egglog(&g);
        assert!(program.contains("MatMulAT"), "MatMulAT not encoded");
        assert!(program.contains("MatMulBT"), "MatMulBT not encoded");

        let mut egraph = egglog::EGraph::default();
        egraph
            .parse_and_run_program(None, &program)
            .expect("egglog failed with backward ops");
    }

    /// E-graph discovers MatMulBT+Add → FusedMatMulBTAdd on backward ops.
    #[test]
    fn test_egglog_discovers_backward_bt_add_fusion() {
        let mut g = Graph::new();
        let grad = g.input("grad", &[4, 8]);
        let w = g.parameter("w", &[4, 8]);
        let prev = g.input("prev_grad", &[4, 4]);
        let bt = g.add_raw_node(
            Op::MatMulBT,
            vec![grad, w],
            crate::graph::TensorType::f32(vec![4, 4]),
        );
        let out = g.add(bt, prev);
        g.set_outputs(vec![out]);

        let (opt, report) = optimize_with_report(&g);
        let output_node = opt.node(opt.outputs()[0]);
        assert!(
            matches!(output_node.op, Op::FusedMatMulBTAdd),
            "expected FusedMatMulBTAdd, got {:?}",
            output_node.op
        );
        assert!(
            report
                .fusions_applied
                .iter()
                .any(|(name, _)| name.contains("BT")),
            "no BT fusion in report"
        );
    }
}
