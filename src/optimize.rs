//! E-graph optimization pass: the graph is encoded into egglog, rewrite
//! rules discover fusions under equality saturation, a traffic-aware
//! cost model extracts the cheapest equivalent term per output, and the
//! extracted terms are stamped back into the graph IR. The e-graph is
//! the single owner of every rewrite decision — there is no parallel
//! pattern-matching path.
//!
//! Scaling: saturation cost is superlinear in node count, so graphs over
//! [`SATURATION_CUTOFF`] are split into segments — repeated regions
//! (transformer layers, detected by `outline`) saturate one instance and
//! stamp the result into every instance; the remaining nodes are chunked
//! into windows under the cutoff. Every node therefore passes through
//! the e-graph exactly once. Cross-segment fusions are not discovered
//! (segment boundaries are opaque leaves) — the same limitation the
//! roadmap notes for block-boundary fusions.
//!
//! Node ids must be topologically ordered (inputs before consumers) for
//! the egglog encoding; graph builders and autodiff maintain this, and
//! `Graph::toposort` restores it after passes that append nodes.

use crate::graph::{Graph, Node, NodeId, Op, TensorType};
use egglog::{Term, TermDag, TermId, ast::Literal};
use std::collections::{HashMap, HashSet};
use std::{fmt, time::Instant};

/// Node-count ceiling for a single egglog saturation. Above this, the
/// graph is segmented (see module docs). Shared-parameter graphs create
/// large e-classes that make pattern matching superlinear: the SmolVLA
/// training graph (~750 nodes) takes minutes unsegmented.
const SATURATION_CUTOFF: usize = 300;

// ---------------------------------------------------------------------------
// HBM-traffic-aware cost model for e-graph extraction.
//
// Per-e-class tensor sizes are built after saturation by evaluating each
// graph node's binding; the cost of an e-node is then the HBM traffic it
// causes: bytes read (inputs) + bytes written (output). A fusion wins by
// exactly the intermediate traffic it eliminates — FusedMatMulAdd(a,b,d)
// saves the write and re-read of the matmul's result tensor — with no
// hand-tuned constants, and unprofitable rewrites (future: Winograd vs
// implicit GEMM, layout conversions, rematerialization) can lose on real
// numbers.
// ---------------------------------------------------------------------------

/// Cost model that prefers the expression with the least HBM traffic.
#[derive(Default, Debug, Clone)]
pub struct FusionCostModel {
    /// e-class value → tensor size in bytes.
    sizes: std::sync::Arc<HashMap<egglog::Value, u64>>,
}

impl FusionCostModel {
    /// Extraction cost is bytes read + bytes written, looked up per
    /// e-class from `sizes`.
    pub fn with_sizes(sizes: HashMap<egglog::Value, u64>) -> Self {
        Self {
            sizes: std::sync::Arc::new(sizes),
        }
    }
}

impl egglog::extract::CostModel<u64> for FusionCostModel {
    fn fold(&self, _head: &str, children_cost: &[u64], head_cost: u64) -> u64 {
        children_cost
            .iter()
            .fold(head_cost, |acc, c| acc.saturating_add(*c))
    }

    fn enode_cost(
        &self,
        _egraph: &egglog::EGraph,
        func: &egglog::Function,
        row: &egglog::FunctionRow,
    ) -> u64 {
        let name = func.name();
        // Leaves exist regardless; their bytes are charged to the ops
        // that read them.
        if name == "Leaf" {
            return 0;
        }
        // row.vals = [args.., output]. Args missing from the map are
        // non-tensor primitives (the node-id ints) and read no HBM.
        if let Some((out, args)) = row.vals.split_last()
            && let Some(&out_bytes) = self.sizes.get(out)
        {
            let read: u64 = args.iter().filter_map(|v| self.sizes.get(v)).sum();
            return read.saturating_add(out_bytes);
        }
        // Unknown output e-class (a rewrite-created tensor that no graph
        // node binds, e.g. the packed matmul inside SwiGLUPacked): fall
        // back to constants that keep fused ops preferred.
        match name {
            "FusedMatMulAdd" | "FusedMatMulATAdd" | "FusedMatMulBTAdd" | "SwiGLUPacked" => 9,
            _ => 10,
        }
    }

    fn base_value_cost(
        &self,
        _egraph: &egglog::EGraph,
        _sort: &egglog::ArcSort,
        _value: egglog::Value,
    ) -> u64 {
        // Ints embedded in constructors are not tensors.
        0
    }
}

/// Report from the e-graph optimization pass.
pub struct OptimizeReport {
    /// The egglog program text of the first segment (for inspection).
    pub egglog_program: String,
    /// Number of e-classes after saturation (summed over segments).
    pub num_eclasses: usize,
    /// Number of e-nodes after saturation (summed over segments).
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
    /// Wall-clock time for extraction + term stamping.
    pub extract_time: std::time::Duration,
    /// Repeated regions outlined for per-block saturation (0 when the
    /// whole graph fit under the saturation cutoff).
    pub outlined_regions: usize,
}

impl OptimizeReport {
    /// An empty report, for code paths that skip optimization (e.g. a
    /// cache hit) but still need to return a report.
    pub fn empty() -> Self {
        Self {
            egglog_program: String::new(),
            num_eclasses: 0,
            num_enodes: 0,
            rules_fired: Vec::new(),
            nodes_before: 0,
            nodes_after: 0,
            fusions_applied: Vec::new(),
            egglog_time: std::time::Duration::ZERO,
            extract_time: std::time::Duration::ZERO,
            outlined_regions: 0,
        }
    }
}

impl fmt::Display for OptimizeReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Optimization Report ===")?;
        writeln!(
            f,
            "Egglog saturation: {:.1}ms ({} e-classes, {} e-nodes{})",
            self.egglog_time.as_secs_f64() * 1000.0,
            self.num_eclasses,
            self.num_enodes,
            if self.outlined_regions > 0 {
                format!(", {} outlined region(s)", self.outlined_regions)
            } else {
                String::new()
            },
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

/// Run e-graph optimization and return the rewritten graph.
pub fn optimize(graph: &Graph) -> Graph {
    let (graph, _report) = optimize_with_report(graph);
    graph
}

/// Like `optimize`, but also returns a detailed report for debugging.
pub fn optimize_with_report(graph: &Graph) -> (Graph, OptimizeReport) {
    let nodes_before = graph.nodes().len();
    let mut g = clone_graph(graph);

    let segments = plan_segments(&g);
    let outlined_regions = segments.iter().filter(|s| s.shifts.len() > 1).count();

    let mut fusions: Vec<(String, u32)> = Vec::new();
    let mut index = build_structural_index(&g);
    let mut first_program = String::new();
    let mut num_eclasses = 0;
    let mut num_enodes = 0;
    let mut egglog_time = std::time::Duration::ZERO;
    let mut extract_time = std::time::Duration::ZERO;

    for seg in &segments {
        process_segment(
            &mut g,
            seg,
            &mut index,
            &mut fusions,
            &mut first_program,
            &mut num_eclasses,
            &mut num_enodes,
            &mut egglog_time,
            &mut extract_time,
        );
    }

    let dce_start = Instant::now();
    sweep_dead_nodes(&mut g);
    extract_time += dce_start.elapsed();

    let nodes_after = g
        .nodes()
        .iter()
        .filter(|n| !matches!(n.op, Op::Nop))
        .count();

    log::info!("optimizer: {} fusions on {} nodes", fusions.len(), nodes_after);
    let mut rules_fired: Vec<(String, usize)> = Vec::new();
    for fusion in &fusions {
        if let Some(entry) = rules_fired.iter_mut().find(|e| e.0 == fusion.0) {
            entry.1 += 1;
        } else {
            rules_fired.push((fusion.0.clone(), 1));
        }
    }
    for (name, count) in &rules_fired {
        log::info!("  {}x {}", count, name);
    }

    let report = OptimizeReport {
        egglog_program: first_program,
        num_eclasses,
        num_enodes,
        rules_fired,
        nodes_before,
        nodes_after,
        fusions_applied: fusions,
        egglog_time,
        extract_time,
        outlined_regions,
    };
    (g, report)
}

/// Dump the whole-graph egglog program (for standalone debugging).
/// Requires topologically-ordered node ids, like `optimize` itself.
pub fn dump_egglog_program(graph: &Graph) -> String {
    let ids: Vec<usize> = graph
        .nodes()
        .iter()
        .filter(|n| !matches!(n.op, Op::Nop))
        .map(|n| n.id as usize)
        .collect();
    let seg = Segment {
        ids,
        shifts: vec![0],
    };
    segment_program(graph, &seg).0
}

// ---------------------------------------------------------------------------
// Segmentation
// ---------------------------------------------------------------------------

/// A unit of saturation: the node ids of one encoded instance, plus the
/// id shift of every instance the extracted terms are stamped into.
/// Repeated regions have `shifts = [0, period, 2*period, ...]`; windows
/// (including a small whole graph) have `shifts = [0]`.
struct Segment {
    ids: Vec<usize>,
    shifts: Vec<usize>,
}

fn plan_segments(g: &Graph) -> Vec<Segment> {
    let n = g.nodes().len();
    let active = g
        .nodes()
        .iter()
        .filter(|n| !matches!(n.op, Op::Nop))
        .count();
    let mut segments = Vec::new();
    let mut covered = vec![false; n];
    if active > SATURATION_CUTOFF {
        for r in crate::outline::detect_repeated_regions(g) {
            for c in covered.iter_mut().skip(r.start).take(r.len()) {
                *c = true;
            }
            segments.push(Segment {
                ids: (r.start..r.start + r.period).collect(),
                shifts: (0..r.count).map(|k| k * r.period).collect(),
            });
        }
    }
    // Chunk everything not covered by a region into windows under the
    // cutoff. Fusion patterns are 2-3 nodes deep, so the coverage lost
    // at window boundaries is small; repetition-less graphs over the
    // cutoff (e.g. large ONNX imports) still get saturated this way.
    let mut window: Vec<usize> = Vec::new();
    for id in 0..n {
        if covered[id] || matches!(g.nodes()[id].op, Op::Nop) {
            continue;
        }
        window.push(id);
        if window.len() == SATURATION_CUTOFF {
            segments.push(Segment {
                ids: std::mem::take(&mut window),
                shifts: vec![0],
            });
        }
    }
    if !window.is_empty() {
        segments.push(Segment {
            ids: window,
            shifts: vec![0],
        });
    }
    segments
}

// ---------------------------------------------------------------------------
// Egglog encoding
// ---------------------------------------------------------------------------

/// The egglog sort and rewrite rules. Named constructors exist only for
/// ops that rewrite rules pattern-match on; every other op — including
/// ones added later — encodes through the arity-generic `Op1..Op6`
/// constructors, tagged with the node id so ops with different
/// attributes (eps, strides, head counts) never unify.
fn egglog_prelude(prog: &mut String) {
    prog.push_str(
        "\
(datatype Op
  (Leaf i64)
  (MatMul Op Op)
  (MatMulAT Op Op)
  (MatMulBT Op Op)
  (FusedMatMulAdd Op Op Op)
  (FusedMatMulATAdd Op Op Op)
  (FusedMatMulBTAdd Op Op Op)
  (Add Op Op)
  (Mul Op Op)
  (Relu Op)
  (Sigmoid Op)
  (Neg Op)
  (Transpose Op)
  (Silu Op)
  (SwiGLU Op Op)
  (SwiGLUPacked Op Op Op)
  (Op1 i64 Op)
  (Op2 i64 Op Op)
  (Op3 i64 Op Op Op)
  (Op4 i64 Op Op Op Op)
  (Op5 i64 Op Op Op Op Op)
  (Op6 i64 Op Op Op Op Op Op)
)

; --- Algebraic simplifications ---
(rewrite (Neg (Neg ?x)) ?x)
(rewrite (Transpose (Transpose ?x)) ?x)
(rewrite (Relu (Relu ?x)) (Relu ?x))

; --- Kernel fusion: Add(MatMul*(a,b), d) -> FusedMatMul*Add(a,b,d) ---
; Both argument orders handled explicitly (no general Add commutativity
; rule, which causes exponential blowup on large graphs).
(rewrite (Add (MatMul ?a ?b) ?d)    (FusedMatMulAdd ?a ?b ?d))
(rewrite (Add ?d (MatMul ?a ?b))    (FusedMatMulAdd ?a ?b ?d))
(rewrite (Add (MatMulAT ?a ?b) ?d)  (FusedMatMulATAdd ?a ?b ?d))
(rewrite (Add ?d (MatMulAT ?a ?b))  (FusedMatMulATAdd ?a ?b ?d))
(rewrite (Add (MatMulBT ?a ?b) ?d)  (FusedMatMulBTAdd ?a ?b ?d))
(rewrite (Add ?d (MatMulBT ?a ?b))  (FusedMatMulBTAdd ?a ?b ?d))

; --- ONNX decomposed op recognition ---
; PyTorch decomposes compound ops when exporting to ONNX. These rules
; recognize the decomposed patterns and fuse them back into compound
; kernels.

; Silu: x * sigmoid(x)
(rewrite (Mul ?x (Sigmoid ?x)) (Silu ?x))
(rewrite (Mul (Sigmoid ?x) ?x) (Silu ?x))

; SwiGLU: silu(gate) * up
(rewrite (Mul (Silu ?gate) ?up) (SwiGLU ?gate ?up))

; Packed SwiGLU: gate and up projections sharing the input become one
; wide matmul over a concatenated weight (the derived parameter is
; created at stamp time; stamping falls back to the unpacked form when
; the weights are not plain 2D parameters).
(rewrite (SwiGLU (MatMul ?h ?wg) (MatMul ?h ?wu)) (SwiGLUPacked ?h ?wg ?wu))

",
    );
    // Saturation is bounded: the deepest rewrite chain is three rules
    // (Mul(x, Sigmoid(x)) -> Silu, Mul(Silu, up) -> SwiGLU, then
    // SwiGLU(MatMul, MatMul) -> SwiGLUPacked), so three iterations reach
    // a fixpoint; the fourth is margin for future rules.
}

/// Returns the named egglog constructor for ops that rewrite rules
/// match on, or `None` for generically-encoded ops.
fn named_constructor(op: &Op) -> Option<&'static str> {
    Some(match op {
        Op::MatMul => "MatMul",
        Op::MatMulAT => "MatMulAT",
        Op::MatMulBT => "MatMulBT",
        Op::FusedMatMulAdd => "FusedMatMulAdd",
        Op::FusedMatMulATAdd => "FusedMatMulATAdd",
        Op::FusedMatMulBTAdd => "FusedMatMulBTAdd",
        Op::Add => "Add",
        Op::Mul => "Mul",
        Op::Relu => "Relu",
        Op::Sigmoid => "Sigmoid",
        Op::Neg => "Neg",
        Op::Transpose => "Transpose",
        Op::Silu => "Silu",
        Op::SwiGLU => "SwiGLU",
        _ => return None,
    })
}

fn node_to_egglog_expr(node: &Node) -> String {
    match node.op {
        Op::Input { .. } | Op::Parameter { .. } | Op::Constant { .. } => {
            format!("(Leaf {})", node.id)
        }
        Op::Nop => unreachable!("Nop nodes are filtered before encoding"),
        ref op => {
            let args: Vec<String> = node.inputs.iter().map(|i| format!("$n{}", i)).collect();
            if let Some(name) = named_constructor(op) {
                format!("({} {})", name, args.join(" "))
            } else {
                assert!(
                    !node.inputs.is_empty() && node.inputs.len() <= 6,
                    "op {:?} with {} inputs exceeds the generic egglog encoding",
                    op,
                    node.inputs.len()
                );
                format!("(Op{} {} {})", node.inputs.len(), node.id, args.join(" "))
            }
        }
    }
}

/// Egglog program for one segment instance: external dependencies become
/// opaque `Leaf` terms, segment nodes are encoded in id order. Returns
/// the program and the external node ids (needed to size their e-classes
/// for traffic-aware extraction).
fn segment_program(g: &Graph, seg: &Segment) -> (String, Vec<usize>) {
    let idset: HashSet<usize> = seg.ids.iter().copied().collect();
    let mut externals: Vec<usize> = Vec::new();
    let mut seen = HashSet::new();
    for &id in &seg.ids {
        let node = &g.nodes()[id];
        if matches!(node.op, Op::Nop) {
            continue;
        }
        for &input in &node.inputs {
            let input = input as usize;
            if !idset.contains(&input) && seen.insert(input) {
                externals.push(input);
            }
        }
    }
    externals.sort_unstable();

    let mut prog = String::new();
    egglog_prelude(&mut prog);
    for &e in &externals {
        prog.push_str(&format!("(let $n{} (Leaf {}))\n", e, e));
    }
    for &id in &seg.ids {
        let node = &g.nodes()[id];
        if matches!(node.op, Op::Nop) {
            continue;
        }
        prog.push_str(&format!("(let $n{} {})\n", id, node_to_egglog_expr(node)));
    }
    // See the comment at the end of `egglog_prelude` for the bound.
    prog.push_str("(run 4)\n");
    (prog, externals)
}

/// Map every node binding (`$n{id}`) to its e-class value and record the
/// tensor's size in bytes — the lookup table for traffic-aware
/// extraction. Nodes sharing an e-class denote the same tensor, so the
/// insert is idempotent; rewrite-created terms (e.g. FusedMatMulAdd)
/// join the e-class of the expression they replaced and need no entry
/// of their own.
fn eclass_sizes(
    graph: &Graph,
    egraph: &mut egglog::EGraph,
    ids: impl Iterator<Item = usize>,
) -> HashMap<egglog::Value, u64> {
    let mut sizes = HashMap::new();
    for id in ids {
        let node = &graph.nodes()[id];
        if matches!(node.op, Op::Nop) {
            continue;
        }
        let var = format!("$n{}", node.id);
        if let Ok((_sort, value)) =
            egraph.eval_expr(&egglog::ast::Expr::Var(egglog::ast::Span::Panic, var))
        {
            sizes.insert(value, node.ty.size_bytes() as u64);
        }
    }
    sizes
}

// ---------------------------------------------------------------------------
// Extraction + stamping
// ---------------------------------------------------------------------------

/// Extraction roots of a segment, as encoded-instance node ids: any node
/// whose value escapes its own instance (consumed outside it, or a graph
/// output), unioned across all instances.
fn segment_roots(g: &Graph, seg: &Segment) -> Vec<usize> {
    let base: HashSet<usize> = seg.ids.iter().copied().collect();
    let mut roots: HashSet<usize> = HashSet::new();
    for &shift in &seg.shifts {
        let inst: HashSet<usize> = base.iter().map(|&i| i + shift).collect();
        for node in g.nodes() {
            if matches!(node.op, Op::Nop) {
                continue;
            }
            if !inst.contains(&(node.id as usize)) {
                for &input in &node.inputs {
                    let input = input as usize;
                    if inst.contains(&input) {
                        roots.insert(input - shift);
                    }
                }
            }
        }
        for &out in g.outputs() {
            let out = out as usize;
            if inst.contains(&out) {
                roots.insert(out - shift);
            }
        }
    }
    let mut v: Vec<usize> = roots.into_iter().collect();
    v.sort_unstable();
    v
}

/// Where each external leaf is read by the encoded instance:
/// ext id → list of (position in `seg.ids`, input slot). Used to
/// translate externals per instance via the instance's actual edges.
fn external_uses(g: &Graph, seg: &Segment) -> HashMap<usize, Vec<(usize, usize)>> {
    let idset: HashSet<usize> = seg.ids.iter().copied().collect();
    let mut uses: HashMap<usize, Vec<(usize, usize)>> = HashMap::new();
    for (pos, &id) in seg.ids.iter().enumerate() {
        let node = &g.nodes()[id];
        if matches!(node.op, Op::Nop) {
            continue;
        }
        for (slot, &input) in node.inputs.iter().enumerate() {
            let input = input as usize;
            if !idset.contains(&input) {
                uses.entry(input).or_default().push((pos, slot));
            }
        }
    }
    uses
}

/// External-leaf translation for one instance: read the instance's own
/// edges at the recorded use sites. Returns `None` (skip the instance)
/// if the sites disagree — a node used both as a shifting chain edge and
/// a shared global, which edge isomorphism allows in principle.
fn instance_ext_map(
    g: &Graph,
    seg: &Segment,
    uses: &HashMap<usize, Vec<(usize, usize)>>,
    shift: usize,
) -> Option<HashMap<usize, NodeId>> {
    let mut map = HashMap::new();
    for (&ext, sites) in uses {
        let mut val: Option<NodeId> = None;
        for &(pos, slot) in sites {
            let v = g.nodes()[seg.ids[pos] + shift].inputs[slot];
            match val {
                Some(prev) if prev != v => return None,
                _ => val = Some(v),
            }
        }
        map.insert(ext, val.unwrap());
    }
    Some(map)
}

#[allow(clippy::too_many_arguments)]
fn process_segment(
    g: &mut Graph,
    seg: &Segment,
    index: &mut HashMap<(&'static str, Vec<NodeId>), NodeId>,
    fusions: &mut Vec<(String, u32)>,
    first_program: &mut String,
    num_eclasses: &mut usize,
    num_enodes: &mut usize,
    egglog_time: &mut std::time::Duration,
    extract_time: &mut std::time::Duration,
) {
    let egglog_start = Instant::now();
    let (program, externals) = segment_program(g, seg);
    if first_program.is_empty() {
        first_program.clone_from(&program);
    }
    let mut egraph = egglog::EGraph::default();
    if let Err(e) = egraph.parse_and_run_program(None, &program) {
        log::warn!(
            "egglog failed on segment of {} nodes: {} — leaving it unoptimized",
            seg.ids.len(),
            e
        );
        *egglog_time += egglog_start.elapsed();
        return;
    }
    let size_ids = externals.iter().copied().chain(seg.ids.iter().copied());
    let cm = FusionCostModel::with_sizes(eclass_sizes(g, &mut egraph, size_ids));

    let roots = segment_roots(g, seg);
    let mut terms: Vec<(usize, TermDag, TermId)> = Vec::new();
    for &root in &roots {
        let var = format!("$n{}", root);
        match egraph.eval_expr(&egglog::ast::Expr::Var(egglog::ast::Span::Panic, var)) {
            Ok((sort, value)) => {
                match egraph.extract_value_with_cost_model(&sort, value, cm.clone()) {
                    Ok((dag, term_id, cost)) => {
                        log::debug!(
                            "extracted $n{} (cost {}): {}",
                            root,
                            cost,
                            dag.to_string(term_id)
                        );
                        terms.push((root, dag, term_id));
                    }
                    Err(e) => log::warn!("extraction failed for $n{}: {}", root, e),
                }
            }
            Err(e) => log::warn!("failed to eval $n{}: {}", root, e),
        }
    }
    let serialized = egraph.serialize(egglog::SerializeConfig::default());
    *num_eclasses += serialized.egraph.class_data.len();
    *num_enodes += serialized.egraph.nodes.len();
    *egglog_time += egglog_start.elapsed();

    // Stamping. All instance translations are computed before any
    // mutation: stamping overwrites root inputs, which may be the very
    // edges the translation reads.
    let stamp_start = Instant::now();
    let uses = external_uses(g, seg);
    let ext_maps: Vec<Option<HashMap<usize, NodeId>>> = seg
        .shifts
        .iter()
        .map(|&shift| instance_ext_map(g, seg, &uses, shift))
        .collect();
    let idset: HashSet<usize> = seg.ids.iter().copied().collect();
    for (&shift, ext_map) in seg.shifts.iter().zip(&ext_maps) {
        let Some(ext_map) = ext_map else {
            log::warn!(
                "segment instance at +{} has ambiguous external edges — left unoptimized",
                shift
            );
            continue;
        };
        for (root, dag, term_id) in &terms {
            let mut stamper = Stamper {
                g,
                index,
                seg_ids: &idset,
                shift,
                ext_map,
                fusions,
                memo: HashMap::new(),
            };
            if let Err(e) = stamper.stamp_root(root + shift, dag, *term_id) {
                log::warn!("stamping $n{} (+{}) failed: {}", root, shift, e);
            }
        }
    }
    *extract_time += stamp_start.elapsed();
}

/// Rebuilds extracted terms in the graph IR. Interior nodes whose
/// children were rewritten are mutated in place (the new inputs are
/// value-equivalent, so every consumer — and every id-carrying attribute
/// like `fwd_node` — stays valid); new fused nodes are appended; roots
/// are overwritten in place so their node ids, types, and output status
/// survive.
struct Stamper<'a> {
    g: &'a mut Graph,
    /// Structural memo for named constructors: (name, children) → node.
    index: &'a mut HashMap<(&'static str, Vec<NodeId>), NodeId>,
    /// Node ids of the encoded instance (terms only reference these).
    seg_ids: &'a HashSet<usize>,
    /// Id shift of the instance being stamped.
    shift: usize,
    /// External-leaf translation for this instance.
    ext_map: &'a HashMap<usize, NodeId>,
    fusions: &'a mut Vec<(String, u32)>,
    /// Per-(dag, instance) term resolution cache.
    memo: HashMap<TermId, NodeId>,
}

impl Stamper<'_> {
    /// Overwrite the root node in place with the extracted term.
    fn stamp_root(&mut self, root: usize, dag: &TermDag, term_id: TermId) -> Result<(), String> {
        match dag.get(term_id).clone() {
            Term::App(ref name, ref children) if named_constructor_exists(name) => {
                let inputs = self.resolve_children(dag, children)?;
                // Unchanged term → nothing to do.
                if named_constructor(&self.g.node(root as u32).op) == Some(name.as_str())
                    && self.g.node(root as u32).inputs == inputs
                {
                    return Ok(());
                }
                self.build_named(name, inputs, Some(root as u32))?;
                Ok(())
            }
            _ => {
                // Generic op (in-place child rewrite), or a leaf/other
                // node the root's e-class collapsed into (Neg∘Neg → x):
                // alias the root to it. The resolved node always has a
                // smaller id (leaves and pristine nodes precede the
                // root topologically), so compile's Identity buffer
                // aliasing sees its input already allocated.
                let resolved = self.resolve(dag, term_id)?;
                if resolved as usize != root {
                    self.g.nodes_mut()[root].op = Op::Identity;
                    self.g.nodes_mut()[root].inputs = vec![resolved];
                }
                Ok(())
            }
        }
    }

    fn resolve_children(&mut self, dag: &TermDag, children: &[TermId]) -> Result<Vec<NodeId>, String> {
        children.iter().map(|&c| self.resolve(dag, c)).collect()
    }

    fn resolve(&mut self, dag: &TermDag, term_id: TermId) -> Result<NodeId, String> {
        if let Some(&id) = self.memo.get(&term_id) {
            return Ok(id);
        }
        let id = match dag.get(term_id).clone() {
            Term::App(ref name, ref children) if name == "Leaf" => {
                self.translate(lit_node_id(dag, children[0])?)?
            }
            Term::App(ref name, ref children) if name.starts_with("Op") => {
                let orig = self.translate(lit_node_id(dag, children[0])?)?;
                let inputs = self.resolve_children(dag, &children[1..])?;
                if self.g.node(orig).inputs != inputs {
                    // Children were rewritten: point this node at the
                    // equivalent producers. Op, attributes, and type are
                    // untouched.
                    self.g.nodes_mut()[orig as usize].inputs = inputs;
                }
                orig
            }
            Term::App(ref name, ref children) => {
                let inputs = self.resolve_children(dag, children)?;
                match self.index.get(&(static_constructor(name)?, inputs.clone())) {
                    Some(&hit) => hit,
                    None => self.build_named(name, inputs, None)?,
                }
            }
            other => return Err(format!("unexpected term {:?}", other)),
        };
        self.memo.insert(term_id, id);
        Ok(id)
    }

    /// Translate an encoded-instance node id to this instance.
    fn translate(&self, raw: usize) -> Result<NodeId, String> {
        if self.seg_ids.contains(&raw) {
            Ok((raw + self.shift) as NodeId)
        } else {
            self.ext_map
                .get(&raw)
                .copied()
                .ok_or_else(|| format!("no translation for external node {}", raw))
        }
    }

    /// Create (or overwrite `target` with) a named-constructor node.
    fn build_named(
        &mut self,
        name: &str,
        inputs: Vec<NodeId>,
        target: Option<NodeId>,
    ) -> Result<NodeId, String> {
        if name == "SwiGLUPacked" {
            return self.build_swiglu_packed(&inputs, target);
        }
        let shape = |id: NodeId| self.g.node(id).ty.shape.clone();
        let ty_of = |id: NodeId| self.g.node(id).ty.clone();
        let rank2 = |id: NodeId| {
            let s = shape(id);
            if s.len() == 2 {
                Ok(s)
            } else {
                Err(format!("{} needs rank-2 operands, got {:?}", name, s))
            }
        };
        let (op, ty, label) = match name {
            "MatMul" => {
                let (a, b) = (rank2(inputs[0])?, rank2(inputs[1])?);
                (Op::MatMul, TensorType::f32(vec![a[0], b[1]]), None)
            }
            "MatMulAT" => {
                let (a, b) = (rank2(inputs[0])?, rank2(inputs[1])?);
                (Op::MatMulAT, TensorType::f32(vec![a[1], b[1]]), None)
            }
            "MatMulBT" => {
                let (a, b) = (rank2(inputs[0])?, rank2(inputs[1])?);
                (Op::MatMulBT, TensorType::f32(vec![a[0], b[0]]), None)
            }
            // The addend has the result shape by Add's own typing.
            "FusedMatMulAdd" => (Op::FusedMatMulAdd, ty_of(inputs[2]), Some("MatMul+Add→FusedMatMulAdd")),
            "FusedMatMulATAdd" => (Op::FusedMatMulATAdd, ty_of(inputs[2]), Some("MatMulAT+Add→FusedMatMulATAdd")),
            "FusedMatMulBTAdd" => (Op::FusedMatMulBTAdd, ty_of(inputs[2]), Some("MatMulBT+Add→FusedMatMulBTAdd")),
            "Add" => (Op::Add, ty_of(inputs[0]), None),
            "Mul" => (Op::Mul, ty_of(inputs[0]), None),
            "Relu" => (Op::Relu, ty_of(inputs[0]), None),
            "Sigmoid" => (Op::Sigmoid, ty_of(inputs[0]), None),
            "Neg" => (Op::Neg, ty_of(inputs[0]), None),
            "Transpose" => {
                let mut s = shape(inputs[0]);
                s.reverse();
                (Op::Transpose, TensorType::f32(s), None)
            }
            "Silu" => (Op::Silu, ty_of(inputs[0]), Some("Mul+Sigmoid→Silu")),
            "SwiGLU" => (Op::SwiGLU, ty_of(inputs[0]), Some("Silu+Mul→SwiGLU")),
            other => return Err(format!("unknown constructor {}", other)),
        };
        let id = self.place(op, inputs.clone(), ty, target);
        self.index.insert((static_constructor(name)?, inputs), id);
        if let Some(label) = label {
            self.fusions.push((label.to_string(), id));
        }
        Ok(id)
    }

    /// SwiGLU(MatMul(h, wg), MatMul(h, wu)) → SwiGLUConcat(MatMul(h, wg|wu))
    /// with a derived concatenated-weight parameter. Falls back to the
    /// equivalent unpacked form when the weights are not plain same-shape
    /// 2D parameters (e.g. ONNX constants).
    fn build_swiglu_packed(
        &mut self,
        inputs: &[NodeId],
        target: Option<NodeId>,
    ) -> Result<NodeId, String> {
        let (h, wg, wu) = (inputs[0], inputs[1], inputs[2]);
        let packable = {
            let (g_node, u_node) = (self.g.node(wg), self.g.node(wu));
            matches!(g_node.op, Op::Parameter { .. })
                && matches!(u_node.op, Op::Parameter { .. })
                && g_node.ty.shape.len() == 2
                && g_node.ty.shape == u_node.ty.shape
                && g_node.ty.dtype == u_node.ty.dtype
                && self.g.node(h).ty.shape.len() == 2
        };
        if !packable {
            let gate = self.lookup_or_build("MatMul", vec![h, wg])?;
            let up = self.lookup_or_build("MatMul", vec![h, wu])?;
            return self.build_named("SwiGLU", vec![gate, up], target);
        }
        let param_name = |id: NodeId| match self.g.node(id).op {
            Op::Parameter { ref name } => name.clone(),
            _ => unreachable!(),
        };
        let (gate_name, up_name) = (param_name(wg), param_name(wu));
        let in_features = self.g.node(wg).ty.shape[0];
        let out_features = self.g.node(wg).ty.shape[1];
        let m = self.g.node(h).ty.shape[0];
        let concat_name = format!("{}+{}", gate_name, up_name);
        // Record the derivation so the runtime fills the packed buffer
        // from the original parameters.
        self.g.derived_params.push(crate::graph::DerivedParam {
            name: concat_name.clone(),
            sources: vec![(gate_name, out_features), (up_name, out_features)],
            rows: in_features,
            transform: crate::graph::ParamTransform::HorizontalConcat,
        });
        let concat_dtype = self.g.node(wg).ty.dtype;
        let concat_w = self.g.add_raw_node(
            Op::Parameter { name: concat_name },
            vec![],
            TensorType::new(vec![in_features, 2 * out_features], concat_dtype),
        );
        let wide_mm = self.g.add_raw_node(
            Op::MatMul,
            vec![h, concat_w],
            TensorType::f32(vec![m, 2 * out_features]),
        );
        let id = self.place(
            Op::SwiGLUConcat,
            vec![wide_mm],
            TensorType::f32(vec![m, out_features]),
            target,
        );
        self.index.insert(("SwiGLUPacked", inputs.to_vec()), id);
        self.fusions.push((
            "SwiGLU(MatMul,MatMul)→SwiGLUConcat(MatMul)".to_string(),
            id,
        ));
        Ok(id)
    }

    fn lookup_or_build(&mut self, name: &str, inputs: Vec<NodeId>) -> Result<NodeId, String> {
        match self.index.get(&(static_constructor(name)?, inputs.clone())) {
            Some(&hit) => Ok(hit),
            None => self.build_named(name, inputs, None),
        }
    }

    /// Write the node into `target` (root stamping keeps the root's id
    /// and type) or append a new node.
    fn place(&mut self, op: Op, inputs: Vec<NodeId>, ty: TensorType, target: Option<NodeId>) -> NodeId {
        match target {
            Some(id) => {
                let node = &mut self.g.nodes_mut()[id as usize];
                node.op = op;
                node.inputs = inputs;
                // ty deliberately untouched: same e-class, same tensor.
                id
            }
            None => self.g.add_raw_node(op, inputs, ty),
        }
    }
}

fn named_constructor_exists(name: &str) -> bool {
    static_constructor(name).is_ok()
}

/// Interns a constructor name to the `'static` string used as the
/// structural-index key.
fn static_constructor(name: &str) -> Result<&'static str, String> {
    Ok(match name {
        "MatMul" => "MatMul",
        "MatMulAT" => "MatMulAT",
        "MatMulBT" => "MatMulBT",
        "FusedMatMulAdd" => "FusedMatMulAdd",
        "FusedMatMulATAdd" => "FusedMatMulATAdd",
        "FusedMatMulBTAdd" => "FusedMatMulBTAdd",
        "Add" => "Add",
        "Mul" => "Mul",
        "Relu" => "Relu",
        "Sigmoid" => "Sigmoid",
        "Neg" => "Neg",
        "Transpose" => "Transpose",
        "Silu" => "Silu",
        "SwiGLU" => "SwiGLU",
        "SwiGLUPacked" => "SwiGLUPacked",
        other => return Err(format!("unknown constructor {}", other)),
    })
}

fn lit_node_id(dag: &TermDag, term_id: TermId) -> Result<usize, String> {
    match dag.get(term_id) {
        Term::Lit(Literal::Int(v)) => Ok(*v as usize),
        other => Err(format!("expected node-id literal, got {:?}", other)),
    }
}

/// Structural memo of the existing graph for named constructors, so
/// term resolution finds each instance's own nodes (and never duplicates
/// an existing equivalent node).
fn build_structural_index(g: &Graph) -> HashMap<(&'static str, Vec<NodeId>), NodeId> {
    let mut index = HashMap::new();
    for node in g.nodes() {
        if let Some(name) = named_constructor(&node.op) {
            index.insert((name, node.inputs.clone()), node.id);
        }
    }
    index
}

/// Nop out nodes no longer reachable from any output. Parameters and
/// inputs are kept even when dead — `set_parameter`/`set_input` address
/// them by name (e.g. the packed-SwiGLU sources feed derived params).
/// CacheWrite executes for its side effect (it mutates the cache buffer
/// in place; decode graphs never read its result), so it is a root.
fn sweep_dead_nodes(g: &mut Graph) {
    let n = g.nodes().len();
    let mut live = vec![false; n];
    let mut stack: Vec<usize> = g.outputs().iter().map(|&o| o as usize).collect();
    stack.extend(
        g.nodes()
            .iter()
            .filter(|node| matches!(node.op, Op::CacheWrite))
            .map(|node| node.id as usize),
    );
    while let Some(id) = stack.pop() {
        if live[id] {
            continue;
        }
        live[id] = true;
        stack.extend(g.nodes()[id].inputs.iter().map(|&i| i as usize));
    }
    for id in 0..n {
        let node = &mut g.nodes_mut()[id];
        if !live[id] && !matches!(node.op, Op::Nop | Op::Parameter { .. } | Op::Input { .. }) {
            node.op = Op::Nop;
            node.inputs.clear();
        }
    }
}

// ---------------------------------------------------------------------------
// Direct graph passes outside the e-graph (inference-only rewrites that
// the training path cannot differentiate through).
// ---------------------------------------------------------------------------

/// Fuse Silu(GroupNorm(x, w, b)) → GroupNormSilu(x, w, b)
///
/// Only fuses if the GroupNorm result is used exclusively by this Silu.
/// This is inference-only (backward pass can't differentiate through the fused op).
pub fn apply_group_norm_silu_fusions(graph: &mut Graph, fusions: &mut Vec<(String, u32)>) {
    let node_ids: Vec<usize> = (0..graph.nodes().len()).collect();
    for &id in &node_ids {
        let node = &graph.nodes()[id];
        if !matches!(node.op, Op::Silu) {
            continue;
        }
        let gn_id = node.inputs[0];
        let gn_node = graph.node(gn_id);
        let (num_groups, eps, channels, spatial) = match gn_node.op {
            Op::GroupNorm {
                num_groups,
                eps,
                channels,
                spatial,
            } => (num_groups, eps, channels, spatial),
            _ => continue,
        };
        // Only fuse if GroupNorm has a single consumer
        let gn_use_count = graph
            .nodes()
            .iter()
            .filter(|n| n.inputs.contains(&gn_id) && !matches!(n.op, Op::Nop))
            .count();
        if gn_use_count != 1 {
            continue;
        }
        if graph.is_output(gn_id) {
            continue;
        }
        let (x, w, b) = (gn_node.inputs[0], gn_node.inputs[1], gn_node.inputs[2]);
        // Rewrite Silu node to GroupNormSilu
        graph.nodes_mut()[id].op = Op::GroupNormSilu {
            num_groups,
            eps,
            channels,
            spatial,
        };
        graph.nodes_mut()[id].inputs = vec![x, w, b];
        // Mark old GroupNorm as Nop
        graph.nodes_mut()[gn_id as usize].op = Op::Nop;
        fusions.push(("GroupNorm+Silu→GroupNormSilu".to_string(), id as u32));
    }
}

/// Rewrite Conv2d(3×3, stride=1) → WinogradConv2d with pre-transformed weights.
///
/// For each matching Conv2d node, creates a derived parameter for the Winograd-transformed
/// weights and rewrites the node to WinogradConv2d.
pub fn apply_winograd_conv_fusions(graph: &mut Graph, fusions: &mut Vec<(String, u32)>) {
    let node_ids: Vec<usize> = (0..graph.nodes().len()).collect();
    for &id in &node_ids {
        let node = &graph.nodes()[id];
        let (in_channels, in_h, in_w, out_channels, kernel_h, kernel_w, stride, padding) =
            match node.op {
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
                    ..
                } => {
                    // Winograd F(2,3) is a 3×3 stride-1 specialization. Only
                    // applies when padding_h == padding_w (symmetric).
                    if padding_h != padding_w {
                        continue;
                    }
                    (
                        in_channels,
                        in_h,
                        in_w,
                        out_channels,
                        kernel_h,
                        kernel_w,
                        stride,
                        padding_h,
                    )
                }
                _ => continue,
            };
        // Only match 3×3 stride-1 convolutions with enough channels to
        // amortize transform overhead (input/output transforms are O(tiles)
        // while matmul savings are O(tiles × Ci)).
        if kernel_h != 3 || kernel_w != 3 || stride != 1 {
            continue;
        }
        if (in_channels * out_channels) < 4096 {
            continue; // too small, GEMM is faster
        }

        let weight_id = node.inputs[1];
        let weight_name = match graph.node(weight_id).op {
            Op::Parameter { ref name } => name.clone(),
            _ => continue,
        };
        let input_id = node.inputs[0];

        // Create Winograd weight parameter name
        let wino_name = format!("{}:winograd", weight_name);

        // Record derivation so runtime can fill this from original weights
        graph.derived_params.push(crate::graph::DerivedParam {
            name: wino_name.clone(),
            sources: vec![(weight_name, (out_channels * in_channels * 9) as usize)],
            rows: 1, // not used for Winograd
            transform: crate::graph::ParamTransform::Winograd3x3 {
                out_channels: out_channels as usize,
                in_channels: in_channels as usize,
            },
        });

        // Create new parameter node for Winograd-transformed weights [16 * Co * Ci]
        let wino_size = 16 * out_channels as usize * in_channels as usize;
        let wino_param = graph.add_raw_node(
            Op::Parameter { name: wino_name },
            vec![],
            TensorType::f32(vec![wino_size]),
        );

        // Rewrite Conv2d → WinogradConv2d
        // Keep original weight as 3rd input for backward pass (grad_input/grad_weight)
        graph.nodes_mut()[id].op = Op::WinogradConv2d {
            in_channels,
            in_h,
            in_w,
            out_channels,
            padding,
        };
        graph.nodes_mut()[id].inputs = vec![input_id, wino_param, weight_id];

        fusions.push(("Conv2d(3x3)→WinogradConv2d".to_string(), id as u32));
    }
}

fn clone_graph(graph: &Graph) -> Graph {
    let mut new_graph = Graph::new();
    for node in graph.nodes() {
        new_graph.add_raw_node(node.op.clone(), node.inputs.clone(), node.ty.clone());
    }
    let num_user = graph.num_user_outputs();
    new_graph.set_outputs(graph.outputs()[..num_user].to_vec());
    new_graph.append_param_grad_outputs(&graph.outputs()[num_user..]);
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

        let program = dump_egglog_program(&g);
        assert!(program.contains("(MatMul"));
        assert!(program.contains("(Leaf 0)"));

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
            if let egglog::CommandOutput::ExtractBest(ref dag, _cost, term_id) = *out {
                let s = dag.to_string(term_id);
                eprintln!("egglog extracted: {}", s);
                assert!(
                    s.contains("FusedMatMulAdd"),
                    "expected FusedMatMulAdd, got: {}",
                    s
                );
                // Verify the term tree structure
                match dag.get(term_id).clone() {
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

    /// Neg(Neg(x)) at the output collapses to an alias of x.
    #[test]
    fn test_double_neg_collapses() {
        let mut g = Graph::new();
        let a = g.input("a", &[4, 8]);
        let b = g.input("b", &[4, 8]);
        let sum = g.add(a, b);
        let n1 = g.neg(sum);
        let n2 = g.neg(n1);
        g.set_outputs(vec![n2]);

        let opt = optimize(&g);
        // The root is rewritten in place to the collapsed expression
        // (either directly as the Add or as an alias of it).
        let out = opt.node(opt.outputs()[0]);
        assert!(
            matches!(out.op, Op::Add | Op::Identity),
            "expected collapsed root, got {:?}",
            out.op
        );
        // The Neg pair is dead.
        let negs = opt.nodes().iter().filter(|n| matches!(n.op, Op::Neg)).count();
        assert_eq!(negs, 0, "dead Neg nodes should be swept");
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
        assert!(program.contains("(run 4)"));
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

        let program = dump_egglog_program(&g);
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
                .any(|entry| entry.0.contains("SwiGLU")),
            "no SwiGLU fusion in report: {:?}",
            report.fusions_applied
        );
        // The fused matmul should have shape [50, 4096] (2*2048)
        let mm_id = output_node.inputs[0];
        let mm_node = opt.node(mm_id);
        assert!(matches!(mm_node.op, Op::MatMul));
        assert_eq!(mm_node.ty.shape, vec![50, 4096]);
        assert_eq!(opt.derived_params.len(), 1);
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

        let program = dump_egglog_program(&g);
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
                .any(|entry| entry.0.contains("BT")),
            "no BT fusion in report"
        );
    }

    /// E-graph recognizes x * sigmoid(x) → Silu(x).
    #[test]
    fn test_silu_fusion() {
        let mut g = Graph::new();
        let x = g.input("x", &[4, 8]);
        let sig = g.sigmoid(x);
        let out = g.mul(x, sig);
        g.set_outputs(vec![out]);

        let (opt, report) = optimize_with_report(&g);
        // The output should now be Silu
        let has_silu = opt.nodes().iter().any(|n| matches!(n.op, Op::Silu));
        assert!(
            has_silu,
            "expected Silu fusion, got nodes: {:?}",
            opt.nodes()
                .iter()
                .map(|n| format!("{:?}", n.op))
                .collect::<Vec<_>>()
        );
        assert!(
            !report.fusions_applied.is_empty() || has_silu,
            "no Silu fusion detected"
        );
    }

    /// Pattern recognition of decomposed Silu+Mul → SwiGLU.
    #[test]
    fn test_swiglu_from_decomposed() {
        let mut g = Graph::new();
        let gate = g.input("gate", &[4, 8]);
        let up = g.input("up", &[4, 8]);
        // Decomposed SwiGLU: silu(gate) * up
        let sig = g.sigmoid(gate);
        let silu = g.mul(gate, sig);
        let out = g.mul(silu, up);
        g.set_outputs(vec![out]);

        let (opt, _report) = optimize_with_report(&g);
        let has_swiglu = opt.nodes().iter().any(|n| matches!(n.op, Op::SwiGLU));
        assert!(
            has_swiglu,
            "expected SwiGLU fusion from decomposed silu*up, got nodes: {:?}",
            opt.nodes()
                .iter()
                .map(|n| format!("{:?}", n.op))
                .collect::<Vec<_>>()
        );
    }

    /// MaxPool2d and GlobalAvgPool survive e-graph optimization unchanged.
    #[test]
    fn test_pool_ops_roundtrip() {
        let mut g = Graph::new();
        let x = g.input("x", &[64 * 8 * 8]);
        let pool = g.max_pool_2d(x, 1, 64, 8, 8, 2, 2, 2, 0);
        let gap = g.global_avg_pool(pool, 1, 64, 16);
        g.set_outputs(vec![gap]);

        let (opt, _report) = optimize_with_report(&g);
        let has_maxpool = opt
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::MaxPool2d { .. }));
        let has_gap = opt
            .nodes()
            .iter()
            .any(|n| matches!(n.op, Op::GlobalAvgPool { .. }));
        assert!(has_maxpool, "MaxPool2d should survive optimization");
        assert!(has_gap, "GlobalAvgPool should survive optimization");
    }

    /// Fusion fires even when the fused-away producer has a second
    /// consumer: the producer stays alive for that consumer and the
    /// fused node feeds the rest. Both paths must survive DCE.
    #[test]
    fn test_shared_producer_keeps_both_paths() {
        let mut g = Graph::new();
        let x = g.input("x", &[4, 8]);
        let w = g.parameter("w", &[8, 8]);
        let d = g.input("d", &[4, 8]);
        let mm = g.matmul(x, w);
        let fused_path = g.add(mm, d);
        let other_path = g.relu(mm);
        g.set_outputs(vec![fused_path, other_path]);

        let opt = optimize(&g);
        let out0 = opt.node(opt.outputs()[0]);
        assert!(matches!(out0.op, Op::FusedMatMulAdd));
        let out1 = opt.node(opt.outputs()[1]);
        assert!(matches!(out1.op, Op::Relu));
        // The shared MatMul must still be live for the Relu path.
        assert!(matches!(opt.node(out1.inputs[0]).op, Op::MatMul));
    }
}

