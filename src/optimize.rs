//! Graph optimization through bounded equality saturation. Ordinary construction
//! extracts a deterministic candidate; calibrated construction retains alternatives
//! until their complete implementations can be measured.
//!
//! In outlined mode, graphs over `SATURATION_CUTOFF` are split into segments:
//! repeated regions
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
pub(crate) mod search;
use egglog::{Term, TermDag, TermId, ast::Literal, extract::Extractor};
use std::collections::{HashMap, HashSet};
use std::{fmt, time::Instant};

/// Node-count ceiling for a single egglog saturation. Above this, the
/// graph is segmented (see module docs). Shared-parameter graphs create
/// large e-classes that make pattern matching superlinear: the SmolVLA
/// training graph (~750 nodes) takes minutes unsegmented.
pub(crate) const SATURATION_CUTOFF: usize = 300;

/// Rewrite strategy used by the graph optimizer.
///
/// Outlined saturation is the production strategy. Windowed and whole-graph
/// modes expose the region-boundary tradeoff for compiler diagnostics.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum OptimizeMode {
    /// Preserve the graph as written (apart from dead-code elimination).
    Off,
    /// Run equality saturation in fixed-size windows, without outlining.
    EgglogWindowed,
    /// Outline repeated regions, then saturate regions and residual windows.
    EgglogOutlined,
    /// Saturate the complete graph in one e-graph. This can scale poorly.
    EgglogWhole,
}

impl OptimizeMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::EgglogWindowed => "egglog-windowed",
            Self::EgglogOutlined => "egglog-outlined",
            Self::EgglogWhole => "egglog-whole",
        }
    }
}

/// Objective used when extracting a representative from each e-class.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ExtractionCost {
    /// Minimize expression-tree nodes.
    AstSize,
    /// Minimize estimated tensor bytes read and written.
    TensorTraffic,
}

impl ExtractionCost {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::AstSize => "ast-size",
            Self::TensorTraffic => "tensor-traffic",
        }
    }
}

/// Configuration for graph-rewrite ablations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct OptimizeConfig {
    pub mode: OptimizeMode,
    pub extraction_cost: ExtractionCost,
    /// Maximum nodes in an outlined region or residual window.
    pub saturation_cutoff: usize,
    /// Skip the Conv2d-to-Winograd rewrite.
    ///
    /// The selection heuristic weighs channel counts only, so a workload
    /// with few channels over a large image sits near its boundary; this
    /// makes which side it should be on measurable without a rebuild.
    pub no_winograd: bool,
    /// Pack SwiGLU projections into one derived parameter buffer.
    #[serde(alias = "greedy_pack_swiglu")]
    pub pack_swiglu: bool,
}

impl Default for OptimizeConfig {
    fn default() -> Self {
        Self {
            mode: OptimizeMode::EgglogOutlined,
            extraction_cost: ExtractionCost::TensorTraffic,
            saturation_cutoff: SATURATION_CUTOFF,
            no_winograd: false,
            pack_swiglu: true,
        }
    }
}

// ---------------------------------------------------------------------------
// Logical tensor traffic for e-graph extraction: input bytes plus output
// bytes, not measured HBM transactions. This tree estimate ignores cache
// reuse, shared subexpressions, arithmetic and occupancy. It orders bounded
// exploration; measured construction compares the lowered implementations.
// ---------------------------------------------------------------------------

/// Cost model that prefers the expression with the least logical tensor traffic.
#[derive(Default, Debug, Clone)]
pub struct FusionCostModel {
    /// e-class value → tensor size in bytes.
    sizes: Option<std::sync::Arc<HashMap<egglog::Value, u64>>>,
}

impl FusionCostModel {
    /// Every non-leaf e-node costs one. Recursive folding therefore
    /// minimizes the extracted expression-tree size.
    pub fn ast_size() -> Self {
        Self { sizes: None }
    }

    /// Extraction cost is bytes read + bytes written, looked up per
    /// e-class from `sizes`.
    pub fn with_sizes(sizes: HashMap<egglog::Value, u64>) -> Self {
        Self {
            sizes: Some(std::sync::Arc::new(sizes)),
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
        // Tile variants are equal implementations. Their cost is a placeholder
        // so extraction can enumerate them; measured search decides.
        if scheduled_matmul(name).is_some() {
            return 1;
        }
        let Some(sizes) = self.sizes.as_ref() else {
            return 1;
        };
        // Values are sort-local: an integer node id can have the same raw
        // value as an unrelated tensor e-class. Only Op arguments read tensors.
        if let Some((out, args)) = row.vals.split_last()
            && let Some(&out_bytes) = sizes.get(out)
        {
            let read = args
                .iter()
                .zip(&func.schema().input)
                .filter(|&(_, sort)| sort.name() == "Op")
                .filter_map(|(value, _)| sizes.get(value))
                .fold(0u64, |total, bytes| total.saturating_add(*bytes));
            return read.saturating_add(out_bytes);
        }
        // Unbound outputs use a structural fallback, with fused ops preferred.
        match name {
            "FusedMatMulAdd" | "FusedMatMulATAdd" | "FusedMatMulBTAdd" | "SwiGLUPacked"
            | "GeGLUPacked" | "SwiGLUPackedBT" | "GeGLUPackedBT" => 9,
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
    /// Rewrite strategy used for this pass.
    pub mode: OptimizeMode,
    /// Extraction objective used for this pass.
    pub extraction_cost: ExtractionCost,
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
    /// Encoding, egglog parsing/saturation, extraction and e-graph statistics.
    pub egglog_time: std::time::Duration,
    /// Term stamping and dead-code elimination.
    pub extract_time: std::time::Duration,
    /// Repeated regions outlined for per-block saturation (0 when the
    /// whole graph fit under the saturation cutoff).
    pub outlined_regions: usize,
    /// Number of independent e-graph saturations.
    pub segments: usize,
    /// Largest segment passed to egglog.
    pub max_segment_nodes: usize,
    /// Roots or segments left unchanged because parsing/extraction failed.
    pub extraction_failures: usize,
}

impl OptimizeReport {
    /// An empty report, for code paths that skip optimization (e.g. a
    /// cache hit) but still need to return a report.
    pub fn empty() -> Self {
        Self {
            mode: OptimizeMode::Off,
            extraction_cost: ExtractionCost::TensorTraffic,
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
            segments: 0,
            max_segment_nodes: 0,
            extraction_failures: 0,
        }
    }
}

impl fmt::Display for OptimizeReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Optimization Report ===")?;
        writeln!(
            f,
            "Mode: {} (cost: {}, segments: {}, max segment: {} nodes)",
            self.mode.as_str(),
            self.extraction_cost.as_str(),
            self.segments,
            self.max_segment_nodes,
        )?;
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
        if self.extraction_failures > 0 {
            writeln!(
                f,
                "Extraction failures: {} (affected roots left unchanged)",
                self.extraction_failures
            )?;
        }
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
    optimize_with_config(graph, OptimizeConfig::default())
}

/// Optimize with an explicit strategy and extraction objective.
pub fn optimize_with_config(graph: &Graph, config: OptimizeConfig) -> (Graph, OptimizeReport) {
    optimize_owned_with_config(clone_graph(graph), config)
}

/// Optimize an owned graph without first duplicating all node payloads.
///
/// Training uses this after autodiff, when the unoptimized graph has no
/// remaining consumer. The borrowed public API above retains its existing
/// behavior for callers that need to keep their source graph.
pub(crate) fn optimize_owned_with_config(
    graph: Graph,
    config: OptimizeConfig,
) -> (Graph, OptimizeReport) {
    match config.mode {
        OptimizeMode::Off => optimize_off(graph, config),
        OptimizeMode::EgglogWindowed | OptimizeMode::EgglogOutlined | OptimizeMode::EgglogWhole => {
            optimize_egglog(graph, config)
        }
    }
}

fn optimize_egglog(mut g: Graph, config: OptimizeConfig) -> (Graph, OptimizeReport) {
    let nodes_before = g.nodes().len();

    let segments = plan_segments(&g, config.mode, config.saturation_cutoff);
    let outlined_regions = segments.iter().filter(|s| s.shifts.len() > 1).count();
    let segment_count = segments.len();
    let max_segment_nodes = segments.iter().map(|s| s.ids.len()).max().unwrap_or(0);

    let mut index = build_structural_index(&g);
    let mut report = OptimizeReport {
        mode: config.mode,
        extraction_cost: config.extraction_cost,
        nodes_before,
        outlined_regions,
        segments: segment_count,
        max_segment_nodes,
        ..OptimizeReport::empty()
    };

    for seg in &segments {
        process_segment(&mut g, seg, &mut index, &mut report, config);
    }

    let dce_start = Instant::now();
    sweep_dead_nodes(&mut g);
    report.extract_time += dce_start.elapsed();

    report.nodes_after = g
        .nodes()
        .iter()
        .filter(|n| !matches!(n.op, Op::Nop))
        .count();

    log::info!(
        "optimizer: {} fusions on {} nodes",
        report.fusions_applied.len(),
        report.nodes_after
    );
    for fusion in &report.fusions_applied {
        if let Some(entry) = report.rules_fired.iter_mut().find(|e| e.0 == fusion.0) {
            entry.1 += 1;
        } else {
            report.rules_fired.push((fusion.0.clone(), 1));
        }
    }
    for &(ref name, count) in &report.rules_fired {
        log::info!("  {}x {}", count, name);
    }

    (g.into_toposort(), report)
}

fn optimize_off(mut g: Graph, config: OptimizeConfig) -> (Graph, OptimizeReport) {
    let nodes_before = g.nodes().len();
    let start = Instant::now();
    sweep_dead_nodes(&mut g);
    let extract_time = start.elapsed();
    let nodes_after = g
        .nodes()
        .iter()
        .filter(|node| !matches!(node.op, Op::Nop))
        .count();
    (
        g.into_toposort(),
        OptimizeReport {
            mode: config.mode,
            extraction_cost: config.extraction_cost,
            egglog_program: String::new(),
            num_eclasses: 0,
            num_enodes: 0,
            rules_fired: Vec::new(),
            nodes_before,
            nodes_after,
            fusions_applied: Vec::new(),
            egglog_time: std::time::Duration::ZERO,
            extract_time,
            outlined_regions: 0,
            segments: 0,
            max_segment_nodes: 0,
            extraction_failures: 0,
        },
    )
}

fn pack_glu_matmul(
    graph: &mut Graph,
    h: NodeId,
    wg: NodeId,
    wu: NodeId,
    transposed: bool,
    requires_full_precision: bool,
) -> Option<NodeId> {
    let (gate, up) = (graph.node(wg), graph.node(wu));
    let Op::Parameter {
        name: ref gate_name,
    } = gate.op
    else {
        return None;
    };
    let Op::Parameter { name: ref up_name } = up.op else {
        return None;
    };
    if gate.ty.shape.len() != 2
        || gate.ty.shape != up.ty.shape
        || gate.ty.dtype != up.ty.dtype
        || graph.node(h).ty.shape.len() != 2
        || (transposed
            && !matches!(
                gate.ty.dtype,
                crate::graph::DType::F32 | crate::graph::DType::F16
            ))
    {
        return None;
    }
    let (in_features, out_features) = if transposed {
        (gate.ty.shape[1], gate.ty.shape[0])
    } else {
        (gate.ty.shape[0], gate.ty.shape[1])
    };
    let dtype = gate.ty.dtype;
    let concat_name = format!(
        "{gate_name}+{up_name}{}",
        if transposed { ":rows" } else { "" }
    );
    let shape = if transposed {
        vec![2 * out_features, in_features]
    } else {
        vec![in_features, 2 * out_features]
    };
    graph.derived_params.push(crate::graph::DerivedParam {
        name: concat_name.clone(),
        sources: vec![
            (gate_name.clone(), out_features),
            (up_name.clone(), out_features),
        ],
        rows: shape[0],
        transform: if transposed {
            crate::graph::ParamTransform::VerticalConcat
        } else {
            crate::graph::ParamTransform::HorizontalConcat
        },
    });
    let concat_w = graph.add_raw_node_with_precision(
        Op::Parameter { name: concat_name },
        vec![],
        TensorType::new(shape, dtype),
        requires_full_precision,
    );
    let m = graph.node(h).ty.shape[0];
    Some(graph.add_raw_node_with_precision(
        if transposed { Op::MatMulBT } else { Op::MatMul },
        vec![h, concat_w],
        TensorType::f32(vec![m, 2 * out_features]),
        requires_full_precision,
    ))
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
    let mut program = String::new();
    egglog_prelude(&mut program, true);
    program.push_str(&segment_program(graph, &seg).0);
    program
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

fn plan_segments(g: &Graph, mode: OptimizeMode, saturation_cutoff: usize) -> Vec<Segment> {
    let n = g.nodes().len();
    let active = g
        .nodes()
        .iter()
        .filter(|n| !matches!(n.op, Op::Nop))
        .count();
    let saturation_cutoff = if mode == OptimizeMode::EgglogWhole {
        n.max(1)
    } else {
        saturation_cutoff.max(1)
    };
    let mut segments = Vec::new();
    let mut covered = vec![false; n];
    if mode == OptimizeMode::EgglogOutlined && active > saturation_cutoff {
        for r in crate::outline::detect_repeated_regions(g) {
            if r.period > saturation_cutoff {
                continue;
            }
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
        if window.len() == saturation_cutoff {
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
    // A derivative may read a forward result, including an output shared with
    // another kernel (for example cross-entropy logits gradients). Keep such
    // cut edges opaque: reconstructing a forward expression under the backward
    // root's precision policy can duplicate its producer and lose that identity.
    segments
        .into_iter()
        .flat_map(|seg| {
            let (full, relaxed) = seg
                .ids
                .into_iter()
                .partition(|&id| g.nodes()[id].requires_full_precision);
            [
                Segment {
                    ids: relaxed,
                    shifts: seg.shifts.clone(),
                },
                Segment {
                    ids: full,
                    shifts: seg.shifts,
                },
            ]
        })
        .filter(|seg| !seg.ids.is_empty())
        .collect()
}

// ---------------------------------------------------------------------------
// Egglog encoding
// ---------------------------------------------------------------------------

/// The egglog sort and rewrite rules. Named constructors exist only for
/// ops that rewrite rules pattern-match on; every other op — including
/// ones added later — encodes through the arity-generic `Op1..Op6`
/// constructors, tagged with the node id so ops with different
/// attributes (eps, strides, head counts) never unify.
fn egglog_prelude(prog: &mut String, pack_swiglu: bool) {
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
  ; Concrete matrix implementations. Same tensor as MatMul / FusedMatMulAdd.
  ; Names encode tile_m, tile_n, k stage, and split count so extraction can
  ; forbid one variant without forbidding the others.
  (M6464k32 Op Op)
  (M6464k16 Op Op)
  (M3232k32 Op Op)
  (M3232k16 Op Op)
  (M6432k32 Op Op)
  (M6464k8s8 Op Op)
  (MA6464k32 Op Op Op)
  (MA6464k16 Op Op Op)
  (MA3232k32 Op Op Op)
  (MA3232k16 Op Op Op)
  (MA6432k32 Op Op Op)
  (MA6464k8s8 Op Op Op)
  (Add Op Op)
  (Mul Op Op)
  (Relu Op)
  (Sigmoid Op)
  (Neg Op)
  (Transpose Op)
  (Silu Op)
  (SwiGLU Op Op)
  (SwiGLUPacked Op Op Op)
  (SwiGLUPackedBT Op Op Op)
  (Gelu Op)
  (GeGLU Op Op)
  (GeGLUPacked Op Op Op)
  (GeGLUPackedBT Op Op Op)
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
(rewrite (FusedMatMulAdd ?a ?b ?d) (Add (MatMul ?a ?b) ?d))
(rewrite (FusedMatMulATAdd ?a ?b ?d) (Add (MatMulAT ?a ?b) ?d))
(rewrite (FusedMatMulBTAdd ?a ?b ?d) (Add (MatMulBT ?a ?b) ?d))

; --- ONNX decomposed op recognition ---
; PyTorch decomposes compound ops when exporting to ONNX. These rules
; recognize the decomposed patterns and fuse them back into compound
; kernels.

; Silu: x * sigmoid(x)
(rewrite (Mul ?x (Sigmoid ?x)) (Silu ?x))
(rewrite (Mul (Sigmoid ?x) ?x) (Silu ?x))

; SwiGLU: silu(gate) * up
(rewrite (Mul (Silu ?gate) ?up) (SwiGLU ?gate ?up))
(rewrite (Mul ?up (Silu ?gate)) (SwiGLU ?gate ?up))

; GeGLU: gelu(gate) * up, then the same HorizontalConcat packing as SwiGLU.
(rewrite (Mul (Gelu ?gate) ?up) (GeGLU ?gate ?up))
(rewrite (Mul ?up (Gelu ?gate)) (GeGLU ?gate ?up))
(rewrite (GeGLU (MatMul ?h ?wg) (MatMul ?h ?wu)) (GeGLUPacked ?h ?wg ?wu))
(rewrite (GeGLU (MatMulBT ?h ?wg) (MatMulBT ?h ?wu)) (GeGLUPackedBT ?h ?wg ?wu))

",
    );
    if pack_swiglu {
        // Stamping creates the derived weight, or retains separate projections
        // when the operands are not compatible 2D parameters.
        prog.push_str("(rewrite (SwiGLU (MatMul ?h ?wg) (MatMul ?h ?wu)) (SwiGLUPacked ?h ?wg ?wu))\n(rewrite (SwiGLU (MatMulBT ?h ?wg) (MatMulBT ?h ?wu)) (SwiGLUPackedBT ?h ?wg ?wu))\n");
    }
    // Saturation is bounded: the deepest rewrite chain is three rules
    // (Mul(x, Sigmoid(x)) -> Silu, Mul(Silu, up) -> SwiGLU, then
    // SwiGLU(MatMul, MatMul) -> SwiGLUPacked), so three iterations reach
    // a fixpoint; the fourth is margin for future rules.
}

fn rule_graph(pack_swiglu: bool) -> egglog::EGraph {
    // Egglog clones share a mutable table-notification list. Searches on
    // different threads must not clone the same initialized database.
    thread_local! {
        static RULES: [std::cell::OnceCell<egglog::EGraph>; 2] =
            const { [const { std::cell::OnceCell::new() }; 2] };
    }
    RULES.with(|rules| {
        rules[usize::from(pack_swiglu)]
            .get_or_init(|| {
                let mut program = String::new();
                egglog_prelude(&mut program, pack_swiglu);
                let mut egraph = egglog::EGraph::default();
                egraph
                    .parse_and_run_program(None, &program)
                    .expect("valid built-in rewrite rules");
                egraph
            })
            .clone()
    })
}

/// Returns the named egglog constructor for ops that rewrite rules
/// match on, or `None` for generically-encoded ops.
fn named_constructor(op: &Op) -> Option<&'static str> {
    Some(match *op {
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
        Op::Gelu => "Gelu",
        Op::GeGLU => "GeGLU",
        _ => return None,
    })
}

fn node_to_egglog_expr(node: &Node) -> String {
    match node.op {
        Op::Input { .. } | Op::Parameter { .. } | Op::Constant { .. } => {
            format!("(Leaf {})", node.id)
        }
        Op::RoPE {
            theta,
            pos_offset: 0,
            ..
        }
        | Op::RoPEGrad {
            theta,
            pos_offset: 0,
            ..
        } if theta.is_finite()
            && theta > 0.0
            && node.ty.shape.first() == Some(&1)
            && node.inputs.len() == 1 =>
        {
            format!("$n{}", node.inputs[0])
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
/// insert is idempotent. Unfusing an existing matrix addition also creates a
/// new intermediate of the same output shape; it must not get a token cost.
fn eclass_sizes(
    graph: &Graph,
    egraph: &egglog::EGraph,
    ids: impl Iterator<Item = usize>,
) -> HashMap<egglog::Value, u64> {
    let mut sizes = HashMap::new();
    for id in ids {
        let node = &graph.nodes()[id];
        if matches!(node.op, Op::Nop) {
            continue;
        }
        let var = format!("$n{}", node.id);
        if let Some(value) = egraph.lookup_function(&var, &[]) {
            sizes.insert(value, node.ty.size_bytes() as u64);
        }
        let product = match node.op {
            Op::FusedMatMulAdd => "MatMul",
            Op::FusedMatMulATAdd => "MatMulAT",
            Op::FusedMatMulBTAdd => "MatMulBT",
            _ => continue,
        };
        let inputs = node.inputs[..2]
            .iter()
            .map(|id| {
                let name = format!("$n{id}");
                egraph.get_function(&name)?;
                egraph.lookup_function(&name, &[])
            })
            .collect::<Option<Vec<_>>>();
        if let Some(inputs) = inputs
            && let Some(value) = egraph.lookup_function(product, &inputs)
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

fn process_segment(
    g: &mut Graph,
    seg: &Segment,
    index: &mut HashMap<(&'static str, Vec<NodeId>, bool), NodeId>,
    report: &mut OptimizeReport,
    config: OptimizeConfig,
) {
    let egglog_start = Instant::now();
    let (program, externals) = segment_program(g, seg);
    if report.egglog_program.is_empty() {
        egglog_prelude(&mut report.egglog_program, config.pack_swiglu);
        report.egglog_program.push_str(&program);
    }
    let mut egraph = rule_graph(config.pack_swiglu);
    if let Err(e) = egraph.parse_and_run_program(None, &program) {
        log::warn!(
            "egglog failed on segment of {} nodes: {} — leaving it unoptimized",
            seg.ids.len(),
            e
        );
        report.extraction_failures += 1;
        report.egglog_time += egglog_start.elapsed();
        return;
    }
    let cm = match config.extraction_cost {
        ExtractionCost::AstSize => FusionCostModel::ast_size(),
        ExtractionCost::TensorTraffic => {
            let size_ids = externals.iter().copied().chain(seg.ids.iter().copied());
            FusionCostModel::with_sizes(eclass_sizes(g, &egraph, size_ids))
        }
    };

    let roots = segment_roots(g, seg);
    let sort = egraph.get_sort_by_name("Op").unwrap().clone();
    let extractor = Extractor::compute_costs_from_rootsorts(Some(vec![sort]), &egraph, cm);
    let mut dag = TermDag::default();
    let mut terms = Vec::new();
    for &root in &roots {
        let var = format!("$n{}", root);
        match egraph.lookup_function(&var, &[]) {
            Some(value) => {
                let extraction = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    extractor.extract_best(&egraph, &mut dag, value)
                }));
                match extraction {
                    Ok(Some((cost, term_id))) => {
                        log::debug!(
                            "extracted $n{} (cost {}): {}",
                            root,
                            cost,
                            dag.to_string(term_id)
                        );
                        terms.push((root, term_id));
                    }
                    Ok(None) => {
                        report.extraction_failures += 1;
                        log::warn!("extraction failed for $n{}", root);
                    }
                    Err(_) => {
                        report.extraction_failures += 1;
                        log::warn!(
                            "egglog panicked while reconstructing $n{} — root left unchanged",
                            root
                        );
                    }
                }
            }
            None => {
                report.extraction_failures += 1;
                log::warn!("missing e-class for $n{}", root);
            }
        }
    }
    let serialized = egraph.serialize(egglog::SerializeConfig::default());
    report.num_eclasses += serialized.egraph.class_data.len();
    report.num_enodes += serialized.egraph.nodes.len();
    report.egglog_time += egglog_start.elapsed();

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
        let Some(ref ext_map) = *ext_map else {
            log::warn!(
                "segment instance at +{} has ambiguous external edges — left unoptimized",
                shift
            );
            continue;
        };
        for &(root, term_id) in &terms {
            let requires_full_precision = g.node((root + shift) as NodeId).requires_full_precision;
            let mut stamper = Stamper {
                g,
                index,
                seg_ids: &idset,
                shift,
                ext_map,
                fusions: &mut report.fusions_applied,
                memo: HashMap::new(),
                requires_full_precision,
            };
            if let Err(e) = stamper.stamp_root(root + shift, &dag, term_id) {
                log::warn!("stamping $n{} (+{}) failed: {}", root, shift, e);
            }
        }
    }
    report.extract_time += stamp_start.elapsed();
}

/// Rebuilds extracted terms in the graph IR. Interior nodes whose
/// children were rewritten are mutated in place (the new inputs are
/// value-equivalent, so every consumer — and every id-carrying attribute
/// like `fwd_node` — stays valid); new fused nodes are appended; roots
/// are overwritten in place so their node ids, types, and output status
/// survive.
struct Stamper<'a> {
    g: &'a mut Graph,
    /// Structural memo for named constructors:
    /// (name, children, precision policy) → node.
    index: &'a mut HashMap<(&'static str, Vec<NodeId>, bool), NodeId>,
    /// Node ids of the encoded instance (terms only reference these).
    seg_ids: &'a HashSet<usize>,
    /// Id shift of the instance being stamped.
    shift: usize,
    /// External-leaf translation for this instance.
    ext_map: &'a HashMap<usize, NodeId>,
    fusions: &'a mut Vec<(String, u32)>,
    /// Per-(dag, instance) term resolution cache.
    memo: HashMap<TermId, NodeId>,
    /// Precision policy of the root currently being reconstructed. Newly
    /// materialized interior nodes must not cross the forward/backward
    /// reduced-precision boundary.
    requires_full_precision: bool,
}

impl Stamper<'_> {
    /// Overwrite the root node in place with the extracted term.
    fn stamp_root(&mut self, root: usize, dag: &TermDag, term_id: TermId) -> Result<(), String> {
        match *dag.get(term_id) {
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

    fn resolve_children(
        &mut self,
        dag: &TermDag,
        children: &[TermId],
    ) -> Result<Vec<NodeId>, String> {
        children.iter().map(|&c| self.resolve(dag, c)).collect()
    }

    fn resolve(&mut self, dag: &TermDag, term_id: TermId) -> Result<NodeId, String> {
        if let Some(&id) = self.memo.get(&term_id) {
            return Ok(id);
        }
        let id = match *dag.get(term_id) {
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
                match self.index.get(&(
                    static_constructor(name)?,
                    inputs.clone(),
                    self.requires_full_precision,
                )) {
                    Some(&hit) => hit,
                    None => self.build_named(name, inputs, None)?,
                }
            }
            ref other => return Err(format!("unexpected term {:?}", other)),
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
        match name {
            "SwiGLUPacked" | "SwiGLUPackedBT" => {
                return self.build_glu_packed(
                    &inputs,
                    target,
                    Op::SwiGLUConcat,
                    static_constructor(name)?,
                );
            }
            "GeGLUPacked" | "GeGLUPackedBT" => {
                return self.build_glu_packed(
                    &inputs,
                    target,
                    Op::GeGLUConcat,
                    static_constructor(name)?,
                );
            }
            _ => {}
        }
        if let Some(spec) = scheduled_matmul(name) {
            let fused = name.starts_with("MA");
            let (op, ty) = if fused {
                (Op::FusedMatMulAdd, self.g.node(inputs[2]).ty.clone())
            } else {
                let a = self.g.node(inputs[0]).ty.shape.clone();
                let b = self.g.node(inputs[1]).ty.shape.clone();
                if a.len() != 2 || b.len() != 2 {
                    return Err(format!("{name} needs rank-2 operands"));
                }
                (Op::MatMul, TensorType::f32(vec![a[0], b[1]]))
            };
            let id = self.place(op, inputs.clone(), ty, target);
            self.g.nodes_mut()[id as usize].matmul_impl = Some(spec);
            self.index.insert(
                (
                    static_constructor(name)?,
                    inputs,
                    self.requires_full_precision,
                ),
                id,
            );
            return Ok(id);
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
            "FusedMatMulAdd" => (
                Op::FusedMatMulAdd,
                ty_of(inputs[2]),
                Some("MatMul+Add→FusedMatMulAdd"),
            ),
            "FusedMatMulATAdd" => (
                Op::FusedMatMulATAdd,
                ty_of(inputs[2]),
                Some("MatMulAT+Add→FusedMatMulATAdd"),
            ),
            "FusedMatMulBTAdd" => (
                Op::FusedMatMulBTAdd,
                ty_of(inputs[2]),
                Some("MatMulBT+Add→FusedMatMulBTAdd"),
            ),
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
            "Gelu" => (Op::Gelu, ty_of(inputs[0]), None),
            "GeGLU" => (Op::GeGLU, ty_of(inputs[0]), Some("Gelu+Mul→GeGLU")),
            other => return Err(format!("unknown constructor {}", other)),
        };
        let id = self.place(op, inputs.clone(), ty, target);
        self.index.insert(
            (
                static_constructor(name)?,
                inputs,
                self.requires_full_precision,
            ),
            id,
        );
        if let Some(label) = label {
            self.fusions.push((label.to_string(), id));
        }
        Ok(id)
    }

    /// SwiGLU/GeGLU(MatMul(h, wg), MatMul(h, wu)) → Concat(MatMul(h, wg|wu))
    /// with a derived concatenated-weight parameter. Falls back to the
    /// equivalent unpacked form when the weights are not plain same-shape
    /// 2D parameters (e.g. ONNX constants).
    fn build_glu_packed(
        &mut self,
        inputs: &[NodeId],
        target: Option<NodeId>,
        concat_op: Op,
        packed_key: &'static str,
    ) -> Result<NodeId, String> {
        let (h, wg, wu) = (inputs[0], inputs[1], inputs[2]);
        let transposed = packed_key.ends_with("BT");
        let matmul = if transposed { "MatMulBT" } else { "MatMul" };
        let unpacked = match concat_op {
            Op::SwiGLUConcat => "SwiGLU",
            Op::GeGLUConcat => "GeGLU",
            _ => unreachable!("glu pack only for SwiGLU/GeGLU concat"),
        };
        let Some(wide_mm) =
            pack_glu_matmul(self.g, h, wg, wu, transposed, self.requires_full_precision)
        else {
            let gate = self.lookup_or_build(matmul, vec![h, wg])?;
            let up = self.lookup_or_build(matmul, vec![h, wu])?;
            return self.build_named(unpacked, vec![gate, up], target);
        };
        let shape = &self.g.node(wide_mm).ty.shape;
        let (m, out_features) = (shape[0], shape[1] / 2);
        let id = self.place(
            concat_op,
            vec![wide_mm],
            TensorType::f32(vec![m, out_features]),
            target,
        );
        self.index.insert(
            (packed_key, inputs.to_vec(), self.requires_full_precision),
            id,
        );
        self.fusions.push((
            format!("{unpacked}({matmul},{matmul})→{unpacked}Concat({matmul})"),
            id,
        ));
        Ok(id)
    }

    fn lookup_or_build(&mut self, name: &str, inputs: Vec<NodeId>) -> Result<NodeId, String> {
        match self.index.get(&(
            static_constructor(name)?,
            inputs.clone(),
            self.requires_full_precision,
        )) {
            Some(&hit) => Ok(hit),
            None => self.build_named(name, inputs, None),
        }
    }

    /// Write the node into `target` (root stamping keeps the root's id
    /// and type) or append a new node.
    fn place(
        &mut self,
        op: Op,
        inputs: Vec<NodeId>,
        ty: TensorType,
        target: Option<NodeId>,
    ) -> NodeId {
        let id = match target {
            Some(id) => {
                let node = &mut self.g.nodes_mut()[id as usize];
                node.op = op;
                node.inputs = inputs;
                // ty deliberately untouched: same e-class, same tensor.
                id
            }
            None => {
                self.g
                    .add_raw_node_with_precision(op, inputs, ty, self.requires_full_precision)
            }
        };
        self.g.nodes_mut()[id as usize].matmul_impl = None;
        id
    }
}

fn named_constructor_exists(name: &str) -> bool {
    static_constructor(name).is_ok()
}

/// Tile equalities used only by measured extraction. Ordinary `optimize`
/// does not run these rules, so a normal build still lowers the logical op.
pub(crate) const TILE_EQUALITY_RULES: &str = "\
(rewrite (MatMul ?a ?b) (M6464k32 ?a ?b))
(rewrite (MatMul ?a ?b) (M6464k16 ?a ?b))
(rewrite (MatMul ?a ?b) (M3232k32 ?a ?b))
(rewrite (MatMul ?a ?b) (M3232k16 ?a ?b))
(rewrite (MatMul ?a ?b) (M6432k32 ?a ?b))
(rewrite (MatMul ?a ?b) (M6464k8s8 ?a ?b))
(rewrite (FusedMatMulAdd ?a ?b ?d) (MA6464k32 ?a ?b ?d))
(rewrite (FusedMatMulAdd ?a ?b ?d) (MA6464k16 ?a ?b ?d))
(rewrite (FusedMatMulAdd ?a ?b ?d) (MA3232k32 ?a ?b ?d))
(rewrite (FusedMatMulAdd ?a ?b ?d) (MA3232k16 ?a ?b ?d))
(rewrite (FusedMatMulAdd ?a ?b ?d) (MA6432k32 ?a ?b ?d))
(rewrite (FusedMatMulAdd ?a ?b ?d) (MA6464k8s8 ?a ?b ?d))
";

pub(crate) fn scheduled_matmul(name: &str) -> Option<crate::graph::MatmulImpl> {
    let spec = |tile_m, tile_n, k_stage, splits| crate::graph::MatmulImpl {
        tile_m,
        tile_n,
        k_stage,
        splits,
    };
    Some(match name {
        "M6464k32" | "MA6464k32" => spec(64, 64, 32, 1),
        "M6464k16" | "MA6464k16" => spec(64, 64, 16, 1),
        "M3232k32" | "MA3232k32" => spec(32, 32, 32, 1),
        "M3232k16" | "MA3232k16" => spec(32, 32, 16, 1),
        "M6432k32" | "MA6432k32" => spec(64, 32, 32, 1),
        "M6464k8s8" | "MA6464k8s8" => spec(64, 64, 8, 8),
        _ => return None,
    })
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
        "SwiGLUPackedBT" => "SwiGLUPackedBT",
        "Gelu" => "Gelu",
        "GeGLU" => "GeGLU",
        "GeGLUPacked" => "GeGLUPacked",
        "GeGLUPackedBT" => "GeGLUPackedBT",
        "M6464k32" => "M6464k32",
        "M6464k16" => "M6464k16",
        "M3232k32" => "M3232k32",
        "M3232k16" => "M3232k16",
        "M6432k32" => "M6432k32",
        "M6464k8s8" => "M6464k8s8",
        "MA6464k32" => "MA6464k32",
        "MA6464k16" => "MA6464k16",
        "MA3232k32" => "MA3232k32",
        "MA3232k16" => "MA3232k16",
        "MA6432k32" => "MA6432k32",
        "MA6464k8s8" => "MA6464k8s8",
        other => return Err(format!("unknown constructor {}", other)),
    })
}

fn lit_node_id(dag: &TermDag, term_id: TermId) -> Result<usize, String> {
    match *dag.get(term_id) {
        Term::Lit(Literal::Int(v)) => Ok(v as usize),
        ref other => Err(format!("expected node-id literal, got {:?}", other)),
    }
}

/// Structural memo of the existing graph for named constructors, so
/// term resolution finds each instance's own nodes (and never duplicates
/// an existing equivalent node).
fn build_structural_index(g: &Graph) -> HashMap<(&'static str, Vec<NodeId>, bool), NodeId> {
    let mut index = HashMap::new();
    for node in g.nodes() {
        if let Some(name) = named_constructor(&node.op) {
            index.insert(
                (name, node.inputs.clone(), node.requires_full_precision),
                node.id,
            );
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
pub fn apply_winograd_conv_fusions(
    graph: &mut Graph,
    fusions: &mut Vec<(String, u32)>,
    config: &OptimizeConfig,
) {
    if config.no_winograd {
        log::info!("Winograd convolution disabled by OptimizeConfig::no_winograd");
        return;
    }
    let mut transformed = HashMap::new();
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
        let requires_full_precision = node.requires_full_precision;
        let weight_name = match graph.node(weight_id).op {
            Op::Parameter { ref name } => name.clone(),
            _ => continue,
        };
        let input_id = node.inputs[0];

        let key = (weight_id, in_channels, out_channels);
        let wino_param = if let Some(&parameter) = transformed.get(&key) {
            graph.nodes_mut()[parameter as usize].requires_full_precision |=
                requires_full_precision;
            parameter
        } else {
            // One transform per logical weight and channel layout.
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
            let parameter = graph.add_raw_node_with_precision(
                Op::Parameter { name: wino_name },
                vec![],
                TensorType::f32(vec![wino_size]),
                requires_full_precision,
            );

            transformed.insert(key, parameter);
            parameter
        };

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
    graph.deep_clone()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn outlined_egglog_is_the_production_default() {
        assert_eq!(OptimizeConfig::default().mode, OptimizeMode::EgglogOutlined);
    }

    #[test]
    fn tensor_traffic_does_not_charge_integer_node_ids() {
        let mut egraph = egglog::EGraph::default();
        egraph
            .parse_and_run_program(
                None,
                "(datatype Op (Leaf i64) (Op1 i64 Op))
                 (let a (Leaf 0)) (let b (Op1 1 a))",
            )
            .unwrap();
        let a = egraph.lookup_function("a", &[]).unwrap();
        let b = egraph.lookup_function("b", &[]).unwrap();
        assert_eq!(b, egraph.base_to_value(1i64));
        let cost = FusionCostModel::with_sizes(HashMap::from([(a, 1024), (b, 2048)]));
        let extractor = Extractor::compute_costs_from_rootsorts(None, &egraph, cost);
        let (bytes, _) = extractor
            .extract_best(&egraph, &mut TermDag::default(), b)
            .unwrap();
        assert_eq!(bytes, 1024 + 2048);
    }

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
        let negs = opt
            .nodes()
            .iter()
            .filter(|n| matches!(n.op, Op::Neg))
            .count();
        assert_eq!(negs, 0, "dead Neg nodes should be swept");
    }

    #[test]
    fn static_rope_identity() {
        for (rows, offset, dynamic) in [(1, 0, false), (1, 1, false), (2, 0, false), (1, 0, true)] {
            let mut g = Graph::new();
            let x = g.input("x", &[rows, 8]);
            let y = if dynamic {
                let pos = g.input_u32("pos", &[1]);
                g.rope_dynamic_offset(x, 10000.0, pos, 4)
            } else {
                g.rope_with_offset(x, 10000.0, offset, 4)
            };
            let grad = g.rope_grad(x, 10000.0, offset, 4);
            g.set_outputs(vec![y, grad]);
            let opt = optimize(&g);
            assert_eq!(
                matches!(opt.node(opt.outputs()[0]).op, Op::Identity),
                rows == 1 && offset == 0 && !dynamic,
                "rows={rows}, offset={offset}, dynamic={dynamic}: {opt}"
            );
            assert_eq!(
                matches!(opt.node(opt.outputs()[1]).op, Op::Identity),
                rows == 1 && offset == 0,
                "rows={rows}, offset={offset}, dynamic={dynamic}: {opt}"
            );
        }
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
        for orientation in 0..3 {
            let mut g = Graph::new();
            let x = g.input("x", &if orientation == 1 { [8, 4] } else { [4, 8] });
            let w = g.parameter("w", &if orientation == 2 { [4, 8] } else { [8, 4] });
            let b = g.input("bias", &[4, 4]);
            let (mm, expected) = match orientation {
                0 => (g.matmul(x, w), Op::FusedMatMulAdd),
                1 => (g.matmul_at(x, w), Op::FusedMatMulATAdd),
                _ => (g.matmul_bt(x, w), Op::FusedMatMulBTAdd),
            };
            let out = g.add(mm, b);
            g.set_outputs(vec![out]);

            let (mut opt, report) = optimize_with_report(&g);
            let output_node = opt.node(opt.outputs()[0]);
            assert_eq!(
                std::mem::discriminant(&output_node.op),
                std::mem::discriminant(&expected)
            );
            assert!(!report.fusions_applied.is_empty());
            for _ in 0..2 {
                opt = optimize_with_report(&opt).0;
                assert_eq!(
                    std::mem::discriminant(&opt.node(opt.outputs()[0]).op),
                    std::mem::discriminant(&expected)
                );
            }
        }
    }

    #[test]
    fn test_optimizer_ablation_modes_share_rewrite_semantics() {
        let mut g = Graph::new();
        let x = g.input("x", &[4, 8]);
        let w = g.parameter("w", &[8, 4]);
        let bias = g.input("bias", &[4, 4]);
        let mm = g.matmul(x, w);
        let out = g.add(mm, bias);
        g.set_outputs(vec![out]);

        let (off, off_report) = optimize_with_config(
            &g,
            OptimizeConfig {
                mode: OptimizeMode::Off,
                ..OptimizeConfig::default()
            },
        );
        assert!(matches!(off.node(off.outputs()[0]).op, Op::Add));
        assert_eq!(off_report.mode, OptimizeMode::Off);

        for mode in [
            OptimizeMode::EgglogWindowed,
            OptimizeMode::EgglogOutlined,
            OptimizeMode::EgglogWhole,
        ] {
            for extraction_cost in [ExtractionCost::AstSize, ExtractionCost::TensorTraffic] {
                let (optimized, report) = optimize_with_config(
                    &g,
                    OptimizeConfig {
                        mode,
                        extraction_cost,
                        saturation_cutoff: 3,
                        ..OptimizeConfig::default()
                    },
                );
                assert!(
                    matches!(
                        optimized.node(optimized.outputs()[0]).op,
                        Op::FusedMatMulAdd
                    ),
                    "{mode:?}/{extraction_cost:?} did not select the fusion"
                );
                assert_eq!(report.mode, mode);
                assert_eq!(report.extraction_cost, extraction_cost);
            }
        }
    }

    #[test]
    fn glu_concat_fusion_preserves_orientation_and_format() {
        for transposed in [false, true] {
            for gelu in [false, true] {
                for dtype in [crate::graph::DType::F32, crate::graph::DType::F16] {
                    let mut g = Graph::new();
                    let h = g.input("h", &[3, 12]);
                    let shape = if transposed { vec![7, 12] } else { vec![12, 7] };
                    let mut project = |name: &str| {
                        let w = g.add_raw_node(
                            Op::Parameter { name: name.into() },
                            vec![],
                            TensorType::new(shape.clone(), dtype),
                        );
                        if transposed {
                            g.matmul_bt(h, w)
                        } else {
                            g.matmul(h, w)
                        }
                    };
                    let gate = project("gate");
                    let up = project("up");
                    let out = if gelu {
                        g.geglu(gate, up)
                    } else {
                        g.swiglu(gate, up)
                    };
                    g.set_outputs(vec![out]);
                    if !gelu {
                        let (unpacked, _) = optimize_with_config(
                            &g,
                            OptimizeConfig {
                                pack_swiglu: false,
                                ..Default::default()
                            },
                        );
                        assert!(unpacked.derived_params.is_empty());
                        assert!(matches!(
                            unpacked.node(unpacked.outputs()[0]).op,
                            Op::SwiGLU
                        ));
                    }
                    for mode in [OptimizeMode::EgglogWhole, OptimizeMode::EgglogOutlined] {
                        for extraction_cost in
                            [ExtractionCost::AstSize, ExtractionCost::TensorTraffic]
                        {
                            let (opt, _) = optimize_with_config(
                                &g,
                                OptimizeConfig {
                                    mode,
                                    extraction_cost,
                                    ..Default::default()
                                },
                            );
                            let output = opt.node(opt.outputs()[0]);
                            assert!(matches!(
                                (&output.op, gelu),
                                (Op::SwiGLUConcat, false) | (Op::GeGLUConcat, true)
                            ));
                            let mm = opt.node(output.inputs[0]);
                            assert!(matches!(
                                (&mm.op, transposed),
                                (Op::MatMul, false) | (Op::MatMulBT, true)
                            ));
                            assert_eq!(mm.ty.shape, [3, 14]);
                            let weight = opt.node(mm.inputs[1]);
                            assert_eq!(weight.ty.dtype, dtype);
                            assert_eq!(
                                weight.ty.shape,
                                if transposed { [14, 12] } else { [12, 14] }
                            );
                            assert_eq!(opt.derived_params.len(), 1);
                            assert!(matches!(
                                (&opt.derived_params[0].transform, transposed),
                                (crate::graph::ParamTransform::HorizontalConcat, false)
                                    | (crate::graph::ParamTransform::VerticalConcat, true)
                            ));
                        }
                    }
                }
            }
        }
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

    /// Extracting a fusion below a generic root appends the fused node. The
    /// optimizer must restore topological order before downstream lifetime
    /// planning sees the rewritten graph.
    #[test]
    fn test_extracted_interior_fusion_is_topologically_ordered() {
        let mut g = Graph::new();
        let grad = g.input("grad", &[4, 8]);
        let w = g.parameter("w", &[4, 8]);
        let previous = g.input("previous", &[4, 4]);
        let bt = g.add_raw_node(
            Op::MatMulBT,
            vec![grad, w],
            crate::graph::TensorType::f32(vec![4, 4]),
        );
        let accumulated = g.add(bt, previous);
        let loss = g.mean_all(accumulated);
        g.set_outputs(vec![loss]);

        let (opt, report) = optimize_with_report(&g);
        assert!(
            report
                .fusions_applied
                .iter()
                .any(|entry| entry.0.contains("BT")),
            "no BT fusion in report"
        );

        let output = opt.node(opt.outputs()[0]);
        assert!(matches!(output.op, Op::MeanAll));
        let fused = opt.node(output.inputs[0]);
        assert!(matches!(fused.op, Op::FusedMatMulBTAdd));
        assert!(
            fused.id < output.id,
            "interior fusion {} must precede consumer {}",
            fused.id,
            output.id,
        );
        for node in opt
            .nodes()
            .iter()
            .filter(|node| !matches!(node.op, Op::Nop))
        {
            assert!(
                node.inputs.iter().all(|&input| input < node.id),
                "node {} has non-topological inputs {:?}",
                node.id,
                node.inputs,
            );
        }
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
