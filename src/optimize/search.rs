//! Bounded equivalent-region extraction for measured implementation selection.
//!
//! This uses the existing egglog rules and graph reconstruction. It deliberately
//! accepts a bounded region and extracts its observable roots together. Timing
//! roots independently would double-count shared work. Each candidate is tuned before
//! comparing its complete execution, not ranked by its untuned kernel timings.

use super::{FusionCostModel, Segment, Stamper};

use crate::{Graph, graph::Op};
use egglog::{
    Term, TermDag, TermId, Value,
    extract::{CostModel, Extractor},
};
use std::{
    collections::{BTreeMap, HashMap, HashSet, VecDeque},
    ops::Range,
    sync::Arc,
};

pub(crate) struct Candidate {
    pub graph: Graph,
    pub expression: String,
}

pub(crate) struct SearchSpace {
    pub candidates: Vec<Candidate>,
    pub truncated: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct Edge {
    head: String,
    inputs: Vec<Value>,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct Cost {
    forbidden: usize,
    estimate: u64,
    // Prefer implementation alternatives only when logical estimates tie.
    unscheduled: usize,
}

impl egglog::extract::Cost for Cost {
    fn identity() -> Self {
        Self {
            forbidden: 0,
            estimate: 0,
            unscheduled: 0,
        }
    }
    fn unit() -> Self {
        Self {
            forbidden: 0,
            estimate: 1,
            unscheduled: 0,
        }
    }
    fn combine(self, other: &Self) -> Self {
        Self {
            forbidden: self.forbidden.saturating_add(other.forbidden),
            estimate: self.estimate.saturating_add(other.estimate),
            unscheduled: self.unscheduled.saturating_add(other.unscheduled),
        }
    }
}

#[derive(Clone)]
struct Excluding {
    costs: FusionCostModel,
    forbidden: Arc<[Edge]>,
}

impl CostModel<Cost> for Excluding {
    fn base_value_cost(&self, _: &egglog::EGraph, _: &egglog::ArcSort, _: Value) -> Cost {
        Cost {
            forbidden: 0,
            estimate: 0,
            unscheduled: 0,
        }
    }

    fn fold(&self, _: &str, children: &[Cost], head: Cost) -> Cost {
        use egglog::extract::Cost as _;
        children
            .iter()
            .fold(head, |cost, child| cost.combine(child))
    }

    fn enode_cost(
        &self,
        egraph: &egglog::EGraph,
        func: &egglog::Function,
        row: &egglog::FunctionRow,
    ) -> Cost {
        Cost {
            forbidden: usize::from(
                self.forbidden
                    .binary_search_by(|edge| {
                        edge.head.as_str().cmp(func.name()).then_with(|| {
                            edge.inputs.as_slice().cmp(&row.vals[..row.vals.len() - 1])
                        })
                    })
                    .is_ok(),
            ),
            estimate: self.costs.enode_cost(egraph, func, row),
            unscheduled: usize::from(super::matrix_family(func.name()) == Some(func.name())),
        }
    }
}

fn edges(
    egraph: &egglog::EGraph,
    terms: &TermDag,
    root: TermId,
) -> Result<Vec<(Value, Edge)>, String> {
    let mut result = Vec::new();
    let mut values = vec![None; terms.size()];
    let mut stack = vec![(root, false)];
    while let Some((id, expanded)) = stack.pop() {
        if values[id].is_some() {
            continue;
        }
        let value = match *terms.get(id) {
            Term::App(ref head, ref args) => {
                if !expanded {
                    stack.push((id, true));
                    stack.extend(args.iter().rev().map(|&arg| (arg, false)));
                    continue;
                }
                let inputs: Vec<_> = args.iter().map(|&arg| values[arg].unwrap()).collect();
                let value = egraph
                    .lookup_function(head, &inputs)
                    .ok_or("extracted term is missing from the e-graph")?;
                if head != "Leaf" {
                    result.push((
                        value,
                        Edge {
                            head: head.clone(),
                            inputs,
                        },
                    ));
                }
                value
            }
            Term::Lit(egglog::ast::Literal::Int(value)) => egraph.base_to_value(value),
            _ => return Err("expected an operator or node-id literal".into()),
        };
        values[id] = Some(value);
    }
    Ok(result)
}

/// Retain equivalent implementations before lowering or kernel tuning.
///
/// Egglog's extractor reconstructs each candidate. Excluding a selected e-node
/// exposes alternative choices, including inside children. This is bounded
/// enumeration, not a globally optimal or k-best schedule search. The estimate
/// orders exploration only; callers must measure complete lowered candidates.
/// `truncated` reports an unfinished search. No GPU measurement happens here.
pub(crate) fn candidates(
    graph: &Graph,
    config: super::OptimizeConfig,
    limit: usize,
) -> Result<SearchSpace, String> {
    let cutoff = config.saturation_cutoff.min(super::SATURATION_CUTOFF);
    if graph.nodes().len() <= cutoff {
        return region_candidates(graph, 0..graph.nodes().len(), config, limit);
    }
    // Parameters and opaque operators need not consume the saturation bound.
    // Extract the rewritable operators together, including independent
    // projections that outlining otherwise leaves outside the repeated body.
    // Do not use sparse regions across mutations: effects need ordered cut edges.
    let ids: Vec<_> = graph
        .nodes()
        .iter()
        .filter(|node| super::named_constructor(&node.op).is_some())
        .map(|node| node.id as usize)
        .collect();
    if limit == 0
        || ids.is_empty()
        || ids.len() > cutoff
        || graph.nodes().iter().any(|node| {
            matches!(
                node.op,
                Op::CacheWrite | Op::CacheWritePrefix | Op::ScatterAdd { .. }
            )
        })
    {
        return Err("rewritable operators exceed a bounded pure region".into());
    }
    segment_candidates(
        graph,
        Segment {
            ids,
            shifts: vec![0],
        },
        config,
        limit,
    )
}

/// Explore a contiguous region without changing the surrounding graph. Its
/// escaping values are extracted together; dependencies outside the region stay
/// opaque. Returned graphs preserve the original inputs, parameters and outputs.
pub(crate) fn region_candidates(
    graph: &Graph,
    region: Range<usize>,
    config: super::OptimizeConfig,
    limit: usize,
) -> Result<SearchSpace, String> {
    if limit == 0
        || region.is_empty()
        || region.len() > super::SATURATION_CUTOFF
        || region.end > graph.nodes().len()
    {
        return Err("expected a bounded region and a positive candidate limit".into());
    }
    let segment = Segment {
        ids: region
            .filter(|&id| !matches!(graph.nodes()[id].op, Op::Nop))
            .collect(),
        shifts: vec![0],
    };
    segment_candidates(graph, segment, config, limit)
}

/// Apply each extracted alternative to all verified instances, then return the
/// complete model. Parameters and cut-edge placement remain model properties,
/// not newly introduced host-visible region inputs.
pub(crate) fn repeated_candidates(
    graph: &Graph,
    region: crate::outline::Region,
    config: super::OptimizeConfig,
    limit: usize,
) -> Result<SearchSpace, String> {
    if limit == 0
        || !crate::outline::detect_repeated_regions(graph).contains(&region)
        || region.period > super::SATURATION_CUTOFF
    {
        return Err("expected a verified repeated region and a positive candidate limit".into());
    }
    segment_candidates(
        graph,
        Segment {
            ids: (region.start..region.start + region.period)
                .filter(|&id| !matches!(graph.nodes()[id].op, Op::Nop))
                .collect(),
            shifts: (0..region.count).map(|i| i * region.period).collect(),
        },
        config,
        limit,
    )
}

fn segment_candidates(
    graph: &Graph,
    segment: Segment,
    config: super::OptimizeConfig,
    limit: usize,
) -> Result<SearchSpace, String> {
    if graph
        .nodes()
        .iter()
        .any(|node| node.inputs.iter().any(|&id| id >= node.id))
    {
        return Err("region search requires topologically ordered nodes".into());
    }
    let roots = super::segment_roots(graph, &segment);
    if roots.is_empty() {
        return Err("region has no observable output".into());
    }
    let full_precision = graph.node(roots[0] as u32).requires_full_precision;
    for id in segment
        .shifts
        .iter()
        .flat_map(|shift| segment.ids.iter().map(move |id| id + shift))
    {
        let node = &graph.nodes()[id];
        match node.op {
            Op::Input { .. } | Op::Parameter { .. } | Op::Constant { .. } => continue,
            Op::CacheWrite | Op::CacheWritePrefix | Op::ScatterAdd { .. } => {
                return Err("stateful regions need an explicit mutation contract".into());
            }
            _ => {}
        }
        if node.requires_full_precision != full_precision {
            return Err("region crosses a precision boundary".into());
        }
        if !(1..=6).contains(&node.inputs.len()) {
            return Err("region contains an unsupported operator arity".into());
        }
    }
    let (mut program, externals) = super::segment_program(graph, &segment);
    let root_name = if roots.len() == 1 {
        format!("$n{}", roots[0])
    } else {
        program.push_str(&format!(
            "(constructor SearchOutputs ({}) Op)\n(let $outputs (SearchOutputs {}))\n",
            vec!["Op"; roots.len()].join(" "),
            roots
                .iter()
                .map(|id| format!("$n{id}"))
                .collect::<Vec<_>>()
                .join(" "),
        ));
        "$outputs".into()
    };
    let mut egraph = super::rule_graph(config.pack_swiglu, true);
    egraph
        .parse_and_run_program(None, &program)
        .map_err(|e| e.to_string())?;
    let sort = egraph.get_sort_by_name("Op").unwrap().clone();
    let value = egraph
        .lookup_function(&root_name, &[])
        .ok_or("missing extraction root")?;
    let costs = match config.extraction_cost {
        super::ExtractionCost::AstSize => FusionCostModel::ast_size(),
        super::ExtractionCost::TensorTraffic => FusionCostModel::with_sizes(super::eclass_sizes(
            graph,
            &egraph,
            segment.ids.iter().chain(&externals).copied(),
        )),
    };
    let ids: HashSet<_> = segment.ids.iter().copied().collect();
    let uses = super::external_uses(graph, &segment);
    let ext_maps = segment
        .shifts
        .iter()
        .map(|&shift| super::instance_ext_map(graph, &segment, &uses, shift))
        .collect::<Option<Vec<_>>>()
        .ok_or("region instances have ambiguous external edges")?;
    let empty = Arc::<[Edge]>::from([]);
    let mut pending = VecDeque::from([empty.clone()]);
    let mut schedules = VecDeque::new();
    let mut visited = HashSet::from([empty]);
    let mut expressions = HashSet::new();
    let mut terms = TermDag::default();
    let mut choices = HashMap::new();
    let mut result = Vec::new();
    let attempts = limit.saturating_mul(segment.ids.len());
    let mut bounded = false;
    for _ in 0..attempts {
        // Alternate structural and schedule coverage. Otherwise a region with
        // many fusion sites can fill the frontier with one identical tile.
        let next = if !result.is_empty() && result.len() % 2 == 0 {
            schedules.pop_front().or_else(|| pending.pop_front())
        } else {
            pending.pop_front().or_else(|| schedules.pop_front())
        };
        let Some(forbidden) = next else {
            break;
        };
        let extractor = Extractor::compute_costs_from_rootsorts(
            Some(vec![sort.clone()]),
            &egraph,
            Excluding {
                costs: costs.clone(),
                forbidden: forbidden.clone(),
            },
        );
        let Some((cost, term)) = extractor.extract_best(&egraph, &mut terms, value) else {
            continue;
        };
        if cost.forbidden != 0 {
            continue;
        }
        let mut branches = Vec::new();
        let mut sites = Vec::new();
        let mut schedule = Vec::new();
        let mut families = BTreeMap::<String, Vec<Edge>>::new();
        for (value, edge) in edges(&egraph, &terms, term)? {
            let branching = *choices.entry(value).or_insert_with(|| {
                extractor
                    .extract_variants(&egraph, &mut TermDag::default(), value, 2)
                    .len()
                    > 1
            });
            if !branching {
                continue;
            }
            if let Some(logical) = super::matrix_family(&edge.head) {
                if edge.head != logical {
                    schedule.push(edge.clone());
                }
                let family: Vec<_> = std::iter::once(logical)
                    .chain(
                        super::matrix_constructors()
                            .iter()
                            .filter(|entry| entry.1 == logical)
                            .map(|entry| entry.0.as_str()),
                    )
                    .map(|name| Edge {
                        head: name.into(),
                        inputs: edge.inputs.clone(),
                    })
                    .collect();
                families
                    .entry(logical.into())
                    .or_default()
                    .extend(family.clone());
                sites.push(family);
            } else {
                families
                    .entry(edge.head.clone())
                    .or_default()
                    .push(edge.clone());
            }
            branches.push(edge);
        }
        // Exclude whole logical families before individual schedules. Adding
        // more tile sizes must not push the unfused form out of a small frontier.
        let exclusions = families
            .into_values()
            .filter(|edges| edges.len() > 1)
            .chain(sites)
            .chain(branches.into_iter().map(|edge| vec![edge]))
            .map(|edges| (false, edges))
            .chain((!schedule.is_empty()).then_some((true, schedule)));
        for (is_schedule, excluded) in exclusions {
            let mut next = forbidden.to_vec();
            next.extend(excluded);
            next.sort_unstable();
            next.dedup();
            if visited.contains(next.as_slice()) {
                continue;
            }
            if visited.len() == attempts {
                bounded = true;
            } else {
                let next = Arc::<[Edge]>::from(next);
                visited.insert(next.clone());
                if is_schedule {
                    schedules.push_back(next);
                } else {
                    pending.push_back(next);
                }
            }
        }
        if !expressions.insert(term) {
            continue;
        }
        let mut candidate = graph.deep_clone();
        let mut index = super::build_structural_index(&candidate);
        let terms_to_stamp = if roots.len() == 1 {
            vec![term]
        } else {
            match *terms.get(term) {
                Term::App(ref head, ref args) if head == "SearchOutputs" => args.clone(),
                _ => return Err("missing joint extraction roots".into()),
            }
        };
        for (&shift, ext_map) in segment.shifts.iter().zip(&ext_maps) {
            for (&root, &term) in roots.iter().zip(&terms_to_stamp) {
                Stamper {
                    g: &mut candidate,
                    index: &mut index,
                    seg_ids: &ids,
                    shift,
                    ext_map,
                    fusions: &mut Vec::new(),
                    memo: HashMap::new(),
                    requires_full_precision: full_precision,
                }
                .stamp_root(root + shift, &terms, term)?;
            }
        }
        super::sweep_dead_nodes(&mut candidate);
        result.push(Candidate {
            graph: candidate.into_toposort(),
            expression: terms.to_string(term),
        });
        if result.len() == limit {
            break;
        }
    }
    Ok(SearchSpace {
        candidates: result,
        truncated: bounded || !pending.is_empty() || !schedules.is_empty(),
    })
}

#[cfg(test)]
mod tests {
    use super::candidates;
    use crate::{Graph, graph::Op};

    fn has_fused_schedule(candidate: &super::Candidate, scheduled: bool) -> bool {
        candidate.graph.nodes().iter().any(|node| {
            matches!(node.op, Op::FusedMatMulAdd) && node.matmul_impl.is_some() == scheduled
        })
    }

    fn has_unfused_schedule(candidate: &super::Candidate, scheduled: bool) -> bool {
        candidate.graph.nodes().iter().any(|node| {
            matches!(node.op, Op::Add)
                && node.inputs.iter().any(|&id| {
                    let child = candidate.graph.node(id);
                    matches!(child.op, Op::MatMul) && child.matmul_impl.is_some() == scheduled
                })
        })
    }

    #[test]
    fn tile_equalities_reach_a_locked_kernel_without_changing_optimize() {
        let mut graph = Graph::new();
        let a = graph.input("a", &[50, 64]);
        let b = graph.parameter("b", &[64, 96]);
        let y = graph.matmul(a, b);
        graph.set_outputs(vec![y]);
        let space = candidates(&graph, Default::default(), 24).unwrap();
        let impls: Vec<_> = space
            .candidates
            .iter()
            .filter_map(|candidate| {
                candidate
                    .graph
                    .nodes()
                    .iter()
                    .find_map(|node| node.matmul_impl)
            })
            .collect();
        assert!(impls.len() >= 2, "{impls:?}");
        assert!(
            impls
                .iter()
                .any(|spec| spec.shape.cols() == 32 && spec.splits == 1)
        );
        assert!(
            impls.iter().any(|spec| spec.shape.k_stage == 16),
            "{impls:?}"
        );
        assert!(impls.iter().any(|spec| spec.splits == 8));
        for unroll in [false, true] {
            assert!(
                impls
                    .iter()
                    .any(|spec| spec.splits == 8 && spec.shape.unroll_k == unroll)
            );
        }
        for candidate in &space.candidates {
            let scheduled = candidate
                .graph
                .nodes()
                .iter()
                .any(|node| node.matmul_impl.is_some());
            let plan = crate::compile::compile(&candidate.graph);
            assert_eq!(
                plan.dispatches
                    .iter()
                    .any(|dispatch| dispatch.schedule_locked),
                scheduled,
                "lock mismatch for {}",
                candidate.expression
            );
        }
        let optimized = crate::optimize::optimize(&graph);
        assert!(
            optimized
                .nodes()
                .iter()
                .all(|node| node.matmul_impl.is_none())
        );
        for candidate in &space.candidates {
            let mut graph = candidate.graph.deep_clone();
            let expected = graph
                .nodes()
                .iter()
                .find(|n| matches!(n.op, Op::MatMul))
                .unwrap()
                .matmul_impl;
            let loss = graph.mean_all(graph.outputs()[0]);
            graph.set_outputs(vec![loss]);
            let differentiated = crate::autodiff::differentiate(&graph);
            assert!(
                differentiated
                    .nodes()
                    .iter()
                    .filter(|n| { matches!(n.op, Op::MatMul | Op::MatMulAT | Op::MatMulBT) })
                    .all(|n| n.matmul_impl == expected),
                "a scheduled contraction lost its derivative schedule"
            );
        }

        // Packing must retain a schedule for the hidden wide projection,
        // including transposed weights and the unpackable-input fallback.
        for transposed in [false, true] {
            for packed in [false, true] {
                for geglu in [false, true] {
                    let mut graph = Graph::new();
                    let h = graph.input("h", &[17, 64]);
                    let dims = if transposed { [96, 64] } else { [64, 96] };
                    let mut projection = |name| {
                        let w = if packed {
                            graph.parameter(name, &dims)
                        } else {
                            graph.input(name, &dims)
                        };
                        if transposed {
                            graph.matmul_bt(h, w)
                        } else {
                            graph.matmul(h, w)
                        }
                    };
                    let gate = projection("gate");
                    let up = projection("up");
                    let y = if geglu {
                        graph.geglu(gate, up)
                    } else {
                        graph.swiglu(gate, up)
                    };
                    graph.set_outputs(vec![y]);
                    let space = candidates(&graph, Default::default(), 4).unwrap();
                    let first = &space.candidates[0].graph;
                    assert_eq!(!first.derived_params.is_empty(), packed);
                    let products: Vec<_> = first
                        .nodes()
                        .iter()
                        .filter(|n| matches!(n.op, Op::MatMul | Op::MatMulBT))
                        .collect();
                    assert_eq!(products.len(), if packed { 1 } else { 2 });
                    assert!(products.iter().all(|n| n.matmul_impl.is_some()));
                    assert!(
                        products
                            .iter()
                            .all(|n| matches!(n.op, Op::MatMulBT) == transposed)
                    );
                }
            }
        }
    }

    #[test]
    #[ignore = "CPU search timing, run separately from correctness tests"]
    #[cfg(feature = "models")]
    fn cpu_search_overhead() {
        use crate::models::{smollm2, smolvla, whisper};
        for model in ["SmolLM2-135M", "SmolVLA", "Whisper-tiny"] {
            let mut graph = Graph::new();
            let output = match model {
                "SmolLM2-135M" => {
                    smollm2::build_graph(&mut graph, &smollm2::Config::smollm2_135m(), 128)
                }
                "SmolVLA" => smolvla::build_action_expert(
                    &mut graph,
                    &smolvla::Config::smolvla_base(),
                    50,
                    16,
                ),
                _ => whisper::build_encoder(&mut graph, &whisper::Config::whisper_tiny(), 1, 3000),
            };
            graph.set_outputs(vec![output]);
            let region = crate::outline::detect_repeated_regions(&graph)[0];
            for extraction_cost in [
                crate::optimize::ExtractionCost::TensorTraffic,
                crate::optimize::ExtractionCost::AstSize,
            ] {
                for sample in 0..6 {
                    let start = std::time::Instant::now();
                    let config = crate::OptimizeConfig {
                        extraction_cost,
                        ..Default::default()
                    };
                    let space = super::candidates(&graph, config, 4)
                        .or_else(|_| super::repeated_candidates(&graph, region, config, 4))
                        .unwrap();
                    println!(
                        "{model} cost={extraction_cost:?} sample={sample} candidates={} truncated={} ms={:.3}",
                        space.candidates.len(),
                        space.truncated,
                        start.elapsed().as_secs_f64() * 1000.0
                    );
                    assert!(!space.candidates.is_empty());
                }
            }
        }
    }

    #[test]
    fn keeps_fused_and_unfused_implementations_before_kernel_tuning() {
        let mut graph = Graph::new();
        let a = graph.input("a", &[3, 7]);
        let b = graph.parameter("b", &[7, 5]);
        let c = graph.input("c", &[3, 5]);
        let product = graph.matmul(a, b);
        let out = graph.add(product, c);
        graph.set_outputs(vec![out]);
        let space = candidates(&graph, Default::default(), 8).unwrap();
        let choices = space.candidates;
        assert!(
            choices
                .iter()
                .any(|c| matches!(c.graph.node(c.graph.outputs()[0]).op, Op::Add))
        );
        assert!(
            choices
                .iter()
                .any(|c| matches!(c.graph.node(c.graph.outputs()[0]).op, Op::FusedMatMulAdd))
        );
        assert!(
            choices
                .iter()
                .all(|c| c.graph.node(c.graph.outputs()[0]).ty.shape == [3, 5])
        );
        assert!(candidates(&graph, Default::default(), 0).is_err());
        let optimized = crate::optimize::optimize(&graph);
        let recovered = candidates(&optimized, Default::default(), 8).unwrap();
        assert!(recovered.candidates.iter().any(|candidate| {
            matches!(
                candidate.graph.node(candidate.graph.outputs()[0]).op,
                Op::Add
            )
        }));
        graph.set_outputs(vec![product, out]);
        let space = candidates(&graph, Default::default(), 8).unwrap();
        assert!(
            space
                .candidates
                .iter()
                .all(|c| c.graph.outputs().len() == 2)
        );
        graph.nodes_mut()[out as usize].requires_full_precision = true;
        assert!(candidates(&graph, Default::default(), 8).is_err());
    }

    #[test]
    fn explores_inner_choices_without_committing_to_the_cheapest_child() {
        let mut graph = Graph::new();
        let a = graph.input("a", &[3, 7]);
        let b = graph.parameter("b", &[7, 5]);
        let c = graph.input("c", &[3, 5]);
        let mm = graph.matmul(a, b);
        let add = graph.add(mm, c);
        let out = graph.neg(add);
        graph.set_outputs(vec![out]);
        let space = candidates(&graph, Default::default(), 8).unwrap();
        for scheduled in [false, true] {
            assert!(
                space
                    .candidates
                    .iter()
                    .any(|c| has_fused_schedule(c, scheduled))
            );
            assert!(
                space
                    .candidates
                    .iter()
                    .any(|c| has_unfused_schedule(c, scheduled)),
                "unfused schedule {scheduled} was crowded out"
            );
        }
        assert!(candidates(&graph, Default::default(), 1).unwrap().truncated);

        let d = graph.input("d", &[3, 7]);
        let e = graph.parameter("e", &[7, 5]);
        let f = graph.input("f", &[3, 5]);
        let mm2 = graph.matmul(d, e);
        let add2 = graph.add(mm2, f);
        let out = graph.mul(add, add2);
        graph.set_outputs(vec![out]);
        let space = candidates(&graph, Default::default(), 16).unwrap();
        assert!(space.truncated);
        assert!(space.candidates.len() > 4);
        let tiles: std::collections::HashSet<_> = space
            .candidates
            .iter()
            .flat_map(|candidate| candidate.graph.nodes().iter().filter_map(|n| n.matmul_impl))
            .collect();
        assert!(
            tiles.len() >= 2,
            "tile equalities did not survive extraction: {tiles:?}"
        );
        let bounded = candidates(&graph, Default::default(), 2).unwrap();
        assert!(bounded.truncated);
        assert_eq!(bounded.candidates.len(), 2);
        assert!(
            bounded
                .candidates
                .iter()
                .any(|c| has_fused_schedule(c, true))
        );
        assert!(
            bounded
                .candidates
                .iter()
                .any(|c| has_unfused_schedule(c, true))
        );
        assert_ne!(
            bounded.candidates[0].expression,
            bounded.candidates[1].expression
        );
        let bounded = candidates(&graph, Default::default(), 4).unwrap();
        let layouts: std::collections::HashSet<_> = bounded
            .candidates
            .iter()
            .flat_map(|candidate| candidate.graph.nodes().iter().filter_map(|n| n.matmul_impl))
            .collect();
        assert!(
            layouts.len() >= 2,
            "fusion choices crowded out schedule coverage"
        );

        // Search only the second pair. External inputs keep their identities,
        // and the first independent pair is not rewritten as a side effect.
        let region = super::region_candidates(
            &graph,
            mm2 as usize..add2 as usize + 1,
            Default::default(),
            8,
        )
        .unwrap();
        assert!(region.candidates.len() >= 2);
        assert!(region.candidates.iter().all(|candidate| {
            let untouched = candidate
                .graph
                .nodes()
                .iter()
                .find(|node| matches!(&node.op, Op::Parameter { name } if name == "b"))
                .unwrap()
                .id;
            candidate.graph.nodes().iter().any(|node| {
                matches!(node.op, Op::MatMul)
                    && node.matmul_impl.is_none()
                    && node.inputs.contains(&untouched)
            })
        }));
        assert!(super::region_candidates(&graph, 0..0, Default::default(), 8).is_err());

        let mut graph = Graph::new();
        let mut h = graph.input("x", &[3, 8]);
        for layer in 0..10 {
            let w = graph.parameter(&format!("w{layer}"), &[8, 8]);
            let c = graph.parameter(&format!("c{layer}"), &[3, 8]);
            let mm = graph.matmul(h, w);
            h = graph.add(mm, c);
        }
        graph.set_outputs(vec![h]);
        let region = crate::outline::detect_repeated_regions(&graph)[0];
        let space = super::repeated_candidates(&graph, region, Default::default(), 8).unwrap();
        assert!(space.truncated);
        assert!(space.candidates.len() >= 2);
        assert!(space.candidates.iter().any(|candidate| {
            candidate
                .graph
                .nodes()
                .iter()
                .filter(|node| matches!(node.op, Op::FusedMatMulAdd) && node.matmul_impl.is_some())
                .count()
                == region.count
        }));
        let context = graph.input("context", &[16, 8]);
        let weight = graph.parameter("context_weight", &[8, 8]);
        let projection = graph.matmul(context, weight);
        graph.set_outputs(vec![h, projection]);
        let sparse = candidates(
            &graph,
            crate::OptimizeConfig {
                saturation_cutoff: 32,
                ..Default::default()
            },
            4,
        )
        .unwrap();
        let scheduled = &sparse.candidates[0].graph;
        let independent = scheduled.outputs()[1];
        assert!(
            scheduled.node(independent).matmul_impl.is_some(),
            "independent projections must not be opaque merely because the model is large"
        );
        assert!(
            scheduled
                .nodes()
                .iter()
                .filter(|node| matches!(node.op, Op::MatMul | Op::FusedMatMulAdd))
                .all(|node| node.matmul_impl.is_some())
        );
        graph.nodes_mut()[region.start + region.period + 2].requires_full_precision = true;
        assert!(super::repeated_candidates(&graph, region, Default::default(), 8).is_err());
    }
}
