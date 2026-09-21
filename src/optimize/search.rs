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
}

impl egglog::extract::Cost for Cost {
    fn identity() -> Self {
        Self {
            forbidden: 0,
            estimate: 0,
        }
    }
    fn unit() -> Self {
        Self {
            forbidden: 0,
            estimate: 1,
        }
    }
    fn combine(self, other: &Self) -> Self {
        Self {
            forbidden: self.forbidden.saturating_add(other.forbidden),
            estimate: self.estimate.saturating_add(other.estimate),
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
            estimate: self.costs.enode_cost(egraph, func, row).max(1),
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
    region_candidates(graph, 0..graph.nodes().len(), config, limit)
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
    let mut egraph = super::rule_graph(config.pack_swiglu);
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
    let mut visited = HashSet::from([empty]);
    let mut expressions = HashSet::new();
    let mut terms = TermDag::default();
    let mut choices = HashMap::new();
    let mut result = Vec::new();
    let attempts = limit.saturating_mul(segment.ids.len());
    let mut bounded = false;
    for _ in 0..attempts {
        let Some(forbidden) = pending.pop_front() else {
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
            families
                .entry(edge.head.clone())
                .or_default()
                .push(edge.clone());
            branches.push(edge);
        }
        // Visit whole-constructor alternatives before their individual sites.
        // Otherwise a small bound explores many nearly identical partial
        // unfusions and can miss the fully unfused family entirely.
        let exclusions = families
            .into_values()
            .filter(|edges| edges.len() > 1)
            .chain(branches.into_iter().map(|edge| vec![edge]));
        for excluded in exclusions {
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
                pending.push_back(next);
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
        truncated: bounded || !pending.is_empty(),
    })
}

#[cfg(test)]
mod tests {
    use super::candidates;
    use crate::{Graph, graph::Op};

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
            for sample in 0..6 {
                let start = std::time::Instant::now();
                let space =
                    super::repeated_candidates(&graph, region, Default::default(), 4).unwrap();
                println!(
                    "{model} sample={sample} region_nodes={} candidates={} truncated={} ms={:.3}",
                    region.period,
                    space.candidates.len(),
                    space.truncated,
                    start.elapsed().as_secs_f64() * 1000.0
                );
                assert!(!space.candidates.is_empty());
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
        assert!(!space.truncated);
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
        assert!(!space.truncated);
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
        assert!(!space.truncated);
        assert!(
            space
                .candidates
                .iter()
                .any(|c| c.expression.starts_with("(Neg (FusedMatMulAdd"))
        );
        assert!(
            space
                .candidates
                .iter()
                .any(|c| c.expression.starts_with("(Neg (Add (MatMul"))
        );
        assert!(candidates(&graph, Default::default(), 1).unwrap().truncated);

        let d = graph.input("d", &[3, 7]);
        let e = graph.parameter("e", &[7, 5]);
        let f = graph.input("f", &[3, 5]);
        let mm2 = graph.matmul(d, e);
        let add2 = graph.add(mm2, f);
        let out = graph.mul(add, add2);
        graph.set_outputs(vec![out]);
        let space = candidates(&graph, Default::default(), 16).unwrap();
        assert!(!space.truncated);
        let forms: std::collections::HashSet<_> = space
            .candidates
            .iter()
            .map(|candidate| candidate.expression.matches("FusedMatMulAdd").count())
            .collect();
        assert_eq!(forms, [0, 1, 2].into_iter().collect());
        assert_eq!(space.candidates.len(), 4);
        let bounded = candidates(&graph, Default::default(), 2).unwrap();
        assert!(bounded.truncated);
        assert_eq!(
            bounded
                .candidates
                .iter()
                .map(|c| c.expression.matches("FusedMatMulAdd").count())
                .collect::<Vec<_>>(),
            [2, 0],
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
        assert_eq!(region.candidates.len(), 2);
        assert!(
            region
                .candidates
                .iter()
                .all(|c| c.expression.matches("FusedMatMulAdd").count() <= 1)
        );
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
        assert!(!space.truncated);
        assert_eq!(space.candidates.len(), 2);
        assert!(space.candidates.iter().any(|candidate| {
            candidate
                .graph
                .nodes()
                .iter()
                .filter(|node| matches!(node.op, Op::FusedMatMulAdd))
                .count()
                == region.count
        }));
        graph.nodes_mut()[region.start + region.period + 2].requires_full_precision = true;
        assert!(super::repeated_candidates(&graph, region, Default::default(), 8).is_err());
    }
}
