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
    collections::{HashMap, HashSet, VecDeque},
    ops::Range,
};

pub struct Candidate {
    pub graph: Graph,
    pub expression: String,
}

pub struct SearchSpace {
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
    forbidden: Vec<Edge>,
}

impl CostModel<Cost> for Excluding {
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
                    .binary_search(&Edge {
                        head: func.name().to_string(),
                        inputs: row.vals[..row.vals.len() - 1].to_vec(),
                    })
                    .is_ok(),
            ),
            estimate: self.costs.enode_cost(egraph, func, row).max(1),
        }
    }
}

fn edges(
    egraph: &mut egglog::EGraph,
    terms: &TermDag,
    root: TermId,
) -> Result<Vec<(Value, Edge)>, String> {
    fn visit(
        egraph: &mut egglog::EGraph,
        terms: &TermDag,
        id: TermId,
        values: &mut HashMap<TermId, Value>,
        edges: &mut Vec<(Value, Edge)>,
    ) -> Result<Value, String> {
        if let Some(&value) = values.get(&id) {
            return Ok(value);
        }
        let value = match *terms.get(id) {
            Term::App(ref head, ref args) => {
                let inputs = args
                    .iter()
                    .map(|&arg| visit(egraph, terms, arg, values, edges))
                    .collect::<Result<Vec<_>, _>>()?;
                let value = egraph
                    .lookup_function(head, &inputs)
                    .ok_or("extracted term is missing from the e-graph")?;
                if head != "Leaf" {
                    edges.push((
                        value,
                        Edge {
                            head: head.clone(),
                            inputs,
                        },
                    ));
                }
                value
            }
            _ => {
                egraph
                    .eval_expr(&terms.term_to_expr(&id, egglog::ast::Span::Panic))
                    .map_err(|e| e.to_string())?
                    .1
            }
        };
        values.insert(id, value);
        Ok(value)
    }
    let mut result = Vec::new();
    visit(egraph, terms, root, &mut HashMap::new(), &mut result)?;
    Ok(result)
}

/// Retain equivalent implementations before lowering or kernel tuning.
///
/// Egglog's extractor reconstructs each candidate. Excluding a selected e-node
/// exposes alternative choices, including inside children. This is bounded
/// enumeration, not a globally optimal or k-best schedule search. The estimate
/// orders exploration only; callers must measure complete lowered candidates.
/// `truncated` reports an unfinished search. No GPU measurement happens here.
pub fn candidates(graph: &Graph, limit: usize) -> Result<SearchSpace, String> {
    region_candidates(graph, 0..graph.nodes().len(), limit)
}

/// Explore a contiguous region without changing the surrounding graph. Its
/// escaping values are extracted together; dependencies outside the region stay
/// opaque. Returned graphs preserve the original inputs, parameters and outputs.
pub fn region_candidates(
    graph: &Graph,
    region: Range<usize>,
    limit: usize,
) -> Result<SearchSpace, String> {
    if limit == 0
        || region.is_empty()
        || region.len() > super::SATURATION_CUTOFF
        || region.end > graph.nodes().len()
    {
        return Err("expected a bounded region and a positive candidate limit".into());
    }
    if graph
        .nodes()
        .iter()
        .any(|node| node.inputs.iter().any(|&id| id >= node.id))
    {
        return Err("region search requires topologically ordered nodes".into());
    }
    let segment = Segment {
        ids: region
            .filter(|&id| !matches!(graph.nodes()[id].op, Op::Nop))
            .collect(),
        shifts: vec![0],
    };
    let roots = super::segment_roots(graph, &segment);
    if roots.is_empty() {
        return Err("region has no observable output".into());
    }
    let full_precision = graph.node(roots[0] as u32).requires_full_precision;
    for &id in &segment.ids {
        let node = &graph.nodes()[id];
        if matches!(
            node.op,
            Op::Input { .. } | Op::Parameter { .. } | Op::Constant { .. }
        ) {
            continue;
        }
        if matches!(
            node.op,
            Op::CacheWrite | Op::CacheWritePrefix | Op::ScatterAdd { .. }
        ) {
            return Err("stateful regions need an explicit mutation contract".into());
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
    let mut egraph = egglog::EGraph::default();
    egraph
        .parse_and_run_program(None, &program)
        .map_err(|e| e.to_string())?;
    let (sort, value) = egraph
        .eval_expr(&egglog::ast::Expr::Var(egglog::ast::Span::Panic, root_name))
        .map_err(|e| e.to_string())?;
    let costs = FusionCostModel::with_sizes(super::eclass_sizes(
        graph,
        &mut egraph,
        segment.ids.iter().chain(&externals).copied(),
    ));
    let ids: HashSet<_> = segment.ids.iter().copied().collect();
    let ext_map: HashMap<_, _> = externals.iter().map(|&id| (id, id as u32)).collect();
    let mut pending = VecDeque::from([Vec::new()]);
    let mut visited = HashSet::from([Vec::new()]);
    let mut expressions = HashSet::new();
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
        let mut terms = TermDag::default();
        let Some((cost, term)) = extractor.extract_best(&egraph, &mut terms, value) else {
            continue;
        };
        if cost.forbidden != 0 {
            continue;
        }
        for (value, edge) in edges(&mut egraph, &terms, term)? {
            let branching = *choices.entry(value).or_insert_with(|| {
                extractor
                    .extract_variants(&egraph, &mut TermDag::default(), value, 2)
                    .len()
                    > 1
            });
            if !branching {
                continue;
            }
            let mut next = forbidden.clone();
            next.push(edge);
            next.sort_unstable();
            next.dedup();
            if visited.contains(&next) {
                continue;
            }
            if visited.len() == attempts {
                bounded = true;
            } else {
                visited.insert(next.clone());
                pending.push_back(next);
            }
        }
        let expression = terms.to_string(term);
        if !expressions.insert(expression.clone()) {
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
        for (&root, term) in roots.iter().zip(terms_to_stamp) {
            Stamper {
                g: &mut candidate,
                index: &mut index,
                seg_ids: &ids,
                shift: 0,
                ext_map: &ext_map,
                fusions: &mut Vec::new(),
                memo: HashMap::new(),
                requires_full_precision: graph.node(root as u32).requires_full_precision,
            }
            .stamp_root(root, &terms, term)?;
        }
        super::sweep_dead_nodes(&mut candidate);
        result.push(Candidate {
            graph: candidate.into_toposort(),
            expression,
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
    fn keeps_fused_and_unfused_implementations_before_kernel_tuning() {
        let mut graph = Graph::new();
        let a = graph.input("a", &[3, 7]);
        let b = graph.parameter("b", &[7, 5]);
        let c = graph.input("c", &[3, 5]);
        let product = graph.matmul(a, b);
        let out = graph.add(product, c);
        graph.set_outputs(vec![out]);
        let space = candidates(&graph, 8).unwrap();
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
        assert!(candidates(&graph, 0).is_err());
        graph.set_outputs(vec![product, out]);
        let space = candidates(&graph, 8).unwrap();
        assert!(!space.truncated);
        assert!(
            space
                .candidates
                .iter()
                .all(|c| c.graph.outputs().len() == 2)
        );
        graph.nodes_mut()[out as usize].requires_full_precision = true;
        assert!(candidates(&graph, 8).is_err());
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
        let space = candidates(&graph, 8).unwrap();
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
        assert!(candidates(&graph, 1).unwrap().truncated);

        let d = graph.input("d", &[3, 7]);
        let e = graph.parameter("e", &[7, 5]);
        let f = graph.input("f", &[3, 5]);
        let mm2 = graph.matmul(d, e);
        let add2 = graph.add(mm2, f);
        let out = graph.mul(add, add2);
        graph.set_outputs(vec![out]);
        let space = candidates(&graph, 16).unwrap();
        assert!(!space.truncated);
        let forms: std::collections::HashSet<_> = space
            .candidates
            .iter()
            .map(|candidate| candidate.expression.matches("FusedMatMulAdd").count())
            .collect();
        assert_eq!(forms, [0, 1, 2].into_iter().collect());
        assert_eq!(space.candidates.len(), 4);

        // Search only the second pair. External inputs keep their identities,
        // and the first independent pair is not rewritten as a side effect.
        let region = super::region_candidates(&graph, mm2 as usize..add2 as usize + 1, 8).unwrap();
        assert_eq!(region.candidates.len(), 2);
        assert!(
            region
                .candidates
                .iter()
                .all(|c| c.expression.matches("FusedMatMulAdd").count() <= 1)
        );
        assert!(super::region_candidates(&graph, 0..0, 8).is_err());
    }
}
