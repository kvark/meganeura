//! Bounded equivalent-region extraction for measured implementation selection.
//!
//! This uses the existing egglog rules and graph reconstruction. It deliberately
//! accepts one small, single-output region: timing unrelated roots independently
//! would double-count shared work. Each candidate must be lowered and tuned before
//! comparing its complete execution, not ranked by its untuned kernel timings.

use super::{FusionCostModel, Segment, Stamper};
use crate::{Graph, graph::Op};
use egglog::{
    Term, TermDag, TermId, Value,
    extract::{CostModel, Extractor},
};
use std::collections::{HashMap, HashSet, VecDeque};

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
        let value = match terms.get(id) {
            Term::App(head, args) => {
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
    if limit == 0 || graph.nodes().len() > super::SATURATION_CUTOFF {
        return Err("expected a bounded region and a positive candidate limit".into());
    }
    let &[root] = graph.outputs() else {
        return Err("region search requires one observable output".into());
    };
    if graph
        .nodes()
        .iter()
        .any(|node| node.inputs.iter().any(|&id| id >= node.id))
    {
        return Err("region search requires topologically ordered nodes".into());
    }
    let segment = Segment {
        ids: graph
            .nodes()
            .iter()
            .filter(|n| !matches!(n.op, Op::Nop))
            .map(|n| n.id as usize)
            .collect(),
        shifts: vec![0],
    };
    let (program, _) = super::segment_program(graph, &segment);
    let mut egraph = egglog::EGraph::default();
    egraph
        .parse_and_run_program(None, &program)
        .map_err(|e| e.to_string())?;
    let (sort, value) = egraph
        .eval_expr(&egglog::ast::Expr::Var(
            egglog::ast::Span::Panic,
            format!("$n{root}"),
        ))
        .map_err(|e| e.to_string())?;
    let costs = FusionCostModel::with_sizes(super::eclass_sizes(
        graph,
        &mut egraph,
        segment.ids.iter().copied(),
    ));
    let ids: HashSet<_> = segment.ids.iter().copied().collect();
    let mut pending = VecDeque::from([Vec::new()]);
    let mut visited = HashSet::from([Vec::new()]);
    let mut expressions = HashSet::new();
    let mut choices = HashMap::new();
    let mut result = Vec::new();
    for _ in 0..limit.saturating_mul(segment.ids.len()) {
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
            if visited.insert(next.clone()) {
                pending.push_back(next);
            }
        }
        let expression = terms.to_string(term);
        if !expressions.insert(expression.clone()) {
            continue;
        }
        let mut candidate = graph.deep_clone();
        let mut index = super::build_structural_index(&candidate);
        Stamper {
            g: &mut candidate,
            index: &mut index,
            seg_ids: &ids,
            shift: 0,
            ext_map: &HashMap::new(),
            fusions: &mut Vec::new(),
            memo: HashMap::new(),
            requires_full_precision: graph.node(root).requires_full_precision,
        }
        .stamp_root(root as usize, &terms, term)?;
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
        truncated: !pending.is_empty(),
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
    }
}
