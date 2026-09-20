//! Bounded equivalent-region extraction for measured implementation selection.
//!
//! This uses the existing egglog rules and graph reconstruction. It deliberately
//! accepts one small, single-output region: timing unrelated roots independently
//! would double-count shared work. Each candidate must be lowered and tuned before
//! comparing its complete execution, not ranked by its untuned kernel timings.

use super::{FusionCostModel, Segment, Stamper};
use crate::{Graph, graph::Op};
use egglog::{TermDag, extract::Extractor};
use std::collections::{HashMap, HashSet};

pub struct Candidate {
    pub graph: Graph,
    pub expression: String,
}

/// Retain root implementation alternatives instead of immediately extracting one.
///
/// This is not a global or k-best graph search: egglog extracts the cheapest
/// children of each root alternative. The limit bounds root alternatives, not
/// saturation. Inputs must therefore already be a bounded, topologically sorted
/// region. Kernel configurations are searched separately for *each* candidate.
pub fn candidates(graph: &Graph, limit: usize) -> Result<Vec<Candidate>, String> {
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
    let extractor = Extractor::compute_costs_from_rootsorts(Some(vec![sort]), &egraph, costs);
    let mut terms = TermDag::default();
    let variants = extractor.extract_variants(&egraph, &mut terms, value, limit);
    let ids: HashSet<_> = segment.ids.iter().copied().collect();
    let mut result = Vec::new();
    for (_, term) in variants {
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
            expression: terms.to_string(term),
        });
    }
    Ok(result)
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
        let choices = candidates(&graph, 8).unwrap();
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
}
