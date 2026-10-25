use super::{BufferRef, ExecutionPlan};
use std::collections::{HashMap, HashSet};

#[derive(Debug)]
pub(crate) struct Preparation {
    pub groups: usize,
    pub inputs: HashSet<BufferRef>,
    pub ready: HashSet<BufferRef>,
}

/// Schedule the requested output dependencies before all remaining dispatches.
/// This runs before memory planning, so values crossing the boundary retain
/// their allocation until their last consumer. Fusion cannot cross stages.
pub(crate) fn schedule(
    plan: &mut ExecutionPlan,
    outputs: &[usize],
    serial: bool,
    fuse: bool,
) -> Preparation {
    assert!(!outputs.is_empty());
    let mut producers = HashMap::new();
    for (index, dispatch) in plan.dispatches.iter().enumerate() {
        for buffer in std::iter::once(&dispatch.output_buffer).chain(&dispatch.extra_outputs) {
            assert!(
                producers.insert(*buffer, index).is_none(),
                "prepared execution requires single-writer buffers"
            );
        }
    }
    let inputs: HashSet<_> = plan.input_buffers.iter().map(|entry| entry.1).collect();
    let mut pending: Vec<_> = outputs
        .iter()
        .map(|&i| {
            *plan
                .output_buffers
                .get(i)
                .expect("unknown preparation output")
        })
        .collect();
    let mut early = vec![false; plan.dispatches.len()];
    let mut ready = HashSet::new();
    while let Some(buffer) = pending.pop() {
        if !ready.insert(buffer) {
            continue;
        }
        if let Some(&producer) = producers.get(&buffer) {
            early[producer] = true;
            pending.extend(plan.dispatches[producer].input_buffers.iter().copied());
        }
    }
    let early_inputs = inputs.intersection(&ready).copied().collect();
    let mut prefix = Vec::new();
    let mut suffix = Vec::new();
    for (index, dispatch) in std::mem::take(&mut plan.dispatches).into_iter().enumerate() {
        if early[index] {
            ready.insert(dispatch.output_buffer);
            ready.extend(dispatch.extra_outputs.iter().copied());
            prefix.push(dispatch);
        } else {
            suffix.push(dispatch);
        }
    }
    let mut groups = super::schedule_sequence(&mut prefix, serial, fuse);
    let prefix_groups = groups.len();
    let suffix_groups = super::schedule_sequence(&mut suffix, serial, fuse);
    let offset = prefix.len();
    groups.extend(
        suffix_groups
            .into_iter()
            .map(|range| range.start + offset..range.end + offset),
    );
    prefix.extend(suffix);
    plan.dispatches = prefix;
    plan.groups = groups;
    Preparation {
        groups: prefix_groups,
        inputs: early_inputs,
        ready,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn plan() -> ExecutionPlan {
        let mut graph = crate::Graph::new();
        let x = graph.input("x", &[4, 4]);
        let w = graph.parameter("w", &[4, 4]);
        let y = graph.matmul(x, w);
        let target = graph.input("late", &[4, 4]);
        let negative = graph.neg(target);
        let error = graph.add(y, negative);
        let squared = graph.mul(error, error);
        let loss = graph.mean_all(squared);
        graph.set_outputs(vec![loss, y]);
        crate::compile::compile(&crate::autodiff::differentiate(&graph))
    }
    #[test]
    fn closure_keeps_late_targets_and_gradients_after_preparation() {
        for (serial, fuse) in [(false, false), (false, true), (true, false)] {
            let mut p = plan();
            let before = p.dispatches.len();
            let layout = schedule(&mut p, &[1, 1], serial, fuse);
            assert!(layout.groups > 0 && layout.groups < p.groups.len());
            let prefix_end = p.groups[layout.groups - 1].end;
            let early_input = p.input_buffers.iter().find(|(n, _)| n == "x").unwrap().1;
            let late = p.input_buffers.iter().find(|(n, _)| n == "late").unwrap().1;
            assert!(layout.inputs.contains(&early_input));
            assert!(!layout.inputs.contains(&late));
            assert!(layout.ready.contains(&p.output_buffers[1]));
            assert!(!layout.ready.contains(&p.output_buffers[0]));
            assert!(
                p.dispatches[..prefix_end]
                    .iter()
                    .all(|d| !d.input_buffers.contains(&late))
            );
            assert!(
                p.param_grad_pairs
                    .iter()
                    .all(|(_, gradient)| !layout.ready.contains(gradient))
            );
            assert!(p.dispatches.len() <= before);
            assert_eq!(p.groups.last().unwrap().end, p.dispatches.len());
        }
    }
    #[test]
    #[should_panic(expected = "unknown preparation output")]
    fn rejects_unknown_output() {
        schedule(&mut plan(), &[42], false, true);
    }
    #[test]
    #[should_panic(expected = "single-writer")]
    fn rejects_ambiguous_writes() {
        let mut p = plan();
        p.dispatches.push(p.dispatches[0].clone());
        schedule(&mut p, &[1], false, true);
    }
}
