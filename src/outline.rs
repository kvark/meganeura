//! Detection of repeated subgraph regions (e.g. transformer layers).
//!
//! Real model graphs are repetitive: N structurally identical blocks
//! differing only in parameter bindings, laid out contiguously in node-id
//! order by the model builders (and preserved by autodiff, which emits
//! backward nodes layer by layer). Detecting that repetition lets the
//! optimizer run e-graph equality saturation on *one* block instance —
//! well under the saturation size cutoff — and apply the extractor's
//! decisions to every instance, instead of skipping egglog entirely on
//! graphs over the cutoff.
//!
//! Detection is two-phase: a cheap per-node structural signature finds
//! candidate (start, period, count) lattices by sequence periodicity,
//! then exact verification checks op/type equality (parameter and input
//! names wildcarded) and edge isomorphism — every edge must either shift
//! with the instance (in-block and chain edges) or point at the same
//! shared global node for all instances.

use crate::graph::{Graph, Op};
use std::hash::{Hash, Hasher};

/// Longest block period considered. Also bounds the outlined egglog
/// program size, keeping saturation fast (the full-graph cutoff is 300).
const MAX_PERIOD: usize = 300;
/// Minimum nodes a region must cover to be worth outlining.
const MIN_COVERAGE: usize = 32;
/// Maximum number of disjoint regions reported (forward + backward
/// regions of a training graph, plus slack).
const MAX_REGIONS: usize = 4;

/// A repeated region: `count` instances of `period` nodes, the first
/// starting at node id `start`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Region {
    pub start: usize,
    pub period: usize,
    pub count: usize,
}

impl Region {
    pub fn len(&self) -> usize {
        self.period * self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    fn overlaps(&self, other: &Region) -> bool {
        self.start < other.start + other.len() && other.start < self.start + self.len()
    }
}

/// Structural signature of a node: op kind + op params + tensor type,
/// with parameter/input names wildcarded (each block instance binds its
/// own weights). Constants hash their data — instances must share exact
/// constant values to be considered equivalent.
fn node_signature(graph: &Graph, id: usize) -> u64 {
    let node = &graph.nodes()[id];
    let mut h = std::collections::hash_map::DefaultHasher::new();
    match node.op {
        Op::Parameter { .. } => "P".hash(&mut h),
        Op::Input { .. } => "I".hash(&mut h),
        Op::Constant { ref data } => {
            "C".hash(&mut h);
            data.len().hash(&mut h);
            for v in data {
                v.to_bits().hash(&mut h);
            }
        }
        // Debug formatting includes op params (num_heads, eps, strides…)
        // so structurally different ops hash differently.
        ref op => format!("{:?}", op).hash(&mut h),
    }
    node.ty.shape.hash(&mut h);
    format!("{:?}", node.ty.dtype).hash(&mut h);
    node.inputs.len().hash(&mut h);
    h.finish()
}

/// Ops are equivalent across instances if equal up to parameter/input
/// names. The Debug-string comparison covers everything else, including
/// constants (their data is part of the Debug output, so instances must
/// share exact constant values).
fn ops_equivalent(a: &Op, b: &Op) -> bool {
    match (a, b) {
        (&Op::Parameter { .. }, &Op::Parameter { .. }) => true,
        (&Op::Input { .. }, &Op::Input { .. }) => true,
        (ref a, ref b) => format!("{:?}", a) == format!("{:?}", b),
    }
}

/// Check that instance `m` and instance `m+1` of the lattice are exact
/// structural copies: equivalent ops and types at each offset, and every
/// input edge either shifts by one period (in-block and chain edges —
/// instance m+1 reading from instance m is `ia + period`) or points at
/// the same shared global node (embeddings, masks, …).
///
/// Comparing *consecutive* pairs matters: the first instance's incoming
/// chain edge points at whatever pre-region node produced the initial
/// hidden state, which is generally not at a lattice position — so a
/// head instance that doesn't pair with its successor is simply dropped
/// rather than invalidating the whole region.
fn pair_isomorphic(graph: &Graph, start: usize, period: usize, m: usize) -> bool {
    let nodes = graph.nodes();
    if start + (m + 2) * period > nodes.len() {
        return false;
    }
    for off in 0..period {
        let a = &nodes[start + m * period + off];
        let b = &nodes[start + (m + 1) * period + off];
        if !ops_equivalent(&a.op, &b.op) || a.ty != b.ty || a.inputs.len() != b.inputs.len() {
            return false;
        }
        for (&ia, &ib) in a.inputs.iter().zip(b.inputs.iter()) {
            let ia = ia as usize;
            let ib = ib as usize;
            if ib == ia + period {
                continue; // lattice edge (in-block or chain)
            }
            if ib == ia {
                continue; // shared global
            }
            return false;
        }
    }
    true
}

/// Refine a candidate lattice to its longest run of consecutive
/// isomorphic instances, or `None` if fewer than two remain.
fn refine_region(graph: &Graph, cand: &Region) -> Option<Region> {
    let mut best: Option<Region> = None;
    let mut run_start = 0;
    let mut run_len = 0; // number of valid consecutive pairs in the run
    for m in 0..cand.count - 1 {
        if pair_isomorphic(graph, cand.start, cand.period, m) {
            if run_len == 0 {
                run_start = m;
            }
            run_len += 1;
        } else {
            run_len = 0;
        }
        if run_len > 0 && best.is_none_or(|b| run_len + 1 > b.count) {
            best = Some(Region {
                start: cand.start + run_start * cand.period,
                period: cand.period,
                count: run_len + 1,
            });
        }
    }
    best
}

/// Find disjoint repeated regions, largest coverage first.
pub fn detect_repeated_regions(graph: &Graph) -> Vec<Region> {
    let n = graph.nodes().len();
    if n < MIN_COVERAGE {
        return Vec::new();
    }
    let sigs: Vec<u64> = (0..n).map(|i| node_signature(graph, i)).collect();

    // Phase 1: collect candidate lattices from signature periodicity.
    // For each period p, maximal runs of `sigs[i] == sigs[i + p]` of
    // length >= p give `run/p + 1` repetitions starting at the run head.
    let mut candidates: Vec<Region> = Vec::new();
    for period in 1..=MAX_PERIOD.min(n / 2) {
        let mut i = 0;
        while i + period < n {
            if sigs[i] != sigs[i + period] {
                i += 1;
                continue;
            }
            let run_start = i;
            while i + period < n && sigs[i] == sigs[i + period] {
                i += 1;
            }
            let run = i - run_start;
            if run >= period {
                candidates.push(Region {
                    start: run_start,
                    period,
                    count: run / period + 1,
                });
            }
            i += 1;
        }
    }
    // Largest coverage first; prefer the smaller (fundamental) period on
    // ties — a 12-layer model also matches at periods of 2, 3, 4… layers
    // with the same coverage, and the one-layer block is the cheapest to
    // saturate while exposing the same in-block patterns.
    candidates.sort_by_key(|r| (std::cmp::Reverse(r.len()), r.period));

    // Phase 2: exact verification, keeping each candidate's longest run
    // of consecutive isomorphic instances.
    let mut accepted: Vec<Region> = Vec::new();
    for cand in candidates {
        if accepted.len() >= MAX_REGIONS {
            break;
        }
        if cand.period > MAX_PERIOD || cand.len() < MIN_COVERAGE {
            continue;
        }
        if accepted.iter().any(|r| r.overlaps(&cand)) {
            continue;
        }
        if let Some(region) = refine_region(graph, &cand) {
            if region.len() >= MIN_COVERAGE && !accepted.iter().any(|r| r.overlaps(&region)) {
                accepted.push(region);
            }
        }
    }
    accepted
}

#[cfg(test)]
mod tests {
    use super::*;

    /// hidden -> [matmul, bias_add, relu] per layer, `layers` times.
    fn layered_mlp(layers: usize, dim: usize) -> Graph {
        let mut g = Graph::new();
        let mut h = g.input("x", &[4, dim]);
        for l in 0..layers {
            let w = g.parameter(&format!("l{l}.w"), &[dim, dim]);
            let b = g.parameter(&format!("l{l}.b"), &[dim]);
            let mm = g.matmul(h, w);
            let ba = g.bias_add(mm, b);
            h = g.relu(ba);
        }
        g.set_outputs(vec![h]);
        g
    }

    #[test]
    fn detects_repeated_layers() {
        let g = layered_mlp(12, 8);
        let regions = detect_repeated_regions(&g);
        assert_eq!(regions.len(), 1, "regions: {regions:?}");
        let r = regions[0];
        // 5 nodes per layer: w, b, matmul, bias_add, relu.
        assert_eq!(r.period, 5);
        assert_eq!(r.count, 12);
        assert_eq!(r.start, 1); // node 0 is the input
    }

    #[test]
    fn rejects_non_repeating_graph() {
        let mut g = Graph::new();
        let x = g.input("x", &[4, 8]);
        let w = g.parameter("w", &[8, 8]);
        let mm = g.matmul(x, w);
        let r = g.relu(mm);
        g.set_outputs(vec![r]);
        assert!(detect_repeated_regions(&g).is_empty());
    }

    #[test]
    fn shape_change_breaks_region() {
        // Two runs of layers with different dims: the detector must not
        // merge them into one region (signatures differ via ty).
        let mut g = Graph::new();
        let mut h = g.input("x", &[4, 8]);
        for l in 0..12 {
            let w = g.parameter(&format!("a{l}.w"), &[8, 8]);
            let mm = g.matmul(h, w);
            h = g.relu(mm);
        }
        let wp = g.parameter("proj", &[8, 16]);
        let mut h2 = g.matmul(h, wp);
        for l in 0..12 {
            let w = g.parameter(&format!("b{l}.w"), &[16, 16]);
            let mm = g.matmul(h2, w);
            h2 = g.relu(mm);
        }
        g.set_outputs(vec![h2]);
        let regions = detect_repeated_regions(&g);
        assert_eq!(regions.len(), 2, "regions: {regions:?}");
        for r in &regions {
            assert_eq!(r.period, 3);
            assert!(r.count >= 11, "count: {}", r.count);
        }
    }

    #[test]
    fn shared_global_edges_allowed() {
        // All layers read the same global scale tensor (shared edge) in
        // addition to the chained hidden state.
        let mut g = Graph::new();
        let mut h = g.input("x", &[4, 8]);
        let scale = g.parameter("scale", &[4, 8]);
        for l in 0..10 {
            let w = g.parameter(&format!("l{l}.w"), &[8, 8]);
            let mm = g.matmul(h, w);
            let sc = g.mul(mm, scale);
            h = g.relu(sc);
        }
        g.set_outputs(vec![h]);
        let regions = detect_repeated_regions(&g);
        assert_eq!(regions.len(), 1, "regions: {regions:?}");
        let r = regions[0];
        assert_eq!(r.period, 4);
        // The head instance reads the graph input at a non-lattice id and
        // is dropped by consecutive-pair verification; 9 of 10 remain.
        assert_eq!(r.count, 9);
        assert_eq!(r.start, 6);
    }

    #[test]
    fn verify_rejects_cross_wired_instances() {
        // Properly chained instances: mm1 reads r0, the exact one-period
        // shift of mm0's input edge.
        let mut g = Graph::new();
        let x = g.input("x", &[4, 8]);
        let w0 = g.parameter("w0", &[8, 8]);
        let mm0 = g.matmul(x, w0);
        let r0 = g.relu(mm0);
        let w1 = g.parameter("w1", &[8, 8]);
        let mm1 = g.matmul(r0, w1);
        let r1 = g.relu(mm1);
        g.set_outputs(vec![r1]);
        assert!(pair_isomorphic(&g, 1, 3, 0));

        // Cross-wired: mm1 reads mm0 instead of r0 — the edge neither
        // shifts by one period nor matches as a shared global.
        let mut g2 = Graph::new();
        let x2 = g2.input("x", &[4, 8]);
        let w0 = g2.parameter("w0", &[8, 8]);
        let mm0 = g2.matmul(x2, w0);
        let _r0 = g2.relu(mm0);
        let w1 = g2.parameter("w1", &[8, 8]);
        let mm1 = g2.matmul(mm0, w1);
        let r1 = g2.relu(mm1);
        g2.set_outputs(vec![r1]);
        assert!(!pair_isomorphic(&g2, 1, 3, 0));
    }
}
