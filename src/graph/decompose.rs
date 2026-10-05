//! Composite decompositions, and recognizing them again.
//!
//! Every composite op ([`OpClass::Composite`]) has a one-level expansion
//! ([`Graph::expand`]) into primitives and other composites.
//! [`Graph::decompose`] applies expansions until only primitives remain.
//! [`Graph::recompose`] runs the other way: at every node, the composite's
//! guesser proposes inputs and attributes, and the proposal holds only if
//! the graph around the node is structurally the composite's own
//! expansion. Recognition is therefore exact, and `recompose(decompose(g))`
//! restores every composite of `g`, so both spellings build the same plan.
//!
//! The expansions are written to be maximally composed: where a composite
//! contains another (a loss containing a log-softmax), it expands to that
//! composite rather than to its primitives. Recomposition runs bottom-up,
//! so the inner composite is recognized first and the outer one then
//! matches its one-level expansion.

use std::collections::HashMap;

use super::{Graph, Node, NodeId, Op, OpClass, TensorType};

/// `sqrt(2/π)` in the GELU kernel's tanh form.
const GELU_SQRT_2_OVER_PI: f32 = 0.797_884_6;
/// The cubic coefficient of GELU's tanh form.
const GELU_CUBIC: f32 = 0.044715;
/// The probability clamp of the BCE loss.
const BCE_EPS: f32 = 1e-7;

impl Graph {
    /// A copy of the graph with every composite replaced by primitives.
    pub fn decompose(&self) -> Graph {
        self.decompose_where(|_| true)
    }

    /// A copy with the composites `select` picks replaced by primitives,
    /// including any composites their expansions contain.
    fn decompose_where(&self, select: impl Fn(&Op) -> bool) -> Graph {
        let mut graph = self.deep_clone();
        while let Some(next) = graph.expand_composites(&select) {
            graph = next;
        }
        graph
    }

    /// One pass replacing each selected composite by its one-level
    /// expansion, or `None` when there is none.
    fn expand_composites(&self, select: &impl Fn(&Op) -> bool) -> Option<Graph> {
        let chosen = |op: &Op| op.class() == OpClass::Composite && select(op);
        if !self.nodes.iter().any(|node| chosen(&node.op)) {
            return None;
        }
        let mut out = Graph {
            nodes: Vec::with_capacity(self.nodes.len()),
            outputs: Vec::new(),
            new_nodes_require_full_precision: false,
            num_param_grad_outputs: 0,
            derived_params: self.derived_params.clone(),
        };
        let mut map: Vec<NodeId> = Vec::with_capacity(self.nodes.len());
        for node in &self.nodes {
            let inputs: Vec<NodeId> = node.inputs.iter().map(|&i| map[i as usize]).collect();
            out.new_nodes_require_full_precision = node.requires_full_precision;
            let id = if chosen(&node.op) {
                let id = out.expand(&node.op, &inputs, &node.ty);
                if node.name.is_some() {
                    out.nodes[id as usize].name.clone_from(&node.name);
                }
                id
            } else {
                let id = out.add_raw_node_with_precision(
                    node.op.clone(),
                    inputs,
                    node.ty.clone(),
                    node.requires_full_precision,
                );
                let new = &mut out.nodes[id as usize];
                new.name.clone_from(&node.name);
                new.matmul_impl = node.matmul_impl;
                id
            };
            map.push(id);
        }
        out.new_nodes_require_full_precision = self.new_nodes_require_full_precision;
        out.outputs = self.outputs.iter().map(|&o| map[o as usize]).collect();
        out.num_param_grad_outputs = self.num_param_grad_outputs;
        Some(out)
    }

    /// Append the one-level expansion of composite `op` applied to
    /// `inputs`, producing a tensor of type `ty`, and return its root.
    #[track_caller]
    pub fn expand(&mut self, op: &Op, inputs: &[NodeId], ty: &TensorType) -> NodeId {
        let arg = |i: usize| inputs[i];
        let root = match *op {
            Op::Softmax => {
                let x = self.rows(arg(0));
                self.decomposed_softmax(x)
            }
            Op::LogSoftmax => {
                let x = self.rows(arg(0));
                let inner = self.node(x).ty.shape[1];
                let max = self.max_inner(x);
                let max = self.broadcast_inner(max, inner);
                let shifted = self.sub(x, max);
                let exp = self.exp(shifted);
                let sum = self.sum_inner(exp);
                let lse = self.log(sum);
                let lse = self.broadcast_inner(lse, inner);
                self.sub(shifted, lse)
            }
            Op::RmsNorm { eps } => {
                let x = self.rows(arg(0));
                self.decomposed_rms_norm(x, arg(1), eps)
            }
            Op::LayerNorm { eps } => {
                // An RMS norm of the centered rows, plus the bias.
                let x = self.rows(arg(0));
                let inner = self.node(x).ty.shape[1];
                let mean = self.mean_inner(x);
                let mean = self.broadcast_inner(mean, inner);
                let centered = self.sub(x, mean);
                let normalized = self.rms_norm(centered, arg(1), eps);
                self.bias_add(normalized, arg(2))
            }
            Op::Silu => {
                let s = self.sigmoid(arg(0));
                self.mul(arg(0), s)
            }
            Op::Gelu => {
                // x · sigmoid(2·√(2/π)·(x + 0.044715·x³)), as the kernel.
                let x = arg(0);
                let square = self.mul(x, x);
                let cube = self.mul(square, x);
                let cube = self.scale(cube, GELU_CUBIC);
                let inner = self.add(x, cube);
                let inner = self.scale(inner, 2.0 * GELU_SQRT_2_OVER_PI);
                let s = self.sigmoid(inner);
                self.mul(x, s)
            }
            Op::Softplus { beta } => {
                // relu(x) + ln(1 + exp(-|βx|)) / β
                let x = arg(0);
                let scaled = self.scale(x, beta);
                let magnitude = self.abs(scaled);
                let tail = self.neg(magnitude);
                let tail = self.exp(tail);
                let tail = self.add_scalar(tail, 1.0);
                let tail = self.log(tail);
                let tail = self.scale(tail, 1.0 / beta);
                let head = self.relu(x);
                self.add(head, tail)
            }
            Op::SwiGLU => {
                let gate = self.silu(arg(0));
                self.mul(gate, arg(1))
            }
            Op::GeGLU => {
                let gate = self.gelu(arg(0));
                self.mul(gate, arg(1))
            }
            Op::MeanAll => {
                let n = self.node(arg(0)).ty.num_elements();
                let sum = self.sum_all(arg(0));
                self.scale(sum, super::mean_factor(n))
            }
            Op::SumRows => {
                let cols = ty.num_elements();
                let rows = self.node(arg(0)).ty.num_elements() / cols.max(1);
                let x = self.view(arg(0), &[rows, cols]);
                let x = self.transpose(x);
                self.sum_inner(x)
            }
            Op::GlobalAvgPool { spatial, .. } => {
                let planes = ty.num_elements();
                let x = self.view(arg(0), &[planes, spatial as usize]);
                self.mean_inner(x)
            }
            Op::NormalizeInnerSum { floor, .. } => {
                let x = arg(0);
                let inner = self.node(x).ty.shape[1];
                let sum = self.sum_inner(x);
                let excess = self.add_scalar(sum, -floor);
                let excess = self.relu(excess);
                let denominator = self.add_scalar(excess, floor);
                let inv = self.recip(denominator);
                let inv = self.broadcast_inner(inv, inner);
                self.mul(x, inv)
            }
            Op::PairwiseSquaredDistance { pairs } => {
                let (left, right) = (arg(0), arg(1));
                let rows = self.node(left).ty.shape[0];
                let left = self.repeat_rows(left, pairs as usize);
                let diff = self.sub(left, right);
                let square = self.mul(diff, diff);
                let sum = self.sum_inner(square);
                self.view(sum, &[rows, pairs as usize])
            }
            Op::PairwiseVectorRejection { pairs } => {
                let (vectors, directions) = (arg(0), arg(1));
                let width = self.node(vectors).ty.shape[1];
                let directions = self.repeat_rows(directions, pairs as usize);
                let product = self.mul(vectors, directions);
                let along = self.sum_inner(product);
                let along = self.broadcast_inner(along, width);
                let projection = self.mul(along, directions);
                self.sub(vectors, projection)
            }
            Op::ExclusiveCumsum { reverse } => {
                // y = x·U, U[k, n] = 1 where k sums into n.
                let n = self.node(arg(0)).ty.shape[1];
                let mut upper = vec![0.0; n * n];
                for k in 0..n {
                    for j in 0..n {
                        if (k < j && !reverse) || (k > j && reverse) {
                            upper[k * n + j] = 1.0;
                        }
                    }
                }
                let upper = self.constant(upper, &[n, n]);
                self.matmul(arg(0), upper)
            }
            Op::ShiftInner { offset } => self.expand_shift(arg(0), offset),
            Op::CrossEntropyLoss => {
                let (logits, labels) = (arg(0), arg(1));
                let rows = self.node(logits).ty.shape[0];
                let log_p = self.log_softmax(logits);
                let terms = self.mul(labels, log_p);
                let sum = self.sum_all(terms);
                self.scale(sum, -1.0 / rows as f32)
            }
            Op::BceLoss => {
                let (pred, labels) = (arg(0), arg(1));
                let n = self.node(pred).ty.num_elements();
                let p = self.clamp(pred, BCE_EPS, 1.0 - BCE_EPS);
                let log_p = self.log(p);
                let hit = self.mul(labels, log_p);
                let not_label = self.neg(labels);
                let not_label = self.add_scalar(not_label, 1.0);
                let not_p = self.neg(p);
                let not_p = self.add_scalar(not_p, 1.0);
                let log_not_p = self.log(not_p);
                let miss = self.mul(not_label, log_not_p);
                let terms = self.add(hit, miss);
                let sum = self.sum_all(terms);
                self.scale(sum, -1.0 / n as f32)
            }
            Op::MulPerChannel { spatial, .. } => {
                let (src, gate) = (arg(0), arg(1));
                let planes = self.node(gate).ty.num_elements();
                let x = self.view(src, &[planes, spatial as usize]);
                let gate = self.view(gate, &[planes, 1]);
                let gate = self.broadcast_inner(gate, spatial as usize);
                self.mul(x, gate)
            }
            Op::AddPerChannel { channels, spatial } => {
                let (src, bias) = (arg(0), arg(1));
                let plane = channels as usize * spatial as usize;
                let batch = self.node(src).ty.num_elements() / plane;
                let x = self.view(src, &[batch, plane]);
                let bias = self.per_channel(bias, channels, spatial);
                self.bias_add(x, bias)
            }
            Op::GroupNorm {
                num_groups,
                eps,
                channels,
                spatial,
            } => {
                let (x, weight, bias) = (arg(0), arg(1), arg(2));
                let total = self.node(x).ty.num_elements();
                let group_len = channels as usize / num_groups as usize * spatial as usize;
                let groups = self.view(x, &[total / group_len, group_len]);
                let mean = self.mean_inner(groups);
                let mean = self.broadcast_inner(mean, group_len);
                let centered = self.sub(groups, mean);
                let square = self.mul(centered, centered);
                let variance = self.mean_inner(square);
                let variance = self.add_scalar(variance, eps);
                let inv = self.rsqrt(variance);
                let inv = self.broadcast_inner(inv, group_len);
                let normalized = self.mul(centered, inv);
                let plane = channels as usize * spatial as usize;
                let normalized = self.view(normalized, &[total / plane, plane]);
                let weight = self.per_channel(weight, channels, spatial);
                let scaled = self.bias_mul(normalized, weight);
                let scaled = self.view(scaled, &[total]);
                self.add_per_channel(scaled, bias, channels, spatial)
            }
            Op::Upsample2x { in_w, .. } => {
                // Each pixel twice along the row, then each row twice.
                let x = arg(0);
                let w = in_w as usize;
                let total = self.node(x).ty.num_elements();
                let rows = total / w;
                let pixels = self.view(x, &[total, 1]);
                let wide = self.broadcast_inner(pixels, 2);
                let wide = self.view(wide, &[total * 2]);
                let rows_u32 = u32::try_from(rows).expect("upsample rows exceed u32");
                let width = u32::try_from(2 * w).expect("upsample width exceeds u32");
                self.concat(wide, wide, rows_u32, 1, 1, width)
            }
            ref other => self.expand_sequence_op(other, inputs, ty),
        };
        self.view(root, &ty.shape)
    }

    /// Placeholder until the remaining composites have expansions.
    #[track_caller]
    fn expand_sequence_op(&mut self, op: &Op, _inputs: &[NodeId], _ty: &TensorType) -> NodeId {
        panic!("{op:?} has no expansion")
    }

    /// `x` with `shape`, through a reshape only when it differs.
    pub(crate) fn view(&mut self, x: NodeId, shape: &[usize]) -> NodeId {
        if self.node(x).ty.shape == shape {
            x
        } else {
            self.reshape(x, shape)
        }
    }

    /// `x` as rows along its last axis.
    fn rows(&mut self, x: NodeId) -> NodeId {
        let shape = &self.node(x).ty.shape;
        let cols = shape.last().copied().unwrap_or(1).max(1);
        let rows = shape.iter().product::<usize>() / cols;
        self.view(x, &[rows, cols])
    }

    /// Each row of `x: [M, D]` repeated `times` times: `[M·times, D]`.
    fn repeat_rows(&mut self, x: NodeId, times: usize) -> NodeId {
        let shape = self.node(x).ty.shape.clone();
        let (rows, width) = (shape[0], shape[1]);
        let flat = self.repeat_axis(x, rows, width, times);
        self.view(flat, &[rows * times, width])
    }

    /// `times` copies of every `[inner]` block of `x`, grouped per `outer`
    /// index, by doubling concatenations: `[outer · times · inner]`.
    pub(crate) fn repeat_axis(
        &mut self,
        x: NodeId,
        outer: usize,
        inner: usize,
        times: usize,
    ) -> NodeId {
        let outer = u32::try_from(outer).expect("repeat outer exceeds u32");
        let inner = u32::try_from(inner).expect("repeat inner exceeds u32");
        let (mut acc, mut acc_n) = (None, 0u32);
        let (mut piece, mut piece_n) = (x, 1u32);
        let mut remaining = times;
        while remaining > 0 {
            if remaining & 1 == 1 {
                acc = Some(match acc {
                    None => piece,
                    Some(a) => self.concat(a, piece, outer, acc_n, piece_n, inner),
                });
                acc_n += piece_n;
            }
            remaining >>= 1;
            if remaining > 0 {
                piece = self.concat(piece, piece, outer, piece_n, piece_n, inner);
                piece_n *= 2;
            }
        }
        acc.expect("repeat at least once")
    }

    /// A per-channel vector `[C]` spread over its planes: `[C·spatial]`.
    fn per_channel(&mut self, v: NodeId, channels: u32, spatial: u32) -> NodeId {
        let column = self.view(v, &[channels as usize, 1]);
        let planes = self.broadcast_inner(column, spatial as usize);
        self.view(planes, &[channels as usize * spatial as usize])
    }

    fn expand_shift(&mut self, x: NodeId, offset: i32) -> NodeId {
        let shape = self.node(x).ty.shape.clone();
        let (rows, cols) = (shape[0], shape[1]);
        let n = cols as i64;
        let o = i64::from(offset);
        if o == 0 {
            return self.materialize(x);
        }
        if o.abs() >= n {
            // Nothing survives: `x > x` is zero for every input, NaN too.
            return self.greater(x, x);
        }
        let (rows_u32, keep, gap) = (rows as u32, (n - o.abs()) as u32, o.unsigned_abs() as u32);
        let zeros = self.constant(vec![0.0; rows * gap as usize], &[rows * gap as usize]);
        if o > 0 {
            let kept = self.split_a(x, rows_u32, keep, gap, 1);
            self.concat(zeros, kept, rows_u32, gap, keep, 1)
        } else {
            let kept = self.split_b(x, rows_u32, gap, keep, 1);
            self.concat(kept, zeros, rows_u32, keep, gap, 1)
        }
    }

    /// The graph with every recognizable expansion replaced by its
    /// composite. Nodes keep their ids; superseded interior nodes are left
    /// for dead-code elimination.
    ///
    /// Composites that occur inside other composites' expansions are
    /// expanded first, so a graph that spells a larger composite partly
    /// with them recomposes as the fully decomposed graph does. Recognition
    /// then runs from the outputs back: the outermost composite claims its
    /// nodes before a smaller one could match part of them.
    pub fn recompose(&self) -> Graph {
        let mut graph = self.deep_clone();
        graph.canonicalize_attributes();
        let mut graph = graph.decompose_where(is_nested);
        let mut templates = HashMap::new();
        for id in (0..graph.nodes.len() as NodeId).rev() {
            if matches!(graph.node(id).op, Op::Nop) {
                continue;
            }
            for (op, inputs) in guesses(&graph, id) {
                if graph.expands_to(id, &op, &inputs, &mut templates) {
                    let node = &mut graph.nodes[id as usize];
                    node.op = op;
                    node.inputs = inputs;
                    node.matmul_impl = None;
                    break;
                }
            }
        }
        graph
    }

    /// Give composites the one spelling their expansion determines, where
    /// an attribute does not change the result: a gate's channel count, an
    /// upsample's split of planes into channels and rows, a shift that
    /// keeps or clears every element.
    fn canonicalize_attributes(&mut self) {
        for id in 0..self.nodes.len() {
            let inputs = self.nodes[id].inputs.clone();
            let canonical = match self.nodes[id].op {
                Op::MulPerChannel { spatial, .. } => Some(Op::MulPerChannel {
                    channels: self.node(inputs[1]).ty.num_elements() as u32,
                    spatial,
                }),
                Op::Upsample2x { in_w, .. } => Some(Op::Upsample2x {
                    channels: 1,
                    in_h: (self.node(inputs[0]).ty.num_elements() / in_w as usize) as u32,
                    in_w,
                }),
                Op::ShiftInner { offset } => {
                    let cols = self.node(inputs[0]).ty.shape[1] as i64;
                    if offset == 0 {
                        Some(Op::Materialize)
                    } else if i64::from(offset).abs() >= cols {
                        Some(Op::ShiftInner {
                            offset: cols as i32,
                        })
                    } else {
                        None
                    }
                }
                _ => None,
            };
            if let Some(op) = canonical {
                self.nodes[id].op = op;
            }
        }
    }

    /// Whether the graph at `root` is exactly the full decomposition of
    /// `op` over `inputs`.
    fn expands_to(
        &self,
        root: NodeId,
        op: &Op,
        inputs: &[NodeId],
        templates: &mut HashMap<String, Option<(Graph, NodeId)>>,
    ) -> bool {
        if op.class() != OpClass::Composite || inputs.iter().any(|&i| i >= root) {
            return false;
        }
        let types: Vec<&TensorType> = inputs.iter().map(|&i| &self.node(i).ty).collect();
        let ty = &self.node(root).ty;
        if !accepts(op, &types, ty) {
            return false;
        }
        let key = format!("{op:?} {types:?} {ty:?}");
        let template = templates
            .entry(key)
            .or_insert_with(|| template(op, &types, ty));
        let Some(&(ref template, top)) = template.as_ref() else {
            return false;
        };
        Matcher {
            template,
            graph: self,
            inputs,
            memo: HashMap::new(),
        }
        .same(top, root)
    }
}

/// Composites that appear inside other composites' expansions.
fn is_nested(op: &Op) -> bool {
    matches!(
        *op,
        Op::Silu
            | Op::Gelu
            | Op::Softmax
            | Op::LogSoftmax
            | Op::RmsNorm { .. }
            | Op::AddPerChannel { .. }
    )
}

/// The full decomposition of `op` over placeholder inputs of `types`, and
/// its root. Placeholders are the first nodes.
fn template(op: &Op, types: &[&TensorType], ty: &TensorType) -> Option<(Graph, NodeId)> {
    let mut graph = Graph::new();
    let placeholders: Vec<NodeId> = types
        .iter()
        .enumerate()
        .map(|(k, t)| {
            graph.add_raw_node(
                Op::Input {
                    name: format!("#{k}"),
                },
                vec![],
                (*t).clone(),
            )
        })
        .collect();
    let top = graph.expand(op, &placeholders, ty);
    graph.set_outputs(vec![top]);
    let full = graph.decompose();
    let top = full.outputs()[0];
    Some((full, top))
}

/// Whether `op` is well-typed over `ins` producing `out`: the contract its
/// builder enforces, which its expansion relies on.
fn accepts(op: &Op, ins: &[&TensorType], out: &TensorType) -> bool {
    use super::DType::F32;
    if ins.iter().any(|t| t.dtype != F32) || out.dtype != F32 {
        return false;
    }
    let elems = |t: &TensorType| t.num_elements();
    let matrix = |t: &TensorType| match t.shape[..] {
        [rows, cols] if rows > 0 && cols > 0 => Some((rows, cols)),
        _ => None,
    };
    let arity = |n: usize| ins.len() == n;
    match *op {
        Op::Silu | Op::Gelu | Op::Softplus { .. } | Op::Softmax | Op::LogSoftmax => {
            arity(1) && ins[0] == out && elems(out) > 0
        }
        Op::SwiGLU | Op::GeGLU => arity(2) && ins[0] == out && ins[1] == out,
        Op::RmsNorm { .. } => {
            arity(2) && ins[0] == out && matrix(out).is_some_and(|(_, n)| ins[1].shape == [n])
        }
        Op::LayerNorm { .. } => {
            arity(3)
                && ins[0] == out
                && matrix(out).is_some_and(|(_, n)| ins[1].shape == [n] && ins[2].shape == [n])
        }
        Op::MeanAll => arity(1) && out.shape == [1],
        Op::SumRows => {
            arity(1) && out.rank() == 1 && elems(out) > 0 && elems(ins[0]) % elems(out) == 0
        }
        Op::GlobalAvgPool { channels, spatial } => {
            arity(1)
                && ins[0].rank() == 1
                && matches!(out.shape[..], [b, c] if c == channels as usize
                    && b * c * spatial as usize == elems(ins[0]))
        }
        Op::NormalizeInnerSum { inner, .. } => {
            arity(1) && ins[0] == out && matrix(out).is_some_and(|(_, n)| n == inner as usize)
        }
        Op::PairwiseSquaredDistance { pairs } => {
            let p = pairs as usize;
            arity(2)
                && matrix(ins[0])
                    .is_some_and(|(m, d)| ins[1].shape == [m * p, d] && out.shape == [m, p])
        }
        Op::PairwiseVectorRejection { pairs } => {
            let p = pairs as usize;
            arity(2)
                && ins[0] == out
                && matrix(ins[1]).is_some_and(|(m, d)| out.shape == [m * p, d])
        }
        Op::ExclusiveCumsum { .. } | Op::ShiftInner { .. } => {
            arity(1) && ins[0] == out && matrix(out).is_some()
        }
        Op::CrossEntropyLoss => {
            arity(2) && matrix(ins[0]).is_some() && ins[1] == ins[0] && out.shape == [1]
        }
        Op::BceLoss => arity(2) && ins[1] == ins[0] && out.shape == [1],
        Op::MulPerChannel { spatial, .. } => {
            arity(2)
                && ins[0] == out
                && out.rank() == 1
                && ins[1].rank() == 1
                && elems(ins[1]) * spatial as usize == elems(out)
        }
        Op::AddPerChannel { channels, spatial } => {
            let plane = channels as usize * spatial as usize;
            arity(2)
                && ins[0] == out
                && out.rank() == 1
                && ins[1].shape == [channels as usize]
                && plane > 0
                && elems(out) % plane == 0
        }
        Op::GroupNorm {
            num_groups,
            channels,
            spatial,
            ..
        } => {
            let plane = channels as usize * spatial as usize;
            arity(3)
                && ins[0] == out
                && out.rank() == 1
                && ins[1].shape == [channels as usize]
                && ins[2].shape == [channels as usize]
                && num_groups > 0
                && channels % num_groups == 0
                && plane > 0
                && elems(out) % plane == 0
        }
        Op::Upsample2x { in_w, .. } => {
            arity(1)
                && ins[0].rank() == 1
                && out.shape == [4 * elems(ins[0])]
                && in_w > 0
                && elems(ins[0]) % in_w as usize == 0
        }
        _ => true,
    }
}

/// Structural equality between a template expansion, whose first nodes are
/// placeholders for `inputs`, and the graph.
struct Matcher<'a> {
    template: &'a Graph,
    graph: &'a Graph,
    inputs: &'a [NodeId],
    memo: HashMap<(NodeId, NodeId), bool>,
}

impl Matcher<'_> {
    fn same(&mut self, t: NodeId, g: NodeId) -> bool {
        if (t as usize) < self.inputs.len() {
            return self.inputs[t as usize] == g;
        }
        if let Some(&known) = self.memo.get(&(t, g)) {
            return known;
        }
        let (tn, gn) = (self.template.node(t), self.graph.node(g));
        let result = tn.ty == gn.ty
            && tn.inputs.len() == gn.inputs.len()
            && same_op(&tn.op, &gn.op)
            && self.same_inputs(tn, gn);
        self.memo.insert((t, g), result);
        result
    }

    fn same_inputs(&mut self, tn: &Node, gn: &Node) -> bool {
        let (ti, gi) = (tn.inputs.clone(), gn.inputs.clone());
        if matches!(tn.op, Op::Add | Op::Mul) && ti.len() == 2 {
            return (self.same(ti[0], gi[0]) && self.same(ti[1], gi[1]))
                || (self.same(ti[0], gi[1]) && self.same(ti[1], gi[0]));
        }
        ti.iter().zip(&gi).all(|(&t, &g)| self.same(t, g))
    }
}

fn same_op(a: &Op, b: &Op) -> bool {
    fn data(op: &Op) -> Option<&[f32]> {
        match *op {
            Op::Constant { ref data } => Some(data),
            _ => None,
        }
    }
    match (data(a), data(b)) {
        (Some(x), Some(y)) => {
            x.len() == y.len() && x.iter().zip(y).all(|(p, q)| p.to_bits() == q.to_bits())
        }
        (None, None) => format!("{a:?}") == format!("{b:?}"),
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Guessers: from a node, the composites it might be the root of. Each only
// has to find the inputs and attributes; `expands_to` checks the rest.
// ---------------------------------------------------------------------------

fn input(g: &Graph, id: NodeId, slot: usize) -> Option<NodeId> {
    g.node(id).inputs.get(slot).copied()
}

/// Skip reshapes, which expansions add to restore the composite's shape.
fn unview(g: &Graph, mut id: NodeId) -> NodeId {
    while matches!(g.node(id).op, Op::Identity) {
        id = g.node(id).inputs[0];
    }
    id
}

/// Both operand orders of a binary node.
fn operands(g: &Graph, id: NodeId) -> Vec<(NodeId, NodeId)> {
    match g.node(id).inputs[..] {
        [a, b] if a == b => vec![(a, b)],
        [a, b] => vec![(a, b), (b, a)],
        _ => vec![],
    }
}

fn is(g: &Graph, id: NodeId, f: impl Fn(&Op) -> bool) -> bool {
    f(&g.node(id).op)
}

fn guesses(g: &Graph, root: NodeId) -> Vec<(Op, Vec<NodeId>)> {
    let mut out = Vec::new();
    let node = g.node(root);
    let core = unview(g, root);
    let core_op = &g.node(core).op;
    match *core_op {
        Op::Mul => {
            for (a, b) in operands(g, core) {
                // softmax: exp(x - max) · bcast(1/sum)
                if is(g, a, |op| matches!(op, Op::Exp))
                    && let Some(shifted) = input(g, a, 0)
                    && let Some(x) = input(g, shifted, 0)
                {
                    out.push((Op::Softmax, vec![unview(g, x), x]));
                }
                // silu: x · sigmoid(x)
                if is(g, b, |op| matches!(op, Op::Sigmoid)) && input(g, b, 0) == Some(a) {
                    out.push((Op::Silu, vec![a]));
                }
                // gelu: x · sigmoid(...)
                if is(g, b, |op| matches!(op, Op::Sigmoid)) {
                    out.push((Op::Gelu, vec![a]));
                }
                // glu: (gate · sigmoid(·)) · up
                if is(g, a, |op| matches!(op, Op::Mul)) {
                    for (gate, s) in operands(g, a) {
                        if is(g, s, |op| matches!(op, Op::Sigmoid)) {
                            out.push((Op::SwiGLU, vec![gate, b]));
                            out.push((Op::GeGLU, vec![gate, b]));
                        }
                    }
                }
                // per-channel gate over flat planes: view(src) · bcast(view(gate))
                if let Some(gate) = input(g, b, 0)
                    && is(g, b, |op| matches!(op, Op::BroadcastInner { .. }))
                {
                    let (src, gate) = (unview(g, a), unview(g, gate));
                    if g.node(src).ty.rank() == 1 && g.node(gate).ty.rank() == 1 {
                        let (channels, spatial) = per_channel_dims(g, node, gate);
                        out.push((Op::MulPerChannel { channels, spatial }, vec![src, gate]));
                    }
                }
            }
        }
        Op::BiasMul => {
            // rms_norm: (x · bcast(rsqrt(mean(x²) + eps))) ⊙ w
            if let (Some(normalized), Some(w)) = (input(g, core, 0), input(g, core, 1)) {
                for (x, inv) in operands(g, normalized) {
                    if let Some(eps) = norm_eps(g, inv) {
                        out.push((Op::RmsNorm { eps }, vec![unview(g, x), w]));
                        out.push((Op::RmsNorm { eps }, vec![x, w]));
                    }
                }
            }
        }
        Op::BiasAdd => {
            // layer_norm: ((x - mean) · inv ⊙ w) + b
            if let (Some(scaled), Some(b)) = (input(g, core, 0), input(g, core, 1))
                && is(g, scaled, |op| matches!(op, Op::BiasMul))
                && let (Some(normalized), Some(w)) = (input(g, scaled, 0), input(g, scaled, 1))
            {
                for (centered, inv) in operands(g, normalized) {
                    if let Some(eps) = norm_eps(g, inv)
                        && let Some(x) = input(g, centered, 0)
                    {
                        out.push((Op::LayerNorm { eps }, vec![unview(g, x), w, b]));
                        out.push((Op::LayerNorm { eps }, vec![x, w, b]));
                    }
                }
            }
            // group_norm: view(groups normalized ⊙ spread(w)) + spread(b)
            if let (Some(scaled), Some(spread_b)) = (input(g, core, 0), input(g, core, 1)) {
                let scaled = unview(g, scaled);
                if is(g, scaled, |op| matches!(op, Op::BiasMul))
                    && let (Some(normalized), Some(spread_w)) =
                        (input(g, scaled, 0), input(g, scaled, 1))
                {
                    let normalized = unview(g, normalized);
                    let spread = |id: NodeId| input(g, unview(g, id), 0).map(|c| unview(g, c));
                    if let (Some(w), Some(bias)) = (spread(spread_w), spread(spread_b)) {
                        for (centered, inv) in operands(g, normalized) {
                            if let Some(eps) = norm_eps(g, inv)
                                && let Some(groups) = input(g, centered, 0)
                            {
                                let channels = g.node(w).ty.num_elements() as u32;
                                let group_len =
                                    g.node(groups).ty.shape.get(1).copied().unwrap_or(0);
                                let plane = g.node(scaled).ty.shape.get(1).copied().unwrap_or(0);
                                if channels > 0 && group_len > 0 && plane % channels as usize == 0 {
                                    let spatial = (plane / channels as usize) as u32;
                                    out.push((
                                        Op::GroupNorm {
                                            num_groups: (plane / group_len) as u32,
                                            eps,
                                            channels,
                                            spatial,
                                        },
                                        vec![unview(g, groups), w, bias],
                                    ));
                                }
                            }
                        }
                    }
                }
            }
            // add_per_channel: view(src) + spread(bias)
            if let (Some(src), Some(bias)) = (input(g, core, 0), input(g, core, 1)) {
                let bias = unview(g, unview(g, bias));
                let bias = input(g, bias, 0).map_or(bias, |b| unview(g, b));
                let channels = g.node(bias).ty.num_elements() as u32;
                let plane = g.node(src).ty.shape.get(1).copied().unwrap_or(0) as u32;
                if channels > 0
                    && plane.is_multiple_of(channels)
                    && g.node(unview(g, src)).ty.rank() == 1
                {
                    out.push((
                        Op::AddPerChannel {
                            channels,
                            spatial: plane / channels,
                        },
                        vec![unview(g, src), bias],
                    ));
                }
            }
        }
        Op::Add => {
            // log_softmax: (x - max) - bcast(log(sum(exp(x - max))))
            for (shifted, _) in operands(g, core) {
                if let Some(x) = input(g, shifted, 0) {
                    out.push((Op::LogSoftmax, vec![unview(g, x)]));
                }
                // softplus: relu(x) + scale(log(1 + exp(-|βx|)), 1/β)
                if is(g, shifted, |op| matches!(op, Op::Relu))
                    && let Some(x) = input(g, shifted, 0)
                {
                    for (_, tail) in operands(g, core) {
                        if let Op::Scale { factor } = g.node(tail).op {
                            out.push((Op::Softplus { beta: 1.0 / factor }, vec![x]));
                        }
                    }
                }
                // pairwise rejection: v - bcast(Σ v·u)·u
                for (v, projection) in operands(g, core) {
                    if is(g, projection, |op| matches!(op, Op::Neg))
                        && let Some(p) = input(g, projection, 0)
                    {
                        for (_, u) in operands(g, p) {
                            if let Some((directions, pairs)) = repeated_rows(g, u) {
                                out.push((
                                    Op::PairwiseVectorRejection { pairs },
                                    vec![v, directions],
                                ));
                            }
                        }
                    }
                }
            }
        }
        Op::Scale { factor } => {
            if let Some(sum) = input(g, core, 0) {
                if is(g, sum, |op| matches!(op, Op::SumAll))
                    && let Some(terms) = input(g, sum, 0)
                {
                    // mean_all: sum · 1/n
                    out.push((Op::MeanAll, vec![terms]));
                    // cross entropy: -Σ labels · log_softmax(logits) / B, the
                    // log-softmax being (logits - max) - lse
                    for (labels, log_p) in operands(g, terms) {
                        if let Some(shifted) = input(g, log_p, 0)
                            && let Some(logits) = input(g, shifted, 0)
                        {
                            out.push((Op::CrossEntropyLoss, vec![unview(g, logits), labels]));
                        }
                    }
                    // bce: -mean(t·log p + (1-t)·log(1-p)), p = clamp(pred)
                    for (hit, _) in operands(g, terms) {
                        for (labels, log_p) in operands(g, hit) {
                            if let Some(p) = input(g, log_p, 0)
                                && let Some(pred) = input(g, p, 0)
                            {
                                out.push((Op::BceLoss, vec![pred, labels]));
                            }
                        }
                    }
                }
                // global average pool: mean over each plane
                if is(g, sum, |op| matches!(op, Op::SumInner))
                    && let Some(planes) = input(g, sum, 0)
                {
                    let x = unview(g, planes);
                    let spatial = g.node(planes).ty.shape.get(1).copied().unwrap_or(0);
                    let total = g.node(x).ty.num_elements();
                    if spatial > 0
                        && g.node(x).ty.rank() == 1
                        && let [batch, channels] = node.ty.shape[..]
                        && batch * channels * spatial == total
                    {
                        out.push((
                            Op::GlobalAvgPool {
                                channels: channels as u32,
                                spatial: spatial as u32,
                            },
                            vec![x],
                        ));
                    }
                }
                let _ = factor;
            }
        }
        Op::SumInner => {
            // sum_rows: sum over the transpose
            if let Some(t) = input(g, core, 0)
                && is(g, t, |op| matches!(op, Op::Transpose))
                && let Some(x) = input(g, t, 0)
            {
                out.push((Op::SumRows, vec![unview(g, x)]));
            }
            // pairwise distance: Σ (repeat(left) - right)²
            if let Some(square) = input(g, core, 0)
                && let Some(diff) = input(g, square, 0)
                && let Some(left) = input(g, diff, 0)
                && let Some(neg) = input(g, diff, 1)
                && let Some(right) = input(g, neg, 0)
                && let Some((left, pairs)) = repeated_rows(g, left)
            {
                out.push((Op::PairwiseSquaredDistance { pairs }, vec![left, right]));
            }
        }
        Op::MatMul => {
            // exclusive cumsum: x · triangular constant
            if let (Some(x), Some(u)) = (input(g, core, 0), input(g, core, 1))
                && let Op::Constant { ref data } = g.node(u).op
                && let [n, cols] = g.node(u).ty.shape[..]
                && n == cols
            {
                let reverse = n > 1 && data[n] == 1.0;
                out.push((Op::ExclusiveCumsum { reverse }, vec![x]));
            }
        }
        Op::Greater => {
            if let [a, b] = node.inputs[..]
                && a == b
            {
                let cols = g.node(a).ty.shape.get(1).copied().unwrap_or(0) as i32;
                out.push((Op::ShiftInner { offset: cols }, vec![a]));
            }
        }
        Op::Concat { .. } => {
            if let [a, b] = g.node(core).inputs[..] {
                // shift: zeros ++ head, or tail ++ zeros
                for (data, keep) in [(b, true), (a, false)] {
                    if let Some(x) = input(g, data, 0) {
                        let cols = g.node(x).ty.shape.get(1).copied().unwrap_or(0) as i32;
                        let kept = g.node(data).ty.num_elements() as i32
                            / g.node(x).ty.shape.first().copied().unwrap_or(1).max(1) as i32;
                        let gap = cols - kept;
                        let offset = if keep { gap } else { -gap };
                        out.push((Op::ShiftInner { offset }, vec![x]));
                    }
                }
                // upsample: each row of the widened image twice
                if a == b
                    && let Some(pixels) = input(g, a, 0).map(|w| unview(g, w))
                    && let Some(x) = input(g, pixels, 0)
                {
                    let x = unview(g, x);
                    if let Some(op) = upsample_dims(g, root, x) {
                        out.push((op, vec![x]));
                    }
                }
            }
        }
        _ => {}
    }
    // normalize_inner_sum: x · bcast(1/(relu(sum - floor) + floor))
    if let Op::Mul = *core_op {
        for (x, inv) in operands(g, core) {
            if let Some(r) = input(g, inv, 0)
                && let Some(d) = input(g, r, 0)
                && let Op::Offset { value: floor } = g.node(d).op
            {
                let inner = g.node(x).ty.shape.get(1).copied().unwrap_or(0) as u32;
                out.push((Op::NormalizeInnerSum { inner, floor }, vec![x]));
            }
        }
    }
    out
}

/// The epsilon of `bcast(rsqrt(mean(·) + eps))`.
fn norm_eps(g: &Graph, inv: NodeId) -> Option<f32> {
    let rsqrt = input(g, inv, 0)?;
    let offset = input(g, rsqrt, 0)?;
    match g.node(offset).op {
        Op::Offset { value } => Some(value),
        _ => None,
    }
}

/// `x: [M, D]` and the repeat count, when `id` repeats its rows.
fn repeated_rows(g: &Graph, id: NodeId) -> Option<(NodeId, u32)> {
    let repeated = &g.node(id).ty;
    let mut base = unview(g, id);
    while let Op::Concat { .. } = g.node(base).op {
        base = unview(g, g.node(base).inputs[0]);
    }
    let x = &g.node(base).ty;
    let (rows, total) = (x.shape.first().copied()?, repeated.shape.first().copied()?);
    (x.rank() == 2 && rows > 0 && total % rows == 0).then_some((base, (total / rows) as u32))
}

/// Channels and plane size of a per-channel gate `[N·C]` over `node`.
fn per_channel_dims(g: &Graph, node: &Node, gate: NodeId) -> (u32, u32) {
    let planes = g.node(gate).ty.num_elements().max(1);
    let spatial = (node.ty.num_elements() / planes) as u32;
    (planes as u32, spatial)
}

/// The upsample op producing `root` from planes `x`.
fn upsample_dims(g: &Graph, root: NodeId, x: NodeId) -> Option<Op> {
    let total = g.node(x).ty.num_elements();
    if g.node(root).ty.num_elements() != 4 * total {
        return None;
    }
    // Rows and width are only fixed together: try every factorization the
    // verification can accept, starting from square planes.
    let concat = unview(g, root);
    let Op::Concat { spatial, .. } = g.node(concat).op else {
        return None;
    };
    let w = spatial as usize / 2;
    (w > 0 && total.is_multiple_of(w)).then(|| Op::Upsample2x {
        channels: 1,
        in_h: (total / w) as u32,
        in_w: w as u32,
    })
}
