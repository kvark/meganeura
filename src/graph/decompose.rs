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
//! Composites a graph already names are left as written.
//!
//! The expansions are written to be maximally composed: where a composite
//! contains another (a loss containing a log-softmax), it expands to that
//! composite rather than to its primitives. Recognition first names those
//! nested composites, then matches the larger ones against templates that
//! name them too.
//!
//! Training builds recognize only composites whose gradient is exactly
//! their decomposition's ([`Graph::recompose_for_training`]), and builds
//! with the optimizer off recognize nothing.

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
    pub fn decompose_where(&self, select: impl Fn(&Op) -> bool) -> Graph {
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
                // Every row twice, every pixel of it twice.
                let w = in_w as usize;
                let rows = self.node(arg(0)).ty.num_elements() / w;
                let x = self.view(arg(0), &[rows, 1, w, 1]);
                self.broadcast_to(x, &[rows, 2, w, 2])
            }
            ref other => self.expand_sequence_op(other, inputs, ty),
        };
        self.view(root, &ty.shape)
    }

    /// Expansions of the rotary-embedding and attention composites.
    #[track_caller]
    fn expand_sequence_op(&mut self, op: &Op, inputs: &[NodeId], ty: &TensorType) -> NodeId {
        let arg = |i: usize| inputs[i];
        match *op {
            Op::RoPE {
                theta,
                pos_offset,
                head_dim,
                freq_factors,
            } => {
                let positions = match inputs.len() {
                    1 => Positions::Offset(pos_offset, None),
                    _ => Positions::Offset(pos_offset, Some(arg(1))),
                };
                let factors = freq_factors.then(|| arg(2));
                self.expand_rope(arg(0), theta, head_dim, positions, factors)
            }
            Op::RoPEPositions { theta, head_dim } => {
                self.expand_rope(arg(0), theta, head_dim, Positions::PerRow(arg(1)), None)
            }
            Op::CausalAttention {
                num_heads,
                num_kv_heads,
                head_dim,
            } => {
                let rows = ty.shape[0];
                let mask = self.window_mask(rows, 0);
                let heads = Heads::new(num_heads, num_kv_heads, head_dim);
                self.attend(arg(0), arg(1), arg(2), heads, Some(mask))
            }
            Op::SlidingWindowAttention {
                num_heads,
                num_kv_heads,
                head_dim,
                window_size,
            } => {
                let rows = ty.shape[0];
                let mask = self.window_mask(rows, window_size as usize);
                let heads = Heads::new(num_heads, num_kv_heads, head_dim);
                self.attend(arg(0), arg(1), arg(2), heads, Some(mask))
            }
            Op::FullAttention {
                num_heads,
                num_kv_heads,
                head_dim,
            }
            | Op::CrossAttention {
                num_heads,
                num_kv_heads,
                head_dim,
            }
            | Op::MultiHeadAttn {
                num_heads,
                num_kv_heads,
                head_dim,
                ..
            } => {
                let heads = Heads::new(num_heads, num_kv_heads, head_dim);
                self.attend(arg(0), arg(1), arg(2), heads, None)
            }
            Op::CachedAttention {
                num_heads,
                num_kv_heads,
                head_dim,
            } => {
                let keys = self.node(arg(1)).ty.shape[0];
                let mask = self.cache_mask(arg(3), keys);
                let heads = Heads::new(num_heads, num_kv_heads, head_dim);
                self.attend(arg(0), arg(1), arg(2), heads, Some(mask))
            }
            Op::BiasedAttention {
                num_heads,
                num_kv_heads,
                head_dim,
                scale_bits,
                causal,
            } => {
                let mask = causal.then(|| self.window_mask(ty.shape[0], 0));
                let heads = Heads::new(num_heads, num_kv_heads, head_dim);
                let scale = f32::from_bits(scale_bits);
                self.attend_biased([arg(0), arg(1), arg(2), arg(3)], heads, scale, mask)
            }
            Op::BiasedCachedAttention {
                num_heads,
                num_kv_heads,
                head_dim,
                scale_bits,
            } => {
                // Every query row shares the cache row biases.
                let (rows, keys) = (ty.shape[0], self.node(arg(1)).ty.shape[0]);
                let mask = self.cache_mask(arg(3), keys);
                let bias = self.repeat_axis(arg(4), num_heads as usize, keys, rows);
                let heads = Heads::new(num_heads, num_kv_heads, head_dim);
                let scale = f32::from_bits(scale_bits);
                self.attend_biased([arg(0), arg(1), arg(2), bias], heads, scale, Some(mask))
            }
            Op::CachedBlockAttention {
                num_heads,
                num_kv_heads,
                head_dim,
                window_size,
            } => {
                // Row `i` sees key `j` when `j - i <= kv_pos`, and with a
                // window `W` also `j - i > kv_pos - W`. Rows from
                // `valid_len` on are unspecified; they come out as zero.
                let rows = ty.shape[0];
                let keys = self.node(arg(1)).ty.shape[0];
                let offsets = self.constant(
                    (0..rows * keys)
                        .map(|n| (n % keys) as f32 - (n / keys) as f32)
                        .collect(),
                    &[rows, keys],
                );
                let pos = self.scalar_row(arg(3), rows * keys);
                let pos = self.view(pos, &[rows, keys]);
                let mut hidden = self.greater(offsets, pos);
                if window_size > 0 {
                    let floor = self.add_scalar(pos, 1.0 - window_size as f32);
                    let old = self.greater(floor, offsets);
                    hidden = self.add(hidden, old);
                }
                let mask = self.scale(hidden, f32::MIN);
                let mask = self.view(mask, &[rows * keys]);
                let heads = Heads::new(num_heads, num_kv_heads, head_dim);
                let out = self.attend(arg(0), arg(1), arg(2), heads, Some(mask));
                let iota = self.constant((0..rows).map(|i| i as f32).collect(), &[rows, 1]);
                let valid = self.scalar_row(arg(4), rows);
                let valid = self.view(valid, &[rows, 1]);
                let keep = self.greater(valid, iota);
                let keep = self.broadcast_inner(keep, ty.shape[1]);
                self.mul(out, keep)
            }
            Op::ChunkedRelativeAttention {
                num_heads,
                head_dim,
                left_context,
                softcap_bits,
            } => self.expand_chunked_relative(
                [arg(0), arg(1), arg(2), arg(3)],
                Heads::new(num_heads, num_heads, head_dim),
                left_context as usize,
                f32::from_bits(softcap_bits),
            ),
            ref other => panic!("{other:?} has no expansion"),
        }
    }

    /// The additive mask hiding cache rows past `kv_pos`, `[keys]`.
    fn cache_mask(&mut self, kv_pos: NodeId, keys: usize) -> NodeId {
        // Key `j` is visible when `j <= kv_pos`.
        let iota = self.constant((0..keys).map(|j| j as f32).collect(), &[1, keys]);
        let pos = self.scalar_row(kv_pos, keys);
        let hidden = self.greater(iota, pos);
        let mask = self.scale(hidden, f32::MIN);
        self.view(mask, &[keys])
    }

    /// [`Self::attend`] with logits `scale · q·k + bias`, the bias holding
    /// `heads · rows · keys` values.
    fn attend_biased(
        &mut self,
        [q, k, v, bias]: [NodeId; 4],
        heads: Heads,
        scale: f32,
        mask: Option<NodeId>,
    ) -> NodeId {
        let (rows, keys) = (self.node(q).ty.shape[0], self.node(k).ty.shape[0]);
        let qh = self.split_heads(q, heads.heads, heads.dim);
        let kh = self.kv_heads(k, heads, keys);
        let vh = self.kv_heads(v, heads, keys);
        let scores = self.batch_matmul_bt(qh, kh);
        let scores = self.scale(scores, scale);
        let bias = self.view(bias, &[heads.heads, rows, keys]);
        let scores = self.add(scores, bias);
        self.weigh(scores, vh, heads, rows, keys, mask)
    }

    /// A `U32` scalar buffer as `[1, n]` copies of its value.
    fn scalar_row(&mut self, scalar: NodeId, n: usize) -> NodeId {
        let value = self.to_f32(scalar);
        let value = self.view(value, &[1, 1]);
        self.broadcast_inner(value, n)
    }

    /// The additive causal mask of `rows` self-attention rows, `[rows²]`,
    /// limited to the last `window` keys when nonzero.
    fn window_mask(&mut self, rows: usize, window: usize) -> NodeId {
        let mask = (0..rows * rows)
            .map(|n| {
                let (i, j) = (n / rows, n % rows);
                let visible = j <= i && (window == 0 || i - j < window);
                if visible { 0.0 } else { f32::MIN }
            })
            .collect();
        self.constant(mask, &[rows * rows])
    }

    /// `x: [rows, heads·dim]` as `[heads, rows, dim]`.
    fn split_heads(&mut self, x: NodeId, heads: usize, dim: usize) -> NodeId {
        let rows = self.node(x).ty.shape[0];
        let x = self.view(x, &[rows, heads, dim]);
        self.permute(x, &[1, 0, 2])
    }

    /// Softmax attention of `q` over `k`/`v` in the shared layout: query
    /// head `h` reads KV head `h / (heads / kv_heads)`, logits are scaled
    /// by `1/√dim`, and `mask` (`[keys]` or `[rows·keys]`) is added to
    /// them when present.
    fn attend(
        &mut self,
        q: NodeId,
        k: NodeId,
        v: NodeId,
        heads: Heads,
        mask: Option<NodeId>,
    ) -> NodeId {
        let (rows, keys) = (self.node(q).ty.shape[0], self.node(k).ty.shape[0]);
        let qh = self.split_heads(q, heads.heads, heads.dim);
        let kh = self.kv_heads(k, heads, keys);
        let vh = self.kv_heads(v, heads, keys);
        let scores = self.batch_matmul_bt(qh, kh);
        let scores = self.scale(scores, heads.scale());
        self.weigh(scores, vh, heads, rows, keys, mask)
    }

    /// `x: [keys, kv_heads·dim]` as `[heads, keys, dim]`, each KV head
    /// repeated for the query heads that share it.
    fn kv_heads(&mut self, x: NodeId, heads: Heads, keys: usize) -> NodeId {
        let xh = self.split_heads(x, heads.kv_heads, heads.dim);
        let group = heads.heads / heads.kv_heads;
        if group == 1 {
            return xh;
        }
        let repeated = self.repeat_axis(xh, heads.kv_heads, keys * heads.dim, group);
        self.view(repeated, &[heads.heads, keys, heads.dim])
    }

    /// Mask, softmax and weigh `vh` by `scores: [heads, rows, keys]`;
    /// returns `[rows, heads·dim]`.
    fn weigh(
        &mut self,
        scores: NodeId,
        vh: NodeId,
        heads: Heads,
        rows: usize,
        keys: usize,
        mask: Option<NodeId>,
    ) -> NodeId {
        let scores = match mask {
            Some(mask) => {
                let width = self.node(mask).ty.num_elements();
                let flat = self.view(scores, &[heads.heads * rows * keys / width, width]);
                self.bias_add(flat, mask)
            }
            None => scores,
        };
        let scores = self.view(scores, &[heads.heads * rows, keys]);
        let weights = self.softmax(scores);
        let weights = self.view(weights, &[heads.heads, rows, keys]);
        let out = self.batch_matmul(weights, vh);
        let out = self.permute(out, &[1, 0, 2]);
        self.view(out, &[rows, heads.heads * heads.dim])
    }

    /// Transformer-XL chunked attention: unscaled logits `q·(k + r)` with
    /// relative key row `left - 1 - (i - j)`, soft-capped, over keys at
    /// distance below `left - 1`.
    fn expand_chunked_relative(
        &mut self,
        [q, k, v, relative]: [NodeId; 4],
        heads: Heads,
        left: usize,
        cap: f32,
    ) -> NodeId {
        let rows = self.node(q).ty.shape[0];
        let qh = self.split_heads(q, heads.heads, heads.dim);
        let kh = self.split_heads(k, heads.heads, heads.dim);
        let vh = self.split_heads(v, heads.heads, heads.dim);
        let rh = self.split_heads(relative, heads.heads, heads.dim);
        let direct = self.batch_matmul_bt(qh, kh);
        // q_i · r_t for every relative row t, then t = left - 1 - (i - j)
        // gathered per (i, j): rows (i, t) of a [rows · left, heads] table.
        // Keys outside the window read row (i, 0); the mask drops them.
        let by_offset = self.batch_matmul_bt(qh, rh);
        let by_offset = self.permute(by_offset, &[1, 2, 0]);
        let table = self.view(by_offset, &[rows * left, heads.heads]);
        let indices: Vec<u32> = (0..rows * rows)
            .map(|n| {
                let (i, j) = (n / rows, n % rows);
                let t = if j <= i && i - j <= left - 2 {
                    left - 1 - (i - j)
                } else {
                    0
                };
                (i * left + t) as u32
            })
            .collect();
        let indices = self.constant_u32(&indices, &[rows * rows]);
        let relative = self.embedding(indices, table);
        let relative = self.view(relative, &[rows, rows, heads.heads]);
        let relative = self.permute(relative, &[2, 0, 1]);
        let logits = self.add(direct, relative);
        let logits = self.scale(logits, 1.0 / cap);
        let logits = self.tanh(logits);
        let logits = self.scale(logits, cap);
        let mask = self.window_mask(rows, left - 1);
        self.weigh(logits, vh, heads, rows, rows, Some(mask))
    }

    /// Rotate pairs `(d, d + dim/2)` of every head by `pos · θ^(-2d/dim)`,
    /// divided by `factors[d]` when present. `θ` and the position offset
    /// stay exact in the graph so recognition reads them back.
    fn expand_rope(
        &mut self,
        x: NodeId,
        theta: f32,
        head_dim: u32,
        positions: Positions,
        factors: Option<NodeId>,
    ) -> NodeId {
        let shape = self.node(x).ty.shape.clone();
        let (rows, width) = (shape[0], shape[1]);
        let dim = head_dim as usize;
        let (heads, half) = (width / dim, dim / 2);
        let pos = match positions {
            Positions::Offset(offset, dynamic) => {
                let iota = self.constant((0..rows).map(|r| r as f32).collect(), &[rows, 1]);
                let pos = self.add_scalar(iota, offset as f32);
                match dynamic {
                    Some(kv) => {
                        let kv = self.scalar_row(kv, rows);
                        let kv = self.view(kv, &[rows, 1]);
                        self.add(pos, kv)
                    }
                    None => pos,
                }
            }
            Positions::PerRow(p) => {
                let p = self.to_f32(p);
                self.view(p, &[rows, 1])
            }
        };
        let theta = self.constant(vec![theta], &[1]);
        let ln_theta = self.log(theta);
        let ln_theta = self.view(ln_theta, &[1, 1]);
        let ln_theta = self.broadcast_inner(ln_theta, half);
        let exponents = self.constant(
            (0..half).map(|d| -2.0 * d as f32 / dim as f32).collect(),
            &[1, half],
        );
        let inv_freq = self.mul(exponents, ln_theta);
        let mut inv_freq = self.exp(inv_freq);
        if let Some(factors) = factors {
            let factors = self.view(factors, &[1, half]);
            let inv = self.recip(factors);
            inv_freq = self.mul(inv_freq, inv);
        }
        let angle = self.matmul(pos, inv_freq);
        let cos = self.cos(angle);
        let cos = self.view(cos, &[rows * half]);
        let sin = self.sin(angle);
        let sin = self.view(sin, &[rows * half]);
        let xh = self.split_heads(x, heads, dim);
        let blocks = u32::try_from(heads * rows).expect("rope rows exceed u32");
        let half_u32 = half as u32;
        let first = self.split_a(xh, blocks, half_u32, half_u32, 1);
        let first = self.view(first, &[heads, rows * half]);
        let second = self.split_b(xh, blocks, half_u32, half_u32, 1);
        let second = self.view(second, &[heads, rows * half]);
        let a = self.bias_mul(first, cos);
        let b = self.bias_mul(second, sin);
        let out_first = self.sub(a, b);
        let c = self.bias_mul(first, sin);
        let d = self.bias_mul(second, cos);
        let out_second = self.add(c, d);
        let out_first = self.view(out_first, &[heads * rows * half]);
        let out_second = self.view(out_second, &[heads * rows * half]);
        let joined = self.concat(out_first, out_second, blocks, half_u32, half_u32, 1);
        let joined = self.view(joined, &[heads, rows, dim]);
        let joined = self.permute(joined, &[1, 0, 2]);
        self.view(joined, &[rows, width])
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
    /// index: `[outer, times, inner]`, one broadcast read.
    pub(crate) fn repeat_axis(
        &mut self,
        x: NodeId,
        outer: usize,
        inner: usize,
        times: usize,
    ) -> NodeId {
        let x = self.view(x, &[outer, 1, inner]);
        self.broadcast_to(x, &[outer, times, inner])
    }

    /// `[rows, width]` zeros, read from one broadcast scalar.
    pub(crate) fn zeros(&mut self, rows: usize, width: usize) -> NodeId {
        let zero = self.constant(vec![0.0], &[1, 1]);
        self.broadcast_to(zero, &[rows, width])
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
        let zeros = self.zeros(rows, gap as usize);
        if o > 0 {
            let kept = self.split_a(x, rows_u32, keep, gap, 1);
            self.concat(zeros, kept, rows_u32, gap, keep, 1)
        } else {
            let kept = self.split_b(x, rows_u32, gap, keep, 1);
            self.concat(kept, zeros, rows_u32, keep, gap, 1)
        }
    }

    /// The graph with every recognizable expansion replaced by its
    /// composite, superseded nodes swept and the rest sorted. Composites
    /// already named stay as written.
    ///
    /// Composites that occur inside other composites' expansions (the
    /// activations, softmaxes and norms) are recognized first, so the larger
    /// ones match them by name, as a graph that names them already does.
    /// Recognition then runs from the outputs back: the outermost composite
    /// claims its nodes before a smaller one could match part of them.
    pub fn recompose(&self) -> Graph {
        self.recompose_where(|_, _| true)
    }

    /// [`Graph::recompose`] restricted to composites whose gradient is
    /// exactly their decomposition's ([`Op::differentiates_as_decomposed`]),
    /// so recognizing a training graph never changes what it learns.
    pub fn recompose_for_training(&self) -> Graph {
        self.recompose_where(|op, inputs| op.differentiates_as_decomposed(inputs.len()))
    }

    fn recompose_where(&self, allow: impl Fn(&Op, &[NodeId]) -> bool) -> Graph {
        let mut graph = self.deep_clone();
        graph.canonicalize_attributes();
        if graph.canonicalize_primitives() {
            // Recognition walks the ids from the outputs back.
            graph = graph.into_toposort();
        }
        let mut templates = HashMap::new();
        graph.recognize(|op, ins| is_nested(op) && allow(op, ins), &mut templates);
        graph.recognize(|op, ins| !is_nested(op) && allow(op, ins), &mut templates);
        // Drop what the composites superseded, so later passes see a graph
        // the size of the one written with composites.
        crate::optimize::sweep_dead_nodes(&mut graph);
        graph.into_toposort()
    }

    /// One sweep from the outputs back, replacing each node that roots the
    /// expansion of a composite `allow` picks by that composite.
    fn recognize(&mut self, allow: impl Fn(&Op, &[NodeId]) -> bool, templates: &mut Templates) {
        // Reshapes of each node: a guess finds an input up to the reshapes
        // around it, and the input is whichever one the expansion reads.
        let mut views: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        for node in &self.nodes {
            if matches!(node.op, Op::Identity) {
                views.entry(node.inputs[0]).or_default().push(node.id);
            }
        }
        for id in (0..self.nodes.len() as NodeId).rev() {
            if matches!(self.node(id).op, Op::Nop) {
                continue;
            }
            // Of the composites the node is the root of, the largest: a
            // smaller one matching is only part of it.
            let mut best: Option<(usize, Op, Vec<NodeId>)> = None;
            for (op, inputs) in guesses(self, id) {
                if !allow(&op, &inputs) {
                    continue;
                }
                for inputs in input_variants(self, &views, &inputs) {
                    if let Some(size) = self.expands_to(id, &op, &inputs, templates)
                        && best.as_ref().is_none_or(|b| size > b.0)
                    {
                        best = Some((size, op.clone(), inputs));
                        break;
                    }
                }
            }
            if let Some((_, op, inputs)) = best {
                let node = &mut self.nodes[id as usize];
                node.op = op;
                node.inputs = inputs;
                node.matmul_impl = None;
            }
        }
    }

    /// Rewrite primitive spellings that differ from the expansions' only
    /// by an exact identity: `recip(sqrt(x))` is `rsqrt(x)`, and the
    /// reciprocal of a row broadcast is the broadcast of the reciprocal.
    /// Whether it appended nodes.
    fn canonicalize_primitives(&mut self) -> bool {
        let before = self.nodes.len();
        for id in 0..before {
            let Op::Recip = self.nodes[id].op else {
                continue;
            };
            let src = self.nodes[id].inputs[0];
            match self.nodes[src as usize].op {
                Op::Sqrt => {
                    let x = self.nodes[src as usize].inputs[0];
                    self.nodes[id].op = Op::Rsqrt;
                    self.nodes[id].inputs = vec![x];
                }
                Op::BroadcastInner { .. } => {
                    // bcast(recip(s)), with a fresh recip of the row values.
                    let s_id = self.nodes[src as usize].inputs[0];
                    let bcast = self.nodes[src as usize].op.clone();
                    let mut ty = self.nodes[s_id as usize].ty.clone();
                    ty.dtype = self.nodes[id].ty.dtype;
                    let inner = match self.nodes[s_id as usize].op {
                        Op::Sqrt => {
                            let x = self.nodes[s_id as usize].inputs[0];
                            self.add_raw_node(Op::Rsqrt, vec![x], ty)
                        }
                        _ => self.add_raw_node(Op::Recip, vec![s_id], ty),
                    };
                    self.nodes[id].op = bcast;
                    self.nodes[id].inputs = vec![inner];
                }
                _ => {}
            }
        }
        self.nodes.len() > before
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
                // Every key visible: one op, whichever spelling.
                Op::FullAttention {
                    num_heads,
                    num_kv_heads,
                    head_dim,
                }
                | Op::CrossAttention {
                    num_heads,
                    num_kv_heads,
                    head_dim,
                }
                | Op::MultiHeadAttn {
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    ..
                } => Some(Op::MultiHeadAttn {
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    is_cross: self.node(inputs[0]).ty.shape[0] != self.node(inputs[1]).ty.shape[0],
                }),
                // A window spanning the sequence is plain causal attention.
                Op::SlidingWindowAttention {
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    window_size,
                } if window_size as usize >= self.node(inputs[0]).ty.shape[0] => {
                    Some(Op::CausalAttention {
                        num_heads,
                        num_kv_heads,
                        head_dim,
                    })
                }
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
    /// `op` over `inputs`, and if so the decomposition's size.
    fn expands_to(
        &self,
        root: NodeId,
        op: &Op,
        inputs: &[NodeId],
        templates: &mut Templates,
    ) -> Option<usize> {
        if op.class() != OpClass::Composite || inputs.iter().any(|&i| i >= root) {
            return None;
        }
        let types: Vec<&TensorType> = inputs.iter().map(|&i| &self.node(i).ty).collect();
        let ty = &self.node(root).ty;
        if !accepts(op, &types, ty) {
            return None;
        }
        let key = format!("{op:?} {types:?} {ty:?}");
        let template = templates
            .entry(key)
            .or_insert_with(|| template(op, &types, ty));
        let &(ref template, top) = template.as_ref()?;
        Matcher {
            template,
            graph: self,
            inputs,
            memo: HashMap::new(),
        }
        .same(top, root)
        .then_some(template.nodes().len())
    }
}

/// Verified templates by op and types: the expansion and its root.
type Templates = HashMap<String, Option<(Graph, NodeId)>>;

/// Where a RoPE row sits: its index plus a static offset and an optional
/// `U32` scalar buffer, or an explicit `U32` position per row.
enum Positions {
    Offset(u32, Option<NodeId>),
    PerRow(NodeId),
}

/// Head layout of an attention op.
#[derive(Clone, Copy)]
struct Heads {
    heads: usize,
    kv_heads: usize,
    dim: usize,
}

impl Heads {
    fn new(heads: u32, kv_heads: u32, dim: u32) -> Self {
        Self {
            heads: heads as usize,
            kv_heads: kv_heads as usize,
            dim: dim as usize,
        }
    }

    /// The logit scale every attention kernel applies.
    fn scale(self) -> f32 {
        1.0 / (self.dim as f32).sqrt()
    }
}

/// `inputs` first, then with inputs swapped for the reshapes around them:
/// the views of each input, and what an input that is itself a view reads.
fn input_variants(
    g: &Graph,
    views: &HashMap<NodeId, Vec<NodeId>>,
    inputs: &[NodeId],
) -> Vec<Vec<NodeId>> {
    const LIMIT: usize = 32;
    let choices: Vec<Vec<NodeId>> = inputs
        .iter()
        .map(|&i| {
            let mut near = vec![i];
            let mut frontier = vec![i];
            while let Some(n) = frontier.pop() {
                for &v in views.get(&n).into_iter().flatten() {
                    if !near.contains(&v) && near.len() < 4 {
                        near.push(v);
                        frontier.push(v);
                    }
                }
            }
            if matches!(g.node(i).op, Op::Identity) && !near.contains(&g.node(i).inputs[0]) {
                near.push(g.node(i).inputs[0]);
            }
            near
        })
        .collect();
    let mut out = vec![Vec::new()];
    for choice in &choices {
        out = out
            .into_iter()
            .flat_map(|prefix| {
                choice.iter().map(move |&c| {
                    let mut next = prefix.clone();
                    next.push(c);
                    next
                })
            })
            .take(LIMIT)
            .collect();
    }
    out
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

/// The decomposition of `op` over placeholder inputs of `types` into
/// primitives and [nested](is_nested) composites, and its root.
/// Placeholders are the first nodes.
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
    // Nested composites recognized as in the graph's first sweep, so both
    // name them over the same views.
    let mut full = graph.decompose();
    if !is_nested(op) {
        full.recognize(|op, _| is_nested(op), &mut HashMap::new());
    }
    let top = full.outputs()[0];
    Some((full, top))
}

/// Whether `op` is well-typed over `ins` producing `out`: the contract its
/// builder enforces, which its expansion relies on.
fn accepts(op: &Op, ins: &[&TensorType], out: &TensorType) -> bool {
    use super::DType::{F32, U32};
    // Only position and length scalars are integers.
    let integer_inputs = matches!(
        *op,
        Op::RoPE { .. }
            | Op::RoPEPositions { .. }
            | Op::CachedAttention { .. }
            | Op::CachedBlockAttention { .. }
            | Op::BiasedCachedAttention { .. }
    );
    if out.dtype != F32 || (!integer_inputs && ins.iter().any(|t| t.dtype != F32)) {
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
        Op::ShiftInner { .. } => arity(1) && ins[0] == out && matrix(out).is_some(),
        Op::CrossEntropyLoss => {
            arity(2) && matrix(ins[0]).is_some() && ins[1] == ins[0] && out.shape == [1]
        }
        Op::BceLoss => arity(2) && ins[1] == ins[0] && out.shape == [1],
        // Per-channel ops and pooling act on flat NCHW tensors.
        Op::MulPerChannel { spatial, .. } => {
            arity(2)
                && ins[0] == out
                && out.rank() == 1
                && elems(ins[1]) * spatial as usize == elems(out)
        }
        Op::AddPerChannel { channels, spatial } => {
            let plane = channels as usize * spatial as usize;
            arity(2)
                && ins[0] == out
                && out.rank() == 1
                && elems(ins[1]) == channels as usize
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
                && elems(ins[1]) == channels as usize
                && elems(ins[2]) == channels as usize
                && num_groups > 0
                && channels % num_groups == 0
                && plane > 0
                && elems(out) % plane == 0
        }
        Op::Upsample2x { in_w, .. } => {
            arity(1)
                && out.shape == [4 * elems(ins[0])]
                && in_w > 0
                && elems(ins[0]) % in_w as usize == 0
        }
        Op::RoPE {
            head_dim,
            freq_factors,
            ..
        } => {
            let d = head_dim as usize;
            let scalar = |t: &TensorType| t.dtype == U32 && t.shape == [1];
            let extra_ok = match ins.len() {
                1 => !freq_factors,
                2 => !freq_factors && scalar(ins[1]),
                3 => {
                    freq_factors && scalar(ins[1]) && ins[2].dtype == F32 && ins[2].shape == [d / 2]
                }
                _ => false,
            };
            extra_ok && rope_x_ok(ins[0], out, d)
        }
        Op::RoPEPositions { head_dim, .. } => {
            arity(2)
                && rope_x_ok(ins[0], out, head_dim as usize)
                && ins[1].dtype == U32
                && ins[1].shape == [out.shape[0]]
        }
        Op::CausalAttention {
            num_heads,
            num_kv_heads,
            head_dim,
        }
        | Op::SlidingWindowAttention {
            num_heads,
            num_kv_heads,
            head_dim,
            ..
        } => {
            arity(3)
                && attention_ok(ins, out, num_heads, num_kv_heads, head_dim)
                && ins[1].shape[0] == out.shape[0]
        }
        Op::FullAttention {
            num_heads,
            num_kv_heads,
            head_dim,
        }
        | Op::CrossAttention {
            num_heads,
            num_kv_heads,
            head_dim,
        }
        | Op::MultiHeadAttn {
            num_heads,
            num_kv_heads,
            head_dim,
            ..
        } => arity(3) && attention_ok(ins, out, num_heads, num_kv_heads, head_dim),
        Op::CachedAttention {
            num_heads,
            num_kv_heads,
            head_dim,
        } => {
            arity(4)
                && attention_ok(&ins[..3], out, num_heads, num_kv_heads, head_dim)
                && ins[3].dtype == U32
                && ins[3].shape == [1]
        }
        Op::CachedBlockAttention {
            num_heads,
            num_kv_heads,
            head_dim,
            ..
        } => {
            arity(5)
                && attention_ok(&ins[..3], out, num_heads, num_kv_heads, head_dim)
                && ins[3..].iter().all(|t| t.dtype == U32 && t.shape == [1])
        }
        Op::BiasedAttention {
            num_heads,
            num_kv_heads,
            head_dim,
            scale_bits,
            causal,
        } => {
            let (rows, keys) = (out.shape[0], ins[1].shape.first().copied().unwrap_or(0));
            arity(4)
                && head_dim <= 512
                && attention_ok(&ins[..3], out, num_heads, num_kv_heads, head_dim)
                && f32::from_bits(scale_bits).is_finite()
                && (!causal || rows == keys)
                && ins[3].shape == [num_heads as usize, rows, keys]
        }
        Op::BiasedCachedAttention {
            num_heads,
            num_kv_heads,
            head_dim,
            scale_bits,
        } => {
            let keys = ins[1].shape.first().copied().unwrap_or(0);
            arity(5)
                && head_dim <= 512
                && attention_ok(&ins[..3], out, num_heads, num_kv_heads, head_dim)
                && f32::from_bits(scale_bits).is_finite()
                && ins[3].dtype == U32
                && ins[3].shape == [1]
                && ins[4].dtype == F32
                && ins[4].shape == [num_heads as usize, keys]
        }
        Op::ChunkedRelativeAttention {
            num_heads,
            head_dim,
            left_context,
            softcap_bits,
        } => {
            let cap = f32::from_bits(softcap_bits);
            arity(4)
                && attention_ok(&ins[..3], out, num_heads, num_heads, head_dim)
                && ins[1].shape == out.shape
                && ins[3].dtype == F32
                && ins[3].shape == [left_context as usize, out.shape[1]]
                && left_context > 1
                && cap > 0.0
                && cap.is_finite()
        }
        _ => true,
    }
}

/// A RoPE input: `[rows, heads · dim]` with an even `dim`.
fn rope_x_ok(x: &TensorType, out: &TensorType, dim: usize) -> bool {
    x == out
        && x.dtype == super::DType::F32
        && dim > 0
        && dim.is_multiple_of(2)
        && matches!(x.shape[..], [rows, width] if rows > 0 && width > 0 && width.is_multiple_of(dim))
}

/// Q, K and V of an attention op in the shared layout.
fn attention_ok(ins: &[&TensorType], out: &TensorType, heads: u32, kv: u32, dim: u32) -> bool {
    let (heads, kv, dim) = (heads as usize, kv as usize, dim as usize);
    let f32_matrix = |t: &TensorType, cols: usize| {
        t.dtype == super::DType::F32 && matches!(t.shape[..], [rows, c] if rows > 0 && c == cols)
    };
    heads > 0
        && kv > 0
        && dim > 0
        && heads.is_multiple_of(kv)
        && ins[0] == out
        && f32_matrix(out, heads * dim)
        && f32_matrix(ins[1], kv * dim)
        && ins[2] == ins[1]
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
        (None, None) => super::key::structural_key(a) == super::key::structural_key(b),
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
                    out.push((Op::Softmax, vec![x]));
                }
                // cached block attention: out · bcast(rows kept)
                if let Some(perm) = Some(unview(g, a))
                    && matches!(g.node(perm).op, Op::Permute { .. })
                    && let Some(product) = input(g, perm, 0).map(|p| unview(g, p))
                    && matches!(g.node(product).op, Op::BatchMatMul)
                {
                    out.extend(attention_guess(g, product, Some(b)));
                }
                // silu: x · sigmoid(x)
                if is(g, b, |op| matches!(op, Op::Sigmoid)) && input(g, b, 0) == Some(a) {
                    out.push((Op::Silu, vec![a]));
                }
                // gelu: x · sigmoid(...)
                if is(g, b, |op| matches!(op, Op::Sigmoid)) {
                    out.push((Op::Gelu, vec![a]));
                }
                // glu: act(gate) · up
                if let Some(gate) = input(g, a, 0) {
                    match g.node(a).op {
                        Op::Silu => out.push((Op::SwiGLU, vec![gate, b])),
                        Op::Gelu => out.push((Op::GeGLU, vec![gate, b])),
                        _ => {}
                    }
                }
                // per-channel gate over flat planes: view(src) · bcast(view(gate))
                if let Some(gate) = input(g, b, 0)
                    && is(g, b, |op| matches!(op, Op::BroadcastInner { .. }))
                {
                    let (src, gate) = (unview(g, a), unview(g, gate));
                    let (channels, spatial) = per_channel_dims(g, node, gate);
                    out.push((Op::MulPerChannel { channels, spatial }, vec![src, gate]));
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
            // layer_norm: rms_norm(x - mean, w) + b
            if let (Some(norm), Some(b)) = (input(g, core, 0), input(g, core, 1))
                && is(g, norm, |op| matches!(op, Op::RmsNorm { .. }))
                && let (Some(centered), Some(w)) = (input(g, norm, 0), input(g, norm, 1))
                && let Op::RmsNorm { eps } = g.node(norm).op
                && let Some(x) = input(g, centered, 0)
            {
                out.push((Op::LayerNorm { eps }, vec![unview(g, x), w, b]));
                out.push((Op::LayerNorm { eps }, vec![x, w, b]));
            }
            // add_per_channel: view(src) + spread(bias)
            if let (Some(src), Some(bias)) = (input(g, core, 0), input(g, core, 1)) {
                let bias = unview(g, unview(g, bias));
                let bias = input(g, bias, 0).map_or(bias, |b| unview(g, b));
                let channels = g.node(bias).ty.num_elements() as u32;
                let plane = g.node(src).ty.shape.get(1).copied().unwrap_or(0) as u32;
                if channels > 0 && plane.is_multiple_of(channels) {
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
        Op::AddPerChannel { .. } => {
            // group_norm: add_per_channel(view(groups normalized ⊙ spread(w)), b)
            if let (Some(scaled), Some(bias)) = (input(g, core, 0), input(g, core, 1)) {
                out.extend(group_norm_guess(g, scaled, bias));
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
                    // cross entropy: -Σ labels · log_softmax(logits) / B
                    for (labels, log_p) in operands(g, terms) {
                        if is(g, log_p, |op| matches!(op, Op::LogSoftmax))
                            && let Some(logits) = input(g, log_p, 0)
                        {
                            out.push((Op::CrossEntropyLoss, vec![logits, labels]));
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
        Op::Permute { .. } => {
            if let Some(inner) = input(g, core, 0) {
                let inner = unview(g, inner);
                match g.node(inner).op {
                    Op::Concat { .. } => out.extend(rope_guess(g, inner)),
                    Op::BatchMatMul => out.extend(attention_guess(g, inner, None)),
                    _ => {}
                }
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
            }
        }
        Op::BroadcastTo => {
            // upsample: each pixel spread over a 2×2 block
            if let Some(x) = input(g, core, 0)
                && let [rows, 2, w, 2] = g.node(core).ty.shape[..]
                && g.node(x).ty.shape[..] == [rows, 1, w, 1]
            {
                out.push((
                    Op::Upsample2x {
                        channels: 1,
                        in_h: rows as u32,
                        in_w: w as u32,
                    },
                    vec![unview(g, x)],
                ));
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

/// Constant data of a node, if it is one.
fn constant(g: &Graph, id: NodeId) -> Option<&[f32]> {
    match g.node(id).op {
        Op::Constant { ref data } => Some(data),
        _ => None,
    }
}

/// The `U32` scalar a `scalar_row` broadcast reads.
fn scalar_source(g: &Graph, id: NodeId) -> Option<NodeId> {
    let bcast = unview(g, id);
    let value = unview(g, input(g, bcast, 0)?);
    matches!(g.node(value).op, Op::ToF32).then(|| input(g, value, 0))?
}

/// The tensor behind a `split_heads` (or its repetition for grouped KV).
fn heads_source(g: &Graph, id: NodeId) -> Option<NodeId> {
    heads_split(g, id).map(|(source, _)| source)
}

/// The tensor behind a `split_heads`, and the split `[heads, rows, dim]`
/// itself, before any repetition for grouped KV.
fn heads_split(g: &Graph, id: NodeId) -> Option<(NodeId, NodeId)> {
    let mut id = unview(g, id);
    if matches!(g.node(id).op, Op::BroadcastTo) {
        id = unview(g, input(g, id, 0)?);
    }
    if !matches!(g.node(id).op, Op::Permute { .. }) {
        return None;
    }
    Some((unview(g, input(g, id, 0)?), id))
}

/// A RoPE whose concatenated halves are `concat`.
fn rope_guess(g: &Graph, concat: NodeId) -> Vec<(Op, Vec<NodeId>)> {
    let mut out = Vec::new();
    let Op::Concat { channels_a, .. } = g.node(concat).op else {
        return out;
    };
    let head_dim = 2 * channels_a;
    let Some(first) = input(g, concat, 0).map(|f| unview(g, f)) else {
        return out;
    };
    for (a, _) in operands(g, first) {
        let (Some(split), Some(cos)) = (input(g, a, 0), input(g, a, 1)) else {
            continue;
        };
        let split = unview(g, split);
        let cos = unview(g, cos);
        let (Some(xh), Some(angle)) = (input(g, split, 0), input(g, cos, 0)) else {
            continue;
        };
        let Some(x) = input(g, xh, 0).map(|x| unview(g, x)) else {
            continue;
        };
        let (Some(pos), Some(inv)) = (input(g, angle, 0), input(g, angle, 1)) else {
            continue;
        };
        // inv_freq = exp(exponents · ln θ), divided by factors when present
        let mut factors = None;
        let mut exp = inv;
        if matches!(g.node(inv).op, Op::Mul) {
            for (e, r) in operands(g, inv) {
                if matches!(g.node(r).op, Op::Recip) {
                    factors = input(g, r, 0).map(|f| unview(g, f));
                    exp = e;
                }
            }
        }
        let theta = input(g, exp, 0).and_then(|product| {
            operands(g, product).into_iter().find_map(|(_, ln)| {
                let log = unview(g, input(g, ln, 0)?);
                let c = input(g, log, 0)?;
                constant(g, c).and_then(|d| d.first().copied())
            })
        });
        let Some(theta) = theta else {
            continue;
        };
        match g.node(pos).op {
            Op::Offset { value } => out.push((
                Op::RoPE {
                    theta,
                    pos_offset: value as u32,
                    head_dim,
                    freq_factors: false,
                },
                vec![x],
            )),
            Op::Add => {
                for (offset, kv) in operands(g, pos) {
                    if let Op::Offset { value } = g.node(offset).op
                        && let Some(kv) = scalar_source(g, kv)
                    {
                        let mut inputs = vec![x, kv];
                        inputs.extend(factors);
                        out.push((
                            Op::RoPE {
                                theta,
                                pos_offset: value as u32,
                                head_dim,
                                freq_factors: factors.is_some(),
                            },
                            inputs,
                        ));
                    }
                }
            }
            _ => {
                let converted = unview(g, pos);
                if matches!(g.node(converted).op, Op::ToF32)
                    && let Some(p) = input(g, converted, 0)
                {
                    out.push((Op::RoPEPositions { theta, head_dim }, vec![x, p]));
                }
            }
        }
    }
    out
}

/// Attention whose value product is `product`; `keep` is the row mask of a
/// cached block, when the root multiplies by one.
fn attention_guess(g: &Graph, product: NodeId, keep: Option<NodeId>) -> Vec<(Op, Vec<NodeId>)> {
    let mut out = Vec::new();
    let (Some(weights), Some(vh)) = (input(g, product, 0), input(g, product, 1)) else {
        return out;
    };
    let Some(v) = heads_source(g, vh) else {
        return out;
    };
    let softmax = unview(g, weights);
    let Some(scores) =
        input(g, softmax, 0).filter(|_| is(g, softmax, |op| matches!(op, Op::Softmax)))
    else {
        return out;
    };
    let mut scores = unview(g, scores);
    let mut mask = None;
    if matches!(g.node(scores).op, Op::BiasAdd) {
        mask = input(g, scores, 1);
        scores = unview(g, input(g, scores, 0).unwrap_or(scores));
    }
    // An additive bias between the scaled scores and the mask.
    let mut bias = None;
    if matches!(g.node(scores).op, Op::Add) {
        for (a, b) in operands(g, scores) {
            if matches!(g.node(a).op, Op::Scale { .. }) {
                scores = a;
                bias = Some(b);
                break;
            }
        }
    }
    let Op::Scale { factor } = g.node(scores).op else {
        return out;
    };
    let Some(logits) = input(g, scores, 0) else {
        return out;
    };
    let qh_of = |id: NodeId| -> Option<(NodeId, NodeId, NodeId, NodeId)> {
        let qh = input(g, id, 0)?;
        let (k, kh) = heads_split(g, input(g, id, 1)?)?;
        Some((qh, heads_source(g, qh)?, k, kh))
    };
    // Chunked relative: cap · tanh((q·k + relative) / cap)
    if matches!(g.node(logits).op, Op::Tanh) {
        let Some(sum) = input(g, logits, 0).and_then(|s| input(g, s, 0)) else {
            return out;
        };
        for (direct, relative) in operands(g, sum) {
            let Some((qh, q, k, _)) = qh_of(direct) else {
                continue;
            };
            // permute(gather(view(permute(batch_matmul_bt(qh, rh)))))
            let rel = (|| {
                let gathered = unview(g, input(g, relative, 0)?);
                let table = unview(g, input(g, gathered, 1)?);
                let by_offset = unview(g, input(g, table, 0)?);
                heads_source(g, input(g, by_offset, 1)?)
            })();
            let (Some(rel), Some(&[heads, _, dim])) = (rel, Some(&g.node(qh).ty.shape[..])) else {
                continue;
            };
            out.push((
                Op::ChunkedRelativeAttention {
                    num_heads: heads as u32,
                    head_dim: dim as u32,
                    left_context: g.node(rel).ty.shape[0] as u32,
                    softcap_bits: factor.to_bits(),
                },
                vec![q, k, v, rel],
            ));
        }
        return out;
    }
    let Some((qh, q, k, kh)) = qh_of(logits) else {
        return out;
    };
    let (&[heads, rows, dim], &[kv_heads, keys, _]) =
        (&g.node(qh).ty.shape[..], &g.node(kh).ty.shape[..])
    else {
        return out;
    };
    let (heads, kv_heads, dim) = (heads as u32, kv_heads as u32, dim as u32);
    if let Some(bias) = bias {
        return biased_guess(g, [q, k, v, bias], [heads, kv_heads, dim], factor, mask);
    }
    match mask.map(|m| unview(g, m)) {
        None => out.push((
            Op::MultiHeadAttn {
                num_heads: heads,
                num_kv_heads: kv_heads,
                head_dim: dim,
                is_cross: rows != keys,
            },
            vec![q, k, v],
        )),
        Some(m) if constant(g, m).is_some() => {
            let data = constant(g, m).unwrap_or_default();
            out.push((
                Op::CausalAttention {
                    num_heads: heads,
                    num_kv_heads: kv_heads,
                    head_dim: dim,
                },
                vec![q, k, v],
            ));
            // The last row sees the window.
            let window = data[data.len().saturating_sub(rows)..]
                .iter()
                .filter(|&&x| x == 0.0)
                .count();
            out.push((
                Op::SlidingWindowAttention {
                    num_heads: heads,
                    num_kv_heads: kv_heads,
                    head_dim: dim,
                    window_size: window as u32,
                },
                vec![q, k, v],
            ));
        }
        Some(m) => {
            // Cached: scale(hidden, MIN), hidden = greater(·, kv_pos) [+ window]
            let Some(hidden) = input(g, m, 0) else {
                return out;
            };
            let mut window = 0;
            let mut comparison = hidden;
            if matches!(g.node(hidden).op, Op::Add) {
                for (a, b) in operands(g, hidden) {
                    if let Some(floor) = input(g, b, 0)
                        && let Op::Offset { value } = g.node(floor).op
                    {
                        window = (1.0 - value) as u32;
                        comparison = a;
                    }
                }
            }
            let Some(pos) = input(g, comparison, 1) else {
                return out;
            };
            let Some(kv_pos) = scalar_source(g, pos) else {
                return out;
            };
            let heads_attrs = (heads, kv_heads, dim);
            match keep {
                None => out.push((
                    Op::CachedAttention {
                        num_heads: heads_attrs.0,
                        num_kv_heads: heads_attrs.1,
                        head_dim: heads_attrs.2,
                    },
                    vec![q, k, v, kv_pos],
                )),
                Some(keep) => {
                    let valid = (|| {
                        let greater = input(g, keep, 0)?;
                        scalar_source(g, input(g, greater, 0)?)
                    })();
                    if let Some(valid) = valid {
                        out.push((
                            Op::CachedBlockAttention {
                                num_heads: heads_attrs.0,
                                num_kv_heads: heads_attrs.1,
                                head_dim: heads_attrs.2,
                                window_size: window,
                            },
                            vec![q, k, v, kv_pos, valid],
                        ));
                    }
                }
            }
        }
    }
    out
}

/// Biased attention around `bias`, by its mask: none, causal (a constant)
/// or a cache prefix, whose bias rows are repeated for every query row.
fn biased_guess(
    g: &Graph,
    [q, k, v, bias]: [NodeId; 4],
    [num_heads, num_kv_heads, head_dim]: [u32; 3],
    scale: f32,
    mask: Option<NodeId>,
) -> Vec<(Op, Vec<NodeId>)> {
    let scale_bits = scale.to_bits();
    let bias = unview(g, bias);
    let biased = |causal| Op::BiasedAttention {
        num_heads,
        num_kv_heads,
        head_dim,
        scale_bits,
        causal,
    };
    let Some(mask) = mask.map(|m| unview(g, m)) else {
        return vec![(biased(false), vec![q, k, v, bias])];
    };
    if constant(g, mask).is_some() {
        return vec![(biased(true), vec![q, k, v, bias])];
    }
    let kv_pos = input(g, mask, 0)
        .and_then(|hidden| input(g, hidden, 1))
        .and_then(|pos| scalar_source(g, pos));
    let mut rows = bias;
    if matches!(g.node(rows).op, Op::BroadcastTo)
        && let Some(source) = input(g, rows, 0)
    {
        rows = unview(g, source);
    }
    kv_pos
        .map(|kv_pos| {
            vec![(
                Op::BiasedCachedAttention {
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    scale_bits,
                },
                vec![q, k, v, kv_pos, rows],
            )]
        })
        .unwrap_or_default()
}

/// Group norm whose per-channel bias add is over `scaled`:
/// `view(groups normalized ⊙ spread(w))`.
fn group_norm_guess(g: &Graph, scaled: NodeId, bias: NodeId) -> Vec<(Op, Vec<NodeId>)> {
    let mut out = Vec::new();
    let scaled = unview(g, scaled);
    if !is(g, scaled, |op| matches!(op, Op::BiasMul)) {
        return out;
    }
    let (Some(normalized), Some(spread_w)) = (input(g, scaled, 0), input(g, scaled, 1)) else {
        return out;
    };
    let normalized = unview(g, normalized);
    let Some(w) = input(g, unview(g, spread_w), 0).map(|c| unview(g, c)) else {
        return out;
    };
    for (centered, inv) in operands(g, normalized) {
        if let Some(eps) = norm_eps(g, inv)
            && let Some(groups) = input(g, centered, 0)
        {
            let channels = g.node(w).ty.num_elements() as u32;
            let group_len = g.node(groups).ty.shape.get(1).copied().unwrap_or(0);
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
    if matches!(g.node(base).op, Op::BroadcastTo) {
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
