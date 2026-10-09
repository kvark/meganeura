//! Lower graph nodes into dispatches.

use super::*;
use crate::kernels::attention_grad::{self, AttentionGrad, Part};
use crate::kernels::conv;

impl<'a> Compiler<'a> {
    pub(super) fn new_with_options(
        graph: &'a Graph,
        options: CompileOptions,
        coop_caps: crate::codegen::CoopCaps,
        shared_memory_bytes: u32,
        allow_reduced_precision_attention_backward: bool,
    ) -> Self {
        if let Some(shape) = options.gemv_shape {
            shape.validate();
        }
        Self {
            graph,
            plan: ExecutionPlan {
                buffers: vec![],
                param_buffers: vec![],
                param_types: HashMap::new(),
                input_buffers: Vec::new(),
                constant_buffers: Vec::new(),
                dispatches: Vec::new(),
                groups: Vec::new(),
                loss_buffer: None,
                output_buffers: Vec::new(),
                param_grad_pairs: Vec::new(),
                lse_buffers: Vec::new(),
                derived_params: Vec::new(),
                weight_buffers: HashMap::new(),
                node_buffers: Vec::new(),
                node_names: Vec::new(),
                knobs: options.knobs,
            },
            node_buffers: HashMap::new(),
            options,
            coop_caps,
            shared_memory_bytes,
            allow_reduced_precision_attention_backward,
            pending_grad_v_buffers: HashMap::new(),
            attention_row_dots: HashMap::new(),
            group_norm_grad_stats: HashMap::new(),
        }
    }

    pub(super) fn into_plan(mut self) -> ExecutionPlan {
        self.compile();

        // Propagate derived parameter info from graph to plan.
        for derived in &self.graph.derived_params {
            if let Some(&(_, buffer)) = self
                .plan
                .param_buffers
                .iter()
                .find(|entry| entry.0 == derived.name)
            {
                let sources = derived
                    .sources
                    .iter()
                    .map(|entry| (entry.0.clone(), entry.1))
                    .collect();
                self.plan
                    .derived_params
                    .push((buffer, sources, derived.transform.clone()));
            }
        }

        self.plan
    }

    pub(super) fn alloc_buffer(&mut self, size_bytes: usize) -> BufferRef {
        let idx = self.plan.buffers.len() as u32;
        self.plan.buffers.push(size_bytes);
        BufferRef(idx)
    }

    /// How many row slices a cooperative [`ShaderEntry::SumRows`] launch needs.
    ///
    /// Eight lanes already share each column. A tall matrix with few column
    /// groups still leaves those lanes on a long chain and the GPU idle.
    /// Short or wide reductions stay one slice so their rounding is unchanged.
    pub(super) fn row_reduction_splits(rows: u32, cols: u32) -> u32 {
        let groups = cols.div_ceil(32).max(1);
        let trip = rows.div_ceil(8);
        if trip <= 64 || groups >= 128 {
            return 1;
        }
        let want = 128u32.div_ceil(groups);
        let cap = trip.div_ceil(32).clamp(2, 256);
        want.clamp(2, cap)
    }

    pub(super) fn push_sum_rows(&mut self, rows: u32, cols: u32, src: BufferRef, dst: BufferRef) {
        let splits = Self::row_reduction_splits(rows, cols);
        if splits == 1 {
            self.plan.dispatches.push(Dispatch {
                shader: ShaderEntry::SumRows,
                workgroups: [cols.div_ceil(32), 1, 1],
                input_buffers: vec![src],
                output_buffer: dst,
                extra_outputs: vec![],
                params: vec![rows, cols, 0, 0],
                ..Default::default()
            });
            return;
        }
        let bytes = (splits as usize)
            .checked_mul(cols as usize)
            .and_then(|n| n.checked_mul(4))
            .expect("row-split reduction size");
        let partial = self.alloc_buffer(bytes);
        self.plan.dispatches.push(Dispatch {
            shader: ShaderEntry::SumRows,
            workgroups: [cols.div_ceil(32), splits, 1],
            input_buffers: vec![src],
            output_buffer: partial,
            extra_outputs: vec![],
            params: vec![rows, cols, 0, splits],
            ..Default::default()
        });
        self.plan.dispatches.push(Dispatch {
            shader: ShaderEntry::SumRows,
            workgroups: [cols.div_ceil(32), 1, 1],
            input_buffers: vec![partial],
            output_buffer: dst,
            extra_outputs: vec![],
            params: vec![splits, cols, 0, 0],
            ..Default::default()
        });
    }

    /// Buffer already allocated for a `CrossEntropyLogitsGrad` on the same
    /// `(logits, labels)` pair. Present only on training graphs.
    /// The (mean, inv_std) buffer for GroupNorm backward over `input`,
    /// emitting its dispatch the first time either gradient asks for it.
    pub(super) fn group_norm_grad_stats(
        &mut self,
        input: NodeId,
        [batch, channels, spatial, num_groups]: [u32; 4],
        eps: f32,
    ) -> BufferRef {
        let key = (input, eps.to_bits());
        if let Some(&stats) = self.group_norm_grad_stats.get(&key) {
            return stats;
        }
        let slots = (batch * num_groups) as usize;
        let stats = self.alloc_buffer(slots * 2 * 4);
        let x = self.get_buffer(input);
        self.plan.dispatches.push(Dispatch {
            shader: ShaderEntry::GroupNormGradStats,
            workgroups: [batch * num_groups, 1, 1],
            input_buffers: vec![x],
            output_buffer: stats,
            extra_outputs: vec![],
            params: vec![batch, channels, spatial, num_groups, eps.to_bits(), 0, 0, 0],
            ..Default::default()
        });
        self.group_norm_grad_stats.insert(key, stats);
        stats
    }

    pub(super) fn ce_logits_grad_buffer(
        &self,
        logits: NodeId,
        labels: NodeId,
    ) -> Option<BufferRef> {
        self.graph.nodes().iter().find_map(|node| {
            if matches!(node.op, Op::CrossEntropyLogitsGrad)
                && node.inputs.first().copied() == Some(logits)
                && node.inputs.get(1).copied() == Some(labels)
            {
                Some(self.get_buffer(node.id))
            } else {
                None
            }
        })
    }

    /// Choose between FlashAttention (BQ>1) and MultiHeadAttn (BQ=1) for
    /// a forward attention dispatch. Returns (shader_entry, workgroups_x).
    pub(super) fn attention_dispatch(
        &self,
        q_seq: u32,
        head_dim: u32,
        num_heads: u32,
        requires_full_precision: bool,
    ) -> (ShaderEntry, [u32; 3]) {
        // Pick the coop-matrix flash forward when the GPU has the
        // 16x16 f16 cooperative_matrix path (NVIDIA, RDNA3, Xe-HPG)
        // and the shape is compatible. ~3.2x faster per dispatch than
        // the scalar kernel on Blackwell. The env var
        // `MEGANEURA_FLASH_FWD_COOP=0` opts back to scalar (regression
        // escape hatch).
        let coop_disabled = !self.options.flash_forward_coop;
        if !coop_disabled
            && !requires_full_precision
            && self.coop_caps.supports_16x16_f16()
            && head_dim >= 16
            && head_dim.is_multiple_of(16)
            // Only power-of-two widths have run on cooperative hardware.
            && head_dim.is_power_of_two()
            && q_seq >= 16
            && crate::codegen::attention_coop_shared_bytes(ShaderGroup::FlashAttentionCoop, head_dim)
                <= u64::from(self.shared_memory_bytes)
        {
            return (
                ShaderEntry::FlashAttentionCoop,
                [q_seq.div_ceil(16), num_heads, 1],
            );
        }
        // The lane split must match forward codegen.
        let (_, tpq) = crate::codegen::attention_lanes(head_dim, self.options.knobs.flash_ept_cap);
        let bq = (self.options.knobs.flash.threads / tpq).max(1);
        if bq >= 2 && q_seq >= bq {
            (
                ShaderEntry::FlashAttention,
                [q_seq.div_ceil(bq), num_heads, 1],
            )
        } else {
            (ShaderEntry::MultiHeadAttn, [q_seq, num_heads, 1])
        }
    }

    /// The attention backward kernel for `part` of `node`, its workgroups
    /// and the scalar paths' elements per thread. A schedule's per-node cap
    /// pins a scalar path; otherwise the family picks the preferred kernel
    /// the target admits.
    pub(super) fn attention_grad_kernel(
        &self,
        node: &Node,
        part: Part,
        [rows, other_rows]: [u32; 2],
        heads: u32,
        head_dim: u32,
    ) -> (AttentionGrad, [u32; 3], u32) {
        let knobs = &self.options.knobs;
        let default_ept = match part {
            Part::Q => knobs.flash_grad_q_ept_cap,
            Part::KV => knobs.flash_grad_kv_ept_cap,
        };
        let problem = attention_grad::Problem {
            part,
            rows,
            other_rows,
            heads,
            head_dim,
            ept_cap: node.attention_ept_cap.unwrap_or(default_ept),
            pinned: node.attention_ept_cap.is_some(),
        };
        let target = attention_grad::Target {
            cooperative_f16: self.coop_caps.supports_16x16_f16(),
            cooperative_f32: self.coop_caps.f32_tile == 8,
            shared_memory_bytes: self.shared_memory_bytes,
            // Rounding dO to f16 loses small derivatives, so the f16 path is
            // an explicit opt-in rather than following from the device
            // advertising f16 matrices.
            reduced_precision: self.allow_reduced_precision_attention_backward,
            prefer: self.options.prefer_attention_grad,
        };
        let kernel = AttentionGrad::select(&problem, &target);
        (kernel, kernel.workgroups(&problem), problem.ept_cap)
    }

    pub(super) fn get_buffer(&self, node: NodeId) -> BufferRef {
        self.node_buffers[&node]
    }

    pub(super) fn is_f32_unit_constant(&self, node: NodeId) -> bool {
        let node = self.graph.node(node);
        node.ty.dtype == DType::F32
            && matches!(
                node.op,
                Op::Constant { ref data }
                    if !data.is_empty()
                        && data.iter().all(|value| value.to_bits() == 1.0_f32.to_bits())
            )
    }

    pub(super) fn emit_sum_inner(
        &mut self,
        input: BufferRef,
        output: BufferRef,
        rows: u32,
        inner: u32,
    ) {
        self.emit_packed_inner_reduction(
            crate::schedule::ReduceOp::Sum,
            input,
            output,
            rows,
            inner,
        );
    }

    /// `[rows, inner]` → `[rows, 1]`, packing narrow rows into one
    /// workgroup so short reductions do not launch mostly idle lanes.
    pub(super) fn emit_packed_inner_reduction(
        &mut self,
        op: crate::schedule::ReduceOp,
        input: BufferRef,
        output: BufferRef,
        rows: u32,
        inner: u32,
    ) {
        const WORKGROUP_SIZE: u32 = 256;
        let rows_per_workgroup = if inner <= 32 { WORKGROUP_SIZE } else { 1 };
        self.emit_inner_reduction(op, input, output, rows, inner, rows_per_workgroup);
    }

    pub(super) fn emit_sum_inner_with_rows_per_workgroup(
        &mut self,
        input: BufferRef,
        output: BufferRef,
        rows: u32,
        inner: u32,
        rows_per_workgroup: u32,
    ) {
        self.emit_inner_reduction(
            crate::schedule::ReduceOp::Sum,
            input,
            output,
            rows,
            inner,
            rows_per_workgroup,
        );
    }

    /// `[rows, inner]` → `[rows, 1]` with `op` combining each row.
    pub(super) fn emit_inner_reduction(
        &mut self,
        op: crate::schedule::ReduceOp,
        input: BufferRef,
        output: BufferRef,
        rows: u32,
        inner: u32,
        rows_per_workgroup: u32,
    ) {
        const WORKGROUP_SIZE: u32 = 256;
        let kernel = ReductionKernel {
            op,
            prologue: PointwiseDAG {
                n_inputs: 1,
                ops: vec![Pw::LoadInput(0)],
                output: 0,
            },
            extra_prologues: vec![],
            epilogue: None,
            n_per_elem: 1,
            n_per_row: 0,
            workgroup_size: WORKGROUP_SIZE,
            rows_per_workgroup,
            gather_elem: Vec::new(),
            input_row_repeats: Vec::new(),
        };
        self.plan.dispatches.push(Dispatch {
            shader: ShaderEntry::Generated,
            workgroups: [rows.div_ceil(rows_per_workgroup), 1, 1],
            input_buffers: vec![input],
            output_buffer: output,
            extra_outputs: vec![],
            params: vec![rows, inner, 1.0_f32.to_bits(), 0],

            kernel: Kernel::Reduction(kernel),
            ..Default::default()
        });
    }

    /// Sum or mean of a whole tensor into `output[0]`.
    ///
    /// One workgroup streams up to 32K elements; beyond that, up to 256
    /// workgroups write partial sums of grid-strided slices and one more
    /// finishes them (and divides, for a mean), so a large loss is not
    /// limited to a single workgroup.
    pub(super) fn emit_reduce_all(
        &mut self,
        shader: ShaderEntry,
        input: BufferRef,
        output: BufferRef,
        len: u32,
    ) {
        const PER_WORKGROUP: u32 = 16 * 1024;
        let groups = (len / PER_WORKGROUP).clamp(1, 256);
        if groups == 1 {
            self.plan.dispatches.push(Dispatch {
                shader,
                workgroups: [1, 1, 1],
                input_buffers: vec![input],
                output_buffer: output,
                extra_outputs: vec![],
                params: vec![len, 0, 0, 0],

                ..Default::default()
            });
            return;
        }
        let partials = self.alloc_buffer(groups as usize * 4);
        self.plan.dispatches.push(Dispatch {
            shader: shader.clone(),
            workgroups: [groups, 1, 1],
            input_buffers: vec![input],
            output_buffer: partials,
            extra_outputs: vec![],
            params: vec![len, 0, 0, 0],
            ..Default::default()
        });
        self.plan.dispatches.push(Dispatch {
            shader,
            workgroups: [1, 1, 1],
            input_buffers: vec![partials],
            output_buffer: output,
            extra_outputs: vec![],
            params: vec![groups, len, 0, 0],
            ..Default::default()
        });
    }

    /// D[row] = dot(d_out[row], o[row]) over `head_dim`-wide rows, for the
    /// attention backward kernels. Sharing D keeps dQ and dK/dV ready at the
    /// same dependency level, preserving overlap and projection-gradient batching.
    pub(super) fn emit_attention_row_dot(
        &mut self,
        d_out: BufferRef,
        o: BufferRef,
        rows: u32,
        head_dim: u32,
    ) -> BufferRef {
        use crate::schedule::ReduceOp;

        let key = (d_out, o, rows, head_dim);
        if let Some(&row_dot) = self.attention_row_dots.get(&key) {
            return row_dot;
        }
        const WORKGROUP_SIZE: u32 = 256;
        let lanes = head_dim.next_power_of_two().clamp(2, WORKGROUP_SIZE);
        let row_dot = self.alloc_buffer(rows as usize * 4);
        self.plan.dispatches.push(Dispatch {
            shader: ShaderEntry::Generated,
            workgroups: [rows.div_ceil(WORKGROUP_SIZE / lanes), 1, 1],
            input_buffers: vec![d_out, o],
            output_buffer: row_dot,
            extra_outputs: vec![],
            params: vec![rows, head_dim, 1.0_f32.to_bits(), 0],
            kernel: Kernel::Reduction(ReductionKernel {
                op: ReduceOp::Sum,
                prologue: PointwiseDAG {
                    n_inputs: 2,
                    ops: vec![Pw::LoadInput(0), Pw::LoadInput(1), Pw::Mul(0, 1)],
                    output: 2,
                },
                extra_prologues: vec![],
                epilogue: None,
                n_per_elem: 2,
                n_per_row: 0,
                workgroup_size: WORKGROUP_SIZE,
                rows_per_workgroup: WORKGROUP_SIZE / lanes,
                gather_elem: Vec::new(),
                input_row_repeats: Vec::new(),
            }),
            ..Default::default()
        });
        self.attention_row_dots.insert(key, row_dot);
        row_dot
    }

    /// Copy into `output`, of extents `dims` (rank 4, leading unit axes
    /// for lower ranks), the input element at `Σ index · strides`: a
    /// permutation, or a broadcast where a stride is zero.
    pub(super) fn emit_strided_copy(
        &mut self,
        input: BufferRef,
        output: BufferRef,
        dims: [u32; 4],
        strides: [u32; 4],
    ) {
        let total = dims
            .iter()
            .try_fold(1u32, |n, &d| n.checked_mul(d))
            .expect("strided copy size exceeds u32");
        self.plan.dispatches.push(Dispatch {
            shader: ShaderEntry::Permute,
            workgroups: linear_grid(total.div_ceil(256)),
            input_buffers: vec![input],
            output_buffer: output,
            params: [&[total, dims[1], dims[2], dims[3]][..], &strides].concat(),
            ..Default::default()
        });
    }

    pub(super) fn emit_broadcast_inner(
        &mut self,
        input: BufferRef,
        output: BufferRef,
        rows: u32,
        inner: u32,
    ) {
        let total = rows
            .checked_mul(inner)
            .expect("inner broadcast element count exceeds u32");
        self.plan.dispatches.push(Dispatch {
            shader: ShaderEntry::GlobalAvgPoolGrad,
            workgroups: [total.div_ceil(256), 1, 1],
            input_buffers: vec![input],
            output_buffer: output,
            extra_outputs: vec![],
            params: vec![total, inner, 1, 0],

            ..Default::default()
        });
    }

    pub(super) fn compile(&mut self) {
        // First pass: allocate buffers for all nodes
        for node in self.graph.nodes() {
            // Cache outputs must alias before allocating views of the result.
            if matches!(node.op, Op::CacheWrite | Op::CacheWritePrefix) {
                let cache = self.get_buffer(node.inputs[1]);
                self.node_buffers.insert(node.id, cache);
                continue;
            }
            // Identity and StopGradient are zero-cost: alias the input
            // buffer. (Identity may also reshape; StopGradient is forward-
            // identity with backward zero, handled in autodiff.)
            if matches!(node.op, Op::Identity | Op::StopGradient) && !node.inputs.is_empty() {
                if let Some(&input_buf) = self.node_buffers.get(&node.inputs[0]) {
                    self.node_buffers.insert(node.id, input_buf);
                    continue;
                }
            }
            let buf = self.alloc_buffer(node.ty.size_bytes());
            self.node_buffers.insert(node.id, buf);

            match node.op {
                Op::Parameter { ref name } => {
                    self.plan.param_buffers.push((name.clone(), buf));
                    self.plan.param_types.insert(buf, node.ty.clone());
                    let wf = WeightFormat::from_dtype(node.ty.dtype);
                    if wf.uses_reduced_storage() {
                        let shape = &node.ty.shape;
                        let (rows, cols) = if shape.len() >= 2 {
                            (shape[0], shape[1])
                        } else {
                            (shape[0], 1)
                        };
                        self.plan.weight_buffers.insert(buf, (wf, rows, cols));
                    }
                }
                Op::Input { ref name } => {
                    self.plan.input_buffers.push((name.clone(), buf));
                }
                Op::Constant { .. } => {}
                Op::MultiHeadAttnGradV { fwd_node, .. } => {
                    self.pending_grad_v_buffers.insert(fwd_node, buf);
                }
                Op::MultiHeadAttn { num_heads, .. }
                | Op::CausalAttention { num_heads, .. }
                | Op::CausalAttentionRoPE { num_heads, .. }
                | Op::SlidingWindowAttention { num_heads, .. }
                | Op::FullAttention { num_heads, .. }
                | Op::CrossAttention { num_heads, .. } => {
                    let q_seq = node.ty.shape[0];
                    // LSE buffer: [0..q_seq*num_heads*2): LSE data (max_score, log_sum_exp per pos×head)
                    let lse_part = q_seq * num_heads as usize * 2;
                    let lse_buf = self.alloc_buffer(lse_part * 4);
                    self.plan.lse_buffers.push((node.id, lse_buf));
                }
                _ => {}
            }
        }

        // Second pass: emit dispatches in topological order.
        // The optimizer may create new nodes at high IDs that are referenced
        // by existing nodes at lower IDs (e.g. SwiGLU concat fusion creates
        // a new MatMul at the end, referenced by the original SwiGLU node).
        // Processing in ID order would dispatch consumers before producers.
        let topo = topological_order(self.graph);
        for &node_id in &topo {
            let first_new = self.plan.dispatches.len();
            self.compile_node(&self.graph.nodes()[node_id as usize]);
            // Stamp provenance on every dispatch this node emitted.
            for d in &mut self.plan.dispatches[first_new..] {
                d.origin.push(node_id);
            }
        }

        // Record the node → buffer map and node names for debug readback.
        let mut node_buffers: Vec<(NodeId, BufferRef)> =
            self.node_buffers.iter().map(|(&n, &b)| (n, b)).collect();
        node_buffers.sort_unstable_by_key(|&(n, _)| n);
        self.plan.node_buffers = node_buffers;
        self.plan.node_names = self
            .graph
            .nodes()
            .iter()
            .filter_map(|n| n.name.clone().map(|name| (n.id, name)))
            .collect();

        // Generate labels for profiling
        for d in &mut self.plan.dispatches {
            d.label = match d.shader {
                // A generated kernel is named after the op that produced it.
                ShaderEntry::Generated => {
                    let op = d.origin.first().map_or_else(
                        || "Generated".to_string(),
                        |&n| {
                            let op = format!("{:?}", self.graph.node(n).op);
                            op.split(|c: char| !c.is_ascii_alphanumeric())
                                .next()
                                .unwrap_or_default()
                                .to_string()
                        },
                    );
                    format!("{op}[{}x{}]", d.params[0], d.params[1])
                }
                ShaderEntry::BlockMatMul
                | ShaderEntry::BlockMatMulAT
                | ShaderEntry::BlockMatMulBT
                | ShaderEntry::BatchMatMul
                | ShaderEntry::BatchMatMulAT
                | ShaderEntry::BatchMatMulBT => {
                    format!(
                        "{:?}[g={},{}x{}x{}]",
                        d.shader, d.params[3], d.params[0], d.params[1], d.params[2]
                    )
                }
                ShaderEntry::MatMul
                | ShaderEntry::FusedMatMulAdd
                | ShaderEntry::MatMulGemv
                | ShaderEntry::MatMulGemvAdd => {
                    format!(
                        "{:?}[{}x{}x{}]",
                        d.shader, d.params[0], d.params[2], d.params[1]
                    )
                }
                ShaderEntry::MatMulAT
                | ShaderEntry::MatMulBT
                | ShaderEntry::MatMulGemvBT
                | ShaderEntry::MatMulGemvBTAdd
                | ShaderEntry::FusedMatMulATAdd
                | ShaderEntry::FusedMatMulBTAdd => {
                    format!(
                        "{:?}[{}x{}x{}]",
                        d.shader, d.params[0], d.params[1], d.params[2]
                    )
                }
                ShaderEntry::MultiHeadAttn
                | ShaderEntry::AttentionGrad(AttentionGrad {
                    path: attention_grad::Path::Rowwise,
                    ..
                }) => {
                    let nh = d.params[2] >> 16;
                    let nkv = d.params[2] & 0xFFFF;
                    format!(
                        "{:?}[q={},kv={},h={}/{}]",
                        d.shader, d.params[0], d.params[1], nh, nkv
                    )
                }
                ShaderEntry::RmsNormGradW
                | ShaderEntry::RmsNormGradWRowPar
                | ShaderEntry::RmsNormGradX
                | ShaderEntry::LayerNormGradWB
                | ShaderEntry::LayerNormGradX => {
                    format!("{:?}[{}x{}]", d.shader, d.params[0], d.params[1])
                }
                _ => {
                    if d.params[0] > 0 {
                        format!("{:?}[{}]", d.shader, d.params[0])
                    } else {
                        format!("{:?}", d.shader)
                    }
                }
            };
            // Prefix the graph-level name when the originating node has one,
            // so profiler rows and dumps read "blk3.mlp.gate: MatMul[...]"
            // instead of a dozen identical "MatMul[...]" entries.
            if let Some(name) = d.origin.iter().find_map(|&n| self.graph.node_name(n)) {
                d.label = format!("{name}: {}", d.label);
            }
        }

        // Outputs layout (from autodiff): [user_outputs..., param_grads...]
        // Where `param_grads` is the last `num_param_grad_outputs()` entries
        // (one per Parameter node, exactly aligned with param_buffers order).
        // For inference (no autodiff), num_param_grad_outputs() == 0 and every
        // output is user-facing.
        let outputs = self.graph.outputs();
        let num_grads = self.graph.num_param_grad_outputs();
        let num_user = outputs.len() - num_grads;

        // Collect user-facing output buffers (accessible via read_output_by_index).
        for &out_id in &outputs[..num_user] {
            self.plan.output_buffers.push(self.get_buffer(out_id));
        }
        if let Some(&loss_id) = outputs.first() {
            self.plan.loss_buffer = Some(self.get_buffer(loss_id));
        }

        // Build param→grad pairs from the trailing grad outputs.
        if num_grads > 0 {
            let param_nodes: Vec<&Node> = self
                .graph
                .nodes()
                .iter()
                .filter(|node| matches!(node.op, Op::Parameter { .. }))
                .collect();
            assert_eq!(
                param_nodes.len(),
                num_grads,
                "autodiff must emit one grad output per Parameter",
            );
            for i in 0..num_grads {
                let grad_node = self.graph.node(outputs[num_user + i]);
                if grad_node.ty.num_elements() != param_nodes[i].ty.num_elements() {
                    // Autodiff uses a scalar zero as the positional placeholder
                    // for a parameter that receives no gradient (for example,
                    // one behind StopGradient). Do not expose that sentinel as
                    // a real gradient: optimizers iterate over the parameter's
                    // full length and would otherwise read past the scalar.
                    assert!(
                        matches!(grad_node.op, Op::Constant { ref data } if data == &[0.0]),
                        "parameter gradient shape mismatch without a zero placeholder",
                    );
                    continue;
                }
                let param_buf = self.plan.param_buffers[i].1;
                let grad_buf = self.get_buffer(outputs[num_user + i]);
                self.plan.param_grad_pairs.push((param_buf, grad_buf));
            }
        }
    }

    pub(super) fn compile_node(&mut self, node: &Node) {
        let out_buf = self.get_buffer(node.id);
        let dispatch_start = self.plan.dispatches.len();

        match node.op {
            // Leaf nodes and dead nodes: no dispatch needed
            Op::Input { .. }
            | Op::Parameter { .. }
            | Op::Constant { .. }
            | Op::Nop
            | Op::Identity
            | Op::StopGradient
            | Op::CrossEntropyLogitsGrad => {}

            Op::Materialize => {
                let input = self.get_buffer(node.inputs[0]);
                let len = node.ty.num_elements() as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::Generated,
                    workgroups: [len.div_ceil(256), 1, 1],
                    input_buffers: vec![input],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![len, 0, 0, 0],
                    kernel: Kernel::Pointwise(PointwiseDAG {
                        n_inputs: 1,
                        ops: vec![Pw::LoadInput(0)],
                        output: 0,
                    }),
                    fusion_barrier: true,
                    ..Default::default()
                });
            }

            Op::MatMul => {
                let a = self.get_buffer(node.inputs[0]);
                let b = self.get_buffer(node.inputs[1]);
                let a_shape = &self.graph.node(node.inputs[0]).ty.shape;
                let b_shape = &self.graph.node(node.inputs[1]).ty.shape;
                let wf = WeightFormat::from_dtype(self.graph.node(node.inputs[1]).ty.dtype);
                let m = a_shape[0] as u32;
                let k = a_shape[1] as u32;
                let n = b_shape[1] as u32;
                if n == 1 && k <= 32 && self.is_f32_unit_constant(node.inputs[1]) {
                    // A narrow multiplication by an all-ones column is a
                    // scalar-order row sum. Keep MatMul in the graph so its
                    // autodiff topology and accumulation order are unchanged;
                    // specialize only the physical forward dispatch.
                    self.emit_sum_inner(a, out_buf, m, k);
                } else if m == 1 && n.is_multiple_of(4) {
                    // K-split GEMV, optionally grouping adjacent output vec4s
                    // within a workgroup for contiguous weight reads.
                    let mut dispatch = Dispatch {
                        shader: ShaderEntry::MatMulGemv,
                        input_buffers: vec![a, b],
                        output_buffer: out_buf,
                        extra_outputs: vec![],
                        params: vec![m, k, n, 0],

                        weight_format: wf,
                        kernel: self.options.gemv_kernel(ShaderGroup::MatMulGemv, wf),
                        ..Default::default()
                    };
                    dispatch.workgroups = gemv_workgroups(n, dispatch.gemv_shape().unwrap());
                    self.plan.dispatches.push(dispatch);
                } else {
                    self.plan.dispatches.push(Dispatch {
                        shader: ShaderEntry::MatMul,
                        workgroups: matmul_workgroups(m, n, 64),
                        input_buffers: vec![a, b],
                        output_buffer: out_buf,
                        extra_outputs: vec![],
                        params: vec![m, k, n, 0],

                        weight_format: wf,
                        ..Default::default()
                    });
                }
            }

            Op::MatMulAT => {
                // C = A^T @ B  (A is [K, M], B is [K, N], C is [M, N])
                let a = self.get_buffer(node.inputs[0]);
                let b = self.get_buffer(node.inputs[1]);
                let a_shape = &self.graph.node(node.inputs[0]).ty.shape;
                let b_shape = &self.graph.node(node.inputs[1]).ty.shape;
                let wf = WeightFormat::from_dtype(self.graph.node(node.inputs[1]).ty.dtype);
                let k = a_shape[0] as u32; // A is [K, M]
                let m = a_shape[1] as u32;
                let n = b_shape[1] as u32; // B is [K, N]
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::MatMulAT,
                    workgroups: matmul_workgroups(m, n, 64),
                    input_buffers: vec![a, b],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![m, n, k, 0],

                    weight_format: wf,
                    ..Default::default()
                });
            }

            Op::MatMulBT => {
                // C = A @ B^T  (A is [M, K], B is [N, K], C is [M, N])
                let a = self.get_buffer(node.inputs[0]);
                let b = self.get_buffer(node.inputs[1]);
                // Block-quantized weights pack along the parameter's first
                // dimension, which is N for a transposed B, while every
                // packed decoder indexes along K. There is no correct
                // reading of a block format on this op, so refuse at
                // compile time rather than emit a kernel that returns
                // plausible numbers. f16 is unaffected: it is an
                // elementwise cast at the same index, not a block layout.
                assert!(
                    !WeightFormat::from_dtype(self.graph.node(node.inputs[1]).ty.dtype)
                        .is_quantized(),
                    "matmul_bt does not support block-quantized weights: their blocks \
                     run along the parameter's first dimension, which is N here, not K"
                );
                let a_shape = &self.graph.node(node.inputs[0]).ty.shape;
                let b_shape = &self.graph.node(node.inputs[1]).ty.shape;
                let wf = WeightFormat::from_dtype(self.graph.node(node.inputs[1]).ty.dtype);
                let m = a_shape[0] as u32; // A is [M, K]
                let k = a_shape[1] as u32;
                let n = b_shape[0] as u32; // B is [N, K]
                if k == 1 && self.is_f32_unit_constant(node.inputs[1]) {
                    // MatMul's input gradient against a unit column is an
                    // exact row broadcast. Preserve the MatMulBT graph node
                    // and its dependency order while avoiding tiled GEMM.
                    self.emit_broadcast_inner(a, out_buf, m, n);
                } else if m == 1 && k.is_multiple_of(4) {
                    // Only f32 and f16 reach here; the assert above turned
                    // every block format away, so the K-split GEMV-BT never
                    // sees packed data.
                    self.plan.dispatches.push(Dispatch {
                        shader: ShaderEntry::MatMulGemvBT,
                        workgroups: row_gemv_workgroups(
                            n.div_ceil(self.options.gemv_shape.map_or(1, |s| s.bt_rows)),
                        ),
                        input_buffers: vec![a, b],
                        output_buffer: out_buf,
                        extra_outputs: vec![],
                        params: vec![m, n, k, 0],

                        weight_format: wf,
                        kernel: self.options.gemv_kernel(ShaderGroup::MatMulGemvBT, wf),
                        ..Default::default()
                    });
                } else {
                    self.plan.dispatches.push(Dispatch {
                        shader: ShaderEntry::MatMulBT,
                        workgroups: matmul_workgroups(m, n, 64),
                        input_buffers: vec![a, b],
                        output_buffer: out_buf,
                        extra_outputs: vec![],
                        params: vec![m, n, k, 0],

                        weight_format: wf,
                        ..Default::default()
                    });
                }
            }

            Op::BlockMatMul
            | Op::BlockMatMulAT { .. }
            | Op::BlockMatMulBT
            | Op::BatchMatMul
            | Op::BatchMatMulAT
            | Op::BatchMatMulBT => {
                let a = self.get_buffer(node.inputs[0]);
                let b = self.get_buffer(node.inputs[1]);
                let a_ty = &self.graph.node(node.inputs[0]).ty;
                let b_ty = &self.graph.node(node.inputs[1]).ty;
                assert_eq!(a_ty.dtype, crate::graph::DType::F32);
                assert_eq!(b_ty.dtype, crate::graph::DType::F32);
                for ty in [a_ty, b_ty, &node.ty] {
                    let elements = ty
                        .shape
                        .iter()
                        .try_fold(1usize, |n, &d| n.checked_mul(d))
                        .expect("block tensor size overflow");
                    assert!(
                        elements <= u32::MAX as usize,
                        "block tensor exceeds shader index range"
                    );
                }
                let (shader, groups, m, n, k) = match node.op {
                    Op::BlockMatMul => (
                        ShaderEntry::BlockMatMul,
                        b_ty.shape[0],
                        a_ty.shape[0],
                        b_ty.shape[2],
                        b_ty.shape[1],
                    ),
                    Op::BlockMatMulAT { groups } => (
                        ShaderEntry::BlockMatMulAT,
                        groups,
                        a_ty.shape[1] / groups,
                        b_ty.shape[1] / groups,
                        a_ty.shape[0],
                    ),
                    Op::BlockMatMulBT => (
                        ShaderEntry::BlockMatMulBT,
                        b_ty.shape[0],
                        a_ty.shape[0],
                        b_ty.shape[1],
                        b_ty.shape[2],
                    ),
                    // The output is `[B, M, N]` and the reduction runs along
                    // the side of `a` that does not land in the output.
                    Op::BatchMatMul => (
                        ShaderEntry::BatchMatMul,
                        node.ty.shape[0],
                        node.ty.shape[1],
                        node.ty.shape[2],
                        a_ty.shape[2],
                    ),
                    Op::BatchMatMulAT => (
                        ShaderEntry::BatchMatMulAT,
                        node.ty.shape[0],
                        node.ty.shape[1],
                        node.ty.shape[2],
                        a_ty.shape[1],
                    ),
                    Op::BatchMatMulBT => (
                        ShaderEntry::BatchMatMulBT,
                        node.ty.shape[0],
                        node.ty.shape[1],
                        node.ty.shape[2],
                        a_ty.shape[2],
                    ),
                    _ => unreachable!(),
                };
                let [m, n, k, groups] = [m, n, k, groups].map(|d| u32::try_from(d).unwrap());
                assert!(
                    groups <= 65535,
                    "block count exceeds portable dispatch limit"
                );
                // Match the ordinary scalar per-block tile choice. This
                // operator does not opt into GEMV or cooperative variants.
                let small = m.div_ceil(64) * n.div_ceil(64) < 16;
                let tile = if small { 32 } else { 64 };
                let workgroups = [n.div_ceil(tile), m.div_ceil(tile), groups];
                assert!(workgroups.iter().all(|&v| v <= 65535));
                self.plan.dispatches.push(Dispatch {
                    shader,
                    workgroups,
                    input_buffers: vec![a, b],
                    output_buffer: out_buf,
                    params: vec![m, n, k, groups],
                    kernel: if small {
                        Kernel::SmallTile
                    } else {
                        Kernel::Default
                    },
                    ..Default::default()
                });
            }

            Op::FusedMatMulAdd => {
                // C = A × B + D (inputs: [a, b, d])
                let a = self.get_buffer(node.inputs[0]);
                let b = self.get_buffer(node.inputs[1]);
                let d = self.get_buffer(node.inputs[2]);
                let a_shape = &self.graph.node(node.inputs[0]).ty.shape;
                let b_shape = &self.graph.node(node.inputs[1]).ty.shape;
                let wf = WeightFormat::from_dtype(self.graph.node(node.inputs[1]).ty.dtype);
                let m = a_shape[0] as u32;
                let k = a_shape[1] as u32;
                let n = b_shape[1] as u32;
                if m == 1 && n.is_multiple_of(4) {
                    let mut dispatch = Dispatch {
                        shader: ShaderEntry::MatMulGemvAdd,
                        input_buffers: vec![a, b, d],
                        output_buffer: out_buf,
                        extra_outputs: vec![],
                        params: vec![m, k, n, 0],

                        weight_format: wf,
                        kernel: self.options.gemv_kernel(ShaderGroup::MatMulGemvAdd, wf),
                        ..Default::default()
                    };
                    dispatch.workgroups = gemv_workgroups(n, dispatch.gemv_shape().unwrap());
                    self.plan.dispatches.push(dispatch);
                } else {
                    self.plan.dispatches.push(Dispatch {
                        shader: ShaderEntry::FusedMatMulAdd,
                        workgroups: matmul_workgroups(m, n, 64),
                        input_buffers: vec![a, b, d],
                        output_buffer: out_buf,
                        extra_outputs: vec![],
                        params: vec![m, k, n, 0],

                        weight_format: wf,
                        ..Default::default()
                    });
                }
            }

            Op::FusedMatMulATAdd => {
                // C = A^T × B + D (inputs: [a, b, d])
                let a = self.get_buffer(node.inputs[0]);
                let b = self.get_buffer(node.inputs[1]);
                let d = self.get_buffer(node.inputs[2]);
                let a_shape = &self.graph.node(node.inputs[0]).ty.shape;
                let b_shape = &self.graph.node(node.inputs[1]).ty.shape;
                let wf = WeightFormat::from_dtype(self.graph.node(node.inputs[1]).ty.dtype);
                let k = a_shape[0] as u32;
                let m = a_shape[1] as u32;
                let n = b_shape[1] as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::FusedMatMulATAdd,
                    workgroups: matmul_workgroups(m, n, 64),
                    input_buffers: vec![a, b, d],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![m, n, k, 0],

                    weight_format: wf,
                    ..Default::default()
                });
            }

            Op::FusedMatMulBTAdd => {
                // C = A × B^T + D (inputs: [a, b, d])
                let a = self.get_buffer(node.inputs[0]);
                let b = self.get_buffer(node.inputs[1]);
                let d = self.get_buffer(node.inputs[2]);
                // Same block-axis mismatch as `Op::MatMulBT`. Graph rewrites
                // can fuse `Add(MatMulBT, ?)` into this op before compile,
                // so the unfused assert would never see a quantized B.
                assert!(
                    !WeightFormat::from_dtype(self.graph.node(node.inputs[1]).ty.dtype)
                        .is_quantized(),
                    "matmul_bt does not support block-quantized weights: their blocks \
                     run along the parameter's first dimension, which is N here, not K"
                );
                let a_shape = &self.graph.node(node.inputs[0]).ty.shape;
                let b_shape = &self.graph.node(node.inputs[1]).ty.shape;
                let wf = WeightFormat::from_dtype(self.graph.node(node.inputs[1]).ty.dtype);
                let m = a_shape[0] as u32;
                let k = a_shape[1] as u32;
                let n = b_shape[0] as u32;
                let gemv = m == 1 && k.is_multiple_of(4);
                self.plan.dispatches.push(Dispatch {
                    shader: if gemv {
                        ShaderEntry::MatMulGemvBTAdd
                    } else {
                        ShaderEntry::FusedMatMulBTAdd
                    },
                    workgroups: if gemv {
                        row_gemv_workgroups(
                            n.div_ceil(self.options.gemv_shape.map_or(1, |s| s.bt_rows)),
                        )
                    } else {
                        matmul_workgroups(m, n, 64)
                    },
                    input_buffers: vec![a, b, d],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![m, n, k, 0],

                    weight_format: wf,
                    kernel: if gemv {
                        self.options.gemv_kernel(ShaderGroup::MatMulGemvBTAdd, wf)
                    } else {
                        Kernel::Default
                    },
                    ..Default::default()
                });
            }

            Op::Add => {
                self.emit_pointwise(pointwise(2, [Pw::Add(0, 1)]), node, out_buf);
            }
            Op::Mul => {
                self.emit_pointwise(pointwise(2, [Pw::Mul(0, 1)]), node, out_buf);
            }
            Op::Greater => {
                self.emit_pointwise(pointwise(2, [Pw::Greater(0, 1)]), node, out_buf);
            }

            Op::BiasAdd => self.emit_row_broadcast(crate::schedule::Pw::Add(0, 1), node, out_buf),

            Op::BiasMul => self.emit_row_broadcast(crate::schedule::Pw::Mul(0, 1), node, out_buf),

            Op::Relu => {
                self.emit_pointwise(pointwise(1, [Pw::Relu(0)]), node, out_buf);
            }
            Op::Sigmoid => {
                self.emit_pointwise(pointwise(1, [Pw::Sigmoid(0)]), node, out_buf);
            }
            Op::Tanh => {
                self.emit_pointwise(pointwise(1, [Pw::Tanh(0)]), node, out_buf);
            }
            Op::Neg => {
                self.emit_pointwise(pointwise(1, [Pw::Neg(0)]), node, out_buf);
            }
            Op::Abs => {
                self.emit_pointwise(pointwise(1, [Pw::Abs(0)]), node, out_buf);
            }
            Op::Log => {
                self.emit_pointwise(pointwise(1, [Pw::Log(0)]), node, out_buf);
            }
            Op::Recip => {
                self.emit_pointwise(pointwise(1, [Pw::Recip(0)]), node, out_buf);
            }
            Op::Exp => {
                self.emit_generated_unary(Pw::Exp(0), node, out_buf);
            }
            Op::Erf => {
                self.emit_generated_unary(Pw::Erf(0), node, out_buf);
            }
            Op::Sin => {
                self.emit_generated_unary(Pw::Sin(0), node, out_buf);
            }
            Op::Cos => {
                self.emit_generated_unary(Pw::Cos(0), node, out_buf);
            }
            Op::ToF32 => {
                self.emit_generated_unary(Pw::U32ToF32(0), node, out_buf);
            }
            Op::ToU32 => {
                self.emit_generated_unary(Pw::F32ToU32(0), node, out_buf);
            }
            Op::Sqrt => {
                self.emit_generated_unary(Pw::Sqrt(0), node, out_buf);
            }
            Op::Rsqrt => {
                self.emit_generated_unary(Pw::Rsqrt(0), node, out_buf);
            }
            Op::Offset { value } => {
                let dag = PointwiseDAG {
                    n_inputs: 1,
                    ops: vec![Pw::LoadInput(0), Pw::const_f32(value), Pw::Add(0, 1)],
                    output: 2,
                };
                self.emit_pointwise(dag, node, out_buf);
            }
            Op::Softplus { beta } => {
                let input = self.get_buffer(node.inputs[0]);
                let len = node.ty.num_elements() as u32;
                let pointwise = softplus::forward(beta);
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::Generated,
                    workgroups: [len.div_ceil(256), 1, 1],
                    input_buffers: vec![input],
                    output_buffer: out_buf,
                    params: vec![len, 0, 0, 0],
                    kernel: Kernel::Pointwise(pointwise),
                    ..Default::default()
                });
            }
            Op::Clamp { min, max } => {
                let input = self.get_buffer(node.inputs[0]);
                let len = node.ty.num_elements() as u32;
                // Use native min/max rather than a ReLU identity. The latter
                // loses every in-range low-magnitude value to cancellation
                // when the bounds are large (for example ±1e10).
                let pointwise = PointwiseDAG {
                    n_inputs: 1,
                    ops: vec![
                        Pw::LoadInput(0),
                        Pw::const_f32(min),
                        Pw::const_f32(max),
                        Pw::Max(0, 1),
                        Pw::Min(3, 2),
                    ],
                    output: 4,
                };
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::Generated,
                    workgroups: [len.div_ceil(256), 1, 1],
                    input_buffers: vec![input],
                    output_buffer: out_buf,
                    params: vec![len, 0, 0, 0],
                    kernel: Kernel::Pointwise(pointwise),
                    ..Default::default()
                });
            }
            Op::Scale { factor } => {
                let input = self.get_buffer(node.inputs[0]);
                let len = node.ty.num_elements() as u32;
                let pointwise = PointwiseDAG {
                    n_inputs: 1,
                    ops: vec![Pw::LoadInput(0), Pw::const_f32(factor), Pw::Mul(0, 1)],
                    output: 2,
                };
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::Generated,
                    workgroups: [len.div_ceil(256), 1, 1],
                    input_buffers: vec![input],
                    output_buffer: out_buf,
                    params: vec![len, 0, 0, 0],
                    kernel: Kernel::Pointwise(pointwise),
                    ..Default::default()
                });
            }
            Op::SoftplusGrad { beta } => {
                let grad_output = self.get_buffer(node.inputs[0]);
                let input = self.get_buffer(node.inputs[1]);
                let len = node.ty.num_elements() as u32;
                let pointwise = softplus::backward(beta);
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::Generated,
                    workgroups: [len.div_ceil(256), 1, 1],
                    input_buffers: vec![grad_output, input],
                    output_buffer: out_buf,
                    params: vec![len, 0, 0, 0],
                    kernel: Kernel::Pointwise(pointwise),
                    ..Default::default()
                });
            }

            Op::SumAll => {
                let input = self.get_buffer(node.inputs[0]);
                let len = self.graph.node(node.inputs[0]).ty.num_elements() as u32;
                self.emit_reduce_all(ShaderEntry::SumAll, input, out_buf, len);
            }

            Op::MeanAll => {
                let input = self.get_buffer(node.inputs[0]);
                let len = self.graph.node(node.inputs[0]).ty.num_elements() as u32;
                self.emit_reduce_all(ShaderEntry::MeanAll, input, out_buf, len);
            }

            Op::SumRows => {
                // [M, N] → [N]. Tall narrow shapes are sliced across extra
                // workgroups; short and wide shapes stay one reduction.
                let input = self.get_buffer(node.inputs[0]);
                let in_shape = &self.graph.node(node.inputs[0]).ty.shape;
                let m = in_shape[0] as u32;
                let n = in_shape[1] as u32;
                self.push_sum_rows(m, n, input, out_buf);
            }

            Op::ExclusiveCumsum { reverse } => {
                // One invocation handles one row. This is deliberately
                // linear in N; the dense-matmul spelling is quadratic.
                let input = self.get_buffer(node.inputs[0]);
                let in_shape = &self.graph.node(node.inputs[0]).ty.shape;
                let m = in_shape[0] as u32;
                let n = in_shape[1] as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::GlobalAvgPoolGrad,
                    workgroups: [m.div_ceil(256), 1, 1],
                    input_buffers: vec![input],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![m, n, 3, u32::from(reverse)],

                    ..Default::default()
                });
            }

            Op::ShiftInner { offset } => {
                let input = self.get_buffer(node.inputs[0]);
                let in_shape = &self.graph.node(node.inputs[0]).ty.shape;
                let m = in_shape[0] as u32;
                let n = in_shape[1] as u32;
                let len = m
                    .checked_mul(n)
                    .expect("shift_inner element count exceeds u32");
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::GlobalAvgPoolGrad,
                    workgroups: [len.div_ceil(256), 1, 1],
                    input_buffers: vec![input],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![len, n, 2, offset as u32],

                    ..Default::default()
                });
            }

            Op::SumInner => {
                // [M, N] → [M, 1]: per-row reduction over the inner axis.
                // Pack narrow rows into one workgroup so SH-width reductions
                // do not launch hundreds of idle lanes per output scalar.
                // Wide reductions keep the existing producer-fusion path.
                // A one-lane packed path retains scalar column-order
                // summation while still allowing pointwise and gather
                // producers to fold into the reduction prologue.
                let input = self.get_buffer(node.inputs[0]);
                let in_shape = &self.graph.node(node.inputs[0]).ty.shape;
                let m = in_shape[0] as u32;
                let n = in_shape[1] as u32;
                self.emit_sum_inner(input, out_buf, m, n);
            }

            Op::MaxInner => {
                let input = self.get_buffer(node.inputs[0]);
                let in_shape = &self.graph.node(node.inputs[0]).ty.shape;
                let (m, n) = (in_shape[0] as u32, in_shape[1] as u32);
                self.emit_packed_inner_reduction(
                    crate::schedule::ReduceOp::Max,
                    input,
                    out_buf,
                    m,
                    n,
                );
            }

            Op::BroadcastInner { inner } => {
                let input = self.get_buffer(node.inputs[0]);
                let rows = node.ty.shape[0] as u32;
                self.emit_broadcast_inner(input, out_buf, rows, inner);
            }

            Op::NormalizeInnerSum { inner, floor } => {
                let input = self.get_buffer(node.inputs[0]);
                let rows = u32::try_from(node.ty.shape[0])
                    .expect("normalize_inner_sum row count exceeds u32");
                let floor = Pw::const_f32(floor);
                let kernel = ReductionKernel {
                    op: crate::schedule::ReduceOp::Sum,
                    prologue: PointwiseDAG {
                        n_inputs: 1,
                        ops: vec![Pw::LoadInput(0)],
                        output: 0,
                    },
                    extra_prologues: vec![],
                    epilogue: Some(ReductionEpilogue {
                        dag: PointwiseDAG {
                            n_inputs: 2,
                            ops: vec![
                                Pw::LoadInput(0),
                                Pw::LoadInput(1),
                                floor,
                                Pw::Sub(1, 2),
                                Pw::Relu(3),
                                Pw::Add(4, 2),
                                Pw::Recip(5),
                                Pw::Mul(0, 6),
                            ],
                            output: 7,
                        },
                        n_per_col_inputs: 0,
                    }),
                    n_per_elem: 1,
                    n_per_row: 0,
                    workgroup_size: 256,
                    rows_per_workgroup: 256,
                    gather_elem: vec![],
                    input_row_repeats: vec![],
                };
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::Generated,
                    workgroups: [rows.div_ceil(256), 1, 1],
                    input_buffers: vec![input],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![rows, inner, 1.0_f32.to_bits(), 0],

                    kernel: Kernel::Reduction(kernel),
                    ..Default::default()
                });
            }

            Op::NormalizeInnerSumGrad { inner, floor } => {
                let grad_output = self.get_buffer(node.inputs[0]);
                let input = self.get_buffer(node.inputs[1]);
                let rows = u32::try_from(node.ty.shape[0])
                    .expect("normalize_inner_sum gradient row count exceeds u32");
                let sum = self.alloc_buffer(rows as usize * std::mem::size_of::<f32>());
                self.emit_sum_inner_with_rows_per_workgroup(input, sum, rows, inner, 256);
                let floor = Pw::const_f32(floor);
                let kernel = ReductionKernel {
                    op: crate::schedule::ReduceOp::Sum,
                    prologue: PointwiseDAG {
                        n_inputs: 3,
                        ops: vec![Pw::LoadInput(0), Pw::LoadInput(1), Pw::Mul(0, 1)],
                        output: 2,
                    },
                    extra_prologues: vec![],
                    epilogue: Some(ReductionEpilogue {
                        dag: PointwiseDAG {
                            n_inputs: 4,
                            ops: vec![
                                Pw::LoadInput(0),
                                Pw::LoadInput(1),
                                Pw::LoadInput(2),
                                Pw::LoadInput(3),
                                floor,
                                Pw::Sub(2, 4),
                                Pw::Relu(5),
                                Pw::Add(6, 4),
                                Pw::Recip(7),
                                Pw::Mul(8, 8),
                                Pw::Neg(9),
                                Pw::Mul(3, 10),
                                Pw::Greater(2, 4),
                                Pw::Mul(11, 12),
                                Pw::Mul(0, 8),
                                Pw::Add(14, 13),
                            ],
                            output: 15,
                        },
                        n_per_col_inputs: 0,
                    }),
                    n_per_elem: 2,
                    n_per_row: 1,
                    workgroup_size: 256,
                    rows_per_workgroup: 256,
                    gather_elem: vec![],
                    input_row_repeats: vec![],
                };
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::Generated,
                    workgroups: [rows.div_ceil(256), 1, 1],
                    input_buffers: vec![grad_output, input, sum],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![rows, inner, 1.0_f32.to_bits(), 0],

                    kernel: Kernel::Reduction(kernel),
                    ..Default::default()
                });
            }

            Op::PairwiseSquaredDistance { pairs } => {
                let left = self.get_buffer(node.inputs[0]);
                let right = self.get_buffer(node.inputs[1]);
                let inner = u32::try_from(self.graph.node(node.inputs[0]).ty.shape[1])
                    .expect("pairwise distance width exceeds u32");
                let total = u32::try_from(node.ty.num_elements())
                    .expect("pairwise distance output element count exceeds u32");
                let kernel = ReductionKernel {
                    op: crate::schedule::ReduceOp::Sum,
                    prologue: PointwiseDAG {
                        n_inputs: 2,
                        ops: vec![
                            Pw::LoadInput(0),
                            Pw::LoadInput(1),
                            Pw::Sub(0, 1),
                            Pw::Mul(2, 2),
                        ],
                        output: 3,
                    },
                    extra_prologues: vec![],
                    epilogue: None,
                    n_per_elem: 2,
                    n_per_row: 0,
                    workgroup_size: 256,
                    rows_per_workgroup: 256,
                    gather_elem: vec![],
                    input_row_repeats: vec![pairs, 1],
                };
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::Generated,
                    workgroups: [total.div_ceil(256), 1, 1],
                    input_buffers: vec![left, right],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![total, inner, 1.0_f32.to_bits(), 0],

                    kernel: Kernel::Reduction(kernel),
                    ..Default::default()
                });
            }

            Op::PairwiseGrad { kind, inner, pairs } => {
                let grad_output = self.get_buffer(node.inputs[0]);
                let first = self.get_buffer(node.inputs[1]);
                let second = self.get_buffer(node.inputs[2]);
                let total = u32::try_from(node.ty.num_elements())
                    .expect("pairwise gradient element count exceeds u32");
                let mode = match kind {
                    PairwiseGradKind::DistanceLeft => 0,
                    PairwiseGradKind::DistanceRight => 1,
                    PairwiseGradKind::RejectionDirections => 2,
                };
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::PairwiseGrad,
                    workgroups: [total.div_ceil(256), 1, 1],
                    input_buffers: vec![grad_output, first, second],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![total, inner, pairs, mode],

                    ..Default::default()
                });
            }

            Op::PairwiseVectorRejection { pairs } => {
                let vectors = self.get_buffer(node.inputs[0]);
                let directions = self.get_buffer(node.inputs[1]);
                let inner = u32::try_from(self.graph.node(node.inputs[1]).ty.shape[1])
                    .expect("pairwise vector width exceeds u32");
                let vector_rows =
                    u32::try_from(node.ty.shape[0]).expect("pairwise vector row count exceeds u32");
                let kernel = ReductionKernel {
                    op: crate::schedule::ReduceOp::Sum,
                    prologue: PointwiseDAG {
                        n_inputs: 2,
                        ops: vec![Pw::LoadInput(0), Pw::LoadInput(1), Pw::Mul(0, 1)],
                        output: 2,
                    },
                    extra_prologues: vec![],
                    epilogue: Some(ReductionEpilogue {
                        dag: PointwiseDAG {
                            n_inputs: 3,
                            ops: vec![
                                Pw::LoadInput(0),
                                Pw::LoadInput(1),
                                Pw::LoadInput(2),
                                Pw::Mul(1, 2),
                                Pw::Sub(0, 3),
                            ],
                            output: 4,
                        },
                        n_per_col_inputs: 0,
                    }),
                    n_per_elem: 2,
                    n_per_row: 0,
                    workgroup_size: 256,
                    rows_per_workgroup: 256,
                    gather_elem: vec![],
                    input_row_repeats: vec![1, pairs],
                };
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::Generated,
                    workgroups: [vector_rows.div_ceil(256), 1, 1],
                    input_buffers: vec![vectors, directions],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![vector_rows, inner, 1.0_f32.to_bits(), 0],

                    kernel: Kernel::Reduction(kernel),
                    ..Default::default()
                });
            }

            Op::Softmax => {
                let input = self.get_buffer(node.inputs[0]);
                let shape = &self.graph.node(node.inputs[0]).ty.shape;
                let batch = shape[0] as u32;
                let features = shape[1] as u32;
                self.emit_softmax_schedule(input, out_buf, batch, features, false);
            }

            Op::LogSoftmax => {
                // x - max - log(sum(exp(x - max))) directly: exact for very
                // negative outputs, where log(softmax(x)) loses them.
                let input = self.get_buffer(node.inputs[0]);
                let shape = &self.graph.node(node.inputs[0]).ty.shape;
                let batch = shape[0] as u32;
                let features = shape[1] as u32;
                self.emit_softmax_schedule(input, out_buf, batch, features, true);
            }

            Op::CrossEntropyLoss => {
                let logits = self.get_buffer(node.inputs[0]);
                let labels = self.get_buffer(node.inputs[1]);
                let shape = &self.graph.node(node.inputs[0]).ty.shape;
                let batch = shape[0] as u32;
                let features = shape[1] as u32;
                // Training autodiff emits CrossEntropyLogitsGrad on the same
                // (logits, labels). Reuse that pre-allocated buffer so the
                // forward kernel's fused grad is the backward value — no
                // second softmax / ones-row matmul. Inference leaves
                // write_grad=0 and binds the loss buffer as a dummy.
                let grad_buf = self.ce_logits_grad_buffer(node.inputs[0], node.inputs[1]);
                let write_grad = u32::from(grad_buf.is_some());
                // One workgroup per batch item (256 threads each), each
                // writing its row's share of the loss. The node's value is
                // their sum, so reduce the rows unless there is only one.
                let partials = if batch > 1 {
                    self.alloc_buffer(batch as usize * 4)
                } else {
                    out_buf
                };
                let grad_buf = grad_buf.unwrap_or(partials);
                self.plan.dispatches.push(Dispatch {
                    shader: if self.graph.node(node.inputs[1]).ty.dtype == DType::U32 {
                        ShaderEntry::CrossEntropyLossIndices
                    } else {
                        ShaderEntry::CrossEntropyLoss
                    },
                    workgroups: [batch, 1, 1],
                    input_buffers: vec![logits, labels],
                    output_buffer: grad_buf,
                    extra_outputs: vec![partials],
                    params: vec![batch, features, write_grad, 0],

                    ..Default::default()
                });
                if batch > 1 {
                    self.emit_reduce_all(ShaderEntry::SumAll, partials, out_buf, batch);
                }
            }

            Op::BceLoss => {
                let pred = self.get_buffer(node.inputs[0]);
                let labels = self.get_buffer(node.inputs[1]);
                let len = self.graph.node(node.inputs[0]).ty.num_elements() as u32;
                let num_wgs = len.div_ceil(256);
                // Each workgroup writes its share of the mean; the node's
                // value is their sum.
                let partials = if num_wgs > 1 {
                    self.alloc_buffer(num_wgs as usize * 4)
                } else {
                    out_buf
                };
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::BceLoss,
                    workgroups: [num_wgs, 1, 1],
                    input_buffers: vec![pred, labels],
                    output_buffer: partials,
                    extra_outputs: vec![],
                    params: vec![len, 0, 0, 0],

                    ..Default::default()
                });
                if num_wgs > 1 {
                    self.emit_reduce_all(ShaderEntry::SumAll, partials, out_buf, num_wgs);
                }
            }

            Op::Transpose => {
                let input = self.get_buffer(node.inputs[0]);
                let shape = &self.graph.node(node.inputs[0]).ty.shape;
                let rank = shape.len();
                let m = shape[rank - 2] as u32;
                let n = shape[rank - 1] as u32;
                let batch = shape[..rank - 2].iter().product::<usize>() as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::Transpose,
                    workgroups: [n.div_ceil(16), m.div_ceil(16), batch],
                    input_buffers: vec![input],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![m, n, 0, 0],

                    ..Default::default()
                });
            }

            Op::Permute { ref perm } => {
                let input = self.get_buffer(node.inputs[0]);
                let in_shape = &self.graph.node(node.inputs[0]).ty.shape;
                // Row-major input strides, then those of each output axis,
                // padded to rank 4 with leading unit axes.
                let mut strides = vec![1usize; in_shape.len()];
                for d in (0..in_shape.len().saturating_sub(1)).rev() {
                    strides[d] = strides[d + 1] * in_shape[d + 1];
                }
                let pad = 4 - perm.len();
                let mut dims = [1u32; 4];
                let mut out_strides = [0u32; 4];
                for (d, &axis) in perm.iter().enumerate() {
                    dims[pad + d] = u32::try_from(in_shape[axis]).expect("permute dim exceeds u32");
                    out_strides[pad + d] =
                        u32::try_from(strides[axis]).expect("permute stride exceeds u32");
                }
                self.emit_strided_copy(input, out_buf, dims, out_strides);
            }

            Op::BroadcastTo => {
                let input = self.get_buffer(node.inputs[0]);
                let in_shape = &self.graph.node(node.inputs[0]).ty.shape;
                // Repeated axes read with stride zero.
                let pad = 4 - in_shape.len();
                let mut dims = [1u32; 4];
                let mut strides = [0u32; 4];
                let mut stride = 1usize;
                for d in (0..in_shape.len()).rev() {
                    dims[pad + d] =
                        u32::try_from(node.ty.shape[d]).expect("broadcast dim exceeds u32");
                    if in_shape[d] != 1 {
                        strides[pad + d] =
                            u32::try_from(stride).expect("broadcast stride exceeds u32");
                    }
                    stride *= in_shape[d];
                }
                self.emit_strided_copy(input, out_buf, dims, strides);
            }

            Op::Silu => {
                self.emit_pointwise(pointwise(1, [Pw::Silu(0)]), node, out_buf);
            }

            Op::SwiGLU => {
                self.emit_pointwise(pointwise(2, [Pw::Silu(0), Pw::Mul(2, 1)]), node, out_buf);
            }

            Op::GeGLU => {
                // geglu(gate, up) = gelu(gate) · up
                let mut ops = gelu_ops(0, 2).to_vec();
                ops.push(Pw::Mul(10, 1));
                self.emit_pointwise(pointwise(2, ops), node, out_buf);
            }

            Op::SwiGLUConcat => {
                // input[M, 2*N] → output[M, N]
                let input = self.get_buffer(node.inputs[0]);
                let out_len = node.ty.num_elements() as u32;
                let half_n = node.ty.shape[1] as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::SwiGLUConcat,
                    workgroups: [out_len.div_ceil(256), 1, 1],
                    input_buffers: vec![input, input], // src_b unused in forward
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![out_len, half_n, 0, 0],

                    ..Default::default()
                });
            }

            Op::GeGLUConcat => {
                let input = self.get_buffer(node.inputs[0]);
                let out_len = node.ty.num_elements() as u32;
                let half_n = node.ty.shape[1] as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::GeGLUConcat,
                    workgroups: [out_len.div_ceil(256), 1, 1],
                    input_buffers: vec![input, input],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![out_len, half_n, 0, 0],

                    ..Default::default()
                });
            }

            Op::SwiGLUConcatGrad => {
                // (grad_out[M,N], input[M,2*N]) → grad_input[M,2*N]
                let grad_out = self.get_buffer(node.inputs[0]);
                let input = self.get_buffer(node.inputs[1]);
                let grad_out_len = self.graph.node(node.inputs[0]).ty.num_elements() as u32;
                let half_n = self.graph.node(node.inputs[0]).ty.shape[1] as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::SwiGLUConcatGrad,
                    workgroups: [grad_out_len.div_ceil(256), 1, 1],
                    input_buffers: vec![input, grad_out],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![grad_out_len, half_n, 0, 0],

                    ..Default::default()
                });
            }

            Op::GeGLUConcatGrad => {
                let grad_out = self.get_buffer(node.inputs[0]);
                let input = self.get_buffer(node.inputs[1]);
                let grad_out_len = self.graph.node(node.inputs[0]).ty.num_elements() as u32;
                let half_n = self.graph.node(node.inputs[0]).ty.shape[1] as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::GeGLUConcatGrad,
                    workgroups: [grad_out_len.div_ceil(256), 1, 1],
                    input_buffers: vec![input, grad_out],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![grad_out_len, half_n, 0, 0],

                    ..Default::default()
                });
            }

            Op::RmsNorm { eps } => {
                let x = self.get_buffer(node.inputs[0]);
                let w = self.get_buffer(node.inputs[1]);
                let shape = &self.graph.node(node.inputs[0]).ty.shape;
                let rows = shape[0] as u32;
                let cols = shape[1] as u32;
                self.emit_rmsnorm_schedule(x, w, out_buf, rows, cols, eps);
            }

            Op::Embedding => {
                let indices = self.get_buffer(node.inputs[0]);
                let table = self.get_buffer(node.inputs[1]);
                let idx_shape = &self.graph.node(node.inputs[0]).ty.shape;
                let tbl_node = self.graph.node(node.inputs[1]);
                let tbl_shape = &tbl_node.ty.shape;
                let seq = idx_shape[0] as u32;
                let hidden = tbl_shape[1] as u32;
                // The table's storage format picks the gather variant, so an
                // f16 table cannot silently be read by the f32 kernel.
                let wf = WeightFormat::from_dtype(tbl_node.ty.dtype);
                let workgroups = seq
                    .checked_mul(hidden)
                    .expect("embedding output exceeds u32 indexing")
                    .div_ceil(256);
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::Embedding,
                    workgroups: [workgroups, 1, 1],
                    input_buffers: vec![indices, table],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![seq, hidden, tbl_shape[0] as u32, 0],

                    weight_format: wf,
                    ..Default::default()
                });
            }

            Op::ToF16 => {
                // Elementwise f32 → f16 cast. UnaryData layout (src, dst,
                // params); dst is an f16 buffer.
                let src = self.get_buffer(node.inputs[0]);
                let len = node.ty.num_elements() as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::ToF16,
                    workgroups: [len.div_ceil(256), 1, 1],
                    input_buffers: vec![src],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![len, 0, 0, 0],

                    ..Default::default()
                });
            }

            Op::ScatterAdd { vocab_size } => {
                let indices = self.get_buffer(node.inputs[0]);
                let src = self.get_buffer(node.inputs[1]);
                let src_shape = &self.graph.node(node.inputs[1]).ty.shape;
                let seq_len = src_shape[0] as u32;
                let embed_dim = src_shape[1] as u32;
                let total = vocab_size as u32 * embed_dim;
                let params = vec![total, seq_len, embed_dim, 0];
                if u64::from(vocab_size as u32) * u64::from(seq_len) > 1_000_000 {
                    self.plan.dispatches.push(Dispatch {
                        shader: ShaderEntry::Generated,
                        workgroups: [total.div_ceil(256), 1, 1],
                        input_buffers: vec![src],
                        output_buffer: out_buf,
                        extra_outputs: vec![],
                        params: vec![total, 0, 0, 0],
                        kernel: Kernel::Pointwise(PointwiseDAG {
                            n_inputs: 1,
                            ops: vec![Pw::const_f32(0.0)],
                            output: 0,
                        }),

                        ..Default::default()
                    });
                    self.plan.dispatches.push(Dispatch {
                        shader: ShaderEntry::ScatterAddAtomic,
                        workgroups: [(seq_len * embed_dim).div_ceil(256), 1, 1],
                        // The output is also an input so scheduling inserts a
                        // global barrier after the zeroing dispatch.
                        input_buffers: vec![indices, src, out_buf],
                        output_buffer: out_buf,
                        extra_outputs: vec![],
                        params,

                        ..Default::default()
                    });
                } else {
                    self.plan.dispatches.push(Dispatch {
                        shader: ShaderEntry::ScatterAdd,
                        workgroups: [total.div_ceil(256), 1, 1],
                        input_buffers: vec![indices, src],
                        output_buffer: out_buf,
                        extra_outputs: vec![],
                        params,

                        ..Default::default()
                    });
                }
            }

            Op::RoPE {
                theta,
                pos_offset,
                head_dim,
                ..
            } => {
                let input = self.get_buffer(node.inputs[0]);
                let shape = &self.graph.node(node.inputs[0]).ty.shape;
                let seq = shape[0] as u32;
                let dim = shape[1] as u32;
                if node.inputs.len() == 3 {
                    // Dynamic offset with per-pair divisors.
                    let offset_buf = self.get_buffer(node.inputs[1]);
                    let factors = self.get_buffer(node.inputs[2]);
                    self.plan.dispatches.push(Dispatch {
                        shader: ShaderEntry::RoPEDynamicFactors,
                        workgroups: [(seq * dim / 2).div_ceil(256), 1, 1],
                        input_buffers: vec![input, offset_buf, factors],
                        output_buffer: out_buf,
                        extra_outputs: vec![],
                        params: vec![seq, dim, theta.to_bits(), pos_offset, head_dim, 0, 0, 0],

                        ..Default::default()
                    });
                } else if node.inputs.len() == 2 {
                    // Dynamic offset: read pos_offset from input buffer
                    let offset_buf = self.get_buffer(node.inputs[1]);
                    self.plan.dispatches.push(Dispatch {
                        shader: ShaderEntry::RoPEDynamic,
                        workgroups: [(seq * dim / 2).div_ceil(256), 1, 1],
                        input_buffers: vec![input, offset_buf],
                        output_buffer: out_buf,
                        extra_outputs: vec![],
                        params: vec![seq, dim, theta.to_bits(), pos_offset, head_dim, 0, 0, 0],

                        ..Default::default()
                    });
                } else {
                    self.plan.dispatches.push(Dispatch {
                        shader: ShaderEntry::RoPE,
                        workgroups: [(seq * dim / 2).div_ceil(256), 1, 1],
                        input_buffers: vec![input],
                        output_buffer: out_buf,
                        extra_outputs: vec![],
                        params: vec![seq, dim, theta.to_bits(), pos_offset, head_dim, 0, 0, 0],

                        ..Default::default()
                    });
                }
            }

            Op::RoPEPositions { theta, head_dim } => {
                let input = self.get_buffer(node.inputs[0]);
                let positions = self.get_buffer(node.inputs[1]);
                let shape = &self.graph.node(node.inputs[0]).ty.shape;
                let seq = shape[0] as u32;
                let dim = shape[1] as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::RoPEPositions,
                    workgroups: [(seq * dim / 2).div_ceil(256), 1, 1],
                    input_buffers: vec![input, positions],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![seq, dim, theta.to_bits(), 0, head_dim, 0, 0, 0],

                    ..Default::default()
                });
            }

            Op::CausalAttention {
                num_heads,
                num_kv_heads,
                head_dim,
            }
            | Op::CausalAttentionRoPE {
                num_heads,
                num_kv_heads,
                head_dim,
                ..
            } => {
                // Route through unified attention shader.
                // kv_seq=0 signals causal mask at runtime.
                let mut q = self.get_buffer(node.inputs[0]);
                let mut k = self.get_buffer(node.inputs[1]);
                let v = self.get_buffer(node.inputs[2]);
                let seq = self.graph.node(node.inputs[0]).ty.shape[0] as u32;
                if let Op::CausalAttentionRoPE { rope_theta, .. } = node.op {
                    // The attention kernels have no rotation of their own:
                    // rotate Q and K into scratch first, as the backward
                    // (which differentiates explicit RoPE nodes) assumes.
                    for (operand, input) in [(&mut q, node.inputs[0]), (&mut k, node.inputs[1])] {
                        let ty = &self.graph.node(input).ty;
                        let dim = ty.shape[1] as u32;
                        let rotated = self.alloc_buffer(ty.size_bytes());
                        self.plan.dispatches.push(Dispatch {
                            shader: ShaderEntry::RoPE,
                            workgroups: [(seq * dim / 2).div_ceil(256), 1, 1],
                            input_buffers: vec![*operand],
                            output_buffer: rotated,
                            extra_outputs: vec![],
                            params: vec![seq, dim, rope_theta.to_bits(), 0, head_dim, 0, 0, 0],
                            ..Default::default()
                        });
                        *operand = rotated;
                    }
                }
                let lse_buf = self.find_lse_buffer(node.id);
                let (shader, workgroups) =
                    self.attention_dispatch(seq, head_dim, num_heads, node.requires_full_precision);
                self.plan.dispatches.push(Dispatch {
                    shader,
                    workgroups,
                    input_buffers: vec![q, k, v],
                    output_buffer: out_buf,
                    extra_outputs: vec![lse_buf],
                    params: vec![seq, 0, (num_heads << 16) | num_kv_heads, head_dim],

                    ..Default::default()
                });
            }

            Op::SlidingWindowAttention {
                num_heads,
                num_kv_heads,
                head_dim,
                window_size,
            } => {
                // Route through unified attention shader.
                // kv_seq=0 signals causal, window_size>0 limits the window.
                let q = self.get_buffer(node.inputs[0]);
                let k = self.get_buffer(node.inputs[1]);
                let v = self.get_buffer(node.inputs[2]);
                let seq = self.graph.node(node.inputs[0]).ty.shape[0] as u32;
                let lse_buf = self.find_lse_buffer(node.id);
                let (shader, workgroups) =
                    self.attention_dispatch(seq, head_dim, num_heads, node.requires_full_precision);
                self.plan.dispatches.push(Dispatch {
                    shader,
                    workgroups,
                    input_buffers: vec![q, k, v],
                    output_buffer: out_buf,
                    extra_outputs: vec![lse_buf],
                    params: vec![
                        seq,
                        0,
                        (num_heads << 16) | num_kv_heads,
                        head_dim,
                        window_size,
                    ],

                    ..Default::default()
                });
            }

            Op::RoPEGrad {
                theta,
                pos_offset,
                head_dim,
            } => {
                let grad_out = self.get_buffer(node.inputs[0]);
                let shape = &self.graph.node(node.inputs[0]).ty.shape;
                let seq = shape[0] as u32;
                let dim = shape[1] as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::RoPEGrad,
                    workgroups: [(seq * dim / 2).div_ceil(256), 1, 1],
                    input_buffers: vec![grad_out],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![seq, dim, theta.to_bits(), pos_offset, head_dim, 0, 0, 0],

                    ..Default::default()
                });
            }

            Op::GroupNorm {
                num_groups,
                eps,
                channels,
                spatial,
            } => {
                let x = self.get_buffer(node.inputs[0]);
                let weight = self.get_buffer(node.inputs[1]);
                let bias = self.get_buffer(node.inputs[2]);
                let total = node.ty.shape[0] as u32;
                let batch = total / (channels * spatial);
                // Two passes so the parallelism scales with the image: one
                // workgroup per slice of a group, rather than one per group.
                let chunks = group_norm_chunks(batch, channels, spatial, num_groups);
                if chunks == 1 {
                    // On small tensors the original kernel is already busy
                    // enough. Keep its single dispatch instead of paying for
                    // an intermediate buffer, another full tensor pass, and a
                    // global barrier merely to split the group once.
                    self.plan.dispatches.push(Dispatch {
                        shader: ShaderEntry::GroupNorm,
                        workgroups: [batch * num_groups, 1, 1],
                        input_buffers: vec![x, weight, bias],
                        output_buffer: out_buf,
                        extra_outputs: vec![],
                        params: vec![batch, channels, spatial, num_groups, eps.to_bits(), 0, 0, 0],

                        ..Default::default()
                    });
                } else {
                    let slices = batch * num_groups * chunks;
                    let partials = self.alloc_buffer(slices as usize * 2 * 4);
                    let params = vec![
                        batch,
                        channels,
                        spatial,
                        num_groups,
                        eps.to_bits(),
                        chunks,
                        0,
                        0,
                    ];
                    self.plan.dispatches.push(Dispatch {
                        shader: ShaderEntry::GroupNormStats,
                        workgroups: [slices, 1, 1],
                        input_buffers: vec![x],
                        output_buffer: partials,
                        extra_outputs: vec![],
                        params: params.clone(),
                        ..Default::default()
                    });
                    self.plan.dispatches.push(Dispatch {
                        shader: ShaderEntry::GroupNormApply,
                        workgroups: [slices, 1, 1],
                        input_buffers: vec![x, partials, weight, bias],
                        output_buffer: out_buf,
                        extra_outputs: vec![],
                        params,

                        ..Default::default()
                    });
                }
            }

            Op::GroupNormSilu {
                num_groups,
                eps,
                channels,
                spatial,
            } => {
                let x = self.get_buffer(node.inputs[0]);
                let weight = self.get_buffer(node.inputs[1]);
                let bias = self.get_buffer(node.inputs[2]);
                let total = node.ty.shape[0] as u32;
                let batch = total / (channels * spatial);
                // Two passes so the parallelism scales with the image: one
                // workgroup per slice of a group, rather than one per group.
                let chunks = group_norm_chunks(batch, channels, spatial, num_groups);
                if chunks == 1 {
                    self.plan.dispatches.push(Dispatch {
                        shader: ShaderEntry::GroupNormSilu,
                        workgroups: [batch * num_groups, 1, 1],
                        input_buffers: vec![x, weight, bias],
                        output_buffer: out_buf,
                        extra_outputs: vec![],
                        params: vec![batch, channels, spatial, num_groups, eps.to_bits(), 0, 0, 0],

                        ..Default::default()
                    });
                } else {
                    let slices = batch * num_groups * chunks;
                    let partials = self.alloc_buffer(slices as usize * 2 * 4);
                    let params = vec![
                        batch,
                        channels,
                        spatial,
                        num_groups,
                        eps.to_bits(),
                        chunks,
                        1,
                        0,
                    ];
                    self.plan.dispatches.push(Dispatch {
                        shader: ShaderEntry::GroupNormStats,
                        workgroups: [slices, 1, 1],
                        input_buffers: vec![x],
                        output_buffer: partials,
                        extra_outputs: vec![],
                        params: params.clone(),
                        ..Default::default()
                    });
                    self.plan.dispatches.push(Dispatch {
                        shader: ShaderEntry::GroupNormApply,
                        workgroups: [slices, 1, 1],
                        input_buffers: vec![x, partials, weight, bias],
                        output_buffer: out_buf,
                        extra_outputs: vec![],
                        params,

                        ..Default::default()
                    });
                }
            }

            Op::GroupNormGradInput {
                num_groups,
                eps,
                channels,
                spatial,
            } => {
                let grad_out = self.get_buffer(node.inputs[0]);
                let input = self.get_buffer(node.inputs[1]);
                let weight = self.get_buffer(node.inputs[2]);
                let total = node.ty.shape[0] as u32;
                let batch = total / (channels * spatial);
                let stats = self.group_norm_grad_stats(
                    node.inputs[1],
                    [batch, channels, spatial, num_groups],
                    eps,
                );
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::GroupNormGradInput,
                    workgroups: [batch * num_groups, 1, 1],
                    input_buffers: vec![grad_out, input, weight, stats],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![batch, channels, spatial, num_groups, eps.to_bits(), 0, 0, 0],

                    ..Default::default()
                });
            }

            Op::GroupNormGradWeightBias {
                num_groups,
                eps,
                channels,
                spatial,
            } => {
                let grad_out = self.get_buffer(node.inputs[0]);
                let input = self.get_buffer(node.inputs[1]);
                let go_total = self.graph.node(node.inputs[0]).ty.shape[0] as u32;
                let batch = go_total / (channels * spatial);
                let stats = self.group_norm_grad_stats(
                    node.inputs[1],
                    [batch, channels, spatial, num_groups],
                    eps,
                );
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::GroupNormGradWeightBias,
                    workgroups: [channels, 1, 1],
                    input_buffers: vec![grad_out, input, stats],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![batch, channels, spatial, num_groups, eps.to_bits(), 0, 0, 0],

                    ..Default::default()
                });
            }

            Op::Concat {
                channels_a,
                channels_b,
                spatial,
            } => {
                let a = self.get_buffer(node.inputs[0]);
                let b = self.get_buffer(node.inputs[1]);
                let total = node.ty.shape[0] as u32;
                let batch = total / ((channels_a + channels_b) * spatial);
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::Concat,
                    workgroups: [total.div_ceil(256), 1, 1],
                    input_buffers: vec![a, b],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![batch, channels_a, channels_b, spatial],

                    ..Default::default()
                });
            }

            Op::SplitA {
                channels_a,
                channels_b,
                spatial,
            } => {
                let x = self.get_buffer(node.inputs[0]);
                let total = node.ty.shape[0] as u32;
                let batch = total / (channels_a * spatial);
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::SplitA,
                    workgroups: [total.div_ceil(256), 1, 1],
                    input_buffers: vec![x],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![batch, channels_a, channels_b, spatial],

                    ..Default::default()
                });
            }

            Op::SplitB {
                channels_a,
                channels_b,
                spatial,
            } => {
                let x = self.get_buffer(node.inputs[0]);
                let total = node.ty.shape[0] as u32;
                let batch = total / (channels_b * spatial);
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::SplitB,
                    workgroups: [total.div_ceil(256), 1, 1],
                    input_buffers: vec![x],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![batch, channels_a, channels_b, spatial],

                    ..Default::default()
                });
            }

            Op::Upsample2x {
                channels,
                in_h,
                in_w,
            } => {
                let x = self.get_buffer(node.inputs[0]);
                let total = node.ty.shape[0] as u32;
                let batch = total / (channels * in_h * 2 * in_w * 2);
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::Upsample2x,
                    workgroups: [total.div_ceil(256), 1, 1],
                    input_buffers: vec![x],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![batch, channels, in_h, in_w],

                    ..Default::default()
                });
            }

            Op::Upsample2xGrad {
                channels,
                in_h,
                in_w,
            } => {
                let grad = self.get_buffer(node.inputs[0]);
                let total = node.ty.shape[0] as u32;
                let batch = total / (channels * in_h * in_w);
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::Upsample2xGrad,
                    workgroups: [total.div_ceil(256), 1, 1],
                    input_buffers: vec![grad],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![batch, channels, in_h, in_w],

                    ..Default::default()
                });
            }

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
            } => {
                let input = self.get_buffer(node.inputs[0]);
                let kernel = self.get_buffer(node.inputs[1]);
                let in_shape = &self.graph.node(node.inputs[0]).ty.shape;
                let out_h = (in_h + 2 * padding_h - kernel_h) / stride + 1;
                let out_w = (in_w + 2 * padding_w - kernel_w) / stride + 1;
                let batch = in_shape[0] as u32 / (in_channels * in_h * in_w);

                // No 1×1 MatMul shortcut: the previous one viewed NCHW
                // input as [batch*H*W, Ci] row-major, which is a transposed
                // (pixel-scrambling) read of the actual [Ci, H*W] layout —
                // and its output was [HW, Co] while every consumer expects
                // [Co, HW]. Matching transposed views in the backward
                // shortcuts made training self-consistent, so gradchecks
                // passed while the network fought a fixed scrambler on
                // every residual projection. The general GEMM path is
                // CPU-parity-verified (tests/inference_parity_large.rs)
                // and handles kernel 1×1 as a degenerate im2col.
                {
                    // Use implicit GEMM: output = weight @ im2col(input)^T
                    // M=Co, N=oH*oW, K=Ci*kH*kW, batched in z dimension.
                    let spatial = out_h * out_w;
                    let tile = conv::register_tile(
                        out_channels,
                        spatial,
                        batch,
                        self.coop_caps.f32_tile > 0,
                    );
                    self.plan.dispatches.push(Dispatch {
                        shader: conv::entry(conv::Kind::Forward, tile),
                        workgroups: [spatial.div_ceil(tile), out_channels.div_ceil(tile), batch],
                        input_buffers: vec![input, kernel],
                        output_buffer: out_buf,
                        extra_outputs: vec![],
                        params: vec![
                            batch,
                            in_channels,
                            in_h,
                            in_w,
                            out_channels,
                            kernel_h,
                            kernel_w,
                            stride,
                            padding_h,
                            out_h,
                            out_w,
                            padding_w,
                        ],
                        kernel: conv::exact_kernel(),

                        ..Default::default()
                    });
                } // else (non-1x1 conv)
            }

            Op::MulPerChannel { channels, spatial } => {
                let src = self.get_buffer(node.inputs[0]);
                let gate = self.get_buffer(node.inputs[1]);
                let len = node.ty.num_elements() as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::MulPerChannel,
                    workgroups: [len.div_ceil(256), 1, 1],
                    input_buffers: vec![src, gate],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![len, spatial, channels, 0],

                    ..Default::default()
                });
            }

            Op::AddPerChannel { channels, spatial } => {
                let src = self.get_buffer(node.inputs[0]);
                let bias = self.get_buffer(node.inputs[1]);
                let len = node.ty.num_elements() as u32;
                // As a pointwise DAG the bias add fuses with the activation
                // after it, so a conv -> bias -> ReLU block writes one
                // activation instead of two.
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::Generated,
                    workgroups: [len.div_ceil(256), 1, 1],
                    input_buffers: vec![src, bias],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![len, 0, 0, 0],
                    kernel: Kernel::Pointwise(PointwiseDAG {
                        n_inputs: 2,
                        ops: vec![
                            Pw::LoadInput(0),
                            Pw::LoadBroadcast {
                                input: 1,
                                divisor: spatial,
                                modulus: channels,
                            },
                            Pw::Add(0, 1),
                        ],
                        output: 2,
                    }),
                    ..Default::default()
                });
            }

            Op::Conv2dDw {
                channels,
                in_h,
                in_w,
                kernel_h,
                kernel_w,
                stride,
                padding_h,
                padding_w,
            } => {
                let input = self.get_buffer(node.inputs[0]);
                let kernel = self.get_buffer(node.inputs[1]);
                let in_shape = &self.graph.node(node.inputs[0]).ty.shape;
                let out_h = (in_h + 2 * padding_h - kernel_h) / stride + 1;
                let out_w = (in_w + 2 * padding_w - kernel_w) / stride + 1;
                let batch = in_shape[0] as u32 / (channels * in_h * in_w);
                // The kernel stages one filter in a 49-entry shared array and
                // puts every (batch, channel) plane on the Z grid axis.
                assert!(
                    kernel_h * kernel_w <= 49,
                    "depthwise convolution supports kernels up to 7x7, got {kernel_h}x{kernel_w}"
                );
                assert!(
                    batch * channels <= 65535,
                    "depthwise convolution batch * channels ({}) exceeds the portable grid limit",
                    batch * channels
                );
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::Conv2dDw,
                    workgroups: [out_w.div_ceil(16), out_h.div_ceil(16), batch * channels],
                    input_buffers: vec![input, kernel],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![
                        batch, channels, in_h, in_w, kernel_h, kernel_w, stride, padding_h, out_h,
                        out_w, padding_w,
                    ],

                    ..Default::default()
                });
            }

            Op::WinogradConv2d {
                in_channels,
                in_h,
                in_w,
                out_channels,
                padding,
                adjoint,
            } => {
                let out_h = in_h + 2 * padding - 2; // 3x3 stride 1
                let out_w = in_w + 2 * padding - 2;
                let batch_size = node.ty.shape[0] as u32 / (out_channels * out_h * out_w);
                let tiles_h = out_h.div_ceil(2);
                let tiles_w = out_w.div_ceil(2);
                let total_tiles = batch_size * tiles_h * tiles_w;

                // Temp buffers
                let input_xform_size = (16 * in_channels * total_tiles * 4) as usize;
                let mm_out_size = (16 * out_channels * total_tiles * 4) as usize;
                let weight_xform =
                    self.alloc_buffer((16 * out_channels * in_channels * 4) as usize);
                let input_xform_buf = self.alloc_buffer(input_xform_size);
                let mm_out_buf = self.alloc_buffer(mm_out_size);

                let input = self.get_buffer(node.inputs[0]);
                let weight = self.get_buffer(node.inputs[1]);

                // Dispatch 0: transform the current weights, [Co·Ci·9] → [16, Co, Ci].
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::WinogradWeightTransform,
                    workgroups: [(out_channels * in_channels).div_ceil(256), 1, 1],
                    input_buffers: vec![weight],
                    output_buffer: weight_xform,
                    extra_outputs: vec![],
                    params: vec![out_channels, in_channels, u32::from(adjoint), 0, 0, 0, 0, 0],

                    ..Default::default()
                });

                // Dispatch 1: Input transform
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::WinogradInputTransform,
                    workgroups: [(total_tiles * in_channels).div_ceil(256), 1, 1],
                    input_buffers: vec![input],
                    output_buffer: input_xform_buf,
                    extra_outputs: vec![],
                    params: vec![
                        batch_size,
                        in_channels,
                        in_h,
                        in_w,
                        padding,
                        tiles_h,
                        tiles_w,
                        total_tiles,
                    ],

                    ..Default::default()
                });

                // Dispatch 2: Batched matmul
                // weight_xform[16, Co, Ci] × input_xform[16, Ci, P] → mm_out[16, Co, P]
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::WinogradBatchedMatMul,
                    workgroups: [total_tiles.div_ceil(64), out_channels.div_ceil(64), 16],
                    input_buffers: vec![weight_xform, input_xform_buf],
                    output_buffer: mm_out_buf,
                    extra_outputs: vec![],
                    params: vec![out_channels, total_tiles, in_channels, 0],

                    ..Default::default()
                });

                // Dispatch 3: Output transform
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::WinogradOutputTransform,
                    workgroups: [(total_tiles * out_channels).div_ceil(256), 1, 1],
                    input_buffers: vec![mm_out_buf],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![
                        batch_size,
                        out_channels,
                        out_h,
                        out_w,
                        tiles_h,
                        tiles_w,
                        total_tiles,
                        0,
                    ],

                    ..Default::default()
                });
            }

            Op::Conv2dGradInput {
                in_channels,
                in_h,
                in_w,
                out_channels,
                kernel_h,
                kernel_w,
                stride,
                padding_h,
                padding_w,
            } => {
                let grad_out = self.get_buffer(node.inputs[0]);
                let kernel = self.get_buffer(node.inputs[1]);
                let out_h = (in_h + 2 * padding_h - kernel_h) / stride + 1;
                let out_w = (in_w + 2 * padding_w - kernel_w) / stride + 1;
                let out_size = node.ty.shape[0] as u32;
                let batch = out_size / (in_channels * in_h * in_w);

                // No 1×1 MatMul shortcut — see the Op::Conv2d arm: the
                // shortcut family used a transposed (NHWC-ish) view of
                // NCHW data. The GEMM path handles kernel 1×1 correctly.
                {
                    // Use implicit GEMM: grad_input = weight_T @ im2col(grad_out)^T
                    // M=Ci, N=H*W, K=Co*kH*kW, batched in z dimension.
                    {
                        let spatial = in_h * in_w;
                        let tile = conv::register_tile(
                            in_channels,
                            spatial,
                            batch,
                            self.coop_caps.f32_tile > 0,
                        );
                        self.plan.dispatches.push(Dispatch {
                            shader: conv::entry(conv::Kind::GradInput, tile),
                            workgroups: [spatial.div_ceil(tile), in_channels.div_ceil(tile), batch],
                            input_buffers: vec![grad_out, kernel],
                            output_buffer: out_buf,
                            extra_outputs: vec![],
                            params: vec![
                                batch,
                                in_channels,
                                in_h,
                                in_w,
                                out_channels,
                                kernel_h,
                                kernel_w,
                                stride,
                                padding_h,
                                out_h,
                                out_w,
                                padding_w,
                            ],
                            kernel: conv::exact_kernel(),

                            ..Default::default()
                        });
                    }
                }
            }

            Op::Conv2dGradWeight {
                in_channels,
                in_h,
                in_w,
                out_channels,
                kernel_h,
                kernel_w,
                stride,
                padding_h,
                padding_w,
            } => {
                let grad_out = self.get_buffer(node.inputs[0]);
                let input = self.get_buffer(node.inputs[1]);
                let out_h = (in_h + 2 * padding_h - kernel_h) / stride + 1;
                let out_w = (in_w + 2 * padding_w - kernel_w) / stride + 1;
                let out_size = self.graph.node(node.inputs[0]).ty.shape[0] as u32;
                let batch = out_size / (out_channels * out_h * out_w);

                // No 1×1 MatMulAT shortcut — see the Op::Conv2d arm: the
                // shortcut family used a transposed (NHWC-ish) view of
                // NCHW data. The GEMM path handles kernel 1×1 correctly.
                {
                    // Use GEMM formulation: grad_weight[Co, Ci*kH*kW] = grad_out_flat[Co, N*oH*oW] @ im2col(input)[N*oH*oW, Ci*kH*kW]
                    let n_total = in_channels * kernel_h * kernel_w; // Ci*kH*kW
                    let m_total = out_channels; // Co
                    // Batch is folded into K, so it does not add workgroups.
                    let tile =
                        conv::register_tile(m_total, n_total, 1, self.coop_caps.f32_tile > 0);
                    self.plan.dispatches.push(Dispatch {
                        shader: conv::entry(conv::Kind::GradWeight, tile),
                        workgroups: [n_total.div_ceil(tile), m_total.div_ceil(tile), 1],
                        input_buffers: vec![grad_out, input],
                        output_buffer: out_buf,
                        extra_outputs: vec![],
                        params: vec![
                            batch,
                            in_channels,
                            in_h,
                            in_w,
                            out_channels,
                            kernel_h,
                            kernel_w,
                            stride,
                            padding_h,
                            out_h,
                            out_w,
                            padding_w,
                        ],
                        kernel: conv::exact_kernel(),

                        ..Default::default()
                    });
                }
            }

            Op::CacheWrite => {
                let new_kv = self.get_buffer(node.inputs[0]);
                let cache = self.get_buffer(node.inputs[1]);
                let kv_pos_input = self.get_buffer(node.inputs[2]);
                let dim = self.graph.node(node.inputs[0]).ty.shape[1] as u32;
                debug_assert_eq!(out_buf, cache);
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::CacheWrite,
                    workgroups: [dim.div_ceil(256), 1, 1],
                    input_buffers: vec![new_kv, cache, kv_pos_input],
                    output_buffer: cache,
                    extra_outputs: vec![],
                    // kv_pos is read from its buffer at run time.
                    params: vec![
                        dim,
                        self.graph.node(node.inputs[1]).ty.shape[0] as u32,
                        0,
                        0,
                    ],

                    ..Default::default()
                });
            }

            Op::CacheWritePrefix => {
                let new_kv = self.get_buffer(node.inputs[0]);
                let cache = self.get_buffer(node.inputs[1]);
                let kv_pos_input = self.get_buffer(node.inputs[2]);
                let valid_len_input = self.get_buffer(node.inputs[3]);
                let new_shape = &self.graph.node(node.inputs[0]).ty.shape;
                let cache_shape = &self.graph.node(node.inputs[1]).ty.shape;
                let block_len = new_shape[0] as u32;
                let dim = new_shape[1] as u32;
                let max_seq = cache_shape[0] as u32;
                debug_assert_eq!(out_buf, cache);
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::CacheWritePrefix,
                    workgroups: [(block_len * dim).div_ceil(256), 1, 1],
                    input_buffers: vec![new_kv, cache, kv_pos_input, valid_len_input],
                    output_buffer: cache,
                    extra_outputs: vec![],
                    params: vec![dim, block_len, max_seq, 0],

                    ..Default::default()
                });
            }

            Op::BiasedAttention {
                num_heads,
                num_kv_heads,
                head_dim,
                scale_bits,
                causal,
            } => {
                let rows = node.ty.shape[0] as u32;
                let keys = self.graph.node(node.inputs[1]).ty.shape[0] as u32;
                let inputs: Vec<_> = node.inputs.iter().map(|&i| self.get_buffer(i)).collect();
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::BiasedAttention,
                    workgroups: [rows, num_heads, 1],
                    // The kernel reads no position here; any buffer binds.
                    input_buffers: vec![inputs[0], inputs[1], inputs[2], inputs[3], inputs[0]],
                    output_buffer: out_buf,
                    params: vec![
                        rows,
                        keys,
                        (num_heads << 16) | num_kv_heads,
                        head_dim,
                        scale_bits,
                        u32::from(causal),
                        rows * keys,
                        keys,
                    ],
                    ..Default::default()
                });
            }

            Op::BiasedCachedAttention {
                num_heads,
                num_kv_heads,
                head_dim,
                scale_bits,
            } => {
                let rows = node.ty.shape[0] as u32;
                let keys = self.graph.node(node.inputs[1]).ty.shape[0] as u32;
                let inputs: Vec<_> = node.inputs.iter().map(|&i| self.get_buffer(i)).collect();
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::BiasedAttention,
                    workgroups: [rows, num_heads, 1],
                    input_buffers: vec![inputs[0], inputs[1], inputs[2], inputs[4], inputs[3]],
                    output_buffer: out_buf,
                    params: vec![
                        rows,
                        keys,
                        (num_heads << 16) | num_kv_heads,
                        head_dim,
                        scale_bits,
                        2,
                        keys,
                        0,
                    ],
                    ..Default::default()
                });
            }

            Op::CachedAttention {
                num_heads,
                num_kv_heads,
                head_dim,
            } => {
                let q = self.get_buffer(node.inputs[0]);
                let k_cache = self.get_buffer(node.inputs[1]);
                let v_cache = self.get_buffer(node.inputs[2]);
                let kv_pos_input = self.get_buffer(node.inputs[3]);
                let q_seq = self.graph.node(node.inputs[0]).ty.shape[0] as u32;
                // The single-query kernel is written for 64-wide heads; the
                // generated multi-query kernel takes any width.
                let (shader, block_queries) = if q_seq == 1 && head_dim == 64 {
                    (ShaderEntry::CachedAttention, 1)
                } else {
                    (
                        ShaderEntry::CachedQueryAttention,
                        crate::codegen::cached_attention_queries(head_dim),
                    )
                };
                self.plan.dispatches.push(Dispatch {
                    shader,
                    workgroups: [q_seq.div_ceil(block_queries), num_heads, 1],
                    input_buffers: vec![q, k_cache, v_cache, kv_pos_input],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![
                        q_seq,
                        num_heads,
                        num_kv_heads,
                        head_dim,
                        self.graph.node(node.inputs[1]).ty.shape[0] as u32,
                        0,
                        0,
                        0,
                    ],

                    ..Default::default()
                });
            }

            Op::CachedBlockAttention {
                num_heads,
                num_kv_heads,
                head_dim,
                window_size,
            } => {
                let q = self.get_buffer(node.inputs[0]);
                let k_cache = self.get_buffer(node.inputs[1]);
                let v_cache = self.get_buffer(node.inputs[2]);
                let kv_pos_input = self.get_buffer(node.inputs[3]);
                let valid_len_input = self.get_buffer(node.inputs[4]);
                let block_len = self.graph.node(node.inputs[0]).ty.shape[0] as u32;
                let max_seq = self.graph.node(node.inputs[1]).ty.shape[0] as u32;
                let mut params = CachedBlockAttentionParams {
                    window_size,
                    num_heads,
                    num_kv_heads,
                    head_dim,
                    block_len,
                    max_seq,
                    splits: 0,
                    _pad: 0,
                };
                let splits = self.options.cached_attention_splits.unwrap_or_else(|| {
                    if block_len == 1 && max_seq > 64 {
                        (max_seq.div_ceil(32)).clamp(2, 16)
                    } else {
                        1
                    }
                });
                assert!(
                    (1..=16).contains(&splits),
                    "cached attention needs 1..=16 splits"
                );
                if splits > 1 {
                    params.splits = splits;
                    let scratch_idx = self.plan.buffers.len() as u32;
                    self.plan.buffers.push(
                        block_len as usize
                            * num_heads as usize
                            * splits as usize
                            * (head_dim + 2) as usize
                            * 4,
                    );
                    let partials = BufferRef(scratch_idx);
                    self.plan.dispatches.push(Dispatch {
                        shader: ShaderEntry::CachedBlockAttentionSplit,
                        workgroups: [block_len, splits, num_heads],
                        input_buffers: vec![q, k_cache, v_cache, kv_pos_input, valid_len_input],
                        output_buffer: partials,
                        extra_outputs: vec![],
                        params: params.to_words(),

                        ..Default::default()
                    });
                    self.plan.dispatches.push(Dispatch {
                        shader: ShaderEntry::CachedBlockAttentionCombine,
                        workgroups: [block_len, num_heads, 1],
                        input_buffers: vec![partials],
                        output_buffer: out_buf,
                        extra_outputs: vec![],
                        params: params.to_words(),

                        ..Default::default()
                    });
                } else {
                    self.plan.dispatches.push(Dispatch {
                        shader: ShaderEntry::CachedBlockAttention,
                        workgroups: [block_len, num_heads, 1],
                        input_buffers: vec![q, k_cache, v_cache, kv_pos_input, valid_len_input],
                        output_buffer: out_buf,
                        extra_outputs: vec![],
                        params: params.to_words(),

                        ..Default::default()
                    });
                }
            }

            Op::ChunkedRelativeAttention {
                num_heads,
                head_dim,
                left_context,
                softcap_bits,
            } => {
                let q = self.get_buffer(node.inputs[0]);
                let k = self.get_buffer(node.inputs[1]);
                let v = self.get_buffer(node.inputs[2]);
                let relative_k = self.get_buffer(node.inputs[3]);
                let seq_len = self.graph.node(node.inputs[0]).ty.shape[0] as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::ChunkedRelativeAttention,
                    workgroups: [seq_len, num_heads, 1],
                    input_buffers: vec![q, k, v, relative_k],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![
                        seq_len,
                        num_heads,
                        head_dim,
                        left_context,
                        softcap_bits,
                        0,
                        0,
                        0,
                    ],

                    ..Default::default()
                });
            }

            Op::PrefixLast => {
                let input = self.get_buffer(node.inputs[0]);
                let valid_len = self.get_buffer(node.inputs[1]);
                let shape = &self.graph.node(node.inputs[0]).ty.shape;
                let rows = shape[0] as u32;
                let cols = shape[1] as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::PrefixLast,
                    workgroups: [cols.div_ceil(256), 1, 1],
                    input_buffers: vec![input, valid_len],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![cols, rows, 0, 0],

                    ..Default::default()
                });
            }

            Op::MaxPool2d {
                channels,
                in_h,
                in_w,
                kernel_h,
                kernel_w,
                stride,
                padding,
            } => {
                let input = self.get_buffer(node.inputs[0]);
                let in_shape = &self.graph.node(node.inputs[0]).ty.shape;
                let batch = in_shape[0] as u32 / (channels * in_h * in_w);
                let out_h = (in_h + 2 * padding - kernel_h) / stride + 1;
                let out_w = (in_w + 2 * padding - kernel_w) / stride + 1;
                let total = batch * channels * out_h * out_w;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::MaxPool2d,
                    workgroups: [total.div_ceil(256), 1, 1],
                    input_buffers: vec![input],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![
                        batch, channels, in_h, in_w, kernel_h, kernel_w, stride, padding, out_h,
                        out_w, 0, 0,
                    ],

                    ..Default::default()
                });
            }

            Op::MaxPool2dGrad {
                channels,
                in_h,
                in_w,
                kernel_h,
                kernel_w,
                stride,
                padding,
            } => {
                let grad_out = self.get_buffer(node.inputs[0]);
                let input = self.get_buffer(node.inputs[1]);
                let total = node.ty.num_elements() as u32;
                let batch = total / (channels * in_h * in_w);
                let out_h = (in_h + 2 * padding - kernel_h) / stride + 1;
                let out_w = (in_w + 2 * padding - kernel_w) / stride + 1;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::MaxPool2dGrad,
                    workgroups: [total.div_ceil(256), 1, 1],
                    input_buffers: vec![grad_out, input],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![
                        batch, channels, in_h, in_w, kernel_h, kernel_w, stride, padding, out_h,
                        out_w, 0, 0,
                    ],

                    ..Default::default()
                });
            }

            Op::GlobalAvgPool { channels, spatial } if spatial > 32 => {
                // One thread per (batch, channel) plane serialises a whole
                // plane per lane: EfficientNet's squeeze-excite pools
                // 112x112 planes, 32 of them at batch 1. Give each plane a
                // workgroup with contiguous loads instead.
                use crate::schedule::{PointwiseDAG, Pw, ReduceOp, ReductionKernel};
                let input = self.get_buffer(node.inputs[0]);
                let rows = node.ty.num_elements() as u32;
                debug_assert_eq!(rows % channels, 0);
                let kernel = ReductionKernel {
                    op: ReduceOp::Sum,
                    prologue: PointwiseDAG {
                        n_inputs: 1,
                        ops: vec![
                            Pw::LoadInput(0),
                            Pw::const_f32(1.0 / spatial as f32),
                            Pw::Mul(0, 1),
                        ],
                        output: 2,
                    },
                    extra_prologues: vec![],
                    epilogue: None,
                    n_per_elem: 1,
                    n_per_row: 0,
                    workgroup_size: 256,
                    rows_per_workgroup: 1,
                    gather_elem: Vec::new(),
                    input_row_repeats: Vec::new(),
                };
                self.plan.dispatches.push(Dispatch {
                    // Generated-reduction routing takes priority over the
                    // sentinel entry in pipeline selection and binding.
                    shader: ShaderEntry::Generated,
                    workgroups: [rows, 1, 1],
                    input_buffers: vec![input],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![rows, spatial, 1.0_f32.to_bits(), 0],
                    kernel: Kernel::Reduction(kernel),
                    ..Default::default()
                });
            }

            Op::GlobalAvgPool { channels, spatial } => {
                let input = self.get_buffer(node.inputs[0]);
                let total_out = node.ty.num_elements() as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::GlobalAvgPool,
                    workgroups: [total_out.div_ceil(256), 1, 1],
                    input_buffers: vec![input],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![channels, spatial, total_out, 0],

                    ..Default::default()
                });
            }

            Op::GlobalAvgPoolGrad {
                channels: _,
                spatial,
            } => {
                let grad_output = self.get_buffer(node.inputs[0]);
                let total = node.ty.num_elements() as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::GlobalAvgPoolGrad,
                    workgroups: [total.div_ceil(256), 1, 1],
                    input_buffers: vec![grad_output],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![total, spatial, 0, 0],

                    ..Default::default()
                });
            }

            Op::Gelu => {
                self.emit_pointwise(pointwise(1, gelu_ops(0, 1)), node, out_buf);
            }

            Op::LayerNorm { eps } => {
                let x = self.get_buffer(node.inputs[0]);
                let w = self.get_buffer(node.inputs[1]);
                let bias = self.get_buffer(node.inputs[2]);
                let shape = &self.graph.node(node.inputs[0]).ty.shape;
                let rows = shape[0] as u32;
                let cols = shape[1] as u32;
                // One workgroup per row. The generated reduction template
                // performs a single reduction per row, which forces the
                // cancelling E[x²] − E[x]² variance; the hand-written
                // kernel takes the mean first, then the squared deviations.
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::LayerNorm,
                    workgroups: [rows, 1, 1],
                    input_buffers: vec![x, w, bias],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![rows, cols, eps.to_bits(), 0],

                    ..Default::default()
                });
            }

            // Full and cross attention are the wrapper's self and cross
            // forms: one lowering, with the key length from K's shape.
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
                let q = self.get_buffer(node.inputs[0]);
                let k = self.get_buffer(node.inputs[1]);
                let v = self.get_buffer(node.inputs[2]);
                let q_seq = self.graph.node(node.inputs[0]).ty.shape[0] as u32;
                let kv_seq = self.graph.node(node.inputs[1]).ty.shape[0] as u32;
                let lse_buf = self.find_lse_buffer(node.id);
                let (shader, workgroups) = self.attention_dispatch(
                    q_seq,
                    head_dim,
                    num_heads,
                    node.requires_full_precision,
                );
                self.plan.dispatches.push(Dispatch {
                    shader,
                    workgroups,
                    input_buffers: vec![q, k, v],
                    output_buffer: out_buf,
                    extra_outputs: vec![lse_buf],
                    params: vec![q_seq, kv_seq, (num_heads << 16) | num_kv_heads, head_dim],

                    ..Default::default()
                });
            }

            Op::MultiHeadAttnGradQ {
                fwd_node,
                num_heads,
                num_kv_heads,
                head_dim,
                ..
            } => {
                let d_out = self.get_buffer(node.inputs[0]);
                let q = self.get_buffer(node.inputs[1]);
                let k = self.get_buffer(node.inputs[2]);
                let v = self.get_buffer(node.inputs[3]);
                let fwd_o = self.get_buffer(fwd_node);
                let lse_buf = self.find_lse_buffer(fwd_node);
                let q_seq = self.graph.node(node.inputs[1]).ty.shape[0] as u32;
                let fwd_op = &self.graph.node(fwd_node).op;
                let is_causal = matches!(
                    fwd_op,
                    Op::CausalAttention { .. }
                        | Op::CausalAttentionRoPE { .. }
                        | Op::SlidingWindowAttention { .. }
                );
                let kv_seq = if is_causal {
                    0
                } else {
                    self.graph.node(node.inputs[2]).ty.shape[0] as u32
                };
                let window_size = match *fwd_op {
                    Op::SlidingWindowAttention { window_size, .. } => window_size,
                    _ => 0,
                };
                let dispatch_kv = if is_causal { q_seq } else { kv_seq };
                let (kernel, workgroups, ept_cap) = self.attention_grad_kernel(
                    node,
                    Part::Q,
                    [q_seq, dispatch_kv],
                    num_heads,
                    head_dim,
                );
                let row_source = if kernel.reads_row_dot() {
                    self.emit_attention_row_dot(d_out, fwd_o, q_seq * num_heads, head_dim)
                } else {
                    fwd_o
                };
                self.plan.dispatches.push(Dispatch {
                    kernel: if kernel.takes_ept_cap() {
                        Kernel::AttentionBackward { ept_cap }
                    } else {
                        Kernel::Default
                    },
                    shader: ShaderEntry::AttentionGrad(kernel),
                    workgroups,
                    input_buffers: vec![d_out, q, k, v, lse_buf, row_source],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![
                        q_seq,
                        kv_seq,
                        (num_heads << 16) | num_kv_heads,
                        head_dim,
                        window_size,
                    ],

                    ..Default::default()
                });
            }

            Op::MultiHeadAttnGradK {
                fwd_node,
                num_heads,
                num_kv_heads,
                head_dim,
                ..
            } => {
                let d_out = self.get_buffer(node.inputs[0]);
                let q = self.get_buffer(node.inputs[1]);
                let k = self.get_buffer(node.inputs[2]);
                let v = self.get_buffer(node.inputs[3]);
                let fwd_o = self.get_buffer(fwd_node);
                let lse_buf = self.find_lse_buffer(fwd_node);
                let q_seq = self.graph.node(node.inputs[1]).ty.shape[0] as u32;
                let fwd_op = &self.graph.node(fwd_node).op;
                let is_causal = matches!(
                    fwd_op,
                    Op::CausalAttention { .. }
                        | Op::CausalAttentionRoPE { .. }
                        | Op::SlidingWindowAttention { .. }
                );
                let kv_seq = if is_causal {
                    0
                } else {
                    self.graph.node(node.inputs[2]).ty.shape[0] as u32
                };
                let window_size = match *fwd_op {
                    Op::SlidingWindowAttention { window_size, .. } => window_size,
                    _ => 0,
                };
                let dispatch_kv = if is_causal { q_seq } else { kv_seq };

                // Use GradV's already-allocated destination, including any
                // views established in the first pass. If there is no GradV
                // node, the fused kernel still needs a scratch destination.
                let dv_buf = match self.pending_grad_v_buffers.remove(&fwd_node) {
                    Some(buf) => buf,
                    None => self.alloc_buffer(self.graph.node(node.inputs[3]).ty.size_bytes()),
                };
                let attention_params = vec![
                    q_seq,
                    kv_seq,
                    (num_heads << 16) | num_kv_heads,
                    head_dim,
                    window_size,
                ];
                let (kernel, workgroups, ept_cap) = self.attention_grad_kernel(
                    node,
                    Part::KV,
                    [dispatch_kv, q_seq],
                    num_kv_heads,
                    head_dim,
                );
                // Most dK/dV kernels need dot(dO, O) for every query row.
                // Reduce it once here; recomputing it in every KV workgroup
                // read O and spent a third of the inner loop's products on it.
                let row_source = if kernel.reads_row_dot() {
                    self.emit_attention_row_dot(d_out, fwd_o, q_seq * num_heads, head_dim)
                } else {
                    fwd_o
                };
                self.plan.dispatches.push(Dispatch {
                    kernel: if kernel.takes_ept_cap() {
                        Kernel::AttentionBackward { ept_cap }
                    } else {
                        Kernel::Default
                    },
                    shader: ShaderEntry::AttentionGrad(kernel),
                    workgroups,
                    input_buffers: vec![d_out, q, k, v, lse_buf, row_source],
                    output_buffer: out_buf,
                    extra_outputs: vec![dv_buf],
                    params: attention_params,

                    ..Default::default()
                });
            }

            Op::MultiHeadAttnGradV { fwd_node, .. } => {
                // GradK emits the fused dK+dV dispatch. Never remap this
                // output here: its views already alias the allocated buffer.
                assert!(
                    !self.pending_grad_v_buffers.contains_key(&fwd_node),
                    "attention GradV for forward node {fwd_node} compiled before fused GradKV"
                );
                return;
            }

            Op::SwiGLUGradGate => {
                // inputs: [grad_out, gate, up]
                let grad_out = self.get_buffer(node.inputs[0]);
                let gate = self.get_buffer(node.inputs[1]);
                let up = self.get_buffer(node.inputs[2]);
                let len = node.ty.num_elements() as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::SwiGLUGradGate,
                    workgroups: [len.div_ceil(256), 1, 1],
                    input_buffers: vec![grad_out, gate, up],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![len, 0, 0, 0],

                    ..Default::default()
                });
            }

            Op::SwiGLUGradUp => {
                // inputs: [grad_out, gate]
                let grad_out = self.get_buffer(node.inputs[0]);
                let gate = self.get_buffer(node.inputs[1]);
                let len = node.ty.num_elements() as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::SwiGLUGradUp,
                    workgroups: [len.div_ceil(256), 1, 1],
                    input_buffers: vec![grad_out, gate],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![len, 0, 0, 0],

                    ..Default::default()
                });
            }

            Op::SiluGrad => {
                // inputs: [grad_out, x]
                let grad_out = self.get_buffer(node.inputs[0]);
                let x = self.get_buffer(node.inputs[1]);
                let len = node.ty.num_elements() as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::SiluGrad,
                    workgroups: [len.div_ceil(256), 1, 1],
                    input_buffers: vec![grad_out, x],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![len, 0, 0, 0],

                    ..Default::default()
                });
            }

            Op::RmsNormGradW { eps } => {
                let dy = self.get_buffer(node.inputs[0]);
                let x = self.get_buffer(node.inputs[1]);
                let w = self.get_buffer(node.inputs[2]);
                let x_shape = &self.graph.node(node.inputs[1]).ty.shape;
                let rows = x_shape[0] as u32;
                let cols = x_shape[1] as u32;
                if rows >= 4 {
                    // Two passes for better GPU occupancy: row blocks write
                    // partial[block, col] = sum(dy * x * rsqrt), then SumRows
                    // reduces the blocks.
                    let block = norm_weight_grad_rows_per_workgroup(rows);
                    let blocks = rows.div_ceil(block);
                    let temp_buf = self.alloc_buffer((blocks as usize) * (cols as usize) * 4);
                    self.plan.dispatches.push(Dispatch {
                        shader: ShaderEntry::RmsNormGradWRowPar,
                        workgroups: [blocks, 1, 1],
                        input_buffers: vec![dy, x, w],
                        output_buffer: temp_buf,
                        extra_outputs: vec![],
                        params: vec![rows, cols, eps.to_bits(), block],

                        ..Default::default()
                    });
                    self.push_sum_rows(blocks, cols, temp_buf, out_buf);
                } else {
                    // Small row count: single-pass is fine
                    self.plan.dispatches.push(Dispatch {
                        shader: ShaderEntry::RmsNormGradW,
                        workgroups: [cols.div_ceil(256), 1, 1],
                        input_buffers: vec![dy, x, w],
                        output_buffer: out_buf,
                        extra_outputs: vec![],
                        params: vec![rows, cols, eps.to_bits(), 0],

                        ..Default::default()
                    });
                }
            }

            Op::RmsNormGradX { eps } => {
                let dy = self.get_buffer(node.inputs[0]);
                let x = self.get_buffer(node.inputs[1]);
                let w = self.get_buffer(node.inputs[2]);
                let x_shape = &self.graph.node(node.inputs[1]).ty.shape;
                let rows = x_shape[0] as u32;
                let cols = x_shape[1] as u32;
                const WG: u32 = 256;
                let lanes_per_row = if (2..=32).contains(&cols) {
                    cols.next_power_of_two()
                } else {
                    0
                };
                let workgroups = WG
                    .checked_div(lanes_per_row)
                    .map_or(rows, |packed_rows| rows.div_ceil(packed_rows));
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::RmsNormGradX,
                    workgroups: [workgroups, 1, 1],
                    input_buffers: vec![dy, x, w],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![rows, cols, eps.to_bits(), lanes_per_row],

                    ..Default::default()
                });
            }

            Op::LayerNormGradWB { eps } => {
                let dy = self.get_buffer(node.inputs[0]);
                let x = self.get_buffer(node.inputs[1]);
                let w = self.get_buffer(node.inputs[2]);
                let x_shape = &self.graph.node(node.inputs[1]).ty.shape;
                let rows = x_shape[0] as u32;
                let cols = x_shape[1] as u32;
                // Every workgroup writes one row of block sums, so the
                // `[cols]` output can only take them directly from one block.
                let block = norm_weight_grad_rows_per_workgroup(rows);
                let blocks = rows.div_ceil(block);
                let partial = if blocks > 1 {
                    self.alloc_buffer((blocks as usize) * (cols as usize) * 4)
                } else {
                    out_buf
                };
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::LayerNormGradWB,
                    workgroups: [blocks, 1, 1],
                    input_buffers: vec![dy, x, w],
                    output_buffer: partial,
                    extra_outputs: vec![],
                    params: vec![rows, cols, eps.to_bits(), block],

                    ..Default::default()
                });
                if blocks > 1 {
                    self.push_sum_rows(blocks, cols, partial, out_buf);
                }
            }

            Op::LayerNormGradX { eps } => {
                let dy = self.get_buffer(node.inputs[0]);
                let x = self.get_buffer(node.inputs[1]);
                let w = self.get_buffer(node.inputs[2]);
                let x_shape = &self.graph.node(node.inputs[1]).ty.shape;
                let rows = x_shape[0] as u32;
                let cols = x_shape[1] as u32;
                self.plan.dispatches.push(Dispatch {
                    shader: ShaderEntry::LayerNormGradX,
                    workgroups: [rows, 1, 1],
                    input_buffers: vec![dy, x, w],
                    output_buffer: out_buf,
                    extra_outputs: vec![],
                    params: vec![rows, cols, eps.to_bits(), 0],

                    ..Default::default()
                });
            }
        }

        // Apply the extracted layout to the ordinary binding contract. GEMV
        // and packed weights retain their own implementations.
        if let Some(spec) = node.matmul_impl
            && self.plan.dispatches.len() == dispatch_start + 1
            && self.plan.dispatches[dispatch_start].shader.is_matmul()
            && !self.plan.dispatches[dispatch_start]
                .weight_format
                .is_quantized()
        {
            let shape = spec.shape;
            assert!(
                shape.legal() && spec.splits > 0,
                "illegal extracted matmul: {spec:?}"
            );
            let dispatch = &mut self.plan.dispatches[dispatch_start];
            dispatch.kernel = Kernel::ScalarMatmul(shape);
            dispatch.workgroups = matmul_workgroups_rect(
                node.ty.shape[0] as u32,
                node.ty.shape[1] as u32,
                shape.rows(),
                shape.cols(),
            );
            // If splitting is illegal (e.g. short K), retain this single-pass
            // layout. Plan deduplication and the report see the actual lowering.
            if spec.splits >= 2 {
                let _ = self
                    .plan
                    .split_matmul(dispatch_start, shape, spec.splits, usize::MAX);
            }
            self.plan.dispatches[dispatch_start].schedule_locked = true;
        }

        // A single graph node can lower to multiple dispatches (for example,
        // row-parallel normalization gradients). Preserve the node's numeric
        // policy on every emitted dispatch so runtime kernel selection cannot
        // silently turn f32 autodiff math into f16-input math.
        for dispatch in &mut self.plan.dispatches[dispatch_start..] {
            dispatch.requires_full_precision = node.requires_full_precision;
        }
    }

    pub(super) fn find_lse_buffer(&self, fwd_node: NodeId) -> BufferRef {
        self.plan
            .lse_buffers
            .iter()
            .find(|item| item.0 == fwd_node)
            .expect("LSE buffer not found for MultiHeadAttn forward node")
            .1
    }

    /// Softmax, or log-softmax with `log`, as two row reductions: the row
    /// max, then the sum of `exp(x - max)` with an epilogue producing
    /// `exp(x - max) / sum` or `(x - max) - log(sum)`.
    pub(super) fn emit_softmax_schedule(
        &mut self,
        input: BufferRef,
        out_buf: BufferRef,
        batch: u32,
        features: u32,
        log: bool,
    ) {
        use crate::schedule::{PointwiseDAG, Pw, ReduceOp, ReductionEpilogue, ReductionKernel};

        const WG: u32 = 256;
        // Give each narrow row a power-of-two lane group and fill the rest of
        // the workgroup with independent rows. This keeps the reduction order
        // while avoiding one mostly idle 256-lane workgroup per short row.
        let rows_per_workgroup = if (2..=32).contains(&features) {
            WG / features.next_power_of_two()
        } else {
            1
        };

        // Allocate intermediate row_max buffer: batch × f32.
        let row_max = self.alloc_buffer((batch as usize) * 4);

        // --- Dispatch 1: row-wise max reduction ---
        let max_prologue = PointwiseDAG {
            n_inputs: 1,
            ops: vec![Pw::LoadInput(0)],
            output: 0,
        };
        let max_kernel = ReductionKernel {
            op: ReduceOp::Max,
            prologue: max_prologue,
            extra_prologues: vec![],
            epilogue: None,
            n_per_elem: 1,
            n_per_row: 0,
            workgroup_size: WG,
            rows_per_workgroup,
            gather_elem: Vec::new(),
            input_row_repeats: Vec::new(),
        };
        self.plan.dispatches.push(Dispatch {
            // Sentinel shader for runtime data-layout selection (UnaryData).
            shader: ShaderEntry::Generated,
            workgroups: [batch.div_ceil(rows_per_workgroup), 1, 1],
            input_buffers: vec![input],
            output_buffer: row_max,
            extra_outputs: vec![],
            params: vec![batch, features, 0, 0],

            kernel: Kernel::Reduction(max_kernel),
            ..Default::default()
        });

        // --- Dispatch 2: sum reduction with exp-subtract prologue + normalize epilogue ---
        // Prologue DAG: inputs are 0=src (per-elem), 1=row_max (per-row).
        //   exp(src - row_max) → scalar contribution.
        let sum_prologue = PointwiseDAG {
            n_inputs: 2,
            ops: vec![
                Pw::LoadInput(0), // v0 = src
                Pw::LoadInput(1), // v1 = row_max
                Pw::Sub(0, 1),    // v2 = src - row_max
                Pw::Exp(2),       // v3 = exp(src - row_max)
            ],
            output: 3,
        };
        // Epilogue: inputs 0=src (per-elem), 1=row_max (per-row),
        //           2=row_sum (reduced scalar, always last).
        let sum_epilogue_dag = PointwiseDAG {
            n_inputs: 3,
            ops: vec![
                Pw::LoadInput(0), // v0 = src
                Pw::LoadInput(1), // v1 = row_max
                Pw::LoadInput(2), // v2 = row_sum
                Pw::Sub(0, 1),    // v3 = src - row_max
                if log {
                    Pw::Log(2) // v4 = log(row_sum)
                } else {
                    Pw::Exp(3) // v4 = exp(src - row_max)
                },
                if log {
                    Pw::Sub(3, 4) // v5 = (src - row_max) - log(row_sum)
                } else {
                    Pw::Div(4, 2) // v5 = exp(...) / row_sum
                },
            ],
            output: 5,
        };
        let sum_kernel = ReductionKernel {
            op: ReduceOp::Sum,
            prologue: sum_prologue,
            extra_prologues: vec![],
            epilogue: Some(ReductionEpilogue {
                dag: sum_epilogue_dag,
                n_per_col_inputs: 0,
            }),
            n_per_elem: 1,
            n_per_row: 1,
            workgroup_size: WG,
            rows_per_workgroup,
            gather_elem: Vec::new(),
            input_row_repeats: Vec::new(),
        };
        self.plan.dispatches.push(Dispatch {
            // Sentinel for runtime data-layout: 1 per-elem + 1 per-row → 2
            // buffer inputs → we key the runtime off `reduction.is_some()`
            // and the kernel's arity, so the shader field is purely a
            // historical leftover here.
            shader: ShaderEntry::Generated,
            workgroups: [batch.div_ceil(rows_per_workgroup), 1, 1],
            input_buffers: vec![input, row_max],
            output_buffer: out_buf,
            extra_outputs: vec![],
            params: vec![batch, features, 0, 0],

            kernel: Kernel::Reduction(sum_kernel),
            ..Default::default()
        });
    }

    /// Emit LayerNorm as a single schedule-template reduction with two
    /// accumulators:
    ///   prologues: x and x*x (sum and sum-of-squares)
    ///   op: Sum
    ///   epilogue: (x - mean) * rsqrt(var + eps) * weight[col] + bias[col]
    ///   where mean = r0/cols, var = r1/cols - mean².
    /// Emit RmsNorm as a single schedule-template reduction:
    ///   prologue: v*v (sum-of-squares)
    ///   op: Sum
    ///   epilogue: src * rsqrt(sum_sq / cols + eps) * weight[col]
    pub(super) fn emit_rmsnorm_schedule(
        &mut self,
        x: BufferRef,
        w: BufferRef,
        out_buf: BufferRef,
        rows: u32,
        cols: u32,
        eps: f32,
    ) {
        let kernel = rmsnorm_kernel(cols, eps);
        let rows_per_workgroup = kernel.rows_per_workgroup;

        // Uses RmsNormData layout: src + bias (per-col weight) + dst + params.
        self.plan.dispatches.push(Dispatch {
            shader: ShaderEntry::Generated,
            workgroups: [rows.div_ceil(rows_per_workgroup), 1, 1],
            input_buffers: vec![x, w],
            output_buffer: out_buf,
            extra_outputs: vec![],
            // The schedule shader embeds epsilon in its DAG, but retain the
            // bits in dispatch metadata as well: runtime may rewrite this
            // dispatch to RmsNormRsqrt for a cooperative matmul prologue.
            params: vec![rows, cols, eps.to_bits(), 0],

            kernel: Kernel::Reduction(kernel),
            ..Default::default()
        });
    }

    pub(super) fn emit_generated_unary(&mut self, op: Pw, node: &Node, out_buf: BufferRef) {
        let input = self.get_buffer(node.inputs[0]);
        let len = node.ty.num_elements() as u32;
        self.plan.dispatches.push(Dispatch {
            shader: ShaderEntry::Generated,
            workgroups: [len.div_ceil(256), 1, 1],
            input_buffers: vec![input],
            output_buffer: out_buf,
            extra_outputs: vec![],
            params: vec![len, 0, 0, 0],

            kernel: Kernel::Pointwise(PointwiseDAG {
                n_inputs: 1,
                ops: vec![Pw::LoadInput(0), op],
                output: 1,
            }),
            ..Default::default()
        });
    }

    /// One elementwise dispatch computing `dag` over the node's inputs.
    pub(super) fn emit_pointwise(&mut self, dag: PointwiseDAG, node: &Node, out_buf: BufferRef) {
        let input_buffers: Vec<BufferRef> = node
            .inputs
            .iter()
            .map(|&input| self.get_buffer(input))
            .collect();
        assert_eq!(
            input_buffers.len(),
            usize::from(dag.n_inputs),
            "{:?}",
            node.op
        );
        let len = node.ty.num_elements() as u32;
        self.plan.dispatches.push(Dispatch {
            shader: ShaderEntry::Generated,
            workgroups: [len.div_ceil(256), 1, 1],
            input_buffers,
            output_buffer: out_buf,
            extra_outputs: vec![],
            params: vec![len, 0, 0, 0],
            kernel: Kernel::Pointwise(dag),
            ..Default::default()
        });
    }

    /// `out[i] = a[i] (+ or *) b[i % len(b)]`: a bias or scale per column.
    /// As a broadcast pointwise DAG it fuses with its neighbours, and a
    /// constant operand (autodiff broadcasts a row with `zeros + b`) folds
    /// to a literal.
    pub(super) fn emit_row_broadcast(
        &mut self,
        combine: crate::schedule::Pw,
        node: &Node,
        out_buf: BufferRef,
    ) {
        let a = self.get_buffer(node.inputs[0]);
        let b = self.get_buffer(node.inputs[1]);
        let len = node.ty.num_elements() as u32;
        let row_len = self.graph.node(node.inputs[1]).ty.num_elements() as u32;
        self.plan.dispatches.push(Dispatch {
            shader: ShaderEntry::Generated,
            workgroups: [len.div_ceil(256), 1, 1],
            input_buffers: vec![a, b],
            output_buffer: out_buf,
            extra_outputs: vec![],
            params: vec![len, 0, 0, 0],
            kernel: Kernel::Pointwise(PointwiseDAG {
                n_inputs: 2,
                ops: vec![
                    Pw::LoadInput(0),
                    Pw::LoadBroadcast {
                        input: 1,
                        divisor: 1,
                        modulus: row_len,
                    },
                    combine,
                ],
                output: 2,
            }),
            ..Default::default()
        });
    }
}
