use super::{BufferRef, Dispatch, ExecutionPlan, ShaderEntry};
use crate::tune::{MatmulTile, TuneClass, TuneError};

impl ExecutionPlan {
    /// Move an unfused cooperative f32 product to the shared-tile native
    /// 16x16 kernel. More than one K partition writes compact M*N partials
    /// that a following SumRows adds up. Preflights before mutating the plan.
    pub(crate) fn tile_native_f32_matmul(
        &mut self,
        index: usize,
        shape: crate::codegen::CooperativeMatmulShape,
        splits: u32,
        max_partial_bytes: usize,
    ) -> Result<(), TuneError> {
        let dispatch = self
            .dispatches
            .get(index)
            .ok_or(TuneError("missing matrix dispatch"))?;
        if dispatch.kernel != super::Kernel::Cooperative
            || dispatch.schedule_locked
            || dispatch.weight_format != super::WeightFormat::F32
            || dispatch.horizontal_batch > 1
            || dispatch.matmul_epilogue.is_some()
            || dispatch.workgroups[2] != 1
            || !dispatch.shader.is_matmul()
        {
            return Err(TuneError(
                "tiled native f32 requires an unfused cooperative product",
            ));
        }
        // The prologue binding layout does not carry an additional addend.
        if dispatch
            .matmul_prologue
            .as_ref()
            .is_some_and(|p| dispatch.input_buffers.len() > 2 + p.factors.len())
        {
            return Err(TuneError(
                "tiled native f32 prologue with addend is unsupported",
            ));
        }
        let (m, n, k) = dispatch.mnk().ok_or(TuneError("not a dense product"))?;
        let rows = crate::codegen::CooperativeMatmulShape::ROWS;
        if !shape.fits_dimensions(m, n, k)
            || !(1..=(k / shape.k_stage).min(65535)).contains(&splits)
            || k.checked_add(splits * shape.k_stage).is_none()
            || m / rows > 65535
            || n / shape.columns > 65535
        {
            return Err(TuneError(
                "tiled native f32 needs complete tiles and a K stage per partition",
            ));
        }
        if dispatch.input_buffers.contains(&dispatch.output_buffer) {
            return Err(TuneError(
                "tiled native f32 cannot alias an input with its output",
            ));
        }
        let columns = m.checked_mul(n).ok_or(TuneError("matrix size overflow"))?;
        let binding_fits = |buffer: Option<&BufferRef>, elements: Option<u32>| {
            buffer.zip(elements).is_some_and(|(b, elements)| {
                (elements as usize)
                    .checked_mul(4)
                    .zip(self.buffers.get(b.0 as usize))
                    .is_some_and(|(required, &available)| required <= available)
            })
        };
        if !binding_fits(dispatch.input_buffers.first(), m.checked_mul(k))
            || !binding_fits(dispatch.input_buffers.get(1), n.checked_mul(k))
            || !binding_fits(Some(&dispatch.output_buffer), Some(columns))
            || (matches!(
                dispatch.shader,
                ShaderEntry::FusedMatMulAdd
                    | ShaderEntry::FusedMatMulATAdd
                    | ShaderEntry::FusedMatMulBTAdd
            ) && !binding_fits(dispatch.input_buffers.get(2), Some(columns)))
        {
            return Err(TuneError("tiled native f32 binding capacity is too small"));
        }
        let mut producer = dispatch.clone();
        producer.kernel = super::Kernel::CooperativeTiled { shape, splits };
        producer.workgroups = [m / rows, n / shape.columns, splits];
        if splits == 1 {
            self.dispatches[index] = producer;
            return Ok(());
        }
        self.splice_k_partials(index, producer, columns, splits, max_partial_bytes)
    }

    /// Replace dispatch `index` by `producer`, redirected to a new buffer of
    /// `splits` partial `columns`-element results, and a SumRows of those
    /// partials into the original output.
    fn splice_k_partials(
        &mut self,
        index: usize,
        mut producer: Dispatch,
        columns: u32,
        splits: u32,
        max_partial_bytes: usize,
    ) -> Result<(), TuneError> {
        if columns.div_ceil(256) > 0xFFFF {
            return Err(TuneError("split-K reduction exceeds dispatch limits"));
        }
        let bytes = columns
            .checked_mul(splits)
            .and_then(|n| (n as usize).checked_mul(4))
            .filter(|&bytes| bytes <= max_partial_bytes)
            .ok_or(TuneError("split-K scratch budget"))?;
        let partial = BufferRef(
            u32::try_from(self.buffers.len()).map_err(|_| TuneError("too many buffers"))?,
        );
        let dispatch = &self.dispatches[index];
        let reduction = Dispatch {
            shader: ShaderEntry::SumRows,
            workgroups: [columns.div_ceil(256), 1, 1],
            input_buffers: vec![partial],
            output_buffer: dispatch.output_buffer,
            params: vec![splits, columns, 1, 0],
            requires_full_precision: dispatch.requires_full_precision,
            fusion_barrier: dispatch.fusion_barrier,
            label: format!("{} split-K reduction", dispatch.label),
            origin: dispatch.origin.clone(),
            ..Default::default()
        };
        producer.output_buffer = partial;
        producer.label = format!("{} split-K {splits}", dispatch.label);
        self.buffers.push(bytes);
        self.dispatches
            .splice(index..index + 1, [producer, reduction]);
        Ok(())
    }

    pub(super) fn split_low_occupancy_conv_weights(&mut self, options: super::ConvWeightSplits) {
        assert!(options.reduction_chunk >= 16 && options.reduction_chunk.is_multiple_of(16));
        let mut bytes = 0usize;
        let mut selections = Vec::new();
        for (index, dispatch) in self.dispatches.iter().enumerate() {
            let Some(class) = TuneClass::from_dispatch(dispatch, None).filter(|c| {
                c.shader == ShaderEntry::Conv2dGradWeightGemm
                    && matches!(dispatch.conv_k_tile(), None | Some(16))
                    && dispatch.input_buffers[0] != dispatch.input_buffers[1]
                    && dispatch.workgroups[0].saturating_mul(dispatch.workgroups[1])
                        < options.workgroup_threshold
            }) else {
                continue;
            };
            let splits = class.k.div_ceil(16).div_ceil(options.reduction_chunk / 16);
            if !(2..=65_535).contains(&splits) {
                continue;
            }
            let Some(total) = (class.m as usize)
                .checked_mul(class.n as usize)
                .and_then(|n| n.checked_mul(splits as usize))
                .and_then(|n| n.checked_mul(4))
                .and_then(|n| n.checked_add(bytes))
                .filter(|&n| n <= options.max_partial_bytes)
            else {
                continue;
            };
            selections.push((index, splits));
            bytes = total;
        }
        if let Err(error) = self.split_conv_weight_gradients(&selections, options.max_partial_bytes)
        {
            log::warn!("keeping unsplit convolution gradients: {error}");
        }
    }

    /// Lower one matrix product (with optional addition) to partials + SumRows.
    /// This is a candidate, not a selection: qualify and time the entire sequence.
    pub(crate) fn split_matmul(
        &mut self,
        index: usize,
        shape: crate::codegen::ScalarMatmulShape,
        splits: u32,
        max_partial_bytes: usize,
    ) -> Result<(), TuneError> {
        if !shape.legal() {
            return Err(TuneError("unsupported split-K tile"));
        }
        let dispatch = self
            .dispatches
            .get(index)
            .ok_or(TuneError("missing matrix dispatch"))?;
        let mut class = TuneClass::from_dispatch(dispatch, None)
            .filter(|c| {
                c.shader.is_matmul()
                    && matches!(
                        c.weight_format,
                        super::WeightFormat::F32 | super::WeightFormat::F16
                    )
            })
            .ok_or(TuneError("split-K requires a scalar matrix product"))?;
        let mut bindings = dispatch.input_buffers.clone();
        bindings.push(dispatch.output_buffer);
        let mut unique = bindings.clone();
        unique.sort_unstable_by_key(|b| b.0);
        unique.dedup();
        if unique.len() != bindings.len() {
            return Err(TuneError("split-K requires distinct bindings"));
        }
        class.binding_bytes = bindings
            .iter()
            .map(|b| self.buffers.get(b.0 as usize).copied())
            .collect::<Option<Vec<_>>>()
            .ok_or(TuneError("invalid matrix binding"))?;
        if !MatmulTile::Scalar(shape).fits(&class)
            || !(2..=0xFFFF).contains(&splits)
            || splits > class.k.div_ceil(shape.k_stage)
            || class.k.checked_add(shape.k_stage - 1).is_none()
            || class.m.div_ceil(shape.rows()) > 0xFFFF
        {
            return Err(TuneError("illegal split-K dimensions or binding capacity"));
        }
        let columns = class
            .m
            .checked_mul(class.n)
            .ok_or(TuneError("matrix size overflow"))?;
        let mut producer = dispatch.clone();
        producer.kernel = super::Kernel::SplitMatmul { shape, splits };
        producer.workgroups = [
            class.n.div_ceil(shape.cols()),
            class.m.div_ceil(shape.rows()),
            splits,
        ];
        self.splice_k_partials(index, producer, columns, splits, max_partial_bytes)
    }

    /// Experimentally lower selected scalar convolution weight gradients to
    /// partials followed by the existing SumRows reduction, before allocating a session.
    ///
    /// Selections are `(dispatch_index, splits)` in this plan's current order.
    /// Preflight every selection before changing anything. The byte cap bounds
    /// the sum of new logical partial capacities, conservatively before aliasing;
    /// the session's ordinary memory preflight still checks actual allocation.
    /// The returned value is this charged sum, not peak memory or tuning scratch.
    ///
    /// This changes reduction order. Callers must qualify numerical behavior and
    /// measure the full sequence; this method neither selects nor qualifies a winner.
    /// The live tile tuner excludes these two-pass entries and cannot swap them.
    pub fn split_conv_weight_gradients(
        &mut self,
        selections: &[(usize, u32)],
        max_partial_bytes: usize,
    ) -> Result<usize, TuneError> {
        let mut selections = selections.to_vec();
        selections.sort_unstable();
        if selections.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            return Err(TuneError("duplicate split-K dispatch selection"));
        }
        let mut replacements = Vec::new();
        let mut total_bytes = 0usize;
        for (index, splits) in selections {
            let dispatch = self
                .dispatches
                .get(index)
                .ok_or(TuneError("split-K dispatch index out of range"))?;
            let mut class = TuneClass::from_dispatch(dispatch, None)
                .filter(|class| {
                    class.shader == ShaderEntry::Conv2dGradWeightGemm
                        && matches!(dispatch.conv_k_tile(), None | Some(16))
                })
                .ok_or(TuneError(
                    "split-K requires an unmodified legal scalar weight gradient",
                ))?;
            let bindings: Vec<_> = dispatch
                .input_buffers
                .iter()
                .chain(std::iter::once(&dispatch.output_buffer))
                .collect();
            for (i, buffer) in bindings.iter().enumerate() {
                if bindings[..i].contains(buffer) {
                    return Err(TuneError("split-K requires distinct logical bindings"));
                }
                class.binding_bytes.push(
                    *self
                        .buffers
                        .get(buffer.0 as usize)
                        .ok_or(TuneError("split-K binding index out of range"))?,
                );
            }
            let tile = MatmulTile::selected(dispatch, None).expect("checked scalar class");
            if !tile.fits(&class) {
                return Err(TuneError("split-K binding capacity is too small"));
            }
            if !(2..=65_535).contains(&splits) || splits > class.k.div_ceil(16) {
                return Err(TuneError(
                    "split-K needs 2..65535 nonempty reduction partitions",
                ));
            }
            let columns = class
                .m
                .checked_mul(class.n)
                .ok_or(TuneError("split-K output index overflow"))?;
            if columns.div_ceil(32) > 65_535 {
                return Err(TuneError(
                    "split-K final reduction exceeds portable dispatch limits",
                ));
            }
            let bytes = columns
                .checked_mul(splits)
                .and_then(|elements| usize::try_from(elements).ok())
                .and_then(|elements| elements.checked_mul(4))
                .ok_or(TuneError("split-K partial index overflow"))?;
            total_bytes = total_bytes
                .checked_add(bytes)
                .filter(|&total| total <= max_partial_bytes)
                .ok_or(TuneError("split-K partial byte budget exceeded"))?;
            let buffer_index = self
                .buffers
                .len()
                .checked_add(replacements.len())
                .and_then(|index| u32::try_from(index).ok())
                .ok_or(TuneError("split-K buffer index overflow"))?;
            let partial = BufferRef(buffer_index);
            let mut producer = dispatch.clone();
            let width = match tile {
                MatmulTile::Tile16 | MatmulTile::SpecializedConv { tile_size: 16, .. } => 16,
                MatmulTile::Tile32 | MatmulTile::SpecializedConv { tile_size: 32, .. } => 32,
                _ => 64,
            };
            producer.shader = match width {
                16 => ShaderEntry::Conv2dGradWeightGemmSplit16,
                32 => ShaderEntry::Conv2dGradWeightGemmSplitSmall,
                _ => ShaderEntry::Conv2dGradWeightGemmSplit,
            };
            producer.workgroups[2] = splits;
            producer.output_buffer = partial;
            producer.label = format!("{} split-K partials ({splits})", dispatch.label);
            let reduction = Dispatch {
                shader: ShaderEntry::SumRows,
                workgroups: [columns.div_ceil(32), 1, 1],
                input_buffers: vec![partial],
                output_buffer: dispatch.output_buffer,
                params: vec![splits, columns, 0, 0],
                requires_full_precision: dispatch.requires_full_precision,
                fusion_barrier: dispatch.fusion_barrier,
                label: format!("{} split-K reduction", dispatch.label),
                origin: dispatch.origin.clone(),
                ..Default::default()
            };
            replacements.push((index, bytes, producer, reduction));
        }
        self.buffers.extend(replacements.iter().map(|r| r.1));
        for (index, _, producer, reduction) in replacements.into_iter().rev() {
            self.dispatches
                .splice(index..index + 1, [producer, reduction]);
        }
        Ok(total_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Graph;

    #[test]
    fn tiled_native_preflight_and_single_dispatch_tuning() {
        use crate::codegen::{CoopConfig, CooperativeMatmulShape};
        let mut graph = Graph::new();
        let a = graph.input("a", &[128, 256]);
        let b = graph.input("b", &[256, 256]);
        let y = graph.matmul(a, b);
        graph.set_outputs(vec![y]);
        let mut base = super::super::compile(&graph);
        base.dispatches[0].kernel = super::super::Kernel::Cooperative;
        let shape = CooperativeMatmulShape {
            columns: 128,
            k_stage: 16,
            prefetch: false,
        };
        for case in 0..10 {
            let mut plan = base.clone();
            let mut candidate = shape;
            let mut budget = usize::MAX;
            match case {
                0 => candidate.columns = 0,
                1 => candidate.k_stage = 64,
                2 => plan.dispatches[0].params[0] = 96,
                3 => plan.dispatches[0].params[1] = 252,
                4 => plan.dispatches[0].params[2] = 192,
                5 => plan.dispatches[0].schedule_locked = true,
                6 => budget = 128 * 256 * 4 * 4 - 1,
                7 => {
                    let b = plan.dispatches[0].input_buffers[1];
                    plan.buffers[b.0 as usize] -= 4;
                }
                // Four partitions need four K stages.
                8 => plan.dispatches[0].params[1] = 48,
                9 => plan.dispatches[0].output_buffer = plan.dispatches[0].input_buffers[0],
                _ => unreachable!(),
            }
            let before = plan.clone();
            assert!(
                plan.tile_native_f32_matmul(0, candidate, 4, budget)
                    .is_err()
            );
            assert_eq!(plan, before);
        }
        for splits in [1, 4] {
            let mut plan = base.clone();
            plan.tile_native_f32_matmul(0, shape, splits, usize::MAX)
                .unwrap();
            assert_eq!(plan.dispatches[0].workgroups, [2, 2, splits]);
            assert_eq!(plan.output_buffers, base.output_buffers);
            let config = CoopConfig {
                tile_size: 16,
                use_f16_input: false,
                compensated: false,
            };
            assert_eq!(
                TuneClass::from_dispatch(&plan.dispatches[0], Some(&config)).is_some(),
                splits == 1
            );
            if splits == 1 {
                assert_eq!(plan.buffers, base.buffers);
                let mut class =
                    TuneClass::from_dispatch(&plan.dispatches[0], Some(&config)).unwrap();
                class.binding_bytes = plan.dispatches[0]
                    .input_buffers
                    .iter()
                    .chain(std::iter::once(&plan.dispatches[0].output_buffer))
                    .map(|b| plan.buffers[b.0 as usize])
                    .collect();
                let selected = MatmulTile::CooperativeTiled(shape);
                assert!(selected.fits(&class));
                assert!(
                    class
                        .challengers(selected, Some(&config))
                        .iter()
                        .any(|c| matches!(c, MatmulTile::CooperativeF32 { .. }))
                );
            } else {
                assert_eq!(*plan.buffers.last().unwrap(), 128 * 256 * 4 * 4);
                assert_eq!(
                    plan.dispatches[1].input_buffers,
                    [plan.dispatches[0].output_buffer]
                );
                super::super::schedule_dispatches(&mut plan, false, true);
                assert_eq!(plan.groups, [0..1, 1..2]);
            }
            let decoded: ExecutionPlan =
                serde_json::from_str(&serde_json::to_string(&plan).unwrap()).unwrap();
            assert_eq!(decoded, plan);
        }
    }

    #[test]
    fn tiled_native_f32_covers_transposes_prologues_and_empty_partitions() {
        use super::super::{Kernel, MatMulPrologue, PrologueLoadKind};
        use crate::codegen::CooperativeMatmulShape;
        let gpu = crate::reference::gpu::shared_context();
        if !gpu
            .capabilities()
            .cooperative_matrix
            .f32_shapes
            .contains(&[16, 16, 16])
        {
            return;
        }
        for columns in [64, 128] {
            for k_stage in [16, 32] {
                let (m, n, k) = (64usize, 128usize, k_stage as usize * 5);
                for transpose in 0..3 {
                    for fusion in 0..3 {
                        let mut graph = Graph::new();
                        let a = graph.input("a", &if transpose == 1 { [k, m] } else { [m, k] });
                        let b = graph.input("b", &if transpose == 2 { [n, k] } else { [k, n] });
                        let y = match transpose {
                            1 => graph.matmul_at(a, b),
                            2 => graph.matmul_bt(a, b),
                            _ => graph.matmul(a, b),
                        };
                        let y = if fusion == 1 {
                            let src = graph.input("src", &[m, n]);
                            graph.add(y, src)
                        } else {
                            y
                        };
                        graph.set_outputs(vec![y]);
                        let mut base = super::super::compile(&crate::optimize::optimize(&graph));
                        assert_eq!(base.dispatches.len(), 1);
                        base.dispatches[0].kernel = Kernel::Cooperative;
                        base.dispatches[0].workgroups = [m as u32 / 32, n as u32 / 32, 1];
                        if fusion == 2 {
                            let mut factors = Vec::new();
                            for (name, count, kind) in [
                                ("row", m, PrologueLoadKind::PerRow),
                                ("col", k, PrologueLoadKind::PerKCol),
                            ] {
                                let buffer = BufferRef(base.buffers.len() as u32);
                                base.buffers.push(count * 4);
                                base.input_buffers.push((name.to_owned(), buffer));
                                base.dispatches[0].input_buffers.push(buffer);
                                factors.push((buffer, kind));
                            }
                            base.dispatches[0].matmul_prologue = Some(MatMulPrologue { factors });
                        }
                        let a: Vec<_> = (0..m * k)
                            .map(|i| ((i * 17 % 101) as f32 - 50.0) * 1e-12)
                            .collect();
                        let b: Vec<_> = (0..k * n)
                            .map(|i| ((i * 31 % 97) as f32 - 48.0) * 1e5)
                            .collect();
                        let src: Vec<_> = (0..m * n)
                            .map(|i| ((i * 7 % 31) as f32 - 15.0) * 0.001)
                            .collect();
                        let row: Vec<_> = (0..m).map(|i| 0.5 + i as f32 * 0.01).collect();
                        let col: Vec<_> = (0..k).map(|i| 0.75 + i as f32 * 0.002).collect();
                        let mut expected = vec![0.0f64; m * n];
                        for r in 0..m {
                            for c in 0..n {
                                let mut value = if fusion == 1 {
                                    f64::from(src[r * n + c])
                                } else {
                                    0.0
                                };
                                for j in 0..k {
                                    let ai = if transpose == 1 { j * m + r } else { r * k + j };
                                    let bi = if transpose == 2 { c * k + j } else { j * n + c };
                                    let av = if fusion == 2 {
                                        a[ai] * row[r] * col[j]
                                    } else {
                                        a[ai]
                                    };
                                    value += f64::from(av) * f64::from(b[bi]);
                                }
                                expected[r * n + c] = value;
                            }
                        }
                        for prefetch in [false, true] {
                            for splits in [1, 4] {
                                let shape = CooperativeMatmulShape {
                                    columns,
                                    k_stage,
                                    prefetch,
                                };
                                let mut plan = base.clone();
                                plan.tile_native_f32_matmul(0, shape, splits, usize::MAX)
                                    .unwrap();
                                let mut session = crate::Session::with_context_opts(
                                    plan,
                                    gpu.clone(),
                                    crate::SessionOptions {
                                        coop: crate::CoopPolicy::NativeF32,
                                        ..Default::default()
                                    },
                                );
                                session.set_input("a", &a);
                                session.set_input("b", &b);
                                if fusion == 1 {
                                    session.set_input("src", &src);
                                }
                                if fusion == 2 {
                                    session.set_input("row", &row);
                                    session.set_input("col", &col);
                                }
                                session.step();
                                session.wait();
                                let got = session.read_output(m * n);
                                let max_ref =
                                    expected.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
                                let mut error2 = 0.0;
                                let mut ref2 = 0.0;
                                for (actual, expected) in got.into_iter().zip(&expected) {
                                    let error = (f64::from(actual) - expected).abs();
                                    assert!(
                                        actual.is_finite() && error <= 1e-10 + 3e-5 * max_ref,
                                        "{shape:?} transpose={transpose} fusion={fusion} splits={splits}: {actual} != {expected}"
                                    );
                                    error2 += error * error;
                                    ref2 += expected * expected;
                                }
                                assert!((error2 / ref2).sqrt() <= 1e-5);
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn opt_in_weight_splits_preserve_defaults_and_respect_budget() {
        let mut graph = Graph::new();
        let x = graph.input("x", &[128 * 3 * 64 * 64]);
        let dy = graph.input("dy", &[128 * 8 * 64 * 64]);
        let dw = graph.conv2d_grad_weight(dy, x, 3, 64, 64, 8, 5, 5, 1, 2, 2);
        graph.set_outputs(vec![dw]);
        let base = super::super::compile(&graph);
        assert_eq!(base.dispatches.len(), 1);
        let capacity = 600 * 1024 * 4;
        for (groups, chunk, budget, expected) in [
            (48, 512, capacity, 2),
            (48, 512, capacity - 1, 1),
            (5, 512, capacity, 1),
            (48, 524288, capacity, 1),
        ] {
            let options = super::super::CompileOptions {
                conv_weight_splits: Some(super::super::ConvWeightSplits {
                    workgroup_threshold: groups,
                    reduction_chunk: chunk,
                    max_partial_bytes: budget,
                }),
                ..Default::default()
            };
            let plan = super::super::compile_with(&graph, &options);
            assert_eq!(plan.dispatches.len(), expected);
            assert_eq!(plan.output_buffers, base.output_buffers);
            if expected == 2 {
                assert_eq!(
                    plan.dispatches[0].shader,
                    ShaderEntry::Conv2dGradWeightGemmSplit16
                );
                assert_eq!(plan.dispatches[0].workgroups, [5, 1, 1024]);
                assert_eq!(plan.dispatches[1].shader, ShaderEntry::SumRows);
                assert_eq!(*plan.buffers.last().unwrap(), capacity);
            } else {
                assert_eq!(plan.dispatches, base.dispatches);
            }
        }
    }

    #[test]
    #[ignore = "GPU dense split-K candidate qualification on an idle device"]
    fn dense_split_candidates_preserve_transposition_and_ragged_edges() {
        let gpu = std::sync::Arc::new(
            crate::init_gpu_context_with(crate::GpuOptions::from_env()).unwrap(),
        );
        for case in 0..6 {
            let transpose = case % 3;
            let add = case >= 3;
            let (m, k, n) = (7, 67, 11);
            let mut graph = Graph::new();
            let a = graph.input("a", &if transpose == 1 { [k, m] } else { [m, k] });
            let b = graph.parameter("b", &if transpose == 2 { [n, k] } else { [k, n] });
            let y = match transpose {
                0 => graph.matmul(a, b),
                1 => graph.matmul_at(a, b),
                _ => graph.matmul_bt(a, b),
            };
            let y = if add {
                let c = graph.input("c", &[m, n]);
                graph.add(y, c)
            } else {
                y
            };
            graph.set_outputs(vec![y]);
            let a: Vec<_> = (0..m * k).map(|i| (i as f32 * 0.21).sin()).collect();
            let b: Vec<_> = (0..k * n).map(|i| (i as f32 * 0.13).cos()).collect();
            let mut reference = vec![0.0_f64; m * n];
            for row in 0..m {
                for col in 0..n {
                    reference[row * n + col] = (0..k)
                        .map(|j| {
                            let ai = if transpose == 1 {
                                j * m + row
                            } else {
                                row * k + j
                            };
                            let bi = if transpose == 2 {
                                col * k + j
                            } else {
                                j * n + col
                            };
                            f64::from(a[ai]) * f64::from(b[bi])
                        })
                        .sum::<f64>()
                        + if add { 0.125 } else { 0.0 };
                }
            }
            for (tile_size, tile_n, k_stage, splits, unroll_k) in [
                (32, 0, 8, 3, false),
                (64, 32, 16, 4, true),
                (32, 64, 32, 2, true),
            ] {
                let mut plan = super::super::compile(&crate::optimize::optimize(&graph));
                let shape = crate::codegen::ScalarMatmulShape {
                    tile_size,
                    tile_n,
                    k_stage,
                    interleave_columns: true,
                    unroll_k,
                };
                let before = serde_json::to_value(&plan).unwrap();
                assert!(plan.split_matmul(0, shape, splits, 0).is_err());
                assert_eq!(before, serde_json::to_value(&plan).unwrap());
                plan.split_matmul(0, shape, splits, 1024 * 1024).unwrap();
                assert!(TuneClass::from_dispatch(&plan.dispatches[0], None).is_none());
                let mut session = crate::Session::with_context_opts(
                    plan,
                    gpu.clone(),
                    crate::SessionOptions {
                        coop: crate::CoopPolicy::Disabled,
                        ..Default::default()
                    },
                );
                session.set_input("a", &a);
                session.set_parameter("b", &b);
                if add {
                    session.set_input("c", &vec![0.125; m * n]);
                }
                session.step();
                session.wait();
                for (actual, expected) in session.read_output(m * n).into_iter().zip(&reference) {
                    assert!(
                        actual.is_finite()
                            && (f64::from(actual) - expected).abs() < 2e-5 + 2e-4 * expected.abs(),
                        "{actual} != {expected}"
                    );
                }
            }
        }
    }

    fn plan() -> (ExecutionPlan, usize) {
        let mut graph = Graph::new();
        let x = graph.input("x", &[3 * 3 * 5 * 7]);
        let w = graph.parameter("w", &[5 * 3 * 2 * 3]);
        let y = graph.conv2d(x, w, 3, 3, 5, 7, 5, 2, 3, 1, 0);
        let loss = graph.sum_all(y);
        graph.set_outputs(vec![loss]);
        let plan = super::super::compile(&crate::autodiff::differentiate(&graph));
        let index = plan
            .dispatches
            .iter()
            .position(|d| {
                matches!(
                    d.shader,
                    ShaderEntry::Conv2dGradWeightGemm
                        | ShaderEntry::Conv2dGradWeightGemmSmall
                        | ShaderEntry::Conv2dGradWeightGemm16
                )
            })
            .unwrap();
        (plan, index)
    }

    #[test]
    fn split_sequence_preserves_logical_output_and_provenance() {
        let (mut plan, index) = plan();
        let original = plan.clone();
        let output = original.dispatches[index].output_buffer;
        let bytes = original.buffers[output.0 as usize] * 3;
        assert_eq!(
            plan.split_conv_weight_gradients(&[(index, 3)], bytes),
            Ok(bytes)
        );
        assert_eq!(plan.buffers.len(), original.buffers.len() + 1);
        assert_eq!(plan.buffers.last(), Some(&bytes));
        assert_eq!(plan.dispatches.len(), original.dispatches.len() + 1);
        let a = &plan.dispatches[index];
        let b = &plan.dispatches[index + 1];
        assert_eq!(a.workgroups[2], 3);
        assert_eq!(a.input_buffers, original.dispatches[index].input_buffers);
        assert_eq!(a.params, original.dispatches[index].params);
        assert!(TuneClass::from_dispatch(a, None).is_none());
        assert_eq!(b.shader, ShaderEntry::SumRows);
        assert_eq!(b.input_buffers, [a.output_buffer]);
        assert_eq!(b.output_buffer, output);
        for dispatch in [a, b] {
            assert_eq!(dispatch.origin, original.dispatches[index].origin);
            assert_eq!(
                dispatch.requires_full_precision,
                original.dispatches[index].requires_full_precision
            );
        }
        assert_eq!(plan.node_buffers, original.node_buffers);
        assert_eq!(plan.param_grad_pairs, original.param_grad_pairs);
        assert_eq!(plan.output_buffers, original.output_buffers);
        assert_eq!(plan.loss_buffer, original.loss_buffer);
        let restored: ExecutionPlan =
            serde_json::from_value(serde_json::to_value(&plan).unwrap()).unwrap();
        assert_eq!(restored.dispatches, plan.dispatches);
    }

    fn rejected(mut plan: ExecutionPlan, selections: &[(usize, u32)], cap: usize) {
        let before = serde_json::to_value(&plan).unwrap();
        assert!(plan.split_conv_weight_gradients(selections, cap).is_err());
        assert_eq!(serde_json::to_value(&plan).unwrap(), before);
    }

    #[test]
    fn all_selections_are_checked_before_any_plan_change() {
        let (mut plan, index) = plan();
        let mut second = plan.dispatches[index].clone();
        let size = plan.buffers[second.output_buffer.0 as usize];
        second.output_buffer = BufferRef(plan.buffers.len() as u32);
        plan.buffers.push(size);
        let other = plan.dispatches.len();
        plan.dispatches.push(second);
        for selections in [
            vec![(index, 3), (index, 2)],
            vec![(index, 3), (other, 1)],
            vec![(index, 3), (usize::MAX, 3)],
        ] {
            rejected(plan.clone(), &selections, usize::MAX);
        }
        rejected(plan.clone(), &[(index, 3), (other, 3)], 6 * size - 1);
        let mut reverse = plan.clone();
        assert_eq!(
            plan.split_conv_weight_gradients(&[(index, 3), (other, 3)], 6 * size),
            Ok(6 * size)
        );
        reverse
            .split_conv_weight_gradients(&[(other, 3), (index, 3)], 6 * size)
            .unwrap();
        assert_eq!(
            serde_json::to_value(plan).unwrap(),
            serde_json::to_value(reverse).unwrap()
        );
    }

    #[test]
    fn split_legality_rejects_bad_geometry_capacities_and_modifiers() {
        let (plan, index) = plan();
        for splits in [0, 1, 5, 65_536, u32::MAX] {
            rejected(plan.clone(), &[(index, splits)], usize::MAX);
        }
        rejected(plan.clone(), &[(index, 2)], 0);
        for change in [
            |d: &mut Dispatch| d.kernel = crate::compile::Kernel::Cooperative,
            |d: &mut Dispatch| d.workgroups[2] = 2,
            |d: &mut Dispatch| d.params[6] = 0,
            |d: &mut Dispatch| d.input_buffers[0] = d.output_buffer,
            |d: &mut Dispatch| d.output_buffer = BufferRef(u32::MAX),
        ] {
            let mut changed = plan.clone();
            change(&mut changed.dispatches[index]);
            rejected(changed, &[(index, 2)], usize::MAX);
        }
        let mut small = plan.clone();
        small.buffers[plan.dispatches[index].output_buffer.0 as usize] -= 4;
        rejected(small, &[(index, 2)], usize::MAX);
        let mut unchanged = plan.clone();
        assert_eq!(unchanged.split_conv_weight_gradients(&[], 0), Ok(0));
        assert_eq!(
            serde_json::to_value(unchanged).unwrap(),
            serde_json::to_value(plan).unwrap()
        );
    }

    #[test]
    fn balanced_tile_partitions_cover_uneven_k_without_overflow() {
        for k in [17u32, 31, 32, 33, 41, 60, 65_537, 12_544, u32::MAX - 15] {
            let tiles = k.div_ceil(16);
            for splits in [2, 3, 7, 16, 65_535].into_iter().filter(|&s| s <= tiles) {
                let mut previous = 0u32;
                for split in 0..splits {
                    let per_split = tiles / splits;
                    let extra = tiles % splits;
                    let first = split.checked_mul(per_split).unwrap() + split.min(extra);
                    let last = first + per_split + u32::from(split < extra);
                    let start = first.checked_mul(16).unwrap();
                    let end = last.checked_mul(16).unwrap().min(k);
                    assert_eq!(start, previous);
                    assert!(end > start);
                    previous = end;
                }
                assert_eq!(previous, k);
            }
        }
    }

    #[test]
    fn partial_indices_and_final_reduction_geometry_are_bounded() {
        for (channels, width, splits, error) in [
            (
                2048,
                32,
                2,
                "split-K final reduction exceeds portable dispatch limits",
            ),
            (1024, 1_048_560, 65_535, "split-K partial index overflow"),
        ] {
            let (mut plan, index) = plan();
            let d = &mut plan.dispatches[index];
            d.shader = ShaderEntry::Conv2dGradWeightGemmSmall;
            d.params = vec![1, channels, 1, width, channels, 1, 1, 1, 0, 1, width, 0];
            d.workgroups = [channels.div_ceil(32), channels.div_ceil(32), 1];
            for &buffer in &d.input_buffers {
                plan.buffers[buffer.0 as usize] = channels as usize * width as usize * 4;
            }
            plan.buffers[d.output_buffer.0 as usize] = channels as usize * channels as usize * 4;
            let before = serde_json::to_value(&plan).unwrap();
            assert_eq!(
                plan.split_conv_weight_gradients(&[(index, splits)], usize::MAX),
                Err(TuneError(error))
            );
            assert_eq!(serde_json::to_value(&plan).unwrap(), before);
        }
    }
}
