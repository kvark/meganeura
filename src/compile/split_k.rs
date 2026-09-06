use super::{BufferRef, Dispatch, ExecutionPlan, ShaderEntry};
use crate::tune::{MatmulTile, TuneClass, TuneError};

impl ExecutionPlan {
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
                .filter(|class| class.shader == ShaderEntry::Conv2dGradWeightGemm)
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
            producer.shader = if tile == MatmulTile::Tile32 {
                ShaderEntry::Conv2dGradWeightGemmSplitSmall
            } else {
                ShaderEntry::Conv2dGradWeightGemmSplit
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
                    ShaderEntry::Conv2dGradWeightGemm | ShaderEntry::Conv2dGradWeightGemmSmall
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
            |d: &mut Dispatch| d.use_coop = true,
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
