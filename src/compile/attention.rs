use super::{BufferRef, CachedBlockAttentionParams, Dispatch, ExecutionPlan, ShaderEntry};
use crate::tune::{TuneAttention, TuneError};

impl TuneAttention {
    pub(crate) fn from_dispatch(d: &Dispatch) -> Option<Self> {
        let CachedBlockAttentionParams {
            window_size,
            num_heads,
            num_kv_heads,
            head_dim,
            block_len,
            max_seq,
            ..
        } = CachedBlockAttentionParams::from_words(&d.params)?;
        if [num_heads, num_kv_heads, head_dim, block_len, max_seq].contains(&0)
            || !num_heads.is_multiple_of(num_kv_heads)
            || head_dim > 512
            || block_len > max_seq
            || num_heads > 0xFFFF
            || block_len > 0xFFFF
            || max_seq > u32::MAX / 2
        {
            return None;
        }
        let last = max_seq - block_len;
        let mut positions = vec![0, last / 2, last];
        positions.dedup();
        let key = Self {
            window_size,
            num_heads,
            num_kv_heads,
            head_dim,
            block_len,
            max_seq,
            positions,
            device_local: [false; 7],
            binding_bytes: Vec::new(),
        };
        key.sizes(1)?;
        Some(key)
    }

    pub(crate) fn sizes(&self, splits: u32) -> Option<Vec<usize>> {
        if !(1..=16).contains(&splits) {
            return None;
        }
        let bytes = |elements: u32| usize::try_from(elements).ok()?.checked_mul(4);
        let query = bytes(
            self.block_len
                .checked_mul(self.num_heads)?
                .checked_mul(self.head_dim)?,
        )?;
        let cache = bytes(
            self.max_seq
                .checked_mul(self.num_kv_heads)?
                .checked_mul(self.head_dim)?,
        )?;
        let partials = if splits == 1 {
            4
        } else {
            bytes(
                self.block_len
                    .checked_mul(self.num_heads)?
                    .checked_mul(splits)?
                    .checked_mul(self.head_dim.checked_add(2)?)?,
            )?
        };
        Some(vec![query, cache, cache, 4, 4, query, partials])
    }

    pub(crate) fn sequence(
        &self,
        source: &Dispatch,
        output: BufferRef,
        partials: BufferRef,
        splits: u32,
    ) -> Vec<Dispatch> {
        let old_entry = format!("{:?}", source.shader);
        let label = |entry: &ShaderEntry| match source.label.rsplit_once(&old_entry) {
            Some((prefix, suffix)) => format!("{prefix}{entry:?}{suffix}"),
            None => format!("{entry:?}"),
        };
        let mut first = source.clone();
        let mut params = CachedBlockAttentionParams::from_words(&source.params)
            .expect("qualified attention parameters");
        params.splits = if splits == 1 { 0 } else { splits };
        params.chunk = if splits == 1 {
            0
        } else {
            self.max_seq.div_ceil(splits)
        };
        first.params = params.to_words();
        first.output_buffer = output;
        first.shader = ShaderEntry::CachedBlockAttention;
        first.label = label(&first.shader);
        first.workgroups = [self.block_len, self.num_heads, 1];
        if splits == 1 {
            return vec![first];
        }
        first.shader = ShaderEntry::CachedBlockAttentionSplit;
        first.label = label(&first.shader);
        first.workgroups = [self.block_len, splits, self.num_heads];
        first.output_buffer = partials;
        let mut combine = first.clone();
        combine.shader = ShaderEntry::CachedBlockAttentionCombine;
        combine.label = label(&combine.shader);
        combine.input_buffers = vec![partials];
        combine.output_buffer = output;
        combine.workgroups = [self.block_len, self.num_heads, 1];
        vec![first, combine]
    }
}

fn plain(d: &Dispatch) -> bool {
    matches!(*d, Dispatch {
        shader: _,
        workgroups: _,
        input_buffers: _,
        output_buffer: _,
        extra_outputs: ref outputs,
        params: _,

        kernel: crate::compile::Kernel::Default,
        horizontal_batch: 0 | 1,
        requires_full_precision: _,
        fusion_barrier: _,
        matmul_epilogue: None,
        gemv_rmsnorm: None,
        matmul_prologue: None,
        label: _,
        origin: _,
        weight_format: crate::compile::WeightFormat::F32,
    } if outputs.is_empty())
}

pub(crate) struct AttentionSequence {
    pub first: usize,
    pub combine: Option<usize>,
    pub initial: u32,
    pub key: TuneAttention,
}

impl AttentionSequence {
    pub fn bindings(&self, plan: &ExecutionPlan) -> Vec<BufferRef> {
        let mut bindings = plan.dispatches[self.first].input_buffers.clone();
        bindings.push(plan.dispatches[self.combine.unwrap_or(self.first)].output_buffer);
        if self.combine.is_some() {
            bindings.push(plan.dispatches[self.first].output_buffer);
        }
        bindings
    }
}

impl ExecutionPlan {
    pub(crate) fn attention_sequences(&self) -> Vec<AttentionSequence> {
        let mut sequences = Vec::new();
        for (first, d) in self.dispatches.iter().enumerate() {
            if !matches!(
                d.shader,
                ShaderEntry::CachedBlockAttention | ShaderEntry::CachedBlockAttentionSplit
            ) || d.input_buffers.len() != 5
                || !plain(d)
            {
                continue;
            }
            let Some(key) = TuneAttention::from_dispatch(d) else {
                continue;
            };
            let combine = if d.shader == ShaderEntry::CachedBlockAttentionSplit {
                let consumers: Vec<_> = self
                    .dispatches
                    .iter()
                    .enumerate()
                    .filter(|&(_, c)| c.input_buffers.contains(&d.output_buffer))
                    .collect();
                let &[(index, c)] = consumers.as_slice() else {
                    continue;
                };
                if index <= first
                    || c.shader != ShaderEntry::CachedBlockAttentionCombine
                    || c.input_buffers != [d.output_buffer]
                    || c.params != d.params
                    || !plain(c)
                    || c.workgroups != [key.block_len, key.num_heads, 1]
                    || self.output_buffers.contains(&d.output_buffer)
                    || self
                        .input_buffers
                        .iter()
                        .chain(&self.param_buffers)
                        .any(|entry| entry.1 == d.output_buffer)
                    || self
                        .constant_buffers
                        .iter()
                        .any(|entry| entry.0 == d.output_buffer)
                    || self.loss_buffer == Some(d.output_buffer)
                    || self
                        .param_grad_pairs
                        .iter()
                        .any(|&(a, b)| a == d.output_buffer || b == d.output_buffer)
                {
                    continue;
                }
                Some(index)
            } else {
                None
            };
            let initial = if combine.is_some() {
                CachedBlockAttentionParams::from_words(&d.params)
                    .unwrap()
                    .splits
            } else {
                1
            };
            if !(1..=16).contains(&initial) {
                continue;
            }
            let output = self.dispatches[combine.unwrap_or(first)].output_buffer;
            let expected = key.sequence(d, output, d.output_buffer, initial);
            if d.workgroups != expected[0].workgroups || d.params != expected[0].params {
                continue;
            }
            let mut sequence = AttentionSequence {
                first,
                combine,
                initial,
                key,
            };
            let bindings = sequence.bindings(self);
            if bindings
                .iter()
                .enumerate()
                .any(|(i, b)| bindings[..i].contains(b))
            {
                continue;
            }
            let Some(sizes) = bindings
                .iter()
                .map(|b| self.buffers.get(b.0 as usize).copied())
                .collect::<Option<Vec<_>>>()
            else {
                continue;
            };
            if sequence
                .key
                .sizes(initial)
                .unwrap()
                .iter()
                .zip(&sizes)
                .any(|(need, have)| need > have)
            {
                continue;
            }
            sequence.key.binding_bytes = sizes[..6].to_vec();
            sequences.push(sequence);
        }
        sequences
    }

    /// Lower cached-attention choices before scheduling and allocation.
    ///
    /// Selections are (producer dispatch index, split count) in the current plan.
    /// One means an unsplit kernel. The budget covers the total logical partial
    /// capacity of selected sequences, before lifetime aliasing. Every selection
    /// is checked before changing the plan. This neither measures nor qualifies
    /// a winner; compare complete plans under the caller's numerical contract.
    pub fn set_attention_splits(
        &mut self,
        selections: &[(usize, u32)],
        max_partial_bytes: usize,
    ) -> Result<usize, TuneError> {
        let sequences = self.attention_sequences();
        let mut selections = selections.to_vec();
        selections.sort_unstable();
        if selections.windows(2).any(|p| p[0].0 == p[1].0) {
            return Err(TuneError("duplicate attention dispatch selection"));
        }
        let mut replacements = std::collections::HashMap::new();
        let mut capacities = Vec::new();
        let mut added = 0;
        let mut bytes = 0usize;
        for (first, splits) in selections {
            let sequence = sequences
                .iter()
                .find(|s| s.first == first)
                .ok_or(TuneError("attention selection is not a legal sequence"))?;
            let size = sequence
                .key
                .sizes(splits)
                .ok_or(TuneError("attention split count must be 1..16"))?[6];
            bytes = bytes
                .checked_add(if splits == 1 { 0 } else { size })
                .filter(|&b| b <= max_partial_bytes)
                .ok_or(TuneError("attention partial byte budget exceeded"))?;
            let source = &self.dispatches[first];
            let output = self.dispatches[sequence.combine.unwrap_or(first)].output_buffer;
            let partial = if sequence.combine.is_some() {
                capacities.push((source.output_buffer, if splits == 1 { 0 } else { size }));
                source.output_buffer
            } else if splits == 1 {
                output
            } else {
                let index = self
                    .buffers
                    .len()
                    .checked_add(added)
                    .and_then(|i| u32::try_from(i).ok())
                    .ok_or(TuneError("attention buffer index overflow"))?;
                added += 1;
                let partial = BufferRef(index);
                capacities.push((partial, size));
                partial
            };
            replacements.insert(
                first,
                sequence.key.sequence(source, output, partial, splits),
            );
            if let Some(combine) = sequence.combine {
                replacements.insert(combine, Vec::new());
            }
        }
        self.buffers.resize(self.buffers.len() + added, 0);
        for (buffer, bytes) in capacities {
            self.buffers[buffer.0 as usize] = bytes;
        }
        self.dispatches = self
            .dispatches
            .drain(..)
            .enumerate()
            .flat_map(|(i, d)| replacements.remove(&i).unwrap_or_else(|| vec![d]))
            .collect();
        Ok(bytes)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn attention_lowering_checks_before_changing_plan() {
        let mut graph = crate::Graph::new();
        let q = graph.input("q", &[1, 8]);
        let k = graph.parameter("k", &[96, 4]);
        let v = graph.parameter("v", &[96, 4]);
        let position = graph.input_u32("position", &[1]);
        let valid = graph.input_u32("valid", &[1]);
        let output = graph.cached_block_attention(q, k, v, position, valid, 2, 1, 4, 37);
        graph.set_outputs(vec![output]);
        let mut plan = crate::compile::compile(&graph);
        let before = serde_json::to_value(&plan).unwrap();
        for (selections, bytes) in [
            (vec![(0, 16)], 1),
            (vec![(0, 2), (0, 4)], 10000),
            (vec![(0, 2), (99, 2)], 10000),
            (vec![(0, 17)], 10000),
        ] {
            assert!(plan.set_attention_splits(&selections, bytes).is_err());
            assert_eq!(serde_json::to_value(&plan).unwrap(), before);
        }
        let outputs = plan.output_buffers.clone();
        for splits in [1, 16, 2, 1] {
            plan.set_attention_splits(&[(0, splits)], 10000).unwrap();
            let sequences = plan.attention_sequences();
            assert_eq!(sequences.len(), 1);
            assert_eq!(sequences[0].initial, splits);
            assert_eq!(plan.dispatches.len(), if splits == 1 { 1 } else { 2 });
            assert_eq!(plan.output_buffers, outputs);
        }
    }
}
