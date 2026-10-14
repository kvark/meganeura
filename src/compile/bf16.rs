//! Fold explicit conversion boundaries, never choose a precision policy.

use super::{ExecutionPlan, Kernel, ShaderEntry};
use crate::schedule::PointwiseStorage;
use std::collections::HashMap;

/// Fill a small output grid without making each partition mostly staging work.
pub(super) fn matrix_splits(m: u32, n: u32, k: u32) -> u32 {
    let groups = m.div_ceil(32).saturating_mul(n.div_ceil(32)).max(1);
    if k < 1024 || groups >= 128 {
        return 1;
    }
    128u32.div_ceil(groups).min(k / 256).min(16)
}

pub(super) fn fuse_boundaries(plan: &mut ExecutionPlan) {
    let protected = plan.externally_visible_buffers();
    loop {
        let mut producers = HashMap::new();
        let mut reads = HashMap::<_, usize>::new();
        for (index, dispatch) in plan.dispatches.iter().enumerate() {
            producers.insert(dispatch.output_buffer, index);
            for &buffer in &dispatch.input_buffers {
                *reads.entry(buffer).or_default() += 1;
            }
        }
        let mut remove = None;
        let mut changed = false;
        for ci in 0..plan.dispatches.len() {
            let consumer = &plan.dispatches[ci];
            // A materialization boundary stays a boundary. A dispatch this
            // pass already typed is allowed to absorb its other conversions.
            if consumer.pointwise().is_some()
                && (!consumer.fusion_barrier
                    || consumer.pointwise_storage != PointwiseStorage::default())
            {
                for (slot, &buffer) in consumer.input_buffers.iter().enumerate() {
                    let Some(&pi) = producers.get(&buffer) else {
                        continue;
                    };
                    let producer = &plan.dispatches[pi];
                    if !matches!(producer.shader, ShaderEntry::FromBF16) {
                        continue;
                    }
                    let input = producer.input_buffers[0];
                    let origin = producer.origin.clone();
                    let consumer = &mut plan.dispatches[ci];
                    consumer.input_buffers[slot] = input;
                    consumer.pointwise_storage.bf16_inputs |= 1 << slot;
                    consumer.fusion_barrier = true;
                    consumer.origin.extend(origin);
                    changed = true;
                    if !protected.contains(&buffer) && reads[&buffer] == 1 {
                        remove = Some(pi);
                    }
                    break;
                }
            }
            if changed {
                break;
            }
            let consumer = &plan.dispatches[ci];
            if !matches!(consumer.shader, ShaderEntry::ToBF16) {
                continue;
            }
            let buffer = consumer.input_buffers[0];
            if protected.contains(&buffer) || reads[&buffer] != 1 {
                continue;
            }
            let Some(&pi) = producers.get(&buffer) else {
                continue;
            };
            let output = consumer.output_buffer;
            let origin = consumer.origin.clone();
            let producer = &mut plan.dispatches[pi];
            if producer.pointwise().is_some()
                && (!producer.fusion_barrier
                    || producer.pointwise_storage != PointwiseStorage::default())
            {
                producer.pointwise_storage.bf16_output = true;
            } else if let Kernel::CooperativeBF16 {
                ref mut output_bf16,
                splits: 1,
            } = producer.kernel
            {
                assert!(!*output_bf16);
                *output_bf16 = true;
            } else if let ShaderEntry::Conv2dBF16 {
                ref mut output_bf16,
                ..
            } = producer.shader
            {
                assert!(!*output_bf16);
                *output_bf16 = true;
            } else {
                continue;
            }
            producer.output_buffer = output;
            producer.fusion_barrier = true;
            producer.origin.extend(origin);
            remove = Some(ci);
            break;
        }
        match remove {
            Some(index) => {
                plan.dispatches.remove(index);
            }
            None if !changed => break,
            None => (),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrix_partitions_keep_short_and_occupied_products_unsplit() {
        for (m, n, k, expected) in [
            (8, 256, 1023, 1),
            (8, 256, 1024, 4),
            (8, 256, 2048, 8),
            (8, 256, 4096, 16),
            (8, 256, 65536, 16),
            (128, 256, 2560, 4),
            (512, 256, 2560, 1),
            (33, 65, 1031, 4),
            (u32::MAX, u32::MAX, 2048, 1),
        ] {
            assert_eq!(matrix_splits(m, n, k), expected, "{m}x{n}x{k}");
        }
    }
}
