//! Fold explicit conversion boundaries, never choose a precision policy.

use super::{ExecutionPlan, Kernel, ShaderEntry};
use crate::schedule::PointwiseStorage;
use std::collections::HashMap;

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
                producer.pointwise_storage.output = crate::DType::BF16;
            } else if let Kernel::CooperativeTyped {
                ref mut types,
                splits: 1,
            } = producer.kernel
            {
                assert_eq!(types.output, crate::DType::F32);
                types.output = crate::DType::BF16;
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
