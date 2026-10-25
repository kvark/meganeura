use super::{BufferRef, Dispatch, ExecutionPlan, Kernel, ShaderEntry};
use crate::{DType, codegen::ImageView};
use std::collections::{BTreeSet, HashMap, HashSet};

struct Uses {
    readers: HashMap<BufferRef, BTreeSet<usize>>,
    writers: HashMap<BufferRef, Vec<usize>>,
    protected: HashSet<BufferRef>,
}

impl Uses {
    fn new(plan: &ExecutionPlan) -> Self {
        let mut result = Self {
            readers: HashMap::new(),
            writers: HashMap::new(),
            protected: plan.externally_visible_buffers(),
        };
        for (i, d) in plan.dispatches.iter().enumerate() {
            let mut reads = d.input_buffers.clone();
            if let Some(p) = d.matmul_prologue.as_ref() {
                for &(buffer, _) in &p.factors {
                    reads.push(buffer);
                    result.protected.insert(buffer);
                }
            }
            if let Some(p) = d.matmul_epilogue.as_ref() {
                for &(buffer, _) in &p.inputs {
                    reads.push(buffer);
                    result.protected.insert(buffer);
                }
            }
            if let Some(p) = d.gemv_rmsnorm.as_ref() {
                reads.push(p.weight);
                result.protected.insert(p.weight);
            }
            for buffer in reads {
                result.readers.entry(buffer).or_default().insert(i);
            }
            for buffer in std::iter::once(d.output_buffer).chain(d.extra_outputs.iter().copied()) {
                result.writers.entry(buffer).or_default().push(i);
            }
        }
        result
    }

    fn private(&self, buffer: BufferRef) -> bool {
        !self.protected.contains(&buffer)
            && self.writers.get(&buffer).is_some_and(|w| w.len() == 1)
            && self
                .readers
                .get(&buffer)
                .is_none_or(|readers| readers.iter().all(|&r| r > self.writers[&buffer][0]))
    }

    fn one(&self, buffer: BufferRef) -> Option<usize> {
        let readers = self.readers.get(&buffer)?;
        if readers.len() == 1 {
            readers.first().copied()
        } else {
            None
        }
    }
}

fn unary(d: &Dispatch, shader: ShaderEntry) -> bool {
    d.shader == shader
        && d.kernel == Kernel::Default
        && d.input_buffers.len() == 1
        && d.extra_outputs.is_empty()
        && d.horizontal_batch <= 1
        && d.matmul_prologue.is_none()
        && d.matmul_epilogue.is_none()
        && d.gemv_rmsnorm.is_none()
        && !d.schedule_locked
}

fn elements(shape: &[u32]) -> Option<u64> {
    shape
        .iter()
        .try_fold(1u64, |n, &v| n.checked_mul(u64::from(v)))
}

fn packed_output(
    plan: &ExecutionPlan,
    uses: &Uses,
    input: BufferRef,
    shape: [u32; 3],
) -> Option<(BufferRef, Vec<usize>)> {
    if !uses.private(input) {
        return None;
    }
    let readers: Vec<_> = uses.readers.get(&input)?.iter().copied().collect();
    if readers.is_empty() {
        return None;
    }
    let [batch, channels, pixels] = shape;
    let padded = channels.div_ceil(32) * 32;
    let elements = elements(&[batch, padded, pixels])?;
    if elements > u64::from(u32::MAX) {
        return None;
    }
    let shader = ShaderEntry::PackChannels {
        dtype: DType::BF16,
        transpose: false,
    };
    if !readers.iter().all(|&i| {
        let d = &plan.dispatches[i];
        unary(d, shader.clone())
            && d.params == [batch, channels, pixels, padded]
            && uses.private(d.output_buffer)
            && plan.buffers[d.output_buffer.0 as usize] as u64 == elements * 2
    }) {
        return None;
    }
    Some((plan.dispatches[readers[0]].output_buffer, readers))
}

pub(super) fn fold(plan: &mut ExecutionPlan) {
    let uses = Uses::new(plan);
    let mut remove = BTreeSet::new();
    let mut replacements = HashMap::new();
    let mut aliases = HashMap::new();
    for (i, original) in plan.dispatches.iter().enumerate() {
        let Kernel::Image(ref kernel) = original.kernel else {
            continue;
        };
        if kernel.view.is_some() || original.params.len() != 8 || original.schedule_locked {
            continue;
        }
        let p = &original.params;
        let Some(input_pixels) = p[2].checked_mul(p[3]) else {
            continue;
        };
        let Some(output_pixels) = p[4].checked_mul(p[5]) else {
            continue;
        };
        if p[0] == 0 || p[1] != kernel.channels || input_pixels == 0 || output_pixels == 0 {
            continue;
        }
        let Some(output_elements) = elements(&[p[0], p[1], output_pixels]) else {
            continue;
        };
        let mut d = original.clone();
        let mut view = ImageView {
            storage: kernel.storage,
            packed: false,
            expansion_width: 0,
            reduction_width: 0,
        };
        let mut drop = BTreeSet::new();
        let mut new_aliases = Vec::new();
        if !kernel.backward {
            if !uses.private(d.output_buffer) {
                continue;
            }
            let Some(mut next) = uses.one(d.output_buffer) else {
                continue;
            };
            let mut pixels = output_pixels;
            if unary(&plan.dispatches[next], ShaderEntry::Upsample2x) {
                let up = &plan.dispatches[next];
                let Some(expanded) = pixels.checked_mul(4) else {
                    continue;
                };
                if up.params.len() != 4
                    || up.params[3] == 0
                    || !pixels.is_multiple_of(up.params[3])
                    || elements(&up.params) != Some(output_elements)
                    || !uses.private(up.output_buffer)
                {
                    continue;
                }
                let Some(cast) = uses.one(up.output_buffer) else {
                    continue;
                };
                view.expansion_width = up.params[3];
                drop.insert(next);
                next = cast;
                pixels = expanded;
            }
            let cast = &plan.dispatches[next];
            let Some(elements) = elements(&[p[0], p[1], pixels]) else {
                continue;
            };
            if !unary(cast, ShaderEntry::ToBF16)
                || !uses.private(cast.output_buffer)
                || elements > u64::from(u32::MAX)
                || cast.params != [elements as u32, 0, 0, 0]
                || plan.buffers[cast.output_buffer.0 as usize] as u64 != elements * 2
            {
                continue;
            }
            d.output_buffer = cast.output_buffer;
            view.storage = DType::BF16;
            drop.insert(next);
            if let Some((target, packs)) =
                packed_output(plan, &uses, d.output_buffer, [p[0], p[1], pixels])
            {
                for pack in packs {
                    drop.insert(pack);
                    let old = plan.dispatches[pack].output_buffer;
                    if old != target {
                        new_aliases.push((old, target));
                    }
                }
                d.output_buffer = target;
                view.packed = true;
            }
        } else {
            if kernel.storage == DType::BF16
                && let Some((target, packs)) =
                    packed_output(plan, &uses, d.output_buffer, [p[0], p[1], input_pixels])
            {
                for pack in packs {
                    drop.insert(pack);
                    let old = plan.dispatches[pack].output_buffer;
                    if old != target {
                        new_aliases.push((old, target));
                    }
                }
                d.output_buffer = target;
                view.packed = true;
            }
            let dy = d.input_buffers[0];
            if !kernel.pool && uses.private(dy) && uses.one(dy) == Some(i) {
                let at = uses.writers[&dy][0];
                let up = &plan.dispatches[at];
                if unary(up, ShaderEntry::Upsample2xGrad)
                    && up.params.len() == 4
                    && up.params[3] > 0
                    && output_pixels.is_multiple_of(up.params[3])
                    && elements(&up.params) == Some(output_elements)
                    && output_elements <= u64::from(u32::MAX) / 4
                    && uses
                        .writers
                        .get(&up.input_buffers[0])
                        .is_none_or(|writers| writers.iter().all(|&w| w < at || w > i))
                {
                    view.reduction_width = up.params[3];
                    d.input_buffers[0] = up.input_buffers[0];
                    drop.insert(at);
                }
            }
        }
        if drop.is_empty() || !remove.is_disjoint(&drop) {
            continue;
        }
        let Kernel::Image(ref mut k) = d.kernel else {
            unreachable!();
        };
        k.view = Some(view);
        replacements.insert(i, d);
        remove.extend(drop);
        aliases.extend(new_aliases);
    }
    let removed = remove.len();
    let changed = replacements.len();
    plan.dispatches = std::mem::take(&mut plan.dispatches)
        .into_iter()
        .enumerate()
        .filter_map(|(i, d)| (!remove.contains(&i)).then(|| replacements.remove(&i).unwrap_or(d)))
        .map(|mut d| {
            for input in &mut d.input_buffers {
                while let Some(&alias) = aliases.get(input) {
                    *input = alias;
                }
            }
            d
        })
        .collect();
    log::info!(
        "image views: {changed} image kernels, {removed} removed dispatches, {} shared packed views",
        aliases.len()
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn fixture() -> Option<(ExecutionPlan, Value)> {
        let directory = std::env::var_os("MEGANEURA_IMAGE_VIEW_FIXTURE")?;
        let directory = std::path::Path::new(&directory);
        let inv: Value = serde_json::from_slice(
            &std::fs::read(
                directory.join("../spatial-value-cost-20261015.AzzHQJ2v/inventory.json"),
            )
            .unwrap(),
        )
        .unwrap();
        let boundary: Value = serde_json::from_slice(
            &std::fs::read(
                directory.join("../spatial-value-cost-20261015.AzzHQJ2v/closed-preflight.json"),
            )
            .unwrap(),
        )
        .unwrap();
        let expected =
            serde_json::from_slice(&std::fs::read(directory.join("view-preflight.json")).unwrap())
                .unwrap();
        let mut graph = crate::Graph::new();
        let y = graph.scalar(0.0);
        graph.set_outputs(vec![y]);
        let mut plan = super::super::compile(&graph);
        plan.buffers = serde_json::from_value(inv["buffers"].clone()).unwrap();
        plan.dispatches = serde_json::from_value(inv["dispatches"].clone()).unwrap();
        plan.output_buffers =
            serde_json::from_value(boundary["protected_buffers"].clone()).unwrap();
        plan.constant_buffers.clear();
        Some((plan, expected))
    }

    #[test]
    fn image_view_actual_plan_matches_independent_proof() {
        let Some((mut plan, proof)) = fixture() else {
            return;
        };
        let original = plan.dispatches.clone();
        let buffers = plan.buffers.clone();
        let visible = plan.externally_visible_buffers();
        let drop: HashSet<usize> =
            serde_json::from_value(proof["removed_dispatches"].clone()).unwrap();
        let aliases: HashMap<u32, u32> = serde_json::from_value(proof["aliases"].clone()).unwrap();
        let changes = proof["modified_images"].as_object().unwrap();
        let mut expected = Vec::new();
        for (i, old) in original.iter().enumerate() {
            if drop.contains(&i) {
                continue;
            }
            let mut d = old.clone();
            if let Some(change) = changes.get(&i.to_string()) {
                let Kernel::Image(ref mut k) = d.kernel else {
                    unreachable!();
                };
                let up = change["upsample"].as_u64().map(|v| v as usize);
                k.view = Some(ImageView {
                    storage: if k.backward { k.storage } else { DType::BF16 },
                    packed: change["layout"] == "NHWC32",
                    expansion_width: if !k.backward {
                        up.map_or(0, |u| original[u].params[3])
                    } else {
                        0
                    },
                    reduction_width: if k.backward {
                        up.map_or(0, |u| original[u].params[3])
                    } else {
                        0
                    },
                });
                d.output_buffer = BufferRef(change["target"][1].as_u64().unwrap() as u32);
                if k.backward
                    && let Some(up) = up
                {
                    d.input_buffers[0] = original[up].input_buffers[0];
                }
            }
            for input in &mut d.input_buffers {
                while let Some(&a) = aliases.get(&input.0) {
                    *input = BufferRef(a);
                }
            }
            expected.push(d);
        }
        fold(&mut plan);
        assert_eq!(plan.dispatches, expected);
        assert_eq!(plan.buffers, buffers);
        assert_eq!(plan.externally_visible_buffers(), visible);
        assert_eq!(original.len() - plan.dispatches.len(), 31);
        println!(
            "13 changed image kernels,31 eliminated dispatches,8 shared views; exact independent plan match"
        );
    }

    #[test]
    fn image_view_protects_observers_and_multiple_writers() {
        let Some((plan, _)) = fixture() else {
            return;
        };
        for kind in 0..4 {
            let mut changed = plan.clone();
            let original = changed.dispatches[123].clone();
            match kind {
                0 => changed.output_buffers.push(original.output_buffer),
                1 => changed.dispatches.push(original.clone()),
                2 => changed
                    .output_buffers
                    .push(changed.dispatches[125].output_buffer),
                _ => {
                    let b = changed.dispatches[125].output_buffer;
                    changed.dispatches[130].matmul_prologue = Some(super::super::MatMulPrologue {
                        factors: vec![(b, super::super::PrologueLoadKind::PerRow)],
                    });
                }
            }
            fold(&mut changed);
            assert!(changed.dispatches.contains(&original), "case {kind}");
        }
        assert_eq!(elements(&[u32::MAX; 4]), None);
    }
}
