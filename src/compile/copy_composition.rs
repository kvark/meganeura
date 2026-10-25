use super::{BufferRef, Dispatch, ExecutionPlan, Kernel, ShaderEntry};
use crate::{
    DType,
    codegen::{CopyKernel, CopyNode, CopySource},
    schedule::{PointwiseDAG, ScalarStorage},
};
use std::collections::{BTreeSet, HashMap, HashSet};

type Source = (BufferRef, Option<usize>);

fn plain(d: &Dispatch, cast: bool) -> bool {
    let shader = if cast {
        matches!(d.shader, ShaderEntry::ToF16 | ShaderEntry::ToBF16)
    } else {
        d.shader == ShaderEntry::Concat
    };
    shader
        && d.kernel == Kernel::Default
        && d.extra_outputs.is_empty()
        && d.horizontal_batch <= 1
        && !d.schedule_locked
        && (cast || !d.fusion_barrier)
        && d.scalar_storage == ScalarStorage::default()
        && d.matmul_prologue.is_none()
        && d.matmul_epilogue.is_none()
        && d.gemv_rmsnorm.is_none()
        && d.input_buffers.len() == if cast { 1 } else { 2 }
        && d.params.len() == 4
        && elements(d, cast).is_some()
}

fn elements(d: &Dispatch, cast: bool) -> Option<u32> {
    let p = &d.params;
    if p.len() != 4 {
        return None;
    }
    if cast {
        return (p[0] > 0).then_some(p[0]);
    }
    if p.contains(&0) {
        return None;
    }
    p[1].checked_add(p[2])?.checked_mul(p[0])?.checked_mul(p[3])
}

struct Uses {
    readers: HashMap<BufferRef, BTreeSet<usize>>,
    writers: HashMap<BufferRef, Vec<usize>>,
    protected: HashSet<BufferRef>,
    inputs: Vec<Vec<Source>>,
    at_endpoint: HashMap<BufferRef, usize>,
}

impl Uses {
    fn new(ds: &[Option<Dispatch>], protected: &HashSet<BufferRef>, endpoint: usize) -> Self {
        let mut result = Self {
            readers: HashMap::new(),
            writers: HashMap::new(),
            protected: protected.clone(),
            inputs: vec![Vec::new(); ds.len()],
            at_endpoint: HashMap::new(),
        };
        let mut versions = HashMap::new();
        for (i, d) in ds
            .iter()
            .enumerate()
            .filter_map(|(i, d)| d.as_ref().map(|d| (i, d)))
        {
            if i == endpoint {
                result.at_endpoint = versions.clone();
            }
            result.inputs[i] = d
                .input_buffers
                .iter()
                .map(|&b| (b, versions.get(&b).copied()))
                .collect();
            let mut hidden = Vec::new();
            if let Some(p) = d.matmul_prologue.as_ref() {
                hidden.extend(p.factors.iter().map(|&(b, _)| b));
            }
            if let Some(p) = d.matmul_epilogue.as_ref() {
                hidden.extend(p.inputs.iter().map(|&(b, _)| b));
            }
            if let Some(p) = d.gemv_rmsnorm.as_ref() {
                hidden.push(p.weight);
            }
            result.protected.extend(hidden.iter().copied());
            for b in d.input_buffers.iter().chain(&hidden) {
                result.readers.entry(*b).or_default().insert(i);
            }
            for b in std::iter::once(&d.output_buffer).chain(&d.extra_outputs) {
                result.writers.entry(*b).or_default().push(i);
                versions.insert(*b, i);
            }
        }
        result
    }

    fn leaves(&self, endpoint: usize, selected: &BTreeSet<usize>) -> Vec<Source> {
        fn visit(
            i: usize,
            uses: &Uses,
            selected: &BTreeSet<usize>,
            seen: &mut HashSet<usize>,
            out: &mut Vec<Source>,
        ) {
            if !seen.insert(i) {
                return;
            }
            for &source in &uses.inputs[i] {
                if let Some(child) = source.1.filter(|j| selected.contains(j)) {
                    visit(child, uses, selected, seen, out);
                } else if !out.contains(&source) {
                    out.push(source);
                }
            }
        }
        let mut out = Vec::new();
        visit(endpoint, self, selected, &mut HashSet::new(), &mut out);
        out
    }
}

pub(super) fn fold(plan: &mut ExecutionPlan) {
    let protected = plan.externally_visible_buffers();
    let mut ds: Vec<_> = std::mem::take(&mut plan.dispatches)
        .into_iter()
        .map(Some)
        .collect();
    let mut changed = false;
    for endpoint in (0..ds.len()).rev() {
        let Some(end) = ds[endpoint].as_ref() else {
            continue;
        };
        let cast = plain(end, true);
        if !cast && !plain(end, false) {
            continue;
        }
        let output = end.output_buffer;
        let uses = Uses::new(&ds, &protected, endpoint);
        let mut selected = BTreeSet::from([endpoint]);
        loop {
            let mut expanded = false;
            for (buffer, producer) in uses.leaves(endpoint, &selected) {
                let Some(i) = producer.filter(|&i| i < endpoint) else {
                    continue;
                };
                let Some(d) = ds[i].as_ref() else {
                    continue;
                };
                if !plain(d, false)
                    || uses.protected.contains(&buffer)
                    || uses.writers[&buffer].len() != 1
                    || !uses.readers[&buffer].is_subset(&selected)
                {
                    continue;
                }
                selected.insert(i);
                let sources = uses.leaves(endpoint, &selected);
                let valid = sources.len() <= PointwiseDAG::MAX_INPUTS as usize
                    && sources.iter().map(|s| s.0).collect::<HashSet<_>>().len() == sources.len()
                    && sources
                        .iter()
                        .all(|&(b, v)| b != output && uses.at_endpoint.get(&b).copied() == v);
                if valid {
                    expanded = true;
                    break;
                }
                selected.remove(&i);
            }
            if !expanded {
                break;
            }
        }
        if selected.len() == 1 {
            continue;
        }
        let sources = uses.leaves(endpoint, &selected);
        let mut nodes = Vec::new();
        let mut refs = HashMap::new();
        for &i in &selected {
            let d = ds[i].as_ref().unwrap();
            let inputs: Vec<_> = uses.inputs[i]
                .iter()
                .map(|source| {
                    if let Some(child) = source.1.filter(|j| selected.contains(j)) {
                        refs[&child]
                    } else {
                        CopySource::Input(sources.iter().position(|s| s == source).unwrap() as u8)
                    }
                })
                .collect();
            if i == endpoint && cast {
                refs.insert(i, inputs[0]);
            } else {
                refs.insert(i, CopySource::Node(nodes.len() as u32));
                nodes.push(CopyNode {
                    inputs: [inputs[0], inputs[1]],
                    widths: [d.params[1], d.params[2]],
                    inner: d.params[3],
                });
            }
        }
        let end = ds[endpoint].as_mut().unwrap();
        let storage = match end.shader {
            ShaderEntry::ToBF16 => DType::BF16,
            ShaderEntry::ToF16 => DType::F16,
            _ => DType::F32,
        };
        let len = elements(end, cast).unwrap();
        end.kernel = Kernel::Copy(CopyKernel {
            n_inputs: sources.len() as u8,
            nodes,
            root: refs[&endpoint],
            storage,
        });
        end.shader = ShaderEntry::Generated;
        end.input_buffers = sources.into_iter().map(|s| s.0).collect();
        end.params = vec![len, 0, 0, 0];
        end.workgroups = [
            len.div_ceil(if storage == DType::F16 { 512 } else { 256 }),
            1,
            1,
        ];
        for i in selected {
            if i != endpoint {
                ds[i] = None;
            }
        }
        changed = true;
    }
    plan.dispatches = ds.into_iter().flatten().collect();
    if changed {
        plan.groups.clear();
    }
}
