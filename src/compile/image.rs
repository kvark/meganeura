use super::{
    BufferRef, Compiler, Dispatch, ExecutionPlan, Kernel, ShaderEntry, pool, rmsnorm_kernel,
};
use crate::{
    DType,
    codegen::{ImageKernel, PoolKernel},
    schedule::{PointwiseDAG, Pw, ScalarStorage},
};
use std::collections::{BTreeSet, HashMap, HashSet};

struct Uses {
    producer: HashMap<BufferRef, usize>,
    readers: HashMap<BufferRef, Vec<usize>>,
}

impl Uses {
    fn new(plan: &ExecutionPlan) -> Self {
        let mut result = Self {
            producer: HashMap::new(),
            readers: HashMap::new(),
        };
        for (i, d) in plan.dispatches.iter().enumerate() {
            for b in std::iter::once(d.output_buffer).chain(d.extra_outputs.iter().copied()) {
                result.producer.insert(b, i);
            }
            for &b in &d.input_buffers {
                result.readers.entry(b).or_default().push(i);
            }
        }
        result
    }

    fn readers(&self, b: BufferRef) -> &[usize] {
        self.readers.get(&b).map_or(&[], Vec::as_slice)
    }
    fn one(&self, b: BufferRef) -> Option<usize> {
        match self.readers(b) {
            &[i] => Some(i),
            _ => None,
        }
    }
}

fn simple(d: &Dispatch) -> bool {
    d.extra_outputs.is_empty()
        && d.horizontal_batch <= 1
        && d.scalar_storage == ScalarStorage::default()
        && !d.fusion_barrier
}

fn transpose(d: &Dispatch, rows: u32, cols: u32, batch: u32) -> bool {
    simple(d)
        && d.shader == ShaderEntry::Transpose
        && d.kernel == Kernel::Default
        && d.params == [rows, cols, 0, 0]
        && d.workgroups[2] == batch
        && d.input_buffers.len() == 1
}

fn bias_dag(channels: u32, spatial: u32) -> PointwiseDAG {
    PointwiseDAG {
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
    }
}

fn identity() -> PointwiseDAG {
    PointwiseDAG {
        n_inputs: 1,
        ops: vec![Pw::LoadInput(0)],
        output: 0,
    }
}

fn sum_tail(
    plan: &ExecutionPlan,
    uses: &Uses,
    mut b: BufferRef,
    cols: u32,
    remove: &mut BTreeSet<usize>,
) -> Option<BufferRef> {
    while plan.buffers[b.0 as usize] != cols as usize * 4 {
        let i = uses.one(b)?;
        let d = &plan.dispatches[i];
        if !simple(d)
            || d.shader != ShaderEntry::SumRows
            || d.kernel != Kernel::Default
            || d.input_buffers != [b]
            || d.params[1] != cols
            || d.params[0] as usize * cols as usize * 4 != plan.buffers[b.0 as usize]
        {
            return None;
        }
        remove.insert(i);
        b = d.output_buffer;
    }
    Some(b)
}

struct Region {
    forward_at: usize,
    backward_at: usize,
    remove: BTreeSet<usize>,
    source: BufferRef,
    storage: DType,
    bias: Option<BufferRef>,
    scale: BufferRef,
    dy: BufferRef,
    y: BufferRef,
    dx: BufferRef,
    dx_type: DType,
    ds: BufferRef,
    db: Option<BufferRef>,
    params: Vec<u32>,
    pool: bool,
}

fn recognize(plan: &ExecutionPlan, uses: &Uses, ni: usize) -> Option<Region> {
    let n = &plan.dispatches[ni];
    if n.params.len() != 4 || n.input_buffers.len() != 2 || !simple(n) {
        return None;
    }
    let (rows, channels, eps) = (n.params[0], n.params[1], n.params[2]);
    if !(1..=256).contains(&channels)
        || n.kernel != Kernel::Reduction(rmsnorm_kernel(channels, f32::from_bits(eps)))
    {
        return None;
    }
    let ti = *uses.producer.get(&n.input_buffers[0])?;
    let t = &plan.dispatches[ti];
    let spatial = *t.params.get(1)?;
    let batch = t.workgroups[2];
    if spatial == 0
        || batch == 0
        || rows != batch.checked_mul(spatial)?
        || !transpose(t, channels, spatial, batch)
    {
        return None;
    }
    let si = uses.readers(n.output_buffer).iter().copied().find(|&i| {
        let d = &plan.dispatches[i];
        simple(d)
            && d.input_buffers == [n.output_buffer]
            && d.pointwise()
                == Some(&PointwiseDAG {
                    n_inputs: 1,
                    ops: vec![Pw::LoadInput(0), Pw::Silu(0)],
                    output: 1,
                })
    })?;
    let sg = uses.readers(n.output_buffer).iter().copied().find(|&i| {
        let d = &plan.dispatches[i];
        simple(d)
            && d.shader == ShaderEntry::SiluGrad
            && d.kernel == Kernel::Default
            && d.input_buffers.len() == 2
            && d.input_buffers[1] == n.output_buffer
            && d.params[0] == rows * channels
    })?;
    let s = &plan.dispatches[si];
    let fy = uses.one(s.output_buffer)?;
    if !transpose(&plan.dispatches[fy], spatial, channels, batch) {
        return None;
    }
    let grad = &plan.dispatches[sg];
    let gt = *uses.producer.get(&grad.input_buffers[0])?;
    if !transpose(&plan.dispatches[gt], channels, spatial, batch) {
        return None;
    }
    let find_grad = |shader| {
        uses.readers(grad.output_buffer).iter().copied().find(|&i| {
            let d = &plan.dispatches[i];
            simple(d)
                && d.shader == shader
                && d.kernel == Kernel::Default
                && d.params[..3] == n.params[..3]
                && d.input_buffers == [grad.output_buffer, n.input_buffers[0], n.input_buffers[1]]
        })
    };
    let gx = find_grad(ShaderEntry::RmsNormGradX)?;
    let gw = find_grad(ShaderEntry::RmsNormGradWRowPar)?;
    let xt = uses.one(plan.dispatches[gx].output_buffer)?;
    if !transpose(&plan.dispatches[xt], spatial, channels, batch) {
        return None;
    }
    let mut remove = BTreeSet::from([ti, ni, si, fy, sg, gt, gx, gw, xt]);
    let ds = sum_tail(
        plan,
        uses,
        plan.dispatches[gw].output_buffer,
        channels,
        &mut remove,
    )?;
    let mut source = t.input_buffers[0];
    let mut bias = None;
    let mut storage = DType::F32;
    let mut dx = plan.dispatches[xt].output_buffer;
    let mut dx_type = DType::F32;
    let mut db = None;
    let mut params = vec![batch, channels, 1, spatial, 1, spatial, eps, 0];
    let mut pooled = false;
    if let Some(&pi) = uses.producer.get(&source) {
        let p = &plan.dispatches[pi];
        if let Kernel::Pool(PoolKernel::Forward {
            ref prologue,
            ref inputs,
        }) = p.kernel
        {
            let q = &p.params;
            if q.len() != 12
                || q[0] != batch
                || q[1] != channels
                || q[4..8] != [2, 2, 2, 0]
                || q[8] * q[9] != spatial
                || q[2] != 2 * q[8]
                || q[3] != 2 * q[9]
                || p.extra_outputs.len() != 1
            {
                return None;
            }
            if *prologue == identity() && inputs.len() == 1 {
                source = p.input_buffers[0];
            } else if *prologue == bias_dag(channels, q[2] * q[3])
                && inputs.len() == 2
                && inputs[1] == DType::F32
            {
                source = p.input_buffers[0];
                bias = Some(p.input_buffers[1]);
            } else {
                return None;
            }
            storage = inputs[0];
            let bi = uses.one(dx)?;
            let b = &plan.dispatches[bi];
            let Kernel::Pool(PoolKernel::Backward { output, sum }) = b.kernel else {
                return None;
            };
            if b.input_buffers != [dx, p.extra_outputs[0]]
                || b.params != p.params
                || b.extra_outputs.len() != usize::from(sum)
            {
                return None;
            }
            dx = b.output_buffer;
            dx_type = output;
            if sum {
                db = Some(sum_tail(
                    plan,
                    uses,
                    b.extra_outputs[0],
                    channels,
                    &mut remove,
                )?);
            }
            params[2] = q[2];
            params[3] = q[3];
            params[4] = q[8];
            params[5] = q[9];
            pooled = true;
            remove.extend([pi, bi]);
        } else if p.pointwise() == Some(&bias_dag(channels, spatial))
            && p.input_buffers.len() == 2
            && p.extra_outputs.is_empty()
            && p.horizontal_batch <= 1
            && !p.scalar_storage.bf16_output
            && p.scalar_storage.bf16_inputs <= 1
            && (!p.fusion_barrier || p.scalar_storage.bf16_inputs == 1)
        {
            source = p.input_buffers[0];
            bias = Some(p.input_buffers[1]);
            if p.scalar_storage.bf16_inputs == 1 {
                storage = DType::BF16;
            }
            remove.insert(pi);
        }
    }
    if !pooled {
        if let Some(sum) = uses
            .readers(dx)
            .iter()
            .copied()
            .find(|&i| pool::spatial_sum(&plan.dispatches[i], batch * channels, spatial))
        {
            db = Some(sum_tail(
                plan,
                uses,
                plan.dispatches[sum].output_buffer,
                channels,
                &mut remove,
            )?);
            remove.insert(sum);
        }
        if let Some(cast) = uses.readers(dx).iter().copied().find(|&i| {
            let d = &plan.dispatches[i];
            simple(d) && d.shader == ShaderEntry::ToBF16 && d.params[0] == rows * channels
        }) {
            if !plan.graph_visible_buffers().contains(&dx)
                && uses
                    .readers(dx)
                    .iter()
                    .all(|&i| i == cast || remove.contains(&i))
            {
                dx = plan.dispatches[cast].output_buffer;
                dx_type = DType::BF16;
                remove.insert(cast);
            }
        }
    }
    if bias.is_some() && db.is_none() {
        return None;
    }
    let y = plan.dispatches[fy].output_buffer;
    let boundaries: HashSet<_> = [Some(y), Some(dx), Some(ds), db]
        .into_iter()
        .flatten()
        .collect();
    let visible = plan.graph_visible_buffers();
    for &i in &remove {
        let d = &plan.dispatches[i];
        for b in std::iter::once(d.output_buffer).chain(d.extra_outputs.iter().copied()) {
            if !boundaries.contains(&b)
                && (visible.contains(&b) || uses.readers(b).iter().any(|j| !remove.contains(j)))
            {
                return None;
            }
        }
    }
    if ni >= sg
        || uses
            .producer
            .get(&source)
            .is_some_and(|i| remove.contains(i))
    {
        return None;
    }
    Some(Region {
        forward_at: ni,
        backward_at: sg,
        remove,
        source,
        storage,
        bias,
        scale: n.input_buffers[1],
        dy: plan.dispatches[gt].input_buffers[0],
        y,
        dx,
        dx_type,
        ds,
        db,
        params,
        pool: pooled,
    })
}

fn allocate(plan: &mut ExecutionPlan, elements: u32) -> BufferRef {
    let b = BufferRef(plan.buffers.len() as u32);
    plan.buffers.push(elements as usize * 4);
    b
}

fn sum(
    plan: &mut ExecutionPlan,
    dispatches: &mut Vec<Dispatch>,
    rows: u32,
    cols: u32,
    src: BufferRef,
    dst: BufferRef,
) {
    let splits = Compiler::row_reduction_splits(rows, cols);
    let partial = if splits == 1 {
        dst
    } else {
        allocate(plan, splits * cols)
    };
    dispatches.push(Dispatch {
        shader: ShaderEntry::SumRows,
        input_buffers: vec![src],
        output_buffer: partial,
        params: vec![rows, cols, 0, if splits == 1 { 0 } else { splits }],
        workgroups: [cols.div_ceil(32), splits, 1],
        requires_full_precision: true,
        ..Default::default()
    });
    if splits > 1 {
        dispatches.push(Dispatch {
            shader: ShaderEntry::SumRows,
            input_buffers: vec![partial],
            output_buffer: dst,
            params: vec![splits, cols, 0, 0],
            workgroups: [cols.div_ceil(32), 1, 1],
            requires_full_precision: true,
            ..Default::default()
        });
    }
}

pub(super) fn fuse(plan: &mut ExecutionPlan) {
    loop {
        let uses = Uses::new(plan);
        let Some(r) = (0..plan.dispatches.len()).find_map(|i| recognize(plan, &uses, i)) else {
            break;
        };
        let channels = r.params[1];
        let rows = r.params[0] * r.params[4] * r.params[5];
        let forward_kernel = ImageKernel {
            view: None,
            channels,
            storage: r.storage,
            pool: r.pool,
            bias: r.bias.is_some(),
            backward: false,
        };
        let groups = rows.div_ceil(forward_kernel.rows_per_workgroup());
        let pooled = allocate(plan, rows * channels);
        let inverse = allocate(plan, rows);
        let winner = r.pool.then(|| allocate(plan, rows * channels));
        let scale_partial = allocate(plan, groups * channels);
        let bias_partial = r.db.map(|_| allocate(plan, groups * channels));
        let mut forward_inputs = vec![r.source, r.scale];
        forward_inputs.extend(r.bias);
        let mut extra = vec![pooled, inverse];
        extra.extend(winner);
        let forward = Dispatch {
            shader: ShaderEntry::Generated,
            kernel: Kernel::Image(forward_kernel),
            input_buffers: forward_inputs,
            output_buffer: r.y,
            extra_outputs: extra,
            params: r.params.clone(),
            workgroups: [groups, 1, 1],
            fusion_barrier: true,
            label: "image_region".into(),
            origin: plan.dispatches[r.forward_at].origin.clone(),
            ..Default::default()
        };
        let mut backward_inputs = vec![r.dy, pooled, inverse, r.scale];
        backward_inputs.extend(winner);
        let mut backward_extra = vec![scale_partial];
        backward_extra.extend(bias_partial);
        let backward = Dispatch {
            shader: ShaderEntry::Generated,
            kernel: Kernel::Image(ImageKernel {
                view: None,
                channels,
                storage: r.dx_type,
                pool: r.pool,
                bias: r.db.is_some(),
                backward: true,
            }),
            input_buffers: backward_inputs,
            output_buffer: r.dx,
            extra_outputs: backward_extra,
            params: r.params,
            workgroups: [groups, 1, 1],
            fusion_barrier: true,
            requires_full_precision: true,
            label: "image_region_grad".into(),
            origin: plan.dispatches[r.backward_at].origin.clone(),
            ..Default::default()
        };
        let mut after = vec![backward];
        sum(plan, &mut after, groups, channels, scale_partial, r.ds);
        if let (Some(src), Some(dst)) = (bias_partial, r.db) {
            sum(plan, &mut after, groups, channels, src, dst);
        }
        let old = std::mem::take(&mut plan.dispatches);
        for (i, d) in old.into_iter().enumerate() {
            if i == r.forward_at {
                plan.dispatches.push(forward.clone());
            }
            if i == r.backward_at {
                plan.dispatches.append(&mut after);
            }
            if !r.remove.contains(&i) {
                plan.dispatches.push(d);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_region_full_plan_coverage() {
        let Ok(path) = std::env::var("IMAGE_REGION_REFERENCE_PLAN") else {
            return;
        };
        let mut plan: ExecutionPlan =
            serde_json::from_reader(std::fs::File::open(path).unwrap()).unwrap();
        let visible = plan.graph_visible_buffers();
        let before = plan.dispatches.len();
        fuse(&mut plan);
        let kernels: Vec<_> = plan
            .dispatches
            .iter()
            .filter_map(|d| {
                if let Kernel::Image(k) = &d.kernel {
                    Some(k)
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(
            kernels.len(),
            16,
            "encoder and decoder, including the spatial stem"
        );
        assert_eq!(kernels.iter().filter(|k| k.pool).count(), 8);
        assert_eq!(plan.graph_visible_buffers(), visible);
        assert!(plan.dispatches.len() < before);
        println!(
            "image_regions=8 dispatches={before}->{}",
            plan.dispatches.len()
        );
    }
}
