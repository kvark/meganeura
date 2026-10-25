//! Same-queue parameter averaging, including compiler-derived dense copies.
use super::*;
use crate::graph::{DType, ParamTransform};

#[derive(Debug, PartialEq, Eq)]
pub enum BlendParameterError {
    DifferentContext,
    PreparedStep,
    InvalidRate,
    UnknownParameter(String),
    IncompatibleParameter(String),
    UnsupportedStorage,
    InvalidGeometry,
    DuplicateParameter,
    AliasedParameters,
}

impl std::fmt::Display for BlendParameterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "parameter blend: {self:?}")
    }
}
impl std::error::Error for BlendParameterError {}

#[derive(blade_macros::ShaderData)]
struct BlendData {
    src: blade_graphics::BufferPiece,
    dst: blade_graphics::BufferPiece,
    params: [u32; 8],
}

#[derive(Clone, Debug)]
struct Region {
    src: BufferRef,
    dst: BufferRef,
    len: u32,
    columns: u32,
    stride: u32,
    offset: u32,
}

fn dense_len(plan: &ExecutionPlan, buffer: BufferRef) -> Result<usize, BlendParameterError> {
    let ty = plan
        .param_types
        .get(&buffer)
        .ok_or(BlendParameterError::UnsupportedStorage)?;
    if ty.dtype != DType::F32
        || plan
            .weight_buffers
            .get(&buffer)
            .is_some_and(|&(format, _, _)| format != crate::compile::WeightFormat::F32)
    {
        return Err(BlendParameterError::UnsupportedStorage);
    }
    let len = ty
        .shape
        .iter()
        .try_fold(1usize, |size, &dim| size.checked_mul(dim))
        .ok_or(BlendParameterError::InvalidGeometry)?;
    if len == 0
        || len > u32::MAX as usize
        || len
            .checked_mul(4)
            .is_none_or(|bytes| bytes > plan.buffers[buffer.0 as usize])
    {
        return Err(BlendParameterError::InvalidGeometry);
    }
    Ok(len)
}

fn region(
    src: BufferRef,
    dst: BufferRef,
    len: usize,
    columns: usize,
    stride: usize,
    offset: usize,
    capacity: usize,
) -> Result<Region, BlendParameterError> {
    if columns == 0
        || stride < columns
        || !len.is_multiple_of(columns)
        || [len, columns, stride, offset, capacity]
            .iter()
            .any(|&v| v > u32::MAX as usize)
        || len == 0
    {
        return Err(BlendParameterError::InvalidGeometry);
    }
    let end = (len / columns - 1)
        .checked_mul(stride)
        .and_then(|row| row.checked_add(columns))
        .and_then(|row| row.checked_add(offset));
    if end.is_none_or(|end| end > capacity) {
        return Err(BlendParameterError::InvalidGeometry);
    }
    Ok(Region {
        src,
        dst,
        len: len as u32,
        columns: columns as u32,
        stride: stride as u32,
        offset: offset as u32,
    })
}

fn plan_updates(
    target: &ExecutionPlan,
    source: &ExecutionPlan,
    names: &[&str],
) -> Result<(Vec<Region>, Vec<Region>), BlendParameterError> {
    let mut seen = std::collections::HashSet::new();
    let mut updates = Vec::new();
    let mut refreshes = Vec::new();
    for &name in names {
        if !seen.insert(name) {
            return Err(BlendParameterError::DuplicateParameter);
        }
        let lookup = |plan: &ExecutionPlan| {
            plan.param_buffers
                .iter()
                .find(|entry| entry.0 == name)
                .map(|entry| entry.1)
                .ok_or_else(|| BlendParameterError::UnknownParameter(name.into()))
        };
        let src = lookup(source)?;
        let dst = lookup(target)?;
        let len = dense_len(source, src)?;
        if len != dense_len(target, dst)? || source.param_types[&src] != target.param_types[&dst] {
            return Err(BlendParameterError::IncompatibleParameter(name.into()));
        }
        updates.push(region(src, dst, len, len, len, 0, len)?);
        for &(derived, ref sources, ref transform) in &target.derived_params {
            if !sources.iter().any(|entry| entry.0 == name) {
                continue;
            }
            let capacity = dense_len(target, derived)?;
            let shape = &target.param_types[&derived].shape;
            if shape.len() != 2 {
                return Err(BlendParameterError::InvalidGeometry);
            }
            let total = sources
                .iter()
                .try_fold(0usize, |acc, item| acc.checked_add(item.1))
                .ok_or(BlendParameterError::InvalidGeometry)?;
            let expected = match *transform {
                ParamTransform::VerticalConcat => shape[0],
                ParamTransform::HorizontalConcat => shape[1],
            };
            if total != expected {
                return Err(BlendParameterError::InvalidGeometry);
            }
            let mut offset = 0usize;
            for &(ref source_name, extent) in sources {
                if source_name == name {
                    let (columns, stride, start, elements) = match *transform {
                        ParamTransform::VerticalConcat => {
                            (shape[1], shape[1], offset * shape[1], extent * shape[1])
                        }
                        ParamTransform::HorizontalConcat => {
                            (extent, shape[1], offset, extent * shape[0])
                        }
                    };
                    if elements != len {
                        return Err(BlendParameterError::InvalidGeometry);
                    }
                    refreshes.push(region(dst, derived, len, columns, stride, start, capacity)?);
                }
                offset += extent;
            }
        }
    }
    Ok((updates, refreshes))
}

fn overlaps(
    a: blade_graphics::BufferPiece,
    a_len: usize,
    b: blade_graphics::BufferPiece,
    b_len: usize,
) -> bool {
    a.buffer == b.buffer && a.offset < b.offset + b_len as u64 && b.offset < a.offset + a_len as u64
}

impl Session {
    /// Blend selected same-shaped dense F32 parameters on the shared GPU queue.
    ///
    /// Computes `rate * source + (1 - rate) * target`, then refreshes affected
    /// dense vertical/horizontal concatenations. Reduced/packed master storage
    /// is rejected. Ordinary shader F32 contraction/underflow rules apply.
    ///
    /// Both sessions track completion: subsequent CPU access, reuse or Drop
    /// cannot race the submitted reads/writes. Neither may have a prepared
    /// forward step. Validation completes before any parameter is changed.
    pub fn blend_parameters_from(
        &mut self,
        source: &mut Session,
        names: &[&str],
        rate: f32,
    ) -> Result<(), BlendParameterError> {
        if !Arc::ptr_eq(&self.gpu, &source.gpu) {
            return Err(BlendParameterError::DifferentContext);
        }
        if self.step_prepared || source.step_prepared {
            return Err(BlendParameterError::PreparedStep);
        }
        if !rate.is_finite() || !(0.0..=1.0).contains(&rate) {
            return Err(BlendParameterError::InvalidRate);
        }
        let (updates, refreshes) = plan_updates(&self.plan, &source.plan, names)?;
        // No update destination may alias a source, another destination or a
        // derived allocation. In particular, a derived copy is not an in-place
        // permutation. Compiler-owned dense copies have distinct allocations.
        let originals = updates.iter().map(|r| r.dst).collect::<Vec<_>>();
        let mut destinations = originals.clone();
        for r in &refreshes {
            if originals.contains(&r.dst) {
                return Err(BlendParameterError::AliasedParameters);
            }
            if !destinations.contains(&r.dst) {
                destinations.push(r.dst);
            }
        }
        for (index, &dst) in destinations.iter().enumerate() {
            let target = self.buffers[dst.0 as usize];
            let target_len = self.plan.buffers[dst.0 as usize];
            for r in &updates {
                if overlaps(
                    target,
                    target_len,
                    source.buffers[r.src.0 as usize],
                    source.plan.buffers[r.src.0 as usize],
                ) {
                    return Err(BlendParameterError::AliasedParameters);
                }
            }
            for &other in &destinations[..index] {
                if overlaps(
                    target,
                    target_len,
                    self.buffers[other.0 as usize],
                    self.plan.buffers[other.0 as usize],
                ) {
                    return Err(BlendParameterError::AliasedParameters);
                }
            }
        }
        if updates.is_empty() {
            return Ok(());
        }
        self.wait();
        if self.blend_pipeline.is_none() {
            use blade_graphics::ShaderData as _;
            let shader = self.gpu.create_shader(blade_graphics::ShaderDesc {
                source: include_str!("../shaders/parameter_blend.wgsl"),
                naga_module: None,
            });
            self.blend_pipeline = Some(self.gpu.create_compute_pipeline(
                blade_graphics::ComputePipelineDesc { full_subgroup_size: None,
                    name: "meganeura_parameter_blend",
                    data_layouts: &[&BlendData::layout()],
                    compute: shader.at("main"),
                },
            ));
        }
        self.encoder.start();
        self.profiled_pass_map.clear();
        for (regions, copies) in [(&updates, false), (&refreshes, true)] {
            if regions.is_empty() {
                continue;
            }
            let mut pass = self.encoder.compute(if copies {
                "parameter_refresh"
            } else {
                "parameter_blend"
            });
            let mut pipeline = pass.with(self.blend_pipeline.as_ref().unwrap());
            for r in regions {
                let blocks = r.len.div_ceil(256);
                let x = blocks.min(65535);
                pipeline.bind(
                    0,
                    &BlendData {
                        src: if copies {
                            self.buffers[r.src.0 as usize]
                        } else {
                            source.buffers[r.src.0 as usize]
                        },
                        dst: self.buffers[r.dst.0 as usize],
                        params: [
                            r.len,
                            r.columns,
                            r.stride,
                            r.offset,
                            rate.to_bits(),
                            (1.0 - rate).to_bits(),
                            u32::from(copies),
                            0,
                        ],
                    },
                );
                pipeline.dispatch([x, blocks.div_ceil(x), 1]);
            }
        }
        let done = self.gpu.submit(&mut self.encoder);
        source.track_submission(done.clone());
        self.sync_point = Some(done);
        Ok(())
    }
}

#[cfg(test)]
mod tests;
