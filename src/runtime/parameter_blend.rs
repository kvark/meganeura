//! Same-queue parameter blending, including compiler-derived dense copies.
use super::*;
use crate::graph::{DType, ParamTransform};

/// Why [`Session::blend_parameters_from`] rejected a request. Every check
/// runs before any parameter changes.
#[derive(Debug, PartialEq, Eq)]
pub enum BlendParameterError {
    /// The sessions use different GPU contexts.
    DifferentContext,
    /// Either session has a prepared step awaiting completion.
    PreparedStep,
    /// The rate is not a finite value in `[0, 1]`.
    InvalidRate,
    /// The name appears more than once in the request.
    DuplicateParameter(String),
    /// Either session has no parameter of this name.
    UnknownParameter(String),
    /// The two sessions' parameters differ in shape or element type.
    IncompatibleParameter(String),
    /// The parameter, or a copy derived from it, is not dense F32 storage.
    UnsupportedStorage(String),
    /// The target writes storage the source reads, as after
    /// [`Session::share_parameter_from`].
    SharedParameter(String),
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

/// Element `i < len` of `src` maps to
/// `dst[offset + (i / columns) * stride + i % columns]`.
#[derive(Debug)]
struct Region {
    src: BufferRef,
    dst: BufferRef,
    len: u32,
    columns: u32,
    stride: u32,
    offset: u32,
}

impl Region {
    fn new(
        src: BufferRef,
        dst: BufferRef,
        len: usize,
        columns: usize,
        stride: usize,
        offset: usize,
    ) -> Self {
        let word = |v: usize| u32::try_from(v).expect("parameter blend region exceeds u32");
        debug_assert!(len > 0 && columns > 0 && stride >= columns && len.is_multiple_of(columns));
        Self {
            src,
            dst,
            len: word(len),
            columns: word(columns),
            stride: word(stride),
            offset: word(offset),
        }
    }
}

/// Element count of a dense F32 parameter buffer.
fn dense_len(plan: &ExecutionPlan, buffer: BufferRef) -> Option<usize> {
    let ty = &plan.param_types[&buffer];
    let quantized = plan
        .weight_buffers
        .get(&buffer)
        .is_some_and(|&(format, _, _)| format != crate::compile::WeightFormat::F32);
    (ty.dtype == DType::F32 && !quantized).then(|| ty.num_elements())
}

/// The named blends, then the derived-copy refreshes they require, each
/// tagged with the parameter name that caused it.
type Updates<'a> = (Vec<(&'a str, Region)>, Vec<(&'a str, Region)>);

fn plan_updates<'a>(
    target: &ExecutionPlan,
    source: &ExecutionPlan,
    names: &[&'a str],
) -> Result<Updates<'a>, BlendParameterError> {
    let mut updates = Vec::new();
    let mut refreshes = Vec::new();
    for (index, &name) in names.iter().enumerate() {
        if names[..index].contains(&name) {
            return Err(BlendParameterError::DuplicateParameter(name.into()));
        }
        let lookup = |plan: &ExecutionPlan| {
            plan.param_buffers
                .iter()
                .find(|entry| entry.0 == name)
                .map(|entry| entry.1)
                .ok_or_else(|| BlendParameterError::UnknownParameter(name.into()))
        };
        let (src, dst) = (lookup(source)?, lookup(target)?);
        if source.param_types[&src] != target.param_types[&dst] {
            return Err(BlendParameterError::IncompatibleParameter(name.into()));
        }
        let unsupported = || BlendParameterError::UnsupportedStorage(name.into());
        let len = dense_len(source, src)
            .zip(dense_len(target, dst))
            .ok_or_else(unsupported)?
            .0;
        updates.push((name, Region::new(src, dst, len, len, len, 0)));
        // Refresh the target's compiler-derived copies of this parameter:
        // `sources` lists each part's extent along the concatenated axis.
        for &(derived, ref sources, ref transform) in &target.derived_params {
            if !sources.iter().any(|part| part.0 == name) {
                continue;
            }
            let capacity = dense_len(target, derived).ok_or_else(unsupported)?;
            let shape = &target.param_types[&derived].shape;
            debug_assert_eq!(shape.len(), 2);
            let mut offset = 0;
            for &(ref part, extent) in sources {
                if part == name {
                    // (columns, first element, rows) of this part.
                    let (columns, start, rows) = match *transform {
                        ParamTransform::VerticalConcat => (shape[1], offset * shape[1], extent),
                        ParamTransform::HorizontalConcat => (extent, offset, shape[0]),
                    };
                    debug_assert_eq!(rows * columns, len);
                    debug_assert!(start + (rows - 1) * shape[1] + columns <= capacity);
                    refreshes.push((
                        name,
                        Region::new(dst, derived, len, columns, shape[1], start),
                    ));
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
    /// Blend same-shaped dense F32 parameters from `source` on the shared
    /// GPU queue: `target = rate * source + (1 - rate) * target`, as for a
    /// slow-moving target network. Dense copies the compiler derived from
    /// a blended parameter (fused concatenations) are refreshed too.
    ///
    /// Both sessions track the submission, so later host access, steps and
    /// drops of either session wait for it. Every check runs before any
    /// parameter changes. Ordinary shader F32 rounding applies.
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
        // No destination may be storage that a blend reads from the source.
        for &(name, ref write) in updates.iter().chain(&refreshes) {
            let dst = write.dst.0 as usize;
            let shared = updates.iter().any(|read| {
                let src = read.1.src.0 as usize;
                overlaps(
                    self.buffers[dst],
                    self.plan.buffers[dst],
                    source.buffers[src],
                    source.plan.buffers[src],
                )
            });
            if shared {
                return Err(BlendParameterError::SharedParameter(name.into()));
            }
        }
        // Derived copies are allocations of their own, so a refresh never
        // overwrites a blended parameter it may still read.
        debug_assert!(refreshes.iter().all(|r| {
            updates.iter().all(|u| {
                let (a, b) = (r.1.dst.0 as usize, u.1.dst.0 as usize);
                !overlaps(
                    self.buffers[a],
                    self.plan.buffers[a],
                    self.buffers[b],
                    self.plan.buffers[b],
                )
            })
        }));
        if updates.is_empty() {
            return Ok(());
        }
        if updates
            .iter()
            .chain(&refreshes)
            .any(|entry| self.is_invariant_input(entry.1.dst))
        {
            // No host write announces these values, so the next step must
            // derive its kept transforms of them again.
            self.invariants_valid.set(false);
        }
        self.wait();
        if self.blend_pipeline.is_none() {
            use blade_graphics::ShaderData as _;
            let shader = self.gpu.create_shader(blade_graphics::ShaderDesc {
                source: include_str!("../shaders/parameter_blend.wgsl"),
                naga_module: None,
            });
            self.blend_pipeline = Some(self.gpu.create_compute_pipeline(
                blade_graphics::ComputePipelineDesc {
                    name: "parameter_blend",
                    data_layouts: &[&BlendData::layout()],
                    compute: shader.at("main"),
                    full_subgroup_size: None,
                },
            ));
        }
        self.encoder.start();
        self.profiled_pass_map.clear();
        // Blends complete before the refreshes read the blended values.
        for (regions, copy) in [(&updates, false), (&refreshes, true)] {
            if regions.is_empty() {
                continue;
            }
            let mut pass = self.encoder.compute(if copy {
                "parameter_refresh"
            } else {
                "parameter_blend"
            });
            let mut pipeline = pass.with(self.blend_pipeline.as_ref().unwrap());
            for r in regions.iter().map(|entry| &entry.1) {
                let blocks = r.len.div_ceil(256);
                let x = blocks.min(65535);
                let src = if copy {
                    self.buffers[r.src.0 as usize]
                } else {
                    source.buffers[r.src.0 as usize]
                };
                pipeline.bind(
                    0,
                    &BlendData {
                        src,
                        dst: self.buffers[r.dst.0 as usize],
                        params: [
                            r.len,
                            r.columns,
                            r.stride,
                            r.offset,
                            rate.to_bits(),
                            (1.0 - rate).to_bits(),
                            u32::from(copy),
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
mod tests {
    use super::*;
    use crate::graph::{DerivedParam, Graph};
    use BlendParameterError as E;

    /// Parameters `a`, `b` and `c`, with `[a | b]` and `[a; c]` derived from
    /// them by the horizontal and vertical concatenations, all as outputs.
    fn graph(rows: usize) -> Graph {
        let mut g = Graph::new();
        let a = g.parameter("a", &[rows, 3]);
        let b = g.parameter("b", &[rows, 2]);
        let c = g.parameter("c", &[2, 3]);
        let horizontal = g.parameter("horizontal", &[rows, 5]);
        let vertical = g.parameter("vertical", &[rows + 2, 3]);
        let derived = |name: &str, sources: [(&str, usize); 2], rows, transform| DerivedParam {
            name: name.into(),
            sources: sources.map(|(part, extent)| (part.into(), extent)).to_vec(),
            rows,
            transform,
        };
        g.derived_params.extend([
            derived(
                "horizontal",
                [("a", 3), ("b", 2)],
                rows,
                ParamTransform::HorizontalConcat,
            ),
            derived(
                "vertical",
                [("a", rows), ("c", 2)],
                rows + 2,
                ParamTransform::VerticalConcat,
            ),
        ]);
        let outputs = [a, b, c, horizontal, vertical].map(|node| g.materialize(node));
        g.set_outputs(outputs.to_vec());
        g
    }

    fn session(graph: &Graph, opts: SessionOptions) -> Session {
        let gpu = crate::reference::gpu::shared_context();
        Session::with_context_opts(crate::compile::compile(graph), gpu, opts)
    }

    fn pair() -> (Session, Session) {
        let default = SessionOptions::default;
        (session(&graph(5), default()), session(&graph(5), default()))
    }

    fn values(len: usize, scale: f32) -> Vec<f32> {
        (0..len).map(|i| (i as f32 - 7.0) * scale).collect()
    }

    /// Small multiples of these scales blend exactly, so the GPU result
    /// must equal the host computation bit for bit.
    #[test]
    fn blend_matches_host_and_refreshes_derived_copies() {
        let (mut target, mut source) = pair();
        for (name, len) in [("a", 15), ("b", 10), ("c", 6)] {
            source.set_parameter(name, &values(len, 0.375));
            target.set_parameter(name, &values(len, -1.25));
        }
        let rate = 0.25;
        target
            .blend_parameters_from(&mut source, &["a", "b"], rate)
            .unwrap();
        // No explicit wait: the step is ordered after the blend.
        target.step();
        let blend = |len| -> Vec<f32> {
            let old = values(len, -1.25);
            let new = values(len, 0.375);
            new.iter()
                .zip(old)
                .map(|(s, t)| rate * s + (1.0 - rate) * t)
                .collect()
        };
        let (a, b, c) = (blend(15), blend(10), values(6, -1.25));
        let horizontal = (0..5)
            .flat_map(|r| a[r * 3..r * 3 + 3].iter().chain(&b[r * 2..r * 2 + 2]))
            .copied()
            .collect();
        let vertical = a.iter().chain(&c).copied().collect();
        for (index, expected) in [a, b, c, horizontal, vertical].iter().enumerate() {
            let mut actual = vec![0.0; expected.len()];
            target.wait_read_output(index, &mut actual);
            assert_eq!(&actual, expected, "output {index}");
        }
        assert_eq!(source.read_params(&["a"])[0], values(15, 0.375));
    }

    fn reject(target: &mut Session, source: &mut Session, names: &[&str], rate: f32) -> E {
        target
            .blend_parameters_from(source, names, rate)
            .unwrap_err()
    }

    #[test]
    fn rejects_different_context() {
        let other = crate::runtime::init_gpu_context_with(crate::GpuOptions::from_env()).unwrap();
        let plan = crate::compile::compile(&graph(5));
        let mut target = Session::with_context(plan, Arc::new(other));
        let mut source = pair().1;
        let error = reject(&mut target, &mut source, &["a"], 0.5);
        assert_eq!(error, E::DifferentContext);
    }

    #[test]
    fn rejects_prepared_step() {
        let opts = SessionOptions {
            prepare_outputs: vec![0],
            ..SessionOptions::default()
        };
        let mut target = session(&graph(5), opts);
        let mut source = pair().1;
        target.prepare_step();
        let error = reject(&mut target, &mut source, &["a"], 0.5);
        assert_eq!(error, E::PreparedStep);
        target.finish_prepared_step();
    }

    #[test]
    fn rejects_invalid_rate() {
        let (mut target, mut source) = pair();
        for rate in [-0.25, 1.5, f32::NAN] {
            assert_eq!(
                reject(&mut target, &mut source, &["a"], rate),
                E::InvalidRate
            );
        }
    }

    #[test]
    fn rejects_duplicate_parameter_before_any_change() {
        let (mut target, mut source) = pair();
        source.set_parameter("a", &values(15, 1.0));
        target.set_parameter("a", &values(15, -1.0));
        let error = reject(&mut target, &mut source, &["a", "b", "a"], 0.5);
        assert_eq!(error, E::DuplicateParameter("a".into()));
        assert_eq!(target.read_params(&["a"])[0], values(15, -1.0));
    }

    #[test]
    fn rejects_unknown_parameter() {
        let (mut target, mut source) = pair();
        let error = reject(&mut target, &mut source, &["a", "missing"], 0.5);
        assert_eq!(error, E::UnknownParameter("missing".into()));
    }

    #[test]
    fn rejects_incompatible_parameter() {
        let mut target = session(&graph(4), SessionOptions::default());
        let mut source = pair().1;
        let error = reject(&mut target, &mut source, &["c", "a"], 0.5);
        assert_eq!(error, E::IncompatibleParameter("a".into()));
    }

    #[test]
    fn rejects_unsupported_storage() {
        let mut g = Graph::new();
        let x = g.input("x", &[2, 4]);
        let w = g.parameter_f16("w", &[4, 4]);
        let y = g.matmul(x, w);
        g.set_outputs(vec![y]);
        let plan = crate::compile::compile(&g);
        let error = plan_updates(&plan, &plan, &["w"]).unwrap_err();
        assert_eq!(error, E::UnsupportedStorage("w".into()));
    }

    #[test]
    fn rejects_shared_parameter() {
        let (mut target, mut source) = pair();
        target.share_parameter_from(&mut source, "b").unwrap();
        let error = reject(&mut target, &mut source, &["a", "b"], 0.5);
        assert_eq!(error, E::SharedParameter("b".into()));
    }
}
