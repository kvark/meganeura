//! Target-aware plan lowering: kernel-family path selection, cooperative tile
//! padding, split-K, late RmsNorm prologue fusion, workgroup cooperative
//! selection, attention precision/shared-memory fitting, and dispatch
//! scheduling.
//!
//! Moving these passes into a single explicit lowering stage keeps compilation
//! target-aware and gives `Session` one finalized [`ExecutionPlan`] rather than
//! re-compiling inside runtime initialization.

use super::{
    AttentionParams, BufferRef, Dispatch, DispatchParams, ExecutionPlan, Kernel, ShaderEntry,
    WeightFormat, fuse_rmsnorm_into_add, fuse_rmsnorm_into_gemv, fuse_rmsnorm_prologues,
    schedule_dispatches, stages,
};
use crate::codegen::{
    CoopCaps, CoopConfig, CooperativeMatmulShape, MatrixTypes, WorkgroupMatmulShape,
};

/// Cooperative-matrix precision policy for session construction and target
/// lowering.
///
/// Explicit typed F16/BF16 products require their native implementation and
/// are not widened or disabled by this policy. Native F32 accumulators do not
/// promise scalar IEEE FMA semantics; see [`crate::reference::native_matrix`].
/// To avoid native matrix arithmetic, use F32 operands with [`Self::Disabled`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CoopPolicy {
    /// Prefer f32 operands and accumulators. On f16-only devices, enable f16 tiles for
    /// work that permits reduced input precision; explicitly pinned and ordinary
    /// derivative dispatches retain scalar f32 operands. Measured search may select
    /// f16 staging for scalar flash attention: forward Q/K and backward Q/K/V, with
    /// forward V, backward dO and dot arithmetic kept f32. Search also considers
    /// device-qualified F16/BF16 forward products, mixed-precision cooperative
    /// attention derivatives, AT/BT derivative products and pointwise convolution
    /// input gradients, subject to the caller's whole-program accuracy and timing
    /// gates. Explicit precision pins are never bypassed.
    #[default]
    Auto,
    /// Allow only f32 cooperative operands and accumulators, without f16
    /// staging or compensation. This is a precision policy, not a vendor
    /// restriction. Devices with only reduced-input cooperative tiles retain
    /// scalar f32 implementations.
    NativeF32,
    /// Disable cooperative selection for ordinary products, using scalar paths.
    /// Explicit typed products still require native cooperative matrices.
    Disabled,
    /// Use f16 tiles without residual compensation, including for
    /// derivative work. Faster than [`Self::Auto`] on f16-only devices,
    /// and can overflow or lose gradient range. Measured search also considers
    /// device-qualified typed F16/BF16 candidates for eligible forward products,
    /// AT/BT derivatives and pointwise convolution input gradients,
    /// subject to the caller's whole-program accuracy and timing gates.
    AllowF16,
}

impl CoopPolicy {
    pub(crate) fn filter_caps(self, mut caps: CoopCaps) -> CoopCaps {
        match self {
            Self::Disabled => caps = CoopCaps::default(),
            Self::NativeF32 => {
                caps.f16_tile = 0;
                caps.f16_attention_rows = 0;
            }
            Self::AllowF16 | Self::Auto => {}
        }
        caps
    }
}

/// Target hardware and policy parameters for [`ExecutionPlan::finalize_for_target`].
pub(crate) struct LowerTarget<'a> {
    pub coop_config: Option<&'a CoopConfig>,
    pub capabilities: &'a blade_graphics::Capabilities,
    pub coop_policy: CoopPolicy,
    pub workgroup_matmul: Option<WorkgroupMatmulShape>,
    pub serial_dispatch: bool,
    pub debug: bool,
    pub prepare_outputs: &'a [usize],
}

impl ExecutionPlan {
    /// Lower a compiled plan into its finalized target-ready form in one pass:
    /// attention tile precision, cooperative/scalar path selection and padding,
    /// scalar split-K, late RmsNorm fusions, native-f32 tiling, workgroup
    /// cooperative selection, flash shared-memory fitting, and barrier-group
    /// scheduling (including horizontal matmul fusion).
    pub(crate) fn finalize_for_target(
        &mut self,
        target: &LowerTarget<'_>,
    ) -> Option<stages::Preparation> {
        apply_attention_precision(self, target.coop_policy);
        select_variants(
            self,
            target.coop_config,
            !target.debug,
            target.coop_policy == CoopPolicy::AllowF16,
        );
        select_workgroup_matmuls(
            self,
            target.capabilities,
            target.workgroup_matmul,
            target.coop_policy,
        );
        if let Some(head_dim) = self
            .dispatches
            .iter()
            .filter(|dispatch| dispatch.shader == ShaderEntry::FlashAttention)
            .map(|dispatch| {
                AttentionParams::from_words(&dispatch.params)
                    .expect("invalid attention parameters")
                    .head_dim
            })
            .max()
        {
            self.knobs
                .flash
                .fit_shared_memory(head_dim, target.capabilities.max_compute_shared_memory_size);
        }

        if target.serial_dispatch {
            log::info!("MEGANEURA_SERIAL_DISPATCH: forcing one dispatch per pass");
        }
        let fuse = !target.serial_dispatch && !target.debug;
        if target.prepare_outputs.is_empty() {
            schedule_dispatches(self, target.serial_dispatch, fuse);
            None
        } else {
            Some(stages::schedule(
                self,
                target.prepare_outputs,
                target.serial_dispatch,
                fuse,
            ))
        }
    }
}

pub(crate) fn tiled_coop_shared_bytes(dispatch: &Dispatch, shape: CooperativeMatmulShape) -> u64 {
    let row_factors = dispatch.matmul_prologue.as_ref().map_or(0, |p| {
        p.factors
            .iter()
            .filter(|f| f.1 == super::PrologueLoadKind::PerRow)
            .count()
    });
    u64::from(shape.shared_bytes()) + row_factors as u64 * 64 * 4
}

/// Honor [`CoopPolicy`] and explicit pins on scalar flash f16 tiles.
///
/// Compile requests f16 staging only through
/// [`crate::CompileOptions::flash_f16_tiles`]. `NativeF32` and `Disabled`
/// clear every request, `AllowF16` enables staging, and `Auto` preserves the
/// compile request. A full-precision pin always clears the bit.
pub(crate) fn apply_attention_precision(plan: &mut ExecutionPlan, coop: CoopPolicy) {
    for dispatch in &mut plan.dispatches {
        let flash = match dispatch.shader {
            ShaderEntry::FlashAttention => true,
            ShaderEntry::AttentionGrad(kernel) => {
                kernel.path == crate::kernels::attention_grad::Path::Flash
            }
            _ => false,
        };
        if !flash || dispatch.requires_full_precision || dispatch.kernel == Kernel::BlockAttention {
            dispatch.attention_f16_tiles = false;
            continue;
        }
        dispatch.attention_f16_tiles = match coop {
            CoopPolicy::NativeF32 | CoopPolicy::Disabled => false,
            CoopPolicy::AllowF16 => true,
            CoopPolicy::Auto => dispatch.attention_f16_tiles,
        };
    }
}

/// Reset any prior workgroup-cooperative kernel selection back to its
/// device-independent base so a reloaded plan can be re-selected for the active
/// adapter.
pub(crate) fn reset_workgroup_matmuls(plan: &mut ExecutionPlan) {
    for dispatch in &mut plan.dispatches {
        if let Kernel::CooperativeWorkgroup { types, splits, .. } = dispatch.kernel {
            let product = dispatch.product().expect("workgroup product dimensions");
            if let Some(types) = types {
                dispatch.kernel = Kernel::CooperativeTyped {
                    types,
                    splits,
                    specialize: false,
                };
                dispatch.workgroups = product
                    .launch(
                        crate::kernels::matmul::Variant::CooperativeTyped { specialize: false },
                        splits,
                    )
                    .workgroups;
            } else {
                dispatch.kernel = Kernel::Default;
                dispatch.workgroups = product
                    .launch(crate::kernels::matmul::Variant::DefaultScalar, 1)
                    .workgroups;
            }
        }
    }
}

/// Select an opt-in workgroup-cooperative matmul shape for eligible F16/BF16
/// products when `coop_policy == CoopPolicy::AllowF16`.
pub(crate) fn select_workgroup_matmuls(
    plan: &mut ExecutionPlan,
    caps: &blade_graphics::Capabilities,
    workgroup_matmul: Option<WorkgroupMatmulShape>,
    coop_policy: CoopPolicy,
) {
    let Some(shape) = workgroup_matmul.filter(|_| coop_policy == CoopPolicy::AllowF16) else {
        return;
    };
    for dispatch in &mut plan.dispatches {
        let (types, splits) = match dispatch.kernel {
            Kernel::CooperativeTyped { types, splits, .. } => (Some(types), splits),
            _ => (None, 1),
        };
        let f32 = matches!(
            dispatch.kernel,
            Kernel::Default | Kernel::SmallTile | Kernel::Cooperative
        );
        if !matches!(
            dispatch.shader,
            ShaderEntry::MatMul
                | ShaderEntry::MatMulAT
                | ShaderEntry::MatMulBT
                | ShaderEntry::FusedMatMulAdd
                | ShaderEntry::FusedMatMulATAdd
                | ShaderEntry::FusedMatMulBTAdd
        ) || dispatch.horizontal_batch > 1
            || dispatch.matmul_prologue.is_some()
            || dispatch
                .matmul_epilogue
                .as_ref()
                .is_some_and(|e| shape.subgroup_rows == 0 || !e.inputs.is_empty())
            || dispatch.full_precision()
            || dispatch.weight_format != WeightFormat::F32
            || (types.is_none() && dispatch.schedule_locked)
            || !(types.is_some() || f32)
        {
            continue;
        }
        if let Some(product) = dispatch.product() {
            if shape.supports_layout(
                caps,
                types.unwrap_or(MatrixTypes::F32_AS_F16),
                Some(product.layout.group()),
                product.m,
                product.n,
                product.k,
            ) && splits > 0
                && splits <= shape.k_blocks(product.k)
            {
                dispatch.kernel = Kernel::CooperativeWorkgroup {
                    shape,
                    types,
                    splits,
                };
                dispatch.workgroups = product
                    .launch(
                        crate::kernels::matmul::Variant::CooperativeWorkgroup(shape),
                        splits,
                    )
                    .workgroups;
            }
        }
    }
}

/// Per-dispatch kernel-variant selection: each convolution and dense product
/// takes the path its kernel family selects (cooperative tiles with their
/// padding, or 32-wide / wide scalar tiles), then scalar split-K and the
/// RmsNorm→matmul prologue fusions that depend on cooperative paths run.
/// Mutates the plan in place. Tuning may replace these initial choices by
/// measurement, asking the same families what is legal.
pub(crate) fn select_variants(
    plan: &mut ExecutionPlan,
    coop_config: Option<&CoopConfig>,
    fuse_prologues: bool,
    allow_raw_f16: bool,
) {
    reset_workgroup_matmuls(plan);
    select_conv_paths(plan, coop_config, allow_raw_f16);
    select_matmul_paths(plan, coop_config, allow_raw_f16);
    select_batch_register_matmuls(plan, coop_config);
    let wg = crate::schedule::GridShape::default().workgroup_size;
    for dispatch in &mut plan.dispatches {
        if let Some(dag) = dispatch.pointwise() {
            let len = dispatch.params.first().copied().unwrap_or(0);
            let work_items = if dag.can_vectorize(dispatch.pointwise_storage, len) {
                len / 4
            } else {
                len
            };
            dispatch.workgroups = [work_items.div_ceil(wg), 1, 1];
        }
    }

    plan.split_long_scalar_products();
    plan.split_low_occupancy_scalar_products();

    // Apply RmsNorm+MatMul prologue fusion only after per-dispatch coop
    // selection. A coop-capable device can still route small matmuls to a
    // scalar pipeline, which has no prologue implementation. The fusion
    // pass filters for kernels that support prologues and records factor
    // buffers as inputs for scheduling and memory planning. Debug sessions skip it —
    // dispatch-level fusion is numerics-neutral, and keeping the RmsNorm
    // output materialized is the point of debug mode.
    if fuse_prologues {
        fuse_rmsnorm_prologues(plan);
        fuse_rmsnorm_into_add(plan);
        fuse_rmsnorm_into_gemv(plan);
    }
    // Aligned native-f32 products share 64-row tiles between four subgroups.
    // Up to 16 K partitions of at least 128 each raise small outputs toward
    // 512 workgroups; 128 columns are used when partitions can still reach
    // that target, 64 otherwise. Ragged, pinned and fused-epilogue products
    // keep their kernel; tuning can challenge unsplit choices.
    if coop_config.is_some_and(|c| c.tile_size == 16 && !c.use_f16_input) {
        for index in (0..plan.dispatches.len()).rev() {
            let dispatch = &plan.dispatches[index];
            let Some((m, n, k)) = dispatch.mnk() else {
                continue;
            };
            let max_splits = crate::kernels::occupancy::NATIVE_F32_TILED_SPLITS.max_splits(k);
            let groups = |columns: u32| u64::from(m / 64) * u64::from(n / columns);
            let wide = n.is_multiple_of(128) && groups(128) * u64::from(max_splits) >= 512;
            let shape = CooperativeMatmulShape {
                columns: if wide { 128 } else { 64 },
                k_stage: 16,
                prefetch: false,
            };
            if !shape.fits_dimensions(m, n, k)
                || tiled_coop_shared_bytes(dispatch, shape) > 16 * 1024
            {
                continue;
            }
            let output_groups = u32::try_from(groups(shape.columns)).unwrap_or(u32::MAX);
            let splits =
                crate::kernels::occupancy::NATIVE_F32_TILED_SPLITS.splits(output_groups, k);
            // Fused, pinned and aliased forms keep their kernel.
            let _ = plan.tile_native_f32_matmul(index, shape, splits, 32 * 1024 * 1024);
        }
    }
}

/// Promote each 64-wide convolution its kernel family admits to a
/// generated cooperative kernel, and size its workgroups and output.
fn select_conv_paths(
    plan: &mut ExecutionPlan,
    coop_config: Option<&CoopConfig>,
    allow_raw_f16: bool,
) {
    use crate::kernels::conv;
    let target = conv::Target {
        cooperative: coop_config.copied(),
        allow_raw_f16,
    };
    for index in 0..plan.dispatches.len() {
        let Some(problem) = conv::Problem::of(&plan.dispatches[index]) else {
            continue;
        };
        let target = conv::Target {
            allow_raw_f16: target.allow_raw_f16 && !plan.dispatches[index].requires_full_precision,
            ..target
        };
        match conv::select(&problem, &target) {
            conv::Path::Compiled => continue,
            // The tiles stage an im2col slab through workgroup memory, which
            // costs more than it saves where no cooperative matrices consume
            // it: on Adreno 740 the register kernel ran D-FINE-N's
            // convolutions 3–22× faster, and about 2× on an RTX 3050 without
            // cooperative matrices.
            conv::Path::Register => {
                let dispatch = &mut plan.dispatches[index];
                dispatch.kernel = Kernel::RegisterConv(problem.register_block());
                dispatch.workgroups = problem.register_workgroups();
                continue;
            }
            conv::Path::Cooperative => {}
        }
        let config = coop_config.expect("cooperative paths need a configuration");
        let geometry = conv::cooperative_geometry(&problem, config)
            .expect("an admitted cooperative convolution is legal");
        let dispatch = &mut plan.dispatches[index];
        dispatch.kernel = Kernel::Cooperative;
        dispatch.shader = problem.cooperative_entry();
        dispatch.workgroups = geometry.grid;
        let output = dispatch.output_buffer;
        let extra = dispatch.input_buffers.get(2).copied();
        // Direct cooperative stores cover complete sub-tiles.
        pad_buffer(plan, output, geometry.output_bytes);
        if let Some(extra) = extra {
            pad_buffer(plan, extra, geometry.output_bytes);
        }
    }
}

/// Pick each dense product's path from its kernel family, and size its
/// workgroups and buffers for it.
fn select_matmul_paths(
    plan: &mut ExecutionPlan,
    coop_config: Option<&CoopConfig>,
    allow_raw_f16: bool,
) {
    use crate::kernels::matmul;
    let target = matmul::Target {
        cooperative: coop_config.copied(),
        allow_raw_f16,
        prefer: plan.knobs.prefer_matmul,
    };
    for index in 0..plan.dispatches.len() {
        let Some(problem) = matmul::Problem::of(&plan.dispatches[index]) else {
            continue;
        };
        let target = matmul::Target {
            allow_raw_f16: target.allow_raw_f16 && !plan.dispatches[index].requires_full_precision,
            ..target
        };
        match matmul::select(&problem, &target) {
            matmul::Path::Cooperative => {
                let config = coop_config.expect("cooperative paths need a configuration");
                let geometry = matmul::cooperative_geometry(&problem, config)
                    .expect("an admitted cooperative product is legal");
                let dispatch = &mut plan.dispatches[index];
                dispatch.kernel = Kernel::Cooperative;
                dispatch.workgroups = geometry.grid;
                let output = dispatch.output_buffer;
                let addend = geometry.pad_addend.then(|| dispatch.input_buffers[2]);
                pad_buffer(plan, output, geometry.output_bytes);
                if let Some(addend) = addend {
                    pad_buffer(plan, addend, geometry.output_bytes);
                }
            }
            // The tiles stage A and B through workgroup memory; without
            // cooperative matrices to consume it, registers are faster: on
            // Adreno 740 D-FINE-N's products ran about 4× faster.
            matmul::Path::Register => {
                let tile = matmul::register_tile(problem.m, problem.n, problem.k)
                    .expect("an admitted register product is aligned");
                let dispatch = &mut plan.dispatches[index];
                dispatch.kernel = Kernel::RegisterMatmul(tile);
                dispatch.workgroups = matmul::register_workgroups(tile, problem.m, problem.n, 1);
            }
            matmul::Path::SmallTile => {
                let dispatch = &mut plan.dispatches[index];
                let product = dispatch.product().expect("dense product");
                dispatch.kernel = Kernel::SmallTile;
                dispatch.workgroups = product.launch(matmul::Variant::SmallScalar, 1).workgroups;
            }
            matmul::Path::WideTile => {
                let dispatch = &mut plan.dispatches[index];
                let product = dispatch.product().expect("dense product");
                dispatch.kernel = Kernel::ScalarMatmul(matmul::WIDE_TILE);
                dispatch.workgroups = product
                    .launch(matmul::Variant::Scalar(matmul::WIDE_TILE), 1)
                    .workgroups;
            }
            matmul::Path::Compiled => {}
        }
    }
}

/// Batch-major products are not in the matmul family; give them the
/// register kernel by the same rules as [`matmul::Path::Register`]. Runs
/// before split-K and horizontal packing, which leave the register kernel
/// alone.
fn select_batch_register_matmuls(plan: &mut ExecutionPlan, coop_config: Option<&CoopConfig>) {
    use crate::kernels::matmul;
    let target = matmul::Target {
        cooperative: coop_config.copied(),
        allow_raw_f16: false,
        prefer: plan.knobs.prefer_matmul,
    };
    for dispatch in &mut plan.dispatches {
        if dispatch.shader != ShaderEntry::BatchMatMul
            || !matches!(dispatch.kernel, Kernel::Default | Kernel::SmallTile)
            || dispatch.schedule_locked
            || dispatch.matmul_epilogue.is_some()
            || dispatch.matmul_prologue.is_some()
            || dispatch.weight_format != WeightFormat::F32
        {
            continue;
        }
        let Some((m, n, k)) = dispatch.mnk() else {
            continue;
        };
        let batch = dispatch.params[3];
        if let Ok(tile) = matmul::register_admits_product([m, n, k], batch, &target) {
            dispatch.kernel = Kernel::RegisterMatmul(tile);
            dispatch.workgroups = matmul::register_workgroups(tile, m, n, batch);
        }
    }
}

/// Grow physical capacity of `buffer` to at least `bytes` while preserving its
/// original logical byte length in `plan.logical_buffers`.
fn pad_buffer(plan: &mut ExecutionPlan, buffer: BufferRef, bytes: usize) {
    let idx = buffer.0 as usize;
    if bytes <= plan.buffers[idx] {
        return;
    }
    if plan.logical_buffers.is_empty() {
        plan.logical_buffers = plan.buffers.clone();
    } else if plan.logical_buffers.len() < plan.buffers.len() {
        let start = plan.logical_buffers.len();
        plan.logical_buffers
            .extend_from_slice(&plan.buffers[start..]);
    }
    plan.buffers[idx] = bytes;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Graph, compile};

    const TILE: WorkgroupMatmulShape = WorkgroupMatmulShape {
        rows: 64,
        columns: 64,
        k_stage: 32,
        threads: 64,
        stage_operands: false,
        subgroup_rows: 0,
        prefetch: false,
    };

    fn caps() -> blade_graphics::Capabilities {
        let mut caps = blade_graphics::Capabilities {
            max_compute_shared_memory_size: 32768,
            shader_float16: true,
            shader_bfloat16: true,
            ..Default::default()
        };
        caps.cooperative_matrix.workgroup = blade_graphics::WorkgroupCooperativeMatrix {
            granularities: vec![blade_graphics::CooperativeMatrixGranularity {
                operand: blade_graphics::CooperativeMatrixOperand::F16,
                size: [16, 16, 16],
                invocations: 64,
            }],
            max_workgroup_size: 256,
            max_dimension: 1024,
            reserved_shared_memory: 8192,
        };
        caps
    }

    fn matmul_plan(m: usize, n: usize, k: usize) -> ExecutionPlan {
        let mut g = Graph::new();
        let a = g.input("a", &[m, k]);
        let b = g.input("b", &[k, n]);
        let y = g.matmul(a, b);
        g.set_outputs(vec![y]);
        compile::compile(&g)
    }

    #[test]
    fn small_block_shapes_preserve_nvidia_scalar_geometry() {
        let coop = CoopConfig {
            tile_size: 16,
            use_f16_input: true,
        };
        for rows in [6, 16] {
            for (inner, columns) in [(1024, 256), (256, 768)] {
                let mut serial = Graph::new();
                let a = serial.input("a", &[rows, inner]);
                let b = serial.parameter("b", &[inner, columns]);
                let y = serial.matmul(a, b);
                serial.set_outputs(vec![y]);
                let mut serial = compile::compile(&serial);
                select_variants(&mut serial, Some(&coop), false, false);

                let mut grouped = Graph::new();
                let a = grouped.input("a", &[rows, 8 * inner]);
                let b = grouped.parameter("b", &[8, inner, columns]);
                let y = grouped.block_matmul(a, b);
                grouped.set_outputs(vec![y]);
                let mut grouped = compile::compile(&grouped);
                select_variants(&mut grouped, Some(&coop), false, false);
                let serial = &serial.dispatches[0];
                let grouped = &grouped.dispatches[0];
                assert!(!serial.use_coop() && !grouped.use_coop());
                let Kernel::SplitMatmul { shape, .. } = serial.kernel else {
                    panic!("{rows}x{inner}x{columns} is not split: {:?}", serial.kernel);
                };
                assert_eq!(shape.tile_size == 32, grouped.use_small_tiles());
                assert_eq!(serial.workgroups[..2], grouped.workgroups[..2]);
                assert_eq!(grouped.workgroups[2], 8);
            }
        }
    }

    #[test]
    fn flash_f16_tiles_follow_the_session_policy() {
        use crate::compile::CompileOptions;
        use crate::kernels::attention_grad::Path;

        let training_plan = |full_precision: bool, backward_tiles: bool| {
            let mut graph = Graph::new();
            let q = graph.parameter("q", &[128, 64]);
            let k = graph.parameter("k", &[128, 64]);
            let v = graph.parameter("v", &[128, 64]);
            if full_precision {
                graph.begin_full_precision_region();
            }
            let y = graph.full_attention(q, k, v, 1, 1, 64);
            let loss = graph.sum_all(y);
            graph.set_outputs(vec![loss]);
            let graph = crate::autodiff::differentiate(&graph);
            let options = CompileOptions {
                flash_f16_tiles: backward_tiles,
                flash_forward_coop: false,
                ..CompileOptions::default()
            };
            compile::compile_with_caps(&graph, &options, CoopCaps::default(), 65_536)
        };
        let tiles = |plan: &ExecutionPlan, forward: bool| -> bool {
            let flags: Vec<bool> = plan
                .dispatches
                .iter()
                .filter(|dispatch| match dispatch.shader {
                    ShaderEntry::FlashAttention => forward,
                    ShaderEntry::AttentionGrad(kernel) => !forward && kernel.path == Path::Flash,
                    _ => false,
                })
                .map(|dispatch| dispatch.attention_f16_tiles)
                .collect();
            assert!(
                !flags.is_empty(),
                "missing flash dispatch, forward={forward}"
            );
            assert!(flags.iter().all(|tiles| *tiles == flags[0]));
            flags[0]
        };

        let ordinary = training_plan(false, false);
        assert!(!tiles(&ordinary, true));
        assert!(!tiles(&ordinary, false));
        let requested = training_plan(false, true);
        assert!(tiles(&requested, true));
        assert!(tiles(&requested, false));

        for (policy, want) in [
            (CoopPolicy::NativeF32, false),
            (CoopPolicy::Disabled, false),
            (CoopPolicy::Auto, true),
            (CoopPolicy::AllowF16, true),
        ] {
            let mut plan = requested.clone();
            apply_attention_precision(&mut plan, policy);
            assert_eq!(tiles(&plan, true), want);
            assert_eq!(tiles(&plan, false), want);
        }

        let mut blocked = requested.clone();
        for d in &mut blocked.dispatches {
            if matches!(d.shader, ShaderEntry::AttentionGrad(_)) {
                d.kernel = Kernel::BlockAttention;
            }
        }
        apply_attention_precision(&mut blocked, CoopPolicy::AllowF16);
        assert!(!tiles(&blocked, false));
        assert!(tiles(&blocked, true));

        let mut pinned = training_plan(true, true);
        apply_attention_precision(&mut pinned, CoopPolicy::AllowF16);
        assert!(!tiles(&pinned, true));
        assert!(!tiles(&pinned, false));
    }

    #[test]
    fn large_workgroup_tiles_respect_device_memory_and_dimensions() {
        let mut device = caps();
        device.max_compute_shared_memory_size = 49152;
        device.cooperative_matrix.workgroup.granularities = [
            blade_graphics::CooperativeMatrixOperand::F16,
            blade_graphics::CooperativeMatrixOperand::Bf16,
        ]
        .map(|operand| blade_graphics::CooperativeMatrixGranularity {
            operand,
            size: [16, 16, 16],
            invocations: 256,
        })
        .to_vec();
        let tile = WorkgroupMatmulShape {
            rows: 128,
            columns: 256,
            k_stage: 32,
            threads: 256,
            stage_operands: false,
            subgroup_rows: 0,
            prefetch: false,
        };
        assert!(tile.supports(&device, MatrixTypes::F32_AS_F16, 1536, 512, 384));
        let deeper = WorkgroupMatmulShape {
            k_stage: 64,
            ..tile
        };
        assert!(!deeper.supports(&device, MatrixTypes::F32_AS_F16, 1536, 512, 384));
        assert!(deeper.supports(&device, MatrixTypes::BF16, 1536, 512, 384));
        assert!(
            !WorkgroupMatmulShape {
                stage_operands: true,
                ..deeper
            }
            .supports(&device, MatrixTypes::BF16, 1536, 512, 384)
        );
        device.cooperative_matrix.workgroup.max_dimension = 128;
        assert!(!tile.supports(&device, MatrixTypes::BF16, 1536, 512, 384));
        assert!(!tile.supports(&device, MatrixTypes::F32_AS_F16, 1536, 512, 384));

        let overflow = WorkgroupMatmulShape {
            rows: u32::MAX - 3,
            columns: 256,
            ..tile
        };
        device.cooperative_matrix.workgroup.max_dimension = u32::MAX;
        assert_eq!(overflow.shared_bytes(MatrixTypes::F32_AS_F16, false), None);
        assert!(!overflow.supports(&device, MatrixTypes::F32_AS_F16, 1536, 512, 384));
    }

    #[test]
    fn workgroup_selection_respects_precision_device_and_shape() {
        for (dims, policy, device, expected) in [
            ((128, 256, 128), CoopPolicy::AllowF16, caps(), true),
            ((129, 256, 128), CoopPolicy::AllowF16, caps(), true),
            ((128, 257, 128), CoopPolicy::AllowF16, caps(), false),
            ((128, 256, 129), CoopPolicy::AllowF16, caps(), false),
            ((128, 256, 128), CoopPolicy::NativeF32, caps(), false),
            ((128, 256, 128), CoopPolicy::Auto, caps(), false),
            ((128, 256, 128), CoopPolicy::Disabled, caps(), false),
            (
                (128, 256, 128),
                CoopPolicy::AllowF16,
                blade_graphics::Capabilities::default(),
                false,
            ),
        ] {
            let mut plan = matmul_plan(dims.0, dims.1, dims.2);
            select_workgroup_matmuls(&mut plan, &device, Some(TILE), policy);
            assert_eq!(
                plan.dispatches
                    .iter()
                    .any(|d| matches!(d.kernel, Kernel::CooperativeWorkgroup { .. })),
                expected
            );
        }
        let mut plan = matmul_plan(128, 256, 128);
        let matrix = plan
            .dispatches
            .iter_mut()
            .find(|d| d.mnk().is_some())
            .unwrap();
        matrix.requires_full_precision = true;
        select_workgroup_matmuls(&mut plan, &caps(), Some(TILE), CoopPolicy::AllowF16);
        assert!(
            !plan
                .dispatches
                .iter()
                .any(|d| matches!(d.kernel, Kernel::CooperativeWorkgroup { .. }))
        );
    }

    #[test]
    fn saved_workgroup_choice_is_reset_before_device_selection() {
        for bf16 in [false, true] {
            let mut plan = matmul_plan(128, 256, 128);
            let matrix = plan
                .dispatches
                .iter_mut()
                .find(|d| d.mnk().is_some())
                .unwrap();
            matrix.kernel = Kernel::CooperativeWorkgroup {
                shape: TILE,
                types: bf16.then_some(MatrixTypes::BF16),
                splits: 1,
            };
            matrix.workgroups = [2, 4, 1];
            let encoded = ron::to_string(&plan).unwrap();
            let mut decoded: ExecutionPlan = ron::from_str(&encoded).unwrap();
            reset_workgroup_matmuls(&mut decoded);
            let matrix = decoded
                .dispatches
                .iter()
                .find(|d| d.mnk().is_some())
                .unwrap();
            if bf16 {
                assert_eq!(
                    matrix.kernel,
                    Kernel::CooperativeTyped {
                        types: MatrixTypes::BF16,
                        splits: 1,
                        specialize: false,
                    }
                );
                assert_eq!(matrix.workgroups, [4, 8, 1]);
            } else {
                assert_eq!(matrix.kernel, Kernel::Default);
                assert_eq!(matrix.workgroups, [4, 2, 1]);
            }
        }
    }

    #[test]
    fn finalize_for_target_preserves_logical_buffer_sizes_when_padding() {
        // 385x384 output requires cooperative tile padding on M (385 -> 416)
        // when lowered to F16 cooperative tiles, while its logical byte size
        // remains exact.
        let mut plan = matmul_plan(385, 384, 64);
        let out_buf = plan.output_buffers[0];
        let logical = plan.buffers[out_buf.0 as usize];
        assert_eq!(logical, 385 * 384 * 4);

        let coop = CoopConfig {
            tile_size: 16,
            use_f16_input: true,
        };
        let device = caps();
        plan.finalize_for_target(&LowerTarget {
            coop_config: Some(&coop),
            capabilities: &device,
            coop_policy: CoopPolicy::AllowF16,
            workgroup_matmul: None,
            serial_dispatch: false,
            debug: false,
            prepare_outputs: &[],
        });

        assert!(plan.buffers[out_buf.0 as usize] > logical);
        assert_eq!(plan.logical_buffer_bytes(out_buf), logical);
        assert!(!plan.groups.is_empty());
    }
}
