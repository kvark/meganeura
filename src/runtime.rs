use crate::compile::{BufferRef, CachedBlockAttentionParams, Dispatch, ExecutionPlan, ShaderEntry};
use crate::kernels::attention_grad::Part as AttentionGradPart;
use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

mod checkpoint;
mod optimizer;
pub(crate) mod search_state;
mod tuning;
pub use crate::tune::TuneOutcome;
pub use tuning::KernelMemo;

type Gpu = blade_graphics::Context;

/// Wait for a command encoder and harvest the submission's timestamps.
///
/// Blade resolves on demand: `last_timing` describes the submission that was
/// just waited on, and borrows its pass names from the encoder, so they are
/// copied out before the next submission reuses that storage.
///
/// `timing` must say whether this context was created with
/// [`blade_graphics::ContextDesc::timing`]. Asking an encoder for timings it
/// was not set up to collect panics, and `Capabilities::timing` reports only
/// what the device *can* do, not what this context enabled — so the answer
/// has to travel with the context rather than be inferred from it. See
/// [`SessionOptions::gpu_timing`].
///
/// Temporary transfer encoders used during model loading go through here too,
/// so their spans reach the trace before they are destroyed.
pub(super) fn wait_for_timed_encoder(
    gpu: &Gpu,
    sync: &blade_graphics::SyncPoint,
    encoder: &mut blade_graphics::CommandEncoder,
    timing: bool,
) -> Result<Option<crate::profiler::GpuTimings>, blade_graphics::DeviceError> {
    let result = gpu.wait_for(sync, !0);
    if !result? {
        return Ok(None);
    }
    if !timing || !gpu.capabilities().timing {
        return Ok(None);
    }
    let borrowed = encoder.last_timing();
    let timings = crate::profiler::GpuTimings {
        passes: borrowed
            .passes
            .iter()
            .map(|&(name, at)| (name.to_owned(), at))
            .collect(),
        done: Some(borrowed.done),
    };
    tracing::debug!(passes = timings.passes.len(), "GPU timestamps resolved");
    crate::profiler::record_gpu_timings(&timings);
    Ok(Some(timings))
}

/// Leave room for pipelines, command buffers, and driver-owned allocations
/// that are not represented by the execution plan's buffer sizes.
const DEVICE_MEMORY_SAFE_PERCENT: u64 = 90;
const DEVICE_MEMORY_RESERVE_PERCENT: u64 = 100 - DEVICE_MEMORY_SAFE_PERCENT;

/// One compute pass of a profiled [`Session::step`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum ProfilePass {
    /// A single dispatch, timestamped under its own label.
    Timed(usize),
    /// Dispatches outside the timing window, batched into one pass. Each
    /// range is the part of one barrier group that falls in this pass; a
    /// barrier separates consecutive ranges, exactly as in normal execution.
    Untimed(Vec<std::ops::Range<usize>>),
}

/// Clip `groups` to `span`, dropping the groups that fall outside it.
fn clip_groups(
    groups: &[std::ops::Range<usize>],
    span: std::ops::Range<usize>,
) -> Vec<std::ops::Range<usize>> {
    groups
        .iter()
        .filter_map(|group| {
            let start = group.start.max(span.start);
            let end = group.end.min(span.end);
            (start < end).then_some(start..end)
        })
        .collect()
}

/// Lay out the compute passes for one profiled step.
///
/// Dispatches inside `window` get a pass each, so Blade timestamps them
/// individually. The rest keep the plan's barrier structure but collapse into
/// a single pass on either side: Blade stops writing timestamps once a
/// submission reaches `limits::PASS_COUNT`, and a pass per barrier group
/// outside the window would spend that budget on dispatches nobody is
/// measuring. Every dispatch still runs exactly once, in plan order.
pub(crate) fn profile_pass_plan(
    groups: &[std::ops::Range<usize>],
    dispatch_count: usize,
    window: std::ops::Range<usize>,
) -> Vec<ProfilePass> {
    let start = window.start.min(dispatch_count);
    let end = window.end.clamp(start, dispatch_count);

    let mut passes = Vec::with_capacity(end - start + 2);
    let head = clip_groups(groups, 0..start);
    if !head.is_empty() {
        passes.push(ProfilePass::Untimed(head));
    }
    passes.extend((start..end).map(ProfilePass::Timed));
    let tail = clip_groups(groups, end..dispatch_count);
    if !tail.is_empty() {
        passes.push(ProfilePass::Untimed(tail));
    }
    passes
}

fn safe_device_memory_remaining(usage: u64, budget: u64) -> u64 {
    let safe_limit = ((budget as u128 * DEVICE_MEMORY_SAFE_PERCENT as u128) / 100u128) as u64;
    safe_limit.saturating_sub(usage)
}

fn ensure_device_memory_budget(gpu: &Gpu, requested: usize, allocation: &str) {
    let stats = gpu.memory_stats();
    if stats.budget == 0 {
        return;
    }
    let requested = u64::try_from(requested).expect("GPU allocation request exceeds u64");
    let remaining = safe_device_memory_remaining(stats.usage, stats.budget);
    assert!(
        requested <= remaining,
        "refusing to oversubscribe GPU memory for {allocation}: requested {:.2} GB with {:.2} GB already used, but the device reports a {:.2} GB budget and Meganeura reserves {DEVICE_MEMORY_RESERVE_PERCENT}% for pipelines and the driver ({:.2} GB safely available); reduce the model, sequence, batch, or microbatch geometry",
        requested as f64 / 1e9,
        stats.usage as f64 / 1e9,
        stats.budget as f64 / 1e9,
        remaining as f64 / 1e9,
    );
    log::debug!(
        "device-memory preflight for {allocation}: {:.1} MB requested, {:.1} MB used, {:.1} MB budget",
        requested as f64 / 1e6,
        stats.usage as f64 / 1e6,
        stats.budget as f64 / 1e6,
    );
}

// scatter_add: var indices (u32), src, dst, params
#[derive(blade_macros::ShaderData)]
struct ScatterAddData {
    indices: blade_graphics::BufferPiece,
    src: blade_graphics::BufferPiece,
    dst: blade_graphics::BufferPiece,
    params: ScatterAddParams,
}

// The ordinary and zeroing entries bind `src` again for their unused row
// scale so every entry point can share one reflected layout.
#[derive(blade_macros::ShaderData)]
struct ScatterAddAtomicData {
    indices: blade_graphics::BufferPiece,
    src: blade_graphics::BufferPiece,
    row_scale: blade_graphics::BufferPiece,
    dst: blade_graphics::BufferPiece,
    params: ScatterAddParams,
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct ScatterAddParams {
    total: u32,
    seq_len: u32,
    embed_dim: u32,
    _pad: u32,
}

/// Summary of GPU memory allocation for a session.
#[derive(Clone, Debug)]
pub struct MemorySummary {
    /// Sum of plan buffer capacities before aliasing, including runtime padding.
    pub total_buffer_bytes: usize,
    /// Resident Adam/LaProp moments; zero until configuration, write or restore.
    pub adam_state_bytes: usize,
    pub grad_accumulator_bytes: usize,
    /// Clip scalar and optional grouped-gradient diagnostic storage.
    pub optimizer_aux_bytes: usize,
    pub num_buffers: usize,
    pub largest_buffer_bytes: usize,
    /// Bytes actually allocated on the GPU after lifetime-based buffer
    /// aliasing, excluding optimizer state and staging.
    pub allocated_buffer_bytes: usize,
    /// Number of physical allocations backing the logical buffers.
    pub num_allocations: usize,
    /// Physical-slot bytes requested with device-local policy.
    /// This is not a query of backend-selected memory types; allocations
    /// may fall back to a different heap.
    pub device_local_bytes: usize,
}

impl MemorySummary {
    /// Resident tensor/optimizer allocation requests, excluding driver objects
    /// and staging. Not a peak or device-budget measurement.
    pub fn total_allocated_bytes(&self) -> usize {
        self.allocated_buffer_bytes
            + self.adam_state_bytes
            + self.grad_accumulator_bytes
            + self.optimizer_aux_bytes
    }
}

/// Per-process GPU memory usage as reported by the graphics API.
///
/// This is the only memory figure that is directly comparable against
/// another engine: unlike [`MemorySummary`], which describes what the
/// execution plan asked for, this includes driver allocations, pipeline
/// objects, and staging that the plan does not account for.
///
/// The value is scoped to the calling process on both backends, so an
/// unrelated workload sharing the GPU does not contaminate it.
#[derive(Clone, Copy, Debug)]
pub struct DeviceMemoryStats {
    /// Bytes currently allocated on the device by this process.
    ///
    /// Vulkan sums `VK_EXT_memory_budget` heap usage over device-local
    /// heaps; Metal reports `MTLDevice.currentAllocatedSize`.
    pub usage_bytes: u64,
    /// Device-local memory available to this process, per the same query.
    pub budget_bytes: u64,
}

impl std::fmt::Display for MemorySummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} graph buffers in {} allocations, {:.1} MB resident ({:.1} MB graph, {:.1} MB device-local, {:.1} MB logical, {:.1} MB adam, {:.1} MB accumulators), largest {:.1} MB",
            self.num_buffers,
            self.num_allocations,
            self.total_allocated_bytes() as f64 / 1e6,
            self.allocated_buffer_bytes as f64 / 1e6,
            self.device_local_bytes as f64 / 1e6,
            self.total_buffer_bytes as f64 / 1e6,
            self.adam_state_bytes as f64 / 1e6,
            self.grad_accumulator_bytes as f64 / 1e6,
            self.largest_buffer_bytes as f64 / 1e6,
        )
    }
}

// ---- ShaderData structs matching codegen global variable names ----

// matmul: var matrix_a, matrix_b, matrix_c, params
#[derive(blade_macros::ShaderData)]
struct MatMulData {
    matrix_a: blade_graphics::BufferPiece,
    matrix_b: blade_graphics::BufferPiece,
    matrix_c: blade_graphics::BufferPiece,
    params: MatMulParams,
}

// MatMul coop with 2-factor prologue — bindings match `matmul_prologue_to_wgsl`
// generated names (`prologue_buf_0`, `prologue_buf_1`).
#[derive(blade_macros::ShaderData)]
struct MatMulPrologue2Data {
    matrix_a: blade_graphics::BufferPiece,
    matrix_b: blade_graphics::BufferPiece,
    matrix_c: blade_graphics::BufferPiece,
    prologue_buf_0: blade_graphics::BufferPiece,
    prologue_buf_1: blade_graphics::BufferPiece,
    params: MatMulParams,
}

// fused_matmul_add: var matrix_a, matrix_b, matrix_c, src (addend), params
#[derive(blade_macros::ShaderData)]
struct FusedMatMulAddData {
    matrix_a: blade_graphics::BufferPiece,
    matrix_b: blade_graphics::BufferPiece,
    matrix_c: blade_graphics::BufferPiece,
    src: blade_graphics::BufferPiece, // addend buffer (named "src" to match codegen)
    params: MatMulParams,
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct MatMulParams {
    m: u32,
    n: u32,
    k: u32,
    _pad: u32,
}

// unary: var src, dst, params
#[derive(blade_macros::ShaderData)]
struct UnaryData {
    src: blade_graphics::BufferPiece,
    dst: blade_graphics::BufferPiece,
    params: UnaryParams,
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct UnaryParams {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// permute: var src, dst, params (eight words)
#[derive(blade_macros::ShaderData)]
pub(crate) struct PermuteData {
    pub(crate) src: blade_graphics::BufferPiece,
    pub(crate) dst: blade_graphics::BufferPiece,
    pub(crate) params: PermuteParams,
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
pub(crate) struct PermuteParams {
    pub(crate) total: u32,
    pub(crate) dims: [u32; 3],
    pub(crate) strides: [u32; 4],
}

// biased attention: q, k, v, bias, kv_pos, dst, params (eight words)
#[derive(blade_macros::ShaderData)]
pub(crate) struct BiasedAttentionData {
    pub(crate) q: blade_graphics::BufferPiece,
    pub(crate) k: blade_graphics::BufferPiece,
    pub(crate) v: blade_graphics::BufferPiece,
    pub(crate) bias: blade_graphics::BufferPiece,
    pub(crate) kv_pos: blade_graphics::BufferPiece,
    pub(crate) dst: blade_graphics::BufferPiece,
    pub(crate) params: [u32; 8],
}

// binary: var src_a, src_b, dst, params
#[derive(blade_macros::ShaderData)]
struct BinaryData {
    src_a: blade_graphics::BufferPiece,
    src_b: blade_graphics::BufferPiece,
    dst: blade_graphics::BufferPiece,
    params: UnaryParams, // same layout: len + padding
}

// ternary (swiglu_grad_gate): var src_a, src_b, src_c, dst, params
#[derive(blade_macros::ShaderData)]
struct TernaryData {
    src_a: blade_graphics::BufferPiece,
    src_b: blade_graphics::BufferPiece,
    src_c: blade_graphics::BufferPiece,
    dst: blade_graphics::BufferPiece,
    params: UnaryParams,
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct BiasAddParams {
    len: u32,
    bias_len: u32,
    _pad0: u32,
    _pad1: u32,
}

// mul_per_channel: var src, gate, dst, params
#[derive(blade_macros::ShaderData)]
struct MulPerChannelData {
    src: blade_graphics::BufferPiece,
    gate: blade_graphics::BufferPiece,
    dst: blade_graphics::BufferPiece,
    params: MulPerChannelParams,
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct MulPerChannelParams {
    len: u32,
    spatial: u32,
    _pad0: u32,
    _pad1: u32,
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct AddPerChannelParams {
    len: u32,
    spatial: u32,
    channels: u32,
    _pad0: u32,
}

// reduce: var src, dst, params (same layout as UnaryData)

// Schedule-template reduction params: { outer, inner, round_one_bits, _pad1 }.
// 16 bytes; same layout as BiasAddParams with a different name for
// clarity at the call site.
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct ReductionParams {
    outer: u32,
    inner: u32,
    round_one_bits: u32,
    table_rows: u32,
}

// Schedule-template reduction bindings — arity 1 (pure): var src, dst, params.
#[derive(blade_macros::ShaderData)]
struct ReductionPass1Data {
    src: blade_graphics::BufferPiece,
    dst: blade_graphics::BufferPiece,
    params: ReductionParams,
}

// Schedule-template reduction bindings — arity 2 with 1 per-elem + 1 per-row.
// Used e.g. for softmax pass 2 (src + row_max).
#[derive(blade_macros::ShaderData)]
struct ReductionPass2RowData {
    src: blade_graphics::BufferPiece,
    per_row_src: blade_graphics::BufferPiece,
    dst: blade_graphics::BufferPiece,
    params: ReductionParams,
}

// rms_norm: var src, bias (weight), dst, params
#[derive(blade_macros::ShaderData)]
struct RmsNormData {
    src: blade_graphics::BufferPiece,
    bias: blade_graphics::BufferPiece, // weight, named "bias" to match binding
    dst: blade_graphics::BufferPiece,
    params: BiasAddParams, // reuse: rows=len, cols=bias_len, _pad x2
}

// split-K combine: var partials, dst, params
#[derive(blade_macros::ShaderData)]
struct CachedBlockAttentionCombineData {
    partials: blade_graphics::BufferPiece,
    dst: blade_graphics::BufferPiece,
    params: CachedBlockAttentionParams,
}

// rms_norm_add: var src, bias (weight), residual, dst, params
#[derive(blade_macros::ShaderData)]
struct RmsNormAddData {
    src: blade_graphics::BufferPiece,
    bias: blade_graphics::BufferPiece,
    residual: blade_graphics::BufferPiece,
    dst: blade_graphics::BufferPiece,
    params: BiasAddParams,
}

// embedding: var indices (u32), src (table), dst, params
#[derive(blade_macros::ShaderData)]
struct EmbeddingData {
    indices: blade_graphics::BufferPiece,
    src: blade_graphics::BufferPiece,
    dst: blade_graphics::BufferPiece,
    params: UnaryParams, // seq in len field
}

// rope: var src, dst, params
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct RoPEParams {
    seq: u32,
    dim: u32,
    theta_bits: u32,
    pos_offset: u32,
    head_dim: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[derive(blade_macros::ShaderData)]
struct RoPEData {
    src: blade_graphics::BufferPiece,
    dst: blade_graphics::BufferPiece,
    params: RoPEParams,
}

// 4-buffer ops: src_a, src_b, bias, dst, params (rms_norm_grad, fused_rms_norm_matmul, etc.)
#[derive(blade_macros::ShaderData)]
struct FourBufData {
    src_a: blade_graphics::BufferPiece,
    src_b: blade_graphics::BufferPiece,
    bias: blade_graphics::BufferPiece,
    dst: blade_graphics::BufferPiece,
    params: MatMulParams,
}

// rope_dynamic: var src, dst, pos_offset_buf, params
#[derive(blade_macros::ShaderData)]
struct RoPEDynamicData {
    src: blade_graphics::BufferPiece,
    dst: blade_graphics::BufferPiece,
    pos_offset_buf: blade_graphics::BufferPiece,
    params: RoPEParams,
}

// rope_dynamic with per-pair divisors: the same four, plus the factors.
#[derive(blade_macros::ShaderData)]
struct RoPEDynamicFactorsData {
    src: blade_graphics::BufferPiece,
    dst: blade_graphics::BufferPiece,
    pos_offset_buf: blade_graphics::BufferPiece,
    factors: blade_graphics::BufferPiece,
    params: RoPEParams,
}

// cache_write: var src, dst (read_write), kv_pos_buf, params
#[derive(blade_macros::ShaderData)]
struct CacheWriteData {
    src: blade_graphics::BufferPiece,
    dst: blade_graphics::BufferPiece,
    kv_pos_buf: blade_graphics::BufferPiece,
    params: UnaryParams, // dim, _pad x3
}

#[derive(blade_macros::ShaderData)]
struct CacheWritePrefixData {
    src: blade_graphics::BufferPiece,
    dst: blade_graphics::BufferPiece,
    kv_pos_buf: blade_graphics::BufferPiece,
    valid_len_buf: blade_graphics::BufferPiece,
    params: MatMulParams, // dim, block_len, max_seq, _pad
}

// group_norm: var src, src_b (weight), bias, dst, params
#[derive(blade_macros::ShaderData)]
struct GroupNormData {
    src: blade_graphics::BufferPiece,
    src_b: blade_graphics::BufferPiece,
    bias: blade_graphics::BufferPiece,
    dst: blade_graphics::BufferPiece,
    params: GroupNormParams,
}

/// Statistics pass: reads the tensor, writes (sum, M2) per slice.
#[derive(blade_macros::ShaderData)]
struct GroupNormStatsData {
    src: blade_graphics::BufferPiece,
    dst: blade_graphics::BufferPiece,
    params: GroupNormParams,
}

/// Normalisation pass: reads the tensor and the partials, writes the result.
#[derive(blade_macros::ShaderData)]
struct GroupNormApplyData {
    src: blade_graphics::BufferPiece,
    src_b: blade_graphics::BufferPiece,
    bias: blade_graphics::BufferPiece,
    dst: blade_graphics::BufferPiece,
    partials: blade_graphics::BufferPiece,
    params: GroupNormParams,
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct GroupNormParams {
    batch: u32,
    channels: u32,
    spatial: u32,
    num_groups: u32,
    eps_bits: u32,
    /// Slices each group is split into, so the parallelism follows the image
    /// rather than the batch. One for the legacy single-pass kernels.
    chunks: u32,
    apply_silu: u32,
    _pad2: u32,
}

// group_norm_grad: var src_a (grad_out), src_b (input), bias (weight), dst, params
// GroupNorm backward: every entry point of the module binds the same set.
// grad_stats reads src_b and writes dst; the gradients read the statistics.
#[derive(blade_macros::ShaderData)]
struct GroupNormGradData {
    src_a: blade_graphics::BufferPiece,
    src_b: blade_graphics::BufferPiece,
    bias: blade_graphics::BufferPiece,
    dst: blade_graphics::BufferPiece,
    stats: blade_graphics::BufferPiece,
    params: GroupNormParams,
}

// concat: var src_a, src_b, dst, params
// (reuses BinaryData layout with UnaryParams → batch, ca, cb, spatial)

// split: var src, dst, params (reuses UnaryData layout)

// upsample: var src, dst, params (reuses UnaryData layout)

// winograd transform: var src, dst, params (8 u32s)
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct WinogradTransformParams {
    p0: u32,
    p1: u32,
    p2: u32,
    p3: u32,
    p4: u32,
    p5: u32,
    p6: u32,
    p7: u32,
}

#[derive(blade_macros::ShaderData)]
struct WinogradTransformData {
    src: blade_graphics::BufferPiece,
    dst: blade_graphics::BufferPiece,
    params: WinogradTransformParams,
}

// conv2d: var src, weight, dst, params (12 u32s = 3 uniform vec4s)
#[derive(blade_macros::ShaderData)]
struct Conv2dData {
    src: blade_graphics::BufferPiece,
    weight: blade_graphics::BufferPiece,
    dst: blade_graphics::BufferPiece,
    params: Conv2dParams,
}

// conv2d_grad_input: var grad_out, weight, dst, params
#[derive(blade_macros::ShaderData)]
struct Conv2dGradInputData {
    grad_out: blade_graphics::BufferPiece,
    weight: blade_graphics::BufferPiece,
    dst: blade_graphics::BufferPiece,
    params: Conv2dParams,
}

// conv2d_grad_weight: var grad_out, src, dst, params
#[derive(blade_macros::ShaderData)]
struct Conv2dGradWeightData {
    grad_out: blade_graphics::BufferPiece,
    src: blade_graphics::BufferPiece,
    dst: blade_graphics::BufferPiece,
    params: Conv2dParams,
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct Conv2dParams {
    batch: u32,
    in_channels: u32,
    in_h: u32,
    in_w: u32,
    out_channels: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride: u32,
    padding_h: u32,
    out_h: u32,
    out_w: u32,
    padding_w: u32,
    kernel_w_multiplier: u32,
    kernel_hw_multiplier: u32,
    column_width_multiplier: u32,
    output_spatial_multiplier: u32,
}

impl From<&Dispatch> for Conv2dParams {
    fn from(dispatch: &Dispatch) -> Self {
        let p = &dispatch.params;
        let column_width = match dispatch.shader {
            ShaderEntry::Conv2dGradInputGemm
            | ShaderEntry::Conv2dGradInputGemmSmall
            | ShaderEntry::Conv2dGradInputGemm16
            | ShaderEntry::Conv2dGradInputGemmCoopGen(..) => p[3],
            _ => p[10],
        };
        Self {
            batch: p[0],
            in_channels: p[1],
            in_h: p[2],
            in_w: p[3],
            out_channels: p[4],
            kernel_h: p[5],
            kernel_w: p[6],
            stride: p[7],
            padding_h: p[8],
            out_h: p[9],
            out_w: p[10],
            padding_w: p[11],
            kernel_w_multiplier: crate::divisor::multiplier(p[6]),
            kernel_hw_multiplier: crate::divisor::multiplier(p[5] * p[6]),
            column_width_multiplier: crate::divisor::multiplier(column_width),
            output_spatial_multiplier: crate::divisor::multiplier(p[9] * p[10]),
        }
    }
}

// conv2d_dw: var src, weight, dst, params (depthwise, no in/out channels split)
#[derive(blade_macros::ShaderData)]
struct Conv2dDwData {
    src: blade_graphics::BufferPiece,
    weight: blade_graphics::BufferPiece,
    dst: blade_graphics::BufferPiece,
    params: Conv2dDwParams,
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct Conv2dDwParams {
    batch: u32,
    channels: u32,
    in_h: u32,
    in_w: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride: u32,
    padding_h: u32,
    out_h: u32,
    out_w: u32,
    padding_w: u32,
    _pad: u32,
}

// max_pool_2d: var src, dst, params
#[derive(blade_macros::ShaderData)]
struct MaxPool2dData {
    src: blade_graphics::BufferPiece,
    dst: blade_graphics::BufferPiece,
    params: MaxPool2dParams,
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct MaxPool2dParams {
    batch: u32,
    channels: u32,
    in_h: u32,
    in_w: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride: u32,
    padding: u32,
    out_h: u32,
    out_w: u32,
    _pad0: u32,
    _pad1: u32,
}

// max_pool_2d_grad: var grad_out, src, dst, params
#[derive(blade_macros::ShaderData)]
struct MaxPool2dGradData {
    grad_out: blade_graphics::BufferPiece,
    src: blade_graphics::BufferPiece,
    dst: blade_graphics::BufferPiece,
    params: MaxPool2dParams,
}

// global_avg_pool: var src, dst, params
#[derive(blade_macros::ShaderData)]
struct GlobalAvgPoolData {
    src: blade_graphics::BufferPiece,
    dst: blade_graphics::BufferPiece,
    params: GlobalAvgPoolParams,
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct GlobalAvgPoolParams {
    channels: u32,
    spatial: u32,
    total_out: u32,
    _pad: u32,
}

// cached_attention: var src_a (q), src_b (k_cache), bias (v_cache), kv_pos_buf, dst, params
#[derive(blade_macros::ShaderData)]
struct CachedAttentionData {
    src_a: blade_graphics::BufferPiece,
    src_b: blade_graphics::BufferPiece,
    bias: blade_graphics::BufferPiece,
    kv_pos_buf: blade_graphics::BufferPiece,
    dst: blade_graphics::BufferPiece,
    params: CachedAttentionParams,
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct CachedAttentionParams {
    queries: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    max_seq: u32,
    _pad: [u32; 3],
}

#[derive(blade_macros::ShaderData)]
struct CachedBlockAttentionData {
    src_a: blade_graphics::BufferPiece,
    src_b: blade_graphics::BufferPiece,
    bias: blade_graphics::BufferPiece,
    kv_pos_buf: blade_graphics::BufferPiece,
    valid_len_buf: blade_graphics::BufferPiece,
    dst: blade_graphics::BufferPiece,
    params: CachedBlockAttentionParams,
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct ChunkedRelativeAttentionParams {
    seq_len: u32,
    num_heads: u32,
    head_dim: u32,
    left_context: u32,
    softcap_bits: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[derive(blade_macros::ShaderData)]
struct ChunkedRelativeAttentionData {
    src_a: blade_graphics::BufferPiece,
    src_b: blade_graphics::BufferPiece,
    bias: blade_graphics::BufferPiece,
    relative_k: blade_graphics::BufferPiece,
    dst: blade_graphics::BufferPiece,
    params: ChunkedRelativeAttentionParams,
}

#[derive(blade_macros::ShaderData)]
struct PrefixLastData {
    src: blade_graphics::BufferPiece,
    valid_len_buf: blade_graphics::BufferPiece,
    dst: blade_graphics::BufferPiece,
    params: MatMulParams, // cols, rows, _pad0, _pad1
}

// layer_norm: var src, src_b (weight), bias, dst, params
#[derive(blade_macros::ShaderData)]
struct LayerNormData {
    src: blade_graphics::BufferPiece,
    src_b: blade_graphics::BufferPiece, // weight
    bias: blade_graphics::BufferPiece,  // bias
    dst: blade_graphics::BufferPiece,
    params: MatMulParams, // rows, cols, eps_bits, _pad
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct SoftmaxParams {
    batch: u32,
    features: u32,
    _pad0: u32,
    _pad1: u32,
}

// cross_entropy: var logits, labels, grad_out, loss_out, params
#[derive(blade_macros::ShaderData)]
struct CrossEntropyData {
    logits: blade_graphics::BufferPiece,
    labels: blade_graphics::BufferPiece,
    grad_out: blade_graphics::BufferPiece,
    loss_out: blade_graphics::BufferPiece,
    params: SoftmaxParams,
}

// bce: var pred, labels, loss_out, params
#[derive(blade_macros::ShaderData)]
struct BceData {
    pred: blade_graphics::BufferPiece,
    labels: blade_graphics::BufferPiece,
    loss_out: blade_graphics::BufferPiece,
    params: UnaryParams, // len, _pad x3
}

// transpose: var src, dst, params
#[derive(blade_macros::ShaderData)]
struct TransposeData {
    src: blade_graphics::BufferPiece,
    dst: blade_graphics::BufferPiece,
    params: TransposeParams,
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct TransposeParams {
    m: u32,
    n: u32,
    _pad0: u32,
    _pad1: u32,
}

// multi_head_attn: var src_a (Q), src_b (K), bias (V), dst, lse, params
#[derive(blade_macros::ShaderData)]
struct MultiHeadAttnData {
    src_a: blade_graphics::BufferPiece,
    src_b: blade_graphics::BufferPiece,
    bias: blade_graphics::BufferPiece,
    dst: blade_graphics::BufferPiece,
    lse: blade_graphics::BufferPiece,
    params: AttentionParams,
}

// multi_head_attn_grad: var d_out (dO), src_a (Q), src_b (K), bias (V), lse, fwd_dst (O), dst (dQ/dK/dV), params
#[derive(blade_macros::ShaderData, Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct AttentionParams {
    q_seq: u32,
    kv_seq: u32,
    packed_heads: u32,
    head_dim: u32,
    window_size: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

#[derive(blade_macros::ShaderData)]
struct MultiHeadAttnGradData {
    d_out: blade_graphics::BufferPiece,
    src_a: blade_graphics::BufferPiece,
    src_b: blade_graphics::BufferPiece,
    bias: blade_graphics::BufferPiece,
    lse: blade_graphics::BufferPiece,
    fwd_dst: blade_graphics::BufferPiece,
    dst: blade_graphics::BufferPiece,
    params: AttentionParams,
}

// Fused GradK+GradV: outputs dK (dst) and dV (dst2)
#[derive(blade_macros::ShaderData)]
struct MultiHeadAttnGradKVData {
    d_out: blade_graphics::BufferPiece,
    src_a: blade_graphics::BufferPiece,
    src_b: blade_graphics::BufferPiece,
    bias: blade_graphics::BufferPiece,
    lse: blade_graphics::BufferPiece,
    fwd_dst: blade_graphics::BufferPiece,
    dst: blade_graphics::BufferPiece,
    dst2: blade_graphics::BufferPiece,
    params: AttentionParams,
}

// ---- Pipeline collection ----

/// A packed B buffer and a 32×32 tile each change the generated WGSL, so
/// both travel in the key: sharing one pipeline across them would run an
/// f32 shader over packed blocks, or a 64×64 shader under a workgroup
/// count computed for 32×32 tiles.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct EpiloguePipelineKey(
    crate::compile::MatMulEpilogue,
    crate::compile::WeightFormat,
    Option<crate::tune::MatmulTile>,
);

fn epilogue_pipeline_key(dispatch: &Dispatch) -> Option<EpiloguePipelineKey> {
    let format = dispatch.weight_format;
    dispatch.matmul_epilogue.as_ref().map(|epilogue| {
        EpiloguePipelineKey(
            epilogue.clone(),
            format,
            crate::tune::MatmulTile::selected(dispatch, None),
        )
    })
}

/// Tile geometry the epilogue shader must be generated for.
///
/// `select_variants` demotes low-occupancy matmuls to 32×32 tiles and
/// recomputes `workgroups` to match. The epilogue path has to follow, or
/// the dispatch runs a 64×64 shader over a 32×32 grid — the store bounds
/// check keeps the result correct, but three quarters of the workgroups
/// do nothing.
fn epilogue_tile(dispatch: &Dispatch) -> crate::codegen::MatMulTile {
    if let Some(shape) = dispatch.scalar_matmul() {
        shape.geometry()
    } else if dispatch.use_small_tiles() {
        crate::codegen::MatMulTile::Small
    } else {
        crate::codegen::MatMulTile::Large
    }
}

/// Identifies one compiled pipeline.
///
/// A dispatch names a `ShaderEntry`, but the pipeline it actually runs
/// also depends on the modifiers the plan attached to it - weight format,
/// tiling, a fused epilogue, and so on. [`Pipelines::key`] resolves that
/// implementation once; preparation builds exactly the selected pipeline.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
enum Variant {
    CoopTiled(
        ShaderEntry,
        crate::codegen::CooperativeMatmulShape,
        u32,
        Vec<crate::compile::PrologueLoadKind>,
    ),
    SplitMatmul(
        ShaderEntry,
        crate::compile::WeightFormat,
        crate::codegen::ScalarMatmulShape,
        u32,
    ),
    SpecializedConv(ShaderEntry, Vec<u32>, u32),
    ScalarMatmul(
        ShaderEntry,
        crate::compile::WeightFormat,
        crate::codegen::ScalarMatmulShape,
    ),
    /// Schedule-template kernels, keyed by kernel content hash. These are
    /// generated from a DAG rather than a shader group, so no `ShaderEntry`
    /// identifies them.
    Reduction(u64),
    Pointwise(u64),
    /// Attention specialization includes the head width and the scalar
    /// backward EPT cap. Both can differ between dispatches in one plan.
    Attention(ShaderEntry, u32, Option<u32>),
    /// Epilogue-fused matmuls, keyed by their actual DAG. The cooperative form
    /// uses workgroup memory to expose accumulator lanes to the epilogue.
    Epilogue(ShaderEntry, EpiloguePipelineKey),
    CoopEpilogue(ShaderEntry, EpiloguePipelineKey),
    /// Prologue-fused coop matmuls. Prologues only apply when `use_coop`
    /// is set. Weight storage and factor kinds determine the shader; buffer
    /// IDs are resolved at dispatch time.
    CoopPrologue(
        ShaderEntry,
        crate::compile::WeightFormat,
        Vec<crate::compile::PrologueLoadKind>,
    ),
    /// GEMV with a RmsNorm folded into its A operand, at a weight format
    /// and measured shape. Packing and width are part of the key so a Q4_0
    /// winner cannot be installed on an f32 fused GEMV of the same extents.
    GemvRmsNorm(
        ShaderEntry,
        crate::compile::WeightFormat,
        crate::codegen::GemvShape,
    ),
    /// The Q8_1-activation, integer-dot K-split GEMV at a measured shape,
    /// with the weight format selecting between the GGML Q4_0 and Meganeura
    /// Q8 kernels.
    GemvIntDot(
        ShaderEntry,
        crate::compile::WeightFormat,
        crate::codegen::GemvShape,
    ),
    /// Same, with the source RmsNorm folded into the prologue.
    GemvRmsNormIntDot(
        ShaderEntry,
        crate::compile::WeightFormat,
        crate::codegen::GemvShape,
    ),
    /// A K-split GEMV at a measured workgroup width and reduction, for the
    /// weight format it reads. Every other axis of the GEMV family — fused
    /// add, transposed B, f16, block-packed — is already in the entry and
    /// the format, so the shape is the only thing this adds.
    Gemv(
        ShaderEntry,
        crate::compile::WeightFormat,
        crate::codegen::GemvShape,
    ),
    /// Non-f32 weight storage (f16, Q4, Q8).
    Weight(ShaderEntry, crate::compile::WeightFormat),
    WeightSmall(ShaderEntry, crate::compile::WeightFormat),
    /// Cooperative-matrix implementation qualified for the session's precision
    /// policy, for the weight format it reads (f32, or f16 with f16 tiles).
    Coop(ShaderEntry, crate::compile::WeightFormat),
    /// Cooperative f16 with hi/lo residual staging (C1).
    CoopCompensated(ShaderEntry),
    /// Same-A matmul pack (D1). The kind is part of the key so a
    /// forward Q/K/V pack (plain f16 coop) cannot share a pipeline with
    /// a backward pack of the same arity (compensated).
    Horizontal(ShaderEntry, u32, HorizMatMulKind),
    SmallTile(ShaderEntry),
    /// The unmodified pipeline.
    Scalar(ShaderEntry),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum HorizMatMulKind {
    Scalar,
    Coop,
    CoopCompensated,
}

fn horiz_kind(dispatch: &Dispatch) -> HorizMatMulKind {
    if dispatch.use_coop_compensated() {
        HorizMatMulKind::CoopCompensated
    } else if dispatch.use_coop() {
        HorizMatMulKind::Coop
    } else {
        HorizMatMulKind::Scalar
    }
}

impl Variant {
    /// The shader entry this pipeline was generated from. `None` for the
    /// schedule-template kernels, which have no entry of their own.
    fn entry(&self) -> Option<&ShaderEntry> {
        match *self {
            Variant::Reduction(_) | Variant::Pointwise(_) => None,
            Variant::Attention(ref e, _, _)
            | Variant::CoopTiled(ref e, _, _, _)
            | Variant::SplitMatmul(ref e, _, _, _)
            | Variant::SpecializedConv(ref e, _, _)
            | Variant::ScalarMatmul(ref e, _, _)
            | Variant::Epilogue(ref e, _)
            | Variant::CoopEpilogue(ref e, _)
            | Variant::CoopPrologue(ref e, _, _)
            | Variant::GemvRmsNorm(ref e, _, _)
            | Variant::Gemv(ref e, _, _)
            | Variant::GemvIntDot(ref e, _, _)
            | Variant::GemvRmsNormIntDot(ref e, _, _)
            | Variant::Weight(ref e, _)
            | Variant::WeightSmall(ref e, _)
            | Variant::Coop(ref e, _)
            | Variant::CoopCompensated(ref e)
            | Variant::Horizontal(ref e, _, _)
            | Variant::SmallTile(ref e)
            | Variant::Scalar(ref e) => Some(e),
        }
    }

    /// Name used by the profiler and by pipeline-statistics dumps.
    fn label(&self) -> String {
        match *self {
            Variant::CoopTiled(ref e, shape, splits, ref prologue) => {
                format!("{e:?}:native-f32-tiled-{shape:?}-split-k-{splits}:{prologue:?}")
            }
            Variant::SplitMatmul(ref e, format, shape, splits) => {
                format!("{e:?}:split-k-{format:?}-{splits}-{shape:?}")
            }
            Variant::ScalarMatmul(ref e, format, shape) => {
                format!("{e:?}:scalar-{format:?}-{shape:?}")
            }
            Variant::SpecializedConv(ref e, ref params, k_tile) => {
                format!("{e:?}:fixed-native-div-k{k_tile}-{params:?}")
            }
            Variant::Reduction(hash) => format!("generated-reduction:{hash:016x}"),
            Variant::Pointwise(hash) => format!("generated-pointwise:{hash:016x}"),
            Variant::Attention(ref e, head_dim, ept) => {
                format!("{e:?}:head-dim-{head_dim}:ept-{ept:?}")
            }
            Variant::Epilogue(ref e, ref key) => epilogue_profile_key(e, key, false),
            Variant::CoopEpilogue(ref e, ref key) => epilogue_profile_key(e, key, true),
            Variant::CoopPrologue(ref e, crate::compile::WeightFormat::F32, ref kinds) => {
                format!("{e:?}:cooperative-prologue:{kinds:?}")
            }
            Variant::CoopPrologue(ref e, format, ref kinds) => {
                format!("{e:?}:cooperative-prologue-{format:?}-weights:{kinds:?}")
            }
            Variant::GemvRmsNorm(ref e, format, shape) => {
                format!("{e:?}:rmsnorm-{format:?}-{shape:?}")
            }
            Variant::GemvIntDot(ref e, format, shape) => {
                format!("{e:?}:gemv-intdot-{format:?}-{shape:?}")
            }
            Variant::GemvRmsNormIntDot(ref e, format, shape) => {
                format!("{e:?}:gemv-rmsnorm-intdot-{format:?}-{shape:?}")
            }
            Variant::Gemv(ref e, format, shape) => format!("{e:?}:gemv-{format:?}-{shape:?}"),
            Variant::Weight(ref e, format) => format!("{e:?}:weight-{format:?}"),
            Variant::WeightSmall(ref e, format) => format!("{e:?}:weight-{format:?}-small-tile"),
            Variant::Coop(ref e, crate::compile::WeightFormat::F32) => format!("{e:?}:cooperative"),
            Variant::Coop(ref e, format) => format!("{e:?}:cooperative-{format:?}-weights"),
            Variant::CoopCompensated(ref e) => format!("{e:?}:cooperative-compensated"),
            Variant::Horizontal(ref e, n, kind) => format!("{e:?}:horizontal-{n}-{kind:?}"),
            Variant::SmallTile(ref e) => format!("{e:?}:small-tile"),
            Variant::Scalar(ref e) => format!("{e:?}:scalar"),
        }
    }
}

/// A compiled pipeline, shared by the sessions on its context that need the
/// same generated code. The last user destroys it.
struct SharedPipeline {
    gpu: Arc<Gpu>,
    raw: blade_graphics::ComputePipeline,
}

impl Drop for SharedPipeline {
    fn drop(&mut self) {
        self.gpu.destroy_compute_pipeline(&mut self.raw);
    }
}

/// Exactly what compilation consumes, so a shared pipeline is the one
/// compiling this module would create. A live pipeline holds its context,
/// so the context address cannot be reused while an entry can upgrade.
#[derive(PartialEq, Eq, Hash)]
struct SharedPipelineKey {
    context: usize,
    entry_point: &'static str,
    layout: String,
    source: String,
}

/// Live pipelines by generated code. Candidate sessions in a measured search
/// differ in a few dispatches, so most of their pipelines are already live in
/// the incumbent. Weak entries never extend a pipeline's lifetime.
#[derive(Default)]
struct SharedPipelines {
    live: HashMap<SharedPipelineKey, std::sync::Weak<SharedPipeline>>,
    prune_at: usize,
}

static SHARED_PIPELINES: std::sync::LazyLock<std::sync::Mutex<SharedPipelines>> =
    std::sync::LazyLock::new(Default::default);

impl SharedPipelines {
    fn get(key: &SharedPipelineKey) -> Option<Arc<SharedPipeline>> {
        let shared = SHARED_PIPELINES.lock().unwrap();
        shared.live.get(key).and_then(std::sync::Weak::upgrade)
    }

    fn insert(key: SharedPipelineKey, pipeline: &Arc<SharedPipeline>) {
        let mut shared = SHARED_PIPELINES.lock().unwrap();
        shared.live.insert(key, Arc::downgrade(pipeline));
        if shared.live.len() >= shared.prune_at {
            shared
                .live
                .retain(|_, pipeline| pipeline.strong_count() != 0);
            shared.prune_at = (2 * shared.live.len()).max(256);
        }
    }
}

struct Pipelines {
    map: HashMap<Variant, Arc<SharedPipeline>>,
    /// Resolved after compilation or tuning, never while recording a step.
    selected: Vec<Variant>,
    /// Codegen knobs the plan was compiled with.
    knobs: crate::compile::TuningKnobs,
    /// Where to write every WGSL the pipeline layer compiles — [`SessionOptions::wgsl_dump_dir`].
    dump_dir: Option<String>,
}

/// Turn a generated module into a Blade shader.
fn create_gen_shader(
    gpu: &Gpu,
    module: crate::codegen::ShaderModule,
) -> Result<blade_graphics::Shader, String> {
    gpu.try_create_shader(blade_graphics::ShaderDesc {
        source: &module.source,
        naga_module: Some(module.module),
    })
    .map_err(|error| error.to_string())
}

fn create_profiled_pipeline(
    gpu: &Gpu,
    name: String,
    layout: &blade_graphics::ShaderDataLayout,
    compute: blade_graphics::ShaderFunction<'_>,
) -> blade_graphics::ComputePipeline {
    let _span = tracing::info_span!(
        "pipeline",
        name = %name,
        trace_min_duration_us = 1_000u64,
    )
    .entered();
    gpu.create_compute_pipeline(blade_graphics::ComputePipelineDesc {
        name: &name,
        data_layouts: &[layout],
        compute,
    })
}

impl Pipelines {
    fn new(
        gpu: &Arc<Gpu>,
        plan: &ExecutionPlan,
        coop_config: Option<&crate::codegen::CoopConfig>,
        wgsl_dump_dir: Option<&str>,
    ) -> Self {
        let mut pipelines = Self {
            map: HashMap::new(),
            selected: Vec::new(),
            knobs: plan.knobs,
            dump_dir: wgsl_dump_dir.map(str::to_string),
        };
        let optimizer: Vec<Dispatch> = if plan.param_grad_pairs.is_empty() {
            Vec::new()
        } else {
            [
                ShaderEntry::SgdUpdate,
                ShaderEntry::AdamUpdate,
                ShaderEntry::GradClipNormSq,
                ShaderEntry::GradClipScale,
                ShaderEntry::AdaptiveGradClip,
                ShaderEntry::GradAccum,
            ]
            .into_iter()
            .map(|shader| Dispatch {
                shader,
                ..Default::default()
            })
            .collect()
        };
        // One job per distinct implementation, in plan order.
        let mut jobs = Vec::new();
        let mut keys = std::collections::HashSet::new();
        for (dispatch, coop) in plan
            .dispatches
            .iter()
            .map(|dispatch| (dispatch, coop_config))
            .chain(optimizer.iter().map(|dispatch| (dispatch, None)))
        {
            let key = Self::key(dispatch);
            if keys.insert(key.clone()) {
                jobs.push((key, dispatch, coop));
            }
        }
        let compiled = Self::compile_all(gpu, plan.knobs, wgsl_dump_dir, &jobs);
        for ((key, dispatch, _), pipeline) in jobs.into_iter().zip(compiled) {
            let pipeline = pipeline.unwrap_or_else(|error| {
                if optimizer.iter().any(|d| std::ptr::eq(d, dispatch)) {
                    panic!("optimizer shader was rejected: {error}")
                } else {
                    panic!("selected shader was rejected: {error}")
                }
            });
            pipelines.map.insert(key, pipeline);
        }
        pipelines.select(&plan.dispatches);
        pipelines
    }

    /// Compile on worker threads. A driver can spend tens of milliseconds on
    /// one pipeline and a plan has hundreds; Blade's context is thread-safe.
    #[allow(clippy::type_complexity)]
    fn compile_all(
        gpu: &Arc<Gpu>,
        knobs: crate::compile::TuningKnobs,
        dump_dir: Option<&str>,
        jobs: &[(Variant, &Dispatch, Option<&crate::codegen::CoopConfig>)],
    ) -> Vec<Result<Arc<SharedPipeline>, String>> {
        let compile = |&(ref key, dispatch, coop): &(Variant, &Dispatch, _)| {
            Self::compile(gpu, knobs, dump_dir, dispatch, key, coop)
        };
        let workers = std::thread::available_parallelism()
            .map_or(1, |n| n.get())
            .min(8)
            .min(jobs.len());
        if workers <= 1 {
            return jobs.iter().map(compile).collect();
        }
        let next = std::sync::atomic::AtomicUsize::new(0);
        let span = tracing::Span::current();
        let mut results: Vec<_> = jobs.iter().map(|_| None).collect();
        std::thread::scope(|scope| {
            let workers: Vec<_> = (0..workers)
                .map(|_| {
                    scope.spawn(|| {
                        let _span = span.enter();
                        let mut compiled = Vec::new();
                        loop {
                            let index = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                            let Some(job) = jobs.get(index) else {
                                break compiled;
                            };
                            compiled.push((index, compile(job)));
                        }
                    })
                })
                .collect();
            for worker in workers {
                let compiled = worker
                    .join()
                    .unwrap_or_else(|panic| std::panic::resume_unwind(panic));
                for (index, result) in compiled {
                    results[index] = Some(result);
                }
            }
        });
        results.into_iter().map(Option::unwrap).collect()
    }

    fn prepare(
        &mut self,
        gpu: &Arc<Gpu>,
        dispatch: &Dispatch,
        coop_config: Option<&crate::codegen::CoopConfig>,
    ) -> Result<(), String> {
        let key = Self::key(dispatch);
        if self.map.contains_key(&key) {
            return Ok(());
        }
        let pipeline = Self::compile(
            gpu,
            self.knobs,
            self.dump_dir.as_deref(),
            dispatch,
            &key,
            coop_config,
        )?;
        self.map.insert(key, pipeline);
        Ok(())
    }

    /// Generate `dispatch`'s implementation `key` and return its pipeline,
    /// shared with any live session on `gpu` that generated the same code.
    fn compile(
        gpu: &Arc<Gpu>,
        knobs: crate::compile::TuningKnobs,
        dump_dir: Option<&str>,
        dispatch: &Dispatch,
        key: &Variant,
        coop_config: Option<&crate::codegen::CoopConfig>,
    ) -> Result<Arc<SharedPipeline>, String> {
        use crate::codegen::ShaderGroup;
        let mut matmul_knobs = crate::codegen::MatmulKnobs {
            k_stage: knobs.matmul_k_stage,
            interleave_columns: knobs.matmul_interleave_columns,
            integer_dot: gpu.capabilities().shader_integer_dot_product,
            unroll_k: false,
        };
        if let Some(shape) = dispatch.scalar_matmul() {
            matmul_knobs.k_stage = shape.k_stage;
            matmul_knobs.interleave_columns = shape.interleave_columns;
            matmul_knobs.unroll_k = shape.unroll_k;
        }
        let group = dispatch.shader.shader_group();
        let mut layout = shader_data_layout(&dispatch.shader);
        let mut entry_point = dispatch.shader.entry_point();
        let cooperative =
            || *coop_config.expect("cooperative dispatch needs a qualified device configuration");
        let module = match *key {
            Variant::CoopTiled(_, shape, splits, _) => {
                let config = cooperative();
                if config.tile_size != 16 || config.use_f16_input || config.compensated {
                    return Err("tiled cooperative matmul requires native 16x16 f32".to_owned());
                }
                if !dispatch
                    .mnk()
                    .is_some_and(|(m, n, k)| shape.fits_dimensions(m, n, k))
                {
                    return Err(
                        "tiled cooperative matmul requires complete aligned tiles".to_owned()
                    );
                }
                let shared_limit = gpu.capabilities().max_compute_shared_memory_size;
                if shared_limit != 0
                    && tiled_coop_shared_bytes(dispatch, shape) > u64::from(shared_limit)
                {
                    return Err("tiled cooperative matmul exceeds device shared memory".to_owned());
                }
                if let Some(ref prologue) = dispatch.matmul_prologue {
                    layout = matmul_with_prologue_layout(prologue.factors.len());
                }
                crate::codegen::generate_tiled_coop_matmul(
                    group,
                    shape,
                    splits,
                    dispatch.matmul_prologue.as_ref(),
                )
            }
            Variant::Scalar(_) => crate::codegen::generate_module(group, matmul_knobs),
            Variant::SmallTile(_) => crate::codegen::generate_module_small(group, matmul_knobs),
            Variant::Weight(_, format) => {
                crate::codegen::generate_module_weighted(group, format, matmul_knobs)
            }
            Variant::WeightSmall(..) => {
                tuning::tile_module(dispatch, crate::tune::MatmulTile::Tile32, matmul_knobs)
            }
            Variant::ScalarMatmul(_, _, shape) => tuning::tile_module(
                dispatch,
                crate::tune::MatmulTile::Scalar(shape),
                matmul_knobs,
            ),
            Variant::SplitMatmul(_, _, shape, splits) => crate::codegen::generate_split_matmul(
                group,
                shape,
                splits,
                dispatch.weight_format,
                matmul_knobs,
            ),
            Variant::SpecializedConv(..) => {
                let tile = crate::tune::MatmulTile::selected(dispatch, None)
                    .expect("specialized convolution");
                tuning::tile_module(dispatch, tile, matmul_knobs)
            }
            Variant::Gemv(_, _, shape)
            | Variant::GemvIntDot(_, _, shape)
            | Variant::GemvRmsNorm(_, _, shape)
            | Variant::GemvRmsNormIntDot(_, _, shape) => {
                if dispatch.gemv_rmsnorm.is_some() {
                    layout = <MatMulRmsNormData as blade_graphics::ShaderData>::layout();
                }
                tuning::tile_module(dispatch, crate::tune::MatmulTile::Gemv(shape), matmul_knobs)
            }
            Variant::Coop(..) | Variant::CoopCompensated(_) => {
                let mut config = cooperative();
                config.compensated = dispatch.use_coop_compensated();
                use crate::codegen::Conv2dCoopDirection;
                match dispatch.shader {
                    ShaderEntry::Conv2dGemmCoopGen(kh, kw, stride) => {
                        crate::codegen::generate_conv2d_coop_module(
                            kh,
                            kw,
                            stride,
                            Conv2dCoopDirection::Forward,
                            &config,
                        )
                    }
                    ShaderEntry::Conv2dGradInputGemmCoopGen(kh, kw, stride) => {
                        crate::codegen::generate_conv2d_coop_module(
                            kh,
                            kw,
                            stride,
                            Conv2dCoopDirection::GradInput,
                            &config,
                        )
                    }
                    _ => crate::codegen::generate_module_coop_weighted(
                        group,
                        &config,
                        dispatch.weight_format,
                    ),
                }
            }
            Variant::Epilogue(..) => crate::codegen::generate_matmul_with_epilogue(
                group,
                dispatch.matmul_epilogue.as_ref(),
                crate::codegen::MatMulOptions {
                    format: dispatch.weight_format,
                    tile: epilogue_tile(dispatch),
                    knobs: matmul_knobs,
                },
            ),
            Variant::CoopEpilogue(..) => crate::codegen::generate_coop_matmul_with_dag_epilogue(
                group,
                &cooperative(),
                dispatch
                    .matmul_epilogue
                    .as_ref()
                    .expect("cooperative epilogue"),
                dispatch.weight_format,
            ),
            Variant::CoopPrologue(..) => {
                let prologue = dispatch
                    .matmul_prologue
                    .as_ref()
                    .expect("cooperative prologue");
                let (add, variant) =
                    crate::codegen::coop_shape(group).expect("cooperative matrix group");
                layout = matmul_with_prologue_layout(prologue.factors.len());
                crate::codegen::gen_matmul_coop_with_prologue(
                    add,
                    variant,
                    &cooperative(),
                    prologue,
                    dispatch.weight_format,
                )
            }
            Variant::Attention(_, hd, ept) => match group {
                ShaderGroup::FlashAttention => crate::codegen::generate_flash_attention_module(
                    hd,
                    knobs.flash_ept_cap,
                    knobs.flash,
                ),
                ShaderGroup::FlashAttentionCoop => {
                    crate::codegen::generate_flash_attention_coop_module(hd)
                }
                ShaderGroup::FlashAttentionCoopF32 => {
                    let config = cooperative();
                    if config.tile_size != 16 || config.use_f16_input || config.compensated {
                        return Err(
                            "native16 attention requires qualified 16x16 f32 matrices".into()
                        );
                    }
                    crate::codegen::generate_flash_attention_coop_f32_module(hd)
                }
                ShaderGroup::AttentionGrad(kernel) => {
                    let tile = if kernel.path
                        == crate::kernels::attention_grad::Path::Cooperative(
                            crate::kernels::attention_grad::Operands::F32,
                        ) {
                        let config = cooperative();
                        if config.use_f16_input || config.compensated {
                            return Err(
                                "native-f32 attention gradients require qualified f32 matrices"
                                    .into(),
                            );
                        }
                        config.tile_size
                    } else {
                        8 // Unused by the other paths.
                    };
                    kernel.generate_with_f32_tile(hd, ept, &knobs, tile)
                }
                ShaderGroup::MultiHeadAttn => crate::codegen::generate_attention_module(hd),
                ShaderGroup::CachedQueryAttention => {
                    crate::codegen::generate_cached_query_attention_module(hd)
                }
                ShaderGroup::CachedBlockAttention
                | ShaderGroup::CachedBlockAttentionSplit
                | ShaderGroup::CachedBlockAttentionCombine => {
                    crate::codegen::generate_cached_attention_module(group, Some(hd))
                }
                _ => unreachable!("non-parameterized attention group {group:?}"),
            },
            Variant::Pointwise(_) => {
                let dag = dispatch.pointwise().expect("pointwise implementation");
                layout = pointwise_data_layout(dag.n_inputs);
                entry_point = crate::schedule::POINTWISE_ENTRY;
                crate::schedule::lower(&crate::schedule::KernelTemplate::Pointwise {
                    dag: dag.clone(),
                    grid: crate::schedule::GridShape::default(),
                })
            }
            Variant::Reduction(_) => {
                let kernel = dispatch.reduction().expect("reduction implementation");
                layout = reduction_data_layout(kernel);
                entry_point = crate::schedule::REDUCTION_ENTRY;
                crate::schedule::lower(&kernel.to_template())
            }
            Variant::Horizontal(_, count, kind) => {
                let coop = match kind {
                    HorizMatMulKind::Scalar => None,
                    HorizMatMulKind::Coop | HorizMatMulKind::CoopCompensated => {
                        let mut config = cooperative();
                        config.compensated = kind == HorizMatMulKind::CoopCompensated;
                        Some(config)
                    }
                };
                layout = horizontal_matmul_layout(count);
                entry_point = "main";
                crate::codegen::generate_horizontal_matmul(group, count, coop.as_ref())
            }
        };
        // Every module a session compiles — the standard, coop, weighted,
        // epilogue-fused and scheduled forms alike — passes through here, so
        // the dump sees exactly what a plan would run, shared or not.
        if let Some(dir) = dump_dir {
            module.dump(dir);
        }
        let shared_key = SharedPipelineKey {
            context: Arc::as_ptr(gpu) as usize,
            entry_point,
            layout: format!("{:?}", layout.bindings),
            source: module.source.clone(),
        };
        if let Some(pipeline) = SharedPipelines::get(&shared_key) {
            return Ok(pipeline);
        }
        let shader = create_gen_shader(gpu, module)?;
        let pipeline = Arc::new(SharedPipeline {
            gpu: Arc::clone(gpu),
            raw: create_profiled_pipeline(gpu, key.label(), &layout, shader.at(entry_point)),
        });
        SharedPipelines::insert(shared_key, &pipeline);
        Ok(pipeline)
    }

    fn attention_head_dim(dispatch: &Dispatch) -> Option<u32> {
        use crate::codegen::ShaderGroup;
        match dispatch.shader.shader_group() {
            ShaderGroup::CachedBlockAttention
            | ShaderGroup::CachedBlockAttentionSplit
            | ShaderGroup::CachedBlockAttentionCombine => {
                CachedBlockAttentionParams::from_words(&dispatch.params).map(|p| p.head_dim)
            }
            ShaderGroup::MultiHeadAttn
            | ShaderGroup::FlashAttention
            | ShaderGroup::FlashAttentionCoop
            | ShaderGroup::FlashAttentionCoopF32
            | ShaderGroup::CachedQueryAttention => dispatch.params.get(3).copied(),
            ShaderGroup::AttentionGrad(kernel) if kernel.specializes_head_dim() => {
                dispatch.params.get(3).copied()
            }
            _ => None,
        }
    }

    /// Resolve one implementation. Geometry and arithmetic must not depend on
    /// which unrelated pipelines happen to have been compiled.
    fn key(dispatch: &Dispatch) -> Variant {
        let entry = dispatch.shader.clone();
        if let crate::compile::Kernel::CooperativeTiled { shape, splits } = dispatch.kernel {
            return Variant::CoopTiled(
                entry,
                shape,
                splits,
                dispatch
                    .matmul_prologue
                    .as_ref()
                    .map(|p| p.factors.iter().map(|f| f.1.clone()).collect())
                    .unwrap_or_default(),
            );
        }
        if let crate::compile::Kernel::SplitMatmul { shape, splits } = dispatch.kernel {
            return Variant::SplitMatmul(entry, dispatch.weight_format, shape, splits);
        }
        if let Some(k_tile) = dispatch.conv_k_tile() {
            return Variant::SpecializedConv(entry, dispatch.params.clone(), k_tile);
        }
        if dispatch.horizontal_batch >= 2 {
            return Variant::Horizontal(entry, dispatch.horizontal_batch, horiz_kind(dispatch));
        }
        if let Some(epilogue) = epilogue_pipeline_key(dispatch) {
            return if dispatch.use_coop() {
                Variant::CoopEpilogue(entry, epilogue)
            } else {
                Variant::Epilogue(entry, epilogue)
            };
        }
        if let Some(shape) = dispatch.scalar_matmul() {
            return Variant::ScalarMatmul(entry, dispatch.weight_format, shape);
        }
        if dispatch.gemv_int_dot() || dispatch.gemv_rmsnorm.is_some() {
            let shape = dispatch
                .gemv_shape()
                .unwrap_or_else(|| crate::codegen::GemvShape::initial(entry.shader_group()));
            return if dispatch.gemv_int_dot() {
                if dispatch.gemv_rmsnorm.is_some() {
                    Variant::GemvRmsNormIntDot(entry, dispatch.weight_format, shape)
                } else {
                    Variant::GemvIntDot(entry, dispatch.weight_format, shape)
                }
            } else {
                Variant::GemvRmsNorm(entry, dispatch.weight_format, shape)
            };
        }
        if let Some(kernel) = dispatch.reduction() {
            return Variant::Reduction(kernel.hash_key());
        }
        if let Some(dag) = dispatch.pointwise() {
            return Variant::Pointwise(dag.hash_key());
        }
        if let Some(dim) = Self::attention_head_dim(dispatch) {
            let ept = match dispatch.kernel {
                crate::compile::Kernel::AttentionBackward { ept_cap } => Some(ept_cap),
                _ => None,
            };
            return Variant::Attention(entry, dim, ept);
        }
        if let Some(shape) = dispatch.gemv_shape() {
            return Variant::Gemv(entry, dispatch.weight_format, shape);
        }
        let cooperative_f16_weights = dispatch.use_coop()
            && !dispatch.use_coop_compensated()
            && dispatch.weight_format == crate::compile::WeightFormat::F16;
        if dispatch.weight_format.uses_reduced_storage() && !cooperative_f16_weights {
            return if dispatch.use_small_tiles() {
                Variant::WeightSmall(entry, dispatch.weight_format)
            } else {
                Variant::Weight(entry, dispatch.weight_format)
            };
        }
        if dispatch.use_coop() {
            if let Some(ref prologue) = dispatch.matmul_prologue {
                return Variant::CoopPrologue(
                    entry,
                    dispatch.weight_format,
                    prologue.factors.iter().map(|f| f.1.clone()).collect(),
                );
            }
            return if dispatch.use_coop_compensated() {
                Variant::CoopCompensated(entry)
            } else {
                Variant::Coop(entry, dispatch.weight_format)
            };
        }
        if dispatch.use_small_tiles() {
            return Variant::SmallTile(entry);
        }
        Variant::Scalar(entry)
    }

    /// The unmodified pipeline for an entry, for the fixed passes -
    /// optimizer step, grad clip, grad accumulate - that are issued
    /// directly rather than through a `Dispatch`, and so never carry a
    /// modifier.
    fn scalar(&self, entry: ShaderEntry) -> &blade_graphics::ComputePipeline {
        &self.map[&Variant::Scalar(entry)].raw
    }

    fn select(&mut self, dispatches: &[Dispatch]) {
        self.selected = dispatches
            .iter()
            .map(|dispatch| {
                let key = Self::key(dispatch);
                assert!(
                    self.map.contains_key(&key),
                    "no pipeline was compiled for {dispatch:?}"
                );
                key
            })
            .collect();
    }

    fn get(&self, dispatch_index: usize) -> &blade_graphics::ComputePipeline {
        &self.map[&self.selected[dispatch_index]].raw
    }

    fn all_pipelines(&self) -> Vec<(&str, &blade_graphics::ComputePipeline)> {
        self.map
            .iter()
            .filter_map(|(variant, pipeline)| {
                variant.entry().map(|e| (e.entry_point(), &pipeline.raw))
            })
            .collect()
    }

    fn all_profile_pipelines(&self) -> Vec<(String, &blade_graphics::ComputePipeline)> {
        self.map
            .iter()
            .map(|(variant, pipeline)| (variant.label(), &pipeline.raw))
            .collect()
    }
}

fn epilogue_profile_key(
    entry: &ShaderEntry,
    epilogue: &EpiloguePipelineKey,
    cooperative: bool,
) -> String {
    use std::hash::{Hash, Hasher};

    let mut hasher = std::hash::DefaultHasher::new();
    epilogue.hash(&mut hasher);
    let variant = if cooperative {
        "cooperative-epilogue"
    } else {
        "scalar-epilogue"
    };
    format!("{entry:?}:{variant}:{:016x}", hasher.finish())
}

fn horizontal_matmul_layout(count: u32) -> blade_graphics::ShaderDataLayout {
    let mut bindings: Vec<(&'static str, blade_graphics::ShaderBinding)> =
        vec![("matrix_a", blade_graphics::ShaderBinding::Buffer)];
    for i in 0..count {
        bindings.push((
            match i {
                0 => "matrix_b0",
                1 => "matrix_b1",
                _ => "matrix_b2",
            },
            blade_graphics::ShaderBinding::Buffer,
        ));
    }
    for i in 0..count {
        bindings.push((
            match i {
                0 => "matrix_c0",
                1 => "matrix_c1",
                _ => "matrix_c2",
            },
            blade_graphics::ShaderBinding::Buffer,
        ));
    }
    bindings.push((
        "params",
        blade_graphics::ShaderBinding::Plain {
            size: std::mem::size_of::<MatMulParams>() as u32,
        },
    ));
    blade_graphics::ShaderDataLayout { bindings }
}

/// ShaderDataLayout for a schedule-template reduction pipeline, chosen
/// by (n_per_elem, n_per_row, n_per_col). Names align with the bindings
/// emitted by `schedule::lower` for reductions.
/// Ordered storage-buffer binding names for a reduction kernel's input
/// streams: each per-element stream, with its `_idx` indices buffer right
/// after it when that stream is a gather; then per-row; then per-col.
/// Excludes `dst` / `params`. Names are `'static` (bounded vocabulary) and
/// match the globals emitted by `schedule::lower_reduction`, so they slot
/// directly into a dynamically-built `ShaderDataLayout` and the matching
/// `DynReductionData::fill` order.
fn reduction_input_binding_names(k: &crate::schedule::ReductionKernel) -> Vec<&'static str> {
    let per_elem: &[&str] = match k.n_per_elem {
        1 => &["src"],
        2 => &["src_a", "src_b"],
        3 => &["src_a", "src_b", "src_c"],
        n => panic!("reduction n_per_elem {n} unsupported"),
    };
    let idx: &[&str] = match k.n_per_elem {
        1 => &["src_idx"],
        2 => &["src_a_idx", "src_b_idx"],
        3 => &["src_a_idx", "src_b_idx", "src_c_idx"],
        _ => &[],
    };
    let per_row: &[&str] = match k.n_per_row {
        0 => &[],
        1 => &["per_row_src"],
        2 => &["per_row_src_a", "per_row_src_b"],
        n => panic!("reduction n_per_row {n} unsupported"),
    };
    let n_per_col = k.epilogue.as_ref().map_or(0, |e| e.n_per_col_inputs);
    let per_col: &[&str] = match n_per_col {
        0 => &[],
        1 => &["bias"],
        2 => &["bias_a", "bias_b"],
        n => panic!("reduction n_per_col {n} unsupported"),
    };
    let mut names = Vec::new();
    for i in 0..k.n_per_elem as usize {
        names.push(per_elem[i]);
        if k.gather_elem.get(i).copied().unwrap_or(false) {
            names.push(idx[i]);
        }
    }
    names.extend_from_slice(per_row);
    names.extend_from_slice(per_col);
    names
}

/// Whether a reduction kernel needs the dynamic binding path (gather
/// streams, or an arity not covered by the three fixed layout structs).
fn reduction_is_dynamic(k: &crate::schedule::ReductionKernel) -> bool {
    let n_per_col = k.epilogue.as_ref().map_or(0, |e| e.n_per_col_inputs);
    let known = matches!(
        (k.n_per_elem, k.n_per_row, n_per_col),
        (1, 0, 0) | (1, 1, 0) | (1, 0, 1)
    );
    !k.gather_elem.iter().all(|&g| !g) || !known
}

/// Dynamically-built reduction bind data: storage buffers (per the
/// `reduction_input_binding_names` order, plus `dst`) followed by the
/// Bindings for a GEMV with its input RmsNorm folded in.
#[derive(blade_macros::ShaderData)]
struct MatMulRmsNormData {
    matrix_a: blade_graphics::BufferPiece,
    norm_w: blade_graphics::BufferPiece,
    matrix_b: blade_graphics::BufferPiece,
    matrix_c: blade_graphics::BufferPiece,
    params: MatMulRmsNormParams,
}

#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct MatMulRmsNormParams {
    m: u32,
    n: u32,
    k: u32,
    eps_bits: u32,
}

/// A run of storage buffers followed by one params uniform.
///
/// Binds a variable number of buffers because the shaders these serve take
/// their operand count as a parameter: a horizontal matmul binds one input
/// per matrix in the pack, and a dynamic reduction binds one per gathered
/// stream. Neither has a fixed arity, so neither can use a `ShaderData`
/// struct with named fields — `bind<D>` ignores `D::layout()` and uses the
/// pipeline's own, which is what makes a single generic shape work for both.
///
/// `layout()` is therefore a placeholder that is never consulted; the
/// pipelines these bind against carry the real layout.
struct BufferListData<P> {
    buffers: Vec<blade_graphics::BufferPiece>,
    params: P,
}

impl<P: blade_graphics::ShaderBindable + Copy> blade_graphics::ShaderData for BufferListData<P> {
    fn layout() -> blade_graphics::ShaderDataLayout {
        blade_graphics::ShaderDataLayout::default()
    }
    fn fill(&self, mut context: blade_graphics::PipelineContext) {
        use blade_graphics::ShaderBindable;
        let mut index = 0u32;
        for b in &self.buffers {
            b.bind_to(&mut context, index);
            index += 1;
        }
        self.params.bind_to(&mut context, index);
    }
}

/// Horizontal matmul bindings: every matrix in the pack, then `params`.
type HorizMatMulData = BufferListData<MatMulParams>;

/// Gather-reduction bindings: one buffer per gathered stream, then `params`.
type DynReductionData = BufferListData<ReductionParams>;

fn reduction_data_layout(
    kernel: &crate::schedule::ReductionKernel,
) -> blade_graphics::ShaderDataLayout {
    use blade_graphics::ShaderData;
    if reduction_is_dynamic(kernel) {
        let mut bindings: Vec<(&'static str, blade_graphics::ShaderBinding)> =
            reduction_input_binding_names(kernel)
                .into_iter()
                .map(|n| (n, blade_graphics::ShaderBinding::Buffer))
                .collect();
        bindings.push(("dst", blade_graphics::ShaderBinding::Buffer));
        bindings.push((
            "params",
            blade_graphics::ShaderBinding::Plain {
                size: std::mem::size_of::<ReductionParams>() as u32,
            },
        ));
        return blade_graphics::ShaderDataLayout { bindings };
    }
    let n_per_col = kernel.epilogue.as_ref().map_or(0, |e| e.n_per_col_inputs);
    match (kernel.n_per_elem, kernel.n_per_row, n_per_col) {
        (1, 0, 0) => ReductionPass1Data::layout(),
        (1, 1, 0) => ReductionPass2RowData::layout(),
        (1, 0, 1) => RmsNormData::layout(),
        other => panic!(
            "reduction kernel with (per_elem, per_row, per_col) = {:?} has no runtime layout",
            other
        ),
    }
}

/// ShaderDataLayout for a schedule-template pointwise pipeline, chosen by
/// DAG input arity. Names align with the bindings emitted by
/// `schedule::lower` for each arity.
fn pointwise_data_layout(n_inputs: u8) -> blade_graphics::ShaderDataLayout {
    use blade_graphics::ShaderData;
    match n_inputs {
        1 => UnaryData::layout(),
        2 => BinaryData::layout(),
        3 => TernaryData::layout(),
        n => panic!("pointwise arity {} has no runtime data layout", n),
    }
}

/// Get the ShaderDataLayout for a coop matmul with an N-factor prologue.
pub fn matmul_with_prologue_layout(n_factors: usize) -> blade_graphics::ShaderDataLayout {
    use blade_graphics::ShaderData;
    match n_factors {
        2 => MatMulPrologue2Data::layout(),
        n => panic!("matmul prologue arity {} has no runtime data layout", n),
    }
}

/// Get the ShaderDataLayout for a given shader entry.
pub fn shader_data_layout(entry: &ShaderEntry) -> blade_graphics::ShaderDataLayout {
    use blade_graphics::ShaderData;
    match *entry {
        // Generated kernels take their layout from their kernel.
        ShaderEntry::Generated => blade_graphics::ShaderDataLayout {
            bindings: Vec::new(),
        },
        ShaderEntry::MatMul
        | ShaderEntry::MatMulAT
        | ShaderEntry::MatMulBT
        | ShaderEntry::MatMulGemv
        | ShaderEntry::MatMulGemvBT => MatMulData::layout(),
        ShaderEntry::BlockMatMul
        | ShaderEntry::BlockMatMulAT
        | ShaderEntry::BlockMatMulBT
        | ShaderEntry::BatchMatMul
        | ShaderEntry::BatchMatMulAT
        | ShaderEntry::BatchMatMulBT => MatMulData::layout(),
        ShaderEntry::FusedMatMulAdd
        | ShaderEntry::FusedMatMulATAdd
        | ShaderEntry::FusedMatMulBTAdd
        | ShaderEntry::MatMulGemvBTAdd
        | ShaderEntry::MatMulGemvAdd => FusedMatMulAddData::layout(),
        ShaderEntry::PairwiseGrad => TernaryData::layout(),
        ShaderEntry::SgdUpdate => optimizer::SgdData::layout(),
        ShaderEntry::AdamUpdate => optimizer::AdamData::layout(),
        ShaderEntry::ScatterAdd => ScatterAddData::layout(),
        ShaderEntry::ScatterAddAtomic => ScatterAddAtomicData::layout(),
        ShaderEntry::SwiGLUConcat
        | ShaderEntry::SwiGLUConcatGrad
        | ShaderEntry::GeGLUConcat
        | ShaderEntry::GeGLUConcatGrad => BinaryData::layout(),
        ShaderEntry::SumAll | ShaderEntry::MeanAll | ShaderEntry::SumRows => UnaryData::layout(),
        ShaderEntry::CrossEntropyLoss | ShaderEntry::CrossEntropyLossIndices => {
            CrossEntropyData::layout()
        }
        ShaderEntry::BceLoss => BceData::layout(),
        ShaderEntry::Transpose => TransposeData::layout(),
        ShaderEntry::Embedding => EmbeddingData::layout(),
        ShaderEntry::ToF16 => UnaryData::layout(),
        ShaderEntry::RoPE | ShaderEntry::RoPEGrad => RoPEData::layout(),
        ShaderEntry::LayerNorm => LayerNormData::layout(),
        ShaderEntry::RmsNormAdd => RmsNormAddData::layout(),
        ShaderEntry::MultiHeadAttn
        | ShaderEntry::FlashAttention
        | ShaderEntry::FlashAttentionCoop
        | ShaderEntry::FlashAttentionCoopF32 => MultiHeadAttnData::layout(),
        ShaderEntry::AttentionGrad(kernel) => match kernel.part {
            AttentionGradPart::Q => MultiHeadAttnGradData::layout(),
            AttentionGradPart::KV => MultiHeadAttnGradKVData::layout(),
        },
        ShaderEntry::SwiGLUGradGate => TernaryData::layout(),
        ShaderEntry::SwiGLUGradUp | ShaderEntry::SiluGrad => BinaryData::layout(),
        ShaderEntry::RmsNormGradW | ShaderEntry::RmsNormGradWRowPar | ShaderEntry::RmsNormGradX => {
            FourBufData::layout()
        }
        ShaderEntry::LayerNormGradWB | ShaderEntry::LayerNormGradX => FourBufData::layout(),
        ShaderEntry::RmsNormRsqrt => UnaryData::layout(),
        ShaderEntry::GroupNorm | ShaderEntry::GroupNormSilu => GroupNormData::layout(),
        ShaderEntry::GroupNormApply => GroupNormApplyData::layout(),
        ShaderEntry::GroupNormStats => GroupNormStatsData::layout(),
        ShaderEntry::GroupNormGradInput
        | ShaderEntry::GroupNormGradWeightBias
        | ShaderEntry::GroupNormGradStats => GroupNormGradData::layout(),
        ShaderEntry::Concat => BinaryData::layout(),
        ShaderEntry::SplitA | ShaderEntry::SplitB => UnaryData::layout(),
        ShaderEntry::Permute => PermuteData::layout(),
        ShaderEntry::BiasedAttention => BiasedAttentionData::layout(),
        ShaderEntry::Upsample2x | ShaderEntry::Upsample2xGrad => UnaryData::layout(),
        ShaderEntry::Conv2dDw => Conv2dDwData::layout(),
        ShaderEntry::MulPerChannel => MulPerChannelData::layout(),
        ShaderEntry::Conv2dGemm
        | ShaderEntry::Conv2dGemmSmall
        | ShaderEntry::Conv2dGemm16
        | ShaderEntry::Conv2dGemmCoopGen(..) => Conv2dData::layout(),
        ShaderEntry::Conv2dGradInputGemm
        | ShaderEntry::Conv2dGradInputGemmSmall
        | ShaderEntry::Conv2dGradInputGemm16
        | ShaderEntry::Conv2dGradInputGemmCoopGen(..) => Conv2dGradInputData::layout(),
        ShaderEntry::Conv2dGradWeightGemm
        | ShaderEntry::Conv2dGradWeightGemmSmall
        | ShaderEntry::Conv2dGradWeightGemm16
        | ShaderEntry::Conv2dGradWeightGemmSplit
        | ShaderEntry::Conv2dGradWeightGemmSplitSmall
        | ShaderEntry::Conv2dGradWeightGemmSplit16 => Conv2dGradWeightData::layout(),
        ShaderEntry::RoPEDynamic | ShaderEntry::RoPEPositions => RoPEDynamicData::layout(),
        ShaderEntry::RoPEDynamicFactors => RoPEDynamicFactorsData::layout(),
        ShaderEntry::CacheWrite => CacheWriteData::layout(),
        ShaderEntry::CacheWritePrefix => CacheWritePrefixData::layout(),
        ShaderEntry::CachedAttention | ShaderEntry::CachedQueryAttention => {
            CachedAttentionData::layout()
        }
        ShaderEntry::CachedBlockAttention | ShaderEntry::CachedBlockAttentionSplit => {
            CachedBlockAttentionData::layout()
        }
        ShaderEntry::CachedBlockAttentionCombine => CachedBlockAttentionCombineData::layout(),
        ShaderEntry::ChunkedRelativeAttention => ChunkedRelativeAttentionData::layout(),
        ShaderEntry::PrefixLast => PrefixLastData::layout(),
        ShaderEntry::MaxPool2d => MaxPool2dData::layout(),
        ShaderEntry::MaxPool2dGrad => MaxPool2dGradData::layout(),
        ShaderEntry::GlobalAvgPool => GlobalAvgPoolData::layout(),
        ShaderEntry::GlobalAvgPoolGrad => UnaryData::layout(),
        ShaderEntry::WinogradInputTransform
        | ShaderEntry::WinogradOutputTransform
        | ShaderEntry::WinogradWeightTransform => WinogradTransformData::layout(),
        ShaderEntry::WinogradBatchedMatMul => MatMulData::layout(),
        ShaderEntry::GradClipNormSq | ShaderEntry::GradClipScale | ShaderEntry::GradAccum => {
            optimizer::GradPassData::layout()
        }
        ShaderEntry::AdaptiveGradClip => optimizer::AdaptiveGradClipData::layout(),
    }
}

// ---- Dispatch recording ----
//
// Ordering the dispatches and partitioning them into barrier groups is the
// compiler's job (`compile::schedule_dispatches`): horizontal fusion changes
// how many dispatches a plan has, so the groups cannot be computed apart from
// it. Recording them is all that is left here.

/// Record the compiled graph, leaving the last chunk open for appended work.
fn record_groups(
    gpu: &Gpu,
    encoder: &mut blade_graphics::CommandEncoder,
    sync_point: &mut Option<blade_graphics::SyncPoint>,
    plan: &ExecutionPlan,
    groups: &[std::ops::Range<usize>],
    pipelines: &Pipelines,
    buffers: &[blade_graphics::BufferPiece],
    chunks: usize,
) {
    let total = groups.len();
    let per_chunk = total.div_ceil(chunks.min(total).max(1));
    let chunk_count = if total == 0 {
        0
    } else {
        total.div_ceil(per_chunk)
    };
    let mut start = 0;
    let mut chunk_index = 0;
    while start < total {
        let end = (start + per_chunk).min(total);
        {
            let label = format!("step {}/{}", chunk_index + 1, chunk_count);
            let mut pass = encoder.compute(&label);
            for (gi, group) in groups.iter().enumerate().take(end).skip(start) {
                if gi > start {
                    pass.barrier();
                }
                for i in group.clone() {
                    let dispatch = &plan.dispatches[i];
                    let pipeline = pipelines.get(i);
                    let mut pc = pass.with(pipeline);
                    Session::bind_dispatch(buffers, dispatch, &mut pc);
                    pc.dispatch(dispatch.workgroups);
                }
            }
        }
        start = end;
        chunk_index += 1;
        if start < total {
            *sync_point = Some(gpu.submit(encoder));
            encoder.start();
        }
    }
}

// ---- Session ----

fn tiled_coop_shared_bytes(
    dispatch: &Dispatch,
    shape: crate::codegen::CooperativeMatmulShape,
) -> u64 {
    let row_factors = dispatch.matmul_prologue.as_ref().map_or(0, |p| {
        p.factors
            .iter()
            .filter(|f| f.1 == crate::compile::PrologueLoadKind::PerRow)
            .count()
    });
    u64::from(shape.shared_bytes()) + row_factors as u64 * 64 * 4
}

/// A compiled, ready-to-execute GPU session.
///
/// Holds all blade-graphics resources: context, buffers, pipelines.
/// Calling `step()` replays the pre-compiled dispatch sequence.
/// Per-dispatch kernel-variant selection at session construction: each
/// convolution and dense product takes the path its kernel family selects
/// (cooperative tiles with their padding, or 32-wide tiles), then the
/// RmsNorm→matmul prologue fusions that depend on cooperative paths run.
/// Mutates the plan in place. Tuning may replace these initial choices by
/// measurement, asking the same families what is legal.
pub(crate) fn select_variants(
    plan: &mut ExecutionPlan,
    coop_config: Option<&crate::codegen::CoopConfig>,
    fuse_prologues: bool,
    allow_raw_f16: bool,
) {
    select_conv_paths(plan, coop_config, allow_raw_f16);
    select_matmul_paths(plan, coop_config, allow_raw_f16);

    // Apply RmsNorm+MatMul prologue fusion only after per-dispatch coop
    // selection. A coop-capable device can still route small matmuls to a
    // scalar pipeline, which has no prologue implementation. The fusion
    // pass filters on `use_coop` and records its factor buffers as declared
    // inputs for scheduling and memory planning. Debug sessions skip it —
    // dispatch-level fusion is numerics-neutral, and keeping the RmsNorm
    // output materialized is the point of debug mode.
    if fuse_prologues {
        crate::compile::fuse_rmsnorm_prologues(plan);
        crate::compile::fuse_rmsnorm_into_add(plan);
        crate::compile::fuse_rmsnorm_into_gemv(plan);
    }
    // Aligned native-f32 products share 64-row tiles between four subgroups.
    // Up to 16 K partitions of at least 128 each raise small outputs toward
    // 512 workgroups; 128 columns are used when partitions can still reach
    // that target, 64 otherwise. Ragged, pinned and fused-epilogue products
    // keep their kernel; tuning can challenge unsplit choices.
    if coop_config.is_some_and(|c| c.tile_size == 16 && !c.use_f16_input && !c.compensated) {
        for index in (0..plan.dispatches.len()).rev() {
            let dispatch = &plan.dispatches[index];
            let Some((m, n, k)) = dispatch.mnk() else {
                continue;
            };
            let max_splits = 1u32 << (k / 128).clamp(1, 16).ilog2();
            let groups = |columns: u32| u64::from(m / 64) * u64::from(n / columns);
            let wide = n.is_multiple_of(128) && groups(128) * u64::from(max_splits) >= 512;
            let shape = crate::codegen::CooperativeMatmulShape {
                columns: if wide { 128 } else { 64 },
                k_stage: 16,
                prefetch: false,
            };
            if !shape.fits_dimensions(m, n, k)
                || tiled_coop_shared_bytes(dispatch, shape) > 16 * 1024
            {
                continue;
            }
            let splits = 512u64
                .div_ceil(groups(shape.columns))
                .next_power_of_two()
                .min(u64::from(max_splits)) as u32;
            // Fused, pinned and aliased forms keep their kernel.
            let _ = plan.tile_native_f32_matmul(index, shape, splits, 32 * 1024 * 1024);
        }
    }
}

/// Promote each 64-wide convolution its kernel family admits to a
/// generated cooperative kernel, and size its workgroups and output.
fn select_conv_paths(
    plan: &mut ExecutionPlan,
    coop_config: Option<&crate::codegen::CoopConfig>,
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
        if conv::select(&problem, &target) != conv::Path::Cooperative {
            continue;
        }
        let config = coop_config.expect("cooperative paths need a configuration");
        let geometry = conv::cooperative_geometry(&problem, config)
            .expect("an admitted cooperative convolution is legal");
        let dispatch = &mut plan.dispatches[index];
        dispatch.kernel = crate::compile::Kernel::Cooperative;
        dispatch.shader = problem.cooperative_entry();
        dispatch.workgroups = geometry.grid;
        // Direct cooperative stores cover complete sub-tiles.
        pad_buffer(
            &mut plan.buffers,
            dispatch.output_buffer,
            geometry.output_bytes,
        );
        if let Some(&extra) = dispatch.input_buffers.get(2) {
            pad_buffer(&mut plan.buffers, extra, geometry.output_bytes);
        }
    }
}

/// Pick each dense product's path from its kernel family, and size its
/// workgroups and buffers for it.
fn select_matmul_paths(
    plan: &mut ExecutionPlan,
    coop_config: Option<&crate::codegen::CoopConfig>,
    allow_raw_f16: bool,
) {
    use crate::kernels::matmul;
    let target = matmul::Target {
        cooperative: coop_config.copied(),
        allow_raw_f16,
    };
    for index in 0..plan.dispatches.len() {
        let Some(problem) = matmul::Problem::of(&plan.dispatches[index]) else {
            continue;
        };
        match matmul::select(&problem, &target) {
            matmul::Path::Cooperative => {
                let config = coop_config.expect("cooperative paths need a configuration");
                let geometry = matmul::cooperative_geometry(&problem, config)
                    .expect("an admitted cooperative product is legal");
                let dispatch = &mut plan.dispatches[index];
                dispatch.kernel = crate::compile::Kernel::Cooperative;
                dispatch.workgroups = geometry.grid;
                pad_buffer(
                    &mut plan.buffers,
                    dispatch.output_buffer,
                    geometry.output_bytes,
                );
                if geometry.pad_addend {
                    let addend = dispatch.input_buffers[2];
                    pad_buffer(&mut plan.buffers, addend, geometry.output_bytes);
                }
            }
            matmul::Path::SmallTile => {
                let dispatch = &mut plan.dispatches[index];
                dispatch.kernel = crate::compile::Kernel::SmallTile;
                dispatch.workgroups = matmul::tiled_workgroups(&problem, 32);
            }
            matmul::Path::Compiled => {}
        }
    }
}

/// Grow `buffer` to at least `bytes`.
fn pad_buffer(buffers: &mut [usize], buffer: BufferRef, bytes: usize) {
    let size = &mut buffers[buffer.0 as usize];
    *size = (*size).max(bytes);
}

#[cfg(test)]
mod block_matmul_variant_tests {
    use super::select_variants;
    use crate::{Graph, codegen::CoopConfig, compile};

    #[test]
    fn small_block_shapes_preserve_nvidia_scalar_geometry() {
        let coop = CoopConfig {
            tile_size: 16,
            use_f16_input: true,
            compensated: false,
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
                assert_eq!(serial.use_small_tiles(), grouped.use_small_tiles());
                assert_eq!(serial.workgroups[..2], grouped.workgroups[..2]);
                assert_eq!(grouped.workgroups[2], 8);
            }
        }
    }
}

#[cfg(test)]
mod tiled_cooperative_policy_tests {
    use super::select_variants;
    use crate::{
        Graph,
        codegen::CoopConfig,
        compile::{self, Kernel},
    };

    #[test]
    fn measured_aligned_products_select_bounded_tiles_and_keep_fallbacks() {
        let config = CoopConfig {
            tile_size: 16,
            use_f16_input: false,
            compensated: false,
        };
        // M, N, K and the expected (columns, splits), measured on MI300X.
        for (m, n, k, expected) in [
            (128, 2048, 2048, Some((128, 16))),
            (128, 2048, 8192, Some((128, 16))),
            (128, 16384, 2048, Some((128, 2))),
            (2048, 2048, 2048, Some((128, 1))),
            (4096, 4096, 1024, Some((128, 1))),
            (256, 2048, 2048, Some((128, 8))),
            (1024, 1536, 576, Some((128, 4))),
            // 128 columns could not reach 512 workgroups with 16 partitions.
            (64, 2048, 4096, Some((64, 16))),
            (64, 256, 2048, Some((64, 16))),
            (256, 576, 1536, Some((64, 8))),
            // Short K stays in one partition.
            (512, 512, 128, Some((64, 1))),
            (128, 2048, 2044, None),
            (96, 2048, 2048, None),
            (128, 2080, 2048, None),
        ] {
            let mut graph = Graph::new();
            let a = graph.input("a", &[m, k]);
            let b = graph.input("b", &[k, n]);
            let y = graph.matmul(a, b);
            graph.set_outputs(vec![y]);
            let mut plan = compile::compile(&graph);
            let original = plan.clone();
            select_variants(&mut plan, Some(&config), false, false);
            let tiled = plan.dispatches.iter().find_map(|d| match d.kernel {
                Kernel::CooperativeTiled { shape, splits } => {
                    assert_eq!(
                        d.workgroups,
                        [m as u32 / 64, n as u32 / shape.columns, splits]
                    );
                    assert_eq!((shape.k_stage, shape.prefetch), (16, false));
                    Some((shape.columns, splits))
                }
                _ => None,
            });
            assert_eq!(tiled, expected, "{m}x{n}x{k}");
            assert_eq!(plan.output_buffers, original.output_buffers);
            if let Some((_, splits)) = expected {
                assert_eq!(plan.dispatches.len(), if splits == 1 { 1 } else { 2 });
                assert_eq!(
                    plan.buffers.len(),
                    original.buffers.len() + usize::from(splits > 1)
                );
                if splits > 1 {
                    assert!(*plan.buffers.last().unwrap() <= 32 * 1024 * 1024);
                }
            }
        }
    }
}

#[cfg(test)]
mod coop_precision_tests {
    use crate::codegen::CoopConfig;

    // Which work keeps f32 operands is the kernel families' admission rule,
    // tested with them; these check whole plans.

    #[test]
    fn coop_f32_matmul_selects_gqa_and_vocabulary_projections() {
        let config = CoopConfig {
            tile_size: 8,
            use_f16_input: false,
            compensated: false,
        };
        for n in [192, 49_152] {
            let mut graph = crate::Graph::new();
            let x = graph.input("x", &[128, 576]);
            let w = graph.parameter("w", &[576, n]);
            let y = graph.matmul(x, w);
            graph.set_outputs(vec![y]);
            let mut plan = crate::compile::compile_with(&graph, &Default::default());
            super::select_variants(&mut plan, Some(&config), false, false);
            assert_eq!(plan.dispatches.len(), 1);
            let dispatch = &plan.dispatches[0];
            assert!(dispatch.use_coop());
            assert_eq!(dispatch.workgroups, [4, n as u32 / 32, 1]);
        }
    }

    #[test]
    fn coop_f32_promotion_and_tuning_agree_at_launch_boundaries() {
        let config = CoopConfig {
            tile_size: 8,
            use_f16_input: false,
            compensated: false,
        };
        let candidate = crate::tune::MatmulTile::CooperativeF32 { tile_size: 8 };
        for groups in [65_535, 65_536] {
            for (m, n) in [(groups * 32, 16), (16, groups * 32)] {
                // Only compile metadata: even the review's 128 MiB output
                // requires no tensor or GPU allocation in this regression.
                let mut graph = crate::Graph::new();
                let a = graph.input("a", &[m, 4]);
                let b = graph.input("b", &[4, n]);
                let y = graph.matmul(a, b);
                graph.set_outputs(vec![y]);
                let mut plan = crate::compile::compile(&graph);
                let before = plan.dispatches[0].clone();
                let buffers = plan.buffers.clone();
                let legal = groups == 65_535;
                let tuning_legal = crate::tune::TuneClass::from_dispatch(&before, Some(&config))
                    .and_then(|class| candidate.buffer_sizes(&class))
                    .is_some();
                assert_eq!(tuning_legal, legal);
                super::select_variants(&mut plan, Some(&config), false, false);
                let after = &plan.dispatches[0];
                assert_eq!(after.use_coop(), legal, "{m}x{n}x4");
                if legal {
                    assert_eq!(
                        after.workgroups,
                        [(m as u32).div_ceil(32), (n as u32).div_ceil(32), 1]
                    );
                } else {
                    assert_eq!(after.workgroups, before.workgroups);
                    assert_eq!(after.kernel, before.kernel);
                    assert_eq!(
                        plan.buffers, buffers,
                        "rejected promotion must not pad buffers"
                    );
                }
            }
        }
    }

    #[test]
    #[cfg(target_pointer_width = "64")]
    fn coop_f32_padding_does_not_wrap_at_four_gib() {
        let config = CoopConfig {
            tile_size: 8,
            use_f16_input: false,
            compensated: false,
        };
        let mut graph = crate::Graph::new();
        let a = graph.input("a", &[32_767, 4]);
        let b = graph.input("b", &[4, 32_768]);
        let y = graph.matmul(a, b);
        graph.set_outputs(vec![y]);
        let mut plan = crate::compile::compile(&graph);
        super::select_variants(&mut plan, Some(&config), false, false);
        let dispatch = &plan.dispatches[0];
        assert!(dispatch.use_coop());
        assert_eq!(
            plan.buffers[dispatch.output_buffer.0 as usize],
            1_usize << 32
        );
    }

    #[test]
    fn compensated_f16_does_not_extend_exponent_range() {
        let value = 1.0e-12_f32;
        let high = half::f16::from_f32(value);
        let low = half::f16::from_f32(value - high.to_f32());

        assert_eq!(high.to_bits(), 0);
        assert_eq!(low.to_bits(), 0);
    }
}

/// Options for session construction beyond the plan itself.
#[derive(Clone, Debug, Default)]
pub struct SessionOptions {
    /// Debug session: disable buffer aliasing and device-local placement so
    /// materialized values stay host-visible and readable via
    /// [`Session::read_node`] after a completed step. `build` also disables
    /// dispatch fusion; graph rewrites are a separate option, and an already
    /// fused plan cannot be unfused here. Costs memory and bandwidth.
    pub debug: bool,
    /// Cooperative-matrix policy (see [`CoopPolicy`]).
    pub coop: CoopPolicy,
    /// The GPU context was created with
    /// [`blade_graphics::ContextDesc::timing`], so its encoders collect pass
    /// timestamps.
    ///
    /// This has to be stated rather than inferred. Asking an encoder for
    /// timings it was not set up to collect panics, and
    /// `Capabilities::timing` reports what the device *can* do, not what a
    /// given context enabled — Blade exposes no way to ask. Set it alongside
    /// [`GpuOptions::timing`]; [`SessionOptions::from_env`] does so from
    /// `MEGANEURA_GPU_TIMING`, which is read before the context exists.
    ///
    /// Leaving it false where the context does collect timestamps is safe:
    /// no timings are harvested, and a profile capture reports
    /// `MissingGpuTimings` rather than misbehaving.
    pub gpu_timing: bool,
    /// Disable buffer lifetime aliasing (independent of `debug`).
    pub no_alias: bool,
    /// Keep every buffer host-visible instead of device-local.
    pub no_device_local: bool,
    /// Fill every allocation that holds no parameter with NaN (all bits set)
    /// instead of zero at session build. Constants are still uploaded, and
    /// inputs are whatever the caller writes. A kernel that reads memory no
    /// dispatch wrote — padding beyond a logical extent, a buffer another
    /// tenant of its allocation left behind, a fused-away producer — then
    /// yields NaN instead of a plausible zero. Testing aid.
    pub poison: bool,
    /// Skip zeroing named parameter buffers during session construction.
    ///
    /// This is disabled by default so an unset parameter has deterministic
    /// zero contents. Checkpoint loaders which guarantee that every parameter
    /// is written before the first step can enable it to avoid touching the
    /// complete model allocation twice.
    pub skip_parameter_zero: bool,
    /// One compute pass per dispatch — serial execution for bisection.
    pub serial_dispatch: bool,
    /// Dump dispatch order, provenance, accesses, and the alias map at
    /// session build.
    pub dump_plan: bool,
    /// Force-pin logical buffers by id/range, e.g. `"3,17,25-40"` — the
    /// aliasing-corruption bisection aid.
    pub pin_buffers: Option<String>,
    /// Largest parameter-arena chunk in bytes (default
    /// [`crate::memplan::ARENA_CHUNK_BYTES`]); smaller values spread the
    /// optimizer over more dispatches, for testing.
    pub arena_chunk_bytes: Option<usize>,
    /// Reuse one staging buffer across `set_parameter` uploads instead of
    /// restaging per parameter.
    pub reuse_upload_staging: bool,
    /// Experimental parameter placement for parameters whose buffers stay
    /// unaliased: host-visible by default, `Some(Memory::DeviceTransient)`
    /// or `Some(Memory::Device)` relocates them for measurement.
    pub device_parameters: Option<blade_graphics::Memory>,
    /// Directory to write every WGSL shader the session's pipeline layer
    /// compiles into, up front and during tuning. Debug hook: each file
    /// names the shader (or its entry point) and a content hash. Nothing
    /// is written when unset.
    pub wgsl_dump_dir: Option<String>,
}

/// How cooperative-matrix hardware may be used.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CoopPolicy {
    /// Prefer f32 operands and accumulators. On f16-only devices, enable f16 tiles for
    /// work that permits reduced input precision; `requires_full_precision`
    /// dispatches retain scalar f32 operands.
    #[default]
    Auto,
    /// Allow only f32 cooperative operands and accumulators, without f16
    /// staging or compensation. This is a precision policy, not a vendor
    /// restriction. Devices with only reduced-input cooperative tiles retain
    /// scalar f32 implementations.
    NativeF32,
    /// Never use cooperative matrices — force the scalar paths.
    Disabled,
    /// Use f16 tiles without residual compensation, including for
    /// derivative work. Faster than [`Self::Auto`] on f16-only devices,
    /// and can overflow or lose gradient range.
    AllowF16,
}

impl CoopPolicy {
    pub(crate) fn filter_caps(
        self,
        mut caps: crate::codegen::CoopCaps,
    ) -> crate::codegen::CoopCaps {
        match self {
            Self::Disabled => caps = crate::codegen::CoopCaps::default(),
            Self::NativeF32 => caps.f16_tile = 0,
            Self::Auto | Self::AllowF16 => {}
        }
        caps
    }
}

/// Why [`Session::read_node`] could not return a value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReadNodeError {
    /// The node id has no buffer in the plan (out of range, or the plan
    /// predates provenance recording).
    UnknownNode,
    /// No node carries this name. Contains the names that do exist.
    UnknownName(Vec<String>),
    /// The value was eliminated by fusion: its buffer is never written.
    /// Rebuild with `build_session_unoptimized` / `MEGANEURA_OPTIMIZER=off`
    /// or mark the node as an output to keep it.
    FusedAway,
    /// The buffer shares a physical allocation with other values
    /// (lifetime aliasing), so its content after a step may belong to a
    /// later tenant. Build the session with [`SessionOptions::debug`] to
    /// pin everything.
    Aliased,
}

impl std::fmt::Display for ReadNodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            ReadNodeError::UnknownNode => write!(f, "node id has no buffer in the plan"),
            ReadNodeError::UnknownName(ref names) => {
                write!(f, "no node with this name; named nodes: {names:?}")
            }
            ReadNodeError::FusedAway => write!(
                f,
                "value was fused into a neighboring kernel and never materialized; \
                 rebuild unoptimized or mark it as an output"
            ),
            ReadNodeError::Aliased => write!(
                f,
                "buffer is lifetime-aliased and may hold a later value; \
                 use a debug session (SessionOptions {{ debug: true }})"
            ),
        }
    }
}

impl std::error::Error for ReadNodeError {}

/// One suspicious dispatch output found by [`Session::step_debug`].
#[derive(Clone, Debug)]
pub struct DispatchAnomaly {
    /// Index into `plan.dispatches`.
    pub dispatch: usize,
    /// Human-readable label (includes the graph-node name when present).
    pub label: String,
    /// Graph node ids this dispatch implements.
    pub origin: Vec<u32>,
    pub has_nan: bool,
    pub has_inf: bool,
    pub max_abs: f32,
}

/// Result of [`Session::step_debug`]: the per-dispatch numeric health scan.
#[derive(Clone, Debug, Default)]
pub struct DebugStepReport {
    /// Dispatches whose scanned primary-output prefix contains NaN/Inf,
    /// listed in plan order. Post-step scanning does not establish root cause.
    pub anomalies: Vec<DispatchAnomaly>,
    /// Dispatches skipped because their output buffer is aliased (only in
    /// non-debug sessions).
    pub skipped_aliased: usize,
}

/// A GPU allocation shared by one or more sessions. The final reference owns
/// destruction, which lets separately compiled execution shapes reuse model
/// parameters and state without duplicate uploads or double-free hazards.
struct PhysicalBuffer {
    gpu: Arc<Gpu>,
    handle: blade_graphics::Buffer,
}

impl Drop for PhysicalBuffer {
    fn drop(&mut self) {
        self.gpu.destroy_buffer(self.handle);
    }
}

struct UploadStaging {
    buffer: blade_graphics::Buffer,
    size: usize,
}

#[derive(Default)]
struct Readback {
    staging: Option<UploadStaging>,
    encoder: Option<blade_graphics::CommandEncoder>,
    // Mapped address and byte count: imported buffers may use a different heap.
    staged: HashMap<(usize, usize), bool>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ShareParameterError {
    DifferentContext,
    UnknownTarget,
    UnknownSource,
    Incompatible { target: usize, source: usize },
}

impl std::fmt::Display for ShareParameterError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::DifferentContext => write!(f, "sessions use different GPU contexts"),
            Self::UnknownTarget => write!(f, "target session has no parameter by that name"),
            Self::UnknownSource => write!(f, "source session has no parameter by that name"),
            Self::Incompatible { target, source } => write!(
                f,
                "parameter storage sizes differ: target needs {target} bytes, source has {source}"
            ),
        }
    }
}

impl std::error::Error for ShareParameterError {}

impl DebugStepReport {
    /// The first reported nonfinite primary-output prefix, if any.
    pub fn first_bad(&self) -> Option<&DispatchAnomaly> {
        self.anomalies.first()
    }
}

pub struct Session {
    gpu: Arc<Gpu>,
    /// Per-logical-buffer view: `buffers[i]` backs `BufferRef(i)`, at its
    /// offset within its physical allocation. Aliased logical buffers share
    /// a handle; the deduplicated owning handles live in `physical_buffers`.
    buffers: Vec<blade_graphics::BufferPiece>,
    /// One handle per physical allocation (what actually gets destroyed).
    physical_buffers: Vec<Arc<PhysicalBuffer>>,
    /// Logical-to-physical mapping, allocation sizes, and per-allocation
    /// device-local placement.
    alias: crate::memplan::AliasPlan,
    pipelines: Pipelines,
    /// Session policy/capabilities after the cooperative smoke test. Tuning
    /// must not re-enable a rejected or explicitly disabled implementation.
    coop_config: Option<crate::codegen::CoopConfig>,
    plan: ExecutionPlan,
    /// Pre-computed barrier groups: each range of dispatch indices shares one
    /// compute pass. Pass boundaries in blade emit ALL_COMMANDS barriers.
    groups: Vec<std::ops::Range<usize>>,
    encoder: blade_graphics::CommandEncoder,
    /// Caller-selected upper bound on the submissions used by `step()`.
    /// See [`Session::set_submission_chunks`]. Always at least 1.
    submission_chunks: usize,
    sync_point: Option<blade_graphics::SyncPoint>,
    /// Latest caller submission carrying work from [`Session::record`], as
    /// reported through [`Session::track_submission`]. Waited on alongside
    /// `sync_point`; it has no timings to harvest from `encoder`.
    external_sync_point: Option<blade_graphics::SyncPoint>,
    /// Calibrated timings harvested when the most recent submission completed.
    last_gpu_timings: Option<crate::profiler::GpuTimings>,
    /// This session's context collects pass timestamps. See
    /// [`SessionOptions::gpu_timing`]; asking an encoder for timings it was
    /// not set up to collect panics, so this gates every harvest.
    gpu_timing: bool,
    /// When set, run in multi-pass mode over this range of dispatch indices:
    /// one compute pass with an individual GPU timestamp per dispatch inside
    /// the range, ordinary grouped passes outside it. Enables
    /// `dump_gpu_timings()`. See [`Session::set_profiling_window`].
    profile_window: Option<std::ops::Range<usize>>,
    /// Plan dispatch behind each compute pass the last profiled `step()`
    /// encoded, in pass order. `None` marks a grouped pass that batched many
    /// dispatches under one timestamp.
    profiled_pass_map: Vec<Option<usize>>,
    /// Debug session: aliasing off, every buffer host-visible, all node
    /// values readable via [`Session::read_node`].
    debug: bool,
    /// Optimizer / gradient scratch (Adam m/v, clip acc, grad-accum) live
    /// in device-local memory. False when `debug` or `no_device_local`.
    optimizer_device: bool,
    /// Per logical buffer: written by any dispatch or session setup.
    written: Vec<bool>,
    /// Active SGD learning rate. When set, every `step()` appends SGD
    /// updates to the same GPU submission (avoiding a separate submit/wait
    /// cycle). Persistent: stays in effect across `step()` calls until
    /// overridden by another `set_learning_rate` / `set_adam` call or
    /// cleared via `clear_optimizer()`.
    pending_lr: Option<f32>,
    /// Adam/LaProp moments `(m, v)` with the optimizer chunks' layout,
    /// absent until configured or restored.
    adam_state: Option<(optimizer::ChunkBuffers, optimizer::ChunkBuffers)>,
    /// Layout of the trainable pairs for the optimizer passes.
    optimizer_chunks: Vec<optimizer::Chunk>,
    /// Host-visible segment table the optimizer passes read.
    optimizer_segments: Option<blade_graphics::Buffer>,
    /// Contents last written to `optimizer_segments`. The table is only
    /// rewritten when it changes, so a step recorded while an earlier one is
    /// in flight does not touch memory the GPU may be reading.
    optimizer_table: Vec<optimizer::Segment>,
    /// Optional exact temporal sum of grouped gradient L2 norms for one
    /// parameter. Adam already visits every scalar gradient, so collecting
    /// this diagnostic does not require another dispatch or shader variant.
    adam_grouped_grad_norm: Option<AdamGroupedGradNorm>,
    /// Single-element f32 holding the sum of squares of all gradient
    /// buffers in the current step(). Written by the final GradClipNormSq
    /// dispatch, consumed by GradClipScale. `None` when the plan has no
    /// trainable parameters.
    grad_clip_acc: Option<blade_graphics::Buffer>,
    /// Two f32 per clip workgroup (see [`grad_clip_workgroups`]): global
    /// clipping uses the first half as squared partial sums for
    /// `grad_clip_acc`, adaptive clipping stores (param², grad²) pairs.
    /// `None` without trainable parameters.
    grad_clip_partials: Option<blade_graphics::Buffer>,
    /// One adaptive-clip scale per parameter.
    agc_scales: Option<blade_graphics::Buffer>,
    /// Persistent per-parameter gradient accumulators (parallel to
    /// `plan.param_grad_pairs`). When `grad_accum_scale` is `Some`, each
    /// `step()` adds `grad * scale` into these instead of letting the
    /// optimizer read the single-step grad — giving PyTorch-style
    /// temporal accumulation. Allocated lazily on first
    /// `set_grad_accumulate`; cleared by `zero_grad`.
    grad_accum: Option<optimizer::ChunkBuffers>,
    /// `Some(scale)` enables temporal accumulation (`scale` = 1/micro so
    /// the accumulator holds the mean grad); `None` = direct optimizer
    /// reads of the per-step grad (default).
    grad_accum_scale: Option<f32>,
    /// Adam step counter.
    adam_step: u32,
    /// Active adaptive optimizer parameters. Adam and LaProp share the same
    /// two moment buffers but differ in whether momentum is accumulated
    /// before or after RMS normalization.
    pending_adam: Option<(f32, f32, f32, f32, AdaptiveOptimizer)>,
    /// Decoupled weight-decay coefficient (AdamW). 0.0 = plain Adam.
    adam_wd: f32,
    /// Maximum L2 norm of the per-step concatenated gradient. When set,
    /// `step()` splits its submission so it can read all gradient buffers
    /// to CPU between backward and optimizer, scale them by
    /// `min(1, max_norm / total_norm)`, and submit the optimizer pass
    /// with bounded gradients. Persists across calls (set once at agent
    /// build, not per-step). `None` (default) is the unclipped fast path.
    pending_grad_clip: Option<f32>,
    /// Per-parameter adaptive gradient clipping `(clip, minimum_param_norm)`.
    pending_agc: Option<(f32, f32)>,
    /// Clip cadence: when grad clipping is enabled, only compute the clip
    /// every N `step()` calls. N=1 (default) clips every step (PyTorch
    /// semantics). N>1 amortizes the extra submit/readback cost — the
    /// NaN-collapse failure mode accumulates over thousands of steps so
    /// clipping every ~5-10 steps still bounds the runaway. Internal
    /// counter `grad_clip_tick` advances each step.
    grad_clip_every: u32,
    grad_clip_tick: u32,
    /// Per-parameter learning rate multipliers, keyed by name prefix.
    /// When the SGD/Adam update applies to a parameter whose name starts
    /// with one of these prefixes, the effective LR is `base_lr * mul`.
    /// Longest-prefix-match wins; default multiplier is 1.0. Empty by
    /// default (preserves base LR for all params).
    lr_multipliers: Vec<(String, f32)>,
    /// Packed HorizontalConcat restage: source uploads write a column
    /// range into this host copy, which is then uploaded as a whole.
    packed_concat_staging: HashMap<crate::compile::BufferRef, Vec<u8>>,
    upload_staging: RefCell<Option<UploadStaging>>,
    readback: RefCell<Readback>,
    reuse_upload_staging: bool,
}

// SAFETY: a session owns or reference-counts every GPU object it contains.
// Blade's Vulkan encoder is not auto-Send only because its persistently mapped
// scratch allocation is represented by a raw pointer; moving the encoder does
// not move or invalidate that allocation. Metal and Vulkan permit their
// command objects to be used from another host thread when externally
// synchronized. Session's mutating/submit/read methods require `&mut self`, so
// transferring ownership is safe; Session intentionally remains !Sync.
unsafe impl Send for Session {}

#[cfg(test)]
mod session_thread_traits {
    use super::Session;

    #[test]
    fn session_can_transfer_between_serial_workers() {
        fn assert_send<T: Send>() {}
        assert_send::<Session>();
    }
}

struct AdamGroupedGradNorm {
    param_index: usize,
    group_size: u32,
    len: usize,
    buffer: blade_graphics::Buffer,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u32)]
enum AdaptiveOptimizer {
    Adam = 0,
    LaProp = 1,
}

/// Identifies a graph-facing slot that can be backed by an imported
/// external buffer. See [`Session::bind_external_buffer`].
#[derive(Clone, Copy, Debug)]
pub enum ExternalSlot<'a> {
    /// Graph input declared by `Graph::input(name, ...)`.
    Input(&'a str),
    /// Graph parameter declared by `Graph::parameter(name, ...)`.
    Parameter(&'a str),
    /// Graph output, by index. Index 0 is the primary output.
    Output(usize),
}

/// Error returned by [`Session::bind_external_buffer`].
#[derive(Debug)]
pub enum ExternalBindError {
    /// No input/parameter of that name, or output index out of range.
    UnknownSlot,
    /// The external handle's backing memory is smaller than the slot.
    TooSmall { required: u64, got: u64 },
}

impl std::fmt::Display for ExternalBindError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::UnknownSlot => write!(f, "no graph slot with the given name/index"),
            Self::TooSmall { required, got } => write!(
                f,
                "external buffer too small: got {got} bytes, need at least {required}"
            ),
        }
    }
}

impl std::error::Error for ExternalBindError {}

/// Error returned by [`Session::record`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecordError {
    /// Temporal gradient accumulation is enabled. Its step needs a
    /// submission boundary between accumulating and applying the update,
    /// which a caller-owned encoder cannot provide.
    GradAccumulation,
}

impl std::fmt::Display for RecordError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::GradAccumulation => write!(
                f,
                "a step with gradient accumulation cannot be recorded into a caller's encoder"
            ),
        }
    }
}

impl std::error::Error for RecordError {}

/// Workgroups the global gradient-norm pass gives one parameter.
///
/// About 16 elements per lane, so small tensors keep one workgroup and the
/// embedding tables that dominate a language model spread across the device
/// instead of streaming through a single workgroup.
/// `piece` advanced by `offset` bytes.
fn piece_at(piece: blade_graphics::BufferPiece, offset: u64) -> blade_graphics::BufferPiece {
    blade_graphics::BufferPiece {
        buffer: piece.buffer,
        offset: piece.offset + offset,
    }
}

fn create_optimizer_buffer(
    gpu: &Gpu,
    name: &str,
    size: u64,
    device_local: bool,
    device_bufs: &mut Vec<(blade_graphics::Buffer, u64)>,
) -> blade_graphics::Buffer {
    let buf = gpu.create_buffer(blade_graphics::BufferDesc {
        name,
        size,
        memory: if device_local {
            blade_graphics::Memory::DeviceTransient
        } else {
            blade_graphics::Memory::Shared
        },
    });
    if device_local {
        device_bufs.push((buf, size));
    } else {
        unsafe {
            std::ptr::write_bytes(buf.data(), 0, size as usize);
        }
    }
    buf
}

/// Equal allocation sizes alone do not make fused parameter contents equal.
fn parameter_storage_matches(
    target_plan: &ExecutionPlan,
    target: BufferRef,
    source_plan: &ExecutionPlan,
    source: BufferRef,
) -> bool {
    let target_recipe = target_plan
        .derived_params
        .iter()
        .find(|derived| derived.0 == target)
        .map(|derived| (&derived.1, &derived.2));
    let source_recipe = source_plan
        .derived_params
        .iter()
        .find(|derived| derived.0 == source)
        .map(|derived| (&derived.1, &derived.2));
    target_plan.buffers[target.0 as usize] == source_plan.buffers[source.0 as usize]
        && target_plan.weight_buffers.get(&target) == source_plan.weight_buffers.get(&source)
        && target_plan.param_types.get(&target) == source_plan.param_types.get(&source)
        && target_recipe == source_recipe
}

impl Session {
    /// Select the safest cooperative matrix config from GPU capabilities.
    /// Prefers f32 operands for training correctness. The f16-input path
    /// remains opt-in because rounding compounds across deep training graphs.
    fn select_coop_config(
        caps: &crate::codegen::CoopCaps,
        policy: CoopPolicy,
    ) -> Option<crate::codegen::CoopConfig> {
        use crate::codegen::CoopConfig;
        // Escape hatch for diagnosing coop-matrix numerical bugs.
        if policy == CoopPolicy::Disabled {
            log::warn!("cooperative matrices disabled by policy — forcing scalar matmul");
            return None;
        }
        let caps = policy.filter_caps(crate::codegen::CoopCaps {
            f16_tile: caps.f16_tile,
            f32_tile: caps.f32_tile,
        });
        log::info!(
            "coop caps: f16_tile={}, f32_tile={}",
            caps.f16_tile,
            caps.f32_tile
        );
        // Prefer f32 cooperative tiles. f16-only devices enable the f16 path for
        // precision-insensitive work; `select_variants` keeps derivative
        // work scalar unless the caller explicitly opts into raw f16 via
        // `CoopPolicy::AllowF16`.
        if caps.f32_tile > 0 {
            Some(CoopConfig {
                tile_size: caps.f32_tile,
                use_f16_input: false,
                compensated: false,
            })
        } else if caps.f16_tile > 0 {
            Some(CoopConfig {
                tile_size: caps.f16_tile,
                use_f16_input: true,
                compensated: false,
            })
        } else {
            None
        }
    }

    /// [`Self::test_coop_matmul`], once per context and configuration.
    /// The test compiles and runs a kernel, and its answer does not change
    /// for a context, so later sessions on it reuse the result.
    fn coop_matmul_qualified(gpu: &Arc<Gpu>, config: &crate::codegen::CoopConfig) -> bool {
        // A listed `Weak` keeps its context address from being reused.
        type Probe = (std::sync::Weak<Gpu>, crate::codegen::CoopConfig, bool);
        static PROBES: std::sync::Mutex<Vec<Probe>> = std::sync::Mutex::new(Vec::new());
        let mut probes = PROBES.lock().unwrap();
        probes.retain(|probe| probe.0.strong_count() != 0);
        if let Some(probe) = probes
            .iter()
            .find(|probe| std::ptr::eq(probe.0.as_ptr(), Arc::as_ptr(gpu)) && probe.1 == *config)
        {
            return probe.2;
        }
        let qualified = Self::test_coop_matmul(gpu, config);
        probes.push((Arc::downgrade(gpu), *config, qualified));
        qualified
    }

    /// Run a tiny cooperative matmul and check the result.
    /// Returns false if the GPU doesn't support the required cooperative
    /// matrix types (e.g. AMD RADV advertises the extension but rejects
    /// the specific f32 matrix shapes).
    fn test_coop_matmul(gpu: &Gpu, config: &crate::codegen::CoopConfig) -> bool {
        use crate::codegen::ShaderGroup;
        use blade_graphics as bg;

        let sm = crate::codegen::generate_module_coop(ShaderGroup::MatMul, config);
        let shader = match gpu.try_create_shader(bg::ShaderDesc {
            source: &sm.source,
            naga_module: Some(sm.module),
        }) {
            Ok(s) => s,
            Err(e) => {
                log::warn!("cooperative matmul shader rejected: {}", e);
                return false;
            }
        };
        let layout = shader_data_layout(&ShaderEntry::MatMul);
        let mut pipeline =
            create_profiled_pipeline(gpu, "coop_probe".to_string(), &layout, shader.at("main"));

        // Test with a multi-tile matmul using varying values. A uniform
        // pattern like A[i,j]=i+1, B[i,j]=j+1 misses bugs where the shader
        // loses index dependence (e.g. coop-load layout mismatches that
        // only surface when neighboring K-indices carry different weights).
        let ot = config.matmul_output_tile() as usize;
        let m: usize = ot * 2; // 2 row tiles
        let inner: usize = ot * 2; // 2 K-tiles (exercises K-loop accumulation)
        let n_out: usize = ot * 2; // 2 column tiles
        let a_size = (m * inner * 4) as u64;
        let b_size = (inner * n_out * 4) as u64;
        let c_size = (m * n_out * 4) as u64;
        let a_buf = gpu.create_buffer(bg::BufferDesc {
            name: "test_a",
            size: a_size,
            memory: bg::Memory::Shared,
        });
        let b_buf = gpu.create_buffer(bg::BufferDesc {
            name: "test_b",
            size: b_size,
            memory: bg::Memory::Shared,
        });
        let c_buf = gpu.create_buffer(bg::BufferDesc {
            name: "test_c",
            size: c_size,
            memory: bg::Memory::Shared,
        });
        // Deterministic varying values that depend on both row and column.
        // Keep magnitudes modest so f16 coop paths don't overflow.
        let fill =
            |idx: usize, dim: usize| -> f32 { (((idx * 31 + dim * 7) % 17) as f32 - 8.0) * 0.125 };
        let a: Vec<f32> = (0..m * inner)
            .map(|index| fill(index / inner, index % inner))
            .collect();
        let b: Vec<f32> = (0..inner * n_out)
            .map(|index| fill(index / n_out + 100, index % n_out + 200))
            .collect();
        let mut expected = vec![0.0f32; m * n_out];
        for i in 0..m {
            for j in 0..n_out {
                let mut acc = 0.0f32;
                for p in 0..inner {
                    acc += a[i * inner + p] * b[p * n_out + j];
                }
                expected[i * n_out + j] = acc;
            }
        }
        // Shared memory can be uncached or write-combined for the host, which
        // makes every read slow: fill it, never read it back on the CPU.
        unsafe {
            std::slice::from_raw_parts_mut(a_buf.data() as *mut f32, a.len()).copy_from_slice(&a);
            std::slice::from_raw_parts_mut(b_buf.data() as *mut f32, b.len()).copy_from_slice(&b);
            std::slice::from_raw_parts_mut(c_buf.data() as *mut f32, m * n_out).fill(0.0);
        }

        let mut encoder = gpu.create_command_encoder(bg::CommandEncoderDesc {
            name: "coop_test",
            buffer_count: 2,
            manual_barriers: false,
        });
        encoder.start();
        {
            let mut pass = encoder.compute("coop_test");
            let mut pc = pass.with(&pipeline);
            let ot = config.matmul_output_tile();
            pc.bind(
                0,
                &MatMulData {
                    matrix_a: a_buf.at(0),
                    matrix_b: b_buf.at(0),
                    matrix_c: c_buf.at(0),
                    params: MatMulParams {
                        m: m as u32,
                        n: n_out as u32,
                        k: inner as u32,
                        _pad: 0,
                    },
                },
            );
            pc.dispatch([(m as u32).div_ceil(ot), (n_out as u32).div_ceil(ot), 1]);
        }
        let sp = gpu.submit(&mut encoder);
        // A standalone scratch submission with no session behind it: nothing
        // is profiling it, so do not ask for timestamps it may not collect.
        let _ = wait_for_timed_encoder(gpu, &sp, &mut encoder, false);

        let result =
            unsafe { std::slice::from_raw_parts(c_buf.data() as *const f32, m * n_out).to_vec() };

        gpu.destroy_command_encoder(&mut encoder);
        gpu.destroy_compute_pipeline(&mut pipeline);
        gpu.destroy_buffer(a_buf);
        gpu.destroy_buffer(b_buf);
        gpu.destroy_buffer(c_buf);

        // Compare against CPU-computed reference. f16 coop paths accumulate
        // in f32 but the inputs are truncated, so leave ~5% slack per term.
        let tol = 0.05 * (inner as f32);
        let mut ok = true;
        let mut first_mismatch = true;
        for i in 0..m {
            for j in 0..n_out {
                let got = result[i * n_out + j];
                let want = expected[i * n_out + j];
                if (got - want).abs() > tol && (got - want).abs() > 1e-3 {
                    if first_mismatch {
                        log::warn!(
                            "coop self-test FAILED (m={m}, n={n_out}, k={inner}, tile={}, tol={tol:.3})",
                            config.tile_size
                        );
                        log::warn!("  got row 0: {:?}", &result[..n_out.min(8)]);
                        log::warn!("  want row 0: {:?}", &expected[..n_out.min(8)]);
                        first_mismatch = false;
                    }
                    ok = false;
                }
            }
        }
        ok
    }

    /// Create a session from a compiled execution plan.
    ///
    /// Reuses the process-default GPU context. Use [`Session::with_context`]
    /// to share a context with an existing Blade-based renderer instead.
    pub fn new(plan: ExecutionPlan) -> Self {
        Self::with_context(plan, default_gpu_context())
    }

    /// Create a session that reuses an externally-owned Blade GPU context,
    /// with explicit session options. [`Session::with_context`] is the
    /// defaults-taking shorthand.
    pub fn with_context_opts(plan: ExecutionPlan, gpu: Arc<Gpu>, opts: SessionOptions) -> Self {
        Self::build_session_impl(plan, gpu, opts, None, false)
    }

    pub(crate) fn with_context_opts_sharing(
        plan: ExecutionPlan,
        gpu: Arc<Gpu>,
        opts: SessionOptions,
        parameter_source: Option<&mut Session>,
    ) -> Self {
        Self::build_session_impl(plan, gpu, opts, parameter_source, false)
    }

    /// A measured-search challenger, constructed on the idle incumbent's
    /// identically stored parameters that neither program writes. Written
    /// parameters are search state, reset between trials, so they stay private.
    pub(crate) fn with_context_opts_inheriting(
        plan: ExecutionPlan,
        gpu: Arc<Gpu>,
        opts: SessionOptions,
        incumbent: &mut Session,
    ) -> Self {
        Self::build_session_impl(plan, gpu, opts, Some(incumbent), true)
    }

    /// Create a session that reuses an externally-owned Blade GPU context.
    ///
    /// Intended for embedding: a host application (renderer, game) creates
    /// the `blade_graphics::Context`, wraps it in `Arc`, and hands a clone
    /// to meganeura. Both sides share the same device and queue. The
    /// session destroys only the resources it allocated — buffers,
    /// pipelines, the command encoder — on drop; the context is released
    /// once the last `Arc` clone is dropped.
    pub fn with_context(plan: ExecutionPlan, gpu: Arc<Gpu>) -> Self {
        Self::build_session_impl(plan, gpu, SessionOptions::default(), None, false)
    }

    fn build_session_impl(
        plan: ExecutionPlan,
        gpu: Arc<Gpu>,
        opts: SessionOptions,
        mut parameter_source: Option<&mut Session>,
        immutable_only: bool,
    ) -> Self {
        if let Some(source) = parameter_source.as_deref_mut() {
            assert!(
                Arc::ptr_eq(&gpu, &source.gpu),
                "parameter source uses a different GPU context"
            );
            source.wait();
        }
        let _session_span = tracing::info_span!(
            "session_init",
            dispatches = plan.dispatches.len(),
            buffers = plan.buffers.len()
        )
        .entered();
        let coop_caps = auto_tune(&gpu, 0).coop_caps;
        let coop_config = {
            let _span = tracing::info_span!("coop_probe").entered();
            Self::select_coop_config(&coop_caps, opts.coop)
                .filter(|config| Self::coop_matmul_qualified(&gpu, config))
        };
        if let Some(ref config) = coop_config {
            log::info!(
                "cooperative matrix enabled (tile={}×{}, {}, f32_tile={}, f16_tile={})",
                config.tile_size,
                config.tile_size,
                if config.use_f16_input {
                    "f16→f32"
                } else {
                    "f32"
                },
                coop_caps.f32_tile,
                coop_caps.f16_tile,
            );
        } else {
            let info = gpu.device_information();
            log::warn!(
                "cooperative matrix not available on {} ({}) (f32_tile={}, f16_tile={}); using naive matmul",
                info.device_name,
                info.driver_name,
                coop_caps.f32_tile,
                coop_caps.f16_tile,
            );
        }

        let mut plan = plan;
        let schedule_span = tracing::info_span!("schedule").entered();

        // Per-dispatch kernel-variant selection: one pass, one owner.
        select_variants(
            &mut plan,
            coop_config.as_ref(),
            !opts.debug,
            opts.coop == CoopPolicy::AllowF16,
        );
        if let Some(head_dim) = plan
            .dispatches
            .iter()
            .filter(|dispatch| dispatch.shader == ShaderEntry::FlashAttention)
            .map(|dispatch| dispatch.params[3])
            .max()
        {
            plan.knobs
                .flash
                .fit_shared_memory(head_dim, gpu.capabilities().max_compute_shared_memory_size);
        }

        // Order the dispatches into barrier groups. The compiler owns this because
        // horizontal fusion changes how many dispatches there are, so the
        // groups cannot be computed independently of it.
        if opts.serial_dispatch {
            log::info!("MEGANEURA_SERIAL_DISPATCH: forcing one dispatch per pass");
        }
        crate::compile::schedule_dispatches(
            &mut plan,
            opts.serial_dispatch,
            !opts.serial_dispatch && !opts.debug,
        );
        let groups = std::mem::take(&mut plan.groups);
        log::info!(
            "{} dispatches → {} barrier groups",
            plan.dispatches.len(),
            groups.len()
        );
        drop(schedule_span);

        // Lifetime-based buffer aliasing: step-local intermediates with
        // disjoint live ranges (at barrier-group granularity) share one
        // physical allocation. See `memplan` for the safety argument.
        // Debug sessions keep every logical buffer distinct and
        // host-visible so any node's value can be read back after a step.
        let memory_plan_span = tracing::info_span!("memory_plan").entered();
        let mut alias = if opts.debug || opts.no_alias {
            if opts.debug {
                log::info!("debug session: buffer aliasing disabled");
            } else {
                log::info!("MEGANEURA_NO_ALIAS: buffer aliasing disabled");
            }
            crate::memplan::plan_no_alias(&plan, &groups, opts.pin_buffers.as_deref())
        } else {
            crate::memplan::plan_buffer_aliasing(&plan, &groups, opts.pin_buffers.as_deref())
        };
        alias.pack_arena(
            &plan,
            opts.arena_chunk_bytes
                .unwrap_or(crate::memplan::ARENA_CHUNK_BYTES),
        );
        let device_parameters = opts.device_parameters;
        // Only a parameter-bearing physical allocation may be
        // relocated, and it must be on its own physical buffer.
        let mut parameter_memory = vec![None; alias.sizes.len()];
        if let Some(memory) = device_parameters {
            for buffer in plan
                .param_buffers
                .iter()
                .map(|entry| entry.1)
                .chain(plan.derived_params.iter().map(|entry| entry.0))
            {
                if plan.input_buffers.iter().any(|entry| entry.1 == buffer)
                    || plan.constant_buffers.iter().any(|entry| entry.0 == buffer)
                    || plan.loss_buffer == Some(buffer)
                    || !plan.buffers[buffer.0 as usize].is_multiple_of(4)
                    || plan
                        .param_types
                        .get(&buffer)
                        .is_some_and(|ty| !ty.size_bytes().is_multiple_of(4))
                {
                    continue;
                }
                let physical = alias.map[buffer.0 as usize];
                // Its own pinned allocation, or a parameter arena.
                assert!(
                    alias.arena.iter().any(|chunk| chunk.params == physical)
                        || alias
                            .map
                            .iter()
                            .enumerate()
                            .all(|(i, &p)| p != physical || i == buffer.0 as usize),
                    "device parameter must have its own pinned allocation"
                );
                alias.device_local[physical] = true;
                parameter_memory[physical] = Some(memory);
            }
        }
        // Step-local intermediates default to device-local memory on the
        // theory that host-visible (ReBAR) traffic is slower on discrete
        // boards; kill switch for measurement and UMA debugging.
        if opts.debug || opts.no_device_local {
            if !opts.debug {
                log::info!("MEGANEURA_NO_DEVICE_LOCAL: all buffers host-visible");
            }
            alias.device_local.fill(false);
        }
        // Install compatible donor bindings before budgeting or allocating.
        // A parameter in an arena leaves that arena as an individual piece;
        // this preserves the donor's offset and lets the optimizer treat the
        // rebound parameter exactly like a post-construction share.
        let mut provided_physical = vec![None; alias.sizes.len()];
        let mut provided_parameters = 0usize;
        if let Some(source) = parameter_source.as_deref() {
            // A measured search resets the parameters either program writes
            // between trials. They are search state and stay private.
            let writes = |plan: &ExecutionPlan| -> std::collections::HashSet<BufferRef> {
                if !immutable_only {
                    return Default::default();
                }
                plan.dispatches
                    .iter()
                    .flat_map(|d| std::iter::once(d.output_buffer).chain(d.extra_outputs.clone()))
                    .collect()
            };
            let (target_writes, source_writes) = (writes(&plan), writes(&source.plan));
            for &(ref name, target) in &plan.param_buffers {
                let Some(source_buffer) = source
                    .plan
                    .param_buffers
                    .iter()
                    .find(|entry| entry.0 == *name)
                    .map(|entry| entry.1)
                else {
                    continue;
                };
                if target_writes.contains(&target) || source_writes.contains(&source_buffer) {
                    continue;
                }
                let target_index = target.0 as usize;
                let source_index = source_buffer.0 as usize;
                // Same bytes, same storage format and the same logical
                // tensor; anything else keeps a private allocation.
                if !parameter_storage_matches(&plan, target, &source.plan, source_buffer) {
                    log::warn!(
                        "parameter `{name}` is stored differently in the source session; \
                         allocating it separately"
                    );
                    continue;
                }
                let source_physical = source.alias.map[source_index];
                let physical = alias.rebind(
                    target_index,
                    plan.buffers[target_index],
                    source.alias.offsets[source_index],
                    source.alias.device_local[source_physical],
                );
                provided_physical.resize(alias.sizes.len(), None);
                provided_physical[physical] =
                    Some(Arc::clone(&source.physical_buffers[source_physical]));
                provided_parameters += 1;
            }
        }
        // A fully donated parameter arena no longer has logical tenants. Keep
        // a harmless minimum-sized backing slot because optimizer metadata
        // retains its physical index, but do not reserve the arena's capacity.
        parameter_memory.resize(alias.sizes.len(), None);
        for physical in 0..alias.sizes.len() {
            if provided_physical[physical].is_none() && !alias.map.contains(&physical) {
                alias.sizes[physical] = 0;
                alias.device_local[physical] = false;
                parameter_memory[physical] = None;
            }
        }
        if provided_parameters != 0 {
            log::info!("reused {provided_parameters} parameter allocations from donor session");
        }
        let alias = alias;
        let optimizer_chunks = optimizer::chunks(&plan, &alias);
        // Debug aid: dump dispatch order, declared accesses, and the
        // alias map for corruption bisection (see MEGANEURA_PIN_BUFS).
        if opts.dump_plan {
            for (gi, range) in groups.iter().enumerate() {
                for i in range.clone() {
                    let d = &plan.dispatches[i];
                    eprintln!(
                        "g{gi:03} d{i:03} {} nodes={:?} coop={} epilogue={} in={:?} out={} wg={:?} params={:?}",
                        if d.label.is_empty() {
                            format!("{:?}", d.shader)
                        } else {
                            d.label.clone()
                        },
                        d.origin,
                        d.use_coop(),
                        d.matmul_epilogue.is_some(),
                        d.input_buffers.iter().map(|b| b.0).collect::<Vec<_>>(),
                        d.output_buffer.0,
                        d.workgroups,
                        d.params
                    );
                }
            }
            for (i, &phys) in alias.map.iter().enumerate() {
                eprintln!(
                    "buf {i:03} ({} B) -> phys {phys} ({} B)",
                    plan.buffers[i], alias.sizes[phys]
                );
            }
        }
        log::info!(
            "buffer aliasing: {} logical buffers ({:.1} MB) -> {} allocations ({:.1} MB, {:.1} MB device-local)",
            plan.buffers.len(),
            alias.logical_bytes(&plan.buffers) as f64 / 1e6,
            alias.sizes.len(),
            alias.physical_bytes() as f64 / 1e6,
            alias.device_local_bytes() as f64 / 1e6,
        );
        let physical_allocation_bytes = alias
            .sizes
            .iter()
            .zip(&provided_physical)
            .try_fold(0usize, |sum, (&size, provided)| {
                sum.checked_add(if provided.is_none() { size.max(4) } else { 0 })
            })
            .expect("session allocation size overflow");
        let planned_allocation_bytes = physical_allocation_bytes
            .checked_add(usize::from(!plan.param_grad_pairs.is_empty()) * 4)
            .expect("session allocation size overflow");
        ensure_device_memory_budget(&gpu, planned_allocation_bytes, "session buffers");
        drop(memory_plan_span);
        let parameter_allocations = {
            let mut allocations = vec![false; alias.sizes.len()];
            for &(_, buffer) in &plan.param_buffers {
                allocations[alias.map[buffer.0 as usize]] = true;
            }
            allocations
        };
        let shared_allocations = alias
            .device_local
            .iter()
            .zip(&provided_physical)
            .filter(|&(device, provided)| !*device && provided.is_none())
            .count();
        let device_allocations = alias
            .device_local
            .iter()
            .zip(&provided_physical)
            .filter(|&(device, provided)| *device && provided.is_none())
            .count();
        let shared_bytes = alias
            .sizes
            .iter()
            .zip(&alias.device_local)
            .zip(&provided_physical)
            .filter_map(|((&size, &device), provided)| {
                (!device && provided.is_none()).then_some(size.max(4))
            })
            .sum::<usize>();
        let device_bytes = physical_allocation_bytes - shared_bytes;
        let zero_on_init: Vec<bool> = alias
            .device_local
            .iter()
            .enumerate()
            .map(|(index, _)| {
                provided_physical[index].is_none()
                    && (!opts.skip_parameter_zero || !parameter_allocations[index])
            })
            .collect();
        let fill_byte: Vec<u8> = parameter_allocations
            .iter()
            .map(|&parameter| if opts.poison && !parameter { 0xFF } else { 0 })
            .collect();
        let shared_zero_allocations = alias
            .device_local
            .iter()
            .zip(&zero_on_init)
            .filter(|&(device, zero)| !*device && *zero)
            .count();
        let shared_zero_bytes = alias
            .sizes
            .iter()
            .zip(&alias.device_local)
            .zip(&zero_on_init)
            .filter_map(|((&size, &device), &zero)| (!device && zero).then_some(size.max(4)))
            .sum::<usize>();
        let buffer_alloc_span = tracing::info_span!(
            "buffer_alloc",
            allocations = alias.sizes.len(),
            bytes = planned_allocation_bytes,
            shared_bytes,
            device_bytes,
        )
        .entered();
        let physical_buffers: Vec<Arc<PhysicalBuffer>> = {
            let _span = tracing::info_span!(
                "buffer_create",
                shared_allocations,
                shared_bytes,
                device_allocations,
                device_bytes,
                trace_min_duration_us = 1_000u64,
            )
            .entered();
            let mut slots = provided_physical;
            let mut create_class = |device_local: bool| {
                for (i, &size) in alias.sizes.iter().enumerate() {
                    if slots[i].is_some() || alias.device_local[i] != device_local {
                        continue;
                    }
                    // Whole vec4s: kernels that read a tensor as
                    // `array<vec4<f32>>` reach its last elements only when
                    // the binding covers the vec4 that holds them.
                    let size = size.max(16).next_multiple_of(16);
                    let handle = gpu.create_buffer(blade_graphics::BufferDesc {
                        name: &format!("buf_{}", i),
                        size: size as u64,
                        memory: if device_local {
                            parameter_memory[i].unwrap_or(blade_graphics::Memory::DeviceTransient)
                        } else {
                            blade_graphics::Memory::Shared
                        },
                    });
                    slots[i] = Some(Arc::new(PhysicalBuffer {
                        gpu: Arc::clone(&gpu),
                        handle,
                    }));
                }
            };
            {
                let _span = tracing::info_span!(
                    "buffer_create_shared",
                    allocations = shared_allocations,
                    bytes = shared_bytes,
                    trace_min_duration_us = 1_000u64,
                )
                .entered();
                create_class(false);
            }
            {
                let _span = tracing::info_span!(
                    "buffer_create_device",
                    allocations = device_allocations,
                    bytes = device_bytes,
                    trace_min_duration_us = 1_000u64,
                )
                .entered();
                create_class(true);
            }
            slots
                .into_iter()
                .map(|slot| slot.expect("every physical allocation was created"))
                .collect()
        };
        // Zero-fill to prevent NaN from uninitialized padding regions (coop
        // tiles read/write full tiles beyond logical dimensions). A caller
        // which promises to initialize all parameters may skip those buffers;
        // every other host-visible allocation remains deterministic.
        {
            let _span = tracing::info_span!(
                "buffer_zero_host",
                allocations = shared_zero_allocations,
                bytes = shared_zero_bytes,
                trace_min_duration_us = 1_000u64,
            )
            .entered();
            for (index, buffer) in physical_buffers.iter().enumerate() {
                if !alias.device_local[index] && zero_on_init[index] {
                    unsafe {
                        std::ptr::write_bytes(
                            buffer.handle.data(),
                            fill_byte[index],
                            alias.sizes[index].max(4),
                        );
                    }
                }
            }
        }
        let buffers: Vec<blade_graphics::BufferPiece> = alias
            .map
            .iter()
            .zip(&alias.offsets)
            .map(|(&p, &offset)| physical_buffers[p].handle.at(offset as u64))
            .collect();

        // Upload constant buffer data (gradient constants, scale factors, etc.)
        {
            let _span = tracing::info_span!(
                "buffer_constants",
                buffers = plan.constant_buffers.len(),
                trace_min_duration_us = 1_000u64,
            )
            .entered();
            for &(buf_ref, ref data) in &plan.constant_buffers {
                let buffer = &buffers[buf_ref.0 as usize];
                unsafe {
                    let ptr = buffer.data() as *mut f32;
                    std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
                }
            }
        }
        drop(buffer_alloc_span);

        let pipelines = {
            let _span = tracing::info_span!("pipeline_set").entered();
            Pipelines::new(
                &gpu,
                &plan,
                coop_config.as_ref(),
                opts.wgsl_dump_dir.as_deref(),
            )
        };
        let mut encoder = {
            let _span = tracing::info_span!("encoder_create").entered();
            gpu.create_command_encoder(blade_graphics::CommandEncoderDesc {
                name: "meganeura",
                buffer_count: 2,
                manual_barriers: false,
            })
        };

        // Zero-fill device-local allocations on the GPU (no host pointer).
        // One submission at build time; the wait below orders it before
        // the first step()'s reads.
        let device_zero_allocations = alias
            .device_local
            .iter()
            .zip(&zero_on_init)
            .filter(|&(device, zero)| *device && *zero)
            .count();
        let device_zero_bytes = alias
            .sizes
            .iter()
            .zip(&alias.device_local)
            .zip(&zero_on_init)
            .filter_map(|((&size, &device), &zero)| (device && zero).then_some(size.max(4)))
            .sum::<usize>();
        if device_zero_allocations != 0 {
            let _span = tracing::info_span!(
                "buffer_zero_gpu",
                allocations = device_zero_allocations,
                bytes = device_zero_bytes,
                trace_min_duration_us = 1_000u64,
            )
            .entered();
            encoder.start();
            {
                let mut transfer = encoder.transfer("zero_device_local");
                for (i, &device_local) in alias.device_local.iter().enumerate() {
                    if device_local && zero_on_init[i] {
                        let size = alias.sizes[i].max(4) as u64;
                        transfer.fill_buffer(physical_buffers[i].handle.at(0), size, fill_byte[i]);
                    }
                }
            }
            let sp = gpu.submit(&mut encoder);
            let _ = wait_for_timed_encoder(&gpu, &sp, &mut encoder, opts.gpu_timing);
        }

        // Both halves matter: the device has to be able to timestamp, and
        // this context has to have asked for it. Blade reports only the first.
        let gpu_timing = opts.gpu_timing && gpu.capabilities().timing;

        let optimizer_device = !opts.no_device_local && !opts.debug;
        let mut optimizer_device_bufs: Vec<(blade_graphics::Buffer, u64)> = Vec::new();
        // Grad-clip accumulator: a single f32. GPU-only after the
        // barriered clip path landed; host-visible only in debug.
        let grad_clip_acc = if !plan.param_grad_pairs.is_empty() {
            Some(create_optimizer_buffer(
                &gpu,
                "grad_clip_acc",
                4,
                optimizer_device,
                &mut optimizer_device_bufs,
            ))
        } else {
            None
        };
        let grad_clip_partials = if !plan.param_grad_pairs.is_empty() {
            // One (param², grad²) pair per optimizer workgroup.
            Some(create_optimizer_buffer(
                &gpu,
                "grad_clip_partials",
                optimizer::clip_slots(&plan) * 8,
                optimizer_device,
                &mut optimizer_device_bufs,
            ))
        } else {
            None
        };
        let agc_scales = if !plan.param_grad_pairs.is_empty() {
            Some(create_optimizer_buffer(
                &gpu,
                "agc_scales",
                plan.param_grad_pairs.len() as u64 * 4,
                optimizer_device,
                &mut optimizer_device_bufs,
            ))
        } else {
            None
        };
        if !optimizer_device_bufs.is_empty() {
            encoder.start();
            {
                let mut transfer = encoder.transfer("zero_optimizer");
                for &(buf, size) in &optimizer_device_bufs {
                    transfer.fill_buffer(buf.at(0), size, 0);
                }
            }
            let sp = gpu.submit(&mut encoder);
            let _ = wait_for_timed_encoder(&gpu, &sp, &mut encoder, opts.gpu_timing);
        }

        // Buffers that some dispatch (or session setup) actually writes.
        // Values whose producer was fused away have allocated-but-never-
        // written buffers; `read_node` reports those instead of returning
        // zeros that look like data.
        let mut written = vec![false; plan.buffers.len()];
        for d in &plan.dispatches {
            written[d.output_buffer.0 as usize] = true;
            for b in &d.extra_outputs {
                written[b.0 as usize] = true;
            }
        }
        for &(_, b) in &plan.param_buffers {
            written[b.0 as usize] = true;
        }
        for &(_, b) in &plan.input_buffers {
            written[b.0 as usize] = true;
        }
        for &(b, _) in &plan.constant_buffers {
            written[b.0 as usize] = true;
        }
        for d in &plan.derived_params {
            written[d.0.0 as usize] = true;
        }
        for &(_, b) in &plan.lse_buffers {
            written[b.0 as usize] = true;
        }

        Self {
            gpu,
            buffers,
            physical_buffers,
            alias,
            pipelines,
            coop_config,
            plan,
            groups,
            encoder,
            submission_chunks: 1,
            sync_point: None,
            external_sync_point: None,
            gpu_timing,
            last_gpu_timings: None,
            profile_window: None,
            profiled_pass_map: Vec::new(),
            debug: opts.debug,
            optimizer_device,
            written,
            pending_lr: None,
            pending_grad_clip: None,
            pending_agc: None,
            grad_clip_every: 1,
            grad_clip_tick: 0,
            grad_clip_acc,
            grad_clip_partials,
            agc_scales,
            grad_accum: None,
            grad_accum_scale: None,
            lr_multipliers: Vec::new(),
            adam_state: None,
            optimizer_chunks,
            optimizer_segments: None,
            optimizer_table: Vec::new(),
            adam_grouped_grad_norm: None,
            adam_step: 0,
            pending_adam: None,
            adam_wd: 0.0,
            packed_concat_staging: HashMap::new(),
            upload_staging: RefCell::new(None),
            readback: RefCell::new(Readback::default()),
            reuse_upload_staging: opts.reuse_upload_staging,
        }
    }

    /// Import an externally-allocated buffer into a graph slot.
    ///
    /// This is true cross-context interop: a producer — a game engine,
    /// renderer, video decoder, other ML framework — allocates the
    /// buffer in *its own* Blade context using `Memory::External(...)`,
    /// then queries the OS handle via the producer context's
    /// `get_external_buffer_source(...)`. Passing that
    /// `ExternalMemorySource` here tells meganeura's context to
    /// re-import the same underlying memory. Both contexts then see
    /// identical bytes without any CPU roundtrip and without sharing a
    /// `blade_graphics::Context`.
    ///
    /// The import replaces the session's internal allocation for
    /// `slot`. After this call, dispatches that read from / write to
    /// that slot operate directly on the shared memory.
    ///
    /// # Handle types
    ///
    /// See [`blade_graphics::ExternalMemorySource`]:
    /// * `Fd(Some(fd))` — Linux/Unix opaque FD (VK_KHR_external_memory_fd).
    /// * `Dma(Some(fd))` — Linux DMA-BUF (VK_EXT_external_memory_dma_buf),
    ///   the usual channel to share with GStreamer / V4L2 / EGL / Wayland.
    /// * `Win32(Some(handle))` / `Win32KMT(Some(handle))` — Windows
    ///   (VK_KHR_external_memory_win32).
    /// * `HostAllocation(ptr)` — a host malloc'd pointer imported by
    ///   both contexts (VK_EXT_external_memory_host). Useful when the
    ///   producer is a CPU-resident tensor, e.g. during a PyTorch
    ///   migration: allocate page-aligned host memory, memcpy the
    ///   tensor in, import on both sides.
    ///
    /// # Ownership & lifetime
    ///
    /// Once imported, the resulting buffer is owned by meganeura's
    /// context — `Session::drop` will destroy it, releasing
    /// meganeura's reference to the shared memory. The producer's
    /// original buffer remains independently owned and must be
    /// destroyed by the producer separately. The underlying allocation
    /// is refcounted by the driver (Vulkan memory objects) or OS
    /// (malloc'd page), so it stays live until both sides release.
    ///
    /// FD/HANDLE ownership follows Vulkan's rules: on a successful
    /// import, the importing driver takes ownership of the handle and
    /// the caller must not close it. On failure the handle is
    /// untouched. For `HostAllocation(ptr)` the caller retains the
    /// pointer and must keep the memory live for the lifetime of the
    /// session.
    ///
    /// # Errors
    ///
    /// * [`ExternalBindError::UnknownSlot`] — no slot with that
    ///   name/index.
    /// * [`ExternalBindError::TooSmall`] — the declared `size` is below
    ///   the slot's requirement.
    ///
    /// # Platform support
    ///
    /// Vulkan only. Blade's Metal and GLES backends currently
    /// `unimplemented!()` for external memory — on those backends the
    /// import call will panic inside Blade.
    ///
    /// # Example
    ///
    /// ```no_run
    /// # use blade_graphics as bg;
    /// # use meganeura::{ExternalSlot, Graph, build_inference_session};
    /// # fn doc(producer: &bg::Context, shared_handle: bg::Buffer, size: u64) {
    /// # let mut g = Graph::new();
    /// # let _ = g.input("x", &[4, 3]);
    /// let handle = producer
    ///     .get_external_buffer_source(shared_handle)
    ///     .expect("buffer was allocated with Memory::External");
    ///
    /// let mut session = build_inference_session(&g); // its own context
    /// session
    ///     .bind_external_buffer(ExternalSlot::Input("x"), handle, size)
    ///     .unwrap();
    /// // session.step() reads x from the shared memory
    /// # }
    /// ```
    pub fn bind_external_buffer(
        &mut self,
        slot: ExternalSlot<'_>,
        source: blade_graphics::ExternalMemorySource,
        size: u64,
    ) -> Result<(), ExternalBindError> {
        let buf_ref = self.resolve_slot(slot)?;
        let required = self.plan.buffers[buf_ref.0 as usize] as u64;
        if size < required {
            return Err(ExternalBindError::TooSmall {
                required,
                got: size,
            });
        }

        // Finish any outstanding GPU work that might still reference the
        // internal buffer, then release it before swapping in the import.
        self.wait();

        let imported = self.gpu.create_buffer(blade_graphics::BufferDesc {
            name: "meganeura_imported",
            size,
            memory: blade_graphics::Memory::External(source),
        });

        let device_local = !blade_graphics::Memory::External(source).is_host_visible();
        let physical = Arc::new(PhysicalBuffer {
            gpu: Arc::clone(&self.gpu),
            handle: imported,
        });
        self.rebind_logical(buf_ref.0 as usize, physical, 0, device_local);
        Ok(())
    }

    /// Make this session's named parameter use the same allocation as a
    /// parameter in another idle session on the same GPU context.
    ///
    /// This is useful when one model needs separately compiled execution
    /// shapes (for example, batched prefill and single-row decode). The
    /// allocation is reference-counted and remains alive until every sharing
    /// session is dropped. Both sessions observe writes to the parameter, so
    /// callers must serialize their execution when the parameter is mutable.
    pub fn share_parameter_from(
        &mut self,
        source: &mut Session,
        name: &str,
    ) -> Result<(), ShareParameterError> {
        if !Arc::ptr_eq(&self.gpu, &source.gpu) {
            return Err(ShareParameterError::DifferentContext);
        }
        self.wait();
        source.wait();
        let target = self
            .plan
            .param_buffers
            .iter()
            .find(|entry| entry.0 == name)
            .map(|entry| entry.1)
            .ok_or(ShareParameterError::UnknownTarget)?;
        let source_buffer = source
            .plan
            .param_buffers
            .iter()
            .find(|entry| entry.0 == name)
            .map(|entry| entry.1)
            .ok_or(ShareParameterError::UnknownSource)?;
        let target_size = self.plan.buffers[target.0 as usize];
        let source_size = source.plan.buffers[source_buffer.0 as usize];
        if target_size != source_size {
            return Err(ShareParameterError::Incompatible {
                target: target_size,
                source: source_size,
            });
        }
        let source_physical = source.alias.map[source_buffer.0 as usize];
        self.rebind_logical(
            target.0 as usize,
            Arc::clone(&source.physical_buffers[source_physical]),
            source.alias.offsets[source_buffer.0 as usize],
            source.alias.device_local[source_physical],
        );
        Ok(())
    }

    /// Back logical buffer `index` with `physical` at byte `offset`. Its old
    /// allocation is replaced when it was the only tenant, and otherwise
    /// kept for the others (an arena, for instance); the optimizer then
    /// updates this parameter on its own.
    fn rebind_logical(
        &mut self,
        index: usize,
        physical: Arc<PhysicalBuffer>,
        offset: usize,
        device_local: bool,
    ) {
        self.buffers[index] = physical.handle.at(offset as u64);
        let target = self
            .alias
            .rebind(index, self.plan.buffers[index], offset, device_local);
        if target == self.physical_buffers.len() {
            self.physical_buffers.push(physical);
        } else {
            self.physical_buffers[target] = physical;
        }
    }

    /// Whether this session's parameter `name` is the same storage as
    /// `other`'s: shared with [`Session::share_parameter_from`] or
    /// [`crate::SessionConfig::share_parameters_from`].
    pub fn shares_parameter(&self, other: &Session, name: &str) -> bool {
        let (Some(mine), Some(theirs)) = (self.param_buffer(name), other.param_buffer(name)) else {
            return false;
        };
        let (mine, theirs) = (mine.0 as usize, theirs.0 as usize);
        Arc::ptr_eq(
            &self.physical_buffers[self.alias.map[mine]],
            &other.physical_buffers[other.alias.map[theirs]],
        ) && self.alias.offsets[mine] == other.alias.offsets[theirs]
    }

    /// Whether parameter `name` and every parameter derived from it already
    /// hold `donor`'s storage, so an initializer need not upload it.
    ///
    /// [`crate::train::build_measured`] constructs each challenger on the
    /// incumbent's identically stored parameters that neither program writes.
    /// Writing such a parameter writes the incumbent's copy too; an
    /// initializer that does upload it must write the same values.
    pub fn inherits_parameter(&self, donor: &Session, name: &str) -> bool {
        // An upload also fills the fused weights derived from `name`. They are
        // named parameters too, and must be the donor's identical derivations.
        let inherits = |name: &str| {
            self.shares_parameter(donor, name)
                && self.param_buffer(name).is_some_and(|mine| {
                    donor.param_buffer(name).is_some_and(|theirs| {
                        parameter_storage_matches(&self.plan, mine, &donor.plan, theirs)
                    })
                })
        };
        inherits(name)
            && self
                .plan
                .derived_params
                .iter()
                .filter(|derived| derived.1.iter().any(|source| source.0 == name))
                .all(|derived| {
                    self.plan
                        .param_buffers
                        .iter()
                        .find(|entry| entry.1 == derived.0)
                        .is_some_and(|entry| inherits(&entry.0))
                })
    }

    /// Size in bytes of the GPU buffer backing the given slot.
    ///
    /// Useful for sizing a shared allocation on the producer side
    /// before calling [`Session::bind_external_buffer`]. Returns `None`
    /// if the slot is not known to the graph.
    pub fn slot_size(&self, slot: ExternalSlot<'_>) -> Option<usize> {
        let buf_ref = self.resolve_slot(slot).ok()?;
        Some(self.plan.buffers[buf_ref.0 as usize])
    }

    fn resolve_slot(&self, slot: ExternalSlot<'_>) -> Result<BufferRef, ExternalBindError> {
        match slot {
            ExternalSlot::Input(name) => self
                .plan
                .input_buffers
                .iter()
                .find(|e| e.0 == name)
                .map(|e| e.1)
                .ok_or(ExternalBindError::UnknownSlot),
            ExternalSlot::Parameter(name) => self
                .plan
                .param_buffers
                .iter()
                .find(|e| e.0 == name)
                .map(|e| e.1)
                .ok_or(ExternalBindError::UnknownSlot),
            ExternalSlot::Output(idx) => self
                .plan
                .output_buffers
                .get(idx)
                .copied()
                .ok_or(ExternalBindError::UnknownSlot),
        }
    }

    /// Set an upper bound on the submissions used by `step()`.
    ///
    /// By default the whole plan goes into a single command buffer. When the
    /// queue is shared with another workload, such as a renderer, splitting
    /// the plan lets that work interleave between submissions without
    /// dividing the model into independently compiled graphs.
    ///
    /// This is a caller-selected scheduling policy, not an automatic tuner.
    /// Meganeura partitions the compiled plan's ordered barrier groups as
    /// evenly as possible by group count; it does not estimate their duration
    /// or observe the latency of other queue users. The requested value is an
    /// upper bound, since a short plan can produce fewer chunks. Profile the
    /// end-to-end workload on the target device: extra submissions have CPU
    /// and driver overhead and can make either workload slower.
    /// [`crate::train::build_measured`] can measure this choice on initialized inputs
    /// when optimizing this graph's latency without competing queue users.
    ///
    /// Correctness across the resulting submission boundaries does not need
    /// extra synchronization. Blade ends every command buffer with a
    /// conservative global memory barrier and opens each new one assuming an
    /// unknown prior producer, and a Vulkan pipeline barrier's scopes cover
    /// commands submitted to the same queue before and after it — not merely
    /// those in the same command buffer.
    ///
    /// Reallocates the command encoder so that no command buffer is re-recorded
    /// while still in flight. Call it once during setup rather than per step.
    /// `chunks` is clamped to at least 1; 1 restores the single-submission
    /// default.
    pub fn set_submission_chunks(&mut self, chunks: usize) {
        let chunks = chunks.max(1);
        if chunks == self.submission_chunks {
            return;
        }
        // The encoder rotates through a fixed ring of command buffers with no
        // internal wait, so recording chunk N+1 while chunk N-1 is still
        // executing would re-record a live buffer. One spare beyond the
        // number in flight keeps that from happening.
        self.wait();
        self.gpu.destroy_command_encoder(&mut self.encoder);
        self.encoder = self
            .gpu
            .create_command_encoder(blade_graphics::CommandEncoderDesc {
                name: "meganeura",
                buffer_count: chunks as u32 + 1,
                manual_barriers: false,
            });
        self.submission_chunks = chunks;
        log::info!("submission chunks: {chunks}");
    }

    /// Enable or disable per-dispatch GPU profiling.
    ///
    /// When enabled, `step()` runs one compute pass per dispatch with
    /// individual GPU timestamps. Call `wait()` and then
    /// `dump_gpu_timings()` to see per-pass timings from the profiled run.
    ///
    /// Blade writes at most [`blade_graphics::limits::PASS_COUNT`] timestamps
    /// per submission and silently drops the rest, so plans larger than that
    /// need [`Session::set_profiling_window`] to be measured in slices.
    pub fn set_profiling(&mut self, enabled: bool) {
        self.profile_window = enabled.then_some(0..self.plan.dispatches.len());
    }

    /// Timestamp only `window`, a range of plan dispatch indices.
    ///
    /// Each dispatch inside the window gets its own compute pass and
    /// timestamp. The dispatches outside it still execute — the step remains
    /// a complete, correct replay — but they collapse into one grouped pass
    /// on either side of the window, keeping the plan's barriers between
    /// barrier groups. That costs at most two of the submission's timestamp
    /// slots regardless of how many dispatches lie outside the window, so a
    /// plan with more dispatches than Blade's per-submission timestamp limit
    /// can be measured by replaying it once per window and stitching the
    /// results together. [`crate::profiler::capture_session_profile`] does
    /// exactly that; prefer it over driving windows by hand.
    ///
    /// `None` restores unprofiled execution. The window is clamped to the
    /// plan, so an over-long range simply times every remaining dispatch.
    pub fn set_profiling_window(&mut self, window: Option<std::ops::Range<usize>>) {
        self.profile_window = window;
    }

    /// Does this session's context collect GPU pass timestamps?
    pub(crate) fn gpu_timing(&self) -> bool {
        self.gpu_timing
    }

    /// The range of dispatch indices `step()` will timestamp individually,
    /// or `None` when it runs in ordinary grouped-pass mode.
    pub fn profiling_window(&self) -> Option<std::ops::Range<usize>> {
        self.profile_window.clone()
    }

    /// Copy the GPU pass timings most recently resolved by Blade.
    ///
    /// The values become available as soon as [`Session::wait`] completes;
    /// callers should normally use [`crate::profiler::capture_session_profile`]
    /// rather than managing timestamp collection directly.
    pub fn gpu_timings(&self) -> Vec<(String, std::time::Duration)> {
        self.last_gpu_timings
            .as_ref()
            .map(|timings| {
                timings
                    .pass_durations()
                    .map(|(name, duration)| (name.to_owned(), duration))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Per-dispatch timings from the most recent profiled [`Session::step`].
    ///
    /// Yields `(plan dispatch index, pass label, duration)` for every dispatch
    /// the active profiling window timestamped on its own, in pass order. The
    /// grouped passes carrying the dispatches outside the window are dropped:
    /// their timestamps cover many dispatches at once and attributing them to
    /// any single one would be a lie.
    ///
    /// Empty when the resolved pass count does not match what the step
    /// encoded, which is how runtime-appended optimizer, gradient-accumulation
    /// and gradient-clipping passes show up. Their timestamps would shift
    /// every following entry, so no attribution is preferable to a wrong one.
    ///
    pub fn profiled_dispatch_timings(&self) -> Vec<(usize, String, std::time::Duration)> {
        let timings = self.gpu_timings();
        if timings.len() != self.profiled_pass_map.len() {
            return Vec::new();
        }
        self.profiled_pass_map
            .iter()
            .zip(timings)
            .filter_map(|(dispatch, (label, duration))| {
                dispatch.map(|index| (index, label, duration))
            })
            .collect()
    }

    /// Stable descriptive key for the pipeline selected by each plan
    /// dispatch, in dispatch order.
    ///
    /// One key per internal pipeline variant, so the key names what will
    /// actually run: scalar, cooperative, small-tile, reduced-weight,
    /// fused RmsNorm, fused epilogue or prologue, attention width, or a
    /// generated pointwise/reduction kernel.
    pub fn dispatch_pipeline_keys(&self) -> Vec<String> {
        self.pipelines.selected.iter().map(Variant::label).collect()
    }

    /// Shared handle to the underlying Blade GPU context.
    ///
    /// Cheap to clone — the context stays alive as long as any `Arc`
    /// reference remains. Useful for wiring a renderer onto the same
    /// device after meganeura has created the context via [`Session::new`].
    pub fn context(&self) -> Arc<blade_graphics::Context> {
        self.gpu.clone()
    }

    /// Query GPU pipeline statistics for all compiled compute pipelines.
    ///
    /// Returns driver-reported statistics (register counts, spill counts,
    /// SIMD width, etc.) for each pipeline executable. The exact statistics
    /// depend on the GPU vendor and backend:
    /// - NVIDIA/Vulkan: register count, spill loads/stores, subgroup size
    /// - Metal: max threads per threadgroup, SIMD width, shared memory
    /// - Other/unsupported: empty Vec
    pub fn get_pipeline_statistics(
        &self,
    ) -> Vec<(String, Vec<blade_graphics::PipelineExecutableInfo>)> {
        let mut result = Vec::new();
        for (name, pipeline) in self.pipelines.all_pipelines() {
            let stats = self.gpu.get_pipeline_statistics(pipeline);
            if !stats.is_empty() {
                result.push((name.to_string(), stats));
            }
        }
        result
    }

    pub(crate) fn get_profile_pipeline_statistics(
        &self,
    ) -> Vec<(String, Vec<blade_graphics::PipelineExecutableInfo>)> {
        let mut result = Vec::new();
        for (name, pipeline) in self.pipelines.all_profile_pipelines() {
            let stats = self.gpu.get_pipeline_statistics(pipeline);
            if !stats.is_empty() {
                result.push((name, stats));
            }
        }
        result.sort_by(|a, b| a.0.cmp(&b.0));
        result
    }
}

// Adapter selection and context creation live in `runtime/context.rs`.
mod context;
pub use context::*;

/// A context to own when the caller did not bring one.
///
/// Deliberately no process-global cache: a session that asked for the
/// default context owns it, and a caller that wants sharing passes its
/// context through [`Session::with_context`] / [`SessionConfig::gpu`] —
/// nothing here decides that for them.
pub(crate) fn default_gpu_context() -> Arc<Gpu> {
    Arc::new(init_gpu_context().expect("failed to initialize blade GPU context"))
}

pub(crate) fn parse_device_id(value: &str) -> Option<u32> {
    let value = value.trim();
    value
        .strip_prefix("0x")
        .or_else(|| value.strip_prefix("0X"))
        .map_or_else(
            || value.parse().ok(),
            |hex| u32::from_str_radix(hex, 16).ok(),
        )
}

#[cfg(test)]
mod parameter_sharing_tests {
    use super::{BufferRef, ExecutionPlan, parameter_storage_matches};
    use crate::graph::{ParamTransform, TensorType};

    fn plan(buffer: BufferRef) -> ExecutionPlan {
        let mut plan = crate::compile::compile(&crate::Graph::new());
        plan.buffers = vec![32; 2];
        plan.param_buffers.push(("packed".into(), buffer));
        plan.param_types.insert(buffer, TensorType::f32(vec![2, 4]));
        plan.derived_params.push((
            buffer,
            vec![("left".into(), 2), ("right".into(), 2)],
            ParamTransform::HorizontalConcat,
        ));
        plan
    }

    #[test]
    fn derived_parameters_require_the_same_recipe() {
        let (target, source) = (BufferRef(0), BufferRef(1));
        let target_plan = plan(target);
        let source_plan = plan(source);
        // Buffer numbering does not affect the values stored in the parameter.
        assert!(parameter_storage_matches(
            &target_plan,
            target,
            &source_plan,
            source
        ));
        for change in 0..4 {
            let mut changed = source_plan.clone();
            match change {
                0 => changed.derived_params[0].2 = ParamTransform::VerticalConcat,
                1 => changed.derived_params[0].1.reverse(),
                2 => changed.derived_params[0].1[0].1 += 1,
                _ => changed.derived_params.clear(),
            }
            assert!(
                !parameter_storage_matches(&target_plan, target, &changed, source),
                "different recipe {change} must stay private"
            );
            assert!(!parameter_storage_matches(
                &changed,
                source,
                &target_plan,
                target
            ));
        }
    }

    #[test]
    fn ordinary_parameters_still_require_matching_storage() {
        let (target, source) = (BufferRef(0), BufferRef(1));
        let mut target_plan = plan(target);
        let mut source_plan = plan(source);
        target_plan.derived_params.clear();
        source_plan.derived_params.clear();
        assert!(parameter_storage_matches(
            &target_plan,
            target,
            &source_plan,
            source
        ));
        for change in 0..3 {
            let mut changed = source_plan.clone();
            match change {
                0 => changed.buffers[source.0 as usize] += 4,
                1 => {
                    changed
                        .param_types
                        .insert(source, TensorType::f32(vec![4, 2]));
                }
                _ => {
                    changed
                        .weight_buffers
                        .insert(source, (crate::compile::WeightFormat::F16, 2, 4));
                }
            }
            assert!(!parameter_storage_matches(
                &target_plan,
                target,
                &changed,
                source
            ));
        }
    }
}

#[cfg(test)]
mod variant_tests {
    use super::{
        Dispatch, HorizMatMulKind, Pipelines, ShaderEntry, Variant, epilogue_pipeline_key,
        epilogue_tile, select_variants,
    };

    fn relu_epilogue() -> crate::compile::MatMulEpilogue {
        crate::compile::MatMulEpilogue {
            dag: crate::schedule::PointwiseDAG {
                n_inputs: 1,
                ops: vec![
                    crate::schedule::Pw::LoadInput(0),
                    crate::schedule::Pw::Relu(0),
                ],
                output: 1,
            },
            inputs: Vec::new(),
        }
    }

    fn matmul(f: impl FnOnce(&mut Dispatch)) -> Variant {
        let mut d = Dispatch {
            shader: ShaderEntry::MatMul,
            ..Default::default()
        };
        f(&mut d);
        Pipelines::key(&d)
    }

    /// The order here is what makes a modifier win over a less specific
    /// one. Reordering it silently changes which kernel every affected
    /// dispatch runs, so it is pinned rather than left to review.
    #[test]
    fn modifiers_outrank_the_scalar_base() {
        assert_eq!(matmul(|_| {}), Variant::Scalar(ShaderEntry::MatMul));

        assert_eq!(
            matmul(|d| {
                d.kernel = crate::compile::Kernel::SmallTile;
                d.weight_format = crate::compile::WeightFormat::F16;
            }),
            Variant::WeightSmall(ShaderEntry::MatMul, crate::compile::WeightFormat::F16),
        );
        assert_eq!(
            matmul(|d| d.kernel = crate::compile::Kernel::Cooperative),
            Variant::Coop(ShaderEntry::MatMul, crate::compile::WeightFormat::F32),
        );
        assert_eq!(
            matmul(|d| {
                d.kernel = crate::compile::Kernel::Cooperative;
                d.weight_format = crate::compile::WeightFormat::F16;
            }),
            Variant::Coop(ShaderEntry::MatMul, crate::compile::WeightFormat::F16),
        );
    }

    #[test]
    fn cooperative_prologues_keep_weight_storage_in_the_key() {
        use crate::compile::{BufferRef, Kernel, MatMulPrologue, PrologueLoadKind, WeightFormat};

        let key = |format| {
            matmul(|dispatch| {
                dispatch.kernel = Kernel::Cooperative;
                dispatch.weight_format = format;
                dispatch.matmul_prologue = Some(MatMulPrologue {
                    factors: vec![
                        (BufferRef(3), PrologueLoadKind::PerRow),
                        (BufferRef(4), PrologueLoadKind::PerKCol),
                    ],
                });
            })
        };
        let (f32, f16) = (key(WeightFormat::F32), key(WeightFormat::F16));
        assert!(matches!(f32, Variant::CoopPrologue(..)));
        assert!(matches!(f16, Variant::CoopPrologue(..)));
        assert_ne!(f32, f16, "mixed-storage prologues must not reuse a shader");
        assert_ne!(f32.label(), f16.label());
    }

    /// A dispatch whose epilogue ops were folded into the matmul has no
    /// standalone dispatch left to run them, so it must not be allowed to
    /// fall back to the unfused kernel.
    #[test]
    fn epilogue_fusion_has_no_fallback() {
        let candidates = matmul(|d| {
            d.matmul_epilogue = Some(relu_epilogue());
        });
        assert!(matches!(candidates, Variant::Epilogue(..)));

        let coop = matmul(|d| {
            d.matmul_epilogue = Some(relu_epilogue());
            d.kernel = crate::compile::Kernel::Cooperative;
        });
        assert!(matches!(coop, Variant::CoopEpilogue(..)));

        let q4 = matmul(|d| {
            d.matmul_epilogue = Some(relu_epilogue());
            d.weight_format = crate::compile::WeightFormat::Q4;
        });
        assert!(matches!(q4, Variant::Epilogue(..)));
        assert_ne!(
            epilogue_pipeline_key(&{
                let mut d = Dispatch {
                    shader: ShaderEntry::MatMul,
                    matmul_epilogue: Some(relu_epilogue()),
                    ..Default::default()
                };
                d.weight_format = crate::compile::WeightFormat::Q4;
                d
            }),
            epilogue_pipeline_key(&Dispatch {
                shader: ShaderEntry::MatMul,
                matmul_epilogue: Some(relu_epilogue()),
                ..Default::default()
            }),
            "Q4 and F32 epilogue pipelines must not share a key"
        );
    }

    /// `select_variants` demotes low-occupancy matmuls to 32×32 tiles and
    /// recomputes `workgroups` for them. The epilogue pipeline has to be
    /// generated for the same geometry, so the tile belongs in the key.
    #[test]
    fn small_tile_epilogue_gets_its_own_pipeline() {
        let large = Dispatch {
            shader: ShaderEntry::MatMul,
            matmul_epilogue: Some(relu_epilogue()),
            ..Default::default()
        };
        let small = Dispatch {
            kernel: crate::compile::Kernel::SmallTile,
            ..large.clone()
        };
        assert_ne!(
            epilogue_pipeline_key(&small),
            epilogue_pipeline_key(&large),
            "32×32 and 64×64 epilogue pipelines must not share a key"
        );
        let mut keys = std::collections::HashSet::new();
        for tile_n in [32, 64] {
            for k_stage in [8, 16, 32] {
                for interleave_columns in [false, true] {
                    let shape = crate::codegen::ScalarMatmulShape {
                        tile_size: 64,
                        tile_n,
                        k_stage,
                        interleave_columns,
                        unroll_k: true,
                    };
                    let dispatch = Dispatch {
                        kernel: crate::compile::Kernel::ScalarMatmul(shape),
                        ..large.clone()
                    };
                    assert!(matches!(Pipelines::key(&dispatch), Variant::Epilogue(..)));
                    assert_eq!(epilogue_tile(&dispatch), shape.geometry());
                    assert!(keys.insert(Pipelines::key(&dispatch)), "{shape:?}");
                    let split = Dispatch {
                        kernel: crate::compile::Kernel::SplitMatmul { shape, splits: 4 },
                        matmul_epilogue: None,
                        ..dispatch
                    };
                    let half = Dispatch {
                        weight_format: crate::compile::WeightFormat::F16,
                        ..split.clone()
                    };
                    assert_ne!(Pipelines::key(&split), Pipelines::key(&half));
                }
            }
        }
    }

    /// An epilogue dispatch resolves to its epilogue pipeline and never
    /// selects `SmallTile` or `Weight`, so it must not be what pulls those
    /// into the compile set. A plain dispatch sharing the group still has
    /// to get them, which is why the guard is per-dispatch and not a
    /// group-wide veto.
    #[test]
    fn epilogue_dispatch_does_not_request_unreachable_variants() {
        let epilogue_small = matmul(|d| {
            d.matmul_epilogue = Some(relu_epilogue());
            d.kernel = crate::compile::Kernel::SmallTile;
            d.weight_format = crate::compile::WeightFormat::Q4;
        });
        assert!(matches!(epilogue_small, Variant::Epilogue(..)));

        // Removing the epilogue still preserves both tile and storage format.
        let plain_small = matmul(|d| {
            d.kernel = crate::compile::Kernel::SmallTile;
            d.weight_format = crate::compile::WeightFormat::Q4;
        });
        assert_eq!(
            plain_small,
            Variant::WeightSmall(ShaderEntry::MatMul, crate::compile::WeightFormat::Q4),
        );
    }

    /// Sessions on one context share the pipelines of identical generated
    /// code, and the last of them destroys each pipeline. Another context
    /// compiles its own.
    #[test]
    fn sessions_share_pipelines_on_one_context() {
        use std::sync::Arc;
        let plan = {
            let mut g = crate::graph::Graph::new();
            let x = g.input("x", &[8, 64]);
            let w = g.parameter("w", &[64, 32]);
            let mm = g.matmul(x, w);
            let out = g.relu(mm);
            g.set_outputs(vec![out]);
            crate::compile::compile(&g)
        };
        let gpu = Arc::new(crate::init_gpu_context().unwrap());
        let first = super::Session::with_context(plan.clone(), gpu.clone());
        let second = super::Session::with_context(plan.clone(), gpu.clone());
        assert!(!first.pipelines.map.is_empty());
        for (variant, pipeline) in &first.pipelines.map {
            assert!(Arc::ptr_eq(pipeline, &second.pipelines.map[variant]));
        }
        let pipelines: Vec<_> = first.pipelines.map.values().map(Arc::downgrade).collect();
        drop(first);
        assert!(pipelines.iter().all(|p| p.strong_count() == 1));
        drop(second);
        assert!(pipelines.iter().all(|p| p.strong_count() == 0));
        let other = super::Session::with_context(
            plan.clone(),
            Arc::new(crate::init_gpu_context().unwrap()),
        );
        let third = super::Session::with_context(plan, gpu);
        for (variant, pipeline) in &other.pipelines.map {
            assert!(!Arc::ptr_eq(pipeline, &third.pipelines.map[variant]));
        }
    }

    /// Preparation and execution use the same key. Check this through
    /// `Pipelines::new`: a variant absent from the map is a kernel that was
    /// never compiled.
    #[test]
    fn epilogue_dispatch_compiles_no_unreachable_pipeline() {
        let gpu = std::sync::Arc::new(crate::init_gpu_context().unwrap());

        // 64×64 f32 matmul: demoted to 32×32 by the occupancy pass.
        let mut demoted = {
            let mut g = crate::graph::Graph::new();
            let x = g.input("x", &[64, 64]);
            let w = g.parameter("w", &[64, 64]);
            let mm = g.matmul(x, w);
            let out = g.relu(mm);
            g.set_outputs(vec![out]);
            crate::compile::compile(&g)
        };
        select_variants(&mut demoted, None, false, false);
        assert!(demoted.dispatches[0].use_small_tiles());
        let pipelines = Pipelines::new(&gpu, &demoted, None, None);
        assert!(
            pipelines
                .map
                .keys()
                .any(|v| matches!(v, Variant::Epilogue(..))),
            "the epilogue pipeline itself must be compiled"
        );
        assert!(
            !pipelines
                .map
                .contains_key(&Variant::SmallTile(ShaderEntry::MatMul)),
            "no dispatch can select SmallTile here, so it must not be built"
        );
        drop(pipelines);

        // Q4 matmul + relu: reduced-storage weights, now fused.
        let mut weighted = {
            let mut g = crate::graph::Graph::new();
            let x = g.input("x", &[8, 64]);
            let w = g.parameter_q4("w", &[64, 128]);
            let mm = g.matmul(x, w);
            let out = g.relu(mm);
            g.set_outputs(vec![out]);
            crate::compile::compile(&g)
        };
        select_variants(&mut weighted, None, false, false);
        assert_eq!(
            weighted.dispatches[0].weight_format,
            crate::compile::WeightFormat::Q4
        );
        let pipelines = Pipelines::new(&gpu, &weighted, None, None);
        assert!(
            !pipelines.map.contains_key(&Variant::Weight(
                ShaderEntry::MatMul,
                crate::compile::WeightFormat::Q4
            )),
            "no dispatch can select the plain weighted kernel here"
        );
        drop(pipelines);
    }

    /// A fused epilogue must not keep a matmul on 64×64 geometry once the
    /// occupancy pass has demoted it: the workgroup count is rewritten for
    /// 32×32, and a 64×64 shader over that grid leaves three quarters of
    /// its workgroups writing nothing.
    #[test]
    fn small_tile_demotion_survives_epilogue_fusion() {
        let mut plan = {
            let mut g = crate::graph::Graph::new();
            let x = g.input("x", &[64, 64]);
            let w = g.parameter("w", &[64, 64]);
            let mm = g.matmul(x, w);
            let out = g.relu(mm);
            g.set_outputs(vec![out]);
            crate::compile::compile(&g)
        };
        assert!(
            plan.dispatches[0].matmul_epilogue.is_some(),
            "expected the relu to fuse into the matmul"
        );
        select_variants(&mut plan, None, false, false);
        let d = &plan.dispatches[0];
        assert!(d.use_small_tiles(), "64×64 matmul should demote to 32×32");
        assert_eq!(
            d.workgroups,
            [2, 2, 1],
            "workgroups must cover the 64×64 output in 32×32 tiles"
        );
        assert_eq!(
            epilogue_tile(d),
            crate::codegen::MatMulTile::Small,
            "the epilogue shader must be generated for the demoted tile"
        );

        let gpu = std::sync::Arc::new(crate::init_gpu_context().unwrap());
        let (m, k, n) = (35, 67, 69);
        let a: Vec<f32> = (0..m * k).map(|i| (i as f32 * 0.21).sin()).collect();
        let b: Vec<f32> = (0..k * n).map(|i| (i as f32 * 0.13).cos()).collect();
        let expected: Vec<f64> = (0..m * n)
            .map(|i| {
                (0..k)
                    .map(|j| f64::from(a[i / n * k + j]) * f64::from(b[j * n + i % n]))
                    .sum::<f64>()
                    .max(0.0)
            })
            .collect();
        for (tile_size, tile_n, k_stage, unroll_k) in
            [(64, 32, 8, true), (32, 64, 16, false), (16, 64, 32, true)]
        {
            let mut graph = crate::Graph::new();
            let x = graph.input("x", &[m, k]);
            let w = graph.parameter("w", &[k, n]);
            let product = graph.matmul(x, w);
            graph.nodes_mut()[product as usize].matmul_impl = Some(crate::graph::MatmulImpl {
                shape: crate::codegen::ScalarMatmulShape {
                    tile_size,
                    tile_n,
                    k_stage,
                    interleave_columns: true,
                    unroll_k,
                },
                splits: 1,
            });
            let y = graph.relu(product);
            graph.set_outputs(vec![y]);
            let plan = crate::compile::compile(&graph);
            assert_eq!(plan.dispatches.len(), 1);
            let mut session = crate::Session::with_context_opts(
                plan,
                gpu.clone(),
                crate::SessionOptions {
                    coop: crate::CoopPolicy::Disabled,
                    ..Default::default()
                },
            );
            session.set_input("x", &a);
            session.set_parameter("w", &b);
            session.step();
            session.wait();
            for (actual, expected) in session.read_output(m * n).into_iter().zip(&expected) {
                assert!(
                    actual.is_finite()
                        && (f64::from(actual) - expected).abs() < 2e-5 + 2e-4 * expected.abs(),
                    "{actual} != {expected}"
                );
            }
        }
    }

    #[test]
    fn horizontal_pack_keys_include_precision() {
        let scalar = matmul(|d| d.horizontal_batch = 3);
        let coop = matmul(|d| {
            d.horizontal_batch = 3;
            d.kernel = crate::compile::Kernel::Cooperative;
        });
        let compensated = matmul(|d| {
            d.horizontal_batch = 3;
            d.kernel = crate::compile::Kernel::CooperativeCompensated;
        });
        assert_eq!(
            scalar,
            Variant::Horizontal(ShaderEntry::MatMul, 3, HorizMatMulKind::Scalar)
        );
        assert_ne!(scalar, coop);
        assert_ne!(coop, compensated);
        assert_eq!(
            compensated,
            Variant::Horizontal(ShaderEntry::MatMul, 3, HorizMatMulKind::CoopCompensated)
        );
    }
}

#[cfg(test)]
mod split_k_tests {
    use super::{BufferRef, Dispatch, ShaderEntry};

    #[test]
    fn split_sequences_get_barriers_and_reuse_nonoverlapping_partials() {
        let mut plan = crate::compile::compile(&crate::Graph::new());
        let bytes = 33 * 33 * 4;
        plan.buffers = vec![bytes; 4];
        plan.input_buffers = vec![
            ("upstream".into(), BufferRef(0)),
            ("input".into(), BufferRef(1)),
        ];
        plan.output_buffers = vec![BufferRef(2), BufferRef(3)];
        let first = Dispatch {
            shader: ShaderEntry::Conv2dGradWeightGemmSmall,
            workgroups: [2, 2, 1],
            input_buffers: vec![BufferRef(0), BufferRef(1)],
            output_buffer: BufferRef(2),
            params: vec![1, 33, 1, 33, 33, 1, 1, 1, 0, 1, 33, 0],
            requires_full_precision: true,
            ..Default::default()
        };
        let mut second = first.clone();
        second.input_buffers[1] = BufferRef(2);
        second.output_buffer = BufferRef(3);
        plan.dispatches = vec![first, second];
        assert_eq!(
            plan.split_conv_weight_gradients(&[(0, 2), (1, 2)], bytes * 4),
            Ok(bytes * 4)
        );
        crate::compile::schedule_dispatches(&mut plan, false, false);
        let groups = plan.groups.clone();
        assert_eq!(groups, [0..1, 1..2, 2..3, 3..4]);
        let alias = crate::memplan::plan_buffer_aliasing(&plan, &groups, None);
        assert_eq!(alias.map[4], alias.map[5]);
        assert_eq!(alias.sizes[alias.map[4]], bytes * 2);
        assert!(alias.device_local[alias.map[4]]);
        for i in 0..4 {
            assert_ne!(alias.map[i], alias.map[4]);
        }
        let no_alias = crate::memplan::plan_no_alias(&plan, &groups, None);
        assert_eq!(
            no_alias.physical_bytes() - alias.physical_bytes(),
            bytes * 2
        );
    }
}

#[cfg(test)]
mod profile_pass_tests {
    use super::{ProfilePass, profile_pass_plan};

    /// Six barrier groups over sixteen dispatches, including two groups of
    /// one so that windows can land on a group boundary and inside a group.
    fn groups() -> Vec<std::ops::Range<usize>> {
        vec![0..3, 3..4, 4..9, 9..10, 10..14, 14..16]
    }

    /// Whatever the window, the plan must still run every dispatch exactly
    /// once and in order, and must not spend more than two passes on the
    /// dispatches outside the window.
    #[test]
    fn every_window_replays_the_whole_plan_in_order() {
        let groups = groups();
        for start in 0..=16 {
            for end in start..=16 {
                let passes = profile_pass_plan(&groups, 16, start..end);
                let mut order = Vec::new();
                let mut timed = Vec::new();
                let mut untimed = 0;
                for pass in passes {
                    match pass {
                        ProfilePass::Timed(index) => {
                            order.push(index);
                            timed.push(index);
                        }
                        ProfilePass::Untimed(spans) => {
                            untimed += 1;
                            order.extend(spans.into_iter().flatten());
                        }
                    }
                }
                assert_eq!(
                    order,
                    (0..16).collect::<Vec<_>>(),
                    "window {start}..{end} did not replay the plan in order"
                );
                assert_eq!(
                    timed,
                    (start..end).collect::<Vec<_>>(),
                    "window {start}..{end} timed the wrong dispatches"
                );
                assert!(
                    untimed <= 2,
                    "window {start}..{end} spent {untimed} passes outside the window"
                );
            }
        }
    }

    #[test]
    fn boundary_plans_preserve_groups_and_clamp_ranges() {
        let passes = profile_pass_plan(&groups(), 16, 5..11);
        assert_eq!(
            passes.first(),
            Some(&ProfilePass::Untimed(vec![0..3, 3..4, 4..5])),
            "the head must keep its group seams and stop at the window"
        );
        assert_eq!(
            passes.last(),
            Some(&ProfilePass::Untimed(vec![11..14, 14..16])),
            "the tail must resume mid-group and keep the remaining seams"
        );
        assert!(
            profile_pass_plan(&groups(), 16, 0..99)
                .iter()
                .all(|pass| matches!(pass, ProfilePass::Timed(_)))
        );
        assert_eq!(
            profile_pass_plan(&groups(), 16, 99..99),
            vec![ProfilePass::Untimed(vec![
                0..3,
                3..4,
                4..9,
                9..10,
                10..14,
                14..16
            ])]
        );
        let inverted = std::ops::Range { start: 9, end: 4 };
        assert_eq!(
            profile_pass_plan(&groups(), 16, inverted),
            vec![
                ProfilePass::Untimed(vec![0..3, 3..4, 4..9]),
                ProfilePass::Untimed(vec![9..10, 10..14, 14..16]),
            ]
        );
        assert!(profile_pass_plan(&[], 0, 0..0).is_empty());
    }
}

#[cfg(test)]
mod device_id_tests {
    use super::parse_device_id;

    #[test]
    fn parses_decimal_and_hex_device_ids() {
        assert_eq!(parse_device_id("29772"), Some(0x744c));
        assert_eq!(parse_device_id("0x744c"), Some(0x744c));
        assert_eq!(parse_device_id("  0X744C  "), Some(0x744c));
        assert_eq!(parse_device_id("not-a-device"), None);
        assert_eq!(parse_device_id("0x"), None);
    }
}

#[cfg(test)]
mod device_memory_budget_tests {
    use super::safe_device_memory_remaining;

    #[test]
    fn reserves_ten_percent_of_the_reported_budget() {
        assert_eq!(safe_device_memory_remaining(100, 1_000), 800);
        assert_eq!(safe_device_memory_remaining(900, 1_000), 0);
        assert_eq!(safe_device_memory_remaining(950, 1_000), 0);
    }

    #[test]
    fn budget_math_does_not_overflow() {
        let safe_limit = ((u64::MAX as u128 * 9) / 10) as u64;
        assert_eq!(safe_device_memory_remaining(0, u64::MAX), safe_limit);
        assert_eq!(safe_device_memory_remaining(safe_limit - 1, u64::MAX), 1);
    }
}

/// Quantize f32 data to Q4_1 format (asymmetric 4-bit, 32-element blocks).
///
/// The data represents a matrix of shape `[rows, cols]` stored row-major.
/// Q4 blocks are formed column-wise: 32 consecutive elements along the row
/// dimension within each column, matching the matmul KTILE=32.
///
/// Per block: d = (max - min) / 15, m = min. Nibble = round((val - m) / d).
/// Returns packed buffer: `[d_f16|m_f16 per block][nibble data]`.
// Host-side Q4/Q8 packing lives in `runtime/quantize.rs`.
mod quantize;
pub use quantize::*;

/// Scatter one source's packed columns into a HorizontalConcat destination.
///
/// Block formats pack along K per column, so a source occupies a column
/// range. Q4 splits headers and nibbles into two regions, so that range is
/// not a single byte span; Q6_K's word padding belongs at the end of the
/// concatenated superblocks, not after each source.
fn scatter_packed_concat_columns(
    dest: &mut [u8],
    src: &[u8],
    fmt: crate::compile::WeightFormat,
    rows: usize,
    src_cols: usize,
    total_cols: usize,
    col_offset: usize,
) {
    assert!(col_offset + src_cols <= total_cols);
    match fmt {
        crate::compile::WeightFormat::F32 | crate::compile::WeightFormat::F16 => {
            let e = if fmt == crate::compile::WeightFormat::F32 {
                4
            } else {
                2
            };
            assert_eq!(src.len(), rows * src_cols * e);
            assert_eq!(dest.len(), rows * total_cols * e);
            for r in 0..rows {
                let s = r * src_cols * e;
                let d = (r * total_cols + col_offset) * e;
                dest[d..d + src_cols * e].copy_from_slice(&src[s..s + src_cols * e]);
            }
        }
        crate::compile::WeightFormat::Q8 => {
            assert!(rows.is_multiple_of(32));
            let bpc = rows / 32;
            let nbytes = bpc * src_cols * 36;
            assert_eq!(src.len(), nbytes);
            let off = col_offset * bpc * 36;
            dest[off..off + nbytes].copy_from_slice(src);
        }
        // The native K-quants differ only in superblock stride. Each
        // source arrives padded to a word, but that tail belongs at the
        // end of the whole parameter, not between two sources' blocks —
        // so only the unpadded span is copied. Q4_K (144) and Q5_K (176)
        // have no tail to strip; Q4_0 (18), Q6_K (210) and Q3_K (110) do.
        fmt @ (crate::compile::WeightFormat::Q40
        | crate::compile::WeightFormat::Q4K
        | crate::compile::WeightFormat::Q5K
        | crate::compile::WeightFormat::Q6K
        | crate::compile::WeightFormat::Q3K) => {
            // Q4_0 blocks 32 elements where the K-quants take 256; the
            // copy is otherwise identical, so the block size joins the
            // stride rather than earning a second arm. The numbers come from
            // `DType::block_geometry`, which `TensorType::size_bytes` and the
            // `Graph::parameter_q*k` asserts also read, so the three cannot
            // drift apart.
            let (block, stride) = fmt
                .dtype()
                .and_then(|d| d.block_geometry())
                .expect("a block-quantized weight format has block geometry");
            assert!(rows.is_multiple_of(block));
            let bpc = rows / block;
            let unpadded = bpc * src_cols * stride;
            assert_eq!(
                src.len(),
                unpadded.next_multiple_of(4),
                "{fmt:?} source should be its superblocks padded to a word"
            );
            let off = col_offset * bpc * stride;
            dest[off..off + unpadded].copy_from_slice(&src[..unpadded]);
        }
        crate::compile::WeightFormat::Q4 => {
            assert!(rows.is_multiple_of(32));
            let bpc = rows / 32;
            let src_blocks = bpc * src_cols;
            let dest_blocks = bpc * total_cols;
            let src_hdr = src_blocks * 4;
            let src_nib = src_blocks * 16;
            assert_eq!(src.len(), src_hdr + src_nib);
            let dest_hdr = dest_blocks * 4;
            let col_blocks = col_offset * bpc;
            dest[col_blocks * 4..col_blocks * 4 + src_hdr].copy_from_slice(&src[..src_hdr]);
            dest[dest_hdr + col_blocks * 16..dest_hdr + col_blocks * 16 + src_nib]
                .copy_from_slice(&src[src_hdr..]);
        }
    }
}

#[cfg(test)]
mod q4_tests {
    use super::*;

    #[test]
    fn q4_roundtrip_identity() {
        // 32x1 column: values from -1.6 to 1.5 in steps of 0.1
        let rows = 32;
        let cols = 1;
        let data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();

        let packed = quantize_q4_0(&data, rows, cols);
        let decoded = dequantize_q4_0(&packed, rows, cols);

        for i in 0..32 {
            let err = (data[i] - decoded[i]).abs();
            assert!(
                err < 0.3,
                "Q4 roundtrip elem[{}]: orig={:.3}, decoded={:.3}, err={:.3}",
                i,
                data[i],
                decoded[i],
                err,
            );
        }
    }

    #[test]
    fn q4_roundtrip_matrix() {
        // 32x4 matrix
        let rows = 32;
        let cols = 4;
        let data: Vec<f32> = (0..rows * cols)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
            .collect();

        let packed = quantize_q4_0(&data, rows, cols);
        let decoded = dequantize_q4_0(&packed, rows, cols);

        let max_err = data
            .iter()
            .zip(decoded.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_err < 0.3,
            "Q4 roundtrip max error {:.4} exceeds 0.3",
            max_err,
        );
    }

    #[test]
    fn q4_packed_concat_matches_quantizing_the_wide_matrix() {
        let rows = 32;
        let left_cols = 2;
        let right_cols = 2;
        let total = left_cols + right_cols;
        let left: Vec<f32> = (0..rows * left_cols)
            .map(|i| ((i % 17) as f32 - 8.0) * 0.05)
            .collect();
        let right: Vec<f32> = (0..rows * right_cols)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.07)
            .collect();
        let mut wide = vec![0.0f32; rows * total];
        for r in 0..rows {
            for c in 0..left_cols {
                wide[r * total + c] = left[r * left_cols + c];
            }
            for c in 0..right_cols {
                wide[r * total + left_cols + c] = right[r * right_cols + c];
            }
        }
        let mut dest = vec![0u8; quantize_q4_0(&wide, rows, total).len()];
        scatter_packed_concat_columns(
            &mut dest,
            &quantize_q4_0(&left, rows, left_cols),
            crate::compile::WeightFormat::Q4,
            rows,
            left_cols,
            total,
            0,
        );
        scatter_packed_concat_columns(
            &mut dest,
            &quantize_q4_0(&right, rows, right_cols),
            crate::compile::WeightFormat::Q4,
            rows,
            right_cols,
            total,
            left_cols,
        );
        assert_eq!(dest, quantize_q4_0(&wide, rows, total));
    }

    #[test]
    fn q6k_packed_concat_strips_per_source_padding() {
        // One superblock is 210 bytes, so each N=1 source pads to 212.
        // Concatenating the padded blobs would insert two zeros in the
        // middle of the superblock stream.
        let left = vec![1u8; 210];
        let right = vec![2u8; 210];
        let pad = |src: &[u8]| {
            let mut v = src.to_vec();
            v.resize(src.len().next_multiple_of(4), 0);
            v
        };
        let mut dest = vec![0u8; 420];
        scatter_packed_concat_columns(
            &mut dest,
            &pad(&left),
            crate::compile::WeightFormat::Q6K,
            256,
            1,
            2,
            0,
        );
        scatter_packed_concat_columns(
            &mut dest,
            &pad(&right),
            crate::compile::WeightFormat::Q6K,
            256,
            1,
            2,
            1,
        );
        assert_eq!(&dest[..210], &left[..]);
        assert_eq!(&dest[210..], &right[..]);
    }

    /// Q3_K's stride is 110, so two `[256, 1]` sources pad to 112 each but
    /// the combined `[256, 2]` parameter is 220 bytes, not 224. Upload
    /// order must not matter, and replacing one source must leave the
    /// other intact.
    #[test]
    fn q3k_packed_concat_sizes_the_pair_without_inner_padding() {
        use crate::compile::WeightFormat;
        use crate::graph::{DType, TensorType};

        let combined = TensorType::new(vec![256, 2], DType::Q3K).size_bytes();
        assert_eq!(combined, 220, "two 110-byte superblocks, already a word");
        assert_eq!(
            TensorType::new(vec![256, 1], DType::Q3K).size_bytes(),
            112,
            "a lone superblock pads to a word"
        );

        let left = vec![1u8; 110];
        let right = vec![2u8; 110];
        let pad = |src: &[u8]| {
            let mut v = src.to_vec();
            v.resize(src.len().next_multiple_of(4), 0);
            v
        };

        // Right first, then left: order must not matter.
        let mut dest = vec![0u8; combined];
        for (src, col) in [(&right, 1usize), (&left, 0usize)] {
            scatter_packed_concat_columns(&mut dest, &pad(src), WeightFormat::Q3K, 256, 1, 2, col);
        }
        assert_eq!(&dest[..110], &left[..]);
        assert_eq!(&dest[110..], &right[..]);

        // Replacing one source leaves the other alone.
        let replacement = vec![3u8; 110];
        scatter_packed_concat_columns(
            &mut dest,
            &pad(&replacement),
            WeightFormat::Q3K,
            256,
            1,
            2,
            0,
        );
        assert_eq!(&dest[..110], &replacement[..]);
        assert_eq!(&dest[110..], &right[..], "the other column must survive");
    }

    /// Q5_K's 176-byte stride is already a word, so its sources carry no
    /// tail and concatenate seamlessly.
    #[test]
    fn q5k_packed_concat_has_no_padding_seam() {
        use crate::compile::WeightFormat;
        use crate::graph::{DType, TensorType};

        assert_eq!(TensorType::new(vec![256, 2], DType::Q5K).size_bytes(), 352);
        let left = vec![4u8; 176];
        let right = vec![5u8; 176];
        let mut dest = vec![0u8; 352];
        for (src, col) in [(&left, 0usize), (&right, 1usize)] {
            scatter_packed_concat_columns(&mut dest, src, WeightFormat::Q5K, 256, 1, 2, col);
        }
        assert_eq!(&dest[..176], &left[..]);
        assert_eq!(&dest[176..], &right[..]);
    }
}

/// Result of [`auto_tune`] — capability snapshot from the connected GPU,
/// suitable for reporting or as the capabilities argument of a
/// capabilities-taking compile call.
#[derive(Clone, Debug)]
pub struct AutoTuneResult {
    /// Cooperative-matrix capabilities. Compile-time decisions
    /// (e.g. flash-attention-coop forward) gate on this.
    pub coop_caps: crate::codegen::CoopCaps,
}

/// Probe the GPU for cooperative_matrix capabilities. The result is a
/// capability description the caller compiles for or reports; there is
/// no shared global to install — everything that reaches the compiler
/// or the pipeline layer takes it as a parameter.
pub fn auto_tune(gpu: &blade_graphics::Context, _head_dim: u32) -> AutoTuneResult {
    let cm = gpu.capabilities().cooperative_matrix;
    let square = |shapes: &[[u32; 3]]| {
        [8, 16]
            .into_iter()
            .find(|&tile| shapes.contains(&[tile; 3]))
            .unwrap_or(0)
    };
    AutoTuneResult {
        coop_caps: crate::codegen::CoopCaps {
            f16_tile: square(&cm.f16_f32_shapes),
            f32_tile: square(&cm.f32_shapes),
        },
    }
}

// Host-side transfers — parameters, inputs, reads, dumps — live in
// `runtime/transfer.rs`. The methods are `pub(super)` because they are
// still `Session` methods; only the grouping is separate.
mod transfer;

// Turning a dispatch into bound shader data lives in `runtime/binding.rs`.
mod binding;

// The execution half of `Session` continues here: waiting, stepping,
// recording, dispatch binding and the optimizer entry points.
impl Session {
    /// Wait for any pending GPU work, including a caller submission reported
    /// through [`Session::track_submission`].
    pub fn wait(&mut self) {
        if let Some(sp) = self.external_sync_point.take() {
            let _span = tracing::info_span!("wait_external").entered();
            let _ = self.gpu.wait_for(&sp, !0);
        }
        if let Some(sp) = self.sync_point.take() {
            let _span = tracing::info_span!("wait").entered();
            self.last_gpu_timings =
                wait_for_timed_encoder(&self.gpu, &sp, &mut self.encoder, self.gpu_timing)
                    .ok()
                    .flatten();
        }
    }

    /// Execute the full dispatch sequence (forward + backward + update).
    pub fn step(&mut self) {
        let _span = tracing::info_span!(
            "step",
            dispatches = self.plan.dispatches.len(),
            groups = self.groups.len(),
            chunks = self.submission_chunks,
        )
        .entered();
        self.wait();

        self.encoder.start();
        self.profiled_pass_map.clear();
        self.encode_step(self.submission_chunks, self.profile_window.clone());
        self.sync_point = Some(self.gpu.submit(&mut self.encoder));
    }

    /// Record one step — the same work as [`Session::step`] — into a
    /// caller's command encoder instead of submitting it, so a host
    /// application can put inference or training in its own submissions
    /// next to its other passes (rendering, simulation, video processing).
    ///
    /// The encoder must belong to this session's context (build the session
    /// with [`crate::SessionConfig::gpu`] or [`Session::with_context`] set to
    /// the application's context) and be started. Nothing is submitted: the
    /// work runs when the caller submits the encoder.
    ///
    /// The encoder must use automatic barriers (`manual_barriers: false`).
    /// Blade then orders every pass after the ones recorded before it, which
    /// is what orders the step's passes among themselves and against the
    /// caller's passes on either side.
    ///
    /// # Data flow
    ///
    /// [`Session::input_buffer`] and [`Session::output_buffer`] name the
    /// session's storage, so the caller's own passes can write inputs and
    /// consume outputs in the same encoder, with no host round trip. Parameter updates of a training
    /// session happen on the GPU as part of the recorded step.
    ///
    /// # Synchronization
    ///
    /// The session cannot see the caller's submission. Host-side access that
    /// needs the GPU idle — [`Session::set_input`], readbacks,
    /// [`Session::wait`], dropping the session — only waits for it after
    /// [`Session::track_submission`] reports the returned sync point.
    /// Without that, the caller must wait before such calls. Recording
    /// several steps before a submission, or recording while earlier
    /// recorded work is in flight, is fine; the recorded steps execute in
    /// queue order.
    ///
    /// Settings baked into the recording (learning rate, Adam step count)
    /// are those current at the call. A parameter layout change, such as
    /// [`Session::share_parameter_from`], rewrites optimizer metadata on the
    /// host at the next recording and requires the GPU to be idle.
    ///
    /// # Limitations
    ///
    /// - Temporal gradient accumulation
    ///   ([`Session::set_grad_accumulate`]) needs a submission boundary
    ///   inside the step and is rejected with
    ///   [`RecordError::GradAccumulation`].
    /// - [`Session::set_submission_chunks`] and the profiling window apply to
    ///   [`Session::step`] only; the caller decides how to split submissions.
    pub fn record(
        &mut self,
        encoder: &mut blade_graphics::CommandEncoder,
    ) -> Result<(), RecordError> {
        if self.grad_accum_scale.is_some() {
            return Err(RecordError::GradAccumulation);
        }
        let _span = tracing::info_span!(
            "record",
            dispatches = self.plan.dispatches.len(),
            groups = self.groups.len(),
        )
        .entered();
        // The recording code writes into `self.encoder`. Lending the
        // caller's encoder in its place keeps a single code path for both.
        std::mem::swap(&mut self.encoder, encoder);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.encode_step(1, None);
        }));
        std::mem::swap(&mut self.encoder, encoder);
        if let Err(panic) = result {
            std::panic::resume_unwind(panic);
        }
        Ok(())
    }

    /// Report the sync point of a caller submission that carries work
    /// recorded by [`Session::record`], so that [`Session::wait`], the
    /// host-side uploads and readbacks, and dropping the session wait for
    /// it. Only the latest submission needs reporting: submissions to one
    /// queue complete in order.
    pub fn track_submission(&mut self, sync_point: blade_graphics::SyncPoint) {
        self.external_sync_point = Some(sync_point);
    }

    /// Record forward, backward and the optimizer into `self.encoder`, which
    /// must be started. `chunks` above one submits all but the last chunk;
    /// with a `profile_window`, the dispatches in it get passes of their own.
    fn encode_step(&mut self, chunks: usize, profile_window: Option<std::ops::Range<usize>>) {
        if let Some(window) = profile_window {
            // Multi-pass mode: one compute pass per dispatch in the window,
            // with per-pass barriers and GPU timestamps. Enables
            // dump_gpu_timings() and profiled_dispatch_timings() after wait().
            let passes = profile_pass_plan(&self.groups, self.plan.dispatches.len(), window);
            for encoded in passes {
                match encoded {
                    ProfilePass::Timed(i) => {
                        let dispatch = &self.plan.dispatches[i];
                        let pipeline = self.pipelines.get(i);
                        let mut pass = self.encoder.compute(&dispatch.label);
                        let mut pc = pass.with(pipeline);
                        Self::bind_dispatch(&self.buffers, dispatch, &mut pc);
                        pc.dispatch(dispatch.workgroups);
                        self.profiled_pass_map.push(Some(i));
                    }
                    ProfilePass::Untimed(spans) => {
                        let label =
                            format!("untimed {}..{}", spans[0].start, spans[spans.len() - 1].end);
                        let mut pass = self.encoder.compute(&label);
                        for (position, span) in spans.into_iter().enumerate() {
                            if position > 0 {
                                pass.barrier();
                            }
                            for i in span {
                                let dispatch = &self.plan.dispatches[i];
                                let pipeline = self.pipelines.get(i);
                                let mut pc = pass.with(pipeline);
                                Self::bind_dispatch(&self.buffers, dispatch, &mut pc);
                                pc.dispatch(dispatch.workgroups);
                            }
                        }
                        self.profiled_pass_map.push(None);
                    }
                }
            }
        } else {
            // Inline-barrier mode: dispatches share one compute pass with
            // lightweight compute-to-compute barriers between groups. Avoids
            // the per-pass overhead of begin_pass/end_pass (timestamps, debug
            // labels) while maintaining correct memory ordering.
            //
            // With `set_submission_chunks`, the groups are spread over
            // several submissions so a co-tenant on the queue (a renderer,
            // typically) can slot its own work between them. The chunk
            // boundary needs no explicit barrier: blade closes each command
            // buffer with a conservative global one.
            record_groups(
                &self.gpu,
                &mut self.encoder,
                &mut self.sync_point,
                &self.plan,
                &self.groups,
                &self.pipelines,
                &self.buffers,
                chunks,
            );
        }

        // Gradient clipping runs after backward and before the optimizer in
        // the same GPU submission. Global clipping measures the concatenated
        // gradient; AGC measures each parameter and its gradient separately.
        let update = if let Some(lr) = self.pending_lr {
            Some(optimizer::Update::Sgd { lr })
        } else {
            self.pending_adam.map(
                |(lr, beta1, beta2, eps, algorithm)| optimizer::Update::Adam {
                    lr,
                    beta1,
                    beta2,
                    eps,
                    algorithm,
                },
            )
        };
        let needs_clip =
            (self.pending_grad_clip.is_some() || self.pending_agc.is_some()) && update.is_some();
        // Cadence: skip the clip on (every-1)/every fraction of steps
        // when grad_clip_every > 1. Default 1 = every step.
        let clip = needs_clip && {
            self.grad_clip_tick = self.grad_clip_tick.wrapping_add(1);
            self.grad_clip_tick.is_multiple_of(self.grad_clip_every)
        };
        self.encode_optimizer(update, clip);
    }

    fn optimizer_len(plan: &ExecutionPlan, param: BufferRef) -> u32 {
        let len = match plan.param_types.get(&param) {
            Some(ty) => {
                assert_eq!(
                    ty.dtype,
                    crate::graph::DType::F32,
                    "optimizer/clip/accumulation requires F32 trainable parameter storage"
                );
                ty.num_elements()
            }
            None => plan.buffers[param.0 as usize] / 4,
        };
        u32::try_from(len).expect("optimizer parameter exceeds u32 element limit")
    }

    // Dispatch binding lives in `runtime/binding.rs`. The methods are
    // `pub(super)` because they are still `Session` methods.
}

impl Session {
    /// Apply SGD updates to all parameters on the GPU.
    pub fn sgd_step(&mut self, learning_rate: f32) {
        let _span = tracing::info_span!("sgd_step").entered();
        self.run_optimizer(optimizer::Update::Sgd { lr: learning_rate });
    }

    /// Configure SGD updates to run after each `step()`.
    ///
    /// `step()` appends all SGD parameter updates to the same GPU
    /// submission as forward+backward — eliminating the submit/wait
    /// overhead of a separate `sgd_step()` call. Persistent: once set,
    /// every subsequent `step()` runs the SGD update at the current
    /// learning rate. Call again to change the rate, switch to Adam via
    /// [`set_adam`](Self::set_adam), or stop optimizer updates via
    /// [`clear_optimizer`](Self::clear_optimizer).
    pub fn set_learning_rate(&mut self, lr: f32) {
        self.pending_lr = Some(lr);
        // Switching optimizers: SGD wins, Adam state stops applying.
        self.pending_adam = None;
    }

    /// Enable global gradient-norm clipping for `step()`. When set,
    /// `step()` reads all `param_grad_pairs` to CPU after backward,
    /// computes the concatenated L2 norm, and scales every gradient
    /// buffer by `min(1, max_norm / total_norm)` before the SGD/Adam
    /// optimizer pass runs. Mirrors PyTorch's
    /// `clip_grad_norm_(parameters, max_norm)`.
    ///
    /// Cost: one extra CPU readback + upload of all gradient buffers
    /// per `step()` (~1-10 ms typical, depending on parameter count).
    /// Use this on tasks where Adam's per-parameter variance estimate
    /// drives the per-step update unbounded — sparse-reward visual
    /// training (Atari) without it eventually NaN-collapses.
    ///
    /// Pass `0.0` or call `disable_grad_clip()` to disable.
    /// Persistent: stays on across `step()` calls until cleared.
    pub fn set_grad_clip_norm(&mut self, max_norm: f32) {
        self.pending_grad_clip = if max_norm > 0.0 { Some(max_norm) } else { None };
        self.pending_agc = None;
    }

    /// Enable adaptive gradient clipping for every trainable parameter.
    ///
    /// Each gradient is scaled to at most
    /// `clip * max(min_param_norm, ||parameter||)`. This matches the
    /// per-parameter transformation used by DreamerV3.
    pub fn set_adaptive_grad_clip(&mut self, clip: f32, min_param_norm: f32) {
        assert!(
            clip >= 0.0 && clip.is_finite(),
            "AGC clip must be finite and non-negative"
        );
        assert!(
            min_param_norm >= 0.0 && min_param_norm.is_finite(),
            "AGC minimum parameter norm must be finite and non-negative"
        );
        self.pending_agc = if clip > 0.0 {
            Some((clip, min_param_norm))
        } else {
            None
        };
        self.pending_grad_clip = None;
    }

    /// Disable gradient clipping for subsequent `step()` calls.
    pub fn disable_grad_clip(&mut self) {
        self.pending_grad_clip = None;
        self.pending_agc = None;
    }

    /// Set clip cadence. `every=1` clips every step (PyTorch default).
    /// `every=5` clips every 5th step and amortizes the extra
    /// submit/readback overhead 5x. The NaN-collapse failure mode
    /// accumulates over thousands of steps, so values up to ~10 are
    /// usually fine; higher cadences trade away clipping precision.
    pub fn set_grad_clip_every(&mut self, every: u32) {
        self.grad_clip_every = every.max(1);
        self.grad_clip_tick = 0;
    }

    /// CPU-side gradient clipping helper. Reads all grad buffers,
    /// computes the L2 norm of the concatenation, and if it exceeds
    /// `max_norm`, scales every buffer in place by `max_norm / norm`.
    /// Returns `(pre_clip_norm, did_clip)`.
    ///
    /// Diagnostic/fallback helper; `step()` uses GPU clipping instead.
    pub fn clip_grad_norm_cpu(&mut self, max_norm: f32) -> (f32, bool) {
        if max_norm <= 0.0 {
            return (0.0, false);
        }
        self.wait();
        // Sum of squares across all gradient buffers.
        let mut total_sq: f64 = 0.0;
        let mut buffers: Vec<(BufferRef, Vec<f32>)> =
            Vec::with_capacity(self.plan.param_grad_pairs.len());
        for &(param_buf, grad_buf) in &self.plan.param_grad_pairs {
            let n = Self::optimizer_len(&self.plan, param_buf) as usize;
            let mut data = vec![0.0f32; n];
            self.read_buffer(grad_buf, &mut data);
            for &v in &data {
                if v.is_finite() {
                    total_sq += (v as f64) * (v as f64);
                }
            }
            buffers.push((grad_buf, data));
        }
        let total_norm = (total_sq as f32).sqrt();
        if !total_norm.is_finite() || total_norm <= max_norm {
            return (total_norm, false);
        }
        let scale = max_norm / total_norm;
        for (grad_buf, mut data) in buffers {
            for v in &mut data {
                *v *= scale;
            }
            self.write_raw_buffer(
                self.buffers[grad_buf.0 as usize],
                bytemuck::cast_slice(&data),
                self.logical_host_visible(grad_buf),
            );
        }
        (total_norm, true)
    }

    /// Set a per-parameter learning rate multiplier, keyed by name
    /// prefix. The effective LR for a parameter is `base_lr * mul`
    /// where `mul` is the multiplier of the LONGEST matching prefix
    /// (default 1.0 if no prefix matches). Applies to both SGD and
    /// Adam updates. Persists across calls; pass `1.0` to clear.
    ///
    /// Use case: a single shared session with mixed-rate training,
    /// e.g. boost the policy head's LR relative to the encoder when
    /// the encoder receives more gradient flow than is healthy. From
    /// kindle's gradient-inspection findings on LunarLander, the
    /// encoder's obs_proj receives 50-100× the gradient of the
    /// policy.fc layers; setting `set_lr_multiplier("policy.", 5.0)`
    /// rebalances this without splitting the session.
    pub fn set_lr_multiplier(&mut self, name_prefix: &str, mul: f32) {
        // Replace existing entry for this prefix, or push new.
        if let Some(pos) = self
            .lr_multipliers
            .iter()
            .position(|entry| entry.0 == name_prefix)
        {
            self.lr_multipliers[pos].1 = mul;
        } else {
            self.lr_multipliers.push((name_prefix.to_string(), mul));
        }
    }

    /// Clear all per-parameter LR multipliers (returns all params to
    /// the base learning rate).
    pub fn clear_lr_multipliers(&mut self) {
        self.lr_multipliers.clear();
    }

    /// Internal: longest-prefix-match lookup. Static helper so it can
    /// be called from inside the SGD/Adam loops without holding `&self`
    /// across the buffer borrow.
    fn lr_multiplier_for_buf(
        param_buffers: &[(String, BufferRef)],
        multipliers: &[(String, f32)],
        buf: BufferRef,
    ) -> f32 {
        if multipliers.is_empty() {
            return 1.0;
        }
        let Some(entry) = param_buffers.iter().find(|e| e.1 == buf) else {
            return 1.0;
        };
        let name = &entry.0;
        let mut best: (usize, f32) = (0, 1.0);
        for m in multipliers {
            let (prefix, mul) = (&m.0, m.1);
            if name.starts_with(prefix.as_str()) && prefix.len() >= best.0 {
                best = (prefix.len(), mul);
            }
        }
        best.1
    }

    /// CPU-fallback SGD update.
    pub fn sgd_step_cpu(&mut self, learning_rate: f32) {
        let _span = tracing::info_span!("sgd_step_cpu").entered();
        self.wait();
        for &(param_buf, grad_buf) in &self.plan.param_grad_pairs {
            let size = Self::optimizer_len(&self.plan, param_buf) as usize;
            let mut param = vec![0.0f32; size];
            let mut grad = vec![0.0f32; size];
            self.read_buffer(param_buf, &mut param);
            self.read_buffer(grad_buf, &mut grad);
            for i in 0..size {
                param[i] -= learning_rate * grad[i];
            }
            self.write_raw_buffer(
                self.buffers[param_buf.0 as usize],
                bytemuck::cast_slice(&param),
                self.logical_host_visible(param_buf),
            );
        }
    }

    /// Apply Adam optimizer updates to all parameters on the GPU.
    pub fn adam_step(&mut self, lr: f32, beta1: f32, beta2: f32, eps: f32) {
        let _span = tracing::info_span!("adam_step").entered();
        self.run_optimizer(optimizer::Update::Adam {
            lr,
            beta1,
            beta2,
            eps,
            algorithm: AdaptiveOptimizer::Adam,
        });
    }

    /// Configure Adam updates to run after each `step()`.
    ///
    /// Analogous to [`set_learning_rate`](Self::set_learning_rate) for
    /// SGD: persistent until reconfigured or cleared. Call again to
    /// change hyperparameters (LR schedule, warmup), switch to SGD via
    /// [`set_learning_rate`](Self::set_learning_rate), or stop optimizer
    /// updates via [`clear_optimizer`](Self::clear_optimizer).
    pub fn set_adam(&mut self, lr: f32, beta1: f32, beta2: f32, eps: f32) {
        self.ensure_adam_state();
        self.pending_adam = Some((lr, beta1, beta2, eps, AdaptiveOptimizer::Adam));
        // Switching optimizers: Adam wins, SGD stops applying.
        self.pending_lr = None;
    }

    /// Configure LaProp updates to run after each `step()`.
    ///
    /// LaProp updates the RMS estimate from the clipped raw gradient,
    /// normalizes that gradient, and only then accumulates momentum. This is
    /// the optimizer ordering used by DreamerV3. Its two state tensors are
    /// checkpoint-compatible with Adam's `m` and `v` buffers.
    pub fn set_laprop(&mut self, lr: f32, beta1: f32, beta2: f32, eps: f32) {
        self.ensure_adam_state();
        self.pending_adam = Some((lr, beta1, beta2, eps, AdaptiveOptimizer::LaProp));
        self.pending_lr = None;
    }

    /// Accumulate the L2 norm of each consecutive gradient group for one
    /// parameter on every Adam update.
    ///
    /// For a `[N, 3]` position parameter, `group_size = 3` produces `N`
    /// values containing the exact temporal sum of each point's gradient
    /// magnitude. Collection is folded into the existing Adam dispatch and
    /// adds no synchronization. Configuring another parameter replaces and
    /// zeroes the previous accumulator.
    pub fn set_adam_grouped_grad_norm(&mut self, name: &str, group_size: usize) {
        assert!(group_size > 0, "gradient group size must be non-zero");
        let group_size = u32::try_from(group_size).expect("gradient group size exceeds u32");
        let param_index = self
            .adam_state_index(name)
            .unwrap_or_else(|| panic!("no Adam state for param: {name}"));
        let param_len = self.param_size(name).expect("parameter exists");
        assert_eq!(
            param_len % group_size as usize,
            0,
            "parameter {name:?} has {param_len} elements, not divisible by group size {group_size}",
        );
        let len = param_len / group_size as usize;
        let byte_len = len * std::mem::size_of::<f32>();

        self.wait();
        if let Some(previous) = self.adam_grouped_grad_norm.take() {
            self.gpu.destroy_buffer(previous.buffer);
        }
        let buffer = self.gpu.create_buffer(blade_graphics::BufferDesc {
            name: "adam_grouped_grad_norm",
            size: (byte_len as u64).max(4),
            memory: blade_graphics::Memory::Shared,
        });
        unsafe {
            std::ptr::write_bytes(buffer.data(), 0, byte_len.max(4));
        }
        self.adam_grouped_grad_norm = Some(AdamGroupedGradNorm {
            param_index,
            group_size,
            len,
            buffer,
        });
    }

    /// Read the grouped gradient-norm totals configured by
    /// [`Session::set_adam_grouped_grad_norm`].
    pub fn read_adam_grouped_grad_norm(&self, name: &str) -> Vec<f32> {
        let accumulator = self
            .adam_grouped_grad_norm
            .as_ref()
            .expect("Adam grouped gradient norm is not configured");
        assert_eq!(
            self.adam_state_index(name),
            Some(accumulator.param_index),
            "Adam grouped gradient norm is configured for another parameter",
        );
        self.read_f32_buffers(
            &[(
                accumulator.buffer.at(0),
                accumulator.len * std::mem::size_of::<f32>(),
            )],
            "adam_grouped_grad_norm_readback",
        )
        .pop()
        .unwrap()
    }

    /// Set the decoupled weight-decay coefficient for the Adam update,
    /// turning it into AdamW: each step also applies `param -= lr * wd * param`,
    /// independent of the gradient. 0.0 (the default) is plain Adam. Persists
    /// across steps until changed; unaffected by [`Self::set_adam`].
    pub fn set_weight_decay(&mut self, wd: f32) {
        self.adam_wd = wd;
    }

    /// Stop running optimizer updates after `step()`.
    ///
    /// Forward+backward still runs (gradients are computed) but no SGD
    /// or Adam pass is appended. Useful when freezing parameters for
    /// evaluation or pinning the model after a warmup phase.
    /// Retains allocated moments and counters so reconfiguration can resume.
    pub fn clear_optimizer(&mut self) {
        self.pending_lr = None;
        self.pending_adam = None;
    }

    /// Enable temporal gradient accumulation over `micro_batches` steps.
    ///
    /// meganeura's static-graph backward *overwrites* each param's grad
    /// buffer every `step()` (it sums contributions within one backward,
    /// not across calls). With accumulation enabled, each `step()`
    /// instead adds `grad / micro_batches` into a persistent accumulator,
    /// and the optimizer (clip + Adam/SGD) reads that accumulator — so
    /// `step()` K times then one optimizer-bearing step trains on the
    /// mean gradient of K micro-batches at single-batch GPU memory.
    ///
    /// Call [`Session::zero_grad`] before each K-step window. Pass
    /// `micro_batches = 1` (or [`Session::clear_grad_accumulate`]) to
    /// restore direct single-step optimizer reads.
    pub fn set_grad_accumulate(&mut self, micro_batches: u32) {
        assert!(micro_batches >= 1);
        if micro_batches == 1 {
            self.grad_accum_scale = None;
            return;
        }
        if self.grad_accum.is_none() {
            ensure_device_memory_budget(
                &self.gpu,
                optimizer::ChunkBuffers::bytes(self),
                "gradient accumulation buffers",
            );
            let mut device_bufs = Vec::new();
            let accumulators = optimizer::ChunkBuffers::new(self, "grad_accum", &mut device_bufs);
            self.grad_accum = Some(accumulators);
            self.zero_optimizer_buffers(&device_bufs, "zero_grad_accum");
        }
        self.grad_accum_scale = Some(1.0 / micro_batches as f32);
    }

    pub fn clear_grad_accumulate(&mut self) {
        self.grad_accum_scale = None;
    }

    /// Zero the gradient accumulators (PyTorch `optimizer.zero_grad()`).
    /// No-op unless [`Session::set_grad_accumulate`] is active.
    pub fn zero_grad(&mut self) {
        if self.grad_accum.is_none() {
            return;
        }
        self.wait();
        let buffers: Vec<_> = self
            .grad_accum
            .as_ref()
            .expect("gradient accumulators")
            .buffers
            .iter()
            .zip(&self.optimizer_chunks)
            .map(|(&buffer, chunk)| (buffer, chunk.bytes as u64))
            .collect();
        if self.optimizer_device {
            self.zero_optimizer_buffers(&buffers, "zero_grad");
        } else {
            for (buffer, size) in buffers {
                unsafe {
                    std::ptr::write_bytes(buffer.data(), 0, size as usize);
                }
            }
        }
    }

    pub fn memory_summary(&self) -> MemorySummary {
        let total: usize = self.plan.buffers.iter().sum();
        let largest = self.plan.buffers.iter().copied().max().unwrap_or(0);
        let chunk_bytes = optimizer::ChunkBuffers::bytes(self);
        let adam_bytes = usize::from(self.adam_state.is_some()) * chunk_bytes * 2;
        let grad_accumulator_bytes = usize::from(self.grad_accum.is_some()) * chunk_bytes;
        let clip_partial_bytes = if self.grad_clip_partials.is_some() {
            optimizer::clip_slots(&self.plan) as usize * 8
        } else {
            0
        };
        let clip_partial_bytes = clip_partial_bytes
            + usize::from(self.agc_scales.is_some()) * self.plan.param_grad_pairs.len() * 4;
        let clip_bytes = usize::from(self.grad_clip_acc.is_some()) * 4 + clip_partial_bytes;
        let optimizer_aux_bytes = clip_bytes
            + self
                .adam_grouped_grad_norm
                .as_ref()
                .map_or(0, |state| (state.len * 4).max(4));
        MemorySummary {
            total_buffer_bytes: total,
            adam_state_bytes: adam_bytes,
            grad_accumulator_bytes,
            optimizer_aux_bytes,
            num_buffers: self.plan.buffers.len(),
            largest_buffer_bytes: largest,
            allocated_buffer_bytes: self.alias.sizes.iter().map(|&size| size.max(4)).sum(),
            num_allocations: self.alias.sizes.len(),
            device_local_bytes: self
                .alias
                .sizes
                .iter()
                .zip(&self.alias.device_local)
                .filter_map(|(&size, &device)| device.then_some(size.max(4)))
                .sum::<usize>()
                + if self.optimizer_device {
                    adam_bytes + grad_accumulator_bytes + clip_bytes
                } else {
                    0
                },
        }
    }

    /// Per-process GPU memory usage, or `None` when the backend cannot
    /// report it.
    ///
    /// Vulkan requires `VK_EXT_memory_budget`; a device without it reports
    /// a zero budget, which is returned as `None` rather than as zero bytes
    /// so that "unsupported" is never recorded as a measurement.
    pub fn device_memory_stats(&self) -> Option<DeviceMemoryStats> {
        let stats = self.gpu.memory_stats();
        // A supported query always reports a non-zero budget; usage alone
        // cannot distinguish "nothing allocated" from "not implemented".
        if stats.budget == 0 {
            return None;
        }
        Some(DeviceMemoryStats {
            usage_bytes: stats.usage,
            budget_bytes: stats.budget,
        })
    }

    pub fn plan(&self) -> &ExecutionPlan {
        &self.plan
    }

    /// Number of barrier groups (compute passes) in the dispatch sequence.
    pub fn num_groups(&self) -> usize {
        self.groups.len()
    }

    /// GPU device and driver name.
    pub fn device_information(&self) -> &blade_graphics::DeviceInformation {
        self.gpu.device_information()
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.wait();
        self.gpu.destroy_command_encoder(&mut self.encoder);
        if let Some(staging) = self.upload_staging.get_mut().take() {
            self.gpu.destroy_buffer(staging.buffer);
        }
        if let Some(mut encoder) = self.readback.get_mut().encoder.take() {
            self.gpu.destroy_command_encoder(&mut encoder);
        }
        if let Some(staging) = self.readback.get_mut().staging.take() {
            self.gpu.destroy_buffer(staging.buffer);
        }
        self.pipelines.map.clear();
        // `buffers` holds aliased copies of these handles; destroy each
        // physical allocation exactly once.
        if let Some((ref m, ref v)) = self.adam_state {
            for &buffer in m.buffers.iter().chain(&v.buffers) {
                self.gpu.destroy_buffer(buffer);
            }
        }
        if let Some(buffer) = self.optimizer_segments {
            self.gpu.destroy_buffer(buffer);
        }
        if let Some(buffer) = self.grad_clip_acc {
            self.gpu.destroy_buffer(buffer);
        }
        if let Some(buffer) = self.grad_clip_partials {
            self.gpu.destroy_buffer(buffer);
        }
        if let Some(buffer) = self.agc_scales {
            self.gpu.destroy_buffer(buffer);
        }
        if let Some(ref accumulator) = self.adam_grouped_grad_norm {
            self.gpu.destroy_buffer(accumulator.buffer);
        }
        if let Some(ref accumulators) = self.grad_accum {
            for &buffer in &accumulators.buffers {
                self.gpu.destroy_buffer(buffer);
            }
        }
    }
}
