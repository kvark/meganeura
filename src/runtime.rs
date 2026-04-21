use crate::compile::{BufferRef, Dispatch, ExecutionPlan, ShaderEntry};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

type Gpu = blade_graphics::Context;

// scatter_add: var indices (u32), src, dst, params
#[derive(blade_macros::ShaderData)]
struct ScatterAddData {
    indices: blade_graphics::BufferPiece,
    src: blade_graphics::BufferPiece,
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
    pub total_buffer_bytes: usize,
    pub adam_state_bytes: usize,
    pub num_buffers: usize,
    pub largest_buffer_bytes: usize,
}

impl std::fmt::Display for MemorySummary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{} buffers, {:.1} MB total ({:.1} MB adam state), largest {:.1} MB",
            self.num_buffers,
            self.total_buffer_bytes as f64 / 1e6,
            self.adam_state_bytes as f64 / 1e6,
            self.largest_buffer_bytes as f64 / 1e6,
        )
    }
}

/// Minimum workgroup count below which the cooperative-matrix (2×2-tile) path is
/// skipped and the scalar tiled matmul is used instead.
///
/// On discrete GPUs (NVIDIA, AMD dGPU), tensor core throughput is so much higher
/// than scalar ALUs that the coop path wins even at moderate workgroup counts.
/// A threshold of 8 allows coop for most practical matmul shapes while avoiding
/// extremely small dispatches (e.g. 1×1 tile grids) where the overhead dominates.
///
/// On integrated GPUs with few CUs, the scalar path may still be faster at low
/// workgroup counts due to limited parallelism, but the coop path self-selects
/// out when the GPU doesn't advertise cooperative matrix support.
const MIN_COOP_WORKGROUPS: u32 = 32;
const MIN_COOP_WORKGROUPS_HIGH_K: u32 = 32;

// ---- ShaderData structs matching codegen global variable names ----

// matmul: var matrix_a, matrix_b, matrix_c, params
#[derive(blade_macros::ShaderData)]
struct MatMulData {
    matrix_a: blade_graphics::BufferPiece,
    matrix_b: blade_graphics::BufferPiece,
    matrix_c: blade_graphics::BufferPiece,
    params: MatMulParams,
}

// fused_rms_norm_matmul_coop: var matrix_a, matrix_b, rsqrt_buf, w_norm, matrix_c, params
#[derive(blade_macros::ShaderData)]
#[allow(unused)] // Coop path is currently disabled
struct FusedRmsNormMatMulCoopData {
    matrix_a: blade_graphics::BufferPiece,
    matrix_b: blade_graphics::BufferPiece,
    rsqrt_buf: blade_graphics::BufferPiece,
    w_norm: blade_graphics::BufferPiece,
    matrix_c: blade_graphics::BufferPiece,
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

// bias_add: var src, bias, dst, params
#[derive(blade_macros::ShaderData)]
struct BiasAddData {
    src: blade_graphics::BufferPiece,
    bias: blade_graphics::BufferPiece,
    dst: blade_graphics::BufferPiece,
    params: BiasAddParams,
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct BiasAddParams {
    len: u32,
    bias_len: u32,
    _pad0: u32,
    _pad1: u32,
}

// sgd: var param, grad, dst, params
#[derive(blade_macros::ShaderData)]
struct SgdData {
    param: blade_graphics::BufferPiece,
    grad: blade_graphics::BufferPiece,
    dst: blade_graphics::BufferPiece,
    params: SgdParams,
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct SgdParams {
    len: u32,
    lr: f32,
    _pad0: u32,
    _pad1: u32,
}

// adam: var param (rw), grad (ro), m (rw), v (rw), params
#[derive(blade_macros::ShaderData)]
struct AdamData {
    param: blade_graphics::BufferPiece,
    grad: blade_graphics::BufferPiece,
    m: blade_graphics::BufferPiece,
    v: blade_graphics::BufferPiece,
    params: AdamParams,
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct AdamParams {
    len: u32,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    step: f32,
    _pad0: u32,
    _pad1: u32,
}

// reduce: var src, dst, params (same layout as UnaryData)

// Schedule-template reduction params: { outer, inner, _pad0, _pad1 }.
// 16 bytes; same layout as BiasAddParams with a different name for
// clarity at the call site.
#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct ReductionParams {
    outer: u32,
    inner: u32,
    _pad0: u32,
    _pad1: u32,
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

// cache_write: var src, dst (read_write), kv_pos_buf, params
#[derive(blade_macros::ShaderData)]
struct CacheWriteData {
    src: blade_graphics::BufferPiece,
    dst: blade_graphics::BufferPiece,
    kv_pos_buf: blade_graphics::BufferPiece,
    params: UnaryParams, // dim, _pad x3
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

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct GroupNormParams {
    batch: u32,
    channels: u32,
    spatial: u32,
    num_groups: u32,
    eps_bits: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

// group_norm_grad: var src_a (grad_out), src_b (input), bias (weight), dst, params
#[derive(blade_macros::ShaderData)]
struct GroupNormGradInputData {
    src_a: blade_graphics::BufferPiece,
    src_b: blade_graphics::BufferPiece,
    bias: blade_graphics::BufferPiece,
    dst: blade_graphics::BufferPiece,
    params: GroupNormParams,
}

// group_norm_grad_weight_bias: var src_a (grad_out), src_b (input), bias (dummy), dst, params
// The bias field is unused by the grad_weight_bias entry point but exists
// in the shared GroupNormGrad module (used by grad_input). We bind a dummy buffer.
#[derive(blade_macros::ShaderData)]
struct GroupNormGradWeightBiasData {
    src_a: blade_graphics::BufferPiece,
    src_b: blade_graphics::BufferPiece,
    bias: blade_graphics::BufferPiece,
    dst: blade_graphics::BufferPiece,
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
    params: MatMulParams, // _reserved, num_heads, num_kv_heads, head_dim
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

// softmax: var src, dst, params
#[derive(blade_macros::ShaderData)]
struct SoftmaxData {
    src: blade_graphics::BufferPiece,
    dst: blade_graphics::BufferPiece,
    params: SoftmaxParams,
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

// bce: var pred, labels, grad_out, loss_out, params
#[derive(blade_macros::ShaderData)]
struct BceData {
    pred: blade_graphics::BufferPiece,
    labels: blade_graphics::BufferPiece,
    grad_out: blade_graphics::BufferPiece,
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

struct Pipelines {
    /// Scalar (default) pipelines.
    map: HashMap<ShaderEntry, blade_graphics::ComputePipeline>,
    /// Cooperative-matrix pipelines for dispatches with `use_coop = true`.
    coop_map: HashMap<ShaderEntry, blade_graphics::ComputePipeline>,
    /// Small-tile (32×32) pipelines for dispatches with `use_small_tiles = true`.
    small_map: HashMap<ShaderEntry, blade_graphics::ComputePipeline>,
    /// Epilogue-fused pipelines keyed by (shader, epilogue chain).
    epilogue_map:
        HashMap<(ShaderEntry, Vec<crate::compile::EpilogueOp>), blade_graphics::ComputePipeline>,
    /// Schedule-template-generated pointwise pipelines, keyed by DAG hash.
    /// Populated for dispatches that carry a `pointwise` DAG.
    pointwise_map: HashMap<u64, blade_graphics::ComputePipeline>,
    /// Schedule-template-generated reduction pipelines, keyed by kernel
    /// content hash. Populated for dispatches that carry a `reduction`
    /// kernel spec.
    reduction_map: HashMap<u64, blade_graphics::ComputePipeline>,
}

impl Pipelines {
    fn new(
        gpu: &Gpu,
        plan: &ExecutionPlan,
        coop_config: Option<&crate::codegen::CoopConfig>,
    ) -> Self {
        use crate::codegen::ShaderGroup;
        use blade_graphics as bg;

        // Collect which shader groups are needed.
        // For matmul entries, compile BOTH scalar and coop if any dispatch uses coop.
        let mut needed: HashSet<ShaderGroup> = HashSet::new();
        let mut needed_coop: HashSet<ShaderGroup> = HashSet::new();
        let mut entries_for_group: HashMap<ShaderGroup, HashSet<ShaderEntry>> = HashMap::new();
        // Extract head_dim from attention dispatches for parameterized shader generation.
        let mut attention_head_dim: Option<u32> = None;

        for dispatch in &plan.dispatches {
            let group = dispatch.shader.shader_group();
            needed.insert(group);
            entries_for_group
                .entry(group)
                .or_default()
                .insert(dispatch.shader.clone());
            // Attention dispatches store head_dim at params[3].
            if group == ShaderGroup::MultiHeadAttn && dispatch.params.len() >= 4 {
                attention_head_dim = Some(dispatch.params[3]);
            }
            if dispatch.use_small_tiles {
                let small_group = match group {
                    ShaderGroup::MatMul => ShaderGroup::MatMulSmall,
                    ShaderGroup::MatMulAdd => ShaderGroup::MatMulSmallAdd,
                    ShaderGroup::MatMulAT => ShaderGroup::MatMulSmallAT,
                    ShaderGroup::MatMulBT => ShaderGroup::MatMulSmallBT,
                    _ => continue,
                };
                needed.insert(small_group);
                entries_for_group
                    .entry(small_group)
                    .or_default()
                    .insert(dispatch.shader.clone());
            }
            if dispatch.use_coop {
                let coop_group = match group {
                    ShaderGroup::MatMul => ShaderGroup::MatMulCoop,
                    ShaderGroup::MatMulAdd => ShaderGroup::MatMulCoopAdd,
                    ShaderGroup::MatMulAT => ShaderGroup::MatMulCoopAT,
                    ShaderGroup::MatMulBT => ShaderGroup::MatMulCoopBT,
                    ShaderGroup::Conv2dGemm => ShaderGroup::Conv2dGemmCoop,
                    ShaderGroup::Conv2dGradInputGemm => ShaderGroup::Conv2dGradInputGemmCoop,
                    ShaderGroup::FusedRmsNormMatMul => ShaderGroup::FusedRmsNormMatMulCoop,
                    _ => continue,
                };
                needed_coop.insert(coop_group);
                entries_for_group
                    .entry(coop_group)
                    .or_default()
                    .insert(dispatch.shader.clone());
            }
        }

        // Always compile SGD and Adam if the plan has trainable parameters.
        if !plan.param_grad_pairs.is_empty() {
            needed.insert(ShaderGroup::Sgd);
            entries_for_group
                .entry(ShaderGroup::Sgd)
                .or_default()
                .insert(ShaderEntry::SgdUpdate);
            needed.insert(ShaderGroup::Adam);
            entries_for_group
                .entry(ShaderGroup::Adam)
                .or_default()
                .insert(ShaderEntry::AdamUpdate);
        }

        let mut map = HashMap::new();
        let mut coop_map = HashMap::new();
        let mut small_map = HashMap::new();

        let compile_group =
            |group: ShaderGroup,
             target: &mut HashMap<ShaderEntry, blade_graphics::ComputePipeline>| {
                let sm = crate::codegen::generate_module(group);
                let shader = gpu.create_shader(bg::ShaderDesc {
                    source: &sm.source,
                    naga_module: Some(sm.module),
                });
                if let Some(entries) = entries_for_group.get(&group) {
                    for entry in entries {
                        let layout = shader_data_layout(entry);
                        let pipeline = gpu.create_compute_pipeline(bg::ComputePipelineDesc {
                            name: entry.entry_point(),
                            data_layouts: &[&layout],
                            compute: shader.at(entry.entry_point()),
                        });
                        target.insert(entry.clone(), pipeline);
                    }
                }
            };

        let small_tile_groups: HashSet<ShaderGroup> = [
            ShaderGroup::MatMulSmall,
            ShaderGroup::MatMulSmallAdd,
            ShaderGroup::MatMulSmallAT,
            ShaderGroup::MatMulSmallBT,
        ]
        .into_iter()
        .collect();

        for &group in &needed {
            if small_tile_groups.contains(&group) {
                compile_group(group, &mut small_map);
            } else if group == ShaderGroup::MultiHeadAttn {
                // Use parameterized attention generator with actual head_dim.
                let hd = attention_head_dim.unwrap_or(64);
                let sm = crate::codegen::generate_attention_module(hd);
                let shader = gpu.create_shader(bg::ShaderDesc {
                    source: &sm.source,
                    naga_module: Some(sm.module),
                });
                if let Some(entries) = entries_for_group.get(&group) {
                    for entry in entries {
                        let layout = shader_data_layout(entry);
                        let pipeline = gpu.create_compute_pipeline(bg::ComputePipelineDesc {
                            name: entry.entry_point(),
                            data_layouts: &[&layout],
                            compute: shader.at(entry.entry_point()),
                        });
                        map.insert(entry.clone(), pipeline);
                    }
                }
            } else {
                compile_group(group, &mut map);
            }
        }
        for &group in &needed_coop {
            if let Some(config) = coop_config {
                // Use the runtime-detected coop config for shader generation.
                let sm = crate::codegen::generate_coop_module(group, config);
                let shader = gpu.create_shader(bg::ShaderDesc {
                    source: &sm.source,
                    naga_module: Some(sm.module),
                });
                if let Some(entries) = entries_for_group.get(&group) {
                    for entry in entries {
                        let layout = shader_data_layout(entry);
                        let pipeline = gpu.create_compute_pipeline(bg::ComputePipelineDesc {
                            name: entry.entry_point(),
                            data_layouts: &[&layout],
                            compute: shader.at(entry.entry_point()),
                        });
                        coop_map.insert(entry.clone(), pipeline);
                    }
                }
            } else {
                compile_group(group, &mut coop_map);
            }
        }

        // Compile epilogue-fused pipelines for dispatches with non-empty epilogue.
        // Prefer the new MatMulEpilogue (PointwiseDAG); fall back to legacy
        // Vec<EpilogueOp> for cached plans that predate the DAG migration.
        let mut epilogue_map = HashMap::new();
        for dispatch in &plan.dispatches {
            if dispatch.matmul_epilogue.is_none() && dispatch.epilogue.is_empty() {
                continue;
            }
            let key = (dispatch.shader.clone(), dispatch.epilogue.clone());
            if epilogue_map.contains_key(&key) {
                continue;
            }
            let group = dispatch.shader.shader_group();
            let sm = if let Some(ref epi) = dispatch.matmul_epilogue {
                crate::codegen::generate_matmul_with_dag_epilogue(group, epi)
            } else {
                crate::codegen::generate_matmul_with_epilogue(group, &dispatch.epilogue)
            };
            let shader = gpu.create_shader(bg::ShaderDesc {
                source: &sm.source,
                naga_module: Some(sm.module),
            });
            let layout = shader_data_layout(&dispatch.shader);
            let pipeline = gpu.create_compute_pipeline(bg::ComputePipelineDesc {
                name: dispatch.shader.entry_point(),
                data_layouts: &[&layout],
                compute: shader.at(dispatch.shader.entry_point()),
            });
            epilogue_map.insert(key, pipeline);
        }

        // Compile schedule-template pointwise pipelines. Each unique DAG
        // (keyed by its content hash) gets one pipeline; the dispatch's
        // existing `shader` field is used only to pick the data layout
        // (UnaryData for n=1 inputs, BinaryData for n=2), which already
        // matches the generated WGSL's binding names.
        let mut pointwise_map: HashMap<u64, blade_graphics::ComputePipeline> = HashMap::new();
        for dispatch in &plan.dispatches {
            let dag = match dispatch.pointwise {
                Some(ref d) => d,
                None => continue,
            };
            let key = dag.hash_key();
            if pointwise_map.contains_key(&key) {
                continue;
            }
            let template = crate::schedule::KernelTemplate::Pointwise {
                dag: dag.clone(),
                grid: crate::schedule::GridShape::default(),
            };
            let sm = crate::schedule::lower(&template);
            let shader = gpu.create_shader(bg::ShaderDesc {
                source: &sm.source,
                naga_module: Some(sm.module),
            });
            let layout = pointwise_data_layout(dag.n_inputs);
            let pipeline = gpu.create_compute_pipeline(bg::ComputePipelineDesc {
                name: crate::schedule::POINTWISE_ENTRY,
                data_layouts: &[&layout],
                compute: shader.at(crate::schedule::POINTWISE_ENTRY),
            });
            pointwise_map.insert(key, pipeline);
        }

        // Compile schedule-template reduction pipelines.
        let mut reduction_map: HashMap<u64, blade_graphics::ComputePipeline> = HashMap::new();
        for dispatch in &plan.dispatches {
            let kernel = match dispatch.reduction {
                Some(ref k) => k,
                None => continue,
            };
            let key = kernel.hash_key();
            if reduction_map.contains_key(&key) {
                continue;
            }
            let sm = crate::schedule::lower(&kernel.to_template());
            let shader = gpu.create_shader(bg::ShaderDesc {
                source: &sm.source,
                naga_module: Some(sm.module),
            });
            let layout = reduction_data_layout(kernel);
            let pipeline = gpu.create_compute_pipeline(bg::ComputePipelineDesc {
                name: crate::schedule::REDUCTION_ENTRY,
                data_layouts: &[&layout],
                compute: shader.at(crate::schedule::REDUCTION_ENTRY),
            });
            reduction_map.insert(key, pipeline);
        }

        Self {
            map,
            coop_map,
            small_map,
            epilogue_map,
            pointwise_map,
            reduction_map,
        }
    }

    fn get(&self, dispatch: &Dispatch) -> &blade_graphics::ComputePipeline {
        if let Some(ref k) = dispatch.reduction {
            if let Some(p) = self.reduction_map.get(&k.hash_key()) {
                return p;
            }
        }
        if let Some(ref dag) = dispatch.pointwise {
            if let Some(p) = self.pointwise_map.get(&dag.hash_key()) {
                return p;
            }
        }
        if !dispatch.epilogue.is_empty() {
            let key = (dispatch.shader.clone(), dispatch.epilogue.clone());
            if let Some(p) = self.epilogue_map.get(&key) {
                return p;
            }
        }
        if dispatch.use_coop {
            if let Some(p) = self.coop_map.get(&dispatch.shader) {
                return p;
            }
        }
        if dispatch.use_small_tiles {
            if let Some(p) = self.small_map.get(&dispatch.shader) {
                return p;
            }
        }
        &self.map[&dispatch.shader]
    }
}

/// ShaderDataLayout for a schedule-template reduction pipeline, chosen
/// by (n_per_elem, n_per_row, n_per_col). Names align with the bindings
/// emitted by `schedule::lower` for reductions.
fn reduction_data_layout(
    kernel: &crate::schedule::ReductionKernel,
) -> blade_graphics::ShaderDataLayout {
    use blade_graphics::ShaderData;
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

/// Get the ShaderDataLayout for a given shader entry.
fn shader_data_layout(entry: &ShaderEntry) -> blade_graphics::ShaderDataLayout {
    use blade_graphics::ShaderData;
    match *entry {
        ShaderEntry::MatMul
        | ShaderEntry::MatMulAT
        | ShaderEntry::MatMulBT
        | ShaderEntry::MatMulGemv
        | ShaderEntry::MatMulGemvBT => MatMulData::layout(),
        ShaderEntry::FusedMatMulAdd
        | ShaderEntry::FusedMatMulATAdd
        | ShaderEntry::FusedMatMulBTAdd
        | ShaderEntry::MatMulGemvAdd => FusedMatMulAddData::layout(),
        ShaderEntry::Relu
        | ShaderEntry::Sigmoid
        | ShaderEntry::Tanh
        | ShaderEntry::Neg
        | ShaderEntry::Abs
        | ShaderEntry::Log
        | ShaderEntry::Recip
        | ShaderEntry::Silu => UnaryData::layout(),
        ShaderEntry::Add | ShaderEntry::Mul | ShaderEntry::Greater | ShaderEntry::SwiGLU => {
            BinaryData::layout()
        }
        ShaderEntry::BiasAdd => BiasAddData::layout(),
        ShaderEntry::SgdUpdate => SgdData::layout(),
        ShaderEntry::AdamUpdate => AdamData::layout(),
        ShaderEntry::ScatterAdd => ScatterAddData::layout(),
        ShaderEntry::SwiGLUConcat | ShaderEntry::SwiGLUConcatGrad => BinaryData::layout(),
        ShaderEntry::SumAll | ShaderEntry::MeanAll | ShaderEntry::SumRows => UnaryData::layout(),
        ShaderEntry::Softmax => SoftmaxData::layout(),
        ShaderEntry::CrossEntropyLoss => CrossEntropyData::layout(),
        ShaderEntry::BceLoss => BceData::layout(),
        ShaderEntry::Transpose => TransposeData::layout(),
        ShaderEntry::RmsNorm => RmsNormData::layout(),
        ShaderEntry::Embedding => EmbeddingData::layout(),
        ShaderEntry::RoPE | ShaderEntry::RoPEGrad => RoPEData::layout(),
        ShaderEntry::Gelu => UnaryData::layout(),
        ShaderEntry::LayerNorm => LayerNormData::layout(),
        ShaderEntry::MultiHeadAttn => MultiHeadAttnData::layout(),
        ShaderEntry::MultiHeadAttnGradQ
        | ShaderEntry::MultiHeadAttnGradK
        | ShaderEntry::MultiHeadAttnGradV => MultiHeadAttnGradData::layout(),
        ShaderEntry::MultiHeadAttnGradKV => MultiHeadAttnGradKVData::layout(),
        ShaderEntry::SwiGLUGradGate => TernaryData::layout(),
        ShaderEntry::SwiGLUGradUp | ShaderEntry::SiluGrad => BinaryData::layout(),
        ShaderEntry::RmsNormGradW
        | ShaderEntry::RmsNormGradWRowPar
        | ShaderEntry::RmsNormGradX => FourBufData::layout(),
        ShaderEntry::LayerNormGradWB | ShaderEntry::LayerNormGradX => FourBufData::layout(),
        ShaderEntry::FusedRmsNormMatMul => FourBufData::layout(),
        ShaderEntry::RmsNormRsqrt => UnaryData::layout(),
        ShaderEntry::GroupNorm | ShaderEntry::GroupNormSilu => GroupNormData::layout(),
        ShaderEntry::GroupNormGradInput => GroupNormGradInputData::layout(),
        ShaderEntry::GroupNormGradWeightBias => GroupNormGradWeightBiasData::layout(),
        ShaderEntry::Concat => BinaryData::layout(),
        ShaderEntry::SplitA | ShaderEntry::SplitB => UnaryData::layout(),
        ShaderEntry::Upsample2x | ShaderEntry::Upsample2xGrad => UnaryData::layout(),
        ShaderEntry::Conv2d => Conv2dData::layout(),
        ShaderEntry::Conv2dGemm | ShaderEntry::Conv2dGemmSmall | ShaderEntry::Conv2dGemmCoop => {
            Conv2dData::layout()
        }
        ShaderEntry::Conv2dGradInput => Conv2dGradInputData::layout(),
        ShaderEntry::Conv2dGradInputGemm
        | ShaderEntry::Conv2dGradInputGemmSmall
        | ShaderEntry::Conv2dGradInputGemmCoop => Conv2dGradInputData::layout(),
        ShaderEntry::Conv2dGradWeight
        | ShaderEntry::Conv2dGradWeightGemm
        | ShaderEntry::Conv2dGradWeightGemmSmall => Conv2dGradWeightData::layout(),
        ShaderEntry::RoPEDynamic => RoPEDynamicData::layout(),
        ShaderEntry::CacheWrite => CacheWriteData::layout(),
        ShaderEntry::CachedAttention => CachedAttentionData::layout(),
        ShaderEntry::MaxPool2d => MaxPool2dData::layout(),
        ShaderEntry::GlobalAvgPool => GlobalAvgPoolData::layout(),
        ShaderEntry::GlobalAvgPoolGrad => UnaryData::layout(),
        ShaderEntry::WinogradInputTransform
        | ShaderEntry::WinogradOutputTransform
        | ShaderEntry::WinogradWeightTransform => WinogradTransformData::layout(),
        ShaderEntry::WinogradBatchedMatMul | ShaderEntry::WinogradBatchedMatMulSmall => {
            MatMulData::layout()
        }
    }
}

// ---- Dispatch scheduling ----

/// Reorder dispatches by dependency level so parallel branches cluster together.
///
/// Level is defined as: 0 for dispatches with no dependencies on other
/// dispatches (only on inputs/params), and `1 + max(level of producers)`
/// otherwise. A stable sort by level produces a valid topological order where
/// all dispatches at the same level are mutually independent — they can share
/// a single compute pass without any barrier between them.
fn reorder_by_level(dispatches: &mut Vec<Dispatch>) {
    let n = dispatches.len();
    if n == 0 {
        return;
    }
    // Map: buffer id → index of the dispatch that writes it.
    let mut producer: HashMap<u32, usize> = HashMap::new();
    let mut levels = vec![0u32; n];
    for (i, dispatch) in dispatches.iter().enumerate() {
        let level = dispatch
            .input_buffers
            .iter()
            .filter_map(|b| producer.get(&b.0))
            .map(|&pred| levels[pred] + 1)
            .max()
            .unwrap_or(0);
        levels[i] = level;
        producer.insert(dispatch.output_buffer.0, i);
        for &extra in &dispatch.extra_outputs {
            producer.insert(extra.0, i);
        }
    }
    // Stable sort by level keeps topological order within a level.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| levels[i]);
    let old = std::mem::take(dispatches);
    *dispatches = order.iter().map(|&i| old[i].clone()).collect();
}

/// Partition the (reordered) dispatch list into barrier groups.
///
/// Dispatches in the same group share one compute pass (no barrier between
/// them). A new group starts whenever a dispatch reads a buffer written by
/// an earlier dispatch in the current group (RAW hazard). After level-based
/// reordering this aligns exactly with level boundaries.
fn compute_groups(dispatches: &[Dispatch]) -> Vec<std::ops::Range<usize>> {
    let mut groups = Vec::new();
    let mut dirty = HashSet::<u32>::new();
    let mut start = 0;
    for (i, dispatch) in dispatches.iter().enumerate() {
        if dispatch.input_buffers.iter().any(|b| dirty.contains(&b.0)) {
            groups.push(start..i);
            start = i;
            dirty.clear();
        }
        dirty.insert(dispatch.output_buffer.0);
        for &extra in &dispatch.extra_outputs {
            dirty.insert(extra.0);
        }
    }
    if !dispatches.is_empty() {
        groups.push(start..dispatches.len());
    }
    groups
}

// ---- Session ----

/// A compiled, ready-to-execute GPU session.
///
/// Holds all blade-graphics resources: context, buffers, pipelines.
/// Calling `step()` replays the pre-compiled dispatch sequence.
pub struct Session {
    gpu: Arc<Gpu>,
    buffers: Vec<blade_graphics::Buffer>,
    pipelines: Pipelines,
    plan: ExecutionPlan,
    /// Pre-computed barrier groups: each range of dispatch indices shares one
    /// compute pass. Pass boundaries in blade emit ALL_COMMANDS barriers.
    groups: Vec<std::ops::Range<usize>>,
    encoder: blade_graphics::CommandEncoder,
    sync_point: Option<blade_graphics::SyncPoint>,
    /// Nanosecond offset (in profiler time) of the most recent GPU submit,
    /// used to place GPU pass timings on the GPU track.
    last_submit_ns: u64,
    /// When true, run in multi-pass mode: one compute pass per dispatch
    /// with individual GPU timestamps. Enables `dump_gpu_timings()`.
    profiling: bool,
    /// Pending SGD learning rate. When set, `step()` appends SGD updates
    /// to the same GPU submission (avoiding a separate submit/wait cycle).
    pending_lr: Option<f32>,
    /// Per-parameter Adam state buffers: (m_buf, v_buf).
    adam_state: Vec<(blade_graphics::Buffer, blade_graphics::Buffer)>,
    /// Adam step counter.
    adam_step: u32,
    /// Pending Adam parameters. When set, `step()` appends Adam updates.
    pending_adam: Option<(f32, f32, f32, f32)>, // (lr, beta1, beta2, eps)
}

impl Session {
    /// Select the best cooperative matrix config from GPU capabilities.
    /// Prefers f16 (more throughput) when available, falls back to f32.
    fn select_coop_config(
        caps: &blade_graphics::CooperativeMatrix,
    ) -> Option<crate::codegen::CoopConfig> {
        use crate::codegen::CoopConfig;
        // Escape hatch for diagnosing coop-matrix numerical bugs.
        if std::env::var("MEGANEURA_DISABLE_COOP").is_ok() {
            log::warn!("MEGANEURA_DISABLE_COOP set — forcing scalar matmul");
            return None;
        }
        if caps.f16_tile > 0 {
            Some(CoopConfig {
                tile_size: caps.f16_tile,
                use_f16_input: true,
            })
        } else if caps.f32_tile > 0 {
            Some(CoopConfig {
                tile_size: caps.f32_tile,
                use_f16_input: false,
            })
        } else {
            None
        }
    }

    /// Run a tiny cooperative matmul and check the result.
    /// Returns false if the GPU doesn't support the required cooperative
    /// matrix types (e.g. AMD RADV advertises the extension but rejects
    /// the specific f32 matrix shapes).
    fn test_coop_matmul(gpu: &Gpu, config: &crate::codegen::CoopConfig) -> bool {
        use crate::codegen::ShaderGroup;
        use blade_graphics as bg;

        let sm = crate::codegen::generate_coop_module(ShaderGroup::MatMulCoop, config);
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
        let mut pipeline = gpu.create_compute_pipeline(bg::ComputePipelineDesc {
            name: "main",
            data_layouts: &[&layout],
            compute: shader.at("main"),
        });

        // Test with a multi-tile matmul using varying values. A uniform
        // pattern like A[i,j]=i+1, B[i,j]=j+1 misses bugs where the shader
        // loses index dependence (e.g. coop-load layout mismatches that
        // only surface when neighboring K-indices carry different weights).
        let ot = config.output_tile() as usize;
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
        let mut expected = vec![0.0f32; m * n_out];
        unsafe {
            let a = std::slice::from_raw_parts_mut(a_buf.data() as *mut f32, m * inner);
            let b = std::slice::from_raw_parts_mut(b_buf.data() as *mut f32, inner * n_out);
            let c = std::slice::from_raw_parts_mut(c_buf.data() as *mut f32, m * n_out);
            // Deterministic varying values that depend on both row and column.
            // Keep magnitudes modest so f16 coop paths don't overflow.
            let fill = |idx: usize, dim: usize| -> f32 {
                (((idx * 31 + dim * 7) % 17) as f32 - 8.0) * 0.125
            };
            for i in 0..m {
                for j in 0..inner {
                    a[i * inner + j] = fill(i, j);
                }
            }
            for i in 0..inner {
                for j in 0..n_out {
                    b[i * n_out + j] = fill(i + 100, j + 200);
                }
            }
            c.fill(0.0);
            for i in 0..m {
                for j in 0..n_out {
                    let mut acc = 0.0f32;
                    for p in 0..inner {
                        acc += a[i * inner + p] * b[p * n_out + j];
                    }
                    expected[i * n_out + j] = acc;
                }
            }
        }

        let mut encoder = gpu.create_command_encoder(bg::CommandEncoderDesc {
            name: "coop_test",
            buffer_count: 2,
        });
        encoder.start();
        {
            let mut pass = encoder.compute("coop_test");
            let mut pc = pass.with(&pipeline);
            let ot = config.output_tile();
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
        let _ = gpu.wait_for(&sp, !0);

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
    /// Initializes a fresh `blade_graphics::Context` owned by this session.
    /// Use [`Session::with_context`] to share a context with an existing
    /// Blade-based renderer.
    pub fn new(plan: ExecutionPlan) -> Self {
        // Safety: we only create one GPU context per session, and the
        // context is used exclusively through this Session.
        let gpu = unsafe {
            blade_graphics::Context::init(blade_graphics::ContextDesc {
                validation: cfg!(debug_assertions),
                timing: true,
                capture: false,
                overlay: false,
                device_id: std::env::var("MEGANEURA_DEVICE_ID")
                    .ok()
                    .and_then(|s| s.parse().ok()),
                ..Default::default()
            })
        }
        .expect("failed to initialize blade GPU context");

        Self::with_context(plan, Arc::new(gpu))
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
        let coop_caps = gpu.capabilities().cooperative_matrix;
        let coop_config = Self::select_coop_config(&coop_caps)
            .filter(|config| Self::test_coop_matmul(&gpu, config));
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
        // Mark individual dispatches for coop and recompute their workgroups.
        // Unlike the old all-or-nothing policy, each dispatch is independently
        // evaluated. Pipelines now stores both scalar and coop variants.
        if let Some(ref config) = coop_config {
            use crate::codegen::ShaderGroup;
            let output_tile = config.output_tile();
            let _half_tile = config.tile_size;
            for dispatch in &mut plan.dispatches {
                let group = dispatch.shader.shader_group();
                // Extract (m, n, k, batch) from dispatch params based on shader group.
                let (m, n, k, batch) = match group {
                    ShaderGroup::MatMul | ShaderGroup::MatMulAdd => (
                        dispatch.params[0],
                        dispatch.params[2],
                        dispatch.params[1],
                        1u32,
                    ),
                    ShaderGroup::Conv2dGemm => {
                        // Forward: C[Co, oH*oW] = W[Co, K] × im2col[K, oH*oW]
                        // params: [batch, in_channels, in_h, in_w, out_channels, kernel_h, kernel_w, stride, padding_h, out_h, out_w, padding_w]
                        let out_ch = dispatch.params[4];
                        let kh = dispatch.params[5];
                        let kw = dispatch.params[6];
                        let in_ch = dispatch.params[1];
                        let out_h = dispatch.params[9];
                        let out_w = dispatch.params[10];
                        (out_ch, out_h * out_w, in_ch * kh * kw, dispatch.params[0])
                    }
                    ShaderGroup::Conv2dGradInputGemm => {
                        // params: [batch, in_channels, in_h, in_w, out_channels, kernel_h, kernel_w, stride, padding, out_h, out_w, ...]
                        let in_ch = dispatch.params[1];
                        let in_h = dispatch.params[2];
                        let in_w = dispatch.params[3];
                        let out_ch = dispatch.params[4];
                        let kh = dispatch.params[5];
                        let kw = dispatch.params[6];
                        (in_ch, in_h * in_w, out_ch * kh * kw, dispatch.params[0])
                    }
                    ShaderGroup::FusedRmsNormMatMul => (
                        dispatch.params[0], // m
                        dispatch.params[1], // n
                        dispatch.params[2], // k
                        1u32,
                    ),
                    ShaderGroup::MatMulAT | ShaderGroup::MatMulBT => (
                        dispatch.params[0],
                        dispatch.params[1],
                        dispatch.params[2],
                        1u32,
                    ),
                    // FusedRmsNormMatMul excluded: coop variant's 64-thread
                    // rsqrt prologue is too slow vs the 256-thread scalar shader.
                    _ => continue,
                };
                let coop_wgs = m.div_ceil(output_tile) * n.div_ceil(output_tile) * batch;
                // Conv2d backward GEMM has all-scalar staging with heavy im2col
                // decomposition (integer division), and only 64 threads vs 256
                // for the scalar shader. Require more workgroups to amortize.
                let is_conv_bwd = matches!(group, ShaderGroup::Conv2dGradInputGemm);
                let min_wgs = if is_conv_bwd {
                    64
                } else if k >= 1024 {
                    MIN_COOP_WORKGROUPS_HIGH_K
                } else {
                    MIN_COOP_WORKGROUPS
                };
                if coop_wgs >= min_wgs {
                    dispatch.use_coop = true;
                    dispatch.workgroups = [m.div_ceil(output_tile), n.div_ceil(output_tile), batch];
                    // coopStore/coopLoad operate on full tiles without per-element
                    // bounds checking. Pad output and addend buffers so edge
                    // tiles don't read/write past the end.
                    let padded_m = m.div_ceil(output_tile) * output_tile;
                    let padded_n = n.div_ceil(output_tile) * output_tile;
                    let padded_bytes = (padded_m * padded_n * batch * 4) as usize;
                    let buf_idx = dispatch.output_buffer.0 as usize;
                    if plan.buffers[buf_idx] < padded_bytes {
                        plan.buffers[buf_idx] = padded_bytes;
                    }
                    // FusedMatMulAdd: also pad the addend (src) buffer.
                    if dispatch.input_buffers.len() > 2 {
                        let src_idx = dispatch.input_buffers[2].0 as usize;
                        if plan.buffers[src_idx] < padded_bytes {
                            plan.buffers[src_idx] = padded_bytes;
                        }
                    }
                }
            }
        }

        // Small-tile selection: use 32×32 tiles when the 64×64 dispatch
        // produces very few workgroups (< 16). The 4× more workgroups from
        // smaller tiles improve occupancy on GPUs with many SMs.
        {
        use crate::codegen::ShaderGroup;
        for dispatch in plan.dispatches.iter_mut() {
            if dispatch.use_coop || dispatch.use_small_tiles {
                continue;
            }
            let group = dispatch.shader.shader_group();
            let wgs_64: u32 = dispatch.workgroups.iter().product();
            let has_small = matches!(
                group,
                ShaderGroup::MatMul
                    | ShaderGroup::MatMulAdd
                    | ShaderGroup::MatMulAT
                    | ShaderGroup::MatMulBT
            );
            if has_small && wgs_64 < 16 {
                dispatch.use_small_tiles = true;
                // 32×32 tiles → double the WG count in each spatial dimension
                dispatch.workgroups[0] *= 2;
                dispatch.workgroups[1] *= 2;
            }
        }
        }

        // Reorder dispatches by dependency level so parallel branches (e.g. Q/K/V
        // projections) cluster together, then partition into barrier groups.
        reorder_by_level(&mut plan.dispatches);
        let groups = if std::env::var("MEGANEURA_SERIAL_DISPATCH").is_ok() {
            // Debug: one dispatch per pass — guarantees serial execution.
            log::info!("MEGANEURA_SERIAL_DISPATCH: forcing one dispatch per pass");
            (0..plan.dispatches.len()).map(|i| i..i + 1).collect()
        } else {
            compute_groups(&plan.dispatches)
        };
        log::info!(
            "{} dispatches → {} barrier groups",
            plan.dispatches.len(),
            groups.len()
        );
        // Validate: no RAW hazard within a group (concurrent dispatch safety).
        for group in &groups {
            let mut written = HashSet::<u32>::new();
            for i in group.clone() {
                let d = &plan.dispatches[i];
                for ib in &d.input_buffers {
                    if written.contains(&ib.0) {
                        log::warn!(
                            "RAW hazard in group: dispatch {} ({:?}) reads buf {} written earlier in same group",
                            i,
                            d.shader,
                            ib.0
                        );
                    }
                }
                written.insert(d.output_buffer.0);
                for eo in &d.extra_outputs {
                    written.insert(eo.0);
                }
            }
        }

        let buffers: Vec<blade_graphics::Buffer> = plan
            .buffers
            .iter()
            .enumerate()
            .map(|(i, &size)| {
                let size = size.max(4);
                let buf = gpu.create_buffer(blade_graphics::BufferDesc {
                    name: &format!("buf_{}", i),
                    size: size as u64,
                    memory: blade_graphics::Memory::Shared,
                });
                // Zero-fill to prevent NaN from uninitialized padding regions
                // (coop tiles read/write full tiles beyond logical dimensions).
                unsafe {
                    std::ptr::write_bytes(buf.data(), 0, size);
                }
                buf
            })
            .collect();

        // Upload constant buffer data (gradient constants, scale factors, etc.)
        for &(buf_ref, ref data) in &plan.constant_buffers {
            let buffer = &buffers[buf_ref.0 as usize];
            unsafe {
                let ptr = buffer.data() as *mut f32;
                std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
            }
        }

        let pipelines = Pipelines::new(&gpu, &plan, coop_config.as_ref());
        let encoder = gpu.create_command_encoder(blade_graphics::CommandEncoderDesc {
            name: "meganeura",
            buffer_count: 2,
        });

        let adam_state = plan
            .param_grad_pairs
            .iter()
            .enumerate()
            .map(|(i, &(param_buf, _))| {
                let size = (plan.buffers[param_buf.0 as usize] as u64).max(4);
                let m_buf = gpu.create_buffer(blade_graphics::BufferDesc {
                    name: &format!("adam_m_{i}"),
                    size,
                    memory: blade_graphics::Memory::Shared,
                });
                let v_buf = gpu.create_buffer(blade_graphics::BufferDesc {
                    name: &format!("adam_v_{i}"),
                    size,
                    memory: blade_graphics::Memory::Shared,
                });
                unsafe {
                    std::ptr::write_bytes(m_buf.data(), 0, size as usize);
                    std::ptr::write_bytes(v_buf.data(), 0, size as usize);
                }
                (m_buf, v_buf)
            })
            .collect();

        Self {
            gpu,
            buffers,
            pipelines,
            plan,
            groups,
            encoder,
            sync_point: None,
            last_submit_ns: 0,
            profiling: false,
            pending_lr: None,
            adam_state,
            adam_step: 0,
            pending_adam: None,
        }
    }

    /// Enable or disable per-dispatch GPU profiling.
    ///
    /// When enabled, `step()` runs one compute pass per dispatch with
    /// individual GPU timestamps. Call `dump_gpu_timings()` after the
    /// *next* `step()` to see per-pass timings from the profiled run.
    pub fn set_profiling(&mut self, enabled: bool) {
        self.profiling = enabled;
    }

    /// Shared handle to the underlying Blade GPU context.
    ///
    /// Cheap to clone — the context stays alive as long as any `Arc`
    /// reference remains. Useful for wiring a renderer onto the same
    /// device after meganeura has created the context via [`Session::new`].
    pub fn context(&self) -> Arc<blade_graphics::Context> {
        self.gpu.clone()
    }

    /// Upload parameter data to GPU buffers.
    pub fn set_parameter(&mut self, name: &str, data: &[f32]) {
        // Check regular parameters first
        for &(ref param_name, buf_ref) in &self.plan.param_buffers {
            if param_name == name {
                self.upload_buffer(buf_ref, bytemuck::cast_slice(data));

                // If this source param feeds a derived param, fill the
                // derived buffer according to the transform type.
                for entry in &self.plan.derived_params {
                    let derived_buf = &entry.0;
                    let sources = entry.1.as_slice();

                    // Check if this source name is referenced
                    if !sources.iter().any(|s| s.0 == name) {
                        continue;
                    }

                    match entry.2 {
                        crate::graph::ParamTransform::HorizontalConcat => {
                            // Row-interleaved layout: for each row, source A's
                            // columns come first, then source B's columns.
                            let total_cols: usize = sources.iter().map(|s| s.1).sum();
                            let buf_f32 = self.plan.buffers[derived_buf.0 as usize] / 4;
                            let rows = buf_f32.checked_div(total_cols).unwrap_or(0);
                            let mut col_offset = 0usize;
                            for src in sources {
                                let src_name = &src.0;
                                let src_cols = src.1;
                                if src_name == name && rows > 0 {
                                    let derived_ptr =
                                        self.buffers[derived_buf.0 as usize].data() as *mut f32;
                                    for r in 0..rows {
                                        let src_start = r * src_cols;
                                        let dst_start = r * total_cols + col_offset;
                                        unsafe {
                                            std::ptr::copy_nonoverlapping(
                                                data[src_start..].as_ptr(),
                                                derived_ptr.add(dst_start),
                                                src_cols,
                                            );
                                        }
                                    }
                                }
                                col_offset += src_cols;
                            }
                        }
                        crate::graph::ParamTransform::Winograd3x3 {
                            out_channels,
                            in_channels,
                        } => {
                            // G matrix for Winograd F(2,3)
                            let g: [[f32; 3]; 4] = [
                                [1.0, 0.0, 0.0],
                                [0.5, 0.5, 0.5],
                                [0.5, -0.5, 0.5],
                                [0.0, 0.0, 1.0],
                            ];
                            let derived_ptr =
                                self.buffers[derived_buf.0 as usize].data() as *mut f32;
                            for co in 0..out_channels {
                                for ci in 0..in_channels {
                                    // Extract 3x3 filter: [Co, Ci, 3, 3]
                                    let base = (co * in_channels + ci) * 9;
                                    let w = [
                                        [data[base], data[base + 1], data[base + 2]],
                                        [data[base + 3], data[base + 4], data[base + 5]],
                                        [data[base + 6], data[base + 7], data[base + 8]],
                                    ];
                                    // u = G × w × G^T (4×4 result)
                                    // First: tmp = G × w (4×3)
                                    let mut tmp = [[0.0f32; 3]; 4];
                                    for r in 0..4 {
                                        for c in 0..3 {
                                            for k in 0..3 {
                                                tmp[r][c] += g[r][k] * w[k][c];
                                            }
                                        }
                                    }
                                    // Then: u = tmp × G^T (4×4)
                                    #[allow(clippy::needless_range_loop)]
                                    for r in 0..4 {
                                        for c in 0..4 {
                                            let mut val = 0.0f32;
                                            for k in 0..3 {
                                                val += tmp[r][k] * g[c][k];
                                                // G^T[k][c] = G[c][k]
                                            }
                                            let alpha = r * 4 + c;
                                            let idx = alpha * out_channels * in_channels
                                                + co * in_channels
                                                + ci;
                                            unsafe {
                                                *derived_ptr.add(idx) = val;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                return;
            }
        }
        panic!("unknown parameter: {}", name);
    }

    /// Upload input data.
    pub fn set_input(&mut self, name: &str, data: &[f32]) {
        for &(ref input_name, buf_ref) in &self.plan.input_buffers {
            if input_name == name {
                self.upload_buffer(buf_ref, bytemuck::cast_slice(data));
                return;
            }
        }
        panic!("unknown input: {}", name);
    }

    /// Upload u32 input data (e.g. token IDs for embedding lookup).
    pub fn set_input_u32(&mut self, name: &str, data: &[u32]) {
        for &(ref input_name, buf_ref) in &self.plan.input_buffers {
            if input_name == name {
                self.upload_buffer(buf_ref, bytemuck::cast_slice(data));
                return;
            }
        }
        panic!("unknown input: {}", name);
    }

    fn upload_buffer(&self, buf_ref: BufferRef, data: &[u8]) {
        let buffer = &self.buffers[buf_ref.0 as usize];
        unsafe {
            let ptr = buffer.data();
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
    }

    /// Read back the loss value.
    pub fn read_loss(&self) -> f32 {
        if let Some(buf_ref) = self.plan.loss_buffer {
            let buffer = &self.buffers[buf_ref.0 as usize];
            let n = self.plan.buffers[buf_ref.0 as usize] / 4;
            unsafe {
                let ptr = buffer.data() as *const f32;
                let slice = std::slice::from_raw_parts(ptr, n);
                slice.iter().sum()
            }
        } else {
            0.0
        }
    }

    /// Diagnostic: print per-dispatch output buffer statistics.
    ///
    /// Scans all dispatches and reports any whose output contains NaN, Inf,
    /// or values exceeding `threshold`. Useful for tracing where numerical
    /// instability first appears in the forward/backward chain.
    pub fn trace_dispatches(&self, threshold: f32) {
        for (i, d) in self.plan.dispatches.iter().enumerate() {
            let buf_size = self.plan.buffers[d.output_buffer.0 as usize];
            let n = buf_size / 4;
            if n == 0 {
                continue;
            }
            let read_n = n.min(65536);
            let mut data = vec![0.0f32; read_n];
            self.read_buffer(d.output_buffer, &mut data);

            let max_abs = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let has_nan = data.iter().any(|v| v.is_nan());
            let has_inf = data.iter().any(|v| v.is_infinite());

            if has_nan || has_inf || max_abs > threshold || (max_abs == 0.0 && n > 100) {
                let label = if d.label.is_empty() {
                    format!("{:?}", d.shader)
                } else {
                    d.label.clone()
                };
                log::warn!(
                    "dispatch {i}: {label} max_abs={max_abs:.3e} nan={has_nan} inf={has_inf} n={n}"
                );
            }
        }
    }

    /// Read back the output tensor (first graph output).
    ///
    /// Returns the data as a `Vec<f32>`. For inference graphs this is the
    /// model's prediction; for training graphs it's the loss scalar.
    pub fn read_output(&self, len: usize) -> Vec<f32> {
        if let Some(buf_ref) = self.plan.loss_buffer {
            let mut out = vec![0.0_f32; len];
            self.read_buffer(buf_ref, &mut out);
            out
        } else {
            Vec::new()
        }
    }

    /// Read back a buffer's contents.
    pub fn read_buffer(&self, buf_ref: BufferRef, out: &mut [f32]) {
        let buffer = &self.buffers[buf_ref.0 as usize];
        unsafe {
            let ptr = buffer.data() as *const f32;
            std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), out.len());
        }
    }

    /// Read back a graph output by index.
    ///
    /// Index 0 is the primary output (logits/loss). Higher indices are
    /// additional outputs (e.g. KV tensors from prefill).
    pub fn read_output_by_index(&self, index: usize, out: &mut [f32]) {
        let buf_ref = self.plan.output_buffers[index];
        self.read_buffer(buf_ref, out);
    }

    /// Number of graph outputs.
    pub fn num_outputs(&self) -> usize {
        self.plan.output_buffers.len()
    }

    /// Look up a parameter's buffer reference by name.
    pub fn param_buffer(&self, name: &str) -> Option<BufferRef> {
        self.plan
            .param_buffers
            .iter()
            .find(|entry| entry.0 == name)
            .map(|entry| entry.1)
    }

    /// Read a parameter buffer's contents by name.
    pub fn read_param(&self, name: &str, out: &mut [f32]) {
        let buf_ref = self
            .param_buffer(name)
            .unwrap_or_else(|| panic!("unknown param: {}", name));
        self.read_buffer(buf_ref, out);
    }

    /// Read a parameter's gradient buffer by name.
    ///
    /// Returns the gradient computed during the last backward pass.
    /// Panics if the parameter has no associated gradient (e.g. inference-only session).
    pub fn read_param_grad(&self, name: &str, out: &mut [f32]) {
        let param_buf = self
            .param_buffer(name)
            .unwrap_or_else(|| panic!("unknown param: {}", name));
        let grad_buf = self
            .plan
            .param_grad_pairs
            .iter()
            .find(|&&(p, _)| p == param_buf)
            .map(|&(_, g)| g)
            .unwrap_or_else(|| panic!("no gradient for param: {}", name));
        self.read_buffer(grad_buf, out);
    }

    /// Upload data into a parameter buffer by name (for initializing KV caches etc.).
    pub fn upload_param(&self, name: &str, data: &[f32]) {
        let buf_ref = self
            .param_buffer(name)
            .unwrap_or_else(|| panic!("unknown param: {}", name));
        self.upload_buffer(buf_ref, bytemuck::cast_slice(data));
    }

    /// Print GPU pass timings from the last completed step.
    ///
    /// Must be called after `step()` + `wait()`, then another `step()`
    /// (which triggers `encoder.start()` to collect timings from the
    /// previous submission).
    pub fn dump_gpu_timings(&self) {
        let timings = self.encoder.timings();
        if timings.is_empty() {
            eprintln!("(no GPU timings available)");
            return;
        }
        let total: std::time::Duration = timings.iter().map(|&(_, d)| d).sum();
        eprintln!(
            "--- GPU pass timings ({} passes, {:.2}ms total) ---",
            timings.len(),
            total.as_secs_f64() * 1000.0
        );

        // Aggregate by shader type
        let mut by_type: std::collections::HashMap<&str, (u32, std::time::Duration)> =
            std::collections::HashMap::new();
        for &(ref name, dur) in timings {
            let entry = by_type.entry(name.as_str()).or_default();
            entry.0 += 1;
            entry.1 += dur;
        }
        let mut sorted: Vec<_> = by_type.into_iter().collect();
        sorted.sort_by_key(|e| std::cmp::Reverse(e.1.1));
        for &(name, (count, dur)) in &sorted {
            let pct = dur.as_secs_f64() / total.as_secs_f64() * 100.0;
            eprintln!(
                "  {:>20}: {:>3}x {:>8.2}ms ({:>5.1}%)",
                name,
                count,
                dur.as_secs_f64() * 1000.0,
                pct
            );
        }
        eprintln!("---");
    }

    /// Wait for any pending GPU work.
    pub fn wait(&mut self) {
        if let Some(sp) = self.sync_point.take() {
            let _span = tracing::info_span!("wait").entered();
            let _ = self.gpu.wait_for(&sp, !0);
        }
    }

    /// Execute the full dispatch sequence (forward + backward + update).
    pub fn step(&mut self) {
        let _span = tracing::info_span!("step").entered();
        self.wait();

        self.encoder.start();
        // After start(), blade exposes GPU timings from the *previous* submission.
        self.drain_gpu_timings();

        if self.profiling {
            // Multi-pass mode: one compute pass per dispatch with per-pass barriers
            // and GPU timestamps. Enables dump_gpu_timings() after the next step().
            for i in 0..self.plan.dispatches.len() {
                let dispatch = &self.plan.dispatches[i];
                let pipeline = self.pipelines.get(dispatch);
                let mut pass = self.encoder.compute(&dispatch.label);
                let mut pc = pass.with(pipeline);
                Self::bind_dispatch(&self.buffers, dispatch, &mut pc);
                pc.dispatch(dispatch.workgroups);
            }
        } else {
            // Inline-barrier mode: all dispatches in a single compute pass
            // with lightweight compute-to-compute barriers between groups.
            // Avoids the per-pass overhead of begin_pass/end_pass (timestamps,
            // debug labels) while maintaining correct memory ordering.
            let mut pass = self.encoder.compute("step");
            for gi in 0..self.groups.len() {
                if gi > 0 {
                    pass.barrier();
                }
                let group = self.groups[gi].clone();
                for i in group {
                    let dispatch = &self.plan.dispatches[i];
                    let pipeline = self.pipelines.get(dispatch);
                    let mut pc = pass.with(pipeline);
                    Self::bind_dispatch(&self.buffers, dispatch, &mut pc);
                    pc.dispatch(dispatch.workgroups);
                }
            }
        }

        // If training, append SGD updates as a final barrier group (all
        // independent, so one pass). This avoids a second submit/wait cycle.
        if !self.plan.param_grad_pairs.is_empty() {
            let lr = self.pending_lr.take();
            if let Some(learning_rate) = lr {
                let pipeline = &self.pipelines.map[&ShaderEntry::SgdUpdate];
                let mut pass = self.encoder.compute("sgd_update");
                for &(param_buf, grad_buf) in &self.plan.param_grad_pairs {
                    let len = (self.plan.buffers[param_buf.0 as usize] / 4) as u32;
                    let mut pc = pass.with(pipeline);
                    pc.bind(
                        0,
                        &SgdData {
                            param: self.buffers[param_buf.0 as usize].at(0),
                            grad: self.buffers[grad_buf.0 as usize].at(0),
                            dst: self.buffers[param_buf.0 as usize].at(0),
                            params: SgdParams {
                                len,
                                lr: learning_rate,
                                _pad0: 0,
                                _pad1: 0,
                            },
                        },
                    );
                    pc.dispatch([len.div_ceil(256), 1, 1]);
                }
            } else if let Some((lr, beta1, beta2, eps)) = self.pending_adam.take() {
                self.adam_step += 1;
                let pipeline = &self.pipelines.map[&ShaderEntry::AdamUpdate];
                let mut pass = self.encoder.compute("adam_update");
                for (idx, &(param_buf, grad_buf)) in self.plan.param_grad_pairs.iter().enumerate() {
                    let len = (self.plan.buffers[param_buf.0 as usize] / 4) as u32;
                    let (ref m_buf, ref v_buf) = self.adam_state[idx];
                    let mut pc = pass.with(pipeline);
                    pc.bind(
                        0,
                        &AdamData {
                            param: self.buffers[param_buf.0 as usize].at(0),
                            grad: self.buffers[grad_buf.0 as usize].at(0),
                            m: m_buf.at(0),
                            v: v_buf.at(0),
                            params: AdamParams {
                                len,
                                lr,
                                beta1,
                                beta2,
                                eps,
                                step: self.adam_step as f32,
                                _pad0: 0,
                                _pad1: 0,
                            },
                        },
                    );
                    pc.dispatch([len.div_ceil(256), 1, 1]);
                }
            }
        }

        self.last_submit_ns = crate::profiler::now_ns();
        self.sync_point = Some(self.gpu.submit(&mut self.encoder));
    }

    fn bind_dispatch(
        buffers: &[blade_graphics::Buffer],
        dispatch: &crate::compile::Dispatch,
        pc: &mut impl blade_graphics::traits::PipelineEncoder,
    ) {
        let buf = |r: BufferRef| buffers[r.0 as usize].at(0);
        // Schedule-template reduction dispatches have priority and route
        // by kernel arity (n_per_elem, n_per_row, n_per_col).
        if let Some(ref k) = dispatch.reduction {
            let params = ReductionParams {
                outer: dispatch.params[0],
                inner: dispatch.params[1],
                _pad0: 0,
                _pad1: 0,
            };
            let n_per_col = k.epilogue.as_ref().map_or(0, |e| e.n_per_col_inputs);
            match (k.n_per_elem, k.n_per_row, n_per_col) {
                (1, 0, 0) => {
                    pc.bind(
                        0,
                        &ReductionPass1Data {
                            src: buf(dispatch.input_buffers[0]),
                            dst: buf(dispatch.output_buffer),
                            params,
                        },
                    );
                }
                (1, 1, 0) => {
                    pc.bind(
                        0,
                        &ReductionPass2RowData {
                            src: buf(dispatch.input_buffers[0]),
                            per_row_src: buf(dispatch.input_buffers[1]),
                            dst: buf(dispatch.output_buffer),
                            params,
                        },
                    );
                }
                (1, 0, 1) => {
                    pc.bind(
                        0,
                        &RmsNormData {
                            src: buf(dispatch.input_buffers[0]),
                            bias: buf(dispatch.input_buffers[1]),
                            dst: buf(dispatch.output_buffer),
                            params: BiasAddParams {
                                len: params.outer,
                                bias_len: params.inner,
                                _pad0: 0,
                                _pad1: 0,
                            },
                        },
                    );
                }
                other => {
                    panic!(
                        "reduction kernel with arity {:?} has no runtime binding layout",
                        other
                    )
                }
            }
            return;
        }
        // Schedule-template pointwise dispatches route by DAG arity, not
        // by the dummy `shader` entry — arity may be 3 after fusion, which
        // no ShaderEntry variant represents.
        if let Some(ref dag) = dispatch.pointwise {
            let params = UnaryParams {
                len: dispatch.params[0],
                _pad0: 0,
                _pad1: 0,
                _pad2: 0,
            };
            match dag.n_inputs {
                1 => {
                    pc.bind(
                        0,
                        &UnaryData {
                            src: buf(dispatch.input_buffers[0]),
                            dst: buf(dispatch.output_buffer),
                            params,
                        },
                    );
                }
                2 => {
                    pc.bind(
                        0,
                        &BinaryData {
                            src_a: buf(dispatch.input_buffers[0]),
                            src_b: buf(dispatch.input_buffers[1]),
                            dst: buf(dispatch.output_buffer),
                            params,
                        },
                    );
                }
                3 => {
                    pc.bind(
                        0,
                        &TernaryData {
                            src_a: buf(dispatch.input_buffers[0]),
                            src_b: buf(dispatch.input_buffers[1]),
                            src_c: buf(dispatch.input_buffers[2]),
                            dst: buf(dispatch.output_buffer),
                            params,
                        },
                    );
                }
                n => panic!("pointwise arity {} has no runtime data layout", n),
            }
            return;
        }
        match dispatch.shader {
            ShaderEntry::MatMul | ShaderEntry::MatMulGemv => {
                pc.bind(
                    0,
                    &MatMulData {
                        matrix_a: buf(dispatch.input_buffers[0]),
                        matrix_b: buf(dispatch.input_buffers[1]),
                        matrix_c: buf(dispatch.output_buffer),
                        params: MatMulParams {
                            m: dispatch.params[0],
                            n: dispatch.params[2],
                            k: dispatch.params[1],
                            _pad: 0,
                        },
                    },
                );
            }
            ShaderEntry::MatMulAT | ShaderEntry::MatMulBT | ShaderEntry::MatMulGemvBT => {
                // params layout: [m, n, k, 0]
                pc.bind(
                    0,
                    &MatMulData {
                        matrix_a: buf(dispatch.input_buffers[0]),
                        matrix_b: buf(dispatch.input_buffers[1]),
                        matrix_c: buf(dispatch.output_buffer),
                        params: MatMulParams {
                            m: dispatch.params[0],
                            n: dispatch.params[1],
                            k: dispatch.params[2],
                            _pad: 0,
                        },
                    },
                );
            }
            ShaderEntry::FusedMatMulAdd | ShaderEntry::MatMulGemvAdd => {
                pc.bind(
                    0,
                    &FusedMatMulAddData {
                        matrix_a: buf(dispatch.input_buffers[0]),
                        matrix_b: buf(dispatch.input_buffers[1]),
                        matrix_c: buf(dispatch.output_buffer),
                        src: buf(dispatch.input_buffers[2]), // addend
                        params: MatMulParams {
                            m: dispatch.params[0],
                            n: dispatch.params[2],
                            k: dispatch.params[1],
                            _pad: 0,
                        },
                    },
                );
            }
            ShaderEntry::FusedMatMulATAdd | ShaderEntry::FusedMatMulBTAdd => {
                // params layout: [m, n, k, 0] (same as AT/BT, no swizzle)
                pc.bind(
                    0,
                    &FusedMatMulAddData {
                        matrix_a: buf(dispatch.input_buffers[0]),
                        matrix_b: buf(dispatch.input_buffers[1]),
                        matrix_c: buf(dispatch.output_buffer),
                        src: buf(dispatch.input_buffers[2]), // addend
                        params: MatMulParams {
                            m: dispatch.params[0],
                            n: dispatch.params[1],
                            k: dispatch.params[2],
                            _pad: 0,
                        },
                    },
                );
            }
            ShaderEntry::Relu
            | ShaderEntry::Sigmoid
            | ShaderEntry::Tanh
            | ShaderEntry::Neg
            | ShaderEntry::Abs
            | ShaderEntry::Log
            | ShaderEntry::Recip
            | ShaderEntry::Silu => {
                pc.bind(
                    0,
                    &UnaryData {
                        src: buf(dispatch.input_buffers[0]),
                        dst: buf(dispatch.output_buffer),
                        params: UnaryParams {
                            len: dispatch.params[0],
                            _pad0: 0,
                            _pad1: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            ShaderEntry::SwiGLUConcat | ShaderEntry::SwiGLUConcatGrad => {
                pc.bind(
                    0,
                    &BinaryData {
                        src_a: buf(dispatch.input_buffers[0]),
                        src_b: buf(dispatch.input_buffers[1]),
                        dst: buf(dispatch.output_buffer),
                        params: UnaryParams {
                            len: dispatch.params[0],
                            _pad0: dispatch.params[1], // half_n
                            _pad1: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            ShaderEntry::Add | ShaderEntry::Mul | ShaderEntry::Greater | ShaderEntry::SwiGLU => {
                pc.bind(
                    0,
                    &BinaryData {
                        src_a: buf(dispatch.input_buffers[0]),
                        src_b: buf(dispatch.input_buffers[1]),
                        dst: buf(dispatch.output_buffer),
                        params: UnaryParams {
                            len: dispatch.params[0],
                            _pad0: 0,
                            _pad1: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            ShaderEntry::BiasAdd => {
                pc.bind(
                    0,
                    &BiasAddData {
                        src: buf(dispatch.input_buffers[0]),
                        bias: buf(dispatch.input_buffers[1]),
                        dst: buf(dispatch.output_buffer),
                        params: BiasAddParams {
                            len: dispatch.params[0],
                            bias_len: dispatch.params[1],
                            _pad0: 0,
                            _pad1: 0,
                        },
                    },
                );
            }
            ShaderEntry::SgdUpdate => {
                pc.bind(
                    0,
                    &SgdData {
                        param: buf(dispatch.input_buffers[0]),
                        grad: buf(dispatch.input_buffers[1]),
                        dst: buf(dispatch.output_buffer),
                        params: SgdParams {
                            len: dispatch.params[0],
                            lr: f32::from_bits(dispatch.params[1]),
                            _pad0: 0,
                            _pad1: 0,
                        },
                    },
                );
            }
            ShaderEntry::SumAll | ShaderEntry::MeanAll => {
                pc.bind(
                    0,
                    &UnaryData {
                        src: buf(dispatch.input_buffers[0]),
                        dst: buf(dispatch.output_buffer),
                        params: UnaryParams {
                            len: dispatch.params[0],
                            _pad0: 0,
                            _pad1: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            ShaderEntry::SumRows => {
                // params[0] = m (rows), params[1] = n (cols)
                pc.bind(
                    0,
                    &UnaryData {
                        src: buf(dispatch.input_buffers[0]),
                        dst: buf(dispatch.output_buffer),
                        params: UnaryParams {
                            len: dispatch.params[0],   // m
                            _pad0: dispatch.params[1], // n
                            _pad1: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            ShaderEntry::Softmax => {
                pc.bind(
                    0,
                    &SoftmaxData {
                        src: buf(dispatch.input_buffers[0]),
                        dst: buf(dispatch.output_buffer),
                        params: SoftmaxParams {
                            batch: dispatch.params[0],
                            features: dispatch.params[1],
                            _pad0: 0,
                            _pad1: 0,
                        },
                    },
                );
            }
            ShaderEntry::CrossEntropyLoss => {
                let loss_buf = dispatch
                    .extra_outputs
                    .first()
                    .copied()
                    .unwrap_or(dispatch.output_buffer);
                pc.bind(
                    0,
                    &CrossEntropyData {
                        logits: buf(dispatch.input_buffers[0]),
                        labels: buf(dispatch.input_buffers[1]),
                        grad_out: buf(dispatch.output_buffer),
                        loss_out: buf(loss_buf),
                        params: SoftmaxParams {
                            batch: dispatch.params[0],
                            features: dispatch.params[1],
                            _pad0: 0,
                            _pad1: 0,
                        },
                    },
                );
            }
            ShaderEntry::BceLoss => {
                pc.bind(
                    0,
                    &BceData {
                        pred: buf(dispatch.input_buffers[0]),
                        labels: buf(dispatch.input_buffers[1]),
                        grad_out: buf(dispatch.output_buffer),
                        loss_out: buf(dispatch.output_buffer),
                        params: UnaryParams {
                            len: dispatch.params[0],
                            _pad0: 0,
                            _pad1: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            ShaderEntry::Transpose => {
                pc.bind(
                    0,
                    &TransposeData {
                        src: buf(dispatch.input_buffers[0]),
                        dst: buf(dispatch.output_buffer),
                        params: TransposeParams {
                            m: dispatch.params[0],
                            n: dispatch.params[1],
                            _pad0: 0,
                            _pad1: 0,
                        },
                    },
                );
            }
            ShaderEntry::RmsNorm => {
                pc.bind(
                    0,
                    &RmsNormData {
                        src: buf(dispatch.input_buffers[0]),
                        bias: buf(dispatch.input_buffers[1]),
                        dst: buf(dispatch.output_buffer),
                        params: BiasAddParams {
                            len: dispatch.params[0],
                            bias_len: dispatch.params[1],
                            _pad0: dispatch.params[2], // eps_bits
                            _pad1: 0,
                        },
                    },
                );
            }
            ShaderEntry::Embedding => {
                pc.bind(
                    0,
                    &EmbeddingData {
                        indices: buf(dispatch.input_buffers[0]),
                        src: buf(dispatch.input_buffers[1]),
                        dst: buf(dispatch.output_buffer),
                        params: UnaryParams {
                            len: dispatch.params[0],
                            _pad0: dispatch.params[1],
                            _pad1: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            ShaderEntry::RoPE | ShaderEntry::RoPEGrad => {
                pc.bind(
                    0,
                    &RoPEData {
                        src: buf(dispatch.input_buffers[0]),
                        dst: buf(dispatch.output_buffer),
                        params: RoPEParams {
                            seq: dispatch.params[0],
                            dim: dispatch.params[1],
                            theta_bits: dispatch.params[2],
                            pos_offset: dispatch.params[3],
                            head_dim: dispatch.params[4],
                            _pad0: 0,
                            _pad1: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            ShaderEntry::Gelu => {
                pc.bind(
                    0,
                    &UnaryData {
                        src: buf(dispatch.input_buffers[0]),
                        dst: buf(dispatch.output_buffer),
                        params: UnaryParams {
                            len: dispatch.params[0],
                            _pad0: 0,
                            _pad1: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            ShaderEntry::LayerNorm => {
                pc.bind(
                    0,
                    &LayerNormData {
                        src: buf(dispatch.input_buffers[0]),
                        src_b: buf(dispatch.input_buffers[1]),
                        bias: buf(dispatch.input_buffers[2]),
                        dst: buf(dispatch.output_buffer),
                        params: MatMulParams {
                            m: dispatch.params[0],
                            n: dispatch.params[1],
                            k: dispatch.params[2],
                            _pad: dispatch.params[3],
                        },
                    },
                );
            }
            ShaderEntry::MultiHeadAttn => {
                pc.bind(
                    0,
                    &MultiHeadAttnData {
                        src_a: buf(dispatch.input_buffers[0]),
                        src_b: buf(dispatch.input_buffers[1]),
                        bias: buf(dispatch.input_buffers[2]),
                        dst: buf(dispatch.output_buffer),
                        lse: buf(dispatch.extra_outputs[0]),
                        params: AttentionParams {
                            q_seq: dispatch.params[0],
                            kv_seq: dispatch.params[1],
                            packed_heads: dispatch.params[2],
                            head_dim: dispatch.params[3],
                            window_size: *dispatch.params.get(4).unwrap_or(&0),
                            _pad0: 0,
                            _pad1: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            ShaderEntry::MultiHeadAttnGradKV => {
                pc.bind(
                    0,
                    &MultiHeadAttnGradKVData {
                        d_out: buf(dispatch.input_buffers[0]),
                        src_a: buf(dispatch.input_buffers[1]),
                        src_b: buf(dispatch.input_buffers[2]),
                        bias: buf(dispatch.input_buffers[3]),
                        lse: buf(dispatch.input_buffers[4]),
                        fwd_dst: buf(dispatch.input_buffers[5]),
                        dst: buf(dispatch.output_buffer),
                        dst2: buf(dispatch.extra_outputs[0]),
                        params: AttentionParams {
                            q_seq: dispatch.params[0],
                            kv_seq: dispatch.params[1],
                            packed_heads: dispatch.params[2],
                            head_dim: dispatch.params[3],
                            window_size: dispatch.params[4],
                            _pad0: 0,
                            _pad1: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            ShaderEntry::MultiHeadAttnGradQ
            | ShaderEntry::MultiHeadAttnGradK
            | ShaderEntry::MultiHeadAttnGradV => {
                pc.bind(
                    0,
                    &MultiHeadAttnGradData {
                        d_out: buf(dispatch.input_buffers[0]),
                        src_a: buf(dispatch.input_buffers[1]),
                        src_b: buf(dispatch.input_buffers[2]),
                        bias: buf(dispatch.input_buffers[3]),
                        lse: buf(dispatch.input_buffers[4]),
                        fwd_dst: buf(dispatch.input_buffers[5]),
                        dst: buf(dispatch.output_buffer),
                        params: AttentionParams {
                            q_seq: dispatch.params[0],
                            kv_seq: dispatch.params[1],
                            packed_heads: dispatch.params[2],
                            head_dim: dispatch.params[3],
                            window_size: dispatch.params[4],
                            _pad0: 0,
                            _pad1: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            ShaderEntry::SwiGLUGradGate => {
                pc.bind(
                    0,
                    &TernaryData {
                        src_a: buf(dispatch.input_buffers[0]), // grad_out
                        src_b: buf(dispatch.input_buffers[1]), // gate
                        src_c: buf(dispatch.input_buffers[2]), // up
                        dst: buf(dispatch.output_buffer),
                        params: UnaryParams {
                            len: dispatch.params[0],
                            _pad0: 0,
                            _pad1: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            ShaderEntry::SwiGLUGradUp => {
                pc.bind(
                    0,
                    &BinaryData {
                        src_a: buf(dispatch.input_buffers[0]), // grad_out
                        src_b: buf(dispatch.input_buffers[1]), // gate
                        dst: buf(dispatch.output_buffer),
                        params: UnaryParams {
                            len: dispatch.params[0],
                            _pad0: 0,
                            _pad1: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            ShaderEntry::SiluGrad => {
                pc.bind(
                    0,
                    &BinaryData {
                        src_a: buf(dispatch.input_buffers[0]), // grad_out
                        src_b: buf(dispatch.input_buffers[1]), // x
                        dst: buf(dispatch.output_buffer),
                        params: UnaryParams {
                            len: dispatch.params[0],
                            _pad0: 0,
                            _pad1: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            ShaderEntry::RmsNormGradW
            | ShaderEntry::RmsNormGradWRowPar
            | ShaderEntry::RmsNormGradX => {
                pc.bind(
                    0,
                    &FourBufData {
                        src_a: buf(dispatch.input_buffers[0]), // dy
                        src_b: buf(dispatch.input_buffers[1]), // x
                        bias: buf(dispatch.input_buffers[2]),  // w
                        dst: buf(dispatch.output_buffer),
                        params: MatMulParams {
                            m: dispatch.params[0], // rows
                            n: dispatch.params[1], // cols
                            k: dispatch.params[2], // eps_bits
                            _pad: dispatch.params[3],
                        },
                    },
                );
            }
            ShaderEntry::LayerNormGradWB | ShaderEntry::LayerNormGradX => {
                pc.bind(
                    0,
                    &FourBufData {
                        src_a: buf(dispatch.input_buffers[0]), // dy
                        src_b: buf(dispatch.input_buffers[1]), // x
                        bias: buf(dispatch.input_buffers[2]),  // w
                        dst: buf(dispatch.output_buffer),
                        params: MatMulParams {
                            m: dispatch.params[0], // rows
                            n: dispatch.params[1], // cols
                            k: dispatch.params[2], // eps_bits
                            _pad: dispatch.params[3],
                        },
                    },
                );
            }
            ShaderEntry::FusedRmsNormMatMul => {
                {
                    // Both scalar and coop paths use FourBufData layout:
                    // src_a=X, src_b=W_proj, bias=W_norm, dst=output
                    // Coop variant: rsqrt computed via workgroup tree reduction
                    // input_buffers: [x, w_norm, w_proj]
                    pc.bind(
                        0,
                        &FourBufData {
                            src_a: buf(dispatch.input_buffers[0]), // X
                            src_b: buf(dispatch.input_buffers[2]), // W_proj
                            bias: buf(dispatch.input_buffers[1]),  // W_norm
                            dst: buf(dispatch.output_buffer),
                            params: MatMulParams {
                                m: dispatch.params[0],
                                n: dispatch.params[1],
                                k: dispatch.params[2],
                                _pad: dispatch.params[3],
                            },
                        },
                    );
                }
            }
            ShaderEntry::RmsNormRsqrt => {
                // bindings: src=X, dst=rsqrt, params=(rows, cols, eps_bits, _pad)
                pc.bind(
                    0,
                    &UnaryData {
                        src: buf(dispatch.input_buffers[0]),
                        dst: buf(dispatch.output_buffer),
                        params: UnaryParams {
                            len: dispatch.params[0],
                            _pad0: dispatch.params[1],
                            _pad1: dispatch.params[2],
                            _pad2: dispatch.params[3],
                        },
                    },
                );
            }
            ShaderEntry::AdamUpdate => {
                unreachable!("AdamUpdate is dispatched via adam_step/set_adam, not bind_dispatch");
            }
            ShaderEntry::ScatterAdd => {
                pc.bind(
                    0,
                    &ScatterAddData {
                        indices: buf(dispatch.input_buffers[0]),
                        src: buf(dispatch.input_buffers[1]),
                        dst: buf(dispatch.output_buffer),
                        params: ScatterAddParams {
                            total: dispatch.params[0],
                            seq_len: dispatch.params[1],
                            embed_dim: dispatch.params[2],
                            _pad: 0,
                        },
                    },
                );
            }
            ShaderEntry::GroupNorm | ShaderEntry::GroupNormSilu => {
                let p = &dispatch.params;
                pc.bind(
                    0,
                    &GroupNormData {
                        src: buf(dispatch.input_buffers[0]),
                        src_b: buf(dispatch.input_buffers[1]),
                        bias: buf(dispatch.input_buffers[2]),
                        dst: buf(dispatch.output_buffer),
                        params: GroupNormParams {
                            batch: p[0],
                            channels: p[1],
                            spatial: p[2],
                            num_groups: p[3],
                            eps_bits: p[4],
                            _pad0: 0,
                            _pad1: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            ShaderEntry::GroupNormGradInput => {
                let p = &dispatch.params;
                pc.bind(
                    0,
                    &GroupNormGradInputData {
                        src_a: buf(dispatch.input_buffers[0]),
                        src_b: buf(dispatch.input_buffers[1]),
                        bias: buf(dispatch.input_buffers[2]),
                        dst: buf(dispatch.output_buffer),
                        params: GroupNormParams {
                            batch: p[0],
                            channels: p[1],
                            spatial: p[2],
                            num_groups: p[3],
                            eps_bits: p[4],
                            _pad0: 0,
                            _pad1: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            ShaderEntry::GroupNormGradWeightBias => {
                let p = &dispatch.params;
                pc.bind(
                    0,
                    &GroupNormGradWeightBiasData {
                        src_a: buf(dispatch.input_buffers[0]),
                        src_b: buf(dispatch.input_buffers[1]),
                        bias: buf(dispatch.input_buffers[1]), // dummy, unused by this entry point
                        dst: buf(dispatch.output_buffer),
                        params: GroupNormParams {
                            batch: p[0],
                            channels: p[1],
                            spatial: p[2],
                            num_groups: p[3],
                            eps_bits: p[4],
                            _pad0: 0,
                            _pad1: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            ShaderEntry::Concat => {
                let p = &dispatch.params;
                pc.bind(
                    0,
                    &BinaryData {
                        src_a: buf(dispatch.input_buffers[0]),
                        src_b: buf(dispatch.input_buffers[1]),
                        dst: buf(dispatch.output_buffer),
                        params: UnaryParams {
                            len: p[0],
                            _pad0: p[1],
                            _pad1: p[2],
                            _pad2: p[3],
                        },
                    },
                );
            }
            ShaderEntry::SplitA | ShaderEntry::SplitB => {
                let p = &dispatch.params;
                pc.bind(
                    0,
                    &UnaryData {
                        src: buf(dispatch.input_buffers[0]),
                        dst: buf(dispatch.output_buffer),
                        params: UnaryParams {
                            len: p[0],
                            _pad0: p[1],
                            _pad1: p[2],
                            _pad2: p[3],
                        },
                    },
                );
            }
            ShaderEntry::Upsample2x | ShaderEntry::Upsample2xGrad => {
                let p = &dispatch.params;
                pc.bind(
                    0,
                    &UnaryData {
                        src: buf(dispatch.input_buffers[0]),
                        dst: buf(dispatch.output_buffer),
                        params: UnaryParams {
                            len: p[0],
                            _pad0: p[1],
                            _pad1: p[2],
                            _pad2: p[3],
                        },
                    },
                );
            }
            ShaderEntry::Conv2d
            | ShaderEntry::Conv2dGemm
            | ShaderEntry::Conv2dGemmSmall
            | ShaderEntry::Conv2dGemmCoop => {
                let p = &dispatch.params;
                pc.bind(
                    0,
                    &Conv2dData {
                        src: buf(dispatch.input_buffers[0]),
                        weight: buf(dispatch.input_buffers[1]),
                        dst: buf(dispatch.output_buffer),
                        params: Conv2dParams {
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
                        },
                    },
                );
            }
            ShaderEntry::Conv2dGradInput
            | ShaderEntry::Conv2dGradInputGemm
            | ShaderEntry::Conv2dGradInputGemmSmall
            | ShaderEntry::Conv2dGradInputGemmCoop => {
                let p = &dispatch.params;
                pc.bind(
                    0,
                    &Conv2dGradInputData {
                        grad_out: buf(dispatch.input_buffers[0]),
                        weight: buf(dispatch.input_buffers[1]),
                        dst: buf(dispatch.output_buffer),
                        params: Conv2dParams {
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
                        },
                    },
                );
            }
            ShaderEntry::Conv2dGradWeight
            | ShaderEntry::Conv2dGradWeightGemm
            | ShaderEntry::Conv2dGradWeightGemmSmall => {
                let p = &dispatch.params;
                pc.bind(
                    0,
                    &Conv2dGradWeightData {
                        grad_out: buf(dispatch.input_buffers[0]),
                        src: buf(dispatch.input_buffers[1]),
                        dst: buf(dispatch.output_buffer),
                        params: Conv2dParams {
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
                        },
                    },
                );
            }
            ShaderEntry::RoPEDynamic => {
                pc.bind(
                    0,
                    &RoPEDynamicData {
                        src: buf(dispatch.input_buffers[0]),
                        dst: buf(dispatch.output_buffer),
                        pos_offset_buf: buf(dispatch.input_buffers[1]),
                        params: RoPEParams {
                            seq: dispatch.params[0],
                            dim: dispatch.params[1],
                            theta_bits: dispatch.params[2],
                            pos_offset: 0,
                            head_dim: dispatch.params[4],
                            _pad0: 0,
                            _pad1: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            ShaderEntry::CacheWrite => {
                pc.bind(
                    0,
                    &CacheWriteData {
                        src: buf(dispatch.input_buffers[0]),
                        dst: buf(dispatch.output_buffer),
                        kv_pos_buf: buf(dispatch.input_buffers[2]),
                        params: UnaryParams {
                            len: dispatch.params[0], // dim
                            _pad0: 0,
                            _pad1: 0,
                            _pad2: 0,
                        },
                    },
                );
            }
            ShaderEntry::CachedAttention => {
                pc.bind(
                    0,
                    &CachedAttentionData {
                        src_a: buf(dispatch.input_buffers[0]),      // Q
                        src_b: buf(dispatch.input_buffers[1]),      // K cache
                        bias: buf(dispatch.input_buffers[2]),       // V cache
                        kv_pos_buf: buf(dispatch.input_buffers[3]), // kv_pos
                        dst: buf(dispatch.output_buffer),
                        params: MatMulParams {
                            m: dispatch.params[0],
                            n: dispatch.params[1],
                            k: dispatch.params[2],
                            _pad: dispatch.params[3],
                        },
                    },
                );
            }
            ShaderEntry::MaxPool2d => {
                pc.bind(
                    0,
                    &MaxPool2dData {
                        src: buf(dispatch.input_buffers[0]),
                        dst: buf(dispatch.output_buffer),
                        params: MaxPool2dParams {
                            batch: dispatch.params[0],
                            channels: dispatch.params[1],
                            in_h: dispatch.params[2],
                            in_w: dispatch.params[3],
                            kernel_h: dispatch.params[4],
                            kernel_w: dispatch.params[5],
                            stride: dispatch.params[6],
                            padding: dispatch.params[7],
                            out_h: dispatch.params[8],
                            out_w: dispatch.params[9],
                            _pad0: dispatch.params[10],
                            _pad1: dispatch.params[11],
                        },
                    },
                );
            }
            ShaderEntry::GlobalAvgPool => {
                pc.bind(
                    0,
                    &GlobalAvgPoolData {
                        src: buf(dispatch.input_buffers[0]),
                        dst: buf(dispatch.output_buffer),
                        params: GlobalAvgPoolParams {
                            channels: dispatch.params[0],
                            spatial: dispatch.params[1],
                            total_out: dispatch.params[2],
                            _pad: dispatch.params[3],
                        },
                    },
                );
            }
            ShaderEntry::GlobalAvgPoolGrad => {
                let p = &dispatch.params;
                pc.bind(
                    0,
                    &UnaryData {
                        src: buf(dispatch.input_buffers[0]),
                        dst: buf(dispatch.output_buffer),
                        params: UnaryParams {
                            len: p[0],
                            _pad0: p[1],
                            _pad1: p[2],
                            _pad2: p[3],
                        },
                    },
                );
            }
            ShaderEntry::WinogradInputTransform
            | ShaderEntry::WinogradOutputTransform
            | ShaderEntry::WinogradWeightTransform => {
                let p = &dispatch.params;
                pc.bind(
                    0,
                    &WinogradTransformData {
                        src: buf(dispatch.input_buffers[0]),
                        dst: buf(dispatch.output_buffer),
                        params: WinogradTransformParams {
                            p0: p[0],
                            p1: p[1],
                            p2: p[2],
                            p3: p[3],
                            p4: p[4],
                            p5: p[5],
                            p6: p[6],
                            p7: p[7],
                        },
                    },
                );
            }
            ShaderEntry::WinogradBatchedMatMul | ShaderEntry::WinogradBatchedMatMulSmall => {
                pc.bind(
                    0,
                    &MatMulData {
                        matrix_a: buf(dispatch.input_buffers[0]),
                        matrix_b: buf(dispatch.input_buffers[1]),
                        matrix_c: buf(dispatch.output_buffer),
                        params: MatMulParams {
                            m: dispatch.params[0],
                            n: dispatch.params[1],
                            k: dispatch.params[2],
                            _pad: dispatch.params[3],
                        },
                    },
                );
            }
        }
    }

    /// Read GPU pass timings from the encoder (available after `encoder.start()`)
    /// and record them on the GPU profiling track.
    fn drain_gpu_timings(&self) {
        let timings = self.encoder.timings();
        if !timings.is_empty() {
            crate::profiler::record_gpu_passes(self.last_submit_ns, timings);
        }
    }

    /// Apply SGD updates to all parameters on the GPU.
    pub fn sgd_step(&mut self, learning_rate: f32) {
        let _span = tracing::info_span!("sgd_step").entered();
        self.wait();
        self.encoder.start();
        self.drain_gpu_timings();

        // All SGD updates are independent (different param/grad buffers),
        // so they share a single compute pass — no barriers between them.
        let pipeline = &self.pipelines.map[&ShaderEntry::SgdUpdate];
        let mut pass = self.encoder.compute("sgd_update");
        for &(param_buf, grad_buf) in &self.plan.param_grad_pairs {
            let len = (self.plan.buffers[param_buf.0 as usize] / 4) as u32;
            let mut pc = pass.with(pipeline);
            pc.bind(
                0,
                &SgdData {
                    param: self.buffers[param_buf.0 as usize].at(0),
                    grad: self.buffers[grad_buf.0 as usize].at(0),
                    dst: self.buffers[param_buf.0 as usize].at(0),
                    params: SgdParams {
                        len,
                        lr: learning_rate,
                        _pad0: 0,
                        _pad1: 0,
                    },
                },
            );
            pc.dispatch([len.div_ceil(256), 1, 1]);
        }
        drop(pass);

        self.last_submit_ns = crate::profiler::now_ns();
        self.sync_point = Some(self.gpu.submit(&mut self.encoder));
    }

    /// Set learning rate for SGD updates fused into the next `step()`.
    ///
    /// When set, `step()` appends all SGD parameter updates to the same
    /// GPU submission as forward+backward — eliminating the submit/wait
    /// overhead of a separate `sgd_step()` call.
    pub fn set_learning_rate(&mut self, lr: f32) {
        self.pending_lr = Some(lr);
    }

    /// CPU-fallback SGD update.
    pub fn sgd_step_cpu(&mut self, learning_rate: f32) {
        let _span = tracing::info_span!("sgd_step_cpu").entered();
        self.wait();
        for &(param_buf, grad_buf) in &self.plan.param_grad_pairs {
            let size = self.plan.buffers[param_buf.0 as usize] / 4;
            let param = &self.buffers[param_buf.0 as usize];
            let grad = &self.buffers[grad_buf.0 as usize];
            unsafe {
                let p = param.data() as *mut f32;
                let g = grad.data() as *const f32;
                for i in 0..size {
                    *p.add(i) -= learning_rate * *g.add(i);
                }
            }
        }
    }

    /// Apply Adam optimizer updates to all parameters on the GPU.
    pub fn adam_step(&mut self, lr: f32, beta1: f32, beta2: f32, eps: f32) {
        let _span = tracing::info_span!("adam_step").entered();
        self.adam_step += 1;
        self.wait();
        self.encoder.start();
        self.drain_gpu_timings();

        let pipeline = &self.pipelines.map[&ShaderEntry::AdamUpdate];
        let mut pass = self.encoder.compute("adam_update");
        for (idx, &(param_buf, grad_buf)) in self.plan.param_grad_pairs.iter().enumerate() {
            let len = (self.plan.buffers[param_buf.0 as usize] / 4) as u32;
            let (ref m_buf, ref v_buf) = self.adam_state[idx];
            let mut pc = pass.with(pipeline);
            pc.bind(
                0,
                &AdamData {
                    param: self.buffers[param_buf.0 as usize].at(0),
                    grad: self.buffers[grad_buf.0 as usize].at(0),
                    m: m_buf.at(0),
                    v: v_buf.at(0),
                    params: AdamParams {
                        len,
                        lr,
                        beta1,
                        beta2,
                        eps,
                        step: self.adam_step as f32,
                        _pad0: 0,
                        _pad1: 0,
                    },
                },
            );
            pc.dispatch([len.div_ceil(256), 1, 1]);
        }
        drop(pass);

        self.last_submit_ns = crate::profiler::now_ns();
        self.sync_point = Some(self.gpu.submit(&mut self.encoder));
    }

    /// Set Adam parameters for updates fused into the next `step()`.
    ///
    /// Analogous to [`set_learning_rate`](Self::set_learning_rate) for SGD.
    pub fn set_adam(&mut self, lr: f32, beta1: f32, beta2: f32, eps: f32) {
        self.pending_adam = Some((lr, beta1, beta2, eps));
    }

    pub fn memory_summary(&self) -> MemorySummary {
        let total: usize = self.plan.buffers.iter().sum();
        let largest = self.plan.buffers.iter().copied().max().unwrap_or(0);
        let adam_bytes: usize = self
            .plan
            .param_grad_pairs
            .iter()
            .map(|&(p, _)| self.plan.buffers[p.0 as usize] * 2)
            .sum();
        MemorySummary {
            total_buffer_bytes: total,
            adam_state_bytes: adam_bytes,
            num_buffers: self.plan.buffers.len(),
            largest_buffer_bytes: largest,
        }
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

    /// Save a training checkpoint (parameters + Adam state) to a safetensors file.
    ///
    /// Saves all parameter values, Adam first/second moment buffers (if any),
    /// and the Adam step counter as metadata.
    #[allow(clippy::pattern_type_mismatch)]
    pub fn save_checkpoint(&mut self, path: &std::path::Path) -> std::io::Result<()> {
        use safetensors::tensor::{Dtype, TensorView};

        self.wait();
        let mut owned_data: Vec<(String, Vec<u8>)> = Vec::new();

        // Collect parameter data
        for (name, buf_ref) in &self.plan.param_buffers {
            let byte_len = self.plan.buffers[buf_ref.0 as usize];
            let mut data = vec![0u8; byte_len];
            unsafe {
                let ptr = self.buffers[buf_ref.0 as usize].data() as *const u8;
                std::ptr::copy_nonoverlapping(ptr, data.as_mut_ptr(), byte_len);
            }
            owned_data.push((name.clone(), data));
        }

        // Collect Adam moment buffers (parallel to param_grad_pairs)
        for (idx, &(param_buf, _)) in self.plan.param_grad_pairs.iter().enumerate() {
            if idx >= self.adam_state.len() {
                break;
            }
            let name = self
                .plan
                .param_buffers
                .iter()
                .find(|(_, br)| *br == param_buf)
                .map(|(n, _)| n.clone())
                .unwrap_or_else(|| format!("param_{}", param_buf.0));
            let byte_len = self.plan.buffers[param_buf.0 as usize];
            for (suffix, buf) in [
                ("adam_m", &self.adam_state[idx].0),
                ("adam_v", &self.adam_state[idx].1),
            ] {
                let key = format!("{suffix}.{name}");
                let mut data = vec![0u8; byte_len];
                unsafe {
                    let ptr = buf.data() as *const u8;
                    std::ptr::copy_nonoverlapping(ptr, data.as_mut_ptr(), byte_len);
                }
                owned_data.push((key, data));
            }
        }

        // Build tensor views
        let views: Vec<(String, TensorView<'_>)> = owned_data
            .iter()
            .map(|(name, data)| {
                let float_len = data.len() / 4;
                (
                    name.clone(),
                    TensorView::new(Dtype::F32, vec![float_len], data).expect("tensor view"),
                )
            })
            .collect();

        // Metadata
        let mut metadata = HashMap::new();
        metadata.insert("adam_step".to_string(), self.adam_step.to_string());

        let buf = safetensors::tensor::serialize(views, &Some(metadata))
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        std::fs::write(path, buf)
    }

    /// Load a training checkpoint from a safetensors file.
    ///
    /// Restores parameter values and Adam optimizer state. The session must
    /// have been created from the same graph (same parameter names/sizes).
    #[allow(clippy::pattern_type_mismatch)]
    pub fn load_checkpoint(&mut self, path: &std::path::Path) -> std::io::Result<()> {
        let file_data = std::fs::read(path)?;
        let (header_size, metadata) = safetensors::SafeTensors::read_metadata(&file_data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
        let st = safetensors::SafeTensors::deserialize(&file_data)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
        let _ = header_size;

        // Restore parameters
        for (name, buf_ref) in &self.plan.param_buffers {
            if let Ok(tensor) = st.tensor(name) {
                self.upload_buffer(*buf_ref, tensor.data());
            } else {
                log::warn!("checkpoint missing parameter: {name}");
            }
        }

        // Restore Adam moment buffers
        for (idx, &(param_buf, _)) in self.plan.param_grad_pairs.iter().enumerate() {
            if idx >= self.adam_state.len() {
                break;
            }
            let name = self
                .plan
                .param_buffers
                .iter()
                .find(|(_, br)| *br == param_buf)
                .map(|(n, _)| n.as_str())
                .unwrap_or("");
            for (suffix, buf) in [
                ("adam_m", &self.adam_state[idx].0),
                ("adam_v", &self.adam_state[idx].1),
            ] {
                let key = format!("{suffix}.{name}");
                if let Ok(tensor) = st.tensor(&key) {
                    unsafe {
                        let ptr = buf.data();
                        std::ptr::copy_nonoverlapping(
                            tensor.data().as_ptr(),
                            ptr,
                            tensor.data().len(),
                        );
                    }
                }
            }
        }

        // Restore adam_step from metadata
        if let Some(ref meta) = *metadata.metadata() {
            if let Some(step_str) = meta.get("adam_step") {
                self.adam_step = step_str.parse::<u32>().unwrap_or(0);
            }
        }

        Ok(())
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.wait();
        self.gpu.destroy_command_encoder(&mut self.encoder);
        for (_, pipeline) in self.pipelines.map.iter_mut() {
            self.gpu.destroy_compute_pipeline(pipeline);
        }
        for (_, pipeline) in self.pipelines.coop_map.iter_mut() {
            self.gpu.destroy_compute_pipeline(pipeline);
        }
        for (_, pipeline) in self.pipelines.small_map.iter_mut() {
            self.gpu.destroy_compute_pipeline(pipeline);
        }
        for (_, pipeline) in self.pipelines.epilogue_map.iter_mut() {
            self.gpu.destroy_compute_pipeline(pipeline);
        }
        for (_, pipeline) in self.pipelines.pointwise_map.iter_mut() {
            self.gpu.destroy_compute_pipeline(pipeline);
        }
        for (_, pipeline) in self.pipelines.reduction_map.iter_mut() {
            self.gpu.destroy_compute_pipeline(pipeline);
        }
        for buffer in &self.buffers {
            self.gpu.destroy_buffer(*buffer);
        }
        for &(m_buf, v_buf) in &self.adam_state {
            self.gpu.destroy_buffer(m_buf);
            self.gpu.destroy_buffer(v_buf);
        }
    }
}
