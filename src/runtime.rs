use crate::compile::{BufferRef, Dispatch, ExecutionPlan, ShaderEntry};
use std::collections::{HashMap, HashSet};

type Gpu = blade_graphics::Context;

fn ceil_div(a: u32, b: u32) -> u32 {
    a.div_ceil(b)
}

/// Minimum workgroup count below which the cooperative-matrix (2×2-tile) path is
/// skipped and the scalar tiled matmul is used instead.
///
/// The 2×2-tile coop kernel launches ceil(m/32)×ceil(n/32) workgroups, each backed
/// by a single wave64. For good occupancy a GPU with ~32 compute units needs ≈ 512
/// concurrent wavefronts.  SmolVLA's chunk_size=50 never produces enough WGs even
/// for its largest matmuls (e.g. m=50, n=2048 → 2×64=128 WGs), so the scalar path
/// runs ≈50% faster for that workload.  Larger batch sizes or model widths that do
/// exceed this threshold will automatically use the coop path.
///
/// For dispatches with a large K (reduction dimension, K ≥ 1024), each coop workgroup
/// does proportionally more arithmetic even when the tile count is low, so a lower
/// threshold applies.  This primarily benefits backward-pass input-gradient matmuls like
/// [50,720]×[2048,720]^T (k=2048) while leaving small-K dispatches (k=720) on the faster
/// scalar path at low tile counts.
const MIN_COOP_WORKGROUPS: u32 = 128;
const MIN_COOP_WORKGROUPS_HIGH_K: u32 = 32; // used when K >= 1024

// ---- ShaderData structs matching codegen global variable names ----

// matmul: var matrix_a, matrix_b, matrix_c, params
#[derive(blade_macros::ShaderData)]
struct MatMulData {
    matrix_a: blade_graphics::BufferPiece,
    matrix_b: blade_graphics::BufferPiece,
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

// reduce: var src, dst, params (same layout as UnaryData)

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
// (same layout as UnaryData)

// causal_attention: var src_a (q), src_b (k), bias (v), dst, params
#[derive(blade_macros::ShaderData)]
struct CausalAttentionData {
    src_a: blade_graphics::BufferPiece,
    src_b: blade_graphics::BufferPiece,
    bias: blade_graphics::BufferPiece, // v, named "bias" to match binding
    dst: blade_graphics::BufferPiece,
    params: MatMulParams, // seq, num_heads, num_kv_heads, head_dim → reuse 4xu32
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
    params: MatMulParams,
}

// multi_head_attn_grad: var d_out (dO), src_a (Q), src_b (K), bias (V), lse, fwd_dst (O), dst (dQ/dK/dV), params
#[derive(blade_macros::ShaderData)]
struct MultiHeadAttnGradData {
    d_out: blade_graphics::BufferPiece,
    src_a: blade_graphics::BufferPiece,
    src_b: blade_graphics::BufferPiece,
    bias: blade_graphics::BufferPiece,
    lse: blade_graphics::BufferPiece,
    fwd_dst: blade_graphics::BufferPiece,
    dst: blade_graphics::BufferPiece,
    params: MatMulParams,
}

// ---- Pipeline collection ----

struct Pipelines {
    /// Scalar (default) pipelines.
    map: HashMap<ShaderEntry, blade_graphics::ComputePipeline>,
    /// Cooperative-matrix pipelines for dispatches with `use_coop = true`.
    coop_map: HashMap<ShaderEntry, blade_graphics::ComputePipeline>,
}

impl Pipelines {
    fn new(gpu: &Gpu, plan: &ExecutionPlan) -> Self {
        use crate::codegen::ShaderGroup;
        use blade_graphics as bg;

        // Collect which shader groups are needed.
        // For matmul entries, compile BOTH scalar and coop if any dispatch uses coop.
        let mut needed: HashSet<ShaderGroup> = HashSet::new();
        let mut needed_coop: HashSet<ShaderGroup> = HashSet::new();
        let mut entries_for_group: HashMap<ShaderGroup, HashSet<ShaderEntry>> = HashMap::new();

        for dispatch in &plan.dispatches {
            let group = dispatch.shader.shader_group();
            needed.insert(group);
            entries_for_group
                .entry(group)
                .or_default()
                .insert(dispatch.shader.clone());
            if dispatch.use_coop {
                let coop_group = match group {
                    ShaderGroup::MatMul => ShaderGroup::MatMulCoop,
                    ShaderGroup::MatMulAdd => ShaderGroup::MatMulCoopAdd,
                    ShaderGroup::MatMulAT => ShaderGroup::MatMulCoopAT,
                    ShaderGroup::MatMulBT => ShaderGroup::MatMulCoopBT,
                    _ => continue,
                };
                needed_coop.insert(coop_group);
                entries_for_group
                    .entry(coop_group)
                    .or_default()
                    .insert(dispatch.shader.clone());
            }
        }

        // Always compile SgdUpdate if the plan has trainable parameters.
        if !plan.param_grad_pairs.is_empty() {
            needed.insert(ShaderGroup::Sgd);
            entries_for_group
                .entry(ShaderGroup::Sgd)
                .or_default()
                .insert(ShaderEntry::SgdUpdate);
        }

        let mut map = HashMap::new();
        let mut coop_map = HashMap::new();

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

        for &group in &needed {
            compile_group(group, &mut map);
        }
        for &group in &needed_coop {
            compile_group(group, &mut coop_map);
        }

        Self { map, coop_map }
    }

    fn get(&self, dispatch: &Dispatch) -> &blade_graphics::ComputePipeline {
        if dispatch.use_coop {
            if let Some(p) = self.coop_map.get(&dispatch.shader) {
                return p;
            }
        }
        &self.map[&dispatch.shader]
    }
}

/// Get the ShaderDataLayout for a given shader entry.
fn shader_data_layout(entry: &ShaderEntry) -> blade_graphics::ShaderDataLayout {
    use blade_graphics::ShaderData;
    match *entry {
        ShaderEntry::MatMul | ShaderEntry::MatMulAT | ShaderEntry::MatMulBT => MatMulData::layout(),
        ShaderEntry::FusedMatMulAdd
        | ShaderEntry::FusedMatMulATAdd
        | ShaderEntry::FusedMatMulBTAdd => FusedMatMulAddData::layout(),
        ShaderEntry::Relu | ShaderEntry::Sigmoid | ShaderEntry::Neg | ShaderEntry::Silu => {
            UnaryData::layout()
        }
        ShaderEntry::Add | ShaderEntry::Mul | ShaderEntry::Greater | ShaderEntry::SwiGLU => {
            BinaryData::layout()
        }
        ShaderEntry::BiasAdd => BiasAddData::layout(),
        ShaderEntry::SgdUpdate => SgdData::layout(),
        ShaderEntry::SumAll | ShaderEntry::MeanAll | ShaderEntry::SumRows => UnaryData::layout(),
        ShaderEntry::Softmax => SoftmaxData::layout(),
        ShaderEntry::CrossEntropyLoss => CrossEntropyData::layout(),
        ShaderEntry::Transpose => TransposeData::layout(),
        ShaderEntry::RmsNorm => RmsNormData::layout(),
        ShaderEntry::Embedding => EmbeddingData::layout(),
        ShaderEntry::RoPE => UnaryData::layout(), // same layout: src, dst, params
        ShaderEntry::CausalAttention => CausalAttentionData::layout(),
        ShaderEntry::Gelu => UnaryData::layout(),
        ShaderEntry::LayerNorm => LayerNormData::layout(),
        ShaderEntry::FullAttention | ShaderEntry::CrossAttention => CausalAttentionData::layout(),
        ShaderEntry::MultiHeadAttn => MultiHeadAttnData::layout(),
        ShaderEntry::MultiHeadAttnGradQ
        | ShaderEntry::MultiHeadAttnGradK
        | ShaderEntry::MultiHeadAttnGradV => MultiHeadAttnGradData::layout(),
        ShaderEntry::SwiGLUGradGate => TernaryData::layout(),
        ShaderEntry::SwiGLUGradUp | ShaderEntry::SiluGrad => BinaryData::layout(),
        ShaderEntry::RmsNormGradW | ShaderEntry::RmsNormGradX => CausalAttentionData::layout(),
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
        if let Some(extra) = dispatch.extra_output {
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
        if let Some(extra) = dispatch.extra_output {
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
    gpu: Gpu,
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
}

impl Session {
    /// Run a tiny cooperative matmul and check the result.
    /// Returns false if the GPU doesn't support the required cooperative
    /// matrix types (e.g. AMD RADV advertises the extension but rejects
    /// the specific f32 matrix shapes).
    fn test_coop_matmul(gpu: &Gpu) -> bool {
        use crate::codegen::ShaderGroup;
        use blade_graphics as bg;

        let sm = crate::codegen::generate_module(ShaderGroup::MatMulCoop);
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

        // 32×32 test: A filled with 0.5, B = identity → C should equal A (all 0.5).
        // Must use N=32 (not 16) so all 4 accumulators of the 2×2-tile coop kernel
        // are within bounds: the second N-tile needs cc1 = tile_col+16+(tid&15) < N,
        // which requires N ≥ 32. With N=16, acc_01 always stages zeros and stores
        // them over valid positions of acc_00, causing the test to fail even on a
        // working implementation.
        const N: usize = 32;
        const BUF_SIZE: u64 = (N * N * 4) as u64;
        let a_buf = gpu.create_buffer(bg::BufferDesc {
            name: "test_a",
            size: BUF_SIZE,
            memory: bg::Memory::Shared,
        });
        let b_buf = gpu.create_buffer(bg::BufferDesc {
            name: "test_b",
            size: BUF_SIZE,
            memory: bg::Memory::Shared,
        });
        let c_buf = gpu.create_buffer(bg::BufferDesc {
            name: "test_c",
            size: BUF_SIZE,
            memory: bg::Memory::Shared,
        });
        unsafe {
            let a = std::slice::from_raw_parts_mut(a_buf.data() as *mut f32, N * N);
            let b = std::slice::from_raw_parts_mut(b_buf.data() as *mut f32, N * N);
            let c = std::slice::from_raw_parts_mut(c_buf.data() as *mut f32, N * N);
            // A = all 0.5
            a.fill(0.5);
            // B = identity
            b.fill(0.0);
            for i in 0..N {
                b[i * N + i] = 1.0;
            }
            // C = 0 (accumulator)
            c.fill(0.0);
        }

        let mut encoder = gpu.create_command_encoder(bg::CommandEncoderDesc {
            name: "coop_test",
            buffer_count: 2,
        });
        encoder.start();
        {
            let mut pass = encoder.compute("coop_test");
            let mut pc = pass.with(&pipeline);
            pc.bind(
                0,
                &MatMulData {
                    matrix_a: a_buf.at(0),
                    matrix_b: b_buf.at(0),
                    matrix_c: c_buf.at(0),
                    params: MatMulParams {
                        m: N as u32,
                        n: N as u32,
                        k: N as u32,
                        _pad: 0,
                    },
                },
            );
            pc.dispatch([1, 1, 1]);
        }
        let sp = gpu.submit(&mut encoder);
        let _ = gpu.wait_for(&sp, !0);

        let result =
            unsafe { std::slice::from_raw_parts(c_buf.data() as *const f32, N * N).to_vec() };

        gpu.destroy_command_encoder(&mut encoder);
        gpu.destroy_compute_pipeline(&mut pipeline);
        gpu.destroy_buffer(a_buf);
        gpu.destroy_buffer(b_buf);
        gpu.destroy_buffer(c_buf);

        // A * I should equal A (all 0.5)
        let ok = result.iter().all(|v| (*v - 0.5).abs() < 0.05);
        if !ok {
            log::warn!(
                "cooperative matmul self-test failed: expected [1,1,1,1], got {:?}",
                result
            );
        }
        ok
    }

    /// Create a session from a compiled execution plan.
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

        let coop_caps = gpu.capabilities().cooperative_matrix;
        // Our coop shader is hardcoded for 16×16 f16 tiles (RDNA3/Volta+).
        // Skip if the device reports a different tile size (e.g. lavapipe 8×8).
        let use_coop_matmul =
            coop_caps.is_supported() && coop_caps.f16_tile == 16 && Self::test_coop_matmul(&gpu);
        if !use_coop_matmul {
            let info = gpu.device_information();
            log::warn!(
                "cooperative matrix not available on {} ({}) (f32_tile={}, f16_tile={}); using naive matmul",
                info.device_name,
                info.driver_name,
                coop_caps.f32_tile,
                coop_caps.f16_tile,
            );
        } else {
            log::info!(
                "cooperative matrix enabled (f32_tile={}, f16_tile={})",
                coop_caps.f32_tile,
                coop_caps.f16_tile,
            );
        }

        let mut plan = plan;
        // Mark individual dispatches for coop and recompute their workgroups.
        // Unlike the old all-or-nothing policy, each dispatch is independently
        // evaluated. Pipelines now stores both scalar and coop variants.
        if use_coop_matmul {
            use crate::codegen::ShaderGroup;
            for dispatch in &mut plan.dispatches {
                let group = dispatch.shader.shader_group();
                let (m, n, k) = match group {
                    ShaderGroup::MatMul | ShaderGroup::MatMulAdd => {
                        (dispatch.params[0], dispatch.params[2], dispatch.params[1])
                    }
                    // Coop AT/BT disabled: backward matmul shapes (K=50 for
                    // weight grads, M=50 for input grads) don't benefit from
                    // coop, and f16 staging precision compounds through layers.
                    ShaderGroup::MatMulAT | ShaderGroup::MatMulBT => continue,
                    _ => continue,
                };
                let coop_wgs = ceil_div(m, 32) * ceil_div(n, 32);
                let min_wgs = if k >= 1024 {
                    MIN_COOP_WORKGROUPS_HIGH_K
                } else {
                    MIN_COOP_WORKGROUPS
                };
                let m_edge_safe = m % 32 <= 16;
                let n_edge_safe = n % 32 <= 16;
                if coop_wgs >= min_wgs && m_edge_safe && n_edge_safe {
                    dispatch.use_coop = true;
                    dispatch.workgroups = [ceil_div(m, 32), ceil_div(n, 32), 1];
                }
            }
        }

        // Reorder dispatches by dependency level so parallel branches (e.g. Q/K/V
        // projections) cluster together, then partition into barrier groups.
        reorder_by_level(&mut plan.dispatches);
        let groups = compute_groups(&plan.dispatches);
        log::info!(
            "{} dispatches → {} barrier groups",
            plan.dispatches.len(),
            groups.len()
        );

        let buffers: Vec<blade_graphics::Buffer> = plan
            .buffers
            .iter()
            .enumerate()
            .map(|(i, &size)| {
                let size = size.max(4);
                gpu.create_buffer(blade_graphics::BufferDesc {
                    name: &format!("buf_{}", i),
                    size: size as u64,
                    memory: blade_graphics::Memory::Shared,
                })
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

        let pipelines = Pipelines::new(&gpu, &plan);
        let encoder = gpu.create_command_encoder(blade_graphics::CommandEncoderDesc {
            name: "meganeura",
            buffer_count: 2,
        });

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
        }
    }

    /// Enable or disable per-dispatch GPU profiling.
    ///
    /// When enabled, `step()` runs one compute pass per dispatch with
    /// individual GPU timestamps. Call `dump_gpu_timings()` after the
    /// *next* `step()` to see per-pass timings from the profiled run.
    /// Note: blade limits timestamps to 100 passes per submission.
    pub fn set_profiling(&mut self, enabled: bool) {
        self.profiling = enabled;
    }

    /// Upload parameter data to GPU buffers.
    pub fn set_parameter(&mut self, name: &str, data: &[f32]) {
        for &(ref param_name, buf_ref) in &self.plan.param_buffers {
            if param_name == name {
                self.upload_buffer(buf_ref, bytemuck::cast_slice(data));
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
            unsafe {
                let ptr = buffer.data() as *const f32;
                *ptr
            }
        } else {
            0.0
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
        sorted.sort_by(|a, b| b.1.1.cmp(&a.1.1));
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
            // Note: blade caps timestamps at 100 passes per submission.
            for i in 0..self.plan.dispatches.len() {
                let dispatch = &self.plan.dispatches[i];
                let pipeline = self.pipelines.get(dispatch);
                let mut pass = self.encoder.compute(&dispatch.label);
                let mut pc = pass.with(pipeline);
                Self::bind_dispatch(&self.buffers, dispatch, &mut pc);
                pc.dispatch(dispatch.workgroups);
            }
        } else {
            // Group mode (default): one compute pass per barrier group.
            // Groups were pre-computed at session creation by reordering dispatches
            // by dependency level and partitioning on RAW hazards. Each pass
            // boundary in blade emits an ALL_COMMANDS barrier, so N groups =
            // N barriers — far fewer than one per dispatch.
            for gi in 0..self.groups.len() {
                let group = self.groups[gi].clone();
                // Name the pass after its first + last dispatch labels
                let pass_name = if group.len() <= 2 {
                    self.plan.dispatches[group.clone()]
                        .iter()
                        .map(|d| d.label.as_str())
                        .collect::<Vec<_>>()
                        .join("+")
                } else {
                    let first = &self.plan.dispatches[group.start].label;
                    let last = &self.plan.dispatches[group.end - 1].label;
                    format!("{}..{}", first, last)
                };
                let mut pass = self.encoder.compute(&pass_name);
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
        match dispatch.shader {
            ShaderEntry::MatMul => {
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
            ShaderEntry::MatMulAT | ShaderEntry::MatMulBT => {
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
            ShaderEntry::FusedMatMulAdd => {
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
            ShaderEntry::Relu | ShaderEntry::Sigmoid | ShaderEntry::Neg | ShaderEntry::Silu => {
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
                pc.bind(
                    0,
                    &CrossEntropyData {
                        logits: buf(dispatch.input_buffers[0]),
                        labels: buf(dispatch.input_buffers[1]),
                        grad_out: buf(dispatch.output_buffer),
                        loss_out: buf(dispatch.output_buffer),
                        params: SoftmaxParams {
                            batch: dispatch.params[0],
                            features: dispatch.params[1],
                            _pad0: 0,
                            _pad1: 0,
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
            ShaderEntry::RoPE => {
                pc.bind(
                    0,
                    &UnaryData {
                        src: buf(dispatch.input_buffers[0]),
                        dst: buf(dispatch.output_buffer),
                        params: UnaryParams {
                            len: dispatch.params[0],
                            _pad0: dispatch.params[1],
                            _pad1: dispatch.params[2],
                            _pad2: 0,
                        },
                    },
                );
            }
            ShaderEntry::CausalAttention
            | ShaderEntry::FullAttention
            | ShaderEntry::CrossAttention => {
                pc.bind(
                    0,
                    &CausalAttentionData {
                        src_a: buf(dispatch.input_buffers[0]),
                        src_b: buf(dispatch.input_buffers[1]),
                        bias: buf(dispatch.input_buffers[2]),
                        dst: buf(dispatch.output_buffer),
                        // Attention params are sequential: [seq/q_seq, num_heads/kv_seq,
                        // num_kv_heads/packed_heads, head_dim]. Use n/k in definition order
                        // so memory layout matches the shader's u32x4 field reads.
                        params: MatMulParams {
                            m: dispatch.params[0],
                            n: dispatch.params[1],
                            k: dispatch.params[2],
                            _pad: dispatch.params[3],
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
                        lse: buf(dispatch
                            .extra_output
                            .expect("MultiHeadAttn needs extra_output")),
                        params: MatMulParams {
                            m: dispatch.params[0],
                            n: dispatch.params[1],
                            k: dispatch.params[2],
                            _pad: dispatch.params[3],
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
                        params: MatMulParams {
                            m: dispatch.params[0],
                            n: dispatch.params[1],
                            k: dispatch.params[2],
                            _pad: dispatch.params[3],
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
            ShaderEntry::RmsNormGradW | ShaderEntry::RmsNormGradX => {
                pc.bind(
                    0,
                    &CausalAttentionData {
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

    pub fn plan(&self) -> &ExecutionPlan {
        &self.plan
    }

    /// Number of barrier groups (compute passes) in the dispatch sequence.
    pub fn num_groups(&self) -> usize {
        self.groups.len()
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
        for buffer in &self.buffers {
            self.gpu.destroy_buffer(*buffer);
        }
    }
}
