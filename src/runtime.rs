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
const MIN_COOP_WORKGROUPS: u32 = 512;

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

// ---- Pipeline collection ----

struct Pipelines {
    map: HashMap<ShaderEntry, blade_graphics::ComputePipeline>,
}

impl Pipelines {
    fn new(gpu: &Gpu, plan: &ExecutionPlan, use_coop_matmul: bool) -> Self {
        use crate::codegen::ShaderGroup;
        use blade_graphics as bg;

        // Collect unique (ShaderGroup, entries) pairs
        let mut needed: HashMap<ShaderGroup, Vec<&ShaderEntry>> = HashMap::new();
        for dispatch in &plan.dispatches {
            let mut group = dispatch.shader.shader_group();
            // Upgrade MatMul/MatMulAdd to cooperative matrix path if supported AND
            // the dispatch has enough workgroups for coop to be efficient.
            if use_coop_matmul && (group == ShaderGroup::MatMul || group == ShaderGroup::MatMulAdd)
            {
                let m = dispatch.params[0];
                let n = dispatch.params[2];
                let coop_wgs = ceil_div(m, 32) * ceil_div(n, 32);
                if coop_wgs >= MIN_COOP_WORKGROUPS {
                    group = if group == ShaderGroup::MatMul {
                        ShaderGroup::MatMulCoop
                    } else {
                        ShaderGroup::MatMulCoopAdd
                    };
                }
            }
            needed.entry(group).or_default().push(&dispatch.shader);
        }

        let mut map = HashMap::new();
        for (group, entries) in &needed {
            let module = crate::codegen::generate_module(*group);
            let shader = gpu.create_shader(bg::ShaderDesc {
                source: "",
                naga_module: Some(module),
            });

            for entry in entries {
                if map.contains_key(*entry) {
                    continue;
                }
                let layout = shader_data_layout(entry);
                let pipeline = gpu.create_compute_pipeline(bg::ComputePipelineDesc {
                    name: (*entry).entry_point(),
                    data_layouts: &[&layout],
                    compute: shader.at((*entry).entry_point()),
                });
                map.insert((*entry).clone(), pipeline);
            }
        }

        Self { map }
    }

    fn get(&self, entry: &ShaderEntry) -> &blade_graphics::ComputePipeline {
        &self.map[entry]
    }
}

/// Get the ShaderDataLayout for a given shader entry.
fn shader_data_layout(entry: &ShaderEntry) -> blade_graphics::ShaderDataLayout {
    use blade_graphics::ShaderData;
    match *entry {
        ShaderEntry::MatMul => MatMulData::layout(),
        ShaderEntry::FusedMatMulAdd => FusedMatMulAddData::layout(),
        ShaderEntry::Relu | ShaderEntry::Sigmoid | ShaderEntry::Neg | ShaderEntry::Silu => {
            UnaryData::layout()
        }
        ShaderEntry::Add | ShaderEntry::Mul | ShaderEntry::Greater | ShaderEntry::SwiGLU => {
            BinaryData::layout()
        }
        ShaderEntry::BiasAdd => BiasAddData::layout(),
        ShaderEntry::SgdUpdate => SgdData::layout(),
        ShaderEntry::SumAll | ShaderEntry::MeanAll => UnaryData::layout(),
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
}

impl Session {
    /// Run a tiny cooperative matmul and check the result.
    /// Returns false if the GPU doesn't support the required cooperative
    /// matrix types (e.g. AMD RADV advertises the extension but rejects
    /// the specific f32 matrix shapes).
    fn test_coop_matmul(gpu: &Gpu) -> bool {
        use crate::codegen::ShaderGroup;
        use blade_graphics as bg;

        let module = crate::codegen::generate_module(ShaderGroup::MatMulCoop);
        let shader = match gpu.try_create_shader(bg::ShaderDesc {
            source: "",
            naga_module: Some(module),
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
                device_id: None,
                ..Default::default()
            })
        }
        .expect("failed to initialize blade GPU context");

        let coop_caps = gpu.capabilities().cooperative_matrix;
        let use_coop_matmul = coop_caps.is_supported() && Self::test_coop_matmul(&gpu);
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
        if use_coop_matmul {
            // Recompute matmul dispatch workgroups for 2×2-tile cooperative kernels.
            // Only upgrade dispatches where the coop WG count meets the occupancy
            // threshold; small matmuls (e.g. SmolVLA's m=50) are left on the scalar
            // path which is faster at low parallelism.
            for dispatch in &mut plan.dispatches {
                if dispatch.shader == ShaderEntry::MatMul
                    || dispatch.shader == ShaderEntry::FusedMatMulAdd
                {
                    let m = dispatch.params[0];
                    let n = dispatch.params[2];
                    let coop_wgs = ceil_div(m, 32) * ceil_div(n, 32);
                    if coop_wgs >= MIN_COOP_WORKGROUPS {
                        dispatch.workgroups = [ceil_div(m, 32), ceil_div(n, 32), 1];
                    }
                    // else: keep scalar workgroups [ceil(n/16), ceil(m/16), 1]
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

        let pipelines = Pipelines::new(&gpu, &plan, use_coop_matmul);
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
        }
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

        if std::env::var("MEGANEURA_MULTI_PASS").is_ok() {
            // Multi-pass mode: one compute pass per dispatch with per-pass barriers
            // and GPU timestamps. Useful for profiling (dump_gpu_timings).
            for i in 0..self.plan.dispatches.len() {
                let dispatch = &self.plan.dispatches[i];
                let pipeline = self.pipelines.get(&dispatch.shader);
                let mut pass = self.encoder.compute(&format!("{:?}", dispatch.shader));
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
                let mut pass = self.encoder.compute("step");
                for i in group {
                    let dispatch = &self.plan.dispatches[i];
                    let pipeline = self.pipelines.get(&dispatch.shader);
                    let mut pc = pass.with(pipeline);
                    Self::bind_dispatch(&self.buffers, dispatch, &mut pc);
                    pc.dispatch(dispatch.workgroups);
                }
            }
        }

        self.last_submit_ns = crate::profiler::now_ns();
        self.sync_point = Some(self.gpu.submit(&mut self.encoder));
    }

    fn bind_dispatch(
        buffers: &[blade_graphics::Buffer],
        dispatch: &crate::compile::Dispatch,
        pc: &mut blade_graphics::PipelineEncoder<'_, '_>,
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
                            k: dispatch.params[1],
                            n: dispatch.params[2],
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

        for &(param_buf, grad_buf) in &self.plan.param_grad_pairs {
            let len = (self.plan.buffers[param_buf.0 as usize] / 4) as u32;
            let pipeline = self.pipelines.get(&ShaderEntry::SgdUpdate);
            let mut pass = self.encoder.compute("sgd_update");
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

        self.last_submit_ns = crate::profiler::now_ns();
        self.sync_point = Some(self.gpu.submit(&mut self.encoder));
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
}

impl Drop for Session {
    fn drop(&mut self) {
        self.wait();
        self.gpu.destroy_command_encoder(&mut self.encoder);
        for (_, pipeline) in self.pipelines.map.iter_mut() {
            self.gpu.destroy_compute_pipeline(pipeline);
        }
        for buffer in &self.buffers {
            self.gpu.destroy_buffer(*buffer);
        }
    }
}
