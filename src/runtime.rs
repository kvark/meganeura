use crate::compile::{BufferRef, ExecutionPlan, ShaderEntry};
use std::collections::HashMap;

type Gpu = blade_graphics::Context;

// ---- ShaderData structs matching codegen global variable names ----

// matmul (cooperative matrix): var matrix_a, matrix_b, matrix_c, params
#[derive(blade_macros::ShaderData)]
struct MatMulData {
    matrix_a: blade_graphics::BufferPiece,
    matrix_b: blade_graphics::BufferPiece,
    matrix_c: blade_graphics::BufferPiece,
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
    src_b: blade_graphics::BufferPiece,  // weight
    bias: blade_graphics::BufferPiece,   // bias
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
    fn new(gpu: &Gpu, plan: &ExecutionPlan) -> Self {
        use crate::codegen::ShaderGroup;
        use blade_graphics as bg;

        // Collect unique (ShaderGroup, entries) pairs
        let mut needed: HashMap<ShaderGroup, Vec<&ShaderEntry>> = HashMap::new();
        for dispatch in &plan.dispatches {
            needed
                .entry(dispatch.shader.shader_group())
                .or_default()
                .push(&dispatch.shader);
        }

        let mut map = HashMap::new();
        for (group, entries) in &needed {
            let wgsl = crate::codegen::generate_wgsl(*group);
            let shader = gpu.create_shader(bg::ShaderDesc {
                source: &wgsl,
                naga_module: None,
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
        ShaderEntry::Relu | ShaderEntry::Sigmoid | ShaderEntry::Neg | ShaderEntry::Silu => {
            UnaryData::layout()
        }
        ShaderEntry::Add | ShaderEntry::Mul | ShaderEntry::Greater => BinaryData::layout(),
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
    encoder: blade_graphics::CommandEncoder,
    sync_point: Option<blade_graphics::SyncPoint>,
    /// Nanosecond offset (in profiler time) of the most recent GPU submit,
    /// used to place GPU pass timings on the GPU track.
    last_submit_ns: u64,
}

impl Session {
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

    /// Wait for any pending GPU work.
    pub fn wait(&mut self) {
        if let Some(sp) = self.sync_point.take() {
            let _span = tracing::info_span!("wait").entered();
            self.gpu.wait_for(&sp, !0);
        }
    }

    /// Execute the full dispatch sequence (forward + backward + update).
    pub fn step(&mut self) {
        let _span = tracing::info_span!("step").entered();
        self.wait();

        self.encoder.start();
        // After start(), blade exposes GPU timings from the *previous* submission.
        self.drain_gpu_timings();

        for i in 0..self.plan.dispatches.len() {
            let dispatch = &self.plan.dispatches[i];
            let pipeline = self.pipelines.get(&dispatch.shader);
            let mut pass = self.encoder.compute(&format!("{:?}", dispatch.shader));
            let mut pc = pass.with(pipeline);
            Self::bind_dispatch(&self.buffers, dispatch, &mut pc);
            pc.dispatch(dispatch.workgroups);
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
            ShaderEntry::Add | ShaderEntry::Mul | ShaderEntry::Greater => {
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
                        params: MatMulParams {
                            m: dispatch.params[0],
                            k: dispatch.params[1],
                            n: dispatch.params[2],
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
