//! Experimental persistent workgroup interpreter, independent of graph lowering.
//!
//! A fixed worker pool claims immutable jobs from a device-side atomic queue.
//! Each job owns its data and executes a sequence of dependent operations on
//! one workgroup. Jobs may have different shapes and instruction streams.
//! Workers live for one batch dispatch; they are not pinned to physical CUs.
//! See `examples/megakernel.rs` for a residual MLP and a separate-dispatch control.

use blade_graphics as bg;
use std::{
    ops::Range,
    sync::Arc,
    time::{Duration, Instant},
};

/// Operations over offsets (in f32 elements) within a job's private data.
#[derive(Clone, Debug)]
pub enum Instruction {
    /// `dst = src * scale + bias`.
    Affine {
        src: u32,
        dst: u32,
        len: u32,
        scale: f32,
        bias: f32,
    },
    /// `dst = a + b`.
    Add { a: u32, b: u32, dst: u32, len: u32 },
    /// `dst = max(src, 0)`.
    Relu { src: u32, dst: u32, len: u32 },
    /// `dst = src / sqrt(mean(src * src) + epsilon)`, without learned weights.
    RmsNorm {
        src: u32,
        dst: u32,
        len: u32,
        epsilon: f32,
    },
    /// `dst[rows] = weights[rows, columns] * src[columns]`.
    /// `weights` is an offset into the batch's shared row-major weight buffer.
    MatVec {
        src: u32,
        weights: u32,
        dst: u32,
        rows: u32,
        columns: u32,
    },
}

/// Independently schedulable work. Inputs and scratch share `data`; every run
/// restores these initial values, including scratch, before dispatching.
#[derive(Clone, Debug, Default)]
pub struct Job {
    pub data: Vec<f32>,
    pub instructions: Vec<Instruction>,
}

#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// Number of workgroups, each with 64 invocations. This is a tuning knob,
    /// not a promise of one resident workgroup per hardware compute unit.
    pub workers: u32,
    /// Must match whether the supplied GPU context enabled timestamps.
    pub gpu_timing: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            workers: 64,
            gpu_timing: false,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Execution {
    Persistent,
    /// Same interpreter and buffers, with one dispatch per instruction depth
    /// and a device memory barrier between depths. A scheduling control, not
    /// the production graph compiler's optimized kernels.
    Separate,
}

#[derive(Debug)]
pub struct Measurement {
    /// Encoding, reset, execution, readback transfer, submission and wait.
    pub wall: Duration,
    /// Compute pass only, including barriers between separate dispatches.
    pub compute: Option<Duration>,
    pub dispatches: u32,
}

#[derive(Debug)]
pub struct Statistics {
    pub completed_per_job: Vec<u32>,
    /// Per persistent worker: [jobs completed, instructions executed].
    /// Zero for the separate-dispatch control.
    pub per_worker: Vec<[u32; 2]>,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct EncodedInstruction {
    op: u32,
    src: u32,
    other: u32,
    dst: u32,
    len: u32,
    columns: u32,
    scale: f32,
    bias: f32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct EncodedJob {
    base: u32,
    first: u32,
    count: u32,
    padding: u32,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    jobs: u32,
    step: u32,
    padding: [u32; 2],
}

#[derive(blade_macros::ShaderData)]
struct Bindings {
    jobs: bg::BufferPiece,
    instructions: bg::BufferPiece,
    weights: bg::BufferPiece,
    arena: bg::BufferPiece,
    queue: bg::BufferPiece,
    workers: bg::BufferPiece,
    params: Params,
}

// Keep byte sizes and strided shader loops below u32 overflow as well as
// element offsets. This prototype uses one-dimensional dispatch grids.
const MAX_ELEMENTS: usize = (u32::MAX / 4) as usize;
const MAX_GROUPS: u32 = 65_535;

fn range(start: u32, len: u32, capacity: usize) -> Result<Range<u32>, String> {
    let end = start.checked_add(len).ok_or("element range overflow")?;
    if len == 0 || end as usize > capacity {
        return Err("empty or out-of-bounds element range".into());
    }
    Ok(start..end)
}

fn check_alias(input: Range<u32>, output: &Range<u32>, in_place: bool) -> Result<(), String> {
    if input.start < output.end && output.start < input.end && !(in_place && input == *output) {
        return Err(
            "overlapping input/output: only exact elementwise or RMS in-place writes are supported"
                .into(),
        );
    }
    Ok(())
}

impl Instruction {
    fn encode(&self, data_len: usize, weight_len: usize) -> Result<EncodedInstruction, String> {
        let mut encoded = EncodedInstruction {
            op: 0,
            src: 0,
            other: 0,
            dst: 0,
            len: 0,
            columns: 0,
            scale: 0.0,
            bias: 0.0,
        };
        match *self {
            Self::Affine {
                src,
                dst,
                len,
                scale,
                bias,
            } => {
                if !scale.is_finite() || !bias.is_finite() {
                    return Err("nonfinite affine constants".into());
                }
                encoded.src = src;
                encoded.dst = dst;
                encoded.len = len;
                encoded.scale = scale;
                encoded.bias = bias;
            }
            Self::Add { a, b, dst, len } => {
                encoded.op = 1;
                encoded.src = a;
                encoded.other = b;
                encoded.dst = dst;
                encoded.len = len;
            }
            Self::Relu { src, dst, len } => {
                encoded.op = 2;
                encoded.src = src;
                encoded.dst = dst;
                encoded.len = len;
            }
            Self::RmsNorm {
                src,
                dst,
                len,
                epsilon,
            } => {
                if !epsilon.is_finite() || epsilon <= 0.0 {
                    return Err("RMS epsilon must be finite and positive".into());
                }
                encoded.op = 3;
                encoded.src = src;
                encoded.dst = dst;
                encoded.len = len;
                encoded.bias = epsilon;
            }
            Self::MatVec {
                src,
                weights,
                dst,
                rows,
                columns,
            } => {
                let size = rows.checked_mul(columns).ok_or("matrix size overflow")?;
                range(weights, size, weight_len)?;
                encoded.op = 4;
                encoded.src = src;
                encoded.other = weights;
                encoded.dst = dst;
                encoded.len = rows;
                encoded.columns = columns;
            }
        }
        let output = range(encoded.dst, encoded.len, data_len)?;
        let input_len = if encoded.op == 4 {
            encoded.columns
        } else {
            encoded.len
        };
        check_alias(
            range(encoded.src, input_len, data_len)?,
            &output,
            encoded.op != 4,
        )?;
        if encoded.op == 1 {
            check_alias(range(encoded.other, encoded.len, data_len)?, &output, true)?;
        }
        Ok(encoded)
    }
}

/// Owns compiled pipelines, an immutable batch, and reusable device storage.
/// Construction validates all addresses and isolates each job's writable arena.
/// Execution is synchronous, so reset, readback and destruction cannot race it.
pub struct Executor {
    gpu: Arc<bg::Context>,
    options: Options,
    pipelines: [bg::ComputePipeline; 2],
    encoder: bg::CommandEncoder,
    buffers: Vec<bg::Buffer>,
    bindings: Bindings,
    initial: bg::Buffer,
    download: bg::Buffer,
    data_bytes: u64,
    queue_bytes: u64,
    worker_bytes: u64,
    ranges: Vec<Range<usize>>,
    depth: u32,
    has_run: bool,
}

impl Executor {
    pub fn new(
        gpu: Arc<bg::Context>,
        jobs: &[Job],
        weights: &[f32],
        options: Options,
    ) -> Result<Self, String> {
        if options.workers == 0 || options.workers > MAX_GROUPS || jobs.len() > MAX_GROUPS as usize
        {
            return Err("workers must be 1..=65535 and jobs at most 65535".into());
        }
        if weights.len() > MAX_ELEMENTS {
            return Err("weight buffer too large".into());
        }
        let mut data = Vec::new();
        let mut encoded_jobs = Vec::new();
        let mut instructions = Vec::new();
        let mut ranges = Vec::new();
        let mut depth = 1;
        for (index, job) in jobs.iter().enumerate() {
            let end = data
                .len()
                .checked_add(job.data.len())
                .ok_or("arena size overflow")?;
            let instruction_end = instructions
                .len()
                .checked_add(job.instructions.len())
                .ok_or("instruction count overflow")?;
            if end > MAX_ELEMENTS || instruction_end > MAX_ELEMENTS / 8 {
                return Err("batch buffers too large".into());
            }
            encoded_jobs.push(EncodedJob {
                base: data.len() as u32,
                first: instructions.len() as u32,
                count: job.instructions.len() as u32,
                padding: 0,
            });
            for (pc, instruction) in job.instructions.iter().enumerate() {
                instructions.push(
                    instruction
                        .encode(job.data.len(), weights.len())
                        .map_err(|error| format!("job {index}, instruction {pc}: {error}"))?,
                );
            }
            depth = depth.max(job.instructions.len() as u32);
            ranges.push(data.len()..end);
            data.extend_from_slice(&job.data);
        }
        let shader = gpu
            .try_create_shader(bg::ShaderDesc {
                source: include_str!("shaders/megakernel.wgsl"),
                naga_module: None,
            })
            .map_err(|error| error.to_string())?;
        let layout = <Bindings as bg::ShaderData>::layout();
        let pipelines = ["persistent", "separate"].map(|entry| {
            gpu.create_compute_pipeline(bg::ComputePipelineDesc {
                name: entry,
                data_layouts: &[&layout],
                compute: shader.at(entry),
            })
        });
        let mut encoder = gpu.create_command_encoder(bg::CommandEncoderDesc {
            name: "megakernel",
            buffer_count: 1,
            manual_barriers: false,
        });
        let mut buffers = Vec::new();
        let mut allocate = |name, size, memory| {
            let buffer = gpu.create_buffer(bg::BufferDesc { name, size, memory });
            buffers.push(buffer);
            buffer
        };
        let data_bytes = (data.len() as u64 * 4).max(4);
        let queue_bytes = (jobs.len() as u64 + 1) * 4;
        let worker_bytes = u64::from(options.workers) * 8;
        let job_buffer = allocate(
            "jobs",
            (encoded_jobs.len() as u64 * 16).max(16),
            bg::Memory::Device,
        );
        let instruction_buffer = allocate(
            "instructions",
            (instructions.len() as u64 * 32).max(32),
            bg::Memory::Device,
        );
        let weight_buffer = allocate(
            "weights",
            (weights.len() as u64 * 4).max(4),
            bg::Memory::Device,
        );
        let initial = allocate("initial data", data_bytes, bg::Memory::Device);
        let arena = allocate("arena", data_bytes, bg::Memory::Device);
        let queue = allocate("queue", queue_bytes, bg::Memory::Device);
        let workers = allocate("worker counts", worker_bytes, bg::Memory::Device);
        let download = allocate(
            "megakernel readback",
            data_bytes + queue_bytes + worker_bytes,
            bg::Memory::Download,
        );
        encoder.start();
        let mut uploads = Vec::new();
        {
            let mut transfer = encoder.transfer("upload megakernel batch");
            if data.is_empty() {
                transfer.fill_buffer(initial.into(), data_bytes, 0);
            }
            for (destination, bytes) in [
                (job_buffer, bytemuck::cast_slice(&encoded_jobs)),
                (instruction_buffer, bytemuck::cast_slice(&instructions)),
                (weight_buffer, bytemuck::cast_slice(weights)),
                (initial, bytemuck::cast_slice(&data)),
            ] {
                if bytes.is_empty() {
                    continue;
                }
                let upload = gpu.create_buffer(bg::BufferDesc {
                    name: "megakernel upload",
                    size: bytes.len() as u64,
                    memory: bg::Memory::Upload,
                });
                unsafe {
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), upload.data(), bytes.len());
                }
                gpu.sync_buffer(upload.into(), bytes.len() as u64, bg::BufferTarget::Data);
                transfer.copy_buffer_to_buffer(
                    upload.into(),
                    destination.into(),
                    bytes.len() as u64,
                );
                uploads.push(upload);
            }
        }
        let sync = gpu.submit(&mut encoder);
        assert!(
            gpu.wait_for(&sync, u32::MAX)
                .expect("megakernel upload failed")
        );
        for upload in uploads {
            gpu.destroy_buffer(upload);
        }
        Ok(Self {
            gpu,
            options,
            pipelines,
            encoder,
            buffers,
            bindings: Bindings {
                jobs: job_buffer.into(),
                instructions: instruction_buffer.into(),
                weights: weight_buffer.into(),
                arena: arena.into(),
                queue: queue.into(),
                workers: workers.into(),
                params: Params {
                    jobs: jobs.len() as u32,
                    step: 0,
                    padding: [0; 2],
                },
            },
            initial,
            download,
            data_bytes,
            queue_bytes,
            worker_bytes,
            ranges,
            depth,
            has_run: false,
        })
    }

    /// Restore inputs and queue, execute a finite batch, and wait for readback.
    pub fn run(&mut self, execution: Execution) -> Measurement {
        let start = Instant::now();
        self.encoder.start();
        {
            let mut transfer = self.encoder.transfer("reset megakernel batch");
            transfer.copy_buffer_to_buffer(
                self.initial.into(),
                self.bindings.arena,
                self.data_bytes,
            );
            transfer.fill_buffer(self.bindings.queue, self.queue_bytes, 0);
            transfer.fill_buffer(self.bindings.workers, self.worker_bytes, 0);
        }
        let dispatches = if execution == Execution::Persistent {
            1
        } else if self.ranges.is_empty() {
            0
        } else {
            self.depth
        };
        {
            let mut pass = self.encoder.compute("megakernel compute");
            for step in 0..dispatches {
                if step != 0 {
                    pass.barrier();
                }
                self.bindings.params.step = step;
                let pipeline_index = if execution == Execution::Persistent {
                    0
                } else {
                    1
                };
                let mut pipeline = pass.with(&self.pipelines[pipeline_index]);
                pipeline.bind(0, &self.bindings);
                pipeline.dispatch([
                    if execution == Execution::Persistent {
                        self.options.workers
                    } else {
                        self.ranges.len() as u32
                    },
                    1,
                    1,
                ]);
            }
        }
        {
            let mut transfer = self.encoder.transfer("megakernel readback");
            transfer.copy_buffer_to_buffer(
                self.bindings.arena,
                self.download.into(),
                self.data_bytes,
            );
            transfer.copy_buffer_to_buffer(
                self.bindings.queue,
                bg::BufferPiece {
                    buffer: self.download,
                    offset: self.data_bytes,
                },
                self.queue_bytes,
            );
            transfer.copy_buffer_to_buffer(
                self.bindings.workers,
                bg::BufferPiece {
                    buffer: self.download,
                    offset: self.data_bytes + self.queue_bytes,
                },
                self.worker_bytes,
            );
        }
        let sync = self.gpu.submit(&mut self.encoder);
        assert!(
            self.gpu
                .wait_for(&sync, u32::MAX)
                .expect("megakernel execution failed")
        );
        let wall = start.elapsed();
        let compute = self.options.gpu_timing.then(|| {
            self.encoder
                .last_timing()
                .pass_durations()
                .find(|&(name, _)| name == "megakernel compute")
                .expect("compute timestamp")
                .1
        });
        self.has_run = true;
        Measurement {
            wall,
            compute,
            dispatches,
        }
    }

    /// Read the complete private arena, including intermediates, after a run.
    pub fn read_job(&self, index: usize) -> Vec<f32> {
        assert!(self.has_run, "run the batch before reading it");
        let range = &self.ranges[index];
        unsafe {
            std::slice::from_raw_parts(
                self.download.data().cast::<f32>().add(range.start),
                range.len(),
            )
            .to_vec()
        }
    }

    pub fn statistics(&self) -> Statistics {
        assert!(self.has_run, "run the batch before reading it");
        unsafe {
            let queue = self
                .download
                .data()
                .add(self.data_bytes as usize)
                .cast::<u32>();
            let workers = self
                .download
                .data()
                .add((self.data_bytes + self.queue_bytes) as usize)
                .cast::<[u32; 2]>();
            Statistics {
                completed_per_job: std::slice::from_raw_parts(queue.add(1), self.ranges.len())
                    .to_vec(),
                per_worker: std::slice::from_raw_parts(workers, self.options.workers as usize)
                    .to_vec(),
            }
        }
    }
}

impl Drop for Executor {
    fn drop(&mut self) {
        self.gpu.destroy_command_encoder(&mut self.encoder);
        for pipeline in &mut self.pipelines {
            self.gpu.destroy_compute_pipeline(pipeline);
        }
        for buffer in self.buffers.drain(..) {
            self.gpu.destroy_buffer(buffer);
        }
    }
}
