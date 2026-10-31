//! Machine profile for measured search.
//!
//! A short calibration with Meganeura's own kernels, bounded by a slice of the
//! tuning budget. It measures how to time this machine (submission round
//! trip, pass and dispatch overhead) and roughly how long work takes there
//! (streaming bandwidth, matmul throughput, parallel width).
//! Nothing is keyed by device name or persisted; a caller can pass a profile
//! through [`crate::TuneCalibration::Profile`] instead.

use super::{Gpu, Pipelines, Session, create_gen_shader, create_profiled_pipeline};
use crate::compile::{BufferRef, Dispatch, ShaderEntry};
use crate::tune::{MachineProfile, median};
use blade_graphics as bg;
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};

/// `dst = src * a + b` over vec4 elements: one read and one write per element.
const STREAM_SHADER: &str = "
struct Params { count: u32, columns: u32, pad0: u32, pad1: u32 }
var<storage> src: array<vec4<f32>>;
var<storage, read_write> dst: array<vec4<f32>>;
var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    let index = id.y * params.columns + id.x;
    if index < params.count {
        dst[index] = src[index] * 1.0001 + vec4<f32>(0.5);
    }
}
";

/// A fixed amount of dependent arithmetic per workgroup, with 16 KiB of
/// workgroup memory so that residency resembles a tiled matmul.
const WIDTH_SHADER: &str = "
struct Params { iterations: u32, pad0: u32, pad1: u32, pad2: u32 }
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;
var<workgroup> tile: array<f32, 4096>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    for (var i = lid; i < 4096u; i += 256u) {
        tile[i] = f32(i) * 0.001;
    }
    workgroupBarrier();
    // Two dependent chains that converge instead of overflowing.
    var x = tile[lid];
    var y = tile[4095u - lid];
    for (var i = 0u; i < params.iterations; i++) {
        x = fma(x, 0.9999, 0.0001);
        y = fma(y, 0.9999, 0.0002);
    }
    workgroupBarrier();
    tile[lid] = x + y;
    workgroupBarrier();
    if lid == 0u {
        dst[group.x] = tile[(group.x + 1u) % 256u];
    }
}
";

#[derive(blade_macros::ShaderData)]
struct StreamData {
    src: bg::BufferPiece,
    dst: bg::BufferPiece,
    params: StreamParams,
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct StreamParams {
    count: u32,
    columns: u32,
    pad0: u32,
    pad1: u32,
}

#[derive(blade_macros::ShaderData)]
struct WidthData {
    dst: bg::BufferPiece,
    params: WidthParams,
}

#[derive(Clone, Copy, bytemuck::Zeroable, bytemuck::Pod)]
#[repr(C)]
struct WidthParams {
    iterations: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

/// Largest workgroup count the parallel-width probe tries.
const MAX_WIDTH: u32 = 8192;
/// Largest square matmul the throughput probe tries.
const MAX_MATMUL: usize = 2048;
/// Largest streaming buffer, per direction.
const MAX_STREAM_BYTES: usize = 64 << 20;

#[derive(Clone, Copy)]
struct Resources {
    budget: Duration,
    scratch_bytes: usize,
}

impl Resources {
    fn covers(self, other: Self) -> bool {
        self.budget >= other.budget && self.scratch_bytes >= other.scratch_bytes
    }
}

#[derive(Default)]
struct ProfileState {
    profile: Option<MachineProfile>,
    // Keep the resource frontier, not independent maxima: (long, small) and
    // (short, large) have not tried the combination (long, large).
    attempted: Vec<Resources>,
}

#[derive(Default)]
struct ProfileCache {
    state: Mutex<ProfileState>,
    measuring: Mutex<()>,
}

impl ProfileCache {
    fn get(&self) -> Option<MachineProfile> {
        self.state.lock().unwrap().profile.clone()
    }

    fn measure(
        &self,
        resources: Resources,
        calibrate: impl FnOnce(&mut MachineProfile),
    ) -> MachineProfile {
        // Do not spend this caller's search budget waiting for another search.
        // Readers keep seeing the last completed profile during an upgrade.
        let Ok(_measuring) = self.measuring.try_lock() else {
            return self.get().unwrap_or_default();
        };
        let mut state = self.state.lock().unwrap();
        let mut profile = state.profile.get_or_insert_default().clone();
        if complete(&profile)
            || resources.budget.is_zero()
            || resources.scratch_bytes <= super::tuning::ENCODER_SCRATCH_BYTES
            || state.attempted.iter().any(|&old| old.covers(resources))
        {
            return profile;
        }
        state.attempted.retain(|&old| !resources.covers(old));
        state.attempted.push(resources);
        drop(state);
        calibrate(&mut profile);
        self.state.lock().unwrap().profile = Some(profile.clone());
        profile
    }
}

fn complete(profile: &MachineProfile) -> bool {
    profile.submit_round_trip.is_some()
        && (!profile.gpu_timestamps || profile.pass_overhead.is_some())
        && profile.dispatch_overhead.is_some()
        && profile.parallel_width.is_some()
        && profile.bandwidth.is_some()
        && profile.matmul_flops.is_some()
}

type ProfileEntry = (Weak<Gpu>, bool, Arc<ProfileCache>);
static PROFILES: Mutex<Vec<ProfileEntry>> = Mutex::new(Vec::new());

/// Lookup only: callers asking whether probes are pending must not submit GPU
/// work, compile pipelines, or wait for an in-progress calibration.
pub(super) fn cached_profile(gpu: &Arc<Gpu>, timestamps: bool) -> Option<MachineProfile> {
    let profiles = PROFILES.lock().unwrap();
    profiles
        .iter()
        .find_map(|&(ref context, timing, ref profile)| {
            (std::ptr::eq(context.as_ptr(), Arc::as_ptr(gpu)) && timing == timestamps)
                .then(|| profile.get())
                .flatten()
        })
}

/// Reuse measurements per context/timing mode, within a soft budget. A partial
/// profile can fill missing measurements with a larger resource allowance;
/// repeated short searches do not keep paying for the same unfinished work.
/// Neither the global lock nor the profile-state lock is held during GPU work.
pub(crate) fn machine_profile(
    gpu: &Arc<Gpu>,
    timestamps: bool,
    budget: Duration,
    max_scratch_bytes: usize,
) -> (MachineProfile, Duration) {
    // A listed `Weak` keeps its context address from being reused.
    let slot = {
        let mut profiles = PROFILES.lock().unwrap();
        profiles.retain(|entry| entry.0.strong_count() != 0);
        if let Some(entry) = profiles
            .iter()
            .find(|entry| std::ptr::eq(entry.0.as_ptr(), Arc::as_ptr(gpu)) && entry.1 == timestamps)
        {
            Arc::clone(&entry.2)
        } else {
            let slot = Arc::new(ProfileCache::default());
            profiles.push((Arc::downgrade(gpu), timestamps, Arc::clone(&slot)));
            slot
        }
    };
    let mut elapsed = Duration::ZERO;
    let profile = slot.measure(
        Resources {
            budget,
            scratch_bytes: max_scratch_bytes,
        },
        |profile| {
            let start = Instant::now();
            calibrate(gpu, timestamps, budget, max_scratch_bytes, profile);
            elapsed = start.elapsed();
            log::info!("machine profile: {profile:?}");
        },
    );
    (profile, elapsed)
}

fn calibrate(
    gpu: &Arc<Gpu>,
    timestamps: bool,
    budget: Duration,
    max_scratch_bytes: usize,
    profile: &mut MachineProfile,
) {
    let Some(buffer_budget) = max_scratch_bytes.checked_sub(super::tuning::ENCODER_SCRATCH_BYTES)
    else {
        return;
    };
    if budget.is_zero() || buffer_budget == 0 {
        return;
    }
    let memory = gpu.memory_stats();
    if memory.budget != 0
        && super::tuning::ENCODER_SCRATCH_BYTES as u64
            > super::safe_device_memory_remaining(memory.usage, memory.budget)
    {
        return;
    }
    let _span = tracing::info_span!("machine_calibration").entered();
    let start = Instant::now();
    // Reserve time for the useful cost-model measurements. Width discovery
    // must not consume the whole allowance before bandwidth/matmul start.
    let mut bench = Bench::new(gpu, timestamps, start + budget / 8, buffer_budget, profile);
    profile.gpu_timestamps = bench.timestamps;
    if let Some(round_trip_ms) = bench.round_trip_ms {
        profile
            .submit_round_trip
            .get_or_insert(Duration::from_secs_f64(round_trip_ms / 1e3));
    }
    bench.overheads(profile);
    bench.deadline = start + budget / 4;
    if profile.parallel_width.is_none() {
        profile.parallel_width = bench.parallel_width();
    }
    bench.deadline = start + budget / 2;
    // Retry a cold pipeline after the first short slice.
    bench.overheads(profile);
    if profile.bandwidth.is_none() {
        profile.bandwidth = bench.bandwidth();
    }
    bench.deadline = start + budget;
    if profile.matmul_flops.is_none() {
        bench.matmul(profile);
    }
    bench.finish();
    profile.elapsed += start.elapsed();
}

/// Timing harness: GPU pass timestamps when the context records them, the
/// host clock around one submission otherwise.
struct Bench<'a> {
    gpu: &'a Arc<Gpu>,
    encoder: bg::CommandEncoder,
    timestamps: bool,
    deadline: Instant,
    /// Empty submit-and-wait, in milliseconds, removed from host timings.
    round_trip_ms: Option<f64>,
    /// Measured timed-pass overhead, in milliseconds, removed from timestamps.
    pass_ms: f64,
    buffers: Vec<bg::Buffer>,
    pipelines: Vec<bg::ComputePipeline>,
    /// Index of the streaming pipeline, compiled on first use.
    stream: Option<usize>,
    max_scratch_bytes: usize,
}

impl<'a> Bench<'a> {
    fn new(
        gpu: &'a Arc<Gpu>,
        timestamps: bool,
        deadline: Instant,
        max_scratch_bytes: usize,
        profile: &MachineProfile,
    ) -> Self {
        let encoder = gpu.create_command_encoder(bg::CommandEncoderDesc {
            name: "machine_calibration",
            buffer_count: 1,
            manual_barriers: false,
        });
        let mut bench = Self {
            gpu,
            encoder,
            timestamps: timestamps && gpu.capabilities().timing,
            deadline,
            round_trip_ms: profile
                .submit_round_trip
                .map(|time| time.as_secs_f64() * 1e3),
            pass_ms: profile.pass_overhead.unwrap_or_default().as_secs_f64() * 1e3,
            buffers: Vec::new(),
            pipelines: Vec::new(),
            stream: None,
            max_scratch_bytes,
        };
        if profile.submit_round_trip.is_some() {
            return bench;
        }
        let mut trips = Vec::new();
        for _ in 0..4 {
            if bench.expired() {
                break;
            }
            let start = Instant::now();
            bench.encoder.start();
            let sync = bench.gpu.submit(&mut bench.encoder);
            let _ = super::wait_for_timed_encoder(bench.gpu, &sync, &mut bench.encoder, false);
            trips.push(start.elapsed().as_secs_f64() * 1e3);
        }
        bench.round_trip_ms = trips.into_iter().reduce(f64::min);
        bench
    }

    fn expired(&self) -> bool {
        Instant::now() >= self.deadline
    }

    fn buffer(&mut self, name: &'static str, bytes: usize) -> bg::Buffer {
        let buffer = self.gpu.create_buffer(bg::BufferDesc {
            name,
            size: bytes as u64,
            memory: bg::Memory::Device,
        });
        // Normal, finite values: uninitialized memory can hold denormals or NaNs.
        self.encoder.start();
        self.encoder
            .transfer("calibration_fill")
            .fill_buffer(buffer.into(), bytes as u64, 0x3c);
        let sync = self.gpu.submit(&mut self.encoder);
        let _ = super::wait_for_timed_encoder(self.gpu, &sync, &mut self.encoder, false);
        self.buffers.push(buffer);
        buffer
    }

    fn release(&mut self, buffers: &[bg::Buffer]) {
        for &buffer in buffers {
            self.gpu.destroy_buffer(buffer);
        }
        self.buffers.retain(|buffer| !buffers.contains(buffer));
    }

    fn stream_pipeline(&mut self) -> Option<usize> {
        if self.stream.is_none() {
            self.stream = self.pipeline::<StreamData>("calibration_stream", STREAM_SHADER);
        }
        self.stream
    }

    fn pipeline<D: bg::ShaderData>(&mut self, name: &str, source: &str) -> Option<usize> {
        if self.expired() {
            return None;
        }
        let module = crate::codegen::ShaderModule::new(source);
        let shader = create_gen_shader(self.gpu, module)
            .map_err(|error| log::warn!("calibration shader {name} rejected: {error}"))
            .ok()?;
        let layout = <D as bg::ShaderData>::layout();
        self.pipelines.push(create_profiled_pipeline(
            self.gpu,
            name.to_owned(),
            &layout,
            shader.at("main"),
        ));
        Some(self.pipelines.len() - 1)
    }

    /// Room for `bytes` more without approaching the device budget.
    fn fits_memory(&self, bytes: usize) -> bool {
        if bytes > self.max_scratch_bytes {
            return false;
        }
        let stats = self.gpu.memory_stats();
        stats.budget == 0
            || (bytes as u64) <= super::safe_device_memory_remaining(stats.usage, stats.budget) / 2
    }

    /// Milliseconds of each of `count` runs of `encode`. With timestamps, all
    /// runs share one submission, one pass each; otherwise each run is its own
    /// submission, timed on the host less the empty round trip.
    fn time(
        &mut self,
        count: usize,
        mut encode: impl FnMut(&mut bg::CommandEncoder, &[bg::ComputePipeline]),
    ) -> Vec<f64> {
        if self.expired() {
            return Vec::new();
        }
        if self.timestamps {
            self.encoder.start();
            for _ in 0..count {
                encode(&mut self.encoder, &self.pipelines);
            }
            let sync = self.gpu.submit(&mut self.encoder);
            let timings = super::wait_for_timed_encoder(self.gpu, &sync, &mut self.encoder, true)
                .ok()
                .flatten();
            if let Some(timings) = timings.filter(|t| t.passes.len() == count) {
                return timings
                    .pass_durations()
                    .map(|(_, duration)| (duration.as_secs_f64() * 1e3 - self.pass_ms).max(0.0))
                    .collect();
            }
            return Vec::new();
        }
        let mut samples = Vec::new();
        for _ in 0..count {
            if self.expired() {
                break;
            }
            let start = Instant::now();
            self.encoder.start();
            encode(&mut self.encoder, &self.pipelines);
            let sync = self.gpu.submit(&mut self.encoder);
            let _ = super::wait_for_timed_encoder(self.gpu, &sync, &mut self.encoder, false);
            samples.push(
                (start.elapsed().as_secs_f64() * 1e3 - self.round_trip_ms.unwrap_or_default())
                    .max(0.0),
            );
        }
        samples
    }

    /// Duration worth timing: long enough that the clock and submission jitter
    /// are small next to it.
    fn target_ms(&self) -> f64 {
        if self.timestamps {
            (20.0 * self.pass_ms).clamp(0.25, 8.0)
        } else {
            (20.0 * self.round_trip_ms.unwrap_or_default()).clamp(1.0, 8.0)
        }
    }

    fn record_stream(
        encoder: &mut bg::CommandEncoder,
        pipelines: &[bg::ComputePipeline],
        stream: usize,
        data: &StreamData,
        grid: [u32; 3],
        dispatches: u32,
    ) {
        let mut pass = encoder.compute("calibration_stream");
        for dispatch in 0..dispatches {
            if dispatch != 0 {
                pass.barrier();
            }
            let mut pc = pass.with(&pipelines[stream]);
            pc.bind(0, data);
            pc.dispatch(grid);
        }
    }

    /// Overhead of a timed pass and of one more dispatch inside a pass, from
    /// a one-workgroup streaming dispatch.
    fn overheads(&mut self, profile: &mut MachineProfile) {
        const DISPATCHES: u32 = 32;
        if profile.dispatch_overhead.is_some()
            && (!self.timestamps || profile.pass_overhead.is_some())
        {
            return;
        }
        let Some(stream) = self.stream_pipeline() else {
            return;
        };
        let bytes = 256 * 16;
        if self.expired() || !self.fits_memory(2 * bytes) {
            return;
        }
        let (src, dst) = (
            self.buffer("calibration_src", bytes),
            self.buffer("calibration_dst", bytes),
        );
        let data = StreamData {
            src: src.into(),
            dst: dst.into(),
            params: StreamParams {
                count: 256,
                columns: 256,
                pad0: 0,
                pad1: 0,
            },
        };
        let single = self.time(DISPATCHES as usize, |encoder, pipelines| {
            Self::record_stream(encoder, pipelines, stream, &data, [1, 1, 1], 1)
        });
        let batched = self.time(3, |encoder, pipelines| {
            Self::record_stream(encoder, pipelines, stream, &data, [1, 1, 1], DISPATCHES)
        });
        if !single.is_empty() && !batched.is_empty() {
            let single = median(&single);
            if self.timestamps {
                let overhead = profile
                    .pass_overhead
                    .get_or_insert(Duration::from_secs_f64(single / 1e3));
                self.pass_ms = overhead.as_secs_f64() * 1e3;
            }
            let extra = (median(&batched) - single).max(0.0) / f64::from(DISPATCHES - 1);
            profile
                .dispatch_overhead
                .get_or_insert(Duration::from_secs_f64(extra / 1e3));
        }
        self.release(&[src, dst]);
    }

    /// Workgroups that run concurrently before a fixed-work kernel slows down.
    fn parallel_width(&mut self) -> Option<u32> {
        if self.expired() || !self.fits_memory(MAX_WIDTH as usize * 4) {
            return None;
        }
        let width = self.pipeline::<WidthData>("calibration_width", WIDTH_SHADER)?;
        if self.expired() {
            return None;
        }
        let dst = self.buffer("calibration_width", MAX_WIDTH as usize * 4);
        let measure = |bench: &mut Self, groups: u32, iterations: u32| {
            let samples = bench.time(3, |encoder, pipelines| {
                let mut pass = encoder.compute("calibration_width");
                let mut pc = pass.with(&pipelines[width]);
                pc.bind(
                    0,
                    &WidthData {
                        dst: dst.into(),
                        params: WidthParams {
                            iterations,
                            pad0: 0,
                            pad1: 0,
                            pad2: 0,
                        },
                    },
                );
                pc.dispatch([groups, 1, 1]);
            });
            (!samples.is_empty()).then(|| median(&samples))
        };
        let result = (|| {
            // Lengthen one workgroup's work until it is worth timing.
            let target = self.target_ms();
            let mut iterations = 256;
            let mut single = measure(self, 1, iterations)?;
            while single < target && iterations < 1 << 22 && !self.expired() {
                iterations *= 2;
                single = measure(self, 1, iterations)?;
            }
            let mut width = 1;
            let mut groups = 2;
            while groups <= MAX_WIDTH && !self.expired() {
                let elapsed = measure(self, groups, iterations)?;
                if elapsed <= 1.25 * single {
                    width = groups;
                }
                if elapsed > 1.5 * single {
                    break;
                }
                groups *= 2;
            }
            Some(width)
        })();
        // Also release after missing timings or a deadline: the next stage
        // gets the same scratch allowance, not an additional allocation.
        self.release(&[dst]);
        result
    }

    /// Streaming bandwidth at the first size worth timing.
    fn bandwidth(&mut self) -> Option<f64> {
        let stream = self.stream_pipeline()?;
        let target = self.target_ms();
        let mut bytes = 1 << 16;
        let mut rate = None;
        while bytes <= MAX_STREAM_BYTES && !self.expired() && self.fits_memory(2 * bytes) {
            let (src, dst) = (
                self.buffer("calibration_src", bytes),
                self.buffer("calibration_dst", bytes),
            );
            let count = (bytes / 16) as u32;
            let groups = count.div_ceil(256);
            let columns_groups = groups.min(32768);
            let data = StreamData {
                src: src.into(),
                dst: dst.into(),
                params: StreamParams {
                    count,
                    columns: columns_groups * 256,
                    pad0: 0,
                    pad1: 0,
                },
            };
            let measure = |bench: &mut Self, repeats: u32| {
                bench.time(3, |encoder, pipelines| {
                    Self::record_stream(
                        encoder,
                        pipelines,
                        stream,
                        &data,
                        [columns_groups, groups.div_ceil(columns_groups), 1],
                        repeats,
                    );
                })
            };
            let samples = measure(self, 1);
            if samples.is_empty() {
                self.release(&[src, dst]);
                return rate;
            }
            let mut elapsed = median(&samples);
            if elapsed == 0.0 && !self.timestamps {
                // Amortize a stale cold-submit estimate over enough useful work
                // to avoid clipping every host sample to zero.
                const REPEATS: u32 = 32;
                let samples = measure(self, REPEATS);
                if !samples.is_empty() {
                    elapsed = median(&samples) / f64::from(REPEATS);
                }
            }
            self.release(&[src, dst]);
            if elapsed > 0.0 {
                rate = Some(2.0 * bytes as f64 / (elapsed / 1e3));
            }
            if elapsed >= target {
                break;
            }
            bytes *= 4;
        }
        rate
    }

    /// Throughput of the ordinary f32 matmul at the first size worth timing,
    /// scaled to a fully occupied machine.
    fn matmul(&mut self, profile: &mut MachineProfile) {
        let target = self.target_ms();
        let mut size = 64;
        while size <= MAX_MATMUL && !self.expired() && self.fits_memory(3 * size * size * 4) {
            let (dispatch, knobs) = matmul_dispatch(size);
            let key = Pipelines::key(&dispatch);
            let pipeline = match Pipelines::compile(self.gpu, knobs, None, &dispatch, &key, None) {
                Ok(pipeline) => pipeline,
                Err(error) => {
                    log::warn!("calibration matmul rejected: {error}");
                    return;
                }
            };
            if self.expired() {
                break;
            }
            let bytes = size * size * 4;
            let buffers = [
                self.buffer("calibration_a", bytes),
                self.buffer("calibration_b", bytes),
                self.buffer("calibration_c", bytes),
            ];
            let pieces: Vec<bg::BufferPiece> = buffers.iter().map(|&b| b.into()).collect();
            let samples = self.time(3, |encoder, _| {
                let mut pass = encoder.compute("calibration_matmul");
                let mut pc = pass.with(&pipeline.raw);
                Session::bind_dispatch(&pieces, &dispatch, &mut pc);
                pc.dispatch(dispatch.workgroups);
            });
            let elapsed = (!samples.is_empty()).then(|| median(&samples));
            let done = elapsed.is_none_or(|elapsed| elapsed >= target)
                || size == MAX_MATMUL
                || self.expired();
            if let Some(elapsed) = elapsed.filter(|&elapsed| elapsed > 0.0) {
                let workgroups = dispatch.workgroups.iter().map(|&w| u64::from(w)).product();
                let flops = 2.0 * (size as f64).powi(3);
                profile.matmul_flops =
                    Some(flops / (elapsed / 1e3) / profile.wave_utilization(workgroups));
            }
            self.release(&buffers);
            if done {
                break;
            }
            size *= 2;
        }
    }

    fn finish(mut self) {
        let buffers = std::mem::take(&mut self.buffers);
        self.release(&buffers);
        for mut pipeline in self.pipelines.drain(..) {
            self.gpu.destroy_compute_pipeline(&mut pipeline);
        }
        self.gpu.destroy_command_encoder(&mut self.encoder);
    }
}

/// The ordinary compiled dispatch for a square f32 product, with scratch
/// bindings A=0, B=1, C=2.
fn matmul_dispatch(size: usize) -> (Dispatch, crate::compile::TuningKnobs) {
    let mut graph = crate::Graph::new();
    let a = graph.input("a", &[size, size]);
    let b = graph.input("b", &[size, size]);
    let y = graph.matmul(a, b);
    graph.set_outputs(vec![y]);
    let plan = crate::compile::compile(&graph);
    let mut dispatch = plan
        .dispatches
        .into_iter()
        .find(|dispatch| dispatch.shader == ShaderEntry::MatMul)
        .expect("a square product compiles to MatMul");
    dispatch.input_buffers = vec![BufferRef(0), BufferRef(1)];
    dispatch.output_buffer = BufferRef(2);
    (dispatch, plan.knobs)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calibration_dispatch_is_the_ordinary_square_product() {
        let (dispatch, _) = matmul_dispatch(128);
        assert_eq!(dispatch.mnk(), Some((128, 128, 128)));
        assert_eq!(dispatch.kernel, crate::compile::Kernel::Default);
    }

    #[test]
    fn partial_profiles_retry_new_resource_combinations_without_repeating_work() {
        let cache = ProfileCache::default();
        let resources = |ms, mib: usize| Resources {
            budget: Duration::from_millis(ms),
            scratch_bytes: mib << 20,
        };
        let unused = |_: &mut MachineProfile| panic!("calibration must be reused");
        assert!(cache.get().is_none());
        assert_eq!(
            cache.measure(resources(0, 64), unused),
            MachineProfile::default()
        );
        assert_eq!(
            cache.measure(resources(100, 1), unused),
            MachineProfile::default()
        );
        let first = cache.measure(resources(100, 2), |profile| {
            profile.bandwidth = Some(1e9);
        });
        assert_eq!(cache.measure(resources(100, 2), unused), first);
        assert_eq!(cache.measure(resources(50, 2), unused), first);
        // More memory with less time is a new opportunity, but must not erase
        // the previous time allowance from the frontier.
        let second = cache.measure(resources(50, 4), |profile| {
            assert_eq!(*profile, first);
            assert_eq!(cache.get(), Some(first.clone()));
            // Another search can read or reuse the prior profile during this
            // upgrade; it neither waits nor starts a concurrent calibration.
            assert_eq!(cache.measure(resources(1000, 64), unused), first);
            profile.matmul_flops = Some(2e9);
        });
        assert_eq!(cache.measure(resources(100, 2), unused), second);
        // Independent maxima would incorrectly suppress this combination.
        let third = cache.measure(resources(100, 4), |profile| {
            assert_eq!(*profile, second);
            profile.parallel_width = Some(8);
        });
        assert_eq!(cache.measure(resources(100, 4), unused), third);
        let complete = cache.measure(resources(200, 4), |profile| {
            profile.submit_round_trip = Some(Duration::from_micros(50));
            profile.dispatch_overhead = Some(Duration::from_micros(1));
        });
        assert_eq!(cache.measure(resources(1000, 64), unused), complete);
    }

    #[test]
    fn larger_allowances_fill_missing_measurements_and_preserve_existing_ones() {
        let gpu = Arc::new(
            crate::init_gpu_context_with(crate::GpuOptions {
                timing: false,
                ..crate::GpuOptions::from_env()
            })
            .unwrap(),
        );
        assert!(cached_profile(&gpu, false).is_none());
        // An expired first slice must not cache an unmeasured zero round trip.
        let (empty, _) = machine_profile(&gpu, false, Duration::from_nanos(1), 64 << 20);
        assert!(empty.submit_round_trip.is_none(), "{empty:?}");
        // Enough for overhead/width and the smallest matmul, but not even the
        // smallest bandwidth probe after reserving the encoder's scratch.
        let resources = super::super::tuning::ENCODER_SCRATCH_BYTES + (64 << 10);
        let (first, _) = machine_profile(&gpu, false, Duration::from_secs(1), resources);
        assert!(first.bandwidth.is_none(), "{first:?}");
        assert_eq!(cached_profile(&gpu, false), Some(first.clone()));
        let (same, elapsed) = machine_profile(&gpu, false, Duration::from_secs(1), resources);
        assert_eq!(same, first);
        assert!(elapsed.is_zero());
        let (second, elapsed) = machine_profile(&gpu, false, Duration::from_secs(2), 64 << 20);
        assert!(complete(&second), "{second:?}");
        assert!(!elapsed.is_zero());
        assert!(second.elapsed >= first.elapsed);
        assert_eq!(cached_profile(&gpu, false), Some(second.clone()));
        macro_rules! preserved {
            ($field:ident) => {
                if first.$field.is_some() {
                    assert_eq!(second.$field, first.$field);
                }
            };
        }
        preserved!(submit_round_trip);
        preserved!(dispatch_overhead);
        preserved!(parallel_width);
        preserved!(matmul_flops);
        let (same, elapsed) = machine_profile(&gpu, false, Duration::from_secs(3), 128 << 20);
        assert_eq!(same, second);
        assert!(elapsed.is_zero());
    }

    #[test]
    #[ignore = "GPU calibration; run with --ignored to inspect a profile"]
    fn calibration_measures_within_its_budget() {
        let gpu = crate::reference::gpu::shared_context();
        let budget = Duration::from_millis(500);
        let mut profile = MachineProfile::default();
        calibrate(
            &gpu,
            crate::GpuOptions::from_env().timing,
            budget,
            64 << 20,
            &mut profile,
        );
        eprintln!("{profile:#?}");
        // Calibration is soft-bounded: one in-flight measurement may overrun.
        assert!(profile.elapsed < budget * 4, "{profile:?}");
        assert!(profile.bandwidth.is_some_and(|rate| rate > 0.0));
        assert!(profile.matmul_flops.is_some_and(|rate| rate > 0.0));
        assert!(profile.parallel_width.is_some_and(|width| width >= 1));
    }
}
