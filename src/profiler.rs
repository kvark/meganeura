//! Profiling infrastructure producing Perfetto binary traces (`.pftrace`).
//!
//! CPU-side work is captured automatically via [`tracing`] spans. GPU pass
//! durations come from blade-graphics hardware timestamp queries. Both land
//! on separate tracks in the resulting trace, viewable in
//! [Perfetto UI](https://ui.perfetto.dev).
//!
//! # Quick start
//!
//! ```ignore
//! meganeura::profiler::init();          // sets up tracing subscriber
//! // ... build session, train ...
//! meganeura::profiler::save("trace.pftrace").unwrap();
//! ```

use serde::Serialize;
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    path::Path,
    sync::{Arc, Mutex, OnceLock},
    time::{Duration, Instant},
};
use tracing::{Subscriber, span};
use tracing_subscriber::{Layer, layer::Context, prelude::*, registry::LookupSpan};

// ---- Track IDs ----

const CPU_TRACK_UUID: u64 = 1;
const GPU_TRACK_UUID: u64 = 2;

// ---- Trace event model ----

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum EventKind {
    SliceBegin = 1,
    SliceEnd = 2,
    Instant = 3,
}

struct TraceEvent {
    name: String,
    timestamp_ns: u64,
    track_uuid: u64,
    kind: EventKind,
}

// ---- Shared profiler state ----

struct ProfilerInner {
    epoch: Instant,
    events: Vec<TraceEvent>,
}

impl ProfilerInner {
    fn now_ns(&self) -> u64 {
        self.epoch.elapsed().as_nanos() as u64
    }
}

static PROFILER: OnceLock<Arc<Mutex<ProfilerInner>>> = OnceLock::new();

fn get_or_init() -> &'static Arc<Mutex<ProfilerInner>> {
    PROFILER.get_or_init(|| {
        Arc::new(Mutex::new(ProfilerInner {
            epoch: Instant::now(),
            events: Vec::with_capacity(8192),
        }))
    })
}

// ---- Public API ----

/// Initialize profiling: installs a global [`tracing`] subscriber that records
/// spans as Perfetto slice events on the CPU track.
///
/// Safe to call multiple times (subsequent calls are no-ops).
/// Must be called *before* any tracing spans you want captured.
pub fn init() {
    let inner = get_or_init().clone();
    let layer = ProfileLayer { inner };
    let subscriber = tracing_subscriber::registry().with(layer);
    // Ignore error if a subscriber is already set.
    let _ = tracing::subscriber::set_global_default(subscriber);
}

/// Record GPU pass timing events on the GPU track.
///
/// `submit_offset_ns` is the nanosecond offset (relative to profiler epoch)
/// when the GPU work was submitted. Pass durations are laid out sequentially
/// starting from that offset.
pub fn record_gpu_passes(submit_offset_ns: u64, passes: &[(String, Duration)]) {
    if let Some(inner) = PROFILER.get() {
        let mut guard = inner.lock().unwrap();
        let mut offset = submit_offset_ns;
        for &(ref name, dur) in passes {
            guard.events.push(TraceEvent {
                name: name.clone(),
                timestamp_ns: offset,
                track_uuid: GPU_TRACK_UUID,
                kind: EventKind::SliceBegin,
            });
            offset += dur.as_nanos() as u64;
            guard.events.push(TraceEvent {
                name: name.clone(),
                timestamp_ns: offset,
                track_uuid: GPU_TRACK_UUID,
                kind: EventKind::SliceEnd,
            });
        }
    }
}

/// Record a single CPU event (for use outside tracing spans).
pub fn record_instant(name: &str) {
    if let Some(inner) = PROFILER.get() {
        let mut guard = inner.lock().unwrap();
        let ts = guard.now_ns();
        guard.events.push(TraceEvent {
            name: name.to_string(),
            timestamp_ns: ts,
            track_uuid: CPU_TRACK_UUID,
            kind: EventKind::Instant,
        });
    }
}

/// Return the nanosecond offset from the profiler epoch (for GPU timing placement).
pub fn now_ns() -> u64 {
    PROFILER
        .get()
        .map(|inner| inner.lock().unwrap().now_ns())
        .unwrap_or(0)
}

/// Number of recorded events (including both CPU spans and GPU passes).
pub fn event_count() -> usize {
    PROFILER
        .get()
        .map(|inner| inner.lock().unwrap().events.len())
        .unwrap_or(0)
}

/// Write all collected events to a Perfetto `.pftrace` binary trace file.
pub fn save(path: impl AsRef<Path>) -> std::io::Result<()> {
    let inner = PROFILER
        .get()
        .ok_or_else(|| std::io::Error::other("profiler not initialized"))?;
    let guard = inner.lock().unwrap();
    write_pftrace(path.as_ref(), &guard.events)
}

// ---- Structured gap profiles ----

/// Options for [`capture_session_profile`].
#[derive(Clone, Debug)]
pub struct CaptureOptions {
    /// Number of separately timestamped executions to retain.
    pub samples: usize,
    /// Median wall time from the normal, uninstrumented benchmark protocol.
    ///
    /// When provided, the artifact reports how much one-pass-per-dispatch
    /// profiling perturbed the workload.
    pub unprofiled_median_ms: Option<f64>,
    /// Query driver-reported pipeline statistics such as register and spill
    /// counts where the backend exposes them.
    pub include_pipeline_statistics: bool,
}

impl Default for CaptureOptions {
    fn default() -> Self {
        Self {
            samples: 3,
            unprofiled_median_ms: None,
            include_pipeline_statistics: true,
        }
    }
}

/// Machine-readable profile of one compiled session and execution shape.
#[derive(Clone, Debug, Serialize)]
pub struct SessionProfile {
    pub schema_version: u32,
    pub timing_contract: String,
    pub device: ProfileDevice,
    pub plan: ProfilePlan,
    pub measurement: ProfileMeasurement,
    pub families: Vec<FamilyProfile>,
    pub dispatches: Vec<DispatchProfile>,
    pub pipeline_statistics: Vec<PipelineProfile>,
}

#[derive(Clone, Debug, Serialize)]
pub struct ProfileDevice {
    pub backend: String,
    pub device_name: String,
    pub driver_name: String,
    pub driver_info: String,
    pub software_emulated: bool,
}

#[derive(Clone, Debug, Serialize)]
pub struct ProfilePlan {
    pub dispatch_count: usize,
    pub forward_dispatch_count: usize,
    pub backward_dispatch_count: usize,
    pub barrier_group_count: usize,
    pub logical_buffer_bytes: usize,
    pub allocated_buffer_bytes: usize,
    pub device_local_bytes: usize,
    pub physical_allocation_count: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct ProfileMeasurement {
    pub sample_count: usize,
    pub unprofiled_median_ms: Option<f64>,
    pub profiled_wall_samples_ms: Vec<f64>,
    pub profiled_wall_median_ms: f64,
    pub gpu_total_samples_ms: Vec<f64>,
    pub gpu_total_median_ms: f64,
    /// Profiled wall median divided by the unprofiled benchmark median.
    pub instrumentation_wall_ratio: Option<f64>,
    /// Timestamped GPU total divided by profiled wall time.
    pub timestamped_gpu_share_of_profiled_wall_pct: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct FamilyProfile {
    pub phase: String,
    pub family: String,
    pub dispatch_count: usize,
    pub timing_samples_ms: Vec<f64>,
    pub median_ms: f64,
    /// Sum of each member dispatch's median. Shares use this additive value
    /// so the family percentages sum to 100%.
    pub dispatch_median_sum_ms: f64,
    pub share_of_dispatch_median_sum_pct: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct DispatchProfile {
    pub index: usize,
    pub phase: String,
    pub family: String,
    pub shader: String,
    pub label: String,
    pub timestamp_label: String,
    pub pipeline: String,
    pub workgroups: [u32; 3],
    pub workgroup_count: u64,
    pub input_buffer_bytes: usize,
    pub output_buffer_bytes: usize,
    pub cooperative: bool,
    pub small_tile: bool,
    pub requires_full_precision: bool,
    pub weight_format: String,
    pub has_prologue: bool,
    pub has_epilogue: bool,
    pub timing_samples_ms: Vec<f64>,
    pub median_ms: f64,
    pub p25_ms: f64,
    pub p75_ms: f64,
    pub share_of_dispatch_median_sum_pct: f64,
}

#[derive(Clone, Debug, Serialize)]
pub struct PipelineProfile {
    pub pipeline: String,
    pub executables: Vec<PipelineExecutableProfile>,
}

#[derive(Clone, Debug, Serialize)]
pub struct PipelineExecutableProfile {
    pub name: String,
    pub statistics: Vec<PipelineStatisticProfile>,
}

#[derive(Clone, Debug, Serialize)]
pub struct PipelineStatisticProfile {
    pub name: String,
    pub description: String,
    pub value: f64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ProfileError {
    ZeroSamples,
    TooManyDispatches {
        count: usize,
        limit: usize,
    },
    MissingGpuTimings {
        sample: usize,
    },
    TimingCount {
        sample: usize,
        expected: usize,
        actual: usize,
    },
}

impl fmt::Display for ProfileError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Self::ZeroSamples => f.write_str("profile sample count must be positive"),
            Self::TooManyDispatches { count, limit } => write!(
                f,
                "session has {count} dispatches, exceeding Blade's {limit}-pass timestamp limit"
            ),
            Self::MissingGpuTimings { sample } => write!(
                f,
                "no GPU timings resolved for profile sample {sample}; set \
                 MEGANEURA_GPU_TIMING=1 before constructing the session"
            ),
            Self::TimingCount {
                sample,
                expected,
                actual,
            } => write!(
                f,
                "profile sample {sample} resolved {actual} timed passes, expected {expected}; \
                 disable runtime-appended optimizer, gradient-accumulation, and \
                 gradient-clipping passes before capture"
            ),
        }
    }
}

impl std::error::Error for ProfileError {}

/// Capture repeated, structured per-dispatch GPU timings for a session.
///
/// Set `MEGANEURA_GPU_TIMING=1` before the session creates its Blade context.
/// Each retained execution runs in one-compute-pass-per-dispatch mode. Two
/// normal executions then advance Blade's command-buffer ring so the hardware
/// timestamps can be read. The returned artifact retains raw samples, reports
/// instrumentation overhead against an optional normal benchmark median, and
/// aggregates dispatches by forward/backward phase and coarse kernel family.
///
/// `prepare` is called immediately before every execution, including the two
/// ring-advance executions. It should restore inputs and any state that the
/// workload mutates.
///
/// The structured dispatch table describes the compiled execution plan.
/// Capture with optimizer, gradient-accumulation, and gradient-clipping passes
/// disabled: those runtime-appended passes do not have plan metadata, and the
/// collector rejects their additional timestamps instead of misattributing
/// them to plan dispatches.
pub fn capture_session_profile(
    session: &mut crate::runtime::Session,
    mut prepare: impl FnMut(&mut crate::runtime::Session),
    options: CaptureOptions,
) -> Result<SessionProfile, ProfileError> {
    if options.samples == 0 {
        return Err(ProfileError::ZeroSamples);
    }

    let dispatch_count = session.plan().dispatches.len();
    let pass_limit = blade_graphics::limits::PASS_COUNT;
    if dispatch_count > pass_limit {
        return Err(ProfileError::TooManyDispatches {
            count: dispatch_count,
            limit: pass_limit,
        });
    }

    let mut timing_samples = vec![Vec::with_capacity(options.samples); dispatch_count];
    let mut timestamp_labels = vec![String::new(); dispatch_count];
    let mut profiled_wall_samples_ms = Vec::with_capacity(options.samples);
    let mut gpu_total_samples_ms = Vec::with_capacity(options.samples);

    for sample in 0..options.samples {
        session.set_profiling(true);
        prepare(session);
        let wall_start = Instant::now();
        session.step();
        session.wait();
        profiled_wall_samples_ms.push(wall_start.elapsed().as_secs_f64() * 1000.0);

        // Blade command encoders retain two command buffers. Reuse the
        // profiled command buffer after two ordinary submissions to resolve
        // its query pool without timing two more instrumented executions.
        session.set_profiling(false);
        for _ in 0..2 {
            prepare(session);
            session.step();
            session.wait();
        }

        let timings = session.gpu_timings();
        if timings.is_empty() {
            return Err(ProfileError::MissingGpuTimings { sample });
        }
        if timings.len() != dispatch_count {
            return Err(ProfileError::TimingCount {
                sample,
                expected: dispatch_count,
                actual: timings.len(),
            });
        }

        let mut total_ms = 0.0;
        for (index, (name, duration)) in timings.into_iter().enumerate() {
            let duration_ms = duration.as_secs_f64() * 1000.0;
            total_ms += duration_ms;
            timing_samples[index].push(duration_ms);
            if sample == 0 {
                timestamp_labels[index] = name;
            }
        }
        gpu_total_samples_ms.push(total_ms);
    }
    session.set_profiling(false);

    let plan = session.plan();
    let pipeline_keys = session.dispatch_pipeline_keys();
    let loss_dispatch = plan.loss_buffer.and_then(|loss| {
        plan.dispatches.iter().rposition(|dispatch| {
            dispatch.output_buffer == loss || dispatch.extra_outputs.contains(&loss)
        })
    });
    let has_backward = !plan.param_grad_pairs.is_empty();

    let phases: Vec<&'static str> = (0..dispatch_count)
        .map(|index| {
            if !has_backward
                || loss_dispatch
                    .map(|last| index <= last)
                    // A differentiated plan should always identify its loss.
                    // If it does not, avoid inventing a backward boundary.
                    .unwrap_or(true)
            {
                "forward"
            } else {
                "backward"
            }
        })
        .collect();

    let dispatch_medians: Vec<f64> = timing_samples
        .iter()
        .map(|samples| quantile(samples, 0.5))
        .collect();
    let dispatch_median_total: f64 = dispatch_medians.iter().sum();

    let mut dispatches = Vec::with_capacity(dispatch_count);
    for (index, dispatch) in plan.dispatches.iter().enumerate() {
        let input_buffer_bytes = dispatch
            .input_buffers
            .iter()
            .map(|buffer| plan.buffers[buffer.0 as usize])
            .sum();
        let output_buffer_bytes = plan.buffers[dispatch.output_buffer.0 as usize]
            + dispatch
                .extra_outputs
                .iter()
                .map(|buffer| plan.buffers[buffer.0 as usize])
                .sum::<usize>();
        let median_ms = dispatch_medians[index];
        dispatches.push(DispatchProfile {
            index,
            phase: phases[index].to_string(),
            family: dispatch.profile_family().to_string(),
            shader: format!("{:?}", dispatch.shader),
            label: dispatch.label.clone(),
            timestamp_label: timestamp_labels[index].clone(),
            pipeline: pipeline_keys[index].clone(),
            workgroups: dispatch.workgroups,
            workgroup_count: dispatch
                .workgroups
                .iter()
                .map(|&value| u64::from(value))
                .product(),
            input_buffer_bytes,
            output_buffer_bytes,
            cooperative: dispatch.use_coop,
            small_tile: dispatch.use_small_tiles,
            requires_full_precision: dispatch.requires_full_precision,
            weight_format: format!("{:?}", dispatch.weight_format),
            has_prologue: dispatch.matmul_prologue.is_some(),
            has_epilogue: dispatch.matmul_epilogue.is_some() || !dispatch.epilogue.is_empty(),
            timing_samples_ms: timing_samples[index].clone(),
            median_ms,
            p25_ms: quantile(&timing_samples[index], 0.25),
            p75_ms: quantile(&timing_samples[index], 0.75),
            share_of_dispatch_median_sum_pct: percentage(median_ms, dispatch_median_total),
        });
    }

    let mut family_indices: BTreeMap<(&str, &str), Vec<usize>> = BTreeMap::new();
    for dispatch in &dispatches {
        family_indices
            .entry((&dispatch.phase, &dispatch.family))
            .or_default()
            .push(dispatch.index);
    }
    let families = family_indices
        .into_iter()
        .map(|((phase, family), indices)| {
            let timing_samples_ms: Vec<f64> = (0..options.samples)
                .map(|sample| {
                    indices
                        .iter()
                        .map(|&index| timing_samples[index][sample])
                        .sum()
                })
                .collect();
            let dispatch_median_sum_ms = indices.iter().map(|&index| dispatch_medians[index]).sum();
            FamilyProfile {
                phase: phase.to_string(),
                family: family.to_string(),
                dispatch_count: indices.len(),
                median_ms: quantile(&timing_samples_ms, 0.5),
                timing_samples_ms,
                dispatch_median_sum_ms,
                share_of_dispatch_median_sum_pct: percentage(
                    dispatch_median_sum_ms,
                    dispatch_median_total,
                ),
            }
        })
        .collect();

    let pipeline_statistics = if options.include_pipeline_statistics {
        let selected_pipelines: BTreeSet<&str> = pipeline_keys.iter().map(String::as_str).collect();
        session
            .get_profile_pipeline_statistics()
            .into_iter()
            .filter(|entry| selected_pipelines.contains(entry.0.as_str()))
            .map(|(pipeline, executables)| PipelineProfile {
                pipeline,
                executables: executables
                    .into_iter()
                    .map(|executable| PipelineExecutableProfile {
                        name: executable.name,
                        statistics: executable
                            .statistics
                            .into_iter()
                            .map(|statistic| PipelineStatisticProfile {
                                name: statistic.name,
                                description: statistic.description,
                                value: statistic.value,
                            })
                            .collect(),
                    })
                    .collect(),
            })
            .collect()
    } else {
        Vec::new()
    };

    let device = session.device_information();
    let memory = session.memory_summary();
    let forward_dispatch_count = phases.iter().filter(|&&phase| phase == "forward").count();
    let backward_dispatch_count = dispatch_count - forward_dispatch_count;
    let profiled_wall_median_ms = quantile(&profiled_wall_samples_ms, 0.5);
    let gpu_total_median_ms = quantile(&gpu_total_samples_ms, 0.5);
    let timing_contract = if device.driver_name == "Metal" {
        "Blade hardware counter samples at compute-encoder boundaries; one compute pass per plan dispatch"
    } else {
        "Blade hardware timestamp intervals at pass boundaries; one compute pass per plan dispatch; Vulkan top-of-pipe intervals include the inter-pass barrier before the following dispatch"
    };

    Ok(SessionProfile {
        schema_version: 1,
        timing_contract: timing_contract.to_string(),
        device: ProfileDevice {
            backend: if device.driver_name == "Metal" {
                "Metal".to_string()
            } else {
                "Vulkan".to_string()
            },
            device_name: device.device_name.clone(),
            driver_name: device.driver_name.clone(),
            driver_info: device.driver_info.clone(),
            software_emulated: device.is_software_emulated,
        },
        plan: ProfilePlan {
            dispatch_count,
            forward_dispatch_count,
            backward_dispatch_count,
            barrier_group_count: session.num_groups(),
            logical_buffer_bytes: memory.total_buffer_bytes,
            allocated_buffer_bytes: memory.allocated_buffer_bytes,
            device_local_bytes: memory.device_local_bytes,
            physical_allocation_count: memory.num_allocations,
        },
        measurement: ProfileMeasurement {
            sample_count: options.samples,
            unprofiled_median_ms: options.unprofiled_median_ms,
            profiled_wall_samples_ms,
            profiled_wall_median_ms,
            gpu_total_samples_ms,
            gpu_total_median_ms,
            instrumentation_wall_ratio: options
                .unprofiled_median_ms
                .filter(|&baseline| baseline > 0.0)
                .map(|baseline| profiled_wall_median_ms / baseline),
            timestamped_gpu_share_of_profiled_wall_pct: percentage(
                gpu_total_median_ms,
                profiled_wall_median_ms,
            ),
        },
        families,
        dispatches,
        pipeline_statistics,
    })
}

/// Save a structured session profile as pretty-printed JSON.
pub fn save_session_profile_json(
    path: impl AsRef<Path>,
    profile: &SessionProfile,
) -> std::io::Result<()> {
    let file = std::fs::File::create(path)?;
    serde_json::to_writer_pretty(std::io::BufWriter::new(file), profile)
        .map_err(std::io::Error::other)
}

fn percentage(part: f64, total: f64) -> f64 {
    if total > 0.0 {
        part / total * 100.0
    } else {
        0.0
    }
}

fn quantile(values: &[f64], q: f64) -> f64 {
    debug_assert!(!values.is_empty());
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    if sorted.len() == 1 {
        return sorted[0];
    }
    let position = (sorted.len() - 1) as f64 * q;
    let low = position.floor() as usize;
    let high = (low + 1).min(sorted.len() - 1);
    let fraction = position - low as f64;
    sorted[low] * (1.0 - fraction) + sorted[high] * fraction
}

// ---- Tracing Layer ----

/// A [`tracing_subscriber::Layer`] that captures span enter/exit as Perfetto
/// slice events on the CPU track.
pub struct ProfileLayer {
    inner: Arc<Mutex<ProfilerInner>>,
}

impl<S> Layer<S> for ProfileLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_enter(&self, id: &span::Id, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            let mut guard = self.inner.lock().unwrap();
            let ts = guard.now_ns();
            guard.events.push(TraceEvent {
                name: span.name().to_string(),
                timestamp_ns: ts,
                track_uuid: CPU_TRACK_UUID,
                kind: EventKind::SliceBegin,
            });
        }
    }

    fn on_exit(&self, id: &span::Id, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            let mut guard = self.inner.lock().unwrap();
            let ts = guard.now_ns();
            guard.events.push(TraceEvent {
                name: span.name().to_string(),
                timestamp_ns: ts,
                track_uuid: CPU_TRACK_UUID,
                kind: EventKind::SliceEnd,
            });
        }
    }

    fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
        let mut guard = self.inner.lock().unwrap();
        let ts = guard.now_ns();
        guard.events.push(TraceEvent {
            name: event.metadata().name().to_string(),
            timestamp_ns: ts,
            track_uuid: CPU_TRACK_UUID,
            kind: EventKind::Instant,
        });
    }
}

// ---- Perfetto binary trace writer ----
//
// Minimal protobuf encoder — just enough to produce valid .pftrace files
// without pulling in prost or other heavy dependencies.

/// Write a Perfetto trace file from collected events.
fn write_pftrace(path: &Path, events: &[TraceEvent]) -> std::io::Result<()> {
    use std::io::Write;
    let mut trace = ProtoBuf::new();

    // Process descriptor packet.
    let mut proc_desc = ProtoBuf::new();
    proc_desc.uint32(1, std::process::id()); // pid
    let mut track_desc = ProtoBuf::new();
    track_desc.uint64(1, 0); // uuid (process track)
    track_desc.message(3, &proc_desc); // process
    track_desc.string(2, "meganeura"); // name
    let mut pkt = ProtoBuf::new();
    pkt.message(60, &track_desc); // track_descriptor
    pkt.uint32(10, 1); // trusted_packet_sequence_id
    trace.message(1, &pkt); // Trace.packet

    // CPU track descriptor.
    let mut td = ProtoBuf::new();
    td.uint64(1, CPU_TRACK_UUID);
    td.uint64(5, 0); // parent_uuid → process
    td.string(2, "CPU");
    let mut pkt = ProtoBuf::new();
    pkt.message(60, &td);
    pkt.uint32(10, 1);
    trace.message(1, &pkt);

    // GPU track descriptor.
    let mut td = ProtoBuf::new();
    td.uint64(1, GPU_TRACK_UUID);
    td.uint64(5, 0); // parent_uuid → process
    td.string(2, "GPU");
    let mut pkt = ProtoBuf::new();
    pkt.message(60, &td);
    pkt.uint32(10, 1);
    trace.message(1, &pkt);

    // Sort events by timestamp so Perfetto sees them in order within the
    // shared packet sequence. GPU pass events are appended after CPU events
    // but carry earlier timestamps (the submit offset), which causes
    // "misplaced End" warnings if written in insertion order.
    let mut sorted: Vec<usize> = (0..events.len()).collect();
    sorted.sort_by_key(|&i| events[i].timestamp_ns);

    // Event packets.
    for &i in &sorted {
        let ev = &events[i];
        let mut te = ProtoBuf::new();
        te.uint64(11, ev.track_uuid); // track_uuid
        te.int32(9, ev.kind as i32); // type enum
        te.string(23, &ev.name); // name

        let mut pkt = ProtoBuf::new();
        pkt.uint64(8, ev.timestamp_ns); // timestamp
        pkt.message(11, &te); // track_event
        pkt.uint32(10, 1); // trusted_packet_sequence_id
        trace.message(1, &pkt);
    }

    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(&trace.buf)?;
    Ok(())
}

// ---- Minimal protobuf encoder ----

struct ProtoBuf {
    buf: Vec<u8>,
}

impl ProtoBuf {
    fn new() -> Self {
        Self {
            buf: Vec::with_capacity(128),
        }
    }

    fn write_varint(&mut self, mut val: u64) {
        loop {
            let byte = (val & 0x7F) as u8;
            val >>= 7;
            if val == 0 {
                self.buf.push(byte);
                return;
            }
            self.buf.push(byte | 0x80);
        }
    }

    fn tag(&mut self, field: u32, wire_type: u32) {
        self.write_varint(((field as u64) << 3) | wire_type as u64);
    }

    fn uint64(&mut self, field: u32, val: u64) {
        self.tag(field, 0);
        self.write_varint(val);
    }

    fn uint32(&mut self, field: u32, val: u32) {
        self.tag(field, 0);
        self.write_varint(val as u64);
    }

    fn int32(&mut self, field: u32, val: i32) {
        self.tag(field, 0);
        // Protobuf int32 uses varint with sign extension to 64 bits.
        self.write_varint(val as u32 as u64);
    }

    fn string(&mut self, field: u32, val: &str) {
        self.tag(field, 2);
        self.write_varint(val.len() as u64);
        self.buf.extend_from_slice(val.as_bytes());
    }

    fn message(&mut self, field: u32, msg: &ProtoBuf) {
        self.tag(field, 2);
        self.write_varint(msg.buf.len() as u64);
        self.buf.extend_from_slice(&msg.buf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_varint_encoding() {
        let mut pb = ProtoBuf::new();
        pb.write_varint(0);
        assert_eq!(pb.buf, &[0]);

        let mut pb = ProtoBuf::new();
        pb.write_varint(1);
        assert_eq!(pb.buf, &[1]);

        let mut pb = ProtoBuf::new();
        pb.write_varint(300);
        assert_eq!(pb.buf, &[0xAC, 0x02]);
    }

    #[test]
    fn test_save_produces_nonempty_file() {
        // Initialize the profiler for this test.
        let inner = get_or_init();
        {
            let mut guard = inner.lock().unwrap();
            guard.events.push(TraceEvent {
                name: "test_span".into(),
                timestamp_ns: 1000,
                track_uuid: CPU_TRACK_UUID,
                kind: EventKind::SliceBegin,
            });
            guard.events.push(TraceEvent {
                name: "test_span".into(),
                timestamp_ns: 2000,
                track_uuid: CPU_TRACK_UUID,
                kind: EventKind::SliceEnd,
            });
            guard.events.push(TraceEvent {
                name: "matmul".into(),
                timestamp_ns: 1200,
                track_uuid: GPU_TRACK_UUID,
                kind: EventKind::SliceBegin,
            });
            guard.events.push(TraceEvent {
                name: "matmul".into(),
                timestamp_ns: 1800,
                track_uuid: GPU_TRACK_UUID,
                kind: EventKind::SliceEnd,
            });
        }

        let dir = std::env::temp_dir().join("meganeura_profiler_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test.pftrace");
        save(&path).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        // Should be a non-trivial protobuf file.
        assert!(
            bytes.len() > 50,
            "trace file too small: {} bytes",
            bytes.len()
        );
        // First byte should be a protobuf tag for field 1, wire type 2 (length-delimited).
        assert_eq!(bytes[0] & 0x07, 2, "expected length-delimited wire type");
        assert_eq!(bytes[0] >> 3, 1, "expected field number 1 (Trace.packet)");

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_dir(&dir);

        // Clean up events for other tests.
        inner.lock().unwrap().events.clear();
    }

    #[test]
    fn test_record_gpu_passes() {
        let inner = get_or_init();
        inner.lock().unwrap().events.clear();

        record_gpu_passes(
            5000,
            &[
                ("relu".into(), Duration::from_nanos(100)),
                ("matmul".into(), Duration::from_nanos(500)),
            ],
        );

        let guard = inner.lock().unwrap();
        assert_eq!(guard.events.len(), 4); // 2 begin + 2 end
        assert_eq!(guard.events[0].name, "relu");
        assert_eq!(guard.events[0].timestamp_ns, 5000);
        assert_eq!(guard.events[0].kind, EventKind::SliceBegin);
        assert_eq!(guard.events[1].timestamp_ns, 5100); // 5000 + 100
        assert_eq!(guard.events[1].kind, EventKind::SliceEnd);
        assert_eq!(guard.events[2].name, "matmul");
        assert_eq!(guard.events[2].timestamp_ns, 5100);
        assert_eq!(guard.events[3].timestamp_ns, 5600); // 5100 + 500

        drop(guard);
        inner.lock().unwrap().events.clear();
    }

    #[test]
    fn test_now_ns_increases() {
        let _ = get_or_init();
        let t1 = now_ns();
        for _ in 0..1000 {
            std::hint::black_box(0);
        }
        let t2 = now_ns();
        assert!(t2 >= t1);
    }

    #[test]
    fn structured_profile_quantiles_interpolate() {
        let values = [4.0, 1.0, 3.0, 2.0];
        assert_eq!(quantile(&values, 0.25), 1.75);
        assert_eq!(quantile(&values, 0.5), 2.5);
        assert_eq!(quantile(&values, 0.75), 3.25);
    }

    #[test]
    fn structured_profile_zero_sample_error_is_actionable() {
        assert_eq!(
            ProfileError::ZeroSamples.to_string(),
            "profile sample count must be positive"
        );
    }
}
