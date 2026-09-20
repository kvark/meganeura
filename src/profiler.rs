//! Profiling infrastructure producing Perfetto binary traces (`.pftrace`).
//!
//! CPU-side work is captured automatically via [`tracing`] spans. Calibrated
//! GPU pass ranges come from Blade's calibrated pass-start and completion
//! timestamps. Both land on separate tracks in the resulting trace, viewable in
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
#[cfg(test)]
use std::time::Duration;
use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    path::Path,
    sync::{Arc, Mutex},
    time::Instant,
};
#[cfg(feature = "profiler")]
use tracing::{Subscriber, field, span};
#[cfg(feature = "profiler")]
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

/// The profiler state: armed once — by `init` or the first GPU-context
/// initialization — and thereafter only read. The event buffer inside is
/// a `Mutex`, so post-arming sharing is lock-sharing as usual and the
/// static is never rewritten.
///
/// [`arm`]: arm
static mut PROFILER: Option<Arc<Mutex<ProfilerInner>>> = None;

/// The profiler state, when it is armed. Everything that records or dumps
/// runs after `init`, or after a GPU context existed to have armed it —
/// so regular paths read through [`profiler`], and only
/// `now_ns`/`event_count`/GPU-timestamp recording keep a soft option.
fn armed() -> Option<&'static Arc<Mutex<ProfilerInner>>> {
    // SAFETY: the static is written only by [`arm`], which runs before any
    // recording or dumping (from `init` and from context initialization),
    // and then never rewritten.
    unsafe { (&raw const PROFILER).as_ref().and_then(|st| st.as_ref()) }
}

/// Arm the in-process event buffer without installing a tracing
/// subscriber.
///
/// Called by GPU-context initialization and by `init`; after that,
/// GPU-timestamp recording and trace dumping never see an unarmed state.
pub fn arm() {
    // SAFETY: arming happens before any recording; gated on being unarmed
    // so it runs once per process, and `init` races nothing — its callers
    // run before GPU work starts.
    if armed().is_none() {
        unsafe {
            PROFILER = Some(Arc::new(Mutex::new(ProfilerInner {
                epoch: Instant::now(),
                events: Vec::with_capacity(8192),
            })));
        }
    }
}

/// The armed profiler; panics when used before [`arm`].
fn profiler() -> &'static Arc<Mutex<ProfilerInner>> {
    armed().expect("profiler used before init: call meganeura::profiler::init first")
}

// ---- Public API ----

/// Initialize profiling.
///
/// Always arms the in-process event buffer used by GPU timestamp
/// recording. With the `profiler` feature, also installs a global
/// [`tracing`] subscriber that records CPU spans as Perfetto slices.
/// Safe to call multiple times (subsequent subscriber installs are
/// no-ops). Must be called *before* any tracing spans you want captured.
pub fn init() {
    init_with_targets(&[]);
}

/// Initialize profiling and include spans emitted by embedding crates.
///
/// Meganeura spans are always captured. Each additional entry is a tracing
/// target prefix, such as `"my_app"`; dependency targets remain out of the
/// performance trace unless explicitly requested.
pub fn init_with_targets(additional_targets: &[&str]) {
    arm();
    #[cfg(not(feature = "profiler"))]
    let _ = additional_targets;
    #[cfg(feature = "profiler")]
    {
        let inner = profiler().clone();
        let mut targets = vec!["meganeura".to_string()];
        targets.extend(
            additional_targets
                .iter()
                .map(|target| (*target).to_string()),
        );
        targets.sort_unstable();
        targets.dedup();
        let layer = ProfileLayer { inner, targets };
        let subscriber = tracing_subscriber::registry().with(layer);
        let _ = tracing::subscriber::set_global_default(subscriber);
    }
}

/// GPU pass timestamps copied out of Blade's borrowed view.
///
/// `blade_graphics::Timing` borrows its pass names from the encoder, which
/// reuses that storage on the next submission, so anything kept past the
/// harvest has to own them.
#[derive(Clone, Debug, Default)]
pub struct GpuTimings {
    /// Pass starts in execution order.
    pub passes: Vec<(String, Instant)>,
    /// Completion time of the final pass.
    pub done: Option<Instant>,
}

impl GpuTimings {
    /// Each pass's start-to-next-start duration, the last ending at `done`.
    pub fn pass_durations(&self) -> impl Iterator<Item = (&str, std::time::Duration)> + '_ {
        self.passes.iter().enumerate().filter_map(move |(i, pass)| {
            let end = match self.passes.get(i + 1) {
                Some(next) => Some(next.1),
                None => self.done,
            }?;
            Some((pass.0.as_str(), end.duration_since(pass.1)))
        })
    }
}

/// Record calibrated GPU pass ranges on the GPU track.
///
/// Blade maps hardware timestamp queries into the same monotonic clock as
/// [`Instant`]. Each pass ends at the next pass start, or at `done` for the
/// final pass. CPU submission time is intentionally not involved.
pub fn record_gpu_timings(timings: &GpuTimings) {
    if armed().is_some() {
        let inner = profiler();
        let mut guard = inner.lock().unwrap();
        for (index, &(ref name, start)) in timings.passes.iter().enumerate() {
            let Some(end) = timings
                .passes
                .get(index + 1)
                .map(|next| next.1)
                .or(timings.done)
            else {
                continue;
            };
            let start_ns = match start.checked_duration_since(guard.epoch) {
                Some(duration) => duration.as_nanos() as u64,
                None => {
                    let delta_ns = guard.epoch.duration_since(start).as_nanos();
                    let timestamp_ns = guard.now_ns();
                    guard.events.push(TraceEvent {
                        name: format!(
                            "gpu_timestamp_rejected pass={} before_epoch_ns={delta_ns}",
                            name
                        ),
                        timestamp_ns,
                        track_uuid: CPU_TRACK_UUID,
                        kind: EventKind::Instant,
                    });
                    continue;
                }
            };
            let end_ns = match end.checked_duration_since(guard.epoch) {
                Some(duration) => duration.as_nanos() as u64,
                None => {
                    let delta_ns = guard.epoch.duration_since(end).as_nanos();
                    let timestamp_ns = guard.now_ns();
                    guard.events.push(TraceEvent {
                        name: format!(
                            "gpu_timestamp_rejected pass={} end_before_epoch_ns={delta_ns}",
                            name
                        ),
                        timestamp_ns,
                        track_uuid: CPU_TRACK_UUID,
                        kind: EventKind::Instant,
                    });
                    continue;
                }
            };
            guard.events.push(TraceEvent {
                name: name.clone(),
                timestamp_ns: start_ns,
                track_uuid: GPU_TRACK_UUID,
                kind: EventKind::SliceBegin,
            });
            guard.events.push(TraceEvent {
                name: name.clone(),
                timestamp_ns: end_ns,
                track_uuid: GPU_TRACK_UUID,
                kind: EventKind::SliceEnd,
            });
        }
    }
}

/// Record a single CPU event (for use outside tracing spans).
pub fn record_instant(name: &str) {
    let inner = profiler();
    let mut guard = inner.lock().unwrap();
    let ts = guard.now_ns();
    guard.events.push(TraceEvent {
        name: name.to_string(),
        timestamp_ns: ts,
        track_uuid: CPU_TRACK_UUID,
        kind: EventKind::Instant,
    });
}

/// Return the nanosecond offset from the profiler epoch (for GPU timing placement).
pub fn now_ns() -> u64 {
    armed()
        .map(|inner| inner.lock().unwrap().now_ns())
        .unwrap_or(0)
}

/// Number of recorded events (including both CPU spans and GPU passes).
pub fn event_count() -> usize {
    armed()
        .map(|inner| inner.lock().unwrap().events.len())
        .unwrap_or(0)
}

/// Write all collected events to a Perfetto `.pftrace` binary trace file.
pub fn save(path: impl AsRef<Path>) -> std::io::Result<()> {
    let inner = profiler();
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
    /// Upper bound on the compute passes one replay may timestamp.
    ///
    /// Defaults to Blade's per-submission limit, which is what plans are
    /// actually measured against. Lower it to cut a plan into more, smaller
    /// windows — each replay then perturbs the schedule less, at the cost of
    /// more replays.
    ///
    /// Clamped to `3 ..= blade_graphics::limits::PASS_COUNT`: two slots
    /// always go to the untimed passes around the window, and raising the
    /// bound past the backend's own limit would only build a window whose
    /// timestamps the backend drops.
    pub max_timed_passes_per_replay: Option<usize>,
}

impl Default for CaptureOptions {
    fn default() -> Self {
        Self {
            samples: 3,
            unprofiled_median_ms: None,
            include_pipeline_statistics: true,
            max_timed_passes_per_replay: None,
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
    /// Resident graph/optimizer buffer requests, not driver allocation or peak.
    pub resident_buffer_bytes: usize,
    pub adam_state_bytes: usize,
    pub grad_accumulator_bytes: usize,
    pub optimizer_aux_bytes: usize,
    pub device_local_bytes: usize,
    pub physical_allocation_count: usize,
}

#[derive(Clone, Debug, Serialize)]
pub struct ProfileMeasurement {
    pub sample_count: usize,
    /// Replays per sample. Each covers a disjoint range of dispatch indices,
    /// because Blade timestamps a bounded number of passes per submission.
    /// One means the whole plan was timed in a single replay.
    pub window_count: usize,
    /// Largest number of dispatches timestamped in one replay.
    pub max_window_dispatches: usize,
    pub unprofiled_median_ms: Option<f64>,
    /// One entry per sample: the sum of that sample's replay wall times.
    pub profiled_wall_samples_ms: Vec<f64>,
    pub profiled_wall_median_ms: f64,
    /// One entry per sample: the sum over that sample's windows, so it covers
    /// every dispatch however many replays it took to time them.
    pub gpu_total_samples_ms: Vec<f64>,
    pub gpu_total_median_ms: f64,
    /// Profiled wall median divided by the unprofiled benchmark median. For a
    /// windowed capture this includes every replay needed for one sample.
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
    /// Graph node ids this dispatch implements (provenance; several after
    /// dispatch-level fusion).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub origin: Vec<u32>,
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
    /// The session reported timings for dispatches other than the window's.
    /// An internal inconsistency rather than a caller error, reported instead
    /// of panicking so that the capture still unwinds through one place and
    /// leaves the session unprofiled.
    Attribution {
        sample: usize,
        detail: String,
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
            Self::Attribution { sample, ref detail } => write!(
                f,
                "profile sample {sample} mis-attributed a window: {detail}"
            ),
            Self::MissingGpuTimings { sample } => write!(
                f,
                "no GPU timings resolved for profile sample {sample}; build the \
                 context with GpuOptions {{ timing: true, .. }} and set \
                 SessionOptions::gpu_timing to match"
            ),
            Self::TimingCount {
                sample,
                expected,
                actual,
            } => write!(
                f,
                "profile sample {sample} recovered {actual} of {expected} timed dispatches; \
                 disable runtime-appended optimizer, gradient-accumulation, and \
                 gradient-clipping passes before capture"
            ),
        }
    }
}

impl std::error::Error for ProfileError {}

/// Split a plan into the dispatch ranges that can be timed one replay each.
///
/// Blade writes at most `pass_limit` timestamps per submission and silently
/// drops the rest, so a replay can time at most that many passes. Two of the
/// slots go to the grouped passes carrying the dispatches on either side of
/// the window, whatever their number — see [`crate::runtime::Session::
/// set_profiling_window`]. A plan that fits in one window yields exactly one,
/// which replays the session the same number of times as before windowing.
fn profile_windows(dispatch_count: usize, pass_limit: usize) -> Vec<std::ops::Range<usize>> {
    let whole_plan = 0..dispatch_count;
    if dispatch_count <= pass_limit {
        return vec![whole_plan];
    }
    let per_window = pass_limit.saturating_sub(2).max(1);
    // Even out the last window rather than leaving it a short remainder: an
    // uneven split would measure the tail under a different pass count than
    // the rest, and pass count is what the instrumentation overhead tracks.
    let count = dispatch_count.div_ceil(per_window);
    let per_window = dispatch_count.div_ceil(count);
    (0..dispatch_count)
        .step_by(per_window)
        .map(|start| start..(start + per_window).min(dispatch_count))
        .collect()
}

/// Capture repeated, structured per-dispatch GPU timings for a session.
///
/// Set `MEGANEURA_GPU_TIMING=1` before the session creates its Blade context.
/// Each retained execution runs in one-compute-pass-per-dispatch mode and
/// resolves its hardware timestamps after the completion fence. The returned
/// artifact retains raw samples, reports instrumentation overhead against an
/// optional normal benchmark median, and aggregates dispatches by
/// forward/backward phase and coarse kernel family.
///
/// Blade timestamps a bounded number of passes per submission, so a plan
/// larger than that is measured as several deterministic replays, each timing
/// one window of dispatch indices and running the rest in grouped passes. The
/// windows tile the plan, so every dispatch is still measured `samples` times;
/// only the replay count grows. `measurement.window_count` reports how many
/// replays one sample took, and is 1 for plans that fit in a single one.
///
/// `prepare` is called immediately before every replay, windows included. It
/// should restore inputs and any state that the workload mutates, so that
/// every window observes the same execution.
///
/// The structured dispatch table describes the compiled execution plan.
/// Capture with optimizer, gradient-accumulation, and gradient-clipping passes
/// disabled: those runtime-appended passes do not have plan metadata, and the
/// collector rejects their additional timestamps instead of misattributing
/// them to plan dispatches.
pub fn capture_session_profile(
    session: &mut crate::runtime::Session,
    prepare: impl FnMut(&mut crate::runtime::Session),
    options: CaptureOptions,
) -> Result<SessionProfile, ProfileError> {
    let result = capture_windows(session, prepare, options);
    // However this ended, the session must not be left timing a window. A
    // caller that handles the error and keeps stepping would otherwise run
    // every later step one pass per dispatch, which is slower and not what
    // it asked for.
    session.set_profiling_window(None);
    result
}

fn capture_windows(
    session: &mut crate::runtime::Session,
    mut prepare: impl FnMut(&mut crate::runtime::Session),
    options: CaptureOptions,
) -> Result<SessionProfile, ProfileError> {
    if options.samples == 0 {
        return Err(ProfileError::ZeroSamples);
    }

    let dispatch_count = session.plan().dispatches.len();
    // Clamped to the backend's own limit at both ends. An override above it
    // does not raise anything — Blade still stops writing timestamps at
    // `PASS_COUNT` — it just asks for one oversized window, which then
    // recovers nothing and fails the capture instead of taking the windowed
    // route that works.
    let pass_limit = options
        .max_timed_passes_per_replay
        .unwrap_or(blade_graphics::limits::PASS_COUNT)
        .clamp(3, blade_graphics::limits::PASS_COUNT);
    let windows = profile_windows(dispatch_count, pass_limit);
    let window_count = windows.len();
    let mut timing_samples = vec![Vec::with_capacity(options.samples); dispatch_count];
    let mut timestamp_labels = vec![String::new(); dispatch_count];
    let mut profiled_wall_samples_ms = Vec::with_capacity(options.samples);
    let mut gpu_total_samples_ms = Vec::with_capacity(options.samples);

    for sample in 0..options.samples {
        let mut total_ms = 0.0;
        let mut wall_ms = 0.0;
        for window in &windows {
            session.set_profiling_window(Some(window.clone()));
            prepare(session);
            let wall_start = Instant::now();
            session.step();
            session.wait();
            wall_ms += wall_start.elapsed().as_secs_f64() * 1000.0;

            let timed = session.profiled_dispatch_timings();
            if timed.len() != window.len() {
                if session.gpu_timings().is_empty() {
                    return Err(ProfileError::MissingGpuTimings { sample });
                }
                return Err(ProfileError::TimingCount {
                    sample,
                    expected: window.len(),
                    actual: timed.len(),
                });
            }

            // The session reports one entry per dispatch in the window, in
            // order. Checking it here keeps a stitching bug from surfacing
            // later as an out-of-bounds index in the aggregation, where the
            // cause would be a good deal harder to see.
            let attributed: Vec<usize> = timed.iter().map(|entry| entry.0).collect();
            if !attributed.iter().copied().eq(window.clone()) {
                return Err(ProfileError::Attribution {
                    sample,
                    detail: format!("window {window:?} came back as {attributed:?}"),
                });
            }

            for (index, name, duration) in timed {
                let duration_ms = duration.as_secs_f64() * 1000.0;
                total_ms += duration_ms;
                timing_samples[index].push(duration_ms);
                if sample == 0 {
                    timestamp_labels[index] = name;
                }
            }
        }
        profiled_wall_samples_ms.push(wall_ms);
        gpu_total_samples_ms.push(total_ms);
    }

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
            origin: dispatch.origin.clone(),
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
            cooperative: dispatch.use_coop(),
            small_tile: dispatch.use_small_tiles(),
            requires_full_precision: dispatch.requires_full_precision,
            weight_format: format!("{:?}", dispatch.weight_format),
            has_prologue: dispatch.matmul_prologue.is_some(),
            has_epilogue: dispatch.matmul_epilogue.is_some(),
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
    let timing_contract = "Blade calibrated pass-start timestamps on the process monotonic clock; each interval ends at the next pass start or final submission completion; one compute pass per timed plan dispatch, with the dispatches outside the replay's window batched into one untimed pass per side";

    Ok(SessionProfile {
        schema_version: 2,
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
            resident_buffer_bytes: memory.total_allocated_bytes(),
            adam_state_bytes: memory.adam_state_bytes,
            grad_accumulator_bytes: memory.grad_accumulator_bytes,
            optimizer_aux_bytes: memory.optimizer_aux_bytes,
            device_local_bytes: memory.device_local_bytes,
            physical_allocation_count: memory.num_allocations,
        },
        measurement: ProfileMeasurement {
            sample_count: options.samples,
            window_count,
            max_window_dispatches: windows.iter().map(|window| window.len()).max().unwrap_or(0),
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
#[cfg(feature = "profiler")]
pub struct ProfileLayer {
    inner: Arc<Mutex<ProfilerInner>>,
    targets: Vec<String>,
}

#[cfg(feature = "profiler")]
impl ProfileLayer {
    fn captures_target(&self, target: &str) -> bool {
        self.targets.iter().any(|prefix| {
            target == prefix
                || target
                    .strip_prefix(prefix)
                    .is_some_and(|suffix| suffix.starts_with("::"))
        })
    }
}

#[cfg(feature = "profiler")]
#[derive(Default)]
struct TraceFields(Vec<(String, String)>);

#[cfg(feature = "profiler")]
impl TraceFields {
    const MIN_DURATION_FIELD: &'static str = "trace_min_duration_us";

    fn record(&mut self, field: &field::Field, value: String) {
        const MAX_VALUE_CHARS: usize = 120;
        let mut value = value;
        if value.chars().count() > MAX_VALUE_CHARS {
            value = value.chars().take(MAX_VALUE_CHARS - 1).collect();
            value.push('\u{2026}');
        }
        if let Some(entry) = self.0.iter_mut().find(|entry| entry.0 == field.name()) {
            entry.1 = value;
        } else {
            self.0.push((field.name().to_string(), value));
        }
    }

    fn display_name(&self, base: &str) -> String {
        let message = self
            .0
            .iter()
            .find_map(|entry| (entry.0 == "message").then_some(entry.1.as_str()));
        let base = message.unwrap_or(base);
        let fields = self
            .0
            .iter()
            .filter(|entry| entry.0 != "message" && entry.0 != Self::MIN_DURATION_FIELD)
            .collect::<Vec<_>>();
        if fields.is_empty() {
            return base.to_string();
        }
        if fields.len() == 1 {
            let field = fields[0];
            return if field.0 == "name" {
                format!("{base} {}", field.1)
            } else {
                format!("{base} {}={}", field.0, field.1)
            };
        }
        let fields = fields
            .iter()
            .map(|entry| format!("{}={}", entry.0, entry.1))
            .collect::<Vec<_>>()
            .join(" ");
        format!("{base} {fields}")
    }

    fn min_duration_ns(&self) -> u64 {
        self.0
            .iter()
            .find_map(|entry| {
                (entry.0 == Self::MIN_DURATION_FIELD)
                    .then(|| entry.1.parse::<u64>().ok())
                    .flatten()
            })
            .unwrap_or(0)
            .saturating_mul(1_000)
    }
}

#[cfg(feature = "profiler")]
impl field::Visit for TraceFields {
    fn record_i64(&mut self, field: &field::Field, value: i64) {
        self.record(field, value.to_string());
    }

    fn record_u64(&mut self, field: &field::Field, value: u64) {
        self.record(field, value.to_string());
    }

    fn record_bool(&mut self, field: &field::Field, value: bool) {
        self.record(field, value.to_string());
    }

    fn record_str(&mut self, field: &field::Field, value: &str) {
        self.record(field, value.to_string());
    }

    fn record_debug(&mut self, field: &field::Field, value: &dyn fmt::Debug) {
        self.record(field, format!("{value:?}"));
    }
}

#[cfg(feature = "profiler")]
struct ProfileSpan {
    name: String,
    min_duration_ns: u64,
    pending_starts: Mutex<Vec<(std::thread::ThreadId, u64)>>,
}

#[cfg(feature = "profiler")]
impl<S> Layer<S> for ProfileLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, attrs: &span::Attributes<'_>, id: &span::Id, ctx: Context<'_, S>) {
        if !self.captures_target(attrs.metadata().target()) {
            return;
        }
        if let Some(span) = ctx.span(id) {
            let mut fields = TraceFields::default();
            attrs.record(&mut fields);
            let name = fields.display_name(span.name());
            span.extensions_mut().insert(ProfileSpan {
                name,
                min_duration_ns: fields.min_duration_ns(),
                pending_starts: Mutex::new(Vec::new()),
            });
        }
    }

    fn on_enter(&self, id: &span::Id, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            let extensions = span.extensions();
            let Some(profile) = extensions.get::<ProfileSpan>() else {
                return;
            };
            let mut guard = self.inner.lock().unwrap();
            let ts = guard.now_ns();
            if profile.min_duration_ns != 0 {
                drop(guard);
                profile
                    .pending_starts
                    .lock()
                    .unwrap()
                    .push((std::thread::current().id(), ts));
                return;
            }
            guard.events.push(TraceEvent {
                name: profile.name.clone(),
                timestamp_ns: ts,
                track_uuid: CPU_TRACK_UUID,
                kind: EventKind::SliceBegin,
            });
        }
    }

    fn on_exit(&self, id: &span::Id, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            let extensions = span.extensions();
            let Some(profile) = extensions.get::<ProfileSpan>() else {
                return;
            };
            let mut guard = self.inner.lock().unwrap();
            let ts = guard.now_ns();
            if profile.min_duration_ns != 0 {
                drop(guard);
                let thread = std::thread::current().id();
                let start = {
                    let mut starts = profile.pending_starts.lock().unwrap();
                    let Some(index) = starts.iter().rposition(|entry| entry.0 == thread) else {
                        return;
                    };
                    starts.remove(index).1
                };
                if ts.saturating_sub(start) < profile.min_duration_ns {
                    return;
                }
                let mut guard = self.inner.lock().unwrap();
                guard.events.push(TraceEvent {
                    name: profile.name.clone(),
                    timestamp_ns: start,
                    track_uuid: CPU_TRACK_UUID,
                    kind: EventKind::SliceBegin,
                });
                guard.events.push(TraceEvent {
                    name: profile.name.clone(),
                    timestamp_ns: ts,
                    track_uuid: CPU_TRACK_UUID,
                    kind: EventKind::SliceEnd,
                });
                return;
            }
            guard.events.push(TraceEvent {
                name: profile.name.clone(),
                timestamp_ns: ts,
                track_uuid: CPU_TRACK_UUID,
                kind: EventKind::SliceEnd,
            });
        }
    }

    fn on_event(&self, event: &tracing::Event<'_>, _ctx: Context<'_, S>) {
        if !self.captures_target(event.metadata().target()) {
            return;
        }
        let mut fields = TraceFields::default();
        event.record(&mut fields);
        let mut guard = self.inner.lock().unwrap();
        let ts = guard.now_ns();
        guard.events.push(TraceEvent {
            name: fields.display_name(event.metadata().name()),
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
    // shared packet sequence. Resolved GPU pass events are appended after
    // CPU events but carry their earlier calibrated hardware timestamps,
    // which causes "misplaced End" warnings if written in insertion order.
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
        arm();
        let inner = profiler();
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
    fn test_record_gpu_timings() {
        arm();
        let inner = profiler();
        let epoch = {
            let mut guard = inner.lock().unwrap();
            guard.events.clear();
            guard.epoch
        };

        record_gpu_timings(&GpuTimings {
            passes: vec![
                ("relu".into(), epoch + Duration::from_nanos(5_000)),
                ("matmul".into(), epoch + Duration::from_nanos(5_100)),
            ],
            done: Some(epoch + Duration::from_nanos(5_550)),
        });

        let guard = inner.lock().unwrap();
        assert_eq!(guard.events.len(), 4); // 2 begin + 2 end
        assert_eq!(guard.events[0].name, "relu");
        assert_eq!(guard.events[0].timestamp_ns, 5000);
        assert_eq!(guard.events[0].kind, EventKind::SliceBegin);
        assert_eq!(guard.events[1].timestamp_ns, 5100);
        assert_eq!(guard.events[1].kind, EventKind::SliceEnd);
        assert_eq!(guard.events[2].name, "matmul");
        assert_eq!(guard.events[2].timestamp_ns, 5100);
        assert_eq!(guard.events[3].timestamp_ns, 5550);

        drop(guard);
        inner.lock().unwrap().events.clear();
    }

    #[test]
    fn test_now_ns_increases() {
        arm();
        let t1 = now_ns();
        for _ in 0..1000 {
            std::hint::black_box(0);
        }
        let t2 = now_ns();
        assert!(t2 >= t1);
    }

    #[cfg(feature = "profiler")]
    #[test]
    fn trace_fields_make_span_names_readable() {
        let empty = TraceFields::default();
        assert_eq!(empty.display_name("pipeline"), "pipeline");

        let one = TraceFields(vec![("name".into(), "MatMul:cooperative".into())]);
        assert_eq!(one.display_name("pipeline"), "pipeline MatMul:cooperative");

        let measured = TraceFields(vec![("source_bytes".into(), "4627".into())]);
        assert_eq!(
            measured.display_name("naga_parse"),
            "naga_parse source_bytes=4627"
        );

        let threshold = TraceFields(vec![(
            TraceFields::MIN_DURATION_FIELD.into(),
            "1000".into(),
        )]);
        assert_eq!(threshold.display_name("pipeline"), "pipeline");
        assert_eq!(threshold.min_duration_ns(), 1_000_000);

        let several = TraceFields(vec![
            ("dispatches".into(), "1034".into()),
            ("buffers".into(), "1703".into()),
        ]);
        assert_eq!(
            several.display_name("session_init"),
            "session_init dispatches=1034 buffers=1703"
        );

        let event = TraceFields(vec![
            ("message".into(), "GPU timestamps resolved".into()),
            ("passes".into(), "8".into()),
        ]);
        assert_eq!(
            event.display_name("event src/runtime.rs:25"),
            "GPU timestamps resolved passes=8"
        );
    }

    #[cfg(feature = "profiler")]
    #[test]
    fn profile_targets_exclude_dependency_noise() {
        let layer = ProfileLayer {
            inner: profiler().clone(),
            targets: vec!["meganeura".into(), "buddy".into()],
        };
        assert!(layer.captures_target("meganeura::runtime"));
        assert!(layer.captures_target("buddy::brain::gemma4"));
        assert!(!layer.captures_target("buddy_system"));
        assert!(!layer.captures_target("libwayshot_xcap"));
    }

    #[cfg(feature = "profiler")]
    #[test]
    fn trace_minimum_duration_omits_short_span() {
        let inner = Arc::new(Mutex::new(ProfilerInner {
            epoch: Instant::now(),
            events: Vec::new(),
        }));
        let layer = ProfileLayer {
            inner: Arc::clone(&inner),
            targets: vec!["meganeura".into()],
        };
        let subscriber = tracing_subscriber::registry().with(layer);
        tracing::subscriber::with_default(subscriber, || {
            let _span = tracing::info_span!(
                target: "meganeura",
                "too_short",
                trace_min_duration_us = 60_000_000u64,
            )
            .entered();
        });
        assert!(inner.lock().unwrap().events.is_empty());
    }

    #[test]
    fn structured_profile_quantiles_interpolate() {
        let values = [4.0, 1.0, 3.0, 2.0];
        assert_eq!(quantile(&values, 0.25), 1.75);
        assert_eq!(quantile(&values, 0.5), 2.5);
        assert_eq!(quantile(&values, 0.75), 3.25);
    }

    /// A window may hold at most `pass_limit - 2` dispatches, must tile the
    /// plan exactly, and must stay a single window for plans that already fit.
    #[test]
    fn profile_windows_tile_the_plan_within_the_timestamp_limit() {
        for (dispatch_count, pass_limit) in [
            (0, 8),
            (1, 8),
            (8, 8),
            (9, 8),
            (100, 8),
            (1034, 1000),
            (5000, 1000),
            (7, 3),
        ] {
            let windows = profile_windows(dispatch_count, pass_limit);
            let mut next = 0;
            for window in &windows {
                assert_eq!(
                    window.start, next,
                    "{dispatch_count}/{pass_limit}: windows must be contiguous"
                );
                // A replay costs one pass per timed dispatch plus one for
                // each side that has dispatches left outside the window.
                let passes = window.len()
                    + usize::from(window.start > 0)
                    + usize::from(window.end < dispatch_count);
                assert!(
                    passes <= pass_limit,
                    "{dispatch_count}/{pass_limit}: window {window:?} needs \
                     {passes} passes, over the {pass_limit} limit"
                );
                next = window.end;
            }
            assert_eq!(
                next, dispatch_count,
                "{dispatch_count}/{pass_limit}: windows must cover the plan"
            );
        }

        // The common case replays exactly as often as before windowing.
        assert_eq!(profile_windows(1000, 1000), vec![0..1000]);
        // An override above the backend limit must not build a window the
        // backend will not timestamp. The clamp happens in the caller, so
        // check the value it computes rather than only the planner.
        let over = CaptureOptions {
            max_timed_passes_per_replay: Some(blade_graphics::limits::PASS_COUNT * 2),
            ..CaptureOptions::default()
        };
        let effective = over
            .max_timed_passes_per_replay
            .unwrap_or(blade_graphics::limits::PASS_COUNT)
            .clamp(3, blade_graphics::limits::PASS_COUNT);
        assert_eq!(effective, blade_graphics::limits::PASS_COUNT);
        let windows = profile_windows(1500, effective);
        assert!(
            windows.len() > 1 && windows.iter().all(|w| w.len() <= effective),
            "a 1500-dispatch plan must still be windowed: {windows:?}"
        );
        // The motivating case: one decode step no longer refuses to profile.
        assert_eq!(profile_windows(1034, 1000).len(), 2);
        // Windows are evened out rather than leaving a stub at the end.
        assert_eq!(profile_windows(1034, 1000), vec![0..517, 517..1034]);
    }

    #[test]
    fn structured_profile_zero_sample_error_is_actionable() {
        assert_eq!(
            ProfileError::ZeroSamples.to_string(),
            "profile sample count must be positive"
        );
    }
}
