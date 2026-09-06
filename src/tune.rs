//! Bounded, opt-in kernel selection. See [`crate::Session::tune_with`].
//!
//! The search space is deliberately small: scalar and native-f32 cooperative
//! tiles for unpacked dense matmuls, with no precision or binding-layout changes.
//! Measurements use synthetic, private scratch, not a live training step.

use crate::codegen::CoopConfig;
use crate::compile::{Dispatch, ShaderEntry};
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Complete key within the supported contiguous-row-major, non-aliasing domain.
/// No winner is transferred between shapes, directions, or memory placements.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TuneClass {
    pub shader: ShaderEntry,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub requires_full_precision: bool,
    /// Placement of A, B, addend (false if absent), and output, respectively.
    pub device_local: [bool; 4],
    /// Declared bytes for A, B, optional addend, output, in binding order.
    /// Candidates cannot borrow unused capacity from an aliased allocation.
    pub binding_bytes: Vec<usize>,
}

/// f32 implementations with identical bindings and logical extents.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MatmulTile {
    Tile32,
    Tile64,
    /// Hardware-native f32 operands/accumulators, never f16 staging.
    CooperativeF32 {
        tile_size: u32,
    },
}

impl MatmulTile {
    pub(crate) fn native_cooperative(config: Option<&CoopConfig>) -> Option<Self> {
        let config = config?;
        if config.use_f16_input || config.compensated || !matches!(config.tile_size, 8 | 16) {
            return None;
        }
        Some(Self::CooperativeF32 {
            tile_size: config.tile_size,
        })
    }

    pub(crate) fn selected(dispatch: &Dispatch, config: Option<&CoopConfig>) -> Option<Self> {
        if dispatch.use_coop {
            Self::native_cooperative(config)
        } else if dispatch.use_small_tiles {
            Some(Self::Tile32)
        } else {
            Some(Self::Tile64)
        }
    }

    pub(crate) fn coop_config(self) -> Option<CoopConfig> {
        match self {
            Self::CooperativeF32 { tile_size } => Some(CoopConfig {
                tile_size,
                use_f16_input: false,
                compensated: false,
            }),
            _ => None,
        }
    }

    pub(crate) fn apply(self, dispatch: &mut Dispatch, class: &TuneClass) {
        dispatch.use_small_tiles = self == Self::Tile32;
        dispatch.use_coop = matches!(self, Self::CooperativeF32 { .. });
        dispatch.use_coop_compensated = false;
        dispatch.scalar_fallback = dispatch
            .use_coop
            .then(|| (dispatch.shader.clone(), Self::Tile64.workgroups(class)));
        dispatch.workgroups = self.workgroups(class);
    }

    fn workgroups(self, class: &TuneClass) -> [u32; 3] {
        let tile = match self {
            Self::Tile32 => 32,
            Self::Tile64 => 64,
            Self::CooperativeF32 { tile_size } => {
                let tile = 2 * tile_size;
                // Cooperative tiles use X for rows, Y for columns.
                return [class.m.div_ceil(tile), class.n.div_ceil(tile), 1];
            }
        };
        // Scalar tiled kernels use X for columns, Y for rows. Derive exact
        // geometry; doubling rounded 64-tile counts overdispatches edges.
        [class.n.div_ceil(tile), class.m.div_ceil(tile), 1]
    }

    pub(crate) fn buffer_sizes(self, class: &TuneClass) -> Option<Vec<usize>> {
        let mut sizes = class.buffer_sizes()?;
        if let Self::CooperativeF32 { tile_size } = self {
            if !matches!(tile_size, 8 | 16)
                || !class.n.is_multiple_of(16)
                || (matches!(
                    class.shader,
                    ShaderEntry::MatMul | ShaderEntry::FusedMatMulAdd
                ) && class.k < 4)
            {
                return None;
            }
            let tile = 2 * tile_size;
            let padded_m = class.m.div_ceil(tile).checked_mul(tile)?;
            let padded_n = class.n.div_ceil(tile).checked_mul(tile)?;
            let bytes = usize::try_from(padded_m.checked_mul(padded_n)?)
                .ok()?
                .checked_mul(4)?;
            *sizes.last_mut()? = bytes;
            if class.has_addend() {
                sizes[2] = bytes;
            }
        }
        if self.workgroups(class).iter().any(|&n| n == 0 || n > 65_535) {
            return None;
        }
        Some(sizes)
    }

    pub(crate) fn fits(self, class: &TuneClass) -> bool {
        let Some(sizes) = self.buffer_sizes(class) else {
            return false;
        };
        sizes.len() == class.binding_bytes.len()
            && sizes
                .iter()
                .zip(&class.binding_bytes)
                .all(|(required, available)| required <= available)
    }
}

impl TuneClass {
    pub(crate) fn from_dispatch(dispatch: &Dispatch, config: Option<&CoopConfig>) -> Option<Self> {
        let addend = dispatch.shader == ShaderEntry::FusedMatMulAdd;
        if !matches!(
            dispatch.shader,
            ShaderEntry::MatMul
                | ShaderEntry::FusedMatMulAdd
                | ShaderEntry::MatMulAT
                | ShaderEntry::MatMulBT
        ) || dispatch.use_coop_compensated
            || (dispatch.use_coop && dispatch.use_small_tiles)
            || dispatch.weight_format.uses_reduced_storage()
            || dispatch.horizontal_batch >= 2
            || dispatch.matmul_prologue.is_some()
            || dispatch.matmul_epilogue.is_some()
            || dispatch.gemv_rmsnorm.is_some()
            || !dispatch.epilogue.is_empty()
            || !dispatch.epilogue_buffers.is_empty()
            || !dispatch.extra_outputs.is_empty()
            || dispatch.pointwise.is_some()
            || dispatch.reduction.is_some()
            || dispatch.input_buffers.len() != if addend { 3 } else { 2 }
            || dispatch.workgroups[2] != 1
            || dispatch.params.len() != 4
            || dispatch.params[3] != 0
        {
            return None;
        }
        let (m, n, k) = match dispatch.shader {
            ShaderEntry::MatMul | ShaderEntry::FusedMatMulAdd => {
                (dispatch.params[0], dispatch.params[2], dispatch.params[1])
            }
            _ => (dispatch.params[0], dispatch.params[1], dispatch.params[2]),
        };
        if m == 0 || n == 0 || k == 0 || m.div_ceil(32) > 65_535 || n.div_ceil(32) > 65_535 {
            return None;
        }
        let class = Self {
            shader: dispatch.shader.clone(),
            m,
            n,
            k,
            requires_full_precision: dispatch.requires_full_precision,
            device_local: [false; 4],
            binding_bytes: Vec::new(),
        };
        let initial = MatmulTile::selected(dispatch, config)?;
        (initial.buffer_sizes(&class).is_some()
            && dispatch.workgroups == initial.workgroups(&class))
        .then_some(class)
    }

    pub(crate) fn has_addend(&self) -> bool {
        self.shader == ShaderEntry::FusedMatMulAdd
    }

    /// A, B, optional addend, output sizes. Reject u32 shader-index overflow,
    /// not just host allocation overflow.
    pub(crate) fn buffer_sizes(&self) -> Option<Vec<usize>> {
        let bytes = |a: u32, b: u32| usize::try_from(a.checked_mul(b)?).ok()?.checked_mul(4);
        let mut sizes = vec![bytes(self.m, self.k)?, bytes(self.k, self.n)?];
        if self.has_addend() {
            sizes.push(bytes(self.m, self.n)?);
        }
        sizes.push(bytes(self.m, self.n)?);
        Some(sizes)
    }

    /// A small deterministic tournament: scalar alternative first, then
    /// native f32 where it fits. Capability/precision/padding are legality;
    /// occupancy and the static large-shape veto do not exclude candidates.
    pub(crate) fn challengers(
        &self,
        initial: MatmulTile,
        config: Option<&CoopConfig>,
    ) -> Vec<MatmulTile> {
        [
            Some(MatmulTile::Tile64),
            Some(MatmulTile::Tile32),
            MatmulTile::native_cooperative(config),
        ]
        .into_iter()
        .flatten()
        .filter(|&tile| tile != initial && tile.fits(self))
        .collect()
    }
}

/// Placement of the private upload/readback buffer, never the kernel bindings.
/// Its default preserves historical reports; new [`TuneOptions`] use Download.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum TuneStaging {
    /// Original bidirectional, GPU-preferred shared staging.
    #[default]
    Shared,
    /// Host-read-optimized staging; still used for uploads as well as readbacks.
    Download,
}

/// Resource bounds and decision policy for [`crate::Session::tune_with`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TuneOptions {
    pub max_classes: usize,
    /// GPU scratch including the upload/readback buffer, not pipelines.
    pub max_scratch_bytes: usize,
    /// Defaults to Download. Does not alter scratch binding placement,
    /// validation or kernel candidates.
    /// Missing historical settings deserialize to the original shared staging.
    #[serde(default)]
    pub staging: TuneStaging,
    /// Soft wall-clock deadline, including compilation and qualification.
    /// An in-flight driver call, GPU submission, or validation cannot be preempted.
    pub max_time: Duration,
    pub warmup_runs: u32,
    /// Complete, alternating baseline/candidate pairs required for a decision.
    pub sample_pairs: usize,
    /// Separate, barrier-delimited dispatches in each timed submission.
    pub dispatches_per_sample: u32,
    /// Required fractional improvement, in addition to a noise margin.
    pub min_improvement: f64,
}

impl Default for TuneOptions {
    fn default() -> Self {
        Self {
            max_classes: 8,
            max_scratch_bytes: 64 * 1024 * 1024,
            staging: TuneStaging::Download,
            max_time: Duration::from_secs(2),
            warmup_runs: 1,
            sample_pairs: 6,
            dispatches_per_sample: 16,
            min_improvement: 0.05,
        }
    }
}

/// An invalid search configuration; rejected before touching the GPU.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TuneError(pub &'static str);

impl std::fmt::Display for TuneError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.0)
    }
}

impl std::error::Error for TuneError {}

impl TuneOptions {
    pub(crate) fn validate(&self) -> Result<(), TuneError> {
        if !(4..=64).contains(&self.sample_pairs) {
            return Err(TuneError("sample_pairs must be between 4 and 64"));
        }
        if !(1..=256).contains(&self.dispatches_per_sample) {
            return Err(TuneError("dispatches_per_sample must be between 1 and 256"));
        }
        if self.warmup_runs > 32 {
            return Err(TuneError("warmup_runs must not exceed 32"));
        }
        if !self.min_improvement.is_finite() || !(0.0..1.0).contains(&self.min_improvement) {
            return Err(TuneError("min_improvement must be finite and in [0, 1)"));
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TuneDecision {
    FasterCandidate,
    KeepBaseline,
    InvalidOutput,
    InvalidTiming,
    TimeBudget,
    ScratchLimit,
    DeviceMemoryBudget,
    ShaderRejected,
}

/// Non-overlapping host wall-time phases within one candidate comparison.
/// `None` means the phase was not reached; an early exit records partial time.
/// Their sum excludes final decision bookkeeping. Historical reports did not
/// measure cleanup; missing measurements are not zero cost.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TunePhaseTimes {
    /// Legality/budget checks, pipeline setup, scratch allocation and bindings.
    /// Includes [`TuneOutcome::compile_time`]; do not add that timer again.
    pub preparation: Option<Duration>,
    /// Nested accounting, already inside `preparation`.
    #[serde(default)]
    pub preparation_breakdown: Option<TunePreparationTimes>,
    /// Input generation, uploads, trial dispatches, readbacks and CPU checks.
    pub qualification: Option<Duration>,
    /// Nested accounting, already inside `qualification`; never add it twice.
    /// `None` means an older report or qualification was not reached.
    #[serde(default)]
    pub qualification_breakdown: Option<TuneQualificationTimes>,
    /// Restoring ordinary-magnitude scratch inputs and warming both variants.
    pub warmup: Option<Duration>,
    /// Paired timing loop, including incomplete/discarded pairs and host checks.
    /// Not the sum of accepted per-dispatch samples or GPU timestamp duration.
    pub sampling: Option<Duration>,
    /// Destruction of the private comparison resources, including early exits.
    #[serde(default)]
    pub cleanup: Option<Duration>,
}

/// Disjoint host wall times within preparation, not GPU timestamps.
/// `None` means historical/unreached work; repeated operations accumulate.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TunePreparationTimes {
    pub checks: Option<Duration>,
    /// Same measurement as [`TuneOutcome::compile_time`]; do not add it twice.
    pub pipelines: Option<Duration>,
    /// Candidate binding allocations with matching memory placement.
    pub buffers: Option<Duration>,
    /// The private upload/readback allocation.
    pub staging: Option<Duration>,
    pub encoder: Option<Duration>,
    /// Dispatch cloning, geometry and pipeline lookup, not GPU execution.
    pub bindings: Option<Duration>,
}

/// Accumulated, disjoint host wall times within qualification, not GPU timestamps.
/// Repeated operations add to each field; `None` means the operation was not
/// reached. Early returns preserve partial time. Bookkeeping and destruction
/// outside these scopes remain in the enclosing qualification time.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TuneQualificationTimes {
    /// Deterministic inputs, padding and NaN output sentinels.
    pub input_preparation: Option<Duration>,
    /// CPU copy into mapped staging memory.
    pub upload_host_copy: Option<Duration>,
    /// Upload encoding, transfer, submission and wait.
    pub upload_transfer: Option<Duration>,
    /// Candidate encoding, dispatch, submission and wait.
    pub dispatch: Option<Duration>,
    /// Readback encoding, transfer, submission and wait.
    pub readback_transfer: Option<Duration>,
    /// Host vector allocation and CPU copy out of mapped staging memory.
    pub readback_host_copy: Option<Duration>,
    /// Full-output finite/parity scans and sampled f64 reference dots.
    pub validation: Option<Duration>,
}

/// Evidence for one candidate comparison within an exact class. Times are
/// batched scratch wall times per dispatch, not GPU timestamps or predicted
/// whole-step latency.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TuneOutcome {
    pub class: TuneClass,
    pub dispatches: usize,
    pub initial: MatmulTile,
    pub candidate: MatmulTile,
    pub selected: MatmulTile,
    pub decision: TuneDecision,
    pub qualified: bool,
    /// Shader rejection or the candidate/input pattern that failed qualification.
    pub failure: Option<String>,
    /// Total comparison cost, including pipeline setup, qualification and timing.
    pub elapsed: Duration,
    pub compile_time: Duration,
    /// Phase accounting is absent in reports written before this instrumentation.
    /// Never reinterpret missing historical measurements as zero-cost phases.
    #[serde(default)]
    pub phase_times: Option<TunePhaseTimes>,
    pub baseline_ms: Vec<f64>,
    pub candidate_ms: Vec<f64>,
    pub baseline_median_ms: Option<f64>,
    pub candidate_median_ms: Option<f64>,
    /// Twice the median absolute deviation of paired time differences.
    /// A conservative selection guard, not a statistical confidence interval.
    pub noise_margin_ms: Option<f64>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TuneReport {
    pub options: TuneOptions,
    pub outcomes: Vec<TuneOutcome>,
    pub eligible_classes: usize,
    /// A class may have up to two challenger comparisons in `outcomes`.
    pub visited_classes: usize,
    pub excluded_dispatches: usize,
    pub class_limit_reached: bool,
    pub time_budget_exhausted: bool,
    pub elapsed: Duration,
}

pub(crate) fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        sorted[mid - 1] / 2.0 + sorted[mid] / 2.0
    } else {
        sorted[mid]
    }
}

/// Reusable two-candidate runner: the callback owns timing and deadline checks.
/// `false` is the baseline, `true` the alternative. Partial pairs never win.
pub(crate) fn measure_pairs(
    pairs: usize,
    mut measure: impl FnMut(bool) -> Option<f64>,
) -> (Vec<f64>, Vec<f64>) {
    let mut baseline = Vec::new();
    let mut candidate = Vec::new();
    for pair in 0..pairs {
        let order = if pair % 2 == 0 {
            [false, true]
        } else {
            [true, false]
        };
        let Some(first) = measure(order[0]) else {
            break;
        };
        let Some(second) = measure(order[1]) else {
            break;
        };
        let times = if order[0] {
            [second, first]
        } else {
            [first, second]
        };
        baseline.push(times[0]);
        candidate.push(times[1]);
    }
    (baseline, candidate)
}

pub(crate) fn decide(outcome: &mut TuneOutcome, options: &TuneOptions) {
    if outcome.baseline_ms.len() != options.sample_pairs
        || outcome.candidate_ms.len() != options.sample_pairs
    {
        outcome.decision = TuneDecision::TimeBudget;
        return;
    }
    if outcome
        .baseline_ms
        .iter()
        .chain(&outcome.candidate_ms)
        .any(|x| !x.is_finite() || *x <= 0.0)
    {
        outcome.decision = TuneDecision::InvalidTiming;
        return;
    }
    let baseline = median(&outcome.baseline_ms);
    let candidate = median(&outcome.candidate_ms);
    let differences: Vec<_> = outcome
        .baseline_ms
        .iter()
        .zip(&outcome.candidate_ms)
        .map(|(b, c)| b - c)
        .collect();
    let gain = median(&differences);
    let deviations: Vec<_> = differences.iter().map(|d| (d - gain).abs()).collect();
    let noise = 2.0 * median(&deviations);
    outcome.baseline_median_ms = Some(baseline);
    outcome.candidate_median_ms = Some(candidate);
    outcome.noise_margin_ms = Some(noise);
    if outcome.qualified
        && baseline - candidate > baseline * options.min_improvement + noise
        && gain > baseline * options.min_improvement + noise
    {
        outcome.selected = outcome.candidate;
        outcome.decision = TuneDecision::FasterCandidate;
    } else {
        outcome.decision = TuneDecision::KeepBaseline;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dispatch() -> Dispatch {
        Dispatch {
            params: vec![33, 17, 65, 0],
            workgroups: [2, 1, 1],
            input_buffers: vec![crate::compile::BufferRef(0), crate::compile::BufferRef(1)],
            output_buffer: crate::compile::BufferRef(2),
            ..Default::default()
        }
    }

    #[test]
    fn complete_geometry_handles_edges_in_both_directions() {
        let mut d = dispatch();
        let class = TuneClass::from_dispatch(&d, None).unwrap();
        assert_eq!((class.m, class.n, class.k), (33, 65, 17));
        MatmulTile::Tile32.apply(&mut d, &class);
        assert_eq!(d.workgroups, [3, 2, 1]);
        assert!(d.use_small_tiles);
        MatmulTile::Tile64.apply(&mut d, &class);
        assert_eq!(d.workgroups, [2, 1, 1]);
        assert!(!d.use_small_tiles);
    }

    #[test]
    fn class_separates_shape_direction_precision_and_placement() {
        let d = dispatch();
        let base = TuneClass::from_dispatch(&d, None).unwrap();
        let mut changed = base.clone();
        changed.k += 1;
        assert_ne!(base, changed);
        changed = base.clone();
        changed.device_local[0] = true;
        assert_ne!(base, changed);
        changed = base.clone();
        changed.requires_full_precision = true;
        assert_ne!(base, changed);
        changed = base.clone();
        changed.binding_bytes = vec![4096; 3];
        assert_ne!(base, changed);
        let mut at = d;
        at.shader = ShaderEntry::MatMulAT;
        at.params = vec![33, 65, 17, 0];
        let at_class = TuneClass::from_dispatch(&at, None).unwrap();
        assert_eq!((at_class.m, at_class.n, at_class.k), (33, 65, 17));
        assert_ne!(base, at_class);
    }

    #[test]
    fn unsupported_modifiers_never_enter_the_search() {
        let base = dispatch();
        let mut variants = vec![base.clone(); 10];
        variants[0].use_coop = true;
        variants[1].horizontal_batch = 2;
        variants[2].weight_format = crate::compile::WeightFormat::F16;
        variants[3].extra_outputs.push(crate::compile::BufferRef(3));
        variants[4].workgroups[2] = 2;
        variants[5].shader = ShaderEntry::MatMulGemv;
        variants[6].params[0] = 0;
        variants[7].params[3] = 1;
        variants[8].params[0] = 32 * 65_535 + 1;
        variants[9].workgroups = [1, 2, 1]; // wrong row/column geometry
        for d in variants {
            assert!(TuneClass::from_dispatch(&d, None).is_none());
        }
        let mut class = TuneClass::from_dispatch(&base, None).unwrap();
        class.m = u32::MAX;
        assert!(class.buffer_sizes().is_none());
    }

    #[test]
    fn trials_alternate_and_discard_incomplete_pairs() {
        let mut calls = Vec::new();
        let (baseline, candidate) = measure_pairs(4, |alternative| {
            calls.push(alternative);
            (calls.len() < 6).then_some(if alternative { 2.0 } else { 3.0 })
        });
        assert_eq!(calls, [false, true, true, false, false, true]);
        assert_eq!(baseline, [3.0, 3.0]);
        assert_eq!(candidate, [2.0, 2.0]);
    }

    fn outcome() -> TuneOutcome {
        TuneOutcome {
            class: TuneClass::from_dispatch(&dispatch(), None).unwrap(),
            dispatches: 3,
            initial: MatmulTile::Tile32,
            candidate: MatmulTile::Tile64,
            selected: MatmulTile::Tile32,
            decision: TuneDecision::KeepBaseline,
            qualified: true,
            failure: None,
            elapsed: Duration::ZERO,
            compile_time: Duration::ZERO,
            phase_times: None,
            baseline_ms: vec![10.0; 6],
            candidate_ms: vec![8.0; 6],
            baseline_median_ms: None,
            candidate_median_ms: None,
            noise_margin_ms: None,
        }
    }

    #[test]
    fn decision_requires_complete_valid_qualified_low_noise_gain() {
        let options = TuneOptions::default();
        let mut win = outcome();
        decide(&mut win, &options);
        assert_eq!(win.selected, MatmulTile::Tile64);
        assert_eq!(win.decision, TuneDecision::FasterCandidate);
        let mut reverse = outcome();
        reverse.initial = MatmulTile::Tile64;
        reverse.candidate = MatmulTile::Tile32;
        reverse.selected = MatmulTile::Tile64;
        decide(&mut reverse, &options);
        assert_eq!(reverse.selected, MatmulTile::Tile32);
        for candidate in [vec![9.6; 6], vec![6.0, 10.0, 6.0, 10.0, 6.0, 10.0]] {
            let mut noisy = outcome();
            noisy.candidate_ms = candidate;
            decide(&mut noisy, &options);
            assert_eq!(noisy.selected, noisy.initial);
        }
        let mut incomplete = outcome();
        incomplete.candidate_ms.pop();
        decide(&mut incomplete, &options);
        assert_eq!(incomplete.decision, TuneDecision::TimeBudget);
        for value in [f64::NAN, f64::INFINITY, 0.0, -1.0] {
            let mut invalid = outcome();
            invalid.candidate_ms[0] = value;
            decide(&mut invalid, &options);
            assert_eq!(invalid.decision, TuneDecision::InvalidTiming);
        }
        let mut invalid = outcome();
        invalid.qualified = false;
        decide(&mut invalid, &options);
        assert_eq!(invalid.selected, invalid.initial);
    }

    #[test]
    fn configuration_validation_is_cpu_only() {
        assert!(TuneOptions::default().validate().is_ok());
        for options in [
            TuneOptions {
                sample_pairs: 0,
                ..Default::default()
            },
            TuneOptions {
                dispatches_per_sample: 0,
                ..Default::default()
            },
            TuneOptions {
                min_improvement: f64::NAN,
                ..Default::default()
            },
        ] {
            assert!(options.validate().is_err());
        }
    }

    #[test]
    fn report_round_trips_settings_and_raw_evidence() {
        let report = TuneReport {
            outcomes: vec![outcome()],
            eligible_classes: 1,
            ..Default::default()
        };
        let json = serde_json::to_string_pretty(&report).unwrap();
        let restored: TuneReport = serde_json::from_str(&json).unwrap();
        assert_eq!(
            restored.outcomes[0].baseline_ms,
            report.outcomes[0].baseline_ms
        );
        assert_eq!(restored.options.max_time, report.options.max_time);
        assert_eq!(restored.outcomes[0].class, report.outcomes[0].class);
    }

    #[test]
    fn phase_times_distinguish_legacy_missing_unreached_and_measured() {
        let mut outcome = outcome();
        outcome.phase_times = Some(TunePhaseTimes {
            preparation: Some(Duration::from_millis(3)),
            preparation_breakdown: Some(TunePreparationTimes {
                buffers: Some(Duration::from_millis(1)),
                ..Default::default()
            }),
            qualification: Some(Duration::ZERO),
            cleanup: Some(Duration::from_millis(1)),
            qualification_breakdown: Some(TuneQualificationTimes {
                upload_host_copy: Some(Duration::ZERO),
                ..Default::default()
            }),
            ..Default::default()
        });
        let mut json = serde_json::to_value(&outcome).unwrap();
        let restored: TuneOutcome = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(restored.phase_times, outcome.phase_times);
        for key in ["preparation_breakdown", "cleanup"] {
            json["phase_times"].as_object_mut().unwrap().remove(key);
        }
        let older: TuneOutcome = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(older.phase_times.unwrap().preparation_breakdown, None);
        assert_eq!(older.phase_times.unwrap().cleanup, None);
        json["phase_times"]
            .as_object_mut()
            .unwrap()
            .remove("qualification_breakdown");
        let older: TuneOutcome = serde_json::from_value(json.clone()).unwrap();
        assert_eq!(older.phase_times.unwrap().qualification_breakdown, None);
        json.as_object_mut().unwrap().remove("phase_times");
        let legacy: TuneOutcome = serde_json::from_value(json).unwrap();
        assert_eq!(legacy.phase_times, None);
    }

    #[test]
    fn staging_round_trips_and_missing_historical_policy_is_shared() {
        assert_eq!(TuneOptions::default().staging, TuneStaging::Download);
        assert_eq!(TuneStaging::default(), TuneStaging::Shared);
        for staging in [TuneStaging::Shared, TuneStaging::Download] {
            let options = TuneOptions {
                staging,
                ..Default::default()
            };
            let mut value = serde_json::to_value(&options).unwrap();
            let restored: TuneOptions = serde_json::from_value(value.clone()).unwrap();
            assert_eq!(restored.staging, staging);
            value.as_object_mut().unwrap().remove("staging");
            let historical: TuneOptions = serde_json::from_value(value).unwrap();
            assert_eq!(historical.staging, TuneStaging::Shared);
        }
    }

    fn native_config(tile_size: u32) -> CoopConfig {
        CoopConfig {
            tile_size,
            use_f16_input: false,
            compensated: false,
        }
    }

    fn class(m: u32, n: u32, k: u32) -> TuneClass {
        let mut class = TuneClass {
            shader: ShaderEntry::MatMul,
            m,
            n,
            k,
            requires_full_precision: true,
            device_local: [false; 4],
            binding_bytes: Vec::new(),
        };
        class.binding_bytes = class.buffer_sizes().unwrap();
        class
    }

    #[test]
    fn native_candidates_ignore_profitability_but_obey_capability_and_precision() {
        let config = native_config(8);
        let native = MatmulTile::CooperativeF32 { tile_size: 8 };
        // One workgroup, and then a dimension above the static 1024 veto.
        for class in [class(16, 16, 8), class(2048, 32, 17)] {
            assert!(
                class
                    .challengers(MatmulTile::Tile64, Some(&config))
                    .contains(&native)
            );
            assert!(
                !class
                    .challengers(MatmulTile::Tile64, None)
                    .contains(&native)
            );
            let f16 = CoopConfig {
                use_f16_input: true,
                ..config
            };
            assert!(
                !class
                    .challengers(MatmulTile::Tile64, Some(&f16))
                    .contains(&native)
            );
            let compensated = CoopConfig {
                compensated: true,
                ..config
            };
            assert!(
                !class
                    .challengers(MatmulTile::Tile64, Some(&compensated))
                    .contains(&native)
            );
        }
        assert!(MatmulTile::native_cooperative(Some(&native_config(4))).is_none());
    }

    #[test]
    fn native_padding_and_addend_capacity_are_legality_not_allocation_requests() {
        let native = MatmulTile::CooperativeF32 { tile_size: 8 };
        let mut class = class(33, 32, 17);
        assert!(!native.fits(&class));
        let required = native.buffer_sizes(&class).unwrap();
        assert_eq!(required, [33 * 17 * 4, 17 * 32 * 4, 48 * 32 * 4]);
        class.binding_bytes = required;
        assert!(native.fits(&class));
        class.shader = ShaderEntry::FusedMatMulAdd;
        class.binding_bytes = class.buffer_sizes().unwrap();
        class.binding_bytes[3] = 48 * 32 * 4;
        assert!(!native.fits(&class)); // output fits, addend does not
        class.binding_bytes[2] = 48 * 32 * 4;
        assert!(native.fits(&class));
        class.n = 17;
        assert!(native.buffer_sizes(&class).is_none());
        class.n = 32;
        class.k = 1;
        assert!(native.buffer_sizes(&class).is_none());
        class.shader = ShaderEntry::MatMulBT;
        assert!(native.buffer_sizes(&class).is_some());
    }

    #[test]
    fn native_geometry_flags_and_scalar_fallback_move_together() {
        let config = native_config(8);
        let native = MatmulTile::CooperativeF32 { tile_size: 8 };
        let class = class(32, 64, 17);
        let mut d = dispatch();
        d.params = vec![class.m, class.k, class.n, 0];
        native.apply(&mut d, &class);
        assert_eq!(d.workgroups, [2, 4, 1]);
        assert!(d.use_coop && !d.use_coop_compensated && !d.use_small_tiles);
        assert_eq!(d.scalar_fallback, Some((ShaderEntry::MatMul, [1, 1, 1])));
        assert!(TuneClass::from_dispatch(&d, Some(&config)).is_some());
        assert!(TuneClass::from_dispatch(&d, None).is_none());
        MatmulTile::Tile32.apply(&mut d, &class);
        assert_eq!(d.workgroups, [2, 1, 1]);
        assert!(!d.use_coop && d.use_small_tiles && d.scalar_fallback.is_none());
        assert!(TuneClass::from_dispatch(&d, Some(&config)).is_some());
    }

    #[test]
    fn an_incomplete_next_challenger_keeps_the_completed_winner() {
        let options = TuneOptions::default();
        let mut first = outcome();
        decide(&mut first, &options);
        let mut next = outcome();
        next.initial = first.selected;
        next.selected = first.selected;
        next.candidate = MatmulTile::CooperativeF32 { tile_size: 8 };
        next.candidate_ms.pop();
        decide(&mut next, &options);
        assert_eq!(next.selected, first.selected);
        assert_eq!(next.decision, TuneDecision::TimeBudget);
        next.candidate_ms.push(8.0);
        decide(&mut next, &options);
        assert_eq!(next.selected, next.candidate);
    }
}
