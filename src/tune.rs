//! Bounded, opt-in kernel selection. See [`crate::Session::tune_with`].
//!
//! The first search space is deliberately small: two scalar f32 tiles for
//! unpacked dense matmuls, with no precision or binding-layout changes.
//! Measurements use synthetic, private scratch, not a live training step.

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
}

/// Scalar f32 implementations with identical bindings and logical extents.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MatmulTile {
    Tile32,
    Tile64,
}

impl MatmulTile {
    pub(crate) fn other(self) -> Self {
        match self {
            Self::Tile32 => Self::Tile64,
            Self::Tile64 => Self::Tile32,
        }
    }

    pub(crate) fn apply(self, dispatch: &mut Dispatch, class: &TuneClass) {
        dispatch.use_small_tiles = self == Self::Tile32;
        dispatch.workgroups = self.workgroups(class);
    }

    fn workgroups(self, class: &TuneClass) -> [u32; 3] {
        let tile = match self {
            Self::Tile32 => 32,
            Self::Tile64 => 64,
        };
        // Scalar tiled kernels use X for columns, Y for rows. Derive exact
        // geometry; doubling rounded 64-tile counts overdispatches edges.
        [class.n.div_ceil(tile), class.m.div_ceil(tile), 1]
    }
}

impl TuneClass {
    pub(crate) fn from_dispatch(dispatch: &Dispatch) -> Option<Self> {
        let addend = dispatch.shader == ShaderEntry::FusedMatMulAdd;
        if !matches!(
            dispatch.shader,
            ShaderEntry::MatMul
                | ShaderEntry::FusedMatMulAdd
                | ShaderEntry::MatMulAT
                | ShaderEntry::MatMulBT
        ) || dispatch.use_coop
            || dispatch.use_coop_compensated
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
        };
        let initial = if dispatch.use_small_tiles {
            MatmulTile::Tile32
        } else {
            MatmulTile::Tile64
        };
        (dispatch.workgroups == initial.workgroups(&class)).then_some(class)
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
}

/// Resource bounds and decision policy for [`crate::Session::tune_with`].
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TuneOptions {
    pub max_classes: usize,
    /// GPU scratch including the shared upload/readback buffer, not pipelines.
    pub max_scratch_bytes: usize,
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
}

/// Evidence for one exact class. Times are batched scratch wall times per
/// dispatch, not GPU timestamps or predicted whole-step latency.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TuneOutcome {
    pub class: TuneClass,
    pub dispatches: usize,
    pub initial: MatmulTile,
    pub selected: MatmulTile,
    pub decision: TuneDecision,
    pub qualified: bool,
    /// Total class cost, including pipeline setup, qualification and timing.
    pub elapsed: Duration,
    pub compile_time: Duration,
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
        outcome.selected = outcome.initial.other();
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
        let class = TuneClass::from_dispatch(&d).unwrap();
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
        let base = TuneClass::from_dispatch(&d).unwrap();
        let mut changed = base.clone();
        changed.k += 1;
        assert_ne!(base, changed);
        changed = base.clone();
        changed.device_local[0] = true;
        assert_ne!(base, changed);
        changed = base.clone();
        changed.requires_full_precision = true;
        assert_ne!(base, changed);
        let mut at = d;
        at.shader = ShaderEntry::MatMulAT;
        at.params = vec![33, 65, 17, 0];
        let at_class = TuneClass::from_dispatch(&at).unwrap();
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
            assert!(TuneClass::from_dispatch(&d).is_none());
        }
        let mut class = TuneClass::from_dispatch(&base).unwrap();
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
            class: TuneClass::from_dispatch(&dispatch()).unwrap(),
            dispatches: 3,
            initial: MatmulTile::Tile32,
            selected: MatmulTile::Tile32,
            decision: TuneDecision::KeepBaseline,
            qualified: true,
            elapsed: Duration::ZERO,
            compile_time: Duration::ZERO,
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
}
