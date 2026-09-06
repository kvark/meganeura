//! CPU-only evidence arithmetic shared by the holdout runner and replay tests.

use serde::{Deserialize, Serialize};

pub fn median(values: &[f64]) -> f64 {
    assert!(!values.is_empty());
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        sorted[mid - 1] * 0.5 + sorted[mid] * 0.5
    } else {
        sorted[mid]
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct PairedTiming {
    pub baseline_median_ms: f64,
    pub tuned_median_ms: f64,
    pub speedup: f64,
    pub paired_gain_median_ms: f64,
    pub paired_noise_margin_ms: f64,
    pub improvement_exceeds_guard: bool,
    pub regression_exceeds_guard: bool,
}

impl PairedTiming {
    pub fn new(baseline: &[f64], tuned: &[f64]) -> Result<Self, &'static str> {
        if baseline.len() != tuned.len()
            || baseline.is_empty()
            || !baseline
                .iter()
                .chain(tuned)
                .all(|v| v.is_finite() && *v > 0.0)
        {
            return Err("timings must be complete, finite, positive pairs");
        }
        let a = median(baseline);
        let b = median(tuned);
        let differences: Vec<_> = baseline.iter().zip(tuned).map(|(a, b)| a - b).collect();
        let gain = median(&differences);
        let noise = 2.0
            * median(
                &differences
                    .iter()
                    .map(|d| (d - gain).abs())
                    .collect::<Vec<_>>(),
            );
        Ok(Self {
            baseline_median_ms: a,
            tuned_median_ms: b,
            speedup: a / b,
            paired_gain_median_ms: gain,
            paired_noise_margin_ms: noise,
            improvement_exceeds_guard: a - b > 0.05 * a + noise && gain > 0.05 * a + noise,
            regression_exceeds_guard: b - a > 0.05 * a + noise && -gain > 0.05 * a + noise,
        })
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TensorComparison {
    pub name: String,
    pub elements: usize,
    pub nonfinite_pairs: usize,
    pub exact: bool,
    pub reference_sum_sq: f64,
    pub candidate_sum_sq: f64,
    pub error_sum_sq: f64,
    pub relative_l2: f64,
    pub max_abs_error: f64,
    pub elementwise_failures: usize,
    pub passed: bool,
}

impl TensorComparison {
    pub fn new(name: String, reference: &[f32], candidate: &[f32]) -> Self {
        assert_eq!(reference.len(), candidate.len(), "tensor layout changed");
        let mut result = Self {
            name,
            elements: reference.len(),
            nonfinite_pairs: 0,
            exact: true,
            reference_sum_sq: 0.0,
            candidate_sum_sq: 0.0,
            error_sum_sq: 0.0,
            relative_l2: 0.0,
            max_abs_error: 0.0,
            elementwise_failures: 0,
            passed: false,
        };
        for (&a, &b) in reference.iter().zip(candidate) {
            result.exact &= a.to_bits() == b.to_bits();
            if !a.is_finite() || !b.is_finite() {
                result.nonfinite_pairs += 1;
                continue;
            }
            let (a, b) = (f64::from(a), f64::from(b));
            let error = (a - b).abs();
            result.reference_sum_sq += a * a;
            result.candidate_sum_sq += b * b;
            result.error_sum_sq += error * error;
            result.max_abs_error = result.max_abs_error.max(error);
            result.elementwise_failures += usize::from(error > 1e-6 + 2e-4 * a.abs());
        }
        result.relative_l2 = (result.error_sum_sq / result.reference_sum_sq.max(1e-30)).sqrt();
        result.passed = result.nonfinite_pairs == 0
            && result.relative_l2 <= 2e-4
            && result.elementwise_failures == 0;
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paired_timing_handles_wins_regressions_noise_and_bad_samples() {
        let win = PairedTiming::new(&[10.0; 4], &[8.0; 4]).unwrap();
        assert!(win.improvement_exceeds_guard && !win.regression_exceeds_guard);
        assert_eq!(win.speedup, 1.25);
        let loss = PairedTiming::new(&[10.0; 4], &[12.0; 4]).unwrap();
        assert!(!loss.improvement_exceeds_guard && loss.regression_exceeds_guard);
        let noisy = PairedTiming::new(&[10.0; 4], &[6.0, 10.0, 6.0, 10.0]).unwrap();
        assert!(!noisy.improvement_exceeds_guard);
        assert!(PairedTiming::new(&[], &[]).is_err());
        assert!(PairedTiming::new(&[1.0], &[1.0, 2.0]).is_err());
        for value in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(PairedTiming::new(&[1.0], &[value]).is_err());
        }
    }

    #[test]
    fn tensor_comparison_rejects_sign_errors_tiny_signal_loss_and_nonfinite_values() {
        for (a, b) in [(1.0, -1.0), (1e-12, 0.0), (f32::NAN, f32::NAN)] {
            assert!(!TensorComparison::new("test".into(), &[a], &[b]).passed);
        }
        let same = TensorComparison::new("test".into(), &[1.0, -2.0], &[1.0, -2.0]);
        assert!(same.passed && same.exact);
        assert_eq!(same.reference_sum_sq, 5.0);
    }
}
