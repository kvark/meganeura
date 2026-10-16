//! Validation policy for native cooperative products with F32 accumulators.
//!
//! Vulkan leaves their internal precision implementation-dependent. In
//! particular, an F32 accumulator does not promise scalar IEEE FMA semantics.
//! This is Meganeura's acceptance budget, not a hardware error-bound theorem.

/// Absolute error budget for a finite native dot product, after rounding its
/// operands to the multiplication format, before narrowing its output.
///
/// Use `sum_abs = Σ|a*b| + |addend|`, and the full K dimension for `terms`.
/// The budget is `gamma(2*K + 1) * sum_abs`, with F32 unit roundoff `u=2^-24`
/// and `gamma(n)=n*u/(1-n*u)`: the usual conservative serial multiply/add
/// rounding budget. It allows native arithmetic error even on exactly
/// representable fixtures, without using FP16/BF16 epsilon for accumulation.
/// Split reductions fit within this operation count; do not multiply it by
/// the partition count. Underflow/overflow and non-finite operands require
/// separate tests. This function deliberately has no absolute or tensor-wide
/// floor, so it cannot excuse flushing a small, well-conditioned result.
///
/// `None` means the scale is invalid or the operation count makes this bound
/// uninformative; callers must not silently accept the result in that case.
pub fn error_bound(terms: usize, sum_abs: f64) -> Option<f64> {
    let nu = (2.0 * terms as f64 + 1.0) * (f64::from(f32::EPSILON) / 2.0);
    if !sum_abs.is_finite() || sum_abs < 0.0 || nu >= 1.0 {
        return None;
    }
    let bound = nu / (1.0 - nu) * sum_abs;
    bound.is_finite().then_some(bound)
}

#[cfg(test)]
mod tests {
    use super::error_bound;

    #[test]
    fn accepts_native_ulp_but_not_wrong_product() {
        let bound = error_bound(16, 1.0).unwrap();
        let observed = f64::from(f32::from_bits(0x3f7fffff));
        assert!((observed - 1.0).abs() <= bound);
        assert!(bound < 1.0 / 1024.0);
        assert!(1.0 > bound); // A missing product must fail.
    }

    #[test]
    fn scales_with_inputs_not_cancelled_output() {
        let bound = error_bound(32, 2.0).unwrap();
        let scale = 2.0f64.powi(-40);
        assert_eq!(error_bound(32, 2.0 * scale), Some(bound * scale));
        let subnormal_sum = 2.0f64.powi(-10);
        assert!(error_bound(32, subnormal_sum).unwrap() < subnormal_sum);
        assert_eq!(error_bound(32, 0.0), Some(0.0));
    }

    #[test]
    fn invalid_budgets_are_not_infinite_tolerance() {
        for scale in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(error_bound(1, scale).is_none());
        }
        assert!(error_bound(1 << 23, 1.0).is_none());
    }
}
