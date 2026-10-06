//! Kernel families: interchangeable implementations of one computation.
//!
//! A family owns everything that varies between its implementations, so a
//! new specialization is a new variant here rather than a new shader entry
//! threaded through every table in the compiler and runtime:
//!
//! - the variants, in order of preference, with the last a fallback that
//!   admits every problem the family accepts;
//! - what each variant admits on a given target, stated as a reason when it
//!   declines ([`Family::admits`]);
//! - its geometry, generated module, binding layout and required
//!   capabilities.
//!
//! The rest of the crate sees one shader entry per family and asks the
//! family about the variant it carries. [`select`] picks the first variant
//! a problem admits, so a specialized path only ever replaces a fallback
//! that computes the same thing, and the tests can run every admitted
//! variant against the same reference.

pub mod attention_grad;

/// Why a variant declined a problem.
pub(crate) type Rejection = &'static str;

/// A family's variants and their admission rules.
pub(crate) trait Family: Copy + Eq + std::fmt::Debug + 'static {
    /// What one dispatch computes: shapes and per-node pins.
    type Problem: std::fmt::Debug;
    /// What the device and compile options allow.
    type Target: std::fmt::Debug;
    /// Every variant, most preferred first. The last is the fallback.
    const PREFERENCE: &'static [Self];

    /// Whether this variant computes `problem` on `target`, and if not, why.
    fn admits(self, problem: &Self::Problem, target: &Self::Target) -> Result<(), Rejection>;
}

/// The variant [`select`] picked, and the reasons every preferred variant
/// declined.
#[derive(Debug)]
pub(crate) struct Selection<F> {
    pub chosen: F,
    /// Read by tests, which check that each variant declines for the
    /// reason it states.
    #[cfg_attr(not(test), allow(dead_code))]
    pub declined: Vec<(F, Rejection)>,
}

/// The most preferred variant of `F` that admits `problem` on `target`.
///
/// Panics when none does: the fallback declines only problems the family
/// does not support at all, which the graph builders refuse earlier.
pub(crate) fn select<F: Family>(problem: &F::Problem, target: &F::Target) -> Selection<F> {
    let mut declined = Vec::new();
    for &variant in F::PREFERENCE {
        match variant.admits(problem, target) {
            Ok(()) => {
                if !declined.is_empty() {
                    log::debug!("{variant:?} for {problem:?}; declined {declined:?}");
                }
                return Selection {
                    chosen: variant,
                    declined,
                };
            }
            Err(reason) => declined.push((variant, reason)),
        }
    }
    panic!("no kernel admits {problem:?} on {target:?}: {declined:?}")
}
