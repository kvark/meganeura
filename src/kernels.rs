//! Kernel families: interchangeable implementations of one computation.
//!
//! A family owns everything that varies between its implementations, so a
//! new specialization is a new variant here rather than a new shader entry
//! threaded through every table in the compiler and runtime:
//!
//! - the variants, in order of preference, with the last a fallback that
//!   admits every problem the family accepts;
//! - what each variant admits on a given target, stated as a reason when it
//!   declines;
//! - its geometry, generated module, binding layout and required
//!   capabilities.
//!
//! The rest of the crate sees one shader entry per family and asks the
//! family about the variant it carries. Selection picks the first variant
//! a problem admits, so a specialized path only ever replaces a fallback
//! that computes the same thing, and the tests can run every admitted
//! variant against the same reference.

pub(crate) mod attention;
pub mod attention_grad;
pub(crate) mod conv;
pub mod matmul;

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

    /// A variant `target` asks to try before the usual order, as a test
    /// pinning each variant in turn does. It is still subject to
    /// [`Family::admits`].
    fn preferred(_target: &Self::Target) -> Option<Self> {
        None
    }

    /// Output rows and columns that one workgroup of this variant covers
    /// for a problem like `problem` on `target`, or `None` where the target
    /// has no such kernel. The [`sweep`] places its problems around
    /// multiples of it, so a variant is tested at its own tile edges as
    /// soon as it states them.
    #[cfg_attr(not(test), allow(dead_code))]
    fn tile(self, problem: &Self::Problem, target: &Self::Target) -> Option<[u32; 2]>;
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
/// Panics when none does. The fallback declines only problems no variant
/// can run: shapes the graph builders refuse, or launches beyond the
/// portable dispatch limit that the preferred tilings stay within.
pub(crate) fn select<F: Family>(problem: &F::Problem, target: &F::Target) -> Selection<F> {
    let mut declined = Vec::new();
    let preferred = F::preferred(target);
    let rest = F::PREFERENCE.iter().filter(|&&v| Some(v) != preferred);
    for &variant in preferred.iter().chain(rest) {
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

/// Every variant of a family against the `f64` reference, at extents that
/// straddle its tiles.
///
/// The variants and their tiles come from the family, so one added to
/// [`Family::PREFERENCE`] is run without a test being written for it. Each
/// variant is preferred in turn and runs wherever it admits the problem. A
/// variant the device lacks, and a problem a variant declines, are printed
/// with the reason the family states.
#[cfg(test)]
pub(crate) mod sweep {
    use super::Family;
    use crate::compile::Dispatch;
    use crate::reference::{Feeds, gpu};
    use crate::{Graph, Mode, Session, graph::Op};

    /// What a family supplies so that each of its variants can be built,
    /// preferred and recognized in a session.
    pub(crate) trait Swept: Family {
        /// What sets one problem apart from another of the same extents: a
        /// product's layout, a convolution's window.
        type Kind: Copy + std::fmt::Debug;
        /// Every kind the family computes.
        const KINDS: &'static [Self::Kind];

        /// A graph whose one dispatch of the family computes `kind` over
        /// `rows` by `columns` outputs, each reduced from `depth` terms.
        fn graph(kind: Self::Kind, extents: [u32; 3]) -> Graph;
        /// The problem that graph compiles to. Should the two ever differ
        /// in what a variant admits, the sweep fails on the variant the
        /// session ran.
        fn problem(kind: Self::Kind, extents: [u32; 3]) -> Self::Problem;
        /// Ask sessions built with `options` to try `self` first.
        fn prefer(self, options: &mut gpu::Options);
        /// What `session`, built with `options`, allows, with `self`
        /// preferred.
        fn target(self, session: &Session, options: &gpu::Options) -> Self::Target;
        /// For a lowered `dispatch` of the family, the variant whose kernel
        /// it runs, or `None` when a later pass replaced the family's choice
        /// with a kernel of its own.
        fn ran(dispatch: &Dispatch) -> Option<Option<Self>>;
    }

    /// Tiles along each axis to try, smallest first. A variant with an
    /// occupancy floor first admits a problem at one of the larger ones.
    const SCALES: [u32; 5] = [1, 2, 4, 8, 16];

    /// Extents of the problem a variant is asked its tile for.
    const SEED: [u32; 3] = [64, 64, 16];

    /// The extents of `scale` whole tiles a side, and ragged ones around
    /// them: one past both edges, one short of both, half a tile past both,
    /// and one row past. The kernels read 16 or 32 terms of a reduction at a
    /// time. The reduction lengths are one such stage for whole tiles, one
    /// term past the longer stage, one short of and one past the shorter,
    /// and the fewest a vector load takes where only a row is added.
    fn lattice([rows, columns]: [u32; 2], scale: u32) -> ([u32; 3], Vec<[u32; 3]>) {
        let (m, n) = (scale * rows, scale * columns);
        let mut ragged = vec![
            [m + 1, n + 1, 33],
            [m - 1, n - 1, 15],
            [m + rows / 2, n + columns / 2, 17],
            [m + 1, n, 4],
        ];
        // A one-row tile has nothing short of it, and no half.
        ragged.retain(|&[rows, columns, _]| rows * columns != 0 && [rows, columns] != [m, n]);
        ([m, n, 16], ragged)
    }

    /// Operands every kernel reads exactly, f16 tiles included: multiples of
    /// 1/64 between -31/64 and 31/64.
    fn feeds(graph: &Graph) -> Feeds {
        let mut feeds = Feeds::new();
        for node in graph.nodes() {
            if let Op::Parameter { ref name } | Op::Input { ref name } = node.op {
                let seed = 17 + 6 * node.id as usize;
                let values: Vec<f32> = (0..node.ty.num_elements())
                    .map(|i| ((i * seed) % 63) as f32 / 64.0 - 31.0 / 64.0)
                    .collect();
                feeds.set(name, &values);
            }
        }
        feeds
    }

    /// Run the sweep for `F` on the shared device.
    ///
    /// Each variant runs each kind at the first scale where it takes whole
    /// tiles, and at the first ragged extents it admits there. The ragged
    /// extents are tried from a different one for each kind, so that
    /// between them the kinds cover every edge.
    pub(crate) fn run<F: Swept>() {
        let defaults = gpu::Options::default();
        // Any session of these options states what the device allows.
        let probe = gpu::session(&F::graph(F::KINDS[0], SEED), &defaults, Mode::Inference);
        let mut idle = Vec::new();
        for &variant in F::PREFERENCE {
            let target = variant.target(&probe, &defaults);
            let mut options = defaults.clone();
            variant.prefer(&mut options);
            let admits = |kind, extents| variant.admits(&F::problem(kind, extents), &target);
            let mut checks = None;
            for (index, &kind) in F::KINDS.iter().enumerate() {
                let Some(tile) = variant.tile(&F::problem(kind, SEED), &target) else {
                    continue;
                };
                let checks = checks.get_or_insert(0);
                let mut reason = "";
                let whole = SCALES.into_iter().find_map(|scale| {
                    let (exact, ragged) = lattice(tile, scale);
                    match admits(kind, exact) {
                        Ok(()) => Some((exact, ragged)),
                        Err(declined) => {
                            reason = declined;
                            None
                        }
                    }
                });
                let Some((exact, mut ragged)) = whole else {
                    println!("{variant:?} {kind:?}: declined, {reason}");
                    continue;
                };
                let first = index % ragged.len();
                ragged.rotate_left(first);
                let tail = ragged.into_iter().find(|&extents| {
                    admits(kind, extents)
                        .map_err(|reason| {
                            println!("{variant:?} {kind:?} {extents:?}: declined, {reason}")
                        })
                        .is_ok()
                });
                for extents in [exact].into_iter().chain(tail) {
                    let what = format!("{variant:?} {kind:?} {extents:?}");
                    let graph = F::graph(kind, extents);
                    let (report, session) = gpu::run_inference(&graph, &feeds(&graph), &options)
                        .unwrap_or_else(|error| panic!("{what}: {error}"));
                    report.assert_passed(&what);
                    let ran: Vec<_> = session
                        .plan()
                        .dispatches
                        .iter()
                        .filter_map(F::ran)
                        .collect();
                    match ran[..] {
                        [Some(ran)] if ran == variant => *checks += 1,
                        // The result above still had to match.
                        [None] => println!("{what}: a later pass replaced the kernel"),
                        _ => panic!("{what}: admitted and preferred, yet the session ran {ran:?}"),
                    }
                }
            }
            match checks {
                Some(0) => idle.push(variant),
                Some(checks) => println!("{variant:?}: {checks} problems match the reference"),
                None => println!("{variant:?}: the device has no such kernel"),
            }
        }
        assert!(
            idle.is_empty(),
            "variants the device has that ran nowhere: {idle:?}"
        );
    }
}
