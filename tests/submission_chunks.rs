//! Splitting a plan across submissions must not change what it computes.
//!
//! `Session::set_submission_chunks` lets a caller choose an upper bound on
//! submissions so that a co-tenant on the GPU queue — a renderer, typically
//! — can interleave its work between chunks instead of waiting behind an
//! entire inference. That only pays off if the chunk boundaries are as sound
//! as the pass boundaries they replace.
//!
//! The argument for soundness is that blade ends every command buffer with a
//! conservative global memory barrier, opens each new one assuming an
//! unknown prior producer, and that a Vulkan pipeline barrier's scopes cover
//! commands submitted to the same queue before and after it, not just those
//! in the same command buffer. A wrong argument there would show up as
//! sporadically stale reads at exactly the seams, so these tests use a deep
//! chain where every layer depends on the previous one, and check equality
//! against the single-submission result.

use crate::support::chain;
use meganeura::train::{Mode, SessionConfig};

/// The chain is deep enough to have many barrier groups, so chunk boundaries
/// land in the middle of real dependencies rather than at a convenient seam.
fn run(chunks: usize, layers: usize, rows: usize, dim: usize, steps: usize) -> Vec<f32> {
    let g = chain::build_chain(layers, rows, dim);
    let (mut session, _) = meganeura::train::build(
        &g,
        SessionConfig {
            mode: Mode::Inference,
            ..crate::support::gpu::config()
        },
    );
    session.set_submission_chunks(chunks);
    chain::seed(&mut session, layers, dim);
    session.set_input("x", &chain::input(rows, dim));

    // Several steps: a boundary hazard can depend on whether a command
    // buffer in the ring is still in flight, which only shows up once the
    // encoder has wrapped around at least once.
    let mut out = vec![0.0f32; rows * dim];
    let mut reference = None;
    for step in 0..steps {
        session.step();
        session.wait();
        session.read_output_by_index(0, &mut out);
        if let Some(ref expected) = reference {
            assert_eq!(&out, expected, "schedule changed repeated graph output");
        }
        if chunks == 1 && step == 0 {
            reference = Some(out.clone());
        }
    }
    reference.unwrap_or(out)
}

#[test]
fn chunked_submission_matches_single_submission() {
    const LAYERS: usize = 8;
    const ROWS: usize = 64;
    const DIM: usize = 64;
    const STEPS: usize = 4;

    let reference = run(1, LAYERS, ROWS, DIM, STEPS);
    assert!(
        reference.iter().all(|v| v.is_finite()),
        "reference output is not finite"
    );
    // A chain of layer norms should not collapse to a constant, or the
    // comparison below would pass for the wrong reason.
    let spread = reference.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b))
        - reference.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    assert!(
        spread > 0.1,
        "reference output is nearly constant: {spread}"
    );

    for chunks in [2, 3, 5, 16] {
        let got = run(chunks, LAYERS, ROWS, DIM, STEPS);
        assert_eq!(
            got.len(),
            reference.len(),
            "chunks={chunks}: output length changed"
        );
        // The same kernels run in the same order on the same data, so this
        // is exact equality, not an approximate comparison. Any difference
        // means a chunk read a buffer before the previous chunk's write
        // landed.
        let worst = got
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert_eq!(
            worst, 0.0,
            "chunks={chunks}: output diverged from the single-submission result by {worst}"
        );
    }
}

/// More chunks than there is work to split, and the degenerate zero, must
/// both behave rather than panic or divide by zero.
#[test]
fn chunk_count_is_clamped_to_something_sensible() {
    let reference = run(1, 2, 8, 16, 1);
    for chunks in [0, 1, 1000] {
        let got = run(chunks, 2, 8, 16, 1);
        let worst = got
            .iter()
            .zip(&reference)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert_eq!(worst, 0.0, "chunks={chunks} changed the result");
    }
}
