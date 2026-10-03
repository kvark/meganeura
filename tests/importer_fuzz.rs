//! The importers take untrusted files, so a malformed one must produce a
//! `Result::Err` and never a panic.
//!
//! `tests/oracle/fuzz.rs` covers random *graphs* through the compiler, which is
//! a different question: those are well-formed by construction. This covers
//! random *bytes* through the readers, which is the question the importers
//! actually face.
//!
//! Every case is arbitrary bytes rather than a well-formed document with one
//! field perturbed, because the failures that matter here are structural: a
//! length that runs past the end of the buffer, a varint that overflows, a
//! shape dimension that does not fit. Those are exactly what a hand-built
//! corpus tends to miss.
//!
//! Run the default cases with `cargo test --test regression importer_fuzz`, or
//! more of them with `PROPTEST_CASES=10000`.
//!
//! Regressions are recorded in `tests/importer_fuzz.proptest-regressions`, one
//! file for the whole module. proptest appends to it per failing test, so if
//! two of these fail at once under parallel test threads one can overwrite the
//! other's entry; run `--test-threads=1` when growing the corpus, which is what
//! the suite does anyway.

use proptest::prelude::*;

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 256,
        // A panic in the reader is the thing being looked for, so a failure
        // has to be reported rather than swallowed.
        failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::WithSource("proptest-regressions"))),
        ..ProptestConfig::default()
    })]

    /// Any bytes at all. Most will fail to parse, which is the correct
    /// outcome; what must never happen is a panic escaping the `Result`.
    #[test]
    fn onnx_arbitrary_bytes_never_panic(bytes in prop::collection::vec(any::<u8>(), 0..512)) {
        let _ = meganeura::load::onnx::load_onnx_bytes(&bytes, None);
    }

    /// Bytes that begin with a plausible protobuf frame, so the reader gets
    /// past the magic-number check and into the field walk — where the
    /// length-delimited slicing lives. Random bytes rarely get that far, so
    /// without this the interesting code paths would go untested.
    #[test]
    fn onnx_well_framed_garbage_never_panic(
        len_field in any::<u64>(),
        tail in prop::collection::vec(any::<u8>(), 0..64),
    ) {
        let mut bytes = Vec::new();
        // A length-delimited field (wire type 2) whose declared length is
        // attacker-chosen and need not match the payload that follows.
        bytes.push(0x12);
        // varint-encode the length.
        let mut v = len_field;
        while v >= 0x80 {
            bytes.push((v as u8) | 0x80);
            v >>= 7;
        }
        bytes.push(v as u8);
        bytes.extend_from_slice(&tail);
        let _ = meganeura::load::onnx::load_onnx_bytes(&bytes, None);
    }

    /// A length that saturates `usize`, which is the case `checked_add` and
    /// `try_from` exist for: `p + len` would wrap and produce an in-range
    /// slice of the wrong bytes.
    ///
    /// This is the input that panicked. The leading filler moves where the
    /// slice would start, so the wrap is exercised at several offsets rather
    /// than only at zero — `len` alone does not change the payload, so without
    /// the filler every case would be the same bytes.
    #[test]
    fn onnx_saturating_length_never_panic(filler in prop::collection::vec(any::<u8>(), 0..24)) {
        let mut bytes = filler;
        // Wire type 2 (length-delimited), then varint for u64::MAX, which is
        // not representable as usize on a 32-bit target and overflows `pos +
        // len` on a 64-bit one.
        bytes.push(0x12);
        let mut v = u64::MAX;
        while v >= 0x80 {
            bytes.push((v as u8) | 0x80);
            v >>= 7;
        }
        bytes.push(v as u8);
        bytes.extend_from_slice(&[0u8; 16]);
        let _ = meganeura::load::onnx::load_onnx_bytes(&bytes, None);
    }

    /// Text-shaped input for the NNEF graph reader, which parses a text format
    /// rather than protobuf. Arbitrary characters, not just printable ones,
    /// because the reader indexes into the string.
    #[test]
    fn nnef_arbitrary_text_never_panic(text in proptest::string::string_regex(".{0,400}").unwrap()) {
        let dir = std::env::temp_dir().join("meganeura-nnef-fuzz");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("graph.nnef");
        std::fs::write(&path, &text).unwrap();
        let _ = meganeura::load::nnef::load_nnef(&path);
        let _ = std::fs::remove_file(&path);
    }
}
