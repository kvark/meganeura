//! Verifies the pure-Rust SentencePiece Unigram tokenizer
//! (`tokenizer::SpmModel`) against the reference `sentencepiece` library.
//!
//! `tests/data/spm_tiny.model` is a small Unigram model trained with the
//! `sentencepiece` Python package; the expected id lists below are that
//! package's `EncodeAsIds` output on the same (lowercase) strings. Matching them
//! exercises the `.model` protobuf parse, the whitespace normalization, the
//! Unigram Viterbi best-path, and the unknown-run merging.

use meganeura::models::magenta_rt::tokenizer::{musiccoca_token_ids, SpmModel};

const MODEL: &[u8] = include_bytes!("data/spm_tiny.model");

#[test]
fn unigram_encode_matches_sentencepiece() {
    let spm = SpmModel::from_bytes(MODEL).expect("parse spm model");
    // Reference: `sp.EncodeAsIds(text)` from the sentencepiece package.
    let cases: &[(&str, &[u32])] = &[
        (
            "upbeat electronic dance",
            &[3, 6, 15, 32, 3, 64, 4, 16, 43, 21, 12, 23, 7, 34],
        ),
        (
            "calm acoustic guitar",
            &[3, 49, 13, 54, 55, 6, 41, 12, 19, 6, 36, 7, 9],
        ),
        ("jazz piano", &[3, 65, 7, 30, 30, 3, 73, 69, 5]),
        (
            "lofi hip hop beats",
            &[3, 17, 28, 8, 24, 8, 15, 24, 5, 15, 3, 32, 11],
        ),
        (
            "funky bass groove",
            &[3, 28, 6, 21, 29, 10, 31, 19, 40, 5, 63, 4],
        ),
        // 'x' and 'q' are out-of-vocab → trailing unknown run collapses to id 0.
        (
            "epic strings xyzqq",
            &[47, 15, 12, 3, 42, 39, 11, 3, 66, 10, 30, 0],
        ),
        // Extra/leading/trailing whitespace collapses + trims (normalizer).
        ("jazz   piano", &[3, 65, 7, 30, 30, 3, 73, 69, 5]),
        ("  upbeat  ", &[3, 6, 15, 32]),
        // 'q' unknown (run → single 0), 'x' is a single-char piece (66).
        ("qqqxxx", &[3, 0, 66, 66, 66]),
        // '▁a' is a single merged piece (no separate dummy-prefix token).
        ("a", &[54]),
        ("", &[]),
    ];
    for (text, want) in cases {
        let got = spm.encode(text);
        assert_eq!(&got, want, "encode({text:?})");
    }
}

#[test]
fn lowercasing_is_applied_by_the_musiccoca_wrapper() {
    let spm = SpmModel::from_bytes(MODEL).expect("parse spm model");
    // The wrapper lowercases, so mixed/upper case must match the lowercase encode
    // (plus the prepended SOS id = 1).
    let lower = spm.encode("funky bass groove");
    let mut want = vec![1u32];
    want.extend_from_slice(&lower);
    assert_eq!(musiccoca_token_ids(&spm, "FUNKY Bass Groove"), want);
}

#[test]
fn musiccoca_prepends_sos_and_truncates() {
    let spm = SpmModel::from_bytes(MODEL).expect("parse spm model");
    let ids = musiccoca_token_ids(&spm, "jazz piano");
    assert_eq!(ids[0], 1, "first id must be the SOS id");
    assert!(ids.len() <= 128);
    assert_eq!(&ids[1..], spm.encode("jazz piano").as_slice());
}
