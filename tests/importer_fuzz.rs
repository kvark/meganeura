//! Importer panic regressions. Run with `PROPTEST_CASES=10000` to expand the
//! corpus; use `--test-threads=1` when recording new failure seeds.

use proptest::{
    prelude::{ProptestConfig, any, prop},
    proptest,
};

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 256,
        failure_persistence: Some(Box::new(proptest::test_runner::FileFailurePersistence::WithSource("proptest-regressions"))),
        ..ProptestConfig::default()
    })]

    #[test]
    fn onnx_arbitrary_bytes_never_panic(bytes in prop::collection::vec(any::<u8>(), 0..512)) {
        let _ = meganeura::load::onnx::load_onnx_bytes(&bytes, None);
    }

    #[test]
    fn onnx_well_framed_garbage_never_panic(
        len_field in any::<u64>(),
        tail in prop::collection::vec(any::<u8>(), 0..64),
    ) {
        let mut bytes = Vec::new();
        bytes.push(0x12);
        let mut v = len_field;
        while v >= 0x80 {
            bytes.push((v as u8) | 0x80);
            v >>= 7;
        }
        bytes.push(v as u8);
        bytes.extend_from_slice(&tail);
        let _ = meganeura::load::onnx::load_onnx_bytes(&bytes, None);
    }

    #[test]
    fn onnx_saturating_length_never_panic(filler in prop::collection::vec(any::<u8>(), 0..24)) {
        let mut bytes = filler;
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

    #[test]
    fn nnef_arbitrary_text_never_panic(text in proptest::string::string_regex(".{0,400}").unwrap()) {
        let dir = std::env::temp_dir().join(format!("meganeura-nnef-fuzz-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("graph.nnef");
        std::fs::write(&path, &text).unwrap();
        let _ = meganeura::load::nnef::load_nnef(&path);
        let _ = std::fs::remove_file(&path);
    }
}
