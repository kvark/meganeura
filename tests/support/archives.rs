use std::io::Read;

pub fn json(compressed: &[u8]) -> serde_json::Value {
    let mut bytes = Vec::new();
    flate2::read::GzDecoder::new(compressed)
        .read_to_end(&mut bytes)
        .expect("invalid compressed experiment record");
    serde_json::from_slice(&bytes).expect("invalid experiment JSON")
}
