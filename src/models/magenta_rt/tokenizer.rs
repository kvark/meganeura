//! SentencePiece **Unigram** tokenizer (pure Rust) for the MusicCoCa text tower.
//!
//! MusicCoCa tokenizes text with a SentencePiece Unigram model (`spm.model`,
//! vocab 64000) before the text encoder. This implements that tokenizer with no
//! external dependency: a minimal `.model` protobuf parser + the Unigram Viterbi
//! encode + the SentencePiece whitespace normalization.
//!
//! Verified against the reference `sentencepiece` library on a trained model in
//! `tests/spm_tokenizer.rs`. (The real MusicCoCa `spm.model` is not bundled in
//! the public HF release; this loads whatever `.model` is provided.)
//!
//! MusicCoCa's recipe (`magenta_rt/musiccoca.py`): lowercase the prompt, encode,
//! truncate to `max_len-1 = 127`, and prepend `target_sos_id = 1`. The text
//! encoder runs the (unpadded) id sequence — see [`musiccoca_token_ids`].

use std::collections::HashMap;

/// SentencePiece "meta" space marker `▁` (U+2581) — what `escape_whitespaces`
/// turns a space into.
const SPACE_MARKER: char = '\u{2581}';
/// SentencePiece unknown penalty added to `min_score` for an unknown char edge.
const UNK_PENALTY: f32 = 10.0;

/// A parsed SentencePiece Unigram model.
pub struct SpmModel {
    /// `id → (piece, log-score)`.
    pieces: Vec<(String, f32)>,
    /// `piece → (id, score)` for Viterbi lookups.
    lookup: HashMap<String, (u32, f32)>,
    unk_id: u32,
    max_piece_chars: usize,
    min_score: f32,
    add_dummy_prefix: bool,
    remove_extra_whitespaces: bool,
    escape_whitespaces: bool,
}

// --- minimal protobuf wire reader ---

fn read_varint(data: &[u8], pos: &mut usize) -> Result<u64, String> {
    let mut result: u64 = 0;
    let mut shift = 0;
    loop {
        let b = *data.get(*pos).ok_or("varint: unexpected EOF")?;
        *pos += 1;
        result |= ((b & 0x7f) as u64) << shift;
        if b & 0x80 == 0 {
            return Ok(result);
        }
        shift += 7;
        if shift >= 64 {
            return Err("varint: too long".into());
        }
    }
}

/// Read a length-delimited (wire type 2) field's bytes.
fn read_bytes<'a>(data: &'a [u8], pos: &mut usize) -> Result<&'a [u8], String> {
    let len = read_varint(data, pos)? as usize;
    let end = *pos + len;
    let slice = data.get(*pos..end).ok_or("bytes: out of range")?;
    *pos = end;
    Ok(slice)
}

/// Skip a field given its wire type.
fn skip_field(data: &[u8], pos: &mut usize, wire: u8) -> Result<(), String> {
    match wire {
        0 => {
            read_varint(data, pos)?;
        }
        1 => *pos += 8,
        2 => {
            let _ = read_bytes(data, pos)?;
        }
        5 => *pos += 4,
        _ => return Err(format!("unsupported wire type {wire}")),
    }
    Ok(())
}

/// One `SentencePiece` sub-message: field 1 = piece (string), field 2 = score
/// (float32), field 3 = type (enum varint).
fn parse_piece(data: &[u8]) -> Result<(String, f32, i32), String> {
    let mut pos = 0;
    let mut piece = String::new();
    let mut score = 0.0_f32;
    let mut ptype = 1; // NORMAL
    while pos < data.len() {
        let tag = read_varint(data, &mut pos)?;
        let field = tag >> 3;
        let wire = (tag & 7) as u8;
        match (field, wire) {
            (1, 2) => {
                piece = std::str::from_utf8(read_bytes(data, &mut pos)?)
                    .map_err(|_| "piece: invalid utf8")?
                    .to_string();
            }
            (2, 5) => {
                let b = data.get(pos..pos + 4).ok_or("score: EOF")?;
                score = f32::from_le_bytes([b[0], b[1], b[2], b[3]]);
                pos += 4;
            }
            (3, 0) => ptype = read_varint(data, &mut pos)? as i32,
            _ => skip_field(data, &mut pos, wire)?,
        }
    }
    Ok((piece, score, ptype))
}

impl SpmModel {
    /// Parse a SentencePiece `.model` (`ModelProto`) from raw bytes.
    ///
    /// Reads `pieces` (field 1) and the `normalizer_spec` (field 3) whitespace
    /// flags; the NFKC `precompiled_charsmap` is not applied (identity for the
    /// ASCII prompts MusicCoCa sees in practice).
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        let mut pos = 0;
        let mut pieces: Vec<(String, f32)> = Vec::new();
        let mut unk_id = 0u32;
        // SentencePiece normalizer defaults.
        let mut add_dummy_prefix = true;
        let mut remove_extra_whitespaces = true;
        let mut escape_whitespaces = true;

        while pos < data.len() {
            let tag = read_varint(data, &mut pos)?;
            let field = tag >> 3;
            let wire = (tag & 7) as u8;
            match (field, wire) {
                (1, 2) => {
                    let (piece, score, ptype) = parse_piece(read_bytes(data, &mut pos)?)?;
                    if ptype == 2 {
                        // UNKNOWN
                        unk_id = pieces.len() as u32;
                    }
                    pieces.push((piece, score));
                }
                (3, 2) => {
                    // NormalizerSpec: field 3/4/5 = add_dummy_prefix /
                    // remove_extra_whitespaces / escape_whitespaces (bool varints).
                    let ns = read_bytes(data, &mut pos)?;
                    let mut p = 0;
                    while p < ns.len() {
                        let t = read_varint(ns, &mut p)?;
                        let f = t >> 3;
                        let w = (t & 7) as u8;
                        match (f, w) {
                            (3, 0) => add_dummy_prefix = read_varint(ns, &mut p)? != 0,
                            (4, 0) => remove_extra_whitespaces = read_varint(ns, &mut p)? != 0,
                            (5, 0) => escape_whitespaces = read_varint(ns, &mut p)? != 0,
                            _ => skip_field(ns, &mut p, w)?,
                        }
                    }
                }
                _ => skip_field(data, &mut pos, wire)?,
            }
        }

        if pieces.is_empty() {
            return Err("no pieces parsed".into());
        }
        let max_piece_chars = pieces
            .iter()
            .map(|p| p.0.chars().count())
            .max()
            .unwrap_or(1);
        let min_score = pieces.iter().map(|p| p.1).fold(f32::INFINITY, f32::min);
        let lookup = pieces
            .iter()
            .enumerate()
            .map(|(i, p)| (p.0.clone(), (i as u32, p.1)))
            .collect();

        Ok(Self {
            pieces,
            lookup,
            unk_id,
            max_piece_chars,
            min_score,
            add_dummy_prefix,
            remove_extra_whitespaces,
            escape_whitespaces,
        })
    }

    /// Number of vocabulary pieces.
    pub fn vocab_size(&self) -> usize {
        self.pieces.len()
    }

    /// SentencePiece whitespace normalization (no NFKC charsmap): collapse/trim
    /// whitespace, prepend the dummy space, and escape spaces to `▁`.
    fn normalize(&self, text: &str) -> String {
        let mut s = text.to_string();
        if self.remove_extra_whitespaces {
            s = s.split_whitespace().collect::<Vec<_>>().join(" ");
        }
        // Empty (or whitespace-only) input encodes to nothing — no dummy prefix.
        if s.is_empty() {
            return String::new();
        }
        if self.add_dummy_prefix {
            s = format!(" {s}");
        }
        if self.escape_whitespaces {
            s = s.replace(' ', &SPACE_MARKER.to_string());
        }
        s
    }

    /// Encode text to token ids via the Unigram Viterbi best-path. Runs of
    /// uncovered characters collapse to a single `unk` id.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let norm = self.normalize(text);
        let chars: Vec<char> = norm.chars().collect();
        let n = chars.len();
        if n == 0 {
            return Vec::new();
        }
        let unk_score = self.min_score - UNK_PENALTY;

        // Viterbi: best[i] = best score reaching boundary i; back[i] = (prev, id?).
        let mut best = vec![f32::NEG_INFINITY; n + 1];
        let mut back: Vec<(usize, Option<u32>)> = vec![(0, None); n + 1];
        best[0] = 0.0;
        for i in 0..n {
            if best[i] == f32::NEG_INFINITY {
                continue;
            }
            let max_l = self.max_piece_chars.min(n - i);
            let mut sub = String::new();
            for l in 1..=max_l {
                sub.push(chars[i + l - 1]);
                if let Some(&(id, score)) = self.lookup.get(&sub) {
                    let cand = best[i] + score;
                    if cand > best[i + l] {
                        best[i + l] = cand;
                        back[i + l] = (i, Some(id));
                    }
                }
            }
            // Unknown single-char fallback.
            let cand = best[i] + unk_score;
            if cand > best[i + 1] {
                best[i + 1] = cand;
                back[i + 1] = (i, None);
            }
        }

        // Backtrack, then merge consecutive unknown segments into one unk id.
        let mut segs: Vec<Option<u32>> = Vec::new();
        let mut pos = n;
        while pos > 0 {
            let (prev, id) = back[pos];
            segs.push(id);
            pos = prev;
        }
        segs.reverse();
        let mut out = Vec::with_capacity(segs.len());
        let mut prev_unk = false;
        for id in segs {
            match id {
                Some(x) => {
                    out.push(x);
                    prev_unk = false;
                }
                None => {
                    if !prev_unk {
                        out.push(self.unk_id);
                        prev_unk = true;
                    }
                }
            }
        }
        out
    }
}

/// MusicCoCa text → token ids: lowercase, encode, truncate to 127, prepend the
/// start-of-sequence id (1). The MusicCoCa text encoder consumes this (unpadded)
/// id sequence; see `models::magenta_rt::musiccoca`.
pub fn musiccoca_token_ids(spm: &SpmModel, text: &str) -> Vec<u32> {
    const SOS_ID: u32 = 1;
    const MAX_LEN: usize = 128;
    let mut ids = spm.encode(&text.to_lowercase());
    ids.truncate(MAX_LEN - 1);
    let mut out = Vec::with_capacity(ids.len() + 1);
    out.push(SOS_ID);
    out.extend(ids);
    out
}
