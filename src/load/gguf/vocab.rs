//! The tokenizer the file carries.
//!
//! GGUF embeds the whole vocabulary — every token, the merge list or the
//! piece scores, and the special-token ids — under `tokenizer.ggml.*`. So
//! a `.gguf` is self-sufficient: turning a prompt into ids needs nothing
//! from the Hub and no `tokenizer.json` alongside.
//!
//! That is also why this is written out rather than delegated. Meganeura
//! keeps `tokenizers` as a dev-dependency precisely so the inference runtime
//! does not drag `onig`'s C library through every cross-compile; reading a
//! vocabulary the file already contains should not undo that.
//!
//! # The two models
//!
//! `tokenizer.ggml.model` picks between them, and they are genuinely
//! different algorithms rather than two spellings of one:
//!
//! - **BPE** (`gpt2`), used by the Qwen, Phi and Llama-3 families. Text is
//!   split by a pre-tokenizer, each piece is recoded into GPT-2's printable
//!   byte alphabet, and adjacent symbols are merged in the order the merge
//!   list ranks them. Every byte is representable, so there is no unknown
//!   token.
//! - **SentencePiece** (`llama`), used by Llama-2, Mistral and Gemma. This
//!   is SentencePiece's *BPE* mode, not its unigram mode — the two are
//!   different algorithms. Spaces become `▁`, and adjacent symbols merge by
//!   *score*, highest first, rather than by a merge list; ties go to the
//!   leftmost pair. Characters outside the vocabulary fall back to one
//!   `<0x..>` token per UTF-8 byte.

use std::collections::HashMap;

use super::{GgufError, GgufModel, GgufValue};

/// Which tokenization algorithm the file declares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TokenizerKind {
    /// Byte-level byte-pair encoding, merged in merge-list order.
    Bpe,
    /// SentencePiece's BPE mode, merged by score. Not its unigram mode —
    /// the two are different algorithms and GGUF names both `llama`.
    SentencePiece,
}

/// GGML's token classes. Only the ones that change behaviour are
/// distinguished here.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TokenKind {
    /// An ordinary piece of text.
    Normal,
    /// A marker such as `<s>` or `<|im_start|>`. Never produced by merging
    /// text, and suppressed when decoding for display.
    Control,
    /// A single raw byte, `<0x41>`. SentencePiece's fallback.
    Byte,
    /// Added after training — a chat-template marker, usually. Matched
    /// literally in the input before anything else runs.
    UserDefined,
}

/// The vocabulary, and everything needed to apply it.
#[derive(Clone, Debug)]
pub struct Vocab {
    /// How text is split before merges apply. Only meaningful for
    /// [`TokenizerKind::Bpe`]; SentencePiece has no separate rule.
    pre: PreTokenizer,
    kind: TokenizerKind,
    tokens: Vec<String>,
    ids: HashMap<String, u32>,
    scores: Vec<f32>,
    kinds: Vec<TokenKind>,
    /// Merge rank by pair; lower merges first. BPE only.
    merges: HashMap<(String, String), u32>,
    /// The id of each raw byte, for SentencePiece's `<0x..>` fallback.
    byte_ids: Vec<Option<u32>>,
    bos: Option<u32>,
    eos: Option<u32>,
    unknown: Option<u32>,
    padding: Option<u32>,
    /// Extra ids that end generation — `<|im_end|>` and friends, which sit
    /// alongside the declared EOS rather than replacing it.
    eog: Vec<u32>,
    add_bos: bool,
    add_eos: bool,
    add_space_prefix: bool,
}

impl Vocab {
    /// Read the vocabulary out of a parsed file.
    ///
    /// Returns [`GgufError::MissingKey`] when the file carries no
    /// vocabulary at all, which a weights-only GGUF legitimately might.
    pub fn from_gguf(model: &GgufModel) -> Result<Self, GgufError> {
        let pre = PreTokenizer::from_name(meta_str(model, "tokenizer.ggml.pre"))?;
        let kind = match meta_str(model, "tokenizer.ggml.model") {
            Some("gpt2") => TokenizerKind::Bpe,
            Some("llama") => TokenizerKind::SentencePiece,
            Some(other) => {
                return Err(GgufError::BadMetadata(format!(
                    "tokenizer model `{other}` is not implemented; this loader \
                     handles `gpt2` (BPE) and `llama` (SentencePiece)"
                )));
            }
            None => return Err(GgufError::MissingKey("tokenizer.ggml.model".to_string())),
        };

        let raw = model
            .metadata
            .get("tokenizer.ggml.tokens")
            .and_then(GgufValue::as_array)
            .ok_or_else(|| GgufError::MissingKey("tokenizer.ggml.tokens".to_string()))?;
        let tokens: Vec<String> = raw
            .iter()
            .map(|v| {
                v.as_str().map(str::to_string).ok_or_else(|| {
                    GgufError::BadMetadata("tokenizer.ggml.tokens holds a non-string".to_string())
                })
            })
            .collect::<Result<_, _>>()?;
        if tokens.is_empty() {
            return Err(GgufError::BadMetadata(
                "tokenizer.ggml.tokens is empty".to_string(),
            ));
        }

        let scores = float_array(model, "tokenizer.ggml.scores", tokens.len());
        let kinds = token_kinds(model, &tokens);

        // Later duplicates lose: GGML resolves a repeated token to its
        // first id, and so must a round trip through this map.
        let mut ids = HashMap::with_capacity(tokens.len());
        for (id, token) in tokens.iter().enumerate() {
            ids.entry(token.clone()).or_insert(id as u32);
        }

        let merges = match kind {
            TokenizerKind::Bpe => merge_ranks(model)?,
            TokenizerKind::SentencePiece => HashMap::new(),
        };

        let byte_ids = byte_ids(kind, &tokens, &ids, &kinds);

        let bos = meta_id(model, "tokenizer.ggml.bos_token_id", tokens.len());
        let eos = meta_id(model, "tokenizer.ggml.eos_token_id", tokens.len());
        let unknown = meta_id(model, "tokenizer.ggml.unknown_token_id", tokens.len());
        let padding = meta_id(model, "tokenizer.ggml.padding_token_id", tokens.len());
        let eog = end_of_generation(model, &tokens, eos);

        Ok(Self {
            pre,
            kind,
            // SentencePiece prepends a space so that a leading word is
            // tokenized the same as a word in the middle of a sentence.
            add_space_prefix: meta_bool(model, "tokenizer.ggml.add_space_prefix")
                .unwrap_or(kind == TokenizerKind::SentencePiece),
            add_bos: meta_bool(model, "tokenizer.ggml.add_bos_token")
                .unwrap_or(kind == TokenizerKind::SentencePiece),
            add_eos: meta_bool(model, "tokenizer.ggml.add_eos_token").unwrap_or(false),
            tokens,
            ids,
            scores,
            kinds,
            merges,
            byte_ids,
            bos,
            eos,
            unknown,
            padding,
            eog,
        })
    }

    /// Which algorithm this vocabulary uses.
    pub fn kind(&self) -> TokenizerKind {
        self.kind
    }

    /// How many tokens the vocabulary holds.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Whether the vocabulary is empty. Never true for a vocabulary that
    /// parsed, but clippy asks and a caller may reasonably check.
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// The beginning-of-sequence id, if the file names one.
    pub fn bos_id(&self) -> Option<u32> {
        self.bos
    }

    /// The end-of-sequence id, if the file names one.
    pub fn eos_id(&self) -> Option<u32> {
        self.eos
    }

    /// The unknown-token id, if the file names one.
    pub fn unknown_id(&self) -> Option<u32> {
        self.unknown
    }

    /// The padding id, if the file names one.
    pub fn padding_id(&self) -> Option<u32> {
        self.padding
    }

    /// Whether `token` should stop generation.
    ///
    /// This is broader than [`Self::eos_id`]: instruction-tuned models end
    /// turns with `<|im_end|>`, `<|eot_id|>` or similar, and a generator
    /// that only watched the declared EOS would run to its token limit
    /// every time.
    pub fn is_end_of_generation(&self, token: u32) -> bool {
        self.eog.contains(&token)
    }

    /// The literal text of one token, without unescaping byte fallbacks or
    /// hiding control markers. For diagnostics; [`Self::decode`] is what
    /// renders output.
    pub fn token_text(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(String::as_str)
    }

    /// Look an exact token up by its literal text.
    pub fn token_id(&self, text: &str) -> Option<u32> {
        self.ids.get(text).copied()
    }

    /// Tokenize `text`, adding whichever of BOS and EOS the file asks for.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        if self.add_bos
            && let Some(bos) = self.bos
        {
            out.push(bos);
        }
        self.encode_into(text, &mut out);
        if self.add_eos
            && let Some(eos) = self.eos
        {
            out.push(eos);
        }
        out
    }

    /// Tokenize `text` and nothing else — no BOS, no EOS.
    ///
    /// What a caller wants when continuing a sequence, or when a chat
    /// template has already placed the markers.
    pub fn encode_plain(&self, text: &str) -> Vec<u32> {
        let mut out = Vec::new();
        self.encode_into(text, &mut out);
        out
    }

    /// Render tokens back to text, dropping control markers.
    pub fn decode(&self, tokens: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in tokens {
            self.decode_one_into(id, &mut bytes);
        }
        // A token boundary can split a multi-byte character, so an
        // incremental caller may hand over a prefix that is not yet valid
        // UTF-8. Replacing is better than refusing to show anything.
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Render one token, for streaming output.
    ///
    /// A single token may be half a character, in which case this yields a
    /// replacement character; [`Self::decode`] over the whole run is exact.
    pub fn decode_token(&self, token: u32) -> String {
        let mut bytes = Vec::new();
        self.decode_one_into(token, &mut bytes);
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Append one token's bytes, which is where the two models differ.
    fn decode_one_into(&self, id: u32, out: &mut Vec<u8>) {
        let Some(text) = self.tokens.get(id as usize) else {
            return;
        };
        match self.kinds.get(id as usize).copied() {
            // Markers are structure, not text. A chat template puts them
            // in; rendering them back would show the scaffolding.
            Some(TokenKind::Control) => {}
            Some(TokenKind::Byte) => {
                if let Some(b) = parse_byte_token(text) {
                    out.push(b);
                } else {
                    out.extend_from_slice(text.as_bytes());
                }
            }
            _ => match self.kind {
                TokenizerKind::Bpe => {
                    // Every char is a stand-in for one byte.
                    for ch in text.chars() {
                        match unicode_to_byte(ch) {
                            Some(b) => out.push(b),
                            // A token added after training may hold real
                            // text rather than alphabet characters.
                            None => {
                                let mut buf = [0u8; 4];
                                out.extend_from_slice(ch.encode_utf8(&mut buf).as_bytes());
                            }
                        }
                    }
                }
                TokenizerKind::SentencePiece => {
                    out.extend_from_slice(text.replace(SPM_SPACE, " ").as_bytes());
                }
            },
        }
    }

    fn encode_into(&self, text: &str, out: &mut Vec<u32>) {
        if text.is_empty() {
            return;
        }
        // Tokens added after training are matched literally and never
        // merged through, so they split the text before anything else.
        for piece in self.split_on_added_tokens(text) {
            match piece {
                Piece::Token(id) => out.push(id),
                Piece::Text(chunk) => match self.kind {
                    TokenizerKind::Bpe => self.encode_bpe(chunk, out),
                    TokenizerKind::SentencePiece => self.encode_spm(chunk, out),
                },
            }
        }
    }

    /// Split around any user-defined token appearing literally in `text`.
    ///
    /// Longest match wins at each position, so `<|im_start|>` is not
    /// mistaken for a shorter marker sharing its prefix.
    fn split_on_added_tokens<'a>(&'a self, text: &'a str) -> Vec<Piece<'a>> {
        let added: Vec<(&str, u32)> = self
            .kinds
            .iter()
            .enumerate()
            .filter(|&(_, k)| matches!(k, TokenKind::UserDefined | TokenKind::Control))
            .filter_map(|(id, _)| {
                let text = self.tokens.get(id)?.as_str();
                // A one-character marker would match ordinary text
                // constantly; real markers are bracketed and longer.
                (text.len() > 1).then_some((text, id as u32))
            })
            .collect();
        if added.is_empty() {
            return vec![Piece::Text(text)];
        }

        let mut pieces = Vec::new();
        let mut start = 0;
        let mut at = 0;
        while at < text.len() {
            if !text.is_char_boundary(at) {
                at += 1;
                continue;
            }
            let rest = &text[at..];
            let hit = added
                .iter()
                .filter(|&&(marker, _)| rest.starts_with(marker))
                .max_by_key(|&&(marker, _)| marker.len());
            if let Some(&(marker, id)) = hit {
                if start < at {
                    pieces.push(Piece::Text(&text[start..at]));
                }
                pieces.push(Piece::Token(id));
                at += marker.len();
                start = at;
            } else {
                at += 1;
            }
        }
        if start < text.len() {
            pieces.push(Piece::Text(&text[start..]));
        }
        pieces
    }

    /// Byte-level BPE: pre-tokenize, recode, then merge by rank.
    fn encode_bpe(&self, text: &str, out: &mut Vec<u32>) {
        for word in self.pre.split(text) {
            // Each byte becomes one printable character, so the merge list
            // — which is written in that alphabet — applies directly.
            let mut symbols: Vec<String> = word
                .bytes()
                .map(|b| byte_to_unicode(b).to_string())
                .collect();
            if symbols.is_empty() {
                continue;
            }

            loop {
                // The lowest-ranked adjacent pair merges first; ties go to
                // the leftmost, which is what the reference does.
                let best = symbols
                    .windows(2)
                    .enumerate()
                    .filter_map(|(i, pair)| {
                        let key = (pair[0].clone(), pair[1].clone());
                        self.merges.get(&key).map(|&rank| (rank, i))
                    })
                    .min();
                let Some((_, at)) = best else { break };
                let merged = format!("{}{}", symbols[at], symbols[at + 1]);
                symbols.splice(at..at + 2, [merged]);
                if symbols.len() == 1 {
                    break;
                }
            }

            for symbol in symbols {
                match self.ids.get(&symbol) {
                    Some(&id) => out.push(id),
                    // A symbol the merges produced but the vocabulary
                    // lacks: fall back to its characters, which are single
                    // bytes and so always present in a byte-level vocab.
                    None => {
                        for ch in symbol.chars() {
                            if let Some(&id) = self.ids.get(&ch.to_string()) {
                                out.push(id);
                            } else if let Some(unk) = self.unknown {
                                out.push(unk);
                            }
                        }
                    }
                }
            }
        }
    }

    /// SentencePiece: normalize, then merge the best-scoring adjacent pair
    /// until none is left in the vocabulary.
    fn encode_spm(&self, text: &str, out: &mut Vec<u32>) {
        let mut normalized = text.replace(' ', SPM_SPACE);
        if self.add_space_prefix && !normalized.starts_with(SPM_SPACE) {
            // So that a leading word tokenizes as it would mid-sentence.
            normalized.insert_str(0, SPM_SPACE);
        }
        let mut symbols: Vec<String> = normalized.chars().map(|c| c.to_string()).collect();
        if symbols.is_empty() {
            return;
        }

        loop {
            // Highest score wins here, where BPE takes the lowest rank —
            // the two models order their merges in opposite directions.
            //
            // Ties go to the *leftmost* pair. GGML's `llm_bigram_spm`
            // comparator breaks an equal score by the lower left index,
            // and `Iterator::max_by` keeps the last maximum rather than
            // the first, so the tie-break has to be written out. With
            // pieces `ab` and `bc` scored equally, `abc` is `[ab, c]`
            // here and `[a, bc]` without it.
            let best = symbols
                .windows(2)
                .enumerate()
                .filter_map(|(i, pair)| {
                    let joined = format!("{}{}", pair[0], pair[1]);
                    let id = *self.ids.get(&joined)?;
                    let score = self.scores.get(id as usize).copied().unwrap_or(0.0);
                    Some((score, i, joined))
                })
                .max_by(|a, b| {
                    a.0.partial_cmp(&b.0)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        // Reversed, so that on an equal score the *lower*
                        // index compares greater and `max_by` keeps it.
                        .then(b.1.cmp(&a.1))
                });
            let Some((_, at, joined)) = best else { break };
            symbols.splice(at..at + 2, [joined]);
            if symbols.len() == 1 {
                break;
            }
        }

        for symbol in symbols {
            match self.ids.get(&symbol) {
                Some(&id) => out.push(id),
                // Outside the vocabulary: emit the UTF-8 bytes as byte
                // tokens, which is what makes SentencePiece lossless.
                None => self.push_byte_fallback(&symbol, out),
            }
        }
    }

    fn push_byte_fallback(&self, symbol: &str, out: &mut Vec<u32>) {
        for &b in symbol.as_bytes() {
            if let Some(Some(id)) = self.byte_ids.get(b as usize).copied() {
                out.push(id);
            } else if let Some(unk) = self.unknown {
                out.push(unk);
            }
        }
    }
}

/// A run of text, or a token matched literally within it.
enum Piece<'a> {
    Text(&'a str),
    Token(u32),
}

/// SentencePiece's visible space.
const SPM_SPACE: &str = "\u{2581}";

// ---------------------------------------------------------------------------
// Pre-tokenization
// ---------------------------------------------------------------------------

/// Which pre-tokenizer a byte-level BPE file declares.
///
/// `tokenizer.ggml.model = gpt2` names the *merge algorithm*, not the rule
/// that decides where merges may apply. Llama 3, Qwen and SmolLM all write
/// `gpt2` there and split text differently, and merges cannot cross a
/// pre-token boundary — so reading the model alone and assuming GPT-2's
/// rule silently retokenizes those files. `1234` is one pre-token under
/// GPT-2, `123|4` under Llama 3 and `1|2|3|4` under Qwen2.
///
/// Each variant here reproduces one of llama.cpp's `regex_exprs` entries
/// exactly. An identifier that maps to any other rule is refused rather
/// than approximated, because the result would be fluent and wrong.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreTokenizer {
    /// The classic GPT-2 rule:
    /// `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)`
    Gpt2,
    /// Llama 3's: case-insensitive contractions, one free character before
    /// a letter run, digits in groups of up to three, and newline-aware
    /// punctuation and whitespace.
    Llama3,
    /// Llama 3's rule with digits taken one at a time.
    Qwen2,
    /// Digits one at a time, then GPT-2's rule over what is left.
    SmolLm,
}

impl PreTokenizer {
    /// Resolve `tokenizer.ggml.pre`.
    ///
    /// Absent, the file predates the key and GPT-2's rule is the only
    /// reading available. Each accepted name is one llama.cpp maps to the
    /// same `regex_exprs` as the variant it is listed under; anything else
    /// — including `default`, which is its own four-pattern cascade — is
    /// refused by name.
    pub fn from_name(name: Option<&str>) -> Result<Self, GgufError> {
        let Some(name) = name else {
            return Ok(Self::Gpt2);
        };
        Ok(match name {
            "gpt-2" | "phi-2" | "jina-es" | "jina-de" | "jina-v2-es" | "jina-v2-de"
            | "gigachat" | "a.x-4.0" | "mellum" | "modern-bert" | "mpt" | "olmo" | "jais"
            | "trillion" | "granite-docling" => Self::Gpt2,
            "llama3" | "llama-v3" | "llama-bpe" | "falcon3" | "falcon-h1" | "pixtral"
            | "midm-2.0" | "lfm2" | "jina-v5-nano" => Self::Llama3,
            "qwen2" | "deepseek-r1-qwen" | "kormo" | "f2llmv2" | "stablelm2" | "hunyuan" => {
                Self::Qwen2
            }
            "smollm" | "codeshell" | "exaone" | "minerva-7b" => Self::SmolLm,
            other => {
                return Err(GgufError::UnsupportedArchitecture(format!(
                    "tokenizer.ggml.pre = `{other}` selects a pre-tokenizer this \
                     loader does not implement; merges cannot cross the boundaries \
                     it would place, so the wrong rule retokenizes the text"
                )));
            }
        })
    }

    /// Split `text` into pre-tokens. Merges apply within these and never
    /// across them.
    fn split<'a>(self, text: &'a str) -> Vec<&'a str> {
        match self {
            Self::Gpt2 => scan_gpt2(text),
            Self::Llama3 => scan_llama3(text, 3),
            Self::Qwen2 => scan_llama3(text, 1),
            // llama.cpp lists `\p{N}` before GPT-2's pattern, and each
            // regex splits the pieces the one before it produced. Severing
            // the digits first is why ` 123` becomes ` `, `1`, `2`, `3`
            // here where GPT-2 alone keeps ` 123` whole.
            Self::SmolLm => split_digits(text)
                .into_iter()
                .flat_map(|piece| {
                    if piece.chars().next().is_some_and(is_digit) {
                        vec![piece]
                    } else {
                        scan_gpt2(piece)
                    }
                })
                .collect(),
        }
    }
}

/// `\p{L}` as std spells it.
///
/// `char::is_alphabetic` is Unicode's Alphabetic property, which is
/// `\p{L}` plus `Nl` and the characters carrying `Other_Alphabetic` —
/// mostly combining marks in Indic and Hebrew scripts. Text in those
/// scripts can therefore split differently here than under a true
/// `\p{L}`; for the Latin, Greek, Cyrillic and CJK text these vocabularies
/// are overwhelmingly built from, the two agree.
fn is_letter(c: char) -> bool {
    c.is_alphabetic()
}

/// `\p{N}` as std spells it — `is_numeric` is `Nd | Nl | No`, which is
/// exactly `\p{N}`.
fn is_digit(c: char) -> bool {
    c.is_numeric()
}

/// Each digit as its own piece, with the spans between them intact.
fn split_digits(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    for (at, c) in text.char_indices() {
        if is_digit(c) {
            if at > start {
                out.push(&text[start..at]);
            }
            out.push(&text[at..at + c.len_utf8()]);
            start = at + c.len_utf8();
        }
    }
    if start < text.len() {
        out.push(&text[start..]);
    }
    out
}

/// The byte length of a contraction at the start of `rest`.
///
/// `case_insensitive` selects between GPT-2's literal lowercase list and
/// the `'[sS]|'[tT]|…` form Llama 3 and Qwen2 use.
fn contraction_len(rest: &str, case_insensitive: bool) -> Option<usize> {
    for suffix in ["'re", "'ve", "'ll", "'s", "'t", "'m", "'d"] {
        let matches = if case_insensitive {
            rest.len() >= suffix.len()
                && rest.as_bytes()[..suffix.len()].eq_ignore_ascii_case(suffix.as_bytes())
        } else {
            rest.starts_with(suffix)
        };
        if matches {
            return Some(suffix.len());
        }
    }
    None
}

/// Split text the way GPT-2's pre-tokenizer regex does.
///
/// The pattern is
/// `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)`,
/// hand-written here rather than run through a regex engine: it is a small
/// deterministic scan, and pulling in a regex crate — or `onig`'s C library,
/// which is exactly what keeping `tokenizers` out of `[dependencies]`
/// avoids — to read a vocabulary the file already contains would be a poor
/// trade.
fn scan_gpt2(text: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let end = text.len();
    let at_byte = |i: usize| chars.get(i).map_or(end, |&(b, _)| b);
    let ch = |i: usize| chars.get(i).map(|&(_, c)| c);

    let mut i = 0;
    while i < chars.len() {
        let start = at_byte(i);

        // Contractions, which the pattern lists first and so match first.
        if ch(i) == Some('\'')
            && let Some(len) = contraction_len(&text[start..], false)
        {
            out.push(&text[start..start + len]);
            i += text[start..start + len].chars().count();
            continue;
        }

        // ` ?\p{L}+`, ` ?\p{N}+` and ` ?[^\s\p{L}\p{N}]+`: an optional
        // single leading space joins the run that follows it.
        let space = ch(i) == Some(' ');
        let head = if space { i + 1 } else { i };
        if let Some(c) = ch(head)
            && !c.is_whitespace()
        {
            let class = classify(c);
            let mut j = head;
            while let Some(c) = ch(j) {
                if c.is_whitespace() || classify(c) != class {
                    break;
                }
                j += 1;
            }
            out.push(&text[start..at_byte(j)]);
            i = j;
            continue;
        }

        // `\s+(?!\S)|\s+`: a whitespace run, except that its final
        // character is left for the next piece when text follows — that is
        // what the negative lookahead buys, and why " a" is one token.
        let mut j = i;
        while ch(j).is_some_and(char::is_whitespace) {
            j += 1;
        }
        if j > i {
            let followed_by_text = ch(j).is_some();
            let stop = if followed_by_text && j - i > 1 {
                j - 1
            } else {
                j
            };
            if stop > i {
                out.push(&text[start..at_byte(stop)]);
            }
            i = if stop > i { stop } else { j };
            continue;
        }

        // Nothing matched, which the pattern makes impossible; advance so
        // a surprising input cannot spin.
        i += 1;
    }
    out
}

/// Split text the way Llama 3's pre-tokenizer regex does, with digits
/// taken `max_digits` at a time.
///
/// The pattern is
/// `(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+`,
/// and Qwen2's differs from it only in `\p{N}` where this has `\p{N}{1,3}`.
/// The alternatives are tried in order, first match winning, exactly as
/// the engine would.
///
/// Three things separate it from [`scan_gpt2`]: contractions match in
/// either case, a letter run may be preceded by any one character that is
/// not a newline, letter or digit rather than only a space, and a digit
/// run is capped and never takes a leading space.
fn scan_llama3(text: &str, max_digits: usize) -> Vec<&str> {
    let mut out = Vec::new();
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let end = text.len();
    let at_byte = |i: usize| chars.get(i).map_or(end, |&(b, _)| b);
    let ch = |i: usize| chars.get(i).map(|&(_, c)| c);
    let is_newline = |c: char| c == '\r' || c == '\n';

    let mut i = 0;
    while i < chars.len() {
        let start = at_byte(i);

        // `(?:'[sS]|'[tT]|…)`
        if ch(i) == Some('\'')
            && let Some(len) = contraction_len(&text[start..], true)
        {
            out.push(&text[start..start + len]);
            i += text[start..start + len].chars().count();
            continue;
        }

        // `[^\r\n\p{L}\p{N}]?\p{L}+` — one optional free character, then
        // at least one letter. The optional character is only consumed if
        // letters actually follow it.
        let prefixed = ch(i).is_some_and(|c| !is_newline(c) && !is_letter(c) && !is_digit(c))
            && ch(i + 1).is_some_and(is_letter);
        if prefixed || ch(i).is_some_and(is_letter) {
            let mut j = if prefixed { i + 1 } else { i };
            while ch(j).is_some_and(is_letter) {
                j += 1;
            }
            out.push(&text[start..at_byte(j)]);
            i = j;
            continue;
        }

        // `\p{N}{1,max}` — no leading space, and capped.
        if ch(i).is_some_and(is_digit) {
            let mut j = i;
            while j < i + max_digits && ch(j).is_some_and(is_digit) {
                j += 1;
            }
            out.push(&text[start..at_byte(j)]);
            i = j;
            continue;
        }

        // ` ?[^\s\p{L}\p{N}]+[\r\n]*`
        let space = ch(i) == Some(' ');
        let head = if space { i + 1 } else { i };
        if ch(head).is_some_and(|c| !c.is_whitespace() && !is_letter(c) && !is_digit(c)) {
            let mut j = head;
            while ch(j).is_some_and(|c| !c.is_whitespace() && !is_letter(c) && !is_digit(c)) {
                j += 1;
            }
            while ch(j).is_some_and(is_newline) {
                j += 1;
            }
            out.push(&text[start..at_byte(j)]);
            i = j;
            continue;
        }

        // The whitespace alternatives. Take the run once and decide which
        // of the three it is.
        let mut j = i;
        while ch(j).is_some_and(char::is_whitespace) {
            j += 1;
        }
        if j > i {
            // `\s*[\r\n]+` is greedy and then backtracks, so it matches
            // through the *last* newline in the run.
            let last_newline = (i..j).rev().find(|&k| ch(k).is_some_and(is_newline));
            let stop = match last_newline {
                Some(k) => k + 1,
                // `\s+(?!\S)` takes the whole run at end of text;
                // otherwise `\s+` leaves the final character for the piece
                // that follows, as in GPT-2's rule.
                None if ch(j).is_some() && j - i > 1 => j - 1,
                None => j,
            };
            out.push(&text[start..at_byte(stop)]);
            i = stop.max(i + 1);
            continue;
        }

        i += 1;
    }
    out
}

/// Which of the pattern's three character runs `c` belongs to.
#[derive(PartialEq, Eq, Clone, Copy)]
enum Class {
    Letter,
    Number,
    Other,
}

fn classify(c: char) -> Class {
    if c.is_alphabetic() {
        Class::Letter
    } else if c.is_numeric() {
        Class::Number
    } else {
        Class::Other
    }
}

// ---------------------------------------------------------------------------
// GPT-2's byte alphabet
//
// Byte-level BPE stores its vocabulary as text, so every byte needs a
// printable stand-in. GPT-2 keeps the bytes that are already printable and
// maps the remaining 68 to U+0100 upward.
// ---------------------------------------------------------------------------

/// Whether a byte stands for itself in the alphabet.
fn byte_is_printable(b: u8) -> bool {
    (b'!'..=b'~').contains(&b) || (0xA1..=0xAC).contains(&b) || (0xAE..=0xFF).contains(&b)
}

/// The character standing for byte `b`.
fn byte_to_unicode(b: u8) -> char {
    if byte_is_printable(b) {
        return b as char;
    }
    // The nth non-printable byte, in order, becomes U+0100 + n.
    let rank = (0..b).filter(|&x| !byte_is_printable(x)).count();
    char::from_u32(0x100 + rank as u32).expect("0x100..0x144 are all valid scalar values")
}

/// The byte a character stands for, inverting [`byte_to_unicode`].
fn unicode_to_byte(c: char) -> Option<u8> {
    let point = c as u32;
    if point < 0x100 {
        let b = point as u8;
        return byte_is_printable(b).then_some(b);
    }
    let rank = point.checked_sub(0x100)? as usize;
    (0u16..256)
        .map(|b| b as u8)
        .filter(|&b| !byte_is_printable(b))
        .nth(rank)
}

/// Parse SentencePiece's `<0x41>` byte token.
fn parse_byte_token(text: &str) -> Option<u8> {
    let hex = text.strip_prefix("<0x")?.strip_suffix('>')?;
    u8::from_str_radix(hex, 16).ok()
}

// ---------------------------------------------------------------------------
// Metadata reads
// ---------------------------------------------------------------------------

fn meta_str<'a>(model: &'a GgufModel, key: &str) -> Option<&'a str> {
    model.metadata.get(key)?.as_str()
}

fn meta_bool(model: &GgufModel, key: &str) -> Option<bool> {
    match *model.metadata.get(key)? {
        GgufValue::Bool(b) => Some(b),
        ref other => other.as_u64().map(|v| v != 0),
    }
}

/// Read a token id, discarding one that does not index the vocabulary.
///
/// Producers write `-1` for "absent" in a signed field, which `as_u64`
/// already rejects; an out-of-range positive id would otherwise panic on
/// first use instead.
fn meta_id(model: &GgufModel, key: &str, vocab_len: usize) -> Option<u32> {
    let raw = model.metadata.get(key)?.as_u64()?;
    let id = u32::try_from(raw).ok()?;
    ((id as usize) < vocab_len).then_some(id)
}

fn float_array(model: &GgufModel, key: &str, len: usize) -> Vec<f32> {
    let mut out = vec![0.0; len];
    if let Some(values) = model.metadata.get(key).and_then(GgufValue::as_array) {
        for (slot, value) in out.iter_mut().zip(values) {
            if let Some(v) = value.as_f64() {
                *slot = v as f32;
            }
        }
    }
    out
}

/// Read `tokenizer.ggml.token_type`, inferring where it is absent.
fn token_kinds(model: &GgufModel, tokens: &[String]) -> Vec<TokenKind> {
    let declared = model
        .metadata
        .get("tokenizer.ggml.token_type")
        .and_then(GgufValue::as_array);
    (0..tokens.len())
        .map(|i| {
            match declared.and_then(|d| d.get(i)).and_then(GgufValue::as_u64) {
                // GGML's llama_token_type: 1 NORMAL, 2 UNKNOWN, 3 CONTROL,
                // 4 USER_DEFINED, 5 UNUSED, 6 BYTE.
                Some(3) => TokenKind::Control,
                Some(4) => TokenKind::UserDefined,
                Some(6) => TokenKind::Byte,
                Some(_) => TokenKind::Normal,
                // Absent: `<0x41>` is unambiguous, and a `<|...|>` marker
                // is conventional enough to read as one.
                None if parse_byte_token(&tokens[i]).is_some() => TokenKind::Byte,
                None if tokens[i].starts_with("<|") && tokens[i].ends_with("|>") => {
                    TokenKind::Control
                }
                None => TokenKind::Normal,
            }
        })
        .collect()
}

/// The merge list, as ranks. Rank is position: earlier merges first.
fn merge_ranks(model: &GgufModel) -> Result<HashMap<(String, String), u32>, GgufError> {
    let raw = model
        .metadata
        .get("tokenizer.ggml.merges")
        .and_then(GgufValue::as_array)
        .ok_or_else(|| GgufError::MissingKey("tokenizer.ggml.merges".to_string()))?;
    let mut out = HashMap::with_capacity(raw.len());
    for (rank, entry) in raw.iter().enumerate() {
        let Some(line) = entry.as_str() else {
            return Err(GgufError::BadMetadata(
                "tokenizer.ggml.merges holds a non-string".to_string(),
            ));
        };
        // Each entry is the two halves separated by one space. The halves
        // are in the byte alphabet, where a literal space is `Ġ`, so
        // splitting on the first space is unambiguous.
        let Some((left, right)) = line.split_once(' ') else {
            return Err(GgufError::BadMetadata(format!(
                "merge `{line}` is not two pieces separated by a space"
            )));
        };
        out.entry((left.to_string(), right.to_string()))
            .or_insert(rank as u32);
    }
    Ok(out)
}

/// The id of each raw byte, for SentencePiece's fallback.
fn byte_ids(
    kind: TokenizerKind,
    tokens: &[String],
    ids: &HashMap<String, u32>,
    kinds: &[TokenKind],
) -> Vec<Option<u32>> {
    let mut out = vec![None; 256];
    match kind {
        TokenizerKind::SentencePiece => {
            for (id, token) in tokens.iter().enumerate() {
                if kinds.get(id) == Some(&TokenKind::Byte)
                    && let Some(b) = parse_byte_token(token)
                {
                    out[b as usize].get_or_insert(id as u32);
                }
            }
        }
        TokenizerKind::Bpe => {
            // Every byte has a single-character token of its own.
            for b in 0..=255u8 {
                out[b as usize] = ids.get(&byte_to_unicode(b).to_string()).copied();
            }
        }
    }
    out
}

/// Ids that should stop generation: the declared EOS, plus the end-of-turn
/// markers instruction-tuned models actually emit.
fn end_of_generation(model: &GgufModel, tokens: &[String], eos: Option<u32>) -> Vec<u32> {
    let mut out = Vec::new();
    let mut push = |id: u32| {
        if !out.contains(&id) {
            out.push(id);
        }
    };
    if let Some(eos) = eos {
        push(eos);
    }
    for key in ["tokenizer.ggml.eot_token_id", "tokenizer.ggml.eom_token_id"] {
        if let Some(id) = meta_id(model, key, tokens.len()) {
            push(id);
        }
    }
    // A chat-tuned model ends its turn with one of these rather than with
    // the declared EOS, and a generator watching only EOS would never stop.
    for (id, token) in tokens.iter().enumerate() {
        if matches!(
            token.as_str(),
            "<|im_end|>" | "<|eot_id|>" | "<|end|>" | "<|endoftext|>" | "<end_of_turn>"
        ) {
            push(id as u32);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    // -- the byte alphabet ------------------------------------------------

    /// A SentencePiece vocabulary of exactly the pieces given, all scored
    /// alike, with no byte fallback and no automatic BOS — so segmentation
    /// is the only thing under test.
    fn tied_score_vocab(pieces: &[&str]) -> Vocab {
        let mut tokens = vec!["<unk>".to_string()];
        let mut scores = vec![0.0f32];
        let mut types = vec![2u32];
        for piece in pieces {
            tokens.push((*piece).to_string());
            scores.push(-1.0);
            types.push(1);
        }
        let mut metadata = std::collections::HashMap::new();
        metadata.insert(
            "tokenizer.ggml.model".to_string(),
            GgufValue::String("llama".to_string()),
        );
        metadata.insert(
            "tokenizer.ggml.tokens".to_string(),
            value_array(tokens.into_iter().map(GgufValue::String).collect()),
        );
        metadata.insert(
            "tokenizer.ggml.scores".to_string(),
            value_array(scores.into_iter().map(GgufValue::F32).collect()),
        );
        metadata.insert(
            "tokenizer.ggml.token_type".to_string(),
            value_array(types.into_iter().map(GgufValue::U32).collect()),
        );
        metadata.insert(
            "tokenizer.ggml.unknown_token_id".to_string(),
            GgufValue::U32(0),
        );
        metadata.insert(
            "tokenizer.ggml.add_bos_token".to_string(),
            GgufValue::Bool(false),
        );
        metadata.insert(
            "tokenizer.ggml.add_eos_token".to_string(),
            GgufValue::Bool(false),
        );
        Vocab::from_gguf(&GgufModel {
            metadata,
            tensors: std::collections::HashMap::new(),
        })
        .expect("a minimal SentencePiece vocabulary")
    }

    /// The pieces `text` encodes to, less the word-prefix marker.
    ///
    /// SentencePiece prefixes a word with U+2581, which
    /// [`tied_score_vocab`] deliberately cannot represent — it carries no
    /// byte fallback, so the prefix arrives as unknowns. Dropping them
    /// leaves the merge order, which is what these tests are about.
    fn merged_pieces<'a>(vocab: &'a Vocab, text: &str) -> Vec<&'a str> {
        vocab
            .encode(text)
            .iter()
            .filter_map(|&id| vocab.token_text(id))
            .filter(|piece| *piece != "<unk>")
            .collect()
    }

    #[test]
    fn an_equal_score_merges_the_leftmost_pair() {
        // `ab` and `bc` both apply to `abc` and score the same. GGML's
        // `llm_bigram_spm` comparator breaks the tie by the lower left
        // index, giving `[ab, c]`; taking the last maximum instead gives
        // `[a, bc]`, which is a different tokenization of the same text.
        let vocab = tied_score_vocab(&["a", "b", "c", "ab", "bc"]);
        assert_eq!(
            merged_pieces(&vocab, "abc"),
            ["ab", "c"],
            "the leftmost tied pair should win"
        );
    }

    #[test]
    fn a_higher_score_still_beats_position() {
        // The tie-break must only apply to ties: `bc` scoring higher has
        // to win even though `ab` is further left.
        let mut vocab = tied_score_vocab(&["a", "b", "c", "ab", "bc"]);
        let bc = vocab.ids["bc"] as usize;
        vocab.scores[bc] = 10.0;
        assert_eq!(merged_pieces(&vocab, "abc"), ["a", "bc"]);
    }

    #[test]
    fn the_pre_tokenizers_disagree_exactly_where_upstream_says_they_do() {
        // Each row is a case where reading `tokenizer.ggml.model = gpt2`
        // and assuming GPT-2's rule would place different boundaries than
        // the file's own pre-tokenizer. Merges cannot cross these, so a
        // wrong rule cannot be repaired by the merge list.
        /// Input, then the boundaries GPT-2, Llama 3 and Qwen2 place.
        type Row = (
            &'static str,
            &'static [&'static str],
            &'static [&'static str],
            &'static [&'static str],
        );
        let cases: [Row; 3] = [
            ("1234", &["1234"], &["123", "4"], &["1", "2", "3", "4"]),
            (
                "I'M here",
                &["I", "'", "M", " here"],
                &["I", "'M", " here"],
                &["I", "'M", " here"],
            ),
            (
                "hi!\nworld",
                &["hi", "!", "\n", "world"],
                &["hi", "!\n", "world"],
                &["hi", "!\n", "world"],
            ),
        ];
        for (input, gpt2, llama3, qwen2) in cases {
            assert_eq!(PreTokenizer::Gpt2.split(input), gpt2, "gpt2 {input:?}");
            assert_eq!(
                PreTokenizer::Llama3.split(input),
                llama3,
                "llama3 {input:?}"
            );
            assert_eq!(PreTokenizer::Qwen2.split(input), qwen2, "qwen2 {input:?}");
        }
    }

    #[test]
    fn smollm_severs_digits_and_keeps_gpt2s_rule_elsewhere() {
        // llama.cpp lists `\p{N}` before GPT-2's pattern, so the digit
        // split happens first and takes the leading space with it.
        assert_eq!(PreTokenizer::SmolLm.split(" 12 ab"), [" ", "1", "2", " ab"]);
        assert_eq!(PreTokenizer::Gpt2.split(" 12 ab"), [" 12", " ab"]);
        // Away from digits the two agree.
        assert_eq!(
            PreTokenizer::SmolLm.split("Hello world"),
            PreTokenizer::Gpt2.split("Hello world")
        );
    }

    #[test]
    fn a_letter_run_may_follow_one_free_character() {
        // `[^\r\n\p{L}\p{N}]?\p{L}+` takes any single non-letter that
        // is not a newline, where GPT-2 only ever joins a space.
        assert_eq!(PreTokenizer::Llama3.split("(word"), ["(word"]);
        assert_eq!(PreTokenizer::Gpt2.split("(word"), ["(", "word"]);
        // But only when letters actually follow it.
        assert_eq!(PreTokenizer::Llama3.split("((("), ["((("]);
    }

    #[test]
    fn an_unimplemented_pre_tokenizer_is_refused_by_name() {
        let err = PreTokenizer::from_name(Some("gpt-4o")).unwrap_err();
        assert!(format!("{err}").contains("gpt-4o"), "{err}");
        // `default` is its own cascade upstream, not a synonym for gpt-2.
        assert!(PreTokenizer::from_name(Some("default")).is_err());
        // Absent, the file predates the key.
        assert_eq!(PreTokenizer::from_name(None).unwrap(), PreTokenizer::Gpt2);
        assert_eq!(
            PreTokenizer::from_name(Some("llama-bpe")).unwrap(),
            PreTokenizer::Llama3
        );
    }

    #[test]
    fn the_byte_alphabet_is_a_bijection() {
        let mut seen = std::collections::HashSet::new();
        for b in 0..=255u8 {
            let c = byte_to_unicode(b);
            assert!(seen.insert(c), "byte {b} collides on {c:?}");
            assert_eq!(unicode_to_byte(c), Some(b), "byte {b} does not round trip");
        }
        assert_eq!(seen.len(), 256);
    }

    #[test]
    fn the_alphabet_matches_gpt2s_own_table() {
        // A few anchors from the reference implementation.
        assert_eq!(byte_to_unicode(b'A'), 'A');
        assert_eq!(byte_to_unicode(b'~'), '~');
        // Space is the first non-printable byte at or below itself: bytes
        // 0..=0x20 are all non-printable, so space has rank 32.
        assert_eq!(byte_to_unicode(b' '), 'Ġ');
        assert_eq!(byte_to_unicode(b'\n'), 'Ċ');
        assert_eq!(byte_to_unicode(0), 'Ā');
        assert_eq!(unicode_to_byte('Ġ'), Some(b' '));
        assert_eq!(unicode_to_byte('Ċ'), Some(b'\n'));
    }

    #[test]
    fn a_character_outside_the_alphabet_has_no_byte() {
        assert_eq!(unicode_to_byte('文'), None);
    }

    // -- pre-tokenization -------------------------------------------------

    #[test]
    fn a_leading_space_joins_the_word_after_it() {
        assert_eq!(scan_gpt2("Hello world"), vec!["Hello", " world"]);
    }

    #[test]
    fn letters_numbers_and_punctuation_are_separate_runs() {
        assert_eq!(scan_gpt2("abc123!!"), vec!["abc", "123", "!!"]);
    }

    #[test]
    fn contractions_split_the_way_the_pattern_lists_them() {
        assert_eq!(scan_gpt2("don't"), vec!["don", "'t"]);
        assert_eq!(scan_gpt2("we've"), vec!["we", "'ve"]);
        assert_eq!(scan_gpt2("they'll"), vec!["they", "'ll"]);
        assert_eq!(scan_gpt2("it's"), vec!["it", "'s"]);
    }

    #[test]
    fn a_run_of_spaces_leaves_its_last_for_the_word() {
        // The lookahead means "a   b" is "a", "  ", " b" — the final space
        // belongs to the word, not the run.
        assert_eq!(scan_gpt2("a   b"), vec!["a", "  ", " b"]);
    }

    #[test]
    fn trailing_whitespace_is_kept_whole() {
        assert_eq!(scan_gpt2("a   "), vec!["a", "   "]);
        assert_eq!(scan_gpt2("hi\n"), vec!["hi", "\n"]);
    }

    #[test]
    fn pretokenizing_never_loses_a_byte() {
        for text in [
            "Hello, world!",
            "a   b",
            "don't stop",
            "  leading",
            "trailing   ",
            "123 + 456 = 579",
            "日本語のテキスト",
            "mixed 文字 and text",
            "\n\nparagraph\n\n",
            "\t\ttabs",
            "",
        ] {
            assert_eq!(
                scan_gpt2(text).concat(),
                text,
                "pretokenizing {text:?} changed it"
            );
        }
    }

    #[test]
    fn unicode_letters_group_with_letters() {
        assert_eq!(scan_gpt2("héllo"), vec!["héllo"]);
        assert_eq!(scan_gpt2(" 日本"), vec![" 日本"]);
    }

    // -- vocabularies -----------------------------------------------------

    fn value_array(items: Vec<GgufValue>) -> GgufValue {
        GgufValue::Array(items)
    }

    fn strings(items: &[&str]) -> GgufValue {
        value_array(
            items
                .iter()
                .map(|s| GgufValue::String((*s).to_string()))
                .collect(),
        )
    }

    /// A byte-level BPE vocabulary over a tiny alphabet, with merges that
    /// build "hello" and " world" out of it.
    fn bpe_model() -> GgufModel {
        let mut tokens: Vec<String> = (0..=255u8)
            .map(|b| byte_to_unicode(b).to_string())
            .collect();
        // Merges, in the order they should apply.
        let merges = ["h e", "he l", "hel l", "hell o", "Ġ w", "Ġw o"];
        for merge in merges {
            let (a, b) = merge.split_once(' ').unwrap();
            tokens.push(format!("{a}{b}"));
        }
        tokens.push("<|endoftext|>".to_string());
        let eos = (tokens.len() - 1) as u32;

        let mut metadata = std::collections::HashMap::new();
        metadata.insert(
            "tokenizer.ggml.model".to_string(),
            GgufValue::String("gpt2".to_string()),
        );
        metadata.insert(
            "tokenizer.ggml.tokens".to_string(),
            value_array(tokens.into_iter().map(GgufValue::String).collect()),
        );
        metadata.insert("tokenizer.ggml.merges".to_string(), strings(&merges));
        metadata.insert(
            "tokenizer.ggml.eos_token_id".to_string(),
            GgufValue::U32(eos),
        );
        metadata.insert(
            "tokenizer.ggml.add_bos_token".to_string(),
            GgufValue::Bool(false),
        );
        GgufModel {
            metadata,
            tensors: std::collections::HashMap::new(),
        }
    }

    /// A SentencePiece vocabulary: pieces with scores, plus byte fallbacks.
    fn spm_model() -> GgufModel {
        let mut tokens = vec!["<unk>".to_string(), "<s>".to_string(), "</s>".to_string()];
        let mut scores = vec![0.0f32, 0.0, 0.0];
        let mut types = vec![2u32, 3, 3];

        // SentencePiece only ever merges two adjacent pieces that are
        // *themselves* in the vocabulary, so reaching `▁hello` needs the
        // whole chain present, exactly as a trained vocabulary has it:
        //   ▁ h e l l o  -(-4)- ll  -(-5)- he  -(-7)- ▁he
        //                -(-3)- ▁hell  -(-1)- ▁hello
        for (piece, score) in [
            ("\u{2581}hello", -1.0f32),
            ("\u{2581}hell", -3.0),
            ("\u{2581}he", -7.0),
            ("\u{2581}h", -8.0),
            ("e", -9.0),
            ("l", -9.0),
            ("o", -9.0),
            ("\u{2581}", -10.0),
            ("h", -9.0),
            ("he", -5.0),
            ("ll", -4.0),
        ] {
            tokens.push(piece.to_string());
            scores.push(score);
            types.push(1);
        }
        for b in 0..=255u8 {
            tokens.push(format!("<0x{b:02X}>"));
            scores.push(-20.0);
            types.push(6);
        }

        let mut metadata = std::collections::HashMap::new();
        metadata.insert(
            "tokenizer.ggml.model".to_string(),
            GgufValue::String("llama".to_string()),
        );
        metadata.insert(
            "tokenizer.ggml.tokens".to_string(),
            value_array(tokens.into_iter().map(GgufValue::String).collect()),
        );
        metadata.insert(
            "tokenizer.ggml.scores".to_string(),
            value_array(scores.into_iter().map(GgufValue::F32).collect()),
        );
        metadata.insert(
            "tokenizer.ggml.token_type".to_string(),
            value_array(types.into_iter().map(GgufValue::U32).collect()),
        );
        metadata.insert("tokenizer.ggml.bos_token_id".to_string(), GgufValue::U32(1));
        metadata.insert("tokenizer.ggml.eos_token_id".to_string(), GgufValue::U32(2));
        metadata.insert(
            "tokenizer.ggml.unknown_token_id".to_string(),
            GgufValue::U32(0),
        );
        GgufModel {
            metadata,
            tensors: std::collections::HashMap::new(),
        }
    }

    #[test]
    fn bpe_merges_in_rank_order() {
        let v = Vocab::from_gguf(&bpe_model()).unwrap();
        assert_eq!(v.kind(), TokenizerKind::Bpe);
        let ids = v.encode("hello world");
        // "hello" merges to one token, " world" gets as far as " wo".
        assert_eq!(v.decode(&ids), "hello world");
        assert_eq!(
            v.token_text(ids[0]),
            Some("hello"),
            "the merge list builds the whole word"
        );
    }

    #[test]
    fn bpe_round_trips_arbitrary_bytes() {
        let v = Vocab::from_gguf(&bpe_model()).unwrap();
        for text in [
            "hello world",
            "Hello, World!",
            "日本語",
            "  spaced  out  ",
            "tabs\tand\nnewlines",
            "emoji 🎉 here",
            "",
        ] {
            assert_eq!(v.decode(&v.encode(text)), text, "round trip of {text:?}");
        }
    }

    #[test]
    fn bpe_adds_no_bos_when_the_file_says_not_to() {
        let v = Vocab::from_gguf(&bpe_model()).unwrap();
        assert_eq!(v.encode("hello"), v.encode_plain("hello"));
    }

    #[test]
    fn spm_prefixes_a_space_and_merges_by_score() {
        let v = Vocab::from_gguf(&spm_model()).unwrap();
        assert_eq!(v.kind(), TokenizerKind::SentencePiece);
        let ids = v.encode_plain("hello");
        // "▁hello" is the best-scoring piece and should win outright.
        assert_eq!(ids.len(), 1);
        assert_eq!(v.token_text(ids[0]), Some("\u{2581}hello"));
        assert_eq!(v.decode(&ids), " hello");
    }

    #[test]
    fn spm_adds_bos_by_default() {
        let v = Vocab::from_gguf(&spm_model()).unwrap();
        let with = v.encode("hello");
        assert_eq!(with[0], v.bos_id().unwrap());
        assert_eq!(&with[1..], &v.encode_plain("hello")[..]);
    }

    #[test]
    fn spm_falls_back_to_bytes_for_what_it_cannot_spell() {
        let v = Vocab::from_gguf(&spm_model()).unwrap();
        let ids = v.encode_plain("é");
        // SentencePiece always prefixes its visible space, so that is the
        // first token; `é` is two UTF-8 bytes and it can spell neither, so
        // two byte tokens follow and no unknown is emitted.
        assert_eq!(ids.len(), 3);
        assert_eq!(v.token_text(ids[0]), Some("\u{2581}"));
        for &id in &ids[1..] {
            assert!(
                v.token_text(id).unwrap().starts_with("<0x"),
                "expected a byte token, got {:?}",
                v.token_text(id)
            );
        }
        assert_eq!(v.decode(&ids), " é");
    }

    #[test]
    fn spm_control_tokens_do_not_render() {
        let v = Vocab::from_gguf(&spm_model()).unwrap();
        let bos = v.bos_id().unwrap();
        let ids = v.encode("hello");
        assert_eq!(ids[0], bos);
        assert!(
            !v.decode(&ids).contains("<s>"),
            "markers are structure, not text"
        );
    }

    #[test]
    fn end_of_generation_covers_more_than_the_declared_eos() {
        let v = Vocab::from_gguf(&bpe_model()).unwrap();
        let eos = v.eos_id().unwrap();
        assert!(v.is_end_of_generation(eos));
        // `<|endoftext|>` is the declared EOS here, so it is the same id;
        // the point is that the set is consulted rather than one id.
        assert!(!v.is_end_of_generation(0));
    }

    #[test]
    fn an_unimplemented_tokenizer_model_says_so() {
        let mut m = bpe_model();
        m.metadata.insert(
            "tokenizer.ggml.model".to_string(),
            GgufValue::String("rwkv".to_string()),
        );
        let err = Vocab::from_gguf(&m).unwrap_err();
        assert!(
            matches!(&err, GgufError::BadMetadata(e) if e.contains("rwkv")),
            "{err:?}"
        );
    }

    #[test]
    fn a_file_with_no_vocabulary_is_a_missing_key_not_a_panic() {
        let empty = GgufModel {
            metadata: std::collections::HashMap::new(),
            tensors: std::collections::HashMap::new(),
        };
        assert!(matches!(
            Vocab::from_gguf(&empty),
            Err(GgufError::MissingKey(_))
        ));
    }

    #[test]
    fn bpe_without_merges_is_refused_rather_than_tokenizing_per_byte() {
        let mut m = bpe_model();
        m.metadata.remove("tokenizer.ggml.merges");
        assert!(matches!(
            Vocab::from_gguf(&m),
            Err(GgufError::MissingKey(k)) if k.contains("merges")
        ));
    }

    #[test]
    fn an_out_of_range_special_id_is_dropped_rather_than_trusted() {
        let mut m = bpe_model();
        m.metadata.insert(
            "tokenizer.ggml.bos_token_id".to_string(),
            GgufValue::U32(999_999),
        );
        let v = Vocab::from_gguf(&m).unwrap();
        assert_eq!(v.bos_id(), None);
    }

    #[test]
    fn a_marker_in_the_prompt_is_matched_literally() {
        let mut m = bpe_model();
        // Add a chat marker after the merges, as a template would use.
        let tokens = match m.metadata.get_mut("tokenizer.ggml.tokens") {
            Some(&mut GgufValue::Array(ref mut tokens)) => tokens,
            _ => unreachable!(),
        };
        tokens.push(GgufValue::String("<|im_start|>".to_string()));
        let marker_id = (tokens.len() - 1) as u32;

        let v = Vocab::from_gguf(&m).unwrap();
        let ids = v.encode_plain("<|im_start|>hello");
        assert_eq!(ids[0], marker_id, "the marker is one token, not its bytes");
        assert_eq!(v.token_text(ids[1]), Some("hello"));
    }

    #[test]
    fn decoding_an_unknown_id_is_empty_rather_than_a_panic() {
        let v = Vocab::from_gguf(&bpe_model()).unwrap();
        assert_eq!(v.decode(&[u32::MAX]), "");
    }

    #[test]
    fn a_byte_token_parses_only_in_its_own_form() {
        assert_eq!(parse_byte_token("<0x41>"), Some(0x41));
        assert_eq!(parse_byte_token("<0xFF>"), Some(0xFF));
        assert_eq!(parse_byte_token("<0x>"), None);
        assert_eq!(parse_byte_token("0x41"), None);
        assert_eq!(parse_byte_token("hello"), None);
    }
}
