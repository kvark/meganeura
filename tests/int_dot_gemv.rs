//! The Q8_1-activation, integer-dot GEMV against GGML Q4_0 and Meganeura Q8
//! weights.
//!
//! This is the one kernel choice in the crate that changes the numbers: the
//! activation is quantized to 8 bits on top of an already-quantized weight.
//! So it cannot be checked the way every other kernel is, against the f32
//! result — it is not computing that. It is checked against a CPU model of
//! the arithmetic it is *supposed* to do — `vec_dot_q4_0_q8_1` and
//! `vec_dot_q8_0_q8_1` — which pins the layout handling, the -8 bias and
//! the `s8` correction term exactly.
//!
//! A second test bounds the error against the f32 path, so that the
//! quantization is shown to be a small perturbation rather than a wrong
//! answer that happens to agree with a wrong reference.

use meganeura::compile::{CompileOptions, ShaderEntry};
use meganeura::train::{Mode, SessionConfig};
use meganeura::{Graph, Session};

const BLOCK: usize = 32;
const BLOCK_BYTES: usize = 18;
const Q8_BLOCK_BYTES: usize = 36;

/// Which weight encoding a case runs: GGML Q4_0 native storage, Meganeura
/// Q8 (GGML Q8_0 after its host repack), or a GGML K-quant superblock
/// format.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Format {
    Q40,
    Q8,
    Q4K,
    Q5K,
    Q6K,
    Q3K,
}

/// One GGML Q4_0 block: an f16 scale then sixteen nibble bytes, byte `j`
/// carrying element `j` low and element `j + 16` high.
fn q40_block(d: f32, nibbles: &[u8; BLOCK]) -> Vec<u8> {
    let mut out = Vec::with_capacity(BLOCK_BYTES);
    out.extend_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
    for j in 0..16 {
        out.push((nibbles[j] & 0x0F) | (nibbles[j + 16] << 4));
    }
    out
}

/// A `[k, n]` weight in Meganeura Q8 layout — Meganeura Q8, i.e. GGML Q8_0
/// after its host repack — column-major in blocks the way packing lays it
/// out: all of column 0's blocks, then column 1's. One block is nine words:
/// the f16 scale in the low half of word zero, then the thirty-two int8
/// quants, four per word.
fn q8_weight(k: usize, n: usize, seed: u32) -> (Vec<u8>, Vec<f32>) {
    let mut state = seed | 1;
    let mut rnd = || {
        state = state.wrapping_mul(747796405).wrapping_add(2891336453);
        let w = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277803737);
        (w >> 22) ^ w
    };
    let mut packed = Vec::new();
    // Row-major [k, n] f32 reference, matching what the graph multiplies.
    let mut reference = vec![0.0f32; k * n];
    for col in 0..n {
        for blk in 0..k / BLOCK {
            let d = 0.01 + (rnd() % 13) as f32 * 0.003;
            let mut block = vec![0u8; Q8_BLOCK_BYTES];
            block[0..2].copy_from_slice(&half::f16::from_f32(d).to_bits().to_le_bytes());
            for (e, slot) in block[4..].iter_mut().enumerate() {
                // Pin both ends of the signed range in every block.
                let q: i32 = match e {
                    0 => -127,
                    1 => 127,
                    _ => (rnd() % 255) as i32 - 127,
                };
                *slot = q as i8 as u8;
                // The f16 round trip is what the GPU sees, so round here too.
                let d16 = half::f16::from_f32(d).to_f32();
                reference[(blk * BLOCK + e) * n + col] = d16 * q as f32;
            }
            packed.extend_from_slice(&block);
        }
    }
    (packed, reference)
}

/// A `[k, n]` packed weight in the requested encoding, column-major in
/// blocks the way packing lays it out: all of column 0's blocks, then
/// column 1's — per 256-element superblock for the K-quants.
fn weight(format: Format, k: usize, n: usize, seed: u32) -> (Vec<u8>, Vec<f32>) {
    match format {
        Format::Q40 => q40_weight(k, n, seed),
        Format::Q8 => q8_weight(k, n, seed),
        Format::Q4K => kquant_weight(format, k, n, seed, 144, q4k_superblock),
        Format::Q5K => kquant_weight(format, k, n, seed, 176, q5k_superblock),
        Format::Q6K => kquant_weight(format, k, n, seed, 210, q6k_superblock),
        Format::Q3K => kquant_weight(format, k, n, seed, 110, q3k_superblock),
    }
}

/// One Q4_K superblock with a realistic spread: per-sub-block scales and
/// mins that actually differ, and quants across the whole nibble range.
fn q4k_superblock(seed: u32) -> Vec<u8> {
    let mut st = seed | 1;
    let mut rnd = || {
        st = st.wrapping_mul(747796405).wrapping_add(2891336453);
        let w = ((st >> ((st >> 28) + 4)) ^ st).wrapping_mul(277803737);
        (w >> 22) ^ w
    };
    let mut b = vec![0u8; 144];
    b[0..2].copy_from_slice(&half::f16::from_f32(0.0035).to_bits().to_le_bytes());
    b[2..4].copy_from_slice(&half::f16::from_f32(0.0021).to_bits().to_le_bytes());
    let sc: Vec<u8> = (0..8).map(|_| (rnd() % 64) as u8).collect();
    let mn: Vec<u8> = (0..8).map(|_| (rnd() % 64) as u8).collect();
    for j in 0..4 {
        b[4 + j] = sc[j] & 63;
        b[8 + j] = mn[j] & 63;
    }
    for j in 4..8 {
        b[8 + j] = (sc[j] & 0x0F) | ((mn[j] & 0x0F) << 4);
        b[j] |= (sc[j] >> 4) << 6;
        b[4 + j] |= (mn[j] >> 4) << 6;
    }
    for i in 0..128 {
        b[16 + i] = (rnd() % 256) as u8;
    }
    b
}

/// One Q5_K superblock: a spread of scales and mins, plus full-range
/// nibbles and high bits.
fn q5k_superblock(seed: u32) -> Vec<u8> {
    let mut st = seed | 1;
    let mut rnd = || {
        st = st.wrapping_mul(747796405).wrapping_add(2891336453);
        let w = ((st >> ((st >> 28) + 4)) ^ st).wrapping_mul(277803737);
        (w >> 22) ^ w
    };
    let mut b = vec![0u8; 176];
    b[0..2].copy_from_slice(&half::f16::from_f32(0.0031).to_bits().to_le_bytes());
    b[2..4].copy_from_slice(&half::f16::from_f32(0.0019).to_bits().to_le_bytes());
    let sc: Vec<u8> = (0..8).map(|_| (rnd() % 64) as u8).collect();
    let mn: Vec<u8> = (0..8).map(|_| (rnd() % 64) as u8).collect();
    for j in 0..4 {
        b[4 + j] = sc[j] & 63;
        b[8 + j] = mn[j] & 63;
    }
    for j in 4..8 {
        b[8 + j] = (sc[j] & 0x0F) | ((mn[j] & 0x0F) << 4);
        b[j] |= (sc[j] >> 4) << 6;
        b[4 + j] |= (mn[j] >> 4) << 6;
    }
    // qh at 16..48 then qs at 48..176.
    for byte in b[16..176].iter_mut() {
        *byte = (rnd() % 256) as u8;
    }
    b
}

/// One Q6_K superblock: full-range ql/qh, int8 scales straddling zero.
fn q6k_superblock(seed: u32) -> Vec<u8> {
    let mut st = seed | 1;
    let mut rnd = || {
        st = st.wrapping_mul(747796405).wrapping_add(2891336453);
        let w = ((st >> ((st >> 28) + 4)) ^ st).wrapping_mul(277803737);
        (w >> 22) ^ w
    };
    let mut b = vec![0u8; 210];
    for byte in b[..192].iter_mut() {
        *byte = (rnd() % 256) as u8;
    }
    for byte in b[192..208].iter_mut() {
        *byte = ((rnd() % 80) as i32 - 40) as i8 as u8;
    }
    b[208..210].copy_from_slice(&half::f16::from_f32(0.0012).to_bits().to_le_bytes());
    b
}

/// One Q3_K superblock: full-range hmask, 2-bit quants and packed scales.
fn q3k_superblock(seed: u32) -> Vec<u8> {
    let mut st = seed | 1;
    let mut rnd = || {
        st = st.wrapping_mul(747796405).wrapping_add(2891336453);
        let w = ((st >> ((st >> 28) + 4)) ^ st).wrapping_mul(277803737);
        (w >> 22) ^ w
    };
    let mut b = vec![0u8; 110];
    for byte in b[..108].iter_mut() {
        *byte = (rnd() % 256) as u8;
    }
    b[108..110].copy_from_slice(&half::f16::from_f32(0.0042).to_bits().to_le_bytes());
    b
}

/// A `[k, n]` K-quant weight, column-major in superblocks, plus its
/// row-major f32 dequantization.
fn kquant_weight(
    format: Format,
    k: usize,
    n: usize,
    seed: u32,
    bytes: usize,
    build: fn(u32) -> Vec<u8>,
) -> (Vec<u8>, Vec<f32>) {
    let sblocks = k / 256;
    let mut packed = Vec::new();
    let mut reference = vec![0.0f32; k * n];
    for col in 0..n {
        for sb in 0..sblocks {
            let sup = build(seed.wrapping_add((col * sblocks + sb) as u32));
            packed.extend_from_slice(&sup);
            for e in 0..256 {
                reference[(sb * 256 + e) * n + col] = element_weight(format, &sup, 0, e / 32, e);
            }
        }
    }
    assert_eq!(packed.len(), n * sblocks * bytes);
    (packed, reference)
}

/// A `[k, n]` GGML Q4_0 weight, column-major in blocks the way packing lays
/// it out: all of column 0's blocks, then column 1's.
fn q40_weight(k: usize, n: usize, seed: u32) -> (Vec<u8>, Vec<f32>) {
    let mut state = seed | 1;
    let mut rnd = || {
        state = state.wrapping_mul(747796405).wrapping_add(2891336453);
        let w = ((state >> ((state >> 28) + 4)) ^ state).wrapping_mul(277803737);
        (w >> 22) ^ w
    };
    let mut packed = Vec::new();
    // Row-major [k, n] f32 reference, matching what the graph multiplies.
    let mut reference = vec![0.0f32; k * n];
    for col in 0..n {
        for blk in 0..k / BLOCK {
            let d = 0.01 + (rnd() % 13) as f32 * 0.003;
            let mut nibbles = [0u8; BLOCK];
            for (e, slot) in nibbles.iter_mut().enumerate() {
                // Pin both ends of the nibble range in every block.
                *slot = match e {
                    0 => 0,
                    1 => 15,
                    _ => (rnd() % 16) as u8,
                };
            }
            packed.extend_from_slice(&q40_block(d, &nibbles));
            // The f16 round trip is what the GPU sees, so round here too.
            let d16 = half::f16::from_f32(d).to_f32();
            for (e, &q) in nibbles.iter().enumerate() {
                reference[(blk * BLOCK + e) * n + col] = d16 * (f32::from(q) - 8.0);
            }
        }
    }
    packed.resize(packed.len().next_multiple_of(4), 0);
    (packed, reference)
}

/// How far a backend may move a scaled activation before it rounds.
///
/// The kernel computes `a * (1 / (amax / 127))`. A backend is free to
/// reassociate that into `(a / amax) * 127`, and the two differ in the last
/// couple of bits — enough to land on opposite sides of a rounding boundary.
/// Metal in particular compiles with fast math under the pinned Blade. This
/// is generous next to an f32 ULP and still far tighter than one quant step.
const REASSOCIATION_SLACK: f32 = 1.0e-4;

/// Quantizations a backend may legitimately choose for one scaled value.
///
/// Both neighbours, not just one. Association is only half the story: at a
/// tie the two directions are equally correct and backends disagree about
/// which to take — WGSL rounds halves to even, Metal's own `round` takes
/// them away from zero — so a value sitting on a boundary has to admit the
/// rounding the reference did *not* choose. Taking only the upward
/// neighbour left a tie with an empty envelope, which is what failed on
/// Metal at `k=64`.
fn quant_choices(scaled: f32) -> [i32; 3] {
    let quant = |v: f32| (v.round_ties_even() as i32).clamp(-127, 127);
    [
        quant(scaled),
        quant(scaled - REASSOCIATION_SLACK),
        quant(scaled + REASSOCIATION_SLACK),
    ]
}

/// `vec_dot_q4_0_q8_1` / `vec_dot_q8_0_q8_1` on the host, as an envelope
/// rather than one number.
///
/// Returns the nominal result and the range the kernel is allowed to land in.
/// Quantization is a rounding decision, and an activation whose scaled value
/// sits within [`REASSOCIATION_SLACK`] of a half-integer may legitimately
/// round either way depending on how the backend associates the scaling.
/// Every element contributes additively, so summing each one's two possible
/// contributions gives an exact bound rather than a guessed tolerance: the
/// envelope stays tight wherever no element is near a boundary.
fn int_dot_reference(
    format: Format,
    a: &[f32],
    packed: &[u8],
    k: usize,
    n: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let blocks = k / BLOCK;
    let mut nominal = vec![0.0f32; n];
    let mut lo = vec![0.0f32; n];
    let mut hi = vec![0.0f32; n];
    for col in 0..n {
        let (mut acc, mut acc_lo, mut acc_hi) = (0.0f32, 0.0f32, 0.0f32);
        for blk in 0..blocks {
            let chunk = &a[blk * BLOCK..(blk + 1) * BLOCK];
            let amax = chunk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let d8 = amax / 127.0;
            let inv = if amax > 0.0 { 1.0 / d8 } else { 0.0 };
            let scaled: Vec<f32> = chunk.iter().map(|v| v * inv).collect();
            let choices: Vec<[i32; 3]> = scaled.iter().map(|&v| quant_choices(v)).collect();
            let q8: Vec<i32> = choices.iter().map(|c| c[0]).collect();

            for e in 0..BLOCK {
                // Q4_0/Q8 have one block per activation block; K-quants
                // address a 256-element superblock per column, and the
                // element's position inside it is the sub-block's 32
                // elements plus the local offset.
                let (at, e_super) = match format {
                    Format::Q40 => ((col * blocks + blk) * BLOCK_BYTES, e),
                    Format::Q8 => ((col * blocks + blk) * Q8_BLOCK_BYTES, e),
                    Format::Q4K => ((col * (blocks / 8) + blk / 8) * 144, (blk % 8) * 32 + e),
                    Format::Q5K => ((col * (blocks / 8) + blk / 8) * 176, (blk % 8) * 32 + e),
                    Format::Q6K => ((col * (blocks / 8) + blk / 8) * 210, (blk % 8) * 32 + e),
                    Format::Q3K => ((col * (blocks / 8) + blk / 8) * 110, (blk % 8) * 32 + e),
                };
                let weight = d8 * element_weight(format, packed, at, e_super / 32, e_super);
                acc += weight * q8[e] as f32;
                let mut low = f32::INFINITY;
                let mut high = f32::NEG_INFINITY;
                for &q in &choices[e] {
                    let contribution = weight * q as f32;
                    low = low.min(contribution);
                    high = high.max(contribution);
                }
                acc_lo += low;
                acc_hi += high;
            }
        }
        nominal[col] = acc;
        lo[col] = acc_lo;
        hi[col] = acc_hi;
    }
    (nominal, lo, hi)
}

/// GGML's `get_scale_min_k4` over the twelve packed scale bytes of a Q4_K
/// or Q5_K superblock, as `(scale, min)` nibble pairs.
fn get_scale_min_k4(j: usize, scales: &[u8]) -> (i32, i32) {
    if j < 4 {
        ((scales[j] & 63) as i32, (scales[j + 4] & 63) as i32)
    } else {
        let sc = ((scales[j + 4] & 0xF) as i32) | (((scales[j - 4] >> 6) as i32) << 4);
        let mn = ((scales[j + 4] >> 4) as i32) | (((scales[j] >> 6) as i32) << 4);
        (sc, mn)
    }
}

/// The dequantized weight for element `e` of the 32-element block `j` that
/// pairs with one Q8_1 activation block. Q4_0/Q8 blocks start at `at`;
/// K-quants address their superblock at `at` and index sub-block `j` inside
/// it. This mirrors GGML's dequantizers — the same reference the K-quant
/// decoders are pinned against — and keeps the f16 round trip the GPU sees.
fn element_weight(format: Format, packed: &[u8], at: usize, j: usize, e: usize) -> f32 {
    match format {
        Format::Q40 => {
            let scale = half::f16::from_le_bytes([packed[at], packed[at + 1]]).to_f32();
            let byte = packed[at + 2 + (e % 16)];
            let nib = if e < 16 { byte & 0x0F } else { byte >> 4 };
            scale * (i32::from(nib) - 8) as f32
        }
        Format::Q8 => {
            let scale = half::f16::from_le_bytes([packed[at], packed[at + 1]]).to_f32();
            scale * i32::from(packed[at + 4 + e] as i8) as f32
        }
        Format::Q4K => {
            let d = half::f16::from_le_bytes([packed[at], packed[at + 1]]).to_f32();
            let dmin = half::f16::from_le_bytes([packed[at + 2], packed[at + 3]]).to_f32();
            let scales = &packed[at + 4..at + 16];
            let (sc, mn) = get_scale_min_k4(j, scales);
            let byte = packed[at + 16 + (e / 64) * 32 + e % 32];
            let q = if (e % 64) < 32 { byte & 0xF } else { byte >> 4 };
            d * sc as f32 * q as f32 - dmin * mn as f32
        }
        Format::Q5K => {
            let d = half::f16::from_le_bytes([packed[at], packed[at + 1]]).to_f32();
            let dmin = half::f16::from_le_bytes([packed[at + 2], packed[at + 3]]).to_f32();
            let scales = &packed[at + 4..at + 16];
            let (sc, mn) = get_scale_min_k4(j, scales);
            let byte = packed[at + 48 + (e / 64) * 32 + e % 32];
            let nib = if (e % 64) < 32 { byte & 0xF } else { byte >> 4 };
            let hi = (packed[at + 16 + e % 32] >> (e / 32)) & 1;
            d * sc as f32 * (nib as i32 + 16 * hi as i32) as f32 - dmin * mn as f32
        }
        Format::Q6K => {
            let d = half::f16::from_le_bytes([packed[at + 208], packed[at + 209]]).to_f32();
            let half = e / 128;
            let within = e % 128;
            let jq = within / 32;
            let l = within % 32;
            let ql = packed[at + half * 64 + l + (jq & 1) * 32];
            let lo = if jq < 2 { ql & 0xF } else { ql >> 4 };
            let qh = packed[at + 128 + half * 32 + l];
            let hi = (qh >> (jq * 2)) & 3;
            let sc = packed[at + 192 + half * 8 + jq * 2 + l / 16] as i8 as i32;
            let q6 = (lo as i32) | ((hi as i32) << 4);
            d * sc as f32 * (q6 - 32) as f32
        }
        Format::Q3K => {
            let d = half::f16::from_le_bytes([packed[at + 108], packed[at + 109]]).to_f32();
            let h = e / 128;
            let jq = (e % 128) / 32;
            let l = e % 32;
            let q2 = (packed[at + 32 + h * 32 + l] >> (jq * 2)) & 3;
            let hbit = (packed[at + l] >> (e / 32)) & 1;
            let idx = e / 16;
            let b = idx % 4;
            let g = idx / 4;
            let src = if g.is_multiple_of(2) { b } else { 4 + b };
            let raw = packed[at + 96 + src];
            let nib = if g < 2 { raw & 0xF } else { raw >> 4 };
            let hi = (packed[at + 96 + 8 + b] >> (g * 2)) & 3;
            let sc = ((nib as i32) | ((hi as i32) << 4)) - 32;
            let quant = q2 as i32 - if hbit == 1 { 0 } else { 4 };
            d * sc as f32 * quant as f32
        }
    }
}

/// Is every output inside the envelope, allowing for float summation noise?
fn within_envelope(got: &[f32], lo: &[f32], hi: &[f32], noise: f32) -> Option<String> {
    assert!(
        got.iter().all(|v| v.is_finite()),
        "the kernel produced a non-finite value"
    );
    for (index, &value) in got.iter().enumerate() {
        if value < lo[index] - noise || value > hi[index] + noise {
            return Some(format!(
                "output {index} = {value} outside [{}, {}] (noise {noise})",
                lo[index], hi[index]
            ));
        }
    }
    None
}

fn run(
    format: Format,
    k: usize,
    n: usize,
    a: &[f32],
    packed: &[u8],
    quantized_activations: bool,
    addend: Option<&[f32]>,
) -> Vec<f32> {
    let mut g = Graph::new();
    let x = g.input("x", &[1, k]);
    let w = match format {
        Format::Q40 => g.parameter_q40("w", &[k, n]),
        Format::Q8 => g.parameter_q8("w", &[k, n]),
        Format::Q4K => g.parameter_q4k("w", &[k, n]),
        Format::Q5K => g.parameter_q5k("w", &[k, n]),
        Format::Q6K => g.parameter_q6k("w", &[k, n]),
        Format::Q3K => g.parameter_q3k("w", &[k, n]),
    };
    let y = g.matmul(x, w);
    let y = if addend.is_some() {
        let d = g.input("d", &[1, n]);
        g.add(y, d)
    } else {
        y
    };
    g.set_outputs(vec![y]);

    let (mut session, _): (Session, _) = meganeura::train::build(
        &g,
        SessionConfig {
            mode: Mode::Inference,
            options: CompileOptions {
                quantized_activations,
                ..CompileOptions::from_env()
            },
            ..SessionConfig::from_env()
        },
    );
    let dispatch = session
        .plan()
        .dispatches
        .iter()
        .find(|d| {
            matches!(
                d.shader,
                ShaderEntry::MatMulGemv | ShaderEntry::MatMulGemvAdd
            )
        })
        .expect("the test graph should compile to one GEMV");
    assert_eq!(
        dispatch.shader,
        if addend.is_some() {
            ShaderEntry::MatMulGemvAdd
        } else {
            ShaderEntry::MatMulGemv
        }
    );
    assert_eq!(dispatch.gemv_int_dot(), quantized_activations);
    session.set_input("x", a);
    if let Some(d) = addend {
        session.set_input("d", d);
    }
    session.set_parameter_packed("w", packed);
    session.step();
    session.wait();
    session.read_output(n)
}

fn run_q40_rmsnorm(
    k: usize,
    n: usize,
    x: &[f32],
    norm_weight: &[f32],
    packed: &[u8],
    expose_normalized: bool,
) -> Vec<f32> {
    let mut g = Graph::new();
    let input = g.input("x", &[1, k]);
    let scale = g.parameter("norm", &[k]);
    let normalized = g.rms_norm(input, scale, 1.0e-6);
    let weight = g.parameter_q40("w", &[k, n]);
    let output = g.matmul(normalized, weight);
    if expose_normalized {
        g.set_outputs(vec![normalized, output]);
    } else {
        g.set_outputs(vec![output]);
    }

    let (mut session, _): (Session, _) = meganeura::train::build(
        &g,
        SessionConfig {
            mode: Mode::Inference,
            options: CompileOptions {
                quantized_activations: true,
                ..CompileOptions::from_env()
            },
            ..SessionConfig::from_env()
        },
    );
    let dispatch = session
        .plan()
        .dispatches
        .iter()
        .find(|dispatch| dispatch.shader == ShaderEntry::MatMulGemv)
        .expect("RmsNorm output should feed a GEMV");
    assert!(dispatch.gemv_int_dot());
    assert_eq!(dispatch.gemv_rmsnorm.is_some(), !expose_normalized);

    session.set_input("x", x);
    session.set_parameter("norm", norm_weight);
    session.set_parameter_packed("w", packed);
    session.step();
    session.wait();
    if expose_normalized {
        let mut values = vec![0.0; n];
        session.read_output_by_index(1, &mut values);
        values
    } else {
        session.read_output(n)
    }
}

/// Largest absolute difference between two outputs.
///
/// `f32::max` returns the non-NaN operand, so folding differences with it
/// reports an all-NaN output as agreeing perfectly. Check finiteness first
/// rather than letting a kernel that produced nothing pass as exact.
fn max_abs_diff(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len());
    assert!(
        got.iter().all(|v| v.is_finite()),
        "the kernel produced a non-finite value"
    );
    got.iter()
        .zip(want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

fn activation(k: usize) -> Vec<f32> {
    (0..k)
        .map(|i| ((i * 37 % 61) as f32 - 30.0) * 0.021)
        .collect()
}

/// The kernel must reproduce the integer arithmetic it claims to do.
///
/// Shapes cover an even and an odd block count, because an 18-byte Q4_0
/// block is not a whole word: the nibble reads are word-aligned for odd
/// blocks and straddle two words for even ones, and only an odd block count
/// pads the buffer at all. Meganeura's Q8 block is nine whole words, so its
/// reads are always aligned — the pairing under test is which int8s feed
/// which dot product. The K-quants span 256-element superblocks whose
/// 210- and 110-byte forms are never whole words either, so their shapes
/// cover whole and odd superblock counts per column.
/// All-ones activation isolates the Q3_K weight arithmetic: with every
/// activation quant at 127 and `d8 = 1/127`, the kernel output times 127
/// must equal the host's element-weight sum exactly. Any misread region of
/// the superblock shows up as a clean integer gap.
#[test]
fn q3k_weight_sum_with_unit_activation() {
    let (k, n) = (256usize, 4usize);
    let a = vec![1.0f32; k];
    let mut sup = vec![0u8; 110];
    let mut st = 77u32 | 1;
    let mut rnd = || {
        st = st.wrapping_mul(747796405).wrapping_add(2891336453);
        let w = ((st >> ((st >> 28) + 4)) ^ st).wrapping_mul(277803737);
        (w >> 22) ^ w
    };
    for b in sup[..108].iter_mut() {
        *b = (rnd() % 256) as u8;
    }
    sup[108..110].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
    let mut packed = Vec::new();
    for _ in 0..n {
        packed.extend_from_slice(&sup);
    }

    // Host per-sub-block sums: sc_j * sum of the sub-block's quants, with
    // the quant = q2 - (hbit ? 0 : 4) and the hmask bit at e/32.
    let byte_at = |off: usize| -> u32 { u32::from(sup[off]) };
    let scale_at = |i: usize| -> f32 {
        let b = i % 4;
        let g = i / 4;
        let src = if g.is_multiple_of(2) { b } else { 4 + b };
        let raw = byte_at(96 + src);
        let nib = if g < 2 { raw & 0xF } else { raw >> 4 };
        let hi = (byte_at(96 + 8 + b) >> (g * 2)) & 3;
        (((nib as i32) | ((hi as i32) << 4)) - 32) as f32
    };
    let mut host_blocks = [0.0f32; 8];
    for e in 0..256usize {
        let j = (e % 128) / 32;
        let l = e % 32;
        let q2 = (byte_at(32 + (e / 128) * 32 + l) >> (j * 2)) & 3;
        let hb = (byte_at(l) >> (e / 32)) & 1;
        let quant = q2 as i32 - if hb == 1 { 0 } else { 4 };
        host_blocks[e / 32] += scale_at(e / 16) * quant as f32;
    }
    // Kernel simulation: the shader's integer form, per sub-block.
    let word_at =
        |off: usize| u32::from_le_bytes([sup[off], sup[off + 1], sup[off + 2], sup[off + 3]]);
    let mut kernel_blocks = [0.0f32; 8];
    for (j8, block) in kernel_blocks.iter_mut().enumerate() {
        let h = j8 / 4;
        let j3 = j8 % 4;
        let qshift = j3 * 2;
        let hshift = j8;
        let mut sum_a = 0i32;
        let mut sum_b = 0i32;
        for i in 0..4usize {
            let q_a = (word_at(32 + h * 32 + i * 4) >> qshift) & 0x03030303;
            let q_b = (word_at(32 + h * 32 + 16 + i * 4) >> qshift) & 0x03030303;
            let hb_a = (word_at(i * 4) >> hshift) & 0x01010101;
            let hb_b = (word_at(16 + i * 4) >> hshift) & 0x01010101;
            for byt in 0..4usize {
                let qa = ((q_a >> (byt * 8)) & 0xFF) as i32;
                let qb = ((q_b >> (byt * 8)) & 0xFF) as i32;
                let ha = ((hb_a >> (byt * 8)) & 1) as i32;
                let hbb = ((hb_b >> (byt * 8)) & 1) as i32;
                sum_a += qa - 4 * (1 - ha);
                sum_b += qb - 4 * (1 - hbb);
            }
        }
        *block = scale_at(j8 * 2) * sum_a as f32 + scale_at(j8 * 2 + 1) * sum_b as f32;
    }
    let got = run(Format::Q3K, k, n, &a, &packed, true, None);
    let host_sum: f32 = host_blocks.iter().sum();
    assert!(
        (got[0] - host_sum).abs() <= 1.0e-2,
        "kernel {:.6} vs host sum {:.6}",
        got[0],
        host_sum
    );
}

#[test]
fn int_dot_gemv_matches_the_q4_0_q8_1_reference() {
    for format in [
        Format::Q40,
        Format::Q8,
        Format::Q4K,
        Format::Q5K,
        Format::Q6K,
        Format::Q3K,
    ] {
        let shapes: &[(usize, usize)] = match format {
            Format::Q40 => &[
                (64usize, 8usize),
                (96, 4),
                (256, 16),
                (6144, 1536),
                (8224, 4),
            ],
            Format::Q8 => &[(64usize, 8usize), (96, 4), (256, 16), (8224, 4)],
            // The envelope only opens where an element sits on a rounding
            // boundary, and each such element contributes up to a full quant
            // step of its weight. At 8192 elements a Q4_K/Q5_K/Q6_K column
            // accumulates enough boundary hits for the envelope to exceed
            // the tightness check below, so the K-quants pin whole and odd
            // superblock counts at a size where the envelope stays tight.
            _ => &[(256usize, 8usize), (512, 4), (768, 4)],
        };
        for (k, n) in shapes {
            let (k, n) = (*k, *n);
            let (packed, _) = weight(format, k, n, (k * n) as u32);
            let a = activation(k);
            let (mut want, mut lo, mut hi) = int_dot_reference(format, &a, &packed, k, n);
            // One representative also exercises the optimizer's fused residual
            // route; keeping it in this table gives the same arithmetic oracle
            // without another near-duplicate test.
            let addend = (format == Format::Q40 && k == 96).then(|| {
                (0..n)
                    .map(|i| (i as f32 - n as f32 * 0.5) * 0.07)
                    .collect::<Vec<_>>()
            });
            if let Some(d) = &addend {
                for ((want, lo), (hi, d)) in want.iter_mut().zip(&mut lo).zip(hi.iter_mut().zip(d))
                {
                    *want += d;
                    *lo += d;
                    *hi += d;
                }
            }
            let got = run(format, k, n, &a, &packed, true, addend.as_deref());

            let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            assert!(scale > 0.05, "k={k} n={n}: reference is ~zero, {scale}");
            // Inside the envelope the quantizer permits, plus summation noise:
            // both sides accumulate the same integer products, so the only other
            // difference is the order of the f32 block sum.
            if let Some(why) = within_envelope(&got, &lo, &hi, scale * 2.0e-5) {
                panic!("{format:?} k={k} n={n}: {why}");
            }

            // The envelope must not be so wide that it stops meaning anything.
            // It only opens where an element sits on a rounding boundary, and
            // one quant step is small next to the result.
            let widest = lo
                .iter()
                .zip(&hi)
                .map(|(a, b)| b - a)
                .fold(0.0f32, f32::max);
            assert!(
                widest <= scale * 0.05,
                "{format:?} k={k} n={n}: the rounding envelope is {widest} against \
                 scale {scale}, which is too loose to catch a wrong kernel"
            );
        }
    }
}

/// Quantizing the activation must be a small perturbation of the f32 path,
/// not a different answer that merely matches a matching reference.
#[test]
fn int_dot_gemv_stays_close_to_the_full_precision_path() {
    for format in [
        Format::Q40,
        Format::Q8,
        Format::Q4K,
        Format::Q5K,
        Format::Q6K,
        Format::Q3K,
    ] {
        for (k, n) in if format == Format::Q6K {
            vec![(256usize, 4usize)]
        } else {
            vec![(512usize, 16usize)]
        } {
            let (packed, dequant) = weight(format, k, n, 9);
            let a = activation(k);

            let exact: Vec<f32> = (0..n)
                .map(|col| (0..k).map(|i| a[i] * dequant[i * n + col]).sum())
                .collect();
            let scale = exact.iter().fold(0.0f32, |m, v| m.max(v.abs()));

            let full = run(format, k, n, &a, &packed, false, None);
            let quantized = run(format, k, n, &a, &packed, true, None);

            let err = |v: &[f32]| max_abs_diff(v, &exact);
            let full_err = err(&full);
            let quantized_err = err(&quantized);
            eprintln!(
                "{format:?} k={k} n={n} scale={scale:.4} f32-path err={full_err:.6} \
             int-dot err={quantized_err:.6}"
            );

            // The f32 path decodes the same weights, so it should be exact bar
            // summation order.
            assert!(
                full_err <= scale * 1.0e-5,
                "{format:?}: the f32 path drifted from the weights: {full_err}"
            );
            // 8-bit activations over a 32-element block: half a step is 1/254 of
            // the block's peak, and errors across blocks partly cancel. Two
            // percent of full scale covers Q4_0, Q8 and the K-quants whose
            // sub-block scales stay modest. Q6_K quantizes its own sub-block
            // scales to signed int8 reaching ±39 while d stays tiny, so half an
            // activation step against one sub-block can be a large absolute
            // error; a dropped correction term there is still an order of
            // magnitude worse than the bound below.
            let tolerance = match format {
                Format::Q6K => 2.0e-1,
                _ => 2.0e-2,
            };
            assert!(
                quantized_err <= scale * tolerance,
                "{format:?}: int-dot drifted too far: {quantized_err} against scale {scale}"
            );
        }
    }
}

#[test]
fn q40_q8_1_rmsnorm_fusion_matches_the_unfused_path() {
    let (k, n) = (1536, 64);
    let x = (0..k)
        .map(|i| ((i * 37 % 257) as f32 - 128.0) * 0.013)
        .collect::<Vec<_>>();
    let norm_weight = (0..k)
        .map(|i| 0.5 + (i * 17 % 101) as f32 * 0.011)
        .collect::<Vec<_>>();
    let (packed, _) = q40_weight(k, n, 0x51a7);

    let fused = run_q40_rmsnorm(k, n, &x, &norm_weight, &packed, false);
    let unfused = run_q40_rmsnorm(k, n, &x, &norm_weight, &packed, true);
    let scale = unfused.iter().fold(0.0f32, |m, value| m.max(value.abs()));
    let difference = max_abs_diff(&fused, &unfused);
    assert!(
        difference <= scale * 0.01,
        "fused Q8_1 activation changed RMSNorm GEMV by {difference} at scale {scale}"
    );
}

#[test]
fn q6k_sub_block_correction_matches_the_reference() {
    let (k, n) = (256usize, 4usize);
    let a = activation(k);
    // For each of the sixteen 16-element sub-blocks of one superblock, zero
    // every scale but that sub-block's, which carries q6 = 17. The exact
    // product is -15 * the sub-block's activation sum, so a misread region
    // shows up immediately.
    for keep in 0..16usize {
        let half = keep / 8;
        let j = (keep % 8) / 2;
        let lbase = (keep % 2) * 16;
        let mut sup = vec![0u8; 210];
        let ql_byte = if j < 2 { 0x01 } else { 0x10 };
        let qlbase = half * 64 + (j & 1) * 32;
        for i in 0..16 {
            sup[qlbase + lbase + i] = ql_byte;
            sup[128 + half * 32 + lbase + i] = 1 << (j * 2);
        }
        sup[192 + keep] = 1;
        sup[208..210].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
        let mut packed = Vec::new();
        for _ in 0..n {
            packed.extend_from_slice(&sup);
        }
        let (want, lo, hi) = int_dot_reference(Format::Q6K, &a, &packed, k, n);
        let got = run(Format::Q6K, k, n, &a, &packed, true, None);
        // The envelope is exact: only this sub-block's elements carry
        // weight, and the quantization choices bound the rest. The noise
        // slack covers backend reassociation of the scale, as elsewhere.
        if let Some(why) = within_envelope(&got, &lo, &hi, want[0].abs() * 2.0e-5) {
            panic!("keep={keep}: {why} (nominal {})", want[0]);
        }
    }
}

#[test]
fn q3k_sub_block_correction_matches_the_reference() {
    let (k, n) = (256usize, 4usize);
    let a = activation(k);
    // Every sub-block live with a distinct scale: sc[i] = i + 1, encoded as
    // nib | hi<<4 = i + 33. Every element's two-bit quant reads 3 with the
    // hmask bit set, so element e carries weight 3 * (e/16 + 1) * d. The
    // kernel must reproduce the same scale attribution for all sixteen.
    let mut sup = vec![0u8; 110];
    for i in 0..16usize {
        let value = (i + 33) as u32;
        let nib = (value & 0xF) as u8;
        let hi = (value >> 4) as u8;
        let src = if (i / 4) % 2 == 0 { i % 4 } else { 4 + (i % 4) };
        let nib_pos = 96 + src;
        if i / 4 < 2 {
            sup[nib_pos] = (sup[nib_pos] & 0xF0) | nib;
        } else {
            sup[nib_pos] = (sup[nib_pos] & 0x0F) | (nib << 4);
        }
        sup[96 + 8 + (i % 4)] |= hi << ((i / 4) * 2);
        let qsbase = 32 + (i / 8) * 32;
        let lbase = (i % 2) * 16;
        for e in 0..16usize {
            // The qs and hmask bytes are shared across groups, so every
            // sub-block ORs its own bits in.
            sup[qsbase + lbase + e] |= 0x3u8 << (((i % 8) / 2) * 2);
            sup[lbase + e] |= (1 << (i / 2)) as u8;
        }
    }
    sup[108..110].copy_from_slice(&half::f16::from_f32(1.0).to_bits().to_le_bytes());
    let mut packed = Vec::new();
    for _ in 0..n {
        packed.extend_from_slice(&sup);
    }
    let (want, lo, hi) = int_dot_reference(Format::Q3K, &a, &packed, k, n);
    let got = run(Format::Q3K, k, n, &a, &packed, true, None);
    eprintln!("q3k all-live: got[0]={:.6} want[0]={:.6}", got[0], want[0]);
    for e in [0usize, 1, 16, 17, 32, 128, 144, 240] {
        eprintln!(
            "  e={e}: host={:.6}",
            element_weight(Format::Q3K, &sup, 0, e / 32, e)
        );
    }
    if let Some(why) = within_envelope(&got, &lo, &hi, want[0].abs() * 2.0e-5) {
        panic!("{why} (nominal {})", want[0]);
    }
}

// The GGML reference comes from the loader itself, which lives behind the
// `gguf` feature; CI runs the suite with --all-features.
#[cfg(feature = "gguf")]
#[test]
fn host_dequant_matches_ggml() {
    use meganeura::load::gguf::{GgmlType, GgufTensor};

    type SuperblockBuild = fn(u32) -> Vec<u8>;
    let cases: [(Format, GgmlType, SuperblockBuild); 4] = [
        (Format::Q4K, GgmlType::Q4K, q4k_superblock),
        (Format::Q5K, GgmlType::Q5K, q5k_superblock),
        (Format::Q6K, GgmlType::Q6K, q6k_superblock),
        (Format::Q3K, GgmlType::Q3K, q3k_superblock),
    ];
    for (format, ty, build) in cases {
        let sup = build(77);
        let reference = GgufTensor::new(vec![256, 1], ty, sup.clone())
            .to_f32()
            .unwrap();
        for e in [
            0usize, 1, 15, 16, 31, 32, 63, 64, 127, 128, 129, 191, 192, 223, 224, 255,
        ] {
            let mine = element_weight(format, &sup, 0, e / 32, e);
            let ggml = reference[e];
            assert!(
                (mine - ggml).abs() <= 1.0e-9,
                "{format:?}: element {e}: host {mine} vs ggml {ggml}"
            );
        }
    }
}
