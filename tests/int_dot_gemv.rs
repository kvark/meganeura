//! The Q8_1-activation, integer-dot GEMV against GGML Q4_0 weights.
//!
//! This is the one kernel choice in the crate that changes the numbers: the
//! activation is quantized to 8 bits on top of an already-quantized weight.
//! So it cannot be checked the way every other kernel is, against the f32
//! result — it is not computing that. It is checked against a CPU model of
//! the arithmetic it is *supposed* to do, `vec_dot_q4_0_q8_1`, which pins
//! the layout handling, the -8 bias and the `s8` correction term exactly.
//!
//! A second test bounds the error against the f32 path, so that the
//! quantization is shown to be a small perturbation rather than a wrong
//! answer that happens to agree with a wrong reference.

use meganeura::compile::{CompileOptions, ShaderEntry};
use meganeura::train::{Mode, SessionConfig};
use meganeura::{Graph, Session};

const BLOCK: usize = 32;
const BLOCK_BYTES: usize = 18;

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

/// A `[k, n]` Q4_0 weight, column-major in blocks the way packing lays it
/// out: all of column 0's blocks, then column 1's.
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

/// `vec_dot_q4_0_q8_1` on the host, as an envelope rather than one number.
///
/// Returns the nominal result and the range the kernel is allowed to land in.
/// Quantization is a rounding decision, and an activation whose scaled value
/// sits within [`REASSOCIATION_SLACK`] of a half-integer may legitimately
/// round either way depending on how the backend associates the scaling.
/// Every element contributes additively, so summing each one's two possible
/// contributions gives an exact bound rather than a guessed tolerance: the
/// envelope stays tight wherever no element is near a boundary.
fn int_dot_reference(
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

            let base = (col * blocks + blk) * BLOCK_BYTES;
            let d4 =
                half::f16::from_bits(u16::from_le_bytes([packed[base], packed[base + 1]])).to_f32();
            // Element e's contribution is `d4 * (q4 - 8) * d8 * q8`, which is
            // additive, so the block's own extremes are the sum of each
            // element's. `s8` is folded in here as the -8 term rather than
            // separately, which is the same arithmetic the kernel does.
            let nibble = |e: usize| -> i32 {
                let byte = packed[base + 2 + (e % 16)];
                i32::from(if e < 16 { byte & 0x0F } else { byte >> 4 })
            };
            for e in 0..BLOCK {
                let weight = d4 * (nibble(e) - 8) as f32 * d8;
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
    k: usize,
    n: usize,
    a: &[f32],
    packed: &[u8],
    quantized_activations: bool,
    addend: Option<&[f32]>,
) -> Vec<f32> {
    let mut g = Graph::new();
    let x = g.input("x", &[1, k]);
    let w = g.parameter_q40("w", &[k, n]);
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
    assert_eq!(dispatch.gemv_int_dot, quantized_activations);
    session.set_input("x", a);
    if let Some(d) = addend {
        session.set_input("d", d);
    }
    session.set_parameter_packed("w", packed);
    session.step();
    session.wait();
    session.read_output(n)
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
/// Shapes cover an even and an odd block count, because an 18-byte block is
/// not a whole word: the nibble reads are word-aligned for odd blocks and
/// straddle two words for even ones, and only an odd block count pads the
/// buffer at all.
#[test]
fn int_dot_gemv_matches_the_q4_0_q8_1_reference() {
    for (k, n) in [(64usize, 8usize), (96, 4), (256, 16), (8224, 4)] {
        let (packed, _) = q40_weight(k, n, (k * n) as u32);
        let a = activation(k);
        let (mut want, mut lo, mut hi) = int_dot_reference(&a, &packed, k, n);
        // One representative also exercises the optimizer's fused residual
        // route; keeping it in this table gives the same arithmetic oracle
        // without another near-duplicate test.
        let addend = (k == 96).then(|| {
            (0..n)
                .map(|i| (i as f32 - n as f32 * 0.5) * 0.07)
                .collect::<Vec<_>>()
        });
        if let Some(d) = &addend {
            for ((want, lo), (hi, d)) in want.iter_mut().zip(&mut lo).zip(hi.iter_mut().zip(d)) {
                *want += d;
                *lo += d;
                *hi += d;
            }
        }
        let got = run(k, n, &a, &packed, true, addend.as_deref());

        let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(scale > 0.05, "k={k} n={n}: reference is ~zero, {scale}");
        // Inside the envelope the quantizer permits, plus summation noise:
        // both sides accumulate the same integer products, so the only other
        // difference is the order of the f32 block sum.
        if let Some(why) = within_envelope(&got, &lo, &hi, scale * 2.0e-5) {
            panic!("k={k} n={n}: {why}");
        }

        // The envelope must not be so wide that it stops meaning anything.
        // It only opens where an element sits on a rounding boundary, and one
        // quant step is worth `d4 * 8 * d8` — small next to the result.
        let widest = lo
            .iter()
            .zip(&hi)
            .map(|(a, b)| b - a)
            .fold(0.0f32, f32::max);
        assert!(
            widest <= scale * 0.05,
            "k={k} n={n}: the rounding envelope is {widest} against scale {scale}, \
             which is too loose to catch a wrong kernel"
        );
    }
}

/// Quantizing the activation must be a small perturbation of the f32 path,
/// not a different answer that merely matches a matching reference.
#[test]
fn int_dot_gemv_stays_close_to_the_full_precision_path() {
    let (k, n) = (512usize, 16usize);
    let (packed, weight) = q40_weight(k, n, 9);
    let a = activation(k);

    let exact: Vec<f32> = (0..n)
        .map(|col| (0..k).map(|i| a[i] * weight[i * n + col]).sum())
        .collect();
    let scale = exact.iter().fold(0.0f32, |m, v| m.max(v.abs()));

    let full = run(k, n, &a, &packed, false, None);
    let quantized = run(k, n, &a, &packed, true, None);

    let err = |v: &[f32]| max_abs_diff(v, &exact);
    let full_err = err(&full);
    let quantized_err = err(&quantized);
    eprintln!(
        "k={k} n={n} scale={scale:.4} f32-path err={full_err:.6} int-dot err={quantized_err:.6}"
    );

    // The f32 path decodes the same weights, so it should be exact bar
    // summation order.
    assert!(
        full_err <= scale * 1.0e-5,
        "the f32 path drifted from the weights: {full_err}"
    );
    // 8-bit activations over a 32-element block: half a step is 1/254 of the
    // block's peak, and errors across blocks partly cancel. Two percent of
    // full scale is loose enough not to be flaky and tight enough that a
    // dropped correction term — `s8` is worth far more than this — fails.
    assert!(
        quantized_err <= scale * 2.0e-2,
        "int-dot drifted too far: {quantized_err} against scale {scale}"
    );
}
