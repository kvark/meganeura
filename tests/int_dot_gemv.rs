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

use meganeura::compile::CompileOptions;
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

/// `vec_dot_q4_0_q8_1` on the host: quantize the activation block to Q8_1,
/// take the integer dot against the raw nibbles, then correct for the -8
/// bias with `s8`.
fn int_dot_reference(a: &[f32], packed: &[u8], k: usize, n: usize) -> Vec<f32> {
    let blocks = k / BLOCK;
    let mut out = vec![0.0f32; n];
    for (col, slot) in out.iter_mut().enumerate() {
        let mut acc = 0.0f32;
        for blk in 0..blocks {
            let chunk = &a[blk * BLOCK..(blk + 1) * BLOCK];
            let amax = chunk.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let d8 = amax / 127.0;
            let inv = if amax > 0.0 { 1.0 / d8 } else { 0.0 };
            let q8: Vec<i32> = chunk
                .iter()
                .map(|v| ((v * inv).round_ties_even() as i32).clamp(-127, 127))
                .collect();
            let s8 = d8 * q8.iter().sum::<i32>() as f32;

            let base = (col * blocks + blk) * BLOCK_BYTES;
            let d4 =
                half::f16::from_bits(u16::from_le_bytes([packed[base], packed[base + 1]])).to_f32();
            let mut sumi = 0i32;
            for j in 0..16 {
                let byte = packed[base + 2 + j];
                sumi += i32::from(byte & 0x0F) * q8[j];
                sumi += i32::from(byte >> 4) * q8[j + 16];
            }
            acc += d4 * (d8 * sumi as f32 - 8.0 * s8);
        }
        *slot = acc;
    }
    out
}

fn run(k: usize, n: usize, a: &[f32], packed: &[u8], quantized_activations: bool) -> Vec<f32> {
    let mut g = Graph::new();
    let x = g.input("x", &[1, k]);
    let w = g.parameter_q40("w", &[k, n]);
    let y = g.matmul(x, w);
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
    session.set_input("x", a);
    session.set_parameter_packed("w", packed);
    session.step();
    session.wait();
    session.read_output(n)
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
    for (k, n) in [(64usize, 8usize), (96, 4), (256, 16), (576, 12)] {
        let (packed, _) = q40_weight(k, n, (k * n) as u32);
        let a = activation(k);
        let want = int_dot_reference(&a, &packed, k, n);
        let got = run(k, n, &a, &packed, true);

        let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        assert!(scale > 0.05, "k={k} n={n}: reference is ~zero, {scale}");
        let worst = got
            .iter()
            .zip(&want)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        // Both sides accumulate the same integer products; only the f32
        // block sum differs in order, so this is float-noise tight.
        assert!(
            worst <= scale * 2.0e-5,
            "k={k} n={n}: max_abs_err={worst}, scale={scale}"
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

    let full = run(k, n, &a, &packed, false);
    let quantized = run(k, n, &a, &packed, true);

    let err = |v: &[f32]| {
        v.iter()
            .zip(&exact)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
    };
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

/// Measuring a shape must not swap the kernel underneath the plan.
///
/// The shape axis is meant to be free to pick among routes to the same
/// answer. This kernel is not one of those routes — it computes something
/// else — so a shape candidate has to be generated inside it. Generating the
/// ordinary GEMV for the same dispatch would quietly put the activation back
/// in f32 and change what the session computes, which is the kind of thing
/// that would show up as an unexplained accuracy change much later.
#[test]
fn tuning_a_shape_keeps_the_int_dot_kernel() {
    use meganeura::compile::ShaderEntry;
    use meganeura::{MatmulTile, TuneOptions};

    let (k, n) = (256usize, 16usize);
    let (packed, _) = q40_weight(k, n, 5);
    let a = activation(k);
    let want = int_dot_reference(&a, &packed, k, n);

    let mut g = Graph::new();
    let x = g.input("x", &[1, k]);
    let w = g.parameter_q40("w", &[k, n]);
    let y = g.matmul(x, w);
    g.set_outputs(vec![y]);
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
    assert!(
        session
            .plan()
            .dispatches
            .iter()
            .any(|d| d.shader == ShaderEntry::MatMulGemv && d.gemv_int_dot),
        "the plan did not route through the int-dot GEMV"
    );

    let report = session
        .tune_with(TuneOptions {
            max_time: std::time::Duration::from_secs(120),
            ..Default::default()
        })
        .unwrap();
    assert!(!report.outcomes.is_empty(), "no shape was challenged");
    for outcome in &report.outcomes {
        assert!(matches!(outcome.candidate, MatmulTile::Gemv(_)));
        assert!(
            outcome.qualified,
            "a shape failed to qualify: {:?}",
            outcome.failure
        );
    }

    // After tuning, the session must still compute the quantized product.
    session.set_input("x", &a);
    session.set_parameter_packed("w", &packed);
    session.step();
    session.wait();
    let got = session.read_output(n);
    let scale = want.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    let worst = got
        .iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    assert!(
        worst <= scale * 2.0e-5,
        "tuning changed the computation: max_abs_err={worst}, scale={scale}"
    );
}
