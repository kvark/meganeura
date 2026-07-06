//! Sampling utilities for Magenta-RT autoregressive generation.
//!
//! - Classifier-free guidance (CFG): combine logits from the positive and
//!   negative (style-masked) batches into a guided distribution.
//! - Top-k temperature sampling: standard recipe used by t5x's
//!   `decoding.temperature_sample` (defaults `temperature=1.1`, `topk=40`).
//!
//! All operations are CPU-side host code; the LLM forward returns logits
//! `[batch=2, vocab]` per step which we feed through [`cfg_combine`] →
//! [`top_k_sample`] → next token.
//!
//! NOTE: PRNG is xorshift64 here, NOT JAX's Threefry. Outputs are stochastic
//! and won't bit-match the official Colab reference. For deterministic
//! comparison we should swap in greedy decoding (`temperature=0`-style argmax).

/// Tiny stateful xorshift64* PRNG — enough for sampling. Reproducible from
/// the seed, no external deps.
#[derive(Clone, Debug)]
pub struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    /// Construct from a u64 seed. Avoids the all-zero state by adding a constant.
    pub fn new(seed: u64) -> Self {
        Self { state: seed.wrapping_add(0x9E3779B97F4A7C15) }
    }
    /// Next u64.
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    /// Uniform f32 in [0, 1).
    pub fn next_f32(&mut self) -> f32 {
        // Top 24 bits → [0, 1) with exact representation.
        ((self.next_u64() >> 40) as f32) / ((1u32 << 24) as f32)
    }
}

/// Combine pos/neg logits via classifier-free guidance.
///
/// `pos` and `neg` are `[vocab]`-shaped logit slices for the positive (real
/// style) and negative (mask-style) batches. Returns the guided logits:
///
///   `out = neg + guidance_weight * (pos - neg)`
///
/// In Magenta-RT this is the per-token step within an autoregressive loop;
/// the caller is responsible for stacking pos/neg into batch=2 on the GPU side.
pub fn cfg_combine(pos: &[f32], neg: &[f32], guidance_weight: f32, out: &mut [f32]) {
    assert_eq!(pos.len(), neg.len());
    assert_eq!(pos.len(), out.len());
    for i in 0..pos.len() {
        out[i] = neg[i] + guidance_weight * (pos[i] - neg[i]);
    }
}

/// Top-k temperature sampling. Returns one token index sampled from the
/// top-`k` logits after softmax with temperature.
///
/// Matches the semantics of t5x's `decoding.temperature_sample`:
/// - Scale logits by `1 / temperature` (higher T → flatter distribution).
/// - Keep only the top-k logits (others get `-inf`).
/// - Apply softmax → categorical sample.
pub fn top_k_sample(logits: &[f32], temperature: f32, k: usize, rng: &mut Xorshift64) -> u32 {
    assert!(temperature > 0.0, "temperature must be > 0");
    let k = k.max(1).min(logits.len());

    // Partial sort by logit (descending). select_nth_unstable_by partitions
    // so that idx[0..k] contains the k largest (in arbitrary order).
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.select_nth_unstable_by(k - 1, |&a, &b| {
        logits[b].partial_cmp(&logits[a]).unwrap_or(std::cmp::Ordering::Equal)
    });
    let topk_idx = &idx[..k];

    // Scaled logits → softmax (numerically stable).
    let inv_t = 1.0 / temperature;
    let max_logit = topk_idx
        .iter()
        .map(|&i| logits[i] * inv_t)
        .fold(f32::NEG_INFINITY, f32::max);
    let mut weights: Vec<f32> = topk_idx
        .iter()
        .map(|&i| ((logits[i] * inv_t) - max_logit).exp())
        .collect();
    let sum: f32 = weights.iter().sum();
    for w in &mut weights {
        *w /= sum;
    }

    // Categorical sample.
    let u = rng.next_f32();
    let mut cum = 0.0;
    for (j, &p) in weights.iter().enumerate() {
        cum += p;
        if u <= cum {
            return topk_idx[j] as u32;
        }
    }
    *topk_idx.last().unwrap() as u32
}

/// Greedy argmax — deterministic, useful for bit-matching reference runs.
/// Ties break to the **first** (lowest) index, matching `np.argmax`/`jnp.argmax`
/// (`Iterator::max_by` would keep the last).
pub fn argmax(logits: &[f32]) -> u32 {
    let mut best = 0usize;
    let mut best_v = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v > best_v {
            best_v = v;
            best = i;
        }
    }
    best as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn xorshift_is_deterministic_from_seed() {
        let mut a = Xorshift64::new(42);
        let mut b = Xorshift64::new(42);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn cfg_combine_blends_correctly() {
        let pos = [1.0, 2.0, 3.0];
        let neg = [0.5, 1.0, 1.5];
        let mut out = [0.0; 3];
        cfg_combine(&pos, &neg, 4.0, &mut out);
        // out = neg + 4*(pos-neg) = [0.5+2, 1+4, 1.5+6]
        assert_eq!(out, [2.5, 5.0, 7.5]);
    }

    #[test]
    fn top_k_sample_picks_from_top_k_only() {
        let mut logits = vec![0.0f32; 10];
        logits[5] = 10.0;
        logits[7] = 9.0;
        let mut rng = Xorshift64::new(42);
        for _ in 0..200 {
            let t = top_k_sample(&logits, 1.0, 2, &mut rng);
            assert!(t == 5 || t == 7, "top-k=2 returned {t}");
        }
    }

    #[test]
    fn top_k_sample_low_temperature_concentrates_on_argmax() {
        let logits = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut rng = Xorshift64::new(0);
        let mut argmax_hits = 0;
        for _ in 0..100 {
            if top_k_sample(&logits, 0.01, 4, &mut rng) == 3 {
                argmax_hits += 1;
            }
        }
        assert!(argmax_hits > 90, "low-T should concentrate on argmax, hits={argmax_hits}");
    }

    #[test]
    fn argmax_picks_largest() {
        assert_eq!(argmax(&[1.0, 5.0, 3.0, 2.0]), 1);
        assert_eq!(argmax(&[]), 0);
    }
}
