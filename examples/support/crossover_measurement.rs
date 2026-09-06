//! Descriptive crossover gates. These are not confidence intervals.

use super::measurement::PairedTiming;
use serde::{Deserialize, Serialize};

pub const PREFIX: usize = 3;
pub const WARMUP: usize = 30;
pub const SETTLING: usize = 5;
pub const CONTROL_PAIRS: usize = 40;
pub const BLOCK_PAIRS: usize = 20;
pub const BLOCKS: usize = 4;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Pair {
    pub step: usize,
    pub first_session: usize,
    pub left_ms: f64,
    pub right_ms: f64,
    pub left_loss: Option<f32>,
    pub right_loss: Option<f32>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Block {
    pub winner_session: usize,
    pub pairs: Vec<Pair>,
}

pub fn winner_order(first: usize) -> [usize; BLOCKS] {
    assert!(first < 2);
    [first, 1 - first, 1 - first, first]
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Confirmation {
    pub control: PairedTiming,
    pub control_stable: bool,
    pub left_winner: PairedTiming,
    pub right_winner: PairedTiming,
    pub pooled: PairedTiming,
    pub decision: String,
}

impl Confirmation {
    pub fn new(
        control: &[Pair],
        blocks: &[Block],
        changed: usize,
        valid: bool,
    ) -> Result<Self, &'static str> {
        if control.len() != CONTROL_PAIRS || blocks.len() != BLOCKS {
            return Err("incomplete control or crossover blocks");
        }
        let first = blocks[0].winner_session;
        if first > 1
            || blocks.iter().map(|b| b.winner_session).collect::<Vec<_>>() != winner_order(first)
        {
            return Err("roles must follow the declared counterbalanced order");
        }
        let control = PairedTiming::new(
            &control.iter().map(|p| p.left_ms).collect::<Vec<_>>(),
            &control.iter().map(|p| p.right_ms).collect::<Vec<_>>(),
        )?;
        let limit = 0.05 * control.baseline_median_ms;
        let control_stable = (control.baseline_median_ms - control.tuned_median_ms).abs() <= limit
            && control.paired_gain_median_ms.abs() <= limit
            && control.paired_noise_margin_ms <= limit;
        let mut baseline = [Vec::new(), Vec::new()];
        let mut tuned = [Vec::new(), Vec::new()];
        let (mut all_baseline, mut all_tuned) = (Vec::new(), Vec::new());
        for block in blocks {
            if block.pairs.len() != BLOCK_PAIRS {
                return Err("incomplete crossover pair block");
            }
            for pair in &block.pairs {
                let (a, b) = if block.winner_session == 0 {
                    (pair.right_ms, pair.left_ms)
                } else {
                    (pair.left_ms, pair.right_ms)
                };
                baseline[block.winner_session].push(a);
                tuned[block.winner_session].push(b);
                all_baseline.push(a);
                all_tuned.push(b);
            }
        }
        let left_winner = PairedTiming::new(&baseline[0], &tuned[0])?;
        let right_winner = PairedTiming::new(&baseline[1], &tuned[1])?;
        let pooled = PairedTiming::new(&all_baseline, &all_tuned)?;
        let guards = [&left_winner, &right_winner, &pooled];
        let decision = if !valid {
            "numerical_failure"
        } else if changed == 0 {
            "unchanged_selection"
        } else if !control_stable {
            "unstable_control"
        } else if guards.iter().all(|g| g.improvement_exceeds_guard) {
            "confirmed_gain"
        } else if guards.iter().all(|g| g.regression_exceeds_guard) {
            "confirmed_regression"
        } else {
            "inconclusive"
        };
        Ok(Self {
            control,
            control_stable,
            left_winner,
            right_winner,
            pooled,
            decision: decision.into(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair(left_ms: f64, right_ms: f64) -> Pair {
        Pair {
            step: 0,
            first_session: 0,
            left_ms,
            right_ms,
            left_loss: None,
            right_loss: None,
        }
    }

    #[test]
    fn crossover_requires_both_roles_stable_controls_changes_and_numerics() {
        let control = vec![pair(10.0, 10.0); CONTROL_PAIRS];
        let mut blocks: Vec<_> = winner_order(0)
            .into_iter()
            .map(|winner_session| Block {
                winner_session,
                pairs: vec![
                    if winner_session == 0 {
                        pair(8.0, 10.0)
                    } else {
                        pair(10.0, 8.0)
                    };
                    BLOCK_PAIRS
                ],
            })
            .collect();
        assert_eq!(
            Confirmation::new(&control, &blocks, 1, true)
                .unwrap()
                .decision,
            "confirmed_gain"
        );
        assert_eq!(
            Confirmation::new(&control, &blocks, 0, true)
                .unwrap()
                .decision,
            "unchanged_selection"
        );
        assert_eq!(
            Confirmation::new(&control, &blocks, 1, false)
                .unwrap()
                .decision,
            "numerical_failure"
        );
        let biased = vec![pair(10.0, 8.0); CONTROL_PAIRS];
        assert_eq!(
            Confirmation::new(&biased, &blocks, 1, true)
                .unwrap()
                .decision,
            "unstable_control"
        );
        let noisy: Vec<_> = (0..CONTROL_PAIRS)
            .map(|i| pair(10.0, if i % 2 == 0 { 9.0 } else { 11.0 }))
            .collect();
        assert_eq!(
            Confirmation::new(&noisy, &blocks, 1, true)
                .unwrap()
                .decision,
            "unstable_control"
        );
        let mut regression = blocks.clone();
        for block in &mut regression {
            for pair in &mut block.pairs {
                std::mem::swap(&mut pair.left_ms, &mut pair.right_ms);
            }
        }
        assert_eq!(
            Confirmation::new(&control, &regression, 1, true)
                .unwrap()
                .decision,
            "confirmed_regression"
        );
        for block in &mut blocks {
            block.pairs = vec![pair(8.0, 10.0); BLOCK_PAIRS];
        }
        assert_eq!(
            Confirmation::new(&control, &blocks, 1, true)
                .unwrap()
                .decision,
            "inconclusive"
        );
        assert!(Confirmation::new(&control[..39], &blocks, 1, true).is_err());
        blocks[0].winner_session = 5;
        assert!(Confirmation::new(&control, &blocks, 1, true).is_err());
    }
}
