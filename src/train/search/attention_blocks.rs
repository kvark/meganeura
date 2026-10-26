//! Local key-block neighbors of a qualified QK/PV program.
use super::{compile, measure};
use compile::{Dispatch, ExecutionPlan, Kernel, ShaderEntry};
use std::collections::{BTreeMap, HashSet, VecDeque};

type Policy = Vec<(Vec<u32>, u32)>;

pub(super) struct Refinements {
    shared_bytes: u32,
    source: Option<(measure::Origin, ExecutionPlan)>,
    pending: VecDeque<(Vec<u32>, u32)>,
    tried: HashSet<(measure::Origin, Policy)>,
}

fn eligible(d: &Dispatch) -> bool {
    d.shader == ShaderEntry::FlashAttentionProducts
        && !d.full_precision()
        && !d.schedule_locked
        && matches!(d.kernel, Kernel::Default | Kernel::AttentionProducts { .. })
}

fn keys(d: &Dispatch) -> u32 {
    match d.kernel {
        Kernel::AttentionProducts { keys } => keys,
        _ => 16,
    }
}

fn policy(plan: &ExecutionPlan) -> Policy {
    plan.dispatches
        .iter()
        .filter(|d| eligible(d))
        .map(|d| (d.params.clone(), keys(d)))
        .collect::<BTreeMap<_, _>>()
        .into_iter()
        .collect()
}

impl Refinements {
    pub(super) fn new(shared_bytes: u32) -> Self {
        Self {
            shared_bytes,
            source: None,
            pending: VecDeque::new(),
            tried: HashSet::new(),
        }
    }

    pub(super) fn selected(&mut self, origin: Option<measure::Origin>, plan: &ExecutionPlan) {
        self.pending.clear();
        self.source = None;
        // Changing the softmax partition changes rounding. Training admission
        // needs its own trajectory checks, beyond the inference experiments.
        if !plan.param_grad_pairs.is_empty() || plan.dispatches.iter().any(|d| d.is_derivative) {
            return;
        }
        let Some(origin) = origin else { return };
        let current = policy(plan);
        self.tried.insert((origin, current.clone()));
        for (params, incumbent) in current {
            let Some(&head_dim) = params.get(3) else {
                continue;
            };
            for keys in [16, 32, 64] {
                if keys != incumbent
                    && crate::codegen::attention_product_shared_bytes(head_dim, keys)
                        <= u64::from(self.shared_bytes)
                {
                    self.pending.push_back((params.clone(), keys));
                }
            }
        }
        if !self.pending.is_empty() {
            self.source = Some((origin, plan.clone()));
        }
    }

    pub(super) fn next(&mut self) -> Option<measure::Program> {
        let selected = self.source.as_ref()?;
        let origin = selected.0;
        let source = &selected.1;
        while let Some((class, key_tile)) = self.pending.pop_front() {
            let mut plan = source.clone();
            for d in plan
                .dispatches
                .iter_mut()
                .filter(|d| eligible(d) && d.params == class)
            {
                d.kernel = if key_tile == 16 {
                    Kernel::Default
                } else {
                    Kernel::AttentionProducts { keys: key_tile }
                };
            }
            if !self.tried.insert((origin, policy(&plan))) {
                continue;
            }
            return Some(measure::Program {
                prepared: true,
                retune: false,
                description: format!(
                    "refine={origin:?}, attention_keys={key_tile}, class={class:?}"
                ),
                plan,
                submission_chunks: None,
                origin: Some(origin),
            });
        }
        None
    }

    /// Carry only matching classes into a new graph/arithmetic proposal.
    pub(super) fn apply(&self, plan: &mut ExecutionPlan) {
        let Some(source) = self.source.as_ref().map(|entry| &entry.1) else {
            return;
        };
        let selected: BTreeMap<_, _> = policy(source).into_iter().collect();
        for d in plan.dispatches.iter_mut().filter(|d| eligible(d)) {
            if let Some(&key_tile) = selected.get(&d.params) {
                d.kernel = if key_tile == 16 {
                    Kernel::Default
                } else {
                    Kernel::AttentionProducts { keys: key_tile }
                };
            }
        }
    }

    pub(super) fn may_remain(&self) -> bool {
        !self.pending.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plan() -> ExecutionPlan {
        let mut plan = compile::compile(&crate::Graph::new());
        for q in [128, 128, 64] {
            plan.dispatches.push(Dispatch {
                shader: ShaderEntry::FlashAttentionProducts,
                params: vec![q, q, (4 << 16) | 2, 64, 0, 0, 0, 0],
                ..Default::default()
            });
        }
        plan
    }

    #[test]
    fn class_neighbors_keep_other_choices_and_accumulate_winners() {
        let mut refinements = Refinements::new(49152);
        let origin = Some(measure::Origin { form: 0, choice: 0 });
        let original = plan();
        refinements.selected(origin, &original);
        let first = refinements.next().unwrap();
        assert_eq!(first.plan.dispatches[0], original.dispatches[0]);
        assert_eq!(keys(&first.plan.dispatches[2]), 32);
        assert_eq!(first.plan.buffers, original.buffers);
        let restored: ExecutionPlan =
            serde_json::from_str(&serde_json::to_string(&first.plan).unwrap()).unwrap();
        assert_eq!(first.plan, restored);
        refinements.selected(origin, &first.plan);
        let second = refinements.next().unwrap();
        assert_eq!(keys(&second.plan.dispatches[2]), 64);
        // Reject that candidate; the next changes the other class while
        // retaining the accepted first class's 32-key choice.
        let combined = refinements.next().unwrap();
        assert_eq!(keys(&combined.plan.dispatches[2]), 32);
        assert_eq!(keys(&combined.plan.dispatches[0]), 32);
        assert_eq!(
            combined.plan.dispatches[0].kernel,
            combined.plan.dispatches[1].kernel
        );
        let mut stripped = combined.plan.clone();
        for d in &mut stripped.dispatches {
            d.kernel = Kernel::Default;
        }
        assert_eq!(stripped, original);
        let mut another_layout = original.clone();
        another_layout.dispatches[0].workgroups = [7, 2, 1];
        refinements.apply(&mut another_layout);
        assert_eq!(another_layout.dispatches[0].workgroups, [7, 2, 1]);
        assert_eq!(keys(&another_layout.dispatches[2]), 32);
    }

    #[test]
    fn pins_training_and_shared_memory_limit_remain_constraints() {
        let origin = Some(measure::Origin { form: 0, choice: 0 });
        for pin in 0..3 {
            let mut p = plan();
            for d in &mut p.dispatches {
                match pin {
                    0 => d.requires_full_precision = true,
                    1 => d.schedule_locked = true,
                    _ => d.is_derivative = true,
                }
            }
            let mut r = Refinements::new(49152);
            r.selected(origin, &p);
            assert!(r.next().is_none());
        }
        let mut p = plan();
        for d in &mut p.dispatches {
            d.params[3] = 128;
        }
        let mut r = Refinements::new(49152);
        r.selected(origin, &p);
        assert!(
            std::iter::from_fn(|| r.next()).all(|p| p
                .plan
                .dispatches
                .iter()
                .all(|d| keys(d) != 64))
        );
        let mut r = Refinements::new(16000);
        r.selected(origin, &plan());
        assert!(r.next().is_none());
    }
}
