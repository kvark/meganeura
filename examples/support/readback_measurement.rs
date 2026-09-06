//! Fixed readback experiment accounting, shared by the runner and CPU replay.
use meganeura::{TuneDecision, TuneReport};
use std::collections::BTreeMap;

pub const PROCESSES: usize = 6;
pub const PREFIX: usize = 3;
pub const MIDDLE: usize = 33;
pub const FINAL: usize = 178;

pub fn search_order(seed: usize, case: usize) -> [usize; 2] {
    let first = (seed + case) % 2;
    [first, 1 - first]
}

pub fn costs(report: &TuneReport) -> Result<BTreeMap<String, f64>, &'static str> {
    if report.outcomes.is_empty() || report.time_budget_exhausted || report.class_limit_reached {
        return Err("incomplete search");
    }
    let mut times = BTreeMap::<String, f64>::new();
    times.insert("elapsed".into(), report.elapsed.as_secs_f64() * 1e3);
    for outcome in &report.outcomes {
        if !outcome.qualified
            || outcome.failure.is_some()
            || !matches!(
                outcome.decision,
                TuneDecision::FasterCandidate | TuneDecision::KeepBaseline
            )
        {
            return Err("unqualified or incomplete comparison");
        }
        let phases = outcome.phase_times.ok_or("missing phases")?;
        let detail = phases
            .qualification_breakdown
            .ok_or("missing qualification breakdown")?;
        for (name, duration) in [
            ("preparation", phases.preparation),
            ("qualification", phases.qualification),
            ("warmup", phases.warmup),
            ("sampling", phases.sampling),
            ("compile", Some(outcome.compile_time)),
            ("input_preparation", detail.input_preparation),
            ("upload_host_copy", detail.upload_host_copy),
            ("upload_transfer", detail.upload_transfer),
            ("dispatch", detail.dispatch),
            ("readback_transfer", detail.readback_transfer),
            ("readback_host_copy", detail.readback_host_copy),
            ("validation", detail.validation),
        ] {
            *times.entry(name.into()).or_default() +=
                duration.ok_or("unreached phase")?.as_secs_f64() * 1e3;
        }
    }
    let nested: f64 = [
        "input_preparation",
        "upload_host_copy",
        "upload_transfer",
        "dispatch",
        "readback_transfer",
        "readback_host_copy",
        "validation",
    ]
    .iter()
    .map(|key| times[*key])
    .sum();
    let remainder = times["qualification"] - nested;
    if remainder < 0.0 {
        return Err("nested times exceed qualification");
    }
    times.insert("qualification_other".into(), remainder);
    Ok(times)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_order_is_balanced_and_missing_phases_are_not_zero_cost() {
        for case in 0..3 {
            assert_eq!(
                (1..=PROCESSES)
                    .filter(|seed| search_order(*seed, case)[0] == 0)
                    .count(),
                3
            );
            for seed in 1..=PROCESSES {
                let order = search_order(seed, case);
                assert_ne!(order[0], order[1]);
            }
        }
        assert!(costs(&TuneReport::default()).is_err());
    }
}
