//! Allocation/cleanup accounting added after the retained readback cohort.
use super::readback;
use meganeura::TuneReport;
use std::collections::BTreeMap;

pub fn costs(report: &TuneReport) -> Result<BTreeMap<String, f64>, &'static str> {
    let mut times = readback::costs(report)?;
    for outcome in &report.outcomes {
        let phases = outcome.phase_times.ok_or("missing phases")?;
        let prep = phases
            .preparation_breakdown
            .ok_or("missing preparation breakdown")?;
        for (key, duration) in [
            ("prep_checks", prep.checks),
            ("prep_pipelines", prep.pipelines),
            ("prep_buffers", prep.buffers),
            ("prep_staging", prep.staging),
            ("prep_encoder", prep.encoder),
            ("prep_bindings", prep.bindings),
            ("cleanup", phases.cleanup),
        ] {
            *times.entry(key.into()).or_default() +=
                duration.ok_or("unreached phase")?.as_secs_f64() * 1e3;
        }
    }
    let nested: f64 = [
        "prep_checks",
        "prep_pipelines",
        "prep_buffers",
        "prep_staging",
        "prep_encoder",
        "prep_bindings",
    ]
    .iter()
    .map(|key| times[*key])
    .sum();
    let remainder = times["preparation"] - nested;
    if remainder < 0.0 {
        return Err("nested times exceed preparation");
    }
    times.insert("preparation_other".into(), remainder);
    // Fresh has no retained resource to release. Reuse records the final release
    // separately, within total search, not within a comparison's cleanup.
    *times.get_mut("cleanup").unwrap() +=
        report.final_cleanup.unwrap_or_default().as_secs_f64() * 1e3;
    times.insert(
        "staging_and_cleanup".into(),
        times["prep_staging"] + times["cleanup"],
    );
    Ok(times)
}
