//! CPU replay of allocation localization and the separately retained reuse cohort.
#[path = "support/staging_evidence.rs"]
mod diagnostic;
#[path = "support/tuning_evidence.rs"]
mod evidence;
#[path = "../examples/support/tuning_measurement.rs"]
mod measurement;
#[path = "../examples/support/preparation_measurement.rs"]
mod preparation;
#[path = "../examples/support/readback_measurement.rs"]
mod readback;
#[path = "support/telemetry.rs"]
mod telemetry;

use meganeura::{TuneReport, TuneStaging, TuneStagingReuse};
use serde_json::Value;
use std::time::Duration;

const PROFILE: &str = include_str!("../docs/experiments/staging-reuse-2026-09-06/profile-01.json");
const PROFILE_PROTOCOL: diagnostic::Protocol = diagnostic::Protocol {
    name: "readback-2026-09-06",
    revision: "42fda56d568293e82b3bc92defe57025bf78c800",
    executable: "e1f4a4c36c314195173a3448ca3fb416616f594056006a2c26b49a98bdfd82cd",
    labels: ["shared", "download"],
    staging: [TuneStaging::Shared, TuneStaging::Download],
    reuse: [TuneStagingReuse::Fresh; 2],
    costs: readback::costs,
};

fn check_preparation(report: &TuneReport) {
    for outcome in &report.outcomes {
        let phases = outcome.phase_times.unwrap();
        let prep = phases.preparation_breakdown.unwrap();
        let parts = [
            prep.checks,
            prep.pipelines,
            prep.buffers,
            prep.staging,
            prep.encoder,
            prep.bindings,
        ];
        assert!(parts.iter().all(|p| p.is_some_and(|p| !p.is_zero())));
        let total: Duration = parts.into_iter().map(Option::unwrap).sum();
        assert_eq!(outcome.compile_time, prep.pipelines.unwrap());
        assert!(total <= phases.preparation.unwrap());
        assert!(!phases.cleanup.unwrap().is_zero());
        let total: Duration = [
            phases.preparation,
            phases.qualification,
            phases.warmup,
            phases.sampling,
            phases.cleanup,
        ]
        .into_iter()
        .map(Option::unwrap)
        .sum();
        assert!(total <= outcome.elapsed);
    }
    let total: Duration = report.outcomes.iter().map(|o| o.elapsed).sum();
    assert!(total + report.final_cleanup.unwrap_or_default() <= report.elapsed);
}

#[test]
fn retained_allocation_profile_replays_without_relabeling_the_readback_cohort() {
    let record: Value = serde_json::from_str(PROFILE).unwrap();
    diagnostic::validate(&record, 1, &PROFILE_PROTOCOL);
    for case in record["cases"].as_array().unwrap() {
        for search in case["searches"].as_array().unwrap() {
            let report: TuneReport = serde_json::from_value(search["report"].clone()).unwrap();
            check_preparation(&report);
            assert_eq!(report.scratch, None);
            assert_eq!(report.final_cleanup, None);
            let costs = preparation::costs(&report).unwrap();
            println!(
                "{} {}: preparation {:.6}, staging {:.6}, buffers {:.6}, encoder {:.6}, cleanup {:.6} ms",
                case["name"],
                search["staging"],
                costs["preparation"],
                costs["prep_staging"],
                costs["prep_buffers"],
                costs["prep_encoder"],
                costs["cleanup"]
            );
        }
    }
}
