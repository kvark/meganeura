//! Fresh versus exact-size, call-local reuse of Download staging.
#[path = "support/tuning_diagnostic.rs"]
mod diagnostic;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use meganeura::{TuneOptions, TuneStaging, TuneStagingReuse};
    diagnostic::run(
        "staging-reuse-2026-09-06",
        ["fresh", "reuse"],
        [TuneStagingReuse::Fresh, TuneStagingReuse::SameSize].map(|staging_reuse| TuneOptions {
            scope: meganeura::TuneScope::Dense,
            staging: TuneStaging::Download,
            staging_reuse,
            max_time: std::time::Duration::from_secs(10),
            ..Default::default()
        }),
    )
}
