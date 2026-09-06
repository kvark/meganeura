//! Shared versus read-optimized private staging; original fresh-buffer policy.
#[path = "support/tuning_diagnostic.rs"]
mod diagnostic;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use meganeura::{TuneOptions, TuneStaging, TuneStagingReuse};
    diagnostic::run(
        "readback-2026-09-06",
        ["shared", "download"],
        [TuneStaging::Shared, TuneStaging::Download].map(|staging| TuneOptions {
            scope: meganeura::TuneScope::Dense,
            staging,
            staging_reuse: TuneStagingReuse::Fresh,
            max_time: std::time::Duration::from_secs(10),
            ..Default::default()
        }),
    )
}
