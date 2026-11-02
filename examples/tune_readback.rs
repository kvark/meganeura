//! Readback and preparation phase costs of the fixed private staging, as a
//! matched control: both arms search with identical options.
#[path = "support/tuning_diagnostic.rs"]
mod diagnostic;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let options = meganeura::TuneOptions {
        scope: meganeura::TuneScope::Dense,
        max_time: std::time::Duration::from_secs(10),
        ..Default::default()
    };
    diagnostic::run(
        "readback-control",
        ["first", "second"],
        [options.clone(), options],
    )
}
