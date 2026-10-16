//! Shared helpers for the test suites.
pub mod gpu;

/// Tuning that compares one challenger at a time, qualifying every challenger,
/// and visits classes in their structural order: what the suites that use the
/// tuner as a per-candidate correctness harness rely on.
#[allow(dead_code)] // Each integration target compiles this shared module independently.
pub fn sequential_tuning() -> meganeura::TuneOptions {
    meganeura::TuneOptions {
        strategy: meganeura::TuneStrategy::Sequential,
        calibration: meganeura::TuneCalibration::Off,
        ..Default::default()
    }
}
