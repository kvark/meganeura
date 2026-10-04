//! One GPU context for the whole test process.
//!
//! The NVIDIA driver stops handing out contexts after roughly ten
//! create/drop cycles in a process. A suite that builds a throwaway session
//! per test crosses that long before it finishes, and `SessionConfig::from_env`
//! now refuses to continue on a different adapter when `MEGANEURA_DEVICE_ID`
//! names one — so those tests fail at setup rather than silently measuring the
//! wrong hardware.
//!
//! Sharing is what these tests want anyway: they compare kernels against the
//! f64 reference, so every case has to run on the same device for the
//! comparison to mean anything. Sessions still isolate each test — buffers,
//! poisoning and the dispatch plan are per-session.

use std::sync::{Arc, OnceLock};

/// A process-wide context, honouring `MEGANEURA_DEVICE_ID` and the other
/// `GpuOptions` environment variables.
pub fn gpu() -> Arc<blade_graphics::Context> {
    static CONTEXT: OnceLock<Arc<blade_graphics::Context>> = OnceLock::new();
    CONTEXT
        .get_or_init(|| {
            Arc::new(
                meganeura::init_gpu_context_with(meganeura::GpuOptions::from_env()).expect(
                    "a GPU context for the test suite. If MEGANEURA_DEVICE_ID names a device that \
                     cannot be opened, that is what this reports.",
                ),
            )
        })
        .clone()
}
/// [`meganeura::SessionConfig::from_env`] on the shared context.
///
/// `from_env` creates its own device-selected context and hands it to exactly
/// one session, so a suite calling it per test exhausts the driver's budget.
/// This applies the same environment overrides without that cost — see
/// `SessionConfig::from_env_with_gpu`.
pub fn config() -> meganeura::SessionConfig<'static> {
    meganeura::SessionConfig::from_env_with_gpu(Some(gpu()))
}

/// [`config`] in inference mode.
pub fn inference_config() -> meganeura::SessionConfig<'static> {
    meganeura::SessionConfig {
        mode: meganeura::Mode::Inference,
        ..config()
    }
}
