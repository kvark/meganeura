//! Device selection and context creation.
//!
//! Split out because nothing here touches a `Session`: it is the layer that
//! answers "which adapter, and with what capabilities" before one exists.

/// GPU context creation options. The library never reads the environment;
/// map device, timing and capture overrides with
/// [`GpuOptions::from_env`] if you want env-driven selection.
#[derive(Clone, Debug, Default)]
pub struct GpuOptions {
    /// Adapter selection by backend-reported numeric device id.
    pub device_id: Option<u32>,
    /// Enable hardware timestamp query pools (needed before the context
    /// exists; feeds `dump_gpu_timings` and the profiler).
    pub timing: bool,
    /// Enable Blade's native-tool capture support, including shader debug
    /// information and command labels. Independent of pass timestamps;
    /// does not change Meganeura's dispatch grouping. Off by default.
    pub capture: bool,
}

/// Create a GPU context with the same environment-independent defaults as
/// [`crate::Session::new`]. Use [`init_gpu_context_with`] for explicit options.
pub fn init_gpu_context() -> Result<blade_graphics::Context, blade_graphics::NotSupportedError> {
    init_gpu_context_with(GpuOptions::default())
}

pub fn init_gpu_context_with(
    options: GpuOptions,
) -> Result<blade_graphics::Context, blade_graphics::NotSupportedError> {
    // Blade panics at init if timing is asked for on a device that cannot
    // timestamp. Ask first and fail the same way every other unsupported
    // request here does.
    if options.timing {
        let can_time = blade_graphics::Context::enumerate()
            .map(|reports| {
                let available = |report: &&blade_graphics::DeviceReport| {
                    matches!(
                        report.status,
                        blade_graphics::DeviceReportStatus::Available { .. }
                    )
                };
                let selected = if let Some(id) = options.device_id {
                    reports.iter().find(|report| report.device_id == id)
                } else {
                    reports
                        .iter()
                        .find(|report| {
                            matches!(
                                report.status,
                                blade_graphics::DeviceReportStatus::Available {
                                    is_default: true,
                                    ..
                                }
                            )
                        })
                        // Metal's context-free enumeration cannot identify
                        // the system default; its first available device is
                        // the one context creation selects.
                        .or_else(|| reports.iter().find(available))
                };
                matches!(
                    selected.map(|report| &report.status),
                    Some(blade_graphics::DeviceReportStatus::Available { caps, .. })
                        if caps.timing
                )
            })
            .unwrap_or(false);
        if !can_time {
            log::warn!("GPU timing requested but no available device can timestamp passes");
            return Err(blade_graphics::NotSupportedError::NoSupportedDeviceFound);
        }
    }
    let _span = tracing::info_span!(
        "gpu_context_init",
        timing = options.timing,
        capture = options.capture,
        device = ?options.device_id
    )
    .entered();
    // From here on there is a context that can record GPU pass ranges, so
    // arm the static profiler state now: every later recording path takes
    // it as given instead of lazily conjuring the state on first use.
    crate::profiler::arm();
    let dev_id = options.device_id;
    unsafe {
        blade_graphics::Context::init(blade_graphics::ContextDesc {
            validation: cfg!(debug_assertions),
            // Opt-in: GPU pass timestamps feed profiling and trace output.
            // Completed timestamp queries are resolved after the submission
            // fence. Keep this off by default to avoid instrumentation cost.
            timing: options.timing,
            capture: options.capture,
            overlay: false,
            device_id: dev_id,
            ..Default::default()
        })
    }
}
