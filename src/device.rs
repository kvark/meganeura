//! Discovery surface for compute devices usable as meganeura execution
//! backends.
//!
//! Today: the GPU via Blade is the only training-capable backend.
//! On-SoC NPUs (Intel NPU, Apple ANE, Qualcomm Hexagon, AMD XDNA) are
//! inference-only and are wired in opportunistically — one backend per
//! vendor SDK, each behind its own cargo feature flag. Intel NPU via
//! OpenVINO is the first to land; the rest will follow the same shape.
//!
//! This module is intentionally narrow: it only enumerates devices and
//! describes them. It does not yet route execution to NPUs — that belongs
//! to a future `Session` variant. Treat this as the discovery half of an
//! eventual heterogeneous runtime.
//!
//! Enumeration is a one-shot probe of each compiled-in backend. The GPU
//! probe transiently initializes a `blade_graphics::Context` to read its
//! device information and drops it before returning; the OpenVINO probe
//! loads the OpenVINO C runtime and asks it for its device list.

/// Coarse classification of a compute device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeviceKind {
    /// GPU exposed through Blade (Vulkan / Metal / DX12). Programmable
    /// via WGSL kernels and the only backend that currently supports
    /// training (gradient + optimizer dispatch).
    Gpu,
    /// Intel on-SoC NPU (Meteor Lake / Lunar Lake / Arrow Lake and
    /// successors), reachable via the OpenVINO runtime. Inference-only.
    IntelNpu,
}

impl DeviceKind {
    /// Short lowercase tag, suitable for logs and CLI output.
    pub fn as_str(self) -> &'static str {
        match self {
            DeviceKind::Gpu => "gpu",
            DeviceKind::IntelNpu => "intel-npu",
        }
    }
}

/// Description of a single addressable compute device.
#[derive(Clone, Debug)]
pub struct DeviceInfo {
    pub kind: DeviceKind,
    /// Human-readable identifier reported by the underlying driver.
    pub name: String,
    /// Backend-native selector for this device. For GPU this is the
    /// string form of the value `MEGANEURA_DEVICE_ID` would take
    /// (`"default"` if unset). For OpenVINO this is the device string
    /// returned by `Core::available_devices` (`"NPU"`, `"NPU.0"`, …).
    pub backend_id: String,
    /// Whether the backend can execute the gradient + optimizer path.
    /// Currently only the GPU backend qualifies; every NPU surface
    /// available on shipped silicon today is inference-only.
    pub supports_training: bool,
}

/// Enumerate every compute device that meganeura can address from this
/// process, across all compiled-in backends.
///
/// The GPU is always probed via Blade. Additional backends contribute
/// when their feature flag is enabled and their runtime is available
/// on the host:
///
/// | Backend     | Feature flag | Contributes              |
/// |-------------|--------------|--------------------------|
/// | Blade GPU   | always       | `DeviceKind::Gpu` (×1)   |
/// | OpenVINO    | `openvino`   | `DeviceKind::IntelNpu`×N |
///
/// A backend that is compiled in but reports no devices (driver missing,
/// hardware absent, runtime load failed) contributes nothing and logs a
/// warning. The function never panics.
pub fn list_devices() -> Vec<DeviceInfo> {
    let mut out = Vec::new();
    out.extend(probe_gpu());
    #[cfg(feature = "openvino")]
    out.extend(openvino_probe::list_intel_npu_devices());
    out
}

fn probe_gpu() -> Option<DeviceInfo> {
    let ctx = match crate::runtime::init_gpu_context() {
        Ok(c) => c,
        Err(e) => {
            log::warn!("blade GPU init failed during device probe: {:?}", e);
            return None;
        }
    };
    let info = ctx.device_information();
    Some(DeviceInfo {
        kind: DeviceKind::Gpu,
        name: info.device_name.clone(),
        backend_id: std::env::var("MEGANEURA_DEVICE_ID")
            .unwrap_or_else(|_| "default".to_string()),
        supports_training: true,
    })
}

#[cfg(feature = "openvino")]
mod openvino_probe {
    use super::{DeviceInfo, DeviceKind};

    // API surface assumed for the `openvino` crate ~0.7. If the
    // installed version differs (older 0.4–0.6 split `Core` differently;
    // newer revisions may take a cache-directory argument to
    // `Core::new`), adjust the three call sites below — the rest of the
    // module is API-agnostic.
    pub fn list_intel_npu_devices() -> Vec<DeviceInfo> {
        let mut core = match openvino::Core::new() {
            Ok(c) => c,
            Err(e) => {
                log::warn!("OpenVINO Core::new failed: {:?}; skipping Intel NPU probe", e);
                return Vec::new();
            }
        };

        let devices = match core.available_devices() {
            Ok(d) => d,
            Err(e) => {
                log::warn!("OpenVINO available_devices failed: {:?}", e);
                return Vec::new();
            }
        };

        devices
            .into_iter()
            .map(|d| d.to_string())
            .filter(|d| d.starts_with("NPU"))
            .map(|backend_id| {
                let name = core
                    .get_property(&backend_id, "FULL_DEVICE_NAME")
                    .map(|v| v.to_string())
                    .unwrap_or_else(|_| backend_id.clone());
                DeviceInfo {
                    kind: DeviceKind::IntelNpu,
                    name,
                    backend_id,
                    supports_training: false,
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn list_devices_does_not_panic() {
        // Don't assert on contents — CI runners vary. Just confirm the
        // probe surface is callable end-to-end.
        let _ = list_devices();
    }

    #[test]
    fn device_kind_tags_are_distinct() {
        assert_ne!(DeviceKind::Gpu.as_str(), DeviceKind::IntelNpu.as_str());
    }
}
