//! Central registry of every `MEGANEURA_*` environment variable — and the
//! ONLY place that reads them.
//!
//! The library core is environment-free: `compile`, `runtime`, `codegen`,
//! and `optimize` accept strongly typed options and never touch
//! `std::env`. Clients that want env-driven behavior (the repo's own
//! examples, benches, and tests; external harnesses like Inferena) opt in
//! explicitly through the `from_env` constructors below, typically:
//!
//! ```no_run
//! # let graph = meganeura::Graph::new();
//! let session = meganeura::build(&graph, meganeura::SessionConfig::from_env()).0;
//! ```
//!
//! Every variable the library reads is declared here — its type, its
//! class, its documentation, and nothing else reads `std::env` for
//! `MEGANEURA_*` names. That buys three things the old scattered
//! `env::var` calls couldn't provide:
//!
//! * **Discoverability** — [`describe`] renders the full table, and a
//!   test pins the README against the registry so docs can't drift.
//! * **Visibility** — [`log_overrides`] (called by every `from_env`
//!   constructor) logs every variable that is actually set, and warns about
//!   `MEGANEURA_*` names it doesn't recognize, so a typo like
//!   `MEGANEURA_DISBLE_COOP=1` fails loudly instead of silently doing
//!   nothing.
//! * **Uniform semantics** — every boolean variable parses the same
//!   way: unset → its default, `0` → off, anything else → on.
//!   (Historically some flags treated *any* value, including `0`, as
//!   on.)
//!
//! Precedence is simple: `from_env` resolves the environment into a
//! value, and any field the caller assigns afterwards wins. A binary
//! that never calls `from_env` is completely environment-independent.
//! The classes below describe intent:
//!
//! * [`VarClass::Diagnostic`] switches exist to change the behavior of
//!   a binary you can't edit — harnesses should honor them by building
//!   their configs through `from_env`.
//! * [`VarClass::Tuning`] variables feed the defaults of
//!   [`TuningKnobs`] / `SessionConfig` knobs.
//! * [`VarClass::Selection`] variables choose external resources
//!   (device, output paths) and have no in-code counterpart.
//! * [`VarClass::External`] names are read by tests/examples, not the
//!   library; they're registered so the unknown-variable warning
//!   doesn't fire on them.

use crate::compile::{CompileOptions, TuningKnobs};
use crate::optimize::{ExtractionCost, OptimizeConfig, OptimizeMode};
use crate::runtime::{CoopPolicy, GpuOptions, SessionOptions};
use crate::train::SessionConfig;
use std::sync::OnceLock;

/// How a variable's value is interpreted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VarKind {
    /// Boolean: unset → default, `"0"` → false, anything else → true.
    Bool,
    /// Unsigned integer.
    U32,
    /// Free-form text (device ids, paths, mode names, ranges).
    Text,
}

/// What kind of decision the variable participates in (see module docs
/// for the precedence rules per class).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VarClass {
    /// Debugging escape hatch; overrides code-level configuration.
    Diagnostic,
    /// Performance knob; provides the *default* for a typed option.
    Tuning,
    /// Selects an external resource (adapter, dump path, mode).
    Selection,
    /// Read by tests/examples only; registered for typo detection.
    External,
}

/// One registered environment variable.
pub struct VarSpec {
    pub name: &'static str,
    pub kind: VarKind,
    pub class: VarClass,
    pub doc: &'static str,
}

impl VarSpec {
    fn raw(&self) -> Option<String> {
        std::env::var(self.name).ok()
    }

    /// Boolean read: unset → `default`, `"0"` → `false`, else `true`.
    pub fn bool_or(&self, default: bool) -> bool {
        debug_assert_eq!(self.kind, VarKind::Bool, "{} is not boolean", self.name);
        match self.raw() {
            None => default,
            Some(v) => v != "0",
        }
    }

    /// `u32` read; unparsable values are ignored with a warning.
    pub fn u32_value(&self) -> Option<u32> {
        debug_assert_eq!(self.kind, VarKind::U32, "{} is not u32", self.name);
        let raw = self.raw()?;
        match raw.parse::<u32>() {
            Ok(v) => Some(v),
            Err(_) => {
                log::warn!("{}={raw:?} is not a u32; ignoring", self.name);
                None
            }
        }
    }

    /// Text read.
    pub fn text(&self) -> Option<String> {
        debug_assert_eq!(self.kind, VarKind::Text, "{} is not text", self.name);
        self.raw()
    }
}

macro_rules! registry {
    ($($ident:ident: $name:literal, $kind:ident, $class:ident, $doc:literal;)*) => {
        $(pub static $ident: VarSpec = VarSpec {
            name: $name,
            kind: VarKind::$kind,
            class: VarClass::$class,
            doc: $doc,
        };)*
        /// Every registered variable.
        pub static REGISTRY: &[&VarSpec] = &[$(&$ident),*];
    };
}

registry! {
    // --- Diagnostic escape hatches (override code configuration) ---
    DISABLE_COOP: "MEGANEURA_DISABLE_COOP", Bool, Diagnostic,
        "Force the portable scalar matmul path; disables cooperative-matrix selection entirely.";
    COOP_F16: "MEGANEURA_COOP_F16", Bool, Diagnostic,
        "Opt in to f16-input cooperative tiles on devices that advertise no f32 tile.";
    FLASH_FWD_COOP: "MEGANEURA_FLASH_FWD_COOP", Bool, Diagnostic,
        "Set to 0 to disable only cooperative flash-attention forward.";
    FLASH_BWD_COOP: "MEGANEURA_FLASH_BWD_COOP", Bool, Diagnostic,
        "Enable the experimental reduced-precision cooperative flash backward.";
    NO_ALIAS: "MEGANEURA_NO_ALIAS", Bool, Diagnostic,
        "Disable buffer lifetime aliasing; every logical buffer gets its own allocation.";
    NO_DEVICE_LOCAL: "MEGANEURA_NO_DEVICE_LOCAL", Bool, Diagnostic,
        "Keep all buffers host-visible instead of device-local.";
    SERIAL_DISPATCH: "MEGANEURA_SERIAL_DISPATCH", Bool, Diagnostic,
        "One compute pass per dispatch — guarantees serial execution for bisection.";
    NO_WINOGRAD: "MEGANEURA_NO_WINOGRAD", Bool, Diagnostic,
        "Skip the Conv2d-to-Winograd rewrite; the selection heuristic weighs channel counts only, so this measures which side of it a workload belongs on.";
    PIN_BUFS: "MEGANEURA_PIN_BUFS", Text, Diagnostic,
        "Force-pin logical buffers by id/range (e.g. \"3,17,25-40\") to bisect aliasing bugs.";
    DUMP_PLAN: "MEGANEURA_DUMP_PLAN", Bool, Diagnostic,
        "Dump dispatch order, provenance, accesses, and the alias map at session build.";
    DUMP_WGSL: "MEGANEURA_DUMP_WGSL", Text, Diagnostic,
        "Directory to write every generated/parsed WGSL shader into.";
    OPTIMIZER: "MEGANEURA_OPTIMIZER", Text, Diagnostic,
        "Rewrite mode: off | greedy | egglog-windowed | egglog-outlined | egglog-whole.";
    EGRAPH_COST: "MEGANEURA_EGRAPH_COST", Text, Diagnostic,
        "Extraction objective: ast-size | tensor-traffic.";
    EGRAPH_CUTOFF: "MEGANEURA_EGRAPH_CUTOFF", U32, Diagnostic,
        "Saturation segment-size ceiling (default 300).";

    // --- Tuning defaults (explicit code-level options win) ---
    TUNE: "MEGANEURA_TUNE", Bool, Tuning,
        "Run Session::tune at build: measure coop vs scalar per kernel family and keep the winner.";
    FLASH_EPT_CAP: "MEGANEURA_FLASH_EPT_CAP", U32, Tuning,
        "Elements-per-thread cap for flash-attention forward codegen (power of two ≥ 2).";
    FLASH_GRAD_Q_EPT_CAP: "MEGANEURA_FLASH_GRAD_Q_EPT_CAP", U32, Tuning,
        "EPT cap for the flash dQ backward kernel.";
    FLASH_GRAD_KV_EPT_CAP: "MEGANEURA_FLASH_GRAD_KV_EPT_CAP", U32, Tuning,
        "EPT cap for the fused flash dK/dV backward kernel.";
    FLASH_BWD_EPT_CAP: "MEGANEURA_FLASH_BWD_EPT_CAP", U32, Tuning,
        "Shared fallback EPT cap for both flash backward kernels.";

    // --- Resource / mode selection ---
    DEVICE_ID: "MEGANEURA_DEVICE_ID", Text, Selection,
        "Adapter selection by backend-reported numeric device id (decimal or 0x-hex).";
    GPU_TIMING: "MEGANEURA_GPU_TIMING", Bool, Selection,
        "Enable hardware timestamp query pools (must be set before the GPU context is created).";

    // --- Read by tests/examples, not the library ---
    TRACE: "MEGANEURA_TRACE", Text, External,
        "Examples convention: write a Perfetto trace to this path.";
    SKIP_BACKPROP: "MEGANEURA_SKIP_BACKPROP", Bool, External,
        "Tests convention: skip MHA-backward tests on software drivers with broken wg reductions.";
}

impl OptimizeConfig {
    /// Read benchmark-oriented overrides while retaining production defaults.
    ///
    /// - `MEGANEURA_OPTIMIZER=off|greedy|egglog-windowed|egglog-outlined|egglog-whole`
    /// - `MEGANEURA_EGRAPH_COST=ast-size|tensor-traffic`
    /// - `MEGANEURA_EGRAPH_CUTOFF=<positive integer>`
    /// - `MEGANEURA_NO_WINOGRAD`
    pub fn from_env() -> Self {
        log_overrides();
        let mut config = Self {
            no_winograd: NO_WINOGRAD.bool_or(false),
            ..Self::default()
        };
        if let Some(value) = OPTIMIZER.text() {
            config.mode = match value.as_str() {
                "off" => OptimizeMode::Off,
                "greedy" => OptimizeMode::Greedy,
                "egglog-windowed" | "windowed" => OptimizeMode::EgglogWindowed,
                "egglog-outlined" | "outlined" => OptimizeMode::EgglogOutlined,
                "egglog-whole" | "whole" => OptimizeMode::EgglogWhole,
                _ => {
                    log::warn!("unknown MEGANEURA_OPTIMIZER={value:?}; using greedy");
                    OptimizeMode::Greedy
                }
            };
        }
        if let Some(value) = EGRAPH_COST.text() {
            config.extraction_cost = match value.as_str() {
                "ast-size" | "ast" | "unit" => ExtractionCost::AstSize,
                "tensor-traffic" | "traffic" => ExtractionCost::TensorTraffic,
                _ => {
                    log::warn!("unknown MEGANEURA_EGRAPH_COST={value:?}; using tensor-traffic");
                    ExtractionCost::TensorTraffic
                }
            };
        }
        if let Some(value) = EGRAPH_CUTOFF.u32_value() {
            if value > 0 {
                config.saturation_cutoff = value as usize;
            } else {
                log::warn!("MEGANEURA_EGRAPH_CUTOFF must be > 0; using the default");
            }
        }
        config
    }
}

impl TuningKnobs {
    /// Platform defaults with `MEGANEURA_FLASH_*_EPT_CAP` overrides.
    /// Caps must be powers of two ≥ 2; invalid values warn and are ignored.
    pub fn from_env() -> Self {
        log_overrides();
        fn cap(var: &VarSpec) -> Option<u32> {
            var.u32_value().filter(|v| {
                let ok = v.is_power_of_two() && *v >= 2;
                if !ok {
                    log::warn!("{}={v} must be a power of two >= 2; ignoring", var.name);
                }
                ok
            })
        }
        let d = Self::default();
        let bwd = cap(&FLASH_BWD_EPT_CAP);
        let fwd = cap(&FLASH_EPT_CAP);
        Self {
            flash_ept_cap: fwd.unwrap_or(d.flash_ept_cap),
            flash_grad_q_ept_cap: cap(&FLASH_GRAD_Q_EPT_CAP)
                .or(bwd)
                .or(fwd)
                .unwrap_or(d.flash_grad_q_ept_cap),
            flash_grad_kv_ept_cap: cap(&FLASH_GRAD_KV_EPT_CAP)
                .or(bwd)
                .or(fwd)
                .unwrap_or(d.flash_grad_kv_ept_cap),
        }
    }
}

impl CompileOptions {
    /// Defaults with the flash-coop switches and EPT caps read from the
    /// environment.
    pub fn from_env() -> Self {
        log_overrides();
        Self {
            knobs: TuningKnobs::from_env(),
            flash_forward_coop: FLASH_FWD_COOP.bool_or(true),
            flash_backward_coop: FLASH_BWD_COOP.bool_or(false),
            ..Self::default()
        }
    }
}

impl SessionOptions {
    /// Defaults with the diagnostic switches and cooperative-matrix policy
    /// read from the environment.
    pub fn from_env() -> Self {
        log_overrides();
        let coop = if DISABLE_COOP.bool_or(false) {
            CoopPolicy::Disabled
        } else if COOP_F16.bool_or(false) {
            CoopPolicy::AllowF16
        } else {
            CoopPolicy::Auto
        };
        Self {
            debug: false,
            coop,
            no_alias: NO_ALIAS.bool_or(false),
            no_device_local: NO_DEVICE_LOCAL.bool_or(false),
            serial_dispatch: SERIAL_DISPATCH.bool_or(false),
            dump_plan: DUMP_PLAN.bool_or(false),
            pin_buffers: PIN_BUFS.text(),
        }
    }
}

impl GpuOptions {
    /// Adapter selection and timestamp collection from the environment.
    pub fn from_env() -> Self {
        log_overrides();
        let device_id = DEVICE_ID.text().and_then(|value| {
            let parsed = crate::runtime::parse_device_id(&value);
            if parsed.is_none() {
                log::warn!(
                    "ignoring invalid MEGANEURA_DEVICE_ID={value:?}; \
                     expected decimal or 0x-prefixed u32"
                );
            }
            parsed
        });
        Self {
            device_id,
            timing: GPU_TIMING.bool_or(false),
        }
    }
}

impl SessionConfig<'_> {
    /// A [`SessionConfig`] with every environment override applied — the
    /// one-liner for harnesses, examples, and env-driven test runs:
    /// compile options, tuning knobs, optimizer mode, diagnostic switches,
    /// coop policy, `MEGANEURA_TUNE`, and (only when `MEGANEURA_DEVICE_ID`
    /// or `MEGANEURA_GPU_TIMING` is set) a GPU context created with those
    /// options. Also installs the WGSL dump directory when
    /// `MEGANEURA_DUMP_WGSL` is set.
    ///
    /// Fields assigned *after* this call win — precedence is simply
    /// "explicit code runs last".
    pub fn from_env() -> Self {
        log_overrides();
        if let Some(dir) = DUMP_WGSL.text() {
            crate::codegen::set_wgsl_dump_dir(dir);
        }
        let gpu_opts = GpuOptions::from_env();
        let gpu = if gpu_opts.device_id.is_some() || gpu_opts.timing {
            match crate::runtime::init_gpu_context_with(gpu_opts) {
                Ok(context) => Some(std::sync::Arc::new(context)),
                Err(e) => {
                    log::warn!("env-selected GPU init failed ({e:?}); using default adapter");
                    None
                }
            }
        } else {
            None
        };
        Self {
            gpu,
            options: CompileOptions::from_env(),
            optimize: OptimizeConfig::from_env(),
            runtime: SessionOptions::from_env(),
            tune: TUNE.bool_or(false),
            ..Self::default()
        }
    }

    /// [`SessionConfig::from_env`] with `mode: Inference`.
    pub fn inference_from_env() -> Self {
        Self {
            mode: crate::train::Mode::Inference,
            ..Self::from_env()
        }
    }

    /// [`SessionConfig::from_env`] with `skip_full_optimize: true`.
    pub fn unoptimized_from_env() -> Self {
        Self {
            skip_full_optimize: true,
            ..Self::from_env()
        }
    }
}

/// Render the registry as an aligned plain-text table.
pub fn describe() -> String {
    use std::fmt::Write;
    let mut out = String::new();
    let width = REGISTRY.iter().map(|v| v.name.len()).max().unwrap_or(0);
    for var in REGISTRY {
        let _ = writeln!(
            out,
            "{:width$}  {:4?}  {:10?}  {}",
            var.name, var.kind, var.class, var.doc
        );
    }
    out
}

/// Log every `MEGANEURA_*` variable that is set (once per process), and
/// warn about set variables the registry doesn't know — almost always a
/// typo. Called from session construction; cheap and idempotent.
pub fn log_overrides() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        let mut active = Vec::new();
        for (name, value) in std::env::vars() {
            if !name.starts_with("MEGANEURA_") {
                continue;
            }
            match REGISTRY.iter().find(|v| v.name == name) {
                Some(var) => {
                    if var.class != VarClass::External {
                        active.push(format!("{name}={value}"));
                    }
                }
                None => log::warn!(
                    "unrecognized environment variable {name}={value} — \
                     not a registered MEGANEURA_* name (typo?); it has no effect"
                ),
            }
        }
        if !active.is_empty() {
            log::info!("environment overrides active: {}", active.join(", "));
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_names_are_unique_and_prefixed() {
        let mut seen = std::collections::HashSet::new();
        for var in REGISTRY {
            assert!(var.name.starts_with("MEGANEURA_"), "{}", var.name);
            assert!(seen.insert(var.name), "duplicate: {}", var.name);
        }
    }

    /// Every user-facing variable must be documented in the README, so
    /// the docs can't drift from the code. External (tests/examples)
    /// names are exempt.
    #[test]
    fn readme_documents_every_variable() {
        let readme = include_str!("../README.md");
        let mut missing = Vec::new();
        for var in REGISTRY {
            if var.class == VarClass::External {
                continue;
            }
            if !readme.contains(var.name) {
                missing.push(var.name);
            }
        }
        assert!(
            missing.is_empty(),
            "README.md does not mention: {missing:?} — document them \
             in the environment-variable table"
        );
    }

    #[test]
    fn no_winograd_reaches_the_typed_option() {
        // Not set: the rewrite stays on.
        unsafe { std::env::remove_var("MEGANEURA_NO_WINOGRAD") };
        assert!(!crate::optimize::OptimizeConfig::from_env().no_winograd);
        // Set: it turns off, and "0" means off in the uniform semantics.
        unsafe { std::env::set_var("MEGANEURA_NO_WINOGRAD", "1") };
        assert!(crate::optimize::OptimizeConfig::from_env().no_winograd);
        unsafe { std::env::set_var("MEGANEURA_NO_WINOGRAD", "0") };
        assert!(!crate::optimize::OptimizeConfig::from_env().no_winograd);
        unsafe { std::env::remove_var("MEGANEURA_NO_WINOGRAD") };
    }

    #[test]
    fn describe_lists_everything() {
        let d = describe();
        for var in REGISTRY {
            assert!(d.contains(var.name));
        }
    }
}
