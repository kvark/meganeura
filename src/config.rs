//! Central registry of every `MEGANEURA_*` environment variable.
//!
//! Every variable the library reads is declared here — its type, its
//! class, its documentation, and nothing else reads `std::env` for
//! `MEGANEURA_*` names. That buys three things the old scattered
//! `env::var` calls couldn't provide:
//!
//! * **Discoverability** — [`describe`] renders the full table, and a
//!   test pins the README against the registry so docs can't drift.
//! * **Visibility** — [`log_overrides`] (called once at session build)
//!   logs every variable that is actually set, and warns about
//!   `MEGANEURA_*` names it doesn't recognize, so a typo like
//!   `MEGANEURA_DISBLE_COOP=1` fails loudly instead of silently doing
//!   nothing.
//! * **Uniform semantics** — every boolean variable parses the same
//!   way: unset → its default, `0` → off, anything else → on.
//!   (Historically some flags treated *any* value, including `0`, as
//!   on.)
//!
//! Precedence is per class:
//!
//! * [`VarClass::Diagnostic`] switches **override everything** — they
//!   exist to change the behavior of a binary you can't edit.
//! * [`VarClass::Tuning`] variables feed **defaults** — an explicitly
//!   constructed [`TuningKnobs`](crate::compile::TuningKnobs) or
//!   `SessionConfig` field wins over the environment.
//! * [`VarClass::Selection`] variables choose external resources
//!   (device, output paths) and have no in-code counterpart.
//! * [`VarClass::External`] names are read by tests/examples, not the
//!   library; they're registered so the unknown-variable warning
//!   doesn't fire on them.

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
    fn describe_lists_everything() {
        let d = describe();
        for var in REGISTRY {
            assert!(d.contains(var.name));
        }
    }
}
