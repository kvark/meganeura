//! Schedule templates — generic kernel generators that lower to Naga IR.
//!
//! This module is the foundation of the generic-fusion plan (see
//! `docs/plan-generic-fusion.md`). Instead of carrying a zoo of hand-written
//! WGSL files, we define a handful of **schedule templates** (pointwise,
//! reduction, matmul±prologue±epilogue, attention) and generate WGSL from
//! parameterized specifications.
//!
//! Design:
//!   - Keep Naga `Module` as our IR; emit WGSL source text, parse via
//!     `naga::front::wgsl`. Parsing is ~100µs — specialization is cheap.
//!   - `PointwiseDAG` is shared across all archetypes (used as prologue /
//!     epilogue on heavy kernels).
//!
//! This initial commit lands only archetype 1 (pointwise) and is
//! deliberately disconnected from the rest of the compiler — it proves
//! the codegen pattern before we rewire call sites.

use std::fmt::Write;

use crate::codegen::ShaderModule;

// -------------------------------------------------------------------------
// PointwiseDAG — shared by all archetypes as a fused elementwise chain.
// -------------------------------------------------------------------------

/// A scalar elementwise operation, executed per-element.
#[derive(Clone, Debug, PartialEq)]
pub enum Pw {
    /// Load input `idx` at the current element position.
    LoadInput(u8),
    /// Literal f32 constant.
    Const(f32),
    /// Binary ops — operate on two value-node indices into [`PointwiseDAG::ops`].
    Add(u16, u16),
    Mul(u16, u16),
    Sub(u16, u16),
    Div(u16, u16),
    Greater(u16, u16),
    /// Unary ops.
    Neg(u16),
    Recip(u16),
    Exp(u16),
    Log(u16),
    Abs(u16),
    Sqrt(u16),
    Rsqrt(u16),
    /// `max(v, 0)` — Relu.
    Relu(u16),
    /// `1 / (1 + exp(-v))` — Sigmoid.
    Sigmoid(u16),
    /// `v * sigmoid(v)` — Silu.
    Silu(u16),
    /// `tanh(v)`.
    Tanh(u16),
}

/// A DAG of scalar elementwise ops with `n_inputs` input streams and one
/// output. Value nodes reference earlier entries by index.
#[derive(Clone, Debug)]
pub struct PointwiseDAG {
    pub n_inputs: u8,
    pub ops: Vec<Pw>,
    /// Index into `ops` — the final output value.
    pub output: u16,
}

impl PointwiseDAG {
    /// Emit WGSL for the DAG body: assumes inputs are available as
    /// `src_0[i]`, `src_1[i]`, … and assigns the result to `let out = ...;`.
    /// Returns the body string. `i` is the element index variable name.
    pub fn emit_body(&self, i: &str) -> String {
        let mut out = String::new();
        for (k, op) in self.ops.iter().enumerate() {
            let _ = write!(out, "            let v{} = ", k);
            match *op {
                Pw::LoadInput(idx) => {
                    let _ = write!(out, "src_{}[{}]", idx, i);
                }
                Pw::Const(c) => {
                    // f32 literal with decimal point, required by WGSL.
                    let _ = write!(out, "{:?}", c);
                }
                Pw::Add(a, b) => {
                    let _ = write!(out, "v{} + v{}", a, b);
                }
                Pw::Mul(a, b) => {
                    let _ = write!(out, "v{} * v{}", a, b);
                }
                Pw::Sub(a, b) => {
                    let _ = write!(out, "v{} - v{}", a, b);
                }
                Pw::Div(a, b) => {
                    let _ = write!(out, "v{} / v{}", a, b);
                }
                Pw::Greater(a, b) => {
                    let _ = write!(out, "select(0.0, 1.0, v{} > v{})", a, b);
                }
                Pw::Neg(a) => {
                    let _ = write!(out, "-v{}", a);
                }
                Pw::Recip(a) => {
                    let _ = write!(out, "1.0 / v{}", a);
                }
                Pw::Exp(a) => {
                    let _ = write!(out, "exp(v{})", a);
                }
                Pw::Log(a) => {
                    let _ = write!(out, "log(v{})", a);
                }
                Pw::Abs(a) => {
                    let _ = write!(out, "abs(v{})", a);
                }
                Pw::Sqrt(a) => {
                    let _ = write!(out, "sqrt(v{})", a);
                }
                Pw::Rsqrt(a) => {
                    let _ = write!(out, "inverseSqrt(v{})", a);
                }
                Pw::Relu(a) => {
                    let _ = write!(out, "max(v{}, 0.0)", a);
                }
                Pw::Sigmoid(a) => {
                    let _ = write!(out, "1.0 / (1.0 + exp(-v{}))", a);
                }
                Pw::Silu(a) => {
                    let _ = write!(out, "v{} / (1.0 + exp(-v{}))", a, a);
                }
                Pw::Tanh(a) => {
                    let _ = write!(out, "tanh(v{})", a);
                }
            }
            out.push_str(";\n");
        }
        out
    }
}

// -------------------------------------------------------------------------
// KernelTemplate — the four archetypes (only Pointwise implemented so far).
// -------------------------------------------------------------------------

/// How threads are assigned to output elements.
#[derive(Clone, Copy, Debug)]
pub struct GridShape {
    /// Elements per workgroup along the X axis. Fixed 256 for now.
    pub workgroup_size: u32,
}

impl Default for GridShape {
    fn default() -> Self {
        Self {
            workgroup_size: 256,
        }
    }
}

/// A schedule template — to be lowered to a Naga [`Module`](naga::Module).
#[derive(Clone, Debug)]
pub enum KernelTemplate {
    Pointwise { dag: PointwiseDAG, grid: GridShape },
    // Reduction, Matmul, Attention — added in later steps of the plan.
}

/// Lower a [`KernelTemplate`] to WGSL + a parsed Naga module.
pub fn lower(t: &KernelTemplate) -> ShaderModule {
    match *t {
        KernelTemplate::Pointwise { ref dag, grid } => lower_pointwise(dag, grid),
    }
}

fn lower_pointwise(dag: &PointwiseDAG, grid: GridShape) -> ShaderModule {
    let mut src = String::new();
    src.push_str(
        "struct Params {\n    len: u32,\n    _pad0: u32,\n    _pad1: u32,\n    _pad2: u32,\n}\n\n",
    );
    for k in 0..dag.n_inputs {
        let _ = writeln!(src, "var<storage> src_{}: array<f32>;", k);
    }
    src.push_str("var<storage, read_write> dst: array<f32>;\n");
    src.push_str("var<uniform> params: Params;\n\n");
    let _ = writeln!(src, "@compute @workgroup_size({})", grid.workgroup_size);
    src.push_str("fn main(@builtin(global_invocation_id) gid: vec3<u32>) {\n");
    src.push_str("    let i = gid.x;\n");
    src.push_str("    if i >= params.len { return; }\n");
    src.push_str(&dag.emit_body("i"));
    let _ = writeln!(src, "            dst[i] = v{};", dag.output);
    src.push_str("}\n");

    let module = naga::front::wgsl::parse_str(&src)
        .unwrap_or_else(|e| panic!("generated WGSL failed to parse:\n{}\n---\n{}", e, src));
    ShaderModule {
        module,
        source: src,
    }
}

// -------------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// The equivalent of `binary.wgsl`'s `add` entry:
    ///   dst[i] = src_a[i] + src_b[i]
    fn add_dag() -> PointwiseDAG {
        PointwiseDAG {
            n_inputs: 2,
            ops: vec![
                Pw::LoadInput(0), // v0 = src_0[i]
                Pw::LoadInput(1), // v1 = src_1[i]
                Pw::Add(0, 1),    // v2 = v0 + v1
            ],
            output: 2,
        }
    }

    /// Equivalent of `binary.wgsl`'s `swiglu`:
    ///   dst[i] = silu(src_a[i]) * src_b[i]
    fn swiglu_dag() -> PointwiseDAG {
        PointwiseDAG {
            n_inputs: 2,
            ops: vec![
                Pw::LoadInput(0), // v0 = gate
                Pw::LoadInput(1), // v1 = up
                Pw::Silu(0),      // v2 = silu(gate)
                Pw::Mul(2, 1),    // v3 = silu(gate) * up
            ],
            output: 3,
        }
    }

    #[test]
    fn lowers_add_to_valid_wgsl() {
        let sm = lower(&KernelTemplate::Pointwise {
            dag: add_dag(),
            grid: GridShape::default(),
        });
        // Sanity: has expected pieces.
        assert!(sm.source.contains("var<storage> src_0: array<f32>;"));
        assert!(sm.source.contains("var<storage> src_1: array<f32>;"));
        assert!(sm.source.contains("dst[i] = v2;"));
        // Naga parse already validated in lower().
        assert!(!sm.module.entry_points.is_empty());
    }

    #[test]
    fn lowers_swiglu_to_valid_wgsl() {
        let sm = lower(&KernelTemplate::Pointwise {
            dag: swiglu_dag(),
            grid: GridShape::default(),
        });
        // Silu expansion present, multiply present.
        assert!(sm.source.contains("1.0 + exp(-v0)"));
        assert!(sm.source.contains("v2 * v1"));
        assert!(sm.source.contains("dst[i] = v3;"));
    }

    #[test]
    fn naga_validates_pointwise_module() {
        use naga::valid::{Capabilities, ValidationFlags, Validator};
        let sm = lower(&KernelTemplate::Pointwise {
            dag: add_dag(),
            grid: GridShape::default(),
        });
        // Blade injects bindings at pipeline creation — validate without them,
        // matching the convention used in codegen.rs.
        let flags = ValidationFlags::all() ^ ValidationFlags::BINDINGS;
        let mut v = Validator::new(flags, Capabilities::all());
        v.validate(&sm.module)
            .unwrap_or_else(|e| panic!("naga validation failed: {:?}\n{}", e, sm.source));
    }
}
