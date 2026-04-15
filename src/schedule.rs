//! Schedule templates — generic kernel generators that lower to Naga IR.
//!
//! Foundation of the generic-fusion plan (see `docs/plan-generic-fusion.md`).
//! Instead of carrying a zoo of hand-written WGSL files, we define a handful
//! of **schedule templates** (pointwise, reduction, matmul±prologue±epilogue,
//! attention) and generate WGSL from parameterized specifications.
//!
//! Design:
//!   - Keep Naga `Module` as our IR; emit WGSL source text, parse via
//!     `naga::front::wgsl`. Parsing is ~100µs — specialization is cheap.
//!   - `PointwiseDAG` is shared across all archetypes (used as prologue /
//!     epilogue on heavy kernels in later steps).
//!   - Generated pointwise kernels for `n_inputs ∈ {1, 2}` use the same
//!     binding names as the existing hand-written `unary.wgsl` / `binary.wgsl`
//!     shaders (`src` / `src_a`+`src_b`), so they plug into the existing
//!     `UnaryData` / `BinaryData` runtime layouts with zero extra plumbing.
//!
//! This commit lands archetype 1 (pointwise) as a standalone lowerer.
//! Wiring into `compile.rs` / `runtime.rs` happens in the next step.

use std::collections::hash_map::DefaultHasher;
use std::fmt::Write;
use std::hash::{Hash, Hasher};

use serde::{Deserialize, Serialize};

use crate::codegen::ShaderModule;

// -------------------------------------------------------------------------
// PointwiseDAG — shared by all archetypes as a fused elementwise chain.
// -------------------------------------------------------------------------

/// A scalar elementwise operation, executed per-element.
///
/// `Hash` derived so that a `PointwiseDAG` can be used as a content-hashed
/// cache key when looking up generated pipelines.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Pw {
    /// Load input stream `idx` at the current element position.
    LoadInput(u8),
    /// Literal f32 constant. Stored as bit pattern so `Hash`/`Eq` are stable.
    Const(u32),
    // Binary ops — reference earlier value-node indices.
    Add(u16, u16),
    Mul(u16, u16),
    Sub(u16, u16),
    Div(u16, u16),
    Greater(u16, u16),
    // Unary ops.
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

impl Pw {
    /// Construct a `Const` from an f32, encoding via bit pattern.
    pub fn const_f32(v: f32) -> Self {
        Pw::Const(v.to_bits())
    }
}

/// A DAG of scalar elementwise ops with `n_inputs` input streams and one
/// output. Value nodes reference earlier entries by index.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PointwiseDAG {
    pub n_inputs: u8,
    pub ops: Vec<Pw>,
    /// Index into `ops` — the final output value.
    pub output: u16,
}

impl PointwiseDAG {
    /// Stable content hash — suitable as a pipeline-cache key.
    pub fn hash_key(&self) -> u64 {
        let mut h = DefaultHasher::new();
        self.hash(&mut h);
        h.finish()
    }

    /// Emit the DAG body as WGSL statements. `input_name(idx)` produces the
    /// expression used to load input stream `idx` at element index `i` (the
    /// caller's element variable — hardcoded to `i` to keep this simple).
    fn emit_body(&self, input_name: impl Fn(u8) -> String) -> String {
        let mut out = String::new();
        for (k, op) in self.ops.iter().enumerate() {
            let _ = write!(out, "    let v{} = ", k);
            match *op {
                Pw::LoadInput(idx) => {
                    let _ = write!(out, "{}[i]", input_name(idx));
                }
                Pw::Const(bits) => {
                    // Reconstruct f32 and emit with a decimal point + `f` suffix
                    // so WGSL parses it as f32 regardless of value.
                    let v = f32::from_bits(bits);
                    if v.is_finite() {
                        let _ = write!(out, "{:?}f", v);
                    } else if v.is_nan() {
                        out.push_str("bitcast<f32>(0x7fc00000u)");
                    } else if v.is_sign_positive() {
                        out.push_str("bitcast<f32>(0x7f800000u)");
                    } else {
                        out.push_str("bitcast<f32>(0xff800000u)");
                    }
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

    /// Binding names for each input stream, chosen to match the existing
    /// hand-written shaders so generated modules plug into the same
    /// runtime data layouts:
    ///   - n=1 → ["src"]           (matches `UnaryData`)
    ///   - n=2 → ["src_a", "src_b"] (matches `BinaryData`)
    ///   - n≥3 → ["src_0", "src_1", …]
    fn input_binding_names(n: u8) -> Vec<String> {
        match n {
            1 => vec!["src".to_string()],
            2 => vec!["src_a".to_string(), "src_b".to_string()],
            n => (0..n).map(|i| format!("src_{}", i)).collect(),
        }
    }
}

// -------------------------------------------------------------------------
// KernelTemplate — the four archetypes (only Pointwise implemented so far).
// -------------------------------------------------------------------------

/// How threads are assigned to output elements.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct GridShape {
    /// Threads per workgroup along the X axis.
    pub workgroup_size: u32,
}

impl Default for GridShape {
    fn default() -> Self {
        Self {
            workgroup_size: 256,
        }
    }
}

/// A schedule template — to be lowered to a Naga [`naga::Module`].
#[derive(Clone, Debug)]
pub enum KernelTemplate {
    Pointwise { dag: PointwiseDAG, grid: GridShape },
    // Reduction, Matmul, Attention — added in later steps of the plan.
}

/// Entry-point name emitted for a `KernelTemplate`. Always `"main"` for
/// pointwise kernels (one entry point per generated module).
pub const POINTWISE_ENTRY: &str = "main";

/// Lower a [`KernelTemplate`] to WGSL + a parsed Naga module.
pub fn lower(t: &KernelTemplate) -> ShaderModule {
    match *t {
        KernelTemplate::Pointwise { ref dag, grid } => lower_pointwise(dag, grid),
    }
}

fn lower_pointwise(dag: &PointwiseDAG, grid: GridShape) -> ShaderModule {
    assert!(
        dag.n_inputs >= 1,
        "PointwiseDAG must have at least one input stream"
    );
    assert!(
        (dag.output as usize) < dag.ops.len(),
        "PointwiseDAG.output is out of range"
    );

    let input_names = PointwiseDAG::input_binding_names(dag.n_inputs);
    let input_name = |idx: u8| input_names[idx as usize].clone();

    let mut src = String::new();
    src.push_str(
        "struct Params {\n    len: u32,\n    _pad0: u32,\n    _pad1: u32,\n    _pad2: u32,\n}\n\n",
    );
    for name in &input_names {
        let _ = writeln!(src, "var<storage> {}: array<f32>;", name);
    }
    src.push_str("var<storage, read_write> dst: array<f32>;\n");
    src.push_str("var<uniform> params: Params;\n\n");
    let _ = writeln!(src, "@compute @workgroup_size({})", grid.workgroup_size);
    let _ = writeln!(
        src,
        "fn {}(@builtin(global_invocation_id) gid: vec3<u32>) {{",
        POINTWISE_ENTRY
    );
    src.push_str("    let i = gid.x;\n");
    src.push_str("    if i >= params.len { return; }\n");
    src.push_str(&dag.emit_body(input_name));
    let _ = writeln!(src, "    dst[i] = v{};", dag.output);
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

    /// Equivalent of `unary.wgsl`'s `relu`: dst[i] = max(src[i], 0).
    fn relu_dag() -> PointwiseDAG {
        PointwiseDAG {
            n_inputs: 1,
            ops: vec![Pw::LoadInput(0), Pw::Relu(0)],
            output: 1,
        }
    }

    /// Equivalent of `binary.wgsl`'s `add`: dst[i] = src_a[i] + src_b[i].
    fn add_dag() -> PointwiseDAG {
        PointwiseDAG {
            n_inputs: 2,
            ops: vec![Pw::LoadInput(0), Pw::LoadInput(1), Pw::Add(0, 1)],
            output: 2,
        }
    }

    /// Equivalent of `binary.wgsl`'s `swiglu`: silu(src_a[i]) * src_b[i].
    fn swiglu_dag() -> PointwiseDAG {
        PointwiseDAG {
            n_inputs: 2,
            ops: vec![
                Pw::LoadInput(0),
                Pw::LoadInput(1),
                Pw::Silu(0),
                Pw::Mul(2, 1),
            ],
            output: 3,
        }
    }

    #[test]
    fn unary_uses_existing_binding_names() {
        let sm = lower(&KernelTemplate::Pointwise {
            dag: relu_dag(),
            grid: GridShape::default(),
        });
        // Match unary.wgsl so UnaryData layout can bind the generated shader.
        assert!(sm.source.contains("var<storage> src: array<f32>;"));
        assert!(
            sm.source
                .contains("var<storage, read_write> dst: array<f32>;")
        );
        assert!(sm.source.contains("fn main("));
        assert!(sm.source.contains("dst[i] = v1;"));
    }

    #[test]
    fn binary_uses_existing_binding_names() {
        let sm = lower(&KernelTemplate::Pointwise {
            dag: add_dag(),
            grid: GridShape::default(),
        });
        // Match binary.wgsl so BinaryData layout can bind the generated shader.
        assert!(sm.source.contains("var<storage> src_a: array<f32>;"));
        assert!(sm.source.contains("var<storage> src_b: array<f32>;"));
        assert!(sm.source.contains("dst[i] = v2;"));
    }

    #[test]
    fn swiglu_body_is_well_formed() {
        let sm = lower(&KernelTemplate::Pointwise {
            dag: swiglu_dag(),
            grid: GridShape::default(),
        });
        assert!(sm.source.contains("1.0 + exp(-v0)"));
        assert!(sm.source.contains("v2 * v1"));
        assert!(sm.source.contains("dst[i] = v3;"));
    }

    #[test]
    fn three_plus_inputs_use_numbered_bindings() {
        // 3-input DAG: dst[i] = src_0 + src_1 + src_2
        let dag = PointwiseDAG {
            n_inputs: 3,
            ops: vec![
                Pw::LoadInput(0),
                Pw::LoadInput(1),
                Pw::LoadInput(2),
                Pw::Add(0, 1),
                Pw::Add(3, 2),
            ],
            output: 4,
        };
        let sm = lower(&KernelTemplate::Pointwise {
            dag,
            grid: GridShape::default(),
        });
        assert!(sm.source.contains("var<storage> src_0: array<f32>;"));
        assert!(sm.source.contains("var<storage> src_1: array<f32>;"));
        assert!(sm.source.contains("var<storage> src_2: array<f32>;"));
    }

    #[test]
    fn hash_is_stable_and_structural() {
        // Same DAG → same hash.
        assert_eq!(relu_dag().hash_key(), relu_dag().hash_key());
        // Different DAGs → different hashes (not a perfect property, but
        // true for these simple cases).
        assert_ne!(relu_dag().hash_key(), add_dag().hash_key());
        assert_ne!(add_dag().hash_key(), swiglu_dag().hash_key());
    }

    #[test]
    fn const_encoding_roundtrips() {
        let dag = PointwiseDAG {
            n_inputs: 1,
            ops: vec![
                Pw::LoadInput(0),
                Pw::const_f32(0.5),
                Pw::Mul(0, 1), // x * 0.5
            ],
            output: 2,
        };
        let sm = lower(&KernelTemplate::Pointwise {
            dag,
            grid: GridShape::default(),
        });
        assert!(sm.source.contains("0.5f"));
    }

    #[test]
    fn naga_validates_pointwise_module() {
        use naga::valid::{Capabilities, ValidationFlags, Validator};
        // Blade injects bindings at pipeline creation — validate without them,
        // matching the convention used in codegen.rs.
        let flags = ValidationFlags::all() ^ ValidationFlags::BINDINGS;
        let mut v = Validator::new(flags, Capabilities::all());

        for dag in [relu_dag(), add_dag(), swiglu_dag()] {
            let sm = lower(&KernelTemplate::Pointwise {
                dag,
                grid: GridShape::default(),
            });
            v.validate(&sm.module).unwrap_or_else(|e| {
                panic!("naga validation failed: {:?}\n{}", e, sm.source);
            });
        }
    }
}
