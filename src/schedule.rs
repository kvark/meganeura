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

    /// Emit the DAG body as WGSL statements. `load(idx)` returns the
    /// complete WGSL expression that loads input stream `idx` at the
    /// current element position — typically something like
    /// `"src_a[i]"` for pointwise or `"src[row * inner + col]"` for
    /// reduction prologues. The indent and `let v{k} = ` prefix are
    /// emitted by this method.
    fn emit_body(&self, load: impl Fn(u8) -> String) -> String {
        let mut out = String::new();
        for (k, op) in self.ops.iter().enumerate() {
            let _ = write!(out, "    let v{} = ", k);
            match *op {
                Pw::LoadInput(idx) => {
                    let _ = write!(out, "{}", load(idx));
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
    ///   - n=1 → ["src"]                        (matches `UnaryData`)
    ///   - n=2 → ["src_a", "src_b"]             (matches `BinaryData`)
    ///   - n=3 → ["src_a", "src_b", "src_c"]    (matches `TernaryData`)
    ///
    /// Panics for n>3 — the runtime does not currently plumb a wider
    /// data layout (see `runtime::pointwise_data_layout`). The fusion
    /// pass caps output arity at 3.
    fn input_binding_names(n: u8) -> Vec<String> {
        match n {
            1 => vec!["src".to_string()],
            2 => vec!["src_a".to_string(), "src_b".to_string()],
            3 => vec![
                "src_a".to_string(),
                "src_b".to_string(),
                "src_c".to_string(),
            ],
            n => panic!("PointwiseDAG arity {} has no binding layout", n),
        }
    }

    /// Fuse a `producer` DAG into one of this DAG's input streams.
    ///
    /// Returns a new DAG equivalent to "whatever `self` does, but wherever
    /// it loaded input `consumer_input_idx`, substitute the output of
    /// `producer` instead". The new DAG's inputs are `producer`'s inputs
    /// followed by `self`'s remaining inputs (all except the fused one),
    /// in original order. Total inputs: `producer.n_inputs + self.n_inputs - 1`.
    ///
    /// Panics if `consumer_input_idx >= self.n_inputs`.
    pub fn fuse_input(&self, consumer_input_idx: u8, producer: &PointwiseDAG) -> PointwiseDAG {
        assert!(
            consumer_input_idx < self.n_inputs,
            "fuse_input: index {} out of range (n_inputs={})",
            consumer_input_idx,
            self.n_inputs
        );

        // Start the merged op list with producer's ops verbatim; their
        // internal indices are already correct since they reference only
        // earlier producer ops.
        let mut ops = producer.ops.clone();

        // Map each of self's op indices to its position in `ops` after
        // remapping. A LoadInput matching the fused input isn't added as
        // a new op — it aliases directly to `producer.output`.
        let mut self_remap: Vec<u16> = Vec::with_capacity(self.ops.len());

        for op in &self.ops {
            if let Pw::LoadInput(j) = *op {
                if j == consumer_input_idx {
                    self_remap.push(producer.output);
                    continue;
                }
            }
            let remapped = match *op {
                Pw::LoadInput(j) => {
                    let new_j = if j < consumer_input_idx {
                        producer.n_inputs + j
                    } else {
                        producer.n_inputs + j - 1
                    };
                    Pw::LoadInput(new_j)
                }
                Pw::Const(bits) => Pw::Const(bits),
                Pw::Add(a, b) => Pw::Add(self_remap[a as usize], self_remap[b as usize]),
                Pw::Mul(a, b) => Pw::Mul(self_remap[a as usize], self_remap[b as usize]),
                Pw::Sub(a, b) => Pw::Sub(self_remap[a as usize], self_remap[b as usize]),
                Pw::Div(a, b) => Pw::Div(self_remap[a as usize], self_remap[b as usize]),
                Pw::Greater(a, b) => Pw::Greater(self_remap[a as usize], self_remap[b as usize]),
                Pw::Neg(a) => Pw::Neg(self_remap[a as usize]),
                Pw::Recip(a) => Pw::Recip(self_remap[a as usize]),
                Pw::Exp(a) => Pw::Exp(self_remap[a as usize]),
                Pw::Log(a) => Pw::Log(self_remap[a as usize]),
                Pw::Abs(a) => Pw::Abs(self_remap[a as usize]),
                Pw::Sqrt(a) => Pw::Sqrt(self_remap[a as usize]),
                Pw::Rsqrt(a) => Pw::Rsqrt(self_remap[a as usize]),
                Pw::Relu(a) => Pw::Relu(self_remap[a as usize]),
                Pw::Sigmoid(a) => Pw::Sigmoid(self_remap[a as usize]),
                Pw::Silu(a) => Pw::Silu(self_remap[a as usize]),
                Pw::Tanh(a) => Pw::Tanh(self_remap[a as usize]),
            };
            self_remap.push(ops.len() as u16);
            ops.push(remapped);
        }

        PointwiseDAG {
            n_inputs: producer.n_inputs + self.n_inputs - 1,
            output: self_remap[self.output as usize],
            ops,
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

/// Associative reduction ops supported by [`KernelTemplate::Reduction`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReduceOp {
    /// Sum: identity 0, accumulator `acc + v`.
    Sum,
    /// Max: identity -inf, accumulator `max(acc, v)`.
    Max,
}

impl ReduceOp {
    fn identity_wgsl(self) -> &'static str {
        match self {
            ReduceOp::Sum => "0.0f",
            ReduceOp::Max => "bitcast<f32>(0xff800000u)", // -inf
        }
    }

    /// Emit WGSL for `acc = combine(acc, v)`.
    fn combine_wgsl(self, acc: &str, v: &str) -> String {
        match self {
            ReduceOp::Sum => format!("{} = {} + {};", acc, acc, v),
            ReduceOp::Max => format!("{} = max({}, {});", acc, acc, v),
        }
    }
}

/// A schedule template — to be lowered to a Naga [`naga::Module`].
#[derive(Clone, Debug)]
pub enum KernelTemplate {
    Pointwise {
        dag: PointwiseDAG,
        grid: GridShape,
    },
    /// Per-row reduction: one workgroup per outer row reduces along the
    /// inner axis. The per-element `prologue` DAG runs first (producing
    /// the scalar contribution to reduce); the tree reduction follows.
    ///
    /// Without an epilogue (`epilogue = None`) the output shape is
    /// `[outer]` — one scalar per row. With an epilogue, the kernel
    /// writes back a transformed value for every element, producing
    /// `[outer, inner]` (the RMSNorm / Softmax / LayerNorm shape).
    ///
    /// Binding names in the generated module:
    ///   - Prologue input streams: same names as pointwise (`src`,
    ///     `src_a`/`src_b`, or `src_a`/`src_b`/`src_c`). Loaded at
    ///     `row * inner + col`.
    ///   - Per-column broadcast inputs: `bias` (1), or `bias_a`/`bias_b`
    ///     (2). Loaded at `col`.
    ///   - `dst: array<f32>` of length `outer` (no epilogue) or
    ///     `outer * inner` (with epilogue).
    ///   - Uniforms: `outer: u32, inner: u32, _pad0: u32, _pad1: u32`.
    Reduction {
        op: ReduceOp,
        /// Per-element transform applied before accumulation. `n_inputs`
        /// determines the number of input streams; `output` is the scalar
        /// contribution to reduce.
        prologue: PointwiseDAG,
        /// Optional per-element map applied after the reduction.
        epilogue: Option<ReductionEpilogue>,
        /// Workgroup size (threads per row). Must be a power of 2 — the
        /// tree reduction assumes it.
        grid: GridShape,
    },
}

/// Per-element write-back DAG for a reduce-then-map reduction.
///
/// The DAG's input layout (by `LoadInput(idx)`):
///   * `0..prologue.n_inputs` — the same per-element streams the
///     prologue sees, loaded at `[row * inner + col]`.
///   * `prologue.n_inputs` — the reduced scalar for this row (broadcast).
///   * `prologue.n_inputs + 1..` — per-column broadcast inputs loaded
///     at `[col]`. `n_per_col_inputs` counts these.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct ReductionEpilogue {
    pub dag: PointwiseDAG,
    pub n_per_col_inputs: u8,
}

/// Entry-point name emitted for any `KernelTemplate`. One entry point per
/// generated module.
pub const POINTWISE_ENTRY: &str = "main";
pub const REDUCTION_ENTRY: &str = "main";

/// Lower a [`KernelTemplate`] to WGSL + a parsed Naga module.
pub fn lower(t: &KernelTemplate) -> ShaderModule {
    match *t {
        KernelTemplate::Pointwise { ref dag, grid } => lower_pointwise(dag, grid),
        KernelTemplate::Reduction {
            op,
            ref prologue,
            ref epilogue,
            grid,
        } => lower_reduction(op, prologue, epilogue.as_ref(), grid),
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
    src.push_str(&dag.emit_body(|idx| format!("{}[i]", input_names[idx as usize])));
    let _ = writeln!(src, "    dst[i] = v{};", dag.output);
    src.push_str("}\n");

    let module = naga::front::wgsl::parse_str(&src)
        .unwrap_or_else(|e| panic!("generated WGSL failed to parse:\n{}\n---\n{}", e, src));
    ShaderModule {
        module,
        source: src,
    }
}

fn lower_reduction(
    op: ReduceOp,
    prologue: &PointwiseDAG,
    epilogue: Option<&ReductionEpilogue>,
    grid: GridShape,
) -> ShaderModule {
    assert!(
        prologue.n_inputs >= 1,
        "reduction prologue must have at least one input stream"
    );
    assert!(
        (prologue.output as usize) < prologue.ops.len(),
        "reduction prologue output index is out of range"
    );
    assert!(
        grid.workgroup_size.is_power_of_two() && grid.workgroup_size >= 2,
        "reduction workgroup_size must be a power of 2 ≥ 2"
    );
    if let Some(epi) = epilogue {
        let expected = prologue.n_inputs + 1 + epi.n_per_col_inputs;
        assert_eq!(
            epi.dag.n_inputs, expected,
            "reduction epilogue expects {} inputs ({} per-elem + 1 reduced + {} per-col), got {}",
            expected, prologue.n_inputs, epi.n_per_col_inputs, epi.dag.n_inputs,
        );
        assert!(
            (epi.dag.output as usize) < epi.dag.ops.len(),
            "reduction epilogue output index is out of range"
        );
    }

    let input_names = PointwiseDAG::input_binding_names(prologue.n_inputs);
    let per_col_names: Vec<String> = epilogue
        .map(|epi| per_col_binding_names(epi.n_per_col_inputs))
        .unwrap_or_default();

    let mut src = String::new();
    src.push_str(
        "struct Params {\n    outer: u32,\n    inner: u32,\n    _pad0: u32,\n    _pad1: u32,\n}\n\n",
    );
    for name in &input_names {
        let _ = writeln!(src, "var<storage> {}: array<f32>;", name);
    }
    for name in &per_col_names {
        let _ = writeln!(src, "var<storage> {}: array<f32>;", name);
    }
    src.push_str("var<storage, read_write> dst: array<f32>;\n");
    src.push_str("var<uniform> params: Params;\n");
    let wg = grid.workgroup_size;
    let _ = writeln!(src, "var<workgroup> wg_data: array<f32, {}>;\n", wg);
    let _ = writeln!(src, "@compute @workgroup_size({})", wg);
    let _ = writeln!(
        src,
        "fn {}(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{",
        REDUCTION_ENTRY
    );
    src.push_str("    let row = wgid.x;\n");
    src.push_str("    let tid = lid.x;\n");
    src.push_str("    if row >= params.outer { return; }\n");
    src.push_str("    let row_offset = row * params.inner;\n");
    let _ = writeln!(src, "    var acc: f32 = {};", op.identity_wgsl());
    src.push_str("    var col = tid;\n");
    src.push_str("    loop {\n");
    src.push_str("        if col >= params.inner { break; }\n");
    // Prologue: per-element DAG; loads read from `src[row_offset + col]`.
    let body = prologue.emit_body(|idx| format!("{}[row_offset + col]", input_names[idx as usize]));
    // Indent each line of the prologue body by 4 extra spaces for the loop.
    for line in body.lines() {
        src.push_str("        ");
        src.push_str(line.trim_start_matches(' '));
        src.push('\n');
    }
    let _ = writeln!(
        src,
        "        {}",
        op.combine_wgsl("acc", &format!("v{}", prologue.output))
    );
    let _ = writeln!(src, "        col += {}u;", wg);
    src.push_str("    }\n");
    src.push_str("    wg_data[tid] = acc;\n");
    src.push_str("    workgroupBarrier();\n");
    // Tree reduction.
    let mut stride = wg / 2;
    src.push_str("    // tree reduction\n");
    while stride > 0 {
        let _ = writeln!(src, "    if tid < {}u {{", stride);
        let combined = op.combine_wgsl("wg_data[tid]", &format!("wg_data[tid + {}u]", stride));
        let _ = writeln!(src, "        {}", combined);
        src.push_str("    }\n");
        src.push_str("    workgroupBarrier();\n");
        stride /= 2;
    }
    match epilogue {
        None => {
            src.push_str("    if tid == 0u { dst[row] = wg_data[0]; }\n");
        }
        Some(epi) => {
            // Reduce-then-map: broadcast the reduced value and write a
            // transformed element back for every col.
            src.push_str("    let reduced_value = wg_data[0];\n");
            src.push_str("    var wcol = tid;\n");
            src.push_str("    loop {\n");
            src.push_str("        if wcol >= params.inner { break; }\n");
            let n_per_elem = prologue.n_inputs as usize;
            let body = epi.dag.emit_body(|idx| {
                let i = idx as usize;
                if i < n_per_elem {
                    format!("{}[row_offset + wcol]", input_names[i])
                } else if i == n_per_elem {
                    "reduced_value".to_string()
                } else {
                    let k = i - (n_per_elem + 1);
                    format!("{}[wcol]", per_col_names[k])
                }
            });
            for line in body.lines() {
                src.push_str("        ");
                src.push_str(line.trim_start_matches(' '));
                src.push('\n');
            }
            let _ = writeln!(src, "        dst[row_offset + wcol] = v{};", epi.dag.output);
            let _ = writeln!(src, "        wcol += {}u;", wg);
            src.push_str("    }\n");
        }
    }
    src.push_str("}\n");

    let module = naga::front::wgsl::parse_str(&src).unwrap_or_else(|e| {
        panic!(
            "generated reduction WGSL failed to parse:\n{}\n---\n{}",
            e, src
        )
    });
    ShaderModule {
        module,
        source: src,
    }
}

/// Binding names for per-column broadcast inputs used by a reduction
/// epilogue:
///   * n=0 → `[]`
///   * n=1 → `["bias"]`      (matches the existing `RmsNormData` layout)
///   * n=2 → `["bias_a", "bias_b"]`
///
/// Panics for n > 2 — wider layouts need additional runtime plumbing.
fn per_col_binding_names(n: u8) -> Vec<String> {
    match n {
        0 => Vec::new(),
        1 => vec!["bias".to_string()],
        2 => vec!["bias_a".to_string(), "bias_b".to_string()],
        n => panic!(
            "reduction epilogue n_per_col_inputs={} has no binding layout (max 2)",
            n
        ),
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
    fn ternary_uses_abc_binding_names() {
        // 3-input DAG: dst[i] = src_a + src_b + src_c — must use abc
        // naming to match the runtime's TernaryData layout.
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
        assert!(sm.source.contains("var<storage> src_a: array<f32>;"));
        assert!(sm.source.contains("var<storage> src_b: array<f32>;"));
        assert!(sm.source.contains("var<storage> src_c: array<f32>;"));
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
    fn fuse_relu_into_add_consumer_replaces_input_a() {
        // consumer: dst = src_a + src_b          (Add)
        // producer: src = relu(src)              (Relu)
        // fused into consumer's input 0:
        //   dst = relu(src_a_new) + src_b_new
        // where src_a_new = original producer input, src_b_new = consumer's
        // remaining input (original consumer input 1).
        let consumer = add_dag();
        let producer = relu_dag();
        let fused = consumer.fuse_input(0, &producer);

        assert_eq!(fused.n_inputs, 2); // 1 (producer) + 2 (consumer) - 1
        let sm = lower(&KernelTemplate::Pointwise {
            dag: fused.clone(),
            grid: GridShape::default(),
        });
        // Binding names collapse back to src_a / src_b for n=2.
        assert!(sm.source.contains("var<storage> src_a: array<f32>;"));
        assert!(sm.source.contains("var<storage> src_b: array<f32>;"));
        // The fused op list: producer's [LoadInput(0), Relu(0)] followed by
        // consumer's [LoadInput(1 remapped), Add(relu_out, other)].
        // Concretely: ops = [LoadInput(0), Relu(0), LoadInput(1), Add(1, 2)].
        assert_eq!(
            fused.ops,
            vec![
                Pw::LoadInput(0),
                Pw::Relu(0),
                Pw::LoadInput(1),
                Pw::Add(1, 2),
            ]
        );
        assert_eq!(fused.output, 3);
    }

    #[test]
    fn fuse_relu_into_add_consumer_input_b() {
        // Fuse relu into the *second* input of Add.
        // Expected: ops = [LoadInput(0)_prod, Relu(0), LoadInput(1)_cons_a, Add(2, 1)]
        // with consumer's input 0 remapped from idx 0 → LoadInput(1).
        let consumer = add_dag();
        let producer = relu_dag();
        let fused = consumer.fuse_input(1, &producer);

        assert_eq!(fused.n_inputs, 2);
        assert_eq!(
            fused.ops,
            vec![
                Pw::LoadInput(0),
                Pw::Relu(0),
                Pw::LoadInput(1),
                Pw::Add(2, 1),
            ]
        );
        assert_eq!(fused.output, 3);
    }

    #[test]
    fn fuse_chain_produces_valid_wgsl() {
        // (relu(x) fused into neg) -> neg(relu(x))
        let producer = relu_dag();
        let consumer = PointwiseDAG {
            n_inputs: 1,
            ops: vec![Pw::LoadInput(0), Pw::Neg(0)],
            output: 1,
        };
        let fused = consumer.fuse_input(0, &producer);
        assert_eq!(fused.n_inputs, 1);

        let sm = lower(&KernelTemplate::Pointwise {
            dag: fused,
            grid: GridShape::default(),
        });
        assert!(sm.source.contains("max(v0, 0.0)"));
        assert!(sm.source.contains("-v1"));

        use naga::valid::{Capabilities, ValidationFlags, Validator};
        let flags = ValidationFlags::all() ^ ValidationFlags::BINDINGS;
        Validator::new(flags, Capabilities::all())
            .validate(&sm.module)
            .unwrap_or_else(|e| panic!("fused DAG invalid: {:?}\n{}", e, sm.source));
    }

    // ---- Reduction archetype ----

    /// Simplest reduction: sum over last axis. Prologue is identity
    /// (pass-through of the one input).
    fn identity_prologue() -> PointwiseDAG {
        PointwiseDAG {
            n_inputs: 1,
            ops: vec![Pw::LoadInput(0)],
            output: 0,
        }
    }

    /// Prologue that squares the input — contribution for sum-of-squares.
    fn square_prologue() -> PointwiseDAG {
        PointwiseDAG {
            n_inputs: 1,
            ops: vec![Pw::LoadInput(0), Pw::Mul(0, 0)],
            output: 1,
        }
    }

    #[test]
    fn reduction_lowers_to_valid_wgsl() {
        let sm = lower(&KernelTemplate::Reduction {
            op: ReduceOp::Sum,
            prologue: identity_prologue(),
            epilogue: None,
            grid: GridShape::default(),
        });
        assert!(sm.source.contains("struct Params"));
        assert!(sm.source.contains("outer: u32"));
        assert!(sm.source.contains("var<workgroup> wg_data"));
        assert!(sm.source.contains("workgroupBarrier"));
        // Identity for sum is 0.
        assert!(sm.source.contains("var acc: f32 = 0.0f"));
        // Dst writes one scalar per row.
        assert!(sm.source.contains("dst[row] = wg_data[0]"));
    }

    #[test]
    fn max_reduction_uses_max_combiner() {
        let sm = lower(&KernelTemplate::Reduction {
            op: ReduceOp::Max,
            prologue: identity_prologue(),
            epilogue: None,
            grid: GridShape::default(),
        });
        // Max identity is -inf via bitcast.
        assert!(sm.source.contains("0xff800000u"));
        // Combiner uses max().
        assert!(sm.source.contains("max(acc,"));
    }

    #[test]
    fn reduction_with_square_prologue_compiles() {
        // Sum of squares — the reduction half of RMSNorm.
        let sm = lower(&KernelTemplate::Reduction {
            op: ReduceOp::Sum,
            prologue: square_prologue(),
            epilogue: None,
            grid: GridShape::default(),
        });
        // Prologue emits the multiply.
        assert!(sm.source.contains("v0 * v0"));
        // And the per-row load expression should reference row_offset.
        assert!(sm.source.contains("src[row_offset + col]"));
    }

    #[test]
    fn reduction_naga_validates() {
        use naga::valid::{Capabilities, ValidationFlags, Validator};
        let flags = ValidationFlags::all() ^ ValidationFlags::BINDINGS;
        let mut v = Validator::new(flags, Capabilities::all());

        for (op, prologue) in [
            (ReduceOp::Sum, identity_prologue()),
            (ReduceOp::Sum, square_prologue()),
            (ReduceOp::Max, identity_prologue()),
        ] {
            let sm = lower(&KernelTemplate::Reduction {
                op,
                prologue,
                epilogue: None,
                grid: GridShape::default(),
            });
            v.validate(&sm.module).unwrap_or_else(|e| {
                panic!("reduction naga validation failed: {:?}\n{}", e, sm.source);
            });
        }
    }

    /// Build the RMSNorm epilogue DAG. Inputs are:
    ///   0 = src[row*inner + col]        (per-element)
    ///   1 = sum_of_squares (broadcast scalar)
    ///   2 = bias[col]                    (per-col)
    /// Output: `src * rsqrt(sum_of_squares * inv_inner + eps) * bias`.
    /// The inverse row length is baked as a Const — it's static at graph
    /// compile time.
    fn rmsnorm_epilogue(inner: f32, eps: f32) -> ReductionEpilogue {
        let inv_inner = Pw::const_f32(1.0 / inner);
        let eps_c = Pw::const_f32(eps);
        let dag = PointwiseDAG {
            n_inputs: 3,
            ops: vec![
                Pw::LoadInput(0), // v0 = src[row,col]
                Pw::LoadInput(1), // v1 = sum_of_squares
                Pw::LoadInput(2), // v2 = bias[col]
                inv_inner,        // v3 = 1/inner
                eps_c,            // v4 = eps
                Pw::Mul(1, 3),    // v5 = mean_sq = sum_sq * inv_inner
                Pw::Add(5, 4),    // v6 = mean_sq + eps
                Pw::Rsqrt(6),     // v7 = rsqrt(...)
                Pw::Mul(0, 7),    // v8 = src * rsqrt
                Pw::Mul(8, 2),    // v9 = (src * rsqrt) * bias
            ],
            output: 9,
        };
        ReductionEpilogue {
            dag,
            n_per_col_inputs: 1,
        }
    }

    #[test]
    fn rmsnorm_reduction_lowers() {
        // Prologue: v*v. Op: Sum. Epilogue: the RMSNorm transform above.
        let sm = lower(&KernelTemplate::Reduction {
            op: ReduceOp::Sum,
            prologue: square_prologue(),
            epilogue: Some(rmsnorm_epilogue(720.0, 1e-5)),
            grid: GridShape::default(),
        });
        // Has the per-col bias binding.
        assert!(sm.source.contains("var<storage> bias: array<f32>;"));
        // Per-element writeback instead of per-row.
        assert!(sm.source.contains("dst[row_offset + wcol]"));
        // Reduced value is read into a local.
        assert!(sm.source.contains("let reduced_value = wg_data[0];"));
        // Per-col load uses wcol.
        assert!(sm.source.contains("bias[wcol]"));
        // Rsqrt appears in generated epilogue.
        assert!(sm.source.contains("inverseSqrt"));
    }

    #[test]
    fn reduction_with_epilogue_naga_validates() {
        use naga::valid::{Capabilities, ValidationFlags, Validator};
        let flags = ValidationFlags::all() ^ ValidationFlags::BINDINGS;
        let sm = lower(&KernelTemplate::Reduction {
            op: ReduceOp::Sum,
            prologue: square_prologue(),
            epilogue: Some(rmsnorm_epilogue(720.0, 1e-5)),
            grid: GridShape::default(),
        });
        Validator::new(flags, Capabilities::all())
            .validate(&sm.module)
            .unwrap_or_else(|e| panic!("RMSNorm reduction invalid: {:?}\n{}", e, sm.source));
    }

    #[test]
    #[should_panic(expected = "reduction epilogue expects")]
    fn reduction_rejects_mismatched_epilogue_arity() {
        // square_prologue has n_inputs=1. A correct epilogue must have
        // n_inputs = 1 + 1 + n_per_col_inputs. Here we pass n_inputs=2
        // with 1 per-col (expected is 3).
        let bad = ReductionEpilogue {
            dag: PointwiseDAG {
                n_inputs: 2,
                ops: vec![Pw::LoadInput(0), Pw::LoadInput(1), Pw::Add(0, 1)],
                output: 2,
            },
            n_per_col_inputs: 1,
        };
        lower(&KernelTemplate::Reduction {
            op: ReduceOp::Sum,
            prologue: square_prologue(),
            epilogue: Some(bad),
            grid: GridShape::default(),
        });
    }

    #[test]
    #[should_panic(expected = "workgroup_size must be a power of 2")]
    fn reduction_rejects_odd_workgroup_size() {
        lower(&KernelTemplate::Reduction {
            op: ReduceOp::Sum,
            prologue: identity_prologue(),
            epilogue: None,
            grid: GridShape {
                workgroup_size: 250, // not power of 2
            },
        });
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
