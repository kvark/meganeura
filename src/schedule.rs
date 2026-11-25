//! Kernel schedules that emit WGSL for Blade to compile.
//!
//! `PointwiseDAG` describes fused scalar expressions shared by standalone
//! pointwise/reduction kernels and matmul epilogues.

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
    /// Load input stream `input` at `(i / divisor) % modulus`, where `i` is
    /// the flat element index: a per-channel bias over NCHW is
    /// `divisor = H*W, modulus = C`. Only flat pointwise kernels can emit
    /// it; see [`PointwiseDAG::has_broadcast`].
    LoadBroadcast {
        input: u8,
        divisor: u32,
        modulus: u32,
    },
    /// Literal f32 constant. Stored as bit pattern so `Hash`/`Eq` are stable.
    Const(u32),
    // Binary ops — reference earlier value-node indices.
    Add(u16, u16),
    Mul(u16, u16),
    Sub(u16, u16),
    Div(u16, u16),
    Min(u16, u16),
    Max(u16, u16),
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
    /// The error function, to about 1.2e-7 relative accuracy.
    Erf(u16),
    Sin(u16),
    Cos(u16),
    /// Reinterpret a loaded word as `u32` and convert it to `f32`.
    U32ToF32(u16),
    /// Truncate a non-negative value to `u32` and store its bits.
    F32ToU32(u16),
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

    /// Evaluate a single-input (`n_inputs == 1`, non-broadcast) DAG in `f64`
    /// for CPU reference qualification (e.g. unary matmul epilogue tuning).
    pub fn eval_unary_f64(&self, input: f64) -> f64 {
        assert_eq!(self.n_inputs, 1, "eval_unary_f64 requires n_inputs == 1");
        let mut vals: Vec<f64> = Vec::with_capacity(self.ops.len());
        for op in &self.ops {
            let v = match *op {
                Pw::LoadInput(0) => input,
                Pw::LoadInput(_) | Pw::LoadBroadcast { .. } => {
                    unreachable!("non-unary load in eval_unary_f64")
                }
                Pw::Const(bits) => f64::from(f32::from_bits(bits)),
                Pw::Add(a, b) => vals[a as usize] + vals[b as usize],
                Pw::Mul(a, b) => vals[a as usize] * vals[b as usize],
                Pw::Sub(a, b) => vals[a as usize] - vals[b as usize],
                Pw::Div(a, b) => vals[a as usize] / vals[b as usize],
                Pw::Min(a, b) => vals[a as usize].min(vals[b as usize]),
                Pw::Max(a, b) => vals[a as usize].max(vals[b as usize]),
                Pw::Greater(a, b) => {
                    if vals[a as usize] > vals[b as usize] {
                        1.0
                    } else {
                        0.0
                    }
                }
                Pw::Neg(a) => -vals[a as usize],
                Pw::Recip(a) => 1.0 / vals[a as usize],
                Pw::Exp(a) => vals[a as usize].exp(),
                Pw::Log(a) => vals[a as usize].ln(),
                Pw::Abs(a) => vals[a as usize].abs(),
                Pw::Sqrt(a) => vals[a as usize].sqrt(),
                Pw::Rsqrt(a) => 1.0 / vals[a as usize].sqrt(),
                Pw::Relu(a) => vals[a as usize].max(0.0),
                Pw::Sigmoid(a) => 1.0 / (1.0 + (-vals[a as usize]).exp()),
                Pw::Silu(a) => {
                    let x = vals[a as usize];
                    x / (1.0 + (-x).exp())
                }
                Pw::Tanh(a) => vals[a as usize].tanh(),
                Pw::Erf(a) => {
                    let x = vals[a as usize];
                    let z = x.abs();
                    let s = x * x;
                    if z < 0.5 {
                        const POLY: [f64; 6] = [
                            std::f64::consts::FRAC_2_SQRT_PI,
                            -0.37612639,
                            0.11283792,
                            -0.026866172,
                            0.0052239776,
                            -0.00085483270,
                        ];
                        let mut p = 0.0;
                        for &c in POLY.iter().rev() {
                            p = p * s + c;
                        }
                        x * p
                    } else {
                        const CHEB: [f64; 10] = [
                            -1.26551223,
                            1.00002368,
                            0.37409196,
                            0.09678418,
                            -0.18628806,
                            0.27886807,
                            -1.13520398,
                            1.48851587,
                            -0.82215223,
                            0.17087277,
                        ];
                        let t = 1.0 / (1.0 + 0.5 * z);
                        let mut poly = 0.0;
                        for &c in CHEB.iter().rev() {
                            poly = poly * t + c;
                        }
                        let c = t * (-s + poly).exp();
                        x.signum() * (1.0 - c)
                    }
                }
                Pw::Sin(a) => vals[a as usize].sin(),
                Pw::Cos(a) => vals[a as usize].cos(),
                Pw::U32ToF32(a) => f64::from((vals[a as usize] as f32).to_bits() as f32),
                Pw::F32ToU32(a) => f64::from(f32::from_bits((vals[a as usize] as f32) as u32)),
            };
            vals.push(v);
        }
        vals[self.output as usize]
    }

    /// Emit the DAG body as WGSL statements. `load(idx)` returns the
    /// complete WGSL expression that loads input stream `idx` at the
    /// current element position — typically something like
    /// `"src_a[i]"` for pointwise or `"src[row * inner + col]"` for
    /// reduction prologues. The indent and `let v{k} = ` prefix are
    /// emitted by this method.
    pub fn emit_body(&self, load: impl Fn(u8) -> String) -> String {
        self.emit_body_with(load, |_, _, _| {
            panic!("broadcast loads are only valid in flat pointwise kernels")
        })
    }

    /// [`Self::emit_body`] for a kernel that can also address broadcast
    /// loads: `broadcast(idx, divisor, modulus)` returns the load expression.
    pub fn emit_body_with(
        &self,
        load: impl Fn(u8) -> String,
        broadcast: impl Fn(u8, u32, u32) -> String,
    ) -> String {
        self.emit_body_impl(load, broadcast, false)
    }

    fn emit_body_impl(
        &self,
        load: impl Fn(u8) -> String,
        broadcast: impl Fn(u8, u32, u32) -> String,
        vec4: bool,
    ) -> String {
        let lit = |s: &str| -> String {
            if vec4 {
                format!("vec4<f32>({s})")
            } else {
                s.to_string()
            }
        };
        let mut out = String::new();
        for (k, op) in self.ops.iter().enumerate() {
            if let Pw::Erf(a) = *op {
                // A Taylor series near zero, where `1 - erfc` would lose
                // the relative precision, and Numerical Recipes' Chebyshev
                // fit of erfc elsewhere.
                let one = lit("1.0");
                let half = lit("0.5");
                let c0 = lit("-1.26551223");
                let c1 = lit("1.00002368");
                let c2 = lit("0.37409196");
                let c3 = lit("0.09678418");
                let c4 = lit("-0.18628806");
                let c5 = lit("0.27886807");
                let c6 = lit("-1.13520398");
                let c7 = lit("1.48851587");
                let c8 = lit("-0.82215223");
                let p0 = lit("1.1283792");
                let p1 = lit("-0.37612639");
                let p2 = lit("0.11283792");
                let p3 = lit("-0.026866172");
                let p4 = lit("0.0052239776");
                let _ = writeln!(
                    out,
                    "    let v{k}_z = abs(v{a});\n    \
                     let v{k}_s = v{a} * v{a};\n    \
                     let v{k}_t = {one} / ({one} + 0.5 * v{k}_z);\n    \
                     let v{k}_c = v{k}_t * exp(-v{k}_s + {c0} + v{k}_t * ({c1} + \
                     v{k}_t * ({c2} + v{k}_t * ({c3} + v{k}_t * ({c4} + \
                     v{k}_t * ({c5} + v{k}_t * ({c6} + v{k}_t * ({c7} + \
                     v{k}_t * ({c8} + v{k}_t * 0.17087277)))))))));\n    \
                     let v{k}_p = v{a} * ({p0} + v{k}_s * ({p1} + v{k}_s * \
                     ({p2} + v{k}_s * ({p3} + v{k}_s * ({p4} + \
                     v{k}_s * -0.00085483270)))));"
                );
                let _ = writeln!(
                    out,
                    "    let v{k} = select(sign(v{a}) * ({one} - v{k}_c), v{k}_p, v{k}_z < {half});"
                );
                continue;
            }
            let _ = write!(out, "    let v{} = ", k);
            match *op {
                Pw::LoadInput(idx) => {
                    let _ = write!(out, "{}", load(idx));
                }
                Pw::LoadBroadcast {
                    input,
                    divisor,
                    modulus,
                } => {
                    let _ = write!(out, "{}", broadcast(input, divisor, modulus));
                }
                Pw::Const(bits) => {
                    // Reconstruct f32 and emit with a decimal point + `f` suffix
                    // so WGSL parses it as f32 regardless of value.
                    let v = f32::from_bits(bits);
                    let scalar = if v.is_finite() {
                        format!("{v:?}f")
                    } else if v.is_nan() {
                        "bitcast<f32>(0x7fc00000u)".to_string()
                    } else if v.is_sign_positive() {
                        "bitcast<f32>(0x7f800000u)".to_string()
                    } else {
                        "bitcast<f32>(0xff800000u)".to_string()
                    };
                    let _ = write!(out, "{}", lit(&scalar));
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
                Pw::Min(a, b) => {
                    let _ = write!(out, "min(v{}, v{})", a, b);
                }
                Pw::Max(a, b) => {
                    let _ = write!(out, "max(v{}, v{})", a, b);
                }
                Pw::Greater(a, b) => {
                    let _ = write!(
                        out,
                        "select({}, {}, v{} > v{})",
                        lit("0.0"),
                        lit("1.0"),
                        a,
                        b
                    );
                }
                Pw::Neg(a) => {
                    let _ = write!(out, "-v{}", a);
                }
                Pw::Recip(a) => {
                    let _ = write!(out, "{} / v{}", lit("1.0"), a);
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
                    let _ = write!(out, "max(v{}, {})", a, lit("0.0"));
                }
                Pw::Sigmoid(a) => {
                    let _ = write!(out, "{} / ({} + exp(-v{}))", lit("1.0"), lit("1.0"), a);
                }
                Pw::Silu(a) => {
                    let _ = write!(out, "v{} / ({} + exp(-v{}))", a, lit("1.0"), a);
                }
                Pw::Erf(_) => unreachable!("emitted above"),
                Pw::Sin(a) => {
                    let _ = write!(out, "sin(v{a})");
                }
                Pw::Cos(a) => {
                    let _ = write!(out, "cos(v{a})");
                }
                Pw::U32ToF32(a) => {
                    if vec4 {
                        let _ = write!(out, "vec4<f32>(bitcast<vec4<u32>>(v{a}))");
                    } else {
                        let _ = write!(out, "f32(bitcast<u32>(v{a}))");
                    }
                }
                Pw::F32ToU32(a) => {
                    if vec4 {
                        let _ = write!(
                            out,
                            "bitcast<vec4<f32>>(vec4<u32>(max(v{a}, vec4<f32>(0.0))))"
                        );
                    } else {
                        let _ = write!(out, "bitcast<f32>(u32(max(v{a}, 0.0)))");
                    }
                }
                Pw::Tanh(a) => {
                    // Saturated outside ±10 (exactly ±1 in f32): some drivers
                    // evaluate tanh through exp(2x), which overflows to NaN.
                    let _ = write!(
                        out,
                        "select(tanh(v{a}), sign(v{a}), abs(v{a}) > {})",
                        lit("10.0")
                    );
                }
            }
            out.push_str(";\n");
        }
        out
    }

    /// Per-input vectorization classification for a 4-wide (`vec4<f32>`)
    /// pointwise kernel. Returns `Some(is_vec4)` when every input can be
    /// declared either as `array<vec4<f32>>` (`true`: contiguous loads or
    /// `divisor == 1` with `modulus % 4 == 0`) or as scalar `array<f32>`
    /// (`false`: channel broadcasts with `divisor % 4 == 0` where all 4 lanes
    /// read the same element). Returns `None` if storage is narrowed, `len` is
    /// not a positive multiple of 4, or any input has mixed/unaligned access.
    pub fn vec4_input_kinds(&self, storage: PointwiseStorage, len: u32) -> Option<Vec<bool>> {
        if storage != PointwiseStorage::default() || len < 4 || !len.is_multiple_of(4) {
            return None;
        }
        let mut kinds: Vec<Option<bool>> = vec![None; self.n_inputs as usize];
        for op in &self.ops {
            match *op {
                Pw::LoadInput(idx) => {
                    let slot = kinds.get_mut(idx as usize)?;
                    if *slot == Some(false) {
                        return None;
                    }
                    *slot = Some(true);
                }
                Pw::LoadBroadcast {
                    input,
                    divisor,
                    modulus,
                } => {
                    let slot = kinds.get_mut(input as usize)?;
                    let is_vec4 = if divisor == 1 && modulus >= 4 && modulus.is_multiple_of(4) {
                        true
                    } else if divisor >= 4 && divisor.is_multiple_of(4) && modulus > 0 {
                        false
                    } else {
                        return None;
                    };
                    if let Some(prev) = *slot
                        && prev != is_vec4
                    {
                        return None;
                    }
                    *slot = Some(is_vec4);
                }
                _ => {}
            }
        }
        // Require every declared input to be referenced so we never retype an
        // unused short buffer as `array<vec4<f32>>`.
        kinds.into_iter().collect()
    }

    /// Whether this pointwise DAG can safely execute 4 elements per thread via
    /// `vec4<f32>` loads and stores.
    pub fn can_vectorize(&self, storage: PointwiseStorage, len: u32) -> bool {
        self.vec4_input_kinds(storage, len).is_some()
    }

    /// Binding names for each input stream:
    ///   - n=1 → `["src"]`
    ///   - n=2..=MAX_POINTWISE_INPUTS → `["src_a", "src_b", ..., "src_g"][..n]`
    fn input_binding_names(n: u8) -> Vec<&'static str> {
        pointwise_input_binding_names(n)
    }

    /// Whether any input is read through a broadcast index rather than at
    /// the current element. Such DAGs cannot move into reduction prologues
    /// or matmul epilogues, whose element positions are not flat indices.
    pub fn has_broadcast(&self) -> bool {
        self.ops
            .iter()
            .any(|op| matches!(op, Pw::LoadBroadcast { .. }))
    }

    /// Whether input stream `idx` is read through a broadcast index.
    pub fn broadcasts_input(&self, idx: u8) -> bool {
        self.ops
            .iter()
            .any(|op| matches!(*op, Pw::LoadBroadcast { input, .. } if input == idx))
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

        let remap_input = |j: u8| {
            if j < consumer_input_idx {
                producer.n_inputs + j
            } else {
                producer.n_inputs + j - 1
            }
        };
        for op in &self.ops {
            match *op {
                Pw::LoadInput(j) if j == consumer_input_idx => {
                    self_remap.push(producer.output);
                    continue;
                }
                // A producer that reads nothing is a literal, which is the
                // same at every index; anything else is only valid at the
                // consumer's own element.
                Pw::LoadBroadcast { input, .. } if input == consumer_input_idx => {
                    assert_eq!(
                        producer.n_inputs, 0,
                        "fuse_input: cannot substitute a computed value for a broadcast load"
                    );
                    self_remap.push(producer.output);
                    continue;
                }
                _ => {}
            }
            let remapped = match *op {
                Pw::LoadInput(j) => Pw::LoadInput(remap_input(j)),
                Pw::LoadBroadcast {
                    input,
                    divisor,
                    modulus,
                } => Pw::LoadBroadcast {
                    input: remap_input(input),
                    divisor,
                    modulus,
                },
                Pw::Const(bits) => Pw::Const(bits),
                Pw::Add(a, b) => Pw::Add(self_remap[a as usize], self_remap[b as usize]),
                Pw::Mul(a, b) => Pw::Mul(self_remap[a as usize], self_remap[b as usize]),
                Pw::Sub(a, b) => Pw::Sub(self_remap[a as usize], self_remap[b as usize]),
                Pw::Div(a, b) => Pw::Div(self_remap[a as usize], self_remap[b as usize]),
                Pw::Min(a, b) => Pw::Min(self_remap[a as usize], self_remap[b as usize]),
                Pw::Max(a, b) => Pw::Max(self_remap[a as usize], self_remap[b as usize]),
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
                Pw::Erf(a) => Pw::Erf(self_remap[a as usize]),
                Pw::Sin(a) => Pw::Sin(self_remap[a as usize]),
                Pw::Cos(a) => Pw::Cos(self_remap[a as usize]),
                Pw::U32ToF32(a) => Pw::U32ToF32(self_remap[a as usize]),
                Pw::F32ToU32(a) => Pw::F32ToU32(self_remap[a as usize]),
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
// KernelTemplate — the pointwise and reduction archetypes.
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
    /// Index of the first maximum. Accumulates (value, lowest column) pairs
    /// and exposes the winning column as an exact F32 scalar, not the value.
    ArgMax,
}

impl ReduceOp {
    fn identity_wgsl(self) -> &'static str {
        match self {
            ReduceOp::Sum => "0.0f",
            ReduceOp::Max | ReduceOp::ArgMax => "bitcast<f32>(0xff800000u)", // -inf
        }
    }

    /// Emit WGSL for `acc = combine(acc, v)`.
    fn combine_wgsl(self, acc: &str, v: &str) -> String {
        match self {
            // Use += compound assignment — NVIDIA's JIT optimizes it
            // measurably better than explicit `x = x + y` form.
            ReduceOp::Sum => format!("{} += {};", acc, v),
            ReduceOp::Max => format!("{} = max({}, {});", acc, acc, v),
            ReduceOp::ArgMax => unreachable!("argmax combines value/index pairs"),
        }
    }
}

/// Emit WGSL keeping the larger value, or the lower column of equal values.
/// An empty accumulator (column `0xffffffff`) takes any nonempty candidate
/// and an empty candidate never wins: comparisons with the infinite identity
/// are not reliable once a shader compiler assumes finite math.
fn combine_argmax(acc: &str, index: &str, value: &str, column: &str) -> String {
    format!(
        "if {column} != 0xffffffffu && ({index} == 0xffffffffu || {value} > {acc} || ({value} == {acc} && {column} < {index})) {{ {acc} = {value}; {index} = {column}; }}"
    )
}

/// The reduced scalar of accumulator `k` at workgroup slot `slot`.
fn reduced_value(op: ReduceOp, k: usize, slot: &str) -> String {
    if op == ReduceOp::ArgMax {
        format!("f32(wg_index_{k}[{slot}])")
    } else {
        format!("wg_data_{k}[{slot}]")
    }
}

/// A schedule template — to be lowered to WGSL source.
#[derive(Clone, Debug)]
pub enum KernelTemplate {
    Pointwise {
        dag: PointwiseDAG,
        grid: GridShape,
    },
    /// Per-row reduction. One workgroup normally reduces one outer row;
    /// narrow reductions may pack several rows into one workgroup. Uniform
    /// input layout is shared by the prologue (pre-reduction per-element
    /// transform) and the optional epilogue (post-reduction per-element map).
    ///
    /// **Input categories**, addressed by `LoadInput(idx)` in the DAGs:
    ///   * **Per-element** `[row * inner + col]` — `n_per_elem` streams.
    ///     Indices `0..n_per_elem`. Binding names match `PointwiseDAG`:
    ///     `src` / `src_a`+`src_b` / `src_a`+`src_b`+`src_c`.
    ///   * **Per-row** `[row]` — `n_per_row` streams. Indices
    ///     `n_per_elem..n_per_elem+n_per_row`. Binding names:
    ///     `per_row_src` (n=1) / `per_row_src_a`+`per_row_src_b` (n=2).
    ///   * **Per-col** `[col]` — `n_per_col` streams, **epilogue only**.
    ///     Indices `n_per_elem+n_per_row..n_per_elem+n_per_row+n_per_col`.
    ///     Binding names: `bias` (n=1) / `bias_a`+`bias_b` (n=2).
    ///   * **Reduced scalar** — the one value produced by the tree
    ///     reduction. **Epilogue only**, at index
    ///     `n_per_elem + n_per_row + n_per_col` (always last).
    ///
    /// Required prologue arity: `n_per_elem + n_per_row`.
    /// Required epilogue arity: `n_per_elem + n_per_row + n_per_col + 1`.
    ///
    /// Without an epilogue the output shape is `[outer]` (one scalar per
    /// row); with an epilogue it is `[outer, inner]`. Uniforms are
    /// `outer: u32, inner: u32, _pad0: u32, _pad1: u32`.
    Reduction {
        op: ReduceOp,
        prologue: PointwiseDAG,
        /// Extra accumulators (see [`ReductionKernel::extra_prologues`]).
        extra_prologues: Vec<PointwiseDAG>,
        epilogue: Option<ReductionEpilogue>,
        /// Number of per-element input streams. Used to split the
        /// prologue/epilogue DAG input-index range into per-elem vs
        /// per-row (see enum-level docs).
        n_per_elem: u8,
        /// Number of per-row broadcast input streams. Loaded at `[row]`.
        n_per_row: u8,
        /// Per-element gather flags (see [`ReductionKernel::gather_elem`]).
        /// Empty = all direct.
        gather_elem: Vec<bool>,
        /// Row-repeat factors for per-element streams (see
        /// [`ReductionKernel::input_row_repeats`]). Empty = all one.
        input_row_repeats: Vec<u32>,
        /// Number of independent rows handled by one workgroup. Must divide
        /// the workgroup size; each row retains a power-of-two lane group.
        rows_per_workgroup: u32,
        /// Total workgroup size. Must be a power of 2 — the tree reduction
        /// within each packed row assumes it.
        grid: GridShape,
    },
}

/// Per-element write-back DAG for a reduce-then-map reduction. See
/// `KernelTemplate::Reduction` for the complete input-index layout.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ReductionEpilogue {
    pub dag: PointwiseDAG,
    pub n_per_col_inputs: u8,
}

/// Serializable, stably-hashable description of a reduction kernel —
/// stored on `compile::Dispatch::reduction` so the runtime can build
/// and cache a pipeline from it.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ReductionKernel {
    pub op: ReduceOp,
    pub prologue: PointwiseDAG,
    /// Additional accumulators: each extra prologue is reduced with the
    /// same `op` over the same input streams, producing one more reduced
    /// scalar. The epilogue sees reduced values at the tail of its input
    /// range in order: `prologue`'s result first, then each extra.
    /// LayerNorm needs `[sum(x), sum(x*x)]`; norm backwards need two
    /// differently-weighted sums.
    #[serde(default)]
    pub extra_prologues: Vec<PointwiseDAG>,
    pub epilogue: Option<ReductionEpilogue>,
    pub n_per_elem: u8,
    pub n_per_row: u8,
    pub workgroup_size: u32,
    /// Independent outer rows reduced by one workgroup. Old cached plans
    /// omit this field and retain the original one-row behavior.
    #[serde(default = "default_rows_per_workgroup")]
    pub rows_per_workgroup: u32,
    /// Which per-element streams are **gathers**: stream `i` loads
    /// `name[name_idx[row] * inner + col]` (the row is indirected through
    /// a `u32` indices buffer `{name}_idx`) instead of the direct
    /// `name[row * inner + col]`. Empty = no gathers (all direct). When
    /// non-empty, length must equal `n_per_elem`. Lets a fused reduction
    /// absorb an `Embedding`-style indexed load of its reduced axis (e.g.
    /// SH `sum_k coeff[cell[row]*K+k]·basis[pix[row]*K+k]`). Gather is only
    /// valid for per-element streams whose gathered axis IS the reduced
    /// axis (table row-stride == inner).
    #[serde(default)]
    pub gather_elem: Vec<bool>,
    /// Per-element stream `i` uses logical row `row / factor[i]`.
    /// This expresses shared-row pairwise inputs without a bespoke shader.
    /// Empty is equivalent to all factors being one.
    #[serde(default)]
    pub input_row_repeats: Vec<u32>,
    /// Keep the scalar shared-memory tree reduction order (disable subgroup
    /// reduction and `vec4` lane grouping). Set by emitters whose late-fused
    /// counterpart or debug-mode parity depends on the exact scalar tree
    /// summation order (currently `RmsNorm` vs `rms_norm_add.wgsl`).
    #[serde(default)]
    pub preserve_tree_order: bool,
}

/// Maximum number of input streams supported by a fused [`PointwiseDAG`].
/// Capped at 7 inputs + 1 output = 8 storage buffers so fused kernels fit
/// devices exposing an 8-storage-buffer-per-stage limit (and stay within the
/// 8-bit [`PointwiseStorage::bf16_inputs`] mask).
pub const MAX_POINTWISE_INPUTS: u8 = 7;

/// Static binding names for `n` pointwise (or per-element reduction) input
/// streams. Preserves `src` / `src_a`..`src_c` for `n <= 3` and extends to
/// `src_g` for wider fused DAGs.
pub(crate) fn pointwise_input_binding_names(n: u8) -> Vec<&'static str> {
    const NAMES: [&str; MAX_POINTWISE_INPUTS as usize] = [
        "src_a", "src_b", "src_c", "src_d", "src_e", "src_f", "src_g",
    ];
    match n {
        1 => vec!["src"],
        2..=MAX_POINTWISE_INPUTS => NAMES[..n as usize].to_vec(),
        _ => panic!(
            "PointwiseDAG arity {n} has no binding layout (supported: 1..={MAX_POINTWISE_INPUTS})"
        ),
    }
}

impl ReductionKernel {
    /// Stable content hash — suitable as a pipeline-cache key.
    pub fn hash_key(&self) -> u64 {
        let mut h = DefaultHasher::new();
        self.hash(&mut h);
        h.finish()
    }

    /// Whether this reduction can safely execute using subgroup intrinsics
    /// instead of a shared-memory tree reduction.
    pub fn can_use_subgroups(&self, subgroup_size: u32) -> bool {
        !self.preserve_tree_order
            && subgroup_size > 0
            && self.rows_per_workgroup == 1
            && self.workgroup_size > 1
    }

    /// Whether this reduction can safely execute 4 elements per thread via
    /// `vec4<f32>` loads (and stores when it has an F32 epilogue).
    ///
    /// Narrow packed rows (`lanes_per_row == 1`) stay scalar to preserve exact
    /// sequential column summation order, `preserve_tree_order` keeps scalar
    /// tree order, and `ArgMax` stays scalar because it tracks per-column
    /// indices.
    pub fn can_vectorize(&self, storage: PointwiseStorage, inner: u32) -> bool {
        !self.preserve_tree_order
            && self.op != ReduceOp::ArgMax
            && self.workgroup_size > self.rows_per_workgroup
            && storage == PointwiseStorage::default()
            && inner >= 4
            && inner.is_multiple_of(4)
    }

    /// Keep reduction arithmetic and scratch F32; optionally narrow its map result.
    #[cfg(test)]
    pub(crate) fn lower_with_storage(&self, storage: PointwiseStorage) -> ShaderModule {
        self.lower_with_options(storage, false, false)
    }

    /// Lower with optional subgroup reduction and 4-wide (`vec4<f32>`) loads/stores.
    pub(crate) fn lower_with_options(
        &self,
        storage: PointwiseStorage,
        use_subgroups: bool,
        vec4: bool,
    ) -> ShaderModule {
        lower_reduction(
            self.op,
            &self.prologue,
            &self.extra_prologues,
            self.epilogue.as_ref(),
            self.n_per_elem,
            self.n_per_row,
            &self.gather_elem,
            &self.input_row_repeats,
            self.rows_per_workgroup,
            GridShape {
                workgroup_size: self.workgroup_size,
            },
            storage,
            use_subgroups,
            vec4,
        )
    }

    /// Lower to the template form consumed by `lower`.
    pub fn to_template(&self) -> KernelTemplate {
        KernelTemplate::Reduction {
            op: self.op,
            prologue: self.prologue.clone(),
            extra_prologues: self.extra_prologues.clone(),
            epilogue: self.epilogue.clone(),
            n_per_elem: self.n_per_elem,
            n_per_row: self.n_per_row,
            gather_elem: self.gather_elem.clone(),
            input_row_repeats: self.input_row_repeats.clone(),
            rows_per_workgroup: self.rows_per_workgroup,
            grid: GridShape {
                workgroup_size: self.workgroup_size,
            },
        }
    }

    /// Total number of `var<storage> ... : array<f32>` input bindings
    /// (per-elem + per-row + per-col), excluding `dst`. Used by the
    /// runtime to pick the right ShaderData layout.
    pub fn n_buffer_inputs(&self) -> u8 {
        self.n_per_elem + self.n_per_row + self.epilogue.as_ref().map_or(0, |e| e.n_per_col_inputs)
    }
}

fn default_rows_per_workgroup() -> u32 {
    1
}

/// Entry-point name emitted for any `KernelTemplate`. One entry point per
/// generated module.
pub const POINTWISE_ENTRY: &str = "main";
pub const REDUCTION_ENTRY: &str = "main";

/// Lower a [`KernelTemplate`] to WGSL source.
pub fn lower(t: &KernelTemplate) -> ShaderModule {
    lower_with_storage(t, PointwiseStorage::default())
}

fn lower_with_storage(t: &KernelTemplate, storage: PointwiseStorage) -> ShaderModule {
    match *t {
        KernelTemplate::Pointwise { ref dag, grid } => lower_pointwise_storage(dag, grid, storage),
        KernelTemplate::Reduction {
            op,
            ref prologue,
            ref extra_prologues,
            ref epilogue,
            n_per_elem,
            n_per_row,
            ref gather_elem,
            ref input_row_repeats,
            rows_per_workgroup,
            grid,
        } => lower_reduction(
            op,
            prologue,
            extra_prologues,
            epilogue.as_ref(),
            n_per_elem,
            n_per_row,
            gather_elem,
            input_row_repeats,
            rows_per_workgroup,
            grid,
            storage,
            false,
            false,
        ),
    }
}

/// Storage only; generated scalar expressions and sensitive consumers stay F32.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PointwiseStorage {
    /// Bit i marks input i as BF16. Unmarked inputs are F32.
    pub bf16_inputs: u8,
    pub bf16_output: bool,
    #[serde(default)]
    pub f16_output: bool,
}

pub fn lower_pointwise_storage(
    dag: &PointwiseDAG,
    grid: GridShape,
    storage: PointwiseStorage,
) -> ShaderModule {
    lower_pointwise_with_options(dag, grid, storage, None)
}

/// Lower a `vec4<f32>`-vectorized pointwise kernel (`4` F32 elements per
/// thread). `input_is_vec4` comes from [`PointwiseDAG::vec4_input_kinds`].
pub fn lower_pointwise_vec4(
    dag: &PointwiseDAG,
    grid: GridShape,
    input_is_vec4: &[bool],
) -> ShaderModule {
    lower_pointwise_with_options(dag, grid, PointwiseStorage::default(), Some(input_is_vec4))
}

fn lower_pointwise_with_options(
    dag: &PointwiseDAG,
    grid: GridShape,
    storage: PointwiseStorage,
    vec4_inputs: Option<&[bool]>,
) -> ShaderModule {
    assert!(
        dag.n_inputs <= MAX_POINTWISE_INPUTS,
        "PointwiseDAG arity {} exceeds max {MAX_POINTWISE_INPUTS}",
        dag.n_inputs
    );
    if dag.n_inputs < 8 {
        assert_eq!(storage.bf16_inputs >> dag.n_inputs, 0);
    }
    assert!(!(storage.bf16_output && storage.f16_output));
    assert!(
        dag.n_inputs >= 1,
        "PointwiseDAG must have at least one input stream"
    );
    assert!(
        (dag.output as usize) < dag.ops.len(),
        "PointwiseDAG.output is out of range"
    );
    if let Some(kinds) = vec4_inputs {
        assert_eq!(storage, PointwiseStorage::default());
        assert_eq!(kinds.len(), dag.n_inputs as usize);
    }

    let input_names = PointwiseDAG::input_binding_names(dag.n_inputs);

    let mut src = String::new();
    if storage.f16_output {
        src.push_str("enable f16;\n");
    }
    if storage.bf16_inputs != 0 || storage.bf16_output {
        src.push_str("enable wgpu_bfloat16;\n");
        src.push_str(include_str!("shaders/bf16_round.wgsl"));
    }
    src.push_str(
        "struct Params {\n    len: u32,\n    _pad0: u32,\n    _pad1: u32,\n    _pad2: u32,\n}\n\n",
    );
    for (i, name) in input_names.iter().enumerate() {
        let ty = if let Some(kinds) = vec4_inputs {
            if kinds[i] { "vec4<f32>" } else { "f32" }
        } else if storage.bf16_inputs & (1 << i) != 0 {
            "bf16"
        } else {
            "f32"
        };
        let _ = writeln!(src, "var<storage> {name}: array<{ty}>;");
    }
    let ty = if vec4_inputs.is_some() {
        "vec4<f32>"
    } else if storage.bf16_output {
        "bf16"
    } else if storage.f16_output {
        "f16"
    } else {
        "f32"
    };
    let _ = writeln!(src, "var<storage, read_write> dst: array<{ty}>;");
    src.push_str("var<uniform> params: Params;\n\n");
    let _ = writeln!(src, "@compute @workgroup_size({})", grid.workgroup_size);
    let _ = writeln!(
        src,
        "fn {}(@builtin(global_invocation_id) gid: vec3<u32>) {{",
        POINTWISE_ENTRY
    );
    src.push_str("    let i = gid.x;\n");
    if let Some(kinds) = vec4_inputs {
        src.push_str("    if i >= params.len / 4u { return; }\n");
        src.push_str(&dag.emit_body_impl(
            |idx| format!("{}[i]", input_names[idx as usize]),
            |idx, divisor, modulus| {
                let name = input_names[idx as usize];
                if kinds[idx as usize] {
                    debug_assert_eq!(divisor, 1);
                    debug_assert!(modulus.is_multiple_of(4));
                    format!("{name}[i % {}u]", modulus / 4)
                } else {
                    format!("vec4<f32>({name}[(i * 4u / {divisor}u) % {modulus}u])")
                }
            },
            true,
        ));
        let _ = writeln!(src, "    dst[i] = v{};", dag.output);
    } else {
        src.push_str("    if i >= params.len { return; }\n");
        let load = |idx: u8, index: String| {
            let value = format!("{}[{index}]", input_names[idx as usize]);
            if storage.bf16_inputs & (1 << idx) != 0 {
                format!("f32({value})")
            } else {
                value
            }
        };
        src.push_str(&dag.emit_body_with(
            |idx| load(idx, "i".into()),
            |idx, divisor, modulus| load(idx, format!("(i / {divisor}u) % {modulus}u")),
        ));
        if storage.bf16_output {
            let _ = writeln!(src, "    dst[i] = round_bf16(v{});", dag.output);
        } else if storage.f16_output {
            let _ = writeln!(src, "    dst[i] = f16(v{});", dag.output);
        } else {
            let _ = writeln!(src, "    dst[i] = v{};", dag.output);
        }
    }
    src.push_str("}\n");

    ShaderModule::with_hint(src, "pointwise")
}

#[allow(clippy::too_many_arguments)]
fn lower_reduction(
    op: ReduceOp,
    prologue: &PointwiseDAG,
    extra_prologues: &[PointwiseDAG],
    epilogue: Option<&ReductionEpilogue>,
    n_per_elem: u8,
    n_per_row: u8,
    gather_elem: &[bool],
    input_row_repeats: &[u32],
    rows_per_workgroup: u32,
    grid: GridShape,
    storage: PointwiseStorage,
    use_subgroups: bool,
    vec4: bool,
) -> ShaderModule {
    assert_eq!(storage.bf16_inputs, 0, "reduction inputs remain F32");
    assert!(!(storage.bf16_output && storage.f16_output));
    assert!(
        epilogue.is_some() || storage == PointwiseStorage::default(),
        "reduction partials remain F32"
    );
    assert!(
        n_per_elem >= 1,
        "reduction must have at least one per-element input stream"
    );
    assert!(
        gather_elem.is_empty() || gather_elem.len() == n_per_elem as usize,
        "gather_elem must be empty or length n_per_elem ({}), got {}",
        n_per_elem,
        gather_elem.len(),
    );
    assert!(
        input_row_repeats.is_empty() || input_row_repeats.len() == n_per_elem as usize,
        "input_row_repeats must be empty or length n_per_elem ({n_per_elem}), got {}",
        input_row_repeats.len(),
    );
    assert!(
        input_row_repeats.iter().all(|&factor| factor > 0),
        "input row repeat factors must be non-zero",
    );
    let is_gather = |i: usize| gather_elem.get(i).copied().unwrap_or(false);
    assert_eq!(
        prologue.n_inputs,
        n_per_elem + n_per_row,
        "prologue arity ({}) must equal n_per_elem + n_per_row ({} + {})",
        prologue.n_inputs,
        n_per_elem,
        n_per_row,
    );
    assert!(
        (prologue.output as usize) < prologue.ops.len(),
        "reduction prologue output index is out of range"
    );
    for (i, extra) in extra_prologues.iter().enumerate() {
        assert_eq!(
            extra.n_inputs, prologue.n_inputs,
            "extra prologue {i} arity must match the primary prologue"
        );
        assert!(
            (extra.output as usize) < extra.ops.len(),
            "extra prologue {i} output index is out of range"
        );
    }
    let n_reduced = 1 + extra_prologues.len();
    assert!(
        grid.workgroup_size.is_power_of_two() && grid.workgroup_size >= 2,
        "reduction workgroup_size must be a power of 2 ≥ 2"
    );
    assert!(
        rows_per_workgroup.is_power_of_two()
            && rows_per_workgroup <= grid.workgroup_size
            && grid.workgroup_size.is_multiple_of(rows_per_workgroup),
        "reduction rows_per_workgroup must be a power of 2 that divides workgroup_size"
    );
    if let Some(epi) = epilogue {
        let expected =
            n_per_elem as usize + n_per_row as usize + epi.n_per_col_inputs as usize + n_reduced;
        assert_eq!(
            epi.dag.n_inputs as usize, expected,
            "reduction epilogue expects {} inputs ({} per-elem + {} per-row + {} per-col + {} reduced), got {}",
            expected, n_per_elem, n_per_row, epi.n_per_col_inputs, n_reduced, epi.dag.n_inputs,
        );
        assert!(
            (epi.dag.output as usize) < epi.dag.ops.len(),
            "reduction epilogue output index is out of range"
        );
    }

    let wg = grid.workgroup_size;
    let lanes_per_row = wg / rows_per_workgroup;
    let use_subgroups = use_subgroups && rows_per_workgroup == 1 && wg > 1;
    let vec4 = vec4
        && op != ReduceOp::ArgMax
        && lanes_per_row > 1
        && storage == PointwiseStorage::default();
    let elem_ty = if vec4 { "vec4<f32>" } else { "f32" };
    let inner_expr = if vec4 { "inner_len" } else { "params.inner" };

    let per_elem_names = PointwiseDAG::input_binding_names(n_per_elem);
    let per_row_names = per_row_binding_names(n_per_row);
    let per_col_names: Vec<String> = epilogue
        .map(|epi| per_col_binding_names(epi.n_per_col_inputs))
        .unwrap_or_default();

    // Builds the `load(idx)` expression for a given DAG input index,
    // respecting the documented per-elem / per-row / per-col / reduced
    // ordering. `in_epilogue = false` disallows per-col and reduced.
    let load_expr = |idx: u8, col_var: &str, in_epilogue: bool| -> String {
        let i = idx as usize;
        let per_elem_end = n_per_elem as usize;
        let per_row_end = per_elem_end + n_per_row as usize;
        if i < per_elem_end {
            let repeat = input_row_repeats.get(i).copied().unwrap_or(1);
            if is_gather(i) {
                let source_row = if repeat == 1 {
                    "row".to_string()
                } else {
                    format!("(row / {repeat}u)")
                };
                // Gather: the row is indirected through a u32 indices
                // buffer, then the gathered row's `inner`-stride slice is
                // indexed by the reduction column. Valid only when the
                // gathered axis IS the reduced axis (table stride == inner).
                // An index past the table reads its last row.
                format!(
                    "{0}[min({0}_idx[{1}], params.table_rows - 1u) * {inner_expr} + {2}]",
                    per_elem_names[i], source_row, col_var
                )
            } else if repeat == 1 {
                format!("{}[row_offset + {}]", per_elem_names[i], col_var)
            } else {
                format!(
                    "{}[(row / {repeat}u) * {inner_expr} + {}]",
                    per_elem_names[i], col_var
                )
            }
        } else if i < per_row_end {
            let expr = format!("{}[row]", per_row_names[i - per_elem_end]);
            if vec4 {
                format!("vec4<f32>({expr})")
            } else {
                expr
            }
        } else {
            assert!(in_epilogue, "prologue cannot reference per-col or reduced");
            let per_col_end = per_row_end + epilogue.map_or(0, |e| e.n_per_col_inputs as usize);
            if i < per_col_end {
                format!("{}[{}]", per_col_names[i - per_row_end], col_var)
            } else {
                // Reduced scalars occupy the tail positions in order.
                let expr = format!("reduced_{}", i - per_col_end);
                if vec4 {
                    format!("vec4<f32>({expr})")
                } else {
                    expr
                }
            }
        }
    };

    let mut src = String::new();
    if storage.f16_output {
        src.push_str("enable f16;\n");
    }
    if storage.bf16_output {
        src.push_str("enable wgpu_bfloat16;\n");
        src.push_str(include_str!("shaders/bf16_round.wgsl"));
    }
    src.push_str(
        "struct Params {\n    outer: u32,\n    inner: u32,\n    round_one_bits: u32,\n    table_rows: u32,\n}\n\n",
    );
    for (i, name) in per_elem_names.iter().enumerate() {
        let _ = writeln!(src, "var<storage> {name}: array<{elem_ty}>;");
        if is_gather(i) {
            // u32 indices buffer for this gather stream.
            let _ = writeln!(src, "var<storage> {name}_idx: array<u32>;");
        }
    }
    for name in &per_row_names {
        let _ = writeln!(src, "var<storage> {name}: array<f32>;");
    }
    for name in &per_col_names {
        let _ = writeln!(src, "var<storage> {name}: array<{elem_ty}>;");
    }
    let output_type = if epilogue.is_some() && vec4 {
        "vec4<f32>"
    } else if storage.f16_output {
        "f16"
    } else if storage.bf16_output {
        "bf16"
    } else {
        "f32"
    };
    let _ = writeln!(src, "var<storage, read_write> dst: array<{output_type}>;");
    src.push_str("var<uniform> params: Params;\n");
    if lanes_per_row > 1 {
        for k in 0..n_reduced {
            let _ = writeln!(src, "var<workgroup> wg_data_{k}: array<f32, {wg}>;");
            if op == ReduceOp::ArgMax {
                let _ = writeln!(src, "var<workgroup> wg_index_{k}: array<u32, {wg}>;");
            }
        }
    }
    src.push('\n');
    let _ = writeln!(src, "@compute @workgroup_size({})", wg);
    if use_subgroups {
        let _ = writeln!(
            src,
            "fn {REDUCTION_ENTRY}(\n    @builtin(workgroup_id) wgid: vec3<u32>,\n    @builtin(local_invocation_id) lid: vec3<u32>,\n    @builtin(subgroup_invocation_id) sg_id: u32,\n    @builtin(subgroup_id) wave_id: u32,\n    @builtin(subgroup_size) wave_size: u32,\n) {{"
        );
        src.push_str("    let row = wgid.x;\n");
        src.push_str("    if row >= params.outer { return; }\n");
        src.push_str("    let lane = lid.x;\n");
        src.push_str("    let row_valid = true;\n");
    } else {
        let _ = writeln!(
            src,
            "fn {}(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{",
            REDUCTION_ENTRY
        );
        src.push_str("    let tid = lid.x;\n");
        let _ = writeln!(src, "    let row_slot = tid / {}u;", lanes_per_row);
        let _ = writeln!(src, "    let lane = tid % {}u;", lanes_per_row);
        let _ = writeln!(
            src,
            "    let row = wgid.x * {}u + row_slot;",
            rows_per_workgroup
        );
        src.push_str("    let row_valid = row < params.outer;\n");
    }
    if vec4 {
        src.push_str("    let inner_len = params.inner / 4u;\n");
    }
    let _ = writeln!(src, "    let row_offset = row * {inner_expr};");
    for k in 0..n_reduced {
        if vec4 {
            let _ = writeln!(
                src,
                "    var acc_v_{k}: vec4<f32> = vec4<f32>({});",
                op.identity_wgsl()
            );
        } else {
            let _ = writeln!(src, "    var acc_{k}: f32 = {};", op.identity_wgsl());
            if op == ReduceOp::ArgMax {
                let _ = writeln!(src, "    var index_{k}: u32 = 0xffffffffu;");
            }
        }
    }
    src.push_str("    var col = lane;\n");
    src.push_str("    if row_valid {\n");
    src.push_str("        loop {\n");
    let _ = writeln!(src, "            if col >= {inner_expr} {{ break; }}");
    // Prologue bodies: `col` is the per-element index variable. Each
    // accumulator's DAG is evaluated in its own block so the `let vN`
    // value names don't collide.
    for (k, dag) in std::iter::once(prologue)
        .chain(extra_prologues.iter())
        .enumerate()
    {
        src.push_str("            {\n");
        let body = dag.emit_body_impl(
            |idx| load_expr(idx, "col", false),
            |_, _, _| panic!("broadcast loads are only valid in flat pointwise kernels"),
            vec4,
        );
        for line in body.lines() {
            src.push_str("                ");
            src.push_str(line.trim_start_matches(' '));
            src.push('\n');
        }
        let prologue_value = if lanes_per_row == 1 {
            // Multiplication by a runtime one prevents a folded producer
            // multiply from contracting with the reduction add. This keeps
            // the fused packed path bit-identical to materializing the
            // producer without a global-memory round trip.
            format!("v{} * bitcast<f32>(params.round_one_bits)", dag.output)
        } else {
            format!("v{}", dag.output)
        };
        let acc = if vec4 {
            format!("acc_v_{k}")
        } else {
            format!("acc_{k}")
        };
        let combine = if op == ReduceOp::ArgMax {
            combine_argmax(&acc, &format!("index_{k}"), &prologue_value, "col")
        } else {
            op.combine_wgsl(&acc, &prologue_value)
        };
        let _ = writeln!(src, "                {combine}");
        src.push_str("            }\n");
    }
    let _ = writeln!(src, "            col += {}u;", lanes_per_row);
    src.push_str("        }\n");
    src.push_str("    }\n");
    if vec4 {
        for k in 0..n_reduced {
            let reduced = match op {
                ReduceOp::Sum => {
                    format!("(acc_v_{k}.x + acc_v_{k}.y) + (acc_v_{k}.z + acc_v_{k}.w)")
                }
                ReduceOp::Max => {
                    format!("max(max(acc_v_{k}.x, acc_v_{k}.y), max(acc_v_{k}.z, acc_v_{k}.w))")
                }
                ReduceOp::ArgMax => unreachable!("ArgMax stays scalar"),
            };
            let _ = writeln!(src, "    let acc_{k}: f32 = {reduced};");
        }
    }

    if lanes_per_row == 1 {
        // One lane owns the entire row — no shared memory or barriers needed.
        for k in 0..n_reduced {
            let value = if op == ReduceOp::ArgMax {
                format!("f32(index_{k})")
            } else {
                format!("acc_{k}")
            };
            let _ = writeln!(src, "    let reduced_{k} = {value};");
        }
    } else if use_subgroups {
        // Subgroup reduction: zero barriers when the whole workgroup fits in
        // one subgroup, and one barrier otherwise (down from log2(wg) + 1).
        let _ = writeln!(src, "    let waves = ({wg}u + wave_size - 1u) / wave_size;");
        for k in 0..n_reduced {
            match op {
                ReduceOp::Sum => {
                    let _ = writeln!(src, "    let wave_val_{k} = subgroupAdd(acc_{k});");
                }
                ReduceOp::Max => {
                    let _ = writeln!(src, "    let wave_val_{k} = subgroupMax(acc_{k});");
                }
                ReduceOp::ArgMax => {
                    // Match the scalar tree path and f64 reference across all
                    // NaN patterns without feeding NaNs into `subgroupMax`
                    // (whose NaN result is undefined in SPIR-V):
                    // - If column 0 (owned by lane 0) is NaN, nothing compares
                    //   greater so column 0 wins with a NaN sentinel value.
                    // - Otherwise, mask NaNs to `-inf` before `subgroupMax`;
                    //   waves with no non-NaN active lanes emit `0xffffffffu`
                    //   so they are skipped by the cross-wave combine.
                    let _ = writeln!(
                        src,
                        "    let is_nan_{k} = (bitcast<u32>(acc_{k}) & 0x7fffffffu) > 0x7f800000u;"
                    );
                    let _ = writeln!(
                        src,
                        "    let lane0_nan_{k} = subgroupAny((lane == 0u) & is_nan_{k});"
                    );
                    let _ = writeln!(
                        src,
                        "    let any_non_nan_{k} = subgroupAny((index_{k} != 0xffffffffu) & !is_nan_{k});"
                    );
                    let _ = writeln!(
                        src,
                        "    let wave_max_{k} = subgroupMax(select(bitcast<f32>(0xff800000u), acc_{k}, !is_nan_{k}));"
                    );
                    let _ = writeln!(
                        src,
                        "    let wave_val_{k} = select(bitcast<f32>(0x7fc00000u), wave_max_{k}, !lane0_nan_{k} & any_non_nan_{k});"
                    );
                    let _ = writeln!(
                        src,
                        "    let wave_match_{k} = subgroupMin(select(0xffffffffu, index_{k}, (index_{k} != 0xffffffffu) & (acc_{k} == wave_val_{k})));"
                    );
                    let _ = writeln!(
                        src,
                        "    let wave_first_{k} = subgroupMin(select(0xffffffffu, index_{k}, !is_nan_{k}));"
                    );
                    let _ = writeln!(
                        src,
                        "    let wave_idx_{k} = select(select(wave_match_{k}, wave_first_{k}, wave_match_{k} == 0xffffffffu), 0u, lane0_nan_{k});"
                    );
                }
            }
        }
        src.push_str("    if waves > 1u {\n");
        src.push_str("        if sg_id == subgroupBroadcastFirst(sg_id) {\n");
        for k in 0..n_reduced {
            let _ = writeln!(src, "            wg_data_{k}[wave_id] = wave_val_{k};");
            if op == ReduceOp::ArgMax {
                let _ = writeln!(src, "            wg_index_{k}[wave_id] = wave_idx_{k};");
            }
        }
        src.push_str("        }\n");
        src.push_str("        workgroupBarrier();\n");
        src.push_str("    }\n");
        for k in 0..n_reduced {
            let _ = writeln!(src, "    var total_{k}: f32 = wave_val_{k};");
            if op == ReduceOp::ArgMax {
                let _ = writeln!(src, "    var total_idx_{k}: u32 = wave_idx_{k};");
            }
        }
        if epilogue.is_none() {
            src.push_str("    if lane == 0u && waves > 1u {\n");
        } else {
            src.push_str("    if waves > 1u {\n");
        }
        for k in 0..n_reduced {
            let _ = writeln!(src, "        total_{k} = {};", op.identity_wgsl());
            if op == ReduceOp::ArgMax {
                let _ = writeln!(src, "        total_idx_{k} = 0xffffffffu;");
            }
        }
        src.push_str("        for (var w = 0u; w < waves; w += 1u) {\n");
        for k in 0..n_reduced {
            let combined = if op == ReduceOp::ArgMax {
                combine_argmax(
                    &format!("total_{k}"),
                    &format!("total_idx_{k}"),
                    &format!("wg_data_{k}[w]"),
                    &format!("wg_index_{k}[w]"),
                )
            } else {
                op.combine_wgsl(&format!("total_{k}"), &format!("wg_data_{k}[w]"))
            };
            let _ = writeln!(src, "            {combined}");
        }
        src.push_str("        }\n");
        src.push_str("    }\n");
        for k in 0..n_reduced {
            let value = if op == ReduceOp::ArgMax {
                format!("f32(total_idx_{k})")
            } else {
                format!("total_{k}")
            };
            let _ = writeln!(src, "    let reduced_{k} = {value};");
        }
    } else {
        for k in 0..n_reduced {
            let _ = writeln!(src, "    wg_data_{k}[tid] = acc_{k};");
            if op == ReduceOp::ArgMax {
                let _ = writeln!(src, "    wg_index_{k}[tid] = index_{k};");
            }
        }
        src.push_str("    workgroupBarrier();\n");
        // Independent tree reductions within each packed row's lane group.
        let mut stride = lanes_per_row / 2;
        src.push_str("    // packed per-row tree reduction\n");
        while stride > 0 {
            let _ = writeln!(src, "    if lane < {}u {{", stride);
            for k in 0..n_reduced {
                let (acc, value) = (
                    format!("wg_data_{k}[tid]"),
                    format!("wg_data_{k}[tid + {stride}u]"),
                );
                let combined = if op == ReduceOp::ArgMax {
                    let index = format!("wg_index_{k}[tid]");
                    let column = format!("wg_index_{k}[tid + {stride}u]");
                    combine_argmax(&acc, &index, &value, &column)
                } else {
                    op.combine_wgsl(&acc, &value)
                };
                let _ = writeln!(src, "        {}", combined);
            }
            src.push_str("    }\n");
            src.push_str("    workgroupBarrier();\n");
            stride /= 2;
        }
        for k in 0..n_reduced {
            let slot = if epilogue.is_none() {
                "tid".to_string()
            } else {
                format!("row_slot * {lanes_per_row}u")
            };
            let value = reduced_value(op, k, &slot);
            let _ = writeln!(src, "    let reduced_{k} = {value};");
        }
    }

    match epilogue {
        None => {
            // K reduced scalars per row, stored interleaved: dst[row*K + k].
            src.push_str("    if lane == 0u && row_valid {\n");
            for k in 0..n_reduced {
                if n_reduced == 1 {
                    let _ = writeln!(src, "        dst[row] = reduced_{k};");
                } else {
                    let _ = writeln!(src, "        dst[row * {n_reduced}u + {k}u] = reduced_{k};");
                }
            }
            src.push_str("    }\n");
        }
        Some(epi) => {
            // Reduce-then-map: broadcast the reduced values and write a
            // transformed element back for every col.
            src.push_str("    var wcol = lane;\n");
            src.push_str("    if row_valid {\n");
            src.push_str("        loop {\n");
            let _ = writeln!(src, "            if wcol >= {inner_expr} {{ break; }}");
            let body = epi.dag.emit_body_impl(
                |idx| load_expr(idx, "wcol", true),
                |_, _, _| panic!("broadcast loads are only valid in flat pointwise kernels"),
                vec4,
            );
            for line in body.lines() {
                src.push_str("            ");
                src.push_str(line.trim_start_matches(' '));
                src.push('\n');
            }
            let result = format!("v{}", epi.dag.output);
            let result = if storage.f16_output {
                format!("f16({result})")
            } else if storage.bf16_output {
                format!("round_bf16({result})")
            } else {
                result
            };
            let _ = writeln!(src, "            dst[row_offset + wcol] = {result};");
            let _ = writeln!(src, "            wcol += {}u;", lanes_per_row);
            src.push_str("        }\n");
            src.push_str("    }\n");
        }
    }
    src.push_str("}\n");

    let module = ShaderModule::with_hint(src, "reduction");
    if use_subgroups {
        module.with_subgroup(0, [wg, 1, 1])
    } else {
        module
    }
}

/// Binding names for per-row broadcast inputs used by the reduction
/// prologue and epilogue:
///   * n=0 → `[]`
///   * n=1 → `["per_row_src"]`
///   * n=2 → `["per_row_src_a", "per_row_src_b"]`
///
/// Panics for n > 2 — wider layouts need additional runtime plumbing.
fn per_row_binding_names(n: u8) -> Vec<String> {
    match n {
        0 => Vec::new(),
        1 => vec!["per_row_src".to_string()],
        2 => vec!["per_row_src_a".to_string(), "per_row_src_b".to_string()],
        n => panic!("reduction n_per_row={} has no binding layout (max 2)", n),
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

    /// dst[i] = max(src[i], 0).
    fn relu_dag() -> PointwiseDAG {
        PointwiseDAG {
            n_inputs: 1,
            ops: vec![Pw::LoadInput(0), Pw::Relu(0)],
            output: 1,
        }
    }

    /// dst[i] = src_a[i] + src_b[i].
    fn add_dag() -> PointwiseDAG {
        PointwiseDAG {
            n_inputs: 2,
            ops: vec![Pw::LoadInput(0), Pw::LoadInput(1), Pw::Add(0, 1)],
            output: 2,
        }
    }

    /// silu(src_a[i]) * src_b[i].
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
        // The names UnaryData binds.
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
        // The names BinaryData binds.
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
            extra_prologues: vec![],
            epilogue: None,
            n_per_elem: 1,
            n_per_row: 0,
            gather_elem: Vec::new(),
            input_row_repeats: Vec::new(),
            rows_per_workgroup: 1,
            grid: GridShape::default(),
        });
        assert!(sm.source.contains("struct Params"));
        assert!(sm.source.contains("outer: u32"));
        assert!(sm.source.contains("var<workgroup> wg_data_0"));
        assert!(sm.source.contains("workgroupBarrier"));
        // Identity for sum is 0.
        assert!(sm.source.contains("var acc_0: f32 = 0.0f"));
        // Dst writes one scalar per row.
        assert!(sm.source.contains("let reduced_0 = wg_data_0[tid];"));
        assert!(sm.source.contains("dst[row] = reduced_0;"));
    }

    #[test]
    fn max_reduction_uses_max_combiner() {
        let sm = lower(&KernelTemplate::Reduction {
            op: ReduceOp::Max,
            prologue: identity_prologue(),
            extra_prologues: vec![],
            epilogue: None,
            n_per_elem: 1,
            n_per_row: 0,
            gather_elem: Vec::new(),
            input_row_repeats: Vec::new(),
            rows_per_workgroup: 1,
            grid: GridShape::default(),
        });
        // Max identity is -inf via bitcast.
        assert!(sm.source.contains("0xff800000u"));
        // Combiner uses max().
        assert!(sm.source.contains("max(acc_0,"));
    }

    #[test]
    fn reduction_with_square_prologue_compiles() {
        // Sum of squares — the reduction half of RMSNorm.
        let sm = lower(&KernelTemplate::Reduction {
            op: ReduceOp::Sum,
            prologue: square_prologue(),
            extra_prologues: vec![],
            epilogue: None,
            n_per_elem: 1,
            n_per_row: 0,
            gather_elem: Vec::new(),
            input_row_repeats: Vec::new(),
            rows_per_workgroup: 1,
            grid: GridShape::default(),
        });
        // Prologue emits the multiply.
        assert!(sm.source.contains("v0 * v0"));
        // And the per-row load expression should reference row_offset.
        assert!(sm.source.contains("src[row_offset + col]"));
    }

    #[test]
    fn reduction_with_gather_streams_emits_indirect_load() {
        // SH sum-over-K: color[row] = sum_col(coeff[cell_idx[row]*K+col]
        //                                      * basis[pix_idx[row]*K+col]).
        // Two per-element GATHER streams, Mul prologue, Sum reduce.
        let prologue = PointwiseDAG {
            n_inputs: 2,
            ops: vec![Pw::LoadInput(0), Pw::LoadInput(1), Pw::Mul(0, 1)],
            output: 2,
        };
        let sm = lower(&KernelTemplate::Reduction {
            op: ReduceOp::Sum,
            prologue,
            extra_prologues: vec![],
            epilogue: None,
            n_per_elem: 2,
            n_per_row: 0,
            gather_elem: vec![true, true],
            input_row_repeats: Vec::new(),
            rows_per_workgroup: 1,
            grid: GridShape::default(),
        });
        // Indices buffers declared for both gather streams.
        assert!(sm.source.contains("var<storage> src_a_idx: array<u32>;"));
        assert!(sm.source.contains("var<storage> src_b_idx: array<u32>;"));
        // Indirect (gathered) loads, not the direct row_offset form.
        assert!(
            sm.source.contains(
                "src_a[min(src_a_idx[row], params.table_rows - 1u) * params.inner + col]"
            )
        );
        assert!(
            sm.source.contains(
                "src_b[min(src_b_idx[row], params.table_rows - 1u) * params.inner + col]"
            )
        );
        assert!(!sm.source.contains("src_a[row_offset"));
        // The prologue multiply still fuses in.
        assert!(sm.source.contains("v0 * v1"));
    }

    #[test]
    fn reduction_with_partial_gather_compiles() {
        // One gather stream + one direct stream (mixed).
        let prologue = PointwiseDAG {
            n_inputs: 2,
            ops: vec![Pw::LoadInput(0), Pw::LoadInput(1), Pw::Mul(0, 1)],
            output: 2,
        };
        let sm = lower(&KernelTemplate::Reduction {
            op: ReduceOp::Sum,
            prologue,
            extra_prologues: vec![],
            epilogue: None,
            n_per_elem: 2,
            n_per_row: 0,
            gather_elem: vec![true, false],
            input_row_repeats: Vec::new(),
            rows_per_workgroup: 1,
            grid: GridShape::default(),
        });
        // Stream 0 gathered, stream 1 direct.
        assert!(
            sm.source.contains(
                "src_a[min(src_a_idx[row], params.table_rows - 1u) * params.inner + col]"
            )
        );
        assert!(sm.source.contains("src_b[row_offset + col]"));
        assert!(sm.source.contains("var<storage> src_a_idx: array<u32>;"));
        assert!(!sm.source.contains("src_b_idx"));
    }

    #[test]
    fn reduction_with_repeated_input_rows_compiles() {
        let prologue = PointwiseDAG {
            n_inputs: 2,
            ops: vec![Pw::LoadInput(0), Pw::LoadInput(1), Pw::Sub(0, 1)],
            output: 2,
        };
        let sm = lower(&KernelTemplate::Reduction {
            op: ReduceOp::Sum,
            prologue,
            extra_prologues: vec![],
            epilogue: None,
            n_per_elem: 2,
            n_per_row: 0,
            gather_elem: Vec::new(),
            input_row_repeats: vec![4, 1],
            rows_per_workgroup: 1,
            grid: GridShape::default(),
        });
        assert!(sm.source.contains("src_a[(row / 4u) * params.inner + col]"));
        assert!(sm.source.contains("src_b[row_offset + col]"));

        use naga::valid::{Capabilities, ValidationFlags, Validator};
        let flags = ValidationFlags::all() ^ ValidationFlags::BINDINGS;
        Validator::new(flags, Capabilities::all())
            .validate(&sm.module)
            .unwrap_or_else(|e| panic!("repeated-row reduction invalid: {:?}\n{}", e, sm.source));
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
                extra_prologues: vec![],
                epilogue: None,
                n_per_elem: 1,
                n_per_row: 0,
                gather_elem: Vec::new(),
                input_row_repeats: Vec::new(),
                rows_per_workgroup: 1,
                grid: GridShape::default(),
            });
            v.validate(&sm.module).unwrap_or_else(|e| {
                panic!("reduction naga validation failed: {:?}\n{}", e, sm.source);
            });
        }
    }

    /// Build the RMSNorm epilogue DAG under the canonical reduction
    /// layout (per-elem, per-row, per-col, reduced). For RMSNorm:
    ///   0 = src[row, col]           (per-elem)
    ///   1 = bias[col]                (per-col)
    ///   2 = sum_of_squares           (reduced — always last)
    /// Output: `src * rsqrt(sum_of_squares * inv_inner + eps) * bias`.
    fn rmsnorm_epilogue(inner: f32, eps: f32) -> ReductionEpilogue {
        let inv_inner = Pw::const_f32(1.0 / inner);
        let eps_c = Pw::const_f32(eps);
        let dag = PointwiseDAG {
            n_inputs: 3,
            ops: vec![
                Pw::LoadInput(0), // v0 = src[row, col]
                Pw::LoadInput(1), // v1 = bias[col]
                Pw::LoadInput(2), // v2 = sum_of_squares
                inv_inner,        // v3 = 1/inner
                eps_c,            // v4 = eps
                Pw::Mul(2, 3),    // v5 = mean_sq = sum_sq * inv_inner
                Pw::Add(5, 4),    // v6 = mean_sq + eps
                Pw::Rsqrt(6),     // v7 = rsqrt(mean_sq + eps)
                Pw::Mul(0, 7),    // v8 = src * rsqrt
                Pw::Mul(8, 1),    // v9 = (src * rsqrt) * bias
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
            extra_prologues: vec![],
            epilogue: Some(rmsnorm_epilogue(720.0, 1e-5)),
            n_per_elem: 1,
            n_per_row: 0,
            gather_elem: Vec::new(),
            input_row_repeats: Vec::new(),
            rows_per_workgroup: 1,
            grid: GridShape::default(),
        });
        // Has the per-col bias binding.
        assert!(sm.source.contains("var<storage> bias: array<f32>;"));
        // Per-element writeback instead of per-row.
        assert!(sm.source.contains("dst[row_offset + wcol]"));
        // Reduced value is read into a local.
        assert!(
            sm.source
                .contains("let reduced_0 = wg_data_0[row_slot * 256u];")
        );
        // Per-col load uses wcol.
        assert!(sm.source.contains("bias[wcol]"));
        // Rsqrt appears in generated epilogue.
        assert!(sm.source.contains("inverseSqrt"));
    }

    /// Softmax composition:
    ///   Dispatch 1: Reduction(Max, identity) → row_max[outer].
    ///   Dispatch 2: Reduction(Sum, exp(src - row_max),
    ///                          epilogue = exp(src - row_max) / row_sum)
    ///               over inputs (src, row_max). Writes [outer, inner].
    ///
    /// This test constructs the dispatch-2 kernel and validates it.
    #[test]
    fn softmax_pass2_lowers() {
        // Prologue: `exp(src - row_max)`. Input 0 = src (per-elem),
        // input 1 = row_max (per-row). n_inputs = 2.
        let prologue = PointwiseDAG {
            n_inputs: 2,
            ops: vec![
                Pw::LoadInput(0), // v0 = src
                Pw::LoadInput(1), // v1 = row_max
                Pw::Sub(0, 1),    // v2 = src - row_max
                Pw::Exp(2),       // v3 = exp(src - row_max)
            ],
            output: 3,
        };
        // Epilogue: `exp(src - row_max) / row_sum`. Inputs: 0=src
        // (per-elem), 1=row_max (per-row), 2=row_sum (reduced, last).
        let epi_dag = PointwiseDAG {
            n_inputs: 3,
            ops: vec![
                Pw::LoadInput(0), // v0 = src
                Pw::LoadInput(1), // v1 = row_max
                Pw::LoadInput(2), // v2 = row_sum
                Pw::Sub(0, 1),    // v3 = src - row_max
                Pw::Exp(3),       // v4 = exp(src - row_max)
                Pw::Div(4, 2),    // v5 = exp(...) / row_sum
            ],
            output: 5,
        };
        let sm = lower(&KernelTemplate::Reduction {
            op: ReduceOp::Sum,
            prologue,
            extra_prologues: vec![],
            epilogue: Some(ReductionEpilogue {
                dag: epi_dag,
                n_per_col_inputs: 0,
            }),
            n_per_elem: 1,
            n_per_row: 1,
            gather_elem: Vec::new(),
            input_row_repeats: Vec::new(),
            rows_per_workgroup: 1,
            grid: GridShape::default(),
        });
        // Bindings: src + per_row_src (for row_max) + dst.
        assert!(sm.source.contains("var<storage> src: array<f32>;"));
        assert!(sm.source.contains("var<storage> per_row_src: array<f32>;"));
        // Per-row load uses `[row]` (the workgroup id), not `[col]`.
        assert!(sm.source.contains("per_row_src[row]"));
        // Writeback shape is [outer, inner].
        assert!(sm.source.contains("dst[row_offset + wcol]"));
        // Prologue has the exp + sub.
        assert!(sm.source.contains("exp(v2)"));
        // Epilogue does the division by the reduced value.
        assert!(sm.source.contains("v4 / v2") || sm.source.contains("/ v"));
        // Naga-valid.
        use naga::valid::{Capabilities, ValidationFlags, Validator};
        let flags = ValidationFlags::all() ^ ValidationFlags::BINDINGS;
        Validator::new(flags, Capabilities::all())
            .validate(&sm.module)
            .unwrap_or_else(|e| panic!("softmax pass-2 invalid: {:?}\n{}", e, sm.source));
    }

    #[test]
    fn reduction_with_epilogue_naga_validates() {
        use naga::valid::{Capabilities, ValidationFlags, Validator};
        let flags = ValidationFlags::all() ^ ValidationFlags::BINDINGS;
        let sm = lower(&KernelTemplate::Reduction {
            op: ReduceOp::Sum,
            prologue: square_prologue(),
            extra_prologues: vec![],
            epilogue: Some(rmsnorm_epilogue(720.0, 1e-5)),
            n_per_elem: 1,
            n_per_row: 0,
            gather_elem: Vec::new(),
            input_row_repeats: Vec::new(),
            rows_per_workgroup: 1,
            grid: GridShape::default(),
        });
        Validator::new(flags, Capabilities::all())
            .validate(&sm.module)
            .unwrap_or_else(|e| panic!("RMSNorm reduction invalid: {:?}\n{}", e, sm.source));
    }

    #[test]
    fn packed_narrow_reduction_lowers_and_validates() {
        use naga::valid::{Capabilities, ValidationFlags, Validator};
        let flags = ValidationFlags::all() ^ ValidationFlags::BINDINGS;
        let sm = lower(&KernelTemplate::Reduction {
            op: ReduceOp::Sum,
            prologue: identity_prologue(),
            extra_prologues: vec![square_prologue()],
            epilogue: None,
            n_per_elem: 1,
            n_per_row: 0,
            gather_elem: Vec::new(),
            input_row_repeats: Vec::new(),
            rows_per_workgroup: 16,
            grid: GridShape::default(),
        });
        assert!(sm.source.contains("let row_slot = tid / 16u;"));
        assert!(sm.source.contains("let lane = tid % 16u;"));
        assert!(sm.source.contains("let row = wgid.x * 16u + row_slot;"));
        assert!(sm.source.contains("if lane < 8u"));
        assert!(sm.source.contains("wg_data_0[tid]"));
        assert!(sm.source.contains("wg_data_1[tid]"));
        assert!(sm.source.contains("dst[row * 2u + 1u]"));
        Validator::new(flags, Capabilities::all())
            .validate(&sm.module)
            .unwrap_or_else(|e| panic!("packed reduction invalid: {:?}\n{}", e, sm.source));
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
            extra_prologues: vec![],
            epilogue: Some(bad),
            n_per_elem: 1,
            n_per_row: 0,
            gather_elem: Vec::new(),
            input_row_repeats: Vec::new(),
            rows_per_workgroup: 1,
            grid: GridShape::default(),
        });
    }

    #[test]
    #[should_panic(expected = "workgroup_size must be a power of 2")]
    fn reduction_rejects_odd_workgroup_size() {
        lower(&KernelTemplate::Reduction {
            op: ReduceOp::Sum,
            prologue: identity_prologue(),
            extra_prologues: vec![],
            epilogue: None,
            n_per_elem: 1,
            n_per_row: 0,
            gather_elem: Vec::new(),
            input_row_repeats: Vec::new(),
            rows_per_workgroup: 1,
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

    #[test]
    fn wide_arity_and_vec4_pointwise_validate_to_spirv() {
        // 6-input DAG exercising wider-than-3 stack-backed DynPointwiseData bindings,
        // broadcasts, and all special vec4 op lowerings (Erf, Tanh, Greater, U32ToF32, F32ToU32).
        let dag = PointwiseDAG {
            n_inputs: 6,
            ops: vec![
                Pw::LoadInput(0),
                Pw::LoadInput(1),
                Pw::LoadBroadcast {
                    input: 2,
                    divisor: 1,
                    modulus: 16,
                },
                Pw::LoadBroadcast {
                    input: 3,
                    divisor: 16,
                    modulus: 8,
                },
                Pw::LoadInput(4),
                Pw::LoadInput(5),
                Pw::Add(0, 1),
                Pw::Add(6, 2),
                Pw::Mul(7, 3),
                Pw::Erf(8),
                Pw::Tanh(9),
                Pw::Greater(10, 4),
                Pw::F32ToU32(11),
                Pw::U32ToF32(12),
                Pw::Silu(13),
                Pw::Sub(14, 5),
                Pw::const_f32(0.25),
                Pw::Add(15, 16),
            ],
            output: 17,
        };
        let kinds = dag
            .vec4_input_kinds(PointwiseStorage::default(), 64)
            .expect("aligned broadcasts should vectorize");
        assert_eq!(kinds, vec![true, true, true, false, true, true]);
        assert!(!dag.can_vectorize(PointwiseStorage::default(), 62));

        let mut sm = lower_pointwise_vec4(&dag, GridShape::default(), &kinds);
        assert!(sm.source.contains("var<storage> src_f: array<vec4<f32>>;"));
        assert!(sm.source.contains("var<storage> src_d: array<f32>;"));
        crate::codegen::validate_shader_for_test(
            &mut sm.module,
            naga::valid::Capabilities::all(),
            (1, 3),
        )
        .unwrap_or_else(|e| panic!("vec4 pointwise spirv failed: {e:?}\n{}", sm.source));
    }

    #[test]
    fn subgroup_and_vec4_reductions_validate_to_spirv() {
        let rms = ReductionKernel {
            op: ReduceOp::Sum,
            prologue: square_prologue(),
            extra_prologues: vec![],
            epilogue: Some(ReductionEpilogue {
                dag: PointwiseDAG {
                    n_inputs: 3,
                    ops: vec![
                        Pw::LoadInput(0),
                        Pw::LoadInput(1),
                        Pw::LoadInput(2),
                        Pw::const_f32(1e-5),
                        Pw::Add(2, 3),
                        Pw::Rsqrt(4),
                        Pw::Mul(0, 5),
                        Pw::Mul(6, 1),
                    ],
                    output: 7,
                },
                n_per_col_inputs: 1,
            }),
            n_per_elem: 1,
            n_per_row: 0,
            workgroup_size: 256,
            rows_per_workgroup: 1,
            gather_elem: vec![],
            input_row_repeats: vec![],
            preserve_tree_order: false,
        };
        assert!(rms.can_vectorize(PointwiseStorage::default(), 64));
        assert!(!rms.can_vectorize(PointwiseStorage::default(), 62));

        for (op, with_epi, vec4) in [
            (ReduceOp::Sum, false, false),
            (ReduceOp::Sum, false, true),
            (ReduceOp::Sum, true, true),
            (ReduceOp::Max, false, true),
            (ReduceOp::ArgMax, false, false),
        ] {
            let mut k = rms.clone();
            k.op = op;
            if !with_epi {
                k.epilogue = None;
                k.prologue = identity_prologue();
            }
            let mut sm = k.lower_with_options(PointwiseStorage::default(), true, vec4);
            assert!(sm.source.contains("subgroup"));
            assert_eq!(
                sm.source.matches("workgroupBarrier()").count(),
                1,
                "subgroup reduction should use at most 1 barrier:\n{}",
                sm.source
            );
            crate::codegen::validate_shader_for_test(
                &mut sm.module,
                naga::valid::Capabilities::all(),
                (1, 3),
            )
            .unwrap_or_else(|e| panic!("reduction spirv failed: {e:?}\n{}", sm.source));
        }

        // Packed single-lane rows emit zero barriers and zero shared arrays.
        let mut packed = rms.clone();
        packed.epilogue = None;
        packed.prologue = identity_prologue();
        packed.rows_per_workgroup = 256;
        let sm = packed.lower_with_options(PointwiseStorage::default(), true, false);
        assert_eq!(sm.source.matches("workgroupBarrier()").count(), 0);
        assert!(!sm.source.contains("var<workgroup>"));
    }
}
