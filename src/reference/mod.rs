//! Reference semantics for every graph op, evaluated in `f64`.
//!
//! This module is the single written definition of what each [`Op`] means.
//! Every kernel, fusion, rewrite and derivative the compiler produces is
//! checked against it:
//!
//! * [`evaluate`] runs any graph, including graphs produced by
//!   [`crate::autodiff::differentiate`], one node at a time with naive loops
//!   in double precision. It never shares code with the GPU lowering.
//! * [`check`] compares a GPU result with a reference result element by
//!   element, using a tolerance scaled to each element's condition rather
//!   than one fixed threshold (see [`magnitudes`]).
//! * [`gradients`] checks the autodiff rules alone, in `f64` on the CPU.
//!
//! The dispatch in [`eval_node`] is an exhaustive `match`: adding an `Op`
//! without giving it reference semantics does not compile.

mod attention;
mod basic;
pub mod gpu;
pub mod gradients;
mod norm;
mod vision;

use crate::graph::{DType, Graph, Node, NodeId, Op};
use std::collections::HashMap;
use std::fmt;

/// A dense tensor of `f64` values in row-major order.
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f64>,
}

impl Tensor {
    pub fn new(shape: Vec<usize>, data: Vec<f64>) -> Self {
        assert_eq!(
            shape.iter().product::<usize>(),
            data.len(),
            "tensor data does not fill shape {shape:?}"
        );
        Self { shape, data }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// Read element `i` as an index (for `U32` inputs).
    fn index(&self, i: usize) -> usize {
        self.data[i] as usize
    }
}

/// Why a graph could not be evaluated.
#[derive(Clone, Debug, PartialEq)]
pub enum Error {
    /// No value was supplied for this input or parameter.
    MissingFeed(String),
    /// A supplied value has the wrong number of elements.
    FeedLength {
        name: String,
        expected: usize,
        got: usize,
    },
    /// The op has no reference semantics for this configuration.
    Unsupported { node: NodeId, reason: String },
    /// The graph or its inputs violate the op's contract.
    Invalid { node: NodeId, reason: String },
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            Error::MissingFeed(ref name) => write!(f, "no value supplied for {name:?}"),
            Error::FeedLength {
                ref name,
                expected,
                got,
            } => write!(f, "{name:?} needs {expected} values, got {got}"),
            Error::Unsupported { node, ref reason } => {
                write!(f, "node %{node} has no reference: {reason}")
            }
            Error::Invalid { node, ref reason } => write!(f, "node %{node} is invalid: {reason}"),
        }
    }
}

impl std::error::Error for Error {}

/// Values for the graph's named inputs and parameters.
///
/// Values are rounded to the precision the GPU receives (`f32`, or `f16`
/// for half-precision parameters) when they are fed, so the reference and
/// the device start from bit-identical operands.
#[derive(Clone, Debug, Default)]
pub struct Feeds {
    values: HashMap<String, Vec<f64>>,
}

impl Feeds {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a floating-point input or parameter, rounded to `f32`.
    pub fn set(&mut self, name: &str, data: &[f32]) -> &mut Self {
        self.values.insert(
            name.to_string(),
            data.iter().map(|&v| f64::from(v)).collect(),
        );
        self
    }

    /// Set a `U32` input.
    pub fn set_u32(&mut self, name: &str, data: &[u32]) -> &mut Self {
        self.values.insert(
            name.to_string(),
            data.iter().map(|&v| f64::from(v)).collect(),
        );
        self
    }

    pub fn get(&self, name: &str) -> Option<&[f64]> {
        self.values.get(name).map(Vec::as_slice)
    }

    /// The value as the `f32` slice the device receives.
    pub fn f32(&self, name: &str) -> Option<Vec<f32>> {
        self.get(name)
            .map(|v| v.iter().map(|&x| x as f32).collect())
    }

    /// The value as `u32` indices.
    pub fn u32(&self, name: &str) -> Option<Vec<u32>> {
        self.get(name)
            .map(|v| v.iter().map(|&x| x as u32).collect())
    }

    /// Fill every floating-point input and parameter of `graph` that has no
    /// value yet with uniform values in `[-scale, scale]`. `U32` inputs are
    /// domain-specific (indices, positions, lengths) and must be set by the
    /// caller.
    pub fn fill_random(&mut self, graph: &Graph, seed: u64, scale: f32) -> &mut Self {
        let mut rng = Rng::new(seed);
        for node in graph.nodes() {
            let name = match node.op {
                Op::Input { ref name } | Op::Parameter { ref name } => name,
                _ => continue,
            };
            if self.values.contains_key(name) || node.ty.dtype == DType::U32 {
                continue;
            }
            let data: Vec<f32> = (0..node.ty.num_elements())
                .map(|_| rng.uniform(-scale, scale))
                .collect();
            self.set(name, &data);
        }
        self
    }
}

/// A small deterministic generator (SplitMix64) so tests need no extra
/// dependency and reproduce from a seed.
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed ^ 0x9E37_79B9_7F4A_7C15)
    }

    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[0, 1)`.
    pub fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    pub fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        (f64::from(lo) + self.unit() * f64::from(hi - lo)) as f32
    }

    /// Uniform integer in `[0, n)`.
    pub fn below(&mut self, n: u32) -> u32 {
        (self.unit() * f64::from(n)) as u32
    }
}

fn round_f16(v: f64) -> f64 {
    f64::from(half::f16::from_f64(v).to_f64() as f32)
}

/// Evaluate every node of `graph`. The result is indexed by node id.
pub fn evaluate(graph: &Graph, feeds: &Feeds) -> Result<Vec<Tensor>, Error> {
    let mut values: Vec<Tensor> = Vec::with_capacity(graph.nodes().len());
    for node in graph.nodes() {
        debug_assert_eq!(node.id as usize, values.len(), "graph is not in id order");
        let inputs: Vec<&Tensor> = node.inputs.iter().map(|&i| &values[i as usize]).collect();
        let value = eval_node(graph, node, &inputs, feeds)?;
        values.push(value);
    }
    Ok(values)
}

/// Evaluate the graph's outputs only.
pub fn evaluate_outputs(graph: &Graph, feeds: &Feeds) -> Result<Vec<Tensor>, Error> {
    let values = evaluate(graph, feeds)?;
    Ok(graph
        .outputs()
        .iter()
        .map(|&o| values[o as usize].clone())
        .collect())
}

fn leaf(node: &Node, name: &str, feeds: &Feeds) -> Result<Vec<f64>, Error> {
    let data = feeds
        .get(name)
        .ok_or_else(|| Error::MissingFeed(name.to_string()))?;
    let expected = node.ty.num_elements();
    if data.len() != expected {
        return Err(Error::FeedLength {
            name: name.to_string(),
            expected,
            got: data.len(),
        });
    }
    match node.ty.dtype {
        DType::F32 | DType::U32 => Ok(data.to_vec()),
        DType::F16 => Ok(data.iter().map(|&v| round_f16(v)).collect()),
        other => Err(Error::Unsupported {
            node: node.id,
            reason: format!("{other:?} parameters are packed; feed their dequantized values"),
        }),
    }
}

/// Evaluate one node from its already-evaluated inputs.
pub fn eval_node(
    graph: &Graph,
    node: &Node,
    inputs: &[&Tensor],
    feeds: &Feeds,
) -> Result<Tensor, Error> {
    let data = match node.op {
        Op::Parameter { ref name } | Op::Input { ref name } => leaf(node, name, feeds)?,
        Op::Constant { ref data } => data.iter().map(|&v| f64::from(v)).collect(),

        Op::MatMul
        | Op::MatMulAT
        | Op::MatMulBT
        | Op::FusedMatMulAdd
        | Op::FusedMatMulATAdd
        | Op::FusedMatMulBTAdd
        | Op::BlockMatMul
        | Op::BlockMatMulAT { .. }
        | Op::BlockMatMulBT
        | Op::Add
        | Op::Mul
        | Op::Greater
        | Op::Relu
        | Op::Sigmoid
        | Op::Tanh
        | Op::Neg
        | Op::Abs
        | Op::Log
        | Op::Recip
        | Op::Exp
        | Op::Softplus { .. }
        | Op::SoftplusGrad { .. }
        | Op::Clamp { .. }
        | Op::Scale { .. }
        | Op::SumAll
        | Op::MeanAll
        | Op::SumRows
        | Op::SumInner
        | Op::BroadcastInner { .. }
        | Op::NormalizeInnerSum { .. }
        | Op::NormalizeInnerSumGrad { .. }
        | Op::ExclusiveCumsum { .. }
        | Op::ShiftInner { .. }
        | Op::Softmax
        | Op::LogSoftmax
        | Op::CrossEntropyLoss
        | Op::CrossEntropyLogitsGrad
        | Op::BceLoss
        | Op::Transpose
        | Op::BiasAdd
        | Op::BiasMul
        | Op::Nop
        | Op::Identity
        | Op::Materialize
        | Op::StopGradient
        | Op::ScatterAdd { .. }
        | Op::Silu
        | Op::SiluGrad
        | Op::Gelu
        | Op::SwiGLU
        | Op::SwiGLUConcat
        | Op::SwiGLUConcatGrad
        | Op::SwiGLUGradGate
        | Op::SwiGLUGradUp
        | Op::GeGLU
        | Op::GeGLUConcat
        | Op::GeGLUConcatGrad
        | Op::Embedding
        | Op::ToF16
        | Op::CacheWrite
        | Op::CacheWritePrefix
        | Op::PrefixLast
        | Op::MulPerChannel { .. }
        | Op::AddPerChannel { .. }
        | Op::GlobalAvgPool { .. }
        | Op::GlobalAvgPoolGrad { .. }
        | Op::Concat { .. }
        | Op::SplitA { .. }
        | Op::SplitB { .. }
        | Op::Upsample2x { .. }
        | Op::Upsample2xGrad { .. } => basic::eval(node, inputs)?,

        Op::RmsNorm { .. }
        | Op::RmsNormGradW { .. }
        | Op::RmsNormGradX { .. }
        | Op::LayerNorm { .. }
        | Op::LayerNormGradWB { .. }
        | Op::LayerNormGradX { .. }
        | Op::RoPE { .. }
        | Op::RoPEGrad { .. }
        | Op::RoPEPositions { .. }
        | Op::PairwiseSquaredDistance { .. }
        | Op::PairwiseVectorRejection { .. }
        | Op::PairwiseGrad { .. } => norm::eval(node, inputs)?,

        Op::Conv2d { .. }
        | Op::Conv2dDw { .. }
        | Op::Conv2dGradInput { .. }
        | Op::Conv2dGradWeight { .. }
        | Op::WinogradConv2d { .. }
        | Op::MaxPool2d { .. }
        | Op::MaxPool2dGrad { .. }
        | Op::GroupNorm { .. }
        | Op::GroupNormSilu { .. }
        | Op::GroupNormGradInput { .. }
        | Op::GroupNormGradWeightBias { .. } => vision::eval(node, inputs)?,

        Op::CausalAttention { .. }
        | Op::CausalAttentionRoPE { .. }
        | Op::FullAttention { .. }
        | Op::CrossAttention { .. }
        | Op::MultiHeadAttn { .. }
        | Op::MultiHeadAttnGradQ { .. }
        | Op::MultiHeadAttnGradK { .. }
        | Op::MultiHeadAttnGradV { .. }
        | Op::SlidingWindowAttention { .. }
        | Op::CachedAttention { .. }
        | Op::CachedBlockAttention { .. }
        | Op::ChunkedRelativeAttention { .. } => attention::eval(graph, node, inputs)?,
    };
    let expected = node.ty.num_elements();
    if data.len() != expected {
        return Err(Error::Invalid {
            node: node.id,
            reason: format!(
                "{:?} produced {} values for type {}",
                node.op,
                data.len(),
                node.ty
            ),
        });
    }
    Ok(Tensor::new(node.ty.shape.clone(), data))
}

/// Per-element scale of the rounding error a correct `f32` evaluation of
/// `node` may make, given exact inputs.
///
/// For a sum of products such as a matmul or convolution, the error bound is
/// proportional to `Σ|aᵢ·bᵢ|`, not to the result: cancellation can make the
/// result tiny while the error stays at the size of the terms. For those
/// ops this evaluates the same op on the absolute values of its inputs.
/// Other ops use the magnitude of the result, or an op-specific bound.
pub fn magnitudes(
    graph: &Graph,
    node: &Node,
    inputs: &[&Tensor],
    output: &Tensor,
) -> Result<Vec<f64>, Error> {
    if let Some(custom) = custom_magnitude(graph, node, inputs, output) {
        return Ok(custom);
    }
    if is_multilinear(&node.op) {
        let absolute: Vec<Tensor> = inputs
            .iter()
            .map(|t| Tensor::new(t.shape.clone(), t.data.iter().map(|v| v.abs()).collect()))
            .collect();
        let refs: Vec<&Tensor> = absolute.iter().collect();
        let abs_out = eval_node(graph, node, &refs, &Feeds::default())?;
        return Ok(abs_out
            .data
            .iter()
            .zip(&output.data)
            .map(|(a, b)| a.max(b.abs()))
            .collect());
    }
    Ok(output.data.iter().map(|v| v.abs()).collect())
}

/// Per-element error scales for every node of `graph`, given the values
/// [`evaluate`] produced.
///
/// Rounding errors made upstream travel through linear ops unchanged in
/// size, however much the values themselves cancel: the gradient of a
/// softmax over one column is exactly zero, yet each term that cancelled
/// was of order one. For ops that are linear in each input (and for 1-Lipschitz
/// selections such as `Relu`) the scale is therefore the op applied to its
/// inputs' scales, or [`magnitudes`] if that is larger. Smooth elementwise
/// ops carry `|f'(x)|` times their input's scale, to first order. Rotations
/// use the absolute rotation matrix. Other ops restart from [`magnitudes`].
pub fn error_scales(graph: &Graph, values: &[Tensor]) -> Result<Vec<Vec<f64>>, Error> {
    let mut scales: Vec<Vec<f64>> = Vec::with_capacity(values.len());
    for node in graph.nodes() {
        let inputs: Vec<&Tensor> = node.inputs.iter().map(|&i| &values[i as usize]).collect();
        let out = &values[node.id as usize];
        let propagates = propagates_scale(&node.op);
        // For a propagating op the scales of its inputs dominate their
        // absolute values, so the op on those scales bounds the op on
        // absolute inputs that `magnitudes` would evaluate again.
        let mut scale = match custom_magnitude(graph, node, &inputs, out) {
            Some(custom) => custom,
            None if propagates => out.data.iter().map(|v| v.abs()).collect(),
            None => magnitudes(graph, node, &inputs, out)?,
        };
        if propagates {
            let input_scales: Vec<Tensor> = node
                .inputs
                .iter()
                .map(|&i| Tensor::new(values[i as usize].shape.clone(), scales[i as usize].clone()))
                .collect();
            let refs: Vec<&Tensor> = input_scales.iter().collect();
            let propagated = eval_node(graph, node, &refs, &Feeds::default())?;
            for (s, p) in scale.iter_mut().zip(&propagated.data) {
                *s = s.max(p.abs());
            }
        } else if matches!(
            node.op,
            Op::RoPE { .. } | Op::RoPEGrad { .. } | Op::RoPEPositions { .. }
        ) {
            let propagated =
                norm::rotation_error_scale(node, &inputs, &scales[node.inputs[0] as usize])?;
            for (s, p) in scale.iter_mut().zip(propagated) {
                *s = s.max(p);
            }
        } else if node.inputs.len() == 1 && basic::derivative(&node.op, 0.0).is_some() {
            // First order: an input error δ becomes |f'(x)|·δ.
            let x = &values[node.inputs[0] as usize];
            let input_scale = &scales[node.inputs[0] as usize];
            for (i, s) in scale.iter_mut().enumerate() {
                let slope = basic::derivative(&node.op, x.data[i]).unwrap_or(0.0);
                *s = s.max(slope.abs() * input_scale[i]);
            }
        }
        scales.push(scale);
    }
    Ok(scales)
}

/// Ops through which an input's error scale bounds the output's.
fn propagates_scale(op: &Op) -> bool {
    is_multilinear(op)
        || matches!(
            *op,
            Op::Mul
                | Op::Neg
                | Op::Scale { .. }
                | Op::Relu
                | Op::Abs
                | Op::Transpose
                | Op::Identity
                | Op::Materialize
                | Op::StopGradient
                | Op::BroadcastInner { .. }
                | Op::BiasMul
                | Op::MulPerChannel { .. }
                | Op::ShiftInner { .. }
                | Op::Concat { .. }
                | Op::SplitA { .. }
                | Op::SplitB { .. }
                | Op::Upsample2x { .. }
                | Op::GlobalAvgPoolGrad { .. }
                | Op::Embedding
                | Op::PrefixLast
        )
}

/// An op-specific error scale, where one is defined.
fn custom_magnitude(
    graph: &Graph,
    node: &Node,
    inputs: &[&Tensor],
    output: &Tensor,
) -> Option<Vec<f64>> {
    basic::magnitude(node, inputs, output)
        .or_else(|| norm::magnitude(node, inputs, output))
        .or_else(|| vision::magnitude(node, inputs, output))
        .or_else(|| attention::magnitude(graph, node, inputs, output))
}

/// Ops whose outputs are sums of products of their inputs, so that the op
/// applied to absolute inputs bounds the size of every summed term.
fn is_multilinear(op: &Op) -> bool {
    matches!(
        *op,
        Op::MatMul
            | Op::MatMulAT
            | Op::MatMulBT
            | Op::FusedMatMulAdd
            | Op::FusedMatMulATAdd
            | Op::FusedMatMulBTAdd
            | Op::BlockMatMul
            | Op::BlockMatMulAT { .. }
            | Op::BlockMatMulBT
            | Op::Add
            | Op::SumAll
            | Op::MeanAll
            | Op::SumRows
            | Op::SumInner
            | Op::ExclusiveCumsum { .. }
            | Op::BiasAdd
            | Op::ScatterAdd { .. }
            | Op::AddPerChannel { .. }
            | Op::GlobalAvgPool { .. }
            | Op::Upsample2xGrad { .. }
            | Op::Conv2d { .. }
            | Op::Conv2dDw { .. }
            | Op::Conv2dGradInput { .. }
            | Op::Conv2dGradWeight { .. }
            | Op::WinogradConv2d { .. }
    )
}

/// Tolerance for comparing an `f32` device result with the reference.
///
/// Element `i` passes when `|got - want| <= rtol * (m_i + floor * max(m))`,
/// where `m` is the per-element [`magnitudes`]. `rtol` bounds relative
/// rounding error; the `floor` term keeps elements that are tiny relative to
/// the rest of the tensor from demanding precision beyond what the others
/// can have (for example `tanh` near zero, evaluated through `exp`).
#[derive(Clone, Copy, Debug)]
pub struct Tolerance {
    pub rtol: f64,
    pub floor: f64,
}

impl Default for Tolerance {
    fn default() -> Self {
        Self {
            rtol: 2e-4,
            floor: 1e-3,
        }
    }
}

/// The worst element of a comparison.
#[derive(Clone, Debug)]
pub struct Mismatch {
    pub index: usize,
    pub got: f64,
    pub want: f64,
    pub allowed: f64,
    /// How many elements exceed their tolerance.
    pub failures: usize,
}

impl fmt::Display for Mismatch {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} element(s) out of tolerance; worst [{}]: got {:e}, want {:e}, error {:e} > allowed {:e}",
            self.failures,
            self.index,
            self.got,
            self.want,
            (self.got - self.want).abs(),
            self.allowed
        )
    }
}

/// Compare `got` against `want`, returning the worst failing element.
/// Non-finite device values fail unless the reference has the same value.
pub fn check(got: &[f32], want: &[f64], magnitude: &[f64], tol: Tolerance) -> Result<(), Mismatch> {
    let got: Vec<f64> = got.iter().map(|&v| f64::from(v)).collect();
    check_f64(&got, want, magnitude, tol)
}

/// [`check`] for values computed in double precision.
pub fn check_f64(
    got: &[f64],
    want: &[f64],
    magnitude: &[f64],
    tol: Tolerance,
) -> Result<(), Mismatch> {
    assert_eq!(got.len(), want.len(), "compared tensors differ in length");
    assert_eq!(magnitude.len(), want.len(), "missing element error scales");
    let max_mag = magnitude
        .iter()
        .copied()
        .filter(|m| m.is_finite())
        .fold(0.0f64, f64::max);
    let mut worst: Option<(f64, Mismatch)> = None;
    let mut failures = 0;
    for (i, ((&g, &w), &m)) in got.iter().zip(want).zip(magnitude).enumerate() {
        let allowed = tol.rtol * (m + tol.floor * max_mag);
        let error = (g - w).abs();
        let ok = if !w.is_finite() || !g.is_finite() {
            g == w || (g.is_nan() && w.is_nan())
        } else {
            allowed.is_finite() && allowed >= 0.0 && error <= allowed
        };
        if ok {
            continue;
        }
        failures += 1;
        let excess = if error.is_finite() {
            error / allowed.max(f64::MIN_POSITIVE)
        } else {
            f64::INFINITY
        };
        if worst.as_ref().is_none_or(|w| excess > w.0) {
            worst = Some((
                excess,
                Mismatch {
                    index: i,
                    got: g,
                    want: w,
                    allowed,
                    failures: 0,
                },
            ));
        }
    }
    match worst {
        None => Ok(()),
        Some((_, mut m)) => {
            m.failures = failures;
            Err(m)
        }
    }
}

/// One compared value.
#[derive(Clone, Debug)]
pub struct Comparison {
    /// What was compared: an output's op, or a parameter's gradient.
    pub what: String,
    pub result: Result<(), Mismatch>,
}

/// Every comparison a check made.
#[derive(Clone, Debug, Default)]
pub struct Report {
    pub comparisons: Vec<Comparison>,
}

impl Report {
    pub fn passed(&self) -> bool {
        self.comparisons.iter().all(|c| c.result.is_ok())
    }

    /// Panic with every failing comparison.
    #[track_caller]
    pub fn assert_passed(&self, context: &str) {
        assert!(self.passed(), "{context}:\n{self}");
    }
}

impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for c in &self.comparisons {
            match c.result {
                Ok(()) => writeln!(f, "  ok   {}", c.what)?,
                Err(ref m) => writeln!(f, "  FAIL {}: {m}", c.what)?,
            }
        }
        Ok(())
    }
}
