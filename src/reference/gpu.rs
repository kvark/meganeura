//! Run a graph on the GPU and compare it with the reference.

use super::{Comparison, Error, Feeds, Report, Tensor, Tolerance, check, error_scales, evaluate};
use crate::graph::{DType, Graph, Op};
use crate::train::{Mode, SessionConfig, build};

/// How to build the session under test.
#[derive(Clone, Debug)]
pub struct Options {
    pub tolerance: Tolerance,
    /// Fill every buffer that holds no parameter with NaN before the step
    /// (see [`crate::SessionOptions::poison`]).
    pub poison: bool,
    /// Compile options for the session under test.
    pub compile: crate::CompileOptions,
    /// Graph-rewrite configuration for the session under test.
    pub optimize: crate::OptimizeConfig,
    /// Cooperative-matrix policy for the session under test.
    pub coop: crate::CoopPolicy,
}

impl Default for Options {
    fn default() -> Self {
        let env = SessionConfig::from_env();
        Self {
            tolerance: Tolerance::default(),
            poison: true,
            compile: env.options,
            optimize: env.optimize,
            coop: env.runtime.coop,
        }
    }
}

impl Options {
    /// The default options and one variant per alternative lowering: the
    /// hand-written pointwise and reduction shaders instead of the
    /// generated ones, and no dispatch fusion. Each must compute the same
    /// values, so suites run their cases under all of them.
    pub fn lowerings() -> Vec<(&'static str, Options)> {
        let default = Options::default();
        let mut hand_pointwise = default.clone();
        hand_pointwise.compile.use_schedule_pointwise = false;
        let mut hand_reduction = default.clone();
        hand_reduction.compile.use_schedule_reduction = false;
        let mut unfused = default.clone();
        unfused.compile.fuse_dispatches = false;
        vec![
            ("default", default),
            ("hand-written pointwise", hand_pointwise),
            ("hand-written reductions", hand_reduction),
            ("unfused", unfused),
        ]
    }

    fn session_config(&self, mode: Mode) -> SessionConfig<'static> {
        let mut config = SessionConfig::from_env();
        config.mode = mode;
        config.options = self.compile.clone();
        config.optimize = self.optimize;
        config.runtime.coop = self.coop;
        config.runtime.poison = self.poison;
        config
    }
}

fn upload(session: &mut crate::Session, graph: &Graph, feeds: &Feeds) -> Result<(), Error> {
    for node in graph.nodes() {
        match node.op {
            Op::Input { ref name } => {
                if node.ty.dtype == DType::U32 {
                    let data = feeds
                        .u32(name)
                        .ok_or_else(|| Error::MissingFeed(name.clone()))?;
                    session.set_input_u32(name, &data);
                } else {
                    let data = feeds
                        .f32(name)
                        .ok_or_else(|| Error::MissingFeed(name.clone()))?;
                    session.set_input(name, &data);
                }
            }
            Op::Parameter { ref name } => {
                let data = feeds
                    .f32(name)
                    .ok_or_else(|| Error::MissingFeed(name.clone()))?;
                session.set_parameter(name, &data);
            }
            _ => {}
        }
    }
    Ok(())
}

/// Reference values and error scales of every node.
fn reference(graph: &Graph, feeds: &Feeds) -> Result<(Vec<Tensor>, Vec<Vec<f64>>), Error> {
    let values = evaluate(graph, feeds)?;
    let scales = error_scales(graph, &values)?;
    Ok((values, scales))
}

/// Build an inference session for `graph`, run one step, and compare every
/// graph output with the reference.
pub fn check_inference(graph: &Graph, feeds: &Feeds, options: &Options) -> Result<Report, Error> {
    let (values, scales) = reference(graph, feeds)?;
    let (mut session, _) = build(graph, options.session_config(Mode::Inference));
    upload(&mut session, graph, feeds)?;
    session.step();
    session.wait();
    let mut report = Report::default();
    for (index, &id) in graph.outputs().iter().enumerate() {
        let node = graph.node(id);
        let (want, magnitude) = (&values[id as usize], &scales[id as usize]);
        let mut got = vec![0.0f32; want.len()];
        session.read_output_by_index(index, &mut got);
        report.comparisons.push(Comparison {
            what: format!("output {index} (%{id} {})", op_name(&node.op)),
            result: check(&got, &want.data, magnitude, options.tolerance),
        });
    }
    Ok(report)
}

/// Build a training session for `graph` (whose first output is a scalar
/// loss), run one step without an optimizer update, and compare the loss,
/// every further output, and every parameter gradient with the reference.
///
/// The reference gradients come from evaluating
/// [`crate::autodiff::differentiate`] in `f64`; [`super::gradients`] checks
/// those against finite differences independently of any kernel.
pub fn check_training(graph: &Graph, feeds: &Feeds, options: &Options) -> Result<Report, Error> {
    let backward = crate::autodiff::differentiate(graph);
    let (values, scales) = reference(&backward, feeds)?;
    let (mut session, _) = build(graph, options.session_config(Mode::Training));
    upload(&mut session, graph, feeds)?;
    session.step();
    session.wait();

    let mut report = Report::default();
    let loss_id = backward.outputs()[0];
    let (want, magnitude) = (&values[loss_id as usize], &scales[loss_id as usize]);
    report.comparisons.push(Comparison {
        what: "loss".to_string(),
        result: check(
            &[session.read_loss()],
            &want.data,
            magnitude,
            options.tolerance,
        ),
    });
    // Further user outputs must survive the forward rewrites and keep their
    // values through the backward pass that shares their buffers.
    for index in 1..backward.num_user_outputs() {
        let id = backward.outputs()[index];
        let (want, magnitude) = (&values[id as usize], &scales[id as usize]);
        let mut got = vec![0.0f32; want.len()];
        if index >= session.num_outputs() {
            report.comparisons.push(Comparison {
                what: format!("output {index} (%{id}) is missing from the session"),
                result: check(&[f32::NAN], &[0.0], &[0.0], options.tolerance),
            });
            continue;
        }
        session.read_output_by_index(index, &mut got);
        report.comparisons.push(Comparison {
            what: format!("output {index} (%{id} {})", op_name(&backward.node(id).op)),
            result: check(&got, &want.data, magnitude, options.tolerance),
        });
    }

    let params = graph.nodes().iter().filter_map(|n| match n.op {
        Op::Parameter { ref name } => Some(name.clone()),
        _ => None,
    });
    let grads = &backward.outputs()[backward.num_user_outputs()..];
    for (name, &grad_id) in params.zip(grads) {
        if let Some(got) = packed_gradient(&session, &name) {
            let (want, magnitude) = (&values[grad_id as usize], &scales[grad_id as usize]);
            report.comparisons.push(Comparison {
                what: format!("gradient of {name} (packed)"),
                result: check(&got, &want.data, magnitude, options.tolerance),
            });
            continue;
        }
        if !session.has_param_grad(&name) {
            // The compiler omits gradients of parameters that do not reach
            // the loss. The reference must agree they are zero.
            let want = &values[grad_id as usize];
            report.comparisons.push(Comparison {
                what: format!("gradient of {name} (absent on device)"),
                result: check(
                    &vec![0.0; want.len()],
                    &want.data,
                    &vec![0.0; want.len()],
                    options.tolerance,
                ),
            });
            continue;
        }
        let (want, magnitude) = (&values[grad_id as usize], &scales[grad_id as usize]);
        let mut got = vec![0.0f32; want.len()];
        session.read_param_grad(&name, &mut got);
        report.comparisons.push(Comparison {
            what: format!("gradient of {name}"),
            result: check(&got, &want.data, magnitude, options.tolerance),
        });
    }
    Ok(report)
}

/// The gradient of parameter `name` when the optimizer replaced it with a
/// packed parameter (for example gate and up projections concatenated for
/// a fused GLU): its slice of the packed parameter's gradient.
fn packed_gradient(session: &crate::Session, name: &str) -> Option<Vec<f32>> {
    use crate::graph::ParamTransform;
    if session.has_param_grad(name) {
        return None;
    }
    let plan = session.plan();
    let &(buffer, ref sources, ref transform) = plan
        .derived_params
        .iter()
        .find(|entry| entry.1.iter().any(|source| source.0 == name))?;
    let packed = &plan.param_buffers.iter().find(|entry| entry.1 == buffer)?.0;
    let ty = plan.param_types.get(&buffer)?;
    let mut grad = vec![0.0f32; ty.num_elements()];
    session.read_param_grad(packed, &mut grad);
    let (rows, width) = (ty.shape[0], ty.num_elements() / ty.shape[0]);
    let offset: usize = sources
        .iter()
        .take_while(|source| source.0 != name)
        .map(|source| source.1)
        .sum();
    let extent = sources.iter().find(|source| source.0 == name)?.1;
    match *transform {
        // Each row holds every source's columns in turn.
        ParamTransform::HorizontalConcat => Some(
            (0..rows)
                .flat_map(|row| grad[row * width + offset..row * width + offset + extent].to_vec())
                .collect(),
        ),
        // Sources are stacked as whole rows.
        ParamTransform::VerticalConcat => {
            Some(grad[offset * width..(offset + extent) * width].to_vec())
        }
        ParamTransform::Winograd3x3 { .. } => None,
    }
}

/// The op's variant name without its fields.
pub fn op_name(op: &Op) -> String {
    let debug = format!("{op:?}");
    debug
        .split(|c: char| !c.is_alphanumeric())
        .next()
        .unwrap_or_default()
        .to_string()
}
