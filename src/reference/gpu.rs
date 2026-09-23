//! Run a graph on the GPU and compare it with the reference.

use super::{Comparison, Error, Feeds, Report, Tensor, Tolerance, check, evaluate, magnitudes};
use crate::graph::{DType, Graph, NodeId, Op};
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
    fn session_config(&self, mode: Mode) -> SessionConfig<'static> {
        let mut config = SessionConfig::from_env();
        config.mode = mode;
        config.options = self.compile.clone();
        config.optimize = self.optimize.clone();
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

/// Reference value and error scale of `node`.
fn reference(graph: &Graph, values: &[Tensor], id: NodeId) -> Result<(Tensor, Vec<f64>), Error> {
    let node = graph.node(id);
    let inputs: Vec<&Tensor> = node.inputs.iter().map(|&i| &values[i as usize]).collect();
    let out = values[id as usize].clone();
    let magnitude = magnitudes(graph, node, &inputs, &out)?;
    Ok((out, magnitude))
}

/// Build an inference session for `graph`, run one step, and compare every
/// graph output with the reference.
pub fn check_inference(graph: &Graph, feeds: &Feeds, options: &Options) -> Result<Report, Error> {
    let values = evaluate(graph, feeds)?;
    let (mut session, _) = build(graph, options.session_config(Mode::Inference));
    upload(&mut session, graph, feeds)?;
    session.step();
    session.wait();
    let mut report = Report::default();
    for (index, &id) in graph.outputs().iter().enumerate() {
        let node = graph.node(id);
        let (want, magnitude) = reference(graph, &values, id)?;
        let mut got = vec![0.0f32; want.len()];
        session.read_output_by_index(index, &mut got);
        report.comparisons.push(Comparison {
            what: format!("output {index} (%{id} {})", op_name(&node.op)),
            result: check(&got, &want.data, &magnitude, options.tolerance),
        });
    }
    Ok(report)
}

/// Build a training session for `graph` (whose first output is a scalar
/// loss), run one step without an optimizer update, and compare the loss
/// and every parameter gradient with the reference.
///
/// The reference gradients come from evaluating
/// [`crate::autodiff::differentiate`] in `f64`; [`super::gradients`] checks
/// those against finite differences independently of any kernel.
pub fn check_training(graph: &Graph, feeds: &Feeds, options: &Options) -> Result<Report, Error> {
    let backward = crate::autodiff::differentiate(graph);
    let values = evaluate(&backward, feeds)?;
    let (mut session, _) = build(graph, options.session_config(Mode::Training));
    upload(&mut session, graph, feeds)?;
    session.step();
    session.wait();

    let mut report = Report::default();
    let loss_id = backward.outputs()[0];
    let (want, magnitude) = reference(&backward, &values, loss_id)?;
    report.comparisons.push(Comparison {
        what: "loss".to_string(),
        result: check(
            &[session.read_loss()],
            &want.data,
            &magnitude,
            options.tolerance,
        ),
    });

    let params = graph.nodes().iter().filter_map(|n| match n.op {
        Op::Parameter { ref name } => Some(name.clone()),
        _ => None,
    });
    let grads = &backward.outputs()[backward.num_user_outputs()..];
    for (name, &grad_id) in params.zip(grads) {
        if !session.has_param_grad(&name) {
            // The compiler omits gradients of parameters that do not reach
            // the loss. The reference must agree they are zero.
            let (want, _) = reference(&backward, &values, grad_id)?;
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
        let (want, magnitude) = reference(&backward, &values, grad_id)?;
        let mut got = vec![0.0f32; want.len()];
        session.read_param_grad(&name, &mut got);
        report.comparisons.push(Comparison {
            what: format!("gradient of {name}"),
            result: check(&got, &want.data, &magnitude, options.tolerance),
        });
    }
    Ok(report)
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
