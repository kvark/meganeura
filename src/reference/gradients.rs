//! Check autodiff rules against finite differences, entirely in `f64`.
//!
//! [`crate::autodiff::differentiate`] turns a forward graph into a graph
//! that also computes parameter gradients. Evaluating both with the
//! reference interpreter separates two questions that a GPU gradient check
//! conflates: is the differentiation rule right (checked here, where finite
//! differences in double precision are accurate to about 1e-9), and does
//! each kernel compute its op (checked by [`super::gpu`]).

use super::{Comparison, Error, Feeds, Report, Rng, Tolerance, check_f64, evaluate};
use crate::graph::{Graph, NodeId, Op};

/// Finite-difference settings.
#[derive(Clone, Copy, Debug)]
pub struct Options {
    /// Central-difference step, relative to `max(1, |x|)`.
    pub step: f64,
    /// Compare every element when a parameter has at most this many.
    pub max_elementwise: usize,
    /// Random directional probes over all parameters at once. Each compares
    /// `(L(x + h·u) − L(x − h·u)) / 2h` with `⟨∇L, u⟩`, so it covers every
    /// element, however small its gradient.
    pub probes: usize,
    pub tolerance: Tolerance,
    pub seed: u64,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            step: 1e-6,
            max_elementwise: 4096,
            probes: 4,
            tolerance: Tolerance {
                rtol: 1e-6,
                floor: 1e-3,
            },
            seed: 0x5eed,
        }
    }
}

fn loss(graph: &Graph, feeds: &Feeds) -> Result<f64, Error> {
    let values = evaluate(graph, feeds)?;
    let out = &values[graph.outputs()[0] as usize];
    Ok(out.data.iter().sum())
}

/// Compare the gradients [`crate::autodiff::differentiate`] produces for
/// every parameter of `graph` with central finite differences of its first
/// output, the scalar loss.
pub fn check(graph: &Graph, feeds: &Feeds, options: &Options) -> Result<Report, Error> {
    let backward = crate::autodiff::differentiate(graph);
    let values = evaluate(&backward, feeds)?;
    let params: Vec<(String, NodeId)> = graph
        .nodes()
        .iter()
        .filter_map(|n| match n.op {
            Op::Parameter { ref name } => Some((name.clone(), n.id)),
            _ => None,
        })
        .collect();
    let grad_ids = &backward.outputs()[backward.num_user_outputs()..];
    let mut analytic = Vec::with_capacity(params.len());
    for (&(ref name, param), &grad) in params.iter().zip(grad_ids) {
        let n = graph.node(param).ty.num_elements();
        let g = &values[grad as usize];
        // A parameter that cannot reach the loss gets a scalar zero.
        let g = if g.len() == n {
            g.data.clone()
        } else if g.data.iter().all(|&v| v == 0.0) {
            vec![0.0; n]
        } else {
            return Err(Error::Invalid {
                node: grad,
                reason: format!("gradient of {name} has {} values for {n}", g.len()),
            });
        };
        analytic.push(g);
    }

    let mut report = Report::default();
    let mut perturbed = feeds.clone();
    let h_for = |x: f64| options.step * x.abs().max(1.0);
    // A central difference cannot resolve changes in the loss below its
    // rounding error, about eps·|L| per evaluation, divided by 2h. Treat
    // that as an absolute floor on every comparison.
    let base_loss = loss(graph, feeds)?;
    let noise = |h: f64| 64.0 * f64::EPSILON * (base_loss.abs() + 1.0) / h;
    let floor = |h: f64| noise(h) / options.tolerance.rtol;

    for (p, name) in params.iter().map(|entry| &entry.0).enumerate() {
        let base = feeds
            .get(name)
            .ok_or_else(|| Error::MissingFeed(name.clone()))?
            .to_vec();
        if base.len() > options.max_elementwise {
            continue;
        }
        let mut numeric = vec![0.0; base.len()];
        for i in 0..base.len() {
            let h = h_for(base[i]);
            let mut x = base.clone();
            x[i] = base[i] + h;
            perturbed.values.insert(name.clone(), x.clone());
            let plus = loss(graph, &perturbed)?;
            x[i] = base[i] - h;
            perturbed.values.insert(name.clone(), x);
            let minus = loss(graph, &perturbed)?;
            numeric[i] = (plus - minus) / (2.0 * h);
        }
        perturbed.values.insert(name.clone(), base.clone());
        let magnitude: Vec<f64> = analytic[p]
            .iter()
            .zip(&numeric)
            .zip(&base)
            .map(|((a, n), &x)| a.abs().max(n.abs()) + floor(h_for(x)))
            .collect();
        report.comparisons.push(Comparison {
            what: format!("d loss / d {name}"),
            result: check_f64(&analytic[p], &numeric, &magnitude, options.tolerance),
        });
    }

    let mut rng = Rng::new(options.seed);
    for probe in 0..options.probes {
        let mut plus = feeds.clone();
        let mut minus = feeds.clone();
        let mut predicted = 0.0;
        let mut scale = 0.0;
        let h = options.step;
        for (p, name) in params.iter().map(|entry| &entry.0).enumerate() {
            let base = feeds
                .get(name)
                .ok_or_else(|| Error::MissingFeed(name.clone()))?;
            let direction: Vec<f64> = (0..base.len()).map(|_| rng.unit() * 2.0 - 1.0).collect();
            for (g, u) in analytic[p].iter().zip(&direction) {
                predicted += g * u;
                scale += (g * u).abs();
            }
            plus.values.insert(
                name.clone(),
                base.iter()
                    .zip(&direction)
                    .map(|(x, u)| x + h * u)
                    .collect(),
            );
            minus.values.insert(
                name.clone(),
                base.iter()
                    .zip(&direction)
                    .map(|(x, u)| x - h * u)
                    .collect(),
            );
        }
        let numeric = (loss(graph, &plus)? - loss(graph, &minus)?) / (2.0 * h);
        report.comparisons.push(Comparison {
            what: format!("directional probe {probe}"),
            result: check_f64(
                &[predicted],
                &[numeric],
                &[scale + floor(h)],
                options.tolerance,
            ),
        });
    }
    Ok(report)
}

/// Reduce `x` to a scalar loss that depends on every element differently:
/// `coef · Σ w ⊙ x` with fixed pseudo-random weights `w`.
///
/// Checking an op through this loss places it mid-graph: its backward must
/// consume a non-uniform upstream gradient scaled by `coef`, which catches
/// backward rules that assume `dL/dy = 1`.
pub fn weighted_loss(graph: &mut Graph, x: NodeId, seed: u64, coef: f32) -> NodeId {
    let ty = graph.node(x).ty.clone();
    let mut rng = Rng::new(seed);
    let weights: Vec<f32> = (0..ty.num_elements())
        .map(|_| rng.uniform(-1.0, 1.0))
        .collect();
    let w = graph.constant(weights, &ty.shape);
    let weighted = graph.mul(x, w);
    let total = graph.sum_all(weighted);
    graph.scale(total, coef)
}
