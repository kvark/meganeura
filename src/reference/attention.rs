//! Placeholder: filled in per op family.

use super::{Error, Tensor};
use crate::graph::{Graph, Node};

pub(super) fn eval(_graph: &Graph, node: &Node, _ins: &[&Tensor]) -> Result<Vec<f64>, Error> {
    Err(Error::Unsupported {
        node: node.id,
        reason: format!("{:?} has no reference yet", node.op),
    })
}

pub(super) fn magnitude(
    _graph: &Graph,
    _node: &Node,
    _ins: &[&Tensor],
    _out: &Tensor,
) -> Option<Vec<f64>> {
    None
}
