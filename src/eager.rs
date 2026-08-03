//! Eager evaluation: inspect any node of a graph *while you build it*.
//!
//! This is meganeura's bridge to the PyTorch-eager development loop. There
//! is no second tracing system and no separate op vocabulary — you build
//! the same [`Graph`] with the same builder methods, and [`Eager::eval`]
//! executes it up to the node you ask about, on the same generated kernels
//! the compiled path uses:
//!
//! ```no_run
//! use meganeura::{Graph, eager::Eager};
//!
//! let mut g = Graph::new();
//! let x = g.input("x", &[2, 4]);
//! let w = g.parameter("w", &[4, 3]);
//! let h = g.matmul(x, w);
//!
//! let mut e = Eager::new();
//! e.set_input("x", vec![1.0; 8]);
//! e.set_parameter("w", vec![0.5; 12]);
//! println!("{}", e.eval(&g, h)); // values, right now
//!
//! let y = g.relu(h);             // keep building the same graph
//! println!("{}", e.eval(&g, y));
//! // ... and when it works: g.set_outputs(vec![y]); build_session(&g)
//! ```
//!
//! Execution model: `eval` compiles the graph **without rewrites or
//! dispatch fusion** (every node materializes, semantics are the direct
//! per-op ones) into a debug session, runs a step, and reads the node's
//! buffer. The session is cached and reused until the graph grows or an
//! input changes; each rebuild recompiles only because the plan changed —
//! pipelines for repeated op shapes are cheap, but this is a development
//! mode: expect whole-prefix re-execution per growth step, and use
//! `build_session` for anything performance-sensitive.

use crate::compile::{self, CompileOptions};
use crate::graph::{Graph, NodeId};
use crate::runtime::{Session, SessionOptions, init_gpu_context};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

/// A materialized tensor value returned by [`Eager::eval`].
#[derive(Clone, Debug, PartialEq)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
}

impl fmt::Display for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Tensor{:?} ", self.shape)?;
        const MAX: usize = 32;
        if self.data.len() <= MAX {
            write!(f, "{:?}", self.data)
        } else {
            let head: Vec<f32> = self.data[..MAX].to_vec();
            write!(f, "{head:?}… ({} elements)", self.data.len())
        }
    }
}

/// Eager evaluator over an externally-owned [`Graph`].
pub struct Eager {
    gpu: Arc<blade_graphics::Context>,
    inputs: HashMap<String, Vec<f32>>,
    inputs_u32: HashMap<String, Vec<u32>>,
    params: HashMap<String, Vec<f32>>,
    /// Cached session + the graph size it was compiled at. Invalidated
    /// when the graph grows or bound data changes.
    session: Option<(Session, usize)>,
    stepped: bool,
}

impl Eager {
    /// Create an eager evaluator on a fresh GPU context.
    pub fn new() -> Self {
        Self::with_context(Arc::new(
            init_gpu_context().expect("failed to initialize blade GPU context"),
        ))
    }

    /// Create an eager evaluator sharing an existing GPU context.
    pub fn with_context(gpu: Arc<blade_graphics::Context>) -> Self {
        Self {
            gpu,
            inputs: HashMap::new(),
            inputs_u32: HashMap::new(),
            params: HashMap::new(),
            session: None,
            stepped: false,
        }
    }

    /// Bind data for an `Op::Input` by name. May be called before or after
    /// the input node exists.
    pub fn set_input(&mut self, name: impl Into<String>, data: Vec<f32>) {
        self.inputs.insert(name.into(), data);
        self.stepped = false;
    }

    /// Bind u32 data for an `Op::Input` by name (token ids etc.).
    pub fn set_input_u32(&mut self, name: impl Into<String>, data: Vec<u32>) {
        self.inputs_u32.insert(name.into(), data);
        self.stepped = false;
    }

    /// Bind data for an `Op::Parameter` by name.
    pub fn set_parameter(&mut self, name: impl Into<String>, data: Vec<f32>) {
        self.params.insert(name.into(), data);
        self.stepped = false;
    }

    /// Evaluate `node` of `g` and return its value. Reuses the cached
    /// session when the graph hasn't grown since the last call.
    pub fn eval(&mut self, g: &Graph, node: NodeId) -> Tensor {
        let n_nodes = g.nodes().len();
        assert!(
            (node as usize) < n_nodes,
            "eval: node {node} not in graph ({n_nodes} nodes)"
        );

        let rebuild = match self.session {
            Some((_, built_at)) => built_at != n_nodes,
            None => true,
        };
        if rebuild {
            self.session = Some((self.build_session(g), n_nodes));
            self.stepped = false;
        }
        let session = &mut self.session.as_mut().expect("just built").0;

        if !self.stepped {
            for (name, data) in &self.params {
                if session.has_parameter(name) {
                    session.set_parameter(name, data);
                }
            }
            for (name, data) in &self.inputs {
                if session.has_input(name) {
                    session.set_input(name, data);
                }
            }
            for (name, data) in &self.inputs_u32 {
                if session.has_input(name) {
                    session.set_input_u32(name, data);
                }
            }
            session.step();
            session.wait();
            self.stepped = true;
        }

        let shape = g.node(node).ty.shape.clone();
        let n: usize = shape.iter().product();
        let mut data = session
            .read_node(node)
            .expect("debug session materializes every node");
        data.truncate(n.max(1));
        Tensor { data, shape }
    }

    fn build_session(&self, g: &Graph) -> Session {
        // No rewrites, no dispatch fusion, and crucially NO toposort
        // renumbering: the caller's NodeIds must stay valid, and `compile`
        // walks nodes in dependency order itself without renaming them.
        // Every node built so far gets a dispatch — eager means "run my
        // code as written", including branches nothing consumes yet.
        let mut g = g.deep_clone();
        if g.outputs().is_empty() {
            // compile() wants at least one output; the last node works —
            // debug sessions materialize everything regardless.
            let last = (g.nodes().len() - 1) as NodeId;
            g.set_outputs(vec![last]);
        }
        let plan = compile::compile_with(
            &g,
            &CompileOptions {
                fuse_dispatches: false,
                ..CompileOptions::default()
            },
        );
        Session::with_context_opts(plan, self.gpu.clone(), SessionOptions { debug: true })
    }
}

impl Default for Eager {
    fn default() -> Self {
        Self::new()
    }
}
