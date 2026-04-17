use crate::{
    autodiff, cache, compile,
    data::DataLoader,
    graph::Graph,
    optimize::{self, OptimizeReport},
    runtime::Session,
};
use std::path::Path;
use std::sync::Arc;

/// Optimizer selection.
#[derive(Clone, Debug)]
pub enum Optimizer {
    /// Stochastic gradient descent.
    Sgd { learning_rate: f32 },
    /// Adam optimizer.
    Adam {
        learning_rate: f32,
        beta1: f32,
        beta2: f32,
        epsilon: f32,
    },
}

impl Optimizer {
    /// SGD with the given learning rate.
    pub fn sgd(lr: f32) -> Self {
        Self::Sgd { learning_rate: lr }
    }

    /// Adam with standard defaults (beta1=0.9, beta2=0.999, eps=1e-8).
    pub fn adam(lr: f32) -> Self {
        Self::Adam {
            learning_rate: lr,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
        }
    }
}

impl Default for Optimizer {
    fn default() -> Self {
        Self::Sgd {
            learning_rate: 0.01,
        }
    }
}

/// Configuration for training.
pub struct TrainConfig {
    /// Optimizer to use (SGD or Adam).
    pub optimizer: Optimizer,
    /// Backward-compatible alias: sets SGD learning rate.
    /// Ignored if `optimizer` is explicitly set to Adam.
    pub learning_rate: f32,
    /// Print loss every `log_interval` steps. 0 disables step logging.
    pub log_interval: usize,
    /// Name of the graph input that receives sample data (e.g. `"x"`).
    pub data_input: String,
    /// Name of the graph input that receives labels (e.g. `"labels"`).
    pub label_input: String,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            optimizer: Optimizer::default(),
            learning_rate: 0.01,
            log_interval: 100,
            data_input: "x".into(),
            label_input: "labels".into(),
        }
    }
}

/// Per-step training metrics passed to [`MetricCallback::on_step`].
#[derive(Clone, Debug)]
pub struct StepMetrics {
    pub epoch: usize,
    pub step: usize,
    pub loss: f32,
}

/// Callback for training events.
pub trait MetricCallback {
    fn on_step(&mut self, _metrics: &StepMetrics) {}
    fn on_epoch(&mut self, _stats: &EpochStats) {}
}

/// Collects per-step loss values for later analysis or plotting.
#[derive(Default)]
pub struct LossHistory {
    pub losses: Vec<f32>,
}

impl MetricCallback for LossHistory {
    fn on_step(&mut self, m: &StepMetrics) {
        self.losses.push(m.loss);
    }
}

/// Per-epoch training statistics.
#[derive(Clone, Debug)]
pub struct EpochStats {
    pub epoch: usize,
    pub avg_loss: f32,
    pub steps: usize,
}

/// Accumulated training history returned by [`Trainer::train`].
#[derive(Clone, Debug, Default)]
pub struct TrainHistory {
    pub epochs: Vec<EpochStats>,
}

impl TrainHistory {
    /// Final average loss (from the last epoch), or `None` if no epochs ran.
    pub fn final_loss(&self) -> Option<f32> {
        self.epochs.last().map(|e| e.avg_loss)
    }
}

/// Drives the training loop over a [`Session`] and [`DataLoader`].
///
/// Encapsulates the epoch → batch → step → SGD update cycle, with
/// configurable logging and loss tracking.
pub struct Trainer {
    session: Session,
    config: TrainConfig,
}

impl Trainer {
    pub fn new(session: Session, config: TrainConfig) -> Self {
        Self { session, config }
    }

    /// Run `epochs` full passes over the data, returning training history.
    pub fn train(&mut self, loader: &mut DataLoader, epochs: usize) -> TrainHistory {
        let mut history = TrainHistory::default();
        for epoch in 0..epochs {
            let stats = self.train_epoch(loader, epoch);
            log::info!(
                "epoch {}: avg_loss = {:.4} ({} steps)",
                stats.epoch,
                stats.avg_loss,
                stats.steps,
            );
            history.epochs.push(stats);
        }
        history
    }

    /// Run a single epoch, returning its statistics.
    pub fn train_epoch(&mut self, loader: &mut DataLoader, epoch: usize) -> EpochStats {
        let _span = tracing::info_span!("train_epoch", epoch).entered();
        loader.shuffle(epoch as u64);
        loader.reset();

        let mut total_loss = 0.0_f32;
        let mut steps = 0usize;

        while let Some(batch) = loader.next_batch() {
            {
                let _span = tracing::info_span!("set_input").entered();
                self.session.set_input(&self.config.data_input, batch.data);
                self.session
                    .set_input(&self.config.label_input, batch.labels);
            }

            // Set optimizer for fused step
            match self.config.optimizer {
                Optimizer::Sgd { learning_rate } => {
                    self.session.set_learning_rate(learning_rate);
                }
                Optimizer::Adam {
                    learning_rate,
                    beta1,
                    beta2,
                    epsilon,
                } => {
                    self.session.set_adam(learning_rate, beta1, beta2, epsilon);
                }
            }
            self.session.step();
            self.session.wait();

            let loss = self.session.read_loss();
            total_loss += loss;

            if self.config.log_interval > 0 && steps.is_multiple_of(self.config.log_interval) {
                log::info!("  epoch {} step {}: loss = {:.4}", epoch, steps, loss);
            }
            steps += 1;
        }

        let avg_loss = if steps > 0 {
            total_loss / steps as f32
        } else {
            0.0
        };
        EpochStats {
            epoch,
            avg_loss,
            steps,
        }
    }

    /// Borrow the underlying session.
    pub fn session(&self) -> &Session {
        &self.session
    }

    /// Mutably borrow the underlying session (e.g. to set parameters).
    pub fn session_mut(&mut self) -> &mut Session {
        &mut self.session
    }

    /// Consume the trainer and return the session.
    pub fn into_session(self) -> Session {
        self.session
    }
}

/// Whether the compiled session runs forward-only or forward + backward
/// + optimizer.
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    /// Forward + backward + parameter updates. Default.
    #[default]
    Training,
    /// Forward only — skips autodiff.
    Inference,
}

/// Parameters for [`build`].
///
/// Replaces the `build_session` / `build_inference_session` /
/// `build_session_with_report` / `build_session_on` / ... zoo with a
/// single entry point. The defaults produce a training session on a
/// fresh Blade GPU context with the default compile options and full
/// optimization enabled.
#[derive(Default)]
pub struct SessionConfig<'a> {
    pub mode: Mode,
    /// Reuse an externally-owned Blade GPU context. `None` initialises
    /// a fresh one inside [`Session`]. Sharing a context with a host
    /// renderer is the motivating use case — see
    /// [`Session::with_context`] for details.
    pub gpu: Option<Arc<blade_graphics::Context>>,
    pub options: compile::CompileOptions,
    /// When set, load a previously-compiled plan from this path if it
    /// matches the graph hash, or save it there after compiling.
    /// Training mode only — the cache is keyed by the forward graph.
    pub cache: Option<&'a Path>,
    /// Skip the post-autodiff full-graph optimization pass. Useful for
    /// debugging gradient flow through aggressively-fused ops. Training
    /// mode only.
    pub skip_full_optimize: bool,
}

/// Build a [`Session`] from a forward-pass graph.
///
/// Runs forward optimization, optional autodiff (training mode),
/// optional full-graph optimization, plan compilation, and GPU session
/// initialization — all the stages the old `build_session_*` family
/// strung together, now driven by [`SessionConfig`].
///
/// Build stages (`optimize_forward`, `autodiff`, `optimize_full`,
/// `compile`, `gpu_init`) are captured as tracing spans and appear in
/// Perfetto traces when profiling is active.
pub fn build(
    forward_graph: &Graph,
    cfg: SessionConfig<'_>,
) -> (Session, OptimizeReport) {
    let _span = tracing::info_span!("build_session").entered();
    log::info!("building {:?} session", cfg.mode);
    log::info!("forward graph:\n{}", forward_graph);

    if cfg.mode == Mode::Training
        && let Some(path) = cfg.cache
    {
        match cache::load_plan(forward_graph, path) {
            Ok(Some(plan)) => {
                log::info!("loaded cached execution plan from {}", path.display());
                let session = make_session(plan, cfg.gpu);
                return (session, OptimizeReport::empty());
            }
            Ok(None) => log::info!("no valid cache found, recompiling"),
            Err(e) => log::warn!("failed to load cache: {}, recompiling", e),
        }
    }

    let (optimized_forward, forward_report) = {
        let _span = tracing::info_span!("optimize_forward").entered();
        optimize::optimize_with_report(forward_graph)
    };
    log::info!("optimized forward: {} nodes", optimized_forward.nodes().len());

    let (final_graph, report) = match cfg.mode {
        Mode::Inference => {
            let mut g = optimized_forward;
            let mut fusions = Vec::new();
            optimize::apply_group_norm_silu_fusions(&mut g, &mut fusions);
            optimize::apply_winograd_conv_fusions(&mut g, &mut fusions);
            for (name, count) in fusions.iter().fold(
                std::collections::BTreeMap::<&str, usize>::new(),
                |mut acc, entry| {
                    *acc.entry(entry.0.as_str()).or_default() += 1;
                    acc
                },
            ) {
                log::info!("inference fusion: {}x {}", count, name);
            }
            (g, forward_report)
        }
        Mode::Training => {
            let sorted = optimized_forward.toposort();
            let full = {
                let _span = tracing::info_span!("autodiff").entered();
                autodiff::differentiate(&sorted)
            };
            log::info!("full graph (forward + backward): {} nodes", full.nodes().len());
            if cfg.skip_full_optimize {
                (full, forward_report)
            } else {
                let _span = tracing::info_span!("optimize_full").entered();
                optimize::optimize_with_report(&full)
            }
        }
    };

    let plan = {
        let _span = tracing::info_span!("compile").entered();
        compile::compile_with(&final_graph, &cfg.options)
    };
    log::info!(
        "execution plan: {} buffers, {} dispatches",
        plan.buffers.len(),
        plan.dispatches.len()
    );

    if cfg.mode == Mode::Training
        && let Some(path) = cfg.cache
    {
        if let Err(e) = cache::save_plan(&plan, forward_graph, path) {
            log::warn!("failed to save cache: {}", e);
        } else {
            log::info!("saved execution plan cache to {}", path.display());
        }
    }

    let session = {
        let _span = tracing::info_span!("gpu_init").entered();
        make_session(plan, cfg.gpu)
    };
    (session, report)
}

fn make_session(
    plan: compile::ExecutionPlan,
    gpu: Option<Arc<blade_graphics::Context>>,
) -> Session {
    match gpu {
        Some(ctx) => Session::with_context(plan, ctx),
        None => Session::new(plan),
    }
}

/// Sugar for `build(g, SessionConfig::default()).0` — the common
/// training case.
pub fn build_session(forward_graph: &Graph) -> Session {
    build(forward_graph, SessionConfig::default()).0
}

/// Sugar for `build(g, SessionConfig { mode: Inference, .. })`.
pub fn build_inference_session(forward_graph: &Graph) -> Session {
    build(
        forward_graph,
        SessionConfig {
            mode: Mode::Inference,
            ..SessionConfig::default()
        },
    )
    .0
}

/// Sugar for `build(g, SessionConfig { skip_full_optimize: true, .. })`.
/// Useful for debugging gradient flow through ops the full-graph
/// optimizer fuses.
pub fn build_session_unoptimized(forward_graph: &Graph) -> Session {
    build(
        forward_graph,
        SessionConfig {
            skip_full_optimize: true,
            ..SessionConfig::default()
        },
    )
    .0
}

/// Run the compile pipeline (autodiff → optimize → compile) without
/// creating a GPU session. Useful for testing compilation in
/// environments without GPU access.
pub fn compile_training_graph(
    forward_graph: &Graph,
) -> (crate::compile::ExecutionPlan, OptimizeReport) {
    let full_graph = autodiff::differentiate(forward_graph);
    let (optimized, report) = optimize::optimize_with_report(&full_graph);
    let plan = compile::compile(&optimized);
    (plan, report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::Graph;

    #[test]
    fn test_compile_training_graph_simple() {
        let mut g = Graph::new();
        let x = g.input("x", &[4, 3]);
        let w = g.parameter("w", &[3, 2]);
        let y = g.matmul(x, w);
        let loss = g.mean_all(y);
        g.set_outputs(vec![loss]);

        let (plan, report) = compile_training_graph(&g);
        assert!(!plan.dispatches.is_empty());
        assert!(plan.loss_buffer.is_some());
        assert_eq!(plan.param_grad_pairs.len(), 1);
        assert_eq!(plan.param_buffers.len(), 1);
        assert_eq!(plan.input_buffers.len(), 1);
        assert!(report.nodes_before > 0);
    }

    #[test]
    fn test_compile_training_graph_mlp() {
        let mut g = Graph::new();
        let x = g.input("x", &[4, 784]);
        let w1 = g.parameter("w1", &[784, 128]);
        let b1 = g.parameter("b1", &[128]);
        let mm1 = g.matmul(x, w1);
        let h1 = g.bias_add(mm1, b1);
        let a1 = g.relu(h1);
        let w2 = g.parameter("w2", &[128, 10]);
        let mm2 = g.matmul(a1, w2);
        let labels = g.input("labels", &[4, 10]);
        let loss = g.cross_entropy_loss(mm2, labels);
        g.set_outputs(vec![loss]);

        let (plan, report) = compile_training_graph(&g);
        // 3 parameters → 3 grad pairs
        assert_eq!(plan.param_grad_pairs.len(), 3);
        assert_eq!(plan.param_buffers.len(), 3);
        assert_eq!(plan.input_buffers.len(), 2); // x and labels
        assert!(plan.loss_buffer.is_some());
        // MLP has no Add(MatMul, x) patterns (uses BiasAdd, not Add),
        // so no matmul+add fusions fire.
        assert!(
            report.fusions_applied.is_empty(),
            "unexpected fusions: {:?}",
            report.fusions_applied
        );
    }

    #[test]
    fn test_train_config_default() {
        let config = TrainConfig::default();
        assert_eq!(config.learning_rate, 0.01);
        assert_eq!(config.log_interval, 100);
        assert_eq!(config.data_input, "x");
        assert_eq!(config.label_input, "labels");
    }

    #[test]
    fn test_train_history_final_loss() {
        let mut h = TrainHistory::default();
        assert_eq!(h.final_loss(), None);
        h.epochs.push(EpochStats {
            epoch: 0,
            avg_loss: 2.5,
            steps: 10,
        });
        h.epochs.push(EpochStats {
            epoch: 1,
            avg_loss: 1.2,
            steps: 10,
        });
        assert_eq!(h.final_loss(), Some(1.2));
    }

    #[test]
    fn test_epoch_stats_fields() {
        let stats = EpochStats {
            epoch: 3,
            avg_loss: 0.42,
            steps: 100,
        };
        assert_eq!(stats.epoch, 3);
        assert!((stats.avg_loss - 0.42).abs() < 1e-6);
        assert_eq!(stats.steps, 100);
    }

    #[test]
    fn test_compile_and_cache_roundtrip() {
        let mut g = Graph::new();
        let x = g.input("x", &[4, 3]);
        let w = g.parameter("w", &[3, 2]);
        let y = g.matmul(x, w);
        let loss = g.mean_all(y);
        g.set_outputs(vec![loss]);

        let (plan, _) = compile_training_graph(&g);

        let dir = std::env::temp_dir().join("meganeura_test_train_cache");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("train_plan.ron");

        cache::save_plan(&plan, &g, &path).unwrap();
        let loaded = cache::load_plan(&g, &path).unwrap().unwrap();
        assert_eq!(loaded.dispatches.len(), plan.dispatches.len());
        assert_eq!(loaded.buffers.len(), plan.buffers.len());
        assert_eq!(loaded.param_grad_pairs.len(), plan.param_grad_pairs.len());

        let _ = std::fs::remove_file(&path);
    }
}
