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

/// Build an inference-only session from a forward-pass graph.
///
/// Skips autodiff (no backward pass). Runs egglog optimization and
/// compiles the graph to a GPU session ready for forward evaluation.
pub fn build_inference_session(forward_graph: &Graph) -> Session {
    build_inference_session_with(forward_graph, &compile::CompileOptions::default())
}

/// Like [`build_inference_session`], but accepts [`compile::CompileOptions`]
/// to toggle experimental paths (e.g. schedule-template codegen).
pub fn build_inference_session_with(
    forward_graph: &Graph,
    options: &compile::CompileOptions,
) -> Session {
    log::info!("building inference session...");
    log::info!("forward graph:\n{}", forward_graph);

    // Optimize with egglog (fusions still help for inference)
    log::info!("running egglog optimization...");
    let mut optimized = optimize::optimize(forward_graph);
    log::info!("optimized graph: {} nodes", optimized.nodes().len());

    // Inference-only fusions (no backward pass needed)
    let mut inference_fusions = Vec::new();
    optimize::apply_group_norm_silu_fusions(&mut optimized, &mut inference_fusions);
    optimize::apply_winograd_conv_fusions(&mut optimized, &mut inference_fusions);
    if !inference_fusions.is_empty() {
        for (name, count) in inference_fusions.iter().fold(
            std::collections::BTreeMap::<&str, usize>::new(),
            |mut acc, entry| {
                *acc.entry(entry.0.as_str()).or_default() += 1;
                acc
            },
        ) {
            log::info!("inference fusion: {}x {}", count, name);
        }
    }

    // Compile to execution plan
    log::info!("compiling execution plan...");
    let plan = compile::compile_with(&optimized, options);
    log::info!(
        "execution plan: {} buffers, {} dispatches",
        plan.buffers.len(),
        plan.dispatches.len()
    );

    // Create GPU session
    log::info!("initializing GPU session...");
    Session::new(plan)
}

/// Like [`build_inference_session`], but reuses an externally-owned
/// Blade GPU context. See [`Session::with_context`] for the motivating
/// use case (a host renderer sharing its device with meganeura).
pub fn build_inference_session_on(
    forward_graph: &Graph,
    gpu: Arc<blade_graphics::Context>,
) -> Session {
    build_inference_session_on_with(forward_graph, gpu, &compile::CompileOptions::default())
}

/// Like [`build_inference_session_with`], but reuses an existing GPU context.
pub fn build_inference_session_on_with(
    forward_graph: &Graph,
    gpu: Arc<blade_graphics::Context>,
    options: &compile::CompileOptions,
) -> Session {
    let mut optimized = optimize::optimize(forward_graph);
    let mut inference_fusions = Vec::new();
    optimize::apply_group_norm_silu_fusions(&mut optimized, &mut inference_fusions);
    optimize::apply_winograd_conv_fusions(&mut optimized, &mut inference_fusions);
    let plan = compile::compile_with(&optimized, options);
    Session::with_context(plan, gpu)
}

/// Build a complete training session from a forward-pass graph.
///
/// This is the main entry point. It:
/// 1. Optimizes the forward graph (SwiGLU fusion, MatMul+Add, etc.)
/// 2. Runs autodiff on the optimized forward graph
/// 3. Optimizes the full graph (backward MatMul+Add fusions)
/// 4. Compiles the optimized graph to an execution plan
/// 5. Creates a GPU session with all resources allocated
pub fn build_session(forward_graph: &Graph) -> Session {
    build_session_with(forward_graph, &compile::CompileOptions::default())
}

/// Like [`build_session`], but accepts [`compile::CompileOptions`] to toggle
/// experimental paths (e.g. schedule-template codegen).
pub fn build_session_with(forward_graph: &Graph, options: &compile::CompileOptions) -> Session {
    let (session, report) = build_session_with_report_and_options(forward_graph, options);
    log::info!("{}", report);
    session
}

/// Like `build_session`, but also returns the optimization report.
///
/// Build-phase stages (autodiff, egglog, compile, gpu_init) are captured
/// as tracing spans and will appear in Perfetto traces when profiling is
/// active.
pub fn build_session_with_report(forward_graph: &Graph) -> (Session, OptimizeReport) {
    build_session_with_report_and_options(forward_graph, &compile::CompileOptions::default())
}

/// [`build_session_with_report`] + [`compile::CompileOptions`].
pub fn build_session_with_report_and_options(
    forward_graph: &Graph,
    options: &compile::CompileOptions,
) -> (Session, OptimizeReport) {
    let _span = tracing::info_span!("build_session").entered();

    log::info!("building training session...");
    log::info!("forward graph:\n{}", forward_graph);

    // Step 1: Optimize forward graph (fuse SwiGLU, MatMul+Add, etc.)
    // This runs BEFORE autodiff so the backward pass differentiates
    // the optimized ops (e.g. SwiGLUConcat instead of separate SwiGLU).
    log::info!("optimizing forward graph...");
    let optimized_forward = {
        let _span = tracing::info_span!("optimize_forward").entered();
        optimize::optimize(forward_graph)
    };
    log::info!(
        "optimized forward: {} nodes",
        optimized_forward.nodes().len()
    );

    // Step 2: Toposort the optimized graph (optimizer may append nodes
    // out of order) then run autodiff. Autodiff iterates in reverse node
    // order, so topological ordering ensures gradients flow correctly.
    let sorted_forward = optimized_forward.toposort();
    log::info!(
        "sorted forward: {} nodes (from {} optimized, {} original)",
        sorted_forward.nodes().len(),
        optimized_forward.nodes().len(),
        forward_graph.nodes().len(),
    );
    let full_graph = {
        let _span = tracing::info_span!("autodiff").entered();
        autodiff::differentiate(&sorted_forward)
    };
    log::info!(
        "full graph (forward + backward): {} nodes",
        full_graph.nodes().len()
    );

    // Step 3: Optimize full graph (fuse backward MatMul+Add, etc.)
    log::info!("optimizing full graph...");
    let (optimized, report) = {
        let _span = tracing::info_span!("optimize_full").entered();
        optimize::optimize_with_report(&full_graph)
    };
    log::info!("optimized graph: {} nodes", optimized.nodes().len());

    // Step 3: Compile to execution plan
    log::info!("compiling execution plan...");
    let plan = {
        let _span = tracing::info_span!("compile").entered();
        compile::compile_with(&optimized, options)
    };
    log::info!(
        "execution plan: {} buffers, {} dispatches",
        plan.buffers.len(),
        plan.dispatches.len()
    );

    // Step 4: Create GPU session
    log::info!("initializing GPU session...");
    let session = {
        let _span = tracing::info_span!("gpu_init").entered();
        Session::new(plan)
    };

    (session, report)
}

/// Like [`build_session`], but reuses an externally-owned Blade GPU
/// context. See [`Session::with_context`] for motivation.
pub fn build_session_on(forward_graph: &Graph, gpu: Arc<blade_graphics::Context>) -> Session {
    build_session_on_with(forward_graph, gpu, &compile::CompileOptions::default())
}

/// Like [`build_session_with`], but reuses an existing GPU context.
pub fn build_session_on_with(
    forward_graph: &Graph,
    gpu: Arc<blade_graphics::Context>,
    options: &compile::CompileOptions,
) -> Session {
    let optimized_forward = optimize::optimize(forward_graph);
    let sorted_forward = optimized_forward.toposort();
    let full_graph = autodiff::differentiate(&sorted_forward);
    let (optimized, _report) = optimize::optimize_with_report(&full_graph);
    let plan = compile::compile_with(&optimized, options);
    Session::with_context(plan, gpu)
}

/// Build a training session without full-graph optimization (step 3).
///
/// Useful for debugging gradient flow issues, since the full-graph optimizer
/// can fuse backward MatMul+Add ops in ways that break gradient propagation.
pub fn build_session_unoptimized(forward_graph: &Graph) -> Session {
    let _span = tracing::info_span!("build_session_unoptimized").entered();

    let optimized_forward = optimize::optimize(forward_graph);
    let sorted_forward = optimized_forward.toposort();
    let full_graph = autodiff::differentiate(&sorted_forward);

    // Skip step 3 (full-graph optimization) — compile directly
    let plan = compile::compile(&full_graph);
    Session::new(plan)
}

/// Build a session, using a cache file if available.
///
/// If the cache exists and the graph hash matches, skip autodiff/optimize/compile.
/// Otherwise, run the full pipeline and save the result.
/// Run the full pipeline (autodiff → optimize → compile) without creating a GPU session.
///
/// Useful for testing the compilation pipeline in environments without GPU access.
pub fn compile_training_graph(
    forward_graph: &Graph,
) -> (crate::compile::ExecutionPlan, OptimizeReport) {
    let full_graph = autodiff::differentiate(forward_graph);
    let (optimized, report) = optimize::optimize_with_report(&full_graph);
    let plan = compile::compile(&optimized);
    (plan, report)
}

pub fn build_session_cached(forward_graph: &Graph, cache_path: &Path) -> Session {
    match cache::load_plan(forward_graph, cache_path) {
        Ok(Some(plan)) => {
            log::info!("loaded cached execution plan from {}", cache_path.display());
            return Session::new(plan);
        }
        Ok(None) => {
            log::info!("no valid cache found, running full pipeline");
        }
        Err(e) => {
            log::warn!("failed to load cache: {}, recompiling", e);
        }
    }

    // Full pipeline — compile and cache the plan before creating the session
    log::info!("building training session...");
    log::info!("forward graph:\n{}", forward_graph);

    let full_graph = autodiff::differentiate(forward_graph);
    let (optimized, report) = optimize::optimize_with_report(&full_graph);
    let plan = compile::compile(&optimized);

    // Save cache before consuming the plan
    if let Err(e) = cache::save_plan(&plan, forward_graph, cache_path) {
        log::warn!("failed to save cache: {}", e);
    } else {
        log::info!("saved execution plan cache to {}", cache_path.display());
    }

    log::info!("{}", report);
    Session::new(plan)
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
