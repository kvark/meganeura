use crate::autodiff;
use crate::cache;
use crate::compile;
use crate::data::DataLoader;
use crate::graph::Graph;
use crate::optimize;
use crate::optimize::OptimizeReport;
use crate::runtime::Session;
use std::path::Path;

/// Configuration for training.
pub struct TrainConfig {
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
            learning_rate: 0.01,
            log_interval: 100,
            data_input: "x".into(),
            label_input: "labels".into(),
        }
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
        loader.shuffle(epoch as u64);
        loader.reset();

        let mut total_loss = 0.0_f32;
        let mut steps = 0usize;

        while let Some(batch) = loader.next_batch() {
            self.session.set_input(&self.config.data_input, batch.data);
            self.session
                .set_input(&self.config.label_input, batch.labels);

            self.session.step();
            self.session.wait();
            self.session.sgd_step_cpu(self.config.learning_rate);

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

/// Build a complete training session from a forward-pass graph.
///
/// This is the main entry point. It:
/// 1. Runs autodiff to build the backward pass
/// 2. Runs egglog optimization on the combined graph
/// 3. Compiles the optimized graph to an execution plan
/// 4. Creates a GPU session with all resources allocated
pub fn build_session(forward_graph: &Graph) -> Session {
    let (session, report) = build_session_with_report(forward_graph);
    log::info!("{}", report);
    session
}

/// Like `build_session`, but also returns the optimization report.
pub fn build_session_with_report(forward_graph: &Graph) -> (Session, OptimizeReport) {
    log::info!("building training session...");
    log::info!("forward graph:\n{}", forward_graph);

    // Step 1: Autodiff
    log::info!("running autodiff...");
    let full_graph = autodiff::differentiate(forward_graph);
    log::info!(
        "full graph (forward + backward): {} nodes",
        full_graph.nodes().len()
    );

    // Step 2: Optimize with egglog
    log::info!("running egglog optimization...");
    let (optimized, report) = optimize::optimize_with_report(&full_graph);
    log::info!("optimized graph: {} nodes", optimized.nodes().len());

    // Step 3: Compile to execution plan
    log::info!("compiling execution plan...");
    let plan = compile::compile(&optimized);
    log::info!(
        "execution plan: {} buffers, {} dispatches",
        plan.buffers.len(),
        plan.dispatches.len()
    );

    // Step 4: Create GPU session
    log::info!("initializing GPU session...");
    (Session::new(plan), report)
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
        // Should have fusions
        assert!(!report.fusions_applied.is_empty());
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
