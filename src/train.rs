use crate::autodiff;
use crate::cache;
use crate::compile;
use crate::graph::Graph;
use crate::optimize;
use crate::optimize::OptimizeReport;
use crate::runtime::Session;
use std::path::Path;

/// Configuration for training.
pub struct TrainConfig {
    pub learning_rate: f32,
    pub log_interval: usize,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            log_interval: 100,
        }
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
pub fn compile_training_graph(forward_graph: &Graph) -> (crate::compile::ExecutionPlan, OptimizeReport) {
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
