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
