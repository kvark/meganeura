use crate::autodiff;
use crate::compile;
use crate::graph::Graph;
use crate::optimize;
use crate::runtime::Session;

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
    log::info!("building training session...");
    log::info!("forward graph:\n{}", forward_graph);

    // Step 1: Autodiff — build backward pass
    log::info!("running autodiff...");
    let full_graph = autodiff::differentiate(forward_graph);
    log::info!(
        "full graph (forward + backward): {} nodes",
        full_graph.nodes().len()
    );

    // Step 2: Optimize with egglog
    log::info!("running egglog optimization...");
    let optimized = optimize::optimize(&full_graph);
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
    Session::new(plan)
}
