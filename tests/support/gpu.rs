//! Shared GPU context and environment overrides for integration tests.

pub use meganeura::reference::gpu::shared_context as gpu;

pub fn config() -> meganeura::SessionConfig<'static> {
    meganeura::SessionConfig::from_env_with_gpu(Some(gpu()))
}

pub fn inference_config() -> meganeura::SessionConfig<'static> {
    meganeura::SessionConfig::inference_from_env_on(gpu())
}

#[allow(dead_code)] // Shared by test binaries with different child modules.
pub fn from_plan(plan: meganeura::compile::ExecutionPlan) -> meganeura::Session {
    meganeura::Session::with_context_opts(plan, gpu(), meganeura::SessionOptions::from_env())
}

/// Keep protocol-run headroom checks without rejecting ordinary shared GPUs
/// or drivers that cannot report `VK_EXT_memory_budget`.
#[allow(dead_code)] // Each integration target compiles this shared module independently.
pub fn assert_protocol_headroom(gpu: &blade_graphics::Context) {
    if std::env::var_os("MEGANEURA_EXPECT_DEVICE_NAME").is_some() {
        let memory = gpu.memory_stats();
        assert!(memory.budget.saturating_sub(memory.usage) >= 2 << 30);
    }
}
