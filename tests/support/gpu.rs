//! Shared GPU context and environment overrides for integration tests.

pub use meganeura::reference::gpu::shared_context as gpu;

pub fn config() -> meganeura::SessionConfig<'static> {
    meganeura::SessionConfig::from_env_with_gpu(Some(gpu()))
}

pub fn inference_config() -> meganeura::SessionConfig<'static> {
    meganeura::SessionConfig::inference_from_env_on(gpu())
}
