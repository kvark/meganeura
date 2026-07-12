use crate::{
    codegen::CoopCaps,
    compile::{CompileOptions, ExecutionPlan},
    graph::Graph,
};
use serde::{Deserialize, Serialize};
use std::{io, path::Path};

/// Increment whenever the serialized execution plan or build pipeline changes
/// in a way that can make an older plan unsafe to reuse.
const CACHE_FORMAT_VERSION: u32 = 2;

/// Cached execution plan with a graph fingerprint for invalidation.
#[derive(Serialize, Deserialize)]
struct CachedPlan {
    #[serde(default)]
    format_version: u32,
    graph_hash: u64,
    /// Hash of mode, compiler options, target capabilities, and environment
    /// switches that affect optimization/code generation. Zero is the public
    /// `save_plan`/`load_plan` compatibility key.
    #[serde(default)]
    build_hash: u64,
    plan: ExecutionPlan,
}

/// Save a compiled execution plan to a RON file.
///
/// The forward graph hash is stored alongside the plan so that
/// stale caches can be detected on load.
pub fn save_plan(plan: &ExecutionPlan, forward_graph: &Graph, path: &Path) -> io::Result<()> {
    save_plan_impl(plan, forward_graph, 0, path)
}

fn save_plan_impl(
    plan: &ExecutionPlan,
    forward_graph: &Graph,
    build_hash: u64,
    path: &Path,
) -> io::Result<()> {
    let cached = CachedPlan {
        format_version: CACHE_FORMAT_VERSION,
        graph_hash: hash_graph(forward_graph),
        build_hash,
        plan: plan.clone(),
    };
    let ron_str = ron::ser::to_string_pretty(&cached, ron::ser::PrettyConfig::default())
        .map_err(io::Error::other)?;
    std::fs::write(path, ron_str)
}

/// Load a previously cached execution plan from a RON file.
///
/// Returns `None` if the file doesn't exist or the graph hash
/// doesn't match (i.e. the forward graph has changed).
pub fn load_plan(forward_graph: &Graph, path: &Path) -> io::Result<Option<ExecutionPlan>> {
    load_plan_impl(forward_graph, 0, path)
}

fn load_plan_impl(
    forward_graph: &Graph,
    build_hash: u64,
    path: &Path,
) -> io::Result<Option<ExecutionPlan>> {
    let data = match std::fs::read_to_string(path) {
        Ok(d) => d,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };
    let cached: CachedPlan =
        ron::from_str(&data).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    if cached.format_version != CACHE_FORMAT_VERSION {
        log::info!(
            "cache invalidated: format version {} != {}",
            cached.format_version,
            CACHE_FORMAT_VERSION,
        );
        return Ok(None);
    }
    if cached.graph_hash != hash_graph(forward_graph) {
        log::info!("cache invalidated: graph hash mismatch");
        return Ok(None);
    }
    if cached.build_hash != build_hash {
        log::info!("cache invalidated: build configuration mismatch");
        return Ok(None);
    }
    Ok(Some(cached.plan))
}

/// Save a plan built by the unified `build()` pipeline.
pub(crate) fn save_build_plan(
    plan: &ExecutionPlan,
    forward_graph: &Graph,
    build_hash: u64,
    path: &Path,
) -> io::Result<()> {
    save_plan_impl(plan, forward_graph, build_hash, path)
}

/// Load a plan built by the unified `build()` pipeline.
pub(crate) fn load_build_plan(
    forward_graph: &Graph,
    build_hash: u64,
    path: &Path,
) -> io::Result<Option<ExecutionPlan>> {
    load_plan_impl(forward_graph, build_hash, path)
}

/// Fingerprint every build input that can alter the execution plan.
///
/// `mode_tag` is deliberately numeric to keep this module independent of the
/// higher-level `train::Mode` type (`0` = training, `1` = inference).
pub(crate) fn hash_build_config(
    options: &CompileOptions,
    mode_tag: u8,
    skip_full_optimize: bool,
    coop_caps: CoopCaps,
) -> u64 {
    #[derive(Serialize)]
    struct BuildFingerprint<'a> {
        package_version: &'a str,
        options: &'a CompileOptions,
        mode_tag: u8,
        skip_full_optimize: bool,
        coop_caps: CoopCaps,
        traffic_cost: bool,
        flash_forward_coop: bool,
        flash_backward_coop: bool,
        flash_ept_cap: u32,
    }

    let fingerprint = BuildFingerprint {
        package_version: env!("CARGO_PKG_VERSION"),
        options,
        mode_tag,
        skip_full_optimize,
        coop_caps,
        traffic_cost: std::env::var("MEGANEURA_NO_TRAFFIC_COST").is_err(),
        flash_forward_coop: std::env::var("MEGANEURA_FLASH_FWD_COOP").as_deref() != Ok("0"),
        flash_backward_coop: std::env::var("MEGANEURA_FLASH_BWD_COOP").as_deref() != Ok("0"),
        flash_ept_cap: crate::codegen::flash_ept_cap(),
    };
    hash_serializable(&fingerprint)
}

/// Compute a semantic fingerprint of a forward graph.
///
/// This includes complete op payloads (epsilon values, convolution geometry,
/// attention configuration, constant contents), tensor dtypes/shapes, edges,
/// outputs, and derived-parameter transforms. The old cache hashed only op
/// discriminants and shapes, allowing semantically different graphs to reuse
/// an unsafe stale plan.
fn hash_graph(graph: &Graph) -> u64 {
    #[derive(Serialize)]
    struct GraphFingerprint<'a> {
        nodes: &'a [crate::graph::Node],
        outputs: &'a [crate::graph::NodeId],
        num_param_grad_outputs: usize,
        derived_params: &'a [crate::graph::DerivedParam],
    }

    hash_serializable(&GraphFingerprint {
        nodes: graph.nodes(),
        outputs: graph.outputs(),
        num_param_grad_outputs: graph.num_param_grad_outputs(),
        derived_params: &graph.derived_params,
    })
}

fn hash_serializable(value: &impl Serialize) -> u64 {
    let encoded = ron::ser::to_string(value).expect("fingerprint serialization cannot fail");
    // Deterministic FNV-1a. This is an invalidation fingerprint, not a
    // cryptographic integrity check; the cache remains trusted local data.
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in encoded.bytes() {
        hash ^= u64::from(byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::compile;
    use crate::graph::Graph;

    #[test]
    fn test_cache_round_trip() {
        let mut g = Graph::new();
        let x = g.input("x", &[4, 784]);
        let w = g.parameter("w", &[784, 128]);
        let y = g.matmul(x, w);
        let h = g.relu(y);
        g.set_outputs(vec![h]);

        let plan = compile::compile(&g);
        let dir = std::env::temp_dir().join("meganeura_test_cache");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_plan.ron");

        // Save
        save_plan(&plan, &g, &path).unwrap();

        // Load with same graph — should succeed
        let loaded = load_plan(&g, &path).unwrap();
        assert!(loaded.is_some());
        let loaded = loaded.unwrap();
        assert_eq!(loaded.buffers.len(), plan.buffers.len());
        assert_eq!(loaded.dispatches.len(), plan.dispatches.len());

        // Load with different graph — should invalidate
        let mut g2 = Graph::new();
        let x2 = g2.input("x", &[4, 784]);
        let w2 = g2.parameter("w", &[784, 256]); // different shape
        let y2 = g2.matmul(x2, w2);
        g2.set_outputs(vec![y2]);

        let loaded2 = load_plan(&g2, &path).unwrap();
        assert!(loaded2.is_none());

        // Cleanup
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_load_missing_file() {
        let g = Graph::new();
        let path = std::env::temp_dir().join("meganeura_nonexistent_cache.ron");
        let result = load_plan(&g, &path).unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_load_corrupt_file() {
        let dir = std::env::temp_dir().join("meganeura_test_cache_corrupt");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("corrupt.ron");
        std::fs::write(&path, "this is not valid RON").unwrap();

        let g = Graph::new();
        let result = load_plan(&g, &path);
        assert!(result.is_err());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_hash_graph_deterministic() {
        let build = || {
            let mut g = Graph::new();
            let x = g.input("x", &[4, 8]);
            let w = g.parameter("w", &[8, 4]);
            let y = g.matmul(x, w);
            g.set_outputs(vec![y]);
            g
        };
        let h1 = hash_graph(&build());
        let h2 = hash_graph(&build());
        assert_eq!(h1, h2, "same graph should produce same hash");
    }

    #[test]
    fn test_hash_graph_differs_on_change() {
        let mut g1 = Graph::new();
        let x = g1.input("x", &[4, 8]);
        let w = g1.parameter("w", &[8, 4]);
        let y = g1.matmul(x, w);
        g1.set_outputs(vec![y]);

        let mut g2 = Graph::new();
        let x2 = g2.input("x", &[4, 8]);
        let w2 = g2.parameter("w", &[8, 5]); // different shape
        let y2 = g2.matmul(x2, w2);
        g2.set_outputs(vec![y2]);

        assert_ne!(hash_graph(&g1), hash_graph(&g2));
    }

    #[test]
    fn test_hash_graph_differs_on_name_change() {
        let mut g1 = Graph::new();
        let x = g1.input("x", &[4, 8]);
        g1.set_outputs(vec![x]);

        let mut g2 = Graph::new();
        let x = g2.input("y", &[4, 8]); // different name
        g2.set_outputs(vec![x]);

        assert_ne!(hash_graph(&g1), hash_graph(&g2));
    }

    #[test]
    fn test_hash_graph_includes_op_payloads() {
        let build = |eps| {
            let mut g = Graph::new();
            let x = g.input("x", &[4, 8]);
            let w = g.parameter("w", &[8]);
            let y = g.rms_norm(x, w, eps);
            g.set_outputs(vec![y]);
            g
        };
        assert_ne!(
            hash_graph(&build(1e-5)),
            hash_graph(&build(1e-6)),
            "changing an op attribute must invalidate the plan",
        );
    }

    #[test]
    fn test_hash_graph_includes_dtype_and_constant_data() {
        let mut f32_graph = Graph::new();
        let p = f32_graph.parameter("w", &[32, 4]);
        f32_graph.set_outputs(vec![p]);

        let mut f16_graph = Graph::new();
        let p = f16_graph.parameter_f16("w", &[32, 4]);
        f16_graph.set_outputs(vec![p]);
        assert_ne!(hash_graph(&f32_graph), hash_graph(&f16_graph));

        let constant_graph = |value| {
            let mut g = Graph::new();
            let c = g.constant(vec![value; 4], &[4]);
            g.set_outputs(vec![c]);
            g
        };
        assert_ne!(
            hash_graph(&constant_graph(1.0)),
            hash_graph(&constant_graph(2.0)),
        );
    }

    #[test]
    fn test_build_hash_includes_mode_options_and_target() {
        let defaults = CompileOptions::default();
        let base = hash_build_config(&defaults, 0, false, CoopCaps::default());
        assert_ne!(
            base,
            hash_build_config(&defaults, 1, false, CoopCaps::default()),
        );
        assert_ne!(
            base,
            hash_build_config(&defaults, 0, true, CoopCaps::default()),
        );
        assert_ne!(
            base,
            hash_build_config(
                &CompileOptions {
                    use_schedule_pointwise: false,
                    ..CompileOptions::default()
                },
                0,
                false,
                CoopCaps::default(),
            ),
        );
        assert_ne!(
            base,
            hash_build_config(
                &defaults,
                0,
                false,
                CoopCaps {
                    f16_tile: 16,
                    f32_tile: 0,
                },
            ),
        );
    }

    #[test]
    fn test_build_cache_rejects_configuration_mismatch() {
        let mut g = Graph::new();
        let x = g.input("x", &[4, 8]);
        g.set_outputs(vec![x]);
        let plan = compile::compile(&g);
        let path = std::env::temp_dir().join("meganeura_test_build_cache.ron");
        save_build_plan(&plan, &g, 11, &path).unwrap();
        assert!(load_build_plan(&g, 11, &path).unwrap().is_some());
        assert!(load_build_plan(&g, 12, &path).unwrap().is_none());
        let _ = std::fs::remove_file(path);
    }
}
