use crate::{
    codegen::CoopCaps,
    compile::{CompileOptions, ExecutionPlan},
    graph::Graph,
    optimize::OptimizeConfig,
};
use serde::{Deserialize, Serialize};
use std::{io, path::Path};

/// Increment whenever the serialized execution plan or build pipeline changes
/// in a way that can make an older plan unsafe to reuse.
// Version 46 pairs gradients with their parameters explicitly, which drops
// the pair an untrained one-element parameter used to get.
// Version 45 raises fused PointwiseDAG arity from 3 to 7 and adds ReductionKernel::preserve_tree_order.
// Version 44 adds the register-blocked attention kernel.
// Version 43 adds ExecutionPlan::uniform_constants (Op::Fill).
// Version 42 adds ExecutionPlan::logical_buffers and full target fingerprinting.
// Version 41 adds subgroup-tiled Winograd and convolution grids, WorkgroupMatmulShape
// subgroup_rows/prefetch, and ScalarMatmulShape vector_fragments.
// Version 40 splits MaxPool2dGrad into a winner-index pass and a gather pass.
// Version 39 adds the ArgMax reduction lowering `Graph::argmax_inner`.
// Version 38 marks parameter-invariant Winograd weight transforms and gives
// low-occupancy Winograd products 32-wide tiles.
// Version 37 records reduced private store output types.
// Version 36 changes F16 attention backward to 32-row workgroups and a
// precomputed F32 row-dot input instead of the forward output.
// Version 35 adds typed Winograd transforms and native half-storage products.
// Version 34 separates GroupNorm statistics by grouping and adds fused
// Winograd residuals and narrow scalar products.
// Version 33 integrates typed workgroup cooperative tile decisions.
// Version 32 adds the opt-in workgroup cooperative product.
// Version 31 replaces the BF16 cooperative kernel variant with typed checked
// products that record operand, multiplication and output types.
// Version 30 uses the ordinary dense matrix entries for BF16, with storage
// precision and K partition count in the cooperative kernel variant.
// Version 29 adds BF16 storage, typed pointwise boundaries and the BF16 matrix
// K partition count. Versions 27–28 were used by pre-merge BF16 builds.
// Version 26 records scalar convolution column interleaving and rejects plans
// from the experimental v24 builds with the legacy attention encoding.
// Version 25 stores RoPE and attention parameters in uniform byte order,
// including padding. RoPE's head_dim moves after axes_dim.
// Version 24 stores explicit key/value and causal-row lengths in forward
// attention dispatches. Version 23 used kv_seq == 0 as a causal sentinel.
// BF16 variants also record matrix stores and pointwise storage; older readers
// would otherwise ignore the storage field and read halves as f32.
// Version 23 allows native16 f32 attention gradients. The same family/path
// serialized by older readers always generated native8, even on native16 GPUs.
// Version 22 records forward GEMV column groups, which set the workgroup count
// an older reader would pair with a one-group kernel.
// Version 21 names attention backward kernels by family, part and path.
// Version 20 changes f32 8x8 cooperative matmul dispatches to 32x32 output tiles
// and bounds-checked addend loads, changing geometry and buffer padding,
// and adds f32 cooperative backward attention.
// Version 19 stores per-dispatch scalar attention-gradient layouts.
// Version 18 restores the flat dispatch representation.
// Version 17 stores typed dispatch operations, including operands and dimensions.
// Version 16 bounds cooperative attention staging by device shared memory.
// Version 15 gives `WinogradConv2d` its logical kernel instead of a derived
// transformed parameter, and adds the adjoint (input-gradient) form.
// Version 14 marks generated kernels as `ShaderEntry::Generated` and drops
// the hand-written elementwise, softmax and RMSNorm entries.
// Version 13 reduces loss partials into a scalar, lowers LayerNorm to the
// two-pass kernel, carries RoPE's static offset into the dynamic kernels, and
// changes which dispatches fusion may merge.
const CACHE_FORMAT_VERSION: u32 = 46;

/// Cached execution plan with a graph fingerprint for invalidation.
#[derive(Serialize, Deserialize)]
struct CachedPlan<Plan = ExecutionPlan> {
    #[serde(default)]
    format_version: u32,
    graph_hash: u64,
    /// Hash of mode, rewrite/compiler options, and target capabilities.
    /// Environment overrides are resolved into those typed options before
    /// hashing. Zero is the public `save_plan`/`load_plan` compatibility key.
    #[serde(default)]
    build_hash: u64,
    plan: Plan,
}

/// Save a compiled execution plan to a RON file.
///
/// The forward graph hash is stored alongside the plan so that
/// stale caches can be detected on load.
pub fn save_plan(plan: &ExecutionPlan, forward_graph: &Graph, path: &Path) -> io::Result<()> {
    save_plan_impl(plan, forward_graph, 0, path)
}

static NEXT_SAVE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

/// Resolve symlinks (including dangling ones) so atomic rename updates the
/// link target instead of replacing the symlink with a regular file.
fn resolve_save_destination(path: &Path) -> io::Result<std::path::PathBuf> {
    let mut cur = path.to_path_buf();
    for _ in 0..40 {
        match std::fs::symlink_metadata(&cur) {
            Ok(meta) if meta.file_type().is_symlink() => {
                let target = std::fs::read_link(&cur)?;
                cur = if target.is_absolute() {
                    target
                } else {
                    cur.parent().map_or(target.clone(), |p| p.join(&target))
                };
            }
            Ok(_) => return Ok(cur),
            Err(err) if err.kind() == io::ErrorKind::NotFound => return Ok(cur),
            Err(err) => return Err(err),
        }
    }
    Err(io::Error::new(
        io::ErrorKind::InvalidInput,
        "too many levels of symbolic links",
    ))
}

fn save_plan_impl(
    plan: &ExecutionPlan,
    forward_graph: &Graph,
    build_hash: u64,
    path: &Path,
) -> io::Result<()> {
    use std::io::Write as _;
    use std::sync::atomic::Ordering;

    let cached = CachedPlan {
        format_version: CACHE_FORMAT_VERSION,
        graph_hash: hash_graph(forward_graph),
        build_hash,
        plan: plan.clone(),
    };
    let ron_str = ron::ser::to_string_pretty(&cached, ron::ser::PrettyConfig::default())
        .map_err(io::Error::other)?;
    let dest = resolve_save_destination(path)?;
    let existing_perms = std::fs::metadata(&dest).ok().map(|m| m.permissions());
    if let Some(parent) = dest.parent()
        && !parent.as_os_str().is_empty()
    {
        std::fs::create_dir_all(parent)?;
    }
    let file_name = dest.file_name().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, "cache path needs a file name")
    })?;
    // Keep the temporary beside the destination for an atomic rename, using
    // `create_new(true)` so we never truncate a sibling file or another writer.
    let (tmp, mut file) = loop {
        let mut name = file_name.to_os_string();
        name.push(format!(
            ".{}-{}.partial",
            std::process::id(),
            NEXT_SAVE.fetch_add(1, Ordering::Relaxed),
        ));
        let candidate = dest.with_file_name(name);
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&candidate)
        {
            Ok(file) => break (candidate, file),
            Err(err) if err.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(err) => return Err(err),
        }
    };
    let write_result = (|| -> io::Result<()> {
        if let Some(perms) = existing_perms {
            file.set_permissions(perms)?;
        }
        file.write_all(ron_str.as_bytes())?;
        file.sync_all()
    })();
    drop(file);
    if let Err(err) = write_result {
        let _ = std::fs::remove_file(&tmp);
        return Err(err);
    }
    if let Err(err) = std::fs::rename(&tmp, &dest) {
        let _ = std::fs::remove_file(&tmp);
        return Err(err);
    }
    Ok(())
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
    // Read compatibility metadata before decoding the versioned plan schema.
    let cached: CachedPlan<serde::de::IgnoredAny> =
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
    let cached: CachedPlan =
        ron::from_str(&data).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
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

/// Hardware and driver identity that governs cached plan reuse.
#[derive(Clone, Debug, Default, Serialize)]
pub(crate) struct BuildTargetFingerprint<'a> {
    pub device_name: &'a str,
    pub driver_name: &'a str,
    pub driver_info: &'a str,
    pub subgroup_size: u32,
    pub coop_caps: CoopCaps,
    pub shared_memory_bytes: u32,
}

/// Fingerprint every build input that can alter the execution plan.
///
/// `mode_tag` is deliberately numeric to keep this module independent of the
/// higher-level `train::Mode` type (`0` = training, `1` = inference).
pub(crate) fn hash_build_config(
    options: &CompileOptions,
    optimize: &OptimizeConfig,
    mode_tag: u8,
    skip_full_optimize: bool,
    target: &BuildTargetFingerprint<'_>,
) -> u64 {
    #[derive(Serialize)]
    struct BuildFingerprint<'a> {
        package_version: &'a str,
        options: &'a CompileOptions,
        optimize: &'a OptimizeConfig,
        mode_tag: u8,
        skip_full_optimize: bool,
        target: &'a BuildTargetFingerprint<'a>,
    }

    let fingerprint = BuildFingerprint {
        package_version: env!("CARGO_PKG_VERSION"),
        options,
        optimize,
        mode_tag,
        skip_full_optimize,
        target,
        // Flash coop flags and EPT caps live in `options`, hashed above.
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
        param_grads: Vec<(crate::graph::NodeId, crate::graph::NodeId)>,
        derived_params: &'a [crate::graph::DerivedParam],
    }

    hash_serializable(&GraphFingerprint {
        nodes: graph.nodes(),
        outputs: graph.outputs(),
        param_grads: graph.param_grads().collect(),
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
        assert_eq!(loaded.param_types, plan.param_types);

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
    fn bf16_storage_plan_round_trip() {
        let mut graph = Graph::new();
        let x = graph.input("x", &[3, 7]);
        let w = graph.parameter("w", &[7, 5]);
        let x = graph.to_bf16(x);
        let w = graph.to_bf16(w);
        let product = graph.matmul(x, w);
        let product = graph.to_bf16(product);
        let product = graph.from_bf16(product);
        let scaled = graph.scale(product, 0.125);
        let scaled = graph.to_bf16(scaled);
        let output = graph.from_bf16(scaled);
        graph.set_outputs(vec![output]);
        let plan = compile::compile(&graph);
        assert!(
            plan.dispatches
                .iter()
                .any(|d| d.pointwise_storage.bf16_output)
        );
        let path =
            std::env::temp_dir().join(format!("meganeura-cache-bf16-{}.ron", std::process::id()));
        save_plan(&plan, &graph, &path).unwrap();
        assert_eq!(load_plan(&graph, &path).unwrap().unwrap(), plan);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn cache_without_logical_parameter_types_is_invalidated() {
        let mut graph = Graph::new();
        let p = graph.parameter("p", &[2, 3]);
        graph.set_outputs(vec![p]);
        let mut plan = compile::compile(&graph);
        plan.param_types.clear();
        let legacy = CachedPlan {
            format_version: 4,
            graph_hash: hash_graph(&graph),
            build_hash: 0,
            plan,
        };
        let path =
            std::env::temp_dir().join(format!("meganeura-cache-v4-{}.ron", std::process::id()));
        std::fs::write(&path, ron::ser::to_string(&legacy).unwrap()).unwrap();
        assert!(load_plan(&graph, &path).unwrap().is_none());
        std::fs::remove_file(path).unwrap();
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
        let optimize = OptimizeConfig::default();
        let base_target = BuildTargetFingerprint {
            device_name: "TestGPU",
            driver_name: "Mesa",
            driver_info: "26.0",
            subgroup_size: 32,
            coop_caps: CoopCaps::default(),
            shared_memory_bytes: 49_152,
        };
        let base = hash_build_config(&defaults, &optimize, 0, false, &base_target);
        assert_ne!(
            base,
            hash_build_config(&defaults, &optimize, 1, false, &base_target),
        );
        assert_ne!(
            base,
            hash_build_config(&defaults, &optimize, 0, true, &base_target),
        );
        assert_ne!(
            base,
            hash_build_config(
                &defaults,
                &optimize,
                0,
                false,
                &BuildTargetFingerprint {
                    shared_memory_bytes: 65_536,
                    ..base_target
                },
            ),
        );
        assert_ne!(
            base,
            hash_build_config(
                &defaults,
                &optimize,
                0,
                false,
                &BuildTargetFingerprint {
                    device_name: "OtherGPU",
                    ..base_target
                },
            ),
        );
        assert_ne!(
            base,
            hash_build_config(
                &defaults,
                &optimize,
                0,
                false,
                &BuildTargetFingerprint {
                    driver_info: "26.1",
                    ..base_target
                },
            ),
        );
        assert_ne!(
            base,
            hash_build_config(
                &defaults,
                &optimize,
                0,
                false,
                &BuildTargetFingerprint {
                    subgroup_size: 64,
                    ..base_target
                },
            ),
        );
        assert_ne!(
            base,
            hash_build_config(
                &CompileOptions {
                    fuse_dispatches: false,
                    ..CompileOptions::default()
                },
                &optimize,
                0,
                false,
                &base_target,
            ),
        );
        assert_ne!(
            base,
            hash_build_config(
                &defaults,
                &optimize,
                0,
                false,
                &BuildTargetFingerprint {
                    coop_caps: CoopCaps {
                        f16_tile: 16,
                        f32_tile: 0,
                        f16_attention_rows: 0,
                    },
                    ..base_target
                },
            ),
        );
    }

    #[test]
    fn build_cache_rejects_rewrite_configuration_changes() {
        use crate::optimize::{ExtractionCost, OptimizeMode};

        let mut graph = Graph::new();
        let x = graph.input("x", &[4, 8]);
        graph.set_outputs(vec![x]);
        let options = CompileOptions::default();
        let defaults = OptimizeConfig::default();
        let target = BuildTargetFingerprint {
            shared_memory_bytes: 49_152,
            ..BuildTargetFingerprint::default()
        };
        let hash = |config: &OptimizeConfig| hash_build_config(&options, config, 1, false, &target);
        let plan = compile::compile(&graph);
        let path = std::env::temp_dir().join(format!(
            "meganeura_test_rewrite_cache_{}.ron",
            std::process::id(),
        ));
        save_build_plan(&plan, &graph, hash(&defaults), &path).unwrap();
        assert!(
            load_build_plan(&graph, hash(&defaults), &path)
                .unwrap()
                .is_some()
        );

        for config in [
            OptimizeConfig {
                mode: OptimizeMode::Off,
                ..defaults
            },
            OptimizeConfig {
                mode: OptimizeMode::EgglogWhole,
                ..defaults
            },
            OptimizeConfig {
                extraction_cost: ExtractionCost::AstSize,
                ..defaults
            },
            OptimizeConfig {
                saturation_cutoff: defaults.saturation_cutoff + 1,
                ..defaults
            },
            OptimizeConfig {
                no_winograd: true,
                ..defaults
            },
            OptimizeConfig {
                pack_swiglu: false,
                ..defaults
            },
        ] {
            assert!(
                load_build_plan(&graph, hash(&config), &path)
                    .unwrap()
                    .is_none(),
                "rewrite configuration reused an incompatible plan: {config:?}",
            );
        }
        std::fs::remove_file(path).unwrap();
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

#[cfg(test)]
mod cache_format_tests {
    use super::*;
    use crate::compile;

    #[test]
    fn legacy_attention_dispatch_caches_are_invalidated() {
        let mut graph = Graph::new();
        let q = graph.input("q", &[4, 8]);
        let k = graph.input("k", &[4, 8]);
        let v = graph.input("v", &[4, 8]);
        let output = graph.causal_attention(q, k, v, 1, 1, 8);
        graph.set_outputs(vec![output]);

        let mut plan = compile::compile(&graph);
        // Version 23 encoded causal attention with kv_seq == 0 and had no
        // causal_rows word. Interpreting these parameters as version 24 would
        // make every query row see zero keys. Experimental convolution builds
        // also wrote this old attention layout with format version 24.
        plan.dispatches[0].params = vec![4, 0, (1 << 16) | 1, 8];
        let mut legacy = CachedPlan {
            format_version: 23,
            graph_hash: hash_graph(&graph),
            build_hash: 0,
            plan,
        };
        let path = std::env::temp_dir().join(format!(
            "meganeura-attention-cache-v23-{}.ron",
            std::process::id()
        ));
        for version in [23, 24] {
            legacy.format_version = version;
            std::fs::write(&path, ron::ser::to_string(&legacy).unwrap()).unwrap();
            assert!(load_plan(&graph, &path).unwrap().is_none());
        }
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn rope_dispatch_cache_v24_is_invalidated() {
        let mut graph = Graph::new();
        let x = graph.input("x", &[3, 32]);
        let output = graph.rope_with_offset(x, 10_000.0, 7, 16);
        graph.set_outputs(vec![output]);
        let plan = compile::compile(&graph);
        let mut legacy_plan = plan.clone();
        // Same length as the new layout, but head_dim occupied word 4.
        legacy_plan.dispatches[0].params = vec![3, 32, 10_000f32.to_bits(), 7, 16, 0, 0, 0];
        let legacy = CachedPlan {
            format_version: 24,
            graph_hash: hash_graph(&graph),
            build_hash: 0,
            plan: legacy_plan,
        };
        let path = std::env::temp_dir().join(format!(
            "meganeura-rope-cache-v24-{}.ron",
            std::process::id()
        ));
        std::fs::write(&path, ron::ser::to_string(&legacy).unwrap()).unwrap();
        assert!(load_plan(&graph, &path).unwrap().is_none());
        save_plan(&plan, &graph, &path).unwrap();
        let restored = load_plan(&graph, &path).unwrap().unwrap();
        assert_eq!(restored.dispatches, plan.dispatches);
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn previous_cache_format_is_invalidated() {
        let mut graph = Graph::new();
        let x = graph.input("x", &[2, 32]);
        let w = graph.parameter_f16("w", &[20, 32]);
        let product = graph.matmul_bt(x, w);
        let output = graph.clamp(product, -3.25, 4.5);
        graph.set_outputs(vec![output]);
        let legacy = CachedPlan {
            format_version: CACHE_FORMAT_VERSION - 1,
            graph_hash: hash_graph(&graph),
            build_hash: 0,
            // Reject the obsolete schema before deserializing the execution plan.
            plan: "legacy execution plan schema",
        };
        let path = std::env::temp_dir().join(format!(
            "meganeura-previous-cache-format-{}.ron",
            std::process::id()
        ));
        std::fs::write(&path, ron::ser::to_string(&legacy).unwrap()).unwrap();
        assert!(load_plan(&graph, &path).unwrap().is_none());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn save_plan_does_not_clobber_preexisting_sibling_files() {
        let mut graph = Graph::new();
        let x = graph.input("x", &[4]);
        let y = graph.relu(x);
        graph.set_outputs(vec![y]);
        let plan = compile::compile(&graph);

        let dir =
            std::env::temp_dir().join(format!("meganeura-cache-collision-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("plan.ron");
        // Pre-populate several upcoming candidate temp file names so even if
        // another test advances `NEXT_SAVE` concurrently, `create_new(true)`
        // hits a collision and skips over existing files without truncating.
        let base = NEXT_SAVE.load(std::sync::atomic::Ordering::Relaxed);
        let colliders: Vec<_> = (base..base + 16)
            .map(|n| dir.join(format!("plan.ron.{}-{n}.partial", std::process::id())))
            .collect();
        for collider in &colliders {
            std::fs::write(collider, b"keep-partial-sibling").unwrap();
        }

        save_plan(&plan, &graph, &path).unwrap();
        for collider in &colliders {
            assert_eq!(std::fs::read(collider).unwrap(), b"keep-partial-sibling");
        }
        let restored = load_plan(&graph, &path).unwrap().unwrap();
        assert_eq!(restored.dispatches, plan.dispatches);

        #[cfg(unix)]
        {
            let target_dir = dir.join("shared");
            let target = target_dir.join("target.ron");
            let link = dir.join("dangling.ron");
            std::os::unix::fs::symlink(&target, &link).unwrap();
            save_plan(&plan, &graph, &link).unwrap();
            assert!(
                std::fs::symlink_metadata(&link)
                    .unwrap()
                    .file_type()
                    .is_symlink(),
                "saving through a dangling symlink must preserve the symlink"
            );
            assert!(
                target.exists(),
                "target file must be created through symlink"
            );
            let restored = load_plan(&graph, &link).unwrap().unwrap();
            assert_eq!(restored.dispatches, plan.dispatches);
        }

        let _ = std::fs::remove_dir_all(&dir);
    }
}
