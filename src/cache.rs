use crate::compile::ExecutionPlan;
use crate::graph::Graph;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::io;
use std::path::Path;

/// Cached execution plan with a graph fingerprint for invalidation.
#[derive(Serialize, Deserialize)]
struct CachedPlan {
    graph_hash: u64,
    plan: ExecutionPlan,
}

/// Save a compiled execution plan to a RON file.
///
/// The forward graph hash is stored alongside the plan so that
/// stale caches can be detected on load.
pub fn save_plan(plan: &ExecutionPlan, forward_graph: &Graph, path: &Path) -> io::Result<()> {
    let cached = CachedPlan {
        graph_hash: hash_graph(forward_graph),
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
    let data = match std::fs::read_to_string(path) {
        Ok(d) => d,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(e) => return Err(e),
    };
    let cached: CachedPlan =
        ron::from_str(&data).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;

    if cached.graph_hash != hash_graph(forward_graph) {
        log::info!("cache invalidated: graph hash mismatch");
        return Ok(None);
    }
    Ok(Some(cached.plan))
}

/// Compute a lightweight fingerprint of a forward graph.
///
/// Hashes the node count, op discriminants, input edges, output list,
/// and tensor shapes. Any structural change invalidates the cache.
fn hash_graph(graph: &Graph) -> u64 {
    let mut hasher = DefaultHasher::new();
    graph.nodes().len().hash(&mut hasher);
    for node in graph.nodes() {
        std::mem::discriminant(&node.op).hash(&mut hasher);
        node.inputs.hash(&mut hasher);
        node.ty.shape.hash(&mut hasher);
        // Hash parameter/input names for identity
        match node.op {
            crate::graph::Op::Parameter { ref name } | crate::graph::Op::Input { ref name } => {
                name.hash(&mut hasher);
            }
            _ => {}
        }
    }
    graph.outputs().hash(&mut hasher);
    hasher.finish()
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
}
