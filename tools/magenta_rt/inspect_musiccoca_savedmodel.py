"""Parse saved_model.pb directly to find the mapping
tf_var_leaves.N → symbolic.variable.name."""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from pathlib import Path
import tensorflow as tf
from tensorflow.core.protobuf import saved_model_pb2

PB_PATH = Path("/x/Hub/models--google--magenta-realtime/snapshots/c05f8d6d608afd588469b7a8ef0929d5a1f8f6bb/savedmodels/musiccoca_mv212f_cpu_compat/saved_model.pb")

print(f"Reading {PB_PATH}...")
with open(PB_PATH, 'rb') as f:
    sm = saved_model_pb2.SavedModel()
    sm.ParseFromString(f.read())

print(f"  meta_graphs: {len(sm.meta_graphs)}")

# Find the object graph proto.
for mg in sm.meta_graphs:
    obj_graph = mg.object_graph_def
    print(f"  object_graph_def nodes: {len(obj_graph.nodes)}")
    if not obj_graph.nodes:
        continue

    # Walk the object graph, building dotted paths for each node.
    # node[0] is root; children link to other nodes by index.
    # Build name → idx map by traversing.
    name_for_idx = {0: ""}
    queue = [(0, "")]
    seen = {0}
    while queue:
        idx, prefix = queue.pop(0)
        node = obj_graph.nodes[idx]
        for child in node.children:
            child_idx = child.node_id
            child_name = child.local_name
            full = (prefix + "." + child_name) if prefix else child_name
            if child_idx not in seen:
                seen.add(child_idx)
                name_for_idx[child_idx] = full
                queue.append((child_idx, full))

    print(f"  Found {len(name_for_idx)} named nodes")

    # Find nodes that are TrackableObjectGraph variables.
    n_vars = 0
    var_list = []  # list of (leaf_idx_if_known, dotted_name, shape)
    for idx, node in enumerate(obj_graph.nodes):
        # Variables have node.variable populated.
        if node.HasField("variable"):
            v = node.variable
            shape = [d.size for d in v.shape.dim] if v.HasField("shape") else []
            name = name_for_idx.get(idx, f"<idx{idx}>")
            var_list.append((idx, name, shape, v.dtype))
            n_vars += 1
    print(f"  Variables: {n_vars}")
    print(f"\n  First 30 variables (idx, name, shape, dtype):")
    for v in var_list[:30]:
        idx, name, shape, dtype = v
        print(f"    [{idx:4d}]  {name:<80s}  shape={shape}  dtype={dtype}")

    # We also need to know the FLAT LEAVES order. Look for the saver_def, the
    # main checkpoint usually maps tf_var_leaves.N → some flat index.
    # In V2 checkpoint format, leaf N corresponds to a particular variable.
    # The "tf_var_leaves" is actually a TF attribute path on the loaded
    # `_UserObject`. To match leaf N to a real variable, we need to find a
    # node whose name is "tf_var_leaves" and whose children are indexed [0], [1] etc.
    for idx, node in enumerate(obj_graph.nodes):
        name = name_for_idx.get(idx, "")
        if "tf_var_leaves" in name:
            # Print children
            children_names = [(c.local_name, c.node_id) for c in node.children]
            if children_names:
                print(f"\n  tf_var_leaves node @ {idx}, name={name}, "
                      f"children_count={len(children_names)}, first_child={children_names[0]}")
                break

    # Detailed: print mapping leaf_N → dotted path
    print(f"\n  Leaf → variable mapping:")
    leaf_pattern_count = 0
    for idx, node in enumerate(obj_graph.nodes):
        name = name_for_idx.get(idx, "")
        if "tf_var_leaves" in name and node.HasField("variable"):
            v = node.variable
            shape = [d.size for d in v.shape.dim] if v.HasField("shape") else []
            print(f"    {name}  shape={shape}")
            leaf_pattern_count += 1
            if leaf_pattern_count > 20:
                print(f"    ... ({n_vars - 20} more)")
                break

    break  # only process first meta_graph
