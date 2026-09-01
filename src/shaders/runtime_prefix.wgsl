// Convert one producer-owned active row count into dispatch-indirect
// arguments for every scalable dispatch in a configured compact branch.

struct PrefixParams {
    entries: u32,
    alignment: u32,
    max_rows: u32,
    _pad: u32,
};

var<storage, read> active_count: array<u32>;
// x = integral row units, y = workgroups per row unit.
var<storage, read> scales: array<vec4<u32>>;
// vec4 stride keeps every VkDispatchIndirectCommand suitably aligned; its
// first three words are exactly (x, y, z).
var<storage, read_write> indirect: array<vec4<u32>>;
var<uniform> params: PrefixParams;

@compute @workgroup_size(64)
fn prepare_runtime_prefix(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if (index >= params.entries) {
        return;
    }
    let active_rows = clamp(active_count[0], 1u, params.max_rows);
    let rows = min(
        ((active_rows + params.alignment - 1u) / params.alignment) * params.alignment,
        params.max_rows,
    );
    let scale = scales[index];
    indirect[index] = vec4<u32>(rows / scale.x * scale.y, 1u, 1u, 0u);
}
