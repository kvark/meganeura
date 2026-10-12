enable wgpu_bfloat16;
enable wgpu_cooperative_matrix;

struct Params { m: u32, n: u32, k: u32, _pad: u32 }
var<storage> matrix_a: array<bf16>;
var<storage> matrix_b: array<bf16>;
var<storage, read_write> matrix_c: array<$OUT_TYPE>;
$ADD_DECL
var<uniform> params: Params;
var<workgroup> stage_a: array<bf16, 1024>;
var<workgroup> stage_b: array<bf16, 1024>;
var<workgroup> result: array<f32, 1024>;
$ROUND

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_index) lane: u32,
        @builtin(subgroup_id) subgroup: u32,
        @builtin(subgroup_size) subgroup_size: u32) {
    let row = group.x * 32u;
    let col = group.y * 32u;
    // Four logical matrix tiles, independent of the device subgroup width.
    // Extra subgroups compute a valid tile but do not store it, keeping the
    // cooperative operations and workgroup barriers in uniform control flow.
    for (var base = 0u; base < 4u; base += 128u / subgroup_size) {
        let tile = base + subgroup;
        let a_offset = ((tile % 4u) / 2u) * 16u * 32u;
        let b_offset = (tile % 2u) * 16u;
        var accumulator = coop_mat16x16<f32, C>();
        for (var start = 0u; start < params.k; start += 32u) {
            for (var index = lane; index < 1024u; index += 128u) {
                let ar = row + index / 32u;
                let ak = start + index % 32u;
                let bk = start + index / 32u;
                let bc = col + index % 32u;
                var a = bf16(0.0f);
                var b = bf16(0.0f);
                if ar < params.m && ak < params.k { a = matrix_a[$A_INDEX]; }
                if bk < params.k && bc < params.n { b = matrix_b[$B_INDEX]; }
                stage_a[index] = a;
                stage_b[index] = b;
            }
            workgroupBarrier();
            let a0 = coopLoadT<coop_mat16x16<bf16, A>>(&stage_a[a_offset], 32u);
            let b0 = coopLoadT<coop_mat16x16<bf16, B>>(&stage_b[b_offset], 32u);
            accumulator = coopMultiplyAdd(a0, b0, accumulator);
            let a1 = coopLoadT<coop_mat16x16<bf16, A>>(&stage_a[a_offset + 16u], 32u);
            let b1 = coopLoadT<coop_mat16x16<bf16, B>>(&stage_b[b_offset + 512u], 32u);
            accumulator = coopMultiplyAdd(a1, b1, accumulator);
            workgroupBarrier();
        }
        if tile < 4u { coopStoreT(accumulator, &result[a_offset + b_offset], 32u); }
    }
    workgroupBarrier();
    for (var offset = lane; offset < 1024u; offset += 128u) {
        let r = row + offset / 32u;
        let c = col + offset % 32u;
        if r < params.m && c < params.n {
            let index = r * params.n + c;
            matrix_c[index] = $STORE(result[offset]$ADD);
        }
    }
}
