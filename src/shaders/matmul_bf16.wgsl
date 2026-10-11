enable wgpu_bfloat16;
enable wgpu_cooperative_matrix;

struct Params { m: u32, n: u32, k: u32, _pad: u32 }
var<storage> matrix_a: array<bf16>;
var<storage> matrix_b: array<bf16>;
var<storage, read_write> matrix_c: array<f32>;
$ADD_DECL
var<uniform> params: Params;
var<workgroup> stage_a: array<bf16, 256>;
var<workgroup> stage_b: array<bf16, 256>;
var<workgroup> result: array<f32, 256>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_index) lane: u32,
        @builtin(subgroup_id) subgroup: u32) {
    let row = group.x * 16u;
    let col = group.y * 16u;
    var accumulator = coop_mat16x16<f32, C>();
    for (var start = 0u; start < params.k; start += 16u) {
        for (var index = lane; index < 256u; index += 64u) {
            let ar = row + index / 16u;
            let ak = start + index % 16u;
            let bk = start + index / 16u;
            let bc = col + index % 16u;
            var a = bf16(0.0f);
            var b = bf16(0.0f);
            if ar < params.m && ak < params.k { a = matrix_a[$A_INDEX]; }
            if bk < params.k && bc < params.n { b = matrix_b[$B_INDEX]; }
            stage_a[index] = a;
            stage_b[index] = b;
        }
        workgroupBarrier();
        let a = coopLoadT<coop_mat16x16<bf16, A>>(&stage_a[0], 16u);
        let b = coopLoadT<coop_mat16x16<bf16, B>>(&stage_b[0], 16u);
        accumulator = coopMultiplyAdd(a, b, accumulator);
        workgroupBarrier();
    }
    if subgroup == 0u { coopStoreT(accumulator, &result[0], 16u); }
    workgroupBarrier();
    for (var offset = lane; offset < 256u; offset += 64u) {
        let r = row + offset / 16u;
        let c = col + offset % 16u;
        if r < params.m && c < params.n {
            let index = r * params.n + c;
            matrix_c[index] = result[offset]$ADD;
        }
    }
}
