enable wgpu_cooperative_matrix;

struct Params {
    m: u32,
    n: u32,
    k: u32,
    _pad: u32,
}

var<storage> matrix_a: array<vec4<f32>>;
var<storage> matrix_b: array<vec4<f32>>;
var<storage, read_write> matrix_c: array<f32>;
var<uniform> params: Params;
var<workgroup> shared_a: array<f32, 256>;
var<workgroup> shared_b: array<f32, 256>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_index) lane: u32, @builtin(subgroup_id) subgroup: u32) {
    let m = params.m;
    let n = params.n;
    let k = params.k;
    let row = lane / 4u;
    let col = (lane % 4u) * 4u;
    let row0 = group.x * 16u;
    let col0 = group.y * 16u;
    var acc = coop_mat16x16<f32,C>();
    for (var t = 0u; t < k; t += 16u) {
        $STAGE_A
        $STAGE_B
        workgroupBarrier();
        let a = coopLoadT<coop_mat16x16<f32,A>>(&shared_a[0], 16u);
        let b = coopLoadT<coop_mat16x16<f32,B>>(&shared_b[0], 16u);
        acc = coopMultiplyAdd(a, b, acc);
        workgroupBarrier();
    }
    let index = row0 * n + col0;
    if subgroup == 0u {
        coopStoreT(acc, &matrix_c[index], n);
    }
}
