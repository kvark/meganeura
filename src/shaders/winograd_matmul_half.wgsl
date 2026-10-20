enable wgpu_cooperative_matrix;
$EXTENSIONS

struct Params { m: u32, n: u32, k: u32, batches: u32 }
var<storage> matrix_a: array<$TYPE>;
var<storage> matrix_b: array<$TYPE>;
var<storage, read_write> matrix_c: array<f32>;
var<uniform> params: Params;

// Lowering admits only complete native tiles. Read the half-sized transform
// buffers without workgroup staging, using the actual subgroup width.
@compute @workgroup_size($THREADS)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(subgroup_id) sg: u32,
        @builtin(subgroup_size) sg_size: u32) {
    let subgroups = $THREADSu / sg_size;
    for (var tile = sg; tile < ($ROWSu / $Mu) * ($COLUMNSu / $Nu); tile += subgroups) {
        let row = group.y * $ROWSu + (tile / ($COLUMNSu / $Nu)) * $Mu;
        let col = group.x * $COLUMNSu + (tile % ($COLUMNSu / $Nu)) * $Nu;
        var acc = $C_TILE();
        for (var kk = 0u; kk < params.k; kk += $Ku) {
            let a = coopLoadT<$A_TILE>(&matrix_a[(group.z * params.m + row) * params.k + kk], params.k);
            let b = coopLoadT<$B_TILE>(&matrix_b[(group.z * params.k + kk) * params.n + col], params.n);
            acc = coopMultiplyAdd(a, b, acc);
        }
        coopStoreT(acc, &matrix_c[(group.z * params.m + row) * params.n + col], params.n);
    }
}
