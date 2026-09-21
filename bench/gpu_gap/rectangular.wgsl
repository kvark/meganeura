enable f16;
enable wgpu_cooperative_matrix;

struct Params { m: u32, n: u32, k: u32, pad: u32 }
var<uniform> params: Params;
var<storage> matrix_a: array<f32>;
var<storage> matrix_b: array<f32>;
var<storage, read_write> matrix_c: array<f32>;
$ADD_DECL
var<workgroup> shared_a: array<f16, $A_SIZE>;
var<workgroup> shared_b: array<f16, $B_SIZE>;
var<workgroup> shared_c: array<f32, $C_SIZE>;

fn weight(index: u32) -> f16 {
    return f16(matrix_b[index]);
}

@compute @workgroup_size($THREADS)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_index) lane: u32,
    @builtin(subgroup_id) subgroup: u32,
) {
    let n = params.n;
    let base_m = group.x * $BM;
    let base_n = group.y * $BN;
    $ACCUMULATORS
    for (var base_k = 0u; base_k < params.k; base_k += $BK) {
        for (var i = lane; i < $BM * $BK / 2u; i += $THREADS) {
            let row = base_m + i / ($BK / 2u);
            let col = base_k + 2u * (i % ($BK / 2u));
            var value = vec2<f16>();
            if row < params.m && col < params.k {
                value.x = f16(matrix_a[row * params.k + col]);
            }
            if row < params.m && col + 1u < params.k {
                value.y = f16(matrix_a[row * params.k + col + 1u]);
            }
            shared_a[2u * i] = value.x;
            shared_a[2u * i + 1u] = value.y;
        }
        for (var i = lane; i < $BN * $BK / 2u; i += $THREADS) {
            let row = base_n + 2u * (i / $BK);
            let col = base_k + i % $BK;
            var value = vec2<f16>();
            if row < params.n && col < params.k {
                value.x = weight(row * params.k + col);
            }
            if row + 1u < params.n && col < params.k {
                value.y = weight((row + 1u) * params.k + col);
            }
            let dst = (i % $BK) * $B_STRIDE + 2u * (i / $BK);
            shared_b[dst] = value.x;
            shared_b[dst + 1u] = value.y;
        }
        workgroupBarrier();
        for (var kk = 0u; kk < $BK; kk += $TK) {
            $MULTIPLIES
        }
        workgroupBarrier();
    }
    $STORES
    workgroupBarrier();
    for (var i = lane; i < $BM * $BN; i += $THREADS) {
        let row = base_m + i / $BN;
        let col = base_n + i % $BN;
        if row < params.m && col < params.n {
            let index = row * params.n + col;
            matrix_c[index] = shared_c[i] $ADD;
        }
    }
}
