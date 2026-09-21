struct Params { m: u32, n: u32, k: u32, pad: u32 }
var<uniform> params: Params;
var<storage> matrix_a: array<f32>;
var<storage> matrix_b: array<f32>;
var<storage, read_write> matrix_c: array<f32>;
$ADD
var<workgroup> shared_a: array<f32, $A_SIZE>;
var<workgroup> shared_b: array<f32, $B_SIZE>;

@compute @workgroup_size($X, $Y)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.y * $X + lid.x;
    let row0 = group.y * $M;
    let col0 = group.x * $N;
    $DECLARATIONS
    for (var k0 = 0u; k0 < params.k; k0 += $K) {
        for (var i = tid; i < $M * $K; i += $THREADS) {
            let row = row0 + i / $K;
            let col = k0 + i % $K;
            let a = select(0.0, matrix_a[row * params.k + col], row < params.m && col < params.k);
            shared_a[(i / $K) * $A_STRIDE + i % $K] = a;
        }
        for (var i = tid; i < $N * $K; i += $THREADS) {
            let row = k0 + i / $N;
            let col = col0 + i % $N;
            let b = select(0.0, matrix_b[row * params.n + col], row < params.k && col < params.n);
            shared_b[(i / $N) * $B_STRIDE + i % $N] = b;
        }
        workgroupBarrier();
        for (var kk = 0u; kk < $K; kk++) { $COMPUTE }
        workgroupBarrier();
    }
    $STORES
}
