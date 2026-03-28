// MatMulAT: C = A^T @ B where A=[K,M], B=[K,N], C=[M,N]
// A is accessed transposed: sA[ly*16+lx] = A[(t+ly)*M + (tile_row+lx)]

struct Params {
    m: u32,
    n: u32,
    k: u32,
    _pad: u32,
}

var<storage> matrix_a: array<f32>;
var<storage> matrix_b: array<f32>;
var<storage, read_write> matrix_c: array<f32>;
var<uniform> params: Params;
var<workgroup> shared_a: array<f32, 256>;
var<workgroup> shared_b: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let lx = lid.x;
    let ly = lid.y;
    let tile_col = wgid.x * 16u;
    let tile_row = wgid.y * 16u;

    var sum = 0.0;
    var t = 0u;
    loop {
        if t >= params.k { break; }

        // Load A tile (transposed): sA[ly*16+lx] = A[(t+ly)*M + (tile_row+lx)]
        let a_row = t + ly;
        let a_col = tile_row + lx;
        let a_in = (a_row < params.k) && (a_col < params.m);
        shared_a[ly * 16u + lx] = select(0.0, matrix_a[a_row * params.m + a_col], a_in);

        // Load B tile: sB[ly*16+lx] = B[(t+ly)*N + (tile_col+lx)]
        let b_row = t + ly;
        let b_col = tile_col + lx;
        let b_in = (b_row < params.k) && (b_col < params.n);
        shared_b[ly * 16u + lx] = select(0.0, matrix_b[b_row * params.n + b_col], b_in);

        workgroupBarrier();

        // Inner product: sum += sA[j][ly] * sB[j][lx]
        for (var j = 0u; j < 16u; j++) {
            sum += shared_a[j * 16u + ly] * shared_b[j * 16u + lx];
        }

        workgroupBarrier();
        t += 16u;
    }

    // Store result
    let row = tile_row + ly;
    let col = tile_col + lx;
    if row < params.m && col < params.n {
        matrix_c[row * params.n + col] = sum;
    }
}
