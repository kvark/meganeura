// MatMulBT: C = A @ B^T where A=[M,K], B=[N,K], C=[M,N]
// B is accessed transposed: sB[ly][lx] = B[(tile_col+ly)*K + (t+lx)]

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

        // Load A tile: sA[ly*16+lx] = A[(tile_row+ly)*K + (t+lx)]
        let a_row = tile_row + ly;
        let a_col = t + lx;
        let a_in = (a_row < params.m) && (a_col < params.k);
        shared_a[ly * 16u + lx] = select(0.0, matrix_a[a_row * params.k + a_col], a_in);

        // Load B tile (transposed): sB[ly*16+lx] = B[(tile_col+ly)*K + (t+lx)]
        let b_row = tile_col + ly;
        let b_col = t + lx;
        let b_in = (b_row < params.n) && (b_col < params.k);
        shared_b[ly * 16u + lx] = select(0.0, matrix_b[b_row * params.k + b_col], b_in);

        workgroupBarrier();

        // Inner product: sum += sA[ly][j] * sB[lx][j]
        for (var j = 0u; j < 16u; j++) {
            sum += shared_a[ly * 16u + j] * shared_b[lx * 16u + j];
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
