// Fused matmul + ReLU: C = relu(A @ B)
// A: [M, K], B: [K, N], C: [M, N]

struct Params {
    m: u32,
    k: u32,
    n: u32,
    _pad: u32,
}

var<storage, read> a: array<f32>;
var<storage, read> b: array<f32>;
var<storage, read_write> c: array<f32>;
var<uniform> params: Params;

const TILE: u32 = 16u;

var<workgroup> tile_a: array<f32, 256>;
var<workgroup> tile_b: array<f32, 256>;

@compute @workgroup_size(16, 16)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let row = gid.y;
    let col = gid.x;
    let local_row = lid.y;
    let local_col = lid.x;

    var sum = 0.0;
    let num_tiles = (params.k + TILE - 1u) / TILE;

    for (var t = 0u; t < num_tiles; t++) {
        let a_col = t * TILE + local_col;
        let b_row = t * TILE + local_row;

        if row < params.m && a_col < params.k {
            tile_a[local_row * TILE + local_col] = a[row * params.k + a_col];
        } else {
            tile_a[local_row * TILE + local_col] = 0.0;
        }

        if b_row < params.k && col < params.n {
            tile_b[local_row * TILE + local_col] = b[b_row * params.n + col];
        } else {
            tile_b[local_row * TILE + local_col] = 0.0;
        }

        workgroupBarrier();

        for (var i = 0u; i < TILE; i++) {
            sum += tile_a[local_row * TILE + i] * tile_b[i * TILE + local_col];
        }

        workgroupBarrier();
    }

    if row < params.m && col < params.n {
        c[row * params.n + col] = max(sum, 0.0);
    }
}
