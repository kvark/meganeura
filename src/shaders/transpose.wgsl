struct Params {
    m: u32,
    n: u32,
    _pad0: u32,
    _pad1: u32,
}

var<storage> src: array<f32>;
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;

// One 16x16 tile per workgroup, staged through shared memory so that both
// the read of `src` rows and the write of `dst` rows are contiguous across
// neighbouring lanes. Writing `dst[col * m + row]` directly strides every
// lane by `m`. The extra column keeps the transposed read off a single bank.
var<workgroup> tile: array<f32, 272>; // 16 x 17

@compute @workgroup_size(16, 16)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    // Read src[row, col] for this tile.
    let col = wgid.x * 16u + lid.x;
    let row = wgid.y * 16u + lid.y;
    if row < params.m && col < params.n {
        tile[lid.y * 17u + lid.x] = src[row * params.n + col];
    }
    workgroupBarrier();

    // dst is [n, m]: its rows are src columns. Lanes along x now walk a dst
    // row, i.e. consecutive src rows of one src column.
    let dst_row = wgid.x * 16u + lid.y;
    let dst_col = wgid.y * 16u + lid.x;
    if dst_row < params.n && dst_col < params.m {
        dst[dst_row * params.m + dst_col] = tile[lid.x * 17u + lid.y];
    }
}
