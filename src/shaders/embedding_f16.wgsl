enable f16;
// Embedding gather from an f16 table → f32 output. Same as embedding.wgsl
// but the source table is f16, halving the bytes read per gathered element
// — the lever for scatter-gather-bandwidth-bound lookups (SH coefficients).
struct Params {
    seq: u32,
    hidden: u32,
    rows: u32, // table rows
    _pad: u32,
}

var<storage> indices: array<u32>;
var<storage> src: array<f16>;
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let total = params.seq * params.hidden;
    if i >= total { return; }
    let row = i / params.hidden;
    let col = i % params.hidden;
    // An id past the table reads its last row rather than past its end.
    let token_id = min(indices[row], params.rows - 1u);
    dst[i] = f32(src[token_id * params.hidden + col]);
}
