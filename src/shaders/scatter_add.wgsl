struct Params {
    total: u32,
    seq_len: u32,
    embed_dim: u32,
    _pad: u32,
}

var<storage> indices: array<u32>;
var<storage> src: array<f32>;
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;

// For each output element (out_row, col), sum src rows where indices[s] == out_row.
// Dispatch: [ceil(vocab*embed/256), 1, 1]
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.total { return; }

    let out_row = i / params.embed_dim;
    let col = i % params.embed_dim;

    var sum = 0.0;
    for (var s = 0u; s < params.seq_len; s++) {
        if indices[s] == out_row {
            sum += src[s * params.embed_dim + col];
        }
    }
    dst[i] = sum;
}
