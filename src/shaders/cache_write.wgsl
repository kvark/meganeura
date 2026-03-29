// cache_write: write [1, dim] into row kv_pos of [max_seq, dim] cache buffer.
// Dispatch: [ceil(dim/256), 1, 1]

struct Params {
    dim: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

var<storage> src: array<f32>;             // new K or V: [1, dim]
var<storage, read_write> dst: array<f32>; // cache: [max_seq, dim]
var<storage> kv_pos_buf: array<u32>;      // [1] — current write position
var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = gid.x;
    if j >= params.dim { return; }
    let kv_pos = kv_pos_buf[0];
    dst[kv_pos * params.dim + j] = src[j];
}
