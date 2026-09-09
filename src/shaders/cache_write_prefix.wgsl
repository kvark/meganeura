// Write a runtime-valid prefix of [block_len, dim] into a KV cache at a
// dynamic row offset.

struct Params {
    dim: u32,
    block_len: u32,
    max_seq: u32,
    _pad: u32,
}

var<storage> src: array<f32>;
var<storage, read_write> dst: array<f32>;
var<storage> kv_pos_buf: array<u32>;
var<storage> valid_len_buf: array<u32>;
var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let linear = gid.x;
    let total = params.block_len * params.dim;
    if linear >= total { return; }
    let row = linear / params.dim;
    let col = linear - row * params.dim;
    let valid_len = min(valid_len_buf[0], params.block_len);
    let position = kv_pos_buf[0] + row;
    if row >= valid_len || position >= params.max_seq { return; }
    dst[position * params.dim + col] = src[linear];
}
