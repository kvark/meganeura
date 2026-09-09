// Select row valid_len-1 from a padded [rows, cols] tensor.

struct Params {
    cols: u32,
    rows: u32,
    _pad0: u32,
    _pad1: u32,
}

var<storage> src: array<f32>;
var<storage> valid_len_buf: array<u32>;
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    if col >= params.cols { return; }
    let valid_len = clamp(valid_len_buf[0], 1u, params.rows);
    dst[col] = src[(valid_len - 1u) * params.cols + col];
}
