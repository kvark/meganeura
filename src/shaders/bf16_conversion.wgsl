enable wgpu_bfloat16;

struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}
var<storage> src: array<$SRC_TYPE>;
var<storage, read_write> dst: array<$DST_TYPE>;
var<uniform> params: Params;

$ROUND

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x < params.len { dst[id.x] = $CONVERT(src[id.x]); }
}
