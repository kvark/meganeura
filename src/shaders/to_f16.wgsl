enable f16;
// Cast f32 → f16, elementwise. Backward is the identity (mixed-precision
// straight-through): the f32 master weights stay the trainable parameter;
// this produces the f16 copy a bandwidth-bound forward read (embedding_f16)
// consumes.
struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

var<storage> src: array<f32>;
var<storage, read_write> dst: array<f16>;
var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.len { return; }
    dst[i] = f16(src[i]);
}
