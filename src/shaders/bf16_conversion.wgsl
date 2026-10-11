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

fn round_bf16(value: f32) -> bf16 {
    let bits = bitcast<u32>(value);
    var rounded = bits + 0x7fffu + ((bits >> 16u) & 1u);
    if (bits & 0x7fffffffu) > 0x7f800000u {
        rounded = bits | 0x00400000u;
    }
    return bf16(bitcast<f32>(rounded & 0xffff0000u));
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x < params.len { dst[id.x] = $CONVERT(src[id.x]); }
}
