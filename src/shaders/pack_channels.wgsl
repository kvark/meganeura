$EXTENSIONS
struct Params { outer: u32, channels: u32, spatial: u32, padded: u32 }
var<storage> src: array<$TYPE>;
var<storage, read_write> dst: array<vec2<$TYPE>>;
var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global: vec3<u32>) {
    let pair = global.x;
    if pair >= params.outer * params.spatial * (params.padded / 2u) { return; }
    let channel = (pair % (params.padded / 2u)) * 2u;
    let row = pair / (params.padded / 2u);
    let spatial = row % params.spatial;
    let outer = row / params.spatial;
    var a = $TYPE(0.0f);
    var b = $TYPE(0.0f);
    if channel < params.channels { a = src[$INDEX_A]; }
    if channel + 1u < params.channels { b = src[$INDEX_B]; }
    dst[pair] = vec2<$TYPE>(a, b);
}
