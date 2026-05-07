// Per-channel broadcast multiply.
//
// Given activations `[N, C, H, W]` and a per-(batch, channel) gate
// `[N, C]`, produce `dst[n, c, h, w] = src[n, c, h, w] * gate[n, c]`.
//
// Used by EfficientNet's Squeeze-and-Excitation block: the gate is
// the output of GAP → fc1 → SiLU → fc2 → sigmoid (shape [N, C]),
// applied as a channel-wise modulation on the depthwise-conv output.
//
// Linear-index encoding: dst[i] = src[i] * gate[i / spatial], where
// `spatial = H * W` (the broadcast factor along the trailing axes).

struct Params {
    len: u32,        // total elements N * C * H * W
    spatial: u32,    // H * W
    _pad0: u32,
    _pad1: u32,
}

var<storage> src: array<f32>;
var<storage> gate: array<f32>;
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.len { return; }
    dst[i] = src[i] * gate[i / params.spatial];
}
