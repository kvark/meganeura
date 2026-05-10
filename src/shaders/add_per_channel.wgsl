// Per-channel broadcast add.
//
// Given activations `[N, C, H, W]` and a per-channel bias `[C]`,
// produce `dst[n, c, h, w] = src[n, c, h, w] + bias[c]`.
//
// Used to fuse a BatchNorm bias term into the post-conv activation
// path: the safetensors store `bn.fused_bias` as a per-channel
// vector (shape `[C]`), and this op broadcasts it across every
// (batch, spatial) position without requiring callers to
// pre-replicate the data into a `[N*C*H*W]`-sized buffer.
//
// Linear-index encoding: dst[i] = src[i] + bias[(i / spatial) % channels],
// where `spatial = H * W`.  The double-mod avoids per-batch replication of
// the bias tensor (compare `mul_per_channel`, which uses a
// `[N*C]`-shaped gate and is keyed by `i / spatial` directly).

struct Params {
    len: u32,        // total elements N * C * H * W
    spatial: u32,    // H * W
    channels: u32,   // C
    _pad0: u32,
}

var<storage> src: array<f32>;
var<storage> bias: array<f32>;
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.len { return; }
    dst[i] = src[i] + bias[(i / params.spatial) % params.channels];
}
