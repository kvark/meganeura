// Per-channel broadcast add.
//
// Given activations `[N, C, H, W]` and a per-channel bias `[C]`,
// produce `dst[n, c, h, w] = src[n, c, h, w] + bias[c]`.
//
// Used by EfficientNet-style BN-fusion: the per-channel bias produced
// by `bias[c] - mean[c] * scale[c] / sqrt(var[c] + eps)` is added once
// per (n, c) tile of the spatial activation. Avoids materialising the
// per-spatial-position broadcast that the resnet pattern uses.
//
// Linear-index encoding: dst[i] = src[i] + bias[(i / spatial) % channels],
// where `spatial = H * W`.

struct Params {
    len: u32,        // total elements N * C * H * W
    spatial: u32,    // H * W
    channels: u32,
    _pad: u32,
}

var<storage> src: array<f32>;
var<storage> bias: array<f32>;
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.len { return; }
    let c = (i / params.spatial) % params.channels;
    dst[i] = src[i] + bias[c];
}
