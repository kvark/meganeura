// Temporal gradient accumulation: acc[i] += grad[i] * scale.
//
// backward overwrites each param's grad buffer every step(); this pass
// adds that fresh grad into a persistent accumulator so gradients sum
// ACROSS step() calls (the PyTorch `.grad +=` semantics meganeura's
// static-graph backward lacks). `scale` is 1/micro_batches so the
// accumulator holds the mean gradient. Cleared by `zero_grad()`.

struct Params {
    len: u32,
    scale: f32,
    _pad0: u32,
    _pad1: u32,
}

var<storage> grad: array<f32>;
var<storage, read_write> acc: array<f32>;
var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if i >= params.len { return; }
    acc[i] = acc[i] + grad[i] * params.scale;
}
