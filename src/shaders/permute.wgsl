// Permute the axes of a tensor of rank at most 4.
// Output index (o0, o1, o2, o3) over dims (d0, d1, d2, d3) reads input
// element o0*s0 + o1*s1 + o2*s2 + o3*s3, where s are the input strides of
// the axes each output axis takes. Lower ranks pad leading dims with 1.
// Dispatch: ceil(total / 256) workgroups over X and Y, each axis within the
// portable limit; the index is gid.x + gid.y · (workgroups along X · 256).
// A zero stride broadcasts that axis.

struct Params {
    total: u32,
    d1: u32,
    d2: u32,
    d3: u32,
    s0: u32,
    s1: u32,
    s2: u32,
    s3: u32,
}

var<storage> src: array<f32>;
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let i = gid.x + gid.y * groups.x * 256u;
    if i >= params.total { return; }
    let o3 = i % params.d3;
    var r = i / params.d3;
    let o2 = r % params.d2;
    r = r / params.d2;
    let o1 = r % params.d1;
    let o0 = r / params.d1;
    dst[i] = src[o0 * params.s0 + o1 * params.s1 + o2 * params.s2 + o3 * params.s3];
}
