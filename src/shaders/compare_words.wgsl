struct Params { len: u32, pad0: u32, pad1: u32, pad2: u32 }
var<storage, read> src_a: array<u32>;
var<storage, read> src_b: array<u32>;
var<storage, read_write> dst: atomic<u32>;
var<uniform> params: Params;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) id: vec3<u32>,
        @builtin(num_workgroups) grid: vec3<u32>) {
    var different = 0u;
    let stride = grid.x * 256u;
    for (var i = id.x; i < params.len; ) {
        different |= select(0u, 1u, src_a[i] != src_b[i]);
        if params.len - i <= stride {
            break;
        }
        i += stride;
    }
    if different != 0u {
        atomicOr(&dst, different);
    }
}
