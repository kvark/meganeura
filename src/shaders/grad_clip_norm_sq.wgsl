// Sum-of-squares of a single gradient buffer, atomically added to a
// shared 1-element accumulator. The accumulator stores f32 bits in a
// u32 atomic; we accumulate via compareExchangeWeak in a CAS loop.
//
// Dispatched once per gradient buffer (one workgroup of 256 threads).
// Strided per-thread accumulation followed by tree reduction in
// shared memory; the workgroup root performs the atomic add.
//
// Pair with `grad_clip_zero` (pre-pass, once per step) and
// `grad_clip_scale` (post-pass, one dispatch per gradient buffer).

struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

var<storage> grad: array<f32>;
var<storage, read_write> acc: array<atomic<u32>>;
var<uniform> params: Params;

var<workgroup> wg_data: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    var s: f32 = 0.0;
    var idx = tid;
    loop {
        if idx >= params.len { break; }
        let g = grad[idx];
        s = s + g * g;
        idx = idx + 256u;
    }
    wg_data[tid] = s;
    workgroupBarrier();

    var stride = 128u;
    loop {
        if stride == 0u { break; }
        if tid < stride {
            wg_data[tid] = wg_data[tid] + wg_data[tid + stride];
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    if tid == 0u {
        let delta = wg_data[0];
        if delta != 0.0 {
            // f32-via-u32 atomicAdd: load current value, compute new
            // value, CAS. Loop until we win the race.
            loop {
                let old_u = atomicLoad(&acc[0]);
                let old_f = bitcast<f32>(old_u);
                let new_f = old_f + delta;
                let new_u = bitcast<u32>(new_f);
                let r = atomicCompareExchangeWeak(&acc[0], old_u, new_u);
                if r.exchanged { break; }
            }
        }
    }
}
