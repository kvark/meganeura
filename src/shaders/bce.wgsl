struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

var<storage> pred: array<f32>;
var<storage> labels: array<f32>;
var<storage, read_write> grad_out: array<f32>;
var<storage, read_write> loss_out: array<f32>;
var<uniform> params: Params;
var<workgroup> wg_buf: array<f32, 256>;

// Binary cross-entropy: -mean(t*log(p) + (1-t)*log(1-p))
// Gradient wrt pred:  (p - t) / (p * (1-p) * N)
// Dispatch: [ceil(len/256), 1, 1], workgroup_size(256)
@compute @workgroup_size(256)
fn main(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
    @builtin(workgroup_id) wgid: vec3<u32>,
) {
    let eps = 1e-7;
    let n = f32(params.len);
    let i = gid.x;

    // Each thread computes gradient and partial loss for one element
    var local_loss = 0.0;
    if i < params.len {
        let p = clamp(pred[i], eps, 1.0 - eps);
        let t = labels[i];
        local_loss = -(t * log(p) + (1.0 - t) * log(1.0 - p));
        grad_out[i] = (p - t) / (p * (1.0 - p) * n);
    }

    // Tree reduction for loss
    wg_buf[tid] = local_loss;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if tid < s {
            wg_buf[tid] += wg_buf[tid + s];
        }
        workgroupBarrier();
    }
    if tid == 0u {
        loss_out[wgid.x] = wg_buf[0] / n;
    }
}
