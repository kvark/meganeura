// Cross-entropy loss with fused softmax gradient.
//
// For each batch item:
//   1. Parallel max reduction over features (numerical stability)
//   2. Parallel sum-exp reduction
//   3. Parallel gradient output + serial loss accumulation
//
// Dispatch: [batch, 1, 1], workgroup_size(256)

struct Params {
    batch: u32,
    features: u32,
    _pad0: u32,
    _pad1: u32,
}

var<storage> logits: array<f32>;
var<storage> labels: array<f32>;
var<storage, read_write> grad_out: array<f32>;
var<storage, read_write> loss_out: array<f32>;
var<uniform> params: Params;
var<workgroup> wg_buf: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    let b = wgid.x;
    let features = params.features;
    let offset = b * features;

    // === Pass 1: parallel max reduction ===
    var local_max = -3.402823e+38; // -FLT_MAX
    var j = tid;
    loop {
        if j >= features { break; }
        local_max = max(local_max, logits[offset + j]);
        j += 256u;
    }
    wg_buf[tid] = local_max;
    workgroupBarrier();

    // Tree reduction for max
    for (var s = 128u; s > 0u; s >>= 1u) {
        if tid < s {
            wg_buf[tid] = max(wg_buf[tid], wg_buf[tid + s]);
        }
        workgroupBarrier();
    }
    let max_val = wg_buf[0];
    workgroupBarrier();

    // === Pass 2: parallel sum-exp reduction ===
    var local_sum = 0.0;
    j = tid;
    loop {
        if j >= features { break; }
        local_sum += exp(logits[offset + j] - max_val);
        j += 256u;
    }
    wg_buf[tid] = local_sum;
    workgroupBarrier();

    // Tree reduction for sum
    for (var s = 128u; s > 0u; s >>= 1u) {
        if tid < s {
            wg_buf[tid] += wg_buf[tid + s];
        }
        workgroupBarrier();
    }
    let log_sum_exp = log(wg_buf[0]) + max_val;
    workgroupBarrier();

    // === Pass 3: parallel gradient + partial loss ===
    let inv_batch = 1.0 / f32(params.batch);
    var local_loss = 0.0;
    j = tid;
    loop {
        if j >= features { break; }
        let log_softmax = logits[offset + j] - log_sum_exp;
        let softmax = exp(log_softmax);
        local_loss -= labels[offset + j] * log_softmax;
        grad_out[offset + j] = (softmax - labels[offset + j]) * inv_batch;
        j += 256u;
    }

    // Reduce loss across threads
    wg_buf[tid] = local_loss;
    workgroupBarrier();
    for (var s = 128u; s > 0u; s >>= 1u) {
        if tid < s {
            wg_buf[tid] += wg_buf[tid + s];
        }
        workgroupBarrier();
    }

    // Thread 0 atomically accumulates the batch loss
    if tid == 0u {
        // Use atomicAdd for loss accumulation across batch items.
        // Since WGSL doesn't have atomicAdd for f32 storage, we accumulate
        // via a simple store-add pattern (only one WG writes per batch item).
        // We initialize loss_out[0] = 0 before dispatch and accumulate here.
        // This is safe because each workgroup handles a different batch item.
        loss_out[b] = wg_buf[0] * inv_batch;
    }
}
