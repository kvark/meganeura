// Cross-entropy loss with fused softmax gradient.
//
// For each batch item:
//   1. One pass over logits and labels: online max / sum-exp (the running
//      sum is rescaled whenever the running max grows) and Σ(labels),
//      combined across lanes in a single tree
//   2. One pass writing the gradient and accumulating the loss
//
// Forward:  L = -Σ labels · log_softmax(logits)
// Gradient: ∂L/∂logits_j = softmax_j · S − labels_j,  where S = Σ_i labels_i.
//
// Most users feed labels that sum to 1 (a probability distribution), for which
// S = 1 and the gradient reduces to the familiar `softmax − labels`. But users
// that feed per-class *weights* that do NOT sum to 1 (e.g. advantage-scaled
// one-hot policy-gradient targets) get the correct general derivative here
// rather than a silent S=1 assumption that produces anti-learning when
// |S| < softmax_j (the small-weight regime).
//
// Dispatch: [batch, 1, 1], workgroup_size(256)

struct Params {
    batch: u32,
    features: u32,
    write_grad: u32,
    _pad1: u32,
}

var<storage> logits: array<f32>;
var<storage> labels: array<f32>;
var<storage, read_write> grad_out: array<f32>;
var<storage, read_write> loss_out: array<f32>;
var<uniform> params: Params;
var<workgroup> wg_buf: array<f32, 256>;
var<workgroup> wg_sum: array<f32, 256>;
var<workgroup> wg_labels: array<f32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_index) tid: u32,
) {
    let b = wgid.x;
    let features = params.features;
    let offset = b * features;

    // === Pass 1: online max / sum-exp and Σ(labels) ===
    // Starting from -FLT_MAX rather than -inf keeps exp(x - max) defined
    // when a lane sees no finite logit.
    var local_max = -3.402823e+38;
    var local_sum = 0.0;
    var local_label_sum = 0.0;
    var j = tid;
    loop {
        if j >= features { break; }
        let x = logits[offset + j];
        if x > local_max {
            local_sum = local_sum * exp(local_max - x) + 1.0;
            local_max = x;
        } else {
            local_sum += exp(x - local_max);
        }
        local_label_sum += labels[offset + j];
        j += 256u;
    }
    wg_buf[tid] = local_max;
    wg_sum[tid] = local_sum;
    wg_labels[tid] = local_label_sum;
    workgroupBarrier();

    for (var s = 128u; s > 0u; s >>= 1u) {
        if tid < s {
            let max_a = wg_buf[tid];
            let max_b = wg_buf[tid + s];
            let merged = max(max_a, max_b);
            wg_sum[tid] = wg_sum[tid] * exp(max_a - merged) + wg_sum[tid + s] * exp(max_b - merged);
            wg_buf[tid] = merged;
            wg_labels[tid] += wg_labels[tid + s];
        }
        workgroupBarrier();
    }
    let log_sum_exp = log(wg_sum[0]) + wg_buf[0];
    let label_sum = wg_labels[0];
    workgroupBarrier();

    // === Pass 2: parallel gradient + partial loss ===
    let inv_batch = 1.0 / f32(params.batch);
    var local_loss = 0.0;
    j = tid;
    loop {
        if j >= features { break; }
        let log_softmax = logits[offset + j] - log_sum_exp;
        let softmax = exp(log_softmax);
        local_loss -= labels[offset + j] * log_softmax;
        if params.write_grad != 0u {
            grad_out[offset + j] = (softmax * label_sum - labels[offset + j]) * inv_batch;
        }
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
