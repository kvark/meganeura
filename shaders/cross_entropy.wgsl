// Cross-entropy loss: L = -mean(sum(labels * log(softmax(logits))))
// Combined softmax + cross-entropy for numerical stability.
// Writes gradient (softmax - labels) into grad_out buffer.

struct Params {
    batch: u32,
    features: u32,
    _pad0: u32,
    _pad1: u32,
}

var<storage, read> logits: array<f32>;
var<storage, read> labels: array<f32>;
var<storage, read_write> grad_out: array<f32>;
var<storage, read_write> loss_out: array<f32>;
var<uniform> params: Params;

@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch = params.batch;
    let features = params.features;
    var total_loss = 0.0;

    for (var b = 0u; b < batch; b++) {
        let offset = b * features;

        // Find max
        var max_val = logits[offset];
        for (var j = 1u; j < features; j++) {
            max_val = max(max_val, logits[offset + j]);
        }

        // Log-sum-exp
        var sum_exp = 0.0;
        for (var j = 0u; j < features; j++) {
            sum_exp += exp(logits[offset + j] - max_val);
        }
        let log_sum_exp = log(sum_exp) + max_val;

        // Loss and gradient for this sample
        for (var j = 0u; j < features; j++) {
            let log_softmax = logits[offset + j] - log_sum_exp;
            let softmax_val = exp(log_softmax);
            total_loss -= labels[offset + j] * log_softmax;
            grad_out[offset + j] = softmax_val - labels[offset + j];
        }
    }

    loss_out[0] = total_loss / f32(batch);
}
