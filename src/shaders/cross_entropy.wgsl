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

@compute @workgroup_size(1)
fn main() {
    var total_loss = 0.0;
    for (var b = 0u; b < params.batch; b++) {
        let offset = b * params.features;

        // Find max for numerical stability
        var max_val = logits[offset];
        for (var j = 1u; j < params.features; j++) {
            max_val = max(max_val, logits[offset + j]);
        }

        // Log-sum-exp
        var sum_exp = 0.0;
        for (var j = 0u; j < params.features; j++) {
            sum_exp += exp(logits[offset + j] - max_val);
        }
        let log_sum_exp = log(sum_exp) + max_val;

        // Loss and gradient
        for (var j = 0u; j < params.features; j++) {
            let log_softmax = logits[offset + j] - log_sum_exp;
            let softmax = exp(log_softmax);
            total_loss -= labels[offset + j] * log_softmax;
            grad_out[offset + j] = (softmax - labels[offset + j]) / f32(params.batch);
        }
    }
    loss_out[0] = total_loss / f32(params.batch);
}
