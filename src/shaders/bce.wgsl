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

// Binary cross-entropy: -mean(t*log(p) + (1-t)*log(1-p))
// Gradient wrt pred:  (p - t) / (p * (1-p) * N)
@compute @workgroup_size(1)
fn main() {
    let eps = 1e-7;  // clamp to avoid log(0)
    var total_loss = 0.0;
    let n = f32(params.len);
    for (var i = 0u; i < params.len; i++) {
        let p = clamp(pred[i], eps, 1.0 - eps);
        let t = labels[i];
        total_loss -= t * log(p) + (1.0 - t) * log(1.0 - p);
        // Gradient: (p - t) / (p * (1-p)) / N
        grad_out[i] = (p - t) / (p * (1.0 - p) * n);
    }
    loss_out[0] = total_loss / n;
}
