// Reduction operations: sum, mean over entire tensor.
// Uses workgroup parallel reduction.

struct Params {
    len: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;
@group(0) @binding(2) var<uniform> params: Params;

const WG_SIZE: u32 = 256u;
var<workgroup> shared: array<f32, 256>;

@compute @workgroup_size(256)
fn sum_all(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = gid.x;
    let local_id = lid.x;

    // Load
    if i < params.len {
        shared[local_id] = input[i];
    } else {
        shared[local_id] = 0.0;
    }
    workgroupBarrier();

    // Tree reduction
    for (var stride = WG_SIZE / 2u; stride > 0u; stride /= 2u) {
        if local_id < stride {
            shared[local_id] += shared[local_id + stride];
        }
        workgroupBarrier();
    }

    // First thread writes partial sum
    // For full reduction, the host dispatches multiple passes
    // or uses atomics. For now, single workgroup.
    if local_id == 0u {
        output[gid.x / WG_SIZE] = shared[0];
    }
}

@compute @workgroup_size(256)
fn mean_all(
    @builtin(global_invocation_id) gid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = gid.x;
    let local_id = lid.x;

    if i < params.len {
        shared[local_id] = input[i];
    } else {
        shared[local_id] = 0.0;
    }
    workgroupBarrier();

    for (var stride = WG_SIZE / 2u; stride > 0u; stride /= 2u) {
        if local_id < stride {
            shared[local_id] += shared[local_id + stride];
        }
        workgroupBarrier();
    }

    if local_id == 0u {
        output[gid.x / WG_SIZE] = shared[0] / f32(params.len);
    }
}

// Softmax: exp(x_i - max(x)) / sum(exp(x_j - max(x)))
// Operates per-row on a [batch, features] tensor.

struct SoftmaxParams {
    batch: u32,
    features: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(3) var<uniform> softmax_params: SoftmaxParams;

@compute @workgroup_size(256)
fn softmax(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if row >= softmax_params.batch { return; }

    let offset = row * softmax_params.features;
    let features = softmax_params.features;

    // Find max for numerical stability
    var max_val = input[offset];
    for (var j = 1u; j < features; j++) {
        max_val = max(max_val, input[offset + j]);
    }

    // Compute exp and sum
    var sum_exp = 0.0;
    for (var j = 0u; j < features; j++) {
        let e = exp(input[offset + j] - max_val);
        output[offset + j] = e;
        sum_exp += e;
    }

    // Normalize
    for (var j = 0u; j < features; j++) {
        output[offset + j] /= sum_exp;
    }
}

// Cross-entropy loss: L = -mean(sum(labels * log(softmax(logits))))
// Combined softmax + cross-entropy for numerical stability.
// Outputs scalar loss and gradient (softmax - labels) in output buffer.

@group(1) @binding(0) var<storage, read> labels: array<f32>;
@group(1) @binding(1) var<storage, read_write> loss_out: array<f32>;

@compute @workgroup_size(1)
fn cross_entropy_loss(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch = softmax_params.batch;
    let features = softmax_params.features;
    var total_loss = 0.0;

    for (var b = 0u; b < batch; b++) {
        let offset = b * features;

        // Find max
        var max_val = input[offset];
        for (var j = 1u; j < features; j++) {
            max_val = max(max_val, input[offset + j]);
        }

        // Log-sum-exp
        var sum_exp = 0.0;
        for (var j = 0u; j < features; j++) {
            sum_exp += exp(input[offset + j] - max_val);
        }
        let log_sum_exp = log(sum_exp) + max_val;

        // Loss and gradient for this sample
        for (var j = 0u; j < features; j++) {
            let log_softmax = input[offset + j] - log_sum_exp;
            let softmax_val = exp(log_softmax);
            total_loss -= labels[offset + j] * log_softmax;
            // Store gradient: softmax - label
            output[offset + j] = softmax_val - labels[offset + j];
        }
    }

    loss_out[0] = total_loss / f32(batch);
}

// Transpose: B = A^T
// A: [M, N] → B: [N, M]

struct TransposeParams {
    m: u32,
    n: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(4) var<uniform> transpose_params: TransposeParams;

@compute @workgroup_size(16, 16)
fn transpose(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.y;
    let col = gid.x;
    if row >= transpose_params.m || col >= transpose_params.n { return; }
    output[col * transpose_params.m + row] = input[row * transpose_params.n + col];
}
