// Flash-decoding split-K combine: merges the per-split online-softmax
// partials of `cached_block_attention_split.wgsl` into the attention
// output. One workgroup per (row, head), 64 threads over head_dim, the
// same shape the fused single kernel dispatches.
//
//   m = max_i m_i
//   l = sum_i exp(m_i - m) * l_i
//   out[d] = sum_i exp(m_i - m) * acc_i[d] / l
//
// Empty splits carry m = -1e30 and l = 0; their weight is exp(-1e30 - m)
// = 0 and they contribute nothing, so any mix of live and empty splits
// reduces exactly to the single-workgroup result.

struct Params {
    window_size: u32,
    num_heads: u32,
    num_kv_heads: u32,
    head_dim: u32,
    block_len: u32,
    max_seq: u32,
    splits: u32,
    chunk: u32,
}

var<storage> partials: array<f32>;
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;

const MAX_VALUES_PER_THREAD: u32 = 8u;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let query_row = wgid.x;
    let head = wgid.y;
    let tid = lid.x;
    if query_row >= params.block_len || head >= params.num_heads { return; }

    let stride = params.head_dim + 2u;
    let row_part = ((query_row * params.num_heads + head) * params.splits)
        * (params.head_dim + 2u);

    // Global max over the splits.
    var m = -1e30;
    for (var i = 0u; i < params.splits; i++) {
        let part = (row_part + i) * stride;
        m = max(m, partials[part + params.head_dim]);
    }

    // Global sum of exp and combined accumulator.
    var l = 0.0;
    for (var i = 0u; i < params.splits; i++) {
        let part = (row_part + i) * stride;
        let w = exp(partials[part + params.head_dim] - m);
        l += w * partials[part + params.head_dim + 1u];
    }

    var my_out: array<f32, 8>;
    for (var lane = 0u; lane < MAX_VALUES_PER_THREAD; lane++) {
        my_out[lane] = 0.0;
    }
    for (var i = 0u; i < params.splits; i++) {
        let part = (row_part + i) * stride;
        let w = exp(partials[part + params.head_dim] - m);
        for (var lane = 0u; lane < MAX_VALUES_PER_THREAD; lane++) {
            let d = lane * 64u + tid;
            if d < params.head_dim {
                my_out[lane] += w * partials[part + d];
            }
        }
    }

    let safe_sum = select(l, 1.0, l == 0.0);
    let dst_base = query_row * params.num_heads * params.head_dim + head * params.head_dim;
    for (var lane = 0u; lane < MAX_VALUES_PER_THREAD; lane++) {
        let d = lane * 64u + tid;
        if d < params.head_dim {
            dst[dst_base + d] = my_out[lane] / safe_sum;
        }
    }
}
