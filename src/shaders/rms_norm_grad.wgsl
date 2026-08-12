// Two entry points:
//   rms_norm_grad_w: gradient wrt weight, dispatch [ceil(cols/256), 1, 1]
//   rms_norm_grad_x: gradient wrt input, dispatch [rows, 1, 1]
// Params field names: m=rows, n=cols, k=eps_bits

struct Params {
    m: u32,
    n: u32,
    k: u32,
    _pad: u32,
}

var<storage> src_a: array<f32>;  // dy
var<storage> src_b: array<f32>;  // x
var<storage> bias: array<f32>;   // w (weight)
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;
var<workgroup> wg_data: array<f32, 256>;

// grad_w[col] = sum_i( dy[i*n+col] * x[i*n+col] * rsqrt_i )
//
// Dispatch: [ceil(cols/256), 1, 1]
// Each thread handles one column. Workgroup cooperatively precomputes
// rsqrt for each row via shared-memory reduction (O(rows × cols/256) per
// thread), then accumulates grad_w in O(rows) per thread.
// Total work: O(rows × cols) — down from the naive O(rows × cols²).
@compute @workgroup_size(256)
fn rms_norm_grad_w(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let col = wgid.x * 256u + lid.x;
    let tid = lid.x;
    let rows = params.m;
    let cols = params.n;
    let eps = bitcast<f32>(params.k);

    // Process one row at a time: cooperative rsqrt then per-column accumulation
    var acc = 0.0;
    for (var i = 0u; i < rows; i++) {
        let offset = i * cols;

        // Cooperative rsqrt: each thread sums a stride of the row
        var ss = 0.0;
        var j = tid;
        loop {
            if j >= cols { break; }
            let v = src_b[offset + j];
            ss += v * v;
            j += 256u;
        }
        wg_data[tid] = ss;
        workgroupBarrier();

        var stride = 128u;
        loop {
            if stride == 0u { break; }
            if tid < stride {
                wg_data[tid] += wg_data[tid + stride];
            }
            workgroupBarrier();
            stride >>= 1u;
        }
        let rsqrt_i = inverseSqrt(wg_data[0] / f32(cols) + eps);

        // Accumulate this row's contribution
        if col < cols {
            acc += src_a[offset + col] * src_b[offset + col] * rsqrt_i;
        }
        workgroupBarrier();
    }

    if col < cols {
        dst[col] = acc;
    }
}

// grad_x[i,j] = rsqrt_i * (dy[i,j]*w[j] - x[i,j] * s_i)
// where s_i = (rsqrt_i^2 / cols) * sum_j(dy[i,j]*w[j]*x[i,j])
@compute @workgroup_size(256)
fn rms_norm_grad_x(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let rows = params.m;
    let cols = params.n;
    let eps = bitcast<f32>(params.k);

    // For narrow rows, params._pad is the power-of-two lane count assigned to
    // each row. Several independent reductions share one workgroup while
    // retaining the same tree order as the full 256-lane kernel.
    if params._pad != 0u {
        let lanes_per_row = params._pad;
        let rows_per_workgroup = 256u / lanes_per_row;
        let row_slot = tid / lanes_per_row;
        let lane = tid % lanes_per_row;
        let packed_row = wgid.x * rows_per_workgroup + row_slot;
        let row_valid = packed_row < rows;
        let packed_offset = packed_row * cols;

        var packed_ss = 0.0;
        if row_valid && lane < cols {
            let x = src_b[packed_offset + lane];
            packed_ss = x * x;
        }
        wg_data[tid] = packed_ss;
        workgroupBarrier();

        var packed_stride = lanes_per_row >> 1u;
        loop {
            if packed_stride == 0u { break; }
            if lane < packed_stride {
                wg_data[tid] += wg_data[tid + packed_stride];
            }
            workgroupBarrier();
            packed_stride >>= 1u;
        }
        let packed_base = row_slot * lanes_per_row;
        let packed_rsqrt = inverseSqrt(wg_data[packed_base] / f32(cols) + eps);
        workgroupBarrier();

        var packed_dot = 0.0;
        if row_valid && lane < cols {
            let index = packed_offset + lane;
            packed_dot = src_a[index] * bias[lane] * src_b[index];
        }
        wg_data[tid] = packed_dot;
        workgroupBarrier();

        packed_stride = lanes_per_row >> 1u;
        loop {
            if packed_stride == 0u { break; }
            if lane < packed_stride {
                wg_data[tid] += wg_data[tid + packed_stride];
            }
            workgroupBarrier();
            packed_stride >>= 1u;
        }
        let packed_s =
            (packed_rsqrt * packed_rsqrt / f32(cols)) * wg_data[packed_base];
        if row_valid && lane < cols {
            let index = packed_offset + lane;
            dst[index] = packed_rsqrt *
                (src_a[index] * bias[lane] - src_b[index] * packed_s);
        }
        return;
    }

    let row = wgid.x;
    if row >= rows { return; }
    let offset = row * cols;

    // Phase 1: Compute rsqrt via shared memory reduction
    var ss = 0.0;
    var j = tid;
    loop {
        if j >= cols { break; }
        let v = src_b[offset + j];
        ss += v * v;
        j += 256u;
    }
    wg_data[tid] = ss;
    workgroupBarrier();

    var stride = 128u;
    loop {
        if stride == 0u { break; }
        if tid < stride {
            wg_data[tid] += wg_data[tid + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }
    let rsqrt_val = inverseSqrt(wg_data[0] / f32(cols) + eps);
    // Every lane must capture the first reduction before wg_data is reused
    // for the dot-product reduction below.
    workgroupBarrier();

    // Phase 2: Compute s_i = (rsqrt^2 / cols) * sum_j(dy[j]*w[j]*x[j])
    var dot = 0.0;
    j = tid;
    loop {
        if j >= cols { break; }
        dot += src_a[offset + j] * bias[j] * src_b[offset + j];
        j += 256u;
    }
    wg_data[tid] = dot;
    workgroupBarrier();

    stride = 128u;
    loop {
        if stride == 0u { break; }
        if tid < stride {
            wg_data[tid] += wg_data[tid + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }
    let s_i = (rsqrt_val * rsqrt_val / f32(cols)) * wg_data[0];

    // Phase 3: Write output
    j = tid;
    loop {
        if j >= cols { break; }
        let dy = src_a[offset + j];
        let w = bias[j];
        let x = src_b[offset + j];
        dst[offset + j] = rsqrt_val * (dy * w - x * s_i);
        j += 256u;
    }
}
