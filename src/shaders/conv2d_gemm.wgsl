// Conv2d forward via implicit GEMM: output = weight @ im2col(input)^T
//
// Computes C[Co, oH*oW] = A[Co, K] × B[K, oH*oW] per batch item,
// where K = Ci*kH*kW and B is the im2col matrix computed on-the-fly.
//
// Register-tiled matmul, workgroup [16,16,1]. BM, TM and K are generated.
//
// Dispatch: [ceil(oH*oW / 64), ceil(Co / 64), batch]

$DIVISOR

$PARAMS_TYPE

var<storage> src: array<f32>;              // input [N, Ci, H, W]
var<storage> weight: array<f32>;           // kernel [Co, Ci, kH, kW] = [Co, K]
var<storage, read_write> dst: array<f32>;  // output [N, Co, oH, oW]
$PARAMS_DECL
var<workgroup> shared_a: array<f32, $SHARED_A_SIZE>; // A tile: [BM, K], padded stride
var<workgroup> shared_b: array<f32, $SHARED_B_SIZE>; // B tile: [K, BM], padded stride

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tx = lid.x;
    let ty = lid.y;
    let n = wgid.z;           // batch index
    let tile_row = wgid.y * $BM_U;  // M (Co) tile start
    let tile_col = wgid.x * $BM_U;  // N (oH*oW) tile start
    let tid = ty * 16u + tx;

    let k_total = params.in_channels * params.kernel_h * params.kernel_w;
    let n_total = params.out_h * params.out_w;
    let m_total = params.out_channels;
    let input_stride = params.in_channels * params.in_h * params.in_w;  // per-batch input size
    let in_hw = params.in_h * params.in_w;

    // Every tile width divides 256, so a thread stages the same B column
    // (output pixel) in every slot and K stage: decompose it once. Its B rows
    // advance by 256/BM per slot, carried through (ci, kh, kw).
    let b_col = tid % $BM_U;
    let hw_idx = tile_col + b_col;
    let oh = divide_exact(hw_idx, params.out_w, params.column_width_multiplier);
    let ow = hw_idx - oh * params.out_w;
    let ih0 = i32(oh * params.stride) - i32(params.padding_h);
    let iw0 = i32(ow * params.stride) - i32(params.padding_w);
    let batch_src = n * input_stride;
    let b_step = split_digits(256u / $BM_U, params.kernel_h, params.kernel_w, params.kernel_w_multiplier, params.kernel_hw_multiplier);
    var b_k = split_digits(tid / $BM_U, params.kernel_h, params.kernel_w, params.kernel_w_multiplier, params.kernel_hw_multiplier);

    $ACC_DECL

    var t = 0u;
    loop {
        if t >= k_total { break; }

        // Load A tile: weight[Co, K].
        for (var e = 0u; e < $STAGE_EPT_U; e++) {
            let flat = tid + e * 256u;
            let row_local = flat / $KTILE_U;  // M dimension (Co)
            let col_local = flat % $KTILE_U;  // K dimension
            let a_row = tile_row + row_local;
            let a_col = t + col_local;
            let in_bounds_a = a_row < m_total && a_col < k_total;
            shared_a[row_local * $A_STRIDE_U + col_local] = select(0.0, weight[a_row * k_total + a_col], in_bounds_a);
        }

        // Load B tile: im2col(input)^T [K, oH*oW].
        // B[k, hw] = input[n, ci, oh*stride+kh-pad, ow*stride+kw-pad]
        for (var e = 0u; e < $STAGE_EPT_U; e++) {
            let row_local = tid / $BM_U + e * (256u / $BM_U);  // K dimension
            let k_idx = t + row_local;

            var val = 0.0;
            if k_idx < k_total && hw_idx < n_total {
                let ih = ih0 + i32(b_k.middle);
                let iw = iw0 + i32(b_k.inner);
                if ih >= 0 && u32(ih) < params.in_h && iw >= 0 && u32(iw) < params.in_w {
                    val = src[batch_src + b_k.outer * in_hw + u32(ih) * params.in_w + u32(iw)];
                }
            }
            shared_b[row_local * $B_STRIDE_U + b_col] = val;
            b_k = advance_digits(b_k, b_step, params.kernel_h, params.kernel_w);
        }

        workgroupBarrier();

        // Compute the register tile over one K stage.
        for (var kk = 0u; kk < $KTILE_U; kk++) {
            $COMPUTE_BODY
        }

        workgroupBarrier();
        t += $KTILE_U;
    }

    // Store: output[n, co, oh*oW+ow] in NCHW layout
    let output_stride = m_total * n_total;  // Co * oH * oW per batch
    let s = $ACC_ARRAY;
    for (var i = 0u; i < $TM_U; i++) {
        for (var j = 0u; j < $TM_U; j++) {
            let co = tile_row + ty * $TM_U + i;
            let hw = tile_col + tx * $TM_U + j;
            if co < m_total && hw < n_total {
                dst[n * output_stride + co * n_total + hw] = s[i][j];
            }
        }
    }
}
