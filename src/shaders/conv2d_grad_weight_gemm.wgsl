// Conv2d backward w.r.t. kernel via implicit GEMM.
//
// grad_weight[Co, Ci*kH*kW] = grad_out_flat[Co, N*oH*oW] × im2col(input)[N*oH*oW, Ci*kH*kW]
// C[Co, Ci*kH*kW] = A[Co, K] × B[K, Ci*kH*kW], K = batch*oH*oW.
//
// Register-tiled matmul, workgroup [16,16,1]. BM, BN, TM, TN and K are generated.
// Dispatch: [ceil(Ci*kH*kW / BN), ceil(Co / BM), 1]

$DIVISOR

$PARAMS_TYPE

var<storage> grad_out: array<f32>;           // [N, Co, oH, oW]
var<storage> src: array<f32>;                // input [N, Ci, H, W]
var<storage, read_write> dst: array<f32>;    // grad_kernel [Co, Ci, kH, kW]
$PARAMS_DECL
var<workgroup> shared_a: array<f32, $SHARED_A_SIZE>;   // A tile: [BM, K]
var<workgroup> shared_b: array<f32, $SHARED_B_SIZE>;   // B tile: [K, BN]

$EXTRA_SHARED

@compute @workgroup_size($WORKGROUP_SIZE)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>$COUNTS$SUBGROUP_PARAMS) {
    let tx = $LOCAL_X;
    let ty = $LOCAL_Y;
    let tile_row = wgid.y * $BM_U;   // M (Co) tile start
    let tile_col = wgid.x * $BN_U;   // N (Ci*kH*kW) tile start
    let tid = ty * 16u + tx;

    let m_total = params.out_channels;                   // Co
    let kernel_hw = params.kernel_h * params.kernel_w;
    let n_total = params.in_channels * kernel_hw;        // Ci*kH*kW
    let go_spatial = params.out_h * params.out_w;        // oH*oW
    let k_total = params.batch * go_spatial;             // N*oH*oW
    let input_spatial = params.in_h * params.in_w;

    $ACC_DECL

    $K_RANGE
    var t = $K_START;
    loop {
        if t >= $K_END { break; }

        // Load A tile: grad_out_flat[Co, N*oH*oW].
        // A[co, n*oH*oW + oh*oW + ow] = grad_out[n, co, oh, ow]
        for (var e = 0u; e < $STAGE_A_EPT_U; e++) {
            let flat = tid + e * $THREADS_U;
            let row_local = flat / $KTILE_U;  // M dimension (Co)
            let col_local = flat % $KTILE_U;  // K dimension
            let co = tile_row + row_local;
            let k_idx = t + col_local;

            var val = 0.0;
            if co < m_total && k_idx < k_total {
                let n = divide_exact(k_idx, go_spatial, params.output_spatial_multiplier);
                let rem = k_idx - n * go_spatial;
                val = grad_out[(n * params.out_channels + co) * go_spatial + rem];
            }
            shared_a[row_local * $KTILE_U + col_local] = val;
        }

        // Load B tile: im2col(input)[N*oH*oW, Ci*kH*kW].
        // B[k_idx, col] where k_idx = n*oH*oW + oh*oW + ow, col = ci*kH*kW + kh*kW + kw
        // B[k_idx, col] = input[n, ci, oh*stride+kh-padding, ow*stride+kw-padding]
        for (var e = 0u; e < $STAGE_B_EPT_U; e++) {
            let flat = tid + e * $THREADS_U;
            let row_local = flat / $BN_U;  // K dimension
            let col_local = flat % $BN_U;  // N dimension (Ci*kH*kW)
            let k_idx = t + row_local;
            let col_idx = tile_col + col_local;

            var val = 0.0;
            if k_idx < k_total && col_idx < n_total {
                // Decompose k_idx → (n, oh, ow)
                let n = divide_exact(k_idx, go_spatial, params.output_spatial_multiplier);
                let rem = k_idx - n * go_spatial;
                let oh = divide_exact(rem, params.out_w, params.column_width_multiplier);
                let ow = rem - oh * params.out_w;
                // Decompose col_idx → (ci, kh, kw)
                let ci = divide_exact(col_idx, kernel_hw, params.kernel_hw_multiplier);
                let k_rem = col_idx - ci * kernel_hw;
                let kh = divide_exact(k_rem, params.kernel_w, params.kernel_w_multiplier);
                let kw = k_rem - kh * params.kernel_w;
                // Input position
                let ih = i32(oh * params.stride + kh) - i32(params.padding_h);
                let iw = i32(ow * params.stride + kw) - i32(params.padding_w);
                if ih >= 0 && u32(ih) < params.in_h && iw >= 0 && u32(iw) < params.in_w {
                    val = src[((n * params.in_channels + ci) * params.in_h + u32(ih)) * params.in_w + u32(iw)];
                }
            }
            shared_b[row_local * $BN_U + col_local] = val;
        }

        workgroupBarrier();

        // Compute the register tile over one K stage.
        for (var kk = 0u; kk < $KTILE_U; kk += $K_STEP_U) {
            $COMPUTE_BODY
        }

        workgroupBarrier();
        t += $KTILE_U;
    }

    // Store: grad_kernel[co, ci*kH*kW + kh*kW + kw]
    // Output layout: [Co, Ci, kH, kW] = [Co, Ci*kH*kW] row-major
    $STORE_PREPARE
    let s = $ACC_ARRAY;
    for (var i = 0u; i < $TM_U; i++) {
        for (var j = 0u; j < $TN_U; j++) {
            let co = tile_row + ty * $TM_U + i;
            let cikk = tile_col + tx * $TN_U + j;
            if co < m_total && cikk < n_total {
                dst[$OUTPUT_OFFSET co * n_total + cikk] = s[i][j];
            }
        }
    }
}
