// Conv2d backward w.r.t. kernel via implicit GEMM.
//
// grad_weight[Co, Ci*kH*kW] = grad_out_flat[Co, N*oH*oW] × im2col(input)[N*oH*oW, Ci*kH*kW]
// C[Co, Ci*kH*kW] = A[Co, K] × B[K, Ci*kH*kW], K = batch*oH*oW.
//
// Register-tiled matmul, workgroup [16,16,1]. BM, TM and K are generated.
// Dispatch: [ceil(Ci*kH*kW / 64), ceil(Co / 64), 1]

$DIVISOR

$PARAMS_TYPE

var<storage> grad_out: array<f32>;           // [N, Co, oH, oW]
var<storage> src: array<f32>;                // input [N, Ci, H, W]
var<storage, read_write> dst: array<f32>;    // grad_kernel [Co, Ci, kH, kW]
$PARAMS_DECL
var<workgroup> shared_a: array<f32, $SHARED_A_SIZE>;   // A tile: [BM, K], padded stride
var<workgroup> shared_b: array<f32, $SHARED_B_SIZE>;   // B tile: [K, BM], padded stride

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>$COUNTS) {
    let tx = lid.x;
    let ty = lid.y;
    let tile_row = wgid.y * $BM_U;   // M (Co) tile start
    let tile_col = wgid.x * $BM_U;   // N (Ci*kH*kW) tile start
    let tid = ty * 16u + tx;

    let m_total = params.out_channels;                   // Co
    let kernel_hw = params.kernel_h * params.kernel_w;
    let n_total = params.in_channels * kernel_hw;        // Ci*kH*kW
    let go_spatial = params.out_h * params.out_w;        // oH*oW
    let k_total = params.batch * go_spatial;             // N*oH*oW
    let input_spatial = params.in_h * params.in_w;

    // Every tile width divides 256, so each thread stages one A column and
    // one B column (ci, kh, kw) throughout; only their K indices move. A's
    // advances by KTILE per stage and B's by 256/BM per slot, carried through
    // (n, oh, ow) instead of divided.
    let b_col = tid % $BM_U;
    let col_idx = tile_col + b_col;
    let ci = divide_exact(col_idx, kernel_hw, params.kernel_hw_multiplier);
    let k_rem = col_idx - ci * kernel_hw;
    let kh = divide_exact(k_rem, params.kernel_w, params.kernel_w_multiplier);
    let kw = k_rem - kh * params.kernel_w;
    let h0 = i32(kh) - i32(params.padding_h);
    let w0 = i32(kw) - i32(params.padding_w);
    let src_channel = ci * input_spatial;

    $ACC_DECL

    $K_RANGE
    var t = $K_START;
    let a_col = tid % $KTILE_U;
    let a_step = split_digits($KTILE_U, params.out_h, params.out_w, params.column_width_multiplier, params.output_spatial_multiplier);
    var a_k = split_digits(t + a_col, params.out_h, params.out_w, params.column_width_multiplier, params.output_spatial_multiplier);
    let b_step = split_digits(256u / $BM_U, params.out_h, params.out_w, params.column_width_multiplier, params.output_spatial_multiplier);
    var b_k = split_digits(t + tid / $BM_U, params.out_h, params.out_w, params.column_width_multiplier, params.output_spatial_multiplier);
    loop {
        if t >= $K_END { break; }

        // Load A tile: grad_out_flat[Co, N*oH*oW].
        // A[co, n*oH*oW + oh*oW + ow] = grad_out[n, co, oh, ow]
        let k_idx_a = t + a_col;
        let rem_a = a_k.middle * params.out_w + a_k.inner;
        for (var e = 0u; e < $STAGE_EPT_U; e++) {
            let row_local = tid / $KTILE_U + e * (256u / $KTILE_U);  // M dimension (Co)
            let co = tile_row + row_local;

            var val = 0.0;
            if co < m_total && k_idx_a < k_total {
                val = grad_out[(a_k.outer * params.out_channels + co) * go_spatial + rem_a];
            }
            shared_a[row_local * $A_STRIDE_U + a_col] = val;
        }
        a_k = advance_digits(a_k, a_step, params.out_h, params.out_w);

        // Load B tile: im2col(input)[N*oH*oW, Ci*kH*kW].
        // B[k_idx, col] where k_idx = n*oH*oW + oh*oW + ow, col = ci*kH*kW + kh*kW + kw
        // B[k_idx, col] = input[n, ci, oh*stride+kh-padding, ow*stride+kw-padding]
        for (var e = 0u; e < $STAGE_EPT_U; e++) {
            let row_local = tid / $BM_U + e * (256u / $BM_U);  // K dimension
            let k_idx = t + row_local;

            var val = 0.0;
            if k_idx < k_total && col_idx < n_total {
                let ih = i32(b_k.middle * params.stride) + h0;
                let iw = i32(b_k.inner * params.stride) + w0;
                if ih >= 0 && u32(ih) < params.in_h && iw >= 0 && u32(iw) < params.in_w {
                    val = src[(b_k.outer * params.in_channels) * input_spatial + src_channel + u32(ih) * params.in_w + u32(iw)];
                }
            }
            shared_b[row_local * $B_STRIDE_U + b_col] = val;
            b_k = advance_digits(b_k, b_step, params.out_h, params.out_w);
        }

        workgroupBarrier();

        // Compute the register tile over one K stage.
        for (var kk = 0u; kk < $KTILE_U; kk++) {
            $COMPUTE_BODY
        }

        workgroupBarrier();
        t += $KTILE_U;
    }

    // Store: grad_kernel[co, ci*kH*kW + kh*kW + kw]
    // Output layout: [Co, Ci, kH, kW] = [Co, Ci*kH*kW] row-major
    let s = $ACC_ARRAY;
    for (var i = 0u; i < $TM_U; i++) {
        for (var j = 0u; j < $TM_U; j++) {
            let co = tile_row + ty * $TM_U + i;
            let cikk = tile_col + tx * $TM_U + j;
            if co < m_total && cikk < n_total {
                dst[$OUTPUT_OFFSET co * n_total + cikk] = s[i][j];
            }
        }
    }
}
