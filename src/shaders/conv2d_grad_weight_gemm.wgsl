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
    let pointwise = params.kernel_h == 1u && params.kernel_w == 1u && params.stride == 1u
        && params.padding_h == 0u && params.padding_w == 0u;

    // Every tile width divides 256, so each thread stages one K position
    // (image, output row, output column) of both tiles: lanes walk the
    // reduction, which is contiguous in NCHW for grad_out and for each
    // im2col column of the input, so a wave reads runs of pixels instead of
    // one element from each of its columns' channel planes. The K position
    // advances by KTILE per stage, carried through (n, oh, ow) instead of
    // divided; a thread's columns are the same every stage and advance by
    // 256/KTILE per slot through (ci, kh, kw).
    let k_lane = tid % $KTILE_U;
    let col_lane = tid / $KTILE_U;
    let col_step = split_digits(256u / $KTILE_U, params.kernel_h, params.kernel_w, params.kernel_w_multiplier, params.kernel_hw_multiplier);
    let col_first = split_digits(tile_col + col_lane, params.kernel_h, params.kernel_w, params.kernel_w_multiplier, params.kernel_hw_multiplier);

    $ACC_DECL

    $K_RANGE
    var t = $K_START;
    let k_step = split_digits($KTILE_U, params.out_h, params.out_w, params.column_width_multiplier, params.output_spatial_multiplier);
    var k_pos = split_digits(t + k_lane, params.out_h, params.out_w, params.column_width_multiplier, params.output_spatial_multiplier);
    loop {
        if t >= $K_END { break; }

        let k_idx = t + k_lane;
        let in_k = k_idx < k_total;

        // Load A tile: grad_out_flat[Co, N*oH*oW].
        // A[co, n*oH*oW + oh*oW + ow] = grad_out[n, co, oh, ow]
        let rem_a = k_pos.middle * params.out_w + k_pos.inner;
        for (var e = 0u; e < $STAGE_EPT_U; e++) {
            let row_local = col_lane + e * (256u / $KTILE_U);  // M dimension (Co)
            let co = tile_row + row_local;

            var val = 0.0;
            if co < m_total && in_k {
                val = grad_out[(k_pos.outer * params.out_channels + co) * go_spatial + rem_a];
            }
            shared_a[row_local * $A_STRIDE_U + k_lane] = val;
        }

        // Load B tile: im2col(input)[N*oH*oW, Ci*kH*kW].
        // B[k_idx, col] where k_idx = n*oH*oW + oh*oW + ow, col = ci*kH*kW + kh*kW + kw
        // B[k_idx, col] = input[n, ci, oh*stride+kh-padding, ow*stride+kw-padding]
        let h0 = i32(k_pos.middle * params.stride) - i32(params.padding_h);
        let w0 = i32(k_pos.inner * params.stride) - i32(params.padding_w);
        let image = k_pos.outer * params.in_channels * input_spatial;
        var c = col_first;
        for (var e = 0u; e < $STAGE_EPT_U; e++) {
            let col_local = col_lane + e * (256u / $KTILE_U);  // N dimension
            var val = 0.0;
            if in_k && tile_col + col_local < n_total {
                // A pointwise window preserves the reduction's spatial index.
                if pointwise {
                    val = src[image + c.outer * input_spatial + rem_a];
                } else {
                    let ih = h0 + i32(c.middle);
                    let iw = w0 + i32(c.inner);
                    if ih >= 0 && u32(ih) < params.in_h && iw >= 0 && u32(iw) < params.in_w {
                        val = src[image + c.outer * input_spatial + u32(ih) * params.in_w + u32(iw)];
                    }
                }
            }
            shared_b[k_lane * $B_STRIDE_U + col_local] = val;
            c = advance_digits(c, col_step, params.kernel_h, params.kernel_w);
        }
        k_pos = advance_digits(k_pos, k_step, params.out_h, params.out_w);

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
            let cikk = tile_col + $OUT_COL;
            if co < m_total && cikk < n_total {
                dst[$OUTPUT_OFFSET co * n_total + cikk] = s[i][j];
            }
        }
    }
}
