// Conv2d backward w.r.t. input via implicit GEMM.
//
// grad_input[n] = weight_T @ im2col(grad_out[n])^T
// where weight_T[ci, co*kH*kW + kh*kW + kw] = weight[co, ci, kh, kw].
// Invert the forward cross-correlation: ih = oh*stride + kh - padding_h.
// Thus oh = (ih + padding_h - kh)/stride when divisible, without flipping weights.
//
// C[Ci, H*W] = A[Ci, K] × B[K, H*W], K = Co*kH*kW, per batch item.
// Register-tiled matmul, workgroup [16,16,1]. BM, TM and K are generated.
//
// Dispatch: [ceil(H*W / 64), ceil(Ci / 64), batch]

$DIVISOR

$PARAMS_TYPE

var<storage> grad_out: array<f32>;         // grad_output [N, Co, oH, oW]
var<storage> weight: array<f32>;           // kernel [Co, Ci, kH, kW]
var<storage, read_write> dst: array<f32>;  // grad_input [N, Ci, H, W]
$PARAMS_DECL
var<workgroup> shared_a: array<f32, $SHARED_A_SIZE>; // A tile: [BM, K], padded stride
var<workgroup> shared_b: array<f32, $SHARED_B_SIZE>; // B tile: [K, BM], padded stride

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tx = lid.x;
    let ty = lid.y;
    let n = wgid.z;                // batch index
    let tile_row = wgid.y * $BM_U;   // M (Ci) tile start
    let tile_col = wgid.x * $BM_U;   // N (H*W) tile start
    let tid = ty * 16u + tx;

    let kernel_hw = params.kernel_h * params.kernel_w;
    let k_total = params.out_channels * kernel_hw;  // Co * kH * kW
    let n_total = params.in_h * params.in_w;        // H * W (grad_input spatial)
    let m_total = params.in_channels;               // Ci
    let go_spatial = params.out_h * params.out_w;    // oH * oW

    let pad_h = i32(params.padding_h);
    let pad_w = i32(params.padding_w);
    let batch_grad = n * params.out_channels * go_spatial;

    // Every tile width divides 256, so each thread stages one A column and
    // one B column (input pixel) throughout; only their K indices move. A's
    // advances by KTILE per stage and B's by 256/BM per slot, carried through
    // (co, kh, kw) instead of divided.
    let a_col = tid % $KTILE_U;
    let a_step = split_digits($KTILE_U, params.kernel_h, params.kernel_w, params.kernel_w_multiplier, params.kernel_hw_multiplier);
    var a_k = split_digits(a_col, params.kernel_h, params.kernel_w, params.kernel_w_multiplier, params.kernel_hw_multiplier);
    let b_col = tid % $BM_U;
    let hw_idx = tile_col + b_col;
    let ih = divide_exact(hw_idx, params.in_w, params.column_width_multiplier);
    let iw = hw_idx - ih * params.in_w;
    let h0 = i32(ih) + pad_h;
    let w0 = i32(iw) + pad_w;
    let b_step = split_digits(256u / $BM_U, params.kernel_h, params.kernel_w, params.kernel_w_multiplier, params.kernel_hw_multiplier);
    var b_k = split_digits(tid / $BM_U, params.kernel_h, params.kernel_w, params.kernel_w_multiplier, params.kernel_hw_multiplier);

    $ACC_DECL

    var t = 0u;
    loop {
        if t >= k_total { break; }

        // Load A tile: weight_T[Ci, K].
        // weight_T[ci, co*kH*kW + kh*kW + kw] = weight[co, ci, kh, kw]
        // weight layout: [Co, Ci, kH, kW] → weight[co * Ci*kH*kW + ci * kH*kW + kh*kW + kw]
        let k_idx_a = t + a_col;
        let k_rem_a = a_k.middle * params.kernel_w + a_k.inner;
        for (var e = 0u; e < $STAGE_EPT_U; e++) {
            let row_local = tid / $KTILE_U + e * (256u / $KTILE_U);  // M dimension (Ci)
            let ci = tile_row + row_local;

            var val = 0.0;
            if ci < m_total && k_idx_a < k_total {
                val = weight[(a_k.outer * m_total + ci) * kernel_hw + k_rem_a];
            }
            shared_a[row_local * $A_STRIDE_U + a_col] = val;
        }
        a_k = advance_digits(a_k, a_step, params.kernel_h, params.kernel_w);

        // Load B tile: im2col(grad_out)^T [K, H*W].
        // B[k, hw] where k = co*kH*kW+kh*kW+kw, hw = ih*W+iw
        // grad_out position: oh = (ih + pad_h - kh) / stride when divisible,
        // likewise ow.
        for (var e = 0u; e < $STAGE_EPT_U; e++) {
            let row_local = tid / $BM_U + e * (256u / $BM_U);  // K dimension
            let k_idx = t + row_local;

            var val = 0.0;
            if k_idx < k_total && hw_idx < n_total {
                let h_off = h0 - i32(b_k.middle);
                let w_off = w0 - i32(b_k.inner);
                if params.stride == 1u {
                    if h_off >= 0 && u32(h_off) < params.out_h && w_off >= 0 && u32(w_off) < params.out_w {
                        val = grad_out[batch_grad + b_k.outer * go_spatial + u32(h_off) * params.out_w + u32(w_off)];
                    }
                } else if h_off >= 0 && w_off >= 0 {
                    let oh = u32(h_off) / params.stride;
                    let ow = u32(w_off) / params.stride;
                    let exact = oh * params.stride == u32(h_off) && ow * params.stride == u32(w_off);
                    if exact && oh < params.out_h && ow < params.out_w {
                        val = grad_out[batch_grad + b_k.outer * go_spatial + oh * params.out_w + ow];
                    }
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

    // Store: grad_input[n, ci, ih*W+iw] in NCHW layout
    let output_stride = m_total * n_total;
    let s = $ACC_ARRAY;
    for (var i = 0u; i < $TM_U; i++) {
        for (var j = 0u; j < $TM_U; j++) {
            let ci = tile_row + ty * $TM_U + i;
            let hw = tile_col + tx * $TM_U + j;
            if ci < m_total && hw < n_total {
                dst[n * output_stride + ci * n_total + hw] = s[i][j];
            }
        }
    }
}
