// Conv2d backward w.r.t. input via implicit GEMM, one stride phase at a time.
//
// grad_input[n, ci, ih, iw] sums weight[co, ci, kh, kw] · grad_out[n, co, oh, ow]
// over the taps with ih + padding_h = oh*stride + kh (likewise w), without
// flipping weights. Input pixels with the same (ih mod stride, iw mod stride)
// phase reach outputs through the same tap residues, so each phase is a dense
// GEMM over only its taps; enumerating every tap for every pixel would make
// all but 1/stride² of a strided gradient's products zero.
//
// Phase (ph, pw) holds ih = ph + stride*r. Its taps are kh = rh + stride*a for
// rh = (ph + padding_h) mod stride and a < taps_h, which reach oh = r + dh - a
// with dh = (ph + padding_h - rh) / stride; likewise w.
// C[Ci, Hs*Ws] = A[Ci, K] × B[K, Hs*Ws], K = Co*taps_h*taps_w, Hs = ceil(H/stride),
// per batch item and phase. At stride 1 there is one phase with every tap.
// Register-tiled matmul, workgroup [16,16,1]. BM, TM and K are generated.
//
// Dispatch: [ceil(Hs*Ws / BM), ceil(Ci / BM), batch*stride²]

$DIVISOR

$PARAMS_TYPE

var<storage> grad_out: array<f32>;         // grad_output [N, Co, oH, oW]
var<storage> weight: array<f32>;           // kernel [Co, Ci, kH, kW]
var<storage, read_write> dst: array<f32>;  // grad_input [N, Ci, H, W]
$PARAMS_DECL
var<workgroup> shared_a: array<f32, $SHARED_A_SIZE>; // A tile: [BM, K], padded stride
var<workgroup> shared_b: array<f32, $SHARED_B_SIZE>; // B tile: [K, BM], padded stride

// (co, a, b) digits of a phase's K index. Its radices vary by phase, so this
// divides; it runs once per thread.
fn split_taps(value: u32, taps_h: u32, taps_w: u32) -> Digits {
    let taps = taps_h * taps_w;
    let rest = value % taps;
    return Digits(value / taps, rest / taps_w, rest % taps_w);
}

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tx = lid.x;
    let ty = lid.y;
    let stride = params.stride;
    let phases = stride * stride;
    let n = wgid.z / phases;            // batch index
    let phase = wgid.z - n * phases;
    let ph = phase / stride;
    let pw = phase - ph * stride;
    let tile_row = wgid.y * $BM_U;   // M (Ci) tile start
    let tile_col = wgid.x * $BM_U;   // N (phase pixel) tile start
    let tid = ty * 16u + tx;

    let rh = (ph + params.padding_h) % stride;
    let rw = (pw + params.padding_w) % stride;
    let taps_h = select(0u, (params.kernel_h - rh + stride - 1u) / stride, rh < params.kernel_h);
    let taps_w = select(0u, (params.kernel_w - rw + stride - 1u) / stride, rw < params.kernel_w);
    let dh = i32((ph + params.padding_h - rh) / stride);
    let dw = i32((pw + params.padding_w - rw) / stride);
    // A phase without taps runs no K stage and stores zeros; one-tap radices
    // keep its digit arithmetic defined.
    let radix_h = max(taps_h, 1u);
    let radix_w = max(taps_w, 1u);
    let phase_w = (params.in_w + stride - 1u) / stride;  // Ws

    let kernel_hw = params.kernel_h * params.kernel_w;
    let k_total = params.out_channels * taps_h * taps_w;          // Co * taps
    let n_total = ((params.in_h + stride - 1u) / stride) * phase_w; // Hs * Ws
    let m_total = params.in_channels;                             // Ci
    let go_spatial = params.out_h * params.out_w;                 // oH * oW
    let batch_grad = n * params.out_channels * go_spatial;

    // Every tile width divides 256, so each thread stages one A column and
    // one B column (phase pixel) throughout; only their K indices move. A's
    // advances by KTILE per stage and B's by 256/BM per slot, carried through
    // (co, a, b) instead of divided.
    let a_col = tid % $KTILE_U;
    let a_step = split_taps($KTILE_U, radix_h, radix_w);
    var a_k = split_taps(a_col, radix_h, radix_w);
    let b_col = tid % $BM_U;
    let hw_idx = tile_col + b_col;
    let r = divide_exact(hw_idx, phase_w, params.column_width_multiplier);
    let c = hw_idx - r * phase_w;
    let h0 = i32(r) + dh;
    let w0 = i32(c) + dw;
    let b_step = split_taps(256u / $BM_U, radix_h, radix_w);
    var b_k = split_taps(tid / $BM_U, radix_h, radix_w);

    $ACC_DECL

    var t = 0u;
    loop {
        if t >= k_total { break; }

        // Load A tile: weight_T[Ci, K] at this phase's taps.
        // weight layout: [Co, Ci, kH, kW] → weight[co * Ci*kH*kW + ci * kH*kW + kh*kW + kw]
        let k_idx_a = t + a_col;
        let tap_a = (rh + stride * a_k.middle) * params.kernel_w + rw + stride * a_k.inner;
        for (var e = 0u; e < $STAGE_EPT_U; e++) {
            let row_local = tid / $KTILE_U + e * (256u / $KTILE_U);  // M dimension (Ci)
            let ci = tile_row + row_local;

            var val = 0.0;
            if ci < m_total && k_idx_a < k_total {
                val = weight[(a_k.outer * m_total + ci) * kernel_hw + tap_a];
            }
            shared_a[row_local * $A_STRIDE_U + a_col] = val;
        }
        a_k = advance_digits(a_k, a_step, radix_h, radix_w);

        // Load B tile: grad_out at oh = r + dh - a, ow = c + dw - b [K, Hs*Ws].
        for (var e = 0u; e < $STAGE_EPT_U; e++) {
            let row_local = tid / $BM_U + e * (256u / $BM_U);  // K dimension
            let k_idx = t + row_local;

            var val = 0.0;
            if k_idx < k_total && hw_idx < n_total {
                let oh = h0 - i32(b_k.middle);
                let ow = w0 - i32(b_k.inner);
                if oh >= 0 && u32(oh) < params.out_h && ow >= 0 && u32(ow) < params.out_w {
                    val = grad_out[batch_grad + b_k.outer * go_spatial + u32(oh) * params.out_w + u32(ow)];
                }
            }
            shared_b[row_local * $B_STRIDE_U + b_col] = val;
            b_k = advance_digits(b_k, b_step, radix_h, radix_w);
        }

        workgroupBarrier();

        // Compute the register tile over one K stage.
        for (var kk = 0u; kk < $KTILE_U; kk++) {
            $COMPUTE_BODY
        }

        workgroupBarrier();
        t += $KTILE_U;
    }

    // Store this phase's pixels of grad_input[n, ci, ih, iw] in NCHW layout;
    // phases past an odd edge have fewer rows or columns than Hs or Ws.
    let in_hw = params.in_h * params.in_w;
    let output_stride = m_total * in_hw;
    let s = $ACC_ARRAY;
    for (var i = 0u; i < $TM_U; i++) {
        for (var j = 0u; j < $TM_U; j++) {
            let ci = tile_row + ty * $TM_U + i;
            let hw = tile_col + $OUT_COL;
            if ci < m_total && hw < n_total {
                let out_r = divide_exact(hw, phase_w, params.column_width_multiplier);
                let ih = ph + stride * out_r;
                let iw = pw + stride * (hw - out_r * phase_w);
                if ih < params.in_h && iw < params.in_w {
                    dst[n * output_stride + ci * in_hw + ih * params.in_w + iw] = s[i][j];
                }
            }
        }
    }
}
