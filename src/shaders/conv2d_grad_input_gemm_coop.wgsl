// Conv2d backward w.r.t. input via implicit GEMM — cooperative matrix variant.
//
// grad_input[n] = weight_T @ im2col(grad_out[n])^T
// C[Ci, H*W] = A[Ci, K] × B[K, H*W], K = Co*kH*kW, per batch item.
//
// Uses 2×2 cooperative matrix tile grid ($OUTPUT_TILE×$OUTPUT_TILE per WG).
// Dispatch: [ceil(Ci / $OUTPUT_TILE), ceil(H*W / $OUTPUT_TILE), batch]

$ENABLE_F16
enable wgpu_cooperative_matrix;

struct Params {
    batch: u32,
    in_channels: u32,
    in_h: u32,
    in_w: u32,
    out_channels: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride: u32,
    padding_h: u32,
    out_h: u32,
    out_w: u32,
    padding_w: u32,
}

var<storage> grad_out: array<f32>;
var<storage> weight: array<f32>;
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;
var<workgroup> shared_a0: array<$ELEM_TYPE, $SHARED_SIZE>;
var<workgroup> shared_a1: array<$ELEM_TYPE, $SHARED_SIZE>;
var<workgroup> shared_b0: array<$ELEM_TYPE, $SHARED_SIZE>;
var<workgroup> shared_b1: array<$ELEM_TYPE, $SHARED_SIZE>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tile_row = wgid.x * $OUTPUT_TILE_U;  // M (Ci)
    let tile_col = wgid.y * $OUTPUT_TILE_U;  // N (H*W)
    let n = wgid.z;                           // batch

    let m_total = params.in_channels;
    let n_total = params.in_h * params.in_w;
    let kernel_hw = params.kernel_h * params.kernel_w;
    let k_total = params.out_channels * kernel_hw;
    let go_spatial = params.out_h * params.out_w;

    let pad_h = i32(params.kernel_h) - 1 - i32(params.padding_h);
    let pad_w = i32(params.kernel_w) - 1 - i32(params.padding_w);

    // C offsets for the 4 output tiles (row-major in [Ci, H*W])
    let c00 = n * m_total * n_total + tile_row * n_total + tile_col;
    let c01 = n * m_total * n_total + tile_row * n_total + (tile_col + $TILE_SIZE_U);
    let c10 = n * m_total * n_total + (tile_row + $TILE_SIZE_U) * n_total + tile_col;
    let c11 = n * m_total * n_total + (tile_row + $TILE_SIZE_U) * n_total + (tile_col + $TILE_SIZE_U);

    let n1_valid = (tile_col + $TILE_SIZE_U) < n_total;
    let m1_valid = (tile_row + $TILE_SIZE_U) < m_total;

    $ACC_INIT

    // Hoisted staging index components
    let src_col = lid.x & $TILE_MASK_U;
    let base_row = lid.x >> $TILE_SHIFT_U;

    var t = 0u;
    loop {
        if t >= k_total { break; }

        let zero_val = $ELEM_ZERO;

        // Stage sa0: B-tile [K, H*W] → im2col(grad_out)^T
        // sa0[flat] = im2col[t+row_local, tile_col+col_local]
        let cc0 = tile_col + src_col;
        let in_n0 = cc0 < n_total;
        for (var e = 0u; e < $STAGING_ITERS_U; e++) {
            let flat = lid.x + e * 64u;
            let tr = t + base_row + e * $ROW_STRIDE_U;
            var val = zero_val;
            if tr < k_total && in_n0 {
                let co = tr / kernel_hw;
                let k_rem = tr - co * kernel_hw;
                let kh = k_rem / params.kernel_w;
                let kw = k_rem - kh * params.kernel_w;
                let ih = cc0 / params.in_w;
                let iw = cc0 - ih * params.in_w;
                if params.stride == 1u {
                    let oh = i32(ih) + pad_h - i32(kh);
                    let ow = i32(iw) + pad_w - i32(kw);
                    if oh >= 0 && u32(oh) < params.out_h && ow >= 0 && u32(ow) < params.out_w {
                        val = $CAST_OPEN grad_out[n * params.out_channels * go_spatial + co * go_spatial + u32(oh) * params.out_w + u32(ow)] $CAST_CLOSE;
                    }
                } else {
                    let h_off = i32(ih) + i32(params.padding_h) - i32(kh);
                    let w_off = i32(iw) + i32(params.padding_w) - i32(kw);
                    let i_stride = i32(params.stride);
                    if h_off >= 0 && w_off >= 0 && (h_off % i_stride) == 0 && (w_off % i_stride) == 0 {
                        let oh = u32(h_off) / params.stride;
                        let ow = u32(w_off) / params.stride;
                        if oh < params.out_h && ow < params.out_w {
                            val = $CAST_OPEN grad_out[n * params.out_channels * go_spatial + co * go_spatial + oh * params.out_w + ow] $CAST_CLOSE;
                        }
                    }
                }
            }
            shared_a0[flat] = val;
        }

        // Stage sa1: B-tile second column block [K, tile_col+TILE..tile_col+2*TILE]
        let cc1 = tile_col + $TILE_SIZE_U + src_col;
        let in_n1 = cc1 < n_total;
        for (var e = 0u; e < $STAGING_ITERS_U; e++) {
            let flat = lid.x + e * 64u;
            let tr = t + base_row + e * $ROW_STRIDE_U;
            var val = zero_val;
            if tr < k_total && in_n1 {
                let co = tr / kernel_hw;
                let k_rem = tr - co * kernel_hw;
                let kh = k_rem / params.kernel_w;
                let kw = k_rem - kh * params.kernel_w;
                let ih = cc1 / params.in_w;
                let iw = cc1 - ih * params.in_w;
                if params.stride == 1u {
                    let oh = i32(ih) + pad_h - i32(kh);
                    let ow = i32(iw) + pad_w - i32(kw);
                    if oh >= 0 && u32(oh) < params.out_h && ow >= 0 && u32(ow) < params.out_w {
                        val = $CAST_OPEN grad_out[n * params.out_channels * go_spatial + co * go_spatial + u32(oh) * params.out_w + u32(ow)] $CAST_CLOSE;
                    }
                } else {
                    let h_off = i32(ih) + i32(params.padding_h) - i32(kh);
                    let w_off = i32(iw) + i32(params.padding_w) - i32(kw);
                    let i_stride = i32(params.stride);
                    if h_off >= 0 && w_off >= 0 && (h_off % i_stride) == 0 && (w_off % i_stride) == 0 {
                        let oh = u32(h_off) / params.stride;
                        let ow = u32(w_off) / params.stride;
                        if oh < params.out_h && ow < params.out_w {
                            val = $CAST_OPEN grad_out[n * params.out_channels * go_spatial + co * go_spatial + oh * params.out_w + ow] $CAST_CLOSE;
                        }
                    }
                }
            }
            shared_a1[flat] = val;
        }

        // Stage sb0: A-tile [Ci, K] → weight_T
        // sb0[flat] = weight_T[tile_row+row_local, t+col_local]
        let tc = t + src_col;
        let in_k = tc < k_total;
        for (var e = 0u; e < $STAGING_ITERS_U; e++) {
            let flat = lid.x + e * 64u;
            let gr = tile_row + base_row + e * $ROW_STRIDE_U;
            var val = zero_val;
            if gr < m_total && in_k {
                let co = tc / kernel_hw;
                let k_rem = tc - co * kernel_hw;
                let kh = k_rem / params.kernel_w;
                let kw = k_rem - kh * params.kernel_w;
                val = $CAST_OPEN weight[(co * m_total + gr) * kernel_hw + kh * params.kernel_w + kw] $CAST_CLOSE;
            }
            shared_b0[flat] = val;
        }

        // Stage sb1: A-tile second row block [Ci+TILE..Ci+2*TILE, K]
        for (var e = 0u; e < $STAGING_ITERS_U; e++) {
            let flat = lid.x + e * 64u;
            let gr = tile_row + $TILE_SIZE_U + base_row + e * $ROW_STRIDE_U;
            var val = zero_val;
            if gr < m_total && in_k {
                let co = tc / kernel_hw;
                let k_rem = tc - co * kernel_hw;
                let kh = k_rem / params.kernel_w;
                let kw = k_rem - kh * params.kernel_w;
                val = $CAST_OPEN weight[(co * m_total + gr) * kernel_hw + kh * params.kernel_w + kw] $CAST_CLOSE;
            }
            shared_b1[flat] = val;
        }

        workgroupBarrier();

        // Cooperative matrix multiply-add: C += A × B
        let a0 = coopLoadT<$COOP_AB>(&shared_b0[0], $TILE_SIZE_U);
        let a1 = coopLoadT<$COOP_AB>(&shared_b1[0], $TILE_SIZE_U);
        let b0 = coopLoadT<$COOP_BA>(&shared_a0[0], $TILE_SIZE_U);
        let b1 = coopLoadT<$COOP_BA>(&shared_a1[0], $TILE_SIZE_U);
        acc00 = coopMultiplyAdd(a0, b0, acc00);
        acc01 = coopMultiplyAdd(a0, b1, acc01);
        acc10 = coopMultiplyAdd(a1, b0, acc10);
        acc11 = coopMultiplyAdd(a1, b1, acc11);

        workgroupBarrier();
        t += $TILE_SIZE_U;
    }

    // Store results to grad_input [N, Ci, H, W] in NCHW layout
    coopStoreT(acc00, &dst[c00], n_total);
    if n1_valid {
        coopStoreT(acc01, &dst[c01], n_total);
    }
    if m1_valid {
        coopStoreT(acc10, &dst[c10], n_total);
    }
    if n1_valid && m1_valid {
        coopStoreT(acc11, &dst[c11], n_total);
    }
}
