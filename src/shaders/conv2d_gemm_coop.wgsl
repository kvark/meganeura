// Conv2d forward via implicit GEMM — cooperative matrix variant.
//
// C[Co, oH*oW] = A[Co, K] × B[K, oH*oW] per batch item,
// where K = Ci*kH*kW and B is the im2col matrix computed on-the-fly.
//
// Uses 2×2 cooperative matrix tile grid ($OUTPUT_TILE×$OUTPUT_TILE per WG).
// Dispatch: [ceil(Co / $OUTPUT_TILE), ceil(oH*oW / $OUTPUT_TILE), batch]

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

var<storage> src: array<f32>;              // input [N, Ci, H, W]
var<storage> weight: array<vec4<f32>>;     // kernel [Co, K] — vec4 for 128-bit loads
var<storage, read_write> dst: array<f32>;  // output [N, Co, oH, oW]
var<uniform> params: Params;
var<workgroup> shared_a0: array<$ELEM_TYPE, $SHARED_SIZE>;
var<workgroup> shared_a1: array<$ELEM_TYPE, $SHARED_SIZE>;
var<workgroup> shared_b0: array<$ELEM_TYPE, $SHARED_SIZE>;
var<workgroup> shared_b1: array<$ELEM_TYPE, $SHARED_SIZE>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tile_row = wgid.x * $OUTPUT_TILE_U;  // M (Co)
    let tile_col = wgid.y * $OUTPUT_TILE_U;  // N (oH*oW)
    let n = wgid.z;                           // batch

    let m_total = params.out_channels;
    let n_total = params.out_h * params.out_w;
    let kernel_hw = params.kernel_h * params.kernel_w;
    let k_total = params.in_channels * kernel_hw;
    let input_stride = params.in_channels * params.in_h * params.in_w;

    // C offsets for the 4 output tiles (row-major in [Co, oH*oW])
    let c00 = n * m_total * n_total + tile_row * n_total + tile_col;
    let c01 = n * m_total * n_total + tile_row * n_total + (tile_col + $TILE_SIZE_U);
    let c10 = n * m_total * n_total + (tile_row + $TILE_SIZE_U) * n_total + tile_col;
    let c11 = n * m_total * n_total + (tile_row + $TILE_SIZE_U) * n_total + (tile_col + $TILE_SIZE_U);

    let n1_valid = (tile_col + $TILE_SIZE_U) < n_total;
    let m1_valid = (tile_row + $TILE_SIZE_U) < m_total;

    $ACC_INIT

    // Hoisted staging index components
    let v4_row = lid.x >> 2u;
    let v4_col = (lid.x & 3u) << 2u;
    let src_col = lid.x & $TILE_MASK_U;
    let base_row = lid.x >> $TILE_SHIFT_U;

    var t = 0u;
    loop {
        if t >= k_total { break; }

        let zero_val = $ELEM_ZERO;

        // Stage sb0: A-tile weight[Co, K] → shared_b0
        // Vec4 staging: weight is dense [Co, K] row-major
        {
            let gr = tile_row + v4_row;
            let tc4 = t + v4_col;
            let flat = v4_row * $TILE_SIZE_U + v4_col;
            if gr < m_total && (tc4 + 4u) <= k_total {
                let v = weight[(gr * k_total + tc4) >> 2u];
                shared_b0[flat] = $CAST_OPEN v.x $CAST_CLOSE;
                shared_b0[flat + 1u] = $CAST_OPEN v.y $CAST_CLOSE;
                shared_b0[flat + 2u] = $CAST_OPEN v.z $CAST_CLOSE;
                shared_b0[flat + 3u] = $CAST_OPEN v.w $CAST_CLOSE;
            } else {
                shared_b0[flat] = zero_val;
                shared_b0[flat + 1u] = zero_val;
                shared_b0[flat + 2u] = zero_val;
                shared_b0[flat + 3u] = zero_val;
            }
        }

        // Stage sb1: A-tile second row block [Co+TILE, K]
        {
            let gr = (tile_row + $TILE_SIZE_U) + v4_row;
            let tc4 = t + v4_col;
            let flat = v4_row * $TILE_SIZE_U + v4_col;
            if gr < m_total && (tc4 + 4u) <= k_total {
                let v = weight[(gr * k_total + tc4) >> 2u];
                shared_b1[flat] = $CAST_OPEN v.x $CAST_CLOSE;
                shared_b1[flat + 1u] = $CAST_OPEN v.y $CAST_CLOSE;
                shared_b1[flat + 2u] = $CAST_OPEN v.z $CAST_CLOSE;
                shared_b1[flat + 3u] = $CAST_OPEN v.w $CAST_CLOSE;
            } else {
                shared_b1[flat] = zero_val;
                shared_b1[flat + 1u] = zero_val;
                shared_b1[flat + 2u] = zero_val;
                shared_b1[flat + 3u] = zero_val;
            }
        }

        // Stage sa0: B-tile im2col(input)[K, oH*oW] → shared_a0
        // Scalar staging: im2col requires index decomposition
        let cc0 = tile_col + src_col;
        let in_n0 = cc0 < n_total;
        for (var e = 0u; e < $STAGING_ITERS_U; e++) {
            let flat = lid.x + e * 64u;
            let tr = t + base_row + e * $ROW_STRIDE_U;
            var val = zero_val;
            if tr < k_total && in_n0 {
                // Decompose k_idx → (ci, kh, kw)
                let ci = tr / kernel_hw;
                let k_rem = tr - ci * kernel_hw;
                let kh = k_rem / params.kernel_w;
                let kw = k_rem - kh * params.kernel_w;
                // Decompose hw_idx → (oh, ow) → (ih, iw)
                let oh = cc0 / params.out_w;
                let ow = cc0 - oh * params.out_w;
                let ih = i32(oh * params.stride + kh) - i32(params.padding_h);
                let iw = i32(ow * params.stride + kw) - i32(params.padding_w);
                if ih >= 0 && u32(ih) < params.in_h && iw >= 0 && u32(iw) < params.in_w {
                    val = $CAST_OPEN src[n * input_stride + ci * params.in_h * params.in_w + u32(ih) * params.in_w + u32(iw)] $CAST_CLOSE;
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
                let ci = tr / kernel_hw;
                let k_rem = tr - ci * kernel_hw;
                let kh = k_rem / params.kernel_w;
                let kw = k_rem - kh * params.kernel_w;
                let oh = cc1 / params.out_w;
                let ow = cc1 - oh * params.out_w;
                let ih = i32(oh * params.stride + kh) - i32(params.padding_h);
                let iw = i32(ow * params.stride + kw) - i32(params.padding_w);
                if ih >= 0 && u32(ih) < params.in_h && iw >= 0 && u32(iw) < params.in_w {
                    val = $CAST_OPEN src[n * input_stride + ci * params.in_h * params.in_w + u32(ih) * params.in_w + u32(iw)] $CAST_CLOSE;
                }
            }
            shared_a1[flat] = val;
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

    // Store results to output [N, Co, oH, oW] in NCHW layout
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
