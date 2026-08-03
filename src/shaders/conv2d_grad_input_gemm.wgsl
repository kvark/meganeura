// Conv2d backward w.r.t. input via implicit GEMM.
//
// grad_input[n] = weight_T @ im2col(grad_out[n])^T
// where weight_T[ci, co*kH*kW + kh*kW + kw] = weight[co, ci, kh, kw]
// and im2col of grad_out uses transposed padding (kH-1-pad, kW-1-pad).
//
// C[Ci, H*W] = A[Ci, K] × B[K, H*W], K = Co*kH*kW, per batch item.
// BM=64, BN=64, KTILE=16, TM=4, TN=4, workgroup [16,16,1]
//
// Dispatch: [ceil(H*W / 64), ceil(Ci / 64), batch]

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
    inv_kernel_w: f32,
    inv_kernel_hw: f32,
    inv_col_w: f32,
    inv_go_spatial: f32,
}

var<storage> grad_out: array<f32>;         // grad_output [N, Co, oH, oW]
var<storage> weight: array<f32>;           // kernel [Co, Ci, kH, kW]
var<storage, read_write> dst: array<f32>;  // grad_input [N, Ci, H, W]
var<uniform> params: Params;
var<workgroup> shared_a: array<f32, $SHARED_SIZE>; // A tile: [64, 16]
var<workgroup> shared_b: array<f32, $SHARED_SIZE>; // B tile: [16, 64]

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

    // Transposed padding for the "flipped convolution"
    let pad_h = i32(params.kernel_h) - 1 - i32(params.padding_h);
    let pad_w = i32(params.kernel_w) - 1 - i32(params.padding_w);

    $ACC_DECL

    var t = 0u;
    loop {
        if t >= k_total { break; }

        // Load A tile: weight_T[Ci, K] → shared_a[64, 16]
        // weight_T[ci, co*kH*kW + kh*kW + kw] = weight[co, ci, kh, kw]
        // weight layout: [Co, Ci, kH, kW] → weight[co * Ci*kH*kW + ci * kH*kW + kh*kW + kw]
        for (var e = 0u; e < $STAGE_EPT_U; e++) {
            let flat = tid + e * 256u;
            let row_local = flat / 16u;  // M dimension (Ci)
            let col_local = flat % 16u;  // K dimension
            let ci = tile_row + row_local;
            let k_idx = t + col_local;

            var val = 0.0;
            if ci < m_total && k_idx < k_total {
                // Decompose k_idx → (co, kh, kw) via reciprocal multiply
                let co = u32(f32(k_idx) * params.inv_kernel_hw);
                let k_rem = k_idx - co * kernel_hw;
                let kh = u32(f32(k_rem) * params.inv_kernel_w);
                let kw = k_rem - kh * params.kernel_w;
                // Read weight[co, ci, kh, kw]
                val = weight[(co * m_total + ci) * kernel_hw + kh * params.kernel_w + kw];
            }
            shared_a[row_local * 16u + col_local] = val;
        }

        // Load B tile: im2col(grad_out)^T [K, H*W] → shared_b[16, 64]
        // B[k, hw] where k = co*kH*kW+kh*kW+kw, hw = ih*W+iw
        // grad_out position: oh = ih + pad_h - kh (for stride=1)
        //                    ow = iw + pad_w - kw
        for (var e = 0u; e < $STAGE_EPT_U; e++) {
            let flat = tid + e * 256u;
            let row_local = flat / $BM_U;  // K dimension (within KTILE=16)
            let col_local = flat % $BM_U;  // N dimension (H*W)
            let k_idx = t + row_local;
            let hw_idx = tile_col + col_local;

            var val = 0.0;
            if k_idx < k_total && hw_idx < n_total {
                let co = u32(f32(k_idx) * params.inv_kernel_hw);
                let k_rem = k_idx - co * kernel_hw;
                let kh = u32(f32(k_rem) * params.inv_kernel_w);
                let kw = k_rem - kh * params.kernel_w;
                let ih = u32(f32(hw_idx) * params.inv_col_w);
                let iw = hw_idx - ih * params.in_w;

                if params.stride == 1u {
                    // Fast path: oh = ih + pad_h - kh, ow = iw + pad_w - kw
                    let oh = i32(ih) + pad_h - i32(kh);
                    let ow = i32(iw) + pad_w - i32(kw);
                    if oh >= 0 && u32(oh) < params.out_h && ow >= 0 && u32(ow) < params.out_w {
                        val = grad_out[n * params.out_channels * go_spatial + co * go_spatial + u32(oh) * params.out_w + u32(ow)];
                    }
                } else {
                    // General stride: oh = (ih + padding - kh) / stride (when divisible)
                    let h_off = i32(ih) + i32(params.padding_h) - i32(kh);
                    let w_off = i32(iw) + i32(params.padding_w) - i32(kw);
                    let i_stride = i32(params.stride);
                    if h_off >= 0 && w_off >= 0 && (h_off % i_stride) == 0 && (w_off % i_stride) == 0 {
                        let oh = u32(h_off) / params.stride;
                        let ow = u32(w_off) / params.stride;
                        if oh < params.out_h && ow < params.out_w {
                            val = grad_out[n * params.out_channels * go_spatial + co * go_spatial + oh * params.out_w + ow];
                        }
                    }
                }
            }
            shared_b[row_local * $BM_U + col_local] = val;
        }

        workgroupBarrier();

        // Compute: 4×4 register-tiled matmul over KTILE=16
        for (var kk = 0u; kk < 16u; kk++) {
            $COMPUTE_BODY
        }

        workgroupBarrier();
        t += 16u;
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
