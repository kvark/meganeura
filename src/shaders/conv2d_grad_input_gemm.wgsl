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
}

var<storage> grad_out: array<f32>;         // grad_output [N, Co, oH, oW]
var<storage> weight: array<f32>;           // kernel [Co, Ci, kH, kW]
var<storage, read_write> dst: array<f32>;  // grad_input [N, Ci, H, W]
var<uniform> params: Params;
var<workgroup> shared_a: array<f32, 1024>; // A tile: [64, 16]
var<workgroup> shared_b: array<f32, 1024>; // B tile: [16, 64]

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tx = lid.x;
    let ty = lid.y;
    let n = wgid.z;                // batch index
    let tile_row = wgid.y * 64u;   // M (Ci) tile start
    let tile_col = wgid.x * 64u;   // N (H*W) tile start
    let tid = ty * 16u + tx;

    let kernel_hw = params.kernel_h * params.kernel_w;
    let k_total = params.out_channels * kernel_hw;  // Co * kH * kW
    let n_total = params.in_h * params.in_w;        // H * W (grad_input spatial)
    let m_total = params.in_channels;               // Ci
    let go_spatial = params.out_h * params.out_w;    // oH * oW

    // Transposed padding for the "flipped convolution"
    let pad_h = i32(params.kernel_h) - 1 - i32(params.padding_h);
    let pad_w = i32(params.kernel_w) - 1 - i32(params.padding_w);

    // 16 accumulator registers (4×4 per thread)
    var s0_0 = 0.0; var s0_1 = 0.0; var s0_2 = 0.0; var s0_3 = 0.0;
    var s1_0 = 0.0; var s1_1 = 0.0; var s1_2 = 0.0; var s1_3 = 0.0;
    var s2_0 = 0.0; var s2_1 = 0.0; var s2_2 = 0.0; var s2_3 = 0.0;
    var s3_0 = 0.0; var s3_1 = 0.0; var s3_2 = 0.0; var s3_3 = 0.0;

    var t = 0u;
    loop {
        if t >= k_total { break; }

        // Load A tile: weight_T[Ci, K] → shared_a[64, 16]
        // weight_T[ci, co*kH*kW + kh*kW + kw] = weight[co, ci, kh, kw]
        // weight layout: [Co, Ci, kH, kW] → weight[co * Ci*kH*kW + ci * kH*kW + kh*kW + kw]
        for (var e = 0u; e < 4u; e++) {
            let flat = tid + e * 256u;
            let row_local = flat / 16u;  // M dimension (Ci)
            let col_local = flat % 16u;  // K dimension
            let ci = tile_row + row_local;
            let k_idx = t + col_local;

            var val = 0.0;
            if ci < m_total && k_idx < k_total {
                // Decompose k_idx → (co, kh, kw)
                let co = k_idx / kernel_hw;
                let k_rem = k_idx - co * kernel_hw;
                let kh = k_rem / params.kernel_w;
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
        for (var e = 0u; e < 4u; e++) {
            let flat = tid + e * 256u;
            let row_local = flat / 64u;  // K dimension (within KTILE=16)
            let col_local = flat % 64u;  // N dimension (H*W)
            let k_idx = t + row_local;
            let hw_idx = tile_col + col_local;

            var val = 0.0;
            if k_idx < k_total && hw_idx < n_total {
                let co = k_idx / kernel_hw;
                let k_rem = k_idx - co * kernel_hw;
                let kh = k_rem / params.kernel_w;
                let kw = k_rem - kh * params.kernel_w;
                let ih = hw_idx / params.in_w;
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
            shared_b[row_local * 64u + col_local] = val;
        }

        workgroupBarrier();

        // Compute: 4×4 register-tiled matmul over KTILE=16
        for (var kk = 0u; kk < 16u; kk++) {
            let a0 = shared_a[(ty * 4u + 0u) * 16u + kk];
            let a1 = shared_a[(ty * 4u + 1u) * 16u + kk];
            let a2 = shared_a[(ty * 4u + 2u) * 16u + kk];
            let a3 = shared_a[(ty * 4u + 3u) * 16u + kk];
            let b0 = shared_b[kk * 64u + tx * 4u + 0u];
            let b1 = shared_b[kk * 64u + tx * 4u + 1u];
            let b2 = shared_b[kk * 64u + tx * 4u + 2u];
            let b3 = shared_b[kk * 64u + tx * 4u + 3u];
            s0_0 += a0 * b0; s0_1 += a0 * b1; s0_2 += a0 * b2; s0_3 += a0 * b3;
            s1_0 += a1 * b0; s1_1 += a1 * b1; s1_2 += a1 * b2; s1_3 += a1 * b3;
            s2_0 += a2 * b0; s2_1 += a2 * b1; s2_2 += a2 * b2; s2_3 += a2 * b3;
            s3_0 += a3 * b0; s3_1 += a3 * b1; s3_2 += a3 * b2; s3_3 += a3 * b3;
        }

        workgroupBarrier();
        t += 16u;
    }

    // Store: grad_input[n, ci, ih*W+iw] in NCHW layout
    let output_stride = m_total * n_total;
    let s = array<array<f32, 4>, 4>(
        array<f32, 4>(s0_0, s0_1, s0_2, s0_3),
        array<f32, 4>(s1_0, s1_1, s1_2, s1_3),
        array<f32, 4>(s2_0, s2_1, s2_2, s2_3),
        array<f32, 4>(s3_0, s3_1, s3_2, s3_3),
    );
    for (var i = 0u; i < 4u; i++) {
        for (var j = 0u; j < 4u; j++) {
            let ci = tile_row + ty * 4u + i;
            let hw = tile_col + tx * 4u + j;
            if ci < m_total && hw < n_total {
                dst[n * output_stride + ci * n_total + hw] = s[i][j];
            }
        }
    }
}
