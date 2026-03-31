// Conv2d backward w.r.t. kernel via implicit GEMM.
//
// grad_weight[Co, Ci*kH*kW] = grad_out_flat[Co, N*oH*oW] × im2col(input)[N*oH*oW, Ci*kH*kW]
// C[Co, Ci*kH*kW] = A[Co, K] × B[K, Ci*kH*kW], K = batch*oH*oW.
//
// BM=64, BN=64, KTILE=16, TM=4, TN=4, workgroup [16,16,1]
// Dispatch: [ceil(Ci*kH*kW / 64), ceil(Co / 64), 1]

struct Params {
    batch: u32,
    in_channels: u32,
    in_h: u32,
    in_w: u32,
    out_channels: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride: u32,
    padding: u32,
    out_h: u32,
    out_w: u32,
    _pad: u32,
}

var<storage> grad_out: array<f32>;           // [N, Co, oH, oW]
var<storage> src: array<f32>;                // input [N, Ci, H, W]
var<storage, read_write> dst: array<f32>;    // grad_kernel [Co, Ci, kH, kW]
var<uniform> params: Params;
var<workgroup> shared_a: array<f32, 1024>;   // A tile: [64, 16]
var<workgroup> shared_b: array<f32, 1024>;   // B tile: [16, 64]

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tx = lid.x;
    let ty = lid.y;
    let tile_row = wgid.y * 64u;   // M (Co) tile start
    let tile_col = wgid.x * 64u;   // N (Ci*kH*kW) tile start
    let tid = ty * 16u + tx;

    let m_total = params.out_channels;                   // Co
    let kernel_hw = params.kernel_h * params.kernel_w;
    let n_total = params.in_channels * kernel_hw;        // Ci*kH*kW
    let go_spatial = params.out_h * params.out_w;        // oH*oW
    let k_total = params.batch * go_spatial;             // N*oH*oW
    let input_spatial = params.in_h * params.in_w;

    // 16 accumulator registers (4×4 per thread)
    var s0_0 = 0.0; var s0_1 = 0.0; var s0_2 = 0.0; var s0_3 = 0.0;
    var s1_0 = 0.0; var s1_1 = 0.0; var s1_2 = 0.0; var s1_3 = 0.0;
    var s2_0 = 0.0; var s2_1 = 0.0; var s2_2 = 0.0; var s2_3 = 0.0;
    var s3_0 = 0.0; var s3_1 = 0.0; var s3_2 = 0.0; var s3_3 = 0.0;

    var t = 0u;
    loop {
        if t >= k_total { break; }

        // Load A tile: grad_out_flat[Co, N*oH*oW] → shared_a[64, 16]
        // A[co, n*oH*oW + oh*oW + ow] = grad_out[n, co, oh, ow]
        for (var e = 0u; e < 4u; e++) {
            let flat = tid + e * 256u;
            let row_local = flat / 16u;  // M dimension (Co)
            let col_local = flat % 16u;  // K dimension
            let co = tile_row + row_local;
            let k_idx = t + col_local;

            var val = 0.0;
            if co < m_total && k_idx < k_total {
                let n = k_idx / go_spatial;
                let rem = k_idx - n * go_spatial;
                let oh = rem / params.out_w;
                let ow = rem - oh * params.out_w;
                val = grad_out[((n * params.out_channels + co) * params.out_h + oh) * params.out_w + ow];
            }
            shared_a[row_local * 16u + col_local] = val;
        }

        // Load B tile: im2col(input)[N*oH*oW, Ci*kH*kW] → shared_b[16, 64]
        // B[k_idx, col] where k_idx = n*oH*oW + oh*oW + ow, col = ci*kH*kW + kh*kW + kw
        // B[k_idx, col] = input[n, ci, oh*stride+kh-padding, ow*stride+kw-padding]
        for (var e = 0u; e < 4u; e++) {
            let flat = tid + e * 256u;
            let row_local = flat / 64u;  // K dimension (within KTILE=16)
            let col_local = flat % 64u;  // N dimension (Ci*kH*kW)
            let k_idx = t + row_local;
            let col_idx = tile_col + col_local;

            var val = 0.0;
            if k_idx < k_total && col_idx < n_total {
                // Decompose k_idx → (n, oh, ow)
                let n = k_idx / go_spatial;
                let rem = k_idx - n * go_spatial;
                let oh = rem / params.out_w;
                let ow = rem - oh * params.out_w;
                // Decompose col_idx → (ci, kh, kw)
                let ci = col_idx / kernel_hw;
                let k_rem = col_idx - ci * kernel_hw;
                let kh = k_rem / params.kernel_w;
                let kw = k_rem - kh * params.kernel_w;
                // Input position
                let ih = i32(oh * params.stride + kh) - i32(params.padding);
                let iw = i32(ow * params.stride + kw) - i32(params.padding);
                if ih >= 0 && u32(ih) < params.in_h && iw >= 0 && u32(iw) < params.in_w {
                    val = src[((n * params.in_channels + ci) * params.in_h + u32(ih)) * params.in_w + u32(iw)];
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

    // Store: grad_kernel[co, ci*kH*kW + kh*kW + kw]
    // Output layout: [Co, Ci, kH, kW] = [Co, Ci*kH*kW] row-major
    let s = array<array<f32, 4>, 4>(
        array<f32, 4>(s0_0, s0_1, s0_2, s0_3),
        array<f32, 4>(s1_0, s1_1, s1_2, s1_3),
        array<f32, 4>(s2_0, s2_1, s2_2, s2_3),
        array<f32, 4>(s3_0, s3_1, s3_2, s3_3),
    );
    for (var i = 0u; i < 4u; i++) {
        for (var j = 0u; j < 4u; j++) {
            let co = tile_row + ty * 4u + i;
            let cikk = tile_col + tx * 4u + j;
            if co < m_total && cikk < n_total {
                dst[co * n_total + cikk] = s[i][j];
            }
        }
    }
}
