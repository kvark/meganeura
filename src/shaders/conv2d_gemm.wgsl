// Conv2d forward via implicit GEMM: output = weight @ im2col(input)^T
//
// Computes C[Co, oH*oW] = A[Co, K] × B[K, oH*oW] per batch item,
// where K = Ci*kH*kW and B is the im2col matrix computed on-the-fly.
//
// Uses the same 64×64 register-tiled matmul as matmul.wgsl:
//   BM=64, BN=64, KTILE=16, TM=4, TN=4, workgroup [16,16,1]
//
// Dispatch: [ceil(oH*oW / 64), ceil(Co / 64), batch]

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
var<storage> weight: array<f32>;           // kernel [Co, Ci, kH, kW] = [Co, K]
var<storage, read_write> dst: array<f32>;  // output [N, Co, oH, oW]
var<uniform> params: Params;
var<workgroup> shared_a: array<f32, 1024>; // A tile: [64, 16]
var<workgroup> shared_b: array<f32, 1024>; // B tile: [16, 64]

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tx = lid.x;
    let ty = lid.y;
    let n = wgid.z;           // batch index
    let tile_row = wgid.y * 64u;  // M (Co) tile start
    let tile_col = wgid.x * 64u;  // N (oH*oW) tile start
    let tid = ty * 16u + tx;

    let k_total = params.in_channels * params.kernel_h * params.kernel_w;
    let n_total = params.out_h * params.out_w;
    let m_total = params.out_channels;
    let input_stride = params.in_channels * params.in_h * params.in_w;  // per-batch input size
    let kernel_hw = params.kernel_h * params.kernel_w;

    // 16 accumulator registers (4×4 per thread)
    var s0_0 = 0.0; var s0_1 = 0.0; var s0_2 = 0.0; var s0_3 = 0.0;
    var s1_0 = 0.0; var s1_1 = 0.0; var s1_2 = 0.0; var s1_3 = 0.0;
    var s2_0 = 0.0; var s2_1 = 0.0; var s2_2 = 0.0; var s2_3 = 0.0;
    var s3_0 = 0.0; var s3_1 = 0.0; var s3_2 = 0.0; var s3_3 = 0.0;

    var t = 0u;
    loop {
        if t >= k_total { break; }

        // Load A tile: weight[Co, K] → shared_a[64, 16]
        // 4 elements per thread (256 threads × 4 = 1024)
        for (var e = 0u; e < 4u; e++) {
            let flat = tid + e * 256u;
            let row_local = flat / 16u;  // M dimension (Co)
            let col_local = flat % 16u;  // K dimension
            let a_row = tile_row + row_local;
            let a_col = t + col_local;
            let in_bounds_a = a_row < m_total && a_col < k_total;
            shared_a[row_local * 16u + col_local] = select(0.0, weight[a_row * k_total + a_col], in_bounds_a);
        }

        // Load B tile: im2col(input)^T [K, oH*oW] → shared_b[16, 64]
        // B[k, hw] = input[n, ci, oh*stride+kh-pad, ow*stride+kw-pad]
        for (var e = 0u; e < 4u; e++) {
            let flat = tid + e * 256u;
            let row_local = flat / 64u;  // K dimension (within KTILE=16)
            let col_local = flat % 64u;  // N dimension (oH*oW)
            let k_idx = t + row_local;
            let hw_idx = tile_col + col_local;

            var val = 0.0;
            if k_idx < k_total && hw_idx < n_total {
                // Decompose k_idx → (ci, kh, kw)
                let ci = k_idx / kernel_hw;
                let k_rem = k_idx - ci * kernel_hw;
                let kh = k_rem / params.kernel_w;
                let kw = k_rem - kh * params.kernel_w;
                // Decompose hw_idx → (oh, ow)
                let oh = hw_idx / params.out_w;
                let ow = hw_idx - oh * params.out_w;
                // Input position
                let ih = i32(oh * params.stride + kh) - i32(params.padding_h);
                let iw = i32(ow * params.stride + kw) - i32(params.padding_w);
                if ih >= 0 && u32(ih) < params.in_h && iw >= 0 && u32(iw) < params.in_w {
                    val = src[n * input_stride + ci * params.in_h * params.in_w + u32(ih) * params.in_w + u32(iw)];
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

    // Store: output[n, co, oh*oW+ow] in NCHW layout
    let output_stride = m_total * n_total;  // Co * oH * oW per batch
    let s = array<array<f32, 4>, 4>(
        array<f32, 4>(s0_0, s0_1, s0_2, s0_3),
        array<f32, 4>(s1_0, s1_1, s1_2, s1_3),
        array<f32, 4>(s2_0, s2_1, s2_2, s2_3),
        array<f32, 4>(s3_0, s3_1, s3_2, s3_3),
    );
    for (var i = 0u; i < 4u; i++) {
        for (var j = 0u; j < 4u; j++) {
            let co = tile_row + ty * 4u + i;
            let hw = tile_col + tx * 4u + j;
            if co < m_total && hw < n_total {
                dst[n * output_stride + co * n_total + hw] = s[i][j];
            }
        }
    }
}
