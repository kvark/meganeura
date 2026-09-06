// Conv2d forward via implicit GEMM: output = weight @ im2col(input)^T
//
// Computes C[Co, oH*oW] = A[Co, K] × B[K, oH*oW] per batch item,
// where K = Ci*kH*kW and B is the im2col matrix computed on-the-fly.
//
// Uses the same 64×64 register-tiled matmul as matmul.wgsl:
//   BM=64, BN=64, KTILE=16, TM=4, TN=4, workgroup [16,16,1]
//
// Dispatch: [ceil(oH*oW / 64), ceil(Co / 64), batch]

$DIVISOR

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
    kernel_w_multiplier: u32,
    kernel_hw_multiplier: u32,
    column_width_multiplier: u32,
    output_spatial_multiplier: u32,
}

var<storage> src: array<f32>;              // input [N, Ci, H, W]
var<storage> weight: array<f32>;           // kernel [Co, Ci, kH, kW] = [Co, K]
var<storage, read_write> dst: array<f32>;  // output [N, Co, oH, oW]
var<uniform> params: Params;
var<workgroup> shared_a: array<f32, $SHARED_SIZE>; // A tile: [64, 16]
var<workgroup> shared_b: array<f32, $SHARED_SIZE>; // B tile: [16, 64]

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tx = lid.x;
    let ty = lid.y;
    let n = wgid.z;           // batch index
    let tile_row = wgid.y * $BM_U;  // M (Co) tile start
    let tile_col = wgid.x * $BM_U;  // N (oH*oW) tile start
    let tid = ty * 16u + tx;

    let k_total = params.in_channels * params.kernel_h * params.kernel_w;
    let n_total = params.out_h * params.out_w;
    let m_total = params.out_channels;
    let input_stride = params.in_channels * params.in_h * params.in_w;  // per-batch input size
    let kernel_hw = params.kernel_h * params.kernel_w;

    $ACC_DECL

    var t = 0u;
    loop {
        if t >= k_total { break; }

        // Load A tile: weight[Co, K] → shared_a[64, 16]
        // 4 elements per thread (256 threads × 4 = 1024)
        for (var e = 0u; e < $STAGE_EPT_U; e++) {
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
        for (var e = 0u; e < $STAGE_EPT_U; e++) {
            let flat = tid + e * 256u;
            let row_local = flat / $BM_U;  // K dimension (within KTILE=16)
            let col_local = flat % $BM_U;  // N dimension (oH*oW)
            let k_idx = t + row_local;
            let hw_idx = tile_col + col_local;

            var val = 0.0;
            if k_idx < k_total && hw_idx < n_total {
                // Decompose k_idx → (ci, kh, kw)
                let ci = divide_exact(k_idx, kernel_hw, params.kernel_hw_multiplier);
                let k_rem = k_idx - ci * kernel_hw;
                let kh = divide_exact(k_rem, params.kernel_w, params.kernel_w_multiplier);
                let kw = k_rem - kh * params.kernel_w;
                // Decompose hw_idx → (oh, ow)
                let oh = divide_exact(hw_idx, params.out_w, params.column_width_multiplier);
                let ow = hw_idx - oh * params.out_w;
                // Input position
                let ih = i32(oh * params.stride + kh) - i32(params.padding_h);
                let iw = i32(ow * params.stride + kw) - i32(params.padding_w);
                if ih >= 0 && u32(ih) < params.in_h && iw >= 0 && u32(iw) < params.in_w {
                    val = src[n * input_stride + ci * params.in_h * params.in_w + u32(ih) * params.in_w + u32(iw)];
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

    // Store: output[n, co, oh*oW+ow] in NCHW layout
    let output_stride = m_total * n_total;  // Co * oH * oW per batch
    let s = $ACC_ARRAY;
    for (var i = 0u; i < $TM_U; i++) {
        for (var j = 0u; j < $TM_U; j++) {
            let co = tile_row + ty * $TM_U + i;
            let hw = tile_col + tx * $TM_U + j;
            if co < m_total && hw < n_total {
                dst[n * output_stride + co * n_total + hw] = s[i][j];
            }
        }
    }
}
