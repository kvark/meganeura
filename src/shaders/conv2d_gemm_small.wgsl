// Conv2d forward via implicit GEMM — small-tile variant.
// BM=32, BN=32, KTILE=16, TM=2, TN=2, workgroup [16,16,1]
// 4× more workgroups than the 64×64 variant for better occupancy on small matrices.
//
// Dispatch: [ceil(oH*oW / 32), ceil(Co / 32), batch]

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

var<storage> src: array<f32>;
var<storage> weight: array<f32>;
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;
var<workgroup> shared_a: array<f32, 512>;  // 32 * 16
var<workgroup> shared_b: array<f32, 512>;  // 16 * 32

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tx = lid.x;
    let ty = lid.y;
    let n = wgid.z;
    let tile_row = wgid.y * 32u;
    let tile_col = wgid.x * 32u;
    let tid = ty * 16u + tx;

    let k_total = params.in_channels * params.kernel_h * params.kernel_w;
    let n_total = params.out_h * params.out_w;
    let m_total = params.out_channels;
    let input_stride = params.in_channels * params.in_h * params.in_w;
    let kernel_hw = params.kernel_h * params.kernel_w;

    var s0_0 = 0.0; var s0_1 = 0.0;
    var s1_0 = 0.0; var s1_1 = 0.0;

    var t = 0u;
    loop {
        if t >= k_total { break; }

        // Load A tile: weight[Co, K] → shared_a[32, 16]
        for (var e = 0u; e < 2u; e++) {
            let flat = tid + e * 256u;
            let row_local = flat / 16u;
            let col_local = flat % 16u;
            let a_row = tile_row + row_local;
            let a_col = t + col_local;
            let ib = a_row < m_total && a_col < k_total;
            shared_a[row_local * 16u + col_local] = select(0.0, weight[a_row * k_total + a_col], ib);
        }

        // Load B tile: im2col(input)^T [K, oH*oW] → shared_b[16, 32]
        for (var e = 0u; e < 2u; e++) {
            let flat = tid + e * 256u;
            let row_local = flat / 32u;
            let col_local = flat % 32u;
            let k_idx = t + row_local;
            let hw_idx = tile_col + col_local;

            var val = 0.0;
            if k_idx < k_total && hw_idx < n_total {
                let ci = k_idx / kernel_hw;
                let k_rem = k_idx - ci * kernel_hw;
                let kh = k_rem / params.kernel_w;
                let kw = k_rem - kh * params.kernel_w;
                let oh = hw_idx / params.out_w;
                let ow = hw_idx - oh * params.out_w;
                let ih = i32(oh * params.stride + kh) - i32(params.padding_h);
                let iw = i32(ow * params.stride + kw) - i32(params.padding_w);
                if ih >= 0 && u32(ih) < params.in_h && iw >= 0 && u32(iw) < params.in_w {
                    val = src[n * input_stride + ci * params.in_h * params.in_w + u32(ih) * params.in_w + u32(iw)];
                }
            }
            shared_b[row_local * 32u + col_local] = val;
        }

        workgroupBarrier();

        for (var kk = 0u; kk < 16u; kk++) {
            let a0 = shared_a[(ty * 2u + 0u) * 16u + kk];
            let a1 = shared_a[(ty * 2u + 1u) * 16u + kk];
            let b0 = shared_b[kk * 32u + tx * 2u + 0u];
            let b1 = shared_b[kk * 32u + tx * 2u + 1u];
            s0_0 += a0 * b0; s0_1 += a0 * b1;
            s1_0 += a1 * b0; s1_1 += a1 * b1;
        }

        workgroupBarrier();
        t += 16u;
    }

    let output_stride = m_total * n_total;
    let s = array<array<f32, 2>, 2>(
        array<f32, 2>(s0_0, s0_1),
        array<f32, 2>(s1_0, s1_1),
    );
    for (var i = 0u; i < 2u; i++) {
        for (var j = 0u; j < 2u; j++) {
            let co = tile_row + ty * 2u + i;
            let hw = tile_col + tx * 2u + j;
            if co < m_total && hw < n_total {
                dst[n * output_stride + co * n_total + hw] = s[i][j];
            }
        }
    }
}
