// Conv2d backward w.r.t. kernel via implicit GEMM — small-tile variant.
// BM=32, BN=32, KTILE=16, TM=2, TN=2, workgroup [16,16,1]
// 4× more workgroups than the 64×64 variant for better occupancy on small matrices.
//
// Dispatch: [ceil(Ci*kH*kW / 32), ceil(Co / 32), 1]

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

var<storage> grad_out: array<f32>;
var<storage> src: array<f32>;
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;
var<workgroup> shared_a: array<f32, 512>;  // 32 * 16
var<workgroup> shared_b: array<f32, 512>;  // 16 * 32

@compute @workgroup_size(16, 16)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let tx = lid.x;
    let ty = lid.y;
    let tile_row = wgid.y * 32u;   // M (Co) tile start
    let tile_col = wgid.x * 32u;   // N (Ci*kH*kW) tile start
    let tid = ty * 16u + tx;

    let m_total = params.out_channels;
    let kernel_hw = params.kernel_h * params.kernel_w;
    let n_total = params.in_channels * kernel_hw;
    let go_spatial = params.out_h * params.out_w;
    let k_total = params.batch * go_spatial;
    let input_spatial = params.in_h * params.in_w;

    var s0_0 = 0.0; var s0_1 = 0.0;
    var s1_0 = 0.0; var s1_1 = 0.0;

    var t = 0u;
    loop {
        if t >= k_total { break; }

        // Load A tile: grad_out_flat[Co, N*oH*oW] → shared_a[32, 16]
        for (var e = 0u; e < 2u; e++) {
            let flat = tid + e * 256u;
            let row_local = flat / 16u;
            let col_local = flat % 16u;
            let co = tile_row + row_local;
            let k_idx = t + col_local;

            var val = 0.0;
            if co < m_total && k_idx < k_total {
                let n = u32(f32(k_idx) * params.inv_go_spatial);
                let rem = k_idx - n * go_spatial;
                let oh = u32(f32(rem) * params.inv_col_w);
                let ow = rem - oh * params.out_w;
                val = grad_out[((n * params.out_channels + co) * params.out_h + oh) * params.out_w + ow];
            }
            shared_a[row_local * 16u + col_local] = val;
        }

        // Load B tile: im2col(input)[N*oH*oW, Ci*kH*kW] → shared_b[16, 32]
        for (var e = 0u; e < 2u; e++) {
            let flat = tid + e * 256u;
            let row_local = flat / 32u;
            let col_local = flat % 32u;
            let k_idx = t + row_local;
            let col_idx = tile_col + col_local;

            var val = 0.0;
            if k_idx < k_total && col_idx < n_total {
                let n = u32(f32(k_idx) * params.inv_go_spatial);
                let rem = k_idx - n * go_spatial;
                let oh = u32(f32(rem) * params.inv_col_w);
                let ow = rem - oh * params.out_w;
                let ci = u32(f32(col_idx) * params.inv_kernel_hw);
                let k_rem = col_idx - ci * kernel_hw;
                let kh = u32(f32(k_rem) * params.inv_kernel_w);
                let kw = k_rem - kh * params.kernel_w;
                let ih = i32(oh * params.stride + kh) - i32(params.padding_h);
                let iw = i32(ow * params.stride + kw) - i32(params.padding_w);
                if ih >= 0 && u32(ih) < params.in_h && iw >= 0 && u32(iw) < params.in_w {
                    val = src[((n * params.in_channels + ci) * params.in_h + u32(ih)) * params.in_w + u32(iw)];
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

    let s = array<array<f32, 2>, 2>(
        array<f32, 2>(s0_0, s0_1),
        array<f32, 2>(s1_0, s1_1),
    );
    for (var i = 0u; i < 2u; i++) {
        for (var j = 0u; j < 2u; j++) {
            let co = tile_row + ty * 2u + i;
            let cikk = tile_col + tx * 2u + j;
            if co < m_total && cikk < n_total {
                dst[co * n_total + cikk] = s[i][j];
            }
        }
    }
}
