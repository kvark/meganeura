// Conv2d backward w.r.t. input via implicit GEMM — small-tile variant.
// BM=32, BN=32, KTILE=16, TM=2, TN=2, workgroup [16,16,1]
// 4× more workgroups than the 64×64 variant for better occupancy on small matrices.
//
// Dispatch: [ceil(H*W / 32), ceil(Ci / 32), batch]

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

    let kernel_hw = params.kernel_h * params.kernel_w;
    let k_total = params.out_channels * kernel_hw;
    let n_total = params.in_h * params.in_w;
    let m_total = params.in_channels;
    let go_spatial = params.out_h * params.out_w;

    let pad_h = i32(params.kernel_h) - 1 - i32(params.padding_h);
    let pad_w = i32(params.kernel_w) - 1 - i32(params.padding_w);

    var s0_0 = 0.0; var s0_1 = 0.0;
    var s1_0 = 0.0; var s1_1 = 0.0;

    var t = 0u;
    loop {
        if t >= k_total { break; }

        // Load A tile: weight_T[Ci, K] → shared_a[32, 16]
        for (var e = 0u; e < 2u; e++) {
            let flat = tid + e * 256u;
            let row_local = flat / 16u;
            let col_local = flat % 16u;
            let ci = tile_row + row_local;
            let k_idx = t + col_local;

            var val = 0.0;
            if ci < m_total && k_idx < k_total {
                let co = u32(f32(k_idx) * params.inv_kernel_hw);
                let k_rem = k_idx - co * kernel_hw;
                let kh = u32(f32(k_rem) * params.inv_kernel_w);
                let kw = k_rem - kh * params.kernel_w;
                val = weight[(co * m_total + ci) * kernel_hw + kh * params.kernel_w + kw];
            }
            shared_a[row_local * 16u + col_local] = val;
        }

        // Load B tile: im2col(grad_out)^T [K, H*W] → shared_b[16, 32]
        for (var e = 0u; e < 2u; e++) {
            let flat = tid + e * 256u;
            let row_local = flat / 32u;
            let col_local = flat % 32u;
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
                    let oh = i32(ih) + pad_h - i32(kh);
                    let ow = i32(iw) + pad_w - i32(kw);
                    if oh >= 0 && u32(oh) < params.out_h && ow >= 0 && u32(ow) < params.out_w {
                        val = grad_out[n * params.out_channels * go_spatial + co * go_spatial + u32(oh) * params.out_w + u32(ow)];
                    }
                } else {
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
            let ci = tile_row + ty * 2u + i;
            let hw = tile_col + tx * 2u + j;
            if ci < m_total && hw < n_total {
                dst[n * output_stride + ci * n_total + hw] = s[i][j];
            }
        }
    }
}
