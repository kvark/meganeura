// Conv2d backward w.r.t. kernel: grad_output[N,Co,oH,oW] × input[N,Ci,H,W] → grad_kernel[Co,Ci,kH,kW]
// Dispatch: [Ci * kW, kH, Co]  workgroup_size(256)
// Each workgroup handles one kernel element (co, ci, kh, kw).
// 256 threads cooperatively accumulate over batch × oH × oW, then tree-reduce.

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

var<storage> grad_out: array<f32>;    // grad_output [N,Co,oH,oW]
var<storage> src: array<f32>;         // input [N,Ci,H,W]
var<storage, read_write> dst: array<f32>;  // grad_kernel [Co,Ci,kH,kW]
var<uniform> params: Params;
var<workgroup> wg_data: array<f32, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let cikw = wgid.x;  // ci * kernel_w + kw
    let kh = wgid.y;
    let co = wgid.z;

    let ci = cikw / params.kernel_w;
    let kw = cikw % params.kernel_w;

    if ci >= params.in_channels || kh >= params.kernel_h || co >= params.out_channels { return; }

    let tid = lid.x;
    let total = params.batch * params.out_h * params.out_w;

    // Each thread accumulates a strided portion of (n, oh, ow)
    var partial = 0.0;
    var idx = tid;
    loop {
        if idx >= total { break; }

        let n = idx / (params.out_h * params.out_w);
        let rem = idx % (params.out_h * params.out_w);
        let oh = rem / params.out_w;
        let ow = rem % params.out_w;

        let ih = i32(oh * params.stride + kh) - i32(params.padding_h);
        let iw = i32(ow * params.stride + kw) - i32(params.padding_w);

        if ih >= 0 && u32(ih) < params.in_h && iw >= 0 && u32(iw) < params.in_w {
            let go_idx = ((n * params.out_channels + co) * params.out_h + oh) * params.out_w + ow;
            let in_idx = ((n * params.in_channels + ci) * params.in_h + u32(ih)) * params.in_w + u32(iw);
            partial += grad_out[go_idx] * src[in_idx];
        }

        idx += 256u;
    }

    // Tree reduction
    wg_data[tid] = partial;
    workgroupBarrier();

    var stride = 128u;
    loop {
        if stride == 0u { break; }
        if tid < stride {
            wg_data[tid] += wg_data[tid + stride];
        }
        workgroupBarrier();
        stride >>= 1u;
    }

    if tid == 0u {
        let k_idx = ((co * params.in_channels + ci) * params.kernel_h + kh) * params.kernel_w + kw;
        dst[k_idx] = wg_data[0];
    }
}
