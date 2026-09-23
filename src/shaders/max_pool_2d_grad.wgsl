// MaxPool2d backward: route every output gradient to the input that won its
// window. A window's winner is its first maximum in row-major (kh, kw)
// order, as in PyTorch; ties go to the earlier element.
//
// One thread per input element gathers from the output windows covering
// it, recomputing each window's winner, so no atomics are needed and the
// result is deterministic.
//
// Dispatch: [ceil(N * C * H * W / 256), 1, 1]

struct Params {
    batch: u32,
    channels: u32,
    in_h: u32,
    in_w: u32,
    kernel_h: u32,
    kernel_w: u32,
    stride: u32,
    padding: u32,
    out_h: u32,
    out_w: u32,
    _pad0: u32,
    _pad1: u32,
}

var<storage> grad_out: array<f32>;
var<storage> src: array<f32>;
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;

// Flat in-plane index of the winner of output window (oh, ow).
fn window_winner(plane_in: u32, oh: u32, ow: u32) -> u32 {
    var best = -3.402823e+38;
    var winner = 0xffffffffu;
    for (var kh = 0u; kh < params.kernel_h; kh++) {
        let ih = oh * params.stride + kh - params.padding;
        if ih >= params.in_h { continue; }
        for (var kw = 0u; kw < params.kernel_w; kw++) {
            let iw = ow * params.stride + kw - params.padding;
            if iw >= params.in_w { continue; }
            let at = ih * params.in_w + iw;
            let value = src[plane_in + at];
            if winner == 0xffffffffu || value > best {
                best = value;
                winner = at;
            }
        }
    }
    return winner;
}

// Output positions o with o * stride - padding <= i < o * stride - padding + k.
fn covering(i: u32, k: u32, out_len: u32) -> vec2<u32> {
    let shifted = i + params.padding;
    let lo = select(0u, (shifted + params.stride - k) / params.stride, shifted + 1u > k);
    let hi = min(shifted / params.stride, out_len - 1u);
    return vec2<u32>(lo, hi);
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let plane_len = params.in_h * params.in_w;
    let total = params.batch * params.channels * plane_len;
    if idx >= total { return; }

    let plane = idx / plane_len;
    let at = idx - plane * plane_len;
    let ih = at / params.in_w;
    let iw = at - ih * params.in_w;
    let plane_in = plane * plane_len;
    let plane_out = plane * params.out_h * params.out_w;

    let rows = covering(ih, params.kernel_h, params.out_h);
    let cols = covering(iw, params.kernel_w, params.out_w);
    var acc = 0.0;
    for (var oh = rows.x; oh <= rows.y; oh++) {
        for (var ow = cols.x; ow <= cols.y; ow++) {
            if window_winner(plane_in, oh, ow) == at {
                acc += grad_out[plane_out + oh * params.out_w + ow];
            }
        }
    }
    dst[idx] = acc;
}
