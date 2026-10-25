@compute @workgroup_size(256)
fn pool(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    let spatial = params.out_h * params.out_w;
    if idx >= params.batch * params.channels * spatial { return; }
    let plane = idx / spatial;
    let oh = (idx / params.out_w) % params.out_h;
    let ow = idx % params.out_w;
    var max_val = -3.402823e+38;
    var best = -3.402823e+38;
    var selected = 0xffffffffu;
    for (var kh = 0u; kh < params.kernel_h; kh++) {
        for (var kw = 0u; kw < params.kernel_w; kw++) {
            let ih = oh * params.stride + kh - params.padding;
            let iw = ow * params.stride + kw - params.padding;
            if ih < params.in_h && iw < params.in_w {
                let at = (plane * params.in_h + ih) * params.in_w + iw;
                let value = sample(at);
                max_val = max(max_val, value);
                if selected == 0xffffffffu || value > best {
                    best = value;
                    selected = at;
                }
            }
        }
    }
    dst[idx] = max_val;
    winner[idx] = selected;
}
