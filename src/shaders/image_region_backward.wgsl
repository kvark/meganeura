var<workgroup> scale_scratch: array<f32, 256>;

@compute @workgroup_size(256)
fn image(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let lane = local.x;
    let channel = lane / ROWS;
    let slot = lane % ROWS;
    let row = group.x * ROWS + slot;
    let valid = channel < CHANNELS && row < params.batch * params.out_h * params.out_w;
    var x = 0.0;
    var inv = 0.0;
    var weight = 0.0;
    var dn = 0.0;
    if valid {
        let at = image_index(row, channel);
        x = pooled[at];
        inv = inverse[row];
        weight = scale[channel];
        let normalized = (x * inv) * weight;
        let sigmoid = 1.0 / (1.0 + exp(-normalized));
        dn = dy[at] * (sigmoid + normalized * sigmoid * (1.0 - sigmoid));
    }
    scratch[lane] = (dn * weight) * x;
    workgroupBarrier();
    for (var stride = LANES / 2u; stride > 0u; stride /= 2u) {
        if channel < stride { scratch[lane] += scratch[lane + stride * ROWS]; }
        workgroupBarrier();
    }
    let correction = (inv * inv / f32(CHANNELS)) * scratch[slot];
    let dx = inv * (dn * weight - x * correction);
    workgroupBarrier();
    if valid {
        let at = image_index(row, channel);
        let chosen = $WINNER;
        for (var offset = 0u; offset < WINDOW * WINDOW; offset += 1u) {
            let value = select(0.0, dx, offset == chosen);
            dst[source_index(row, channel, offset)] = $STORE;
        }
    }
    scratch[lane] = dx;
    scale_scratch[lane] = (dn * x) * inv;
    workgroupBarrier();
    for (var stride = ROWS / 2u; stride > 0u; stride /= 2u) {
        if slot < stride {
            scratch[lane] += scratch[lane + stride];
            scale_scratch[lane] += scale_scratch[lane + stride];
        }
        workgroupBarrier();
    }
    if slot == 0u && channel < CHANNELS {
        partial_scale[group.x * CHANNELS + channel] = scale_scratch[lane];
        $BIAS_STORE
    }
}
