@compute @workgroup_size(256)
fn image(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let lane = local.x;
    let channel = lane / ROWS;
    let slot = lane % ROWS;
    let row = group.x * ROWS + slot;
    let valid = channel < CHANNELS && row < params.batch * params.out_h * params.out_w;
    var value = 0.0;
    var chosen = 0u;
    if valid {
        let b = $BIAS;
        value = f32(src[source_index(row, channel, 0u)]) + b;
        var maximum = max(-3.402823e+38, value);
        for (var offset = 1u; offset < WINDOW * WINDOW; offset += 1u) {
            let sample = f32(src[source_index(row, channel, offset)]) + b;
            maximum = max(maximum, sample);
            if sample > value { value = sample; chosen = offset; }
        }
        if WINDOW > 1u { value = maximum; }
    }
    scratch[lane] = value * value;
    workgroupBarrier();
    for (var stride = LANES / 2u; stride > 0u; stride /= 2u) {
        if channel < stride { scratch[lane] += scratch[lane + stride * ROWS]; }
        workgroupBarrier();
    }
    let inv = inverseSqrt(scratch[slot] * (1.0 / f32(CHANNELS)) + bitcast<f32>(params.eps));
    if valid {
        let at = image_index(row, channel);
        let normalized = (value * inv) * scale[channel];
        dst[at] = normalized / (1.0 + exp(-normalized));
        pooled[at] = value;
        $SAVE_WINNER
        if channel == 0u { inverse[row] = inv; }
    }
}
