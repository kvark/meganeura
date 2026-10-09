    // Prologue: sum of squares over A, reduced across the workgroup.
    var ss = 0.0;
    var si = lid.x;
    loop {
        if si >= k { break; }
        let v = $NORM_VALUE;
        ss += v * v;
        si += $WORKGROUP_SIZEu;
    }
    scale_buf[lid.x] = ss;
    workgroupBarrier();
    var sstride = $WORKGROUP_SIZEu / 2u;
    loop {
        if sstride == 0u { break; }
        if lid.x < sstride {
            scale_buf[lid.x] += scale_buf[lid.x + sstride];
        }
        workgroupBarrier();
        sstride >>= 1u;
    }
    if lid.x == 0u {
        inv_rms = inverseSqrt(scale_buf[0] / f32(k) + bitcast<f32>(params.eps_bits));
    }
    workgroupBarrier();
