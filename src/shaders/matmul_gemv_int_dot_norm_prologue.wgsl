    // Prologue: sum of squares over A, reduced across the workgroup.
    var ss = 0.0;
    var si = lane;
    loop {
        if si >= k { break; }
        let v = matrix_a[si];
        ss += v * v;
        si += LANES;
    }
    scale_buf[lane] = ss;
    workgroupBarrier();
    var sstride = LANES / 2u;
    loop {
        if sstride == 0u { break; }
        if lane < sstride {
            scale_buf[lane] += scale_buf[lane + sstride];
        }
        workgroupBarrier();
        sstride >>= 1u;
    }
    if lane == 0u {
        inv_rms = inverseSqrt(scale_buf[0] / f32(k) + bitcast<f32>(params.eps_bits));
    }
    workgroupBarrier();
