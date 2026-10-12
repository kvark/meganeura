struct Params {
    seq: u32,
    dim: u32,
    theta_bits: u32,
    // Static offset, added to the buffer offset by the dynamic kernels.
    pos_offset: u32,
    axes_dim: vec3<u32>,
    head_dim: u32,
}
