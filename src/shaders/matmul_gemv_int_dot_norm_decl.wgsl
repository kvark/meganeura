var<storage> norm_w: array<f32>;
var<workgroup> scale_buf: array<f32, $WORKGROUP_SIZEu>;
var<workgroup> inv_rms: f32;
