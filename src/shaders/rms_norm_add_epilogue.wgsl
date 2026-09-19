var<storage> residual: array<f32>;

fn apply_rms_norm_epilogue(index: u32, value: f32) -> f32 {
    return value + residual[index];
}
