// Zero a 1-element scalar accumulator buffer used as the global
// gradient-norm-squared accumulator for grad clipping.
// Run once per step() before any GradClipNormSq dispatches.
//
// `acc[0]` holds f32 sum-of-squares stored as u32 bits; writing 0u is
// equivalent to writing 0.0f (positive-zero).

var<storage, read_write> acc: array<u32>;

@compute @workgroup_size(1)
fn main() {
    acc[0] = 0u;
}
