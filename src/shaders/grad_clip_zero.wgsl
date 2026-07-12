// Zero a 1-element scalar accumulator buffer used as the global
// gradient-norm-squared accumulator for grad clipping.
// Run once per step() before any GradClipNormSq dispatches.
//
// `acc[0]` holds the f32 sum-of-squares.

var<storage, read_write> acc: array<f32>;

@compute @workgroup_size(1)
fn main() {
    acc[0] = 0.0;
}
