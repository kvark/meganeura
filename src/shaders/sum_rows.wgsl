struct Params {
    m: u32,
    n: u32,
    _pad0: u32,
    _pad1: u32,
}

var<storage> src: array<f32>;
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;

// Sum across rows: dst[col] = sum over row of src[row * n + col]
@compute @workgroup_size(256)
fn sum_rows(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    if col >= params.n { return; }
    var acc = 0.0;
    for (var row = 0u; row < params.m; row++) {
        acc += src[row * params.n + col];
    }
    dst[col] = acc;
}

// Exclusive cumulative sum along each row. `_pad0 != 0` selects reverse
// order, which is the transpose operation used by autodiff.
@compute @workgroup_size(64)
fn exclusive_cumsum(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if row >= params.m { return; }

    let base = row * params.n;
    var acc = 0.0;
    for (var offset = 0u; offset < params.n; offset++) {
        let col = select(offset, params.n - 1u - offset, params._pad0 != 0u);
        let index = base + col;
        let value = src[index];
        dst[index] = acc;
        acc += value;
    }
}

// Shift each row by the signed offset encoded in `_pad0`.
@compute @workgroup_size(256)
fn shift_inner(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    let len = params.m * params.n;
    if index >= len { return; }

    let row = index / params.n;
    let col = i32(index % params.n);
    let source_col = col - bitcast<i32>(params._pad0);
    if source_col >= 0 && source_col < i32(params.n) {
        dst[index] = src[row * params.n + u32(source_col)];
    } else {
        dst[index] = 0.0;
    }
}
