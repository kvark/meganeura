struct Params {
    m: u32,
    n: u32,
    _pad0: u32,
    _pad1: u32,
}

var<storage> src: array<f32>;
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;
var<workgroup> partials: array<f32, 256>;

// Sum across rows: 32 adjacent columns per workgroup, with 8 row lanes per
// column. Threads along X issue contiguous reads while Y supplies reduction
// parallelism for tall matrices.
@compute @workgroup_size(32, 8)
fn sum_rows(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let col = wgid.x * 32u + lid.x;
    var acc = 0.0;
    if col < params.n {
        for (var row = lid.y; row < params.m; row += 8u) {
            acc += src[row * params.n + col];
        }
    }

    let index = lid.y * 32u + lid.x;
    partials[index] = acc;
    workgroupBarrier();

    if lid.y < 4u {
        partials[index] += partials[index + 128u];
    }
    workgroupBarrier();
    if lid.y < 2u {
        partials[index] += partials[index + 64u];
    }
    workgroupBarrier();
    if lid.y == 0u && col < params.n {
        dst[col] = partials[lid.x] + partials[32u + lid.x];
    }
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
