struct Params {
    total: u32,
    seq_len: u32,
    embed_dim: u32,
    _pad: u32,
}

var<storage> indices: array<u32>;
var<storage> src: array<f32>;
var<storage> row_scale: array<f32>;
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;

// Column-parallel scatter-add. Each thread owns one embedding column and
// walks the source rows sequentially, so duplicate indices accumulate
// correctly without storage atomics.
//
// The previous CAS loop used device-scope `atomic<u32>` on a storage
// buffer. Blade enables `vulkanMemoryModel` without
// `vulkanMemoryModelDeviceScope`, which is VUID 06265 on cooperative-
// matrix Vulkan contexts (the same hole grad-clip closed).
//
// Parallelism is `embed_dim` rather than `seq_len * embed_dim`. That is
// enough for the embedding widths we train (768–4096) and is portable
// to Metal and strict Vulkan.
//
// `_pad` 0: unscaled. `_pad` 1 or 2: multiply by `row_scale[s]` (the
// two values are historical work-mapping modes; they are the same math).
// Dispatch: [ceil(embed_dim / 256), 1, 1]
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let col = gid.x;
    if col >= params.embed_dim { return; }

    let output_rows = params.total / params.embed_dim;
    let scaled = params._pad != 0u;
    for (var s = 0u; s < params.seq_len; s++) {
        var value = src[s * params.embed_dim + col];
        if scaled {
            value *= row_scale[s];
        }
        if value == 0.0 { continue; }
        let output_row = indices[s];
        if output_row >= output_rows { continue; }
        dst[output_row * params.embed_dim + col] += value;
    }
}
