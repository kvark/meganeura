struct Params {
    total: u32,
    seq_len: u32,
    embed_dim: u32,
    _pad: u32,
}

var<storage> indices: array<u32>;
var<storage> src: array<f32>;
var<storage> row_scale: array<f32>;
var<storage, read_write> dst: array<atomic<u32>>;
var<uniform> params: Params;

// Blade enables Vulkan memory-model device scope when cooperative matrices
// require the Vulkan memory model, so storage CAS is valid on that path.
// `_pad == 2` maps one invocation to each narrow source row. `_pad > 2`
// encodes a scale-group width as `_pad - 2`; other modes map one invocation
// to each source element.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if params._pad == 2u {
        let source_row = gid.x;
        if source_row >= params.seq_len { return; }

        let scale = row_scale[source_row];
        if scale == 0.0 { return; }
        let output_row = indices[source_row];
        let output_rows = params.total / params.embed_dim;
        if output_row >= output_rows { return; }

        for (var column = 0u; column < params.embed_dim; column += 1u) {
            let value = src[source_row * params.embed_dim + column] * scale;
            if value == 0.0 { continue; }

            let output_index = output_row * params.embed_dim + column;
            var old_bits = atomicLoad(&dst[output_index]);
            loop {
                let old_value = bitcast<f32>(old_bits);
                let new_bits = bitcast<u32>(old_value + value);
                let result = atomicCompareExchangeWeak(&dst[output_index], old_bits, new_bits);
                if result.exchanged {
                    break;
                }
                old_bits = result.old_value;
            }
        }
        return;
    }

    let source_index = gid.x;
    let source_len = params.seq_len * params.embed_dim;
    if source_index >= source_len { return; }

    let source_row = source_index / params.embed_dim;
    var value = src[source_index];
    if params._pad == 1u {
        value *= row_scale[source_row];
    } else if params._pad > 2u {
        value *= row_scale[source_index / (params._pad - 2u)];
    }
    if value == 0.0 { return; }

    let column = source_index % params.embed_dim;
    let output_row = indices[source_row];
    let output_rows = params.total / params.embed_dim;
    if output_row >= output_rows { return; }

    let output_index = output_row * params.embed_dim + column;
    var old_bits = atomicLoad(&dst[output_index]);
    loop {
        let old_value = bitcast<f32>(old_bits);
        let new_bits = bitcast<u32>(old_value + value);
        let result = atomicCompareExchangeWeak(&dst[output_index], old_bits, new_bits);
        if result.exchanged {
            break;
        }
        old_bits = result.old_value;
    }
}
