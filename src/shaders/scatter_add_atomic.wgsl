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

@compute @workgroup_size(256)
fn zero(@builtin(global_invocation_id) gid: vec3<u32>) {
    let index = gid.x;
    if index < params.total {
        atomicStore(&dst[index], 0u);
    }
}

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let source_index = gid.x;
    let source_len = params.seq_len * params.embed_dim;
    if source_index >= source_len { return; }

    // Sparse gradients are common when gathers are followed by masks or
    // activation gates. Avoid contending on the destination for updates that
    // cannot change it. This is especially important for padded sequences,
    // where many invocations would otherwise serialize while adding zero.
    let value = src[source_index];
    if value == 0.0 { return; }

    let source_row = source_index / params.embed_dim;
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

@compute @workgroup_size(256)
fn row_mul(@builtin(global_invocation_id) gid: vec3<u32>) {
    let source_index = gid.x;
    let source_len = params.seq_len * params.embed_dim;
    if source_index >= source_len { return; }

    let source_row = source_index / params.embed_dim;
    let value = row_scale[source_row] * src[source_index];
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
