struct Params {
    // `main` interprets these as M and N. The window entry points interpret
    // them as batch, channels, height, width, window, and shift. Keeping one
    // common uniform layout lets all pure rearrangements share this module.
    dim0: u32,
    dim1: u32,
    dim2: u32,
    dim3: u32,
    dim4: u32,
    dim5: u32,
    _pad0: u32,
    _pad1: u32,
}

var<storage> src: array<f32>;
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let m = params.dim0;
    let n = params.dim1;
    let col = gid.x;
    let row = gid.y;
    if row >= m || col >= n { return; }
    // dst[col * m + row] = src[row * n + col]
    dst[col * m + row] = src[row * n + col];
}

// Reversible NCHW <-> window-token permutation. The packed row-major matrix
// has shape [window^2, batch * windows_y * windows_x * channels]. Every
// (batch, window, channel-head) can consequently use the ordinary attention
// implementation as an independent head.

@compute @workgroup_size(256)
fn window_pack(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch_count = params.dim0;
    let channels = params.dim1;
    let height = params.dim2;
    let width = params.dim3;
    let window = params.dim4;
    let shift = params.dim5;
    let windows_y = (height + shift + window - 1u) / window;
    let windows_x = (width + shift + window - 1u) / window;
    let window_count = windows_y * windows_x;
    let inner = batch_count * window_count * channels;
    let window_area = window * window;
    let total = window_area * inner;
    let i = gid.x;
    if i >= total { return; }

    let token = i / inner;
    let column = i % inner;
    let channel = column % channels;
    let batch_window = column / channels;
    let batch = batch_window / window_count;
    let window_id = batch_window % window_count;
    let window_y = window_id / windows_x;
    let window_x = window_id % windows_x;
    let token_y = token / window;
    let token_x = token % window;

    // The shifted grid starts at (-shift, -shift). Signed coordinates make
    // the leading padding explicit and avoid unsigned underflow.
    let y = i32(window_y * window + token_y) - i32(shift);
    let x = i32(window_x * window + token_x) - i32(shift);
    if x >= 0 && x < i32(width) && y >= 0 && y < i32(height) {
        let source = ((batch * channels + channel) * height + u32(y))
            * width + u32(x);
        dst[i] = src[source];
    } else {
        dst[i] = 0.0;
    }
}

@compute @workgroup_size(256)
fn window_merge(@builtin(global_invocation_id) gid: vec3<u32>) {
    let batch_count = params.dim0;
    let channels = params.dim1;
    let height = params.dim2;
    let width = params.dim3;
    let window = params.dim4;
    let shift = params.dim5;
    let total = batch_count * channels * height * width;
    let i = gid.x;
    if i >= total { return; }

    let x = i % width;
    let y = (i / width) % height;
    let channel = (i / (width * height)) % channels;
    let batch = i / (channels * height * width);

    let windows_y = (height + shift + window - 1u) / window;
    let windows_x = (width + shift + window - 1u) / window;
    let window_count = windows_y * windows_x;
    let shifted_x = x + shift;
    let shifted_y = y + shift;
    let window_x = shifted_x / window;
    let window_y = shifted_y / window;
    let token_x = shifted_x % window;
    let token_y = shifted_y % window;
    let token = token_y * window + token_x;
    let window_id = window_y * windows_x + window_x;
    let inner = batch_count * window_count * channels;
    let column = (batch * window_count + window_id) * channels + channel;
    dst[i] = src[token * inner + column];
}
