struct Params {
    batch: u32,
    channels: u32,
    in_h: u32,
    in_w: u32,
    out_h: u32,
    out_w: u32,
    eps: u32,
    _pad: u32,
}
var<uniform> params: Params;
var<workgroup> scratch: array<f32, 256>;

fn image_index(row: u32, channel: u32) -> u32 {
    let pixels = params.out_h * params.out_w;
    return (row / pixels * CHANNELS + channel) * pixels + row % pixels;
}

fn source_index(row: u32, channel: u32, offset: u32) -> u32 {
    let pixels = params.out_h * params.out_w;
    let position = row % pixels;
    let y = position / params.out_w * WINDOW + offset / WINDOW;
    let x = position % params.out_w * WINDOW + offset % WINDOW;
    return ((row / pixels * CHANNELS + channel) * params.in_h + y) * params.in_w + x;
}
