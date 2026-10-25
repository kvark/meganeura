struct Params {
    m: u32, n: u32, k: u32, groups: u32,
    splits: u32, _pad0: u32, _pad1: u32, _pad2: u32,
}

fn grouped_span(k: u32, splits: u32, part_id: u32) -> vec2<u32> {
    let tiles = k / 32u + select(0u, 1u, k % 32u != 0u);
    let count = tiles / splits;
    let extra = tiles % splits;
    let first = part_id * count + min(part_id, extra);
    let start = first * 32u;
    let length = (count + select(0u, 1u, part_id < extra)) * 32u;
    return vec2<u32>(start, start + min(k - start, length));
}
