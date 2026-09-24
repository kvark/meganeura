// Optimizer kernels run one dispatch per arena chunk (see
// `memplan::ArenaChunk`). Its parameters are segments of the bound buffers,
// all addressed from one base. Workgroup `g` covers elements
// [(g - first_group) * TILE, + TILE) of the segment whose groups contain g,
// so every parameter is split the same way however it is packed.

struct Segment {
    // First element, counted from the bound buffers' base.
    offset: u32,
    // Elements.
    len: u32,
    // First workgroup of this segment.
    first_group: u32,
    // Position in the plan's parameter/gradient pairs.
    index: u32,
    // Per-parameter learning-rate multiplier.
    lr_scale: f32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}

const TILE: u32 = 1024u;
const WORKGROUP: u32 = 256u;

var<storage> segments: array<Segment>;

// Linear workgroup index: grids wider than the dispatch limit wrap into y.
fn group_index(wgid: vec3<u32>, groups: vec3<u32>) -> u32 {
    return wgid.y * groups.x + wgid.x;
}

// The segment whose workgroups include `group`; segments are sorted by
// their first workgroup.
fn find_segment(count: u32, group: u32) -> Segment {
    var lo = 0u;
    var hi = count;
    loop {
        if hi - lo <= 1u { break; }
        let mid = (lo + hi) / 2u;
        if segments[mid].first_group <= group {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    return segments[lo];
}
