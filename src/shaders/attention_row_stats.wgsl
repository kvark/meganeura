// Row dots D = dot(dO, O), followed by Q/dO maxima per 16-row tile and head.
// Scaled attention backward reuses the maxima across all key workgroups.
var<storage> src_a: array<f32>; // dO
var<storage> src_b: array<f32>; // O
var<storage> src_c: array<f32>; // Q
var<storage, read_write> dst: array<f32>;
var<uniform> params: vec4<u32>; // query rows, heads, head width, padding

var<workgroup> partial: array<vec2<f32>, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(subgroup_id) subgroup: u32,
    @builtin(subgroup_size) width: u32,
    @builtin(subgroup_invocation_id) lane: u32,
) {
    let subgroups = 256u / width;
    var maximum = vec2<f32>(0.0);
    // A subgroup owns each row; only the final tile maximum crosses subgroups.
    for (var r = subgroup; r < 16u; r += subgroups) {
        let row = group.x * 16u + r;
        let index = row * params.y + group.y;
        var dot = 0.0;
        var local = vec2<f32>(0.0);
        if row < params.x {
            for (var col = lane; col < params.z; col += width) {
                let i = index * params.z + col;
                let dout = src_a[i];
                dot += dout * src_b[i];
                local = max(local, abs(vec2<f32>(src_c[i], dout)));
            }
        }
        let sum = subgroupAdd(dot);
        maximum = max(maximum, subgroupMax(local));
        if lane == 0u && row < params.x { dst[index] = sum; }
    }
    if lane == 0u { partial[subgroup] = maximum; }
    workgroupBarrier();
    if lid.x == 0u {
        var tile_maximum = vec2<f32>(0.0);
        for (var group = 0u; group < subgroups; group++) {
            tile_maximum = max(tile_maximum, partial[group]);
        }
        let offset = params.x * params.y + 2u * (group.x * params.y + group.y);
        dst[offset] = tile_maximum.x;
        dst[offset + 1u] = tile_maximum.y;
    }
}
