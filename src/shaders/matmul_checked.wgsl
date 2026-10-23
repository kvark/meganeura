$EXTENSIONS
$DECL
var<workgroup> stage_a: array<$SHARED_TYPE, $SHARED_COUNT>;
var<workgroup> stage_b: array<$SHARED_TYPE, $SHARED_COUNT>;
var<workgroup> result: array<f32, 1024>;
$ROUND

@compute @workgroup_size($THREADS)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_index) lane: u32,
        @builtin(subgroup_id) subgroup: u32,
        @builtin(subgroup_size) subgroup_size: u32) {
    $DIMS
    $RANGE
    let row = group.x * 32u;
    let col = group.y * 32u;
    // Pair rectangular tiles only when there are too few subgroups to cover
    // all eight hardware tiles in one pass.
    let subgroups = $THREADSu / subgroup_size;
    let paired = $TILE_Mu == 8u && subgroups < 8u;
    let tile_rows = select($TILE_Mu, 16u, paired);
    let tile_count = 64u / tile_rows;
    // Extra subgroups compute a valid tile but do not store it, keeping the
    // cooperative operations and workgroup barriers in uniform control flow.
    for (var base = 0u; base < tile_count; base += subgroups) {
        let tile = base + subgroup;
        let a_offset = ((tile % tile_count) / 2u) * tile_rows * (32u / $PACKu);
        let b_offset = (tile % 2u) * (16u / $PACKu);
        var accumulator = $C_TILE();
        var lower = $C_TILE();
        for (var start = first; start < end; start += 32u) {
            $STAGE
            workgroupBarrier();
            let a0 = coopLoadT<$A_TILE>(&stage_a[a_offset], 32u / $PACKu);
            let b0 = coopLoadT<$B_TILE>(&stage_b[b_offset], 32u / $PACKu);
            accumulator = coopMultiplyAdd(a0, b0, accumulator);
            let a1 = coopLoadT<$A_TILE>(&stage_a[a_offset + 16u / $PACKu], 32u / $PACKu);
            let b1 = coopLoadT<$B_TILE>(&stage_b[b_offset + 512u / $PACKu], 32u / $PACKu);
            accumulator = coopMultiplyAdd(a1, b1, accumulator);
            // A loop avoids Naga's conservative subgroup-size conditional check.
            for (var lower_row = $TILE_Mu; lower_row < tile_rows; lower_row += $TILE_Mu) {
                let a2 = coopLoadT<$A_TILE>(&stage_a[a_offset + 256u / $PACKu], 32u / $PACKu);
                let a3 = coopLoadT<$A_TILE>(&stage_a[a_offset + 272u / $PACKu], 32u / $PACKu);
                lower = coopMultiplyAdd(a2, b0, lower);
                lower = coopMultiplyAdd(a3, b1, lower);
            }
            workgroupBarrier();
        }
        if tile < tile_count {
            coopStoreT(accumulator, &result[(a_offset + b_offset) * $PACKu], 32u);
            if paired {
                coopStoreT(lower, &result[(a_offset + b_offset) * $PACKu + 256u], 32u);
            }
        }
    }
    workgroupBarrier();
    for (var offset = lane; offset < 1024u; offset += $THREADSu) {
        let r = row + offset / 32u;
        let c = col + offset % 32u;
        if r < m && c < n {
            let index = $INDEX;
            $DEST[index] = $STORE(result[offset]$ADD);
        }
    }
}
