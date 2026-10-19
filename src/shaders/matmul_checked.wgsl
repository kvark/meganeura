$EXTENSIONS
$DECL
var<workgroup> stage_a: array<$MULTIPLY_TYPE, 1024>;
var<workgroup> stage_b: array<$MULTIPLY_TYPE, 1024>;
var<workgroup> result: array<f32, 1024>;
$ROUND

@compute @workgroup_size(128)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_index) lane: u32,
        @builtin(subgroup_id) subgroup: u32,
        @builtin(subgroup_size) subgroup_size: u32) {
    $DIMS
    $RANGE
    let row = group.x * 32u;
    let col = group.y * 32u;
    // Four 16x16 output tiles; rectangular hardware tiles keep two accumulators.
    // Extra subgroups compute a valid tile but do not store it, keeping the
    // cooperative operations and workgroup barriers in uniform control flow.
    for (var base = 0u; base < 4u; base += 128u / subgroup_size) {
        let tile = base + subgroup;
        let a_offset = ((tile % 4u) / 2u) * 16u * 32u;
        let b_offset = (tile % 2u) * 16u;
        var accumulator = $C_TILE();
        var lower = $C_TILE();
        for (var start = first; start < end; start += 32u) {
            for (var index = lane; index < 1024u; index += 128u) {
                let ar = row + index / 32u;
                let ak = start + index % 32u;
                let bk = start + index / 32u;
                let bc = col + index % 32u;
                var a = $MULTIPLY_TYPE(0.0f);
                var b = $MULTIPLY_TYPE(0.0f);
                if ar < m && ak < k { a = $A_CAST($A_LOAD); }
                if bk < k && bc < n { b = $B_CAST($B_LOAD); }
                stage_a[index] = a;
                stage_b[index] = b;
            }
            workgroupBarrier();
            let a0 = coopLoadT<$A_TILE>(&stage_a[a_offset], 32u);
            let b0 = coopLoadT<$B_TILE>(&stage_b[b_offset], 32u);
            accumulator = coopMultiplyAdd(a0, b0, accumulator);
            let a1 = coopLoadT<$A_TILE>(&stage_a[a_offset + 16u], 32u);
            let b1 = coopLoadT<$B_TILE>(&stage_b[b_offset + 512u], 32u);
            accumulator = coopMultiplyAdd(a1, b1, accumulator);
            if $TILE_Mu == 8u {
                let a2 = coopLoadT<$A_TILE>(&stage_a[a_offset + 256u], 32u);
                let a3 = coopLoadT<$A_TILE>(&stage_a[a_offset + 272u], 32u);
                lower = coopMultiplyAdd(a2, b0, lower);
                lower = coopMultiplyAdd(a3, b1, lower);
            }
            workgroupBarrier();
        }
        if tile < 4u {
            coopStoreT(accumulator, &result[a_offset + b_offset], 32u);
            if $TILE_Mu == 8u {
                coopStoreT(lower, &result[a_offset + b_offset + 256u], 32u);
            }
        }
    }
    workgroupBarrier();
    for (var offset = lane; offset < 1024u; offset += 128u) {
        let r = row + offset / 32u;
        let c = col + offset % 32u;
        if r < m && c < n {
            let index = $INDEX;
            $DEST[index] = $STORE(result[offset]$ADD);
        }
    }
}
