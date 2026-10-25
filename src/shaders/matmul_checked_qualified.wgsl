$EXTENSIONS

$DECL
$STAGE_DECL
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
    // Four logical matrix tiles, independent of the device subgroup width.
    // Extra subgroups compute a valid tile but do not store it, keeping the
    // cooperative operations and workgroup barriers in uniform control flow.
    for (var base = 0u; base < 4u; base += 128u / subgroup_size) {
        let tile = base + subgroup;
        let a_offset = ((tile % 4u) / 2u) * 16u * 32u;
        let b_offset = (tile % 2u) * 16u;
        var accumulator = coop_mat16x16<f32, C>();
        $PARTIAL_INIT
        $STAGE_INIT
        for (var start = first; start < end; start += 32u) {
            $STAGE_LOAD
            workgroupBarrier();
            $MATRIX_PRODUCT
            $PARTIAL_STEP
            $STAGE_ADVANCE
            workgroupBarrier();
        }
        $PARTIAL_FINISH
        if tile < 4u { coopStoreT(accumulator, &result[a_offset + b_offset], 32u); }
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
