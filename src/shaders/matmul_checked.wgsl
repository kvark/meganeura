// @section main
$EXTENSIONS
$DECL
var<workgroup> stage_a: array<$MULTIPLY_TYPE, 1024>;
var<workgroup> stage_b: array<$MULTIPLY_TYPE, 1024>;
$RESULT_DECL
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
    // Pair eight-row tiles when there are too few subgroups to cover all
    // row blocks in one pass. Eight-column tiles also compute the right half.
    let subgroups = $THREADSu / subgroup_size;
    let paired = $TILE_Mu == 8u && subgroups < 8u;
    let tile_rows = select($TILE_Mu, 16u, paired);
    let tile_count = 64u / tile_rows;
    // Extra subgroups compute a valid tile but do not store it, keeping the
    // cooperative operations and workgroup barriers in uniform control flow.
    for (var base = 0u; base < tile_count; base += subgroups) {
        let tile = base + subgroup;
        let a_offset = ((tile % tile_count) / 2u) * tile_rows * 32u;
        let b_offset = (tile % 2u) * 16u;
        var accumulator = $C_TILE();
        var lower = $C_TILE();
        var right = $C_TILE();
        var lower_right = $C_TILE();
        for (var start = first; start < end; start += 32u) {
            $STAGE
            workgroupBarrier();
            for (var inner = 0u; inner < 32u; inner += $TILE_Ku) {
                let a = coopLoadT<$A_TILE>(&stage_a[a_offset + inner], 32u);
                let b = coopLoadT<$B_TILE>(&stage_b[b_offset + inner * 32u], 32u);
                accumulator = coopMultiplyAdd(a, b, accumulator);
                var a_lower = a;
                // Loops keep cooperative operations in uniform control flow
                // despite Naga's conservative subgroup-size conditional check.
                for (var lower_row = $TILE_Mu; lower_row < tile_rows; lower_row += $TILE_Mu) {
                    a_lower = coopLoadT<$A_TILE>(&stage_a[a_offset + lower_row * 32u + inner], 32u);
                    lower = coopMultiplyAdd(a_lower, b, lower);
                }
                for (var right_col = $TILE_Nu; right_col < 16u; right_col += $TILE_Nu) {
                    let b_right = coopLoadT<$B_TILE>(&stage_b[b_offset + inner * 32u + right_col], 32u);
                    right = coopMultiplyAdd(a, b_right, right);
                    for (var lower_row = $TILE_Mu; lower_row < tile_rows; lower_row += $TILE_Mu) {
                        lower_right = coopMultiplyAdd(a_lower, b_right, lower_right);
                    }
                }
            }
            workgroupBarrier();
        }
        if tile < tile_count$STORE_VALID {
            coopStoreT(accumulator, $C_BASE, $RESULT_STRIDE);
            if paired {
                coopStoreT(lower, $C_LOWER, $RESULT_STRIDE);
            }
            if $TILE_Nu == 8u {
                coopStoreT(right, $C_RIGHT, $RESULT_STRIDE);
                if paired {
                    coopStoreT(lower_right, $C_LOWER_RIGHT, $RESULT_STRIDE);
                }
            }
        }
    }
    $RESULT_SCATTER
}

// @section scatter
workgroupBarrier();
    for (var offset = lane; offset < 1024u; offset += $THREADSu) {
        let r = row + offset / 32u;
        let c = col + offset % 32u;
        if r < m && c < n {
            let index = $INDEX;
            $DEST[index] = $STORE(result[offset]$ADD);
        }
    }
