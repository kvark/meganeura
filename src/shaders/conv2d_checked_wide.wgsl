// @section main
$EXTENSIONS
$DECL
var<workgroup> stage_a: array<$MULTIPLY_TYPE, $WIDE_A_SIZEu>;
var<workgroup> stage_b: array<$MULTIPLY_TYPE, $WIDE_B_SIZEu>;
$RESULT_DECL
$ROUND

@compute @workgroup_size($THREADS)
fn main(@builtin(workgroup_id) group: vec3<u32>,
        @builtin(local_invocation_index) lane: u32,
        @builtin(subgroup_id) subgroup: u32,
        @builtin(subgroup_size) subgroup_size: u32) {
    $DIMS
    $RANGE
    let row = group.x * $WIDE_ROWSu;
    let col = group.y * $WIDE_COLUMNSu;
    // Each subgroup owns a 32x32 block of native accumulators. A runtime
    // width other than the one the invocation count assumed takes extra
    // rounds, or leaves subgroups computing a copy they do not store, so
    // cooperative operations and barriers stay in uniform control flow.
    let subgroups = max($THREADSu / subgroup_size, 1u);
    for (var base = 0u; base < $WIDE_BLOCKSu; base += subgroups) {
        let block = base + subgroup;
        let block_row = ((block % $WIDE_BLOCKSu) / $WIDE_BLOCK_COLUMNSu) * 32u;
        let block_col = ((block % $WIDE_BLOCKSu) % $WIDE_BLOCK_COLUMNSu) * 32u;
$WIDE_ACCUMULATORS
        for (var start = first; start < end; start += 32u) {
$WIDE_STAGE
            workgroupBarrier();
            for (var inner = 0u; inner < 32u; inner += $TILE_Ku) {
$WIDE_PRODUCTS
            }
            workgroupBarrier();
        }
        if block < $WIDE_BLOCKSu {
$WIDE_STORES
        }
    }
$RESULT_SCATTER
}

// @section stage
            for (var index = lane; index < $WIDE_A_SIZEu; index += $THREADSu) {
                let ar = row + index / 32u;
                let ak = start + index % 32u;
                var a = $MULTIPLY_TYPE(0.0f);
                if all(vec2<bool>((ar < m), (ak < k))) { a = $A_CAST($A_LOAD); }
                stage_a[index] = a;
            }
            for (var index = lane; index < $WIDE_B_SIZEu; index += $THREADSu) {
                let bk = start + index / $WIDE_COLUMNSu;
                let bc = col + index % $WIDE_COLUMNSu;
                var b = $MULTIPLY_TYPE(0.0f);
                if all(vec2<bool>((bk < k), (bc < n))) { b = $B_CAST($B_LOAD); }
                stage_b[index] = b;
            }

// @section vector_stage
            for (var index = lane; index < $WIDE_A_VECTORSu; index += $THREADSu) {
                let ar = row + index / 8u;
                let ak = start + (index % 8u) * 4u;
                var a = vec4<$MULTIPLY_TYPE>();
                if all(vec2<bool>((ar < m), (ak < k))) {
                    let v = weight[(ar * k + ak) / 4u];
                    a = $A_VECTOR;
                }
                stage_a[index*4u] = a.x; stage_a[index*4u+1u] = a.y;
                stage_a[index*4u+2u] = a.z; stage_a[index*4u+3u] = a.w;
            }
            for (var index = lane; index < $WIDE_B_VECTORSu; index += $THREADSu) {
                let bk = start + index / $WIDE_COLUMN_VECTORSu;
                let bc = col + (index % $WIDE_COLUMN_VECTORSu) * 4u;
                var b = vec4<$MULTIPLY_TYPE>();
                if all(vec2<bool>((bk < k), (bc < n))) {
                    let v = src[((group.z * k + bk) * n + bc) / 4u];
                    b = $B_VECTOR;
                }
                stage_b[index*4u] = b.x; stage_b[index*4u+1u] = b.y;
                stage_b[index*4u+2u] = b.z; stage_b[index*4u+3u] = b.w;
            }

// @section scatter
    workgroupBarrier();
    for (var offset = lane; offset < $WIDE_RESULTu; offset += $THREADSu) {
        let r = row + offset / $WIDE_COLUMNSu;
        let c = col + offset % $WIDE_COLUMNSu;
        if r < m && c < n {
            let index = $INDEX;
            $DEST[index] = $STORE(result[offset]$ADD);
        }
    }
