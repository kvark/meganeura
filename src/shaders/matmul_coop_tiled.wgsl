// @section header
enable wgpu_cooperative_matrix;
$EXTENSIONS
$PARAMS_TYPE
var<storage> matrix_a: array<vec4<$A_TYPE>>;
$MATRIX_BINDINGS
$ADD_DECL
$PROLOGUE_DECL
var<uniform> params: Params;
$SHARED_DECL
$RESULT_DECL
$ROUND
$CACHE_DECL

// @section kernel
$ENTRY_SIGNATURE {
    $DIMS
    let tile_row = wgid.x * 64u;
    let tile_col = wgid.y * $COLUMNSu;
    $RANGE
    $OUTPUT_LAYOUT
    $CACHE_INIT
    // Subgroup width is not a pipeline requirement in Blade. Every width
    // executes the same four logical tiles, with uniform outer-loop trips
    // and workgroup barriers even when some subgroups have no matrix tile.
    for (var wave_base = 0u; wave_base < 4u; wave_base += $THREADSu / sg_size) {
        let wave = wave_base + sg;
        // Extra subgroups compute a valid tile but do not store it. Keeping
        // cooperative loads and arithmetic in uniform flow also satisfies
        // Naga's conservative cooperative-operation uniformity analysis.
        let wr = (wave % 4u) / 2u;
        let wc = wave % 2u;
        $ACC_INIT
        $VARIABLES
        $FIRST_LOAD
        for (var t = begin; t < end; t += $STAGEu) {
            $STEP_LOAD
            $WRITES
            workgroupBarrier();
            $NEXT_LOAD
            $MULTIPLY
            $PARTIAL_STEP
            workgroupBarrier();
        }
        $PARTIAL_FINISH
        if wave < 4u {
            $STORES
        }
        $SCATTER
    }
}
