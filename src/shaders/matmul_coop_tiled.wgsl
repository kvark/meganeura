// @section header
enable wgpu_cooperative_matrix;
struct Params { m: u32, n: u32, k: u32, _pad: u32, }
var<storage> matrix_a: array<vec4<f32>>;
$MATRIX_BINDINGS
$ADD_DECL
$PROLOGUE_DECL
var<uniform> params: Params;
var<workgroup> sa: array<f32, $A_SIZE>;
var<workgroup> sb: array<f32, $B_SIZE>;
$CACHE_DECL

// @section kernel
$ENTRY_SIGNATURE {
    let m = params.m;
    let n = params.n;
    let k = params.k;
    let tile_row = wgid.x * 64u;
    let tile_col = wgid.y * $COLUMNSu;
    let chunk = ((k / $STAGEu + $SPLITSu - 1u) / $SPLITSu) * $STAGEu;
    let begin = wgid.z * chunk;
    let end = min(k, begin + chunk);
    let output_base = wgid.z * m * n;
    $CACHE_INIT
    // Subgroup width is not a pipeline requirement in Blade. Every width
    // executes the same four logical tiles, with uniform outer-loop trips
    // and workgroup barriers even when some subgroups have no matrix tile.
    for (var wave_base = 0u; wave_base < 4u; wave_base += 256u / sg_size) {
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
            workgroupBarrier();
        }
        if wave < 4u {
            $STORES
        }
    }
}
