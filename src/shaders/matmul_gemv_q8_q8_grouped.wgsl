// Share each K32 activation quantization across several vec4 output groups.
// Only the first contiguous lane group packs A. All groups reach the barriers,
// including inactive output/K tails; each group reduces its own F32 dot sum.
// The single-column-group kernel keeps its existing barrier-free quantizer.
struct Params {
    m: u32,
    n: u32,
    k: u32,
    eps_bits: u32,
}

var<storage> matrix_a: array<f32>;
$NORM_DECL
var<storage> matrix_b: array<u32>;
var<storage, read_write> matrix_c: array<vec4<f32>>;
$ADDEND_DECL
var<uniform> params: Params;
const LANES: u32 = $LANESu;
const GROUPS: u32 = $COLUMN_GROUPSu;
var<workgroup> a_scale: array<f32, LANES>;
// Consecutive packing lanes write consecutive words at each byte position.
var<workgroup> a_quant: array<u32, 8u * LANES>;
var<workgroup> reduce_buf: array<vec4<f32>, LANES * GROUPS>;

$PACKED_DOT_HELPER
$A_FN_DECL

@compute @workgroup_size(LANES * GROUPS)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let column_group = lid.x / LANES;
    let lane = lid.x % LANES;
    let col4 = wgid.x * GROUPS + column_group;
    let n_v4 = params.n / 4u;
    let k = params.k;
$NORM_PROLOGUE
    let blocks = k / 32u;
    var acc = vec4<f32>(0.0);
    var tile = 0u;
    loop {
        if tile >= blocks { break; }
        let blk = tile + lane;
        if column_group == 0u && blk < blocks {
            // Preserve the inline Q8_1 quantizer's scale, reciprocal and RNE.
            let abase = blk * 32u;
            var amax = 0.0;
            for (var i = 0u; i < 32u; i++) {
                amax = max(amax, abs(a_val(abase + i)));
            }
            let d8 = amax / 127.0;
            let inv = select(0.0, 1.0 / d8, amax > 0.0);
            a_scale[lane] = d8;
            for (var j = 0u; j < 8u; j++) {
                var word = 0u;
                for (var b = 0u; b < 4u; b++) {
                    let q = clamp(i32(round(a_val(abase + j * 4u + b) * inv)), -127, 127);
                    word |= (bitcast<u32>(q) & 0xFFu) << (b * 8u);
                }
                a_quant[j * LANES + lane] = word;
            }
        }
        workgroupBarrier();
        if col4 < n_v4 && blk < blocks {
            let d8 = a_scale[lane];
            var u: array<u32, 8>;
            for (var j = 0u; j < 8u; j++) {
                u[j] = a_quant[j * LANES + lane];
            }
            let col = col4 * 4u;
            for (var c = 0u; c < 4u; c++) {
                let base = ((col + c) * blocks + blk) * 9u;
                var sumi = 0;
                for (var j = 0u; j < 8u; j++) {
                    sumi += dot_q8_q8_packed(matrix_b[base + 1u + j], u[j]);
                }
                acc[c] += unpack2x16float(matrix_b[base] & 0xFFFFu).x * (d8 * f32(sumi));
            }
        }
        // All readers finish before the packing group overwrites the tile.
        workgroupBarrier();
        tile += LANES;
    }
$REDUCTION
    if lane == 0u && col4 < n_v4 {
        matrix_c[col4] = $TOTAL$ADDEND;
    }
}
