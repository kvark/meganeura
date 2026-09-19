// N-tile GEMV for M=1: C[1, N] = A[1, K] × B[K, N].
//
// llama.cpp's MUL_MAT_VEC walks K coalesced because ggml stores each output
// row contiguously along K. Meganeura stores B as [K, N], so the coalesced
// direction is N. Each workgroup owns 32 consecutive output columns. 32
// warps K-split those columns: warp w walks k in [w*K/32, (w+1)*K/32).
// Adjacent lanes at a fixed k read consecutive N — one 128-byte transaction
// per k-step — then five barriers combine the 32 partials.
//
// Dispatch: N/32 workgroups × 1024 threads. 32 warps per WG recover the
// occupancy a 256-thread N-tile lost on seq=1 (96 WGs × 8 warps was too
// few on a 48-SM GPU). Used for wide N (packed FFN-up).
// Codegen fills the dollar-token slots for plain, fused-add, and
// RmsNorm-folded forms.

struct Params {
    m: u32,
    n: u32,
    k: u32,
    $PAD_FIELD: u32,
}

var<storage> matrix_a: array<f32>;
$NORM_W_DECL
var<storage> matrix_b: array<f32>;
var<storage, read_write> matrix_c: array<f32>;
$SRC_DECL
var<uniform> params: Params;

const N_TILE: u32 = 32u;
const K_WARPS: u32 = 32u;
const WG_SIZE: u32 = 1024u;

$WG_EXTRA
var<workgroup> reduce_buf: array<f32, 1024>;

fn a_at(kk: u32) -> f32 {
    return $A_AT_BODY;
}

@compute @workgroup_size(1024)
fn main(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
    let lane = lid.x;
    let warp = lane / N_TILE;
    let lane_in_warp = lane % N_TILE;
    let col = wgid.x * N_TILE + lane_in_warp;
    let n = params.n;
    let k = params.k;

    $PROLOGUE

    var acc = 0.0;
    if col < n {
        let k_lo = warp * k / K_WARPS;
        let k_hi = (warp + 1u) * k / K_WARPS;
        var kk = k_lo;
        let k_hi4 = k_lo + ((k_hi - k_lo) / 4u) * 4u;
        loop {
            if kk >= k_hi4 { break; }
            acc = fma(a_at(kk), matrix_b[kk * n + col], acc);
            acc = fma(a_at(kk + 1u), matrix_b[(kk + 1u) * n + col], acc);
            acc = fma(a_at(kk + 2u), matrix_b[(kk + 2u) * n + col], acc);
            acc = fma(a_at(kk + 3u), matrix_b[(kk + 3u) * n + col], acc);
            kk = kk + 4u;
        }
        loop {
            if kk >= k_hi { break; }
            acc = fma(a_at(kk), matrix_b[kk * n + col], acc);
            kk = kk + 1u;
        }
    }

    reduce_buf[lane] = acc;
    workgroupBarrier();
    if warp < 16u { reduce_buf[lane] += reduce_buf[lane + 512u]; }
    workgroupBarrier();
    if warp < 8u { reduce_buf[lane] += reduce_buf[lane + 256u]; }
    workgroupBarrier();
    if warp < 4u { reduce_buf[lane] += reduce_buf[lane + 128u]; }
    workgroupBarrier();
    if warp < 2u { reduce_buf[lane] += reduce_buf[lane + 64u]; }
    workgroupBarrier();
    if warp < 1u { reduce_buf[lane] += reduce_buf[lane + 32u]; }
    workgroupBarrier();
    if warp == 0u && col < n {
        matrix_c[col] = reduce_buf[lane]$ADDEND;
    }
}
