// Columns of the `k` largest values of each row, largest first, ties to the
// lower column, as exact F32 indices: src [rows, cols] -> dst [rows, k].
//
// One workgroup per row. A radix select over order-preserving u32 keys finds
// the k-th largest key T in four 8-bit passes, counting into a shared
// histogram. Each thread then scans one contiguous chunk of the row: keys
// above T all belong to the result, and keys equal to T are taken by column,
// lowest first, which an exclusive scan of the per-chunk counts orders. The
// selection is sorted by (key descending, column ascending) with a bitonic
// network in workgroup memory, padded to a power of two.
//
// Short rows take the narrow mode (`narrow != 0`) instead: one thread per
// row, each pick the best entry that comes after the previous pick in
// result order, k scans of the row with no shared state.

struct Params {
    rows: u32,
    cols: u32,
    k: u32,
    narrow: u32,
}

var<storage> src: array<f32>;
var<storage, read_write> dst: array<f32>;
var<uniform> params: Params;

const THREADS: u32 = 256u;
// The largest k, a power of two.
const SELECTED: u32 = 1024u;

var<workgroup> histogram: array<atomic<u32>, 256>;
var<workgroup> above_counts: array<u32, 256>;
var<workgroup> tie_counts: array<u32, 256>;
var<workgroup> keys: array<u32, SELECTED>;
var<workgroup> columns: array<u32, SELECTED>;
// The k-th key's fixed prefix, and how many keys equal to it are wanted.
var<workgroup> state: array<u32, 2>;

// A u32 that orders like the float: negatives reversed below positives.
// Both zeros map to +0 so they tie, as they compare equal.
fn sort_key(value: f32) -> u32 {
    var bits = bitcast<u32>(value);
    if value == 0.0 {
        bits = 0u;
    }
    if (bits & 0x80000000u) != 0u {
        return ~bits;
    }
    return bits | 0x80000000u;
}

// Whether (key a, column a) comes before (key b, column b) in the result.
fn first(key_a: u32, column_a: u32, key_b: u32, column_b: u32) -> bool {
    return key_a > key_b || (key_a == key_b && column_a < column_b);
}

// The best entry of a short row that comes after (prev_key, prev_column)
// in result order, scanning every column. Picks are strictly ordered, so
// nothing needs marking as taken.
fn pick_after(base: u32, cols: u32, first_pick: bool, prev_key: u32, prev_column: u32) -> vec2<u32> {
    var best_key = 0u;
    var best_column = 0xffffffffu;
    for (var c = 0u; c < cols; c++) {
        let key = sort_key(src[base + c]);
        let after = first_pick || first(prev_key, prev_column, key, c);
        if after && (best_column == 0xffffffffu || first(key, c, best_key, best_column)) {
            best_key = key;
            best_column = c;
        }
    }
    return vec2<u32>(best_key, best_column);
}

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) t: u32) {
    let cols = params.cols;
    let k = params.k;
    // Uniform across the dispatch, so the barriers below stay uniform.
    if params.narrow != 0u {
        let row = wid.x * THREADS + t;
        if row < params.rows {
            var pick = vec2<u32>(0u, 0u);
            for (var i = 0u; i < k; i++) {
                pick = pick_after(row * cols, cols, i == 0u, pick.x, pick.y);
                dst[row * k + i] = f32(pick.y);
            }
        }
        return;
    }
    let row = wid.x;
    let base = row * cols;

    if t == 0u {
        state[0] = 0u;
        state[1] = k;
    }
    // Radix select: fix the key's top byte, then the next, keeping how many
    // of the wanted elements still lie among keys with the fixed prefix.
    var mask = 0u;
    for (var round = 0u; round < 4u; round++) {
        let shift = 24u - round * 8u;
        atomicStore(&histogram[t], 0u);
        workgroupBarrier();
        let prefix = state[0];
        for (var c = t; c < cols; c += THREADS) {
            let key = sort_key(src[base + c]);
            if (key & mask) == prefix {
                atomicAdd(&histogram[(key >> shift) & 255u], 1u);
            }
        }
        workgroupBarrier();
        // Each thread owns one bucket and sums the buckets above it; the
        // one where the wanted element falls fixes the next byte, so no
        // thread walks the histogram alone.
        above_counts[t] = atomicLoad(&histogram[t]);
        workgroupBarrier();
        let need = state[1];
        var higher = 0u;
        for (var b = t + 1u; b < 256u; b++) {
            higher += above_counts[b];
        }
        let count = above_counts[t];
        workgroupBarrier();
        if higher < need && higher + count >= need {
            state[1] = need - higher;
            state[0] = prefix | (t << shift);
        }
        mask |= 255u << shift;
        workgroupBarrier();
    }
    let kth = state[0];
    let ties = state[1];
    // Every key above the k-th belongs to the result; they fill its front.
    let above_total = k - ties;

    let chunk = (cols + THREADS - 1u) / THREADS;
    let start = min(t * chunk, cols);
    let end = min(start + chunk, cols);
    var above = 0u;
    var tied = 0u;
    for (var c = start; c < end; c++) {
        let key = sort_key(src[base + c]);
        if key > kth {
            above += 1u;
        } else if key == kth {
            tied += 1u;
        }
    }
    above_counts[t] = above;
    tie_counts[t] = tied;
    workgroupBarrier();
    if t == 0u {
        var above_sum = 0u;
        var tie_sum = 0u;
        for (var i = 0u; i < THREADS; i++) {
            let a = above_counts[i];
            let e = tie_counts[i];
            above_counts[i] = above_sum;
            tie_counts[i] = tie_sum;
            above_sum += a;
            tie_sum += e;
        }
    }
    workgroupBarrier();
    var slot = above_counts[t];
    var rank = tie_counts[t];
    for (var c = start; c < end; c++) {
        let key = sort_key(src[base + c]);
        if key > kth {
            keys[slot] = key;
            columns[slot] = c;
            slot += 1u;
        } else if key == kth {
            if rank < ties {
                keys[above_total + rank] = key;
                columns[above_total + rank] = c;
            }
            rank += 1u;
        }
    }
    var padded = 1u;
    while padded < k {
        padded <<= 1u;
    }
    // Padding sorts after every real entry.
    for (var i = k + t; i < padded; i += THREADS) {
        keys[i] = 0u;
        columns[i] = 0xffffffffu;
    }
    workgroupBarrier();

    // Bitonic sort into result order.
    for (var size = 2u; size <= padded; size <<= 1u) {
        for (var stride = size >> 1u; stride > 0u; stride >>= 1u) {
            for (var i = t; i < padded / 2u; i += THREADS) {
                let lo = 2u * stride * (i / stride) + (i % stride);
                let hi = lo + stride;
                let forward = (lo & size) == 0u;
                let ka = keys[lo];
                let ca = columns[lo];
                let kb = keys[hi];
                let cb = columns[hi];
                if first(ka, ca, kb, cb) != forward {
                    keys[lo] = kb;
                    columns[lo] = cb;
                    keys[hi] = ka;
                    columns[hi] = ca;
                }
            }
            workgroupBarrier();
        }
    }

    for (var i = t; i < k; i += THREADS) {
        dst[row * k + i] = f32(columns[i]);
    }
}
