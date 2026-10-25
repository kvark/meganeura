enable wgpu_memory_fence;

struct Node {
    extent: u32, data_offset: u32, first_task: u32, task_count: u32,
    first_pred: u32, pred_count: u32, first_succ: u32, succ_count: u32,
}
struct Params {
    node_count: u32, task_count: u32, word_count: u32, seed: u32,
    bitmap_offset: u32, pending_offset: u32, tiles_offset: u32, counts_offset: u32,
    workers_offset: u32, worker_count: u32, padding0: u32, padding1: u32,
}
var<storage, read_write> payload: array<u32>;
var<storage, read> incoming: array<vec4<u32>>;
var<storage, read> nodes: array<Node>;
var<storage, read> edges: array<u32>;
var<storage, read> tasks: array<vec2<u32>>;
var<storage, read_write> state: array<atomic<u32>>;
var<uniform> params: Params;
var<workgroup> selected: u32;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) worker: vec3<u32>, @builtin(local_invocation_index) lane: u32) {
    for (var iteration = 0u; iteration <= params.task_count; iteration += 1u) {
        if lane == 0u {
            selected = 0xffffffffu;
            for (var probe = 0u; probe < params.word_count; probe += 1u) {
                let word = (worker.x + probe) % params.word_count;
                let bits = atomicLoad(&state[params.bitmap_offset + word]);
                if bits != 0u {
                    let bit = firstTrailingBit(bits);
                    let mask = 1u << bit;
                    let old = atomicAnd(&state[params.bitmap_offset + word], ~mask);
                    if (old & mask) != 0u {
                        selected = word * 32u + bit;
                        storageFence();
                        break;
                    }
                }
            }
        }
        let task = workgroupUniformLoad(&selected);
        if task == 0xffffffffu {
            return;
        }
        if task >= params.task_count {
            if lane == 0u { atomicOr(&state[1], 1u); }
            return;
        }
        storageFence();
        let descriptor = tasks[task];
        let node_id = descriptor.x;
        let node = nodes[node_id];
        let index = descriptor.y * 64u + lane;
        if index < node.extent {
            var value = params.seed + (node_id + 1u) * 1009u + index;
            for (var p = 0u; p < node.pred_count; p += 1u) {
                let predecessor = nodes[edges[node.first_pred + p]];
                let source = (index * 17u + node_id * 13u + params.seed) % predecessor.extent;
                let offset = predecessor.data_offset + source;
                value += incoming[offset / 4u][offset % 4u];
            }
            for (var delay = 0u; delay < (node_id % 5u) * 3u; delay += 1u) {
                value = value * 1664525u + 1013904223u;
            }
            payload[node.data_offset + index] = value;
        }
        storageFence();
        workgroupBarrier();
        if lane == 0u {
            storageFence();
            atomicAdd(&state[params.counts_offset + task], 1u);
            let tiles = atomicSub(&state[params.tiles_offset + node_id], 1u);
            if tiles == 0u { atomicOr(&state[1], 2u); }
            if tiles == 1u {
                storageFence();
                for (var s = 0u; s < node.succ_count; s += 1u) {
                    let successor_id = edges[node.first_succ + s];
                    storageFence();
                    let pending = atomicSub(&state[params.pending_offset + successor_id], 1u);
                    if pending == 0u { atomicOr(&state[1], 4u); }
                    if pending == 1u {
                        storageFence();
                        let successor = nodes[successor_id];
                        for (var tile = 0u; tile < successor.task_count; tile += 1u) {
                            let next = successor.first_task + tile;
                            atomicOr(&state[params.bitmap_offset + next / 32u], 1u << (next % 32u));
                        }
                    }
                }
            }
            if atomicSub(&state[0], 1u) == 0u { atomicOr(&state[1], 8u); }
            atomicAdd(&state[params.workers_offset + worker.x], 1u);
        }
    }
    if lane == 0u { atomicOr(&state[1], 16u); }
}
