enable wgpu_memory_fence;
enable wgpu_binding_array;
$ENABLES

struct GraphNode {
    // Logical grid, low 16 bits handler, high 16 bits tiles per claim.
    grid: vec4<u32>,
    // Parameter offset, logical tiles, successor offset, successor count.
    work: vec4<u32>,
}
struct GraphConfig {
    nodes: u32,
    ready_words: u32,
    counters: u32,
    worker_counts: u32,
    workers: u32,
    track_workers: u32,
    padding: vec2<u32>,
}
var<storage> graph_nodes: array<GraphNode>;
var<storage> graph_edges: array<u32>;
var<storage> graph_parameters: array<u32>;
var<storage, read_write> graph_queue: array<atomic<u32>>;
var<uniform> graph_config: GraphConfig;
var<workgroup> graph_ticket: vec4<u32>;
var<workgroup> graph_node_grid: vec4<u32>;
var<workgroup> graph_node_work: vec4<u32>;
var<workgroup> graph_parameter_words: array<u32, $PARAMETER_WORDS>;
var<workgroup> graph_geometry_error: array<atomic<u32>, 4>;

$HANDLERS
$BUFFERS
$SHARED

fn graph_claim(worker: u32, lane: u32) -> vec4<u32> {
    if lane == 0u {
        graph_ticket = vec4<u32>(0xffffffffu);
        if atomicLoad(&graph_queue[0]) == 0u {
            graph_ticket.x = 0xfffffffeu;
        } else {
            for (var probe = 0u; probe < graph_config.ready_words; probe += 1u) {
                let word = (worker + probe) % graph_config.ready_words;
                var bits = atomicLoad(&graph_queue[2u + word]);
                while bits != 0u {
                    let bit = firstTrailingBit(bits);
                    let mask = 1u << bit;
                    let node_index = word * 32u + bit;
                    let node = graph_nodes[node_index];
                    let count = max(node.work.y, 1u);
                    let counter = graph_config.counters + 3u * node_index;
                    let batch = node.grid.w >> 16u;
                    let tile = atomicAdd(&graph_queue[counter], batch);
                    if tile + batch >= count {
                        atomicAnd(&graph_queue[2u + word], ~mask);
                    }
                    if tile < count {
                        storageFence();
                        graph_node_grid = node.grid;
                        graph_node_work = node.work;
                        graph_ticket = vec4<u32>(node_index, tile, node.work.x, node.grid.w & 0xffffu);
                        break;
                    }
                    bits &= ~mask;
                }
                if graph_ticket.x != 0xffffffffu { break; }
            }
        }
    }
    return workgroupUniformLoad(&graph_ticket);
}

@compute @workgroup_size($THREADS)
fn graph_main(@builtin(local_invocation_index) physical_lane: u32,
              @builtin(workgroup_id) worker: vec3<u32>$SUBGROUPS) {
    $GEOMETRY
    loop {
        let ticket = graph_claim(worker.x, physical_lane);
        if ticket.x == 0xfffffffeu { return; }
        if ticket.x == 0xffffffffu { continue; }
        let node = GraphNode(workgroupUniformLoad(&graph_node_grid),
                             workgroupUniformLoad(&graph_node_work));
        for (var word = physical_lane; word < $PARAMETER_WORDSu; word += $THREADSu) {
            graph_parameter_words[word] = graph_parameters[ticket.z + word];
        }
        let graph_parameter_values = workgroupUniformLoad(&graph_parameter_words);
        storageFence();
        if node.work.y != 0u {
            let grid = node.grid.xyz;
            switch ticket.w {
                $CASES
                default: {
                    if physical_lane == 0u { atomicOr(&graph_queue[1], 1u); }
                }
            }
        }
        storageFence();
        workgroupBarrier();
        if physical_lane == 0u {
            storageFence();
            let counter = graph_config.counters + 3u * ticket.x;
            let completed = node.grid.w >> 16u;
            let remaining = atomicSub(&graph_queue[counter + 1u], completed);
            if remaining < completed { atomicOr(&graph_queue[1], 2u); }
            if remaining == completed {
                storageFence();
                for (var edge = 0u; edge < node.work.w; edge += 1u) {
                    let successor = graph_edges[node.work.z + edge];
                    let dependency = graph_config.counters + 3u * successor + 2u;
                    storageFence();
                    let pending = atomicSub(&graph_queue[dependency], 1u);
                    if pending == 0u { atomicOr(&graph_queue[1], 4u); }
                    if pending == 1u {
                        storageFence();
                        atomicOr(&graph_queue[2u + successor / 32u], 1u << (successor % 32u));
                    }
                }
                if atomicSub(&graph_queue[0], 1u) == 0u {
                    atomicOr(&graph_queue[1], 8u);
                }
            }
            if graph_config.track_workers != 0u && node.work.y != 0u {
                atomicAdd(&graph_queue[graph_config.worker_counts
                    + ticket.x * graph_config.workers + worker.x], completed);
            }
        }
        workgroupBarrier();
    }
}
