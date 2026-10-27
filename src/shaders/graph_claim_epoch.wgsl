fn graph_claim(worker: u32, lane: u32) -> vec4<u32> {
    if lane == 0u {
        graph_ticket = vec4<u32>(0xffffffffu);
        loop {
            let remaining = atomicLoad(&graph_queue[0]);
            if remaining == 0u {
                graph_ticket.x = 0xfffffffeu;
                break;
            }
            if remaining == graph_idle_remaining { continue; }
            if remaining != graph_acquired_remaining {
                storageFence();
                graph_acquired_remaining = remaining;
            }
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
            if graph_ticket.x != 0xffffffffu { break; }
            graph_idle_remaining = remaining;
        }
    }
    return workgroupUniformLoad(&graph_ticket);
}
