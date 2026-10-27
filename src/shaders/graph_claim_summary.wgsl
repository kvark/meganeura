// The summary is advisory. Only the original ready bitmap and tile RMW grant
// ownership; a failed summary scan always falls back before caching an epoch.
fn graph_claim_word(word: u32, repair_summary: bool) -> bool {
    let summary = graph_config.padding.x + word / 32u;
    let summary_mask = 1u << (word % 32u);
    var bits = atomicLoad(&graph_queue[2u + word]);
    if bits == 0u {
        if !repair_summary {
            atomicAnd(&graph_queue[summary], ~summary_mask);
        }
        return false;
    }
    if repair_summary {
        atomicOr(&graph_queue[summary], summary_mask);
    }
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
            let old_bits = atomicAnd(&graph_queue[2u + word], ~mask);
            if old_bits == mask {
                atomicAnd(&graph_queue[summary], ~summary_mask);
            }
        }
        if tile < count {
            storageFence();
            graph_node_grid = node.grid;
            graph_node_work = node.work;
            graph_ticket = vec4<u32>(node_index, tile, node.work.x, node.grid.w & 0xffffu);
            return true;
        }
        bits &= ~mask;
    }
    return false;
}

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
            let first_word = worker % graph_config.ready_words;
            let first_bit = first_word % 32u;
            var summary_index = first_word / 32u;
            // Split the starting summary word so traversal wraps in the same
            // cyclic preference, including a partial final summary word.
            for (var probe = 0u; probe <= graph_config.padding.y; probe += 1u) {
                if probe == graph_config.padding.y && first_bit == 0u { break; }
                var hints = atomicLoad(&graph_queue[graph_config.padding.x + summary_index]);
                if probe == 0u { hints &= 0xffffffffu << first_bit; }
                if probe == graph_config.padding.y { hints &= (1u << first_bit) - 1u; }
                while hints != 0u {
                    let bit = firstTrailingBit(hints);
                    let word = summary_index * 32u + bit;
                    if word < graph_config.ready_words && graph_claim_word(word, false) { break; }
                    hints &= hints - 1u;
                }
                if graph_ticket.x != 0xffffffffu { break; }
                summary_index += 1u;
                if summary_index == graph_config.padding.y { summary_index = 0u; }
            }
            if graph_ticket.x != 0xffffffffu { break; }
            // Summary updates may race or be stale. This authoritative scan
            // preserves the original epoch-wait progress argument and repairs
            // missing hints when it encounters a nonempty ready word.
            for (var probe = 0u; probe < graph_config.ready_words; probe += 1u) {
                let word = (worker + probe) % graph_config.ready_words;
                if graph_claim_word(word, true) { break; }
            }
            if graph_ticket.x != 0xffffffffu { break; }
            graph_idle_remaining = remaining;
        }
    }
    return workgroupUniformLoad(&graph_ticket);
}
