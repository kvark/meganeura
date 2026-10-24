struct PersistentParams {
    grids: array<vec4<u32>, 3>,
    starts: vec4<u32>,
    ends: vec4<u32>,
    successors: vec4<u32>,
}
var<storage, read_write> persistent_queue: array<atomic<u32>>;
var<uniform> persistent_params: PersistentParams;
// Preserve the 16-byte placement needed by cooperative matrix scratch.
var<workgroup> persistent_ticket: array<u32, 4>;

fn persistent_claim(worker: u32, lane: u32) -> u32 {
    for (var retry = 0u; retry < 64u; retry += 1u) {
        if lane == 0u {
            persistent_ticket[0] = 0xffffffffu;
            if atomicLoad(&persistent_queue[0]) == 0u {
                persistent_ticket[0] = 0xfffffffeu;
            } else {
                for (var probe = 0u; probe < persistent_params.successors.w; probe += 1u) {
                    let word = (worker + probe) % persistent_params.successors.w;
                    let bits = atomicLoad(&persistent_queue[8u + word]);
                    if bits != 0u {
                        let bit = firstTrailingBit(bits);
                        let mask = 1u << bit;
                        let old = atomicAnd(&persistent_queue[8u + word], ~mask);
                        if (old & mask) != 0u {
                            persistent_ticket[0] = word * 32u + bit;
                            storageFence();
                            break;
                        }
                    }
                }
            }
        }
        let ticket = workgroupUniformLoad(&persistent_ticket[0]);
        if ticket != 0xffffffffu {
            return ticket;
        }
    }
    return 0xffffffffu;
}

@compute @workgroup_size($WORKGROUP)
fn persistent_main(@builtin(local_invocation_id) lid: vec3<u32>,
                   @builtin(workgroup_id) worker: vec3<u32>,
                   @builtin(local_invocation_index) lane: u32$SUBGROUPS) {
    for (var iteration = 0u; iteration <= persistent_params.ends.w; iteration += 1u) {
        let ticket = persistent_claim(worker.x, lane);
        if ticket >= 0xfffffffeu {
            return;
        }
        if ticket >= persistent_params.ends.w {
            if lane == 0u { atomicOr(&persistent_queue[1], 1u); }
            return;
        }
        storageFence();
        $HANDLERS
        storageFence();
        workgroupBarrier();
        if lane == 0u {
            storageFence();
            var node = 0u;
            if ticket >= persistent_params.ends.x && ticket < persistent_params.ends.y { node = 1u; }
            if ticket >= persistent_params.ends.y && ticket < persistent_params.ends.z { node = 2u; }
            let remaining = atomicSub(&persistent_queue[2u + node], 1u);
            if remaining == 0u { atomicOr(&persistent_queue[1], 2u); }
            if remaining == 1u {
                storageFence();
                for (var successor = 0u; successor < 3u; successor += 1u) {
                    if (persistent_params.successors[node] & (1u << successor)) != 0u {
                        storageFence();
                        let pending = atomicSub(&persistent_queue[5u + successor], 1u);
                        if pending == 0u { atomicOr(&persistent_queue[1], 4u); }
                        if pending == 1u {
                            storageFence();
                            for (var next = persistent_params.starts[successor];
                                 next < persistent_params.ends[successor]; next += 1u) {
                                atomicOr(&persistent_queue[8u + next / 32u], 1u << (next % 32u));
                            }
                        }
                    }
                }
            }
            if atomicSub(&persistent_queue[0], 1u) == 0u {
                atomicOr(&persistent_queue[1], 8u);
            }
            if $TRACK_WORKERS {
                atomicAdd(&persistent_queue[persistent_params.starts.w + worker.x], 1u);
            }
        }
    }
    if lane == 0u { atomicOr(&persistent_queue[1], 16u); }
}
