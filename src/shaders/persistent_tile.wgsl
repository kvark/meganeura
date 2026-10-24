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

@compute @workgroup_size($WORKGROUP)
fn persistent_main(@builtin(local_invocation_id) lid: vec3<u32>,
                   @builtin(workgroup_id) worker: vec3<u32>,
                   @builtin(local_invocation_index) lane: u32$SUBGROUPS) {
    loop {
        if lane == 0u {
            persistent_ticket[0] = atomicAdd(&persistent_queue[0], 1u);
            if $TRACK_WORKERS && persistent_ticket[0] < persistent_params.ends.w {
                atomicAdd(&persistent_queue[1u + worker.x], 1u);
            }
        }
        let ticket = workgroupUniformLoad(&persistent_ticket[0]);
        if ticket >= persistent_params.ends.w {
            break;
        }
        $HANDLERS
        // Finish the body before another ticket or reuse of reduction scratch.
        workgroupBarrier();
    }
}
