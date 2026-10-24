struct PersistentParams {
    grids: array<vec4<u32>, 3>,
    starts: vec4<u32>,
    ends: vec4<u32>,
}
var<storage, read_write> persistent_queue: array<atomic<u32>>;
var<uniform> persistent_params: PersistentParams;
var<workgroup> persistent_ticket: u32;

@compute @workgroup_size($WORKGROUP)
fn persistent_main(@builtin(local_invocation_id) lid: vec3<u32>,
                   @builtin(workgroup_id) worker: vec3<u32>) {
    loop {
        if lid.x == 0u {
            persistent_ticket = atomicAdd(&persistent_queue[0], 1u);
            if $TRACK_WORKERS && persistent_ticket < persistent_params.ends.w {
                atomicAdd(&persistent_queue[1u + worker.x], 1u);
            }
        }
        let ticket = workgroupUniformLoad(&persistent_ticket);
        if ticket >= persistent_params.ends.w {
            break;
        }
        $HANDLERS
        // Finish the body before another ticket or reuse of reduction scratch.
        workgroupBarrier();
    }
}
