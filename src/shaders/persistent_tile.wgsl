struct PersistentParams {
    grid: vec3<u32>,
    count: u32,
}
var<storage, read_write> persistent_queue: array<atomic<u32>>;
var<uniform> persistent_params: PersistentParams;
var<workgroup> persistent_ticket: u32;

@compute @workgroup_size($WORKGROUP)
fn persistent_main(@builtin(local_invocation_id) lid: vec3<u32>) {
    loop {
        if lid.x == 0u {
            persistent_ticket = atomicAdd(&persistent_queue[0], 1u);
        }
        let ticket = workgroupUniformLoad(&persistent_ticket);
        if ticket >= persistent_params.count {
            break;
        }
        let grid = persistent_params.grid;
        let logical = vec3<u32>(ticket % grid.x, (ticket / grid.x) % grid.y,
                                ticket / (grid.x * grid.y));
        execute_tile(logical, lid);
        // Every invocation finishes the body before the next ticket or reuse
        // of reduction scratch. No tensor is published to another worker.
        workgroupBarrier();
    }
}
