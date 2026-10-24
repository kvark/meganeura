if ticket >= persistent_params.starts[$SLOT] && ticket < persistent_params.ends[$SLOT] {
    let grid = persistent_params.grids[$SLOT].xyz;
    let local_ticket = ticket - persistent_params.starts[$SLOT];
    let logical = vec3<u32>(local_ticket % grid.x, (local_ticket / grid.x) % grid.y,
                            local_ticket / (grid.x * grid.y));
    $CALL;
    if $CHAIN {
        // Each producer and consumer of this tile belongs to this workgroup.
        // No workgroup reads another workgroup's output inside the chain.
        storageBarrier();
    }
}
