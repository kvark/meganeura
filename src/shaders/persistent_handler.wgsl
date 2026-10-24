if ticket >= persistent_params.starts[$SLOT] && ticket < persistent_params.ends[$SLOT] {
    let grid = persistent_params.grids[$SLOT].xyz;
    let local_ticket = ticket - persistent_params.starts[$SLOT];
    let logical = vec3<u32>(local_ticket % grid.x, (local_ticket / grid.x) % grid.y,
                            local_ticket / (grid.x * grid.y));
    h$SLOT_execute_tile(logical, lid);
}
