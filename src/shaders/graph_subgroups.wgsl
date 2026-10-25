if physical_lane == 0u { atomicStore(&graph_geometry_error[0], 0u); }
workgroupBarrier();
if $SUBGROUP_THREADSu % graph_subgroup_size != 0u
    || graph_num_subgroups * graph_subgroup_size != $THREADSu
    || physical_lane != graph_subgroup_id * graph_subgroup_size + graph_subgroup_lane {
    atomicOr(&graph_geometry_error[0], 1u);
}
workgroupBarrier();
if physical_lane == 0u { graph_ticket.x = atomicLoad(&graph_geometry_error[0]); }
if workgroupUniformLoad(&graph_ticket).x != 0u {
    if physical_lane == 0u { atomicOr(&graph_queue[1], 16u); }
    return;
}
