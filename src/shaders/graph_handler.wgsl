case $INDEXu: {
    let lane = $LANE;
    let lid = vec3<u32>(lane % $Xu, (lane / $Xu) % $Yu, lane / ($Xu * $Yu));
    $SUBGROUPS
    let graph_tile_group = min($GROUP, $LAST_GROUPu);
    let graph_active_lane = ($GROUP) < (node.grid.w >> 16u);
    let logical = vec3<u32>($LOGICAL_X, $LOGICAL_Y, $LOGICAL_Z);
    $PARAMETERS
    $CALL;
}
