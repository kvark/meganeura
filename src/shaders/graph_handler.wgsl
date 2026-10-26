case $INDEXu: {
$CLAIM_GEOMETRY    let lane = $LANE;
    let lid = vec3<u32>(lane % $Xu, (lane / $Xu) % $Yu, lane / ($Xu * $Yu));
    $SUBGROUPS
    let graph_tile_group = min($GROUP, $LAST_GROUP);
    let graph_active_lane = ($GROUP) < $ACTIVE_GROUPS;
    $PARAMETERS
    $REPEAT_BEGIN
    let logical = vec3<u32>($LOGICAL_X, $LOGICAL_Y, $LOGICAL_Z);
    $CALL;
    $REPEAT_END
$COMPLETE}
