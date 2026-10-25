case $INDEXu: {
    let lane = $LANE;
    let lid = vec3<u32>(lane % $Xu, (lane / $Xu) % $Yu, lane / ($Xu * $Yu));
    $SUBGROUPS
    $ACTIVE
    $PARAMETERS
    $CALL;
}
