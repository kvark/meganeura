var tile_max = max_score;
for (var i = 0u; i < $BKV_U; i++) {
    let kv_pos = t + i;
    if valid && kv_pos >= my_kv_start && kv_pos < my_kv_len {
        let score = wg_scores[i * $WG_SIZE_U + grp_base] * scale;
        tile_max = max(tile_max, score);
    }
}
let correction = exp(max_score - tile_max);
sum_exp *= correction;
$RESCALE_OUTPUT
for (var i = 0u; i < $BKV_U; i++) {
    let kv_pos = t + i;
    if valid && kv_pos >= my_kv_start && kv_pos < my_kv_len {
        let score = wg_scores[i * $WG_SIZE_U + grp_base] * scale;
        let weight = exp(score - tile_max);
        sum_exp += weight;
        $ACCUMULATE_OUTPUT
    }
}
max_score = tile_max;
workgroupBarrier();
