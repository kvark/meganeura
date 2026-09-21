// Rescale once per score tile, not once per key. Masked tiles preserve state.
var tile_max = max_score;
for (var i = 0u; i < $BKV_U; i++) {
    let kv_pos = t + i;
    if valid && kv_pos >= my_kv_start && kv_pos < my_kv_len {
        let score = wg_scores[grp_base + i * $TPQ_U] * scale;
        tile_max = max(tile_max, score);
    }
}
let correction = exp(max_score - tile_max);
sum_exp *= correction;
$RESCALE_OUTPUT
for (var i = 0u; i < $BKV_U; i++) {
    let kv_pos = t + i;
    if valid && kv_pos >= my_kv_start && kv_pos < my_kv_len {
        let score = wg_scores[grp_base + i * $TPQ_U] * scale;
        let weight = exp(score - tile_max);
        sum_exp += weight;
        let v_base = kv_pos * kv_dim + kv_head_off;
        $ACCUMULATE_OUTPUT
    }
}
max_score = tile_max;
workgroupBarrier();
