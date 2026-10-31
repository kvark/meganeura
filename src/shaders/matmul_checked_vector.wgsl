{
    let out_pos = $OUTPUT_BASE + select((index % 8u) * 4u, index / 8u, $OUTPUT_OUTER);
    let k_pos = start + select(index / 8u, (index % 8u) * 4u, $OUTPUT_OUTER);
    let at = $GLOBAL_INDEX;
    var value = vec4<$MULTIPLY_TYPE>();
    if $UNCHECKED || (out_pos < $OUTPUT_LIMIT && k_pos < k) {
        let remaining = select($OUTPUT_LIMIT - out_pos, k - k_pos, $OUTPUT_OUTER);
        if $UNCHECKED || (at % 4u == 0u && remaining >= 4u) {
            let raw = matrix_$OPERAND[at / 4u];
            value = $VECTOR;
        } else {
            for (var e = 0u; e < 4u; e++) {
                let out_e = out_pos + select(e, 0u, $OUTPUT_OUTER);
                let k_e = k_pos + select(0u, e, $OUTPUT_OUTER);
                if out_e < $OUTPUT_LIMIT && k_e < k {
                    value[e] = $CAST(matrix_$OPERAND[(at + e) / 4u][(at + e) % 4u]);
                }
            }
        }
    }
    for (var e = 0u; e < 4u; e++) {
        let linear = index * 4u + e;
        let transposed = (linear % 32u) * 32u + linear / 32u;
        stage_$OPERAND[select(linear, transposed, $TRANSPOSE)] = value[e];
    }
}
