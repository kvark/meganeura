let a0 = coopLoadT<coop_mat16x16<$MULTIPLY_TYPE, A>>(&stage_a[a_offset], 32u);
let b0 = coopLoadT<coop_mat16x16<$MULTIPLY_TYPE, B>>(&stage_b[b_offset], 32u);
accumulator = coopMultiplyAdd(a0, b0, accumulator);
let a1 = coopLoadT<coop_mat16x16<$MULTIPLY_TYPE, A>>(&stage_a[a_offset + 16u], 32u);
let b1 = coopLoadT<coop_mat16x16<$MULTIPLY_TYPE, B>>(&stage_b[b_offset + 512u], 32u);
accumulator = coopMultiplyAdd(a1, b1, accumulator);
