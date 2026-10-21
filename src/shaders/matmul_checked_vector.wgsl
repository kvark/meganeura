            for (var index = lane; index < 256u; index += 128u) {
                let ar0 = row + select(index / 8u, (index % 8u) * 4u, $TRANSPOSE_A);
                let ak0 = start + select((index % 8u) * 4u, index / 8u, $TRANSPOSE_A);
                let bk0 = start + select(index / 8u, (index % 8u) * 4u, $TRANSPOSE_B);
                let bc0 = col + select((index % 8u) * 4u, index / 8u, $TRANSPOSE_B);
                let ai = select(ar0 * k + ak0, ak0 * m + ar0, $TRANSPOSE_A);
                let bi = select(bk0 * n + bc0, bc0 * k + bk0, $TRANSPOSE_B);
                var a = vec4<$MULTIPLY_TYPE>();
                var b = vec4<$MULTIPLY_TYPE>();
                if ar0 < m && ak0 < k {
                    let remaining = select(k - ak0, m - ar0, $TRANSPOSE_A);
                    if ai % 4u == 0u && remaining >= 4u {
                        let v = matrix_a[ai / 4u];
                        a = $A_VECTOR;
                    } else {
                        for (var e = 0u; e < 4u; e++) {
                            let ar = ar0 + select(0u, e, $TRANSPOSE_A);
                            let ak = ak0 + select(e, 0u, $TRANSPOSE_A);
                            if ar < m && ak < k { a[e] = $A_CAST($A_LOAD); }
                        }
                    }
                }
                if bk0 < k && bc0 < n {
                    let remaining = select(n - bc0, k - bk0, $TRANSPOSE_B);
                    if bi % 4u == 0u && remaining >= 4u {
                        let v = matrix_b[bi / 4u];
                        b = $B_VECTOR;
                    } else {
                        for (var e = 0u; e < 4u; e++) {
                            let bk = bk0 + select(0u, e, $TRANSPOSE_B);
                            let bc = bc0 + select(e, 0u, $TRANSPOSE_B);
                            if bk < k && bc < n { b[e] = $B_CAST($B_LOAD); }
                        }
                    }
                }
                for (var e = 0u; e < 4u; e++) {
                    let linear = index * 4u + e;
                    let transposed = (linear % 32u) * 32u + linear / 32u;
                    stage_a[select(linear, transposed, $TRANSPOSE_A)] = a[e];
                    stage_b[select(linear, transposed, $TRANSPOSE_B)] = b[e];
                }
            }
