            for (var index = lane; index < 1024u; index += $THREADSu) {
                // Read adjacent storage elements even when the operand is transposed.
                let transposed = (index % 32u) * 32u + index / 32u;
                let a_index = select(index, transposed, $TRANSPOSE_A);
                let b_index = select(index, transposed, $TRANSPOSE_B);
                let ar = row + a_index / 32u;
                let ak = start + a_index % 32u;
                let bk = start + b_index / 32u;
                let bc = col + b_index % 32u;
                var a = $MULTIPLY_TYPE(0.0f);
                var b = $MULTIPLY_TYPE(0.0f);
                if all(vec2<bool>((ar < m), (ak < k))) { a = $A_CAST($A_LOAD); }
                if all(vec2<bool>((bk < k), (bc < n))) { b = $B_CAST($B_LOAD); }
                stage_a[a_index] = a;
                stage_b[b_index] = b;
            }
