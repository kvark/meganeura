            for (var index = lane; index < 1024u; index += 128u) {
                let ar = row + index / 32u;
                let ak = start + index % 32u;
                let bk = start + index / 32u;
                let bc = col + index % 32u;
                var a = $MULTIPLY_TYPE(0.0f);
                var b = $MULTIPLY_TYPE(0.0f);
                if ar < m && ak < k { a = $A_CAST($A_LOAD); }
                if bk < k && bc < n { b = $B_CAST($B_LOAD); }
                stage_a[index] = a;
                stage_b[index] = b;
            }
