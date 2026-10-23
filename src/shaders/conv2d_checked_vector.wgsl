for (var index = lane; index < 256u; index += $THREADSu) {
                let ar = row + index / 8u;
                let ak = start + (index % 8u) * 4u;
                let bk = start + index / 8u;
                let bc = col + (index % 8u) * 4u;
                var a = vec4<$MULTIPLY_TYPE>();
                var b = vec4<$MULTIPLY_TYPE>();
                if all(vec2<bool>((ar < m), (ak < k))) {
                    let v = weight[(ar * k + ak) / 4u];
                    a = $A_VECTOR;
                }
                if all(vec2<bool>((bk < k), (bc < n))) {
                    let v = src[((group.z * k + bk) * n + bc) / 4u];
                    b = $B_VECTOR;
                }
                stage_a[index] = a;
                stage_b[index] = b;
            }
