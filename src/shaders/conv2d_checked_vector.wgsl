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
                // Keep scalar pointers for coopLoadT: Naga's Metal helper
                // does not yet convert vector pointers or their row strides.
                stage_a[index*4u] = a.x; stage_a[index*4u+1u] = a.y;
                stage_a[index*4u+2u] = a.z; stage_a[index*4u+3u] = a.w;
                stage_b[index*4u] = b.x; stage_b[index*4u+1u] = b.y;
                stage_b[index*4u+2u] = b.z; stage_b[index*4u+3u] = b.w;
            }
