pub(crate) const SHADER: &str = include_str!("shaders/divisor.wgsl");

pub(crate) fn multiplier(divisor: u32) -> u32 {
    assert_ne!(divisor, 0, "index divisor must be positive");
    // Division by one bypasses multiplication in the shader.
    ((1u64 << 32) / u64::from(divisor)) as u32
}

#[cfg(test)]
mod tests {
    use super::multiplier;

    fn cases() -> Vec<[u32; 4]> {
        let mut divisors: Vec<_> = (1..=256).collect();
        let mut numerators: Vec<_> = (0..=256).collect();
        for bit in 1..32 {
            let value = 1u32 << bit;
            for offset in -2i64..=2 {
                if let Ok(value) = u32::try_from(i64::from(value) + offset) {
                    numerators.push(value);
                    if value != 0 {
                        divisors.push(value);
                    }
                }
            }
        }
        numerators.extend([u32::MAX - 1, u32::MAX]);
        divisors.extend([u32::MAX - 1, u32::MAX]);
        let mut cases = Vec::new();
        for divisor in divisors {
            for &value in &numerators {
                cases.push([value, divisor, multiplier(divisor), 0]);
            }
            for quotient in [1, 2, 3, 41, 47, 55, 65535, u32::MAX / divisor] {
                let boundary = u64::from(divisor) * u64::from(quotient);
                for value in [boundary.saturating_sub(1), boundary, boundary + 1] {
                    if let Ok(value) = u32::try_from(value) {
                        cases.push([value, divisor, multiplier(divisor), 0]);
                    }
                }
            }
        }
        let mut state = 0x243f6a88u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        for i in 0..100_000 {
            let value = next();
            let raw = next();
            let divisor = if i % 2 == 0 { raw } else { raw & 0xffff }.max(1);
            cases.push([value, divisor, multiplier(divisor), 0]);
        }
        cases
    }

    #[test]
    fn reciprocal_estimate_is_at_most_one_low_without_overflow() {
        for [value, divisor, reciprocal, _] in cases() {
            if divisor == 1 {
                assert_eq!(reciprocal, 0);
                continue;
            }
            let estimate = (u64::from(value) * u64::from(reciprocal)) >> 32;
            let expected = value / divisor;
            assert!(estimate <= u64::from(expected));
            assert!(u64::from(expected) - estimate <= 1);
            let product = u32::try_from(estimate * u64::from(divisor)).unwrap();
            let remainder = value.checked_sub(product).unwrap();
            assert_eq!(estimate as u32 + u32::from(remainder >= divisor), expected);
        }
    }

    #[test]
    #[should_panic(expected = "index divisor must be positive")]
    fn zero_divisor_is_rejected() {
        multiplier(0);
    }

    #[test]
    fn gpu_exact_division_and_multiply_high() {
        use bg::ShaderData;
        use blade_graphics as bg;

        #[derive(blade_macros::ShaderData)]
        struct Data {
            src: bg::BufferPiece,
            dst: bg::BufferPiece,
        }

        let source = format!(
            "{}\n\
             var<storage> src: array<vec4<u32>>;\n\
             var<storage, read_write> dst: array<vec2<u32>>;\n\
             @compute @workgroup_size(64)\n\
             fn main(@builtin(global_invocation_id) id: vec3<u32>) {{\n\
                 if id.x < arrayLength(&src) {{\n\
                     let input = src[id.x];\n\
                     dst[id.x] = vec2<u32>(divide_exact(input.x, input.y, input.z),\n\
                         multiply_high(input.x, input.y));\n\
                 }}\n\
             }}",
            super::SHADER
        );
        let gpu = crate::init_gpu_context().unwrap();
        let shader = gpu.create_shader(bg::ShaderDesc {
            source: &source,
            naga_module: None,
        });
        let mut pipeline = gpu.create_compute_pipeline(bg::ComputePipelineDesc {
            name: "exact_division_test",
            data_layouts: &[&Data::layout()],
            compute: shader.at("main"),
        });
        let inputs = cases();
        let src = gpu.create_buffer(bg::BufferDesc {
            name: "exact_division_inputs",
            size: (inputs.len() * 16) as u64,
            memory: bg::Memory::Shared,
        });
        let dst = gpu.create_buffer(bg::BufferDesc {
            name: "exact_division_outputs",
            size: (inputs.len() * 8) as u64,
            memory: bg::Memory::Shared,
        });
        unsafe {
            std::slice::from_raw_parts_mut(src.data().cast::<[u32; 4]>(), inputs.len())
                .copy_from_slice(&inputs);
            std::slice::from_raw_parts_mut(dst.data().cast::<[u32; 2]>(), inputs.len())
                .fill([u32::MAX; 2]);
        }
        let mut encoder = gpu.create_command_encoder(bg::CommandEncoderDesc {
            name: "exact_division_test",
            buffer_count: 1,
            manual_barriers: false,
        });
        encoder.start();
        {
            let mut pass = encoder.compute("exact_division_test");
            let mut command = pass.with(&pipeline);
            command.bind(
                0,
                &Data {
                    src: src.at(0),
                    dst: dst.at(0),
                },
            );
            command.dispatch([(inputs.len() as u32).div_ceil(64), 1, 1]);
        }
        let sync = gpu.submit(&mut encoder);
        assert!(gpu.wait_for(&sync, !0).unwrap());
        let outputs = unsafe {
            std::slice::from_raw_parts(dst.data().cast::<[u32; 2]>(), inputs.len()).to_vec()
        };
        gpu.destroy_command_encoder(&mut encoder);
        gpu.destroy_compute_pipeline(&mut pipeline);
        gpu.destroy_buffer(src);
        gpu.destroy_buffer(dst);
        for ([value, divisor, _, _], [quotient, high]) in inputs.into_iter().zip(outputs) {
            assert_eq!(quotient, value / divisor, "{value} / {divisor}");
            assert_eq!(
                u64::from(high),
                (u64::from(value) * u64::from(divisor)) >> 32,
                "high product: {value} * {divisor}"
            );
        }
    }
}
