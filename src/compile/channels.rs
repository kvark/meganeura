use super::{BufferRef, Dispatch, ExecutionPlan, ShaderEntry};
use crate::{
    DType,
    codegen::{ConvolutionPass, MatrixTypes},
};

impl ExecutionPlan {
    pub(crate) fn pack_convolution_channels(&mut self, shared_bytes: u32) {
        // Both convolution layouts fit below the forward tile's shared budget.
        if shared_bytes < 2 * 64 * 40 * 2 + 64 * 64 * 4 {
            return;
        }
        let mut dispatches = Vec::with_capacity(self.dispatches.len());
        for mut dispatch in std::mem::take(&mut self.dispatches) {
            let (pass, kernel_h, kernel_w, stride, splits, types) = match dispatch.shader {
                ShaderEntry::Conv2dBF16 {
                    kernel_h,
                    kernel_w,
                    stride,
                    output_bf16,
                } => (
                    ConvolutionPass::Forward,
                    kernel_h,
                    kernel_w,
                    stride,
                    1,
                    MatrixTypes {
                        output: if output_bf16 { DType::BF16 } else { DType::F32 },
                        ..MatrixTypes::BF16
                    },
                ),
                ShaderEntry::Conv2dGradientTyped {
                    weight_gradient,
                    kernel_h,
                    kernel_w,
                    stride,
                    splits,
                    types,
                } if types == MatrixTypes::BF16 => (
                    if weight_gradient {
                        ConvolutionPass::WeightGradient
                    } else {
                        ConvolutionPass::InputGradient
                    },
                    kernel_h,
                    kernel_w,
                    stride,
                    splits,
                    types,
                ),
                _ => {
                    dispatches.push(dispatch);
                    continue;
                }
            };
            let p = &dispatch.params;
            let input_spatial = u64::from(p[2]) * u64::from(p[3]);
            let output_spatial = u64::from(p[9]) * u64::from(p[10]);
            let window = u64::from(kernel_h) * u64::from(kernel_w);
            let input_channels = u64::from(p[1]).div_ceil(32) * 32;
            let output_channels = u64::from(p[4]).div_ceil(32) * 32;
            let (Some(input_window), Some(output_window)) = (
                input_channels.checked_mul(window),
                output_channels.checked_mul(window),
            ) else {
                dispatches.push(dispatch);
                continue;
            };
            let (dimensions, grid) = match pass {
                ConvolutionPass::Forward => (
                    [
                        (p[0], p[1], input_spatial, false),
                        (p[4], p[1], window, false),
                    ],
                    [
                        u64::from(p[4]).div_ceil(64),
                        output_spatial.div_ceil(64),
                        u64::from(p[0]),
                    ],
                ),
                ConvolutionPass::InputGradient => (
                    [
                        (p[0], p[4], output_spatial, false),
                        (p[1], p[4], window, true),
                    ],
                    [
                        u64::from(p[1]).div_ceil(64),
                        input_spatial.div_ceil(64),
                        u64::from(p[0]),
                    ],
                ),
                ConvolutionPass::WeightGradient => (
                    [
                        (p[0], p[4], output_spatial, false),
                        (p[0], p[1], input_spatial, false),
                    ],
                    [
                        u64::from(p[4]).div_ceil(64),
                        input_window.div_ceil(64),
                        u64::from(splits),
                    ],
                ),
            };
            let eligible = !dispatch.schedule_locked
                && p[1] >= 16
                && p[4] >= 16
                && grid.iter().all(|&n| n > 0 && n <= 65535)
                && (pass == ConvolutionPass::WeightGradient || grid.iter().product::<u64>() >= 64)
                && input_spatial <= u64::from(u32::MAX)
                && output_spatial <= u64::from(u32::MAX)
                && input_window <= u64::from(u32::MAX)
                && output_window <= u64::from(u32::MAX)
                && dimensions.iter().all(|&(outer, channels, spatial, _)| {
                    let padded = u64::from(channels).div_ceil(32) * 32;
                    spatial <= u64::from(u32::MAX)
                        && u64::from(outer)
                            .checked_mul(spatial)
                            .and_then(|n| n.checked_mul(padded))
                            .is_some_and(|n| n <= 32 << 20 && (n / 2).div_ceil(256) <= 65535)
                });
            if !eligible {
                dispatches.push(dispatch);
                continue;
            }
            for (slot, (outer, channels, spatial, transpose)) in dimensions.into_iter().enumerate()
            {
                let padded = u64::from(channels).div_ceil(32) * 32;
                let elements = u64::from(outer) * spatial * padded;
                let output = BufferRef(self.buffers.len().try_into().unwrap());
                self.buffers.push((elements * 2).try_into().unwrap());
                dispatches.push(Dispatch {
                    shader: ShaderEntry::PackChannels {
                        dtype: DType::BF16,
                        transpose,
                    },
                    workgroups: [(elements / 2).div_ceil(256).try_into().unwrap(), 1, 1],
                    input_buffers: vec![dispatch.input_buffers[slot]],
                    output_buffer: output,
                    params: vec![outer, channels, spatial as u32, padded as u32],
                    fusion_barrier: true,
                    origin: dispatch.origin.clone(),
                    ..Default::default()
                });
                dispatch.input_buffers[slot] = output;
            }
            dispatch.shader = ShaderEntry::Conv2dPacked {
                pass,
                splits,
                kernel_h,
                kernel_w,
                stride,
                types,
            };
            dispatch.workgroups = grid.map(|n| n as u32);
            dispatches.push(dispatch);
        }
        self.dispatches = dispatches;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn derivative_packing_preserves_splits_and_bounded_fallbacks() {
        let mut graph = crate::Graph::new();
        let x = graph.parameter("x", &[64 * 17 * 9 * 11]);
        let w = graph.parameter("w", &[33 * 17 * 8]);
        let x = graph.to_bf16(x);
        let w = graph.to_bf16(w);
        let y = graph.conv2d_hw(x, w, 64, 17, 9, 11, 33, 2, 4, 1, 0, 2);
        let loss = graph.sum_all(y);
        graph.set_outputs(vec![loss]);
        let graph = crate::autodiff::differentiate_with_bf16_gradients(&graph);
        let original = crate::compile::compile(&graph);
        let mut packed = original.clone();
        packed.pack_convolution_channels(32768);
        for (pass, splits) in [
            (ConvolutionPass::InputGradient, 1),
            (ConvolutionPass::WeightGradient, 2),
        ] {
            assert!(packed.dispatches.iter().any(|d| matches!(d.shader, ShaderEntry::Conv2dPacked { pass: p, splits: s, .. } if p == pass && s == splits)));
        }
        assert_eq!(
            packed
                .dispatches
                .iter()
                .filter(|d| matches!(
                    d.shader,
                    ShaderEntry::PackChannels {
                        transpose: true,
                        ..
                    }
                ))
                .count(),
            1
        );
        for weight in [false, true] {
            for change in 0..5 {
                let mut plan = original.clone();
                plan.dispatches.retain(|d| matches!(d.shader, ShaderEntry::Conv2dGradientTyped { weight_gradient, .. } if weight_gradient == weight));
                assert_eq!(plan.dispatches.len(), 1);
                let d = &mut plan.dispatches[0];
                match change {
                    1 => d.schedule_locked = true,
                    2 => d.params[1] = 3,
                    3 => d.params[0] = u32::MAX,
                    4 => {
                        d.params[2] = u32::MAX;
                        d.params[3] = u32::MAX;
                    }
                    _ => (),
                }
                let before = serde_json::to_vec(&plan).unwrap();
                plan.pack_convolution_channels(if change == 0 { 16384 } else { 32768 });
                assert_eq!(serde_json::to_vec(&plan).unwrap(), before);
            }
        }
    }

    fn plan() -> ExecutionPlan {
        let mut graph = crate::Graph::new();
        let x = graph.input("x", &[64 * 17 * 8 * 8]);
        let w = graph.input("w", &[33 * 17 * 9]);
        let x = graph.to_bf16(x);
        let w = graph.to_bf16(w);
        let y = graph.conv2d_hw(x, w, 64, 17, 8, 8, 33, 3, 3, 1, 1, 1);
        graph.set_outputs(vec![y]);
        crate::compile::compile(&graph)
    }
    #[test]
    fn channel_packing_admission_bounds_and_fallbacks() {
        let mut packed = plan();
        packed.pack_convolution_channels(32768);
        assert_eq!(
            packed
                .dispatches
                .iter()
                .filter(|d| matches!(d.shader, ShaderEntry::PackChannels { .. }))
                .count(),
            2
        );
        assert_eq!(
            packed
                .dispatches
                .iter()
                .filter(|d| matches!(d.shader, ShaderEntry::Conv2dPacked { .. }))
                .count(),
            1
        );
        for (shared, change) in [
            (26623, 0),
            (32768, 1),
            (32768, 2),
            (32768, 3),
            (32768, 4),
            (32768, 5),
        ] {
            let mut original = plan();
            let d = original
                .dispatches
                .iter_mut()
                .find(|d| matches!(d.shader, ShaderEntry::Conv2dBF16 { .. }))
                .unwrap();
            match change {
                1 => d.params[1] = 3,
                2 => d.params[0] = 1,
                3 => d.params[2] = 65536,
                4 => {
                    d.params[0] = u32::MAX;
                    d.params[2] = u32::MAX;
                    d.params[3] = u32::MAX;
                }
                5 => d.schedule_locked = true,
                _ => (),
            }
            let before = serde_json::to_vec(&original).unwrap();
            original.pack_convolution_channels(shared);
            assert_eq!(serde_json::to_vec(&original).unwrap(), before);
        }
    }
}
