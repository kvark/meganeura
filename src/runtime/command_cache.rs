use super::Gpu;
use std::ops::Range;

#[derive(Default)]
pub(super) struct Cache {
    pub enabled: bool,
    chunks: Vec<(Range<usize>, blade_graphics::CommandEncoder)>,
    pub recordings: u64,
    pub replays: u64,
}

impl Cache {
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            ..Default::default()
        }
    }
    pub fn clear(&mut self, gpu: &Gpu) {
        for (_, mut encoder) in self.chunks.drain(..) {
            gpu.destroy_command_encoder(&mut encoder);
        }
    }

    pub fn execute(
        &mut self,
        gpu: &Gpu,
        primary: &mut blade_graphics::CommandEncoder,
        range: Range<usize>,
        record: impl FnOnce(&mut blade_graphics::CommandEncoder),
    ) {
        let index = match self.chunks.iter().position(|entry| entry.0 == range) {
            Some(index) => {
                self.replays += 1;
                index
            }
            None => {
                let mut secondary =
                    gpu.create_secondary_command_encoder(blade_graphics::CommandEncoderDesc {
                        name: "cached graph chunk",
                        buffer_count: 1,
                        manual_barriers: false,
                    });
                secondary.start();
                record(&mut secondary);
                secondary.finish_secondary();
                self.chunks.push((range, secondary));
                self.recordings += 1;
                self.chunks.len() - 1
            }
        };
        primary.execute_secondary(&self.chunks[index].1);
    }
}
