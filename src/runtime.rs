use crate::compile::{BufferRef, Dispatch, ExecutionPlan, ShaderEntry};
use std::collections::HashMap;

/// A compiled, ready-to-execute GPU session.
///
/// Holds all blade-graphics resources: context, buffers, pipelines.
/// Calling `step()` replays the pre-compiled dispatch sequence.
pub struct Session {
    gpu: blade_graphics::Context,
    buffers: Vec<blade_graphics::Buffer>,
    pipelines: HashMap<ShaderEntry, blade_graphics::ComputePipeline>,
    plan: ExecutionPlan,
    sync_point: Option<blade_graphics::SyncPoint>,
}

impl Session {
    /// Create a session from a compiled execution plan.
    ///
    /// This allocates all GPU buffers and creates compute pipelines.
    pub fn new(plan: ExecutionPlan) -> Self {
        // Safety: we only create one GPU context per session, and the
        // context is used exclusively through this Session.
        let gpu = unsafe {
            blade_graphics::Context::init(blade_graphics::ContextDesc {
                validation: cfg!(debug_assertions),
                capture: false,
                overlay: false,
                device_id: 0,
                ..Default::default()
            })
        }
        .expect("failed to initialize blade GPU context");

        // Allocate GPU buffers
        let buffers: Vec<blade_graphics::Buffer> = plan
            .buffers
            .iter()
            .enumerate()
            .map(|(i, &size)| {
                let size = size.max(4); // minimum 4 bytes
                gpu.create_buffer(blade_graphics::BufferDesc {
                    name: &format!("buf_{}", i),
                    size: size as u64,
                    memory: blade_graphics::Memory::Shared,
                })
            })
            .collect();

        // Collect unique shader files needed
        let mut shader_files: HashMap<&str, Vec<&ShaderEntry>> = HashMap::new();
        for dispatch in &plan.dispatches {
            shader_files
                .entry(dispatch.shader.shader_file())
                .or_default()
                .push(&dispatch.shader);
        }

        // Create compute pipelines
        let mut pipelines = HashMap::new();
        for (file, entries) in &shader_files {
            let source = std::fs::read_to_string(file)
                .unwrap_or_else(|e| panic!("failed to read shader {}: {}", file, e));
            let shader = gpu.create_shader(blade_graphics::ShaderDesc { source: &source });

            for entry in entries {
                if pipelines.contains_key(*entry) {
                    continue;
                }
                let pipeline =
                    gpu.create_compute_pipeline(blade_graphics::ComputePipelineDesc {
                        name: (*entry).entry_point(),
                        data_layouts: &[],
                        compute: shader.at((*entry).entry_point()),
                    });
                pipelines.insert((*entry).clone(), pipeline);
            }
        }

        Self {
            gpu,
            buffers,
            pipelines,
            plan,
            sync_point: None,
        }
    }

    /// Upload parameter data to GPU buffers.
    pub fn set_parameter(&mut self, name: &str, data: &[f32]) {
        for &(ref param_name, buf_ref) in &self.plan.param_buffers {
            if param_name == name {
                self.upload_buffer(buf_ref, bytemuck::cast_slice(data));
                return;
            }
        }
        panic!("unknown parameter: {}", name);
    }

    /// Upload input data.
    pub fn set_input(&mut self, name: &str, data: &[f32]) {
        for &(ref input_name, buf_ref) in &self.plan.input_buffers {
            if input_name == name {
                self.upload_buffer(buf_ref, bytemuck::cast_slice(data));
                return;
            }
        }
        panic!("unknown input: {}", name);
    }

    fn upload_buffer(&self, buf_ref: BufferRef, data: &[u8]) {
        let buffer = &self.buffers[buf_ref.0 as usize];
        unsafe {
            let ptr = buffer.data();
            std::ptr::copy_nonoverlapping(data.as_ptr(), ptr, data.len());
        }
    }

    /// Read back the loss value.
    pub fn read_loss(&self) -> f32 {
        if let Some(buf_ref) = self.plan.loss_buffer {
            let buffer = &self.buffers[buf_ref.0 as usize];
            unsafe {
                let ptr = buffer.data() as *const f32;
                *ptr
            }
        } else {
            0.0
        }
    }

    /// Read back a buffer's contents.
    pub fn read_buffer(&self, buf_ref: BufferRef, out: &mut [f32]) {
        let buffer = &self.buffers[buf_ref.0 as usize];
        unsafe {
            let ptr = buffer.data() as *const f32;
            std::ptr::copy_nonoverlapping(ptr, out.as_mut_ptr(), out.len());
        }
    }

    /// Wait for any pending GPU work.
    pub fn wait(&mut self) {
        if let Some(sp) = self.sync_point.take() {
            self.gpu.wait_for(&sp, !0);
        }
    }

    /// Execute the full dispatch sequence (forward + backward + update).
    pub fn step(&mut self) {
        self.wait();

        let mut encoder = self.gpu.create_command_encoder(blade_graphics::CommandEncoderDesc {
            name: "meganeura_step",
            buffer_count: self.plan.dispatches.len() as u32 + 1,
        });

        for dispatch in &self.plan.dispatches {
            self.execute_dispatch(&mut encoder, dispatch);
        }

        self.sync_point = Some(self.gpu.submit(&mut encoder));
    }

    fn execute_dispatch(
        &self,
        encoder: &mut blade_graphics::CommandEncoder,
        _dispatch: &Dispatch,
    ) {
        // Note: The actual binding depends on blade's ShaderData system.
        // For now, we encode a compute pass per dispatch.
        // The real binding logic requires knowing blade's internal
        // pipeline layout expectations, which varies by shader.
        //
        // This is a structural sketch — full binding will require
        // blade_macros::ShaderData structs generated per shader.
        {
            let mut _pass = encoder.compute(_dispatch.shader.entry_point());
            // TODO: bind shader data and dispatch
            // let mut pipe = pass.with(&self.pipelines[&dispatch.shader]);
            // pipe.dispatch(dispatch.workgroups);
        }
    }

    /// Apply SGD updates to all parameters.
    pub fn sgd_step(&mut self, _learning_rate: f32) {
        self.wait();

        let mut encoder = self.gpu.create_command_encoder(blade_graphics::CommandEncoderDesc {
            name: "sgd_update",
            buffer_count: self.plan.param_grad_pairs.len() as u32 + 1,
        });

        for &(param_buf, _grad_buf) in &self.plan.param_grad_pairs {
            let _param_size = self.plan.buffers[param_buf.0 as usize];
            // TODO: dispatch SGD shader
            // For now, do CPU-side SGD as a fallback
        }

        self.sync_point = Some(self.gpu.submit(&mut encoder));
    }

    /// CPU-fallback SGD update (used until GPU SGD shader binding is complete).
    pub fn sgd_step_cpu(&mut self, learning_rate: f32) {
        self.wait();
        for &(param_buf, grad_buf) in &self.plan.param_grad_pairs {
            let size = self.plan.buffers[param_buf.0 as usize] / 4;
            let param = &self.buffers[param_buf.0 as usize];
            let grad = &self.buffers[grad_buf.0 as usize];
            unsafe {
                let p = param.data() as *mut f32;
                let g = grad.data() as *const f32;
                for i in 0..size {
                    *p.add(i) -= learning_rate * *g.add(i);
                }
            }
        }
    }

    pub fn plan(&self) -> &ExecutionPlan {
        &self.plan
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.wait();
        for buffer in &self.buffers {
            self.gpu.destroy_buffer(*buffer);
        }
    }
}
