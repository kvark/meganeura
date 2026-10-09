//! Parameter uploads, input writes and result readback.

use super::*;

impl Session {
    /// Whether the plan has a parameter with this name.
    pub fn has_parameter(&self, name: &str) -> bool {
        self.plan.param_buffers.iter().any(|entry| entry.0 == name)
    }

    /// Whether the plan has an input with this name.
    pub fn has_input(&self, name: &str) -> bool {
        self.plan.input_buffers.iter().any(|entry| entry.0 == name)
    }

    /// Upload parameter data to GPU buffers.
    pub fn set_parameter(&mut self, name: &str, data: &[f32]) {
        // Host writes target the same host-coherent allocation read by GPU
        // dispatches. Finish the previous submission before overwriting it;
        // callers should not need a hidden `wait()` correctness precondition.
        self.wait();
        // Check regular parameters first
        for &(ref param_name, buf_ref) in &self.plan.param_buffers {
            if param_name == name {
                if let Some(&(fmt, rows, cols)) = self.plan.weight_buffers.get(&buf_ref) {
                    let packed = quantize::encode_parameter(name, data, fmt, rows, cols);
                    self.upload_parameter_bytes(buf_ref, &packed);
                } else {
                    self.upload_parameter_bytes(buf_ref, bytemuck::cast_slice(data));
                }

                // If this source param feeds a derived param, fill the
                // derived buffer according to the transform type.
                let derived: Vec<_> = self
                    .plan
                    .derived_params
                    .iter()
                    .filter(|entry| entry.1.iter().any(|s| s.0 == name))
                    .cloned()
                    .collect();
                for (derived_buf, sources, transform) in derived {
                    let sources = sources.as_slice();
                    match transform {
                        crate::graph::ParamTransform::VerticalConcat => {
                            let ty = &self.plan.param_types[&derived_buf];
                            let bytes = match ty.dtype {
                                crate::graph::DType::F32 => bytemuck::cast_slice(data).to_vec(),
                                crate::graph::DType::F16 => data
                                    .iter()
                                    .flat_map(|&v| half::f16::from_f32(v).to_le_bytes())
                                    .collect(),
                                _ => unreachable!("row concatenation requires dense weights"),
                            };
                            self.copy_parameter_rows(derived_buf, name, &bytes, sources);
                        }
                        crate::graph::ParamTransform::HorizontalConcat => {
                            let total_cols: usize = sources.iter().map(|s| s.1).sum();
                            let derived_fmt = self
                                .plan
                                .weight_buffers
                                .get(&derived_buf)
                                .map(|&(f, _, _)| f)
                                .unwrap_or(crate::compile::WeightFormat::F32);

                            if derived_fmt.uses_reduced_storage() {
                                // Packing runs per column, so encode this source
                                // and merge it into the canonical packed staging.
                                // This also keeps f32 and packed upload calls in
                                // sync when callers mix the two APIs.
                                let &(_, rows, _) =
                                    self.plan.weight_buffers.get(&derived_buf).unwrap();
                                let src_cols = sources
                                    .iter()
                                    .find(|src| src.0 == name)
                                    .map(|src| src.1)
                                    .unwrap();
                                let packed = quantize::encode_parameter(
                                    name,
                                    data,
                                    derived_fmt,
                                    rows,
                                    src_cols,
                                );
                                self.restage_packed_concat(derived_buf, name, &packed, sources);
                            } else {
                                // f32: direct copy into GPU buffer
                                let buf_f32 =
                                    self.plan.param_types.get(&derived_buf).map_or(
                                        self.plan.buffers[derived_buf.0 as usize] / 4,
                                        |ty| ty.num_elements(),
                                    );
                                let rows = buf_f32.checked_div(total_cols).unwrap_or(0);
                                let mut col_offset = 0usize;
                                for src in sources {
                                    if src.0 == name && rows > 0 {
                                        let src_cols = src.1;
                                        if self.logical_host_visible(derived_buf) {
                                            let derived_ptr = self.buffers[derived_buf.0 as usize]
                                                .data()
                                                as *mut f32;
                                            for r in 0..rows {
                                                let src_start = r * src_cols;
                                                let dst_start = r * total_cols + col_offset;
                                                unsafe {
                                                    std::ptr::copy_nonoverlapping(
                                                        data[src_start..].as_ptr(),
                                                        derived_ptr.add(dst_start),
                                                        src_cols,
                                                    );
                                                }
                                            }
                                        } else {
                                            self.copy_parameter_columns(
                                                buf_ref,
                                                derived_buf,
                                                data,
                                                rows,
                                                src_cols,
                                                total_cols,
                                                col_offset,
                                            );
                                        }
                                    }
                                    col_offset += src.1;
                                }
                            }
                        }
                    }
                }

                return;
            }
        }
        panic!("unknown parameter: {}", name);
    }

    /// Upload a reduced-storage parameter in the exact representation
    /// described by the graph's [`crate::graph::DType`].
    ///
    /// This is intended for checkpoint readers which can losslessly transcode
    /// an external packed tensor into Meganeura's storage layout. Unlike
    /// [`Self::set_parameter`], this method does not dequantize or requantize
    /// the values. The byte count is validated against the logical tensor
    /// size before anything is copied to the device.
    ///
    /// Derived gate/up concatenations are restaged in their storage format.
    /// There is no K-quant encoder here, and
    /// Q4/Q6_K packed blobs are not a byte-append of their sources.
    pub fn set_parameter_packed(&mut self, name: &str, data: &[u8]) {
        self.wait();
        for &(ref param_name, buf_ref) in &self.plan.param_buffers {
            if param_name != name {
                continue;
            }
            let ty = self
                .plan
                .param_types
                .get(&buf_ref)
                .unwrap_or_else(|| panic!("parameter `{name}` has no tensor type"));
            assert!(
                crate::compile::WeightFormat::from_dtype(ty.dtype).uses_reduced_storage(),
                "parameter `{name}` is not packed; use set_parameter"
            );
            let expected = ty.size_bytes();
            assert_eq!(
                data.len(),
                expected,
                "packed parameter `{name}` needs {expected} bytes, got {}",
                data.len()
            );
            self.upload_parameter_bytes(buf_ref, data);
            let derived: Vec<_> = self
                .plan
                .derived_params
                .iter()
                .filter(|entry| entry.1.iter().any(|s| s.0 == name))
                .cloned()
                .collect();
            for (derived_buf, sources, transform) in derived {
                match transform {
                    crate::graph::ParamTransform::HorizontalConcat => {
                        self.restage_packed_concat(derived_buf, name, data, &sources);
                    }
                    crate::graph::ParamTransform::VerticalConcat => {
                        self.copy_parameter_rows(derived_buf, name, data, &sources);
                    }
                }
            }
            return;
        }
        panic!("unknown parameter: {name}");
    }

    pub(super) fn copy_parameter_rows(
        &self,
        derived_buf: BufferRef,
        name: &str,
        data: &[u8],
        sources: &[(String, usize)],
    ) {
        let ty = &self.plan.param_types[&derived_buf];
        assert!(matches!(
            ty.dtype,
            crate::graph::DType::F32 | crate::graph::DType::F16
        ));
        assert_eq!(ty.shape.len(), 2);
        assert_eq!(ty.shape[0], sources.iter().map(|s| s.1).sum::<usize>());
        let row_bytes = ty.shape[1]
            * if ty.dtype == crate::graph::DType::F16 {
                2
            } else {
                4
            };
        let host_visible = self.logical_host_visible(derived_buf);
        let mut offset = 0usize;
        for &(ref source, rows) in sources {
            let bytes = rows * row_bytes;
            if source == name {
                assert_eq!(data.len(), bytes);
                if host_visible || (offset.is_multiple_of(4) && bytes.is_multiple_of(4)) {
                    self.write_raw_buffer_at(
                        piece_at(self.buffers[derived_buf.0 as usize], offset as u64),
                        data,
                        host_visible,
                    );
                } else {
                    // Odd f16 row ranges need a word-aligned transfer. Preserve
                    // the current device image, including checkpoint/shared
                    // updates, instead of keeping a stale host-side copy.
                    let capacity = self.plan.buffers[derived_buf.0 as usize];
                    assert!(capacity.is_multiple_of(4));
                    let mut image = vec![0.0f32; capacity / 4];
                    self.read_buffer(derived_buf, &mut image);
                    let image: &mut [u8] = bytemuck::cast_slice_mut(&mut image);
                    image[offset..offset + bytes].copy_from_slice(data);
                    self.upload_buffer(derived_buf, image);
                }
            }
            offset += bytes;
        }
    }

    pub(super) fn restage_packed_concat(
        &mut self,
        derived_buf: BufferRef,
        name: &str,
        data: &[u8],
        sources: &[(String, usize)],
    ) {
        let ty = self
            .plan
            .param_types
            .get(&derived_buf)
            .unwrap_or_else(|| panic!("derived concat has no tensor type"));
        assert_eq!(ty.shape.len(), 2);
        let rows = ty.shape[0];
        let total_cols: usize = sources.iter().map(|s| s.1).sum();
        assert_eq!(ty.shape[1], total_cols);
        let expected = ty.size_bytes();
        let fmt = crate::compile::WeightFormat::from_dtype(ty.dtype);
        let mut staging = self
            .packed_concat_staging
            .remove(&derived_buf)
            .unwrap_or_else(|| vec![0u8; expected]);
        if staging.len() != expected {
            staging = vec![0u8; expected];
        }
        let mut col_offset = 0usize;
        for src in sources {
            if src.0 == name {
                scatter_packed_concat_columns(
                    &mut staging,
                    data,
                    fmt,
                    rows,
                    src.1,
                    total_cols,
                    col_offset,
                );
            }
            col_offset += src.1;
        }
        self.upload_parameter_bytes(derived_buf, &staging);
        self.packed_concat_staging.insert(derived_buf, staging);
    }

    /// Upload input data.
    pub fn set_input(&mut self, name: &str, data: &[f32]) {
        self.wait();
        for &(ref input_name, buf_ref) in &self.plan.input_buffers {
            if input_name == name {
                self.upload_input_bytes(buf_ref, bytemuck::cast_slice(data));
                return;
            }
        }
        panic!("unknown input: {}", name);
    }

    /// Raw host pointer and byte size for the named input slot's
    /// backing buffer. Input buffers are pinned by the memory plan and
    /// allocated as `Memory::Shared` (device-local + host-visible +
    /// host-coherent), so writes through this pointer go straight into
    /// the GPU-side buffer — no staging, no explicit upload, no
    /// `VK_EXT_external_memory_host` import needed. (Step-local
    /// intermediates may live in `Memory::Device` instead, but those
    /// are never exposed through this API.)
    ///
    /// Returns `None` if the input is absent or its external allocation is not host-visible.
    ///
    /// # Ordering
    ///
    /// Host writes to host-coherent memory are made visible to a
    /// subsequent `step()` via Vulkan's implicit host-memory-domain
    /// barrier on queue submit — no explicit flush required. A
    /// `wait()` from the previous `step()` must have completed
    /// before the host begins writing, otherwise the next frame
    /// races the GPU's in-flight read of the previous frame.
    ///
    /// # Safety
    ///
    /// The returned pointer is valid as long as the `Session`
    /// lives. The caller must not write beyond `size_bytes` and
    /// must respect the alignment of whatever type they interpret
    /// the memory as.
    pub fn input_host_ptr(&self, name: &str) -> Option<(*mut u8, usize)> {
        for &(ref input_name, buf_ref) in &self.plan.input_buffers {
            if input_name == name {
                if !self.logical_host_visible(buf_ref) {
                    return None;
                }
                let buffer = self.buffers[buf_ref.0 as usize];
                let size = self.plan.buffers[buf_ref.0 as usize];
                return Some((buffer.data(), size));
            }
        }
        None
    }

    /// Return the underlying blade `BufferPiece` for the named input
    /// slot, so a sibling compute pipeline running on the *same* shared
    /// `blade_graphics::Context` (see [`Session::with_context`]) can
    /// dispatch directly into it — no fd/dmabuf interop, no
    /// `bind_external_buffer` round-trip.
    ///
    /// Returns `None` if no input with that name exists.  The returned
    /// `BufferPiece` is a lightweight `Copy` handle into the
    /// session-owned buffer; it remains valid for the lifetime of the
    /// `Session`.  Writes through it must respect the same ordering
    /// rules as [`Session::input_host_ptr`] (the previous `step()`'s
    /// `wait()` must have completed before the producer begins
    /// writing).
    pub fn input_buffer(&self, name: &str) -> Option<blade_graphics::BufferPiece> {
        for &(ref input_name, buf_ref) in &self.plan.input_buffers {
            if input_name == name {
                return Some(self.buffers[buf_ref.0 as usize]);
            }
        }
        None
    }

    /// Return the underlying blade `BufferPiece` for a graph output.
    ///
    /// Graph outputs are pinned by the memory planner, so the returned handle
    /// remains stable for the lifetime of the session. This lets a sibling
    /// compute pipeline on the same [`blade_graphics::Context`] consume a
    /// prediction directly, without a device-to-host readback followed by an
    /// upload.
    ///
    /// # Ordering
    ///
    /// Call [`Session::step`] before recording or submitting the consumer.
    /// Submissions to the shared context's queue are ordered, so the consumer
    /// can be submitted after `step()` without a host-side [`Session::wait`].
    /// The handle must not be used after the session is destroyed.
    pub fn output_buffer(&self, index: usize) -> Option<blade_graphics::BufferPiece> {
        let &buf_ref = self.plan.output_buffers.get(index)?;
        Some(self.buffers[buf_ref.0 as usize])
    }

    /// Upload u32 input data (e.g. token IDs for embedding lookup).
    pub fn set_input_u32(&mut self, name: &str, data: &[u32]) {
        self.wait();
        for &(ref input_name, buf_ref) in &self.plan.input_buffers {
            if input_name == name {
                self.upload_input_bytes(buf_ref, bytemuck::cast_slice(data));
                return;
            }
        }
        panic!("unknown input: {}", name);
    }

    fn upload_input_bytes(&self, buffer: BufferRef, data: &[u8]) {
        let capacity = self.plan.buffers[buffer.0 as usize];
        let logical = self
            .plan
            .input_types
            .get(&buffer)
            .map_or(capacity, |ty| ty.size_bytes());
        self.upload_padded_bytes(buffer, data, logical);
    }

    pub(super) fn upload_parameter_bytes(&self, buffer: BufferRef, data: &[u8]) {
        let capacity = self.plan.buffers[buffer.0 as usize];
        let logical = self
            .plan
            .param_types
            .get(&buffer)
            .map_or(capacity, |ty| ty.size_bytes());
        self.upload_padded_bytes(buffer, data, logical);
    }

    fn upload_padded_bytes(&self, buffer: BufferRef, data: &[u8], logical: usize) {
        let capacity = self.plan.buffers[buffer.0 as usize];
        if data.len() == logical && logical < capacity {
            let mut padded = vec![0; capacity];
            padded[..logical].copy_from_slice(data);
            self.upload_buffer(buffer, &padded);
        } else {
            // Retain compatibility with callers supplying a full padded slot;
            // every other short/mismatched upload remains an error.
            self.upload_buffer(buffer, data);
        }
    }

    pub(super) fn copy_parameter_columns(
        &self,
        source: BufferRef,
        destination: BufferRef,
        data: &[f32],
        rows: usize,
        columns: usize,
        destination_columns: usize,
        column_offset: usize,
    ) {
        assert!(columns + column_offset <= destination_columns);
        assert!(rows * destination_columns * 4 <= self.plan.buffers[destination.0 as usize]);
        assert!(rows * columns <= data.len());
        if columns == 0 || rows == 0 {
            return;
        }
        let source_is_f32 = self
            .plan
            .param_types
            .get(&source)
            .is_none_or(|ty| ty.dtype == crate::graph::DType::F32)
            && self
                .plan
                .weight_buffers
                .get(&source)
                .is_none_or(|&(format, _, _)| !format.uses_reduced_storage());
        if !source_is_f32 {
            for row in 0..rows {
                self.write_raw_buffer_at(
                    piece_at(
                        self.buffers[destination.0 as usize],
                        ((row * destination_columns + column_offset) * 4) as u64,
                    ),
                    bytemuck::cast_slice(&data[row * columns..(row + 1) * columns]),
                    false,
                );
            }
            return;
        }
        assert!(rows * columns * 4 <= self.plan.buffers[source.0 as usize]);
        let mut encoder = self
            .gpu
            .create_command_encoder(blade_graphics::CommandEncoderDesc {
                name: "parameter_columns",
                buffer_count: 1,
                manual_barriers: false,
            });
        encoder.start();
        {
            let mut transfer = encoder.transfer("parameter_columns");
            for row in 0..rows {
                transfer.copy_buffer_to_buffer(
                    piece_at(self.buffers[source.0 as usize], (row * columns * 4) as u64),
                    piece_at(
                        self.buffers[destination.0 as usize],
                        ((row * destination_columns + column_offset) * 4) as u64,
                    ),
                    (columns * 4) as u64,
                );
            }
        }
        let sync = self.gpu.submit(&mut encoder);
        let _ = wait_for_timed_encoder(&self.gpu, &sync, &mut encoder, self.gpu_timing);
        self.gpu.destroy_command_encoder(&mut encoder);
    }

    pub(super) fn upload_buffer(&self, buf_ref: BufferRef, data: &[u8]) {
        let buffer = self.buffers[buf_ref.0 as usize];
        let expected = self.plan.buffers[buf_ref.0 as usize];
        // All upload paths (set_parameter, set_input, set_input_u32,
        // upload_param, gradient clip rewrite, checkpoint restore) are
        // *full-buffer* writes — partial uploads silently leave the
        // tail of the GPU buffer at whatever its previous contents were
        // (zero on first allocation, stale data afterwards) and propagate
        // as garbage through every kernel that reads the slot. The
        // canonical instance was kindle's V2-S BN bias parameter, declared
        // as `[batch * channels * area]` in the graph but loaded from a
        // `[channels]` safetensor — at batch=1 the conv kernel's per-area
        // broadcast covered the gap, but at batch>1 lane slices 1..N
        // were uninitialized and the agent silently failed to learn for
        // 50 000 steps. Asserting here catches the class.
        assert_eq!(
            data.len(),
            expected,
            "upload_buffer: byte-size mismatch for buffer {} — got {} bytes, slot expects {}. \
             Likely a parameter shape mismatch between the graph declaration and the source data \
             (e.g. graph says [batch * channels * area], safetensor says [channels]).",
            buf_ref.0,
            data.len(),
            expected,
        );
        self.write_raw_buffer(buffer, data, self.logical_host_visible(buf_ref));
    }

    /// Whether this logical buffer's physical allocation is CPU-accessible.
    ///
    /// Do not probe `Buffer::data()` for this: on Metal, `Memory::Device`
    /// (`MTLStorageModePrivate`) still returns a non-null `contents()`
    /// pointer, and touching it trips
    /// `validateCPUWriteable` / `validateCPUReadable`.
    pub(super) fn logical_host_visible(&self, buf_ref: BufferRef) -> bool {
        !self.alias.device_local[self.alias.map[buf_ref.0 as usize]]
    }

    /// Write `data` into a GPU buffer, staging through a host-visible
    /// allocation when the destination is device-local.
    pub(super) fn write_raw_buffer(
        &self,
        buffer: blade_graphics::BufferPiece,
        data: &[u8],
        host_visible: bool,
    ) {
        self.write_raw_buffer_at(buffer, data, host_visible);
    }

    pub(super) fn write_raw_buffer_at(
        &self,
        destination: blade_graphics::BufferPiece,
        data: &[u8],
        host_visible: bool,
    ) {
        if data.is_empty() {
            return;
        }
        if host_visible {
            unsafe {
                std::ptr::copy_nonoverlapping(data.as_ptr(), destination.data(), data.len());
            }
            return;
        }
        let staging_bytes = data.len().clamp(4, 16 * 1024 * 1024);
        let mut cached = self.upload_staging.borrow_mut();
        if cached
            .as_ref()
            .is_some_and(|staging| staging.size < staging_bytes)
        {
            self.gpu.destroy_buffer(cached.take().unwrap().buffer);
        }
        let staging = cached.get_or_insert_with(|| UploadStaging {
            buffer: self.gpu.create_buffer(blade_graphics::BufferDesc {
                name: "upload_staging",
                size: staging_bytes as u64,
                memory: blade_graphics::Memory::Upload,
            }),
            size: staging_bytes,
        });
        let mut encoder = self
            .gpu
            .create_command_encoder(blade_graphics::CommandEncoderDesc {
                name: "upload_staging",
                buffer_count: 1,
                manual_barriers: false,
            });
        for (index, chunk) in data.chunks(staging.size).enumerate() {
            unsafe {
                std::ptr::copy_nonoverlapping(chunk.as_ptr(), staging.buffer.data(), chunk.len());
            }
            encoder.start();
            encoder.transfer("upload_staging").copy_buffer_to_buffer(
                staging.buffer.at(0),
                destination
                    .buffer
                    .at(destination.offset + (index * staging.size) as u64),
                chunk.len() as u64,
            );
            let sync = self.gpu.submit(&mut encoder);
            let _ = wait_for_timed_encoder(&self.gpu, &sync, &mut encoder, self.gpu_timing);
        }
        self.gpu.destroy_command_encoder(&mut encoder);
        if !self.reuse_upload_staging {
            self.gpu.destroy_buffer(cached.take().unwrap().buffer);
        }
    }

    pub(super) fn read_raw_f32(
        &self,
        buffer: blade_graphics::BufferPiece,
        out: &mut [f32],
        host_visible: bool,
    ) {
        if out.is_empty() {
            return;
        }
        let direct = |out: &mut [f32]| unsafe {
            std::ptr::copy_nonoverlapping(buffer.data() as *const f32, out.as_mut_ptr(), out.len());
        };
        let mut readback = self.readback.borrow_mut();
        if host_visible {
            let key = (buffer.data() as usize, std::mem::size_of_val(out));
            if let Some(&staged) = readback.staged.get(&key) {
                if staged {
                    self.read_staged_f32(buffer, out, &mut readback);
                } else {
                    direct(out);
                }
                return;
            }
            // A mapped device heap need not be CPU-cached. Measure the actual
            // allocation instead of assuming that host-visible means fast reads.
            let mut direct_time = std::time::Duration::MAX;
            let mut staged_time = std::time::Duration::MAX;
            for _ in 0..3 {
                let start = std::time::Instant::now();
                direct(out);
                direct_time = direct_time.min(start.elapsed());
                let bits: Vec<_> = out.iter().map(|x| x.to_bits()).collect();
                let start = std::time::Instant::now();
                self.read_staged_f32(buffer, out, &mut readback);
                staged_time = staged_time.min(start.elapsed());
                assert!(
                    out.iter().zip(bits).all(|(a, b)| a.to_bits() == b),
                    "readback changed buffer contents"
                );
            }
            let staged = staged_time.as_secs_f64() < direct_time.as_secs_f64() * 0.9;
            log::debug!(
                "readback {} bytes: mapped {direct_time:?}, staged {staged_time:?}, use staging={staged}",
                key.1
            );
            readback.staged.insert(key, staged);
            return;
        }
        self.read_staged_f32(buffer, out, &mut readback);
    }

    pub(super) fn read_staged_f32(
        &self,
        buffer: blade_graphics::BufferPiece,
        out: &mut [f32],
        readback: &mut Readback,
    ) {
        let cached = &mut readback.staging;
        let staging_bytes = std::mem::size_of_val(out).clamp(4, 16 * 1024 * 1024);
        if cached.as_ref().is_some_and(|s| s.size < staging_bytes) {
            self.gpu.destroy_buffer(cached.take().unwrap().buffer);
        }
        let staging = cached.get_or_insert_with(|| UploadStaging {
            buffer: self.gpu.create_buffer(blade_graphics::BufferDesc {
                name: "readback_staging",
                size: staging_bytes as u64,
                memory: blade_graphics::Memory::Download,
            }),
            size: staging_bytes,
        });
        let encoder = readback.encoder.get_or_insert_with(|| {
            self.gpu
                .create_command_encoder(blade_graphics::CommandEncoderDesc {
                    name: "readback",
                    buffer_count: 1,
                    manual_barriers: false,
                })
        });
        for (index, chunk) in out.chunks_mut(staging.size / 4).enumerate() {
            encoder.start();
            encoder.transfer("readback_copy").copy_buffer_to_buffer(
                piece_at(buffer, (index * staging.size) as u64),
                staging.buffer.at(0),
                std::mem::size_of_val(chunk) as u64,
            );
            let sync = self.gpu.submit(encoder);
            wait_for_timed_encoder(&self.gpu, &sync, encoder, self.gpu_timing)
                .expect("readback submission failed");
            unsafe {
                std::ptr::copy_nonoverlapping(
                    staging.buffer.data() as *const f32,
                    chunk.as_mut_ptr(),
                    chunk.len(),
                );
            }
        }
    }

    /// Read back the loss value.
    pub fn read_loss(&self) -> f32 {
        if let Some(buf_ref) = self.plan.loss_buffer {
            let buffer = self.buffers[buf_ref.0 as usize];
            let n = self.plan.buffers[buf_ref.0 as usize] / 4;
            if !self.logical_host_visible(buf_ref) {
                let mut values = vec![0.0; n];
                self.read_raw_f32(buffer, &mut values, false);
                return values.iter().sum();
            }
            unsafe {
                let ptr = buffer.data() as *const f32;
                let slice = std::slice::from_raw_parts(ptr, n);
                slice.iter().sum()
            }
        } else {
            0.0
        }
    }

    /// True when a logical buffer's content can be trusted after `step()`:
    /// it is the only tenant of its physical allocation.
    pub(super) fn buffer_unaliased(&self, buf: BufferRef) -> bool {
        let index = buf.0 as usize;
        (0..self.alias.map.len())
            .all(|other| other == index || !self.alias.overlap(&self.plan.buffers, index, other))
    }

    /// True when every tenant of this allocation is an immutable constant
    /// with the same bitwise payload. Such aliases remain readable because no
    /// step can replace their contents with a later value.
    pub(super) fn buffer_has_equivalent_constant_aliases(&self, buf: BufferRef) -> bool {
        let phys = self.alias.map[buf.0 as usize];
        let Some(reference) = self
            .plan
            .constant_buffers
            .iter()
            .find_map(|&(buffer, ref data)| (buffer == buf).then_some(data))
        else {
            return false;
        };
        let tenant_count = (0..self.alias.map.len())
            .filter(|&other| {
                self.alias
                    .overlap(&self.plan.buffers, buf.0 as usize, other)
            })
            .count();
        let mut constant_count = 0usize;
        for &(buffer, ref data) in &self.plan.constant_buffers {
            if self.alias.map[buffer.0 as usize] != phys {
                continue;
            }
            constant_count += 1;
            if data.len() != reference.len()
                || !data
                    .iter()
                    .zip(reference)
                    .all(|(left, right)| left.to_bits() == right.to_bits())
            {
                return false;
            }
        }
        tenant_count > 1 && tenant_count == constant_count
    }

    /// Read back the value of any graph node after a step.
    ///
    /// Works for materialized nodes in a debug session
    /// ([`SessionOptions::debug`] / `SessionConfig::debug()`); in a normal
    /// session it works for values whose buffer is not lifetime-aliased
    /// (params, inputs, outputs, and whatever the alias planner left
    /// unshared). Bit-identical immutable constants remain readable when they
    /// share an allocation. Use [`Session::read_node_by_name`] to address
    /// nodes named via `Graph::named` (or `nn` layers, which name their
    /// outputs).
    pub fn read_node(&self, node: crate::graph::NodeId) -> Result<Vec<f32>, ReadNodeError> {
        let buf = self
            .plan
            .node_buffers
            .binary_search_by_key(&node, |&(n, _)| n)
            .map(|i| self.plan.node_buffers[i].1)
            .map_err(|_| ReadNodeError::UnknownNode)?;
        if !self.written[buf.0 as usize] {
            return Err(ReadNodeError::FusedAway);
        }
        if !self.debug
            && !self.buffer_unaliased(buf)
            && !self.buffer_has_equivalent_constant_aliases(buf)
        {
            return Err(ReadNodeError::Aliased);
        }
        let n = self.plan.buffers[buf.0 as usize] / 4;
        let mut out = vec![0.0f32; n];
        self.read_buffer(buf, &mut out);
        Ok(out)
    }

    /// Read back a node's value by the name attached via `Graph::named`.
    /// When several nodes share a name (a reused module), the last one wins.
    pub fn read_node_by_name(&self, name: &str) -> Result<Vec<f32>, ReadNodeError> {
        let node = self
            .plan
            .node_names
            .iter()
            .rev()
            .find(|entry| entry.1 == name)
            .map(|&(id, _)| id)
            .ok_or_else(|| {
                ReadNodeError::UnknownName(
                    self.plan
                        .node_names
                        .iter()
                        .map(|entry| entry.1.clone())
                        .collect(),
                )
            })?;
        self.read_node(node)
    }

    /// Run a full step, then scan at most 65,536 f32 elements of each primary
    /// dispatch output in plan order. Extra outputs and runtime-appended
    /// optimizer state are not scanned. Aliased outputs are skipped outside
    /// debug mode; overwritten values and nonfinite tails can be missed.
    /// `first_bad()` identifies the first reported prefix, not necessarily
    /// the root cause. Active optimizer, accumulation and KV updates still run.
    pub fn step_debug(&mut self) -> DebugStepReport {
        self.step();
        self.wait();
        let mut report = DebugStepReport::default();
        for (i, d) in self.plan.dispatches.iter().enumerate() {
            if !self.debug && !self.buffer_unaliased(d.output_buffer) {
                report.skipped_aliased += 1;
                continue;
            }
            let buf_size = self.plan.buffers[d.output_buffer.0 as usize];
            let n = (buf_size / 4).min(65536);
            if n == 0 {
                continue;
            }
            let mut data = vec![0.0f32; n];
            self.read_buffer(d.output_buffer, &mut data);
            let max_abs = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let has_nan = data.iter().any(|v| v.is_nan());
            let has_inf = data.iter().any(|v| v.is_infinite());
            if has_nan || has_inf {
                report.anomalies.push(DispatchAnomaly {
                    dispatch: i,
                    label: if d.label.is_empty() {
                        format!("{:?}", d.shader)
                    } else {
                        d.label.clone()
                    },
                    origin: d.origin.clone(),
                    has_nan,
                    has_inf,
                    max_abs,
                });
            }
        }
        report
    }

    /// Diagnostic: print per-dispatch output buffer statistics.
    ///
    /// Scans all dispatches and reports any whose output contains NaN, Inf,
    /// or values exceeding `threshold`. Useful for tracing where numerical
    /// instability first appears in the forward/backward chain.
    pub fn trace_dispatches(&self, threshold: f32) {
        for (i, d) in self.plan.dispatches.iter().enumerate() {
            let buf_size = self.plan.buffers[d.output_buffer.0 as usize];
            let n = buf_size / 4;
            if n == 0 {
                continue;
            }
            let read_n = n.min(65536);
            let mut data = vec![0.0f32; read_n];
            self.read_buffer(d.output_buffer, &mut data);

            let max_abs = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let has_nan = data.iter().any(|v| v.is_nan());
            let has_inf = data.iter().any(|v| v.is_infinite());

            if has_nan || has_inf || max_abs > threshold || (max_abs == 0.0 && n > 100) {
                let label = if d.label.is_empty() {
                    format!("{:?}", d.shader)
                } else {
                    d.label.clone()
                };
                log::warn!(
                    "dispatch {i}: {label} max_abs={max_abs:.3e} nan={has_nan} inf={has_inf} n={n}"
                );
            }
        }
    }

    /// Read back the output tensor (first graph output).
    ///
    /// Returns the data as a `Vec<f32>`. For inference graphs this is the
    /// model's prediction; for training graphs it's the loss scalar.
    pub fn read_output(&self, len: usize) -> Vec<f32> {
        if let Some(buf_ref) = self.plan.loss_buffer {
            let mut out = vec![0.0_f32; len];
            self.read_buffer(buf_ref, &mut out);
            out
        } else {
            Vec::new()
        }
    }

    /// Read back a buffer's contents.
    ///
    /// Host-visible buffers (params, inputs, outputs, loss) read
    /// directly through the mapped pointer. Device-local buffers
    /// (intermediates, parameter gradients) take a staging round-trip.
    pub fn read_buffer(&self, buf_ref: BufferRef, out: &mut [f32]) {
        assert!(
            std::mem::size_of_val(out) <= self.plan.buffers[buf_ref.0 as usize],
            "read_buffer: F32 view exceeds buffer capacity"
        );
        self.read_raw_f32(
            self.buffers[buf_ref.0 as usize],
            out,
            self.logical_host_visible(buf_ref),
        );
    }

    /// Read complete buffers as F32 views in one staged transfer, preserving
    /// request order. Includes any declared padding, just like [`Self::read_buffer`].
    /// Call after waiting for the producing step. Useful for full-output checks
    /// without separately probing mapped reads for every allocation.
    pub fn read_buffers(&self, buffers: &[BufferRef]) -> Vec<Vec<f32>> {
        let requests: Vec<_> = buffers
            .iter()
            .map(|b| (self.buffers[b.0 as usize], self.plan.buffers[b.0 as usize]))
            .collect();
        self.read_f32_buffers(&requests, "buffer_readback")
    }

    /// Read back a graph output by index.
    ///
    /// Index 0 is the primary output (logits/loss). Higher indices are
    /// additional outputs (e.g. KV tensors from prefill).
    pub fn read_output_by_index(&self, index: usize, out: &mut [f32]) {
        let buf_ref = self.plan.output_buffers[index];
        self.read_buffer(buf_ref, out);
    }

    /// Wait for pending work and read a graph output.
    ///
    /// Queues a staged download before waiting on the CPU. Mapped reads and
    /// the initial readback probe still wait before accessing the buffer.
    pub fn wait_read_output(&mut self, index: usize, out: &mut [f32]) {
        let buf_ref = self.plan.output_buffers[index];
        let buffer = self.buffers[buf_ref.0 as usize];
        let staged = !self.logical_host_visible(buf_ref)
            || self
                .readback
                .borrow()
                .staged
                .get(&(buffer.data() as usize, std::mem::size_of_val(out)))
                == Some(&true);
        if !staged {
            self.wait();
        }
        self.read_buffer(buf_ref, out);
        self.wait();
    }

    /// Number of graph outputs.
    pub fn num_outputs(&self) -> usize {
        self.plan.output_buffers.len()
    }

    /// Look up a parameter's buffer reference by name.
    pub fn param_buffer(&self, name: &str) -> Option<BufferRef> {
        self.plan
            .param_buffers
            .iter()
            .find(|entry| entry.0 == name)
            .map(|entry| entry.1)
    }

    /// Read an F32 parameter buffer's contents by name.
    pub fn read_param(&self, name: &str, out: &mut [f32]) {
        let buf_ref = self
            .param_buffer(name)
            .unwrap_or_else(|| panic!("unknown param: {}", name));
        self.assert_f32_parameter(buf_ref);
        self.read_buffer(buf_ref, out);
    }

    pub(super) fn assert_f32_parameter(&self, buffer: BufferRef) {
        assert!(
            self.plan
                .param_types
                .get(&buffer)
                .is_none_or(|ty| ty.dtype == crate::graph::DType::F32),
            "parameter read requires F32 storage"
        );
    }

    pub(super) fn read_f32_buffers(
        &self,
        buffers: &[(blade_graphics::BufferPiece, usize)],
        label: &'static str,
    ) -> Vec<Vec<f32>> {
        if buffers.is_empty() {
            return Vec::new();
        }

        let mut total_bytes = 0_usize;
        let requests: Vec<_> = buffers
            .iter()
            .map(|&(buffer, byte_len)| {
                assert_eq!(byte_len % std::mem::size_of::<f32>(), 0);
                let offset = total_bytes;
                total_bytes += byte_len;
                (buffer, offset, byte_len)
            })
            .collect();
        let staging = self.gpu.create_buffer(blade_graphics::BufferDesc {
            name: label,
            size: (total_bytes as u64).max(4),
            memory: blade_graphics::Memory::Download,
        });
        let mut encoder = self
            .gpu
            .create_command_encoder(blade_graphics::CommandEncoderDesc {
                name: label,
                buffer_count: 1,
                manual_barriers: false,
            });
        encoder.start();
        {
            let mut transfer = encoder.transfer(label);
            for &(buffer, offset, byte_len) in &requests {
                if byte_len != 0 {
                    transfer.copy_buffer_to_buffer(
                        buffer,
                        staging.at(offset as u64),
                        byte_len as u64,
                    );
                }
            }
        }
        let sync = self.gpu.submit(&mut encoder);
        let _ = wait_for_timed_encoder(&self.gpu, &sync, &mut encoder, self.gpu_timing);

        let mut outputs = Vec::with_capacity(requests.len());
        for &(_, offset, byte_len) in &requests {
            let len = byte_len / std::mem::size_of::<f32>();
            let mut output = vec![0.0_f32; len];
            unsafe {
                std::ptr::copy_nonoverlapping(
                    staging.data().add(offset) as *const f32,
                    output.as_mut_ptr(),
                    len,
                );
            }
            outputs.push(output);
        }
        self.gpu.destroy_command_encoder(&mut encoder);
        self.gpu.destroy_buffer(staging);
        outputs
    }

    /// Read several full F32 parameter buffers with one GPU transfer.
    ///
    /// Shared parameter memory is fast for GPU access and CPU uploads, but
    /// can be very slow for CPU reads on a discrete GPU. This stages every
    /// requested parameter into cached download memory before copying it to
    /// the returned vectors. Results have the same order as `names`.
    pub fn read_params(&self, names: &[&str]) -> Vec<Vec<f32>> {
        let buffers: Vec<_> = names
            .iter()
            .map(|name| {
                let buf_ref = self
                    .param_buffer(name)
                    .unwrap_or_else(|| panic!("unknown param: {name}"));
                assert!(
                    matches!(
                        self.plan.weight_buffers.get(&buf_ref).map(|entry| entry.0),
                        None | Some(crate::compile::WeightFormat::F32)
                    ),
                    "parameter {name:?} is not an F32 buffer",
                );
                self.assert_f32_parameter(buf_ref);
                let byte_len = self.param_size(name).expect("parameter exists") * 4;
                (self.buffers[buf_ref.0 as usize], byte_len)
            })
            .collect();
        self.read_f32_buffers(&buffers, "parameter_readback")
    }

    /// Read a parameter's gradient buffer by name.
    ///
    /// Returns the gradient computed during the last backward pass.
    /// Panics if the parameter has no associated gradient (e.g. inference-only session).
    pub fn read_param_grad(&self, name: &str, out: &mut [f32]) {
        let param_buf = self
            .param_buffer(name)
            .unwrap_or_else(|| panic!("unknown param: {}", name));
        let grad_buf = self
            .plan
            .param_grad_pairs
            .iter()
            .find(|&&(p, _)| p == param_buf)
            .map(|&(_, g)| g)
            .unwrap_or_else(|| panic!("no gradient for param: {}", name));
        self.read_buffer(grad_buf, out);
    }

    /// Enumerate all parameter names in this session.
    ///
    /// Useful for diagnostic loops that want to inspect every parameter
    /// without hardcoding names. Order matches the underlying
    /// `ExecutionPlan::param_buffers` (compile-order, stable across runs
    /// of the same graph).
    pub fn param_names(&self) -> Vec<&str> {
        self.plan
            .param_buffers
            .iter()
            .map(|entry| entry.0.as_str())
            .collect()
    }

    /// Logical element count for a parameter, independent of storage/padding. Returns
    /// `None` if the name doesn't exist.
    pub fn param_size(&self, name: &str) -> Option<usize> {
        let buf_ref = self.param_buffer(name)?;
        Some(self.plan.param_types.get(&buf_ref).map_or(
            self.plan.buffers[buf_ref.0 as usize] / std::mem::size_of::<f32>(),
            |ty| ty.num_elements(),
        ))
    }

    /// Returns true iff the parameter has an associated gradient buffer
    /// (i.e. was reached by the backward pass and isn't inference-only).
    pub fn has_param_grad(&self, name: &str) -> bool {
        let Some(param_buf) = self.param_buffer(name) else {
            return false;
        };
        self.plan
            .param_grad_pairs
            .iter()
            .any(|&(p, _)| p == param_buf)
    }

    /// Read the Adam first-moment buffer (`m`) for a parameter into
    /// `out`. `out.len()` must equal the parameter's element count.
    /// Returns zeros without allocating before optimizer initialization.
    /// Panics if the parameter has no gradient. Used together with
    /// [`Session::write_adam_m`] / [`Session::write_adam_v`] /
    /// [`Session::set_adam_step_count`] to carry optimizer state
    /// across a session rebuild — e.g. when the parameter table
    /// reshaped due to a topology change like RadFoam densification.
    pub fn read_adam_m(&self, name: &str, out: &mut [f32]) {
        let idx = self
            .adam_state_index(name)
            .unwrap_or_else(|| panic!("no Adam state for param: {name}"));
        let n = self.param_size(name).expect("param exists; size known");
        assert_eq!(
            out.len(),
            n,
            "read_adam_m: out.len()={} but param '{name}' has {n} elements",
            out.len()
        );
        if let Some((buffer, _)) = self.adam_moments(idx) {
            self.read_raw_f32(buffer, out, !self.optimizer_device);
        } else {
            out.fill(0.0);
        }
    }

    /// Read the Adam second-moment buffer (`v`) for a parameter. See
    /// [`Session::read_adam_m`].
    pub fn read_adam_v(&self, name: &str, out: &mut [f32]) {
        let idx = self
            .adam_state_index(name)
            .unwrap_or_else(|| panic!("no Adam state for param: {name}"));
        let n = self.param_size(name).expect("param exists; size known");
        assert_eq!(
            out.len(),
            n,
            "read_adam_v: out.len()={} but param '{name}' has {n} elements",
            out.len()
        );
        if let Some((_, buffer)) = self.adam_moments(idx) {
            self.read_raw_f32(buffer, out, !self.optimizer_device);
        } else {
            out.fill(0.0);
        }
    }

    /// Read both Adam moment buffers for several parameters with one GPU
    /// transfer through cached download memory. Results have the same order
    /// as `names`.
    pub fn read_adam_states(&self, names: &[&str]) -> Vec<(Vec<f32>, Vec<f32>)> {
        if self.adam_state.is_none() {
            return names
                .iter()
                .map(|name| {
                    assert!(
                        self.adam_state_index(name).is_some(),
                        "no Adam state for param: {name}"
                    );
                    let n = self.param_size(name).expect("parameter exists");
                    (vec![0.0; n], vec![0.0; n])
                })
                .collect();
        }
        let buffers: Vec<_> = names
            .iter()
            .flat_map(|name| {
                let idx = self
                    .adam_state_index(name)
                    .unwrap_or_else(|| panic!("no Adam state for param: {name}"));
                let byte_len = self.param_size(name).expect("param exists; size known")
                    * std::mem::size_of::<f32>();
                let (m, v) = self.adam_moments(idx).expect("Adam state");
                [(m, byte_len), (v, byte_len)]
            })
            .collect();
        let mut values = self
            .read_f32_buffers(&buffers, "adam_state_readback")
            .into_iter();
        let mut states = Vec::with_capacity(names.len());
        for _ in names {
            states.push((values.next().unwrap(), values.next().unwrap()));
        }
        debug_assert!(values.next().is_none());
        states
    }

    /// Write the Adam first-moment buffer for a parameter. `data.len()`
    /// must equal the parameter's element count. See
    /// [`Session::read_adam_m`] for the carry-over use case.
    pub fn write_adam_m(&mut self, name: &str, data: &[f32]) {
        self.wait();
        let idx = self
            .adam_state_index(name)
            .unwrap_or_else(|| panic!("no Adam state for param: {name}"));
        let n = self.param_size(name).expect("param exists; size known");
        assert_eq!(
            data.len(),
            n,
            "write_adam_m: data.len()={} but param '{name}' has {n} elements",
            data.len()
        );
        self.ensure_adam_state();
        self.write_raw_buffer(
            self.adam_moments(idx).expect("Adam state").0,
            bytemuck::cast_slice(data),
            !self.optimizer_device,
        );
    }

    /// Write the Adam second-moment buffer for a parameter. See
    /// [`Session::write_adam_m`].
    pub fn write_adam_v(&mut self, name: &str, data: &[f32]) {
        self.wait();
        let idx = self
            .adam_state_index(name)
            .unwrap_or_else(|| panic!("no Adam state for param: {name}"));
        let n = self.param_size(name).expect("param exists; size known");
        assert_eq!(
            data.len(),
            n,
            "write_adam_v: data.len()={} but param '{name}' has {n} elements",
            data.len()
        );
        self.ensure_adam_state();
        self.write_raw_buffer(
            self.adam_moments(idx).expect("Adam state").1,
            bytemuck::cast_slice(data),
            !self.optimizer_device,
        );
    }

    /// Current Adam step counter (`t`). Adam's bias correction uses
    /// `1 - β₁ᵗ` / `1 - β₂ᵗ`, so this number is part of the optimizer
    /// state that must round-trip across a session rebuild for
    /// [`read_adam_m`](Session::read_adam_m) carry-over to be exact.
    pub fn adam_step_count(&self) -> u32 {
        self.adam_step
    }

    /// Set the Adam step counter. Pair with [`Session::write_adam_m`] /
    /// [`Session::write_adam_v`] when restoring optimizer state into a
    /// freshly built session.
    pub fn set_adam_step_count(&mut self, t: u32) {
        self.adam_step = t;
    }

    pub(super) fn adam_state_index(&self, name: &str) -> Option<usize> {
        let param_buf = self.param_buffer(name)?;
        self.plan
            .param_grad_pairs
            .iter()
            .position(|&(p, _)| p == param_buf)
    }

    /// Bulk read of per-parameter gradient L2 norms (Frobenius for
    /// matrices). Returns `(name, ‖grad‖₂)` pairs, in compile order,
    /// for every parameter that has a gradient buffer.
    ///
    /// The dominant diagnostic for "which parameter is exploding /
    /// vanishing under the current loss". Cost is one buffer copy +
    /// one sum-of-squares pass per parameter; meant for periodic
    /// inspection (every N steps), not per-step.
    pub fn read_all_param_grad_norms(&self) -> Vec<(String, f32)> {
        let mut out = Vec::with_capacity(self.plan.param_buffers.len());
        let mut scratch: Vec<f32> = Vec::new();
        for entry in &self.plan.param_buffers {
            let name = &entry.0;
            if !self.has_param_grad(name) {
                continue;
            }
            let n = self.param_size(name).expect("param exists; size known");
            scratch.resize(n, 0.0);
            self.read_param_grad(name, &mut scratch);
            let sum_sq: f32 = scratch.iter().map(|&v| v * v).sum();
            out.push((name.clone(), sum_sq.sqrt()));
        }
        out
    }

    /// Bulk read of F32 per-parameter weight L2 norms. Same shape as
    /// `read_all_param_grad_norms`. Useful for computing
    /// gradient-to-weight ratios as a stability indicator.
    pub fn read_all_param_norms(&self) -> Vec<(String, f32)> {
        let mut out = Vec::with_capacity(self.plan.param_buffers.len());
        let mut scratch: Vec<f32> = Vec::new();
        for entry in &self.plan.param_buffers {
            let name = &entry.0;
            let n = self.param_size(name).expect("param exists; size known");
            scratch.resize(n, 0.0);
            self.read_param(name, &mut scratch);
            let sum_sq: f32 = scratch.iter().map(|&v| v * v).sum();
            out.push((name.clone(), sum_sq.sqrt()));
        }
        out
    }

    /// Print a summary of the largest gradient norms (descending).
    /// Intended for occasional diagnostic dumps when training is
    /// behaving unexpectedly. Includes the gradient/weight ratio
    /// (if the weight is nonzero) so divergent updates are visible.
    pub fn dump_grad_summary(&self, top_n: usize) {
        let grads = self.read_all_param_grad_norms();
        let weights = self.read_all_param_norms();
        let weights_lookup: std::collections::HashMap<&str, f32> = weights
            .iter()
            .map(|entry| (entry.0.as_str(), entry.1))
            .collect();
        let mut sorted: Vec<&(String, f32)> = grads.iter().collect();
        sorted.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let n = sorted.len().min(top_n.max(1));
        eprintln!(
            "--- top {} param grads (out of {} with grads) ---",
            n,
            sorted.len()
        );
        for entry in sorted.iter().take(n) {
            let (name, g) = (&entry.0, entry.1);
            let w = weights_lookup.get(name.as_str()).copied().unwrap_or(0.0);
            let ratio = if w > 0.0 { g / w } else { f32::NAN };
            eprintln!(
                "  {:>40}  ‖g‖={:.3e}  ‖w‖={:.3e}  g/w={:.3e}",
                name, g, w, ratio
            );
        }
    }

    /// Upload data into a parameter buffer by name (for initializing KV caches etc.).
    pub fn upload_param(&mut self, name: &str, data: &[f32]) {
        self.wait();
        let buf_ref = self
            .param_buffer(name)
            .unwrap_or_else(|| panic!("unknown param: {}", name));
        self.upload_buffer(buf_ref, bytemuck::cast_slice(data));
    }

    /// Print GPU pass timings from the last completed step.
    ///
    /// Must be called after `step()` + `wait()`.
    pub fn dump_gpu_timings(&self) {
        let timings = self.gpu_timings();
        if timings.is_empty() {
            eprintln!("(no GPU timings available)");
            return;
        }
        let total: std::time::Duration = timings.iter().map(|&(_, d)| d).sum();
        eprintln!(
            "--- GPU pass timings ({} passes, {:.2}ms total) ---",
            timings.len(),
            total.as_secs_f64() * 1000.0
        );

        // Aggregate by shader type
        let mut by_type: std::collections::HashMap<&str, (u32, std::time::Duration)> =
            std::collections::HashMap::new();
        for &(ref name, dur) in &timings {
            let entry = by_type.entry(name.as_str()).or_default();
            entry.0 += 1;
            entry.1 += dur;
        }
        let mut sorted: Vec<_> = by_type.into_iter().collect();
        sorted.sort_by_key(|e| std::cmp::Reverse(e.1.1));
        for &(name, (count, dur)) in &sorted {
            let pct = dur.as_secs_f64() / total.as_secs_f64() * 100.0;
            eprintln!(
                "  {:>20}: {:>3}x {:>8.2}ms ({:>5.1}%)",
                name,
                count,
                dur.as_secs_f64() * 1000.0,
                pct
            );
        }
        eprintln!("---");
    }
}
