use super::Session;
use crate::{
    compile::{BufferRef, ExecutionPlan},
    graph::{DType, TensorType},
};
use safetensors::{
    SafeTensors,
    tensor::{Dtype, TensorView},
};
use std::{
    collections::{BTreeMap, HashMap, HashSet},
    io,
};

#[derive(serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct LogicalLayout {
    parameters: BTreeMap<String, TensorType>,
    adam_parameters: Vec<String>,
}

struct Parameter {
    name: String,
    buffer: BufferRef,
    ty: TensorType,
    byte_len: usize,
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message.into())
}

fn logical_bytes(ty: &TensorType) -> io::Result<usize> {
    let elements = ty
        .shape
        .iter()
        .try_fold(1usize, |n, &dim| n.checked_mul(dim));
    elements
        .and_then(|n| match ty.dtype {
            DType::F32 | DType::U32 => n.checked_mul(4),
            DType::F16 => n.checked_mul(2),
            DType::Q4_0 => n.div_ceil(32).checked_mul(20),
            DType::Q8_0 => n.div_ceil(32).checked_mul(36),
        })
        .ok_or_else(|| invalid("checkpoint logical shape overflows byte size"))
}

fn parameters(plan: &ExecutionPlan) -> io::Result<Vec<Parameter>> {
    let mut seen = HashSet::new();
    plan.param_buffers
        .iter()
        .map(|&(ref name, buffer)| {
            if !seen.insert(name) {
                return Err(invalid(format!("duplicate parameter name {name:?}")));
            }
            let ty = plan.param_types.get(&buffer).ok_or_else(|| {
                invalid(format!(
                    "logical type missing for {name:?}; recompile the execution plan"
                ))
            })?;
            let byte_len = logical_bytes(ty)?;
            if byte_len > plan.buffers[buffer.0 as usize] {
                return Err(invalid(format!(
                    "logical parameter {name:?} exceeds its allocation"
                )));
            }
            Ok(Parameter {
                name: name.clone(),
                buffer,
                ty: ty.clone(),
                byte_len,
            })
        })
        .collect()
}

fn tensor_layout(ty: &TensorType) -> io::Result<(Dtype, Vec<usize>)> {
    Ok(match ty.dtype {
        DType::F32 => (Dtype::F32, ty.shape.clone()),
        DType::F16 => (Dtype::F16, ty.shape.clone()),
        DType::U32 => (Dtype::U32, ty.shape.clone()),
        DType::Q4_0 | DType::Q8_0 => (Dtype::U8, vec![logical_bytes(ty)?]),
    })
}

#[derive(Clone, Copy)]
enum Target {
    Parameter(BufferRef),
    Moment { index: usize, second: bool },
}

struct Write<'a> {
    target: Target,
    data: &'a [u8],
}

struct Restore<'a> {
    writes: Vec<Write<'a>>,
    adam_step: Option<u32>,
    reset_moments: bool,
}

fn validate_tensor(
    tensor: &TensorView<'_>,
    name: &str,
    bytes: usize,
    layout: Option<(Dtype, Vec<usize>)>,
) -> io::Result<()> {
    if tensor.data().len() != bytes {
        return Err(invalid(format!(
            "checkpoint tensor {name:?} has {} bytes, expected {bytes}",
            tensor.data().len()
        )));
    }
    if let Some((dtype, shape)) = layout {
        if tensor.dtype() != dtype || tensor.shape() != shape {
            return Err(invalid(format!(
                "checkpoint tensor {name:?} has {:?}{:?}, expected {dtype:?}{shape:?}",
                tensor.dtype(),
                tensor.shape()
            )));
        }
    }
    Ok(())
}

/// Pure preflight: no session, GPU allocation or mutation is reachable here.
fn validate<'a>(plan: &ExecutionPlan, data: &'a [u8]) -> io::Result<Restore<'a>> {
    let (_, header) = SafeTensors::read_metadata(data).map_err(|e| invalid(e.to_string()))?;
    let tensors = SafeTensors::deserialize(data).map_err(|e| invalid(e.to_string()))?;
    let metadata = header.metadata().as_ref();
    let get = |key: &str| metadata.and_then(|m| m.get(key));
    let version = get("meganeura_checkpoint_format")
        .map(|v| {
            v.parse::<u32>()
                .map_err(|_| invalid(format!("invalid meganeura_checkpoint_format {v:?}")))
        })
        .transpose()?
        .unwrap_or(1);
    if !(1..=3).contains(&version) {
        return Err(invalid(format!("unsupported checkpoint format {version}")));
    }
    let adam_step = get("adam_step")
        .map(|v| {
            v.parse::<u32>()
                .map_err(|_| invalid(format!("invalid adam_step {v:?}")))
        })
        .transpose()?;
    let params = parameters(plan)?;
    let mut writes = Vec::new();
    if version == 3 {
        let layout: LogicalLayout = serde_json::from_str(
            get("meganeura_logical_layout")
                .ok_or_else(|| invalid("checkpoint missing meganeura_logical_layout"))?,
        )
        .map_err(|e| invalid(e.to_string()))?;
        if adam_step.is_none() {
            return Err(invalid("checkpoint missing adam_step"));
        }
        let expected_types: BTreeMap<_, _> = params
            .iter()
            .map(|p| (p.name.clone(), p.ty.clone()))
            .collect();
        if layout.parameters != expected_types {
            return Err(invalid(
                "checkpoint parameter names, logical shapes or storage types do not match the plan",
            ));
        }
        let mut expected_names = HashSet::new();
        for param in &params {
            expected_names.insert(param.name.clone());
            let tensor = tensors
                .tensor(&param.name)
                .map_err(|_| invalid(format!("checkpoint missing parameter {:?}", param.name)))?;
            validate_tensor(
                &tensor,
                &param.name,
                param.byte_len,
                Some(tensor_layout(&param.ty)?),
            )?;
            writes.push(Write {
                target: Target::Parameter(param.buffer),
                data: tensor.data(),
            });
        }
        for name in layout.adam_parameters {
            let param = params
                .iter()
                .find(|p| p.name == name)
                .ok_or_else(|| invalid(format!("unknown Adam parameter {name:?}")))?;
            if param.ty.dtype != DType::F32 {
                return Err(invalid(format!("Adam parameter {name:?} is not F32")));
            }
            for (suffix, second) in [("adam_m", false), ("adam_v", true)] {
                let key = format!("{suffix}.{name}");
                if !expected_names.insert(key.clone()) {
                    return Err(invalid(format!("duplicate checkpoint tensor {key:?}")));
                }
                let tensor = tensors
                    .tensor(&key)
                    .map_err(|_| invalid(format!("checkpoint missing tensor {key:?}")))?;
                validate_tensor(
                    &tensor,
                    &key,
                    param.byte_len,
                    Some((Dtype::F32, param.ty.shape.clone())),
                )?;
                if let Some(index) = plan
                    .param_grad_pairs
                    .iter()
                    .position(|&(p, _)| p == param.buffer)
                {
                    writes.push(Write {
                        target: Target::Moment { index, second },
                        data: tensor.data(),
                    });
                }
            }
        }
        if expected_names.len() != tensors.len()
            || tensors
                .names()
                .iter()
                .any(|n| !expected_names.contains(n.as_str()))
        {
            return Err(invalid("checkpoint contains unexpected tensors"));
        }
    } else {
        // Legacy files used flattened physical allocations. Preserve their
        // partial-load behavior, but validate every applicable write first.
        for param in &params {
            if let Ok(tensor) = tensors.tensor(&param.name) {
                let bytes = plan.buffers[param.buffer.0 as usize];
                let (dtype, _) = tensor_layout(&param.ty)?;
                let shape = vec![bytes / dtype.size()];
                validate_tensor(
                    &tensor,
                    &param.name,
                    bytes,
                    (version == 2).then_some((dtype, shape)),
                )?;
                writes.push(Write {
                    target: Target::Parameter(param.buffer),
                    data: tensor.data(),
                });
            } else {
                log::warn!("legacy checkpoint missing parameter: {}", param.name);
            }
        }
        for (index, &(param_buffer, _)) in plan.param_grad_pairs.iter().enumerate() {
            let param = params
                .iter()
                .find(|p| p.buffer == param_buffer)
                .ok_or_else(|| invalid("unnamed optimizer parameter"))?;
            for (suffix, second) in [("adam_m", false), ("adam_v", true)] {
                let name = format!("{suffix}.{}", param.name);
                if let Ok(tensor) = tensors.tensor(&name) {
                    if param.ty.dtype != DType::F32 {
                        return Err(invalid(format!(
                            "Adam parameter {:?} is not F32",
                            param.name
                        )));
                    }
                    let bytes = plan.buffers[param_buffer.0 as usize];
                    validate_tensor(
                        &tensor,
                        &name,
                        bytes,
                        (version == 2).then_some((Dtype::F32, vec![bytes / 4])),
                    )?;
                    writes.push(Write {
                        target: Target::Moment { index, second },
                        data: tensor.data(),
                    });
                }
            }
        }
    }
    Ok(Restore {
        writes,
        adam_step,
        reset_moments: version == 3,
    })
}

fn restored_weight_staging(
    plan: &ExecutionPlan,
    writes: &[Write<'_>],
) -> io::Result<HashMap<BufferRef, Vec<f32>>> {
    use crate::compile::WeightFormat;
    let mut staging = HashMap::new();
    for &(buffer, _, _) in &plan.derived_params {
        let Some(&(format, rows, cols)) = plan.weight_buffers.get(&buffer) else {
            continue;
        };
        let Some(write) = writes
            .iter()
            .find(|w| matches!(w.target, Target::Parameter(b) if b == buffer))
        else {
            continue;
        };
        let values = match format {
            WeightFormat::F16 => write
                .data
                .as_chunks::<2>()
                .0
                .iter()
                .map(|&b| half::f16::from_bits(u16::from_le_bytes(b)).to_f32())
                .collect(),
            WeightFormat::Q4 | WeightFormat::Q8 => {
                let elements = rows
                    .checked_mul(cols)
                    .ok_or_else(|| invalid("derived weight size overflow"))?;
                let block_bytes = if format == WeightFormat::Q4 { 20 } else { 36 };
                let bytes = (elements / 32)
                    .checked_mul(block_bytes)
                    .ok_or_else(|| invalid("derived weight size overflow"))?;
                if !rows.is_multiple_of(32) || write.data.len() < bytes {
                    return Err(invalid("invalid packed derived weight layout"));
                }
                if format == WeightFormat::Q4 {
                    super::dequantize_q4_0(write.data, rows, cols)
                } else {
                    super::dequantize_q8_0(write.data, rows, cols)
                }
            }
            WeightFormat::F32 => continue,
        };
        staging.insert(buffer, values);
    }
    Ok(staging)
}

impl Session {
    /// Save a training checkpoint (parameters + Adam state) to a safetensors file.
    ///
    /// Format 3 stores logical shapes/storage types, parameter payloads without
    /// device padding, allocated Adam/LaProp moments and the step counter.
    /// Packed Q4/Q8 tensors remain bytes with an explicit logical-type manifest.
    /// This is not a complete session snapshot: optimizer configuration,
    /// accumulation windows, clip cadence and diagnostic totals are not saved.
    #[allow(clippy::pattern_type_mismatch)]
    pub fn save_checkpoint(&mut self, path: &std::path::Path) -> std::io::Result<()> {
        struct CheckpointTensor {
            name: String,
            buffer: blade_graphics::Buffer,
            offset: usize,
            byte_len: usize,
            dtype: Dtype,
            shape: Vec<usize>,
            host_visible: bool,
        }

        let wall_start = std::time::Instant::now();
        let wait_start = std::time::Instant::now();
        self.wait();
        let wait_duration = wait_start.elapsed();
        let collection_start = std::time::Instant::now();
        let mut tensors = Vec::new();
        let mut total_bytes = 0_usize;
        let params = parameters(&self.plan)?;
        let mut layout = LogicalLayout {
            parameters: params
                .iter()
                .map(|p| (p.name.clone(), p.ty.clone()))
                .collect(),
            adam_parameters: Vec::new(),
        };

        // Collect parameter data
        for param in &params {
            let byte_len = param.byte_len;
            let (dtype, shape) = tensor_layout(&param.ty)?;
            let offset = total_bytes
                .checked_add(3)
                .ok_or_else(|| invalid("checkpoint size overflow"))?
                / 4
                * 4;
            tensors.push(CheckpointTensor {
                name: param.name.clone(),
                buffer: self.buffers[param.buffer.0 as usize],
                offset,
                byte_len,
                dtype,
                shape,
                host_visible: self.logical_host_visible(param.buffer),
            });
            total_bytes = offset
                .checked_add(byte_len)
                .ok_or_else(|| invalid("checkpoint size overflow"))?;
        }

        // Collect Adam moment buffers (parallel to param_grad_pairs)
        for (idx, &(param_buf, _)) in self.plan.param_grad_pairs.iter().enumerate() {
            if idx >= self.adam_state.len() {
                break;
            }
            let param = params
                .iter()
                .find(|p| p.buffer == param_buf)
                .ok_or_else(|| invalid("unnamed optimizer parameter"))?;
            if param.ty.dtype != DType::F32 {
                return Err(invalid(format!(
                    "Adam parameter {:?} is not F32",
                    param.name
                )));
            }
            layout.adam_parameters.push(param.name.clone());
            let byte_len = param.byte_len;
            for (suffix, buf) in [
                ("adam_m", &self.adam_state[idx].0),
                ("adam_v", &self.adam_state[idx].1),
            ] {
                let offset = total_bytes
                    .checked_add(3)
                    .ok_or_else(|| invalid("checkpoint size overflow"))?
                    / 4
                    * 4;
                tensors.push(CheckpointTensor {
                    name: format!("{suffix}.{}", param.name),
                    buffer: *buf,
                    offset,
                    byte_len,
                    dtype: Dtype::F32,
                    shape: param.ty.shape.clone(),
                    host_visible: !self.optimizer_device,
                });
                total_bytes = offset
                    .checked_add(byte_len)
                    .ok_or_else(|| invalid("checkpoint size overflow"))?;
            }
        }
        let mut names = HashSet::new();
        for tensor in &tensors {
            if !names.insert(&tensor.name) {
                return Err(invalid(format!(
                    "checkpoint tensor name collision: {:?}",
                    tensor.name
                )));
            }
            if tensor.byte_len % 4 != 0 && !tensor.host_visible {
                return Err(invalid("unaligned device-only checkpoint tensor"));
            }
        }
        let mut metadata = HashMap::new();
        metadata.insert("meganeura_checkpoint_format".to_string(), "3".to_string());
        metadata.insert("adam_step".to_string(), self.adam_step.to_string());
        metadata.insert(
            "meganeura_logical_layout".to_string(),
            serde_json::to_string(&layout).map_err(io::Error::other)?,
        );
        let collection_duration = collection_start.elapsed();

        // Host-visible parameter allocations are optimized for GPU access and
        // CPU writes; reading them directly is extremely slow on a discrete
        // GPU. Snapshot every tensor through one GPU transfer instead.
        let readback_start = std::time::Instant::now();
        let staging = self.gpu.create_buffer(blade_graphics::BufferDesc {
            name: "checkpoint_readback",
            size: (total_bytes as u64).max(4),
            memory: blade_graphics::Memory::Download,
        });
        let mut encoder = self
            .gpu
            .create_command_encoder(blade_graphics::CommandEncoderDesc {
                name: "checkpoint_readback",
                buffer_count: 1,
                manual_barriers: false,
            });
        encoder.start();
        {
            let mut transfer = encoder.transfer("checkpoint_readback");
            for tensor in &tensors {
                let aligned_len = tensor.byte_len / 4 * 4;
                if aligned_len != 0 {
                    transfer.copy_buffer_to_buffer(
                        tensor.buffer.at(0),
                        staging.at(tensor.offset as u64),
                        aligned_len as u64,
                    );
                }
            }
        }
        let sync = self.gpu.submit(&mut encoder);
        let _ = self.gpu.wait_for(&sync, !0);
        for tensor in &tensors {
            let aligned_len = tensor.byte_len / 4 * 4;
            let tail_len = tensor.byte_len - aligned_len;
            if tail_len != 0 {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        tensor.buffer.data().add(aligned_len),
                        staging.data().add(tensor.offset + aligned_len),
                        tail_len,
                    );
                }
            }
        }
        let readback_duration = readback_start.elapsed();

        // Build tensor views
        let view_start = std::time::Instant::now();
        let staging_data = unsafe { std::slice::from_raw_parts(staging.data(), total_bytes) };
        let views: Vec<(String, TensorView<'_>)> = tensors
            .iter()
            .map(|tensor| {
                let data = &staging_data[tensor.offset..tensor.offset + tensor.byte_len];
                (
                    tensor.name.clone(),
                    TensorView::new(tensor.dtype, tensor.shape.clone(), data).expect("tensor view"),
                )
            })
            .collect();
        let view_duration = view_start.elapsed();

        let persist_start = std::time::Instant::now();
        let result = safetensors::tensor::serialize_to_file(views, &Some(metadata), path)
            .map_err(|e| std::io::Error::other(e.to_string()));
        let persist_duration = persist_start.elapsed();
        let bytes = if result.is_ok() {
            std::fs::metadata(path).map_or(0, |file| file.len())
        } else {
            0
        };
        self.gpu.destroy_command_encoder(&mut encoder);
        self.gpu.destroy_buffer(staging);
        log::info!(
            "checkpoint timing: wall={:.3}s wait={:.3}s collect={:.3}s \
             readback={:.3}s views={:.3}s persist={:.3}s bytes={bytes}",
            wall_start.elapsed().as_secs_f64(),
            wait_duration.as_secs_f64(),
            collection_duration.as_secs_f64(),
            readback_duration.as_secs_f64(),
            view_duration.as_secs_f64(),
            persist_duration.as_secs_f64(),
        );
        result
    }

    /// Restore a checkpoint after validating every applicable tensor and metadata field.
    ///
    /// Format 3 requires the same named logical parameter shapes/storage types,
    /// but permits different device padding. Inference sessions validate and
    /// ignore saved moments; training sessions restore them lazily. Parameters
    /// without saved moments get zero moments, not earlier target state.
    /// Formats 1/2 retain legacy physical-size and partial-load compatibility.
    ///
    /// Validation/I/O errors leave tensors, moments and counters unchanged.
    /// This is not rollback after device loss/allocation failure. Optimizer
    /// configuration and accumulation/clip/diagnostic state are not restored;
    /// use a fresh session or explicitly manage those training-loop boundaries.
    pub fn load_checkpoint(&mut self, path: &std::path::Path) -> io::Result<()> {
        let data = std::fs::read(path)?;
        let restore = validate(&self.plan, &data)?;
        for write in &restore.writes {
            let (capacity, host_visible) = match write.target {
                Target::Parameter(param) => (
                    self.plan.buffers[param.0 as usize],
                    self.logical_host_visible(param),
                ),
                Target::Moment { index, .. } => (
                    self.plan.buffers[self.plan.param_grad_pairs[index].0.0 as usize],
                    !self.optimizer_device,
                ),
            };
            if capacity % 4 != 0 && !host_visible {
                return Err(invalid("unaligned device-only checkpoint destination"));
            }
        }
        let staging = restored_weight_staging(&self.plan, &restore.writes)?;
        let moment_indices: HashSet<_> = restore
            .writes
            .iter()
            .filter_map(|w| match w.target {
                Target::Moment { index, .. } => Some(index),
                _ => None,
            })
            .collect();
        self.wait();
        if !moment_indices.is_empty() {
            self.ensure_adam_state();
        }
        if restore.reset_moments {
            for (index, &(m, v)) in self.adam_state.iter().enumerate() {
                if moment_indices.contains(&index) {
                    continue;
                }
                let param = self.plan.param_grad_pairs[index].0;
                let zeros = vec![0; self.plan.buffers[param.0 as usize].max(4)];
                self.write_raw_buffer(&m, &zeros, !self.optimizer_device);
                self.write_raw_buffer(&v, &zeros, !self.optimizer_device);
            }
        }
        for write in restore.writes {
            let (buffer, capacity, host_visible) = match write.target {
                Target::Parameter(param) => (
                    self.buffers[param.0 as usize],
                    self.plan.buffers[param.0 as usize],
                    self.logical_host_visible(param),
                ),
                Target::Moment { index, second } => {
                    let param = self.plan.param_grad_pairs[index].0;
                    let (m, v) = self.adam_state[index];
                    (
                        if second { v } else { m },
                        self.plan.buffers[param.0 as usize],
                        !self.optimizer_device,
                    )
                }
            };
            if write.data.len() == capacity {
                self.write_raw_buffer(&buffer, write.data, host_visible);
            } else {
                let mut padded = vec![0; capacity];
                padded[..write.data.len()].copy_from_slice(write.data);
                self.write_raw_buffer(&buffer, &padded, host_visible);
            }
        }
        if let Some(step) = restore.adam_step {
            self.adam_step = step;
        }
        self.weight_staging.extend(staging);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    type Tensors = BTreeMap<String, (Dtype, Vec<usize>, Vec<u8>)>;

    fn plan() -> ExecutionPlan {
        let mut graph = crate::Graph::new();
        let a = graph.parameter("a", &[2, 3]);
        let b = graph.parameter("b", &[2, 3]);
        let sum = graph.add(a, b);
        let loss = graph.mean_all(sum);
        graph.set_outputs(vec![loss]);
        crate::compile::compile(&crate::autodiff::differentiate(&graph))
    }

    fn fixture(plan: &ExecutionPlan) -> (Tensors, HashMap<String, String>) {
        let params = parameters(plan).unwrap();
        let layout = LogicalLayout {
            parameters: params
                .iter()
                .map(|p| (p.name.clone(), p.ty.clone()))
                .collect(),
            adam_parameters: params.iter().map(|p| p.name.clone()).collect(),
        };
        let mut tensors = BTreeMap::new();
        for p in params {
            for name in [
                p.name.clone(),
                format!("adam_m.{}", p.name),
                format!("adam_v.{}", p.name),
            ] {
                tensors.insert(name, (Dtype::F32, p.ty.shape.clone(), vec![0; p.byte_len]));
            }
        }
        (
            tensors,
            HashMap::from([
                ("meganeura_checkpoint_format".into(), "3".into()),
                ("adam_step".into(), "7".into()),
                (
                    "meganeura_logical_layout".into(),
                    serde_json::to_string(&layout).unwrap(),
                ),
            ]),
        )
    }

    fn encode(tensors: &Tensors, metadata: HashMap<String, String>) -> Vec<u8> {
        let views: Vec<_> = tensors
            .iter()
            .map(|(name, entry)| {
                (
                    name.clone(),
                    TensorView::new(entry.0, entry.1.clone(), &entry.2).unwrap(),
                )
            })
            .collect();
        safetensors::tensor::serialize(views, &Some(metadata)).unwrap()
    }

    #[test]
    fn logical_checkpoint_preflight_rejects_incomplete_and_mismatched_records() {
        let plan = plan();
        let (tensors, metadata) = fixture(&plan);
        let data = encode(&tensors, metadata);
        let restore = validate(&plan, &data).unwrap();
        assert_eq!(restore.writes.len(), 6);
        assert_eq!(restore.adam_step, Some(7));
        assert!(validate(&plan, &data[..data.len() - 1]).is_err());
        assert!(validate(&plan, &data[..7]).is_err());
        for case in 0..15 {
            let (mut tensors, mut metadata) = fixture(&plan);
            match case {
                0 => {
                    metadata.insert("adam_step".into(), "-1".into());
                }
                1 => {
                    metadata.insert("meganeura_checkpoint_format".into(), "4".into());
                }
                2 => {
                    tensors.remove("b");
                }
                3 => {
                    tensors.get_mut("b").unwrap().1 = vec![3, 2];
                }
                4 => {
                    tensors.get_mut("b").unwrap().0 = Dtype::I32;
                }
                5 => {
                    tensors.insert("adam_v.b".into(), (Dtype::F32, vec![5], vec![0; 20]));
                }
                6 => {
                    tensors.remove("adam_v.b");
                }
                7 => {
                    tensors.insert("extra".into(), (Dtype::F32, vec![1], vec![0; 4]));
                }
                8..=10 => {
                    let mut layout: LogicalLayout =
                        serde_json::from_str(&metadata["meganeura_logical_layout"]).unwrap();
                    match case {
                        8 => layout.parameters.get_mut("b").unwrap().shape = vec![3, 2],
                        9 => layout.adam_parameters.push("b".into()),
                        _ => layout.adam_parameters.push("unknown".into()),
                    }
                    metadata.insert(
                        "meganeura_logical_layout".into(),
                        serde_json::to_string(&layout).unwrap(),
                    );
                }
                11 => {
                    metadata.remove("meganeura_logical_layout");
                }
                12 => {
                    metadata.remove("adam_step");
                }
                13 => {
                    metadata.insert("meganeura_checkpoint_format".into(), "bad".into());
                }
                _ => {
                    metadata.insert("meganeura_checkpoint_format".into(), "0".into());
                }
            }
            assert!(
                validate(&plan, &encode(&tensors, metadata)).is_err(),
                "corruption case {case} accepted"
            );
        }
    }

    #[test]
    fn logical_checkpoint_is_independent_of_padding_and_inference_moment_allocation() {
        let mut plan = plan();
        let (mut tensors, metadata) = fixture(&plan);
        let data = encode(&tensors, metadata.clone());
        for &(_, buffer) in &plan.param_buffers {
            plan.buffers[buffer.0 as usize] += 128;
        }
        assert_eq!(validate(&plan, &data).unwrap().writes.len(), 6);
        plan.param_grad_pairs.clear();
        assert_eq!(validate(&plan, &data).unwrap().writes.len(), 2);
        tensors.remove("adam_v.b");
        assert!(validate(&plan, &encode(&tensors, metadata)).is_err());
    }

    #[test]
    fn checkpoint_logical_byte_counts_are_checked() {
        assert!(logical_bytes(&TensorType::f32(vec![usize::MAX])).is_err());
        assert!(logical_bytes(&TensorType::f32(vec![usize::MAX, 2])).is_err());
        assert_eq!(logical_bytes(&TensorType::f16(vec![3])).unwrap(), 6);
        assert_eq!(
            logical_bytes(&TensorType::new(vec![32, 33], DType::Q4_0)).unwrap(),
            660
        );
        assert_eq!(
            logical_bytes(&TensorType::new(vec![32, 33], DType::Q8_0)).unwrap(),
            1188
        );
    }

    #[test]
    fn legacy_checkpoint_partial_loads_remain_preflighted() {
        for version in [1, 2] {
            let plan = plan();
            let (mut tensors, mut metadata) = fixture(&plan);
            metadata.insert("meganeura_checkpoint_format".into(), version.to_string());
            metadata.remove("meganeura_logical_layout");
            tensors.remove("b");
            for entry in tensors.values_mut() {
                entry.1 = vec![entry.2.len() / 4];
            }
            let data = encode(&tensors, metadata.clone());
            assert_eq!(validate(&plan, &data).unwrap().writes.len(), 5);
            tensors.insert("adam_v.b".into(), (Dtype::F32, vec![9], vec![0; 36]));
            assert!(validate(&plan, &encode(&tensors, metadata)).is_err());
        }
    }
}
