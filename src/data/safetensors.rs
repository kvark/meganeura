//! Safetensors model weight loading.
//!
//! Loads named tensors from `.safetensors` files (local or downloaded from
//! the HuggingFace Hub) into f32 data that can be uploaded to a [`Session`].
//!
//! # Example
//!
//! ```no_run
//! use meganeura::data::safetensors::SafeTensorsModel;
//!
//! let model = SafeTensorsModel::download("dacorvo/mnist-mlp").unwrap();
//! for (name, info) in model.tensor_info() {
//!     println!("{}: shape={:?}", name, info.shape);
//! }
//! let data = model.tensor_f32("input_layer.weight").unwrap();
//! ```

use safetensors::SafeTensors;
use std::collections::HashMap;
use std::path::PathBuf;

/// Information about a tensor in a safetensors file.
#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub shape: Vec<usize>,
    pub dtype: safetensors::Dtype,
}

/// Parsed safetensors weights, loaded from a local file or downloaded
/// from the HuggingFace Hub.
pub struct SafeTensorsModel {
    /// Raw bytes of the safetensors file (kept alive for zero-copy access).
    data: Vec<u8>,
    /// Cached tensor metadata.
    info: HashMap<String, TensorInfo>,
}

impl SafeTensorsModel {
    /// Download a model from HuggingFace Hub by repository ID.
    ///
    /// Downloads `model.safetensors` from the given repo (e.g. `"dacorvo/mnist-mlp"`).
    pub fn download(repo_id: &str) -> Result<Self, Box<dyn std::error::Error>> {
        Self::download_file(repo_id, "model.safetensors")
    }

    /// Download a specific safetensors file from a HuggingFace Hub repo.
    pub fn download_file(
        repo_id: &str,
        filename: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        log::info!("downloading {}/{} from HuggingFace Hub...", repo_id, filename);
        let api = hf_hub::api::sync::Api::new()?;
        let repo = api.model(repo_id.to_string());
        let path = repo.get(filename)?;
        log::info!("cached at: {}", path.display());
        Self::load(path)
    }

    /// Load a model from a local safetensors file.
    pub fn load(path: PathBuf) -> Result<Self, Box<dyn std::error::Error>> {
        let data = std::fs::read(&path)?;
        let tensors = SafeTensors::deserialize(&data)?;

        let mut info = HashMap::new();
        for (name, view) in tensors.iter() {
            info.insert(
                name.to_string(),
                TensorInfo {
                    shape: view.shape().to_vec(),
                    dtype: view.dtype(),
                },
            );
        }

        log::info!("loaded {} tensors from {}", info.len(), path.display());
        Ok(Self { data, info })
    }

    /// List all tensor names and their metadata.
    pub fn tensor_info(&self) -> &HashMap<String, TensorInfo> {
        &self.info
    }

    /// Read a tensor as f32 data.
    ///
    /// For F32 tensors this is a direct reinterpretation. Other dtypes
    /// are not currently supported and will return an error.
    pub fn tensor_f32(&self, name: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let tensors = SafeTensors::deserialize(&self.data)?;
        let view = tensors
            .tensor(name)
            .map_err(|e| format!("tensor '{}': {}", name, e))?;

        if view.dtype() != safetensors::Dtype::F32 {
            return Err(format!(
                "tensor '{}' has dtype {:?}, expected F32",
                name,
                view.dtype()
            )
            .into());
        }

        let bytes = view.data();
        // safetensors stores data in little-endian, which matches x86/ARM
        let floats: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        Ok(floats)
    }

    /// Read a tensor as f32 data, auto-converting from BF16 if necessary.
    ///
    /// Supports F32 (direct) and BF16 (converted). Other dtypes return an error.
    pub fn tensor_f32_auto(&self, name: &str) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let tensors = SafeTensors::deserialize(&self.data)?;
        let view = tensors
            .tensor(name)
            .map_err(|e| format!("tensor '{}': {}", name, e))?;

        match view.dtype() {
            safetensors::Dtype::F32 => {
                let bytes = view.data();
                let floats: Vec<f32> = bytes
                    .chunks_exact(4)
                    .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                    .collect();
                Ok(floats)
            }
            safetensors::Dtype::BF16 => {
                // BF16 is a truncated F32: same sign + exponent bits,
                // fewer mantissa bits. Widening is a simple 16-bit left
                // shift (zero-fills the lower mantissa bits).
                let bytes = view.data();
                let floats: Vec<f32> = bytes
                    .chunks_exact(2)
                    .map(|chunk| {
                        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                        f32::from_bits((bits as u32) << 16)
                    })
                    .collect();
                Ok(floats)
            }
            other => Err(format!(
                "tensor '{}' has dtype {:?}, expected F32 or BF16",
                name, other
            )
            .into()),
        }
    }

    /// Read a tensor as f32 and transpose, auto-converting from BF16 if necessary.
    pub fn tensor_f32_auto_transposed(
        &self,
        name: &str,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let info = self
            .info
            .get(name)
            .ok_or_else(|| format!("tensor '{}' not found", name))?;

        if info.shape.len() != 2 {
            return Err(format!(
                "tensor '{}' has {} dims, expected 2 for transpose",
                name,
                info.shape.len()
            )
            .into());
        }

        let data = self.tensor_f32_auto(name)?;
        let rows = info.shape[0];
        let cols = info.shape[1];
        let mut transposed = vec![0.0_f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                transposed[c * rows + r] = data[r * cols + c];
            }
        }
        Ok(transposed)
    }

    /// Read a tensor as f32 and transpose it from (rows, cols) to (cols, rows).
    ///
    /// PyTorch Linear layers store weights as (out_features, in_features),
    /// but meganeura's matmul expects (in_features, out_features).
    pub fn tensor_f32_transposed(
        &self,
        name: &str,
    ) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
        let info = self
            .info
            .get(name)
            .ok_or_else(|| format!("tensor '{}' not found", name))?;

        if info.shape.len() != 2 {
            return Err(format!(
                "tensor '{}' has {} dims, expected 2 for transpose",
                name,
                info.shape.len()
            )
            .into());
        }

        let data = self.tensor_f32(name)?;
        let rows = info.shape[0];
        let cols = info.shape[1];
        let mut transposed = vec![0.0_f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                transposed[c * rows + r] = data[r * cols + c];
            }
        }
        Ok(transposed)
    }
}
