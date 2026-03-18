//! MNIST IDX file format parser.
//!
//! Supports both raw and gzip-compressed files.

use super::DataLoader;
use std::io::{self, Read};
use std::path::Path;

/// MNIST dataset loaded into memory.
///
/// Images are flattened to `[N, 784]` as `f32` in `[0, 1]`.
/// Labels are one-hot encoded to `[N, 10]`.
pub struct MnistDataset {
    /// Flattened image data, shape `[n, 784]`, values in `[0, 1]`.
    pub images: Vec<f32>,
    /// One-hot label data, shape `[n, 10]`.
    pub labels: Vec<f32>,
    /// Number of samples.
    pub n: usize,
}

impl MnistDataset {
    /// Load MNIST from the standard IDX files.
    ///
    /// * `images_path` – path to `train-images-idx3-ubyte` (or `t10k-…`).
    /// * `labels_path` – path to `train-labels-idx1-ubyte` (or `t10k-…`).
    pub fn load(images_path: &Path, labels_path: &Path) -> io::Result<Self> {
        let images_raw = std::fs::read(images_path)?;
        let labels_raw = std::fs::read(labels_path)?;
        Self::from_buffers(&images_raw, &labels_raw)
    }

    /// Load gzip-compressed IDX files (the common download format).
    pub fn load_gz(images_path: &Path, labels_path: &Path) -> io::Result<Self> {
        let images_raw = read_gz(images_path)?;
        let labels_raw = read_gz(labels_path)?;
        Self::from_buffers(&images_raw, &labels_raw)
    }

    fn from_buffers(images_raw: &[u8], labels_raw: &[u8]) -> io::Result<Self> {
        let images = parse_idx_images(images_raw)?;
        let labels = parse_idx_labels(labels_raw)?;

        if images.n != labels.n {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "image count ({}) != label count ({})",
                    images.n, labels.n
                ),
            ));
        }

        Ok(Self {
            images: images.data,
            labels: labels.data,
            n: images.n,
        })
    }

    /// Create a [`DataLoader`] from this dataset.
    pub fn loader(self, batch_size: usize) -> DataLoader {
        DataLoader::new(self.images, self.labels, 784, 10, batch_size)
    }
}

fn read_gz(path: &Path) -> io::Result<Vec<u8>> {
    let file = std::fs::File::open(path)?;
    let mut decoder = flate2::read::GzDecoder::new(file);
    let mut buf = Vec::new();
    decoder.read_to_end(&mut buf)?;
    Ok(buf)
}

struct ParsedImages {
    data: Vec<f32>,
    n: usize,
}

struct ParsedLabels {
    data: Vec<f32>,
    n: usize,
}

fn read_u32_be(buf: &[u8], offset: usize) -> u32 {
    u32::from_be_bytes([
        buf[offset],
        buf[offset + 1],
        buf[offset + 2],
        buf[offset + 3],
    ])
}

fn parse_idx_images(buf: &[u8]) -> io::Result<ParsedImages> {
    if buf.len() < 16 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "IDX image file too short",
        ));
    }
    let magic = read_u32_be(buf, 0);
    if magic != 0x00000803 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "bad IDX image magic: 0x{:08x}, expected 0x00000803",
                magic
            ),
        ));
    }
    let n = read_u32_be(buf, 4) as usize;
    let rows = read_u32_be(buf, 8) as usize;
    let cols = read_u32_be(buf, 12) as usize;
    let pixels = rows * cols; // 784 for MNIST
    let expected = 16 + n * pixels;
    if buf.len() < expected {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "IDX image file truncated: need {} bytes, got {}",
                expected,
                buf.len()
            ),
        ));
    }
    let data: Vec<f32> = buf[16..16 + n * pixels]
        .iter()
        .map(|&b| b as f32 / 255.0)
        .collect();
    Ok(ParsedImages { data, n })
}

fn parse_idx_labels(buf: &[u8]) -> io::Result<ParsedLabels> {
    if buf.len() < 8 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "IDX label file too short",
        ));
    }
    let magic = read_u32_be(buf, 0);
    if magic != 0x00000801 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "bad IDX label magic: 0x{:08x}, expected 0x00000801",
                magic
            ),
        ));
    }
    let n = read_u32_be(buf, 4) as usize;
    let expected = 8 + n;
    if buf.len() < expected {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "IDX label file truncated: need {} bytes, got {}",
                expected,
                buf.len()
            ),
        ));
    }
    // One-hot encode to 10 classes
    let mut data = vec![0.0_f32; n * 10];
    for i in 0..n {
        let label = buf[8 + i] as usize;
        if label >= 10 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("label {} out of range at index {}", label, i),
            ));
        }
        data[i * 10 + label] = 1.0;
    }
    Ok(ParsedLabels { data, n })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_idx_images() {
        // Construct a minimal IDX image file: 2 images of 2x2 pixels
        let mut buf = Vec::new();
        buf.extend_from_slice(&0x00000803_u32.to_be_bytes()); // magic
        buf.extend_from_slice(&2_u32.to_be_bytes()); // n
        buf.extend_from_slice(&2_u32.to_be_bytes()); // rows
        buf.extend_from_slice(&2_u32.to_be_bytes()); // cols
        buf.extend_from_slice(&[0, 128, 255, 64, 32, 96, 192, 160]); // pixel data

        let parsed = parse_idx_images(&buf).unwrap();
        assert_eq!(parsed.n, 2);
        assert_eq!(parsed.data.len(), 8); // 2 * 2 * 2
        assert_eq!(parsed.data[0], 0.0);
        assert!((parsed.data[1] - 128.0 / 255.0).abs() < 1e-6);
        assert!((parsed.data[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_parse_idx_labels() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&0x00000801_u32.to_be_bytes()); // magic
        buf.extend_from_slice(&3_u32.to_be_bytes()); // n
        buf.extend_from_slice(&[7, 2, 0]); // labels

        let parsed = parse_idx_labels(&buf).unwrap();
        assert_eq!(parsed.n, 3);
        assert_eq!(parsed.data.len(), 30); // 3 * 10
        // Sample 0: label=7
        assert_eq!(parsed.data[7], 1.0);
        assert_eq!(parsed.data[0], 0.0);
        // Sample 1: label=2
        assert_eq!(parsed.data[10 + 2], 1.0);
        // Sample 2: label=0
        assert_eq!(parsed.data[20], 1.0);
    }

    #[test]
    fn test_parse_idx_bad_magic() {
        let buf = [0, 0, 0, 0, 0, 0, 0, 1, 0];
        assert!(parse_idx_labels(&buf).is_err());
    }

    #[test]
    fn test_parse_idx_truncated() {
        let mut buf = Vec::new();
        buf.extend_from_slice(&0x00000801_u32.to_be_bytes());
        buf.extend_from_slice(&100_u32.to_be_bytes()); // claims 100 labels
        buf.extend_from_slice(&[0, 1]); // only 2 bytes of data
        assert!(parse_idx_labels(&buf).is_err());
    }
}
