//! Data loading utilities for training.
//!
//! Provides a [`DataLoader`] that yields mini-batches from an in-memory
//! dataset, and an [`MnistDataset`] that reads the standard IDX file format.

pub mod safetensors;
pub mod mnist;

pub use mnist::MnistDataset;

/// A single batch of input data and labels.
pub struct Batch<'a> {
    pub data: &'a [f32],
    pub labels: &'a [f32],
}

/// Iterates over a dataset in mini-batches, with optional shuffling.
///
/// Data is stored as flat `f32` slices in row-major order. Each sample
/// occupies `sample_size` contiguous elements in `data` and `label_size`
/// elements in `labels`.
pub struct DataLoader {
    data: Vec<f32>,
    labels: Vec<f32>,
    sample_size: usize,
    label_size: usize,
    batch_size: usize,
    /// Permutation of sample indices (shuffled each epoch).
    indices: Vec<usize>,
    pos: usize,
    /// Scratch buffers for gathering shuffled batches.
    batch_data: Vec<f32>,
    batch_labels: Vec<f32>,
}

impl DataLoader {
    /// Create a loader from pre-loaded flat arrays.
    ///
    /// * `data` – all samples concatenated, length must be `n * sample_size`.
    /// * `labels` – all labels concatenated, length must be `n * label_size`.
    /// * `batch_size` – number of samples per batch.
    pub fn new(
        data: Vec<f32>,
        labels: Vec<f32>,
        sample_size: usize,
        label_size: usize,
        batch_size: usize,
    ) -> Self {
        let n = data.len() / sample_size;
        assert_eq!(
            data.len(),
            n * sample_size,
            "data length not divisible by sample_size"
        );
        assert_eq!(
            labels.len(),
            n * label_size,
            "labels length not divisible by label_size"
        );
        assert!(
            n >= batch_size,
            "dataset ({n} samples) smaller than batch_size ({batch_size})"
        );
        let indices: Vec<usize> = (0..n).collect();
        Self {
            data,
            labels,
            sample_size,
            label_size,
            batch_size,
            indices,
            pos: 0,
            batch_data: vec![0.0; batch_size * sample_size],
            batch_labels: vec![0.0; batch_size * label_size],
        }
    }

    /// Number of samples in the dataset.
    pub fn len(&self) -> usize {
        self.indices.len()
    }

    /// Whether the dataset is empty.
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Number of complete batches per epoch.
    pub fn num_batches(&self) -> usize {
        self.len() / self.batch_size
    }

    /// Shuffle sample order using a simple LCG seeded with `seed`.
    pub fn shuffle(&mut self, seed: u64) {
        // Fisher-Yates shuffle with a lightweight LCG PRNG.
        let n = self.indices.len();
        let mut state = seed.wrapping_add(1);
        for i in (1..n).rev() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let j = (state >> 33) as usize % (i + 1);
            self.indices.swap(i, j);
        }
    }

    /// Reset the iterator to the beginning (without shuffling).
    pub fn reset(&mut self) {
        self.pos = 0;
    }

    /// Return the next mini-batch, or `None` if the epoch is exhausted.
    ///
    /// The returned slices borrow internal scratch buffers and are valid
    /// until the next call to `next_batch`, `shuffle`, or `reset`.
    pub fn next_batch(&mut self) -> Option<Batch<'_>> {
        let remaining = self.len() - self.pos;
        if remaining < self.batch_size {
            return None;
        }
        // Gather samples according to the current permutation.
        for b in 0..self.batch_size {
            let idx = self.indices[self.pos + b];
            let src = idx * self.sample_size..(idx + 1) * self.sample_size;
            let dst = b * self.sample_size..(b + 1) * self.sample_size;
            self.batch_data[dst].copy_from_slice(&self.data[src]);

            let lsrc = idx * self.label_size..(idx + 1) * self.label_size;
            let ldst = b * self.label_size..(b + 1) * self.label_size;
            self.batch_labels[ldst].copy_from_slice(&self.labels[lsrc]);
        }
        self.pos += self.batch_size;
        Some(Batch {
            data: &self.batch_data,
            labels: &self.batch_labels,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataloader_basic() {
        // 8 samples, sample_size=3, label_size=2, batch_size=4
        let data: Vec<f32> = (0..24).map(|i| i as f32).collect();
        let labels: Vec<f32> = (0..16).map(|i| i as f32 * 0.1).collect();
        let mut loader = DataLoader::new(data, labels, 3, 2, 4);

        assert_eq!(loader.len(), 8);
        assert_eq!(loader.num_batches(), 2);

        let b1 = loader.next_batch().unwrap();
        assert_eq!(b1.data.len(), 12); // 4 * 3
        assert_eq!(b1.labels.len(), 8); // 4 * 2
        // First batch should be samples 0..4 in order (no shuffle)
        assert_eq!(b1.data[0], 0.0);
        assert_eq!(b1.data[3], 3.0); // start of sample 1

        let b2 = loader.next_batch().unwrap();
        assert_eq!(b2.data[0], 12.0); // start of sample 4

        // Epoch exhausted
        assert!(loader.next_batch().is_none());
    }

    #[test]
    fn test_dataloader_reset() {
        let data: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let labels: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let mut loader = DataLoader::new(data, labels, 3, 2, 2);

        let _ = loader.next_batch();
        loader.reset();
        let b = loader.next_batch().unwrap();
        assert_eq!(b.data[0], 0.0); // back to start
    }

    #[test]
    fn test_dataloader_shuffle() {
        let data: Vec<f32> = (0..30).map(|i| i as f32).collect();
        let labels: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let mut loader = DataLoader::new(data, labels, 3, 1, 5);

        loader.shuffle(42);
        let b = loader.next_batch().unwrap();
        // Just check the batch is valid — 5 samples of size 3
        assert_eq!(b.data.len(), 15);
        assert_eq!(b.labels.len(), 5);
    }

    #[test]
    fn test_dataloader_partial_last_batch_dropped() {
        // 5 samples, batch_size=2 → 2 full batches, last sample dropped
        let data: Vec<f32> = vec![0.0; 10];
        let labels: Vec<f32> = vec![0.0; 5];
        let mut loader = DataLoader::new(data, labels, 2, 1, 2);

        assert_eq!(loader.num_batches(), 2);
        assert!(loader.next_batch().is_some());
        assert!(loader.next_batch().is_some());
        assert!(loader.next_batch().is_none());
    }

    #[test]
    #[should_panic(expected = "dataset")]
    fn test_dataloader_too_small() {
        let data = vec![0.0; 6]; // 2 samples
        let labels = vec![0.0; 2];
        DataLoader::new(data, labels, 3, 1, 5); // batch_size > n
    }
}
