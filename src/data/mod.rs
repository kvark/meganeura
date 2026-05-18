//! Data loading utilities for training.
//!
//! Provides a [`DataLoader`] that yields mini-batches from an in-memory
//! dataset, and an [`MnistDataset`] that reads the standard IDX file format.
//!
//! Each loader exposes one or more named input streams; on every batch
//! the corresponding flat slice is gathered for that stream and bound to
//! a graph input of the same name via `Session::set_input`.

pub mod mnist;
pub mod safetensors;

pub use mnist::MnistDataset;

/// A single input stream definition: a named per-sample flat array.
///
/// `data.len()` must equal `n * sample_size` for some `n`; all streams in
/// a loader must agree on `n`.
pub struct InputStream {
    /// Name of the graph input bound to this stream
    /// (matches `Graph::input(name, …)`).
    pub name: String,
    /// Concatenated per-sample values, length `n * sample_size`.
    pub data: Vec<f32>,
    /// Number of `f32`s per sample.
    pub sample_size: usize,
}

impl InputStream {
    pub fn new(name: impl Into<String>, data: Vec<f32>, sample_size: usize) -> Self {
        Self {
            name: name.into(),
            data,
            sample_size,
        }
    }
}

/// A single mini-batch.
///
/// Each entry is `(input_name, gathered_slice)`; the slice borrows from
/// the loader's internal scratch and stays valid until the next call to
/// `next_batch`, `shuffle`, or `reset`.
pub struct Batch<'a> {
    pub tensors: Vec<(&'a str, &'a [f32])>,
}

impl<'a> Batch<'a> {
    /// Lookup a stream by name.
    pub fn get(&self, name: &str) -> Option<&'a [f32]> {
        for &(k, v) in &self.tensors {
            if k == name {
                return Some(v);
            }
        }
        None
    }
}

struct StreamState {
    name: String,
    data: Vec<f32>,
    sample_size: usize,
    scratch: Vec<f32>,
}

/// Iterates over a dataset in mini-batches, with optional shuffling.
///
/// Each named [`InputStream`] is gathered into per-batch scratch and
/// returned via [`Batch::tensors`]. The simple (data, labels) shape is
/// covered by [`DataLoader::new`]; multi-input graphs (e.g. blade-volume-
/// train's `cell_indices`, `dt`, `mask`, `labels`) use
/// [`DataLoader::with_streams`].
pub struct DataLoader {
    streams: Vec<StreamState>,
    batch_size: usize,
    /// Permutation of sample indices (shuffled each epoch).
    indices: Vec<usize>,
    pos: usize,
}

impl DataLoader {
    /// Create a loader with the canonical two-stream `(x, labels)` shape.
    ///
    /// Stream names are `"x"` and `"labels"`. Use [`with_streams`] for
    /// graphs that take more than two inputs.
    ///
    /// [`with_streams`]: Self::with_streams
    pub fn new(
        data: Vec<f32>,
        labels: Vec<f32>,
        sample_size: usize,
        label_size: usize,
        batch_size: usize,
    ) -> Self {
        Self::with_streams(
            vec![
                InputStream::new("x", data, sample_size),
                InputStream::new("labels", labels, label_size),
            ],
            batch_size,
        )
    }

    /// Create a loader from one or more named input streams.
    ///
    /// All streams must contain the same number of samples
    /// (`data.len() / sample_size` must agree). Pair with
    /// [`TrainConfig::default`](crate::TrainConfig::default) — the
    /// [`Trainer`](crate::Trainer) calls `set_input(name, …)` for each
    /// stream returned by `next_batch`, so stream names must match
    /// graph-input names.
    pub fn with_streams(streams: Vec<InputStream>, batch_size: usize) -> Self {
        assert!(!streams.is_empty(), "DataLoader needs at least one stream");
        let n0 = streams[0].data.len() / streams[0].sample_size;
        assert_eq!(
            streams[0].data.len(),
            n0 * streams[0].sample_size,
            "stream '{}' length not divisible by sample_size",
            streams[0].name,
        );
        for s in &streams[1..] {
            let n = s.data.len() / s.sample_size;
            assert_eq!(
                s.data.len(),
                n * s.sample_size,
                "stream '{}' length not divisible by sample_size",
                s.name,
            );
            assert_eq!(
                n, n0,
                "stream '{}' has {} samples, expected {} (must match first stream)",
                s.name, n, n0,
            );
        }
        assert!(
            n0 >= batch_size,
            "dataset ({n0} samples) smaller than batch_size ({batch_size})",
        );

        let states: Vec<StreamState> = streams
            .into_iter()
            .map(|s| {
                let scratch = vec![0.0; batch_size * s.sample_size];
                StreamState {
                    name: s.name,
                    data: s.data,
                    sample_size: s.sample_size,
                    scratch,
                }
            })
            .collect();
        let indices: Vec<usize> = (0..n0).collect();
        Self {
            streams: states,
            batch_size,
            indices,
            pos: 0,
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

    /// Names of the input streams, in the order returned by `next_batch`.
    pub fn input_names(&self) -> impl Iterator<Item = &str> {
        self.streams.iter().map(|s| s.name.as_str())
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
        for s in &mut self.streams {
            for b in 0..self.batch_size {
                let idx = self.indices[self.pos + b];
                let src = idx * s.sample_size..(idx + 1) * s.sample_size;
                let dst = b * s.sample_size..(b + 1) * s.sample_size;
                s.scratch[dst].copy_from_slice(&s.data[src]);
            }
        }
        self.pos += self.batch_size;
        let tensors = self
            .streams
            .iter()
            .map(|s| (s.name.as_str(), s.scratch.as_slice()))
            .collect();
        Some(Batch { tensors })
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
        let b1_data = b1.get("x").unwrap();
        let b1_labels = b1.get("labels").unwrap();
        assert_eq!(b1_data.len(), 12); // 4 * 3
        assert_eq!(b1_labels.len(), 8); // 4 * 2
        // First batch should be samples 0..4 in order (no shuffle)
        assert_eq!(b1_data[0], 0.0);
        assert_eq!(b1_data[3], 3.0); // start of sample 1

        let b2 = loader.next_batch().unwrap();
        assert_eq!(b2.get("x").unwrap()[0], 12.0); // start of sample 4

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
        assert_eq!(b.get("x").unwrap()[0], 0.0); // back to start
    }

    #[test]
    fn test_dataloader_shuffle() {
        let data: Vec<f32> = (0..30).map(|i| i as f32).collect();
        let labels: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let mut loader = DataLoader::new(data, labels, 3, 1, 5);

        loader.shuffle(42);
        let b = loader.next_batch().unwrap();
        // Just check the batch is valid — 5 samples of size 3
        assert_eq!(b.get("x").unwrap().len(), 15);
        assert_eq!(b.get("labels").unwrap().len(), 5);
    }

    #[test]
    fn test_dataloader_three_streams() {
        // Multi-input shape (e.g. blade-volume-train: cell_indices, dt,
        // mask, labels). Three streams, batch_size=2, n=4.
        let stream_a: Vec<f32> = (0..8).map(|i| i as f32).collect(); // sample_size=2
        let stream_b: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect(); // sample_size=4
        let stream_c: Vec<f32> = (0..4).map(|i| i as f32 + 100.0).collect(); // sample_size=1
        let mut loader = DataLoader::with_streams(
            vec![
                InputStream::new("a", stream_a, 2),
                InputStream::new("b", stream_b, 4),
                InputStream::new("c", stream_c, 1),
            ],
            2,
        );

        let names: Vec<&str> = loader.input_names().collect();
        assert_eq!(names, vec!["a", "b", "c"]);

        let batch = loader.next_batch().unwrap();
        assert_eq!(batch.get("a").unwrap().len(), 4); // batch_size * 2
        assert_eq!(batch.get("b").unwrap().len(), 8); // batch_size * 4
        assert_eq!(batch.get("c").unwrap().len(), 2); // batch_size * 1

        // First batch = samples 0, 1 (no shuffle yet).
        assert_eq!(batch.get("a").unwrap(), &[0.0, 1.0, 2.0, 3.0]);
        assert_eq!(batch.get("c").unwrap(), &[100.0, 101.0]);
    }

    #[test]
    #[should_panic(expected = "expected 4")]
    fn test_dataloader_stream_count_mismatch_rejected() {
        // Stream a has 4 samples (8 / 2), stream b has 3 samples (6 / 2).
        let a: Vec<f32> = vec![0.0; 8];
        let b: Vec<f32> = vec![0.0; 6];
        DataLoader::with_streams(
            vec![InputStream::new("a", a, 2), InputStream::new("b", b, 2)],
            2,
        );
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
