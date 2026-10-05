//! Building blocks spelled in primitives: what audio and sequence models
//! such as Lyria (Magenta RealTime) need beyond the composite set.
//!
//! None of these is an op of its own. Each is a short composition of
//! primitives, so it runs, differentiates and checks against the reference
//! like any other graph, and needs no kernel or rewrite rule. Tensors in
//! NCHW layout are flat, as for convolution, with their extents given by an
//! [`Nchw`].

use super::{Graph, NodeId};

/// Extents of a flat NCHW tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Nchw {
    pub batch: u32,
    pub channels: u32,
    pub h: u32,
    pub w: u32,
}

impl Nchw {
    pub fn len(self) -> usize {
        self.batch as usize * self.channels as usize * self.h as usize * self.w as usize
    }

    pub fn is_empty(self) -> bool {
        self.len() == 0
    }

    /// Batch × channel planes.
    fn planes(self) -> u32 {
        self.batch * self.channels
    }
}

/// T5's relative-position bucket of a query `relative` positions after a
/// key (`query - key`, as flaxformer counts), with `num_buckets` buckets
/// reaching out to `max_distance`. Bidirectionally, keys after the query
/// take the upper half of the buckets. Computed in `f32`, as the reference
/// implementations do.
pub fn t5_bucket(relative: i32, num_buckets: u32, max_distance: u32, bidirectional: bool) -> u32 {
    let mut n = relative;
    let mut base = 0;
    let mut buckets = num_buckets;
    if bidirectional {
        buckets /= 2;
        if n < 0 {
            base = buckets;
            n = -n;
        }
    } else {
        n = n.max(0);
    }
    let exact = buckets / 2;
    let n = n as u32;
    if n < exact {
        return base + n;
    }
    let log_ratio = (n as f32 / exact as f32).ln() / (max_distance as f32 / exact as f32).ln();
    let large = (exact as f32 + log_ratio * (buckets - exact) as f32) as u32;
    base + large.min(buckets - 1)
}

impl Graph {
    /// ELU with α = 1: `x` where positive, `exp(x) - 1` elsewhere, as
    /// `relu(x) + exp(-relu(-x)) - 1`.
    pub fn elu(&mut self, x: NodeId) -> NodeId {
        let positive = self.relu(x);
        let negated = self.neg(x);
        let negative = self.relu(negated);
        let negative = self.neg(negative);
        let tail = self.exp(negative);
        let tail = self.add_scalar(tail, -1.0);
        self.add(positive, tail)
    }

    /// Crop `[top, bottom]` rows and `[left, right]` columns from every
    /// plane of `x`. Returns a flat `[batch, channels, h - top - bottom,
    /// w - left - right]`.
    #[track_caller]
    pub fn crop_2d(
        &mut self,
        x: NodeId,
        dims: Nchw,
        [top, bottom]: [u32; 2],
        [left, right]: [u32; 2],
    ) -> NodeId {
        assert!(
            top + bottom < dims.h && left + right < dims.w,
            "crop leaves nothing"
        );
        let rows = dims.h - top - bottom;
        let mut out = x;
        // Rows of every plane, then columns of every row.
        if top > 0 {
            out = self.split_b(out, dims.planes(), top, dims.h - top, dims.w);
        }
        if bottom > 0 {
            out = self.split_a(out, dims.planes(), rows, bottom, dims.w);
        }
        let lines = dims.planes() * rows;
        if left > 0 {
            out = self.split_b(out, lines, left, dims.w - left, 1);
        }
        if right > 0 {
            out = self.split_a(out, lines, dims.w - left - right, right, 1);
        }
        self.flat(out)
    }

    /// Channel-to-width pixel shuffle: `[b, c, h, w] → [b, c / f, h, w · f]`
    /// with `out[b, c, h, f·w + k] = x[b, k · c/f + c, h, w]`.
    #[track_caller]
    pub fn pixel_shuffle_w(&mut self, x: NodeId, dims: Nchw, factor: u32) -> NodeId {
        assert!(
            factor > 0 && dims.channels.is_multiple_of(factor),
            "channels must divide by the factor"
        );
        // The channel group k moves past every (c, h, w) of its slice.
        let block = (dims.channels / factor * dims.h * dims.w) as usize;
        let x = self.view(x, &[dims.batch as usize, factor as usize, block]);
        let out = self.permute(x, &[0, 2, 1]);
        self.flat(out)
    }

    /// Insert `stride - 1` zeros between neighbours along each row:
    /// `[b, c, h, w] → [b, c, h, w · stride - (stride - 1)]`.
    #[track_caller]
    pub fn dilate_w(&mut self, x: NodeId, dims: Nchw, stride: u32) -> NodeId {
        assert!(stride > 0, "stride must be positive");
        if stride == 1 {
            return x;
        }
        let pixels = dims.len() as u32;
        let gap = dims.len() * (stride as usize - 1);
        let zeros = self.zeros(1, gap);
        let zeros = self.view(zeros, &[gap]);
        // Each pixel followed by its zeros, then the trailing zeros of a
        // row dropped.
        let spaced = self.concat(x, zeros, pixels, 1, stride - 1, 1);
        let lines = dims.planes() * dims.h;
        let width = dims.w * stride;
        let out = self.split_a(spaced, lines, width - (stride - 1), stride - 1, 1);
        self.flat(out)
    }

    /// Insert `stride - 1` zero rows between neighbouring rows:
    /// `[b, c, h, w] → [b, c, h · stride - (stride - 1), w]`.
    #[track_caller]
    pub fn dilate_h(&mut self, x: NodeId, dims: Nchw, stride: u32) -> NodeId {
        assert!(stride > 0, "stride must be positive");
        if stride == 1 {
            return x;
        }
        let rows = dims.planes() * dims.h;
        let gap = rows as usize * (stride as usize - 1) * dims.w as usize;
        let zeros = self.zeros(1, gap);
        let zeros = self.view(zeros, &[gap]);
        let spaced = self.concat(x, zeros, rows, dims.w, (stride - 1) * dims.w, 1);
        let height = dims.h * stride;
        let out = self.split_a(
            spaced,
            dims.planes(),
            (height - (stride - 1)) * dims.w,
            (stride - 1) * dims.w,
            1,
        );
        self.flat(out)
    }

    /// Nearest-neighbour upsampling by whole factors:
    /// `[b, c, h, w] → [b, c, h · scale_h, w · scale_w]`.
    #[track_caller]
    pub fn upsample_nearest(
        &mut self,
        x: NodeId,
        dims: Nchw,
        scale_h: u32,
        scale_w: u32,
    ) -> NodeId {
        assert!(scale_h > 0 && scale_w > 0, "scales must be positive");
        let rows = (dims.planes() * dims.h) as usize;
        let w = dims.w as usize;
        let x = self.view(x, &[rows, 1, w, 1]);
        let out = self.broadcast_to(x, &[rows, scale_h as usize, w, scale_w as usize]);
        self.flat(out)
    }

    /// Transposed convolution with PyTorch's `ConvTranspose2d` kernel
    /// layout `[in_channels, out_channels, kernel_h, kernel_w]` (flat) and
    /// separate strides. The output is `(in - 1) · stride - 2 · padding +
    /// kernel` along each axis.
    ///
    /// It runs as a forward convolution of the zero-dilated input with the
    /// flipped, channel-transposed kernel, which reaches the convolution's
    /// fastest lowering.
    #[track_caller]
    pub fn conv_transpose_2d(
        &mut self,
        x: NodeId,
        kernel: NodeId,
        dims: Nchw,
        out_channels: u32,
        [kernel_h, kernel_w]: [u32; 2],
        [stride_h, stride_w]: [u32; 2],
        [padding_h, padding_w]: [u32; 2],
    ) -> NodeId {
        let (ci, co) = (dims.channels as usize, out_channels as usize);
        let taps = (kernel_h * kernel_w) as usize;
        assert_eq!(
            self.node(kernel).ty.num_elements(),
            ci * co * taps,
            "kernel size"
        );
        // Forward kernel [out, in, kh, kw] = transposed[in, out, flipped].
        let k = self.view(kernel, &[ci, co, taps]);
        let k = self.permute(k, &[1, 0, 2]);
        let k = self.view(k, &[co * ci, taps]);
        let mut flip = vec![0.0; taps * taps];
        for t in 0..taps {
            flip[t * taps + (taps - 1 - t)] = 1.0;
        }
        let flip = self.constant(flip, &[taps, taps]);
        let k = self.matmul(k, flip);
        let k = self.flat(k);

        let x = self.dilate_h(x, dims, stride_h);
        let tall = Nchw {
            h: (dims.h - 1) * stride_h + 1,
            ..dims
        };
        let x = self.dilate_w(x, tall, stride_w);
        let dilated = Nchw {
            w: (dims.w - 1) * stride_w + 1,
            ..tall
        };
        // Pad by kernel - 1 - padding; a larger padding crops instead.
        let pad = |k: u32, p: u32| (k - 1).saturating_sub(p);
        let crop = |k: u32, p: u32| p.saturating_sub(k - 1);
        let (pad_h, pad_w) = (pad(kernel_h, padding_h), pad(kernel_w, padding_w));
        let out = self.conv2d_hw(
            x,
            k,
            dims.batch,
            dims.channels,
            dilated.h,
            dilated.w,
            out_channels,
            kernel_h,
            kernel_w,
            1,
            pad_h,
            pad_w,
        );
        let (crop_h, crop_w) = (crop(kernel_h, padding_h), crop(kernel_w, padding_w));
        if crop_h == 0 && crop_w == 0 {
            return out;
        }
        let full = Nchw {
            batch: dims.batch,
            channels: out_channels,
            h: dilated.h + 2 * pad_h - kernel_h + 1,
            w: dilated.w + 2 * pad_w - kernel_w + 1,
        };
        self.crop_2d(out, full, [crop_h, crop_h], [crop_w, crop_w])
    }

    /// T5's relative-position bias `[heads, q_len, kv_len]` from a learned
    /// `[heads, num_buckets]` table, for queries and keys at positions
    /// `0..q_len` and `0..kv_len`: a gather by the constant bucket map.
    /// Feed it to [`Graph::biased_attention`] with scale 1.
    #[track_caller]
    pub fn t5_relative_bias(
        &mut self,
        table: NodeId,
        [q_len, kv_len]: [usize; 2],
        max_distance: u32,
        bidirectional: bool,
    ) -> NodeId {
        let (heads, buckets) = self.t5_table(table);
        let indices: Vec<u32> = (0..q_len * kv_len)
            .map(|n| {
                let (i, j) = ((n / kv_len) as i32, (n % kv_len) as i32);
                t5_bucket(i - j, buckets as u32, max_distance, bidirectional)
            })
            .collect();
        let indices = self.constant_u32(&indices, &[q_len * kv_len]);
        let by_bucket = self.transpose(table);
        let bias = self.embedding(indices, by_bucket);
        let bias = self.transpose(bias);
        self.view(bias, &[heads, q_len, kv_len])
    }

    /// T5's relative-position bias of a query at the runtime position
    /// `kv_pos` (a `U32` scalar) against cache rows `0..max_seq`:
    /// `[heads, max_seq]`, for [`Graph::biased_cached_attention`]. The bias
    /// of every distance is gathered once from the table; the step's row
    /// is a gather from that at offsets computed from `kv_pos`.
    #[track_caller]
    pub fn t5_relative_bias_cached(
        &mut self,
        table: NodeId,
        kv_pos: NodeId,
        max_seq: usize,
        max_distance: u32,
        bidirectional: bool,
    ) -> NodeId {
        let (_, buckets) = self.t5_table(table);
        // Distance `query - key` from -(max_seq - 1) to max_seq - 1, at
        // index distance + max_seq - 1.
        let span = 2 * max_seq - 1;
        let indices: Vec<u32> = (0..span)
            .map(|t| {
                t5_bucket(
                    t as i32 - (max_seq as i32 - 1),
                    buckets as u32,
                    max_distance,
                    bidirectional,
                )
            })
            .collect();
        let indices = self.constant_u32(&indices, &[span]);
        let by_bucket = self.transpose(table);
        let by_distance = self.embedding(indices, by_bucket);
        // Key j sits at index kv_pos - j + max_seq - 1.
        let pos = self.to_f32(kv_pos);
        let pos = self.view(pos, &[1, 1]);
        let pos = self.broadcast_inner(pos, max_seq);
        let pos = self.view(pos, &[max_seq]);
        let offsets = self.constant(
            (0..max_seq).map(|j| (max_seq - 1 - j) as f32).collect(),
            &[max_seq],
        );
        let index = self.add(pos, offsets);
        let index = self.to_u32(index);
        let row = self.embedding(index, by_distance);
        self.transpose(row)
    }

    /// `[heads, buckets]` of a T5 bias table.
    #[track_caller]
    fn t5_table(&self, table: NodeId) -> (usize, usize) {
        match self.node(table).ty.shape[..] {
            [heads, buckets] if buckets >= 2 => (heads, buckets),
            ref shape => panic!("T5 bias table must be [heads, buckets], got {shape:?}"),
        }
    }

    /// `x` as a flat vector.
    fn flat(&mut self, x: NodeId) -> NodeId {
        let n = self.node(x).ty.num_elements();
        self.view(x, &[n])
    }
}
