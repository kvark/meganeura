//! Magenta-RT: Google DeepMind's real-time music generation model.
//!
//! Three components:
//! - [`spectrostream`]: SoundStream-style 48 kHz stereo RVQ audio codec (encoder + decoder).
//! - [`musiccoca`]: text/audio joint embedder with RVQ-quantized output.
//! - [`llm`]: encoder-decoder Depthformer (T5 1.1 variant with hierarchical decoder).
//!
//! The high-level pipeline (per 2-sec chunk):
//! ```text
//! text prompt → MusicCoCa → 6 style tokens ─┐
//! 10s of audio context tokens (4 RVQ depths)─┴→ Encoder (1006 tokens)
//!                                                  ↓
//!                                            Decoder (800 = 50 frames × 16 RVQ)
//!                                                  ↓
//!                                            SpectroStream decoder → 48kHz stereo
//! ```
//!
//! Inference uses classifier-free guidance (batch=2: pos + neg style) with
//! temperature=1.1, top-k=40 sampling.
//!
//! Weight-loading map: weights are exported from the official magenta-realtime
//! python lib (TF SavedModels + T5X jax params) via
//! `tools/magenta_rt/convert_colab.{py,ipynb}` and saved as safetensors with a
//! companion `manifest.json`. See [`llm::LlmConfig`] for the parameter naming
//! convention.

pub mod llm;
pub mod musiccoca;
pub mod sampling;
pub mod spectrostream;

/// Top-level configuration mirroring `system.MagentaRTConfiguration` in
/// the official magenta_rt python library.
#[derive(Clone, Debug)]
pub struct MagentaRtConfig {
    /// Audio chunk length in seconds (2.0).
    pub chunk_length_sec: f32,
    /// Context window in seconds (10.0).
    pub context_length_sec: f32,
    /// Crossfade overlap between chunks in seconds (0.04 = 40 ms = 1 codec frame).
    pub crossfade_length_sec: f32,
    /// 48000 Hz.
    pub codec_sample_rate: u32,
    /// 25 Hz codec frame rate.
    pub codec_frame_rate: u32,
    /// 2 (stereo).
    pub codec_num_channels: u32,
    /// 1024 RVQ codebook entries per codebook level.
    pub codec_rvq_codebook_size: u32,
    /// 768-d MusicCoCa joint embedding.
    pub style_embedding_dim: u32,
    /// 1024 RVQ codebook entries for style.
    pub style_rvq_codebook_size: u32,
    /// 4 codec RVQ depths fed into LLM encoder (truncated from 64).
    pub encoder_codec_rvq_depth: u32,
    /// 6 style RVQ depths fed into LLM encoder.
    pub encoder_style_rvq_depth: u32,
    /// 16 codec RVQ depths produced by LLM decoder.
    pub decoder_codec_rvq_depth: u32,
}

impl Default for MagentaRtConfig {
    /// Production Magenta-RT config (matches the `MagentaRTT5X` default in system.py).
    fn default() -> Self {
        Self {
            chunk_length_sec: 2.0,
            context_length_sec: 10.0,
            crossfade_length_sec: 0.04,
            codec_sample_rate: 48000,
            codec_frame_rate: 25,
            codec_num_channels: 2,
            codec_rvq_codebook_size: 1024,
            style_embedding_dim: 768,
            style_rvq_codebook_size: 1024,
            encoder_codec_rvq_depth: 4,
            encoder_style_rvq_depth: 6,
            decoder_codec_rvq_depth: 16,
        }
    }
}

impl MagentaRtConfig {
    /// Samples per codec frame: `sample_rate / frame_rate` = 1920.
    pub fn frame_length_samples(&self) -> u32 {
        self.codec_sample_rate / self.codec_frame_rate
    }
    /// Samples per chunk: `chunk_length_sec * sample_rate` = 96000.
    pub fn chunk_length_samples(&self) -> u32 {
        (self.chunk_length_sec * self.codec_sample_rate as f32) as u32
    }
    /// Codec frames per chunk: `chunk_length_sec * frame_rate` = 50.
    pub fn chunk_length_frames(&self) -> u32 {
        (self.chunk_length_sec * self.codec_frame_rate as f32) as u32
    }
    /// Codec frames per context window: `context_length_sec * frame_rate` = 250.
    pub fn context_length_frames(&self) -> u32 {
        (self.context_length_sec * self.codec_frame_rate as f32) as u32
    }
    /// 1 (40 ms × 25 Hz).
    pub fn crossfade_length_frames(&self) -> u32 {
        (self.crossfade_length_sec * self.codec_frame_rate as f32) as u32
    }
    /// 1920 samples (1 codec frame at 48 kHz).
    pub fn crossfade_length_samples(&self) -> u32 {
        (self.crossfade_length_sec * self.codec_sample_rate as f32) as u32
    }

    // --- LLM vocab layout (mirrors `MagentaRTConfiguration` properties) ---

    /// 0.
    pub fn vocab_pad_token(&self) -> u32 { 0 }
    /// 1.
    pub fn vocab_mask_token(&self) -> u32 { 1 }
    /// 2.
    pub fn vocab_codec_offset(&self) -> u32 { 2 }
    /// `decoder_codec_rvq_depth * codec_rvq_codebook_size` = 16384.
    pub fn vocab_codec_size(&self) -> u32 {
        self.decoder_codec_rvq_depth * self.codec_rvq_codebook_size
    }
    /// `vocab_codec_offset + vocab_codec_size + 1024` (1024 reserved/unused) = 17410.
    pub fn vocab_style_offset(&self) -> u32 {
        self.vocab_codec_offset() + self.vocab_codec_size() + 1024
    }
    /// `encoder_style_rvq_depth * style_rvq_codebook_size` = 6144.
    pub fn vocab_style_size(&self) -> u32 {
        self.encoder_style_rvq_depth * self.style_rvq_codebook_size
    }
    /// Effective vocab size used by the model = 23554.
    pub fn vocab_size(&self) -> u32 {
        self.vocab_style_offset() + self.vocab_style_size()
    }
    /// Padded vocab the LLM was trained with = 29698 (tokens above
    /// [`Self::vocab_size`] are unused).
    pub fn vocab_size_pretrained(&self) -> u32 { 29698 }

    /// Total encoder input length: `context_frames * encoder_codec_rvq_depth + encoder_style_rvq_depth` = 1006.
    pub fn encoder_input_length(&self) -> u32 {
        self.context_length_frames() * self.encoder_codec_rvq_depth + self.encoder_style_rvq_depth
    }
    /// Total decoder output length: `chunk_frames * decoder_codec_rvq_depth` = 800.
    pub fn decoder_output_length(&self) -> u32 {
        self.chunk_length_frames() * self.decoder_codec_rvq_depth
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_matches_magenta_rt_paper_dims() {
        let c = MagentaRtConfig::default();
        assert_eq!(c.frame_length_samples(), 1920);
        assert_eq!(c.chunk_length_samples(), 96000);
        assert_eq!(c.chunk_length_frames(), 50);
        assert_eq!(c.context_length_frames(), 250);
        assert_eq!(c.crossfade_length_frames(), 1);
        assert_eq!(c.crossfade_length_samples(), 1920);
        assert_eq!(c.vocab_codec_size(), 16384);
        assert_eq!(c.vocab_style_offset(), 17410);
        assert_eq!(c.vocab_size(), 23554);
        assert_eq!(c.encoder_input_length(), 1006);
        assert_eq!(c.decoder_output_length(), 800);
    }
}
