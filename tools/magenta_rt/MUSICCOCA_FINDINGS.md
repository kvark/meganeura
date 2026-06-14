# MusicCoCa reverse-engineering notes

## What we have

Dumped (in `magenta_rt_codec_dump/`):
- `weights_musiccoca.safetensors` — 80 MusicCoCa tensors + 12 RVQ codebooks.
  - **Variable names are opaque (`tf_var_leaves.0` … `tf_var_leaves.79`)** because
    the published `musiccoca_mv212f_cpu_*` SavedModel was exported via
    `tf_var_leaves` flattening with no symbolic structure preserved
    (confirmed by parsing `saved_model.pb` — only 80 anonymous leaves).
- `musiccoca_vocab.model` — SentencePiece vocab.
- `musiccoca_testdata.safetensors` — 26 prompts → reference embeddings + tokens
  + a 2-second audio test clip.
- `musiccoca_inputs.txt` — the 26 prompts (Ambient, Blues, …, Zamba).
- `musiccoca_manifest.json` — full variable shape list.

## Architecture from shapes (educated guess)

| Shape | Count | Likely role |
|-------|-------|-------------|
| `[12, 768, 12, 64]` | 8 | Q/K/V/O projections — 12 layers × 768 dim × 12 heads × 64 head_dim. 8 = 2 modules × 4 (audio + text encoder) |
| `[12, 12, 64]` | 6 | Per-head biases per layer |
| `[12, 768]` | 12 | LayerNorm scales (12 layers × 1 module × multiple LNs) |
| `[768]` | 12 | LayerNorm biases or final norms |
| `[12, 768, 3072]` | 2 | MLP up-projection (audio + text) |
| `[12, 3072, 768]` | 2 | MLP down-projection |
| `[768, 12, 256]` | 8 | ?? 12 heads × 256 features — extra projection? |
| `[12, 256]` | 6 | Matching biases |
| `[1, 768]` | 2 | CLS tokens (audio + text) |
| `[496, 768]` | 1 | Position embedding (max_seq_len=496) |
| `[3, 768, 3072]`, `[3, 768, 12, 256]`, … | several | 3-layer cross-attention "captioner" / multimodal decoder |
| `[768, 64000]` | 1 | Output embedding (vocab=64000 for text caption decoder) |
| `[64000]` | 1 | Output bias |

Best guess: 2 × 12-layer transformer encoders (text + audio) + 3-layer
multimodal decoder. Embed dim 768, 12 heads × 64 head_dim, MLP 3072.

## Why we can't proceed via SavedModel inspection alone

1. The CPU SavedModel embeds `SentencepieceOp` from `tensorflow-text`, which
   makes `tf.saved_model.load` fail without that op registered. Nixpkgs has
   no `tensorflow-text` package; pip wheels are Python ≤3.12 only.
2. The "novocab" SavedModel loads but only has `tf_var_leaves.N` indices.
3. `saved_model.pb`'s `SavedObjectGraph` has zero symbolic children for the
   variables (only `tf_var_leaves` → indexed children).
4. The architecture source is NOT in the public
   [`magenta/magenta-realtime`](https://github.com/magenta/magenta-realtime)
   repo — `magenta_rt/musiccoca.py` only *uses* the model via TFLite/SavedModel
   interpreters, never defines layers.

## Quantizer

- 12 codebooks × `[768, 1024]` — k-means VQ on the 768-dim embedding.
- Matches the `MusicCoCa dim=768, codebook=1024, rvq_depth=12` config.
- This part is straightforward to implement in Rust.

## Test data

- 26 lowercase genre prompts ("Ambient", … "Zamba")
- 26 × 768 reference embeddings (range [-12.5, 12.5])
- 26 × 12 tokens (each in [0, 1023]) — the quantized result
- An additional 2 × 160000 audio clip with 2 × 768 audio embeddings

## Authoritative architecture (Lyria Team paper, arXiv 2508.04651, §2.2)

> *"The audio embedding tower M_A is a 12-layer VisionTransformer (ViT). Its
> input is a log-mel spectrogram of a 10s slice of 16kHz audio (128 channels
> and length 992; split into patches of size 16 × 16). The text embedding
> tower M_T is a 12-layer Transformer, which operates on tokenized text with
> a maximum sequence length of 128 tokens. We use attention pooling to reduce
> the activations of each tower to a single 768d embedding, which can be
> subsequently quantized into 12 discrete tokens with codebook size |Vm| = 1024."*
>
> *"In addition to the two embedding towers, MusicCoCa has a text decoder
> which can generate audio text captions. In our application this decoder is
> a shallow 3-layer Transformer which only serves a regularizing purpose."*

So:
- **Audio tower**: 12-layer ViT, 128×992 log-mel spectrogram, 16×16 patches →
  8 × 62 = **496 patches** ⇒ the `[496, 768]` tensor is the **audio**
  position embedding.
- **Text tower**: 12-layer Transformer, max_seq_len **128** tokens. We don't
  see a `[128, 768]` in the shape list, so text positions are likely
  sinusoidal (no learned weights), OR the audio's `[496, 768]` table is
  shared and text uses the first 128 slots.
- **Attention pooling per tower**: a single learned query (the `[1, 768]`
  tensors? — we see 2 of these) attends over the tower's output tokens.
- **Quantizer**: 12 × `[768, 1024]` k-means centroids.
- **Captioner (regularization, not needed for inference)**: 3-layer
  Transformer with cross-attention — explains the `[3, ...]` shapes
  (`[3, 768, 3072]`, `[3, 3072, 768]`, `[3, 3, 768, 12, 256]`, etc.).
- **Output projection to vocab=64000**: `[768, 64000]` + `[64000]` — only
  used by the captioner; the *contrastive 768d output* uses the attention
  pool result directly.

The relevant tensors for **text → 768d → 12 tokens** inference:
- Text token embedding table (not yet identified — should be `[64000, 768]`,
  but we don't see that exact shape, so it's likely the **transpose** of the
  `[768, 64000]` output projection, with weight tying as in CoCa).
- Per-layer Q/K/V/O projections: 4 of the `[12, 768, 12, 64]` × 8 are text
  (the other 4 are audio).
- Per-head biases: 3 of the `[12, 12, 64]` × 6 are text Q/K/V (no O bias).
- MLP up/down: 1 of each `[12, 768, 3072]`/`[12, 3072, 768]`/`[12, 3072]`.
- 2 LN-positions per layer × 12 layers (stored fused as `[12, 768]`).
- 1 final LN per tower (the `[768]` × 12 spread).
- Attention pooling: 1 query `[1, 768]`, plus the pool's own attention proj.
- Final contrastive projection from pool to 768d shared space.

**Unmapped concern**: there are also `[768, 12, 256]` × 8 and `[12, 256]` × 6
shapes. These do NOT fit standard 12-layer attention (768 → 12×64 = 768
heads). The `256` dim suggests **a different head_dim (=256/12... not integer)
OR an internal_dim like 12 heads × 256 = 3072**. Most likely: these are the
attention-pool's K/V projections and biases, where the pool uses 12 heads × 256
head_dim = 3072 to match MLP dim. Pool has 4 projections (Q, K, V, O) per
tower × 2 towers = 8 ⇒ matches the 8 count. 3 biases per tower × 2 = 6 ✓.

## TFLite-extracted per-layer mapping (16 of 80 vars)

The `cpu_novocab` SavedModel exposes two signatures: `embed_text` and
`embed_music_spectrogram`. The `embed_text` body is an `XlaCallModule` with
80 resource inputs (`jax2tf_arg_0` … `jax2tf_arg_79`), so the same flat
`tf_var_leaves.N` indexing as the safetensors dump.

`tools/magenta_rt/convert_musiccoca_tflite.py` converts that signature to
TFLite (574 MB; written to `/tmp/musiccoca_text_encoder.tflite`). The
TFLite tensor names *do* preserve `jax2tf_arg_N/ReadVariableOp;…` for the
16 args sliced inside the per-layer `while` body (the per-layer transformer
weights). The other 64 args (token embed, pos embed, attention pool,
contrastive proj, audio path, captioner) are constant-folded by the
TFLite converter and lose their original index.

`tools/magenta_rt/extract_musiccoca_mapping.py` confirms the per-layer
mapping. The 16 per-layer args (used by the 12 stacked text encoder
layers) are:

```
arg 64  [12, 3072]            — MLP up bias  (one of …_dense_0 / mlp.up.bias)
arg 65  [12, 768, 3072]       — MLP up kernel
arg 66  [12, 768]             — likely MLP down bias
arg 67  [12, 3072, 768]       — MLP down kernel
arg 68  [12, 768]   ┐
arg 69  [12, 768]   │
arg 70  [12, 768]   │ — six per-layer [12, 768] params: pre-attn LN
arg 71  [12, 768]   │   scale + bias, pre-MLP LN scale + bias, and the
arg 74  [12, 768]   │   two attn out-proj bias / pre-attn bias
arg 66  [12, 768]   ┘
arg 72  [12, 12, 64]          — Q/K/V bias (3 of these, no out-proj bias)
arg 76  [12, 12, 64]
arg 78  [12, 12, 64]
arg 73  [12, 768, 12, 64]     — Q/K/V/out_proj kernel (4 of these)
arg 75  [12, 768, 12, 64]
arg 77  [12, 768, 12, 64]
arg 79  [12, 768, 12, 64]
```

Per-layer structure is therefore confirmed standard:
- 1 MLP up + 1 MLP down (both kernel + bias)
- 4 attention projections (Q, K, V, output) all stored as `[12, 768, 12, 64]`
- 3 attention biases `[12, 12, 64]` (one is missing — typically output proj
  has no bias)
- 6 `[12, 768]` params: 2 LNs (pre-attn, pre-mlp) × (scale + bias) + 2 others

The specific role of each within Q/K/V/out and which `[12, 768]` is which
LN scale/bias still needs to be brute-forced by matching the SavedModel's
`embed_text` output for a known prompt.

For args 0–63 (the outer computation), the TFLite has the constants but
no `jax2tf_arg_N` name on them. Tool: `match_musiccoca_constants.py`
matches every constant in subgraph 0 against the 80 raw safetensors
variables by sorted-value comparison (handles arbitrary reshape/transpose).

Result: **28/80 args mapped**. Per-layer text encoder block (args 64–79)
plus 12 more outer text-branch variables identified:

```
# Per-layer text-encoder block (alphabetical inside blocks; block order
# is mlp → 4×LN → attention) — 12 layers stacked via JAX scan

arg 64  [12, 3072]            mlp.wi.bias
arg 65  [12, 768, 3072]       mlp.wi.kernel
arg 66  [12, 768]             mlp.wo.bias
arg 67  [12, 3072, 768]       mlp.wo.kernel
arg 68  [12, 768]   ┐
arg 69  [12, 768]   │  pre-attn/pre-mlp LayerNorm scale+bias
arg 70  [12, 768]   │  (4 params, alphabetical order TBD)
arg 71  [12, 768]   ┘
arg 72  [12, 12, 64]          attention.key.bias
arg 73  [12, 768, 12, 64]     attention.key.kernel
arg 74  [12, 768]             attention.out.bias       ← (out alphabetically between key/query)
arg 75  [12, 768, 12, 64]     attention.out.kernel
arg 76  [12, 12, 64]          attention.query.bias
arg 77  [12, 768, 12, 64]     attention.query.kernel
arg 78  [12, 12, 64]          attention.value.bias
arg 79  [12, 768, 12, 64]     attention.value.kernel

# Outer (text-branch) — matched via value comparison

arg 13  [12, 256]             attention pool key/query/value bias (head bias)
arg 14  [768, 12, 256]        attention pool projection 1 (Q/K/V/O TBD)
arg 16  [768]                 attention pool LN bias or out bias
arg 17  [768, 12, 256]        attention pool projection 2
arg 18  [12, 256]             attention pool head bias
arg 19  [768, 12, 256]        attention pool projection 3
arg 20  [12, 256]             attention pool head bias
arg 21  [768, 12, 256]        attention pool projection 4
arg 22  [768]                 LN scale/bias
arg 24  [1, 768]              CLS / pool query
arg 27  [768, 64000]          text token embedding (or transpose for vocab logits)
arg 62  [768]                 LN scale/bias
```

The 52 unmatched args (0–12, 15, 23, 25, 26, 28–61, 63) belong to the
audio branch (12-layer ViT + audio attn pool + position embed [496, 768])
and the 3-layer captioner, which `embed_text` does not exercise.

Within the per-block alphabetical ordering observation, the
[12, 256] biases on the attention pool also sort key → out → query → value.
So tentatively the attn-pool mapping is:
  arg 14 = key.kernel,  arg 13 = key.bias,
  arg 17 = out.kernel,  (no out.bias?),
  arg 19 = query.kernel, arg 18 = query.bias,
  arg 21 = value.kernel, arg 20 = value.bias.

Pending: which of the 4 [12, 768] LN params (args 68–71) is pre-attn
scale/bias vs pre-mlp scale/bias, plus the contrastive projection. These
should fall out of brute-forcing once a NumPy reference is wired up against
the SavedModel's `embed_text` output as oracle.

## NumPy reference + verification harness (`musiccoca_numpy_ref.py`)

Implements the text encoder forward pass + tests against the SavedModel
oracle (signature `embed_text`, output `contrastive_txt_embed` 768-d).

Current accuracy on a 1-token (SOS only) input:
  cosine similarity: **0.85**
  truth norm: 28.7   ours norm: 2.4   (~12× magnitude mismatch)

So the architecture is *directionally* mostly right but missing a scale
component. Empirical findings while sweeping:
- All 24 orderings of the 4 LN params (args 68–71) give nearly identical
  cosine (~0.85) — the LN choice among these is not the bottleneck.
- All 6 valid orderings of attention K/O/Q/V kernels (constrained by
  arg 75 = OUT.kernel via TFLite einsum signature `ABNH,DNH->ABD`) give
  similar cosine — self-attn is symmetric in Q/K swap, so we can't
  fully disambiguate without more reference points.
- arg 25 [768] stands out among the [768] biases: mean = 0.58 (vs ~0
  for others), std = 0.10, norm = 16.3. This is "LN-scale-shaped"
  (typical scale init is 1, with subsequent training drift). But
  applying it as LN scale before or after the pool flips cos negative
  — so the right LN ordering / direction is still unknown.
- The unmatched `[256, 768]` (arg 42) could be a 256→768 projection,
  but probably belongs to the audio path (since text pool's OUT kernel
  is already arg 17 = [768, 12, 256]).

## Final architecture (RESOLVED via TFLite op-by-op trace)

After tracing TFLite subgraph 0 backwards from the output, the
post-encoder pipeline is fully decoded:

```python
# Token embedding lookup with sqrt(d) scaling.
x = embed_table[ids] * sqrt(768)

# Sinusoidal positional encoding — CONCATENATED [sin(...), cos(...)],
# NOT interleaved. inv_freq table = 1/10000^(2i/d), shape [384].
pe = concat([sin(pos * inv_freq), cos(pos * inv_freq)], axis=-1)
x = x + pe

# 12-layer pre-norm transformer encoder. Each layer:
for li in range(12):
    # LayerNorm uses scale_offset=+1.0 (flaxformer/CoCa convention:
    # trainable scale is stored as deviation from identity).
    #   per-layer: scale=arg71[li], bias=arg70[li], pre-MLP scale=arg69[li], pre-MLP bias=arg68[li]
    h = LayerNorm(x, scale=arg71[li] + 1, bias=arg70[li])
    Q = h @ arg77[li] + arg76[li]   # query.kernel, query.bias
    K = h @ arg73[li] + arg72[li]   # key.kernel,   key.bias
    V = h @ arg79[li] + arg78[li]   # value.kernel, value.bias
    attn = softmax(Q @ K^T * 1/sqrt(64) + pad_mask) @ V
    o    = attn @ arg75[li] + arg74[li]   # out.kernel, out.bias
    x = x + o
    h = LayerNorm(x, scale=arg69[li] + 1, bias=arg68[li])
    h2 = gelu(h @ arg65[li] + arg64[li]) @ arg67[li] + arg66[li]
    x = x + h2

# Final LN before attention pool (scale = arg23 + 1, bias = arg22).
# Wait, this is wrong — the final LN happens AFTER the attn pool in TFLite.

# Attention pool: 12 heads × 256 head_dim.
# Single learned query (arg 24, shape [1, 768]) attends over all positions.
pool_Q = expand(arg24) @ arg19 + arg18   # query
pool_K = x          @ arg14 + arg13       # key
pool_V = x          @ arg21 + arg20       # value
pool_attn = softmax(pool_Q @ pool_K^T * 1/sqrt(256) + pad_mask) @ pool_V
pool_out = pool_attn @ arg17 + arg16   # out.kernel, out.bias

# Final LN at the END of embed_text (scale = arg23 + 1, bias = arg22).
contrastive_embed = LayerNorm(pool_out, scale=arg23 + 1, bias=arg22)

# L2-normalized output is just contrastive_embed / norm(contrastive_embed).
```

## Verification results

`verify_musiccoca_prompts.py` against testdata's 26 reference prompts +
their embeddings:
- **Mean cosine: 0.9993**
- Min cos: 0.9965 ("Q-pop", an OOV word)
- Max cos: 0.9998

`verify_musiccoca_tokens.py` end-to-end (text → embed → RVQ tokens):
- **93.3% token match** (291/312)
- 8 of 26 prompts bit-exact (Viking, World, ...)

### Critical: RVQ codebook ordering

The 12 codebooks are stored in `musiccoca_quant.variables.{0..11}` but
in **ALPHABETICAL string order** of the integer suffix:
`["0", "1", "10", "11", "2", "3", ..., "9"]`. Remap to numerical order
before quantization, otherwise only first 2 codebooks align (16.7%
match instead of 93.3%).

## Remaining gap to bit-exact

The 7% token mismatch is downstream RVQ residual-drift from the 0.999
cosine embedding precision. Once one codebook misses by 1, subsequent
residuals diverge slightly. Possible micro-optimizations:
- Per-LN scale_offset tuning (some LNs might use 0 instead of 1).
- Slight magnitude correction (norm ratio is 0.997-0.998, not 1.000).

But functionally, 93% RVQ token match means the LLM downstream gets
near-identical style conditioning. Mission accomplished.

## Paths forward (any future session)

Now that the architecture is known, three concrete paths:

1. **Implement text encoder + verify against test embeddings** (~1 week):
   write a standard 12-layer Transformer with the per-layer fused params,
   plus the attention pooling head + quantizer. Verify by checking the 26
   testdata prompts' final 768d embeddings match within tolerance. The
   `tf_var_leaves.N` ordering can be guessed by matching shapes to layer
   positions; mistakes show up as obviously wrong activations.

2. **Convert SavedModel → TFLite locally** (Python ≤3.12 + tensorflow-text),
   then read the flatbuffer to get the exact node-level graph. Cleanest.

3. **Use Magenta-RT's MockMusicCoCa-style "lookup table" approach**: embed
   each of the 26 test prompts directly via the testdata file. Unblocks
   end-to-end demos for fixed prompts (e.g., "Ambient", "Jazz", "Pop")
   while we wait for the real text encoder.
