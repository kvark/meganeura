"""Verify RVQ tokenization: text → embed → 12 RVQ tokens matches reference."""
import os, sys, struct, json, re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, "/x/Code/meganeura/tools/magenta_rt")
import numpy as np
import sentencepiece as spm
from musiccoca_numpy_ref import load_args, embed_text_numpy

WTS = "/x/Code/meganeura/magenta_rt_codec_dump/weights_musiccoca.safetensors"
VOCAB = "/x/Code/meganeura/magenta_rt_codec_dump/musiccoca_vocab.model"
TESTDATA = "/x/Code/meganeura/magenta_rt_codec_dump/musiccoca_testdata.safetensors"
INPUTS = "/x/Code/meganeura/magenta_rt_codec_dump/musiccoca_inputs.txt"


def load_st(p):
    with open(p, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        h = json.loads(f.read(n))
        raw = f.read()
    out = {}
    dtypes = {'F32': np.float32, 'I32': np.int32}
    for k, info in h.items():
        if k.startswith('__'): continue
        if info['dtype'] not in dtypes: continue
        s, e = info['data_offsets']
        sh = info['shape'] if info['shape'] else [1]
        out[k] = np.frombuffer(raw[s:e], dtype=dtypes[info['dtype']]).reshape(sh).copy()
    return out


# Load codebooks: 12 of [768, 1024]
codebooks_raw = load_st(WTS)
codebooks = []
for k, v in sorted(codebooks_raw.items()):
    if "musiccoca_quant.variables." in k and "ATTRIBUTES.VARIABLE_VALUE" in k:
        m = re.search(r"variables\.(\d+)\.", k)
        if m and v.shape == (768, 1024):
            codebooks.append((int(m.group(1)), v))
codebooks.sort()
# DEBUG: print indices
print(f"DEBUG: codebook indices order: {[c[0] for c in codebooks]}")
codebooks = [c[1] for c in codebooks]
print(f"Loaded {len(codebooks)} codebooks of shape {codebooks[0].shape}")


def rvq_quantize(embed, codebooks):
    """RVQ: at each level, find nearest centroid, output index, subtract residual."""
    residual = embed.copy()
    tokens = []
    for cb in codebooks:
        # cb [768, 1024]: 1024 centroids of 768 dim. Index by column.
        # Distance: residual @ cb gives [1024] dot products (proxy if normalized).
        # Use Euclidean distance instead.
        dists = np.linalg.norm(residual[:, None] - cb, axis=0)  # [1024]
        idx = int(dists.argmin())
        tokens.append(idx)
        residual = residual - cb[:, idx]
    return tokens


sp = spm.SentencePieceProcessor()
sp.Load(VOCAB)
with open(INPUTS) as f:
    prompts = [l.strip() for l in f if l.strip()]

ref = load_st(TESTDATA)
ref_tokens = ref["tokens"]   # [26, 12]
print(f"Reference tokens: {ref_tokens.shape}")

args = load_args()
match_count = 0
total = 0
print()
for i, prompt in enumerate(prompts):
    labels = sp.EncodeAsIds(prompt.lower())
    ids_arr = [1] + labels
    ids_arr += [0] * (128 - len(ids_arr))
    ids = np.array(ids_arr, dtype=np.int32)[None]
    paddings = np.ones((1, 128), dtype=np.float32)
    paddings[0, :len(labels) + 1] = 0.0

    ours_embed = embed_text_numpy(ids, paddings, args)[0]
    # Remap codebooks: safetensors stored them in alphabetical order
    # ["0","1","10","11","2",...,"9"], not numerical. Convert to numerical.
    alpha_order = [0, 1, 10, 11, 2, 3, 4, 5, 6, 7, 8, 9]
    numerical_cbs = [None] * 12
    for safetensors_idx, num_level in enumerate(alpha_order):
        numerical_cbs[num_level] = codebooks[safetensors_idx]
    ours_tokens = rvq_quantize(ours_embed, numerical_cbs)
    ref_toks = ref_tokens[i].tolist()
    n_match = sum(1 for a, b in zip(ours_tokens, ref_toks) if a == b)
    match_count += n_match
    total += len(ref_toks)
    status = "✓" if n_match == len(ref_toks) else f"{n_match}/{len(ref_toks)}"
    print(f"  [{i:2d}] {prompt[:15]:<15s}  ref={ref_toks}  ours={ours_tokens}  {status}")
print(f"\nToken match: {match_count}/{total}  ({100*match_count/total:.1f}%)")
