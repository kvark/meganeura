"""Verify NumPy text encoder against the 26 reference prompts + their embeddings."""
import os, sys, struct, json, re
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, "/x/Code/meganeura/tools/magenta_rt")
import numpy as np
import sentencepiece as spm
from musiccoca_numpy_ref import load_args, embed_text_numpy

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


sp = spm.SentencePieceProcessor()
sp.Load(VOCAB)
print(f"SentencePiece loaded: vocab_size={sp.GetPieceSize()}")

with open(INPUTS) as f:
    prompts = [line.strip() for line in f if line.strip()]

ref = load_st(TESTDATA)
ref_emb = ref["embeddings"]  # [26, 768]
print(f"Reference: {ref_emb.shape}")

args = load_args()

# For each prompt, tokenize, run encoder, compare.
print()
cos_list = []
for i, prompt in enumerate(prompts):
    # Lower + tokenize.
    labels = sp.EncodeAsIds(prompt.lower())
    # SOS at start, pad to 128.
    max_len = 128
    sos_id = 1
    n = min(len(labels), max_len - 1)
    ids_arr = [sos_id] + labels[:n]
    ids_arr += [0] * (max_len - len(ids_arr))
    ids = np.array(ids_arr, dtype=np.int32)[None]  # [1, 128]
    paddings = np.ones((1, max_len), dtype=np.float32)
    paddings[0, :n + 1] = 0.0

    ours = embed_text_numpy(ids, paddings, args)[0]
    truth = ref_emb[i]
    cos = np.dot(ours, truth) / (np.linalg.norm(ours) * np.linalg.norm(truth) + 1e-9)
    cos_list.append(cos)
    print(f"  [{i:2d}] {prompt[:20]:<20s}  tokens={labels[:5]}{'...' if len(labels)>5 else ''}  cos={cos:.4f}")

print(f"\nMean cosine vs testdata embeddings: {np.mean(cos_list):.4f}")
print(f"Min cosine: {min(cos_list):.4f}")
print(f"Max cosine: {max(cos_list):.4f}")
