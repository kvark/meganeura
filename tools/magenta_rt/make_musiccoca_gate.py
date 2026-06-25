#!/usr/bin/env python3
"""Build the MusicCoCa real-weight gate bundle for `tests/musiccoca_real_weight.rs`.

Produces `musiccoca_gate.safetensors` (all f32) from the dumped weights +
testdata:
  - ids        [26, maxlen]  SentencePiece token ids per prompt (SOS=1 prefix,
                             lowercased), zero-padded; the encoder reads
                             `lens[i]` valid tokens.
  - lens       [26]          valid token count per prompt.
  - ref_embeddings [26, 768] reference contrastive embeddings (testdata).
  - ref_tokens [26, 12]      reference RVQ style tokens (testdata).
  - codebooks  [12, 768, 1024] RVQ codebooks in **RVQ-level order**
                             (`numeric_codebook_order` = [0,1,4,5,6,7,8,9,10,11,2,3]).

The tokenization was verified to reproduce the reference embeddings at cosine
1.0000 via the `embed_text` SavedModel oracle; the level-ordered codebooks
reproduce the reference tokens at 100% (the raw numeric order gives ~17%).

Run after `dump_musiccoca.py`. Needs `sentencepiece` + the dumped
`weights_musiccoca.safetensors`, `musiccoca_testdata.safetensors`, and the
`vocabularies/musiccoca_mv212f_vocab.model` (auto-downloaded).
"""
import os
from pathlib import Path

import numpy as np
import sentencepiece as spm
from huggingface_hub import snapshot_download
from safetensors.numpy import load_file, save_file

DUMP = Path(os.environ.get("MAGENTA_RT_DUMP", "magenta_rt_codec_dump"))


def numeric_codebook_order(depth: int):
    """RVQ-level → checkpoint-variable index. The 12 quantizer variables are
    string-sorted in the checkpoint (`0,1,10,11,2,…`); level L's codebook is the
    L-th in that sorted order. Matches `musiccoca::numeric_codebook_order`."""
    keys = sorted(range(depth), key=str)
    order = [0] * depth
    for pos, level in enumerate(keys):
        order[level] = pos
    return order


def main():
    w = load_file(str(DUMP / "weights_musiccoca.safetensors"))
    td = load_file(str(DUMP / "musiccoca_testdata.safetensors"))
    ref_emb = td["embeddings"].astype(np.float32)  # [26, 768]
    ref_tok = td["tokens"].astype(np.float32)  # [26, 12]

    # Codebooks in RVQ-level order.
    order = numeric_codebook_order(12)
    var = [w[f"musiccoca_quant.variables.{i}..ATTRIBUTES.VARIABLE_VALUE"] for i in range(12)]
    codebooks = np.stack([var[order[L]] for L in range(12)]).astype(np.float32)  # [12,768,1024]

    # Tokenize the 26 prompts (lowercase + SOS=1 prefix).
    root = Path(
        snapshot_download(
            "google/magenta-realtime",
            allow_patterns=[
                "vocabularies/musiccoca_mv212f_vocab.model",
                "testdata/musiccoca_mv212/*.txt",
            ],
        )
    )
    prompts = [
        p
        for p in next((root / "testdata/musiccoca_mv212").glob("*.txt")).read_text().splitlines()
        if p.strip()
    ]
    sp = spm.SentencePieceProcessor(
        model_file=str(root / "vocabularies/musiccoca_mv212f_vocab.model")
    )
    ids_list = [[1] + sp.encode(p.lower(), out_type=int) for p in prompts]
    maxlen = max(len(x) for x in ids_list)
    ids = np.zeros((len(ids_list), maxlen), np.float32)
    lens = np.zeros((len(ids_list),), np.float32)
    for i, x in enumerate(ids_list):
        ids[i, : len(x)] = x
        lens[i] = len(x)

    out = DUMP / "musiccoca_gate.safetensors"
    save_file(
        {
            "ids": ids,
            "lens": lens,
            "ref_embeddings": ref_emb,
            "ref_tokens": ref_tok,
            "codebooks": codebooks,
        },
        str(out),
    )
    print(f"wrote {out} ({len(prompts)} prompts, maxlen={maxlen})")


if __name__ == "__main__":
    main()
