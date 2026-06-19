#!/usr/bin/env python3
"""Fetch the Magenta-RT v1 LLM checkpoint manifest (tensor names + shapes).

Lists every `target.*` weight in the T5X tensorstore checkpoint
`google/magenta-realtime :: checkpoints/llm_base_x4286_c1860k` and reads each
tensor's `.zarray` metadata for its shape/dtype. Writes `llm_base_manifest.json`.

This is the JSON-only manifest (no weights) the completion plan calls for to
settle the LLM architecture; it needs only HuggingFace metadata access, not the
multi-GB weight download. CC-BY-4.0, ungated.
"""
import json, re, urllib.request

REPO = "google/magenta-realtime"
CKPT = "checkpoints/llm_base_x4286_c1860k"
API = f"https://huggingface.co/api/models/{REPO}/tree/main/{CKPT}"
RAW = f"https://huggingface.co/{REPO}/raw/main/{CKPT}"


def _get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "meganeura"}), timeout=60)


def list_tensors():
    paths, cursor = [], None
    for _ in range(100):
        r = _get(API + (f"?cursor={cursor}" if cursor else ""))
        data = json.load(r)
        if not data:
            break
        paths += [x["path"].split(CKPT + "/", 1)[1] for x in data]
        link = r.headers.get("Link", "")
        m = re.search(r"cursor=([^&>]+)", link)
        if 'rel="next"' in link and m:
            cursor = m.group(1)
        else:
            break
    return sorted(p for p in paths if p.startswith("target."))


def zarray(name):
    z = json.load(_get(f"{RAW}/{name}/.zarray"))
    return z["shape"], z["dtype"]


def norm(s):
    return re.sub(r"layers_\d+", "layers_N", re.sub(r"depth_layers_\d+", "depth_layers_N", s))


def main():
    names = list_tensors()
    # One .zarray fetch per unique (layer-collapsed) pattern, then expand.
    shape_by_pattern = {}
    for n in names:
        p = norm(n)
        if p not in shape_by_pattern:
            shape_by_pattern[p] = zarray(n)
    manifest = {n: {"shape": shape_by_pattern[norm(n)][0], "dtype": shape_by_pattern[norm(n)][1]} for n in names}
    out = {
        "repo": REPO,
        "checkpoint": CKPT,
        "format": "t5x/flaxformer tensorstore (zarr)",
        "num_tensors": len(manifest),
        "tensors": manifest,
    }
    with open("tools/magenta_rt/llm_base_manifest.json", "w") as f:
        json.dump(out, f, indent=1, sort_keys=True)
    print(f"wrote {len(manifest)} tensors")


if __name__ == "__main__":
    main()
