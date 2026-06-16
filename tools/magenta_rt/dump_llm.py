"""Dump Magenta-RT LLM (T5X base checkpoint) target.* parameters to safetensors.

The HF checkpoint uses tensorstore/zarr format where each parameter is a
directory with .zarray (metadata) + one or more chunk files. For T5X
inference checkpoints, chunks are typically un-sharded (just "0" or "0.0").
"""
import os, json, struct
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
from pathlib import Path
import numpy as np
from huggingface_hub import snapshot_download
from safetensors.numpy import save_file as save_safetensors

DUMP = Path("magenta_rt_codec_dump")
DUMP.mkdir(exist_ok=True)


def read_zarr_param(path: Path):
    """Read a single zarr-format param from `path/` (containing .zarray + chunks)."""
    z_file = path / ".zarray"
    meta = json.loads(z_file.read_text())
    shape = tuple(meta["shape"])
    dtype_str = meta["dtype"]
    chunks = tuple(meta["chunks"])

    # Map zarr dtype to numpy.
    np_dtype = np.dtype(dtype_str)

    # Allocate output.
    out = np.zeros(shape, dtype=np_dtype)

    # Enumerate chunk indices.
    n_chunks_per_axis = []
    for s, c in zip(shape, chunks):
        n_chunks_per_axis.append((s + c - 1) // c)

    import itertools
    for chunk_idx in itertools.product(*[range(n) for n in n_chunks_per_axis]):
        chunk_name = ".".join(str(i) for i in chunk_idx)
        chunk_file = path / chunk_name
        if not chunk_file.exists():
            raise FileNotFoundError(f"missing chunk: {chunk_file}")
        chunk_bytes = chunk_file.read_bytes()
        # Each chunk is raw f32 bytes (no compression for T5X inference checkpoints).
        # Decompress if needed (T5X uses gzip sometimes).
        # Try raw first; if size doesn't match, try gzip.
        chunk_size = int(np.prod(chunks)) * np_dtype.itemsize
        if len(chunk_bytes) == chunk_size:
            chunk_arr = np.frombuffer(chunk_bytes, dtype=np_dtype).reshape(chunks)
        else:
            # Try gzip decompression.
            import gzip
            chunk_arr = np.frombuffer(gzip.decompress(chunk_bytes), dtype=np_dtype).reshape(chunks)
        # Place into output array.
        slices = tuple(
            slice(ci * cs, min((ci + 1) * cs, s))
            for ci, cs, s in zip(chunk_idx, chunks, shape)
        )
        # Crop chunk_arr if it overflows.
        chunk_slices = tuple(
            slice(0, slc.stop - slc.start)
            for slc in slices
        )
        out[slices] = chunk_arr[chunk_slices]

    return out


def main():
    print("Snapshotting Magenta-RT LLM base checkpoint (target.* only)...")
    root = Path(snapshot_download(
        "google/magenta-realtime",
        allow_patterns=[
            "checkpoints/llm_base_x4286_c1860k/target.*",
        ],
    ))
    print(f"  root: {root}")

    ckpt = root / "checkpoints" / "llm_base_x4286_c1860k"
    print(f"  ckpt: {ckpt}")

    # Walk all subdirectories under ckpt to find .zarray files.
    print("Scanning param paths...")
    param_paths = []
    for zfile in ckpt.rglob(".zarray"):
        param_paths.append(zfile.parent)
    print(f"  found {len(param_paths)} parameters")

    # Read each param.
    weights = {}
    total_bytes = 0
    for i, ppath in enumerate(sorted(param_paths)):
        rel = ppath.relative_to(ckpt)
        # Convert path components to dotted name. The .v suffix is for "value"
        # (vs other slots), drop it.
        name = ".".join(rel.parts)
        if name.endswith(".v"):
            name = name[:-2]
        arr = read_zarr_param(ppath)
        weights[name] = np.ascontiguousarray(arr.astype(np.float32 if arr.dtype != np.int32 else np.int32))
        total_bytes += arr.nbytes
        if i < 10 or i > len(param_paths) - 5:
            print(f"  [{i:4d}] {name[:80]:<80s}  shape={list(arr.shape)}  dtype={arr.dtype}")
        elif i == 10:
            print(f"  ... (suppressing {len(param_paths) - 14} middle entries) ...")

    print(f"\nTotal: {len(weights)} tensors, {total_bytes / 1024 / 1024:.1f} MB")
    out_st = DUMP / "weights_llm_base.safetensors"
    save_safetensors(weights, str(out_st))
    print(f"Saved to {out_st}")

    # Manifest.
    manifest = {
        "checkpoint": "llm_base_x4286_c1860k",
        "tensors": [
            {"name": k, "shape": list(v.shape), "dtype": str(v.dtype)}
            for k, v in sorted(weights.items())
        ],
    }
    (DUMP / "llm_base_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Saved manifest to {DUMP / 'llm_base_manifest.json'}")


if __name__ == "__main__":
    main()
