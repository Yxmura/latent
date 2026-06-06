"""Inspect GGUF metadata from a partial download (no tensor data needed).

Reads only the GGUF header (general.* fields, tensor info, etc.) and
prints it. Does not require the full file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def read_gguf_metadata(path: Path) -> dict:
    """Read just the GGUF header (no tensor data).

    GGUF format:
      - magic: 'GGUF' (4 bytes)
      - version: u32
      - tensor_count: u64
      - metadata_kv_count: u64
      - metadata_kv[]: each is key (string), type (u32 enum), value
      - tensor_infos[]: name (string), n_dims (u32), dims (u64[]), type (u32), offset (u64)
      - padding to alignment
      - tensor data
    """
    with open(path, "rb") as f:
        data = f.read()

    if data[:4] != b"GGUF":
        raise ValueError(f"Not a GGUF file (got {data[:4]!r})")

    version = np.frombuffer(data[4:8], dtype=np.uint32)[0]
    tensor_count = np.frombuffer(data[8:16], dtype=np.uint64)[0]
    kv_count = np.frombuffer(data[16:24], dtype=np.uint64)[0]

    pos = 24
    metadata = {}
    for _ in range(int(kv_count)):
        # key: length-prefixed string
        key_len = np.frombuffer(data[pos:pos+8], dtype=np.uint64)[0]
        pos += 8
        key = data[pos:pos+int(key_len)].decode("utf-8")
        pos += int(key_len)
        # type
        vtype = np.frombuffer(data[pos:pos+4], dtype=np.uint32)[0]
        pos += 4
        # value
        val, pos = _read_kv_value(data, pos, int(vtype))
        metadata[key] = val

    # Tensor infos (just names, shapes, types, offsets)
    tensors = []
    for _ in range(int(tensor_count)):
        name_len = np.frombuffer(data[pos:pos+8], dtype=np.uint64)[0]
        pos += 8
        name = data[pos:pos+int(name_len)].decode("utf-8")
        pos += int(name_len)
        n_dims = np.frombuffer(data[pos:pos+4], dtype=np.uint32)[0]
        pos += 4
        dims = np.frombuffer(data[pos:pos+8*int(n_dims)], dtype=np.uint64).tolist()
        pos += 8 * int(n_dims)
        ttype = np.frombuffer(data[pos:pos+4], dtype=np.uint32)[0]
        pos += 4
        offset = np.frombuffer(data[pos:pos+8], dtype=np.uint64)[0]
        pos += 8
        tensors.append({
            "name": name,
            "shape": list(dims),
            "type": int(ttype),
            "offset": int(offset),
        })

    return {
        "version": int(version),
        "tensor_count": int(tensor_count),
        "kv_count": int(kv_count),
        "metadata": metadata,
        "tensors": tensors,
    }


def _read_kv_value(data: bytes, pos: int, vtype: int):
    """Decode a single KV value at pos, return (value, new_pos)."""
    # gguf-py 0.19.0 GGUFValueType enum:
    # 0=UINT8, 1=INT8, 2=UINT16, 3=INT16, 4=UINT32, 5=INT32,
    # 6=FLOAT32, 7=BOOL, 8=STRING, 9=ARRAY, 10=UINT64, 11=INT64, 12=FLOAT64
    if vtype == 0:
        return int(data[pos]), pos + 1
    if vtype == 1:
        return int(np.frombuffer(data[pos:pos+1], dtype=np.int8)[0]), pos + 1
    if vtype == 2:
        return int(np.frombuffer(data[pos:pos+2], dtype=np.uint16)[0]), pos + 2
    if vtype == 3:
        return int(np.frombuffer(data[pos:pos+2], dtype=np.int16)[0]), pos + 2
    if vtype == 4:
        return int(np.frombuffer(data[pos:pos+4], dtype=np.uint32)[0]), pos + 4
    if vtype == 5:
        return int(np.frombuffer(data[pos:pos+4], dtype=np.int32)[0]), pos + 4
    if vtype == 6:  # FLOAT32
        return float(np.frombuffer(data[pos:pos+4], dtype=np.float32)[0]), pos + 4
    if vtype == 7:  # BOOL
        return bool(data[pos]), pos + 1
    if vtype == 8:  # STRING
        slen = np.frombuffer(data[pos:pos+8], dtype=np.uint64)[0]
        pos += 8
        s = data[pos:pos+int(slen)].decode("utf-8", errors="replace")
        return s, pos + int(slen)
    if vtype == 9:  # ARRAY
        atype = np.frombuffer(data[pos:pos+4], dtype=np.uint32)[0]
        pos += 4
        alen = np.frombuffer(data[pos:pos+8], dtype=np.uint64)[0]
        pos += 8
        items = []
        for _ in range(int(alen)):
            item, pos = _read_kv_value(data, pos, int(atype))
            items.append(item)
        return items, pos
    if vtype == 10:
        return int(np.frombuffer(data[pos:pos+8], dtype=np.uint64)[0]), pos + 8
    if vtype == 11:
        return int(np.frombuffer(data[pos:pos+8], dtype=np.int64)[0]), pos + 8
    if vtype == 12:  # FLOAT64
        return float(np.frombuffer(data[pos:pos+8], dtype=np.float64)[0]), pos + 8
    raise ValueError(f"Unknown GGUF value type: {vtype}")


def summarize(info: dict) -> str:
    out = []
    out.append(f"GGUF version: {info['version']}")
    out.append(f"Tensor count: {info['tensor_count']}")
    out.append(f"Metadata KV count: {info['kv_count']}")
    out.append("")
    out.append("=== General metadata ===")
    for key in sorted(info["metadata"]):
        if "." in key and key.split(".")[0] in {"tokenizer", "split"}:
            continue
        out.append(f"  {key} = {info['metadata'][key]!r}")
    out.append("")
    out.append("=== Tensor summary (grouped by name pattern) ===")
    patterns: dict[str, list[dict]] = {}
    for t in info["tensors"]:
        # Strip blk.N. prefix
        n = t["name"]
        if n.startswith("blk."):
            parts = n.split(".", 2)
            if len(parts) >= 3:
                pattern = "%s.%s" % (parts[0], parts[2].rsplit(".", 1)[0] if parts[2].count(".") > 0 else parts[2])
            else:
                pattern = n
        else:
            pattern = n
        patterns.setdefault(pattern, []).append(t)
    for pat, ts in sorted(patterns.items()):
        shapes = [tuple(t["shape"]) for t in ts]
        types = set(t["type"] for t in ts)
        out.append(f"  {pat}: {len(ts)} tensors, shapes={shapes[:3]}{'...' if len(shapes) > 3 else ''}, types={types}")
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a partial GGUF download.")
    parser.add_argument("gguf", type=Path)
    parser.add_argument("--latent-tensors-only", action="store_true",
                        help="Show only tensors matching LATENT patterns")
    args = parser.parse_args()

    info = read_gguf_metadata(args.gguf)
    print(summarize(info))

    if args.latent_tensors_only:
        print()
        print("=== LATENT-related tensors ===")
        latent_patterns = ["ffn_latent", "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps", "ffn_gate_inp"]
        for t in info["tensors"]:
            if any(p in t["name"] for p in latent_patterns):
                print(f"  {t['name']}: shape={t['shape']} type={t['type']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
