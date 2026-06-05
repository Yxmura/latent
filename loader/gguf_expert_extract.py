#!/usr/bin/env python3
"""Inspect or extract expert tensors from a GGUF file for LATENT compression."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


def add_gguf_to_path(path: Path | None) -> None:
    if path is not None:
        sys.path.insert(0, str(path))
        return
    candidate = Path(__file__).resolve().parents[1] / "vendor" / "llama.cpp-v4" / "gguf-py"
    if candidate.exists():
        sys.path.insert(0, str(candidate))


def expert_tensors(tensors: Iterable[object], pattern: re.Pattern[str]):
    for tensor in tensors:
        name = getattr(tensor, "name")
        if pattern.search(name):
            yield tensor


def tensor_to_dense(tensor: object, gguf_module: object) -> np.ndarray:
    tensor_type = getattr(tensor, "tensor_type")
    data = getattr(tensor, "data")
    try:
        dense = gguf_module.dequantize(data, tensor_type)
    except NotImplementedError as exc:
        raise TypeError(
            f"{getattr(tensor, 'name')} has unsupported type {tensor_type}; "
            "gguf-py has no dequantizer for this tensor type"
        ) from exc
    return np.asarray(dense, dtype=np.float32)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gguf", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="Write dense selected tensors to this NPZ.")
    parser.add_argument("--gguf-py", type=Path, help="Path to llama.cpp/gguf-py if not vendored.")
    parser.add_argument(
        "--pattern",
        default=r"ffn_(gate|up|down).*exps|ffn_(gate|up|down)_exps",
        help="Regex used to identify expert tensors.",
    )
    parser.add_argument("--list-only", action="store_true")
    args = parser.parse_args()

    add_gguf_to_path(args.gguf_py)
    import gguf  # type: ignore
    from gguf import GGUFReader  # type: ignore

    reader = GGUFReader(args.gguf)
    pattern = re.compile(args.pattern)
    selected = list(expert_tensors(reader.tensors, pattern))
    summary = [
        {
            "name": tensor.name,
            "shape": [int(x) for x in tensor.shape.tolist()],
            "type": str(tensor.tensor_type),
            "n_elements": int(tensor.n_elements),
            "n_bytes": int(tensor.n_bytes),
        }
        for tensor in selected
    ]
    print(json.dumps({"tensor_count": len(summary), "tensors": summary}, indent=2))

    if args.list_only:
        return 0
    if not args.output:
        raise SystemExit("--output is required unless --list-only is set")

    arrays = {}
    for tensor in selected:
        arrays[tensor.name] = tensor_to_dense(tensor, gguf)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **arrays)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
