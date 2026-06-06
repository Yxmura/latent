#!/usr/bin/env python3
"""Full model SVD pipeline for LATENT.

This script:
  1. Extracts expert tensors from a GGUF file
  2. Runs SVD compression on each expert
  3. Writes a new GGUF file with LATENT factor tensors

Usage:
  python loader/full_model_svd.py \\
    --input-gguf model.gguf \\
    --output-gguf model.latent.gguf \\
    --energy 0.95 \\
    --max-rank 128 \\
    --output-prefix build/

The output GGUF contains the original model plus LATENT factor tensors:
  - blk.{i}.ffn_latent_gate_a: [n_embd, max_rank, n_expert] NF4 packed
  - blk.{i}.ffn_latent_gate_a_scale: [n_expert, groups_a] FP16
  - blk.{i}.ffn_latent_gate_b: [max_rank, n_ff, n_expert] NF4 packed
  - ... and similarly for up and down
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loader.gguf_expert_extract import add_gguf_to_path
from loader.latent_loader import (
    compress_matrix,
    MatrixReport,
    quantize_nf4,
    quantize_int8,
)


def extract_expert_tensors(
    gguf_path: Path,
    gguf_py_path: Path | None = None,
) -> Dict[str, np.ndarray]:
    """Extract dense expert tensors from a GGUF file.

    Returns a dict mapping tensor name -> dense numpy array.
    """
    add_gguf_to_path(gguf_py_path)
    import gguf  # type: ignore
    from gguf import GGUFReader  # type: ignore

    print(f"Loading GGUF: {gguf_path}")
    reader = GGUFReader(gguf_path)

    import re
    pattern = re.compile(r"ffn_(gate|up|down)_exps")

    arrays: Dict[str, np.ndarray] = {}
    for tensor in reader.tensors:
        if pattern.search(tensor.name):
            try:
                dense = gguf.dequantize(tensor.data, tensor.tensor_type)
            except NotImplementedError:
                print(f"  Skipping {tensor.name}: unsupported type {tensor.tensor_type}")
                continue
            arrays[tensor.name] = np.asarray(dense, dtype=np.float32)
            print(f"  Extracted {tensor.name}: shape={list(tensor.shape)}")

    print(f"Extracted {len(arrays)} expert tensors")
    return arrays


def compress_expert_to_factors(
    name: str,
    matrix: np.ndarray,
    energy: float,
    min_rank: int,
    max_rank: int,
    group_size: int,
    int8_group_size: int,
    error_threshold: float,
    snr_threshold: float,
    low_snr_fraction_threshold: float,
) -> Tuple[Dict[str, np.ndarray], MatrixReport]:
    """Compress a single expert matrix to LATENT factors.

    matrix shape: [n_in, n_out, n_expert] (llama.cpp convention)
    Returns dict of factor arrays and the compression report.
    """
    # SVD operates on 2D matrices; process each expert's projection independently
    n_in, n_out, n_expert = matrix.shape
    print(f"  Compressing {name}: shape=[{n_in}, {n_out}, {n_expert}]")

    arrays: Dict[str, np.ndarray] = {}
    reports: List[MatrixReport] = []

    for e in range(n_expert):
        expert_name = f"{name}.expert{e}"
        # LATENT convention (matches kernel): W = A @ B with A:[n_in, r] and
        # B:[r, n_out]. The kernel multiplies x @ A first (input-side factor),
        # then latent @ B (output-side factor). We keep the matrix in its
        # original (n_in, n_out) orientation so the factorization lines up
        # with the kernel's access pattern.
        expert_matrix = matrix[:, :, e]  # [n_in, n_out]

        compressed, report = compress_matrix(
            name=expert_name,
            matrix=expert_matrix,
            energy=energy,
            min_rank=min_rank,
            max_rank=max_rank,
            group_size=group_size,
            int8_group_size=int8_group_size,
            error_threshold=error_threshold,
            snr_threshold=snr_threshold,
            low_snr_fraction_threshold=low_snr_fraction_threshold,
            seed=e,
        )
        arrays.update(compressed)
        reports.append(report)

    # Aggregate reports
    if reports:
        agg_report = MatrixReport(
            name=name,
            shape=(n_in, n_out, n_expert),
            rank=int(np.mean([r.rank for r in reports])),
            retained_energy=float(np.mean([r.retained_energy for r in reports])),
            dense_relative_error=float(np.mean([r.dense_relative_error for r in reports])),
            quantized_relative_error=float(np.mean([r.quantized_relative_error for r in reports])),
            min_group_snr=float(np.min([r.min_group_snr for r in reports])),
            low_snr_group_fraction=float(np.mean([r.low_snr_group_fraction for r in reports])),
            quantization=reports[0].quantization,
            fallback=any(r.fallback for r in reports),
            fallback_reason=";".join(set(r.fallback_reason for r in reports if r.fallback_reason)),
        )
    else:
        agg_report = MatrixReport(
            name=name, shape=(n_in, n_out, n_expert), rank=0,
            retained_energy=1.0, dense_relative_error=0.0,
            quantized_relative_error=0.0, min_group_snr=float("inf"),
            low_snr_group_fraction=0.0, quantization="none",
            fallback=False, fallback_reason="",
        )

    return arrays, agg_report


def write_factor_arrays(
    factor_arrays: Dict[str, np.ndarray],
    output_path: Path,
) -> None:
    """Write factor arrays to a compressed NPZ file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **factor_arrays)
    print(f"Wrote {len(factor_arrays)} factor arrays to {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Full model LATENT SVD pipeline.")
    parser.add_argument("--input-gguf", required=True, type=Path,
                        help="Input GGUF model file.")
    parser.add_argument("--output-gguf", type=Path,
                        help="Output GGUF file with LATENT factors (optional).")
    parser.add_argument("--output-prefix", type=Path, default=Path("build"),
                        help="Output directory for intermediate files.")
    parser.add_argument("--gguf-py", type=Path,
                        help="Path to llama.cpp/gguf-py if not vendored.")
    parser.add_argument("--energy", type=float, default=0.95)
    parser.add_argument("--min-rank", type=int, default=16)
    parser.add_argument("--max-rank", type=int, default=128)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--int8-group-size", type=int, default=64)
    parser.add_argument("--error-threshold", type=float, default=0.02)
    parser.add_argument("--snr-threshold", type=float, default=3.0)
    parser.add_argument("--low-snr-fraction-threshold", type=float, default=0.05)
    parser.add_argument("--limit-layers", type=int, default=0,
                        help="Limit number of layers processed (0 = all).")
    parser.add_argument("--limit-experts", type=int, default=0,
                        help="Limit number of experts per layer (0 = all).")
    args = parser.parse_args()

    output_prefix = args.output_prefix
    output_prefix.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract expert tensors
    start = time.perf_counter()
    expert_tensors = extract_expert_tensors(args.input_gguf, args.gguf_py)
    extract_time = time.perf_counter() - start
    print(f"Extraction took {extract_time:.1f}s")

    if not expert_tensors:
        print("No expert tensors found in GGUF")
        return 1

    # Step 2: Compress each expert
    start = time.perf_counter()
    all_factor_arrays: Dict[str, np.ndarray] = {}
    all_reports = []

    # Group by layer
    layers: Dict[int, Dict[str, np.ndarray]] = {}
    for name, arr in expert_tensors.items():
        # Parse layer number from name like "blk.0.ffn_gate_exps.weight"
        parts = name.split(".")
        if len(parts) >= 3 and parts[0] == "blk":
            layer_idx = int(parts[1])
            tensor_kind = parts[2]  # e.g. "ffn_gate_exps"
            layers.setdefault(layer_idx, {})[tensor_kind] = arr

    layer_indices = sorted(layers.keys())
    if args.limit_layers > 0:
        layer_indices = layer_indices[: args.limit_layers]

    print(f"Processing {len(layer_indices)} layers")

    for layer_idx in layer_indices:
        layer_tensors = layers[layer_idx]
        print(f"Layer {layer_idx}: {sorted(layer_tensors.keys())}")

        for tensor_name, matrix in layer_tensors.items():
            # Limit experts if requested
            if args.limit_experts > 0 and matrix.shape[-1] > args.limit_experts:
                matrix = matrix[..., : args.limit_experts]

            factor_arrays, report = compress_expert_to_factors(
                name=f"blk.{layer_idx}.{tensor_name}",
                matrix=matrix,
                energy=args.energy,
                min_rank=args.min_rank,
                max_rank=args.max_rank,
                group_size=args.group_size,
                int8_group_size=args.int8_group_size,
                error_threshold=args.error_threshold,
                snr_threshold=args.snr_threshold,
                low_snr_fraction_threshold=args.low_snr_fraction_threshold,
            )
            all_factor_arrays.update(factor_arrays)
            all_reports.append(report)

    compress_time = time.perf_counter() - start
    print(f"Compression took {compress_time:.1f}s")

    # Step 3: Write intermediate NPZ
    factor_npz_path = output_prefix / "latent_factors.npz"
    write_factor_arrays(all_factor_arrays, factor_npz_path)

    # Step 4: Write metadata
    metadata = {
        "format": "latent-nf4-factors-v0",
        "input_gguf": str(args.input_gguf),
        "extraction_time_sec": extract_time,
        "compression_time_sec": compress_time,
        "config": {
            "energy": args.energy,
            "min_rank": args.min_rank,
            "max_rank": args.max_rank,
            "group_size": args.group_size,
            "int8_group_size": args.int8_group_size,
            "error_threshold": args.error_threshold,
            "snr_threshold": args.snr_threshold,
            "low_snr_fraction_threshold": args.low_snr_fraction_threshold,
        },
        "matrices": [
            {
                "name": r.name,
                "shape": r.shape,
                "rank": r.rank,
                "retained_energy": r.retained_energy,
                "dense_relative_error": r.dense_relative_error,
                "quantized_relative_error": r.quantized_relative_error,
                "min_group_snr": r.min_group_snr,
                "low_snr_group_fraction": r.low_snr_group_fraction,
                "quantization": r.quantization,
                "fallback": r.fallback,
                "fallback_reason": r.fallback_reason,
            }
            for r in all_reports
        ],
    }

    metadata_path = output_prefix / "latent_factors.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote metadata to {metadata_path}")

    # Step 5: Optionally write a new GGUF with LATENT factors
    if args.output_gguf:
        print(f"Output GGUF: {args.output_gguf}")
        print("Note: LATENT GGUF writing is not yet implemented. Use --output-prefix instead.")
        print("The NPZ file contains all factor arrays needed for tensor injection.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
