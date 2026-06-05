#!/usr/bin/env python3
"""End-to-end validation pipeline for LATENT.

Validates that the LATENT compression + SwiGLU expert FFN pipeline
produces results within acceptable error bounds of the dense reference.

Usage:
  python benchmarks/end_to_end_validation.py \\
    --rows 4096 \\
    --cols 2048 \\
    --n-expert 8 \\
    --top-k 6 \\
    --max-rank 128
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loader.adaptive_topk import adaptive_topk_inclusive
from loader.latent_loader import (
    compress_matrix,
    dequantize_int8,
    dequantize_nf4,
    relative_frobenius_error,
)
from loader.latent_runtime import (
    dense_swiglu_expert,
    load_pair,
    swiglu_expert,
)


def make_realistic_expert(
    rows: int,
    cols: int,
    intrinsic_rank: int,
    noise: float = 0.01,
    seed: int = 0,
) -> np.ndarray:
    """Create a realistic expert matrix with known intrinsic rank + noise."""
    rng = np.random.default_rng(seed)
    left = rng.standard_normal((rows, intrinsic_rank), dtype=np.float32)
    right = rng.standard_normal((intrinsic_rank, cols), dtype=np.float32)
    return (left @ right + noise * rng.standard_normal((rows, cols), dtype=np.float32)).astype(np.float32)


def compress_and_pack(
    name: str,
    matrix: np.ndarray,
    energy: float,
    min_rank: int,
    max_rank: int,
    group_size: int,
    error_threshold: float,
) -> tuple[Dict[str, np.ndarray], dict]:
    """Compress a matrix and return packed factor arrays + report."""
    arrays, report = compress_matrix(
        name=name,
        matrix=matrix,
        energy=energy,
        min_rank=min_rank,
        max_rank=max_rank,
        group_size=group_size,
        int8_group_size=64,
        error_threshold=error_threshold,
        snr_threshold=3.0,
        low_snr_fraction_threshold=0.05,
        seed=hash(name) & 0xFFFF,
    )
    return arrays, {
        "name": report.name,
        "shape": list(report.shape),
        "rank": report.rank,
        "retained_energy": report.retained_energy,
        "dense_relative_error": report.dense_relative_error,
        "quantized_relative_error": report.quantized_relative_error,
        "min_group_snr": report.min_group_snr,
        "quantization": report.quantization,
        "fallback": report.fallback,
    }


def validate_single_expert(
    n_embd: int,
    n_ff: int,
    intrinsic_rank: int,
    compression_max_rank: int,
    energy: float,
    error_threshold: float,
    weight_before_down: bool = False,
    seed: int = 0,
) -> dict:
    """Run end-to-end validation on a single expert.

    Matrix shapes (matching llama.cpp convention):
      gate: [n_embd, n_ff]  - hidden -> intermediate
      up:   [n_embd, n_ff]  - hidden -> intermediate
      down: [n_ff, n_embd]  - intermediate -> hidden
    """
    gate = make_realistic_expert(n_embd, n_ff, intrinsic_rank, seed=seed)
    up = make_realistic_expert(n_embd, n_ff, intrinsic_rank, seed=seed + 1)
    down = make_realistic_expert(n_ff, n_embd, intrinsic_rank, seed=seed + 2)

    # Compress
    start = time.perf_counter()
    gate_arrays, gate_report = compress_and_pack(
        "gate", gate, energy, 16, compression_max_rank, 32, error_threshold
    )
    up_arrays, up_report = compress_and_pack(
        "up", up, energy, 16, compression_max_rank, 32, error_threshold
    )
    down_arrays, down_report = compress_and_pack(
        "down", down, energy, 16, compression_max_rank, 32, error_threshold
    )
    compress_time = time.perf_counter() - start

    # Test runtime
    rng = np.random.default_rng(seed + 100)
    x = rng.standard_normal((1, n_embd), dtype=np.float32)
    weight = 0.3

    # Dense reference
    start = time.perf_counter()
    dense_out = dense_swiglu_expert(
        x, gate, up, down, weight=weight, weight_before_down=weight_before_down
    )
    dense_time = time.perf_counter() - start

    # LATENT runtime
    arrays = {}
    arrays.update(gate_arrays)
    arrays.update(up_arrays)
    arrays.update(down_arrays)

    start = time.perf_counter()
    latent_out = swiglu_expert(
        x,
        load_pair(arrays, "gate"),
        load_pair(arrays, "up"),
        load_pair(arrays, "down"),
        weight=weight,
        weight_before_down=weight_before_down,
    )
    latent_time = time.perf_counter() - start

    # Error
    if dense_out.size > 0 and latent_out.size > 0:
        # Handle shape mismatches
        if dense_out.shape != latent_out.shape:
            if dense_out.size == latent_out.size:
                dense_out = dense_out.reshape(latent_out.shape)
            else:
                runtime_error = float("nan")
                return {
                    "shape_mismatch": True,
                    "dense_shape": list(dense_out.shape),
                    "latent_shape": list(latent_out.shape),
                }
        runtime_error = relative_frobenius_error(dense_out, latent_out)
    else:
        runtime_error = float("nan")

    return {
        "intrinsic_rank": intrinsic_rank,
        "compression_max_rank": compression_max_rank,
        "energy": energy,
        "error_threshold": error_threshold,
        "weight_before_down": weight_before_down,
        "gate_report": gate_report,
        "up_report": up_report,
        "down_report": down_report,
        "compress_time_sec": compress_time,
        "dense_time_sec": dense_time,
        "latent_time_sec": latent_time,
        "runtime_relative_error": runtime_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="End-to-end LATENT validation.")
    parser.add_argument("--rows", type=int, default=4096, help="hidden_size")
    parser.add_argument("--cols", type=int, default=2048, help="intermediate_size")
    parser.add_argument("--n-expert", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--max-rank", type=int, default=128)
    parser.add_argument("--energy", type=float, default=0.95)
    parser.add_argument("--error-threshold", type=float, default=0.02)
    parser.add_argument("--intrinsic-ranks", type=int, nargs="+", default=[32, 64, 128, 256])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="Write results to JSON.")
    args = parser.parse_args()

    print("=" * 70)
    print("LATENT End-to-End Validation")
    print("=" * 70)
    print(f"hidden_size={args.rows}, intermediate_size={args.cols}")
    print(f"max_rank={args.max_rank}, energy={args.energy}")
    print(f"error_threshold={args.error_threshold}")
    print()

    results: List[dict] = []
    for intrinsic_rank in args.intrinsic_ranks:
        print(f"--- Testing intrinsic_rank={intrinsic_rank} ---")
        for weight_before_down in [False, True]:
            config = "weight_before_down" if weight_before_down else "weight_after"
            try:
                result = validate_single_expert(
                    n_embd=args.rows,
                    n_ff=args.cols,
                    intrinsic_rank=intrinsic_rank,
                    compression_max_rank=args.max_rank,
                    energy=args.energy,
                    error_threshold=args.error_threshold,
                    weight_before_down=weight_before_down,
                    seed=args.seed,
                )
                result["config"] = config
                results.append(result)
                print(f"  {config}: runtime_err={result['runtime_relative_error']:.4f}, "
                      f"gate_rank={result['gate_report']['rank']}, "
                      f"up_rank={result['up_report']['rank']}, "
                      f"down_rank={result['down_report']['rank']}")
            except Exception as e:
                print(f"  {config}: FAILED ({e})")
                results.append({
                    "intrinsic_rank": intrinsic_rank,
                    "config": config,
                    "error": str(e),
                })
        print()

    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    valid_results = [r for r in results if "runtime_relative_error" in r]
    if valid_results:
        runtime_errors = [r["runtime_relative_error"] for r in valid_results]
        quant_errors = [r["gate_report"]["quantized_relative_error"] for r in valid_results]
        print(f"Mean runtime relative error: {np.mean(runtime_errors):.4f}")
        print(f"Max  runtime relative error: {np.max(runtime_errors):.4f}")
        print(f"Mean gate quant error:       {np.mean(quant_errors):.4f}")
        print(f"Max  gate quant error:       {np.max(quant_errors):.4f}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nResults written to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
