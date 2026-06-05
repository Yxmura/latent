#!/usr/bin/env python3
"""Memory-savings benchmark for LATENT.

Measures the per-matrix, per-layer, and aggregate compression ratio of
LATENT on synthetic expert matrices with configurable intrinsic rank.

The goal is to quantify the *theoretical maximum* VRAM savings from
representing experts as low-rank A/B factors instead of dense matrices.

Usage:
  python benchmarks/memory_savings_bench.py \\
    --hidden-size 4096 \\
    --intermediate-size 2048 \\
    --n-expert 256 \\
    --max-rank 128 \\
    --energy 0.95
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loader.latent_loader import compress_matrix


def _bytes(n: int, dtype_bytes: int) -> int:
    return n * dtype_bytes


def measure_matrix(
    name: str,
    matrix: np.ndarray,
    energy: float,
    max_rank: int,
    error_threshold: float,
) -> dict:
    """Compress one matrix and measure bytes before / after."""
    rows, cols = matrix.shape
    dense_bytes = _bytes(rows * cols, 4)  # FP32 baseline
    n_expert = 1

    # We need a per-expert shape for compress_matrix
    arrays, report = compress_matrix(
        name=name,
        matrix=matrix,
        energy=energy,
        min_rank=8,
        max_rank=max_rank,
        group_size=32,
        int8_group_size=64,
        error_threshold=error_threshold,
        snr_threshold=3.0,
        low_snr_fraction_threshold=0.05,
        seed=hash(name) & 0xFFFF,
    )

    # Factor storage: NF4 stores 4 bits per element + FP16 scales per group
    rank = report.rank
    # A: [rows, rank], B: [rank, cols]
    # Each factor has nf4 + scales (FP16 per 32 elements) + shape metadata
    def factor_bytes(rows_: int, cols_: int) -> int:
        n_elem = rows_ * cols_
        nf4_bytes = (n_elem + 1) // 2  # 4 bits packed
        n_groups = (n_elem + 31) // 32
        scale_bytes = n_groups * 2  # FP16 scales
        shape_bytes = 8  # two int64s
        return nf4_bytes + scale_bytes + shape_bytes

    a_bytes = factor_bytes(rows, rank)
    b_bytes = factor_bytes(rank, cols)
    latent_bytes = a_bytes + b_bytes

    return {
        "name": name,
        "rows": rows,
        "cols": cols,
        "rank": rank,
        "dense_bytes": dense_bytes,
        "latent_bytes": latent_bytes,
        "savings_ratio": dense_bytes / max(latent_bytes, 1),
        "fallback": report.fallback,
        "retained_energy": report.retained_energy,
        "quantized_relative_error": report.quantized_relative_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="LATENT memory savings benchmark.")
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--intermediate-size", type=int, default=2048)
    parser.add_argument("--n-expert", type=int, default=256)
    parser.add_argument("--max-rank", type=int, default=128)
    parser.add_argument("--energy", type=float, default=0.95)
    parser.add_argument("--error-threshold", type=float, default=0.02)
    parser.add_argument("--intrinsic-rank", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("=" * 72)
    print("LATENT Memory Savings Benchmark")
    print("=" * 72)
    print(f"hidden_size        = {args.hidden_size}")
    print(f"intermediate_size  = {args.intermediate_size}")
    print(f"n_expert / layer   = {args.n_expert}")
    print(f"max_rank           = {args.max_rank}")
    print(f"energy             = {args.energy}")
    print(f"intrinsic_rank     = {args.intrinsic_rank}")
    print()

    rng = np.random.default_rng(args.seed)

    # One layer's worth of three expert matrices (gate / up / down)
    n_ff = args.intermediate_size
    n_embd = args.hidden_size
    ir = args.intrinsic_rank

    # gate / up: [n_embd, n_ff]
    gate = (rng.standard_normal((n_embd, ir), dtype=np.float32)
            @ rng.standard_normal((ir, n_ff), dtype=np.float32))
    up = (rng.standard_normal((n_embd, ir), dtype=np.float32)
          @ rng.standard_normal((ir, n_ff), dtype=np.float32))
    # down: [n_ff, n_embd]
    down = (rng.standard_normal((n_ff, ir), dtype=np.float32)
            @ rng.standard_normal((ir, n_embd), dtype=np.float32))

    gate_r = measure_matrix("gate", gate, args.energy, args.max_rank, args.error_threshold)
    up_r = measure_matrix("up", up, args.energy, args.max_rank, args.error_threshold)
    down_r = measure_matrix("down", down, args.energy, args.max_rank, args.error_threshold)

    per_expert_dense = gate_r["dense_bytes"] + up_r["dense_bytes"] + down_r["dense_bytes"]
    per_expert_latent = gate_r["latent_bytes"] + up_r["latent_bytes"] + down_r["latent_bytes"]
    n_layers = 1
    per_layer_dense = per_expert_dense * args.n_expert
    per_layer_latent = per_expert_latent * args.n_expert
    total_dense_mb = per_layer_dense * n_layers / (1024 * 1024)
    total_latent_mb = per_layer_latent * n_layers / (1024 * 1024)
    savings = total_dense_mb - total_latent_mb
    ratio = total_dense_mb / max(total_latent_mb, 1e-9)

    print("Per-expert (gate + up + down):")
    print(f"  dense  : {per_expert_dense / 1024:.1f} KiB")
    print(f"  latent : {per_expert_latent / 1024:.1f} KiB "
          f"(rank={gate_r['rank']}/{up_r['rank']}/{down_r['rank']})")
    print(f"  ratio  : {per_expert_dense / max(per_expert_latent, 1):.2f}x")
    print()
    print(f"Per-layer (×{args.n_expert} experts):")
    print(f"  dense  : {per_layer_dense / (1024 * 1024):.1f} MiB")
    print(f"  latent : {per_layer_latent / (1024 * 1024):.1f} MiB")
    print()
    print(f"Per-layer VRAM savings: {savings:.1f} MiB ({ratio:.2f}x compression)")
    print()
    print("Note: This is a *theoretical* compression bound for synthetic matrices.")
    print("Real DeepSeek V4 Flash experts may have higher intrinsic rank, reducing")
    print("the achievable ratio. The full-model SVD pipeline is the source of truth.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
