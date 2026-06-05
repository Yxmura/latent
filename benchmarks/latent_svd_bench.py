#!/usr/bin/env python3
"""Synthetic LATENT compression benchmark."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loader.latent_loader import compress_matrix


def make_matrix(rows: int, cols: int, intrinsic_rank: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    left = rng.standard_normal((rows, intrinsic_rank), dtype=np.float32)
    right = rng.standard_normal((intrinsic_rank, cols), dtype=np.float32)
    noise = 0.01 * rng.standard_normal((rows, cols), dtype=np.float32)
    return left @ right + noise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--cols", type=int, default=512)
    parser.add_argument("--intrinsic-rank", type=int, default=64)
    parser.add_argument("--max-rank", type=int, default=128)
    parser.add_argument("--energy", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    matrix = make_matrix(args.rows, args.cols, args.intrinsic_rank, args.seed)
    start = time.perf_counter()
    _, report = compress_matrix(
        "synthetic",
        matrix,
        energy=args.energy,
        min_rank=16,
        max_rank=args.max_rank,
        group_size=32,
        int8_group_size=64,
        error_threshold=0.02,
        snr_threshold=3.0,
        low_snr_fraction_threshold=0.05,
        seed=args.seed,
    )
    elapsed = time.perf_counter() - start
    print(f"shape={report.shape}")
    print(f"rank={report.rank}")
    print(f"retained_energy={report.retained_energy:.6f}")
    print(f"dense_relative_error={report.dense_relative_error:.6f}")
    print(f"quantized_relative_error={report.quantized_relative_error:.6f}")
    print(f"fallback={report.fallback}")
    print(f"elapsed_sec={elapsed:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
