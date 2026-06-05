"""Tests for the full model SVD pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loader.full_model_svd import compress_expert_to_factors


def make_synthetic_experts(n_in: int, n_out: int, n_expert: int, intrinsic_rank: int, seed: int) -> np.ndarray:
    """Generate synthetic expert matrices with known intrinsic rank."""
    rng = np.random.default_rng(seed)
    matrix = np.zeros((n_in, n_out, n_expert), dtype=np.float32)
    for e in range(n_expert):
        left = rng.standard_normal((n_in, intrinsic_rank), dtype=np.float32)
        right = rng.standard_normal((intrinsic_rank, n_out), dtype=np.float32)
        matrix[:, :, e] = left @ right
    return matrix


def test_compress_expert_to_factors_runs():
    """Test that the compression pipeline runs on synthetic data."""
    matrix = make_synthetic_experts(
        n_in=64, n_out=32, n_expert=2, intrinsic_rank=4, seed=42
    )

    factor_arrays, report = compress_expert_to_factors(
        name="blk.0.ffn_gate_exps",
        matrix=matrix,
        energy=0.99,
        min_rank=4,
        max_rank=8,
        group_size=32,
        int8_group_size=64,
        error_threshold=0.5,
        snr_threshold=0.0,
        low_snr_fraction_threshold=1.0,
    )

    assert "blk.0.ffn_gate_exps.expert0.a.nf4" in factor_arrays
    assert "blk.0.ffn_gate_exps.expert0.a.scales" in factor_arrays
    assert "blk.0.ffn_gate_exps.expert0.a.shape" in factor_arrays
    assert "blk.0.ffn_gate_exps.expert0.b.nf4" in factor_arrays

    # The report should reflect the compression
    assert report.shape == (64, 32, 2)
    assert report.retained_energy > 0.9
    assert not report.fallback


def test_compress_expert_to_factors_multiple_experts():
    """Test that multiple experts are compressed independently."""
    matrix = make_synthetic_experts(
        n_in=32, n_out=16, n_expert=4, intrinsic_rank=3, seed=123
    )

    factor_arrays, report = compress_expert_to_factors(
        name="blk.1.ffn_up_exps",
        matrix=matrix,
        energy=0.99,
        min_rank=2,
        max_rank=4,
        group_size=32,
        int8_group_size=64,
        error_threshold=0.5,
        snr_threshold=0.0,
        low_snr_fraction_threshold=1.0,
    )

    # All 4 experts should be present
    for e in range(4):
        assert f"blk.1.ffn_up_exps.expert{e}.a.nf4" in factor_arrays
        assert f"blk.1.ffn_up_exps.expert{e}.b.nf4" in factor_arrays

    # Shape should be the full expert grid
    assert report.shape == (32, 16, 4)


if __name__ == "__main__":
    test_compress_expert_to_factors_runs()
    test_compress_expert_to_factors_multiple_experts()
    print("All tests passed")
