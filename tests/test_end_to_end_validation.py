"""Tests for the end-to-end validation pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.end_to_end_validation import (
    compress_and_pack,
    make_realistic_expert,
    validate_single_expert,
)


def test_make_realistic_expert():
    """Test that synthetic expert generation produces expected shapes."""
    matrix = make_realistic_expert(64, 32, 8, seed=42)
    assert matrix.shape == (64, 32)
    assert matrix.dtype == np.float32
    # Should have non-trivial values
    assert np.linalg.norm(matrix) > 0


def test_compress_and_pack_returns_arrays_and_report():
    """Test that compress_and_pack produces factor arrays and a report."""
    matrix = make_realistic_expert(64, 32, 8, seed=42)
    arrays, report = compress_and_pack(
        "test", matrix, energy=0.99, min_rank=4, max_rank=16, group_size=32, error_threshold=0.5
    )
    assert "test.a.nf4" in arrays
    assert "test.a.scales" in arrays
    assert "test.a.shape" in arrays
    assert "test.b.nf4" in arrays
    assert "test.b.scales" in arrays
    assert "test.b.shape" in arrays
    assert report["shape"] == [64, 32]
    assert report["rank"] >= 4
    assert not report["fallback"]


def test_validate_single_expert_weight_after():
    """Test end-to-end validation with weight applied after FFN."""
    result = validate_single_expert(
        n_embd=64,
        n_ff=32,
        intrinsic_rank=8,
        compression_max_rank=16,
        energy=0.99,
        error_threshold=0.5,
        weight_before_down=False,
        seed=42,
    )
    assert "runtime_relative_error" in result
    assert result["gate_report"]["rank"] >= 4
    assert result["up_report"]["rank"] >= 4
    assert result["down_report"]["rank"] >= 4
    # Runtime error should be finite
    assert np.isfinite(result["runtime_relative_error"])


def test_validate_single_expert_weight_before_down():
    """Test end-to-end validation with DeepSeek V4 weight_before_down=True."""
    result = validate_single_expert(
        n_embd=64,
        n_ff=32,
        intrinsic_rank=8,
        compression_max_rank=16,
        energy=0.99,
        error_threshold=0.5,
        weight_before_down=True,
        seed=42,
    )
    assert "runtime_relative_error" in result
    # Weight-before-down should produce a finite error
    assert np.isfinite(result["runtime_relative_error"])


def test_validate_single_expert_higher_max_rank_lower_error():
    """Increasing max_rank (with the same intrinsic rank) should reduce runtime error."""
    err_fewer = validate_single_expert(
        n_embd=64, n_ff=32, intrinsic_rank=16,
        compression_max_rank=16, energy=0.99, error_threshold=0.5, seed=42
    )["runtime_relative_error"]

    err_more = validate_single_expert(
        n_embd=64, n_ff=32, intrinsic_rank=16,
        compression_max_rank=32, energy=0.99, error_threshold=0.5, seed=42
    )["runtime_relative_error"]

    # More capacity (higher max_rank) -> lower runtime error
    assert err_more <= err_fewer
