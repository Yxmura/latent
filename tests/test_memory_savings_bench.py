"""Tests for the memory savings benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.memory_savings_bench import _bytes, measure_matrix


def test_bytes_helper():
    """Test that _bytes correctly multiplies count by dtype size."""
    assert _bytes(1024, 2) == 2048
    assert _bytes(1024, 4) == 4096
    assert _bytes(0, 4) == 0


def test_measure_matrix_compresses():
    """Test that measure_matrix returns a reasonable result."""
    rng = np.random.default_rng(42)
    matrix = rng.standard_normal((64, 32), dtype=np.float32)
    result = measure_matrix(
        name="test",
        matrix=matrix,
        energy=0.99,
        max_rank=16,
        error_threshold=0.5,
    )
    assert result["name"] == "test"
    assert result["rows"] == 64
    assert result["cols"] == 32
    assert result["rank"] >= 4
    assert result["dense_bytes"] == 64 * 32 * 4
    # Latent should be smaller than dense
    assert result["latent_bytes"] < result["dense_bytes"]
    assert result["savings_ratio"] > 1.0


def test_measure_matrix_low_intrinsic_high_compression():
    """A matrix with low intrinsic rank should compress much better."""
    rng = np.random.default_rng(42)
    # Intrinsic rank 4
    matrix = (
        rng.standard_normal((64, 4), dtype=np.float32)
        @ rng.standard_normal((4, 32), dtype=np.float32)
    )
    result = measure_matrix(
        name="lowrank",
        matrix=matrix,
        energy=0.99,
        max_rank=32,
        error_threshold=0.5,
    )
    # With intrinsic rank 4, latent should be vastly smaller than dense
    assert result["savings_ratio"] > 5.0
    assert not result["fallback"]
