"""Tests for the adaptive top-k module."""

from __future__ import annotations

import numpy as np

from loader.adaptive_topk import (
    adaptive_topk_batch,
    adaptive_topk_inclusive,
    expected_savings,
)


def test_inclusive_check_basic():
    """Test the basic inclusive threshold check."""
    # Example from PRISM.md: scores [0.45, 0.22, 0.15, 0.08, 0.05, 0.03]
    scores = np.array([0.45, 0.22, 0.15, 0.08, 0.05, 0.03], dtype=np.float32)
    total = scores.sum()
    target = 0.95 * total  # 0.931
    # cum progression: 0.45, 0.67, 0.82, 0.90, 0.95, 0.98
    # cum + score:    0.90, 0.89, 0.97, 0.98, 1.00, 1.01
    # threshold 0.931: stop when cum+score > 0.931
    # i=0: 0+0.45=0.45 <= 0.931, keep (cum=0.45)
    # i=1: 0.45+0.22=0.67 <= 0.931, keep (cum=0.67)
    # i=2: 0.67+0.15=0.82 <= 0.931, keep (cum=0.82)
    # i=3: 0.82+0.08=0.90 <= 0.931, keep (cum=0.90)
    # i=4: 0.90+0.05=0.95 > 0.931, stop
    n_keep = adaptive_topk_inclusive(scores, threshold=0.95)
    assert n_keep == 4


def test_inclusive_check_threshold_one():
    """Threshold 1.0 should keep all experts."""
    scores = np.array([0.45, 0.22, 0.15, 0.08, 0.05, 0.03], dtype=np.float32)
    n_keep = adaptive_topk_inclusive(scores, threshold=1.0)
    assert n_keep == 6


def test_inclusive_check_threshold_low():
    """Very low threshold should keep zero or one expert."""
    scores = np.array([0.45, 0.22, 0.15, 0.08, 0.05, 0.03], dtype=np.float32)
    n_keep = adaptive_topk_inclusive(scores, threshold=0.1)
    # threshold 0.1 of total 0.98 = 0.098
    # First score 0.45 alone exceeds 0.098, so inclusive check returns 0
    assert n_keep == 0


def test_inclusive_check_all_zero():
    """All-zero scores should keep 0 experts."""
    scores = np.zeros(6, dtype=np.float32)
    n_keep = adaptive_topk_inclusive(scores, threshold=0.95)
    assert n_keep == 0


def test_inclusive_check_empty():
    """Empty input should return 0."""
    scores = np.array([], dtype=np.float32)
    n_keep = adaptive_topk_inclusive(scores, threshold=0.95)
    assert n_keep == 0


def test_batch_processing():
    """Test batch processing of adaptive top-k."""
    rng = np.random.default_rng(42)
    scores = rng.standard_normal((10, 256), dtype=np.float32)
    scores = np.exp(scores) / np.exp(scores).sum(axis=1, keepdims=True)

    keep_mask, n_kept = adaptive_topk_batch(scores, top_k=6, threshold=0.95)
    assert keep_mask.shape == (10, 6)
    assert n_kept.shape == (10,)
    # All n_kept should be in [1, 6]
    assert np.all(n_kept >= 1)
    assert np.all(n_kept <= 6)


def test_savings_typical():
    """Test that expected savings is in a reasonable range."""
    # Per PRISM.md, typical savings at threshold 0.95 is ~17%
    savings = expected_savings(threshold=0.95, n_samples=1000, n_experts=256, top_k=6, seed=0)
    # With uniform-ish softmax over 256 experts, top-6 will dominate;
    # the threshold 0.95 should drop a few experts, yielding 10-50% savings
    assert 0.0 < savings < 0.8
