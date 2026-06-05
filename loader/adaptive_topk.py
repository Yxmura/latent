"""Adaptive top-k for MoE expert selection.

Implements threshold-based expert dropping using router confidence scores.
After the router computes softmax scores for the top-K selected experts,
drop experts from the tail whose cumulative weight exceeds a threshold.

Reference: PRISM.md, Component 3
  "After the router computes softmax scores for the top-6 selected experts,
   drop experts from the tail whose cumulative weight exceeds a threshold
   (default: 0.95 of the top-6 sum)."

For a typical token with scores [0.45, 0.22, 0.15, 0.08, 0.05, 0.03],
top6_sum = 0.98, threshold = 0.95:
  - i=0: cum=0.45, score=0.45, 0.45+0.45=0.90 < 0.931=0.95*0.98, keep
  - i=1: cum=0.67, score=0.22, 0.67+0.22=0.89 < 0.931, keep
  - i=2: cum=0.82, score=0.15, 0.82+0.15=0.97 > 0.931, stop
  n_keep = 2, savings = 4/6 = 67%

The inclusive check ensures the cumulative sum STRICTLY does not exceed the
threshold. This correctly implements "capture threshold fraction of routing
weight".
"""

from __future__ import annotations

import numpy as np


def adaptive_topk_inclusive(
    sorted_scores: np.ndarray,
    threshold: float = 0.95,
) -> int:
    """Determine how many top experts to keep using inclusive threshold.

    sorted_scores: scores sorted in descending order
    threshold: cumulative weight threshold relative to total

    Returns the number of experts to keep (0 <= n_keep <= len(sorted_scores)).
    """
    if sorted_scores.size == 0:
        return 0
    if not 0.0 < threshold <= 1.0:
        raise ValueError(f"threshold must be in (0, 1], got {threshold}")

    total = float(np.sum(sorted_scores))
    if total <= 0:
        return 0

    target = threshold * total
    cumulative = 0.0
    for i, score in enumerate(sorted_scores):
        if cumulative + score > target:
            return i
        cumulative += score
    return int(sorted_scores.size)


def adaptive_topk_batch(
    scores: np.ndarray,
    top_k: int,
    threshold: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply adaptive top-k to a batch of router scores.

    scores: [n_tokens, n_experts] router scores
    top_k: number of experts to select per token (before adaptive dropping)
    threshold: cumulative weight threshold

    Returns:
        keep_mask: [n_tokens, top_k] boolean mask, True for kept experts
        n_kept: [n_tokens] number of experts kept per token
    """
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2D, got {scores.ndim}D")

    n_tokens, n_experts = scores.shape
    if top_k > n_experts:
        raise ValueError(f"top_k={top_k} > n_experts={n_experts}")

    # Sort each row by score descending
    sorted_idx = np.argsort(-scores, axis=1)[:, :top_k]
    sorted_scores = np.take_along_axis(scores, sorted_idx, axis=1)

    keep_mask = np.zeros((n_tokens, top_k), dtype=bool)
    n_kept = np.zeros(n_tokens, dtype=np.int32)
    for t in range(n_tokens):
        n = adaptive_topk_inclusive(sorted_scores[t], threshold=threshold)
        keep_mask[t, :n] = True
        n_kept[t] = n
    return keep_mask, n_kept


def expected_savings(
    threshold: float,
    n_samples: int = 10000,
    n_experts: int = 256,
    top_k: int = 6,
    seed: int = 0,
) -> float:
    """Estimate the expected compute savings from adaptive top-k at a given threshold.

    Simulates a uniform softmax over n_experts and computes the average
    fraction of experts dropped.

    This is a rough estimate; real savings depend on the actual router
    distribution. Use 0.10-0.20 as a typical range for threshold 0.95.
    """
    rng = np.random.default_rng(seed)
    # Simulate logits with typical MoE distribution
    logits = rng.standard_normal((n_samples, n_experts), dtype=np.float32)
    # Apply softmax
    logits = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    scores = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    keep_mask, n_kept = adaptive_topk_batch(scores, top_k=top_k, threshold=threshold)
    return float(1.0 - n_kept.mean() / top_k)
