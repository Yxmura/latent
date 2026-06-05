#!/usr/bin/env python3
"""LATENT load-time expert compression prototype.

This script compresses expert FFN matrices into low-rank A/B factors:

    W ~= A @ B

where A = U * sqrt(S) and B = sqrt(S) * Vt from a truncated SVD. The
factors are quantized with NF4 and per-group FP16 scales.

The current input format is an NPZ file containing dense matrices. This keeps
the compression path testable before binding it to a specific llama.cpp GGUF
fork and tensor naming scheme.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np


NF4_CODEBOOK = np.array(
    [
        -1.0,
        -0.6961928,
        -0.52507305,
        -0.3949175,
        -0.28444138,
        -0.18477343,
        -0.09105004,
        0.0,
        0.0795803,
        0.1609302,
        0.2461123,
        0.33791524,
        0.44070983,
        0.562617,
        0.72295684,
        1.0,
    ],
    dtype=np.float32,
)


@dataclass
class MatrixReport:
    name: str
    shape: Tuple[int, int]
    rank: int
    retained_energy: float
    dense_relative_error: float
    quantized_relative_error: float
    min_group_snr: float
    low_snr_group_fraction: float
    quantization: str
    fallback: bool
    fallback_reason: str


def randomized_svd(
    matrix: np.ndarray,
    target_rank: int,
    oversample: int = 16,
    power_iterations: int = 1,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute a truncated randomized SVD for a 2D matrix."""
    if matrix.ndim != 2:
        raise ValueError(f"expected 2D matrix, got shape {matrix.shape}")
    m, n = matrix.shape
    rank = min(target_rank, m, n)
    sketch_rank = min(rank + oversample, m, n)
    rng = np.random.default_rng(seed)
    omega = rng.standard_normal((n, sketch_rank), dtype=np.float32)

    y = matrix @ omega
    for _ in range(power_iterations):
        y = matrix @ (matrix.T @ y)

    q, _ = np.linalg.qr(y, mode="reduced")
    b = q.T @ matrix
    u_hat, s, vt = np.linalg.svd(b, full_matrices=False)
    u = q @ u_hat
    return u[:, :rank], s[:rank], vt[:rank, :]


def choose_rank(singular_values: np.ndarray, energy: float, min_rank: int, max_rank: int) -> int:
    if not 0 < energy <= 1:
        raise ValueError("energy must be in (0, 1]")
    total = float(np.sum(singular_values**2))
    if total == 0:
        return min_rank
    cumulative = np.cumsum(singular_values**2) / total
    rank = int(np.searchsorted(cumulative, energy) + 1)
    return max(min_rank, min(rank, max_rank, singular_values.size))


def make_factors(u: np.ndarray, s: np.ndarray, vt: np.ndarray, rank: int) -> Tuple[np.ndarray, np.ndarray]:
    root = np.sqrt(s[:rank]).astype(np.float32)
    a = u[:, :rank].astype(np.float32) * root[np.newaxis, :]
    b = root[:, np.newaxis] * vt[:rank, :].astype(np.float32)
    return a, b


def quantize_nf4(values: np.ndarray, group_size: int = 32) -> Tuple[np.ndarray, np.ndarray, Tuple[int, ...]]:
    """Quantize values to packed NF4 nibbles plus one FP16 scale per group."""
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    original_shape = values.shape
    groups = int(math.ceil(flat.size / group_size))
    padded_size = groups * group_size
    padded = np.zeros(padded_size, dtype=np.float32)
    padded[: flat.size] = flat
    grouped = padded.reshape(groups, group_size)

    scales = np.max(np.abs(grouped), axis=1).astype(np.float32)
    scales[scales == 0] = 1.0
    normalized = grouped / scales[:, np.newaxis]
    distances = np.abs(normalized[..., np.newaxis] - NF4_CODEBOOK[np.newaxis, np.newaxis, :])
    codes = np.argmin(distances, axis=-1).astype(np.uint8).reshape(-1)

    if codes.size % 2:
        codes = np.pad(codes, (0, 1), constant_values=7)
    packed = (codes[0::2] | (codes[1::2] << 4)).astype(np.uint8)
    return packed, scales.astype(np.float16), original_shape


def dequantize_nf4(
    packed: np.ndarray,
    scales: np.ndarray,
    original_shape: Tuple[int, ...],
    group_size: int = 32,
) -> np.ndarray:
    codes = np.empty(packed.size * 2, dtype=np.uint8)
    codes[0::2] = packed & 0x0F
    codes[1::2] = packed >> 4
    groups = int(math.ceil(np.prod(original_shape) / group_size))
    codes = codes[: groups * group_size].reshape(groups, group_size)
    decoded = NF4_CODEBOOK[codes] * scales.astype(np.float32)[:, np.newaxis]
    return decoded.reshape(-1)[: int(np.prod(original_shape))].reshape(original_shape).astype(np.float32)


def quantize_int8(values: np.ndarray, group_size: int = 64) -> Tuple[np.ndarray, np.ndarray, Tuple[int, ...]]:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    original_shape = values.shape
    groups = int(math.ceil(flat.size / group_size))
    padded = np.zeros(groups * group_size, dtype=np.float32)
    padded[: flat.size] = flat
    grouped = padded.reshape(groups, group_size)
    scales = np.max(np.abs(grouped), axis=1).astype(np.float32) / 127.0
    scales[scales == 0] = 1.0
    quantized = np.rint(grouped / scales[:, np.newaxis]).clip(-127, 127).astype(np.int8)
    return quantized.reshape(-1), scales.astype(np.float16), original_shape


def dequantize_int8(
    quantized: np.ndarray,
    scales: np.ndarray,
    original_shape: Tuple[int, ...],
    group_size: int = 64,
) -> np.ndarray:
    groups = int(math.ceil(np.prod(original_shape) / group_size))
    grouped = quantized[: groups * group_size].reshape(groups, group_size).astype(np.float32)
    decoded = grouped * scales.astype(np.float32)[:, np.newaxis]
    return decoded.reshape(-1)[: int(np.prod(original_shape))].reshape(original_shape).astype(np.float32)


def quantization_snr_by_group(reference: np.ndarray, candidate: np.ndarray, group_size: int) -> np.ndarray:
    ref = np.asarray(reference, dtype=np.float32).reshape(-1)
    cand = np.asarray(candidate, dtype=np.float32).reshape(-1)
    groups = int(math.ceil(ref.size / group_size))
    padded_ref = np.zeros(groups * group_size, dtype=np.float32)
    padded_cand = np.zeros(groups * group_size, dtype=np.float32)
    padded_ref[: ref.size] = ref
    padded_cand[: cand.size] = cand
    ref_g = padded_ref.reshape(groups, group_size)
    err_g = (padded_ref - padded_cand).reshape(groups, group_size)
    signal = np.linalg.norm(ref_g, axis=1)
    noise = np.linalg.norm(err_g, axis=1)
    return signal / np.maximum(noise, 1e-12)


def relative_frobenius_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    denom = np.linalg.norm(reference, ord="fro")
    if denom == 0:
        return 0.0
    return float(np.linalg.norm(reference - candidate, ord="fro") / denom)


def compress_matrix(
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
    seed: int,
) -> Tuple[Dict[str, np.ndarray], MatrixReport]:
    full_s = np.linalg.svd(matrix, compute_uv=False)
    rank = choose_rank(full_s, energy=energy, min_rank=min_rank, max_rank=max_rank)
    u, s, vt = randomized_svd(matrix, target_rank=rank, seed=seed)
    a, b = make_factors(u, s, vt, rank)
    dense_reconstruction = a @ b
    dense_error = relative_frobenius_error(matrix, dense_reconstruction)

    a_packed, a_scales, a_shape = quantize_nf4(a, group_size=group_size)
    b_packed, b_scales, b_shape = quantize_nf4(b, group_size=group_size)
    aq = dequantize_nf4(a_packed, a_scales, a_shape, group_size=group_size)
    bq = dequantize_nf4(b_packed, b_scales, b_shape, group_size=group_size)
    quant_error = relative_frobenius_error(matrix, aq @ bq)
    snr = np.concatenate([
        quantization_snr_by_group(a, aq, group_size),
        quantization_snr_by_group(b, bq, group_size),
    ])
    min_group_snr = float(np.min(snr)) if snr.size else float("inf")
    low_snr_fraction = float(np.mean(snr < snr_threshold)) if snr.size else 0.0

    retained = float(np.sum(s[:rank] ** 2) / np.sum(full_s**2)) if np.sum(full_s**2) else 1.0
    fallback_reason = ""
    use_int8 = low_snr_fraction > low_snr_fraction_threshold
    if use_int8:
        a_i8, a_i8_scales, a_i8_shape = quantize_int8(a, group_size=int8_group_size)
        b_i8, b_i8_scales, b_i8_shape = quantize_int8(b, group_size=int8_group_size)
        aq_i8 = dequantize_int8(a_i8, a_i8_scales, a_i8_shape, group_size=int8_group_size)
        bq_i8 = dequantize_int8(b_i8, b_i8_scales, b_i8_shape, group_size=int8_group_size)
        int8_error = relative_frobenius_error(matrix, aq_i8 @ bq_i8)
        if int8_error < quant_error:
            quant_error = int8_error
            arrays = {
                f"{name}.a.int8": a_i8,
                f"{name}.a.scales": a_i8_scales,
                f"{name}.b.int8": b_i8,
                f"{name}.b.scales": b_i8_scales,
                f"{name}.a.shape": np.array(a_i8_shape, dtype=np.int32),
                f"{name}.b.shape": np.array(b_i8_shape, dtype=np.int32),
            }
            quantization = "int8_g%d" % int8_group_size
            fallback_reason = "nf4_low_group_snr"
        else:
            use_int8 = False

    if not use_int8:
        arrays = {
            f"{name}.a.nf4": a_packed,
            f"{name}.a.scales": a_scales,
            f"{name}.b.nf4": b_packed,
            f"{name}.b.scales": b_scales,
            f"{name}.a.shape": np.array(a_shape, dtype=np.int32),
            f"{name}.b.shape": np.array(b_shape, dtype=np.int32),
        }
        quantization = "nf4_g%d" % group_size

    dense_fallback = quant_error > error_threshold
    if dense_fallback:
        arrays[f"{name}.dense_fallback"] = matrix.astype(np.float16)
        fallback_reason = "relative_error_threshold"

    report = MatrixReport(
        name=name,
        shape=tuple(int(x) for x in matrix.shape),
        rank=rank,
        retained_energy=retained,
        dense_relative_error=dense_error,
        quantized_relative_error=quant_error,
        min_group_snr=min_group_snr,
        low_snr_group_fraction=low_snr_fraction,
        quantization=quantization,
        fallback=dense_fallback,
        fallback_reason=fallback_reason,
    )
    return arrays, report


def iter_matrices(npz: np.lib.npyio.NpzFile) -> Iterable[Tuple[str, np.ndarray]]:
    for key in sorted(npz.files):
        value = np.asarray(npz[key])
        if value.ndim == 2:
            yield key, value.astype(np.float32)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compress dense expert matrices into LATENT NF4 factors.")
    parser.add_argument("--input", required=True, type=Path, help="Input NPZ with 2D dense matrices.")
    parser.add_argument("--output", required=True, type=Path, help="Output NPZ for packed factors.")
    parser.add_argument("--metadata", type=Path, help="Output JSON metadata path.")
    parser.add_argument("--energy", type=float, default=0.95, help="Target retained spectral energy.")
    parser.add_argument("--min-rank", type=int, default=16)
    parser.add_argument("--max-rank", type=int, default=128)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--int8-group-size", type=int, default=64)
    parser.add_argument("--error-threshold", type=float, default=0.02)
    parser.add_argument("--snr-threshold", type=float, default=3.0)
    parser.add_argument("--low-snr-fraction-threshold", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.group_size <= 0:
        raise ValueError("--group-size must be positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = args.metadata or args.output.with_suffix(".json")

    compressed: Dict[str, np.ndarray] = {}
    reports = []
    with np.load(args.input) as npz:
        for index, (name, matrix) in enumerate(iter_matrices(npz)):
            arrays, report = compress_matrix(
                name=name,
                matrix=matrix,
                energy=args.energy,
                min_rank=args.min_rank,
                max_rank=args.max_rank,
                group_size=args.group_size,
                int8_group_size=args.int8_group_size,
                error_threshold=args.error_threshold,
                snr_threshold=args.snr_threshold,
                low_snr_fraction_threshold=args.low_snr_fraction_threshold,
                seed=args.seed + index,
            )
            compressed.update(arrays)
            reports.append(report)

    np.savez_compressed(args.output, **compressed)
    metadata = {
        "format": "latent-nf4-factors-v0",
        "input": str(args.input),
        "output": str(args.output),
        "assumptions": {
            "gguf_tensor_shapes_verified": False,
            "kernel_integrated_with_llama_cpp": False,
            "quality_requires_end_to_end_eval": True,
        },
        "matrices": [asdict(report) for report in reports],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"compressed_matrices": len(reports), "metadata": str(metadata_path)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
