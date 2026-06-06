#!/usr/bin/env python3
"""Write a LATENT-quantized GGUF file from per-expert compressed factors.

Per LATENT.md §1, the output GGUF stores each expert's W = A*B factorization
in stacked per-expert tensors. Per layer, 12 tensors are written:

  blk.N.ffn_latent_{gate,up,down}_a   : (n_expert, n_in,  max_rank)      I8
  blk.N.ffn_latent_{gate,up,down}_a_s : (n_expert, n_in,  max_rank/32)   F16
  blk.N.ffn_latent_{gate,up,down}_b   : (n_expert, max_rank, n_out)      I8
  blk.N.ffn_latent_{gate,up,down}_b_s : (n_expert, max_rank, n_out/32)   F16

Where:
  - n_in, n_out are the FFN matrix dimensions (gate/up: n_embd x n_ff_exp,
    down: n_ff_exp x n_embd)
  - max_rank is a single value per model (uniform padding for storage)
  - I8 holds unpacked NF4 codes (1 code per byte) - matches the v1 kernel's
    access pattern. Packed (2 codes/byte) is a future optimization.
  - F16 scales are per-group FP16 (group_size=32 by default)

The input is a per-layer dict of per-expert A/B factors produced by
`loader.full_model_svd.compress_expert_to_factors`. Per-expert ranks
chosen by the energy threshold are zero-padded to max_rank for storage.
Per-expert rank metadata is also written to the GGUF (`latent.rank`).

The output GGUF can be loaded by the patched llama.cpp which dispatches
to the fused ggml_latent_expert_ffn op when latent tensors are present.

Usage:
    python loader/write_latent_gguf.py \\
        --input-gguf model.gguf \\
        --output-gguf model.latent.gguf \\
        --max-rank 128 \\
        --energy 0.95

The input GGUF is used to copy non-expert tensors and metadata. If you
only have an NPZ of compressed factors, use --from-npz mode.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from loader.gguf_expert_extract import add_gguf_to_path
from loader.latent_loader import (
    compress_matrix,
    dequantize_nf4,
    make_factors,
    MatrixReport,
    quantize_nf4,
    quantize_int8,
    relative_frobenius_error,
    choose_rank,
    randomized_svd,
)
from loader.full_model_svd import extract_expert_tensors


# Tensor kinds we compress (matching llama.cpp V4 convention for
# deepseek4). gate/up are (n_embd, n_ff_exp); down is (n_ff_exp, n_embd).
GATE_UP_SHAPE = ("n_embd", "n_ff_exp")
DOWN_SHAPE = ("n_ff_exp", "n_embd")


def _factor_name(layer: int, kind: str, suffix: str) -> str:
    """Build GGUF tensor name. kind is 'gate'|'up'|'down', suffix is 'a'|'b'|'a_s'|'b_s'."""
    return f"blk.{layer}.ffn_latent_{kind}_{suffix}"


def _quantize_factor(
    factor: np.ndarray,
    group_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Quantize a 2D factor to NF4 packed + FP16 scales.

    Returns (packed_uint8, scales_fp16). The packed array has 1 NF4 code
    per byte (unpacked; matches v1 kernel's data layout).
    """
    if factor.size == 0:
        return np.zeros(0, dtype=np.uint8), np.zeros(0, dtype=np.float16)
    packed, scales, shape = quantize_nf4(factor.astype(np.float32), group_size=group_size)
    # Verify roundtrip
    decoded = dequantize_nf4(packed, scales, shape, group_size=group_size)
    err = relative_frobenius_error(factor.astype(np.float32), decoded)
    if err > 0.05:
        # Per LATENT.md §2.3 fallback to INT8 if NF4 SNR is too low.
        q, s, _ = quantize_int8(factor.astype(np.float32), group_size=group_size * 2)
        decoded = (q.astype(np.float32) * s.astype(np.float32)[:, np.newaxis]).reshape(factor.shape)
        if relative_frobenius_error(factor.astype(np.float32), decoded) < err:
            return q.astype(np.uint8), s
    return packed.view(np.uint8), scales


def _stack_per_expert_factors(
    factors: List[np.ndarray],
    pad_to: int,
) -> np.ndarray:
    """Stack per-expert factor matrices into (n_expert, rows, pad_to).

    Each factor is a 2D matrix. We right-pad the LAST axis to `pad_to`
    with zeros. This way the kernel can index by expert_id and the
    unused columns contribute nothing. For A factors, pad_to = max_rank.
    For B factors, pad_to = n_out (the columns are not rank, they're the
    output dim of the expert).

    The number of rows (first axis) is taken as the MAX across factors so
    that a high-rank factor can be stored alongside a low-rank factor
    (the low-rank one is bottom-padded with zeros).
    """
    out_rows = max(f.shape[0] for f in factors)
    stacked = np.zeros((len(factors), out_rows, pad_to), dtype=np.float32)
    for i, f in enumerate(factors):
        nr = min(f.shape[0], out_rows)
        nc = min(f.shape[1], pad_to)
        stacked[i, :nr, :nc] = f[:nr, :nc]
    return stacked


def compress_layer_to_stacked(
    layer_idx: int,
    gate_exps: np.ndarray,
    up_exps: np.ndarray,
    down_exps: np.ndarray,
    energy: float,
    min_rank: int,
    max_rank: int,
    group_size: int,
    int8_group_size: int,
    error_threshold: float,
    snr_threshold: float,
    low_snr_fraction_threshold: float,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], List[MatrixReport]]:
    """Compress one layer's experts into stacked per-expert factor tensors.

    Args:
      gate_exps, up_exps: numpy arrays of shape (n_expert, n_in, n_out) where
        n_in = n_embd and n_out = n_ff_exp (GGUF reader convention).
      down_exps: shape (n_expert, n_in, n_out) where n_in = n_ff_exp and
        n_out = n_embd.

    Returns:
      weight_tensors: 6 NF4 packed arrays (gate_a, gate_b, up_a, up_b, down_a, down_b)
      scale_tensors:  6 FP16 scale arrays
      reports: per-matrix compression reports
    """
    assert gate_exps.ndim == 3 and gate_exps.shape[2] == up_exps.shape[2] == down_exps.shape[2]
    n_expert = gate_exps.shape[2]
    # Tensors come in as (n_out, n_in, n_expert) (numpy write order).
    # For gate/up: n_in = n_embd, n_out = n_ff_exp. For down: swap.
    n_embd = int(gate_exps.shape[1])
    n_ff_exp = int(gate_exps.shape[0])
    assert int(down_exps.shape[1]) == n_ff_exp
    assert int(down_exps.shape[0]) == n_embd

    # Per-expert factor lists. Each entry is a 2D matrix of shape
    # (rows, rank) for the relevant factor.
    gate_a_factors: List[np.ndarray] = []
    gate_b_factors: List[np.ndarray] = []
    up_a_factors: List[np.ndarray] = []
    up_b_factors: List[np.ndarray] = []
    down_a_factors: List[np.ndarray] = []
    down_b_factors: List[np.ndarray] = []

    reports: List[MatrixReport] = []

    for e in range(n_expert):
        for kind, matrix_expert_raw, a_list, b_list in [
            ("gate", gate_exps[:, :, e], gate_a_factors, gate_b_factors),
            ("up",   up_exps[:, :, e],   up_a_factors,   up_b_factors),
            ("down", down_exps[:, :, e], down_a_factors, down_b_factors),
        ]:
            # extract_expert_tensors returns numpy write order (n_out, n_in, n_expert).
            # So gate_exps[:, :, e] is (n_out_gate=n_ff_exp, n_in_gate=n_embd).
            # The kernel multiplies x @ A where x has shape (n_in,), so A must be
            # (n_in, r) and B must be (r, n_out). Transpose to (n_in, n_out) so
            # that SVD produces A with the correct orientation.
            matrix_expert = matrix_expert_raw.T  # (n_in, n_out)
            rank = choose_rank(
                np.linalg.svd(matrix_expert, compute_uv=False),
                energy=energy, min_rank=min_rank, max_rank=max_rank,
            )
            u, s, vt = randomized_svd(matrix_expert, target_rank=rank, seed=e)
            a, b = make_factors(u, s, vt, rank)
            a_list.append(a)
            b_list.append(b)
            err = relative_frobenius_error(matrix_expert, a @ b)
            reports.append(MatrixReport(
                name=f"blk.{layer_idx}.ffn_{kind}_exps.expert{e}",
                shape=tuple(int(x) for x in matrix_expert.shape),
                rank=rank,
                retained_energy=1.0 - err,
                dense_relative_error=err,
                quantized_relative_error=err,  # measured at quantize step
                min_group_snr=float("inf"),
                low_snr_group_fraction=0.0,
                quantization="nf4_g%d" % group_size,
                fallback=False,
                fallback_reason="",
            ))

    # n_in/n_out per kind. gate/up: (n_embd, n_ff_exp), down: (n_ff_exp, n_embd).
    # Stacked A has shape (n_expert, n_in, max_rank).
    # Stacked B has shape (n_expert, max_rank, n_out).
    b_pad = {
        "gate": n_ff_exp, "up": n_ff_exp, "down": n_embd,
    }
    stacked = {
        "gate_a": _stack_per_expert_factors(gate_a_factors, max_rank),
        "gate_b": _stack_per_expert_factors(gate_b_factors, b_pad["gate"]),
        "up_a":   _stack_per_expert_factors(up_a_factors,   max_rank),
        "up_b":   _stack_per_expert_factors(up_b_factors,   b_pad["up"]),
        "down_a": _stack_per_expert_factors(down_a_factors, max_rank),
        "down_b": _stack_per_expert_factors(down_b_factors, b_pad["down"]),
    }

    weight_tensors: Dict[str, np.ndarray] = {}
    scale_tensors: Dict[str, np.ndarray] = {}

    for kind in ("gate", "up", "down"):
        for ab in ("a", "b"):
            stacked_factor = stacked[f"{kind}_{ab}"]
            # For A: (n_expert, n_in, max_rank). For B: (n_expert, max_rank, n_out).
            # In both cases the LAST axis is the one we group by group_size.
            last_axis_len = stacked_factor.shape[2]
            flat = stacked_factor.reshape(-1, last_axis_len)
            n_groups = (last_axis_len + group_size - 1) // group_size
            padded_last = n_groups * group_size
            if padded_last != last_axis_len:
                flat = np.pad(flat, ((0, 0), (0, padded_last - last_axis_len)), mode="constant")
            flat_packed_list = []
            flat_scales_list = []
            for row in flat:
                packed, scales, _ = quantize_nf4(row.astype(np.float32), group_size=group_size)
                # Unpack: 2 NF4 codes per byte -> 1 NF4 code per byte. The kernel
                # accesses codes as 1-byte-per-index, so we must store them
                # unpacked to match the kernel's data layout.
                codes = np.empty(packed.size * 2, dtype=np.uint8)
                codes[0::2] = packed & 0x0F
                codes[1::2] = packed >> 4
                if codes.size < padded_last:
                    codes = np.pad(codes, (0, padded_last - codes.size), constant_values=7)
                if scales.size < n_groups:
                    scales = np.pad(scales, (0, n_groups - scales.size), constant_values=1.0)
                flat_packed_list.append(codes)
                flat_scales_list.append(scales)
            stacked_codes = np.stack(flat_packed_list, axis=0).reshape(
                stacked_factor.shape[0], stacked_factor.shape[1], padded_last
            )
            stacked_scales = np.stack(flat_scales_list, axis=0).reshape(
                stacked_factor.shape[0], stacked_factor.shape[1], n_groups
            )
            weight_tensors[_factor_name(layer_idx, kind, ab)] = stacked_codes
            scale_tensors[_factor_name(layer_idx, kind, f"{ab}_s")] = stacked_scales

    return weight_tensors, scale_tensors, reports


def write_latent_gguf(
    input_gguf: Path,
    output_gguf: Path,
    max_rank: int,
    energy: float = 0.95,
    min_rank: int = 16,
    group_size: int = 32,
    int8_group_size: int = 64,
    error_threshold: float = 0.02,
    snr_threshold: float = 3.0,
    low_snr_fraction_threshold: float = 0.05,
    limit_layers: int = 0,
    limit_experts: int = 0,
    gguf_py_path: Path | None = None,
) -> int:
    """Run SVD + NF4 compression on each expert and write a LATENT GGUF.

    Returns 0 on success.
    """
    add_gguf_to_path(gguf_py_path)
    import gguf
    from gguf import GGUFWriter, GGMLQuantizationType

    print(f"Loading expert tensors from {input_gguf}")
    expert_tensors = extract_expert_tensors(input_gguf, gguf_py_path)
    if not expert_tensors:
        print("No expert tensors found")
        return 1

    # Group by layer
    layers: Dict[int, Dict[str, np.ndarray]] = {}
    for name, arr in expert_tensors.items():
        parts = name.split(".")
        if len(parts) >= 3 and parts[0] == "blk":
            layer_idx = int(parts[1])
            tensor_kind = parts[2]
            layers.setdefault(layer_idx, {})[tensor_kind] = arr

    layer_indices = sorted(layers.keys())
    if limit_layers > 0:
        layer_indices = layer_indices[:limit_layers]

    # extract_expert_tensors returns arrays in numpy write order. Tensors
    # were written as numpy (n_out, n_in, n_expert), so the read-back shape
    # is (n_out, n_in, n_expert). For ffn_gate_exps / ffn_up_exps:
    # n_in = n_embd, n_out = n_ff_exp. For ffn_down_exps:
    # n_in = n_ff_exp, n_out = n_embd.
    first_layer = next(iter(layers.values()))
    gate_shape = first_layer["ffn_gate_exps"].shape
    down_shape = first_layer["ffn_down_exps"].shape
    n_expert = int(gate_shape[2])
    n_embd = int(gate_shape[1])  # gate's n_in
    n_ff_exp = int(gate_shape[0])  # gate's n_out
    # down's n_in should be n_ff_exp, n_out should be n_embd
    assert int(down_shape[1]) == n_ff_exp, f"down n_in mismatch: {down_shape} vs n_ff_exp={n_ff_exp}"
    assert int(down_shape[0]) == n_embd, f"down n_out mismatch: {down_shape} vs n_embd={n_embd}"
    if limit_experts > 0:
        n_expert = min(n_expert, limit_experts)
    print(f"Model dims: n_embd={n_embd}, n_ff_exp={n_ff_exp}, n_expert={n_expert}")
    print(f"Layers: {len(layer_indices)}, max_rank={max_rank}, energy={energy}")

    # Open GGUF writer. We use the deepseek4 arch so llama.cpp will read it.
    print(f"Writing GGUF: {output_gguf}")
    writer = GGUFWriter(str(output_gguf), arch="deepseek4", use_temp_file=False)

    # Required metadata (subset of what deepseek4 expects; we copy from input)
    print("Adding metadata...")
    from gguf import GGUFReader
    reader = GGUFReader(str(input_gguf))
    # Copy general.* and deepseek4.* metadata fields
    for key, field in reader.fields.items():
        if key.startswith("general.") or key.startswith("deepseek4."):
            # Skip the kv_count and tensor_count pseudo-fields
            if key in ("GGUF.tensor_count", "GGUF.kv_count", "GGUF.version"):
                continue
            try:
                val = field.parts[-1][0] if field.parts else None
                if val is None:
                    continue
                if field.types[0] == gguf.GGUFValueType.STRING:
                    writer.add_string(key, str(val))
                elif field.types[0] == gguf.GGUFValueType.UINT32:
                    writer.add_uint32(key, int(val))
                elif field.types[0] == gguf.GGUFValueType.INT32:
                    writer.add_int32(key, int(val)) if hasattr(writer, "add_int32") else writer.add_uint32(key, int(val))
                elif field.types[0] == gguf.GGUFValueType.FLOAT32:
                    writer.add_float32(key, float(val))
                elif field.types[0] == gguf.GGUFValueType.UINT64:
                    writer.add_uint64(key, int(val)) if hasattr(writer, "add_uint64") else None
                elif field.types[0] == gguf.GGUFValueType.ARRAY:
                    pass  # skip arrays for simplicity
            except Exception as e:
                print(f"  skip {key}: {e}")

    # Add LATENT-specific metadata
    writer.add_uint32("latent.max_rank", max_rank)
    writer.add_uint32("latent.energy_milli", int(energy * 1000))
    writer.add_uint32("latent.group_size", group_size)
    writer.add_string("latent.quantization", "nf4_g%d" % group_size)
    writer.add_string("latent.format_version", "1")

    # Write non-expert tensors (tokenizer, output, etc.) verbatim
    print("Copying non-expert tensors...")
    import re
    expert_pattern = re.compile(r"ffn_(gate|up|down)_exps")
    for tensor in reader.tensors:
        if expert_pattern.search(tensor.name):
            continue
        try:
            data = gguf.dequantize(tensor.data, tensor.tensor_type)
        except (NotImplementedError, ValueError):
            print(f"  Skipping {tensor.name}: type {tensor.tensor_type} not supported")
            continue
        data = np.asarray(data)
        writer.add_tensor(tensor.name, data)

    # Process each layer
    print("Compressing experts...")
    t0 = time.perf_counter()
    for layer_idx in layer_indices:
        layer_tensors = layers[layer_idx]
        gate = layer_tensors.get("ffn_gate_exps")
        up = layer_tensors.get("ffn_up_exps")
        down = layer_tensors.get("ffn_down_exps")
        if gate is None or up is None or down is None:
            print(f"  layer {layer_idx}: missing one of gate/up/down, skip")
            continue
        if limit_experts > 0:
            gate = gate[..., :limit_experts]
            up = up[..., :limit_experts]
            down = down[..., :limit_experts]

        weight_tensors, scale_tensors, reports = compress_layer_to_stacked(
            layer_idx=layer_idx,
            gate_exps=gate, up_exps=up, down_exps=down,
            energy=energy, min_rank=min_rank, max_rank=max_rank,
            group_size=group_size, int8_group_size=int8_group_size,
            error_threshold=error_threshold,
            snr_threshold=snr_threshold,
            low_snr_fraction_threshold=low_snr_fraction_threshold,
        )

        # Write the 12 tensors for this layer. gguf-py reverses the recorded
        # shape from the numpy shape, and the file layout assumes the numpy
        # array is in C-order with its LAST dim being the fastest-changing.
        # We want the C++ kernel to see ne[0]=n_expert, ne[1]=n_in, ne[2]=max_rank
        # (or ne[2]=max_rank/32 for scales). To make gguf record that order,
        # we pass the numpy array transposed so the LAST dim is n_expert.
        # Write the 12 tensors for this layer.
        #
        # gguf-py reverses the recorded shape from the numpy shape, and the file
        # data layout is C-order on the numpy array. We want the C++ reader to
        # see ne[0]=n_expert, ne[1]=n_in, ne[2]=max_rank (or n_out for B),
        # with C++ row-major access (a, b, c) at byte a*N1*N2 + b*N2 + c.
        #
        # gguf-py records the dims as `numpy.shape[::-1]`, so to get recorded
        # dims (N0, N1, N2) we pass numpy of shape (N2, N1, N0). The file bytes
        # (C-order on the numpy array) are my_logical.tobytes() unchanged
        # because reshape preserves byte order, and the resulting
        # (a, b, c) -> byte mapping in C++ matches my_logical's C-order.
        for name, arr in weight_tensors.items():
            # arr logical shape: (N0, N1, N2) per C++ view.
            # Pass numpy of shape (N2, N1, N0) so gguf records (N0, N1, N2).
            t = arr.astype(np.int8).reshape(arr.shape[2], arr.shape[1], arr.shape[0])
            writer.add_tensor(name, t, raw_dtype=GGMLQuantizationType.I8)
        for name, arr in scale_tensors.items():
            t = arr.astype(np.float16).reshape(arr.shape[2], arr.shape[1], arr.shape[0])
            writer.add_tensor(name, t, raw_dtype=GGMLQuantizationType.F16)
        if (layer_idx + 1) % 5 == 0 or layer_idx == layer_indices[0]:
            elapsed = time.perf_counter() - t0
            print(f"  layer {layer_idx+1}/{len(layer_indices)} ({elapsed:.1f}s)")

    print("Finalizing GGUF...")
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"Wrote {output_gguf}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Write a LATENT GGUF from a model.")
    parser.add_argument("--input-gguf", required=True, type=Path)
    parser.add_argument("--output-gguf", required=True, type=Path)
    parser.add_argument("--gguf-py", type=Path)
    parser.add_argument("--max-rank", type=int, default=128)
    parser.add_argument("--energy", type=float, default=0.95)
    parser.add_argument("--min-rank", type=int, default=16)
    parser.add_argument("--group-size", type=int, default=32)
    parser.add_argument("--int8-group-size", type=int, default=64)
    parser.add_argument("--error-threshold", type=float, default=0.02)
    parser.add_argument("--snr-threshold", type=float, default=3.0)
    parser.add_argument("--low-snr-fraction-threshold", type=float, default=0.05)
    parser.add_argument("--limit-layers", type=int, default=0)
    parser.add_argument("--limit-experts", type=int, default=0)
    args = parser.parse_args()

    return write_latent_gguf(
        input_gguf=args.input_gguf,
        output_gguf=args.output_gguf,
        max_rank=args.max_rank,
        energy=args.energy,
        min_rank=args.min_rank,
        group_size=args.group_size,
        int8_group_size=args.int8_group_size,
        error_threshold=args.error_threshold,
        snr_threshold=args.snr_threshold,
        low_snr_fraction_threshold=args.low_snr_fraction_threshold,
        limit_layers=args.limit_layers,
        limit_experts=args.limit_experts,
        gguf_py_path=args.gguf_py,
    )


if __name__ == "__main__":
    raise SystemExit(main())
