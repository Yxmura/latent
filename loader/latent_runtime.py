"""CPU reference runtime for LATENT factor artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from loader.latent_loader import dequantize_int8, dequantize_nf4


@dataclass
class FactorPair:
    a: np.ndarray
    b: np.ndarray


def _load_factor(arrays: Mapping[str, np.ndarray], name: str, side: str) -> np.ndarray:
    shape = tuple(int(x) for x in arrays[f"{name}.{side}.shape"])
    scales = arrays[f"{name}.{side}.scales"]
    nf4_key = f"{name}.{side}.nf4"
    int8_key = f"{name}.{side}.int8"
    if nf4_key in arrays:
        return dequantize_nf4(arrays[nf4_key], scales, shape, group_size=32)
    if int8_key in arrays:
        return dequantize_int8(arrays[int8_key], scales, shape, group_size=64)
    raise KeyError(f"missing {name}.{side} factor")


def load_pair(arrays: Mapping[str, np.ndarray], name: str) -> FactorPair:
    return FactorPair(a=_load_factor(arrays, name, "a"), b=_load_factor(arrays, name, "b"))


def project(x: np.ndarray, pair: FactorPair) -> np.ndarray:
    return (np.asarray(x, dtype=np.float32) @ pair.a) @ pair.b


def silu(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    # Compute in float64 to avoid overflow on large negative x
    x64 = x.astype(np.float64)
    result = x64 / (1.0 + np.exp(-x64))
    # Clamp to finite values for safety
    return np.clip(result, -1e30, 1e30).astype(np.float32)


def swiglu_expert(
    x: np.ndarray,
    gate: FactorPair,
    up: FactorPair,
    down: FactorPair,
    weight: float = 1.0,
    weight_before_down: bool = False,
) -> np.ndarray:
    hidden = silu(project(x, gate)) * project(x, up)
    if weight_before_down:
        hidden = hidden * np.float32(weight)
    return project(hidden, down)


def dense_swiglu_expert(
    x: np.ndarray,
    gate: np.ndarray,
    up: np.ndarray,
    down: np.ndarray,
    weight: float = 1.0,
    weight_before_down: bool = False,
) -> np.ndarray:
    hidden = silu(np.asarray(x, dtype=np.float32) @ gate) * (np.asarray(x, dtype=np.float32) @ up)
    if weight_before_down:
        hidden = hidden * np.float32(weight)
    return hidden @ down
