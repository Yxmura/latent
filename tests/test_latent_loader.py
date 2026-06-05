import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from loader.gguf_expert_extract import tensor_to_dense
from loader.latent_loader import (
    choose_rank,
    compress_matrix,
    dequantize_nf4,
    quantize_nf4,
    relative_frobenius_error,
)
from loader.latent_runtime import dense_swiglu_expert, load_pair, swiglu_expert


def make_low_rank_matrix(rows=64, cols=48, rank=8, seed=123):
    rng = np.random.default_rng(seed)
    left = rng.standard_normal((rows, rank), dtype=np.float32)
    right = rng.standard_normal((rank, cols), dtype=np.float32)
    return left @ right


def test_choose_rank_respects_energy_and_bounds():
    singular_values = np.array([4.0, 2.0, 1.0, 0.5], dtype=np.float32)
    rank = choose_rank(singular_values, energy=0.90, min_rank=1, max_rank=4)
    assert rank == 2
    assert choose_rank(singular_values, energy=0.99, min_rank=3, max_rank=3) == 3


def test_nf4_roundtrip_preserves_shape_and_reasonable_error():
    values = np.linspace(-2.0, 2.0, 133, dtype=np.float32).reshape(7, 19)
    packed, scales, shape = quantize_nf4(values, group_size=16)
    restored = dequantize_nf4(packed, scales, shape, group_size=16)
    assert restored.shape == values.shape
    assert relative_frobenius_error(values, restored) < 0.12


def test_compress_low_rank_matrix_without_dense_fallback():
    matrix = make_low_rank_matrix()
    arrays, report = compress_matrix(
        name="blk.0.ffn_gate.0",
        matrix=matrix,
        energy=0.999,
        min_rank=4,
        max_rank=16,
        group_size=32,
        int8_group_size=64,
        error_threshold=0.35,
        snr_threshold=0.0,
        low_snr_fraction_threshold=1.0,
        seed=7,
    )
    assert report.rank >= 8
    assert report.dense_relative_error < 1e-4
    assert report.quantized_relative_error < 0.35
    assert not report.fallback
    assert "blk.0.ffn_gate.0.a.nf4" in arrays
    assert "blk.0.ffn_gate.0.dense_fallback" not in arrays


def test_cli_writes_npz_and_metadata(tmp_path: Path):
    input_path = tmp_path / "dense.npz"
    output_path = tmp_path / "latent.npz"
    metadata_path = tmp_path / "latent.json"
    np.savez(input_path, **{"blk.0.ffn_up.0": make_low_rank_matrix(32, 24, 4)})

    result = subprocess.run(
        [
            sys.executable,
            "loader/latent_loader.py",
            "--input",
            str(input_path),
            "--output",
            str(output_path),
            "--metadata",
            str(metadata_path),
            "--energy",
            "0.99",
            "--max-rank",
            "8",
            "--error-threshold",
            "0.5",
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    assert "compressed_matrices" in result.stdout
    assert output_path.exists()
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["format"] == "latent-nf4-factors-v0"
    assert metadata["assumptions"]["gguf_tensor_shapes_verified"] is False
    assert len(metadata["matrices"]) == 1


def test_cpu_reference_swiglu_matches_low_rank_artifact():
    rng = np.random.default_rng(99)
    x = rng.standard_normal((3, 16), dtype=np.float32)
    gate = make_low_rank_matrix(16, 12, 4, seed=1)
    up = make_low_rank_matrix(16, 12, 4, seed=2)
    down = make_low_rank_matrix(12, 16, 4, seed=3)

    arrays = {}
    for name, matrix in [("gate", gate), ("up", up), ("down", down)]:
        compressed, report = compress_matrix(
            name=name,
            matrix=matrix,
            energy=0.999,
            min_rank=4,
            max_rank=8,
            group_size=32,
            int8_group_size=64,
            error_threshold=1.0,
            snr_threshold=0.0,
            low_snr_fraction_threshold=1.0,
            seed=10,
        )
        assert not report.fallback
        arrays.update(compressed)

    latent = swiglu_expert(x, load_pair(arrays, "gate"), load_pair(arrays, "up"), load_pair(arrays, "down"))
    dense = dense_swiglu_expert(x, gate, up, down)
    assert relative_frobenius_error(dense, latent) < 0.35


def test_cpu_reference_supports_deepseek4_weight_before_down():
    rng = np.random.default_rng(1234)
    x = rng.standard_normal((2, 8), dtype=np.float32)
    gate = make_low_rank_matrix(8, 6, 3, seed=4)
    up = make_low_rank_matrix(8, 6, 3, seed=5)
    down = make_low_rank_matrix(6, 8, 3, seed=6)
    weight = 0.25

    before = dense_swiglu_expert(x, gate, up, down, weight=weight, weight_before_down=True)
    after = dense_swiglu_expert(x, gate, up, down, weight=1.0, weight_before_down=False) * weight

    assert np.allclose(before, after, rtol=1e-5, atol=1e-5)


def test_gguf_tensor_to_dense_uses_dequantizer():
    class FakeTensor:
        name = "blk.0.ffn_gate_exps.weight"
        tensor_type = "fake"
        data = np.array([1, 2, 3], dtype=np.uint8)

    class FakeGguf:
        @staticmethod
        def dequantize(data, tensor_type):
            assert tensor_type == "fake"
            return data.astype(np.float32) + 0.5

    dense = tensor_to_dense(FakeTensor(), FakeGguf)
    assert dense.dtype == np.float32
    assert np.allclose(dense, np.array([1.5, 2.5, 3.5], dtype=np.float32))
