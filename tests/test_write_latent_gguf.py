"""End-to-end test: build a small synthetic deepseek4-shaped GGUF, run
the LATENT writer on it, and verify the output has the expected structure
and reconstruction quality.
"""

from __future__ import annotations

import gc
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import gguf
from loader.latent_loader import dequantize_nf4
from loader.write_latent_gguf import write_latent_gguf


def make_synthetic_deepseek4_gguf(path: str) -> None:
    """Build a small GGUF with deepseek4 architecture and 2 expert layers.

    GGUF stores shapes in reversed order from numpy. We want the reader to see:
        blk.{L}.ffn_gate_exps : [n_expert=4, n_in=32, n_out=16]
    so we write the numpy array with shape (n_out, n_in, n_expert) = (16, 32, 4).

    Each expert is generated with intrinsic rank 4 so the SVD + NF4 path can
    reconstruct it well at max_rank=4.
    """
    n_embd = 32
    n_ff_exp = 16
    n_expert = 4
    n_layer = 2
    intrinsic_rank = 4
    rng = np.random.default_rng(42)

    def make_experts(n_in: int, n_out: int, k: int) -> np.ndarray:
        arr = np.zeros((n_out, n_in, n_expert), dtype=np.float32)
        for e in range(n_expert):
            left = rng.standard_normal((n_out, k), dtype=np.float32)
            right = rng.standard_normal((k, n_in), dtype=np.float32)
            arr[:, :, e] = left @ right
        return arr

    w = gguf.GGUFWriter(path, arch="deepseek4", use_temp_file=False)
    w.add_uint32("deepseek4.block_count", n_layer)
    w.add_uint32("deepseek4.embedding_length", n_embd)
    w.add_uint32("deepseek4.feed_forward_length", n_ff_exp)
    w.add_uint32("deepseek4.expert_count", n_expert)
    w.add_uint32("deepseek4.expert_used_count", 2)
    w.add_uint32("deepseek4.attention.head_count", 4)
    w.add_uint32("deepseek4.attention.head_count_kv", 4)
    w.add_float32("deepseek4.attention.layer_norm_rms_epsilon", 1e-6)
    w.add_string("general.name", "synthetic-ds4-test")
    w.add_tensor("blk.0.ffn_gate_inp", np.random.randn(n_embd, n_expert).astype(np.float32))
    w.add_tensor("blk.1.ffn_gate_inp", np.random.randn(n_embd, n_expert).astype(np.float32))
    for layer in range(n_layer):
        w.add_tensor(f"blk.{layer}.ffn_gate_exps",
                     make_experts(n_embd, n_ff_exp, intrinsic_rank))
        w.add_tensor(f"blk.{layer}.ffn_up_exps",
                     make_experts(n_embd, n_ff_exp, intrinsic_rank))
        w.add_tensor(f"blk.{layer}.ffn_down_exps",
                     make_experts(n_ff_exp, n_embd, intrinsic_rank))
        w.add_tensor(f"blk.{layer}.ffn_norm",
                     (rng.standard_normal((n_embd,)).astype(np.float32) * 0.1 + 1.0))
    w.add_tensor("output_norm", (rng.standard_normal((n_embd,)).astype(np.float32) * 0.1 + 1.0))
    w.add_tensor("output", (rng.standard_normal((1024, n_embd)).astype(np.float32) * 0.01))
    w.add_string("tokenizer.ggml.model", "gpt2")
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def test_synthetic_roundtrip(tmpdir: str) -> None:
    src = os.path.join(tmpdir, "src.gguf")
    dst = os.path.join(tmpdir, "latent.gguf")
    make_synthetic_deepseek4_gguf(src)
    print(f"  src: {os.path.getsize(src)} bytes")

    rc = write_latent_gguf(
        input_gguf=Path(src),
        output_gguf=Path(dst),
        max_rank=4,
        energy=0.99,
        min_rank=2,
        group_size=4,
    )
    assert rc == 0, f"write_latent_gguf returned {rc}"
    assert os.path.getsize(dst) > 0, "empty output"
    print(f"  dst: {os.path.getsize(dst)} bytes")

    r = gguf.GGUFReader(dst)
    assert "latent.max_rank" in r.fields, "missing latent.max_rank"
    assert r.fields["latent.max_rank"].parts[-1][0] == 4
    assert "latent.energy_milli" in r.fields
    tensor_names = {t.name for t in r.tensors}
    tensors_by_name = {t.name: t for t in r.tensors}
    for layer in range(2):
        for kind in ("gate", "up", "down"):
            for ab in ("a", "b"):
                assert f"blk.{layer}.ffn_latent_{kind}_{ab}" in tensor_names, \
                    f"missing blk.{layer}.ffn_latent_{kind}_{ab}"
            # Per-expert ranks uint8 tensor: (n_expert, 3)
            assert f"blk.{layer}.ffn_latent_ranks" in tensor_names, \
                f"missing blk.{layer}.ffn_latent_ranks"
    # Validate tensor shapes match the per-expert stacked layout consumed
    # by the CUDA kernel in kernels/latent_fused_expert_ffn.cu.
    # gate/up: gate_a   ne=[n_expert, n_embd, padded_max_rank // 2]  (packed, 2 codes/byte)
    #          gate_a_s ne=[n_expert, n_embd, n_groups_a]
    #          gate_b   ne=[n_expert, padded_max_rank, padded_n_ff // 2]
    #          gate_b_s ne=[n_expert, padded_max_rank, n_groups_b]
    # down:     down_a   ne=[n_expert, n_ff, padded_max_rank // 2]
    #          down_b   ne=[n_expert, padded_max_rank, n_embd // 2]
    n_expert = 4
    n_embd = 32
    n_ff = 16
    max_rank = 4
    n_groups = max_rank // 4  # group_size=4
    for layer in range(2):
        for kind in ("gate", "up"):
            a = tensors_by_name[f"blk.{layer}.ffn_latent_{kind}_a"]
            a_s = tensors_by_name[f"blk.{layer}.ffn_latent_{kind}_a_s"]
            b = tensors_by_name[f"blk.{layer}.ffn_latent_{kind}_b"]
            b_s = tensors_by_name[f"blk.{layer}.ffn_latent_{kind}_b_s"]
            assert tuple(a.shape) == (n_expert, n_embd, max_rank // 2), \
                f"{a.name} shape {a.shape} != {(n_expert, n_embd, max_rank // 2)}"
            assert tuple(a_s.shape) == (n_expert, n_embd, n_groups), \
                f"{a_s.name} shape {a_s.shape} != {(n_expert, n_embd, n_groups)}"
            assert tuple(b.shape) == (n_expert, max_rank, n_ff // 2), \
                f"{b.name} shape {b.shape} != {(n_expert, max_rank, n_ff // 2)}"
            assert tuple(b_s.shape) == (n_expert, max_rank, n_ff // 4), \
                f"{b_s.name} shape {b_s.shape} != {(n_expert, max_rank, n_ff // 4)}"
            assert a.tensor_type.name == "I8"
            assert b.tensor_type.name == "I8"
            assert a_s.tensor_type.name == "F16"
            assert b_s.tensor_type.name == "F16"
        for kind in ("down",):
            a = tensors_by_name[f"blk.{layer}.ffn_latent_{kind}_a"]
            a_s = tensors_by_name[f"blk.{layer}.ffn_latent_{kind}_a_s"]
            b = tensors_by_name[f"blk.{layer}.ffn_latent_{kind}_b"]
            b_s = tensors_by_name[f"blk.{layer}.ffn_latent_{kind}_b_s"]
            assert tuple(a.shape) == (n_expert, n_ff, max_rank // 2), \
                f"{a.name} shape {a.shape} != {(n_expert, n_ff, max_rank // 2)}"
            assert tuple(a_s.shape) == (n_expert, n_ff, n_groups), \
                f"{a_s.name} shape {a_s.shape} != {(n_expert, n_ff, n_groups)}"
            assert tuple(b.shape) == (n_expert, max_rank, n_embd // 2), \
                f"{b.name} shape {b.shape} != {(n_expert, max_rank, n_embd // 2)}"
            assert tuple(b_s.shape) == (n_expert, max_rank, n_embd // 4), \
                f"{b_s.name} shape {b_s.shape} != {(n_expert, max_rank, n_embd // 4)}"
    # Check per-expert ranks tensor: (n_expert, 3) uint8
    for layer in range(2):
        r = tensors_by_name[f"blk.{layer}.ffn_latent_ranks"]
        assert tuple(r.shape) == (n_expert, 3), \
            f"{r.name} shape {r.shape} != {(n_expert, 3)}"
        assert r.tensor_type.name == "I8"
        ranks_data = np.asarray(r.data, dtype=np.uint8).reshape(r.shape)
        assert np.all(ranks_data >= 2), f"ranks too small: {ranks_data.min()}"
        assert np.all(ranks_data <= max_rank), f"ranks exceed max_rank: {ranks_data}"
    assert "output_norm" in tensor_names
    assert "output" in tensor_names
    assert "blk.0.ffn_norm" in tensor_names
    assert "blk.0.ffn_gate_exps" not in tensor_names
    print("  structure OK")


def test_synthetic_reconstruction(tmpdir: str) -> None:
    """Verify the writer's factor reconstruction matches the original experts.

    We re-read the latent GGUF, dequantize the factor tensors, multiply A @ B,
    and compare to the original dense expert matrix.
    """
    src = os.path.join(tmpdir, "src.gguf")
    dst = os.path.join(tmpdir, "latent.gguf")
    make_synthetic_deepseek4_gguf(src)
    write_latent_gguf(
        input_gguf=Path(src),
        output_gguf=Path(dst),
        max_rank=4,
        energy=0.99,
        min_rank=2,
        group_size=4,
    )

    src_reader = gguf.GGUFReader(src)
    orig = {t.name: t for t in src_reader.tensors}
    dst_reader = gguf.GGUFReader(dst)
    lat = {t.name: t for t in dst_reader.tensors}

    max_rank = 4
    group_size = 4
    n_groups_a = max_rank // group_size

    for layer in range(2):
        for kind, orig_name, n_in, n_out in [
            ("gate", f"blk.{layer}.ffn_gate_exps", 32, 16),
            ("up",   f"blk.{layer}.ffn_up_exps",   32, 16),
            ("down", f"blk.{layer}.ffn_down_exps", 16, 32),
        ]:
            orig_data = gguf.dequantize(orig[orig_name].data, orig[orig_name].tensor_type)
            orig_data = np.transpose(orig_data, (2, 1, 0))  # (n_expert, n_in, n_out)
            n_expert = orig_data.shape[0]

            a_name = f"blk.{layer}.ffn_latent_{kind}_a"
            b_name = f"blk.{layer}.ffn_latent_{kind}_b"
            a_s_name = f"blk.{layer}.ffn_latent_{kind}_a_s"
            b_s_name = f"blk.{layer}.ffn_latent_{kind}_b_s"

            a_codes = np.asarray(lat[a_name].data, dtype=np.uint8).reshape(lat[a_name].shape)
            b_codes = np.asarray(lat[b_name].data, dtype=np.uint8).reshape(lat[b_name].shape)
            a_scales = np.asarray(lat[a_s_name].data, dtype=np.float16).reshape(lat[a_s_name].shape)
            b_scales = np.asarray(lat[b_s_name].data, dtype=np.float16).reshape(lat[b_s_name].shape)

            for e in range(n_expert):
                # Storage layout: 2 NF4 codes per byte (packed). We must unpack them
                # into 1-code-per-byte before dequantizing.
                from loader.latent_loader import NF4_CODEBOOK

                # Unpack gate_a_codes (each byte -> 2 codes)
                a_packed = a_codes[e]  # (n_in, padded_rank // 2)
                a_codes_e = np.empty((a_packed.shape[0], a_packed.shape[1] * 2), dtype=np.uint8)
                a_codes_e[:, 0::2] = a_packed & 0x0F
                a_codes_e[:, 1::2] = a_packed >> 4

                # Unpack gate_b_codes
                b_packed = b_codes[e]  # (padded_rank, n_out // 2)
                b_codes_e = np.empty((b_packed.shape[0], b_packed.shape[1] * 2), dtype=np.uint8)
                b_codes_e[:, 0::2] = b_packed & 0x0F
                b_codes_e[:, 1::2] = b_packed >> 4

                a_scales_e = a_scales[e]  # (n_in, n_groups_a)
                b_scales_e = b_scales[e]  # (padded_rank, n_groups_b)

                a_decoded = np.zeros((a_codes_e.shape[0], max_rank), dtype=np.float32)
                for row in range(a_codes_e.shape[0]):
                    for g in range(a_scales_e.shape[1]):
                        s = a_codes_e[row, g * group_size : (g + 1) * group_size]
                        a_decoded[row, g * group_size : (g + 1) * group_size] = (
                            NF4_CODEBOOK[s] * a_scales_e[row, g]
                        )
                b_decoded = np.zeros(b_codes_e.shape, dtype=np.float32)
                n_groups_b = b_scales_e.shape[1]
                group_size_b = b_codes_e.shape[1] // n_groups_b if n_groups_b else b_codes_e.shape[1]
                for row in range(b_codes_e.shape[0]):
                    for g in range(n_groups_b):
                        s = b_codes_e[row, g * group_size_b : (g + 1) * group_size_b]
                        b_decoded[row, g * group_size_b : (g + 1) * group_size_b] = (
                            NF4_CODEBOOK[s] * b_scales_e[row, g]
                        )
                # Truncate A back to max_rank (the padding columns are zero anyway)
                a_decoded = a_decoded[:, :max_rank]
                W_recon = a_decoded @ b_decoded
                W_orig = orig_data[e]
                err = np.linalg.norm(W_recon - W_orig) / max(np.linalg.norm(W_orig), 1e-9)
                # With max_rank=4 (full), energy=0.99, group_size=4, NF4 round-trip
                # error should be modest. Allow 20% to absorb the test's group_size
                # quantization on tiny matrices.
                assert err < 0.20, f"{orig_name}.expert{e} reconstruction err={err:.4f}"
    print("  reconstruction OK")


def main() -> int:
    rc = 1
    with tempfile.TemporaryDirectory() as tmp:
        try:
            test_synthetic_roundtrip(tmp)
            test_synthetic_reconstruction(tmp)
            print("PASS")
            rc = 0
        finally:
            gc.collect()
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
