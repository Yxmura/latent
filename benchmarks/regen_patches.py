"""Apply all source changes to the v4 fork copy, then generate 3 patches.

This regenerates the 3 patches (002, 003, 004) from a clean v4 fork copy.
The patches cover LATENT model integration, fused kernel, and graph helper.
"""

import re
import shutil
import subprocess
import sys
from pathlib import Path

VENDOR = Path("C:/Users/marce/Documents/GitHub/latent/vendor/llama.cpp-v4")
PATCHGEN = Path("C:/Users/marce/Documents/GitHub/latent/build/patchgen/v4")
PATCHES = Path("C:/Users/marce/Documents/GitHub/latent/patches")


def reset():
    if PATCHGEN.exists():
        shutil.rmtree(PATCHGEN)
    shutil.copytree(VENDOR, PATCHGEN)


def apply_text(rel: str, old: str, new: str, *, must_replace: bool = True):
    fp = PATCHGEN / rel
    with open(fp, "r", newline="") as f:
        c = f.read()
    if old not in c:
        if must_replace:
            print(f"ERROR: old not found in {rel}: {old[:80]!r}")
            raise SystemExit(1)
        return
    c = c.replace(old, new, 1)
    with open(fp, "w", newline="") as f:
        f.write(c)


def apply_step_1_arch_h():
    """Add 6 LATENT tensor enums + LLM_KV_LATENT_MAX_RANK."""
    apply_text(
        "src/llama-arch.h",
        """    LLM_TENSOR_FFN_EXP_PROBS_B,
    LLM_TENSOR_FFN_LATENT_DOWN,
    LLM_TENSOR_FFN_LATENT_UP,
""",
        """    LLM_TENSOR_FFN_EXP_PROBS_B,
    LLM_TENSOR_FFN_LATENT_DOWN,
    LLM_TENSOR_FFN_LATENT_UP,
    LLM_TENSOR_FFN_LATENT_GATE_A,
    LLM_TENSOR_FFN_LATENT_GATE_B,
    LLM_TENSOR_FFN_LATENT_UP_A,
    LLM_TENSOR_FFN_LATENT_UP_B,
    LLM_TENSOR_FFN_LATENT_DOWN_A,
    LLM_TENSOR_FFN_LATENT_DOWN_B,
""",
    )
    apply_text(
        "src/llama-arch.h",
        "    LLM_KV_HASH_LAYER_COUNT,\n",
        "    LLM_KV_HASH_LAYER_COUNT,\n    LLM_KV_LATENT_MAX_RANK,\n",
    )


def apply_step_2_arch_cpp():
    """Add 6 LATENT tensor names + 6 LLM_TENSOR_INFOS + LLM_KV_LATENT_MAX_RANK string."""
    apply_text(
        "src/llama-arch.cpp",
        """    { LLM_TENSOR_FFN_LATENT_UP,                          "blk.%d.ffn_latent_up" },
""",
        """    { LLM_TENSOR_FFN_LATENT_UP,                          "blk.%d.ffn_latent_up" },
    { LLM_TENSOR_FFN_LATENT_GATE_A,                      "blk.%d.ffn_latent_gate_a" },
    { LLM_TENSOR_FFN_LATENT_GATE_B,                      "blk.%d.ffn_latent_gate_b" },
    { LLM_TENSOR_FFN_LATENT_UP_A,                        "blk.%d.ffn_latent_up_a" },
    { LLM_TENSOR_FFN_LATENT_UP_B,                        "blk.%d.ffn_latent_up_b" },
    { LLM_TENSOR_FFN_LATENT_DOWN_A,                      "blk.%d.ffn_latent_down_a" },
    { LLM_TENSOR_FFN_LATENT_DOWN_B,                      "blk.%d.ffn_latent_down_b" },
""",
    )
    apply_text(
        "src/llama-arch.cpp",
        """    {LLM_TENSOR_FFN_LATENT_UP,              {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
""",
        """    {LLM_TENSOR_FFN_LATENT_UP,              {LLM_TENSOR_LAYER_REPEATING, GGML_OP_MUL}},
    // LATENT per-expert stacked A/B factors (handled by fused op, not mul)
    {LLM_TENSOR_FFN_LATENT_GATE_A,          {LLM_TENSOR_LAYER_REPEATING, GGML_OP_NONE}},
    {LLM_TENSOR_FFN_LATENT_GATE_B,          {LLM_TENSOR_LAYER_REPEATING, GGML_OP_NONE}},
    {LLM_TENSOR_FFN_LATENT_UP_A,            {LLM_TENSOR_LAYER_REPEATING, GGML_OP_NONE}},
    {LLM_TENSOR_FFN_LATENT_UP_B,            {LLM_TENSOR_LAYER_REPEATING, GGML_OP_NONE}},
    {LLM_TENSOR_FFN_LATENT_DOWN_A,          {LLM_TENSOR_LAYER_REPEATING, GGML_OP_NONE}},
    {LLM_TENSOR_FFN_LATENT_DOWN_B,          {LLM_TENSOR_LAYER_REPEATING, GGML_OP_NONE}},
""",
    )
    apply_text(
        "src/llama-arch.cpp",
        """    { LLM_KV_HASH_LAYER_COUNT,                  "%s.hash_layer_count"                  },
""",
        """    { LLM_KV_HASH_LAYER_COUNT,                  "%s.hash_layer_count"                  },
    { LLM_KV_LATENT_MAX_RANK,                   "%s.latent.max_rank"                   },
""",
    )


def apply_step_3_struct():
    """Add 12 LATENT tensor struct fields + hparams n_latent_max_rank."""
    apply_text(
        "src/llama-model.h",
        """    // ff MoE latent proj
    struct ggml_tensor * ffn_latent_down = nullptr;
    struct ggml_tensor * ffn_latent_up   = nullptr;
""",
        """    // ff MoE latent proj (LATENT: per-expert stacked A/B factors, NF4 + FP16 scales)
    struct ggml_tensor * ffn_latent_gate_a   = nullptr;
    struct ggml_tensor * ffn_latent_gate_a_s = nullptr;
    struct ggml_tensor * ffn_latent_gate_b   = nullptr;
    struct ggml_tensor * ffn_latent_gate_b_s = nullptr;
    struct ggml_tensor * ffn_latent_up_a     = nullptr;
    struct ggml_tensor * ffn_latent_up_a_s   = nullptr;
    struct ggml_tensor * ffn_latent_up_b     = nullptr;
    struct ggml_tensor * ffn_latent_up_b_s   = nullptr;
    struct ggml_tensor * ffn_latent_down_a   = nullptr;
    struct ggml_tensor * ffn_latent_down_a_s = nullptr;
    struct ggml_tensor * ffn_latent_down_b   = nullptr;
    struct ggml_tensor * ffn_latent_down_b_s = nullptr;

    // ff MoE latent proj (Nemotron 3 Super dense per-layer projections)
    struct ggml_tensor * ffn_latent_down = nullptr;
    struct ggml_tensor * ffn_latent_up   = nullptr;
""",
    )
    apply_text(
        "src/llama-hparams.h",
        "    uint32_t n_attn_out_groups  = 0;\n",
        "    uint32_t n_attn_out_groups  = 0;\n    uint32_t n_latent_max_rank  = 0;\n",
    )


def apply_step_4_deepseek4_load():
    """Update deepseek4.cpp: read max_rank, create 12 tensors, switch graph build."""
    apply_text(
        "src/models/deepseek4.cpp",
        """    ml.get_key(LLM_KV_HASH_LAYER_COUNT,                  hparams.n_hash_layers);
    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS,              hparams.nextn_predict_layers, false);
""",
        """    ml.get_key(LLM_KV_HASH_LAYER_COUNT,                  hparams.n_hash_layers);
    ml.get_key(LLM_KV_LATENT_MAX_RANK,                   hparams.n_latent_max_rank, false);
    ml.get_key(LLM_KV_NEXTN_PREDICT_LAYERS,              hparams.nextn_predict_layers, false);
""",
    )
    apply_text(
        "src/models/deepseek4.cpp",
        """    const int64_t n_hc              = hparams.n_hc;
    const int64_t hc_dim            = n_hc * n_embd;
""",
        """    const int64_t n_hc              = hparams.n_hc;
    const int64_t n_latent_rank     = hparams.n_latent_max_rank;
    const int64_t hc_dim            = n_hc * n_embd;
""",
    )
    apply_text(
        "src/models/deepseek4.cpp",
        """        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);

        layer.ffn_gate_shexp""",
        """        layer.ffn_up_exps   = create_tensor(tn(LLM_TENSOR_FFN_UP_EXPS,   "weight", i), {n_embd,   n_ff_exp, n_expert}, 0);

        // LATENT: per-expert stacked A/B factors, NF4 packed + FP16 per-group scales.
        // Tensors are TENSOR_NOT_REQUIRED: dense GGUFs without them fall through
        // to the existing build_moe_ffn path. Shapes use max_rank from the
        // `latent.max_rank` metadata (or 0 if absent -> latent tensors are
        // skipped). A is the input-side factor, B is the output-side factor.
        if (n_latent_rank > 0) {
            // gate: W = A @ B with A:[n_embd, r], B:[r, n_ff_exp]
            layer.ffn_latent_gate_a   = create_tensor(tn(LLM_TENSOR_FFN_LATENT_GATE_A,   "weight", i), {n_expert, n_embd,   n_latent_rank     }, TENSOR_NOT_REQUIRED);
            layer.ffn_latent_gate_a_s = create_tensor(tn(LLM_TENSOR_FFN_LATENT_GATE_A,   "scales", i), {n_expert, n_embd,   n_latent_rank / 32}, TENSOR_NOT_REQUIRED);
            layer.ffn_latent_gate_b   = create_tensor(tn(LLM_TENSOR_FFN_LATENT_GATE_B,   "weight", i), {n_expert, n_latent_rank, n_ff_exp     }, TENSOR_NOT_REQUIRED);
            layer.ffn_latent_gate_b_s = create_tensor(tn(LLM_TENSOR_FFN_LATENT_GATE_B,   "scales", i), {n_expert, n_latent_rank, n_ff_exp / 32}, TENSOR_NOT_REQUIRED);
            // up: same shape as gate
            layer.ffn_latent_up_a     = create_tensor(tn(LLM_TENSOR_FFN_LATENT_UP_A,     "weight", i), {n_expert, n_embd,   n_latent_rank     }, TENSOR_NOT_REQUIRED);
            layer.ffn_latent_up_a_s   = create_tensor(tn(LLM_TENSOR_FFN_LATENT_UP_A,     "scales", i), {n_expert, n_embd,   n_latent_rank / 32}, TENSOR_NOT_REQUIRED);
            layer.ffn_latent_up_b     = create_tensor(tn(LLM_TENSOR_FFN_LATENT_UP_B,     "weight", i), {n_expert, n_latent_rank, n_ff_exp     }, TENSOR_NOT_REQUIRED);
            layer.ffn_latent_up_b_s   = create_tensor(tn(LLM_TENSOR_FFN_LATENT_UP_B,     "scales", i), {n_expert, n_latent_rank, n_ff_exp / 32}, TENSOR_NOT_REQUIRED);
            // down: A:[n_ff_exp, r], B:[r, n_embd]
            layer.ffn_latent_down_a   = create_tensor(tn(LLM_TENSOR_FFN_LATENT_DOWN_A,   "weight", i), {n_expert, n_ff_exp, n_latent_rank     }, TENSOR_NOT_REQUIRED);
            layer.ffn_latent_down_a_s = create_tensor(tn(LLM_TENSOR_FFN_LATENT_DOWN_A,   "scales", i), {n_expert, n_ff_exp, n_latent_rank / 32}, TENSOR_NOT_REQUIRED);
            layer.ffn_latent_down_b   = create_tensor(tn(LLM_TENSOR_FFN_LATENT_DOWN_B,   "weight", i), {n_expert, n_latent_rank, n_embd     }, TENSOR_NOT_REQUIRED);
            layer.ffn_latent_down_b_s = create_tensor(tn(LLM_TENSOR_FFN_LATENT_DOWN_B,   "scales", i), {n_expert, n_latent_rank, n_embd / 32}, TENSOR_NOT_REQUIRED);
        }

        layer.ffn_gate_shexp""",
    )


def apply_step_5_deepseek4_graph():
    """Switch build_moe_ffn call to dispatch to build_moe_ffn_latent when latent present."""
    apply_text(
        "src/models/deepseek4.cpp",
        """        ggml_tensor * moe_out = build_moe_ffn(cur,
                layer.ffn_gate_inp,
                layer.ffn_up_exps,
                layer.ffn_gate_exps,
                layer.ffn_down_exps,
                layer.ffn_exp_probs_b,
                n_expert, n_expert_used,
                LLM_FFN_SILU, hparams.expert_weights_norm,
                hparams.expert_weights_scale,
                (llama_expert_gating_func_type) hparams.expert_gating_func,
                il,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                nullptr,
                selected);""",
        """        ggml_tensor * moe_out;
        if (layer.ffn_latent_gate_a != nullptr) {
            // LATENT path: 12 stacked per-expert A/B factor tensors are
            // present, dispatch to the fused ggml_latent_expert_ffn op via
            // build_moe_ffn_latent. Routing logic is the same as dense.
            moe_out = build_moe_ffn_latent(cur,
                    layer.ffn_gate_inp,
                    layer.ffn_exp_probs_b,
                    n_expert, n_expert_used,
                    hparams.expert_weights_scale,
                    (llama_expert_gating_func_type) hparams.expert_gating_func,
                    il,
                    nullptr,
                    layer.ffn_latent_gate_a, layer.ffn_latent_gate_a_s,
                    layer.ffn_latent_gate_b, layer.ffn_latent_gate_b_s,
                    layer.ffn_latent_up_a,   layer.ffn_latent_up_a_s,
                    layer.ffn_latent_up_b,   layer.ffn_latent_up_b_s,
                    layer.ffn_latent_down_a, layer.ffn_latent_down_a_s,
                    layer.ffn_latent_down_b, layer.ffn_latent_down_b_s,
                    selected);
        } else {
            moe_out = build_moe_ffn(cur,
                    layer.ffn_gate_inp,
                    layer.ffn_up_exps,
                    layer.ffn_gate_exps,
                    layer.ffn_down_exps,
                    layer.ffn_exp_probs_b,
                    n_expert, n_expert_used,
                    LLM_FFN_SILU, hparams.expert_weights_norm,
                    hparams.expert_weights_scale,
                    (llama_expert_gating_func_type) hparams.expert_gating_func,
                    il,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    nullptr,
                    selected);
        }""",
    )


def gen_patches():
    """Generate the 3 patches by isolating hunks by file."""
    # Use git diff with file path restrictions to split patches by category
    def gen(out: Path, paths: list[str]):
        args = ["git", "-C", str(PATCHGEN), "diff", "--no-color", "--"] + paths
        result = subprocess.run(args, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ERROR: git diff failed: {result.stderr}")
            raise SystemExit(1)
        out.write_text(result.stdout)
        return out.stat().st_size

    p2 = gen(PATCHES / "002-latent-gguf-tensors.patch", [
        "src/llama-arch.h",
        "src/llama-arch.cpp",
        "src/llama-model.h",
        "src/llama-hparams.h",
        "src/models/deepseek4.cpp",
    ])
    p3 = gen(PATCHES / "003-latent-fused-kernel.patch", [
        "ggml/include/ggml.h",
        "ggml/src/ggml-backend-meta.cpp",
        "ggml/src/ggml-cpu/ggml-cpu.c",
        "ggml/src/ggml-cpu/ops.cpp",
        "ggml/src/ggml-cpu/ops.h",
        "ggml/src/ggml-cuda/ggml-cuda.cu",
        "ggml/src/ggml-cuda/latent-fused-expert-ffn.cu",
        "ggml/src/ggml-cuda/latent-fused-expert-ffn.cuh",
        "ggml/src/ggml-cuda/unary.cu",
        "ggml/src/ggml-cuda/unary.cuh",
        "ggml/src/ggml.c",
    ])
    p4 = gen(PATCHES / "004-latent-graph-integration.patch", [
        "src/llama-graph.cpp",
        "src/llama-graph.h",
    ])
    print(f"Patch 002: {p2} bytes")
    print(f"Patch 003: {p3} bytes")
    print(f"Patch 004: {p4} bytes")


def main():
    print("Resetting patchgen copy...")
    reset()
    print("Applying LATENT model changes (002)...")
    apply_step_1_arch_h()
    apply_step_2_arch_cpp()
    apply_step_3_struct()
    apply_step_4_deepseek4_load()
    apply_step_5_deepseek4_graph()
    # patch 003/004 are pre-existing; for regen we need to apply the kernel + graph changes.
    # For now, just generate patch 002 from this state. Patches 003/004 are pre-existing
    # and their content is what was already generated (and the existing 003 has whitespace
    # issues; the 004 was regenerated successfully).
    print("Generating patches...")
    gen_patches()
    print("Done.")


if __name__ == "__main__":
    main()
