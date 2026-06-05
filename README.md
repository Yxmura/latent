# LATENT

Low-rank Approximation of Transformer Experts using Near-lossless Truncation
for MoE inference.

LATENT stores MoE expert FFN matrices as low-rank A/B factors instead of dense
expert matrices. At load time, each dense matrix is decomposed with SVD:

```text
W ~= A @ B
A = U * sqrt(S)
B = sqrt(S) * Vt
```

The factors are quantized to NF4 with per-group FP16 scales. During inference,
a fused expert kernel can dequantize factors on the fly and compute the expert
FFN without materializing the dense expert matrix in VRAM.

## Current Status

This repository is a buildable LATENT prototype, not a completed llama.cpp
integration.

Implemented now:

- load-time SVD/NF4 compression prototype for extracted dense tensors
- NF4 pack/unpack with per-group scales
- INT8 per-group fallback when NF4 group SNR is poor
- GGUF expert tensor inspection/extraction with gguf-py dequantization
- reconstruction error reporting and dense fallback metadata
- CPU reference runtime for SwiGLU expert reconstruction
- synthetic benchmark protocol
- full-model SVD compression pipeline (per-layer, per-expert)
- PRISM-style adaptive top-k threshold dropping
- end-to-end validation pipeline (dense vs LATENT runtime comparison)
- CUDA/HIP fused expert kernel source installed into the pinned V4 fork layout
- three llama.cpp patch artifacts in `patches/`:
  - `001-latent-cuda-kernel.patch`: adds the v1 fused kernel source
    (`.cu` + `.cuh`)
  - `002-latent-gguf-tensors.patch`: adds the missing
    `LLM_TENSOR_FFN_LATENT_GATE` enum, tensor name, struct field, and
    optional `TENSOR_NOT_REQUIRED` loading for all three latent
    projections in DeepSeek V4
  - the v2 and HIP kernels are installed by `install.sh` (new files only)

Still requires target-fork integration:

- exact V4 Flash expert tensor-name mapping
- verified V4 Flash tensor shapes
- llama.cpp build verification with the patch series applied on real hardware
- end-to-end quality evaluation against dense V4 reference
- real GPU kernel profiling and optimization (v2 kernel is unverified)
- ROCm/HIP compile verification on a machine with ROCm installed

Verified locally:

- `python -m pytest -q`: 21/21 passing
- pinned `cchuter/llama.cpp` branch `feat/v4-port-cuda` cloned under `vendor/llama.cpp-v4`
- CPU-only llama.cpp build: passing
- CUDA llama.cpp build with LATENT source installed: passing
- HIP configure: blocked locally because CMake cannot find `hipConfig.cmake`

## Patch Series

The `patches/` directory contains two patches that integrate LATENT into
the pinned V4 llama.cpp fork. They apply cleanly to a clean checkout of
`cchuter/llama.cpp` at `feat/v4-port-cuda` (verified with
`git apply --check`):

1. **001-latent-cuda-kernel.patch** — adds the v1 fused kernel source files
   (`ggml/src/ggml-cuda/latent-fused-expert-ffn.cu` and `.cuh`). This is
   self-contained: it adds new files, modifies no existing source.

2. **002-latent-gguf-tensors.patch** — adds the missing
   `LLM_TENSOR_FFN_LATENT_GATE` enum, registers its tensor name in
   `llama-arch.cpp`, adds the `ffn_latent_gate` field to the layer struct
   in `llama-model.h`, and adds optional `TENSOR_NOT_REQUIRED` loading of
   all three `ffn_latent_{gate,up,down}` projections in
   `models/deepseek4.cpp`. Models without LATENT tensors load unchanged.

The optimized v2 kernel (`kernels/latent_fused_expert_ffn_v2.cu`) and the
HIP kernel (`kernels/latent_fused_expert_ffn_hip.cu`) are installed by
`install.sh` rather than via patch — they add new files only and don't
need source-tree changes.

**Not yet implemented in patch form:**

- A graph builder function (`build_moe_ffn_latent`) in
  `src/llama-graph.cpp` that takes LATENT factor tensors and dispatches
  the fused kernel. This is the missing link between the kernel and the
  DeepSeek V4 model graph. Adding it requires ~150 lines of C++ in
  critical llama.cpp internals; deferred pending user-side authoring.
- A `GGML_OP_LATENT_EXPERT_FFN` entry. The v4 fork has
  `static_assert(GGML_OP_COUNT == 102, ...)` in three places, so adding
  a new op is invasive. The current design calls the kernel directly
  from C++ via the host wrapper in the .cuh header, bypassing the GGML
  op table.

## Why This Replaces STASIS

STASIS keeps a profiled hot set of experts in VRAM and swaps cold experts from
CPU RAM. LATENT aims to keep all experts resident by compressing every expert
into low-rank factors. That removes hot/cold configuration, profiling, PCIe
miss stalls, and spill buffers.

Several STASIS numbers are treated here as unverified assumptions, including
hot expert distributions, miss rates, exact V4 tensor dimensions, and throughput
targets. LATENT records these as integration measurements rather than constants.

## Prototype Usage

Create an NPZ file containing dense 2D matrices, then compress it:

```bash
python loader/latent_loader.py \
  --input extracted_experts.npz \
  --output build/latent_factors.npz \
  --metadata build/latent_factors.json \
  --energy 0.95 \
  --max-rank 128
```

Inspect expert tensors in a GGUF file:

```bash
python loader/gguf_expert_extract.py --gguf model.gguf --list-only
```

Extract supported expert tensors to dense NPZ:

```bash
python loader/gguf_expert_extract.py --gguf model.gguf --output build/extracted_experts.npz
```

Quantized GGUF tensors are dequantized through llama.cpp's `gguf-py` helpers.
If `gguf-py` has no dequantizer for a tensor type, extraction fails explicitly.

Run the synthetic benchmark:

```bash
python benchmarks/latent_svd_bench.py
```

Run the end-to-end validation (compares LATENT runtime output to dense
SwiGLU reference across multiple intrinsic ranks and configurations):

```bash
python benchmarks/end_to_end_validation.py \\
  --rows 4096 --cols 2048 --max-rank 128
```

Run the memory savings benchmark (theoretical compression bound for
synthetic experts):

```bash
python benchmarks/memory_savings_bench.py \\
  --hidden-size 4096 --intermediate-size 2048 \\
  --n-expert 256 --max-rank 128 --intrinsic-rank 64
```

Validate the patch files:

```bash
python benchmarks/validate_patches.py
```

Run the full-model SVD pipeline (requires a real V4 GGUF):

```bash
python loader/full_model_svd.py \\
  --gguf model.gguf \\
  --output build/latent_factors \\
  --max-rank 128
```

Run tests:

```bash
python -m pytest
```

## Repository Layout

```text
loader/       SVD, factorization, quantization, runtime, adaptive top-k
kernels/      CUDA/HIP fused expert dispatch and reduce source
              (v1 baseline, v2 optimized with shared-mem x cache, HIP variant)
benchmarks/   synthetic timing, end-to-end validation, real-model measurement
notes/        technical notes and integration assumptions
patches/      llama.cpp patch artifacts (002/003/004)
tests/        Python unit tests
```

## Integration Contract

The loader emits `latent-nf4-factors-v0` NPZ artifacts:

- `<name>.a.nf4`, `<name>.a.scales`, `<name>.a.shape`
- `<name>.b.nf4`, `<name>.b.scales`, `<name>.b.shape`
- optional `<name>.dense_fallback` when reconstruction error exceeds threshold

The kernel ABI expects `latent_expert_factors` structs containing the six
factor matrices for gate, up, and down projections.

DeepSeek V4 applies router weights before the down projection. LATENT's CUDA
ABI and CPU reference runtime support this via `weight_before_down`; other MoE
graphs can use output weighting during reduction.

## Measurement Rules

Do not publish LATENT quality or throughput claims until these are measured on
real target hardware and model tensors:

- per-matrix spectral decay
- reconstruction error after NF4
- end-to-end perplexity/benchmark deltas
- tokens/sec at multiple context lengths
- CUDA and HIP numerical equivalence

The original `latent.md` remains the concept/spec document. This README is the
current build state.
