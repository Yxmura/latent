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
- CUDA/HIP fused expert kernel source installed into the pinned V4 fork layout
- installer plus `patches/001-latent-cuda-kernel.patch`

Still requires target-fork integration:

- exact V4 Flash expert tensor-name mapping
- verified V4 Flash tensor shapes
- llama.cpp graph/op registration for LATENT factor tensors
- end-to-end quality evaluation
- real GPU kernel profiling and optimization
- ROCm/HIP compile verification on a machine with ROCm installed

Verified locally:

- `python -m pytest -q`: passing
- pinned `cchuter/llama.cpp` branch `feat/v4-port-cuda` cloned under `vendor/llama.cpp-v4`
- CPU-only llama.cpp build: passing
- CUDA llama.cpp build with LATENT source installed: passing
- HIP configure: blocked locally because CMake cannot find `hipConfig.cmake`

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

Run tests:

```bash
python -m pytest
```

## Repository Layout

```text
loader/       SVD, factorization, quantization, metadata output
kernels/      CUDA/HIP fused expert dispatch and reduce source
benchmarks/   synthetic timing and real-model measurement protocol
notes/        technical notes and integration assumptions
patches/      llama.cpp patch artifacts
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
