# LATENT Implementation Audit

Current audit against `latent.md`.

## Implemented And Verified

- SVD factorization into A/B factors with singular values absorbed.
- NF4 per-group quantization and dequantization.
- INT8 fallback path for low NF4 group SNR.
- Dense FP16 fallback metadata when reconstruction error exceeds threshold.
- Synthetic compression benchmark.
- CPU reference SwiGLU runtime.
- DeepSeek V4 weight-before-down semantics in CPU runtime and CUDA ABI.
- Variable rank per matrix in CUDA ABI using matrix shapes instead of a global rank.
- GGUF tensor listing/extraction tool with gguf-py dequantization.
- Installer for the pinned cchuter V4 fork.
- CUDA backend compile with LATENT source installed.

## Partially Implemented

- Fused CUDA/HIP kernel exists and compiles under CUDA, but is not invoked by
  llama.cpp graph execution yet.
- HIP source placement is correct for this fork because HIP globs
  `ggml/src/ggml-cuda/*.cu`, but local HIP compile is unverified because ROCm is
  not installed.
- GGUF tooling can inspect tensors and extract/dequantize tensor formats
  supported by llama.cpp's `gguf-py`.

## Not Implemented Yet

- Exact V4 Flash tensor-name mapping on a real expert GGUF.
- Runtime loading of `.npz` LATENT factor artifacts into llama.cpp tensors.
- New GGML op or graph rewrite to replace gate/up/down `MUL_MAT_ID` chains with
  LATENT factor dispatch.
- End-to-end model execution with LATENT factors.
- Full 11,008-expert compression of V4 Flash.
- Quality evaluation: perplexity, code, math, and long-context benchmarks.
- Throughput measurement on RTX 3090 or W7800.
- HIP build/run verification on AMD hardware.
- Calibration pass for KL divergence and expert rank expansion.

## Local Verification Evidence

- `python -m pytest -q`: 6 tests pass.
- `cmake --build build-latent-cpu --config Release -j 4`: llama.cpp CPU build passes.
- `cmake --build build-latent-cuda --config Release -j 4`: llama.cpp CUDA build passes with LATENT installed.
- `cmake -S . -B build-latent-hip -DGGML_HIP=ON ...`: fails before compile because `hipConfig.cmake` is missing locally.

## Completion Status

LATENT is not 100% complete. The remaining required work is graph/runtime
integration plus real model and hardware validation.
