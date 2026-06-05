#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f "CMakeLists.txt" ]]; then
  echo "Run this from the root of a llama.cpp checkout." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_DIR="ggml/src/ggml-cuda"
TOOLS_DIR="tools/latent"

if [[ ! -d "${CUDA_DIR}" ]]; then
  echo "Expected ${CUDA_DIR}; this installer targets the cchuter/llama.cpp V4 layout." >&2
  exit 1
fi

mkdir -p "${TOOLS_DIR}"
cp "${SCRIPT_DIR}/loader/latent_loader.py"            "${TOOLS_DIR}/latent_loader.py"
cp "${SCRIPT_DIR}/loader/gguf_expert_extract.py"      "${TOOLS_DIR}/gguf_expert_extract.py"

if compgen -G "${SCRIPT_DIR}/patches/*.patch" > /dev/null; then
  for patch in "${SCRIPT_DIR}"/patches/*.patch; do
    if git apply --check "${patch}" 2>/dev/null; then
      echo "Applying ${patch}"
      git apply "${patch}"
    else
      echo "Skipping ${patch}; it does not apply cleanly, likely because LATENT files are already installed."
    fi
  done
else
  cp "${SCRIPT_DIR}/kernels/latent_fused_expert_ffn.cu" "${CUDA_DIR}/latent-fused-expert-ffn.cu"
  cp "${SCRIPT_DIR}/kernels/latent_fused_expert_ffn.h"  "${CUDA_DIR}/latent-fused-expert-ffn.cuh"
  cat <<'EOF'
LATENT files installed:

- ggml/src/ggml-cuda/latent-fused-expert-ffn.cu
- ggml/src/ggml-cuda/latent-fused-expert-ffn.cuh
- tools/latent/latent_loader.py
- tools/latent/gguf_expert_extract.py

No llama.cpp patches were applied because this repository currently contains
no graph/op rewrite patch. The CUDA and HIP CMake files glob ggml-cuda/*.cu, so
the LATENT kernel source is included in CUDA/HIP backend builds. Runtime use
still requires a graph integration patch after GGUF tensor names and factor
storage tensors are verified.
EOF
fi

cp "${SCRIPT_DIR}/kernels/latent_fused_expert_ffn.cu" "${CUDA_DIR}/latent-fused-expert-ffn.cu"
cp "${SCRIPT_DIR}/kernels/latent_fused_expert_ffn.h"  "${CUDA_DIR}/latent-fused-expert-ffn.cuh"
