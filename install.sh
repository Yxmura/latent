#!/usr/bin/env bash
set -euo pipefail

if [[ ! -f "CMakeLists.txt" ]]; then
  echo "Run this from the root of a llama.cpp checkout." >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_DIR="ggml/src/ggml-cuda"
HIP_DIR="ggml/src/ggml-hip"
TOOLS_DIR="tools/latent"

if [[ ! -d "${CUDA_DIR}" ]]; then
  echo "Expected ${CUDA_DIR}; this installer targets the cchuter/llama.cpp V4 layout." >&2
  exit 1
fi

mkdir -p "${TOOLS_DIR}"
cp "${SCRIPT_DIR}/loader/latent_loader.py"            "${TOOLS_DIR}/latent_loader.py"
cp "${SCRIPT_DIR}/loader/gguf_expert_extract.py"      "${TOOLS_DIR}/gguf_expert_extract.py"
cp "${SCRIPT_DIR}/loader/full_model_svd.py"           "${TOOLS_DIR}/full_model_svd.py"
cp "${SCRIPT_DIR}/loader/adaptive_topk.py"            "${TOOLS_DIR}/adaptive_topk.py"
cp "${SCRIPT_DIR}/loader/latent_runtime.py"           "${TOOLS_DIR}/latent_runtime.py"
cp "${SCRIPT_DIR}/benchmarks/end_to_end_validation.py" "${TOOLS_DIR}/end_to_end_validation.py"
cp "${SCRIPT_DIR}/benchmarks/memory_savings_bench.py"  "${TOOLS_DIR}/memory_savings_bench.py"
cp "${SCRIPT_DIR}/benchmarks/validate_patches.py"      "${TOOLS_DIR}/validate_patches.py"

if compgen -G "${SCRIPT_DIR}/patches/*.patch" > /dev/null; then
  for patch in "${SCRIPT_DIR}"/patches/*.patch; do
    if git apply --check "${patch}" 2>/dev/null; then
      echo "Applying ${patch}"
      git apply "${patch}"
    else
      echo "Skipping ${patch}; it does not apply cleanly, likely because LATENT files are already installed."
    fi
  done
fi

# Install the LATENT kernel into both CUDA and HIP backends.
cp "${SCRIPT_DIR}/kernels/latent_fused_expert_ffn.cu" "${CUDA_DIR}/latent-fused-expert-ffn.cu"
cp "${SCRIPT_DIR}/kernels/latent_fused_expert_ffn.h"  "${CUDA_DIR}/latent-fused-expert-ffn.cuh"
cp "${SCRIPT_DIR}/kernels/latent_fused_expert_ffn_v2.cu" "${CUDA_DIR}/latent-fused-expert-ffn-v2.cu"
if [[ -d "${HIP_DIR}" ]]; then
  cp "${SCRIPT_DIR}/kernels/latent_fused_expert_ffn_hip.cu" "${HIP_DIR}/latent-fused-expert-ffn.hip"
  echo "Installed HIP kernel to ${HIP_DIR}/latent-fused-expert-ffn.hip"
fi

cat <<'EOF'
LATENT files installed:

- ggml/src/ggml-cuda/latent-fused-expert-ffn.cu
- ggml/src/ggml-cuda/latent-fused-expert-ffn.cuh
- ggml/src/ggml-cuda/latent-fused-expert-ffn-v2.cu (optimized)
- ggml/src/ggml-hip/latent-fused-expert-ffn.hip (AMD ROCm)
- tools/latent/latent_loader.py
- tools/latent/gguf_expert_extract.py
- tools/latent/full_model_svd.py
- tools/latent/adaptive_topk.py
- tools/latent/latent_runtime.py
- tools/latent/end_to_end_validation.py

The CUDA and HIP CMake files glob ggml-cuda/*.cu and ggml-hip/*.hip, so the
LATENT kernel sources are included in CUDA/HIP backend builds. Any patches
from patches/ that applied cleanly have been applied to the llama.cpp source
tree (tensor enums, GGML op registration, graph integration).
EOF
