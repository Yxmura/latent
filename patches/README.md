# Patch Artifacts

`001-latent-cuda-kernel.patch` adds the LATENT fused expert CUDA/HIP source to
the pinned cchuter V4 fork layout under `ggml/src/ggml-cuda`.

Further patch work should wait until these facts are verified from a real V4
Flash GGUF and runtime:

- backend op signatures
- CUDA/HIP source layout
- GGUF expert tensor names and orientation
- graph node used for MoE expert FFN
- command-line flag conventions

Expected future patches:

- `002-latent-factor-loader.patch`
- `003-latent-graph-op.patch`
- `004-latent-cli-flags.patch`
