# Expert Subspace Decomposition

LATENT compresses each MoE expert projection matrix into low-rank factors and
stores those factors in NF4. It is designed as a replacement for hot/cold expert
memory management.

## Implemented Prototype

`loader/latent_loader.py` accepts an NPZ file with dense 2D matrices. For each
matrix it:

1. computes singular values for rank selection
2. runs randomized SVD
3. creates `A = U * sqrt(S)` and `B = sqrt(S) * Vt`
4. quantizes A and B to NF4 with FP16 scales
5. reports dense and quantized reconstruction error
6. stores a dense FP16 fallback if the error threshold is exceeded

This deliberately avoids pretending that GGUF extraction is solved before the
target V4 Flash fork and tensor names are verified.

## Unverified Inputs

These must be measured from the actual model artifacts:

- hidden size
- intermediate size
- number of layers
- number of experts per layer
- exact gate/up/down tensor orientation
- source quantization format
- router top-k behavior

The design spec discusses likely dimensions and throughput targets, but the
implementation treats them as assumptions.

## Kernel Direction

The CUDA/HIP baseline kernel uses the row-vector convention:

```text
y = x @ W
W ~= A @ B
```

So each projection computes:

```text
latent = x @ A
output = latent @ B
```

The current kernel is scalar and portable. After correctness is established, the
next optimization steps are:

- coalesced factor layout for B reads
- warp-level reductions
- persistent dispatch across layers
- CUDA-specific and HIP-specific fast paths
- benchmark-guided shared-memory layout

## Relationship to PRISM

PRISM's DSV4 HIP work and adaptive top-k can remain useful. LATENT changes the
expert storage model: all experts are compressed and resident, so the STASIS
hot/cold loader, profiler dependency, spill buffers, and DMA pipeline are no
longer core components.
