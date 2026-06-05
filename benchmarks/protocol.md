# LATENT Benchmark Protocol

This protocol separates prototype measurements from model claims.

## Synthetic Compression

Run:

```bash
python benchmarks/latent_svd_bench.py --rows 1024 --cols 512 --intrinsic-rank 64
```

Record:

- selected rank
- dense reconstruction error
- NF4 reconstruction error
- elapsed compression time
- fallback status

## Real Expert Tensors

Before publishing throughput or quality claims, extract actual V4 Flash expert
matrices from the target GGUF/fork and verify:

- tensor names for gate, up, and down experts
- hidden and intermediate dimensions
- number of layers and experts per layer
- top-k routing count
- quantized source format and dequantization path

Then run `loader/latent_loader.py` on extracted FP16/FP32 matrices and summarize
per-layer/per-matrix reconstruction error.

## Quality

End-to-end quality must compare dense or original-quantized inference against
LATENT factors on:

- perplexity on at least one general text set
- code generation benchmark
- math/reasoning benchmark
- long-context prompt suite

Do not treat synthetic Frobenius error as a substitute for model quality.
