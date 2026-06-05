#pragma once

#include <stdint.h>

#if defined(__HIP_PLATFORM_AMD__)
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#define LATENT_DEVICE __device__
#define LATENT_GLOBAL __global__
#define LATENT_SYNC __syncthreads()
#else
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#define LATENT_DEVICE __device__
#define LATENT_GLOBAL __global__
#define LATENT_SYNC __syncthreads()
#endif

struct latent_nf4_matrix {
    const uint8_t * data;
    const half * scales;
    int rows;
    int cols;
    int group_size;
};

struct latent_expert_factors {
    latent_nf4_matrix gate_a;
    latent_nf4_matrix gate_b;
    latent_nf4_matrix up_a;
    latent_nf4_matrix up_b;
    latent_nf4_matrix down_a;
    latent_nf4_matrix down_b;
};

struct latent_dispatch_args {
    const half * x;
    const int32_t * expert_ids;
    const half * expert_weights;
    const latent_expert_factors * experts;
    half * expert_outputs;
    int hidden_size;
    int intermediate_size;
    int top_k;
    int weight_before_down;
};

struct latent_reduce_args {
    const half * expert_outputs;
    const half * expert_weights;
    half * output;
    int hidden_size;
    int top_k;
    int apply_weights;
};

extern "C" LATENT_GLOBAL void latent_fused_expert_dispatch(latent_dispatch_args args);
extern "C" LATENT_GLOBAL void latent_expert_reduce(latent_reduce_args args);
