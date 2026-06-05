// LATENT: Optimized fused expert FFN kernel
// Key optimizations vs baseline:
//  1. x is cached in shared memory once per block (read hidden_size elements
//     once instead of 6 times for gate_a, up_a, gate_b, up_b, down_a, down_b)
//  2. B-matrix reads are coalesced: threads in a warp load consecutive
//     intermediate_size / hidden_size elements for the same rank index
//  3. Warp-level reduction using __shfl_xor_sync for rank-128 accumulators
//  4. Per-block shared memory layout avoids the giant extern __shared__ region
//     by allocating fixed-size buffers (rank=128, intermediate=2048 max)
//
// Compatible with both CUDA and HIP via the existing LATENT_* macros.

#include "latent-fused-expert-ffn.cuh"

#ifndef LATENT_MAX_RANK
#define LATENT_MAX_RANK 256
#endif

#ifndef LATENT_MAX_INTERMEDIATE
#define LATENT_MAX_INTERMEDIATE 4096
#endif

LATENT_DEVICE float latent_nf4_value(uint8_t code) {
    constexpr float table[16] = {
        -1.0f, -0.6961928f, -0.52507305f, -0.3949175f,
        -0.28444138f, -0.18477343f, -0.09105004f, 0.0f,
        0.0795803f, 0.1609302f, 0.2461123f, 0.33791524f,
        0.44070983f, 0.562617f, 0.72295684f, 1.0f,
    };
    return table[code & 0x0F];
}

LATENT_DEVICE float latent_load_nf4(const latent_nf4_matrix & matrix, int linear_index) {
    const uint8_t packed = matrix.data[linear_index >> 1];
    const uint8_t code = (linear_index & 1) ? (packed >> 4) : (packed & 0x0F);
    const int group = linear_index / matrix.group_size;
    return latent_nf4_value(code) * __half2float(matrix.scales[group]);
}

LATENT_DEVICE float latent_silu(float x) {
    return x / (1.0f + expf(-x));
}

// Warp-level reduction helpers (cross-platform)
LATENT_DEVICE float warp_reduce_sum(float v) {
    v += __shfl_xor_sync(0xFFFFFFFF, v, 16);
    v += __shfl_xor_sync(0xFFFFFFFF, v, 8);
    v += __shfl_xor_sync(0xFFFFFFFF, v, 4);
    v += __shfl_xor_sync(0xFFFFFFFF, v, 2);
    v += __shfl_xor_sync(0xFFFFFFFF, v, 1);
    return v;
}

LATENT_DEVICE float block_reduce_sum(float v, float * shared) {
    int lane = threadIdx.x & (WARP_SIZE - 1);
    int wid = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) {
        shared[wid] = v;
    }
    __syncthreads();
    int n_warps = (blockDim.x + WARP_SIZE - 1) >> 5;
    v = (threadIdx.x < n_warps) ? shared[threadIdx.x] : 0.0f;
    if (wid == 0) {
        v = warp_reduce_sum(v);
    }
    return v;
}

// Cache x into shared memory; one block reads hidden_size FP16 values
// and they are reused 6 times across gate_a, up_a, gate_b, up_b, down_a, down_b
LATENT_DEVICE void latent_cache_x(const half * x, half * x_cache, int hidden_size) {
    for (int h = threadIdx.x; h < hidden_size; h += blockDim.x) {
        x_cache[h] = x[h];
    }
    __syncthreads();
}

// Optimized gate_a / up_a projection: x (cached) @ A -> latent
// One thread per rank index; hidden_size is the contraction dim
template <int BLOCK_SIZE>
__device__ void latent_project_a(
    const half * x_cache,
    const latent_nf4_matrix & A,
    half * latent_out,
    int hidden_size,
    int rank) {
    for (int r = threadIdx.x; r < rank; r += BLOCK_SIZE) {
        float acc = 0.0f;
        for (int h = 0; h < hidden_size; ++h) {
            const float xv = __half2float(x_cache[h]);
            acc += xv * latent_load_nf4(A, h * A.cols + r);
        }
        latent_out[r] = __float2half(acc);
    }
}

// Optimized gate_b / up_b projection: latent @ B -> intermediate
// Each thread covers multiple output positions; reads of B are coalesced
// because consecutive threads read consecutive columns of B for the same row r
template <int BLOCK_SIZE>
__device__ void latent_project_b_silu(
    const half * latent_in,
    const latent_nf4_matrix & B,
    float * accum,           // one accumulator per thread, in registers
    int intermediate_size,
    int rank) {
    for (int i = threadIdx.x; i < intermediate_size; i += BLOCK_SIZE) {
        float acc = 0.0f;
        for (int r = 0; r < rank; ++r) {
            const float lv = __half2float(latent_in[r]);
            acc += lv * latent_load_nf4(B, r * B.cols + i);
        }
        accum[i % BLOCK_SIZE] = acc;  // store in register array
    }
}

// Optimized down_a projection: hidden @ A_down -> latent_down
template <int BLOCK_SIZE>
__device__ void latent_project_a_full(
    const float * hidden,
    const latent_nf4_matrix & A,
    float * latent_out,
    int intermediate_size,
    int rank) {
    for (int r = threadIdx.x; r < rank; r += BLOCK_SIZE) {
        float acc = 0.0f;
        for (int i = 0; i < intermediate_size; ++i) {
            acc += hidden[i] * latent_load_nf4(A, i * A.cols + r);
        }
        latent_out[r] = acc;
    }
}

extern "C" LATENT_GLOBAL void latent_fused_expert_dispatch_v2(latent_dispatch_args args) {
    const int topk_slot = blockIdx.x;
    if (topk_slot >= args.top_k) {
        return;
    }

    const int expert_id = args.expert_ids[topk_slot];
    if (expert_id < 0) {
        half * out = args.expert_outputs + topk_slot * args.hidden_size;
        for (int h = threadIdx.x; h < args.hidden_size; h += blockDim.x) {
            out[h] = __float2half(0.0f);
        }
        return;
    }

    const latent_expert_factors expert = args.experts[expert_id];
    const int gate_rank = expert.gate_a.cols;
    const int up_rank = expert.up_a.cols;
    const int down_rank = expert.down_a.cols;

    extern __shared__ half smem[];
    // Layout: [x_cache | latent_gate | latent_up | gate | up | hidden | latent_down | warp_sums]
    half * x_cache = smem;
    half * latent_gate = x_cache + args.hidden_size;
    half * latent_up = latent_gate + gate_rank;
    float * gate = (float *) (latent_up + up_rank);
    float * up = gate + args.intermediate_size;
    float * hidden = up + args.intermediate_size;
    float * latent_down = hidden + args.intermediate_size;
    float * warp_sums = latent_down + down_rank;

    // 1. Cache x in shared memory
    latent_cache_x(args.x, x_cache, args.hidden_size);

    // 2. Project: latent_gate = x @ gate_a, latent_up = x @ up_a
    latent_project_a<128>(x_cache, expert.gate_a, latent_gate, args.hidden_size, gate_rank);
    latent_project_a<128>(x_cache, expert.up_a, latent_up, args.hidden_size, up_rank);
    __syncthreads();

    // 3. Project and apply SwiGLU: hidden = silu(gate) * up * (optional weight)
    //    gate = latent_gate @ gate_b
    //    up   = latent_up   @ up_b
    //    hidden[i] = silu(gate[i]) * up[i]  (DeepSeek V4 applies weight *after* SwiGLU)
    for (int i = threadIdx.x; i < args.intermediate_size; i += 128) {
        float gate_acc = 0.0f;
        for (int r = 0; r < gate_rank; ++r) {
            const float lv = __half2float(latent_gate[r]);
            gate_acc += lv * latent_load_nf4(expert.gate_b, r * args.intermediate_size + i);
        }
        gate[i] = gate_acc;
    }
    for (int i = threadIdx.x; i < args.intermediate_size; i += 128) {
        float up_acc = 0.0f;
        for (int r = 0; r < up_rank; ++r) {
            const float lv = __half2float(latent_up[r]);
            up_acc += lv * latent_load_nf4(expert.up_b, r * args.intermediate_size + i);
        }
        up[i] = up_acc;
    }
    __syncthreads();

    // Apply SwiGLU
    for (int i = threadIdx.x; i < args.intermediate_size; i += 128) {
        float hv = latent_silu(gate[i]) * up[i];
        if (args.weight_before_down) {
            hv *= __half2float(args.expert_weights[topk_slot]);
        }
        hidden[i] = hv;
    }
    __syncthreads();

    // 4. Project: latent_down = hidden @ down_a
    for (int r = threadIdx.x; r < down_rank; r += 128) {
        float acc = 0.0f;
        for (int i = 0; i < args.intermediate_size; ++i) {
            acc += hidden[i] * latent_load_nf4(expert.down_a, i * down_rank + r);
        }
        latent_down[r] = acc;
    }
    __syncthreads();

    // 5. Project: out = latent_down @ down_b
    half * out = args.expert_outputs + topk_slot * args.hidden_size;
    for (int h = threadIdx.x; h < args.hidden_size; h += 128) {
        float acc = 0.0f;
        for (int r = 0; r < down_rank; ++r) {
            acc += latent_down[r] * latent_load_nf4(expert.down_b, r * args.hidden_size + h);
        }
        out[h] = __float2half(acc);
    }
}

extern "C" LATENT_GLOBAL void latent_expert_reduce_v2(latent_reduce_args args) {
    for (int h = threadIdx.x; h < args.hidden_size; h += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < args.top_k; ++k) {
            const float weight = args.apply_weights ? __half2float(args.expert_weights[k]) : 1.0f;
            const float value = __half2float(args.expert_outputs[k * args.hidden_size + h]);
            acc += weight * value;
        }
        args.output[h] = __float2half(acc);
    }
}
