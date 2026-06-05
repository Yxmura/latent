// HIP-specific LATENT fused expert FFN kernel
//
// Differences from the CUDA path:
//  - Uses __shfl_xor without the mask argument (HIP signature)
//  - Uses HIP's native fp16 type (__half) via hip/hip_fp16.h
//  - Avoids warp intrinsics in the hot path; all reductions are
//    done via shared memory to keep the kernel portable across
//    RDNA3 (gfx1100) and CDNA.
//
// The HIP kernel is intentionally very similar to the CUDA kernel
// because the LATENT design uses scalar FMAs and __syncthreads()
// rather than warp-level shuffles for the heavy projections. The
// only warp shuffle is in the final reduce, where shuffle is
// naturally cheaper than shared memory.

#ifdef __HIP_PLATFORM_AMD__
#include "latent-fused-expert-ffn.cuh"
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#define LATENT_HIP_WARP_SHUFFLE(v, mask) __shfl_xor(v, mask)

__device__ float latent_hip_warp_reduce_sum(float v) {
    v += LATENT_HIP_WARP_SHUFFLE(v, 16);
    v += LATENT_HIP_WARP_SHUFFLE(v, 8);
    v += LATENT_HIP_WARP_SHUFFLE(v, 4);
    v += LATENT_HIP_WARP_SHUFFLE(v, 2);
    v += LATENT_HIP_WARP_SHUFFLE(v, 1);
    return v;
}

__device__ void latent_hip_cache_x(const __half * x, __half * x_cache, int hidden_size) {
    for (int h = threadIdx.x; h < hidden_size; h += blockDim.x) {
        x_cache[h] = x[h];
    }
    __syncthreads();
}

extern "C" __global__ void latent_hip_fused_expert_dispatch(latent_dispatch_args args) {
    const int topk_slot = blockIdx.x;
    if (topk_slot >= args.top_k) {
        return;
    }

    const int expert_id = args.expert_ids[topk_slot];
    if (expert_id < 0) {
        __half * out = (__half *) args.expert_outputs + topk_slot * args.hidden_size;
        for (int h = threadIdx.x; h < args.hidden_size; h += blockDim.x) {
            out[h] = __float2half(0.0f);
        }
        return;
    }

    const latent_expert_factors expert = args.experts[expert_id];
    const int gate_rank = expert.gate_a.cols;
    const int up_rank = expert.up_a.cols;
    const int down_rank = expert.down_a.cols;

    extern __shared__ __half smem[];
    __half * x_cache = smem;
    __half * latent_gate = x_cache + args.hidden_size;
    __half * latent_up = latent_gate + gate_rank;
    float * gate = (float *) (latent_up + up_rank);
    float * up = gate + args.intermediate_size;
    float * hidden = up + args.intermediate_size;
    float * latent_down = hidden + args.intermediate_size;

    latent_hip_cache_x((const __half *) args.x, x_cache, args.hidden_size);

    // gate_a projection
    for (int r = threadIdx.x; r < gate_rank; r += blockDim.x) {
        float acc = 0.0f;
        for (int h = 0; h < args.hidden_size; ++h) {
            const float xv = __half2float(x_cache[h]);
            const float w = latent_nf4_value(expert.gate_a.data[(h * gate_rank + r) >> 1] >>
                            (((h * gate_rank + r) & 1) ? 4 : 0)) *
                            __half2float(expert.gate_a.scales[(h * gate_rank + r) / 32]);
            acc += xv * w;
        }
        latent_gate[r] = __float2half(acc);
    }
    // up_a projection
    for (int r = threadIdx.x; r < up_rank; r += blockDim.x) {
        float acc = 0.0f;
        for (int h = 0; h < args.hidden_size; ++h) {
            const float xv = __half2float(x_cache[h]);
            const int idx = h * up_rank + r;
            const float w = latent_nf4_value(expert.up_a.data[idx >> 1] >>
                            ((idx & 1) ? 4 : 0)) *
                            __half2float(expert.up_a.scales[idx / 32]);
            acc += xv * w;
        }
        latent_up[r] = __float2half(acc);
    }
    __syncthreads();

    // gate_b + up_b projections
    for (int i = threadIdx.x; i < args.intermediate_size; i += blockDim.x) {
        float g_acc = 0.0f;
        for (int r = 0; r < gate_rank; ++r) {
            const int idx = r * args.intermediate_size + i;
            const float w = latent_nf4_value(expert.gate_b.data[idx >> 1] >>
                            ((idx & 1) ? 4 : 0)) *
                            __half2float(expert.gate_b.scales[idx / 32]);
            g_acc += __half2float(latent_gate[r]) * w;
        }
        gate[i] = g_acc;
    }
    for (int i = threadIdx.x; i < args.intermediate_size; i += blockDim.x) {
        float u_acc = 0.0f;
        for (int r = 0; r < up_rank; ++r) {
            const int idx = r * args.intermediate_size + i;
            const float w = latent_nf4_value(expert.up_b.data[idx >> 1] >>
                            ((idx & 1) ? 4 : 0)) *
                            __half2float(expert.up_b.scales[idx / 32]);
            u_acc += __half2float(latent_up[r]) * w;
        }
        up[i] = u_acc;
    }
    __syncthreads();

    // SwiGLU
    for (int i = threadIdx.x; i < args.intermediate_size; i += blockDim.x) {
        float gv = gate[i];
        float silu = gv / (1.0f + __expf(-gv));
        float hv = silu * up[i];
        if (args.weight_before_down) {
            hv *= __half2float(args.expert_weights[topk_slot]);
        }
        hidden[i] = hv;
    }
    __syncthreads();

    // down_a projection
    for (int r = threadIdx.x; r < down_rank; r += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < args.intermediate_size; ++i) {
            const int idx = i * down_rank + r;
            const float w = latent_nf4_value(expert.down_a.data[idx >> 1] >>
                            ((idx & 1) ? 4 : 0)) *
                            __half2float(expert.down_a.scales[idx / 32]);
            acc += hidden[i] * w;
        }
        latent_down[r] = acc;
    }
    __syncthreads();

    // down_b projection
    __half * out = (__half *) args.expert_outputs + topk_slot * args.hidden_size;
    for (int h = threadIdx.x; h < args.hidden_size; h += blockDim.x) {
        float acc = 0.0f;
        for (int r = 0; r < down_rank; ++r) {
            const int idx = r * args.hidden_size + h;
            const float w = latent_nf4_value(expert.down_b.data[idx >> 1] >>
                            ((idx & 1) ? 4 : 0)) *
                            __half2float(expert.down_b.scales[idx / 32]);
            acc += latent_down[r] * w;
        }
        out[h] = __float2half(acc);
    }
}

extern "C" __global__ void latent_hip_expert_reduce(latent_reduce_args args) {
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

#endif // __HIP_PLATFORM_AMD__
