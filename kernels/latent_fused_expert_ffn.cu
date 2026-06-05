#include "latent-fused-expert-ffn.cuh"

LATENT_DEVICE float latent_nf4_value(uint8_t code) {
    // NF4 codebook from QLoRA/bitsandbytes. Keep this in sync with loader.
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

extern "C" LATENT_GLOBAL void latent_fused_expert_dispatch(latent_dispatch_args args) {
    // One block per selected expert. This portable baseline intentionally uses
    // scalar accumulation so CUDA and HIP can share one implementation while
    // graph integration and tensor naming are pinned down.
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
    half * latent_gate = smem;
    half * latent_up = latent_gate + gate_rank;
    half * gate = latent_up + up_rank;
    half * up = gate + args.intermediate_size;
    half * hidden = up + args.intermediate_size;
    half * latent_down = hidden + args.intermediate_size;

    for (int r = threadIdx.x; r < gate_rank; r += blockDim.x) {
        float gate_acc = 0.0f;
        for (int h = 0; h < args.hidden_size; ++h) {
            const float xv = __half2float(args.x[h]);
            gate_acc += xv * latent_load_nf4(expert.gate_a, h * gate_rank + r);
        }
        latent_gate[r] = __float2half(gate_acc);
    }
    for (int r = threadIdx.x; r < up_rank; r += blockDim.x) {
        float up_acc = 0.0f;
        for (int h = 0; h < args.hidden_size; ++h) {
            const float xv = __half2float(args.x[h]);
            up_acc += xv * latent_load_nf4(expert.up_a, h * up_rank + r);
        }
        latent_up[r] = __float2half(up_acc);
    }
    LATENT_SYNC;

    for (int i = threadIdx.x; i < args.intermediate_size; i += blockDim.x) {
        float gate_acc = 0.0f;
        float up_acc = 0.0f;
        for (int r = 0; r < gate_rank; ++r) {
            gate_acc += __half2float(latent_gate[r]) * latent_load_nf4(expert.gate_b, r * args.intermediate_size + i);
        }
        for (int r = 0; r < up_rank; ++r) {
            up_acc += __half2float(latent_up[r]) * latent_load_nf4(expert.up_b, r * args.intermediate_size + i);
        }
        gate[i] = __float2half(gate_acc);
        up[i] = __float2half(up_acc);
        float hidden_value = latent_silu(gate_acc) * up_acc;
        if (args.weight_before_down) {
            hidden_value *= __half2float(args.expert_weights[topk_slot]);
        }
        hidden[i] = __float2half(hidden_value);
    }
    LATENT_SYNC;

    for (int r = threadIdx.x; r < down_rank; r += blockDim.x) {
        float acc = 0.0f;
        for (int i = 0; i < args.intermediate_size; ++i) {
            acc += __half2float(hidden[i]) * latent_load_nf4(expert.down_a, i * down_rank + r);
        }
        latent_down[r] = __float2half(acc);
    }
    LATENT_SYNC;

    half * out = args.expert_outputs + topk_slot * args.hidden_size;
    for (int h = threadIdx.x; h < args.hidden_size; h += blockDim.x) {
        float acc = 0.0f;
        for (int r = 0; r < down_rank; ++r) {
            acc += __half2float(latent_down[r]) * latent_load_nf4(expert.down_b, r * args.hidden_size + h);
        }
        out[h] = __float2half(acc);
    }
}

extern "C" LATENT_GLOBAL void latent_expert_reduce(latent_reduce_args args) {
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
