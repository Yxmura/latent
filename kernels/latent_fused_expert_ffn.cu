#include "latent-fused-expert-ffn.cuh"

#include <cstdint>
#include <vector>

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

extern "C" LATENT_GLOBAL void latent_fused_expert_dispatch(latent_dispatch_args args) {
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

extern "C" LATENT_GLOBAL void latent_per_expert_copy(
        const half * __restrict__ expert_outputs,
        float       * __restrict__ dst,
        int hidden_size,
        int slot) {
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= hidden_size) {
        return;
    }
    dst[slot * hidden_size + h] = __half2float(expert_outputs[slot * hidden_size + h]);
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

void ggml_cuda_latent_expert_ffn(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src_x        = dst->src[0];
    const ggml_tensor * src_ids      = dst->src[1];
    const ggml_tensor * src_weights  = dst->src[2];
    const ggml_tensor * src_gate_a   = dst->src[3];
    const ggml_tensor * src_gate_a_s = dst->src[4];
    const ggml_tensor * src_gate_b   = dst->src[5];
    const ggml_tensor * src_gate_b_s = dst->src[6];
    const ggml_tensor * src_up_a     = dst->src[7];
    const ggml_tensor * src_up_a_s   = dst->src[8];
    const ggml_tensor * src_up_b     = dst->src[9];
    const ggml_tensor * src_up_b_s   = dst->src[10];
    const ggml_tensor * src_down_a   = dst->src[11];
    const ggml_tensor * src_down_a_s = dst->src[12];
    const ggml_tensor * src_down_b   = dst->src[13];
    const ggml_tensor * src_down_b_s = dst->src[14];

    // Per-expert stacked layout produced by loader/write_latent_gguf.py:
    //   gate_a:   ne[0]=n_expert, ne[1]=n_embd,   ne[2]=padded_max_rank
    //   gate_b:   ne[0]=n_expert, ne[1]=padded_max_rank, ne[2]=padded_n_ff
    //   gate_a_s: ne[0]=n_expert, ne[1]=n_embd,   ne[2]=n_groups_a
    //   gate_b_s: ne[0]=n_expert, ne[1]=padded_max_rank, ne[2]=n_groups_b
    //   down_a:   ne[0]=n_expert, ne[1]=n_ff,     ne[2]=padded_max_rank
    //   down_b:   ne[0]=n_expert, ne[1]=padded_max_rank, ne[2]=padded_n_embd
    //   (padded_* axes are rounded up to a multiple of group_size for NF4)
    const int n_embd        = (int) src_x->ne[0];
    const int n_tokens      = (int) src_x->ne[1];
    const int n_expert_used = (int) src_ids->ne[0];
    const int n_expert      = (int) src_gate_a->ne[0];
    const int n_in_a        = (int) src_gate_a->ne[1];
    const int padded_max    = (int) src_gate_a->ne[2];
    const int padded_n_ff   = (int) src_gate_b->ne[2];
    const int n_groups_a    = (int) src_gate_a_s->ne[2];
    const int n_groups_b    = (int) src_gate_b_s->ne[2];
    const int n_groups_d_a  = (int) src_down_a_s->ne[2];
    const int n_groups_d_b  = (int) src_down_b_s->ne[2];
    const int n_ff          = padded_n_ff;  // kernel iterates the padded dim directly
    (void) n_in_a;

    GGML_ASSERT(n_embd == n_in_a);
    GGML_ASSERT(src_x->type == GGML_TYPE_F16);
    GGML_ASSERT(src_weights->type == GGML_TYPE_F16);
    GGML_ASSERT(src_ids->type == GGML_TYPE_I32);
    GGML_ASSERT(src_gate_a->type == GGML_TYPE_I8);
    GGML_ASSERT(src_gate_b->type == GGML_TYPE_I8);
    GGML_ASSERT(src_up_a->type == GGML_TYPE_I8);
    GGML_ASSERT(src_up_b->type == GGML_TYPE_I8);
    GGML_ASSERT(src_down_a->type == GGML_TYPE_I8);
    GGML_ASSERT(src_down_b->type == GGML_TYPE_I8);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    auto build_factor = [](const uint8_t * data_base,
                           const half *     scales_base,
                           int rows,
                           int cols,
                           int group_size) -> latent_nf4_matrix {
        latent_nf4_matrix m;
        m.data = data_base;
        m.scales = scales_base;
        m.rows = rows;
        m.cols = cols;
        m.group_size = group_size;
        return m;
    };

    const size_t gate_a_bytes    = (size_t) n_embd    * (size_t) padded_max;
    const size_t gate_b_bytes    = (size_t) padded_max * (size_t) padded_n_ff;
    const size_t down_a_bytes    = (size_t) n_ff      * (size_t) padded_max;
    const size_t down_b_bytes    = (size_t) padded_max * (size_t) n_embd;
    const size_t gate_a_s_values = (size_t) n_embd    * (size_t) n_groups_a;
    const size_t gate_b_s_values = (size_t) padded_max * (size_t) n_groups_b;
    const size_t down_a_s_values = (size_t) n_ff      * (size_t) n_groups_d_a;
    const size_t down_b_s_values = (size_t) padded_max * (size_t) n_groups_d_b;

    const int group_size_a = (n_groups_a > 0) ? (padded_max / n_groups_a) : padded_max;
    const int group_size_b = (n_groups_b > 0) ? (padded_n_ff / n_groups_b) : padded_n_ff;
    const int group_size_d = (n_groups_d_a > 0) ? (padded_max / n_groups_d_a) : padded_max;
    const int group_size_e = (n_groups_d_b > 0) ? (n_embd / n_groups_d_b) : n_embd;

    std::vector<latent_expert_factors> host_experts(n_expert);
    for (int e = 0; e < n_expert; ++e) {
        host_experts[e].gate_a = build_factor(
            (const uint8_t *) src_gate_a->data + (size_t) e * gate_a_bytes,
            (const half *)     src_gate_a_s->data + (size_t) e * gate_a_s_values,
            n_embd, padded_max, group_size_a);

        host_experts[e].gate_b = build_factor(
            (const uint8_t *) src_gate_b->data + (size_t) e * gate_b_bytes,
            (const half *)     src_gate_b_s->data + (size_t) e * gate_b_s_values,
            padded_max, padded_n_ff, group_size_b);

        host_experts[e].up_a = build_factor(
            (const uint8_t *) src_up_a->data + (size_t) e * gate_a_bytes,
            (const half *)     src_up_a_s->data + (size_t) e * gate_a_s_values,
            n_embd, padded_max, group_size_a);

        host_experts[e].up_b = build_factor(
            (const uint8_t *) src_up_b->data + (size_t) e * gate_b_bytes,
            (const half *)     src_up_b_s->data + (size_t) e * gate_b_s_values,
            padded_max, padded_n_ff, group_size_b);

        host_experts[e].down_a = build_factor(
            (const uint8_t *) src_down_a->data + (size_t) e * down_a_bytes,
            (const half *)     src_down_a_s->data + (size_t) e * down_a_s_values,
            n_ff, padded_max, group_size_d);

        host_experts[e].down_b = build_factor(
            (const uint8_t *) src_down_b->data + (size_t) e * down_b_bytes,
            (const half *)     src_down_b_s->data + (size_t) e * down_b_s_values,
            padded_max, n_embd, group_size_e);
    }

    latent_expert_factors * device_experts = nullptr;
    cudaMallocAsync(&device_experts, (size_t) n_expert * sizeof(latent_expert_factors), ctx.stream());
    cudaMemcpyAsync(device_experts, host_experts.data(),
                    (size_t) n_expert * sizeof(latent_expert_factors),
                    cudaMemcpyHostToDevice, ctx.stream());

    const size_t shared_mem_bytes = (size_t)(2 * max_rank + 3 * n_ff + max_rank) * sizeof(half);
    const int    threads           = 128;

    for (int t = 0; t < n_tokens; ++t) {
        const half    * x_t       = (const half *)    src_x->data       + (size_t) t * n_embd;
        const int32_t * ids_t     = (const int32_t *) src_ids->data     + (size_t) t * n_expert_used;
        const half    * weights_t = (const half *)    src_weights->data + (size_t) t * n_expert_used;
        float         * out_t     = (float *)         dst->data        + (size_t) t * n_embd * n_expert_used;

        half * device_per_expert = nullptr;
        cudaMallocAsync(&device_per_expert,
                        (size_t) n_expert_used * n_embd * sizeof(half),
                        ctx.stream());

        latent_dispatch_args dargs{};
        dargs.x = x_t;
        dargs.expert_ids = ids_t;
        dargs.expert_weights = weights_t;
        dargs.experts = device_experts;
        dargs.expert_outputs = device_per_expert;
        dargs.hidden_size = n_embd;
        dargs.intermediate_size = n_ff;
        dargs.top_k = n_expert_used;
        dargs.weight_before_down = 0;

        latent_fused_expert_dispatch<<<n_expert_used, threads, shared_mem_bytes, ctx.stream()>>>(dargs);

        const int copy_threads = 128;
        const int copy_blocks  = (n_embd + copy_threads - 1) / copy_threads;
        latent_per_expert_copy<<<copy_blocks, copy_threads, 0, ctx.stream()>>>(
            device_per_expert, out_t, n_embd, 0);
        for (int slot = 1; slot < n_expert_used; ++slot) {
            latent_per_expert_copy<<<copy_blocks, copy_threads, 0, ctx.stream()>>>(
                device_per_expert, out_t, n_embd, slot);
        }

        cudaFreeAsync(device_per_expert, ctx.stream());
    }

    cudaFreeAsync(device_experts, ctx.stream());
}
