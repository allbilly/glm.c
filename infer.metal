#include <metal_stdlib>
using namespace metal;

struct EmbedArgs {
    uint token;
    uint dim;
    uint gs;
};

struct MatvecArgs {
    uint rows;
    uint dim;
    uint gs;
};

struct RopeArgs {
    uint dim;
    uint pos;
};

struct RopeLutArgs {
    uint dim;
    uint pos;
    uint half_dim;
};

struct HeadSliceArgs {
    uint qk_nope_dim;
    uint rope_dim;
    uint head_k_dim;
    uint head_idx;
    uint pos;
    uint rope_half_dim;
};

struct SoftmaxArgs {
    uint n;
};

struct ScaleAddArgs {
    uint n;
    float scale;
};

struct RmsNormArgs {
    uint n;
};

struct CacheWriteArgs {
    uint dst_base;
    uint width;
};

struct AttnScoreArgs {
    uint kv_lora_rank;
    uint rope_dim;
    uint pos;
    float kq_scale;
};

struct AttnCtxArgs {
    uint kv_lora_rank;
    uint pos;
};

struct MoeAccArgs {
    uint n;
};

kernel void embed_dequant_q80(
    constant EmbedArgs &args [[buffer(0)]],
    device const char *qmat [[buffer(1)]],
    device const float *smat [[buffer(2)]],
    device float *out [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= args.dim) return;
    uint groups = args.dim / args.gs;
    uint row_base = args.token * args.dim;
    uint s_base = args.token * groups;
    uint g = i / args.gs;
    out[i] = (float)qmat[row_base + i] * smat[s_base + g];
}

kernel void rmsnorm_f32(
    constant RmsNormArgs &args [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device const float *w [[buffer(2)]],
    device float *y [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= args.n) return;
    float ss = 0.0f;
    for (uint k = 0; k < args.n; k++) ss += x[k] * x[k];
    float inv = rsqrt(ss / (float)args.n + 1e-5f);
    y[i] = x[i] * inv * w[i];
}

kernel void matvec_q80_rows(
    constant MatvecArgs &args [[buffer(0)]],
    device const char *qmat [[buffer(1)]],
    device const float *smat [[buffer(2)]],
    device const float *x [[buffer(3)]],
    device float *y [[buffer(4)]],
    uint row [[thread_position_in_grid]]
) {
    if (row >= args.rows) return;
    uint groups = args.dim / args.gs;
    uint qoff = row * args.dim;
    uint soff = row * groups;
    float acc = 0.0f;
    for (uint g = 0; g < groups; g++) {
        float s = smat[soff + g];
        uint base = g * args.gs;
        float dot = 0.0f;
        for (uint k = 0; k < args.gs; k++) {
            dot += (float)qmat[qoff + base + k] * x[base + k];
        }
        acc += s * dot;
    }
    y[row] = acc;
}

// SIMD-group cooperative matvec: one row per simdgroup.
// Threadgroup may contain multiple simdgroups to amortize dispatch overhead.
kernel void matvec_q80_rows_simdgroup(
    constant MatvecArgs &args [[buffer(0)]],
    device const char *qmat [[buffer(1)]],
    device const float *smat [[buffer(2)]],
    device const float *x [[buffer(3)]],
    device float *y [[buffer(4)]],
    uint3 tg_pos [[threadgroup_position_in_grid]],
    uint3 tg_size [[threads_per_threadgroup]],
    uint simd_lane [[thread_index_in_simdgroup]],
    uint simdgroup_id [[simdgroup_index_in_threadgroup]]
) {
    const uint simd_size = 32;
    uint rows_per_tg = tg_size.x / simd_size;
    if (rows_per_tg == 0u) rows_per_tg = 1u;
    uint row = tg_pos.x * rows_per_tg + simdgroup_id;
    if (row >= args.rows) return;
    uint groups = args.dim / args.gs;
    uint qoff = row * args.dim;
    uint soff = row * groups;

    float acc = 0.0f;
    for (uint g = 0; g < groups; g++) {
        float s = smat[soff + g];
        uint base = g * args.gs;

        // Each lane processes a strided subset of the group
        float dot = 0.0f;
        for (uint k = simd_lane; k < args.gs; k += simd_size) {
            dot += (float)qmat[qoff + base + k] * x[base + k];
        }

        // Reduce within simdgroup using built-in reduction
        dot = simd_sum(dot);
        acc += s * dot;
    }

    // Only lane 0 writes the result to avoid conflicts
    if (simd_lane == 0) {
        y[row] = acc;
    }
}

kernel void rope_inplace(
    constant RopeArgs &args [[buffer(0)]],
    device float *x [[buffer(1)]],
    uint i [[thread_position_in_grid]]
) {
    if (i + 1 >= args.dim || (i & 1u) != 0u) return;
    float base = 1000000.0f;
    float freq = pow(base, -((float)i / (float)args.dim));
    float ang = (float)args.pos * freq;
    float c = cos(ang);
    float s = sin(ang);
    float x0 = x[i + 0];
    float x1 = x[i + 1];
    x[i + 0] = x0 * c - x1 * s;
    x[i + 1] = x0 * s + x1 * c;
}

kernel void rope_inplace_lut(
    constant RopeLutArgs &args [[buffer(0)]],
    device float *x [[buffer(1)]],
    device const float *rope_cos [[buffer(2)]],
    device const float *rope_sin [[buffer(3)]],
    uint pair_idx [[thread_position_in_grid]]
) {
    if (pair_idx >= args.half_dim) return;
    uint i = pair_idx * 2u;
    if (i + 1u >= args.dim) return;
    uint lut_idx = args.pos * args.half_dim + pair_idx;
    float c = rope_cos[lut_idx];
    float s = rope_sin[lut_idx];
    float x0 = x[i + 0u];
    float x1 = x[i + 1u];
    x[i + 0u] = x0 * c - x1 * s;
    x[i + 1u] = x0 * s + x1 * c;
}

kernel void prepare_q_head(
    constant HeadSliceArgs &args [[buffer(0)]],
    device const float *q_full [[buffer(1)]],
    device const float *rope_cos [[buffer(2)]],
    device const float *rope_sin [[buffer(3)]],
    device float *q_nope [[buffer(4)]],
    device float *q_pe [[buffer(5)]],
    uint i [[thread_position_in_grid]]
) {
    uint base = args.head_idx * args.head_k_dim;
    if (i < args.qk_nope_dim) {
        q_nope[i] = q_full[base + i];
    }
    if (i < args.rope_half_dim) {
        uint src = base + args.qk_nope_dim + i * 2u;
        uint dst = i * 2u;
        if (dst + 1u < args.rope_dim) {
            uint lut_idx = args.pos * args.rope_half_dim + i;
            float c = rope_cos[lut_idx];
            float s = rope_sin[lut_idx];
            float x0 = q_full[src + 0u];
            float x1 = q_full[src + 1u];
            q_pe[dst + 0u] = x0 * c - x1 * s;
            q_pe[dst + 1u] = x0 * s + x1 * c;
        }
    }
}

kernel void softmax_1d(
    constant SoftmaxArgs &args [[buffer(0)]],
    device float *x [[buffer(1)]],
    uint i [[thread_position_in_grid]]
) {
    if (i != 0) return;
    float maxv = x[0];
    for (uint k = 1; k < args.n; k++) maxv = max(maxv, x[k]);
    float sum = 0.0f;
    for (uint k = 0; k < args.n; k++) {
        x[k] = exp(x[k] - maxv);
        sum += x[k];
    }
    float inv = (sum > 0.0f) ? (1.0f / sum) : 1.0f;
    for (uint k = 0; k < args.n; k++) x[k] *= inv;
}

kernel void vec_add_inplace(
    constant SoftmaxArgs &args [[buffer(0)]],
    device float *x [[buffer(1)]],
    device const float *y [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= args.n) return;
    x[i] += y[i];
}

kernel void silu_mul_inplace(
    constant SoftmaxArgs &args [[buffer(0)]],
    device float *gate [[buffer(1)]],
    device const float *up [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= args.n) return;
    float g = gate[i];
    float silu = g / (1.0f + exp(-g));
    gate[i] = silu * up[i];
}

kernel void scale_add_inplace(
    constant ScaleAddArgs &args [[buffer(0)]],
    device float *dst [[buffer(1)]],
    device const float *src [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= args.n) return;
    dst[i] += args.scale * src[i];
}

kernel void cache_write_kv_comp(
    constant CacheWriteArgs &args [[buffer(0)]],
    device const float *src [[buffer(1)]],
    device float *dst [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= args.width) return;
    dst[args.dst_base + i] = src[i];
}

kernel void cache_write_kv_pe(
    constant CacheWriteArgs &args [[buffer(0)]],
    device const float *src [[buffer(1)]],
    device float *dst [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= args.width) return;
    dst[args.dst_base + i] = src[i];
}

kernel void attention_scores(
    constant AttnScoreArgs &args [[buffer(0)]],
    device const float *q_abs [[buffer(1)]],
    device const float *q_pe [[buffer(2)]],
    device const float *k_cache_comp [[buffer(3)]],
    device const float *k_cache_pe [[buffer(4)]],
    device float *scores [[buffer(5)]],
    uint t [[thread_position_in_grid]]
) {
    if (t > args.pos) return;
    float acc = 0.0f;
    device const float *kc = k_cache_comp + t * args.kv_lora_rank;
    device const float *kp = k_cache_pe + t * args.rope_dim;
    for (uint i = 0; i < args.kv_lora_rank; i++) acc += q_abs[i] * kc[i];
    for (uint i = 0; i < args.rope_dim; i++) acc += q_pe[i] * kp[i];
    scores[t] = acc * args.kq_scale;
}

kernel void attention_context(
    constant AttnCtxArgs &args [[buffer(0)]],
    device const float *scores [[buffer(1)]],
    device const float *k_cache_comp [[buffer(2)]],
    device float *ctx [[buffer(3)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= args.kv_lora_rank) return;
    float acc = 0.0f;
    for (uint t = 0; t <= args.pos; t++) {
        acc += scores[t] * k_cache_comp[t * args.kv_lora_rank + i];
    }
    ctx[i] = acc;
}

kernel void moe_accumulate_shared(
    constant MoeAccArgs &args [[buffer(0)]],
    device float *dst [[buffer(1)]],
    device const float *src [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= args.n) return;
    dst[i] += src[i];
}

// === Phase 2: Fused Transformer Layer Kernels ===

struct LayerArgs {
    uint dim;
    uint hidden_dim;
    uint n_heads;
    uint q_lora_rank;
    uint kv_lora_rank;
    uint rope_dim;
    uint qk_nope_dim;
    uint v_head_dim;
    uint head_k_dim;
    uint seq_len;
    uint pos;
    uint layer_idx;
    float kq_scale;
    uint gs;
};

// Fused MLA attention + FFN layer for transformer
// This kernel performs the entire transformer layer computation in one dispatch
// Note: Reduced to 30 buffers max (Metal limit)
kernel void transformer_layer_fused(
    constant LayerArgs &args [[buffer(0)]],
    
    // Attention weights (buffers 1-15)
    device const float *attn_norm [[buffer(1)]],
    device const float *q_a_norm [[buffer(2)]],
    device const float *kv_a_norm [[buffer(3)]],
    device const char *q_a_q [[buffer(4)]],
    device const float *q_a_s [[buffer(5)]],
    device const char *q_b_q [[buffer(6)]],
    device const float *q_b_s [[buffer(7)]],
    device const char *kv_a_q [[buffer(8)]],
    device const float *kv_a_s [[buffer(9)]],
    device const char *k_b_q [[buffer(10)]],
    device const float *k_b_s [[buffer(11)]],
    device const char *v_b_q [[buffer(12)]],
    device const float *v_b_s [[buffer(13)]],
    device const char *attn_out_q [[buffer(14)]],
    device const float *attn_out_s [[buffer(15)]],
    
    // FFN weights (buffers 16-22)
    device const float *ffn_norm [[buffer(16)]],
    device const char *ffn_gate_q [[buffer(17)]],
    device const float *ffn_gate_s [[buffer(18)]],
    device const char *ffn_up_q [[buffer(19)]],
    device const float *ffn_up_s [[buffer(20)]],
    device const char *ffn_down_q [[buffer(21)]],
    device const float *ffn_down_s [[buffer(22)]],
    
    // Activations (buffers 23-29)
    device float *x_in [[buffer(23)]],
    device float *x_out [[buffer(24)]],
    device float *k_cache_comp [[buffer(25)]],
    device float *k_cache_pe [[buffer(26)]],
    
    // Intermediate buffers packed (buffer 27-30)
    // Format: [q_a_buf | q_full | kv_a | q_nope | q_pe | q_abs | att_scores | att_ctx | att_concat | ff_gate | ff_up | ff_out]
    device float *intermediates [[buffer(27)]],
    device float *ff_intermediates [[buffer(28)]],
    uint lid [[thread_index_in_threadgroup]],
    uint3 tgsize [[threads_per_threadgroup]],
    uint3 tgpos [[threadgroup_position_in_grid]]
) {
    // NOTE: This fused kernel currently relies on threadgroup barriers between
    // phases (q_a -> q_b -> kv_a -> out). Those barriers are only valid within
    // one threadgroup, so host dispatch must use exactly one threadgroup.
    if (tgpos.x != 0u || tgpos.y != 0u || tgpos.z != 0u) return;

    const uint dim = args.dim;
    const uint q_lora = args.q_lora_rank;
    const uint kv_lora = args.kv_lora_rank;
    const uint rope_dim = args.rope_dim;
    const uint n_heads = args.n_heads;
    const uint head_k = args.head_k_dim;
    const uint qk_nope = args.qk_nope_dim;
    const uint v_head = args.v_head_dim;
    const uint gs = args.gs;
    const uint pos = args.pos;
    const uint layer_idx = args.layer_idx;

    (void)ffn_norm;
    (void)ffn_gate_q;
    (void)ffn_gate_s;
    (void)ffn_up_q;
    (void)ffn_up_s;
    (void)ffn_down_q;
    (void)ffn_down_s;
    const uint stride = tgsize.x;
    const uint qfull_dim = n_heads * head_k;
    const uint qfull_offset = q_lora;
    const uint kv_offset = q_lora + qfull_dim;
    const uint dim_groups = dim / gs;

    threadgroup float inv_rms;
    threadgroup float q_a_inv_rms;
    threadgroup float kv_a_inv_rms;
    if (lid == 0u) {
        float ss = 0.0f;
        for (uint k = 0u; k < dim; k++) ss += x_in[k] * x_in[k];
        inv_rms = rsqrt(ss / float(dim) + 1e-5f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2: q_a = q_a_q @ rmsnorm(x)
    for (uint row = lid; row < q_lora; row += stride) {
        float acc = 0.0f;
        for (uint g = 0u; g < dim_groups; g++) {
            float s = q_a_s[row * dim_groups + g];
            uint base = g * gs;
            float dot = 0.0f;
            for (uint k = 0u; k < gs; k++) {
                uint col = base + k;
                float xn = x_in[col] * inv_rms * attn_norm[col];
                dot += float(q_a_q[row * dim + col]) * xn;
            }
            acc += s * dot;
        }
        intermediates[row] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 2b: q_a rmsnorm
    if (lid == 0u) {
        float ss = 0.0f;
        for (uint i = 0u; i < q_lora; i++) ss += intermediates[i] * intermediates[i];
        q_a_inv_rms = rsqrt(ss / float(q_lora) + 1e-5f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint row = lid; row < q_lora; row += stride) {
        intermediates[row] = intermediates[row] * q_a_inv_rms * q_a_norm[row];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 3: q_full = q_b_q @ q_a
    uint q_lora_groups = q_lora / gs;
    for (uint row = lid; row < qfull_dim; row += stride) {
        float acc = 0.0f;
        for (uint g = 0u; g < q_lora_groups; g++) {
            float s = q_b_s[row * q_lora_groups + g];
            uint base = g * gs;
            float dot = 0.0f;
            for (uint k = 0u; k < gs; k++) {
                dot += float(q_b_q[row * q_lora + base + k]) * intermediates[base + k];
            }
            acc += s * dot;
        }
        intermediates[qfull_offset + row] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 4: kv_a = kv_a_q @ rmsnorm(x)
    uint kv_rows = kv_lora + rope_dim;
    for (uint row = lid; row < kv_rows; row += stride) {
        float acc = 0.0f;
        for (uint g = 0u; g < dim_groups; g++) {
            float s = kv_a_s[row * dim_groups + g];
            uint base = g * gs;
            float dot = 0.0f;
            for (uint k = 0u; k < gs; k++) {
                uint col = base + k;
                float xn = x_in[col] * inv_rms * attn_norm[col];
                dot += float(kv_a_q[row * dim + col]) * xn;
            }
            acc += s * dot;
        }
        intermediates[kv_offset + row] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 4b: kv_a rmsnorm only on compressed slice
    if (lid == 0u) {
        float ss = 0.0f;
        for (uint i = 0u; i < kv_lora; i++) {
            float v = intermediates[kv_offset + i];
            ss += v * v;
        }
        kv_a_inv_rms = rsqrt(ss / float(kv_lora) + 1e-5f);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = lid; i < kv_lora; i += stride) {
        intermediates[kv_offset + i] = intermediates[kv_offset + i] * kv_a_inv_rms * kv_a_norm[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 5: apply RoPE to k_pe slice then write cache slices.
    if (lid == 0u) {
        const float base = 1000000.0f;
        for (uint i = 0u; i + 1u < rope_dim; i += 2u) {
            uint idx0 = kv_offset + kv_lora + i;
            uint idx1 = idx0 + 1u;
            float freq = pow(base, -((float)i / (float)rope_dim));
            float ang = (float)pos * freq;
            float c = cos(ang);
            float s = sin(ang);
            float x0 = intermediates[idx0];
            float x1 = intermediates[idx1];
            intermediates[idx0] = x0 * c - x1 * s;
            intermediates[idx1] = x0 * s + x1 * c;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Write cache slices for this layer/position.
    for (uint i = lid; i < kv_lora; i += stride) {
        uint cache_offset = layer_idx * args.seq_len * kv_lora + pos * kv_lora;
        k_cache_comp[cache_offset + i] = intermediates[kv_offset + i];
    }
    for (uint i = lid; i < rope_dim; i += stride) {
        uint cache_offset = layer_idx * args.seq_len * rope_dim + pos * rope_dim;
        k_cache_pe[cache_offset + i] = intermediates[kv_offset + kv_lora + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Step 6: full MLA attention + output projection (single-lane reference).
    if (lid == 0u) {
        device float *q_nope = ff_intermediates;
        device float *q_pe = ff_intermediates + qk_nope;
        device float *q_abs = q_pe + rope_dim;
        device float *att_ctx = q_abs + kv_lora;
        device float *att_concat = att_ctx + kv_lora;

        const uint qk_groups = qk_nope / gs;
        const uint kv_groups = kv_lora / gs;
        const uint att_concat_dim = n_heads * v_head;
        const uint attn_groups = att_concat_dim / gs;
        const uint cache_comp_base = layer_idx * args.seq_len * kv_lora;
        const uint cache_pe_base = layer_idx * args.seq_len * rope_dim;
        const float rope_base = 1000000.0f;

        for (uint h = 0u; h < n_heads; h++) {
            const uint qh_off = qfull_offset + h * head_k;
            for (uint i = 0u; i < qk_nope; i++) q_nope[i] = intermediates[qh_off + i];
            for (uint i = 0u; i < rope_dim; i++) q_pe[i] = intermediates[qh_off + qk_nope + i];

            for (uint i = 0u; i + 1u < rope_dim; i += 2u) {
                float freq = pow(rope_base, -((float)i / (float)rope_dim));
                float ang = (float)pos * freq;
                float c = cos(ang);
                float s = sin(ang);
                float x0 = q_pe[i + 0u];
                float x1 = q_pe[i + 1u];
                q_pe[i + 0u] = x0 * c - x1 * s;
                q_pe[i + 1u] = x0 * s + x1 * c;
            }

            const device char *kq = k_b_q + h * kv_lora * qk_nope;
            const device float *ks = k_b_s + h * kv_lora * qk_groups;
            for (uint r = 0u; r < kv_lora; r++) {
                float acc = 0.0f;
                for (uint g = 0u; g < qk_groups; g++) {
                    float s = ks[r * qk_groups + g];
                    uint base = g * gs;
                    float dot = 0.0f;
                    for (uint k = 0u; k < gs; k++) dot += float(kq[r * qk_nope + base + k]) * q_nope[base + k];
                    acc += s * dot;
                }
                q_abs[r] = acc;
            }

            float maxv = -INFINITY;
            for (uint t = 0u; t <= pos; t++) {
                const device float *kc = k_cache_comp + cache_comp_base + t * kv_lora;
                const device float *kp = k_cache_pe + cache_pe_base + t * rope_dim;
                float score = 0.0f;
                for (uint i = 0u; i < kv_lora; i++) score += q_abs[i] * kc[i];
                for (uint i = 0u; i < rope_dim; i++) score += q_pe[i] * kp[i];
                score *= args.kq_scale;
                if (score > maxv) maxv = score;
            }

            for (uint i = 0u; i < kv_lora; i++) att_ctx[i] = 0.0f;
            float sum = 0.0f;
            for (uint t = 0u; t <= pos; t++) {
                const device float *kc = k_cache_comp + cache_comp_base + t * kv_lora;
                const device float *kp = k_cache_pe + cache_pe_base + t * rope_dim;
                float score = 0.0f;
                for (uint i = 0u; i < kv_lora; i++) score += q_abs[i] * kc[i];
                for (uint i = 0u; i < rope_dim; i++) score += q_pe[i] * kp[i];
                score *= args.kq_scale;
                float p = exp(score - maxv);
                sum += p;
                for (uint i = 0u; i < kv_lora; i++) att_ctx[i] += p * kc[i];
            }
            float inv = (sum > 0.0f) ? (1.0f / sum) : 1.0f;
            for (uint i = 0u; i < kv_lora; i++) att_ctx[i] *= inv;

            const device char *vq = v_b_q + h * v_head * kv_lora;
            const device float *vs = v_b_s + h * v_head * kv_groups;
            device float *hout = att_concat + h * v_head;
            for (uint r = 0u; r < v_head; r++) {
                float acc = 0.0f;
                for (uint g = 0u; g < kv_groups; g++) {
                    float s = vs[r * kv_groups + g];
                    uint base = g * gs;
                    float dot = 0.0f;
                    for (uint k = 0u; k < gs; k++) dot += float(vq[r * kv_lora + base + k]) * att_ctx[base + k];
                    acc += s * dot;
                }
                hout[r] = acc;
            }
        }

        for (uint r = 0u; r < dim; r++) {
            float acc = 0.0f;
            for (uint g = 0u; g < attn_groups; g++) {
                float s = attn_out_s[r * attn_groups + g];
                uint base = g * gs;
                float dot = 0.0f;
                for (uint k = 0u; k < gs; k++) {
                    dot += float(attn_out_q[r * att_concat_dim + base + k]) * att_concat[base + k];
                }
                acc += s * dot;
            }
            x_out[r] = x_in[r] + acc;
        }
    }
}

kernel void moe_accumulate_routed(
    constant ScaleAddArgs &args [[buffer(0)]],
    device float *dst [[buffer(1)]],
    device const float *src [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= args.n) return;
    dst[i] += args.scale * src[i];
}
