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
        for (uint k = 0; k < args.gs; k++) dot += (float)qmat[qoff + base + k] * x[base + k];
        acc += s * dot;
    }
    y[row] = acc;
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

kernel void moe_accumulate_routed(
    constant ScaleAddArgs &args [[buffer(0)]],
    device float *dst [[buffer(1)]],
    device const float *src [[buffer(2)]],
    uint i [[thread_position_in_grid]]
) {
    if (i >= args.n) return;
    dst[i] += args.scale * src[i];
}
