#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#if defined(_OPENMP)
#include <omp.h>
#endif

#include "model.h"
#include "backend.h"

static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_queue = nil;
static id<MTLLibrary> g_library = nil;
static NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *g_pipelines = nil;
static NSMutableDictionary<NSValue *, id<MTLBuffer>> *g_nocopy_cache = nil;
static id<MTLComputePipelineState> g_ps_matvec = nil;
static id<MTLBuffer> g_matvec_args = nil;

static int g_cfg_native = -1;
static int g_cfg_hybrid_matvec = -1;
static int g_cfg_native_strict = -1;
static int g_cfg_native_unsafe = -1;
static int g_cfg_native_ffn_cpu = -1;
static int g_cfg_native_parity = -1;
static int g_cfg_native_parity_all_pos = -1;
static int g_cfg_native_parity_layers = -1;
static int g_cfg_native_parity_stage = -2;
static float g_cfg_native_parity_tol = -1.0f;
static int g_warned_native_requires_unsafe = 0;
static int g_omp_tuned = 0;

// Persistent command buffer for batching matvec operations
static id<MTLCommandBuffer> g_persistent_cb = nil;
static id<MTLComputeCommandEncoder> g_persistent_encoder = nil;
static int g_matvec_batch_count = 0;
static const int g_matvec_batch_flush_threshold = 64;  // Flush after N matvecs to avoid timeout

static int env_flag_cached(const char *name, int *slot, int default_value) {
    if (!slot) return default_value;
    if (*slot >= 0) return *slot;
    const char *v = getenv(name);
    if (!v || !v[0]) {
        *slot = default_value ? 1 : 0;
        return *slot;
    }
    if (strcmp(v, "0") == 0 || strcmp(v, "false") == 0 || strcmp(v, "False") == 0 || strcmp(v, "FALSE") == 0) {
        *slot = 0;
    } else {
        *slot = 1;
    }
    return *slot;
}

static int env_int_cached(const char *name, int *slot, int default_value) {
    if (!slot) return default_value;
    if (*slot >= 0) return *slot;
    const char *v = getenv(name);
    if (!v || !v[0]) {
        *slot = default_value;
        return *slot;
    }
    *slot = atoi(v);
    return *slot;
}

static float env_float_cached(const char *name, float *slot, float default_value) {
    if (!slot) return default_value;
    if (*slot >= 0.0f) return *slot;
    const char *v = getenv(name);
    if (!v || !v[0]) {
        *slot = default_value;
        return *slot;
    }
    *slot = (float)atof(v);
    return *slot;
}

static int metal_native_enabled(void) {
    return env_flag_cached("GLM_METAL_NATIVE", &g_cfg_native, 0);
}

static int metal_hybrid_matvec_enabled(void) {
    return env_flag_cached("GLM_METAL_HYBRID_MATVEC", &g_cfg_hybrid_matvec, 0);
}

static int metal_native_strict(void) {
    return env_flag_cached("GLM_METAL_NATIVE_STRICT", &g_cfg_native_strict, 0);
}

static int metal_native_unsafe_enabled(void) {
    return env_flag_cached("GLM_METAL_NATIVE_UNSAFE", &g_cfg_native_unsafe, 0);
}

static int metal_native_ffn_cpu_enabled(void) {
    return env_flag_cached("GLM_METAL_NATIVE_FFN_CPU", &g_cfg_native_ffn_cpu, 1);
}

static int metal_native_parity_enabled(void) {
    return env_flag_cached("GLM_METAL_NATIVE_PARITY", &g_cfg_native_parity, 0);
}

static int metal_native_parity_all_pos(void) {
    return env_flag_cached("GLM_METAL_NATIVE_PARITY_ALL_POS", &g_cfg_native_parity_all_pos, 0);
}

static int metal_native_parity_layers(void) {
    return env_int_cached("GLM_METAL_NATIVE_PARITY_LAYERS", &g_cfg_native_parity_layers, 0);
}

enum {
    GLM_NATIVE_PARITY_STAGE_FFN = 0,
    GLM_NATIVE_PARITY_STAGE_ATTN = 1,
    GLM_NATIVE_PARITY_STAGE_BOTH = 2,
};

static int metal_native_parity_stage(void) {
    if (g_cfg_native_parity_stage != -2) return g_cfg_native_parity_stage;
    const char *v = getenv("GLM_METAL_NATIVE_PARITY_STAGE");
    if (!v || !v[0]) {
        g_cfg_native_parity_stage = GLM_NATIVE_PARITY_STAGE_BOTH;
        return g_cfg_native_parity_stage;
    }
    if (strcmp(v, "attn") == 0 || strcmp(v, "attention") == 0) {
        g_cfg_native_parity_stage = GLM_NATIVE_PARITY_STAGE_ATTN;
    } else if (strcmp(v, "ffn") == 0 || strcmp(v, "layer") == 0) {
        g_cfg_native_parity_stage = GLM_NATIVE_PARITY_STAGE_FFN;
    } else {
        g_cfg_native_parity_stage = GLM_NATIVE_PARITY_STAGE_BOTH;
    }
    return g_cfg_native_parity_stage;
}

static float metal_native_parity_tol(void) {
    return env_float_cached("GLM_METAL_NATIVE_PARITY_TOL", &g_cfg_native_parity_tol, 1e-3f);
}

static void metal_tune_omp_if_needed(void) {
#if defined(_OPENMP)
    if (g_omp_tuned) return;
    int threads = 0;
    const char *metal_threads = getenv("GLM_METAL_OMP_THREADS");
    if (metal_threads && metal_threads[0]) {
        threads = atoi(metal_threads);
    } else {
        const char *omp_threads = getenv("OMP_NUM_THREADS");
        if (!omp_threads || !omp_threads[0]) threads = 4;
    }
    if (threads > 0) {
        omp_set_num_threads(threads);
        fprintf(stderr, "[glm-metal] using OMP threads=%d for CPU-compatible path\n", threads);
    }
    g_omp_tuned = 1;
#endif
}

typedef struct {
    id<MTLBuffer> x;
    id<MTLBuffer> xn;
    id<MTLBuffer> xb;
    id<MTLBuffer> ff_gate;
    id<MTLBuffer> ff_up;
    id<MTLBuffer> logits;
    id<MTLBuffer> k_cache_comp;
    id<MTLBuffer> k_cache_pe;
    id<MTLBuffer> q_a;
    id<MTLBuffer> q_full;
    id<MTLBuffer> kv_a;
    id<MTLBuffer> q_nope;
    id<MTLBuffer> q_pe;
    id<MTLBuffer> q_abs;
    id<MTLBuffer> att_scores;
    id<MTLBuffer> att_ctx;
    id<MTLBuffer> att_concat;
    id<MTLBuffer> layer_intermediates;
    id<MTLBuffer> layer_ff_intermediates;
    id<MTLBuffer> moe_gate;
    id<MTLBuffer> moe_up;
    id<MTLBuffer> moe_router;
    id<MTLBuffer> moe_out_acc;
    id<MTLBuffer> moe_topk_idx;
    id<MTLBuffer> moe_topk_w;
} MetalRunStateBuffers;

// Layer-specific GPU buffers
typedef struct {
    id<MTLBuffer> attn_norm;
    id<MTLBuffer> q_a_norm;
    id<MTLBuffer> kv_a_norm;
    id<MTLBuffer> q_a_q;
    id<MTLBuffer> q_a_s;
    id<MTLBuffer> q_b_q;
    id<MTLBuffer> q_b_s;
    id<MTLBuffer> kv_a_q;
    id<MTLBuffer> kv_a_s;
    id<MTLBuffer> k_b_q;
    id<MTLBuffer> k_b_s;
    id<MTLBuffer> v_b_q;
    id<MTLBuffer> v_b_s;
    id<MTLBuffer> attn_out_q;
    id<MTLBuffer> attn_out_s;
    id<MTLBuffer> ffn_norm;
    id<MTLBuffer> gate_inp;
    id<MTLBuffer> exp_probs_b;
    id<MTLBuffer> gate_sh_q;
    id<MTLBuffer> gate_sh_s;
    id<MTLBuffer> up_sh_q;
    id<MTLBuffer> up_sh_s;
    id<MTLBuffer> down_sh_q;
    id<MTLBuffer> down_sh_s;
} MetalLayerBuffers;

typedef struct {
    id<MTLBuffer> output_norm;
    id<MTLBuffer> tok_q;
    id<MTLBuffer> tok_s;
    id<MTLBuffer> out_q;
    id<MTLBuffer> out_s;
    id<MTLBuffer> l0_ffn_norm;
    id<MTLBuffer> l0_ffn_gate_q;
    id<MTLBuffer> l0_ffn_gate_s;
    id<MTLBuffer> l0_ffn_up_q;
    id<MTLBuffer> l0_ffn_up_s;
    id<MTLBuffer> l0_ffn_down_q;
    id<MTLBuffer> l0_ffn_down_s;
} MetalModelBuffers;

static MetalRunStateBuffers g_run = {0};
static MetalModelBuffers g_model = {0};
static MetalLayerBuffers *g_layers = NULL;
static int g_n_layers = 0;

static id<MTLBuffer> make_shared_buffer(NSUInteger length, const char *name) {
    if (length == 0) return nil;
    id<MTLBuffer> buf = [g_device newBufferWithLength:length options:MTLResourceStorageModeShared];
    if (!buf) {
        fprintf(stderr, "[glm-metal] failed to allocate buffer %s (%lu bytes)\n", name, (unsigned long)length);
    }
    return buf;
}

static id<MTLBuffer> make_nocopy_buffer(const void *ptr, NSUInteger length, const char *name) {
    if (!ptr || length == 0) return nil;
    id<MTLBuffer> buf = [g_device newBufferWithBytesNoCopy:(void *)ptr
                                                    length:length
                                                   options:MTLResourceStorageModeShared
                                               deallocator:nil];
    if (!buf) {
        fprintf(stderr, "[glm-metal] failed to create model buffer %s\n", name);
    }
    return buf;
}

static id<MTLBuffer> cached_nocopy_buffer(const void *ptr, NSUInteger length, const char *name) {
    if (!ptr || length == 0) return nil;
    NSValue *key = [NSValue valueWithPointer:(void *)ptr];
    id<MTLBuffer> cached = g_nocopy_cache[key];
    if (cached != nil && cached.length >= length) return cached;
    id<MTLBuffer> buf = [g_device newBufferWithBytesNoCopy:(void *)ptr
                                                    length:length
                                                   options:MTLResourceStorageModeShared
                                               deallocator:nil];
    if (!buf) {
        fprintf(stderr, "[glm-metal] failed to create cached buffer %s\n", name);
        return nil;
    }
    g_nocopy_cache[key] = buf;
    return buf;
}

// Upload quantized weights to GPU (copy since weights are read-only)
static id<MTLBuffer> upload_weight_buffer(const void *host_ptr, NSUInteger length, const char *name) {
    if (!host_ptr || length == 0) return nil;
    id<MTLBuffer> buf = [g_device newBufferWithBytes:host_ptr length:length options:MTLResourceStorageModeShared];
    if (!buf) {
        fprintf(stderr, "[glm-metal] failed to upload weight buffer %s (%lu bytes)\n", name, (unsigned long)length);
        return nil;
    }
    return buf;
}

static double monotonic_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static int cmp_double_asc(const void *a, const void *b) {
    const double da = *(const double *)a;
    const double db = *(const double *)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

static double percentile(const double *vals, int n, double pct) {
    if (!vals || n <= 0) return 0.0;
    if (pct <= 0.0) return vals[0];
    if (pct >= 100.0) return vals[n - 1];
    double rank = (pct / 100.0) * (double)(n - 1);
    int lo = (int)rank;
    int hi = lo + 1;
    if (hi >= n) return vals[n - 1];
    double frac = rank - (double)lo;
    return vals[lo] * (1.0 - frac) + vals[hi] * frac;
}

static int ensure_pipeline(const char *name) {
    NSString *kname = [NSString stringWithUTF8String:name];
    id<MTLComputePipelineState> ps = g_pipelines[kname];
    if (ps != nil) return 0;

    id<MTLFunction> fn = [g_library newFunctionWithName:kname];
    if (!fn) {
        fprintf(stderr, "[glm-metal] missing kernel: %s\n", name);
        return -1;
    }
    NSError *err = nil;
    ps = [g_device newComputePipelineStateWithFunction:fn error:&err];
    if (!ps) {
        fprintf(stderr, "[glm-metal] failed to build kernel %s: %s\n", name, err.localizedDescription.UTF8String);
        return -1;
    }
    g_pipelines[kname] = ps;
    return 0;
}

static void *pipeline_lookup_or_crash(const char *name) {
    NSString *kname = [NSString stringWithUTF8String:name];
    id<MTLComputePipelineState> ps = g_pipelines[kname];
    if (ps == nil) {
        fprintf(stderr, "[glm-metal] fatal: kernel not loaded: %s\n", name);
        abort();
    }
    return (__bridge void *)ps;
}

static void dispatch2(id<MTLComputeCommandEncoder> encoder, const char *name,
                      unsigned int tgx, unsigned int tgy, unsigned int tgs,
                      const void *params, size_t params_size, void **buffers, size_t buffer_count) {
    id<MTLComputePipelineState> ps = (__bridge id<MTLComputePipelineState>)pipeline_lookup_or_crash(name);
    [encoder setComputePipelineState:ps];
    if (params && params_size > 0) {
        [encoder setBytes:params length:params_size atIndex:0];
    }
    static const NSUInteger offsets[32] = {0};
    if (buffer_count > 0) {
        if (buffer_count > 31) {
            fprintf(stderr, "[glm-metal] too many buffers for dispatch: %zu\n", buffer_count);
            abort();
        }
        id<MTLBuffer> __unsafe_unretained mtl_buffers[31];
        for (size_t i = 0; i < buffer_count; i++) {
            mtl_buffers[i] = (__bridge id<MTLBuffer>)buffers[i];
        }
        [encoder setBuffers:mtl_buffers offsets:offsets withRange:NSMakeRange(1, (NSUInteger)buffer_count)];
    }
    [encoder dispatchThreadgroups:MTLSizeMake(tgx, tgy, 1) threadsPerThreadgroup:MTLSizeMake(tgs, 1, 1)];
}

static void dispatch(id<MTLComputeCommandEncoder> encoder, const char *name,
                     unsigned int tgs_count, unsigned int tgs,
                     const void *params, size_t params_size, void **buffers, size_t buffer_count) {
    dispatch2(encoder, name, tgs_count, 1, tgs, params, params_size, buffers, buffer_count);
}

// CPU helper functions for native forward
static void rmsnorm_cpu(float *o, const float *x, const float *w, int n) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * (double)x[i];
    float mean = (float)(ss / (double)n);
    float inv = 1.0f / sqrtf(mean + 1e-5f);
    for (int i = 0; i < n; i++) o[i] = x[i] * inv * w[i];
}

static void matvec_q80_cpu(const int8_t *qmat, const float *smat, const float *x, float *y, int rows, int dim, int gs) {
    int groups = dim / gs;
    for (int r = 0; r < rows; r++) {
        float acc = 0.0f;
        for (int g = 0; g < groups; g++) {
            float s = smat[r * groups + g];
            int base = g * gs;
            float dot = 0.0f;
            for (int k = 0; k < gs; k++) {
                dot += (float)qmat[r * dim + base + k] * x[base + k];
            }
            acc += s * dot;
        }
        y[r] = acc;
    }
}

static void matvec_f32_rows_cpu(const float *a, const float *x, float *y, int rows, int cols) {
    for (int r = 0; r < rows; r++) {
        const float *row = a + (size_t)r * (size_t)cols;
        float acc = 0.0f;
        for (int c = 0; c < cols; c++) acc += row[c] * x[c];
        y[r] = acc;
    }
}

static inline float silu_cpu(float x) {
    return x / (1.0f + expf(-x));
}

static void apply_l0_dense_ffn_cpu(const Runtime *rt, RunState *s, int gs) {
    if (!rt || !s || !rt->has_l0_ffn) return;
    const int dim = rt->cfg.dim;
    const int hidden = rt->cfg.hidden_dim;
    rmsnorm_cpu(s->xn, s->x, rt->l0_ffn_norm, dim);
    matvec_q80_cpu(rt->l0_ffn_gate_q, rt->l0_ffn_gate_s, s->xn, s->ff_gate, hidden, dim, gs);
    matvec_q80_cpu(rt->l0_ffn_up_q, rt->l0_ffn_up_s, s->xn, s->ff_up, hidden, dim, gs);
    for (int i = 0; i < hidden; i++) s->ff_gate[i] = silu_cpu(s->ff_gate[i]) * s->ff_up[i];
    matvec_q80_cpu(rt->l0_ffn_down_q, rt->l0_ffn_down_s, s->ff_gate, s->xb, dim, hidden, gs);
    for (int i = 0; i < dim; i++) s->x[i] += s->xb[i];
}

static void apply_moe_shared_ffn_layer_cpu(const Runtime *rt, RunState *s, int gs, int layer_idx) {
    if (!rt || !s) return;
    if (!rt->has_moe_shared || layer_idx <= 0 || layer_idx >= rt->cfg.n_layers) return;
    if (rt->cfg.moe_ffn_dim <= 0) return;

    const LayerMoeShared *w = &rt->moe_layers[layer_idx];
    if (!w || !w->ffn_norm) return;

    const int dim = rt->cfg.dim;
    const int moe_dim = rt->cfg.moe_ffn_dim;
    const int n_routed = rt->cfg.n_routed_experts;
    const int top_k = rt->cfg.n_experts_used;

    rmsnorm_cpu(s->xn, s->x, w->ffn_norm, dim);

    matvec_f32_rows_cpu(w->gate_inp, s->xn, s->moe_router, n_routed, dim);
    for (int e = 0; e < n_routed; e++) {
        float logit = s->moe_router[e];
        s->moe_router[e] = 1.0f / (1.0f + expf(-logit));
    }

    for (int k = 0; k < top_k; k++) {
        s->moe_topk_idx[k] = -1;
        s->moe_topk_w[k] = -1e30f;
    }
    for (int e = 0; e < n_routed; e++) {
        float p = s->moe_router[e];
        float sel = p + w->exp_probs_b[e];
        int insert_pos = -1;
        for (int k = 0; k < top_k; k++) {
            int idx = s->moe_topk_idx[k];
            float cur_sel = (idx >= 0) ? (s->moe_router[idx] + w->exp_probs_b[idx]) : -INFINITY;
            if (sel > cur_sel) {
                insert_pos = k;
                break;
            }
        }
        if (insert_pos < 0) continue;
        for (int k = top_k - 1; k > insert_pos; k--) {
            s->moe_topk_idx[k] = s->moe_topk_idx[k - 1];
            s->moe_topk_w[k] = s->moe_topk_w[k - 1];
        }
        s->moe_topk_idx[insert_pos] = e;
        s->moe_topk_w[insert_pos] = p;
    }

    float top_sum = 0.0f;
    for (int k = 0; k < top_k; k++) if (s->moe_topk_idx[k] >= 0) top_sum += s->moe_topk_w[k];
    if (top_sum > 0.0f) {
        float denom = top_sum < 6.103515625e-5f ? 6.103515625e-5f : top_sum;
        float inv = 1.0f / denom;
        for (int k = 0; k < top_k; k++) s->moe_topk_w[k] *= inv;
    }

    memset(s->moe_out_acc, 0, (size_t)dim * sizeof(float));

    int disable_routed = 0;
    const char *disable_env = getenv("GLM_DISABLE_MOE_ROUTED");
    if (disable_env && disable_env[0] && disable_env[0] != '0') disable_routed = 1;

    if (!disable_routed && rt->has_moe_routed && rt->moe_routed_layers) {
        const LayerMoeRouted *rw = &rt->moe_routed_layers[layer_idx];
        if (rw->gate_q && rw->gate_s && rw->up_q && rw->up_s && rw->down_q && rw->down_s) {
            const int dim_groups = dim / gs;
            const int moe_groups = moe_dim / gs;
            const float routed_scale = 1.8f;
            for (int k = 0; k < top_k; k++) {
                int e = s->moe_topk_idx[k];
                if (e < 0) continue;
                float w_e = s->moe_topk_w[k] * routed_scale;

                const int8_t *gq = rw->gate_q + (size_t)e * (size_t)moe_dim * (size_t)dim;
                const float *gsr = rw->gate_s + (size_t)e * (size_t)moe_dim * (size_t)dim_groups;
                const int8_t *uq = rw->up_q + (size_t)e * (size_t)moe_dim * (size_t)dim;
                const float *usr = rw->up_s + (size_t)e * (size_t)moe_dim * (size_t)dim_groups;
                const int8_t *dq = rw->down_q + (size_t)e * (size_t)dim * (size_t)moe_dim;
                const float *dsr = rw->down_s + (size_t)e * (size_t)dim * (size_t)moe_groups;

                matvec_q80_cpu(gq, gsr, s->xn, s->moe_gate, moe_dim, dim, gs);
                matvec_q80_cpu(uq, usr, s->xn, s->moe_up, moe_dim, dim, gs);
                for (int i = 0; i < moe_dim; i++) s->moe_gate[i] = silu_cpu(s->moe_gate[i]) * s->moe_up[i];
                matvec_q80_cpu(dq, dsr, s->moe_gate, s->xb, dim, moe_dim, gs);
                for (int i = 0; i < dim; i++) s->moe_out_acc[i] += w_e * s->xb[i];
            }
        }
    }

    matvec_q80_cpu(w->gate_sh_q, w->gate_sh_s, s->xn, s->moe_gate, moe_dim, dim, gs);
    matvec_q80_cpu(w->up_sh_q, w->up_sh_s, s->xn, s->moe_up, moe_dim, dim, gs);
    for (int i = 0; i < moe_dim; i++) s->moe_gate[i] = silu_cpu(s->moe_gate[i]) * s->moe_up[i];
    matvec_q80_cpu(w->down_sh_q, w->down_sh_s, s->moe_gate, s->xb, dim, moe_dim, gs);

    for (int i = 0; i < dim; i++) s->x[i] += s->xb[i] + s->moe_out_acc[i];
}

static int compare_layer_checkpoint(const float *gpu,
                                    const float *cpu,
                                    int n,
                                    double *l1_out,
                                    float *linf_out,
                                    int *idx_out) {
    if (!gpu || !cpu || n <= 0) return -1;
    double l1 = 0.0;
    float linf = 0.0f;
    int idx = 0;
    for (int i = 0; i < n; i++) {
        float d = fabsf(gpu[i] - cpu[i]);
        l1 += (double)d;
        if (d > linf) {
            linf = d;
            idx = i;
        }
    }
    if (l1_out) *l1_out = l1;
    if (linf_out) *linf_out = linf;
    if (idx_out) *idx_out = idx;
    return 0;
}

static int alloc_copy_blob(void **dst, const void *src, size_t bytes) {
    if (!dst) return -1;
    *dst = NULL;
    if (!src || bytes == 0) return 0;
    void *p = malloc(bytes);
    if (!p) return -1;
    memcpy(p, src, bytes);
    *dst = p;
    return 0;
}

static void runstate_backup_free(RunState *b) {
    if (!b) return;
    free(b->x);
    free(b->xn);
    free(b->xb);
    free(b->ff_gate);
    free(b->ff_up);
    free(b->logits);
    free(b->k_cache_comp);
    free(b->k_cache_pe);
    free(b->q_a);
    free(b->q_full);
    free(b->kv_a);
    free(b->q_nope);
    free(b->q_pe);
    free(b->q_abs);
    free(b->att_scores);
    free(b->att_ctx);
    free(b->att_concat);
    free(b->moe_gate);
    free(b->moe_up);
    free(b->moe_router);
    free(b->moe_out_acc);
    free(b->moe_topk_idx);
    free(b->moe_topk_w);
    memset(b, 0, sizeof(*b));
}

static int runstate_backup_from(const Runtime *rt, const RunState *src, RunState *dst) {
    if (!rt || !src || !dst) return -1;
    memset(dst, 0, sizeof(*dst));

    const RuntimeConfig *cfg = &rt->cfg;
    dst->cache_cap = src->cache_cap;
    dst->n_attn_layers = src->n_attn_layers;

    size_t dim = (size_t)cfg->dim;
    size_t hidden = (size_t)cfg->hidden_dim;
    size_t vocab = (size_t)cfg->vocab_size;

    if (alloc_copy_blob((void **)&dst->x, src->x, dim * sizeof(float)) != 0 ||
        alloc_copy_blob((void **)&dst->xn, src->xn, dim * sizeof(float)) != 0 ||
        alloc_copy_blob((void **)&dst->xb, src->xb, dim * sizeof(float)) != 0 ||
        alloc_copy_blob((void **)&dst->ff_gate, src->ff_gate, hidden * sizeof(float)) != 0 ||
        alloc_copy_blob((void **)&dst->ff_up, src->ff_up, hidden * sizeof(float)) != 0 ||
        alloc_copy_blob((void **)&dst->logits, src->logits, vocab * sizeof(float)) != 0) {
        runstate_backup_free(dst);
        return -1;
    }

    if (src->n_attn_layers > 0) {
        size_t cache_comp = (size_t)src->n_attn_layers * (size_t)src->cache_cap * (size_t)cfg->kv_lora_rank * sizeof(float);
        size_t cache_pe = (size_t)src->n_attn_layers * (size_t)src->cache_cap * (size_t)cfg->rope_dim * sizeof(float);
        if (alloc_copy_blob((void **)&dst->k_cache_comp, src->k_cache_comp, cache_comp) != 0 ||
            alloc_copy_blob((void **)&dst->k_cache_pe, src->k_cache_pe, cache_pe) != 0 ||
            alloc_copy_blob((void **)&dst->q_a, src->q_a, (size_t)cfg->q_lora_rank * sizeof(float)) != 0 ||
            alloc_copy_blob((void **)&dst->q_full, src->q_full, (size_t)cfg->n_heads * (size_t)cfg->l0_head_k_dim * sizeof(float)) != 0 ||
            alloc_copy_blob((void **)&dst->kv_a, src->kv_a, (size_t)(cfg->kv_lora_rank + cfg->rope_dim) * sizeof(float)) != 0 ||
            alloc_copy_blob((void **)&dst->q_nope, src->q_nope, (size_t)cfg->l0_qk_nope_dim * sizeof(float)) != 0 ||
            alloc_copy_blob((void **)&dst->q_pe, src->q_pe, (size_t)cfg->rope_dim * sizeof(float)) != 0 ||
            alloc_copy_blob((void **)&dst->q_abs, src->q_abs, (size_t)cfg->kv_lora_rank * sizeof(float)) != 0 ||
            alloc_copy_blob((void **)&dst->att_scores, src->att_scores, (size_t)src->cache_cap * sizeof(float)) != 0 ||
            alloc_copy_blob((void **)&dst->att_ctx, src->att_ctx, (size_t)cfg->kv_lora_rank * sizeof(float)) != 0 ||
            alloc_copy_blob((void **)&dst->att_concat, src->att_concat, (size_t)cfg->n_heads * (size_t)cfg->l0_v_head_dim * sizeof(float)) != 0) {
            runstate_backup_free(dst);
            return -1;
        }
    }

    if (rt->has_moe_shared && cfg->moe_ffn_dim > 0) {
        if (alloc_copy_blob((void **)&dst->moe_gate, src->moe_gate, (size_t)cfg->moe_ffn_dim * sizeof(float)) != 0 ||
            alloc_copy_blob((void **)&dst->moe_up, src->moe_up, (size_t)cfg->moe_ffn_dim * sizeof(float)) != 0 ||
            alloc_copy_blob((void **)&dst->moe_router, src->moe_router, (size_t)cfg->n_routed_experts * sizeof(float)) != 0 ||
            alloc_copy_blob((void **)&dst->moe_out_acc, src->moe_out_acc, dim * sizeof(float)) != 0 ||
            alloc_copy_blob((void **)&dst->moe_topk_idx, src->moe_topk_idx, (size_t)cfg->n_experts_used * sizeof(int)) != 0 ||
            alloc_copy_blob((void **)&dst->moe_topk_w, src->moe_topk_w, (size_t)cfg->n_experts_used * sizeof(float)) != 0) {
            runstate_backup_free(dst);
            return -1;
        }
    }

    return 0;
}

static void runstate_restore_from_backup(const Runtime *rt, const RunState *backup, RunState *dst) {
    if (!rt || !backup || !dst) return;
    const RuntimeConfig *cfg = &rt->cfg;
    dst->cache_cap = backup->cache_cap;
    dst->n_attn_layers = backup->n_attn_layers;

    if (dst->x && backup->x) memcpy(dst->x, backup->x, (size_t)cfg->dim * sizeof(float));
    if (dst->xn && backup->xn) memcpy(dst->xn, backup->xn, (size_t)cfg->dim * sizeof(float));
    if (dst->xb && backup->xb) memcpy(dst->xb, backup->xb, (size_t)cfg->dim * sizeof(float));
    if (dst->ff_gate && backup->ff_gate) memcpy(dst->ff_gate, backup->ff_gate, (size_t)cfg->hidden_dim * sizeof(float));
    if (dst->ff_up && backup->ff_up) memcpy(dst->ff_up, backup->ff_up, (size_t)cfg->hidden_dim * sizeof(float));
    if (dst->logits && backup->logits) memcpy(dst->logits, backup->logits, (size_t)cfg->vocab_size * sizeof(float));

    if (dst->n_attn_layers > 0) {
        if (dst->k_cache_comp && backup->k_cache_comp) {
            memcpy(dst->k_cache_comp,
                   backup->k_cache_comp,
                   (size_t)dst->n_attn_layers * (size_t)dst->cache_cap * (size_t)cfg->kv_lora_rank * sizeof(float));
        }
        if (dst->k_cache_pe && backup->k_cache_pe) {
            memcpy(dst->k_cache_pe,
                   backup->k_cache_pe,
                   (size_t)dst->n_attn_layers * (size_t)dst->cache_cap * (size_t)cfg->rope_dim * sizeof(float));
        }
        if (dst->q_a && backup->q_a) memcpy(dst->q_a, backup->q_a, (size_t)cfg->q_lora_rank * sizeof(float));
        if (dst->q_full && backup->q_full) memcpy(dst->q_full, backup->q_full, (size_t)cfg->n_heads * (size_t)cfg->l0_head_k_dim * sizeof(float));
        if (dst->kv_a && backup->kv_a) memcpy(dst->kv_a, backup->kv_a, (size_t)(cfg->kv_lora_rank + cfg->rope_dim) * sizeof(float));
        if (dst->q_nope && backup->q_nope) memcpy(dst->q_nope, backup->q_nope, (size_t)cfg->l0_qk_nope_dim * sizeof(float));
        if (dst->q_pe && backup->q_pe) memcpy(dst->q_pe, backup->q_pe, (size_t)cfg->rope_dim * sizeof(float));
        if (dst->q_abs && backup->q_abs) memcpy(dst->q_abs, backup->q_abs, (size_t)cfg->kv_lora_rank * sizeof(float));
        if (dst->att_scores && backup->att_scores) memcpy(dst->att_scores, backup->att_scores, (size_t)dst->cache_cap * sizeof(float));
        if (dst->att_ctx && backup->att_ctx) memcpy(dst->att_ctx, backup->att_ctx, (size_t)cfg->kv_lora_rank * sizeof(float));
        if (dst->att_concat && backup->att_concat) memcpy(dst->att_concat, backup->att_concat, (size_t)cfg->n_heads * (size_t)cfg->l0_v_head_dim * sizeof(float));
    }

    if (rt->has_moe_shared && cfg->moe_ffn_dim > 0) {
        if (dst->moe_gate && backup->moe_gate) memcpy(dst->moe_gate, backup->moe_gate, (size_t)cfg->moe_ffn_dim * sizeof(float));
        if (dst->moe_up && backup->moe_up) memcpy(dst->moe_up, backup->moe_up, (size_t)cfg->moe_ffn_dim * sizeof(float));
        if (dst->moe_router && backup->moe_router) memcpy(dst->moe_router, backup->moe_router, (size_t)cfg->n_routed_experts * sizeof(float));
        if (dst->moe_out_acc && backup->moe_out_acc) memcpy(dst->moe_out_acc, backup->moe_out_acc, (size_t)cfg->dim * sizeof(float));
        if (dst->moe_topk_idx && backup->moe_topk_idx) memcpy(dst->moe_topk_idx, backup->moe_topk_idx, (size_t)cfg->n_experts_used * sizeof(int));
        if (dst->moe_topk_w && backup->moe_topk_w) memcpy(dst->moe_topk_w, backup->moe_topk_w, (size_t)cfg->n_experts_used * sizeof(float));
    }
}

// Persistent command buffer management for batching matvec operations
static void matvec_batch_begin(void) {
    if (!g_queue) return;
    if (g_persistent_cb == nil) {
        g_persistent_cb = [g_queue commandBuffer];
        if (g_persistent_cb) {
            g_persistent_encoder = [g_persistent_cb computeCommandEncoder];
            g_matvec_batch_count = 0;
        }
    }
}

static void matvec_batch_flush(void) {
    if (g_persistent_encoder) {
        [g_persistent_encoder endEncoding];
        g_persistent_encoder = nil;
    }
    if (g_persistent_cb) {
        [g_persistent_cb commit];
        [g_persistent_cb waitUntilCompleted];
        if (g_persistent_cb.error) {
            fprintf(stderr, "[glm-metal] batch command failed: %s\n", 
                    g_persistent_cb.error.localizedDescription.UTF8String);
        }
        g_persistent_cb = nil;
    }
    g_matvec_batch_count = 0;
}

static void matvec_batch_ensure_encoder(void) {
    if (!g_persistent_encoder && g_persistent_cb) {
        g_persistent_encoder = [g_persistent_cb computeCommandEncoder];
    }
}

static int matvec_batch_should_flush(void) {
    return g_matvec_batch_count >= g_matvec_batch_flush_threshold;
}

int glm_metal_init(const Runtime *rt, RunState *st) {
    (void)st;
    if (!rt) return -1;
    if (g_device != nil) return 0;

    g_device = MTLCreateSystemDefaultDevice();
    if (!g_device) {
        fprintf(stderr, "[glm-metal] no Metal device available\n");
        return -1;
    }
    g_queue = [g_device newCommandQueue];
    if (!g_queue) {
        fprintf(stderr, "[glm-metal] failed to create command queue\n");
        return -1;
    }

    NSString *cwd = [[NSFileManager defaultManager] currentDirectoryPath];
    NSString *libPath = [cwd stringByAppendingPathComponent:@"infer.metallib"];
    NSURL *libURL = [NSURL fileURLWithPath:libPath];
    NSError *err = nil;
    g_library = [g_device newLibraryWithURL:libURL error:&err];
    if (!g_library) {
        fprintf(stderr, "[glm-metal] failed to load %s: %s\n",
                libPath.UTF8String, err ? err.localizedDescription.UTF8String : "unknown error");
        return -1;
    }

    g_pipelines = [[NSMutableDictionary alloc] init];
    g_nocopy_cache = [[NSMutableDictionary alloc] init];
    const char *required[] = {
        "embed_dequant_q80",
        "rmsnorm_f32",
        "matvec_q80_rows",
        "matvec_q80_rows_simdgroup",
        "rope_inplace",
        "softmax_1d",
        "vec_add_inplace",
        "silu_mul_inplace",
        "scale_add_inplace",
        "cache_write_kv_comp",
        "cache_write_kv_pe",
        "attention_scores",
        "attention_context",
        "moe_accumulate_shared",
        "moe_accumulate_routed",
        NULL
    };
    for (int i = 0; required[i] != NULL; i++) {
        if (ensure_pipeline(required[i]) != 0) return -1;
    }
    if (metal_native_enabled()) {
        if (ensure_pipeline("transformer_layer_fused") != 0) return -1;
    }

    g_ps_matvec = g_pipelines[@"matvec_q80_rows_simdgroup"];
    if (!g_ps_matvec) {
        fprintf(stderr, "[glm-metal] failed to cache matvec pipeline\n");
        return -1;
    }
    g_matvec_args = [g_device newBufferWithLength:sizeof(uint32_t) * 3 options:MTLResourceStorageModeShared];
    if (!g_matvec_args) {
        fprintf(stderr, "[glm-metal] failed to allocate matvec arg buffer\n");
        return -1;
    }

    const size_t dim = (size_t)rt->cfg.dim;
    const size_t vocab = (size_t)rt->cfg.vocab_size;
    const size_t gs = (size_t)rt->cfg.group_size;
    g_model.output_norm = make_nocopy_buffer(rt->output_norm, dim * sizeof(float), "output_norm");
    g_model.tok_q = make_nocopy_buffer(rt->tok_q, vocab * dim * sizeof(int8_t), "tok_q");
    g_model.tok_s = make_nocopy_buffer(rt->tok_s, vocab * (dim / gs) * sizeof(float), "tok_s");
    g_model.out_q = make_nocopy_buffer(rt->out_q, vocab * dim * sizeof(int8_t), "out_q");
    g_model.out_s = make_nocopy_buffer(rt->out_s, vocab * (dim / gs) * sizeof(float), "out_s");
    if (!g_model.output_norm || !g_model.tok_q || !g_model.tok_s || !g_model.out_q || !g_model.out_s) return -1;

    fprintf(stderr,
            "[glm-metal] initialized device=%s native=%s native_unsafe=%s hybrid_matvec=%s\n",
            g_device.name.UTF8String,
            metal_native_enabled() ? "on" : "off",
            metal_native_unsafe_enabled() ? "on" : "off",
            metal_hybrid_matvec_enabled() ? "on" : "off");
    return 0;
}

// Upload all model weights to GPU for full native forward
static int glm_metal_upload_weights(const Runtime *rt) {
    if (!rt || !g_device) return -1;
    
    const RuntimeConfig *cfg = &rt->cfg;
    const size_t dim = (size_t)cfg->dim;
    const size_t vocab = (size_t)cfg->vocab_size;
    const size_t gs = (size_t)cfg->group_size;
    const int n_layers = cfg->n_layers;
    
    // Upload base model weights
    g_model.output_norm = upload_weight_buffer(rt->output_norm, dim * sizeof(float), "output_norm");
    g_model.tok_q = upload_weight_buffer(rt->tok_q, vocab * dim * sizeof(int8_t), "tok_q");
    g_model.tok_s = upload_weight_buffer(rt->tok_s, vocab * (dim / gs) * sizeof(float), "tok_s");
    g_model.out_q = upload_weight_buffer(rt->out_q, vocab * dim * sizeof(int8_t), "out_q");
    g_model.out_s = upload_weight_buffer(rt->out_s, vocab * (dim / gs) * sizeof(float), "out_s");
    
    // Upload L0 FFN weights if present
    if (rt->has_l0_ffn) {
        g_model.l0_ffn_norm = upload_weight_buffer(rt->l0_ffn_norm, dim * sizeof(float), "l0_ffn_norm");
        g_model.l0_ffn_gate_q = upload_weight_buffer(rt->l0_ffn_gate_q, cfg->hidden_dim * dim * sizeof(int8_t), "l0_ffn_gate_q");
        g_model.l0_ffn_gate_s = upload_weight_buffer(rt->l0_ffn_gate_s, cfg->hidden_dim * (dim / gs) * sizeof(float), "l0_ffn_gate_s");
        g_model.l0_ffn_up_q = upload_weight_buffer(rt->l0_ffn_up_q, cfg->hidden_dim * dim * sizeof(int8_t), "l0_ffn_up_q");
        g_model.l0_ffn_up_s = upload_weight_buffer(rt->l0_ffn_up_s, cfg->hidden_dim * (dim / gs) * sizeof(float), "l0_ffn_up_s");
        g_model.l0_ffn_down_q = upload_weight_buffer(rt->l0_ffn_down_q, dim * cfg->hidden_dim * sizeof(int8_t), "l0_ffn_down_q");
        g_model.l0_ffn_down_s = upload_weight_buffer(rt->l0_ffn_down_s, dim * (cfg->hidden_dim / gs) * sizeof(float), "l0_ffn_down_s");
    }
    
    // Allocate layer buffers
    g_n_layers = n_layers;
    g_layers = (MetalLayerBuffers *)calloc(n_layers, sizeof(MetalLayerBuffers));
    if (!g_layers) {
        fprintf(stderr, "[glm-metal] failed to allocate layer buffer array\n");
        return -1;
    }
    
    // Upload per-layer weights
    for (int l = 0; l < n_layers; l++) {
        MetalLayerBuffers *lb = &g_layers[l];
        
        if (rt->has_all_attn) {
            const LayerAttnWeights *w = &rt->attn_layers[l];
            lb->attn_norm = upload_weight_buffer(w->attn_norm, dim * sizeof(float), "attn_norm");
            lb->q_a_norm = upload_weight_buffer(w->q_a_norm, cfg->q_lora_rank * sizeof(float), "q_a_norm");
            lb->kv_a_norm = upload_weight_buffer(w->kv_a_norm, cfg->kv_lora_rank * sizeof(float), "kv_a_norm");
            
            lb->q_a_q = upload_weight_buffer(w->q_a_q, cfg->q_lora_rank * dim * sizeof(int8_t), "q_a_q");
            lb->q_a_s = upload_weight_buffer(w->q_a_s, cfg->q_lora_rank * (dim / gs) * sizeof(float), "q_a_s");
            lb->q_b_q = upload_weight_buffer(w->q_b_q, cfg->n_heads * cfg->l0_head_k_dim * cfg->q_lora_rank * sizeof(int8_t), "q_b_q");
            lb->q_b_s = upload_weight_buffer(w->q_b_s, cfg->n_heads * cfg->l0_head_k_dim * (cfg->q_lora_rank / gs) * sizeof(float), "q_b_s");
            lb->kv_a_q = upload_weight_buffer(w->kv_a_q, (cfg->kv_lora_rank + cfg->rope_dim) * dim * sizeof(int8_t), "kv_a_q");
            lb->kv_a_s = upload_weight_buffer(w->kv_a_s, (cfg->kv_lora_rank + cfg->rope_dim) * (dim / gs) * sizeof(float), "kv_a_s");
            
            // Upload k_b, v_b per head (packed arrays)
            lb->k_b_q = upload_weight_buffer(w->k_b_q, cfg->n_heads * cfg->kv_lora_rank * cfg->l0_qk_nope_dim * sizeof(int8_t), "k_b_q");
            lb->k_b_s = upload_weight_buffer(w->k_b_s, cfg->n_heads * cfg->kv_lora_rank * (cfg->l0_qk_nope_dim / gs) * sizeof(float), "k_b_s");
            lb->v_b_q = upload_weight_buffer(w->v_b_q, cfg->n_heads * cfg->l0_v_head_dim * cfg->kv_lora_rank * sizeof(int8_t), "v_b_q");
            lb->v_b_s = upload_weight_buffer(w->v_b_s, cfg->n_heads * cfg->l0_v_head_dim * (cfg->kv_lora_rank / gs) * sizeof(float), "v_b_s");
            
            lb->attn_out_q = upload_weight_buffer(w->attn_out_q, dim * cfg->n_heads * cfg->l0_v_head_dim * sizeof(int8_t), "attn_out_q");
            lb->attn_out_s = upload_weight_buffer(w->attn_out_s, dim * (cfg->n_heads * cfg->l0_v_head_dim / gs) * sizeof(float), "attn_out_s");
        }
        
        if (rt->has_moe_shared && l > 0) {
            const LayerMoeShared *w = &rt->moe_layers[l];
            lb->ffn_norm = upload_weight_buffer(w->ffn_norm, dim * sizeof(float), "ffn_norm");
            lb->gate_inp = upload_weight_buffer(w->gate_inp, cfg->n_routed_experts * dim * sizeof(float), "gate_inp");
            lb->exp_probs_b = upload_weight_buffer(w->exp_probs_b, cfg->n_routed_experts * sizeof(float), "exp_probs_b");
            lb->gate_sh_q = upload_weight_buffer(w->gate_sh_q, cfg->moe_ffn_dim * dim * sizeof(int8_t), "gate_sh_q");
            lb->gate_sh_s = upload_weight_buffer(w->gate_sh_s, cfg->moe_ffn_dim * (dim / gs) * sizeof(float), "gate_sh_s");
            lb->up_sh_q = upload_weight_buffer(w->up_sh_q, cfg->moe_ffn_dim * dim * sizeof(int8_t), "up_sh_q");
            lb->up_sh_s = upload_weight_buffer(w->up_sh_s, cfg->moe_ffn_dim * (dim / gs) * sizeof(float), "up_sh_s");
            lb->down_sh_q = upload_weight_buffer(w->down_sh_q, dim * cfg->moe_ffn_dim * sizeof(int8_t), "down_sh_q");
            lb->down_sh_s = upload_weight_buffer(w->down_sh_s, dim * (cfg->moe_ffn_dim / gs) * sizeof(float), "down_sh_s");
        }
    }
    
    fprintf(stderr, "[glm-metal] uploaded %d layers to GPU\n", n_layers);
    return 0;
}

void glm_metal_prepare(const Runtime *rt, RunState *st) {
    if (!rt || !st) return;
    const RuntimeConfig *cfg = &rt->cfg;
    if (st->cache_cap <= 0 || st->cache_cap > cfg->seq_len) {
        fprintf(stderr, "[glm-metal] invalid cache_cap: %d (seq_len=%d)\n", st->cache_cap, cfg->seq_len);
        return;
    }
    if (st->n_attn_layers < 0 || st->n_attn_layers > cfg->n_layers) {
        fprintf(stderr, "[glm-metal] invalid attention layer count: %d (n_layers=%d)\n", st->n_attn_layers, cfg->n_layers);
        return;
    }

    if (!metal_native_enabled() || !metal_native_unsafe_enabled()) {
        fprintf(stderr, "[glm-metal] native forward disabled; skipping full GPU weight upload\n");
        return;
    }

    // Upload all weights to GPU for native forward
    if (glm_metal_upload_weights(rt) != 0) {
        fprintf(stderr, "[glm-metal] failed to upload weights\n");
        return;
    }

    g_run.x = make_shared_buffer((NSUInteger)cfg->dim * sizeof(float), "x");
    g_run.xn = make_shared_buffer((NSUInteger)cfg->dim * sizeof(float), "xn");
    g_run.xb = make_shared_buffer((NSUInteger)cfg->dim * sizeof(float), "xb");
    g_run.ff_gate = make_shared_buffer((NSUInteger)cfg->hidden_dim * sizeof(float), "ff_gate");
    g_run.ff_up = make_shared_buffer((NSUInteger)cfg->hidden_dim * sizeof(float), "ff_up");
    g_run.logits = make_shared_buffer((NSUInteger)cfg->vocab_size * sizeof(float), "logits");

    g_run.k_cache_comp = make_shared_buffer((NSUInteger)st->n_attn_layers * (NSUInteger)st->cache_cap * (NSUInteger)cfg->kv_lora_rank * sizeof(float), "k_cache_comp");
    g_run.k_cache_pe = make_shared_buffer((NSUInteger)st->n_attn_layers * (NSUInteger)st->cache_cap * (NSUInteger)cfg->rope_dim * sizeof(float), "k_cache_pe");
    g_run.q_a = make_shared_buffer((NSUInteger)cfg->q_lora_rank * sizeof(float), "q_a");
    g_run.q_full = make_shared_buffer((NSUInteger)cfg->n_heads * (NSUInteger)cfg->l0_head_k_dim * sizeof(float), "q_full");
    g_run.kv_a = make_shared_buffer((NSUInteger)(cfg->kv_lora_rank + cfg->rope_dim) * sizeof(float), "kv_a");
    g_run.q_nope = make_shared_buffer((NSUInteger)cfg->l0_qk_nope_dim * sizeof(float), "q_nope");
    g_run.q_pe = make_shared_buffer((NSUInteger)cfg->rope_dim * sizeof(float), "q_pe");
    g_run.q_abs = make_shared_buffer((NSUInteger)cfg->kv_lora_rank * sizeof(float), "q_abs");
    g_run.att_scores = make_shared_buffer((NSUInteger)st->cache_cap * sizeof(float), "att_scores");
    g_run.att_ctx = make_shared_buffer((NSUInteger)cfg->kv_lora_rank * sizeof(float), "att_ctx");
    g_run.att_concat = make_shared_buffer((NSUInteger)cfg->n_heads * (NSUInteger)cfg->l0_v_head_dim * sizeof(float), "att_concat");
    NSUInteger layer_intermediate_elems = (NSUInteger)cfg->q_lora_rank +
                                          (NSUInteger)cfg->n_heads * (NSUInteger)cfg->l0_head_k_dim +
                                          (NSUInteger)cfg->kv_lora_rank +
                                          (NSUInteger)cfg->rope_dim;
    NSUInteger layer_ff_elems = (NSUInteger)(cfg->hidden_dim > cfg->moe_ffn_dim ? cfg->hidden_dim : cfg->moe_ffn_dim);
    if (layer_ff_elems < (NSUInteger)cfg->dim) layer_ff_elems = (NSUInteger)cfg->dim;
    g_run.layer_intermediates = make_shared_buffer(layer_intermediate_elems * sizeof(float), "layer_intermediates");
    g_run.layer_ff_intermediates = make_shared_buffer(layer_ff_elems * sizeof(float), "layer_ff_intermediates");

    g_run.moe_gate = make_shared_buffer((NSUInteger)cfg->moe_ffn_dim * sizeof(float), "moe_gate");
    g_run.moe_up = make_shared_buffer((NSUInteger)cfg->moe_ffn_dim * sizeof(float), "moe_up");
    g_run.moe_router = make_shared_buffer((NSUInteger)cfg->n_routed_experts * sizeof(float), "moe_router");
    g_run.moe_out_acc = make_shared_buffer((NSUInteger)cfg->dim * sizeof(float), "moe_out_acc");
    g_run.moe_topk_idx = make_shared_buffer((NSUInteger)cfg->n_experts_used * sizeof(int), "moe_topk_idx");
    g_run.moe_topk_w = make_shared_buffer((NSUInteger)cfg->n_experts_used * sizeof(float), "moe_topk_w");

    if (!g_run.layer_intermediates || !g_run.layer_ff_intermediates) {
        fprintf(stderr, "[glm-metal] failed to allocate native layer intermediate buffers\n");
        return;
    }
    
    fprintf(stderr, "[glm-metal] prepared GPU buffers for native forward\n");
}

// Native GPU forward pass - full token inference on GPU (Phase 3)
int glm_metal_forward_token_native(const Runtime *rt, RunState *st, int token, int pos, int debug_mode) {
    if (!rt || !st) return -1;
    if (!g_device || !g_queue) {
        fprintf(stderr, "[glm-metal] native forward: device not initialized\n");
        return -1;
    }
    if (!g_layers || g_n_layers != rt->cfg.n_layers) {
        fprintf(stderr, "[glm-metal] native forward: weights not uploaded\n");
        return -1;
    }
    
    // Get pipeline for fused layer kernel
    id<MTLComputePipelineState> ps_layer = g_pipelines[@"transformer_layer_fused"];
    if (!ps_layer) {
        fprintf(stderr, "[glm-metal] native forward: layer kernel not found\n");
        return -1;
    }
    
    const RuntimeConfig *cfg = &rt->cfg;
    const int n_layers = cfg->n_layers;
    const uint32_t dim = (uint32_t)cfg->dim;
    const uint32_t hidden_dim = (uint32_t)cfg->hidden_dim;
    const int cpu_ffn_enabled = metal_native_ffn_cpu_enabled();
    const int parity_requested = metal_native_parity_enabled() && (pos == 0 || metal_native_parity_all_pos());
    const int parity_stage = metal_native_parity_stage();
    float *cpu_layer_ckpt = NULL;
    float *cpu_attn_ckpt = NULL;
    int cpu_layers_written = 0;
    int cpu_attn_layers_written = 0;
    int parity_layers = 0;
    int parity_failed = 0;
    float parity_tol = metal_native_parity_tol();
    
    // Create argument buffer for layer args
    struct LayerArgs {
        uint32_t dim;
        uint32_t hidden_dim;
        uint32_t n_heads;
        uint32_t q_lora_rank;
        uint32_t kv_lora_rank;
        uint32_t rope_dim;
        uint32_t qk_nope_dim;
        uint32_t v_head_dim;
        uint32_t head_k_dim;
        uint32_t seq_len;
        uint32_t pos;
        uint32_t layer_idx;
        float kq_scale;
        uint32_t gs;
    };
    
    // Allocate intermediate buffers for each layer (ping-pong buffers: x <-> x_out)
    id<MTLBuffer> buf_x = g_run.x;
    id<MTLBuffer> buf_x_out = g_run.xb;

    if (parity_requested) {
        parity_layers = n_layers;
        int requested_layers = metal_native_parity_layers();
        if (requested_layers > 0 && requested_layers < parity_layers) parity_layers = requested_layers;
        int need_ffn = (parity_stage == GLM_NATIVE_PARITY_STAGE_FFN || parity_stage == GLM_NATIVE_PARITY_STAGE_BOTH);
        int need_attn = (parity_stage == GLM_NATIVE_PARITY_STAGE_ATTN || parity_stage == GLM_NATIVE_PARITY_STAGE_BOTH);
        if (need_ffn) cpu_layer_ckpt = (float *)calloc((size_t)parity_layers * (size_t)cfg->dim, sizeof(float));
        if (need_attn) cpu_attn_ckpt = (float *)calloc((size_t)parity_layers * (size_t)cfg->dim, sizeof(float));
        if ((need_ffn && !cpu_layer_ckpt) || (need_attn && !cpu_attn_ckpt)) {
            fprintf(stderr, "[glm-metal] native parity: out of memory\n");
            free(cpu_layer_ckpt);
            free(cpu_attn_ckpt);
            return -1;
        }
        int ckpt_status = glm_cpu_forward_token_dual_checkpoints(rt,
                                                                 st,
                                                                 token,
                                                                 pos,
                                                                 debug_mode,
                                                                 cpu_attn_ckpt,
                                                                 cfg->dim,
                                                                 parity_layers,
                                                                 &cpu_attn_layers_written,
                                                                 cpu_layer_ckpt,
                                                                 cfg->dim,
                                                                 parity_layers,
                                                                 &cpu_layers_written);
        if (ckpt_status != 0) {
            fprintf(stderr, "[glm-metal] native parity: failed to collect CPU checkpoints\n");
            free(cpu_layer_ckpt);
            free(cpu_attn_ckpt);
            cpu_layer_ckpt = NULL;
            cpu_attn_ckpt = NULL;
            parity_layers = 0;
            cpu_layers_written = 0;
            cpu_attn_layers_written = 0;
        } else {
            fprintf(stderr,
                    "[glm-metal] native parity: enabled for pos=%d ffn_layers=%d attn_layers=%d stage=%d tol=%.6g\n",
                    pos,
                    cpu_layers_written,
                    cpu_attn_layers_written,
                    parity_stage,
                    (double)parity_tol);
            if (parity_stage == GLM_NATIVE_PARITY_STAGE_BOTH) {
                fprintf(stderr, "[glm-metal] native parity: comparing both post-attn and post-ffn checkpoints\n");
            }
        }
    }

    if (parity_requested && !cpu_layer_ckpt && !cpu_attn_ckpt) {
        fprintf(stderr,
                "[glm-metal] native parity: no checkpoint buffers enabled; stage=%d\n",
                parity_stage);
    }

    float *gpu_attn_snapshot = NULL;
    int need_gpu_attn_snapshot = (parity_stage == GLM_NATIVE_PARITY_STAGE_ATTN || parity_stage == GLM_NATIVE_PARITY_STAGE_BOTH);
    if (need_gpu_attn_snapshot && cpu_ffn_enabled) {
        gpu_attn_snapshot = (float *)malloc((size_t)cfg->dim * sizeof(float));
        if (!gpu_attn_snapshot) {
            free(cpu_layer_ckpt);
            free(cpu_attn_ckpt);
            fprintf(stderr, "[glm-metal] native parity: failed to allocate GPU attn snapshot buffer\n");
            return -1;
        }
    }

    int per_layer_sync = cpu_ffn_enabled || (cpu_layer_ckpt != NULL) || (cpu_attn_ckpt != NULL);
    id<MTLCommandBuffer> cb = nil;
    if (!per_layer_sync) {
        cb = [g_queue commandBuffer];
        if (!cb) {
            fprintf(stderr, "[glm-metal] native forward: failed to create command buffer\n");
            free(cpu_layer_ckpt);
            free(cpu_attn_ckpt);
            free(gpu_attn_snapshot);
            return -1;
        }
    }
    
    // Step 1: Token embedding lookup (GPU)
    // For now, do this on CPU and copy to GPU
    // TODO: Implement GPU embed lookup kernel
    int gs = (int)cfg->group_size;
    int n_groups = cfg->dim / gs;
    const int8_t *eq = rt->tok_q + (size_t)token * (size_t)cfg->dim;
    const float *es = rt->tok_s + (size_t)token * (size_t)n_groups;
    // Dequantize on CPU for now
    float *x_host = (float *)malloc(cfg->dim * sizeof(float));
    for (int i = 0; i < cfg->dim; i++) {
        int group = i / gs;
        x_host[i] = (float)eq[i] * es[group];
    }
    memcpy(g_run.x.contents, x_host, cfg->dim * sizeof(float));
    free(x_host);

    // Step 2: Process all layers
    for (int layer = 0; layer < n_layers; layer++) {
        id<MTLCommandBuffer> cb_layer = cb ? cb : [g_queue commandBuffer];
        if (!cb_layer) {
            fprintf(stderr, "[glm-metal] native forward: failed to create layer command buffer\n");
            free(cpu_layer_ckpt);
            return -1;
        }
        id<MTLComputeCommandEncoder> encoder = [cb_layer computeCommandEncoder];
        if (!encoder) {
            fprintf(stderr, "[glm-metal] native forward: failed to create layer encoder\n");
            free(cpu_layer_ckpt);
            return -1;
        }
        
        // Set up layer arguments
        id<MTLBuffer> args_buf = [g_device newBufferWithLength:sizeof(struct LayerArgs) options:MTLResourceStorageModeShared];
        struct LayerArgs *args = (struct LayerArgs *)args_buf.contents;
        args->dim = dim;
        args->hidden_dim = hidden_dim;
        args->n_heads = (uint32_t)cfg->n_heads;
        args->q_lora_rank = (uint32_t)cfg->q_lora_rank;
        args->kv_lora_rank = (uint32_t)cfg->kv_lora_rank;
        args->rope_dim = (uint32_t)cfg->rope_dim;
        args->qk_nope_dim = (uint32_t)cfg->l0_qk_nope_dim;
        args->v_head_dim = (uint32_t)cfg->l0_v_head_dim;
        args->head_k_dim = (uint32_t)cfg->l0_head_k_dim;
        args->seq_len = (uint32_t)st->cache_cap;
        args->pos = (uint32_t)pos;
        args->layer_idx = (uint32_t)layer;
        args->kq_scale = 1.0f / sqrtf((float)cfg->l0_head_k_dim);
        args->gs = (uint32_t)gs;
        
        MetalLayerBuffers *lb = &g_layers[layer];
        
        // Set pipeline
        [encoder setComputePipelineState:ps_layer];
        
        // Set buffers - attention weights
        [encoder setBuffer:args_buf offset:0 atIndex:0];
        [encoder setBuffer:lb->attn_norm offset:0 atIndex:1];
        [encoder setBuffer:lb->q_a_norm offset:0 atIndex:2];
        [encoder setBuffer:lb->kv_a_norm offset:0 atIndex:3];
        [encoder setBuffer:lb->q_a_q offset:0 atIndex:4];
        [encoder setBuffer:lb->q_a_s offset:0 atIndex:5];
        [encoder setBuffer:lb->q_b_q offset:0 atIndex:6];
        [encoder setBuffer:lb->q_b_s offset:0 atIndex:7];
        [encoder setBuffer:lb->kv_a_q offset:0 atIndex:8];
        [encoder setBuffer:lb->kv_a_s offset:0 atIndex:9];
        [encoder setBuffer:lb->k_b_q offset:0 atIndex:10];
        [encoder setBuffer:lb->k_b_s offset:0 atIndex:11];
        [encoder setBuffer:lb->v_b_q offset:0 atIndex:12];
        [encoder setBuffer:lb->v_b_s offset:0 atIndex:13];
        [encoder setBuffer:lb->attn_out_q offset:0 atIndex:14];
        [encoder setBuffer:lb->attn_out_s offset:0 atIndex:15];
        
        // FFN weights
        if (layer == 0 && rt->has_l0_ffn) {
            [encoder setBuffer:g_model.l0_ffn_norm offset:0 atIndex:16];
            [encoder setBuffer:g_model.l0_ffn_gate_q offset:0 atIndex:17];
            [encoder setBuffer:g_model.l0_ffn_gate_s offset:0 atIndex:18];
            [encoder setBuffer:g_model.l0_ffn_up_q offset:0 atIndex:19];
            [encoder setBuffer:g_model.l0_ffn_up_s offset:0 atIndex:20];
            [encoder setBuffer:g_model.l0_ffn_down_q offset:0 atIndex:21];
            [encoder setBuffer:g_model.l0_ffn_down_s offset:0 atIndex:22];
        } else if (rt->has_moe_shared && layer > 0) {
            [encoder setBuffer:lb->ffn_norm offset:0 atIndex:16];
            [encoder setBuffer:lb->gate_sh_q offset:0 atIndex:17];
            [encoder setBuffer:lb->gate_sh_s offset:0 atIndex:18];
            [encoder setBuffer:lb->up_sh_q offset:0 atIndex:19];
            [encoder setBuffer:lb->up_sh_s offset:0 atIndex:20];
            [encoder setBuffer:lb->down_sh_q offset:0 atIndex:21];
            [encoder setBuffer:lb->down_sh_s offset:0 atIndex:22];
        } else {
            // No FFN for this layer - set dummy buffers
            [encoder setBuffer:g_run.x offset:0 atIndex:16];
            [encoder setBuffer:g_run.x offset:0 atIndex:17];
            [encoder setBuffer:g_run.x offset:0 atIndex:18];
            [encoder setBuffer:g_run.x offset:0 atIndex:19];
            [encoder setBuffer:g_run.x offset:0 atIndex:20];
            [encoder setBuffer:g_run.x offset:0 atIndex:21];
            [encoder setBuffer:g_run.x offset:0 atIndex:22];
        }
        
        // Activations - use ping-pong buffers
        id<MTLBuffer> input_buf = (layer % 2 == 0) ? buf_x : buf_x_out;
        id<MTLBuffer> output_buf = (layer % 2 == 0) ? buf_x_out : buf_x;
        
        [encoder setBuffer:input_buf offset:0 atIndex:23];
        [encoder setBuffer:output_buf offset:0 atIndex:24];
        [encoder setBuffer:g_run.k_cache_comp offset:0 atIndex:25];
        [encoder setBuffer:g_run.k_cache_pe offset:0 atIndex:26];
        
        // Intermediate buffers
        [encoder setBuffer:g_run.layer_intermediates offset:0 atIndex:27];
        [encoder setBuffer:g_run.layer_ff_intermediates offset:0 atIndex:28];
        
        // Dispatch
        // Fused kernel currently requires single-threadgroup execution because
        // it uses threadgroup barriers across phase boundaries.
        const unsigned int tgs = 256;
        const unsigned int tgx = 1;
        [encoder dispatchThreadgroups:MTLSizeMake(tgx, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgs, 1, 1)];
        [encoder endEncoding];

        if (cb == nil) {
            [cb_layer commit];
            [cb_layer waitUntilCompleted];
            if (cb_layer.error) {
                fprintf(stderr, "[glm-metal] native layer=%d failed: %s\n", layer, cb_layer.error.localizedDescription.UTF8String);
                free(cpu_layer_ckpt);
                free(cpu_attn_ckpt);
                free(gpu_attn_snapshot);
                return -1;
            }

            float *gpu_vec = (float *)output_buf.contents;
            if (gpu_attn_snapshot) {
                memcpy(gpu_attn_snapshot, gpu_vec, cfg->dim * sizeof(float));
            }

            if (cpu_ffn_enabled) {
                memcpy(st->x, gpu_vec, cfg->dim * sizeof(float));
                if (layer == 0) {
                    apply_l0_dense_ffn_cpu(rt, st, gs);
                } else {
                    apply_moe_shared_ffn_layer_cpu(rt, st, gs, layer);
                }
                memcpy(gpu_vec, st->x, cfg->dim * sizeof(float));
            }

            if (cpu_attn_ckpt && layer < cpu_attn_layers_written) {
                const float *cpu_vec = cpu_attn_ckpt + (size_t)layer * (size_t)cfg->dim;
                const float *gpu_cmp = gpu_attn_snapshot ? gpu_attn_snapshot : gpu_vec;
                double l1 = 0.0;
                float linf = 0.0f;
                int max_idx = 0;
                compare_layer_checkpoint(gpu_cmp, cpu_vec, cfg->dim, &l1, &linf, &max_idx);
                fprintf(stderr,
                        "[glm-metal-parity] stage=attn layer=%d l1=%.6e linf=%.6e idx=%d gpu=%.6f cpu=%.6f\n",
                        layer,
                        l1,
                        (double)linf,
                        max_idx,
                        gpu_cmp[max_idx],
                        cpu_vec[max_idx]);
                if (linf > parity_tol) parity_failed = 1;
            }

            if (cpu_layer_ckpt && layer < cpu_layers_written) {
                const float *cpu_vec = cpu_layer_ckpt + (size_t)layer * (size_t)cfg->dim;
                double l1 = 0.0;
                float linf = 0.0f;
                int max_idx = 0;
                compare_layer_checkpoint(gpu_vec, cpu_vec, cfg->dim, &l1, &linf, &max_idx);
                fprintf(stderr,
                        "[glm-metal-parity] stage=ffn layer=%d l1=%.6e linf=%.6e idx=%d gpu=%.6f cpu=%.6f\n",
                        layer,
                        l1,
                        (double)linf,
                        max_idx,
                        gpu_vec[max_idx],
                        cpu_vec[max_idx]);
                if (linf > parity_tol) parity_failed = 1;
            }
        }
    }

    // Step 3: Final RMSNorm + LM head (TODO: implement GPU kernel)
    // For now, read output back to CPU and do final steps there.
    id<MTLBuffer> final_output = (n_layers % 2 == 0) ? buf_x : buf_x_out;

    // Step 4: Read back to CPU for final processing.
    if (cb != nil) {
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) {
            fprintf(stderr, "[glm-metal] native forward failed: %s\n", cb.error.localizedDescription.UTF8String);
            free(cpu_layer_ckpt);
            free(cpu_attn_ckpt);
            free(gpu_attn_snapshot);
            return -1;
        }
    }

    // Final processing on CPU for now (RMSNorm + LM head)
    memcpy(st->x, final_output.contents, cfg->dim * sizeof(float));

    // Do final steps on CPU
    rmsnorm_cpu(st->xn, st->x, rt->output_norm, cfg->dim);
    matvec_q80_cpu(rt->out_q, rt->out_s, st->xn, st->logits, cfg->vocab_size, cfg->dim, gs);

    free(cpu_layer_ckpt);
    free(cpu_attn_ckpt);
    free(gpu_attn_snapshot);
    if (parity_failed) {
        fprintf(stderr,
                "[glm-metal] native parity failed: at least one layer exceeded tol=%.6g\n",
                (double)parity_tol);
        return -2;
    }

    return 0;
}

int glm_metal_forward_token(const Runtime *rt, RunState *st, int token, int pos, int debug_mode) {
    if (!rt || !st) return -1;
    if (pos < 0 || pos >= st->cache_cap) {
        fprintf(stderr, "[glm-metal] invalid position: %d (cache_cap=%d)\n", pos, st->cache_cap);
        return -1;
    }
    if (st->n_attn_layers < 0 || st->n_attn_layers > rt->cfg.n_layers) {
        fprintf(stderr, "[glm-metal] invalid attention layer count: %d (n_layers=%d)\n", st->n_attn_layers, rt->cfg.n_layers);
        return -1;
    }

    RunState backup_state;
    memset(&backup_state, 0, sizeof(backup_state));
    int backup_ready = 0;

    int parity_mode = metal_native_parity_enabled();
    int parity_this_pos = parity_mode && (pos == 0 || metal_native_parity_all_pos());

    if (metal_native_enabled()) {
        if (!metal_native_unsafe_enabled()) {
            if (!g_warned_native_requires_unsafe) {
                fprintf(stderr,
                        "[glm-metal] GLM_METAL_NATIVE is set, but native kernels are experimental; "
                        "set GLM_METAL_NATIVE_UNSAFE=1 to execute them\n");
                g_warned_native_requires_unsafe = 1;
            }
        } else {
            int strict_mode = metal_native_strict();
            int run_native_this_pos = (!parity_mode || parity_this_pos);
            int need_backup = run_native_this_pos && ((!strict_mode) || parity_this_pos);

            if (need_backup) {
                if (runstate_backup_from(rt, st, &backup_state) == 0) {
                    backup_ready = 1;
                } else {
                    fprintf(stderr, "[glm-metal] native fallback backup failed; disabling fallback for this token\n");
                }
            }

            if (run_native_this_pos) {
                int st_native = glm_metal_forward_token_native(rt, st, token, pos, debug_mode);
                if (st_native == 0 && !parity_this_pos) {
                    if (backup_ready) runstate_backup_free(&backup_state);
                    return 0;
                }

                if (strict_mode && st_native != 0) {
                    if (backup_ready) runstate_backup_free(&backup_state);
                    fprintf(stderr, "[glm-metal] native strict mode enabled; aborting on native failure\n");
                    return st_native;
                }

                if (!backup_ready) {
                    fprintf(stderr, "[glm-metal] native path needs fallback restore but no backup exists\n");
                    return st_native != 0 ? st_native : -1;
                }

                runstate_restore_from_backup(rt, &backup_state, st);
                runstate_backup_free(&backup_state);

                if (parity_this_pos && st_native == 0) {
                    fprintf(stderr, "[glm-metal] native parity probe complete; executing CPU reference path\n");
                } else {
                    fprintf(stderr, "[glm-metal] native forward failed, falling back to CPU-compatible path\n");
                }
            }
        }
    }

    int prev_mode = glm_backend_metal_matvec_enabled();
    metal_tune_omp_if_needed();
    if (metal_hybrid_matvec_enabled()) {
        glm_set_metal_matvec_mode(1);
    }
    int status = glm_cpu_forward_token(rt, st, token, pos, debug_mode);
    if (!prev_mode) {
        glm_set_metal_matvec_mode(0);
    }
    return status;
}

int glm_metal_matvec_rows(const int8_t *qmat, const float *smat, const float *x, float *y, int rows, int dim, int gs) {
    if (!g_device || !g_queue) return -1;
    if (!qmat || !smat || !x || !y) return -1;
    if (rows <= 0 || dim <= 0 || gs <= 0 || (dim % gs) != 0) return -1;

    const NSUInteger q_bytes = (NSUInteger)rows * (NSUInteger)dim * sizeof(int8_t);
    const NSUInteger s_bytes = (NSUInteger)rows * (NSUInteger)(dim / gs) * sizeof(float);
    const NSUInteger x_bytes = (NSUInteger)dim * sizeof(float);
    const NSUInteger y_bytes = (NSUInteger)rows * sizeof(float);

    id<MTLBuffer> q_buf = cached_nocopy_buffer(qmat, q_bytes, "qmat");
    id<MTLBuffer> s_buf = cached_nocopy_buffer(smat, s_bytes, "smat");
    id<MTLBuffer> x_buf = cached_nocopy_buffer(x, x_bytes, "x");
    id<MTLBuffer> y_buf = cached_nocopy_buffer(y, y_bytes, "y");

    if (!q_buf || !s_buf || !x_buf || !y_buf) return -1;
    if (!g_ps_matvec || !g_matvec_args) return -1;

    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    if (!cb) return -1;
    id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
    if (!encoder) return -1;

    uint32_t *args = (uint32_t *)g_matvec_args.contents;
    args[0] = (uint32_t)rows;
    args[1] = (uint32_t)dim;
    args[2] = (uint32_t)gs;

    // Use SIMD-group cooperative reduction: 32 threads (1 simdgroup) per row
    const unsigned int simd_size = 32;
    const unsigned int tgx = (unsigned int)rows;  // One threadgroup per row
    static const NSUInteger offsets[4] = {0, 0, 0, 0};
    id<MTLBuffer> mtl_buffers[4] = {q_buf, s_buf, x_buf, y_buf};
    [encoder setComputePipelineState:g_ps_matvec];
    [encoder setBuffer:g_matvec_args offset:0 atIndex:0];
    [encoder setBuffers:mtl_buffers offsets:offsets withRange:NSMakeRange(1, 4)];
    [encoder dispatchThreadgroups:MTLSizeMake(tgx, 1, 1) 
              threadsPerThreadgroup:MTLSizeMake(simd_size, 1, 1)];
    [encoder endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) {
        fprintf(stderr, "[glm-metal] matvec command failed: %s\n", cb.error.localizedDescription.UTF8String);
        return -1;
    }
    return 0;
}

int glm_metal_microbench(const Runtime *rt, const char *kernel_family, int iters, int warmup) {
    if (!g_device || !g_queue || !g_pipelines || !rt || !kernel_family) return -1;
    if (iters <= 0) iters = 64;
    if (warmup < 0) warmup = 0;

    double *samples = (double *)calloc((size_t)iters, sizeof(double));
    double *sorted = (double *)calloc((size_t)iters, sizeof(double));
    if (!samples || !sorted) {
        free(samples);
        free(sorted);
        return -1;
    }

    id<MTLComputePipelineState> ps0 = nil;
    id<MTLComputePipelineState> ps1 = nil;
    id<MTLComputePipelineState> ps2 = nil;
    NSUInteger active_bytes = 0;

    uint32_t mat_rows = 0, mat_dim = 0, mat_gs = 0;
    uint32_t vec_n = 0;
    uint32_t rope_dim = 0, rope_pos = 127;
    uint32_t att_kv = 0, att_rope = 0, att_pos = 0;
    float att_scale = 1.0f;

    id<MTLBuffer> b1 = nil;
    id<MTLBuffer> b2 = nil;
    id<MTLBuffer> b3 = nil;
    id<MTLBuffer> b4 = nil;
    id<MTLBuffer> b5 = nil;
    id<MTLBuffer> b6 = nil;

    if (strcmp(kernel_family, "matvec_q80_rows") == 0 || strcmp(kernel_family, "matvec_q80_rows_simdgroup") == 0) {
        mat_rows = (uint32_t)rt->cfg.hidden_dim;
        if (mat_rows < 64u) mat_rows = 64u;
        if (mat_rows > 4096u) mat_rows = 4096u;
        mat_dim = (uint32_t)rt->cfg.dim;
        mat_gs = (uint32_t)rt->cfg.group_size;
        if (mat_dim == 0 || mat_gs == 0 || (mat_dim % mat_gs) != 0) {
            fprintf(stderr, "[glm-kbench] invalid matvec shape (rows=%u dim=%u gs=%u)\n", mat_rows, mat_dim, mat_gs);
            free(samples);
            free(sorted);
            return -1;
        }

        NSUInteger q_bytes = (NSUInteger)mat_rows * (NSUInteger)mat_dim * sizeof(int8_t);
        NSUInteger s_bytes = (NSUInteger)mat_rows * (NSUInteger)(mat_dim / mat_gs) * sizeof(float);
        NSUInteger x_bytes = (NSUInteger)mat_dim * sizeof(float);
        NSUInteger y_bytes = (NSUInteger)mat_rows * sizeof(float);
        b1 = make_shared_buffer(q_bytes, "kbench_q");
        b2 = make_shared_buffer(s_bytes, "kbench_s");
        b3 = make_shared_buffer(x_bytes, "kbench_x");
        b4 = make_shared_buffer(y_bytes, "kbench_y");
        if (!b1 || !b2 || !b3 || !b4) {
            free(samples);
            free(sorted);
            return -1;
        }
        memset(b1.contents, 1, q_bytes);
        float *s = (float *)b2.contents;
        for (NSUInteger i = 0; i < s_bytes / sizeof(float); i++) s[i] = 0.01f;
        float *x = (float *)b3.contents;
        for (NSUInteger i = 0; i < x_bytes / sizeof(float); i++) x[i] = 1.0f;

        if (strcmp(kernel_family, "matvec_q80_rows_simdgroup") == 0) {
            ps0 = g_pipelines[@"matvec_q80_rows_simdgroup"];
        } else {
            ps0 = g_ps_matvec;
        }
        active_bytes = q_bytes + s_bytes + x_bytes + y_bytes;
    } else if (strcmp(kernel_family, "rmsnorm_f32") == 0) {
        vec_n = (uint32_t)rt->cfg.dim;
        if (vec_n == 0) vec_n = 1024;
        NSUInteger bytes = (NSUInteger)vec_n * sizeof(float);
        b1 = make_shared_buffer(bytes, "kbench_x");
        b2 = make_shared_buffer(bytes, "kbench_w");
        b3 = make_shared_buffer(bytes, "kbench_y");
        if (!b1 || !b2 || !b3) {
            free(samples);
            free(sorted);
            return -1;
        }
        float *x = (float *)b1.contents;
        float *w = (float *)b2.contents;
        for (NSUInteger i = 0; i < bytes / sizeof(float); i++) {
            x[i] = 1.0f;
            w[i] = 1.0f;
        }
        ps0 = g_pipelines[@"rmsnorm_f32"];
        active_bytes = bytes * 3u;
    } else if (strcmp(kernel_family, "rope_inplace") == 0) {
        rope_dim = rt->cfg.rope_dim > 0 ? (uint32_t)rt->cfg.rope_dim : (uint32_t)rt->cfg.dim;
        if (rope_dim < 2u) rope_dim = 2u;
        if ((rope_dim & 1u) != 0u) rope_dim--;
        NSUInteger bytes = (NSUInteger)rope_dim * sizeof(float);
        b1 = make_shared_buffer(bytes, "kbench_rope_x");
        if (!b1) {
            free(samples);
            free(sorted);
            return -1;
        }
        float *x = (float *)b1.contents;
        for (NSUInteger i = 0; i < bytes / sizeof(float); i++) x[i] = (float)(i % 17) * 0.125f;
        ps0 = g_pipelines[@"rope_inplace"];
        active_bytes = bytes * 2u;
    } else if (strcmp(kernel_family, "attention_scores_context") == 0) {
        att_kv = (uint32_t)rt->cfg.kv_lora_rank;
        att_rope = (uint32_t)rt->cfg.rope_dim;
        if (att_kv == 0 || att_rope == 0) {
            fprintf(stderr, "[glm-kbench] attention kernels require kv_lora_rank and rope_dim\n");
            free(samples);
            free(sorted);
            return -1;
        }
        att_pos = 255u;
        if ((uint32_t)rt->cfg.seq_len > 0 && att_pos + 1u > (uint32_t)rt->cfg.seq_len) {
            att_pos = (uint32_t)rt->cfg.seq_len - 1u;
        }
        if (att_pos < 1u) att_pos = 1u;
        uint32_t tlen = att_pos + 1u;
        uint32_t head_k = (uint32_t)rt->cfg.l0_head_k_dim;
        if (head_k == 0) head_k = att_kv + att_rope;
        att_scale = 1.0f / sqrtf((float)head_k);

        NSUInteger q_abs_bytes = (NSUInteger)att_kv * sizeof(float);
        NSUInteger q_pe_bytes = (NSUInteger)att_rope * sizeof(float);
        NSUInteger k_comp_bytes = (NSUInteger)tlen * (NSUInteger)att_kv * sizeof(float);
        NSUInteger k_pe_bytes = (NSUInteger)tlen * (NSUInteger)att_rope * sizeof(float);
        NSUInteger scores_bytes = (NSUInteger)tlen * sizeof(float);
        NSUInteger ctx_bytes = (NSUInteger)att_kv * sizeof(float);

        b1 = make_shared_buffer(q_abs_bytes, "kbench_q_abs");
        b2 = make_shared_buffer(q_pe_bytes, "kbench_q_pe");
        b3 = make_shared_buffer(k_comp_bytes, "kbench_k_comp");
        b4 = make_shared_buffer(k_pe_bytes, "kbench_k_pe");
        b5 = make_shared_buffer(scores_bytes, "kbench_scores");
        b6 = make_shared_buffer(ctx_bytes, "kbench_ctx");
        if (!b1 || !b2 || !b3 || !b4 || !b5 || !b6) {
            free(samples);
            free(sorted);
            return -1;
        }

        float *q_abs = (float *)b1.contents;
        for (NSUInteger i = 0; i < q_abs_bytes / sizeof(float); i++) q_abs[i] = 0.25f;
        float *q_pe = (float *)b2.contents;
        for (NSUInteger i = 0; i < q_pe_bytes / sizeof(float); i++) q_pe[i] = 0.125f;
        float *k_comp = (float *)b3.contents;
        for (NSUInteger i = 0; i < k_comp_bytes / sizeof(float); i++) k_comp[i] = 0.03125f;
        float *k_pe = (float *)b4.contents;
        for (NSUInteger i = 0; i < k_pe_bytes / sizeof(float); i++) k_pe[i] = 0.015625f;

        ps0 = g_pipelines[@"attention_scores"];
        ps1 = g_pipelines[@"softmax_1d"];
        ps2 = g_pipelines[@"attention_context"];
        active_bytes = q_abs_bytes + q_pe_bytes + (k_comp_bytes * 2u) + k_pe_bytes + (scores_bytes * 2u) + ctx_bytes;
    } else {
        fprintf(stderr, "[glm-kbench] unknown kernel family: %s\n", kernel_family);
        free(samples);
        free(sorted);
        return -1;
    }

    if (!ps0) {
        fprintf(stderr, "[glm-kbench] missing pipeline for %s\n", kernel_family);
        free(samples);
        free(sorted);
        return -1;
    }

    const int total = warmup + iters;
    for (int i = 0; i < total; i++) {
        double t0 = monotonic_ms();
        id<MTLCommandBuffer> cb = [g_queue commandBuffer];
        if (!cb) {
            free(samples);
            free(sorted);
            return -1;
        }
        id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
        if (!encoder) {
            free(samples);
            free(sorted);
            return -1;
        }

        if (strcmp(kernel_family, "matvec_q80_rows") == 0) {
            uint32_t *args = (uint32_t *)g_matvec_args.contents;
            args[0] = mat_rows;
            args[1] = mat_dim;
            args[2] = mat_gs;
            const unsigned int tgs = 128;
            const unsigned int tgx = (mat_rows + tgs - 1u) / tgs;
            static const NSUInteger offsets[4] = {0, 0, 0, 0};
            id<MTLBuffer> mtl_buffers[4] = {b1, b2, b3, b4};
            [encoder setComputePipelineState:ps0];
            [encoder setBuffer:g_matvec_args offset:0 atIndex:0];
            [encoder setBuffers:mtl_buffers offsets:offsets withRange:NSMakeRange(1, 4)];
            [encoder dispatchThreadgroups:MTLSizeMake(tgx, 1, 1) threadsPerThreadgroup:MTLSizeMake(tgs, 1, 1)];
        } else if (strcmp(kernel_family, "matvec_q80_rows_simdgroup") == 0) {
            uint32_t *args = (uint32_t *)g_matvec_args.contents;
            args[0] = mat_rows;
            args[1] = mat_dim;
            args[2] = mat_gs;
            // simdgroup variant: one row per simdgroup (32 threads)
            const unsigned int simd_size = 32;
            const unsigned int tgx = mat_rows;
            static const NSUInteger offsets[4] = {0, 0, 0, 0};
            id<MTLBuffer> mtl_buffers[4] = {b1, b2, b3, b4};
            [encoder setComputePipelineState:ps0];
            [encoder setBuffer:g_matvec_args offset:0 atIndex:0];
            [encoder setBuffers:mtl_buffers offsets:offsets withRange:NSMakeRange(1, 4)];
            [encoder dispatchThreadgroups:MTLSizeMake(tgx, 1, 1) threadsPerThreadgroup:MTLSizeMake(simd_size, 1, 1)];
        } else if (strcmp(kernel_family, "rmsnorm_f32") == 0) {
            struct {
                uint32_t n;
            } args = {vec_n};
            const unsigned int tgs = 256;
            const unsigned int tgx = (vec_n + tgs - 1u) / tgs;
            void *buffers[] = {(__bridge void *)b1, (__bridge void *)b2, (__bridge void *)b3};
            dispatch(encoder, "rmsnorm_f32", tgx, tgs, &args, sizeof(args), buffers, 3);
        } else if (strcmp(kernel_family, "rope_inplace") == 0) {
            struct {
                uint32_t dim;
                uint32_t pos;
            } args = {rope_dim, rope_pos};
            const unsigned int tgs = 256;
            const unsigned int tgx = (rope_dim + tgs - 1u) / tgs;
            void *buffers[] = {(__bridge void *)b1};
            dispatch(encoder, "rope_inplace", tgx, tgs, &args, sizeof(args), buffers, 1);
        } else {
            struct {
                uint32_t kv_lora_rank;
                uint32_t rope_dim;
                uint32_t pos;
                float kq_scale;
            } score_args = {att_kv, att_rope, att_pos, att_scale};
            struct {
                uint32_t n;
            } softmax_args = {att_pos + 1u};
            struct {
                uint32_t kv_lora_rank;
                uint32_t pos;
            } ctx_args = {att_kv, att_pos};

            void *score_buffers[] = {
                (__bridge void *)b1,
                (__bridge void *)b2,
                (__bridge void *)b3,
                (__bridge void *)b4,
                (__bridge void *)b5,
            };
            const unsigned int score_tgs = 128;
            const unsigned int score_tgx = (att_pos + 1u + score_tgs - 1u) / score_tgs;
            dispatch(encoder, "attention_scores", score_tgx, score_tgs, &score_args, sizeof(score_args), score_buffers, 5);

            void *softmax_buffers[] = {(__bridge void *)b5};
            dispatch(encoder, "softmax_1d", 1, 1, &softmax_args, sizeof(softmax_args), softmax_buffers, 1);

            void *ctx_buffers[] = {(__bridge void *)b5, (__bridge void *)b3, (__bridge void *)b6};
            const unsigned int ctx_tgs = 128;
            const unsigned int ctx_tgx = (att_kv + ctx_tgs - 1u) / ctx_tgs;
            dispatch(encoder, "attention_context", ctx_tgx, ctx_tgs, &ctx_args, sizeof(ctx_args), ctx_buffers, 3);
        }

        [encoder endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        if (cb.error) {
            fprintf(stderr, "[glm-kbench] command failed: %s\n", cb.error.localizedDescription.UTF8String);
            free(samples);
            free(sorted);
            return -1;
        }

        double elapsed = monotonic_ms() - t0;
        if (i >= warmup) samples[i - warmup] = elapsed;
    }

    double total_ms = 0.0;
    for (int i = 0; i < iters; i++) {
        total_ms += samples[i];
        sorted[i] = samples[i];
    }
    qsort(sorted, (size_t)iters, sizeof(double), cmp_double_asc);
    double avg_ms = total_ms / (double)iters;
    double p50 = percentile(sorted, iters, 50.0);
    double p95 = percentile(sorted, iters, 95.0);
    double bw_gbps = (avg_ms > 0.0) ? ((double)active_bytes / (avg_ms / 1000.0) / 1e9) : 0.0;

    fprintf(stderr,
            "[glm-kbench] kernel=%s iters=%d warmup=%d avg_ms=%.4f p50_ms=%.4f p95_ms=%.4f active_bytes=%lu est_bw_gbps=%.3f\n",
            kernel_family, iters, warmup, avg_ms, p50, p95, (unsigned long)active_bytes, bw_gbps);

    free(samples);
    free(sorted);
    return 0;
}

void glm_metal_free(void) {
    // Clean up persistent command buffer
    if (g_persistent_encoder) {
        [g_persistent_encoder endEncoding];
        g_persistent_encoder = nil;
    }
    if (g_persistent_cb) {
        [g_persistent_cb commit];
        [g_persistent_cb waitUntilCompleted];
        g_persistent_cb = nil;
    }
    g_matvec_batch_count = 0;
    
    // Free layer buffers
    if (g_layers) {
        free(g_layers);
        g_layers = NULL;
    }
    g_n_layers = 0;
    
    g_run = (MetalRunStateBuffers){0};
    g_model = (MetalModelBuffers){0};
    g_matvec_args = nil;
    g_ps_matvec = nil;
    g_nocopy_cache = nil;
    g_pipelines = nil;
    g_library = nil;
    g_queue = nil;
    g_device = nil;
}
