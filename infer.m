#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "model.h"
#include "backend.h"

static id<MTLDevice> g_device = nil;
static id<MTLCommandQueue> g_queue = nil;
static id<MTLLibrary> g_library = nil;
static NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *g_pipelines = nil;

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
    id<MTLBuffer> moe_gate;
    id<MTLBuffer> moe_up;
    id<MTLBuffer> moe_router;
    id<MTLBuffer> moe_out_acc;
    id<MTLBuffer> moe_topk_idx;
    id<MTLBuffer> moe_topk_w;
} MetalRunStateBuffers;

typedef struct {
    id<MTLBuffer> output_norm;
    id<MTLBuffer> tok_q;
    id<MTLBuffer> tok_s;
    id<MTLBuffer> out_q;
    id<MTLBuffer> out_s;
} MetalModelBuffers;

static MetalRunStateBuffers g_run = {0};
static MetalModelBuffers g_model = {0};

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
    const char *required[] = {
        "embed_dequant_q80",
        "rmsnorm_f32",
        "matvec_q80_rows",
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

    const size_t dim = (size_t)rt->cfg.dim;
    const size_t vocab = (size_t)rt->cfg.vocab_size;
    const size_t gs = (size_t)rt->cfg.group_size;
    g_model.output_norm = make_nocopy_buffer(rt->output_norm, dim * sizeof(float), "output_norm");
    g_model.tok_q = make_nocopy_buffer(rt->tok_q, vocab * dim * sizeof(int8_t), "tok_q");
    g_model.tok_s = make_nocopy_buffer(rt->tok_s, vocab * (dim / gs) * sizeof(float), "tok_s");
    g_model.out_q = make_nocopy_buffer(rt->out_q, vocab * dim * sizeof(int8_t), "out_q");
    g_model.out_s = make_nocopy_buffer(rt->out_s, vocab * (dim / gs) * sizeof(float), "out_s");
    if (!g_model.output_norm || !g_model.tok_q || !g_model.tok_s || !g_model.out_q || !g_model.out_s) return -1;

    fprintf(stderr, "[glm-metal] initialized device=%s\n", g_device.name.UTF8String);
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

    g_run.moe_gate = make_shared_buffer((NSUInteger)cfg->moe_ffn_dim * sizeof(float), "moe_gate");
    g_run.moe_up = make_shared_buffer((NSUInteger)cfg->moe_ffn_dim * sizeof(float), "moe_up");
    g_run.moe_router = make_shared_buffer((NSUInteger)cfg->n_routed_experts * sizeof(float), "moe_router");
    g_run.moe_out_acc = make_shared_buffer((NSUInteger)cfg->dim * sizeof(float), "moe_out_acc");
    g_run.moe_topk_idx = make_shared_buffer((NSUInteger)cfg->n_experts_used * sizeof(int), "moe_topk_idx");
    g_run.moe_topk_w = make_shared_buffer((NSUInteger)cfg->n_experts_used * sizeof(float), "moe_topk_w");
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

    glm_set_metal_matvec_mode(1);
    int status = glm_cpu_forward_token(rt, st, token, pos, debug_mode);
    glm_set_metal_matvec_mode(0);
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

    id<MTLBuffer> q_buf = [g_device newBufferWithBytesNoCopy:(void *)qmat
                                                       length:q_bytes
                                                      options:MTLResourceStorageModeShared
                                                  deallocator:nil];
    id<MTLBuffer> s_buf = [g_device newBufferWithBytesNoCopy:(void *)smat
                                                       length:s_bytes
                                                      options:MTLResourceStorageModeShared
                                                  deallocator:nil];
    id<MTLBuffer> x_buf = [g_device newBufferWithBytesNoCopy:(void *)x
                                                       length:x_bytes
                                                      options:MTLResourceStorageModeShared
                                                  deallocator:nil];
    id<MTLBuffer> y_buf = [g_device newBufferWithBytesNoCopy:(void *)y
                                                       length:y_bytes
                                                      options:MTLResourceStorageModeShared
                                                  deallocator:nil];

    if (!q_buf || !s_buf || !x_buf || !y_buf) return -1;

    id<MTLCommandBuffer> cb = [g_queue commandBuffer];
    if (!cb) return -1;
    id<MTLComputeCommandEncoder> encoder = [cb computeCommandEncoder];
    if (!encoder) return -1;

    struct {
        uint32_t rows;
        uint32_t dim;
        uint32_t gs;
    } args = {(uint32_t)rows, (uint32_t)dim, (uint32_t)gs};

    void *buffers[] = {
        (__bridge void *)q_buf,
        (__bridge void *)s_buf,
        (__bridge void *)x_buf,
        (__bridge void *)y_buf,
    };
    const unsigned int tgs = 128;
    const unsigned int tgx = ((unsigned int)rows + tgs - 1u) / tgs;
    dispatch(encoder, "matvec_q80_rows", tgx, tgs, &args, sizeof(args), buffers, 4);
    [encoder endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) {
        fprintf(stderr, "[glm-metal] matvec command failed: %s\n", cb.error.localizedDescription.UTF8String);
        return -1;
    }
    return 0;
}

void glm_metal_free(void) {
    g_run = (MetalRunStateBuffers){0};
    g_model = (MetalModelBuffers){0};
    g_pipelines = nil;
    g_library = nil;
    g_queue = nil;
    g_device = nil;
}
