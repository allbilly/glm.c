#include <stdio.h>
#include <stdlib.h>

#include "backend.h"

int glm_cpu_forward_token(const Runtime *rt, RunState *st, int token, int pos, int debug_mode);

#if defined(__APPLE__) && defined(GLM_ENABLE_METAL)
int glm_metal_init(const Runtime *rt, RunState *st);
void glm_metal_prepare(const Runtime *rt, RunState *st);
int glm_metal_forward_token(const Runtime *rt, RunState *st, int token, int pos, int debug_mode);
void glm_metal_free(void);
int glm_metal_matvec_rows(const int8_t *qmat, const float *smat, const float *x, float *y, int rows, int dim, int gs);
#else
static int glm_metal_init(const Runtime *rt, RunState *st) {
    (void)rt;
    (void)st;
    return -1;
}
static void glm_metal_prepare(const Runtime *rt, RunState *st) {
    (void)rt;
    (void)st;
}
static int glm_metal_forward_token(const Runtime *rt, RunState *st, int token, int pos, int debug_mode) {
    (void)rt;
    (void)st;
    (void)token;
    (void)pos;
    (void)debug_mode;
    return -1;
}
static void glm_metal_free(void) {}
static int glm_metal_matvec_rows(const int8_t *qmat, const float *smat, const float *x, float *y, int rows, int dim, int gs) {
    (void)qmat;
    (void)smat;
    (void)x;
    (void)y;
    (void)rows;
    (void)dim;
    (void)gs;
    return -1;
}
#endif

static int g_metal_matvec_mode = 0;
static int g_metal_matvec_min_rows = -1;

void glm_set_metal_matvec_mode(int enabled) {
    g_metal_matvec_mode = enabled ? 1 : 0;
}

int glm_backend_metal_matvec_enabled(void) {
    return g_metal_matvec_mode;
}

static int metal_matvec_min_rows(void) {
    if (g_metal_matvec_min_rows < 0) {
        const char *env = getenv("GLM_METAL_MATVEC_MIN_ROWS");
        g_metal_matvec_min_rows = env ? atoi(env) : 32768;
        if (g_metal_matvec_min_rows < 1) g_metal_matvec_min_rows = 1;
    }
    return g_metal_matvec_min_rows;
}

int glm_backend_try_metal_matvec(const int8_t *qmat, const float *smat, const float *x, float *y, int rows, int dim, int gs) {
    if (!g_metal_matvec_mode) return -1;
    if (rows < metal_matvec_min_rows()) return -1;
    return glm_metal_matvec_rows(qmat, smat, x, y, rows, dim, gs);
}

int glm_backend_init(BackendType backend, const Runtime *rt, RunState *st) {
    if (backend != BACKEND_METAL) return 0;
#if !defined(__APPLE__) || !defined(GLM_ENABLE_METAL)
    fprintf(stderr, "[glm-metal] unavailable on this platform, using CPU backend\n");
    return 0;
#else
    if (glm_metal_init(rt, st) != 0) {
        fprintf(stderr, "[glm-metal] init failed, falling back to CPU backend\n");
        return 0;
    }
    glm_metal_prepare(rt, st);
    return 1;
#endif
}

int glm_backend_forward_token(int use_metal, const Runtime *rt, RunState *st, int token, int pos, int debug_mode) {
    if (use_metal) return glm_metal_forward_token(rt, st, token, pos, debug_mode);
    return glm_cpu_forward_token(rt, st, token, pos, debug_mode);
}

void glm_backend_free(int use_metal) {
    if (use_metal) glm_metal_free();
}
