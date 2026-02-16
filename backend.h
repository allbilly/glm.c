#ifndef GLM47_FLASH_BACKEND_H
#define GLM47_FLASH_BACKEND_H

#include <stdint.h>
#include "model.h"

int glm_backend_init(BackendType backend, const Runtime *rt, RunState *st);
int glm_backend_forward_token(int use_metal, const Runtime *rt, RunState *st, int token, int pos, int debug_mode);
void glm_backend_free(int use_metal);
int glm_backend_metal_microbench(int use_metal, const Runtime *rt, const char *kernel_family, int iters, int warmup);

void glm_set_metal_matvec_mode(int enabled);
int glm_backend_metal_matvec_enabled(void);
int glm_backend_try_metal_matvec(const int8_t *qmat, const float *smat, const float *x, float *y, int rows, int dim, int gs);

#endif
