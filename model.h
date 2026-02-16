#ifndef GLM47_FLASH_MODEL_H
#define GLM47_FLASH_MODEL_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint32_t magic;
    uint32_t version;
    int32_t dim;
    int32_t hidden_dim;
    int32_t n_layers;
    int32_t n_heads;
    int32_t n_kv_heads;
    int32_t vocab_size;
    int32_t seq_len;
    int32_t rope_dim;
    int32_t q_lora_rank;
    int32_t kv_lora_rank;
    int32_t kv_mqa_width;
    int32_t n_routed_experts;
    int32_t n_experts_used;
    int32_t n_shared_experts;
    int32_t bos_id;
    int32_t eos_id;
    uint64_t group_size;
    uint64_t off_norm;
    uint64_t off_tok_q;
    uint64_t off_tok_s;
    uint64_t off_out_q;
    uint64_t off_out_s;
    uint64_t total_bytes;
    uint64_t off_l0_ffn_norm;
    uint64_t off_l0_ffn_gate_q;
    uint64_t off_l0_ffn_gate_s;
    uint64_t off_l0_ffn_up_q;
    uint64_t off_l0_ffn_up_s;
    uint64_t off_l0_ffn_down_q;
    uint64_t off_l0_ffn_down_s;
    int32_t l0_qk_nope_dim;
    int32_t l0_head_k_dim;
    int32_t l0_v_head_dim;
    int32_t l0_flags;
    int32_t moe_ffn_dim;
} RuntimeConfig;

typedef struct {
    const float *attn_norm;
    const float *q_a_norm;
    const float *kv_a_norm;
    const int8_t *q_a_q;
    const float *q_a_s;
    const int8_t *q_b_q;
    const float *q_b_s;
    const int8_t *kv_a_q;
    const float *kv_a_s;
    const int8_t *k_b_q;
    const float *k_b_s;
    const int8_t *v_b_q;
    const float *v_b_s;
    const int8_t *attn_out_q;
    const float *attn_out_s;
} LayerAttnWeights;

typedef struct {
    const float *ffn_norm;
    const float *gate_inp;
    const float *exp_probs_b;
    const int8_t *gate_sh_q;
    const float *gate_sh_s;
    const int8_t *up_sh_q;
    const float *up_sh_s;
    const int8_t *down_sh_q;
    const float *down_sh_s;
} LayerMoeShared;

typedef struct {
    const int8_t *gate_q;
    const float *gate_s;
    const int8_t *up_q;
    const float *up_s;
    const int8_t *down_q;
    const float *down_s;
} LayerMoeRouted;

typedef struct Runtime {
    RuntimeConfig cfg;
    unsigned char *data;
    size_t file_size;

    const float *output_norm;
    const int8_t *tok_q;
    const float *tok_s;
    const int8_t *out_q;
    const float *out_s;

    int has_l0_ffn;
    const float *l0_ffn_norm;
    const int8_t *l0_ffn_gate_q;
    const float *l0_ffn_gate_s;
    const int8_t *l0_ffn_up_q;
    const float *l0_ffn_up_s;
    const int8_t *l0_ffn_down_q;
    const float *l0_ffn_down_s;

    int has_l0_attn;
    const float *l0_attn_norm;
    const float *l0_q_a_norm;
    const float *l0_kv_a_norm;
    const int8_t *l0_q_a_q;
    const float *l0_q_a_s;
    const int8_t *l0_q_b_q;
    const float *l0_q_b_s;
    const int8_t *l0_kv_a_q;
    const float *l0_kv_a_s;
    const int8_t *l0_k_b_q;
    const float *l0_k_b_s;
    const int8_t *l0_v_b_q;
    const float *l0_v_b_s;
    const int8_t *l0_attn_out_q;
    const float *l0_attn_out_s;

    int has_all_attn;
    LayerAttnWeights *attn_layers;

    int has_moe_shared;
    LayerMoeShared *moe_layers;
    int has_moe_routed;
    LayerMoeRouted *moe_routed_layers;
} Runtime;

typedef struct RunState {
    float *x;
    float *xn;
    float *xb;
    float *ff_gate;
    float *ff_up;
    float *logits;

    int cache_cap;
    int n_attn_layers;
    float *k_cache_comp;
    float *k_cache_pe;
    float *q_a;
    float *q_full;
    float *kv_a;
    float *q_nope;
    float *q_pe;
    float *q_abs;
    float *att_scores;
    float *att_ctx;
    float *att_concat;
    float *moe_gate;
    float *moe_up;
    float *moe_router;
    float *moe_out_acc;
    int *moe_topk_idx;
    float *moe_topk_w;
} RunState;

typedef enum {
    BACKEND_CPU = 0,
    BACKEND_METAL = 1,
} BackendType;

int glm_cpu_forward_token(const Runtime *rt, RunState *st, int token, int pos, int debug_mode);
int glm_app_main(int argc, char **argv);

#endif
