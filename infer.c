#include <ctype.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "backend.h"

#if defined(_OPENMP)
#include <omp.h>
#endif

#if defined(GLM_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#define GLM_HAS_CBLAS 1
#elif defined(GLM_USE_CBLAS)
#include <cblas.h>
#define GLM_HAS_CBLAS 1
#else
#define GLM_HAS_CBLAS 0
#endif

#if defined _WIN32
#include "win.h"
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

#define MAGIC 0x67343763u
#define VERSION 4u
#define HEADER_SIZE 256
#define PROMPT_BUFFER_SIZE 32768

#define FLAG_ALL_ATTN 1
#define FLAG_NATIVE_COVERAGE_FULL 2

typedef struct {
    char *bytes;
    uint32_t len;
} Token;

typedef struct {
    uint32_t max_token_len;
    uint32_t bos_id;
    uint32_t eos_id;
    size_t n_tokens;
    Token *tokens;
    float *scores;
    int ascii_map[256];
    uint16_t byte_to_cp[256];
    int16_t cp_to_byte[512];
} Tokenizer;


static int encode_prompt_bpe(const Tokenizer *tok, const char *prompt, int *ids, int max_ids, int bos);
static void assert_finite_slice(const char *name, const float *x, int n, int debug_mode);

static void die(const char *msg) {
    fprintf(stderr, "%s\n", msg);
    exit(EXIT_FAILURE);
}

static int64_t file_size_of(const char *path) {
    struct stat st;
    if (stat(path, &st) != 0) return -1;
    return st.st_size;
}

static int utf8_encode_cp(uint32_t cp, char out[4]) {
    if (cp <= 0x7Fu) {
        out[0] = (char)cp;
        return 1;
    }
    if (cp <= 0x7FFu) {
        out[0] = (char)(0xC0u | (cp >> 6));
        out[1] = (char)(0x80u | (cp & 0x3Fu));
        return 2;
    }
    if (cp <= 0xFFFFu) {
        out[0] = (char)(0xE0u | (cp >> 12));
        out[1] = (char)(0x80u | ((cp >> 6) & 0x3Fu));
        out[2] = (char)(0x80u | (cp & 0x3Fu));
        return 3;
    }
    out[0] = (char)(0xF0u | (cp >> 18));
    out[1] = (char)(0x80u | ((cp >> 12) & 0x3Fu));
    out[2] = (char)(0x80u | ((cp >> 6) & 0x3Fu));
    out[3] = (char)(0x80u | (cp & 0x3Fu));
    return 4;
}

static size_t utf8_decode_cp(const unsigned char *s, size_t len, uint32_t *cp) {
    if (len == 0) return 0;
    unsigned char b0 = s[0];
    if (b0 < 0x80u) {
        *cp = b0;
        return 1;
    }
    if ((b0 & 0xE0u) == 0xC0u && len >= 2) {
        unsigned char b1 = s[1];
        if ((b1 & 0xC0u) == 0x80u) {
            uint32_t v = ((uint32_t)(b0 & 0x1Fu) << 6) | (uint32_t)(b1 & 0x3Fu);
            if (v >= 0x80u) {
                *cp = v;
                return 2;
            }
        }
    } else if ((b0 & 0xF0u) == 0xE0u && len >= 3) {
        unsigned char b1 = s[1];
        unsigned char b2 = s[2];
        if ((b1 & 0xC0u) == 0x80u && (b2 & 0xC0u) == 0x80u) {
            uint32_t v = ((uint32_t)(b0 & 0x0Fu) << 12) | ((uint32_t)(b1 & 0x3Fu) << 6) | (uint32_t)(b2 & 0x3Fu);
            if (v >= 0x800u && !(v >= 0xD800u && v <= 0xDFFFu)) {
                *cp = v;
                return 3;
            }
        }
    } else if ((b0 & 0xF8u) == 0xF0u && len >= 4) {
        unsigned char b1 = s[1];
        unsigned char b2 = s[2];
        unsigned char b3 = s[3];
        if ((b1 & 0xC0u) == 0x80u && (b2 & 0xC0u) == 0x80u && (b3 & 0xC0u) == 0x80u) {
            uint32_t v = ((uint32_t)(b0 & 0x07u) << 18) |
                         ((uint32_t)(b1 & 0x3Fu) << 12) |
                         ((uint32_t)(b2 & 0x3Fu) << 6) |
                         (uint32_t)(b3 & 0x3Fu);
            if (v >= 0x10000u && v <= 0x10FFFFu) {
                *cp = v;
                return 4;
            }
        }
    }
    *cp = b0;
    return 1;
}

static void build_byte_unicode_maps(Tokenizer *tok) {
    int bs[256];
    int cs[256];
    int n = 0;
    for (int b = 33; b <= 126; b++) bs[n++] = b;
    for (int b = 161; b <= 172; b++) bs[n++] = b;
    for (int b = 174; b <= 255; b++) bs[n++] = b;
    for (int i = 0; i < n; i++) cs[i] = bs[i];

    int extra = 0;
    for (int b = 0; b < 256; b++) {
        int present = 0;
        for (int i = 0; i < n; i++) {
            if (bs[i] == b) {
                present = 1;
                break;
            }
        }
        if (!present) {
            bs[n] = b;
            cs[n] = 256 + extra;
            n++;
            extra++;
        }
    }
    if (n != 256) die("invalid byte-unicode map construction");

    for (int i = 0; i < 256; i++) tok->byte_to_cp[i] = 0;
    for (int i = 0; i < 512; i++) tok->cp_to_byte[i] = -1;

    for (int i = 0; i < 256; i++) {
        int b = bs[i];
        int cp = cs[i];
        tok->byte_to_cp[b] = (uint16_t)cp;
        if (cp >= 0 && cp < 512) tok->cp_to_byte[cp] = (int16_t)b;
    }
}

static int append_decoded_token_bytes(const Tokenizer *tok, const Token *t, char *out, size_t *out_n, size_t out_cap) {
    if (!tok || !t || !t->bytes) return 0;
    const unsigned char *s = (const unsigned char *)t->bytes;
    size_t i = 0;
    while (i < t->len) {
        uint32_t cp = 0;
        size_t used = utf8_decode_cp(s + i, (size_t)t->len - i, &cp);
        if (used == 0) return 0;
        int mapped = (cp < 512u) ? tok->cp_to_byte[cp] : -1;
        if (mapped >= 0) {
            if (*out_n + 1 >= out_cap) return 0;
            out[(*out_n)++] = (char)mapped;
        } else {
            if (*out_n + used >= out_cap) return 0;
            memcpy(out + *out_n, s + i, used);
            *out_n += used;
        }
        i += used;
    }
    return 1;
}

static void putchar_safe_byte(unsigned char c) {
    if (c == '\n' || c == '\t' || c >= 0x20u) putchar((int)c);
    else putchar(' ');
}

static void tokenizer_init(Tokenizer *tok) {
    memset(tok, 0, sizeof(*tok));
    for (int i = 0; i < 256; i++) tok->ascii_map[i] = -1;
    build_byte_unicode_maps(tok);
}

static void tokenizer_free(Tokenizer *tok) {
    if (tok->tokens != NULL) {
        for (size_t i = 0; i < tok->n_tokens; i++) free(tok->tokens[i].bytes);
        free(tok->tokens);
    }
    free(tok->scores);
    tokenizer_init(tok);
}

static void build_ascii_map(Tokenizer *tok) {
    for (size_t i = 0; i < tok->n_tokens; i++) {
        if (tok->tokens[i].len == 1) {
            unsigned char c = (unsigned char)tok->tokens[i].bytes[0];
            tok->ascii_map[c] = (int)i;
        }
    }
}

static void load_tokenizer(const char *checkpoint_path, Tokenizer *tok, int vocab_size) {
    tokenizer_init(tok);

    size_t n = strlen(checkpoint_path) + strlen(".tokenizer") + 1;
    char *path = (char *)malloc(n);
    if (!path) die("out of memory");
    snprintf(path, n, "%s.tokenizer", checkpoint_path);

    FILE *f = fopen(path, "rb");
    free(path);
    if (!f) die("failed to open tokenizer sidecar");

    if (fread(&tok->max_token_len, sizeof(uint32_t), 1, f) != 1 ||
        fread(&tok->bos_id, sizeof(uint32_t), 1, f) != 1 ||
        fread(&tok->eos_id, sizeof(uint32_t), 1, f) != 1) {
        fclose(f);
        die("failed tokenizer header read");
    }

    tok->n_tokens = (size_t)vocab_size;
    tok->tokens = (Token *)calloc(tok->n_tokens, sizeof(Token));
    tok->scores = (float *)calloc(tok->n_tokens, sizeof(float));
    if (!tok->tokens) {
        fclose(f);
        die("out of memory");
    }
    if (!tok->scores) {
        fclose(f);
        die("out of memory");
    }

    for (size_t i = 0; i < tok->n_tokens; i++) {
        float score;
        uint32_t len;
        if (fread(&score, sizeof(float), 1, f) != 1 || fread(&len, sizeof(uint32_t), 1, f) != 1) {
            fclose(f);
            die("failed tokenizer token header read");
        }
        (void)score;
        tok->scores[i] = score;
        tok->tokens[i].len = len;
        tok->tokens[i].bytes = (char *)malloc((size_t)len + 1);
        if (!tok->tokens[i].bytes) {
            fclose(f);
            die("out of memory");
        }
        if (len > 0 && fread(tok->tokens[i].bytes, 1, len, f) != len) {
            fclose(f);
            die("failed tokenizer token bytes read");
        }
        tok->tokens[i].bytes[len] = '\0';
    }
    fclose(f);
    build_ascii_map(tok);
}

static int load_template_text(const char *checkpoint_path, const char *suffix, char *out, size_t out_cap) {
    if (out_cap == 0) return 0;
    size_t n = strlen(checkpoint_path) + strlen(suffix) + 1;
    char *path = (char *)malloc(n);
    if (!path) die("out of memory");
    snprintf(path, n, "%s%s", checkpoint_path, suffix);
    FILE *f = fopen(path, "rb");
    free(path);
    if (!f) return 0;
    size_t got = fread(out, 1, out_cap - 1, f);
    fclose(f);
    out[got] = '\0';
    return got > 0;
}

static void render_template_user(const char *templ, const char *user, char *out, size_t out_cap) {
    if (out_cap == 0) return;
    if (!templ || templ[0] == '\0') {
        snprintf(out, out_cap, "%s", user ? user : "");
        return;
    }
    const char *p = strstr(templ, "%s");
    if (!p) {
        snprintf(out, out_cap, "%s%s", templ, user ? user : "");
        return;
    }
    size_t pre = (size_t)(p - templ);
    size_t wrote = 0;
    if (pre > 0) {
        size_t n = pre < (out_cap - 1) ? pre : (out_cap - 1);
        memcpy(out, templ, n);
        wrote = n;
    }
    if (wrote < out_cap - 1 && user) {
        size_t room = out_cap - 1 - wrote;
        size_t ulen = strlen(user);
        size_t n = ulen < room ? ulen : room;
        memcpy(out + wrote, user, n);
        wrote += n;
    }
    const char *post = p + 2;
    if (wrote < out_cap - 1 && *post) {
        size_t room = out_cap - 1 - wrote;
        size_t plen = strlen(post);
        size_t n = plen < room ? plen : room;
        memcpy(out + wrote, post, n);
        wrote += n;
    }
    out[wrote] = '\0';
}

static void runtime_close(Runtime *rt) {
    free(rt->attn_layers);
    free(rt->moe_layers);
    free(rt->moe_routed_layers);
#if !defined _WIN32
    if (rt->data != NULL && rt->file_size > 0) {
        munmap(rt->data, rt->file_size);
    }
#endif
    memset(rt, 0, sizeof(*rt));
}

static int offset_ok(size_t file_size, uint64_t off, size_t bytes) {
    if ((size_t)off > file_size) return 0;
    if (bytes > file_size - (size_t)off) return 0;
    return 1;
}

static void runtime_open(Runtime *rt, const char *checkpoint) {
    memset(rt, 0, sizeof(*rt));

    int64_t fs = file_size_of(checkpoint);
    if (fs <= 0) die("failed to stat checkpoint");
    rt->file_size = (size_t)fs;

#if !defined _WIN32
    int fd = open(checkpoint, O_RDONLY);
    if (fd < 0) die("failed to open checkpoint");
    rt->data = (unsigned char *)mmap(NULL, rt->file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (rt->data == MAP_FAILED) die("mmap failed");
#endif

    if (rt->file_size < HEADER_SIZE) die("checkpoint too small");
    memset(&rt->cfg, 0, sizeof(RuntimeConfig));
    size_t copy = sizeof(RuntimeConfig) < HEADER_SIZE ? sizeof(RuntimeConfig) : HEADER_SIZE;
    memcpy(&rt->cfg, rt->data, copy);

    if (rt->cfg.magic != MAGIC) die("invalid checkpoint magic");
    if (rt->cfg.version != 2u && rt->cfg.version != 3u && rt->cfg.version != VERSION) die("unsupported checkpoint version");
    if ((size_t)rt->cfg.total_bytes != rt->file_size) die("checkpoint size mismatch");
    if (rt->cfg.group_size == 0) die("invalid group size");
    if (rt->cfg.dim <= 0 || rt->cfg.hidden_dim <= 0 || rt->cfg.vocab_size <= 0) die("invalid dimensions in checkpoint");
    if (rt->cfg.dim % (int)rt->cfg.group_size != 0) die("dim must be divisible by group size");
    if (rt->cfg.hidden_dim % (int)rt->cfg.group_size != 0) die("hidden_dim must be divisible by group size");
    if (rt->cfg.q_lora_rank % (int)rt->cfg.group_size != 0) die("q_lora_rank must be divisible by group size");
    if (rt->cfg.kv_lora_rank % (int)rt->cfg.group_size != 0) die("kv_lora_rank must be divisible by group size");

    size_t dim = (size_t)rt->cfg.dim;
    size_t hidden = (size_t)rt->cfg.hidden_dim;
    size_t vocab = (size_t)rt->cfg.vocab_size;
    size_t gs = (size_t)rt->cfg.group_size;
    size_t dim_groups = dim / gs;
    size_t hidden_groups = hidden / gs;

    size_t norm_bytes = dim * sizeof(float);
    size_t tok_q_bytes = vocab * dim * sizeof(int8_t);
    size_t tok_s_bytes = vocab * dim_groups * sizeof(float);
    size_t out_q_bytes = vocab * dim * sizeof(int8_t);
    size_t out_s_bytes = vocab * dim_groups * sizeof(float);

    if (!offset_ok(rt->file_size, rt->cfg.off_norm, norm_bytes) ||
        !offset_ok(rt->file_size, rt->cfg.off_tok_q, tok_q_bytes) ||
        !offset_ok(rt->file_size, rt->cfg.off_tok_s, tok_s_bytes) ||
        !offset_ok(rt->file_size, rt->cfg.off_out_q, out_q_bytes) ||
        !offset_ok(rt->file_size, rt->cfg.off_out_s, out_s_bytes)) {
        die("checkpoint base tensor offsets out of bounds");
    }

    rt->output_norm = (const float *)(rt->data + rt->cfg.off_norm);
    rt->tok_q = (const int8_t *)(rt->data + rt->cfg.off_tok_q);
    rt->tok_s = (const float *)(rt->data + rt->cfg.off_tok_s);
    rt->out_q = (const int8_t *)(rt->data + rt->cfg.off_out_q);
    rt->out_s = (const float *)(rt->data + rt->cfg.off_out_s);

    // Optional extension block for version 3+: layer-0 dense FFN weights.
    rt->has_l0_ffn = 0;
    if (rt->cfg.version >= 3u && rt->cfg.off_l0_ffn_norm != 0) {
        size_t l0_norm_bytes = dim * sizeof(float);
        size_t l0_gate_q_bytes = hidden * dim * sizeof(int8_t);
        size_t l0_gate_s_bytes = hidden * dim_groups * sizeof(float);
        size_t l0_up_q_bytes = hidden * dim * sizeof(int8_t);
        size_t l0_up_s_bytes = hidden * dim_groups * sizeof(float);
        size_t l0_down_q_bytes = dim * hidden * sizeof(int8_t);
        size_t l0_down_s_bytes = dim * hidden_groups * sizeof(float);

        if (!offset_ok(rt->file_size, rt->cfg.off_l0_ffn_norm, l0_norm_bytes) ||
            !offset_ok(rt->file_size, rt->cfg.off_l0_ffn_gate_q, l0_gate_q_bytes) ||
            !offset_ok(rt->file_size, rt->cfg.off_l0_ffn_gate_s, l0_gate_s_bytes) ||
            !offset_ok(rt->file_size, rt->cfg.off_l0_ffn_up_q, l0_up_q_bytes) ||
            !offset_ok(rt->file_size, rt->cfg.off_l0_ffn_up_s, l0_up_s_bytes) ||
            !offset_ok(rt->file_size, rt->cfg.off_l0_ffn_down_q, l0_down_q_bytes) ||
            !offset_ok(rt->file_size, rt->cfg.off_l0_ffn_down_s, l0_down_s_bytes)) {
            die("checkpoint layer-0 FFN offsets out of bounds");
        }

        rt->l0_ffn_norm = (const float *)(rt->data + rt->cfg.off_l0_ffn_norm);
        rt->l0_ffn_gate_q = (const int8_t *)(rt->data + rt->cfg.off_l0_ffn_gate_q);
        rt->l0_ffn_gate_s = (const float *)(rt->data + rt->cfg.off_l0_ffn_gate_s);
        rt->l0_ffn_up_q = (const int8_t *)(rt->data + rt->cfg.off_l0_ffn_up_q);
        rt->l0_ffn_up_s = (const float *)(rt->data + rt->cfg.off_l0_ffn_up_s);
        rt->l0_ffn_down_q = (const int8_t *)(rt->data + rt->cfg.off_l0_ffn_down_q);
        rt->l0_ffn_down_s = (const float *)(rt->data + rt->cfg.off_l0_ffn_down_s);
        rt->has_l0_ffn = 1;
    }

    // Optional extension block for version 4+: layer-0 MLA attention.
    rt->has_l0_attn = 0;
    rt->has_all_attn = 0;
    size_t off_after_l0_attn = (size_t)rt->cfg.off_l0_ffn_down_s + dim * hidden_groups * sizeof(float);
    if (rt->cfg.version >= 4u && (rt->cfg.l0_flags & FLAG_ALL_ATTN) != 0) {
        int qk_nope = rt->cfg.l0_qk_nope_dim;
        int head_k = rt->cfg.l0_head_k_dim;
        int v_head = rt->cfg.l0_v_head_dim;
        if (qk_nope <= 0 || head_k <= 0 || v_head <= 0) die("invalid v4 mla dims");
        if (head_k != qk_nope + rt->cfg.rope_dim) die("invalid v4 mla shape relation");
        if (qk_nope % (int)rt->cfg.group_size != 0) die("qk_nope dim must be divisible by group size");
        if ((rt->cfg.n_heads * v_head) % (int)rt->cfg.group_size != 0) die("attn out width must be divisible by group size");

        size_t q_lora = (size_t)rt->cfg.q_lora_rank;
        size_t kv_lora = (size_t)rt->cfg.kv_lora_rank;
        size_t rope = (size_t)rt->cfg.rope_dim;
        size_t n_heads = (size_t)rt->cfg.n_heads;
        size_t qk_nope_sz = (size_t)qk_nope;
        size_t head_k_sz = (size_t)head_k;
        size_t v_head_sz = (size_t)v_head;
        size_t q_lora_groups = q_lora / gs;
        size_t kv_lora_groups = kv_lora / gs;
        size_t qk_nope_groups = qk_nope_sz / gs;
        size_t attn_out_cols = n_heads * v_head_sz;
        size_t attn_out_groups = attn_out_cols / gs;

        size_t bytes_l0_attn_norm = dim * sizeof(float);
        size_t bytes_l0_q_a_norm = q_lora * sizeof(float);
        size_t bytes_l0_kv_a_norm = kv_lora * sizeof(float);
        size_t bytes_l0_q_a_q = q_lora * dim * sizeof(int8_t);
        size_t bytes_l0_q_a_s = q_lora * dim_groups * sizeof(float);
        size_t bytes_l0_q_b_q = n_heads * head_k_sz * q_lora * sizeof(int8_t);
        size_t bytes_l0_q_b_s = n_heads * head_k_sz * q_lora_groups * sizeof(float);
        size_t bytes_l0_kv_a_q = (kv_lora + rope) * dim * sizeof(int8_t);
        size_t bytes_l0_kv_a_s = (kv_lora + rope) * dim_groups * sizeof(float);
        size_t bytes_l0_k_b_q = n_heads * kv_lora * qk_nope_sz * sizeof(int8_t);
        size_t bytes_l0_k_b_s = n_heads * kv_lora * qk_nope_groups * sizeof(float);
        size_t bytes_l0_v_b_q = n_heads * v_head_sz * kv_lora * sizeof(int8_t);
        size_t bytes_l0_v_b_s = n_heads * v_head_sz * kv_lora_groups * sizeof(float);
        size_t bytes_l0_attn_out_q = dim * attn_out_cols * sizeof(int8_t);
        size_t bytes_l0_attn_out_s = dim * attn_out_groups * sizeof(float);

        size_t off = (size_t)rt->cfg.off_l0_ffn_down_s + dim * hidden_groups * sizeof(float);
        if (!offset_ok(rt->file_size, off, bytes_l0_attn_norm)) die("l0 attn_norm out of bounds");
        rt->l0_attn_norm = (const float *)(rt->data + off);
        off += bytes_l0_attn_norm;
        if (!offset_ok(rt->file_size, off, bytes_l0_q_a_norm)) die("l0 q_a_norm out of bounds");
        rt->l0_q_a_norm = (const float *)(rt->data + off);
        off += bytes_l0_q_a_norm;
        if (!offset_ok(rt->file_size, off, bytes_l0_kv_a_norm)) die("l0 kv_a_norm out of bounds");
        rt->l0_kv_a_norm = (const float *)(rt->data + off);
        off += bytes_l0_kv_a_norm;
        if (!offset_ok(rt->file_size, off, bytes_l0_q_a_q)) die("l0 q_a_q out of bounds");
        rt->l0_q_a_q = (const int8_t *)(rt->data + off);
        off += bytes_l0_q_a_q;
        if (!offset_ok(rt->file_size, off, bytes_l0_q_a_s)) die("l0 q_a_s out of bounds");
        rt->l0_q_a_s = (const float *)(rt->data + off);
        off += bytes_l0_q_a_s;
        if (!offset_ok(rt->file_size, off, bytes_l0_q_b_q)) die("l0 q_b_q out of bounds");
        rt->l0_q_b_q = (const int8_t *)(rt->data + off);
        off += bytes_l0_q_b_q;
        if (!offset_ok(rt->file_size, off, bytes_l0_q_b_s)) die("l0 q_b_s out of bounds");
        rt->l0_q_b_s = (const float *)(rt->data + off);
        off += bytes_l0_q_b_s;
        if (!offset_ok(rt->file_size, off, bytes_l0_kv_a_q)) die("l0 kv_a_q out of bounds");
        rt->l0_kv_a_q = (const int8_t *)(rt->data + off);
        off += bytes_l0_kv_a_q;
        if (!offset_ok(rt->file_size, off, bytes_l0_kv_a_s)) die("l0 kv_a_s out of bounds");
        rt->l0_kv_a_s = (const float *)(rt->data + off);
        off += bytes_l0_kv_a_s;
        if (!offset_ok(rt->file_size, off, bytes_l0_k_b_q)) die("l0 k_b_q out of bounds");
        rt->l0_k_b_q = (const int8_t *)(rt->data + off);
        off += bytes_l0_k_b_q;
        if (!offset_ok(rt->file_size, off, bytes_l0_k_b_s)) die("l0 k_b_s out of bounds");
        rt->l0_k_b_s = (const float *)(rt->data + off);
        off += bytes_l0_k_b_s;
        if (!offset_ok(rt->file_size, off, bytes_l0_v_b_q)) die("l0 v_b_q out of bounds");
        rt->l0_v_b_q = (const int8_t *)(rt->data + off);
        off += bytes_l0_v_b_q;
        if (!offset_ok(rt->file_size, off, bytes_l0_v_b_s)) die("l0 v_b_s out of bounds");
        rt->l0_v_b_s = (const float *)(rt->data + off);
        off += bytes_l0_v_b_s;
        if (!offset_ok(rt->file_size, off, bytes_l0_attn_out_q)) die("l0 attn_out_q out of bounds");
        rt->l0_attn_out_q = (const int8_t *)(rt->data + off);
        off += bytes_l0_attn_out_q;
        if (!offset_ok(rt->file_size, off, bytes_l0_attn_out_s)) die("l0 attn_out_s out of bounds");
        rt->l0_attn_out_s = (const float *)(rt->data + off);
        off += bytes_l0_attn_out_s;
        if (off > rt->file_size) die("l0 mla offsets overflow");

        rt->has_l0_attn = 1;
        off_after_l0_attn = off;
    }

    // Optional appended block: per-layer attention tensors for all layers.
    size_t off_after_all_attn = off_after_l0_attn;
    if (rt->has_l0_attn) {
        int qk_nope = rt->cfg.l0_qk_nope_dim;
        int head_k = rt->cfg.l0_head_k_dim;
        int v_head = rt->cfg.l0_v_head_dim;
        size_t q_lora = (size_t)rt->cfg.q_lora_rank;
        size_t kv_lora = (size_t)rt->cfg.kv_lora_rank;
        size_t rope = (size_t)rt->cfg.rope_dim;
        size_t n_heads = (size_t)rt->cfg.n_heads;
        size_t qk_nope_sz = (size_t)qk_nope;
        size_t head_k_sz = (size_t)head_k;
        size_t v_head_sz = (size_t)v_head;
        size_t q_lora_groups = q_lora / gs;
        size_t kv_lora_groups = kv_lora / gs;
        size_t qk_nope_groups = qk_nope_sz / gs;
        size_t attn_out_cols = n_heads * v_head_sz;
        size_t attn_out_groups = attn_out_cols / gs;

        size_t bytes_attn_norm = dim * sizeof(float);
        size_t bytes_q_a_norm = q_lora * sizeof(float);
        size_t bytes_kv_a_norm = kv_lora * sizeof(float);
        size_t bytes_q_a_q = q_lora * dim * sizeof(int8_t);
        size_t bytes_q_a_s = q_lora * dim_groups * sizeof(float);
        size_t bytes_q_b_q = n_heads * head_k_sz * q_lora * sizeof(int8_t);
        size_t bytes_q_b_s = n_heads * head_k_sz * q_lora_groups * sizeof(float);
        size_t bytes_kv_a_q = (kv_lora + rope) * dim * sizeof(int8_t);
        size_t bytes_kv_a_s = (kv_lora + rope) * dim_groups * sizeof(float);
        size_t bytes_k_b_q = n_heads * kv_lora * qk_nope_sz * sizeof(int8_t);
        size_t bytes_k_b_s = n_heads * kv_lora * qk_nope_groups * sizeof(float);
        size_t bytes_v_b_q = n_heads * v_head_sz * kv_lora * sizeof(int8_t);
        size_t bytes_v_b_s = n_heads * v_head_sz * kv_lora_groups * sizeof(float);
        size_t bytes_attn_out_q = dim * attn_out_cols * sizeof(int8_t);
        size_t bytes_attn_out_s = dim * attn_out_groups * sizeof(float);
        size_t bytes_per_layer =
            bytes_attn_norm + bytes_q_a_norm + bytes_kv_a_norm + bytes_q_a_q + bytes_q_a_s + bytes_q_b_q + bytes_q_b_s +
            bytes_kv_a_q + bytes_kv_a_s + bytes_k_b_q + bytes_k_b_s + bytes_v_b_q + bytes_v_b_s + bytes_attn_out_q + bytes_attn_out_s;
        size_t n_layers = (size_t)rt->cfg.n_layers;
        size_t bytes_all_layers = bytes_per_layer * n_layers;

        if (off_after_l0_attn < rt->file_size) {
            if (bytes_all_layers > rt->file_size - off_after_l0_attn) {
                die("incomplete all-layer attention block");
            }
            rt->attn_layers = (LayerAttnWeights *)calloc(n_layers, sizeof(LayerAttnWeights));
            if (!rt->attn_layers) die("out of memory");

            size_t off = off_after_l0_attn;
            for (size_t li = 0; li < n_layers; li++) {
                LayerAttnWeights *w = &rt->attn_layers[li];
                if (!offset_ok(rt->file_size, off, bytes_attn_norm)) die("all-attn attn_norm out of bounds");
                w->attn_norm = (const float *)(rt->data + off);
                off += bytes_attn_norm;
                if (!offset_ok(rt->file_size, off, bytes_q_a_norm)) die("all-attn q_a_norm out of bounds");
                w->q_a_norm = (const float *)(rt->data + off);
                off += bytes_q_a_norm;
                if (!offset_ok(rt->file_size, off, bytes_kv_a_norm)) die("all-attn kv_a_norm out of bounds");
                w->kv_a_norm = (const float *)(rt->data + off);
                off += bytes_kv_a_norm;
                if (!offset_ok(rt->file_size, off, bytes_q_a_q)) die("all-attn q_a_q out of bounds");
                w->q_a_q = (const int8_t *)(rt->data + off);
                off += bytes_q_a_q;
                if (!offset_ok(rt->file_size, off, bytes_q_a_s)) die("all-attn q_a_s out of bounds");
                w->q_a_s = (const float *)(rt->data + off);
                off += bytes_q_a_s;
                if (!offset_ok(rt->file_size, off, bytes_q_b_q)) die("all-attn q_b_q out of bounds");
                w->q_b_q = (const int8_t *)(rt->data + off);
                off += bytes_q_b_q;
                if (!offset_ok(rt->file_size, off, bytes_q_b_s)) die("all-attn q_b_s out of bounds");
                w->q_b_s = (const float *)(rt->data + off);
                off += bytes_q_b_s;
                if (!offset_ok(rt->file_size, off, bytes_kv_a_q)) die("all-attn kv_a_q out of bounds");
                w->kv_a_q = (const int8_t *)(rt->data + off);
                off += bytes_kv_a_q;
                if (!offset_ok(rt->file_size, off, bytes_kv_a_s)) die("all-attn kv_a_s out of bounds");
                w->kv_a_s = (const float *)(rt->data + off);
                off += bytes_kv_a_s;
                if (!offset_ok(rt->file_size, off, bytes_k_b_q)) die("all-attn k_b_q out of bounds");
                w->k_b_q = (const int8_t *)(rt->data + off);
                off += bytes_k_b_q;
                if (!offset_ok(rt->file_size, off, bytes_k_b_s)) die("all-attn k_b_s out of bounds");
                w->k_b_s = (const float *)(rt->data + off);
                off += bytes_k_b_s;
                if (!offset_ok(rt->file_size, off, bytes_v_b_q)) die("all-attn v_b_q out of bounds");
                w->v_b_q = (const int8_t *)(rt->data + off);
                off += bytes_v_b_q;
                if (!offset_ok(rt->file_size, off, bytes_v_b_s)) die("all-attn v_b_s out of bounds");
                w->v_b_s = (const float *)(rt->data + off);
                off += bytes_v_b_s;
                if (!offset_ok(rt->file_size, off, bytes_attn_out_q)) die("all-attn attn_out_q out of bounds");
                w->attn_out_q = (const int8_t *)(rt->data + off);
                off += bytes_attn_out_q;
                if (!offset_ok(rt->file_size, off, bytes_attn_out_s)) die("all-attn attn_out_s out of bounds");
                w->attn_out_s = (const float *)(rt->data + off);
                off += bytes_attn_out_s;
            }
            rt->has_all_attn = 1;
            off_after_all_attn = off;
        }
    }

    // Optional appended block: shared-expert FFN tensors for MoE layers 1..n_layers-1.
    rt->has_moe_shared = 0;
    rt->has_moe_routed = 0;
    size_t off_after_moe_shared = off_after_all_attn;
    if (rt->cfg.moe_ffn_dim > 0 && rt->cfg.n_layers > 1 && off_after_all_attn < rt->file_size) {
        size_t moe_dim = (size_t)rt->cfg.moe_ffn_dim;
        if (moe_dim % gs != 0) die("moe_ffn_dim must be divisible by group size");
        size_t moe_groups = moe_dim / gs;
        size_t n_layers = (size_t)rt->cfg.n_layers;
        size_t n_routed = (size_t)rt->cfg.n_routed_experts;
        size_t dim_groups_local = dim / gs;

        size_t bytes_ffn_norm = dim * sizeof(float);
        size_t bytes_gate_inp = n_routed * dim * sizeof(float);
        size_t bytes_exp_probs_b = n_routed * sizeof(float);
        size_t bytes_gate_sh_q = moe_dim * dim * sizeof(int8_t);
        size_t bytes_gate_sh_s = moe_dim * dim_groups_local * sizeof(float);
        size_t bytes_up_sh_q = moe_dim * dim * sizeof(int8_t);
        size_t bytes_up_sh_s = moe_dim * dim_groups_local * sizeof(float);
        size_t bytes_down_sh_q = dim * moe_dim * sizeof(int8_t);
        size_t bytes_down_sh_s = dim * moe_groups * sizeof(float);
        size_t bytes_per_layer_shared =
            bytes_ffn_norm + bytes_gate_inp + bytes_exp_probs_b + bytes_gate_sh_q + bytes_gate_sh_s +
            bytes_up_sh_q + bytes_up_sh_s + bytes_down_sh_q + bytes_down_sh_s;
        size_t n_moe_layers = n_layers - 1;
        size_t bytes_total_shared = bytes_per_layer_shared * n_moe_layers;

        if (bytes_total_shared > rt->file_size - off_after_all_attn) die("incomplete moe shared block");
        rt->moe_layers = (LayerMoeShared *)calloc(n_layers, sizeof(LayerMoeShared));
        if (!rt->moe_layers) die("out of memory");

        size_t off = off_after_all_attn;
        for (size_t li = 1; li < n_layers; li++) {
            LayerMoeShared *w = &rt->moe_layers[li];
            if (!offset_ok(rt->file_size, off, bytes_ffn_norm)) die("moe ffn_norm out of bounds");
            w->ffn_norm = (const float *)(rt->data + off);
            off += bytes_ffn_norm;
            if (!offset_ok(rt->file_size, off, bytes_gate_inp)) die("moe gate_inp out of bounds");
            w->gate_inp = (const float *)(rt->data + off);
            off += bytes_gate_inp;
            if (!offset_ok(rt->file_size, off, bytes_exp_probs_b)) die("moe exp_probs_b out of bounds");
            w->exp_probs_b = (const float *)(rt->data + off);
            off += bytes_exp_probs_b;
            if (!offset_ok(rt->file_size, off, bytes_gate_sh_q)) die("moe gate_sh_q out of bounds");
            w->gate_sh_q = (const int8_t *)(rt->data + off);
            off += bytes_gate_sh_q;
            if (!offset_ok(rt->file_size, off, bytes_gate_sh_s)) die("moe gate_sh_s out of bounds");
            w->gate_sh_s = (const float *)(rt->data + off);
            off += bytes_gate_sh_s;
            if (!offset_ok(rt->file_size, off, bytes_up_sh_q)) die("moe up_sh_q out of bounds");
            w->up_sh_q = (const int8_t *)(rt->data + off);
            off += bytes_up_sh_q;
            if (!offset_ok(rt->file_size, off, bytes_up_sh_s)) die("moe up_sh_s out of bounds");
            w->up_sh_s = (const float *)(rt->data + off);
            off += bytes_up_sh_s;
            if (!offset_ok(rt->file_size, off, bytes_down_sh_q)) die("moe down_sh_q out of bounds");
            w->down_sh_q = (const int8_t *)(rt->data + off);
            off += bytes_down_sh_q;
            if (!offset_ok(rt->file_size, off, bytes_down_sh_s)) die("moe down_sh_s out of bounds");
            w->down_sh_s = (const float *)(rt->data + off);
            off += bytes_down_sh_s;
        }
        rt->has_moe_shared = 1;
        off_after_moe_shared = off;

        // Optional routed experts block for MoE layers 1..n_layers-1.
        if (off_after_moe_shared < rt->file_size) {
            size_t bytes_gate_q = n_routed * moe_dim * dim * sizeof(int8_t);
            size_t bytes_gate_s = n_routed * moe_dim * dim_groups_local * sizeof(float);
            size_t bytes_up_q = n_routed * moe_dim * dim * sizeof(int8_t);
            size_t bytes_up_s = n_routed * moe_dim * dim_groups_local * sizeof(float);
            size_t bytes_down_q = n_routed * dim * moe_dim * sizeof(int8_t);
            size_t bytes_down_s = n_routed * dim * moe_groups * sizeof(float);
            size_t bytes_per_layer_routed = bytes_gate_q + bytes_gate_s + bytes_up_q + bytes_up_s + bytes_down_q + bytes_down_s;
            size_t bytes_total_routed = bytes_per_layer_routed * n_moe_layers;
            if (bytes_total_routed > rt->file_size - off_after_moe_shared) die("incomplete moe routed block");

            rt->moe_routed_layers = (LayerMoeRouted *)calloc(n_layers, sizeof(LayerMoeRouted));
            if (!rt->moe_routed_layers) die("out of memory");
            off = off_after_moe_shared;
            for (size_t li = 1; li < n_layers; li++) {
                LayerMoeRouted *w = &rt->moe_routed_layers[li];
                if (!offset_ok(rt->file_size, off, bytes_gate_q)) die("moe routed gate_q out of bounds");
                w->gate_q = (const int8_t *)(rt->data + off);
                off += bytes_gate_q;
                if (!offset_ok(rt->file_size, off, bytes_gate_s)) die("moe routed gate_s out of bounds");
                w->gate_s = (const float *)(rt->data + off);
                off += bytes_gate_s;
                if (!offset_ok(rt->file_size, off, bytes_up_q)) die("moe routed up_q out of bounds");
                w->up_q = (const int8_t *)(rt->data + off);
                off += bytes_up_q;
                if (!offset_ok(rt->file_size, off, bytes_up_s)) die("moe routed up_s out of bounds");
                w->up_s = (const float *)(rt->data + off);
                off += bytes_up_s;
                if (!offset_ok(rt->file_size, off, bytes_down_q)) die("moe routed down_q out of bounds");
                w->down_q = (const int8_t *)(rt->data + off);
                off += bytes_down_q;
                if (!offset_ok(rt->file_size, off, bytes_down_s)) die("moe routed down_s out of bounds");
                w->down_s = (const float *)(rt->data + off);
                off += bytes_down_s;
            }
            rt->has_moe_routed = 1;
        }
    }
}

static void rmsnorm(float *o, const float *x, const float *w, int n) {
    double ss = 0.0;
    for (int i = 0; i < n; i++) ss += (double)x[i] * (double)x[i];
    float mean = (float)(ss / (double)n);
    float inv = 1.0f / sqrtf(mean + 1e-5f);
    for (int i = 0; i < n; i++) o[i] = x[i] * inv * w[i];
}

static void dequant_row_q80(const int8_t *qrow, const float *srow, float *out, int dim, int gs) {
    int n_groups = dim / gs;
    for (int g = 0; g < n_groups; g++) {
        float s = srow[g];
        int base = g * gs;
        for (int k = 0; k < gs; k++) {
            out[base + k] = (float)qrow[base + k] * s;
        }
    }
}

static void matvec_q80_rows(const int8_t *qmat, const float *smat, const float *x, float *y, int rows, int dim, int gs) {
    int n_groups = dim / gs;
#if defined(_OPENMP)
#pragma omp parallel for schedule(static) if (rows >= 64)
#endif
    for (int r = 0; r < rows; r++) {
        const int8_t *qrow = qmat + (size_t)r * dim;
        const float *srow = smat + (size_t)r * n_groups;
        float acc = 0.0f;
        for (int g = 0; g < n_groups; g++) {
            float s = srow[g];
            int base = g * gs;
            float dot = 0.0f;
#if defined(_OPENMP)
#pragma omp simd reduction(+:dot)
#endif
            for (int k = 0; k < gs; k++) dot += (float)qrow[base + k] * x[base + k];
            acc += s * dot;
        }
        y[r] = acc;
    }
}

static void dequant_row(const int8_t *qrow, const float *srow, float *out, int dim, int gs) {
    dequant_row_q80(qrow, srow, out, dim, gs);
}

static void matvec_rows(const int8_t *qmat, const float *smat, const float *x, float *y, int rows, int dim, int gs) {
    if (glm_backend_try_metal_matvec(qmat, smat, x, y, rows, dim, gs) == 0) {
        return;
    }
    matvec_q80_rows(qmat, smat, x, y, rows, dim, gs);
}

static void runstate_init(RunState *s) {
    memset(s, 0, sizeof(*s));
}

static void runstate_free(RunState *s) {
    free(s->x);
    free(s->xn);
    free(s->xb);
    free(s->ff_gate);
    free(s->ff_up);
    free(s->logits);
    free(s->k_cache_comp);
    free(s->k_cache_pe);
    free(s->q_a);
    free(s->q_full);
    free(s->kv_a);
    free(s->q_nope);
    free(s->q_pe);
    free(s->q_abs);
    free(s->att_scores);
    free(s->att_ctx);
    free(s->att_concat);
    free(s->moe_gate);
    free(s->moe_up);
    free(s->moe_router);
    free(s->moe_out_acc);
    free(s->moe_topk_idx);
    free(s->moe_topk_w);
    runstate_init(s);
}

static void runstate_build(RunState *s, const Runtime *rt, int cache_cap) {
    runstate_init(s);
    const RuntimeConfig *cfg = &rt->cfg;
    s->x = (float *)calloc((size_t)cfg->dim, sizeof(float));
    s->xn = (float *)calloc((size_t)cfg->dim, sizeof(float));
    s->xb = (float *)calloc((size_t)cfg->dim, sizeof(float));
    s->ff_gate = (float *)calloc((size_t)cfg->hidden_dim, sizeof(float));
    s->ff_up = (float *)calloc((size_t)cfg->hidden_dim, sizeof(float));
    s->logits = (float *)calloc((size_t)cfg->vocab_size, sizeof(float));
    s->cache_cap = cache_cap;
    s->n_attn_layers = rt->has_all_attn ? cfg->n_layers : (rt->has_l0_attn ? 1 : 0);

    if (s->n_attn_layers > 0) {
        s->k_cache_comp = (float *)calloc((size_t)s->n_attn_layers * (size_t)cache_cap * (size_t)cfg->kv_lora_rank, sizeof(float));
        s->k_cache_pe = (float *)calloc((size_t)s->n_attn_layers * (size_t)cache_cap * (size_t)cfg->rope_dim, sizeof(float));
        s->q_a = (float *)calloc((size_t)cfg->q_lora_rank, sizeof(float));
        s->q_full = (float *)calloc((size_t)cfg->n_heads * (size_t)cfg->l0_head_k_dim, sizeof(float));
        s->kv_a = (float *)calloc((size_t)(cfg->kv_lora_rank + cfg->rope_dim), sizeof(float));
        s->q_nope = (float *)calloc((size_t)cfg->l0_qk_nope_dim, sizeof(float));
        s->q_pe = (float *)calloc((size_t)cfg->rope_dim, sizeof(float));
        s->q_abs = (float *)calloc((size_t)cfg->kv_lora_rank, sizeof(float));
        s->att_scores = (float *)calloc((size_t)cache_cap, sizeof(float));
        s->att_ctx = (float *)calloc((size_t)cfg->kv_lora_rank, sizeof(float));
        s->att_concat = (float *)calloc((size_t)cfg->n_heads * (size_t)cfg->l0_v_head_dim, sizeof(float));
    }
    if (rt->has_moe_shared && cfg->moe_ffn_dim > 0) {
        s->moe_gate = (float *)calloc((size_t)cfg->moe_ffn_dim, sizeof(float));
        s->moe_up = (float *)calloc((size_t)cfg->moe_ffn_dim, sizeof(float));
        s->moe_router = (float *)calloc((size_t)cfg->n_routed_experts, sizeof(float));
        s->moe_out_acc = (float *)calloc((size_t)cfg->dim, sizeof(float));
        s->moe_topk_idx = (int *)calloc((size_t)cfg->n_experts_used, sizeof(int));
        s->moe_topk_w = (float *)calloc((size_t)cfg->n_experts_used, sizeof(float));
    }

    if (!s->x || !s->xn || !s->xb || !s->ff_gate || !s->ff_up || !s->logits ||
        (s->n_attn_layers > 0 && (!s->k_cache_comp || !s->k_cache_pe || !s->q_a || !s->q_full || !s->kv_a ||
                             !s->q_nope || !s->q_pe || !s->q_abs || !s->att_scores || !s->att_ctx || !s->att_concat)) ||
        (rt->has_moe_shared && cfg->moe_ffn_dim > 0 &&
         (!s->moe_gate || !s->moe_up || !s->moe_router || !s->moe_out_acc || !s->moe_topk_idx || !s->moe_topk_w))) {
        die("out of memory");
    }
}

static void runstate_copy_from(const Runtime *rt, const RunState *src, RunState *dst) {
    if (!rt || !src || !dst) return;
    const RuntimeConfig *cfg = &rt->cfg;

    memcpy(dst->x, src->x, (size_t)cfg->dim * sizeof(float));
    memcpy(dst->xn, src->xn, (size_t)cfg->dim * sizeof(float));
    memcpy(dst->xb, src->xb, (size_t)cfg->dim * sizeof(float));
    memcpy(dst->ff_gate, src->ff_gate, (size_t)cfg->hidden_dim * sizeof(float));
    memcpy(dst->ff_up, src->ff_up, (size_t)cfg->hidden_dim * sizeof(float));
    memcpy(dst->logits, src->logits, (size_t)cfg->vocab_size * sizeof(float));

    if (src->n_attn_layers > 0) {
        memcpy(dst->k_cache_comp,
               src->k_cache_comp,
               (size_t)src->n_attn_layers * (size_t)src->cache_cap * (size_t)cfg->kv_lora_rank * sizeof(float));
        memcpy(dst->k_cache_pe,
               src->k_cache_pe,
               (size_t)src->n_attn_layers * (size_t)src->cache_cap * (size_t)cfg->rope_dim * sizeof(float));
        memcpy(dst->q_a, src->q_a, (size_t)cfg->q_lora_rank * sizeof(float));
        memcpy(dst->q_full, src->q_full, (size_t)cfg->n_heads * (size_t)cfg->l0_head_k_dim * sizeof(float));
        memcpy(dst->kv_a, src->kv_a, (size_t)(cfg->kv_lora_rank + cfg->rope_dim) * sizeof(float));
        memcpy(dst->q_nope, src->q_nope, (size_t)cfg->l0_qk_nope_dim * sizeof(float));
        memcpy(dst->q_pe, src->q_pe, (size_t)cfg->rope_dim * sizeof(float));
        memcpy(dst->q_abs, src->q_abs, (size_t)cfg->kv_lora_rank * sizeof(float));
        memcpy(dst->att_scores, src->att_scores, (size_t)src->cache_cap * sizeof(float));
        memcpy(dst->att_ctx, src->att_ctx, (size_t)cfg->kv_lora_rank * sizeof(float));
        memcpy(dst->att_concat, src->att_concat, (size_t)cfg->n_heads * (size_t)cfg->l0_v_head_dim * sizeof(float));
    }

    if (rt->has_moe_shared && cfg->moe_ffn_dim > 0) {
        memcpy(dst->moe_gate, src->moe_gate, (size_t)cfg->moe_ffn_dim * sizeof(float));
        memcpy(dst->moe_up, src->moe_up, (size_t)cfg->moe_ffn_dim * sizeof(float));
        memcpy(dst->moe_router, src->moe_router, (size_t)cfg->n_routed_experts * sizeof(float));
        memcpy(dst->moe_out_acc, src->moe_out_acc, (size_t)cfg->dim * sizeof(float));
        memcpy(dst->moe_topk_idx, src->moe_topk_idx, (size_t)cfg->n_experts_used * sizeof(int));
        memcpy(dst->moe_topk_w, src->moe_topk_w, (size_t)cfg->n_experts_used * sizeof(float));
    }
}

static float silu(float x) {
    return x / (1.0f + expf(-x));
}

static float dot_f32(const float *a, const float *b, int n) {
#if GLM_HAS_CBLAS
    if (n >= 128) {
        return cblas_sdot(n, a, 1, b, 1);
    }
#endif
    float s = 0.0f;
    for (int i = 0; i < n; i++) s += a[i] * b[i];
    return s;
}

static void matvec_f32_rows(const float *a, const float *x, float *y, int rows, int cols) {
#if GLM_HAS_CBLAS
    if (rows > 0 && cols > 0) {
        cblas_sgemv(CblasRowMajor, CblasNoTrans, rows, cols, 1.0f, a, cols, x, 1, 0.0f, y, 1);
        return;
    }
#endif
    for (int r = 0; r < rows; r++) {
        const float *row = a + (size_t)r * (size_t)cols;
        y[r] = dot_f32(row, x, cols);
    }
}

static float sum_f32(const float *x, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += x[i];
    return sum;
}

static void debug_print_tensor_sum(const char *name, const float *x, int n) {
    if (!name || !x || n <= 0) return;
    fprintf(stderr, "[glm-layer-sum] %s sum=%.6f\n", name, sum_f32(x, n));
}

static void softmax_inplace(float *x, int n) {
    float maxv = x[0];
    for (int i = 1; i < n; i++) if (x[i] > maxv) maxv = x[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        x[i] = expf(x[i] - maxv);
        sum += x[i];
    }
    if (sum <= 0.0f) return;
    float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) x[i] *= inv;
}

static void apply_rope_inplace(float *x, int dim, int pos) {
    const float base = 1000000.0f;
    for (int i = 0; i + 1 < dim; i += 2) {
        float freq = powf(base, -((float)i / (float)dim));
        float ang = (float)pos * freq;
        float c = cosf(ang);
        float s = sinf(ang);
        float x0 = x[i + 0];
        float x1 = x[i + 1];
        x[i + 0] = x0 * c - x1 * s;
        x[i + 1] = x0 * s + x1 * c;
    }
}

static void apply_mla_attention_layer(const Runtime *rt, const LayerAttnWeights *w, RunState *s, int gs, int pos, int layer_idx, int debug_mode) {
    if (!w) return;
    if (pos < 0 || pos >= s->cache_cap) die("position exceeds attention cache capacity");
    if (layer_idx < 0 || layer_idx >= s->n_attn_layers) die("invalid attention layer index");

    const int dim = rt->cfg.dim;
    const int n_heads = rt->cfg.n_heads;
    const int q_lora = rt->cfg.q_lora_rank;
    const int kv_lora = rt->cfg.kv_lora_rank;
    const int rope_dim = rt->cfg.rope_dim;
    const int qk_nope = rt->cfg.l0_qk_nope_dim;
    const int head_k = rt->cfg.l0_head_k_dim;
    const int v_head = rt->cfg.l0_v_head_dim;
    const float kq_scale = 1.0f / sqrtf((float)head_k);

    rmsnorm(s->xn, s->x, w->attn_norm, dim);
    assert_finite_slice("attn_xn", s->xn, dim, debug_mode);

    matvec_rows(w->q_a_q, w->q_a_s, s->xn, s->q_a, q_lora, dim, gs);
    rmsnorm(s->q_a, s->q_a, w->q_a_norm, q_lora);
    assert_finite_slice("attn_q_a", s->q_a, q_lora, debug_mode);
    matvec_rows(w->q_b_q, w->q_b_s, s->q_a, s->q_full, n_heads * head_k, q_lora, gs);
    assert_finite_slice("attn_q_full", s->q_full, n_heads * head_k, debug_mode);

    matvec_rows(w->kv_a_q, w->kv_a_s, s->xn, s->kv_a, kv_lora + rope_dim, dim, gs);
    rmsnorm(s->kv_a, s->kv_a, w->kv_a_norm, kv_lora);
    assert_finite_slice("attn_kv_a", s->kv_a, kv_lora, debug_mode);

    size_t layer_base = (size_t)layer_idx * (size_t)s->cache_cap;
    size_t cache_slots = (size_t)s->n_attn_layers * (size_t)s->cache_cap;
    size_t write_slot = layer_base + (size_t)pos;
    if (write_slot >= cache_slots) die("attention cache write slot out of bounds");
    float *cache_comp = s->k_cache_comp + (layer_base + (size_t)pos) * (size_t)kv_lora;
    float *cache_pe = s->k_cache_pe + (layer_base + (size_t)pos) * (size_t)rope_dim;
    memcpy(cache_comp, s->kv_a, (size_t)kv_lora * sizeof(float));
    memcpy(cache_pe, s->kv_a + kv_lora, (size_t)rope_dim * sizeof(float));
    apply_rope_inplace(cache_pe, rope_dim, pos);
    assert_finite_slice("attn_cache_pe", cache_pe, rope_dim, debug_mode);

    for (int h = 0; h < n_heads; h++) {
        const float *qh = s->q_full + (size_t)h * (size_t)head_k;
        memcpy(s->q_nope, qh, (size_t)qk_nope * sizeof(float));
        memcpy(s->q_pe, qh + qk_nope, (size_t)rope_dim * sizeof(float));
        apply_rope_inplace(s->q_pe, rope_dim, pos);
        assert_finite_slice("attn_q_pe", s->q_pe, rope_dim, debug_mode);

        const int8_t *k_b_q_h = w->k_b_q + (size_t)h * (size_t)kv_lora * (size_t)qk_nope;
        const float *k_b_s_h = w->k_b_s + (size_t)h * (size_t)kv_lora * (size_t)(qk_nope / gs);
        matvec_rows(k_b_q_h, k_b_s_h, s->q_nope, s->q_abs, kv_lora, qk_nope, gs);
        assert_finite_slice("attn_q_abs", s->q_abs, kv_lora, debug_mode);

        for (int t = 0; t <= pos; t++) {
            size_t read_slot = layer_base + (size_t)t;
            if (read_slot >= cache_slots) die("attention cache read slot out of bounds");
            const float *kc = s->k_cache_comp + (layer_base + (size_t)t) * (size_t)kv_lora;
            const float *kp = s->k_cache_pe + (layer_base + (size_t)t) * (size_t)rope_dim;
            float score = dot_f32(s->q_abs, kc, kv_lora) + dot_f32(s->q_pe, kp, rope_dim);
            s->att_scores[t] = score * kq_scale;
        }
        softmax_inplace(s->att_scores, pos + 1);
        assert_finite_slice("attn_scores", s->att_scores, pos + 1, debug_mode);
        if (debug_mode && pos <= 1 && h == 0) {
            float s0 = s->att_scores[0];
            float s1 = (pos > 0) ? s->att_scores[1] : 0.0f;
            fprintf(stderr, "[glm-attn-score] layer=%d pos=%d head=0 s0=%.6f s1=%.6f\n", layer_idx, pos, s0, s1);
            fprintf(stderr, "[glm-rope-sum] layer=%d pos=%d q_pe=%.6f k_pe=%.6f\n", layer_idx, pos, sum_f32(s->q_pe, rope_dim), sum_f32(cache_pe, rope_dim));
        }

        memset(s->att_ctx, 0, (size_t)kv_lora * sizeof(float));
        for (int t = 0; t <= pos; t++) {
            const float *kc = s->k_cache_comp + (layer_base + (size_t)t) * (size_t)kv_lora;
            float p = s->att_scores[t];
            for (int i = 0; i < kv_lora; i++) s->att_ctx[i] += p * kc[i];
        }
        assert_finite_slice("attn_ctx", s->att_ctx, kv_lora, debug_mode);

        const int8_t *v_b_q_h = w->v_b_q + (size_t)h * (size_t)v_head * (size_t)kv_lora;
        const float *v_b_s_h = w->v_b_s + (size_t)h * (size_t)v_head * (size_t)(kv_lora / gs);
        matvec_rows(v_b_q_h, v_b_s_h, s->att_ctx, s->att_concat + (size_t)h * (size_t)v_head, v_head, kv_lora, gs);
        assert_finite_slice("attn_concat_h", s->att_concat + (size_t)h * (size_t)v_head, v_head, debug_mode);
    }

    assert_finite_slice("attn_concat", s->att_concat, n_heads * v_head, debug_mode);
    matvec_rows(w->attn_out_q, w->attn_out_s, s->att_concat, s->xb, dim, n_heads * v_head, gs);
    assert_finite_slice("attn_xb", s->xb, dim, debug_mode);
    for (int i = 0; i < dim; i++) s->x[i] += s->xb[i];
    assert_finite_slice("attn_x", s->x, dim, debug_mode);
}

static void apply_l0_dense_ffn(const Runtime *rt, RunState *s, int gs, int debug_mode) {
    if (!rt->has_l0_ffn) return;
    rmsnorm(s->xn, s->x, rt->l0_ffn_norm, rt->cfg.dim);
    assert_finite_slice("l0_ffn_xn", s->xn, rt->cfg.dim, debug_mode);
    matvec_rows(rt->l0_ffn_gate_q, rt->l0_ffn_gate_s, s->xn, s->ff_gate, rt->cfg.hidden_dim, rt->cfg.dim, gs);
    matvec_rows(rt->l0_ffn_up_q, rt->l0_ffn_up_s, s->xn, s->ff_up, rt->cfg.hidden_dim, rt->cfg.dim, gs);
    assert_finite_slice("l0_ffn_gate_pre", s->ff_gate, rt->cfg.hidden_dim, debug_mode);
    assert_finite_slice("l0_ffn_up", s->ff_up, rt->cfg.hidden_dim, debug_mode);
    for (int i = 0; i < rt->cfg.hidden_dim; i++) {
        s->ff_gate[i] = silu(s->ff_gate[i]) * s->ff_up[i];
    }
    assert_finite_slice("l0_ffn_gate", s->ff_gate, rt->cfg.hidden_dim, debug_mode);
    matvec_rows(rt->l0_ffn_down_q, rt->l0_ffn_down_s, s->ff_gate, s->xb, rt->cfg.dim, rt->cfg.hidden_dim, gs);
    assert_finite_slice("l0_ffn_xb", s->xb, rt->cfg.dim, debug_mode);
    for (int i = 0; i < rt->cfg.dim; i++) {
        s->x[i] += s->xb[i];
    }
    assert_finite_slice("l0_ffn_x", s->x, rt->cfg.dim, debug_mode);
}

static void apply_moe_shared_ffn_layer(const Runtime *rt, RunState *s, int gs, int layer_idx, int debug_mode, int pos) {
    if (!rt->has_moe_shared) return;
    if (layer_idx <= 0 || layer_idx >= rt->cfg.n_layers) return;
    if (rt->cfg.moe_ffn_dim <= 0) return;
    const LayerMoeShared *w = &rt->moe_layers[layer_idx];
    if (!w->ffn_norm) return;

    const int dim = rt->cfg.dim;
    const int moe_dim = rt->cfg.moe_ffn_dim;
    const int n_routed = rt->cfg.n_routed_experts;

    rmsnorm(s->xn, s->x, w->ffn_norm, dim);
    assert_finite_slice("moe_xn", s->xn, dim, debug_mode);

    // Router probabilities for routed experts. For DeepSeek2, expert selection uses
    // probs + exp_probs_b, but expert weights themselves remain unbiased probs.
    matvec_f32_rows(w->gate_inp, s->xn, s->moe_router, n_routed, dim);
    for (int e = 0; e < n_routed; e++) {
        float logit = s->moe_router[e];
        s->moe_router[e] = 1.0f / (1.0f + expf(-logit)); // sigmoid gating
    }
    assert_finite_slice("moe_router", s->moe_router, n_routed, debug_mode);

    // Select top-k experts.
    const int top_k = rt->cfg.n_experts_used;
    for (int k = 0; k < top_k; k++) {
        s->moe_topk_idx[k] = -1;
        s->moe_topk_w[k] = -1e30f;
    }
    for (int e = 0; e < n_routed; e++) {
        float p = s->moe_router[e];
        float sel = p + w->exp_probs_b[e];
        int pos = -1;
        for (int k = 0; k < top_k; k++) {
            int idx = s->moe_topk_idx[k];
            float cur_sel = (idx >= 0) ? (s->moe_router[idx] + w->exp_probs_b[idx]) : -INFINITY;
            if (sel > cur_sel) {
                pos = k;
                break;
            }
        }
        if (pos < 0) continue;
        for (int k = top_k - 1; k > pos; k--) {
            s->moe_topk_idx[k] = s->moe_topk_idx[k - 1];
            s->moe_topk_w[k] = s->moe_topk_w[k - 1];
        }
        s->moe_topk_idx[pos] = e;
        s->moe_topk_w[pos] = p;
    }
    float top_sum = 0.0f;
    for (int k = 0; k < top_k; k++) {
        if (s->moe_topk_idx[k] >= 0) top_sum += s->moe_topk_w[k];
    }
    if (top_sum > 0.0f) {
        // Match llama.cpp MoE normalization: clamp denominator to min f16 normal.
        float denom = top_sum < 6.103515625e-5f ? 6.103515625e-5f : top_sum;
        float inv = 1.0f / denom;
        for (int k = 0; k < top_k; k++) s->moe_topk_w[k] *= inv;
    }
    if (debug_mode && layer_idx >= 38 && top_k >= 4) {
        int s_ids = 0;
        float norm_sum = 0.0f;
        for (int k = 0; k < top_k; k++) {
            if (s->moe_topk_idx[k] >= 0) {
                s_ids += s->moe_topk_idx[k];
                norm_sum += s->moe_topk_w[k];
            }
        }
        fprintf(stderr, "[glm-moe-topk] layer=%d ids=%d,%d,%d,%d id_sum=%d pre_sum=%.6f norm_sum=%.6f w=%.6f,%.6f,%.6f,%.6f\n",
                layer_idx, s->moe_topk_idx[0], s->moe_topk_idx[1], s->moe_topk_idx[2], s->moe_topk_idx[3], s_ids, top_sum, norm_sum,
                s->moe_topk_w[0], s->moe_topk_w[1], s->moe_topk_w[2], s->moe_topk_w[3]);
    }

    memset(s->moe_out_acc, 0, (size_t)dim * sizeof(float));
    int disable_moe_routed = 0;
    const char *disable_moe_routed_env = getenv("GLM_DISABLE_MOE_ROUTED");
    if (disable_moe_routed_env && disable_moe_routed_env[0] && disable_moe_routed_env[0] != '0') {
        disable_moe_routed = 1;
    }
    if (!disable_moe_routed && rt->has_moe_routed && rt->moe_routed_layers) {
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

                matvec_rows(gq, gsr, s->xn, s->moe_gate, moe_dim, dim, gs);
                matvec_rows(uq, usr, s->xn, s->moe_up, moe_dim, dim, gs);
                assert_finite_slice("moe_routed_gate_pre", s->moe_gate, moe_dim, debug_mode);
                assert_finite_slice("moe_routed_up", s->moe_up, moe_dim, debug_mode);
                for (int i = 0; i < moe_dim; i++) s->moe_gate[i] = silu(s->moe_gate[i]) * s->moe_up[i];
                assert_finite_slice("moe_routed_gate", s->moe_gate, moe_dim, debug_mode);
                matvec_rows(dq, dsr, s->moe_gate, s->xb, dim, moe_dim, gs);
                assert_finite_slice("moe_routed_xb", s->xb, dim, debug_mode);
                for (int i = 0; i < dim; i++) s->moe_out_acc[i] += w_e * s->xb[i];
            }
        }
    }

    matvec_rows(w->gate_sh_q, w->gate_sh_s, s->xn, s->moe_gate, moe_dim, dim, gs);
    matvec_rows(w->up_sh_q, w->up_sh_s, s->xn, s->moe_up, moe_dim, dim, gs);
    assert_finite_slice("moe_shared_gate_pre", s->moe_gate, moe_dim, debug_mode);
    assert_finite_slice("moe_shared_up", s->moe_up, moe_dim, debug_mode);
    for (int i = 0; i < moe_dim; i++) {
        s->moe_gate[i] = silu(s->moe_gate[i]) * s->moe_up[i];
    }
    assert_finite_slice("moe_shared_gate", s->moe_gate, moe_dim, debug_mode);
    matvec_rows(w->down_sh_q, w->down_sh_s, s->moe_gate, s->xb, dim, moe_dim, gs);
    assert_finite_slice("moe_shared_xb", s->xb, dim, debug_mode);
    if (debug_mode) {
        fprintf(stderr, "[glm-layer-sum] ffn_moe_out-%d sum=%.6f\n", layer_idx, sum_f32(s->moe_out_acc, dim));
        fprintf(stderr, "[glm-layer-sum] ffn_shexp-%d sum=%.6f\n", layer_idx, sum_f32(s->xb, dim));
    }
    assert_finite_slice("moe_out_acc", s->moe_out_acc, dim, debug_mode);
    for (int i = 0; i < dim; i++) {
        s->x[i] += s->xb[i] + s->moe_out_acc[i];
    }
    assert_finite_slice("moe_x", s->x, dim, debug_mode);
}

static int argmax(const float *x, int n) {
    int idx = 0;
    float best = x[0];
    for (int i = 1; i < n; i++) {
        if (x[i] > best) {
            best = x[i];
            idx = i;
        }
    }
    return idx;
}

static int sample_logits_temperature(const float *logits, int n, float temperature) {
    if (temperature <= 0.0f) return argmax(logits, n);

    float max_logit = logits[0];
    for (int i = 1; i < n; i++) {
        if (logits[i] > max_logit) max_logit = logits[i];
    }

    const double inv_temp = 1.0 / (double)temperature;
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += exp(((double)logits[i] - (double)max_logit) * inv_temp);
    }
    if (!(sum > 0.0) || !isfinite(sum)) return argmax(logits, n);

    const double r = ((double)rand() / ((double)RAND_MAX + 1.0)) * sum;
    double cdf = 0.0;
    for (int i = 0; i < n; i++) {
        cdf += exp(((double)logits[i] - (double)max_logit) * inv_temp);
        if (cdf >= r) return i;
    }
    return n - 1;
}

static void debug_print_prompt_embd_batch(const Runtime *rt, const int *prompt_ids, int n_prompt) {
    if (n_prompt <= 0) return;

    const int dim = rt->cfg.dim;
    const int gs = (int)rt->cfg.group_size;
    const int n_groups = dim / gs;
    const int head_vals = dim < 8 ? dim : 8;
    const int head_rows = n_prompt < 4 ? n_prompt : 4;

    float *row = (float *)malloc((size_t)dim * sizeof(float));
    if (!row) die("out of memory");

    double sum = 0.0;
    fprintf(stderr,
            "[glm-debug-callback] embd = (f32) GET_ROWS(token_embd.weight{%d, %d, 1, 1}, inp_tokens{%d, 1, 1, 1}) = {%d, %d, 1, 1}\n",
            dim, rt->cfg.vocab_size, n_prompt, dim, n_prompt);

    for (int t = 0; t < n_prompt; t++) {
        int tok = prompt_ids[t];
        if (tok < 0 || tok >= rt->cfg.vocab_size) tok = 0;
        const int8_t *eq = rt->tok_q + (size_t)tok * (size_t)dim;
        const float *es = rt->tok_s + (size_t)tok * (size_t)n_groups;
        dequant_row(eq, es, row, dim, gs);

        for (int i = 0; i < dim; i++) sum += row[i];

        if (t < head_rows) {
            fprintf(stderr, "[glm-debug-callback]   token[%d]=%d row_head:", t, tok);
            for (int i = 0; i < head_vals; i++) {
                fprintf(stderr, " %.4f", row[i]);
            }
            if (dim > head_vals) fprintf(stderr, " ...");
            fprintf(stderr, "\n");
        }
    }
    if (n_prompt > head_rows) {
        fprintf(stderr, "[glm-debug-callback]   ... (%d more tokens)\n", n_prompt - head_rows);
    }
    fprintf(stderr, "[glm-debug-callback]   sum = %.6f\n", sum);

    free(row);
}

static void print_token_safe(const Tokenizer *tok, const Token *t) {
    if (!tok || !t || !t->bytes || t->len == 0) return;
    const unsigned char *s = (const unsigned char *)t->bytes;
    size_t i = 0;
    while (i < t->len) {
        uint32_t cp = 0;
        size_t used = utf8_decode_cp(s + i, (size_t)t->len - i, &cp);
        if (used == 0) break;
        int mapped = (cp < 512u) ? tok->cp_to_byte[cp] : -1;
        if (mapped >= 0) {
            putchar_safe_byte((unsigned char)mapped);
        } else {
            for (size_t j = 0; j < used; j++) putchar_safe_byte(s[i + j]);
        }
        i += used;
    }
}

static int lookup_token_id(const Tokenizer *tok, const char *bytes, uint32_t len) {
    if (len == 1) {
        int id = tok->ascii_map[(unsigned char)bytes[0]];
        if (id >= 0 && tok->tokens[id].len == 1 && tok->tokens[id].bytes[0] == bytes[0]) return id;
    }
    for (size_t i = 0; i < tok->n_tokens; i++) {
        if (tok->tokens[i].len != len) continue;
        if (len == 0) continue;
        if (memcmp(tok->tokens[i].bytes, bytes, len) == 0) return (int)i;
    }
    return -1;
}

static int encode_prompt_bpe(const Tokenizer *tok, const char *prompt, int *ids, int max_ids, int bos) {
    if (max_ids <= 0) return 0;

    int n = 0;
    if (bos >= 0 && n < max_ids) ids[n++] = bos;
    if (!prompt || prompt[0] == '\0') return n;

    const unsigned char *p = (const unsigned char *)prompt;
    while (*p && n < max_ids) {
        int id = -1;
        int consumed = 1;

        // Special-token path: "<...>" up to 64 bytes.
        if (*p == '<') {
            int end = -1;
            for (int k = 0; p[k] != '\0' && k < 64; k++) {
                if (p[k] == '>') {
                    end = k;
                    break;
                }
            }
            if (end >= 0) {
                uint32_t slen = (uint32_t)(end + 1);
                id = lookup_token_id(tok, (const char *)p, slen);
                if (id >= 0) consumed = end + 1;
            }
        }

        if (id < 0) {
            char mapped[4];
            int mapped_len = utf8_encode_cp((uint32_t)tok->byte_to_cp[*p], mapped);
            id = lookup_token_id(tok, mapped, (uint32_t)mapped_len);
            if (id < 0) {
                char raw = (char)(*p);
                id = lookup_token_id(tok, &raw, 1);
            }
        }

        if (id >= 0 && n < max_ids) ids[n++] = id;
        p += consumed;
    }

    // Greedy BPE merges by merge score.
    char *merge_buf = (char *)malloc((size_t)tok->max_token_len * 2u + 4u);
    if (!merge_buf) die("out of memory");
    while (1) {
        float best = -1e30f;
        int best_id = -1;
        int best_idx = -1;

        for (int i = 0; i < n - 1; i++) {
            int a = ids[i];
            int b = ids[i + 1];
            if (a < 0 || b < 0 || a >= (int)tok->n_tokens || b >= (int)tok->n_tokens) continue;
            uint32_t la = tok->tokens[a].len;
            uint32_t lb = tok->tokens[b].len;
            uint32_t l = la + lb;
            if (l == 0) continue;
            memcpy(merge_buf, tok->tokens[a].bytes, la);
            memcpy(merge_buf + la, tok->tokens[b].bytes, lb);
            int merged = lookup_token_id(tok, merge_buf, l);
            if (merged >= 0 && tok->scores[merged] > best) {
                best = tok->scores[merged];
                best_id = merged;
                best_idx = i;
            }
        }

        if (best_idx < 0) break;
        ids[best_idx] = best_id;
        for (int i = best_idx + 1; i < n - 1; i++) ids[i] = ids[i + 1];
        n--;
    }
    free(merge_buf);
    return n;
}

static void usage(const char *prog) {
    fprintf(stderr, "Usage: %s <checkpoint.bin> [-n N] [-i PROMPT|-f PROMPT_FILE] [-m completion|chat] [-T CONTEXT] [-t TEMP] [--seed N] [--backend cpu|metal] [--debug] [--self-test-tokenizer] [--bench-mode full|prefill|decode] [--bench-report PATH]\n", prog);
}

typedef enum {
    BENCH_MODE_OFF = 0,
    BENCH_MODE_FULL,
    BENCH_MODE_PREFILL,
    BENCH_MODE_DECODE,
} BenchMode;

typedef struct {
    int n;
    double mean_ms;
    double p50_ms;
    double p95_ms;
    double tok_s;
    double active_bytes;
    double bw_gbps;
} BenchPhaseReport;

static double monotonic_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1000000.0;
}

static int latency_push(double **arr, int *n, int *cap, double value) {
    if (*n >= *cap) {
        int next = (*cap == 0) ? 128 : (*cap * 2);
        double *tmp = (double *)realloc(*arr, (size_t)next * sizeof(double));
        if (!tmp) return 0;
        *arr = tmp;
        *cap = next;
    }
    (*arr)[(*n)++] = value;
    return 1;
}

static int cmp_double_asc(const void *a, const void *b) {
    const double da = *(const double *)a;
    const double db = *(const double *)b;
    if (da < db) return -1;
    if (da > db) return 1;
    return 0;
}

static double percentile_sorted(const double *sorted, int n, double pct) {
    if (!sorted || n <= 0) return 0.0;
    if (pct <= 0.0) return sorted[0];
    if (pct >= 100.0) return sorted[n - 1];
    double rank = (pct / 100.0) * (double)(n - 1);
    int lo = (int)rank;
    int hi = lo + 1;
    if (hi >= n) return sorted[n - 1];
    double frac = rank - (double)lo;
    return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
}

static double q80_bytes(size_t rows, size_t cols, size_t gs) {
    if (rows == 0 || cols == 0 || gs == 0) return 0.0;
    size_t groups = cols / gs;
    double q = (double)rows * (double)cols * (double)sizeof(int8_t);
    double s = (double)rows * (double)groups * (double)sizeof(float);
    return q + s;
}

static double estimate_model_active_bytes(const Runtime *rt) {
    if (!rt) return 0.0;
    const RuntimeConfig *cfg = &rt->cfg;
    size_t dim = (size_t)cfg->dim;
    size_t hidden = (size_t)cfg->hidden_dim;
    size_t vocab = (size_t)cfg->vocab_size;
    size_t gs = (size_t)cfg->group_size;
    if (gs == 0 || dim == 0) return 0.0;

    double bytes = 0.0;
    bytes += (double)dim * sizeof(int8_t) + (double)(dim / gs) * sizeof(float);
    bytes += (double)dim * sizeof(float);
    bytes += q80_bytes(vocab, dim, gs);

    if (rt->has_l0_ffn) {
        bytes += (double)dim * sizeof(float);
        bytes += q80_bytes(hidden, dim, gs);
        bytes += q80_bytes(hidden, dim, gs);
        bytes += q80_bytes(dim, hidden, gs);
    }

    int n_attn_layers = rt->has_all_attn ? cfg->n_layers : (rt->has_l0_attn ? 1 : 0);
    if (n_attn_layers > 0 && cfg->l0_qk_nope_dim > 0 && cfg->l0_head_k_dim > 0 && cfg->l0_v_head_dim > 0) {
        size_t q_lora = (size_t)cfg->q_lora_rank;
        size_t kv_lora = (size_t)cfg->kv_lora_rank;
        size_t rope = (size_t)cfg->rope_dim;
        size_t n_heads = (size_t)cfg->n_heads;
        size_t qk_nope = (size_t)cfg->l0_qk_nope_dim;
        size_t head_k = (size_t)cfg->l0_head_k_dim;
        size_t v_head = (size_t)cfg->l0_v_head_dim;
        size_t attn_out = n_heads * v_head;

        double per_layer = 0.0;
        per_layer += (double)dim * sizeof(float);
        per_layer += (double)q_lora * sizeof(float);
        per_layer += (double)kv_lora * sizeof(float);
        per_layer += q80_bytes(q_lora, dim, gs);
        per_layer += q80_bytes(n_heads * head_k, q_lora, gs);
        per_layer += q80_bytes(kv_lora + rope, dim, gs);
        per_layer += q80_bytes(n_heads * kv_lora, qk_nope, gs);
        per_layer += q80_bytes(n_heads * v_head, kv_lora, gs);
        per_layer += q80_bytes(dim, attn_out, gs);

        bytes += per_layer * (double)n_attn_layers;
    }

    if (rt->has_moe_shared && cfg->moe_ffn_dim > 0 && cfg->n_layers > 1) {
        size_t moe_dim = (size_t)cfg->moe_ffn_dim;
        size_t n_routed = (size_t)cfg->n_routed_experts;
        int n_moe_layers = cfg->n_layers - 1;

        double shared = 0.0;
        shared += (double)dim * sizeof(float);
        shared += (double)n_routed * (double)dim * sizeof(float);
        shared += (double)n_routed * sizeof(float);
        shared += q80_bytes(moe_dim, dim, gs);
        shared += q80_bytes(moe_dim, dim, gs);
        shared += q80_bytes(dim, moe_dim, gs);
        bytes += shared * (double)n_moe_layers;

        if (rt->has_moe_routed && cfg->n_experts_used > 0) {
            double per_expert = 0.0;
            per_expert += q80_bytes(moe_dim, dim, gs);
            per_expert += q80_bytes(moe_dim, dim, gs);
            per_expert += q80_bytes(dim, moe_dim, gs);
            bytes += per_expert * (double)cfg->n_experts_used * (double)n_moe_layers;
        }
    }

    return bytes;
}

static double estimate_cache_active_bytes(const Runtime *rt, const RunState *st, int pos) {
    if (!rt || !st || pos < 0 || st->n_attn_layers <= 0) return 0.0;
    double kv = (double)rt->cfg.kv_lora_rank;
    double rope = (double)rt->cfg.rope_dim;
    double heads = (double)rt->cfg.n_heads;
    double layers = (double)st->n_attn_layers;
    double t = (double)(pos + 1);
    double bytes_f = (double)sizeof(float);

    double read_scores = heads * t * (kv + rope) * bytes_f;
    double read_ctx = heads * t * kv * bytes_f;
    double write_cache = (kv + rope) * bytes_f;
    return layers * (read_scores + read_ctx + write_cache);
}

static BenchPhaseReport build_phase_report(const double *samples, int n, double model_bytes, double cache_sum_bytes) {
    BenchPhaseReport r = {0};
    if (!samples || n <= 0) return r;

    double *sorted = (double *)malloc((size_t)n * sizeof(double));
    if (!sorted) return r;
    double total = 0.0;
    for (int i = 0; i < n; i++) {
        sorted[i] = samples[i];
        total += samples[i];
    }
    qsort(sorted, (size_t)n, sizeof(double), cmp_double_asc);

    r.n = n;
    r.mean_ms = total / (double)n;
    r.p50_ms = percentile_sorted(sorted, n, 50.0);
    r.p95_ms = percentile_sorted(sorted, n, 95.0);
    r.tok_s = (r.mean_ms > 0.0) ? (1000.0 / r.mean_ms) : 0.0;
    r.active_bytes = model_bytes + (cache_sum_bytes / (double)n);
    r.bw_gbps = (r.mean_ms > 0.0) ? ((r.active_bytes / (r.mean_ms / 1000.0)) / 1e9) : 0.0;
    free(sorted);
    return r;
}

static void print_phase_report(const char *phase, BenchPhaseReport r) {
    if (!phase || r.n <= 0) return;
    fprintf(stderr,
            "[glm-bench] phase=%s tokens=%d tok_s=%.3f mean_ms=%.3f p50_ms=%.3f p95_ms=%.3f active_bytes=%.0f est_bw_gbps=%.3f\n",
            phase, r.n, r.tok_s, r.mean_ms, r.p50_ms, r.p95_ms, r.active_bytes, r.bw_gbps);
}

static void write_bench_report_json(const char *path,
                                    const char *backend,
                                    BenchMode mode,
                                    int prompt_tokens,
                                    int generated_tokens,
                                    BenchPhaseReport prefill,
                                    BenchPhaseReport decode) {
    if (!path || path[0] == '\0') return;
    FILE *f = fopen(path, "w");
    if (!f) {
        fprintf(stderr, "[glm-bench] warning: failed to open report path: %s\n", path);
        return;
    }
    const char *mode_s = mode == BENCH_MODE_PREFILL ? "prefill" : (mode == BENCH_MODE_DECODE ? "decode" : "full");
    fprintf(f, "{\n");
    fprintf(f, "  \"backend\": \"%s\",\n", backend ? backend : "cpu");
    fprintf(f, "  \"mode\": \"%s\",\n", mode_s);
    fprintf(f, "  \"prompt_tokens\": %d,\n", prompt_tokens);
    fprintf(f, "  \"generated_tokens\": %d,\n", generated_tokens);
    fprintf(f, "  \"prefill\": {\n");
    fprintf(f, "    \"tokens\": %d,\n", prefill.n);
    fprintf(f, "    \"tok_s\": %.6f,\n", prefill.tok_s);
    fprintf(f, "    \"mean_ms\": %.6f,\n", prefill.mean_ms);
    fprintf(f, "    \"p50_ms\": %.6f,\n", prefill.p50_ms);
    fprintf(f, "    \"p95_ms\": %.6f,\n", prefill.p95_ms);
    fprintf(f, "    \"active_bytes\": %.0f,\n", prefill.active_bytes);
    fprintf(f, "    \"est_bw_gbps\": %.6f\n", prefill.bw_gbps);
    fprintf(f, "  },\n");
    fprintf(f, "  \"decode\": {\n");
    fprintf(f, "    \"tokens\": %d,\n", decode.n);
    fprintf(f, "    \"tok_s\": %.6f,\n", decode.tok_s);
    fprintf(f, "    \"mean_ms\": %.6f,\n", decode.mean_ms);
    fprintf(f, "    \"p50_ms\": %.6f,\n", decode.p50_ms);
    fprintf(f, "    \"p95_ms\": %.6f,\n", decode.p95_ms);
    fprintf(f, "    \"active_bytes\": %.0f,\n", decode.active_bytes);
    fprintf(f, "    \"est_bw_gbps\": %.6f\n", decode.bw_gbps);
    fprintf(f, "  }\n");
    fprintf(f, "}\n");
    fclose(f);
}

static char *read_text_file_alloc(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;

    if (fseek(f, 0, SEEK_END) != 0) {
        fclose(f);
        return NULL;
    }
    long n = ftell(f);
    if (n < 0) {
        fclose(f);
        return NULL;
    }
    if (fseek(f, 0, SEEK_SET) != 0) {
        fclose(f);
        return NULL;
    }

    size_t size = (size_t)n;
    char *buf = (char *)malloc(size + 1);
    if (!buf) {
        fclose(f);
        return NULL;
    }
    size_t got = fread(buf, 1, size, f);
    fclose(f);
    if (got != size) {
        free(buf);
        return NULL;
    }
    buf[size] = '\0';
    return buf;
}

static void assert_finite_slice(const char *name, const float *x, int n, int debug_mode) {
    if (!debug_mode) return;
    for (int i = 0; i < n; i++) {
        if (!isfinite(x[i])) {
            fprintf(stderr, "[error] non-finite value in %s at i=%d: %f\n", name, i, x[i]);
            die("numerical sanity check failed");
        }
    }
    if (debug_mode) {
        float minv = x[0], maxv = x[0], sum = 0.0f;
        for (int i = 0; i < n; i++) {
            if (x[i] < minv) minv = x[i];
            if (x[i] > maxv) maxv = x[i];
            sum += x[i];
        }
        fprintf(stderr, "[debug] %s stats: min=%.6f max=%.6f mean=%.6f\n", name, minv, maxv, sum / (float)n);
    }
}

static int tokenizer_roundtrip_check(const Tokenizer *tok, const char *text, int bos_id) {
    int ids[2048];
    int n = encode_prompt_bpe(tok, text, ids, (int)(sizeof(ids) / sizeof(ids[0])), bos_id);
    if (n <= 0) return 0;

    char out[8192];
    size_t out_n = 0;
    int start = 0;
    if (n > 0 && ids[0] == bos_id) start = 1;

    for (int i = start; i < n; i++) {
        int id = ids[i];
        if (id < 0 || id >= (int)tok->n_tokens) return 0;
        const Token *t = &tok->tokens[id];
        if (!append_decoded_token_bytes(tok, t, out, &out_n, sizeof(out))) return 0;
    }
    out[out_n] = '\0';
    return strcmp(out, text) == 0;
}

static int run_tokenizer_self_test(const Tokenizer *tok, int bos_id) {
    const char *cases[] = {
        "1+1=",
        "abc123",
        "sum(2,3)",
        "z=",
        "Hello i am batman. My name is",
        "line1\nline2",
    };
    int ok = 1;
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        int pass = tokenizer_roundtrip_check(tok, cases[i], bos_id);
        fprintf(stderr, "[tokenizer-selftest] \"%s\" -> %s\n", cases[i], pass ? "PASS" : "FAIL");
        if (!pass) ok = 0;
    }
    return ok;
}

static void print_topk_debug(const float *logits, const Tokenizer *tok, int vocab_size, int k) {
    if (k <= 0) return;
    int *idx = (int *)calloc((size_t)k, sizeof(int));
    float *val = (float *)calloc((size_t)k, sizeof(float));
    if (!idx || !val) die("out of memory");
    for (int i = 0; i < k; i++) {
        idx[i] = -1;
        val[i] = -INFINITY;
    }

    for (int i = 0; i < vocab_size; i++) {
        float x = logits[i];
        int pos = -1;
        for (int j = 0; j < k; j++) {
            if (x > val[j]) {
                pos = j;
                break;
            }
        }
        if (pos < 0) continue;
        for (int j = k - 1; j > pos; j--) {
            val[j] = val[j - 1];
            idx[j] = idx[j - 1];
        }
        val[pos] = x;
        idx[pos] = i;
    }

    fprintf(stderr, "[debug] top-%d logits:\n", k);
    for (int i = 0; i < k; i++) {
        if (idx[i] < 0) continue;
        fprintf(stderr, "  #%d id=%d logit=%.6f token=\"", i + 1, idx[i], val[i]);
        const Token *t = &tok->tokens[idx[i]];
        for (uint32_t j = 0; j < t->len; j++) {
            unsigned char c = (unsigned char)t->bytes[j];
            if (isprint(c) && c != '"') fputc((int)c, stderr);
            else fputc(' ', stderr);
        }
        fprintf(stderr, "\"\n");
    }

    free(idx);
    free(val);
}

static int glm_cpu_forward_token_impl(const Runtime *rt,
                                      RunState *st,
                                      int token,
                                      int pos,
                                      int debug_mode,
                                      float *attn_layer_out,
                                      int attn_layer_stride,
                                      int max_attn_layers,
                                      int *written_attn_layers,
                                      float *layer_out,
                                      int layer_stride,
                                      int max_layers,
                                      int *written_layers) {
    if (!rt || !st) return -1;
    if (token < 0 || token >= rt->cfg.vocab_size) token = 0;
    if (written_layers) *written_layers = 0;
    if (written_attn_layers) *written_attn_layers = 0;

    int attn_checkpoints_enabled = (attn_layer_out != NULL && attn_layer_stride >= rt->cfg.dim && max_attn_layers > 0);
    int checkpoints_enabled = (layer_out != NULL && layer_stride >= rt->cfg.dim && max_layers > 0);
    int attn_layers_written = 0;
    int layers_written = 0;

    LayerAttnWeights l0_attn_weights = {
        .attn_norm = rt->l0_attn_norm,
        .q_a_norm = rt->l0_q_a_norm,
        .kv_a_norm = rt->l0_kv_a_norm,
        .q_a_q = rt->l0_q_a_q,
        .q_a_s = rt->l0_q_a_s,
        .q_b_q = rt->l0_q_b_q,
        .q_b_s = rt->l0_q_b_s,
        .kv_a_q = rt->l0_kv_a_q,
        .kv_a_s = rt->l0_kv_a_s,
        .k_b_q = rt->l0_k_b_q,
        .k_b_s = rt->l0_k_b_s,
        .v_b_q = rt->l0_v_b_q,
        .v_b_s = rt->l0_v_b_s,
        .attn_out_q = rt->l0_attn_out_q,
        .attn_out_s = rt->l0_attn_out_s,
    };

    int gs = (int)rt->cfg.group_size;
    int n_groups = rt->cfg.dim / gs;
    const int8_t *eq = rt->tok_q + (size_t)token * (size_t)rt->cfg.dim;
    const float *es = rt->tok_s + (size_t)token * (size_t)n_groups;
    dequant_row(eq, es, st->x, rt->cfg.dim, gs);
    assert_finite_slice("embed", st->x, rt->cfg.dim, debug_mode);
    if (debug_mode && pos == 0) {
        debug_print_tensor_sum("embd", st->x, rt->cfg.dim);
    }

    if (rt->has_all_attn) {
        for (int l = 0; l < rt->cfg.n_layers; l++) {
            float sum_before_attn = 0.0f;
            if (debug_mode && pos == 0) {
                sum_before_attn = sum_f32(st->x, rt->cfg.dim);
            }
            apply_mla_attention_layer(rt, &rt->attn_layers[l], st, gs, pos, l, debug_mode);
            if (attn_checkpoints_enabled && l < max_attn_layers) {
                memcpy(attn_layer_out + (size_t)l * (size_t)attn_layer_stride,
                       st->x,
                       (size_t)rt->cfg.dim * sizeof(float));
                attn_layers_written = l + 1;
            }
            float sum_before_ffn = 0.0f;
            if (debug_mode && pos == 0) {
                char name[32];
                snprintf(name, sizeof(name), "Vcur-%d", l);
                fprintf(stderr, "[glm-layer-sum] %s sum=%.6f\n", name, sum_f32(st->kv_a, rt->cfg.kv_lora_rank));
                snprintf(name, sizeof(name), "kqv_out-%d", l);
                fprintf(stderr, "[glm-layer-sum] %s sum=%.6f\n", name, sum_f32(st->att_concat, rt->cfg.n_heads * rt->cfg.l0_v_head_dim));
                snprintf(name, sizeof(name), "ffn_inp-%d", l);
                debug_print_tensor_sum(name, st->x, rt->cfg.dim);
                snprintf(name, sizeof(name), "attn_out-%d", l);
                fprintf(stderr, "[glm-layer-sum] %s sum=%.6f\n", name, sum_f32(st->x, rt->cfg.dim) - sum_before_attn);
                sum_before_ffn = sum_f32(st->x, rt->cfg.dim);
            }
            if (l == 0) apply_l0_dense_ffn(rt, st, gs, debug_mode);
            else apply_moe_shared_ffn_layer(rt, st, gs, l, debug_mode, pos);

            if (checkpoints_enabled && l < max_layers) {
                memcpy(layer_out + (size_t)l * (size_t)layer_stride, st->x, (size_t)rt->cfg.dim * sizeof(float));
                layers_written = l + 1;
            }

            if (debug_mode && pos == 0) {
                char name[32];
                snprintf(name, sizeof(name), "l_out-%d", l);
                debug_print_tensor_sum(name, st->x, rt->cfg.dim);
                snprintf(name, sizeof(name), "ffn_out-%d", l);
                float ffn_out_sum = sum_f32(st->x, rt->cfg.dim) - sum_before_ffn;
                fprintf(stderr, "[glm-layer-sum] %s sum=%.6f\n", name, ffn_out_sum);
            }
        }
        assert_finite_slice("attn_stack_out", st->x, rt->cfg.dim, debug_mode);
    } else if (rt->has_l0_attn) {
        apply_mla_attention_layer(rt, &l0_attn_weights, st, gs, pos, 0, debug_mode);
        if (attn_checkpoints_enabled && max_attn_layers > 0) {
            memcpy(attn_layer_out, st->x, (size_t)rt->cfg.dim * sizeof(float));
            attn_layers_written = 1;
        }
        assert_finite_slice("l0_attn_out", st->x, rt->cfg.dim, debug_mode);
        apply_l0_dense_ffn(rt, st, gs, debug_mode);
        if (checkpoints_enabled && max_layers > 0) {
            memcpy(layer_out, st->x, (size_t)rt->cfg.dim * sizeof(float));
            layers_written = 1;
        }
        if (debug_mode && pos == 0) {
            debug_print_tensor_sum("l_out-0", st->x, rt->cfg.dim);
        }
    } else {
        apply_l0_dense_ffn(rt, st, gs, debug_mode);
        if (checkpoints_enabled && max_layers > 0) {
            memcpy(layer_out, st->x, (size_t)rt->cfg.dim * sizeof(float));
            layers_written = 1;
        }
        if (debug_mode && pos == 0) {
            debug_print_tensor_sum("l_out-0", st->x, rt->cfg.dim);
        }
    }
    if (rt->has_l0_ffn) assert_finite_slice("l0_ffn_out", st->x, rt->cfg.dim, debug_mode);
    rmsnorm(st->xn, st->x, rt->output_norm, rt->cfg.dim);
    assert_finite_slice("final_rmsnorm", st->xn, rt->cfg.dim, debug_mode);
    if (debug_mode && pos == 0) {
        debug_print_tensor_sum("result_norm", st->xn, rt->cfg.dim);
    }
    matvec_rows(rt->out_q, rt->out_s, st->xn, st->logits, rt->cfg.vocab_size, rt->cfg.dim, gs);
    assert_finite_slice("logits", st->logits, rt->cfg.vocab_size, debug_mode);
    if (debug_mode && pos == 0) {
        debug_print_tensor_sum("result_output", st->logits, rt->cfg.vocab_size);
    }
    if (written_layers) *written_layers = layers_written;
    if (written_attn_layers) *written_attn_layers = attn_layers_written;
    return 0;
}

int glm_cpu_forward_token(const Runtime *rt, RunState *st, int token, int pos, int debug_mode) {
    return glm_cpu_forward_token_impl(rt, st, token, pos, debug_mode, NULL, 0, 0, NULL, NULL, 0, 0, NULL);
}

int glm_cpu_forward_token_checkpoints(const Runtime *rt,
                                      const RunState *src,
                                      int token,
                                      int pos,
                                      int debug_mode,
                                      float *layer_out,
                                      int layer_stride,
                                      int max_layers,
                                      int *written_layers) {
    if (!rt || !src || !layer_out || layer_stride < rt->cfg.dim || max_layers <= 0) return -1;
    if (src->cache_cap <= 0) return -1;

    RunState tmp;
    runstate_build(&tmp, rt, src->cache_cap);
    runstate_copy_from(rt, src, &tmp);

    int local_written = 0;
    int rc = glm_cpu_forward_token_impl(rt,
                                        &tmp,
                                        token,
                                        pos,
                                        debug_mode,
                                        NULL,
                                        0,
                                        0,
                                        NULL,
                                        layer_out,
                                        layer_stride,
                                        max_layers,
                                        &local_written);
    if (written_layers) *written_layers = local_written;
    runstate_free(&tmp);
    return rc;
}

int glm_cpu_forward_token_dual_checkpoints(const Runtime *rt,
                                           const RunState *src,
                                           int token,
                                           int pos,
                                           int debug_mode,
                                           float *attn_layer_out,
                                           int attn_layer_stride,
                                           int max_attn_layers,
                                           int *written_attn_layers,
                                           float *layer_out,
                                           int layer_stride,
                                           int max_layers,
                                           int *written_layers) {
    if (!rt || !src) return -1;
    if (src->cache_cap <= 0) return -1;
    if (attn_layer_out && (attn_layer_stride < rt->cfg.dim || max_attn_layers <= 0)) return -1;
    if (layer_out && (layer_stride < rt->cfg.dim || max_layers <= 0)) return -1;

    RunState tmp;
    runstate_build(&tmp, rt, src->cache_cap);
    runstate_copy_from(rt, src, &tmp);

    int local_written_attn = 0;
    int local_written = 0;
    int rc = glm_cpu_forward_token_impl(rt,
                                        &tmp,
                                        token,
                                        pos,
                                        debug_mode,
                                        attn_layer_out,
                                        attn_layer_stride,
                                        max_attn_layers,
                                        &local_written_attn,
                                        layer_out,
                                        layer_stride,
                                        max_layers,
                                        &local_written);
    if (written_attn_layers) *written_attn_layers = local_written_attn;
    if (written_layers) *written_layers = local_written;
    runstate_free(&tmp);
    return rc;
}

int glm_app_main(int argc, char **argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    const char *checkpoint = argv[1];
    int n_tokens = 100;
    const char *prompt = "";
    const char *prompt_path = NULL;
    const char *mode = "completion";
    int context_limit = 0;
    float temperature = 0.0f;
    int seed = -1;
    int debug_mode = 0;
    int self_test_tokenizer = 0;
    int backend_set = 0;
    BackendType backend = BACKEND_CPU;
    BenchMode bench_mode = BENCH_MODE_OFF;
    const char *bench_report_path = NULL;

    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) n_tokens = atoi(argv[++i]);
        else if (strcmp(argv[i], "-i") == 0 && i + 1 < argc) prompt = argv[++i];
        else if (strcmp(argv[i], "-f") == 0 && i + 1 < argc) prompt_path = argv[++i];
        else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) mode = argv[++i];
        else if (strcmp(argv[i], "-T") == 0 && i + 1 < argc) context_limit = atoi(argv[++i]);
        else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) temperature = (float)atof(argv[++i]);
        else if (strcmp(argv[i], "--seed") == 0 && i + 1 < argc) seed = atoi(argv[++i]);
        else if (strcmp(argv[i], "--debug") == 0) debug_mode = 1;
        else if (strcmp(argv[i], "--self-test-tokenizer") == 0) self_test_tokenizer = 1;
        else if (strcmp(argv[i], "--bench-mode") == 0 && i + 1 < argc) {
            const char *v = argv[++i];
            if (strcmp(v, "full") == 0) bench_mode = BENCH_MODE_FULL;
            else if (strcmp(v, "prefill") == 0) bench_mode = BENCH_MODE_PREFILL;
            else if (strcmp(v, "decode") == 0) bench_mode = BENCH_MODE_DECODE;
            else {
                fprintf(stderr, "Error: unknown bench mode '%s' (expected full|prefill|decode)\n", v);
                return 1;
            }
        }
        else if (strcmp(argv[i], "--bench-report") == 0 && i + 1 < argc) {
            bench_report_path = argv[++i];
        }
        else if (strcmp(argv[i], "--backend") == 0 && i + 1 < argc) {
            const char *v = argv[++i];
            if (strcmp(v, "cpu") == 0) {
                backend = BACKEND_CPU;
                backend_set = 1;
            } else if (strcmp(v, "metal") == 0) {
                backend = BACKEND_METAL;
                backend_set = 1;
            } else {
                fprintf(stderr, "Error: unknown backend '%s' (expected cpu|metal)\n", v);
                return 1;
            }
        }
        else {
            usage(argv[0]);
            return 1;
        }
    }
    if (!backend_set) {
        const char *env_backend = getenv("GLM_BACKEND");
        if (env_backend && strcmp(env_backend, "metal") == 0) backend = BACKEND_METAL;
    }
    if (temperature < 0.0f) {
        fprintf(stderr, "Error: temperature must be >= 0.\n");
        return 1;
    }
    if (context_limit < 0) {
        fprintf(stderr, "Error: context limit must be >= 0.\n");
        return 1;
    }
    if (prompt[0] != '\0' && prompt_path != NULL) {
        fprintf(stderr, "Error: use only one of -i or -f.\n");
        usage(argv[0]);
        return 1;
    }

    Runtime rt;
    runtime_open(&rt, checkpoint);

    Tokenizer tok;
    load_tokenizer(checkpoint, &tok, rt.cfg.vocab_size);

    if (bench_mode == BENCH_MODE_OFF) {
        printf("[glm4.7-flash] loaded checkpoint\n");
        printf("  dim=%d layers=%d heads=%d kv_heads=%d vocab=%d seq_len=%d\n",
               rt.cfg.dim, rt.cfg.n_layers, rt.cfg.n_heads, rt.cfg.n_kv_heads, rt.cfg.vocab_size, rt.cfg.seq_len);
        printf("  mode=%s version=%u\n", mode, rt.cfg.version);
        const char *attn_tag = rt.has_all_attn ? "+mla_attn_all" : (rt.has_l0_attn ? "+l0_mla_attn" : "");
        printf("  forward=embed%s%s%s%s+final_norm+lm_head\n",
               attn_tag,
               rt.has_l0_ffn ? "+l0_dense_ffn" : "",
               rt.has_moe_shared ? "+moe_shared_ffn" : "",
               rt.has_moe_routed ? "+moe_routed" : "");
        printf("  blocks: l0_attn=%s all_attn=%s l0_ffn=%s moe_shared=%s moe_routed=%s native_coverage=%s\n",
               rt.has_l0_attn ? "on" : "off",
               rt.has_all_attn ? "on" : "off",
               rt.has_l0_ffn ? "on" : "off",
               rt.has_moe_shared ? "on" : "off",
               rt.has_moe_routed ? "on" : "off",
               (rt.cfg.l0_flags & FLAG_NATIVE_COVERAGE_FULL) ? "full" : "partial");
        printf("  attn_layers_active=%d\n", rt.has_all_attn ? rt.cfg.n_layers : (rt.has_l0_attn ? 1 : 0));
        printf("  backend=%s\n", backend == BACKEND_METAL ? "metal" : "cpu");
    }

    if (self_test_tokenizer) {
        int pass = run_tokenizer_self_test(&tok, rt.cfg.bos_id);
        tokenizer_free(&tok);
        runtime_close(&rt);
        return pass ? 0 : 2;
    }

    unsigned int rng_seed = (seed >= 0) ? (unsigned int)seed : ((unsigned int)getpid() ^ (unsigned int)n_tokens);
    srand(rng_seed);
    if (debug_mode) {
        fprintf(stderr, "[debug] sampler: temperature=%.6f seed=%u\n", temperature, rng_seed);
    }

    char *prompt_file_buf = NULL;
    if (prompt_path != NULL) {
        prompt_file_buf = read_text_file_alloc(prompt_path);
        if (!prompt_file_buf) {
            fprintf(stderr, "Error: failed to read prompt file: %s\n", prompt_path);
            tokenizer_free(&tok);
            runtime_close(&rt);
            return 1;
        }
        prompt = prompt_file_buf;
    }

    char input[PROMPT_BUFFER_SIZE];
    char rendered_prompt[PROMPT_BUFFER_SIZE];
    char prompt_template[PROMPT_BUFFER_SIZE];
    if (strcmp(mode, "chat") == 0 && strlen(prompt) == 0) {
        printf("chat> ");
        fflush(stdout);
        if (!fgets(input, sizeof(input), stdin)) {
            free(prompt_file_buf);
            tokenizer_free(&tok);
            runtime_close(&rt);
            return 0;
        }
        if (load_template_text(checkpoint, ".template", prompt_template, sizeof(prompt_template))) {
            if (strstr(prompt_template, "{{") != NULL || strstr(prompt_template, "{%") != NULL) {
                snprintf(prompt_template, sizeof(prompt_template), "<|user|>\n%%s\n<|assistant|>\n");
            }
            render_template_user(prompt_template, input, rendered_prompt, sizeof(rendered_prompt));
            prompt = rendered_prompt;
        } else {
            prompt = input;
        }
    }

    if (bench_mode == BENCH_MODE_OFF && strlen(prompt) > 0) {
        printf("%s", prompt);
        if (prompt[strlen(prompt) - 1] != '\n') putchar('\n');
    }

    int *prompt_ids = (int *)calloc((size_t)PROMPT_BUFFER_SIZE, sizeof(int));
    if (!prompt_ids) die("out of memory");
    // Completion prompts in GGUF tools are tokenized without implicit BOS.
    int prompt_bos = (strcmp(mode, "chat") == 0) ? rt.cfg.bos_id : -1;
    int n_prompt = encode_prompt_bpe(&tok, prompt, prompt_ids, PROMPT_BUFFER_SIZE, prompt_bos);
    if (n_prompt <= 0) n_prompt = encode_prompt_bpe(&tok, "", prompt_ids, PROMPT_BUFFER_SIZE, prompt_bos);
    if (n_prompt <= 0) {
        prompt_ids[0] = (rt.cfg.bos_id >= 0 && rt.cfg.bos_id < rt.cfg.vocab_size) ? rt.cfg.bos_id : 0;
        n_prompt = 1;
    }
    if (debug_mode) {
        fprintf(stderr, "[debug] prompt_ids (%d):", n_prompt);
        int show = n_prompt < 12 ? n_prompt : 12;
        for (int i = 0; i < show; i++) fprintf(stderr, " %d", prompt_ids[i]);
        if (n_prompt > show) fprintf(stderr, " ...");
        fprintf(stderr, "\n");
        debug_print_prompt_embd_batch(&rt, prompt_ids, n_prompt);
    }

    int cache_cap = n_prompt + n_tokens + 8;
    if (cache_cap < 1) cache_cap = 1;
    if (cache_cap > rt.cfg.seq_len) cache_cap = rt.cfg.seq_len;
    if (cache_cap > PROMPT_BUFFER_SIZE) cache_cap = PROMPT_BUFFER_SIZE;
    if (context_limit > 0 && cache_cap > context_limit) cache_cap = context_limit;
    if (cache_cap < n_prompt) cache_cap = n_prompt;

    RunState st;
    runstate_build(&st, &rt, cache_cap);
    int use_metal = glm_backend_init(backend, &rt, &st);

    int token = prompt_ids[0];
    if (token < 0 || token >= rt.cfg.vocab_size) token = 0;
    int prompt_index = 1;
    int generated = 0;
    int forward_failed = 0;

    double *prefill_lat_ms = NULL;
    double *decode_lat_ms = NULL;
    int prefill_n = 0, decode_n = 0;
    int prefill_cap = 0, decode_cap = 0;
    double prefill_cache_sum = 0.0;
    double decode_cache_sum = 0.0;
    double model_active_bytes = estimate_model_active_bytes(&rt);

    for (int pos = 0; pos < cache_cap; pos++) {
        if (bench_mode == BENCH_MODE_PREFILL && pos >= n_prompt) break;

        int in_prefill_phase = pos < n_prompt;
        double t0_ms = monotonic_ms();
        int fwd_status = glm_backend_forward_token(use_metal, &rt, &st, token, pos, debug_mode);
        double elapsed_ms = monotonic_ms() - t0_ms;
        if (fwd_status != 0) {
            fprintf(stderr, "[error] forward failed at pos=%d\n", pos);
            forward_failed = 1;
            break;
        }

        if (bench_mode != BENCH_MODE_OFF) {
            if (in_prefill_phase) {
                if (!latency_push(&prefill_lat_ms, &prefill_n, &prefill_cap, elapsed_ms)) die("out of memory");
                prefill_cache_sum += estimate_cache_active_bytes(&rt, &st, pos);
            } else {
                if (!latency_push(&decode_lat_ms, &decode_n, &decode_cap, elapsed_ms)) die("out of memory");
                decode_cache_sum += estimate_cache_active_bytes(&rt, &st, pos);
            }
        }

        if (debug_mode && prompt_index >= n_prompt && generated == 0) {
            print_topk_debug(st.logits, &tok, rt.cfg.vocab_size, 5);
        }

        int next = -1;
        if (prompt_index < n_prompt) {
            next = prompt_ids[prompt_index++];
            if (next < 0 || next >= rt.cfg.vocab_size) next = 0;
        } else {
            next = sample_logits_temperature(st.logits, rt.cfg.vocab_size, temperature);
            if (next == rt.cfg.bos_id && rt.cfg.vocab_size > 1) next = (next + 1) % rt.cfg.vocab_size;
            if (next == rt.cfg.eos_id) {
                if (bench_mode == BENCH_MODE_OFF) break;
                next = (next + 1) % rt.cfg.vocab_size;
            }
            if (bench_mode == BENCH_MODE_OFF) print_token_safe(&tok, &tok.tokens[next]);
            generated++;
            if (generated >= n_tokens) break;
        }
        token = next;
    }

    if (bench_mode == BENCH_MODE_OFF) {
        putchar('\n');
    } else {
        BenchPhaseReport prefill_report = build_phase_report(prefill_lat_ms, prefill_n, model_active_bytes, prefill_cache_sum);
        BenchPhaseReport decode_report = build_phase_report(decode_lat_ms, decode_n, model_active_bytes, decode_cache_sum);
        if (bench_mode == BENCH_MODE_FULL || bench_mode == BENCH_MODE_PREFILL) {
            print_phase_report("prefill", prefill_report);
        }
        if (bench_mode == BENCH_MODE_FULL || bench_mode == BENCH_MODE_DECODE) {
            print_phase_report("decode", decode_report);
        }
        write_bench_report_json(bench_report_path,
                                backend == BACKEND_METAL ? "metal" : "cpu",
                                bench_mode,
                                n_prompt,
                                generated,
                                prefill_report,
                                decode_report);
    }

    free(prefill_lat_ms);
    free(decode_lat_ms);

    glm_backend_free(use_metal);
    free(prompt_ids);
    free(prompt_file_buf);
    runstate_free(&st);
    tokenizer_free(&tok);
    runtime_close(&rt);
    return forward_failed ? 1 : 0;
}
