// model.m — Model weight management implementation

#import <Foundation/Foundation.h>
#import <mach/mach.h>
#import <sys/mman.h>
#include "model.h"
#include "dequant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// --- Tensor name patterns for common architectures ---

// Most LLMs use "model.layers.N.WEIGHT_NAME" or "blk.N.WEIGHT_NAME"
// GGUF standardizes to "blk.N.*" naming convention

static const void *find_layer_tensor(const model_t *m, uint32_t layer_idx,
                                     const char *suffix, ggml_type_t *out_type,
                                     size_t *out_size) {
    char name[256];
    snprintf(name, sizeof(name), "blk.%u.%s", layer_idx, suffix);

    const gguf_tensor_info_t *ti = gguf_find_tensor(m->gf, name);
    if (!ti) return NULL;

    if (out_type) *out_type = ti->type;
    if (out_size) *out_size = ti->size_bytes;
    return gguf_tensor_data(m->gf, ti);
}

static const void *find_global_tensor(const gguf_file_t *gf, const char *name,
                                      ggml_type_t *out_type) {
    const gguf_tensor_info_t *ti = gguf_find_tensor(gf, name);
    if (!ti) return NULL;
    if (out_type) *out_type = ti->type;
    return gguf_tensor_data(gf, ti);
}

// --- Extract architecture-specific hyperparameters ---

static bool extract_params(model_t *m) {
    gguf_file_t *gf = m->gf;
    model_params_t *p = &m->params;

    // Architecture name (e.g., "llama", "qwen2", "gemma")
    p->arch = gguf_get_str(gf, "general.architecture");
    if (!p->arch) {
        fprintf(stderr, "model: missing general.architecture\n");
        return false;
    }

    // Build arch-prefixed key names
    char key[256];

    snprintf(key, sizeof(key), "%s.block_count", p->arch);
    p->n_layers = gguf_get_u32(gf, key, 0);

    snprintf(key, sizeof(key), "%s.attention.head_count", p->arch);
    p->n_heads = gguf_get_u32(gf, key, 0);

    snprintf(key, sizeof(key), "%s.attention.head_count_kv", p->arch);
    p->n_kv_heads = gguf_get_u32(gf, key, p->n_heads);  // default: MHA

    snprintf(key, sizeof(key), "%s.embedding_length", p->arch);
    p->dim = gguf_get_u32(gf, key, 0);

    snprintf(key, sizeof(key), "%s.feed_forward_length", p->arch);
    p->ffn_dim = gguf_get_u32(gf, key, 0);

    snprintf(key, sizeof(key), "%s.context_length", p->arch);
    p->context_length = gguf_get_u32(gf, key, 2048);

    snprintf(key, sizeof(key), "%s.rope.freq_base", p->arch);
    p->rope_freq_base = gguf_get_f32(gf, key, 10000.0f);

    snprintf(key, sizeof(key), "%s.attention.layer_norm_rms_epsilon", p->arch);
    p->rms_norm_eps = gguf_get_f32(gf, key, 1e-5f);

    // Vocab size from tokenizer metadata
    const gguf_kv_t *vocab_kv = gguf_find_kv(gf, "tokenizer.ggml.tokens");
    if (vocab_kv && vocab_kv->type == GGUF_TYPE_ARRAY) {
        p->vocab_size = (uint32_t)vocab_kv->value.arr.count;
    } else {
        // Fallback: infer from embedding tensor
        const gguf_tensor_info_t *embd = gguf_find_tensor(gf, "token_embd.weight");
        if (embd && embd->n_dims >= 2) {
            p->vocab_size = (uint32_t)embd->dims[1];
        }
    }

    // Derived
    p->head_dim = p->n_heads > 0 ? p->dim / p->n_heads : 0;

    // Validate
    if (p->n_layers == 0 || p->dim == 0 || p->n_heads == 0) {
        fprintf(stderr, "model: incomplete hyperparameters (layers=%u, dim=%u, heads=%u)\n",
                p->n_layers, p->dim, p->n_heads);
        return false;
    }

    return true;
}

// --- Public API ---

model_t *model_open(const char *path) {
    gguf_file_t *gf = gguf_open(path);
    if (!gf) return NULL;

    model_t *m = calloc(1, sizeof(model_t));
    m->gf = gf;

    if (!extract_params(m)) {
        model_close(m);
        return NULL;
    }

    // Locate global tensors
    m->token_embd  = find_global_tensor(gf, "token_embd.weight", &m->token_embd_type);
    m->output_norm = find_global_tensor(gf, "output_norm.weight", &m->output_norm_type);
    m->output      = find_global_tensor(gf, "output.weight", &m->output_type);

    // Some models share embedding and output weights
    if (!m->output) {
        m->output = m->token_embd;
        m->output_type = m->token_embd_type;
    }

    if (!m->token_embd) {
        fprintf(stderr, "model: missing token_embd.weight\n");
        model_close(m);
        return NULL;
    }

    return m;
}

void model_close(model_t *m) {
    if (!m) return;
    gguf_close(m->gf);
    free(m);
}

bool model_layer_weights(const model_t *m, uint32_t layer_idx, layer_weights_t *lw) {
    memset(lw, 0, sizeof(*lw));

    lw->attn_norm = find_layer_tensor(m, layer_idx, "attn_norm.weight",
                                      &lw->attn_norm_type, NULL);
    lw->wq = find_layer_tensor(m, layer_idx, "attn_q.weight",
                               &lw->wq_type, &lw->wq_size);
    lw->wk = find_layer_tensor(m, layer_idx, "attn_k.weight",
                               &lw->wk_type, &lw->wk_size);
    lw->wv = find_layer_tensor(m, layer_idx, "attn_v.weight",
                               &lw->wv_type, &lw->wv_size);
    lw->wo = find_layer_tensor(m, layer_idx, "attn_output.weight",
                               &lw->wo_type, &lw->wo_size);

    lw->ffn_norm = find_layer_tensor(m, layer_idx, "ffn_norm.weight",
                                     &lw->ffn_norm_type, NULL);
    lw->w1 = find_layer_tensor(m, layer_idx, "ffn_gate.weight",
                               &lw->w1_type, &lw->w1_size);
    lw->w2 = find_layer_tensor(m, layer_idx, "ffn_down.weight",
                               &lw->w2_type, &lw->w2_size);
    lw->w3 = find_layer_tensor(m, layer_idx, "ffn_up.weight",
                               &lw->w3_type, &lw->w3_size);

    // Fill FP16 pointers if dequantized weights are available
    if (m->attn_fp16_ready && layer_idx < m->params.n_layers) {
        lw->wq_fp16       = m->attn_wq_fp16[layer_idx];
        lw->wk_fp16       = m->attn_wk_fp16[layer_idx];
        lw->wv_fp16       = m->attn_wv_fp16[layer_idx];
        lw->wo_fp16       = m->attn_wo_fp16[layer_idx];
        lw->attn_norm_fp16 = m->attn_norm_fp16[layer_idx];
    }

    return lw->wq != NULL && lw->wk != NULL && lw->wv != NULL && lw->wo != NULL;
}

void model_prefetch_layer(const model_t *m, uint32_t layer_idx) {
    layer_weights_t lw;
    if (!model_layer_weights(m, layer_idx, &lw)) return;

    // Prefetch all weight pages for this layer
    const void *ptrs[] = { lw.wq, lw.wk, lw.wv, lw.wo, lw.w1, lw.w2, lw.w3,
                           lw.attn_norm, lw.ffn_norm };
    size_t sizes[] = { lw.wq_size, lw.wk_size, lw.wv_size, lw.wo_size,
                       lw.w1_size, lw.w2_size, lw.w3_size, 0, 0 };

    for (int i = 0; i < 9; i++) {
        if (ptrs[i] && sizes[i] > 0) {
            madvise((void *)ptrs[i], sizes[i], MADV_WILLNEED);
        }
    }
}

void model_evict_layer(const model_t *m, uint32_t layer_idx) {
    layer_weights_t lw;
    if (!model_layer_weights(m, layer_idx, &lw)) return;

    const void *ptrs[] = { lw.wq, lw.wk, lw.wv, lw.wo, lw.w1, lw.w2, lw.w3 };
    size_t sizes[] = { lw.wq_size, lw.wk_size, lw.wv_size, lw.wo_size,
                       lw.w1_size, lw.w2_size, lw.w3_size };

    for (int i = 0; i < 7; i++) {
        if (ptrs[i] && sizes[i] > 0) {
            madvise((void *)ptrs[i], sizes[i], MADV_DONTNEED);
        }
    }
}

// Dequantize a weight matrix from Q4_K or Q6_K to FP16
// Returns allocated FP16 buffer [out_rows * in_cols]
static _Float16 *dequant_weight_fp16(const void *data, ggml_type_t type,
                                     int out_rows, int in_cols) {
    _Float16 *buf = malloc((size_t)out_rows * in_cols * sizeof(_Float16));
    if (!buf) return NULL;
    for (int r = 0; r < out_rows; r++) {
        if (type == GGML_TYPE_Q4_K) {
            dequant_q4k_row_f16(data, r, buf + (size_t)r * in_cols, in_cols);
        } else if (type == GGML_TYPE_Q6_K) {
            dequant_q6k_row_f16(data, r, buf + (size_t)r * in_cols, in_cols);
        } else {
            fprintf(stderr, "model: unsupported quant type %d for dequant\n", type);
            free(buf);
            return NULL;
        }
    }
    return buf;
}

// Dequantize RMSNorm weight (F32 → FP16)
static _Float16 *dequant_norm_fp16(const void *data, int dim) {
    _Float16 *buf = malloc((size_t)dim * sizeof(_Float16));
    if (!buf) return NULL;
    const float *src = (const float *)data;
    for (int i = 0; i < dim; i++) {
        buf[i] = (_Float16)src[i];
    }
    return buf;
}

size_t model_dequant_attention(model_t *m) {
    uint32_t nl = m->params.n_layers;
    uint32_t dim = m->params.dim;
    uint32_t kv_dim = m->params.n_kv_heads * m->params.head_dim;
    size_t total = 0;

    m->attn_wq_fp16   = calloc(nl, sizeof(_Float16 *));
    m->attn_wk_fp16   = calloc(nl, sizeof(_Float16 *));
    m->attn_wv_fp16   = calloc(nl, sizeof(_Float16 *));
    m->attn_wo_fp16   = calloc(nl, sizeof(_Float16 *));
    m->attn_norm_fp16 = calloc(nl, sizeof(_Float16 *));

    for (uint32_t l = 0; l < nl; l++) {
        layer_weights_t lw;
        if (!model_layer_weights(m, l, &lw)) {
            fprintf(stderr, "model: failed to get layer %u weights\n", l);
            continue;
        }

        m->attn_wq_fp16[l] = dequant_weight_fp16(lw.wq, lw.wq_type, dim, dim);
        m->attn_wk_fp16[l] = dequant_weight_fp16(lw.wk, lw.wk_type, kv_dim, dim);
        m->attn_wv_fp16[l] = dequant_weight_fp16(lw.wv, lw.wv_type, kv_dim, dim);
        m->attn_wo_fp16[l] = dequant_weight_fp16(lw.wo, lw.wo_type, dim, dim);
        m->attn_norm_fp16[l] = dequant_norm_fp16(lw.attn_norm, dim);

        total += (size_t)dim * dim * 2;           // Wq
        total += (size_t)kv_dim * dim * 2;         // Wk
        total += (size_t)kv_dim * dim * 2;         // Wv
        total += (size_t)dim * dim * 2;            // Wo
        total += (size_t)dim * 2;                  // norm
    }

    m->attn_fp16_ready = true;
    return total;
}

void model_free_attention_fp16(model_t *m) {
    if (!m->attn_fp16_ready) return;
    uint32_t nl = m->params.n_layers;

    for (uint32_t l = 0; l < nl; l++) {
        free(m->attn_wq_fp16[l]);
        free(m->attn_wk_fp16[l]);
        free(m->attn_wv_fp16[l]);
        free(m->attn_wo_fp16[l]);
        free(m->attn_norm_fp16[l]);
    }
    free(m->attn_wq_fp16);  m->attn_wq_fp16 = NULL;
    free(m->attn_wk_fp16);  m->attn_wk_fp16 = NULL;
    free(m->attn_wv_fp16);  m->attn_wv_fp16 = NULL;
    free(m->attn_wo_fp16);  m->attn_wo_fp16 = NULL;
    free(m->attn_norm_fp16); m->attn_norm_fp16 = NULL;
    m->attn_fp16_ready = false;
}

void model_print_info(const model_t *m) {
    const model_params_t *p = &m->params;
    size_t rss = model_get_rss();

    printf("=== Fiber-Inference Model Info ===\n");
    printf("Architecture:    %s\n", p->arch);
    printf("Layers:          %u\n", p->n_layers);
    printf("Dimension:       %u\n", p->dim);
    printf("Heads:           %u (KV: %u)\n", p->n_heads, p->n_kv_heads);
    printf("Head dim:        %u\n", p->head_dim);
    printf("FFN dim:         %u\n", p->ffn_dim);
    printf("Vocab size:      %u\n", p->vocab_size);
    printf("Context length:  %u\n", p->context_length);
    printf("RoPE freq base:  %.1f\n", p->rope_freq_base);
    printf("RMS norm eps:    %e\n", p->rms_norm_eps);
    printf("File size:       %.2f GB\n", m->gf->file_size / (1024.0 * 1024.0 * 1024.0));
    printf("Tensors:         %llu\n", m->gf->n_tensors);
    printf("Data offset:     0x%zx\n", m->gf->data_offset);
    printf("Current RSS:     %.1f MB\n", rss / (1024.0 * 1024.0));

    // Print embedding type
    printf("Embedding type:  %s\n", ggml_type_name(m->token_embd_type));

    // Print first layer weight types
    layer_weights_t lw;
    if (model_layer_weights(m, 0, &lw)) {
        printf("Layer 0 types:   Q=%s K=%s V=%s O=%s\n",
               ggml_type_name(lw.wq_type), ggml_type_name(lw.wk_type),
               ggml_type_name(lw.wv_type), ggml_type_name(lw.wo_type));
        printf("                 W1=%s W2=%s W3=%s\n",
               ggml_type_name(lw.w1_type), ggml_type_name(lw.w2_type),
               ggml_type_name(lw.w3_type));
    }
    printf("=================================\n");
}

size_t model_get_rss(void) {
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    kern_return_t ret = task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                                 (task_info_t)&info, &count);
    if (ret != KERN_SUCCESS) return 0;
    return info.resident_size;
}
