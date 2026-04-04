// model.h — Model weight management with mmap and layer prefetch/evict
// Wraps GGUF parser with LLM-specific layer access patterns.

#ifndef MODEL_H
#define MODEL_H

#include "gguf.h"
#include <stdbool.h>

// Model hyperparameters (extracted from GGUF metadata)
typedef struct {
    uint32_t n_layers;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t dim;            // embedding dimension
    uint32_t head_dim;       // dim / n_heads
    uint32_t ffn_dim;        // intermediate/feed-forward dimension
    uint32_t vocab_size;
    uint32_t context_length;
    float    rope_freq_base;
    float    rms_norm_eps;
    const char *arch;        // e.g. "llama", "qwen2", "gemma"
} model_params_t;

// Layer weight pointers (into mmap'd GGUF data)
typedef struct {
    // Attention
    const void *attn_norm;   // RMSNorm weight
    const void *wq;          // Query projection
    const void *wk;          // Key projection
    const void *wv;          // Value projection
    const void *wo;          // Output projection
    // FFN
    const void *ffn_norm;    // RMSNorm weight
    const void *w1;          // Gate projection (up)
    const void *w2;          // Down projection
    const void *w3;          // Up projection
    // Quantization types
    ggml_type_t attn_norm_type;
    ggml_type_t wq_type;
    ggml_type_t wk_type;
    ggml_type_t wv_type;
    ggml_type_t wo_type;
    ggml_type_t ffn_norm_type;
    ggml_type_t w1_type;
    ggml_type_t w2_type;
    ggml_type_t w3_type;
    // Sizes in bytes
    size_t wq_size, wk_size, wv_size, wo_size;
    size_t w1_size, w2_size, w3_size;
    // Dequantized FP16 attention weights (for ANE, allocated by model_dequant_attention)
    _Float16 *wq_fp16;       // [dim, dim]
    _Float16 *wk_fp16;       // [kv_dim, dim]
    _Float16 *wv_fp16;       // [kv_dim, dim]
    _Float16 *wo_fp16;       // [dim, dim]
    _Float16 *attn_norm_fp16; // [dim]
} layer_weights_t;

// Model handle
typedef struct {
    gguf_file_t    *gf;
    model_params_t  params;
    // Global weights
    const void     *token_embd;       // embedding table
    const void     *output_norm;      // final RMSNorm
    const void     *output;           // classifier / lm_head
    ggml_type_t     token_embd_type;
    ggml_type_t     output_norm_type;
    ggml_type_t     output_type;
    // Per-layer dequantized FP16 attention weights (for ANE, Phase 2)
    _Float16      **attn_wq_fp16;     // [n_layers] array of pointers, each [dim * dim]
    _Float16      **attn_wk_fp16;     // [n_layers], each [kv_dim * dim]
    _Float16      **attn_wv_fp16;     // [n_layers], each [kv_dim * dim]
    _Float16      **attn_wo_fp16;     // [n_layers], each [dim * dim]
    _Float16      **attn_norm_fp16;   // [n_layers], each [dim]
    bool            attn_fp16_ready;  // true after model_dequant_attention()
} model_t;

// Open model from GGUF file
model_t *model_open(const char *path);

// Close model
void model_close(model_t *m);

// Get layer weights (pointers into mmap'd data)
bool model_layer_weights(const model_t *m, uint32_t layer_idx, layer_weights_t *lw);

// Prefetch next layer's pages into RAM (call from background thread)
void model_prefetch_layer(const model_t *m, uint32_t layer_idx);

// Evict layer's pages from RAM (tell OS we're done)
void model_evict_layer(const model_t *m, uint32_t layer_idx);

// Dequantize attention weights (Q4_K/Q6_K → FP16) for ANE.
// Returns total bytes allocated. Call model_free_attention_fp16 after ANE compile.
size_t model_dequant_attention(model_t *m);

// Free dequantized FP16 attention weights (after ANE has baked them)
void model_free_attention_fp16(model_t *m);

// Print model info summary
void model_print_info(const model_t *m);

// Get current RSS in bytes
size_t model_get_rss(void);

#endif // MODEL_H
