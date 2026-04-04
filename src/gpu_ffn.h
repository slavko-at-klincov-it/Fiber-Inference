// gpu_ffn.h — GPU inference pipeline (Metal)

#ifndef GPU_FFN_H
#define GPU_FFN_H

#include "model.h"
#include "kv_cache.h"

typedef struct gpu_context gpu_context_t;

gpu_context_t *gpu_init(const model_t *m);
void gpu_free(gpu_context_t *ctx);

// Embed single token → internal buf_x (FP16)
void gpu_embed(gpu_context_t *ctx, const model_t *m, int token_id);

// Forward: all layers + classifier in one GPU submission. Returns ms.
double gpu_forward_token(gpu_context_t *ctx, const model_t *m,
                         kv_cache_t *kv, int pos);

// F32 logits (vocab_size elements)
const float *gpu_get_logits(gpu_context_t *ctx);

// Direct access to internal FP16 hidden state buffer (dim elements)
_Float16 *gpu_get_buf_x(gpu_context_t *ctx);

// FFN-only forward for one layer: reads buf_x, writes back to buf_x.
// Used by ANE prefill path (ANE does attention, GPU does FFN).
void gpu_forward_ffn_layer(gpu_context_t *ctx, const model_t *m,
                            uint32_t layer_idx);

// Classifier-only: RMSNorm + output projection on current buf_x → logits.
void gpu_forward_classifier(gpu_context_t *ctx, const model_t *m);

#ifdef __OBJC__
id<MTLDevice> gpu_get_device(gpu_context_t *ctx);
#endif

#endif
