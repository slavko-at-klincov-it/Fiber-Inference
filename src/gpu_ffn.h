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

#ifdef __OBJC__
id<MTLDevice> gpu_get_device(gpu_context_t *ctx);
#endif

#endif
