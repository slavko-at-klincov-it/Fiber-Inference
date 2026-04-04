// ane_attn.h — ANE attention context for Fiber-Inference Phase 2
#ifndef ANE_ATTN_H
#define ANE_ATTN_H

#include <stdbool.h>
#include "model.h"

typedef struct ane_attn_context ane_attn_context_t;

// Initialize ANE subsystem. Returns NULL if ANE unavailable.
ane_attn_context_t *ane_attn_init(void);

// Compile ANE attention kernels for all layers.
// max_prefill_seq: maximum sequence length for prefill (e.g. 512).
// Must be called after model_dequant_attention().
bool ane_attn_compile(ane_attn_context_t *ctx, model_t *m, int max_prefill_seq);

// Run ANE attention for one layer.
// x_in: [dim * seq] FP16 channels-first. out: [out_ch * seq] FP16.
bool ane_attn_eval_layer(ane_attn_context_t *ctx, int layer,
                          const _Float16 *x_in, int dim, int seq,
                          _Float16 *out, int out_ch);

// Quick validation: run layer 0 on test data, print PASS/FAIL
void ane_attn_validate(ane_attn_context_t *ctx, const model_t *m);

// Batched prefill: ANE attention + GPU FFN for all prompt tokens.
// Populates KV cache and leaves logits ready for sampling.
// Returns prefill tok/s.
#include "gpu_ffn.h"
#include "kv_cache.h"
double ane_prefill_batch(ane_attn_context_t *ctx, gpu_context_t *gpu,
                          const model_t *m, kv_cache_t *kv,
                          const int *tokens, int n_tokens);

// Free ANE context and compiled kernels
void ane_attn_free(ane_attn_context_t *ctx);

#endif // ANE_ATTN_H
