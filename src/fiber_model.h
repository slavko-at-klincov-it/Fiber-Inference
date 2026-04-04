// fiber_model.h — Synthetic Fiber-768 model (random FP16 weights)
// For benchmarking the architecture without a trained model
#ifndef FIBER_MODEL_H
#define FIBER_MODEL_H

#include <stdbool.h>
#include <stddef.h>
#include "fiber_arch.h"

typedef struct {
    // Per-layer weights (FP16, [out, in] row-major)
    _Float16 *wq[FIBER_LAYERS];        // [dim, dim]
    _Float16 *wk[FIBER_LAYERS];        // [kv_dim, dim]
    _Float16 *wv[FIBER_LAYERS];        // [kv_dim, dim]
    _Float16 *wo[FIBER_LAYERS];        // [dim, dim]
    _Float16 *w1[FIBER_LAYERS];        // [ffn_dim, dim]
    _Float16 *w3[FIBER_LAYERS];        // [ffn_dim, dim]
    _Float16 *w2[FIBER_LAYERS];        // [dim, ffn_dim]
    _Float16 *attn_norm[FIBER_LAYERS];  // [dim]
    _Float16 *ffn_norm[FIBER_LAYERS];   // [dim]
    // Pre-converted FP32 FFN weights for AMX (cblas_sgemm)
    float *w1_f32[FIBER_LAYERS];        // [ffn_dim, dim]
    float *w3_f32[FIBER_LAYERS];        // [ffn_dim, dim]
    float *w2_f32[FIBER_LAYERS];        // [dim, ffn_dim]
    float *ffn_norm_f32[FIBER_LAYERS];  // [dim]
    // Global
    _Float16 *embedding;    // [vocab, dim]
    _Float16 *output_norm;  // [dim]
    _Float16 *output;       // [vocab, dim] (classifier)
    size_t total_bytes;
} fiber_model_t;

// Create synthetic model with random FP16 weights
fiber_model_t *fiber_model_create(void);

// Free all weights
void fiber_model_free(fiber_model_t *fm);

#endif
