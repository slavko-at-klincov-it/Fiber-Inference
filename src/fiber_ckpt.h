// fiber_ckpt.h — Load BLZT checkpoint from ANE-Training
#ifndef FIBER_CKPT_H
#define FIBER_CKPT_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#include "fiber_model.h"

// BLZT checkpoint header (from ANE-Training/training/train_pipeline.m)
typedef struct {
    uint32_t magic;         // 0x424C5A54
    uint32_t version;       // 2
    int step, total_steps;
    int n_layers, vocab_size, dim, hidden_dim, n_heads, seq_len;
    float lr, loss;
    double cum_compile, cum_train, cum_wall;
    int cum_steps, cum_batches, adam_t;
} CkptHdr;

// Load Stories-110M checkpoint into fiber_model_t.
// Adapts dimensions (hidden=2048, MHA 12/12 heads, 12 layers).
// Returns NULL on failure.
fiber_model_t *fiber_model_load_blzt(const char *path);

// Load karpathy/llama2.c format (stories110M.bin, pretrained).
// Header: 7 ints + sequential FP32 weights.
fiber_model_t *fiber_model_load_karpathy(const char *path);

#endif
