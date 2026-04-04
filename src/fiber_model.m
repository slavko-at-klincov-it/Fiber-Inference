// fiber_model.m — Synthetic Fiber-768 model generator
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "fiber_model.h"

// Xavier-like random init scaled for FP16
static _Float16 *rand_fp16(int rows, int cols) {
    size_t n = (size_t)rows * cols;
    _Float16 *buf = malloc(n * sizeof(_Float16));
    float scale = sqrtf(2.0f / (float)(rows + cols));
    for (size_t i = 0; i < n; i++) {
        float r = ((float)(rand() % 10000) / 5000.0f - 1.0f) * scale;
        buf[i] = (_Float16)r;
    }
    return buf;
}

// Ones init for norm weights
static _Float16 *ones_fp16(int n) {
    _Float16 *buf = malloc(n * sizeof(_Float16));
    for (int i = 0; i < n; i++) buf[i] = (_Float16)1.0f;
    return buf;
}

fiber_model_t *fiber_model_create(void) {
    srand(42);
    fiber_model_t *fm = calloc(1, sizeof(*fm));
    size_t total = 0;

    int dim = FIBER_DIM, kv_dim = FIBER_KV_DIM, ffn = FIBER_FFN_DIM, vocab = FIBER_VOCAB;

    for (int l = 0; l < FIBER_LAYERS; l++) {
        fm->wq[l] = rand_fp16(dim, dim);         total += dim * dim * 2;
        fm->wk[l] = rand_fp16(kv_dim, dim);      total += kv_dim * dim * 2;
        fm->wv[l] = rand_fp16(kv_dim, dim);      total += kv_dim * dim * 2;
        fm->wo[l] = rand_fp16(dim, dim);          total += dim * dim * 2;
        fm->w1[l] = rand_fp16(ffn, dim);          total += ffn * dim * 2;
        fm->w3[l] = rand_fp16(ffn, dim);          total += ffn * dim * 2;
        fm->w2[l] = rand_fp16(dim, ffn);          total += dim * ffn * 2;
        fm->attn_norm[l] = ones_fp16(dim);        total += dim * 2;
        fm->ffn_norm[l]  = ones_fp16(dim);        total += dim * 2;

        // FP32 versions of ALL weights (for ane_weight_fp16 and cblas)
        int dd = dim * dim, kd = kv_dim * dim, fd = ffn * dim, df = dim * ffn;
        fm->wq_f32[l] = malloc(dd * sizeof(float));
        fm->wk_f32[l] = malloc(kd * sizeof(float));
        fm->wv_f32[l] = malloc(kd * sizeof(float));
        fm->wo_f32[l] = malloc(dd * sizeof(float));
        fm->w1_f32[l] = malloc(fd * sizeof(float));
        fm->w3_f32[l] = malloc(fd * sizeof(float));
        fm->w2_f32[l] = malloc(df * sizeof(float));
        fm->attn_norm_f32[l] = malloc(dim * sizeof(float));
        fm->ffn_norm_f32[l] = malloc(dim * sizeof(float));
        for (int i = 0; i < dd; i++) fm->wq_f32[l][i] = (float)fm->wq[l][i];
        for (int i = 0; i < kd; i++) fm->wk_f32[l][i] = (float)fm->wk[l][i];
        for (int i = 0; i < kd; i++) fm->wv_f32[l][i] = (float)fm->wv[l][i];
        for (int i = 0; i < dd; i++) fm->wo_f32[l][i] = (float)fm->wo[l][i];
        for (int i = 0; i < fd; i++) fm->w1_f32[l][i] = (float)fm->w1[l][i];
        for (int i = 0; i < fd; i++) fm->w3_f32[l][i] = (float)fm->w3[l][i];
        for (int i = 0; i < df; i++) fm->w2_f32[l][i] = (float)fm->w2[l][i];
        for (int i = 0; i < dim; i++) fm->attn_norm_f32[l][i] = 1.0f;
        for (int i = 0; i < dim; i++) fm->ffn_norm_f32[l][i] = 1.0f;
        total += (dd*2 + kd*2 + fd*2 + df + dim*2) * sizeof(float);
    }

    fm->embedding   = rand_fp16(vocab, dim);      total += vocab * dim * 2;
    fm->output_norm = ones_fp16(dim);              total += dim * 2;
    fm->output      = rand_fp16(vocab, dim);       total += vocab * dim * 2;
    fm->total_bytes = total;

    printf("Fiber-768 model: %d layers, %.1f MB weights\n",
           FIBER_LAYERS, total / (1024.0 * 1024.0));
    return fm;
}

void fiber_model_free(fiber_model_t *fm) {
    if (!fm) return;
    for (int l = 0; l < FIBER_LAYERS; l++) {
        free(fm->wq[l]); free(fm->wk[l]); free(fm->wv[l]); free(fm->wo[l]);
        free(fm->w1[l]); free(fm->w3[l]); free(fm->w2[l]);
        free(fm->attn_norm[l]); free(fm->ffn_norm[l]);
        free(fm->wq_f32[l]); free(fm->wk_f32[l]); free(fm->wv_f32[l]); free(fm->wo_f32[l]);
        free(fm->w1_f32[l]); free(fm->w3_f32[l]); free(fm->w2_f32[l]);
        free(fm->attn_norm_f32[l]); free(fm->ffn_norm_f32[l]);
    }
    free(fm->embedding); free(fm->output_norm); free(fm->output);
    free(fm);
}
