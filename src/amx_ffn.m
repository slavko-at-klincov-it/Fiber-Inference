// amx_ffn.m — AMX FFN pipeline using Accelerate/cblas
#import <Foundation/Foundation.h>
#include <Accelerate/Accelerate.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "amx_ffn.h"

void amx_forward_ffn_batch(_Float16 *x, int dim, int ffn_dim, int seq,
                            const _Float16 *w1, const _Float16 *w3,
                            const _Float16 *w2, const _Float16 *ffn_norm,
                            float eps) {
    size_t ds = (size_t)dim * seq;
    size_t fs = (size_t)ffn_dim * seq;
    size_t wd_hd = (size_t)ffn_dim * dim;
    size_t wd_dh = (size_t)dim * ffn_dim;

    // Convert x from channels-first FP16 [dim, seq] to row-major FP32 [dim, seq]
    // (channels-first IS row-major with dim rows and seq columns)
    float *xf = malloc(ds * sizeof(float));
    for (size_t i = 0; i < ds; i++) xf[i] = (float)x[i];

    // RMSNorm per token (column-wise in [dim, seq] layout)
    float *xnorm = malloc(ds * sizeof(float));
    for (int t = 0; t < seq; t++) {
        float ss = 0;
        for (int d = 0; d < dim; d++) {
            float v = xf[(size_t)d * seq + t];
            ss += v * v;
        }
        float rrms = 1.0f / sqrtf(ss / dim + eps);
        for (int d = 0; d < dim; d++) {
            xnorm[(size_t)d * seq + t] = xf[(size_t)d * seq + t] * rrms * (float)ffn_norm[d];
        }
    }

    // Convert weights FP16 → FP32
    float *w1f = malloc(wd_hd * sizeof(float));
    float *w3f = malloc(wd_hd * sizeof(float));
    float *w2f = malloc(wd_dh * sizeof(float));
    for (size_t i = 0; i < wd_hd; i++) w1f[i] = (float)w1[i];
    for (size_t i = 0; i < wd_hd; i++) w3f[i] = (float)w3[i];
    for (size_t i = 0; i < wd_dh; i++) w2f[i] = (float)w2[i];

    // h1 = W1 @ xnorm: [ffn, dim] × [dim, seq] → [ffn, seq]
    float *h1 = malloc(fs * sizeof(float));
    float *h3 = malloc(fs * sizeof(float));
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                ffn_dim, seq, dim, 1.0f, w1f, dim, xnorm, seq, 0.0f, h1, seq);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                ffn_dim, seq, dim, 1.0f, w3f, dim, xnorm, seq, 0.0f, h3, seq);
    free(w1f); free(w3f); free(xnorm);

    // SiLU gate: silu = (h1 / (1 + exp(-h1))) * h3
    for (size_t i = 0; i < fs; i++) {
        float s = h1[i] / (1.0f + expf(-h1[i]));
        h1[i] = s * h3[i];
    }
    free(h3);

    // ffn_out = W2 @ silu: [dim, ffn] × [ffn, seq] → [dim, seq]
    float *ffn_out = malloc(ds * sizeof(float));
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                dim, seq, ffn_dim, 1.0f, w2f, ffn_dim, h1, seq, 0.0f, ffn_out, seq);
    free(w2f); free(h1);

    // Residual add + convert back to FP16
    for (size_t i = 0; i < ds; i++) {
        x[i] = (_Float16)(xf[i] + ffn_out[i]);
    }
    free(xf); free(ffn_out);
}

// FP32 weights version — skips per-call FP16→FP32 conversion
void amx_forward_ffn_batch_f32(_Float16 *x, int dim, int ffn_dim, int seq,
                                const float *w1, const float *w3,
                                const float *w2, const float *ffn_norm,
                                float eps) {
    size_t ds = (size_t)dim * seq;
    size_t fs = (size_t)ffn_dim * seq;

    // Convert x FP16 → FP32
    float *xf = malloc(ds * sizeof(float));
    for (size_t i = 0; i < ds; i++) xf[i] = (float)x[i];

    // RMSNorm per token
    float *xnorm = malloc(ds * sizeof(float));
    for (int t = 0; t < seq; t++) {
        float ss = 0;
        for (int d = 0; d < dim; d++) {
            float v = xf[(size_t)d * seq + t];
            ss += v * v;
        }
        float rrms = 1.0f / sqrtf(ss / dim + eps);
        for (int d = 0; d < dim; d++) {
            xnorm[(size_t)d * seq + t] = xf[(size_t)d * seq + t] * rrms * ffn_norm[d];
        }
    }

    // Matmuls — weights already FP32, no conversion needed
    float *h1 = malloc(fs * sizeof(float));
    float *h3 = malloc(fs * sizeof(float));
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                ffn_dim, seq, dim, 1.0f, w1, dim, xnorm, seq, 0.0f, h1, seq);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                ffn_dim, seq, dim, 1.0f, w3, dim, xnorm, seq, 0.0f, h3, seq);
    free(xnorm);

    // SiLU gate
    for (size_t i = 0; i < fs; i++) {
        float s = h1[i] / (1.0f + expf(-h1[i]));
        h1[i] = s * h3[i];
    }
    free(h3);

    // W2 matmul
    float *ffn_out = malloc(ds * sizeof(float));
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                dim, seq, ffn_dim, 1.0f, w2, ffn_dim, h1, seq, 0.0f, ffn_out, seq);
    free(h1);

    // Residual + FP16
    for (size_t i = 0; i < ds; i++) {
        x[i] = (_Float16)(xf[i] + ffn_out[i]);
    }
    free(xf); free(ffn_out);
}
