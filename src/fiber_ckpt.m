// fiber_ckpt.m — Load BLZT checkpoint from ANE-Training
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fiber_ckpt.h"

static _Float16 *f32_to_f16(const float *src, int n) {
    _Float16 *dst = malloc(n * sizeof(_Float16));
    for (int i = 0; i < n; i++) dst[i] = (_Float16)src[i];
    return dst;
}

static float *dup_f32(const float *src, int n) {
    float *dst = malloc(n * sizeof(float));
    memcpy(dst, src, n * sizeof(float));
    return dst;
}

fiber_model_t *fiber_model_load_blzt(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return NULL; }

    CkptHdr h;
    fread(&h, sizeof(h), 1, f);
    if (h.magic != 0x424C5A54) {
        fprintf(stderr, "Not a BLZT checkpoint (magic=0x%X)\n", h.magic);
        fclose(f); return NULL;
    }

    printf("BLZT checkpoint: %s\n", path);
    printf("  Step %d/%d, loss=%.3f\n", h.step, h.total_steps, h.loss);
    printf("  dim=%d, hidden=%d, heads=%d, layers=%d, vocab=%d, seq=%d\n",
           h.dim, h.hidden_dim, h.n_heads, h.n_layers, h.vocab_size, h.seq_len);

    int dim = h.dim, hid = h.hidden_dim, heads = h.n_heads;
    int nl = h.n_layers, vocab = h.vocab_size;
    int wq_sz = dim * dim;  // MHA: Q/K/V/O all dim×dim
    int wo_sz = dim * dim;
    int w1_sz = hid * dim;
    int w2_sz = dim * hid;

    fiber_model_t *fm = calloc(1, sizeof(*fm));
    size_t total = 0;

    // Temporary FP32 buffers for reading
    float *tmp_big = malloc((size_t)(w1_sz > wq_sz ? w1_sz : wq_sz) * sizeof(float));

    for (int l = 0; l < nl && l < FIBER_LAYERS; l++) {
        // Read Wq
        fread(tmp_big, 4, wq_sz, f);
        fm->wq[l] = f32_to_f16(tmp_big, wq_sz);
        fm->wq_f32[l] = dup_f32(tmp_big, wq_sz);
        // Wk
        fread(tmp_big, 4, wq_sz, f);
        fm->wk[l] = f32_to_f16(tmp_big, wq_sz);
        fm->wk_f32[l] = dup_f32(tmp_big, wq_sz);
        // Wv
        fread(tmp_big, 4, wq_sz, f);
        fm->wv[l] = f32_to_f16(tmp_big, wq_sz);
        fm->wv_f32[l] = dup_f32(tmp_big, wq_sz);
        // Wo
        fread(tmp_big, 4, wo_sz, f);
        fm->wo[l] = f32_to_f16(tmp_big, wo_sz);
        fm->wo_f32[l] = dup_f32(tmp_big, wo_sz);
        // W1 (gate)
        fread(tmp_big, 4, w1_sz, f);
        fm->w1[l] = f32_to_f16(tmp_big, w1_sz);
        fm->w1_f32[l] = dup_f32(tmp_big, w1_sz);
        // W2 (down)
        fread(tmp_big, 4, w2_sz, f);
        fm->w2[l] = f32_to_f16(tmp_big, w2_sz);
        fm->w2_f32[l] = dup_f32(tmp_big, w2_sz);
        // W3 (up)
        fread(tmp_big, 4, w1_sz, f);
        fm->w3[l] = f32_to_f16(tmp_big, w1_sz);
        fm->w3_f32[l] = dup_f32(tmp_big, w1_sz);
        // attn_norm
        float *norm_tmp = malloc(dim * sizeof(float));
        fread(norm_tmp, 4, dim, f);
        fm->attn_norm[l] = f32_to_f16(norm_tmp, dim);
        fm->attn_norm_f32[l] = norm_tmp;
        // ffn_norm
        float *fnorm_tmp = malloc(dim * sizeof(float));
        fread(fnorm_tmp, 4, dim, f);
        fm->ffn_norm[l] = f32_to_f16(fnorm_tmp, dim);
        fm->ffn_norm_f32[l] = fnorm_tmp;

        // Skip Adam states (m and v for each weight)
        long adam_skip = (long)(wq_sz*2 + wq_sz*2 + wq_sz*2 + wo_sz*2 +
                                w1_sz*2 + w2_sz*2 + w1_sz*2 + dim*2 + dim*2) * 4;
        fseek(f, adam_skip, SEEK_CUR);

        total += (wq_sz*4 + w1_sz*2 + w2_sz + dim*2) * (sizeof(_Float16) + sizeof(float));
    }

    // rms_final
    float *rms_final = malloc(dim * sizeof(float));
    fread(rms_final, 4, dim, f);
    fm->output_norm = f32_to_f16(rms_final, dim);
    free(rms_final);

    // Skip Adam for rms_final
    fseek(f, (long)dim * 2 * 4, SEEK_CUR);

    // embedding (also used as output/classifier in weight-tied models)
    float *embed = malloc((size_t)vocab * dim * sizeof(float));
    fread(embed, 4, vocab * dim, f);
    fm->embedding = f32_to_f16(embed, vocab * dim);
    fm->output = f32_to_f16(embed, vocab * dim); // weight tying
    free(embed);

    fclose(f);
    free(tmp_big);

    fm->total_bytes = total;
    printf("Loaded %d layers, %.1f MB\n", nl, total / (1024.0 * 1024.0));
    return fm;
}
