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

// ============================================================
// karpathy/llama2.c format loader (stories110M.bin)
// ============================================================

fiber_model_t *fiber_model_load_karpathy(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); return NULL; }

    // Header: 7 ints
    int config[7];
    fread(config, sizeof(int), 7, f);
    int dim = config[0], hidden = config[1], nl = config[2];
    int heads = config[3], kv_heads = config[4], vocab = config[5], seq = config[6];

    // kv_heads=0 means MHA (same as heads)
    if (kv_heads == 0) kv_heads = heads;
    // negative vocab means no weight tying
    int shared_weights = (vocab > 0);
    if (vocab < 0) vocab = -vocab;

    printf("karpathy model: %s\n", path);
    printf("  dim=%d, hidden=%d, heads=%d, kv=%d, layers=%d, vocab=%d, seq=%d\n",
           dim, hidden, heads, kv_heads, nl, vocab, seq);

    fiber_model_t *fm = calloc(1, sizeof(*fm));
    size_t total = 0;
    int dd = dim * dim, hd = hidden * dim, dh = dim * hidden;

    // 1. Token embedding [vocab, dim]
    float *embed = malloc((size_t)vocab * dim * sizeof(float));
    fread(embed, 4, (size_t)vocab * dim, f);
    fm->embedding = f32_to_f16(embed, vocab * dim);
    total += vocab * dim * 2;

    // 2. Per-layer weights (in karpathy order: rms_att for ALL layers first, then wq, etc.)
    // WAIT — karpathy format stores ALL rms_att first, then ALL wq, etc.?
    // Let me check: the code in ANE-Training reads per-layer sequentially.
    // Actually, looking at the llama2.c code: weights are stored as:
    //   rms_att_weight[layer][dim]  — ALL layers
    //   wq[layer][dim*dim]          — ALL layers
    //   wk[layer][dim*dim]          — ALL layers
    //   wv[layer][dim*dim]          — ALL layers
    //   wo[layer][dim*dim]          — ALL layers
    //   rms_ffn_weight[layer][dim]  — ALL layers
    //   w1[layer][hidden*dim]       — ALL layers
    //   w2[layer][dim*hidden]       — ALL layers
    //   w3[layer][hidden*dim]       — ALL layers
    //   rms_final[dim]

    // Read all rms_att
    for (int l = 0; l < nl && l < FIBER_LAYERS; l++) {
        float *tmp = malloc(dim * sizeof(float));
        fread(tmp, 4, dim, f);
        fm->attn_norm[l] = f32_to_f16(tmp, dim);
        fm->attn_norm_f32[l] = tmp;
    }
    // Read all wq
    for (int l = 0; l < nl && l < FIBER_LAYERS; l++) {
        float *tmp = malloc(dd * sizeof(float));
        fread(tmp, 4, dd, f);
        fm->wq[l] = f32_to_f16(tmp, dd);
        fm->wq_f32[l] = tmp;
        total += dd * (2 + 4);
    }
    // Read all wk
    int kv_dim = kv_heads * (dim / heads);
    int kd = kv_dim * dim;
    for (int l = 0; l < nl && l < FIBER_LAYERS; l++) {
        float *tmp = malloc(kd * sizeof(float));
        fread(tmp, 4, kd, f);
        fm->wk[l] = f32_to_f16(tmp, kd);
        fm->wk_f32[l] = tmp;
    }
    // Read all wv
    for (int l = 0; l < nl && l < FIBER_LAYERS; l++) {
        float *tmp = malloc(kd * sizeof(float));
        fread(tmp, 4, kd, f);
        fm->wv[l] = f32_to_f16(tmp, kd);
        fm->wv_f32[l] = tmp;
    }
    // Read all wo
    for (int l = 0; l < nl && l < FIBER_LAYERS; l++) {
        float *tmp = malloc(dd * sizeof(float));
        fread(tmp, 4, dd, f);
        fm->wo[l] = f32_to_f16(tmp, dd);
        fm->wo_f32[l] = tmp;
    }
    // Read all rms_ffn
    for (int l = 0; l < nl && l < FIBER_LAYERS; l++) {
        float *tmp = malloc(dim * sizeof(float));
        fread(tmp, 4, dim, f);
        fm->ffn_norm[l] = f32_to_f16(tmp, dim);
        fm->ffn_norm_f32[l] = tmp;
    }
    // Read all w1
    for (int l = 0; l < nl && l < FIBER_LAYERS; l++) {
        float *tmp = malloc(hd * sizeof(float));
        fread(tmp, 4, hd, f);
        fm->w1[l] = f32_to_f16(tmp, hd);
        fm->w1_f32[l] = tmp;
    }
    // Read all w2
    for (int l = 0; l < nl && l < FIBER_LAYERS; l++) {
        float *tmp = malloc(dh * sizeof(float));
        fread(tmp, 4, dh, f);
        fm->w2[l] = f32_to_f16(tmp, dh);
        fm->w2_f32[l] = tmp;
    }
    // Read all w3
    for (int l = 0; l < nl && l < FIBER_LAYERS; l++) {
        float *tmp = malloc(hd * sizeof(float));
        fread(tmp, 4, hd, f);
        fm->w3[l] = f32_to_f16(tmp, hd);
        fm->w3_f32[l] = tmp;
    }
    // rms_final
    float *rms_final = malloc(dim * sizeof(float));
    fread(rms_final, 4, dim, f);
    fm->output_norm = f32_to_f16(rms_final, dim);
    free(rms_final);

    // Output weights: shared with embedding if shared_weights, else read separately
    if (shared_weights) {
        fm->output = f32_to_f16(embed, vocab * dim);
    } else {
        // Skip freq_cis (cos + sin tables)
        fseek(f, (long)(dim / heads) * seq * 4 * 2, SEEK_CUR);
        float *out_w = malloc((size_t)vocab * dim * sizeof(float));
        fread(out_w, 4, (size_t)vocab * dim, f);
        fm->output = f32_to_f16(out_w, vocab * dim);
        free(out_w);
    }
    free(embed);

    fclose(f);
    fm->total_bytes = total;
    printf("Loaded %d layers (%.1f MB)\n", nl, total / (1024.0 * 1024.0));
    return fm;
}
