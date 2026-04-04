// ane_attn.m — ANE attention for Fiber-Inference Phase 2
#import <Foundation/Foundation.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ane_attn.h"
#include "ane.h"
#include "ane_mil.h"
#include "model.h"
#include "gpu_ffn.h"
#include "kv_cache.h"
#include "dequant.h"
#include "timer.h"
#include <Accelerate/Accelerate.h>

struct ane_attn_context {
    bool initialized;
    int max_prefill_seq;
    // Per-layer compiled kernels (22 layers)
    ANEKernel **kernels;
    int n_layers;
    // Reusable padded I/O buffers (allocated on first use)
    _Float16 *pad_in;
    _Float16 *pad_out;
};

ane_attn_context_t *ane_attn_init(void) {
    int rc = ane_init();
    if (rc != 0) {
        fprintf(stderr, "ANE init failed (rc=%d)\n", rc);
        return NULL;
    }

    ANEDeviceInfo info = ane_device_info();
    printf("ANE: %s, %d cores\n", info.arch, info.num_cores);

    ane_attn_context_t *ctx = calloc(1, sizeof(*ctx));
    ctx->initialized = true;
    return ctx;
}

// Build cos/sin RoPE tables as ANEWeight blobs
static ANEWeight make_rope_weight(const char *name, int hd2, int seq, float base, bool is_sin) {
    // Compute frequency table and cos/sin values
    size_t n = (size_t)hd2 * seq;
    _Float16 *data = malloc(n * sizeof(_Float16));

    for (int i = 0; i < hd2; i++) {
        float freq = 1.0f / powf(base, (float)(2 * i) / (float)(hd2 * 2));
        for (int p = 0; p < seq; p++) {
            float angle = (float)p * freq;
            float val = is_sin ? sinf(angle) : cosf(angle);
            data[(size_t)i * seq + p] = (_Float16)val;
        }
    }

    // Build ANEWeight with 128-byte header + FP16 data
    size_t data_bytes = n * sizeof(_Float16);
    size_t total = 128 + data_bytes;
    uint8_t *blob = calloc(1, total);

    // Global header
    blob[0] = 0x01;
    blob[4] = 0x02;

    // Per-chunk header at offset 64
    blob[64] = 0xEF; blob[65] = 0xBE; blob[66] = 0xAD; blob[67] = 0xDE; // DEADBEEF LE
    blob[68] = 0x01;
    // size at offset 64+8=72
    memcpy(blob + 72, &data_bytes, sizeof(uint64_t));
    // offset at 64+16=80
    uint64_t payload_off = 128;
    memcpy(blob + 80, &payload_off, sizeof(uint64_t));

    // Payload
    memcpy(blob + 128, data, data_bytes);
    free(data);

    ANEWeight w;
    w.name = name;
    w.data = blob;
    w.len = total;
    return w;
}

// Build sliding window causal mask as ANEWeight blob
// window=0 means full causal (no window limit)
static ANEWeight make_sliding_window_mask(const char *name, int seq, int window) {
    size_t n = (size_t)seq * seq;
    _Float16 *data = malloc(n * sizeof(_Float16));
    _Float16 neg_inf = (_Float16)(-65504.0f);
    for (int i = 0; i < seq; i++) {
        for (int j = 0; j < seq; j++) {
            bool causal = (j <= i);
            bool in_window = (window <= 0) || ((i - j) < window);
            data[(size_t)i * seq + j] = (causal && in_window) ? (_Float16)0.0f : neg_inf;
        }
    }
    size_t data_bytes = n * sizeof(_Float16);
    size_t total = 128 + data_bytes;
    uint8_t *blob = calloc(1, total);
    blob[0] = 0x01; blob[4] = 0x02;
    blob[64] = 0xEF; blob[65] = 0xBE; blob[66] = 0xAD; blob[67] = 0xDE;
    blob[68] = 0x01;
    memcpy(blob + 72, &data_bytes, sizeof(uint64_t));
    uint64_t payload_off = 128;
    memcpy(blob + 80, &payload_off, sizeof(uint64_t));
    memcpy(blob + 128, data, data_bytes);
    free(data);
    ANEWeight w;
    w.name = name;
    w.data = blob;
    w.len = total;
    return w;
}

// Build causal mask as ANEWeight blob
static ANEWeight make_causal_mask(const char *name, int seq) {
    size_t n = (size_t)seq * seq;
    _Float16 *data = malloc(n * sizeof(_Float16));

    // Lower triangle + diagonal = 0, upper triangle = -inf (FP16)
    _Float16 neg_inf = (_Float16)(-65504.0f);
    for (int i = 0; i < seq; i++) {
        for (int j = 0; j < seq; j++) {
            data[(size_t)i * seq + j] = (j <= i) ? (_Float16)0.0f : neg_inf;
        }
    }

    size_t data_bytes = n * sizeof(_Float16);
    size_t total = 128 + data_bytes;
    uint8_t *blob = calloc(1, total);
    blob[0] = 0x01; blob[4] = 0x02;
    blob[64] = 0xEF; blob[65] = 0xBE; blob[66] = 0xAD; blob[67] = 0xDE;
    blob[68] = 0x01;
    memcpy(blob + 72, &data_bytes, sizeof(uint64_t));
    uint64_t payload_off = 128;
    memcpy(blob + 80, &payload_off, sizeof(uint64_t));
    memcpy(blob + 128, data, data_bytes);
    free(data);

    ANEWeight w;
    w.name = name;
    w.data = blob;
    w.len = total;
    return w;
}

// Build ANEWeight blob from already-FP16 data (no conversion needed)
static ANEWeight make_fp16_weight(const char *name, const _Float16 *src, int rows, int cols) {
    size_t wsize = (size_t)rows * cols * 2;
    size_t total = 128 + wsize;
    uint8_t *buf = calloc(total, 1);

    buf[0] = 0x01; buf[4] = 0x02;
    buf[64] = 0xEF; buf[65] = 0xBE; buf[66] = 0xAD; buf[67] = 0xDE;
    buf[68] = 0x01;
    *(uint32_t *)(buf + 72) = (uint32_t)wsize;
    *(uint32_t *)(buf + 80) = 128;
    memcpy(buf + 128, src, wsize);

    return (ANEWeight){.data = buf, .len = total, .name = name};
}

bool ane_attn_compile(ane_attn_context_t *ctx, model_t *m, int max_prefill_seq) {
    if (!ctx || !ctx->initialized) return false;
    if (!m->attn_fp16_ready) {
        fprintf(stderr, "ANE: attention weights not dequantized\n");
        return false;
    }

    uint32_t nl = m->params.n_layers;
    uint32_t dim = m->params.dim;
    uint32_t n_heads = m->params.n_heads;
    uint32_t n_kv_heads = m->params.n_kv_heads;
    uint32_t head_dim = m->params.head_dim;
    int kv_dim = n_kv_heads * head_dim;
    int seq = max_prefill_seq;

    ctx->max_prefill_seq = seq;
    ctx->n_layers = nl;
    ctx->kernels = calloc(nl, sizeof(ANEKernel *));

    // Generate MIL text (same for all layers, only weights differ)
    NSString *mil = gen_sdpa_prefill_mil(dim, n_heads, n_kv_heads, head_dim,
                                          seq, m->params.rope_freq_base,
                                          m->params.rms_norm_eps);
    const char *mil_c = [mil UTF8String];
    size_t mil_len = strlen(mil_c);
    printf("ANE MIL: %zu bytes generated\n", mil_len);

    // Build shared weights (cos, sin, mask)
    ANEWeight w_cos = make_rope_weight("@model_path/weights/cos.bin",
                                        head_dim / 2, seq, m->params.rope_freq_base, false);
    ANEWeight w_sin = make_rope_weight("@model_path/weights/sin.bin",
                                        head_dim / 2, seq, m->params.rope_freq_base, true);
    ANEWeight w_mask = make_causal_mask("@model_path/weights/mask.bin", seq);

    // I/O sizes
    size_t in_bytes = (size_t)dim * seq * sizeof(_Float16);
    int out_ch = dim + 2 * kv_dim;
    size_t out_bytes = (size_t)out_ch * seq * sizeof(_Float16);

    int compiled = 0;
    for (uint32_t l = 0; l < nl; l++) {
        // Per-layer attention weights (already FP16)
        ANEWeight w_rms = make_fp16_weight("@model_path/weights/rms1.bin",
                                            m->attn_norm_fp16[l], 1, dim);
        ANEWeight w_wq = make_fp16_weight("@model_path/weights/wq.bin",
                                           m->attn_wq_fp16[l], dim, dim);
        ANEWeight w_wk = make_fp16_weight("@model_path/weights/wk.bin",
                                           m->attn_wk_fp16[l], kv_dim, dim);
        ANEWeight w_wv = make_fp16_weight("@model_path/weights/wv.bin",
                                           m->attn_wv_fp16[l], kv_dim, dim);
        ANEWeight w_wo = make_fp16_weight("@model_path/weights/wo.bin",
                                           m->attn_wo_fp16[l], dim, dim);

        ANEWeight weights[] = { w_rms, w_wq, w_wk, w_wv, w_wo, w_cos, w_sin, w_mask };
        int n_weights = 8;

        ctx->kernels[l] = ane_compile(mil_c, mil_len,
                                       weights, n_weights,
                                       1, &in_bytes,
                                       1, &out_bytes,
                                       ANE_QOS_BACKGROUND);

        // Free per-layer weight blobs (ANE bakes them at compile)
        free(w_rms.data);
        free(w_wq.data);
        free(w_wk.data);
        free(w_wv.data);
        free(w_wo.data);

        if (ctx->kernels[l]) {
            compiled++;
            if (ane_sram_spill(ctx->kernels[l])) {
                fprintf(stderr, "ANE: layer %u SRAM spill detected\n", l);
            }
        } else {
            fprintf(stderr, "ANE: layer %u compile FAILED\n", l);
        }
    }

    // Free shared weight blobs
    free((void *)w_cos.data);
    free((void *)w_sin.data);
    free((void *)w_mask.data);

    printf("ANE: compiled %d/%u kernels (budget used: %d/119)\n",
           compiled, nl, ane_compile_count());
    return compiled == (int)nl;
}

// Run ANE attention for one layer on a test input and validate output is non-zero.
// x_in: [dim * seq] FP16 input (channels-first: x[d][t] = x_in[d * seq + t])
// out:  [out_ch * seq] FP16 output where out_ch = dim + 2*kv_dim
bool ane_attn_eval_layer(ane_attn_context_t *ctx, int layer,
                          const _Float16 *x_in, int dim, int seq,
                          _Float16 *out, int out_ch) {
    if (!ctx || layer >= ctx->n_layers || !ctx->kernels[layer]) return false;

    ANEKernel *k = ctx->kernels[layer];
    int compiled_seq = ctx->max_prefill_seq;
    size_t in_bytes = (size_t)dim * compiled_seq * sizeof(_Float16);
    size_t out_bytes = (size_t)out_ch * compiled_seq * sizeof(_Float16);

    // Zero-copy I/O via IOSurface pointers (from libane)
    ane_lock_input(k, 0);
    _Float16 *io_in = (_Float16 *)ane_input_ptr(k, 0);

    if (seq == compiled_seq) {
        // Fast path: direct copy, no padding
        memcpy(io_in, x_in, in_bytes);
    } else {
        // Padded path: zero-fill then copy per channel
        memset(io_in, 0, in_bytes);
        for (int d = 0; d < dim; d++) {
            memcpy(io_in + (size_t)d * compiled_seq,
                   x_in + (size_t)d * seq,
                   seq * sizeof(_Float16));
        }
    }
    ane_unlock_input(k, 0);

    // Eval
    if (!ane_eval(k, ANE_QOS_BACKGROUND)) return false;

    // Read output directly from IOSurface
    ane_lock_output(k, 0);
    _Float16 *io_out = (_Float16 *)ane_output_ptr(k, 0);

    if (seq == compiled_seq) {
        memcpy(out, io_out, out_bytes);
    } else {
        for (int c = 0; c < out_ch; c++) {
            memcpy(out + (size_t)c * seq,
                   io_out + (size_t)c * compiled_seq,
                   seq * sizeof(_Float16));
        }
    }
    ane_unlock_output(k, 0);
    return true;
}

// Quick validation: run ANE on random-ish input, check output is non-zero
void ane_attn_validate(ane_attn_context_t *ctx, const model_t *m) {
    if (!ctx || ctx->n_layers == 0) return;

    int dim = m->params.dim;
    int kv_dim = m->params.n_kv_heads * m->params.head_dim;
    int out_ch = dim + 2 * kv_dim;
    int seq = 4; // small test

    // Create simple test input
    _Float16 *x_in = calloc(dim * seq, sizeof(_Float16));
    for (int i = 0; i < dim * seq; i++) {
        x_in[i] = (_Float16)(0.01f * (float)(i % 100 - 50));
    }

    _Float16 *out = calloc(out_ch * seq, sizeof(_Float16));

    printf("\n--- ANE Validation ---\n");
    bool ok = ane_attn_eval_layer(ctx, 0, x_in, dim, seq, out, out_ch);
    if (!ok) {
        printf("Layer 0: EVAL FAILED\n");
    } else {
        // Check output statistics
        float max_val = 0, sum = 0;
        int nonzero = 0;
        for (int i = 0; i < dim * seq; i++) { // check wo_out portion only
            float v = fabsf((float)out[i]);
            if (v > max_val) max_val = v;
            sum += v;
            if (v > 1e-6f) nonzero++;
        }
        printf("Layer 0: max=%.4f, mean=%.6f, nonzero=%d/%d — %s\n",
               max_val, sum / (dim * seq), nonzero, dim * seq,
               (nonzero > 0 && max_val > 1e-4f) ? "PASS" : "FAIL");
    }

    free(x_in);
    free(out);
}

// Batched prefill: ANE attention + GPU FFN for all prompt tokens
double ane_prefill_batch(ane_attn_context_t *ctx, gpu_context_t *gpu,
                          const model_t *m, kv_cache_t *kv,
                          const int *tokens, int n_tokens) {
    timer_init();
    uint64_t t0 = timer_now();
    int dim = m->params.dim;
    int n_kv_heads = m->params.n_kv_heads;
    int head_dim = m->params.head_dim;
    int kv_dim = n_kv_heads * head_dim;
    int out_ch = dim + 2 * kv_dim;
    int seq = n_tokens;  // actual token count
    int n_layers = m->params.n_layers;
    int work_seq = ctx->max_prefill_seq; // padded working size (>= 128)

    // Limit to compiled max sequence length
    if (seq > work_seq) seq = work_seq;

    uint64_t t_embed = timer_now();
    // 1. Embed all tokens → [dim, work_seq] FP16 (channels-first, zero-padded)
    _Float16 *x = calloc((size_t)dim * work_seq, sizeof(_Float16));
    _Float16 *token_buf = calloc(dim, sizeof(_Float16));
    for (int t = 0; t < seq; t++) {
        // Dequant embedding for this token
        switch (m->token_embd_type) {
            case GGML_TYPE_Q4_K:
                dequant_q4k_row_f16(m->token_embd, tokens[t], token_buf, dim);
                break;
            case GGML_TYPE_Q6_K:
                dequant_q6k_row_f16(m->token_embd, tokens[t], token_buf, dim);
                break;
            case GGML_TYPE_F16:
                memcpy(token_buf, (_Float16 *)m->token_embd + (size_t)tokens[t] * dim,
                       dim * sizeof(_Float16));
                break;
            default:
                break;
        }
        // Copy to channels-first: x[d * work_seq + t]
        for (int d = 0; d < dim; d++) {
            x[(size_t)d * work_seq + t] = token_buf[d];
        }
    }
    free(token_buf);

    double embed_ms = timer_ms(t_embed, timer_now());

    // Working buffers for ANE output (work_seq size for consistent MPS dimensions)
    _Float16 *ane_out = calloc((size_t)out_ch * work_seq, sizeof(_Float16));
    double total_ane_ms = 0, total_ffn_ms = 0, total_kv_ms = 0, total_res_ms = 0;

    // 2. For each layer: ANE attention → residual → GPU FFN → residual
    for (int l = 0; l < n_layers; l++) {
        // a. ANE: full-batch attention (RMSNorm + QKV + RoPE + SDPA + Wo)
        uint64_t ta = timer_now();
        bool ok = ane_attn_eval_layer(ctx, l, x, dim, work_seq, ane_out, out_ch);
        total_ane_ms += timer_ms(ta, timer_now());
        if (!ok) {
            fprintf(stderr, "ANE: prefill layer %d eval failed\n", l);
            break;
        }

        // b. Residual add on CPU (full work_seq width for consistency)
        uint64_t tr = timer_now();
        for (size_t i = 0; i < (size_t)dim * work_seq; i++) {
            x[i] += ane_out[i];
        }
        total_res_ms += timer_ms(tr, timer_now());

        // c. Store K and V into KV cache
        uint64_t tk = timer_now();
        //    ANE K output: channels [dim .. dim+kv_dim-1], channels-first [kv_dim, seq]
        //    KV cache layout: [head][pos][dim] = cache[h * max_seq * head_dim + pos * head_dim + d]
        _Float16 *k_out = ane_out + (size_t)dim * work_seq;      // [kv_dim, work_seq]
        _Float16 *v_out = ane_out + (size_t)(dim + kv_dim) * work_seq; // [kv_dim, work_seq]
        _Float16 *k_cache = (_Float16 *)((id<MTLBuffer>)(kv->k_cache[l])).contents;
        _Float16 *v_cache = (_Float16 *)((id<MTLBuffer>)(kv->v_cache[l])).contents;

        for (int h = 0; h < n_kv_heads; h++) {
            for (int t = 0; t < seq; t++) {
                for (int d = 0; d < head_dim; d++) {
                    int ch = h * head_dim + d; // channel index in kv_dim
                    int cache_idx = h * kv->max_seq * head_dim + t * head_dim + d;
                    k_cache[cache_idx] = k_out[(size_t)ch * work_seq + t];
                    v_cache[cache_idx] = v_out[(size_t)ch * work_seq + t];
                }
            }
        }
        total_kv_ms += timer_ms(tk, timer_now());

        // d. GPU batched FFN: 1 command buffer per layer (MPS matmul)
        uint64_t tf = timer_now();
        gpu_forward_ffn_batch(gpu, m, l, x, work_seq);
        total_ffn_ms += timer_ms(tf, timer_now());
    }

    // 3. Classifier: copy last token to GPU, run classifier
    {
        _Float16 *gpu_x = gpu_get_buf_x(gpu);
        for (int d = 0; d < dim; d++) {
            gpu_x[d] = x[(size_t)d * work_seq + (seq - 1)];
        }
        gpu_forward_classifier(gpu, m);
    }

    free(x);
    free(ane_out);

    double ms = timer_ms(t0, timer_now());
    double classified_ms = total_ane_ms + total_ffn_ms + total_kv_ms + total_res_ms + embed_ms;
    fprintf(stderr, "\n=== Prefill Breakdown (%d tokens, %d layers) ===\n", seq, n_layers);
    fprintf(stderr, "  Embed:     %6.1f ms\n", embed_ms);
    fprintf(stderr, "  ANE attn:  %6.1f ms (%5.1f ms/layer)\n", total_ane_ms, total_ane_ms/n_layers);
    fprintf(stderr, "  Residual:  %6.1f ms (%5.1f ms/layer)\n", total_res_ms, total_res_ms/n_layers);
    fprintf(stderr, "  KV cache:  %6.1f ms (%5.1f ms/layer)\n", total_kv_ms, total_kv_ms/n_layers);
    fprintf(stderr, "  GPU FFN:   %6.1f ms (%5.1f ms/layer)\n", total_ffn_ms, total_ffn_ms/n_layers);
    fprintf(stderr, "  Unaccounted: %5.1f ms (ANE I/O, memcpy, classifier)\n", ms - classified_ms);
    fprintf(stderr, "  Total:     %6.1f ms (%.1f tok/s)\n", ms, (double)seq / (ms / 1000.0));
    fprintf(stderr, "===========================================\n");
    return (double)seq / (ms / 1000.0); // tok/s
}

void ane_attn_free(ane_attn_context_t *ctx) {
    if (!ctx) return;
    if (ctx->kernels) {
        for (int l = 0; l < ctx->n_layers; l++) {
            if (ctx->kernels[l]) ane_free(ctx->kernels[l]);
        }
        free(ctx->kernels);
    }
    free(ctx->pad_in);
    free(ctx->pad_out);
    free(ctx);
}
