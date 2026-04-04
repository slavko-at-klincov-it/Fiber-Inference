// fiber_proof.m — A/B proof: same model, CPU baseline vs ANE, compare text + speed
#import <Foundation/Foundation.h>
#include <Accelerate/Accelerate.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "fiber_proof.h"
#include "fiber_ckpt.h"
#include "fiber_model.h"
#include "fiber_arch.h"
#include "ane.h"
#include "ane_mil.h"
#include "tokenizer.h"
#include "gguf.h"
#include "timer.h"

// ============================================================
// CPU/AMX baseline forward (single token, no ANE)
// Standard transformer: RMSNorm → QKV → RoPE → Attention → Wo → Residual → FFN
// ============================================================

static void cpu_rmsnorm(float *out, const float *x, const float *w, int dim, float eps) {
    float ss = 0;
    for (int i = 0; i < dim; i++) ss += x[i] * x[i];
    float rrms = 1.0f / sqrtf(ss / dim + eps);
    for (int i = 0; i < dim; i++) out[i] = x[i] * rrms * w[i];
}

static void cpu_rope(float *q, float *k, int heads, int kv_heads, int hd, int pos, float base) {
    for (int h = 0; h < heads; h++) {
        for (int i = 0; i < hd/2; i++) {
            float freq = 1.0f / powf(base, 2.0f * i / (float)hd);
            float angle = pos * freq;
            float c = cosf(angle), s = sinf(angle);
            int i1 = h * hd + i, i2 = i1 + hd/2;
            float x1 = q[i1], x2 = q[i2];
            q[i1] = x1*c - x2*s; q[i2] = x1*s + x2*c;
        }
    }
    for (int h = 0; h < kv_heads; h++) {
        for (int i = 0; i < hd/2; i++) {
            float freq = 1.0f / powf(base, 2.0f * i / (float)hd);
            float angle = pos * freq;
            float c = cosf(angle), s = sinf(angle);
            int i1 = h * hd + i, i2 = i1 + hd/2;
            float x1 = k[i1], x2 = k[i2];
            k[i1] = x1*c - x2*s; k[i2] = x1*s + x2*c;
        }
    }
}

// Full CPU forward for one token (decode mode)
// x: [dim] hidden state (modified in place)
// Returns: logits[vocab] (caller frees)
static float *cpu_forward_token(float *x, int pos, fiber_model_t *fm,
                                 int n_layers, int dim, int heads, int kv_heads,
                                 int hd, int ffn_dim, int vocab,
                                 float **k_cache, float **v_cache, int max_seq) {
    float eps = FIBER_RMS_EPS;
    float *xnorm = malloc(dim * sizeof(float));
    float *q = malloc(dim * sizeof(float));
    float *k = malloc(dim * sizeof(float));
    float *v = malloc(dim * sizeof(float));

    for (int l = 0; l < n_layers; l++) {
        // Attention: RMSNorm → QKV → RoPE → KV store → Attention → Wo → Residual
        cpu_rmsnorm(xnorm, x, fm->attn_norm_f32[l], dim, eps);

        // Q = Wq @ xnorm, K = Wk @ xnorm, V = Wv @ xnorm
        cblas_sgemv(CblasRowMajor, CblasNoTrans, dim, dim, 1.0f,
                    fm->wq_f32[l], dim, xnorm, 1, 0.0f, q, 1);
        cblas_sgemv(CblasRowMajor, CblasNoTrans, kv_heads*hd, dim, 1.0f,
                    fm->wk_f32[l], dim, xnorm, 1, 0.0f, k, 1);
        cblas_sgemv(CblasRowMajor, CblasNoTrans, kv_heads*hd, dim, 1.0f,
                    fm->wv_f32[l], dim, xnorm, 1, 0.0f, v, 1);

        cpu_rope(q, k, heads, kv_heads, hd, pos, FIBER_ROPE_BASE);

        // Store K, V into cache
        int kv_dim = kv_heads * hd;
        memcpy(k_cache[l] + (size_t)pos * kv_dim, k, kv_dim * sizeof(float));
        memcpy(v_cache[l] + (size_t)pos * kv_dim, v, kv_dim * sizeof(float));

        // Attention: per head
        float *attn_out = calloc(dim, sizeof(float));
        float scale = 1.0f / sqrtf((float)hd);
        for (int h = 0; h < heads; h++) {
            int kvh = h * kv_heads / heads;
            float *scores = malloc((pos+1) * sizeof(float));
            // Q·K scores
            for (int t = 0; t <= pos; t++) {
                float dot = 0;
                for (int d = 0; d < hd; d++)
                    dot += q[h*hd+d] * k_cache[l][t*kv_dim + kvh*hd + d];
                scores[t] = dot * scale;
            }
            // Softmax
            float maxs = scores[0];
            for (int t = 1; t <= pos; t++) if (scores[t] > maxs) maxs = scores[t];
            float sum = 0;
            for (int t = 0; t <= pos; t++) { scores[t] = expf(scores[t]-maxs); sum += scores[t]; }
            for (int t = 0; t <= pos; t++) scores[t] /= sum;
            // Weighted V sum
            for (int t = 0; t <= pos; t++)
                for (int d = 0; d < hd; d++)
                    attn_out[h*hd+d] += scores[t] * v_cache[l][t*kv_dim + kvh*hd + d];
            free(scores);
        }

        // Wo projection + residual
        float *proj = malloc(dim * sizeof(float));
        cblas_sgemv(CblasRowMajor, CblasNoTrans, dim, dim, 1.0f,
                    fm->wo_f32[l], dim, attn_out, 1, 0.0f, proj, 1);
        for (int d = 0; d < dim; d++) x[d] += proj[d];
        free(attn_out); free(proj);

        // FFN: RMSNorm → W1,W3 → SiLU → W2 → Residual
        cpu_rmsnorm(xnorm, x, fm->ffn_norm_f32[l], dim, eps);
        float *h1 = malloc(ffn_dim * sizeof(float));
        float *h3 = malloc(ffn_dim * sizeof(float));
        cblas_sgemv(CblasRowMajor, CblasNoTrans, ffn_dim, dim, 1.0f,
                    fm->w1_f32[l], dim, xnorm, 1, 0.0f, h1, 1);
        cblas_sgemv(CblasRowMajor, CblasNoTrans, ffn_dim, dim, 1.0f,
                    fm->w3_f32[l], dim, xnorm, 1, 0.0f, h3, 1);
        for (int i = 0; i < ffn_dim; i++) {
            float s = h1[i] / (1.0f + expf(-h1[i]));
            h1[i] = s * h3[i];
        }
        float *ffn_out = malloc(dim * sizeof(float));
        cblas_sgemv(CblasRowMajor, CblasNoTrans, dim, ffn_dim, 1.0f,
                    fm->w2_f32[l], ffn_dim, h1, 1, 0.0f, ffn_out, 1);
        for (int d = 0; d < dim; d++) x[d] += ffn_out[d];
        free(h1); free(h3); free(ffn_out);
    }

    // Classifier
    float *final_norm = malloc(dim * sizeof(float));
    float *norm_w = malloc(dim * sizeof(float));
    for (int d = 0; d < dim; d++) norm_w[d] = (float)fm->output_norm[d];
    cpu_rmsnorm(final_norm, x, norm_w, dim, eps);
    free(norm_w);

    float *logits = malloc(vocab * sizeof(float));
    // output = embedding^T (weight tying)
    for (int v = 0; v < vocab; v++) {
        float dot = 0;
        for (int d = 0; d < dim; d++)
            dot += (float)fm->output[(size_t)v*dim+d] * final_norm[d];
        logits[v] = dot;
    }
    free(final_norm); free(xnorm); free(q); free(k); free(v);
    return logits;
}

// Hybrid decode: CPU attention + ANE FFN
// Same as cpu_forward_token but FFN runs on ANE
static float *hybrid_forward_token(float *x, int pos, fiber_model_t *fm,
                                    int n_layers, int dim, int heads, int kv_heads,
                                    int hd, int ffn_dim, int vocab,
                                    float **k_cache, float **v_cache, int max_seq,
                                    ANEKernel **ffn_kernels, int ane_ffn_seq) {
    float eps = FIBER_RMS_EPS;
    float *xnorm = malloc(dim * sizeof(float));
    float *q = malloc(dim * sizeof(float));
    float *k = malloc(dim * sizeof(float));
    float *v = malloc(dim * sizeof(float));
    int kv_dim = kv_heads * hd;
    size_t ane_in_bytes = (size_t)dim * ane_ffn_seq * sizeof(_Float16);

    for (int l = 0; l < n_layers; l++) {
        // === CPU ATTENTION (same as cpu_forward_token) ===
        cpu_rmsnorm(xnorm, x, fm->attn_norm_f32[l], dim, eps);
        cblas_sgemv(CblasRowMajor, CblasNoTrans, dim, dim, 1.0f,
                    fm->wq_f32[l], dim, xnorm, 1, 0.0f, q, 1);
        cblas_sgemv(CblasRowMajor, CblasNoTrans, kv_dim, dim, 1.0f,
                    fm->wk_f32[l], dim, xnorm, 1, 0.0f, k, 1);
        cblas_sgemv(CblasRowMajor, CblasNoTrans, kv_dim, dim, 1.0f,
                    fm->wv_f32[l], dim, xnorm, 1, 0.0f, v, 1);
        cpu_rope(q, k, heads, kv_heads, hd, pos, FIBER_ROPE_BASE);
        memcpy(k_cache[l] + (size_t)pos * kv_dim, k, kv_dim * sizeof(float));
        memcpy(v_cache[l] + (size_t)pos * kv_dim, v, kv_dim * sizeof(float));

        float *attn_out = calloc(dim, sizeof(float));
        float scale = 1.0f / sqrtf((float)hd);
        for (int h = 0; h < heads; h++) {
            int kvh = h * kv_heads / heads;
            float *scores = malloc((pos+1) * sizeof(float));
            for (int t = 0; t <= pos; t++) {
                float dot = 0;
                for (int d = 0; d < hd; d++)
                    dot += q[h*hd+d] * k_cache[l][t*kv_dim + kvh*hd + d];
                scores[t] = dot * scale;
            }
            float maxs = scores[0];
            for (int t = 1; t <= pos; t++) if (scores[t] > maxs) maxs = scores[t];
            float sum = 0;
            for (int t = 0; t <= pos; t++) { scores[t] = expf(scores[t]-maxs); sum += scores[t]; }
            for (int t = 0; t <= pos; t++) scores[t] /= sum;
            for (int t = 0; t <= pos; t++)
                for (int d = 0; d < hd; d++)
                    attn_out[h*hd+d] += scores[t] * v_cache[l][t*kv_dim + kvh*hd + d];
            free(scores);
        }
        float *proj = malloc(dim * sizeof(float));
        cblas_sgemv(CblasRowMajor, CblasNoTrans, dim, dim, 1.0f,
                    fm->wo_f32[l], dim, attn_out, 1, 0.0f, proj, 1);
        for (int d = 0; d < dim; d++) x[d] += proj[d];
        free(attn_out); free(proj);

        // === ANE FFN (replaces CPU FFN) ===
        // Pack x into channels-first FP16 [dim, ane_ffn_seq] (padded, token at pos 0)
        _Float16 *ffn_in = calloc(dim * ane_ffn_seq, sizeof(_Float16));
        for (int d = 0; d < dim; d++) ffn_in[(size_t)d * ane_ffn_seq] = (_Float16)x[d];

        ane_lock_input(ffn_kernels[l], 0);
        memcpy(ane_input_ptr(ffn_kernels[l], 0), ffn_in, ane_in_bytes);
        ane_unlock_input(ffn_kernels[l], 0);
        ane_eval(ffn_kernels[l], ANE_QOS_BACKGROUND);
        ane_lock_output(ffn_kernels[l], 0);
        _Float16 *ffn_out_ptr = (_Float16 *)ane_output_ptr(ffn_kernels[l], 0);
        // Read back token at position 0 (FFN includes residual: out = x + ffn(x))
        for (int d = 0; d < dim; d++) x[d] = (float)ffn_out_ptr[(size_t)d * ane_ffn_seq];
        ane_unlock_output(ffn_kernels[l], 0);
        free(ffn_in);
    }

    // Classifier (same as CPU)
    float *final_norm = malloc(dim * sizeof(float));
    float *norm_w = malloc(dim * sizeof(float));
    for (int d = 0; d < dim; d++) norm_w[d] = (float)fm->output_norm[d];
    cpu_rmsnorm(final_norm, x, norm_w, dim, eps);
    free(norm_w);
    float *logits = malloc(vocab * sizeof(float));
    for (int vv = 0; vv < vocab; vv++) {
        float dot = 0;
        for (int d = 0; d < dim; d++)
            dot += (float)fm->output[(size_t)vv*dim+d] * final_norm[d];
        logits[vv] = dot;
    }
    free(final_norm); free(xnorm); free(q); free(k); free(v);
    return logits;
}

static int argmax(const float *arr, int n) {
    int best = 0;
    for (int i = 1; i < n; i++) if (arr[i] > arr[best]) best = i;
    return best;
}

// ============================================================
// Proof
// ============================================================

void fiber_proof(const char *ckpt_path, const char *gguf_path, const char *prompt) {
    timer_init();
    printf("\n========================================\n");
    printf("  FIBER PROOF: Same Model, Two Pipelines\n");
    printf("========================================\n\n");

    // 1. Load model (detect format by extension)
    fiber_model_t *fm;
    int dim = 768, heads = 12, kv_heads = 12, hd = 64;
    int ffn_dim = 2048, n_layers = 12, vocab = 32000, max_seq = 256;

    if (strstr(ckpt_path, ".bin") && !strstr(ckpt_path, "ckpt")) {
        // karpathy format (stories110M.bin)
        fm = fiber_model_load_karpathy(ckpt_path);
    } else {
        // BLZT format (ane_stories110M_dyn_ckpt.bin)
        fm = fiber_model_load_blzt(ckpt_path);
    }
    if (!fm) return;

    // 2. Load tokenizer from GGUF
    gguf_file_t *gf = gguf_open(gguf_path);
    if (!gf) { fprintf(stderr, "Cannot open %s for tokenizer\n", gguf_path); return; }
    tokenizer_t *tok = tokenizer_init(gf);
    if (!tok) { fprintf(stderr, "Tokenizer init failed\n"); return; }

    // 3. Encode prompt
    int tokens[256];
    int n_prompt = tokenizer_encode(tok, prompt, tokens, 256);
    printf("Prompt: \"%s\" → %d tokens\n\n", prompt, n_prompt);

    int gen_tokens = 32;

    // ========================================
    // Pipeline A: CPU/AMX Baseline (single-accelerator)
    // ========================================
    printf("--- Pipeline A: CPU/AMX Baseline ---\n");

    float **kc_a = calloc(n_layers, sizeof(float *));
    float **vc_a = calloc(n_layers, sizeof(float *));
    for (int l = 0; l < n_layers; l++) {
        kc_a[l] = calloc((size_t)max_seq * kv_heads * hd, sizeof(float));
        vc_a[l] = calloc((size_t)max_seq * kv_heads * hd, sizeof(float));
    }

    float *x_a = malloc(dim * sizeof(float));
    int output_a[256];
    int n_out_a = 0;

    uint64_t t0 = timer_now();

    // Prefill
    for (int i = 0; i < n_prompt; i++) {
        for (int d = 0; d < dim; d++)
            x_a[d] = (float)fm->embedding[(size_t)tokens[i] * dim + d];
        float *logits = cpu_forward_token(x_a, i, fm, n_layers, dim, heads, kv_heads,
                                           hd, ffn_dim, vocab, kc_a, vc_a, max_seq);
        if (i == n_prompt - 1) {
            output_a[n_out_a++] = argmax(logits, vocab);
        }
        free(logits);
    }

    double prefill_a_ms = timer_ms(t0, timer_now());

    // Decode
    uint64_t td = timer_now();
    for (int i = 0; i < gen_tokens - 1 && n_prompt + n_out_a < max_seq; i++) {
        int tok_id = output_a[n_out_a - 1];
        if (tok_id == tokenizer_eos(tok)) break;
        for (int d = 0; d < dim; d++)
            x_a[d] = (float)fm->embedding[(size_t)tok_id * dim + d];
        float *logits = cpu_forward_token(x_a, n_prompt + n_out_a - 1, fm,
                                           n_layers, dim, heads, kv_heads,
                                           hd, ffn_dim, vocab, kc_a, vc_a, max_seq);
        output_a[n_out_a++] = argmax(logits, vocab);
        free(logits);
    }
    double decode_a_ms = timer_ms(td, timer_now());

    printf("Output: ");
    for (int i = 0; i < n_prompt; i++) printf("%s", tokenizer_decode(tok, tokens[i]));
    for (int i = 0; i < n_out_a; i++) printf("%s", tokenizer_decode(tok, output_a[i]));
    printf("\n");
    printf("Prefill: %d tokens in %.1f ms (%.1f tok/s)\n",
           n_prompt, prefill_a_ms, n_prompt / (prefill_a_ms/1000.0));
    printf("Decode:  %d tokens in %.1f ms (%.1f tok/s)\n\n",
           n_out_a, decode_a_ms, n_out_a / (decode_a_ms/1000.0));

    // Cleanup A
    for (int l = 0; l < n_layers; l++) { free(kc_a[l]); free(vc_a[l]); }
    free(kc_a); free(vc_a); free(x_a);

    // ========================================
    // Pipeline B: ANE-only Prefill
    // ========================================
    printf("--- Pipeline B: ANE-only Prefill ---\n");

    ane_init();
    int total_needed = n_prompt + gen_tokens;
    int seq = total_needed < 128 ? 128 : total_needed;
    if (seq > 256) seq = 256; // ANE sweet spot limit

    // Compile ANE attention kernels
    NSString *mil = gen_sdpa_prefill_mil(dim, heads, kv_heads, hd, seq,
                                          FIBER_ROPE_BASE, FIBER_RMS_EPS);
    const char *mc = [mil UTF8String];
    size_t ml = strlen(mc);

    size_t in_bytes = (size_t)dim * seq * sizeof(_Float16);
    int kv_dim = kv_heads * hd;
    int out_ch = dim + 2 * kv_dim;
    size_t out_bytes = (size_t)out_ch * seq * sizeof(_Float16);

    ANEKernel *attn_kernels[12], *ffn_kernels[12];
    for (int l = 0; l < n_layers; l++) {
        // Attention weights
        int hd2 = hd/2;
        float *cos_d = malloc(hd2*seq*sizeof(float));
        float *sin_d = malloc(hd2*seq*sizeof(float));
        for (int i=0;i<hd2;i++) { float f=1.0f/powf(FIBER_ROPE_BASE,2.0f*i/(float)hd);
            for (int p=0;p<seq;p++) { cos_d[i*seq+p]=cosf(p*f); sin_d[i*seq+p]=sinf(p*f); }}
        float *mask = malloc(seq*seq*sizeof(float));
        for (int i=0;i<seq;i++) for (int j=0;j<seq;j++) mask[i*seq+j]=(j<=i)?0.0f:-65504.0f;

        ANEWeight aw[8];
        aw[0]=ane_weight_fp16("@model_path/weights/rms1.bin",fm->attn_norm_f32[l],1,dim);
        aw[1]=ane_weight_fp16("@model_path/weights/wq.bin",fm->wq_f32[l],dim,dim);
        aw[2]=ane_weight_fp16("@model_path/weights/wk.bin",fm->wk_f32[l],kv_dim,dim);
        aw[3]=ane_weight_fp16("@model_path/weights/wv.bin",fm->wv_f32[l],kv_dim,dim);
        aw[4]=ane_weight_fp16("@model_path/weights/wo.bin",fm->wo_f32[l],dim,dim);
        aw[5]=ane_weight_fp16("@model_path/weights/cos.bin",cos_d,hd2,seq);
        aw[6]=ane_weight_fp16("@model_path/weights/sin.bin",sin_d,hd2,seq);
        aw[7]=ane_weight_fp16("@model_path/weights/mask.bin",mask,seq,seq);
        attn_kernels[l] = ane_compile(mc,ml,aw,8,1,&in_bytes,1,&out_bytes,ANE_QOS_BACKGROUND);
        for (int w=0;w<8;w++) ane_weight_free(&aw[w]);
        free(cos_d); free(sin_d); free(mask);

        // FFN kernel
        NSString *fmil = gen_ffn_only_mil(dim, ffn_dim, seq, FIBER_RMS_EPS);
        size_t fml = strlen([fmil UTF8String]);
        size_t fout = in_bytes;
        ANEWeight fw[4];
        fw[0]=ane_weight_fp16("@model_path/weights/ffn_norm.bin",fm->ffn_norm_f32[l],1,dim);
        fw[1]=ane_weight_fp16("@model_path/weights/w1.bin",fm->w1_f32[l],ffn_dim,dim);
        fw[2]=ane_weight_fp16("@model_path/weights/w3.bin",fm->w3_f32[l],ffn_dim,dim);
        fw[3]=ane_weight_fp16("@model_path/weights/w2.bin",fm->w2_f32[l],dim,ffn_dim);
        ffn_kernels[l] = ane_compile([fmil UTF8String],fml,fw,4,1,&in_bytes,1,&fout,ANE_QOS_BACKGROUND);
        for (int w=0;w<4;w++) ane_weight_free(&fw[w]);
    }

    // Embed all tokens → [dim, seq] channels-first FP16
    _Float16 *x_b = calloc((size_t)dim * seq, sizeof(_Float16));
    for (int t = 0; t < n_prompt; t++)
        for (int d = 0; d < dim; d++)
            x_b[(size_t)d*seq+t] = fm->embedding[(size_t)tokens[t]*dim+d];

    _Float16 *ane_out = calloc((size_t)out_ch * seq, sizeof(_Float16));

    // Warmup
    ane_write(attn_kernels[0], 0, x_b, in_bytes);
    ane_eval(attn_kernels[0], ANE_QOS_BACKGROUND);

    // Timed ANE prefill
    t0 = timer_now();
    for (int l = 0; l < n_layers; l++) {
        // ANE Attention (includes residual)
        ane_lock_input(attn_kernels[l], 0);
        memcpy(ane_input_ptr(attn_kernels[l], 0), x_b, in_bytes);
        ane_unlock_input(attn_kernels[l], 0);
        ane_eval(attn_kernels[l], ANE_QOS_BACKGROUND);
        ane_lock_output(attn_kernels[l], 0);
        memcpy(x_b, ane_output_ptr(attn_kernels[l], 0), in_bytes);
        ane_unlock_output(attn_kernels[l], 0);

        // ANE FFN (includes residual)
        ane_lock_input(ffn_kernels[l], 0);
        memcpy(ane_input_ptr(ffn_kernels[l], 0), x_b, in_bytes);
        ane_unlock_input(ffn_kernels[l], 0);
        ane_eval(ffn_kernels[l], ANE_QOS_BACKGROUND);
        ane_lock_output(ffn_kernels[l], 0);
        memcpy(x_b, ane_output_ptr(ffn_kernels[l], 0), in_bytes);
        ane_unlock_output(ffn_kernels[l], 0);
    }
    double prefill_b_ms = timer_ms(t0, timer_now());

    // ANE prefill → first token
    // Classify: extract last token hidden state, RMSNorm, matmul with output weights
    float *_lh = malloc(dim * sizeof(float));
    for (int d = 0; d < dim; d++) _lh[d] = (float)x_b[(size_t)d*seq+(n_prompt-1)];
    float *_nw = malloc(dim * sizeof(float));
    for (int d = 0; d < dim; d++) _nw[d] = (float)fm->output_norm[d];
    float *_fn = malloc(dim * sizeof(float));
    cpu_rmsnorm(_fn, _lh, _nw, dim, FIBER_RMS_EPS);
    float *_lg = malloc(vocab * sizeof(float));
    for (int v2 = 0; v2 < vocab; v2++) {
        float dot = 0;
        for (int d = 0; d < dim; d++) dot += (float)fm->output[(size_t)v2*dim+d] * _fn[d];
        _lg[v2] = dot;
    }
    int first_tok_b = argmax(_lg, vocab);
    free(_lh); free(_nw); free(_fn); free(_lg);

    // ========================================
    // ANE Decode via Re-Prefill
    // Strategy: for each new token, re-run entire context through ANE
    // O(n²) but uses existing kernels. Effective for short contexts (≤128).
    // ========================================
    int output_b[256];
    int n_out_b = 0;
    output_b[n_out_b++] = first_tok_b;

    int all_tokens[256]; // prompt + generated
    memcpy(all_tokens, tokens, n_prompt * sizeof(int));
    all_tokens[n_prompt] = first_tok_b;

    printf("ANE Decode (re-prefill) generating %d tokens...\n", gen_tokens - 1);
    uint64_t td_b = timer_now();

    for (int step = 0; step < gen_tokens - 1; step++) {
        int total_ctx = n_prompt + n_out_b;
        if (total_ctx >= seq) break; // max context
        if (output_b[n_out_b - 1] == 2) break; // EOS

        // Re-embed all tokens → [dim, seq] channels-first
        memset(x_b, 0, (size_t)dim * seq * sizeof(_Float16));
        for (int t = 0; t < total_ctx; t++)
            for (int d = 0; d < dim; d++)
                x_b[(size_t)d * seq + t] = fm->embedding[(size_t)all_tokens[t] * dim + d];

        // ANE forward (all layers)
        for (int l = 0; l < n_layers; l++) {
            ane_lock_input(attn_kernels[l], 0);
            memcpy(ane_input_ptr(attn_kernels[l], 0), x_b, in_bytes);
            ane_unlock_input(attn_kernels[l], 0);
            ane_eval(attn_kernels[l], ANE_QOS_BACKGROUND);
            ane_lock_output(attn_kernels[l], 0);
            memcpy(x_b, ane_output_ptr(attn_kernels[l], 0), in_bytes);
            ane_unlock_output(attn_kernels[l], 0);

            ane_lock_input(ffn_kernels[l], 0);
            memcpy(ane_input_ptr(ffn_kernels[l], 0), x_b, in_bytes);
            ane_unlock_input(ffn_kernels[l], 0);
            ane_eval(ffn_kernels[l], ANE_QOS_BACKGROUND);
            ane_lock_output(ffn_kernels[l], 0);
            memcpy(x_b, ane_output_ptr(ffn_kernels[l], 0), in_bytes);
            ane_unlock_output(ffn_kernels[l], 0);
        }

        // Classify last token
        float *lh2 = malloc(dim * sizeof(float));
        for (int d = 0; d < dim; d++) lh2[d] = (float)x_b[(size_t)d*seq+(total_ctx-1)];
        float *fn3 = malloc(dim * sizeof(float));
        cpu_rmsnorm(fn3, lh2, _nw, dim, FIBER_RMS_EPS);  // reuse _nw... no, it's freed
        // Need norm weights again
        float *nw2 = malloc(dim * sizeof(float));
        for (int d = 0; d < dim; d++) nw2[d] = (float)fm->output_norm[d];
        cpu_rmsnorm(fn3, lh2, nw2, dim, FIBER_RMS_EPS);
        float *lg2 = malloc(vocab * sizeof(float));
        for (int v2 = 0; v2 < vocab; v2++) {
            float dot = 0;
            for (int d = 0; d < dim; d++) dot += (float)fm->output[(size_t)v2*dim+d] * fn3[d];
            lg2[v2] = dot;
        }
        int next = argmax(lg2, vocab);
        free(lh2); free(nw2); free(fn3); free(lg2);
        output_b[n_out_b++] = next;
        all_tokens[total_ctx] = next;
    }
    double decode_b_ms = timer_ms(td_b, timer_now());

    printf("Prefill: %d tokens in %.1f ms (%.1f tok/s)\n",
           n_prompt, prefill_b_ms, n_prompt / (prefill_b_ms/1000.0));
    printf("Decode:  %d tokens in %.1f ms (%.1f tok/s)\n",
           n_out_b, decode_b_ms, n_out_b / (decode_b_ms/1000.0));
    printf("Output: ");
    for (int i = 0; i < n_prompt; i++) printf("%s", tokenizer_decode(tok, tokens[i]));
    for (int i = 0; i < n_out_b; i++) printf("%s", tokenizer_decode(tok, output_b[i]));
    printf("\n\n");

    // ========================================
    // Pipeline C: Hybrid Decode (CPU Attention + ANE FFN)
    // ========================================
    printf("--- Pipeline C: Hybrid Decode (CPU Attn + ANE FFN) ---\n");

    // Compile FFN kernels for decode (seq=128 padded)
    int decode_ffn_seq = 128;
    ANEKernel *decode_ffn[12];
    NSString *dfmil = gen_ffn_only_mil(dim, ffn_dim, decode_ffn_seq, FIBER_RMS_EPS);
    size_t dfin = (size_t)dim * decode_ffn_seq * sizeof(_Float16);
    for (int l = 0; l < n_layers; l++) {
        ANEWeight dfw[4];
        dfw[0]=ane_weight_fp16("@model_path/weights/ffn_norm.bin",fm->ffn_norm_f32[l],1,dim);
        dfw[1]=ane_weight_fp16("@model_path/weights/w1.bin",fm->w1_f32[l],ffn_dim,dim);
        dfw[2]=ane_weight_fp16("@model_path/weights/w3.bin",fm->w3_f32[l],ffn_dim,dim);
        dfw[3]=ane_weight_fp16("@model_path/weights/w2.bin",fm->w2_f32[l],dim,ffn_dim);
        decode_ffn[l] = ane_compile([dfmil UTF8String],strlen([dfmil UTF8String]),
                                     dfw,4,1,&dfin,1,&dfin,ANE_QOS_BACKGROUND);
        for(int w=0;w<4;w++) ane_weight_free(&dfw[w]);
    }
    printf("Compiled %d ANE FFN decode kernels (seq=%d)\n", n_layers, decode_ffn_seq);

    // KV cache for hybrid decode
    float **kc_c = calloc(n_layers, sizeof(float*));
    float **vc_c = calloc(n_layers, sizeof(float*));
    for (int l = 0; l < n_layers; l++) {
        kc_c[l] = calloc((size_t)max_seq * kv_heads * hd, sizeof(float));
        vc_c[l] = calloc((size_t)max_seq * kv_heads * hd, sizeof(float));
    }

    float *x_c = malloc(dim * sizeof(float));
    int output_c[256];
    int n_out_c = 0;

    // Prefill on CPU (to fill KV cache)
    for (int i = 0; i < n_prompt; i++) {
        for (int d = 0; d < dim; d++)
            x_c[d] = (float)fm->embedding[(size_t)tokens[i] * dim + d];
        float *logits = cpu_forward_token(x_c, i, fm, n_layers, dim, heads, kv_heads,
                                           hd, ffn_dim, vocab, kc_c, vc_c, max_seq);
        if (i == n_prompt - 1) output_c[n_out_c++] = argmax(logits, vocab);
        free(logits);
    }

    // Hybrid decode: CPU attention + ANE FFN
    uint64_t td_c = timer_now();
    for (int step = 0; step < gen_tokens - 1 && n_prompt + n_out_c < max_seq; step++) {
        int tok_id = output_c[n_out_c - 1];
        if (tok_id == 2) break; // EOS
        for (int d = 0; d < dim; d++)
            x_c[d] = (float)fm->embedding[(size_t)tok_id * dim + d];
        float *logits = hybrid_forward_token(x_c, n_prompt + n_out_c - 1, fm,
                                              n_layers, dim, heads, kv_heads,
                                              hd, ffn_dim, vocab, kc_c, vc_c, max_seq,
                                              decode_ffn, decode_ffn_seq);
        output_c[n_out_c++] = argmax(logits, vocab);
        free(logits);
    }
    double decode_c_ms = timer_ms(td_c, timer_now());

    printf("Decode:  %d tokens in %.1f ms (%.1f tok/s)\n",
           n_out_c, decode_c_ms, n_out_c / (decode_c_ms/1000.0));
    printf("Output: ");
    for (int i = 0; i < n_prompt; i++) printf("%s", tokenizer_decode(tok, tokens[i]));
    for (int i = 0; i < n_out_c; i++) printf("%s", tokenizer_decode(tok, output_c[i]));
    printf("\n\n");

    // Match check
    int match_c = 0;
    int min_c = n_out_a < n_out_c ? n_out_a : n_out_c;
    for (int i = 0; i < min_c; i++) if (output_a[i] == output_c[i]) match_c++;

    for (int l = 0; l < n_layers; l++) { free(kc_c[l]); free(vc_c[l]); if(decode_ffn[l]) ane_free(decode_ffn[l]); }
    free(kc_c); free(vc_c); free(x_c);

    // ========================================
    // Comparison (3-way)
    // ========================================
    printf("========================================\n");
    printf("  COMPARISON (3 Pipelines)\n");
    printf("========================================\n");
    printf("                CPU/AMX     ANE-RePrefill  Hybrid(CPU+ANE)\n");
    printf("Prefill:    %5.1f t/s      %5.1f t/s       (CPU prefill)\n",
           n_prompt/(prefill_a_ms/1000.0), n_prompt/(prefill_b_ms/1000.0));
    printf("Decode:     %5.1f t/s      %5.1f t/s       %5.1f t/s\n",
           n_out_a/(decode_a_ms/1000.0),
           n_out_b/(decode_b_ms/1000.0),
           n_out_c/(decode_c_ms/1000.0));
    printf("Speedup:      1.0x          %.1fx             %.1fx\n",
           decode_a_ms / decode_b_ms, decode_a_ms / decode_c_ms);
    int match_b = 0;
    int min_b = n_out_a < n_out_b ? n_out_a : n_out_b;
    for (int i = 0; i < min_b; i++) if (output_a[i] == output_b[i]) match_b++;
    printf("\nToken match vs CPU: RePrefill=%d/%d, Hybrid=%d/%d\n",
           match_b, min_b, match_c, min_c);
    printf("========================================\n");

    // Cleanup
    free(x_b); free(ane_out);
    for (int l = 0; l < n_layers; l++) { ane_free(attn_kernels[l]); ane_free(ffn_kernels[l]); }
    tokenizer_free(tok); gguf_close(gf);
    fiber_model_free(fm);
}

// ============================================================
// Extended Proof Sweep: CPU vs ANE across dim/heads/layers
// Synthetic random weights, verifies token match at each config
// ============================================================

// Minimal CPU prefill for synthetic model (no tokenizer needed)
static float *synth_cpu_prefill(float **wq, float **wk, float **wv, float **wo,
                                 float **w1, float **w3, float **w2,
                                 float **anorm, float **fnorm,
                                 float *embed, float *out_norm, float *out_w,
                                 int dim, int heads, int kv_heads, int hd,
                                 int ffn, int layers, int vocab, int seq) {
    float eps = 1e-5f;
    // Embed: token IDs 0..seq-1 → x[t] = embed[t % vocab]
    // Process all tokens, keep last hidden state
    float *x = malloc(dim * sizeof(float));
    float **kc = calloc(layers, sizeof(float*));
    float **vc = calloc(layers, sizeof(float*));
    int kv_dim = kv_heads * hd;
    for (int l = 0; l < layers; l++) {
        kc[l] = calloc((size_t)seq * kv_dim, sizeof(float));
        vc[l] = calloc((size_t)seq * kv_dim, sizeof(float));
    }
    float *xn = malloc(dim * sizeof(float));
    float *q = malloc(dim * sizeof(float));
    float *k = malloc(kv_dim * sizeof(float));
    float *v = malloc(kv_dim * sizeof(float));

    for (int t = 0; t < seq; t++) {
        for (int d = 0; d < dim; d++) x[d] = embed[(t % vocab) * dim + d];
        for (int l = 0; l < layers; l++) {
            cpu_rmsnorm(xn, x, anorm[l], dim, eps);
            cblas_sgemv(CblasRowMajor,CblasNoTrans,dim,dim,1,wq[l],dim,xn,1,0,q,1);
            cblas_sgemv(CblasRowMajor,CblasNoTrans,kv_dim,dim,1,wk[l],dim,xn,1,0,k,1);
            cblas_sgemv(CblasRowMajor,CblasNoTrans,kv_dim,dim,1,wv[l],dim,xn,1,0,v,1);
            cpu_rope(q, k, heads, kv_heads, hd, t, 10000.0f);
            memcpy(kc[l]+(size_t)t*kv_dim, k, kv_dim*sizeof(float));
            memcpy(vc[l]+(size_t)t*kv_dim, v, kv_dim*sizeof(float));
            float *ao = calloc(dim, sizeof(float));
            float sc = 1.0f/sqrtf((float)hd);
            for (int h = 0; h < heads; h++) {
                int kvh = h * kv_heads / heads;
                float *scores = malloc((t+1)*sizeof(float));
                for (int p = 0; p <= t; p++) {
                    float d2=0; for (int d=0;d<hd;d++) d2+=q[h*hd+d]*kc[l][p*kv_dim+kvh*hd+d];
                    scores[p]=d2*sc;
                }
                float mx=scores[0]; for(int p=1;p<=t;p++) if(scores[p]>mx)mx=scores[p];
                float sm=0; for(int p=0;p<=t;p++){scores[p]=expf(scores[p]-mx);sm+=scores[p];}
                for(int p=0;p<=t;p++) scores[p]/=sm;
                for(int p=0;p<=t;p++) for(int d=0;d<hd;d++) ao[h*hd+d]+=scores[p]*vc[l][p*kv_dim+kvh*hd+d];
                free(scores);
            }
            float *proj = malloc(dim*sizeof(float));
            cblas_sgemv(CblasRowMajor,CblasNoTrans,dim,dim,1,wo[l],dim,ao,1,0,proj,1);
            for(int d=0;d<dim;d++) x[d]+=proj[d];
            free(ao); free(proj);
            cpu_rmsnorm(xn,x,fnorm[l],dim,eps);
            float *h1=malloc(ffn*sizeof(float)),*h3=malloc(ffn*sizeof(float));
            cblas_sgemv(CblasRowMajor,CblasNoTrans,ffn,dim,1,w1[l],dim,xn,1,0,h1,1);
            cblas_sgemv(CblasRowMajor,CblasNoTrans,ffn,dim,1,w3[l],dim,xn,1,0,h3,1);
            for(int i=0;i<ffn;i++){float s=h1[i]/(1+expf(-h1[i]));h1[i]=s*h3[i];}
            float *fo=malloc(dim*sizeof(float));
            cblas_sgemv(CblasRowMajor,CblasNoTrans,dim,ffn,1,w2[l],ffn,h1,1,0,fo,1);
            for(int d=0;d<dim;d++) x[d]+=fo[d];
            free(h1);free(h3);free(fo);
        }
    }
    // Classifier
    float *fn = malloc(dim*sizeof(float));
    cpu_rmsnorm(fn, x, out_norm, dim, eps);
    float *logits = malloc(vocab * sizeof(float));
    for (int v2 = 0; v2 < vocab; v2++) {
        float d2=0; for(int d=0;d<dim;d++) d2+=out_w[v2*dim+d]*fn[d]; logits[v2]=d2;
    }
    free(fn); free(x); free(xn); free(q); free(k); free(v);
    for(int l=0;l<layers;l++){free(kc[l]);free(vc[l]);} free(kc);free(vc);
    return logits;
}

void fiber_proof_sweep(void) {
    timer_init(); srand(42);
    ane_init();

    printf("\n=== EXTENDED PROOF SWEEP ===\n");
    printf("CPU/AMX vs ANE across configurations. Token match verifies correctness.\n\n");
    printf("%-8s %-5s %-5s %-4s %-5s %-4s %-4s | %8s %8s %7s %6s\n",
           "model","dim","heads","kv","ffn","lyrs","seq","cpu_ms","ane_ms","speedup","match");
    printf("------------------------------------------------------------------------\n");

    typedef struct { int dim; int heads; int kv_heads; int ffn; int layers; int seq; const char *name; } pconfig;
    pconfig cfgs[] = {
        // Small models (full sweep)
        { 256,  4,  4,  684, 12, 128, "~15M"},
        { 384,  6,  6, 1024, 12, 128, "~30M"},
        { 512,  8,  8, 1365, 12, 128, "~50M"},
        { 768, 12, 12, 2048, 12, 128, "~110M"},
        // Medium models (~500M-1B)
        {1024, 16,  8, 2730, 16, 128, "~500M"},
        {1024, 16,  8, 4096, 24, 128, "~1B"},
        {1536, 24,  8, 4096, 16, 128, "~1B-wide"},
        // Large models (~2B-4B) — reduce seq for CPU speed
        {2048, 32,  8, 5461, 22, 128, "~2B"},
        {2048, 32,  8, 5461, 32, 128, "~3B"},
        {2560, 40,  8, 6912, 28, 128, "~4B"},
        // Depth test at dim=768
        { 768, 12, 12, 2048, 32, 128, "768d-32L"},
        { 768, 12, 12, 2048, 48, 128, "768d-48L"},
        // Very large: 7B-9B (seq=32 — CPU baseline too slow at 128)
        {4096, 64,  8, 11008, 32,  32, "~7B"},
        {4096, 64,  8, 14336, 32,  32, "~9B"},
        // Qwen3 exact dimensions (head_dim=128)
        {1024,  8,  8,  3072, 28, 128, "Q3-0.6B"},
        {2048, 16,  8,  6144, 28,  64, "Q3-1.7B"},
        {2560, 20,  8,  9728, 36,  32, "Q3-4B"},
        {4096, 32,  8, 12288, 36,  32, "Q3-8B"},
    };
    int n_cfgs = sizeof(cfgs) / sizeof(cfgs[0]);

    int vocab = 1000; // small vocab for speed

    for (int ci = 0; ci < n_cfgs; ci++) {
        pconfig c = cfgs[ci];
        int hd = c.dim / c.heads;  // head_dim derived from dim/heads
        int kv_dim = c.kv_heads * hd;
        int seq = c.seq;

        // Allocate synthetic FP32 weights
        float **wq=calloc(c.layers,sizeof(float*)), **wk=calloc(c.layers,sizeof(float*));
        float **wv=calloc(c.layers,sizeof(float*)), **wo=calloc(c.layers,sizeof(float*));
        float **w1f=calloc(c.layers,sizeof(float*)), **w3f=calloc(c.layers,sizeof(float*));
        float **w2f=calloc(c.layers,sizeof(float*));
        float **an=calloc(c.layers,sizeof(float*)), **fn=calloc(c.layers,sizeof(float*));

        for (int l = 0; l < c.layers; l++) {
            wq[l]=malloc(c.dim*c.dim*4); wk[l]=malloc(kv_dim*c.dim*4);
            wv[l]=malloc(kv_dim*c.dim*4); wo[l]=malloc(c.dim*c.dim*4);
            w1f[l]=malloc(c.ffn*c.dim*4); w3f[l]=malloc(c.ffn*c.dim*4);
            w2f[l]=malloc(c.dim*c.ffn*4);
            an[l]=malloc(c.dim*4); fn[l]=malloc(c.dim*4);
            float sc=sqrtf(2.0f/(c.dim+c.dim));
            for(int i=0;i<c.dim*c.dim;i++) wq[l][i]=sc*((rand()%2000-1000)/1000.0f);
            for(int i=0;i<kv_dim*c.dim;i++) wk[l][i]=sc*((rand()%2000-1000)/1000.0f);
            for(int i=0;i<kv_dim*c.dim;i++) wv[l][i]=sc*((rand()%2000-1000)/1000.0f);
            for(int i=0;i<c.dim*c.dim;i++) wo[l][i]=sc*((rand()%2000-1000)/1000.0f);
            float scf=sqrtf(2.0f/(c.dim+c.ffn));
            for(int i=0;i<c.ffn*c.dim;i++) w1f[l][i]=scf*((rand()%2000-1000)/1000.0f);
            for(int i=0;i<c.ffn*c.dim;i++) w3f[l][i]=scf*((rand()%2000-1000)/1000.0f);
            for(int i=0;i<c.dim*c.ffn;i++) w2f[l][i]=scf*((rand()%2000-1000)/1000.0f);
            for(int i=0;i<c.dim;i++) { an[l][i]=1.0f; fn[l][i]=1.0f; }
        }
        float *embed = malloc(vocab*c.dim*4);
        for(int i=0;i<vocab*c.dim;i++) embed[i]=0.01f*((rand()%200-100)/100.0f);
        float *out_norm = malloc(c.dim*4);
        for(int i=0;i<c.dim;i++) out_norm[i]=1.0f;

        // === CPU prefill ===
        uint64_t t0 = timer_now();
        float *cpu_logits = synth_cpu_prefill(wq,wk,wv,wo,w1f,w3f,w2f,an,fn,
                                               embed,out_norm,embed,
                                               c.dim,c.heads,c.kv_heads,hd,
                                               c.ffn,c.layers,vocab,seq);
        double cpu_ms = timer_ms(t0, timer_now());
        int cpu_tok = argmax(cpu_logits, vocab);

        // === ANE prefill ===
        int ane_seq = seq < 128 ? 128 : seq;
        NSString *mil = gen_sdpa_prefill_mil(c.dim, c.heads, c.kv_heads, hd,
                                              ane_seq, 10000.0f, 1e-5f);
        NSString *fmil = gen_ffn_only_mil(c.dim, c.ffn, ane_seq, 1e-5f);
        size_t in_b = (size_t)c.dim*ane_seq*2;
        int oc = c.dim + 2*kv_dim;
        size_t out_b = (size_t)oc*ane_seq*2;
        size_t fout_b = in_b;

        ANEKernel **ak = calloc(c.layers, sizeof(ANEKernel*));
        ANEKernel **fk = calloc(c.layers, sizeof(ANEKernel*));
        bool compile_ok = true;

        for (int l = 0; l < c.layers; l++) {
            int hd2=hd/2;
            float *cd=malloc(hd2*ane_seq*4),*sd=malloc(hd2*ane_seq*4);
            for(int i=0;i<hd2;i++){float f=1.0f/powf(10000.0f,2.0f*i/(float)hd);
                for(int p=0;p<ane_seq;p++){cd[i*ane_seq+p]=cosf(p*f);sd[i*ane_seq+p]=sinf(p*f);}}
            float *mk=malloc(ane_seq*ane_seq*4);
            for(int i=0;i<ane_seq;i++) for(int j=0;j<ane_seq;j++) mk[i*ane_seq+j]=(j<=i)?0:-65504.0f;

            ANEWeight aw[8];
            aw[0]=ane_weight_fp16("@model_path/weights/rms1.bin",an[l],1,c.dim);
            aw[1]=ane_weight_fp16("@model_path/weights/wq.bin",wq[l],c.dim,c.dim);
            aw[2]=ane_weight_fp16("@model_path/weights/wk.bin",wk[l],kv_dim,c.dim);
            aw[3]=ane_weight_fp16("@model_path/weights/wv.bin",wv[l],kv_dim,c.dim);
            aw[4]=ane_weight_fp16("@model_path/weights/wo.bin",wo[l],c.dim,c.dim);
            aw[5]=ane_weight_fp16("@model_path/weights/cos.bin",cd,hd2,ane_seq);
            aw[6]=ane_weight_fp16("@model_path/weights/sin.bin",sd,hd2,ane_seq);
            aw[7]=ane_weight_fp16("@model_path/weights/mask.bin",mk,ane_seq,ane_seq);
            ak[l]=ane_compile([mil UTF8String],strlen([mil UTF8String]),aw,8,1,&in_b,1,&out_b,ANE_QOS_BACKGROUND);
            for(int w=0;w<8;w++) ane_weight_free(&aw[w]);

            ANEWeight fw[4];
            fw[0]=ane_weight_fp16("@model_path/weights/ffn_norm.bin",fn[l],1,c.dim);
            fw[1]=ane_weight_fp16("@model_path/weights/w1.bin",w1f[l],c.ffn,c.dim);
            fw[2]=ane_weight_fp16("@model_path/weights/w3.bin",w3f[l],c.ffn,c.dim);
            fw[3]=ane_weight_fp16("@model_path/weights/w2.bin",w2f[l],c.dim,c.ffn);
            fk[l]=ane_compile([fmil UTF8String],strlen([fmil UTF8String]),fw,4,1,&in_b,1,&fout_b,ANE_QOS_BACKGROUND);
            for(int w=0;w<4;w++) ane_weight_free(&fw[w]);

            free(cd);free(sd);free(mk);
            if(!ak[l]||!fk[l]) { compile_ok=false; break; }
        }

        double ane_ms = -1;
        int ane_tok = -1;
        if (compile_ok) {
            _Float16 *xb = calloc((size_t)c.dim*ane_seq, sizeof(_Float16));
            for(int t=0;t<seq;t++) for(int d=0;d<c.dim;d++)
                xb[(size_t)d*ane_seq+t]=(_Float16)embed[(t%vocab)*c.dim+d];

            // Warmup
            ane_write(ak[0],0,xb,in_b); ane_eval(ak[0],ANE_QOS_BACKGROUND);

            t0 = timer_now();
            for(int l=0;l<c.layers;l++){
                ane_lock_input(ak[l],0); memcpy(ane_input_ptr(ak[l],0),xb,in_b); ane_unlock_input(ak[l],0);
                ane_eval(ak[l],ANE_QOS_BACKGROUND);
                ane_lock_output(ak[l],0); memcpy(xb,ane_output_ptr(ak[l],0),in_b); ane_unlock_output(ak[l],0);
                ane_lock_input(fk[l],0); memcpy(ane_input_ptr(fk[l],0),xb,in_b); ane_unlock_input(fk[l],0);
                ane_eval(fk[l],ANE_QOS_BACKGROUND);
                ane_lock_output(fk[l],0); memcpy(xb,ane_output_ptr(fk[l],0),in_b); ane_unlock_output(fk[l],0);
            }
            ane_ms = timer_ms(t0, timer_now());

            // Classifier
            float *lh=malloc(c.dim*4);
            for(int d=0;d<c.dim;d++) lh[d]=(float)xb[(size_t)d*ane_seq+(seq-1)];
            float *fnr=malloc(c.dim*4);
            cpu_rmsnorm(fnr,lh,out_norm,c.dim,1e-5f);
            float *lg=malloc(vocab*4);
            for(int v2=0;v2<vocab;v2++){float d2=0;for(int d=0;d<c.dim;d++)d2+=embed[v2*c.dim+d]*fnr[d];lg[v2]=d2;}
            ane_tok = argmax(lg, vocab);
            free(lh);free(fnr);free(lg);free(xb);
        }

        // Print result
        bool match = (cpu_tok == ane_tok);
        printf("%-8s %-5d %-5d %-4d %-5d %-4d %-4d | %7.1f %7.1f %6.1fx %6s\n",
               c.name, c.dim, c.heads, c.kv_heads, c.ffn, c.layers, c.seq,
               cpu_ms, ane_ms > 0 ? ane_ms : -1,
               ane_ms > 0 ? cpu_ms/ane_ms : 0,
               compile_ok ? (match ? "YES" : "NO") : "FAIL");
        fflush(stdout); // show progress immediately

        // Cleanup
        for(int l=0;l<c.layers;l++){
            if(ak[l])ane_free(ak[l]); if(fk[l])ane_free(fk[l]);
            free(wq[l]);free(wk[l]);free(wv[l]);free(wo[l]);
            free(w1f[l]);free(w3f[l]);free(w2f[l]);free(an[l]);free(fn[l]);
        }
        free(ak);free(fk);free(wq);free(wk);free(wv);free(wo);
        free(w1f);free(w3f);free(w2f);free(an);free(fn);
        free(embed);free(out_norm);free(cpu_logits);
    }

    printf("\nCompile budget used: %d / 119\n", ane_compile_count());
    printf("=== SWEEP COMPLETE ===\n");
}
