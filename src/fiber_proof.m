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

    // 1. Load model
    fiber_model_t *fm = fiber_model_load_blzt(ckpt_path);
    if (!fm) return;

    int dim = 768, heads = 12, kv_heads = 12, hd = 64;
    int ffn_dim = 2048, n_layers = 12, vocab = 32000, max_seq = 256;

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
    int seq = n_prompt < 128 ? 128 : n_prompt;

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

    // Classifier on last token
    float *last_h = malloc(dim * sizeof(float));
    for (int d = 0; d < dim; d++) last_h[d] = (float)x_b[(size_t)d*seq+(n_prompt-1)];
    float *norm_w = malloc(dim * sizeof(float));
    for (int d = 0; d < dim; d++) norm_w[d] = (float)fm->output_norm[d];
    float *final_n = malloc(dim * sizeof(float));
    cpu_rmsnorm(final_n, last_h, norm_w, dim, FIBER_RMS_EPS);
    float *logits_b = malloc(vocab * sizeof(float));
    for (int v = 0; v < vocab; v++) {
        float dot = 0;
        for (int d = 0; d < dim; d++) dot += (float)fm->output[(size_t)v*dim+d] * final_n[d];
        logits_b[v] = dot;
    }
    int first_tok_b = argmax(logits_b, vocab);

    printf("Output (prefill only): ");
    for (int i = 0; i < n_prompt; i++) printf("%s", tokenizer_decode(tok, tokens[i]));
    printf("%s", tokenizer_decode(tok, first_tok_b));
    printf("...\n");
    printf("Prefill: %d tokens in %.1f ms (%.1f tok/s)\n\n",
           n_prompt, prefill_b_ms, n_prompt / (prefill_b_ms/1000.0));

    // ========================================
    // Comparison
    // ========================================
    printf("========================================\n");
    printf("  COMPARISON\n");
    printf("========================================\n");
    printf("CPU/AMX Prefill: %6.1f ms (%6.1f tok/s)\n", prefill_a_ms, n_prompt/(prefill_a_ms/1000.0));
    printf("ANE     Prefill: %6.1f ms (%6.1f tok/s)\n", prefill_b_ms, n_prompt/(prefill_b_ms/1000.0));
    printf("Speedup:         %.1fx\n", prefill_a_ms / prefill_b_ms);
    printf("\nFirst generated token:\n");
    printf("  CPU/AMX: [%d] \"%s\"\n", output_a[0], tokenizer_decode(tok, output_a[0]));
    printf("  ANE:     [%d] \"%s\"\n", first_tok_b, tokenizer_decode(tok, first_tok_b));
    printf("  Match:   %s\n", output_a[0] == first_tok_b ? "YES" : "NO");
    printf("========================================\n");

    // Cleanup
    free(last_h); free(norm_w); free(final_n); free(logits_b);
    free(x_b); free(ane_out);
    for (int l = 0; l < n_layers; l++) { ane_free(attn_kernels[l]); ane_free(ffn_kernels[l]); }
    tokenizer_free(tok); gguf_close(gf);
    fiber_model_free(fm);
}
