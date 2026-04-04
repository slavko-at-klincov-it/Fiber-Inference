// main.m — Fiber-Inference CLI: GPU-only inference (Phase 1)
// Token-by-token generation with layer-by-layer mmap prefetch/evict

#import <Foundation/Foundation.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>

#include "model.h"
#include "tokenizer.h"
#include "sampler.h"
#include "kv_cache.h"
#include "gpu_ffn.h"
#include "ane_attn.h"
#include "ane.h"
#include "ane_mil.h"
#include "fiber_model.h"
#include "fiber_ckpt.h"
#include "amx_ffn.h"
#include "timer.h"
#include <Accelerate/Accelerate.h>

#define MAX_PROMPT_TOKENS 4096

static void print_usage(const char *prog) {
    fprintf(stderr, "Usage: %s --model <path.gguf> [options]\n", prog);
    fprintf(stderr, "\nOptions:\n");
    fprintf(stderr, "  --model <path>      Path to GGUF model file (required)\n");
    fprintf(stderr, "  --prompt <text>     Input prompt (default: \"Hello\")\n");
    fprintf(stderr, "  --tokens <n>        Max tokens to generate (default: 128)\n");
    fprintf(stderr, "  --ctx-len <n>       Max context length (default: model max)\n");
    fprintf(stderr, "  --temperature <f>   Sampling temperature (default: 0.7)\n");
    fprintf(stderr, "  --top-k <n>         Top-k sampling (default: 40)\n");
    fprintf(stderr, "  --top-p <f>         Top-p (nucleus) sampling (default: 0.9)\n");
    fprintf(stderr, "  --list-tensors      List all tensors and exit\n");
    fprintf(stderr, "  --benchmark         Print per-token timing\n");
    fprintf(stderr, "  --help              Show this help\n");
}

static void list_tensors(const model_t *m) {
    printf("\n=== Tensor Index (%llu tensors) ===\n", m->gf->n_tensors);
    for (uint64_t i = 0; i < m->gf->n_tensors; i++) {
        const gguf_tensor_info_t *ti = &m->gf->tensors[i];
        char name[51];
        size_t nlen = ti->name.len < 50 ? ti->name.len : 50;
        memcpy(name, ti->name.str, nlen);
        name[nlen] = '\0';

        char shape[31];
        if (ti->n_dims == 1) snprintf(shape, 31, "[%llu]", ti->dims[0]);
        else snprintf(shape, 31, "[%llu, %llu]", ti->dims[0], ti->dims[1]);

        printf("  %-45s %-6s %s\n", name, ggml_type_name(ti->type), shape);
    }
}

// Process one token: embed + all layers + classifier in one GPU submission
static void forward_token(gpu_context_t *gpu, const model_t *m,
                          kv_cache_t *kv, int token, int pos) {
    gpu_embed(gpu, m, token);
    gpu_forward_token(gpu, m, kv, pos);
}

// ============================================================
// Fiber-768 Architecture Benchmark (synthetic model, no GGUF)
// ============================================================
static void run_fiber768_bench(const char *ckpt_path) {
    timer_init();
    printf("\n=== Fiber-768 Architecture Benchmark ===\n");

    // 1. Create or load model
    uint64_t t0 = timer_now();
    fiber_model_t *fm;
    int actual_layers, actual_dim, actual_ffn, actual_heads, actual_kv_heads;

    if (ckpt_path) {
        fm = fiber_model_load_blzt(ckpt_path);
        if (!fm) return;
        // Stories-110M: dim=768, hidden=2048, heads=12, kv=12 (MHA), 12 layers
        actual_layers = 12; actual_dim = 768; actual_ffn = 2048;
        actual_heads = 12; actual_kv_heads = 12;
    } else {
        fm = fiber_model_create();
        actual_layers = FIBER_LAYERS; actual_dim = FIBER_DIM; actual_ffn = FIBER_FFN_DIM;
        actual_heads = FIBER_HEADS; actual_kv_heads = FIBER_KV_HEADS;
    }
    printf("dim=%d, heads=%d (kv=%d), ffn=%d, layers=%d, max_seq=%d\n\n",
           actual_dim, actual_heads, actual_kv_heads, actual_ffn,
           actual_layers, FIBER_MAX_SEQ);
    printf("Model ready in %.1f ms\n", timer_ms(t0, timer_now()));

    // 2. Init ANE
    if (ane_init() != 0) { fprintf(stderr, "ANE init failed\n"); return; }
    ANEDeviceInfo info = ane_device_info();
    printf("ANE: %s, %d cores\n", info.arch, info.num_cores);

    // 3. Compile ANE attention kernels
    int seq = FIBER_MAX_SEQ;
    int dim = actual_dim;
    int kv_dim = actual_kv_heads * FIBER_HEAD_DIM;
    int out_ch = dim + 2 * kv_dim;
    int n_layers_use = actual_layers;

    printf("\nCompiling ANE kernels for seq=%d...\n", seq);
    t0 = timer_now();

    NSString *mil = gen_sdpa_prefill_mil(dim, actual_heads, actual_kv_heads,
                                          FIBER_HEAD_DIM, seq,
                                          FIBER_ROPE_BASE, FIBER_RMS_EPS);
    const char *mil_c = [mil UTF8String];
    size_t mil_len = strlen(mil_c);

    // Compile one kernel per layer (different weights baked in)
    ANEKernel *kernels[FIBER_LAYERS];
    size_t in_bytes = (size_t)dim * seq * sizeof(_Float16);
    size_t out_bytes_ane = (size_t)out_ch * seq * sizeof(_Float16);
    int compiled = 0;

    for (int l = 0; l < n_layers_use; l++) {
        // Build weight blobs from synthetic FP16 weights
        ANEWeight weights[8];
        weights[0] = ane_weight_fp16("@model_path/weights/rms1.bin",
                                      fm->attn_norm_f32[l], 1, dim);
        weights[1] = ane_weight_fp16("@model_path/weights/wq.bin",
                                      fm->wq_f32[l], dim, dim);
        weights[2] = ane_weight_fp16("@model_path/weights/wk.bin",
                                      fm->wk_f32[l], kv_dim, dim);
        weights[3] = ane_weight_fp16("@model_path/weights/wv.bin",
                                      fm->wv_f32[l], kv_dim, dim);
        weights[4] = ane_weight_fp16("@model_path/weights/wo.bin",
                                      fm->wo_f32[l], dim, dim);

        // cos/sin tables
        int hd2 = FIBER_HEAD_DIM / 2;
        float *cos_data = malloc(hd2 * seq * sizeof(float));
        float *sin_data = malloc(hd2 * seq * sizeof(float));
        for (int i = 0; i < hd2; i++) {
            float freq = 1.0f / powf(FIBER_ROPE_BASE, 2.0f * i / (float)FIBER_HEAD_DIM);
            for (int p = 0; p < seq; p++) {
                cos_data[i * seq + p] = cosf(p * freq);
                sin_data[i * seq + p] = sinf(p * freq);
            }
        }
        weights[5] = ane_weight_fp16("@model_path/weights/cos.bin", cos_data, hd2, seq);
        weights[6] = ane_weight_fp16("@model_path/weights/sin.bin", sin_data, hd2, seq);

        float *mask = malloc(seq * seq * sizeof(float));
        for (int i = 0; i < seq; i++)
            for (int j = 0; j < seq; j++)
                mask[i * seq + j] = (j <= i) ? 0.0f : -65504.0f;
        weights[7] = ane_weight_fp16("@model_path/weights/mask.bin", mask, seq, seq);

        kernels[l] = ane_compile(mil_c, mil_len, weights, 8,
                                  1, &in_bytes, 1, &out_bytes_ane, ANE_QOS_BACKGROUND);
        if (kernels[l]) compiled++;

        for (int w = 0; w < 8; w++) ane_weight_free(&weights[w]);
        free(cos_data); free(sin_data); free(mask);
    }
    printf("ANE: compiled %d/%d kernels in %.1f ms\n",
           compiled, n_layers_use, timer_ms(t0, timer_now()));

    if (compiled < n_layers_use) {
        fprintf(stderr, "ANE compile failed, aborting\n");
        fiber_model_free(fm);
        return;
    }

    // 4. Create test input (synthetic tokens embedded)
    _Float16 *x = calloc((size_t)dim * seq, sizeof(_Float16));
    for (int t = 0; t < seq; t++)
        for (int d = 0; d < dim; d++)
            x[(size_t)d * seq + t] = fm->embedding[(size_t)(t % FIBER_VOCAB) * dim + d];

    _Float16 *ane_out = calloc((size_t)out_ch * seq, sizeof(_Float16));

    // 5. Benchmark: ANE Attention + AMX FFN
    printf("\n--- Benchmark: ANE Attention + AMX FFN ---\n");

    // Warmup
    ane_write(kernels[0], 0, x, in_bytes);
    ane_eval(kernels[0], ANE_QOS_BACKGROUND);

    // Timed run
    double ane_total = 0, amx_total = 0;
    t0 = timer_now();

    for (int l = 0; l < n_layers_use; l++) {
        // ANE Attention
        uint64_t ta = timer_now();
        ane_lock_input(kernels[l], 0);
        memcpy(ane_input_ptr(kernels[l], 0), x, in_bytes);
        ane_unlock_input(kernels[l], 0);
        ane_eval(kernels[l], ANE_QOS_BACKGROUND);
        ane_lock_output(kernels[l], 0);
        memcpy(ane_out, ane_output_ptr(kernels[l], 0), out_bytes_ane);
        ane_unlock_output(kernels[l], 0);
        ane_total += timer_ms(ta, timer_now());

        // Attention output now includes residual
        memcpy(x, ane_out, (size_t)dim * seq * sizeof(_Float16));

        // AMX FFN
        uint64_t tf = timer_now();
        amx_forward_ffn_batch_f32(x, dim, actual_ffn, seq,
                                   fm->w1_f32[l], fm->w3_f32[l], fm->w2_f32[l],
                                   fm->ffn_norm_f32[l], FIBER_RMS_EPS);
        amx_total += timer_ms(tf, timer_now());
    }

    double total_ms = timer_ms(t0, timer_now());
    double tok_per_s = (double)seq / (total_ms / 1000.0);

    printf("\n=== Fiber-768 Results (%d tokens, %d layers) ===\n", seq, n_layers_use);
    printf("  ANE Attention: %6.1f ms (%5.2f ms/layer)\n", ane_total, ane_total/n_layers_use);
    printf("  AMX FFN:       %6.1f ms (%5.2f ms/layer)\n", amx_total, amx_total/n_layers_use);
    printf("  Other:         %6.1f ms\n", total_ms - ane_total - amx_total);
    printf("  Total:         %6.1f ms (%.1f tok/s)\n", total_ms, tok_per_s);
    printf("===============================================\n");

    // ================================================================
    // Benchmark B: ANE Attention + ANE FFN (everything on ANE)
    // ================================================================
    printf("\n--- Benchmark B: ANE Attention + ANE FFN ---\n");
    printf("Compiling ANE FFN kernels...\n");

    NSString *ffn_mil = gen_ffn_only_mil(dim, actual_ffn, seq, FIBER_RMS_EPS);
    const char *ffn_mil_c = [ffn_mil UTF8String];
    size_t ffn_mil_len = strlen(ffn_mil_c);
    size_t ffn_out_bytes = (size_t)dim * seq * sizeof(_Float16);

    ANEKernel *ffn_kernels[FIBER_LAYERS];
    int ffn_compiled = 0;
    t0 = timer_now();
    for (int l = 0; l < n_layers_use; l++) {
        ANEWeight fw[4];
        // Use FP32 versions for ane_weight_fp16 (which expects float* and converts)
        fw[0] = ane_weight_fp16("@model_path/weights/ffn_norm.bin",
                                 fm->ffn_norm_f32[l], 1, dim);
        fw[1] = ane_weight_fp16("@model_path/weights/w1.bin",
                                 fm->w1_f32[l], actual_ffn, dim);
        fw[2] = ane_weight_fp16("@model_path/weights/w3.bin",
                                 fm->w3_f32[l], actual_ffn, dim);
        fw[3] = ane_weight_fp16("@model_path/weights/w2.bin",
                                 fm->w2_f32[l], dim, actual_ffn);

        ffn_kernels[l] = ane_compile(ffn_mil_c, ffn_mil_len, fw, 4,
                                      1, &in_bytes, 1, &ffn_out_bytes, ANE_QOS_BACKGROUND);
        if (ffn_kernels[l]) ffn_compiled++;
        for (int w = 0; w < 4; w++) ane_weight_free(&fw[w]);
    }
    printf("ANE FFN: compiled %d/%d in %.1f ms (budget: %d/119)\n",
           ffn_compiled, n_layers_use, timer_ms(t0, timer_now()), ane_compile_count());

    if (ffn_compiled == n_layers_use) {
        // Reset x to embeddings
        for (int t = 0; t < seq; t++)
            for (int d = 0; d < dim; d++)
                x[(size_t)d * seq + t] = fm->embedding[(size_t)(t % FIBER_VOCAB) * dim + d];

        // Warmup
        ane_write(ffn_kernels[0], 0, x, in_bytes);
        ane_eval(ffn_kernels[0], ANE_QOS_BACKGROUND);

        double ane_attn2 = 0, ane_ffn2 = 0;
        t0 = timer_now();

        for (int l = 0; l < n_layers_use; l++) {
            // ANE Attention
            uint64_t ta = timer_now();
            ane_lock_input(kernels[l], 0);
            memcpy(ane_input_ptr(kernels[l], 0), x, in_bytes);
            ane_unlock_input(kernels[l], 0);
            ane_eval(kernels[l], ANE_QOS_BACKGROUND);
            ane_lock_output(kernels[l], 0);
            memcpy(ane_out, ane_output_ptr(kernels[l], 0), out_bytes_ane);
            ane_unlock_output(kernels[l], 0);
            ane_attn2 += timer_ms(ta, timer_now());

            // Attention output now includes residual: oo = x + attn(x)
            // Extract x_after_attn from first dim channels of ane_out
            memcpy(x, ane_out, (size_t)dim * seq * sizeof(_Float16));

            // ANE FFN (output includes residual: out = x + ffn(x))
            uint64_t tf = timer_now();
            ane_lock_input(ffn_kernels[l], 0);
            memcpy(ane_input_ptr(ffn_kernels[l], 0), x, in_bytes);
            ane_unlock_input(ffn_kernels[l], 0);
            ane_eval(ffn_kernels[l], ANE_QOS_BACKGROUND);
            ane_lock_output(ffn_kernels[l], 0);
            memcpy(x, ane_output_ptr(ffn_kernels[l], 0), ffn_out_bytes);
            ane_unlock_output(ffn_kernels[l], 0);
            ane_ffn2 += timer_ms(tf, timer_now());
        }

        double total2 = timer_ms(t0, timer_now());
        printf("\n=== Fiber-768 ANE-Only Results (%d tokens, %d layers) ===\n", seq, n_layers_use);
        printf("  ANE Attention: %6.1f ms (%5.2f ms/layer)\n", ane_attn2, ane_attn2/n_layers_use);
        printf("  ANE FFN:       %6.1f ms (%5.2f ms/layer)\n", ane_ffn2, ane_ffn2/n_layers_use);
        printf("  Other:         %6.1f ms\n", total2 - ane_attn2 - ane_ffn2);
        printf("  Total:         %6.1f ms (%.1f tok/s)\n", total2, (double)seq/(total2/1000.0));
        printf("====================================================\n");

        // === Text Generation (if real weights loaded) ===
        if (ckpt_path && fm->output) {
            // Extract last token hidden state from x [dim, seq] channels-first
            float *last_hidden = malloc(dim * sizeof(float));
            for (int d = 0; d < dim; d++)
                last_hidden[d] = (float)x[(size_t)d * seq + (seq - 1)];

            // Final RMSNorm
            float ss = 0;
            for (int d = 0; d < dim; d++) ss += last_hidden[d] * last_hidden[d];
            float rrms = 1.0f / sqrtf(ss / dim + FIBER_RMS_EPS);
            float *norm_w = malloc(dim * sizeof(float));
            for (int d = 0; d < dim; d++) norm_w[d] = (float)fm->output_norm[d];
            for (int d = 0; d < dim; d++) last_hidden[d] *= rrms * norm_w[d];
            free(norm_w);

            // Classifier: logits = embedding^T @ last_hidden (vocab_size logits)
            int vocab = FIBER_VOCAB;
            float *logits = calloc(vocab, sizeof(float));
            // embedding is [vocab, dim] FP16. Compute dot product per vocab entry.
            for (int v = 0; v < vocab; v++) {
                float dot = 0;
                for (int d = 0; d < dim; d++)
                    dot += (float)fm->output[(size_t)v * dim + d] * last_hidden[d];
                logits[v] = dot;
            }

            // Top-5 tokens
            printf("\n--- Text Generation (greedy from last token) ---\n");
            printf("Top-5 logits: ");
            for (int k = 0; k < 5; k++) {
                int best = 0;
                for (int v = 1; v < vocab; v++)
                    if (logits[v] > logits[best]) best = v;
                printf("[%d]=%.2f ", best, logits[best]);
                logits[best] = -1e9f; // mask for next iteration
            }
            printf("\n");
            free(logits); free(last_hidden);
        }

        for (int l = 0; l < n_layers_use; l++) ane_free(ffn_kernels[l]);
    }

    // Cleanup
    for (int l = 0; l < n_layers_use; l++) ane_free(kernels[l]);
    free(x); free(ane_out);
    fiber_model_free(fm);
}

int main(int argc, char *argv[]) {
    @autoreleasepool {
        const char *model_path = NULL;
        const char *prompt = "Hello";
        int max_gen_tokens = 128;
        uint32_t ctx_len = 0;
        float temperature = 0.7f;
        int top_k = 40;
        float top_p = 0.9f;
        int list_tensors_flag = 0;
        int benchmark_flag = 0;
        int no_ane_flag = 0;
        const char *arch_mode = NULL;

        static struct option long_opts[] = {
            {"model",        required_argument, 0, 'm'},
            {"prompt",       required_argument, 0, 'p'},
            {"tokens",       required_argument, 0, 'n'},
            {"ctx-len",      required_argument, 0, 'c'},
            {"temperature",  required_argument, 0, 'T'},
            {"top-k",        required_argument, 0, 'k'},
            {"top-p",        required_argument, 0, 'P'},
            {"list-tensors", no_argument,       0, 't'},
            {"benchmark",    no_argument,       0, 'b'},
            {"no-ane",       no_argument,       0, 'G'},
            {"arch",         required_argument, 0, 'A'},
            {"help",         no_argument,       0, 'h'},
            {0, 0, 0, 0}
        };

        int opt;
        while ((opt = getopt_long(argc, argv, "m:p:n:c:T:k:P:tbh", long_opts, NULL)) != -1) {
            switch (opt) {
                case 'm': model_path = optarg; break;
                case 'p': prompt = optarg; break;
                case 'n': max_gen_tokens = atoi(optarg); break;
                case 'c': ctx_len = (uint32_t)atoi(optarg); break;
                case 'T': temperature = (float)atof(optarg); break;
                case 'k': top_k = atoi(optarg); break;
                case 'P': top_p = (float)atof(optarg); break;
                case 't': list_tensors_flag = 1; break;
                case 'b': benchmark_flag = 1; break;
                case 'G': no_ane_flag = 1; break;
                case 'A': arch_mode = optarg; break;
                case 'h': print_usage(argv[0]); return 0;
                default:  print_usage(argv[0]); return 1;
            }
        }

        // Fiber-768 architecture benchmark mode
        // --arch fiber768 (synthetic) or --arch fiber768:/path/to/ckpt.bin (real weights)
        if (arch_mode && strncmp(arch_mode, "fiber768", 8) == 0) {
            const char *ckpt = strchr(arch_mode, ':');
            run_fiber768_bench(ckpt ? ckpt + 1 : NULL);
            return 0;
        }

        if (!model_path) {
            fprintf(stderr, "Error: --model or --arch fiber768 is required\n");
            print_usage(argv[0]);
            return 1;
        }

        timer_init();
        printf("Fiber-Inference v0.2 (Phase 1: GPU-Only Baseline)\n\n");

        // ======= Load model =======
        printf("Loading: %s\n", model_path);
        uint64_t t0 = timer_now();
        model_t *m = model_open(model_path);
        uint64_t t1 = timer_now();
        if (!m) { fprintf(stderr, "Failed to open model\n"); return 1; }
        printf("Model loaded in %.1f ms\n\n", timer_ms(t0, t1));
        model_print_info(m);

        if (list_tensors_flag) {
            list_tensors(m);
            model_close(m);
            return 0;
        }

        if (ctx_len == 0 || ctx_len > m->params.context_length)
            ctx_len = m->params.context_length;

        // ======= Initialize tokenizer =======
        t0 = timer_now();
        tokenizer_t *tok = tokenizer_init(m->gf);
        t1 = timer_now();
        if (!tok) { fprintf(stderr, "Failed to init tokenizer\n"); model_close(m); return 1; }
        printf("Tokenizer initialized in %.1f ms\n", timer_ms(t0, t1));

        // ======= Initialize GPU =======
        t0 = timer_now();
        gpu_context_t *gpu = gpu_init(m);
        t1 = timer_now();
        if (!gpu) { fprintf(stderr, "Failed to init GPU\n"); tokenizer_free(tok); model_close(m); return 1; }
        printf("GPU initialized in %.1f ms\n", timer_ms(t0, t1));

        // ======= Initialize ANE =======
        ane_attn_context_t *ane_ctx = NULL;
        if (no_ane_flag) {
            printf("ANE disabled (--no-ane)\n");
        } else {
        t0 = timer_now();
        ane_ctx = ane_attn_init();
        t1 = timer_now();
        if (ane_ctx) {
            printf("ANE initialized in %.1f ms\n", timer_ms(t0, t1));

            // Dequantize attention weights for ANE (Q4_K/Q6_K → FP16)
            t0 = timer_now();
            size_t dq_bytes = model_dequant_attention(m);
            t1 = timer_now();
            printf("Dequantized attention weights: %.1f MB in %.1f ms\n",
                   dq_bytes / (1024.0 * 1024.0), timer_ms(t0, t1));

            // ANE compile deferred until we know prompt length (after tokenize)
        } else {
            printf("ANE not available, using GPU-only mode\n");
        }
        } // end !no_ane_flag

        // ======= Initialize KV cache =======
        kv_cache_t *kv = kv_cache_init(gpu_get_device(gpu),
                                        m->params.n_layers,
                                        m->params.n_kv_heads,
                                        m->params.head_dim,
                                        ctx_len);
        if (!kv) {
            fprintf(stderr, "Failed to init KV cache\n");
            gpu_free(gpu); tokenizer_free(tok); model_close(m);
            return 1;
        }

        // ======= Initialize sampler =======
        sampler_t sampler;
        sampler_init(&sampler, m->params.vocab_size, temperature, top_k, top_p);

        // ======= Encode prompt =======
        int prompt_tokens[MAX_PROMPT_TOKENS];
        int n_prompt = tokenizer_encode(tok, prompt, prompt_tokens, MAX_PROMPT_TOKENS);

        if (n_prompt == 0) {
            prompt_tokens[0] = tokenizer_bos(tok);
            n_prompt = 1;
        }

        printf("\nPrompt: \"%s\" → %d tokens\n", prompt, n_prompt);

        // ANE compile for prompt length (min 128 — ANE matmul dimension constraint)
        if (ane_ctx && n_prompt > 4) {
            t0 = timer_now();
            int ane_seq = n_prompt < 128 ? 128 : n_prompt;
            bool ane_ok = ane_attn_compile(ane_ctx, m, ane_seq);
            t1 = timer_now();
            if (ane_ok) {
                printf("ANE: %d kernels for max_seq=%d in %.1f ms\n",
                       m->params.n_layers, ane_seq, timer_ms(t0, t1));
                model_free_attention_fp16(m);
                gpu_init_ffn_batch(gpu, m, ane_seq);
            } else {
                fprintf(stderr, "ANE compile failed, using GPU-only\n");
                ane_attn_free(ane_ctx);
                ane_ctx = NULL;
            }
        }

        printf("RSS after init: %.1f MB\n", model_get_rss() / (1024.0 * 1024.0));
        printf("\n--- Output ---\n");
        for (int i = 0; i < n_prompt; i++) {
            printf("%s", tokenizer_decode(tok, prompt_tokens[i]));
        }
        fflush(stdout);

        // ======= Prefill: process all prompt tokens =======
        t0 = timer_now();
        double prefill_tps = 0;
        if (ane_ctx && n_prompt > 4) {
            // ANE batched prefill: ANE attention + GPU FFN
            prefill_tps = ane_prefill_batch(ane_ctx, gpu, m, kv,
                                             prompt_tokens, n_prompt);
        } else {
            // GPU-only prefill (token by token)
            for (int i = 0; i < n_prompt; i++) {
                forward_token(gpu, m, kv, prompt_tokens[i], i);
            }
        }
        t1 = timer_now();
        double prefill_ms = timer_ms(t0, t1);

        int next_token = sampler_sample(&sampler, gpu_get_logits(gpu));

        // ======= Decode: generate tokens one by one =======
        int pos = n_prompt;
        int gen_count = 0;
        uint64_t gen_start = timer_now();

        for (int i = 0; i < max_gen_tokens && pos < (int)ctx_len; i++) {
            const char *decoded = tokenizer_decode(tok, next_token);
            printf("%s", decoded);
            fflush(stdout);
            gen_count++;

            if (next_token == tokenizer_eos(tok)) break;

            forward_token(gpu, m, kv, next_token, pos);
            next_token = sampler_sample(&sampler, gpu_get_logits(gpu));
            pos++;
        }

        uint64_t gen_end = timer_now();

        // ======= Stats =======
        printf("\n\n--- Stats ---\n");
        printf("Prefill: %d tokens in %.1f ms (%.1f tok/s)\n",
               n_prompt, prefill_ms,
               n_prompt / (prefill_ms / 1000.0));

        if (gen_count > 0) {
            double gen_ms = timer_ms(gen_start, gen_end);
            printf("Decode:  %d tokens in %.1f ms (%.1f tok/s)\n",
                   gen_count, gen_ms,
                   gen_count / (gen_ms / 1000.0));
        }

        printf("Peak RSS: %.1f MB\n", model_get_rss() / (1024.0 * 1024.0));

        kv_cache_free(kv);
        ane_attn_free(ane_ctx);
        gpu_free(gpu);
        tokenizer_free(tok);
        model_close(m);
    }
    return 0;
}
