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
#include "timer.h"

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
                case 'h': print_usage(argv[0]); return 0;
                default:  print_usage(argv[0]); return 1;
            }
        }

        if (!model_path) {
            fprintf(stderr, "Error: --model is required\n");
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
