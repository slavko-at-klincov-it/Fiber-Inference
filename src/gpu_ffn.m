// gpu_ffn.m — Optimized GPU inference: zero-copy, single encoder, FP16 buffers

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include "gpu_ffn.h"
#include "model.h"
#include "kv_cache.h"
#include "timer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

struct gpu_context {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;

    id<MTLComputePipelineState> pipe_q4k, pipe_q5k, pipe_q6k;
    id<MTLComputePipelineState> pipe_q4k_f32, pipe_q6k_f32;  // F32 output for classifier
    id<MTLComputePipelineState> pipe_rmsnorm, pipe_silu, pipe_residual;
    id<MTLComputePipelineState> pipe_rope_qk, pipe_kv_store, pipe_attention;

    id<MTLBuffer> buf_model;   // zero-copy mmap
    const void *mmap_base;

    // Working buffers — ALL FP16 except logits
    id<MTLBuffer> buf_x, buf_xnorm, buf_q, buf_k, buf_v;
    id<MTLBuffer> buf_attn_out, buf_proj, buf_h1, buf_h3, buf_silu;
    id<MTLBuffer> buf_logits;  // F32 for sampling precision

    model_params_t params;
};

static id<MTLBuffer> make_buf(id<MTLDevice> d, size_t n) {
    return [d newBufferWithLength:n options:MTLResourceStorageModeShared];
}

static id<MTLComputePipelineState> pipe_for_type(gpu_context_t *c, ggml_type_t t) {
    if (t == GGML_TYPE_Q4_K) return c->pipe_q4k;
    if (t == GGML_TYPE_Q5_K) return c->pipe_q5k;
    if (t == GGML_TYPE_Q6_K) return c->pipe_q6k;
    return c->pipe_q4k;
}

static id<MTLComputePipelineState> pipe_for_type_f32(gpu_context_t *c, ggml_type_t t) {
    if (t == GGML_TYPE_Q6_K) return c->pipe_q6k_f32;
    return c->pipe_q4k_f32;
}

static size_t woff(const gpu_context_t *c, const void *p) {
    return (const uint8_t *)p - (const uint8_t *)c->mmap_base;
}

// Encode matvec (FP16 output)
static void enc_mv(id<MTLComputeCommandEncoder> e, id<MTLComputePipelineState> p,
                   id<MTLBuffer> wb, size_t wo, id<MTLBuffer> in, id<MTLBuffer> out,
                   int id_dim, int od) {
    [e setComputePipelineState:p];
    [e setBuffer:wb offset:wo atIndex:0];
    [e setBuffer:in offset:0 atIndex:1];
    [e setBuffer:out offset:0 atIndex:2];
    [e setBytes:&id_dim length:4 atIndex:3];
    [e setBytes:&od length:4 atIndex:4];
    [e dispatchThreadgroups:MTLSizeMake(od,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
}

static void enc_rms(id<MTLComputeCommandEncoder> e, id<MTLComputePipelineState> p,
                    id<MTLBuffer> x, id<MTLBuffer> wb, size_t wo,
                    id<MTLBuffer> out, int dim, float eps) {
    [e setComputePipelineState:p];
    [e setBuffer:x offset:0 atIndex:0];
    [e setBuffer:wb offset:wo atIndex:1];
    [e setBuffer:out offset:0 atIndex:2];
    [e setBytes:&dim length:4 atIndex:3];
    [e setBytes:&eps length:4 atIndex:4];
    [e dispatchThreadgroups:MTLSizeMake(1,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
}

// CPU dequant — shared implementation in dequant.h
#include "dequant.h"

// ============================================================
// Init
// ============================================================

static id<MTLComputePipelineState> mp(id<MTLLibrary> lib, NSString *name) {
    NSError *err = nil;
    id<MTLFunction> fn = [lib newFunctionWithName:name];
    if (!fn) { fprintf(stderr, "gpu: missing '%s'\n", [name UTF8String]); return nil; }
    return [lib.device newComputePipelineStateWithFunction:fn error:&err];
}

gpu_context_t *gpu_init(const model_t *m) {
    @autoreleasepool {
        gpu_context_t *c = calloc(1, sizeof(gpu_context_t));
        c->params = m->params;
        c->mmap_base = m->gf->data;
        c->device = MTLCreateSystemDefaultDevice();
        if (!c->device) { free(c); return NULL; }
        c->queue = [c->device newCommandQueue];
        printf("GPU: %s\n", [c->device.name UTF8String]);

        NSError *err = nil;
        NSString *src = [NSString stringWithContentsOfFile:@"metal/kernels.metal"
                         encoding:NSUTF8StringEncoding error:&err];
        if (!src) { fprintf(stderr, "gpu: cannot read kernels\n"); free(c); return NULL; }
        MTLCompileOptions *opts = [[MTLCompileOptions alloc] init];
        opts.fastMathEnabled = YES;
        id<MTLLibrary> lib = [c->device newLibraryWithSource:src options:opts error:&err];
        if (!lib) { fprintf(stderr, "gpu: compile: %s\n", [[err description] UTF8String]); free(c); return NULL; }

        c->pipe_q4k      = mp(lib, @"q4k_matvec");
        c->pipe_q5k      = mp(lib, @"q5k_matvec");
        c->pipe_q6k      = mp(lib, @"q6k_matvec");
        c->pipe_q4k_f32  = mp(lib, @"q4k_matvec_f32");
        c->pipe_q6k_f32  = mp(lib, @"q6k_matvec_f32");
        c->pipe_rmsnorm   = mp(lib, @"rmsnorm");
        c->pipe_silu      = mp(lib, @"silu_gate");
        c->pipe_residual  = mp(lib, @"residual_add");
        c->pipe_rope_qk   = mp(lib, @"rope_qk");
        c->pipe_kv_store  = mp(lib, @"kv_store_kv");
        c->pipe_attention  = mp(lib, @"attention_decode");

        if (!c->pipe_q4k || !c->pipe_rmsnorm || !c->pipe_attention) {
            fprintf(stderr, "gpu: pipeline creation failed\n"); free(c); return NULL;
        }

        // Zero-copy model buffer
        size_t ps = getpagesize();
        size_t aligned = (m->gf->file_size + ps - 1) & ~(ps - 1);
        c->buf_model = [c->device newBufferWithBytesNoCopy:(void *)m->gf->data
                        length:aligned options:MTLResourceStorageModeShared deallocator:nil];

        // FP16 working buffers
        model_params_t p = c->params;
        int kv = p.n_kv_heads * p.head_dim;
        int mx = p.dim > p.ffn_dim ? p.dim : p.ffn_dim;
        c->buf_x        = make_buf(c->device, p.dim * 2);
        c->buf_xnorm    = make_buf(c->device, p.dim * 2);
        c->buf_q        = make_buf(c->device, p.dim * 2);
        c->buf_k        = make_buf(c->device, kv * 2);
        c->buf_v        = make_buf(c->device, kv * 2);
        c->buf_attn_out = make_buf(c->device, p.dim * 2);
        c->buf_proj     = make_buf(c->device, mx * 2);
        c->buf_h1       = make_buf(c->device, p.ffn_dim * 2);
        c->buf_h3       = make_buf(c->device, p.ffn_dim * 2);
        c->buf_silu     = make_buf(c->device, p.ffn_dim * 2);
        c->buf_logits   = make_buf(c->device, p.vocab_size * 4); // F32!

        printf("GPU: zero-copy %.1f MB, FP16 buffers\n", aligned / (1024.0*1024.0));
        return c;
    }
}

void gpu_free(gpu_context_t *c) { if (c) free(c); }
id<MTLDevice> gpu_get_device(gpu_context_t *c) { return c->device; }
const float *gpu_get_logits(gpu_context_t *c) { return (const float *)c->buf_logits.contents; }
const float *gpu_get_hidden(gpu_context_t *c) { return NULL; } // buf_x is FP16 now
_Float16 *gpu_get_buf_x(gpu_context_t *c) { return (_Float16 *)c->buf_x.contents; }

// FFN-only for one layer (own command buffer, synchronous)
void gpu_forward_ffn_layer(gpu_context_t *c, const model_t *m, uint32_t layer_idx) {
    @autoreleasepool {
        model_params_t p = c->params;
        int dim = p.dim, ffn = p.ffn_dim;
        float eps = p.rms_norm_eps;

        layer_weights_t lw;
        model_layer_weights(m, layer_idx, &lw);

        id<MTLCommandBuffer> cb = [c->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        // RMSNorm(x) → xnorm
        enc_rms(enc, c->pipe_rmsnorm, c->buf_x, c->buf_model, woff(c, lw.ffn_norm),
                c->buf_xnorm, dim, eps);
        // W1 matmul
        enc_mv(enc, pipe_for_type(c, lw.w1_type), c->buf_model, woff(c, lw.w1),
               c->buf_xnorm, c->buf_h1, dim, ffn);
        // W3 matmul
        enc_mv(enc, pipe_for_type(c, lw.w3_type), c->buf_model, woff(c, lw.w3),
               c->buf_xnorm, c->buf_h3, dim, ffn);
        // SiLU gate
        [enc setComputePipelineState:c->pipe_silu];
        [enc setBuffer:c->buf_h1 offset:0 atIndex:0];
        [enc setBuffer:c->buf_h3 offset:0 atIndex:1];
        [enc setBuffer:c->buf_silu offset:0 atIndex:2];
        [enc setBytes:&ffn length:4 atIndex:3];
        [enc dispatchThreads:MTLSizeMake(ffn,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        // W2 matmul
        enc_mv(enc, pipe_for_type(c, lw.w2_type), c->buf_model, woff(c, lw.w2),
               c->buf_silu, c->buf_proj, ffn, dim);
        // Residual add: x += ffn_out
        [enc setComputePipelineState:c->pipe_residual];
        [enc setBuffer:c->buf_x offset:0 atIndex:0];
        [enc setBuffer:c->buf_proj offset:0 atIndex:1];
        [enc setBytes:&dim length:4 atIndex:2];
        [enc dispatchThreads:MTLSizeMake(dim,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];

        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }
}

// Classifier-only: final RMSNorm + output matmul → logits
void gpu_forward_classifier(gpu_context_t *c, const model_t *m) {
    @autoreleasepool {
        model_params_t p = c->params;
        int dim = p.dim, vocab = p.vocab_size;
        float eps = p.rms_norm_eps;

        id<MTLCommandBuffer> cb = [c->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        enc_rms(enc, c->pipe_rmsnorm, c->buf_x, c->buf_model, woff(c, m->output_norm),
                c->buf_xnorm, dim, eps);
        [enc setComputePipelineState:pipe_for_type_f32(c, m->output_type)];
        [enc setBuffer:c->buf_model offset:woff(c, m->output) atIndex:0];
        [enc setBuffer:c->buf_xnorm offset:0 atIndex:1];
        [enc setBuffer:c->buf_logits offset:0 atIndex:2];
        [enc setBytes:&dim length:4 atIndex:3];
        [enc setBytes:&vocab length:4 atIndex:4];
        [enc dispatchThreadgroups:MTLSizeMake(vocab,1,1)
            threadsPerThreadgroup:MTLSizeMake(256,1,1)];

        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
    }
}

// ============================================================
// Embedding (CPU → FP16 buf_x)
// ============================================================

void gpu_embed(gpu_context_t *c, const model_t *m, int token_id) {
    int dim = m->params.dim;
    _Float16 *x = (_Float16 *)c->buf_x.contents;
    switch (m->token_embd_type) {
        case GGML_TYPE_F32: {
            const float *e = (const float *)m->token_embd;
            for (int i = 0; i < dim; i++) x[i] = (_Float16)e[(size_t)token_id * dim + i];
            break;
        }
        case GGML_TYPE_F16:
            memcpy(x, (const _Float16 *)m->token_embd + (size_t)token_id * dim, dim * 2);
            break;
        case GGML_TYPE_Q4_K: dequant_q4k_row_f16(m->token_embd, token_id, x, dim); break;
        case GGML_TYPE_Q6_K: dequant_q6k_row_f16(m->token_embd, token_id, x, dim); break;
        default: memset(x, 0, dim * 2); break;
    }
}

// ============================================================
// Forward: ALL layers + classifier in ONE encoder, ONE CB
// ============================================================

double gpu_forward_token(gpu_context_t *c, const model_t *m,
                         kv_cache_t *kv, int pos) {
    @autoreleasepool {
        model_params_t p = c->params;
        int dim = p.dim, kv_dim = p.n_kv_heads * p.head_dim, ffn = p.ffn_dim;
        float eps = p.rms_norm_eps, theta = p.rope_freq_base;
        int nh = p.n_heads, nkv = p.n_kv_heads, hd = p.head_dim;

        uint64_t t0 = timer_now();
        id<MTLCommandBuffer> cb = [c->queue commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];

        for (uint32_t l = 0; l < p.n_layers; l++) {
            layer_weights_t lw;
            model_layer_weights(m, l, &lw);

            // RMSNorm → xnorm
            enc_rms(enc, c->pipe_rmsnorm, c->buf_x, c->buf_model, woff(c, lw.attn_norm),
                    c->buf_xnorm, dim, eps);

            // QKV
            enc_mv(enc, pipe_for_type(c, lw.wq_type), c->buf_model, woff(c, lw.wq),
                   c->buf_xnorm, c->buf_q, dim, dim);
            enc_mv(enc, pipe_for_type(c, lw.wk_type), c->buf_model, woff(c, lw.wk),
                   c->buf_xnorm, c->buf_k, dim, kv_dim);
            enc_mv(enc, pipe_for_type(c, lw.wv_type), c->buf_model, woff(c, lw.wv),
                   c->buf_xnorm, c->buf_v, dim, kv_dim);

            // Fused RoPE Q+K
            [enc setComputePipelineState:c->pipe_rope_qk];
            [enc setBuffer:c->buf_q offset:0 atIndex:0];
            [enc setBuffer:c->buf_k offset:0 atIndex:1];
            [enc setBytes:&nh length:4 atIndex:2];
            [enc setBytes:&nkv length:4 atIndex:3];
            [enc setBytes:&hd length:4 atIndex:4];
            [enc setBytes:&pos length:4 atIndex:5];
            [enc setBytes:&theta length:4 atIndex:6];
            [enc dispatchThreads:MTLSizeMake(nh * hd / 2, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];

            // Fused KV store
            {
                int ms = kv->max_seq, ke = nkv * hd;
                [enc setComputePipelineState:c->pipe_kv_store];
                [enc setBuffer:c->buf_k offset:0 atIndex:0];
                [enc setBuffer:c->buf_v offset:0 atIndex:1];
                [enc setBuffer:kv->k_cache[l] offset:0 atIndex:2];
                [enc setBuffer:kv->v_cache[l] offset:0 atIndex:3];
                [enc setBytes:&nkv length:4 atIndex:4];
                [enc setBytes:&hd length:4 atIndex:5];
                [enc setBytes:&ms length:4 atIndex:6];
                [enc setBytes:&pos length:4 atIndex:7];
                [enc dispatchThreads:MTLSizeMake(ke, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(MIN(256, ke), 1, 1)];
            }

            // Attention
            {
                int ms = kv->max_seq, sl = pos + 1;
                float scale = 1.0f / sqrtf((float)hd);
                [enc setComputePipelineState:c->pipe_attention];
                [enc setBuffer:c->buf_q offset:0 atIndex:0];
                [enc setBuffer:kv->k_cache[l] offset:0 atIndex:1];
                [enc setBuffer:kv->v_cache[l] offset:0 atIndex:2];
                [enc setBuffer:c->buf_attn_out offset:0 atIndex:3];
                [enc setBytes:&nh length:4 atIndex:4];
                [enc setBytes:&nkv length:4 atIndex:5];
                [enc setBytes:&hd length:4 atIndex:6];
                [enc setBytes:&ms length:4 atIndex:7];
                [enc setBytes:&sl length:4 atIndex:8];
                [enc setBytes:&scale length:4 atIndex:9];
                [enc dispatchThreadgroups:MTLSizeMake(nh, 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
            }

            // Wo + residual
            enc_mv(enc, pipe_for_type(c, lw.wo_type), c->buf_model, woff(c, lw.wo),
                   c->buf_attn_out, c->buf_proj, dim, dim);
            [enc setComputePipelineState:c->pipe_residual];
            [enc setBuffer:c->buf_x offset:0 atIndex:0];
            [enc setBuffer:c->buf_proj offset:0 atIndex:1];
            [enc setBytes:&dim length:4 atIndex:2];
            [enc dispatchThreads:MTLSizeMake(dim,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];

            // FFN: RMSNorm → W1,W3 → SiLU → W2 → residual
            enc_rms(enc, c->pipe_rmsnorm, c->buf_x, c->buf_model, woff(c, lw.ffn_norm),
                    c->buf_xnorm, dim, eps);
            enc_mv(enc, pipe_for_type(c, lw.w1_type), c->buf_model, woff(c, lw.w1),
                   c->buf_xnorm, c->buf_h1, dim, ffn);
            enc_mv(enc, pipe_for_type(c, lw.w3_type), c->buf_model, woff(c, lw.w3),
                   c->buf_xnorm, c->buf_h3, dim, ffn);
            [enc setComputePipelineState:c->pipe_silu];
            [enc setBuffer:c->buf_h1 offset:0 atIndex:0];
            [enc setBuffer:c->buf_h3 offset:0 atIndex:1];
            [enc setBuffer:c->buf_silu offset:0 atIndex:2];
            [enc setBytes:&ffn length:4 atIndex:3];
            [enc dispatchThreads:MTLSizeMake(ffn,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            enc_mv(enc, pipe_for_type(c, lw.w2_type), c->buf_model, woff(c, lw.w2),
                   c->buf_silu, c->buf_proj, ffn, dim);
            [enc setComputePipelineState:c->pipe_residual];
            [enc setBuffer:c->buf_x offset:0 atIndex:0];
            [enc setBuffer:c->buf_proj offset:0 atIndex:1];
            [enc setBytes:&dim length:4 atIndex:2];
            [enc dispatchThreads:MTLSizeMake(dim,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        }

        // Classifier: RMSNorm → output projection (F32 logits)
        {
            int vocab = p.vocab_size;
            enc_rms(enc, c->pipe_rmsnorm, c->buf_x, c->buf_model, woff(c, m->output_norm),
                    c->buf_xnorm, dim, eps);
            [enc setComputePipelineState:pipe_for_type_f32(c, m->output_type)];
            [enc setBuffer:c->buf_model offset:woff(c, m->output) atIndex:0];
            [enc setBuffer:c->buf_xnorm offset:0 atIndex:1];
            [enc setBuffer:c->buf_logits offset:0 atIndex:2];
            [enc setBytes:&dim length:4 atIndex:3];
            [enc setBytes:&vocab length:4 atIndex:4];
            [enc dispatchThreadgroups:MTLSizeMake(vocab,1,1)
                threadsPerThreadgroup:MTLSizeMake(256,1,1)];
        }

        [enc endEncoding];
        [cb commit];
        [cb waitUntilCompleted];
        return timer_ms(t0, timer_now());
    }
}
