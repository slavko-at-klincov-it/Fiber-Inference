// kv_cache.m — Simple KV cache using MTLBuffer per layer

#import <Metal/Metal.h>
#include "kv_cache.h"
#include <stdio.h>
#include <stdlib.h>

kv_cache_t *kv_cache_init(id<MTLDevice> device, int n_layers,
                           int n_kv_heads, int head_dim, int max_seq) {
    kv_cache_t *kv = calloc(1, sizeof(kv_cache_t));
    kv->n_layers = n_layers;
    kv->n_kv_heads = n_kv_heads;
    kv->head_dim = head_dim;
    kv->max_seq = max_seq;

    // Each cache: [n_kv_heads, max_seq, head_dim] in FP16
    size_t layer_bytes = (size_t)n_kv_heads * max_seq * head_dim * sizeof(_Float16);

    kv->k_cache = (__strong id<MTLBuffer> *)calloc(n_layers, sizeof(id<MTLBuffer>));
    kv->v_cache = (__strong id<MTLBuffer> *)calloc(n_layers, sizeof(id<MTLBuffer>));

    for (int l = 0; l < n_layers; l++) {
        kv->k_cache[l] = [device newBufferWithLength:layer_bytes
                                  options:MTLResourceStorageModeShared];
        kv->v_cache[l] = [device newBufferWithLength:layer_bytes
                                  options:MTLResourceStorageModeShared];

        // Zero-init
        memset(kv->k_cache[l].contents, 0, layer_bytes);
        memset(kv->v_cache[l].contents, 0, layer_bytes);
    }

    printf("KV Cache: %d layers, %d KV-heads, head_dim=%d, max_seq=%d (%.1f MB)\n",
           n_layers, n_kv_heads, head_dim, max_seq,
           (2.0 * n_layers * layer_bytes) / (1024.0 * 1024.0));

    return kv;
}

void kv_cache_free(kv_cache_t *kv) {
    if (!kv) return;
    // ARC handles MTLBuffer release
    free(kv->k_cache);
    free(kv->v_cache);
    free(kv);
}
