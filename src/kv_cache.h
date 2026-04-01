// kv_cache.h — Simple KV cache with MTLBuffer per layer

#ifndef KV_CACHE_H
#define KV_CACHE_H

#ifdef __OBJC__
#import <Metal/Metal.h>
#endif

#include <stdint.h>

typedef struct {
    int n_layers;
    int n_kv_heads;
    int head_dim;
    int max_seq;
#ifdef __OBJC__
    // ARC-managed array of MTLBuffers
    id<MTLBuffer> __strong *k_cache;  // [n_layers]
    id<MTLBuffer> __strong *v_cache;  // [n_layers]
#else
    void **k_cache;
    void **v_cache;
#endif
} kv_cache_t;

#ifdef __OBJC__
kv_cache_t *kv_cache_init(id<MTLDevice> device, int n_layers,
                           int n_kv_heads, int head_dim, int max_seq);
#endif

void kv_cache_free(kv_cache_t *kv);

#endif // KV_CACHE_H
