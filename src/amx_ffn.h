// amx_ffn.h — AMX/Accelerate FFN pipeline for Fiber-768
// Uses cblas_sgemm (dispatches to AMX hardware) for batched matmul
#ifndef AMX_FFN_H
#define AMX_FFN_H

#include "fiber_arch.h"

// Forward FFN for one layer using AMX (cblas_sgemm).
// x: [dim, seq] FP16 channels-first — modified in-place (residual added).
// w1, w3: [ffn_dim, dim] FP16. w2: [dim, ffn_dim] FP16.
// ffn_norm: [dim] FP16. eps: RMSNorm epsilon.
void amx_forward_ffn_batch(_Float16 *x, int dim, int ffn_dim, int seq,
                            const _Float16 *w1, const _Float16 *w3,
                            const _Float16 *w2, const _Float16 *ffn_norm,
                            float eps);

// FP32 weights version — no per-call conversion needed (pre-converted at init)
void amx_forward_ffn_batch_f32(_Float16 *x, int dim, int ffn_dim, int seq,
                                const float *w1, const float *w3,
                                const float *w2, const float *ffn_norm,
                                float eps);

#endif
