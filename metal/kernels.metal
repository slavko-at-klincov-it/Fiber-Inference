// kernels.metal — Optimized Metal compute kernels for Fiber-Inference
// FP16 I/O with F32 internal accumulation, simd_sum reductions, fused dispatches

#include <metal_stdlib>
using namespace metal;

// ============================================================
// Helper: get_scale_min_k4 — shared by Q4_K and Q5_K
// ============================================================
static inline void get_scale_min_k4(int j, device const uint8_t *sc,
                                     thread float &scale, thread float &min_val) {
    if (j < 4) {
        scale = float(sc[j] & 63);
        min_val = float(sc[j + 4] & 63);
    } else {
        scale = float((sc[j + 4] & 0xF) | ((sc[j - 4] >> 6) << 4));
        min_val = float((sc[j + 4] >> 4) | ((sc[j] >> 6) << 4));
    }
}

// ============================================================
// SIMD-accelerated reduction helper
// 256 threads = 8 SIMD groups of 32. simd_sum reduces within each group,
// then 8 values are reduced via threadgroup memory (3 barriers instead of 8).
// ============================================================
static inline float simd_reduce_sum(float val, uint tid, threadgroup float *shared) {
    float simd_total = simd_sum(val);
    if ((tid & 31) == 0) shared[tid >> 5] = simd_total;  // lane 0 of each simdgroup
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Final reduction across 8 simdgroups
    if (tid < 8) {
        float v = shared[tid];
        v += simd_shuffle_xor(v, 4);
        v += simd_shuffle_xor(v, 2);
        v += simd_shuffle_xor(v, 1);
        shared[0] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return shared[0];
}

// ============================================================
// Q4_K Multi-Row MatVec — processes 4 rows per threadgroup
// Input vector loaded once into threadgroup memory, reused across rows.
// 256 threads = 4 rows × 64 threads/row (2 SIMD groups per row)
// ============================================================
kernel void q4k_matvec_mr(
    device const uint8_t *weight [[buffer(0)]],
    device const half *input      [[buffer(1)]],
    device half *output            [[buffer(2)]],
    constant int &in_dim           [[buffer(3)]],
    constant int &out_dim          [[buffer(4)]],
    uint tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    const int ROWS_PER_TG = 4;
    const int THREADS_PER_ROW = 64; // 2 SIMD groups
    int local_row = int(tid) / THREADS_PER_ROW;
    int local_tid = int(tid) % THREADS_PER_ROW;
    int row = int(tgid) * ROWS_PER_TG + local_row;

    if (row >= out_dim) return;

    const int bpr = in_dim / 256;

    // Load input into threadgroup memory (shared across 4 rows)
    threadgroup half tg_input[2048]; // max in_dim we support
    for (int i = int(tid); i < in_dim; i += 256) {
        tg_input[i] = input[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Compute dot product for this row
    device const uint8_t *rb = weight + uint64_t(row) * bpr * 144;
    float sum = 0.0f;

    // Each thread processes 4 elements per block (64 threads × 4 = 256 per block)
    for (int b = 0; b < bpr; b++) {
        device const uint8_t *bl = rb + b * 144;
        float d = float(*(device const half *)bl);
        float dm = float(*(device const half *)(bl + 2));
        device const uint8_t *sc = bl + 4, *qs = bl + 16;

        // Process 4 elements per thread (256 elements / 64 threads)
        for (int sub = 0; sub < 4; sub++) {
            int e = local_tid + sub * 64;
            int g = e/64, s = (e/32)&1, w = e%32;
            float sv, mv;
            get_scale_min_k4(g*2+s, sc, sv, mv);
            float val = d * sv * float(s==0 ? (qs[g*32+w]&0xF) : (qs[g*32+w]>>4)) - dm * mv;
            sum += val * float(tg_input[b*256+e]);
        }
    }

    // Reduce within the 64-thread row group (2 SIMD groups)
    float simd_total = simd_sum(sum);
    threadgroup float row_shared[8]; // 2 simd groups per row × 4 rows
    int simd_idx = local_tid / 32;
    if ((local_tid & 31) == 0) row_shared[local_row * 2 + simd_idx] = simd_total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (local_tid == 0) {
        float result = row_shared[local_row * 2] + row_shared[local_row * 2 + 1];
        output[row] = half(result);
    }
}

// ============================================================
// Q4_K Fused Dequant + MatVec — FP16 in, FP16 out (original, 1 row/tg)
// ============================================================
kernel void q4k_matvec(
    device const uint8_t *weight [[buffer(0)]],
    device const half *input      [[buffer(1)]],
    device half *output            [[buffer(2)]],
    constant int &in_dim           [[buffer(3)]],
    constant int &out_dim          [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    const int bpr = in_dim / 256;
    device const uint8_t *rb = weight + uint64_t(row) * bpr * 144;
    float sum = 0.0f;
    for (int b = 0; b < bpr; b++) {
        device const uint8_t *bl = rb + b * 144;
        float d = float(*(device const half *)bl);
        float dm = float(*(device const half *)(bl + 2));
        device const uint8_t *sc = bl + 4, *qs = bl + 16;
        int e = int(tid), g = e/64, s = (e/32)&1, w = e%32;
        float sv, mv;
        get_scale_min_k4(g*2+s, sc, sv, mv);
        float val = d * sv * float(s==0 ? (qs[g*32+w]&0xF) : (qs[g*32+w]>>4)) - dm * mv;
        sum += val * float(input[b*256+e]);
    }
    threadgroup float shared[8];
    float result = simd_reduce_sum(sum, tid, shared);
    if (tid == 0) output[row] = half(result);
}

// ============================================================
// Q5_K Fused Dequant + MatVec — FP16 in, FP16 out
// ============================================================
kernel void q5k_matvec(
    device const uint8_t *weight [[buffer(0)]],
    device const half *input      [[buffer(1)]],
    device half *output            [[buffer(2)]],
    constant int &in_dim           [[buffer(3)]],
    constant int &out_dim          [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    const int bpr = in_dim / 256;
    device const uint8_t *rb = weight + uint64_t(row) * bpr * 176;
    float sum = 0.0f;
    for (int b = 0; b < bpr; b++) {
        device const uint8_t *bl = rb + b * 176;
        float d = float(*(device const half *)bl);
        float dm = float(*(device const half *)(bl + 2));
        device const uint8_t *sc = bl+4, *qh = bl+16, *qs = bl+48;
        int e = int(tid), g = e/64, s = (e/32)&1, w = e%32;
        float sv, mv;
        get_scale_min_k4(g*2+s, sc, sv, mv);
        float nib = float(s==0 ? (qs[g*32+w]&0xF) : (qs[g*32+w]>>4));
        float hi = float((qh[w] >> (g*2+s)) & 1);
        float val = d * sv * (nib + hi * 16.0f) - dm * mv;
        sum += val * float(input[b*256+e]);
    }
    threadgroup float shared[8];
    float result = simd_reduce_sum(sum, tid, shared);
    if (tid == 0) output[row] = half(result);
}

// ============================================================
// Q6_K Fused Dequant + MatVec — FP16 in, FP16 out
// ============================================================
kernel void q6k_matvec(
    device const uint8_t *weight [[buffer(0)]],
    device const half *input      [[buffer(1)]],
    device half *output            [[buffer(2)]],
    constant int &in_dim           [[buffer(3)]],
    constant int &out_dim          [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    const int bpr = in_dim / 256;
    device const uint8_t *rb = weight + uint64_t(row) * bpr * 210;
    float sum = 0.0f;
    for (int b = 0; b < bpr; b++) {
        device const uint8_t *bl = rb + b * 210;
        device const uint8_t *ql = bl, *qh = bl + 128;
        device const int8_t *sc = (device const int8_t *)(bl + 192);
        float d = float(*(device const half *)(bl + 208));
        int e = int(tid);
        int hi = e/128, w128 = e%128, s32 = w128/32, l = w128%32;
        int ql_idx = hi*64 + (s32&1)*32 + l;
        float q_low = float((s32>=2) ? (ql[ql_idx]>>4) : (ql[ql_idx]&0xF));
        float q_hi = float((qh[hi*32+l] >> (s32*2)) & 3);
        float q6 = q_low + q_hi * 16.0f - 32.0f;
        float val = d * float(sc[hi*8 + s32*2 + l/16]) * q6;
        sum += val * float(input[b*256+e]);
    }
    threadgroup float shared[8];
    float result = simd_reduce_sum(sum, tid, shared);
    if (tid == 0) output[row] = half(result);
}

// ============================================================
// Q6_K MatVec with F32 output — for classifier logits only
// ============================================================
kernel void q6k_matvec_f32(
    device const uint8_t *weight [[buffer(0)]],
    device const half *input      [[buffer(1)]],
    device float *output           [[buffer(2)]],
    constant int &in_dim           [[buffer(3)]],
    constant int &out_dim          [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    const int bpr = in_dim / 256;
    device const uint8_t *rb = weight + uint64_t(row) * bpr * 210;
    float sum = 0.0f;
    for (int b = 0; b < bpr; b++) {
        device const uint8_t *bl = rb + b * 210;
        device const uint8_t *ql = bl, *qh = bl + 128;
        device const int8_t *sc = (device const int8_t *)(bl + 192);
        float d = float(*(device const half *)(bl + 208));
        int e = int(tid);
        int hi = e/128, w128 = e%128, s32 = w128/32, l = w128%32;
        int ql_idx = hi*64 + (s32&1)*32 + l;
        float q_low = float((s32>=2) ? (ql[ql_idx]>>4) : (ql[ql_idx]&0xF));
        float q_hi = float((qh[hi*32+l] >> (s32*2)) & 3);
        float q6 = q_low + q_hi * 16.0f - 32.0f;
        float val = d * float(sc[hi*8 + s32*2 + l/16]) * q6;
        sum += val * float(input[b*256+e]);
    }
    threadgroup float shared[8];
    float result = simd_reduce_sum(sum, tid, shared);
    if (tid == 0) output[row] = result;
}

// F32-output variants for Q4K and Q5K (classifier with these types)
kernel void q4k_matvec_f32(
    device const uint8_t *weight [[buffer(0)]],
    device const half *input      [[buffer(1)]],
    device float *output           [[buffer(2)]],
    constant int &in_dim           [[buffer(3)]],
    constant int &out_dim          [[buffer(4)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]])
{
    const int bpr = in_dim / 256;
    device const uint8_t *rb = weight + uint64_t(row) * bpr * 144;
    float sum = 0.0f;
    for (int b = 0; b < bpr; b++) {
        device const uint8_t *bl = rb + b * 144;
        float d = float(*(device const half *)bl);
        float dm = float(*(device const half *)(bl + 2));
        device const uint8_t *sc = bl + 4, *qs = bl + 16;
        int e = int(tid), g = e/64, s = (e/32)&1, w = e%32;
        float sv, mv;
        get_scale_min_k4(g*2+s, sc, sv, mv);
        float val = d * sv * float(s==0 ? (qs[g*32+w]&0xF) : (qs[g*32+w]>>4)) - dm * mv;
        sum += val * float(input[b*256+e]);
    }
    threadgroup float shared[8];
    float result = simd_reduce_sum(sum, tid, shared);
    if (tid == 0) output[row] = result;
}

// ============================================================
// RMSNorm — FP16 in/out, F32 weights (from model buffer)
// ============================================================
kernel void rmsnorm(
    device const half *x   [[buffer(0)]],
    device const float *w  [[buffer(1)]],
    device half *out       [[buffer(2)]],
    constant int &dim      [[buffer(3)]],
    constant float &eps    [[buffer(4)]],
    uint tid [[thread_position_in_threadgroup]])
{
    threadgroup float shared[8];
    float ss = 0.0f;
    for (int i = int(tid); i < dim; i += 256) ss += float(x[i]) * float(x[i]);
    float total = simd_reduce_sum(ss, tid, shared);
    float rrms = 1.0f / sqrt(total / float(dim) + eps);
    for (int i = int(tid); i < dim; i += 256)
        out[i] = half(float(x[i]) * rrms * w[i]);
}

// ============================================================
// Batched RMSNorm — channels-first [dim, seq] layout
// One thread per token, loops over dim
// ============================================================
kernel void rmsnorm_batch(
    device const half *x   [[buffer(0)]],   // [dim, seq] channels-first
    device const half *w   [[buffer(1)]],   // [dim] FP16 norm weights
    device half *out       [[buffer(2)]],   // [dim, seq] channels-first
    constant int &dim      [[buffer(3)]],
    constant int &seq      [[buffer(4)]],
    constant float &eps    [[buffer(5)]],
    uint t [[thread_position_in_grid]])
{
    if (int(t) >= seq) return;
    float ss = 0.0f;
    for (int d = 0; d < dim; d++) {
        float v = float(x[d * seq + t]);
        ss += v * v;
    }
    float rrms = 1.0f / sqrt(ss / float(dim) + eps);
    for (int d = 0; d < dim; d++) {
        out[d * seq + t] = half(float(x[d * seq + t]) * rrms * float(w[d]));
    }
}

// ============================================================
// SiLU Gate — FP16
// ============================================================
kernel void silu_gate(
    device const half *h1 [[buffer(0)]],
    device const half *h3 [[buffer(1)]],
    device half *out      [[buffer(2)]],
    constant int &n       [[buffer(3)]],
    uint id [[thread_position_in_grid]])
{
    if (int(id) >= n) return;
    float v = float(h1[id]);
    out[id] = half((v / (1.0f + exp(-v))) * float(h3[id]));
}

// ============================================================
// Residual Add — FP16
// ============================================================
kernel void residual_add(
    device half *x        [[buffer(0)]],
    device const half *y  [[buffer(1)]],
    constant int &n       [[buffer(2)]],
    uint id [[thread_position_in_grid]])
{
    if (int(id) >= n) return;
    x[id] = half(float(x[id]) + float(y[id]));
}

// ============================================================
// Fused RoPE for Q and K — single dispatch
// Applies RoPE to Q [n_heads * head_dim] and K [n_kv_heads * head_dim]
// Thread id covers max(n_heads, n_kv_heads) * head_dim / 2
// ============================================================
kernel void rope_qk(
    device half *Q             [[buffer(0)]],
    device half *K             [[buffer(1)]],
    constant int &n_heads      [[buffer(2)]],
    constant int &n_kv_heads   [[buffer(3)]],
    constant int &head_dim     [[buffer(4)]],
    constant int &pos          [[buffer(5)]],
    constant float &theta_base [[buffer(6)]],
    uint id [[thread_position_in_grid]])
{
    int half_hd = head_dim / 2;
    int head = int(id) / half_hd;
    int i    = int(id) % half_hd;

    float freq = 1.0f / pow(theta_base, 2.0f * float(i) / float(head_dim));
    float angle = float(pos) * freq;
    float c = cos(angle), s = sin(angle);

    // Apply to Q
    if (head < n_heads) {
        int i1 = head * head_dim + i, i2 = i1 + half_hd;
        float x1 = float(Q[i1]), x2 = float(Q[i2]);
        Q[i1] = half(x1*c - x2*s);
        Q[i2] = half(x1*s + x2*c);
    }
    // Apply to K (reuse same freq/angle computation)
    if (head < n_kv_heads) {
        int i1 = head * head_dim + i, i2 = i1 + half_hd;
        float x1 = float(K[i1]), x2 = float(K[i2]);
        K[i1] = half(x1*c - x2*s);
        K[i2] = half(x1*s + x2*c);
    }
}

// ============================================================
// Fused KV Store — stores both K and V in one dispatch
// ============================================================
kernel void kv_store_kv(
    device const half *K_src [[buffer(0)]],
    device const half *V_src [[buffer(1)]],
    device half *K_cache     [[buffer(2)]],
    device half *V_cache     [[buffer(3)]],
    constant int &n_kv_heads [[buffer(4)]],
    constant int &head_dim   [[buffer(5)]],
    constant int &max_seq    [[buffer(6)]],
    constant int &pos        [[buffer(7)]],
    uint id [[thread_position_in_grid]])
{
    int total = n_kv_heads * head_dim;
    if (int(id) >= total) return;
    int h = int(id) / head_dim, d = int(id) % head_dim;
    int cache_idx = h * max_seq * head_dim + pos * head_dim + d;
    K_cache[cache_idx] = K_src[id];
    V_cache[cache_idx] = V_src[id];
}

// ============================================================
// Attention Decode — Q FP16, KV cache FP16, output FP16
// ============================================================
#define MAX_CTX 4096

kernel void attention_decode(
    device const half *Q       [[buffer(0)]],
    device const half *K_cache [[buffer(1)]],
    device const half *V_cache [[buffer(2)]],
    device half *output        [[buffer(3)]],
    constant int &n_heads      [[buffer(4)]],
    constant int &n_kv_heads   [[buffer(5)]],
    constant int &head_dim     [[buffer(6)]],
    constant int &max_seq      [[buffer(7)]],
    constant int &seq_len      [[buffer(8)]],
    constant float &scale      [[buffer(9)]],
    uint head [[threadgroup_position_in_grid]],
    uint tid  [[thread_position_in_threadgroup]])
{
    int kv_head = int(head) * n_kv_heads / n_heads;
    int hd = head_dim, sl = seq_len;
    device const half *q = Q + int(head) * hd;
    device const half *K = K_cache + kv_head * max_seq * hd;
    device const half *V = V_cache + kv_head * max_seq * hd;

    threadgroup float scores[MAX_CTX];
    threadgroup float shared[8];

    // Phase 1: Scores
    for (int t = int(tid); t < sl; t += 256) {
        float s = 0.0f;
        for (int d = 0; d < hd; d++) s += float(q[d]) * float(K[t*hd+d]);
        scores[t] = s * scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 2: Softmax max
    float lmax = -INFINITY;
    for (int t = int(tid); t < sl; t += 256) lmax = max(lmax, scores[t]);
    float gmax = simd_reduce_sum(lmax, tid, shared);  // abuse: works for max too? No.
    // Use proper max reduction
    {
        threadgroup float smax[8];
        float sm = simd_max(lmax);
        if ((tid & 31) == 0) smax[tid >> 5] = sm;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < 8) {
            float v = smax[tid];
            v = max(v, simd_shuffle_xor(v, 4));
            v = max(v, simd_shuffle_xor(v, 2));
            v = max(v, simd_shuffle_xor(v, 1));
            smax[0] = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        gmax = smax[0];
    }

    // Softmax exp + sum
    float lsum = 0.0f;
    for (int t = int(tid); t < sl; t += 256) {
        scores[t] = exp(scores[t] - gmax);
        lsum += scores[t];
    }
    float gsum = simd_reduce_sum(lsum, tid, shared);
    float inv_sum = 1.0f / gsum;
    for (int t = int(tid); t < sl; t += 256) scores[t] *= inv_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: V weighted sum
    for (int d = int(tid); d < hd; d += 256) {
        float out = 0.0f;
        for (int t = 0; t < sl; t++) out += scores[t] * float(V[t*hd+d]);
        output[int(head) * hd + d] = half(out);
    }
}
