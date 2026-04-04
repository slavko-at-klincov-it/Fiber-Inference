// sweep.m — Systematic hardware benchmark for architecture design
// Measures ANE Attention, GPU FFN (MPS), and AMX (cblas) across dimensions
// No GGUF model needed — generates synthetic weights

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <Accelerate/Accelerate.h>
#include <mach/mach_time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ane.h"
#include "ane_mil.h"

// Timer
static mach_timebase_info_data_t tbi;
static void tinit(void) { mach_timebase_info(&tbi); }
static uint64_t tnow(void) { return mach_absolute_time(); }
static double tms(uint64_t a, uint64_t b) {
    return (double)(b-a) * tbi.numer / tbi.denom / 1e6;
}

// ============================================================
// Sweep 1: ANE Attention
// ============================================================

static void sweep_ane_attention(void) {
    printf("\n=== SWEEP 1: ANE ATTENTION ===\n");
    printf("%-6s %-6s %-6s %-6s | %8s %5s %10s\n",
           "dim", "heads", "hd", "seq", "ms", "spill", "GFLOPS");
    printf("----------------------------------------------\n");

    typedef struct { int dim; int heads; int hd; int seq; } config_t;
    config_t configs[] = {
        // Vary dim (heads=dim/64, head_dim=64, seq=128)
        { 256,  4, 64, 128},
        { 384,  6, 64, 128},
        { 512,  8, 64, 128},
        { 768, 12, 64, 128},
        {1024, 16, 64, 128},
        {2048, 32, 64, 128},
        // Vary head_dim (dim=512, heads=512/hd, seq=128)
        { 512, 16, 32, 128},
        { 512,  8, 64, 128},  // duplicate for consistency check
        { 512,  4,128, 128},
        // Vary seq (dim=768, heads=12, hd=64)
        { 768, 12, 64, 128},  // duplicate
        { 768, 12, 64, 256},
        { 768, 12, 64, 512},
    };
    int n_configs = sizeof(configs) / sizeof(configs[0]);

    for (int ci = 0; ci < n_configs; ci++) {
        config_t c = configs[ci];
        int kv_heads = c.heads;  // MHA for simplicity (no GQA in sweep)
        int kv_dim = kv_heads * c.hd;
        int out_ch = c.dim + 2 * kv_dim;

        // Ensure seq >= 128 (ANE constraint)
        int compile_seq = c.seq < 128 ? 128 : c.seq;

        // Generate MIL
        NSString *mil = gen_sdpa_prefill_mil(c.dim, c.heads, kv_heads, c.hd,
                                              compile_seq, 10000.0f, 1e-5f);
        const char *mil_c = [mil UTF8String];
        size_t mil_len = strlen(mil_c);

        // Generate random FP16 weights
        int n_weights_needed = 8; // rms, wq, wk, wv, wo, cos, sin, mask
        ANEWeight weights[8];

        // Simple random FP32 data for weights
        int wq_n = c.dim * c.dim;
        float *wq_data = malloc(wq_n * sizeof(float));
        for (int i = 0; i < wq_n; i++) wq_data[i] = 0.01f * ((float)(rand() % 200 - 100));

        int wk_n = kv_dim * c.dim;
        float *wk_data = malloc(wk_n * sizeof(float));
        for (int i = 0; i < wk_n; i++) wk_data[i] = 0.01f * ((float)(rand() % 200 - 100));

        weights[0] = ane_weight_fp16("@model_path/weights/rms1.bin", wq_data, 1, c.dim);
        weights[1] = ane_weight_fp16("@model_path/weights/wq.bin", wq_data, c.dim, c.dim);
        weights[2] = ane_weight_fp16("@model_path/weights/wk.bin", wk_data, kv_dim, c.dim);
        weights[3] = ane_weight_fp16("@model_path/weights/wv.bin", wk_data, kv_dim, c.dim);
        weights[4] = ane_weight_fp16("@model_path/weights/wo.bin", wq_data, c.dim, c.dim);

        // cos/sin tables
        int hd2 = c.hd / 2;
        float *cos_data = malloc(hd2 * compile_seq * sizeof(float));
        float *sin_data = malloc(hd2 * compile_seq * sizeof(float));
        for (int i = 0; i < hd2; i++) {
            float freq = 1.0f / powf(10000.0f, 2.0f * i / (float)(c.hd));
            for (int p = 0; p < compile_seq; p++) {
                cos_data[i * compile_seq + p] = cosf(p * freq);
                sin_data[i * compile_seq + p] = sinf(p * freq);
            }
        }
        weights[5] = ane_weight_fp16("@model_path/weights/cos.bin", cos_data, hd2, compile_seq);
        weights[6] = ane_weight_fp16("@model_path/weights/sin.bin", sin_data, hd2, compile_seq);

        // Causal mask
        int mask_n = compile_seq * compile_seq;
        float *mask_data = malloc(mask_n * sizeof(float));
        for (int i = 0; i < compile_seq; i++)
            for (int j = 0; j < compile_seq; j++)
                mask_data[i * compile_seq + j] = (j <= i) ? 0.0f : -65504.0f;
        weights[7] = ane_weight_fp16("@model_path/weights/mask.bin", mask_data, compile_seq, compile_seq);

        size_t in_bytes = (size_t)c.dim * compile_seq * sizeof(_Float16);
        size_t out_bytes = (size_t)out_ch * compile_seq * sizeof(_Float16);

        ANEKernel *k = ane_compile(mil_c, mil_len, weights, n_weights_needed,
                                    1, &in_bytes, 1, &out_bytes, ANE_QOS_BACKGROUND);

        if (!k) {
            printf("%-6d %-6d %-6d %-6d | %8s %5s %10s\n",
                   c.dim, c.heads, c.hd, c.seq, "FAIL", "-", "-");
        } else {
            bool spill = ane_sram_spill(k);

            // Run warmup + 5 eval
            _Float16 *in_buf = calloc(c.dim * compile_seq, sizeof(_Float16));
            for (int i = 0; i < c.dim * compile_seq && i < 1000; i++)
                in_buf[i] = (_Float16)(0.01f * (i % 100 - 50));
            _Float16 *out_buf = calloc(out_ch * compile_seq, sizeof(_Float16));

            ane_write(k, 0, in_buf, in_bytes);
            ane_eval(k, ANE_QOS_BACKGROUND); // warmup

            int runs = 5;
            double times[5];
            for (int r = 0; r < runs; r++) {
                ane_write(k, 0, in_buf, in_bytes);
                uint64_t t0 = tnow();
                ane_eval(k, ANE_QOS_BACKGROUND);
                times[r] = tms(t0, tnow());
            }

            // Median
            for (int i = 0; i < runs-1; i++)
                for (int j = i+1; j < runs; j++)
                    if (times[j] < times[i]) { double t=times[i]; times[i]=times[j]; times[j]=t; }
            double median_ms = times[runs/2];

            // FLOPS estimate: QKV(3*dim*dim*seq) + Attn(2*heads*seq*seq*hd) + Wo(dim*dim*seq)
            double flops = (double)c.seq * (4.0*c.dim*c.dim + 2.0*c.heads*c.seq*c.hd);
            double gflops = flops / (median_ms * 1e6);

            printf("%-6d %-6d %-6d %-6d | %7.2f %5s %9.1f\n",
                   c.dim, c.heads, c.hd, c.seq,
                   median_ms, spill ? "YES" : "no", gflops);

            free(in_buf); free(out_buf);
            ane_free(k);
        }

        // Cleanup
        for (int i = 0; i < n_weights_needed; i++) ane_weight_free(&weights[i]);
        free(wq_data); free(wk_data); free(cos_data); free(sin_data); free(mask_data);
    }
}

// ============================================================
// Sweep 2: GPU FFN (MPS MatrixMultiplication)
// ============================================================

static void sweep_gpu_ffn(id<MTLDevice> dev, id<MTLCommandQueue> queue) {
    printf("\n=== SWEEP 2: GPU FFN (MPS) ===\n");
    printf("%-6s %-6s %-5s %-6s | %8s %10s\n",
           "dim", "ffn", "ratio", "seq", "ms", "GFLOPS");
    printf("----------------------------------------------\n");

    typedef struct { int dim; int ffn; int seq; } fconfig_t;
    fconfig_t configs[] = {
        // Vary FFN ratio (dim=768, seq=128)
        { 768, 1536, 128},  // 2.0x
        { 768, 2048, 128},  // 2.67x
        { 768, 2304, 128},  // 3.0x
        { 768, 3072, 128},  // 4.0x
        { 768, 4608, 128},  // 6.0x
        // Vary dim (ratio=3x, seq=128)
        { 256,  768, 128},
        { 384, 1152, 128},
        { 512, 1536, 128},
        { 768, 2304, 128},  // duplicate
        {1024, 3072, 128},
        {2048, 6144, 128},
        // Vary seq (dim=768, ffn=2304, ratio=3x)
        { 768, 2304,  32},
        { 768, 2304,  64},
        { 768, 2304, 128},  // duplicate
        { 768, 2304, 256},
        { 768, 2304, 512},
    };
    int n_configs = sizeof(configs) / sizeof(configs[0]);

    for (int ci = 0; ci < n_configs; ci++) {
        @autoreleasepool {
            fconfig_t c = configs[ci];
            float ratio = (float)c.ffn / c.dim;

            // Allocate buffers
            id<MTLBuffer> buf_in  = [dev newBufferWithLength:c.dim * c.seq * 2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> buf_w1  = [dev newBufferWithLength:c.ffn * c.dim * 2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> buf_w3  = [dev newBufferWithLength:c.ffn * c.dim * 2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> buf_w2  = [dev newBufferWithLength:c.dim * c.ffn * 2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> buf_h1  = [dev newBufferWithLength:c.ffn * c.seq * 2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> buf_h3  = [dev newBufferWithLength:c.ffn * c.seq * 2 options:MTLResourceStorageModeShared];
            id<MTLBuffer> buf_out = [dev newBufferWithLength:c.dim * c.seq * 2 options:MTLResourceStorageModeShared];

            // Fill with random FP16
            _Float16 *p = buf_w1.contents;
            for (int i = 0; i < c.ffn * c.dim; i++) p[i] = (_Float16)(0.01f * (rand() % 200 - 100));
            p = buf_w3.contents;
            for (int i = 0; i < c.ffn * c.dim; i++) p[i] = (_Float16)(0.01f * (rand() % 200 - 100));
            p = buf_w2.contents;
            for (int i = 0; i < c.dim * c.ffn; i++) p[i] = (_Float16)(0.01f * (rand() % 200 - 100));
            p = buf_in.contents;
            for (int i = 0; i < c.dim * c.seq; i++) p[i] = (_Float16)(0.01f * (rand() % 200 - 100));

            // MPS objects
            MPSMatrixDescriptor *d_in = [MPSMatrixDescriptor matrixDescriptorWithRows:c.dim columns:c.seq rowBytes:c.seq*2 dataType:MPSDataTypeFloat16];
            MPSMatrixDescriptor *d_h  = [MPSMatrixDescriptor matrixDescriptorWithRows:c.ffn columns:c.seq rowBytes:c.seq*2 dataType:MPSDataTypeFloat16];
            MPSMatrixDescriptor *d_wh = [MPSMatrixDescriptor matrixDescriptorWithRows:c.ffn columns:c.dim rowBytes:c.dim*2 dataType:MPSDataTypeFloat16];
            MPSMatrixDescriptor *d_wd = [MPSMatrixDescriptor matrixDescriptorWithRows:c.dim columns:c.ffn rowBytes:c.ffn*2 dataType:MPSDataTypeFloat16];

            MPSMatrix *m_in  = [[MPSMatrix alloc] initWithBuffer:buf_in descriptor:d_in];
            MPSMatrix *m_w1  = [[MPSMatrix alloc] initWithBuffer:buf_w1 descriptor:d_wh];
            MPSMatrix *m_w3  = [[MPSMatrix alloc] initWithBuffer:buf_w3 descriptor:d_wh];
            MPSMatrix *m_w2  = [[MPSMatrix alloc] initWithBuffer:buf_w2 descriptor:d_wd];
            MPSMatrix *m_h1  = [[MPSMatrix alloc] initWithBuffer:buf_h1 descriptor:d_h];
            MPSMatrix *m_h3  = [[MPSMatrix alloc] initWithBuffer:buf_h3 descriptor:d_h];
            MPSMatrix *m_out = [[MPSMatrix alloc] initWithBuffer:buf_out descriptor:d_in];

            MPSMatrixMultiplication *mul_hd = [[MPSMatrixMultiplication alloc]
                initWithDevice:dev resultRows:c.ffn resultColumns:c.seq interiorColumns:c.dim];
            MPSMatrixMultiplication *mul_dh = [[MPSMatrixMultiplication alloc]
                initWithDevice:dev resultRows:c.dim resultColumns:c.seq interiorColumns:c.ffn];

            // Warmup
            {
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                [mul_hd encodeToCommandBuffer:cb leftMatrix:m_w1 rightMatrix:m_in resultMatrix:m_h1];
                [cb commit]; [cb waitUntilCompleted];
            }

            // Measure: W1@x + W3@x + W2@silu (3 matmuls = FFN core)
            int runs = 5;
            double times[5];
            for (int r = 0; r < runs; r++) {
                id<MTLCommandBuffer> cb = [queue commandBuffer];
                [mul_hd encodeToCommandBuffer:cb leftMatrix:m_w1 rightMatrix:m_in resultMatrix:m_h1];
                [mul_hd encodeToCommandBuffer:cb leftMatrix:m_w3 rightMatrix:m_in resultMatrix:m_h3];
                [mul_dh encodeToCommandBuffer:cb leftMatrix:m_w2 rightMatrix:m_h1 resultMatrix:m_out];
                uint64_t t0 = tnow();
                [cb commit]; [cb waitUntilCompleted];
                times[r] = tms(t0, tnow());
            }

            // Median
            for (int i = 0; i < runs-1; i++)
                for (int j = i+1; j < runs; j++)
                    if (times[j] < times[i]) { double t=times[i]; times[i]=times[j]; times[j]=t; }
            double median_ms = times[runs/2];

            // FLOPS: W1(ffn*dim*seq) + W3(ffn*dim*seq) + W2(dim*ffn*seq) = seq*(2*ffn*dim + dim*ffn)
            double flops = (double)c.seq * (3.0 * c.ffn * c.dim) * 2.0; // ×2 for MAC
            double gflops = flops / (median_ms * 1e6);

            printf("%-6d %-6d %-5.1f %-6d | %7.2f %9.1f\n",
                   c.dim, c.ffn, ratio, c.seq, median_ms, gflops);
        }
    }
}

// ============================================================
// Sweep 3: AMX (cblas_sgemm)
// ============================================================

static void sweep_amx(void) {
    printf("\n=== SWEEP 3: AMX (cblas_sgemm FP32) ===\n");
    printf("%-6s %-6s %-6s | %8s %10s\n",
           "M", "K", "N", "ms", "GFLOPS");
    printf("----------------------------------------------\n");

    typedef struct { int M; int K; int N; } mconfig_t;
    mconfig_t configs[] = {
        // FFN-like: W1@x = [ffn, dim] × [dim, seq]
        // dim=768, vary ffn and seq
        {2304,  768, 128},  // 3x ratio, seq=128
        {1536,  768, 128},  // 2x
        {3072,  768, 128},  // 4x
        {4608,  768, 128},  // 6x
        // Vary dim
        { 768,  256, 128},
        {1152,  384, 128},
        {1536,  512, 128},
        {2304,  768, 128},  // dup
        {3072, 1024, 128},
        {6144, 2048, 128},
        // Vary seq
        {2304,  768,  32},
        {2304,  768,  64},
        {2304,  768, 128},  // dup
        {2304,  768, 256},
        {2304,  768, 512},
    };
    int n_configs = sizeof(configs) / sizeof(configs[0]);

    for (int ci = 0; ci < n_configs; ci++) {
        mconfig_t c = configs[ci];

        float *A = malloc((size_t)c.M * c.K * sizeof(float));
        float *B = malloc((size_t)c.K * c.N * sizeof(float));
        float *C = malloc((size_t)c.M * c.N * sizeof(float));

        for (int i = 0; i < c.M * c.K; i++) A[i] = 0.01f * (rand() % 200 - 100);
        for (int i = 0; i < c.K * c.N; i++) B[i] = 0.01f * (rand() % 200 - 100);

        // Warmup
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    c.M, c.N, c.K, 1.0f, A, c.K, B, c.N, 0.0f, C, c.N);

        int runs = 10;
        double times[10];
        for (int r = 0; r < runs; r++) {
            uint64_t t0 = tnow();
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        c.M, c.N, c.K, 1.0f, A, c.K, B, c.N, 0.0f, C, c.N);
            times[r] = tms(t0, tnow());
        }

        // Median
        for (int i = 0; i < runs-1; i++)
            for (int j = i+1; j < runs; j++)
                if (times[j] < times[i]) { double t=times[i]; times[i]=times[j]; times[j]=t; }
        double median_ms = times[runs/2];

        double flops = 2.0 * c.M * c.K * c.N;
        double gflops = flops / (median_ms * 1e6);

        printf("%-6d %-6d %-6d | %7.3f %9.1f\n",
               c.M, c.K, c.N, median_ms, gflops);

        free(A); free(B); free(C);
    }
}

// ============================================================
// Main
// ============================================================

int main(int argc, char *argv[]) {
    @autoreleasepool {
        tinit();
        srand(42);

        printf("=== Fiber-Inference Hardware Sweep ===\n");
        printf("Apple M4 Mac Mini: 10-core GPU, 16-core ANE, AMX/SME\n\n");

        // Init ANE
        if (ane_init() != 0) {
            fprintf(stderr, "ANE init failed\n");
            return 1;
        }
        ANEDeviceInfo info = ane_device_info();
        printf("ANE: %s, %d cores\n", info.arch, info.num_cores);

        // Init GPU
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [dev newCommandQueue];
        printf("GPU: %s\n\n", [dev.name UTF8String]);

        // Run sweeps
        sweep_ane_attention();
        sweep_gpu_ffn(dev, queue);
        sweep_amx();

        printf("\n=== SWEEP COMPLETE ===\n");
    }
    return 0;
}
