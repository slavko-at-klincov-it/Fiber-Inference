// test_dynamic.m — Test dynamic weights on ANE (compile-once, swap per layer)
#import <Foundation/Foundation.h>
#include <mach/mach_time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ane.h"

static mach_timebase_info_data_t tbi;
static void tinit(void) { mach_timebase_info(&tbi); }
static uint64_t tnow(void) { return mach_absolute_time(); }
static double tms(uint64_t a, uint64_t b) { return (double)(b-a)*tbi.numer/tbi.denom/1e6; }

int main(void) {
    @autoreleasepool {
        tinit(); srand(42);
        ane_init();
        printf("=== Dynamic Weights Test ===\n\n");

        int dim = 768, ffn = 2048, seq = 128;

        // Use libane's built-in dynamic matmul generator
        // W1: [ffn, dim] @ [dim, seq] → [ffn, seq]
        //   IC=dim, OC=ffn, input [1, dim, 1, seq+ffn]
        char *mil_w1 = ane_mil_linear_dynamic(dim, ffn, seq);
        if (!mil_w1) { printf("MIL gen failed\n"); return 1; }

        size_t in_bytes_w1 = (size_t)dim * (seq + ffn) * sizeof(_Float16);
        size_t out_bytes_w1 = (size_t)ffn * seq * sizeof(_Float16);

        printf("Compiling dynamic W1 kernel (in=%zu, out=%zu)...\n", in_bytes_w1, out_bytes_w1);
        ANEKernel *k_w1 = ane_compile(mil_w1, strlen(mil_w1), NULL, 0,
                                       1, &in_bytes_w1, 1, &out_bytes_w1, ANE_QOS_BACKGROUND);
        free(mil_w1);

        if (!k_w1) { printf("Dynamic W1 compile FAILED\n"); return 1; }
        printf("Dynamic W1 compile OK!\n");

        // W2: [dim, ffn] @ [ffn, seq] → [dim, seq]
        //   IC=ffn, OC=dim, input [1, ffn, 1, seq+dim]
        char *mil_w2 = ane_mil_linear_dynamic(ffn, dim, seq);
        size_t in_bytes_w2 = (size_t)ffn * (seq + dim) * sizeof(_Float16);
        size_t out_bytes_w2 = (size_t)dim * seq * sizeof(_Float16);

        ANEKernel *k_w2 = ane_compile(mil_w2, strlen(mil_w2), NULL, 0,
                                       1, &in_bytes_w2, 1, &out_bytes_w2, ANE_QOS_BACKGROUND);
        free(mil_w2);
        if (!k_w2) { printf("Dynamic W2 compile FAILED\n"); return 1; }
        printf("Dynamic W2 compile OK!\n");

        // Generate random weights for 12 layers
        int n_layers = 12;
        float **w1_data = malloc(n_layers * sizeof(float*));
        float **w2_data = malloc(n_layers * sizeof(float*));
        for (int l = 0; l < n_layers; l++) {
            w1_data[l] = malloc(ffn * dim * sizeof(float));
            w2_data[l] = malloc(dim * ffn * sizeof(float));
            for (int i = 0; i < ffn*dim; i++) w1_data[l][i] = 0.001f*(rand()%2000-1000);
            for (int i = 0; i < dim*ffn; i++) w2_data[l][i] = 0.001f*(rand()%2000-1000);
        }

        // Input activation
        float *act = malloc(dim * seq * sizeof(float));
        for (int i = 0; i < dim*seq; i++) act[i] = 0.01f*(rand()%200-100);

        // Warmup
        ane_write_dynamic_weights(k_w1, 0, w1_data[0], dim, ffn, seq);
        ane_eval(k_w1, ANE_QOS_BACKGROUND);

        // Benchmark: 12 layers of W1 matmul with dynamic weights
        printf("\n--- 12-Layer Dynamic W1 Matmul ---\n");
        uint64_t t0 = tnow();
        for (int l = 0; l < n_layers; l++) {
            ane_write_dynamic_weights(k_w1, 0, w1_data[l], dim, ffn, seq);
            ane_eval(k_w1, ANE_QOS_BACKGROUND);
        }
        double total_w1 = tms(t0, tnow());
        printf("12 layers W1: %.1f ms (%.2f ms/layer)\n", total_w1, total_w1/n_layers);

        // Benchmark: 12 layers of W2 matmul
        printf("\n--- 12-Layer Dynamic W2 Matmul ---\n");
        ane_write_dynamic_weights(k_w2, 0, w2_data[0], ffn, dim, seq);
        ane_eval(k_w2, ANE_QOS_BACKGROUND); // warmup

        t0 = tnow();
        for (int l = 0; l < n_layers; l++) {
            ane_write_dynamic_weights(k_w2, 0, w2_data[l], ffn, dim, seq);
            ane_eval(k_w2, ANE_QOS_BACKGROUND);
        }
        double total_w2 = tms(t0, tnow());
        printf("12 layers W2: %.1f ms (%.2f ms/layer)\n", total_w2, total_w2/n_layers);

        printf("\n--- Total ---\n");
        printf("W1 + W2 per layer: %.2f ms\n", (total_w1+total_w2)/n_layers);
        printf("Compile budget: %d / 119 (only 2 kernels for ALL layers!)\n", ane_compile_count());

        // Cleanup
        ane_free(k_w1); ane_free(k_w2);
        for (int l = 0; l < n_layers; l++) { free(w1_data[l]); free(w2_data[l]); }
        free(w1_data); free(w2_data); free(act);

        printf("\n=== DONE ===\n");
    }
    return 0;
}
