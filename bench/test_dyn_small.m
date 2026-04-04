#import <Foundation/Foundation.h>
#include <stdio.h>
#include "ane.h"

int main(void) {
    @autoreleasepool {
        ane_init();
        printf("=== Small Dynamic Weight Test ===\n");

        // Small dimensions to see if dynamic works at all
        int tests[][3] = {{64, 64, 32}, {128, 128, 32}, {256, 256, 32}, {256, 768, 32}, {768, 2048, 32}};
        for (int t = 0; t < 5; t++) {
            int ic = tests[t][0], oc = tests[t][1], seq = tests[t][2];
            char *mil = ane_mil_linear_dynamic(ic, oc, seq);
            if (!mil) { printf("IC=%d OC=%d SEQ=%d: MIL gen failed\n", ic, oc, seq); continue; }
            size_t in_b = (size_t)ic * (seq + oc) * 2; // FP16
            size_t out_b = (size_t)oc * seq * 2;
            ANEKernel *k = ane_compile(mil, strlen(mil), NULL, 0, 1, &in_b, 1, &out_b, ANE_QOS_BACKGROUND);
            free(mil);
            printf("IC=%-4d OC=%-4d SEQ=%-3d in=%.1fKB → %s\n",
                   ic, oc, seq, in_b/1024.0, k ? "OK" : "FAIL");
            if (k) ane_free(k);
        }
        printf("Budget: %d/119\n", ane_compile_count());
    }
    return 0;
}
