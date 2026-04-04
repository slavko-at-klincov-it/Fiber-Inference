// test_sliding.m — Test sliding window attention at seq > 256
#import <Foundation/Foundation.h>
#include <mach/mach_time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ane.h"
#include "ane_mil.h"

static mach_timebase_info_data_t tbi;
static void tinit(void) { mach_timebase_info(&tbi); }
static uint64_t tnow(void) { return mach_absolute_time(); }
static double tms(uint64_t a, uint64_t b) { return (double)(b-a)*tbi.numer/tbi.denom/1e6; }

// Build sliding window mask
static ANEWeight make_sw_mask(const char *name, int seq, int window) {
    size_t n = (size_t)seq * seq;
    float *data = malloc(n * sizeof(float));
    for (int i = 0; i < seq; i++)
        for (int j = 0; j < seq; j++) {
            bool causal = (j <= i);
            bool in_win = (window <= 0) || ((i - j) < window);
            data[i*seq+j] = (causal && in_win) ? 0.0f : -65504.0f;
        }
    ANEWeight w = ane_weight_fp16(name, data, seq, seq);
    free(data);
    return w;
}

static double test_attn(int dim, int heads, int kv, int hd, int seq, int window) {
    NSString *mil = gen_sdpa_prefill_mil(dim, heads, kv, hd, seq, 10000.0f, 1e-5f);
    int kv_dim = kv * hd, hd2 = hd/2;
    float *wf = malloc(dim*dim*4), *kf = malloc(kv_dim*dim*4), *nf = malloc(dim*4);
    for(int i=0;i<dim*dim;i++) wf[i]=0.001f*(rand()%2000-1000);
    for(int i=0;i<kv_dim*dim;i++) kf[i]=0.001f*(rand()%2000-1000);
    for(int i=0;i<dim;i++) nf[i]=1.0f;
    float *cd=malloc(hd2*seq*4),*sd=malloc(hd2*seq*4);
    for(int i=0;i<hd2;i++){float f=1.0f/powf(10000.0f,2.0f*i/(float)hd);
        for(int p=0;p<seq;p++){cd[i*seq+p]=cosf(p*f);sd[i*seq+p]=sinf(p*f);}}

    ANEWeight aw[8];
    aw[0]=ane_weight_fp16("@model_path/weights/rms1.bin",nf,1,dim);
    aw[1]=ane_weight_fp16("@model_path/weights/wq.bin",wf,dim,dim);
    aw[2]=ane_weight_fp16("@model_path/weights/wk.bin",kf,kv_dim,dim);
    aw[3]=ane_weight_fp16("@model_path/weights/wv.bin",kf,kv_dim,dim);
    aw[4]=ane_weight_fp16("@model_path/weights/wo.bin",wf,dim,dim);
    aw[5]=ane_weight_fp16("@model_path/weights/cos.bin",cd,hd2,seq);
    aw[6]=ane_weight_fp16("@model_path/weights/sin.bin",sd,hd2,seq);
    aw[7]=make_sw_mask("@model_path/weights/mask.bin",seq,window);

    int out_ch = dim + 2*kv_dim;
    size_t in_b=dim*seq*2, out_b=out_ch*seq*2;
    ANEKernel *k = ane_compile([mil UTF8String],strlen([mil UTF8String]),aw,8,1,&in_b,1,&out_b,ANE_QOS_BACKGROUND);
    for(int w2=0;w2<8;w2++) ane_weight_free(&aw[w2]);
    free(wf);free(kf);free(nf);free(cd);free(sd);

    if (!k) return -1;

    _Float16 *ibuf = calloc(dim*seq, sizeof(_Float16));
    for(int i=0;i<1000;i++) ibuf[i]=(_Float16)(0.01f*(i%100-50));
    ane_write(k,0,ibuf,in_b); ane_eval(k,ANE_QOS_BACKGROUND); // warmup

    double times[5];
    for(int r=0;r<5;r++){
        ane_write(k,0,ibuf,in_b);
        uint64_t t0=tnow();
        ane_eval(k,ANE_QOS_BACKGROUND);
        times[r]=tms(t0,tnow());
    }
    for(int i=0;i<4;i++) for(int j=i+1;j<5;j++)
        if(times[j]<times[i]){double t=times[i];times[i]=times[j];times[j]=t;}

    free(ibuf); ane_free(k);
    return times[2];
}

int main(void) {
    @autoreleasepool {
        tinit(); srand(42);
        ane_init();
        printf("=== Sliding Window Attention Test ===\n");
        printf("dim=768, heads=12, MHA, hd=64\n\n");
        printf("%-5s %-7s | %8s %10s\n", "seq", "window", "ms", "tok/s");
        printf("----------------------------------\n");

        typedef struct { int seq; int window; } tcfg;
        tcfg tests[] = {
            {128, 0},     // full causal (baseline)
            {256, 0},     // full causal
            {256, 128},   // sliding window
            {384, 0},     // full causal (SRAM stress)
            {384, 128},   // sliding window
            {384, 192},   // larger window
            {512, 0},     // full causal (known slow)
            {512, 128},   // sliding window
            {512, 256},   // larger window
        };
        int n = sizeof(tests)/sizeof(tests[0]);

        for (int i = 0; i < n; i++) {
            tcfg t = tests[i];
            double ms = test_attn(768, 12, 12, 64, t.seq, t.window);
            double tps = (ms > 0) ? (double)t.seq / (ms/1000.0) : 0;
            printf("%-5d %-7s | %7.2f %10.0f\n",
                   t.seq,
                   t.window == 0 ? "full" : [[@(t.window) stringValue] UTF8String],
                   ms > 0 ? ms : -1, tps);
            fflush(stdout);
        }

        printf("\nCompile budget: %d / 119\n", ane_compile_count());
        printf("=== DONE ===\n");
    }
    return 0;
}
