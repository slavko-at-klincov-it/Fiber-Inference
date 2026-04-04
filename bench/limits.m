// limits.m — Systematic limit testing for Fiber-768 ANE architecture
// Tests: sequence length, layer count, dimension, memory, precision
#import <Foundation/Foundation.h>
#include <mach/mach_time.h>
#include <mach/mach.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ane.h"
#include "ane_mil.h"

static mach_timebase_info_data_t tbi;
static void tinit(void) { mach_timebase_info(&tbi); }
static uint64_t tnow(void) { return mach_absolute_time(); }
static double tms(uint64_t a, uint64_t b) { return (double)(b-a)*tbi.numer/tbi.denom/1e6; }

static size_t get_rss(void) {
    struct mach_task_basic_info info;
    mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
    task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t)&info, &count);
    return info.resident_size;
}

// Compile + run one ANE attention kernel, return eval time in ms (-1 on failure)
static double test_ane_attention(int dim, int heads, int kv_heads, int head_dim,
                                  int seq, int warmup, int runs) {
    NSString *mil = gen_sdpa_prefill_mil(dim, heads, kv_heads, head_dim,
                                          seq, 10000.0f, 1e-5f);
    const char *mc = [mil UTF8String];
    size_t ml = strlen(mc);

    int kv_dim = kv_heads * head_dim;
    int out_ch = dim + 2 * kv_dim;

    // Random weights
    int hd2 = head_dim / 2;
    float *wdata = malloc(dim * dim * sizeof(float));
    float *kdata = malloc(kv_dim * dim * sizeof(float));
    for (int i = 0; i < dim*dim; i++) wdata[i] = 0.001f*(rand()%2000-1000);
    for (int i = 0; i < kv_dim*dim; i++) kdata[i] = 0.001f*(rand()%2000-1000);

    float *cos_d = malloc(hd2*seq*sizeof(float)), *sin_d = malloc(hd2*seq*sizeof(float));
    for (int i=0;i<hd2;i++) { float f=1.0f/powf(10000.0f,2.0f*i/(float)(head_dim));
        for (int p=0;p<seq;p++) { cos_d[i*seq+p]=cosf(p*f); sin_d[i*seq+p]=sinf(p*f); }}
    float *mask = malloc(seq*seq*sizeof(float));
    for (int i=0;i<seq;i++) for (int j=0;j<seq;j++) mask[i*seq+j]=(j<=i)?0.0f:-65504.0f;

    ANEWeight w[8];
    w[0]=ane_weight_fp16("@model_path/weights/rms1.bin",wdata,1,dim);
    w[1]=ane_weight_fp16("@model_path/weights/wq.bin",wdata,dim,dim);
    w[2]=ane_weight_fp16("@model_path/weights/wk.bin",kdata,kv_dim,dim);
    w[3]=ane_weight_fp16("@model_path/weights/wv.bin",kdata,kv_dim,dim);
    w[4]=ane_weight_fp16("@model_path/weights/wo.bin",wdata,dim,dim);
    w[5]=ane_weight_fp16("@model_path/weights/cos.bin",cos_d,hd2,seq);
    w[6]=ane_weight_fp16("@model_path/weights/sin.bin",sin_d,hd2,seq);
    w[7]=ane_weight_fp16("@model_path/weights/mask.bin",mask,seq,seq);

    size_t in_b = (size_t)dim*seq*2, out_b = (size_t)out_ch*seq*2;
    ANEKernel *k = ane_compile(mc, ml, w, 8, 1, &in_b, 1, &out_b, ANE_QOS_BACKGROUND);

    for (int i=0;i<8;i++) ane_weight_free(&w[i]);
    free(wdata); free(kdata); free(cos_d); free(sin_d); free(mask);

    if (!k) return -1.0;

    _Float16 *ibuf = calloc(dim*seq, sizeof(_Float16));
    for (int i=0;i<dim*seq&&i<1000;i++) ibuf[i]=(_Float16)(0.01f*(i%100-50));

    // Warmup
    for (int r=0;r<warmup;r++) { ane_write(k,0,ibuf,in_b); ane_eval(k,ANE_QOS_BACKGROUND); }

    // Timed
    double times[20];
    int nruns = runs < 20 ? runs : 20;
    for (int r=0;r<nruns;r++) {
        ane_write(k,0,ibuf,in_b);
        uint64_t t0=tnow();
        ane_eval(k,ANE_QOS_BACKGROUND);
        times[r]=tms(t0,tnow());
    }

    // Median
    for (int i=0;i<nruns-1;i++) for (int j=i+1;j<nruns;j++)
        if (times[j]<times[i]) { double t=times[i]; times[i]=times[j]; times[j]=t; }

    free(ibuf); ane_free(k);
    return times[nruns/2];
}

// Same for FFN-only kernel
static double test_ane_ffn(int dim, int ffn_dim, int seq, int warmup, int runs) {
    NSString *mil = gen_ffn_only_mil(dim, ffn_dim, seq, 1e-5f);
    const char *mc = [mil UTF8String];
    size_t ml = strlen(mc);

    float *wd = malloc(ffn_dim*dim*sizeof(float));
    float *w2d = malloc(dim*ffn_dim*sizeof(float));
    float *nd = malloc(dim*sizeof(float));
    for (int i=0;i<ffn_dim*dim;i++) wd[i]=0.001f*(rand()%2000-1000);
    for (int i=0;i<dim*ffn_dim;i++) w2d[i]=0.001f*(rand()%2000-1000);
    for (int i=0;i<dim;i++) nd[i]=1.0f;

    ANEWeight w[4];
    w[0]=ane_weight_fp16("@model_path/weights/ffn_norm.bin",nd,1,dim);
    w[1]=ane_weight_fp16("@model_path/weights/w1.bin",wd,ffn_dim,dim);
    w[2]=ane_weight_fp16("@model_path/weights/w3.bin",wd,ffn_dim,dim);
    w[3]=ane_weight_fp16("@model_path/weights/w2.bin",w2d,dim,ffn_dim);

    size_t in_b = (size_t)dim*seq*2, out_b = in_b;
    ANEKernel *k = ane_compile(mc, ml, w, 4, 1, &in_b, 1, &out_b, ANE_QOS_BACKGROUND);

    for (int i=0;i<4;i++) ane_weight_free(&w[i]);
    free(wd); free(w2d); free(nd);

    if (!k) return -1.0;

    _Float16 *ibuf = calloc(dim*seq, sizeof(_Float16));
    for (int i=0;i<dim*seq&&i<1000;i++) ibuf[i]=(_Float16)(0.01f*(i%100-50));

    for (int r=0;r<warmup;r++) { ane_write(k,0,ibuf,in_b); ane_eval(k,ANE_QOS_BACKGROUND); }

    double times[20];
    int nruns = runs < 20 ? runs : 20;
    for (int r=0;r<nruns;r++) {
        ane_write(k,0,ibuf,in_b);
        uint64_t t0=tnow();
        ane_eval(k,ANE_QOS_BACKGROUND);
        times[r]=tms(t0,tnow());
    }
    for (int i=0;i<nruns-1;i++) for (int j=i+1;j<nruns;j++)
        if (times[j]<times[i]) { double t=times[i]; times[i]=times[j]; times[j]=t; }

    free(ibuf); ane_free(k);
    return times[nruns/2];
}

int main(void) {
    @autoreleasepool {
        tinit(); srand(42);
        ane_init();
        printf("=== Fiber Architecture Limit Tests ===\n\n");

        // =============================================
        // TEST 1: Sequence Length Scaling
        // =============================================
        printf("--- TEST 1: Sequence Length (dim=768, heads=12, MHA) ---\n");
        printf("%-6s | %8s %8s %10s | %8s %8s %10s\n",
               "seq", "attn_ms", "attn_tps", "spill", "ffn_ms", "ffn_tps", "layer_tps");
        printf("-----------------------------------------------------------\n");

        int seqs[] = {128, 192, 256, 384, 512, 768, 1024};
        for (int si = 0; si < 7; si++) {
            int s = seqs[si];
            if (s < 128) continue;
            double attn = test_ane_attention(768, 12, 12, 64, s, 2, 5);
            double ffn = test_ane_ffn(768, 2048, s, 2, 5);
            double attn_tps = (attn > 0) ? (double)s / (attn/1000.0) : 0;
            double ffn_tps = (ffn > 0) ? (double)s / (ffn/1000.0) : 0;
            double layer_ms = (attn > 0 && ffn > 0) ? attn + ffn : -1;
            double layer_tps = (layer_ms > 0) ? (double)s / (layer_ms/1000.0) : 0;
            printf("%-6d | %7.2f %8.0f %10s | %7.2f %8.0f %10.0f\n",
                   s,
                   attn > 0 ? attn : -1, attn_tps, attn > 0 ? "ok" : "FAIL",
                   ffn > 0 ? ffn : -1, ffn_tps, layer_tps);
        }

        // =============================================
        // TEST 2: Dimension Scaling
        // =============================================
        printf("\n--- TEST 2: Dimension Scaling (seq=256, ratio=2.67x, MHA) ---\n");
        printf("%-6s %-6s %-6s | %8s %8s | %8s %8s | %8s\n",
               "dim", "heads", "ffn", "attn_ms", "attn_tps", "ffn_ms", "ffn_tps", "RSS_MB");
        printf("--------------------------------------------------------------\n");

        typedef struct { int dim; int heads; int ffn; } dconfig;
        dconfig dims[] = {
            {256, 4, 684}, {384, 6, 1024}, {512, 8, 1365},
            {768, 12, 2048}, {1024, 16, 2730}, {1536, 24, 4096}, {2048, 32, 5461}
        };
        for (int di = 0; di < 7; di++) {
            dconfig d = dims[di];
            size_t rss_before = get_rss();
            double attn = test_ane_attention(d.dim, d.heads, d.heads, 64, 256, 1, 3);
            double ffn = test_ane_ffn(d.dim, d.ffn, 256, 1, 3);
            size_t rss_after = get_rss();
            double attn_tps = (attn > 0) ? 256.0 / (attn/1000.0) : 0;
            double ffn_tps = (ffn > 0) ? 256.0 / (ffn/1000.0) : 0;
            printf("%-6d %-6d %-6d | %7.2f %8.0f | %7.2f %8.0f | %7.1f\n",
                   d.dim, d.heads, d.ffn,
                   attn > 0 ? attn : -1, attn_tps,
                   ffn > 0 ? ffn : -1, ffn_tps,
                   rss_after / (1024.0*1024.0));
        }

        // =============================================
        // TEST 3: GQA Ratio Impact
        // =============================================
        printf("\n--- TEST 3: GQA Ratio (dim=768, heads=12, seq=256) ---\n");
        printf("%-6s %-8s | %8s %8s\n", "kv_h", "ratio", "attn_ms", "attn_tps");
        printf("---------------------------------------\n");

        int kv_heads[] = {12, 6, 4, 3, 2, 1};
        for (int ki = 0; ki < 6; ki++) {
            int kvh = kv_heads[ki];
            double attn = test_ane_attention(768, 12, kvh, 64, 256, 2, 5);
            double tps = (attn > 0) ? 256.0 / (attn/1000.0) : 0;
            printf("%-6d %-8s | %7.2f %8.0f\n",
                   kvh, kvh==12?"MHA":"GQA",
                   attn > 0 ? attn : -1, tps);
        }

        // =============================================
        // TEST 4: ANE Compile Budget
        // =============================================
        printf("\n--- TEST 4: Compile Budget ---\n");
        printf("Kernels compiled so far: %d / 119\n", ane_compile_count());
        printf("Remaining budget: %d kernels\n", 119 - ane_compile_count());

        printf("\n=== LIMIT TESTS COMPLETE ===\n");
        printf("Peak RSS: %.1f MB\n", get_rss() / (1024.0*1024.0));
    }
    return 0;
}
