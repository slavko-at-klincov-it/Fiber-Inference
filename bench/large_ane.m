// large_ane.m — ANE-only benchmarks for 7B-9B and Qwen3 dimensions
// No CPU baseline (too slow at dim=4096+). Measures only ANE throughput.
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

typedef struct { const char *name; int dim; int heads; int kv; int ffn; int layers; int seq; } cfg_t;

static double bench_ane_layer(cfg_t c) {
    int hd = c.dim / c.heads;
    int kv_dim = c.kv * hd;
    int out_ch = c.dim + 2 * kv_dim;
    int seq = c.seq < 128 ? 128 : c.seq;

    // Attention kernel
    NSString *amil = gen_sdpa_prefill_mil(c.dim, c.heads, c.kv, hd, seq, 1000000.0f, 1e-5f);
    int hd2 = hd/2;
    float *wf = malloc(c.dim*c.dim*4);
    float *kf = malloc(kv_dim*c.dim*4);
    for(int i=0;i<c.dim*c.dim;i++) wf[i]=0.001f*(rand()%2000-1000);
    for(int i=0;i<kv_dim*c.dim;i++) kf[i]=0.001f*(rand()%2000-1000);
    float *cd=malloc(hd2*seq*4),*sd=malloc(hd2*seq*4);
    for(int i=0;i<hd2;i++){float f=1.0f/powf(1000000.0f,2.0f*i/(float)hd);
        for(int p=0;p<seq;p++){cd[i*seq+p]=cosf(p*f);sd[i*seq+p]=sinf(p*f);}}
    float *mk=malloc(seq*seq*4);
    for(int i=0;i<seq;i++) for(int j=0;j<seq;j++) mk[i*seq+j]=(j<=i)?0:-65504.0f;

    ANEWeight aw[8];
    aw[0]=ane_weight_fp16("@model_path/weights/rms1.bin",wf,1,c.dim);
    aw[1]=ane_weight_fp16("@model_path/weights/wq.bin",wf,c.dim,c.dim);
    aw[2]=ane_weight_fp16("@model_path/weights/wk.bin",kf,kv_dim,c.dim);
    aw[3]=ane_weight_fp16("@model_path/weights/wv.bin",kf,kv_dim,c.dim);
    aw[4]=ane_weight_fp16("@model_path/weights/wo.bin",wf,c.dim,c.dim);
    aw[5]=ane_weight_fp16("@model_path/weights/cos.bin",cd,hd2,seq);
    aw[6]=ane_weight_fp16("@model_path/weights/sin.bin",sd,hd2,seq);
    aw[7]=ane_weight_fp16("@model_path/weights/mask.bin",mk,seq,seq);

    size_t in_b=(size_t)c.dim*seq*2, out_b=(size_t)out_ch*seq*2;
    ANEKernel *ak = ane_compile([amil UTF8String],strlen([amil UTF8String]),aw,8,1,&in_b,1,&out_b,ANE_QOS_BACKGROUND);
    for(int w=0;w<8;w++) ane_weight_free(&aw[w]);

    // FFN kernel
    NSString *fmil = gen_ffn_only_mil(c.dim, c.ffn, seq, 1e-5f);
    float *ff=malloc(c.ffn*c.dim*4);
    float *f2=malloc(c.dim*c.ffn*4);
    float *fn=malloc(c.dim*4);
    for(int i=0;i<c.ffn*c.dim;i++) ff[i]=0.001f*(rand()%2000-1000);
    for(int i=0;i<c.dim*c.ffn;i++) f2[i]=0.001f*(rand()%2000-1000);
    for(int i=0;i<c.dim;i++) fn[i]=1.0f;
    ANEWeight fw[4];
    fw[0]=ane_weight_fp16("@model_path/weights/ffn_norm.bin",fn,1,c.dim);
    fw[1]=ane_weight_fp16("@model_path/weights/w1.bin",ff,c.ffn,c.dim);
    fw[2]=ane_weight_fp16("@model_path/weights/w3.bin",ff,c.ffn,c.dim);
    fw[3]=ane_weight_fp16("@model_path/weights/w2.bin",f2,c.dim,c.ffn);
    size_t fout=in_b;
    ANEKernel *fk = ane_compile([fmil UTF8String],strlen([fmil UTF8String]),fw,4,1,&in_b,1,&fout,ANE_QOS_BACKGROUND);
    for(int w=0;w<4;w++) ane_weight_free(&fw[w]);
    free(wf);free(kf);free(cd);free(sd);free(mk);free(ff);free(f2);free(fn);

    if (!ak || !fk) { if(ak)ane_free(ak); if(fk)ane_free(fk); return -1; }

    // Benchmark 1 layer (attn + ffn)
    _Float16 *x = calloc(c.dim*seq, sizeof(_Float16));
    for(int i=0;i<1000&&i<c.dim*seq;i++) x[i]=(_Float16)(0.01f*(i%100-50));

    // Warmup
    ane_write(ak,0,x,in_b); ane_eval(ak,ANE_QOS_BACKGROUND);

    // Measure
    int runs = 3;
    double best = 1e9;
    for (int r = 0; r < runs; r++) {
        ane_lock_input(ak,0); memcpy(ane_input_ptr(ak,0),x,in_b); ane_unlock_input(ak,0);
        uint64_t t0 = tnow();
        ane_eval(ak,ANE_QOS_BACKGROUND);
        ane_lock_output(ak,0); memcpy(x,ane_output_ptr(ak,0),in_b); ane_unlock_output(ak,0);
        ane_lock_input(fk,0); memcpy(ane_input_ptr(fk,0),x,in_b); ane_unlock_input(fk,0);
        ane_eval(fk,ANE_QOS_BACKGROUND);
        ane_lock_output(fk,0); memcpy(x,ane_output_ptr(fk,0),fout); ane_unlock_output(fk,0);
        double ms = tms(t0, tnow());
        if (ms < best) best = ms;
    }

    free(x); ane_free(ak); ane_free(fk);
    return best;
}

int main(void) {
    @autoreleasepool {
        tinit(); srand(42);
        ane_init();
        ANEDeviceInfo info = ane_device_info();
        printf("=== Large Model ANE Benchmarks ===\n");
        printf("Apple M4 ANE: %s, %d cores\n\n", info.arch, info.num_cores);

        cfg_t cfgs[] = {
            // Reference (known good)
            {"768-hd64",  768, 12, 12, 2048, 12, 128},  // head_dim=64
            {"768-hd128", 768,  6,  6, 2048, 12, 128},  // head_dim=128 (test)
            {"~1B",    1024, 16,  8, 4096, 24, 128},
            {"~2B",    2048, 32,  8, 5461, 22, 128},
            // 7B-9B class
            {"~7B-hd64",  4096, 64,  8, 11008, 32, 128},
            {"~9B-hd64",  4096, 64,  8, 14336, 32, 128},
            // Qwen3 exact dimensions (head_dim=128)
            {"Qwen3-0.6B", 1024,  8,  8,  3072, 28, 128},
            {"Qwen3-1.7B", 2048, 16,  8,  6144, 28, 128},
            {"Q3-4B-fix",  2560, 32,  8,  9728, 36, 128},  // 32 heads (hd=80), kv=8 (ratio=4, integer)
            {"Qwen3-8B",   4096, 32,  8, 12288, 36, 128},
        };
        int n = sizeof(cfgs)/sizeof(cfgs[0]);

        printf("%-12s %-5s %-5s %-3s %-4s %-6s %-4s | %8s %8s %8s %7s\n",
               "model","dim","heads","kv","hd","ffn","lyrs",
               "1L_ms","total_ms","tok/s","RSS_MB");
        printf("--------------------------------------------------------------------------\n");

        for (int i = 0; i < n; i++) {
            cfg_t c = cfgs[i];
            int hd = c.dim / c.heads;
            size_t rss0 = get_rss();

            double layer_ms = bench_ane_layer(c);
            double total_ms = layer_ms > 0 ? layer_ms * c.layers : -1;
            double tps = total_ms > 0 ? (double)c.seq / (total_ms / 1000.0) : 0;

            printf("%-12s %-5d %-5d %-3d %-4d %-6d %-4d | %7.2f %8.1f %8.0f %6.1f\n",
                   c.name, c.dim, c.heads, c.kv, hd, c.ffn, c.layers,
                   layer_ms > 0 ? layer_ms : -1,
                   total_ms > 0 ? total_ms : -1,
                   tps,
                   get_rss() / (1024.0*1024.0));
            fflush(stdout);
        }

        printf("\nCompile budget: %d / 119\n", ane_compile_count());
        printf("=== COMPLETE ===\n");
    }
    return 0;
}
