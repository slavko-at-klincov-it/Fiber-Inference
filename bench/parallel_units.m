// parallel_units.m — Test ALL units working simultaneously on the SAME model
// The question nobody has tested: does GPU + AMX + ANE together beat GPU alone?
// M4_RE Exp 06 proved 103% efficiency when running in parallel.
// MLX uses only GPU (~7 TFLOPS). 11+ TFLOPS sit unused.

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#include <Accelerate/Accelerate.h>
#include <mach/mach_time.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ane.h"
#include "ane_mil.h"

static mach_timebase_info_data_t tbi;
static void tinit(void) { mach_timebase_info(&tbi); }
static uint64_t tnow(void) { return mach_absolute_time(); }
static double tms(uint64_t a, uint64_t b) { return (double)(b-a)*tbi.numer/tbi.denom/1e6; }

// ============================================================
// Test 1: GPU alone (MPS MatMul) — baseline
// ============================================================
static double test_gpu_alone(id<MTLDevice> dev, id<MTLCommandQueue> queue,
                              int dim, int ffn, int seq, int layers) {
    id<MTLBuffer> bx = [dev newBufferWithLength:dim*seq*2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bw = [dev newBufferWithLength:ffn*dim*2 options:MTLResourceStorageModeShared];
    id<MTLBuffer> bh = [dev newBufferWithLength:ffn*seq*2 options:MTLResourceStorageModeShared];
    _Float16 *p = bx.contents; for(int i=0;i<dim*seq;i++) p[i]=(_Float16)(0.01f*(rand()%200-100));
    p = bw.contents; for(int i=0;i<ffn*dim;i++) p[i]=(_Float16)(0.001f*(rand()%2000-1000));

    MPSMatrixDescriptor *dx=[MPSMatrixDescriptor matrixDescriptorWithRows:dim columns:seq rowBytes:seq*2 dataType:MPSDataTypeFloat16];
    MPSMatrixDescriptor *dw=[MPSMatrixDescriptor matrixDescriptorWithRows:ffn columns:dim rowBytes:dim*2 dataType:MPSDataTypeFloat16];
    MPSMatrixDescriptor *dh=[MPSMatrixDescriptor matrixDescriptorWithRows:ffn columns:seq rowBytes:seq*2 dataType:MPSDataTypeFloat16];
    MPSMatrix *mx=[MPSMatrix.alloc initWithBuffer:bx descriptor:dx];
    MPSMatrix *mw=[MPSMatrix.alloc initWithBuffer:bw descriptor:dw];
    MPSMatrix *mh=[MPSMatrix.alloc initWithBuffer:bh descriptor:dh];
    MPSMatrixMultiplication *mul=[MPSMatrixMultiplication.alloc initWithDevice:dev resultRows:ffn resultColumns:seq interiorColumns:dim];

    // Warmup
    {id<MTLCommandBuffer> cb=[queue commandBuffer]; [mul encodeToCommandBuffer:cb leftMatrix:mw rightMatrix:mx resultMatrix:mh]; [cb commit];[cb waitUntilCompleted];}

    // Measure: layers × W1 matmul (simulates FFN compute load)
    uint64_t t0 = tnow();
    for (int l = 0; l < layers; l++) {
        id<MTLCommandBuffer> cb=[queue commandBuffer];
        [mul encodeToCommandBuffer:cb leftMatrix:mw rightMatrix:mx resultMatrix:mh];
        [cb commit];[cb waitUntilCompleted];
    }
    return tms(t0, tnow());
}

// ============================================================
// Test 2: AMX alone (cblas_sgemm) — baseline
// ============================================================
static double test_amx_alone(int dim, int ffn, int seq, int layers) {
    float *x = malloc(dim*seq*4);
    float *w = malloc(ffn*dim*4);
    float *h = malloc(ffn*seq*4);
    for(int i=0;i<dim*seq;i++) x[i]=0.01f*(rand()%200-100);
    for(int i=0;i<ffn*dim;i++) w[i]=0.001f*(rand()%2000-1000);

    // Warmup
    cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,ffn,seq,dim,1,w,dim,x,seq,0,h,seq);

    uint64_t t0 = tnow();
    for (int l = 0; l < layers; l++) {
        cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,ffn,seq,dim,1,w,dim,x,seq,0,h,seq);
    }
    double ms = tms(t0, tnow());
    free(x); free(w); free(h);
    return ms;
}

// ============================================================
// Test 3: ANE alone (attention kernel) — baseline
// ============================================================
static double test_ane_alone(int dim, int heads, int kv, int hd, int ffn, int seq, int layers) {
    int kv_dim = kv * hd, ane_seq = seq < 128 ? 128 : seq;
    NSString *amil = gen_sdpa_prefill_mil(dim, heads, kv, hd, ane_seq, 10000.0f, 1e-5f);
    int hd2=hd/2;
    float *wf=malloc(dim*dim*4),*kf=malloc(kv_dim*dim*4),*nf=malloc(dim*4);
    for(int i=0;i<dim*dim;i++) wf[i]=0.001f*(rand()%2000-1000);
    for(int i=0;i<kv_dim*dim;i++) kf[i]=0.001f*(rand()%2000-1000);
    for(int i=0;i<dim;i++) nf[i]=1.0f;
    float *cd=malloc(hd2*ane_seq*4),*sd=malloc(hd2*ane_seq*4);
    for(int i=0;i<hd2;i++){float f=1.0f/powf(10000.0f,2.0f*i/(float)hd);for(int p=0;p<ane_seq;p++){cd[i*ane_seq+p]=cosf(p*f);sd[i*ane_seq+p]=sinf(p*f);}}
    float *mk=malloc(ane_seq*ane_seq*4);
    for(int i=0;i<ane_seq;i++)for(int j=0;j<ane_seq;j++)mk[i*ane_seq+j]=(j<=i)?0:-65504.0f;

    ANEWeight aw[8];
    aw[0]=ane_weight_fp16("@model_path/weights/rms1.bin",nf,1,dim);
    aw[1]=ane_weight_fp16("@model_path/weights/wq.bin",wf,dim,dim);
    aw[2]=ane_weight_fp16("@model_path/weights/wk.bin",kf,kv_dim,dim);
    aw[3]=ane_weight_fp16("@model_path/weights/wv.bin",kf,kv_dim,dim);
    aw[4]=ane_weight_fp16("@model_path/weights/wo.bin",wf,dim,dim);
    aw[5]=ane_weight_fp16("@model_path/weights/cos.bin",cd,hd2,ane_seq);
    aw[6]=ane_weight_fp16("@model_path/weights/sin.bin",sd,hd2,ane_seq);
    aw[7]=ane_weight_fp16("@model_path/weights/mask.bin",mk,ane_seq,ane_seq);
    int oc=dim+2*kv_dim;
    size_t ib=dim*ane_seq*2, ob=oc*ane_seq*2;
    ANEKernel *ak=ane_compile([amil UTF8String],strlen([amil UTF8String]),aw,8,1,&ib,1,&ob,ANE_QOS_BACKGROUND);
    for(int w=0;w<8;w++) ane_weight_free(&aw[w]);
    free(wf);free(kf);free(nf);free(cd);free(sd);free(mk);
    if(!ak) return -1;

    _Float16 *x=calloc(dim*ane_seq,sizeof(_Float16));
    for(int i=0;i<1000;i++) x[i]=(_Float16)(0.01f*(i%100-50));
    ane_write(ak,0,x,ib); ane_eval(ak,ANE_QOS_BACKGROUND);

    uint64_t t0=tnow();
    for(int l=0;l<layers;l++){
        ane_write(ak,0,x,ib);
        ane_eval(ak,ANE_QOS_BACKGROUND);
    }
    double ms=tms(t0,tnow());
    free(x); ane_free(ak);
    return ms;
}

// ============================================================
// Test 4: GPU + AMX PARALLEL (GPU does attention, AMX does FFN)
// ============================================================
typedef struct {
    int dim, ffn, seq, layers;
    double ms;
} amx_thread_args;

static void *amx_thread_func(void *arg) {
    amx_thread_args *a = (amx_thread_args *)arg;
    a->ms = test_amx_alone(a->dim, a->ffn, a->seq, a->layers);
    return NULL;
}

static double test_gpu_amx_parallel(id<MTLDevice> dev, id<MTLCommandQueue> queue,
                                     int dim, int ffn, int seq, int layers) {
    // Start AMX on background thread
    amx_thread_args args = {dim, ffn, seq, layers, 0};
    pthread_t amx_tid;

    uint64_t t0 = tnow();
    pthread_create(&amx_tid, NULL, amx_thread_func, &args);

    // GPU on main thread (simultaneously)
    double gpu_ms = test_gpu_alone(dev, queue, dim, ffn, seq, layers);

    pthread_join(amx_tid, NULL);
    double total = tms(t0, tnow());

    printf("    (GPU=%.1fms, AMX=%.1fms, wall=%.1fms, overlap=%.0f%%)\n",
           gpu_ms, args.ms, total,
           (gpu_ms + args.ms - total) / (gpu_ms + args.ms) * 100);
    return total;
}

// ============================================================
// Test 5: GPU + ANE PARALLEL
// ============================================================
typedef struct {
    int dim, heads, kv, hd, ffn, seq, layers;
    double ms;
} ane_thread_args;

static void *ane_thread_func(void *arg) {
    ane_thread_args *a = (ane_thread_args *)arg;
    a->ms = test_ane_alone(a->dim, a->heads, a->kv, a->hd, a->ffn, a->seq, a->layers);
    return NULL;
}

static double test_gpu_ane_parallel(id<MTLDevice> dev, id<MTLCommandQueue> queue,
                                     int dim, int heads, int kv, int hd, int ffn,
                                     int seq, int layers) {
    ane_thread_args args = {dim, heads, kv, hd, ffn, seq, layers, 0};
    pthread_t ane_tid;

    uint64_t t0 = tnow();
    pthread_create(&ane_tid, NULL, ane_thread_func, &args);
    double gpu_ms = test_gpu_alone(dev, queue, dim, ffn, seq, layers);
    pthread_join(ane_tid, NULL);
    double total = tms(t0, tnow());

    printf("    (GPU=%.1fms, ANE=%.1fms, wall=%.1fms, overlap=%.0f%%)\n",
           gpu_ms, args.ms > 0 ? args.ms : -1, total,
           args.ms > 0 ? (gpu_ms + args.ms - total) / (gpu_ms + args.ms) * 100 : 0);
    return total;
}

// ============================================================
// Test 6: ALL THREE — GPU + AMX + ANE
// ============================================================
static double test_all_three(id<MTLDevice> dev, id<MTLCommandQueue> queue,
                              int dim, int heads, int kv, int hd, int ffn,
                              int seq, int layers) {
    amx_thread_args amx_args = {dim, ffn, seq, layers, 0};
    ane_thread_args ane_args = {dim, heads, kv, hd, ffn, seq, layers, 0};
    pthread_t amx_tid, ane_tid;

    uint64_t t0 = tnow();
    pthread_create(&amx_tid, NULL, amx_thread_func, &amx_args);
    pthread_create(&ane_tid, NULL, ane_thread_func, &ane_args);
    double gpu_ms = test_gpu_alone(dev, queue, dim, ffn, seq, layers);
    pthread_join(amx_tid, NULL);
    pthread_join(ane_tid, NULL);
    double total = tms(t0, tnow());

    double serial = gpu_ms + amx_args.ms + (ane_args.ms > 0 ? ane_args.ms : 0);
    printf("    (GPU=%.1f, AMX=%.1f, ANE=%.1f, wall=%.1f, efficiency=%.0f%%)\n",
           gpu_ms, amx_args.ms, ane_args.ms > 0 ? ane_args.ms : -1, total,
           serial / total * 100);
    return total;
}

int main(void) {
    @autoreleasepool {
        tinit(); srand(42);
        ane_init();
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        id<MTLCommandQueue> queue = [dev newCommandQueue];

        printf("=== PARALLEL UNIT TEST: Does GPU+AMX+ANE together beat GPU alone? ===\n");
        printf("M4 Mac Mini: GPU ~7 TFLOPS, ANE ~19 TFLOPS, AMX ~1.6 TFLOPS\n");
        printf("M4_RE Exp 06: 103%% efficiency when all run simultaneously\n\n");

        typedef struct { const char *name; int dim; int heads; int kv; int hd; int ffn; int layers; int seq; } cfg;
        cfg tests[] = {
            {"768d-12L",  768, 12, 12, 64, 2048, 12, 128},
            {"1024d-12L", 1024, 16, 16, 64, 2730, 12, 128},
            {"1024d-28L", 1024, 8, 8, 128, 3072, 28, 128},
        };
        int nt = sizeof(tests)/sizeof(tests[0]);

        for (int ti = 0; ti < nt; ti++) {
            cfg c = tests[ti];
            printf("--- %s (dim=%d, ffn=%d, %dL, seq=%d) ---\n", c.name, c.dim, c.ffn, c.layers, c.seq);

            double gpu = test_gpu_alone(dev, queue, c.dim, c.ffn, c.seq, c.layers);
            double amx = test_amx_alone(c.dim, c.ffn, c.seq, c.layers);
            double ane = test_ane_alone(c.dim, c.heads, c.kv, c.hd, c.ffn, c.seq, c.layers);

            printf("  Solo:     GPU=%.1fms  AMX=%.1fms  ANE=%.1fms\n", gpu, amx, ane>0?ane:-1);

            // GPU work = layers matmuls. Calculate throughput.
            double flops_per_layer = 2.0 * c.ffn * c.dim * c.seq; // one matmul
            double gpu_gflops = (flops_per_layer * c.layers) / (gpu * 1e6);
            double amx_gflops = (flops_per_layer * c.layers) / (amx * 1e6);
            printf("  GFLOPS:   GPU=%.0f  AMX=%.0f\n", gpu_gflops, amx_gflops);

            printf("  GPU+AMX:  ");
            double par_ga = test_gpu_amx_parallel(dev, queue, c.dim, c.ffn, c.seq, c.layers);

            printf("  GPU+ANE:  ");
            double par_gn = test_gpu_ane_parallel(dev, queue, c.dim, c.heads, c.kv, c.hd, c.ffn, c.seq, c.layers);

            printf("  ALL 3:    ");
            double par_all = test_all_three(dev, queue, c.dim, c.heads, c.kv, c.hd, c.ffn, c.seq, c.layers);

            // Total work done in wall time (throughput comparison)
            printf("\n  THROUGHPUT (work / wall_time):\n");
            printf("    GPU alone:      %.1f ms → %.0f GFLOPS\n", gpu, gpu_gflops);
            printf("    GPU+AMX:        %.1f ms → %.0f GFLOPS (%.1fx)\n", par_ga,
                   (flops_per_layer * c.layers * 2) / (par_ga * 1e6), // 2x work (GPU+AMX each do layers matmuls)
                   (gpu + amx) / par_ga); // speedup = serial_time / parallel_time
            printf("    GPU+ANE:        %.1f ms → %.1fx\n", par_gn, (gpu + (ane>0?ane:0)) / par_gn);
            printf("    GPU+AMX+ANE:    %.1f ms → %.1fx\n", par_all, (gpu + amx + (ane>0?ane:0)) / par_all);
            printf("\n");
        }

        printf("Compile budget: %d / 119\n", ane_compile_count());
        printf("=== DONE ===\n");
    }
    return 0;
}
