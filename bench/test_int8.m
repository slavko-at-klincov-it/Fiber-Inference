// test_int8.m — Test INT8 weights on ANE vs FP16
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

int main(void) {
    @autoreleasepool {
        tinit(); srand(42);
        ane_init();
        printf("=== INT8 vs FP16 Weight Test ===\n\n");

        int dim = 768, heads = 12, kv = 12, hd = 64, ffn = 2048, seq = 128;
        int kv_dim = kv * hd;

        // Generate random FP32 weights
        float *wf = malloc(dim*dim*4);
        float *kf = malloc(kv_dim*dim*4);
        float *ff = malloc(ffn*dim*4);
        float *f2 = malloc(dim*ffn*4);
        for(int i=0;i<dim*dim;i++) wf[i]=0.01f*((rand()%200)-100);
        for(int i=0;i<kv_dim*dim;i++) kf[i]=0.01f*((rand()%200)-100);
        for(int i=0;i<ffn*dim;i++) ff[i]=0.01f*((rand()%200)-100);
        for(int i=0;i<dim*ffn;i++) f2[i]=0.01f*((rand()%200)-100);

        // cos/sin/mask
        int hd2=hd/2;
        float *cd=malloc(hd2*seq*4),*sd=malloc(hd2*seq*4);
        for(int i=0;i<hd2;i++){float f=1.0f/powf(10000.0f,2.0f*i/(float)hd);
            for(int p=0;p<seq;p++){cd[i*seq+p]=cosf(p*f);sd[i*seq+p]=sinf(p*f);}}
        float *mk=malloc(seq*seq*4);
        for(int i=0;i<seq;i++) for(int j=0;j<seq;j++) mk[i*seq+j]=(j<=i)?0:-65504.0f;
        float *nf=malloc(dim*4);
        for(int i=0;i<dim;i++) nf[i]=1.0f;

        // ====== FP16 Attention ======
        printf("--- FP16 Attention ---\n");
        NSString *amil = gen_sdpa_prefill_mil(dim,heads,kv,hd,seq,10000.0f,1e-5f);
        ANEWeight aw16[8];
        aw16[0]=ane_weight_fp16("@model_path/weights/rms1.bin",nf,1,dim);
        aw16[1]=ane_weight_fp16("@model_path/weights/wq.bin",wf,dim,dim);
        aw16[2]=ane_weight_fp16("@model_path/weights/wk.bin",kf,kv_dim,dim);
        aw16[3]=ane_weight_fp16("@model_path/weights/wv.bin",kf,kv_dim,dim);
        aw16[4]=ane_weight_fp16("@model_path/weights/wo.bin",wf,dim,dim);
        aw16[5]=ane_weight_fp16("@model_path/weights/cos.bin",cd,hd2,seq);
        aw16[6]=ane_weight_fp16("@model_path/weights/sin.bin",sd,hd2,seq);
        aw16[7]=ane_weight_fp16("@model_path/weights/mask.bin",mk,seq,seq);

        int out_ch = dim + 2*kv_dim;
        size_t in_b = dim*seq*2, out_b = out_ch*seq*2;
        ANEKernel *k16 = ane_compile([amil UTF8String],strlen([amil UTF8String]),
                                      aw16,8,1,&in_b,1,&out_b,ANE_QOS_BACKGROUND);
        for(int w=0;w<8;w++) ane_weight_free(&aw16[w]);

        if (!k16) { printf("FP16 compile FAILED\n"); return 1; }
        printf("FP16 compile OK\n");

        _Float16 *ibuf = calloc(dim*seq, sizeof(_Float16));
        for(int i=0;i<1000;i++) ibuf[i]=(_Float16)(0.01f*(i%100-50));
        ane_write(k16,0,ibuf,in_b); ane_eval(k16,ANE_QOS_BACKGROUND);

        double fp16_times[5];
        for(int r=0;r<5;r++){
            ane_write(k16,0,ibuf,in_b);
            uint64_t t0=tnow();
            ane_eval(k16,ANE_QOS_BACKGROUND);
            fp16_times[r]=tms(t0,tnow());
        }
        for(int i=0;i<4;i++) for(int j=i+1;j<5;j++)
            if(fp16_times[j]<fp16_times[i]){double t=fp16_times[i];fp16_times[i]=fp16_times[j];fp16_times[j]=t;}
        printf("FP16 eval: %.2f ms (median)\n", fp16_times[2]);

        // ====== INT8 Attention ======
        printf("\n--- INT8 Attention ---\n");
        float scale_rms, scale_wq, scale_wk, scale_wv, scale_wo;
        ANEWeight aw8[8];
        aw8[0]=ane_weight_fp16("@model_path/weights/rms1.bin",nf,1,dim); // norm stays FP16
        aw8[1]=ane_weight_int8("@model_path/weights/wq.bin",wf,dim,dim,&scale_wq);
        aw8[2]=ane_weight_int8("@model_path/weights/wk.bin",kf,kv_dim,dim,&scale_wk);
        aw8[3]=ane_weight_int8("@model_path/weights/wv.bin",kf,kv_dim,dim,&scale_wv);
        aw8[4]=ane_weight_int8("@model_path/weights/wo.bin",wf,dim,dim,&scale_wo);
        aw8[5]=ane_weight_fp16("@model_path/weights/cos.bin",cd,hd2,seq);
        aw8[6]=ane_weight_fp16("@model_path/weights/sin.bin",sd,hd2,seq);
        aw8[7]=ane_weight_fp16("@model_path/weights/mask.bin",mk,seq,seq);

        ANEKernel *k8 = ane_compile([amil UTF8String],strlen([amil UTF8String]),
                                     aw8,8,1,&in_b,1,&out_b,ANE_QOS_BACKGROUND);
        for(int w=0;w<8;w++) ane_weight_free(&aw8[w]);

        if (!k8) {
            printf("INT8 compile FAILED — ANE may not support mixed FP16/INT8 weights\n");
        } else {
            printf("INT8 compile OK (scales: wq=%.4f wk=%.4f)\n", scale_wq, scale_wk);

            ane_write(k8,0,ibuf,in_b); ane_eval(k8,ANE_QOS_BACKGROUND);

            double int8_times[5];
            for(int r=0;r<5;r++){
                ane_write(k8,0,ibuf,in_b);
                uint64_t t0=tnow();
                ane_eval(k8,ANE_QOS_BACKGROUND);
                int8_times[r]=tms(t0,tnow());
            }
            for(int i=0;i<4;i++) for(int j=i+1;j<5;j++)
                if(int8_times[j]<int8_times[i]){double t=int8_times[i];int8_times[i]=int8_times[j];int8_times[j]=t;}
            printf("INT8 eval: %.2f ms (median)\n", int8_times[2]);
            printf("Speedup: %.2fx\n", fp16_times[2] / int8_times[2]);
            ane_free(k8);
        }

        // ====== FP16 FFN ======
        printf("\n--- FP16 FFN ---\n");
        NSString *fmil = gen_ffn_only_mil(dim, ffn, seq, 1e-5f);
        ANEWeight fw16[4];
        fw16[0]=ane_weight_fp16("@model_path/weights/ffn_norm.bin",nf,1,dim);
        fw16[1]=ane_weight_fp16("@model_path/weights/w1.bin",ff,ffn,dim);
        fw16[2]=ane_weight_fp16("@model_path/weights/w3.bin",ff,ffn,dim);
        fw16[3]=ane_weight_fp16("@model_path/weights/w2.bin",f2,dim,ffn);
        size_t fin=dim*seq*2;
        ANEKernel *fk16 = ane_compile([fmil UTF8String],strlen([fmil UTF8String]),
                                       fw16,4,1,&fin,1,&fin,ANE_QOS_BACKGROUND);
        for(int w=0;w<4;w++) ane_weight_free(&fw16[w]);
        if (!fk16) { printf("FP16 FFN compile FAILED\n"); } else {
            ane_write(fk16,0,ibuf,fin); ane_eval(fk16,ANE_QOS_BACKGROUND);
            double ftimes[5];
            for(int r=0;r<5;r++){
                ane_write(fk16,0,ibuf,fin);
                uint64_t t0=tnow();
                ane_eval(fk16,ANE_QOS_BACKGROUND);
                ftimes[r]=tms(t0,tnow());
            }
            for(int i=0;i<4;i++) for(int j=i+1;j<5;j++)
                if(ftimes[j]<ftimes[i]){double t=ftimes[i];ftimes[i]=ftimes[j];ftimes[j]=t;}
            printf("FP16 FFN: %.2f ms\n", ftimes[2]);

            // INT8 FFN
            printf("\n--- INT8 FFN ---\n");
            float s1,s3,s2;
            ANEWeight fw8[4];
            fw8[0]=ane_weight_fp16("@model_path/weights/ffn_norm.bin",nf,1,dim);
            fw8[1]=ane_weight_int8("@model_path/weights/w1.bin",ff,ffn,dim,&s1);
            fw8[2]=ane_weight_int8("@model_path/weights/w3.bin",ff,ffn,dim,&s3);
            fw8[3]=ane_weight_int8("@model_path/weights/w2.bin",f2,dim,ffn,&s2);
            ANEKernel *fk8 = ane_compile([fmil UTF8String],strlen([fmil UTF8String]),
                                          fw8,4,1,&fin,1,&fin,ANE_QOS_BACKGROUND);
            for(int w=0;w<4;w++) ane_weight_free(&fw8[w]);
            if (!fk8) { printf("INT8 FFN compile FAILED\n"); } else {
                ane_write(fk8,0,ibuf,fin); ane_eval(fk8,ANE_QOS_BACKGROUND);
                double i8times[5];
                for(int r=0;r<5;r++){
                    ane_write(fk8,0,ibuf,fin);
                    uint64_t t0=tnow();
                    ane_eval(fk8,ANE_QOS_BACKGROUND);
                    i8times[r]=tms(t0,tnow());
                }
                for(int i=0;i<4;i++) for(int j=i+1;j<5;j++)
                    if(i8times[j]<i8times[i]){double t=i8times[i];i8times[i]=i8times[j];i8times[j]=t;}
                printf("INT8 FFN: %.2f ms\n", i8times[2]);
                printf("Speedup: %.2fx\n", ftimes[2] / i8times[2]);
                ane_free(fk8);
            }
            ane_free(fk16);
        }

        ane_free(k16);
        free(ibuf);free(wf);free(kf);free(ff);free(f2);free(cd);free(sd);free(mk);free(nf);

        printf("\nCompile budget: %d / 119\n", ane_compile_count());
        printf("=== DONE ===\n");
    }
    return 0;
}
