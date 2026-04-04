// ane_mil.h — MIL text generator for GQA SDPA on ANE
// Adapted from ANE-Training/training/stories_mil.h for inference with GQA + RoPE
#pragma once
#import <Foundation/Foundation.h>
#include <math.h>

#define MIL_HDR \
    @"program(1.3)\n[buildInfo = dict<string, string>({{\"coremlc-component-MIL\", \"3510.2.1\"}, " \
    "{\"coremlc-version\", \"3505.4.1\"}, {\"coremltools-component-milinternal\", \"\"}, " \
    "{\"coremltools-version\", \"9.0\"}})]\n{\n"
#define CONV_CONST \
    "        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n" \
    "        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n" \
    "        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n" \
    "        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n" \
    "        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"

// Generate fused SDPA kernel for prefill (RMSNorm + QKV + RoPE + SDPA + Wo)
// Input:  [1, dim, 1, seq] FP16
// Output: [1, dim + 2*kv_dim, 1, seq] FP16 = concat(wo_out, k_roped, v)
// Weights baked: Wq, Wk, Wv, Wo, rms_norm, cos_table, sin_table, causal_mask
static NSString *gen_sdpa_prefill_mil(int dim, int n_heads, int n_kv_heads,
                                       int head_dim, int seq,
                                       float rope_base, float rms_eps) {
    int kv_dim = n_kv_heads * head_dim;
    int gqa_ratio = n_heads / n_kv_heads;
    if (n_kv_heads > 1 && n_heads % n_kv_heads != 0) {
        fprintf(stderr, "ANE MIL: n_heads=%d not divisible by n_kv_heads=%d (GQA ratio %.1f)\n",
                n_heads, n_kv_heads, (float)n_heads / n_kv_heads);
        return nil;
    }
    float scale = 1.0f / sqrtf((float)head_dim);
    float invd = 1.0f / (float)dim;
    int hd2 = head_dim / 2;

    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", dim, seq];

    // === RMSNorm ===
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> sq = mul(x=x,y=x)[name=string(\"sq\")];\n", dim, seq];
    [m appendString:@"        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([1])];\n"];
    [m appendString:@"        bool kd = const()[name=string(\"kd\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss = reduce_sum(x=sq,axes=rax,keep_dims=kd)[name=string(\"ss\")];\n", seq];
    [m appendFormat:@"        fp16 invd = const()[name=string(\"invd\"), val=fp16(%f)];\n", invd];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss2 = mul(x=ss,y=invd)[name=string(\"ss2\")];\n", seq];
    [m appendFormat:@"        fp16 eps = const()[name=string(\"eps\"), val=fp16(%e)];\n", rms_eps];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss3 = add(x=ss2,y=eps)[name=string(\"ss3\")];\n", seq];
    [m appendString:@"        fp16 nhalf = const()[name=string(\"nhalf\"), val=fp16(-0.5)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> rrms = pow(x=ss3,y=nhalf)[name=string(\"rrms\")];\n", seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xr = mul(x=x,y=rrms)[name=string(\"xr\")];\n", dim, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> rw = const()[name=string(\"rw\"), val=tensor<fp16, [1,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/rms1.bin\"), offset=uint64(64)))];\n", dim, dim];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xn = mul(x=xr,y=rw)[name=string(\"xn\")];\n", dim, seq];

    // === QKV Projections (1x1 conv) ===
    [m appendString:@CONV_CONST];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wq = const()[name=string(\"Wq\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wq.bin\"), offset=uint64(64)))];\n", dim,dim,dim,dim];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wk = const()[name=string(\"Wk\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wk.bin\"), offset=uint64(64)))];\n", kv_dim,dim,kv_dim,dim];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wv = const()[name=string(\"Wv\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wv.bin\"), offset=uint64(64)))];\n", kv_dim,dim,kv_dim,dim];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> Wo = const()[name=string(\"Wo\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/wo.bin\"), offset=uint64(64)))];\n", dim,dim,dim,dim];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> qf = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wq,x=xn)[name=string(\"cq\")];\n", dim,seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> kf = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wk,x=xn)[name=string(\"ck\")];\n", kv_dim,seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> vf = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wv,x=xn)[name=string(\"cv\")];\n", kv_dim,seq];

    // === Reshape Q: [1, dim, 1, seq] → [1, n_heads, head_dim, seq] ===
    [m appendFormat:@"        tensor<int32, [4]> qsh = const()[name=string(\"qsh\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", n_heads, head_dim, seq];
    [m appendString:@"        tensor<int32, [4]> pm = const()[name=string(\"pm\"), val=tensor<int32, [4]>([0,1,3,2])];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> q4 = reshape(shape=qsh,x=qf)[name=string(\"rq\")];\n", n_heads, head_dim, seq];

    // === Reshape K,V: [1, kv_dim, 1, seq] → [1, n_kv_heads, head_dim, seq] ===
    [m appendFormat:@"        tensor<int32, [4]> kvsh = const()[name=string(\"kvsh\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", n_kv_heads, head_dim, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k4 = reshape(shape=kvsh,x=kf)[name=string(\"rk\")];\n", n_kv_heads, head_dim, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> v4 = reshape(shape=kvsh,x=vf)[name=string(\"rv\")];\n", n_kv_heads, head_dim, seq];

    // === RoPE on Q: split head_dim into halves, apply rotation ===
    // cos_table, sin_table: [1, 1, hd/2, seq] — broadcast over heads
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> cos_t = const()[name=string(\"cos_t\"), val=tensor<fp16, [1,1,%d,%d]>(BLOBFILE(path=string(\"@model_path/weights/cos.bin\"), offset=uint64(64)))];\n", hd2, seq, hd2, seq];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> sin_t = const()[name=string(\"sin_t\"), val=tensor<fp16, [1,1,%d,%d]>(BLOBFILE(path=string(\"@model_path/weights/sin.bin\"), offset=uint64(64)))];\n", hd2, seq, hd2, seq];

    // Q RoPE: q4 shape is [1, n_heads, head_dim, seq]
    // slice first half [0:hd/2] and second half [hd/2:hd] along dim=2
    [m appendFormat:@"        tensor<int32, [4]> sb = const()[name=string(\"sb\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendFormat:@"        tensor<int32, [4]> se1 = const()[name=string(\"se1\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", n_heads, hd2, seq];
    [m appendFormat:@"        tensor<int32, [4]> sb2 = const()[name=string(\"sb2\"), val=tensor<int32, [4]>([0,0,%d,0])];\n", hd2];
    [m appendFormat:@"        tensor<int32, [4]> se2 = const()[name=string(\"se2\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", n_heads, head_dim, seq];
    // Q halves
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> q_lo = slice_by_index(begin=sb,end=se1,x=q4)[name=string(\"qlo\")];\n", n_heads, hd2, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> q_hi = slice_by_index(begin=sb2,end=se2,x=q4)[name=string(\"qhi\")];\n", n_heads, hd2, seq];
    // q_rot_lo = q_lo * cos - q_hi * sin
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> qc1 = mul(x=q_lo,y=cos_t)[name=string(\"qc1\")];\n", n_heads, hd2, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> qs1 = mul(x=q_hi,y=sin_t)[name=string(\"qs1\")];\n", n_heads, hd2, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> qr_lo = sub(x=qc1,y=qs1)[name=string(\"qrlo\")];\n", n_heads, hd2, seq];
    // q_rot_hi = q_hi * cos + q_lo * sin
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> qc2 = mul(x=q_hi,y=cos_t)[name=string(\"qc2\")];\n", n_heads, hd2, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> qs2 = mul(x=q_lo,y=sin_t)[name=string(\"qs2\")];\n", n_heads, hd2, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> qr_hi = add(x=qc2,y=qs2)[name=string(\"qrhi\")];\n", n_heads, hd2, seq];
    // concat back: [1, n_heads, head_dim, seq]
    [m appendString:@"        int32 cax2 = const()[name=string(\"cax2\"), val=int32(2)];\n"];
    [m appendString:@"        bool cid2 = const()[name=string(\"cid2\"), val=bool(false)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> q_rot = concat(axis=cax2,interleave=cid2,values=(qr_lo,qr_hi))[name=string(\"qrot\")];\n", n_heads, head_dim, seq];

    // === RoPE on K (same pattern, n_kv_heads instead of n_heads) ===
    [m appendFormat:@"        tensor<int32, [4]> kse1 = const()[name=string(\"kse1\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", n_kv_heads, hd2, seq];
    [m appendFormat:@"        tensor<int32, [4]> kse2 = const()[name=string(\"kse2\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", n_kv_heads, head_dim, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k_lo = slice_by_index(begin=sb,end=kse1,x=k4)[name=string(\"klo\")];\n", n_kv_heads, hd2, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k_hi = slice_by_index(begin=sb2,end=kse2,x=k4)[name=string(\"khi\")];\n", n_kv_heads, hd2, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> kc1 = mul(x=k_lo,y=cos_t)[name=string(\"kc1\")];\n", n_kv_heads, hd2, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> ks1 = mul(x=k_hi,y=sin_t)[name=string(\"ks1\")];\n", n_kv_heads, hd2, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> kr_lo = sub(x=kc1,y=ks1)[name=string(\"krlo\")];\n", n_kv_heads, hd2, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> kc2 = mul(x=k_hi,y=cos_t)[name=string(\"kc2\")];\n", n_kv_heads, hd2, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> ks2 = mul(x=k_lo,y=sin_t)[name=string(\"ks2\")];\n", n_kv_heads, hd2, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> kr_hi = add(x=kc2,y=ks2)[name=string(\"krhi\")];\n", n_kv_heads, hd2, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k_rot = concat(axis=cax2,interleave=cid2,values=(kr_lo,kr_hi))[name=string(\"krot\")];\n", n_kv_heads, head_dim, seq];

    // === GQA: repeat-interleave K,V from n_kv_heads to n_heads ===
    // tile gives [k0,k1,k2,k3,k0,...] but GQA needs [k0,k0,...,k1,k1,...]
    // Fix: reshape [1,nkv,hd,seq] → [1,nkv,1,hd*seq] → tile dim2 → [1,nkv,gqa,hd*seq] → reshape [1,nh,hd,seq]
    if (gqa_ratio > 1) {
        int hd_x_seq = head_dim * seq;
        // K: flatten last 2 dims, tile, reshape
        [m appendFormat:@"        tensor<int32, [4]> gqa_f = const()[name=string(\"gqa_f\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", n_kv_heads, hd_x_seq];
        [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> k_flat = reshape(shape=gqa_f,x=k_rot)[name=string(\"kflat\")];\n", n_kv_heads, hd_x_seq];
        [m appendFormat:@"        tensor<int32, [4]> gqa_t = const()[name=string(\"gqa_t\"), val=tensor<int32, [4]>([1,1,%d,1])];\n", gqa_ratio];
        [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k_rep = tile(reps=gqa_t,x=k_flat)[name=string(\"krep\")];\n", n_kv_heads, gqa_ratio, hd_x_seq];
        [m appendFormat:@"        tensor<int32, [4]> gqa_r = const()[name=string(\"gqa_r\"), val=tensor<int32, [4]>([1,%d,%d,%d])];\n", n_heads, head_dim, seq];
        [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k_tiled = reshape(shape=gqa_r,x=k_rep)[name=string(\"ktile\")];\n", n_heads, head_dim, seq];
        // V: same pattern
        [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> v_flat = reshape(shape=gqa_f,x=v4)[name=string(\"vflat\")];\n", n_kv_heads, hd_x_seq];
        [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> v_rep = tile(reps=gqa_t,x=v_flat)[name=string(\"vrep\")];\n", n_kv_heads, gqa_ratio, hd_x_seq];
        [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> v_tiled = reshape(shape=gqa_r,x=v_rep)[name=string(\"vtile\")];\n", n_heads, head_dim, seq];
    }

    // === Transpose to [1, heads, seq, head_dim] for matmul ===
    NSString *q_name = @"q_rot";
    NSString *k_name = (gqa_ratio > 1) ? @"k_tiled" : @"k_rot";
    NSString *v_name = (gqa_ratio > 1) ? @"v_tiled" : @"v4";
    int attn_heads = n_heads; // after tiling

    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> q = transpose(perm=pm,x=%@)[name=string(\"tq\")];\n", attn_heads, seq, head_dim, q_name];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> k = transpose(perm=pm,x=%@)[name=string(\"tk\")];\n", attn_heads, seq, head_dim, k_name];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> v = transpose(perm=pm,x=%@)[name=string(\"tv\")];\n", attn_heads, seq, head_dim, v_name];

    // === Attention: Q @ K^T * scale + mask → softmax → @ V ===
    [m appendString:@"        bool tx = const()[name=string(\"tx\"), val=bool(false)];\n"];
    [m appendString:@"        bool ty = const()[name=string(\"ty\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> sc1 = matmul(transpose_x=tx,transpose_y=ty,x=q,y=k)[name=string(\"mm1\")];\n", attn_heads, seq, seq];
    [m appendFormat:@"        fp16 scv = const()[name=string(\"scv\"), val=fp16(%f)];\n", scale];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> sc2 = mul(x=sc1,y=scv)[name=string(\"scl\")];\n", attn_heads, seq, seq];
    [m appendFormat:@"        tensor<fp16, [1,1,%d,%d]> cm = const()[name=string(\"cm\"), val=tensor<fp16, [1,1,%d,%d]>(BLOBFILE(path=string(\"@model_path/weights/mask.bin\"), offset=uint64(64)))];\n", seq, seq, seq, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> ms = add(x=sc2,y=cm)[name=string(\"msk\")];\n", attn_heads, seq, seq];
    [m appendString:@"        int32 sax = const()[name=string(\"sax\"), val=int32(-1)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> aw = softmax(axis=sax,x=ms)[name=string(\"sm\")];\n", attn_heads, seq, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> a4 = matmul(transpose_x=tx,transpose_y=tx,x=aw,y=v)[name=string(\"mm2\")];\n", attn_heads, seq, head_dim];

    // === Reshape back to [1, dim, 1, seq] and output projection ===
    [m appendFormat:@"        tensor<fp16, [1,%d,%d,%d]> at = transpose(perm=pm,x=a4)[name=string(\"ta\")];\n", attn_heads, head_dim, seq];
    [m appendFormat:@"        tensor<int32, [4]> os = const()[name=string(\"os\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", dim, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> af = reshape(shape=os,x=at)[name=string(\"ra\")];\n", dim, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> wo_out = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=Wo,x=af)[name=string(\"co\")];\n", dim, seq];
    // Residual: oo = x + wo_out (attention residual inside ANE)
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> oo = add(x=x,y=wo_out)[name=string(\"ares\")];\n", dim, seq];

    // === Output: concat(attn_residual, k_roped_flat, v_flat) for KV-cache ===
    // Flatten k_rot back to [1, kv_dim, 1, seq] and v4 to same
    [m appendFormat:@"        tensor<int32, [4]> kvflat = const()[name=string(\"kvflat\"), val=tensor<int32, [4]>([1,%d,1,%d])];\n", kv_dim, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> kout = reshape(shape=kvflat,x=k_rot)[name=string(\"kout\")];\n", kv_dim, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> vout = reshape(shape=kvflat,x=v4)[name=string(\"vout\")];\n", kv_dim, seq];
    [m appendString:@"        int32 cax = const()[name=string(\"cax\"), val=int32(1)];\n"];
    [m appendString:@"        bool cid = const()[name=string(\"cid\"), val=bool(false)];\n"];
    int out_ch = dim + 2 * kv_dim;
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = concat(axis=cax,interleave=cid,values=(oo,kout,vout))[name=string(\"cat\")];\n", out_ch, seq];
    [m appendString:@"    } -> (out);\n}\n"];

    return m;
}

// Generate pure FFN kernel for ANE (no attention, just SwiGLU FFN)
// Input:  [1, dim, 1, seq] FP16
// Output: [1, dim, 1, seq] FP16 (FFN output, before residual)
// Weights baked: W1 [ffn,dim], W3 [ffn,dim], W2 [dim,ffn], ffn_norm [dim]
// Based on gdc-lm/src/mil_gen.h gen_gdc_fused_ffn pattern (9 nodes/layer)
static NSString *gen_ffn_only_mil(int dim, int ffn_dim, int seq, float rms_eps) {
    float invd = 1.0f / (float)dim;
    NSMutableString *m = [NSMutableString string];
    [m appendString:MIL_HDR];
    [m appendFormat:@"    func main<ios18>(tensor<fp16, [1, %d, 1, %d]> x) {\n", dim, seq];

    // RMSNorm (8 nodes)
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> sq = mul(x=x,y=x)[name=string(\"sq\")];\n", dim, seq];
    [m appendString:@"        tensor<int32, [1]> rax = const()[name=string(\"rax\"), val=tensor<int32, [1]>([1])];\n"];
    [m appendString:@"        bool kd = const()[name=string(\"kd\"), val=bool(true)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss = reduce_sum(x=sq,axes=rax,keep_dims=kd)[name=string(\"ss\")];\n", seq];
    [m appendFormat:@"        fp16 invd = const()[name=string(\"invd\"), val=fp16(%f)];\n", invd];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss2 = mul(x=ss,y=invd)[name=string(\"ss2\")];\n", seq];
    [m appendFormat:@"        fp16 eps = const()[name=string(\"eps\"), val=fp16(%e)];\n", rms_eps];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> ss3 = add(x=ss2,y=eps)[name=string(\"ss3\")];\n", seq];
    [m appendString:@"        fp16 nhalf = const()[name=string(\"nhalf\"), val=fp16(-0.5)];\n"];
    [m appendFormat:@"        tensor<fp16, [1,1,1,%d]> rrms = pow(x=ss3,y=nhalf)[name=string(\"rrms\")];\n", seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xr = mul(x=x,y=rrms)[name=string(\"xr\")];\n", dim, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,1]> rw = const()[name=string(\"rw\"), val=tensor<fp16, [1,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/ffn_norm.bin\"), offset=uint64(64)))];\n", dim, dim];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> xn = mul(x=xr,y=rw)[name=string(\"xn\")];\n", dim, seq];

    // Conv constants
    [m appendString:@"        string pt = const()[name=string(\"pt\"), val=string(\"valid\")];\n"];
    [m appendString:@"        tensor<int32, [2]> st = const()[name=string(\"st\"), val=tensor<int32, [2]>([1,1])];\n"];
    [m appendString:@"        tensor<int32, [4]> pd = const()[name=string(\"pd\"), val=tensor<int32, [4]>([0,0,0,0])];\n"];
    [m appendString:@"        tensor<int32, [2]> dl = const()[name=string(\"dl\"), val=tensor<int32, [2]>([1,1])];\n"];
    [m appendString:@"        int32 gr = const()[name=string(\"gr\"), val=int32(1)];\n"];

    // SwiGLU FFN (9 nodes: 3 conv + sigmoid + 2 mul)
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W1 = const()[name=string(\"W1\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w1.bin\"), offset=uint64(64)))];\n", ffn_dim, dim, ffn_dim, dim];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W3 = const()[name=string(\"W3\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w3.bin\"), offset=uint64(64)))];\n", ffn_dim, dim, ffn_dim, dim];
    [m appendFormat:@"        tensor<fp16, [%d,%d,1,1]> W2 = const()[name=string(\"W2\"), val=tensor<fp16, [%d,%d,1,1]>(BLOBFILE(path=string(\"@model_path/weights/w2.bin\"), offset=uint64(64)))];\n", dim, ffn_dim, dim, ffn_dim];

    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> h1 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W1,x=xn)[name=string(\"c1\")];\n", ffn_dim, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> h3 = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W3,x=xn)[name=string(\"c3\")];\n", ffn_dim, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> sg = sigmoid(x=h1)[name=string(\"sg\")];\n", ffn_dim, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> si = mul(x=h1,y=sg)[name=string(\"si\")];\n", ffn_dim, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> gt = mul(x=si,y=h3)[name=string(\"gt\")];\n", ffn_dim, seq];
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> ffn_out = conv(dilations=dl,groups=gr,pad=pd,pad_type=pt,strides=st,weight=W2,x=gt)[name=string(\"c2\")];\n", dim, seq];

    // Residual: output = input + ffn_out
    [m appendFormat:@"        tensor<fp16, [1,%d,1,%d]> out = add(x=x,y=ffn_out)[name=string(\"res\")];\n", dim, seq];

    [m appendString:@"    } -> (out);\n}\n"];
    return m;
}

// end of ane_mil.h
