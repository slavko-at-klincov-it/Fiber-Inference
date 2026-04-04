// dequant.h — CPU dequantization helpers (Q4_K, Q6_K → FP16)
// Shared between gpu_ffn.m (embedding) and model.m (ANE weight prep)
#pragma once
#include <stdint.h>
#include <string.h>

// Dequantize one row of Q4_K data to FP16
// data: pointer to Q4_K block data, row: row index, out: FP16 output, dim: row width
static inline void dequant_q4k_row_f16(const void *data, int row, _Float16 *out, int dim) {
    int bpr = dim / 256;
    const uint8_t *base = (const uint8_t *)data + (size_t)row * bpr * 144;
    for (int b = 0; b < bpr; b++) {
        const uint8_t *bl = base + b * 144;
        _Float16 dh, dmh; memcpy(&dh, bl, 2); memcpy(&dmh, bl+2, 2);
        float d = (float)dh, dm = (float)dmh;
        const uint8_t *sc = bl+4, *qs = bl+16;
        for (int j = 0; j < 4; j++) {
            float s1,m1,s2,m2; int is = j*2;
            if(is<4){s1=(float)(sc[is]&63);m1=(float)(sc[is+4]&63);}
            else{s1=(float)((sc[is+4]&0xF)|((sc[is-4]>>6)<<4));m1=(float)((sc[is+4]>>4)|((sc[is]>>6)<<4));}
            is=j*2+1;
            if(is<4){s2=(float)(sc[is]&63);m2=(float)(sc[is+4]&63);}
            else{s2=(float)((sc[is+4]&0xF)|((sc[is-4]>>6)<<4));m2=(float)((sc[is+4]>>4)|((sc[is]>>6)<<4));}
            for(int l=0;l<32;l++){
                uint8_t qv=qs[j*32+l];
                out[b*256+j*64+l]    = (_Float16)(d*s1*(float)(qv&0xF)-dm*m1);
                out[b*256+j*64+l+32] = (_Float16)(d*s2*(float)(qv>>4)-dm*m2);
            }
        }
    }
}

// Dequantize one row of Q6_K data to FP16
static inline void dequant_q6k_row_f16(const void *data, int row, _Float16 *out, int dim) {
    int bpr = dim / 256;
    const uint8_t *base = (const uint8_t *)data + (size_t)row * bpr * 210;
    for (int b = 0; b < bpr; b++) {
        const uint8_t *bl = base + b * 210;
        const uint8_t *ql = bl, *qh = bl+128;
        const int8_t *sc = (const int8_t *)(bl+192);
        _Float16 dh; memcpy(&dh, bl+208, 2); float d = (float)dh;
        int bi = b * 256;
        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; l++) {
                int is = l/16;
                const int8_t q1 = (int8_t)((ql[l]&0xF)|(((qh[l]>>0)&3)<<4))-32;
                const int8_t q2 = (int8_t)((ql[l+32]&0xF)|(((qh[l]>>2)&3)<<4))-32;
                const int8_t q3 = (int8_t)((ql[l]>>4)|(((qh[l]>>4)&3)<<4))-32;
                const int8_t q4 = (int8_t)((ql[l+32]>>4)|(((qh[l]>>6)&3)<<4))-32;
                out[bi+n+l]    = (_Float16)(d*(float)sc[is+0]*(float)q1);
                out[bi+n+l+32] = (_Float16)(d*(float)sc[is+2]*(float)q2);
                out[bi+n+l+64] = (_Float16)(d*(float)sc[is+4]*(float)q3);
                out[bi+n+l+96] = (_Float16)(d*(float)sc[is+6]*(float)q4);
            }
            ql += 64; qh += 32; sc += 8;
        }
    }
}
