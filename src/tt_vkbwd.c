#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "wubu_vk.h"

static double corr1(const float *a, const float *b, int n) {
    double ma = 0, mb = 0;
    for (int i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
    ma /= n; mb /= n;
    double va = 0, vb = 0, cov = 0;
    for (int i = 0; i < n; i++) {
        double da = a[i] - ma, db = b[i] - mb;
        va += da * da; vb += db * db; cov += da * db;
    }
    return cov / sqrt(va * vb);
}

/* CPU reference conv1d_bwd (matches wubu_train.c) */
static void cpu_bwd(const float *in, const float *w, const float *dout,
                    int in_ch, int out_ch, int k, int stride, int pad, int dil,
                    int n_in, int n_out, float *din, float *dw, float *db) {
    memset(dw, 0, (size_t)out_ch * in_ch * k * sizeof(float));
    if (db) memset(db, 0, (size_t)out_ch * sizeof(float));
    memset(din, 0, (size_t)in_ch * n_in * sizeof(float));
    for (int ic = 0; ic < in_ch; ic++) {
        for (int oc = 0; oc < out_ch; oc++) {
            const float *wv = w + (size_t)oc * in_ch * k;
            const float *do_ = dout + (size_t)oc * n_out;
            float *din_row = din + (size_t)ic * n_in;
            float *dw_row = dw + (size_t)oc * in_ch * k + (size_t)ic * k;
            for (int j = 0; j < n_out; j++) {
                float g = do_[j];
                int j_in = j * stride - pad;
                for (int tap = 0; tap < k; tap++) {
                    int src = j_in + dil * tap;
                    if (src >= 0 && src < n_in) {
                        float x = in[(size_t)ic * n_in + src];
                        dw_row[tap] += x * g;
                        din_row[src] += g * wv[(size_t)ic * k + tap];
                    }
                }
            }
        }
    }
    if (db) {
        for (int oc = 0; oc < out_ch; oc++) {
            float acc = 0.0f;
            for (int j = 0; j < n_out; j++) acc += dout[(size_t)oc * n_out + j];
            db[oc] = acc;
        }
    }
}

int main(void) {
    WuBuVk *vk = wubu_vk_create();
    if (!vk) { printf("vk create FAIL\n"); return 1; }
    /* small test: in_ch=8, out_ch=4, k=3, n=64 */
    int in_ch = 8, out_ch = 4, k = 3, n = 64, pad = 1, dil = 1;
    size_t in_n = (size_t)in_ch * n, w_n = (size_t)out_ch * in_ch * k;
    float *in = (float *)malloc(in_n * 4);
    float *w = (float *)malloc(w_n * 4);
    float *dout = (float *)malloc((size_t)out_ch * n * 4);
    float *din_c = (float *)malloc(in_n * 4);
    float *din_g = (float *)malloc(in_n * 4);
    float *dw_c = (float *)malloc(w_n * 4);
    float *dw_g = (float *)malloc(w_n * 4);
    float *db_c = (float *)malloc((size_t)out_ch * 4);
    float *db_g = (float *)malloc((size_t)out_ch * 4);
    unsigned s = 7;
    for (size_t i = 0; i < in_n; i++) { s = s * 1664525u + 1013904223u; in[i] = ((int)(s >> 8) % 2000 - 1000) / 1000.0f; }
    for (size_t i = 0; i < w_n; i++) { s = s * 1664525u + 1013904223u; w[i] = ((int)(s >> 8) % 2000 - 1000) / 1000.0f; }
    for (size_t i = 0; i < (size_t)out_ch * n; i++) { s = s * 1664525u + 1013904223u; dout[i] = ((int)(s >> 8) % 2000 - 1000) / 1000.0f; }

    cpu_bwd(in, w, dout, in_ch, out_ch, k, 1, pad, dil, n, n, din_c, dw_c, db_c);
    printf("cpu done: dw[0]=%.4f din[0]=%.4f db[0]=%.4f\n", dw_c[0], din_c[0], db_c[0]);

    int rc = wubu_vk_bwd_conv1d(vk, in, in_ch, w, dout, out_ch, k, 1, pad, dil, n, n,
                                din_g, dw_g, db_g);
    printf("vk bwd conv rc=%d\n", rc);
    printf("vk: dw[0]=%.4f din[0]=%.4f db[0]=%.4f\n", dw_g[0], din_g[0], db_g[0]);
    double cw = corr1(dw_c, dw_g, (int)w_n);
    double cd = corr1(din_c, din_g, (int)in_n);
    printf("dw corr=%.8f  din corr=%.8f\n", cw, cd);

    /* convT bwd test: in_ch=4, out_ch=8, k=3, stride=2, n_in=32, n_out=64 */
    int in_ch_t = 4, out_ch_t = 8, k_t = 3, n_in_t = 32, stride_t = 2, pad_t = 1;
    int n_out_t = (n_in_t - 1) * stride_t - 2 * pad_t + k_t;
    size_t in_n_t = (size_t)in_ch_t * n_in_t, w_n_t = (size_t)in_ch_t * out_ch_t * k_t;
    float *in_t = (float *)malloc(in_n_t * 4);
    float *w_t = (float *)malloc(w_n_t * 4);
    float *dout_t = (float *)malloc((size_t)out_ch_t * n_out_t * 4);
    float *din_tc = (float *)malloc(in_n_t * 4);
    float *din_tg = (float *)malloc(in_n_t * 4);
    float *dw_tc = (float *)malloc(w_n_t * 4);
    float *dw_tg = (float *)malloc(w_n_t * 4);
    float *db_tc = (float *)malloc((size_t)out_ch_t * 4);
    float *db_tg = (float *)malloc((size_t)out_ch_t * 4);
    s = 99;
    for (size_t i = 0; i < in_n_t; i++) { s = s * 1664525u + 1013904223u; in_t[i] = ((int)(s >> 8) % 2000 - 1000) / 1000.0f; }
    for (size_t i = 0; i < w_n_t; i++) { s = s * 1664525u + 1013904223u; w_t[i] = ((int)(s >> 8) % 2000 - 1000) / 1000.0f; }
    for (size_t i = 0; i < (size_t)out_ch_t * n_out_t; i++) { s = s * 1664525u + 1013904223u; dout_t[i] = ((int)(s >> 8) % 2000 - 1000) / 1000.0f; }
    /* CPU convT bwd */
    memset(dw_tc, 0, w_n_t * 4);
    memset(din_tc, 0, in_n_t * 4);
    for (int ic = 0; ic < in_ch_t; ic++) {
        for (int oc = 0; oc < out_ch_t; oc++) {
            const float *wv = w_t + (size_t)(ic * out_ch_t + oc) * k_t;
            const float *do_ = dout_t + (size_t)oc * n_out_t;
            float *din_row = din_tc + (size_t)ic * n_in_t;
            float *dw_row = dw_tc + (size_t)(ic * out_ch_t + oc) * k_t;
            for (int i = 0; i < n_in_t; i++) {
                int j_start = i * stride_t - pad_t;
                float din_acc = 0.0f;
                float x = in_t[(size_t)ic * n_in_t + i];
                for (int tap = 0; tap < k_t; tap++) {
                    int j = j_start + tap;
                    if (j >= 0 && j < n_out_t) {
                        float g = do_[j];
                        din_acc += g * wv[tap];
                        dw_row[tap] += x * g;
                    }
                }
                din_row[i] += din_acc;
            }
        }
    }
    for (int oc = 0; oc < out_ch_t; oc++) {
        float acc = 0.0f;
        for (int j = 0; j < n_out_t; j++) acc += dout_t[(size_t)oc * n_out_t + j];
        db_tc[oc] = acc;
    }
    int rc2 = wubu_vk_bwd_convt1d(vk, in_t, in_ch_t, w_t, dout_t, out_ch_t, k_t, stride_t, pad_t,
                                  n_in_t, n_out_t, din_tg, dw_tg, db_tg);
    printf("vk bwd convt rc=%d\n", rc2);
    printf("cpu: dw[0]=%.4f din[0]=%.4f db[0]=%.4f\n", dw_tc[0], din_tc[0], db_tc[0]);
    printf("vk : dw[0]=%.4f din[0]=%.4f db[0]=%.4f\n", dw_tg[0], din_tg[0], db_tg[0]);
    printf("convt dw corr=%.8f din corr=%.8f db corr=%.8f\n",
           corr1(dw_tc, dw_tg, (int)w_n_t), corr1(din_tc, din_tg, (int)in_n_t),
           corr1(db_tc, db_tg, out_ch_t));

    /* bwd_act tests: lrelu (mode 0) + tanh (mode 1) */
    int na = 1000;
    float *x = (float *)malloc(na * 4);
    float *dout_a = (float *)malloc(na * 4);
    float *din_a = (float *)malloc(na * 4);
    float *din_ref = (float *)malloc(na * 4);
    s = 555;
    for (int i = 0; i < na; i++) { s = s * 1664525u + 1013904223u; x[i] = ((int)(s >> 8) % 4000 - 2000) / 1000.0f; }
    for (int i = 0; i < na; i++) { s = s * 1664525u + 1013904223u; dout_a[i] = ((int)(s >> 8) % 2000 - 1000) / 1000.0f; }
    for (int i = 0; i < na; i++) din_ref[i] = dout_a[i] * (x[i] > 0 ? 1.0f : 0.1f);
    int rc3 = wubu_vk_bwd_act(vk, x, dout_a, din_a, 0, (size_t)na);
    printf("vk bwd lrelu rc=%d corr=%.8f\n", rc3, corr1(din_ref, din_a, na));
    for (int i = 0; i < na; i++) din_ref[i] = dout_a[i] * (1.0f - x[i] * x[i]);
    rc3 = wubu_vk_bwd_act(vk, x, dout_a, din_a, 1, (size_t)na);
    printf("vk bwd tanh  rc=%d corr=%.8f\n", rc3, corr1(din_ref, din_a, na));

    /* conv_post-like bwd: in_ch=32, out_ch=1, k=7, n=5120 (grid 1 x 20) */
    int np = 5120;
    float *in_p = (float *)malloc((size_t)32 * np * 4);
    float *w_p = (float *)malloc((size_t)32 * 7 * 4);
    float *dout_p = (float *)malloc((size_t)np * 4);
    float *din_p = (float *)malloc((size_t)32 * np * 4);
    float *dw_p = (float *)malloc((size_t)32 * 7 * 4);
    float *db_p = (float *)malloc(4);
    s = 777;
    for (int i = 0; i < 32 * np; i++) { s = s * 1664525u + 1013904223u; in_p[i] = ((int)(s >> 8) % 2000 - 1000) / 1000.0f; }
    for (int i = 0; i < 32 * 7; i++) { s = s * 1664525u + 1013904223u; w_p[i] = ((int)(s >> 8) % 2000 - 1000) / 1000.0f; }
    for (int i = 0; i < np; i++) { s = s * 1664525u + 1013904223u; dout_p[i] = ((int)(s >> 8) % 2000 - 1000) / 1000.0f; }
    memset(din_p, 0, (size_t)32 * np * 4);
    memset(dw_p, 0, (size_t)32 * 7 * 4);
    memset(db_p, 0, 4);
    int rc4 = wubu_vk_bwd_conv1d(vk, in_p, 32, w_p, dout_p, 1, 7, 1, 3, 1, np, np,
                          din_p, dw_p, db_p);
 printf("vk bwd conv_post rc=%d dw[0]=%.4f\n", rc4, dw_p[0]);

 /* lrelu_bwd with 163840 elements (32*5120) — the crash dims */
 float *x_big = (float *)malloc(163840 * 4);
 float *do_big = (float *)malloc(163840 * 4);
 float *di_big = (float *)malloc(163840 * 4);
 s = 888;
 for (int i = 0; i < 163840; i++) { s = s * 1664525u + 1013904223u; x_big[i] = ((int)(s >> 8) % 4000 - 2000) / 1000.0f; }
 for (int i = 0; i < 163840; i++) { s = s * 1664525u + 1013904223u; do_big[i] = ((int)(s >> 8) % 2000 - 1000) / 1000.0f; }
 int rc5 = wubu_vk_bwd_act(vk, x_big, do_big, di_big, 0, 163840);
 printf("vk bwd lrelu BIG rc=%d di[0]=%.4f\n", rc5, di_big[0]);
 fflush(stdout);

 wubu_vk_destroy(vk);
 return 0;
 }
