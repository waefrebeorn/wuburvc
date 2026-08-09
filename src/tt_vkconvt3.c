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
    if (va == 0 || vb == 0) return 0.0;
    return cov / sqrt(va * vb);
}

int main(void) {
    WuBuVk *vk = wubu_vk_create();
    if (!vk) { printf("vk create FAIL\n"); return 1; }
    /* real upsample convT dims (L=3): in_ch=64, out_ch=32, k=16, stride=10, n_in=2560, n_out=5120 */
    int ic3 = 64, oc3 = 32, k3 = 16, st3 = 10, nin3 = 2560, nout3 = 5120;
    size_t in3n = (size_t)ic3 * nin3, w3n = (size_t)ic3 * oc3 * k3;
    float *in3 = (float *)malloc(in3n * 4);
    float *w3 = (float *)malloc(w3n * 4);
    float *do3 = (float *)malloc((size_t)oc3 * nout3 * 4);
    float *di3 = (float *)calloc(in3n, 4);
    float *dw3 = (float *)calloc(w3n, 4);
    float *db3 = (float *)calloc((size_t)oc3, 4);
    unsigned s = 321;
    for (size_t i = 0; i < in3n; i++) { s = s * 1664525u + 1013904223u; in3[i] = ((int)(s >> 8) % 2000 - 1000) / 1000.0f; }
    for (size_t i = 0; i < w3n; i++) { s = s * 1664525u + 1013904223u; w3[i] = ((int)(s >> 8) % 2000 - 1000) / 1000.0f; }
    for (size_t i = 0; i < (size_t)oc3 * nout3; i++) { s = s * 1664525u + 1013904223u; do3[i] = ((int)(s >> 8) % 2000 - 1000) / 1000.0f; }
    int rc6 = wubu_vk_bwd_convt1d(vk, in3, ic3, w3, do3, oc3, k3, st3, 3, nin3, nout3, di3, dw3, db3);
    /* CPU reference */
    float *di_c = (float *)calloc(in3n, 4);
    float *dw_c = (float *)calloc(w3n, 4);
    float *db_c = (float *)calloc((size_t)oc3, 4);
    for (int ic = 0; ic < ic3; ic++) {
        for (int oc = 0; oc < oc3; oc++) {
            const float *wv = w3 + (size_t)(ic * oc3 + oc) * k3;
            const float *do_ = do3 + (size_t)oc * nout3;
            float *din_row = di_c + (size_t)ic * nin3;
            float *dw_row = dw_c + (size_t)(ic * oc3 + oc) * k3;
            for (int i = 0; i < nin3; i++) {
                int j_start = i * st3 - 3;
                float din_acc = 0.0f;
                float x = in3[(size_t)ic * nin3 + i];
                for (int tap = 0; tap < k3; tap++) {
                    int j = j_start + tap;
                    if (j >= 0 && j < nout3) {
                        float g = do_[j];
                        din_acc += g * wv[tap];
                        dw_row[tap] += x * g;
                    }
                }
                din_row[i] += din_acc;
            }
        }
    }
    for (int oc = 0; oc < oc3; oc++) {
        float acc = 0.0f;
        for (int j = 0; j < nout3; j++) acc += do3[(size_t)oc * nout3 + j];
        db_c[oc] = acc;
    }
    double cdi = corr1(di_c, di3, (int)in3n);
    double cdw = corr1(dw_c, dw3, (int)w3n);
    double cdb = corr1(db_c, db3, oc3);
    printf("convt L3 rc=%d din corr=%.8f dw corr=%.8f db corr=%.8f di[0]=%.6f\n", rc6, cdi, cdw, cdb, di3[0]);
    fflush(stdout);
    wubu_vk_destroy(vk);
    return 0;
}
