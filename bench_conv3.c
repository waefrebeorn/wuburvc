#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "wubu_vk.h"
static void cpu_conv1d(const float *in, int in_ch, int n, const float *w, const float *b,
                       int out_ch, int k, int stride, int pad, int dil, float *out, int n_out) {
    for (int oc = 0; oc < out_ch; oc++)
        for (int j = 0; j < n_out; j++) {
            float acc = b[oc];
            for (int ic = 0; ic < in_ch; ic++) {
                int base = ic * n, wb = (oc * in_ch + ic) * k;
                for (int tap = 0; tap < k; tap++) {
                    int src = j * stride + tap * dil - pad;
                    if (src >= 0 && src < n) acc += in[base + src] * w[wb + tap];
                }
            }
            out[oc * n_out + j] = acc;
        }
}
int main(int argc, char **argv) {
    int in_ch = argc > 1 ? atoi(argv[1]) : 8, n = argc > 2 ? atoi(argv[2]) : 129;
    int out_ch = argc > 3 ? atoi(argv[3]) : 8, k = argc > 4 ? atoi(argv[4]) : 3;
    int pad = argc > 5 ? atoi(argv[5]) : 3;
    size_t in_sz = (size_t)in_ch * n, w_sz = (size_t)out_ch * in_ch * k;
    float *in = malloc(in_sz*4), *w = malloc(w_sz*4), *b = malloc(out_ch*4);
    float *out = malloc((size_t)out_ch*n*4), *ref = malloc((size_t)out_ch*n*4);
    for (size_t i = 0; i < in_sz; i++) in[i] = ((float)(i % 7)) * 0.01f - 0.03f;
    for (size_t i = 0; i < w_sz; i++)  w[i]  = ((float)(i % 11)) * 0.001f - 0.005f;
    for (int i = 0; i < out_ch; i++)   b[i]  = 0.01f * i - 0.1f;
    cpu_conv1d(in, in_ch, n, w, b, out_ch, k, 1, pad, 1, ref, n);
    WuBuVk *vk = wubu_vk_create(); if (!vk) return 2;
    wubu_vk_conv1d(vk, in, in_ch, n, w, b, out_ch, k, 1, pad, 1, out, n);
    double maxdiff = 0; size_t bad = 0, total = (size_t)out_ch*n;
    for (size_t i = 0; i < total; i++) {
        double d = fabs((double)out[i] - ref[i]);
        if (d > maxdiff) maxdiff = d;
        if (d > 1e-3) { bad++; if (bad <= 5) printf("  bad[%zu] j=%zu oc=%zu out=%.4f ref=%.4f\n", i, i % n, i / n, out[i], ref[i]); }
    }
    printf("pad=%d in=%dx%d out=%dx%d k=%d: maxdiff=%.6f bad=%zu/%zu\n", pad, in_ch, n, out_ch, n, k, maxdiff, bad, total);
    wubu_vk_destroy(vk); return 0;
}
