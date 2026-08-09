/* bench_conv2.c — microbenchmark + CORRECTNESS check vs naive CPU conv.
 * Usage: bench_conv2.exe in_ch n out_ch k  (defaults 192 370 512 7)
 * Verifies the tiled VK conv matches a naive CPU conv (maxdiff) AND times it.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include "wubu_vk.h"

static void cpu_conv1d(const float *in, int in_ch, int n,
                       const float *w, const float *b,
                       int out_ch, int k, int stride, int pad, int dil,
                       float *out, int n_out) {
    for (int oc = 0; oc < out_ch; oc++) {
        for (int j = 0; j < n_out; j++) {
            float acc = b[oc];
            for (int ic = 0; ic < in_ch; ic++) {
                int base = ic * n;
                int wb = (oc * in_ch + ic) * k;
                for (int tap = 0; tap < k; tap++) {
                    int src = j * stride + tap * dil - pad;
                    if (src >= 0 && src < n)
                        acc += in[base + src] * w[wb + tap];
                }
            }
            out[oc * n_out + j] = acc;
        }
    }
}

int main(int argc, char **argv) {
    int in_ch = argc > 1 ? atoi(argv[1]) : 192;
    int n     = argc > 2 ? atoi(argv[2]) : 370;
    int out_ch= argc > 3 ? atoi(argv[3]) : 512;
    int k     = argc > 4 ? atoi(argv[4]) : 7;
    int reps  = argc > 5 ? atoi(argv[5]) : 10;

    size_t in_sz = (size_t)in_ch * n;
    size_t w_sz  = (size_t)out_ch * in_ch * k;
    float *in  = (float *)malloc(in_sz * 4);
    float *w   = (float *)malloc(w_sz * 4);
    float *b   = (float *)malloc(out_ch * 4);
    float *out = (float *)malloc((size_t)out_ch * n * 4);
    float *ref = (float *)malloc((size_t)out_ch * n * 4);
    for (size_t i = 0; i < in_sz; i++) in[i] = ((float)(i % 7)) * 0.01f - 0.03f;
    for (size_t i = 0; i < w_sz; i++)  w[i]  = ((float)(i % 11)) * 0.001f - 0.005f;
    for (int i = 0; i < out_ch; i++)   b[i]  = 0.01f * i - 0.1f;

    int pad = k / 2;
    cpu_conv1d(in, in_ch, n, w, b, out_ch, k, 1, pad, 1, ref, n);

    WuBuVk *vk = wubu_vk_create();
    if (!vk) { fprintf(stderr, "VK_CREATE_FAIL\n"); return 2; }
    int n_out = n;
    if (wubu_vk_conv1d(vk, in, in_ch, n, w, b, out_ch, k, 1, pad, 1, out, n_out) != 0) {
        fprintf(stderr, "CONV_FAIL\n"); return 3;
    }
    double maxdiff = 0.0, sum = 0.0, sumsq = 0.0;
    for (size_t i = 0; i < (size_t)out_ch * n_out; i++) {
        double d = fabs((double)out[i] - ref[i]);
        if (d > maxdiff) maxdiff = d;
        sum += d; sumsq += d * d;
    }
    size_t cnt = (size_t)out_ch * n_out;
    printf("CORRECT maxdiff=%.6f meandiff=%.6f rms=%.6f (n=%zu)\n",
           maxdiff, sum / cnt, sqrt(sumsq / cnt), cnt);

    double t0 = omp_get_wtime();
    for (int r = 0; r < reps; r++)
        wubu_vk_conv1d(vk, in, in_ch, n, w, b, out_ch, k, 1, pad, 1, out, n_out);
    double t1 = omp_get_wtime();
    printf("BENCH conv in=%dx%d out=%dx%d k=%d reps=%d: %.3f ms/call (%.1f GMAC/s)\n",
           in_ch, n, out_ch, n_out, k, reps,
           (t1 - t0) / reps * 1000.0,
           (double)reps * out_ch * n_out * in_ch * k / (t1 - t0) / 1e9);
    wubu_vk_destroy(vk);
    free(in); free(w); free(b); free(out); free(ref);
    return 0;
}
