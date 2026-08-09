/* bench_conv.c — microbenchmark: one wubu_vk_conv1d call, timed.
 * Usage: bench_conv.exe in_ch n out_ch k  (defaults 512 370 256 7)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "wubu_vk.h"

int main(int argc, char **argv) {
    int in_ch = argc > 1 ? atoi(argv[1]) : 512;
    int n     = argc > 2 ? atoi(argv[2]) : 370;
    int out_ch= argc > 3 ? atoi(argv[3]) : 256;
    int k     = argc > 4 ? atoi(argv[4]) : 7;
    int reps  = argc > 5 ? atoi(argv[5]) : 20;

    size_t in_sz = (size_t)in_ch * n;
    size_t w_sz  = (size_t)out_ch * in_ch * k;
    float *in  = (float *)malloc(in_sz * 4);
    float *w   = (float *)malloc(w_sz * 4);
    float *b   = (float *)malloc(out_ch * 4);
    float *out = (float *)malloc((size_t)out_ch * n * 4);
    for (size_t i = 0; i < in_sz; i++) in[i] = ((float)(i % 7)) * 0.01f;
    for (size_t i = 0; i < w_sz; i++)  w[i]  = ((float)(i % 11)) * 0.001f;
    for (int i = 0; i < out_ch; i++)   b[i]  = 0.01f * i;

    WuBuVk *vk = wubu_vk_create();
    if (!vk) { fprintf(stderr, "VK_CREATE_FAIL\n"); return 2; }
    int n_out = n;
    double t0 = omp_get_wtime();
    for (int r = 0; r < reps; r++) {
        if (wubu_vk_conv1d(vk, in, in_ch, n, w, b, out_ch, k, 1, k / 2, 1, out, n_out) != 0) {
            fprintf(stderr, "CONV_FAIL\n"); return 3;
        }
    }
    double t1 = omp_get_wtime();
    printf("BENCH conv in=%dx%d out=%dx%d k=%d reps=%d: %.3f ms/call (%.1f GMAC/s)\n",
           in_ch, n, out_ch, n_out, k, reps,
           (t1 - t0) / reps * 1000.0,
           (double)reps * out_ch * n_out * in_ch * k / (t1 - t0) / 1e9);
    wubu_vk_destroy(vk);
    free(in); free(w); free(b); free(out);
    return 0;
}
