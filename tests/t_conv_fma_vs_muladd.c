/* t_conv_fma_vs_muladd.c — does mul+add beat FMA in the ACTUAL conv1d shape?
 * OC-blocked register conv, 32-sample j-tiles, k=3, in_ch=out_ch=192, n=798
 * (the real MRF L3 conv shape). Two kernels: FMA path (current) and
 * mul+add path. Both accumulate 4-oc x 4 tiles. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <immintrin.h>

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* FMA register conv — mirrors conv1d_c's interior path exactly */
static void conv_fma(const float *in, int in_ch, int n, const float *w,
                     int out_ch, int k, int pad, int dil, float *out) {
    const int OC_BLK = 4;
    int j_in_lo = pad;
    int j_in_hi = n - (k - 1) * dil + pad;
    int jr_max = n - 32 - (k - 1) * dil + pad;
    if (jr_max > j_in_hi - 32) jr_max = j_in_hi - 32;
    for (int oc0 = 0; oc0 < out_ch; oc0 += OC_BLK) {
        int n_oc = OC_BLK;
        float *orow0[OC_BLK]; const float *wv0[OC_BLK];
        for (int oc = 0; oc < n_oc; oc++) {
            orow0[oc] = out + (size_t)(oc0 + oc) * n;
            wv0[oc] = w + (size_t)(oc0 + oc) * in_ch * k;
        }
        int jr = j_in_lo;
        for (; jr <= jr_max; jr += 32) {
            __m256 a0[OC_BLK], a1[OC_BLK], a2[OC_BLK], a3[OC_BLK];
            for (int oc = 0; oc < n_oc; oc++) {
                a0[oc] = _mm256_setzero_ps(); a1[oc] = _mm256_setzero_ps();
                a2[oc] = _mm256_setzero_ps(); a3[oc] = _mm256_setzero_ps();
            }
            for (int ick = 0; ick < in_ch; ick++) {
                const float *ir2 = in + (size_t)ick * n + jr - pad;
                for (int tap = 0; tap < k; tap++) {
                    __m256 iv0 = _mm256_loadu_ps(ir2 + tap * dil);
                    __m256 iv1 = _mm256_loadu_ps(ir2 + 8 + tap * dil);
                    __m256 iv2 = _mm256_loadu_ps(ir2 + 16 + tap * dil);
                    __m256 iv3 = _mm256_loadu_ps(ir2 + 24 + tap * dil);
                    for (int oc = 0; oc < n_oc; oc++) {
                        float wt = wv0[oc][(size_t)ick * k + tap];
                        __m256 wv8 = _mm256_set1_ps(wt);
                        a0[oc] = _mm256_fmadd_ps(iv0, wv8, a0[oc]);
                        a1[oc] = _mm256_fmadd_ps(iv1, wv8, a1[oc]);
                        a2[oc] = _mm256_fmadd_ps(iv2, wv8, a2[oc]);
                        a3[oc] = _mm256_fmadd_ps(iv3, wv8, a3[oc]);
                    }
                }
            }
            for (int oc = 0; oc < n_oc; oc++) {
                _mm256_storeu_ps(orow0[oc] + jr, a0[oc]);
                _mm256_storeu_ps(orow0[oc] + jr + 8, a1[oc]);
                _mm256_storeu_ps(orow0[oc] + jr + 16, a2[oc]);
                _mm256_storeu_ps(orow0[oc] + jr + 24, a3[oc]);
            }
        }
    }
}

/* mul+add version — same structure, separate mul (FP0/1) + add (FP2/3) */
static void conv_muladd(const float *in, int in_ch, int n, const float *w,
                        int out_ch, int k, int pad, int dil, float *out) {
    const int OC_BLK = 4;
    int j_in_lo = pad;
    int j_in_hi = n - (k - 1) * dil + pad;
    int jr_max = n - 32 - (k - 1) * dil + pad;
    if (jr_max > j_in_hi - 32) jr_max = j_in_hi - 32;
    for (int oc0 = 0; oc0 < out_ch; oc0 += OC_BLK) {
        int n_oc = OC_BLK;
        float *orow0[OC_BLK]; const float *wv0[OC_BLK];
        for (int oc = 0; oc < n_oc; oc++) {
            orow0[oc] = out + (size_t)(oc0 + oc) * n;
            wv0[oc] = w + (size_t)(oc0 + oc) * in_ch * k;
        }
        int jr = j_in_lo;
        for (; jr <= jr_max; jr += 32) {
            __m256 a0[OC_BLK], a1[OC_BLK], a2[OC_BLK], a3[OC_BLK];
            for (int oc = 0; oc < n_oc; oc++) {
                a0[oc] = _mm256_setzero_ps(); a1[oc] = _mm256_setzero_ps();
                a2[oc] = _mm256_setzero_ps(); a3[oc] = _mm256_setzero_ps();
            }
            for (int ick = 0; ick < in_ch; ick++) {
                const float *ir2 = in + (size_t)ick * n + jr - pad;
                for (int tap = 0; tap < k; tap++) {
                    __m256 iv0 = _mm256_loadu_ps(ir2 + tap * dil);
                    __m256 iv1 = _mm256_loadu_ps(ir2 + 8 + tap * dil);
                    __m256 iv2 = _mm256_loadu_ps(ir2 + 16 + tap * dil);
                    __m256 iv3 = _mm256_loadu_ps(ir2 + 24 + tap * dil);
                    for (int oc = 0; oc < n_oc; oc++) {
                        float wt = wv0[oc][(size_t)ick * k + tap];
                        __m256 wv8 = _mm256_set1_ps(wt);
                        __m256 p0 = _mm256_mul_ps(iv0, wv8);
                        a0[oc] = _mm256_add_ps(p0, a0[oc]);
                        p0 = _mm256_mul_ps(iv1, wv8);
                        a1[oc] = _mm256_add_ps(p0, a1[oc]);
                        p0 = _mm256_mul_ps(iv2, wv8);
                        a2[oc] = _mm256_add_ps(p0, a2[oc]);
                        p0 = _mm256_mul_ps(iv3, wv8);
                        a3[oc] = _mm256_add_ps(p0, a3[oc]);
                    }
                }
            }
            for (int oc = 0; oc < n_oc; oc++) {
                _mm256_storeu_ps(orow0[oc] + jr, a0[oc]);
                _mm256_storeu_ps(orow0[oc] + jr + 8, a1[oc]);
                _mm256_storeu_ps(orow0[oc] + jr + 16, a2[oc]);
                _mm256_storeu_ps(orow0[oc] + jr + 24, a3[oc]);
            }
        }
    }
}

int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 798;
    int iters = argc > 2 ? atoi(argv[2]) : 200;
    int in_ch = 192, out_ch = 192, k = 3, pad = 1, dil = 1;
    float *in = malloc((size_t)in_ch * n * sizeof(float));
    float *w = malloc((size_t)out_ch * in_ch * k * sizeof(float));
    float *o1 = calloc((size_t)out_ch * n, sizeof(float));
    float *o2 = calloc((size_t)out_ch * n, sizeof(float));
    srand(3);
    for (int i = 0; i < in_ch * n; i++) in[i] = ((float)rand()/RAND_MAX-0.5f)*0.2f;
    for (int i = 0; i < out_ch * in_ch * k; i++) w[i] = ((float)rand()/RAND_MAX-0.5f)*0.02f;

    /* correctness: maxdiff between the two kernels */
    conv_fma(in, in_ch, n, w, out_ch, k, pad, dil, o1);
    conv_muladd(in, in_ch, n, w, out_ch, k, pad, dil, o2);
    double maxdiff = 0; int nbad = 0;
    for (int i = 0; i < out_ch * n; i++) {
        double d = fabs((double)o1[i] - o2[i]);
        if (d > maxdiff) maxdiff = d;
        if (d > 1e-6) nbad++;
    }
    printf("n=%d maxdiff(fma vs muladd)=%.6f nbad=%d (of %d)\n", n, maxdiff, nbad, out_ch*n);

    /* speed */
    double t0 = now_s();
    for (int it = 0; it < iters; it++) conv_fma(in, in_ch, n, w, out_ch, k, pad, dil, o1);
    double t1 = now_s();
    for (int it = 0; it < iters; it++) conv_muladd(in, in_ch, n, w, out_ch, k, pad, dil, o2);
    double t2 = now_s();
    double macs = (double)in_ch * out_ch * k * n * iters;
    printf("FMA    : %.3f s  (%.2f GMAC/s)\n", t1-t0, macs/(t1-t0)/1e9);
    printf("MUL+ADD: %.3f s  (%.2f GMAC/s)\n", t2-t1, macs/(t2-t1)/1e9);
    printf("ratio  : %.3fx\n", (t1-t0)/(t2-t1));
    free(in); free(w); free(o1); free(o2);
    return 0;
}
