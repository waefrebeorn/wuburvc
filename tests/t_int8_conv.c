/* t_int8_conv.c — DA benchmark v2: INT8 W8A8 conv (im2col + Q8_0 GEMM) vs
 * FP32 AVX2 conv on Zen2 (Ryzen 5 3600). Shape = MRF conv1d: in=192, out=192,
 * k=3 (and 7/11), n=8192. im2col makes the reduction dim (ic,tap) contiguous
 * so maddubs can pair lanes correctly.
 * Build: gcc -O3 -mavx2 -mfma -march=znver2 -fopenmp t_int8_conv.c -o t_int8_conv.exe
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <immintrin.h>

/* FP32 AVX2 input-stationary conv (mirror of conv1d_c interior) */
static void conv_fp32(const float *in, int in_ch, int n, const float *w,
                      int out_ch, int k, int pad, int dil, float *out) {
    int n_out = n - dil * (k - 1) + 2 * pad - 1 + 1; /* stride 1 */
    memset(out, 0, (size_t)out_ch * n_out * sizeof(float));
    const int TILE = 2048;
#pragma omp parallel for schedule(dynamic, 4) num_threads(8) if(n_out >= TILE)
    for (int jb = 0; jb < n_out; jb += TILE) {
        int j_hi = jb + TILE < n_out ? jb + TILE : n_out;
        enum { OC_BLK = 4 };
        int j_in_lo = jb > pad ? jb : pad;   /* max(jb,pad) */
        int j_in_hi = n - (k - 1) * dil + pad;
        if (j_in_hi > j_hi) j_in_hi = j_hi;
        int jr_max = n - 32 - (k - 1) * dil + pad;
        if (jr_max > j_in_hi - 32) jr_max = j_in_hi - 32;
        for (int oc0 = 0; oc0 < out_ch; oc0 += OC_BLK) {
            int n_oc = OC_BLK;
            if (oc0 + n_oc > out_ch) n_oc = out_ch - oc0;
            float *orow0[OC_BLK]; const float *wv0[OC_BLK]; float bias0[OC_BLK];
            for (int oc = 0; oc < n_oc; oc++) {
                orow0[oc] = out + (size_t)(oc0 + oc) * n_out;
                wv0[oc] = w + (size_t)(oc0 + oc) * in_ch * k;
                bias0[oc] = 0.0f;
            }
            for (int j = jb; j < j_in_lo && j < j_hi; j++)
                for (int oc = 0; oc < n_oc; oc++) {
                    const float *wv = wv0[oc];
                    float acc = orow0[oc][j];
                    for (int ick = 0; ick < in_ch; ick++) {
                        const float *ir2 = in + (size_t)ick * n;
                        const float *wv2 = wv + (size_t)ick * k;
                        for (int tap = 0; tap < k; tap++) {
                            int src = j + tap * dil - pad;
                            if (src >= 0 && src < n) acc += ir2[src] * wv2[tap];
                        }
                    }
                    orow0[oc][j] = acc;
                }
            int jr = j_in_lo;
            for (; jr <= jr_max; jr += 32) {
                __m256 a0[OC_BLK], a1[OC_BLK], a2[OC_BLK], a3[OC_BLK];
                for (int oc = 0; oc < n_oc; oc++) {
                    a0[oc] = _mm256_loadu_ps(orow0[oc] + jr);
                    a1[oc] = _mm256_loadu_ps(orow0[oc] + jr + 8);
                    a2[oc] = _mm256_loadu_ps(orow0[oc] + jr + 16);
                    a3[oc] = _mm256_loadu_ps(orow0[oc] + jr + 24);
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
            for (; jr < j_in_hi; jr++)
                for (int oc = 0; oc < n_oc; oc++) {
                    const float *wv = wv0[oc];
                    float acc = orow0[oc][jr];
                    for (int ick = 0; ick < in_ch; ick++) {
                        const float *ir2 = in + (size_t)ick * n;
                        const float *wv2 = wv + (size_t)ick * k;
                        for (int tap = 0; tap < k; tap++)
                            acc += ir2[jr + tap * dil - pad] * wv2[tap];
                    }
                    orow0[oc][jr] = acc;
                }
        }
    }
}

/* INT8 W8A8 conv via im2col + Q8_0-style GEMM.
 * Activations: per-tile uint8 (offset 128), scale sa.
 * Weights: per-oc int8, scale sw[oc].
 * im2col matrix A[j][(ic,tap)] (uint8, +128 offset), K = in_ch*k.
 * dot = (Σ A·Wq - 128*ΣWq) * sa * sw[oc].
 */
static void conv_int8(const float *in, int in_ch, int n, const float *w,
                      int out_ch, int k, int pad, int dil, float *out) {
    int n_out = n - dil * (k - 1) + 2 * pad - 1 + 1;
    memset(out, 0, (size_t)out_ch * n_out * sizeof(float));
    int K = in_ch * k;
    const int TILE = 2048;
    /* quantize weights once: per-oc int8 + ΣWq */
    char *wq = (char *)malloc((size_t)out_ch * K);
    int *sumWq = (int *)malloc((size_t)out_ch * sizeof(int));
    float *sw = (float *)malloc((size_t)out_ch * sizeof(float));
    for (int oc = 0; oc < out_ch; oc++) {
        float wmax = 0.0f;
        for (int i = 0; i < K; i++) { float v = fabsf(w[(size_t)oc * K + i]); if (v > wmax) wmax = v; }
        if (wmax < 1e-9f) wmax = 1e-9f;
        sw[oc] = wmax / 127.0f;
        int s = 0;
        for (int i = 0; i < K; i++) {
            int q = (int)lrintf(w[(size_t)oc * K + i] / sw[oc]);
            if (q < -128) q = -128; if (q > 127) q = 127;
            wq[(size_t)oc * K + i] = (char)q;
            s += q;
        }
        sumWq[oc] = s;
    }
#pragma omp parallel for schedule(dynamic, 4) num_threads(8) if(n_out >= TILE)
    for (int jb = 0; jb < n_out; jb += TILE) {
        int j_hi = jb + TILE < n_out ? jb + TILE : n_out;
        int j_lo = jb > pad ? jb : pad;   /* max(jb,pad): first j with all taps valid */
        int j_in_hi = n - (k - 1) * dil + pad;
        if (j_in_hi > j_hi) j_in_hi = j_hi;
        /* im2col tile: A[(j-jb)*K + (ic*k+tap)] = in[ic][j+tap*dil-pad] (+128) */
        int jn = j_hi - jb;
        unsigned char *A = (unsigned char *)malloc((size_t)jn * K);
        /* per-tile activation scale */
        float amax = 0.0f;
        for (int j = j_lo; j < j_in_hi; j++)
            for (int ic = 0; ic < in_ch; ic++) {
                float v = fabsf(in[(size_t)ic * n + j]);
                if (v > amax) amax = v;
            }
        float sa = amax < 1e-9f ? 1e-9f : amax / 127.0f;
        for (int ic = 0; ic < in_ch; ic++) {
            const float *row = in + (size_t)ic * n;
            for (int tap = 0; tap < k; tap++) {
                int col = ic * k + tap;
                int off = tap * dil - pad;
                for (int j = j_lo; j < j_in_hi; j++) {
                    int src = j + off;
                    if (src < 0 || src >= n) continue;
                    int q = (int)lrintf(row[src] / sa) + 128;
                    if (q < 0) q = 0; if (q > 255) q = 255;
                    A[(size_t)(j - jb) * K + col] = (unsigned char)q;
                }
            }
        }
        /* GEMM: out[oc][j] = Σ_A A[j][i]*wq[oc][i] ; OC_BLK=8, j scalar-loop */
        enum { OC_BLK = 8 };
        for (int oc0 = 0; oc0 < out_ch; oc0 += OC_BLK) {
            int n_oc = OC_BLK;
            if (oc0 + n_oc > out_ch) n_oc = out_ch - oc0;
            float *orow0[OC_BLK];
            const char *wq0[OC_BLK];
            float fac[OC_BLK], corr[OC_BLK];
            for (int oc = 0; oc < n_oc; oc++) {
                orow0[oc] = out + (size_t)(oc0 + oc) * n_out;
                wq0[oc] = wq + (size_t)(oc0 + oc) * K;
                fac[oc] = sa * sw[oc0 + oc];
                corr[oc] = 128.0f * (float)sumWq[oc0 + oc] * sa * sw[oc0 + oc];
            }
            for (int j = j_lo; j < j_in_hi; j++) {
                const unsigned char *arow = A + (size_t)(j - jb) * K;
                int i = 0;
                __m256i s0[OC_BLK], s1[OC_BLK];
                for (int oc = 0; oc < n_oc; oc++) { s0[oc] = _mm256_setzero_si256(); s1[oc] = _mm256_setzero_si256(); }
                const __m256i ones16 = _mm256_set1_epi16(1);
                for (; i + 32 <= K; i += 32) {
                    __m256i av = _mm256_loadu_si256((const __m256i *)(arow + i));
                    for (int oc = 0; oc < n_oc; oc++) {
                        __m256i wv = _mm256_loadu_si256((const __m256i *)(wq0[oc] + i));
                        __m256i p = _mm256_maddubs_epi16(av, wv);      /* 16 int16 */
                        s0[oc] = _mm256_add_epi32(s0[oc], _mm256_madd_epi16(p, ones16));
                    }
                }
                for (; i < K; i++)
                    for (int oc = 0; oc < n_oc; oc++)
                        s0[oc] = _mm256_add_epi32(s0[oc], _mm256_set1_epi32((int)arow[i] * (int)wq0[oc][i]));
                for (int oc = 0; oc < n_oc; oc++) {
                    int v[8]; _mm256_storeu_si256((__m256i *)v, s0[oc]);
                    long acc = 0; for (int z = 0; z < 8; z++) acc += v[z];
                    orow0[oc][j] = (float)acc * fac[oc] - corr[oc];
                }
            }
            /* boundary rows outside j_lo..j_in_hi: FP32 fallback */
            for (int oc = 0; oc < n_oc; oc++) {
                const float *wv = w + (size_t)(oc0 + oc) * K;
                for (int j = jb; j < j_lo && j < j_hi; j++) {
                    float acc = 0.0f;
                    for (int ic = 0; ic < in_ch; ic++) {
                        const float *ir2 = in + (size_t)ic * n;
                        const float *wv2 = wv + (size_t)ic * k;
                        for (int tap = 0; tap < k; tap++) {
                            int src = j + tap * dil - pad;
                            if (src >= 0 && src < n) acc += ir2[src] * wv2[tap];
                        }
                    }
                    orow0[oc][j] = acc;
                }
            }
        }
        free(A);
    }
    free(wq); free(sumWq); free(sw);
}

int main(int argc, char **argv) {
    int in_ch = 192, out_ch = 192, k = 3, n = 8192, dil = 1, reps = 20;
    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) k = atoi(argv[2]);
    if (argc > 3) reps = atoi(argv[3]);
    int pad = k / 2;
    float *in = (float *)malloc((size_t)in_ch * n * sizeof(float));
    float *w  = (float *)malloc((size_t)out_ch * in_ch * k * sizeof(float));
    int n_out = n - dil * (k - 1) + 2 * pad - 1 + 1;
    float *o1 = (float *)malloc((size_t)out_ch * n_out * sizeof(float));
    float *o2 = (float *)malloc((size_t)out_ch * n_out * sizeof(float));
    srand(42);
    for (int i = 0; i < in_ch * n; i++) in[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    for (int i = 0; i < out_ch * in_ch * k; i++) w[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.2f;
    conv_fp32(in, in_ch, n, w, out_ch, k, pad, dil, o1);
    conv_int8(in, in_ch, n, w, out_ch, k, pad, dil, o2);
    double se = 0.0, mx = 0.0;
    for (int i = 0; i < out_ch * n_out; i++) {
        double d = fabs((double)o1[i] - o2[i]);
        se += d * d; if (d > mx) mx = d;
    }
    double rmse = sqrt(se / (out_ch * n_out));
    double rms1 = 0.0; for (int i = 0; i < out_ch * n_out; i++) rms1 += (double)o1[i] * o1[i];
    rms1 = sqrt(rms1 / (out_ch * n_out));
    printf("shape in=%d out=%d k=%d n=%d n_out=%d\n", in_ch, out_ch, k, n, n_out);
    printf("INT8 vs FP32: rmse=%.6f maxdiff=%.6f (rms=%.4f) rel=%.3f%%\n",
           rmse, mx, rms1, 100.0 * rmse / (rms1 + 1e-9));
    double t1 = omp_get_wtime();
    for (int r = 0; r < reps; r++) conv_fp32(in, in_ch, n, w, out_ch, k, pad, dil, o1);
    double t2 = omp_get_wtime();
    for (int r = 0; r < reps; r++) conv_int8(in, in_ch, n, w, out_ch, k, pad, dil, o2);
    double t3 = omp_get_wtime();
    double macs = (double)out_ch * in_ch * k * n_out * reps;
    printf("FP32: %.3fs  %.1f GMAC/s\n", t2 - t1, macs / 1e9 / (t2 - t1));
    printf("INT8: %.3fs  %.1f GMAC/s\n", t3 - t2, macs / 1e9 / (t3 - t2));
    printf("SPEEDUP: %.2fx\n", (t2 - t1) / (t3 - t2));
    free(in); free(w); free(o1); free(o2);
    return 0;
}
