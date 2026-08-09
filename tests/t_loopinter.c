/* t_loopinter.c — A/B: original oc0-outer AVX conv vs jr-outer (interchanged)
 * on Zen2. Byte-identity check + timing. Same boundary code both sides.
 * Build: gcc -O3 -mavx2 -mfma -march=znver2 -fopenmp t_loopinter.c -o t_loopinter.exe
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <immintrin.h>

/* ORIGINAL: oc0 outer, jr inner */
static void conv_orig(const float *in, int in_ch, int n, const float *w,
                      int out_ch, int k, int pad, int dil, float *out) {
    int n_out = n - dil * (k - 1) + 2 * pad - 1 + 1;
    memset(out, 0, (size_t)out_ch * n_out * sizeof(float));
    const int TILE = 2048;
#pragma omp parallel for schedule(dynamic, 4) num_threads(8) if(n_out >= TILE)
    for (int jb = 0; jb < n_out; jb += TILE) {
        int j_hi = jb + TILE < n_out ? jb + TILE : n_out;
        enum { OC_BLK = 4 };
        int j_in_lo = jb > pad ? jb : pad;
        int j_in_hi = n - (k - 1) * dil + pad;
        if (j_in_hi > j_hi) j_in_hi = j_hi;
        int jr_max = n - 32 - (k - 1) * dil + pad;
        if (jr_max > j_in_hi - 32) jr_max = j_in_hi - 32;
        for (int oc0 = 0; oc0 < out_ch; oc0 += OC_BLK) {
            int n_oc = OC_BLK;
            if (oc0 + n_oc > out_ch) n_oc = out_ch - oc0;
            float *orow0[OC_BLK]; const float *wv0[OC_BLK];
            for (int oc = 0; oc < n_oc; oc++) {
                orow0[oc] = out + (size_t)(oc0 + oc) * n_out;
                wv0[oc] = w + (size_t)(oc0 + oc) * in_ch * k;
            }
            for (int j = jb; j < j_in_lo && j < j_hi; j++)
                for (int oc = 0; oc < n_oc; oc++) {
                    const float *wv = wv0[oc];
                    float acc = 0.0f;
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
                    float acc = 0.0f;
                    for (int ick = 0; ick < in_ch; ick++) {
                        const float *ir2 = in + (size_t)ick * n;
                        const float *wv2 = wv + (size_t)ick * k;
                        for (int tap = 0; tap < k; tap++) {
                            int src = jr + tap * dil - pad;
                            if (src >= 0 && src < n) acc += ir2[src] * wv2[tap];
                        }
                    }
                    orow0[oc][jr] = acc;
                }
        }
    }
}

/* INTERCHANGED: jr outer, oc0 inner — 24KB input block stays L1-hot */
static void conv_inter(const float *in, int in_ch, int n, const float *w,
                       int out_ch, int k, int pad, int dil, float *out) {
    int n_out = n - dil * (k - 1) + 2 * pad - 1 + 1;
    memset(out, 0, (size_t)out_ch * n_out * sizeof(float));
    const int TILE = 2048;
#pragma omp parallel for schedule(dynamic, 4) num_threads(8) if(n_out >= TILE)
    for (int jb = 0; jb < n_out; jb += TILE) {
        int j_hi = jb + TILE < n_out ? jb + TILE : n_out;
        enum { OC_BLK = 4 };
        int j_in_lo = jb > pad ? jb : pad;
        int j_in_hi = n - (k - 1) * dil + pad;
        if (j_in_hi > j_hi) j_in_hi = j_hi;
        int jr_max = n - 32 - (k - 1) * dil + pad;
        if (jr_max > j_in_hi - 32) jr_max = j_in_hi - 32;
        /* boundaries identical to original (oc0 outer over the edge rows) */
        for (int oc0 = 0; oc0 < out_ch; oc0 += OC_BLK) {
            int n_oc = OC_BLK;
            if (oc0 + n_oc > out_ch) n_oc = out_ch - oc0;
            const float *wv0[OC_BLK];
            for (int oc = 0; oc < n_oc; oc++) wv0[oc] = w + (size_t)(oc0 + oc) * in_ch * k;
            for (int j = jb; j < j_in_lo && j < j_hi; j++)
                for (int oc = 0; oc < n_oc; oc++) {
                    const float *wv = wv0[oc];
                    float acc = 0.0f;
                    for (int ick = 0; ick < in_ch; ick++) {
                        const float *ir2 = in + (size_t)ick * n;
                        const float *wv2 = wv + (size_t)ick * k;
                        for (int tap = 0; tap < k; tap++) {
                            int src = j + tap * dil - pad;
                            if (src >= 0 && src < n) acc += ir2[src] * wv2[tap];
                        }
                    }
                    out[(size_t)(oc0 + oc) * n_out + j] = acc;
                }
        }
        /* interior: jr outer, oc0 inner */
        int jr = j_in_lo;
        for (; jr <= jr_max; jr += 32) {
            for (int oc0 = 0; oc0 < out_ch; oc0 += OC_BLK) {
                int n_oc = OC_BLK;
                if (oc0 + n_oc > out_ch) n_oc = out_ch - oc0;
                float *orow0[OC_BLK]; const float *wv0[OC_BLK];
                for (int oc = 0; oc < n_oc; oc++) {
                    orow0[oc] = out + (size_t)(oc0 + oc) * n_out;
                    wv0[oc] = w + (size_t)(oc0 + oc) * in_ch * k;
                }
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
        }
        /* scalar tail: rows (jr after register loop) .. j_in_hi — jr-outer,
         * oc0-inner (same as the interior, so ALL oc blocks get their rows) */
        for (; jr < j_in_hi; jr++) {
            for (int oc0 = 0; oc0 < out_ch; oc0 += OC_BLK) {
                int n_oc = OC_BLK;
                if (oc0 + n_oc > out_ch) n_oc = out_ch - oc0;
                for (int oc = 0; oc < n_oc; oc++) {
                    const float *wv = w + (size_t)(oc0 + oc) * in_ch * k;
                    float acc = 0.0f;
                    for (int ick = 0; ick < in_ch; ick++) {
                        const float *ir2 = in + (size_t)ick * n;
                        const float *wv2 = wv + (size_t)ick * k;
                        for (int tap = 0; tap < k; tap++) {
                            int src = jr + tap * dil - pad;
                            if (src >= 0 && src < n) acc += ir2[src] * wv2[tap];
                        }
                    }
                    out[(size_t)(oc0 + oc) * n_out + jr] = acc;
                }
            }
        }
    }
}

int main(int argc, char **argv) {
    int in_ch = 192, out_ch = 192, k = 3, n = 8192, dil = 1, reps = 30;
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
    conv_orig(in, in_ch, n, w, out_ch, k, pad, dil, o1);
    conv_inter(in, in_ch, n, w, out_ch, k, pad, dil, o2);
    double mx = 0.0; int nbad = 0;
    for (int i = 0; i < out_ch * n_out; i++) {
        double d = fabs((double)o1[i] - o2[i]);
        if (d > mx) mx = d;
        if (d > 1e-6) nbad++;
    }
    printf("shape in=%d out=%d k=%d n=%d n_out=%d\n", in_ch, out_ch, k, n, n_out);
    printf("inter vs orig: maxdiff=%.9f nbad=%d\n", mx, nbad);
    double t1 = omp_get_wtime();
    for (int r = 0; r < reps; r++) conv_orig(in, in_ch, n, w, out_ch, k, pad, dil, o1);
    double t2 = omp_get_wtime();
    for (int r = 0; r < reps; r++) conv_inter(in, in_ch, n, w, out_ch, k, pad, dil, o2);
    double t3 = omp_get_wtime();
    double macs = (double)out_ch * in_ch * k * n_out * reps;
    printf("orig (oc0-outer): %.3fs  %.1f GMAC/s\n", t2 - t1, macs / 1e9 / (t2 - t1));
    printf("inter (jr-outer): %.3fs  %.1f GMAC/s\n", t3 - t2, macs / 1e9 / (t3 - t2));
    printf("SPEEDUP: %.2fx\n", (t2 - t1) / (t3 - t2));
    free(in); free(w); free(o1); free(o2);
    return 0;
}
