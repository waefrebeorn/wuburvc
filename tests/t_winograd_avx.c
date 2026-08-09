/* t_winograd_avx.c — honest AVX2 comparison: Winograd F(2,3) vs the engine's
 * OC-blocked AVX2 FMA conv (the real kernel), same shape. If Winograd doesn't
 * beat the real kernel on Zen2, don't wire it.
 * Build: gcc -O3 -mavx2 -mfma -march=znver2 -fopenmp t_winograd_avx.c -o t_winograd_avx.exe
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <immintrin.h>

/* THE ENGINE KERNEL v2 (loop-interchanged: jr outer, oc0 inner so the
 * 24KB input block for a 32-sample jr stays L1-hot across all oc-blocks;
 * original re-reads the whole tile from L3 48x) */
static void conv_avx(const float *in, int in_ch, int n, const float *w,
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
        int n_oc = OC_BLK;
        float *orow0[OC_BLK]; const float *wv0[OC_BLK];
        for (int oc = 0; oc < OC_BLK; oc++) { orow0[oc] = NULL; wv0[oc] = NULL; }
        /* interior: jr OUTER (input block L1-hot), oc0 inner */
        for (int jr = j_in_lo; jr <= jr_max; jr += 32) {
            for (int oc0 = 0; oc0 < out_ch; oc0 += OC_BLK) {
                int n_ocb = OC_BLK;
                if (oc0 + n_ocb > out_ch) n_ocb = out_ch - oc0;
                float *orowb[OC_BLK]; const float *wvb[OC_BLK];
                for (int oc = 0; oc < n_ocb; oc++) {
                    orowb[oc] = out + (size_t)(oc0 + oc) * n_out;
                    wvb[oc] = w + (size_t)(oc0 + oc) * in_ch * k;
                }
                __m256 a0[OC_BLK], a1[OC_BLK], a2[OC_BLK], a3[OC_BLK];
                for (int oc = 0; oc < n_ocb; oc++) {
                    a0[oc] = _mm256_loadu_ps(orowb[oc] + jr);
                    a1[oc] = _mm256_loadu_ps(orowb[oc] + jr + 8);
                    a2[oc] = _mm256_loadu_ps(orowb[oc] + jr + 16);
                    a3[oc] = _mm256_loadu_ps(orowb[oc] + jr + 24);
                }
                for (int ick = 0; ick < in_ch; ick++) {
                    const float *ir2 = in + (size_t)ick * n + jr - pad;
                    for (int tap = 0; tap < k; tap++) {
                        __m256 iv0 = _mm256_loadu_ps(ir2 + tap * dil);
                        __m256 iv1 = _mm256_loadu_ps(ir2 + 8 + tap * dil);
                        __m256 iv2 = _mm256_loadu_ps(ir2 + 16 + tap * dil);
                        __m256 iv3 = _mm256_loadu_ps(ir2 + 24 + tap * dil);
                        for (int oc = 0; oc < n_ocb; oc++) {
                            float wt = wvb[oc][(size_t)ick * k + tap];
                            __m256 wv8 = _mm256_set1_ps(wt);
                            a0[oc] = _mm256_fmadd_ps(iv0, wv8, a0[oc]);
                            a1[oc] = _mm256_fmadd_ps(iv1, wv8, a1[oc]);
                            a2[oc] = _mm256_fmadd_ps(iv2, wv8, a2[oc]);
                            a3[oc] = _mm256_fmadd_ps(iv3, wv8, a3[oc]);
                        }
                    }
                }
                for (int oc = 0; oc < n_ocb; oc++) {
                    _mm256_storeu_ps(orowb[oc] + jr, a0[oc]);
                    _mm256_storeu_ps(orowb[oc] + jr + 8, a1[oc]);
                    _mm256_storeu_ps(orowb[oc] + jr + 16, a2[oc]);
                    _mm256_storeu_ps(orowb[oc] + jr + 24, a3[oc]);
                }
            }
        }
        /* boundaries: FP32 full-sum (cheap, few samples) */
        for (int oc0 = 0; oc0 < out_ch; oc0 += OC_BLK) {
            int n_ocb = OC_BLK;
            if (oc0 + n_ocb > out_ch) n_ocb = out_ch - oc0;
            for (int j = jb; j < j_in_lo && j < j_hi; j++)
                for (int oc = 0; oc < n_ocb; oc++) {
                    const float *wv = w + (size_t)(oc0 + oc) * in_ch * k;
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
            for (int j = j_in_hi; j < j_hi; j++)
                for (int oc = 0; oc < n_ocb; oc++) {
                    const float *wv = w + (size_t)(oc0 + oc) * in_ch * k;
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
    }
}

/* AVX2 Winograd F(2,3): 2 outputs from 4 inputs, 4 mults per (oc,ic) pair.
 * Process 8 output-position pairs (16 outputs) per iteration with 8-lane
 * vectors: for each pair p, the 4 inputs d0..d3 are 8-lane vectors (one per
 * pair), weights are scalars (same for all pairs). */
static void conv_wino3_avx(const float *in, int in_ch, int n, const float *w,
                           int out_ch, int k, int pad, int dil, float *out) {
    int n_out = n - dil * (k - 1) + 2 * pad - 1 + 1;
    memset(out, 0, (size_t)out_ch * n_out * sizeof(float));
    if (!(k == 3 && dil == 1 && pad == 1)) { conv_avx(in, in_ch, n, w, out_ch, k, pad, dil, out); return; }
    const int TILE = 2048;
#pragma omp parallel for schedule(dynamic, 4) num_threads(8) if(n_out >= TILE)
    for (int jb = 0; jb < n_out; jb += TILE) {
        int j_hi = jb + TILE < n_out ? jb + TILE : n_out;
        int j_lo = jb > pad ? jb : pad;
        int j_in_hi = n - (k - 1) * dil + pad;
        if (j_in_hi > j_hi) j_in_hi = j_hi;
        for (int oc = 0; oc < out_ch; oc++) {
            const float *wv = w + (size_t)oc * in_ch * k;
            float *orow = out + (size_t)oc * n_out;
            /* accumulate 16 outputs at a time (8 pairs) */
            int j = j_lo;
            for (; j + 15 < j_in_hi; j += 16) {
                __m256 acc0 = _mm256_setzero_ps(), acc1 = _mm256_setzero_ps();
                for (int ic = 0; ic < in_ch; ic++) {
                    const float *ir2 = in + (size_t)ic * n;
                    /* d0 = in[j-1 + 0..7], d1 = in[j+0..7], d2 = in[j+1..8], d3 = in[j+2..9]
                     * for pairs p=0..7 (outputs 2p, 2p+1) */
                    __m256 d0 = _mm256_loadu_ps(ir2 + j - 1);
                    __m256 d1 = _mm256_loadu_ps(ir2 + j);
                    __m256 d2 = _mm256_loadu_ps(ir2 + j + 1);
                    __m256 d3 = _mm256_loadu_ps(ir2 + j + 2);
                    float g0 = wv[(size_t)ic * 3 + 0];
                    float g1 = wv[(size_t)ic * 3 + 1];
                    float g2 = wv[(size_t)ic * 3 + 2];
                    /* precompute weight transforms (scalars) */
                    float G0 = g0;
                    float G1 = (g0 + g1 + g2) * 0.5f;
                    float G2 = (g0 - g1 + g2) * 0.5f;
                    float G3 = g2;
                    /* B^T d (4 values per pair) */
                    __m256 t0 = _mm256_sub_ps(d0, d2);   /* d0 - d2 */
                    __m256 t1 = _mm256_add_ps(d1, d2);   /* d1 + d2 */
                    __m256 t2 = _mm256_sub_ps(d2, d1);   /* d2 - d1 */
                    __m256 t3 = _mm256_sub_ps(d1, d3);   /* d1 - d3 */
                    /* m1 = t0*G0 ; m2 = t1*G1 ; m3 = t2*G2 ; m4 = t3*G3 */
                    __m256 m1 = _mm256_mul_ps(t0, _mm256_set1_ps(G0));
                    __m256 m2 = _mm256_mul_ps(t1, _mm256_set1_ps(G1));
                    __m256 m3 = _mm256_mul_ps(t2, _mm256_set1_ps(G2));
                    __m256 m4 = _mm256_mul_ps(t3, _mm256_set1_ps(G3));
                    /* y0 = m1+m2+m3 ; y1 = m2-m3-m4 */
                    __m256 y0 = _mm256_add_ps(_mm256_add_ps(m1, m2), m3);
                    __m256 y1 = _mm256_sub_ps(_mm256_sub_ps(m2, m3), m4);
                    acc0 = _mm256_add_ps(acc0, y0);
                    acc1 = _mm256_add_ps(acc1, y1);
                }
                /* scatter acc0 → orow[j+0..7], acc1 → orow[j+8..15] */
                _mm256_storeu_ps(orow + j, acc0);
                _mm256_storeu_ps(orow + j + 8, acc1);
            }
            for (; j < j_in_hi; j++) {
                float acc = 0.0f;
                for (int ic = 0; ic < in_ch; ic++) {
                    const float *ir2 = in + (size_t)ic * n;
                    const float *wv2 = wv + (size_t)ic * 3;
                    for (int tap = 0; tap < 3; tap++) {
                        int src = j + tap - pad;
                        if (src >= 0 && src < n) acc += ir2[src] * wv2[tap];
                    }
                }
                orow[j] = acc;
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
    /* o1 = original (oc0-outer) kernel — compile-time selected below */
    conv_wino3_avx(in, in_ch, n, w, out_ch, k, pad, dil, o1);   /* k!=3 → conv_avx_orig fallback? no: conv_wino3_avx calls conv_avx for k!=3, so o1 = interchanged! */
    /* We need BOTH: interchanged (conv_avx) and original. Recreate original
     * via the k!=3 path is NOT enough — copy original conv as conv_orig. */
    (void)o1; (void)o2;
    printf("shape in=%d out=%d k=%d n=%d n_out=%d\n", in_ch, out_ch, k, n, n_out);
    printf("NOTE: this build only contains the INTERCHANGED kernel; timing below is self-relative.\n");
    double t1 = omp_get_wtime();
    for (int r = 0; r < reps; r++) conv_avx(in, in_ch, n, w, out_ch, k, pad, dil, o2);
    double t2 = omp_get_wtime();
    double macs = (double)out_ch * in_ch * k * n_out * reps;
    printf("interchanged: %.3fs  %.1f GMAC/s\n", t2 - t1, macs / 1e9 / (t2 - t1));
    free(in); free(w); free(o1); free(o2);
    return 0;
}
