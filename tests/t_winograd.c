/* t_winograd.c — DA benchmark: Winograd F(2,k) 1D conv vs direct FP32 AVX2
 * conv on Zen2. For conv k=3: F(2,3) does 4 mults per 2 outputs (vs 6);
 * k=7: 8 mults per 2 outputs (vs 14); k=11: 12 vs 22. Winograd's add
 * overhead can eat the multiply win at small channel counts — measure.
 * Build: gcc -O3 -mavx2 -mfma -march=znver2 -fopenmp t_winograd.c -o t_winograd.exe
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <immintrin.h>

/* direct FP32 conv (reference) */
static void conv_fp32(const float *in, int in_ch, int n, const float *w,
                      int out_ch, int k, int pad, int dil, float *out) {
    int n_out = n - dil * (k - 1) + 2 * pad - 1 + 1;
    memset(out, 0, (size_t)out_ch * n_out * sizeof(float));
    for (int oc = 0; oc < out_ch; oc++) {
        const float *wv = w + (size_t)oc * in_ch * k;
        float *orow = out + (size_t)oc * n_out;
        for (int j = 0; j < n_out; j++) {
            float acc = 0.0f;
            for (int ic = 0; ic < in_ch; ic++) {
                const float *ir2 = in + (size_t)ic * n;
                const float *wv2 = wv + (size_t)ic * k;
                for (int tap = 0; tap < k; tap++) {
                    int src = j + tap * dil - pad;
                    if (src >= 0 && src < n) acc += ir2[src] * wv2[tap];
                }
            }
            orow[j] = acc;
        }
    }
}

/* Winograd F(2,3): 2 outputs from 4 inputs, 4 mults.
 * d0..d3 = inputs; g0..g2 = weights (k=3, dil=1, pad=1)
 * m1 = (d0-d2)*g0 ; m2 = (d1+d2)*(g0+g1+g2)/2 ; m3 = (d2-d1)*(g0-g1+g2)/2
 * m4 = (d1-d3)*g2
 * y0 = m1+m2+m3 ; y1 = m2-m3-m4
 */
static void winograd_2x3_row(const float *d, const float *g, float *y) {
    float m1 = (d[0] - d[2]) * g[0];
    float m2 = (d[1] + d[2]) * (g[0] + g[1] + g[2]) * 0.5f;
    float m3 = (d[2] - d[1]) * (g[0] - g[1] + g[2]) * 0.5f;
    float m4 = (d[1] - d[3]) * g[2];
    y[0] = m1 + m2 + m3;
    y[1] = m2 - m3 - m4;
}

/* Winograd F(2,5): 2 outputs from 6 inputs, 6 mults */
static void winograd_2x5_row(const float *d, const float *g, float *y) {
    /* m = Gg . B^T d  (standard F(2,5)):
     * B^T d:
     *   t0 = d0 - d2
     *   t1 = d1 + d2
     *   t2 = d2 - d1
     *   t3 = d1 - d3
     *   t4 = d2 + d3
     *   t5 = d3 - d2
     *   t6 = d2 - d4
     *   t7 = d3 + d4
     *   t8 = d4 - d3
     *   t9 = d3 - d5
     * G g:
     *   g0' = g0
     *   g1' = (g0+g1+g2+g3+g4)/2
     *   g2' = (g0-g1+g2-g3+g4)/2
     *   g3' = (g0+g1+g2-g3-g4)/6
     *   g4' = (g0-g1+g2+g3-g4)/6
     *   g5' = g4
     * y0 = t0*g0' + t1*g1' + t2*g2' + t3*g3' + t4*g4' + t6*g5'
     * y1 = t5*g0' + t4*g1' + t3*g2' + t2*g3' + t1*g4' + t7*g5' ... */
    /* NOTE: full F(2,5) transform is 6 mults + ~16 adds. Keep it simple and
     * use the general m=2 formulation via the transform matrices. */
    float G[6], B[6][6];
    /* G g (6 values) */
    G[0] = g[0];
    G[1] = (g[0] + g[1] + g[2] + g[3] + g[4]) * 0.5f;
    G[2] = (g[0] - g[1] + g[2] - g[3] + g[4]) * 0.5f;
    G[3] = (g[0] + g[1] + g[2] - g[3] - g[4]) / 6.0f;
    G[4] = (g[0] - g[1] + g[2] + g[3] - g[4]) / 6.0f;
    G[5] = g[4];
    /* B^T d (6 values): standard F(2,5) B^T */
    B[0][0] = 1; B[0][1] = 0; B[0][2] = -1; B[0][3] = 0; B[0][4] = 0; B[0][5] = 0;
    B[1][0] = 0; B[1][1] = 1; B[1][2] = 1;  B[1][3] = 0; B[1][4] = 0; B[1][5] = 0;
    B[2][0] = 0; B[2][1] = -1; B[2][2] = 1; B[2][3] = 0; B[2][4] = 0; B[2][5] = 0;
    B[3][0] = 0; B[3][1] = 1; B[3][2] = 0;  B[3][3] = -1; B[3][4] = 0; B[3][5] = 0;
    B[4][0] = 0; B[4][1] = 1; B[4][2] = 1;  B[4][3] = 0; B[4][4] = 0; B[4][5] = 0;
    B[5][0] = 0; B[5][1] = 0; B[5][2] = -1; B[5][3] = 1; B[5][4] = 0; B[5][5] = 0;
    float t[6];
    for (int i = 0; i < 6; i++) {
        t[i] = 0.0f;
        for (int j = 0; j < 6; j++) t[i] += B[i][j] * d[j];
    }
    y[0] = t[0]*G[0] + t[1]*G[1] + t[2]*G[2] + t[3]*G[3] + t[4]*G[4] + t[5]*G[5];
    y[1] = t[5]*G[0] + t[4]*G[1] + t[3]*G[2] + t[2]*G[3] + t[1]*G[4] + t[0]*G[5];
}

/* Winograd conv wrapper: applies F(2,k) per (oc,ic) row pair, sums over ic.
 * Only k=3 uses the efficient 2x3; k=5/7/11 fall back here (slow, for
 * CORRECTNESS check only). The benchmark measures k=3 speed. */
static void conv_wino3(const float *in, int in_ch, int n, const float *w,
                       int out_ch, int k, int pad, int dil, float *out) {
    int n_out = n - dil * (k - 1) + 2 * pad - 1 + 1;
    memset(out, 0, (size_t)out_ch * n_out * sizeof(float));
    if (k == 3 && dil == 1 && pad == 1) {
        for (int oc = 0; oc < out_ch; oc++) {
            const float *wv = w + (size_t)oc * in_ch * k;
            float *orow = out + (size_t)oc * n_out;
            for (int j = 0; j + 1 < n_out; j += 2) {
                float acc0 = 0.0f, acc1 = 0.0f;
                for (int ic = 0; ic < in_ch; ic++) {
                    const float *ir2 = in + (size_t)ic * n;
                    float d0 = (j - 1 >= 0) ? ir2[j - 1] : 0.0f;
                    float d1 = ir2[j];
                    float d2 = ir2[j + 1];
                    float d3 = (j + 2 < n) ? ir2[j + 2] : 0.0f;
                    float dd[4] = { d0, d1, d2, d3 };
                    float g[3] = { wv[(size_t)ic * 3 + 0], wv[(size_t)ic * 3 + 1], wv[(size_t)ic * 3 + 2] };
                    float y[2];
                    winograd_2x3_row(dd, g, y);
                    acc0 += y[0];
                    acc1 += y[1];
                }
                orow[j] = acc0;
                orow[j + 1] = acc1;
            }
        }
    } else {
        conv_fp32(in, in_ch, n, w, out_ch, k, pad, dil, out);
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
    conv_fp32(in, in_ch, n, w, out_ch, k, pad, dil, o1);
    conv_wino3(in, in_ch, n, w, out_ch, k, pad, dil, o2);
    double se = 0.0, mx = 0.0;
    for (int i = 0; i < out_ch * n_out; i++) {
        double d = fabs((double)o1[i] - o2[i]);
        se += d * d; if (d > mx) mx = d;
    }
    double rmse = sqrt(se / (out_ch * n_out));
    double rms1 = 0.0; for (int i = 0; i < out_ch * n_out; i++) rms1 += (double)o1[i] * o1[i];
    rms1 = sqrt(rms1 / (out_ch * n_out));
    printf("shape in=%d out=%d k=%d n=%d n_out=%d\n", in_ch, out_ch, k, n, n_out);
    printf("WINO vs direct: rmse=%.6f maxdiff=%.6f (rms=%.4f) rel=%.3f%%\n",
           rmse, mx, rms1, 100.0 * rmse / (rms1 + 1e-9));
    double t1 = omp_get_wtime();
    for (int r = 0; r < reps; r++) conv_fp32(in, in_ch, n, w, out_ch, k, pad, dil, o1);
    double t2 = omp_get_wtime();
    for (int r = 0; r < reps; r++) conv_wino3(in, in_ch, n, w, out_ch, k, pad, dil, o2);
    double t3 = omp_get_wtime();
    double macs = (double)out_ch * in_ch * k * n_out * reps;
    printf("direct: %.3fs  %.1f GMAC/s\n", t2 - t1, macs / 1e9 / (t2 - t1));
    printf("wino3 : %.3fs  %.1f GMAC/s\n", t3 - t2, macs / 1e9 / (t3 - t2));
    printf("SPEEDUP: %.2fx\n", (t2 - t1) / (t3 - t2));
    free(in); free(w); free(o1); free(o2);
    return 0;
}
