/* t_int8_peak.c — DA microbench: maddubs INT8 dot throughput vs FMA dot
 * throughput on this exact Zen2 (Ryzen 5 3600). Answers: does the int8 path
 * have any compute headroom at all before we build an NHWC conv kernel?
 * Build: gcc -O3 -mavx2 -mfma -march=znver2 t_int8_peak.c -o t_int8_peak.exe
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <immintrin.h>

/* FMA dot: 8 lanes, 1 FMA per 8 MACs — 2 FMA pipes → 16 FLOP/cycle */
static float dot_fma(const float *a, const float *b, int n, int iters) {
    float r = 0.0f;
    __m256 acc = _mm256_setzero_ps();
    for (int it = 0; it < iters; it++) {
        __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
        for (int i = 0; i < n; i += 16) {
            __m256 a0 = _mm256_loadu_ps(a + i), b0 = _mm256_loadu_ps(b + i);
            __m256 a1 = _mm256_loadu_ps(a + i + 8), b1 = _mm256_loadu_ps(b + i + 8);
            s0 = _mm256_fmadd_ps(a0, b0, s0);
            s1 = _mm256_fmadd_ps(a1, b1, s1);
        }
        acc = _mm256_add_ps(acc, s0);
        acc = _mm256_add_ps(acc, s1);
    }
    float v[8]; _mm256_storeu_ps(v, acc);
    for (int i = 0; i < 8; i++) r += v[i];
    return r;
}

/* maddubs INT8 dot: 16 lanes of 8-bit pairs → 8 int16 per instr.
 * Accumulate int32. 16 MACs per maddubs instruction. */
static long dot_int8(const unsigned char *a, const char *b, int n, int iters) {
    long r = 0;
    __m256i acc = _mm256_setzero_si256();
    const __m256i ones16 = _mm256_set1_epi16(1);
    for (int it = 0; it < iters; it++) {
        __m256i s0 = _mm256_setzero_si256(), s1 = _mm256_setzero_si256();
        for (int i = 0; i < n; i += 32) {
            __m256i a0 = _mm256_loadu_si256((const __m256i *)(a + i));
            __m256i b0 = _mm256_loadu_si256((const __m256i *)(b + i));
            __m256i p0 = _mm256_maddubs_epi16(a0, b0);   /* 16 int16 lanes */
            __m256i p1 = _mm256_madd_epi16(p0, ones16);  /* 8 int32 lanes */
            s0 = _mm256_add_epi32(s0, p1);
            s1 = _mm256_add_epi32(s1, p1);
        }
        acc = _mm256_add_epi32(acc, s0);
        acc = _mm256_add_epi32(acc, s1);
    }
    int v[8]; _mm256_storeu_si256((__m256i *)v, acc);
    for (int i = 0; i < 8; i++) r += v[i];
    return r;
}

/* maddubs with SAME-lane-accumulate only (no widen, int16 acc — Q8_0 style) */
static long dot_int8_16(const unsigned char *a, const char *b, int n, int iters) {
    long r = 0;
    __m256i acc = _mm256_setzero_si256();
    for (int it = 0; it < iters; it++) {
        __m256i s0 = _mm256_setzero_si256(), s1 = _mm256_setzero_si256();
        for (int i = 0; i < n; i += 32) {
            __m256i a0 = _mm256_loadu_si256((const __m256i *)(a + i));
            __m256i b0 = _mm256_loadu_si256((const __m256i *)(b + i));
            __m256i p0 = _mm256_maddubs_epi16(a0, b0);
            s0 = _mm256_add_epi16(s0, p0);
            s1 = _mm256_add_epi16(s1, p0);
        }
        acc = _mm256_add_epi16(acc, s0);
        acc = _mm256_add_epi16(acc, s1);
    }
    short v[16]; _mm256_storeu_si256((__m256i *)v, acc);
    for (int i = 0; i < 16; i++) r += v[i];
    return r;
}

int main(int argc, char **argv) {
    int n = 1 << 16;      /* 64K elements */
    int iters = 20000;
    if (argc > 1) n = atoi(argv[1]);
    if (argc > 2) iters = atoi(argv[2]);
    float *a = (float *)malloc((size_t)n * sizeof(float));
    float *b = (float *)malloc((size_t)n * sizeof(float));
    unsigned char *ai = (unsigned char *)malloc((size_t)n);
    char *bi = (char *)malloc((size_t)n);
    srand(42);
    for (int i = 0; i < n; i++) { a[i] = (float)(rand() % 2000 - 1000) / 100.0f; b[i] = (float)(rand() % 100 - 50) / 100.0f; ai[i] = (unsigned char)(rand() & 0xFF); bi[i] = (char)(rand() & 0xFF); }
    /* warmup */
    volatile float wf = dot_fma(a, b, n, 10);
    volatile long wi = dot_int8(ai, bi, n, 10);
    volatile long wi16 = dot_int8_16(ai, bi, n, 10);
    /* timing */
    int reps = 3;
    double t1 = omp_get_wtime();
    for (int r = 0; r < reps; r++) wf = dot_fma(a, b, n, iters);
    double t2 = omp_get_wtime();
    for (int r = 0; r < reps; r++) wi = dot_int8(ai, bi, n, iters);
    double t3 = omp_get_wtime();
    for (int r = 0; r < reps; r++) wi16 = dot_int8_16(ai, bi, n, iters);
    double t4 = omp_get_wtime();
    double macs = (double)n * iters * reps;
    printf("n=%d iters=%d\n", n, iters);
    printf("FMA        : %.3fs  %7.1f GMAC/s  (8 MAC/instr, 2 pipes)\n", t2 - t1, macs / 1e9 / (t2 - t1));
    printf("maddubs+32 : %.3fs  %7.1f GMAC/s  (16 MAC/instr + widen)\n", t3 - t2, macs / 1e9 / (t3 - t2));
    printf("maddubs+16 : %.3fs  %7.1f GMAC/s  (16 MAC/instr, int16 acc)\n", t4 - t3, macs / 1e9 / (t4 - t3));
    printf("RATIO vs FMA: int8-widen %.2fx  int8-int16 %.2fx\n",
           (t2 - t1) / (t3 - t2), (t2 - t1) / (t4 - t3));
    printf("checks: %.2f %ld %ld\n", (double)wf, wi, wi16);
    free(a); free(b); free(ai); free(bi);
    return 0;
}
