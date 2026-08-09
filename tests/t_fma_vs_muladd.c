/* t_fma_vs_muladd.c — Zen2: does mul+add (FP0/1 + FP2/3) beat FMA (FP0/1 only)?
 * The SO answer (Peter Cordes) says Zen2 sustains 2 vaddpd + 2 vmulpd/clock on
 * separate ports, vs 2 FMA/clock on one port pair — mul+add should win ~1.1-2x
 * for FMA-latency-bound loops. Verify on THIS Ryzen 3600. */
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <immintrin.h>

static double now_s(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* 8 independent accumulators, FMA chain (Intel-style) */
static float bench_fma(float *a, float *b, int n, int iters) {
    float r = 0;
    for (int it = 0; it < iters; it++) {
        __m256 acc0 = _mm256_set1_ps(0.001f), acc1 = acc0, acc2 = acc0, acc3 = acc0;
        __m256 acc4 = acc0, acc5 = acc0, acc6 = acc0, acc7 = acc0;
        for (int i = 0; i + 8 <= n; i += 8) {
            __m256 x = _mm256_loadu_ps(a + i);
            __m256 y = _mm256_loadu_ps(b + i);
            acc0 = _mm256_fmadd_ps(x, y, acc0);
            acc1 = _mm256_fmadd_ps(x, y, acc1);
            acc2 = _mm256_fmadd_ps(x, y, acc2);
            acc3 = _mm256_fmadd_ps(x, y, acc3);
            acc4 = _mm256_fmadd_ps(x, y, acc4);
            acc5 = _mm256_fmadd_ps(x, y, acc5);
            acc6 = _mm256_fmadd_ps(x, y, acc6);
            acc7 = _mm256_fmadd_ps(x, y, acc7);
        }
        acc0 = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3)),
                             _mm256_add_ps(_mm256_add_ps(acc4, acc5), _mm256_add_ps(acc6, acc7)));
        r += ((float*)&acc0)[0];
    }
    return r;
}

/* same with separate mul+add (Zen2: mul on FP0/1, add on FP2/3) */
static float bench_muladd(float *a, float *b, int n, int iters) {
    float r = 0;
    for (int it = 0; it < iters; it++) {
        __m256 acc0 = _mm256_set1_ps(0.001f), acc1 = acc0, acc2 = acc0, acc3 = acc0;
        __m256 acc4 = acc0, acc5 = acc0, acc6 = acc0, acc7 = acc0;
        for (int i = 0; i + 8 <= n; i += 8) {
            __m256 x = _mm256_loadu_ps(a + i);
            __m256 y = _mm256_loadu_ps(b + i);
            __m256 p0 = _mm256_mul_ps(x, y);
            acc0 = _mm256_add_ps(p0, acc0);
            acc1 = _mm256_add_ps(p0, acc1);
            acc2 = _mm256_add_ps(p0, acc2);
            acc3 = _mm256_add_ps(p0, acc3);
            acc4 = _mm256_add_ps(p0, acc4);
            acc5 = _mm256_add_ps(p0, acc5);
            acc6 = _mm256_add_ps(p0, acc6);
            acc7 = _mm256_add_ps(p0, acc7);
        }
        acc0 = _mm256_add_ps(_mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3)),
                             _mm256_add_ps(_mm256_add_ps(acc4, acc5), _mm256_add_ps(acc6, acc7)));
        r += ((float*)&acc0)[0];
    }
    return r;
}

int main(void) {
    const int n = 1 << 20, iters = 2000;
    float *a = (float*)malloc(n * sizeof(float));
    float *b = (float*)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) { a[i] = (float)(i % 7) * 0.01f; b[i] = (float)(i % 13) * 0.01f; }
    /* warmup */
    bench_fma(a, b, n, 50); bench_muladd(a, b, n, 50);
    double t0 = now_s(); volatile float r1 = bench_fma(a, b, n, iters);
    double t1 = now_s(); volatile float r2 = bench_muladd(a, b, n, iters);
    double t2 = now_s();
    double flops = (double)n * iters * 16; /* 8 accs x 8 lanes x (mul+add) */
    printf("FMA    : %.3f s  (%.1f GFLOP/s)  r=%f\n", t1 - t0, flops / (t1 - t0) / 1e9, r1);
    printf("MUL+ADD: %.3f s  (%.1f GFLOP/s)  r=%f\n", t2 - t1, flops / (t2 - t1) / 1e9, r2);
    printf("ratio muladd/fma = %.3fx\n", (t1 - t0) / (t2 - t1));
    free(a); free(b);
    return 0;
}
