#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* only the two kernels under test — no engine headers */
void conv1d_c(const float *in, int in_ch, int n,
              const float *w, const float *b,
              int out_ch, int k, int stride, int pad, int dil,
              float *out);
void conv1d_c_fused(const float *in, int in_ch, int n,
                    const float *w, const float *b,
                    int out_ch, int k, int stride, int pad, int dil,
                    float *out, float in_slope, const float *resid);

/* weak stub definitions for the symbols real.c references but this test
 * never calls (link-time only; bodies never execute) */
float wubu_fastexp(float x) __attribute__((weak));
float wubu_fastexp(float x) { return x; }
int wubu_generator_nsf_cuda(void) __attribute__((weak));
int wubu_generator_nsf_cuda(void) { return -1; }
void *wubu_vk_create(void) __attribute__((weak));
void *wubu_vk_create(void) { return 0; }
int wubu_vk_generator_nsf(void) __attribute__((weak));
int wubu_vk_generator_nsf(void) { return -1; }
int wubu_rvc_find_tensor(void) __attribute__((weak));
int wubu_rvc_find_tensor(void) { return 0; }

static void lrelu(float *x, size_t n, float slope) {
    for (size_t i = 0; i < n; i++) x[i] = x[i] > 0 ? x[i] : slope * x[i];
}

/* OLD sequence: x = conv2(lrelu(conv1(lrelu(x)))) + x, via 7 passes */
static void old_pair(float *x, float *tmp, float *out,
                     const float *d1, const float *d2,
                     int ch, int n, int k, int pad1, int dil, int pad2) {
    memcpy(tmp, x, (size_t)ch * n * sizeof(float));
    lrelu(tmp, (size_t)ch * n, 0.1f);
    conv1d_c(tmp, ch, n, d1, NULL, ch, k, 1, pad1, dil, out);
    lrelu(out, (size_t)ch * n, 0.1f);
    conv1d_c(out, ch, n, d2, NULL, ch, k, 1, pad2, 1, tmp);
    for (int i = 0; i < ch * n; i++) tmp[i] += x[i];
    memcpy(x, tmp, (size_t)ch * n * sizeof(float));
}

/* NEW sequence: same math, 2 fused calls, in-place residual */
static void new_pair(float *x, float *out,
                     const float *d1, const float *d2,
                     int ch, int n, int k, int pad1, int dil, int pad2) {
    conv1d_c_fused(x, ch, n, d1, NULL, ch, k, 1, pad1, dil, out, 0.1f, NULL);
    conv1d_c_fused(out, ch, n, d2, NULL, ch, k, 1, pad2, 1, x, 0.1f, x);
}

int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 798;
    int ch = 192;
    int pairs = argc > 2 ? atoi(argv[2]) : 3;
    float *xa = malloc((size_t)ch * n * sizeof(float));
    float *xb = malloc((size_t)ch * n * sizeof(float));
    float *tmp = malloc((size_t)ch * n * sizeof(float));
    float *out = malloc((size_t)ch * n * sizeof(float));
    float *d1 = malloc((size_t)ch * ch * 11 * sizeof(float));
    float *d2 = malloc((size_t)ch * ch * 11 * sizeof(float));
    srand(11);
    for (int i = 0; i < ch * n; i++) { xa[i] = xb[i] = ((float)rand()/RAND_MAX-0.5f)*0.2f; }
    for (int i = 0; i < ch * ch * 11; i++) { d1[i] = d2[i] = ((float)rand()/RAND_MAX-0.5f)*0.02f; }

    int ks[3] = {3, 7, 11};
    int dils[3] = {1, 3, 5};
    for (int p = 0; p < pairs; p++) {
        int k = ks[p % 3];
        int dil = dils[p % 3];
        int pad1 = dil * (k - 1) / 2;
        int pad2 = k / 2;
        old_pair(xa, tmp, out, d1, d2, ch, n, k, pad1, dil, pad2);
        new_pair(xb, out, d1, d2, ch, n, k, pad1, dil, pad2);
    }
    double maxdiff = 0; int nbad = 0;
    for (int i = 0; i < ch * n; i++) {
        double d = fabs((double)xa[i] - xb[i]);
        if (d > maxdiff) maxdiff = d;
        if (d > 1e-6) nbad++;
    }
    printf("n=%d pairs=%d maxdiff=%.9f nbad=%d (of %d)\n", n, pairs, maxdiff, nbad, ch*n);
    printf("xa[0..3]=%.6f %.6f %.6f %.6f  xb[0..3]=%.6f %.6f %.6f %.6f\n",
           xa[0], xa[1], xa[2], xa[3], xb[0], xb[1], xb[2], xb[3]);
    free(xa); free(xb); free(tmp); free(out); free(d1); free(d2);
    return (maxdiff <= 1e-6) ? 0 : 1;
}
