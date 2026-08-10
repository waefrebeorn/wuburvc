#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "wubu_train.h"
int main(void) {
    int n = 512, sr = 16000;  /* single frame for w=512, hop=128 -> n_frames=1 */
    float *a = malloc((size_t)n * 4), *b = malloc((size_t)n * 4);
    for (int i = 0; i < n; i++) {
        float t = (float)i / sr;
        a[i] = 0.5f * sinf(2 * (float)M_PI * 220 * t) + 0.2f * sinf(2 * (float)M_PI * 440 * t);
        b[i] = 0.4f * sinf(2 * (float)M_PI * 222 * t) + 0.15f * sinf(2 * (float)M_PI * 444 * t);
    }
    float *g = calloc((size_t)n, 4);
    float L0 = wubu_stft_loss_grad(a, b, n, sr, g, 1);
    printf("L0=%.6f g[100]=%.6f g[300]=%.6f\n", L0, g[100], g[300]);
    float h = 1e-4f;
    for (int t = 0; t < 6; t++) {
        int idx = 40 + t * 80;
        float save = a[idx];
        a[idx] = save + h; float Lp = wubu_stft_loss_grad(a, b, n, sr, g, 1);
        a[idx] = save - h; float Lm = wubu_stft_loss_grad(a, b, n, sr, g, 1);
        a[idx] = save;
        float fd = (Lp - Lm) / (2 * h);
        float *g2 = calloc((size_t)n, 4);
        wubu_stft_loss_grad(a, b, n, sr, g2, 1);
        printf("idx=%3d fd=%+.6f an=%+.6f ratio=%.4f\n", idx, fd, g2[idx], fd / (g2[idx] + 1e-12));
        free(g2);
    }
    return 0;
}
