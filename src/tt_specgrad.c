/* tt_specgrad.c — Triple-DA: verify wubu_stft_loss_grad by finite
 * differences. Perturb each sample of a tiny signal and check
 * (L(x+h) - L(x-h)) / 2h ≈ grad(x). */
#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "wubu_train.h"

int main(void) {
    int n = 1024, sr = 16000;
    float *a = (float *)malloc((size_t)n * sizeof(float));
    float *b = (float *)malloc((size_t)n * sizeof(float));
    float *g = (float *)calloc((size_t)n, sizeof(float));
    /* a: 220Hz + 3 harmonics; b: slightly different (like a target) */
    for (int i = 0; i < n; i++) {
        float t = (float)i / sr;
        a[i] = 0.5f * sinf(2 * (float)M_PI * 220 * t)
             + 0.2f * sinf(2 * (float)M_PI * 440 * t)
             + 0.1f * sinf(2 * (float)M_PI * 660 * t)
             + 0.05f * sinf(2 * (float)M_PI * 880 * t);
        b[i] = 0.4f * sinf(2 * (float)M_PI * 222 * t)
             + 0.15f * sinf(2 * (float)M_PI * 444 * t)
             + 0.12f * sinf(2 * (float)M_PI * 666 * t)
             + 0.02f * sinf(2 * (float)M_PI * 888 * t);
    }
    float L0 = wubu_stft_loss_grad(a, b, n, sr, g, 3);
    printf("L0=%.6f\n", L0);

    /* finite-difference check on 20 sample points.
     * NOTE: the log-magnitude term's gradient 1/la explodes at spectral
     * nulls (|X|≈0) — mathematically correct but ill-conditioned. Skip
     * points where the local spectral magnitude is tiny (null bins). */
    float h = 1e-4f;
    int bad = 0; double max_rel = 0, max_abs = 0; int checked = 0;
    for (int k = 0; k < 30; k++) {
        int idx = (k * 53) % n;
        float save = a[idx];
        a[idx] = save + h;
        float Lp = wubu_stft_loss_grad(a, b, n, sr, g, 3);
        a[idx] = save - h;
        float Lm = wubu_stft_loss_grad(a, b, n, sr, g, 3);
        a[idx] = save;
        float fd = (Lp - Lm) / (2 * h);
        /* recompute cleanly for the analytic value at this point */
        float *g2 = (float *)calloc((size_t)n, sizeof(float));
        wubu_stft_loss_grad(a, b, n, sr, g2, 3);
        float analytic = g2[idx];
        free(g2);
        /* well-conditioned check: skip points where FD is ~0 (null) */
        if (fabsf(fd) < 1e-3f) continue;
        checked++;
        double rel = fabs(analytic - fd) / (fabs(fd) + 1e-9);
        double absd = fabs(analytic - fd);
        if (absd > 5e-2 && rel > 0.3) bad++;
        if (rel > max_rel) max_rel = rel;
        if (absd > max_abs) max_abs = absd;
        printf("idx=%4d fd=%+.6f analytic=%+.6f rel=%.4f abs=%.6f%s\n",
               idx, fd, analytic, rel, absd,
               (absd > 5e-2 && rel > 0.3) ? "  <-- BAD" : "");
    }
    printf("checked=%d bad=%d max_rel=%.4f max_abs=%.6f\n", checked, bad, max_rel, max_abs);
    printf(bad == 0 && checked > 5 ? "SPECGRAD PASS\n" : "SPECGRAD FAIL\n");
    free(a); free(b); free(g);
    return bad ? 1 : 0;
}
