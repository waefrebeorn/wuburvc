#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Minimal replica of the direct-DFT gradient to calibrate the exact
 * normalization. Loss = 0.5·(Σ|Δlin|/nbins + Σ|Δlog|/nbins) per frame,
 * averaged over frames. Single scale, single frame first. */
int main(void) {
    int w = 128, sr = 16000, n = w;
    float *a = malloc((size_t)n * 4), *b = malloc((size_t)n * 4);
    for (int i = 0; i < n; i++) {
        float t = (float)i / sr;
        a[i] = 0.5f * sinf(2 * (float)M_PI * 220 * t) + 0.2f * sinf(2 * (float)M_PI * 440 * t);
        b[i] = 0.4f * sinf(2 * (float)M_PI * 222 * t) + 0.15f * sinf(2 * (float)M_PI * 444 * t);
    }
    float *win = malloc((size_t)w * 4);
    for (int i = 0; i < w; i++) win[i] = 1.0f;   /* flat: isolate DFT factor */
    int nbins = w / 2 + 1;

    /* forward magnitudes of a and b */
    float *re_a = malloc((size_t)nbins * 4), *im_a = malloc((size_t)nbins * 4);
    float *mag_b = malloc((size_t)nbins * 4);
    for (int k = 0; k < nbins; k++) {
        float ra = 0, ia = 0, rb = 0, ib = 0;
        for (int t = 0; t < w; t++) {
            float ph = -2.0f * (float)M_PI * k * t / w;
            ra += a[t] * win[t] * cosf(ph); ia += a[t] * win[t] * sinf(ph);
            rb += b[t] * win[t] * cosf(ph); ib += b[t] * win[t] * sinf(ph);
        }
        re_a[k] = ra; im_a[k] = ia;
        mag_b[k] = sqrtf(rb * rb + ib * ib);
    }
    float dsum = 0, dlog = 0;
    for (int k = 1; k < nbins; k++) {
        float ma = sqrtf(re_a[k] * re_a[k] + im_a[k] * im_a[k]);
        dsum += fabsf(ma - mag_b[k]);
        dlog += fabsf(logf(ma + 1e-10f) - logf(mag_b[k] + 1e-10f));
    }
    float L = 0.5f * (dsum / nbins + dlog / nbins);
    printf("L = %.6f\n", L);

    /* analytic gradient of L w.r.t a[n]:  dm_k·(Re cos - Im sin)/ma */
    float *g = calloc((size_t)n, 4);
    for (int k = 1; k < nbins; k++) {
        float ma = sqrtf(re_a[k] * re_a[k] + im_a[k] * im_a[k]);
        float la = ma + 1e-10f, lb = mag_b[k] + 1e-10f;
        float dm = 0.5f * ((ma > mag_b[k] ? 1 : -1) + (la > lb ? 1 : -1) / la) / nbins;
        for (int t = 0; t < w; t++) {
            float ph = -2.0f * (float)M_PI * k * t / w;
            g[t] += dm * (re_a[k] * cosf(ph) - im_a[k] * sinf(ph)) / ma * win[t];
        }
    }
    /* FD */
    float h = 1e-3f;
    for (int t = 0; t < 8; t++) {
        int idx = 10 + t * 15;
        float save = a[idx];
        a[idx] = save + h;
        /* recompute L */
        for (int k = 0; k < nbins; k++) {
            float ra = 0, ia = 0;
            for (int tt = 0; tt < w; tt++) {
                float ph = -2.0f * (float)M_PI * k * tt / w;
                ra += a[tt] * win[tt] * cosf(ph); ia += a[tt] * win[tt] * sinf(ph);
            }
            re_a[k] = ra; im_a[k] = ia;
        }
        float dsp = 0, dlp = 0;
        for (int k = 1; k < nbins; k++) {
            float ma = sqrtf(re_a[k] * re_a[k] + im_a[k] * im_a[k]);
            dsp += fabsf(ma - mag_b[k]);
            dlp += fabsf(logf(ma + 1e-10f) - logf(mag_b[k] + 1e-10f));
        }
        float Lp = 0.5f * (dsp / nbins + dlp / nbins);
        a[idx] = save - h;
        for (int k = 0; k < nbins; k++) {
            float ra = 0, ia = 0;
            for (int tt = 0; tt < w; tt++) {
                float ph = -2.0f * (float)M_PI * k * tt / w;
                ra += a[tt] * win[tt] * cosf(ph); ia += a[tt] * win[tt] * sinf(ph);
            }
            re_a[k] = ra; im_a[k] = ia;
        }
        float dsm = 0, dlm = 0;
        for (int k = 1; k < nbins; k++) {
            float ma = sqrtf(re_a[k] * re_a[k] + im_a[k] * im_a[k]);
            dsm += fabsf(ma - mag_b[k]);
            dlm += fabsf(logf(ma + 1e-10f) - logf(mag_b[k] + 1e-10f));
        }
        float Lm = 0.5f * (dsm / nbins + dlm / nbins);
        a[idx] = save;
        float fd = (Lp - Lm) / (2 * h);
        printf("idx=%3d fd=%+.6f an=%+.6f ratio=%.4f\n", idx, fd, g[idx], fd / (g[idx] + 1e-12));
    }
    return 0;
}
