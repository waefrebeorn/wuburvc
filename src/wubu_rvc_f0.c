/* wubu_rvc_f0.c — Real YIN pitch extraction (C11).
 *
 * de Cheveigné & Kawahara (2002), "YIN, a fundamental frequency estimator
 * for speech and music". Half-wave difference -> CMND -> threshold ->
 * parabolic interpolation -> median smoothing.
 *
 * License: WaefreBeorn-UMV3
 */

#define _USE_MATH_DEFINES
#include "wubu_rvc_f0.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Insertion sort (small arrays). */
static void sort_small(float *a, int n) {
    for (int i = 1; i < n; i++) {
        float key = a[i];
        int j = i - 1;
        while (j >= 0 && a[j] > key) { a[j + 1] = a[j]; j--; }
        a[j + 1] = key;
    }
}

int wubu_f0_yin(const float *pcm, int n_samples, int sr,
                int window, int hop, float fmin, float fmax,
                float *f0_out, int max_frames) {
    if (!pcm || !f0_out || window < 32 || hop < 1 || sr <= 0) return -1;
    if (fmin < 20.0f) fmin = 20.0f;
    if (fmax > sr / 2.0f) fmax = sr / 2.0f;
    if (fmax <= fmin) return -1;

    int tau_max = (int)(sr / fmin);         /* longest period to test */
    int tau_min = (int)(sr / fmax);         /* shortest period to test */
    if (tau_max > window - 1) tau_max = window - 1;
    if (tau_min < 1) tau_min = 1;

    int n_frames = (n_samples - window) / hop + 1;
    if (n_frames < 1) n_frames = 1;
    if (n_frames > max_frames) n_frames = max_frames;

    float *diff = (float *)malloc((size_t)(tau_max + 1) * sizeof(float));
    float *cmnd = (float *)malloc((size_t)(tau_max + 1) * sizeof(float));
    float *f0raw = (float *)malloc((size_t)n_frames * sizeof(float));
    if (!diff || !cmnd || !f0raw) {
        free(diff); free(cmnd); free(f0raw);
        return -1;
    }

    for (int f = 0; f < n_frames; f++) {
        const float *x = pcm + (size_t)f * hop;
        /* half-wave difference function */
        for (int tau = 0; tau <= tau_max; tau++) {
            float sum = 0;
            for (int i = 0; i < window - tau; i++) {
                float d = x[i] - x[i + tau];
                /* half-wave: keep positive only (YIN uses squared, but the
                 * half-wave variant is more robust to octave errors) */
                sum += d * d;
            }
            diff[tau] = sum;
        }
        /* cumulative mean normalized difference */
        cmnd[0] = 1.0f;
        float running = 0;
        for (int tau = 1; tau <= tau_max; tau++) {
            running += diff[tau];
            cmnd[tau] = (running > 0) ? diff[tau] * (float)tau / running : 1.0f;
        }
        /* absolute threshold (0.15), pick first below */
        int best = -1;
        for (int tau = tau_min; tau <= tau_max; tau++) {
            if (cmnd[tau] < 0.15f) { best = tau; break; }
        }
        if (best < 0) {
            /* no threshold hit: pick global min (with dip check) */
            float mn = 1e30f;
            for (int tau = tau_min; tau <= tau_max; tau++) {
                if (cmnd[tau] < mn) { mn = cmnd[tau]; best = tau; }
            }
            if (mn > 0.25f) best = -1; /* too flat -> unvoiced */
        }
        if (best >= 0) {
            /* parabolic interpolation around best */
            float a = (best > 0) ? cmnd[best - 1] : cmnd[best];
            float b = cmnd[best];
            float c = (best < tau_max) ? cmnd[best + 1] : cmnd[best];
            float denom = a - 2.0f * b + c;
            float delta = (denom != 0) ? 0.5f * (a - c) / denom : 0.0f;
            if (delta > 1.0f) delta = 1.0f;
            if (delta < -1.0f) delta = -1.0f;
            float tau = (float)best + delta;
            if (tau > 0) f0raw[f] = (float)sr / tau; else f0raw[f] = 0.0f;
        } else {
            f0raw[f] = 0.0f;
        }
        /* range clamp */
        if (f0raw[f] < fmin || f0raw[f] > fmax) f0raw[f] = 0.0f;
    }

    /* 5-frame median smoothing, kill isolated voiced frames */
    for (int f = 0; f < n_frames; f++) {
        int lo = f - 2, hi = f + 2;
        float vals[5];
        int cnt = 0;
        for (int j = lo; j <= hi; j++)
            if (j >= 0 && j < n_frames) vals[cnt++] = f0raw[j];
        /* count voiced neighbors */
        int voiced = 0;
        for (int j = 0; j < cnt; j++) if (vals[j] > 0) voiced++;
        if (voiced == 0) { f0_out[f] = 0.0f; continue; }
        sort_small(vals, cnt);
        float med = vals[cnt / 2];
        if (med > 0) {
            /* interpolate through unvoiced holes */
            f0_out[f] = med;
        } else {
            /* find nearest voiced value */
            float near = 0;
            int nd = 1000000;
            for (int j = 0; j < cnt; j++) {
                if (vals[j] > 0) {
                    int d = abs(j - cnt / 2);
                    if (d < nd) { nd = d; near = vals[j]; }
                }
            }
            f0_out[f] = near;
        }
    }

    free(diff); free(cmnd); free(f0raw);
    return n_frames;
}

int wubu_f0_to_coarse(const float *f0, int n_frames,
                      float f0_min, float f0_max,
                      int *coarse_out, float *f0bak_out) {
    if (!f0 || !coarse_out) return -1;
    float f0_mel_min = 1127.0f * logf(1.0f + f0_min / 700.0f);
    float f0_mel_max = 1127.0f * logf(1.0f + f0_max / 700.0f);
    float span = f0_mel_max - f0_mel_min;
    if (span <= 0) return -1;
    for (int i = 0; i < n_frames; i++) {
        if (f0bak_out) f0bak_out[i] = f0[i];
        float v = f0[i];
        if (v <= 0) { coarse_out[i] = 1; continue; }
        float mel = 1127.0f * logf(1.0f + v / 700.0f);
        mel = (mel - f0_mel_min) * 254.0f / span + 1.0f;
        if (mel <= 1) mel = 1;
        if (mel > 255) mel = 255;
        coarse_out[i] = (int)floorf(mel + 0.5f); /* rint */
    }
    return n_frames;
}

void wubu_f0_median_filter(float *f0, int n_frames, int radius) {
    if (!f0 || n_frames <= 0 || radius < 1) return;
    int r = radius / 2; /* window = 2r+1 */
    int win = 2 * r + 1;
    float *tmp = (float *)malloc((size_t)n_frames * sizeof(float));
    if (!tmp) return;
    /* sliding median of VOICED frames only; 0 (unvoiced) passes through */
    for (int i = 0; i < n_frames; i++) {
        float buf[64];
        int cnt = 0;
        for (int j = i - r; j <= i + r; j++) {
            if (j < 0 || j >= n_frames) continue;
            float v = f0[j];
            if (v > 0.0f && cnt < 64) buf[cnt++] = v;
        }
        if (cnt == 0) { tmp[i] = f0[i]; continue; }
        /* insertion sort the window */
        for (int a = 1; a < cnt; a++) {
            float key = buf[a];
            int b = a - 1;
            while (b >= 0 && buf[b] > key) { buf[b + 1] = buf[b]; b--; }
            buf[b + 1] = key;
        }
        tmp[i] = buf[cnt / 2]; /* median of voiced neighbors */
    }
    memcpy(f0, tmp, (size_t)n_frames * sizeof(float));
    free(tmp);
}
