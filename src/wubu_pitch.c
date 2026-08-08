/* wubu_pitch.c — phase-vocoder pitch shifter (no tempo change).
 *
 * Algorithm (classic phase vocoder, bin-reassignment variant):
 *   1. STFT (hann, 75% overlap, N=2048, H=512).
 *   2. Per frame: FFT; for each bin estimate the instantaneous frequency
 *      omega[k] = binDelta + princarg(phase_m[k] - phase_{m-1}[k] - binDelta)
 *      (binDelta = 2*pi*k*H/N — the expected phase advance at hop H).
 *   3. Reassign each source bin k to dest k' = round(k*alpha) (alpha =
 *      2^(semitones/12)): magnitude is copied, the DEST phase advances by
 *      alpha * omega[k] per frame — this is what changes the pitch while
 *      keeping the frame timeline (duration) intact.
 *   4. ISTFT + hann + overlap-add.
 *
 * Good enough for voice (octave/±few-semitone shifts); phase-locked bin
 * reassignment avoids the worst "phasiness".
 *
 * C11, self-contained (own radix-2 FFT), opaque state, minimal includes.
 * License: WaefreBeorn-UMV3
 */
#include "wubu_pitch.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define PV_N 2048
#define PV_H 512
#define PV_BINS (PV_N / 2 + 1)

typedef struct {
    float re, im;
} Cpx;

static void cfft(Cpx *a, int n, int inv) {
    /* iterative radix-2 FFT (Cooley-Tukey, bit-reversal) */
    for (int i = 1, j = 0; i < n; i++) {
        int bit = n >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) { Cpx t = a[i]; a[i] = a[j]; a[j] = t; }
    }
    for (int len = 2; len <= n; len <<= 1) {
        double ang = 2.0 * M_PI / len * (inv ? -1.0 : 1.0);
        Cpx wl = { (float)cos(ang), (float)sin(ang) };
        for (int i = 0; i < n; i += len) {
            Cpx w = { 1.0f, 0.0f };
            for (int j = 0; j < len / 2; j++) {
                Cpx u = a[i + j];
                Cpx v = { a[i + j + len / 2].re * w.re - a[i + j + len / 2].im * w.im,
                          a[i + j + len / 2].re * w.im + a[i + j + len / 2].im * w.re };
                a[i + j] = (Cpx){ u.re + v.re, u.im + v.im };
                a[i + j + len / 2] = (Cpx){ u.re - v.re, u.im - v.im };
                Cpx nw = { w.re * wl.re - w.im * wl.im, w.re * wl.im + w.im * wl.re };
                w = nw;
            }
        }
    }
    if (inv)
        for (int i = 0; i < n; i++) { a[i].re /= n; a[i].im /= n; }
}

static float princarg(float p) {
    while (p > (float)M_PI) p -= 2.0f * (float)M_PI;
    while (p < -(float)M_PI) p += 2.0f * (float)M_PI;
    return p;
}

int wubu_pitch_shift(const float *in, int n, int sr, float semitones,
                     float *out) {
    if (!in || !out || n <= 0 || sr <= 0) return -1;
    if (fabsf(semitones) < 0.05f) {
        if (out != in) memcpy(out, in, (size_t)n * sizeof(float));
        return 0;
    }
    double alpha = pow(2.0, semitones / 12.0);
    const int N = PV_N, H = PV_H;

    float *win = (float *)malloc((size_t)N * sizeof(float));
    float *acc = (float *)calloc((size_t)(n + N), sizeof(float));
    float *wsum = (float *)calloc((size_t)(n + N), sizeof(float));
    Cpx *X = (Cpx *)malloc((size_t)N * sizeof(Cpx));
    Cpx *Y = (Cpx *)malloc((size_t)N * sizeof(Cpx));
    Cpx *prev = (Cpx *)calloc((size_t)PV_BINS, sizeof(Cpx));   /* prev frame spectrum */
    float *phout = (float *)calloc((size_t)PV_BINS, sizeof(float)); /* dest phases */
    int *taken = (int *)calloc((size_t)PV_BINS, sizeof(int));       /* strongest source per dest */
    float *magmax = (float *)calloc((size_t)PV_BINS, sizeof(float));
    if (!win || !acc || !wsum || !X || !Y || !prev || !phout || !taken || !magmax) {
        free(win); free(acc); free(wsum); free(X); free(Y); free(prev);
        free(phout); free(taken); free(magmax);
        return -1;
    }
    for (int k = 0; k < N; k++) {
        /* periodic hann */
        win[k] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * k / N));
    }

    int n_frames = (n - N) / H + 1;
    if (n_frames < 1) n_frames = 1;
    const float bin_delta = 2.0f * (float)M_PI * H / N; /* per-bin phase advance per hop */

    for (int m = 0; m < n_frames; m++) {
        const float *xf = in + (size_t)m * H;
        for (int k = 0; k < N; k++) {
            float v = (m * H + k < n) ? xf[k] * win[k] : 0.0f;
            X[k].re = v; X[k].im = 0.0f;
        }
        cfft(X, N, 0);

        memset(Y, 0, (size_t)N * sizeof(Cpx));
        memset(taken, 0, (size_t)PV_BINS * sizeof(int));
        memset(magmax, 0, (size_t)PV_BINS * sizeof(float));

        for (int k = 0; k < PV_BINS; k++) {
            float mag = sqrtf(X[k].re * X[k].re + X[k].im * X[k].im);
            float ph = atan2f(X[k].im, X[k].re);
            if (m == 0) {
                /* first frame: no previous phase — direct assignment */
                int kd = (int)lroundf(k * (float)alpha);
                if (kd > 0 && kd <= N / 2 && mag > magmax[kd]) {
                    magmax[kd] = mag;
                    phout[kd] = ph;
                    taken[kd] = 1;
                }
            } else {
                float omega = bin_delta * k + princarg(ph - atan2f(prev[k].im, prev[k].re) - bin_delta * k);
                int kd = (int)lroundf(k * (float)alpha);
                if (kd > 0 && kd <= N / 2 && mag > magmax[kd]) {
                    magmax[kd] = mag;
                    /* dest phase advances by alpha * omega per frame */
                    phout[kd] = phout[kd] + (float)alpha * omega;
                    taken[kd] = 1;
                }
            }
        }
        /* build the output spectrum (conjugate-symmetric for a real signal) */
        for (int kd = 0; kd < PV_BINS; kd++) {
            if (!taken[kd]) continue;
            Y[kd].re = magmax[kd] * cosf(phout[kd]);
            Y[kd].im = magmax[kd] * sinf(phout[kd]);
        }
        for (int kd = 1; kd < N / 2; kd++) {
            Y[N - kd].re = Y[kd].re;
            Y[N - kd].im = -Y[kd].im;
        }
        cfft(Y, N, 1);

        /* overlap-add with the same hann (synthesis window = analysis) */
        int base = m * H;
        for (int k = 0; k < N; k++) {
            if (base + k >= n + N) break;
            float v = Y[k].re * win[k];
            acc[base + k] += v;
            wsum[base + k] += win[k] * win[k];
        }
        memcpy(prev, X, (size_t)PV_BINS * sizeof(Cpx));
    }

    for (int i = 0; i < n; i++) {
        float w = wsum[i] > 1e-9f ? wsum[i] : 1.0f;
        out[i] = acc[i] / w;
    }

    free(win); free(acc); free(wsum); free(X); free(Y); free(prev);
    free(phout); free(taken); free(magmax);
    return 0;
}
