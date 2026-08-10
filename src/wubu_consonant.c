/* wubu_consonant.c — spectral-flatness voicing/consonant detection.
 *
 * See wubu_consonant.h. Computes per-frame Wiener entropy (spectral
 * flatness) from a Hann-windowed STFT, then combines with the f0 contour
 * to emit a uv mask: voiced = peaked spectrum AND (f0 > 0 OR strong low-freq
 * energy); unvoiced = flat spectrum (noise-like) — s/sh/f, plosive bursts,
 * breath. The mask feeds the generator so unvoiced frames get noise
 * excitation (source-filter SOTA).
 *
 * License: WaefreBeorn-UMV3
 */

#define _USE_MATH_DEFINES
#include "wubu_consonant.h"
#include "wubu_fft.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

int wubu_consonant_uv(const float *pcm, int n_samples, int sr,
                      const float *f0, int n_frames,
                      float *uv_out, float *flat_out,
                      int window, int hop) {
    if (!pcm || !uv_out || n_frames < 1) return -1;
    if (window < 64 || hop < 1 || sr <= 0) return -1;
    int nfft = 1;
    while (nfft < window) nfft <<= 1;

    WuBuCpx *spec = (WuBuCpx *)malloc((size_t)nfft * sizeof(WuBuCpx));
    float *hann = (float *)malloc((size_t)window * sizeof(float));
    if (!spec || !hann) { free(spec); free(hann); return -1; }
    for (int i = 0; i < window; i++)
        hann[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / (float)(window - 1)));

    int nbins = nfft / 2 + 1;
    float log_min = 1e-12f;

    for (int fi = 0; fi < n_frames; fi++) {
        int start = fi * hop;
        if (start + window > n_samples) break;
        const float *x = pcm + start;

        memset(spec, 0, (size_t)nfft * sizeof(WuBuCpx));
        for (int i = 0; i < window; i++) {
            spec[i].re = x[i] * hann[i];
            spec[i].im = 0.0f;
        }
        wubu_fft(spec, nfft, 0);

        /* magnitude + log-domain sums for Wiener entropy */
        double sum_log = 0.0, sum_mag = 0.0;
        int valid = 0;
        for (int b = 0; b < nbins; b++) {
            float m = sqrtf(spec[b].re * spec[b].re + spec[b].im * spec[b].im);
            if (m > 1e-9f) {
                sum_log += logf(m);
                sum_mag += m;
                valid++;
            }
        }
        float flatness = 1.0f;
        if (valid > 1 && sum_mag > 1e-9f) {
            /* Wiener entropy = exp(mean log) / mean — 0 peaked, 1 flat */
            double geo = exp(sum_log / (double)valid);
            double arith = sum_mag / (double)valid;
            flatness = (float)(geo / (arith + log_min));
            if (flatness < 0.0f) flatness = 0.0f;
            if (flatness > 1.0f) flatness = 1.0f;
        }

        /* energy in the frame (voiced frames are usually louder) */
        double e = 0.0;
        for (int i = 0; i < window; i++) e += x[i] * x[i];
        float rms = (float)sqrt(e / (double)window);

        /* voicing decision:
         *   - f0 says voiced AND spectrum peaked  -> voiced
         *   - flat spectrum (noise-like)          -> unvoiced/consonant
         *   - very quiet frame                    -> unvoiced (breath/stop)
         * Threshold: flatness > 0.6 is clearly noise-like; < 0.35 clearly
         * harmonic. In between, trust f0 if present. */
        int f0_voiced = (f0 && f0[fi] > 0.0f);
        float uv;
        if (flatness > 0.65f) {
            uv = 0.0f;                       /* flat noise -> unvoiced */
        } else if (flatness < 0.35f) {
            uv = 1.0f;                       /* peaked harmonics -> voiced */
        } else if (f0_voiced && flatness < 0.55f) {
            uv = 1.0f;                       /* f0 + moderate peak -> voiced */
        } else {
            uv = 0.0f;
        }
        /* very quiet frames: treat as unvoiced (plosive gap, breath) */
        if (rms < 0.01f) uv = 0.0f;

        uv_out[fi] = uv;
        if (flat_out) flat_out[fi] = flatness;
    }

    free(spec); free(hann);
    return n_frames;
}
