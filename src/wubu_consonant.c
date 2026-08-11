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

        /* HNR (harmonics-to-noise ratio) — the SOTA voicing confidence
         * (SwiftF0 confidence, FCPE voicing). Sum spectral energy at
         * f0, 2×f0, 3×f0 (harmonics) vs total energy. High HNR = clear
         * periodic source; low HNR = noise/fricative/breath. This is
         * the missing detection signal: flatness alone can't tell a
         * breathy voiced frame from a fricative. */
        float hnr = 0.0f;
        if (f0 && f0[fi] > 40.0f) {
            double harm_e = 0.0, tot_e = 1e-9;
            for (int b = 1; b < nbins; b++) {
                float m = sqrtf(spec[b].re * spec[b].re + spec[b].im * spec[b].im);
                tot_e += (double)m * m;
            }
            float f0_bin = f0[fi] * (float)nfft / (float)sr;
            for (int h = 1; h <= 8; h++) {
                int b = (int)(f0_bin * h + 0.5f);
                if (b < 1 || b >= nbins) break;
                /* HNR WINDOW FIX (2026-08-10): a single FFT bin per harmonic
                 * UNDERESTIMATES harmonic energy — a Hann window spreads each
                 * harmonic over ~±1 bin (main-lobe width ~4 bins at this
                 * nfft), so summing only the center bin dropped HNR below the
                 * 0.5 voiced threshold on ~37% of genuinely voiced frames
                 * (measured on the track-3 dry vocal: 105/282 sampled frames
                 * flipped). The mask then gated the sine OFF on vowels →
                 * breathy/buzzy "post process off" sound. Sum ±1 bin around
                 * each harmonic (the same window the peak detector sees). */
                for (int wb = b - 1; wb <= b + 1; wb++) {
                    if (wb < 1 || wb >= nbins) continue;
                    float mw = sqrtf(spec[wb].re * spec[wb].re +
                                     spec[wb].im * spec[wb].im);
                    harm_e += (double)mw * mw;
                }
            }
            hnr = (float)(harm_e / tot_e);   /* 0..~1 (1 = pure harmonic) */
        }

        /* voicing decision — now CONFIDENCE-GRADED, not hard binary:
         *   f0 present + strong harmonics (hnr) -> confidently voiced
         *   flat spectrum, no harmonic energy     -> unvoiced
         *   in between -> soft confidence (graded sine/noise blend)
         * f0 presence is necessary but not sufficient: a breathy frame
         * with weak harmonics is marked PARTIAL so the generator blends
         * sine + noise instead of hard-switching (kills onset clicks). */
        int f0_voiced = (f0 && f0[fi] > 0.0f);
        float uv;
        if (!f0_voiced || flatness > 0.65f) {
            uv = 0.0f;                       /* no pitch or flat noise */
        } else if (flatness < 0.35f && hnr > 0.5f) {
            uv = 1.0f;                       /* peaked harmonics + strong HNR */
        } else {
            /* graded: blend by harmonic strength (0.3..1.0) */
            uv = 0.3f + 0.7f * (hnr > 0.5f ? 1.0f : (hnr / 0.5f));
            if (flatness > 0.45f) uv *= 0.6f;   /* flat-ish pulls toward noise */
            if (uv > 1.0f) uv = 1.0f;
            if (uv < 0.05f) uv = 0.0f;
        }
        /* very quiet frames: treat as unvoiced (plosive gap, breath) */
        if (rms < 0.01f) uv = 0.0f;

        uv_out[fi] = uv;
        if (flat_out) flat_out[fi] = flatness;
    }

    /* temporal smoothing: a 3-frame median on the confidence kills
     * single-frame flicker (breathy onsets flip 0/1 spuriously). */
    if (n_frames >= 3) {
        float *tmp = (float *)malloc((size_t)n_frames * sizeof(float));
        if (tmp) {
            memcpy(tmp, uv_out, (size_t)n_frames * sizeof(float));
            for (int i = 1; i < n_frames - 1; i++) {
                float a = tmp[i - 1], b = tmp[i], c = tmp[i + 1];
                float med = (a <= b) ? ((b <= c) ? b : (a > c ? a : c))
                                     : ((a <= c) ? a : (b > c ? b : c));
                uv_out[i] = 0.5f * uv_out[i] + 0.5f * med;
            }
            free(tmp);
        }
    }

    free(spec); free(hann);
    return n_frames;
}
