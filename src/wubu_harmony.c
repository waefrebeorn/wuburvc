/* wubu_harmony.c — dual-fundamental (harmony) pitch detection.
 *
 * See wubu_harmony.h. Harmonic-sum salience over a Hann-windowed STFT.
 * Detects the lead fundamental (continuity-stabilized) and a second
 * simultaneous fundamental (harmony) when present, so the generator can
 * render both tones.
 *
 * License: WaefreBeorn-UMV3
 */

#define _USE_MATH_DEFINES
#include "wubu_harmony.h"
#include "wubu_fft.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* number of harmonics summed for salience */
#define WUBU_HARM_NHARM 10

typedef struct {
    float freq;    /* candidate fundamental (Hz) */
    float sal;     /* harmonic-sum salience (post-bonus) */
    float raw;     /* raw salience BEFORE continuity bonus (harmony gate) */
    int   idx;     /* grid index */
} WubuHarmCand;

static int cmp_cand_desc(const void *a, const void *b) {
    const WubuHarmCand *ca = (const WubuHarmCand *)a;
    const WubuHarmCand *cb = (const WubuHarmCand *)b;
    if (cb->sal > ca->sal) return 1;
    if (cb->sal < ca->sal) return -1;
    return 0;
}

/* is f2 a (near-)harmonic multiple of f1? tolerate ±0.6 semitones.
 * Returns 1 if f2/f1 rounds to a small integer (2..6) or is within
 * 0.6 semitones of f1 itself (unison). */
static int is_harmonic_multiple(float f1, float f2) {
    if (f1 <= 0 || f2 <= 0) return 1;
    float ratio = f2 / f1;
    if (ratio < 0.9f) return 1;         /* below unison -> treat as alias */
    /* exact integer multiple (octave/fifth-up-subharmonic traps) */
    int n = (int)lroundf(ratio);
    if (n >= 2 && n <= 6) {
        float cents = 1200.0f * log2f(ratio / (float)n);
        if (fabsf(cents) < 70.0f) return 1;
    }
    if (fabsf(ratio - 1.0f) < 0.05f) return 1;
    return 0;
}

/* semitone-quantized harmony interval search.
 * Autotune harmony vocals are quantized to exact semitones by definition.
 * Instead of grabbing whatever spectral peak is second-strongest (which
 * jumps between partials of BOTH voices), we search the salience map for a
 * candidate at a MUSICAL interval above the lead: major/minor 3rd, 4th,
 * 5th, 6th, octave, + octave variants. Returns the best harmony frequency
 * and its relative gain, or 0 if none. */
static float harmony_at_interval(const WubuHarmCand *cands, int n_cand,
                                 float lead, float lead_raw_sal,
                                 const float *cand_f, int n_cand_total,
                                 float *out_gain) {
    if (lead <= 0.0f || lead_raw_sal <= 1e-9f) return 0.0f;
    /* musical intervals (semitones above lead). NO octave/15/16/17/19/24:
     * a singer's own 2nd/3rd partial sits at exactly those frequencies, so
     * an "octave harmony" would be a false positive from the lead's own
     * overtone, not a second voice. Real pop/autotune harmony = 3rds, 4ths,
     * 5ths, 6ths (and a 7th). */
    static const int st[] = {3, 4, 5, 7, 9, 10};
    static const int nst = 6;
    float best = 0.0f, best_gain = 0.0f;
    for (int i = 0; i < nst; i++) {
        float target = lead * powf(2.0f, st[i] / 12.0f);
        if (target < 50.0f || target > 1100.0f) continue;
        /* nearest grid candidate to target */
        float best_d = 1e9f;
        int bi = -1;
        for (int c = 0; c < n_cand_total; c++) {
            float d = fabsf(cand_f[c] - target);
            if (d < best_d) { best_d = d; bi = c; }
        }
        if (bi < 0) continue;
        /* allow ±0.5 semitone tolerance; use the raw salience at that bin */
        float tol_hz = lead * powf(2.0f, 0.5f / 12.0f) - lead * powf(2.0f, -0.5f / 12.0f);
        if (best_d > tol_hz) continue;
        float s = cands[bi].raw;
        float rel = s / lead_raw_sal;
        if (rel >= 0.22f && rel > best_gain) {
            best = cands[bi].freq;
            best_gain = rel > 1.0f ? 1.0f : rel;
        }
    }
    if (out_gain) *out_gain = best_gain;
    return best;
}

int wubu_harmony_detect(const float *pcm, int n_samples, int sr,
                        const float *primary_in, int n_frames,
                        float *primary_out, float *harmony_out,
                        float *harmony_gain, int max_frames,
                        int window, int hop, float fmin, float fmax) {
    if (!pcm || (!primary_out && !harmony_out && !harmony_gain)) return -1;
    if (n_frames < 1 || n_frames > max_frames) return -1;
    if (window < 64 || hop < 1 || sr <= 0) return -1;
    if (fmax <= fmin || fmin < 20.0f) return -1;
    /* FFT must be power of 2 */
    int nfft = 1;
    while (nfft < window) nfft <<= 1;

    /* log-spaced candidate grid: ~36 per octave (33-cent steps) */
    int n_cand = 0;
    {
        float f = fmin;
        while (f <= fmax) { f *= powf(2.0f, 1.0f / 36.0f); n_cand++; }
        n_cand += 2;
    }
    float *cand_f = (float *)malloc((size_t)n_cand * sizeof(float));
    WubuHarmCand *cands = (WubuHarmCand *)malloc((size_t)n_cand * sizeof(WubuHarmCand));
    WuBuCpx *spec = (WuBuCpx *)malloc((size_t)nfft * sizeof(WuBuCpx));
    float *hann = (float *)malloc((size_t)window * sizeof(float));
    if (!cand_f || !cands || !spec || !hann) {
        free(cand_f); free(cands); free(spec); free(hann); return -1;
    }
    /* Hann window */
    for (int i = 0; i < window; i++)
        hann[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / (float)(window - 1)));
    /* grid */
    float f = fmin;
    for (int i = 0; i < n_cand; i++) { cand_f[i] = f; f *= powf(2.0f, 1.0f / 36.0f); }

    float prev_primary = 0.0f;   /* continuity anchor across frames */
    float bin_hz = (float)sr / (float)nfft;

    for (int fi = 0; fi < n_frames; fi++) {
        int start = fi * hop;
        if (start + window > n_samples) break;   /* short tail: stop */
        const float *x = pcm + start;

        /* window + FFT */
        memset(spec, 0, (size_t)nfft * sizeof(WuBuCpx));
        for (int i = 0; i < window; i++) {
            spec[i].re = x[i] * hann[i];
            spec[i].im = 0.0f;
        }
        wubu_fft(spec, nfft, 0);

        /* magnitude (half spectrum) */
        int nbins = nfft / 2 + 1;
        float *mag = (float *)malloc((size_t)nbins * sizeof(float));
        if (!mag) { free(cand_f); free(cands); free(spec); free(hann); return -1; }
        for (int b = 0; b < nbins; b++)
            mag[b] = sqrtf(spec[b].re * spec[b].re + spec[b].im * spec[b].im);

        /* salience over candidate grid */
        for (int c = 0; c < n_cand; c++) {
            float cf = cand_f[c];
            if (cf < fmin || cf > fmax) { cands[c].sal = 0.0f; cands[c].freq = cf; cands[c].idx = c; continue; }
            float s = 0.0f;
            for (int k = 1; k <= WUBU_HARM_NHARM; k++) {
                float hz = cf * (float)k;
                if (hz >= (float)sr * 0.45f) break;
                int b = (int)lroundf(hz / bin_hz);
                if (b >= 1 && b < nbins) s += mag[b] / (float)k;
            }
            cands[c].sal = s;
            cands[c].raw = s;
            cands[c].freq = cf;
            cands[c].idx = c;
        }

        /* continuity bonus toward previous primary (stops note-flipping) */
        if (prev_primary > 0.0f) {
            for (int c = 0; c < n_cand; c++) {
                if (cands[c].sal <= 0.0f) continue;
                float cents = 1200.0f * fabsf(log2f(cands[c].freq / prev_primary));
                if (cents < 60.0f) cands[c].sal *= 3.0f;   /* strong pull */
                else if (cents < 150.0f) cands[c].sal *= 1.5f;
            }
        }

        /* top candidates: global argmax (post-bonus) is the lead; then
         * collect the next local maxima for context. Strict local-max over
         * the whole grid misses plateau peaks (adjacent grid points with
         * equal salience), so use a running global max. */
        WubuHarmCand top[5] = {{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}};
        int ntop = 0;
        {
            /* 1. global max */
            int gmax = -1;
            float gsal = 0.0f;
            for (int c = 0; c < n_cand; c++) {
                if (cands[c].sal > gsal) { gsal = cands[c].sal; gmax = c; }
            }
            if (gmax >= 0 && cands[gmax].sal > 1e-6f) {
                top[ntop++] = cands[gmax];
            }
            /* 2. additional local maxima (excluding the global) */
            for (int c = 0; c < n_cand; c++) {
                if (c == gmax) continue;
                float left  = (c > 0) ? cands[c-1].sal : 0.0f;
                float right = (c < n_cand-1) ? cands[c+1].sal : 0.0f;
                if (cands[c].sal <= left || cands[c].sal <= right) continue;
                if (cands[c].sal <= 1e-6f) continue;
                if (ntop < 5) { top[ntop++] = cands[c]; }
                else {
                    int w = 0;
                    for (int t = 1; t < ntop; t++) if (top[t].sal < top[w].sal) w = t;
                    if (cands[c].sal > top[w].sal) top[w] = cands[c];
                }
            }
        }
        qsort(top, (size_t)ntop, sizeof(WubuHarmCand), cmp_cand_desc);

        /* pick the lead: global salience max, refined toward the input
         * contour (RMVPE) when it agrees within ~1 semitone — RMVPE is
         * far more accurate on monophonic frames than our 36-grid spectral
         * pick (which can land a grid step flat when adjacent points map
         * to the same FFT bin). The continuity bonus already stabilizes
         * note-flipping; this refinement nails the exact cents. */
        float lead = 0.0f, harm = 0.0f, gain = 0.0f;
        if (ntop >= 1 && top[0].sal > 1e-6f) {
            lead = top[0].freq;
            if (primary_in && primary_in[fi] > 0.0f) {
                float cents = 1200.0f * fabsf(log2f(primary_in[fi] / lead));
                if (cents < 110.0f) lead = primary_in[fi];
            }
        }
        /* harmony: search the FULL salience map at semitone-quantized
         * musical intervals above the lead (autotune harmony = exact
         * semitones). This replaces the "second-strongest peak" heuristic
         * which flipped between partials of both voices. */
        if (lead > 0.0f) {
            float lead_raw = (ntop >= 1) ? top[0].raw : 0.0f;
            harm = harmony_at_interval(cands, n_cand, lead, lead_raw,
                                       cand_f, n_cand, &gain);
        }

        /* If detection failed, hold the input contour (RMVPE is usually
         * right on monophonic frames) — only override when we have a real
         * peak. */
        if (lead <= 0.0f && primary_in && primary_in[fi] > 0.0f)
            lead = primary_in[fi];

        if (lead <= 0.0f) lead = 0.0f;
        if (primary_out) primary_out[fi] = lead;
        if (harmony_out) harmony_out[fi] = harm;
        if (harmony_gain) harmony_gain[fi] = (harm > 0.0f) ? gain : 0.0f;

        if (lead > 0.0f) prev_primary = lead;
        else if (primary_in && primary_in[fi] > 0.0f) prev_primary = primary_in[fi];

        free(mag);
    }

    free(cand_f); free(cands); free(spec); free(hann);
    return n_frames;
}
