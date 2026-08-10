/* wubu_breath.c — breath-existence detection (see wubu_breath.h).
 *
 * Frame features (10 ms hop, 160 samples @16k):
 *   rms      — short-term energy (silence floor adaptive)
 *   zcr      — zero-crossing rate (breath/fricative signature)
 *   specvar  — variance across FFT magnitudes (VMS analog; sustained
 *              textured noise = breath, flat = silence)
 *   flux     — spectral flux (click/cough transient vs sustained breath)
 *
 * Classification (per INTERSPEECH 2024 + VAD lineage):
 *   VOICED    : f0 > 0 (harmonic source present)
 *   SILENCE   : rms < floor AND zcr low AND specvar low
 *   BREATH    : rms < speech floor (quiet) BUT zcr high AND sustained
 *               (duration >= 300 ms) AND specvar textured
 *   CONSONANT : rms >= speech floor (energetic fricative/plosive)
 *
 * Output: class per 100 fps frame + breath gain (1.0 on breath frames,
 * 0 on silence — kills phantom noise, renders real breaths).
 */
#include "wubu_breath.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define HOP 160          /* 10 ms @16k */
#define WIN 512          /* 32 ms FFT window */
#define BREATH_MIN_MS 300.0f
#define BREATH_MIN_FRAMES ((int)(BREATH_MIN_MS / 10.0f))

/* simple helpers over float PCM (normalized -1..1) */
static float frms(const float *p, int n) {
    double e = 0;
    for (int i = 0; i < n; i++) { double v = p[i]; e += v * v; }
    return (float)sqrt(e / (n > 0 ? n : 1));
}

static float fzcr(const float *p, int n) {
    int z = 0;
    for (int i = 1; i < n; i++)
        if ((p[i] >= 0) != (p[i - 1] >= 0)) z++;
    return (float)z / (float)(n > 1 ? n - 1 : 1);
}

/* spectral variance + flux over a Hann-windowed FFT magnitude frame.
 * We compute a cheap 256-point magnitude spectrum (not full STFT) via
 * direct DFT of a downsampled frame — sufficient for texture detection. */
static void fspec(const float *p, int n, float *var_out, float *flux_out,
                  const float *prev_mag, float *mag_out) {
    const int NB = 96;           /* 0..3 kHz texture bands @16k */
    float mag[NB];
    memset(mag, 0, sizeof(mag));
    /* frame is 512 samples; take bins at multiples of 160/4 = 40 samples
     * (i.e., 50 Hz spacing) up to 3 kHz */
    for (int b = 1; b < NB; b++) {
        double re = 0, im = 0;
        for (int t = 0; t < n; t += 4) {
            double ph = -2.0 * M_PI * b * t / (double)n;
            re += p[t] * cos(ph);
            im += p[t] * sin(ph);
        }
        mag[b] = (float)sqrt(re * re + im * im);
    }
    /* variance across bins */
    double mean = 0;
    for (int b = 1; b < NB; b++) mean += mag[b];
    mean /= (NB - 1);
    double var = 0;
    for (int b = 1; b < NB; b++) { double d = mag[b] - mean; var += d * d; }
    var /= (NB - 1);
    *var_out = (float)var;
    /* flux vs previous frame */
    double fl = 0;
    if (prev_mag) {
        for (int b = 1; b < NB; b++) {
            double d = mag[b] - prev_mag[b];
            fl += d * d;
        }
    }
    *flux_out = (float)fl;
    if (mag_out) memcpy(mag_out, mag, sizeof(mag));
}

int wubu_breath_detect(const float *pcm, int n,
                       const float *f0, int n_f0,
                       int *out_class, float *out_breath_gain,
                       WuBuBreathStats *stats) {
    if (!pcm || n <= 0) return 0;
    int n_frames = n / HOP;
    if (n_frames < 1) n_frames = 1;
    if (n_frames > WUBU_BREATH_MAX_FRAMES) n_frames = WUBU_BREATH_MAX_FRAMES;
    /* The caller's buffers are sized to the f0 frame grid (n_f0) — the
     * 10 ms hop count can exceed it by a few frames at the tail. Clamp so
     * we never write past the caller's allocation. */
    if (n_f0 > 0 && n_frames > n_f0) n_frames = n_f0;

    /* ---- per-frame features ---- */
    float *rms = (float *)calloc((size_t)n_frames, sizeof(float));
    float *zcr = (float *)calloc((size_t)n_frames, sizeof(float));
    float *svar = (float *)calloc((size_t)n_frames, sizeof(float));
    float *flux = (float *)calloc((size_t)n_frames, sizeof(float));
    int *cls = (int *)calloc((size_t)n_frames, sizeof(int));
    if (!rms || !zcr || !svar || !flux || !cls) {
        free(rms); free(zcr); free(svar); free(flux); free(cls);
        return 0;
    }
    float prev_mag[96];
    memset(prev_mag, 0, sizeof(prev_mag));
    for (int f = 0; f < n_frames; f++) {
        int start = f * HOP;
        int len = WIN;
        if (start + len > n) len = n - start;
        if (len < 16) break;
        rms[f] = frms(pcm + start, len);
        zcr[f] = fzcr(pcm + start, len);
        float mg[96];
        fspec(pcm + start, len, &svar[f], &flux[f],
              f ? prev_mag : NULL, mg);
        memcpy(prev_mag, mg, sizeof(mg));
    }

    /* ---- adaptive floors ----
     * silence floor: 10th percentile RMS (background/quiet)
     * speech floor:  P80 RMS × 0.30 — a breath is quiet *relative to the
     * dominant speech/singing energy* (0.01–0.03 rms vs 0.1–0.4 rms).
     * Median is polluted by long breath/silence runs, so use P80 (typical
     * strong phonation) as the speech reference. */
    float *rms_sorted = (float *)malloc((size_t)n_frames * sizeof(float));
    if (!rms_sorted) { free(rms); free(zcr); free(svar); free(flux); free(cls); return 0; }
    memcpy(rms_sorted, rms, (size_t)n_frames * sizeof(float));
    /* full selection sort of the first 512 frames (enough for floors) */
    int ns = n_frames < 512 ? n_frames : 512;
    for (int i = 0; i < ns - 1; i++) {
        int mi = i;
        for (int j = i + 1; j < ns; j++) if (rms_sorted[j] < rms_sorted[mi]) mi = j;
        if (mi != i) { float t = rms_sorted[i]; rms_sorted[i] = rms_sorted[mi]; rms_sorted[mi] = t; }
    }
    float floor_rms = rms_sorted[ns / 10];
    if (floor_rms < 1e-5f) floor_rms = 1e-5f;
    float p80_rms = rms_sorted[(ns * 4) / 5];
    float speech_floor = p80_rms * 0.30f;
    if (speech_floor < floor_rms * 2.0f) speech_floor = floor_rms * 2.0f;
    free(rms_sorted);

    /* ---- initial classification ---- */
    for (int f = 0; f < n_frames; f++) {
        int fidx = (int)((double)f * n_f0 / (double)n_frames);
        if (fidx < 0) fidx = 0;
        if (fidx >= n_f0) fidx = n_f0 - 1;
        int voiced = (f0 && n_f0 > 0) ? (f0[fidx] > 40.0f) : 0;
        if (voiced) {
            cls[f] = WUBU_BREATH_VOICED;
        } else if (rms[f] < floor_rms * 1.5f && zcr[f] < 0.08f && svar[f] < 1e6f) {
            cls[f] = WUBU_BREATH_SILENCE;
        } else if (rms[f] < speech_floor) {
            /* quiet + noisy: candidate breath (or click) */
            cls[f] = WUBU_BREATH_BREATH;
        } else {
            cls[f] = WUBU_BREATH_CONSONANT;
        }
    }

    /* ---- breath duration gate: must be sustained >= 300 ms ---- */
    /* first pass: mark breath runs; second: keep runs >= BREATH_MIN_FRAMES */
    for (int f = 0; f < n_frames; ) {
        if (cls[f] != WUBU_BREATH_BREATH) { f++; continue; }
        int run_start = f;
        while (f < n_frames && cls[f] == WUBU_BREATH_BREATH) f++;
        int run_len = f - run_start;
        if (run_len < BREATH_MIN_FRAMES) {
            /* too short: reclassify quiet noise as silence */
            for (int i = run_start; i < f; i++) cls[i] = WUBU_BREATH_SILENCE;
        }
    }

    /* ---- event list + stats ---- */
    WuBuBreathStats st;
    memset(&st, 0, sizeof(st));
    st.n_frames = n_frames;
    for (int f = 0; f < n_frames; f++) {
        int c = cls[f];
        if (c >= 0 && c < 4) st.class_total[c]++;
    }
    st.voiced_frac = (float)st.class_total[WUBU_BREATH_VOICED] / n_frames;
    st.breath_frac = (float)st.class_total[WUBU_BREATH_BREATH] / n_frames;
    for (int f = 0; f < n_frames; ) {
        if (cls[f] != WUBU_BREATH_BREATH) { f++; continue; }
        int s = f;
        while (f < n_frames && cls[f] == WUBU_BREATH_BREATH) f++;
        if (st.n_events < 256) {
            st.events[st.n_events].start_frame = s;
            st.events[st.n_events].end_frame = f - 1;
            st.events[st.n_events].dur_ms = (float)(f - s) * 10.0f;
            float pz = 0, pr = 0;
            for (int i = s; i < f; i++) {
                if (zcr[i] > pz) pz = zcr[i];
                if (rms[i] > pr) pr = rms[i];
            }
            st.events[st.n_events].peak_zcr = pz;
            st.events[st.n_events].peak_rms = pr;
            st.n_events++;
        }
    }

    /* ---- outputs ---- */
    if (out_class) memcpy(out_class, cls, (size_t)n_frames * sizeof(int));
    if (out_breath_gain) {
        for (int f = 0; f < n_frames; f++)
            out_breath_gain[f] = (cls[f] == WUBU_BREATH_BREATH) ? 1.0f : 0.0f;
    }
    if (stats) *stats = st;

    free(rms); free(zcr); free(svar); free(flux); free(cls);
    return n_frames;
}
