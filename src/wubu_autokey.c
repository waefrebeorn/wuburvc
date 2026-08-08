/* wubu_autokey.c — automatic key adaptation (see wubu_autokey.h). */
#include "wubu_autokey.h"
#include "wubu_rvc_f0.h"
#include "wubu_audioio.h"
#include "wubu_rvc_real.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double voiced_mean(const float *f0, int n) {
    double s = 0; int c = 0;
    for (int i = 0; i < n; i++) if (f0[i] > 0) { s += log(f0[i]); c++; }
    return c ? exp(s / c) : 0.0;
}

/* measure output-vs-input tracking: median cents drift + octave-flip rate */
static void measure_drift(const float *in_f0, int n_in,
                          const float *out_f0, int n_out,
                          float *drift_cents, float *flip_rate) {
    int n = n_in < n_out ? n_in : n_out;
    float *diffs = (float *)malloc((size_t)n * sizeof(float));
    int c = 0, flips = 0;
    for (int i = 0; i < n; i++) {
        if (in_f0[i] > 0 && out_f0[i] > 0) {
            float cents = 1200.0f * log2f(out_f0[i] / in_f0[i]);
            diffs[c++] = cents;
            if (fabsf(log2f(out_f0[i] / in_f0[i])) > 0.5f) flips++;
        }
    }
    *flip_rate = c ? (float)flips / c : 1.0f;
    if (c == 0) { *drift_cents = 0; free(diffs); return; }
    /* median */
    for (int a = 1; a < c; a++) {
        float key = diffs[a]; int b = a - 1;
        while (b >= 0 && diffs[b] > key) { diffs[b + 1] = diffs[b]; b--; }
        diffs[b + 1] = key;
    }
    *drift_cents = diffs[c / 2];
    free(diffs);
}

float wubu_autokey_calibrate(WuBuRVCModel *model, WuBuHubert *hb,
                             WuBuRmvpe *rm,
                             const float *pcm16, int n16,
                             const float *f0_hz, int n_f0,
                             int content_dim, int sr_out, int ups_total,
                             float noise_scale, int use_snake,
                             int probe_secs, float *drift_cents) {
    *drift_cents = 0.0f;
    if (!model || !pcm16 || !f0_hz || !hb || !rm) return 0.0f;
    if (probe_secs < 3) probe_secs = 3;
    int probe_n16 = probe_secs * 16000;
    if (probe_n16 > n16) probe_n16 = n16;
    int probe_frames = probe_n16 / 160;
    if (probe_frames > n_f0) probe_frames = n_f0;
    if (probe_frames < 200) return 0.0f;

    /* pick a VOICED window for the probe — the start of the file is often
     * an instrumental intro with no f0 (an all-unvoiced probe calibrates
     * nothing). Scan for the first stretch with >30% voicing. */
    int start_frame = 0;
    {
        int best = -1, best_vc = 0;
        for (int i = 0; i + probe_frames <= n_f0; i += probe_frames / 2) {
            int vc = 0;
            for (int j = i; j < i + probe_frames; j++)
                if (f0_hz[j] > 0) vc++;
            if (vc > best_vc) { best_vc = vc; best = i; }
            if (vc > probe_frames * 0.4) break; /* good enough */
        }
        if (best >= 0 && best_vc > probe_frames / 5) start_frame = best;
    }
    const float *probe_pcm = pcm16 + (size_t)start_frame * 160;
    const float *probe_f0 = f0_hz + start_frame;
    if (start_frame > 0)
        fprintf(stderr, "[autokey] probe window starts at frame %d (%.1f s)\n",
                start_frame, (double)start_frame / 100.0);

    double in_mean = voiced_mean(probe_f0, probe_frames);
    /* candidates: 0 first, then the octave that lifts a low input into the
     * model's conditioned range (or drops a very high input) */
    float cands[3]; int nc = 1; cands[0] = 0.0f;
    if (in_mean > 0) {
        if (in_mean < 350.0) { cands[nc++] = 12.0f; }
        else if (in_mean > 700.0) { cands[nc++] = -12.0f; }
    }

    float best_shift = 0.0f, best_score = 1e30f;
    float best_drift = 0.0f;

    for (int ci = 0; ci < nc; ci++) {
        float s = cands[ci];
        float gain = powf(2.0f, s / 12.0f);
        /* probe f0 shifted + coarse */
        float *pf0 = (float *)malloc((size_t)(probe_frames + 2) * sizeof(float));
        int *pcoarse = (int *)malloc((size_t)(probe_frames + 2) * sizeof(int));
        float *pnsf = (float *)malloc((size_t)(probe_frames + 2) * sizeof(float));
        if (!pf0 || !pcoarse || !pnsf) { free(pf0); free(pcoarse); free(pnsf); break; }
        for (int i = 0; i < probe_frames; i++) pf0[i] = probe_f0[i] * gain;
        wubu_f0_to_coarse(pf0, probe_frames, 50.0f, 1100.0f, pcoarse, pnsf);

        /* content for the probe */
        int pT = 0, maxT = probe_n16 / 320 + 8;
        float *content = (float *)malloc((size_t)maxT * content_dim * sizeof(float));
        if (!content) { free(pf0); free(pcoarse); free(pnsf); break; }
        pT = wubu_hubert_extract_real(hb, probe_pcm, probe_n16, 2, content,
                                      maxT * content_dim);
        if (pT <= 0) { free(content); free(pf0); free(pcoarse); free(pnsf); break; }
        int pT2 = pT * 2;
        int pf = pT2 < probe_frames ? pT2 : probe_frames;
        /* nearest ×2 upsampling + col-major transpose in one pass */
        float *cmaj = (float *)calloc((size_t)content_dim * pT2, sizeof(float));
        if (!cmaj) { free(content); free(pf0); free(pcoarse); free(pnsf); break; }
        for (int j = 0; j < pT2; j++)
            for (int d = 0; d < content_dim; d++)
                cmaj[(size_t)d * pT2 + j] = content[(size_t)(j / 2) * content_dim + d];
        free(content);

        /* synth probe */
        float *pout = (float *)calloc((size_t)pT2 * ups_total + 1024, sizeof(float));
        if (!pout) { free(cmaj); free(pf0); free(pcoarse); free(pnsf); break; }
        int pn = wubu_rvc_synthesize_real(model, cmaj, pf, content_dim,
                                          pcoarse, pnsf, 0, noise_scale,
                                          pout, pT2 * ups_total + 1024, use_snake);
        free(cmaj); free(pf0); free(pcoarse); free(pnsf);
        if (pn <= 0) { free(pout); break; }

        /* output f0 (resample to 16k, RMVPE) */
        int o16 = 0;
        float *p16 = (float *)malloc((size_t)(pn + 1024) * sizeof(float));
        float *of0 = (float *)calloc((size_t)(pn / 10 + 16), sizeof(float));
        if (!p16 || !of0) { free(pout); free(p16); free(of0); break; }
        o16 = wubu_audio_resample_sinc(pout, pn, sr_out, 16000, p16);
        int onf = wubu_rmvpe_f0(rm, p16, o16, of0, o16 / 160 + 8);
        free(p16); free(pout);

        /* the output is at the FED pitch; compare to the UNSHIFTED input
         * (what we want after the -s restore) */
        float drift = 0, flip = 0;
        if (onf > 0) {
            float *ref = (float *)malloc((size_t)probe_frames * sizeof(float));
            if (ref) {
                for (int i = 0; i < probe_frames; i++) ref[i] = probe_f0[i]; /* input key */
                measure_drift(ref, probe_frames, of0, onf, &drift, &flip);
                free(ref);
            }
        }
        free(of0);

        /* score: octave flips dominate, then |drift| */
        float score = flip * 100.0f + fabsf(drift) / 12.0f;
        fprintf(stderr, "[autokey] probe s=%+.0f: drift %+.0f cents, flips %.0f%% (score %.1f)\n",
                s, drift, flip * 100.0f, score);
        if (score < best_score) { best_score = score; best_shift = s; best_drift = drift; }
    }

    /* fine correction: if the model still pulls at the best feed, fold the
     * residual into the shift (output then lands on the input's key) */
    if (fabsf(best_drift) > 30.0f && fabsf(best_drift) < 500.0f) {
        best_shift -= best_drift / 1200.0f * 12.0f; /* semitones */
    }
    *drift_cents = best_drift;
    if (fabsf(best_shift) < 0.5f) best_shift = 0.0f;
    fprintf(stderr, "[autokey] chosen feed shift %+.1f semitones (input mean %.0f Hz)\n",
            best_shift, in_mean);
    return best_shift;
}

float wubu_autokey_load(const char *model_dir, float *cached_mean,
                        float *cached_drift) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/model_key.json", model_dir);
    FILE *f = fopen(path, "rb");
    if (!f) return 0.0f;
    float shift = 0, mean = 0, drift = 0;
    int got = fscanf(f, "{\"shift\":%f,\"input_mean\":%f,\"drift\":%f}", &shift, &mean, &drift);
    fclose(f);
    if (got == 3) { *cached_mean = mean; *cached_drift = drift; return shift; }
    return 0.0f;
}

float wubu_autokey_input_mean(const float *f0, int n) {
    return (float)voiced_mean(f0, n);
}

int wubu_autokey_save(const char *model_dir, float shift, float input_mean,
                      float drift) {
    char path[1024];
    snprintf(path, sizeof(path), "%s/model_key.json", model_dir);
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    fprintf(f, "{\"shift\":%.1f,\"input_mean\":%.1f,\"drift\":%.1f}", shift, input_mean, drift);
    fclose(f);
    return 0;
}
