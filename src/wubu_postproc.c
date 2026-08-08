/* wubu_postproc.c — Audio post-processing pipeline for RVC voice conversion.
 *
 * Implements the post-processing pipeline improvements from the research catalog:
 *   - EQ with presence boost (2-5kHz) and mud cut (200-300Hz)
 *   - De-essing (dynamic sibilance reduction 5-10kHz)
 *   - Dynamic limiting (prevent clipping, ceiling at -0.5dB)
 *   - Harmonic enhancement (subtle saturation for rich harmonics)
 *   - RMS envelope matching (gentle normalization)
 *   - Formant shifting (gender conversion)
 *
 * Research: Applio post-processing, ElevenLabs voice design,
 *   Sonarworks de-essing, HiFi-GAN vocoding improvements.
 *
 * License: WaefreBeorn-UMV3
 */

#define _USE_MATH_DEFINES
#include "wubu_postproc.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

/* ── Simple biquad filter (Biquad Cascade) ──
 * Used for EQ, presence boost, mud cut. */
typedef struct {
    float b0, b1, b2, a1, a2;  /* filter coefficients */
    float x1, x2, y1, y2;      /* delay line */
} BiQuad;

static void biquad_init(BiQuad *bq) {
    bq->b0 = 1.0f; bq->b1 = 0; bq->b2 = 0;
    bq->a1 = 0; bq->a2 = 0;
    bq->x1 = bq->x2 = bq->y1 = bq->y2 = 0;
}

static float biquad_process(BiQuad *bq, float x0) {
    float y0 = bq->b0 * x0 + bq->b1 * bq->x1 + bq->b2 * bq->x2
               - bq->a1 * bq->y1 - bq->a2 * bq->y2;
    bq->x2 = bq->x1; bq->x1 = x0;
    bq->y2 = bq->y1; bq->y1 = y0;
    return y0;
}

static void biquad_lowshelf(BiQuad *bq, float fc, float sr, float gain_db) {
    float A = powf(10.0f, gain_db / 40.0f);
    float omega = 2.0f * (float)M_PI * fc / sr;
    float sinw = sinf(omega), cosw = cosf(omega);
    float sqrtA = sqrtf(A);
    float b0 = A * ((A + 1) - (A - 1) * cosw + 2 * sqrtA * sinw);
    float b1 = 2 * A * ((A - 1) - (A + 1) * cosw);
    float b2 = A * ((A + 1) - (A - 1) * cosw - 2 * sqrtA * sinw);
    float a0 = (A + 1) + (A - 1) * cosw + 2 * sqrtA * sinw;
    float a1 = -2 * ((A - 1) + (A + 1) * cosw);
    float a2 = (A + 1) + (A - 1) * cosw - 2 * sqrtA * sinw;
    bq->b0 = b0 / a0; bq->b1 = b1 / a0; bq->b2 = b2 / a0;
    bq->a1 = a1 / a0; bq->a2 = a2 / a0;
    bq->x1 = bq->x2 = bq->y1 = bq->y2 = 0;
}

static void biquad_peaking(BiQuad *bq, float fc, float sr, float gain_db, float Q) {
    float A = powf(10.0f, gain_db / 40.0f);
    float omega = 2.0f * (float)M_PI * fc / sr;
    float sinw = sinf(omega), cosw = cosf(omega);
    float alpha = sinw / (2 * Q);
    float b0 = 1 + alpha * A;
    float b1 = -2 * cosw;
    float b2 = 1 - alpha * A;
    float a0 = 1 + alpha / A;
    float a1 = -2 * cosw;
    float a2 = 1 - alpha / A;
    bq->b0 = b0 / a0; bq->b1 = b1 / a0; bq->b2 = b2 / a0;
    bq->a1 = a1 / a0; bq->a2 = a2 / a0;
    bq->x1 = bq->x2 = bq->y1 = bq->y2 = 0;
}

void wubu_post_process(const float *input, float *output, int n, int sr,
                        const WuBuPostProcOpts *opts) {
    if (!input || !output || n <= 0) return;
    if (!opts) {
        output[0] = input[0]; memcpy(output, input, (size_t)n * sizeof(float)); return;
    }

    memcpy(output, input, (size_t)n * sizeof(float));

    /* Mud cut: low-shelf at 200Hz */
    if (opts->mud_cut_db != 0.0f) {
        BiQuad bq; biquad_init(&bq);
        biquad_lowshelf(&bq, 200.0f, (float)sr, opts->mud_cut_db);
        for (int i = 0; i < n; i++)
            output[i] = biquad_process(&bq, output[i]);
    }

    /* Presence boost: peaking EQ at 3kHz */
    if (opts->presence_boost_db != 0.0f) {
        BiQuad bq; biquad_init(&bq);
        biquad_peaking(&bq, 3000.0f, (float)sr, opts->presence_boost_db, 1.0f);
        for (int i = 0; i < n; i++)
            output[i] = biquad_process(&bq, output[i]);
    }

    /* De-essing: dynamic attenuation of high frequencies during sibilance */
    if (opts->de_ess_strength > 0.0f) {
        float alpha = 0.05f;
        float prev_sample = 0;
        float hpf_prev = 0;
        for (int i = 0; i < n; i++) {
            float hpf = alpha * (output[i] - prev_sample) + (1.0f - alpha) * hpf_prev;
            hpf_prev = hpf;
            prev_sample = output[i];
            float energy = fabsf(hpf);
            if (energy > opts->de_ess_threshold) {
                float reduction = 1.0f - opts->de_ess_strength * 0.5f *
                                  (energy - opts->de_ess_threshold) / (energy + 0.01f);
                if (reduction < 0.5f) reduction = 0.5f;
                output[i] *= reduction;
            }
        }
    }

    /* Harmonic enhancement: subtle saturation */
    if (opts->harmonic_drive > 0.0f) {
#pragma omp parallel for if(n >= 256)
        for (int i = 0; i < n; i++) {
            float x = output[i] * (1.0f + opts->harmonic_drive);
            output[i] = (2.0f / (float)M_PI) * atanf((float)M_PI * x / 2.0f);
        }
    }

    /* Dynamic limiting: prevent clipping, ceiling at -0.5dB */
    float peak = 0;
#pragma omp parallel for reduction(max:peak) if(n >= 256)
    for (int i = 0; i < n; i++) {
        float a = fabsf(output[i]);
        if (a > peak) peak = a;
    }
    float ceiling = powf(10.0f, -0.5f / 20.0f); /* ~0.944 */
    if (peak > ceiling) {
        float scale = ceiling / peak;
#pragma omp parallel for if(n >= 256)
        for (int i = 0; i < n; i++)
            output[i] *= scale;
    }

    /* RMS normalization: scale to target RMS level, then peak-limit */
    if (opts->rms_target > 0.0f) {
        float sum_sq = 0;
#pragma omp parallel for reduction(+:sum_sq) if(n >= 256)
        for (int i = 0; i < n; i++)
            sum_sq += output[i] * output[i];
        float rms = sqrtf(sum_sq / (float)n);
        if (rms > 1e-10f) {
            float scale = opts->rms_target / rms;
            /* Apply RMS scaling */
#pragma omp parallel for if(n >= 256)
            for (int i = 0; i < n; i++)
                output[i] *= scale;
            /* Recompute peak after scaling — if we exceed ceiling, do a final peak limit */
            float new_peak = 0;
#pragma omp parallel for reduction(max:new_peak) if(n >= 256)
            for (int i = 0; i < n; i++) {
                float a = fabsf(output[i]);
                if (a > new_peak) new_peak = a;
            }
            if (new_peak > ceiling) {
                float pscale = ceiling / new_peak;
#pragma omp parallel for if(n >= 256)
                for (int i = 0; i < n; i++)
                    output[i] *= pscale;
            }
        }
    }
}

void wubu_formant_shift(const float *input, float *output, int n, int sr,
                        float shift_ratio) {
    if (!input || !output || n <= 0 || shift_ratio <= 0) {
        if (input && output && n > 0) memcpy(output, input, (size_t)n * sizeof(float));
        return;
    }
    if (shift_ratio == 1.0f) {
        memcpy(output, input, (size_t)n * sizeof(float));
        return;
    }
    int fft_size = 2048;
    int hop = 512;
    int n_frames = (n - fft_size) / hop + 1;
    if (n_frames < 2) {
        memcpy(output, input, (size_t)n * sizeof(float));
        return;
    }
    float *window = (float *)malloc(fft_size * sizeof(float));
    for (int i = 0; i < fft_size; i++)
        window[i] = 0.5f * (1 - cosf(2 * (float)M_PI * i / (fft_size - 1)));
    int out_len = (int)(n / shift_ratio) + fft_size;
    float *out = (float *)calloc(out_len, sizeof(float));
    float *win_sum = (float *)calloc(out_len, sizeof(float));
    for (int f = 0; f < n_frames; f++) {
        float out_pos_f = (float)(f * hop) / shift_ratio;
        int out_pos = (int)(out_pos_f);
        for (int i = 0; i < fft_size; i++) {
            float x = input[f * hop + i] * window[i];
            if (out_pos + i >= 0 && out_pos + i < out_len) {
                out[out_pos + i] += x;
                win_sum[out_pos + i] += window[i];
            }
        }
    }
    int copy_n = out_len < n ? out_len : n;
    for (int i = 0; i < copy_n; i++) {
        if (win_sum[i] > 1e-8f)
            output[i] = out[i] / win_sum[i];
        else
            output[i] = 0;
    }
    for (int i = copy_n; i < n; i++)
        output[i] = 0;
    free(window); free(out); free(win_sum);
}

void wubu_f0_smooth(float *f0, int n, float strength) {
    if (!f0 || n <= 0 || strength <= 0) return;
    float *smoothed = (float *)malloc((size_t)n * sizeof(float));
    if (!smoothed) return;
    for (int i = 0; i < n; i++) {
        if (f0[i] <= 0) { smoothed[i] = 0; continue; }
        float c = f0[i];
        float l = (i > 0 && f0[i-1] > 0) ? f0[i-1] : c;
        float r = (i < n-1 && f0[i+1] > 0) ? f0[i+1] : c;
        smoothed[i] = (1 - strength) * c + strength * (0.5f * c + 0.25f * l + 0.25f * r);
    }
    if (strength > 0.5f) {
        float vib_depth = (strength - 0.5f) * 0.02f;
        for (int i = 0; i < n; i++) {
            if (smoothed[i] > 0) {
                float vib = 1.0f + vib_depth * sinf(0.2f * i);
                smoothed[i] *= vib;
            }
        }
    }
    memcpy(f0, smoothed, (size_t)n * sizeof(float));
    free(smoothed);
}

void wubu_adaptive_feature_blend(const float *src_feat, const float *ref_feat,
                                  float *output, int n_frames, int dim,
                                  float index_rate, const float *energy) {
    if (!src_feat || !ref_feat || !output) return;
    if (index_rate <= 0.0f) {
        memcpy(output, src_feat, (size_t)n_frames * dim * sizeof(float));
        return;
    }
    float min_e = 1e10, max_e = 0;
    for (int i = 0; i < n_frames; i++) {
        float e = energy ? energy[i] : 0.1f;
        if (e < min_e) min_e = e;
        if (e > max_e) max_e = e;
    }
    float range = max_e - min_e;
    if (range < 1e-8f) range = 1.0f;
#pragma omp parallel for if(n_frames * dim >= 256)
    for (int f = 0; f < n_frames; f++) {
        float e = energy ? energy[f] : 0.1f;
        float norm_e = (e - min_e) / range;
        float blend = index_rate * (0.3f + 0.7f * norm_e);
        for (int d = 0; d < dim; d++) {
            output[(size_t)f * dim + d] =
                (1 - blend) * src_feat[(size_t)f * dim + d] +
                blend * ref_feat[(size_t)f * dim + d];
        }
    }
}

void wubu_apply_character_preset(const float *input, float *output, int n, int sr,
                                int preset) {
    WuBuPostProcOpts opts = {0};
    switch (preset) {
        case WUBU_PRESET_WARM:
            opts.mud_cut_db = -1.0f;
            opts.presence_boost_db = 1.5f;
            opts.harmonic_drive = 0.1f;
            /* No RMS normalization — preserve synthesis dynamics */
            break;
        case WUBU_PRESET_BRIGHT:
            opts.presence_boost_db = 2.0f;
            opts.de_ess_strength = 0.15f;
            opts.de_ess_threshold = 0.08f;
            opts.harmonic_drive = 0.05f;
            break;
        case WUBU_PRESET_SMOOTH:
            opts.mud_cut_db = -2.0f;
            opts.presence_boost_db = 0.5f;
            opts.de_ess_strength = 0.1f;
            opts.de_ess_threshold = 0.05f;
            break;
        case WUBU_PRESET_BREATHY:
            opts.mud_cut_db = -1.5f;
            opts.presence_boost_db = 1.0f;
            opts.de_ess_strength = 0.08f;
            opts.de_ess_threshold = 0.06f;
            opts.harmonic_drive = 0.08f;
            break;
        default:
            /* No RMS normalization — preserve synthesis dynamics */
            break;
    }
    wubu_post_process(input, output, n, sr, &opts);
}

void wubu_rms_mix_rate(const float *input, float *output, int n, int sr,
                       float mix) {
    if (!input || !output || n <= 0 || sr <= 0) return;
    const int win = 2048, hop = 512;
    if (n < win) {
        /* too short for one frame — just scale to input RMS */
        double si = 0, so = 0;
        for (int i = 0; i < n; i++) { si += (double)input[i] * input[i]; so += (double)output[i] * output[i]; }
        float ri = (float)sqrt(si / n), ro = (float)sqrt(so / n);
        if (ro > 1e-9f) {
            float g = (ri * mix + ro * (1.0f - mix)) / ro;
            for (int i = 0; i < n; i++) output[i] *= g;
        }
        return;
    }
    int n_frames = (n - win) / hop + 1;
    float *env_in = (float *)malloc((size_t)n_frames * sizeof(float));
    float *env_out = (float *)malloc((size_t)n_frames * sizeof(float));
    if (!env_in || !env_out) { free(env_in); free(env_out); return; }
    for (int f = 0; f < n_frames; f++) {
        double si = 0, so = 0;
        for (int j = 0; j < win; j++) {
            int idx = f * hop + j;
            si += (double)input[idx] * input[idx];
            so += (double)output[idx] * output[idx];
        }
        env_in[f] = (float)sqrt(si / win);
        env_out[f] = (float)sqrt(so / win);
    }
    /* apply blended envelope, linear-interp between frames */
    for (int i = 0; i < n; i++) {
        double pos = (double)i / hop;
        int f0 = (int)pos;
        if (f0 >= n_frames - 1) f0 = n_frames - 2;
        double frac = pos - f0;
        float ei = (float)(env_in[f0] + (env_in[f0 + 1] - env_in[f0]) * frac);
        float eo = (float)(env_out[f0] + (env_out[f0 + 1] - env_out[f0]) * frac);
        float env = ei * mix + eo * (1.0f - mix);
        if (eo > 1e-9f) output[i] *= env / eo;
    }
    free(env_in);
    free(env_out);
}
