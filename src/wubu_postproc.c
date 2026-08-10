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
#include "wubu_fft.h"
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

static void biquad_highpass(BiQuad *bq, float fc, float sr, float Q) {
    float omega = 2.0f * (float)M_PI * fc / sr;
    float sinw = sinf(omega), cosw = cosf(omega);
    float alpha = sinw / (2 * Q);
    float b0 = (1 + cosw) / 2;
    float b1 = -(1 + cosw);
    float b2 = (1 + cosw) / 2;
    float a0 = 1 + alpha;
    float a1 = -2 * cosw;
    float a2 = 1 - alpha;
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

    /* De-essing: split-band dynamic sibilance reduction at 5-10kHz.
     * Proper implementation: high-pass the 5kHz band, follow its energy
     * envelope (fast attack ~2ms, slow release ~80ms), and attenuate ONLY
     * that band when sibilance crosses threshold. The old one-pole HPF at
     * alpha=0.05 was a ~400Hz broadband detector — it barely moved. */
    if (opts->de_ess_strength > 0.0f) {
        float fc = 5000.0f;                       /* sibilance band edge */
        float Q = 0.7071f;
        float attack = 1.0f - expf(-1.0f / (0.002f * (float)sr));   /* 2 ms */
        float release = 1.0f - expf(-1.0f / (0.080f * (float)sr));  /* 80 ms */
        BiQuad hp; biquad_init(&hp);
        biquad_highpass(&hp, fc, (float)sr, Q);
        float env = 0.0f;
        for (int i = 0; i < n; i++) {
            float s = biquad_process(&hp, output[i]);
            float e = fabsf(s);
            env = (e > env) ? env + attack * (e - env)
                            : env + release * (e - env);
            if (env > opts->de_ess_threshold) {
                float over = (env - opts->de_ess_threshold) / (env + 1e-6f);
                /* reduction ramps 1.0 → (1 - strength) as sibilance peaks */
                float att = 1.0f - opts->de_ess_strength * over;
                if (att < 0.5f) att = 0.5f;
                output[i] -= s * (1.0f - att);   /* remove part of the HF band */
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
    float *slow = (float *)malloc((size_t)n * sizeof(float));
    float *vib = (float *)malloc((size_t)n * sizeof(float));
    if (!smoothed || !slow || !vib) {
        free(smoothed); free(slow); free(vib); return;
    }

    /* Vibrato-aware smoothing (VibE-SVC, arXiv:2605.20794): vibrato is a
     * separable HIGH-FREQUENCY component of the f0 contour (4-7 Hz = period
     * ~15-25 frames @100fps). Naive smoothing (old code) destroyed it and
     * then injected a FAKE fixed-rate sine — phase-locked to the frame
     * index, ignoring the singer's real vibrato. Instead:
     *   1. slow   = heavy moving average (the pitch "carrier")
     *   2. vib    = median-3 minus slow  (the singer's actual vibrato,
     *               which survives the slow MA but frame-to-frame tracking
     *               jitter is removed by the median)
     *   3. out    = slow + vib, blended by strength against the original
     * So tracking jitter is smoothed but the singer's natural vibrato is
     * PRESERVED at its original depth/frequency. */
    const int MA = 5;   /* 50 ms @100fps — passes 4-7 Hz vibrato, kills jitter */
    for (int i = 0; i < n; i++) {
        if (f0[i] <= 0) { slow[i] = 0; vib[i] = 0; continue; }
        /* moving average over voiced neighbors only */
        float acc = 0; int cnt = 0;
        for (int k = -MA; k <= MA; k++) {
            int j = i + k;
            if (j >= 0 && j < n && f0[j] > 0) { acc += f0[j]; cnt++; }
        }
        slow[i] = cnt > 0 ? acc / cnt : f0[i];
        /* median-3 of the original (kills single-frame jitter, keeps
         * slower vibrato) */
        float l = (i > 0 && f0[i-1] > 0) ? f0[i-1] : f0[i];
        float r = (i < n-1 && f0[i+1] > 0) ? f0[i+1] : f0[i];
        float c = f0[i];
        float med = (l <= c) ? ((c <= r) ? c : (l > r ? l : r))
                             : ((l <= r) ? l : (c > r ? c : r));
        vib[i] = med - slow[i];   /* actual vibrato + low-freq drift */
    }

    for (int i = 0; i < n; i++) {
        if (f0[i] <= 0) { smoothed[i] = 0; continue; }
        float target = slow[i] + vib[i];   /* jitter removed, vibrato kept */
        smoothed[i] = (1.0f - strength) * f0[i] + strength * target;
    }

    memcpy(f0, smoothed, (size_t)n * sizeof(float));
    free(smoothed); free(slow); free(vib);
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

/* ── Artifact spectral gate (AF-Vocoder concept) ──
 * Vocoder artifact frames are spectral outliers: flat (noise-like) AND
 * isolated (harmonic neighbors). Detect per-frame Wiener entropy + HNR;
 * a frame that is flat while BOTH neighbors are harmonic gets a soft-knee
 * attenuation (it's a digital clatter burst, not a real fricative —
 * real fricatives come in runs). In-place safe. */
void wubu_artifact_gate(float *audio, int n, int sr, float strength) {
    if (!audio || n <= 0 || sr <= 0 || strength <= 0.0f) return;
    const int win = 512, hop = 128;   /* 10.7ms frame @48k — bursts dominate */
    if (n < win) return;
    int n_frames = (n - win) / hop + 1;
    if (n_frames < 3) return;

    float *flat = (float *)malloc((size_t)n_frames * sizeof(float));
    float *gains = (float *)malloc((size_t)n_frames * sizeof(float));
    if (!flat || !gains) { free(flat); free(gains); return; }

    /* per-frame flatness (Wiener entropy) via proper FFT */
    int nfft = 512;
    WuBuCpx *spec = (WuBuCpx *)malloc((size_t)nfft * sizeof(WuBuCpx));
    float *hann = (float *)malloc((size_t)win * sizeof(float));
    if (!spec || !hann) { free(spec); free(hann); free(flat); free(gains); return; }
    for (int i = 0; i < win; i++)
        hann[i] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / (float)(win - 1)));
    for (int f = 0; f < n_frames; f++) {
        memset(spec, 0, (size_t)nfft * sizeof(WuBuCpx));
        for (int i = 0; i < win; i++)
            spec[i].re = audio[f * hop + i] * hann[i];
        wubu_fft(spec, nfft, 0);
        double sum_log = 0, sum_mag = 0;
        int valid = 0;
        for (int b = 1; b < nfft / 2; b++) {
            double m = sqrt(spec[b].re * spec[b].re + spec[b].im * spec[b].im);
            if (m > 1e-9) { sum_log += log(m); sum_mag += m; valid++; }
        }
        float flatness = 1.0f;
        if (valid > 1 && sum_mag > 1e-9) {
            double geo = exp(sum_log / valid);
            double arith = sum_mag / valid;
            flatness = (float)(geo / (arith + 1e-12));
            if (flatness < 0) flatness = 0;
            if (flatness > 1) flatness = 1;
        }
        flat[f] = flatness;
    }
    free(spec); free(hann);
    /* gate: SHORT flat runs get attenuated, LONG flat runs survive.
     * Discriminator (AF-Vocoder insight): a real fricative/consonant is a
     * sustained noise run (100-300 ms); a vocoder artifact is a short
     * digital clatter burst (10-50 ms). At hop=128 @48k = 2.67 ms/frame,
     * that means artifacts are <= ~18 frames, fricatives >= ~37 frames.
     * Use 15 frames (~40 ms) as the cutoff — a wide margin above real
     * fricatives, safely below long clatter. */
    const int MAX_ARTIFACT_RUN = 15;
    for (int f = 0; f < n_frames; f++) gains[f] = 1.0f;
    for (int f = 0; f < n_frames; ) {
        if (flat[f] < 0.65f) { f++; continue; }
        int run_start = f;
        while (f < n_frames && flat[f] >= 0.65f) f++;
        int run_len = f - run_start;
        if (run_len >= 1 && run_len <= MAX_ARTIFACT_RUN) {
            /* short clatter burst — attenuate */
            for (int i = run_start; i < f; i++) {
                float depth = (flat[i] - 0.65f) / 0.35f;
                if (depth > 1) depth = 1;
                if (depth < 0) depth = 0;
                gains[i] = 1.0f - strength * depth;
            }
        }
        /* runs > MAX_ARTIFACT_RUN: real fricative, leave alone */
    }
    /* apply with linear interpolation between frames (no clicks) */
    for (int i = 0; i < n; i++) {
        double pos = (double)i / hop;
        int f0 = (int)pos;
        if (f0 >= n_frames - 1) f0 = n_frames - 2;
        double frac = pos - f0;
        float g = (float)(gains[f0] + (gains[f0 + 1] - gains[f0]) * frac);
        audio[i] *= g;
    }
    free(flat); free(gains);
}
