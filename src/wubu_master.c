/* wubu_master.c — stereo mastering chain (C11).
 *
 * Chain: EQ -> compressor -> saturation -> stereo width -> limiter -> loudness.
 * Biquad filters use the RBJ audio EQ cookbook (peaking/shelf/pass).
 * Compressor: feed-forward, RMS detector with attack/release smoothing.
 * Limiter: peak gain computer with release smoothing.
 * Loudness: RMS-based target with true-peak safety ceiling.
 *
 * License: WaefreBeorn-UMV3
 */
#define _USE_MATH_DEFINES
#include "wubu_master.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

static void stage_peak(const char *tag, const float *lr, int n) {
    if (!getenv("WUBU_MASTER_DBG")) return;
    float pk = 0.0f;
    double sum = 0.0;
    for (int i = 0; i < n * 2; i++) {
        float a = fabsf(lr[i]);
        if (a > pk) pk = a;
        sum += (double)lr[i] * lr[i];
    }
    float rms = (float)sqrt(sum / (2.0 * n));
    fprintf(stderr, "[master] %s: peak %.4f rms %.4f\n", tag, pk, rms);
}

static float db2lin(float db) { return powf(10.0f, db / 20.0f); }
static float lin2db(float v) {
    return (v > 1e-12f) ? 20.0f * log10f(v) : -120.0f;
}

/* ── biquad (RBJ) — double precision: filters near Nyquist/DC (e.g. a 30 Hz
 * highpass at 48 kHz) put poles on the unit circle; float32 rounding tips
 * them unstable and the filter blows up over long files. ── */
typedef struct {
    double b0, b1, b2, a1, a2;
    double z1, z2;
} Biquad;

static void biquad_init(Biquad *q, WuBuEqType type, float f, float gain_db,
                        float qf, float sr) {
    memset(q, 0, sizeof(*q));
    if (f <= 0.0f || f >= sr * 0.5f) return;
    double A = pow(10.0, gain_db / 40.0);
    double w0 = 2.0 * M_PI * f / sr;
    double cosw = cos(w0);
    double sinw = sin(w0);
    double alpha = sinw / (2.0 * qf);
    double b0, b1, b2, a0, a1, a2;
    switch (type) {
    case WUBU_EQ_PEAK:
        b0 = 1.0f + alpha * A; b1 = -2.0f * cosw; b2 = 1.0f - alpha * A;
        a0 = 1.0f + alpha / A; a1 = -2.0f * cosw; a2 = 1.0f - alpha / A;
        break;
    case WUBU_EQ_LOWSHELF:
        b0 = A * ((A + 1.0f) - (A - 1.0f) * cosw + 2.0f * sqrtf(A) * alpha);
        b1 = 2.0f * A * ((A - 1.0f) - (A + 1.0f) * cosw);
        b2 = A * ((A + 1.0f) - (A - 1.0f) * cosw - 2.0f * sqrtf(A) * alpha);
        a0 = (A + 1.0f) + (A - 1.0f) * cosw + 2.0f * sqrtf(A) * alpha;
        a1 = -2.0f * ((A - 1.0f) + (A + 1.0f) * cosw);
        a2 = (A + 1.0f) + (A - 1.0f) * cosw - 2.0f * sqrtf(A) * alpha;
        break;
    case WUBU_EQ_HIGHSHELF:
        b0 = A * ((A + 1.0f) + (A - 1.0f) * cosw + 2.0f * sqrtf(A) * alpha);
        b1 = -2.0f * A * ((A - 1.0f) + (A + 1.0f) * cosw);
        b2 = A * ((A + 1.0f) + (A - 1.0f) * cosw - 2.0f * sqrtf(A) * alpha);
        a0 = (A + 1.0f) - (A - 1.0f) * cosw + 2.0f * sqrtf(A) * alpha;
        a1 = 2.0f * ((A - 1.0f) - (A + 1.0f) * cosw);
        a2 = (A + 1.0f) - (A - 1.0f) * cosw - 2.0f * sqrtf(A) * alpha;
        break;
    case WUBU_EQ_HIGHPASS:
        b0 = (1.0f + cosw) / 2.0f; b1 = -(1.0f + cosw); b2 = b0;
        a0 = 1.0f + alpha; a1 = -2.0f * cosw; a2 = 1.0f - alpha;
        break;
    case WUBU_EQ_LOWPASS:
    default:
        b0 = (1.0f - cosw) / 2.0f; b1 = 1.0f - cosw; b2 = b0;
        a0 = 1.0f + alpha; a1 = -2.0f * cosw; a2 = 1.0f - alpha;
        break;
    }
    q->b0 = b0 / a0; q->b1 = b1 / a0; q->b2 = b2 / a0;
    q->a1 = a1 / a0; q->a2 = a2 / a0;
    if (getenv("WUBU_MASTER_DBG"))
        fprintf(stderr, "[biquad] type=%d f=%.0f g=%.2f q=%.3f -> b0=%.6f b1=%.6f b2=%.6f a1=%.6f a2=%.6f\n",
                (int)type, f, gain_db, qf, q->b0, q->b1, q->b2, q->a1, q->a2);
}

static float biquad_run(Biquad *q, float x) {
    double y = q->b0 * x + q->z1;
    q->z1 = q->b1 * x - q->a1 * y + q->z2;
    q->z2 = q->b2 * x - q->a2 * y;
    return (float)y;
}

/* ── envelope smoothing coefficient from ms ── */
static float coeff_ms(float ms, float sr) {
    if (ms <= 0.0f) return 1.0f;
    return 1.0f - expf(-1.0f / (ms * 0.001f * sr));
}

void wubu_master_default(WuBuMasterOpts *opts) {
    memset(opts, 0, sizeof(*opts));
    /* gentle low rumble cut + presence lift */
    opts->eq[0].type = WUBU_EQ_HIGHPASS; opts->eq[0].f = 30.0f; opts->eq[0].q = 0.707f;
    opts->eq[1].type = WUBU_EQ_PEAK;     opts->eq[1].f = 120.0f; opts->eq[1].gain_db = -1.5f; opts->eq[1].q = 1.0f;
    opts->eq[2].type = WUBU_EQ_PEAK;     opts->eq[2].f = 3000.0f; opts->eq[2].gain_db = 1.0f; opts->eq[2].q = 0.8f;
    opts->eq[3].type = WUBU_EQ_HIGHSHELF;opts->eq[3].f = 10000.0f; opts->eq[3].gain_db = 0.5f; opts->eq[3].q = 0.707f;
    opts->n_eq = 4;
    opts->comp.threshold_db = -18.0f;
    opts->comp.ratio = 2.0f;
    opts->comp.attack_ms = 12.0f;
    opts->comp.release_ms = 180.0f;
    opts->comp.makeup_db = 3.0f;
    opts->comp.stereo_link = 1;
    opts->sat.drive = 0.12f;
    opts->sat.mix = 0.4f;
    opts->stereo.width = 1.0f;
    opts->limiter.ceiling_db = -1.0f;
    opts->limiter.release_ms = 60.0f;
    opts->loud.rms_target = -18.0f;
    opts->loud.true_peak_db = -1.0f;
}

int wubu_master_process(const WuBuMasterOpts *opts, float *lr, int n, int sr) {
    if (!opts || !lr || n <= 0 || sr <= 0) return -1;

    /* ── 1. EQ (per channel, chain of biquads) ── */
    if (opts->n_eq > 0) {
        Biquad *lq = (Biquad *)calloc((size_t)opts->n_eq, sizeof(Biquad));
        Biquad *rq = (Biquad *)calloc((size_t)opts->n_eq, sizeof(Biquad));
        if (!lq || !rq) { free(lq); free(rq); return -1; }
        for (int i = 0; i < opts->n_eq; i++) {
            biquad_init(&lq[i], opts->eq[i].type, opts->eq[i].f,
                        opts->eq[i].gain_db, opts->eq[i].q > 0 ? opts->eq[i].q : 0.707f, (float)sr);
            rq[i] = lq[i];
        }
        for (int i = 0; i < n; i++) {
            float xl = lr[i * 2], xr = lr[i * 2 + 1];
            for (int b = 0; b < opts->n_eq; b++) {
                xl = biquad_run(&lq[b], xl);
                xr = biquad_run(&rq[b], xr);
            }
            lr[i * 2] = xl;
            lr[i * 2 + 1] = xr;
        }
        free(lq); free(rq);
    }
    stage_peak("eq", lr, n);

    /* ── 2. compressor ── */
    if (opts->comp.ratio > 1.0f) {
        float thr = db2lin(opts->comp.threshold_db);
        float slope = 1.0f / opts->comp.ratio;
        float att = coeff_ms(opts->comp.attack_ms, (float)sr);
        float rel = coeff_ms(opts->comp.release_ms, (float)sr);
        float makeup = db2lin(opts->comp.makeup_db);
        float env = 1.0f;
        for (int i = 0; i < n; i++) {
            float xl = lr[i * 2], xr = lr[i * 2 + 1];
            float side = opts->comp.stereo_link
                       ? (fabsf(xl) > fabsf(xr) ? fabsf(xl) : fabsf(xr))
                       : 0.5f * (fabsf(xl) + fabsf(xr));
            float g = 1.0f;
            if (side > thr) {
                float over_db = lin2db(side) - opts->comp.threshold_db;
                float red_db = over_db * (1.0f - slope);
                g = db2lin(-red_db);
            }
            float target = (g < env) ? g : env + rel * (g - env);
            env += att * (target - env);  /* fast attack path via target */
            lr[i * 2] = xl * env * makeup;
            lr[i * 2 + 1] = xr * env * makeup;
        }
    }
    stage_peak("comp", lr, n);

    /* ── 3. saturation ── */
    if (opts->sat.drive > 0.0f && opts->sat.mix > 0.0f) {
        float pre = 1.0f + 3.0f * opts->sat.drive;
        float wet = opts->sat.mix;
        for (int i = 0; i < n * 2; i++) {
            float x = lr[i];
            float s = tanhf(x * pre);
            lr[i] = x * (1.0f - wet) + s * wet;
        }
    }
    stage_peak("sat", lr, n);

    /* ── 4. stereo width ── */
    if (opts->stereo.width > 0.0f && opts->stereo.width != 1.0f) {
        float w = opts->stereo.width;
        for (int i = 0; i < n; i++) {
            float l = lr[i * 2], r = lr[i * 2 + 1];
            float m = 0.5f * (l + r);
            float s = 0.5f * (l - r);
            lr[i * 2] = m + s * w;
            lr[i * 2 + 1] = m - s * w;
        }
    }
    stage_peak("width", lr, n);

    /* ── 5. limiter (TRUE-PEAK, 4x inter-sample detection) ──
     * Sample-level peaks miss inter-sample overshoot - the reconstructed
     * analog waveform between samples can exceed the ceiling and clip on
     * DAC/codec ("random clipping" with zero sample-level clips). Estimate
     * the true peak via 4x linear interpolation between adjacent samples
     * and drive the gain computer from that, fast attack, slow release. */
    {
        float ceiling = db2lin(opts->limiter.ceiling_db);
        float att = coeff_ms(0.5f, (float)sr); /* ~0.5 ms attack */
        float rel = coeff_ms(opts->limiter.release_ms, (float)sr);
        float gain = 1.0f;
        for (int i = 0; i < n; i++) {
            float xl0 = lr[i * 2], xr0 = lr[i * 2 + 1];
            float xl1 = (i + 1 < n) ? lr[(i + 1) * 2] : xl0;
            float xr1 = (i + 1 < n) ? lr[(i + 1) * 2 + 1] : xr0;
            float tp = 0.0f;
            for (int k = 0; k < 4; k++) {
                float f = (float)k / 4.0f;
                float sl = fabsf(xl0 + (xl1 - xl0) * f);
                float sr = fabsf(xr0 + (xr1 - xr0) * f);
                if (sl > tp) tp = sl;
                if (sr > tp) tp = sr;
            }
            float target = (tp * gain > ceiling) ? ceiling / tp : 1.0f;
            float gnew = (target < gain) ? gain + att * (target - gain)
                                         : gain + rel * (target - gain);
            gain = gnew;
            lr[i * 2] = xl0 * gain;
            lr[i * 2 + 1] = xr0 * gain;
        }
    }
    stage_peak("limiter", lr, n);

    /* ── 6. loudness normalize (RMS target + true-peak safety) ── */
    if (opts->loud.rms_target < 0.0f) {
        double sum = 0.0;
        for (int i = 0; i < n * 2; i++) sum += (double)lr[i] * lr[i];
        float rms = (float)sqrt(sum / (2.0 * n));
        if (rms > 1e-9f) {
            float target = db2lin(opts->loud.rms_target);
            float gain = target / rms;
            /* true-peak safety */
            float peak = 0.0f;
            for (int i = 0; i < n; i++) {
                float xl0 = lr[i * 2] * gain, xr0 = lr[i * 2 + 1] * gain;
                float xl1 = (i + 1 < n) ? lr[(i + 1) * 2] * gain : xl0;
                float xr1 = (i + 1 < n) ? lr[(i + 1) * 2 + 1] * gain : xr0;
                for (int k = 0; k < 4; k++) {
                    float f = (float)k / 4.0f;
                    float sl = fabsf(xl0 + (xl1 - xl0) * f);
                    float sr = fabsf(xr0 + (xr1 - xr0) * f);
                    if (sl > peak) peak = sl;
                    if (sr > peak) peak = sr;
                }
            }
            float max_gain = db2lin(opts->loud.true_peak_db) / (peak + 1e-12f);
            if (max_gain < gain) gain = max_gain;
            for (int i = 0; i < n * 2; i++) lr[i] *= gain;
        }
    }
    stage_peak("loud", lr, n);
    return 0;
}
