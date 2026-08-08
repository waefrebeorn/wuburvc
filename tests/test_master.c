/* test_master.c — WuBuDesk mastering suite verification.
 *
 * Triple-DA: generate a synthetic stereo mix (kick + bass + vocal-ish tone
 * with dynamics), run through the C11 mastering chain, verify:
 *   1. no NaN/Inf,
 *   2. true-peak stays under the ceiling,
 *   3. RMS lands at the target,
 *   4. length preserved.
 *
 * Build:
 *   cc -std=c11 -O2 -I src src/test_master.c src/wubu_master.c src/wubu_audioio.c \
 *      -lm -o build/test_master.exe
 *
 * License: WaefreBeorn-UMV3
 */
#define _USE_MATH_DEFINES
#include "wubu_master.h"
#include "wubu_audioio.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static float db2lin(float db) { return powf(10.0f, db / 20.0f); }

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("=== WuBuDesk Mastering Suite Test ===\n\n");

    const int sr = 48000;
    const int n = sr * 8; /* 8 seconds */
    float *lr = (float *)malloc((size_t)n * 2 * sizeof(float));
    if (!lr) return 1;

    /* build a musical-ish test signal with dynamics:
     * kick at 50 Hz decaying, bass at 110 Hz, vocal-ish tone 300 Hz with
     * amplitude modulation, hi-hat noise bursts. */
    srand(42);
    for (int i = 0; i < n; i++) {
        double t = (double)i / sr;
        double kick = 0.0, bass = 0.0, vox = 0.0, hat = 0.0;
        double ph = fmod(t, 0.5);       /* phase within 2 Hz beat */
        if (ph < 0.12) {                /* kick transient */
            double env = exp(-ph * 40.0);
            kick = 0.7 * sin(2 * M_PI * 50 * ph) * env;
        }
        bass = 0.4 * sin(2 * M_PI * 110 * t);
        double am = 0.5 + 0.5 * sin(2 * M_PI * 0.5 * t); /* phrase envelope */
        vox = 0.35 * sin(2 * M_PI * 300 * t) * am;
        double hph = fmod(t, 0.125);
        if (hph < 0.02)
            hat = 0.1 * ((double)rand() / RAND_MAX - 0.5) * exp(-hph * 150.0);
        float l = (float)(kick + bass + vox + hat);
        float r = (float)(kick + bass + vox * 0.8 - hat * 0.5);
        lr[i * 2] = l;
        lr[i * 2 + 1] = r;
    }
    printf("[1] test signal: %d samples @ %d Hz (8 s)\n", n, sr);

    WuBuMasterOpts opts;
    wubu_master_default(&opts);
    printf("[2] mastering chain: %d EQ bands, comp %.0f:1 @ %.0f dB, sat %.2f, "
           "limit %.0f dB, RMS %.0f\n",
           opts.n_eq, opts.comp.ratio, opts.comp.threshold_db, opts.sat.drive,
           opts.limiter.ceiling_db, opts.loud.rms_target);

    wubu_master_process(&opts, lr, n, sr);

    /* 1. NaN/Inf check */
    int bad = 0;
    float peak = 0.0f;
    double sum = 0.0;
    for (int i = 0; i < n * 2; i++) {
        if (!isfinite(lr[i])) bad++;
        float a = fabsf(lr[i]);
        if (a > peak) peak = a;
        sum += (double)lr[i] * lr[i];
    }
    float rms = (float)sqrt(sum / (2.0 * n));
    float rms_db = 20.0f * log10f(rms + 1e-12f);
    float peak_db = 20.0f * log10f(peak + 1e-12f);
    printf("[3] post-master: peak %.4f (%.1f dBFS), RMS %.4f (%.1f dBFS)\n",
           peak, peak_db, rms, rms_db);
    printf("     non-finite samples: %d\n", bad);
    if (bad) { printf("     🔴 FAIL: NaN/Inf in output\n"); return 1; }
    printf("     ✅ no NaN/Inf\n");

    float ceiling = db2lin(opts.limiter.ceiling_db) * 1.001f;
    if (peak <= ceiling) printf("     ✅ peak under limiter ceiling\n");
    else { printf("     🔴 FAIL: peak %.4f > ceiling %.4f\n", peak, ceiling); return 1; }

    float target_db = opts.loud.rms_target;
    if (rms_db > target_db + 1.5f || rms_db < target_db - 1.5f) {
        printf("     🔴 FAIL: RMS %.1f far from target %.1f\n", rms_db, target_db);
        return 1;
    }
    printf("     ✅ RMS within 1.5 dB of target\n");

    /* 4. write + reread roundtrip */
    wubu_audio_write("outputs/master_test.wav", lr, n, sr, 1);
    WuBuAudio *rt = wubu_audio_read_stereo("outputs/master_test.wav");
    if (!rt || rt->n != n || rt->sr != sr) {
        printf("     🔴 FAIL: wav roundtrip\n");
        return 1;
    }
    wubu_audio_free(rt);
    printf("     ✅ wav write/read roundtrip\n");

    free(lr);
    printf("\nALL MASTER TESTS PASS\n");
    return 0;
}
